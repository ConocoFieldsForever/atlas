//! eft::menu — Tarkov-style start menu / map manager ("stash screen").
//!
//! Shown when the viewer is launched with NO pack (bare `eft_viewer`). Scans `packs/` and
//! presents one card per known map: install state, pack size on disk, built age, tarkov.dev
//! intel sync age, data-completeness ticks, and a game-update check — each pack's manifest
//! carries the `sourceFingerprint` of the game install it was extracted from (stamped by
//! `assemble_bevy`/`tools/stamp_fingerprint.py`); the menu recomputes the fingerprint from the
//! live install (stat-only, milliseconds) and flags mismatches as "GAME UPDATED - REBUILD".
//!
//! PLAY reuses the map-switch mechanism (spawn self with the pack argv + exit) so the render
//! path keeps its build-once buffers. DELETE removes the pack dir after an explicit confirm.

use bevy::prelude::*;

/// One known/installed map row.
pub struct MapEntry {
    /// Display title ("Streets of Tarkov").
    pub title: &'static str,
    /// Dataset/dir stem ("streets").
    pub key: &'static str,
    /// packs/<key>.eftpack when present on disk.
    pub pack_dir: Option<String>,
    pub size_bytes: u64,
    /// Age in days of manifest.json (pack build) and loot.json (tarkov.dev sync).
    pub built_days: Option<f64>,
    pub intel_days: Option<f64>,
    pub has_volume: bool,
    pub has_grass: bool,
    pub has_gamedata: bool,
    pub has_icons: bool,
    /// None = pack unstamped (unknown vintage); Some(true) = matches the live install.
    pub fp_match: Option<bool>,
}

/// A running `tools/build_map.py <map>` pipeline: stdout+stderr stream into `log` from a
/// reader thread; the UI shows the tail + the latest `[STAGE i/N]` marker. One at a time.
pub struct BuildJob {
    pub key: String,
    child: std::sync::Arc<std::sync::Mutex<std::process::Child>>,
    log: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Some(exit-ok) once the child has been reaped.
    pub result: Option<bool>,
    /// Wall-clock build start — feeds the loading bar's ESTIMATED TIME readout.
    pub started: std::time::Instant,
}

impl BuildJob {
    pub fn spawn(key: &str, game_dir: &str) -> std::io::Result<Self> {
        use std::io::BufRead;
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000; // no console popping over the menu
        let root = crate::paths::repo_root().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "python kit not found (tools/build_map.py) beside the exe or in the cwd",
            )
        })?;
        let mut child = std::process::Command::new(crate::paths::python_exe(root))
            .current_dir(root) // kit-relative paths inside build_map.py resolve from its root
            .env("EFT_GAME_DATA", game_dir) // menu-selected install drives the pipeline
            .arg("tools/build_map.py")
            .args(key.split([',', ' ']).filter(|s| !s.is_empty()))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()?;
        let log = std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::<String>::with_capacity(512),
        ));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let push = |log: &std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
                    line: String| {
            let mut l = log.lock().unwrap();
            if l.len() >= 500 {
                l.pop_front();
            }
            l.push_back(line);
        };
        for pipe in [
            child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let log = log.clone();
            std::thread::spawn(move || {
                let rd = std::io::BufReader::new(pipe);
                for line in rd.lines().map_while(Result::ok) {
                    push(&log, line);
                }
            });
        }
        {
            // Reaper: flags completion without blocking the UI thread.
            let done_t = done.clone();
            let log_t = log.clone();
            let child = std::sync::Arc::new(std::sync::Mutex::new(child));
            let child2 = child.clone();
            std::thread::spawn(move || {
                let (done, log) = (done_t, log_t);
                loop {
                    match child2.lock().unwrap().try_wait() {
                        Ok(Some(status)) => {
                            push(
                                &log,
                                format!(
                                    "[exit {}]",
                                    status.code().map_or("?".into(), |c| c.to_string())
                                ),
                            );
                            done.store(status.success(), std::sync::atomic::Ordering::SeqCst);
                            // done flag semantics: stored success; presence read via result parse
                            break;
                        }
                        Ok(None) => std::thread::sleep(std::time::Duration::from_millis(300)),
                        Err(_) => break,
                    }
                }
            });
            return Ok(Self {
                key: key.to_string(),
                child,
                log,
                done,
                result: None,
                started: std::time::Instant::now(),
            });
        }
    }

    pub fn cancel(&self) {
        let _ = self.child.lock().unwrap().kill();
    }

    /// Entire captured log (for the COPY LOG button — the panel only shows a tail).
    pub fn full_log(&self) -> String {
        let l = self.log.lock().unwrap();
        let mut s = String::with_capacity(l.iter().map(|x| x.len() + 1).sum());
        for line in l.iter() {
            s.push_str(line);
            s.push('\n');
        }
        s
    }

    /// (last stage marker, tail lines, finished?, success?)
    pub fn snapshot(&self, tail: usize) -> (String, Vec<String>, bool, bool) {
        let l = self.log.lock().unwrap();
        let stage = l
            .iter()
            .rev()
            .find(|s| s.starts_with("[STAGE") || s.starts_with("[BUILD"))
            .cloned()
            .unwrap_or_else(|| "starting...".into());
        let lines: Vec<String> = l.iter().rev().take(tail).cloned().collect();
        let finished = l.iter().rev().any(|s| s.starts_with("[exit"));
        let ok = l.iter().any(|s| s.starts_with("[BUILD OK]"))
            || self.done.load(std::sync::atomic::Ordering::SeqCst);
        (stage, lines.into_iter().rev().collect(), finished, ok)
    }
}

/// Present ONLY in menu mode (bare launch, no pack): drives the fullscreen menu UI and
/// suppresses the in-raid panels (ui.rs checks for this resource).
#[derive(Resource)]
pub struct MenuState {
    pub entries: Vec<MapEntry>,
    pub game_fp: Option<String>,
    pub game_dir: String,
    pub total_bytes: u64,
    /// Index pending delete confirmation.
    pub confirm_delete: Option<usize>,
    /// Index showing rebuild instructions.
    pub show_rebuild: Option<usize>,
    /// The one in-flight pipeline build, if any.
    pub build: Option<BuildJob>,
    /// Build panel: raw log tail visible? Collapsed by default; auto-expands on failure.
    pub show_log: bool,
    /// Footer editor buffer for the game-install path.
    pub game_dir_edit: String,
}

/// The standard map roster (dataset key -> display name). Packs on disk that aren't in this
/// list still show up (title falls back to the key).
const KNOWN_MAPS: &[(&str, &str)] = &[
    ("lighthouse", "Lighthouse"),
    ("interchange", "Interchange"),
    ("factory", "Factory"),
    ("customs", "Customs"),
    ("woods", "Woods"),
    ("shoreline", "Shoreline"),
    ("reserve", "Reserve"),
    ("labs", "The Lab"),
    ("ground_zero", "Ground Zero"),
    ("streets", "Streets of Tarkov"),
    ("labyrinth", "The Labyrinth"),
];

/// FNV-1a 64 over "name|size|mtime_s;" of the game's top-level asset files, sorted by name.
/// MUST stay byte-identical to tools/stamp_fingerprint.py (the python stamper).
pub fn game_fingerprint(game_data: &str) -> Option<String> {
    let mut entries: Vec<(String, u64, u64)> = Vec::new();
    for e in std::fs::read_dir(game_data).ok()? {
        let Ok(e) = e else { continue };
        let Ok(md) = e.metadata() else { continue };
        if !md.is_file() {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        if !(name.starts_with("level")
            || name.ends_with(".assets")
            || name.ends_with(".resS")
            || name.ends_with(".resource"))
        {
            continue;
        }
        let mtime = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())?;
        entries.push((name, md.len(), mtime));
    }
    if entries.is_empty() {
        return None;
    }
    entries.sort();
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for (n, size, mt) in &entries {
        for b in format!("{n}|{size}|{mt};").bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1_0000_0001_B3);
        }
    }
    Some(format!("{h:016x}"))
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(path) {
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                if md.is_dir() {
                    total += dir_size(&e.path());
                } else {
                    total += md.len();
                }
            }
        }
    }
    total
}

fn age_days(path: &std::path::Path) -> Option<f64> {
    let md = std::fs::metadata(path).ok()?;
    let m = md.modified().ok()?;
    Some(m.elapsed().ok()?.as_secs_f64() / 86_400.0)
}

/// Scan packs/ and build the menu model. Called at startup and after a delete.
pub fn scan(game_fp: &Option<String>) -> (Vec<MapEntry>, u64) {
    let mut entries: Vec<MapEntry> = Vec::new();
    let mut total = 0u64;
    let installed = |key: &str| {
        crate::paths::packs_root()
            .join(format!("{key}.eftpack"))
            .to_string_lossy()
            .into_owned()
    };
    // Known roster first, then any extra packs on disk.
    let mut extra: Vec<String> = std::fs::read_dir(crate::paths::packs_root())
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.strip_suffix(".eftpack").map(str::to_string)
                })
                .filter(|k| KNOWN_MAPS.iter().all(|(key, _)| key != k))
                .collect()
        })
        .unwrap_or_default();
    extra.sort();
    let roster = KNOWN_MAPS
        .iter()
        .map(|(k, t)| (*k, *t))
        .chain(extra.iter().map(|k| (k.as_str(), k.as_str())));
    for (key, title) in roster {
        let dir = installed(key);
        let p = std::path::Path::new(&dir);
        let manifest = p.join("manifest.json");
        if !manifest.is_file() {
            entries.push(MapEntry {
                title: Box::leak(title.to_string().into_boxed_str()),
                key: Box::leak(key.to_string().into_boxed_str()),
                pack_dir: None,
                size_bytes: 0,
                built_days: None,
                intel_days: None,
                has_volume: false,
                has_grass: false,
                has_gamedata: false,
                has_icons: false,
                fp_match: None,
            });
            continue;
        }
        let size = dir_size(p);
        total += size;
        let man: serde_json::Value = std::fs::read_to_string(&manifest)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let stamped = man.get("sourceFingerprint").and_then(|v| v.as_str());
        let fp_match = match (stamped, game_fp.as_deref()) {
            (Some(s), Some(g)) => Some(s == g),
            _ => None,
        };
        let has_volume = man
            .get("sidecars")
            .and_then(|s| s.get("volume"))
            .and_then(|v| v.as_str())
            .map(|v| std::path::Path::new(v).exists() || p.join(v).exists())
            .unwrap_or(false)
            || p.join("volume.bin").exists();
        entries.push(MapEntry {
            title: Box::leak(title.to_string().into_boxed_str()),
            key: Box::leak(key.to_string().into_boxed_str()),
            pack_dir: Some(dir.clone()),
            size_bytes: size,
            built_days: age_days(&manifest),
            intel_days: age_days(&p.join("loot.json"))
                .or_else(|| age_days(&crate::paths::shared_dir().join("loot.json"))),
            has_volume,
            has_grass: p.join("grass.bin").exists(),
            has_gamedata: p.join("gamedata.json").exists(),
            has_icons: p.join("icons").is_dir()
                || crate::paths::shared_dir().join("icons").is_dir(),
            fp_match,
        });
    }
    (entries, total)
}

/// Small persisted viewer config beside the EXE (portable-app style; cwd fallback for old
/// installs) — resolution in paths::config_path.
fn config_path() -> std::path::PathBuf {
    crate::paths::config_path()
}

fn config_game_dir() -> Option<String> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config_path()).ok()?).ok()?;
    v.get("gameData").and_then(|s| s.as_str()).map(str::to_string)
}

pub fn save_config_game_dir(dir: &str) {
    let mut v: serde_json::Value = std::fs::read_to_string(config_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    v["gameData"] = serde_json::Value::String(dir.to_string());
    let _ = std::fs::write(config_path(), serde_json::to_string_pretty(&v).unwrap_or_default());
}

/// Looks like an EscapeFromTarkov_Data dir? (has level files / globalgamemanagers)
pub fn valid_game_dir(dir: &str) -> bool {
    let p = std::path::Path::new(dir);
    p.join("globalgamemanagers").exists()
        || p.join("level0").exists()
        || std::fs::read_dir(p)
            .map(|rd| {
                rd.flatten()
                    .any(|e| e.file_name().to_string_lossy().starts_with("sharedassets"))
            })
            .unwrap_or(false)
}

/// BSG launcher registry entry -> "<InstallLocation>\EscapeFromTarkov_Data".
fn registry_game_dir() -> Option<String> {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("reg")
        .args([
            "query",
            r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov",
            "/v",
            "InstallLocation",
        ])
        .creation_flags(0x0800_0000)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let loc = s
        .lines()
        .find(|l| l.trim_start().starts_with("InstallLocation"))?
        .split("REG_SZ")
        .nth(1)?
        .trim()
        .to_string();
    (!loc.is_empty()).then(|| format!(r"{loc}\EscapeFromTarkov_Data"))
}

/// Autodetect priority: EFT_GAME_DATA env > saved config > launcher registry > drive probe.
pub fn detect_game_dir() -> String {
    if let Ok(d) = std::env::var("EFT_GAME_DATA") {
        if !d.is_empty() {
            return d;
        }
    }
    if let Some(d) = config_game_dir().filter(|d| valid_game_dir(d)) {
        return d;
    }
    if let Some(d) = registry_game_dir().filter(|d| valid_game_dir(d)) {
        return d;
    }
    for drive in ["C", "D", "E", "F", "G"] {
        for tail in [
            r"\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data",
            r"\Battlestate Games\EFT\EscapeFromTarkov_Data",
            r"\Games\Escape from Tarkov\EscapeFromTarkov_Data",
        ] {
            let d = format!("{drive}:{tail}");
            if valid_game_dir(&d) {
                return d;
            }
        }
    }
    r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data".to_string()
}

/// One-time, menu-startup extraction of the real CCTV menu prop (menu_fx 3D decor):
/// if packs/shared/menu/camera.{bin,png} are missing and the game install resolves, run
/// tools/extract_menu_prop.py SYNCHRONOUSLY (dataset-first, UnityPy game-file fallback)
/// before the menu scans packs. Bounded wait (~30 s): on timeout the child is left to
/// finish on its own (files then exist for the NEXT launch) and this launch just uses the
/// vector camera. Any failure falls through silently to the vector camera — the menu
/// always draws. Runs before Bevy's first frame, so a short block here is invisible.
fn ensure_menu_prop(game_dir: &str) {
    let menu_dir = crate::paths::shared_dir().join("menu");
    if menu_dir.join("camera.bin").is_file() && menu_dir.join("camera.png").is_file() {
        return; // already extracted on this machine — never re-run
    }
    if !valid_game_dir(game_dir) {
        eprintln!("menu prop: game install not found - keeping the vector camera");
        return;
    }
    let Some(root) = crate::paths::repo_root() else {
        return; // no python kit beside the exe/cwd — shipped-lite bundle, vector camera
    };
    let script = root.join("tools").join("extract_menu_prop.py");
    if !script.is_file() {
        return;
    }
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    eprintln!("menu prop: extracting the real CCTV prop (one-time, local-only)...");
    let child = std::process::Command::new(crate::paths::python_exe(root))
        .current_dir(root)
        .env("EFT_GAME_DATA", game_dir)
        .arg("tools/extract_menu_prop.py")
        .arg("--out")
        .arg(&menu_dir)
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(std::process::Stdio::inherit()) // its ASCII [menu-prop] lines go to our console
        .stderr(std::process::Stdio::inherit())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("menu prop: could not start extractor: {e}");
            return;
        }
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!(
                    "menu prop: extractor finished ({})",
                    if status.success() { "ok" } else { "failed - vector camera" }
                );
                return;
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    // Never kill it: let it finish in the background so the files are
                    // there next launch; this launch simply keeps the vector camera.
                    eprintln!("menu prop: extractor still running after 30s - vector camera this launch");
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            Err(e) => {
                eprintln!("menu prop: extractor wait failed: {e}");
                return;
            }
        }
    }
}

pub fn build_state() -> MenuState {
    let game_dir = detect_game_dir();
    // Real-asset menu decor: one-time local extraction BEFORE the pack scan / first frame.
    ensure_menu_prop(&game_dir);
    let game_fp = game_fingerprint(&game_dir);
    let (entries, total_bytes) = scan(&game_fp);
    // EFT_MENU_BUILD=<map>[,--dry-run] auto-starts a build on menu open (CLI/testing hook).
    let build = std::env::var("EFT_MENU_BUILD")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|k| BuildJob::spawn(&k, &game_dir).ok());
    MenuState {
        entries,
        game_fp,
        game_dir_edit: game_dir.clone(),
        game_dir,
        total_bytes,
        confirm_delete: None,
        show_rebuild: None,
        build,
        show_log: false,
    }
}

fn fmt_size(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0} MB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        "-".to_string()
    }
}

fn fmt_age(days: Option<f64>) -> String {
    match days {
        Some(d) if d < 1.0 => "today".to_string(),
        Some(d) => format!("{:.0}d ago", d),
        None => "-".to_string(),
    }
}

/// Fullscreen menu UI (EguiPrimaryContextPass). Only registered/active in menu mode.
#[cfg(feature = "egui")]
pub fn menu_ui(
    mut contexts: bevy_egui::EguiContexts,
    state: Option<ResMut<MenuState>>,
    mut switch: ResMut<crate::MapSwitch>,
    // Present only when the real-asset 3D CCTV spawned (menu_fx::spawn_menu_prop): flips
    // the CentralPanel transparent so the 3D world shows through, and suppresses the
    // vector-drawn camera (exactly one of the two decors ever renders).
    prop3d: Option<Res<crate::menu_fx::MenuCamProp>>,
) {
    use bevy_egui::egui::{self, Color32, RichText};
    let Some(mut state) = state else { return };
    let real_prop = prop3d.is_some();
    let Ok(ctx) = contexts.ctx_mut() else { return };

    // The menu animates without input now (camera LED blink / servo slew / idle patrol and
    // the loading-bar pulse): keep frames coming even when no events arrive, at a faster
    // cadence while a pipeline build is streaming.
    let build_running = state.build.as_ref().is_some_and(|b| b.result.is_none());
    ctx.request_repaint_after(std::time::Duration::from_millis(if build_running {
        50
    } else {
        80
    }));

    // EFT gear-screen scheme: near-black field, charcoal panels with 1px steel borders,
    // thin uppercase type, desaturated bone/beige text, muted green/red state colors.
    const BG: Color32 = Color32::from_rgb(9, 9, 9);
    const HEADER: Color32 = Color32::from_rgb(16, 16, 16);
    const CARD: Color32 = Color32::from_rgb(20, 20, 19);
    const BORDER: Color32 = Color32::from_rgb(48, 47, 44);
    const BONE: Color32 = Color32::from_rgb(215, 211, 203);
    const BEIGE: Color32 = Color32::from_rgb(199, 178, 153);
    const DIM: Color32 = Color32::from_rgb(110, 107, 100);
    const OK: Color32 = Color32::from_rgb(127, 174, 106);
    const WARN: Color32 = Color32::from_rgb(200, 140, 50);
    const BAD: Color32 = Color32::from_rgb(176, 65, 62);

    // EFT UI is all hard corners — square every widget while the menu owns the screen.
    ctx.style_mut(|s| {
        let z = egui::CornerRadius::ZERO;
        s.visuals.widgets.noninteractive.corner_radius = z;
        s.visuals.widgets.inactive.corner_radius = z;
        s.visuals.widgets.hovered.corner_radius = z;
        s.visuals.widgets.active.corner_radius = z;
        s.visuals.widgets.open.corner_radius = z;
        s.visuals.window_corner_radius = z;
        s.visuals.menu_corner_radius = z;
    });

    egui::TopBottomPanel::top("menu_header")
        .frame(egui::Frame::new().fill(HEADER).inner_margin(egui::Margin::symmetric(24, 10)))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("TARKOV STASH").color(BONE).size(24.0).strong());
                ui.add_space(14.0);
                ui.label(RichText::new("|  MAP").color(BEIGE).size(13.0));
                ui.label(RichText::new("SELECT LOCATION").color(DIM).size(13.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(fmt_size(state.total_bytes)).color(BEIGE).size(13.0),
                    );
                    ui.label(RichText::new("PACKS ON DISK").color(DIM).size(11.0));
                });
            });
        });

    // With the real 3D prop active the CentralPanel goes TRANSPARENT: the 3D world behind
    // (menu-mode ClearColor is the same #090909, set in main.rs) becomes the field and the
    // prop shows through the right gutter. Header/cards keep their own opaque fills, so the
    // list looks identical either way.
    egui::CentralPanel::default()
        .frame(
            egui::Frame::new()
                .fill(if real_prop { Color32::TRANSPARENT } else { BG })
                .inner_margin(24.0),
        )
        .show(ctx, |ui| {
            // Backdrop decor: wall-mounted CCTV servo-tracking the cursor (menu_fx). Painted
            // FIRST so every widget added later layers over it (same-layer paint order); it is
            // pure painter output — no widget, no Sense — so it can never steal clicks.
            // Skipped when the REAL 3D prop is rendering in its place (never both).
            if !real_prop {
                crate::menu_fx::security_camera(ui, ui.max_rect());
            }

            let mut delete_now: Option<usize> = None;
            let mut rescan = false;
            let mut set_confirm: Option<usize> = None;
            let mut set_rebuild: Option<usize> = None;
            let mut start_build: Option<String> = None;
            let confirm_idx = state.confirm_delete;
            // A FINISHED job (result latched) no longer blocks buttons — the log panel lingers
            // until CLOSE, but DELETE/BUILD on the rows must come back as soon as it's done.
            let building_key = state
                .build
                .as_ref()
                .filter(|b| b.result.is_none())
                .map(|b| b.key.clone());
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Right gutter: keep the map rows clear of the camera decor zone top-right
                // (the camera is painted behind, so without this it would never be seen).
                ui.set_max_width((ui.available_width() - 166.0).max(430.0));
                for i in 0..state.entries.len() {
                    let e = &state.entries[i];
                    let installed = e.pack_dir.is_some();
                    egui::Frame::new()
                        .fill(CARD)
                        .stroke(egui::Stroke::new(1.0, BORDER))
                        .corner_radius(0.0)
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.set_min_height(34.0);
                                let title = RichText::new(e.title).size(19.0).strong().color(
                                    if installed { BEIGE } else { DIM },
                                );
                                ui.add_sized([220.0, 30.0], egui::Label::new(title));
                                // Status badge.
                                let (txt, col) = if !installed {
                                    ("NOT INSTALLED", DIM)
                                } else {
                                    match e.fp_match {
                                        Some(true) => ("READY", OK),
                                        // Game-file hashes changed since this pack was built.
                                        Some(false) => ("GAME FILES UPDATED", WARN),
                                        None => ("READY (unstamped)", WARN),
                                    }
                                };
                                ui.label(RichText::new(txt).color(col).size(12.0).strong());
                                ui.add_space(10.0);
                                if installed {
                                    ui.label(RichText::new(fmt_size(e.size_bytes)).color(BEIGE));
                                    ui.label(
                                        RichText::new(format!("built {}", fmt_age(e.built_days)))
                                            .color(DIM)
                                            .size(11.0),
                                    );
                                    ui.label(
                                        RichText::new(format!("intel {}", fmt_age(e.intel_days)))
                                            .color(DIM)
                                            .size(11.0),
                                    );
                                    let tick = |on: bool, s: &str| {
                                        RichText::new(s).size(11.0).color(if on { OK } else { DIM })
                                    };
                                    ui.label(tick(e.has_volume, "light"));
                                    ui.label(tick(e.has_grass, "grass"));
                                    ui.label(tick(e.has_gamedata, "zones"));
                                    ui.label(tick(e.has_icons, "icons"));
                                }
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let this_building =
                                            building_key.as_deref() == Some(e.key);
                                        let any_building = building_key.is_some();
                                        if installed {
                                            let play = egui::Button::new(
                                                RichText::new("PLAY").strong().color(Color32::BLACK),
                                            )
                                            .fill(BEIGE)
                                            .corner_radius(0.0);
                                            if ui.add_sized([84.0, 30.0], play).clicked() {
                                                switch.0 = e.pack_dir.clone();
                                            }
                                            // UPDATE sits BETWEEN delete and play: shown when the
                                            // game-file hashes no longer match the pack's stamp —
                                            // re-runs the pipeline so the data catches up.
                                            if e.fp_match == Some(false) {
                                                let upd = egui::Button::new(
                                                    RichText::new(if this_building {
                                                        "UPDATING..."
                                                    } else {
                                                        "UPDATE"
                                                    })
                                                    .color(Color32::BLACK)
                                                    .strong(),
                                                )
                                                .fill(WARN)
                                                .corner_radius(0.0);
                                                if ui
                                                    .add_enabled_ui(!any_building, |ui| {
                                                        ui.add_sized([84.0, 30.0], upd).on_hover_text(
                                                            "game files changed since this pack was built - \
                                                             run the pipeline again (data may be out of date)",
                                                        )
                                                    })
                                                    .inner
                                                    .clicked()
                                                {
                                                    start_build = Some(e.key.to_string());
                                                }
                                            }
                                            // Tarkov-style destructive button: red fill, black text.
                                            let del_btn = |t: &str| {
                                                egui::Button::new(
                                                    RichText::new(t).color(Color32::BLACK).strong(),
                                                )
                                                .fill(BAD)
                                                .corner_radius(0.0)
                                            };
                                            if confirm_idx == Some(i) {
                                                if ui
                                                    .add_sized([120.0, 30.0], del_btn("CONFIRM"))
                                                    .clicked()
                                                {
                                                    delete_now = Some(i);
                                                }
                                            } else if ui
                                                .add_enabled_ui(!this_building, |ui| {
                                                    ui.add_sized([84.0, 30.0], del_btn("DELETE"))
                                                })
                                                .inner
                                                .clicked()
                                            {
                                                set_confirm = Some(i);
                                            }
                                        } else {
                                            let b = egui::Button::new(RichText::new(
                                                if this_building { "BUILDING..." } else { "BUILD" },
                                            ));
                                            if ui.add_enabled(!any_building, b).clicked() {
                                                start_build = Some(e.key.to_string());
                                            }
                                        }
                                    },
                                );
                            });
                        });
                    ui.add_space(6.0);
                }
            });

            if let Some(i) = set_confirm {
                state.confirm_delete = Some(i);
            }
            if let Some(i) = set_rebuild {
                state.show_rebuild = Some(i);
            }
            if let Some(i) = delete_now {
                if let Some(dir) = state.entries[i].pack_dir.clone() {
                    // Safety: only ever delete inside the resolved packs root (entry paths are
                    // built from it, but belt-and-braces against future edits).
                    let p = std::path::Path::new(&dir);
                    if p.starts_with(crate::paths::packs_root()) {
                        match std::fs::remove_dir_all(p) {
                            Ok(()) => info!("menu: deleted {dir}"),
                            Err(e) => error!("menu: delete {dir} failed: {e}"),
                        }
                    } else {
                        error!("menu: refusing to delete outside packs root: {dir}");
                    }
                    rescan = true;
                }
                state.confirm_delete = None;
            }
            if rescan {
                let (entries, total) = scan(&state.game_fp);
                state.entries = entries;
                state.total_bytes = total;
            }

            let _ = set_rebuild; // (legacy instructions window removed — BUILD runs the pipeline)

            // Kick a build (one at a time; tools/build_map.py streams staged output).
            if let Some(key) = start_build {
                match BuildJob::spawn(&key, &state.game_dir) {
                    Ok(job) => {
                        info!("menu: building '{key}' via tools/build_map.py");
                        state.build = Some(job);
                        state.show_log = false; // fresh panel starts with the log collapsed
                    }
                    Err(e) => error!("menu: failed to start build for {key}: {e}"),
                }
            }

            // ---- Build progress (EFT loader style): segmented bar + stage line; the raw
            // streaming tail stays collapsed behind SHOW LOG and auto-expands on failure ----
            let mut clear_build = false;
            let mut toggle_log = false;
            let mut expand_log = false;
            // Auto-refresh the map rows the moment the pipeline finishes (the panel itself
            // stays up until CLOSE so the result remains readable). result doubles as the
            // "already rescanned" latch; a failure force-expands the log tail so the error
            // lines are visible without a click.
            if let Some(job) = &mut state.build {
                let (_, _, finished, ok) = job.snapshot(0);
                if finished && job.result.is_none() {
                    job.result = Some(ok);
                    rescan = true;
                    if !ok {
                        expand_log = true;
                    }
                }
            }
            if expand_log {
                state.show_log = true;
            }
            let show_log = state.show_log;
            if let Some(job) = &state.build {
                let (stage, tail, finished, ok) = job.snapshot(12);
                let failed = finished && !ok;
                ui.add_space(10.0);
                egui::Frame::new()
                    .fill(HEADER)
                    .stroke(egui::Stroke::new(1.0, BORDER))
                    .inner_margin(10.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("BUILDING: {}", job.key.to_uppercase()))
                                    .color(BONE)
                                    .strong(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if finished {
                                        let col = if ok { OK } else { BAD };
                                        let txt = if ok { "DONE" } else { "FAILED" };
                                        ui.label(RichText::new(txt).color(col).strong());
                                        if ui.button("CLOSE").clicked() {
                                            clear_build = true;
                                        }
                                    } else if ui
                                        .button(RichText::new("CANCEL").color(BAD))
                                        .clicked()
                                    {
                                        job.cancel();
                                    }
                                    // The tail is hidden by default — the loader bar carries
                                    // the status; the raw log is one click away.
                                    if ui
                                        .button(if show_log { "HIDE LOG" } else { "SHOW LOG" })
                                        .clicked()
                                    {
                                        toggle_log = true;
                                    }
                                    // Full captured log (the panel shows only a tail) — for
                                    // diagnosing which stage failed / sharing the output.
                                    if ui.button("COPY LOG").clicked() {
                                        ui.ctx().copy_text(job.full_log());
                                    }
                                },
                            );
                        });
                        // Progress from the [STAGE i/N] markers: finished stages count full,
                        // the running stage counts half; [BUILD OK] pins 100%.
                        let frac = if stage.starts_with("[BUILD OK]") || (finished && ok) {
                            1.0
                        } else {
                            stage
                                .strip_prefix("[STAGE ")
                                .and_then(|s| s.split(']').next())
                                .and_then(|s| {
                                    let (i, n) = s.split_once('/')?;
                                    let i: f32 = i.trim().parse().ok()?;
                                    let n: f32 = n.trim().parse().ok()?;
                                    let done_stage = stage.contains(": done")
                                        || stage.contains(": skipped");
                                    Some(((i - 1.0 + if done_stage { 1.0 } else { 0.5 }) / n)
                                        .clamp(0.0, 1.0))
                                })
                                .unwrap_or(0.0)
                        };
                        // "LOADING OBJECTS..." style stage line for the loader bar: the text
                        // between the [STAGE] marker and its status suffix, uppercased and
                        // ASCII-whitelisted (menu glyph set is plain ASCII only).
                        let stage_txt = if failed {
                            "BUILD FAILED".to_string()
                        } else if finished {
                            "BUILD COMPLETE".to_string()
                        } else {
                            let mut s = stage
                                .split(']')
                                .nth(1)
                                .unwrap_or("")
                                .split(':')
                                .next()
                                .unwrap_or("")
                                .trim()
                                .to_ascii_uppercase();
                            s.retain(|c| c.is_ascii_graphic() || c == ' ');
                            s.truncate(38);
                            if s.is_empty() {
                                s = "STARTING".into();
                            }
                            s.push_str("...");
                            s
                        };
                        ui.add_space(8.0);
                        crate::menu_fx::eft_loading_bar(
                            ui,
                            frac,
                            &stage_txt,
                            job.started.elapsed().as_secs_f32(),
                            failed,
                        );
                        if show_log {
                            ui.add_space(6.0);
                            for line in &tail {
                                ui.label(
                                    RichText::new(line).color(DIM).size(11.0).monospace(),
                                );
                            }
                        }
                    });
            }
            if toggle_log {
                state.show_log = !state.show_log;
            }
            if clear_build {
                state.build = None;
                rescan = true;
            }
            if rescan {
                let (entries, total) = scan(&state.game_fp);
                state.entries = entries;
                state.total_bytes = total;
            }

            ui.add_space(8.0);
            ui.separator();
            // Game install path: autodetected (env > saved > registry > probe), editable here;
            // SET validates, persists to eft_viewer.config.json and re-fingerprints the packs.
            ui.horizontal(|ui| {
                ui.label(RichText::new("GAME INSTALL").color(DIM).size(11.0));
                let mut edit = state.game_dir_edit.clone();
                ui.add(
                    egui::TextEdit::singleline(&mut edit)
                        .desired_width(520.0)
                        .font(egui::TextStyle::Monospace),
                );
                state.game_dir_edit = edit;
                let dirty = state.game_dir_edit != state.game_dir;
                if ui
                    .add_enabled(dirty, egui::Button::new(RichText::new("SET").color(BONE)))
                    .clicked()
                {
                    if valid_game_dir(&state.game_dir_edit) {
                        state.game_dir = state.game_dir_edit.clone();
                        save_config_game_dir(&state.game_dir);
                        state.game_fp = game_fingerprint(&state.game_dir);
                        let (entries, total) = scan(&state.game_fp);
                        state.entries = entries;
                        state.total_bytes = total;
                    } else {
                        error!("menu: '{}' does not look like EscapeFromTarkov_Data", state.game_dir_edit);
                    }
                }
                match &state.game_fp {
                    Some(fp) => ui.label(
                        RichText::new(format!("[{}]", &fp[..8])).color(OK).size(11.0),
                    ),
                    None => ui.label(
                        RichText::new("NOT FOUND - set the EscapeFromTarkov_Data path")
                            .color(WARN)
                            .size(11.0),
                    ),
                };
            });
        });
}
