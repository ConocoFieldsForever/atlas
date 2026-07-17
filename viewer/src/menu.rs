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
}

impl BuildJob {
    pub fn spawn(key: &str) -> std::io::Result<Self> {
        use std::io::BufRead;
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000; // no console popping over the menu
        let mut child = std::process::Command::new("python")
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
            });
        }
    }

    pub fn cancel(&self) {
        let _ = self.child.lock().unwrap().kill();
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
    let installed = |key: &str| format!("packs/{key}.eftpack");
    // Known roster first, then any extra packs on disk.
    let mut extra: Vec<String> = std::fs::read_dir("packs")
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
            intel_days: age_days(&p.join("loot.json")),
            has_volume,
            has_grass: p.join("grass.bin").exists(),
            has_gamedata: p.join("gamedata.json").exists(),
            has_icons: p.join("icons").is_dir(),
            fp_match,
        });
    }
    (entries, total)
}

pub fn build_state() -> MenuState {
    let game_dir = std::env::var("EFT_GAME_DATA").unwrap_or_else(|_| {
        r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data".to_string()
    });
    let game_fp = game_fingerprint(&game_dir);
    let (entries, total_bytes) = scan(&game_fp);
    // EFT_MENU_BUILD=<map>[,--dry-run] auto-starts a build on menu open (CLI/testing hook).
    let build = std::env::var("EFT_MENU_BUILD")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|k| BuildJob::spawn(&k).ok());
    MenuState {
        entries,
        game_fp,
        game_dir,
        total_bytes,
        confirm_delete: None,
        show_rebuild: None,
        build,
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
) {
    use bevy_egui::egui::{self, Color32, RichText};
    let Some(mut state) = state else { return };
    let Ok(ctx) = contexts.ctx_mut() else { return };

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

    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(BG).inner_margin(24.0))
        .show(ctx, |ui| {

            let mut delete_now: Option<usize> = None;
            let mut rescan = false;
            let mut set_confirm: Option<usize> = None;
            let mut set_rebuild: Option<usize> = None;
            let mut start_build: Option<String> = None;
            let confirm_idx = state.confirm_delete;
            let building_key = state.build.as_ref().map(|b| b.key.clone());
            egui::ScrollArea::vertical().show(ui, |ui| {
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
                                        Some(false) => ("GAME UPDATED - REBUILD", BAD),
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
                                            if confirm_idx == Some(i) {
                                                if ui
                                                    .button(RichText::new("CONFIRM DELETE").color(BAD))
                                                    .clicked()
                                                {
                                                    delete_now = Some(i);
                                                }
                                            } else if ui
                                                .add_enabled(!this_building, egui::Button::new("DELETE"))
                                                .clicked()
                                            {
                                                set_confirm = Some(i);
                                            }
                                            if e.fp_match == Some(false) {
                                                let b = egui::Button::new(
                                                    RichText::new(if this_building {
                                                        "BUILDING..."
                                                    } else {
                                                        "REBUILD"
                                                    })
                                                    .color(WARN),
                                                );
                                                if ui.add_enabled(!any_building, b).clicked() {
                                                    start_build = Some(e.key.to_string());
                                                }
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
                    match std::fs::remove_dir_all(&dir) {
                        Ok(()) => info!("menu: deleted {dir}"),
                        Err(e) => error!("menu: delete {dir} failed: {e}"),
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
                match BuildJob::spawn(&key) {
                    Ok(job) => {
                        info!("menu: building '{key}' via tools/build_map.py");
                        state.build = Some(job);
                    }
                    Err(e) => error!("menu: failed to start build for {key}: {e}"),
                }
            }

            // ---- Build progress (Tarkov task-log style): stage header + streaming tail ----
            let mut clear_build = false;
            if let Some(job) = &state.build {
                let (stage, tail, finished, ok) = job.snapshot(12);
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
                            ui.label(RichText::new(&stage).color(
                                if stage.contains("FAILED") { BAD } else { BEIGE },
                            ));
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
                                    } else {
                                        ui.spinner();
                                        if ui.button(RichText::new("CANCEL").color(BAD)).clicked()
                                        {
                                            job.cancel();
                                        }
                                    }
                                },
                            );
                        });
                        ui.add_space(4.0);
                        for line in &tail {
                            ui.label(
                                RichText::new(line).color(DIM).size(11.0).monospace(),
                            );
                        }
                    });
                if finished && !ok {
                    // leave the panel up so the failure tail stays readable
                }
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
            ui.horizontal(|ui| {
                match &state.game_fp {
                    Some(fp) => ui.label(
                        RichText::new(format!("game install: {} [{}]", state.game_dir, &fp[..8]))
                            .color(DIM)
                            .size(11.0),
                    ),
                    None => ui.label(
                        RichText::new(format!("game install not found at {}", state.game_dir))
                            .color(WARN)
                            .size(11.0),
                    ),
                };
            });
        });
}
