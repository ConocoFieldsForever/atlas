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
    MenuState {
        entries,
        game_fp,
        game_dir,
        total_bytes,
        confirm_delete: None,
        show_rebuild: None,
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

    const BG: Color32 = Color32::from_rgb(12, 12, 11);
    const CARD: Color32 = Color32::from_rgb(22, 22, 20);
    const BEIGE: Color32 = Color32::from_rgb(199, 178, 153);
    const DIM: Color32 = Color32::from_rgb(120, 115, 105);
    const OK: Color32 = Color32::from_rgb(120, 160, 90);
    const WARN: Color32 = Color32::from_rgb(200, 140, 50);
    const BAD: Color32 = Color32::from_rgb(190, 70, 60);

    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(BG).inner_margin(24.0))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                ui.label(RichText::new("TARKOV STASH").color(BEIGE).size(34.0).strong());
                ui.label(RichText::new("RAID PLANNER - SELECT LOCATION").color(DIM).size(13.0));
            });
            ui.add_space(14.0);

            let mut delete_now: Option<usize> = None;
            let mut rescan = false;
            let mut set_confirm: Option<usize> = None;
            let mut set_rebuild: Option<usize> = None;
            let confirm_idx = state.confirm_delete;
            egui::ScrollArea::vertical().show(ui, |ui| {
                for i in 0..state.entries.len() {
                    let e = &state.entries[i];
                    let installed = e.pack_dir.is_some();
                    egui::Frame::new()
                        .fill(CARD)
                        .corner_radius(3.0)
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
                                        if installed {
                                            let play = egui::Button::new(
                                                RichText::new("PLAY").strong().color(Color32::BLACK),
                                            )
                                            .fill(BEIGE);
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
                                            } else if ui.button("DELETE").clicked() {
                                                set_confirm = Some(i);
                                            }
                                            if e.fp_match == Some(false)
                                                && ui.button("REBUILD").clicked()
                                            {
                                                set_rebuild = Some(i);
                                            }
                                        } else if ui.button("HOW TO BUILD").clicked() {
                                            set_rebuild = Some(i);
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

            if let Some(i) = state.show_rebuild {
                let key = state.entries.get(i).map(|e| e.key).unwrap_or("<map>");
                let mut open = true;
                egui::Window::new("Build / rebuild map data")
                    .collapsible(false)
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.label(format!(
                            "Run the extraction pipeline for '{key}' (see extraction/README.md):"
                        ));
                        let cmd = format!(
                            "python extraction/unity/eft_extract_v2.py {key} && \
                             python -m eft_pipeline.assemble_bevy {key} && \
                             python extraction/intel/extract_gamedata.py {key} && \
                             python tools/stamp_fingerprint.py packs/{key}.eftpack"
                        );
                        ui.code(&cmd);
                        if ui.button("Copy command").clicked() {
                            ui.ctx().copy_text(cmd);
                        }
                    });
                if !open {
                    state.show_rebuild = None;
                }
            }

            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("packs on disk: {}", fmt_size(state.total_bytes)))
                        .color(BEIGE),
                );
                ui.add_space(16.0);
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
