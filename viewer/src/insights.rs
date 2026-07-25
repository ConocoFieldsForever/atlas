//! insights.rs — the INSIGHTS tab: netcode position breadcrumbs mined from EFT's own logs.
//!
//! EFT's application log sporadically prints the player's true world position in plain text —
//! netcode speed-limit events of the shape
//! `Reason:PacketsQueue, Position:(-24.995, 21.394, 109.302), ... CurrentState:Run` — a few per
//! raid (docs/GAME_DATA_SOURCES.md, finding A11). Nobody asks for these; the game just writes
//! them. Harvested across EVERY session folder on disk they become a per-map trail of everywhere
//! the netcode happened to pin you: free breadcrumbs between screenshot fixes, entirely passive.
//!
//! Positions are attributed to the raid active at that point of the file (the most recent
//! `scene preset path:maps/<bundle>.bundle` line above them — same bundle→map table the live
//! link uses) and bridged with the pipeline-wide X-flip, viewer = (-x, y, z).
//!
//! The scan runs on a plain thread (it is pure file I/O over a few tens of MB) once at first
//! use, with a REFRESH button to pick up the current session's newest lines.

use bevy::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;

/// One harvested netcode position, in viewer space.
pub struct Crumb {
    pub pos: Vec3,
    /// Movement state the netcode reported ("Run", "Sprint", ...); empty when absent.
    pub state: String,
    /// The log line's own timestamp prefix ("2026-07-24 21:52:27"); empty when unparsable.
    pub when: String,
}

#[derive(Default)]
pub struct ScanResult {
    pub by_map: HashMap<String, Vec<Crumb>>,
    /// How many log session folders were read (shown in the footer as provenance).
    pub sessions: usize,
}

#[derive(Resource, Default)]
pub struct Insights {
    pub data: ScanResult,
    /// A scan has completed at least once (the panel shows counts, not a spinner).
    pub scanned: bool,
    /// Draw the current map's crumbs in the world (gizmo dots).
    pub show_on_map: bool,
    /// In-flight scan, if any.
    rx: Option<Mutex<Receiver<ScanResult>>>,
}

impl Insights {
    fn scanning(&self) -> bool {
        self.rx.is_some()
    }
}

/// Every candidate Logs root, mirroring game_watch::latest_log_folder's layout knowledge:
/// `detect_game_dir` returns the DATA dir; EFT's logs live beside the EXE one level up.
fn logs_roots() -> Vec<PathBuf> {
    let game = PathBuf::from(crate::menu::detect_game_dir());
    let mut out = Vec::new();
    for c in [
        Some(game.join("Logs")),
        game.parent().map(|p| p.join("Logs")),
        Some(game.join("build").join("Logs")),
    ]
    .into_iter()
    .flatten()
    {
        if c.is_dir() && !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// Parse `Position:(x, y, z)` out of a line. Bounds-checked, never panics on malformed text.
fn parse_position(line: &str) -> Option<Vec3> {
    let start = line.find("Position:(")? + "Position:(".len();
    let end = start + line[start..].find(')')?;
    let mut it = line[start..end].split(',').map(|v| v.trim().parse::<f32>());
    match (it.next(), it.next(), it.next()) {
        (Some(Ok(x)), Some(Ok(y)), Some(Ok(z))) => {
            (x.is_finite() && y.is_finite() && z.is_finite()).then(|| Vec3::new(-x, y, z))
        }
        _ => None,
    }
}

/// Blocking scan of every session folder — runs on the worker thread only.
fn scan() -> ScanResult {
    let mut out = ScanResult::default();
    for root in logs_roots() {
        let Ok(rd) = std::fs::read_dir(&root) else { continue };
        let mut folders: Vec<PathBuf> = rd
            .flatten()
            .filter(|e| {
                e.file_name().to_string_lossy().starts_with("log_")
                    && e.file_type().map(|t| t.is_dir()).unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        folders.sort(); // chronological (names embed the date)
        for folder in folders {
            let Ok(rd) = std::fs::read_dir(&folder) else { continue };
            let mut apps: Vec<PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    let n = p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
                    n.contains("application") && n.ends_with(".log")
                })
                .collect();
            if apps.is_empty() {
                continue;
            }
            apps.sort(); // _000, _001, ... rotation order
            out.sessions += 1;
            let mut cur_map: Option<&'static str> = None;
            for app in apps {
                let Ok(text) = std::fs::read_to_string(&app) else { continue };
                for line in text.lines() {
                    if let Some(rest) = line.split("scene preset path:maps/").nth(1) {
                        if let Some(bundle) = rest.split(".bundle").next() {
                            cur_map = crate::game_watch::bundle_to_map(bundle.trim());
                        }
                        continue;
                    }
                    if !line.contains("Position:(") {
                        continue;
                    }
                    let Some(map) = cur_map else { continue };
                    let Some(pos) = parse_position(line) else { continue };
                    let state = line
                        .split("CurrentState:")
                        .nth(1)
                        .map(|s| {
                            s.chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>()
                        })
                        .unwrap_or_default();
                    // "2026-07-24 21:52:27.492|..." -> keep to the seconds.
                    let when = line.split('|').next().unwrap_or("").chars().take(19).collect();
                    let v = out.by_map.entry(map.to_string()).or_default();
                    if v.len() < 5000 {
                        // runaway-log backstop; a real map sees a few crumbs per raid
                        v.push(Crumb { pos, state, when });
                    }
                }
            }
        }
    }
    out
}

fn start_scan(ins: &mut Insights) {
    if ins.scanning() {
        return;
    }
    let (tx, rx): (Sender<ScanResult>, Receiver<ScanResult>) = std::sync::mpsc::channel();
    ins.rx = Some(Mutex::new(rx));
    std::thread::Builder::new()
        .name("atlas-insights-scan".into())
        .spawn(move || {
            let _ = tx.send(scan());
        })
        .ok();
}

fn poll_scan(mut ins: ResMut<Insights>) {
    let done = match ins.rx.as_ref().and_then(|m| m.lock().ok()) {
        Some(rx) => rx.try_recv().ok(),
        None => None,
    };
    if let Some(res) = done {
        let n: usize = res.by_map.values().map(Vec::len).sum();
        info!("insights: {} netcode positions across {} maps ({} sessions)", n, res.by_map.len(), res.sessions);
        ins.data = res;
        ins.scanned = true;
        ins.rx = None;
    }
}

/// The map id of the loaded pack ("streets"), if any.
fn current_map(pack: Option<&crate::render::LoadedPack>) -> Option<String> {
    pack.and_then(|p| {
        p.0.root.file_name()?.to_str()?.strip_suffix(".eftpack").map(str::to_string)
    })
}

/// Gizmo dots for the current map's crumbs (gated by the panel's "show on map" toggle).
fn draw_crumbs(
    ins: Res<Insights>,
    pack: Option<Res<crate::render::LoadedPack>>,
    mut gizmos: Gizmos,
) {
    if !ins.show_on_map || !ins.scanned {
        return;
    }
    let Some(map) = current_map(pack.as_deref()) else { return };
    let Some(crumbs) = ins.data.by_map.get(&map) else { return };
    for c in crumbs {
        // A small upright diamond: visible from above (the map view) and from ground level.
        let p = c.pos;
        gizmos.line(p - Vec3::Y * 0.4, p + Vec3::Y * 0.4, Color::srgb(0.45, 0.78, 0.98));
        gizmos.circle(
            bevy::math::Isometry3d::new(p, Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
            0.35,
            Color::srgb(0.45, 0.78, 0.98),
        );
    }
}

/// The right-panel tab body (same SidePanel slot + gating pattern as ui::level_panel).
#[cfg(feature = "egui")]
fn insights_panel(
    mut contexts: bevy_egui::EguiContexts,
    tab: Res<crate::ui::RightPanelTab>,
    menu: Option<Res<crate::menu::MenuState>>,
    pack: Option<Res<crate::render::LoadedPack>>,
    mut ins: ResMut<Insights>,
    mut cam_cmd: ResMut<crate::CameraCommand>,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};
    if menu.is_some() || *tab != crate::ui::RightPanelTab::Insights {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    // First open kicks the scan (plain file I/O on its own thread; the panel stays responsive).
    if !ins.scanned && !ins.scanning() {
        start_scan(&mut ins);
    }
    const DIM: egui::Color32 = theme::MUTED;
    let cur = current_map(pack.as_deref());
    egui::SidePanel::right("insights_panel")
        .default_width(300.0)
        .frame(theme::panel_frame())
        .show(ctx, |ui| {
            ui.label(theme::title("INSIGHTS"));
            ui.label(
                RichText::new(
                    "Positions the game's own netcode wrote into its logs - a passive trail of \
where you have been, a few points per raid.",
                )
                .size(10.0)
                .color(DIM),
            );
            ui.add_space(theme::SP_MD);
            ui.horizontal(|ui| {
                if ui.add_enabled(!ins.scanning(), egui::Button::new(RichText::new("REFRESH").size(11.0))).clicked() {
                    start_scan(&mut ins);
                }
                if ins.scanning() {
                    ui.spinner();
                    ui.label(RichText::new("scanning logs\u{2026}").size(10.0).color(DIM));
                }
            });
            if !ins.scanned {
                return;
            }
            ui.add_space(theme::SP_MD);
            let mut show = ins.show_on_map;
            if ui.checkbox(&mut show, RichText::new("Show on map").size(11.0)).changed() {
                ins.show_on_map = show;
            }
            ui.add_space(theme::SP_MD);

            // Per-map counts, current map first + expanded with fly-to rows.
            let mut maps: Vec<(&String, usize)> =
                ins.data.by_map.iter().map(|(k, v)| (k, v.len())).collect();
            maps.sort_by(|a, b| b.1.cmp(&a.1));
            ui.label(RichText::new("POSITIONS BY MAP").color(DIM).size(11.0));
            for (map, n) in &maps {
                let here = cur.as_deref() == Some(map.as_str());
                ui.label(
                    RichText::new(format!(
                        "{}  {n}{}",
                        crate::inspect::prettify(map),
                        if here { "  (this map)" } else { "" }
                    ))
                    .size(11.0)
                    .color(if here { theme::TEXT_BRIGHT } else { DIM }),
                );
            }
            let Some(curmap) = cur else { return };
            let Some(crumbs) = ins.data.by_map.get(&curmap) else { return };
            ui.add_space(theme::SP_MD);
            ui.label(RichText::new("THIS MAP - NEWEST FIRST").color(DIM).size(11.0));
            egui::ScrollArea::vertical().show(ui, |ui| {
                for c in crumbs.iter().rev().take(200) {
                    ui.horizontal(|ui| {
                        if ui.small_button("go").on_hover_text("Fly the camera here").clicked() {
                            cam_cmd.fly_to = Some(c.pos);
                        }
                        let label = if c.state.is_empty() {
                            c.when.clone()
                        } else {
                            format!("{}  \u{00B7}  {}", c.when, c.state)
                        };
                        ui.label(RichText::new(label).size(10.0).color(theme::TEXT_BRIGHT));
                    });
                }
                if crumbs.len() > 200 {
                    ui.label(
                        RichText::new(format!("\u{2026} and {} older", crumbs.len() - 200))
                            .size(10.0)
                            .color(DIM),
                    );
                }
            });
        });
}

pub struct InsightsPlugin;
impl Plugin for InsightsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Insights>()
            .add_systems(Update, (poll_scan, draw_crumbs));
        #[cfg(feature = "egui")]
        app.add_systems(bevy_egui::EguiPrimaryContextPass, insights_panel);
    }
}
