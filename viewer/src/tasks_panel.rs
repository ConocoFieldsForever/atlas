//! tasks_panel.rs — the revamped TASKS / QUESTS tab (a self-contained right-panel module).
//!
//! This SUPERSEDES the old quest checklist that lived inside `ui::layers_panel`. It is built as a
//! drop-in module for the toolbar-tab router: the router owns the `SidePanel`/egui frame and calls
//! [`tasks_panel_ui`] once per frame with a single `SystemParam` bundle ([`TasksPanelParams`]).
//!
//! WHAT IT DOES (vs the old section):
//!   * loads the FULL global quest catalog (`tasks.json`, build_tasks.py) into [`TaskCatalog`] with
//!     every field the schema carries — objective TYPE, required ITEMS (+count/FIR), quest items,
//!     mark items, kill targets, extract names, and per-map objective LOCATIONS. The old runtime
//!     model (`poi::QuestData`) only kept this-map objective zones, so it could not show items or
//!     off-map tasks. We keep `poi::QuestData` untouched (it still drives the on-map markers).
//!   * TRACK on/off per task -> `ui::QuestTracker.active`, the SAME set `poi::apply_quest_visibility`
//!     and `poi::draw_quest_outlines` already read, so the existing marker/outline wiring is reused
//!     verbatim (tracking a task focuses the map to it; checking one also flips the Quest layer on).
//!   * FILTER BY MAP (default: the current pack's map, from `manifest.dataset`), Kappa/Lightkeeper,
//!     level gate, and a text search over task/trader/item names. Tasks are GROUPED BY TRADER.
//!   * REQUIRED ITEMS with ICONS via a mirror of `inspect`'s icon cache (`packs/shared/icons/<slug>.png`,
//!     the same `icon_slug` contract). Items whose icon isn't cached degrade to a text chip.
//!   * OBJECTIVE LOCATIONS: each objective that has a zone/point on this map gets a "go" button
//!     (`CameraCommand::fly_to`) and a "route" button (`pathfind::RouteRequest`).
//!
//! egui 0.32 / bevy_egui 0.37; drawn in `EguiPrimaryContextPass` (the router's schedule). Labels are
//! ASCII plus the whitelisted glyphs only: `\u{00D7}` (x) `\u{2192}` (->) `\u{00B7}` (.) `\u{25CF}`
//! (dot) `\u{2026}` (...) `\u{2264}` (<=).
//!
//! Clone-edit-compare discipline (mirrors `ui::layers_panel`): editing a `ResMut` through an egui
//! widget marks the resource CHANGED every frame it renders, which would make the poi visibility
//! systems rewrite every quest marker per frame. So the tracker / layer toggle are edited on LOCAL
//! copies and written back only on a real delta.

use bevy::prelude::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ============================================================================================
// DATA MODEL — the full quest catalog (loaded once from tasks.json; map-agnostic).
// ============================================================================================

/// The global task catalog, loaded from `tasks.json` at startup. Empty by default so the resource
/// always exists (the panel just shows "no tasks" without a file). `loaded` latches after the first
/// attempt (with a pack up) so the one-shot loader stops re-running.
#[derive(Resource, Default)]
pub struct TaskCatalog {
    pub tasks: Vec<TaskDef>,
    /// Set once the load has run (whether or not a file was found), so `load_task_catalog` is a
    /// single bool check per frame thereafter.
    pub loaded: bool,
    /// Provenance string for the footer ("tarkov.dev").
    pub source: String,
}

/// One task with every field the panel surfaces. Read only by the egui panel; the `allow` keeps the
/// non-egui build (`--no-default-features`) warning-clean, matching `ui.rs`/`poi.rs` house style.
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct TaskDef {
    pub id: String,
    pub name: String,
    /// Trader display name ("Prapor"); empty when the task has no giver.
    pub trader: String,
    /// Primary map key ("customs"); empty for map-less tasks.
    pub map: String,
    /// EVERY map this task touches (primary + objective/zone/location maps) — the map filter uses it.
    pub maps: Vec<String>,
    /// 0 = no level gate.
    pub min_level: u32,
    pub kappa: bool,
    pub lk: bool,
    /// Prerequisite task NAMES (shown so the planner can see ordering).
    pub requires: Vec<String>,
    pub objectives: Vec<ObjectiveDef>,
}

/// One objective. Fields are all optional in the schema and default to empty/false here.
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct ObjectiveDef {
    /// Objective TYPE: visit / giveItem / findItem / shoot / mark / plantItem / extract /
    /// findQuestItem / giveQuestItem / buildWeapon / skill / ... — drives the tag + colour.
    pub kind: String,
    pub desc: String,
    pub optional: bool,
    /// 0 = unspecified; else "need N".
    pub count: u32,
    /// Found-in-raid requirement.
    pub fir: bool,
    /// Handover/find item display names (may be a long "any X" enumeration).
    pub items: Vec<String>,
    /// The quest item to find/plant/hand over (findQuestItem/giveQuestItem/plantQuestItem).
    pub quest_item: Option<String>,
    /// The tool used to complete a `mark` objective (e.g. "MS2000 Marker").
    pub marker_item: Option<String>,
    /// Kill targets for `shoot` objectives (enemy names — no icons).
    pub targets: Vec<String>,
    /// Extract name for `extract` objectives.
    pub exit: Option<String>,
    /// Objective zones (a task can span maps; each carries its own map key + bridged position).
    pub zones: Vec<ObjZone>,
    /// Quest-item find spots (bridged points), for objectives without an explicit zone.
    pub item_locations: Vec<ObjLoc>,
}

/// A single objective zone bridged to viewer space.
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct ObjZone {
    pub map: String,
    /// Point (already viewer space); None for outline-only zones.
    pub pos: Option<Vec3>,
    /// true when the zone carries a footprint polygon (drawn on the map by poi.rs).
    pub has_outline: bool,
}

/// A quest-item find location: a map key + the candidate points (viewer space).
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct ObjLoc {
    pub map: String,
    pub pts: Vec<Vec3>,
}

// ---- tasks.json raw shapes (deserialize; then converted to the model above) --------------------

#[derive(Deserialize)]
struct RawFile {
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    tasks: Vec<RawTask>,
}
#[derive(Deserialize)]
struct RawTask {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    trader: Option<String>,
    #[serde(default)]
    map: Option<String>,
    #[serde(default)]
    maps: Vec<String>,
    #[serde(default, rename = "minLevel")]
    min_level: u32,
    #[serde(default)]
    kappa: bool,
    #[serde(default)]
    lk: bool,
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    objectives: Vec<RawObj>,
}
#[derive(Deserialize)]
struct RawObj {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    desc: String,
    #[serde(default)]
    optional: bool,
    #[serde(default)]
    count: u32,
    #[serde(default)]
    fir: bool,
    #[serde(default)]
    items: Vec<String>,
    #[serde(default, rename = "questItem")]
    quest_item: Option<String>,
    #[serde(default, rename = "markerItem")]
    marker_item: Option<String>,
    #[serde(default)]
    targets: Vec<String>,
    #[serde(default)]
    exit: Option<String>,
    #[serde(default)]
    zones: Vec<RawZone>,
    #[serde(default, rename = "itemLocations")]
    item_locations: Vec<RawLoc>,
}
#[derive(Deserialize)]
struct RawZone {
    #[serde(default)]
    map: Option<String>,
    #[serde(default)]
    pos: Option<[f32; 3]>,
    #[serde(default)]
    outline: Option<Vec<[f32; 3]>>,
}
#[derive(Deserialize)]
struct RawLoc {
    #[serde(default)]
    map: Option<String>,
    #[serde(default)]
    pts: Vec<[f32; 3]>,
}

fn convert_task(t: RawTask) -> TaskDef {
    let objectives = t
        .objectives
        .into_iter()
        .map(|o| ObjectiveDef {
            kind: o.kind,
            desc: o.desc,
            optional: o.optional,
            count: o.count,
            fir: o.fir,
            items: o.items,
            quest_item: o.quest_item.filter(|s| !s.is_empty()),
            marker_item: o.marker_item.filter(|s| !s.is_empty()),
            targets: o.targets,
            exit: o.exit.filter(|s| !s.is_empty()),
            zones: o
                .zones
                .into_iter()
                .map(|z| ObjZone {
                    map: z.map.unwrap_or_default(),
                    pos: z.pos.map(Vec3::from),
                    has_outline: z.outline.map(|v| v.len() >= 3).unwrap_or(false),
                })
                .collect(),
            item_locations: o
                .item_locations
                .into_iter()
                .map(|l| ObjLoc {
                    map: l.map.unwrap_or_default(),
                    pts: l.pts.into_iter().map(Vec3::from).collect(),
                })
                .collect(),
        })
        .collect();
    TaskDef {
        id: t.id,
        name: t.name,
        trader: t.trader.unwrap_or_default(),
        map: t.map.unwrap_or_default(),
        maps: t.maps,
        min_level: t.min_level,
        kappa: t.kappa,
        lk: t.lk,
        requires: t.requires,
        objectives,
    }
}

/// dataset "interchange_v2" -> tasks.json map key "interchange" (strip a `_vN` suffix). Same rule as
/// `poi::map_key` (kept local so the module is standalone).
fn map_key(dataset: &str) -> String {
    if let Some((base, ver)) = dataset.rsplit_once("_v") {
        if !ver.is_empty() && ver.chars().all(|c| c.is_ascii_digit()) {
            return base.to_string();
        }
    }
    dataset.to_string()
}

/// Resolve `tasks.json` WITHOUT a hardcoded path, mirroring `poi::resolve_sidecar`: env override,
/// then next to the pack, then the pack-parent `shared/`, then the shared dir, then the cwd.
fn resolve_tasks_json(root: Option<&Path>) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("EFT_TASKS_JSON") {
        let pb = PathBuf::from(&p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Some(root) = root {
        let pb = root.join("tasks.json");
        if pb.is_file() {
            return Some(pb);
        }
        if let Some(shared) = root.parent().map(|p| p.join("shared").join("tasks.json")) {
            if shared.is_file() {
                return Some(shared);
            }
        }
    }
    let shared = crate::paths::shared_dir().join("tasks.json");
    if shared.is_file() {
        return Some(shared);
    }
    let cwd = PathBuf::from("tasks.json");
    if cwd.is_file() {
        return Some(cwd);
    }
    None
}

/// One-shot loader (Update, guarded by `TaskCatalog::loaded` — same pattern as `ui::load_bookmarks`):
/// once the pack is up, read + parse `tasks.json` into the catalog. A missing/corrupt file just
/// leaves the catalog empty (the tab shows a hint). Runs on Update rather than Startup so it never
/// races the pack's insertion order.
fn load_task_catalog(
    mut cat: ResMut<TaskCatalog>,
    pack: Option<Res<crate::render::LoadedPack>>,
) {
    if cat.loaded {
        return;
    }
    let Some(pack) = pack else {
        return; // no pack yet (start-menu mode) — try again next frame
    };
    cat.loaded = true; // latch: one real attempt once the pack exists
    let root = pack.0.root.as_path();
    let Some(path) = resolve_tasks_json(Some(root)) else {
        warn!("tasks_panel: no tasks.json found (set EFT_TASKS_JSON or drop it in packs/shared) — Tasks tab empty");
        return;
    };
    match std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<RawFile>(&s).ok())
    {
        Some(rf) => {
            cat.source = rf.source.unwrap_or_else(|| "tarkov.dev".into());
            cat.tasks = rf.tasks.into_iter().map(convert_task).collect();
            info!("tasks_panel: {} tasks loaded from {}", cat.tasks.len(), path.display());
        }
        None => warn!("tasks_panel: failed to parse {}", path.display()),
    }
}

/// The module's plugin: owns the catalog resource + the one-shot loader (and, under egui, the icon
/// cache). Deliberately adds NOTHING to `UiPlugin` — the engineer adds this plugin once and calls
/// [`tasks_panel_ui`] from the right-panel router.
pub struct TasksPanelPlugin;
impl Plugin for TasksPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TaskCatalog>()
            .add_systems(Update, load_task_catalog);
        #[cfg(feature = "egui")]
        app.init_resource::<TaskIconCache>();
    }
    /// Tolerate double-registration: `main.rs` wires this (data owner) and the parallel router work
    /// might too. `init_resource` is idempotent and `load_task_catalog` is `loaded`-guarded, so a
    /// second add is harmless — and avoids Bevy's default duplicate-plugin panic.
    fn is_unique(&self) -> bool {
        false
    }
}

// ============================================================================================
// ICON CACHE — a mirror of inspect.rs's private `IconCache` (share-or-mirror: mirror keeps this
// module standalone; the few textures that overlap loose-loot/lock icons are negligible).
// ============================================================================================

/// Lazy per-slug icon texture cache for the task cards. `None` entries memoize a miss so the disk is
/// probed at most once per slug. Item-keyed (not map-keyed): loads from `<pack>/icons` then the
/// cross-map `packs/shared/icons`. Same `<slug>.png` contract as `inspect::icon_slug` / fetch_icons.
#[cfg(feature = "egui")]
#[derive(Resource, Default)]
pub struct TaskIconCache {
    tex: std::collections::HashMap<String, Option<bevy_egui::egui::TextureHandle>>,
}

#[cfg(feature = "egui")]
impl TaskIconCache {
    /// Fetch (and cache) the texture for `slug`, or `None` if no `<slug>.png` exists in either dir.
    fn get(
        &mut self,
        ctx: &bevy_egui::egui::Context,
        root: Option<&Path>,
        shared: Option<&Path>,
        slug: &str,
    ) -> Option<bevy_egui::egui::TextureHandle> {
        use bevy_egui::egui;
        if let Some(hit) = self.tex.get(slug) {
            return hit.clone();
        }
        let file = format!("{slug}.png");
        let try_open = |dir: Option<&Path>| dir.and_then(|d| image::open(d.join(&file)).ok());
        let loaded = try_open(root).or_else(|| try_open(shared)).map(|img| {
            let rgba = img.into_rgba8();
            let (w, h) = rgba.dimensions();
            let ci = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw());
            ctx.load_texture(format!("taskicon:{slug}"), ci, egui::TextureOptions::LINEAR)
        });
        self.tex.insert(slug.to_string(), loaded.clone());
        loaded
    }
}

// ============================================================================================
// PANEL UI STATE (egui) — ephemeral, panel-owned. Lives in a `Local` so nothing reacts to it and
// no resource registration is needed; it resets on each launch (a map switch restarts the viewer),
// which is exactly the desired "default to the current map" behaviour.
// ============================================================================================

#[cfg(feature = "egui")]
pub struct TasksUiState {
    search: String,
    /// Requirement 2 default: show only the current map's tasks. Toggle off for all maps.
    this_map_only: bool,
    /// Client-side "checked off" objective ids — a session-only progress feel (no game state).
    done: std::collections::HashSet<String>,
}
#[cfg(feature = "egui")]
impl Default for TasksUiState {
    fn default() -> Self {
        Self {
            search: String::new(),
            this_map_only: true,
            done: std::collections::HashSet::new(),
        }
    }
}

// ============================================================================================
// SYSTEM-PARAM BUNDLE — one bundle so the router stays under Bevy's 16-param limit.
// ============================================================================================

#[cfg(feature = "egui")]
#[derive(bevy::ecs::system::SystemParam)]
pub struct TasksPanelParams<'w, 's> {
    /// The full quest catalog (this module's `load_task_catalog` fills it).
    pub catalog: Res<'w, TaskCatalog>,
    /// The tracked-task set + filter row — the SAME resource poi.rs reads for marker visibility.
    pub tracker: ResMut<'w, crate::ui::QuestTracker>,
    /// Only `quests` (the master Quest-layer toggle) is touched — checking a task flips it on so
    /// the markers actually appear.
    pub toggles: ResMut<'w, crate::ui::LayerToggles>,
    /// Panel-local UI state (search / map filter / checked objectives).
    pub ui_state: bevy::ecs::system::Local<'s, TasksUiState>,
    /// Task-item icon cache (mirror of inspect's).
    pub icons: ResMut<'w, TaskIconCache>,
    /// The loaded pack — current map key (`manifest.dataset`) + the icon dirs.
    pub pack: Option<Res<'w, crate::render::LoadedPack>>,
    /// "fly to" an objective location.
    pub cam_cmd: ResMut<'w, crate::CameraCommand>,
    /// "route here" / "route tracked" (the :8091 pathfind server).
    pub route: MessageWriter<'w, crate::pathfind::RouteRequest>,
    /// Last route status/length for the readout.
    pub route_result: Res<'w, crate::pathfind::RouteResult>,
    /// Pathfind server state — the route buttons gate on it running.
    pub server: Res<'w, crate::pathfind::PathfindServer>,
}

// ============================================================================================
// PUBLIC ENTRY — the router calls this once per frame with the bundle above.
// ============================================================================================

/// Render the Tasks tab into `ui`. The caller owns the `SidePanel`/frame; this fills a
/// right-panel-width column. See the module doc for the full contract.
#[cfg(feature = "egui")]
pub fn tasks_panel_ui(ui: &mut bevy_egui::egui::Ui, p: &mut TasksPanelParams) {
    use bevy_egui::egui::{self, Color32, CollapsingHeader, RichText};
    use crate::pathfind::{RouteRequest, RouteStatus, ServerStatus};

    // Destructure the bundle into independent field references so egui closures capture disjoint
    // fields cleanly (no whole-`p` borrow conflicts).
    let TasksPanelParams {
        catalog,
        tracker,
        toggles,
        ui_state,
        icons,
        pack,
        cam_cmd,
        route,
        route_result,
        server,
    } = p;

    // ---- Tarkov gear-screen palette (charcoal cards, bone/beige text, amber accent, square) ----
    const ACCENT: Color32 = Color32::from_rgb(232, 194, 122); // warm amber
    const BONE: Color32 = Color32::from_rgb(208, 200, 178);
    const HDR: Color32 = Color32::from_rgb(160, 164, 160);
    const MUTED: Color32 = Color32::from_rgb(122, 124, 118);
    const CARD_BG: Color32 = Color32::from_rgb(26, 28, 29);
    const CARD_BORDER: Color32 = Color32::from_rgb(58, 60, 55);
    const KAPPA: Color32 = Color32::from_rgb(212, 175, 95);
    const LK: Color32 = Color32::from_rgb(120, 200, 210);
    const FIR: Color32 = Color32::from_rgb(126, 190, 120);
    const TRACKED: Color32 = Color32::from_rgb(150, 138, 232); // quest purple accent when tracked

    // Local copies for the clone-edit-compare write-back (see module doc).
    let mut tr = (**tracker).clone();
    let mut quests_on = toggles.quests;

    // Current map key + icon dirs, resolved once from the pack.
    let (cur_map, icon_root, icon_shared): (Option<String>, Option<PathBuf>, Option<PathBuf>) =
        match pack {
            Some(lp) => {
                let root = lp.0.root.clone();
                let shared = root.parent().map(|pp| pp.join("shared").join("icons"));
                (Some(map_key(&lp.0.manifest.dataset)), Some(root.join("icons")), shared)
            }
            None => (None, None, None),
        };

    ui.spacing_mut().item_spacing = egui::vec2(8.0, 6.0);

    // ---- HEADER ----
    ui.horizontal(|ui| {
        ui.label(RichText::new("TASKS").color(ACCENT).size(17.0).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if !tr.active.is_empty()
                && ui
                    .small_button("untrack all")
                    .on_hover_text("clear every tracked task")
                    .clicked()
            {
                tr.active.clear();
            }
        });
    });

    // ---- MAP FILTER + master Quest-layer toggle ----
    ui.horizontal(|ui| {
        let map_name = cur_map
            .as_deref()
            .map(|m| titlecase_key(m))
            .unwrap_or_else(|| "(no map)".to_string());
        ui.checkbox(&mut ui_state.this_map_only, RichText::new("this map only").size(12.0))
            .on_hover_text("show only tasks that touch the loaded map");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(map_name).color(ACCENT).size(12.0));
        });
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("\u{25CF}").color(TRACKED).size(11.0));
        ui.checkbox(&mut quests_on, RichText::new("show markers on map").size(12.0))
            .on_hover_text("the Quest overlay layer (tracked tasks focus it to themselves)");
    });

    // ---- SEARCH ----
    ui.add(
        egui::TextEdit::singleline(&mut ui_state.search)
            .desired_width(f32::INFINITY)
            .hint_text("Search tasks / traders / items\u{2026}"),
    );

    // ---- FILTER ROW (reuses the tracker's own filter fields, so it composes with poi.rs) ----
    ui.horizontal(|ui| {
        ui.checkbox(&mut tr.kappa_only, RichText::new("Kappa").size(12.0));
        ui.checkbox(&mut tr.lk_only, RichText::new("Lightkeeper").size(12.0));
    });
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut tr.max_level).range(0..=79));
        ui.label(RichText::new("\u{2264} Lvl  (0 = any)").size(12.0).color(MUTED));
    });

    // ---- FILTER + GROUP the catalog by trader ----
    let q = ui_state.search.trim().to_lowercase();
    let (kappa_only, lk_only, max_level, this_map_only) =
        (tr.kappa_only, tr.lk_only, tr.max_level, ui_state.this_map_only);
    let cur = cur_map.as_deref();
    let on_map = |t: &TaskDef| match cur {
        Some(m) => t.map == m || t.maps.iter().any(|x| x == m),
        None => true,
    };
    let matches_q = |t: &TaskDef| {
        if q.is_empty() {
            return true;
        }
        if t.name.to_lowercase().contains(&q) || t.trader.to_lowercase().contains(&q) {
            return true;
        }
        t.objectives.iter().any(|o| {
            o.items.iter().any(|i| i.to_lowercase().contains(&q))
                || o.quest_item.as_deref().is_some_and(|i| i.to_lowercase().contains(&q))
                || o.marker_item.as_deref().is_some_and(|i| i.to_lowercase().contains(&q))
        })
    };
    let mut groups: std::collections::BTreeMap<String, Vec<&TaskDef>> = std::collections::BTreeMap::new();
    let mut shown = 0usize;
    for t in &catalog.tasks {
        if this_map_only && !on_map(t) {
            continue;
        }
        if kappa_only && !t.kappa {
            continue;
        }
        if lk_only && !t.lk {
            continue;
        }
        if max_level > 0 && t.min_level > 0 && t.min_level > max_level {
            continue;
        }
        if !matches_q(t) {
            continue;
        }
        let trader = if t.trader.is_empty() { "Other".to_string() } else { t.trader.clone() };
        groups.entry(trader).or_default().push(t);
        shown += 1;
    }
    for v in groups.values_mut() {
        v.sort_by(|a, b| a.min_level.cmp(&b.min_level).then_with(|| a.name.cmp(&b.name)));
    }

    // ---- SUMMARY + route-tracked controls ----
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{shown} tasks  \u{00B7}  {} tracked", tr.active.len()))
                .size(10.0)
                .color(MUTED),
        );
    });
    let pf_running = server.status == ServerStatus::Running;
    ui.horizontal(|ui| {
        if ui
            .add_enabled(pf_running && !tr.active.is_empty(), egui::Button::new("Route tracked"))
            .on_hover_text("shortest tour through every tracked task's on-map objectives")
            .clicked()
        {
            let mut dests: Vec<Vec3> = Vec::new();
            if let Some(m) = cur {
                for t in catalog.tasks.iter().filter(|t| tr.active.contains(&t.id)) {
                    for o in &t.objectives {
                        if let Some(pos) = obj_location(o, m) {
                            dests.push(pos);
                        }
                    }
                }
            }
            if !dests.is_empty() {
                route.write(RouteRequest { start: None, dests, optimize_order: true });
            }
        }
        if ui.button("Clear route").clicked() {
            route.write(RouteRequest { start: None, dests: Vec::new(), optimize_order: false });
        }
    });
    match &route_result.status {
        RouteStatus::Pending => {
            ui.label(RichText::new("routing\u{2026}").size(11.0).color(ACCENT));
        }
        RouteStatus::Ok => {
            ui.label(
                RichText::new(format!(
                    "Route  {:.0} m  ({} stops)",
                    route_result.dist,
                    route_result.points.len()
                ))
                .size(11.0)
                .color(Color32::from_gray(210)),
            );
        }
        RouteStatus::Error(e) => {
            ui.label(RichText::new(e.as_str()).size(10.0).color(Color32::from_rgb(210, 96, 84)));
        }
        RouteStatus::Idle => {}
    }
    if !pf_running {
        ui.label(
            RichText::new("start the pathfind server (Layers tab) to enable routing")
                .size(9.0)
                .italics()
                .color(MUTED),
        );
    }

    ui.add_space(4.0);
    ui.separator();

    // ---- TASK LIST (trader groups, collapsible; each task a charcoal card) ----
    egui::ScrollArea::vertical()
        .id_salt("tasks_body")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if catalog.tasks.is_empty() {
                ui.label(
                    RichText::new("no tasks.json loaded")
                        .size(11.0)
                        .italics()
                        .color(MUTED),
                );
                return;
            }
            if shown == 0 {
                ui.label(
                    RichText::new("no tasks match the current filters")
                        .size(11.0)
                        .italics()
                        .color(MUTED),
                );
                return;
            }
            for (trader, tasks) in &groups {
                CollapsingHeader::new(
                    RichText::new(format!("{trader}   \u{00B7} {}", tasks.len()))
                        .color(HDR)
                        .size(12.0)
                        .strong(),
                )
                .id_salt(format!("trader_{trader}"))
                .default_open(true)
                .show(ui, |ui| {
                    for t in tasks {
                        let tracked = tr.active.contains(&t.id);
                        let frame = egui::Frame::new()
                            .fill(CARD_BG)
                            .inner_margin(egui::Margin::same(8))
                            .corner_radius(0.0)
                            .stroke(egui::Stroke::new(
                                1.0,
                                if tracked { TRACKED } else { CARD_BORDER },
                            ));
                        frame.show(ui, |ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(6.0, 3.0);

                            // -- title row: [track] name .............. [locate] --
                            ui.horizontal(|ui| {
                                let mut on = tracked;
                                if ui
                                    .checkbox(&mut on, "")
                                    .on_hover_text("track this task (focuses the map markers to it)")
                                    .changed()
                                {
                                    if on {
                                        tr.active.insert(t.id.clone());
                                        quests_on = true; // tracking is pointless with the layer off
                                    } else {
                                        tr.active.remove(&t.id);
                                    }
                                }
                                ui.label(
                                    RichText::new(&t.name)
                                        .color(if tracked { ACCENT } else { BONE })
                                        .size(13.0)
                                        .strong(),
                                );
                                // "locate" flies to the first on-map objective of the task.
                                let here = cur.and_then(|m| task_first_location(t, m));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if let Some(pos) = here {
                                            if ui
                                                .small_button(RichText::new("locate").size(11.0))
                                                .on_hover_text("fly to this task on the map")
                                                .clicked()
                                            {
                                                cam_cmd.fly_to = Some(pos);
                                            }
                                        }
                                    },
                                );
                            });

                            // -- meta line: Lvl / Kappa / LK / off-map note --
                            ui.horizontal_wrapped(|ui| {
                                if t.min_level > 0 {
                                    ui.label(
                                        RichText::new(format!("Lvl {}", t.min_level))
                                            .size(10.0)
                                            .color(MUTED),
                                    );
                                }
                                if t.kappa {
                                    ui.label(RichText::new("Kappa").size(10.0).color(KAPPA));
                                }
                                if t.lk {
                                    ui.label(RichText::new("Lightkeeper").size(10.0).color(LK));
                                }
                                // In "all maps" mode, note where a task's objectives actually are.
                                if !this_map_only && !t.maps.is_empty() {
                                    ui.label(
                                        RichText::new(t.maps.iter().map(|m| titlecase_key(m)).collect::<Vec<_>>().join(", "))
                                            .size(10.0)
                                            .color(MUTED),
                                    );
                                }
                            });
                            if !t.requires.is_empty() {
                                ui.label(
                                    RichText::new(format!("after: {}", t.requires.join(", ")))
                                        .size(9.0)
                                        .italics()
                                        .color(MUTED),
                                );
                            }

                            // -- REQUIRED ITEMS (icons; degrade to text) --
                            let mut missing_icons = 0usize;
                            for o in &t.objectives {
                                // Collect this objective's item names (handover/find + quest + mark).
                                let mut names: Vec<&str> = o.items.iter().map(|s| s.as_str()).collect();
                                if let Some(qi) = &o.quest_item {
                                    names.push(qi.as_str());
                                }
                                if let Some(mi) = &o.marker_item {
                                    names.push(mi.as_str());
                                }
                                if names.is_empty() {
                                    continue;
                                }
                                // Badge: "need N" and/or FIR.
                                ui.horizontal(|ui| {
                                    let verb = item_verb(&o.kind);
                                    let mut b = String::from(verb);
                                    if o.count > 1 {
                                        b.push_str(&format!("  \u{00D7}{}", o.count));
                                    }
                                    ui.label(RichText::new(b).size(10.0).color(BONE));
                                    if o.fir {
                                        ui.label(
                                            RichText::new("FIR")
                                                .size(9.0)
                                                .strong()
                                                .color(FIR),
                                        )
                                        .on_hover_text("must be Found In Raid");
                                    }
                                });
                                // Icon/text chips (cap the "any X" enumerations).
                                const CAP: usize = 10;
                                ui.horizontal_wrapped(|ui| {
                                    for name in names.iter().take(CAP) {
                                        let slug = crate::inspect::icon_slug(name);
                                        let tex = icons.get(
                                            ui.ctx(),
                                            icon_root.as_deref(),
                                            icon_shared.as_deref(),
                                            &slug,
                                        );
                                        if let Some(tex) = tex {
                                            let sz = tex.size_vec2();
                                            let s = 20.0 / sz.y.max(1.0);
                                            ui.image((tex.id(), sz * s)).on_hover_text(*name);
                                        } else {
                                            missing_icons += 1;
                                            ui.label(
                                                RichText::new(short_item(name))
                                                    .size(10.0)
                                                    .color(MUTED),
                                            )
                                            .on_hover_text(*name);
                                        }
                                    }
                                    if names.len() > CAP {
                                        ui.label(
                                            RichText::new(format!("\u{2026} +{}", names.len() - CAP))
                                                .size(10.0)
                                                .color(MUTED),
                                        );
                                    }
                                });
                            }
                            let _ = missing_icons; // (kept for a possible "N need CDN icons" hint)

                            // -- OBJECTIVES CHECKLIST (progress feel) --
                            let total = t.objectives.len();
                            let done_n = t
                                .objectives
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| ui_state.done.contains(&obj_key(&t.id, *i)))
                                .count();
                            ui.add_space(2.0);
                            ui.label(
                                RichText::new(format!("Objectives  {done_n}/{total}"))
                                    .size(10.0)
                                    .color(if done_n == total && total > 0 { FIR } else { HDR }),
                            );
                            for (i, o) in t.objectives.iter().enumerate() {
                                let key = obj_key(&t.id, i);
                                let is_done = ui_state.done.contains(&key);
                                ui.horizontal(|ui| {
                                    // client-side "check off" toggle
                                    let mut d = is_done;
                                    if ui.checkbox(&mut d, "").changed() {
                                        if d {
                                            ui_state.done.insert(key.clone());
                                        } else {
                                            ui_state.done.remove(&key);
                                        }
                                    }
                                    let (tag, dot) = obj_tag(&o.kind);
                                    ui.label(RichText::new("\u{25CF}").color(dot).size(10.0));
                                    ui.label(RichText::new(tag).size(9.0).strong().color(dot));
                                    // go/route to this objective when it has an on-map location
                                    let here = cur.and_then(|m| obj_location(o, m));
                                    if let Some(pos) = here {
                                        if ui.small_button(RichText::new("go").size(10.0)).clicked() {
                                            cam_cmd.fly_to = Some(pos);
                                        }
                                        if ui
                                            .add_enabled(pf_running, egui::Button::new(RichText::new("route").size(10.0)))
                                            .clicked()
                                        {
                                            route.write(RouteRequest {
                                                start: None,
                                                dests: vec![pos],
                                                optimize_order: false,
                                            });
                                        }
                                    }
                                });
                                // objective description (strikethrough when checked off)
                                let mut desc = RichText::new(&o.desc).size(11.0);
                                desc = if is_done {
                                    desc.strikethrough().color(MUTED)
                                } else {
                                    desc.color(BONE)
                                };
                                ui.horizontal_wrapped(|ui| {
                                    ui.add_space(18.0);
                                    ui.label(desc);
                                    if o.optional {
                                        ui.label(RichText::new("(optional)").size(9.0).italics().color(MUTED));
                                    }
                                });
                                // kill targets / extract name (no icons)
                                if !o.targets.is_empty() {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.add_space(18.0);
                                        ui.label(
                                            RichText::new(format!("targets: {}", o.targets.join(", ")))
                                                .size(9.0)
                                                .color(MUTED),
                                        );
                                    });
                                }
                                if let Some(exit) = &o.exit {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.add_space(18.0);
                                        ui.label(
                                            RichText::new(format!("exit: {exit}")).size(9.0).color(MUTED),
                                        );
                                    });
                                }
                            }
                        });
                        ui.add_space(4.0);
                    }
                });
            }
        });

    // ---- footer provenance ----
    ui.add_space(4.0);
    ui.label(
        RichText::new(format!(
            "tasks + item names: {}  \u{00B7}  icons: tarkov.dev CDN cache",
            if catalog.source.is_empty() { "tarkov.dev" } else { catalog.source.as_str() }
        ))
        .size(9.0)
        .italics()
        .color(MUTED),
    );

    // ---- WRITE-BACK (only on a real delta, so poi's is_changed() gates stay meaningful) ----
    if tr != **tracker {
        **tracker = tr;
    }
    if quests_on != toggles.quests {
        toggles.quests = quests_on;
    }
}

// ---- small helpers (egui-gated) ---------------------------------------------------------------

/// The first objective location for a task that falls on `map_key` (zone point, else find-spot).
#[cfg(feature = "egui")]
fn task_first_location(t: &TaskDef, map_key: &str) -> Option<Vec3> {
    t.objectives.iter().find_map(|o| obj_location(o, map_key))
}

/// An objective's location on `map_key`: a zone point first, else the first quest-item find spot.
#[cfg(feature = "egui")]
fn obj_location(o: &ObjectiveDef, map_key: &str) -> Option<Vec3> {
    for z in &o.zones {
        if z.map == map_key {
            if let Some(p) = z.pos {
                return Some(p);
            }
        }
    }
    for l in &o.item_locations {
        if l.map == map_key {
            if let Some(p) = l.pts.first() {
                return Some(*p);
            }
        }
    }
    None
}

/// A stable id for an objective's checked-off state (task id + index; objective ids aren't always
/// present/unique across the schema, so the index keys it locally).
#[cfg(feature = "egui")]
fn obj_key(task_id: &str, i: usize) -> String {
    format!("{task_id}#{i}")
}

/// Short ACTION verb for an item-bearing objective's badge.
#[cfg(feature = "egui")]
fn item_verb(kind: &str) -> &'static str {
    match kind {
        "giveItem" | "giveQuestItem" => "Hand in",
        "findItem" | "findQuestItem" => "Find",
        "plantItem" | "plantQuestItem" => "Plant",
        "buildWeapon" => "Build",
        "sellItem" => "Sell",
        "useItem" => "Use",
        "mark" => "Mark with",
        _ => "Need",
    }
}

/// (short ASCII tag, dot colour) per objective TYPE — the map key from `obj_tag` reads at a glance.
#[cfg(feature = "egui")]
fn obj_tag(kind: &str) -> (&'static str, bevy_egui::egui::Color32) {
    use bevy_egui::egui::Color32;
    match kind {
        "visit" => ("GO TO", Color32::from_rgb(126, 190, 120)),
        "mark" => ("MARK", Color32::from_rgb(232, 194, 122)),
        "shoot" => ("KILL", Color32::from_rgb(214, 92, 72)),
        "extract" => ("EXFIL", Color32::from_rgb(92, 200, 160)),
        "giveItem" | "giveQuestItem" => ("HAND IN", Color32::from_rgb(96, 156, 226)),
        "findItem" => ("FIND", Color32::from_rgb(196, 178, 120)),
        "findQuestItem" => ("PICK UP", Color32::from_rgb(176, 132, 226)),
        "plantItem" | "plantQuestItem" => ("PLANT", Color32::from_rgb(226, 150, 92)),
        "buildWeapon" => ("BUILD", Color32::from_rgb(180, 180, 170)),
        "skill" => ("SKILL", Color32::from_rgb(150, 150, 145)),
        "traderLevel" | "traderStanding" => ("TRADER", Color32::from_rgb(150, 150, 145)),
        "taskStatus" => ("QUEST", Color32::from_rgb(150, 150, 145)),
        "sellItem" => ("SELL", Color32::from_rgb(96, 156, 226)),
        "useItem" => ("USE", Color32::from_rgb(180, 180, 170)),
        "experience" => ("XP", Color32::from_rgb(150, 150, 145)),
        _ => ("DO", Color32::from_rgb(150, 150, 145)),
    }
}

/// Trim a long item name to ~24 chars for a text chip (icons carry the full name on hover).
#[cfg(feature = "egui")]
fn short_item(name: &str) -> String {
    if name.chars().count() <= 24 {
        name.to_string()
    } else {
        let mut s: String = name.chars().take(23).collect();
        s.push('\u{2026}');
        s
    }
}

/// Map key -> display ("ground_zero" -> "Ground Zero"). ASCII only.
#[cfg(feature = "egui")]
fn titlecase_key(key: &str) -> String {
    key.split(['_', '-'])
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
