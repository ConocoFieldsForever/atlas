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
    pub xp: i64,
    pub image: Option<String>,
    pub faction: Option<String>,
    pub restartable: bool,
    pub delay_min: i64,
    pub delay_max: i64,
    pub trader_reqs: Vec<TraderRequirement>,
    pub rewards: TaskRewards,
    /// Prerequisite task NAMES (shown so the planner can see ordering).
    pub requires: Vec<String>,
    pub objectives: Vec<ObjectiveDef>,
}

/// One objective. Fields are all optional in the schema and default to empty/false here.
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct ObjectiveDef {
    pub id: String,
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
    pub required_keys: Vec<Vec<ItemRef>>,
    pub weapons: Vec<String>,
    pub weapon_mods: Vec<String>,
    pub wearing: Vec<String>,
    pub not_wearing: Vec<String>,
    pub use_any: Vec<String>,
    pub distance: Option<CompareValue>,
    pub body_parts: Vec<String>,
    pub shot_type: Option<String>,
    pub time_window: Option<[f32; 2]>,
    pub min_durability: Option<f32>,
    pub max_durability: Option<f32>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct ItemRef {
    #[serde(rename = "n")]
    pub name: String,
    #[serde(rename = "s")]
    pub short: Option<String>,
    #[serde(rename = "pr")]
    pub price: Option<i64>,
    pub count: Option<i64>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct CompareValue {
    #[serde(rename = "compareMethod")]
    pub compare: Option<String>,
    pub value: f32,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct TraderRequirement {
    pub trader: String,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub compare: Option<String>,
    pub value: f32,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct StandingReward {
    pub trader: String,
    pub value: Option<f32>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct OfferReward {
    pub trader: String,
    pub level: Option<i64>,
    pub item: Option<String>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct SkillReward {
    pub name: Option<String>,
    pub level: Option<f32>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct TaskRewards {
    pub items: Vec<ItemRef>,
    pub standing: Vec<StandingReward>,
    pub offers: Vec<OfferReward>,
    pub skills: Vec<SkillReward>,
    pub traders: Vec<String>,
    pub achievements: Vec<String>,
    pub customization: Vec<String>,
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
    xp: i64,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    faction: Option<String>,
    #[serde(default)]
    restartable: bool,
    #[serde(default, rename = "delayMin")]
    delay_min: i64,
    #[serde(default, rename = "delayMax")]
    delay_max: i64,
    #[serde(default, rename = "traderReqs")]
    trader_reqs: Vec<TraderRequirement>,
    #[serde(default)]
    rewards: TaskRewards,
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    objectives: Vec<RawObj>,
}
#[derive(Deserialize)]
struct RawObj {
    #[serde(default)]
    id: String,
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
    #[serde(default, rename = "requiredKeys")]
    required_keys: Vec<Vec<ItemRef>>,
    #[serde(default)]
    weapons: Vec<String>,
    #[serde(default, rename = "weaponMods")]
    weapon_mods: Vec<String>,
    #[serde(default)]
    wearing: Vec<String>,
    #[serde(default, rename = "notWearing")]
    not_wearing: Vec<String>,
    #[serde(default, rename = "useAny")]
    use_any: Vec<String>,
    #[serde(default)]
    distance: Option<CompareValue>,
    #[serde(default, rename = "bodyParts")]
    body_parts: Vec<String>,
    #[serde(default, rename = "shotType")]
    shot_type: Option<String>,
    #[serde(default, rename = "timeWindow")]
    time_window: Option<[f32; 2]>,
    #[serde(default, rename = "minDurability")]
    min_durability: Option<f32>,
    #[serde(default, rename = "maxDurability")]
    max_durability: Option<f32>,
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
            id: o.id,
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
            required_keys: o.required_keys,
            weapons: o.weapons,
            weapon_mods: o.weapon_mods,
            wearing: o.wearing,
            not_wearing: o.not_wearing,
            use_any: o.use_any,
            distance: o.distance,
            body_parts: o.body_parts,
            shot_type: o.shot_type,
            time_window: o.time_window,
            min_durability: o.min_durability,
            max_durability: o.max_durability,
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
        xp: t.xp,
        image: t.image.filter(|s| !s.is_empty()),
        faction: t.faction.filter(|s| !s.is_empty()),
        restartable: t.restartable,
        delay_min: t.delay_min,
        delay_max: t.delay_max,
        trader_reqs: t.trader_reqs,
        rewards: t.rewards,
        requires: t.requires,
        objectives,
    }
}

/// Canonical map key for tasks.json joins: the manifest's `map` (canonical id) if present, else the
/// dataset dir basename with a `_vN` suffix stripped. Same rule as `poi::map_key`.
fn map_key(manifest: &crate::eftpack::Manifest) -> String {
    if !manifest.map.is_empty() {
        return manifest.map.clone();
    }
    let dataset = &manifest.dataset;
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
    epoch: Res<crate::render::MapEpoch>,
    mut loaded_epoch: Local<Option<u64>>,
) {
    // Epoch-tracked (not a one-shot latch): an in-place swap re-resolves tasks.json against the new
    // pack root (usually the same shared file → identical catalog; a pack-local one is picked up).
    if *loaded_epoch == Some(epoch.0) {
        return;
    }
    let Some(pack) = pack else {
        return; // no pack yet (start-menu mode) — try again next frame
    };
    *loaded_epoch = Some(epoch.0);
    cat.loaded = true;
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
    task_art: std::collections::HashMap<String, Option<bevy_egui::egui::TextureHandle>>,
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

    fn get_task(
        &mut self,
        ctx: &bevy_egui::egui::Context,
        task_dir: Option<&Path>,
        task_id: &str,
    ) -> Option<bevy_egui::egui::TextureHandle> {
        use bevy_egui::egui;
        if let Some(hit) = self.task_art.get(task_id) {
            return hit.clone();
        }
        let loaded = task_dir
            .and_then(|dir| image::open(dir.join(format!("{task_id}.png"))).ok())
            .map(|img| {
                let rgba = img.into_rgba8();
                let (w, h) = rgba.dimensions();
                let image = egui::ColorImage::from_rgba_unmultiplied(
                    [w as usize, h as usize],
                    rgba.as_raw(),
                );
                ctx.load_texture(
                    format!("taskart:{task_id}"),
                    image,
                    egui::TextureOptions::LINEAR,
                )
            });
        self.task_art.insert(task_id.to_string(), loaded.clone());
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
}
#[cfg(feature = "egui")]
impl Default for TasksUiState {
    fn default() -> Self {
        Self {
            // EFT_TASK_SEARCH seeds the search box (screenshots / power users), mirroring EFT_TAB.
            search: std::env::var("EFT_TASK_SEARCH").unwrap_or_default(),
            this_map_only: true,
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
    /// "route here" / "route tracked" (in-process CPU routing, nav.rs).
    pub route: MessageWriter<'w, crate::pathfind::RouteRequest>,
    /// Last route status/length for the readout.
    pub route_result: Res<'w, crate::pathfind::RouteResult>,
    /// Pathfind server state — the route buttons gate on it running.
    pub server: Res<'w, crate::pathfind::PathfindServer>,
    pub progress: ResMut<'w, crate::progress::PlayerProgress>,
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
        progress,
    } = p;

    // ---- Palette: all from the single source of truth (ui_theme); thin local aliases keep the
    // body readable. This file's BONE/MUTED/CARD used to DRIFT from menu.rs/ui.rs — now unified. ----
    use crate::ui_theme as theme;
    const ACCENT: Color32 = theme::ACCENT;
    const BONE: Color32 = theme::BONE;
    const MUTED: Color32 = theme::MUTED;
    // Slightly stronger than the default card border: an untracked task card's body is near-panel
    // charcoal, so a faint 1px line let cards melt together in a long list (review finding). The
    // brighter steel + the dark title bar give each card a clear boundary.
    const CARD_BORDER: Color32 = theme::BORDER_STRONG;
    const FIR: Color32 = theme::OK;
    const TRACKED: Color32 = theme::TRACKED; // quest purple accent when tracked

    // Local copies for the clone-edit-compare write-back (see module doc).
    let mut tr = (**tracker).clone();
    tr.active = progress.tracked.clone();
    let mut done = progress.done.clone();
    let mut owned_keys = progress.owned_keys.clone();
    let mut quests_on = toggles.quests;

    // Current map key + icon dirs, resolved once from the pack.
    let (cur_map, icon_root, icon_shared, task_art_dir):
        (Option<String>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) =
        match pack {
            Some(lp) => {
                let root = lp.0.root.clone();
                let shared_root = root.parent().map(|pp| pp.join("shared"));
                let shared = shared_root.as_ref().map(|p| p.join("icons"));
                let task_art = shared_root.as_ref().map(|p| p.join("task_images"));
                (Some(map_key(&lp.0.manifest)), Some(root.join("icons")), shared, task_art)
            }
            None => (None, None, None, Some(crate::paths::shared_dir().join("task_images"))),
        };

    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);

    // ---- HEADER ----
    ui.horizontal(|ui| {
        ui.label(theme::title("TASKS"));
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

    // ---- LEVEL FILTER (the Kappa / Lightkeeper toggles were dropped: niche Tarkov jargon the user
    // didn't want. The kappa_only / lk_only flags stay in the model, defaulting off, so they can be
    // reinstated later without touching the filter loop below.) ----
    ui.horizontal(|ui| {
        ui.label(RichText::new("Max level").size(12.0).color(MUTED));
        ui.add(egui::DragValue::new(&mut tr.max_level).range(0..=79).speed(1.0))
            .on_hover_text("hide tasks whose required level is above this (0 = show every level)");
        if tr.max_level == 0 {
            ui.label(RichText::new("(any)").size(11.0).color(theme::FAINT));
        }
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
                || o.required_keys.iter().flatten().any(|k| k.name.to_lowercase().contains(&q))
                || o.weapons.iter().chain(&o.weapon_mods).chain(&o.wearing).any(|i| i.to_lowercase().contains(&q))
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
                    for (i, o) in t.objectives.iter().enumerate() {
                        if done.contains(&obj_key(&t.id, o, i)) {
                            continue;
                        }
                        if let Some(pos) = obj_location(o, m) {
                            dests.push(pos);
                        }
                    }
                }
            }
            if !dests.is_empty() {
                route.write(RouteRequest { start: None, dests, optimize_order: true, ..Default::default() });
            }
        }
        if ui.button("Clear route").clicked() {
            route.write(RouteRequest { start: None, dests: Vec::new(), optimize_order: false, ..Default::default() });
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
                    route_result.stop_count
                ))
                .size(11.0)
                .color(Color32::from_gray(210)),
            );
        }
        RouteStatus::Error(e) => {
            ui.label(RichText::new(e.as_str()).size(theme::SIZE_CAPTION).color(theme::DANGER_TEXT));
        }
        RouteStatus::Idle => {}
    }
    if !pf_running {
        ui.label(
            RichText::new("no route data for this map \u{2014} rebuild it from the start menu")
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
                CollapsingHeader::new(theme::section_header(trader, tasks.len()))
                .id_salt(format!("trader_{trader}"))
                .default_open(true)
                .show(ui, |ui| {
                    for t in tasks {
                        let tracked = tr.active.contains(&t.id);
                        let card_border = if tracked { TRACKED } else { CARD_BORDER };
                        // One card per task: a DARK title bar (name + Track toggle on the right), a
                        // thin meta line, then a SUBTASK TABLE — each objective is a row with a fixed
                        // left ICON gutter (big, left-aligned item art) and a click-to-toggle text
                        // column. No checkboxes anywhere: the title's Track button drives tracking and
                        // clicking an objective line marks it done/not-done.
                        let total = t.objectives.len();
                        let done_n = t
                            .objectives
                            .iter()
                            .enumerate()
                            .filter(|(i, o)| done.contains(&obj_key(&t.id, o, *i)))
                            .count();
                        let all_done = total > 0 && done_n == total;
                        // Task block: a flush RAIL title bar (no padding around it) sits directly on a
                        // padded body; ONE border is painted around both (below) so the flush title
                        // reads as part of the container. Subtask rows inside are divided by separator lines.
                        ui.spacing_mut().item_spacing.y = 0.0; // title bar + body touch (no seam gap)
                        let title_resp = egui::Frame::new().fill(theme::RAIL).inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.horizontal(|ui| {
                                if let Some(tex) = icons.get_task(ui.ctx(), task_art_dir.as_deref(), &t.id) {
                                    ui.add(egui::Image::new(&tex).fit_to_exact_size(egui::vec2(42.0, 30.0)));
                                }
                                // Track button on the RIGHT first (right_to_left); the name fills the space
                                // to its left and truncates, so a long name never pushes the button off.
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    let btn = if tracked {
                                        theme::button_filled("Tracked", TRACKED, Color32::BLACK)
                                    } else {
                                        egui::Button::new(RichText::new("Track").size(12.0).color(theme::BEIGE)).corner_radius(0.0)
                                    };
                                    if ui
                                        .add(btn)
                                        .on_hover_text(if tracked {
                                            "stop tracking (removes its markers + zones)"
                                        } else {
                                            "track this task (shows its markers + zones on the map)"
                                        })
                                        .clicked()
                                    {
                                        if tracked {
                                            tr.active.remove(&t.id);
                                        } else {
                                            tr.active.insert(t.id.clone());
                                            quests_on = true; // tracking is pointless with the layer off
                                        }
                                    }
                                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(&t.name)
                                                    .color(if tracked { TRACKED } else { theme::TEXT_BRIGHT })
                                                    .size(theme::SIZE_CARD_TITLE)
                                                    .strong(),
                                            )
                                            .truncate(),
                                        )
                                        .on_hover_text(&t.name);
                                    });
                                });
                            });
                        });
                        let body_resp = egui::Frame::new().fill(theme::CARD).inner_margin(egui::Margin { left: 10, right: 8, top: 6, bottom: 8 }).corner_radius(0.0).show(ui, |ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(6.0, 5.0);

                            // ===== META: level + progress (left) · locate (right) =====
                            ui.horizontal(|ui| {
                                let mut bits: Vec<String> = Vec::new();
                                if t.min_level > 0 {
                                    bits.push(format!("Lvl {}", t.min_level));
                                }
                                if t.xp > 0 {
                                    bits.push(format!("{} XP", crate::inspect::money(t.xp)));
                                }
                                if let Some(faction) = &t.faction {
                                    bits.push(faction.clone());
                                }
                                if total > 0 {
                                    bits.push(format!("{done_n}/{total} done"));
                                }
                                if !bits.is_empty() {
                                    ui.label(
                                        RichText::new(bits.join("   \u{00B7}   "))
                                            .size(10.0)
                                            .color(if all_done { FIR } else { MUTED }),
                                    );
                                }
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if let Some(pos) = cur.and_then(|m| task_first_location(t, m)) {
                                        if ui
                                            .small_button(RichText::new("locate").size(10.0).color(ACCENT))
                                            .on_hover_text("fly to this task on the map")
                                            .clicked()
                                        {
                                            cam_cmd.fly_to = Some(pos);
                                        }
                                    }
                                });
                            });
                            // prereqs / off-map note folded into ONE tiny line
                            let mut note: Vec<String> = Vec::new();
                            if !this_map_only && !t.maps.is_empty() {
                                note.push(t.maps.iter().map(|m| titlecase_key(m)).collect::<Vec<_>>().join(", "));
                            }
                            if !t.requires.is_empty() {
                                note.push(format!("after: {}", t.requires.join(", ")));
                            }
                            if !note.is_empty() {
                                ui.label(RichText::new(note.join("   \u{00B7}   ")).size(9.0).italics().color(MUTED));
                            }
                            let mut conditions: Vec<String> = t.trader_reqs.iter().map(|r| {
                                format!("{} {} {}", r.trader, r.kind.as_deref().unwrap_or("requirement"), trim_num(r.value))
                            }).collect();
                            if t.restartable { conditions.push("restartable".into()); }
                            if t.delay_max > 0 {
                                conditions.push(format!("available after {}-{} min", t.delay_min / 60, t.delay_max / 60));
                            }
                            if !conditions.is_empty() {
                                ui.label(RichText::new(conditions.join("   \u{00B7}   ")).size(9.0).color(MUTED));
                            }
                            let reward = reward_summary(&t.rewards);
                            if !reward.is_empty() {
                                CollapsingHeader::new(RichText::new(format!("Rewards  {reward}")).size(10.0).color(FIR))
                                    .id_salt(format!("rewards_{}", t.id))
                                    .show(ui, |ui| {
                                        for line in reward_lines(&t.rewards) {
                                            ui.label(RichText::new(line).size(9.5).color(BONE));
                                        }
                                    });
                            }

                            // ===== SUBTASK TABLE: fixed left icon/glyph gutter + click-to-toggle rows =====
                            // Every row shares a fixed-width left gutter so the art lines up in ONE
                            // column: a 32px item ICON when the item has cached art, else a painter-drawn
                            // TYPE GLYPH (crosshair / door / flag / pin / ...). A DONE row is dimmed whole
                            // and its gutter shows a green check. Clicking the gutter, the tag, OR the
                            // text toggles done — the whole line is the hit target. A thin separator line
                            // divides each subtask (and the first from the meta header above).
                            const GUTTER: f32 = 40.0;
                            const ICON: f32 = 32.0;
                            const CHIP_CAP: usize = 3;
                            ui.spacing_mut().item_spacing.y = 4.0; // snug rows around the divider lines
                            for (i, o) in t.objectives.iter().enumerate() {
                                ui.separator(); // separating line between subtasks
                                let key = obj_key(&t.id, o, i);
                                let is_done = done.contains(&key);
                                let (tag, dot) = obj_tag(&o.kind);
                                let here = cur.and_then(|m| obj_location(o, m));
                                // item names for this objective (handover/find + quest + mark tool)
                                let mut names: Vec<&str> = o.items.iter().map(|s| s.as_str()).collect();
                                if let Some(qi) = &o.quest_item { names.push(qi.as_str()); }
                                if let Some(mi) = &o.marker_item { names.push(mi.as_str()); }

                                // done state dims the whole row (text + tag + counts), gutter shows a check.
                                let body_col = if is_done { MUTED } else { BONE };
                                let tag_col = if is_done { MUTED } else { dot };
                                let mut toggle = false;

                                ui.horizontal_top(|ui| {
                                    // -- LEFT: fixed gutter (item icon / type glyph / done check) --
                                    let mut first_has_icon = false;
                                    let (grect, gresp) =
                                        ui.allocate_exact_size(egui::vec2(GUTTER, ICON), egui::Sense::click());
                                    let ibox = egui::Rect::from_min_size(grect.min, egui::vec2(ICON, ICON));
                                    if is_done {
                                        theme::paint_check(ui.painter(), ibox, FIR);
                                    } else {
                                        if let Some(name) = names.first() {
                                            let slug = crate::inspect::icon_slug(name);
                                            if let Some(tex) = icons.get(
                                                ui.ctx(), icon_root.as_deref(), icon_shared.as_deref(), &slug,
                                            ) {
                                                let sz = tex.size_vec2();
                                                let s = (ICON / sz.x.max(1.0)).min(ICON / sz.y.max(1.0));
                                                let irect = egui::Rect::from_center_size(ibox.center(), sz * s);
                                                ui.painter().image(
                                                    tex.id(),
                                                    irect,
                                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                                    Color32::WHITE,
                                                );
                                                first_has_icon = true;
                                            }
                                        }
                                        if !first_has_icon {
                                            theme::paint_obj_glyph(ui.painter(), ibox, o.kind.as_str(), dot);
                                        }
                                    }
                                    if gresp
                                        .on_hover_cursor(egui::CursorIcon::PointingHand)
                                        .on_hover_text(if first_has_icon { names[0] } else { o.kind.as_str() })
                                        .clicked()
                                    {
                                        toggle = true;
                                    }

                                    // -- RIGHT: text column (tag + clickable text + meta) --
                                    ui.vertical(|ui| {
                                        ui.spacing_mut().item_spacing = egui::vec2(5.0, 2.0);
                                        ui.horizontal_wrapped(|ui| {
                                            if ui
                                                .add(egui::Label::new(RichText::new(tag).size(9.0).strong().color(tag_col))
                                                    .sense(egui::Sense::click()))
                                                .on_hover_cursor(egui::CursorIcon::PointingHand)
                                                .on_hover_text(o.kind.as_str())
                                                .clicked()
                                            { toggle = true; }
                                            let body = if o.desc.is_empty() { "(objective)".to_string() } else { o.desc.clone() };
                                            let mut rt = RichText::new(body).size(11.5).color(body_col);
                                            if is_done { rt = rt.strikethrough(); }
                                            if ui
                                                .add(egui::Label::new(rt).sense(egui::Sense::click()))
                                                .on_hover_cursor(egui::CursorIcon::PointingHand)
                                                .on_hover_text("click to mark done / not done")
                                                .clicked()
                                            { toggle = true; }
                                            if o.count > 1 {
                                                ui.label(RichText::new(format!("\u{00D7}{}", o.count)).size(10.0).strong()
                                                    .color(if is_done { MUTED } else { BONE }));
                                            }
                                            if o.fir {
                                                ui.label(RichText::new("FIR").size(9.0).strong().color(if is_done { MUTED } else { FIR }))
                                                    .on_hover_text("Found In Raid");
                                            }
                                            if o.optional {
                                                ui.label(RichText::new("opt").size(9.0).italics().color(MUTED))
                                                    .on_hover_text("optional objective");
                                            }
                                            // extra item names (beyond the one shown as the gutter icon) —
                                            // capped so a big "any of these" enumeration (some objectives
                                            // list thousands of acceptable items) can't blow up the row OR
                                            // allocate per-name every frame: only CHIP_CAP chips render and
                                            // the "+N more" hover samples a BOUNDED slice, never the full list.
                                            let skip = if first_has_icon { 1 } else { 0 };
                                            let extra_total = names.len().saturating_sub(skip);
                                            for name in names.iter().copied().skip(skip).take(CHIP_CAP) {
                                                theme::chip(ui, &short_item(name), MUTED).on_hover_text(name);
                                            }
                                            if extra_total > CHIP_CAP {
                                                let sample: Vec<&str> =
                                                    names.iter().copied().skip(skip + CHIP_CAP).take(12).collect();
                                                let mut hover = sample.join(", ");
                                                let covered = CHIP_CAP + sample.len();
                                                if extra_total > covered {
                                                    hover.push_str(&format!(" \u{2026} (+{} more)", extra_total - covered));
                                                }
                                                ui.label(RichText::new(format!("+{} more", extra_total - CHIP_CAP)).size(9.0).color(MUTED))
                                                    .on_hover_text(hover);
                                            }
                                        });
                                        // secondary: kill targets / extract / go / route
                                        if !o.targets.is_empty() || o.exit.is_some() || here.is_some() {
                                            ui.horizontal_wrapped(|ui| {
                                                if !o.targets.is_empty() {
                                                    ui.label(RichText::new(o.targets.join(", ")).size(9.0).color(MUTED))
                                                        .on_hover_text("kill targets");
                                                }
                                                if let Some(exit) = &o.exit {
                                                    ui.label(RichText::new(format!("@{exit}")).size(9.0).color(MUTED))
                                                        .on_hover_text("extract");
                                                }
                                                if let Some(pos) = here {
                                                    // go/route AUTO-TRACK the task: tracking is what
                                                    // shows its markers + TRIGGER-REGION zones on the
                                                    // map (poi.rs draws tracked tasks' objective
                                                    // outlines/walls) — flying or routing to an
                                                    // objective without seeing its region is useless.
                                                    let mut engage = |tr: &mut crate::ui::QuestTracker, quests_on: &mut bool| {
                                                        if !tr.active.contains(&t.id) {
                                                            tr.active.insert(t.id.clone());
                                                        }
                                                        *quests_on = true;
                                                    };
                                                    if ui.small_button(RichText::new("go").size(10.0))
                                                        .on_hover_text("fly here (tracks the task so its zones show)").clicked()
                                                    {
                                                        engage(&mut tr, &mut quests_on);
                                                        cam_cmd.fly_to = Some(pos);
                                                    }
                                                    if ui
                                                        .add_enabled(pf_running, egui::Button::new(RichText::new("route").size(10.0)))
                                                        .on_hover_text("route here (tracks the task so its zones show)").clicked()
                                                    {
                                                        engage(&mut tr, &mut quests_on);
                                                        route.write(RouteRequest {
                                                            start: None,
                                                            dests: vec![pos],
                                                            optimize_order: false,
                                                            labels: vec![t.name.clone()],
                                                            ..Default::default()
                                                        });
                                                    }
                                                }
                                            });
                                        }
                                        let restrictions = objective_restrictions(o);
                                        if !restrictions.is_empty() {
                                            ui.horizontal_wrapped(|ui| {
                                                for r in restrictions {
                                                    theme::chip(ui, &r, ACCENT);
                                                }
                                            });
                                        }
                                        if !o.required_keys.is_empty() {
                                            ui.horizontal_wrapped(|ui| {
                                                ui.label(RichText::new("KEYS").size(9.0).strong().color(ACCENT));
                                                for key_group in &o.required_keys {
                                                    for item in key_group {
                                                        let mut have = owned_keys.contains(&item.name);
                                                        if ui.checkbox(&mut have, RichText::new(&item.name).size(9.5)).changed() {
                                                            if have { owned_keys.insert(item.name.clone()); }
                                                            else { owned_keys.remove(&item.name); }
                                                        }
                                                    }
                                                }
                                            });
                                        }
                                    });
                                });
                                if toggle {
                                    if is_done { done.remove(&key); } else { done.insert(key.clone()); }
                                }
                            }
                        });
                        // ONE border around the whole block (flush title bar + body) so the title reads
                        // as part of the container; TRACKED purple when tracked, else steel.
                        let block = title_resp.response.rect.union(body_resp.response.rect);
                        ui.painter().rect_stroke(block, 0.0, egui::Stroke::new(1.0, card_border), egui::StrokeKind::Inside);
                        ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0); // restore for the next task
                        ui.add_space(6.0);
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
    if progress.tracked != tracker.active {
        progress.tracked = tracker.active.clone();
    }
    if progress.done != done {
        progress.done = done;
    }
    if progress.owned_keys != owned_keys {
        progress.owned_keys = owned_keys;
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
/// present/unique across the schema, so the index keys it locally). Ungated: the game link
/// (game_watch.rs) writes the same keys when a task finishes in-game.
pub(crate) fn obj_key(task_id: &str, objective: &ObjectiveDef, i: usize) -> String {
    if objective.id.is_empty() {
        format!("{task_id}#{i}")
    } else {
        format!("{task_id}:{}", objective.id)
    }
}

#[cfg(feature = "egui")]
fn trim_num(value: f32) -> String {
    if value.fract().abs() < 0.001 { format!("{value:.0}") } else { format!("{value:.2}") }
}

#[cfg(feature = "egui")]
fn objective_restrictions(o: &ObjectiveDef) -> Vec<String> {
    let mut out = Vec::new();
    if !o.weapons.is_empty() { out.push(format!("weapon: {}", o.weapons.join(" / "))); }
    if !o.weapon_mods.is_empty() { out.push(format!("mods: {}", o.weapon_mods.join(" / "))); }
    if !o.wearing.is_empty() { out.push(format!("wear: {}", o.wearing.join(" / "))); }
    if !o.not_wearing.is_empty() { out.push(format!("without: {}", o.not_wearing.join(" / "))); }
    if !o.use_any.is_empty() { out.push(format!("use: {}", o.use_any.join(" / "))); }
    if let Some(d) = &o.distance {
        out.push(format!("distance {} {} m", d.compare.as_deref().unwrap_or("at"), trim_num(d.value)));
    }
    if !o.body_parts.is_empty() { out.push(format!("hit: {}", o.body_parts.join(" / "))); }
    if let Some(shot) = &o.shot_type { out.push(format!("shot: {shot}")); }
    if let Some([from, until]) = o.time_window { out.push(format!("time {}:00-{}:00", trim_num(from), trim_num(until))); }
    if o.min_durability.is_some() || o.max_durability.is_some() {
        out.push(format!(
            "durability {}-{}%",
            o.min_durability.map(trim_num).unwrap_or_else(|| "0".into()),
            o.max_durability.map(trim_num).unwrap_or_else(|| "100".into())
        ));
    }
    out
}

#[cfg(feature = "egui")]
fn reward_summary(r: &TaskRewards) -> String {
    let mut bits = Vec::new();
    if !r.items.is_empty() { bits.push(format!("{} item{}", r.items.len(), if r.items.len() == 1 { "" } else { "s" })); }
    if !r.standing.is_empty() { bits.push(format!("{} rep", r.standing.len())); }
    if !r.offers.is_empty() { bits.push(format!("{} unlock{}", r.offers.len(), if r.offers.len() == 1 { "" } else { "s" })); }
    bits.join("  \u{00B7}  ")
}

#[cfg(feature = "egui")]
fn reward_lines(r: &TaskRewards) -> Vec<String> {
    let mut out = Vec::new();
    for item in &r.items {
        let count = item.count.filter(|n| *n > 1).map(|n| format!(" x{n}")).unwrap_or_default();
        let price = item.price.filter(|v| *v > 0).map(|v| format!(" (~{} R)", crate::inspect::money(v))).unwrap_or_default();
        out.push(format!("{}{}{}", item.name, count, price));
    }
    for s in &r.standing { out.push(format!("{} reputation {:+}", s.trader, s.value.unwrap_or(0.0))); }
    for o in &r.offers { out.push(format!("Unlock: {}{} at {}", o.item.as_deref().unwrap_or("offer"), o.level.map(|v| format!(" LL{v}")).unwrap_or_default(), o.trader)); }
    for s in &r.skills { out.push(format!("Skill: {} +{}", s.name.as_deref().unwrap_or("skill"), trim_num(s.level.unwrap_or(0.0)))); }
    for t in &r.traders { out.push(format!("Unlock trader: {t}")); }
    for a in &r.achievements { out.push(format!("Achievement: {a}")); }
    for c in &r.customization { out.push(format!("Customization: {c}")); }
    out
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
