//! ui.rs — right-hand LAYER-TOGGLE panel (egui).
//!
//! A `SidePanel::right` with checkboxes to show/hide map overlay layers. The LOOT layer is
//! fully wired (master toggle + per-class filters + a min-value filter driving `LootClass`
//! marker visibility; the same `min_value` also prunes Map Intel's value-tagged loose loot). The
//! other layers (PMC/scav spawns, extracts, doors, interactables) are present as the framework
//! and light up as their data/overlays land (extract_semantics.py → semantics.json).
//!
//! `LayerToggles` + `apply_loot_visibility` exist even without the `egui` feature (so the loot
//! markers still respect a programmatic default); only the panel itself is egui-gated.

use crate::loot::LootClass;
use bevy::prelude::*;
use std::collections::BTreeMap;

/// The set of loot classes shown in the panel (order preserved by BTreeMap).
const LOOT_CLASSES: &[&str] = &[
    "weapon", "medical", "safe", "register", "bag", "crate", "tech", "stash", "furniture", "body",
];

#[derive(Resource, Clone, PartialEq)]
pub struct LayerToggles {
    pub loot: bool,
    /// Collapse dense point layers into camera-distance grid cells.
    pub cluster_dense: bool,
    /// class -> shown. Missing class defaults to shown.
    pub loot_classes: BTreeMap<String, bool>,
    /// Min ruble value for VALUE-TAGGED markers (`poi::MarkerValue`: container `ev` estimates +
    /// loose-loot prices); 0 = filter off. ONE filter shared by loot containers and Map Intel's
    /// loose loot, set from the Loot section's "min value" row. Untagged markers never filter.
    pub min_value: i64,
    /// GLOBAL "hide inactive" filter: hides every marker tagged `poi::SceneInactive` (gamedata
    /// records serialized `active: false` — disabled exfils, low-power minefields, off sniper
    /// zones, disabled doors/loot points) and their zone outlines. COMPOSES with the layer
    /// toggles like `min_value`; untagged markers never filter. ON by default (inactive hidden) so a
    /// fresh map isn't cluttered with disabled markers; `EFT_LAYERS=showinactive` starts it off, and
    /// it's a one-click toggle in the panel — inactive intel still matters when planning (a disabled
    /// exfil can be event-enabled mid-wipe).
    pub hide_inactive: bool,
    pub pmc_spawns: bool,
    pub scav_spawns: bool,
    pub bosses: bool,
    pub extracts: bool,
    pub doors: bool,
    pub interactables: bool,
    // ---- MAP INTEL (loot.json v2) ----
    pub locks: bool,
    pub hazards: bool,
    pub switches: bool,
    pub transits: bool,
    pub stationary: bool,
    pub loose: bool,
    // ---- TYPED GAME DATA (gamedata.json) ----
    pub minefields: bool,
    pub sniper_zones: bool,
    // ---- QUESTS (tasks.json) ----
    pub quests: bool,
}

impl Default for LayerToggles {
    fn default() -> Self {
        // `EFT_LAYERS=pmc,scav,boss,extract,door,interact,lock,hazard,switch,transit,stationary,loose`
        // pre-enables layers (dev/testing); normally only loot is on and the rest are toggled in
        // the panel.
        let on: std::collections::HashSet<String> = std::env::var("EFT_LAYERS")
            .ok()
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_default();
        let has = |k: &str| on.contains(k);
        Self {
            loot: !has("noloot"),
            cluster_dense: !has("nocluster"),
            loot_classes: LOOT_CLASSES.iter().map(|c| (c.to_string(), true)).collect(),
            min_value: 0,
            hide_inactive: !has("showinactive"),
            pmc_spawns: has("pmc"),
            scav_spawns: has("scav"),
            bosses: has("boss"),
            extracts: has("extract"),
            doors: has("door"),
            interactables: has("interact"),
            locks: has("lock"),
            hazards: has("hazard"),
            switches: has("switch"),
            transits: has("transit"),
            stationary: has("stationary"),
            loose: has("loose"),
            minefields: has("minefield"),
            sniper_zones: has("sniper"),
            quests: has("quest"),
        }
    }
}

/// Marker-search box state: the live query string. Matched (case-insensitively) against every
/// marker's `MarkerInfo` title/subtitle; a click flies the camera (`CameraCommand`) to the hit.
#[derive(Resource, Default)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct UiSearch {
    pub query: String,
}

/// Quest-tracker state: the checked ("active") task ids + the filter row. `active` drives per-task
/// marker visibility (poi::apply_quest_visibility) and the outline gizmo; the filters just prune
/// the checklist. `max_level == 0` means no level cap. Always present (poi.rs reads `active`).
#[derive(Resource, Default, Clone, PartialEq)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct QuestTracker {
    pub active: std::collections::HashSet<String>,
    pub kappa_only: bool,
    pub lk_only: bool,
    /// 0 = no cap.
    pub max_level: u32,
}

/// One pinned marker in the raid plan. `entity` ties the pin back to the live marker so pins
/// whose marker despawned self-prune; title/pos/value are snapshotted at pin time so the row
/// renders without re-resolving the marker every frame.
#[derive(Clone, PartialEq)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct PlanPin {
    pub entity: Entity,
    pub title: String,
    pub pos: Vec3,
    /// Estimated ruble value carried over from the marker's `poi::MarkerValue` (0 = unpriced).
    pub value: i64,
}

/// The raid plan: markers pinned from their inspect cards ("pin" button, inspect.rs). Read and
/// pruned by the panel's "Raid plan" section; mutations are click-gated so change detection
/// stays quiet.
#[derive(Resource, Default)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct PlanList {
    pub pins: Vec<PlanPin>,
}

/// One saved camera view. `pos` is the exact camera position at save time; `target` is the point
/// ~20 m along the camera forward that a bookmark click flies to (`CameraCommand::fly_to` reframes
/// with the standard offset — see the panel's views row). Both persist so an exact-pose restore
/// can be added later without a schema change.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct Bookmark {
    pub name: String,
    pub pos: [f32; 3],
    pub target: [f32; 3],
}

/// Saved camera views, persisted per map to `<pack>/bookmarks.json` (loaded once the pack is up,
/// written on every real change from the panel).
#[derive(Resource, Default, Clone, PartialEq)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct Bookmarks {
    pub views: Vec<Bookmark>,
    /// Set once the per-pack bookmarks.json load has run (whether or not the file existed).
    pub loaded: bool,
}

/// Position-HUD toggle: the small top-left live camera-coords readout (`pos_hud`). Default ON;
/// flipped by the "position HUD" checkbox in the panel footer.
#[derive(Resource)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct PosHud(pub bool);
impl Default for PosHud {
    fn default() -> Self {
        Self(true)
    }
}

/// Which settings group the right panel shows. Selected by the vertical icon toolbar; the
/// content panels (layers/camera/tasks) all render into the SAME `SidePanel::right` slot and
/// early-return when they aren't the active tab, so only one shows per frame.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub enum RightPanelTab {
    /// Map-overlay visibility (loot/spawns/extracts/hazards/quests/…) — the original panel.
    #[default]
    Visibility,
    /// Camera settings (FOV, exposure, fly speed, walk mode).
    Camera,
    /// Task / quest tracker (revamped module).
    Tasks,
    /// Navigation: place your position + route to extracts (navigate_panel module).
    Navigate,
    /// Level controls: power switches (toggle the lights each one drives).
    Level,
}

pub struct UiPlugin;
impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LayerToggles>()
            .init_resource::<UiSearch>()
            .init_resource::<QuestTracker>()
            .init_resource::<PlanList>()
            .init_resource::<Bookmarks>()
            .init_resource::<PosHud>()
            // EFT_TAB=camera|tasks|nav|vis seeds the initial right-panel tab (screenshots / power users).
            .insert_resource(match std::env::var("EFT_TAB").as_deref() {
                Ok("camera") => RightPanelTab::Camera,
                Ok("tasks") => RightPanelTab::Tasks,
                Ok("nav") | Ok("route") => RightPanelTab::Navigate,
                Ok("level") => RightPanelTab::Level,
                _ => RightPanelTab::Visibility,
            })
            // apply_loot_visibility ordered AFTER spawn_loot so a swap-respawn's fresh markers are
            // made visible (auto-sync point). teardown_ui drops per-map UI state on a swap.
            .add_systems(
                Update,
                (apply_loot_visibility.after(crate::loot::spawn_loot), load_bookmarks),
            )
            .add_systems(
                Update,
                teardown_ui.run_if(resource_changed::<crate::render::MapEpoch>),
            );
        // egui UI MUST run in EguiPrimaryContextPass (between egui's begin/end frame); in
        // plain Update the context has no fonts yet and `ctx_mut()` panics (bevy_egui 0.37).
        // toolbar_panel FIRST (rightmost narrow rail) then the tab content (to its left).
        #[cfg(feature = "egui")]
        app.add_systems(
            bevy_egui::EguiPrimaryContextPass,
            // .chain(): egui panel STACKING follows .show() order, so the toolbar must run first
            // (rightmost rail) and the content panels second (to its left). layers/camera/tasks
            // share the "map_layers" slot and each early-returns unless it's the active tab.
            // fit_camera_viewport LAST: once all right-side panels are laid out, shrink the 3D
            // camera viewport to the free central area so the scene re-centers instead of hiding
            // behind the panel.
            (
                toolbar_panel,
                layers_panel,
                camera_panel,
                level_panel,
                tasks_tab,
                crate::navigate_panel::navigate_tab,
                pos_hud,
                // NOTE: the in-raid EN/RU toggle is intentionally NOT registered (finding 8). It
                // flipped the shared Lang but the raid panels (navigate/tasks) are hardcoded
                // English, so it changed only the badge and misrepresented that RU took effect.
                // Language is set in the START MENU (which IS fully localized) until raid
                // localization exists; `lang_toggle` was removed rather than lie in-raid.
                map_loading_indicator,
                map_load_error_panel,
                fit_camera_viewport,
            )
                .chain(),
        );
    }
}

/// A small centered "Loading <map>…" toast shown while an in-place map swap is loading off-thread
/// (the previous map keeps rendering behind it, so the switch never freezes the frame).
#[cfg(feature = "egui")]
fn map_loading_indicator(
    mut contexts: bevy_egui::EguiContexts,
    menu: Option<Res<crate::menu::MenuState>>,
    pending: Res<crate::PendingMapLoad>,
    // The GPU build streams textures across many frames AFTER the .eftpack file finishes loading
    // (which is all `PendingMapLoad` tracks). Honor the render world's build flag too, so the toast
    // stays up for the WHOLE load — file load + GPU build — not just the file load.
    gpu_load: Option<Res<crate::render::GpuLoadSignal>>,
    pack: Option<Res<crate::render::LoadedPack>>,
) {
    use bevy_egui::egui::{self, RichText};
    use crate::ui_theme as theme;
    if menu.is_some() {
        return;
    }
    let building = gpu_load.as_ref().map(|s| s.in_progress()).unwrap_or(false);
    // Name to show: the loading file's name while it loads; once loaded, the pack's dataset name
    // (the GPU build phase); fall back to a generic label.
    let owned_name;
    let name = if let Some(n) = pending.loading() {
        n
    } else if building {
        owned_name = pack
            .as_ref()
            .map(|p| p.0.manifest.dataset.clone())
            .unwrap_or_else(|| "map".to_string());
        owned_name.as_str()
    } else {
        return; // nothing loading and no GPU build in progress
    };
    let label = titlecase(name);
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Area::new(egui::Id::new("map_loading"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 46.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme::CARD_TRANSLUCENT)
                .stroke(egui::Stroke::new(1.0, theme::ACCENT))
                .inner_margin(egui::Margin::symmetric(16, 9))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!("Loading  {label}\u{2026}"))
                            .size(14.0)
                            .strong()
                            .color(theme::TEXT_BRIGHT),
                    );
                });
        });
}

/// A failed async PLAY (corrupt/partial pack) used to leave a blank window with no message and no
/// way back (finding 4). This centered error card shows what failed + a "Back to menu" button
/// (relaunches into the start menu) and a "Dismiss" that just clears the error. Only shown outside
/// menu mode when `MapLoadError` is set.
#[cfg(feature = "egui")]
fn map_load_error_panel(
    mut contexts: bevy_egui::EguiContexts,
    menu: Option<Res<crate::menu::MenuState>>,
    mut err: ResMut<crate::MapLoadError>,
    mut back: ResMut<crate::ReturnToMenu>,
) {
    use bevy_egui::egui::{self, RichText};
    use crate::ui_theme as theme;
    if menu.is_some() || err.0.is_none() {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let msg = err.0.clone().unwrap_or_default();
    let mut dismiss = false;
    let mut go_back = false;
    egui::Area::new(egui::Id::new("map_load_error"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme::CARD)
                .stroke(egui::Stroke::new(1.0, theme::DANGER))
                .inner_margin(egui::Margin::symmetric(20, 16))
                .show(ui, |ui| {
                    ui.set_max_width(460.0);
                    ui.label(
                        RichText::new("MAP FAILED TO LOAD")
                            .size(16.0)
                            .strong()
                            .color(theme::DANGER_TEXT),
                    );
                    ui.add_space(6.0);
                    ui.label(RichText::new(&msg).size(12.0).color(theme::TEXT_BRIGHT));
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("The pack may be corrupt or incomplete \u{2014} rebuild it from the start menu.")
                            .size(11.0)
                            .color(theme::MUTED),
                    );
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        if ui.add(theme::primary_button("BACK TO MENU")).clicked() {
                            go_back = true;
                        }
                        if ui.button("Dismiss").clicked() {
                            dismiss = true;
                        }
                    });
                });
        });
    if go_back {
        back.0 = true;
        err.0 = None;
    } else if dismiss {
        err.0 = None;
    }
}

// (removed) `lang_toggle`: the in-raid EN/RU switch (finding 8). It persisted the shared `Lang` but
// the raid panels (navigate_panel / tasks_panel) hardcode English, so clicking RU changed only the
// language badge while viewing a map — a false claim that RU localization works in-raid. Language is
// chosen in the START MENU (fully localized) instead. `lang_switch_area` (menu.rs) is unchanged and
// still drives the menu toggle; re-register a raid toggle here once the raid panels are localized.

/// Re-center the 3D scene in the area egui leaves free (the window minus the right-side rail +
/// content panel) so it isn't just hidden behind the panel. We do NOT shrink the camera's viewport:
/// bevy_egui derives egui's own screen size from this camera's render target, so shrinking it feeds
/// back into `available_rect` and collapses the panel. Instead we apply an OFF-AXIS (lens-shift)
/// projection via `sub_camera_view`, which changes only the projection matrix — the render target
/// (and thus egui) stays the full window. Shifting the rendered content left by half the panel width
/// puts whatever WAS at window-center at the center of the free region. `world_to_viewport` /
/// `viewport_to_world` read the same shifted matrix, so marker billboards + the pick ray stay
/// consistent. Cleared in start-menu mode or when nothing occupies the sides.
#[cfg(feature = "egui")]
fn fit_camera_viewport(
    mut contexts: bevy_egui::EguiContexts,
    menu: Option<Res<crate::menu::MenuState>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut cam: Query<&mut bevy::camera::Camera, With<crate::render::CullCamera>>,
) {
    if menu.is_some() {
        if let Ok(mut c) = cam.single_mut() {
            if c.sub_camera_view.is_some() {
                c.sub_camera_view = None; // menu owns the whole screen — no shift
            }
        }
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let avail = ctx.available_rect(); // free central region (egui points), stable: we never shrink the target
    let ppp = ctx.pixels_per_point();
    let Ok(window) = windows.single() else {
        return;
    };
    let win_w = window.resolution.physical_width() as f32;
    let win_h = window.resolution.physical_height() as f32;
    if win_w < 1.0 || win_h < 1.0 {
        return;
    }
    let vis_w = (avail.width() * ppp).clamp(0.0, win_w);
    let panel_w = (win_w - vis_w).max(0.0);
    let Ok(mut camera) = cam.single_mut() else {
        return;
    };
    // No side panel (e.g. hide-all) -> centered full-window, no shift.
    if panel_w < 4.0 {
        if camera.sub_camera_view.is_some() {
            camera.sub_camera_view = None;
        }
        return;
    }
    // Lens-shift the content left by panel_w/2 px (offset.x on a full-window virtual sensor).
    let sub = bevy::camera::SubCameraView {
        full_size: UVec2::new(win_w as u32, win_h as u32),
        offset: Vec2::new(panel_w * 0.5, 0.0),
        size: UVec2::new(win_w as u32, win_h as u32),
    };
    let same = matches!(
        &camera.sub_camera_view,
        Some(s) if s.full_size == sub.full_size && s.size == sub.size
            && (s.offset - sub.offset).abs().max_element() < 0.5
    );
    if !same {
        camera.sub_camera_view = Some(sub);
    }
}

/// Load `<pack>/bookmarks.json` into `Bookmarks` whenever the map epoch advances (initial load +
/// every in-place swap). Epoch-tracked (not a one-shot bool) so a swap reloads the NEW pack's views
/// — and it reloads BEFORE the egui `layers_panel` write-back that frame, so the old map's views
/// can't serialize into the new pack's file. A missing/corrupt file just means an empty list.
fn load_bookmarks(
    mut bm: ResMut<Bookmarks>,
    pack: Option<Res<crate::render::LoadedPack>>,
    epoch: Res<crate::render::MapEpoch>,
    mut loaded_epoch: Local<Option<u64>>,
) {
    if *loaded_epoch == Some(epoch.0) {
        return;
    }
    let Some(pack) = pack else {
        return;
    };
    *loaded_epoch = Some(epoch.0);
    let path = pack.0.root.join("bookmarks.json");
    bm.views = std::fs::read_to_string(&path)
        .ok()
        .and_then(|txt| serde_json::from_str::<Vec<Bookmark>>(&txt).ok())
        .unwrap_or_default();
    bm.loaded = true;
}

/// In-place map swap: drop the per-map UI state whose `Entity` refs point into the OLD map's
/// markers (recycled ids would silently resolve to wrong new markers) and the quest tracker set
/// (the new map's task ids differ). Filter/view PREFERENCES are kept. `Bookmarks` reload is handled
/// by `load_bookmarks` (epoch-tracked); loot/POI/quest marker visibility by their epoch guards.
fn teardown_ui(mut plan: ResMut<PlanList>) {
    plan.pins.clear();
}

/// Show/hide loot markers by the master toggle AND the per-class filter AND the min-value
/// filter. Only touches the markers when the toggles change (true on the first run too, so the
/// initial state is applied once the markers exist), so it's ~free per frame.
fn apply_loot_visibility(
    toggles: Res<LayerToggles>,
    epoch: Res<crate::render::MapEpoch>,
    cam: Query<&GlobalTransform, With<crate::render::CullCamera>>,
    mut q: Query<(
        &LootClass,
        Option<&crate::poi::MarkerValue>,
        &GlobalTransform,
        Option<&crate::poi::DenseMarker>,
        &mut Visibility,
    )>,
) {
    // Re-apply on a toggle change OR a map swap (fresh markers spawn Hidden and the swap didn't
    // touch the toggles).
    if !toggles.cluster_dense && !toggles.is_changed() && !epoch.is_changed() {
        return;
    }
    let camera = cam.single().ok().map(|t| t.translation()).unwrap_or(Vec3::ZERO);
    let mut occupied = std::collections::HashSet::new();
    for (cls, val, gt, dense, mut vis) in &mut q {
        let mut shown = vis_for(&toggles, &cls.0, val) == Visibility::Visible;
        if shown && toggles.cluster_dense && dense.is_some() {
            let p = gt.translation();
            let distance = Vec2::new(p.x - camera.x, p.z - camera.z).length();
            let cell = if distance > 320.0 { 35.0 } else if distance > 140.0 { 14.0 } else { 0.0 };
            if cell > 0.0 {
                shown = occupied.insert((cls.0.clone(), (p.x / cell).floor() as i32, (p.z / cell).floor() as i32));
            }
        }
        *vis = if shown { Visibility::Visible } else { Visibility::Hidden };
    }
}

fn vis_for(t: &LayerToggles, cls: &str, val: Option<&crate::poi::MarkerValue>) -> Visibility {
    let shown = t.loot
        && t.loot_classes.get(cls).copied().unwrap_or(true)
        && crate::poi::value_passes(t.min_value, val);
    if shown {
        Visibility::Visible
    } else {
        Visibility::Hidden
    }
}

#[cfg(feature = "egui")]
fn titlecase(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// The min-value filter steps offered by the panel's "min value" selector (0 = Off). ASCII-only
/// labels — the default egui font has no ruble sign, and `inspect::money` doesn't emit one.
#[cfg(feature = "egui")]
const MIN_VALUE_STEPS: &[(i64, &str)] = &[
    (0, "Off"),
    (50_000, "50k"),
    (100_000, "100k"),
    (250_000, "250k"),
    (500_000, "500k"),
    (1_000_000, "1M"),
];

/// Short label for the current min-value ("Off"/"50k"/…); off-step values (none today) fall back
/// to the thousands-separated `inspect::money` form.
#[cfg(feature = "egui")]
fn min_value_label(v: i64) -> String {
    MIN_VALUE_STEPS
        .iter()
        .find(|(s, _)| *s == v)
        .map(|(_, n)| (*n).to_string())
        .unwrap_or_else(|| crate::inspect::money(v))
}

#[cfg(feature = "egui")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
/// Bundled params for the map dropdown + "Graphics (experimental)" section — keeps
/// `layers_panel` under Bevy's 16-system-param limit. Also carries the raid-planning state
/// (pin list, camera bookmarks, position-HUD toggle, camera transform) for the same reason.
#[derive(bevy::ecs::system::SystemParam)]
struct GfxUiParams<'w, 's> {
    gfx: ResMut<'w, crate::render::GfxSettings>,
    /// Forced LOD level (graphics-panel LOD selector); meaningful on --alllod packs, no-op on lean.
    forced_lod: ResMut<'w, crate::ForcedLod>,
    /// Active render path — the graphics panel greys out the GPU-driven-only controls when the
    /// viewer fell back to the M0/Standard path (which don't consume them; finding 9).
    render_path: Option<Res<'w, crate::render::RenderPath>>,
    map_switch: ResMut<'w, crate::MapSwitch>,
    /// Active toolbar tab — layers_panel early-returns unless this is `Visibility` (bundled here
    /// to keep layers_panel under the 16-system-param limit).
    tab: Res<'w, RightPanelTab>,
    /// Present only in start-menu mode (bare launch) — the panel stands down entirely.
    menu: Option<Res<'w, crate::menu::MenuState>>,
    pack: Option<Res<'w, crate::render::LoadedPack>>,
    /// (display name, pack path) list, scanned once from the packs/ dir beside the current pack.
    pack_list: bevy::ecs::system::Local<'s, Option<Vec<(String, String)>>>,
    /// Raid plan pins (inspect-card "pin" button fills it; the panel section lists/prunes it).
    plan: ResMut<'w, PlanList>,
    /// Saved camera views (persisted per pack to bookmarks.json).
    bookmarks: ResMut<'w, Bookmarks>,
    /// Position-HUD on/off (footer checkbox).
    hud: ResMut<'w, PosHud>,
    /// The fly-cam transform (root-level entity, so `Transform` IS world space) for "save view".
    cam: Query<'w, 's, &'static Transform, With<crate::render::CullCamera>>,
    /// Typed gamedata.json zone state — the footer credits the game files when it's live.
    gamedata: Res<'w, crate::poi::GameDataZones>,
    map_meta: Res<'w, crate::poi::MapIntelMeta>,
    progress: ResMut<'w, crate::progress::PlayerProgress>,
    /// Scene-inactive markers, counted next to the "hide inactive" filter checkbox (walls
    /// excluded — a zone would otherwise count twice: marker + wall).
    inactive: Query<
        'w,
        's,
        (),
        (
            bevy::prelude::With<crate::poi::SceneInactive>,
            bevy::prelude::Without<crate::poi::ZoneWall>,
        ),
    >,
}

#[cfg(feature = "egui")]
fn layers_panel(
    mut contexts: bevy_egui::EguiContexts,
    mut gfx_ui: GfxUiParams,
    mut toggles_res: ResMut<LayerToggles>,
    mut search: ResMut<UiSearch>,
    mut tracker_res: ResMut<QuestTracker>,
    quest_data: Res<crate::poi::QuestData>,
    key_catalog: Res<crate::poi::KeyCatalog>,
    markers: Query<(
        &crate::inspect::MarkerInfo,
        &GlobalTransform,
        Option<&crate::poi::PoiLayer>,
        Option<&crate::loot::LootClass>,
        Option<&crate::poi::QuestMarkerTask>,
        Option<&crate::poi::MarkerValue>,
        Option<&crate::poi::SceneInactive>,
    )>,
    // Zone-wall ribbons share their zone's `PoiLayer` for visibility but are scenery — keep
    // them out of the per-layer marker counts and the extract-routing destinations (a wall's
    // transform is identity; it would route the tour through the world origin).
    poi_q: Query<&crate::poi::PoiLayer, Without<crate::poi::ZoneWall>>,
    loot_q: Query<&crate::loot::LootClass>,
    mut cam_cmd: ResMut<crate::CameraCommand>,
    mut route_writer: MessageWriter<crate::pathfind::RouteRequest>,
    server: Res<crate::pathfind::PathfindServer>,
) {
    use bevy_egui::egui::{self, Color32, CollapsingHeader, RichText};
    use crate::pathfind::{RouteRequest, ServerStatus};
    use crate::poi::PoiLayer;
    if gfx_ui.menu.is_some() {
        return; // start-menu mode: menu.rs owns the whole screen
    }
    if *gfx_ui.tab != RightPanelTab::Visibility {
        return; // another tab owns the content panel this frame
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    // Clone-edit-compare (Codex review): passing `&mut toggles.x` from a ResMut into egui widgets
    // marks the resource CHANGED every frame the panel renders, which made apply_poi_visibility /
    // apply_loot_visibility / apply_quest_visibility rewrite every marker's Visibility per frame.
    // Widgets edit these copies; the deltas are written back once at the end only if real.
    let mut toggles = toggles_res.clone();
    let mut tracker = tracker_res.clone();
    // Same clone-compare for the bookmarks (write-back also persists to bookmarks.json) and the
    // HUD toggle. The raid-plan list is NOT cloned: its mutations are all click-gated, so it
    // never dirties change detection from mere rendering.
    let mut bm = gfx_ui.bookmarks.clone();
    let mut hud_on = gfx_ui.hud.0;
    // All colors come from the single source of truth (ui_theme); these are thin local aliases so
    // the panel body reads cleanly. No drifted literal values live here anymore.
    use crate::ui_theme as theme;
    const ACCENT: Color32 = theme::ACCENT;
    const MUTED: Color32 = theme::MUTED;
    const KEYCARD: Color32 = theme::VIOLET;

    // Per-layer marker counts (cheap: a few thousand markers, once per focused frame). Shown as a
    // dim number after each row so the planner can gauge density without enabling the layer.
    let mut poi_counts = [0usize; 16];
    for l in &poi_q {
        poi_counts[*l as usize] += 1;
    }
    let mut loot_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for c in &loot_q {
        *loot_counts.entry(c.0.clone()).or_default() += 1;
    }
    let loot_total: usize = loot_counts.values().sum();

    // Theme-standard side-panel frame (square, charcoal, panel margin). Per-widget RichText below
    // carries the rest of the look; global egui defaults are themed once in `apply_global_style`.
    egui::SidePanel::right("map_layers")
        .resizable(false)
        .frame(theme::panel_frame())
        .default_width(248.0)
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = theme::ITEM_SPACING;

            // ---- STICKY header + search (stay put while the sections scroll) ----
            ui.add_space(theme::SP_XS);
            ui.horizontal(|ui| {
                ui.label(theme::title("MAP  LAYERS"));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("hide all")
                        .on_hover_text("turn every overlay off")
                        .clicked()
                    {
                        hide_all(&mut toggles);
                    }
                });
            });
            // ---- Map dropdown (switching restarts the viewer into the selected pack) ----
            let cur_pack_root = gfx_ui
                .pack
                .as_ref()
                .map(|p| p.0.root.to_string_lossy().replace('\\', "/"));
            let packs = gfx_ui.pack_list.get_or_insert_with(|| {
                // Scan the packs/ dir next to the loaded pack (or ./packs as fallback), once.
                let dir = cur_pack_root
                    .as_deref()
                    .and_then(|r| std::path::Path::new(r).parent().map(|p| p.to_path_buf()))
                    .unwrap_or_else(|| crate::paths::packs_root().to_path_buf());
                let mut v: Vec<(String, String)> = std::fs::read_dir(&dir)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter_map(|e| {
                        let p = e.path();
                        let name = p.file_name()?.to_str()?.strip_suffix(".eftpack")?.to_string();
                        // a real pack has a manifest (skips half-built fleet output)
                        p.join("manifest.json").is_file()
                            .then(|| (name, p.to_string_lossy().replace('\\', "/")))
                    })
                    .collect();
                v.sort();
                v
            });
            let cur_name = cur_pack_root
                .as_deref()
                .and_then(|r| r.rsplit('/').next())
                .and_then(|n| n.strip_suffix(".eftpack"))
                .unwrap_or("(none)")
                .to_string();
            ui.horizontal(|ui| {
                ui.label(RichText::new("map").color(MUTED).size(11.0));
                egui::ComboBox::from_id_salt("map_select")
                    .selected_text(RichText::new(&cur_name).color(ACCENT).size(12.0))
                    .width(170.0)
                    .show_ui(ui, |ui| {
                        for (name, path) in packs.iter() {
                            if ui
                                .selectable_label(*name == cur_name, name)
                                .on_hover_text("switch to this map in place (no relaunch)")
                                .clicked()
                                && *name != cur_name
                            {
                                gfx_ui.map_switch.0 = Some(path.clone());
                            }
                        }
                    });
            });
            // ---- CAMERA BOOKMARKS (per map, persisted to <pack>/bookmarks.json). "save view"
            // snapshots the fly-cam; clicking a row flies to the stored TARGET via the standard
            // `CameraCommand` framing (fly_to is the only camera command — the exact saved pose
            // isn't restorable without touching main.rs, so pos is stored but unused for now).
            ui.horizontal(|ui| {
                ui.label(RichText::new("views").color(MUTED).size(11.0));
                if ui
                    .small_button("save view")
                    .on_hover_text("bookmark the current camera view (persists per map)")
                    .clicked()
                {
                    if let Ok(tf) = gfx_ui.cam.single() {
                        let pos = tf.translation;
                        let target = pos + tf.forward() * 20.0;
                        let mut n = bm.views.len() + 1;
                        while bm.views.iter().any(|b| b.name == format!("View {n}")) {
                            n += 1;
                        }
                        bm.views.push(Bookmark {
                            name: format!("View {n}"),
                            pos: pos.to_array(),
                            target: target.to_array(),
                        });
                    }
                }
            });
            let mut bm_remove: Option<usize> = None;
            for (i, b) in bm.views.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    if ui
                        .selectable_label(false, RichText::new(&b.name).size(12.0))
                        .on_hover_text("fly to this view")
                        .clicked()
                    {
                        cam_cmd.fly_to = Some(Vec3::from(b.target));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(RichText::new("\u{00D7}").size(12.0)).clicked() {
                            bm_remove = Some(i);
                        }
                    });
                });
            }
            if let Some(i) = bm_remove {
                bm.views.remove(i);
            }
            ui.add_space(4.0);
            ui.add(
                egui::TextEdit::singleline(&mut search.query)
                    .desired_width(f32::INFINITY)
                    .hint_text("Search markers\u{2026}"),
            );
            let q = search.query.trim().to_lowercase();
            if !q.is_empty() {
                // (rank, info, position, poi layer, loot class, quest task, value) — the
                // layer/class/task let a click auto-enable whatever hidden layer the hit lives
                // on; the value lets it also lift the min-value filter when that alone hides
                // the hit. Ranked: exact title > title prefix > title substring > subtitle or
                // detail-only match (an exact "RB-VO" must beat "RB-VO marked key" fragments).
                let mut hits = Vec::new();
                for (info, gt, layer, cls, qtask, val, inact) in &markers {
                    let tl = info.title.to_lowercase();
                    let rank = if tl == q {
                        0u8
                    } else if tl.starts_with(&q) {
                        1
                    } else if tl.contains(&q) {
                        2
                    } else if info.subtitle.to_lowercase().contains(&q)
                        || info.detail.iter().any(|d| d.to_lowercase().contains(&q))
                    {
                        3
                    } else {
                        continue;
                    };
                    hits.push((rank, info, gt.translation(), layer, cls, qtask, val, inact));
                }
                hits.sort_by_key(|h| h.0);
                let total = hits.len();
                ui.add_space(2.0);
                ui.label(RichText::new(format!("{total} results")).size(10.0).color(MUTED));
                egui::ScrollArea::vertical()
                    .id_salt("marker_search")
                    .max_height(200.0)
                    .show(ui, |ui| {
                        for (_, info, pos, layer, cls, qtask, val, inact) in hits.iter().take(25) {
                            // Is the hit's layer/class currently toggled off? (Clicking enables it.)
                            let hidden = if let Some(task) = qtask {
                                !toggles.quests
                                    || (!tracker.active.is_empty()
                                        && !tracker.active.contains(&task.0))
                            } else if let Some(l) = layer {
                                !*layer_toggle_mut(&mut toggles, **l)
                            } else if let Some(c) = cls {
                                !(toggles.loot
                                    && toggles.loot_classes.get(&c.0).copied().unwrap_or(true))
                            } else {
                                false
                            };
                            // Value-tagged hits (containers / loose loot) can ALSO be hidden by
                            // the min-value filter even with their layer on — surface that as
                            // "(filtered)" and lift the filter on click, else the fly-to lands
                            // on empty ground. Scene-inactive hits under the "hide inactive"
                            // filter get the exact same treatment.
                            let value_hidden =
                                !crate::poi::value_passes(toggles.min_value, *val);
                            let inactive_hidden = toggles.hide_inactive && inact.is_some();
                            // Second column: the subtitle, or — when only a detail line matched —
                            // that matching detail line, so the hit shows WHY it matched.
                            let second = if info.title.to_lowercase().contains(&q)
                                || info.subtitle.to_lowercase().contains(&q)
                            {
                                info.subtitle.as_str()
                            } else {
                                info.detail
                                    .iter()
                                    .find(|d| d.to_lowercase().contains(&q))
                                    .map(|s| s.as_str())
                                    .unwrap_or(info.subtitle.as_str())
                            };
                            let label =
                                RichText::new(format!("{}  \u{00B7}  {}", info.title, second));
                            ui.horizontal(|ui| {
                                if ui.selectable_label(false, label).clicked() {
                                    cam_cmd.fly_to = Some(*pos);
                                    // Flying to an invisible marker is useless — turn its layer on.
                                    if let Some(task) = qtask {
                                        toggles.quests = true;
                                        if !tracker.active.is_empty()
                                            && !tracker.active.contains(&task.0)
                                        {
                                            tracker.active.insert(task.0.clone());
                                        }
                                    } else if let Some(l) = layer {
                                        *layer_toggle_mut(&mut toggles, **l) = true;
                                    } else if let Some(c) = cls {
                                        toggles.loot = true;
                                        toggles.loot_classes.insert(c.0.clone(), true);
                                    }
                                    // ... and lift whichever global filter would keep this
                                    // hit invisible (min value / hide inactive).
                                    if value_hidden {
                                        toggles.min_value = 0;
                                    }
                                    if inactive_hidden {
                                        toggles.hide_inactive = false;
                                    }
                                }
                                if hidden {
                                    ui.label(RichText::new("(off)").size(10.0).color(MUTED));
                                } else if value_hidden || inactive_hidden {
                                    ui.label(RichText::new("(filtered)").size(10.0).color(MUTED));
                                }
                            });
                        }
                        if total > 25 {
                            ui.label(
                                RichText::new(format!("\u{2026} +{} more", total - 25))
                                    .size(10.0)
                                    .color(MUTED),
                            );
                        }
                    });
            }
            ui.add_space(4.0);
            ui.separator();

            // ---- SCROLLABLE body: all sections in collapsible groups so the panel never overflows ----
            egui::ScrollArea::vertical()
                .id_salt("panel_body")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if !gfx_ui.map_meta.name.is_empty() {
                        CollapsingHeader::new(RichText::new("Map overview").size(12.0).strong())
                            .id_salt("sec_map_overview")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label(RichText::new(&gfx_ui.map_meta.name).size(13.0).strong().color(ACCENT));
                                let mut raid = Vec::new();
                                if let Some(mins) = gfx_ui.map_meta.raid_minutes { raid.push(format!("{mins} min raid")); }
                                if let Some(players) = &gfx_ui.map_meta.players { raid.push(format!("{players} players")); }
                                if !raid.is_empty() { ui.label(RichText::new(raid.join("  \u{00B7}  ")).size(10.0).color(MUTED)); }
                                if !gfx_ui.map_meta.enemies.is_empty() {
                                    ui.label(RichText::new(format!("Enemies: {}", gfx_ui.map_meta.enemies.join(", "))).size(9.5).color(MUTED));
                                }
                                if !gfx_ui.map_meta.description.is_empty() {
                                    ui.label(RichText::new(&gfx_ui.map_meta.description).size(9.5).italics().color(MUTED));
                                }
                            });
                    }
                    // ===== RAID PLAN (markers pinned from their inspect cards) =====
                    // Self-prune pins whose marker entity despawned; write back only when
                    // something was actually dropped so change detection stays quiet.
                    let alive: Vec<PlanPin> = gfx_ui
                        .plan
                        .pins
                        .iter()
                        .filter(|p| markers.get(p.entity).is_ok())
                        .cloned()
                        .collect();
                    if alive.len() != gfx_ui.plan.pins.len() {
                        gfx_ui.plan.pins = alive;
                    }
                    let n_pins = gfx_ui.plan.pins.len();
                    CollapsingHeader::new(section_hdr("Raid plan", n_pins))
                        .id_salt("sec_plan")
                        .default_open(true)
                        .show(ui, |ui| {
                            if n_pins == 0 {
                                ui.label(
                                    RichText::new("pin markers from their info cards")
                                        .size(10.0)
                                        .italics()
                                        .color(MUTED),
                                );
                                return;
                            }
                            let mut remove: Option<usize> = None;
                            for (i, pin) in gfx_ui.plan.pins.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    let mut row = pin.title.clone();
                                    if pin.value > 0 {
                                        row.push_str(&format!(
                                            "  {}",
                                            crate::inspect::money(pin.value)
                                        ));
                                    }
                                    if ui
                                        .selectable_label(false, RichText::new(row).size(12.0))
                                        .on_hover_text("fly to this pin")
                                        .clicked()
                                    {
                                        cam_cmd.fly_to = Some(pin.pos);
                                    }
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui
                                                .small_button(
                                                    RichText::new("\u{00D7}").size(12.0),
                                                )
                                                .clicked()
                                            {
                                                remove = Some(i);
                                            }
                                        },
                                    );
                                });
                            }
                            if let Some(i) = remove {
                                gfx_ui.plan.pins.remove(i);
                            }
                            let total: i64 =
                                gfx_ui.plan.pins.iter().map(|p| p.value.max(0)).sum();
                            if total > 0 {
                                ui.label(
                                    RichText::new(format!(
                                        "total  {}",
                                        crate::inspect::money(total)
                                    ))
                                    .size(11.0)
                                    .color(Color32::from_gray(210)),
                                );
                            }
                            ui.add_space(2.0);
                            // Routing needs the :8091 server up — same gate as the Pathfinding
                            // section's "Route: nearest extract" button.
                            let pf_running = server.status == ServerStatus::Running;
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(pf_running, egui::Button::new("Route plan"))
                                    .on_hover_text(
                                        "shortest tour through every pin (pathfind server)",
                                    )
                                    .clicked()
                                {
                                    let dests: Vec<Vec3> =
                                        gfx_ui.plan.pins.iter().map(|p| p.pos).collect();
                                    if !dests.is_empty() {
                                        route_writer.write(RouteRequest {
                                            start: None,
                                            dests,
                                            optimize_order: true,
                                            ..Default::default()
                                        });
                                    }
                                }
                                if ui.button("Clear").clicked() {
                                    gfx_ui.plan.pins.clear();
                                }
                            });
                        });

                    // ===== LOOT =====
                    // The active min-value filter is surfaced in the header so it's visible even
                    // when the section is collapsed.
                    let loot_name = if toggles.min_value > 0 {
                        format!("Loot (min {})", min_value_label(toggles.min_value))
                    } else {
                        "Loot".to_string()
                    };
                    CollapsingHeader::new(section_hdr(&loot_name, loot_total))
                        .id_salt("sec_loot")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(
                                &mut toggles.loot,
                                RichText::new("Raw loot").size(14.0).strong(),
                            );
                            ui.checkbox(&mut toggles.cluster_dense, "adaptive marker clustering")
                                .on_hover_text("at long range, show one representative per grid cell to reduce clutter");
                            let loot_on = toggles.loot;
                            for (cls, on) in toggles.loot_classes.iter_mut() {
                                let n = loot_counts.get(cls).copied().unwrap_or(0);
                                ui.horizontal(|ui| {
                                    ui.add_space(10.0);
                                    let sw = if loot_on {
                                        theme::loot_class_color(cls)
                                    } else {
                                        Color32::from_gray(70)
                                    };
                                    theme::swatch(ui, sw);
                                    ui.add_enabled_ui(loot_on, |ui| {
                                        ui.checkbox(on, titlecase(cls));
                                    });
                                    theme::count_tag(ui, n);
                                });
                            }
                            // ---- MIN VALUE — ONE filter shared by the containers above and Map
                            // Intel's loose loot (both carry a `poi::MarkerValue`); every other
                            // marker kind ignores it. min_value lives in `LayerToggles`, so the
                            // apply systems re-run on change exactly like the toggles.
                            ui.add_space(2.0);
                            ui.horizontal(|ui| {
                                ui.add_space(10.0);
                                ui.label(RichText::new("min value").size(12.0).color(MUTED));
                                egui::ComboBox::from_id_salt("loot_min_value")
                                    .width(76.0)
                                    .selected_text(min_value_label(toggles.min_value))
                                    .show_ui(ui, |ui| {
                                        for &(v, name) in MIN_VALUE_STEPS {
                                            ui.selectable_value(&mut toggles.min_value, v, name);
                                        }
                                    });
                            });
                            ui.label(
                                RichText::new("also filters Map Intel \u{2192} Loose loot")
                                    .size(9.0)
                                    .italics()
                                    .color(MUTED),
                            );
                            // ---- HIDE INACTIVE — the OTHER global filter, kept beside min
                            // value: hides markers/outlines whose gamedata record is disabled
                            // in the game scene (poi::SceneInactive; cards say "Inactive in
                            // scene"). Composes with every layer toggle.
                            ui.horizontal(|ui| {
                                ui.add_space(10.0);
                                ui.checkbox(&mut toggles.hide_inactive, "hide inactive")
                                    .on_hover_text(
                                        "hide markers disabled in the game scene \
                                         (inactive exfils, low-power minefields, \u{2026})",
                                    );
                                theme::count_tag(ui, gfx_ui.inactive.iter().count());
                            });
                        });

                    // ===== SPAWNS & POIS =====
                    let spawn_total = poi_counts[PoiLayer::PmcSpawn as usize]
                        + poi_counts[PoiLayer::ScavSpawn as usize]
                        + poi_counts[PoiLayer::Boss as usize]
                        + poi_counts[PoiLayer::Extract as usize]
                        + poi_counts[PoiLayer::Door as usize]
                        + poi_counts[PoiLayer::Interactable as usize];
                    CollapsingHeader::new(section_hdr("Spawns & POIs", spawn_total))
                        .id_salt("sec_spawns")
                        .default_open(false)
                        .show(ui, |ui| {
                            poi_row(ui, &mut toggles.pmc_spawns, "PMC spawns", PoiLayer::PmcSpawn, &poi_counts);
                            poi_row(ui, &mut toggles.scav_spawns, "Scav spawns", PoiLayer::ScavSpawn, &poi_counts);
                            poi_row(ui, &mut toggles.bosses, "Bosses", PoiLayer::Boss, &poi_counts);
                            poi_row(ui, &mut toggles.extracts, "Extracts", PoiLayer::Extract, &poi_counts);
                            poi_row(ui, &mut toggles.doors, "Doors", PoiLayer::Door, &poi_counts);
                            // Name-classified props from the game files (jackets/weapon
                            // boxes/safes); mixes real lootables with decorative twins, so it
                            // reads "props", not "interactables".
                            poi_row(ui, &mut toggles.interactables, "Loot props", PoiLayer::Interactable, &poi_counts);
                        });

                    // ===== MAP INTEL =====
                    let intel_total = poi_counts[PoiLayer::Lock as usize]
                        + poi_counts[PoiLayer::Hazard as usize]
                        + poi_counts[PoiLayer::Switch as usize]
                        + poi_counts[PoiLayer::Transit as usize]
                        + poi_counts[PoiLayer::Stationary as usize]
                        + poi_counts[PoiLayer::LooseLoot as usize]
                        + poi_counts[PoiLayer::Minefield as usize]
                        + poi_counts[PoiLayer::SniperZone as usize];
                    CollapsingHeader::new(section_hdr("Map Intel", intel_total))
                        .id_salt("sec_intel")
                        .default_open(false)
                        .show(ui, |ui| {
                            poi_row(ui, &mut toggles.locks, "Locks & keys", PoiLayer::Lock, &poi_counts);
                            poi_row(ui, &mut toggles.hazards, "Hazards", PoiLayer::Hazard, &poi_counts);
                            // TYPED zones from the game files (gamedata.json): markers + red /
                            // orange footprint outlines (poi::draw_gamedata_outlines).
                            poi_row(ui, &mut toggles.minefields, "Minefields", PoiLayer::Minefield, &poi_counts);
                            poi_row(ui, &mut toggles.sniper_zones, "Sniper zones", PoiLayer::SniperZone, &poi_counts);
                            poi_row(ui, &mut toggles.switches, "Switches", PoiLayer::Switch, &poi_counts);
                            poi_row(ui, &mut toggles.transits, "Transits", PoiLayer::Transit, &poi_counts);
                            poi_row(ui, &mut toggles.stationary, "Stationary guns", PoiLayer::Stationary, &poi_counts);
                            poi_row(ui, &mut toggles.loose, "Loose loot", PoiLayer::LooseLoot, &poi_counts);

                            // ---- KEYS FOR THIS MAP (aggregated from the lock markers, price desc;
                            // poi::KeyCatalog). Click a key -> locks layer on + fly to a lock it opens.
                            if !key_catalog.keys.is_empty() {
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new("Keys for this map").size(11.0).color(MUTED),
                                );
                                for k in &key_catalog.keys {
                                    let mut row = format!(
                                        "{}  \u{00D7}{}",
                                        k.name,
                                        k.lock_positions.len()
                                    );
                                    if let Some(pr) = k.price.filter(|&p| p > 0) {
                                        row.push_str(&format!("  {}", crate::inspect::money(pr)));
                                    }
                                    // Keycards read violet (matches the marker/card accent).
                                    let text = if k.card {
                                        RichText::new(row).size(12.0).color(KEYCARD)
                                    } else {
                                        RichText::new(row).size(12.0)
                                    };
                                    ui.horizontal(|ui| {
                                        let mut owned = gfx_ui.progress.owns_key(&k.name);
                                        if ui.checkbox(&mut owned, "").on_hover_text("mark key owned for route planning").changed() {
                                            if owned { gfx_ui.progress.owned_keys.insert(k.name.clone()); }
                                            else { gfx_ui.progress.owned_keys.retain(|x| !x.eq_ignore_ascii_case(&k.name)); }
                                        }
                                        if ui.selectable_label(false, text).clicked() {
                                            toggles.locks = true;
                                            if let Some(p) = k.lock_positions.first() { cam_cmd.fly_to = Some(*p); }
                                        }
                                    });
                                }
                            }
                        });

                    // ===== QUESTS (visibility only; tracking/filters/objectives live in the
                    //       Tasks tab — the checklist icon in the toolbar) =====
                    CollapsingHeader::new(section_hdr(
                        "Quests",
                        poi_counts[PoiLayer::Quest as usize],
                    ))
                    .id_salt("sec_quests")
                    .default_open(false)
                    .show(ui, |ui| {
                        poi_row(ui, &mut toggles.quests, "Show quest markers", PoiLayer::Quest, &poi_counts);
                        ui.label(
                            RichText::new("track tasks, items + objectives in the Tasks tab")
                                .size(10.0)
                                .color(MUTED),
                        );
                    });

                    // (Pathfinding moved to its own Navigation tab — navigate_panel.rs. Position
                    // placement + the extract table + route status all live there now.)

                    // ---- Graphics (experimental): live toggles for the render features. ----
                    // Edits go through a local copy so change-detection only fires on a real
                    // tweak (a bare &mut through ResMut would mark the resource changed every
                    // frame the sliders render).
                    CollapsingHeader::new(section_hdr("Graphics (experimental)", 0))
                    .id_salt("sec_gfx")
                    .default_open(false)
                    .show(ui, |ui| {
                        let mut g = gfx_ui.gfx.clone();
                        // Finding 9: fog / sky-refl / emissive / shadows / grass / cull / LOD ride the
                        // GPU-driven shader uniforms and do NOTHING on the M0 (fixed flat-light) or
                        // Standard (Bevy PBR) fallbacks. Grey them out there so a fallback user can't
                        // fiddle dead sliders. Bloom / grade LUT / SSAO / sharpen run in the shared
                        // camera+post chain on every path, so they stay enabled.
                        let is_gpu = gfx_ui
                            .render_path
                            .as_deref()
                            .map(|p| *p == crate::render::RenderPath::GpuDriven)
                            .unwrap_or(true);
                        if !is_gpu {
                            ui.label(
                                RichText::new("compatibility renderer: some effects below need the GPU-driven path")
                                    .size(10.0)
                                    .italics()
                                    .color(theme::WARN),
                            );
                        }
                        ui.add_enabled(is_gpu, egui::Slider::new(&mut g.fog, 0.0..=2.0).text("fog"));
                        ui.add_enabled(is_gpu, egui::Slider::new(&mut g.sky_refl, 0.0..=2.0).text("sky reflections"));
                        ui.add_enabled(is_gpu, egui::Slider::new(&mut g.emissive, 0.0..=3.0).text("emissive"));
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut g.bloom, "bloom");
                            ui.add_enabled(
                                g.bloom,
                                egui::Slider::new(&mut g.bloom_intensity, 0.0..=0.3),
                            );
                        });
                        ui.add_enabled_ui(g.grade_available, |ui| {
                            ui.checkbox(&mut g.grade, "game grade LUT")
                                .on_hover_text("the game's own display chain; off = TonyMcMapface fallback");
                            ui.add_enabled(
                                g.grade && g.grade_available,
                                egui::Slider::new(&mut g.grade_exposure, 0.2..=4.0).text("exposure"),
                            );
                            ui.add_enabled(
                                g.grade && g.grade_available,
                                egui::Checkbox::new(&mut g.vignette, "vignette"),
                            );
                        });
                        ui.add_enabled_ui(g.shadows_available && is_gpu, |ui| {
                            ui.checkbox(&mut g.shadows, "sun shadows")
                                .on_hover_text("real-time cascades; marginal on the baked-GI look");
                        });
                        ui.add_enabled(is_gpu, egui::Checkbox::new(&mut g.grass, "grass"));
                        ui.add_enabled(
                            is_gpu,
                            egui::Slider::new(&mut g.cull_px, 0.0..=8.0)
                                .text("prop cull px")
                                .clamping(egui::SliderClamping::Always),
                        );
                        ui.add_enabled(
                            is_gpu,
                            egui::Slider::new(&mut g.cull_px_grass, 0.0..=16.0)
                                .text("grass cull px"),
                        );
                        ui.checkbox(&mut g.ssao, "SSAO (contact shading)")
                            .on_hover_text("depth-based ambient occlusion \u{2014} crevices/corners darken like the game");
                        ui.add_enabled(
                            g.ssao,
                            egui::Slider::new(&mut g.ssao_intensity, 0.0..=2.0).text("ssao intensity"),
                        );
                        ui.add_enabled(
                            g.ssao,
                            egui::Slider::new(&mut g.ssao_radius, 0.2..=2.0).text("ssao radius m"),
                        );
                        ui.add_enabled(
                            g.grade && g.grade_available,
                            egui::Slider::new(&mut g.sharpen, 0.0..=1.0).text("sharpen"),
                        )
                        .on_hover_text("EFT-style unsharp mask (the game ships ~0.5); needs the grade LUT");
                        // ---- lighting (live: rides the LightGrid uniform, no rebuild) ----
                        ui.separator();
                        ui.add_enabled(is_gpu, egui::Checkbox::new(&mut g.lights, "practical lights"))
                            .on_hover_text("realtime lamps/spots (maps with the direct/indirect light split); no effect on legacy full-bake packs");
                        ui.add_enabled(
                            is_gpu && g.lights,
                            egui::Slider::new(&mut g.light_intensity, 0.0..=3.0).text("light intensity"),
                        );
                        ui.add_enabled(
                            is_gpu,
                            egui::Slider::new(&mut g.sun_diffuse, 0.0..=2.5).text("sun diffuse"),
                        )
                        .on_hover_text("direct-sun fill on indirect-bake maps (1 = shipped); no-op where the bake already includes the sun");
                        ui.add_enabled(
                            is_gpu,
                            egui::Slider::new(&mut g.gi_intensity, 0.25..=2.0).text("GI brightness"),
                        )
                        .on_hover_text("baked ambient / global-illumination level");
                        // ---- photoreal extras (camera post; work on every render path) ----
                        ui.separator();
                        ui.checkbox(&mut g.dof, "depth of field")
                            .on_hover_text("bokeh focus blur (experimental)");
                        ui.add_enabled(
                            g.dof,
                            egui::Slider::new(&mut g.dof_focal_m, 1.0..=120.0)
                                .logarithmic(true)
                                .text("focus dist m"),
                        );
                        ui.add_enabled(
                            g.dof,
                            egui::Slider::new(&mut g.dof_fstop, 0.5..=16.0)
                                .logarithmic(true)
                                .text("f-stop"),
                        );
                        ui.add(egui::Slider::new(&mut g.chroma, 0.0..=0.05).text("chromatic aberration"))
                            .on_hover_text("subtle lens fringing; the game's own chain ships a touch of it");
                        // DISTANCE LOD (LOD_DISTANCE_PLAN.md): draw coarser mesh shells for distant
                        // objects. A LIVE cull-uniform switch — no rebuild. Meaningful only on an
                        // --alllod pack (multiple shells per group); a no-op on lean LOD0-only packs.
                        ui.add_enabled_ui(is_gpu, |ui| {
                            ui.checkbox(&mut g.lod_distance, "Distance LOD")
                                .on_hover_text("Swap in coarser shells for far geometry (needs an --alllod pack)");
                            ui.add_enabled(
                                g.lod_distance,
                                egui::Slider::new(&mut g.lod_bias, 0.25..=4.0).logarithmic(true).text("LOD bias"),
                            )
                            .on_hover_text(">1 holds finer detail to a greater distance; <1 switches coarse sooner");
                            ui.horizontal(|ui| {
                                ui.label("force shell");
                                let mut f = g.lod_force;
                                egui::ComboBox::from_id_salt("lod_force")
                                    .selected_text(if f < 0 { "off".to_string() } else { f.to_string() })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut f, -1, "off");
                                        for l in 0..=4 {
                                            ui.selectable_value(&mut f, l, l.to_string());
                                        }
                                    });
                                g.lod_force = f;
                            });
                        });
                        if ui.small_button("reset to defaults").clicked() {
                            let keep = (g.grade_available, g.shadows_available);
                            g = crate::render::GfxSettings::default();
                            g.grade_available = keep.0;
                            g.shadows_available = keep.1;
                        }
                        if g != *gfx_ui.gfx {
                            *gfx_ui.gfx = g;
                        }
                        ui.label(
                            RichText::new("changes apply live; defaults = shipped look")
                                .size(10.0)
                                .italics()
                                .color(MUTED),
                        );
                    });

                    ui.add_space(6.0);
                    // Provenance: with gamedata.json live, extracts/minefields/sniper zones/doors
                    // are TYPED data read from the game's own scene MonoBehaviours; otherwise the
                    // extracts fall back to tarkov.dev (extracts_dev) and doors to the name
                    // classifier.
                    let provenance = if gfx_ui.gamedata.live {
                        "exfils/mines/snipers: game files  \u{2022}  spawns/intel: tarkov.dev"
                    } else {
                        "spawns/extracts/intel: tarkov.dev  \u{2022}  doors/props: game files"
                    };
                    ui.label(
                        RichText::new(provenance)
                            .size(9.0)
                            .italics()
                            .color(MUTED),
                    );
                    ui.add_space(2.0);
                    ui.checkbox(&mut hud_on, RichText::new("position HUD").size(11.0))
                        .on_hover_text("live camera coords, top-left - copy for callouts");
                });
        });

    // Write back ONLY on real change so downstream is_changed() gates stay meaningful.
    if toggles != *toggles_res {
        *toggles_res = toggles;
    }
    if tracker != *tracker_res {
        *tracker_res = tracker;
    }
    if hud_on != gfx_ui.hud.0 {
        gfx_ui.hud.0 = hud_on;
    }
    // Bookmarks: a real change (save view / remove) also persists to <pack>/bookmarks.json.
    // A write failure only warns — the in-memory list still updates for this session.
    if bm != *gfx_ui.bookmarks {
        if let Some(pack) = gfx_ui.pack.as_ref() {
            let path = pack.0.root.join("bookmarks.json");
            match serde_json::to_string_pretty(&bm.views) {
                Ok(txt) => {
                    if let Err(e) = std::fs::write(&path, txt) {
                        warn!("ui: bookmarks save failed ({}): {e}", path.display());
                    }
                }
                Err(e) => warn!("ui: bookmarks serialize failed: {e}"),
            }
        }
        *gfx_ui.bookmarks = bm;
    }
}

/// POSITION HUD — a small live camera-coords readout (top-left, under the pick readout) with a
/// "copy" button so callouts can be shared. Toggled by `PosHud` (panel-footer checkbox). Styled
/// to match the pick readout: dark translucent box, small light text.
#[cfg(feature = "egui")]
fn pos_hud(
    mut contexts: bevy_egui::EguiContexts,
    hud: Res<PosHud>,
    menu: Option<Res<crate::menu::MenuState>>,
    cams: Query<&Transform, With<crate::render::CullCamera>>,
) {
    use bevy_egui::egui::{self, RichText};
    if !hud.0 || menu.is_some() {
        return; // hidden in start-menu mode (no raid context)
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let Ok(tf) = cams.single() else {
        return;
    };
    let p = tf.translation;
    // Camera ANGLE from the transform forward, in the EXACT convention `EFT_POSE`/`setup` REBUILD the
    // rotation with: `Ry(yaw)·Rx(pitch)` gives forward = (-cos p·sin yaw, sin p, -cos p·cos yaw). To
    // reproduce THIS forward we invert that: yaw = atan2(-fwd.x, -fwd.z), pitch = asin(fwd.y). (The old
    // atan2(fwd.x, -fwd.z) yielded the NEGATED yaw, so a copied pose fed back to EFT_POSE mirrored the
    // view across X — the reproducibility bug.)
    let fwd = *tf.forward();
    let yaw_deg = (-fwd.x).atan2(-fwd.z).to_degrees();
    let pitch_deg = fwd.y.clamp(-1.0, 1.0).asin().to_degrees();
    let dim = crate::ui_theme::SECTION;
    let bright = crate::ui_theme::TEXT_BRIGHT;
    let pos_s = format!("{:.1} {:.1} {:.1}", p.x, p.y, p.z);
    let ang_s = format!("{:.1} {:.1}", yaw_deg, pitch_deg);
    // One-line capture of the FULL camera pose (position + look angle) for reproducing a view.
    let capture = format!(
        "pos={:.4},{:.4},{:.4} yaw={:.5} pitch={:.5} fwd={:.6},{:.6},{:.6}",
        p.x, p.y, p.z, yaw_deg, pitch_deg, fwd.x, fwd.y, fwd.z
    );
    egui::Area::new(egui::Id::new("pos_hud"))
        .fixed_pos(egui::pos2(8.0, 36.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(crate::ui_theme::HUD_BG)
                .inner_margin(egui::Margin::same(6))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("POS").size(11.0).color(dim));
                        ui.label(RichText::new(&pos_s).size(13.0).color(bright));
                        ui.add_space(8.0);
                        ui.label(RichText::new("YAW/PITCH").size(11.0).color(dim));
                        ui.label(RichText::new(&ang_s).size(13.0).color(bright));
                        if ui.small_button("copy").clicked() {
                            // Copy the FULL pose so the exact angle is captured, not just xyz.
                            ui.ctx().copy_text(capture.clone());
                        }
                    });
                });
        });
}

/// Vertical icon toolbar (a thin rail on the window's right edge). Each vector-drawn icon
/// selects which settings group the content panel shows. Shown BEFORE the content panels so it
/// occupies the outermost (rightmost) slot.
#[cfg(feature = "egui")]
fn toolbar_panel(
    mut contexts: bevy_egui::EguiContexts,
    mut tab: ResMut<RightPanelTab>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut go_menu: ResMut<crate::ReturnToMenu>,
) {
    use bevy_egui::egui;
    use crate::ui_theme as theme;
    if menu.is_some() {
        return; // start menu owns the screen (and themes egui itself)
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    // Theme egui's own defaults once per frame (square corners, spacing, widget fills + text) so
    // every in-raid panel/card/button matches without per-widget restyling. toolbar_panel is the
    // first UI system in the raid chain, so this runs before the content panels each frame.
    theme::apply_global_style(ctx);
    let cur = *tab;
    egui::SidePanel::right("toolbar")
        .exact_width(46.0)
        .resizable(false)
        .frame(egui::Frame::new().fill(theme::RAIL).inner_margin(egui::Margin::symmetric(3, 8)))
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            // Top house = "Menu": back to the start menu (map manager). Sits above the tab icons,
            // separated — it's an action, not a tab, so it never shows the active-tab highlight.
            if theme::rail_button(ui, false, 4, "Menu", "Back to menu (map manager)") {
                go_menu.0 = true;
            }
            ui.add_space(3.0);
            ui.separator();
            ui.add_space(3.0);
            if theme::rail_button(ui, cur == RightPanelTab::Visibility, 0, "Layers", "Visibility layers") {
                *tab = RightPanelTab::Visibility;
            }
            if theme::rail_button(ui, cur == RightPanelTab::Camera, 1, "Camera", "Camera") {
                *tab = RightPanelTab::Camera;
            }
            if theme::rail_button(ui, cur == RightPanelTab::Tasks, 2, "Tasks", "Tasks") {
                *tab = RightPanelTab::Tasks;
            }
            if theme::rail_button(ui, cur == RightPanelTab::Navigate, 3, "Nav", "Navigation \u{00B7} routes") {
                *tab = RightPanelTab::Navigate;
            }
            // Bottom house = "Map": map-specific controls (level / power switches). The second of the
            // two house icons — the caption is what tells it apart from the top "Menu" one.
            if theme::rail_button(ui, cur == RightPanelTab::Level, 4, "Map", "Map \u{00B7} level & power controls") {
                *tab = RightPanelTab::Level;
            }
        });
}

/// Level-controls tab: flip the map's POWER SWITCHES (each toggles the exact light bank it drives,
/// derived from the game's own switch->LampController links). Extracts are shown in the map-overlay
/// layers (Visibility tab) already, so they are NOT duplicated here.
/// Renders into the same right-panel slot as the other tabs, gated on the active tab.
#[cfg(feature = "egui")]
fn level_panel(
    mut contexts: bevy_egui::EguiContexts,
    tab: Res<RightPanelTab>,
    menu: Option<Res<crate::menu::MenuState>>,
    pack: Option<Res<crate::render::LoadedPack>>,
    mut gfx: ResMut<crate::render::GfxSettings>,
    mut cam: Query<&mut Transform, With<crate::render::CullCamera>>,
) {
    use bevy_egui::egui::{self, RichText};
    use crate::ui_theme as theme;
    if menu.is_some() || *tab != RightPanelTab::Level {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let Some(pack) = pack else { return };
    const DIM: bevy_egui::egui::Color32 = theme::MUTED;
    let mut mask = gfx.light_groups; // clone-edit-compare so mere rendering never dirties change-detection
    egui::SidePanel::right("map_layers")
        .default_width(300.0)
        .frame(theme::panel_frame())
        .show(ctx, |ui| {
            ui.label(theme::title("LEVEL CONTROLS"));
            ui.add_space(theme::SP_MD);

            // ---- POWER ----
            ui.label(RichText::new("POWER").color(DIM).size(11.0));
            if pack.0.switches.is_empty() {
                ui.label(RichText::new("No power switches on this map.").color(DIM).size(11.0));
            } else {
                for (i, sw) in pack.0.switches.iter().enumerate() {
                    let g = sw.group_idx;
                    ui.horizontal(|ui| {
                        if g >= 0 && g < 32 {
                            let bit = 1u32 << g;
                            let mut on = mask & bit != 0;
                            let label = if pack.0.switches.len() == 1 {
                                format!("Power  ({} lamps)", sw.count)
                            } else {
                                format!("Power {}  ({} lamps)", i + 1, sw.count)
                            };
                            if ui.checkbox(&mut on, label).changed() {
                                if on { mask |= bit } else { mask &= !bit }
                            }
                        } else {
                            ui.add_enabled(false, egui::Checkbox::new(&mut false, "Power (no lights)"));
                        }
                        if ui.small_button("go").on_hover_text("jump to the switch").clicked() {
                            if let Ok(mut t) = cam.single_mut() {
                                // stand a few metres back + above the lever, looking at it
                                let p = sw.world_pos + Vec3::new(0.0, 2.0, 5.0);
                                t.translation = p;
                                t.look_at(sw.world_pos, Vec3::Y);
                            }
                        }
                    });
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.small_button("All on").clicked() {
                        for sw in &pack.0.switches {
                            if (0..32).contains(&sw.group_idx) {
                                mask |= 1 << sw.group_idx;
                            }
                        }
                    }
                    if ui.small_button("All off").clicked() {
                        mask = 0;
                    }
                });
                ui.label(
                    RichText::new("Maps spawn un-powered (dark). Flip a switch to light its bank \u{2014} or click the switch in the world.")
                        .color(DIM)
                        .size(10.0),
                );
            }
        });
    if mask != gfx.light_groups {
        gfx.light_groups = mask; // one write only on a real change
    }
}

/// Camera-settings tab: FOV, exposure, fly speed (scroll-adjustable), walk-mode toggle. Renders
/// into the same content slot as layers_panel, gated on the active tab.
#[cfg(feature = "egui")]
fn camera_panel(
    mut contexts: bevy_egui::EguiContexts,
    tab: Res<RightPanelTab>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut cam: ResMut<crate::CameraSettings>,
    mut gfx: ResMut<crate::render::GfxSettings>,
) {
    use bevy_egui::egui::{self, RichText};
    use crate::ui_theme as theme;
    if menu.is_some() || *tab != RightPanelTab::Camera {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    const DIM: bevy_egui::egui::Color32 = theme::MUTED;
    // Clone-edit-compare so merely rendering the sliders doesn't dirty change detection every
    // frame (only real edits write back — same discipline as the graphics panel).
    let mut fov = cam.fov_deg;
    let mut fly = cam.fly_speed;
    let mut walk = cam.walk_mode;
    let mut expo = gfx.grade_exposure;
    egui::SidePanel::right("map_layers")
        .default_width(300.0)
        .frame(theme::panel_frame())
        .show(ctx, |ui| {
            ui.label(theme::title("CAMERA"));
            ui.add_space(theme::SP_MD);

            ui.label(RichText::new("FIELD OF VIEW").color(DIM).size(11.0));
            ui.add(egui::Slider::new(&mut fov, 20.0..=110.0).suffix("\u{00B0}").text(""));
            ui.add_space(6.0);

            ui.label(RichText::new("EXPOSURE").color(DIM).size(11.0));
            ui.add(egui::Slider::new(&mut expo, 0.2..=4.0).text(""));
            ui.add_space(6.0);

            ui.label(RichText::new("FLY SPEED  (scroll wheel)").color(DIM).size(11.0));
            ui.add(egui::Slider::new(&mut fly, 2.0..=1500.0).logarithmic(true).suffix(" m/s").text(""));
            ui.add_space(10.0);

            ui.checkbox(&mut walk, "Walk mode (ground-follow + jump)");
            ui.label(
                RichText::new(
                    "WASD walk, Space jump, Shift sprint; scroll scales walk speed + jump height",
                )
                .color(DIM)
                .size(10.0),
            );
        });
    if fov != cam.fov_deg {
        cam.fov_deg = fov;
    }
    if fly != cam.fly_speed {
        cam.fly_speed = fly;
    }
    if walk != cam.walk_mode {
        cam.walk_mode = walk;
    }
    if expo != gfx.grade_exposure {
        gfx.grade_exposure = expo;
    }
}

/// Tasks tab: opens the shared content slot and delegates to the revamped `tasks_panel` module
/// (trader-grouped task cards, required-item icons, objective go/route, map filter). Gated on the
/// active tab like the other content panels.
#[cfg(feature = "egui")]
fn tasks_tab(
    mut contexts: bevy_egui::EguiContexts,
    tab: Res<RightPanelTab>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut params: crate::tasks_panel::TasksPanelParams,
) {
    use bevy_egui::egui;
    if menu.is_some() || *tab != RightPanelTab::Tasks {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::SidePanel::right("map_layers")
        .default_width(320.0)
        .frame(crate::ui_theme::panel_frame())
        .show(ctx, |ui| {
            crate::tasks_panel::tasks_panel_ui(ui, &mut params);
        });
}

/// Section header text: name + a dim count of markers in that section. Thin wrapper over the shared
/// `ui_theme::section_header` so every section title (here + the Tasks tab) is one style.
#[cfg(feature = "egui")]
fn section_hdr(name: &str, count: usize) -> bevy_egui::egui::RichText {
    crate::ui_theme::section_header(name, count)
}

/// Turn every overlay layer off (the panel's "hide all" quick action).
#[cfg(feature = "egui")]
fn hide_all(t: &mut LayerToggles) {
    t.loot = false;
    t.pmc_spawns = false;
    t.scav_spawns = false;
    t.bosses = false;
    t.extracts = false;
    t.doors = false;
    t.interactables = false;
    t.locks = false;
    t.hazards = false;
    t.switches = false;
    t.transits = false;
    t.stationary = false;
    t.loose = false;
    t.minefields = false;
    t.sniper_zones = false;
    t.quests = false;
}

/// The `LayerToggles` field owning a POI layer's visibility — the same mapping as poi.rs
/// `apply_poi_visibility` — so search can flip a hidden layer on when one of its hits is clicked.
#[cfg(feature = "egui")]
fn layer_toggle_mut(t: &mut LayerToggles, l: crate::poi::PoiLayer) -> &mut bool {
    use crate::poi::PoiLayer as P;
    match l {
        P::PmcSpawn => &mut t.pmc_spawns,
        P::ScavSpawn => &mut t.scav_spawns,
        P::Boss => &mut t.bosses,
        P::Extract => &mut t.extracts,
        P::Door => &mut t.doors,
        P::Interactable => &mut t.interactables,
        P::Lock => &mut t.locks,
        P::Hazard => &mut t.hazards,
        P::Switch => &mut t.switches,
        P::Transit => &mut t.transits,
        P::Stationary => &mut t.stationary,
        P::LooseLoot => &mut t.loose,
        P::Quest => &mut t.quests,
        P::Minefield => &mut t.minefields,
        P::SniperZone => &mut t.sniper_zones,
    }
}

/// One POI toggle row: colour swatch + checkbox + a right-aligned dim marker count. Uses the shared
/// theme swatch (colour from `poi::poi_look`, matching the on-map marker) + count tag.
#[cfg(feature = "egui")]
fn poi_row(
    ui: &mut bevy_egui::egui::Ui,
    on: &mut bool,
    label: &str,
    l: crate::poi::PoiLayer,
    counts: &[usize; 16],
) {
    ui.horizontal(|ui| {
        ui.add_space(crate::ui_theme::SP_XS);
        crate::ui_theme::swatch(ui, crate::ui_theme::poi_color(l));
        ui.checkbox(on, label);
        crate::ui_theme::count_tag(ui, counts[l as usize]);
    });
}
