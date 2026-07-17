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
    /// class -> shown. Missing class defaults to shown.
    pub loot_classes: BTreeMap<String, bool>,
    /// Min ruble value for VALUE-TAGGED markers (`poi::MarkerValue`: container `ev` estimates +
    /// loose-loot prices); 0 = filter off. ONE filter shared by loot containers and Map Intel's
    /// loose loot, set from the Loot section's "min value" row. Untagged markers never filter.
    pub min_value: i64,
    /// GLOBAL "hide inactive" filter: hides every marker tagged `poi::SceneInactive` (gamedata
    /// records serialized `active: false` — disabled exfils, low-power minefields, off sniper
    /// zones, disabled doors/loot points) and their zone outlines. COMPOSES with the layer
    /// toggles like `min_value`; untagged markers never filter. Off by default: inactive intel
    /// still matters when planning (a disabled exfil can be event-enabled mid-wipe).
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
            loot_classes: LOOT_CLASSES.iter().map(|c| (c.to_string(), true)).collect(),
            min_value: 0,
            hide_inactive: has("hideinactive"),
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
            // EFT_TAB=camera|tasks|vis seeds the initial right-panel tab (screenshots / power users).
            .insert_resource(match std::env::var("EFT_TAB").as_deref() {
                Ok("camera") => RightPanelTab::Camera,
                Ok("tasks") => RightPanelTab::Tasks,
                _ => RightPanelTab::Visibility,
            })
            .add_systems(Update, (apply_loot_visibility, load_bookmarks));
        // egui UI MUST run in EguiPrimaryContextPass (between egui's begin/end frame); in
        // plain Update the context has no fonts yet and `ctx_mut()` panics (bevy_egui 0.37).
        // toolbar_panel FIRST (rightmost narrow rail) then the tab content (to its left).
        #[cfg(feature = "egui")]
        app.add_systems(
            bevy_egui::EguiPrimaryContextPass,
            // .chain(): egui panel STACKING follows .show() order, so the toolbar must run first
            // (rightmost rail) and the content panel second (to its left).
            (toolbar_panel, layers_panel, camera_panel, pos_hud).chain(),
        );
    }
}

/// One-shot: once the pack is loaded, read `<pack>/bookmarks.json` into `Bookmarks`. After the
/// first success it's a single bool check per frame. A missing/corrupt file just means an empty
/// list (the next save recreates it).
fn load_bookmarks(mut bm: ResMut<Bookmarks>, pack: Option<Res<crate::render::LoadedPack>>) {
    if bm.loaded {
        return;
    }
    let Some(pack) = pack else {
        return;
    };
    let path = pack.0.root.join("bookmarks.json");
    let views = std::fs::read_to_string(&path)
        .ok()
        .and_then(|txt| serde_json::from_str::<Vec<Bookmark>>(&txt).ok())
        .unwrap_or_default();
    bm.views = views;
    bm.loaded = true;
}

/// Show/hide loot markers by the master toggle AND the per-class filter AND the min-value
/// filter. Only touches the markers when the toggles change (true on the first run too, so the
/// initial state is applied once the markers exist), so it's ~free per frame.
fn apply_loot_visibility(
    toggles: Res<LayerToggles>,
    mut q: Query<(&LootClass, Option<&crate::poi::MarkerValue>, &mut Visibility)>,
) {
    if !toggles.is_changed() {
        return;
    }
    for (cls, val, mut vis) in &mut q {
        *vis = vis_for(&toggles, &cls.0, val);
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

/// Vivid, distinct legend colour per loot class (doubles as the on-map key).
#[cfg(feature = "egui")]
fn class_color(cls: &str) -> bevy_egui::egui::Color32 {
    use bevy_egui::egui::Color32;
    match cls {
        "weapon" => Color32::from_rgb(214, 92, 72),
        "medical" => Color32::from_rgb(92, 200, 122),
        "safe" => Color32::from_rgb(235, 190, 74),
        "register" => Color32::from_rgb(84, 162, 235),
        "bag" => Color32::from_rgb(205, 150, 92),
        "crate" => Color32::from_rgb(196, 162, 108),
        "tech" => Color32::from_rgb(176, 112, 226),
        "furniture" => Color32::from_rgb(162, 138, 116),
        "stash" => Color32::from_rgb(150, 150, 150),
        "body" => Color32::from_rgb(222, 74, 74),
        _ => Color32::from_rgb(180, 180, 180),
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
    extracts: Query<
        (
            &crate::poi::PoiLayer,
            &GlobalTransform,
            Option<&crate::poi::ExtractFaction>,
        ),
        Without<crate::poi::ZoneWall>,
    >,
    mut cam_cmd: ResMut<crate::CameraCommand>,
    mut route_writer: MessageWriter<crate::pathfind::RouteRequest>,
    mut server_cmd: MessageWriter<crate::pathfind::ServerCmd>,
    route_result: Res<crate::pathfind::RouteResult>,
    server: Res<crate::pathfind::PathfindServer>,
) {
    use bevy_egui::egui::{self, Color32, CollapsingHeader, RichText};
    use crate::pathfind::{RouteRequest, RouteStatus, ServerCmd, ServerStatus};
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
    const ACCENT: Color32 = Color32::from_rgb(232, 194, 122); // warm tactical amber
    const HDR: Color32 = Color32::from_rgb(160, 164, 160);
    const MUTED: Color32 = Color32::from_rgb(120, 122, 120);
    const DIMCOUNT: Color32 = Color32::from_rgb(110, 112, 110);
    const PANEL_BG: Color32 = Color32::from_rgb(20, 22, 23);
    const KEYCARD: Color32 = Color32::from_rgb(184, 115, 235); // violet = Color::srgb(0.72,0.45,0.92)

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

    // Style ONLY this panel's frame (a global ctx.set_style() was painting a fullscreen white
    // layer over the 3D scene). Per-widget RichText below carries the rest of the look.
    let frame = egui::Frame::side_top_panel(&ctx.style())
        .fill(PANEL_BG)
        .inner_margin(egui::Margin::same(14));
    egui::SidePanel::right("map_layers")
        .resizable(false)
        .frame(frame)
        .default_width(248.0)
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 6.0);

            // ---- STICKY header + search (stay put while the sections scroll) ----
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("MAP  LAYERS").color(ACCENT).size(17.0).strong());
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
                                .on_hover_text("switch map (restarts the viewer)")
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
                    CollapsingHeader::new(section_hdr("Raid plan", n_pins, HDR))
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
                    CollapsingHeader::new(section_hdr(&loot_name, loot_total, HDR))
                        .id_salt("sec_loot")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(
                                &mut toggles.loot,
                                RichText::new("Raw loot").size(14.0).strong(),
                            );
                            let loot_on = toggles.loot;
                            for (cls, on) in toggles.loot_classes.iter_mut() {
                                let n = loot_counts.get(cls).copied().unwrap_or(0);
                                ui.horizontal(|ui| {
                                    ui.add_space(10.0);
                                    let swatch =
                                        if loot_on { class_color(cls) } else { Color32::from_gray(70) };
                                    ui.label(RichText::new("\u{25CF}").color(swatch).size(12.0));
                                    ui.add_enabled_ui(loot_on, |ui| {
                                        ui.checkbox(on, titlecase(cls));
                                    });
                                    count_tag(ui, n, DIMCOUNT);
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
                                count_tag(ui, gfx_ui.inactive.iter().count(), DIMCOUNT);
                            });
                        });

                    // ===== SPAWNS & POIS =====
                    let spawn_total = poi_counts[PoiLayer::PmcSpawn as usize]
                        + poi_counts[PoiLayer::ScavSpawn as usize]
                        + poi_counts[PoiLayer::Boss as usize]
                        + poi_counts[PoiLayer::Extract as usize]
                        + poi_counts[PoiLayer::Door as usize]
                        + poi_counts[PoiLayer::Interactable as usize];
                    CollapsingHeader::new(section_hdr("Spawns & POIs", spawn_total, HDR))
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
                    CollapsingHeader::new(section_hdr("Map Intel", intel_total, HDR))
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
                                    if ui.selectable_label(false, text).clicked() {
                                        toggles.locks = true;
                                        if let Some(p) = k.lock_positions.first() {
                                            cam_cmd.fly_to = Some(*p);
                                        }
                                    }
                                }
                            }
                        });

                    // ===== TASKS / QUESTS =====
                    CollapsingHeader::new(section_hdr(
                        "Tasks / Quests",
                        poi_counts[PoiLayer::Quest as usize],
                        HDR,
                    ))
                    .id_salt("sec_quests")
                    .default_open(false)
                    .show(ui, |ui| {
                        poi_row(ui, &mut toggles.quests, "Show quest markers", PoiLayer::Quest, &poi_counts);
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut tracker.kappa_only, "Kappa");
                            ui.checkbox(&mut tracker.lk_only, "Lightkeeper");
                        });
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut tracker.max_level).range(0..=79));
                            ui.label(RichText::new("\u{2264} Lvl").size(12.0).color(MUTED));
                        });

                        let (kappa_only, lk_only, max_level) =
                            (tracker.kappa_only, tracker.lk_only, tracker.max_level);
                        let shown: Vec<&crate::poi::QuestEntry> = quest_data
                            .tasks
                            .iter()
                            .filter(|t| {
                                if kappa_only && !t.kappa {
                                    return false;
                                }
                                if lk_only && !t.lk {
                                    return false;
                                }
                                if max_level > 0 {
                                    if let Some(ml) = t.min_level {
                                        if ml > max_level {
                                            return false;
                                        }
                                    }
                                }
                                true
                            })
                            .collect();
                        ui.add_space(2.0);
                        ui.label(
                            RichText::new(format!("{} tasks", shown.len())).size(10.0).color(MUTED),
                        );
                        egui::ScrollArea::vertical()
                            .id_salt("quest_list")
                            .max_height(240.0)
                            .show(ui, |ui| {
                                for t in &shown {
                                    ui.horizontal(|ui| {
                                        let mut on = tracker.active.contains(&t.id);
                                        if ui.checkbox(&mut on, "").changed() {
                                            if on {
                                                tracker.active.insert(t.id.clone());
                                            } else {
                                                tracker.active.remove(&t.id);
                                            }
                                        }
                                        let name =
                                            if t.name.is_empty() { "Task" } else { t.name.as_str() };
                                        if ui
                                            .selectable_label(false, RichText::new(name).size(13.0))
                                            .clicked()
                                        {
                                            if let Some(pos) = t
                                                .objectives
                                                .first()
                                                .and_then(|o| o.zones.first())
                                                .map(|z| z.pos)
                                            {
                                                cam_cmd.fly_to = Some(pos);
                                            }
                                        }
                                    });
                                    let mut tags = if t.trader.is_empty() {
                                        String::new()
                                    } else {
                                        t.trader.clone()
                                    };
                                    if let Some(ml) = t.min_level.filter(|&l| l > 0) {
                                        if !tags.is_empty() {
                                            tags.push_str("  \u{00B7}  ");
                                        }
                                        tags.push_str(&format!("Lvl {ml}"));
                                    }
                                    if t.kappa {
                                        if !tags.is_empty() {
                                            tags.push_str("  \u{00B7}  ");
                                        }
                                        tags.push_str("Kappa");
                                    }
                                    if !tags.is_empty() {
                                        ui.label(RichText::new(tags).size(10.0).color(MUTED));
                                    }
                                }
                            });

                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Route active").clicked() {
                                let dests: Vec<Vec3> = quest_data
                                    .tasks
                                    .iter()
                                    .filter(|t| tracker.active.contains(&t.id))
                                    .flat_map(|t| {
                                        t.objectives
                                            .iter()
                                            .filter_map(|o| o.zones.first().map(|z| z.pos))
                                    })
                                    .collect();
                                if !dests.is_empty() {
                                    route_writer.write(RouteRequest {
                                        start: None,
                                        dests,
                                        optimize_order: true,
                                    });
                                }
                            }
                            if ui.button("Clear route").clicked() {
                                route_writer.write(RouteRequest {
                                    start: None,
                                    dests: Vec::new(),
                                    optimize_order: false,
                                });
                            }
                        });
                        ui.add_space(2.0);
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
                                ui.label(
                                    RichText::new(e.as_str())
                                        .size(10.0)
                                        .color(Color32::from_rgb(210, 96, 84)),
                                );
                            }
                            RouteStatus::Idle => {}
                        }
                    });

                    // ===== PATHFINDING (server start/stop) =====
                    CollapsingHeader::new(RichText::new("Pathfinding").color(HDR).size(12.0).strong())
                        .id_salt("sec_pathfind")
                        .default_open(false)
                        .show(ui, |ui| {
                            let (dot, txt, col) = match server.status {
                                ServerStatus::Running => {
                                    ("\u{25CF}", "server running", Color32::from_rgb(120, 210, 130))
                                }
                                ServerStatus::Starting => {
                                    ("\u{25CF}", "server starting\u{2026}", ACCENT)
                                }
                                ServerStatus::Stopped => {
                                    ("\u{25CF}", "server stopped", Color32::from_gray(130))
                                }
                            };
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(dot).color(col).size(11.0));
                                ui.label(RichText::new(txt).color(col).size(12.0));
                            });
                            let running = server.status == ServerStatus::Running;
                            let starting = server.status == ServerStatus::Starting;
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(!running && !starting, egui::Button::new("Start server"))
                                    .clicked()
                                {
                                    server_cmd.write(ServerCmd::Start);
                                }
                                if ui
                                    .add_enabled(running || starting, egui::Button::new("Stop server"))
                                    .clicked()
                                {
                                    server_cmd.write(ServerCmd::Stop);
                                }
                            });
                            // One-click route from the camera through every extract — the `chain`
                            // query re-orders the stops, so the nearest extract comes first.
                            if ui
                                .add_enabled(running, egui::Button::new("Route: nearest extract"))
                                .clicked()
                            {
                                let dests: Vec<Vec3> = extracts
                                    .iter()
                                    .filter(|(l, _, _)| **l == PoiLayer::Extract)
                                    .map(|(_, gt, _)| gt.translation())
                                    .collect();
                                if !dests.is_empty() {
                                    route_writer.write(RouteRequest {
                                        start: None,
                                        dests,
                                        optimize_order: true,
                                    });
                                }
                            }
                            ui.label(
                                RichText::new(
                                    "on-demand routing via the local :8091 GPU server (first query loads the map \u{2248}30 s)",
                                )
                                .size(10.0)
                                .italics()
                                .color(MUTED),
                            );
                        });

                    // ---- Graphics (experimental): live toggles for the render features. ----
                    // Edits go through a local copy so change-detection only fires on a real
                    // tweak (a bare &mut through ResMut would mark the resource changed every
                    // frame the sliders render).
                    CollapsingHeader::new(
                        RichText::new("Graphics (experimental)").color(HDR).size(12.0).strong(),
                    )
                    .id_salt("sec_gfx")
                    .default_open(false)
                    .show(ui, |ui| {
                        let mut g = gfx_ui.gfx.clone();
                        ui.add(egui::Slider::new(&mut g.fog, 0.0..=2.0).text("fog"));
                        ui.add(egui::Slider::new(&mut g.sky_refl, 0.0..=2.0).text("sky reflections"));
                        ui.add(egui::Slider::new(&mut g.emissive, 0.0..=3.0).text("emissive"));
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
                        ui.add_enabled_ui(g.shadows_available, |ui| {
                            ui.checkbox(&mut g.shadows, "sun shadows")
                                .on_hover_text("real-time cascades; marginal on the baked-GI look");
                        });
                        ui.checkbox(&mut g.grass, "grass");
                        ui.add(
                            egui::Slider::new(&mut g.cull_px, 0.0..=8.0)
                                .text("prop cull px")
                                .clamping(egui::SliderClamping::Always),
                        );
                        ui.add(
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
    use bevy_egui::egui::{self, Color32, RichText};
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
    // Camera ANGLE from the transform forward (same convention as FlyCam yaw/pitch and
    // apply_camera_command): yaw = atan2(fwd.x, -fwd.z), pitch = asin(fwd.y). Degrees for reading.
    let fwd = *tf.forward();
    let yaw_deg = fwd.x.atan2(-fwd.z).to_degrees();
    let pitch_deg = fwd.y.clamp(-1.0, 1.0).asin().to_degrees();
    let dim = Color32::from_rgb(160, 164, 160);
    let bright = Color32::from_rgb(230, 245, 230);
    let pos_s = format!("{:.1} {:.1} {:.1}", p.x, p.y, p.z);
    let ang_s = format!("{:.1} {:.1}", yaw_deg, pitch_deg);
    // One-line capture of the FULL camera pose (position + look angle) for reproducing a view.
    let capture = format!(
        "pos={:.2},{:.2},{:.2} yaw={:.2} pitch={:.2} fwd={:.3},{:.3},{:.3}",
        p.x, p.y, p.z, yaw_deg, pitch_deg, fwd.x, fwd.y, fwd.z
    );
    egui::Area::new(egui::Id::new("pos_hud"))
        .fixed_pos(egui::pos2(8.0, 36.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_rgba_unmultiplied(0, 0, 0, 153))
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
) {
    use bevy_egui::egui::{self, Color32};
    if menu.is_some() {
        return; // start menu owns the screen
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    const RAIL: Color32 = Color32::from_rgb(16, 16, 15);
    const ACTIVE: Color32 = Color32::from_rgb(199, 178, 153); // beige
    const IDLE: Color32 = Color32::from_rgb(120, 116, 108);
    let cur = *tab;
    egui::SidePanel::right("toolbar")
        .exact_width(40.0)
        .resizable(false)
        .frame(egui::Frame::new().fill(RAIL).inner_margin(egui::Margin::symmetric(4, 8)))
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 6.0;
            // (tab, icon-id) — draw an icon button; returns true on click.
            let mut btn = |ui: &mut egui::Ui, this: RightPanelTab, kind: u8, tip: &str| {
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::click());
                let active = cur == this;
                let col = if active { ACTIVE } else { IDLE };
                if active {
                    ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(34, 33, 30));
                } else if resp.hovered() {
                    ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(26, 25, 23));
                }
                paint_tool_icon(ui.painter(), rect, kind, col);
                resp.on_hover_text(tip).clicked()
            };
            if btn(ui, RightPanelTab::Visibility, 0, "Visibility layers") {
                *tab = RightPanelTab::Visibility;
            }
            if btn(ui, RightPanelTab::Camera, 1, "Camera") {
                *tab = RightPanelTab::Camera;
            }
            if btn(ui, RightPanelTab::Tasks, 2, "Tasks") {
                *tab = RightPanelTab::Tasks;
            }
        });
}

/// Vector icon inside `rect`: 0 = eye, 1 = camera, 2 = tasks/checklist. Painter primitives only
/// (no image assets — keeps the shippable exe free of game-derived art).
#[cfg(feature = "egui")]
fn paint_tool_icon(
    painter: &bevy_egui::egui::Painter,
    rect: bevy_egui::egui::Rect,
    kind: u8,
    c: bevy_egui::egui::Color32,
) {
    use bevy_egui::egui::{self, Color32, Stroke};
    let ctr = rect.center();
    let s = Stroke::new(1.6, c);
    match kind {
        0 => {
            // eye: lens (two arcs approximated by an ellipse outline) + pupil
            let pts: Vec<egui::Pos2> = (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.0 * std::f32::consts::TAU;
                    egui::pos2(ctr.x + t.cos() * 9.0, ctr.y + t.sin() * 5.0)
                })
                .collect();
            painter.add(egui::Shape::closed_line(pts, s));
            painter.circle_filled(ctr, 2.6, c);
        }
        1 => {
            // camera: body + top viewfinder bump + lens
            let body = egui::Rect::from_center_size(ctr, egui::vec2(20.0, 13.0));
            painter.rect_stroke(body, 1.0, s, egui::StrokeKind::Middle);
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(ctr.x - 5.0, body.top() - 3.0), egui::vec2(7.0, 3.0)),
                0.0,
                c,
            );
            painter.circle_stroke(ctr, 4.2, s);
            painter.circle_filled(egui::pos2(body.right() - 2.5, body.top() + 2.5), 1.0, c);
        }
        _ => {
            // tasks: three rows, each a small box + a line (a checklist)
            for r in 0..3 {
                let y = rect.top() + 9.0 + r as f32 * 7.5;
                let bx = egui::Rect::from_min_size(egui::pos2(rect.left() + 5.0, y - 2.5), egui::vec2(5.0, 5.0));
                painter.rect_stroke(bx, 0.0, Stroke::new(1.3, c), egui::StrokeKind::Middle);
                if r == 0 {
                    // a check in the first box
                    painter.line_segment([egui::pos2(bx.left() + 1.0, y), egui::pos2(bx.center().x, bx.bottom() - 1.0)], Stroke::new(1.3, c));
                    painter.line_segment([egui::pos2(bx.center().x, bx.bottom() - 1.0), egui::pos2(bx.right(), bx.top())], Stroke::new(1.3, c));
                }
                painter.line_segment(
                    [egui::pos2(bx.right() + 3.0, y), egui::pos2(rect.right() - 5.0, y)],
                    Stroke::new(1.4, if r == 0 { c } else { Color32::from_rgb(90, 87, 80) }),
                );
            }
        }
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
    use bevy_egui::egui::{self, Color32, RichText};
    if menu.is_some() || *tab != RightPanelTab::Camera {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    const BONE: Color32 = Color32::from_rgb(215, 211, 203);
    const DIM: Color32 = Color32::from_rgb(120, 116, 108);
    // Clone-edit-compare so merely rendering the sliders doesn't dirty change detection every
    // frame (only real edits write back — same discipline as the graphics panel).
    let mut fov = cam.fov_deg;
    let mut fly = cam.fly_speed;
    let mut walk = cam.walk_mode;
    let mut expo = gfx.grade_exposure;
    egui::SidePanel::right("map_layers")
        .default_width(300.0)
        .frame(egui::Frame::new().fill(Color32::from_rgb(18, 18, 17)).inner_margin(10.0))
        .show(ctx, |ui| {
            ui.label(RichText::new("CAMERA").color(BONE).size(16.0).strong());
            ui.add_space(8.0);

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
                    "walk locomotion lands next; scroll will scale walk speed + jump height",
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

/// Section header text: name + a dim count of markers in that section.
#[cfg(feature = "egui")]
fn section_hdr(name: &str, count: usize, col: bevy_egui::egui::Color32) -> bevy_egui::egui::RichText {
    use bevy_egui::egui::RichText;
    if count > 0 {
        RichText::new(format!("{name}   \u{00B7} {count}")).color(col).size(12.0).strong()
    } else {
        RichText::new(name).color(col).size(12.0).strong()
    }
}

/// Right-aligned dim marker count for a row.
#[cfg(feature = "egui")]
fn count_tag(ui: &mut bevy_egui::egui::Ui, n: usize, col: bevy_egui::egui::Color32) {
    use bevy_egui::egui::{Align, Layout, RichText};
    if n == 0 {
        return;
    }
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        ui.label(RichText::new(n.to_string()).size(10.0).color(col));
    });
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

/// egui swatch colour for a POI layer (matches the on-map marker colour).
#[cfg(feature = "egui")]
fn poi_swatch(l: crate::poi::PoiLayer) -> bevy_egui::egui::Color32 {
    let (c, _, _) = crate::poi::poi_look(l);
    let s = c.to_srgba();
    bevy_egui::egui::Color32::from_rgb(
        (s.red * 255.0) as u8,
        (s.green * 255.0) as u8,
        (s.blue * 255.0) as u8,
    )
}

/// One POI toggle row: colour swatch + checkbox + a right-aligned dim marker count.
#[cfg(feature = "egui")]
fn poi_row(
    ui: &mut bevy_egui::egui::Ui,
    on: &mut bool,
    label: &str,
    l: crate::poi::PoiLayer,
    counts: &[usize; 16],
) {
    use bevy_egui::egui::{Color32, RichText};
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        ui.label(RichText::new("\u{25CF}").color(poi_swatch(l)).size(12.0));
        ui.checkbox(on, label);
        count_tag(ui, counts[l as usize], Color32::from_rgb(110, 112, 110));
    });
}
