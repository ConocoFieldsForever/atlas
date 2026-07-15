//! ui.rs — right-hand LAYER-TOGGLE panel (egui).
//!
//! A `SidePanel::right` with checkboxes to show/hide map overlay layers. The LOOT layer is
//! fully wired (master toggle + per-class filters driving `LootClass` marker visibility). The
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

#[derive(Resource)]
pub struct LayerToggles {
    pub loot: bool,
    /// class -> shown. Missing class defaults to shown.
    pub loot_classes: BTreeMap<String, bool>,
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
#[derive(Resource, Default)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct QuestTracker {
    pub active: std::collections::HashSet<String>,
    pub kappa_only: bool,
    pub lk_only: bool,
    /// 0 = no cap.
    pub max_level: u32,
}

pub struct UiPlugin;
impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LayerToggles>()
            .init_resource::<UiSearch>()
            .init_resource::<QuestTracker>()
            .add_systems(Update, apply_loot_visibility);
        // egui UI MUST run in EguiPrimaryContextPass (between egui's begin/end frame); in
        // plain Update the context has no fonts yet and `ctx_mut()` panics (bevy_egui 0.37).
        #[cfg(feature = "egui")]
        app.add_systems(bevy_egui::EguiPrimaryContextPass, layers_panel);
    }
}

/// Show/hide loot markers by the master toggle AND the per-class filter. Only touches the
/// markers when the toggles change (true on the first run too, so the initial state is applied
/// once the markers exist), so it's ~free per frame.
fn apply_loot_visibility(toggles: Res<LayerToggles>, mut q: Query<(&LootClass, &mut Visibility)>) {
    if !toggles.is_changed() {
        return;
    }
    for (cls, mut vis) in &mut q {
        *vis = vis_for(&toggles, &cls.0);
    }
}

fn vis_for(t: &LayerToggles, cls: &str) -> Visibility {
    let shown = t.loot && t.loot_classes.get(cls).copied().unwrap_or(true);
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

#[cfg(feature = "egui")]
#[allow(clippy::too_many_arguments)]
fn layers_panel(
    mut contexts: bevy_egui::EguiContexts,
    mut toggles: ResMut<LayerToggles>,
    mut search: ResMut<UiSearch>,
    mut tracker: ResMut<QuestTracker>,
    quest_data: Res<crate::poi::QuestData>,
    markers: Query<(&crate::inspect::MarkerInfo, &GlobalTransform)>,
    mut cam_cmd: ResMut<crate::CameraCommand>,
    mut route_writer: MessageWriter<crate::pathfind::RouteRequest>,
    route_result: Res<crate::pathfind::RouteResult>,
) {
    use bevy_egui::egui::{self, Color32, RichText};
    use crate::pathfind::{RouteRequest, RouteStatus};
    use crate::poi::PoiLayer;
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    const ACCENT: Color32 = Color32::from_rgb(232, 194, 122); // warm tactical amber
    const HDR: Color32 = Color32::from_rgb(150, 154, 150);
    const MUTED: Color32 = Color32::from_rgb(120, 122, 120);
    const PANEL_BG: Color32 = Color32::from_rgb(20, 22, 23);

    // Style ONLY this panel's frame (a global ctx.set_style() was painting a fullscreen white
    // layer over the 3D scene). Per-widget RichText below carries the rest of the look.
    let frame = egui::Frame::side_top_panel(&ctx.style())
        .fill(PANEL_BG)
        .inner_margin(egui::Margin::same(14));
    egui::SidePanel::right("map_layers")
        .resizable(false)
        .frame(frame)
        .default_width(232.0)
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 7.0);
            ui.add_space(4.0);
            ui.label(RichText::new("MAP  LAYERS").color(ACCENT).size(17.0).strong());
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);

            // ---- Marker search (finds any marker by name -> fly the camera to it) ----
            ui.add(
                egui::TextEdit::singleline(&mut search.query)
                    .desired_width(f32::INFINITY)
                    .hint_text("Search markers\u{2026}"),
            );
            let q = search.query.trim().to_lowercase();
            if !q.is_empty() {
                // Case-insensitive substring over title (and subtitle) of every marker.
                let mut hits: Vec<(&crate::inspect::MarkerInfo, Vec3)> = Vec::new();
                for (info, gt) in &markers {
                    if info.title.to_lowercase().contains(&q)
                        || info.subtitle.to_lowercase().contains(&q)
                    {
                        hits.push((info, gt.translation()));
                    }
                }
                let total = hits.len();
                ui.add_space(2.0);
                ui.label(RichText::new(format!("{total} results")).size(10.0).color(MUTED));
                egui::ScrollArea::vertical()
                    .id_salt("marker_search")
                    .max_height(220.0)
                    .show(ui, |ui| {
                        for (info, pos) in hits.iter().take(25) {
                            let label = RichText::new(format!(
                                "{}  \u{00B7}  {}",
                                info.title, info.subtitle
                            ));
                            if ui.selectable_label(false, label).clicked() {
                                cam_cmd.fly_to = Some(*pos);
                            }
                        }
                        if total > 25 {
                            ui.label(
                                RichText::new(format!("\u{2026} +{} more", total - 25))
                                    .size(10.0)
                                    .color(MUTED),
                            );
                        }
                    });
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(2.0);
            }

            // ---- Loot layer (functional) ----
            ui.checkbox(&mut toggles.loot, RichText::new("Raw loot").size(15.0).strong());
            let loot_on = toggles.loot;
            ui.add_space(2.0);
            for (cls, on) in toggles.loot_classes.iter_mut() {
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    let swatch = if loot_on { class_color(cls) } else { Color32::from_gray(70) };
                    ui.label(RichText::new("\u{25CF}").color(swatch).size(12.0)); // ●
                    ui.add_enabled_ui(loot_on, |ui| {
                        ui.checkbox(on, titlecase(cls));
                    });
                });
            }

            ui.add_space(12.0);
            ui.label(RichText::new("SPAWNS  &  POIS").color(HDR).size(11.0).strong());
            ui.add_space(2.0);
            ui.separator();
            ui.add_space(2.0);
            poi_row(ui, &mut toggles.pmc_spawns, "PMC spawns", PoiLayer::PmcSpawn);
            poi_row(ui, &mut toggles.scav_spawns, "Scav spawns", PoiLayer::ScavSpawn);
            poi_row(ui, &mut toggles.bosses, "Bosses", PoiLayer::Boss);
            poi_row(ui, &mut toggles.extracts, "Extracts", PoiLayer::Extract);
            poi_row(ui, &mut toggles.doors, "Doors", PoiLayer::Door);
            poi_row(ui, &mut toggles.interactables, "Interactables", PoiLayer::Interactable);
            ui.add_space(8.0);
            ui.label(
                RichText::new("PMC/scav/boss: tarkov.dev  \u{2022}  extracts/doors: game data")
                    .size(10.0)
                    .italics()
                    .color(MUTED),
            );

            ui.add_space(12.0);
            ui.label(RichText::new("MAP  INTEL").color(HDR).size(11.0).strong());
            ui.add_space(2.0);
            ui.separator();
            ui.add_space(2.0);
            poi_row(ui, &mut toggles.locks, "Locks & keys", PoiLayer::Lock);
            poi_row(ui, &mut toggles.hazards, "Hazards", PoiLayer::Hazard);
            poi_row(ui, &mut toggles.switches, "Switches", PoiLayer::Switch);
            poi_row(ui, &mut toggles.transits, "Transits", PoiLayer::Transit);
            poi_row(ui, &mut toggles.stationary, "Stationary guns", PoiLayer::Stationary);
            poi_row(ui, &mut toggles.loose, "Loose loot", PoiLayer::LooseLoot);
            poi_row(ui, &mut toggles.quests, "Tasks / quests", PoiLayer::Quest);

            // ---- TASK TRACKER (checklist + filters + on-demand route) ----
            // The `quests` poi_row above stays the master on/off; this section refines WHICH tasks
            // are shown/routed. Selecting tasks focuses the quest markers to them (poi.rs) and
            // draws their objective-zone outlines.
            ui.add_space(12.0);
            ui.label(RichText::new("TASKS  /  QUESTS").color(HDR).size(11.0).strong());
            ui.add_space(2.0);
            ui.separator();
            ui.add_space(2.0);

            // Filter row: Kappa / Lightkeeper toggles + a max-level cap (0 = any).
            ui.horizontal(|ui| {
                ui.checkbox(&mut tracker.kappa_only, "Kappa");
                ui.checkbox(&mut tracker.lk_only, "Lightkeeper");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut tracker.max_level).range(0..=79));
                ui.label(RichText::new("\u{2264} Lvl").size(12.0).color(MUTED));
            });

            // This map's tasks passing the filters (borrows QuestData; filter reads copied out so
            // the checklist can still mutate `tracker.active` below).
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
            ui.add_space(4.0);
            ui.label(RichText::new(format!("({} tasks)", shown.len())).size(10.0).color(MUTED));
            egui::ScrollArea::vertical()
                .id_salt("quest_list")
                .max_height(260.0)
                .show(ui, |ui| {
                    for t in &shown {
                        ui.horizontal(|ui| {
                            // Checkbox synced to the tracker's active set (temp bool + apply-on-change).
                            let mut on = tracker.active.contains(&t.id);
                            if ui.checkbox(&mut on, "").changed() {
                                if on {
                                    tracker.active.insert(t.id.clone());
                                } else {
                                    tracker.active.remove(&t.id);
                                }
                            }
                            // Click the name to fly to the first objective's first zone.
                            let name = if t.name.is_empty() { "Task" } else { t.name.as_str() };
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
                        // Dim "{trader} \u{00B7} Lvl {min} \u{00B7} Kappa" tag line.
                        let mut tags =
                            if t.trader.is_empty() { String::new() } else { t.trader.clone() };
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

            // Route buttons: chain through every active task's objectives' first zones (server
            // optimizes the order); an empty request clears the polyline.
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Route active").clicked() {
                    let dests: Vec<Vec3> = quest_data
                        .tasks
                        .iter()
                        .filter(|t| tracker.active.contains(&t.id))
                        .flat_map(|t| {
                            t.objectives.iter().filter_map(|o| o.zones.first().map(|z| z.pos))
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
            // Route status (from pathfind.rs) — a single dim line under the buttons.
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

/// One POI toggle row: colour swatch + checkbox.
#[cfg(feature = "egui")]
fn poi_row(
    ui: &mut bevy_egui::egui::Ui,
    on: &mut bool,
    label: &str,
    l: crate::poi::PoiLayer,
) {
    use bevy_egui::egui::RichText;
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        ui.label(RichText::new("\u{25CF}").color(poi_swatch(l)).size(12.0));
        ui.checkbox(on, label);
    });
}
