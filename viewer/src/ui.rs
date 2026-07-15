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

pub struct UiPlugin;
impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LayerToggles>()
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
fn layers_panel(mut contexts: bevy_egui::EguiContexts, mut toggles: ResMut<LayerToggles>) {
    use bevy_egui::egui::{self, Color32, RichText};
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
