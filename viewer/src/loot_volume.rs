//! Loot-VALUE volume: a 3-D grid over the pack bounds accumulating the ruble value of every loot
//! marker, for the Analysis tab's "where is the money" view.
//!
//! Only the CPU side lives here (grid build + normalisation + the numbers the legend prints). The
//! renderer consumes `LootVolume` separately, so the volume can be inspected and trusted before any
//! pixels depend on it.
//!
//! Value comes from the SAME components the markers draw from — `MarkerValue` is `ev` on containers
//! (loot.rs) and `pr` on loose loot (poi.rs) — so the volume can never disagree with the map about
//! what a spot is worth. Loose loot is a `PoiLayer::LooseLoot` split rather than a new tag, for the
//! same reason.
use bevy::prelude::*;

use crate::poi::{MarkerValue, PoiLayer, SceneInactive};

/// Grid resolution. 8 m lands streets at ~210 x 15 x 250 — the same order as the SH irradiance grid
/// (216 x 14 x 256) that already ships in every pack, so the memory profile is known-good.
pub const DEFAULT_CELL_M: f32 = 8.0;

#[derive(Resource)]
pub struct LootVolumeSettings {
    /// Draw the volume (and fade the world geometry behind it).
    pub enabled: bool,
    /// Fold `PoiLayer::LooseLoot` into the totals as well as containers.
    pub include_loose: bool,
    pub cell_m: f32,
    /// World-geometry alpha while the volume is up. 1.0 = untouched.
    pub geometry_alpha: f32,
}

impl Default for LootVolumeSettings {
    fn default() -> Self {
        Self {
            // `EFT_LOOT_VOLUME=1` starts it on, matching EFT_LOD / EFT_SHOW_DISABLED. Without a launch
            // override the grid is only reachable by clicking, so a headless run cannot verify it.
            enabled: std::env::var("EFT_LOOT_VOLUME").map(|v| v.trim() == "1").unwrap_or(false),
            include_loose: std::env::var("EFT_LOOT_VOLUME_LOOSE").map(|v| v.trim() == "1").unwrap_or(false),
            cell_m: DEFAULT_CELL_M,
            geometry_alpha: 0.25,
        }
    }
}

/// The built grid. `cells` is row-major x-major: idx = (z * dim.1 + y) * dim.0 + x.
#[derive(Resource, Default)]
pub struct LootVolume {
    pub dim: (usize, usize, usize),
    pub origin: Vec3,
    pub cell_m: f32,
    pub cells: Vec<f32>,
    /// Normalisation ceiling. The 99th percentile of NON-EMPTY cells, not the max: a couple of
    /// megasafes otherwise saturate one cell and flatten the entire rest of the map to black.
    pub hot: f32,
    pub total_value: f64,
    pub filled_cells: usize,
    pub markers_used: usize,
    /// Bumped on every rebuild so the renderer can tell a stale upload from a current one.
    pub generation: u64,
}

impl LootVolume {
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dim.1 + y) * self.dim.0 + x
    }

    /// Value in the cell containing `p`, or 0 outside the grid.
    pub fn sample_world(&self, p: Vec3) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let l = (p - self.origin) / self.cell_m;
        if l.x < 0.0 || l.y < 0.0 || l.z < 0.0 {
            return 0.0;
        }
        let (x, y, z) = (l.x as usize, l.y as usize, l.z as usize);
        if x >= self.dim.0 || y >= self.dim.1 || z >= self.dim.2 {
            return 0.0;
        }
        self.cells[self.idx(x, y, z)]
    }
}

/// Rebuild when the INPUTS actually change. Deliberately compares values through a `Local` instead
/// of `is_changed()`: egui draws these toggles with `&mut` on a `ResMut`, and `deref_mut` marks the
/// resource changed every frame the panel is open — an `is_changed()` gate here would rebuild the
/// grid continuously (the same livelock that stalled the map load on the disabled-geometry toggle).
/// The marker count is folded in so the grid also refreshes when a map swap replaces the markers.
#[allow(clippy::type_complexity)]
pub fn build_loot_volume(
    settings: Res<LootVolumeSettings>,
    pack: Option<Res<crate::render::LoadedPack>>,
    markers: Query<(
        &GlobalTransform,
        &MarkerValue,
        Option<&PoiLayer>,
        Option<&SceneInactive>,
    )>,
    toggles: Res<crate::ui::LayerToggles>,
    mut vol: ResMut<LootVolume>,
    mut last: Local<Option<(bool, bool, u32, i64, bool)>>,
) {
    let Some(pack) = pack else { return };
    // Key on the SETTINGS only — never on the marker count. poi.rs de-clutters dense markers by
    // distance, so that count churns as the camera moves, and folding it in rebuilt a 5.6M-cell grid
    // (306x61x302 on streets) on those frames. Ticking the box has to build ONCE, then stay put.
    let key = (
        settings.enabled,
        settings.include_loose,
        settings.cell_m.to_bits(),
        toggles.min_value,
        toggles.hide_inactive,
    );
    // ...but a first build that found NOTHING is retried: on the frame the box is ticked the loot
    // markers may not have spawned yet, and a settings-only key would then latch an empty grid
    // forever with no way back short of toggling something.
    let n_markers = markers.iter().len();
    let empty_but_markers_exist = vol.filled_cells == 0 && n_markers > 0;
    if *last == Some(key) && !(settings.enabled && empty_but_markers_exist) {
        return;
    }
    *last = Some(key);

    if !settings.enabled {
        *vol = LootVolume::default();
        return;
    }

    let b = pack.0.manifest.bounds;
    let (lo, hi) = (Vec3::new(b[0], b[1], b[2]), Vec3::new(b[3], b[4], b[5]));
    let cell = settings.cell_m.max(1.0);
    let span = (hi - lo).max(Vec3::splat(cell));
    let dim = (
        (span.x / cell).ceil() as usize + 1,
        (span.y / cell).ceil() as usize + 1,
        (span.z / cell).ceil() as usize + 1,
    );
    let n = dim.0 * dim.1 * dim.2;
    // Sanity ceiling: a pathological cell_m would otherwise try to allocate the map as f32s.
    if n == 0 || n > 64_000_000 {
        warn!("loot volume: {dim:?} = {n} cells is out of range for cell_m={cell} — not building");
        *vol = LootVolume::default();
        return;
    }

    let mut cells = vec![0.0f32; n];
    let (mut used, mut total) = (0usize, 0f64);
    for (gt, val, layer, inactive) in markers.iter() {
        if val.0 <= 0 {
            continue;
        }
        let is_loose = matches!(layer, Some(PoiLayer::LooseLoot));
        if is_loose && !settings.include_loose {
            continue;
        }
        // Agree with what the map is SHOWING: the same min-value floor and inactive filter the
        // markers obey, so a spot the user filtered out cannot still glow in the volume.
        if val.0 < toggles.min_value {
            continue;
        }
        if toggles.hide_inactive && inactive.is_some() {
            continue;
        }
        let p = gt.translation();
        let l = (p - lo) / cell;
        if l.x < 0.0 || l.y < 0.0 || l.z < 0.0 {
            continue;
        }
        let (x, y, z) = (l.x as usize, l.y as usize, l.z as usize);
        if x >= dim.0 || y >= dim.1 || z >= dim.2 {
            continue;
        }
        cells[(z * dim.1 + y) * dim.0 + x] += val.0 as f32;
        used += 1;
        total += val.0 as f64;
    }

    // 99th percentile of NON-EMPTY cells (see `hot`).
    let mut nz: Vec<f32> = cells.iter().copied().filter(|&v| v > 0.0).collect();
    let filled = nz.len();
    nz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let hot = if nz.is_empty() {
        0.0
    } else {
        nz[((nz.len() as f32 * 0.99) as usize).min(nz.len() - 1)]
    };

    let gen = vol.generation.wrapping_add(1);
    *vol = LootVolume {
        dim,
        origin: lo,
        cell_m: cell,
        cells,
        hot,
        total_value: total,
        filled_cells: filled,
        markers_used: used,
        generation: gen,
    };
    info!(
        "loot volume: {}x{}x{} @ {:.0} m — {} markers, {} filled cells, total {:.0}, hot(p99) {:.0}",
        dim.0, dim.1, dim.2, cell, used, filled, total, hot
    );
}

pub struct LootVolumePlugin;

impl Plugin for LootVolumePlugin {
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<LootGizmos>()
            .add_systems(Startup, configure_loot_gizmos)
            .init_resource::<LootVolumeSettings>()
            .init_resource::<LootVolume>()
            .add_systems(Update, (build_loot_volume, draw_loot_volume, draw_loot_outlines).chain())
            // The panel MUST live in EguiPrimaryContextPass, not Update: that is the pass egui lays
            // its areas out in, and it is what makes the panel capture the pointer. Drawn from Update
            // it still rendered, but egui never registered the area, so `is_pointer_over_area()` stayed
            // false and every click went through the checkbox into the scene pick raycast.
            .add_systems(bevy_egui::EguiPrimaryContextPass, analysis_panel);
    }
}

/// The Analysis tab. Owns its own panel (the `insights` module's pattern) rather than adding to
/// `layers_panel`, which is already at the system-param ceiling.
#[allow(clippy::too_many_arguments)]
fn analysis_panel(
    mut contexts: bevy_egui::EguiContexts,
    tab: Res<crate::ui::RightPanelTab>,
    focus: Res<crate::overlay::OverlayFocus>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut settings: ResMut<LootVolumeSettings>,
    vol: Res<LootVolume>,
    toggles: Res<crate::ui::LayerToggles>,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, Color32, RichText};
    if menu.is_some() || focus.0 || *tab != crate::ui::RightPanelTab::Analysis {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    const DIM: Color32 = theme::MUTED;
    egui::SidePanel::right("analysis_panel")
        .default_width(300.0)
        .frame(theme::panel_frame())
        .show(ctx, |ui| {
            ui.label(theme::title("ANALYSIS"));
            ui.label(
                RichText::new(
                    "Loot VALUE as a 3-D volume: every cell is the total worth of the loot inside \
it, so the bright regions are where the money is - not where the items are.",
                )
                .size(10.0)
                .color(DIM),
            );
            ui.add_space(theme::SP_MD);

            ui.checkbox(&mut settings.enabled, "loot value volume")
                .on_hover_text("draw the value grid and fade the world geometry so it reads through walls");
            ui.add_enabled_ui(settings.enabled, |ui| {
                ui.checkbox(&mut settings.include_loose, "include loose loot")
                    .on_hover_text(
                        "fold loose loot into the totals as well as containers. On streets this is \
the difference between ~5M and ~339M of mapped value, so leaving it off measures containers only.",
                    );
                ui.horizontal(|ui| {
                    ui.label(RichText::new("cell").size(11.0).color(DIM));
                    ui.add(egui::Slider::new(&mut settings.cell_m, 2.0..=32.0).suffix(" m"));
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("geometry").size(11.0).color(DIM));
                    ui.add(egui::Slider::new(&mut settings.geometry_alpha, 0.0..=1.0).text("alpha"));
                });
            });

            ui.add_space(theme::SP_MD);
            ui.separator();
            ui.add_space(theme::SP_SM);

            // ---- LEGEND. Reads its endpoints from the SAME `hot` the shader normalises by, so the
            // scale printed here cannot drift from the colours on screen.
            ui.label(RichText::new("VALUE").size(10.0).color(DIM));
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 14.0),
                egui::Sense::hover(),
            );
            let n = 48;
            for i in 0..n {
                let t = i as f32 / (n - 1) as f32;
                let x0 = rect.left() + rect.width() * (i as f32 / n as f32);
                let x1 = rect.left() + rect.width() * ((i + 1) as f32 / n as f32);
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x0, rect.top()),
                        egui::pos2(x1 + 1.0, rect.bottom()),
                    ),
                    0.0,
                    ramp_color(t),
                );
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("0").size(10.0).color(DIM));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("{} +", money(vol.hot as f64)))
                            .size(10.0)
                            .color(DIM),
                    );
                });
            });

            ui.add_space(theme::SP_SM);
            if !settings.enabled {
                ui.label(RichText::new("volume off").size(10.0).color(DIM));
            } else if vol.filled_cells == 0 {
                // Never a silent blank: an empty grid has a cause, and min value is the usual one.
                ui.label(
                    RichText::new(if toggles.min_value > 0 {
                        "no loot passes the current min-value filter"
                    } else {
                        "no loot markers on this map"
                    })
                    .size(10.0)
                    .color(theme::WARN),
                );
            } else {
                for (k, v) in [
                    ("grid", format!("{}x{}x{} @ {:.0} m", vol.dim.0, vol.dim.1, vol.dim.2, vol.cell_m)),
                    ("markers", format!("{}", vol.markers_used)),
                    ("filled cells", format!("{}", vol.filled_cells)),
                    ("mapped value", money(vol.total_value)),
                    ("hottest (p99)", money(vol.hot as f64)),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(k).size(10.0).color(DIM));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(v).size(10.0));
                        });
                    });
                }
            }
        });
}

/// Perceptual-ish cool->hot ramp (blue -> cyan -> green -> amber -> red). Shared by the legend and
/// (later) the volume shader, so the two cannot disagree about what a colour means.
pub fn ramp_rgb(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    const STOPS: [[f32; 3]; 5] = [
        [0.05, 0.10, 0.45],
        [0.05, 0.65, 0.75],
        [0.25, 0.80, 0.30],
        [0.95, 0.70, 0.15],
        [0.90, 0.15, 0.10],
    ];
    let s = t * (STOPS.len() - 1) as f32;
    let i = (s.floor() as usize).min(STOPS.len() - 2);
    let f = s - i as f32;
    let (a, b) = (STOPS[i], STOPS[i + 1]);
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

fn ramp_color(t: f32) -> bevy_egui::egui::Color32 {
    let c = ramp_rgb(t);
    bevy_egui::egui::Color32::from_rgb(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}

/// Compact ruble figure for the legend/readout (1.2M, 43k, 900).
fn money(v: f64) -> String {
    if v >= 1.0e9 {
        format!("{:.1}B", v / 1.0e9)
    } else if v >= 1.0e6 {
        format!("{:.1}M", v / 1.0e6)
    } else if v >= 1.0e3 {
        format!("{:.0}k", v / 1.0e3)
    } else {
        format!("{v:.0}")
    }
}

/// One drawn cell.
#[derive(Component)]
struct LootVoxel;

/// Spawn/refresh the drawn cells. Deliberately ORDINARY Bevy entities (`Mesh3d` + a blended
/// `StandardMaterial`) rather than a bespoke render node: the filled-cell count is ~1k on streets,
/// which is nothing next to the 186k world instances, and this way the volume inherits depth,
/// culling and TAA from the existing pipeline instead of re-deriving them.
///
/// Keyed on `generation`, so it redraws exactly when the grid actually changed — never per frame.
fn draw_loot_volume(
    mut commands: Commands,
    vol: Res<LootVolume>,
    settings: Res<LootVolumeSettings>,
    existing: Query<Entity, With<LootVoxel>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut drawn: Local<u64>,
) {
    let want = if settings.enabled { vol.generation } else { 0 };
    if *drawn == want {
        return;
    }
    *drawn = want;
    for e in existing.iter() {
        commands.entity(e).despawn();
    }
    if !settings.enabled || vol.filled_cells == 0 || vol.hot <= 0.0 {
        return;
    }

    let cube = meshes.add(Cuboid::new(vol.cell_m, vol.cell_m, vol.cell_m));
    let mut n = 0usize;
    for z in 0..vol.dim.2 {
        for y in 0..vol.dim.1 {
            for x in 0..vol.dim.0 {
                let v = vol.cells[vol.idx(x, y, z)];
                if v <= 0.0 {
                    continue;
                }
                // Normalise against the p99 ceiling, then compress: value is heavily long-tailed, so
                // a linear ramp leaves all but the richest cells indistinguishable.
                let t = (v / vol.hot).clamp(0.0, 1.0).powf(0.45);
                let c = ramp_rgb(t);
                // Alpha rides the same curve: cheap cells stay ghostly, rich ones read as solid.
                let a = 0.10 + 0.55 * t;
                let m = mats.add(StandardMaterial {
                    base_color: Color::srgba(c[0], c[1], c[2], a),
                    emissive: LinearRgba::new(c[0] * 2.0, c[1] * 2.0, c[2] * 2.0, 1.0),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    cull_mode: None,
                    ..default()
                });
                let p = vol.origin
                    + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * vol.cell_m;
                commands.spawn((
                    Mesh3d(cube.clone()),
                    MeshMaterial3d(m),
                    Transform::from_translation(p),
                    LootVoxel,
                    Name::new("loot_voxel"),
                ));
                n += 1;
            }
        }
    }
    info!("loot volume: drew {n} cells (gen {})", vol.generation);
}

/// Its own gizmo group so it can carry a NEGATIVE depth bias: that is what makes the volume read
/// THROUGH walls. Fading the world instead would mean pushing all 186k instances into the blend
/// pass (`blend_class` is baked per mesh and drives gpu_cull's opaque/blend reset), which costs a
/// full back-to-front sort. Drawing the volume in front is the same answer from the cheap side, and
/// it is the wireframe reading of the request.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct LootGizmos;

fn configure_loot_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (cfg, _) = store.config_mut::<LootGizmos>();
    cfg.depth_bias = -1.0; // in front of everything
    cfg.line.width = 1.5;
}

/// Wireframe cell outlines, drawn every frame through geometry. Cheap: ~1k cuboids of 12 lines.
fn draw_loot_outlines(
    mut giz: Gizmos<LootGizmos>,
    vol: Res<LootVolume>,
    settings: Res<LootVolumeSettings>,
) {
    if !settings.enabled || vol.hot <= 0.0 || vol.cells.is_empty() {
        return;
    }
    // `geometry_alpha` now means what it can actually deliver: how strongly the volume punches
    // through geometry. 1.0 = full-strength outlines, 0.0 = outlines off (solid cells only).
    let k = settings.geometry_alpha.clamp(0.0, 1.0);
    if k <= 0.001 {
        return;
    }
    let s = Vec3::splat(vol.cell_m);
    for z in 0..vol.dim.2 {
        for y in 0..vol.dim.1 {
            for x in 0..vol.dim.0 {
                let v = vol.cells[vol.idx(x, y, z)];
                if v <= 0.0 {
                    continue;
                }
                let t = (v / vol.hot).clamp(0.0, 1.0).powf(0.45);
                let c = ramp_rgb(t);
                let p = vol.origin
                    + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * vol.cell_m;
                giz.cuboid(
                    Transform::from_translation(p).with_scale(s),
                    Color::srgba(c[0], c[1], c[2], (0.25 + 0.6 * t) * k),
                );
            }
        }
    }
}
