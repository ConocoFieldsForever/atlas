//! eft::menu_fx — start-menu decor + custom widgets.
//!
//! * [`eft_loading_bar`] — game-loader-style segmented progress bar for the build panel:
//!   khaki segments with 1px edges, pulsing frontier segment, stage text + percent upper-left,
//!   ESTIMATED TIME mm:ss upper-right, small midpoint tick.
//! * [`security_camera`] — vector-drawn wall CCTV (egui painter only) in the top-right that
//!   servo-tracks the mouse cursor (yaw/pitch clamped cone, exponential slew), blinking red
//!   LED, idle patrol sweep when the pointer leaves the window. This is the SHIPPED look and
//!   the permanent fallback.
//! * [`spawn_menu_prop`] / [`menu_prop_update`] — the REAL game CCTV (Street_Camera_01 from
//!   lighthouse) as a 3D cursor-tracking prop, when a MACHINE-LOCAL extraction exists in
//!   packs/shared/menu/ (camera.bin + camera.png, written by tools/extract_menu_prop.py —
//!   packs/ is gitignored, the asset is never committed/shipped). Menu mode only; when the
//!   files are absent/corrupt the [`MenuCamProp`] resource is never inserted and menu.rs
//!   keeps painting the vector camera instead (exactly one of the two ever renders).
//!
//! Palette mirrors menu.rs (near-black field / charcoal panels / bone text) but runs a step
//! dimmer so the decor stays background. ASCII-only labels (glyph whitelist).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, PrimitiveTopology, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;
use bevy_egui::egui::{self, Color32, Rect, Shape, Stroke, StrokeKind, pos2, vec2};

// ---- shared UI palette: sourced from the single source of truth (ui_theme). The wall-CCTV
// prop's own material colours (BODY/DARK/SEAM/METAL, inside `security_camera`) stay local — they
// are a physical object's finish, not UI design tokens. ----
use crate::ui_theme as theme;
const KHAKI: Color32 = theme::BEIGE; // filled loader segment (tactical beige = PLAY / active tab)
const RED: Color32 = theme::DANGER; // failed-loader recolour / camera LED
const TXT: Color32 = theme::BONE;
const DIM: Color32 = theme::MUTED;

fn scale_rgb(c: Color32, f: f32) -> Color32 {
    Color32::from_rgb(
        (c.r() as f32 * f) as u8,
        (c.g() as f32 * f) as u8,
        (c.b() as f32 * f) as u8,
    )
}

/// EFT-loader-style segmented progress bar.
///
/// * `frac` in [0,1] — stage-derived progress; fully filled segments are khaki, the frontier
///   segment pulses (~1.5 Hz) while work is running, the rest stay dark charcoal.
/// * `stage_text` — "LOADING OBJECTS..." style line, drawn upper-left with the percent under it.
/// * `elapsed_secs` — build wall time; drives the ESTIMATED TIME readout upper-right
///   (naive `elapsed/frac - elapsed`, shown as "--:--" until `frac` is meaningful).
/// * `failed` — recolors filled segments the menu red and freezes the pulse.
pub fn eft_loading_bar(
    ui: &mut egui::Ui,
    frac: f32,
    stage_text: &str,
    elapsed_secs: f32,
    failed: bool,
) {
    use egui::RichText;
    const SEGS: usize = 12; // the game loader reads as ~10-14 wide segments
    const GAP: f32 = 2.0;
    const BAR_H: f32 = 18.0;
    const EMPTY: Color32 = theme::INSET; // unfilled segment well
    const EDGE: Color32 = theme::BORDER; // segment edge
    const FRAME: Color32 = theme::BORDER_STRONG; // bar frame

    let frac = frac.clamp(0.0, 1.0);
    let eta = if failed || frac < 0.05 || elapsed_secs < 2.0 {
        "--:--".to_string() // no sane estimate yet (or never will be)
    } else if frac >= 1.0 {
        "00:00".to_string()
    } else {
        let est = (elapsed_secs / frac - elapsed_secs).clamp(0.0, 99.0 * 60.0 + 59.0);
        format!("{:02}:{:02}", (est / 60.0) as u32, (est % 60.0) as u32)
    };

    // Header row: stage + percent upper-left, ESTIMATED TIME + mm:ss upper-right.
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(
                RichText::new(stage_text)
                    .color(if failed { RED } else { TXT })
                    .size(12.0)
                    .strong(),
            );
            ui.label(RichText::new(format!("{:.0}%", frac * 100.0)).color(DIM).size(11.0));
        });
        ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
            ui.label(RichText::new("ESTIMATED TIME").color(DIM).size(10.0));
            ui.label(RichText::new(eta).color(TXT).size(12.0).strong());
        });
    });

    ui.add_space(6.0); // headroom for the midpoint tick above the frame
    let (rect, _) =
        ui.allocate_exact_size(vec2(ui.available_width(), BAR_H), egui::Sense::hover());
    let p = ui.painter();
    p.rect_stroke(rect, 0.0, Stroke::new(1.0, FRAME), StrokeKind::Inside);
    // Midpoint tick (the game loader's 50% marker) — a small notch above the frame. With an
    // even segment count the center also lands exactly on a gap, which reads as a divider.
    let cx = rect.center().x.floor() + 0.5;
    p.line_segment(
        [pos2(cx, rect.top() - 5.0), pos2(cx, rect.top() - 1.0)],
        Stroke::new(1.0, DIM),
    );

    let inner = rect.shrink(2.0);
    let seg_w = (inner.width() - GAP * (SEGS as f32 - 1.0)) / SEGS as f32;
    if seg_w < 2.0 {
        return; // panel squeezed to nothing — keep the frame, skip segments
    }
    let t = ui.input(|i| i.time) as f32;
    let frontier = (frac * SEGS as f32).floor() as usize; // == SEGS when frac hits 1.0
    let base = if failed { RED } else { KHAKI };
    for s in 0..SEGS {
        let x0 = inner.left() + s as f32 * (seg_w + GAP);
        let r = Rect::from_min_size(pos2(x0, inner.top()), vec2(seg_w, inner.height()));
        if s < frontier {
            p.rect_filled(r, 0.0, base);
        } else if s == frontier && frac < 1.0 {
            // Frontier segment: brightness pulse ~1.5 Hz. The sine is sharpened (^1.6) so it
            // dwells dim and snaps bright — reads mechanical rather than "breathing".
            let pulse = if failed {
                0.5 // static half-bright red stub marks where the build died
            } else {
                let w = 0.5 + 0.5 * (t * 1.5 * std::f32::consts::TAU).sin();
                0.35 + 0.65 * w.powf(1.6)
            };
            p.rect_filled(r, 0.0, scale_rgb(base, pulse));
            p.rect_stroke(r, 0.0, Stroke::new(1.0, EDGE), StrokeKind::Inside);
        } else {
            p.rect_filled(r, 0.0, EMPTY);
            p.rect_stroke(r, 0.0, Stroke::new(1.0, EDGE), StrokeKind::Inside);
        }
    }
}

/// Wall-mounted CCTV camera decor: paints directly with the panel painter, so call it BEFORE
/// the map list and it layers behind everything added later. It creates no widget and senses
/// nothing — pure painter output can never steal pointer input.
///
/// Tracking model (2D fake-3D, screen coords, y down):
/// * pitch = angular offset of the cursor from the rest direction (down-left), clamped to
///   +-25 deg — rotates the whole body in screen space (the mount is side-on to us).
/// * yaw   = lateral position of the cursor across the panel mapped to +-40 deg — shown by
///   foreshortening the body and skewing the front face + lens sideways along the normal.
/// * both angles slew with a frame-rate-independent exponential (servo feel, not glued).
/// * pointer outside the window -> slow idle patrol sweep instead.
///
/// This vector body is what SHIPS; when a machine-local packs/shared/menu extraction exists,
/// menu.rs skips this call and the real 3D prop ([`spawn_menu_prop`]) renders instead.
/// Menu backdrop: a slowly spinning, glowing wireframe globe rendered "just out of focus" (each
/// edge drawn as a wide soft halo + a softer core — no crisp 1px line — so it reads as a defocused
/// hologram). Latitude circles + longitude meridians of a unit sphere, tilted, orthographically
/// projected; back-facing arcs are dimmed so it reads as a sphere. The cursor influences the spin:
/// its horizontal offset from the globe centre sets a target angular velocity (drag it faster /
/// reverse it), with inertia; its vertical offset nudges the tilt. Idle = gentle constant spin.
/// Painter-only (no assets), the same idiom as `security_camera` which it replaces.
pub fn wireframe_globe(ui: &egui::Ui, panel: Rect) {
    use std::f32::consts::{PI, TAU};
    let ctx = ui.ctx();
    let t_now = ctx.input(|i| i.time) as f32;
    let dt = ctx.input(|i| i.stable_dt).min(0.1);
    let _ = t_now;

    // Backdrop placement: centred in the window, sitting behind the menu content.
    let center = panel.center();
    let radius = panel.height().min(panel.width()) * 0.40;

    // ---- spin + tilt state (persist across frames): angle phi, angular velocity omega, tilt ----
    let id = egui::Id::new("menu_fx_globe");
    let (mut phi, mut omega, mut tilt) = ctx
        .data(|d| d.get_temp::<(f32, f32, f32)>(id))
        .unwrap_or((0.0, 0.30, 0.42));
    const BASE_SPIN: f32 = 0.30; // idle rad/s
    let (target_omega, target_tilt) = match ctx.input(|i| i.pointer.hover_pos()) {
        Some(m) => {
            let nx = ((m.x - center.x) / (panel.width() * 0.5)).clamp(-1.4, 1.4);
            let ny = ((m.y - center.y) / (panel.height() * 0.5)).clamp(-1.0, 1.0);
            (BASE_SPIN + nx * 1.7, 0.42 + ny * 0.28) // cursor drags the spin + tips the axis
        }
        None => (BASE_SPIN, 0.42),
    };
    omega += (target_omega - omega) * (1.0 - (-2.5 * dt).exp());
    tilt += (target_tilt - tilt) * (1.0 - (-3.0 * dt).exp());
    phi += omega * dt;
    ctx.data_mut(|d| d.insert_temp(id, (phi, omega, tilt)));
    let (st, ct) = tilt.sin_cos();

    // Project a unit-sphere point (Y-spin already folded into longitudes) to screen + depth.
    // z after tilt: +toward viewer .. -away; drives the front/back brightness so it reads 3D.
    let proj = |x: f32, y: f32, z: f32| -> (egui::Pos2, f32) {
        let y2 = y * ct - z * st;
        let z2 = y * st + z * ct;
        (pos2(center.x + x * radius, center.y - y2 * radius), z2)
    };

    let painter = ui.painter();

    // Soft neon ATMOSPHERE behind the globe (the "blur"): concentric translucent discs building a
    // radial haze, brightest near the surface and fading out — a cheap bloom/defocus glow.
    for i in 0..9 {
        let f = i as f32 / 8.0;
        let r = radius * (1.18 - f * 0.5); // large -> small
        let a = 0.045 * f; // fainter outside, denser toward the core
        painter.circle_filled(
            center,
            r,
            Color32::from_rgba_unmultiplied(30, 140, 170, (a * 255.0) as u8),
        );
    }

    // Glowing neon polyline: wide soft HALO + mid glow + bright CORE, alpha-scaled by depth so the
    // back of the sphere fades. Holographic teal-cyan.
    let glow = |pts: &[(egui::Pos2, f32)]| {
        for w in pts.windows(2) {
            let (a, za) = w[0];
            let (b, zb) = w[1];
            let depth = (((za + zb) * 0.5) + 1.0) * 0.5; // 0 back .. 1 front
            let k = 0.22 + 0.78 * depth;
            let col = |al: f32| {
                Color32::from_rgba_unmultiplied(130, 240, 252, (al * 255.0).clamp(0.0, 255.0) as u8)
            };
            painter.line_segment([a, b], Stroke::new(7.0, col(k * 0.09))); // wide out-of-focus halo
            painter.line_segment([a, b], Stroke::new(3.2, col(k * 0.26))); // mid glow
            painter.line_segment([a, b], Stroke::new(1.4, col(k * 0.85))); // bright neon core
        }
    };

    // Latitude circles.
    for band in 1..6 {
        let lat = -PI / 2.0 + PI * (band as f32 / 6.0);
        let (sl, cl) = lat.sin_cos();
        let pts: Vec<(egui::Pos2, f32)> = (0..=64)
            .map(|i| {
                let th = i as f32 / 64.0 * TAU + phi;
                proj(cl * th.cos(), sl, cl * th.sin())
            })
            .collect();
        glow(&pts);
    }
    // Meridians as FULL great circles (each pole-to-pole circle = 2 apparent vertical lines): 6
    // great circles -> 12 evenly-spaced verticals with NO gap (the old half-arcs only covered ~150
    // deg of longitude, so half the vertical lines were missing).
    for mrd in 0..6 {
        let lon = mrd as f32 / 6.0 * PI + phi;
        let (sn, cs) = lon.sin_cos();
        let pts: Vec<(egui::Pos2, f32)> = (0..=64)
            .map(|i| {
                let ang = i as f32 / 64.0 * TAU; // sweep the whole great circle (front + back halves)
                let (sa, ca) = ang.sin_cos();
                proj(ca * cs, sa, ca * sn)
            })
            .collect();
        glow(&pts);
    }
    // Poles: a bright node so the spin axis reads.
    for pole in [-1.0f32, 1.0] {
        let (p, z) = proj(0.0, pole, 0.0);
        let d = ((z + 1.0) * 0.5).clamp(0.0, 1.0);
        painter.circle_filled(p, 2.6, Color32::from_rgba_unmultiplied(190, 248, 255, (d * 200.0) as u8));
    }
}

// ============================================================================================
// NEON 3D GLOBE — the real "shader" backdrop: a UV sphere whose only visible surface is a lat/long
// grid, drawn as bright HDR emissive with ADDITIVE blend, so the menu camera's Bloom halos it into
// a glowing neon wireframe (true glow/blur, unlike the egui painter). A real 3D object => it spins,
// the cursor steers it, and it sits behind the (transparent) menu UI. Menu mode only.
// ============================================================================================

/// Procedural lat/long grid MASK: bright on grid lines (soft edges), black between. Multiplied by
/// the material's HDR emissive so only the lines glow.
fn globe_grid_texture() -> Image {
    const W: usize = 1024;
    const H: usize = 512;
    const NLON: usize = 18; // meridians (vertical)
    const NLAT: usize = 9; // parallels (horizontal)
    // distance-to-nearest-line -> GAUSSIAN soft 0..1 intensity: fuzzy, wide-falloff lines so the
    // globe reads soft/blurred (volumetric) rather than as crisp wireframe edges.
    let line = |frac: f32, n: usize| -> f32 {
        let c = frac * n as f32;
        let d = (c - c.round()).abs();
        (-(d * d) / (2.0 * 0.055 * 0.055)).exp()
    };
    let mut data = vec![0u8; W * H * 4];
    for y in 0..H {
        let v = (y as f32 + 0.5) / H as f32;
        let lat_i = line(v, NLAT);
        for x in 0..W {
            let u = (x as f32 + 0.5) / W as f32;
            let b = (lat_i.max(line(u, NLON)) * 255.0) as u8;
            let o = (y * W + x) * 4;
            data[o] = b;
            data[o + 1] = b;
            data[o + 2] = b;
            data[o + 3] = b;
        }
    }
    Image::new(
        Extent3d { width: W as u32, height: H as u32, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Spin state carried on the globe entity (angle + angular velocity). The spin AXIS is a fixed
/// tilted pole (below) — the cursor only changes the speed, so the axis never wobbles.
#[derive(Component)]
pub struct MenuGlobe {
    phi: f32,
    omega: f32,
}

/// The globe's polar spin axis: mostly vertical, tilted slightly toward the camera + a touch to the
/// side, so meridians sweep side-to-side like a real spinning globe (not a face-on wheel).
const GLOBE_AXIS: Vec3 = Vec3::new(0.13, 1.0, 0.22);

/// Spawn the neon globe at the menu camera's look target (origin) — big enough to fill the backdrop.
pub fn spawn_menu_globe(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let grid = images.add(globe_grid_texture());
    // ALPHA-BLEND (not additive): the grid texture drives BOTH the emissive glow AND the alpha, so
    // between-line texels are transparent (see-through globe) and the lines DON'T additively stack
    // where front+back overlap (that stacking blew the edge-on latitudes out to white). NOT `unlit`
    // — Bevy's unlit path skips emissive; lit + black base + HDR emissive = glowing lines only.
    let mat = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        base_color_texture: Some(grid.clone()), // alpha = line mask -> transparent between lines
        emissive: LinearRgba::rgb(0.1, 1.8, 2.5), // HDR cyan; softer since lines are wide + Bloom hot
        emissive_texture: Some(grid),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        double_sided: true,
        ..default()
    });
    let mesh = meshes.add(Sphere::new(18.0).mesh().uv(60, 30));
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(mat),
        Transform::from_xyz(0.0, 0.0, 0.0),
        MenuGlobe { phi: 0.0, omega: 0.09 },
        Name::new("menu_neon_globe"),
    ));

    // Faint inner glow ORB: a translucent emissive sphere filling the wireframe so the globe reads as
    // a glowing VOLUME, not just lines. The menu camera's (boosted) Bloom hazes it into a soft
    // volumetric core; grazing angles at the silhouette make it denser at the rim. Default cull_mode
    // (Back) shows only the front hemisphere.
    let core = materials.add(StandardMaterial {
        base_color: Color::srgba(0.03, 0.26, 0.34, 0.17),
        emissive: LinearRgba::rgb(0.03, 0.75, 1.1),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(15.0).mesh().uv(32, 16))),
        MeshMaterial3d(core),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Name::new("menu_globe_core"),
    ));
    info!("menu: spawned neon 3D globe");
}

/// Spin the globe around its FIXED tilted pole; the cursor's horizontal offset only changes the
/// spin SPEED (drag it faster / reverse it), with inertia — so the axis stays put.
pub fn menu_globe_update(
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut q: Query<(&mut Transform, &mut MenuGlobe)>,
) {
    let dt = time.delta_secs().min(0.1);
    let cursor_nx = windows
        .single()
        .ok()
        .and_then(|w| w.cursor_position().map(|c| (c.x / w.width() - 0.5) * 2.0));
    let axis = GLOBE_AXIS.normalize();
    for (mut tf, mut g) in &mut q {
        let target = 0.09 + cursor_nx.unwrap_or(0.0) * 0.5; // much slower idle; gentle cursor drag
        g.omega += (target - g.omega) * (1.0 - (-2.5 * dt).exp());
        g.phi += g.omega * dt;
        tf.rotation = Quat::from_axis_angle(axis, g.phi);
    }
}

// ============================================================================================
// NEON LOW-POLY TRIANGLE TERRAIN — the interactive menu backdrop. A triangulated grid (row/col +
// diagonal edges = triangles) in the 3D world, HDR-emissive so the camera's Bloom halos it into a
// glowing neon wireframe. Each frame the vertex heights are recomputed: gentle idle swells + a
// radial RIPPLE emanating from the cursor's point on the ground (a real camera->plane raycast), so
// moving the mouse pushes waves across the terrain. Evokes a Tarkov map surface.
// ============================================================================================

const TN: usize = 56; // grid is TN x TN vertices
const TCELL: f32 = 2.5; // cell size (world units)

#[derive(Component)]
pub struct MenuTerrain {
    mesh: Handle<Mesh>,
}

pub fn spawn_menu_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let half = (TN as f32 - 1.0) * TCELL * 0.5;
    let mut pos = Vec::with_capacity(TN * TN);
    for j in 0..TN {
        for i in 0..TN {
            pos.push([i as f32 * TCELL - half, 0.0, j as f32 * TCELL - half]);
        }
    }
    // Indexed LineList: row + column + one diagonal per cell (the diagonal splits each quad into two
    // triangles). Indices are fixed; only the vertex heights change per frame.
    let vid = |i: usize, j: usize| (j * TN + i) as u32;
    let mut indices: Vec<u32> = Vec::new();
    for j in 0..TN {
        for i in 0..TN {
            if i + 1 < TN {
                indices.extend([vid(i, j), vid(i + 1, j)]);
            }
            if j + 1 < TN {
                indices.extend([vid(i, j), vid(i, j + 1)]);
            }
            if i + 1 < TN && j + 1 < TN {
                indices.extend([vid(i, j), vid(i + 1, j + 1)]);
            }
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_indices(Indices::U32(indices));
    let handle = meshes.add(mesh);

    let mat = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(0.08, 1.7, 2.4), // HDR cyan -> Bloom neon
        alpha_mode: AlphaMode::Add, // additive glowing lines (single layer -> no blowout)
        ..default()
    });
    commands.spawn((
        Mesh3d(handle.clone()),
        MeshMaterial3d(mat),
        Transform::from_xyz(0.0, 0.0, 0.0),
        MenuTerrain { mesh: handle },
        Name::new("menu_terrain"),
    ));
    info!("menu: spawned neon triangle terrain");
}

pub fn menu_terrain_update(
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam: Query<(&GlobalTransform, &Camera), With<crate::render::CullCamera>>,
    q: Query<&MenuTerrain>,
) {
    let t = time.elapsed_secs();
    // Cursor -> point on the ground plane (y=0), via a camera->cursor raycast. None when off-screen.
    let cursor_pt = (|| {
        let w = windows.single().ok()?;
        let cpos = w.cursor_position()?;
        let (gt, camera) = cam.single().ok()?;
        let ray = camera.viewport_to_world(gt, cpos).ok()?;
        let dy = ray.direction.y;
        if dy.abs() < 1e-4 {
            return None;
        }
        let dist = -ray.origin.y / dy;
        (dist > 0.0).then(|| {
            let p = ray.get_point(dist);
            Vec2::new(p.x, p.z)
        })
    })();
    let half = (TN as f32 - 1.0) * TCELL * 0.5;
    for terrain in &q {
        let Some(mesh) = meshes.get_mut(&terrain.mesh) else {
            continue;
        };
        let mut pos = Vec::with_capacity(TN * TN);
        for j in 0..TN {
            for i in 0..TN {
                let x = i as f32 * TCELL - half;
                let z = j as f32 * TCELL - half;
                // gentle idle swells
                let mut y = 1.5 * ((x * 0.05 + t * 0.25).sin() + (z * 0.06 - t * 0.2).cos());
                // radial ripple from the cursor's ground point (decays with distance)
                if let Some(c) = cursor_pt {
                    let d = ((x - c.x) * (x - c.x) + (z - c.y) * (z - c.y)).sqrt();
                    y += 5.5 * (d * 0.17 - t * 3.2).sin() * (-d * 0.045).exp();
                }
                pos.push([x, y, z]);
            }
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    }
}

// ============================================================================================
// INTERCHANGE-INSPIRED NEON WIREFRAME EXFIL — the menu backdrop. A stylized, DERIVATIVE low-poly
// schematic of the railway / elevated-overpass corner of Interchange (the razor-wire highway
// overpass on its pillars, the rail line with boxcars running out from under it, power-transmission
// pylons receding into the distance, stacked shipping containers, a gantry crane, and a few
// conifers) — the vantage an in-game camera pose is derived from, but hand-built entirely from
// primitives so it carries NO game geometry and ships safely with the app. Rendered as HDR-emissive
// lines so the camera Bloom halos it into a glowing hologram; a gentle idle drift + cursor parallax
// keep it alive.
// ============================================================================================

#[derive(Component)]
pub struct MenuScene {
    yaw: f32,
    tilt: f32,
}

fn wire_edge(v: &mut Vec<[f32; 3]>, a: Vec3, b: Vec3) {
    v.push([a.x, a.y, a.z]);
    v.push([b.x, b.y, b.z]);
}

/// 12 edges of an axis-aligned box, appended as LineList vertex pairs.
fn wire_box(v: &mut Vec<[f32; 3]>, lo: Vec3, hi: Vec3) {
    let c = [
        Vec3::new(lo.x, lo.y, lo.z),
        Vec3::new(hi.x, lo.y, lo.z),
        Vec3::new(hi.x, lo.y, hi.z),
        Vec3::new(lo.x, lo.y, hi.z),
        Vec3::new(lo.x, hi.y, lo.z),
        Vec3::new(hi.x, hi.y, lo.z),
        Vec3::new(hi.x, hi.y, hi.z),
        Vec3::new(lo.x, hi.y, hi.z),
    ];
    for &(a, b) in &[
        (0, 1), (1, 2), (2, 3), (3, 0), // floor ring
        (4, 5), (5, 6), (6, 7), (7, 4), // roof ring
        (0, 4), (1, 5), (2, 6), (3, 7), // uprights
    ] {
        wire_edge(v, c[a], c[b]);
    }
}

/// A flat XZ grid of lines at height `y`.
fn wire_grid(v: &mut Vec<[f32; 3]>, x0: f32, x1: f32, z0: f32, z1: f32, y: f32, nx: usize, nz: usize) {
    for i in 0..=nx {
        let x = x0 + (x1 - x0) * i as f32 / nx as f32;
        wire_edge(v, Vec3::new(x, y, z0), Vec3::new(x, y, z1));
    }
    for j in 0..=nz {
        let z = z0 + (z1 - z0) * j as f32 / nz as f32;
        wire_edge(v, Vec3::new(x0, y, z), Vec3::new(x1, y, z));
    }
}

/// A run of razor-wire concertina: overlapping vertical circles (X-Y plane) marching along X.
fn wire_coils(v: &mut Vec<[f32; 3]>, x0: f32, x1: f32, y_top: f32, z: f32, r: f32, count: usize) {
    let seg = 10;
    let n = count.max(2);
    for c in 0..n {
        let cx = x0 + (x1 - x0) * c as f32 / (n as f32 - 1.0);
        let cy = y_top + r;
        for k in 0..seg {
            let a0 = k as f32 / seg as f32 * std::f32::consts::TAU;
            let a1 = (k + 1) as f32 / seg as f32 * std::f32::consts::TAU;
            wire_edge(
                v,
                Vec3::new(cx + r * a0.cos(), cy + r * a0.sin(), z),
                Vec3::new(cx + r * a1.cos(), cy + r * a1.sin(), z),
            );
        }
    }
}

/// A simplified lattice power-transmission pylon at (cx,cz): 4 tapering legs, horizontal belts, and
/// two cross-arms near the top. Returns the arm height so wires can hang from it.
fn wire_pylon(v: &mut Vec<[f32; 3]>, cx: f32, cz: f32, base_hw: f32, top_hw: f32, h: f32) -> f32 {
    let corner = |hw: f32, i: usize| {
        let s = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)][i];
        (hw * s.0, hw * s.1)
    };
    for i in 0..4 {
        let (bx, bz) = corner(base_hw, i);
        let (tx, tz) = corner(top_hw, i);
        wire_edge(v, Vec3::new(cx + bx, 0.0, cz + bz), Vec3::new(cx + tx, h, cz + tz));
    }
    for &f in &[0.3f32, 0.6, 0.85] {
        let hw = base_hw + (top_hw - base_hw) * f;
        let y = h * f;
        for i in 0..4 {
            let (ax, az) = corner(hw, i);
            let (bx, bz) = corner(hw, (i + 1) % 4);
            wire_edge(v, Vec3::new(cx + ax, y, cz + az), Vec3::new(cx + bx, y, cz + bz));
        }
    }
    let arm_y = h * 0.82;
    for &(ay, aw) in &[(arm_y, base_hw * 2.4), (h * 0.95, base_hw * 1.7)] {
        wire_edge(v, Vec3::new(cx - aw, ay, cz), Vec3::new(cx + aw, ay, cz));
        wire_edge(v, Vec3::new(cx - aw, ay, cz), Vec3::new(cx - aw, ay + 3.0, cz));
        wire_edge(v, Vec3::new(cx + aw, ay, cz), Vec3::new(cx + aw, ay + 3.0, cz));
    }
    arm_y
}

/// Build the scene as two line sets: (dim ground/pylons/trees, bright structures) for depth.
/// Layout extends into -Z (away from the menu camera at +Z); rails run toward the camera.
fn build_exfil_scene() -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut g = Vec::new();
    let mut s = Vec::new();

    // ground
    wire_grid(&mut g, -220.0, 220.0, -240.0, 110.0, 0.0, 26, 22);

    // --- elevated overpass deck crossing along X, with pillars + razor wire on the near rail ---
    let (dz0, dz1, dy0, dy1) = (-58.0f32, -40.0f32, 22.0f32, 27.0f32);
    wire_box(&mut s, Vec3::new(-200.0, dy0, dz0), Vec3::new(200.0, dy1, dz1));
    wire_grid(&mut s, -200.0, 200.0, dz0 + 3.0, dz1 - 3.0, dy1, 40, 2);
    let mut px = -180.0;
    while px <= 180.0 {
        wire_box(&mut s, Vec3::new(px - 4.0, 0.0, dz0 + 4.0), Vec3::new(px + 4.0, dy0, dz1 - 4.0));
        px += 40.0;
    }
    wire_coils(&mut s, -196.0, 196.0, dy1, dz0 + 1.0, 2.6, 62);

    // --- power pylons receding into the distance + drooping catenary wires ---
    let pylons = [
        (-150.0f32, -90.0f32),
        (-80.0, -120.0),
        (0.0, -150.0),
        (85.0, -185.0),
        (165.0, -220.0),
    ];
    for &(cx, cz) in &pylons {
        wire_pylon(&mut g, cx, cz, 6.0, 1.6, 46.0);
    }
    let arm_y = 46.0 * 0.82;
    for w in pylons.windows(2) {
        let (a, b) = (w[0], w[1]);
        for &off in &[-13.0f32, 0.0, 13.0] {
            let steps = 10;
            for k in 0..steps {
                let t0 = k as f32 / steps as f32;
                let t1 = (k + 1) as f32 / steps as f32;
                let sag = |t: f32| -24.0 * (t * (1.0 - t));
                let p0 = Vec3::new(a.0 + off + (b.0 - a.0) * t0, arm_y + sag(t0), a.1 + (b.1 - a.1) * t0);
                let p1 = Vec3::new(a.0 + off + (b.0 - a.0) * t1, arm_y + sag(t1), a.1 + (b.1 - a.1) * t1);
                wire_edge(&mut g, p0, p1);
            }
        }
    }

    // --- railway: two rails toward the camera, sleepers, boxcars ---
    let (rail_x, gauge) = (-52.0f32, 7.0f32);
    let (rz0, rz1) = (-40.0f32, 95.0f32);
    wire_edge(&mut s, Vec3::new(rail_x, 0.4, rz0), Vec3::new(rail_x, 0.4, rz1));
    wire_edge(&mut s, Vec3::new(rail_x + gauge, 0.4, rz0), Vec3::new(rail_x + gauge, 0.4, rz1));
    let mut sz = rz0;
    while sz <= rz1 {
        wire_edge(&mut s, Vec3::new(rail_x - 1.5, 0.2, sz), Vec3::new(rail_x + gauge + 1.5, 0.2, sz));
        sz += 4.0;
    }
    for &bz in &[-8.0f32, 18.0, 46.0] {
        wire_box(&mut s, Vec3::new(rail_x - 1.0, 0.6, bz - 7.5), Vec3::new(rail_x + gauge + 1.0, 10.5, bz + 7.5));
    }

    // --- stacked shipping containers, foreground ---
    for &(cx, cz, cy) in &[
        (8.0f32, 52.0f32, 0.0f32),
        (30.0, 52.0, 0.0),
        (10.0, 54.0, 9.2),
        (-16.0, 66.0, 0.0),
        (52.0, 60.0, 0.0),
    ] {
        wire_box(&mut s, Vec3::new(cx, cy, cz), Vec3::new(cx + 20.0, cy + 9.0, cz + 13.0));
    }

    // --- gantry crane: mast + angled lattice jib ---
    let (mx, mz) = (-92.0f32, 6.0f32);
    wire_box(&mut s, Vec3::new(mx - 2.5, 0.0, mz - 2.5), Vec3::new(mx + 2.5, 34.0, mz + 2.5));
    let (ja, jb) = (Vec3::new(mx, 31.0, mz), Vec3::new(mx + 46.0, 41.0, mz));
    let (ja2, jb2) = (Vec3::new(mx, 27.0, mz), Vec3::new(mx + 44.0, 37.5, mz));
    wire_edge(&mut s, ja, jb);
    wire_edge(&mut s, ja2, jb2);
    for k in 0..6 {
        let t = k as f32 / 6.0;
        wire_edge(&mut s, ja.lerp(jb, t), ja2.lerp(jb2, t));
    }

    // --- conifers on the left (dim) ---
    for &(tx, tz, th) in &[
        (-150.0f32, 30.0f32, 16.0f32),
        (-172.0, 8.0, 20.0),
        (-140.0, -10.0, 14.0),
        (-186.0, -30.0, 22.0),
    ] {
        wire_edge(&mut g, Vec3::new(tx, 0.0, tz), Vec3::new(tx, th, tz));
        for &(dx, dz) in &[(5.0f32, 0.0f32), (0.0, 5.0)] {
            wire_edge(&mut g, Vec3::new(tx - dx, th * 0.35, tz - dz), Vec3::new(tx, th, tz));
            wire_edge(&mut g, Vec3::new(tx + dx, th * 0.35, tz + dz), Vec3::new(tx, th, tz));
            wire_edge(&mut g, Vec3::new(tx - dx, th * 0.35, tz - dz), Vec3::new(tx + dx, th * 0.35, tz + dz));
        }
    }

    (g, s)
}

pub fn spawn_menu_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let (ground, structures) = build_exfil_scene();
    let mut mk = |verts: Vec<[f32; 3]>| {
        let mut m = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
        meshes.add(m)
    };
    let ground_mesh = mk(ground);
    let struct_mesh = mk(structures);
    let ground_mat = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(0.02, 0.30, 0.45), // dim teal ground/pylons/trees
        alpha_mode: AlphaMode::Add,
        ..default()
    });
    let struct_mat = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(0.06, 1.5, 2.1), // HDR cyan -> Bloom neon
        alpha_mode: AlphaMode::Add,
        ..default()
    });
    for (mesh, mat) in [(ground_mesh, ground_mat), (struct_mesh, struct_mat)] {
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::default(),
            MenuScene { yaw: 0.0, tilt: 0.0 },
            Name::new("menu_exfil_wire"),
        ));
    }
    info!("menu: spawned Interchange-inspired neon wireframe exfil backdrop");
}

pub fn menu_scene_update(
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut q: Query<(&mut Transform, &mut MenuScene)>,
) {
    let dt = time.delta_secs();
    let t = time.elapsed_secs();
    let (cnx, cny) = windows
        .single()
        .ok()
        .and_then(|w| {
            w.cursor_position()
                .map(|c| ((c.x / w.width().max(1.0) - 0.5) * 2.0, (c.y / w.height().max(1.0) - 0.5) * 2.0))
        })
        .unwrap_or((0.0, 0.0));
    // idle sway + cursor parallax (holographic feel) — gentle, the scene is deep
    let target_yaw = (t * 0.11).sin() * 0.04 + cnx * 0.10;
    let target_tilt = -cny * 0.035;
    for (mut tf, mut sc) in &mut q {
        let k = 1.0 - (-3.0 * dt).exp();
        sc.yaw += (target_yaw - sc.yaw) * k;
        sc.tilt += (target_tilt - sc.tilt) * k;
        tf.rotation = Quat::from_rotation_y(sc.yaw) * Quat::from_rotation_x(sc.tilt);
    }
}

pub fn security_camera(ui: &egui::Ui, panel: Rect) {
    use std::f32::consts::{PI, TAU};
    const BODY: Color32 = Color32::from_rgb(42, 42, 40); // #2a2a28
    const DARK: Color32 = Color32::from_rgb(28, 28, 27); // #1c1c1b
    const EDGE: Color32 = Color32::from_rgb(19, 19, 18);
    const SEAM: Color32 = Color32::from_rgb(52, 51, 47);
    const METAL: Color32 = Color32::from_rgb(35, 35, 34);

    let ctx = ui.ctx();
    let t = ctx.input(|i| i.time) as f32;
    let dt = ctx.input(|i| i.stable_dt).min(0.1);

    // Mount: plate on the panel's right edge, arm reaching left-down to the pivot joint.
    let plate = Rect::from_min_max(
        pos2(panel.right() - 12.0, panel.top() + 24.0),
        pos2(panel.right() - 4.0, panel.top() + 62.0),
    );
    let pivot = pos2(plate.left() - 34.0, plate.center().y + 10.0);

    // ---- target pose ----
    const YAW_MAX: f32 = 40.0 * PI / 180.0;
    const PITCH_MAX: f32 = 25.0 * PI / 180.0;
    let rest = 2.62_f32; // rad (~150 deg): down-left, toward the middle of the map list
    let (yaw_t, pitch_t) = match ctx.input(|i| i.pointer.hover_pos()) {
        Some(m) => {
            let v = m - pivot;
            // pitch: wrapped screen-space angle to the cursor, relative to rest
            let mut d = v.y.atan2(v.x) - rest;
            while d > PI {
                d -= TAU;
            }
            while d < -PI {
                d += TAU;
            }
            // yaw: how far across the panel the cursor sits. Directly under the camera
            // (v.x ~ 0) turns it toward the wall (+), far screen-left turns it into the
            // scene (-). Linear in x reads better than atan2-vs-depth at this size.
            let yn = (1.0 + 2.0 * v.x / panel.width().max(1.0)).clamp(-1.0, 1.0);
            (yn * YAW_MAX, d.clamp(-PITCH_MAX, PITCH_MAX))
        }
        // Idle patrol: slow side-to-side sweep (~12 s period) with a faint nod.
        None => (
            (t * 0.5).sin() * YAW_MAX * 0.85,
            (t * 0.23).sin() * 0.05 - 0.06,
        ),
    };

    // ---- servo slew: exponential approach, frame-rate independent (alpha = 1-e^-k*dt) ----
    let id = egui::Id::new("menu_fx_cam_pose");
    let (mut yaw, mut pitch) =
        ctx.data(|d| d.get_temp::<(f32, f32)>(id)).unwrap_or((0.0, -0.05));
    let a = 1.0 - (-6.0 * dt).exp();
    yaw += (yaw_t - yaw) * a;
    pitch += (pitch_t - pitch) * a;
    ctx.data_mut(|d| d.insert_temp(id, (yaw, pitch)));

    // ---- geometry ----
    let ang = rest + pitch;
    let u = egui::Vec2::angled(ang); // body forward axis
    let n = vec2(-u.y, u.x); // body normal; +n is the "top" side for the rest pose
    let len = 56.0 * (0.78 + 0.22 * yaw.cos()); // yaw foreshortens the body
    let (wb, wf) = (13.0_f32, 10.0_f32); // back / front half-widths (tapered box)
    let sk = n * (yaw.sin() * 9.0); // yaw skews the front face + lens laterally

    let b = pivot + vec2(1.0, 9.0); // body back-center hangs under the bracket joint
    let f = b + u * len;
    let fc = f + sk; // front-face (lens) center

    let p = ui.painter();

    // Faint surveillance cone out of the lens — barely-there, sells the tracking.
    p.add(Shape::convex_polygon(
        vec![
            fc,
            fc + egui::Vec2::angled(ang + 0.15) * 88.0,
            fc + egui::Vec2::angled(ang - 0.15) * 88.0,
        ],
        Color32::from_rgba_unmultiplied(200, 196, 180, 6),
        Stroke::NONE,
    ));

    // Wall plate + screws.
    p.rect_filled(plate, 0.0, DARK);
    p.rect_stroke(plate, 0.0, Stroke::new(1.0, SEAM), StrokeKind::Inside);
    for dy in [-13.0, 13.0] {
        p.circle_filled(pos2(plate.center().x, plate.center().y + dy), 1.3, SEAM);
    }
    // Arm + pivot joint + short bracket down to the body.
    p.line_segment([plate.left_center(), pivot], Stroke::new(4.0, METAL));
    p.line_segment([pivot, b], Stroke::new(2.5, METAL));
    p.circle_filled(pivot, 4.5, DARK);
    p.circle_stroke(pivot, 4.5, Stroke::new(1.0, SEAM));

    // Body: tapered quad from back (at the bracket) to the skewed front face.
    p.add(Shape::convex_polygon(
        vec![b + n * wb, fc + n * wf, fc - n * wf, b - n * wb],
        BODY,
        Stroke::new(1.0, EDGE),
    ));
    // Darker rear cap (power/cable block).
    let bp = |du: f32, w: f32| b + u * du + n * w;
    p.add(Shape::convex_polygon(
        vec![
            bp(-3.0, wb + 1.5),
            bp(7.0, wb + 1.5),
            bp(7.0, -(wb + 1.5)),
            bp(-3.0, -(wb + 1.5)),
        ],
        DARK,
        Stroke::new(1.0, EDGE),
    ));
    // Panel seams across the housing (skew interpolated toward the front).
    for fs in [0.42_f32, 0.68] {
        let c = b + u * (len * fs) + sk * fs;
        let w = wb + (wf - wb) * fs - 1.0;
        p.line_segment([c + n * w, c - n * w], Stroke::new(1.0, METAL));
    }
    // Hood / sun visor over the lens (protrudes past the front on the top side).
    let hp = |du: f32, w: f32| fc + u * du + n * w;
    p.add(Shape::convex_polygon(
        vec![
            hp(-2.0, wf + 1.5),
            hp(7.0, wf + 1.5),
            hp(7.0, wf - 3.0),
            hp(-2.0, wf - 3.0),
        ],
        DARK,
        Stroke::new(1.0, EDGE),
    ));
    // Lens: dark glass, ring, dim curved glint.
    p.circle_filled(fc, 6.0, Color32::from_rgb(16, 16, 16));
    p.circle_stroke(fc, 6.0, Stroke::new(1.0, SEAM));
    p.circle_filled(fc, 3.2, Color32::from_rgb(24, 25, 26));
    p.circle_filled(
        fc + vec2(-1.8, -1.8),
        1.3,
        Color32::from_rgba_unmultiplied(150, 158, 158, 90),
    );

    // Record LED, top-rear: sharp 0.8 s on/off blink with a soft glow halo, independent of
    // the tracking pose.
    let led = b + u * 9.0 + n * (wb - 3.5);
    if t.rem_euclid(0.8) < 0.4 {
        p.circle_filled(led, 7.5, Color32::from_rgba_unmultiplied(RED.r(), RED.g(), RED.b(), 18));
        p.circle_filled(led, 4.5, Color32::from_rgba_unmultiplied(RED.r(), RED.g(), RED.b(), 55));
        p.circle_filled(led, 2.1, Color32::from_rgb(226, 82, 74));
    } else {
        p.circle_filled(led, 1.8, Color32::from_rgb(58, 34, 32));
    }
}

// ============================ real-asset 3D menu prop ============================
//
// The REAL Street_Camera_01 CCTV from the lighthouse dataset, spawned as a Bevy 3D
// entity parented to the menu camera. Loaded from packs/shared/menu/camera.{bin,png}
// (machine-local extraction — tools/extract_menu_prop.py; packs/ is gitignored so the
// game asset is never committed or shipped). Bevy's png/gltf features are OFF in this
// build, so BOTH the mesh and the texture are built manually from raw bytes (the same
// pattern as main.rs build_sky_cubemap): Mesh::new + insert_attribute, and image-crate
// decode -> Image::new(Rgba8UnormSrgb).

/// Tracking cone + servo feel: ported 1:1 from the vector `security_camera` above.
const YAW_MAX: f32 = 40.0 * std::f32::consts::PI / 180.0;
const PITCH_MAX: f32 = 25.0 * std::f32::consts::PI / 180.0;
/// Servo slew rate (alpha = 1 - e^(-K*dt), frame-rate independent).
const SLEW_K: f32 = 6.0;
/// LED blink period/duty: 0.8 s cycle, on for the first half (matches the vector LED).
const LED_PERIOD: f32 = 0.8;
const LED_ON: f32 = 0.4;
/// Camera-space framing: prop distance in front of the menu camera, and the on-screen
/// anchor (px from the right edge / from the top) it is steered onto each frame.
/// menu.rs reserves a 166 px right gutter (+24 px CentralPanel margin) for the decor.
const PROP_DIST: f32 = 3.5;
const GUTTER_CENTER_FROM_RIGHT: f32 = 24.0 + 166.0 * 0.5;
const PROP_CENTER_FROM_TOP: f32 = 170.0;
/// Target world size of the prop's largest axis, meters (in game it stands ~0.47 m).
const PROP_SIZE_M: f32 = 0.5;
/// Menu camera vertical FOV — must match the `PerspectiveProjection::default()` the
/// menu-mode `setup` (main.rs) leaves on the camera.
const MENU_FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// Present ONLY when the real-asset prop actually spawned; menu.rs uses its absence to
/// fall back to the vector `security_camera` (never both).
#[derive(Resource)]
pub struct MenuCamProp {
    prop: Entity,
    led_mat: Handle<StandardMaterial>,
    yaw: f32,
    pitch: f32,
    /// Last LED state actually written to the material (only mutate the asset on edges,
    /// so the material isn't re-uploaded every frame).
    led_on: bool,
}

struct PropData {
    mesh: Mesh,
    image: Image,
    /// Half-extents of the CENTERED mesh (raw dataset units) — drives scale + LED anchor.
    half: Vec3,
}

/// Parse packs/shared/menu/camera.bin + camera.png. Any structural problem returns None
/// (the caller falls back to the vector camera) with an ASCII diagnostic on stderr.
///
/// camera.bin layout (little-endian, written by tools/extract_menu_prop.py):
///   [u32 vert_count][u32 index_count]
///   [pos f32x3 * n][normal f32x3 * n][uv f32x2 * n][indices u32 * m]
fn load_prop_data(dir: &std::path::Path) -> Option<PropData> {
    let bin = match std::fs::read(dir.join("camera.bin")) {
        Ok(b) => b,
        Err(_) => return None, // absent = the normal shipped state; stay quiet
    };
    if bin.len() < 8 {
        eprintln!("menu prop: camera.bin truncated ({} bytes)", bin.len());
        return None;
    }
    let nv = u32::from_le_bytes(bin[0..4].try_into().unwrap()) as usize;
    let ni = u32::from_le_bytes(bin[4..8].try_into().unwrap()) as usize;
    let expect = 8 + nv * 32 + ni * 4;
    if nv == 0 || ni == 0 || ni % 3 != 0 || nv > 4_000_000 || bin.len() != expect {
        eprintln!(
            "menu prop: camera.bin corrupt (verts {nv} indices {ni} len {} expect {expect})",
            bin.len()
        );
        return None;
    }
    let f = |off: usize| f32::from_le_bytes(bin[off..off + 4].try_into().unwrap());
    let (pos_off, nrm_off, uv_off, idx_off) = (8, 8 + nv * 12, 8 + nv * 24, 8 + nv * 32);
    let mut pos = Vec::with_capacity(nv);
    let mut nrm = Vec::with_capacity(nv);
    let mut uv = Vec::with_capacity(nv);
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for i in 0..nv {
        let p = [f(pos_off + i * 12), f(pos_off + i * 12 + 4), f(pos_off + i * 12 + 8)];
        if !p.iter().all(|v| v.is_finite()) {
            eprintln!("menu prop: camera.bin has non-finite positions");
            return None;
        }
        lo = lo.min(Vec3::from(p));
        hi = hi.max(Vec3::from(p));
        pos.push(p);
        nrm.push([f(nrm_off + i * 12), f(nrm_off + i * 12 + 4), f(nrm_off + i * 12 + 8)]);
        uv.push([f(uv_off + i * 8), f(uv_off + i * 8 + 4)]);
    }
    let mut idx = Vec::with_capacity(ni);
    for i in 0..ni {
        let v = u32::from_le_bytes(bin[idx_off + i * 4..idx_off + i * 4 + 4].try_into().unwrap());
        if v as usize >= nv {
            eprintln!("menu prop: camera.bin index {v} out of range ({nv} verts)");
            return None;
        }
        idx.push(v);
    }
    let half = ((hi - lo) * 0.5).max(Vec3::splat(1e-4));

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, nrm);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(idx));

    // Albedo: manual decode (bevy's png asset feature is off in this trimmed build).
    let png = std::fs::read(dir.join("camera.png")).ok()?;
    let decoded = match image::load_from_memory(&png) {
        Ok(d) => d.to_rgba8(),
        Err(e) => {
            eprintln!("menu prop: camera.png decode failed: {e}");
            return None;
        }
    };
    let (w, h) = decoded.dimensions();
    let image = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        decoded.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    Some(PropData { mesh, image, half })
}

/// Startup (menu mode only, after main.rs `setup`): spawn the real CCTV as a child of the
/// menu camera, plus a red LED sphere on the lens hood and a small fill light (the menu
/// world only has the dim analytic key light — without a fill the prop reads near-black).
/// No asset on disk -> no resource -> menu.rs paints the vector camera as before.
pub fn spawn_menu_prop(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    cam: Query<Entity, With<crate::FlyCam>>,
) {
    let dir = crate::paths::shared_dir().join("menu");
    let Some(data) = load_prop_data(&dir) else {
        eprintln!("menu prop: no local extraction in packs/shared/menu - using the vector camera");
        return;
    };
    let Ok(cam_e) = cam.single() else {
        eprintln!("menu prop: no camera entity; keeping the vector camera");
        return;
    };

    let half = data.half;
    let scale = PROP_SIZE_M / (half.max_element() * 2.0);
    let verts = data.mesh.count_vertices();

    let body_mat = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(data.image)),
        perceptual_roughness: 0.7,
        // Double-sided: dataset OBJs are UnityPy X-flipped with reversed winding (the pack
        // pipeline conjugates that per instance; a standalone prop must not cull on it).
        cull_mode: None,
        double_sided: true,
        ..default()
    });
    let led_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.02, 0.02),
        emissive: LinearRgba::new(14.0, 0.9, 0.8, 1.0), // HDR-hot so Bloom halos it
        ..default()
    });

    // Mount geometry (centered raw units): wall plate at +Y, housing below center, lens
    // facing +Z. The LED sits on the right flank of the lens hood.
    let led_pos = Vec3::new(half.x * 0.45, -half.y * 0.35, half.z * 0.8);
    let led_r = half.max_element() * 0.045;

    let prop = commands
        .spawn((
            Mesh3d(meshes.add(data.mesh)),
            MeshMaterial3d(body_mat),
            // Placed/steered every frame by menu_prop_update; this is just a sane seed.
            Transform::from_translation(Vec3::new(2.2, 0.9, -PROP_DIST))
                .with_scale(Vec3::splat(scale)),
        ))
        .id();
    let led = commands
        .spawn((
            Mesh3d(meshes.add(Sphere::new(led_r))),
            MeshMaterial3d(led_mat.clone()),
            Transform::from_translation(led_pos),
        ))
        .id();
    // Fill light: camera-space up-left of the prop, so the housing front reads. Menu-only
    // world => nothing else is close enough to catch it.
    let fill = commands
        .spawn((
            PointLight {
                color: Color::srgb(1.0, 0.97, 0.9),
                intensity: 500_000.0,
                range: 12.0,
                shadows_enabled: false,
                ..default()
            },
            Transform::from_translation(Vec3::new(0.9, 1.8, -1.8)),
        ))
        .id();
    commands.entity(prop).add_child(led);
    commands.entity(cam_e).add_child(prop);
    commands.entity(cam_e).add_child(fill);
    commands.insert_resource(MenuCamProp {
        prop,
        led_mat,
        yaw: 0.0,
        pitch: -0.05,
        led_on: true,
    });
    eprintln!("menu prop: real CCTV loaded ({verts} verts) from packs/shared/menu");
}

/// Per-frame (menu mode only): steer the prop onto the right-gutter screen anchor for the
/// CURRENT window size, servo-track the cursor (idle patrol sweep when it is outside the
/// window), and blink the LED. Same constants/feel as the vector `security_camera`.
pub fn menu_prop_update(
    prop: Option<ResMut<MenuCamProp>>,
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut transforms: Query<&mut Transform>,
) {
    let Some(mut st) = prop else { return };
    let Ok(win) = windows.single() else { return };
    let (w, h) = (win.width().max(1.0), win.height().max(1.0));
    let t = time.elapsed_secs();
    let dt = time.delta_secs().min(0.1);

    // ---- target pose: cursor NDC -> clamped yaw/pitch; no cursor -> idle patrol ----
    let (yaw_t, pitch_t) = match win.cursor_position() {
        Some(c) => {
            let nx = (2.0 * c.x / w - 1.0).clamp(-1.0, 1.0);
            let ny_up = (1.0 - 2.0 * c.y / h).clamp(-1.0, 1.0);
            // +yaw turns the lens screen-right; +pitch (rot_x) tilts it down, so cursor-up
            // (ny_up > 0) needs a negative pitch.
            (nx * YAW_MAX, -ny_up * PITCH_MAX)
        }
        // Idle patrol: slow side-to-side sweep with a faint nod (vector-cam constants).
        None => (
            (t * 0.5).sin() * YAW_MAX * 0.85,
            (t * 0.23).sin() * 0.05 - 0.06,
        ),
    };
    let a = 1.0 - (-SLEW_K * dt).exp();
    st.yaw += (yaw_t - st.yaw) * a;
    st.pitch += (pitch_t - st.pitch) * a;

    // ---- camera-space framing: put the prop at the right-gutter anchor at PROP_DIST ----
    // Perspective: at distance d the view half-height is d*tan(fov/2); x/y follow from the
    // anchor's NDC. Recomputed per frame so window resizes keep it glued to the gutter.
    let half_h = PROP_DIST * (MENU_FOV_Y * 0.5).tan();
    let half_w = half_h * (w / h);
    let ndc_x = 2.0 * (w - GUTTER_CENTER_FROM_RIGHT) / w - 1.0;
    let ndc_y = 1.0 - 2.0 * PROP_CENTER_FROM_TOP.min(h * 0.35) / h;
    if let Ok(mut tf) = transforms.get_mut(st.prop) {
        tf.translation = Vec3::new(ndc_x * half_w, ndc_y * half_h, -PROP_DIST);
        tf.rotation = Quat::from_rotation_y(st.yaw) * Quat::from_rotation_x(st.pitch);
    }

    // ---- LED blink: hard 0.8 s on/off; mutate the material asset only on edges ----
    let on = t.rem_euclid(LED_PERIOD) < LED_ON;
    if on != st.led_on {
        st.led_on = on;
        if let Some(m) = materials.get_mut(&st.led_mat) {
            m.emissive = if on {
                LinearRgba::new(14.0, 0.9, 0.8, 1.0)
            } else {
                LinearRgba::new(0.02, 0.002, 0.002, 1.0)
            };
        }
    }
}
