//! pick.rs — double-LEFT-click "identify geometry" debug tool.
//!
//! This is the native analogue of the web viewer's debug pick. On a double
//! left-click it casts a world ray from the cursor and intersects it AGAINST THE
//! PACK DATA — the per-instance affines and the decoded mesh TRIANGLES — **not**
//! the rendered pixels. That distinction is the whole point: a prop can be
//! present in the pack with valid geometry, a valid transform and a normal opaque
//! material yet render completely invisible (a render bug). Clicking where it
//! *should* be and getting a triangle hit proves the geometry IS in the data and
//! names the exact mesh/instance/material to blame.
//!
//! Pipeline per pick:
//!   1. cursor -> world ray via `Camera::viewport_to_world` (CullCamera view).
//!   2. BROADPHASE: ray vs each instance's WORLD bounding sphere (local sphere
//!      from `Pack::bounding_spheres()`, transformed by the instance affine with
//!      the shear-correct max-column-norm radius scale — same convention as the
//!      GPU cull). Collect (instance, t_near>=0).
//!   3. NARROWPHASE: nearest up to 96 spheres, decode that mesh on demand, push
//!      the RAY into mesh-local space with the affine inverse (direction NOT
//!      renormalized, so the Möller–Trumbore `t` is already the WORLD distance),
//!      and test every triangle double-sided. Keep the globally nearest hit.
//!   4. Report the hit mesh/instance/material both ON-SCREEN (Bevy UI Text) and
//!      to the console (with the top-5 nearby sphere candidates for diagnosis).

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::render::{CullCamera, LoadedPack};

/// Two left-presses within this window (seconds) count as a double-click.
const DOUBLE_CLICK_SECS: f32 = 0.4;
/// Cap on triangle-tested meshes per pick (nearest spheres first) so a click in
/// a dense area can't stall the frame decoding thousands of meshes.
const MAX_NARROWPHASE: usize = 96;
/// Möller–Trumbore parallel-ray / min-distance epsilon (world metres).
const T_EPS: f32 = 1.0e-4;

/// Timing state for the double-click detector. NEG_INFINITY so the very first
/// click is never mistaken for the second half of a double.
#[derive(Resource)]
struct PickState {
    last_left: f32,
}
impl Default for PickState {
    fn default() -> Self {
        Self {
            last_left: f32::NEG_INFINITY,
        }
    }
}

/// Cached per-mesh LOCAL bounding spheres (computed once, on the first pick).
/// This is NOT "preloading all triangles" — it's 4 floats per mesh, exactly what
/// the cull already keeps — it just spares us recomputing every centroid on each
/// click. Triangles are still decoded on demand, per pick.
#[derive(Resource, Default)]
struct PickSpheres(Vec<[f32; 4]>);

/// Marker for the single on-screen readout text node.
#[derive(Component)]
struct PickReadout;

pub struct PickPlugin;

impl Plugin for PickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PickState>()
            .init_resource::<PickSpheres>()
            .add_systems(Startup, spawn_pick_ui)
            .add_systems(Update, pick_system);
    }
}

/// Top-left, dark, semi-transparent readout. Spawned once; the pick system just
/// rewrites its string.
fn spawn_pick_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("PICK  double-click to identify geometry"),
        TextFont {
            font_size: 13.0,
            ..default()
        },
        TextColor(Color::srgb(0.90, 0.96, 0.90)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            padding: UiRect::all(Val::Px(6.0)),
            max_width: Val::Px(760.0),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        PickReadout,
    ));
}

/// A resolved triangle hit against the pack data.
struct Hit {
    t: f32,
    inst: usize,
    mesh_id: usize,
    material_id: u32,
}

#[allow(clippy::too_many_arguments)]
fn pick_system(
    mouse: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    mut state: ResMut<PickState>,
    mut spheres: ResMut<PickSpheres>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<CullCamera>>,
    pack: Option<Res<LoadedPack>>,
    pointer_on_ui: Res<crate::inspect::PointerOnUi>,
    keys: Res<ButtonInput<KeyCode>>,
    ui_kb: Res<crate::inspect::UiWantsKeyboard>,
    mut place: ResMut<crate::pathfind::PlaceMode>,
    mut start_pt: ResMut<crate::pathfind::StartPoint>,
    mut readout: Query<&mut Text, With<PickReadout>>,
) {
    // Esc cancels an armed place-position mode (checked before the click gate so it works without
    // any click) — unless a text field has focus, where Esc means "defocus the field".
    if place.0 && keys.just_pressed(KeyCode::Escape) && !ui_kb.0 {
        place.0 = false;
    }
    // ---- click gate --------------------------------------------------------
    // Ignore clicks landing on egui panels (Codex review: UI double-clicks were triggering the
    // expensive full-geometry raycast).
    if !mouse.just_pressed(MouseButton::Left) || pointer_on_ui.0 {
        return;
    }
    // An ARMED place mode (Navigation tab "PLACE ON MAP") makes the next single click place the
    // route start on the clicked surface; SHIFT-click does the same without arming (power users).
    // A plain DOUBLE-click identifies geometry. All share the one full-geometry raycast below.
    let place_start =
        place.0 || keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let now = time.elapsed_secs();
    let is_double = (now - state.last_left) <= DOUBLE_CLICK_SECS;
    state.last_left = now;
    if place_start {
        // A placement click must not PRIME the double-click detector: without this, a quick
        // follow-up plain click would read as a double and fire the identify raycast.
        state.last_left = f32::NEG_INFINITY;
    } else if !is_double {
        return; // first (or lone) plain click — arm and wait for the second
    }
    if is_double {
        // Consume, so a third rapid click needs two fresh presses to re-fire.
        state.last_left = f32::NEG_INFINITY;
    }

    let set_text = |readout: &mut Query<&mut Text, With<PickReadout>>, s: String| {
        if let Ok(mut t) = readout.single_mut() {
            t.0 = s;
        }
    };

    // ---- gather cursor + camera + pack (bail cleanly on any miss) ----------
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        return;
    };
    let Some(pack) = pack.as_ref() else {
        info!("PICK (no pack loaded)");
        set_text(&mut readout, "PICK  (no pack loaded)".to_string());
        return;
    };
    let pack = &pack.0;

    let ro: Vec3 = ray.origin;
    let rd: Vec3 = *ray.direction; // unit world direction

    // ---- cache the local bounding spheres on first use ---------------------
    if spheres.0.is_empty() {
        match pack.bounding_spheres() {
            Ok(s) => spheres.0 = s,
            Err(e) => {
                warn!("PICK: bounding_spheres failed: {e:#}");
                set_text(&mut readout, "PICK  (geometry unavailable)".to_string());
                return;
            }
        }
    }
    let spheres = &spheres.0;
    if pack.instances.is_empty() || spheres.is_empty() {
        set_text(&mut readout, "PICK  (empty pack)".to_string());
        return;
    }

    // ---- BROADPHASE: ray vs each instance's world bounding sphere ----------
    // world sphere: center' = affine * local_center; radius' = local_r * max
    // column norm of the linear part (conservative under shear/scale/mirror —
    // matches the cull's radius scale).
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (i, inst) in pack.instances.iter().enumerate() {
        let mid = inst.mesh_id as usize;
        if mid >= spheres.len() {
            continue;
        }
        let s = spheres[mid];
        let aff = inst.affine3a();
        let local_c = Vec3::new(s[0], s[1], s[2]);
        let wc = aff.transform_point3(local_c);
        let m3 = aff.matrix3;
        let cscale = m3
            .x_axis
            .length()
            .max(m3.y_axis.length())
            .max(m3.z_axis.length());
        let wr = s[3] * cscale;
        if let Some(t) = ray_sphere(ro, rd, wc, wr) {
            candidates.push((i, t));
        }
    }

    // nearest first — drives both narrowphase order and the sphere-only fallback.
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // ---- NARROWPHASE: Möller–Trumbore vs the actual pack triangles ---------
    let mut best: Option<Hit> = None;
    for &(inst_idx, _t_near) in candidates.iter().take(MAX_NARROWPHASE) {
        let inst = &pack.instances[inst_idx];
        let mid = inst.mesh_id as usize;
        let Some(m) = pack.manifest.meshes.get(mid) else {
            continue;
        };
        let aff = inst.affine3a();
        // A singular linear part would make the inverse NaN/Inf; skip the
        // triangle test for it (it still counts as a sphere candidate).
        if aff.matrix3.determinant().abs() < 1.0e-12 {
            continue;
        }
        let geom = match pack.mesh_geom(m) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let inv = aff.inverse();
        // Ray -> mesh-local. Do NOT renormalize the direction: with ld = M^-1*rd,
        // a local hit param `t` satisfies affine*(lo + t*ld) = ro + t*rd, so `t`
        // is already the WORLD-space distance and hits compare directly.
        let lo = inv.transform_point3(ro);
        let ld = inv.transform_vector3(rd);

        let idx = &geom.indices;
        let pos = &geom.positions;
        for sm in &m.submeshes {
            let start = sm.idx_start as usize;
            let end = start.saturating_add(sm.idx_count as usize);
            if end > idx.len() {
                continue;
            }
            // `end <= idx.len()` was checked above; step whole triangles.
            let mut tri = start;
            while tri + 3 <= end {
                let ia = idx[tri] as usize;
                let ib = idx[tri + 1] as usize;
                let ic = idx[tri + 2] as usize;
                tri += 3;
                if ia >= pos.len() || ib >= pos.len() || ic >= pos.len() {
                    continue;
                }
                let v0 = Vec3::from_array(pos[ia]);
                let v1 = Vec3::from_array(pos[ib]);
                let v2 = Vec3::from_array(pos[ic]);
                if let Some(t) = ray_triangle(lo, ld, v0, v1, v2) {
                    if best.as_ref().map_or(true, |h| t < h.t) {
                        best = Some(Hit {
                            t,
                            inst: inst_idx,
                            mesh_id: mid,
                            material_id: sm.material_id,
                        });
                    }
                }
            }
        }
    }

    // ---- build the top-5 nearby sphere-candidate list for the log ----------
    let top5: Vec<String> = candidates
        .iter()
        .take(5)
        .map(|&(i, tn)| {
            let mid = pack.instances[i].mesh_id as usize;
            let name = pack
                .manifest
                .meshes
                .get(mid)
                .map(|m| m.name.as_str())
                .unwrap_or("?");
            format!("{name}(inst {i}, d={tn:.1})")
        })
        .collect();
    let cand_str = if top5.is_empty() {
        "none".to_string()
    } else {
        top5.join(", ")
    };

    // ---- resolve + report --------------------------------------------------
    let material_role = |mid: u32| -> String {
        pack.materials
            .iter()
            .find(|m| m.id == mid)
            .map(|m| m.role.clone())
            .unwrap_or_else(|| "?".to_string())
    };

    if let Some(h) = best {
        let mesh_name = pack
            .manifest
            .meshes
            .get(h.mesh_id)
            .map(|m| m.name.as_str())
            .unwrap_or("?");
        let role = material_role(h.material_id);
        let wp = ro + rd * h.t;
        if place_start {
            // Drop the route start ("you are here") on the clicked surface; disarm the mode.
            start_pt.0 = Some(wp);
            place.0 = false;
            let line = format!(
                "POSITION set  ({:.1}, {:.1}, {:.1})  \u{2014} routes begin here",
                wp.x, wp.y, wp.z
            );
            info!("{line}");
            set_text(&mut readout, line);
            return;
        }
        let line = format!(
            "PICK  {mesh}  #{id}  inst {inst}  mat {mat} {role}  d={dist:.1}m  ({x:.1}, {y:.1}, {z:.1})",
            mesh = mesh_name,
            id = h.mesh_id,
            inst = h.inst,
            mat = h.material_id,
            role = role,
            dist = h.t,
            x = wp.x,
            y = wp.y,
            z = wp.z,
        );
        info!("{line}  | candidates: {cand_str}");
        set_text(&mut readout, line);
    } else if place_start {
        // Armed placement but the ray hit no surface (sky / off-map): stay armed so the next
        // click can try again, and say so instead of dumping an identify readout.
        set_text(
            &mut readout,
            "POSITION  no surface there \u{2014} click on the map geometry (Esc cancels)".to_string(),
        );
    } else if let Some(&(inst_idx, t_near)) = candidates.first() {
        // No triangle hit, but the ray passed through some bounding spheres —
        // report the nearest as a sphere-only near-miss.
        let inst = &pack.instances[inst_idx];
        let mid = inst.mesh_id as usize;
        let mesh_name = pack
            .manifest
            .meshes
            .get(mid)
            .map(|m| m.name.as_str())
            .unwrap_or("?");
        let mat0 = pack
            .manifest
            .meshes
            .get(mid)
            .and_then(|m| m.submeshes.first())
            .map(|s| s.material_id)
            .unwrap_or(0);
        let role = material_role(mat0);
        let wp = ro + rd * t_near;
        let line = format!(
            "PICK  {mesh}  #{id}  inst {inst}  mat {mat} {role}  d={dist:.1}m  ({x:.1}, {y:.1}, {z:.1})  (sphere-only)",
            mesh = mesh_name,
            id = mid,
            inst = inst_idx,
            mat = mat0,
            role = role,
            dist = t_near,
            x = wp.x,
            y = wp.y,
            z = wp.z,
        );
        info!("{line}  | candidates: {cand_str}");
        set_text(&mut readout, line);
    } else {
        info!("PICK (nothing hit)  | candidates: {cand_str}");
        set_text(&mut readout, "PICK  (nothing hit)".to_string());
    }
}

/// Ray vs sphere. `rd` MUST be unit length. Returns the nearest non-negative
/// entry `t` (0 if the origin is inside the sphere), or None if the sphere is
/// entirely behind the ray or missed.
#[inline]
fn ray_sphere(ro: Vec3, rd: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ro - center;
    let b = oc.dot(rd);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c; // a == 1 (rd unit)
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t0 = -b - sq;
    let t1 = -b + sq;
    if t1 < 0.0 {
        return None; // whole sphere behind the ray
    }
    Some(if t0 >= 0.0 { t0 } else { 0.0 })
}

/// Double-sided Möller–Trumbore. `d` need NOT be unit length; the returned `t`
/// is in units of `d` (see caller: `d` is the affine-inverse-mapped ray, so `t`
/// is the world distance). Double-sided = no back-face cull, so it hits geometry
/// regardless of winding / mirror (exactly what an invisible prop needs).
#[inline]
fn ray_triangle(o: Vec3, d: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1.0e-8 {
        return None; // ray parallel to triangle
    }
    let inv_det = 1.0 / det;
    let tvec = o - v0;
    let u = tvec.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tvec.cross(e1);
    let v = d.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv_det;
    if t > T_EPS {
        Some(t)
    } else {
        None
    }
}
