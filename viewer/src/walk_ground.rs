//! eft::walk_ground — ground-height query for the walk camera (UI_ROADMAP §1c, design-workflow spec).
//!
//! Free-fly can't "walk the map": there's no floor to stand on. This builds a static 2.5D query
//! over the pack's world geometry so a walk camera can follow floors/stairs/terrain and land jumps
//! on the RIGHT storey in multi-floor buildings.
//!
//! Design (from the read-only design workflow, Approach A):
//!  - Enumerate every WORLD triangle (mesh-local verts × each instance's affine — NEVER decompose).
//!  - Keep only near-horizontal WALKABLE faces (world face-normal n.y > 0.5, computed from the
//!    transformed positions via cross product — mirror/shear-proof, unlike the stored local normals).
//!  - Bucket each kept triangle's index into every XZ grid cell its footprint overlaps (big slabs
//!    span many cells — insert-by-AABB or you get fall-through seams at cell borders).
//!  - `ground_height(x,z, feet_y, step_up)` = the GREATEST surface Y at (x,z) that is ≤ feet_y+step_up.
//!    "Highest surface below the feet (+ a small step allowance)" is what makes multi-floor correct:
//!    on floor 3 the floor-5 slab is above the feet and excluded; the STEP_UP lets you mount stairs
//!    and curbs but not tabletops. Terrain tiles are ordinary meshes, so their triangles are already
//!    in the grid — no separate heightfield needed.
//!
//! Built ONCE (lazily, on first walk activation) off the main thread's critical path via the compute
//! task pool. Per-query cost is O(triangles in a 3×3 cell block), non-allocating, no mesh decode.

use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use crate::eftpack::Pack;

/// Camera eye height above the feet/ground (m).
pub const EYE_HEIGHT: f32 = 1.7;
/// Max height the feet may rise to select a surface (mount stairs/curbs, not tabletops) (m).
pub const STEP_UP: f32 = 0.45;
/// Snappy game-feel gravity (not real 9.8) (m/s²).
pub const GRAVITY: f32 = 20.0;
/// Jump apex height per unit walk-speed (m per m/s) — apex = JUMP_K·walk_speed, so one scroll
/// gesture juices both movement speed and hop height together.
pub const JUMP_K: f32 = 0.12;
/// XZ grid cell size (m).
const CELL: f32 = 3.0;
/// Only faces at least this upward (n.y/|n|) are walkable (~≤60° from horizontal).
const HORIZ_MIN: f32 = 0.5;
/// If the camera falls this far below the last known ground, treat as fell-through-world.
pub const KILL_DROP: f32 = 60.0;

/// Per-camera walk locomotion state (lives on the CullCamera+FlyCam entity).
#[derive(Component, Default)]
pub struct WalkState {
    /// Vertical velocity (m/s); gravity integrates it, jump sets it.
    pub vy: f32,
    /// Standing on a surface this frame (vs airborne).
    pub grounded: bool,
    /// Last resolved ground Y (for the fell-through-world backstop).
    pub last_ground_y: f32,
    /// Whether last_ground_y is valid yet.
    pub has_ground: bool,
}

/// Prebuilt walkable-surface query. `tris` stores each kept world triangle ONCE (36 B); `cells`
/// holds only u32 indices into it, so a slab spanning many cells duplicates the index, not the tri.
#[derive(Resource)]
pub struct GroundGrid {
    origin_xz: Vec2,
    inv_cell: f32,
    nx: u32,
    nz: u32,
    tris: Vec<[Vec3; 3]>,
    cells: Vec<Vec<u32>>,
}

impl GroundGrid {
    /// Enumerate + filter + bucket every walkable world triangle. Parallel over instances.
    pub fn build(pack: &Pack) -> Self {
        let t0 = std::time::Instant::now();
        let instances = &pack.instances;
        let pool = ComputeTaskPool::get();
        let threads = pool.thread_num().max(1);
        let chunk = instances.len().div_ceil(threads).max(1);

        // Phase 1 (parallel): each worker collects its kept world triangles.
        let per_thread: Vec<Vec<[Vec3; 3]>> = pool.scope(|s| {
            for c in instances.chunks(chunk) {
                s.spawn(async move {
                    let mut out: Vec<[Vec3; 3]> = Vec::new();
                    for inst in c {
                        let aff = inst.affine3a();
                        // Skip singular instances (degenerate affine → garbage transform).
                        if aff.matrix3.determinant().abs() < 1e-12 {
                            continue;
                        }
                        let Some(m) = pack.manifest.meshes.get(inst.mesh_id as usize) else {
                            continue;
                        };
                        let Ok(geom) = pack.mesh_geom(m) else { continue };
                        for sm in &m.submeshes {
                            let start = sm.idx_start as usize;
                            let end = start + sm.idx_count as usize;
                            let idx = &geom.indices;
                            let mut i = start;
                            while i + 2 < end + 1 && i + 2 < idx.len() {
                                let (a, b, cc) =
                                    (idx[i] as usize, idx[i + 1] as usize, idx[i + 2] as usize);
                                i += 3;
                                let (Some(&pa), Some(&pb), Some(&pc)) = (
                                    geom.positions.get(a),
                                    geom.positions.get(b),
                                    geom.positions.get(cc),
                                ) else {
                                    continue;
                                };
                                let w0 = aff.transform_point3(Vec3::from_array(pa));
                                let w1 = aff.transform_point3(Vec3::from_array(pb));
                                let w2 = aff.transform_point3(Vec3::from_array(pc));
                                // World face normal from transformed verts (mirror/shear-proof).
                                let n = (w1 - w0).cross(w2 - w0);
                                let len = n.length();
                                if len < 1e-9 {
                                    continue; // degenerate
                                }
                                if n.y / len > HORIZ_MIN {
                                    out.push([w0, w1, w2]);
                                }
                            }
                        }
                    }
                    out
                });
            }
        });

        // Merge.
        let mut tris: Vec<[Vec3; 3]> = Vec::with_capacity(per_thread.iter().map(Vec::len).sum());
        for v in per_thread {
            tris.extend(v);
        }

        if tris.is_empty() {
            return Self {
                origin_xz: Vec2::ZERO,
                inv_cell: 1.0 / CELL,
                nx: 1,
                nz: 1,
                tris,
                cells: vec![Vec::new()],
            };
        }

        // XZ AABB of all kept triangles.
        let (mut min_x, mut min_z, mut max_x, mut max_z) =
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for t in &tris {
            for v in t {
                min_x = min_x.min(v.x);
                min_z = min_z.min(v.z);
                max_x = max_x.max(v.x);
                max_z = max_z.max(v.z);
            }
        }
        let origin_xz = Vec2::new(min_x, min_z);
        let inv_cell = 1.0 / CELL;
        let nx = (((max_x - min_x) * inv_cell).ceil() as u32 + 1).max(1);
        let nz = (((max_z - min_z) * inv_cell).ceil() as u32 + 1).max(1);
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); (nx * nz) as usize];

        // Bucket each triangle index into EVERY cell its XZ AABB overlaps.
        for (ti, t) in tris.iter().enumerate() {
            let (mut tmnx, mut tmnz, mut tmxx, mut tmxz) =
                (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for v in t {
                tmnx = tmnx.min(v.x);
                tmnz = tmnz.min(v.z);
                tmxx = tmxx.max(v.x);
                tmxz = tmxz.max(v.z);
            }
            let cx0 = (((tmnx - min_x) * inv_cell) as i64).clamp(0, nx as i64 - 1);
            let cx1 = (((tmxx - min_x) * inv_cell) as i64).clamp(0, nx as i64 - 1);
            let cz0 = (((tmnz - min_z) * inv_cell) as i64).clamp(0, nz as i64 - 1);
            let cz1 = (((tmxz - min_z) * inv_cell) as i64).clamp(0, nz as i64 - 1);
            for cz in cz0..=cz1 {
                for cx in cx0..=cx1 {
                    cells[(cz as u32 * nx + cx as u32) as usize].push(ti as u32);
                }
            }
        }

        let refs: usize = cells.iter().map(Vec::len).sum();
        info!(
            "walk_ground: {} walkable tris, {}x{} cells ({} index refs), built in {:.0}ms",
            tris.len(),
            nx,
            nz,
            refs,
            t0.elapsed().as_secs_f32() * 1000.0
        );
        Self {
            origin_xz,
            inv_cell,
            nx,
            nz,
            tris,
            cells,
        }
    }

    /// Highest walkable surface Y at (x,z) that is ≤ `feet_y + step_up`. None = no ground (void).
    pub fn ground_height(&self, x: f32, z: f32, feet_y: f32, step_up: f32) -> Option<f32> {
        let cx = ((x - self.origin_xz.x) * self.inv_cell).floor() as i64;
        let cz = ((z - self.origin_xz.y) * self.inv_cell).floor() as i64;
        let cap = feet_y + step_up;
        let mut best: Option<f32> = None;
        // 3×3 block around the cell (seam safety net).
        for dz in -1..=1 {
            for dx in -1..=1 {
                let gx = cx + dx;
                let gz = cz + dz;
                if gx < 0 || gz < 0 || gx >= self.nx as i64 || gz >= self.nz as i64 {
                    continue;
                }
                for &ti in &self.cells[(gz as u32 * self.nx + gx as u32) as usize] {
                    let t = &self.tris[ti as usize];
                    if let Some(y) = tri_height_at(t, x, z) {
                        if y <= cap && best.is_none_or(|b| y > b) {
                            best = Some(y);
                        }
                    }
                }
            }
        }
        best
    }
}

/// Barycentric surface Y of triangle `t` at XZ point (x,z), or None if (x,z) is outside its XZ
/// projection. Uses the standard 2D barycentric on the XZ plane.
fn tri_height_at(t: &[Vec3; 3], x: f32, z: f32) -> Option<f32> {
    let (a, b, c) = (t[0], t[1], t[2]);
    let v0x = b.x - a.x;
    let v0z = b.z - a.z;
    let v1x = c.x - a.x;
    let v1z = c.z - a.z;
    let denom = v0x * v1z - v1x * v0z;
    if denom.abs() < 1e-12 {
        return None; // degenerate in XZ (vertical sliver)
    }
    let inv = 1.0 / denom;
    let px = x - a.x;
    let pz = z - a.z;
    // s,t barycentric of (px,pz) in the (v0,v1) basis.
    let s = (px * v1z - v1x * pz) * inv;
    let w = (v0x * pz - px * v0z) * inv;
    if s < -1e-4 || w < -1e-4 || s + w > 1.0 + 1e-4 {
        return None;
    }
    Some(a.y + s * (b.y - a.y) + w * (c.y - a.y))
}

/// Initial upward velocity for a jump, derived from walk speed so scrolling faster also jumps
/// higher (apex = JUMP_K·walk_speed): vy = sqrt(2·G·JUMP_K·walk_speed).
pub fn jump_velocity(walk_speed: f32) -> f32 {
    (2.0 * GRAVITY * JUMP_K * walk_speed.max(0.0)).sqrt()
}
