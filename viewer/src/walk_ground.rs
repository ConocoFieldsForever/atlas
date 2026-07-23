//! eft::walk_ground — ground-height + wall-collision query for the walk camera (UI_ROADMAP §1c).
//!
//! Free-fly can't "walk the map": there's no floor to stand on and nothing to stop you clipping
//! through walls. This builds two static spatial grids over the pack's world geometry:
//!  - a GROUND grid (near-horizontal faces) so the walk camera follows floors/stairs/terrain and
//!    lands jumps on the RIGHT storey in multi-floor buildings, and
//!  - a WALL grid (steep faces) so a player-sized capsule gets pushed back out of walls/fences.
//!
//! Design (from the read-only design workflow, Approach A):
//!  - Enumerate every WORLD triangle ONCE (mesh-local verts × each instance's affine — NEVER
//!    decompose the transform; use the transformed positions).
//!  - Classify by world face-normal (cross product of transformed edges — mirror/shear-proof):
//!      n.y/|n| > HORIZ_MIN  → GROUND (walkable),
//!      n.y/|n| < WALL_MAX_NY → WALL (collision; skip tiny clutter under WALL_MIN_AREA).
//!    Faces in between (gentle ramps) are neither — you walk them but they never block.
//!  - Bucket each kept triangle's index into every XZ grid cell its footprint overlaps (big slabs
//!    span many cells — insert-by-AABB or you get fall-through seams / tunneling at cell borders).
//!  - `ground_height(x,z, feet_y, step_up)` = the GREATEST surface Y at (x,z) that is ≤ feet_y+step_up.
//!    "Highest surface below the feet (+ a small step allowance)" is what makes multi-floor correct.
//!  - `resolve_walls(pos, feet_y)` = push the body capsule out of any wall triangle it penetrates.
//!
//! Built ONCE (lazily, on first walk activation) off the main thread's critical path via the compute
//! task pool. Per-query cost is O(triangles in a 3×3 cell block), non-allocating, no mesh decode.

use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use crate::eftpack::Pack;

/// Camera eye height above the feet/ground (m).
pub const EYE_HEIGHT: f32 = 1.7;
/// Total body height feet→head for the collision capsule (m).
pub const PLAYER_HEIGHT: f32 = 1.8;
/// Max height the feet may rise to select a surface (mount stairs/curbs, not tabletops) (m).
/// 0.5 clears the taller EFT stair treads / low ledges that 0.45 stopped short of, while staying
/// well under a real wall so you still can't climb waist-high geometry.
pub const STEP_UP: f32 = 0.5;
/// Snappy game-feel gravity (not real 9.8) (m/s²).
pub const GRAVITY: f32 = 20.0;
/// Jump apex height per unit walk-speed (m per m/s) — apex = JUMP_K·walk_speed, so one scroll
/// gesture juices both movement speed and hop height together.
pub const JUMP_K: f32 = 0.12;
/// XZ grid cell size (m).
const CELL: f32 = 3.0;
/// Only faces at least this upward (n.y/|n|) are walkable (~≤60° from horizontal).
const HORIZ_MIN: f32 = 0.5;
/// Faces steeper than this (n.y/|n| below it) are WALLS — collision, not ground (~≥68° from flat).
const WALL_MAX_NY: f32 = 0.38;
/// Skip wall triangles smaller than this (m²): trims tiny clutter/railings so the wall grid stays
/// affordable; real walls/fences/barriers are large quads and survive.
const WALL_MIN_AREA: f32 = 0.04;
/// Player collision radius (m) — horizontal half-width of the body capsule.
pub const PLAYER_RADIUS: f32 = 0.32;
/// If the camera falls this far below the last known ground, treat as fell-through-world.
pub const KILL_DROP: f32 = 60.0;
/// Head-bob: peak vertical eye offset (m) and phase advance (rad per metre walked). Deliberately
/// subtle — a faint footstep cadence, not a seasick lurch.
pub const BOB_AMP: f32 = 0.03;
pub const BOB_RATE: f32 = 6.5;

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
    /// Head-bob phase (radians), advanced by distance walked while grounded.
    pub bob_phase: f32,
    /// The cosmetic head-bob Y offset applied last frame — removed at the start of the next frame so
    /// the bob never feeds back into the ground/step physics (which run on the un-bobbed eye height).
    pub last_bob: f32,
}

/// One XZ-bucketed triangle grid. `tris` stores each world triangle ONCE (36 B); `cells` holds only
/// u32 indices into it, so a slab spanning many cells duplicates the index, not the triangle.
struct Grid {
    origin_xz: Vec2,
    inv_cell: f32,
    nx: u32,
    nz: u32,
    tris: Vec<[Vec3; 3]>,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    fn empty() -> Self {
        Self { origin_xz: Vec2::ZERO, inv_cell: 1.0 / CELL, nx: 1, nz: 1, tris: Vec::new(), cells: vec![Vec::new()] }
    }

    /// Bucket a set of world triangles into an XZ grid by AABB footprint.
    fn from_tris(tris: Vec<[Vec3; 3]>) -> Self {
        if tris.is_empty() {
            return Self::empty();
        }
        let (mut min_x, mut min_z, mut max_x, mut max_z) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
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
        for (ti, t) in tris.iter().enumerate() {
            let (mut tmnx, mut tmnz, mut tmxx, mut tmxz) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
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
        Self { origin_xz, inv_cell, nx, nz, tris, cells }
    }

    /// Cell coords for an XZ point (may be out of range).
    #[inline]
    fn cell_of(&self, x: f32, z: f32) -> (i64, i64) {
        (
            ((x - self.origin_xz.x) * self.inv_cell).floor() as i64,
            ((z - self.origin_xz.y) * self.inv_cell).floor() as i64,
        )
    }
}

/// Prebuilt walk queries: a ground grid (stand/jump) and a wall grid (collision).
#[derive(Resource)]
pub struct GroundGrid {
    ground: Grid,
    walls: Grid,
}

impl GroundGrid {
    /// Enumerate + classify + bucket every world triangle. Parallel over instances, single pass.
    pub fn build(pack: &Pack) -> Self {
        let t0 = std::time::Instant::now();
        let instances = &pack.instances;
        let pool = ComputeTaskPool::get();
        let threads = pool.thread_num().max(1);
        let chunk = instances.len().div_ceil(threads).max(1);

        // Phase 1 (parallel): each worker collects (ground, wall) world triangles.
        let per_thread: Vec<(Vec<[Vec3; 3]>, Vec<[Vec3; 3]>)> = pool.scope(|s| {
            for (ci, c) in instances.chunks(chunk).enumerate() {
                let base = ci * chunk;
                s.spawn(async move {
                    let mut ground: Vec<[Vec3; 3]> = Vec::new();
                    let mut walls: Vec<[Vec3; 3]> = Vec::new();
                    for (j, inst) in c.iter().enumerate() {
                        // All-LOD pack: only the default shell contributes collision triangles
                        // (else the ground/wall grids double-count overlapping shells).
                        if !pack.is_default_lod(base + j) {
                            continue;
                        }
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
                                let (Some(pa), Some(pb), Some(pc)) = (
                                    geom.positions.get(a),
                                    geom.positions.get(b),
                                    geom.positions.get(cc),
                                ) else {
                                    continue;
                                };
                                let w0 = aff.transform_point3(Vec3::from_array(*pa));
                                let w1 = aff.transform_point3(Vec3::from_array(*pb));
                                let w2 = aff.transform_point3(Vec3::from_array(*pc));
                                // World face normal from transformed verts (mirror/shear-proof).
                                let n = (w1 - w0).cross(w2 - w0);
                                let len = n.length();
                                if len < 1e-9 {
                                    continue; // degenerate
                                }
                                let ny = n.y / len;
                                if ny > HORIZ_MIN {
                                    ground.push([w0, w1, w2]);
                                } else if ny.abs() < WALL_MAX_NY && len * 0.5 >= WALL_MIN_AREA {
                                    // Steep enough to be a wall, big enough to matter. (len = 2·area.)
                                    walls.push([w0, w1, w2]);
                                }
                            }
                        }
                    }
                    (ground, walls)
                });
            }
        });

        // Merge each class.
        let mut ground_tris: Vec<[Vec3; 3]> =
            Vec::with_capacity(per_thread.iter().map(|(g, _)| g.len()).sum());
        let mut wall_tris: Vec<[Vec3; 3]> =
            Vec::with_capacity(per_thread.iter().map(|(_, w)| w.len()).sum());
        for (g, w) in per_thread {
            ground_tris.extend(g);
            wall_tris.extend(w);
        }

        let ground = Grid::from_tris(ground_tris);
        let walls = Grid::from_tris(wall_tris);
        info!(
            "walk_ground: {} walkable + {} wall tris, ground {}x{} / walls {}x{} cells, built in {:.0}ms",
            ground.tris.len(),
            walls.tris.len(),
            ground.nx,
            ground.nz,
            walls.nx,
            walls.nz,
            t0.elapsed().as_secs_f32() * 1000.0
        );
        Self { ground, walls }
    }

    /// Highest walkable surface Y at (x,z) that is ≤ `feet_y + step_up`. None = no ground (void).
    pub fn ground_height(&self, x: f32, z: f32, feet_y: f32, step_up: f32) -> Option<f32> {
        let g = &self.ground;
        let (cx, cz) = g.cell_of(x, z);
        let cap = feet_y + step_up;
        let mut best: Option<f32> = None;
        // 3×3 block around the cell (seam safety net).
        for dz in -1..=1 {
            for dx in -1..=1 {
                let gx = cx + dx;
                let gz = cz + dz;
                if gx < 0 || gz < 0 || gx >= g.nx as i64 || gz >= g.nz as i64 {
                    continue;
                }
                for &ti in &g.cells[(gz as u32 * g.nx + gx as u32) as usize] {
                    let t = &g.tris[ti as usize];
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

    /// Push the body capsule (vertical segment at XZ=p.xz, feet_y → feet_y+PLAYER_HEIGHT, radius
    /// PLAYER_RADIUS) out of any wall triangle it penetrates. Returns the corrected XZ.
    /// Correction is purely horizontal so a slightly-tilted wall never launches or sinks the player.
    pub fn resolve_walls(&self, p: Vec3, feet_y: f32) -> Vec2 {
        let g = &self.walls;
        if g.tris.is_empty() {
            return Vec2::new(p.x, p.z);
        }
        let mut px = p.x;
        let mut pz = p.z;
        // Capsule samples: shins, waist, head. Shin sample sits above STEP_UP so low curbs the
        // ground query already steps onto don't read as walls.
        let samples = [feet_y + STEP_UP + 0.1, feet_y + 1.0, feet_y + PLAYER_HEIGHT - 0.15];
        // Two relaxation passes so inside-corners (two walls at once) settle.
        for _ in 0..2 {
            let (cx, cz) = g.cell_of(px, pz);
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let gx = cx + dx;
                    let gz = cz + dz;
                    if gx < 0 || gz < 0 || gx >= g.nx as i64 || gz >= g.nz as i64 {
                        continue;
                    }
                    for &ti in &g.cells[(gz as u32 * g.nx + gx as u32) as usize] {
                        let t = &g.tris[ti as usize];
                        for &sy in &samples {
                            let s = Vec3::new(px, sy, pz);
                            let c = closest_point_on_tri(s, t);
                            let d = s - c;
                            let dist = d.length();
                            if dist < PLAYER_RADIUS {
                                let hlen = (d.x * d.x + d.z * d.z).sqrt();
                                if hlen > 1e-4 {
                                    let push = PLAYER_RADIUS - dist;
                                    px += d.x / hlen * push;
                                    pz += d.z / hlen * push;
                                }
                            }
                        }
                    }
                }
            }
        }
        Vec2::new(px, pz)
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

/// Closest point on triangle `t` to point `p` (Ericson, Real-Time Collision Detection §5.1.5).
fn closest_point_on_tri(p: Vec3, t: &[Vec3; 3]) -> Vec3 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // vertex region A
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // vertex region B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v; // edge AB
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // vertex region C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w; // edge AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w; // edge BC
    }
    // Interior: barycentric projection.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Initial upward velocity for a jump, derived from walk speed so scrolling faster also jumps
/// higher (apex = JUMP_K·walk_speed): vy = sqrt(2·G·JUMP_K·walk_speed).
pub fn jump_velocity(walk_speed: f32) -> f32 {
    (2.0 * GRAVITY * JUMP_K * walk_speed.max(0.0)).sqrt()
}
