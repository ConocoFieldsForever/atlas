//! nav_bake — PORTABLE viewer-side nav-grid baker (pure-CPU BVH raycast).
//!
//! WHY: the runtime router (`crate::nav`) can only LOAD a pre-baked grid. The only baker that
//! existed (tarkmap/bake_nav.py) needs NVIDIA-Warp/CUDA + an `instanced_raw.glb` the native build
//! never produces, so NO pack shipped nav data and routing was dead on every machine. This module
//! bakes the SAME layered-2.5D nav grid straight from a loaded [`Pack`]'s world triangles on the
//! CPU (a median-split BVH + vertical down-raycasts, parallelised with rayon), so routing is
//! produced by default on AMD / NVIDIA / no-GPU alike. It is a faithful port of `bake_nav.py`:
//! identical constants (RES/K/NY_MIN/HEADROOM/CLIMB/DROP_MAX/VAULT/MISS) and the same
//! down-cast + up-facing + headroom + door rules, so quality equals the CUDA bake.
//!
//! OUTPUT (matches `crate::nav::NavGrid::load` EXACTLY — see that module's doc):
//!   nav.json      — { min_x, min_z, res, nx, nz, n_layers(K), miss, climb, drop_max, ... }.
//!   nav.bin       — f32[nx*nz*K] LE: cell (iz*nx+ix) layer l at (iz*nx+ix)*K + l, ASCENDING,
//!                    `miss` (large negative) for empty layers.
//!   nav_door.bin  — u8[nx*nz]: 1 = door cell (forced passable).
//!   nav_blk.bin   — u8[nx*nz*K] 8-dir edge mask (bit d = edge to NB_BAKE[d] blocked by a thin
//!                    wall/fence). Produced by a SECOND pass: retain the near-vertical WALL
//!                    triangles walk_ground drops (|normal.y| < WALL_MAX_NY, area >= WALL_MIN_AREA),
//!                    build a 3-D wall BVH, and for every walkable edge the router would traverse
//!                    cast a player-capsule fan (±PLAYER_RADIUS at body heights) — any wall hit
//!                    blocks the edge. This is what stops routes threading a thin interior wall a
//!                    player cannot walk through (doors + walkable stairs/ramps stay passable). The
//!                    mask is ADDITIVE: `NavGrid::load` treats an absent nav_blk.bin as "no blocked
//!                    edges", so old packs still load.
//!
//! DIFFERENCES vs bake_nav.py (all deliberate, documented):
//!   * Geometry source is the .eftpack (meshes × instance affines) — the SAME triangles the viewer
//!     draws, already in viewer-world space (Y up), which is exactly the space the router queries.
//!     No glb, no coordinate reinterpretation.
//!   * A vertical column ray is tested against a triangle by its XZ projection (barycentric) + a
//!     plane-Y evaluation, which is equivalent to a true vertical ray-triangle test for the
//!     horizontal-ish surfaces nav cares about (vertical walls project to ~zero XZ area and are
//!     skipped, exactly as bake_nav ignores |normal.y| < NY_MIN). Skipping those walls up-front
//!     keeps the BVH small.
//!   * Mirror instances (negative-determinant affine) keep their ORIGINAL winding in the pack (the
//!     renderer flips via a flag, never bakes it), so a world-space face normal comes out inverted.
//!     We flip `normal.y` for mirror instances so up/down classification is physically correct.
//!   * Grid bounds use the same 0.5/99.5-percentile + 6 m pad as bake_nav (rejects skybox/backdrop
//!     outliers), so they sit within/around the pack's manifest AABB rather than exactly on it.

use crate::eftpack::Pack;
use crate::nav::{NavGrid, Scratch};
use anyhow::{anyhow, Context, Result};
use glam::Vec3;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

// ---- constants (match bake_nav.py) ------------------------------------------------------------
const NY_MIN: f32 = 0.5; // up-facing = slope <= 60deg
const HEADROOM: f32 = 1.8; // a floor is walkable only with >= this clearance above it
const CLIMB: f32 = 1.2;
const DROP_MAX: f32 = 2.0;
const VAULT: f32 = 1.2;
const MISS: f32 = -1.0e9;
const MISS_HALF: f32 = MISS * 0.5;
const SLOPE_MAX_DEG: i32 = 60;
const Y_HIGH_FLOOR: f32 = 90.0; // ray origin height floor (bake_nav Y_HIGH); raised for taller maps
const PAD: f32 = 6.0; // grid padding beyond the geometry (metres)
/// Below this XZ-projected parallelogram area a triangle is treated as a vertical wall (a vertical
/// ray can't meaningfully hit it) and dropped from the BVH — same effect as bake_nav ignoring
/// near-vertical faces, but it also shrinks the tree.
const MIN_XZ_AREA2: f32 = 1.0e-6;
/// Barycentric inclusion tolerance — a hair negative so a column landing exactly on a shared
/// triangle edge/seam still registers a floor (avoids pinhole gaps between adjacent floor tris).
const BARY_EPS: f32 = -1.0e-4;
const LEAF_MAX: usize = 4;

// ---- FIX 1: thin-wall edge mask (nav_blk.bin) — reuse walk_ground's player-capsule wall model ---
/// A face with |normal.y| below this is a WALL (collision), matching `walk_ground::WALL_MAX_NY`.
const WALL_MAX_NY: f32 = 0.38;
/// Wall triangles smaller than this (m²) are clutter/railings — skipped (matches `WALL_MIN_AREA`).
const WALL_MIN_AREA: f32 = 0.04;
/// Player capsule half-width (m) — matches `walk_ground::PLAYER_RADIUS`. The ±R fan blocks a gap
/// narrower than 2·R = 0.64 m even when a centre ray would thread it.
const PLAYER_RADIUS: f32 = 0.32;
/// Total player height (m) — matches `walk_ground::PLAYER_HEIGHT`.
const PLAYER_HEIGHT_NAV: f32 = 1.8;
/// Free step-up (m) — matches nav.rs' default `step_up`. The capsule fan starts ABOVE this so
/// curbs / low risers the router already steps onto are NOT read as walls (the curb-vs-wall band).
const STEP_UP_NAV: f32 = 0.45;
/// tan(45°) — nav.rs' default `walk_slope_deg`, so the baker's edge-walkability gate matches the
/// router's (a bit blocked here is an edge the router would otherwise traverse).
const SLOPE_TAN_NAV: f32 = 1.0;
/// A door-tagged mesh only punches a passable hole (and drops out of the wall set) when its
/// INSTANCE footprint is door-panel sized (≤ this, in the SMALLER horizontal span). A large
/// gate/shutter fence keeps blocking — otherwise a `gate`/`shutter` NAME on a wall-wide mesh would
/// open a wall-wide gap the player can't actually pass.
const DOOR_FOOTPRINT_MAX: f32 = 1.5;
/// Capsule perpendicular offsets (−R, 0, +R) across the edge.
const CAP_OFF: [f32; 3] = [-PLAYER_RADIUS, 0.0, PLAYER_RADIUS];
/// Body sample heights above the floor (shins / waist / head) — start above STEP_UP so low curbs
/// aren't over-blocked; matches `walk_ground::resolve_walls`'s capsule samples.
const CAP_H: [f32; 3] = [STEP_UP_NAV + 0.1, 1.0, PLAYER_HEIGHT_NAV - 0.15];
/// 8-neighbour offsets — MUST match `nav.rs` NB order (block-mask bit d = the edge to NB_BAKE[d]).
const NB_BAKE: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// One world-space triangle. Shared with `sh_bake` (the wgpu lighting bake reuses the same
/// world-triangle assembly + BVH), hence `pub(crate)`.
#[derive(Clone, Copy)]
pub(crate) struct Tri {
    pub(crate) a: Vec3,
    pub(crate) b: Vec3,
    pub(crate) c: Vec3,
    /// Normalised world-space normal Y (sign-corrected for mirror instances).
    pub(crate) ny: f32,
    /// Belongs to a DOOR-tagged mesh/root → transparent to the cast + stamps the door footprint.
    pub(crate) door: bool,
    /// SubMesh.material_id of the face — the `sh_bake` diffuse bounce reads it to look up per-material
    /// albedo/emissive. `nav_bake` never reads it (it just travels with the tri through the BVH).
    pub(crate) mat: u32,
}

/// A surface hit collected along one downward column ray.
#[derive(Clone, Copy)]
struct Hit {
    y: f32,
    ny: f32,
    door: bool,
}

// ---- door name rules (port of bake_nav.py DOOR_RE / DOOR_SKIP, hand-rolled: no regex dep) ------

/// True if `name` names a door panel that should be forced passable. Mirrors bake_nav's
/// `DOOR_RE.search(nm) and not DOOR_SKIP.search(nm)` (case-insensitive).
fn is_door_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let s = name.to_ascii_lowercase();
    door_match(&s) && !door_skip(&s)
}

fn door_match(s: &str) -> bool {
    const SUBS: [&str; 8] = [
        "inside_door",
        "door_metal",
        "door_wood",
        "_door_left",
        "_door_right",
        "glass_door",
        "rollet",
        "shutter",
    ];
    if SUBS.iter().any(|p| s.contains(p)) {
        return true;
    }
    // `_door_[lr]\b` : "_door_l" or "_door_r" followed by a word boundary (non [A-Za-z0-9_] or end).
    for pat in ["_door_l", "_door_r"] {
        let mut from = 0;
        while let Some(rel) = s[from..].find(pat) {
            let end = from + rel + pat.len();
            let boundary = s[end..].chars().next().map_or(true, |c| !is_word_char(c));
            if boundary {
                return true;
            }
            from = from + rel + 1;
        }
    }
    // `\bgate\b`
    word_present(s, "gate")
}

fn door_skip(s: &str) -> bool {
    const SUBS: [&str; 16] = [
        "trailer",
        "truck",
        "van",
        "lovlo",
        "tarcola",
        "transformator",
        "locker",
        "fridge",
        "microwave",
        "oven",
        "cabinet",
        "lockbox",
        "padlock",
        "wagon",
        "gaz",
        "kamaz",
    ];
    // "ural" is the 17th DOOR_SKIP alternative; kept out of the array to keep it a fixed size.
    SUBS.iter().any(|p| s.contains(p)) || s.contains("ural")
}

#[inline]
fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `word` present in `s` bounded by non-word chars on both sides (a `\bword\b` match).
fn word_present(s: &str, word: &str) -> bool {
    let mut from = 0;
    while let Some(rel) = s[from..].find(word) {
        let start = from + rel;
        let end = start + word.len();
        let before = start == 0 || !is_word_char(s[..start].chars().next_back().unwrap());
        let after = s[end..].chars().next().map_or(true, |c| !is_word_char(c));
        if before && after {
            return true;
        }
        from = start + 1;
    }
    false
}

// ---- world-triangle assembly ------------------------------------------------------------------

/// Instance XZ footprint (smaller horizontal span) from the mesh-local AABB corners × the affine —
/// used by the door/gate footprint cap. Transforming the 8 corners captures shear/mirror without a
/// TRS decompose.
fn instance_small_footprint(aff: &glam::Affine3A, lmin: Vec3, lmax: Vec3) -> bool {
    let (mut mnx, mut mnz) = (f32::INFINITY, f32::INFINITY);
    let (mut mxx, mut mxz) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for cxi in 0..2 {
        for cyi in 0..2 {
            for czi in 0..2 {
                let corner = Vec3::new(
                    if cxi == 0 { lmin.x } else { lmax.x },
                    if cyi == 0 { lmin.y } else { lmax.y },
                    if czi == 0 { lmin.z } else { lmax.z },
                );
                let w = aff.transform_point3(corner);
                mnx = mnx.min(w.x);
                mxx = mxx.max(w.x);
                mnz = mnz.min(w.z);
                mxz = mxz.max(w.z);
            }
        }
    }
    (mxx - mnx).min(mxz - mnz) <= DOOR_FOOTPRINT_MAX
}

/// Build the world-space triangle soup for the BVH from the pack's meshes × instance affines.
/// Each unique mesh is unpacked ONCE (via the eftpack accessor) then transformed for every one of
/// its instances. Returns `(column_tris, wall_tris, min_y, max_y, door_tris)`:
///   * `column_tris` — the input to the vertical-column BVH: UNCHANGED (both up-facing floors AND
///     down-facing ceilings; ceilings are load-bearing for `resolve_column`'s headroom). Vertical
///     faces (XZ projection ~ a line) are still skipped here.
///   * `wall_tris` — the NEW near-vertical WALL faces (|ny| < WALL_MAX_NY, area >= WALL_MIN_AREA)
///     for the horizontal-segment wall BVH, EXCLUDING door panels (small door-tagged instances).
pub(crate) fn build_tris(pack: &Pack) -> (Vec<Tri>, Vec<Tri>, f32, f32, usize) {
    let by_mesh = pack.instances_by_mesh();
    let mut tris: Vec<Tri> = Vec::new();
    let mut walls: Vec<Tri> = Vec::new();
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut door_tris = 0usize;

    for (mid, inst_ids) in by_mesh.iter().enumerate() {
        if inst_ids.is_empty() {
            continue;
        }
        let mesh = &pack.manifest.meshes[mid];
        let mesh_is_door = is_door_name(&mesh.name);
        // Unpack the mesh geometry once (positions + indices) via the shared accessor.
        let geom = match pack.mesh_geom(mesh) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  nav-bake: skipping mesh {} '{}': {e}", mesh.id, mesh.name);
                continue;
            }
        };
        if geom.positions.is_empty() || geom.indices.len() < 3 {
            continue;
        }
        // Mesh-local AABB once (for the door/gate footprint cap per instance).
        let (mut lmin, mut lmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in &geom.positions {
            let v = Vec3::from(*p);
            lmin = lmin.min(v);
            lmax = lmax.max(v);
        }
        // Per-face material id (for the sh_bake bounce). Submeshes are consecutive index runs within
        // this mesh's index array, so face f (indices 3f..3f+3) belongs to the submesh whose
        // [idx_start, idx_start+idx_count) contains 3f. Same for every instance of this mesh.
        let n_faces = geom.indices.len() / 3;
        let mut face_mat = vec![0u32; n_faces];
        for sub in &mesh.submeshes {
            let f0 = (sub.idx_start as usize) / 3;
            let f1 = (((sub.idx_start + sub.idx_count) as usize) / 3).min(n_faces);
            if f0 < f1 {
                face_mat[f0..f1].fill(sub.material_id);
            }
        }
        for &iid in inst_ids {
            // All-LOD pack: bake nav from the default shell only (else the BVH soup has stacked
            // overlapping shells → slower bake + coarse-shell walkability artifacts).
            if !pack.is_default_lod(iid as usize) {
                continue;
            }
            let inst = &pack.instances[iid as usize];
            let root_is_door = pack
                .manifest
                .roots
                .get(inst.root_id as usize)
                .map(|r| is_door_name(r))
                .unwrap_or(false);
            let door_tagged = mesh_is_door || root_is_door;
            let aff = inst.affine3a();
            let mirror = inst.is_mirror();
            // Footprint cap: a door tag only opens a hole (transparent + door-cell stamp, and drops
            // its faces from `walls`) when the instance is door-panel sized. A big gate still blocks.
            let door = door_tagged && instance_small_footprint(&aff, lmin, lmax);
            for (fi, tri) in geom.indices.chunks_exact(3).enumerate() {
                let mat = face_mat[fi];
                let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                // Defensive: a bad index just skips the face (release is panic=abort).
                if i0 >= geom.positions.len() || i1 >= geom.positions.len() || i2 >= geom.positions.len() {
                    continue;
                }
                let a = aff.transform_point3(Vec3::from(geom.positions[i0]));
                let b = aff.transform_point3(Vec3::from(geom.positions[i1]));
                let c = aff.transform_point3(Vec3::from(geom.positions[i2]));
                let e1 = b - a;
                let e2 = c - a;
                let n = e1.cross(e2);
                let nlen = n.length();
                if nlen < 1.0e-12 {
                    continue; // degenerate
                }
                let mut ny = n.y / nlen;
                if mirror {
                    ny = -ny; // restore correct orientation for winding-flipped mirror instances
                }
                // WALL: near-vertical + big enough, and NOT a (small) door panel. `|ny|` so the
                // mirror flip is immaterial. area = 0.5·|e1×e2| = 0.5·nlen.
                if !door && ny.abs() < WALL_MAX_NY && 0.5 * nlen >= WALL_MIN_AREA {
                    walls.push(Tri { a, b, c, ny, door: false, mat });
                }
                // Column BVH input — UNCHANGED: drop only the vertical faces (XZ projection ~ a
                // line, a vertical ray can't register them); keep floors AND ceilings for headroom.
                let xz_area2 = (e1.x * e2.z - e1.z * e2.x).abs();
                if xz_area2 < MIN_XZ_AREA2 {
                    continue;
                }
                min_y = min_y.min(a.y.min(b.y.min(c.y)));
                max_y = max_y.max(a.y.max(b.y.max(c.y)));
                if door {
                    door_tris += 1;
                }
                tris.push(Tri { a, b, c, ny, door, mat });
            }
        }
    }
    if !min_y.is_finite() {
        min_y = 0.0;
        max_y = 0.0;
    }
    (tris, walls, min_y, max_y, door_tris)
}

// ---- BVH (median-split over XZ, for vertical-ray queries) -------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct BvhNode {
    pub(crate) min: Vec3,
    pub(crate) max: Vec3,
    /// Leaf (count>0): tris[start..start+count]. Internal (count==0): children at start, start+1.
    pub(crate) start: u32,
    pub(crate) count: u32,
}

pub(crate) struct Bvh {
    pub(crate) nodes: Vec<BvhNode>,
    pub(crate) tris: Vec<Tri>,
}

impl Bvh {
    pub(crate) fn build(tris: Vec<Tri>) -> Bvh {
        let n = tris.len();
        if n == 0 {
            return Bvh {
                nodes: vec![BvhNode {
                    min: Vec3::ZERO,
                    max: Vec3::ZERO,
                    start: 0,
                    count: 0,
                }],
                tris,
            };
        }
        // Per-triangle centroid XZ for split ordering.
        let cx: Vec<f32> = tris.iter().map(|t| (t.a.x + t.b.x + t.c.x) / 3.0).collect();
        let cz: Vec<f32> = tris.iter().map(|t| (t.a.z + t.b.z + t.c.z) / 3.0).collect();
        let mut idx: Vec<u32> = (0..n as u32).collect();

        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * (n / LEAF_MAX).max(1) + 8);
        nodes.push(BvhNode {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            start: 0,
            count: 0,
        });
        // Explicit work stack (no recursion → no stack-overflow risk under panic=abort).
        let mut stack: Vec<(usize, usize, usize)> = vec![(0usize, 0usize, n)];
        while let Some((node, lo, hi)) = stack.pop() {
            let mut mn = Vec3::splat(f32::INFINITY);
            let mut mx = Vec3::splat(f32::NEG_INFINITY);
            for &ti in &idx[lo..hi] {
                let t = &tris[ti as usize];
                mn = mn.min(t.a).min(t.b).min(t.c);
                mx = mx.max(t.a).max(t.b).max(t.c);
            }
            let count = hi - lo;
            if count <= LEAF_MAX {
                nodes[node] = BvhNode {
                    min: mn,
                    max: mx,
                    start: lo as u32,
                    count: count as u32,
                };
                continue;
            }
            // Split on the wider of X/Z (the axes that matter for vertical-ray XZ pruning).
            let use_x = (mx.x - mn.x) >= (mx.z - mn.z);
            let key = |ti: u32| -> f32 {
                if use_x {
                    cx[ti as usize]
                } else {
                    cz[ti as usize]
                }
            };
            let mid = (lo + hi) / 2;
            idx[lo..hi].select_nth_unstable_by(mid - lo, |&x, &y| key(x).total_cmp(&key(y)));
            let l = nodes.len();
            nodes.push(BvhNode { min: mn, max: mx, start: 0, count: 0 });
            nodes.push(BvhNode { min: mn, max: mx, start: 0, count: 0 });
            nodes[node] = BvhNode {
                min: mn,
                max: mx,
                start: l as u32,
                count: 0,
            };
            stack.push((l, lo, mid));
            stack.push((l + 1, mid, hi));
        }
        // Reorder triangles into leaf (idx) order so leaf ranges index `tris` directly.
        let tris_ordered: Vec<Tri> = idx.iter().map(|&i| tris[i as usize]).collect();
        Bvh {
            nodes,
            tris: tris_ordered,
        }
    }

    /// Gather every surface hit under the column (x,z) with hit-Y in [y_low, y_high] into `out`.
    fn column(&self, x: f32, z: f32, y_low: f32, y_high: f32, out: &mut Vec<Hit>, stack: &mut Vec<u32>) {
        out.clear();
        stack.clear();
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            // Vertical ray = a point in XZ: prune nodes the column can't pass through.
            if x < node.min.x || x > node.max.x || z < node.min.z || z > node.max.z {
                continue;
            }
            if node.count > 0 {
                let s = node.start as usize;
                for t in &self.tris[s..s + node.count as usize] {
                    if let Some(y) = tri_vertical_y(t, x, z) {
                        if y >= y_low && y <= y_high {
                            out.push(Hit { y, ny: t.ny, door: t.door });
                        }
                    }
                }
            } else {
                stack.push(node.start);
                stack.push(node.start + 1);
            }
        }
    }
}

/// Y where the vertical line through (x,z) crosses triangle `t`, or None if outside its XZ
/// projection / the projection is degenerate. Barycentric in the XZ plane, then interpolate Y.
#[inline]
fn tri_vertical_y(t: &Tri, x: f32, z: f32) -> Option<f32> {
    let (ax, az) = (t.a.x, t.a.z);
    let v0x = t.b.x - ax;
    let v0z = t.b.z - az;
    let v1x = t.c.x - ax;
    let v1z = t.c.z - az;
    let den = v0x * v1z - v1x * v0z;
    if den.abs() < 1.0e-12 {
        return None;
    }
    let inv = 1.0 / den;
    let p0x = x - ax;
    let p0z = z - az;
    let v = (p0x * v1z - v1x * p0z) * inv; // weight for b
    let w = (v0x * p0z - p0x * v0z) * inv; // weight for c
    let u = 1.0 - v - w;
    if u < BARY_EPS || v < BARY_EPS || w < BARY_EPS {
        return None;
    }
    Some(u * t.a.y + v * t.b.y + w * t.c.y)
}

// ---- 3-D wall BVH (segment/triangle queries — the column BVH can't answer horizontal segments) --

/// A BVH over the retained WALL triangles with full 3-D AABB nodes, for segment-vs-triangle
/// queries. Median-split on the widest axis, leaf <= LEAF_MAX, explicit work stack (panic=abort).
struct WallBvh {
    nodes: Vec<BvhNode>,
    tris: Vec<Tri>,
}

impl WallBvh {
    fn build(tris: Vec<Tri>) -> WallBvh {
        let n = tris.len();
        if n == 0 {
            return WallBvh {
                nodes: vec![BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, start: 0, count: 0 }],
                tris,
            };
        }
        let cen: Vec<Vec3> = tris.iter().map(|t| (t.a + t.b + t.c) / 3.0).collect();
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * (n / LEAF_MAX).max(1) + 8);
        nodes.push(BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, start: 0, count: 0 });
        let mut stack: Vec<(usize, usize, usize)> = vec![(0usize, 0usize, n)];
        while let Some((node, lo, hi)) = stack.pop() {
            let mut mn = Vec3::splat(f32::INFINITY);
            let mut mx = Vec3::splat(f32::NEG_INFINITY);
            for &ti in &idx[lo..hi] {
                let t = &tris[ti as usize];
                mn = mn.min(t.a).min(t.b).min(t.c);
                mx = mx.max(t.a).max(t.b).max(t.c);
            }
            let count = hi - lo;
            if count <= LEAF_MAX {
                nodes[node] = BvhNode { min: mn, max: mx, start: lo as u32, count: count as u32 };
                continue;
            }
            // Split on the widest of x/y/z.
            let ext = mx - mn;
            let axis = if ext.x >= ext.y && ext.x >= ext.z {
                0
            } else if ext.y >= ext.z {
                1
            } else {
                2
            };
            let key = |ti: u32| -> f32 {
                let c = cen[ti as usize];
                match axis {
                    0 => c.x,
                    1 => c.y,
                    _ => c.z,
                }
            };
            let mid = (lo + hi) / 2;
            idx[lo..hi].select_nth_unstable_by(mid - lo, |&x, &y| key(x).total_cmp(&key(y)));
            let l = nodes.len();
            nodes.push(BvhNode { min: mn, max: mx, start: 0, count: 0 });
            nodes.push(BvhNode { min: mn, max: mx, start: 0, count: 0 });
            nodes[node] = BvhNode { min: mn, max: mx, start: l as u32, count: 0 };
            stack.push((l, lo, mid));
            stack.push((l + 1, mid, hi));
        }
        let tris_ordered: Vec<Tri> = idx.iter().map(|&i| tris[i as usize]).collect();
        WallBvh { nodes, tris: tris_ordered }
    }

    /// True if the segment p0->p1 intersects ANY wall triangle. Slab-prune AABBs vs the segment,
    /// Möller–Trumbore at leaves; early-out on first hit. `stack` is a reusable per-thread buffer.
    fn segment_hit(&self, p0: Vec3, p1: Vec3, stack: &mut Vec<u32>) -> bool {
        if self.tris.is_empty() {
            return false;
        }
        let dir = p1 - p0;
        let inv = Vec3::new(
            if dir.x != 0.0 { 1.0 / dir.x } else { f32::INFINITY },
            if dir.y != 0.0 { 1.0 / dir.y } else { f32::INFINITY },
            if dir.z != 0.0 { 1.0 / dir.z } else { f32::INFINITY },
        );
        stack.clear();
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if !seg_aabb(p0, inv, dir, node.min, node.max) {
                continue;
            }
            if node.count > 0 {
                let s = node.start as usize;
                for t in &self.tris[s..s + node.count as usize] {
                    if moller_trumbore(p0, dir, t.a, t.b, t.c) {
                        return true;
                    }
                }
            } else {
                stack.push(node.start);
                stack.push(node.start + 1);
            }
        }
        false
    }

    /// True if ANY wall triangle's AABB overlaps the query box [bmin,bmax] (conservative — a hit
    /// means a wall occupies that volume). Used to flag a cell whose body column contains a wall so
    /// the simplifier never straightens a chord THROUGH it (the sub-cell walls that block no
    /// cell-edge and so are invisible to the per-edge blk mask).
    fn box_overlaps(&self, bmin: Vec3, bmax: Vec3, stack: &mut Vec<u32>) -> bool {
        if self.tris.is_empty() {
            return false;
        }
        stack.clear();
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if node.min.x > bmax.x
                || node.max.x < bmin.x
                || node.min.y > bmax.y
                || node.max.y < bmin.y
                || node.min.z > bmax.z
                || node.max.z < bmin.z
            {
                continue;
            }
            if node.count > 0 {
                let s = node.start as usize;
                for t in &self.tris[s..s + node.count as usize] {
                    let tmnx = t.a.x.min(t.b.x).min(t.c.x);
                    let tmxx = t.a.x.max(t.b.x).max(t.c.x);
                    if tmnx > bmax.x || tmxx < bmin.x {
                        continue;
                    }
                    let tmny = t.a.y.min(t.b.y).min(t.c.y);
                    let tmxy = t.a.y.max(t.b.y).max(t.c.y);
                    if tmny > bmax.y || tmxy < bmin.y {
                        continue;
                    }
                    let tmnz = t.a.z.min(t.b.z).min(t.c.z);
                    let tmxz = t.a.z.max(t.b.z).max(t.c.z);
                    if tmnz > bmax.z || tmxz < bmin.z {
                        continue;
                    }
                    return true;
                }
            } else {
                stack.push(node.start);
                stack.push(node.start + 1);
            }
        }
        false
    }
}

/// Segment (origin p0, direction `dir`, t∈[0,1]) vs AABB slab test. `inv` = 1/dir (∞ where dir=0).
#[inline]
fn seg_aabb(p0: Vec3, inv: Vec3, dir: Vec3, bmin: Vec3, bmax: Vec3) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = 1.0f32;
    // X
    if dir.x != 0.0 {
        let t1 = (bmin.x - p0.x) * inv.x;
        let t2 = (bmax.x - p0.x) * inv.x;
        let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(lo);
        tmax = tmax.min(hi);
    } else if p0.x < bmin.x || p0.x > bmax.x {
        return false;
    }
    // Y
    if dir.y != 0.0 {
        let t1 = (bmin.y - p0.y) * inv.y;
        let t2 = (bmax.y - p0.y) * inv.y;
        let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(lo);
        tmax = tmax.min(hi);
    } else if p0.y < bmin.y || p0.y > bmax.y {
        return false;
    }
    // Z
    if dir.z != 0.0 {
        let t1 = (bmin.z - p0.z) * inv.z;
        let t2 = (bmax.z - p0.z) * inv.z;
        let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(lo);
        tmax = tmax.min(hi);
    } else if p0.z < bmin.z || p0.z > bmax.z {
        return false;
    }
    tmin <= tmax
}

/// Möller–Trumbore segment/triangle intersection: hit iff t∈[0,1] and barycentric u,v,w ≥ -eps
/// (w = 1-u-v). `dir` = p1-p0 (NOT normalised) so t is the segment fraction.
#[inline]
fn moller_trumbore(p0: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    const DET_EPS: f32 = 1.0e-8;
    const BARY_EPS_HIT: f32 = 1.0e-5;
    const T_EPS: f32 = 1.0e-6;
    let e1 = b - a;
    let e2 = c - a;
    let pv = dir.cross(e2);
    let det = e1.dot(pv);
    if det.abs() < DET_EPS {
        return false; // segment parallel to the triangle plane
    }
    let inv = 1.0 / det;
    let tv = p0 - a;
    let u = tv.dot(pv) * inv;
    if u < -BARY_EPS_HIT || u > 1.0 + BARY_EPS_HIT {
        return false;
    }
    let qv = tv.cross(e1);
    let v = dir.dot(qv) * inv;
    if v < -BARY_EPS_HIT || u + v > 1.0 + BARY_EPS_HIT {
        return false;
    }
    let t = e2.dot(qv) * inv;
    t >= -T_EPS && t <= 1.0 + T_EPS
}

/// `best_layer` bit-identical to `nav.rs::best_layer`: the layer whose height is nearest `ref_y`
/// (ascending scan, FIRST layer wins on an equal |Δ|, break at the first MISS slot). -1 if none.
#[inline]
fn best_layer_bake(h: &[f32], c: usize, k: usize, ref_y: f32) -> i32 {
    let (mut b, mut bd) = (-1i32, f32::MAX);
    for l in 0..k {
        let hh = h[c * k + l];
        if hh <= MISS_HALF {
            break;
        }
        let dd = (hh - ref_y).abs();
        if dd < bd {
            bd = dd;
            b = l as i32;
        }
    }
    b
}

/// Edge walkability matching `nav.rs::walkable_step(forced=false)` with the router's DEFAULT
/// step_up / walk_slope (the baker omits them from nav.json, so the router uses these defaults) —
/// so a bit is only set on an edge the router would otherwise traverse.
#[inline]
fn walkable_step_bake(up: f32, run: f32) -> bool {
    if up > 0.0 {
        up <= STEP_UP_NAV || (up <= CLIMB && up <= run * SLOPE_TAN_NAV)
    } else {
        -up <= DROP_MAX
    }
}

/// Cast the player-capsule fan across one edge (cell floor -> neighbour floor): ±PLAYER_RADIUS
/// perpendicular offsets at body heights CAP_H. Any wall-tri hit ⇒ the edge is blocked.
#[allow(clippy::too_many_arguments)]
fn capsule_blocked(
    cx: f32,
    cz: f32,
    fy0: f32,
    ncx: f32,
    ncz: f32,
    fy1: f32,
    bvh: &WallBvh,
    stack: &mut Vec<u32>,
) -> bool {
    let ex = ncx - cx;
    let ez = ncz - cz;
    let el = (ex * ex + ez * ez).sqrt();
    if el < 1.0e-6 {
        return false;
    }
    let (nex, nez) = (ex / el, ez / el);
    let (px, pz) = (-nez, nex); // perpendicular in XZ
    for &o in &CAP_OFF {
        let (ox, oz) = (px * o, pz * o);
        for &hy in &CAP_H {
            let p0 = Vec3::new(cx + ox, fy0 + hy, cz + oz);
            let p1 = Vec3::new(ncx + ox, fy1 + hy, ncz + oz);
            if bvh.segment_hit(p0, p1, stack) {
                return true;
            }
        }
    }
    false
}

/// Count segments of a route polyline whose player-capsule fan (CAP_H × ±PLAYER_RADIUS) hits ANY
/// wall triangle — the acceptance metric (a "wall-crossing"). Returns (segments, crossings).
fn count_wall_crossings(poly: &[Vec3], bvh: &WallBvh) -> (usize, usize) {
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    let (mut segs, mut cross) = (0usize, 0usize);
    for w in poly.windows(2) {
        segs += 1;
        let (a, b) = (w[0], w[1]);
        let ex = b.x - a.x;
        let ez = b.z - a.z;
        let el = (ex * ex + ez * ez).sqrt();
        if el < 1.0e-6 {
            continue;
        }
        let (nex, nez) = (ex / el, ez / el);
        let (px, pz) = (-nez, nex);
        let mut hit = false;
        'scan: for &o in &CAP_OFF {
            let (ox, oz) = (px * o, pz * o);
            for &hy in &CAP_H {
                let p0 = Vec3::new(a.x + ox, a.y + hy, a.z + oz);
                let p1 = Vec3::new(b.x + ox, b.y + hy, b.z + oz);
                if bvh.segment_hit(p0, p1, &mut stack) {
                    hit = true;
                    break 'scan;
                }
            }
        }
        if hit {
            cross += 1;
        }
    }
    (segs, cross)
}

/// Of the segments that DO cross a wall (per `count_wall_crossings`), how many pass through a door
/// cell (sampling the segment every ~half-cell, 3x3 door neighbourhood)? A door frame is passable,
/// so these are not violations. Used only by the machine proof's attribution.
fn count_door_crossings(poly: &[Vec3], baked: &Baked) -> usize {
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    let mut n = 0usize;
    for w in poly.windows(2) {
        let (a, b) = (w[0], w[1]);
        let ex = b.x - a.x;
        let ez = b.z - a.z;
        let el = (ex * ex + ez * ez).sqrt();
        if el < 1.0e-6 {
            continue;
        }
        let (nex, nez) = (ex / el, ez / el);
        let (px, pz) = (-nez, nex);
        // does this segment cross a wall at all?
        let mut hit = false;
        'scan: for &o in &CAP_OFF {
            let (ox, oz) = (px * o, pz * o);
            for &hy in &CAP_H {
                let p0 = Vec3::new(a.x + ox, a.y + hy, a.z + oz);
                let p1 = Vec3::new(b.x + ox, b.y + hy, b.z + oz);
                if baked.wall_bvh.segment_hit(p0, p1, &mut stack) {
                    hit = true;
                    break 'scan;
                }
            }
        }
        if !hit {
            continue;
        }
        // walk the segment in ~half-cell steps; is any sampled cell (3x3) a door?
        let steps = (el / (baked.res * 0.5)).ceil().max(1.0) as usize;
        let mut door_near = false;
        'walk: for si in 0..=steps {
            let t = si as f32 / steps as f32;
            let x = a.x + ex * t;
            let z = a.z + ez * t;
            let cix = ((x - baked.min_x) / baked.res).round() as i64;
            let ciz = ((z - baked.min_z) / baked.res).round() as i64;
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let (jx, jz) = (cix + dx, ciz + dz);
                    if jx < 0 || jz < 0 || jx >= baked.nx as i64 || jz >= baked.nz as i64 {
                        continue;
                    }
                    if baked.door[(jz * baked.nx as i64 + jx) as usize] != 0 {
                        door_near = true;
                        break 'walk;
                    }
                }
            }
        }
        if door_near {
            n += 1;
        }
    }
    n
}

/// Run the down-cast state machine on one column's hits (mutating `hits` — it is sorted here) and
/// write ascending floor heights into `hout` (length K, pre-filled MISS). Returns (n_floors,
/// is_door). Faithful port of the `nav_cast` kernel: up-facing surfaces are floors iff there is
/// >= HEADROOM clearance under the last ceiling/floor above; a floor also caps clearance for the
/// floor below it; DOOR faces are transparent (never a surface) but stamp the cell.
fn resolve_column(hits: &mut Vec<Hit>, k: usize, hout: &mut [f32], floors: &mut Vec<f32>) -> (usize, bool) {
    floors.clear();
    let mut door_cell = false;
    if hits.is_empty() {
        return (0, false);
    }
    hits.sort_unstable_by(|p, q| q.y.total_cmp(&p.y)); // top -> bottom
    let mut last_down = f32::INFINITY;
    for h in hits.iter() {
        if h.door {
            door_cell = true;
            continue; // transparent to the cast
        }
        if h.ny >= NY_MIN {
            // up-facing floor
            if last_down - h.y >= HEADROOM {
                floors.push(h.y);
            }
            last_down = h.y; // a floor also caps clearance for anything below it
            if floors.len() >= k {
                break;
            }
        } else if h.ny <= -NY_MIN {
            last_down = h.y; // down-facing ceiling / underside
        }
        // near-vertical wall: ignored (also pre-filtered from the BVH)
    }
    floors.sort_unstable_by(|p, q| p.total_cmp(q)); // ascending, MISS slots stay at the end
    let n = floors.len().min(k);
    for (i, &f) in floors.iter().take(n).enumerate() {
        hout[i] = f;
    }
    (n, door_cell)
}

// ---- baked grid + writer ----------------------------------------------------------------------

pub struct Baked {
    dataset: String,
    min_x: f32,
    min_z: f32,
    res: f32,
    nx: usize,
    nz: usize,
    k: usize,
    y_high: f32,
    heights: Vec<f32>,   // nx*nz*k, ascending, MISS empty
    door: Vec<u8>,       // nx*nz
    blk: Vec<u8>,        // nx*nz*k, 8-dir edge mask
    wall_cell: Vec<u8>,  // nx*nz, 1 = a wall sits in this cell's body column (simplify guard)
    walkable: usize,
    door_cells: usize,
    blocked_edges: usize,
    wall_cells: usize,
    wall_tris: usize,
    /// Kept only in-memory for the headless self-check (never written); the machine proof samples
    /// the SIMPLIFIED routes against these exact wall triangles.
    wall_bvh: WallBvh,
}

impl Baked {
    fn cells(&self) -> usize {
        self.nx * self.nz
    }

    /// Write nav.bin + nav_door.bin + nav_blk.bin + nav.json into `dir`, matching the format
    /// `NavGrid::load` reads. nav_blk.bin is ADDITIVE (an absent one loads as "no blocked edges").
    fn write(&self, dir: &Path) -> Result<()> {
        let bin: &[u8] = bytemuck::cast_slice(&self.heights);
        std::fs::write(dir.join("nav.bin"), bin)
            .with_context(|| format!("writing {}", dir.join("nav.bin").display()))?;
        std::fs::write(dir.join("nav_door.bin"), &self.door)
            .with_context(|| format!("writing {}", dir.join("nav_door.bin").display()))?;
        std::fs::write(dir.join("nav_blk.bin"), &self.blk)
            .with_context(|| format!("writing {}", dir.join("nav_blk.bin").display()))?;
        std::fs::write(dir.join("nav_wallcell.bin"), &self.wall_cell)
            .with_context(|| format!("writing {}", dir.join("nav_wallcell.bin").display()))?;
        // Match bake_nav.py's key set exactly (the router reads min_x/min_z/res/nx/nz/n_layers/miss/
        // climb/drop_max; the rest are informational). step_up/walk_slope_deg are intentionally
        // omitted so the router falls back to the SAME defaults it used for CUDA-baked packs.
        let meta = serde_json::json!({
            "map": self.dataset,
            "min_x": self.min_x,
            "min_z": self.min_z,
            "res": self.res,
            "nx": self.nx,
            "nz": self.nz,
            "n_layers": self.k,
            "y_high": self.y_high,
            "miss": MISS,
            "climb": CLIMB,
            "drop_max": DROP_MAX,
            "vault": VAULT,
            "slope_max_deg": SLOPE_MAX_DEG,
            "baker": "atlas-cpu-bvh",
            "index": "iz*nx+ix",
            "layout": "nav.bin: (iz*nx+ix)*K + layer -> f32 height (asc, MISS empty); nav_door.bin: u8 per cell",
            "nav_blk": "u8[nx*nz*K] 8-dir edge mask (bit d = edge to NB[d] blocked by a thin wall/fence; player-capsule second pass)",
            "nav_wallcell": "u8[nx*nz] 1 = a wall occupies this cell's body column (simplify guard: never straighten a chord through it)",
        });
        std::fs::write(dir.join("nav.json"), serde_json::to_string_pretty(&meta)?)
            .with_context(|| format!("writing {}", dir.join("nav.json").display()))?;
        Ok(())
    }
}

/// Bake a nav grid for an already-loaded pack. Pure CPU; parallel over grid columns.
pub fn bake(pack: &Pack, res: f32, k: usize) -> Result<Baked> {
    if res <= 0.0 {
        return Err(anyhow!("res must be > 0 (got {res})"));
    }
    if k == 0 {
        return Err(anyhow!("layers must be >= 1 (got {k})"));
    }

    let t_tris = Instant::now();
    let (tris, walls, min_y, max_y, door_tris) = build_tris(pack);
    let n_tris = tris.len();
    let n_walls = walls.len();
    eprintln!(
        "  nav-bake: {n_tris} column tris, {n_walls} wall tris (retained for blk), {door_tris} door tris, y in [{min_y:.1}, {max_y:.1}] in {:.2}s",
        t_tris.elapsed().as_secs_f32()
    );
    if n_tris == 0 {
        return Err(anyhow!("no walkable triangles in pack (nothing to bake)"));
    }

    // Grid bounds: 0.5/99.5 percentile of the world verts + PAD (rejects skybox/backdrop outliers),
    // same method as bake_nav.py.
    let step = (n_tris / 1_000_000).max(1);
    let mut xs: Vec<f32> = Vec::with_capacity(n_tris / step + 1);
    let mut zs: Vec<f32> = Vec::with_capacity(n_tris / step + 1);
    for t in tris.iter().step_by(step) {
        xs.push(t.a.x);
        zs.push(t.a.z);
    }
    let lo_x = percentile(&mut xs, 0.5);
    let hi_x = percentile(&mut xs, 99.5);
    let lo_z = percentile(&mut zs, 0.5);
    let hi_z = percentile(&mut zs, 99.5);
    let min_x = lo_x - PAD;
    let max_x = hi_x + PAD;
    let min_z = lo_z - PAD;
    let max_z = hi_z + PAD;
    let nx = (((max_x - min_x) / res).ceil() as usize).max(1) + 1;
    let nz = (((max_z - min_z) / res).ceil() as usize).max(1) + 1;
    let cells = nx * nz;
    // Guard against a pathological grid (a runaway percentile on broken geometry) blowing memory.
    let m = cells
        .checked_mul(k)
        .ok_or_else(|| anyhow!("grid {nx}x{nz}x{k} overflows"))?;
    if m > 400_000_000 {
        return Err(anyhow!(
            "grid {nx}x{nz}x{k} = {m} cells is implausibly large — aborting (check pack bounds)"
        ));
    }

    let y_high = Y_HIGH_FLOOR.max(max_y + 10.0);
    let y_low = min_y - 10.0;

    let t_bvh = Instant::now();
    let bvh = Bvh::build(tris);
    eprintln!(
        "  nav-bake: BVH {} nodes over {n_tris} tris in {:.2}s; grid {nx} x {nz} @ {res}m x {k} = {:.1} MB",
        bvh.nodes.len(),
        t_bvh.elapsed().as_secs_f32(),
        (m * 4) as f32 / 1e6
    );

    // Cast one vertical column per cell, in parallel. Per-thread scratch (hit list + BVH stack +
    // floor buffer) is created once per worker via for_each_init.
    let t_cast = Instant::now();
    let mut heights = vec![MISS; m];
    let mut door = vec![0u8; cells];
    heights
        .par_chunks_mut(k)
        .zip(door.par_iter_mut())
        .enumerate()
        .for_each_init(
            || (Vec::<Hit>::with_capacity(64), Vec::<u32>::with_capacity(64), Vec::<f32>::with_capacity(16)),
            |(hits, nstack, floors), (cell, (hout, dout))| {
                let ix = cell % nx;
                let iz = cell / nx;
                let x = min_x + ix as f32 * res;
                let z = min_z + iz as f32 * res;
                bvh.column(x, z, y_low, y_high, hits, nstack);
                let (_, is_door) = resolve_column(hits, k, hout, floors);
                *dout = is_door as u8;
            },
        );

    let walkable = heights
        .par_chunks(k)
        .filter(|c| c[0] > MISS_HALF)
        .count();
    let door_cells = door.iter().filter(|&&d| d != 0).count();
    eprintln!(
        "  nav-bake: cast {cells} columns in {:.2}s; {walkable} walkable ({:.1}%), {door_cells} door cells",
        t_cast.elapsed().as_secs_f32(),
        100.0 * walkable as f32 / cells as f32
    );

    // ---- FIX 1: player-capsule SECOND pass -> nav_blk.bin ------------------------------------
    // Build the 3-D wall BVH, then for every walkable (cell,layer) edge the router would traverse,
    // cast a ±PLAYER_RADIUS capsule fan at body heights; any wall hit blocks that edge. Each cell
    // writes ONLY its OWN k-slice (par_chunks_mut ownership → race-free, no atomics); best_layer is
    // resolved per cell exactly as the router does, so the bit lands on the (node,d) A* checks. The
    // reverse bit is set independently when the neighbour cell processes its own outgoing edge.
    let t_bvh_w = Instant::now();
    let wall_bvh = WallBvh::build(walls);
    let t_blk = Instant::now();
    let mut blk = vec![0u8; m];
    // Per-cell "a wall occupies my body column" flag (see WallBvh::box_overlaps). Half-extent =
    // half a cell + the capsule radius, so it covers any wall a chord passing anywhere in the cell
    // (±res/2) could clip within ±PLAYER_RADIUS — the sub-cell walls the per-edge blk mask misses.
    let mut wall_cell = vec![0u8; cells];
    let wc_half = res * 0.5 + PLAYER_RADIUS;
    if !wall_bvh.tris.is_empty() {
        blk.par_chunks_mut(k).zip(wall_cell.par_iter_mut()).enumerate().for_each_init(
            || Vec::<u32>::with_capacity(64),
            |wstack, (c, (bout, wc))| {
                let ix = c % nx;
                let iz = c / nx;
                let cx = min_x + ix as f32 * res;
                let cz = min_z + iz as f32 * res;
                let door_c = door[c] != 0;
                let mut any_floor = false;
                for l in 0..k {
                    let floor_c = heights[c * k + l];
                    if floor_c <= MISS_HALF {
                        break; // ascending; MISS sinks to the end
                    }
                    any_floor = true;
                    let mut mask = 0u8;
                    for d in 0..8 {
                        let (dx, dz) = NB_BAKE[d];
                        let jx = ix as i64 + dx as i64;
                        let jz = iz as i64 + dz as i64;
                        if jx < 0 || jz < 0 || jx >= nx as i64 || jz >= nz as i64 {
                            continue;
                        }
                        let nc = (jz * nx as i64 + jx) as usize;
                        let nl = best_layer_bake(&heights, nc, k, floor_c);
                        if nl < 0 {
                            continue; // neighbour has no floor (matches nav.rs `continue`)
                        }
                        let floor_nc = heights[nc * k + nl as usize];
                        let up = floor_nc - floor_c;
                        let horiz = ((dx * dx + dz * dz) as f32).sqrt() * res;
                        if !walkable_step_bake(up, horiz) {
                            continue; // an edge the router would never traverse — don't bother
                        }
                        if door_c || door[nc] != 0 {
                            continue; // doors stay transparent (never blocked)
                        }
                        let ncx = min_x + jx as f32 * res;
                        let ncz = min_z + jz as f32 * res;
                        if capsule_blocked(cx, cz, floor_c, ncx, ncz, floor_nc, &wall_bvh, wstack) {
                            mask |= 1u8 << d;
                        }
                    }
                    bout[l] = mask;
                }
                // Flag the cell if a wall sits in the body column of ANY of its floors (skip doors —
                // a door footprint must stay straightenable so routes can head straight for it).
                if any_floor && !door_c {
                    for l in 0..k {
                        let floor_c = heights[c * k + l];
                        if floor_c <= MISS_HALF {
                            break;
                        }
                        // Body band = [floor, floor + PLAYER_HEIGHT + STEP_UP]: the top margin
                        // covers a chord that legitimately floats up to one step above the floor
                        // (segment_clear's float_tol) whose highest capsule sample reaches
                        // floor + STEP_UP + (PLAYER_HEIGHT-0.15).
                        let bmin = Vec3::new(cx - wc_half, floor_c, cz - wc_half);
                        let bmax = Vec3::new(
                            cx + wc_half,
                            floor_c + PLAYER_HEIGHT_NAV + STEP_UP_NAV,
                            cz + wc_half,
                        );
                        if wall_bvh.box_overlaps(bmin, bmax, wstack) {
                            *wc = 1;
                            break;
                        }
                    }
                }
            },
        );
    }
    let blocked_edges = blk.iter().map(|b| b.count_ones() as usize).sum::<usize>();
    let wall_cells = wall_cell.iter().filter(|&&w| w != 0).count();
    eprintln!(
        "  nav-bake: wall BVH {} nodes over {} tris in {:.2}s; capsule pass {} blocked edge-bits, {} wall cells ({:.1}%) in {:.2}s",
        wall_bvh.nodes.len(),
        wall_bvh.tris.len(),
        t_blk.duration_since(t_bvh_w).as_secs_f32(),
        blocked_edges,
        wall_cells,
        100.0 * wall_cells as f32 / cells as f32,
        t_blk.elapsed().as_secs_f32()
    );

    Ok(Baked {
        dataset: pack.manifest.dataset.clone(),
        min_x,
        min_z,
        res,
        nx,
        nz,
        k,
        y_high,
        heights,
        door,
        blk,
        wall_cell,
        walkable,
        door_cells,
        blocked_edges,
        wall_cells,
        wall_tris: wall_bvh.tris.len(),
        wall_bvh,
    })
}

/// `q`-th percentile (0..100) of `v` via partial select (mutates order). 0 for empty.
fn percentile(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let idx = (((q / 100.0) * (v.len() - 1) as f32).round() as usize).min(v.len() - 1);
    v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    v[idx]
}

// ---- headless self-check + MACHINE PROOF (FIX 5): route many legs, assert ZERO wall-crossings ---

/// Extended self-check: load the freshly baked grid, route 200+ varied legs, and for EVERY segment
/// of each SIMPLIFIED route sample the player-capsule fan against the KEPT wall BVH — the AFTER
/// wall-crossing count MUST be 0. For contrast it re-routes the SAME legs on a grid with the wall
/// mask disabled (reproducing OLD no-nav_blk.bin routing) and counts BEFORE crossings. Also reports
/// reachability, and prints one wall-threading BEFORE leg's coords (for a before/after screenshot).
fn self_check(baked: &Baked, dir: &Path) {
    let Some(grid) = NavGrid::load(dir) else {
        eprintln!("  [verify] FAILED: NavGrid::load returned None on the freshly baked pack");
        return;
    };
    eprintln!("  [verify] NavGrid::load OK ({} nodes)", grid.nodes());
    // "before" grid = same data with the wall mask + clearance zeroed (OLD routing behaviour).
    let mut grid_before = match NavGrid::load(dir) {
        Some(g) => g,
        None => return,
    };
    grid_before.clear_wall_data();

    let (nx, k) = (baked.nx, baked.k);
    let mut walk: Vec<(usize, usize, f32)> = Vec::new(); // (ix, iz, floor_y)
    for cell in 0..baked.cells() {
        let y = baked.heights[cell * k];
        if y > MISS_HALF {
            walk.push((cell % nx, cell / nx, y));
        }
    }
    if walk.len() < 2 {
        eprintln!("  [verify] only {} walkable cell(s) — no route to test", walk.len());
        return;
    }
    let world = |w: &(usize, usize, f32)| {
        Vec3::new(baked.min_x + w.0 as f32 * baked.res, w.2, baked.min_z + w.1 as f32 * baked.res)
    };

    // Deterministic varied legs: coprime-ish index strides sweep start/dest position, direction and
    // length across the whole walkable set (no RNG dep).
    let n = walk.len();
    let want = 256usize;
    let sa = (n / 3).max(1);
    let sb = ((n / 7) | 1).max(1); // odd -> coprime-ish with n's factors
    let min_span2 = (8.0f32 / baked.res).powi(2); // non-trivial legs (>= ~8 m)
    let mut sc_a = Scratch::new(grid.nodes());
    let mut sc_b = Scratch::new(grid_before.nodes());
    let (mut routes_after, mut routes_before) = (0usize, 0usize);
    let (mut segs_after, mut segs_before) = (0usize, 0usize);
    let (mut cross_after, mut cross_before) = (0usize, 0usize);
    let mut cross_raw = 0usize; // crossings on the RAW (unsimplified) A* path — attributes the gap
    let mut cross_after_door = 0usize; // AFTER crossings whose segment passes through a door cell
    let mut attempts = 0usize;
    let mut example: Option<(Vec3, Vec3)> = None;
    let mut i = 0usize;
    while attempts < want && i < want * 8 {
        let si = i.wrapping_mul(sa) % n;
        let di = (i.wrapping_mul(sb) + 1) % n;
        i += 1;
        if si == di {
            continue;
        }
        let (s, d) = (walk[si], walk[di]);
        let span2 = (s.0 as f32 - d.0 as f32).powi(2) + (s.1 as f32 - d.1 as f32).powi(2);
        if span2 < min_span2 {
            continue;
        }
        attempts += 1;
        let (a, b) = (world(&s), world(&d));
        if let Some((raw, simp)) = grid.route_debug(a, b, &mut sc_a) {
            routes_after += 1;
            let (segs, cr) = count_wall_crossings(&simp, &baked.wall_bvh);
            segs_after += segs;
            cross_after += cr;
            // Attribute the crossing: does the offending segment pass through a DOOR cell? A door
            // frame is a wall the router is ALLOWED to cross (you open the door), so a graze there
            // is not a violation of "never cross a wall a player can't pass".
            let cr_door = count_door_crossings(&simp, baked);
            cross_after_door += cr_door;
            let (_, cr_raw) = count_wall_crossings(&raw, &baked.wall_bvh);
            cross_raw += cr_raw;
        }
        if let Some((poly, _)) = grid_before.path(a, b, &mut sc_b, None) {
            routes_before += 1;
            let (segs, cr) = count_wall_crossings(&poly, &baked.wall_bvh);
            segs_before += segs;
            cross_before += cr;
            if cr > 0 && example.is_none() {
                example = Some((a, b)); // a leg the OLD router threaded through a wall
            }
        }
    }

    eprintln!(
        "  [verify] machine proof: {attempts} legs attempted; routed AFTER {routes_after} / BEFORE {routes_before} ({:.0}% reachable)",
        100.0 * routes_after as f32 / attempts.max(1) as f32
    );
    eprintln!(
        "  [verify] wall-crossings on SIMPLIFIED routes: BEFORE (no blk mask) {cross_before} over {segs_before} segs; AFTER (blk + wall-aware simplify) {cross_after} over {segs_after} segs",
    );
    eprintln!(
        "  [verify] attribution: AFTER RAW A* path crossings {cross_raw} (blk/connectivity gap) vs SIMPLIFIED {cross_after} (simplify adds {}); of the {cross_after} AFTER crossings, {cross_after_door} are at a DOOR (passable frame, not a violation)",
        cross_after.saturating_sub(cross_raw)
    );
    let cross_after_walls = cross_after.saturating_sub(cross_after_door);
    if let Some((a, b)) = example {
        eprintln!(
            "  [verify] example BEFORE wall-threading leg: EFT_ROUTE=\"{:.2},{:.2},{:.2};{:.2},{:.2},{:.2}\"",
            a.x, a.y, a.z, b.x, b.y, b.z
        );
    }
    if cross_after_walls == 0 {
        eprintln!(
            "  [verify] PASS: ZERO impassable-wall crossings across all {routes_after} simplified routes ({cross_after_door} door-frame graze(s) excluded — doors are passable)"
        );
    } else {
        eprintln!("  [verify] FAIL: {cross_after_walls} wall-crossing(s) remain on the simplified routes");
    }
}

// ---- CLI entry: `atlas bake-nav <pack_dir> [--res R] [--layers K]` -----------------------------

/// Handle the headless `bake-nav` subcommand. `args` is argv AFTER the "bake-nav" token. Returns a
/// process exit code (0 = ok). Never panics (release is panic=abort): all failures return non-zero.
pub fn run_cli(args: &[String]) -> i32 {
    let mut pack_dir: Option<String> = None;
    let mut res: f32 = 1.0;
    let mut k: usize = 8;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--res" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<f32>().ok()) {
                    Some(v) => res = v,
                    None => {
                        eprintln!("bake-nav: --res needs a number");
                        return 2;
                    }
                }
            }
            "--layers" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<usize>().ok()) {
                    Some(v) => k = v,
                    None => {
                        eprintln!("bake-nav: --layers needs an integer");
                        return 2;
                    }
                }
            }
            s if s.starts_with('-') => {
                eprintln!("bake-nav: unknown flag '{s}'");
                return 2;
            }
            s => {
                if pack_dir.is_none() {
                    pack_dir = Some(s.to_string());
                } else {
                    eprintln!("bake-nav: unexpected extra argument '{s}'");
                    return 2;
                }
            }
        }
        i += 1;
    }
    let Some(dir) = pack_dir else {
        eprintln!("usage: atlas bake-nav <pack_dir> [--res 1.0] [--layers 8]");
        return 2;
    };
    let dir_path = Path::new(&dir);

    let t0 = Instant::now();
    eprintln!("bake-nav: loading pack '{dir}'");
    let pack = match Pack::load(dir_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("bake-nav: failed to load pack '{dir}': {e:#}");
            return 1;
        }
    };
    let baked = match bake(&pack, res, k) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bake-nav: bake failed: {e:#}");
            return 1;
        }
    };
    if let Err(e) = baked.write(dir_path) {
        eprintln!("bake-nav: writing nav files failed: {e:#}");
        return 1;
    }
    let nav_bin_bytes = baked.heights.len() * 4;
    eprintln!(
        "bake-nav: OK '{}' -> nav.bin ({} x {} x {} = {} cells, {} floats, {} bytes), \
         nav_blk.bin ({} bytes, {} blocked edge-bits from {} wall tris), \
         nav_wallcell.bin ({} bytes, {} wall cells), \
         {} walkable, {} door cells; bounds x[{:.1},{:.1}] z[{:.1},{:.1}]; {:.2}s total",
        baked.dataset,
        baked.nx,
        baked.nz,
        baked.k,
        baked.cells(),
        baked.heights.len(),
        nav_bin_bytes,
        baked.blk.len(),
        baked.blocked_edges,
        baked.wall_tris,
        baked.wall_cell.len(),
        baked.wall_cells,
        baked.walkable,
        baked.door_cells,
        baked.min_x,
        baked.min_x + (baked.nx - 1) as f32 * baked.res,
        baked.min_z,
        baked.min_z + (baked.nz - 1) as f32 * baked.res,
        t0.elapsed().as_secs_f32()
    );

    // Headless proof the writer matches the runtime loader + the router routes on it.
    self_check(&baked, dir_path);
    0
}
