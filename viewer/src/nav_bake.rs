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
//! nav_blk.bin (thin-wall edge mask) is intentionally NOT emitted — the router treats an absent
//! mask as "no blocked edges" and rebuilds walkability from the heights at load, so it is optional.
//! Producing it would need general horizontal body-height raycasts (a whole second pass); deferred.
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

/// One world-space triangle that a vertical ray can hit (walls pre-filtered out).
#[derive(Clone, Copy)]
struct Tri {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    /// Normalised world-space normal Y (sign-corrected for mirror instances).
    ny: f32,
    /// Belongs to a DOOR-tagged mesh/root → transparent to the cast + stamps the door footprint.
    door: bool,
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

/// Build the world-space triangle soup for the BVH from the pack's meshes × instance affines.
/// Each unique mesh is unpacked ONCE (via the eftpack accessor) then transformed for every one of
/// its instances. Vertical walls and degenerate faces are dropped. Also tracks the true vertical
/// extent (for the ray origin) and the door-triangle count.
fn build_tris(pack: &Pack) -> (Vec<Tri>, f32, f32, usize) {
    let by_mesh = pack.instances_by_mesh();
    let mut tris: Vec<Tri> = Vec::new();
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
        for &iid in inst_ids {
            let inst = &pack.instances[iid as usize];
            let root_is_door = pack
                .manifest
                .roots
                .get(inst.root_id as usize)
                .map(|r| is_door_name(r))
                .unwrap_or(false);
            let door = mesh_is_door || root_is_door;
            let aff = inst.affine3a();
            let mirror = inst.is_mirror();
            for tri in geom.indices.chunks_exact(3) {
                let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                // Defensive: a bad index just skips the face (release is panic=abort).
                if i0 >= geom.positions.len() || i1 >= geom.positions.len() || i2 >= geom.positions.len() {
                    continue;
                }
                let a = aff.transform_point3(Vec3::from(geom.positions[i0]));
                let b = aff.transform_point3(Vec3::from(geom.positions[i1]));
                let c = aff.transform_point3(Vec3::from(geom.positions[i2]));
                // Drop triangles whose XZ footprint is ~a line (vertical walls): a vertical ray
                // cannot register them, and skipping them keeps the BVH lean.
                let e1 = b - a;
                let e2 = c - a;
                let xz_area2 = (e1.x * e2.z - e1.z * e2.x).abs();
                if xz_area2 < MIN_XZ_AREA2 {
                    continue;
                }
                let n = e1.cross(e2);
                let nlen = n.length();
                if nlen < 1.0e-12 {
                    continue; // degenerate
                }
                let mut ny = n.y / nlen;
                if mirror {
                    ny = -ny; // restore correct orientation for winding-flipped mirror instances
                }
                min_y = min_y.min(a.y.min(b.y.min(c.y)));
                max_y = max_y.max(a.y.max(b.y.max(c.y)));
                if door {
                    door_tris += 1;
                }
                tris.push(Tri { a, b, c, ny, door });
            }
        }
    }
    if !min_y.is_finite() {
        min_y = 0.0;
        max_y = 0.0;
    }
    (tris, min_y, max_y, door_tris)
}

// ---- BVH (median-split over XZ, for vertical-ray queries) -------------------------------------

#[derive(Clone, Copy)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    /// Leaf (count>0): tris[start..start+count]. Internal (count==0): children at start, start+1.
    start: u32,
    count: u32,
}

struct Bvh {
    nodes: Vec<BvhNode>,
    tris: Vec<Tri>,
}

impl Bvh {
    fn build(tris: Vec<Tri>) -> Bvh {
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
    heights: Vec<f32>, // nx*nz*k, ascending, MISS empty
    door: Vec<u8>,     // nx*nz
    walkable: usize,
    door_cells: usize,
}

impl Baked {
    fn cells(&self) -> usize {
        self.nx * self.nz
    }

    /// Write nav.bin + nav_door.bin + nav.json into `dir`, matching the format `NavGrid::load` reads.
    fn write(&self, dir: &Path) -> Result<()> {
        let bin: &[u8] = bytemuck::cast_slice(&self.heights);
        std::fs::write(dir.join("nav.bin"), bin)
            .with_context(|| format!("writing {}", dir.join("nav.bin").display()))?;
        std::fs::write(dir.join("nav_door.bin"), &self.door)
            .with_context(|| format!("writing {}", dir.join("nav_door.bin").display()))?;
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
    let (tris, min_y, max_y, door_tris) = build_tris(pack);
    let n_tris = tris.len();
    eprintln!(
        "  nav-bake: {n_tris} world tris (walls dropped), {door_tris} door tris, y in [{min_y:.1}, {max_y:.1}] in {:.2}s",
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
        walkable,
        door_cells,
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

// ---- headless self-check: prove NavGrid::load + a path query succeed on the fresh bake ---------

fn self_check(baked: &Baked, dir: &Path) {
    let Some(grid) = NavGrid::load(dir) else {
        eprintln!("  [verify] FAILED: NavGrid::load returned None on the freshly baked pack");
        return;
    };
    eprintln!("  [verify] NavGrid::load OK ({} nodes)", grid.nodes());

    // Collect walkable cell centres (lowest floor) as candidate route endpoints.
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
    // Start = the walkable cell nearest the walkable centroid (well inside the map).
    let (mcx, mcz) = (
        walk.iter().map(|w| w.0).sum::<usize>() as f32 / walk.len() as f32,
        walk.iter().map(|w| w.1).sum::<usize>() as f32 / walk.len() as f32,
    );
    let start = *walk
        .iter()
        .min_by(|a, b| {
            let da = (a.0 as f32 - mcx).powi(2) + (a.1 as f32 - mcz).powi(2);
            let db = (b.0 as f32 - mcx).powi(2) + (b.1 as f32 - mcz).powi(2);
            da.total_cmp(&db)
        })
        .unwrap();
    let world = |w: &(usize, usize, f32)| {
        Vec3::new(baked.min_x + w.0 as f32 * baked.res, w.2, baked.min_z + w.1 as f32 * baked.res)
    };
    let a = world(&start);
    // Try walkable cells as the destination NEAREST-first (skipping trivially-close ones): a cell a
    // few metres away on the same connected floor is the strongest proof the router routes on this
    // grid; far corners are often genuinely disconnected and would only waste the attempt budget.
    let mut cands: Vec<&(usize, usize, f32)> = walk.iter().collect();
    let span2 = |d: &&(usize, usize, f32)| {
        (d.0 as f32 - start.0 as f32).powi(2) + (d.1 as f32 - start.1 as f32).powi(2)
    };
    cands.sort_by(|p, q| span2(p).total_cmp(&span2(q)));
    let mut scratch = Scratch::new(grid.nodes());
    let min_span = (5.0 / baked.res).powi(2); // want a non-trivial leg (>= ~5 m)
    let mut tried = 0;
    for d in &cands {
        if span2(d) < min_span {
            continue;
        }
        tried += 1;
        if tried > 200 {
            break;
        }
        let b = world(d);
        if let Some((poly, len)) = grid.path(a, b, &mut scratch, None) {
            eprintln!(
                "  [verify] path OK: cell({},{}) -> cell({},{}) = {} node(s), {:.1} m (from ({:.1},{:.1},{:.1}) to ({:.1},{:.1},{:.1}))",
                start.0, start.1, d.0, d.1, poly.len(), len, a.x, a.y, a.z, b.x, b.y, b.z
            );
            return;
        }
    }
    eprintln!("  [verify] NavGrid loaded + query ran, but no route between the sampled endpoints (disconnected floors?)");
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
         {} walkable, {} door cells; bounds x[{:.1},{:.1}] z[{:.1},{:.1}]; {:.2}s total",
        baked.dataset,
        baked.nx,
        baked.nz,
        baked.k,
        baked.cells(),
        baked.heights.len(),
        nav_bin_bytes,
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
