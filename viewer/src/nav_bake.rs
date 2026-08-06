//! nav_bake — PORTABLE viewer-side nav-grid baker (pure-CPU BVH raycast).
//!
//! WHY: the runtime router (`crate::nav`) can only LOAD a pre-baked grid. The only baker that
//! existed (tarkmap/bake_nav.py) needs NVIDIA-Warp/CUDA + an `instanced_raw.glb` the native build
//! never produces, so NO pack shipped nav data and routing was dead on every machine. This module
//! bakes the SAME layered-2.5D nav grid straight from a loaded [`Pack`]'s world triangles on the
//! CPU (a median-split BVH + vertical down-raycasts, parallelised with rayon), so routing is
//! produced by default on AMD / NVIDIA / no-GPU alike.
//!
//! WALKABILITY IS THE GAME'S, NOT OURS. The rules come from Unity's `NavMeshProjectSettings`
//! (extracted by `extraction/unity/eft_extract_nav.py` into packs/shared/nav_agents.json) — see
//! [`NavAgent`]. We bake against `Humanoid`: radius 0.30, height 1.70, slope 48 deg, climb 0.38,
//! minRegionArea 2 m². The decisive pair is `ledgeDropHeight = 0` and `maxJumpAcrossDistance = 0`,
//! true of EVERY agent EFT ships: those are the only settings that create drop-down and jump
//! off-mesh links, so the game's navmesh has none, and a descent is bounded exactly like a climb.
//!
//! GEOMETRY IS THE PHYSICS WORLD, NOT THE VISIBLE ONE. `build_tris` bakes render meshes AND the
//! pack's physics colliders (`add_collider_tris`), because most of what you collide with has no
//! renderer at all — on interchange, 131,945 of 141,347 colliders. Unity does the same thing via
//! `NavMeshSurface.m_UseGeometry = PhysicsColliders`. Colliders are selected by LAYER NAME (EFT
//! splits movement collision `LowPolyCollider` from ballistics collision `HighPolyCollider`), and
//! triggers are skipped since a Unity trigger has no contact response.
//!
//! `EFT_NAV_LEGACY=1` restores the pre-derivation constants and `EFT_NAV_COLLIDERS=0` drops back to
//! render-only geometry, so any claim about what these changed can be produced as an A/B.
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

// ---- agent descriptor: READ FROM THE GAME, never hand-tuned -----------------------------------
// EFT stores its pathfinding recipe in Unity's `NavMeshProjectSettings` (an ENGINE type, so it is
// readable despite the encrypted il2cpp metadata). `eft_extract_nav.py` lifts it verbatim into
// packs/shared/nav_agents.json. We bake against `Humanoid`, the default agent type (agentTypeID 0).
//
// The two fields that mattered most were the ones nobody would have guessed: EVERY agent EFT ships
// has `ledgeDropHeight = 0` and `maxJumpAcrossDistance = 0`. In Unity those are the only settings
// that generate drop-down and jump-across off-mesh links, so the game's navmesh contains NO drops
// and NO jumps at all — a bot can only move where the surface is continuous within `agentClimb`.
// Our router allowed a flat 2.0 m free fall in any direction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NavAgent {
    pub(crate) radius: f32,
    /// Required clearance above a floor (Unity `agentHeight`).
    pub(crate) height: f32,
    pub(crate) slope_deg: f32,
    /// tan(slope_deg) — the per-metre rise of the steepest surface still considered walkable.
    pub(crate) slope_tan: f32,
    /// Max step onto a DISCONTINUITY (kerb, stair riser).
    pub(crate) climb: f32,
    /// Unity `ledgeDropHeight`: 0 on every EFT agent (no drop-down links exist).
    pub(crate) ledge_drop: f32,
    /// Unity `minRegionArea` (m²) — islands smaller than this are discarded by the bake.
    pub(crate) min_region_area: f32,
    /// Where the values came from, for the bake log ("game" or "fallback").
    pub(crate) source: &'static str,
}

impl NavAgent {
    /// Fallback used only when packs/shared/nav_agents.json is absent — the previously hand-tuned
    /// numbers, so a pack without the sidecar bakes exactly as it did before.
    const FALLBACK: NavAgent = NavAgent {
        radius: 0.30,
        height: 1.8,
        slope_deg: 48.0,
        slope_tan: 1.110_613,
        climb: 0.38,
        ledge_drop: 0.0,
        min_region_area: 2.0,
        source: "fallback",
    };

    /// `EFT_NAV_LEGACY=1` — the exact pre-game-derived rules (60 deg surface recording, 1.8 m
    /// headroom, and a flat 2 m free-fall via `ledge_drop`). Kept as a measurement knob so any
    /// claim about what the game-derived rules changed can be produced as an A/B, not asserted.
    const LEGACY: NavAgent = NavAgent {
        radius: 0.30,
        height: 1.8,
        slope_deg: 60.0,
        slope_tan: 1.732_051,
        climb: 0.38,
        ledge_drop: 2.0,
        min_region_area: 0.0,
        source: "legacy",
    };

    /// Largest legal height CHANGE across one edge of horizontal length `run`, in either direction.
    /// A continuous surface that passed the slope filter can rise or fall at most `run·tan(slope)`
    /// over that span; a discontinuity is only crossable up to `climb`. With `ledgeDropHeight = 0`
    /// there is nothing else — no free fall, in particular none of the old flat 2 m drop.
    ///
    /// HARD-CAPPED AT [`VAULT`]: whatever the slope term works out to, a single edge may never
    /// span more height than a player can actually vault. On a 1 m grid the `run·tan(48°)` term
    /// reaches 1.11 m orthogonally and **1.57 m diagonally** — above the 1.2 m vault — so without
    /// this clamp the diagonal moves were quietly the most permissive edges on the grid, which is
    /// exactly how a route steps onto something it has no business standing on.
    #[inline]
    pub(crate) fn max_step(&self, run: f32) -> f32 {
        if self.ledge_drop > 0.0 {
            // Only the LEGACY A/B profile takes this path: a flat free-fall allowance, independent
            // of edge length. No agent EFT ships has a non-zero ledgeDropHeight.
            return self.ledge_drop;
        }
        self.climb.max(run * self.slope_tan).min(VAULT)
    }
}

/// The agent descriptor for this process, loaded once from the shared pack tier.
pub(crate) fn agent() -> &'static NavAgent {
    static A: std::sync::OnceLock<NavAgent> = std::sync::OnceLock::new();
    A.get_or_init(|| {
        if std::env::var("EFT_NAV_LEGACY").as_deref() == Ok("1") {
            return NavAgent::LEGACY;
        }
        load_agent().unwrap_or(NavAgent::FALLBACK)
    })
}

/// `EFT_NAV_COLLIDERS=0` bakes from render geometry only (the pre-collider input), for A/B.
fn colliders_enabled() -> bool {
    std::env::var("EFT_NAV_COLLIDERS").as_deref() != Ok("0")
}

fn load_agent() -> Option<NavAgent> {
    let txt = std::fs::read_to_string(crate::paths::shared_dir().join("nav_agents.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    let list = v.get("agents")?.as_array()?;
    // Prefer the default `Humanoid` (agentTypeID 0); otherwise take the first entry.
    let a = list
        .iter()
        .find(|a| a.get("name").and_then(|n| n.as_str()) == Some("Humanoid"))
        .or_else(|| list.first())?;
    let f = |k: &str, d: f32| {
        a.get(k)
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(d)
    };
    let slope_deg = f("agentSlope", 48.0).clamp(10.0, 80.0);
    Some(NavAgent {
        radius: f("agentRadius", 0.30),
        height: f("agentHeight", 1.8),
        slope_deg,
        slope_tan: slope_deg.to_radians().tan(),
        climb: f("agentClimb", 0.38),
        ledge_drop: f("ledgeDropHeight", 0.0),
        min_region_area: f("minRegionArea", 2.0),
        source: "game",
    })
}

// ---- constants (match bake_nav.py) ------------------------------------------------------------
const VAULT: f32 = 1.2;
const MISS: f32 = -1.0e9;
const MISS_HALF: f32 = MISS * 0.5;
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
/// Wall triangles smaller than this (m²) are clutter — skipped, UNLESS they are tall (see
/// `WALL_MIN_SPAN_Y`). Area alone is the wrong test for a barrier: measured on streets, 95-100% of
/// every fence/railing's vertical triangles fall under 0.04 m² because a bar or slat is thin, so
/// the whole fence was dropped from the wall set and pathfinding walked straight through it.
const WALL_MIN_AREA: f32 = 0.04;
/// ...so ALSO keep any near-vertical triangle spanning at least this much height (m), whatever its
/// area. A fence bar, railing upright or palisade slat is thin but TALL; genuine clutter (bolts,
/// trim, small props) is small in every dimension. This is what makes fences block.
const WALL_MIN_SPAN_Y: f32 = 0.40;
/// Player capsule half-width (m) — matches `walk_ground::PLAYER_RADIUS`. The ±R fan blocks a gap
/// narrower than 2·R = 0.64 m even when a centre ray would thread it.
const PLAYER_RADIUS: f32 = 0.32;
/// Total player height (m) — matches `walk_ground::PLAYER_HEIGHT`.
const PLAYER_HEIGHT_NAV: f32 = 1.8;
/// Free step-up (m) — the capsule fan starts ABOVE this so curbs / low risers the router already
/// steps onto are NOT read as walls (the curb-vs-wall band).
///
/// This is the SAMPLING constant only. The value the router actually gates edges on is
/// `free_step(res)` below, which is resolution-dependent; see the comment there for why the two
/// must never be conflated again.
const STEP_UP_NAV: f32 = 0.45;

/// The free step the ROUTER will use, as a function of grid resolution — and therefore the step the
/// BAKER must validate edges against. One definition, both sides.
///
/// These were two numbers. The baker hardcoded 0.45 while `Baked::write` shipped
/// `res * tan(55 deg)` (0.714 at res 0.5) into nav.json, so the router traversed 4,445 edges the
/// capsule pass had never tested — every one a potential route through a wall, and exactly the two
/// crossings the self-check kept reporting. Proven by re-baking with `EFT_NAV_STEP=0.45`: forcing
/// the router back onto the baker's constant returned the check to ZERO crossings.
///
/// Why it is resolution-dependent: a 0.5 m cell cannot represent a stair tread. A real EFT interior
/// stair (`Sparja_stairs_LOD0`) rises 4.02 m over a 3.17 m run, so one cell step spans ~0.634 m of
/// rise. Gate below that and every stair-accessed floor becomes a sealed island. So the allowance is
/// the grid's ALIASING limit, floored at the agent's own climb and capped at what a player clears
/// unaided. Bake finer and it shrinks by itself.
#[inline]
pub(crate) fn free_step(res: f32) -> f32 {
    (res * 55.0_f32.to_radians().tan()).clamp(agent().climb, VAULT)
}
// (The old `SLOPE_TAN_NAV = tan(45°)` stand-in is gone: the baker and the router now BOTH use the
// game's `agentSlope` — the baker via `agent().slope_tan`, the router via nav.json's
// `walk_slope_deg`, which this baker emits. There is no second slope number to keep in sync.)
/// A door-tagged mesh only punches a passable hole (and drops out of the wall set) when its
/// INSTANCE footprint is door-panel sized (≤ this, in the SMALLER horizontal span). A large
/// gate/shutter fence keeps blocking — otherwise a `gate`/`shutter` NAME on a wall-wide mesh would
/// open a wall-wide gap the player can't actually pass.
const DOOR_FOOTPRINT_MAX: f32 = 1.5;
/// Capsule perpendicular offsets (−R, 0, +R) across the edge.
/// Five samples, not three: with only (-R, 0, +R) a fence with ~0.3 m bar spacing could be
/// threaded -- every sample landing in a gap -- which is exactly how routes crossed railings even
/// when the bars WERE in the wall set. The player is a solid 0.64 m-wide capsule; sampling it
/// densely is the cheap approximation of sweeping it.
const CAP_OFF: [f32; 5] = [
    -PLAYER_RADIUS,
    -PLAYER_RADIUS * 0.5,
    0.0,
    PLAYER_RADIUS * 0.5,
    PLAYER_RADIUS,
];
/// Body sample heights above the floor (shins / waist / head) — start above STEP_UP so low curbs
/// aren't over-blocked; matches `walk_ground::resolve_walls`'s capsule samples.
const CAP_H: [f32; 3] = [STEP_UP_NAV + 0.1, 1.0, PLAYER_HEIGHT_NAV - 0.15];
/// 8-neighbour offsets — MUST match `nav.rs` NB order (block-mask bit d = the edge to NB_BAKE[d]).
const NB_BAKE: [(i32, i32); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

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
                if i0 >= geom.positions.len()
                    || i1 >= geom.positions.len()
                    || i2 >= geom.positions.len()
                {
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
                let span_y = a.y.max(b.y).max(c.y) - a.y.min(b.y).min(c.y);
                if !door
                    && ny.abs() < WALL_MAX_NY
                    && (0.5 * nlen >= WALL_MIN_AREA || span_y >= WALL_MIN_SPAN_Y)
                {
                    walls.push(Tri {
                        a,
                        b,
                        c,
                        ny,
                        door: false,
                        mat,
                    });
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
                tris.push(Tri {
                    a,
                    b,
                    c,
                    ny,
                    door,
                    mat,
                });
            }
        }
    }
    // ---- PHYSICS COLLIDERS -------------------------------------------------------------------
    // The loop above walks RENDER meshes, so it only sees geometry you can look at. The world the
    // player collides with is the physics world, and on interchange 131,945 of 141,347 colliders
    // have no renderer at all — invisible walls, kerbs, railings and blockers that every route
    // baked so far walked straight through. Unity bakes its own navmesh from exactly this
    // (NavMeshSurface.m_UseGeometry = PhysicsColliders), so this is the game's own input, not an
    // approximation of it.
    let n_col = add_collider_tris(pack, &mut tris, &mut walls, &mut min_y, &mut max_y);
    if n_col > 0 {
        eprintln!("  nav-bake: +{n_col} triangles from physics colliders");
    }

    if !min_y.is_finite() {
        min_y = 0.0;
        max_y = 0.0;
    }
    (tris, walls, min_y, max_y, door_tris)
}

/// Unity layers whose SOLID colliders form the world you walk against. Selected by NAME from the
/// pack's `layerNames` (straight out of TagManager) so no layer index is hardcoded — EFT separates
/// movement collision (`LowPolyCollider`) from ballistics collision (`HighPolyCollider`), and only
/// the former plus terrain, doors, the map border and invisible glass should stop a route.
///
/// `HighPolyCollider` is deliberately EXCLUDED: it is the fine hit-detection shell that sits on top
/// of the same objects, so including it would double every surface for no navigational gain.
const NAV_COLLIDER_LAYERS: [&str; 6] = [
    "LowPolyCollider",
    "DoorLowPolyCollider",
    "Terrain",
    "LevelBorder",
    "TransparentCollider",
    "Default",
];

/// Tessellate the pack's physics colliders into the nav triangle soup. Returns the triangle count.
///
/// Triggers are skipped: a Unity trigger has no contact response, so it cannot block movement.
/// (Interchange's 5,763 `Swamp_collider` boxes are triggers on the `Triggers` layer — swamp
/// splash/sound volumes — as are its 26,450 `Foliage` bush volumes.)
fn add_collider_tris(
    pack: &Pack,
    tris: &mut Vec<Tri>,
    walls: &mut Vec<Tri>,
    min_y: &mut f32,
    max_y: &mut f32,
) -> usize {
    if pack.colliders.is_empty() || !colliders_enabled() {
        return 0;
    }
    let before = tris.len();
    let mut skipped_layer = 0usize;
    let mut skipped_trigger = 0usize;
    let mut skipped_navignore = 0usize;
    let mut skipped_worldfloor = 0usize;
    // WORLD BACKSTOP GATE. EFT ships a single map-spanning box on the MOVEMENT layer whose job is
    // to stop anything that falls out of the level -- `TEMP_GROUND_COLIDER`, ~1 m thick, centred at
    // y = -16. It is solid, it is on `LowPolyCollider`, and by every rule above it is walkable
    // ground, so the bake laid a continuous floor at y = -15.5 under the entire map: 97.3% of
    // streets' floored cells sat on it. Routes then ran 18 m underground (invisible, but "walkable"
    // and 272 m long), and reachability read as near-perfect because everything connects down there.
    //
    // Gate it STRUCTURALLY rather than by name: a floor a player stands on is never a half-kilometre
    // slab a metre thick. streets ships 1874 x 1.0 x 1898 m; ground_zero ships one too; interchange
    // does not, which is why every previous nav measurement (all on interchange) never saw this.
    //
    // Do NOT compare against `manifest.bounds` -- those span 2434 x 2401 m on streets because they
    // include the distant skyline backdrop, so a 1898 m slab looks small next to them. Absolute
    // dimensions are the honest test: SLAB_MIN_SPAN wide and SLAB_MAX_THICK thin, together, is a
    // backstop and nothing else.
    const SLAB_MIN_SPAN: f32 = 500.0;
    const SLAB_MAX_THICK: f32 = 5.0;
    // Scratch reused across colliders so a 43k-collider map doesn't churn the allocator.
    let mut verts: Vec<Vec3> = Vec::with_capacity(64);
    let mut idx: Vec<[u32; 3]> = Vec::with_capacity(64);

    for c in &pack.colliders {
        if c.is_trigger() {
            skipped_trigger += 1;
            continue;
        }
        // The GAME's own navmesh exclusion. `NavMeshModifier.m_IgnoreFromBuild` is BSG saying this
        // object is not navigation geometry; the extractor decodes it and the packer stores it, and
        // until now nothing read it. Honouring it is strictly better than any heuristic we could
        // invent for the same objects — it is the authored answer. Streets carries 2,329 of them.
        if c.flags & crate::eftpack::col_flags::NAV_IGNORE != 0 {
            skipped_navignore += 1;
            continue;
        }
        if !NAV_COLLIDER_LAYERS.contains(&pack.layer_name(c.layer)) {
            skipped_layer += 1;
            continue;
        }
        // A wide, thin box is a world backstop, not a floor (see the gate above).
        if c.kind == 0 {
            let m3 = c.affine3a().matrix3;
            let s = Vec3::from(c.shape);
            let ext = (m3.x_axis * s.x).abs() + (m3.y_axis * s.y).abs() + (m3.z_axis * s.z).abs();
            if ext.x.max(ext.z) >= SLAB_MIN_SPAN && ext.y <= SLAB_MAX_THICK {
                if skipped_worldfloor == 0 {
                    eprintln!(
                        "  nav-bake: WORLD BACKSTOP ignored: {:.0} x {:.1} x {:.0} m solid on \
                         layer '{}' — a slab this wide and this thin is a fall-out-of-the-world \
                         catcher, not a walkable floor",
                        ext.x,
                        ext.y,
                        ext.z,
                        pack.layer_name(c.layer)
                    );
                }
                skipped_worldfloor += 1;
                continue;
            }
        }
        verts.clear();
        idx.clear();
        match c.kind {
            0 => shape_box(
                Vec3::from(c.center),
                Vec3::from(c.shape),
                &mut verts,
                &mut idx,
            ),
            1 => shape_sphere(Vec3::from(c.center), c.shape[0], &mut verts, &mut idx),
            2 => shape_capsule(
                Vec3::from(c.center),
                c.shape[0],
                c.shape[1],
                c.shape[2] as u32,
                &mut verts,
                &mut idx,
            ),
            3 => {
                let Some((vb, ib)) = pack.collider_mesh_geom(c.mesh_id) else {
                    continue;
                };
                verts.reserve(vb.len() / 12);
                for i in 0..vb.len() / 12 {
                    verts.push(crate::eftpack::read_vec3(vb, i * 12));
                }
                idx.reserve(ib.len() / 12);
                for t in 0..ib.len() / 12 {
                    let b = t * 12;
                    idx.push([
                        crate::eftpack::read_u32(ib, b),
                        crate::eftpack::read_u32(ib, b + 4),
                        crate::eftpack::read_u32(ib, b + 8),
                    ]);
                }
            }
            _ => continue,
        }
        let aff = c.affine3a();
        let mirror = c.flags & crate::eftpack::col_flags::MIRROR != 0;
        // A door's collision panel must stay PASSABLE, exactly as the render path treats a
        // door-named mesh: transparent to the column cast and kept out of the wall set, with the
        // cell stamped as a door so the router may force an edge through it. EFT gives doors their
        // own layer, so this needs no name matching at all. Without it every mall door became a
        // solid wall and interior reachability fell off a cliff.
        let is_door = pack.layer_name(c.layer) == "DoorLowPolyCollider";
        for t in &idx {
            let (i0, i1, i2) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
                continue;
            }
            let a = aff.transform_point3(verts[i0]);
            let b = aff.transform_point3(verts[i1]);
            let cc = aff.transform_point3(verts[i2]);
            let (e1, e2) = (b - a, cc - a);
            let n = e1.cross(e2);
            let nlen = n.length();
            if nlen < 1.0e-12 {
                continue;
            }
            let mut ny = n.y / nlen;
            if mirror {
                ny = -ny;
            }
            // Same wall/floor split as the render path: near-vertical faces become blocking walls,
            // horizontal-ish ones become floor candidates for the column raycast.
            let span_y = a.y.max(b.y).max(cc.y) - a.y.min(b.y).min(cc.y);
            if !is_door
                && ny.abs() < WALL_MAX_NY
                && (0.5 * nlen >= WALL_MIN_AREA || span_y >= WALL_MIN_SPAN_Y)
            {
                walls.push(Tri {
                    a,
                    b,
                    c: cc,
                    ny,
                    door: false,
                    mat: 0,
                });
            }
            let xz_area2 = (e1.x * e2.z - e1.z * e2.x).abs();
            if xz_area2 < MIN_XZ_AREA2 {
                continue;
            }
            *min_y = min_y.min(a.y.min(b.y.min(cc.y)));
            *max_y = max_y.max(a.y.max(b.y.max(cc.y)));
            tris.push(Tri {
                a,
                b,
                c: cc,
                ny,
                door: is_door,
                mat: 0,
            });
        }
    }
    eprintln!(
        "  nav-bake: colliders {} total -> {} used ({} triggers skipped, {} off-layer, \
         {} nav-ignored by the game, {} map-spanning world backstop)",
        pack.colliders.len(),
        pack.colliders.len()
            - skipped_trigger
            - skipped_layer
            - skipped_navignore
            - skipped_worldfloor,
        skipped_trigger,
        skipped_layer,
        skipped_navignore,
        skipped_worldfloor
    );
    tris.len() - before
}

/// Recast's `rcFilterLedgeSpans`, scaled to this grid and bounded by what a player can VAULT.
///
/// A surface you cannot get down from in one move is a surface you cannot be standing on. Recast
/// marks a span unwalkable when the drop to any orthogonal neighbour exceeds the agent's climb; we
/// use [`VAULT`] instead, because that is the real limit on how much height a player can cross in
/// one move, and at 1 m cells the tighter climb value erodes ordinary kerbs and stair heads.
///
/// This is what removes the "path in the air": the top of a truss beam, a pipe run or a gantry has
/// a multi-metre drop on EVERY side, so every one of its cells is a ledge and the whole surface
/// disappears. A real floor only loses its outermost ring, because its interior cells have
/// neighbours at their own height.
///
/// A neighbour with NO floor at or below `h + VAULT` is SKIPPED rather than counted as a cliff —
/// at 1 m resolution that case is a wall or solid interior, not open air, and treating it as a drop
/// would eat every floor that meets a wall.
fn filter_ledge_spans(heights: &mut [f32], nx: usize, nz: usize, k: usize) -> usize {
    let cells = nx * nz;
    let mut kill = vec![false; cells * k];
    let mut pruned = 0usize;
    const ORTHO: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    for c in 0..cells {
        let (ix, iz) = ((c % nx) as i64, (c / nx) as i64);
        for l in 0..k {
            let h = heights[c * k + l];
            if h <= MISS_HALF {
                break; // floors are ascending; MISS trails
            }
            for (dx, dz) in ORTHO {
                let (jx, jz) = (ix + dx, iz + dz);
                if jx < 0 || jz < 0 || jx >= nx as i64 || jz >= nz as i64 {
                    continue; // off-grid is the map border, not a cliff
                }
                let nc = jz as usize * nx + jx as usize;
                // Highest neighbour floor that is still steppable-onto or droppable-to.
                let mut best = f32::NEG_INFINITY;
                for nl in 0..k {
                    let nh = heights[nc * k + nl];
                    if nh <= MISS_HALF {
                        break;
                    }
                    if nh <= h + VAULT && nh > best {
                        best = nh;
                    }
                }
                if best == f32::NEG_INFINITY {
                    continue; // nothing to step to on this side: wall/solid, not open air
                }
                if best - h < -VAULT {
                    kill[c * k + l] = true;
                    pruned += 1;
                    break;
                }
            }
        }
    }

    // Re-compact each touched cell so the ascending-floors / trailing-MISS invariant survives.
    if pruned > 0 {
        let mut keep: Vec<f32> = Vec::with_capacity(k);
        for c in 0..cells {
            if !(0..k).any(|l| kill[c * k + l]) {
                continue;
            }
            keep.clear();
            for l in 0..k {
                let h = heights[c * k + l];
                if h > MISS_HALF && !kill[c * k + l] {
                    keep.push(h);
                }
            }
            for l in 0..k {
                heights[c * k + l] = keep.get(l).copied().unwrap_or(MISS);
            }
        }
    }
    pruned
}

/// Recast's `minRegionArea` filter: flood the (cell,layer) graph over the edges the router would
/// traverse and blank every region whose area is below `agent().min_region_area`. Returns the
/// number of nodes blanked.
///
/// Blanking a node means shifting its layer out of the cell's ascending height list, so the
/// invariant `nav.bin` relies on (floors ascending, MISS slots trailing) is preserved.
fn prune_small_regions(
    heights: &mut [f32],
    door: &[u8],
    blk: &mut [u8],
    nx: usize,
    nz: usize,
    k: usize,
    res: f32,
) -> usize {
    let a = agent();
    let min_cells = (a.min_region_area / (res * res)).ceil().max(1.0) as usize;
    if min_cells <= 1 {
        return 0; // nothing can be smaller than one node
    }
    let cells = nx * nz;
    let nodes = cells * k;
    let mut seen = vec![false; nodes];
    let mut kill = vec![false; nodes];
    let mut stack: Vec<u32> = Vec::new();
    let mut region: Vec<u32> = Vec::new();

    // The ROUTER's neighbour test, not an approximation of it. This closure decides which nodes
    // count as one region, so `minRegionArea` prunes exactly the islands the router cannot leave.
    // Its old comment claimed "same neighbour test the router uses"; it was wrong three ways, and
    // every one of them made the baker MORE connected than the router, which is the direction that
    // silently ships sealed islands:
    //
    //   1. A forced (door) up-step was unbounded — `up >= 0.0` accepted ANY rise, the exact rule
    //      nav.rs documents as removed for authorising a +9.9 m hop onto a roof. The router caps
    //      every up-move at `vault`, however it is authorised. Measured on interchange: 3,618 of
    //      108,657 forced door edges were accepted here and refused by the router, up to +30.8 m.
    //   2. No diagonal corner test at all, while every router expansion runs `diag_ok`. Streets
    //      had 254,187 diagonals the baker walked and the router will not, keeping 2,765
    //      sub-threshold islands alive (~1,258 m2) that a start can snap onto and never leave.
    //   3. The forced DOWN bound was `max_step(run)` (0.555 orthogonal / 0.785 diagonal) against
    //      the router's flat `drop_max` — different, and orientation-dependent.
    let ortho_ok_bake = |ix: i64, iz: i64, h_ref: f32, blk_c: u8, o: usize| -> bool {
        if (blk_c >> o) & 1 != 0 {
            return false;
        }
        let (dx, dz) = (NB_BAKE[o].0 as i64, NB_BAKE[o].1 as i64);
        let (jx, jz) = (ix + dx, iz + dz);
        if jx < 0 || jz < 0 || jx >= nx as i64 || jz >= nz as i64 {
            return false;
        }
        let oc = (jz * nx as i64 + jx) as usize;
        let nl = best_layer_bake(heights, oc, k, h_ref);
        if nl < 0 {
            return false;
        }
        let up = heights[oc * k + nl as usize] - h_ref;
        let run = ((dx * dx + dz * dz) as f32).sqrt() * res;
        walkable_step_bake(up, run, res)
    };
    let step_ok = |c: usize, l: usize, nc: usize, d: usize| -> Option<usize> {
        let blk_c = blk[c * k + l];
        if (blk_c >> d) & 1 != 0 {
            return None;
        }
        let h = heights[c * k + l];
        let nl = best_layer_bake(heights, nc, k, h);
        if nl < 0 {
            return None;
        }
        let nh = heights[nc * k + nl as usize];
        let (dx, dz) = (NB_BAKE[d].0 as f32, NB_BAKE[d].1 as f32);
        let run = (dx * dx + dz * dz).sqrt() * res;
        let forced = door[c] != 0 || door[nc] != 0;
        let up = nh - h;
        // Mirrors nav.rs `walkable_step(up, run, forced)`: a door waives the step/slope rule but
        // NOT the absolute vault ceiling, and never the drop limit.
        let ok = if forced {
            // nav.json ships `drop_max = free_step(res)`, so that is the router's limit.
            (up >= 0.0 && up <= VAULT) || (up < 0.0 && -up <= free_step(res))
        } else {
            walkable_step_bake(up, run, res)
        };
        if !ok {
            return None;
        }
        let (ix, iz) = ((c % nx) as i64, (c / nx) as i64);
        if dx != 0.0 && dz != 0.0 && !forced {
            let (o1, o2) = match d {
                4 => (0usize, 2usize),
                5 => (0, 3),
                6 => (1, 2),
                7 => (1, 3),
                _ => (0, 0),
            };
            if !ortho_ok_bake(ix, iz, h, blk_c, o1) || !ortho_ok_bake(ix, iz, h, blk_c, o2) {
                return None;
            }
        }
        Some(nl as usize)
    };

    let mut pruned = 0usize;
    for c0 in 0..cells {
        for l0 in 0..k {
            let n0 = c0 * k + l0;
            if seen[n0] || heights[n0] <= MISS_HALF {
                continue;
            }
            region.clear();
            stack.clear();
            stack.push(n0 as u32);
            seen[n0] = true;
            while let Some(n) = stack.pop() {
                region.push(n);
                let (c, l) = (n as usize / k, n as usize % k);
                let (ix, iz) = ((c % nx) as i64, (c / nx) as i64);
                for (d, (dx, dz)) in NB_BAKE.iter().enumerate() {
                    let (jx, jz) = (ix + *dx as i64, iz + *dz as i64);
                    if jx < 0 || jz < 0 || jx >= nx as i64 || jz >= nz as i64 {
                        continue;
                    }
                    let nc = jz as usize * nx + jx as usize;
                    if let Some(nl) = step_ok(c, l, nc, d) {
                        let nn = nc * k + nl;
                        if !seen[nn] {
                            seen[nn] = true;
                            stack.push(nn as u32);
                        }
                    }
                }
            }
            // Area is counted in distinct CELLS (a two-storey stairwell is not "twice the area").
            let mut distinct: std::collections::HashSet<usize> =
                std::collections::HashSet::with_capacity(region.len());
            for &n in &region {
                distinct.insert(n as usize / k);
            }
            if distinct.len() < min_cells {
                for &n in &region {
                    kill[n as usize] = true;
                    pruned += 1;
                }
            }
        }
    }
    if pruned > 0 {
        // Re-compact each touched cell so surviving floors stay ascending with MISS trailing.
        //
        // `blk` MUST be permuted by the SAME keep-mask. It is indexed per (cell, LAYER) and the
        // router reads it by layer index, so compacting heights alone silently re-points every
        // surviving layer at another layer's block mask. Concretely: a cell holding [speck, deck]
        // where the deck has a railing bit; prune the speck, the deck slides to layer 0, and the
        // router reads the speck's empty mask and walks through the railing. That is precisely the
        // routes-through-walls failure this grid exists to prevent, so the two arrays are compacted
        // together or not at all.
        let mut keep: Vec<f32> = Vec::with_capacity(k);
        let mut keep_blk: Vec<u8> = Vec::with_capacity(k);
        for c in 0..cells {
            if !(0..k).any(|l| kill[c * k + l]) {
                continue;
            }
            keep.clear();
            keep_blk.clear();
            for l in 0..k {
                let h = heights[c * k + l];
                if h > MISS_HALF && !kill[c * k + l] {
                    keep.push(h);
                    keep_blk.push(blk[c * k + l]);
                }
            }
            for l in 0..k {
                heights[c * k + l] = keep.get(l).copied().unwrap_or(MISS);
                blk[c * k + l] = keep_blk.get(l).copied().unwrap_or(0);
            }
        }
    }
    pruned
}

/// Unity BoxCollider (`m_Center` + full `m_Size`) -> 12 triangles.
fn shape_box(center: Vec3, size: Vec3, v: &mut Vec<Vec3>, idx: &mut Vec<[u32; 3]>) {
    let h = size * 0.5;
    for &sz in &[-1.0f32, 1.0] {
        for &sy in &[-1.0f32, 1.0] {
            for &sx in &[-1.0f32, 1.0] {
                v.push(center + Vec3::new(h.x * sx, h.y * sy, h.z * sz));
            }
        }
    }
    // corner index = (z<<2)|(y<<1)|x
    const F: [[u32; 3]; 12] = [
        [0, 2, 1],
        [1, 2, 3], // -z
        [4, 5, 6],
        [5, 7, 6], // +z
        [0, 1, 4],
        [1, 5, 4], // -y
        [2, 6, 3],
        [3, 6, 7], // +y
        [0, 4, 2],
        [2, 4, 6], // -x
        [1, 3, 5],
        [3, 7, 5], // +x
    ];
    idx.extend_from_slice(&F);
}

/// Latitude/longitude sphere. Coarse on purpose: a collider sphere only has to be right to well
/// under the nav cell size, and nav grids are baked at ~1 m.
fn shape_sphere(center: Vec3, r: f32, v: &mut Vec<Vec3>, idx: &mut Vec<[u32; 3]>) {
    const RINGS: u32 = 6; // latitude bands
    const SEGS: u32 = 10; // longitude segments
    for i in 0..=RINGS {
        let phi = std::f32::consts::PI * i as f32 / RINGS as f32;
        let (sp, cp) = phi.sin_cos();
        for j in 0..SEGS {
            let th = std::f32::consts::TAU * j as f32 / SEGS as f32;
            let (st, ct) = th.sin_cos();
            v.push(center + Vec3::new(r * sp * ct, r * cp, r * sp * st));
        }
    }
    for i in 0..RINGS {
        for j in 0..SEGS {
            let a = i * SEGS + j;
            let b = i * SEGS + (j + 1) % SEGS;
            let c = (i + 1) * SEGS + j;
            let d = (i + 1) * SEGS + (j + 1) % SEGS;
            // OUTWARD winding. `resolve_column` classifies a surface purely on the sign of `ny`,
            // so an inward-wound primitive has its top read as a CEILING and its underside read as
            // a FLOOR -- inventing a walkable surface in mid-air, the exact artifact this bake is
            // meant to remove. Covered by `collider_primitives_are_wound_outward`.
            idx.push([a, b, c]);
            idx.push([b, d, c]);
        }
    }
}

/// Unity CapsuleCollider: a cylinder of `height` (total, including the two hemisphere caps) with
/// radius `r`, aligned to `dir` (0=X, 1=Y, 2=Z). Approximated by a capped cylinder — exact enough
/// at nav resolution, and the caps matter only for headroom.
fn shape_capsule(
    center: Vec3,
    r: f32,
    height: f32,
    dir: u32,
    v: &mut Vec<Vec3>,
    idx: &mut Vec<[u32; 3]>,
) {
    const SEGS: u32 = 10;
    let half = (height * 0.5 - r).max(0.0); // cylindrical half-length between the cap centres
    let axis = match dir {
        0 => Vec3::X,
        2 => Vec3::Z,
        _ => Vec3::Y,
    };
    // Two orthogonal radial axes.
    let u = if axis.x.abs() < 0.9 {
        Vec3::X.cross(axis)
    } else {
        Vec3::Y.cross(axis)
    }
    .normalize();
    let w = axis.cross(u);
    // Rings at -half-r (pole), -half, +half, +half+r (pole): a cylinder plus flat-ish caps.
    for &(off, rad) in &[
        (-(half + r), 0.0f32),
        (-half, r),
        (half, r),
        (half + r, 0.0),
    ] {
        for j in 0..SEGS {
            let th = std::f32::consts::TAU * j as f32 / SEGS as f32;
            let (st, ct) = th.sin_cos();
            v.push(center + axis * off + (u * ct + w * st) * rad);
        }
    }
    for i in 0..3u32 {
        for j in 0..SEGS {
            let a = i * SEGS + j;
            let b = i * SEGS + (j + 1) % SEGS;
            let c = (i + 1) * SEGS + j;
            let d = (i + 1) * SEGS + (j + 1) % SEGS;
            // OUTWARD winding. `resolve_column` classifies a surface purely on the sign of `ny`,
            // so an inward-wound primitive has its top read as a CEILING and its underside read as
            // a FLOOR -- inventing a walkable surface in mid-air, the exact artifact this bake is
            // meant to remove. Covered by `collider_primitives_are_wound_outward`.
            idx.push([a, b, c]);
            idx.push([b, d, c]);
        }
    }
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
            nodes.push(BvhNode {
                min: mn,
                max: mx,
                start: 0,
                count: 0,
            });
            nodes.push(BvhNode {
                min: mn,
                max: mx,
                start: 0,
                count: 0,
            });
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
    fn column(
        &self,
        x: f32,
        z: f32,
        y_low: f32,
        y_high: f32,
        out: &mut Vec<Hit>,
        stack: &mut Vec<u32>,
    ) {
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
                            out.push(Hit {
                                y,
                                ny: t.ny,
                                door: t.door,
                            });
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
                nodes: vec![BvhNode {
                    min: Vec3::ZERO,
                    max: Vec3::ZERO,
                    start: 0,
                    count: 0,
                }],
                tris,
            };
        }
        let cen: Vec<Vec3> = tris.iter().map(|t| (t.a + t.b + t.c) / 3.0).collect();
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * (n / LEAF_MAX).max(1) + 8);
        nodes.push(BvhNode {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            start: 0,
            count: 0,
        });
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
            nodes.push(BvhNode {
                min: mn,
                max: mx,
                start: 0,
                count: 0,
            });
            nodes.push(BvhNode {
                min: mn,
                max: mx,
                start: 0,
                count: 0,
            });
            nodes[node] = BvhNode {
                min: mn,
                max: mx,
                start: l as u32,
                count: 0,
            };
            stack.push((l, lo, mid));
            stack.push((l + 1, mid, hi));
        }
        let tris_ordered: Vec<Tri> = idx.iter().map(|&i| tris[i as usize]).collect();
        WallBvh {
            nodes,
            tris: tris_ordered,
        }
    }

    /// True if the segment p0->p1 intersects ANY wall triangle. Slab-prune AABBs vs the segment,
    /// Möller–Trumbore at leaves; early-out on first hit. `stack` is a reusable per-thread buffer.
    fn segment_hit(&self, p0: Vec3, p1: Vec3, stack: &mut Vec<u32>) -> bool {
        if self.tris.is_empty() {
            return false;
        }
        let dir = p1 - p0;
        let inv = Vec3::new(
            if dir.x != 0.0 {
                1.0 / dir.x
            } else {
                f32::INFINITY
            },
            if dir.y != 0.0 {
                1.0 / dir.y
            } else {
                f32::INFINITY
            },
            if dir.z != 0.0 {
                1.0 / dir.z
            } else {
                f32::INFINITY
            },
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

    /// Like [`Self::segment_hit`] but returns the triangle that was hit. Only used by the
    /// self-check's failure report: "1 crossing" is not a diagnosis, and the triangle's own normal
    /// and vertical span is what says whether the router clipped a real wall, a stair riser, or a
    /// door panel that should never have been in the wall set at all.
    fn segment_hit_tri(&self, p0: Vec3, p1: Vec3, stack: &mut Vec<u32>) -> Option<Tri> {
        if self.tris.is_empty() {
            return None;
        }
        let dir = p1 - p0;
        let inv = Vec3::new(
            if dir.x != 0.0 {
                1.0 / dir.x
            } else {
                f32::INFINITY
            },
            if dir.y != 0.0 {
                1.0 / dir.y
            } else {
                f32::INFINITY
            },
            if dir.z != 0.0 {
                1.0 / dir.z
            } else {
                f32::INFINITY
            },
        );
        stack.clear();
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if !seg_aabb(p0, inv, dir, node.min, node.max) {
                continue;
            }
            if node.count > 0 {
                let st = node.start as usize;
                for t in &self.tris[st..st + node.count as usize] {
                    if moller_trumbore(p0, dir, t.a, t.b, t.c) {
                        return Some(*t);
                    }
                }
            } else {
                stack.push(node.start);
                stack.push(node.start + 1);
            }
        }
        None
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
                    // The three tests above only compare the triangle's own AABB with the query
                    // box, and that is not the same question. A long slanted wall triangle has a
                    // huge, mostly EMPTY bounding box: accepting on that alone reports a wall in
                    // every cell the box covers, and the clearance pass then deletes floors metres
                    // away from any actual wall. Streets lost 674,570 floors this way, taking whole
                    // interiors (and the ground the spawns stand on) with them. Finish the job with
                    // the exact separating-axis test.
                    if tri_box_overlap(t.a, t.b, t.c, bmin, bmax) {
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
}

/// Exact triangle vs axis-aligned box overlap (Akenine-Möller separating-axis test).
///
/// The 13 axes are: the 3 box normals, the triangle normal, and the 9 edge cross-products. The
/// caller has already done the 3 box-normal tests as a cheap AABB reject, but they are repeated
/// here so this function is correct standalone.
#[inline]
fn tri_box_overlap(a: Vec3, b: Vec3, c: Vec3, bmin: Vec3, bmax: Vec3) -> bool {
    let ctr = (bmin + bmax) * 0.5;
    let h = (bmax - bmin) * 0.5;
    let (v0, v1, v2) = (a - ctr, b - ctr, c - ctr);
    let (e0, e1, e2) = (v1 - v0, v2 - v1, v0 - v2);

    // 9 edge cross-product axes. For axis = edge x boxAxis, the box's projected radius is a fixed
    // combination of two half-extents, so each test is a handful of multiplies.
    //
    // ALL THREE vertices are projected on every axis. The classic Akenine-Moller listing projects
    // only two, because for a given (edge, box-axis) pair the third vertex's projection coincides
    // with one of them -- but WHICH two depends on the edge: (v0,v2) for e0/e1 on the X and Y
    // axes, (v1,v2) for e0 on Z, (v0,v1) for e1 on Z and for e2 on X and Y. This code applied one
    // fixed pattern to all three edges, so on several edge/axis combinations it measured an
    // interval that did not contain the triangle's true extent and reported a SEPARATING axis where
    // none existed. False negatives only: overlapping triangles read as clear. That is the worst
    // possible direction here, because both callers use this to find walls -- the clearance pass
    // and the per-cell `wall_cell` flag -- so missed overlaps become walls the grid does not know
    // about. It let the simplifier straighten a 31 m chord through a 24 m building facade at
    // [-302.5, 3.4, 235.7] with `wall_cell` reading 0 the whole way across.
    //
    // Projecting all three is one extra multiply-add per axis and cannot be got wrong.
    let sep = |p0: f32, p1: f32, p2: f32, rad: f32| -> bool {
        p0.min(p1).min(p2) > rad || p0.max(p1).max(p2) < -rad
    };
    for e in [e0, e1, e2] {
        let (fx, fy, fz) = (e.x.abs(), e.y.abs(), e.z.abs());
        // axis = e x (1,0,0) = (0, e.z, -e.y)
        if sep(
            e.z * v0.y - e.y * v0.z,
            e.z * v1.y - e.y * v1.z,
            e.z * v2.y - e.y * v2.z,
            fz * h.y + fy * h.z,
        ) {
            return false;
        }
        // axis = e x (0,1,0) = (-e.z, 0, e.x)
        if sep(
            -e.z * v0.x + e.x * v0.z,
            -e.z * v1.x + e.x * v1.z,
            -e.z * v2.x + e.x * v2.z,
            fz * h.x + fx * h.z,
        ) {
            return false;
        }
        // axis = e x (0,0,1) = (e.y, -e.x, 0)
        if sep(
            e.y * v0.x - e.x * v0.y,
            e.y * v1.x - e.x * v1.y,
            e.y * v2.x - e.x * v2.y,
            fy * h.x + fx * h.y,
        ) {
            return false;
        }
    }

    // 3 box-normal axes.
    for k in 0..3 {
        let (p0, p1, p2) = (v0[k], v1[k], v2[k]);
        if p0.min(p1).min(p2) > h[k] || p0.max(p1).max(p2) < -h[k] {
            return false;
        }
    }

    // Triangle-plane axis.
    let n = e0.cross(e1);
    let d = -n.dot(v0);
    let r = h.x * n.x.abs() + h.y * n.y.abs() + h.z * n.z.abs();
    d.abs() <= r
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
fn walkable_step_bake(up: f32, run: f32, res: f32) -> bool {
    let a = agent();
    let step = free_step(res);
    if up > 0.0 {
        // slope_tan is the GAME's agentSlope now, so the baker's gate and the router's agree
        // (nav.json ships walk_slope_deg = agentSlope; both sides read the same number). `step` is
        // shared with the router the same way — see `free_step`.
        up <= VAULT && (up <= step || (up <= a.climb && up <= run * a.slope_tan))
    } else {
        // DOWN mirrors the router exactly: nav.json ships `drop_max = free_step(res)`, so a descent
        // is bounded by the same aliasing allowance an ascent is. (Symmetry matters: you must be
        // able to walk back DOWN the staircase you were just allowed to climb.)
        -up <= step.max(a.max_step(run)).min(VAULT)
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
            // SLOPED: `hy` above the floor at EACH end, so the ray tracks the surface the agent
            // would actually walk. Both flat variants were wrong in opposite directions. Flat at
            // the DESTINATION height (`fy1 + hy` at both ends) floats above every riser the router
            // permits, so routes climbed waist-high ledges, barriers and crates. Flat at the SOURCE
            // height (`fy0 + hy` at both ends) is the mirror failure: once the free step became
            // `res*tan(55 deg)` = 0.714 m, a stair riser of 0.634 m per cell straddles the lowest
            // sample at 0.55, so every stair edge the router was finally allowed to take got
            // capsule-blocked right back — the sealed-island bug, reintroduced by the fix for it.
            // Sloped clears both: over a stair the ray rises with the treads and misses the risers,
            // while across level ground it is exactly the old flat ray and still hits real walls.
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

/// Where the capsule fan first touches a wall along `poly`, located to about half a cell, together
/// with whether a door cell sits at that spot. A leg reported as "1 crossing" is only actionable if
/// it names the PLACE — the endpoints of a 250 m route say nothing about which wall it clipped.
fn locate_crossings(
    poly: &[Vec3],
    bvh: &WallBvh,
    res: f32,
    door: &[u8],
    nx: usize,
    nz: usize,
    min_x: f32,
    min_z: f32,
    max_out: usize,
) -> Vec<(Vec3, bool)> {
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    let mut out = Vec::new();
    for w in poly.windows(2) {
        if out.len() >= max_out {
            break;
        }
        let (a, b) = (w[0], w[1]);
        let ex = b.x - a.x;
        let ez = b.z - a.z;
        let el = (ex * ex + ez * ez).sqrt();
        if el < 1.0e-6 {
            continue;
        }
        let (px, pz) = (-ez / el, ex / el);
        let steps = (el / (res * 0.5)).ceil().max(1.0) as usize;
        for si in 0..steps {
            let p = a + (b - a) * (si as f32 / steps as f32);
            let q = a + (b - a) * ((si + 1) as f32 / steps as f32);
            let mut hit: Option<(Tri, f32)> = None;
            'scan: for &o in &CAP_OFF {
                let (ox, oz) = (px * o, pz * o);
                for &hy in &CAP_H {
                    let p0 = Vec3::new(p.x + ox, p.y + hy, p.z + oz);
                    let p1 = Vec3::new(q.x + ox, q.y + hy, q.z + oz);
                    if let Some(t) = bvh.segment_hit_tri(p0, p1, &mut stack) {
                        hit = Some((t, hy));
                        break 'scan;
                    }
                }
            }
            let Some((t, hy)) = hit else { continue };
            let ty = [t.a.y, t.b.y, t.c.y];
            let (tlo, thi) = (
                ty.iter().copied().fold(f32::MAX, f32::min),
                ty.iter().copied().fold(f32::MIN, f32::max),
            );
            eprintln!(
                "  [verify]     tri y[{:.2},{:.2}] span {:.2} m, ny {:.2}, ray at floor+{:.2};                  parent seg [{:.2},{:.2},{:.2}]->[{:.2},{:.2},{:.2}] len {:.2} m",
                tlo, thi, thi - tlo, t.ny, hy, a.x, a.y, a.z, b.x, b.y, b.z, el
            );
            let mid = (p + q) * 0.5;
            let cix = ((mid.x - min_x) / res).round() as i64;
            let ciz = ((mid.z - min_z) / res).round() as i64;
            let mut at_door = false;
            'ring: for dz in -1..=1 {
                for dx in -1..=1 {
                    let (jx, jz) = (cix + dx, ciz + dz);
                    if jx < 0 || jz < 0 || jx >= nx as i64 || jz >= nz as i64 {
                        continue;
                    }
                    if door[(jz * nx as i64 + jx) as usize] != 0 {
                        at_door = true;
                        break 'ring;
                    }
                }
            }
            out.push((mid, at_door));
            if out.len() >= max_out {
                break;
            }
        }
    }
    out
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
        let fan_hits = |p: Vec3, q: Vec3, st: &mut Vec<u32>| -> bool {
            for &o in &CAP_OFF {
                let (ox, oz) = (px * o, pz * o);
                for &hy in &CAP_H {
                    let p0 = Vec3::new(p.x + ox, p.y + hy, p.z + oz);
                    let p1 = Vec3::new(q.x + ox, q.y + hy, q.z + oz);
                    if baked.wall_bvh.segment_hit(p0, p1, st) {
                        return true;
                    }
                }
            }
            false
        };
        if !fan_hits(a, b, &mut stack) {
            continue;
        }
        // LOCATE the hit before excusing it. This used to ask "does ANY cell in a 3x3 ring along
        // this whole segment hold a door?", which excuses a segment that grazes a wall at one end
        // because a door happens to sit at the other -- and with 9.5k door cells on streets a long
        // leg almost always passes near one. That turned the acceptance metric into a rubber stamp
        // (151 of 151 crossings "at a door"). Walk the segment in half-cell steps, find the sub-
        // segments that actually hit, and require a door within ONE cell of that spot.
        let steps = (el / (baked.res * 0.5)).ceil().max(1.0) as usize;
        let mut excused = true;
        let mut any_hit = false;
        for si in 0..steps {
            let t0 = si as f32 / steps as f32;
            let t1 = (si + 1) as f32 / steps as f32;
            let p = a + (b - a) * t0;
            let q = a + (b - a) * t1;
            if !fan_hits(p, q, &mut stack) {
                continue;
            }
            any_hit = true;
            let mid = (p + q) * 0.5;
            let cix = ((mid.x - baked.min_x) / baked.res).round() as i64;
            let ciz = ((mid.z - baked.min_z) / baked.res).round() as i64;
            let mut door_here = false;
            'ring: for dz in -1..=1 {
                for dx in -1..=1 {
                    let (jx, jz) = (cix + dx, ciz + dz);
                    if jx < 0 || jz < 0 || jx >= baked.nx as i64 || jz >= baked.nz as i64 {
                        continue;
                    }
                    if baked.door[(jz * baked.nx as i64 + jx) as usize] != 0 {
                        door_here = true;
                        break 'ring;
                    }
                }
            }
            if !door_here {
                excused = false; // a hit with no door at it: a real wall crossing
                break;
            }
        }
        // Sub-stepping can miss a hit the full-length ray found (the fan is sampled, not swept);
        // if that happens, do NOT excuse it -- an unlocated hit is not a door hit.
        if any_hit && excused {
            n += 1;
        }
    }
    n
}

/// Run the down-cast state machine on one column's hits (mutating `hits` — it is sorted here) and
/// write ascending floor heights into `hout` (length K, pre-filled MISS). Returns (n_floors,
/// is_door). Faithful port of the `nav_cast` kernel: up-facing surfaces are floors iff there is
/// >= `agentHeight` clearance under the last ceiling/floor above; a floor also caps clearance for
/// the floor below it; DOOR faces are transparent (never a surface) but stamp the cell.
///
/// The up-facing threshold is `cos(agentSlope)`, i.e. Recast's `rcMarkWalkableTriangles`: a surface
/// steeper than the agent's slope limit is not a floor at all. It used to be a flat 60°, which
/// recorded 48-60° rubble/embankments as walkable ground the game would never navigate.
/// Returns `(floors_written, is_door, floors_found)`. `floors_found > k` means the column had more
/// walkable surfaces than the grid can hold and the highest ones were dropped — surfaced by the
/// caller, because a layer budget that silently truncates is how the street went missing.
fn resolve_column(
    hits: &mut Vec<Hit>,
    k: usize,
    hout: &mut [f32],
    floors: &mut Vec<f32>,
) -> (usize, bool, usize) {
    floors.clear();
    let mut door_cell = false;
    if hits.is_empty() {
        return (0, false, 0);
    }
    let a = agent();
    let ny_min = a.slope_deg.to_radians().cos();
    hits.sort_unstable_by(|p, q| q.y.total_cmp(&p.y)); // top -> bottom
    let mut last_down = f32::INFINITY;
    for h in hits.iter() {
        if h.door {
            door_cell = true;
            continue; // transparent to the cast
        }
        if h.ny >= ny_min {
            // up-facing floor
            if last_down - h.y >= a.height {
                floors.push(h.y);
            }
            last_down = h.y; // a floor also caps clearance for anything below it
                             // NO early break at `floors.len() >= k`. The scan runs top -> bottom, so stopping once
                             // k floors were collected kept the TOPMOST k and threw the rest away — and the sort
                             // below, which exists precisely to keep the LOWEST k, then had nothing left to choose
                             // from. On Streets that silently deleted the street itself under every building with
                             // more than k stacked surfaces: a player spawn at y = 0.6 found no floor within 8 m and
                             // snapped 20 m up onto the roof, stranding it on a rooftop island. Scanning the whole
                             // column costs a few hits per cell and is the difference between a map that is 40%
                             // connected and one that is whole.
        } else if h.ny <= -ny_min {
            last_down = h.y; // down-facing ceiling / underside
        }
        // near-vertical wall: ignored (also pre-filtered from the BVH)
    }
    floors.sort_unstable_by(|p, q| p.total_cmp(q)); // ascending, MISS slots stay at the end
    let n = floors.len().min(k);
    for (i, &f) in floors.iter().take(n).enumerate() {
        hout[i] = f;
    }
    (n, door_cell, floors.len())
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
    heights: Vec<f32>,  // nx*nz*k, ascending, MISS empty
    door: Vec<u8>,      // nx*nz
    blk: Vec<u8>,       // nx*nz*k, 8-dir edge mask
    wall_cell: Vec<u8>, // nx*nz, 1 = a wall sits in this cell's body column (simplify guard)
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
        // climb/drop_max; the rest are informational). `walk_slope_deg` is now EMITTED (it is the
        // game's agentSlope, no longer a guess the router had to default); `step_up` stays omitted
        // so the router keeps its own default.
        let a = agent();
        // The steepest stair a person still walks up is ~55 deg; at `res` metres per cell that
        // aliases to `res * tan(55)` of rise between adjacent samples. Never below the agent's own
        // climb, never above a vault.
        let stair_step = free_step(self.res);
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
            "climb": a.climb,
            // ledgeDropHeight = 0 on every EFT agent, so a descent is bounded by the same
            // continuous-surface rule as an ascent, NOT by a free-fall allowance. The router
            // applies `max(drop_max, run * tan(walk_slope_deg))` per edge.
            "drop_max": stair_step,
            // STAIRS. A grid this coarse cannot represent a tread. `Sparja_stairs_LOD0` (a normal
            // EFT interior stair) rises 4.02 m over a 3.17 m run — 51.7 deg — in ~13 treads of
            // 0.24 m going and 0.31 m rise. One 0.5 m cell step along it therefore spans about two
            // treads, ~0.62 m of rise, which the agent's own 0.38 m climb rejects outright. The
            // result is not a slightly worse route: EVERY upper floor reached only by stairs became
            // a sealed island. The office at (-99, 7, 285) was 705 nodes with a 0.5 m height band
            // and no path to the street; with this it joins the 1,342,298-node main component.
            //
            // So the free-step allowance is the ALIASING limit of the grid, not a physical stride:
            // res * tan(stair pitch), floored at the agent's climb and capped by what a player can
            // actually clear unaided (VAULT). Bake finer and this shrinks by itself.
            "step_up": stair_step,
            "ledge_drop_height": a.ledge_drop,
            "vault": VAULT,
            "slope_max_deg": a.slope_deg,
            "walk_slope_deg": a.slope_deg,
            "agent_radius": a.radius,
            "agent_height": a.height,
            "min_region_area": a.min_region_area,
            "agent_source": a.source,
            "baker": "atlas-cpu-bvh",
            // Read back by NavGrid::load; a mismatch is reported at error level. See
            // crate::nav::BAKER_VERSION for the bump policy.
            "baker_version": crate::nav::BAKER_VERSION,
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
/// Half-width (m) of the passable disc stamped around a typed door's pivot. Door leaves are ~1 m
/// wide and the pivot sits at the hinge, so this must span the leaf plus a little for the frame.
const DOOR_STAMP_R: f32 = 1.1;

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
    // Footprint probes for the agent-radius erosion: the four cardinal points at agentRadius.
    let erode_climb = agent().climb;
    let er = agent().radius.max(0.05);
    let footprint_probes: [(f32, f32); 4] = [(er, 0.0), (-er, 0.0), (0.0, er), (0.0, -er)];
    #[allow(non_snake_case)]
    let FOOTPRINT_PROBES = footprint_probes;
    let mut heights = vec![MISS; m];
    let mut door = vec![0u8; cells];
    // Columns holding more walkable surfaces than `k` can store. The highest ones are dropped
    // (the lowest k are kept, since that is where agents walk), but a map where this is large is
    // a map baked with too few layers, and that must be visible rather than inferred later from
    // a spawn that mysteriously snapped onto a roof.
    let layer_overflow = std::sync::atomic::AtomicUsize::new(0);
    let deepest_column = std::sync::atomic::AtomicUsize::new(0);
    heights
        .par_chunks_mut(k)
        .zip(door.par_iter_mut())
        .enumerate()
        .for_each_init(
            // `support` is the footprint probe's output buffer and must be K long, exactly like
            // the per-cell `hout` slice `resolve_column` normally writes into.
            || {
                (
                    Vec::<Hit>::with_capacity(64),
                    Vec::<u32>::with_capacity(64),
                    Vec::<f32>::with_capacity(16),
                    vec![MISS; k],
                )
            },
            |(hits, nstack, floors, support), (cell, (hout, dout))| {
                let ix = cell % nx;
                let iz = cell / nx;
                let x = min_x + ix as f32 * res;
                let z = min_z + iz as f32 * res;
                bvh.column(x, z, y_low, y_high, hits, nstack);
                let (n, is_door, found) = resolve_column(hits, k, hout, floors);
                *dout = is_door as u8;
                if found > k {
                    layer_overflow.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    deepest_column.fetch_max(found, std::sync::atomic::Ordering::Relaxed);
                }
                // AGENT-RADIUS EROSION (Recast `rcErodeWalkableArea`), done by SUPERSAMPLING the
                // agent's footprint rather than by dilating the grid.
                //
                // One ray at the cell CENTRE makes a whole 1 m^2 cell walkable when all it hit was
                // a 0.2 m truss beam or a pipe top. Those chain into ramps, and routes climb
                // gantries and roofs. Unity never has this: it erodes the walkable area by
                // agentRadius, deleting any surface narrower than the agent outright.
                //
                // Grid dilation cannot express that at 1 m cells (radius 0.30 m is sub-cell), but
                // the footprint test can: probe the four cardinal points at agentRadius and keep a
                // floor only where the surface is still there within one climb step. A beam,
                // railing or pipe fails on both sides; a walkway, kerb or stair tread passes.
                //
                // MEASURED to be the effective fix, against the alternative (ledge filter) and both
                // together -- real wall crossings on the 256-leg self-check:
                //     neither 76  |  ledge only 108  |  EROSION ONLY 5  |  both 21
                if n > 0 && !is_door {
                    for l in 0..n {
                        let h = hout[l];
                        if h <= MISS_HALF {
                            break;
                        }
                        let mut supported = true;
                        for &(ox, oz) in &FOOTPRINT_PROBES {
                            bvh.column(x + ox, z + oz, y_low, y_high, hits, nstack);
                            let (sn, _, _) = resolve_column(hits, k, support, floors);
                            // Supported if ANY floor under the offset probe is within a climb step
                            // of this one -- the agent's edge still has ground beneath it.
                            if !support[..sn]
                                .iter()
                                .any(|&sh| (sh - h).abs() <= erode_climb)
                            {
                                supported = false;
                                break;
                            }
                        }
                        if !supported {
                            hout[l] = MISS;
                        }
                    }
                    // Re-compact so surviving floors stay ascending with MISS trailing.
                    let mut w = 0usize;
                    for l in 0..k {
                        let h = hout[l];
                        if h > MISS_HALF {
                            hout[w] = h;
                            w += 1;
                        }
                    }
                    for l in w..k {
                        hout[l] = MISS;
                    }
                }
            },
        );

    // Ledge filter runs BEFORE the wall/blk pass and the region prune, so both of those see the
    // final floor set (a blocked-edge bit on a floor that no longer exists is wasted work, and a
    // region's area must be measured over surviving cells).
    let t_ledge = Instant::now();
    // OPT-IN (`EFT_NAV_LEDGE=1`), and OFF by default — it MEASURED WORSE at this resolution.
    // Real wall crossings on the 256-leg self-check: neither 76 | ledge only 108 | erosion only 5 |
    // both 21. Deleting a cell at every drop-off costs 1 m of walkable surface per edge on a 1 m
    // grid, which severs road decks and walkway edges (routes then fall to the terrain UNDER a
    // road) and forces detours that graze walls. Recast can afford the rule because it runs at
    // 0.1667 m voxels, where one voxel of erosion is 6× finer than one of our cells. Kept, off, for
    // the day this bakes at a finer resolution.
    let n_ledge = if std::env::var("EFT_NAV_LEDGE").as_deref() == Ok("1") {
        filter_ledge_spans(&mut heights, nx, nz, k)
    } else {
        0
    };
    if n_ledge > 0 {
        eprintln!(
            "  nav-bake: ledge filter (drop > {VAULT} m vault on any side): removed {n_ledge} floor(s) in {:.2}s",
            t_ledge.elapsed().as_secs_f32()
        );
    }

    let n_over = layer_overflow.load(std::sync::atomic::Ordering::Relaxed);
    if n_over > 0 {
        let deepest = deepest_column.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "  nav-bake: LAYER OVERFLOW: {n_over} column(s) hold more than {k} walkable surfaces              (deepest {deepest}); the lowest {k} are kept and the surfaces above them dropped.              Re-bake with --layers {} to keep all of them.",
            deepest.next_power_of_two().max(k)
        );
    }
    let walkable = heights.par_chunks(k).filter(|c| c[0] > MISS_HALF).count();
    // Stamp door cells from the TYPED door table (gamedata.json), not just from column-ray hits.
    //
    // A door panel is a near-vertical sheet, so its XZ projection is ~a line and it is deliberately
    // skipped when building the column BVH (`MIN_XZ_AREA2`). Only the panel's thin horizontal caps
    // survived, and a cell was stamped only if the ray at its exact CENTRE happened to clip one --
    // so in practice almost no door registered (measured: 21 of interchange's 479 typed doors; ZERO
    // on icebreaker). Door cells are what force an opening to stay passable through the wall mask,
    // so a doorway that never stamps is a doorway the router treats as sealed.
    //
    // The typed doors carry a real world pivot, so stamp a small disc around each one. This is the
    // game's own door table -- derived, not a name-regex guess -- and it covers locked doors too:
    // locked doors stay PASSABLE (the player may hold the key); the route surfaces which key it
    // needs rather than refusing to path.
    let mut stamped_doors = 0usize;
    let mut sealed_doors = 0usize;
    for d in &pack.doors {
        // A door is a portal only if it CAN open.
        //
        // Locked-WITH-a-key stays passable on the reasoning above: the player may hold the key and
        // the route names which one. Locked with NO key is a different object entirely — nothing in
        // the game opens it — so stamping a door cell there carves a permanent hole through a wall
        // that is permanently shut, and the router will happily plan through it. Streets ships 27.
        if d.state.eq_ignore_ascii_case("locked") && d.key_id.is_none() {
            sealed_doors += 1;
            continue;
        }
        let cx = ((d.pivot.x - min_x) / res).round() as i64;
        let cz = ((d.pivot.z - min_z) / res).round() as i64;
        let r = (DOOR_STAMP_R / res).ceil() as i64;
        let mut any = false;
        for dz in -r..=r {
            for dx in -r..=r {
                let (gx, gz) = (cx + dx, cz + dz);
                if gx < 0 || gz < 0 || gx >= nx as i64 || gz >= nz as i64 {
                    continue;
                }
                // Disc, not square: a square would punch holes in walls flanking the doorway.
                let (wx, wz) = (dx as f32 * res, dz as f32 * res);
                if wx * wx + wz * wz > DOOR_STAMP_R * DOOR_STAMP_R {
                    continue;
                }
                let ci = gz as usize * nx + gx as usize;
                // Only stamp cells that actually have a floor — a door disc must not invent
                // walkable space in a void.
                if heights[ci * k] > MISS_HALF {
                    door[ci] = 1;
                    any = true;
                }
            }
        }
        stamped_doors += any as usize;
    }
    eprintln!(
        "  nav-bake: door stamp: {stamped_doors}/{} typed doors marked a nav door cell \
         ({sealed_doors} locked with no key left SEALED)",
        pack.doors.len()
    );
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

    // ---- AGENT-CLEARANCE PASS (Recast `rcErodeWalkableArea`, done properly) --------------------
    //
    // The existing footprint erosion asks "is there still FLOOR beside this cell?" — it never asks
    // "is there a WALL next to it?". A cell 0.1 m from a wall passes it: floor exists on all four
    // sides. So the walkable set contains cells the player capsule does not fit in, and the router
    // then needs a separate per-edge capsule mask (`blk`) to stop routes squeezing through. That
    // mask is what severs connectivity — it takes wall-crossings from 3814 to 75 but costs 57 of
    // 256 legs, and at finer resolutions it severs harder.
    //
    // Unity does not work that way. Recast erodes the walkable field by `agentRadius` ONCE, and
    // connectivity is then plain adjacency in what survives. Do the same: delete any floor whose
    // agent body box intersects a wall. The box is [±r] in XZ over [h+free_step, h+agentHeight], with
    // r = the game's own agentRadius.
    //
    // The payoff is structural. Once every walkable cell is >= r from any wall, two ORTHOGONALLY
    // adjacent walkable cells cannot have a wall between them whenever `res <= 2r` — the wall would
    // have to be within r of one of the two centres. So at res <= 0.6 m the wall-crossing guarantee
    // comes from geometry instead of from a mask, and `blk` can go. (Diagonals span res*sqrt(2) and
    // need res <= 0.42 for the same argument; below that they rely on the existing `diag_ok`
    // two-orthogonal rule, which is why diagonals keep it.)
    //
    // MEASURED against the alternatives on interchange, every config on the same fixed,
    // game-derived inputs (231 patrol legs BSG asserts a bot walks; 278 spawns -> Emercom):
    //
    //   res  clearance  blk | wall-cross  reach AFTER/BEFORE  ledge-violations  patrol  spawns
    //   1.0    off      on  |     14          193 / 250        1013 @ 1.40 m    219/231  278/278
    //   0.5    ON       off |    187          176 / 185              0          217/231  278/278
    //   0.5    ON       ON  |      0          191 / 191              0          216/231  278/278
    //   0.5    off      on  |      4          163 / 232              0             -        -
    //
    // Read the `191 / 191` row: with clearance on, `blk` costs NOTHING. It cost 57 legs at 1.0 m
    // and 69 at 0.5 m without clearance. That is the whole point — once the walkable set is
    // capsule-consistent, the per-edge mask has nothing left to block, and the two mechanisms stop
    // fighting. Dropping `blk` instead (the 187 row) fails badly: `diag_ok` is DRIVEN by blk, so
    // turning it off also disables the diagonal guard, and the simplifier loses its wall guard.
    //
    // Off with `EFT_NAV_CLEARANCE=0` for A/B.
    let clearance_on = std::env::var("EFT_NAV_CLEARANCE").as_deref() != Ok("0");
    if clearance_on && !wall_bvh.tris.is_empty() {
        let t_clr = Instant::now();
        let a = agent();
        let (r, body) = (a.radius, a.height);
        // The band starts above what the agent can simply STEP ONTO, not at the floor. It used to
        // start at h + 0.05, which meant any near-vertical triangle rising out of the floor deleted
        // that floor -- and a STAIR RISER is exactly that. Every tread on every staircase has the
        // next riser (0.63 m at res 0.5) within the agent's 0.30 m radius, so the pass was deleting
        // the stairs themselves, along with kerbs, door thresholds and any low ledge the step rule
        // already permits. That is what shattered the walkable set into 8k islands bounded by cells
        // with no ground floor. Geometry you can step onto is not an obstruction; only what stands
        // above the free step is.
        let foot = free_step(res).max(0.05);
        let removed = std::sync::atomic::AtomicUsize::new(0);
        heights.par_chunks_mut(k).enumerate().for_each_init(
            || Vec::<u32>::with_capacity(64),
            |wstack, (c, hs)| {
                // Doors are exempt for the same reason they are exempt everywhere else: a doorway
                // is narrower than the agent by design, and the game models it as passable.
                if door[c] != 0 {
                    return;
                }
                let ix = c % nx;
                let iz = c / nx;
                let cx = min_x + ix as f32 * res;
                let cz = min_z + iz as f32 * res;
                let mut w = 0usize;
                for l in 0..k {
                    let h = hs[l];
                    if h <= MISS_HALF {
                        break; // ascending, MISS trails
                    }
                    let bmin = Vec3::new(cx - r, h + foot, cz - r);
                    let bmax = Vec3::new(cx + r, h + body, cz + r);
                    if wall_bvh.box_overlaps(bmin, bmax, wstack) {
                        removed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        continue; // capsule does not fit — not walkable
                    }
                    hs[w] = h;
                    w += 1;
                }
                // Re-compact: the nav.bin invariant is floors ASCENDING with MISS trailing, and
                // every reader breaks at the first MISS.
                for l in w..k {
                    hs[l] = MISS;
                }
            },
        );
        eprintln!(
            "  nav-bake: agent-clearance pass (r={r:.2} m, body={body:.2} m): removed {} floor(s) \
             the player capsule does not fit in, in {:.2}s",
            removed.load(std::sync::atomic::Ordering::Relaxed),
            t_clr.elapsed().as_secs_f32()
        );
    }

    // `blk`/`wall_cell` exist to stop routes squeezing past walls the cell-centre sample missed.
    // With the clearance pass on they are redundant by construction (see above), and they are the
    // main cost to connectivity, so allow turning them off: `EFT_NAV_BLK=0`.
    let blk_on = std::env::var("EFT_NAV_BLK").as_deref() != Ok("0");
    let t_blk = Instant::now();
    let mut blk = vec![0u8; m];
    // Per-cell "a wall occupies my body column" flag (see WallBvh::box_overlaps). Half-extent =
    // half a cell + the capsule radius, so it covers any wall a chord passing anywhere in the cell
    // (±res/2) could clip within ±PLAYER_RADIUS — the sub-cell walls the per-edge blk mask misses.
    let mut wall_cell = vec![0u8; cells];
    let wc_half = res * 0.5 + PLAYER_RADIUS;
    // Runs over whatever `heights` currently holds. It has to be repeatable because pruning below
    // changes `heights`, and every bit in `blk` encodes a test against a SPECIFIC neighbour layer.
    let capsule_pass = |heights: &[f32], blk: &mut [u8], wall_cell: &mut [u8]| {
        blk.par_chunks_mut(k)
            .zip(wall_cell.par_iter_mut())
            .enumerate()
            .for_each_init(
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
                            if !walkable_step_bake(up, horiz, res) {
                                continue; // an edge the router would never traverse — don't bother
                            }
                            if door_c || door[nc] != 0 {
                                continue; // doors stay transparent (never blocked)
                            }
                            let ncx = min_x + jx as f32 * res;
                            let ncz = min_z + jz as f32 * res;
                            if capsule_blocked(
                                cx, cz, floor_c, ncx, ncz, floor_nc, &wall_bvh, wstack,
                            ) {
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
                            // Body band top = the highest point the ACCEPTANCE ray can reach: a chord
                            // may ride up to `chord_rise_max` above the floor, and its topmost capsule
                            // sample sits CAP_H[2] above that. This was `PLAYER_HEIGHT + STEP_UP_NAV`,
                            // citing `segment_clear`'s float_tol — a symbol that no longer exists, and
                            // a constant (0.45) the rise tolerance stopped tracking when the free step
                            // became res-dependent. It left ~0.11 m of band unguarded at res 0.5 and
                            // 0.60 m at res 1.0, which cannot route anyone through a wall (it is above
                            // head height) but DOES let count_wall_crossings report a crossing on a
                            // chord segment_clear certified as clear, polluting the metric that is
                            // supposed to be zero.
                            let band = crate::nav::chord_rise_max(free_step(res)) + CAP_H[2];
                            let bmin = Vec3::new(cx - wc_half, floor_c, cz - wc_half);
                            let bmax = Vec3::new(cx + wc_half, floor_c + band, cz + wc_half);
                            if wall_bvh.box_overlaps(bmin, bmax, wstack) {
                                *wc = 1;
                                break;
                            }
                        }
                    }
                },
            );
    };
    if blk_on && !wall_bvh.tris.is_empty() {
        capsule_pass(&heights, &mut blk, &mut wall_cell);
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

    // ---- Recast `minRegionArea`: discard islands too small to stand on ------------------------
    // Unity's build settings ship minRegionArea = 2.0 m² for every EFT agent. Recast drops any
    // connected walkable region below it, which is what stops one-cell specks -- the top of a
    // bollard, a lamp housing, a pipe flange -- being navmesh. They matter here because a route
    // snapped onto a speck is a route that can never leave it.
    //
    // Run AFTER the capsule pass so connectivity is measured over the edges the ROUTER will
    // actually traverse (blocked edges included), not over raw floor adjacency.
    let n_pruned = prune_small_regions(&mut heights, &door, &mut blk, nx, nz, k, res);
    if n_pruned > 0 {
        let a = agent();
        eprintln!(
            "  nav-bake: minRegionArea {:.1} m² ({} cell(s) @ {}m): pruned {} node(s) in undersized islands",
            a.min_region_area,
            (a.min_region_area / (res * res)).ceil().max(1.0) as usize,
            res,
            n_pruned
        );
        // RE-RUN the capsule pass on the pruned heights. Permuting `blk` alongside `heights` (which
        // prune_small_regions does) keeps each surviving layer pointing at its OWN mask, but it
        // cannot fix the other half of the staleness: a bit was computed against whichever
        // neighbour layer `best_layer` chose at the time, and deleting that layer makes the router
        // choose a different one — an edge whose capsule test was never run. That is a route
        // through a wall from a pass that was meant to prevent them, and it is not hypothetical:
        // it is the single crossing that survived on streets at [-302.5, 3.4, 235.7].
        if blk_on && !wall_bvh.tris.is_empty() {
            let t_re = Instant::now();
            for b in blk.iter_mut() {
                *b = 0;
            }
            for w in wall_cell.iter_mut() {
                *w = 0;
            }
            capsule_pass(&heights, &mut blk, &mut wall_cell);
            eprintln!(
                "  nav-bake: capsule pass re-run on the pruned grid: {} blocked edge-bits, {} wall cells in {:.2}s",
                blk.iter().map(|b| b.count_ones() as usize).sum::<usize>(),
                wall_cell.iter().filter(|&&w| w != 0).count(),
                t_re.elapsed().as_secs_f32()
            );
        }
    }

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

/// Does a drawn route stay ON the floor it claims to walk?
///
/// The wall-crossing metric CANNOT see this. `walls` holds only near-vertical triangles
/// (`|ny| < WALL_MAX_NY`), so every horizontal floor and ceiling is excluded from that BVH by
/// construction — a segment that passes straight through a storey slab scores zero wall crossings.
/// That blind spot is exactly the "route goes through a ceiling" report.
///
/// So sample the polyline every `STEP` metres and ask the grid what floor exists under each sample.
/// A sample further than `TOL` from the nearest floor at its own XZ is airborne or buried: the line
/// is not on any walkable surface there.
type FloorWorst = Option<(Vec3, (Vec3, Vec3))>;
fn count_floor_violations(poly: &[Vec3], grid: &NavGrid) -> (usize, usize, f32, FloorWorst) {
    const STEP: f32 = 1.0;
    const TOL: f32 = 1.5; // generous: the drawn line rides ~0.1 m proud, and stairs read as steps
    let (mut samples, mut bad, mut worst) = (0usize, 0usize, 0.0f32);
    let mut worst_at: Option<Vec3> = None;
    let mut worst_seg: Option<(Vec3, Vec3)> = None;
    for w in poly.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        let n = (len / STEP).ceil().max(1.0) as usize;
        for i in 0..=n {
            let p = a.lerp(b, i as f32 / n as f32);
            samples += 1;
            if !grid.on_floor(p.x, p.z, p.y, TOL) {
                bad += 1;
                // Depth of the miss, measured against the containing cell (None = over a void).
                let d = grid
                    .floor_near(p.x, p.z, p.y)
                    .map_or(99.0, |f| (p.y - f).abs());
                if d > worst {
                    worst = d;
                    worst_at = Some(p);
                    worst_seg = Some((a, b));
                }
            }
        }
    }
    (
        samples,
        bad,
        worst,
        worst_at.map(|p| (p, worst_seg.unwrap_or((p, p)))),
    )
}

/// Could a BOT actually walk this drawn line?
///
/// Floor adherence proves the line sits ON a floor; it does not prove the floor UNDER the line is
/// continuous. A route can hug the ground the whole way and still step off a 2 m ledge, or up onto
/// a crate — the "traverse the top of a fuel tanker" class of bug. Unity forbids both: every EFT
/// agent has `ledgeDropHeight = 0` and `maxJumpAcrossDistance = 0`, so the navmesh contains no
/// drop-down and no jump links at all, and a bot may only move where the surface is CONTINUOUS
/// within one climb step (or along a slope).
///
/// So walk the polyline in short increments and track the floor beneath it, always taking the layer
/// nearest the previous one so a multi-storey column cannot make the surface teleport. Any change
/// larger than the agent's own `max_step` for that increment is a step no bot could take.
/// Returns (samples, illegal, worst delta).
fn count_ledge_violations(poly: &[Vec3], grid: &NavGrid) -> (usize, usize, f32) {
    const STEP: f32 = 0.25; // short: a coarse stride hides a ledge inside one interval
                            // The floor can only change where the GRID changes, so judge a delta against the most
                            // permissive single move the router itself allows — a diagonal step, run = res*sqrt(2). Using
                            // max_step(STEP) instead would flag every legal 1 m router step as a violation and measure the
                            // sampling stride rather than the route. Anything over this is a step the router should never
                            // have taken, at any resolution.
    let limit = agent().max_step(grid.res * std::f32::consts::SQRT_2);
    let (mut samples, mut bad, mut worst) = (0usize, 0usize, 0.0f32);
    let mut prev: Option<f32> = None;
    for w in poly.windows(2) {
        let (a, b) = (w[0], w[1]);
        let n = (a.distance(b) / STEP).ceil().max(1.0) as usize;
        for i in 0..=n {
            let p = a.lerp(b, i as f32 / n as f32);
            // Track the SAME surface: nearest layer to where we already are, not to the drawn y.
            let here = grid.surface_near(p.x, p.z, prev.unwrap_or(p.y));
            let Some(h) = here else {
                prev = None;
                continue;
            };
            if let Some(ph) = prev {
                samples += 1;
                let d = (h - ph).abs();
                if d > limit {
                    bad += 1;
                    if d > worst {
                        worst = d;
                    }
                }
            }
            prev = Some(h);
        }
    }
    (samples, bad, worst)
}

// ---- headless self-check + MACHINE PROOF (FIX 5): route many legs, assert ZERO wall-crossings ---

/// Extended self-check: load the freshly baked grid, route 200+ varied legs, and for EVERY segment
/// of each SIMPLIFIED route sample the player-capsule fan against the KEPT wall BVH — the AFTER
/// wall-crossing count MUST be 0. For contrast it re-routes the SAME legs on a grid with the wall
/// mask disabled (reproducing OLD no-nav_blk.bin routing) and counts BEFORE crossings. Also reports
/// reachability, and prints one wall-threading BEFORE leg's coords (for a before/after screenshot).
/// How many pairs of NON-ADJACENT legs of a tour cross each other in plan view, and how far the
/// walked distance exceeds the straight line through the same stops.
///
/// This is the "does the route double back on itself" question. It is deliberately measured on the
/// STOP sequence rather than on the dense polyline: a walkable route legitimately weaves around
/// obstacles at metre scale, and counting those weaves would drown the signal. A tour whose
/// stop-to-stop legs cross is an ORDERING failure — the run visits A, walks past B to C, then comes
/// back for B — which is what a player sees as an obviously silly route.
fn tour_self_crossings(stops: &[Vec3]) -> usize {
    // Proper segment intersection in XZ, endpoints excluded (consecutive legs share one by
    // construction, and three collinear stops are not a crossing).
    let side = |a: Vec3, b: Vec3, p: Vec3| -> f32 {
        (b.x - a.x) * (p.z - a.z) - (b.z - a.z) * (p.x - a.x)
    };
    let mut n = 0usize;
    for i in 0..stops.len().saturating_sub(1) {
        for j in (i + 2)..stops.len().saturating_sub(1) {
            let (a, b, c, d) = (stops[i], stops[i + 1], stops[j], stops[j + 1]);
            let (d1, d2) = (side(c, d, a), side(c, d, b));
            let (d3, d4) = (side(a, b, c), side(a, b, d));
            if ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0)) {
                n += 1;
            }
        }
    }
    n
}

/// Number of independent verification workers to use for one grid.
///
/// Every worker owns one [`Scratch`] (four 4-byte arrays per nav node), which is about 396 MiB on
/// Customs. Blindly using Rayon's full machine-wide thread count can therefore exhaust RAM on the
/// exact large maps whose verification takes longest. Cap the default at four CPUs and a 1 GiB
/// aggregate scratch budget. Advanced benchmarkers can tune the two caps without changing output.
fn verification_jobs(nodes: usize) -> usize {
    const BYTES_PER_NODE: usize = 16;
    const DEFAULT_MEMORY_MIB: usize = 1024;
    const MAX_AUTO_JOBS: usize = 4;

    let cpus = std::thread::available_parallelism().map_or(1, usize::from);
    let memory_mib = std::env::var("EFT_NAV_VERIFY_MEMORY_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MEMORY_MIB);
    let per_worker = nodes.saturating_mul(BYTES_PER_NODE).max(1);
    let memory_jobs = memory_mib
        .saturating_mul(1024 * 1024)
        .checked_div(per_worker)
        .unwrap_or(1)
        .max(1);
    let automatic = cpus.min(MAX_AUTO_JOBS).min(memory_jobs).max(1);
    std::env::var("EFT_NAV_VERIFY_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(automatic)
        .min(cpus)
        .min(memory_jobs)
        .max(1)
}

fn verification_chunk_len(items: usize, jobs: usize) -> usize {
    items.div_ceil(jobs.max(1)).max(1)
}

/// Route the map's REAL loot runs and hold them to the same wall/floor/ledge rules as the random
/// legs. Worth doing separately because random walkable cells are overwhelmingly open street: the
/// planner's stops are containers and loose spawns, which sit indoors, in corners, on shelves and
/// behind doors — exactly the geometry a cell-centre sampler under-covers, and exactly where a
/// route that clips a wall is most likely and least visible.
///
/// Runs the shipped planner (`planner::solve`), not a reimplementation of it, so the thing under
/// test is the stitched multi-leg tour the user actually sees, seams included.
fn loot_plan_check(baked: &Baked, dir: &Path, grid: &NavGrid, jobs: usize) {
    let Ok(txt) = std::fs::read_to_string(dir.join("gamedata.json")) else {
        return;
    };
    let Ok(gd) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return;
    };
    let pt = |v: &serde_json::Value| -> Option<Vec3> {
        let a = v.get("pos")?.as_array()?;
        (a.len() >= 3).then(|| {
            Vec3::new(
                a[0].as_f64().unwrap_or(0.0) as f32,
                a[1].as_f64().unwrap_or(0.0) as f32,
                a[2].as_f64().unwrap_or(0.0) as f32,
            )
        })
    };
    let empty = vec![];
    let mut cands: Vec<crate::planner::Cand> = Vec::new();
    for key in ["containers", "loose_points"] {
        for e in gd.get(key).and_then(|v| v.as_array()).unwrap_or(&empty) {
            let Some(pos) = pt(e) else { continue };
            cands.push(crate::planner::Cand {
                name: e
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                value: 10_000,
                score_value: 10_000.0,
                pos,
                loot_s: 8.0,
            });
        }
    }
    let extracts: Vec<(String, Vec3)> = gd
        .get("exfils")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty)
        .iter()
        .filter_map(|e| Some((e.get("name")?.as_str()?.to_string(), pt(e)?)))
        .collect();
    if cands.len() < 8 || extracts.is_empty() {
        return;
    }
    // Starts: player spawns, spread deterministically across the list.
    let spawns: Vec<Vec3> = gd
        .get("spawn_points")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty)
        .iter()
        .filter(|s| {
            s.get("categories")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().any(|c| c.as_str() == Some("player")))
                .unwrap_or(false)
        })
        .filter_map(pt)
        .collect();
    if spawns.is_empty() {
        return;
    }
    let want = 12usize.min(spawns.len());
    let stride = (spawns.len() / want).max(1);
    let starts: Vec<Vec3> = (0..want)
        .map(|i| spawns[(i * stride) % spawns.len()])
        .collect();

    #[derive(Default)]
    struct LootCheck {
        planned: bool,
        failed: bool,
        stops: usize,
        metres: f32,
        segs: usize,
        cross: usize,
        cross_door: usize,
        selfx: usize,
        straight: f32,
        lg_n: usize,
        lg_bad: usize,
        lg_worst: f32,
        fs_n: usize,
        fs_bad: usize,
        fs_worst: f32,
        worst_leg: Option<(Vec3, usize)>,
    }

    let t_plans = Instant::now();
    // `par_chunks`, rather than one Rayon task per spawn, bounds concurrent full-grid scratch
    // buffers to `jobs`. Each chunk is serial internally and results collect in original spawn
    // order, so the first-offender report remains deterministic.
    let batches: Vec<Vec<LootCheck>> = starts
        .par_chunks(verification_chunk_len(starts.len(), jobs))
        .map(|batch| {
            batch
                .iter()
                .map(|&start| {
                    let Ok(plan) = crate::planner::solve(
                        grid,
                        start,
                        cands.clone(),
                        extracts.clone(),
                        12,
                        1800.0,
                        None,
                    ) else {
                        return LootCheck {
                            failed: true,
                            ..Default::default()
                        };
                    };
                    // start -> every stop -> the extract it actually ends at. The extract leg is
                    // part of total_dist, so leaving it out of the straight-line baseline would
                    // inflate the ratio and make an honest route look like a bad one.
                    let mut seq: Vec<Vec3> = vec![start];
                    seq.extend(plan.stops.iter().map(|st| st.pos));
                    if let Some(e) = extracts.iter().find(|e| e.0 == plan.extract) {
                        seq.push(e.1);
                    }
                    let (segs, cross) = count_wall_crossings(&plan.polyline, &baked.wall_bvh);
                    let cross_door = count_door_crossings(&plan.polyline, baked);
                    let (lg_n, lg_bad, lg_worst) = count_ledge_violations(&plan.polyline, grid);
                    let (fs_n, fs_bad, fs_worst, _) = count_floor_violations(&plan.polyline, grid);
                    LootCheck {
                        planned: true,
                        stops: plan.stops.len(),
                        metres: plan.total_dist,
                        segs,
                        cross,
                        cross_door,
                        selfx: tour_self_crossings(&seq),
                        straight: seq.windows(2).map(|w| w[0].distance(w[1])).sum(),
                        lg_n,
                        lg_bad,
                        lg_worst,
                        fs_n,
                        fs_bad,
                        fs_worst,
                        worst_leg: (cross > cross_door).then_some((start, cross - cross_door)),
                        ..Default::default()
                    }
                })
                .collect()
        })
        .collect();

    let (mut planned, mut failed) = (0usize, 0usize);
    let (mut segs, mut cross, mut cross_door) = (0usize, 0usize, 0usize);
    let (mut stops, mut metres) = (0usize, 0.0f32);
    let (mut selfx, mut straight_total) = (0usize, 0.0f32);
    let (mut lg_n, mut lg_bad, mut lg_worst) = (0usize, 0usize, 0.0f32);
    let (mut fs_n, mut fs_bad, mut fs_worst) = (0usize, 0usize, 0.0f32);
    let mut worst_leg: Option<(Vec3, usize)> = None;
    for result in batches.into_iter().flatten() {
        planned += result.planned as usize;
        failed += result.failed as usize;
        stops += result.stops;
        metres += result.metres;
        segs += result.segs;
        cross += result.cross;
        cross_door += result.cross_door;
        selfx += result.selfx;
        straight_total += result.straight;
        lg_n += result.lg_n;
        lg_bad += result.lg_bad;
        lg_worst = lg_worst.max(result.lg_worst);
        fs_n += result.fs_n;
        fs_bad += result.fs_bad;
        fs_worst = fs_worst.max(result.fs_worst);
        if worst_leg.is_none() {
            worst_leg = result.worst_leg;
        }
    }
    eprintln!(
        "  [verify] loot plans: evaluated {want} spawn(s) on {jobs} CPU worker(s) in {:.2}s",
        t_plans.elapsed().as_secs_f32()
    );
    if planned == 0 {
        eprintln!("  [verify] loot plans: none of {want} spawn(s) could produce a run");
        return;
    }
    let real = cross.saturating_sub(cross_door);
    eprintln!(
        "  [verify] loot plans: {planned}/{want} solved ({failed} refused), {stops} stop(s), \
         {metres:.0} m of tour over {segs} segment(s)"
    );
    eprintln!(
        "  [verify] loot plans: wall-crossings {real} (+{cross_door} door-frame graze(s)); \
         ledge {lg_bad}/{lg_n} worst {lg_worst:.2} m; floor {fs_bad}/{fs_n} worst {fs_worst:.2} m"
    );
    eprintln!(
        "  [verify] loot plans: {selfx} self-crossing leg pair(s); walked {:.2}x the straight line \
         through the same stops",
        metres / straight_total.max(1.0)
    );
    if let Some((at, n)) = worst_leg {
        eprintln!(
            "  [verify] loot plans: first offending run starts at [{:.1},{:.1},{:.1}] with {n} crossing(s)",
            at.x, at.y, at.z
        );
    }
    if real == 0 && lg_bad == 0 && fs_bad == 0 {
        eprintln!(
            "  [verify] loot plans: PASS (no wall crossing, no illegal ledge, never off the floor)"
        );
    } else {
        eprintln!("  [verify] loot plans: FAIL");
    }
}

fn self_check(baked: &Baked, dir: &Path) {
    let t_verify = Instant::now();
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
        eprintln!(
            "  [verify] only {} walkable cell(s) — no route to test",
            walk.len()
        );
        return;
    }
    let world = |w: &(usize, usize, f32)| {
        Vec3::new(
            baked.min_x + w.0 as f32 * baked.res,
            w.2,
            baked.min_z + w.1 as f32 * baked.res,
        )
    };

    // Deterministic varied legs: coprime-ish index strides sweep start/dest position, direction and
    // length across the whole walkable set (no RNG dep).
    let n = walk.len();
    let want = 256usize;
    // Strides must be genuinely COPRIME with n, not "coprime-ish". `n/3` shares n's factor of 3
    // whenever 3 | n, so `i*sa % n` cycled through THREE distinct start cells and the 256 "varied"
    // legs were 256 samples of 3 start points — which is how a bake could report 8 offending legs
    // that were really one bad corner cell counted 7 times. Walk up from a low-discrepancy
    // (golden-ratio) offset until gcd == 1, which always terminates since gcd(1, n) == 1.
    let coprime = |mut st: usize| -> usize {
        let gcd = |mut a: usize, mut b: usize| {
            while b != 0 {
                let t = a % b;
                a = b;
                b = t;
            }
            a
        };
        st = st.max(1).min(n - 1);
        while gcd(st, n) != 1 {
            st -= 1;
            if st == 0 {
                return 1;
            }
        }
        st
    };
    let sa = coprime((n as f64 * 0.618_033_988_7) as usize);
    let sb = coprime((n as f64 * 0.381_966_011_3) as usize);
    let min_span2 = (8.0f32 / baked.res).powi(2); // non-trivial legs (>= ~8 m)
    let jobs = verification_jobs(grid.nodes());
    let scratch_mib = grid.nodes().saturating_mul(16) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "  [verify] route proof: {jobs} CPU worker(s), about {scratch_mib:.0} MiB scratch each"
    );

    // Build the exact deterministic leg list first. Parallel execution below consumes this list in
    // order-preserving chunks, so counts, first examples and failure details remain byte-for-byte
    // deterministic regardless of worker scheduling.
    let mut legs: Vec<(Vec3, Vec3)> = Vec::with_capacity(want);
    let mut i = 0usize;
    while legs.len() < want && i < want * 8 {
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
        legs.push((world(&s), world(&d)));
    }

    #[derive(Default)]
    struct AfterCheck {
        routed: bool,
        segs: usize,
        cross: usize,
        cross_raw: usize,
        cross_door: usize,
        lg_total: usize,
        lg_bad: usize,
        lg_worst: f32,
        fs_total: usize,
        fs_bad: usize,
        fs_worst: f32,
        fs_at: FloorWorst,
        bad_wall: Option<(Vec3, Vec3, usize, Vec<(Vec3, bool)>)>,
    }

    let t_after = Instant::now();
    let after_batches: Vec<Vec<AfterCheck>> = legs
        .par_chunks(verification_chunk_len(legs.len(), jobs))
        .map(|batch| {
            let mut scratch = Scratch::new(grid.nodes());
            batch
                .iter()
                .map(|&(a, b)| {
                    let Some((raw, simp)) = grid.route_debug(a, b, &mut scratch) else {
                        return AfterCheck::default();
                    };
                    let (segs, cross) = count_wall_crossings(&simp, &baked.wall_bvh);
                    // Attribute the crossing: does the offending segment pass through a DOOR cell?
                    let cross_door = count_door_crossings(&simp, baked);
                    let (_, cross_raw) = count_wall_crossings(&raw, &baked.wall_bvh);
                    let (lg_total, lg_bad, lg_worst) = count_ledge_violations(&simp, &grid);
                    let (fs_total, fs_bad, fs_worst, fs_at) = count_floor_violations(&simp, &grid);
                    let bad_wall = (cross > cross_door).then(|| {
                        let at = locate_crossings(
                            &simp,
                            &baked.wall_bvh,
                            baked.res,
                            &baked.door,
                            baked.nx,
                            baked.nz,
                            baked.min_x,
                            baked.min_z,
                            4,
                        );
                        (a, b, cross - cross_door, at)
                    });
                    AfterCheck {
                        routed: true,
                        segs,
                        cross,
                        cross_raw,
                        cross_door,
                        lg_total,
                        lg_bad,
                        lg_worst,
                        fs_total,
                        fs_bad,
                        fs_worst,
                        fs_at,
                        bad_wall,
                    }
                })
                .collect()
        })
        .collect();

    let (mut routes_after, mut routes_before) = (0usize, 0usize);
    let (mut segs_after, mut segs_before) = (0usize, 0usize);
    let (mut cross_after, mut cross_before) = (0usize, 0usize);
    let mut cross_raw = 0usize; // crossings on the RAW (unsimplified) A* path — attributes the gap
    let mut cross_after_door = 0usize; // AFTER crossings whose segment passes through a door cell
                                       // Floor adherence: the wall metric is blind to horizontal surfaces (see
                                       // `count_floor_violations`), so track how much of each drawn route is nowhere near a floor.
    let (mut fs_total, mut fs_bad, mut fs_worst) = (0usize, 0usize, 0.0f32);
    let mut fs_at: FloorWorst = None;
    // Ledge legality: the floor UNDER the drawn line must be continuous within one climb step.
    let (mut lg_total, mut lg_bad, mut lg_worst) = (0usize, 0usize, 0.0f32);
    // Offending legs, so a failure is a place to go LOOK at rather than a number.
    let mut bad_walls: Vec<(Vec3, Vec3, usize, Vec<(Vec3, bool)>)> = Vec::new();
    let attempts = legs.len();
    let mut example: Option<(Vec3, Vec3)> = None;
    for result in after_batches.into_iter().flatten() {
        routes_after += result.routed as usize;
        segs_after += result.segs;
        cross_after += result.cross;
        cross_after_door += result.cross_door;
        cross_raw += result.cross_raw;
        lg_total += result.lg_total;
        lg_bad += result.lg_bad;
        lg_worst = lg_worst.max(result.lg_worst);
        fs_total += result.fs_total;
        fs_bad += result.fs_bad;
        if result.fs_worst > fs_worst {
            fs_worst = result.fs_worst;
            fs_at = result.fs_at;
        }
        if bad_walls.len() < 8 {
            if let Some(bad) = result.bad_wall {
                bad_walls.push(bad);
            }
        }
    }
    let after_secs = t_after.elapsed().as_secs_f32();

    #[derive(Default)]
    struct BeforeCheck {
        routed: bool,
        segs: usize,
        cross: usize,
        example: Option<(Vec3, Vec3)>,
    }
    let t_before = Instant::now();
    let before_batches: Vec<Vec<BeforeCheck>> = legs
        .par_chunks(verification_chunk_len(legs.len(), jobs))
        .map(|batch| {
            let mut scratch = Scratch::new(grid_before.nodes());
            batch
                .iter()
                .map(|&(a, b)| {
                    let Some((poly, _)) = grid_before.path(a, b, &mut scratch, None) else {
                        return BeforeCheck::default();
                    };
                    let (segs, cross) = count_wall_crossings(&poly, &baked.wall_bvh);
                    BeforeCheck {
                        routed: true,
                        segs,
                        cross,
                        example: (cross > 0).then_some((a, b)),
                    }
                })
                .collect()
        })
        .collect();
    for result in before_batches.into_iter().flatten() {
        routes_before += result.routed as usize;
        segs_before += result.segs;
        cross_before += result.cross;
        if example.is_none() {
            example = result.example; // first leg the OLD router threaded through a wall
        }
    }
    let before_secs = t_before.elapsed().as_secs_f32();
    eprintln!("  [verify] route proof timing: AFTER {after_secs:.2}s; BEFORE {before_secs:.2}s");

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
    eprintln!(
        "  [verify] floor adherence: {fs_bad}/{fs_total} sampled metres of the SIMPLIFIED routes sit \
         >1.5 m from any floor at their own XZ (worst {fs_worst:.1} m) — this is the metric the \
         wall check cannot see, since walls exclude horizontal surfaces"
    );
    eprintln!(
        "  [verify] ledge legality: {lg_bad}/{lg_total} sampled steps along the SIMPLIFIED routes          exceed the most permissive single router move ({:.2} m for a diagonal at res {:.2}); worst          {lg_worst:.2} m — every EFT agent has ledgeDropHeight = 0, so a bot may only move where          the surface is continuous",
        agent().max_step(grid.res * std::f32::consts::SQRT_2),
        grid.res
    );
    if let Some((p, (sa, sb))) = fs_at {
        eprintln!(
            "  [verify]   worst floor sample ({:.1}, {:.1}, {:.1}) lies on the chord              ({:.1},{:.1},{:.1}) -> ({:.1},{:.1},{:.1})  [chord length {:.1} m, dy {:.1} m]",
            p.x, p.y, p.z, sa.x, sa.y, sa.z, sb.x, sb.y, sb.z,
            sa.distance(sb),
            sb.y - sa.y
        );
        let floors = grid.floors_at(p.x, p.z);
        eprintln!("  [verify]   floors present at that XZ: {floors:?}");
    }
    for (a, b, c, at) in &bad_walls {
        eprintln!(
            "  [verify]   wall leg: ({:.0},{:.0},{:.0}) -> ({:.0},{:.0},{:.0})  {c} crossing(s)",
            a.x, a.y, a.z, b.x, b.y, b.z
        );
        for (p, door) in at {
            eprintln!(
                "  [verify]     hits wall at [{:.1},{:.1},{:.1}]{}",
                p.x,
                p.y,
                p.z,
                if *door { "  (a door cell is here)" } else { "" }
            );
        }
    }
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
        eprintln!(
            "  [verify] FAIL: {cross_after_walls} wall-crossing(s) remain on the simplified routes"
        );
    }
    let t_loot = Instant::now();
    loot_plan_check(baked, dir, &grid, jobs);
    eprintln!(
        "  [verify] timing: machine proof {:.2}s; loot plans {:.2}s; total {:.2}s",
        after_secs + before_secs,
        t_loot.elapsed().as_secs_f32(),
        t_verify.elapsed().as_secs_f32()
    );
}

// ---- CLI entry: `atlas check-nav <pack_dir> --to "<exfil>" [--side ...]` -----------------------

/// Route EVERY spawn point in the pack to one extract and report the ones that cannot get there.
///
/// This is the acceptance test a nav bake actually has to pass. The self-check's random cell pairs
/// measure the grid in the abstract; this measures the thing a player cares about — can you leave
/// the map from where the game puts you. Spawn points and extracts both come from gamedata.json
/// (the game's own tables), so nothing here is authored.
///
/// Exfils are matched on the SERIALIZED name; the game's English display names live in the shared
/// tarkov.dev locale (e.g. `SE Exfil` is "Emercom Checkpoint"), so either string is accepted.
pub fn run_check_cli(args: &[String]) -> i32 {
    let mut pack_dir: Option<String> = None;
    let mut want: Option<String> = None;
    let mut side = "all".to_string();
    let mut from_pt: Option<Vec3> = None;
    let mut loot_mode = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--from" => {
                i += 1;
                let parsed = args.get(i).and_then(|v| {
                    let n: Vec<f32> = v
                        .split(',')
                        .filter_map(|t| t.trim().parse::<f32>().ok())
                        .collect();
                    (n.len() == 3).then(|| Vec3::new(n[0], n[1], n[2]))
                });
                match parsed {
                    Some(v) => from_pt = Some(v),
                    None => {
                        eprintln!("check-nav: --from needs x,y,z");
                        return 2;
                    }
                }
            }
            "--loot" => loot_mode = true,
            "--to" => {
                i += 1;
                match args.get(i) {
                    Some(v) => want = Some(v.clone()),
                    None => {
                        eprintln!("check-nav: --to needs an exfil name");
                        return 2;
                    }
                }
            }
            "--side" => {
                i += 1;
                match args.get(i) {
                    Some(v) => side = v.to_lowercase(),
                    None => {
                        eprintln!("check-nav: --side needs pmc|scav|all");
                        return 2;
                    }
                }
            }
            s if pack_dir.is_none() => pack_dir = Some(s.to_string()),
            s => {
                eprintln!("check-nav: unexpected argument '{s}'");
                return 2;
            }
        }
        i += 1;
    }
    let Some(dir) = pack_dir else {
        eprintln!(
            "usage: atlas check-nav <pack_dir> [--to \"<exfil name>\"] [--from x,y,z] [--loot] \
             [--side player|pmc|scav|all|patrols]\n  \
             player = every spawn a human can start on (categories contains `player`, side \
             all/pmc) — the set that matters; pmc = the literal `side=pmc` points only\n  --from routes ONE arbitrary point (a spot on the road, say) to every matched exfil, asserting walls/floor/ledges on the route it draws\n  --loot runs the REAL loot-run planner over the containers + loose points and asserts the same properties on the tour it returns"
        );
        return 2;
    };
    let root = Path::new(&dir);

    let gd: serde_json::Value = match std::fs::read_to_string(root.join("gamedata.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
    {
        Some(v) => v,
        None => {
            eprintln!("check-nav: no readable gamedata.json in {dir}");
            return 1;
        }
    };
    let Some(grid) = NavGrid::load(root) else {
        eprintln!("check-nav: no nav grid in {dir} — run `atlas bake-nav` first");
        return 1;
    };

    // Locale: the serialized exfil id -> English display name (tarkov.dev `maps_en`), so `--to`
    // accepts either. Missing cache is fine; matching then falls back to the serialized name.
    let locale: serde_json::Value = std::fs::read_to_string(
        crate::paths::shared_dir()
            .join(".tarkov-json-cache")
            .join("maps_en.json"),
    )
    .ok()
    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    .and_then(|v| v.get("data").cloned())
    .unwrap_or(serde_json::Value::Null);
    let display = |name: &str| -> String {
        locale
            .get(name)
            .and_then(|v| v.as_str())
            .unwrap_or(name)
            .to_string()
    };

    let empty = vec![];
    let exfils = gd
        .get("exfils")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let want_s = want.unwrap_or_default();
    let wl = want_s.to_lowercase();
    let mut targets: Vec<(String, String, Vec3)> = Vec::new();
    for e in exfils {
        let name = e
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let disp = display(&name);
        let p = e.get("pos").and_then(|v| v.as_array());
        let Some(p) = p else { continue };
        if p.len() < 3 {
            continue;
        }
        let pos = Vec3::new(
            p[0].as_f64().unwrap_or(0.0) as f32,
            p[1].as_f64().unwrap_or(0.0) as f32,
            p[2].as_f64().unwrap_or(0.0) as f32,
        );
        if wl.is_empty() || name.to_lowercase().contains(&wl) || disp.to_lowercase().contains(&wl) {
            targets.push((name, disp, pos));
        }
    }
    if targets.is_empty() {
        eprintln!("check-nav: no exfil matches '{want_s}'. Available:");
        for e in exfils {
            let n = e.get("name").and_then(|v| v.as_str()).unwrap_or("");
            eprintln!("    {:<34} {}", n, display(n));
        }
        return 1;
    }

    // --patrols: route every PatrolWay leg and report the ones whose route CLIMBS far above both
    // of its endpoints. A patrol leg between two ground-level waypoints has no business going up a
    // gantry or onto a roof, so this names the offending legs (and the height they reach) instead
    // of leaving "the path goes up in the air" to be diagnosed by eye.
    if side == "patrols" {
        let ways = gd
            .get("patrol_ways")
            .and_then(|v| v.as_array())
            .unwrap_or(&empty);
        let pt = |v: &serde_json::Value| -> Option<Vec3> {
            let a = v.as_array()?;
            if a.len() < 3 {
                return None;
            }
            Some(Vec3::new(
                a[0].as_f64()? as f32,
                a[1].as_f64()? as f32,
                a[2].as_f64()? as f32,
            ))
        };
        let mut sc = Scratch::new(grid.nodes());
        let (mut legs, mut routed, mut climbers) = (0usize, 0usize, 0usize);
        let mut worst: Vec<(f32, Vec3, Vec3, Vec3)> = Vec::new();
        for w in ways {
            let Some(pts) = w.get("points").and_then(|v| v.as_array()) else {
                continue;
            };
            let ps: Vec<Vec3> = pts.iter().filter_map(pt).collect();
            for pair in ps.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                legs += 1;
                let Some((poly, _)) = grid.path(a, b, &mut sc, None) else {
                    continue;
                };
                routed += 1;
                let base = a.y.max(b.y);
                let mut top = f32::NEG_INFINITY;
                let mut at = a;
                for p in &poly {
                    if p.y > top {
                        top = p.y;
                        at = *p;
                    }
                }
                let rise = top - base;
                if rise > 2.0 {
                    climbers += 1;
                    worst.push((rise, at, a, b));
                }
            }
        }
        worst.sort_by(|x, y| y.0.total_cmp(&x.0));
        println!("\n=== patrol legs that climb above BOTH endpoints ===");
        println!("  {legs} legs, {routed} routed, {climbers} climb >2 m above their endpoints");
        for (rise, at, a, b) in worst.iter().take(25) {
            println!(
                "   +{:6.1} m  peak [{:8.1},{:7.1},{:8.1}]   leg [{:.0},{:.0},{:.0}] -> [{:.0},{:.0},{:.0}]",
                rise, at.x, at.y, at.z, a.x, a.y, a.z, b.x, b.y, b.z
            );
        }
        return if climbers > 0 { 1 } else { 0 };
    }

    // The walkable set's connectivity, computed with the ROUTER's own edge rule. "Spawn X cannot
    // reach exfil Y" has two entirely different causes -- X never snapped onto the mesh at all, or
    // X snapped into a sealed island -- and reporting a bare percentage cannot distinguish them.
    // Every earlier pass at this reachability number was tuning knobs without knowing which of the
    // two it was looking at.
    let (comp, comp_sizes) = grid.components();
    let mut order: Vec<usize> = (0..comp_sizes.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(comp_sizes[i]));
    let total_nodes: u64 = comp_sizes.iter().map(|&v| v as u64).sum();
    let cell_m2 = (grid.res * grid.res) as f64;
    println!("\n=== walkable connectivity ===");
    println!(
        "  {} component(s) over {} walkable node(s)",
        comp_sizes.len(),
        total_nodes
    );
    // Extent + vertical band per component. "8034 components" is a number; "this 122k-node island
    // spans the whole map but only 2 m of height" is a diagnosis.
    let mut bbox: Vec<(Vec3, Vec3)> =
        vec![(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)); comp_sizes.len()];
    for (n, &lb) in comp.iter().enumerate() {
        if lb < 0 {
            continue;
        }
        let p = grid.node_pos(n);
        let e = &mut bbox[lb as usize];
        e.0 = e.0.min(p);
        e.1 = e.1.max(p);
    }
    for &i in order.iter().take(6) {
        let (lo, hi) = bbox[i];
        println!(
            "   #{:<6} {:>10} node(s)  {:>5.1}%  ~{:>8.0} m2   x[{:>7.0},{:>7.0}] z[{:>7.0},{:>7.0}] y[{:>6.1},{:>6.1}]",
            i,
            comp_sizes[i],
            100.0 * comp_sizes[i] as f64 / total_nodes.max(1) as f64,
            comp_sizes[i] as f64 * cell_m2,
            lo.x, hi.x, lo.z, hi.z, lo.y, hi.y
        );
    }
    let main_comp = order.first().copied().map(|i| i as i32).unwrap_or(-1);
    let comp_of = |p: Vec3| grid.component_at(&comp, p);
    let comp_desc = |c: Option<i32>| match c {
        None => "NO SNAP (no floor within 8 m)".to_string(),
        Some(c) if c == main_comp => format!("#{c} (main)"),
        Some(c) => format!(
            "#{c} SEALED ISLAND, {} node(s) ~{:.0} m2",
            comp_sizes[c as usize],
            comp_sizes[c as usize] as f64 * cell_m2
        ),
    };
    for (name, disp, tgt) in &targets {
        println!(
            "  exfil {:<26} {} -> {}",
            name,
            disp,
            comp_desc(comp_of(*tgt))
        );
    }

    // --loot: run the REAL planner over this pack's own loot and hold the tours to the properties
    // a grid can check. The flag used to be parsed and then ignored -- `loot_mode` was assigned and
    // never read -- while the usage text promised it "runs the REAL loot-run planner". A flag that
    // lies is worse than a missing one, because it reports a check that never ran.
    //
    // What it CANNOT do here is the wall assertion: that needs the collider BVH, which only exists
    // during a bake. `bake-nav` runs the identical planner with the wall test attached; this is the
    // fast pass you can run against a finished pack without re-baking, and it says so.
    if loot_mode {
        let mut rc = 0;
        let pt = |v: &serde_json::Value| -> Option<Vec3> {
            let a = v.get("pos")?.as_array()?;
            (a.len() >= 3).then(|| {
                Vec3::new(
                    a[0].as_f64().unwrap_or(0.0) as f32,
                    a[1].as_f64().unwrap_or(0.0) as f32,
                    a[2].as_f64().unwrap_or(0.0) as f32,
                )
            })
        };
        let mut cands: Vec<crate::planner::Cand> = Vec::new();
        for key in ["containers", "loose_points"] {
            for e in gd.get(key).and_then(|v| v.as_array()).unwrap_or(&empty) {
                let Some(pos) = pt(e) else { continue };
                cands.push(crate::planner::Cand {
                    name: e
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    value: 10_000,
                    score_value: 10_000.0,
                    pos,
                    loot_s: 8.0,
                });
            }
        }
        let ex_all: Vec<(String, Vec3)> = targets.iter().map(|(n, _, p)| (n.clone(), *p)).collect();
        let starts: Vec<Vec3> = match from_pt {
            Some(f) => vec![f],
            None => gd
                .get("spawn_points")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty)
                .iter()
                .filter(|sp| {
                    sp.get("categories")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().any(|c| c.as_str() == Some("player")))
                        .unwrap_or(false)
                })
                .filter_map(pt)
                .collect(),
        };
        if cands.len() < 8 || ex_all.is_empty() || starts.is_empty() {
            eprintln!(
                "check-nav --loot: need loot, extracts and a start (have {} loot, {} exfil(s), {} start(s))",
                cands.len(),
                ex_all.len(),
                starts.len()
            );
            return 1;
        }
        let want = 12usize.min(starts.len());
        let stride = (starts.len() / want).max(1);
        println!(
            "\n=== loot plans ({want} start(s), {} candidate stop(s)) ===",
            cands.len()
        );
        let (mut solved, mut refused) = (0usize, 0usize);
        // Bucketed by CAUSE. "7 refused" on its own reads as a routing failure, and on interchange
        // it is not one: every refusal there is this harness's own budget (1800 s, 12 stops, every
        // candidate flat-valued) failing to fit a tour from a start far from the loot mass, while
        // the field reached the map fine. A count that cannot tell those apart sends you looking
        // in the router for a tuning artifact.
        let mut why: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let (mut lg_n, mut lg_bad, mut lg_worst) = (0usize, 0usize, 0.0f32);
        let (mut fs_n, mut fs_bad, mut fs_worst) = (0usize, 0usize, 0.0f32);
        let (mut selfx, mut walked, mut straight) = (0usize, 0.0f32, 0.0f32);
        for i in 0..want {
            let start = starts[(i * stride) % starts.len()];
            match crate::planner::solve(
                &grid,
                start,
                cands.clone(),
                ex_all.clone(),
                12,
                1800.0,
                None,
            ) {
                Ok(plan) => {
                    solved += 1;
                    walked += plan.total_dist;
                    let (n, b, w) = count_ledge_violations(&plan.polyline, &grid);
                    lg_n += n;
                    lg_bad += b;
                    lg_worst = lg_worst.max(w);
                    let (n2, b2, w2, _) = count_floor_violations(&plan.polyline, &grid);
                    fs_n += n2;
                    fs_bad += b2;
                    fs_worst = fs_worst.max(w2);
                    let mut seq: Vec<Vec3> = vec![start];
                    seq.extend(plan.stops.iter().map(|st| st.pos));
                    if let Some(e) = ex_all.iter().find(|e| e.0 == plan.extract) {
                        seq.push(e.1);
                    }
                    selfx += tour_self_crossings(&seq);
                    straight += seq.windows(2).map(|w| w[0].distance(w[1])).sum::<f32>();
                    println!(
                        "  OK   [{:>7.1},{:>6.1},{:>7.1}]  {:>2} stop(s), {:>5.0} m, exits {}",
                        start.x,
                        start.y,
                        start.z,
                        plan.stops.len(),
                        plan.total_dist,
                        plan.extract
                    );
                }
                Err(e) => {
                    refused += 1;
                    *why.entry(if e.contains("off the walkable mesh") {
                        "start is not on the nav mesh (ROUTING)"
                    } else if e.contains("budget") {
                        "no tour fits the time budget (TUNING, not routing)"
                    } else if e.contains("extract") {
                        "no extract reachable within the budget"
                    } else {
                        "no loot above the value filter"
                    })
                    .or_default() += 1;
                    println!(
                        "  none [{:>7.1},{:>6.1},{:>7.1}]  {e}",
                        start.x, start.y, start.z
                    );
                }
            }
        }
        if solved == 0 {
            println!("  no start could produce a run");
            return 1;
        }
        println!(
            "  {solved}/{want} solved ({refused} refused); ledge {lg_bad}/{lg_n} worst {lg_worst:.2} m; \
             floor {fs_bad}/{fs_n} worst {fs_worst:.2} m"
        );
        for (cause, n) in &why {
            println!("    {n:>3} refused: {cause}");
        }
        println!(
            "  {selfx} self-crossing leg pair(s); walked {:.2}x the straight line through the same stops",
            walked / straight.max(1.0)
        );
        if lg_bad > 0 || fs_bad > 0 {
            rc = 1;
            println!("  FAIL: a planned tour leaves the floor or takes a step the router forbids");
        } else {
            println!(
                "  PASS: every tour stays on the floor and every step is one the router allows.\n  \
                 NOTE: wall crossings are NOT checked here (that needs the collider BVH) \u{2014} \
                 `atlas bake-nav <pack>` runs this same planner with the wall test attached."
            );
        }
        return rc;
    }

    // --from: one arbitrary point (a spot on the road, a place the user got stuck) to every matched
    // exfil. Reports the component first, so a failure names its cause instead of just failing, and
    // asserts on the drawn route the two properties a grid can check without the collider BVH:
    // the line stays on a floor, and every step it takes is one the router itself would allow.
    if let Some(from) = from_pt {
        let mut sc = Scratch::new(grid.nodes());
        let mut rc = 0;
        let fc = comp_of(from);
        println!(
            "\n=== from [{:.1}, {:.1}, {:.1}] ===",
            from.x, from.y, from.z
        );
        println!("  component: {}", comp_desc(fc));
        if let Some(c) = fc {
            let (lo, hi) = bbox[c as usize];
            println!(
                "  island extent: x[{:.1},{:.1}] z[{:.1},{:.1}] y[{:.1},{:.1}]",
                lo.x, hi.x, lo.z, hi.z, lo.y, hi.y
            );
            if c != main_comp {
                let br = grid.island_boundary(&comp, c, 6);
                println!(
                    "  what seals it: {} wall edge(s), {} step edge(s) (min rise {:.2} m, min drop {:.2} m), {} diagonal edge(s)",
                    br.wall, br.step, br.min_up, br.min_down, br.diag
                );
                for (a, b, why, up) in &br.examples {
                    println!(
                        "    {:<5} [{:>7.1},{:>6.1},{:>7.1}] -> [{:>7.1},{:>6.1},{:>7.1}]  rise {:+.2} m",
                        why, a.x, a.y, a.z, b.x, b.y, b.z, up
                    );
                }
            }
        }
        let (mut ok, mut bad) = (0usize, 0usize);
        for (name, disp, tgt) in &targets {
            let tc = comp_of(*tgt);
            // The visualization must not change WHICH routes exist. It used to: `path_traced` was
            // a copy of `path` that never grew the island rescue or the destination re-snap, so
            // ticking "visualize search" quietly downgraded the router and exfils whose recorded
            // point is a trigger-volume centre stopped resolving. They share one implementation
            // now, and this asserts they still agree.
            let traced = grid.path_traced(from, *tgt, &mut sc, None, 4096).is_some();
            match grid.path(from, *tgt, &mut sc, None) {
                Some((pts, len)) if pts.len() >= 2 => {
                    ok += 1;
                    if !traced {
                        rc = 1;
                        println!(
                            "  DISAGREE {name:<26} plain routing finds a path but the traced                              (visualize search) variant does not"
                        );
                    }
                    let (lg_n, lg_bad, lg_worst) = count_ledge_violations(&pts, &grid);
                    let (fs_n, fs_bad, fs_worst, _) = count_floor_violations(&pts, &grid);
                    let straight = from.distance(*tgt).max(0.001);
                    println!(
                        "  OK   {:<26} {:>7.0} m walked ({:.2}x straight), {} pt(s); \
                         ledge {}/{} worst {:.2} m; floor {}/{} worst {:.2} m",
                        name,
                        len,
                        len / straight,
                        pts.len(),
                        lg_bad,
                        lg_n,
                        lg_worst,
                        fs_bad,
                        fs_n,
                        fs_worst
                    );
                    if lg_bad > 0 || fs_bad > 0 {
                        rc = 1;
                    }
                }
                _ => {
                    bad += 1;
                    rc = 1;
                    if traced {
                        println!(
                            "  DISAGREE {name:<26} the traced variant finds a path but plain                              routing does not"
                        );
                    }
                    println!(
                        "  FAIL {:<26} {} -- exfil is in {}",
                        name,
                        if fc.is_some() && fc == tc {
                            "same component, so this is an A* failure, not connectivity"
                        } else {
                            "different component"
                        },
                        comp_desc(tc)
                    );
                }
            }
            let _ = disp;
        }
        println!("  {ok} reachable, {bad} not");
        return rc;
    }

    let spawns = gd
        .get("spawn_points")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let mut sc = Scratch::new(grid.nodes());
    let mut rc = 0;

    for (name, disp, tgt) in &targets {
        let mut ok = 0usize;
        let mut fail: Vec<(String, String, Vec3)> = Vec::new();
        let mut skipped = 0usize;
        for s in spawns {
            let sside = s
                .get("side")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase();
            // `--side player` selects where a HUMAN actually starts, which is not the same set as
            // `--side pmc`. On streets the 40 `side=pmc` points carry categories bit8/bit16/bit32
            // and no `player` at all, while the 241 real player starts are `side=all` with
            // `categories=[player]` — so a plain side match had never once tested the spawns the
            // game drops a PMC on.
            let is_player = s
                .get("categories")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().any(|c| c.as_str() == Some("player")))
                .unwrap_or(false);
            let keep = match side.as_str() {
                "all" => true,
                "player" => is_player && matches!(sside.as_str(), "all" | "pmc"),
                other => sside == other,
            };
            if !keep {
                continue;
            }
            let Some(p) = s.get("pos").and_then(|v| v.as_array()) else {
                continue;
            };
            if p.len() < 3 {
                continue;
            }
            let from = Vec3::new(
                p[0].as_f64().unwrap_or(0.0) as f32,
                p[1].as_f64().unwrap_or(0.0) as f32,
                p[2].as_f64().unwrap_or(0.0) as f32,
            );
            let sname = s
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string();
            match grid.path(from, *tgt, &mut sc, None) {
                Some((pts, _)) if pts.len() >= 2 => ok += 1,
                _ => {
                    skipped += 0;
                    fail.push((sname, sside, from))
                }
            }
        }
        let total = ok + fail.len() + skipped;
        let pct = if total > 0 {
            100.0 * ok as f32 / total as f32
        } else {
            0.0
        };
        println!(
            "\n=== {} ({}) at [{:.1}, {:.1}, {:.1}] ===",
            disp, name, tgt.x, tgt.y, tgt.z
        );
        println!("  {ok}/{total} spawns can reach it ({pct:.1}%)   [side filter: {side}]");
        if !fail.is_empty() {
            rc = 1;
            println!("  UNREACHABLE ({}):", fail.len());
            // Bucket the failures by CAUSE before listing them: a hundred spawns sharing one
            // sealed island is one bug, while a hundred spawns in the main component failing to
            // route is a completely different one, and the flat list of coordinates this used to
            // print made the two look identical.
            let mut by_cause: std::collections::BTreeMap<String, usize> = Default::default();
            for (_, _, p) in &fail {
                *by_cause.entry(comp_desc(comp_of(*p))).or_default() += 1;
            }
            for (cause, n) in &by_cause {
                println!("    {n:>4} x {cause}");
            }
            for (n, sd, p) in fail.iter().take(12) {
                println!(
                    "    {:<22} {:<5} at [{:>8.1},{:>7.1},{:>8.1}]   EFT_ROUTE=\"{:.2},{:.2},{:.2};{:.2},{:.2},{:.2}\"",
                    n, sd, p.x, p.y, p.z, p.x, p.y, p.z, tgt.x, tgt.y, tgt.z
                );
            }
            if fail.len() > 12 {
                println!("    ... and {} more", fail.len() - 12);
            }
        }
    }
    rc
}

// ---- CLI entry: `atlas bake-nav <pack_dir> [--res R] [--layers K]` -----------------------------

/// Handle the headless `bake-nav` subcommand. `args` is argv AFTER the "bake-nav" token. Returns a
/// process exit code (0 = ok). Never panics (release is panic=abort): all failures return non-zero.
pub fn run_cli(args: &[String]) -> i32 {
    let mut pack_dir: Option<String> = None;
    // 0.5 m, not 1 m. At 1 m the grid cannot tell a 1.11 m LEDGE from a 48-degree ramp (the slope
    // term allows `run * tan(slope)` per step), so routes stepped off ledges: 1013 illegal steps
    // with a worst of 1.40 m on the self-check. At 0.5 m that is zero. It also halves the height
    // the diagonal rule permits, which is what makes the clearance pass sufficient. Costs 4x the
    // grid (nav.bin 43 MB -> 173 MB).
    let mut res: f32 = 0.5;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every generated collider triangle must face OUTWARD. `resolve_column` classifies a surface
    /// purely on the sign of `ny` (`nav_bake.rs` up-facing = floor, down-facing = ceiling), so an
    /// inward-wound primitive has its top read as a ceiling and its underside read as a FLOOR —
    /// i.e. it invents a walkable surface in mid-air, the exact artifact this bake exists to avoid.
    fn assert_outward(verts: &[Vec3], idx: &[[u32; 3]], centre: Vec3, what: &str) {
        for (i, t) in idx.iter().enumerate() {
            let (a, b, c) = (
                verts[t[0] as usize],
                verts[t[1] as usize],
                verts[t[2] as usize],
            );
            let n = (b - a).cross(c - a);
            // A lat/long sphere collapses all SEGS vertices onto each pole, so the polar band is
            // made of zero-area triangles with no meaningful orientation. `add_collider_tris`
            // discards them on the same basis (`nlen < 1e-12`), so skip them here rather than
            // asserting on floating-point noise.
            if n.length() < 1.0e-6 {
                continue;
            }
            let outward = (a + b + c) / 3.0 - centre;
            assert!(
                n.dot(outward) > 0.0,
                "{what}: triangle {i} is wound INWARD (n·out = {})",
                n.dot(outward)
            );
        }
    }

    #[test]
    fn collider_primitives_are_wound_outward() {
        let (mut v, mut i) = (Vec::new(), Vec::new());
        shape_box(Vec3::ZERO, Vec3::splat(2.0), &mut v, &mut i);
        assert_outward(&v, &i, Vec3::ZERO, "box");

        v.clear();
        i.clear();
        shape_sphere(Vec3::ZERO, 1.0, &mut v, &mut i);
        assert_outward(&v, &i, Vec3::ZERO, "sphere");

        for dir in 0..3u32 {
            v.clear();
            i.clear();
            shape_capsule(Vec3::ZERO, 0.5, 2.0, dir, &mut v, &mut i);
            assert_outward(&v, &i, Vec3::ZERO, "capsule");
        }
    }

    /// The baker decides which edges to capsule-test; the router decides which edges to walk. If
    /// the baker is STRICTER, it skips the wall test on an edge the router will take, and that
    /// edge's block bit stays 0 — a route through a wall, which is the failure this whole grid is
    /// built to prevent. They must agree exactly.
    #[test]
    fn baker_and_router_agree_on_walkability() {
        let a = agent();
        // Sweep the RESOLUTIONS we actually bake at. The old test pinned one grid built from
        // hardcoded defaults, so it stayed green through the exact divergence it exists to catch:
        // the baker gated on a constant 0.45 while nav.json shipped res*tan(55 deg) (0.714 at
        // res 0.5) to the router, handing it 4,445 edges the capsule pass never tested.
        for res in [0.25f32, 0.5, 1.0] {
            let step = free_step(res);
            let grid = NavGrid::test_grid(a.climb, a.slope_deg, VAULT, step);
            for mul in [1.0f32, std::f32::consts::SQRT_2] {
                let run = res * mul;
                let mut up = -2.5f32;
                while up <= 2.5 {
                    let baker = walkable_step_bake(up, run, res);
                    let router = grid.walkable_step_pub(up, run, false);
                    assert_eq!(
                        baker, router,
                        "disagree at res={res} up={up:.3} run={run:.3}:                          baker={baker} router={router} (step={step:.3})"
                    );
                    // FORCED (door) edges too. Sweeping only forced=false is exactly why the
                    // prune flood's unbounded `up >= 0.0` survived: the one test that exists to
                    // catch baker/router divergence never looked at the branch that diverged.
                    let bf = (up >= 0.0 && up <= VAULT) || (up < 0.0 && -up <= free_step(res));
                    let rf = grid.walkable_step_pub(up, run, true);
                    assert_eq!(
                        bf, rf,
                        "FORCED disagree at res={res} up={up:.3} run={run:.3}:                          baker={bf} router={rf}"
                    );
                    up += 0.01;
                }
            }
        }
    }

    /// `tri_box_overlap` must never report a separating axis that does not exist: it is what both
    /// the clearance pass and `wall_cell` use to find walls, so a false "clear" is a wall the grid
    /// does not know about. Cross-checked against a dense point sample of each triangle, which is
    /// slow but assumption-free — an analytic "reference" would just be the same algorithm twice.
    #[test]
    fn tri_box_overlap_has_no_false_negatives() {
        // A deterministic spread of awkward triangles: axis-aligned, oblique, thin slivers, and
        // the tall thin vertical facade that actually broke this (24 m of wall, 0.3 m of width).
        let tris = [
            (
                Vec3::new(0.0, -12.0, 0.0),
                Vec3::new(0.0, 12.0, 0.0),
                Vec3::new(0.3, 12.0, 0.2),
            ),
            (
                Vec3::new(-1.0, 0.1, -1.0),
                Vec3::new(1.0, 0.1, -1.0),
                Vec3::new(0.0, 0.1, 1.0),
            ),
            (
                Vec3::new(-2.0, -2.0, -2.0),
                Vec3::new(2.0, 1.0, -0.5),
                Vec3::new(0.1, 2.0, 1.7),
            ),
            (
                Vec3::new(0.4, -3.0, 0.4),
                Vec3::new(0.41, 3.0, 0.4),
                Vec3::new(0.4, 0.0, 0.41),
            ),
            (
                Vec3::new(-5.0, 0.6, 0.2),
                Vec3::new(5.0, 0.6, 0.25),
                Vec3::new(0.0, 0.65, 0.3),
            ),
        ];
        let h = Vec3::new(0.55, 1.075, 0.55);
        let mut checked = 0usize;
        for (ti, (a, b, c)) in tris.iter().enumerate() {
            // Sweep the box centre over a lattice around the triangle.
            for gx in -6..=6 {
                for gy in -6..=6 {
                    for gz in -6..=6 {
                        let ctr = Vec3::new(gx as f32 * 0.4, gy as f32 * 0.4, gz as f32 * 0.4);
                        let (bmin, bmax) = (ctr - h, ctr + h);
                        let got = tri_box_overlap(*a, *b, *c, bmin, bmax);
                        if got {
                            continue; // a positive is allowed to be conservative
                        }
                        // Claimed clear: no barycentric sample of the triangle may lie inside.
                        const N: usize = 40;
                        for i in 0..=N {
                            for j in 0..=(N - i) {
                                let (u, v) = (i as f32 / N as f32, j as f32 / N as f32);
                                let p = *a + (*b - *a) * u + (*c - *a) * v;
                                assert!(
                                    p.x < bmin.x
                                        || p.x > bmax.x
                                        || p.y < bmin.y
                                        || p.y > bmax.y
                                        || p.z < bmin.z
                                        || p.z > bmax.z,
                                    "tri {ti} reported CLEAR of box centred {ctr:?} but contains                                      the point {p:?}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "test sampled nothing");
    }

    /// `nav.bin`'s contract: floors ascending, `MISS` only ever trailing, so every reader can stop
    /// at the first MISS. Both mutating passes re-compact, and a bug there silently truncates a
    /// cell's floor list for every consumer.
    fn assert_invariant(h: &[f32], k: usize, what: &str) {
        for (c, cell) in h.chunks(k).enumerate() {
            let mut seen_miss = false;
            let mut prev = f32::NEG_INFINITY;
            for (l, &v) in cell.iter().enumerate() {
                if v <= MISS_HALF {
                    seen_miss = true;
                } else {
                    assert!(
                        !seen_miss,
                        "{what}: cell {c} layer {l} is a floor AFTER a MISS"
                    );
                    assert!(
                        v >= prev,
                        "{what}: cell {c} layer {l} breaks ascending order"
                    );
                    prev = v;
                }
            }
        }
    }

    #[test]
    fn ledge_filter_preserves_the_height_invariant() {
        let (nx, nz, k) = (6usize, 6usize, 4usize);
        let mut h = vec![MISS; nx * nz * k];
        for c in 0..nx * nz {
            h[c * k] = 10.0; // continuous ground everywhere
        }
        // An isolated pillar top 8 m up in the middle: a ledge on every side.
        let mid = (nz / 2) * nx + nx / 2;
        h[mid * k + 1] = 18.0;
        let removed = filter_ledge_spans(&mut h, nx, nz, k);
        assert!(
            removed > 0,
            "the pillar top should have been removed as a ledge"
        );
        assert_invariant(&h, k, "after filter_ledge_spans");
        assert_eq!(
            h[mid * k],
            10.0,
            "ground must survive and compact to layer 0"
        );
        assert!(h[mid * k + 1] <= MISS_HALF, "the pillar top must be gone");
    }
}
