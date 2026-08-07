//! sh_bake — PORTABLE CPU SH irradiance-volume baker (rayon; NO CUDA, NO GPU vendor lock).
//!
//! WHY: the only lighting bake that existed (`extraction/bake/bake_volume2.py`) needs NVIDIA-Warp +
//! a CUDA GPU, so AMD / Intel / no-CUDA builds skip it and fall back to the flat realtime path (the
//! render review's "surface detail looks dogshit" on those machines). This bakes the SAME
//! world-space L1 SH radiance volume the viewer already consumes, on the CPU in parallel (rayon),
//! reusing `nav_bake`'s world-triangle assembly + BVH (shared code, one geometry path). Invoked
//! headless as `atlas bake-sh <pack_dir>`, exactly like `atlas bake-nav`.
//!
//! MILESTONES:
//!   M1 SKY-VISIBILITY — per probe, cast a Fibonacci sphere of rays; a ray that escapes the geometry
//!      sees a neutral sky gradient, an occluded ray sees nothing. Project the visible sky radiance
//!      into L1 SH -> genuine directional sky-occlusion lighting (open areas bright + directional,
//!      enclosed areas darker) on any GPU.
//!   M2 PRACTICALS (this milestone) — add the pack's live point/spot lights: physical 1/d^2 falloff
//!      with a smooth range window + spot cones, each SHADOW-TESTED against the same BVH, projected
//!      as delta lights into the SH. Fills interiors that see no sky. Mirrors bake_volume2.py's direct
//!      pass exactly (LIGHT_SCALE=6, MIN_D2=0.25), so a portable bake matches the author-side CUDA one.
//!   M3 DIFFUSE BOUNCE — a second pass re-casts a Fibonacci sphere; at each NEAREST surface hit it
//!      gathers irradiance E(hit,n) from the pass-A grid (trilinear + cosine convolution) and
//!      re-emits albedo/pi * E + emissive. Albedo/emissive = the per-material MEAN of the pack's own
//!      source PNGs (colored bounce -> a red container bounces red = Warp parity). Nearest-hit uses
//!      front-to-back BVH traversal; toggle the whole pass with EFT_SH_BOUNCE=0.
//! The OUTPUT FORMAT is already final (volume.json + volume.bin, the exact layout the viewer's
//! `load_sh_volume` reads) so each milestone just improves the numbers, never the plumbing.
//!
//! OUTPUT (byte-compatible with bake_volume2.py — see packs/*/volume.json `layout`):
//!   <pack>/volume.json  — { min, max, dims:[nx,ny,nz], spacing, coeffs:4, channels:3, layout, sun_dir, bounces }
//!   <pack>/volume.bin   — f16 LE, probe-major (idx = ((z*ny)+y)*nx + x), 12 halfs/probe =
//!                          4 L1 SH coeffs x RGB, RADIANCE SH (Y00, Y1-1=y, Y10=z, Y11=x).

use crate::eftpack::Pack;
use crate::nav_bake::{build_tris, Bvh, Tri};
use anyhow::{anyhow, Context, Result};
use glam::Vec3;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

// ---- tunables (match bake_volume2.py where it matters; env-overridable) -----------------------
const XZ_TARGET: f32 = 3.0; // target XZ probe spacing floor (m)
const Y_SPACING: f32 = 4.0; // Y probe spacing (m)
const RAY_EPS: f32 = 0.02; // ray origin push-off so a probe on a surface doesn't self-hit
const MIN_D2: f32 = 0.25; // 1/d^2 clamp within 0.5m of a bulb (probe-on-lamp); matches bake_volume2
const LIGHT_RANGE_FLOOR: f32 = 4.0; // floor a light's BAKE range (bake_volume2 `max(r,4.0)`): the probe
                                    // grid is coarse (~4-7m cells) so a sub-floor light would influence
                                    // ~no probes and vanish. Bake-only; realtime keeps the authored range.
const GRID_MAX_XY: u64 = 8192; // ny*nz cap (WebGL tex height, same as bake_volume2)
const GRID_MAX_X: usize = 4096;
const GRID_MAX_PROBES: u64 = 2_600_000;

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}

/// Which pass-A engine to use. `Auto` (default) tries the vendor-neutral wgpu-compute backend and
/// silently falls back to the rayon CPU pass if there's no usable GPU (or the map won't fit even
/// chunked); `Gpu` forces the GPU attempt (still falls back on failure, with a warning); `Cpu` skips
/// the GPU entirely. Pass B (the diffuse bounce) always runs on the CPU for now.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Auto,
    Gpu,
    Cpu,
}

/// The baked grid + SH payload, ready to serialize.
struct Baked {
    min: [f32; 3],
    max: [f32; 3],
    dims: [usize; 3],
    spacing: [f32; 3],
    sun_dir: [f32; 3],
    /// nx*ny*nz*12 f16 halfs (as u16), probe-major.
    halfs: Vec<u16>,
    /// PER-PROBE VALIDITY, probe-major, 0 = the probe is buried in geometry, 255 = fully open.
    /// Unity's Adaptive Probe Volumes call this "probe validity" and derive it from the fraction of
    /// sampling rays that hit a BACKFACE (a probe inside a wall sees the wall's inside). The viewer
    /// weights its trilinear taps by it, which is what stops a probe sealed inside a wall from
    /// bleeding its (near-black, or wrongly-lit) value onto the surfaces around it.
    validity: Vec<u8>,
    inside_solid: usize,
    /// diffuse bounces baked in (0 = direct only, 1 = one bounce).
    bounces: usize,
    /// false = INDIRECT-ONLY bake (practicals NOT baked; rendered real-time instead — direct/indirect
    /// split). true = the legacy full bake (sky + practicals + bounce all in the SH).
    direct: bool,
}

// ---- Fibonacci sphere + SH basis --------------------------------------------------------------

/// The i-th of `n` near-uniform directions on the unit sphere (spherical Fibonacci).
#[inline]
fn fib_dir(i: usize, n: usize) -> Vec3 {
    const GA: f32 = 2.399_963_2; // golden angle
    let z = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = GA * i as f32;
    Vec3::new(r * phi.cos(), z, r * phi.sin()) // Y-up
}

/// L1 real SH basis evaluated at unit direction `d`, matching the volume.json `layout`:
/// 0=Y00, 1=Y1-1(y), 2=Y10(z), 3=Y11(x).
#[inline]
fn sh_basis(d: Vec3) -> [f32; 4] {
    [0.282_095, 0.488_603 * d.y, 0.488_603 * d.z, 0.488_603 * d.x]
}

/// Neutral grayscale sky radiance seen along an escaping ray (M1). Brighter toward the zenith, a dim
/// floor at/below the horizon, no color tint (matches bake_volume2's "NEUTRAL gray sky"). Intensity
/// is `EFT_SKY` (default tuned so open ground reads roughly game-neutral; refined for parity in M2).
#[inline]
fn sky(d: Vec3, scale: f32) -> Vec3 {
    let up = d.y.clamp(-1.0, 1.0);
    let g = (0.35 + 0.75 * up.max(0.0)) * scale; // horizon ~0.35*scale, zenith ~1.1*scale
    Vec3::splat(g)
}

// ---- 3D ray vs the (nav) BVH: any-hit occlusion test ------------------------------------------

/// True if the ray `o + t*d` (t in (eps, t_max)) hits ANY triangle in the BVH. For a SKY ray pass
/// `t_max = f32::INFINITY` (does the ray escape to the sky?); for a SHADOW ray to a bulb at distance
/// `dist` pass `t_max = dist - 0.1` (is there an occluder BETWEEN probe and bulb?). Slab-prunes each
/// node's AABB, Möller–Trumbore at leaves, early-out on first hit. Reuses the SAME `Bvh` (nodes +
/// tris) `nav_bake` builds — the node AABBs are full 3-D, so the X/Z-split tree is a valid (if not
/// Y-optimal) accelerator for arbitrary rays.
fn ray_occluded(bvh: &Bvh, o: Vec3, d: Vec3, t_max: f32, stack: &mut Vec<u32>) -> bool {
    if bvh.tris.is_empty() {
        return false;
    }
    let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    stack.clear();
    stack.push(0);
    while let Some(ni) = stack.pop() {
        let node = bvh.nodes[ni as usize];
        // Ray-AABB slab test (t in (RAY_EPS, inf)).
        let t0 = (node.min - o) * inv;
        let t1 = (node.max - o) * inv;
        let tmin = t0.min(t1);
        let tmax = t0.max(t1);
        let enter = tmin.x.max(tmin.y).max(tmin.z).max(RAY_EPS);
        let exit = tmax.x.min(tmax.y).min(tmax.z);
        if enter > exit || enter > t_max {
            continue; // ray misses the box, or the box starts beyond the segment end
        }
        if node.count > 0 {
            let s = node.start as usize;
            for t in &bvh.tris[s..s + node.count as usize] {
                if ray_tri(o, d, t, t_max) {
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

/// Möller–Trumbore: the ray `o + t*d` crosses triangle `t` at some t in (RAY_EPS, t_max) -> Some(t).
#[inline]
fn ray_tri_t(o: Vec3, d: Vec3, t: &Tri, t_max: f32) -> Option<f32> {
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1.0e-8 {
        return None; // parallel
    }
    let inv = 1.0 / det;
    let tv = o - t.a;
    let u = tv.dot(p) * inv;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = tv.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let tt = e2.dot(q) * inv;
    if tt > RAY_EPS && tt < t_max {
        Some(tt)
    } else {
        None
    }
}

/// Any-hit boolean form (the occlusion / shadow test).
#[inline]
fn ray_tri(o: Vec3, d: Vec3, t: &Tri, t_max: f32) -> bool {
    ray_tri_t(o, d, t, t_max).is_some()
}

/// NEAREST hit of the ray `o + t*d` (t in (RAY_EPS, t_max)) against the BVH -> (t, tri_index into
/// `bvh.tris`). Used by the M3 diffuse-bounce pass, which needs the closest surface (to read its
/// normal + gather irradiance there). Unlike `ray_occluded` there is NO early-out; instead the
/// current nearest `best_t` prunes any box whose entry is already farther, so it stays fast.
fn ray_hit(bvh: &Bvh, o: Vec3, d: Vec3, t_max: f32, stack: &mut Vec<u32>) -> Option<(f32, usize)> {
    if bvh.tris.is_empty() {
        return None;
    }
    let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    let mut best_t = t_max;
    let mut best_face: Option<usize> = None;
    stack.clear();
    stack.push(0);
    while let Some(ni) = stack.pop() {
        let node = bvh.nodes[ni as usize];
        let t0 = (node.min - o) * inv;
        let t1 = (node.max - o) * inv;
        let tmin = t0.min(t1);
        let tmax = t0.max(t1);
        let enter = tmin.x.max(tmin.y).max(tmin.z).max(RAY_EPS);
        let exit = tmax.x.min(tmax.y).min(tmax.z);
        if enter > exit || enter > best_t {
            continue; // miss, or the whole box is farther than the nearest hit so far
        }
        if node.count > 0 {
            let s = node.start as usize;
            for fi in s..s + node.count as usize {
                if let Some(t) = ray_tri_t(o, d, &bvh.tris[fi], best_t) {
                    if t < best_t {
                        best_t = t;
                        best_face = Some(fi);
                    }
                }
            }
        } else {
            // front-to-back: process the child the ray ENTERS sooner first, so `best_t` tightens fast
            // and the farther subtree is pruned by the `enter > best_t` test above. Ordering only
            // affects speed, never which hit is nearest.
            let (a, b) = (node.start, node.start + 1);
            let na = bvh.nodes[a as usize];
            let nb = bvh.nodes[b as usize];
            let ea = ((na.min - o) * inv).min((na.max - o) * inv).max_element();
            let eb = ((nb.min - o) * inv).min((nb.max - o) * inv).max_element();
            if ea <= eb {
                stack.push(b); // far pushed first -> near (a) popped first
                stack.push(a);
            } else {
                stack.push(a);
                stack.push(b);
            }
        }
    }
    best_face.map(|f| (best_t, f))
}

/// Fraction of `dirs` that hit a BACKFACE from `o` — Unity APV's probe-validity metric.
///
/// A probe floating in open space sees front faces (or sky); a probe buried inside a wall or under
/// terrain sees that geometry's INSIDE, i.e. triangles whose geometric normal points the same way
/// as the ray. Unity marks a probe invalid past ~25% backfaces and then weights interpolation by
/// validity ("Validity and Normal Based" leak reduction). We store the continuous ratio and let the
/// shader weight with it, which degrades more gracefully than a hard flag.
fn backface_ratio(bvh: &Bvh, o: Vec3, dirs: &[(Vec3, [f32; 4])], stride: usize, stack: &mut Vec<u32>) -> f32 {
    let mut hit = 0u32;
    let mut back = 0u32;
    let mut i = 0;
    while i < dirs.len() {
        let d = dirs[i].0;
        if let Some((_, fi)) = ray_hit(bvh, o, d, f32::INFINITY, stack) {
            hit += 1;
            let t = &bvh.tris[fi];
            // geometric normal from the winding; a hit is a BACKFACE when it faces along the ray
            let n = (t.b - t.a).cross(t.c - t.a);
            if n.dot(d) > 0.0 {
                back += 1;
            }
        }
        i += stride;
    }
    if hit == 0 { 0.0 } else { back as f32 / hit as f32 }
}

/// VIRTUAL OFFSET (Unity Adaptive Probe Volumes): find a lighting position for a probe that is
/// buried in geometry.
///
/// Rejecting an in-wall probe at runtime only turns it into a HOLE, and the hole is its own
/// artefact: measured on interchange, a window sat 98% of the way into its cell, so the probe
/// carrying nearly all the trilinear weight was the invalid one and the cell collapsed to zero GI
/// — a hard, probe-grid-aligned rectangle painted across the glass. Runtime offsetting cannot
/// rescue it either, because it can only step along the shaded surface's normal while the lighting
/// cliff may run along a different axis entirely (that window's normal is +X; its cliff runs in Z).
///
/// Unity's fix is to MOVE the probe out of the collider before lighting it, so the probe holds real
/// light instead of being excluded. Same idea here: search outward for the nearest position whose
/// backface ratio says "open space", and bake there. Returns the probe's own position when it is
/// already valid or when nothing valid is within reach (deep inside rock — correctly still dark).
fn virtual_offset(bvh: &Bvh, o: Vec3, spacing: [f32; 3], dirs: &[(Vec3, [f32; 4])],
                  stride: usize, stack: &mut Vec<u32>) -> Vec3 {
    // 26 neighbour directions (the 3x3x3 cube minus the centre), tried nearest-first so a probe
    // moves the SHORTEST distance that reaches open space — Unity likewise caps the push and
    // prefers the smallest one, because a probe dragged far away stops representing its cell.
    // Candidate directions: the 6 face axes first (a probe buried in a wall almost always escapes
    // straight out of one), then the 8 cube diagonals for corners. Nearest distance first, so a
    // probe moves the SHORTEST way into open space — Unity likewise caps and minimises the push,
    // because a probe dragged far stops representing its own cell.
    const DIRS: [[f32; 3]; 14] = [
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
    ];
    let step = spacing[0].min(spacing[1]).min(spacing[2]);
    for frac in [0.5f32, 1.0] {
        let mut best: Option<(f32, Vec3)> = None;
        for d in DIRS.iter() {
            let dir = Vec3::new(d[0], d[1], d[2]).normalize();
            let c = o + dir * (step * frac);
            let r = backface_ratio(bvh, c, dirs, stride, stack);
            if r < 0.25 && best.map_or(true, |(br, _)| r < br) {
                best = Some((r, c));
            }
        }
        if let Some((_, c)) = best {
            return c;
        }
    }
    o
}

// ---- grid + bake ------------------------------------------------------------------------------

/// `q`-th percentile (0..100) of a mutable slice via partial select.
fn percentile(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let idx = (((q / 100.0) * (v.len() - 1) as f32).round() as usize).min(v.len() - 1);
    v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    v[idx]
}

/// World-space origin of probe `pi` in the probe-major grid (idx = ((z*ny)+y)*nx + x).
#[inline]
fn probe_o(pi: usize, nx: usize, ny: usize, gmin: Vec3, spacing: [f32; 3]) -> Vec3 {
    let x = pi % nx;
    let y = (pi / nx) % ny;
    let z = pi / (nx * ny);
    Vec3::new(
        gmin.x + x as f32 * spacing[0],
        gmin.y + y as f32 * spacing[1],
        gmin.z + z as f32 * spacing[2],
    )
}

/// Trilinear irradiance E(p, n) from the direct radiance-SH grid `sh` (pass-A f32 coeffs), via cosine
/// convolution (A0=pi, A1=2pi/3) — the SAME reconstruction the viewer's `load_sh_volume` and
/// bake_volume2.py's `irr_at` use. `inv_sp` = 1/spacing. Clamped to >= 0 per channel.
#[inline]
fn irr_at(sh: &[[Vec3; 4]], gmin: Vec3, inv_sp: Vec3, dims: [usize; 3], p: Vec3, n: Vec3) -> Vec3 {
    let [nx, ny, nz] = dims;
    let gx = ((p.x - gmin.x) * inv_sp.x).clamp(0.0, (nx - 1) as f32);
    let gy = ((p.y - gmin.y) * inv_sp.y).clamp(0.0, (ny - 1) as f32);
    let gz = ((p.z - gmin.z) * inv_sp.z).clamp(0.0, (nz - 1) as f32);
    let (x0, y0, z0) = (gx.floor() as usize, gy.floor() as usize, gz.floor() as usize);
    let (x1, y1, z1) = ((x0 + 1).min(nx - 1), (y0 + 1).min(ny - 1), (z0 + 1).min(nz - 1));
    let (fx, fy, fz) = (gx - x0 as f32, gy - y0 as f32, gz - z0 as f32);
    let mut a = [Vec3::ZERO; 4];
    for k in 0..8 {
        let (xi, wx) = if k & 1 != 0 { (x1, fx) } else { (x0, 1.0 - fx) };
        let (yi, wy) = if k & 2 != 0 { (y1, fy) } else { (y0, 1.0 - fy) };
        let (zi, wz) = if k & 4 != 0 { (z1, fz) } else { (z0, 1.0 - fz) };
        let w = wx * wy * wz;
        let s = &sh[(zi * ny + yi) * nx + xi];
        for c in 0..4 {
            a[c] += s[c] * w;
        }
    }
    // E(n) = pi*Y00*c0 + (2pi/3)*Y1*(c1*n.y + c2*n.z + c3*n.x)
    let e = a[0] * 0.886_226_9 + (a[1] * n.y + a[2] * n.z + a[3] * n.x) * 1.023_326_7;
    Vec3::new(e.x.max(0.0), e.y.max(0.0), e.z.max(0.0))
}

// ---- M3b: per-material albedo/emissive LUTs (mean of the pack's own source PNGs) --------------

/// sRGB(u8) -> linear(f32) LUT (gamma 2.2, matching bake_volume2.py's `_srgb_lut`).
fn srgb_lin_lut() -> [f32; 256] {
    let mut l = [0f32; 256];
    let mut i = 0;
    while i < 256 {
        l[i] = ((i as f32) / 255.0).powf(2.2);
        i += 1;
    }
    l
}

/// Mean LINEAR rgb of a texture PNG (whole image), or None if it can't be decoded.
fn tex_mean_linear(path: &Path, lut: &[f32; 256]) -> Option<Vec3> {
    let img = image::open(path).ok()?.to_rgb8();
    let raw = img.as_raw();
    if raw.len() < 3 {
        return None;
    }
    let (mut sr, mut sg, mut sb) = (0f64, 0f64, 0f64);
    for px in raw.chunks_exact(3) {
        sr += lut[px[0] as usize] as f64;
        sg += lut[px[1] as usize] as f64;
        sb += lut[px[2] as usize] as f64;
    }
    let n = (raw.len() / 3) as f64;
    Some(Vec3::new((sr / n) as f32, (sg / n) as f32, (sb / n) as f32))
}

/// Per-material albedo + emissive radiance LUTs (indexed by material id) for the diffuse bounce.
/// albedo = mean(source albedo PNG, LINEAR) * tint  (or tint*0.5 when untextured — bake_volume2's
/// untextured fallback); emissive = factor*hdr * mean(emissive PNG) (or *1 if no map), clamped to a
/// sane ceiling. Every UNIQUE texture is decoded ONCE, in parallel (the pack ships its own tex/).
fn build_material_luts(pack: &Pack) -> (Vec<Vec3>, Vec<Vec3>) {
    let lut = srgb_lin_lut();
    let max_id = pack.materials.iter().map(|m| m.id).max().unwrap_or(0) as usize;
    let mut albedo = vec![Vec3::splat(0.3); max_id + 1]; // neutral grey for id gaps
    let mut emissive = vec![Vec3::ZERO; max_id + 1];

    // unique texture paths (albedo + emissive maps) referenced by any material
    let mut want: std::collections::HashSet<String> = std::collections::HashSet::new();
    for m in &pack.materials {
        if let Some(a) = &m.albedo {
            want.insert(a.clone());
        }
        if let Some(e) = &m.emissive {
            if let Some(t) = &e.texture {
                want.insert(t.clone());
            }
        }
    }
    let paths: Vec<String> = want.into_iter().collect();
    let t_tex = Instant::now();
    let means: std::collections::HashMap<String, Vec3> = paths
        .par_iter()
        .filter_map(|rel| tex_mean_linear(&pack.root.join(rel), &lut).map(|v| (rel.clone(), v)))
        .collect();
    eprintln!(
        "  sh-bake: decoded {}/{} unique bounce textures (mean albedo) in {:.2}s",
        means.len(),
        paths.len(),
        t_tex.elapsed().as_secs_f32()
    );

    for m in &pack.materials {
        let i = m.id as usize;
        let tint = Vec3::new(m.tint[0], m.tint[1], m.tint[2]);
        albedo[i] = match m.albedo.as_ref().and_then(|a| means.get(a)) {
            Some(mean) => *mean * tint,
            None => tint * 0.5,
        }
        // energy bound: a diffuse albedo is physically <= 1 (reference clamps too). No-op on LDR
        // base-color packs, but guards against a future HDR tint over-brightening the bounce.
        .clamp(Vec3::ZERO, Vec3::ONE);
        if let Some(e) = &m.emissive {
            let f = Vec3::new(e.factor[0], e.factor[1], e.factor[2]) * e.hdr;
            let cov = e.texture.as_ref().and_then(|t| means.get(t)).copied().unwrap_or(Vec3::ONE);
            emissive[i] = (f * cov).min(Vec3::splat(8.0)); // clamp HDR emissive (bake_volume2 EMIS_MAX)
        }
    }
    (albedo, emissive)
}

fn bake(pack: &Pack, backend: Backend) -> Result<Baked> {
    let n_dir = env_usize("EFT_SH_RAYS", 256).max(8);
    let sky_scale = env_f32("EFT_SKY", 2.0);
    let light_scale = env_f32("EFT_LIGHT_SCALE", 6.0); // Unity color*intensity -> SH radiance (bake_volume2 default)

    // --- geometry: reuse nav_bake's world-triangle assembly (floors+ceilings+walls = the occluder) ---
    let t_geo = Instant::now();
    let (mut column_tris, wall_tris, min_y, max_y, _door) = build_tris(pack);
    let n_col = column_tris.len();
    column_tris.extend_from_slice(&wall_tris); // full occluder soup
    let n_tris = column_tris.len();
    if n_tris == 0 {
        return Err(anyhow!("no triangles in pack (nothing to bake)"));
    }
    eprintln!(
        "  sh-bake: {n_tris} occluder tris ({n_col} column + {} wall), y in [{min_y:.1},{max_y:.1}] in {:.2}s",
        wall_tris.len(),
        t_geo.elapsed().as_secs_f32()
    );

    // --- probe grid: 1/99-pct XZ core + data-driven Y band, ~XZ_TARGET spacing, auto-widened ---
    let step = (n_tris / 500_000).max(1);
    let mut xs: Vec<f32> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    let mut zs: Vec<f32> = Vec::new();
    for t in column_tris.iter().step_by(step) {
        xs.push(t.a.x);
        ys.push(t.a.y);
        zs.push(t.a.z);
    }
    let (lo_x, hi_x) = (percentile(&mut xs.clone(), 1.0), percentile(&mut xs, 99.0));
    let (lo_z, hi_z) = (percentile(&mut zs.clone(), 1.0), percentile(&mut zs, 99.0));
    let ylo = percentile(&mut ys.clone(), 0.5) - 2.0;
    let yhi = percentile(&mut ys, 99.7) + 4.0;
    let gmin = Vec3::new(lo_x, ylo, lo_z);
    let gmax = Vec3::new(hi_x.max(lo_x + 1.0), yhi.max(ylo + 1.0), hi_z.max(lo_z + 1.0));
    let ext = gmax - gmin;
    let mut sxz = XZ_TARGET.max(((ext.x * ext.z) / 120_000.0).sqrt());
    let (mut nx, mut ny, mut nz);
    let mut guard = 0;
    loop {
        nx = ((ext.x / sxz).ceil() as usize + 1).max(2);
        ny = ((ext.y / Y_SPACING).ceil() as usize + 1).max(2);
        nz = ((ext.z / sxz).ceil() as usize + 1).max(2);
        let ok = (ny as u64 * nz as u64) <= GRID_MAX_XY
            && nx <= GRID_MAX_X
            && (nx as u64 * ny as u64 * nz as u64) <= GRID_MAX_PROBES;
        guard += 1;
        if ok || guard > 64 {
            break;
        }
        sxz *= 1.15;
    }
    let dims = [nx, ny, nz];
    let spacing = [
        ext.x / (nx.max(2) - 1) as f32,
        ext.y / (ny.max(2) - 1) as f32,
        ext.z / (nz.max(2) - 1) as f32,
    ];
    let n_probe = nx * ny * nz;
    eprintln!(
        "  sh-bake: grid {nx} x {ny} x {nz} = {n_probe} probes @ [{:.2},{:.2},{:.2}]m, {n_dir} rays/probe",
        spacing[0], spacing[1], spacing[2]
    );

    // --- BVH (shared with nav_bake) ---
    let t_bvh = Instant::now();
    let bvh = Bvh::build(column_tris);
    eprintln!("  sh-bake: BVH {} nodes in {:.2}s", bvh.nodes.len(), t_bvh.elapsed().as_secs_f32());

    // --- sun_dir: reuse the pack's brightest directional-ish light if any, else a neutral default ---
    let sun_dir = pack_sun_dir(pack);

    // --- M2: live practical lights (already viewer-world, color = linear*intensity, spot cones baked) ---
    let lights = &pack.lights;
    eprintln!(
        "  sh-bake: {} live practical light(s), shadow-tested @ scale {light_scale}",
        lights.len()
    );

    // Precompute the sky-ray directions + their SH bases once (shared across all probes).
    let dirs: Vec<(Vec3, [f32; 4])> = (0..n_dir).map(|i| { let d = fib_dir(i, n_dir); (d, sh_basis(d)) }).collect();
    let norm = 4.0 * std::f32::consts::PI / n_dir as f32;

    // Direct/indirect split: with EFT_SH_INDIRECT=1 (`bake-sh --indirect-only`) the M2 practicals are
    // NOT baked into the SH — they render as real-time direct lights instead. The volume then carries
    // only INDIRECT lighting (sky M1 + bounce M3) and volume.json is marked "direct": false, which the
    // viewer reads to auto-enable the realtime light grid (crisp per-light interiors, no double-count).
    let indirect_only = std::env::var("EFT_SH_INDIRECT").map(|v| v.trim() == "1").unwrap_or(false);
    if indirect_only {
        eprintln!("  sh-bake: INDIRECT-ONLY — practicals skipped (baked = sky + bounce; direct=false)");
    }

    // ================= PROBE VALIDITY + VIRTUAL OFFSET (Unity APV) =================================
    // Both must run BEFORE pass A: virtual offset changes WHERE a probe is lit from.
    let t_v = Instant::now();
    let v_stride = (n_dir / 64).max(1);           // ~64 rays: "am I inside a solid" is low-frequency
    let mut validity: Vec<u8> = vec![255; n_probe];
    validity.par_iter_mut().enumerate().for_each_init(
        || Vec::<u32>::with_capacity(64),
        |stack, (pi, out)| {
            let o = probe_o(pi, nx, ny, gmin, spacing);
            let r = backface_ratio(&bvh, o, &dirs, v_stride, stack);
            // Unity treats >=25% backfaces as invalid; ramp to zero over that range so a probe near
            // a wall fades out of the interpolation rather than popping.
            *out = (((1.0 - (r / 0.25)).clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
        },
    );
    let n_invalid = validity.iter().filter(|&&v| v < 128).count();
    eprintln!(
        "  sh-bake: validity {n_probe} probes in {:.2}s ({} invalid / in-geometry, {:.1}%)",
        t_v.elapsed().as_secs_f32(), n_invalid, 100.0 * n_invalid as f32 / n_probe.max(1) as f32
    );

    // VIRTUAL OFFSET: light an invalid probe from open space instead of leaving a hole. Restricted
    // to probes that BORDER valid space — a probe deep inside rock has nowhere useful to go and is
    // correctly dark, and skipping those keeps the search affordable (the useful set is a small
    // fraction of the invalid ones). EFT_SH_VO=0 disables.
    let vo_on = std::env::var("EFT_SH_VO").map(|v| v.trim() != "0").unwrap_or(true);
    let mut positions: Vec<Vec3> = Vec::new();
    let mut n_moved = 0usize;
    if vo_on && n_invalid > 0 {
        let t_vo = Instant::now();
        let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
        let borders_valid = |pi: usize| -> bool {
            let x = pi % nx;
            let y = (pi / nx) % ny;
            let z = pi / (nx * ny);
            let mut n = 0u8;
            for (dx, dy, dz) in [(1i32, 0i32, 0i32), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                let (a, b, c) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                if a < 0 || b < 0 || c < 0 || a >= nx as i32 || b >= ny as i32 || c >= nz as i32 {
                    continue;
                }
                if validity[idx(a as usize, b as usize, c as usize)] >= 128 {
                    n += 1;
                }
            }
            n > 0
        };
        positions = (0..n_probe).map(|pi| probe_o(pi, nx, ny, gmin, spacing)).collect();
        let moved: Vec<bool> = positions
            .par_iter_mut()
            .enumerate()
            .map_init(
                || Vec::<u32>::with_capacity(64),
                |stack, (pi, pos)| {
                    if validity[pi] >= 128 || !borders_valid(pi) {
                        return false;
                    }
                    let c = virtual_offset(&bvh, *pos, spacing, &dirs, v_stride * 4, stack);
                    if c != *pos {
                        *pos = c;
                        return true;
                    }
                    false
                },
            )
            .collect();
        n_moved = moved.iter().filter(|&&m| m).count();
        eprintln!(
            "  sh-bake: virtual offset {} probes relocated out of geometry in {:.2}s",
            n_moved, t_vo.elapsed().as_secs_f32()
        );
        // A relocated probe is lit from open space, so it is no longer a hole: mark it valid.
        for (pi, m) in moved.iter().enumerate() {
            if *m {
                validity[pi] = 255;
            }
        }
    }
    let probe_pos = |pi: usize| -> Vec3 {
        if positions.is_empty() { probe_o(pi, nx, ny, gmin, spacing) } else { positions[pi] }
    };

    // ================= PASS A — direct: sky-visibility (M1) + shadow-tested practicals (M2) =========
    // GPU (vendor-neutral wgpu compute; the tri/node BVH is chunked across storage bindings to keep
    // even interchange/streets-scale giants on-GPU) when available + it fits; else the rayon CPU pass
    // below. Kept in f32 (NOT packed yet) so the M3 diffuse bounce can trilinearly gather irradiance.
    let t_bake = Instant::now();
    // PER-PROBE pass A, shared by the full CPU path and the relocated-probe fixup below.
    // Returns (sh, saw_no_sky) — the second is the inside-solid diagnostic.
    let pass_a_probe = |pi: usize, stack: &mut Vec<u32>| -> ([Vec3; 4], bool) {
        let o = probe_pos(pi);
        // --- M1: sky-visibility (each escaping Fibonacci ray sees a neutral sky gradient) ---
        let mut sh = [Vec3::ZERO; 4];
        let mut n_sky = 0u32;
        for (d, basis) in &dirs {
            if ray_occluded(&bvh, o, *d, f32::INFINITY, stack) {
                continue; // occluded -> sees no sky
            }
            n_sky += 1;
            let l = sky(*d, sky_scale);
            for c in 0..4 {
                sh[c] += l * basis[c];
            }
        }
        for c in 0..4 {
            sh[c] *= norm; // sky is a hemisphere integral -> weight by the ray solid angle dw
        }
        // --- M2: live practicals (delta lights; added AFTER the sky*norm scale, NOT weighted by
        //     norm — a point light is a single direction, not a solid-angle sample). SKIPPED under
        //     --indirect-only (they render as real-time direct lights instead). ---
        if !indirect_only {
            for lgt in lights {
                // Switch-controlled lights are OFF in the runtime's default power state (the
                // renderer builds its records with all groups off, mask 0), so baking them lit
                // burns a permanently-powered mall into the volume and makes the lever inert:
                // a volume with "direct": true disables the realtime grid entirely. The extractor
                // was changed to keep these records, tagged `on:false` with a `group`, precisely so
                // the viewer could switch them; neither SH baker learned. Interchange ships 109
                // such records, all intensity > 0, all inside the bake volume. Skip the whole
                // group, not just the on:false ones, because the default state zeroes all of it.
                if lgt.group_idx >= 0 {
                    continue;
                }
                let tol = lgt.pos - o;
                let dist = tol.length();
                let r = lgt.range.max(LIGHT_RANGE_FLOOR); // floor to match the CUDA reference
                if dist <= 0.05 || dist >= r {
                    continue; // on the bulb, or out of range
                }
                let dl = tol / dist;
                let spot = if lgt.cos_outer > -1.5 {
                    // spot cone smoothstep; point lights (sentinel cos_outer=-2.0) stay factor 1
                    let cosang = -dl.dot(lgt.dir);
                    ((cosang - lgt.cos_outer) / (lgt.cos_inner - lgt.cos_outer + 1.0e-4)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                if spot <= 0.0 {
                    continue; // outside the cone
                }
                let x = dist / r;
                let win = (1.0 - x * x * x * x).clamp(0.0, 1.0);
                let at = win * win / (dist * dist).max(MIN_D2);
                if ray_occluded(&bvh, o, dl, dist - 0.1, stack) {
                    continue; // occluder between probe and bulb -> shadowed
                }
                let rad = lgt.color * (at * spot * light_scale);
                let basis = sh_basis(dl);
                for c in 0..4 {
                    sh[c] += rad * basis[c];
                }
            }
        }
        (sh, n_sky == 0)
    };

    // The `n_moved == 0` gate that used to sit here is gone, because both of the requirements
    // its post-mortem set for removing it are now met. (1) The GPU pass reads TRUE probe
    // positions from an uploaded buffer instead of deriving them from (gmin, spacing, index), so
    // virtual offset is honoured on the GPU itself and the CPU fixup that used to re-light the
    // relocated probes has nothing left to fix. (2) Device loss is survivable at every layer: the
    // non-panicking uncaptured-error handler keeps wgpu's poll from aborting the process, the
    // opening batch is scaled down by BVH depth so a giant scene cannot blow the ~2 s watchdog
    // before the adaptive sizing has a measurement, the all-zero net catches a reset that still
    // returns a buffer, and if the process dies anyway the BUILD retries the stage with
    // EFT_BAKE_CPU=1 (tools/build_map.py) -- the stale-volume freshness check from the original
    // post-mortem is what arms that retry.
    let gpu_a = if matches!(backend, Backend::Gpu | Backend::Auto) {
        crate::sh_bake_gpu::pass_a_gpu(
            &bvh, lights, &positions, gmin, spacing, dims, n_dir, sky_scale, light_scale,
            indirect_only,
        )
    } else {
        None
    };
    let used_gpu = gpu_a.is_some();
    let (sh_a, inside_solid): (Vec<[Vec3; 4]>, usize) = if let Some(v) = gpu_a {
        (v, 0) // inside-solid is a CPU-pass diagnostic only (never serialized); GPU path skips the count
    } else {
        if backend == Backend::Gpu {
            eprintln!("  sh-bake: --backend gpu unavailable — falling back to the CPU pass A");
        }
        let inside = std::sync::atomic::AtomicUsize::new(0);
        let mut sh_a: Vec<[Vec3; 4]> = vec![[Vec3::ZERO; 4]; n_probe];
        sh_a.par_iter_mut().enumerate().for_each_init(
            || Vec::<u32>::with_capacity(64),
            |stack, (pi, out)| {
                let (sh, no_sky) = pass_a_probe(pi, stack);
                if no_sky {
                    inside.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                *out = sh;
            },
        );
        (sh_a, inside.into_inner())
    };
    eprintln!(
        "  sh-bake: pass A (direct) {n_probe} probes in {:.2}s ({} fully-occluded/inside-solid){}",
        t_bake.elapsed().as_secs_f32(),
        inside_solid,
        if used_gpu { " [GPU]" } else { " [CPU]" }
    );

    // ================= PASS B — one diffuse bounce (M3) =============================================
    // Re-cast a Fibonacci sphere; at each NEAREST surface hit, gather irradiance E(hit,n) from the
    // pass-A grid (trilinear + cosine convolution) and re-emit albedo/pi * E + emissive back into the
    // SH. Albedo/emissive = the per-material mean of the pack's own source PNGs (colored bounce = Warp
    // parity). Toggle with EFT_SH_BOUNCE=0 (then the pack is byte-identical to the M2 direct bake).
    // The bounce is low-frequency, so half the sky-ray count is plenty and ~halves the (costly)
    // nearest-hit pass. Override with EFT_SH_BOUNCE_RAYS.
    let bounce_rays = env_usize("EFT_SH_BOUNCE_RAYS", 128).clamp(1, 4096);
    let albedo_boost = env_f32("EFT_SH_ALBEDO_BOOST", 1.0); // global multiplier on per-material albedo
    let emis_gain = env_f32("EFT_SH_EMIS_GAIN", 1.0); // global multiplier on emissive-surface gather
    let do_bounce = env_usize("EFT_SH_BOUNCE", 1) > 0 && !bvh.tris.is_empty();
    let mut halfs = vec![0u16; n_probe * 12];
    let bounces = if do_bounce {
        let t_b = Instant::now();
        // per-material albedo/emissive (mean of the pack's source PNGs) — colored bounce = Warp parity
        let (albedo_lut, emis_lut) = build_material_luts(pack);
        let inv_pi_boost = std::f32::consts::FRAC_1_PI * albedo_boost; // albedo/pi * boost
        // GPU bounce (nearest-hit re-cast + trilinear gather + per-material re-emit, combined with pass
        // A) — EXPERIMENTAL, opt-in via EFT_SH_GPU_BOUNCE=1. The nearest-hit bounce is far costlier and
        // more per-probe-variable than pass A's any-hit occlusion, so a hot batch can exceed the OS GPU
        // watchdog (~2 s) and reset the device; TDR-safe batches for it would be so small that giant-map
        // GPU pass B barely beats the rayon pass anyway. Default is the reliable CPU bounce below.
        // The n_moved gate is gone here for the same reason as pass A's: the shader now reads
        // every probe's TRUE position from the uploaded buffer, so a relocated probe casts its
        // bounce rays from where it actually sits. That mattered doubly for this pass -- the
        // shipped bakes are --indirect-only, where the bounce IS essentially the whole signal,
        // and 25.7% of interchange's probes are relocated.
        let gpu_b = if matches!(backend, Backend::Gpu | Backend::Auto)
            && env_usize("EFT_SH_GPU_BOUNCE", 0) > 0
        {
            crate::sh_bake_gpu::pass_b_gpu(
                &bvh, &sh_a, &positions, gmin, spacing, [nx, ny, nz], bounce_rays,
                &albedo_lut, &emis_lut, inv_pi_boost, emis_gain,
            )
        } else {
            None
        };
        if let Some(combined) = gpu_b {
            // combined = pass A + bounce*bnorm (done on GPU); just pack to f16
            halfs.par_chunks_mut(12).zip(combined.par_iter()).for_each(|(out, c)| {
                for k in 0..4 {
                    out[k * 3] = f16_bits(c[k].x);
                    out[k * 3 + 1] = f16_bits(c[k].y);
                    out[k * 3 + 2] = f16_bits(c[k].z);
                }
            });
            eprintln!(
                "  sh-bake: pass B (1 diffuse bounce, per-material colored albedo, {bounce_rays} rays) [GPU] in {:.2}s",
                t_b.elapsed().as_secs_f32()
            );
        } else {
        let bdirs: Vec<(Vec3, [f32; 4])> =
            (0..bounce_rays).map(|i| { let d = fib_dir(i, bounce_rays); (d, sh_basis(d)) }).collect();
        let bnorm = 4.0 * std::f32::consts::PI / bounce_rays as f32;
        let inv_sp = Vec3::new(1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]);
        // full mesh diagonal (BVH root AABB) — a bounce ray may hit geometry well outside the probe band
        let root = bvh.nodes[0];
        let max_dist = (root.max - root.min).length() * 1.2;
        halfs.par_chunks_mut(12).enumerate().for_each_init(
            || Vec::<u32>::with_capacity(64),
            |stack, (pi, out)| {
                let o = probe_pos(pi);
                let mut sh_b = [Vec3::ZERO; 4];
                for (d, basis) in &bdirs {
                    if let Some((t, face)) = ray_hit(&bvh, o, *d, max_dist, stack) {
                        let tri = &bvh.tris[face];
                        let mut n = (tri.b - tri.a).cross(tri.c - tri.a).normalize_or_zero();
                        if n.dot(*d) > 0.0 {
                            n = -n; // orient the surface toward the incoming ray
                        }
                        let h = o + *d * t + n * 0.05; // hit point, nudged off the surface
                        let e = irr_at(&sh_a, gmin, inv_sp, [nx, ny, nz], h, n);
                        let mat = tri.mat as usize;
                        let alb = albedo_lut.get(mat).copied().unwrap_or(Vec3::splat(0.3));
                        let emi = emis_lut.get(mat).copied().unwrap_or(Vec3::ZERO);
                        let rad = e * alb * inv_pi_boost + emi * emis_gain; // albedo/pi * E + emissive
                        for c in 0..4 {
                            sh_b[c] += rad * basis[c];
                        }
                    }
                }
                let a = &sh_a[pi];
                for c in 0..4 {
                    let v = a[c] + sh_b[c] * bnorm; // direct + (bounce hemisphere integral)
                    out[c * 3] = f16_bits(v.x);
                    out[c * 3 + 1] = f16_bits(v.y);
                    out[c * 3 + 2] = f16_bits(v.z);
                }
            },
        );
        eprintln!(
            "  sh-bake: pass B (1 diffuse bounce, per-material colored albedo, {bounce_rays} rays) [CPU] in {:.2}s",
            t_b.elapsed().as_secs_f32()
        );
        }
        1
    } else {
        // bounce off -> pack pass A directly (identical to the M2 direct bake)
        halfs.par_chunks_mut(12).enumerate().for_each(|(pi, out)| {
            let a = &sh_a[pi];
            for c in 0..4 {
                out[c * 3] = f16_bits(a[c].x);
                out[c * 3 + 1] = f16_bits(a[c].y);
                out[c * 3 + 2] = f16_bits(a[c].z);
            }
        });
        0
    };

    Ok(Baked {
        min: [gmin.x, gmin.y, gmin.z],
        max: [gmax.x, gmax.y, gmax.z],
        dims,
        spacing,
        sun_dir,
        halfs,
        validity,
        inside_solid,
        bounces,
        direct: !indirect_only,
    })
}

/// Sun direction for the viewer's sky-reflection term. The pack's practical lights are point/spot
/// (no directional sun in the set), so M1 uses a fixed neutral default matching interchange's baked
/// `sun_dir`; a real per-map sun can be threaded from the extraction sidecar in a later milestone.
fn pack_sun_dir(_pack: &Pack) -> [f32; 3] {
    [0.449, 0.799, -0.400]
}

// ---- f16 (IEEE half) pack — no new dep --------------------------------------------------------

/// f32 -> IEEE-754 half (round-to-nearest-even-ish; good enough for radiance SH), returned as bits.
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if ((bits >> 23) & 0xff) == 0xff {
        // inf/nan
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow -> 0
        }
        let m = (mant | 0x80_0000) >> (14 - exp) as u32;
        return sign | (m as u16);
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

// ---- write volume.json + volume.bin -----------------------------------------------------------

fn write_volume(b: &Baked, dir: &Path) -> Result<()> {
    let bin: &[u8] = bytemuck::cast_slice(&b.halfs);
    std::fs::write(dir.join("volume.bin"), bin)
        .with_context(|| format!("writing {}", dir.join("volume.bin").display()))?;
    // Validity is a SEPARATE file, not appended to volume.bin: an older viewer keeps loading the
    // volume byte-for-byte as before and simply renders without leak reduction.
    std::fs::write(dir.join("volume_valid.bin"), &b.validity)
        .with_context(|| format!("writing {}", dir.join("volume_valid.bin").display()))?;
    let meta = serde_json::json!({
        "min": b.min, "max": b.max,
        "dims": [b.dims[0], b.dims[1], b.dims[2]],
        "spacing": b.spacing,
        "coeffs": 4, "channels": 3,
        "layout": "float16 LE, probe-major; probe index = ((z*ny)+y)*nx + x; within each probe 12 halfs ordered coeff0.r,coeff0.g,coeff0.b, coeff1.r,coeff1.g,coeff1.b, coeff2.r,coeff2.g,coeff2.b, coeff3.r,coeff3.g,coeff3.b. Coeffs are RADIANCE SH (L1 real basis): 0=Y00(0.282095), 1=Y1-1(0.488603*y), 2=Y10(0.488603*z), 3=Y11(0.488603*x). Viewer reconstructs IRRADIANCE via cosine convolution A0=pi, A1=2pi/3.",
        "sun_dir": b.sun_dir,
        "bounces": b.bounces,
        // false = INDIRECT-ONLY (practicals excluded; the viewer renders them real-time). The viewer
        // auto-enables the realtime light grid when this is present and false (direct/indirect split).
        "direct": b.direct,
        // Per-probe validity (Unity APV): u8 per probe, probe-major, SAME index as volume.bin.
        // 255 = open space, 0 = buried in geometry. The viewer weights its trilinear taps by this so
        // probes sealed inside walls stop bleeding onto the surfaces around them.
        "validity": "volume_valid.bin",
        "validity_layout": "u8 per probe, probe-major, same index as volume.bin; 255=valid/open, 0=inside geometry (backface ratio >= 0.25)",
        "baker": "atlas-cpu-sh (sh_bake.rs, M3: sky-visibility + shadow-tested practicals + diffuse bounce; APV-style probe validity)",
    });
    std::fs::write(dir.join("volume.json"), serde_json::to_string_pretty(&meta)?)
        .with_context(|| format!("writing {}", dir.join("volume.json").display()))?;
    Ok(())
}

// ---- CLI: `atlas bake-sh <pack_dir> [--rays N]` -----------------------------------------------

/// Handle the headless `bake-sh` subcommand. Returns a process exit code (0 = ok). Never panics.
pub fn run_cli(args: &[String]) -> i32 {
    let mut pack_dir: Option<String> = None;
    // WORKER/RENDERER COORDINATION: heavy GPU compute next to an interactive renderer risks a
    // Windows TDR (~2 s/dispatch), and a TDR resets the ADAPTER — losing the viewer's device too,
    // not just ours. So when a viewer holds the interactive-GPU lease we take the CPU path by
    // default (minutes, not seconds, but it cannot take the viewer down). An explicit
    // `--backend gpu` still overrides. See viewer/src/gpu_lease.rs.
    // Settings ▸ General ▸ "Force CPU processing": the build spawner exports EFT_BAKE_CPU=1 and it
    // inherits down to every bake child. Same escape hatch as an explicit `--backend cpu` — for
    // machines whose GPU driver crashes/TDRs on the compute bake (an explicit --backend flag on the
    // command line still overrides, same as the lease rule below).
    let mut backend = if std::env::var("EFT_BAKE_CPU").map(|v| v.trim() == "1").unwrap_or(false) {
        eprintln!("  bake-sh: EFT_BAKE_CPU=1 (Force CPU processing) -- using the CPU backend");
        Backend::Cpu
    } else if crate::gpu_lease::busy() {
        eprintln!("  bake-sh: a viewer holds the GPU -- using the CPU backend (pass --backend gpu to override)");
        Backend::Cpu
    } else {
        Backend::Auto
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rays" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<usize>().ok()) {
                    Some(v) => std::env::set_var("EFT_SH_RAYS", v.to_string()),
                    None => {
                        eprintln!("bake-sh: --rays needs an integer");
                        return 2;
                    }
                }
            }
            // Direct/indirect split: bake only INDIRECT (sky + bounce); the practicals become
            // real-time direct lights in the viewer. Marks volume.json "direct": false.
            "--indirect-only" | "--indirect" => std::env::set_var("EFT_SH_INDIRECT", "1"),
            // Pass-A engine: auto (default; GPU with silent CPU fallback), gpu (force GPU attempt),
            // cpu (rayon only). The GPU path is vendor-neutral wgpu compute (NVIDIA + AMD).
            "--backend" => {
                i += 1;
                backend = match args.get(i).map(String::as_str) {
                    Some("auto") => Backend::Auto,
                    Some("gpu") => Backend::Gpu,
                    Some("cpu") => Backend::Cpu,
                    other => {
                        eprintln!("bake-sh: --backend needs auto|gpu|cpu (got {other:?})");
                        return 2;
                    }
                };
            }
            s if s.starts_with('-') => {
                eprintln!("bake-sh: unknown flag '{s}'");
                return 2;
            }
            s => {
                if pack_dir.is_none() {
                    pack_dir = Some(s.to_string());
                } else {
                    eprintln!("bake-sh: unexpected extra argument '{s}'");
                    return 2;
                }
            }
        }
        i += 1;
    }
    let Some(dir) = pack_dir else {
        eprintln!("usage: atlas bake-sh <pack_dir> [--rays 256] [--backend auto|gpu|cpu] [--indirect-only]");
        return 2;
    };
    let dir_path = Path::new(&dir);
    let t0 = Instant::now();
    eprintln!("bake-sh: loading pack '{dir}'");
    let pack = match Pack::load(dir_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("bake-sh: failed to load pack '{dir}': {e:#}");
            return 1;
        }
    };
    let baked = match bake(&pack, backend) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bake-sh: bake failed: {e:#}");
            return 1;
        }
    };
    if let Err(e) = write_volume(&baked, dir_path) {
        eprintln!("bake-sh: writing volume files failed: {e:#}");
        return 1;
    }
    eprintln!(
        "bake-sh: OK -> volume.bin ({} probes x 12 halfs = {} bytes), volume.json; {} inside-solid; {:.2}s total",
        baked.dims[0] * baked.dims[1] * baked.dims[2],
        baked.halfs.len() * 2,
        baked.inside_solid,
        t0.elapsed().as_secs_f32()
    );
    0
}
