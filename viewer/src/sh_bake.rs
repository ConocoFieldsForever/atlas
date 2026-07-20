//! sh_bake — PORTABLE CPU SH irradiance-volume baker (rayon; NO CUDA, NO GPU vendor lock).
//!
//! WHY: the only lighting bake that existed (`extraction/bake/bake_volume2.py`) needs NVIDIA-Warp +
//! a CUDA GPU, so AMD / Intel / no-CUDA builds skip it and fall back to the flat realtime path (the
//! render review's "surface detail looks dogshit" on those machines). This bakes the SAME
//! world-space L1 SH radiance volume the viewer already consumes, on the CPU in parallel (rayon),
//! reusing `nav_bake`'s world-triangle assembly + BVH (shared code, one geometry path). Invoked
//! headless as `atlas bake-sh <pack_dir>`, exactly like `atlas bake-nav`.
//!
//! MILESTONE 1 (this file): SKY-VISIBILITY bake — per probe, cast a Fibonacci sphere of rays; a ray
//! that escapes the geometry sees a neutral sky gradient, an occluded ray sees nothing. Project the
//! visible sky radiance into L1 SH. That alone replaces the flat dummy SH with genuine, directional
//! sky-occlusion lighting (open areas bright + directional, enclosed areas darker) on any GPU.
//! M2 adds shadow-tested practical lights; M3 adds the diffuse bounce + emissive. The OUTPUT FORMAT
//! is already final (volume.json + volume.bin, the exact layout `NavGrid`/the viewer's `load_sh_volume`
//! reads) so each milestone just improves the numbers, never the plumbing.
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
const GRID_MAX_XY: u64 = 8192; // ny*nz cap (WebGL tex height, same as bake_volume2)
const GRID_MAX_X: usize = 4096;
const GRID_MAX_PROBES: u64 = 2_600_000;

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
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
    inside_solid: usize,
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

/// True if the ray `o + t*d` (t in (eps, inf)) hits ANY triangle in the BVH — i.e. the ray does NOT
/// escape to the sky. Slab-prunes each node's AABB, Möller–Trumbore at leaves, early-out on first hit.
/// Reuses the SAME `Bvh` (nodes + tris) `nav_bake` builds — the node AABBs are full 3-D, so the
/// X/Z-split tree is a valid (if not Y-optimal) accelerator for arbitrary rays.
fn ray_occluded(bvh: &Bvh, o: Vec3, d: Vec3, stack: &mut Vec<u32>) -> bool {
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
        if enter > exit {
            continue;
        }
        if node.count > 0 {
            let s = node.start as usize;
            for t in &bvh.tris[s..s + node.count as usize] {
                if ray_tri(o, d, t) {
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

/// Möller–Trumbore: does the ray `o + t*d` cross triangle `t` at some t > RAY_EPS?
#[inline]
fn ray_tri(o: Vec3, d: Vec3, t: &Tri) -> bool {
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1.0e-8 {
        return false; // parallel
    }
    let inv = 1.0 / det;
    let tv = o - t.a;
    let u = tv.dot(p) * inv;
    if u < 0.0 || u > 1.0 {
        return false;
    }
    let q = tv.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let tt = e2.dot(q) * inv;
    tt > RAY_EPS
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

fn bake(pack: &Pack) -> Result<Baked> {
    let n_dir = env_usize("EFT_SH_RAYS", 256).max(8);
    let sky_scale = env_f32("EFT_SKY", 2.0);

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

    // --- per-probe sky-visibility SH, parallel over Z-rows ---
    let t_bake = Instant::now();
    let inside = std::sync::atomic::AtomicUsize::new(0);
    let mut halfs = vec![0u16; n_probe * 12];
    // Precompute the ray directions + their SH bases once (shared across all probes).
    let dirs: Vec<(Vec3, [f32; 4])> = (0..n_dir).map(|i| { let d = fib_dir(i, n_dir); (d, sh_basis(d)) }).collect();
    let norm = 4.0 * std::f32::consts::PI / n_dir as f32;
    // Parallelize over probes directly (each probe independent). Per-thread BVH stack via for_each_init.
    halfs
        .par_chunks_mut(12)
        .enumerate()
        .for_each_init(
            || Vec::<u32>::with_capacity(64),
            |stack, (pi, out)| {
                let x = pi % nx;
                let y = (pi / nx) % ny;
                let z = pi / (nx * ny);
                let o = Vec3::new(
                    gmin.x + x as f32 * spacing[0],
                    gmin.y + y as f32 * spacing[1],
                    gmin.z + z as f32 * spacing[2],
                );
                let mut sh = [Vec3::ZERO; 4];
                let mut n_sky = 0u32;
                for (d, basis) in &dirs {
                    if ray_occluded(&bvh, o, *d, stack) {
                        continue; // occluded -> sees no sky (M1: hits contribute nothing)
                    }
                    n_sky += 1;
                    let l = sky(*d, sky_scale);
                    for c in 0..4 {
                        sh[c] += l * basis[c];
                    }
                }
                if n_sky == 0 {
                    inside.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                for c in 0..4 {
                    let v = sh[c] * norm;
                    out[c * 3] = f16_bits(v.x);
                    out[c * 3 + 1] = f16_bits(v.y);
                    out[c * 3 + 2] = f16_bits(v.z);
                }
            },
        );
    let inside_solid = inside.into_inner();
    eprintln!(
        "  sh-bake: baked {n_probe} probes in {:.2}s ({} fully-occluded/inside-solid)",
        t_bake.elapsed().as_secs_f32(),
        inside_solid
    );

    Ok(Baked {
        min: [gmin.x, gmin.y, gmin.z],
        max: [gmax.x, gmax.y, gmax.z],
        dims,
        spacing,
        sun_dir,
        halfs,
        inside_solid,
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
    let meta = serde_json::json!({
        "min": b.min, "max": b.max,
        "dims": [b.dims[0], b.dims[1], b.dims[2]],
        "spacing": b.spacing,
        "coeffs": 4, "channels": 3,
        "layout": "float16 LE, probe-major; probe index = ((z*ny)+y)*nx + x; within each probe 12 halfs ordered coeff0.r,coeff0.g,coeff0.b, coeff1.r,coeff1.g,coeff1.b, coeff2.r,coeff2.g,coeff2.b, coeff3.r,coeff3.g,coeff3.b. Coeffs are RADIANCE SH (L1 real basis): 0=Y00(0.282095), 1=Y1-1(0.488603*y), 2=Y10(0.488603*z), 3=Y11(0.488603*x). Viewer reconstructs IRRADIANCE via cosine convolution A0=pi, A1=2pi/3.",
        "sun_dir": b.sun_dir,
        "bounces": 0,
        "baker": "atlas-cpu-sh (sh_bake.rs, M1 sky-visibility)",
    });
    std::fs::write(dir.join("volume.json"), serde_json::to_string_pretty(&meta)?)
        .with_context(|| format!("writing {}", dir.join("volume.json").display()))?;
    Ok(())
}

// ---- CLI: `atlas bake-sh <pack_dir> [--rays N]` -----------------------------------------------

/// Handle the headless `bake-sh` subcommand. Returns a process exit code (0 = ok). Never panics.
pub fn run_cli(args: &[String]) -> i32 {
    let mut pack_dir: Option<String> = None;
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
        eprintln!("usage: atlas bake-sh <pack_dir> [--rays 256]");
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
    let baked = match bake(&pack) {
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
