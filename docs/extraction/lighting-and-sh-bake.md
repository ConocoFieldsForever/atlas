## Contents

1. [What EFT ships and what it does not](#1-what-eft-ships-and-what-it-does-not)
2. [Light extraction: Unity `Light` → `lights_<level>.json`](#2-light-extraction-unity-light--lights_leveljson)
3. [Light placement: the conjugation reduced to a point and a vector](#3-light-placement-the-conjugation-reduced-to-a-point-and-a-vector)
4. [Reducing a raw light to a bake/runtime record](#4-reducing-a-raw-light-to-a-bakeruntime-record)
5. [Occluder geometry and the BVH](#5-occluder-geometry-and-the-bvh)
6. [The probe grid: bounds, spacing, caps](#6-the-probe-grid-bounds-spacing-caps)
7. [SH convention: basis, radiance vs irradiance](#7-sh-convention-basis-radiance-vs-irradiance)
8. [Probe validity and virtual offset](#8-probe-validity-and-virtual-offset)
9. [Pass A - sky visibility + shadow-tested practicals](#9-pass-a--sky-visibility--shadow-tested-practicals)
10. [Pass B - one diffuse bounce with per-material albedo](#10-pass-b--one-diffuse-bounce-with-per-material-albedo)
11. [`volume.bin` / `volume_valid.bin` / `volume.json` byte formats](#11-volumebin--volume_validbin--volumejson-byte-formats)
12. [GPU backend: buffer layouts, chunking, TDR-safe batching](#12-gpu-backend-buffer-layouts-chunking-tdr-safe-batching)
13. [How the viewer samples the volume](#13-how-the-viewer-samples-the-volume)
14. [The direct/indirect split and the realtime light grid](#14-the-directindirect-split-and-the-realtime-light-grid)
15. [Invariants and failure signatures](#15-invariants-and-failure-signatures)
16. [Environment knobs](#16-environment-knobs)

---

## 1. What EFT ships and what it does not

**There is no baked lighting to rip:**

| Asset class | Status in the shipped levels |
|---|---|
| Lightmaps | None. All 11 `LightmapSettings` are empty; every `MeshRenderer` has `m_LightmapIndex = 65535` (Unity's "not lightmapped" sentinel). |
| `ReflectionProbe` | None authored at map scale. |
| `LightProbeGroup` / `LightProbes` | None. |
| `Cubemap` | None (map-scale sky cubemaps are re-derived separately). |
| Directional light (sun) | **Zero `Light` components of `m_Type == 1`** on the mainline maps. `RenderSettings.m_Sun` is null. The sun is driven by a day/night **script** at runtime, not by a scene object. Any Directional that does appear in a scene is deliberately discarded (`viewer/src/eftpack.rs:669-671` returns `Unsupported`). |
| Point / Spot lights ("practicals") | The real lighting. Interchange: **7802 `Light` components → 1285 live** after the enabled/active filter. Historic unfiltered export: ~7800 raw = 4403 Point + 3399 Spot. |

**Unverified:** the asset-class survey above (11 empty `LightmapSettings`, `m_LightmapIndex = 65535`, absent `ReflectionProbe`/`LightProbeGroup`/`Cubemap`, null `RenderSettings.m_Sun`) and the light counts (7802 → 1285; ~7800 raw = 4403 Point + 3399 Spot) were not re-probed against game data for this document. The 1285 figure is independently confirmed by `viewer/assets/shaders/gpu_draw.wgsl:216`; the rest carry over from earlier probes of the same maps and should be re-measured before being relied on. Only the discard behaviour is confirmed in code.

Consequences for a reimplementation:

- Any "sun" in the output is **synthetic**. The baker writes a fixed fallback `sun_dir = [0.449, 0.799, -0.400]` (`viewer/src/sh_bake.rs:887-889`) purely so the viewer's sky-reflection term has a direction. Do not present it as game data.
- Indoor lighting must be reconstructed by ray-tracing the point/spot set. Nothing else exists.
- Directional / `Rectangle` / `Disc` light types are **skipped, never coerced** into the point loop (`viewer/src/eftpack.rs:669-671`). Coercing a Directional into a point light drops a runaway bulb at the sun's transform position.

---

## 2. Light extraction: Unity `Light` → `lights_<level>.json`

Authority: `extraction/unity/eft_extract_lights.py`. Lights live in the map's `*_light` scene (Interchange = `level64`); multi-district maps split them across several scenes and **all** sidecars must be merged.

**Type enum** (`m_Type`, `extraction/unity/eft_extract_lights.py:28`): `0=Spot, 1=Directional, 2=Point, 3=Rectangle, 4=Disc`.

**Read scalars from the type tree, not from the object accessor.** Attribute access silently misses `m_Type`, defaulting every light to Point (`:194-197`). Read `m_Color{r,g,b,a}` (linear), `m_Type`, `m_Range`, `m_SpotAngle` (FULL cone, degrees), `m_InnerSpotAngle`, `m_Shadows.m_Type` off `read_typetree()`.

**World transform.** Walk `m_Father` up to the root (depth guard 256), then multiply back down: `W = Π TRS(t)` for `t` in reversed chain (`:134-149`). Position is `W[:3,3]`; the rotation quaternion (xyzw) comes from the orthonormal `W[:3,:3]` via the standard trace/largest-diagonal branch (`:184-193`).

**Liveness filter** (this is load-bearing - without it Factory_Rework imports ~5x too dense):
- `Light.m_Enabled` must be true (`:169`), AND
- every ancestor GameObject's `m_IsActive` must be true (`:141-144`).
- `intensity <= 0` with no controller → dropped as a placeholder (`:157-165`).
- **Pooled-at-origin cull**: drop when `|x| < 0.01 and |z| < 0.01` (`:181`).

**Controller-driven intensity.** Newer maps (Icebreaker+, partly Factory_Rework) serialize `m_Intensity = 0` and drive the lamp from a sibling MonoBehaviour on the same GameObject (`:45-83`). Decode by raw payload size:
- **220 bytes** → `struct.unpack("<55f")`; **`f[9]` = intensity**, accepted when `0.05 < f[9] < 60`.
- **92 bytes** → `struct.unpack("<23f")`; **`f[9]` = intensity**, accepted when `0.03 < f[9] < 60`.

The payload size gates and both acceptance windows are confirmed in code. **Unverified:** the field semantics and the statistics behind them - that `f[9]` is intensity, that `f[8]`/`f[28]` resemble spotAngle/range, and the 10–40 m vs authored 3–7 m distribution argument - are attested only by the source comments at `:45-83` (`:70-73` for the range argument). The design rule they support is unchanged: only intensity is taken from the controller, and **range, angle and colour stay authored**; overriding range makes every lamp span a whole deck and washes out all contrast.

**Power-switch groups.** A map's power lever is an `EFT.Interactive.Switch` whose serialized `PPtr[]` resolves *entirely* to `EFT.Interactive.LampController`; that array is the exact bank it powers. Class names come from each MonoBehaviour's `m_Script → MonoScript.m_ClassName` (readable even though the IL2CPP payload is not), so nothing keys on a GameObject name and the rule is map-agnostic (`:89-123`). Each controlled light is tagged `"group": "<level>:<switchGO>"` and **force-kept even when it ships off** (`:176-179`) - the mall is dark until the lever is thrown, and a dropped record makes the lever inert.

**Output record** (rounding in parentheses, `:202-215`):

```
{ name, type: "Point"|"Spot"|"Directional",
  position:[x,y,z]        // Unity world, 4 dp
  rotation:[x,y,z,w]      // Unity world quat, 5 dp
  color:[r,g,b,a]         // LINEAR, 4 dp
  intensity,              // 4 dp
  range,                  // meters, 3 dp
  spotAngle,              // FULL cone degrees, 2 dp
  innerSpotAngle,         // FULL cone degrees, 2 dp
  shadowType,             // int
  on: bool,               // false only survives when `group` is present (or --all)
  group?: "<level>:<switchGO>" }
```

Written to `lights_<level>.json` (or `lights_<level>_all.json` with `--all`, which also keeps ungrouped disabled banks tagged `on:false`). The consumer must tolerate both a bare array and `{"lights":[...]}` (`viewer/src/eftpack.rs:641-646`).

---

## 3. Light placement: the conjugation reduced to a point and a vector

Geometry uses the instance conjugation `G·M·G⁻¹` with the EFT handedness flip `G3 = diag(-1, 1, 1)`. A light has no mesh, so the conjugation collapses:

```
position:      p_world = G3 · p_unity        =  (-x,  y,  z)
spot forward:  f       = R_quat · (0,0,1)    // Unity forward is +Z
               dir     = normalize(G3 · f)   =  normalize(-f.x, f.y, f.z)
```

`viewer/src/eftpack.rs:685-697`. `G3` is an involution (`G3 = G3⁻¹`), which is why the two-sided conjugation degenerates to a single application on each of a point and a direction.

**Verification datum** (*carried over from an earlier probe; not re-measured for this document*): with the flip omitted, the light-set bounding box has `maxX = 654.8` while the geometry's is `502.1` - lights sit outside the map. X-flipped, `maxX = 474.4`, inside. The rule itself is map-agnostic: `G3` comes from the global coordinate matrix, never a per-map constant.

**Cone angles.** `spotAngle` / `innerSpotAngle` are **full** cone angles in degrees:

```
cos_outer = cos(radians(spotAngle)      * 0.5)
cos_inner = cos(radians(innerSpotAngle) * 0.5)
if cos_inner <= cos_outer: cos_inner = cos_outer + 1e-3   // smoothstep needs inner > outer
```

**Point-light sentinels:** `dir = (0,0,0)`, `cos_outer = -2.0`, `cos_inner = -1.0`. The `-2.0` is what every downstream cone test branches on (`cos_outer > -1.5` ⇒ spot), and it also makes a naive `smoothstep(cos_outer, cos_inner, cosang)` evaluate to 1 everywhere without a branch.

**Intensity is folded into colour at parse time**: `color = linear_rgb * intensity` (`viewer/src/eftpack.rs:687`). Everything downstream treats `color` as radiant colour.

---

## 4. Reducing a raw light to a bake/runtime record

`viewer/src/eftpack.rs:664-717`. Runtime struct:

```
Light { pos: vec3, color: vec3, range: f32, dir: vec3,
        cos_outer: f32, cos_inner: f32, group_idx: i32 }
```

Rejections:
- type not point/spot → `Unsupported` (counted + warned in aggregate, never silently coerced).
- `(!on && group.is_none()) || intensity <= 0 || range <= 0` → `Inactive` (`:674`).
- `group_idx` is a dense index assigned first-seen from the group string, shared with the switch table so a switch and its lights agree; `-1` = always on.

---

## 5. Occluder geometry and the BVH

Both bake passes ray-trace the same world-space triangle soup the navmesh baker builds, so there is exactly one geometry path.

`build_tris(pack)` (`viewer/src/nav_bake.rs:404`) returns `(column_tris, wall_tris, min_y, max_y, door_count)`. The SH baker concatenates walls into columns to form the full occluder set (`viewer/src/sh_bake.rs:472-474`) - the nav baker keeps them separate, the light bake needs both.

```
Tri  { a, b, c : vec3 (world),      // nav_bake.rs:262-273
       ny   : f32,                  // world normal Y (sign-corrected for mirrored instances)
       door : bool,
       mat  : u32 }                 // SubMesh.material_id - the bounce's albedo LUT key

BvhNode { min, max : vec3,          // nav_bake.rs:1069-1075
          start, count : u32 }      // leaf: tris[start .. start+count]
                                    // internal (count == 0): children at start, start+1
```

The tree is a median split over **XZ only**, but node AABBs are full 3-D, so it is a valid (if not Y-optimal) accelerator for arbitrary rays (`viewer/src/sh_bake.rs:125-129`).

**Ray tests** - `RAY_EPS = 0.02` m is both the slab-entry clamp and the triangle `t_min` (`viewer/src/sh_bake.rs:43`):

- Slab: `enter = max(min(t0,t1).{x,y,z}, RAY_EPS)`, `exit = min(max(t0,t1).{x,y,z})`; reject when `enter > exit || enter > t_max` (`:139-147`).
- Möller–Trumbore, parallel threshold `|det| < 1e-8`, accepts `t ∈ (RAY_EPS, t_max)` (`:166-191`).
- Any-hit (occlusion) early-outs on the first hit. Nearest-hit has no early-out; instead `best_t` prunes any box whose entry is already farther, and children are pushed **far-first** so the near child pops first and tightens `best_t` (`:234-248`). Ordering affects speed only, never which hit is nearest.
- GPU walks use a fixed 64-entry stack with `sp < 63` push guards (`viewer/assets/shaders/sh_bake.wgsl:90,105-106`).

Two `t_max` conventions carry all the meaning: the **sky ray** is unbounded (does the ray escape?), the **shadow ray** uses `dist - 0.1` (is anything *between* probe and bulb?). "Unbounded" is spelled differently per backend: `f32::INFINITY` on the CPU path (`viewer/src/sh_bake.rs:653`, `:267`) versus the finite sentinel `1.0e30` in the GPU port (`viewer/assets/shaders/sh_bake.wgsl:148`). This is the one place the two backends are not literally the same constant (§12).

---

## 6. The probe grid: bounds, spacing, caps

`viewer/src/sh_bake.rs:485-528`.

**Bounds.** By this point `column_tris` has already been extended with `wall_tris` (`:474`), so the percentile sampling runs over the **full occluder soup** (floors + ceilings + walls), not over column triangles only. Sample the `a` vertex of every `step`-th triangle, `step = max(n_tris / 500_000, 1)` (`:486`; the loop is at `:490`):

```
lo_x, hi_x = 1st, 99th percentile of x
lo_z, hi_z = 1st, 99th percentile of z
ylo = 0.5th  percentile of y - 2.0
yhi = 99.7th percentile of y + 4.0
gmin = (lo_x, ylo, lo_z);  gmax = (max(hi_x, lo_x+1), max(yhi, ylo+1), max(hi_z, lo_z+1))
```

The percentile clip is what keeps a single stray far-flung triangle from ballooning the grid. `ylo` sitting *below* all walkable ground is deliberate and is later compensated for in the shader (§13).

**Spacing.**

```
XZ_TARGET = 3.0 m,  Y_SPACING = 4.0 m
sxz = max(3.0, sqrt(ext.x * ext.z / 120000))
loop:
  nx = max(ceil(ext.x / sxz) + 1, 2)
  ny = max(ceil(ext.y / 4.0) + 1, 2)
  nz = max(ceil(ext.z / sxz) + 1, 2)
  ok = (ny*nz <= 8192) && (nx <= 4096) && (nx*ny*nz <= 2_600_000)
  break if ok or 64 iterations;  sxz *= 1.15
spacing[axis] = ext[axis] / (n[axis] - 1)          // NOT sxz / Y_SPACING
```

The `ny*nz <= 8192` cap is a WebGL texture-height limit inherited from the reference baker; `nx <= 4096` and the 2.6 M probe budget are the other two.

**The spacing invariant.** Spacing is *recomputed* as `extent/(n-1)`, so probe `i` sits at exactly `min + i*spacing` and probe `n-1` lands exactly on `max`. Every consumer (bounce gather, shader 8-tap, texel-center uvw) assumes this. Emitting the *target* spacing instead of the derived one shifts every sample by a fraction of a cell.

**Probe index (probe-major, X fastest):**

```
pi = (z * ny + y) * nx + x
x = pi % nx;  y = (pi / nx) % ny;  z = pi / (nx * ny)
world = gmin + (x, y, z) * spacing
```

`viewer/src/sh_bake.rs:342-351`. This ordering is identical to a GPU 3-D texture's texel order, which is why the upload is a straight memcpy (§13).

---

## 7. SH convention: basis, radiance vs irradiance

**Sampling directions - spherical Fibonacci** (`viewer/src/sh_bake.rs:97-103`), golden angle `GA = 2.3999632`:

```
z   = 1 - (2i + 1) / n
r   = sqrt(max(1 - z², 0))
phi = GA * i
d   = (r·cos(phi),  z,  r·sin(phi))          // Y-up
```

**L1 real basis, in the exact stored order** (`viewer/src/sh_bake.rs:108-110`):

```
basis[0] = 0.282095            // Y00
basis[1] = 0.488603 * d.y      // Y1-1
basis[2] = 0.488603 * d.z      // Y10
basis[3] = 0.488603 * d.x      // Y11
```

Note the mapping: **coeff 1 ↔ y, coeff 2 ↔ z, coeff 3 ↔ x.** This is not the textbook `(x,y,z)` ordering; it is fixed by the format and repeated verbatim in the bake shaders and the viewer. Getting it wrong tilts every dominant-light direction and every bounce gather by a coordinate swap.

**Stored coefficients are RADIANCE SH, not irradiance.** Irradiance is reconstructed at sample time by cosine convolution (`A0 = π`, `A1 = 2π/3`):

```
E(n) = 0.8862269 · c0  +  1.0233267 · (c1·n.y + c2·n.z + c3·n.x)      then clamp per channel ≥ 0
       ^ π·0.282095       ^ (2π/3)·0.488603
```

`viewer/src/sh_bake.rs:376-378`, `viewer/assets/shaders/sh_bounce.wgsl:144`.

The renderer folds the Lambert `1/π` in and uses the divided form:

```
E(n)/π = 0.282095 · c0  +  0.325735 · (c1·n.y + c2·n.z + c3·n.x)
```

`viewer/assets/shaders/gpu_draw.wgsl:336-340`. `0.325735 = (2/3)·0.488603`.

**Radiance toward a direction L** (used for the dominant-light and directional-occlusion terms) uses the *unconvolved* basis: `0.282095·c0 + 0.488603·(c1·L.y + c2·L.z + c3·L.x)` (`viewer/assets/shaders/gpu_draw.wgsl:504-506`).

**Solid-angle normalization.** Anything integrated over the sphere by Monte-Carlo (sky, bounce) is multiplied by `norm = 4π / n_rays` after accumulation. Anything that is a **delta light** (a point/spot bulb - one direction, not a solid-angle sample) is added *after* that scale and is **not** multiplied by `norm`. Mixing these up scales the practicals by `4π/256 ≈ 0.049` and the interiors go black.

---

## 8. Probe validity and virtual offset

Unity's Adaptive Probe Volumes terminology, reimplemented.

**Backface ratio** (`viewer/src/sh_bake.rs:261-279`). Fire every `stride`-th Fibonacci direction, take the **nearest** hit, compute the geometric normal from the winding `n = (b-a) × (c-a)`, and count the hit as a backface when `n · d > 0`. Ratio = backfaces / hits; 0 when nothing was hit. A probe in open space sees front faces or sky; a probe buried in a wall sees that wall's inside.

**Validity byte** (`:561-571`), computed with `v_stride = max(n_dir/64, 1)` ⇒ ~64 rays (this is a low-frequency question):

```
validity_u8 = round( clamp(1 - r/0.25, 0, 1) * 255 )
```

so `r ≥ 0.25` ⇒ 0. Unity flags a probe invalid past 25% backfaces; storing the continuous ramp and letting the shader weight by it degrades more gracefully than a hard flag. "Invalid" is reported as `< 128`.

**Virtual offset** (`viewer/src/sh_bake.rs:295-326`). Rejecting an in-wall probe at runtime only converts it into a hole, and the hole is its own artefact: measured on Interchange, a window sat 98% of the way into its cell, so the probe carrying nearly all the trilinear weight was the invalid one and the cell collapsed to zero GI - a hard, probe-grid-aligned rectangle painted across the glass. Runtime offsetting cannot fix it either, because it can only step along the *shaded surface's* normal (+X on that window) while the lighting cliff runs along a different axis (Z).

So move the probe before lighting it:

```
candidates = 6 face axes, then 8 cube diagonals (14 total), each normalized
step = min(spacing.x, spacing.y, spacing.z)
for frac in [0.5, 1.0]:
    for dir in candidates:
        c = o + normalize(dir) * (step * frac)
        r = backface_ratio(c, stride = v_stride * 4)     // ~16 rays
        keep the candidate with the LOWEST r among those with r < 0.25
    if any candidate kept: return it
return o                                                  // deep inside rock: correctly still dark
```

Nearest radius first, so a probe moves the shortest distance that reaches open space - a probe dragged far stops representing its own cell.

**Restrictions and bookkeeping:**
- Only run for probes with `validity < 128` **and** at least one of the 6 axis neighbours valid (`:589-604`). A probe deep inside rock has nowhere useful to go, and skipping it keeps the search affordable.
- A relocated probe is lit from open space, so it is no longer a hole: set its `validity = 255` (`:630-634`).
- **The relocated position must be used by every subsequent pass.** The relocation rate is large enough that deriving positions from `(gmin, spacing, index)` is wrong - *unverified: the specific figure is attested only by source comments, 25.7% at `viewer/src/sh_bake.rs:787` and 26% at `viewer/assets/shaders/sh_bake.wgsl:136`; not re-measured.* Both GPU shaders therefore read a per-probe position buffer (`viewer/assets/shaders/sh_bake.wgsl:138`, `sh_bounce.wgsl:164`); when nothing moved the host uploads the plain grid positions, so the two regimes are numerically identical.

---

## 9. Pass A - sky visibility + shadow-tested practicals

`viewer/src/sh_bake.rs:647-712`; GPU port `viewer/assets/shaders/sh_bake.wgsl:127-190`. One thread / one parallel task per probe.

**M1 - sky visibility.** `n_dir = 256` by default (min 8). For each Fibonacci direction: cast an **any-hit** ray with an unbounded `t_max` (`f32::INFINITY` on CPU, the `1.0e30` sentinel in WGSL, §5). If occluded, the ray sees nothing. If it escapes, it sees a neutral grayscale sky gradient:

```
sky(d) = (0.35 + 0.75 * max(d.y, 0)) * sky_scale     // horizon 0.35·s, zenith 1.1·s, no tint
c[k] += sky(d) * basis[k]  for k in 0..4
```

`sky_scale = 2.0` default. After the loop, **`c[k] *= 4π / n_dir`**.

**M2 - practicals** (added *after* the norm scale, *not* multiplied by it). Per light:

```
if light.group_idx >= 0: skip                    // see below
tol  = light.pos - probe;  dist = |tol|
r    = max(light.range, 4.0)                     // LIGHT_RANGE_FLOOR, bake-only
if dist <= 0.05 or dist >= r: skip
dl   = tol / dist
spot = 1.0
if cos_outer > -1.5:                             // -2.0 sentinel ⇒ point light
    cosang = -dot(dl, light.dir)
    spot = clamp((cosang - cos_outer) / (cos_inner - cos_outer + 1e-4), 0, 1)
if spot <= 0: skip
x    = dist / r
win  = clamp(1 - x⁴, 0, 1)
at   = win² / max(dist², 0.25)                   // MIN_D2 = 0.25 ⇒ clamp within 0.5 m of a bulb
if any_hit(probe, dl, t_max = dist - 0.1): skip  // shadowed
rad  = light.color * (at * spot * light_scale)   // light_scale = 6.0
c[k] += rad * basis(dl)[k]
```

This matches the CUDA reference (`extraction/bake/bake_volume2.py:455-470`) except for the spot-branch threshold: the reference tests `co > -0.999` (`:461`), Rust and WGSL test `cos_outer > -1.5`. Both are inert against the `-2.0` point sentinel, so a portable bake still reproduces the author-side one.

**`LIGHT_RANGE_FLOOR = 4.0` is bake-only.** The probe grid is coarse (~3–7 m cells), so a sub-floor light would influence roughly no probes and vanish entirely. Realtime rendering keeps the authored range (`viewer/src/sh_bake.rs:45-47`).

**Switch-controlled lights are excluded from the bake** (`viewer/src/sh_bake.rs:678-680`). The runtime's default power state is all-groups-off (mask 0), so baking them lit burns a permanently-powered mall into the volume *and* makes the lever inert, because a `"direct": true` volume disables the realtime grid entirely. Skip the whole **group**, not just the records tagged `on:false` - the default state zeroes all of it. *Unverified: the figure "Interchange ships 109 such records, all with intensity > 0, all inside the bake volume" is attested only by the source comment at `viewer/src/sh_bake.rs:675-676`.*

**Indirect-only mode** (`--indirect-only` / `EFT_SH_INDIRECT=1`) skips the entire M2 block; the volume then carries sky + bounce only and `volume.json` is marked `"direct": false`. This is what ships.

---

## 10. Pass B - one diffuse bounce with per-material albedo

`viewer/src/sh_bake.rs:768-856`; GPU port `viewer/assets/shaders/sh_bounce.wgsl`.

Pass A is kept in **f32** (not yet packed to f16) precisely so pass B can gather from it.

```
bounce_rays = 128 (clamped 1..4096);  bnorm = 4π / bounce_rays
max_dist    = |bvh.root.max - bvh.root.min| * 1.2      // a bounce ray may hit outside the probe band

for each Fibonacci direction d:
    (t, face) = NEAREST hit from the probe position, t_max = max_dist;  skip on miss
    n = normalize((b-a) × (c-a));  if dot(n, d) > 0: n = -n     // orient toward the incoming ray
    h = probe + d*t + n*0.05                                    // nudge off the surface
    E = irr_at(sh_A, h, n)                                      // trilinear + cosine convolution, ≥ 0
    rad = E * albedo[tri.mat] * (1/π · albedo_boost) + emissive[tri.mat] * emis_gain
    b[k] += rad * basis(d)[k]

out[k] = sh_A[k] + b[k] * bnorm
```

Half the sky-ray count is plenty (the bounce is low-frequency) and it roughly halves the costly nearest-hit pass.

**`irr_at`** (`viewer/src/sh_bake.rs:357-379`) is the *same* reconstruction the viewer uses: clamp `(p - gmin)/spacing` to `[0, n-1]` per axis, 8-corner trilinear on the raw coefficients, then the `0.8862269 / 1.0233267` cosine convolution, clamped ≥ 0. SH interpolates linearly, so interpolating coefficients and then reconstructing is correct.

**Per-material LUTs** (`viewer/src/sh_bake.rs:415-463`), built once by decoding every unique texture the pack references, in parallel:

```
sRGB→linear:  lut[i] = (i/255)^2.2                        // gamma 2.2, matches the reference baker
mean(tex)   = arithmetic mean of the LINEAR rgb over ALL pixels of the PNG

albedo[id]   = clamp( mean(albedo_png) * tint, 0, 1 )     // untextured fallback: tint * 0.5
emissive[id] = min( factor * hdr * coverage, 8.0 )        // coverage = mean(emissive_png) or 1.0
default for id gaps = (0.3, 0.3, 0.3)
```

The `clamp(...,1)` is an energy bound (a diffuse albedo is physically ≤ 1); the `8.0` ceiling is the reference baker's `EMIS_MAX`. Using the per-material mean of the pack's own source PNGs is what makes the bounce **coloured** - a red container bounces red, painted asphalt bounces dark.

Setting `EFT_SH_BOUNCE=0` packs pass A directly and writes `"bounces": 0`.

---

## 11. `volume.bin` / `volume_valid.bin` / `volume.json` byte formats

### `volume.bin`

- **IEEE-754 half (f16), little-endian.**
- **Probe-major**, `pi = ((z*ny) + y)*nx + x`.
- **12 halfs = 24 bytes per probe**, ordered:

```
byte  0.. 5 : c0.r c0.g c0.b     (Y00)
byte  6..11 : c1.r c1.g c1.b     (Y1-1, ∝ y)
byte 12..17 : c2.r c2.g c2.b     (Y10,  ∝ z)
byte 18..23 : c3.r c3.g c3.b     (Y11,  ∝ x)
```

- File size must be **at least** `nx * ny * nz * 24`. The loader computes `need = n_probes * 24` (`viewer/src/render/gpu_driven.rs:1110`) and tests `if bin.len() < need` (`:1111`), so it rejects **undersized** files only; an over-long `volume.bin` loads normally and the trailing bytes are ignored (`:1110-1118`).
- Values are **radiance**, unbounded positive HDR (§7).

The f32→f16 packer (`viewer/src/sh_bake.rs:894-914`) is a hand-rolled truncating converter - no rounding dependency:

```
sign = (bits >> 16) & 0x8000
exp  = ((bits >> 23) & 0xff) - 127 + 15
mant = bits & 0x7fffff
if (bits >> 23) & 0xff == 0xff: return sign | 0x7c00 | (0x200 if mant else 0)   // inf / nan
if exp >= 0x1f: return sign | 0x7c00                                            // overflow → inf
if exp <= 0:    return sign if exp < -10                                        // underflow → 0
                else sign | ((mant | 0x800000) >> (14 - exp))                   // subnormal
return sign | (exp << 10) | (mant >> 13)
```

### `volume_valid.bin`

**u8 per probe, probe-major, exactly the same index as `volume.bin`.** `255` = open space, `0` = buried in geometry (backface ratio ≥ 0.25). Length must equal `nx*ny*nz`; a mismatch means the two files came from different bakes and the mask must be ignored rather than misapplied (`viewer/src/render/gpu_driven.rs:1197-1210`).

It is a **separate file** on purpose: a consumer that predates it loads `volume.bin` byte-for-byte as before and simply renders without leak reduction.

### `volume.json`

`viewer/src/sh_bake.rs:926-943`:

```json
{ "min": [x,y,z], "max": [x,y,z],
  "dims": [nx,ny,nz], "spacing": [sx,sy,sz],
  "coeffs": 4, "channels": 3,
  "layout": "<the full self-describing string>",
  "sun_dir": [0.449,0.799,-0.400],
  "bounces": 1,
  "direct": false,
  "validity": "volume_valid.bin",
  "validity_layout": "u8 per probe, probe-major, same index as volume.bin; 255=valid/open, 0=inside geometry (backface ratio >= 0.25)",
  "baker": "<identifier>" }
```

Optional, read by the renderer but **never written by the baker** (`write_volume`, `viewer/src/sh_bake.rs:926-943`, does not emit it): `"gi_intensity"` - a per-map GI multiplier so a dark bake can be lifted without a rebuild. The serde field is declared at `viewer/src/render/gpu_driven.rs:1005-1010`; the value is sanitized at `:1174-1177` (`.filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(1.0)`), and the same filter guards the `EFT_GI` override at `:5276-5280`.

`coeffs != 4 || channels != 3` is a hard reject (`:1088-1094`).

---

## 12. GPU backend: buffer layouts, chunking, TDR-safe batching

`viewer/src/sh_bake_gpu.rs`. Vendor-neutral compute (no CUDA); both passes are faithful ports, and the only constant that differs from the CPU path is the unbounded-ray sentinel (`1.0e30` vs `f32::INFINITY`, §5), which cannot change a hit result at map scale - GPU and CPU produce matching volumes.

**Byte strides** (`:19-31`). Index fields are packed into the vec3 padding as **real u32**, never round-tripped through an f32 load (some GPUs denorm-flush that):

```
TRI_STRIDE   48  = a.xyz f32 + mat u32 | b.xyz f32 + pad | c.xyz f32 + pad
NODE_STRIDE  32  = min.xyz f32 + start u32 | max.xyz f32 + count u32
LIGHT_STRIDE 48  = (pos.xyz, range) | (color.rgb, cos_outer) | (dir.xyz, cos_inner)
MAT_STRIDE   32  = (albedo.xyz, _) | (emissive.xyz, _)
positions    16  = (x, y, z, 0) per probe
output       48  = 12 f32 per probe
```

**Chunking.** A single storage binding is capped at `max_storage_buffer_binding_size` (a u32 ⇒ ≤ 4 GiB, and 2 GiB on some Vulkan drivers), which Interchange (~5.8 GiB of tris) and Streets exceed. Tris and nodes are split across up to **3** bindings each; the shader indexes globally:

```
chunk = i / per_chunk;   local = i % per_chunk       // sh_bake.wgsl:43-56
tpc = cap / 48;  npc = cap / 32
```

Needing more than 3 chunks defers to CPU (`viewer/src/sh_bake_gpu.rs:124-129`).

**Bindings.** Pass A: `0` uniform params, `1..3` tris, `4..6` nodes, `7` lights, `8` output (read_write), `9` positions - requires ≥ 9 storage buffers per stage. Pass B: `1..3` tris (with material ids), `4..6` nodes, `7` pass-A grid, `8` materials, `9` output, `10` positions - requires ≥ 10. Workgroup size 64; dispatch `ceil(count / 64)`.

**Params structs are different sizes** (`viewer/src/sh_bake_gpu.rs:33-54`, mirrored in the WGSL):

```
ParamsA  6 x vec4 =  96 bytes:  gmin, spacing, dims, counts, consts, chunk
ParamsB  7 x vec4 = 112 bytes:  gmin, spacing, inv_sp, dims, counts, fconst, chunk
```

Pass B adds `inv_sp` = `1/spacing` with `.w = bnorm` and renames `consts` → `fconst` (`viewer/assets/shaders/sh_bounce.wgsl:14-22`). `chunk.z` is the **probe batch offset**, rewritten between dispatches (pass B writes it at `offset_of!(ParamsB, chunk) + 8`, `:568`). Sizing the pass-B uniform at 96 bytes truncates `chunk` - including that batch offset - so every batch after the first addresses the wrong probes.

**Adaptive TDR-safe batching** (`viewer/src/sh_bake_gpu.rs:626-719`). A single whole-map dispatch is fastest but blows the ~2 s Windows GPU watchdog; the driver then resets the device and the output comes back as zeros. So:

```
budget hi = 0.9 s (clamp 0.3..1.4)
first batch:  depth  = log2(max(n_nodes, 2))
              shrink = 2^(max(depth - 18, 0) / 2)      // 1x under 2^18 nodes, 16x at 2^26
              bsz    = max(4096 / shrink, 128), then max(bsz, 64)
per batch:    measure the poll wall time dt
              dt < hi        → bsz *= 2
              dt > hi * 1.6  → bsz /= 2  (floor 2048)
              else             hold
              clamp(bsz, 2048, n_probe)
```

Sizing by **dispatch wall time**, not per-probe rate: small batches under-utilize the GPU and report an inflated rate, which traps rate-based sizing at tiny batches forever. The first batch must scale with scene cost because adaptive sizing only has a measurement from batch 2 onward. *Unverified: the supporting anecdote - a fixed 4096-probe opener alone losing the device on a 129 M-tri / 67 M-node scene - is attested only by the source comment at `viewer/src/sh_bake_gpu.rs:656-661`.*

**Four independent safety layers**, all required:
1. `on_uncaptured_error` must **log and continue**. wgpu's default handler panics; a mid-batch device loss surfaces as "Parent device is lost" from `poll` and aborts the whole build stage, skipping the graceful fallback (`:106-108`).
2. `checked_mapped` pushes OOM + Validation error scopes **before** `create_buffer(mapped_at_creation: true)` and pops them **before** mapping. On an allocation failure wgpu hands back an error buffer and `get_mapped_range_mut()` **panics** - the OOM only becomes catchable at scope-pop, after the panic would have fired (`:138-154`).
3. `poll` returning `Err` mid-batch → return `None` → CPU fallback.
4. **All-zero net**: a watchdog reset silently zeros the buffer while `poll` still returns Ok. A real bake always has structure, so if > 98% of the output floats are exactly `0.0`, treat it as a TDR and fall back (`:711-717`).

**Backend selection** (`viewer/src/sh_bake.rs:963-971`): `EFT_BAKE_CPU=1` or an interactive viewer holding the GPU lease → CPU (a TDR resets the *adapter*, taking the viewer's device down too). Otherwise Auto: try GPU, fall back silently. An explicit `--backend gpu|cpu` overrides. **GPU pass B is opt-in** (`EFT_SH_GPU_BOUNCE=1`): nearest-hit is far costlier and far more per-probe-variable than any-hit occlusion, so TDR-safe batches for it would be so small that it barely beats the CPU.

---

## 13. How the viewer samples the volume

`viewer/src/render/gpu_driven.rs:1070-1230` (load), `viewer/assets/shaders/gpu_draw.wgsl:213-448` (sample).

**Repack, no float conversion.** The 12 halfs are shuffled into three per-channel buffers whose texel is `(c0,c1,c2,c3)`:

```
R texels ← source half indices 0, 3, 6,  9
G texels ← source half indices 1, 4, 7, 10
B texels ← source half indices 2, 5, 8, 11
```

Uploaded as three **RGBA16Float 3-D textures** of size `(nx, ny, nz)` - probe order (x-fastest → y → z) is exactly 3-D texel order, so it is a direct copy. Hardware trilinear then interpolates each SH coefficient for free, which is correct because SH interpolates linearly. Validity uploads as an **R8Unorm 3-D** texture on the same grid. Sampler is Linear with ClampToEdge on all three axes.

**Uniform, 64 bytes, four vec4** (`viewer/src/render/gpu_driven.rs:438-453`):

```
vol_min       : (min.xyz,        gi_intensity)
vol_inv_extent: (1/(max-min).xyz, normal_bias = 0.75 m)
dims          : (nx, ny, nz as f32, ground_over_top)
spacing       : (sx, sy, sz,       0)
```

**Texel-center uvw** - the single most error-prone line:

```
uvw = ((p - min) / spacing + 0.5) / dims
```

`viewer/assets/shaders/gpu_draw.wgsl:307-309`. Probes sit at `min + i*spacing`; a 3-D texture's texel *centers* sit at `(i+0.5)/N`. The align-corners form `(p-min)/extent` is off by up to half a texel, which blends ~40% of the below-floor probe layer (whose L1 is inverted) into every ground-height hardware sample. The manual 8-tap uses integer `textureLoad` and never suffered this - which is exactly why diffuse GI once matched across the volume boundary while the sun/dominant terms did not.

**The manual 8-tap** (`gpu_draw.wgsl:370-448`) replaces hardware trilinear for the diffuse term:

```
wp   = clamp(world_pos, min, min + extent)
sp   = wp + n_bias * 0.75                    // bias along the SURFACE normal
grid = clamp((sp - min) / spacing, 0, dims - 1);  base = floor(grid);  f = grid - base
per corner:
    tw = trilinear weight
    pv = textureLoad(sh_valid, ipc, 0).x                     // same integer coord as the SH tap
    wn = max(dot(normalize(probe_pos - wp + n_bias*1e-3), n_bias), 0) * pv
    w  = tw * (wn + 1e-4)                                    // the epsilon MUST ride tw
    accumulate w * max(E(n_eval), 0)
result = mix(hardware_trilinear, sum/wsum, smoothstep(0, 4e-3, wsum))
```

Two independent rejections, both APV: **validity** (baked, view-independent - the actual wall-leak fix) and **normal** (cheap, catches the below-slab case the bake cannot know about - a probe can be perfectly valid and still be on the wrong side of the floor you stand on).

Two details that look cosmetic and are not:
- The epsilon must be `tw * (wn + 1e-4)`, not `tw*wn + 1e-4`. As a constant floor it survives `tw → 0`, so a probe contributes even at zero trilinear weight; crossing a cell boundary swaps which 8 probes are in the octet and those floor terms change discontinuously - a hard-edged rectangular patch wherever a bright probe enters or leaves the set. Worst on windows, where the outward normal rejects every indoor probe and the floor terms alone decide the result.
- The handover to the hardware fallback must be a `smoothstep`, not an `if`. As a hard threshold it is a cliff mid-shading-function: neighbouring pixels either side reconstruct from completely different estimators, painting probe-grid-aligned rectangles with a texel staircase along the edge.

**Separate bias and eval directions.** `sh_irradiance_b(pos, n_eval, n_bias)` exists because the environment-reflection lookup passed the mirror vector as the normal, so `R` also drove the probe-position bias. The grid holds a hard indoor/outdoor cliff at glass, and as `R` swings across a pane the biased sample slides back and forth over that cliff, producing an axis-aligned bright rectangle. *Unverified: the measured magnitude - outdoor probes ~1.2–2.1 vs indoor ~0.01–0.04, a 40x step across one cell - is attested only by the source comment at `viewer/assets/shaders/gpu_draw.wgsl:361-364`.* Bias by the true surface normal; evaluate in the mirror direction.

**Out-of-volume redirect** (`gpu_draw.wgsl:311-328`). The grid's bounds are triangle-density-derived and collapse around dense hubs, so big open maps under-cover. Beyond the AABB, clamping smears whatever the nearest edge probe held into infinite razor-straight bands. Instead:

```
t_out   = smoothstep(0, 2*max(spacing), max component of the outside distance)
sky_pos = (clamp(p).x, max.y - 0.5*spacing.y, clamp(p).z)      // the volume's own top probe row
result  = mix(local_8tap, sample(sky_pos) * ground_over_top, t_out)
```

`ground_over_top` = mean layer-1 `c0` luma ÷ mean top-layer `c0` luma over probes whose luma > 0.05, Rec.709 weights, clamped `[0.5, 1.5]` (`viewer/src/render/gpu_driven.rs:1133-1194`). Blend **results**, not positions - a position lerp walks the sample through mid-air heights where the hemisphere weights collapse.

**Dominant light for specular** (`gpu_draw.wgsl:467-513`). The L1 band encodes the linear part of incident radiance; its luminance-weighted direction is the "sun-ish" light:

```
dom = (lum(c3), lum(c1), lum(c2))              // x from Y11, y from Y1-1, z from Y10
L   = normalize(dom);  mag = |dom|             // mag < 1e-4 ⇒ no dominant light
directionality = clamp(mag / (1.73205 * lum(c0)), 0, 1)     // √3 for an ideal directional source
radiance(L)    = 0.282095*c0 + 0.488603*(c1*L.y + c2*L.z + c3*L.x), scaled by directionality
```

The sample is clamped to `wp.y ≥ min.y + spacing.y` (a full cell above the volume floor) because the bottom probe layer sits below all walkable ground by construction (§6) and its inverted L1 otherwise halves magnitude and directionality map-wide.

---

## 14. The direct/indirect split and the realtime light grid

`"direct": false` in `volume.json` means the practicals were **excluded** from the bake. The renderer reads it and auto-enables the realtime light path so the two never double-count (`viewer/src/render/gpu_driven.rs:3440-3469`). The shipped configuration is `--indirect-only`: baked soft indirect GI + crisp real-time direct practicals.

**Static CSR light grid** (`viewer/src/render/gpu_driven.rs:522-690`):

```
cell   = median(light ranges) clamped [4, 12] m, then raised so 256 cells/axis cover the extent
dims   = clamp(ceil(extent / cell), 1, 256) per axis; cell *= 1.5 while cells > 4_000_000
offsets: nCells+1 entries, base-included (base = nCells+1); cell i's lights = grid[grid[i] .. grid[i+1]]
```

A light is inserted into every cell its range-sphere AABB overlaps. A grid that stops short of the lights degenerates to the all-lights-in-one-edge-cell worst case.

**Runtime attenuation differs from the bake's on purpose** (`viewer/assets/shaders/gpu_draw.wgsl:566-569`):

```
runtime: win = saturate(1 - d²/r²);  atten = win² / max(d², 0.0625)   // r is the AUTHORED range
bake:    win = clamp(1 - (d/r)⁴, 0, 1);  at = win² / max(d², 0.25)    // r = max(range, 4.0)
```

The bake's floors exist because the probe grid is coarse; the runtime has per-pixel resolution and needs neither.

**Power groups.** Lights carry `group_idx` parallel to the packed light records. The default state is mask 0 (everything off); a toggle rewrites the light buffer zeroing the **colour lane** of every light whose group bit is clear, leaving positions and the CSR grid untouched (`viewer/src/render/gpu_driven.rs:6929-6963`).

**Directional occlusion without shadow maps.** Because the indirect SH is occlusion-aware, each realtime light is gated by `radiance(toward light) / ambient`, smoothstepped over `[0.12, 0.85]` and disabled when ambient < 1e-3 (`gpu_draw.wgsl:572-582`). A direction blocked by a wall reads dark, so leaking lights soften - at zero extra data.

---

## 15. Invariants and failure signatures

| Invariant | Failure signature when broken |
|---|---|
| `spacing = extent / (dims - 1)`, probe `i` at `min + i*spacing` | Every sample shifted by a fraction of a cell; GI slides relative to geometry, worst near walls. |
| Probe index `((z*ny)+y)*nx + x`, X fastest | Volume appears transposed; lighting from the wrong part of the map, often mirrored. |
| Coefficient order `(Y00, y, z, x)` | Dominant-light direction and bounce gather rotate by a coordinate swap; specular highlights point the wrong way. |
| Stored values are **radiance**; consumer applies `A0=π, A1=2π/3` | Applying the convolution twice ⇒ flat washed-out. Omitting it ⇒ too dark by `0.8862269/0.282095 = π ≈ 3.14x` on the L0/ambient band and `1.0233267/0.488603 = 2π/3 ≈ 2.09x` on L1 - the error is band-dependent, never a single uniform factor. |
| Sky/bounce scaled by `4π/n_rays`; delta lights **not** | Practicals scaled by ~0.049 ⇒ interiors go black (or, inverted, blow out). |
| `volume.bin` length ≥ `nx*ny*nz*24` | Undersized ⇒ loader rejects the whole volume; map falls back to flat ambient. Oversized loads and the tail is ignored. |
| `volume_valid.bin` length == `nx*ny*nz`, **from the same bake** | Validity mask applied to a grid it does not describe: valid probes masked out, buried probes weighted in. Light leaks through walls in a probe-aligned pattern. |
| `volume.bin`, `volume.json`, `volume_valid.bin` written together and referenced together in the manifest | A stale volume baked for older geometry - the grid's bounds no longer match the map. Interiors near-black or lit through moved walls. |
| Uvw is texel-center `((p-min)/spacing + 0.5)/dims` | Half-texel offset blends the inverted below-floor layer into every ground sample; dominant light halves in magnitude. |
| Switch-controlled lights excluded from the bake | A permanently-powered mall burned into the volume, and the power lever does nothing. |
| Relocated probe positions used by *every* pass | Probes rescued from inside walls are re-lit from inside the wall; on `--indirect-only` bakes the bounce (essentially the whole signal) is cast from inside solids. |
| 8-tap epsilon rides `tw` (`tw*(wn+1e-4)`) | Hard-edged axis-aligned rectangles appearing/disappearing at probe-cell boundaries; a cream block across window glass. |
| Estimator handover is `smoothstep`, not `if` | Probe-grid-aligned rectangles with a texel staircase along the threshold crossing. |
| Pass-B uniform sized 7 × vec4 = 112 bytes | A 96-byte pass-B uniform truncates `chunk`, dropping the probe batch offset: every batch after the first writes the wrong probes. |
| Uncaptured GPU errors log instead of panicking | A mid-batch device loss aborts the build stage instead of falling back to CPU. |
| Error scopes popped **before** mapping a `mapped_at_creation` buffer | Hard process crash on VRAM exhaustion instead of a clean CPU fallback. |
| All-zero output treated as a TDR | A watchdog reset ships a fully black volume that looks like a successful bake. |

---

## 16. Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `EFT_SH_RAYS` | 256 (min 8) | Pass-A Fibonacci ray count per probe. |
| `EFT_SKY` | 2.0 | Sky gradient scale. |
| `EFT_LIGHT_SCALE` | 6.0 | Unity `color*intensity` → SH radiance. Shared with the realtime grid. |
| `EFT_SH_INDIRECT` | 0 | `1` = indirect-only (`--indirect-only`): skip practicals, write `"direct": false`. |
| `EFT_SH_BOUNCE` | 1 | `0` disables pass B (`"bounces": 0`). |
| `EFT_SH_BOUNCE_RAYS` | 128 (1..4096) | Bounce ray count. |
| `EFT_SH_ALBEDO_BOOST` | 1.0 | Global multiplier on per-material albedo. |
| `EFT_SH_EMIS_GAIN` | 1.0 | Global multiplier on the emissive gather. |
| `EFT_SH_VO` | 1 | `0` disables virtual offset. |
| `EFT_SH_GPU_BOUNCE` | 0 | `1` runs pass B on the GPU (experimental). |
| `EFT_SH_GPU_BATCH_S` | 0.9 (0.3–1.4) | Target seconds per dispatch. |
| `EFT_SH_GPU_BATCH0` | derived from BVH depth | Override the first batch size. |
| `EFT_SH_GPU_CAP_MB` | adapter limit | Force a smaller per-binding cap to exercise multi-chunk on a small map. |
| `EFT_BAKE_CPU` | 0 | `1` forces the CPU backend everywhere (driver-crash escape hatch). |
| `EFT_GI` | sidecar `gi_intensity`, else 1.0 | Viewer-side GI multiplier override; non-finite or negative values fall back to 1.0 (`viewer/src/render/gpu_driven.rs:5276-5280`). |

CLI: `bake-sh <pack_dir> [--rays N] [--backend auto|gpu|cpu] [--indirect-only]` (`viewer/src/sh_bake.rs:952-1050`).