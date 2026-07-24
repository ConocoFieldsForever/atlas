// eft::gpu_draw â€” M2 GPU-driven indirect surface shader, M3 textured, M3b1 BLEND.
//
// The M2 counterpart of instancing_m0.wgsl. Instead of instance-rate vertex attributes,
// the per-instance affine is FETCHED from a GPU-resident storage buffer, indexed through
// the compacted `visible` list the compute cull produced:
//
//   real_instance = visible[@builtin(instance_index)]
//
// @builtin(instance_index) already ranges over [first_instance, first_instance+count)
// where first_instance = the mesh's instance_base, so `visible[instance_index]` reads the
// mesh's survivors directly (NO manual "first_instance + local" offset â€” that GL-ism would
// double-offset on wgpu).
//
// THE #1 RULE (tarkov-unity-extraction): apply the FULL ROW-MAJOR 3x4 affine (incl shear +
// mirror) to raw verts; transform normals by the COFACTOR matrix; double-sided via
// cull_mode=None + a front-facing flip. NEVER TRS-decompose. Identical vertex math to the M0
// shader â€” only the instance FETCH changed.
//
// M3 TEXTURES (bindless albedo + cutout + tint):
//   Materials vary PER SUBMESH, but a whole mesh draws in ONE multi_draw_indexed_indirect and
//   wgpu 0.17.3 exposes NO @builtin(draw_id). So the material id rides on the VERTEX: each vertex
//   is tagged (@location(3), Uint32) with its submesh's GLOBAL materialId (0..4408). Verified:
//   across all 1719 multi-submesh meshes the submeshes reference DISJOINT vertex sets, so every
//   vertex belongs to exactly one submesh â€” no boundary-vertex duplication is needed. The vertex
//   stage passes the id @interpolate(flat) to the fragment, which indexes the material table SSBO
//   -> a bindless albedo texture. flat is REQUIRED: a u32 cannot be perspective-interpolated.
//
// M3b1 BLEND (this file specializes into TWO pipelines off ONE source):
//   The material `flags` now also carry bit1 = MAT_FLAG_BLEND (role âˆˆ {decal,glass,water} or
//   alphaMode=BLEND, set in build_cpu_data). The Rust `specialize()` compiles this shader TWICE
//   with the same entry points, discriminated by the `BLEND_PASS` shader_def:
//     * OPAQUE build (no BLEND_PASS): discards BLEND materials, writes depth, outputs alpha 1.0.
//     * BLEND  build (BLEND_PASS defined): discards non-BLEND (opaque/cutout) materials, no depth
//       write, alpha-blends, outputs the REAL computed albedo.a (tex.a*tint.a, or tint.a
//       untextured). Both ride the ONE Transparent3d pass; the opaque item is enqueued at a large
//       negative distance so it runs FIRST and lays down depth the blend item tests against.
//   SHADER_DEF NAME: "BLEND_PASS" (Rust must push ShaderDefVal "BLEND_PASS" into the FragmentState
//   shader_defs iff key.blend_pass). The class-discard is LEXICALLY BEFORE the top-level
//   textureSample so naga uniformity is preserved (the sample stays in uniform control flow).
//
// group(0) = Bevy's mesh view bind group (reused via SetMeshViewBindGroup<0> so
// position_world_to_clip resolves). group(1) = our two instance/visible storage buffers.
// group(2) = M3 material table + bindless albedo array + sampler + Phase-2b bindless normal-map
// array (see Rust `material_layout`).

#import bevy_pbr::view_transformations::position_world_to_clip
// Phase 1.6 GGX spec: the camera world position (for the view vector V) rides on the Bevy
// mesh-view uniform at group(0) @binding(0) — already bound via SetMeshViewBindGroup<0>.
// `view.world_position.xyz` is the eye position. (view_transformations imports this same
// binding internally as `view_bindings`; we import the `view` symbol directly for the .xyz.)
#import bevy_pbr::mesh_view_bindings::view

struct InstanceGpu {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    ids: vec4<u32>,
    sphere: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceGpu>;
@group(1) @binding(1) var<storage, read> visible: array<u32>;

// M3 material table â€” one entry per materials.json material (id == index, 4409 entries).
// 160 bytes, 16-aligned (Phase 2b added the normal block @64; #6 added the detail block @80).
// Matches the Rust `MaterialGpu` POD byte-for-byte.
//   albedo_index : index into `albedo_tex`; 0xFFFFFFFF sentinel = untextured (use tint/white).
//   flags        : bit0 = cutout (role=cutout / alphaMode=MASK) -> discard when alpha < cutoff.
//                  bit1 = blend  (role âˆˆ {decal,glass,water} / alphaMode=BLEND) -> BLEND pass.
//                  bit2 = softcutout (Custom/Vert Paint SoftCutout Decal) -> feather via color.a.
//                  bit3 = water  (role=water) -> dark wet sheen, not the white tint fallback.
//   alpha_cutoff : PER-MATERIAL cutoff (NOT a global 0.5).
//   uv_xform     : (sx,sy,ox,oy) REFERENCE ONLY â€” tiling is already baked into the vertex UVs
//                  (manifest.conventions.uvTilingBaked=true). Do NOT apply it (double-tile trap).
//   tint         : linear rgba; rgb multiplies albedo. a is the BLEND-pass opacity (M3b1: used).
//   vp           : SoftCutout params [_AlphaStrength, _Cutoff, _AlphaHeight, 0] (M3b2). BLEND-pass
//                  coverage = clamp(color.a*vp.x - (vp.y - vp.z), 0, 1). Zeros for non-SoftCutout.
//   normal_index : index into `normal_tex`; 0xFFFFFFFF sentinel = no normal map (Phase 2b).
//   normal_flags : bit0 = green-flip (DirectX Y-down; negate sampled n.y).
//   normal_scale : tangent xy multiplier (default 1.0).
struct MaterialGpu {
    albedo_index: u32,
    flags: u32,
    alpha_cutoff: f32,
    roughness: f32,    // Phase 1.6: repurposed from _pad (offset 12) — GGX spec roughness,
                       // clamped [0.03,1.0] CPU-side (glass=0.05 -> sharp highlight).
    uv_xform: vec4<f32>,
    tint: vec4<f32>,
    vp: vec4<f32>,
    // Phase 2b normal mapping: 4th 16-byte block @64 (size -> 80). Byte-identical to Rust POD.
    normal_index: u32,
    normal_flags: u32,
    normal_scale: f32,
    _pad2: u32,
    // #6 Detail maps: 5 more 16-byte blocks @80 (size -> 160). Byte-identical to the Rust POD.
    //   detail_albedo_index / detail_normal_index : indices into the SAME albedo_tex / normal_tex
    //       bindless arrays as the base textures (the 2 shared detail PNGs are appended there).
    //   detail_flags : bit0 = has detail albedo, bit1 = has detail normal.
    //   detail_albedo_uv / detail_normal_uv : RAW _Detail*Map_ST (sx,sy,ox,oy); the fragment builds
    //       a RELATIVE transform against uv_xform (base _MainTex_ST is baked into the vertex UVs).
    //   detail_params : x=albedo strength, y=normal scale, z=fade start(8m), w=fade end(15m).
    //   detail_mean_gain : xyz = offline mean(sample_linear×4.5948) for the mean-neutralize; w=1.
    detail_albedo_index: u32,      // @80
    detail_normal_index: u32,      // @84
    detail_flags: u32,             // @88
    _detpad: u32,                  // @92
    detail_albedo_uv: vec4<f32>,   // @96
    detail_normal_uv: vec4<f32>,   // @112
    detail_params: vec4<f32>,      // @128
    detail_mean_gain: vec4<f32>,   // @144
    // Emissive (windows/monitors/signs/lamps): 16-byte block @160 (size -> 176). Declared as
    // FOUR SCALARS, not u32+vec3 (a vec3 would 16-align and silently desync from the Rust POD).
    emissive_index: u32,           // @160  bindless albedo_tex index, or MAT_EMISSIVE_NONE
    em_r: f32,                     // @164  linear rgb emissive = factor × hdr (CPU-precomputed)
    em_g: f32,                     // @168
    em_b: f32,                     // @172
    // Parallax (steep/occlusion) mapping @176 (size -> 192). Byte-identical to the Rust POD.
    parallax_index: u32,           // @176  bindless albedo_tex index of the grayscale HEIGHT map, or NONE
    parallax_scale: f32,           // @180  Unity _Parallax amount (max tangent-space UV offset)
    _ppad0: u32,                   // @184
    _ppad1: u32,                   // @188  -> 192
};

// Phase 1.6 GGX specular tuning: global multiplier on the dielectric spec lobe. 1.0 = physical.
// Dial down if broad interior highlights read too hot, up if wet floors/glass look flat.
const SPEC_STRENGTH: f32 = 1.5;
// Environment-reflection strength (M4): glossy / glass / water reflect the baked SH volume sampled
// in the mirror direction (fresnel + gloss weighted), giving the reflections + pop the flat baked
// diffuse alone can't. 0 = off (pure Phase-1.6 look). Matte surfaces are unaffected (gloss ~0).
const ENV_REFL_STRENGTH: f32 = 1.6;
// Analytic sky reflection (sky_reflect): a crisp cool gradient, scaled by the LOCAL SH exposure so
// it never exceeds how lit the spot is (indoor floors stay dark, outdoor surfaces pop). The sun
// glint is OWNED by the GGX lobe (dom light) — a second analytic sun disk here double-counted it
// and was ~14x the real sun's angular size.
const SKY_REFL_GAIN: f32 = 1.45;       // how much brighter than the local SH the sky reads (outdoor pop)

// --- Distance fog / aerial perspective ---------------------------------------
// The single biggest "real EFT" cue outdoors: overcast Baltic haze milks out geometry with
// distance toward the horizon color. Exponential-squared in distance, faded DOWN indoors via the
// SH directionality (isotropic probe = interior = little haze) with a floor (overcast outdoor
// directionality is itself low — a hard gate would kill fog on the terrain it targets), and faded
// UP slightly with view height drop so low coastal sightlines haze harder than top-down views.
const FOG_DENSITY: f32 = 0.00075;              // ~17% haze at 600 m, ~50% at 1.2 km
// Haze color: sits BETWEEN scene radiance (~0.2-0.4 pre-tonemap) and the sky horizon (~0.6) —
// matching the horizon exactly overpowered mid-distance geometry (bright milk veil at 300 m).
const FOG_COLOR: vec3<f32> = vec3<f32>(0.44, 0.49, 0.58);
const FOG_INDOOR_FLOOR: f32 = 0.2;             // fraction of fog that survives indoors

const MAT_ALBEDO_NONE: u32 = 0xFFFFFFFFu; // sentinel: material has no albedo texture
const MAT_NORMAL_NONE: u32 = 0xFFFFFFFFu;  // sentinel: material has no normal map (Phase 2b)
const MAT_EMISSIVE_NONE: u32 = 0xFFFFFFFFu; // sentinel: material has no emissive texture
const MAT_FLAG_CUTOUT: u32 = 1u;          // bit0: alpha-tested (MASK) surface
const MAT_FLAG_BLEND: u32 = 2u;           // bit1: alpha-blended (decal/glass/water/BLEND) surface
const MAT_FLAG_SOFTCUTOUT: u32 = 4u;      // bit2: Vert Paint SoftCutout decal (feather via color.a)
const MAT_FLAG_WATER: u32 = 8u;           // bit3: water/mirror (dark wet sheen, not white fallback)
const MAT_FLAG_TERRAIN: u32 = 16u;        // bit4: MicroSplat terrain (splat-blend 12 layers; slice in _pad2)
const MAT_FLAG_DETAIL: u32 = 32u;         // bit5: #6 detail maps (secondary albedo/normal; never on terrain)
const MAT_FLAG_RFA: u32 = 64u;            // bit6: per-pixel roughness = 1 - RAW tex.a (smoothness-in-alpha)
const MAT_FLAG_VP: u32 = 128u;            // bit7: vert-paint 3-layer splat (VpGpu at _pad2)
const MAT_FLAG_PARALLAX: u32 = 2048u;     // bit11: steep parallax mapping (offset UV by parallax_index height)
const MAT_FLAG_PUDDLE_LUMA: u32 = 256u;   // bit8: puddle shape mask in luma(rgb), not alpha (atlas)
const MAT_FLAG_WATER_MATTE: u32 = 512u;   // bit9: STRETCHED floor water-decal (tire marks / wet-ground) -> matte, no mirror
const MAT_FLAG_DECAL: u32 = 1024u;        // bit10: plain surface decal; mask ALL lighting terms by texture coverage
const DETAIL_HAS_ALBEDO: u32 = 1u;        // detail_flags bit0: has detail albedo texture
const DETAIL_HAS_NORMAL: u32 = 2u;        // detail_flags bit1: has detail normal texture
const DETAIL_UNITY_GAIN: f32 = 4.5948;    // Unity Standard detail ×2 expressed in linear space

@group(2) @binding(0) var<storage, read> materials: array<MaterialGpu>;
@group(2) @binding(1) var albedo_tex: binding_array<texture_2d<f32>>;
@group(2) @binding(2) var albedo_samp: sampler;
// Phase 2b: bindless normal-map array (LINEAR data). Reuses `albedo_samp`; indexed non-uniformly
// by m.normal_index (same device feature as albedo). Sentinel MAT_NORMAL_NONE = no normal map.
@group(2) @binding(3) var normal_tex: binding_array<texture_2d<f32>>;

// #1 MicroSplat terrain splat table (byte-identical to Rust `TerrainSplatGpu`). All indices are
// into the SAME `albedo_tex` array. Layer i weight = control map (i/4) channel (i%4);
// layer_uv = terrainUV01 * layer_rep[i]. ctrl_idx is 4 slices × 3 maps at [slice*3 + k].
struct TerrainSplat {
    layer_albedo: array<u32, 12>,
    layer_rep: array<f32, 12>,
    ctrl_idx: array<u32, 48>,   // up to 16 slices × 3 maps at [slice*3 + k] (slice names from sidecar)
};
@group(2) @binding(4) var<storage, read> terrain_splat: TerrainSplat;

// Vert-Paint 3-layer splat table (byte-identical to Rust `VpGpu`, 112 B). One entry per
// MAT_FLAG_VP material, indexed by `m._pad2`. The EXACT game blend (RE'd from the DX11
// fragment, validated in the web viewer's _vpsplat.js):
//   w_i = pow(Heights_i(raw_uv) * COLOR_0_i, blend), normalized;
//   albedo = Σ w_i · layer_i(uv_i) · tint_i.
// Layer 0's ST is baked into the mesh UVs (uvTilingBaked), so raw_uv = (uv - uv0.zw)/uv0.xy;
// the heights mask samples at raw_uv, layer i at raw_uv*uvi.xy+uvi.zw (layer 0 round-trips).
struct VpGpu {
    tex: vec4<u32>,     // x,y,z = layer albedo indices; w = heights index or MAT_ALBEDO_NONE
    uv0: vec4<f32>,     // raw per-layer _MainTex_ST (sx,sy,ox,oy)
    uv1: vec4<f32>,
    uv2: vec4<f32>,
    tint0: vec4<f32>,   // rgb tint; tint0.w = heights blend sharpness
    tint1: vec4<f32>,
    tint2: vec4<f32>,
};
@group(2) @binding(5) var<storage, read> vp_table: array<VpGpu>;

// --- Phase 1: baked SH-GI irradiance volume (group 3) ------------------------
// Replaces the flat `ambient + N·L` hack with the RTX-baked spherical-harmonics
// irradiance volume that ALREADY integrates the full scene lighting (neutral sky +
// warm sun + 1285 live practical lights + 1 diffuse bounce). Matching Unity =
// sampling this volume per-fragment for diffuse GI — NO per-light buffer is needed.
//
// Three RGBA16Float 3D textures, ONE PER COLOR CHANNEL; each texel = (c0,c1,c2,c3),
// the L1 RADIANCE-SH coeffs for that channel at that probe. Hardware trilinear
// interpolates each SH coeff across probes for free (correct: SH interpolates
// linearly). `sh.vol_min.xyz` = world-space min corner of the probe AABB, `.w` =
// gi_intensity; `sh.vol_inv_extent.xyz` = 1/(max-min) to map world_pos -> [0,1] uvw.
struct ShVolume {
    vol_min: vec4<f32>,        // xyz = world min, w = gi_intensity
    vol_inv_extent: vec4<f32>, // xyz = 1/(max-min), w = normal_bias (meters)
    dims: vec4<f32>,           // xyz = (nx, ny, nz) probe grid dims, w unused
    spacing: vec4<f32>,        // xyz = (sx, sy, sz) probe spacing (meters), w unused
};
@group(3) @binding(0) var<uniform> sh: ShVolume;
@group(3) @binding(1) var sh_r: texture_3d<f32>;
@group(3) @binding(2) var sh_g: texture_3d<f32>;
@group(3) @binding(3) var sh_b: texture_3d<f32>;
@group(3) @binding(4) var sh_samp: sampler;

// --- #5 Dynamic sun shadows (added to the EXISTING SH/lighting group 3; NOT a 5th bind group) ----
// A 2-cascade near-field contact CSM. The SH volume above already bakes the BROAD sun shadow; these
// two near cascades only add the missing high-frequency contact edge, and the combination below is
// gated + capped so it can only SUBTRACT a small, bounded amount of light (anti double-darkening).
// Byte-identical to the Rust `SunShadowUniform` (208 bytes).
struct SunShadowUniform {
    view_proj: array<mat4x4<f32>, 2>, // per-cascade world->light-clip (0..1 depth ortho)
    split_depths: vec4<f32>,          // x=far0(15) y=far1(80) z=overlap(0.10) w=enabled(1/0)
    sun_dir_texel: vec4<f32>,         // xyz=Lsun (toward sun), w=1/shadow_map_size (PCF texel)
    texel_world: vec4<f32>,           // x=cascade0 world texel, y=cascade1 world texel (bias units)
    combine: vec4<f32>,               // x=diffuse cap(0.12) y=fade start(65) z=fade end(80) w=debug
    // Runtime graphics scales from the UI (all 1.0 = shipped look):
    // x = fog density scale (0 = fog off), y = sky-reflection gain scale, z = emissive scale.
    gfx: vec4<f32>,
};
@group(3) @binding(5) var<uniform> sun: SunShadowUniform;
@group(3) @binding(6) var shadow_map: texture_depth_2d_array;
@group(3) @binding(7) var shadow_cmp: sampler_comparison;

// --- REALTIME point/spot lights (added to lighting group 3; NOT a 5th bind group) ----------------
// EFT lights its maps with realtime lights; the pack ships the raw set. A static world CSR grid
// (built once on the CPU) buckets lights into cells; each fragment loops only the handful whose
// range-sphere covers its cell. `params.z` (rt_enabled) is auto-selected on the CPU vs the baked SH
// volume so the two never double-count. Byte-identical to the Rust `LightGridUniform` (48 bytes).
struct LightGrid {
    grid_min: vec4<f32>,   // xyz = grid world-min corner, w = cell size (meters)
    grid_dims: vec4<u32>,  // xyz = grid dims, w = n_lights (0 => skip the loop)
    params: vec4<f32>,     // x = light_scale, y = ambient_scale, z = rt_enabled (1/0), w unused
};
@group(3) @binding(8) var<uniform> lgrid: LightGrid;
// 3 vec4 per light: v0=(pos.xyz,range) v1=(color.rgb,cos_outer) v2=(dir.xyz,cos_inner).
@group(3) @binding(9) var<storage, read> lights: array<vec4<f32>>;
// CSR: [0..=nCells] offsets (base-included) then concatenated per-cell light indices.
@group(3) @binding(10) var<storage, read> light_grid: array<u32>;

// Reconstruct diffuse IRRADIANCE (÷π folded in: cosine-convolved A0=π, A1=2π/3; the
// π cancels the Lambert 1/π) from the L1 radiance SH at `world_pos`, for surface
// normal `n`. Per channel: E/π = 0.282095*c0 + 0.325735*(c1*n.y + c2*n.z + c3*n.x).
//
// FALLBACK ONLY (sh_irradiance_hw): the ORIGINAL hardware-trilinear reconstruction.
// textureSampleLevel blindly averages all 8 corner probes — INCLUDING probes flood-
// filled "inside/below solid" (dark). On flat concrete their darkness bled up as ~3 m
// banded shadows (the probe spacing). Kept as the enclosed-point fallback for the new
// hemisphere-weighted manual tap below (so a fully-enclosed sample can't go black).
// Uses textureSampleLevel (explicit LOD 0, no mips) so it needs NO derivatives and
// can be called anywhere — even after a `discard` — avoiding naga's uniformity
// constraint (unlike the albedo textureSample, which must stay in uniform flow).
// OUT-OF-VOLUME redirect (universal): the probe grid can under-cover big open maps — its bounds
// are tri-density-derived and collapse around dense hubs (icebreaker's SHIP shrank the grid to a
// 222x480 m box under a 600x700 m ice field). Beyond the AABB the samplers used to CLAMP, smearing
// whatever the nearest EDGE probe held (ship-shadowed interior columns included) into razor-straight
// infinite bands ("giant shadow across the ice"; same on lighthouse). In the game those areas are
// open-sky-lit — so out-of-volume reads slide the sample point to the volume's own TOP LAYER above
// the nearest edge: real open-sky probes, giving the exact ambient AND sun direction/strength the
// open ground inside gets. Fades in over ~2 cells; inside samples are untouched. No per-map data.
// Texel-center uvw for the HARDWARE-sampled SH reads. Probes sit at world = min + i*spacing and a
// 3D texture's texel CENTERS sit at (i+0.5)/N — the old align-corners mapping ((p-min)/extent) was
// off by up to half a texel, so a ground-height sample blended ~40% of the below-floor probe layer
// (whose L1 is inverted) into the dominant light even when aimed exactly at layer 1. The manual
// 8-tap (textureLoad, integer texels) never suffered this — which is why diffuse GI matched across
// the volume boundary but the sun/dominant terms did not.
fn sh_uvw(p: vec3<f32>) -> vec3<f32> {
    return ((p - sh.vol_min.xyz) / sh.spacing.xyz + vec3<f32>(0.5)) / sh.dims.xyz;
}

fn sh_outside_t(p: vec3<f32>) -> f32 {
    let ext = vec3<f32>(1.0) / max(sh.vol_inv_extent.xyz, vec3<f32>(1e-9));
    let lo = sh.vol_min.xyz;
    let hi = lo + ext;
    let d = max(max(lo - p, p - hi), vec3<f32>(0.0));
    let margin = 2.0 * max(sh.spacing.x, max(sh.spacing.y, sh.spacing.z));
    return smoothstep(0.0, max(margin, 1e-3), max(d.x, max(d.y, d.z)));
}
fn sh_effective_pos(p: vec3<f32>) -> vec3<f32> {
    let ext = vec3<f32>(1.0) / max(sh.vol_inv_extent.xyz, vec3<f32>(1e-9));
    let lo = sh.vol_min.xyz;
    let hi = lo + ext;
    let edge = clamp(p, lo, hi);
    let sky = vec3<f32>(edge.x, hi.y - 0.5 * sh.spacing.y, edge.z); // top-layer probe row = open sky
    return mix(p, sky, sh_outside_t(p));
}

fn sh_irradiance_hw(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let wp = sh_effective_pos(world_pos);
    let uvw = sh_uvw(wp);
    let cr = textureSampleLevel(sh_r, sh_samp, uvw, 0.0); // (c0,c1,c2,c3) for R
    let cg = textureSampleLevel(sh_g, sh_samp, uvw, 0.0);
    let cb = textureSampleLevel(sh_b, sh_samp, uvw, 0.0);
    let w = vec3<f32>(n.y, n.z, n.x) * 0.325735;          // weights for c1(∝y),c2(∝z),c3(∝x)
    let e = vec3<f32>(
        0.282095 * cr.x + cr.y * w.x + cr.z * w.y + cr.w * w.z,
        0.282095 * cg.x + cg.y * w.x + cg.z * w.y + cg.w * w.z,
        0.282095 * cb.x + cb.y * w.x + cb.z * w.y + cb.w * w.z);
    // redirected (out-of-volume) samples scaled from top-of-dome to ground-equivalent sky
    return max(e, vec3<f32>(0.0)) * mix(1.0, sh.dims.w, sh_outside_t(world_pos));
}

// MANUAL 8-tap irradiance-volume sample (irradiance-volume leak fix). Replaces the
// hardware trilinear above. It (a) trilinear-weights the 8 corner probes, (b) REJECTS
// probes that don't sit in the shaded surface's hemisphere via a normal-direction
// weight `wn` (~0 for probes below the slab / behind the surface — the ones that were
// leaking dark bands), and (c) offsets the sample point along the normal (normal-bias)
// so the shading point pulls away from the solid it sits on. `textureLoad` on a
// texture_3d fetches an exact probe texel (no filtering, sampler ignored) so no
// derivatives are required — still callable after a `discard`.
fn sh_irradiance(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let wp = sh_effective_pos(world_pos);
    let bias = sh.vol_inv_extent.w;
    let sp   = wp + n * bias;
    let dims = sh.dims.xyz;
    let spacing = sh.spacing.xyz;
    let grid = clamp((sp - sh.vol_min.xyz) / spacing, vec3<f32>(0.0), dims - vec3<f32>(1.0)); // continuous grid coords
    let base = floor(grid);
    let f    = grid - base;
    var sum  = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let o  = vec3<f32>(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u));
        let pc = clamp(base + o, vec3<f32>(0.0), dims - vec3<f32>(1.0));
        let ipc = vec3<i32>(pc);
        let probe_pos = sh.vol_min.xyz + pc * spacing;
        let tw3 = mix(vec3<f32>(1.0) - f, f, o);      // per-axis trilinear weight
        let tw  = tw3.x * tw3.y * tw3.z;
        let dir = probe_pos - wp;
        let wn  = max(dot(normalize(dir + n * 1e-3), n), 0.0);  // ~0 for below-slab / back-facing probes (the leak)
        let w   = tw * wn + 1e-4;
        let cr = textureLoad(sh_r, ipc, 0);
        let cg = textureLoad(sh_g, ipc, 0);
        let cb = textureLoad(sh_b, ipc, 0);
        let ww = vec3<f32>(n.y, n.z, n.x) * 0.325735;  // weights for c1,c2,c3
        let e  = vec3<f32>(
            0.282095 * cr.x + cr.y * ww.x + cr.z * ww.y + cr.w * ww.z,
            0.282095 * cg.x + cg.y * ww.x + cg.z * ww.y + cg.w * ww.z,
            0.282095 * cb.x + cb.y * ww.x + cb.z * ww.y + cb.w * ww.z);
        sum  = sum + w * max(e, vec3<f32>(0.0));
        wsum = wsum + w;
    }
    // Fallback: if the hemisphere weighting rejected essentially everything (fully
    // enclosed point), don't go black — use the plain hardware-trilinear reconstruction.
    if (wsum < 1e-3) { return sh_irradiance_hw(wp, n); }
    return (sum / wsum) * mix(1.0, sh.dims.w, sh_outside_t(world_pos));
}

// --- Phase 1.6: dominant light direction + radiance from the SH volume -------
// GGX spec needs a directional light. We DERIVE it from the SAME baked SH volume the
// diffuse samples, so the highlight stays consistent with the baked lighting (no new
// data). The L1 band of a radiance-SH encodes the linear (directional) part of the
// incident radiance; its luminance-weighted direction is the "sun-ish" dominant light,
// and reconstructing the radiance in that direction gives its color/intensity.
//
// Sampled via textureSampleLevel (explicit LOD 0, no derivatives) — like sh_irradiance_hw —
// so it is safe to call after the class-discard (naga uniformity preserved).
struct DomLight {
    dir: vec3<f32>,      // normalized dominant light direction L
    radiance: vec3<f32>, // SH radiance reconstructed toward L (>= 0), the light's rgb
    mag: f32,            // dominant magnitude; < 1e-4 => no dominant light (skip spec)
    directionality: f32, // #5: L1/L0 ratio, normalized to [0,1]. ~1 => crisp directional (sun-lit),
                         // ~0 => flat/isotropic (already-baked-shadow). Gates the shadow term so we
                         // don't re-darken places the SH volume already shadowed (double-darkening).
};
fn sh_dominant_light(world_pos: vec3<f32>) -> DomLight {
    var wp = sh_effective_pos(world_pos);
    // The grid's BOTTOM probe layer sits below all walkable ground BY CONSTRUCTION (the baker's
    // y-band starts at the 0.5-percentile minus 2 m), and those below-surface probes carry an
    // INVERTED L1 that plain trilinear mixes into every ground-level sample — halving the dominant
    // light's magnitude and directionality map-wide (icebreaker open ice: mag 0.95/dir 0.20 as-is
    // vs 1.85/0.31 one layer up, matching the top layer's 1.84/0.30). The irradiance 8-tap already
    // rejects these probes via its hemisphere weight; give the dominant light the same protection
    // by clamping its sample a full cell above the volume floor.
    wp.y = max(wp.y, sh.vol_min.y + sh.spacing.y);
    let uvw = sh_uvw(wp);
    let cr = textureSampleLevel(sh_r, sh_samp, uvw, 0.0); // (c0,c1,c2,c3) for R
    let cg = textureSampleLevel(sh_g, sh_samp, uvw, 0.0);
    let cb = textureSampleLevel(sh_b, sh_samp, uvw, 0.0);
    // luminance-weight each directional coeff across channels (Rec.709 luma).
    let lw  = vec3<f32>(0.2126, 0.7152, 0.0722);
    let lc1 = dot(vec3<f32>(cr.y, cg.y, cb.y), lw); // coeff1 = Y1-1 (∝ y)
    let lc2 = dot(vec3<f32>(cr.z, cg.z, cb.z), lw); // coeff2 = Y10  (∝ z)
    let lc3 = dot(vec3<f32>(cr.w, cg.w, cb.w), lw); // coeff3 = Y11  (∝ x)
    var out: DomLight;
    let dom  = vec3<f32>(lc3, lc1, lc2); // x from Y11, y from Y1-1, z from Y10
    let dmag = length(dom);
    out.mag  = dmag;
    // #5: directionality = |L1| / (sqrt(3)*L0). For an ideal directional source the L1/L0 ratio
    // approaches sqrt(3); diffuse/isotropic lighting approaches 0. l0 is the luminance of the
    // constant (ambient) band. Used by the shadow gate so flat-lit (already baked-shadow) points
    // receive little or no further attenuation.
    let l0 = max(dot(vec3<f32>(cr.x, cg.x, cb.x), lw), 1e-4);
    out.directionality = clamp(dmag / (1.73205 * l0), 0.0, 1.0);
    if (dmag < 1e-4) {
        out.dir = vec3<f32>(0.0, 1.0, 0.0); // sentinel; caller skips on mag
        out.radiance = vec3<f32>(0.0);
        return out;
    }
    let L = dom / dmag;
    out.dir = L;
    // radiance toward L per channel: 0.282095*c0 + 0.488603*(c1*L.y + c2*L.z + c3*L.x).
    let rr = 0.282095 * cr.x + 0.488603 * (cr.y * L.y + cr.z * L.z + cr.w * L.x);
    let rg = 0.282095 * cg.x + 0.488603 * (cg.y * L.y + cg.z * L.z + cg.w * L.x);
    let rb = 0.282095 * cb.x + 0.488603 * (cb.y * L.y + cb.z * L.z + cb.w * L.x);
    // Scale by directionality: the raw reconstruction includes the L0 AMBIENT band, which
    // inflated the GGX "sun" ~6.8x in flat-lit (isotropic) probes — indoor floors got sun
    // glints from what is actually ambient. Genuinely sun-lit areas (directionality -> 1)
    // keep their current tuning; isotropic probes smoothly lose the phantom highlight.
    // NOTE: no dims.w scale here — the ground/top ratio applies to the AMBIENT band only
    // (measured: c0 layer1/top ~0.9, but dominant-band radiance layer1/top ~1.0).
    out.radiance = max(vec3<f32>(rr, rg, rb), vec3<f32>(0.0)) * out.directionality;
    return out;
}

// --- REALTIME point/spot light evaluation ------------------------------------
// Loops the lights bucketed into this fragment's grid cell, accumulating Lambert diffuse (÷π folded,
// so it drops straight into the SH `gi` irradiance term) and GGX/Cook-Torrance specular using the
// SAME dielectric BRDF the SH-dominant path uses. Reads only storage buffers + arithmetic (NO
// derivatives / textureSample), so it is uniformity-safe to call anywhere — even after a discard.
struct RtLight {
    diffuse: vec3<f32>, // irradiance/π (pre-albedo), add into the SH `gi` term
    spec: vec3<f32>,    // GGX specular radiance, add into spec_rgb
};
fn eval_realtime_lights(world_pos: vec3<f32>, N: vec3<f32>, V: vec3<f32>, rough: f32) -> RtLight {
    var acc_d = vec3<f32>(0.0);
    var acc_s = vec3<f32>(0.0);
    if (lgrid.params.z > 0.5 && lgrid.grid_dims.w > 0u) {
        let dims = lgrid.grid_dims.xyz;
        let cellf = clamp(floor((world_pos - lgrid.grid_min.xyz) / lgrid.grid_min.w),
                          vec3<f32>(0.0), vec3<f32>(dims) - vec3<f32>(1.0));
        let cell = vec3<u32>(cellf);
        let ci = (cell.z * dims.y + cell.y) * dims.x + cell.x;
        let s = light_grid[ci];
        let e = light_grid[ci + 1u];
        let NdotV = max(dot(N, V), 1e-3);
        let a  = rough * rough;
        let a2 = a * a;
        let sk = (rough + 1.0) * (rough + 1.0) / 8.0; // Smith k (direct lighting)
        let scale = lgrid.params.x;
        // SH-directional occlusion for the realtime lights (cross-platform, ~free — reuses the baked
        // INDIRECT SH). After the direct/indirect split the SH carries sky + bounce ONLY (practicals
        // are realtime), and it is occlusion-aware, so a direction blocked by a wall reads DARK. Gating
        // each light by radiance(toward-light)/ambient softly attenuates lights that leak from behind
        // geometry, with NO per-light shadow map. amb~0 (no volume / fully-black probe) -> disabled.
        let occ_uvw = clamp(sh_uvw(sh_effective_pos(world_pos)), vec3<f32>(0.0), vec3<f32>(1.0));
        let occ_cr = textureSampleLevel(sh_r, sh_samp, occ_uvw, 0.0);
        let occ_cg = textureSampleLevel(sh_g, sh_samp, occ_uvw, 0.0);
        let occ_cb = textureSampleLevel(sh_b, sh_samp, occ_uvw, 0.0);
        let OCC_LUMA = vec3<f32>(0.299, 0.587, 0.114);
        let occ_amb = 0.282095 * dot(vec3<f32>(occ_cr.x, occ_cg.x, occ_cb.x), OCC_LUMA);
        for (var k = s; k < e; k = k + 1u) {
            let li = light_grid[k];
            let L0 = lights[li * 3u];
            let L1 = lights[li * 3u + 1u];
            let L2 = lights[li * 3u + 2u];
            let Lp = L0.xyz; let range = L0.w;
            let lcol = L1.rgb; let cos_outer = L1.w;
            let ldir = L2.xyz; let cos_inner = L2.w;
            let Lv = Lp - world_pos;
            let d2 = dot(Lv, Lv);
            if (d2 >= range * range) { continue; }
            let d = sqrt(max(d2, 1e-8));
            let l = Lv / d;
            let win = saturate(1.0 - d2 / (range * range));
            let atten = win * win / max(d2, 0.0625);          // (1-(d/r)^2)^2 / d^2, near-singularity capped
            let cosang = dot(-l, ldir);
            let spot = smoothstep(cos_outer, cos_inner, cosang); // point: cos_outer=-2,cos_inner=-1 => 1
            let ndl = max(dot(N, l), 0.0);
            if (ndl <= 0.0 || spot <= 0.0) { continue; }
            // Directional occlusion from the indirect SH: reconstruct radiance TOWARD the light (`l`)
            // and compare to the isotropic ambient. Ratio ~1 in an open/uniform direction, <1 toward a
            // walled-off one -> the light dims. Folded into `radiance` so it gates diffuse AND spec.
            let occ_rl = 0.282095 * vec3<f32>(occ_cr.x, occ_cg.x, occ_cb.x)
                + 0.488603 * (vec3<f32>(occ_cr.y, occ_cg.y, occ_cb.y) * l.y
                            + vec3<f32>(occ_cr.z, occ_cg.z, occ_cb.z) * l.z
                            + vec3<f32>(occ_cr.w, occ_cg.w, occ_cb.w) * l.x);
            let occ = select(1.0,
                smoothstep(0.12, 0.85, dot(max(occ_rl, vec3<f32>(0.0)), OCC_LUMA) / (occ_amb + 1e-4)),
                occ_amb > 1.0e-3);
            let radiance = lcol * (scale * atten * spot * occ);
            acc_d = acc_d + radiance * ndl;
            // GGX/Cook-Torrance specular — identical BRDF to the SH-dominant lobe (dielectric F0=0.04).
            let H = normalize(V + l);
            let NdotH = max(dot(N, H), 0.0);
            let VdotH = max(dot(V, H), 0.0);
            let dd = (NdotH * NdotH * (a2 - 1.0) + 1.0);
            let D  = a2 / (3.14159265 * dd * dd);
            let gv = NdotV / (NdotV * (1.0 - sk) + sk);
            let gl = ndl / (ndl * (1.0 - sk) + sk);
            let G  = gv * gl;
            let F  = 0.04 + 0.96 * pow(1.0 - VdotH, 5.0);
            acc_s = acc_s + radiance * (D * G * F / (4.0 * NdotV * ndl + 1e-4)) * ndl * SPEC_STRENGTH;
        }
    }
    return RtLight(acc_d * 0.31830989, acc_s); // 1/π on the diffuse (matches the SH ÷π convention)
}

// --- Analytic sky environment for glossy reflections -------------------------
// The baked SH volume in the mirror direction reads as a dull grey blob, so glass / water reflecting
// it look flat. Real glossy surfaces reflect the SKY, which is crisper (a directional gradient) and,
// outdoors, brighter. We synthesize a cool vertical gradient (brighter toward the zenith); the SUN
// glint is owned solely by the GGX lobe (which is properly shadow-gated) — a second analytic disk
// here double-counted the sun at ~14x its real angular size. CRITICAL: the gradient is scaled by
// `level` — the LOCAL exposure (luma of the SH reflection) — so it can NEVER exceed how lit the spot
// actually is. Indoors (dark SH probe) the "sky" stays dark; only sky-exposed outdoor surfaces get
// the bright reflection. This is what keeps interior glossy floors from blowing out.
// f32-safe sine for world-space wave phases: sin() of a raw world-scaled coordinate reaches
// ±1200 rad at map edges, where GPU fast-sin collapses into STRUCTURED PRECISION NOISE — this
// was the sea's screen-space checkerboard AND the km-scale dark "shadow streak" beat bands
// (survived every shading A/B; pinned by zeroing the ripple). fract() first keeps the argument
// in the accurate [0,2π) domain; the input is in CYCLES (phase/2π).
fn rsin(phase_cycles: f32) -> f32 {
    return sin(fract(phase_cycles) * 6.2831853);
}

fn sky_reflect(R: vec3<f32>, level: f32) -> vec3<f32> {
    let up = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.66, 0.72, 0.82);
    let zenith  = vec3<f32>(0.92, 0.98, 1.10);
    // anchored to local exposure; sun.gfx.y = runtime UI gain scale (1 = shipped look)
    return mix(horizon, zenith, up * up) * (level * SKY_REFL_GAIN * sun.gfx.y);
}

// --- Distance fog / aerial perspective ----------------------------------------
// Exp² haze toward the overcast horizon color — the strongest single "real EFT" cue outdoors
// (Lighthouse sightlines run 1-2 km; razor-sharp distant terrain is the tell of a renderer).
// Gated DOWN indoors via the SH directionality (an isotropic probe = interior) with a floor:
// overcast outdoor directionality is itself low, so a hard zero-gate would kill fog on the very
// terrain it targets. rgb-only (alpha untouched) — correct under non-premultiplied ALPHA_BLENDING,
// since whatever is BEHIND a transparent was already fogged.
fn apply_fog(rgb: vec3<f32>, world_pos: vec3<f32>, directionality: f32) -> vec3<f32> {
    let d = distance(view.world_position.xyz, world_pos);
    let dens = FOG_DENSITY * sun.gfx.x; // runtime density scale (0 = fog off, 1 = shipped)
    let f = 1.0 - exp(-(d * dens) * (d * dens));
    let gate = mix(FOG_INDOOR_FLOOR, 1.0, smoothstep(0.03, 0.20, directionality));
    return mix(rgb, FOG_COLOR, f * gate);
}

// --- #5 sun-shadow PCF sampling ----------------------------------------------
// Sample one cascade `c` for the receiver point `p` with GEOMETRIC normal `Ng`. Returns visibility
// in [0,1] (1 = fully lit, 0 = fully shadowed). textureSampleCompareLevel samples at LOD 0 (no
// derivatives) so this is safe in the non-uniform control flow of the cascade select below.
fn sample_cascade(p: vec3<f32>, Ng: vec3<f32>, c: u32) -> f32 {
    let world_texel = sun.texel_world[c];

    // Receiver-plane offset (acne fix): push along the GEOMETRIC normal (normal maps must NOT wobble
    // the receiver offset) plus a small nudge toward the sun.
    let offset_p =
        p
        + Ng * (1.5 * world_texel)
        + sun.sun_dir_texel.xyz * (0.25 * world_texel);

    let q = sun.view_proj[c] * vec4<f32>(offset_p, 1.0);
    let ndc = q.xyz / q.w;

    // Outside this cascade's frustum -> treat as lit (the caller's cascade select / far fade owns
    // the transition). ndc.z is the conventional 0..1 light-space depth.
    if (any(ndc.xy < vec2<f32>(-1.0)) || any(ndc.xy > vec2<f32>(1.0))
        || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let dt = sun.sun_dir_texel.w; // 1/shadow_map_size

    var sum = 0.0;
    // Weighted 3x3 tent [1,2,1]^2, total weight 16.
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let wx = select(1.0, 2.0, x == 0);
            let wy = select(1.0, 2.0, y == 0);
            sum += wx * wy * textureSampleCompareLevel(
                shadow_map,
                shadow_cmp,
                uv + vec2<f32>(f32(x), f32(y)) * dt,
                i32(c),
                ndc.z
            );
        }
    }
    return sum / 16.0;
}

// Cascade select by VIEW-SPACE depth with a short blend across the overlap, then fade the whole
// effect to fully lit over the far contact range. Returns visibility in [0,1].
fn sun_shadow_visibility(p: vec3<f32>, Ng: vec3<f32>, view_depth: f32) -> f32 {
    // Blend cascade 0 -> 1 across 13.5..15 m (just inside the split, per the 10% overlap fit).
    if (view_depth < 13.5) {
        return sample_cascade(p, Ng, 0u);
    } else if (view_depth < 15.0) {
        let v0 = sample_cascade(p, Ng, 0u);
        let v1 = sample_cascade(p, Ng, 1u);
        return mix(v0, v1, (view_depth - 13.5) / (15.0 - 13.5));
    }
    return sample_cascade(p, Ng, 1u);
}

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_index: u32, // Uint32; per-submesh global materialId, tagged in build_cpu_data
    @location(4) color: vec4<f32>,    // M3b2: per-vertex COLOR_0 vert-paint weight (SoftCutout coverage on .a)
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) material_index: u32,
    @location(3) color: vec4<f32>, // interpolated (NOT flat): SoftCutout feathers across the tri
    @location(4) world_pos: vec3<f32>, // Phase1 SH-GI: world position -> irradiance-volume uvw
};

// cofactor(linear 3x3) = det Â· inverse-transpose; columns = cross products of the linear
// columns. Correct normal transform under shear / non-uniform scale / mirror, no decompose.
fn cofactor(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(cross(c1, c2), cross(c2, c0), cross(c0, c1));
}

// Phase 2b screen-space (cotangent-frame derivative) TBN. There are NO stored vertex tangents,
// so the tangent basis is reconstructed per-fragment from the position + uv derivatives. MUST be
// called in UNIFORM control flow (dpdx/dpdy require it — same naga rule as the albedo sample).
// `N` = geometric (back-face-flipped) world normal; `p` = world position; `uv` = surface uv;
// `n_ts` = tangent-space normal already unpacked to [-1,1] (with scale + green-flip applied).
fn perturb_normal(N: vec3<f32>, p: vec3<f32>, uv: vec2<f32>, n_ts: vec3<f32>) -> vec3<f32> {
    let dp1 = dpdx(p); let dp2 = dpdy(p);
    let duv1 = dpdx(uv); let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, N); let dp1perp = cross(N, dp1);
    let T = dp2perp * duv1.x + dp1perp * duv2.x;
    let B = dp2perp * duv1.y + dp1perp * duv2.y;
    let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
    return normalize(mat3x3<f32>(T * invmax, B * invmax, N) * n_ts);
}

// #6 Detail maps: RNM (Reoriented Normal Mapping) blend of a base and a detail tangent-space
// normal. Combines the two in TANGENT space so a single cotangent-frame transform (perturb_normal)
// maps the result to world space ONCE — avoids double-applying the TBN. `base`/`detail` are the
// unpacked [-1,1] tangent-space normals.
fn blend_rnm(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    let t = base + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * (dot(t, u) / max(t.z, 1e-4)) - u);
}

// #6 Detail maps: build the RELATIVE UV transform from the baked base UV to the detail UV. The base
// `_MainTex_ST` (bsx,bsy,box,boy) is ALREADY baked into o.uv (and V was flipped afterward), so the
// authored detail `_Detail*Map_ST` (dsx,dsy,dox,doy) cannot be applied directly. Returns
// (scale.x, scale.y, offset.x, offset.y) such that `detail_uv = o.uv * scale + offset`. The Y
// offset term (1 - doy - ry*(1 - boy)) undoes the baked V flip. Guards a ~0 base scale.
fn detail_xform(base_st: vec4<f32>, det_st: vec4<f32>) -> vec4<f32> {
    let bsx = select(1.0, base_st.x, abs(base_st.x) > 1e-6);
    let bsy = select(1.0, base_st.y, abs(base_st.y) > 1e-6);
    let rx = det_st.x / bsx;
    let ry = det_st.y / bsy;
    return vec4<f32>(
        rx,
        ry,
        det_st.z - base_st.z * rx,
        1.0 - det_st.w - ry * (1.0 - base_st.w),
    );
}

@vertex
fn vertex(v: Vertex, @builtin(instance_index) instance_index: u32) -> VOut {
    // B5: clamp both indirections (visible[] then instances[]). @builtin(instance_index) is bounded by
    // the indirect draw args, and visible[] entries come from cs_cull, so both are in-bounds for
    // well-formed packs (no-op); the clamp only prevents an OOB fetch (AMD garbage / NVIDIA 0) if a
    // draw arg or compaction slot were ever corrupt.
    let vi = min(instance_index, arrayLength(&visible) - 1u);
    let real = min(visible[vi], arrayLength(&instances) - 1u);
    let inst = instances[real];

    // rebuild the linear 3x3 columns + translation from the ROW-MAJOR 3x4.
    let col0 = vec3<f32>(inst.m0.x, inst.m1.x, inst.m2.x);
    let col1 = vec3<f32>(inst.m0.y, inst.m1.y, inst.m2.y);
    let col2 = vec3<f32>(inst.m0.z, inst.m1.z, inst.m2.z);
    let t = vec3<f32>(inst.m0.w, inst.m1.w, inst.m2.w);
    let lin = mat3x3<f32>(col0, col1, col2);

    let world = lin * v.position + t;

    var o: VOut;
    o.clip = position_world_to_clip(world);
#ifdef DECAL_NDC_PUSH
    // Coplanar-decal separation in CLIP space, NOT the rasterizer DepthBiasState (whose `constant`
    // is `constant * 2^(exponent(z)-23)` on Depth32Float — it rides the depth exponent, drifts as the
    // camera zooms/rotates, so no magnitude is stable). After the perspective divide this is exactly
    // +eps on z_ndc, exponent-INDEPENDENT. Reverse-Z (near=1, GreaterEqual): +z = toward camera = wins.
    //
    // This def is ONLY on the decal COLOR passes (DecalColor / Overlay), which DON'T write depth, so a
    // comfortably LARGE eps can't peter-pan (it moves only the depth used for the test, never a written
    // depth or the screen xy). It must clear EVERY coplanar surface underneath: not just the opaque
    // road but OTHER overlapping road decals' depth-prepass writes — the real bug is two stacked
    // SoftCutout roads (e.g. Bus_stop_road_01 + _02) whose color passes each failed GreaterEqual against
    // the OTHER's prepass. The depth prepass (DecalDepth) does NOT get this push (it writes raw depth
    // for void occlusion; a push there WOULD peter-pan). eps large enough to clear coplanar rounding,
    // small enough that genuinely-closer geometry (walls, props) still occludes the decal.
    o.clip.z = o.clip.z + 1.0e-3 * o.clip.w;
#endif
    o.world_normal = normalize(cofactor(col0, col1, col2) * v.normal);
    o.uv = v.uv;
    o.material_index = v.material_index;
    o.color = v.color;
    o.world_pos = world; // Phase1 SH-GI: fragment samples the irradiance volume at this point
    return o;
}

// Steep parallax-occlusion mapping (MAT_FLAG_PARALLAX): march the tangent-space view ray against the
// grayscale HEIGHT map and return the UV where it first pierces the surface, so a flat panel fakes
// recessed relief (Unity _ParallaxMap; the Factory basement "fake rooms" ride this). The screen-space
// derivatives are passed IN (the caller computed them in uniform control flow), so the whole march can
// live inside the per-material `if` — textureSampleGrad takes them explicitly, needing no implicit
// derivatives. Height = 1 - sample.g (white = surface top, black = deepest recess).
fn parallax_uv(uv: vec2<f32>, dwx: vec3<f32>, dwy: vec3<f32>, dux: vec2<f32>, duy: vec2<f32>,
               gN: vec3<f32>, wp: vec3<f32>, pidx: u32, scale: f32) -> vec2<f32> {
    // DISTANCE FADE: beyond ~25m the surface covers few pixels, the screen-derivative frame turns
    // per-2x2-quad noisy, and the march offset shimmers under camera motion ("texture jitter"). The
    // relief is invisible at that range anyway — fade out over 25..50m and skip entirely past it.
    let dvec = view.world_position.xyz - wp;
    let dist = length(dvec);
    let fade = 1.0 - smoothstep(25.0, 50.0, dist);
    if (fade <= 0.001) { return uv; }
    // Mikkelsen cotangent frame from the world-pos + uv screen derivatives (matches perturb_normal).
    // Degenerate-UV guard: on constant/screen-axis-aligned UV surfaces these cross terms collapse to
    // ~0 and normalize() emits NaN that rides the UV into garbage texels (per-quad sparkle) — bail.
    let t_raw = dwx * duy.y - dwy * dux.y;
    let b_raw = dwy * dux.x - dwx * duy.x;
    if (dot(t_raw, t_raw) < 1e-16 || dot(b_raw, b_raw) < 1e-16) { return uv; }
    let T = normalize(t_raw);
    let B = normalize(b_raw);
    let Vw = dvec / max(dist, 1e-4);
    let Vts = vec3<f32>(dot(Vw, T), dot(Vw, B), dot(Vw, gN));
    let vz = max(Vts.z, 0.15);                       // clamp grazing (bound the max offset)
    let num = mix(32.0, 8.0, clamp(vz, 0.0, 1.0));   // more layers at grazing angles (larger offset)
    let layer = 1.0 / num;
    let dtex = (Vts.xy / vz) * (scale * fade) / num; // per-layer UV step, relief fading with distance
    var cuv = uv;
    var cl = 0.0;
    var h = 1.0 - textureSampleGrad(albedo_tex[pidx], albedo_samp, cuv, dux, duy).g;
    for (var i = 0; i < 32; i = i + 1) {
        if (cl >= h) { break; }
        cuv = cuv - dtex;
        h = 1.0 - textureSampleGrad(albedo_tex[pidx], albedo_samp, cuv, dux, duy).g;
        cl = cl + layer;
    }
    // Occlusion interpolation between the last (below-surface) and previous (above-surface) samples.
    let prev = cuv + dtex;
    let after = h - cl;
    let before = (1.0 - textureSampleGrad(albedo_tex[pidx], albedo_samp, prev, dux, duy).g) - (cl - layer);
    let w = clamp(after / max(after - before, 1e-4), 0.0, 1.0);
    return mix(cuv, prev, w);
}

@fragment
fn fragment(o: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // B5: clamp the material index (per-vertex Uint32 from the pack) into the materials table. The
    // global materialId equals the materials.json array index for a well-formed pack (in-bounds
    // no-op); the clamp keeps an OOB id from reading garbage on AMD.
    let m = materials[min(o.material_index, arrayLength(&materials) - 1u)];

    // #1 terrain: UV derivatives computed here in UNIFORM control flow, so the per-layer
    // textureSampleGrad calls inside the (non-uniform) terrain branch need no implicit
    // derivatives and keep correct mipmapping.
    let duv_dx = dpdx(o.uv);
    let duv_dy = dpdy(o.uv);

    // --- Parallax (steep) mapping ------------------------------------------------
    // Offset the base albedo/normal UV along the tangent-space view ray using the height map, so a
    // flat panel fakes recessed relief (Unity _ParallaxMap). The WORLD-pos derivatives are taken here
    // UNCONDITIONALLY (uniform control flow) so the per-material march below is legal; the march uses
    // textureSampleGrad (explicit gradients) and needs no implicit derivatives. `puv == o.uv` for the
    // ~all non-parallax materials, so their albedo/normal sampling is byte-identical.
    let dwp_dx = dpdx(o.world_pos);
    let dwp_dy = dpdy(o.world_pos);
    var puv = o.uv;
    if ((m.flags & MAT_FLAG_PARALLAX) != 0u) {
        let gN = select(-normalize(o.world_normal), normalize(o.world_normal), front);
        puv = parallax_uv(o.uv, dwp_dx, dwp_dy, duv_dx, duv_dy, gN, o.world_pos,
                          m.parallax_index, m.parallax_scale);
    }

    // --- #6 Detail maps: shared gate + distance fade -----------------------------
    // Strictly ADDITIVE and gated: `has_detail` is false for every non-detail material AND for all
    // terrain (terrain owns albedo/normal via the splat branch and must never enter here), so those
    // materials are byte-identical to before. The detail albedo/normal are sampled with
    // textureSampleGrad (explicit gradients) below, so they are legal in the per-material non-uniform
    // branches. Fade the whole effect out over detail_params.z..w (8..15 m) so detail tiling doesn't
    // shimmer in the distance.
    let has_detail = (m.flags & MAT_FLAG_DETAIL) != 0u && (m.flags & MAT_FLAG_TERRAIN) == 0u;
    let has_detail_albedo = has_detail && (m.detail_flags & DETAIL_HAS_ALBEDO) != 0u;
    let has_detail_normal = has_detail && (m.detail_flags & DETAIL_HAS_NORMAL) != 0u;
    let detail_dist = distance(view.world_position.xyz, o.world_pos);
    let detail_fade = 1.0 - smoothstep(m.detail_params.z, m.detail_params.w, detail_dist);

    // The pass class-discard is done AFTER the albedo sample below (Codex P0): a non-uniform
    // `discard` placed BEFORE textureSample makes the sample's implicit derivatives non-uniform
    // and FAILS naga validation, so both pipelines would fail to create. Sample first (uniform
    // control flow), THEN discard the wrong material class.

    // --- Albedo -----------------------------------------------------------------
    // Sample with the RAW baked vertex UV. Per manifest.conventions for this pack:
    //   uvVFlipBaked = true  -> V is already flipped; do NOT do 1.0 - uv.y (upside-down trap).
    //   uvTilingBaked = true -> tiling is already in the UVs; do NOT apply m.uv_xform (double-tile
    //                           trap). m.uv_xform is carried for reference only.
    // (tarkov-unity-extraction Â§4: a texture-space flip/re-tile means the geometry is wrong;
    //  never compensate on the texture â€” honor the declared, baked convention.)
    let has_albedo = m.albedo_index != MAT_ALBEDO_NONE;

    // Sample UNCONDITIONALLY, at top-level (uniform) control flow: textureSample computes
    // implicit derivatives and WGSL/naga requires that in uniform control flow, so it must NOT
    // sit inside the per-fragment `if (has_albedo)` branch (which is non-uniform). Untextured
    // materials sample slot 0 and discard the result. The index is non-uniform ACROSS fragments
    // within one draw -> requires SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    // (guarded Rust-side); that feature covers the non-uniform INDEX, not non-uniform control flow.
    let idx = select(0u, m.albedo_index, has_albedo);
    let tex = textureSampleGrad(albedo_tex[idx], albedo_samp, puv, duv_dx, duv_dy);

    // Emissive (windows / monitors / signs / lamps): same uniform-flow pattern as the albedo —
    // sample slot 0 unconditionally and select the result. Lives in the sRGB albedo array
    // (conventions.colorSpace.emissive == "srgb"). em_rgb = tex × (factor·hdr), added to the lit
    // returns below; with the HDR target + Bloom, hot emitters bleed like the game's.
    let has_em = m.emissive_index != MAT_EMISSIVE_NONE;
    let eidx = select(0u, m.emissive_index, has_em);
    let em_tex = textureSample(albedo_tex[eidx], albedo_samp, o.uv).rgb;
    // sun.gfx.z = runtime emissive scale from the UI (1 = shipped look).
    let em_rgb = select(vec3<f32>(0.0), em_tex * vec3<f32>(m.em_r, m.em_g, m.em_b) * sun.gfx.z, has_em);

    // A2C: screen-space alpha gradient width for the cutout coverage ramp (fwidth is a derivative
    // op — MUST run here in uniform control flow, before any non-uniform discard). Equals
    // fwidth(albedo.a) for textured cutouts since tint.a is flat per material.
    let cutout_aw = max(fwidth(tex.a * m.tint.a), 1e-3);
    // Pixel footprint on the world XZ plane (meters/pixel) — derivative op, so computed HERE in
    // uniform control flow. Deep water band-limits its procedural ripple with this: an octave
    // whose cycles-per-pixel nears Nyquist fades out instead of aliasing (the un-limited sines
    // re-emerged as kilometer-scale interference BEATS — broad dark fresnel bands that read as
    // giant "shadow streaks" across the sea, plus fine moiré at grazing angles).
    let wxz_footprint = max(
        abs(dpdx(o.world_pos.xz)) + abs(dpdy(o.world_pos.xz)),
        vec2<f32>(1e-6),
    );

    // --- Shading normal (Phase 2b normal mapping) --------------------------------
    // Computed ONCE here, in UNIFORM control flow, BEFORE any discard. BOTH the normal
    // textureSample AND the screen-space TBN's dpdx/dpdy need implicit derivatives, so — exactly
    // like the albedo sample above — they must NOT sit inside a per-fragment `if` (which is
    // non-uniform because m.normal_index derives from the flat material_index). So we sample slot
    // `nidx` and run perturb_normal UNCONDITIONALLY (uniform flow), then `select` the perturbed
    // normal only for materials that actually have a normal map. N is reused by the diffuse GI,
    // the dominant-light spec, and GGX below.
    var N = normalize(o.world_normal);
    if (!front) {
        N = -N; // double-sided: flip for back faces (inward shells / mirrors)
    }
    // #5: capture the GEOMETRIC (back-face-flipped) normal BEFORE normal mapping — the shadow
    // receiver-plane bias uses it (a normal map must not wobble the depth-compare offset).
    let Ng = N;
    let has_normal = m.normal_index != MAT_NORMAL_NONE;
    let nidx = select(0u, m.normal_index, has_normal); // untextured -> slot 0, result discarded
    // Read only XY (tangent) and RECONSTRUCT Z. Normal maps are stored BC5 (Rg two-channel, Z is
    // redundant); reading .z would give 0. This is also correct for the legacy raw-RGB normals
    // (Z = sqrt(1 - x² - y²) regardless), so it is a drop-in. Matches the detail-normal decode below.
    var base_xy = textureSampleGrad(normal_tex[nidx], albedo_samp, puv, duv_dx, duv_dy).xy * 2.0 - vec2<f32>(1.0);
    if ((m.normal_flags & 1u) != 0u) { base_xy.y = -base_xy.y; } // DirectX green-flip (no derivative op)
    base_xy = base_xy * m.normal_scale;
    var base_ts = vec3<f32>(base_xy, sqrt(max(1.0 - dot(base_xy, base_xy), 1.0e-4)));
    // Flat (0,0,1) when the material has no base normal map, so a detail-only material still has a
    // valid base to RNM-blend against (blend_rnm(flat, detail) == detail).
    base_ts = select(vec3<f32>(0.0, 0.0, 1.0), base_ts, has_normal);

    // #6 Detail normal: sample the detail normal (SAME bindless array), decode with the SAME
    // green-flip convention as the base, scale by strength × distance fade, keep a valid hemisphere,
    // then RNM-blend into the base tangent-space normal. textureSampleGrad uses explicit gradients
    // (scaled by the relative UV scale) so this optional sample is legal in the non-uniform branch.
    // The base+detail combine in TANGENT space; the single perturb_normal below is the ONLY TBN
    // transform (no double-application).
    var combined_ts = base_ts;
    if (has_detail_normal) {
        let dn = detail_xform(m.uv_xform, m.detail_normal_uv);
        let dn_uv = o.uv * dn.xy + dn.zw;
        let dn_s = textureSampleGrad(
            normal_tex[m.detail_normal_index], albedo_samp, dn_uv,
            duv_dx * dn.xy, duv_dy * dn.xy
        ).xy;
        var dxy = dn_s * 2.0 - vec2<f32>(1.0);
        if ((m.normal_flags & 1u) != 0u) { dxy.y = -dxy.y; }
        dxy = dxy * (m.detail_params.y * detail_fade);
        // Keep a valid hemisphere after the authored strength (avoid a NaN z / over-tilt).
        let d2 = dot(dxy, dxy);
        if (d2 > 0.99) { dxy = dxy * sqrt(0.99 / d2); }
        let detail_ts = vec3<f32>(dxy, sqrt(max(1.0 - dot(dxy, dxy), 1e-4)));
        combined_ts = blend_rnm(base_ts, detail_ts);
    }
    // Single cotangent-frame transform for base+detail combined (uniform control flow: dpdx/dpdy).
    let has_any_normal = has_normal || has_detail_normal;
    let N_mapped = perturb_normal(N, o.world_pos, o.uv, combined_ts);
    N = select(N, N_mapped, has_any_normal);

    // Pass class-discard (M3b1) — AFTER the samples above (Codex P0). Each pass keeps only its
    // class; the discard is fine here because no derivative-requiring op follows it.
    let is_blend = (m.flags & MAT_FLAG_BLEND) != 0u;
    let is_softcutout = (m.flags & MAT_FLAG_SOFTCUTOUT) != 0u;
#ifdef DECAL_DEPTH_PASS
    // Coverage-only depth prepass for SoftCutout road/track surfaces.
    if (!is_softcutout) { discard; }
#else
#ifdef DECAL_COLOR_PASS
    // The color pass uses a stronger bias than the coverage depth pass and never writes depth,
    // preventing overlapping road pieces from fighting as the view direction changes.
    if (!is_softcutout) { discard; }
#else
#ifdef BLEND_PASS
    let is_overlay = (m.flags & (MAT_FLAG_DECAL | MAT_FLAG_WATER)) != 0u;
#ifdef OVERLAY_PASS
    if (!is_blend || is_softcutout || !is_overlay) { discard; }
#else
    // True transparency (glass) has no coplanar bias; surface overlays use OVERLAY_PASS.
    if (!is_blend || is_softcutout || is_overlay) { discard; }
#endif
#else
    if (is_blend) { discard; }  // OPAQUE pipeline: keep everything except BLEND (cutout stays)
#endif
#endif
#endif

    var albedo = m.tint; // untextured (sentinel) -> tint over implicit white
    if (has_albedo) {
        albedo = tex * m.tint; // _MainTex * _Color
    }

    // #1 MicroSplat terrain: replace the single-texture albedo with the splat blend of the 12
    // layers, weighted by this slice's 3 control maps (layer i weight = ctrl(i/4).chan(i%4)).
    // textureSampleGrad takes explicit gradients, so the whole blend is safe in this branch.
    if ((m.flags & MAT_FLAG_TERRAIN) != 0u) {
        let base = m._pad2 * 3u;
        let c0 = textureSampleGrad(albedo_tex[terrain_splat.ctrl_idx[base + 0u]], albedo_samp, o.uv, duv_dx, duv_dy);
        let c1 = textureSampleGrad(albedo_tex[terrain_splat.ctrl_idx[base + 1u]], albedo_samp, o.uv, duv_dx, duv_dy);
        let c2 = textureSampleGrad(albedo_tex[terrain_splat.ctrl_idx[base + 2u]], albedo_samp, o.uv, duv_dx, duv_dy);
        var w = array<f32, 12>(c0.r, c0.g, c0.b, c0.a, c1.r, c1.g, c1.b, c1.a, c2.r, c2.g, c2.b, c2.a);
        var acc = vec3<f32>(0.0);
        var wsum = 0.0;
        for (var i = 0u; i < 12u; i = i + 1u) {
            let wi = w[i];
            if (wi <= 0.002) { continue; }
            let rep = terrain_splat.layer_rep[i];
            let la = textureSampleGrad(albedo_tex[terrain_splat.layer_albedo[i]], albedo_samp,
                                       o.uv * rep, duv_dx * rep, duv_dy * rep);
            acc = acc + wi * la.rgb;
            wsum = wsum + wi;
        }
        albedo = vec4<f32>(acc / max(wsum, 0.002), 1.0) * m.tint;
    }

    // --- Vert-Paint 3-layer splat (Custom/Vert Paint SoftCutout Decal + Shader Solid) --------
    // Replaces the single layer-0 sample with the game's height-splat blend (see VpGpu above).
    // Without this, a parking lot whose layer 0 is road_sand tiled a rust-orange blotch grid
    // where the game blends asphalt/gravel/sand by the painted COLOR_0 weights.
    // vp_smooth < 0 = "not a vp material" (also gates the matte-roughness override below).
    var vp_smooth = -1.0;
    if ((m.flags & MAT_FLAG_VP) != 0u) {
        let v = vp_table[m._pad2];
        // Relative transforms from the BAKED base UV (layer0 ST baked in, then V-FLIPPED —
        // assemble_bevy v_baked = 1-(v*sy0+oy0)) to each layer's / the heights mask's own frame.
        // detail_xform is the SAME V-flip-aware math the detail maps already validate; the naive
        // `(uv - zw)/xy` un-bake ignored the flip and shifted layers/heights by up to half a
        // tile on 136 shipped materials (Codex audit C2).
        let x1 = detail_xform(v.uv0, v.uv1);
        let x2 = detail_xform(v.uv0, v.uv2);
        let xh = detail_xform(v.uv0, vec4<f32>(1.0, 1.0, 0.0, 0.0));
        // Heights control mask (R/G/B = per-layer coverage) in its own frame — sampling each
        // layer's tiled uv instead was noise-at-3-scales fighting itself (web-viewer parity).
        var h = vec3<f32>(1.0);
        if (v.tex.w != MAT_ALBEDO_NONE) {
            h = textureSampleGrad(albedo_tex[v.tex.w], albedo_samp, o.uv * xh.xy + xh.zw,
                                  duv_dx * xh.xy, duv_dy * xh.xy).rgb;
        }
        let hw = h * o.color.rgb;
        let hs = hw.x + hw.y + hw.z;
        // Zero coverage (unpainted / "Solid" variant / missing COLOR_0) falls back to the BASE
        // layer — normalizing near-zero weights washed to an equal 3-way blend (codex fix, web).
        var w = vec3<f32>(1.0, 0.0, 0.0);
        if (hs > 1e-5) {
            w = pow(max(hw, vec3<f32>(1e-4)), vec3<f32>(max(v.tint0.w, 1.0)));
            w = w / max(w.x + w.y + w.z, 1e-4);
        }
        let u1 = o.uv * x1.xy + x1.zw;
        let u2 = o.uv * x2.xy + x2.zw;
        let a0 = textureSampleGrad(albedo_tex[v.tex.x], albedo_samp, o.uv, duv_dx, duv_dy);
        let a1 = textureSampleGrad(albedo_tex[v.tex.y], albedo_samp, u1,
                                   duv_dx * x1.xy, duv_dy * x1.xy);
        let a2 = textureSampleGrad(albedo_tex[v.tex.z], albedo_samp, u2,
                                   duv_dx * x2.xy, duv_dy * x2.xy);
        var spl = w.x * a0.rgb * v.tint0.rgb + w.y * a1.rgb * v.tint1.rgb + w.z * a2.rgb * v.tint2.rgb;
        // Near-black resolve (dark layer tints / bad mask) falls back to layer 0 (web parity).
        if (dot(spl, vec3<f32>(0.299, 0.587, 0.114)) < 0.02) {
            spl = a0.rgb * v.tint0.rgb;
        }
        albedo = vec4<f32>(spl, albedo.a) * m.tint;
        // Smoothness rides the layer alphas; the roughness override below compresses it (matte).
        vp_smooth = w.x * a0.a + w.y * a1.a + w.z * a2.a;
    }

    // --- #6 Detail albedo (mean-neutralized contrast) ----------------------------
    // Multiply the base albedo by the detail texture, but NEUTRALIZED: the detail map's own average
    // brightness/tint would darken/recolor the whole surface, so we divide out its offline-measured
    // mean (after Unity's ×4.5948 linear gain) — the result has mean ~1.0, so the detail adds only
    // LOCAL contrast, not a global shift. Distance-faded; NEVER touches albedo.a; terrain excluded by
    // `has_detail`. The detail albedo lives in the SAME sRGB `albedo_tex` array, so this sample is
    // already LINEAR (matching the base albedo path). Gradients are scaled by the relative UV scale.
    if (has_detail_albedo) {
        let da = detail_xform(m.uv_xform, m.detail_albedo_uv);
        let da_uv = o.uv * da.xy + da.zw;
        let detail_lin = textureSampleGrad(
            albedo_tex[m.detail_albedo_index], albedo_samp, da_uv,
            duv_dx * da.xy, duv_dy * da.xy
        ).rgb;
        let unity_gain = detail_lin * DETAIL_UNITY_GAIN;
        let neutral = unity_gain / max(m.detail_mean_gain.xyz, vec3<f32>(1e-3));
        let weight = clamp(m.detail_params.x * detail_fade, 0.0, 1.0);
        albedo = vec4<f32>(
            albedo.rgb * mix(vec3<f32>(1.0), clamp(neutral, vec3<f32>(0.25), vec3<f32>(4.0)), weight),
            albedo.a
        );
    }

    // Cutout / alpha-test (role=cutout, alphaMode=MASK). Per-material cutoff. discard needs no
    // derivatives, so a non-uniform branch here is fine. (Cutout is an OPAQUE-pass material â€”
    // BLEND_PASS never reaches it because the class-discard above already dropped non-blend.)
    // Alpha-test on the COMPUTED albedo.a (tex.a*tint.a, or tint.a when untextured) so an
    // untextured cutout with tint.a < cutoff still discards (Codex P2), not just textured ones.
    // A2C: the threshold sits at HALF the cutoff so alpha-to-coverage can dither the lower half
    // of the coverage ramp (the return below outputs the fwidth-remapped coverage); the hard
    // discard still kills fully-transparent texels so depth stays clean.
    if ((m.flags & MAT_FLAG_CUTOUT) != 0u && albedo.a < 0.5 * m.alpha_cutoff) {
        discard;
    }

    // --- Lighting (baked SH-GI irradiance volume; Phase 1) -----------------------
    // Sample the RTX-baked SH irradiance volume per-fragment. It ALREADY integrates
    // neutral sky + warm sun + 1285 practical lights + 1 diffuse bounce, so this
    // single diffuse GI term MATCHES Unity with no per-light evaluation. Output is
    // LINEAR HDR radiance — Bevy's Tonemapping node runs after this pass, so do NOT
    // tonemap or gamma-encode here. `N` is the normal-mapped shading normal computed at the
    // top (perturbed by normal_tex when present, else the back-face-flipped geometric normal).
    // Ambient floor: keep enclosed interiors (checkout nooks, deep aisles) from crushing to
    // pure black. Baked GI reaches very little light into these pockets, and against the AgX
    // grade a low-albedo prop there reads as *invisible*. max() only lifts the darkest areas.
    // Subtle floor (0.03): prevents pure-black crushed interiors without washing shadows.
    // (The 0.25 test proved the invisible register is NOT a darkness issue — real render bug.)
    let ambient_floor = vec3<f32>(0.03);
    let gi = max(sh_irradiance(o.world_pos, N), ambient_floor);

    // Dominant baked light dir/color/directionality — DERIVED once from the SH volume and shared by
    // BOTH the #5 shadow gate (below) and the GGX spec lobe (further down).
    let dom = sh_dominant_light(o.world_pos);

    // --- #5 Dynamic sun shadow term (double-darkening-safe) ----------------------
    // `shadow_event` in [0,1] is how strongly this fragment is in a NEW sun shadow that the baked SH
    // does NOT already represent. It is the product of three gates so it can only fire where the
    // baked lighting is genuinely direct-sun-lit:
    //   * SH directionality  — flat/isotropic (already-shadowed) points are ~0.
    //   * dom·Lsun alignment — the SH dominant light must actually BE the sun.
    //   * N·Lsun             — the surface must face the sun.
    // times a far contact fade, times the PCF occlusion (1 - visibility). Fully gated off (0) when
    // the feature is disabled (sun_dir missing or not EFT_SHADOWS=1), so the render is identical to today.
    var shadow_event = 0.0;
    var sun_diffuse = vec3<f32>(0.0); // B4-M: additive direct-sun diffuse (indirect-only bakes)
    if (sun.split_depths.w > 0.5) {
        let Lsun = sun.sun_dir_texel.xyz;
        let align = dot(dom.dir, Lsun);
        let NdotSun = dot(N, Lsun);
        let sun_lit_gate =
              smoothstep(0.10, 0.35, dom.directionality)
            * smoothstep(0.75, 0.95, align)
            * smoothstep(0.05, 0.35, NdotSun);
        let view_depth = -(view.view_from_world * vec4<f32>(o.world_pos, 1.0)).z;
        let contact_fade = 1.0 - smoothstep(sun.combine.y, sun.combine.z, view_depth); // 65..80 m
        let shadow_vis = sun_shadow_visibility(o.world_pos, Ng, view_depth);
        shadow_event = sun_lit_gate * contact_fade * (1.0 - shadow_vis);
        // B4-M: the SH carries sky + bounce ONLY (indirect-only bake), so there is no direct-sun
        // Lambert term and sunlit exteriors read flat vs the game. Add a sun diffuse on surfaces
        // FACING the sun that the shadow map says are lit. Gate on N·Lsun × shadow_vis ONLY — NOT
        // `sun_lit_gate`, whose SH-directionality/align requirement is ~0 on an intentionally
        // sky-dominant indirect SH and would cancel the very term we want. Tinted by dom.radiance
        // (the map's daylight colour/level -> auto-scales, no blow-out, no new uniform). Only when
        // realtime lighting is on. Strength = lgrid.params.w (EFT_SUN_DIFFUSE, set to 0 on a FULL
        // bake so the already-integrated sun is never double-counted). Live-tunable, no rebuild.
        if (lgrid.params.w > 0.0) {
            sun_diffuse = dom.radiance * (lgrid.params.w * max(NdotSun, 0.0) * shadow_vis);
        }
    }

    // Anti double-darkening combination: the SH volume ALREADY integrates the broad sun shadow, so
    // the contact term may only remove a BOUNDED fraction of the REMOVABLE diffuse (everything above
    // the 0.03 ambient floor), never the floor itself. Cap = combine.x (0.12): a sunlit contact edge
    // loses at most 12% of its above-floor diffuse; an already-dark SH region (low gate) is untouched
    // (< ~1% change). `gi_shadowed >= ambient_floor` component-wise is preserved. combine.w==1
    // (EFT_SHADOW_DEBUG=1) zeroes the diffuse coeff -> specular-only diagnostic.
    let diffuse_cap = select(sun.combine.x, 0.0, sun.combine.w > 0.5);
    let removable = max(gi - ambient_floor, vec3<f32>(0.0));
    let gi_shadowed = gi - removable * (diffuse_cap * shadow_event);
    // `lit` is computed BELOW, after `rough` + the realtime-light loop, so the realtime diffuse
    // folds into the same albedo × irradiance term (see eval_realtime_lights).

    // --- Specular: dielectric GGX / Cook-Torrance ON TOP of the SH diffuse (Phase 1.6) -----
    // Unity-Standard-style spec lobe lit from the SH volume's own dominant light dir + color,
    // so the highlight is consistent with the baked GI (no new data). Dielectric: F0 = 0.04
    // (every material is metallic≈0). Roughness is per-material (glass=0.05 -> a sharp glint).
    // N is the Phase-2b normal-mapped shading normal computed at the top (back-face-flipped
    // geometric normal, perturbed by the tangent-space normal map when the material has one).
    // Shared view + mirror-reflection direction (used by the GGX lobe AND the env reflection below).
    let V = normalize(view.world_position.xyz - o.world_pos);
    let NdotV = max(dot(N, V), 1e-3);
    let R = reflect(-V, N);
    let fresnel_v = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0); // Schlick view-fresnel, dielectric F0=0.04

    // Per-pixel roughness: RFA materials (82% of the pack — Unity Standard smoothness-in-alpha)
    // derive roughness from the RAW tex.a (NOT albedo.a — tint.a would bias it); everything else
    // keeps the per-material constant. Water floors at 0.10 so its sun glint is a believable
    // overcast smudge rather than a pinprick (the analytic sun disk that used to fake width is gone).
    var rough = clamp(m.roughness, 0.03, 1.0);
    if ((m.flags & MAT_FLAG_RFA) != 0u && has_albedo) {
        rough = clamp(1.0 - tex.a, 0.06, 1.0);
    }
    if ((m.flags & MAT_FLAG_WATER) != 0u) {
        rough = max(rough, 0.10);
    }
    // Vert-paint ground matte: the splat layers pack smoothness in alpha, but raw 1-smoothness
    // read near-mirror (wet-glossy road slabs popping off the matte terrain). Compress ×0.30 and
    // floor at 0.72 — asphalt/dirt are matte in-game (web-viewer-validated constants).
    if (vp_smooth >= 0.0) {
        rough = clamp(1.0 - 0.30 * vp_smooth, 0.72, 1.0);
    }

    // --- REALTIME point/spot lights ----------------------------------------------
    // Direct contribution from the pack's realtime lights (auto-gated vs the baked SH volume so they
    // never double-count). Diffuse folds into the SH `gi` irradiance (÷π-consistent); spec adds to the
    // GGX lobe. `ambient_scale` (params.y, default 1.0) trims the SH ambient — on the no-CUDA path the
    // SH is a flat ~0.28 dummy, so this is the gentle base the realtime bubbles sit on. Both terms are
    // zero when realtime is off (params.z==0), so the baked-SH path renders byte-identically.
    let rt = eval_realtime_lights(o.world_pos, N, V, rough);
    // SH ambient (× gi_intensity × ambient_scale) + the realtime diffuse (which carries its own
    // light_scale, independent of gi_intensity), all modulated by albedo.
    let lit = albedo.rgb * (gi_shadowed * sh.vol_min.w * lgrid.params.y + rt.diffuse + sun_diffuse);

    var spec_rgb = vec3<f32>(0.0);
    if (dom.mag >= 1e-4) {
        let Ld = dom.dir;
        let H = normalize(V + Ld);
        let NdotL = max(dot(N, Ld), 0.0);
        let NdotH = max(dot(N, H), 0.0);
        let VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0) {
            let a  = rough * rough;
            let a2 = a * a;
            let d  = (NdotH * NdotH * (a2 - 1.0) + 1.0);
            let D  = a2 / (3.14159265 * d * d);                       // GGX/Trowbridge-Reitz NDF
            let sk = (rough + 1.0) * (rough + 1.0) / 8.0;             // Smith k (direct lighting)
            let gv = NdotV / (NdotV * (1.0 - sk) + sk);
            let gl = NdotL / (NdotL * (1.0 - sk) + sk);
            let G  = gv * gl;                                         // Smith geometry
            let F  = 0.04 + 0.96 * pow(1.0 - VdotH, 5.0);            // Schlick, dielectric F0=0.04
            spec_rgb = dom.radiance * (D * G * F / (4.0 * NdotV * NdotL + 1e-4)) * NdotL * SPEC_STRENGTH;
        }
    }
    // #5: the GGX lobe is the ONLY real-time directional-looking term, and it is NOT baked into the
    // SH volume, so it takes the FULL shadow (unlike the capped diffuse). Safe from double-darkening
    // because sun_lit_gate already required SH directionality + sun alignment.
    // The realtime-light GGX (rt.spec) is added AFTER the sun-shadow attenuation — it's lit by the
    // practicals, not the sun, so the sun-shadow term must not darken it.
    spec_rgb = spec_rgb * (1.0 - shadow_event) + rt.spec;

    // --- Environment reflection (analytic sky + sun, anchored to the baked SH) --------------------
    // Two environments: the SH volume (scene-accurate color, but a dull ground-level probe) and an
    // analytic overcast SKY (crisp + bright, from sky_reflect). Glossier surfaces see more of the
    // sky; rougher ones fade back to the soft SH probe. Fresnel brightens toward grazing angles (why
    // glass / wet floors / water flash at glancing view). Matte surfaces untouched (gloss ~0). Sun-
    // shadowed like the GGX lobe. `sh_env`/`sky_env` are reused by the water/glass blend branches.
    let sh_env = max(sh_irradiance(o.world_pos, R), ambient_floor) * sh.vol_min.w;
    let refl_level = dot(sh_env, vec3<f32>(0.2126, 0.7152, 0.0722)); // local exposure anchor (luma)
    let sky_env = sky_reflect(R, refl_level);
    let gloss = 1.0 - rough; // per-pixel when RFA — glossy pipes/tiles pop, worn surfaces go matte
    let env = mix(sh_env, max(sh_env, sky_env), gloss);
    let refl_rgb = env * (fresnel_v * gloss * ENV_REFL_STRENGTH) * (1.0 - shadow_event);

    // Water class is needed by BOTH passes now: textured puddles blend (P2), untextured DEEP
    // water (sea / basins) is OPAQUE (P1) so depth sorts it correctly under glass.
    let is_water = (m.flags & MAT_FLAG_WATER) != 0u;

#ifdef DECAL_DEPTH_PASS
    // Softcutout road/track coverage for the DEPTH-ONLY prepass. Coverage is the
    // PER-VERTEX COLOR_0.a modulated by the SoftCutout params (tex.a is SMOOTHNESS here, NOT
    // coverage). Alpha-to-coverage writes only covered samples; color writes are disabled in the
    // pipeline. The extra ×color.a keeps feather tails soft where _AlphaStrength would re-saturate.
    //   coverage = clamp(color.a*_AlphaStrength - (_Cutoff - _AlphaHeight), 0, 1) * color.a
    let coverage = clamp(o.color.a * m.vp.x - (m.vp.y - m.vp.z), 0.0, 1.0) * o.color.a;
    return vec4<f32>(0.0, 0.0, 0.0, coverage);
#else
#ifdef DECAL_COLOR_PASS
    // Continuous premultiplied color feather. Depth was established by DECAL_DEPTH_PASS; this pass
    // uses a slightly stronger bias, tests but does not write depth, so overlapping roads blend in
    // stable phase order rather than competing in the depth buffer.
    let coverage = clamp(o.color.a * m.vp.x - (m.vp.y - m.vp.z), 0.0, 1.0) * o.color.a;
    let col = apply_fog(lit + spec_rgb, o.world_pos, dom.directionality);
    return vec4<f32>(col * coverage, coverage);
#else
#ifdef BLEND_PASS
    // Transparent/overlay pass: emit premultiplied rgb plus the real computed opacity.
    // Softcutout roads are handled by the dedicated depth + color pair above.
    //  * Water/mirror (role=water): untextured water had albedo=tint=WHITE -> a flat white slab.
    //    Emit a translucent dark wet sheen instead (animated flow deferred).
    //  * Other blend (glass / plain decal): keep the tex.a*tint.a coverage.
    if (is_water) {
        // PUDDLE — the game's `Decal/Water Deferred Decal`, RE'd from its DX11 fragment and
        // reproduced in the forward BLEND pass. The hardware SrcAlpha/OneMinusSrcAlpha blend gives
        // result = col*a + road*(1-a) = lerp(road, col, a), matching the decal's G-buffer albedo
        // lerp — it DARKENS/wets the road, it does NOT paint a black body over it.
        //   coverage = saturate((mask + COLOR_0.a) * _FadeStrength=1.52)   (soft, no hard cutoff)
        //   mask channel: LUMA(rgb) for atlas puddles (alpha≡1), else raw tex.a  (per-mat flag)
        //   F = pow(1 - NdotV*_Fresnel=0.354, 2)  -> ~0.42 head-on..1 grazing (never black)
        //   reflection gated to the INTERIOR (coverage>0.5); edges only wet-darken.
        let mask_ch = select(tex.a, dot(tex.rgb, vec3<f32>(0.299, 0.587, 0.114)),
                             (m.flags & MAT_FLAG_PUDDLE_LUMA) != 0u);
        // The game's DXBC is `saturate((mask + COLOR_0.a) * _FadeStrength=1.52)`. EFT's puddle DECAL
        // meshes carry NO vertex-color channel (0/38 water meshes on lighthouse have one), so their
        // COLOR_0 is the shader default — and for these deferred decals the game renders SOFT,
        // mask-driven puddles, which is only consistent with a default COLOR_0.a of 0 (a default of
        // 1.0 would saturate every puddle to a hard opaque slab). Our assembler wrongly writes
        // opaque-white COLOR_0 for non-vp meshes, so `o.color.a` = 1.0 here and (mask + 1.0)*1.52
        // hard-slabbed the puddle. Force COLOR_0.a = 0 -> the game's exact mask-driven coverage.
        let color0_a = 0.0; // puddle decals have no painted COLOR_0; Unity's decal default is 0, not 1
        let raw_coverage = clamp((mask_ch + color0_a) * 1.52, 0.0, 1.0);
        // BC/mip filtering can leave a few percent of mask outside the authored puddle. Multiplying
        // that tail by the material tint made the physical quad boundary visible. Suppress only the
        // near-zero tail with a smooth ramp; authored mid/high coverage remains unchanged.
        let coverage = raw_coverage * smoothstep(0.015, 0.10, raw_coverage);
        // A large STRETCHED floor decal (texture mapped at tens-to-hundreds of meters per repeat,
        // flagged per-material at load) is matte wet-ground / tire marks, NOT a reflective puddle —
        // kill the sky mirror + sun glint so it reads as a dark decal. Real puddles keep both.
        let matte = (m.flags & MAT_FLAG_WATER_MATTE) != 0u;
        let ndv = max(NdotV, 1e-3);
        let fr = pow(1.0 - ndv * 0.354, 2.0);              // game _Fresnel
        let refl = max(sh_env, sky_env) * (fr * 0.88);     // _ReflectionStrength, sharp sky mirror
        let refl_mix = select(clamp(ndv * ndv * 2.0 * max(coverage - 0.5, 0.0), 0.0, 1.0) * fr, 0.0, matte);
        let wet = m.tint.rgb * gi;                          // dark wet asphalt (grey _Color), GI-lit
        let col = mix(wet, refl, refl_mix) + select(spec_rgb, vec3<f32>(0.0), matte);
        let a = coverage * m.tint.a;                        // _Color.a = overall puddle strength
        // PREMULTIPLIED output: rgb = col*a preserves the puddle's intended lerp(road, col, a) —
        // the reflection is DELIBERATELY weighted by coverage here (it wets the road), unlike glass.
        return vec4<f32>(apply_fog(col, o.world_pos, dom.directionality) * a, a);
    }
    // Plain surface decal: PREMULTIPLIED, with EVERY contribution coverage-masked. Decal atlases
    // commonly have transparent RGB texels around the visible pothole/stain/tire mark. Letting
    // specular or environment light escape the mask paints the atlas quad as a pale rectangle even
    // where albedo.a == 0 (the Lighthouse pothole atlas is 61% fully transparent). This is the
    // material-class distinction that the shared glass/decal branch previously lost.
    if ((m.flags & MAT_FLAG_DECAL) != 0u) {
        let col = apply_fog(lit + spec_rgb + refl_rgb + em_rgb, o.world_pos, dom.directionality);
        return vec4<f32>(col * albedo.a, albedo.a);
    }
    // Glass: PREMULTIPLIED. Transmission alpha scales only the DIFFUSE (lit) — the env reflection,
    // GGX glint and emissive are ADDED at full strength so a clear pane mirrors the bright overcast
    // sky instead of reading as a dark tinted slab. Emissive rides here too (lit windows/signage).
    return vec4<f32>(
        apply_fog(lit, o.world_pos, dom.directionality) * albedo.a + spec_rgb + refl_rgb + em_rgb,
        albedo.a
    );
#else
    // DEEP WATER (untextured role=water: the sea + treatment basins) — OPAQUE pass, so depth-write
    // sorts it under glass and no pale clear-color bleeds through. Energy-balanced fresnel between
    // a dark teal body (IGNORE the white tint) and the sky mirror, over a procedural ripple that
    // perturbs the EXISTING shading normal (keeps any authored wave normal map) and fades with
    // distance (raw sin ripples alias into sparkle at km range — and fog owns the far look anyway).
    if (is_water && !has_albedo) {
        let d = distance(view.world_position.xyz, o.world_pos);
        // Amp falls off hard past ~250 m: the derivative footprint underreads on this one
        // 6.8 km quad at grazing, so the Nyquist gates alone leave faint residue; beyond a
        // few hundred meters fog owns the look and the game's sea reads flat anyway.
        let ripple_amp = (0.06 / (1.0 + d * 0.004)) * (1.0 - smoothstep(150.0, 350.0, d));
        let wp = o.world_pos.xz * 0.35;
        // Nyquist band-limit per octave (shader-side "mip" for the procedural sines): fade an
        // octave to zero as its cycles-per-pixel pass 1/4 -> 1/2. The old amplitude-only distance
        // fade left the FREQUENCY aliasing: undersampled sines beat into kilometer-scale dark
        // fresnel bands ("shadow streaks") + fine moiré at altitude/grazing views. Derivative
        // footprint (wxz_footprint, uniform-flow above) covers distance AND grazing angle with
        // no tuned distance constants; shore-level close-ups keep both octaves untouched.
        let cyc = 0.35 * max(wxz_footprint.x, wxz_footprint.y) / 6.2831853;
        // Conservative gates (fade well before Nyquist): the screen-derivative footprint
        // UNDERestimates at extreme grazing on this one giant quad, so 0.25..0.5 still let a
        // ~2px wave alias faintly. Near-shore footprints are centimeters — unaffected.
        let w1 = 1.0 - smoothstep(0.10, 0.22, cyc);        // base octave (~18 m wavelength)
        let w2 = 1.0 - smoothstep(0.10, 0.22, cyc * 3.4);  // detail octave (~5 m wavelength)
        let wc = wp / 6.2831853; // phase in CYCLES for the f32-safe rsin (see rsin above)
        let dxy = (w1 * vec2<f32>(rsin(wc.x + wc.y * 0.6), rsin(wc.x * 0.7 - wc.y))
                 + 0.5 * w2 * vec2<f32>(rsin(wc.x * 3.1 + wc.y * 2.3), rsin(wc.y * 3.7 - wc.x * 2.9)))
                 * ripple_amp;
        // Base the water normal on WORLD UP, not the interpolated mesh/map normal: the sea mesh's
        // per-vertex normals crosshatch at the quad-grid period (fine screen stripes + the wide
        // dark fresnel bands the user reported as "shadow streaks" — verified by red-paint diag +
        // pixel sampling: the pattern lives in this branch and survives ripple/SSAO/sh_env A/Bs).
        // EFT water is horizontal; the game's shader owns the surface with its own wave normals,
        // which the band-limited ripple above emulates.
        let Nup = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), !front);
        let Nw = normalize(Nup + vec3<f32>(dxy.x, 0.0, dxy.y));
        let Rw = reflect(-V, Nw);
        let NwV = max(dot(Nw, V), 1e-3);
        let wf = 0.02 + 0.98 * pow(1.0 - NwV, 5.0);
        // Sample the water's SH terms one probe layer ABOVE the surface: sea-level probes straddle
        // the water plane (below-water = occluded/dark, above = bright sky), and their trilinear
        // alternation at the probe-grid period (~5 m) checkerboarded the whole sea — the pattern
        // rode in through refl_level (the sky_reflect luma anchor), which the red-paint diagnostic
        // + pixel sampling pinned to this branch after ripple/mesh-normal/sh_env-max A/Bs all
        // came back byte-identical. The surface's environment is the AIR above it by definition.
        // _hw variant (blind trilinear): the vis-weighted 8-tap alternates with the bake's
        // visibility texture even above the surface; open sky needs no leak protection.
        let p_air = o.world_pos + vec3<f32>(0.0, max(sh.spacing.y, 1.0), 0.0);
        let gi_w = max(sh_irradiance_hw(p_air, Nw), ambient_floor) * sh.vol_min.w;
        let env_w = max(sh_irradiance_hw(p_air, Rw), ambient_floor) * sh.vol_min.w;
        let lvl_w = dot(env_w, vec3<f32>(0.2126, 0.7152, 0.0722));
        let refl = max(env_w, sky_reflect(Rw, lvl_w));
        let deep = vec3<f32>(0.015, 0.045, 0.060);
        // Reinhard-compress the sky mirror: the baked open-sky SH is HDR-bright (sky_scale ≈ 2), and
        // reflecting it raw washed the whole sea to a pale slab (term-bisected: refl*wf ≈ 4x the
        // body+glint terms). The game's overcast sea is a MUTED sheen over a dark teal body — compress
        // the reflection into [0,1) and weight it down so the body color owns the look head-on and the
        // sky only takes over toward grazing angles (wf -> 1 keeps the horizon bright, like the game).
        let refl_c = refl / (vec3<f32>(1.0) + refl);
        // Keep the INPUT luminance low: the pack's warm grade LUT (+exposure) renders any bright
        // pale-blue as cream (grade-off A/B measured [152,170,172] -> graded beige). The game's sea
        // is dark enough that the same warm grade reads dark green-grey — so (a) the body ABSORBS:
        // clamp the open-sky irradiance instead of scaling linearly with it, and (b) the sky mirror
        // stays grazing-weighted (bright horizon band like the game, dark head-on body).
        // The pack's game grade LUT is LDR-authored: after the x1.35 exposure it CLIPS every
        // channel >= ~0.74 linear to one flat highlight plateau (~(244,227,191) -> the "white slab").
        // So the water must arrive BELOW that plateau or its color is unrecoverable. Cap the radiance
        // hue-preservingly: dark teal body head-on (~0.18), brighter grazing band (~0.28) like the
        // game's horizon sheen — never into the LUT clip. (Codex-diagnosed via LUT evaluation.)
        let body = deep * min(gi_w, vec3<f32>(1.2));
        let raw_col = body * (1.0 - wf) + refl_c * (wf * mix(0.08, 0.24, wf)) + spec_rgb;
        let peak = max(raw_col.r, max(raw_col.g, raw_col.b));
        let water_cap = mix(0.18, 0.28, wf);
        let col = raw_col * min(1.0, water_cap / max(peak, 1.0e-5));
        // Open water is never indoors: directionality=1 keeps the full outdoor fog. The per-pixel
        // dom.directionality (unlifted SH) alternated with the probe grid and MODULATED THE FOG
        // into the residual banding (survived the gi/env/spec eliminations above).
        return vec4<f32>(apply_fog(col, o.world_pos, 1.0), 1.0);
    }
    // OPAQUE pass: diffuse + GGX glint + the glossy env reflection (matte surfaces: refl_rgb ~0)
    // + emissive, all under the aerial-perspective fog. Cutouts output the fwidth-remapped
    // coverage ramp for alpha-to-coverage (MSAA dithers the edge); everything else is alpha 1.0.
    let is_cut = (m.flags & MAT_FLAG_CUTOUT) != 0u;
    let cov = clamp((albedo.a - m.alpha_cutoff) / cutout_aw + 0.5, 0.0, 1.0);
    return vec4<f32>(
        apply_fog(lit + spec_rgb + refl_rgb + em_rgb, o.world_pos, dom.directionality),
        select(1.0, cov, is_cut)
    );
#endif
#endif
#endif
}
