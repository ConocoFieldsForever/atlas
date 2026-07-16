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
    em_b: f32,                     // @172  -> 176
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
fn sh_irradiance_hw(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let uvw = (world_pos - sh.vol_min.xyz) * sh.vol_inv_extent.xyz;
    let cr = textureSampleLevel(sh_r, sh_samp, uvw, 0.0); // (c0,c1,c2,c3) for R
    let cg = textureSampleLevel(sh_g, sh_samp, uvw, 0.0);
    let cb = textureSampleLevel(sh_b, sh_samp, uvw, 0.0);
    let w = vec3<f32>(n.y, n.z, n.x) * 0.325735;          // weights for c1(∝y),c2(∝z),c3(∝x)
    let e = vec3<f32>(
        0.282095 * cr.x + cr.y * w.x + cr.z * w.y + cr.w * w.z,
        0.282095 * cg.x + cg.y * w.x + cg.z * w.y + cg.w * w.z,
        0.282095 * cb.x + cb.y * w.x + cb.z * w.y + cb.w * w.z);
    return max(e, vec3<f32>(0.0)); // clamp >= 0 (SH rings negative)
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
    let bias = sh.vol_inv_extent.w;
    let sp   = world_pos + n * bias;
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
        let dir = probe_pos - world_pos;
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
    if (wsum < 1e-3) { return sh_irradiance_hw(world_pos, n); }
    return sum / wsum;
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
    let uvw = (world_pos - sh.vol_min.xyz) * sh.vol_inv_extent.xyz;
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
    out.radiance = max(vec3<f32>(rr, rg, rb), vec3<f32>(0.0)) * out.directionality;
    return out;
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
    let real = visible[instance_index];
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
    o.world_normal = normalize(cofactor(col0, col1, col2) * v.normal);
    o.uv = v.uv;
    o.material_index = v.material_index;
    o.color = v.color;
    o.world_pos = world; // Phase1 SH-GI: fragment samples the irradiance volume at this point
    return o;
}

@fragment
fn fragment(o: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let m = materials[o.material_index];

    // #1 terrain: UV derivatives computed here in UNIFORM control flow, so the per-layer
    // textureSampleGrad calls inside the (non-uniform) terrain branch need no implicit
    // derivatives and keep correct mipmapping.
    let duv_dx = dpdx(o.uv);
    let duv_dy = dpdy(o.uv);

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
    let tex = textureSample(albedo_tex[idx], albedo_samp, o.uv);

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
    let nt = textureSample(normal_tex[nidx], albedo_samp, o.uv).rgb;
    var base_ts = nt * 2.0 - vec3<f32>(1.0);
    if ((m.normal_flags & 1u) != 0u) { base_ts.y = -base_ts.y; } // DirectX green-flip (no derivative op)
    base_ts = vec3<f32>(base_ts.xy * m.normal_scale, max(base_ts.z, 1e-3));
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
#ifdef BLEND_PASS
    if (!is_blend) { discard; } // BLEND pipeline: keep only decal/glass/water/alphaMode=BLEND
#else
    if (is_blend) { discard; }  // OPAQUE pipeline: keep everything except BLEND (cutout stays)
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
    let lit = albedo.rgb * gi_shadowed * sh.vol_min.w; // vol_min.w = gi_intensity

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
    spec_rgb = spec_rgb * (1.0 - shadow_event);

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

#ifdef BLEND_PASS
    // BLEND pass: emit the REAL computed opacity. Non-premultiplied to match the pipeline's
    // BlendState::ALPHA_BLENDING (src=SrcAlpha, dst=OneMinusSrcAlpha), i.e. Unity _Color*_MainTex.
    //
    // Three coverage laws (M3b2), by material class:
    //  * SoftCutout (Custom/Vert Paint SoftCutout Decal): coverage is the PER-VERTEX COLOR_0.a
    //    modulated by the SoftCutout params — tex.a is SMOOTHNESS here, NOT coverage. This
    //    feathers roads / tire-tracks into the terrain (soft edges), fixing the floating-slab /
    //    solid-quad look. rgb stays the lit (tex.rgb*tint.rgb).
    //      coverage = clamp(color.a*_AlphaStrength - (_Cutoff - _AlphaHeight), 0, 1)
    //    (matches the RE'd EFT road shader; NO polygonOffset — the feather + depth-write-off
    //    handle coplanarity, per tarkmap-road-terrain-matte-and-hole-bake.)
    //  * Water/mirror (role=water): untextured water had albedo=tint=WHITE -> a flat white slab.
    //    Emit a translucent dark wet sheen instead (animated flow deferred).
    //  * Other blend (glass / plain decal): keep the tex.a*tint.a coverage.
    let is_softcutout = (m.flags & MAT_FLAG_SOFTCUTOUT) != 0u;
    if (is_softcutout) {
        // Roads/decals are matte ground overlays — no env reflection (keeps asphalt from mirroring).
        let coverage = clamp(o.color.a * m.vp.x - (m.vp.y - m.vp.z), 0.0, 1.0);
        return vec4<f32>(apply_fog(lit + spec_rgb, o.world_pos, dom.directionality), coverage);
    } else if (is_water) {
        // PUDDLE (textured water — untextured DEEP water is opaque-pass now, see #else path).
        // A thin wet film: albedo.a is the puddle_noise coverage but tex.a*tint.a (tint.a≈0.3)
        // crushed it to ~7% -> divide tint.a back out to recover the mask, remap to a clean
        // feathered SHAPE, and let opacity follow it so edges dissolve into the dry ground.
        // Energy-balanced fresnel mix (body*(1-wf) + refl*wf): no additive pedestal, so the film
        // is dark from above and a mirror only toward grazing — like a real puddle.
        let wf = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);      // water fresnel (F0≈0.02..0.04)
        let refl = max(sh_env, sky_env);                   // crisp bright sky mirror
        let raw = albedo.a / max(m.tint.a, 1e-3);          // ~ puddle_noise coverage in [0,1]
        let mask = smoothstep(0.12, 0.45, raw);            // feathered puddle shape
        let body = m.tint.rgb * gi;                        // dark wet film, GI-lit
        let col = body * (1.0 - wf) + refl * wf + spec_rgb;
        let a = mask * clamp(0.45 + 0.55 * wf, 0.0, 1.0);  // shape-masked; grazing -> opaque mirror
        return vec4<f32>(apply_fog(col, o.world_pos, dom.directionality), a);
    }
    // Glass / plain decal: env reflection (glass is glossy -> strong) + the GGX glint make it read as
    // reflective, not a flat tinted pane. Emissive rides here too (lit windows / signage panes).
    return vec4<f32>(
        apply_fog(lit + spec_rgb + refl_rgb + em_rgb, o.world_pos, dom.directionality),
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
        let ripple_amp = 0.06 / (1.0 + d * 0.004);
        let wp = o.world_pos.xz * 0.35;
        let dxy = (vec2<f32>(sin(wp.x + wp.y * 0.6), sin(wp.x * 0.7 - wp.y))
                 + 0.5 * vec2<f32>(sin(wp.x * 3.1 + wp.y * 2.3), sin(wp.y * 3.7 - wp.x * 2.9)))
                 * ripple_amp;
        let Nw = normalize(N + vec3<f32>(dxy.x, 0.0, dxy.y));
        let Rw = reflect(-V, Nw);
        let NwV = max(dot(Nw, V), 1e-3);
        let wf = 0.02 + 0.98 * pow(1.0 - NwV, 5.0);
        let refl = max(sh_env, sky_reflect(Rw, refl_level));
        let deep = vec3<f32>(0.015, 0.045, 0.060);
        let col = deep * gi * (1.0 - wf) + refl * wf + spec_rgb;
        return vec4<f32>(apply_fog(col, o.world_pos, dom.directionality), 1.0);
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
}
