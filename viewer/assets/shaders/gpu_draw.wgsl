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
// 80 bytes, 16-aligned (Phase 2b added the normal block). Matches the Rust `MaterialGpu` POD.
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
};

// Phase 1.6 GGX specular tuning: global multiplier on the dielectric spec lobe. 1.0 = physical.
// Dial down if broad interior highlights read too hot, up if wet floors/glass look flat.
const SPEC_STRENGTH: f32 = 1.0;

const MAT_ALBEDO_NONE: u32 = 0xFFFFFFFFu; // sentinel: material has no albedo texture
const MAT_NORMAL_NONE: u32 = 0xFFFFFFFFu;  // sentinel: material has no normal map (Phase 2b)
const MAT_FLAG_CUTOUT: u32 = 1u;          // bit0: alpha-tested (MASK) surface
const MAT_FLAG_BLEND: u32 = 2u;           // bit1: alpha-blended (decal/glass/water/BLEND) surface
const MAT_FLAG_SOFTCUTOUT: u32 = 4u;      // bit2: Vert Paint SoftCutout decal (feather via color.a)
const MAT_FLAG_WATER: u32 = 8u;           // bit3: water/mirror (dark wet sheen, not white fallback)

@group(2) @binding(0) var<storage, read> materials: array<MaterialGpu>;
@group(2) @binding(1) var albedo_tex: binding_array<texture_2d<f32>>;
@group(2) @binding(2) var albedo_samp: sampler;
// Phase 2b: bindless normal-map array (LINEAR data). Reuses `albedo_samp`; indexed non-uniformly
// by m.normal_index (same device feature as albedo). Sentinel MAT_NORMAL_NONE = no normal map.
@group(2) @binding(3) var normal_tex: binding_array<texture_2d<f32>>;

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
    out.radiance = max(vec3<f32>(rr, rg, rb), vec3<f32>(0.0));
    return out;
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
    let has_normal = m.normal_index != MAT_NORMAL_NONE;
    let nidx = select(0u, m.normal_index, has_normal); // untextured -> slot 0, result discarded
    let nt = textureSample(normal_tex[nidx], albedo_samp, o.uv).rgb;
    var n_ts = nt * 2.0 - vec3<f32>(1.0);
    if ((m.normal_flags & 1u) != 0u) { n_ts.y = -n_ts.y; } // DirectX green-flip (no derivative op)
    n_ts = vec3<f32>(n_ts.xy * m.normal_scale, max(n_ts.z, 1e-3));
    let N_mapped = perturb_normal(N, o.world_pos, o.uv, n_ts);
    N = select(N, N_mapped, has_normal);

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

    // Cutout / alpha-test (role=cutout, alphaMode=MASK). Per-material cutoff. discard needs no
    // derivatives, so a non-uniform branch here is fine. (Cutout is an OPAQUE-pass material â€”
    // BLEND_PASS never reaches it because the class-discard above already dropped non-blend.)
    // Alpha-test on the COMPUTED albedo.a (tex.a*tint.a, or tint.a when untextured) so an
    // untextured cutout with tint.a < cutoff still discards (Codex P2), not just textured ones.
    if ((m.flags & MAT_FLAG_CUTOUT) != 0u && albedo.a < m.alpha_cutoff) {
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
    let lit = albedo.rgb * gi * sh.vol_min.w; // vol_min.w = gi_intensity

    // --- Specular: dielectric GGX / Cook-Torrance ON TOP of the SH diffuse (Phase 1.6) -----
    // Unity-Standard-style spec lobe lit from the SH volume's own dominant light dir + color,
    // so the highlight is consistent with the baked GI (no new data). Dielectric: F0 = 0.04
    // (every material is metallic≈0). Roughness is per-material (glass=0.05 -> a sharp glint).
    // N is the Phase-2b normal-mapped shading normal computed at the top (back-face-flipped
    // geometric normal, perturbed by the tangent-space normal map when the material has one).
    let dom = sh_dominant_light(o.world_pos);
    var spec_rgb = vec3<f32>(0.0);
    if (dom.mag >= 1e-4) {
        let V = normalize(view.world_position.xyz - o.world_pos);
        let Ld = dom.dir;
        let H = normalize(V + Ld);
        let NdotL = max(dot(N, Ld), 0.0);
        let NdotV = max(dot(N, V), 1e-4);
        let NdotH = max(dot(N, H), 0.0);
        let VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0) {
            let rough = clamp(m.roughness, 0.03, 1.0);
            let a  = rough * rough;
            let a2 = a * a;
            let d  = (NdotH * NdotH * (a2 - 1.0) + 1.0);
            let D  = a2 / (3.14159265 * d * d);                       // GGX/Trowbridge-Reitz NDF
            let k  = (rough + 1.0) * (rough + 1.0) / 8.0;             // Smith k (direct lighting)
            let gv = NdotV / (NdotV * (1.0 - k) + k);
            let gl = NdotL / (NdotL * (1.0 - k) + k);
            let G  = gv * gl;                                         // Smith geometry
            let F  = 0.04 + 0.96 * pow(1.0 - VdotH, 5.0);            // Schlick, dielectric F0=0.04
            spec_rgb = dom.radiance * (D * G * F / (4.0 * NdotV * NdotL + 1e-4)) * NdotL * SPEC_STRENGTH;
        }
    }

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
    let is_water = (m.flags & MAT_FLAG_WATER) != 0u;
    if (is_softcutout) {
        let coverage = clamp(o.color.a * m.vp.x - (m.vp.y - m.vp.z), 0.0, 1.0);
        return vec4<f32>(lit + spec_rgb, coverage);
    } else if (is_water) {
        // Dark wet sheen, modulated by GI so it isn't a flat slab decoupled from the
        // scene lighting; the GGX spec on top is what actually reads as a wet/mirror glint.
        return vec4<f32>(vec3<f32>(0.03, 0.04, 0.05) * gi + spec_rgb, 0.35);
    }
    // Glass / plain decal: spec is what makes glass read as shiny.
    return vec4<f32>(lit + spec_rgb, albedo.a);
#else
    // OPAQUE pass: fully opaque (blend materials were already discarded above).
    return vec4<f32>(lit + spec_rgb, 1.0);
#endif
}
