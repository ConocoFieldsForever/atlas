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
// group(2) = M3 material table + bindless albedo array + sampler (see Rust `material_layout`).

#import bevy_pbr::view_transformations::position_world_to_clip

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
// 48 bytes, 16-aligned. Matches the Rust `MaterialGpu` POD.
//   albedo_index : index into `albedo_tex`; 0xFFFFFFFF sentinel = untextured (use tint/white).
//   flags        : bit0 = cutout (role=cutout / alphaMode=MASK) -> discard when alpha < cutoff.
//                  bit1 = blend  (role âˆˆ {decal,glass,water} / alphaMode=BLEND) -> BLEND pass.
//   alpha_cutoff : PER-MATERIAL cutoff (NOT a global 0.5).
//   uv_xform     : (sx,sy,ox,oy) REFERENCE ONLY â€” tiling is already baked into the vertex UVs
//                  (manifest.conventions.uvTilingBaked=true). Do NOT apply it (double-tile trap).
//   tint         : linear rgba; rgb multiplies albedo. a is the BLEND-pass opacity (M3b1: used).
struct MaterialGpu {
    albedo_index: u32,
    flags: u32,
    alpha_cutoff: f32,
    _pad: u32,
    uv_xform: vec4<f32>,
    tint: vec4<f32>,
};

const MAT_ALBEDO_NONE: u32 = 0xFFFFFFFFu; // sentinel: material has no albedo texture
const MAT_FLAG_CUTOUT: u32 = 1u;          // bit0: alpha-tested (MASK) surface
const MAT_FLAG_BLEND: u32 = 2u;           // bit1: alpha-blended (decal/glass/water/BLEND) surface

@group(2) @binding(0) var<storage, read> materials: array<MaterialGpu>;
@group(2) @binding(1) var albedo_tex: binding_array<texture_2d<f32>>;
@group(2) @binding(2) var albedo_samp: sampler;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_index: u32, // Uint32; per-submesh global materialId, tagged in build_cpu_data
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) material_index: u32,
};

// cofactor(linear 3x3) = det Â· inverse-transpose; columns = cross products of the linear
// columns. Correct normal transform under shear / non-uniform scale / mirror, no decompose.
fn cofactor(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(cross(c1, c2), cross(c2, c0), cross(c0, c1));
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

    // Pass class-discard (M3b1) — AFTER the sample above (Codex P0). Each pass keeps only its
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

    // --- Lighting (flat sun + ambient; full lighting deferred to M3b) ------------
    var n = normalize(o.world_normal);
    if (!front) {
        n = -n;   // double-sided: flip for back faces (inward shells / mirrors)
    }
    let key = normalize(vec3<f32>(0.4, 1.0, 0.3));
    let ndl = clamp(dot(n, key), 0.0, 1.0);
    let ambient = 0.28;
    let lit = albedo.rgb * (ambient + (1.0 - ambient) * ndl);

#ifdef BLEND_PASS
    // BLEND pass: emit the REAL computed opacity. albedo.a = tex.a*tint.a (textured) or tint.a
    // (untextured water/glass/decal). Non-premultiplied to match the pipeline's
    // BlendState::ALPHA_BLENDING (src=SrcAlpha, dst=OneMinusSrcAlpha), i.e. Unity _Color*_MainTex.
    return vec4<f32>(lit, albedo.a);
#else
    // OPAQUE pass: fully opaque (blend materials were already discarded above).
    return vec4<f32>(lit, 1.0);
#endif
}
