// #5 SUN SHADOW PASS — depth-only (+ a minimal alpha-aware fragment). Reuses the camera-culled
// `visible[]` + instance buffer (group(0), SAME layout as the main draw's group(1)) and the shared
// interleaved vertex buffer READ-ONLY, but transforms into the sun's orthographic light space
// instead of the camera. The pipeline writes only depth into one cascade layer of the shadow map;
// the main draw (gpu_draw.wgsl) then projects each fragment's world position into this same light
// space and PCF-compares against this depth to derive a bounded contact-shadow term.
//
// WHY A FRAGMENT STAGE (depth-only would normally need none): this scene appends 109k alpha-cutout
// grass cross-quads (gpu_driven.rs:942) and BLEND decals/water/glass. Without a discard-capable
// fragment, grass casts SOLID rectangular card shadows and blend surfaces cast their full geometry.
// So the fragment discards BLEND materials outright and alpha-tests CUTOUT materials, exactly like
// the main opaque pass, using textureSampleGrad with gradients taken in uniform control flow.
//
// GROUPS (match the Rust shadow pipeline layout):
//   group(0) = ssbo_layout  (instances + visible), SAME resource as the main draw's group(1).
//   group(1) = ShadowCascadeUniform (this cascade's world->light-clip matrix + Lsun/texel).
//   group(2) = material_layout (material table + bindless albedo + sampler), SAME resource as the
//              main draw's group(2). Only bindings 0/1/2 are referenced here; the extra bindings
//              (normal array, terrain splat) are present in the bind group but simply unused.

struct InstanceGpu {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    ids: vec4<u32>,
    sphere: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> instances: array<InstanceGpu>;
@group(0) @binding(1) var<storage, read> visible: array<u32>;

// group(1): per-cascade uniform. Byte-identical to the Rust `ShadowCascadeUniform`.
struct ShadowCascadeUniform {
    // world -> sun light clip (conventional 0..1-depth orthographic). Column-major Mat4 upload.
    view_proj: mat4x4<f32>,
    // xyz = Lsun (points TOWARD the sun; light travels along -Lsun), w = 1/shadow_map_size (PCF texel).
    dir_texel: vec4<f32>,
};
@group(1) @binding(0) var<uniform> cascade: ShadowCascadeUniform;

// group(2): the SAME material table + bindless albedo array + sampler as the main draw. Byte-identical
// MaterialGpu (only the leading fields we actually read are used, but the struct must match the SSBO
// stride so `materials[o.material_index]` indexes correctly).
struct MaterialGpu {
    albedo_index: u32,
    flags: u32,
    alpha_cutoff: f32,
    roughness: f32,
    uv_xform: vec4<f32>,
    tint: vec4<f32>,
    vp: vec4<f32>,
    normal_index: u32,
    normal_flags: u32,
    normal_scale: f32,
    _pad2: u32,
};
@group(2) @binding(0) var<storage, read> materials: array<MaterialGpu>;
@group(2) @binding(1) var albedo_tex: binding_array<texture_2d<f32>>;
@group(2) @binding(2) var albedo_samp: sampler;

const MAT_ALBEDO_NONE: u32 = 0xFFFFFFFFu; // sentinel: material has no albedo texture
const MAT_FLAG_CUTOUT: u32 = 1u;          // bit0: alpha-tested (MASK) surface -> alpha-test the caster
const MAT_FLAG_BLEND: u32 = 2u;           // bit1: alpha-blended (decal/glass/water) -> never cast solid

struct ShadowVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) material_index: u32,
};

// Vertex layout is position-only-plus (loc0 pos @0, loc2 uv @24, loc3 material @32; stride 52). The
// interleaved buffer's normal (@12) and color (@36) attrs are simply not declared on this pipeline.
@vertex
fn vertex(
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_index: u32,
    @builtin(instance_index) ii: u32,
) -> ShadowVOut {
    let inst = instances[visible[ii]];
    let col0 = vec3<f32>(inst.m0.x, inst.m1.x, inst.m2.x);
    let col1 = vec3<f32>(inst.m0.y, inst.m1.y, inst.m2.y);
    let col2 = vec3<f32>(inst.m0.z, inst.m1.z, inst.m2.z);
    let t = vec3<f32>(inst.m0.w, inst.m1.w, inst.m2.w);
    let world = mat3x3<f32>(col0, col1, col2) * position + t;

    var o: ShadowVOut;
    o.clip = cascade.view_proj * vec4<f32>(world, 1.0);
    o.uv = uv;
    o.material_index = material_index;
    return o;
}

// Alpha-aware caster fragment. No color target (depth-only). Derivatives are taken at top level in
// uniform control flow; the CUTOUT sample uses textureSampleGrad with those explicit gradients so it
// stays valid even after the (non-uniform) BLEND discard — same naga-uniformity discipline as the
// main shader's terrain branch.
@fragment
fn fragment(o: ShadowVOut) {
    let m = materials[o.material_index];

    let duv_dx = dpdx(o.uv);
    let duv_dy = dpdy(o.uv);

    // BLEND surfaces (decals/water/glass) must not cast their full geometry as shadow.
    if ((m.flags & MAT_FLAG_BLEND) != 0u) {
        discard;
    }

    // CUTOUT surfaces (grass/foliage): alpha-test the caster so gaps between blades don't write
    // depth (otherwise 109k grass cross-quads project solid rectangular cards).
    if ((m.flags & MAT_FLAG_CUTOUT) != 0u) {
        let idx = select(0u, m.albedo_index, m.albedo_index != MAT_ALBEDO_NONE);
        let a = textureSampleGrad(albedo_tex[idx], albedo_samp, o.uv, duv_dx, duv_dy).a * m.tint.a;
        if (a < m.alpha_cutoff) {
            discard;
        }
    }
    // Opaque (and passing-cutout) fragments fall through: the pipeline writes depth normally.
}
