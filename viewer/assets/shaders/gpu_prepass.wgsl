// NORMAL PREPASS — geometric world normal + roughness into an Rgba16Float target, with its own
// 1x Depth32Float. The forward main pass writes only color, so every screen-space effect here was
// depth-only: SSAO reconstructed FACE normals from depth derivatives (faceted on curved surfaces,
// haloed at edges), and SSR was impossible. This pass is the enabler: it re-draws the OPAQUE scene
// through the SAME camera-culled `visible[]` + indirect buffers the main pass consumes (nothing is
// re-culled), writing per-pixel `vec4(world_normal, roughness)` that SSAO reads today and SSR can
// read next.
//
// GEOMETRIC normal, deliberately not normal-mapped: SSAO over normal-mapped normals turns surface
// detail into AO noise (the industry default is the geometric/vertex normal), and the one surface
// whose shading normal really matters to SSR — water — computes its ripple normal in its own branch
// anyway. If SSR later wants mapped normals on walls/roads, that is a v2 of this pass (sample the
// normal map here), not a different architecture.
//
// Modeled on gpu_shadow.wgsl (the other "same buffers, different camera" pass):
//   group(0) = ssbo_layout  (instances + visible), SAME resource as the main draw's group(1).
//   group(1) = PrepassUniform (camera world->clip + params).
//   group(2) = material_layout, SAME resource as the main draw's group(2): the material table for
//              the BLEND/CUTOUT alpha discipline, the bindless albedo array for the alpha test.
//
// GRASS is excluded twice (belt & braces, same as shadows): the Rust node skips the grass mesh
// range in the multidraw, and the vertex guard below degenerates any grass instance that slips
// through. AO at grass-blade scale is noise, and the 11.5M-clump fragment bill is real.

struct InstanceGpu {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    ids: vec4<u32>,
    sphere: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> instances: array<InstanceGpu>;
@group(0) @binding(1) var<storage, read> visible: array<u32>;

// group(1): byte-identical to the Rust `PrepassUniform` (80 B: mat4 + vec4).
struct PrepassUniform {
    // world -> camera clip (Bevy reverse-z perspective). Column-major Mat4 upload.
    view_proj: mat4x4<f32>,
    // x = albedo_tex bindless array length (descriptor-index clamp — WGSL has no arrayLength for a
    //     binding_array, so the count is uploaded, exactly like the shadow pass). yzw pad.
    params: vec4<f32>,
};
@group(1) @binding(0) var<uniform> prepass: PrepassUniform;

// group(2): SAME material table + bindless albedo as the main draw. Byte-identical MaterialGpu —
// the array STRIDE must be the full 192-byte record or every material index > 0 reads garbage and
// the misdecoded albedo_index becomes an out-of-range bindless descriptor, which FAULTS AMD
// hardware (two field device-losses). Keep in lockstep with the Rust `GpuMaterial` (192 B,
// asserted at gpu_driven.rs), gpu_draw.wgsl and gpu_shadow.wgsl — shader_material_stride.rs pins.
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
    _detail_idx: vec4<u32>,     // @80  stride only (see gpu_shadow.wgsl for the full story)
    _detail_auv: vec4<f32>,     // @96
    _detail_nuv: vec4<f32>,     // @112
    _detail_par: vec4<f32>,     // @128
    _detail_mg: vec4<f32>,      // @144
    _emissive: vec4<u32>,       // @160
    _parallax: vec4<u32>,       // @176 -> 192 B total
};
@group(2) @binding(0) var<storage, read> materials: array<MaterialGpu>;
@group(2) @binding(1) var albedo_tex: binding_array<texture_2d<f32>>;
@group(2) @binding(2) var albedo_samp: sampler;

const MAT_ALBEDO_NONE: u32 = 0xFFFFFFFFu;
const MAT_FLAG_CUTOUT: u32 = 1u;
const MAT_FLAG_BLEND: u32 = 2u;
const MAT_FLAG_WATER: u32 = 8u;           // bit3 — the class channel marks water pixels for SSR/TAA
const MAT_FLAG_RFA: u32 = 64u;            // bit6 — roughness = 1 - RAW tex.a (Unity smoothness-in-alpha)

// Verbatim from gpu_draw.wgsl: octahedral decode of the Snorm16x2 vertex normal…
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let s = vec2<f32>(select(-1.0, 1.0, n.x >= 0.0), select(-1.0, 1.0, n.y >= 0.0));
        let xy = (vec2<f32>(1.0) - abs(vec2<f32>(n.y, n.x))) * s;
        n = vec3<f32>(xy.x, xy.y, n.z);
    }
    return normalize(n);
}

// …and the cofactor normal transform (det·inverse-transpose): correct under shear / non-uniform
// scale / mirror with no decompose. Same math, same reasons (tarkov-unity-extraction skill §1).
fn cofactor(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(cross(c1, c2), cross(c2, c0), cross(c0, c1));
}

struct PrepassVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) material_index: u32,
    @location(2) world_normal: vec3<f32>,
};

@vertex
fn vertex(
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_index: u32,
    @builtin(instance_index) ii: u32,
) -> PrepassVOut {
    // B5: clamp both indirections, same as gpu_draw/gpu_shadow.
    let vi = min(ii, arrayLength(&visible) - 1u);
    let inst = instances[min(visible[vi], arrayLength(&instances) - 1u)];
    // Grass never enters the prepass (see header). Degenerate + beyond far clip = zero fragments.
    if (inst.ids.z == 1u) {
        var g: PrepassVOut;
        g.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        g.uv = vec2<f32>(0.0);
        g.material_index = 0u;
        g.world_normal = vec3<f32>(0.0, 1.0, 0.0);
        return g;
    }
    let col0 = vec3<f32>(inst.m0.x, inst.m1.x, inst.m2.x);
    let col1 = vec3<f32>(inst.m0.y, inst.m1.y, inst.m2.y);
    let col2 = vec3<f32>(inst.m0.z, inst.m1.z, inst.m2.z);
    let t = vec3<f32>(inst.m0.w, inst.m1.w, inst.m2.w);
    let world = mat3x3<f32>(col0, col1, col2) * position + t;

    var o: PrepassVOut;
    o.clip = prepass.view_proj * vec4<f32>(world, 1.0);
    o.uv = uv;
    o.material_index = material_index;
    o.world_normal = normalize(cofactor(col0, col1, col2) * oct_decode(normal_oct));
    return o;
}

struct PrepassOut {
    @location(0) normal_rough: vec4<f32>,
};

@fragment
fn fragment(o: PrepassVOut, @builtin(front_facing) front: bool) -> PrepassOut {
    // B5: clamp the material index (same discipline as every other pass).
    let m = materials[min(o.material_index, arrayLength(&materials) - 1u)];

    // Derivatives in UNIFORM control flow, before any discard (naga uniformity rule).
    let duv_dx = dpdx(o.uv);
    let duv_dy = dpdy(o.uv);

    // BLEND surfaces (decals/water/glass) are not part of the opaque normal buffer: their normals
    // belong to whatever opaque surface is BEHIND them, which is what SSAO/SSR should see.
    if ((m.flags & MAT_FLAG_BLEND) != 0u) {
        discard;
    }
    // CUTOUT alpha test, or foliage/fences write solid card normals (same rule as the shadow pass).
    let n_tex = u32(max(prepass.params.x, 0.0));
    if ((m.flags & MAT_FLAG_CUTOUT) != 0u && m.albedo_index != MAT_ALBEDO_NONE && n_tex > 0u) {
        // CLAMP the bindless descriptor index — an OOB binding_array index faults AMD outright.
        let idx = min(m.albedo_index, n_tex - 1u);
        let a = textureSampleGrad(albedo_tex[idx], albedo_samp, o.uv, duv_dx, duv_dy).a * m.tint.a;
        if (a < m.alpha_cutoff) {
            discard;
        }
    }

    var N = normalize(o.world_normal);
    if (!front) {
        N = -N; // double-sided shells/mirrors: same flip as the main pass
    }
    // ROUGHNESS, forward-matching (Phase 1 acceptance: "deep-water pixels contain ... forward-
    // matching roughness"). 82% of materials are RFA — Unity Standard smoothness in the RAW
    // texture alpha — and the first prepass shipped the per-material constant instead, which
    // would have fed SSR wrong roughness across most of the scene (Codex review finding, verified
    // against gpu_draw.wgsl's per-pixel block). Same clamps, same water floor as the forward pass.
    var rough = clamp(m.roughness, 0.03, 1.0);
    if ((m.flags & MAT_FLAG_RFA) != 0u && m.albedo_index != MAT_ALBEDO_NONE && n_tex > 0u) {
        let ridx = min(m.albedo_index, n_tex - 1u);
        let raw_a = textureSampleGrad(albedo_tex[ridx], albedo_samp, o.uv, duv_dx, duv_dy).a;
        rough = clamp(1.0 - raw_a, 0.06, 1.0);
    }
    let is_water = (m.flags & MAT_FLAG_WATER) != 0u;
    if (is_water) {
        rough = max(rough, 0.10);
    }
    // CLASS CHANNEL, encoded in .w so no second target is needed:
    //   w in [0,1]   = static opaque, value IS the roughness
    //   w in [2,3]   = WATER, roughness = w - 2 (SSR must use the animated water normal path;
    //                  TAA treats it as reactive)
    //   w == 0 exact = no prepass data (the clear) — consumers fall back / reject history
    // Grass and blend surfaces never get here (discarded/degenerated), so they read as no-data.
    var w = max(rough, 1.0e-3); // never exactly 0 for real geometry — 0 is the no-data sentinel
    if (is_water) {
        w = rough + 2.0;
    }
    var out: PrepassOut;
    out.normal_rough = vec4<f32>(N, w);
    return out;
}
