// eft::gpu_cull â€” M2 GPU-driven compute frustum-cull + per-mesh compaction.
//
// One-source-of-truth compute shader for the M2 path (render/gpu_driven.rs). Two
// entry points, dispatched as SEPARATE compute passes each frame (separate passes =
// automatic wgpu barrier between them, so cs_reset's writes are visible to cs_cull):
//
//   cs_reset  â€” one thread per MESH: rewrite that mesh's DrawIndexedIndirectArgs from
//               its static meshMeta (index_count/first_index/base_vertex/instance_base)
//               and atomicStore instance_count = 0. Regenerating every field each frame
//               means the indirect buffer never needs CPU initialization and has no
//               stale-data hazard.
//   cs_cull   â€” one thread per INSTANCE: test the PRECOMPUTED conservative world
//               bounding sphere against the 6 frustum planes; survivors do
//               slot = atomicAdd(indirect[meshId].instance_count, 1) and write their
//               global instance index into visible[instance_base + slot].
//
// Because the pack stores instances GROUPED-BY-MESH and CONTIGUOUS, each mesh owns the
// static region [instance_base, instance_base+instance_count); first_instance =
// instance_base is a compile-time constant (from meshMeta), so NO global prefix-sum is
// needed. The draw shader (gpu_draw.wgsl) reads visible[@builtin(instance_index)]
// directly â€” @builtin(instance_index) already includes first_instance.
//
// The #1 rule (tarkov-unity-extraction): the world sphere was built on the CPU from the
// FULL 3x4 affine using the CONSERVATIVE Frobenius-norm radius scale ||L||_F (a guaranteed
// >= operator-norm upper bound) â€” NEVER max-column-norm (a lower bound that underestimates
// under shear and wrongly culls visible geometry). No decompose.

struct InstanceGpu {
    m0: vec4<f32>,      // ROW-MAJOR world 3x4 affine, row 0 (incl shear+mirror)
    m1: vec4<f32>,      // row 1
    m2: vec4<f32>,      // row 2
    ids: vec4<u32>,     // x=mesh_id  y=flags  z=class (1=grass -> bigger screen-size cull)  w=pad
    sphere: vec4<f32>,  // xyz = world-space center, w = conservative world radius
};

struct MeshMeta {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    instance_base: u32,
    instance_count: u32,
    blend_class: u32,   // 0 = opaque-only, 1 = blend-only, 2 = mixed (draws in both passes)
    pad1: u32,
    pad2: u32,
};

// wgpu DrawIndexedIndirectArgs â€” instance_count is atomic so cs_cull can bump it.
struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct CullGlobals {
    frustum: array<vec4<f32>, 6>,   // 6 world planes, NORMALIZED, inward (visible: dot(n,c)+w >= -r)
    counts: vec4<u32>,              // x=instance_count  y=mesh_count  z=bitcast f32 k_grass  w=pad
    // Screen-size cull anchor: xyz = camera world pos, w = k_general where
    // k = min_px / (0.5 * viewport_h * proj11). Cull when sphere.w < k * distance(cam, center)
    // (the sphere subtends fewer than min_px pixels). Zeros = disabled (frame-0 seed).
    cam_k: vec4<f32>,
};

@group(0) @binding(0) var<uniform> G: CullGlobals;
@group(0) @binding(1) var<storage, read>        instances: array<InstanceGpu>;
@group(0) @binding(2) var<storage, read>        mesh_meta: array<MeshMeta>;
@group(0) @binding(3) var<storage, read_write>  visible: array<u32>;
@group(0) @binding(4) var<storage, read_write>  indirect: array<DrawArgs>;        // P1 opaque (+ shadow casters)
@group(0) @binding(5) var<storage, read_write>  indirect_blend: array<DrawArgs>;  // P2 per-mesh blend draws

@compute @workgroup_size(64)
fn cs_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= G.counts.y) { return; }
    let mm = mesh_meta[m];
    // Class-split indirect args: the OPAQUE buffer zeroes blend-only meshes (P1 + the shadow
    // casters skip them entirely); the BLEND buffer zeroes opaque-only meshes. Mixed meshes
    // keep their full index run in BOTH (the fragment class-discard splits the materials).
    let opaque_count = select(mm.index_count, 0u, mm.blend_class == 1u);
    let blend_count  = select(0u, mm.index_count, mm.blend_class != 0u);
    indirect[m].index_count = opaque_count;
    indirect[m].first_index = mm.first_index;
    indirect[m].base_vertex = mm.base_vertex;
    indirect[m].first_instance = mm.instance_base;   // static per-mesh region base
    atomicStore(&indirect[m].instance_count, 0u);
    indirect_blend[m].index_count = blend_count;
    indirect_blend[m].first_index = mm.first_index;
    indirect_blend[m].base_vertex = mm.base_vertex;
    indirect_blend[m].first_instance = mm.instance_base;
    atomicStore(&indirect_blend[m].instance_count, 0u);
}

fn sphere_visible(center: vec3<f32>, radius: f32) -> bool {
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let p = G.frustum[i];
        if (dot(p.xyz, center) + p.w < -radius) {
            return false;
        }
    }
    return true;
}

#ifdef CULL_COMPUTE_SPHERE
// Optional GPU-side world sphere, used ONLY when the CPU precompute is disabled (in that
// mode `sphere` carries the LOCAL center/radius instead of the world sphere). The radius
// scale is the Frobenius norm of the linear 3x3, ||L||_F = sqrt(|c0|^2+|c1|^2+|c2|^2):
// a GUARANTEED upper bound on the operator norm (sigma_max <= ||L||_F <= sqrt(3)*sigma_max),
// so it NEVER under-culls. NEVER use max-column-norm (a lower bound). Mirrors the CPU
// `gpu_driven::conservative_radius_scale`.
fn world_sphere_from_affine(inst: InstanceGpu) -> vec4<f32> {
    let c0 = vec3<f32>(inst.m0.x, inst.m1.x, inst.m2.x);
    let c1 = vec3<f32>(inst.m0.y, inst.m1.y, inst.m2.y);
    let c2 = vec3<f32>(inst.m0.z, inst.m1.z, inst.m2.z);
    let t  = vec3<f32>(inst.m0.w, inst.m1.w, inst.m2.w);
    let lin = mat3x3<f32>(c0, c1, c2);
    let center = lin * inst.sphere.xyz + t;                  // sphere.xyz = LOCAL center
    let frob = sqrt(dot(c0, c0) + dot(c1, c1) + dot(c2, c2));
    return vec4<f32>(center, inst.sphere.w * frob);          // sphere.w = LOCAL radius
}
#endif

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= G.counts.x) { return; }
    let inst = instances[i];
#ifdef CULL_COMPUTE_SPHERE
    let sphere = world_sphere_from_affine(inst);
#else
    let sphere = inst.sphere;   // CPU-precomputed conservative world sphere (default path)
#endif
    if (!sphere_visible(sphere.xyz, sphere.w)) { return; }

    // Screen-size cull: drop instances whose bounding sphere subtends fewer than min_px pixels
    // (grass uses a larger threshold — 100k+ ~1.3 m clumps are invisible way before the far
    // plane and dominated the draw cost). k==0 (frame-0 seed / EFT_CULL_PX=0) disables.
    let k = select(G.cam_k.w, bitcast<f32>(G.counts.z), inst.ids.z == 1u);
    if (k > 0.0) {
        let d = max(distance(G.cam_k.xyz, sphere.xyz), 1e-3);
        if (sphere.w < k * d) { return; }
    }

    // B5: clamp the instance-supplied mesh id before it indexes mesh_meta / indirect / indirect_blend
    // (all sized == mesh_count). Well-formed packs are always in-bounds (no-op); a malformed id must
    // not read/write out of bounds — AMD returns garbage on OOB (NVIDIA returns 0), which would let a
    // stray instance corrupt an unrelated mesh's draw args.
    let mesh_id = min(inst.ids.x, arrayLength(&mesh_meta) - 1u);
    let base = mesh_meta[mesh_id].instance_base;
    // The OPAQUE buffer's counter is the CANONICAL slot allocator for visible[]; the blend
    // buffer's counter converges to the same total (same survivors), so both passes read the
    // identical visible[base .. base+count) range.
    let slot = atomicAdd(&indirect[mesh_id].instance_count, 1u);
    atomicAdd(&indirect_blend[mesh_id].instance_count, 1u);
    // B5: clamp the compaction write index into visible[] (sized == instance_total). In-bounds for
    // well-formed packs; the clamp only guards a corrupt base/slot from stomping foreign memory.
    let vi = min(base + slot, arrayLength(&visible) - 1u);
    visible[vi] = i;
}
