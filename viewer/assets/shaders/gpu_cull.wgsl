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
    ids: vec4<u32>,     // x=mesh_id  y=flags  z=class(1=grass)+lod bits(8=is_default,9..12=lod_index,13..31=lod_group id)  w=lod window (pack2x16float(near',far'); 0=sentinel/always-draw)
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
    // Distance-LOD: x=proj11, y=lod_bias, z=mode (0=max detail, 1=distance, 2=force shell w), w=forced shell.
    lod_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> G: CullGlobals;
@group(0) @binding(1) var<storage, read>        instances: array<InstanceGpu>;
@group(0) @binding(2) var<storage, read>        mesh_meta: array<MeshMeta>;
@group(0) @binding(3) var<storage, read_write>  visible: array<u32>;
@group(0) @binding(4) var<storage, read_write>  indirect: array<DrawArgs>;        // P1 opaque (+ shadow casters)
@group(0) @binding(5) var<storage, read_write>  indirect_blend: array<DrawArgs>;  // P2 per-mesh blend draws
@group(0) @binding(6) var<storage, read>        lod_centers: array<vec4<f32>>;    // B1 per-group ref center (indexed by ids.z>>13)

// LINEAR INVOCATION INDEX ACROSS A 2-D DISPATCH.
//
// `max_compute_workgroups_per_dimension` is 65,535 on essentially every adapter — a hard Vulkan /
// D3D limit, not a soft wgpu default. At @workgroup_size(64) a 1-D dispatch therefore tops out at
// 4,194,240 invocations, and woods ships 11,572,828 instances once its grass is built (883 MiB of
// them). Dispatching 180,826 groups in X was a validation error: wgpu invalidated the encoder and
// every later pass reported only "Encoder is invalid", so the map rendered at 2 fps with no
// message naming the real cause.
//
// The host now splits the group count over X and Y (see `dispatch_2d` in gpu_driven.rs) and each
// entry point reconstructs its index from BOTH. The X stride is read from `num_workgroups` rather
// than passed in a uniform, so the shader and the host cannot drift apart — there is no new struct
// field to keep in sync.
fn linear_index(gid: vec3<u32>, ng: vec3<u32>) -> u32 {
    return gid.y * (ng.x * 64u) + gid.x;
}

@compute @workgroup_size(64)
fn cs_reset(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) ng: vec3<u32>) {
    let m = linear_index(gid, ng);
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
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(num_workgroups) ng: vec3<u32>) {
    let i = linear_index(gid, ng);
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
    let is_grass = inst.ids.z == 1u;
    let k = select(G.cam_k.w, bitcast<f32>(G.counts.z), is_grass);
    if (k > 0.0) {
        let d = max(distance(G.cam_k.xyz, sphere.xyz), 1e-3);
        if (sphere.w < k * d) { return; }
        // GRASS DISTANCE CLAMP (counts.w, 0 = off). The screen-size test above is already a distance
        // test — it rejects past `sphere.w / k` — but `k` is derived as px/(0.5·viewport_h·proj11),
        // so the grass horizon SCALES WITH RESOLUTION AND FOV: 1080p -> 1440p is 1.33x viewport
        // height, which pushes grass 33% further out for the same setting and costs proportionally
        // more. It also varies per grass KIND, because the cull distance depends on each clump's
        // bounding radius and woods ships 15 kinds. A metre limit is resolution-independent,
        // FOV-independent, uniform across kinds, and is a number a user can reason about.
        if (is_grass) {
            let lim = bitcast<f32>(G.counts.w);
            if (lim > 0.0 && d > lim) { return; }
        }
    }

    // Distance-LOD shell selection. ids.w == 0 is the sentinel (always draw): lean packs, ungrouped
    // instances, and single-present-shell groups. Only multi-shell instances carry a window.
    if (inst.ids.w != 0u) {
        let mode = u32(G.lod_params.z);
        if (mode == 0u) {
            // Max detail (default / today's look): draw only the default (finest-present) shell.
            if ((inst.ids.z & 256u) == 0u) { return; }
        } else if (mode == 1u) {
            // Distance-based: draw shell i iff dist in (near'_i, far'_i] * proj11 * bias. B1: measure
            // from the group's SHARED reference center (all shells/renderers of a group switch as a
            // unit) — NOT this shell's own bounding-sphere centroid, which differs per LOD mesh and
            // would leave a double-draw/hole seam at every boundary. Group id rides ids.z bits 13+.
            let ab = unpack2x16float(inst.ids.w);
            let gid = min(inst.ids.z >> 13u, arrayLength(&lod_centers) - 1u);
            let d = max(distance(G.cam_k.xyz, lod_centers[gid].xyz), 1e-3);
            let m = G.lod_params.x * G.lod_params.y;
            var lo = ab.x * m;
            var hi = ab.y * m;
            // STAGGERED transition, using the game's own fade width (lod_centers[gid].w, derived from
            // Unity's ftw/srh — see the CPU side). Without this, every instance in a group swaps shell
            // at the SAME distance, so a stand of trees pops in unison and reads as a glitch.
            //
            // Each instance jitters its own boundaries by up to +-half the band, hashed from its index
            // so the offset is FIXED for that instance: it must not change frame to frame, or the
            // instance would flicker between shells while the camera holds still. The boundaries of
            // adjacent shells move together (both derive from the same hash and band), so the windows
            // still tile and no distance is left undrawn or double-drawn.
            //
            // This is a stagger, NOT Unity's alpha cross-fade: a true dithered fade needs a per-visible
            // -instance fade weight passed from this shader to the fragment stage, which means a new
            // storage buffer in the draw's bind group. Deliberately not done here.
            let band = lod_centers[gid].w;
            if (band > 0.0) {
                // Hash the GROUP id — NOT the instance index — to a stable [-0.5, 0.5).
                //
                // BUG FIX (user-visible as "objects disappear briefly when zooming"): the first
                // version hashed `i`, the instance index. A group's shells are DIFFERENT instances,
                // so LOD0 and LOD1 drew with different jitters: LOD0 stopped at far*(1+b*j0) while
                // LOD1 started at the same boundary scaled by j1 != j0 — leaving a distance band
                // where NEITHER shell drew (or both did). A zoom changes proj11, which sweeps every
                // instance's d*m metric through its boundaries, so the gap crossed the screen as a
                // blink. Keying the hash on the group id gives every shell of a group the SAME
                // offset, so their windows tile exactly again — which is what the comment below
                // always claimed. The stagger still varies BETWEEN groups, which is its whole job.
                var h = (inst.ids.z >> 13u) * 747796405u + 2891336453u;
                h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
                let j = f32((h >> 22u) ^ h) * (1.0 / 4294967296.0) - 0.5;
                lo = lo * (1.0 + band * j);
                hi = hi * (1.0 + band * j);
            }
            if (!(d > lo && d <= hi)) { return; }
        } else {
            // Force a single shell index (debug).
            if (((inst.ids.z >> 9u) & 15u) != u32(G.lod_params.w)) { return; }
        }
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

// ---------------------------------------------------------------------------------------------
// PER-INSTANCE TRANSPARENT ORDERING (cs_sort_blend)
//
// `cs_cull` compacts survivors with `slot = atomicAdd(...)`, so the order of instances INSIDE a
// mesh's visible[] run is whatever order the GPU's atomics happened to retire — and it differs
// every frame. That is invisible for opaque geometry (depth sorts it) but the Blend pass draws
// with depth-write OFF, so the composite of two overlapping panes of the SAME mesh depends
// entirely on that order: two windows seen through each other FLICKERED between two shadings
// with the camera completely still (field report: Nikitskaya_2_Outdoor_Glass_04 instances 19539 +
// 19540 on interchange).
//
// Unity avoids this by keeping every transparent surface its own renderer and sorting them
// individually, back to front, each frame. Our GPU-driven path batches all instances of a mesh
// into ONE indirect draw, so we have to restore that ordering ourselves: after the cull, sort each
// blend mesh's visible run by distance from the camera, FARTHEST FIRST. One invocation per blend
// mesh (counts are tiny — 6,235 blend instances across 1,377 meshes on interchange, ~4.5 each), so
// a straight insertion sort is the right tool: no scratch memory, stable, and optimal on the
// nearly-sorted runs we re-sort every frame.
//
// Opaque draws read the same visible[] range and do not care about order, so reordering is safe
// for mixed-class meshes too.
@group(0) @binding(7) var<storage, read> blend_mesh_ids: array<u32>;

// Beyond this a single-threaded insertion sort would cost more than the artefact: leave the run in
// cull order (as before this pass existed). Nothing in the shipped packs comes close.
const SORT_MAX: u32 = 1024u;

@compute @workgroup_size(64)
fn cs_sort_blend(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(num_workgroups) ng: vec3<u32>) {
    let k = linear_index(gid, ng);
    if (k >= arrayLength(&blend_mesh_ids)) { return; }
    let mesh_id = min(blend_mesh_ids[k], arrayLength(&mesh_meta) - 1u);
    let base = mesh_meta[mesh_id].instance_base;
    // Survivor count written by cs_cull this frame (the opaque counter is the canonical allocator).
    let count = atomicLoad(&indirect[mesh_id].instance_count);
    if (count < 2u || count > SORT_MAX) { return; }
    let last = min(base + count, arrayLength(&visible)) - base;

    // Insertion sort visible[base .. base+last) by DESCENDING distance to the camera, so the
    // farthest pane is drawn first and nearer glass composites over it.
    for (var i: u32 = 1u; i < last; i = i + 1u) {
        let vi = visible[base + i];
        let ci = instances[min(vi, arrayLength(&instances) - 1u)].sphere.xyz;
        let di = distance(G.cam_k.xyz, ci);
        var j: i32 = i32(i) - 1;
        loop {
            if (j < 0) { break; }
            let vj = visible[base + u32(j)];
            let cj = instances[min(vj, arrayLength(&instances) - 1u)].sphere.xyz;
            if (distance(G.cam_k.xyz, cj) >= di) { break; }   // already farther: position found
            visible[base + u32(j) + 1u] = vj;
            j = j - 1;
        }
        visible[base + u32(j) + 1u] = vi;
    }
}
