// eft::cull — GPU-driven per-instance frustum (+ optional occlusion) cull with compaction.
// Ported MATH from tarkmap/out/_cullgl.js (per-instance world bounding sphere + sphere-vs-
// frustum) and tarkmap/out/_lod.js (screen-height LOD selection). Replaces the web CPU
// compaction + camera-motion throttle with a compute pass that compacts visible instances
// and fills DrawIndexedIndirect args for indirect multidraw.
//
// KEY MATH (no TRS-decompose — the skill's #1 rule):
//   world center = lin3x3 * meshBS.center + translation
//   world radius = meshBS.radius * max(|col0|, |col1|, |col2|)   // max linear-column norm =
//     shear / non-uniform-scale / mirror correct effective radius, applied WITHOUT decomposing.
//   visible  <=>  for all 6 frustum planes: dot(plane.xyz, center) + plane.w >= -radius
//
// COMPACTION MODEL: assemble_bevy groups instances by (mesh, sub_sig), so every meshId owns a
// CONTIGUOUS region [instance_base, instance_base+count) in instances.bin. We compact IN PLACE
// within each region: visible instances are written to out[instance_base + localSlot] where
// localSlot = atomicAdd(vis_count[meshId], 1). The compacted region stays contiguous, so one
// DrawIndexedIndirect per submesh (first_instance = instance_base, instance_count = vis_count)
// draws exactly the survivors. Non-visible tail slots are simply not drawn.
//
// PASSES (separate entry points, dispatched in order each frame):
//   cs_clear     — zero vis_count[] (one thread per mesh)
//   cs_cull      — one thread per instance: frustum(+occlusion+LOD) test -> compact
//   cs_indirect  — one thread per submesh-draw: write DrawIndexedIndirect from vis_count[mesh]

// -------------------------------------------------------------------------------------------
// Layouts. Input InstanceRaw mirrors instances.bin (affine 3x4 + meshId/lod/root/flags) plus a
// loader-filled material_id. Output InstanceGpu matches instanced.wgsl's instance vertex attrs
// (loc4..7): m0,m1,m2 (vec4 f32) + ids (vec4 u32 = mesh_id, material_id, flags, lod_packed).
// -------------------------------------------------------------------------------------------
struct InstanceRaw {
    m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,   // ROW-MAJOR world 3x4 affine (incl shear)
    ids0: vec4<u32>,   // x=mesh_id  y=lodGroup(bits, -1=none)  z=lodIndex(bits)  w=root_id
    ids1: vec4<u32>,   // x=flags    y=material_id              z,w=pad
};

struct InstanceGpu {
    m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,
    ids: vec4<u32>,    // x=mesh_id y=material_id z=flags w=(lodGroup<<16 | lodIndex&0xffff)
};

struct MeshBounds {
    center_radius: vec4<f32>,   // xyz local-space bounding-sphere center, w radius
    region: vec4<u32>,          // x=instance_base y=instance_count z,w=pad
};

// DrawIndexedIndirect (wgpu): index_count, instance_count, first_index, base_vertex(u32 bits), first_instance.
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: u32,
    first_instance: u32,
};

// per-submesh draw template (static, from the manifest): the index range + which mesh it belongs to.
struct SubmeshDraw {
    index_count: u32,
    first_index: u32,
    base_vertex: u32,
    mesh_id: u32,
};

// LOD group SoA (optional; _lod.js screen-height selection). center+size and the srh table.
struct LodGroup {
    center_size: vec4<f32>,   // xyz world center, w size(world)
    srh: vec4<u32>,           // x=srh_offset y=rcnt(renderable LOD count) z,w=pad
};

struct CullParams {
    frustum: array<vec4<f32>, 6>,   // 6 world-space planes, NORMALIZED (visible: dot(n,c)+w >= -r)
    camera: vec4<f32>,              // xyz camera pos, w = tan(fovV/2)
    lod: vec4<f32>,                 // x=lodBias y=clampCoarsest(0/1) z=lodEnable(0/1) w=pad
    counts: vec4<u32>,              // x=instance_count y=mesh_count z=submesh_count w=flags(bit0=occlusion)
};

@group(0) @binding(0) var<uniform> P: CullParams;
@group(0) @binding(1) var<storage, read>        instances_in: array<InstanceRaw>;
@group(0) @binding(2) var<storage, read>        mesh_bounds: array<MeshBounds>;
@group(0) @binding(3) var<storage, read_write>  instances_out: array<InstanceGpu>;
@group(0) @binding(4) var<storage, read_write>  vis_count: array<atomic<u32>>;   // per mesh
@group(0) @binding(5) var<storage, read_write>  draws: array<DrawIndexedIndirect>;
@group(0) @binding(6) var<storage, read>        submeshes: array<SubmeshDraw>;
// optional LOD
@group(0) @binding(7) var<storage, read>        lod_groups: array<LodGroup>;
@group(0) @binding(8) var<storage, read>        lod_srh: array<f32>;
// optional occlusion: hierarchical-Z depth pyramid (prev frame). Bound only when occlusion on.
@group(0) @binding(9) var hzb: texture_2d<f32>;
@group(0) @binding(10) var hzb_samp: sampler;

// max linear-column norm — the shear/mirror-correct effective scale (no decompose).
fn max_col_scale(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> f32 {
    return max(length(c0), max(length(c1), length(c2)));
}

fn frustum_visible(center: vec3<f32>, radius: f32) -> bool {
    for (var i: i32 = 0; i < 6; i = i + 1) {
        let pl = P.frustum[i];
        if (dot(pl.xyz, center) + pl.w < -radius) {
            return false;
        }
    }
    return true;
}

// _lod.js screen-height gate. Because assemble_bevy's LOD-shell dedup ships ONE (finest)
// instance per group, this collapses to a RENDERABILITY predicate: keep if the group is within
// any renderable LOD threshold (lev>=0), or always keep when clampCoarsest holds the world
// complete at distance. Full multi-LOD swap logic is kept in comments for future multi-LOD packs.
fn lod_visible(lod_group: i32) -> bool {
    if (P.lod.z < 0.5 || lod_group < 0) { return true; }       // LOD disabled or untagged -> keep
    let grp = lod_groups[u32(lod_group)];
    let dx = P.camera.xyz - grp.center_size.xyz;
    let dist = max(length(dx), 1e-4);
    let k = P.lod.x / (2.0 * P.camera.w);                       // lodBias / (2*tan(fovV/2))
    let Heff = grp.center_size.w * k / dist;
    let off = grp.srh.x; let rc = grp.srh.y;
    var lev: i32 = -1;
    for (var j: u32 = 0u; j < rc; j = j + 1u) {                 // srh descending -> first match = finest
        if (Heff >= lod_srh[off + j]) { lev = i32(j); break; }
    }
    // clampCoarsest: hold the coarsest renderable instead of popping the group out at distance.
    if (lev < 0 && P.lod.y > 0.5 && rc > 0u) { lev = i32(rc) - 1; }
    return lev >= 0;
    // Multi-LOD future: keep only if instance.lodIndex == resolveApplied(lev, avail, rc).
}

// Optional HZB occlusion. STUB: returns true (conservative) until the depth-pyramid pass is
// wired. Real test: project the world sphere to screen, sample the coarsest HZB mip covering
// its screen extent, keep if sphere front depth <= sampled max depth. See SHADER_PORT_MAP.md.
fn occlusion_visible(center: vec3<f32>, radius: f32) -> bool {
    return true;
}

// ------------------------------------------ passes ------------------------------------------

@compute @workgroup_size(64)
fn cs_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= P.counts.y) { return; }
    atomicStore(&vis_count[m], 0u);
}

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.counts.x) { return; }
    let inst = instances_in[i];
    let mesh_id = inst.ids0.x;

    // world bounding sphere (no decompose).
    let col0 = vec3<f32>(inst.m0.x, inst.m1.x, inst.m2.x);
    let col1 = vec3<f32>(inst.m0.y, inst.m1.y, inst.m2.y);
    let col2 = vec3<f32>(inst.m0.z, inst.m1.z, inst.m2.z);
    let translation = vec3<f32>(inst.m0.w, inst.m1.w, inst.m2.w);
    let bs = mesh_bounds[mesh_id];
    let lin = mat3x3<f32>(col0, col1, col2);
    let center = lin * bs.center_radius.xyz + translation;
    let radius = bs.center_radius.w * max_col_scale(col0, col1, col2);

    let lod_group = bitcast<i32>(inst.ids0.y);
    if (!lod_visible(lod_group)) { return; }
    if (!frustum_visible(center, radius)) { return; }
    if ((P.counts.w & 1u) != 0u && !occlusion_visible(center, radius)) { return; }

    // compact into this mesh's contiguous region.
    let slot = atomicAdd(&vis_count[mesh_id], 1u);
    let out_idx = bs.region.x + slot;

    var o: InstanceGpu;
    o.m0 = inst.m0; o.m1 = inst.m1; o.m2 = inst.m2;
    let lod_packed = (inst.ids0.y << 16u) | (inst.ids0.z & 0xffffu);
    o.ids = vec4<u32>(mesh_id, inst.ids1.y, inst.ids1.x, lod_packed);
    instances_out[out_idx] = o;
}

@compute @workgroup_size(64)
fn cs_indirect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= P.counts.z) { return; }
    let sm = submeshes[d];
    let vis = atomicLoad(&vis_count[sm.mesh_id]);
    let base = mesh_bounds[sm.mesh_id].region.x;
    var out: DrawIndexedIndirect;
    out.index_count = sm.index_count;
    out.instance_count = vis;                 // survivors of the cull for this mesh
    out.first_index = sm.first_index;
    out.base_vertex = sm.base_vertex;
    out.first_instance = base;                // start of this mesh's compacted region
    draws[d] = out;
}
