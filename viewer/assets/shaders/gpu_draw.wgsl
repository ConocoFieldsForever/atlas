// eft::gpu_draw â€” M2 GPU-driven indirect surface shader.
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
// cull_mode=None + a front-facing flip. NEVER TRS-decompose. Identical math to the M0
// shader â€” only the instance FETCH changed.
//
// group(0) = Bevy's mesh view bind group (reused via SetMeshViewBindGroup<0> so
// position_world_to_clip resolves). group(1) = our two storage buffers.

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

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
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
    return o;
}

@fragment
fn fragment(o: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(o.world_normal);
    if (!front) {
        n = -n;   // double-sided: flip for back faces (inward shells / mirrors)
    }
    let key = normalize(vec3<f32>(0.4, 1.0, 0.3));
    let ndl = clamp(dot(n, key), 0.0, 1.0);
    let ambient = 0.28;
    let base = vec3<f32>(0.66, 0.68, 0.70);
    let lit = base * (ambient + (1.0 - ambient) * ndl);
    return vec4<f32>(lit, 1.0);
}
