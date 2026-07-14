// eft M0 instanced surface shader — the WIRED first-pixel draw.
//
// Consumes the per-instance FULL ROW-MAJOR 3x4 affine (incl shear + mirror) as three
// instance-rate vec4 rows, plus the mesh vertex (position/normal/uv). It applies the
// affine to RAW verts and transforms the normal by the COFACTOR matrix — the #1 rule:
// NEVER TRS-decompose. Mirrors (det<0) are correct with ZERO baking because the pipeline
// draws double-sided (cull off) and the cofactor matrix flips the normal sign for det<0.
//
// M0 lighting is a flat sun+ambient lambert (no textures/materials yet — that is M2/M3 in
// instanced.wgsl). Output goes through Bevy's normal main-pass → tonemapping.
//
// Reuses Bevy's mesh view bind group (group 0) via the standard view-transform helper.

#import bevy_pbr::view_transformations::position_world_to_clip

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // instance-rate (step_mode = Instance):
    @location(3) m0: vec4<f32>,   // affine row 0 = (m00 m01 m02 m03)
    @location(4) m1: vec4<f32>,   // affine row 1
    @location(5) m2: vec4<f32>,   // affine row 2
    @location(6) ids: vec4<u32>,  // x = flags (bit0 MIRROR), y = materialId (reserved)
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

// cofactor(linear 3x3) = det · inverse-transpose. Columns = cross products of the linear
// columns. Correct normal transform under shear / non-uniform scale / mirror WITHOUT
// decomposing or inverting.
fn cofactor(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(cross(c1, c2), cross(c2, c0), cross(c0, c1));
}

@vertex
fn vertex(v: Vertex) -> VOut {
    // rebuild linear 3x3 columns + translation from the ROW-MAJOR 3x4.
    let col0 = vec3<f32>(v.m0.x, v.m1.x, v.m2.x);
    let col1 = vec3<f32>(v.m0.y, v.m1.y, v.m2.y);
    let col2 = vec3<f32>(v.m0.z, v.m1.z, v.m2.z);
    let t = vec3<f32>(v.m0.w, v.m1.w, v.m2.w);
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
    // double-sided: flip for back faces so inward shells / mirrors shade correctly.
    if (!front) {
        n = -n;
    }
    let key = normalize(vec3<f32>(0.4, 1.0, 0.3));   // overcast key stand-in (real sun_dir is M3)
    let ndl = clamp(dot(n, key), 0.0, 1.0);
    let ambient = 0.28;
    let base = vec3<f32>(0.66, 0.68, 0.70);
    let lit = base * (ambient + (1.0 - ambient) * ndl);
    return vec4<f32>(lit, 1.0);
}
