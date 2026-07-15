// #5 SUN SHADOW PASS — depth-only. Reuses the camera-culled `visible[]` + instance buffer
// (same group as the main draw's group(1)) and the shared interleaved vertex buffer, but
// transforms into the sun's orthographic light space instead of the camera. No fragment
// stage: the pipeline writes only depth into the shadow map. The main draw then projects each
// fragment's world position into this same light space and PCF-compares against this depth.

struct InstanceGpu {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    ids: vec4<u32>,
    sphere: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> instances: array<InstanceGpu>;
@group(0) @binding(1) var<storage, read> visible: array<u32>;

struct SunUniform {
    // world -> sun light clip (orthographic). Row-major mat4 (bevy Mat4 col-major upload).
    view_proj: mat4x4<f32>,
    // xyz = sun direction (toward the sun), w = shadow map texel size (for PCF).
    dir_texel: vec4<f32>,
};
@group(1) @binding(0) var<uniform> sun: SunUniform;

// Only position (location 0) is needed; the interleaved buffer's other attrs are ignored via
// a position-only vertex layout on the shadow pipeline (Rust side).
@vertex
fn vertex(@location(0) position: vec3<f32>, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
    let inst = instances[visible[ii]];
    let col0 = vec3<f32>(inst.m0.x, inst.m1.x, inst.m2.x);
    let col1 = vec3<f32>(inst.m0.y, inst.m1.y, inst.m2.y);
    let col2 = vec3<f32>(inst.m0.z, inst.m1.z, inst.m2.z);
    let t = vec3<f32>(inst.m0.w, inst.m1.w, inst.m2.w);
    let world = mat3x3<f32>(col0, col1, col2) * position + t;
    return sun.view_proj * vec4<f32>(world, 1.0);
}
