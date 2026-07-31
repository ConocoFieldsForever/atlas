// DEPTH PYRAMID (Phase 1 substrate, docs/GRAPHICS_PLAN.md) — a reverse-z MAX-reduction mip chain
// over the normal prepass's 1x depth, shared by every consumer that needs hierarchical depth:
// SSR's hierarchical trace (Phase 6) and Hi-Z occlusion culling (Phase 3). ONE pyramid, built once
// per frame, per the plan's "SSR, contact shadows and Hi-Z should not each build their own".
//
// Reverse-z: NEAR = 1, FAR = 0. An occluder query asks "is anything in this footprint closer than
// my depth?", so the reduction keeps the MAXIMUM (nearest) of each 2x2 — a conservative nearest-
// surface bound per texel footprint. Odd sizes: the gather clamps, so edge texels self-duplicate;
// conservative for max-reduction.
//
// Two entry points, one bind group shape (src texture, dst storage mip):
//   cs_copy   — mip 0: read the prepass DEPTH texture (depth format needs textureLoad-as-depth).
//   cs_reduce — mip i: 2x2 max of mip i-1 (both R32Float storage/sampled).

@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var src_mip: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let d = textureLoad(src_depth, vec2<i32>(gid.xy), 0);
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn cs_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let sdims = vec2<i32>(textureDimensions(src_mip));
    let base = vec2<i32>(gid.xy) * 2;
    let p00 = clamp(base, vec2<i32>(0), sdims - 1);
    let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), sdims - 1);
    let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), sdims - 1);
    let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), sdims - 1);
    // MIN reduce (reverse-z: smaller = farther). Each texel of mip i is the FARTHEST depth in
    // its 2x2 footprint, so the whole tile is provably at-or-nearer — exactly what the Hi-Z
    // occlusion test needs ("everything here is nearer than the sphere" => cull). Was MAX
    // (nearest) from the speculative Phase-1 build, which no pass ever consumed.
    let m = min(
        min(textureLoad(src_mip, p00, 0).r, textureLoad(src_mip, p10, 0).r),
        min(textureLoad(src_mip, p01, 0).r, textureLoad(src_mip, p11, 0).r),
    );
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(m, 0.0, 0.0, 0.0));
}
