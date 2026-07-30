// TAA (Phase 2, docs/GRAPHICS_PLAN.md) — temporal accumulation over the HDR scene, between the
// SSAO composite and Bloom, so the grade LUT tonemaps the converged image like everything else.
//
// Sources of truth per pixel:
//   * CURRENT color   — the post-process source (HDR, MSAA-resolved).
//   * DEPTH           — the prepass 1x depth (reverse-z). No prepass data = no reprojection.
//   * CLASS           — the prepass normal target's w lane: 0 = no data (sky/blend/grass),
//                       [0,1] = static opaque, [2,3] = water (reactive: its shading animates).
//   * HISTORY         — our own ping-pong Rgba16Float, valid only while `params.has_history > 0`.
//
// Reprojection is CAMERA-ONLY, exactly as the plan warns: doors/grass/water move without motion
// vectors, so those classes are handled by REACTIVITY (blend toward current) instead of pretending
// their history is valid. The plan's acceptance is "no trails longer than two frames" — reactive
// classes converge in one to two frames by construction.
//
// Neighborhood VARIANCE clipping (not min/max clamp): the standard mean±γσ AABB in YCoCg keeps
// history that merely drifts inside the local distribution while rejecting genuine disocclusions;
// γ = 1.0 is conservative (less ghosting, slightly less smoothing).

struct TaaParams {
    // Previous frame's clip_from_world (UNJITTERED), for world reprojection.
    prev_clip_from_world: mat4x4<f32>,
    // Current frame's view-from-clip (inverse of unjittered clip_from_view) …
    inv_proj: mat4x4<f32>,
    // … and world-from-view, so depth -> world without a second matrix multiply on the CPU.
    world_from_view: mat4x4<f32>,
    // x = blend alpha for converged static pixels (history weight = 1-x), y = has_history (1/0),
    // z,w = viewport size in px.
    p: vec4<f32>,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;      // prepass 1x depth
@group(0) @binding(4) var normal_tex: texture_2d<f32>;      // prepass normal+class (w lane)
@group(0) @binding(5) var<uniform> params: TaaParams;

struct FsIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FsIn {
    var out: FsIn;
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    out.uv = uv;
    out.clip = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}
fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

@fragment
fn fs_taa(in: FsIn) -> @location(0) vec4<f32> {
    let cur = textureSampleLevel(scene_tex, scene_samp, in.uv, 0.0);
    if (params.p.y < 0.5) {
        return cur; // no history yet (first frame / invalidation): pass through
    }
    let px = vec2<i32>(in.clip.xy);
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let c = clamp(px, vec2<i32>(0), dims - 1);
    let d = textureLoad(depth_tex, c, 0);
    let class_w = textureLoad(normal_tex, c, 0).w;

    // No prepass data (sky, blend surfaces, grass): camera reprojection is not defined for what
    // is actually moving there. Blend lightly against the SCREEN-SAME history texel — this still
    // damps FXAA/shading shimmer on sky and foliage without inventing motion.
    if (class_w <= 0.0 || d <= 1e-7) {
        let hist0 = textureSampleLevel(history_tex, scene_samp, in.uv, 0.0);
        return mix(hist0, cur, 0.5);
    }

    // World position from prepass depth, then into LAST frame's clip.
    let ndc = vec3<f32>(in.uv.x * 2.0 - 1.0, 1.0 - 2.0 * in.uv.y, d);
    let vpos4 = params.inv_proj * vec4<f32>(ndc, 1.0);
    let wpos = (params.world_from_view * vec4<f32>(vpos4.xyz / vpos4.w, 1.0)).xyz;
    let prev = params.prev_clip_from_world * vec4<f32>(wpos, 1.0);
    if (prev.w <= 0.0) {
        return cur;
    }
    let pndc = prev.xyz / prev.w;
    let puv = vec2<f32>(pndc.x * 0.5 + 0.5, 0.5 - pndc.y * 0.5);
    if (any(puv < vec2<f32>(0.0)) || any(puv > vec2<f32>(1.0))) {
        return cur; // history off-screen: disocclusion by camera motion
    }
    let hist = textureSampleLevel(history_tex, scene_samp, puv, 0.0);

    // 3x3 neighborhood variance bound in YCoCg.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let s = rgb_to_ycocg(
                textureLoad(scene_tex, clamp(px + vec2<i32>(x, y), vec2<i32>(0), dims - 1), 0).rgb,
            );
            m1 += s;
            m2 += s * s;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    let sigma = sqrt(max(m2 - m1 * m1, vec3<f32>(0.0)));
    let lo = m1 - sigma;
    let hi = m1 + sigma;
    let hist_y = clamp(rgb_to_ycocg(hist.rgb), lo, hi);
    let hist_c = ycocg_to_rgb(hist_y);

    // Blend: static opaque converges hard; WATER is reactive (its shading animates with no motion
    // vectors, so long history would smear the chop the user just got).
    var alpha = params.p.x;
    if (class_w >= 2.0) {
        alpha = max(alpha, 0.5);
    }
    let outc = mix(hist_c, cur.rgb, alpha);
    return vec4<f32>(outc, cur.a);
}

// Trivial blit: the TAA resolve renders into the HISTORY texture (it must persist to next frame),
// and this copies it onto the post-process destination. Two tiny passes beat guessing whether the
// ViewTarget's internal texture carries COPY_SRC/COPY_DST usage.
@fragment
fn fs_blit(in: FsIn) -> @location(0) vec4<f32> {
    return textureSampleLevel(scene_tex, scene_samp, in.uv, 0.0);
}
