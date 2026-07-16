// eft::ssao — depth-only screen-space ambient occlusion (Graphics (experimental) toggle).
//
// The custom GPU-driven path renders no normal prepass, so this is CLASSIC depth-reconstructed
// SSAO: view-space position from the (multisampled, reverse-z) depth buffer, face normal from
// neighboring texels, a per-pixel-rotated spiral of range-checked horizon taps, then a MULTIPLY
// onto the scene color. Runs between the main pass and Bloom, so the grade LUT tonemaps the
// occluded color like everything else. Physically it darkens *all* light (not just ambient) —
// the classic SSAO approximation; intensity is UI-tunable and the whole pass is off by default.
//
// Distance fade (p.w) keeps AO out of the fog band — far geometry is haze-lit, not crevice-lit.

struct SsaoParams {
    inv_proj: mat4x4<f32>,  // view-from-clip (Bevy reverse-z infinite projection inverse)
    // x = world radius (m), y = intensity, z = power, w = fade-end view distance (m)
    p: vec4<f32>,
    // x,y = viewport px, z = proj11 (1/tan(fov_y/2)), w = pad
    vp: vec4<f32>,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var depth_tex: texture_depth_multisampled_2d;
@group(0) @binding(3) var<uniform> ao: SsaoParams;

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

// View-space position of a pixel (sample 0 of the MSAA depth — AO is a low-frequency term).
fn view_pos_at(px: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    let c = clamp(px, vec2<i32>(0), dims - 1);
    let d = textureLoad(depth_tex, c, 0);
    let uv = (vec2<f32>(c) + 0.5) / vec2<f32>(dims);
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, d);
    let v = ao.inv_proj * vec4<f32>(ndc, 1.0);
    return v.xyz / v.w;
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

const TAPS: i32 = 10;

@fragment
fn fs_ssao(in: FsIn) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(scene_tex, scene_samp, in.uv, 0.0);
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let px = vec2<i32>(in.clip.xy);

    let d0 = textureLoad(depth_tex, clamp(px, vec2<i32>(0), dims - 1), 0);
    if (d0 <= 1e-7) { // reverse-z far plane = sky: nothing to occlude
        return color;
    }
    let P = view_pos_at(px, dims);
    let Px = view_pos_at(px + vec2<i32>(1, 0), dims);
    let Py = view_pos_at(px + vec2<i32>(0, 1), dims);
    var N = normalize(cross(Px - P, Py - P));
    if (dot(N, -P) < 0.0) { N = -N; } // face the camera (view looks down -Z)

    // Project the world-space radius to pixels at this depth; clamp so the kernel neither
    // vanishes at range nor explodes point-blank.
    let view_z = max(-P.z, 0.05);
    let r_px = clamp(ao.p.x * ao.vp.z * 0.5 * ao.vp.y / view_z, 2.0, 64.0);

    let rot = hash12(vec2<f32>(px)) * 6.28318;
    var occ = 0.0;
    for (var i = 0; i < TAPS; i = i + 1) {
        let ang = rot + f32(i) * 2.39996; // golden-angle spiral
        let rad = sqrt((f32(i) + 0.5) / f32(TAPS)) * r_px;
        let off = vec2<i32>(vec2<f32>(cos(ang), sin(ang)) * rad);
        let S = view_pos_at(px + off, dims);
        let v = S - P;
        let d2 = dot(v, v);
        if (d2 < 1e-6) { continue; }
        // Horizon term with a distance falloff: only geometry within ~radius counts.
        let falloff = 1.0 / (1.0 + d2 / (ao.p.x * ao.p.x));
        occ += max(0.0, dot(N, v) * inverseSqrt(d2) - 0.08) * falloff;
    }
    var a = 1.0 - clamp(occ * (2.0 / f32(TAPS)), 0.0, 1.0);
    a = pow(a, ao.p.z);
    // Fade AO out with view distance — the fog band owns the far field.
    let fade = 1.0 - smoothstep(ao.p.w * 0.6, ao.p.w, view_z);
    a = mix(1.0, a, ao.p.y * fade);
    return vec4<f32>(color.rgb * a, color.a);
}
