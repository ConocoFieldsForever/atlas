// eft::ssao — SSAO as a PRE-MAIN AO LANE (Graphics (experimental) toggle).
//
// Runs AFTER the normal/depth prepass and BEFORE the main pass, writing an R8 occlusion factor
// the main pass samples during OPAQUE shading (ambient + lamp diffuse). This replaces the old
// post-multiply over the finished frame, which darkened every pixel by the occlusion of whatever
// the PREPASS saw there — glass is excluded from the prepass, so a pane was darkened by the AO of
// the interior BEHIND it (the "SSAO through glass" bug). As a lane, BLEND surfaces simply don't
// sample it (ao = 1), and AO correctly scales only the ambient terms instead of sun + emissive.
//
// Positions reconstruct from the prepass 1x depth (reverse-z, cleared to 0 = sky); normals come
// from the prepass target with a per-pixel derivative fallback where it wrote none (blend/grass).
// Distance fade (p.w) keeps AO out of the fog band — far geometry is haze-lit, not crevice-lit.

struct SsaoParams {
    inv_proj: mat4x4<f32>,  // view-from-clip (Bevy reverse-z infinite projection inverse)
    // world -> view rotation (full matrix uploaded; only the 3x3 is used, on direction vectors).
    // Needed because the prepass stores WORLD-space normals and this shader works in view space.
    view_from_world: mat4x4<f32>,
    // x = world radius (m), y = intensity, z = power, w = fade-end view distance (m)
    p: vec4<f32>,
    // x,y = viewport px, z = proj11 (1/tan(fov_y/2)), w = reserved
    vp: vec4<f32>,
};

@group(0) @binding(0) var depth_tex: texture_depth_2d;   // prepass depth (1x, reverse-z, 0 = sky)
// The prepass normal target (world normal.xyz + class/roughness.w). Zero where the prepass wrote
// nothing (sky / blend / grass) — those pixels take the derivative fallback below.
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> ao: SsaoParams;

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

// View-space position of a pixel from the prepass depth.
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
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let px = vec2<i32>(in.clip.xy);

    let d0 = textureLoad(depth_tex, clamp(px, vec2<i32>(0), dims - 1), 0);
    if (d0 <= 1e-7) { // reverse-z far plane = sky: nothing to occlude
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let P = view_pos_at(px, dims);
    // REAL surface normal from the prepass when it wrote this pixel; derivative face normal
    // otherwise (the prepass clears to zero, so sky / blend surfaces / the excluded grass read
    // (0,0,0) and take the fallback — one code path, per-pixel choice). The rasterized normal
    // neither facets on curves nor halos at silhouettes like the derivative one does.
    var N: vec3<f32>;
    let ndims = vec2<i32>(textureDimensions(normal_tex));
    let npx = clamp(px, vec2<i32>(0), ndims - 1);
    let nr = textureLoad(normal_tex, npx, 0);
    if (dot(nr.xyz, nr.xyz) > 0.1) {
        N = normalize((ao.view_from_world * vec4<f32>(nr.xyz, 0.0)).xyz);
    } else {
        let Px = view_pos_at(px + vec2<i32>(1, 0), dims);
        let Py = view_pos_at(px + vec2<i32>(0, 1), dims);
        N = normalize(cross(Px - P, Py - P));
    }
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
    return vec4<f32>(a, 0.0, 0.0, 1.0);
}
