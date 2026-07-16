// eft::grade — EFT display-chain post pass (grade LUT + tonemap-OFF display encode).
// Ported MATH from tarkmap/out/_gradegl.js (the WebGL ShaderPass; cleaner reference than the
// TSL variant). The pass OUTPUT IS ALREADY DISPLAY-ENCODED — the sRGB/Hejl encode is BAKED
// INTO THE LUT. Therefore, in the Bevy render graph:
//   * the scene is rendered with NO tonemapping (Tonemapping::None), linear HDR;
//   * this pass runs last;
//   * its target MUST be a NON-sRGB (Unorm) surface/texture — writing to an sRGB target would
//     gamma-encode the already-encoded output twice.
//
// LUT: the web ships a 512x512 RGBA8 raw-bytes file (eft_grade_lut.bin) that is a 64^3 LUT
// tiled 8x8 with a manual blue-slice trilinear in-shader. We convert it to a REAL 64x64x64
// 3D texture at pack/load time so HW trilinear does the blue-slice lerp for free and there is
// no tile-seam math. The 2D-tiled fallback formula is preserved in the comments below.
//
// SHAPER: the LUT domain map is p = sqrt(clamp(lin/4, 0, 1)) — NOT sRGB/log. This lets the
// 64^3 LUT cover HDR up to 4.0. Get it wrong and the LUT reads the wrong slice.

struct GradeParams {
    // x = exposure (default 0.18; eye-adaptation may override).
    exposure: f32,
    // EFT-style unsharp-mask strength on the pre-LUT linear scene (0 = off; game ships ~0.5).
    sharpen: f32,
    _pad1: f32, _pad2: f32,
    // vignette tuning (PRISM): aspect divisors + smoothstep edges + strength.
    // xy = (1.15, 0.95) axis divisors; z,w = smoothstep(0.55, 1.25).
    vig: vec4<f32>,
    vig_strength: vec4<f32>,   // x = 0.488
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;   // linear HDR scene RT
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var lut3d: texture_3d<f32>;       // 64^3, Linear filter, ClampToEdge, NoColorSpace
@group(0) @binding(3) var lut_samp: sampler;
@group(0) @binding(4) var<uniform> grade: GradeParams;

struct FsIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle: 3 verts covering the screen, uv in [0,1]. The clip-space Y is FLIPPED
// (Bevy's fullscreen_vertex_shader convention): uv (0,0) = clip (-1,+1) = top-left pixel, so
// texture V grows downward in step with framebuffer Y — without this the image is upside down.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FsIn {
    var out: FsIn;
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));  // (0,0),(2,0),(0,2)
    out.uv = uv;
    out.clip = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

// LUT sample via a REAL 3D texture. c = pre-exposed LINEAR color.
// shaper p = sqrt(clamp(c/4,0,1)) in [0,1]; HW trilinear covers the blue-slice lerp.
fn lut_sample(c: vec3<f32>) -> vec3<f32> {
    let p = sqrt(clamp(c / 4.0, vec3<f32>(0.0), vec3<f32>(1.0)));
    return textureSampleLevel(lut3d, lut_samp, p, 0.0).rgb;
}

// --- 2D-tiled fallback (kept for reference; only if the LUT stays a 512x512 8x8 atlas) ---
//   let u = sqrt(clamp(c/4,0,1)) * 63.0;
//   let b0 = floor(u.b); let f = u.b - b0; let b1 = min(b0 + 1.0, 63.0);
//   let xy = vec2(u.r, u.g) + 0.5;
//   let uv0 = (vec2(u.b_mod8, floor(b0/8)) * 64.0 + xy) / 512.0;   // etc.
//   return mix(tex(uv0).rgb, tex(uv1).rgb, f);

@fragment
fn fs_grade(in: FsIn) -> @location(0) vec4<f32> {
    var scene = textureSampleLevel(scene_tex, scene_samp, in.uv, 0.0).rgb;
    // EFT-style sharpen: 4-tap unsharp mask on the pre-LUT linear scene (the game applies a
    // strong sharpen in its own post chain). max() guards ringing below zero.
    if (grade.sharpen > 0.0) {
        let ts = 1.0 / vec2<f32>(textureDimensions(scene_tex));
        let n = textureSampleLevel(scene_tex, scene_samp, in.uv + vec2<f32>(ts.x, 0.0), 0.0).rgb
              + textureSampleLevel(scene_tex, scene_samp, in.uv - vec2<f32>(ts.x, 0.0), 0.0).rgb
              + textureSampleLevel(scene_tex, scene_samp, in.uv + vec2<f32>(0.0, ts.y), 0.0).rgb
              + textureSampleLevel(scene_tex, scene_samp, in.uv - vec2<f32>(0.0, ts.y), 0.0).rgb;
        scene = max(scene + (scene - n * 0.25) * grade.sharpen, vec3<f32>(0.0));
    }
    let lin = scene * grade.exposure;
    let g = lut_sample(lin);

    // PRISM vignette: e = (uv-0.5)*2 / (1.15, 0.95); vig = 1 - smoothstep(0.55,1.25,|e|)*0.488.
    let e = (in.uv - 0.5) * 2.0 / grade.vig.xy;
    let vig = 1.0 - smoothstep(grade.vig.z, grade.vig.w, length(e)) * grade.vig_strength.x;

    // Already DISPLAY-ENCODED (LUT baked the sRGB encode). No further tonemap / sRGB.
    return vec4<f32>(g * vig, 1.0);
}
