// eft::fpv_cam — analog FPV video-link post pass (drone mode only).
//
// Emulates a 5.8 GHz analog VTX/CMOS chain the way it actually degrades:
//  * ever-present fine luma noise (even at full RSSI an analog feed is never clean),
//  * per-scanline horizontal tear bursts (sync instability) that grow as signal drops,
//  * NTSC-style chroma fringing (R/B sampled a touch apart, wider when weak),
//  * faint scanlines + a slow hum bar,
//  * full-frame snow breakup as RSSI approaches zero.
//
// Runs AFTER the grade/tonemap on the HDR ping-pong target (display-referred values in [0,1] —
// which is correct: this noise rides the *video signal*, not the scene light). `signal` (0..1)
// comes from the CPU RF model: pilot-to-drone range + wall/floor crossings (render/fpv_cam.rs).

struct FpvParams {
    time: f32,
    signal: f32,     // 0 = static, 1 = clean-ish
    intensity: f32,  // user master gain for the whole effect
    aspect: f32,
    enabled: f32,    // <0.5 = pass-through
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var<uniform> fx: FpvParams;

struct FsIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle, Bevy y-flip convention (same as grade.wgsl).
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FsIn {
    var out: FsIn;
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    out.uv = uv;
    out.clip = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

// Cheap 2D hash (Hoskins-style) — plenty for video noise.
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_fpv(in: FsIn) -> @location(0) vec4<f32> {
    let uv = in.uv;
    if (fx.enabled < 0.5 || fx.intensity <= 0.001) {
        return textureSampleLevel(scene_tex, scene_samp, uv, 0.0);
    }
    let t = fx.time;
    let gain = clamp(fx.intensity, 0.0, 1.0);
    let sig = clamp(fx.signal, 0.0, 1.0);
    let weak = 1.0 - sig;

    // --- Sync instability: per-scanline horizontal tear, gated in bursts ------------------
    let rows = 480.0; // NTSC-ish line count
    let row = floor(uv.y * rows);
    // Global burst gate: fires occasionally when strong, near-constantly when weak.
    let burst_seed = hash12(vec2<f32>(floor(t * 9.0), 3.71));
    let burst = step(1.0 - (0.03 + 0.75 * weak), burst_seed);
    // Which rows tear inside a burst (re-rolled ~60 Hz), and by how much.
    let row_roll = hash12(vec2<f32>(row, floor(t * 61.0)));
    let row_sel = step(1.0 - (0.06 + 0.5 * weak), hash12(vec2<f32>(row * 1.7, floor(t * 13.0) + 9.1)));
    let tear = (row_roll - 0.5) * burst * row_sel * (0.004 + 0.10 * weak) * gain;
    var suv = vec2<f32>(uv.x + tear, uv.y);

    // --- Chroma fringing: R/B sampled slightly apart (wider when weak / tearing) ----------
    let shift = (0.0006 + 0.004 * weak + 0.02 * abs(tear)) * gain;
    let cr = textureSampleLevel(scene_tex, scene_samp, suv + vec2<f32>(shift, 0.0), 0.0).r;
    let cg = textureSampleLevel(scene_tex, scene_samp, suv, 0.0).g;
    let cb = textureSampleLevel(scene_tex, scene_samp, suv - vec2<f32>(shift, 0.0), 0.0).b;
    var col = vec3<f32>(cr, cg, cb);

    // --- Analog softness + desaturation as the link degrades ------------------------------
    let luma = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    col = mix(col, vec3<f32>(luma), (0.08 + 0.45 * weak) * gain);

    // --- Luma noise: fine grain always, snow that eats the picture as RSSI dies -----------
    // Noise texel scale ~ one scanline tall, a couple px wide, re-rolled per frame-ish tick.
    let n = hash12(uv * vec2<f32>(640.0 * fx.aspect, rows) + vec2<f32>(t * 97.3, t * 61.7));
    let grain = (n - 0.5) * (0.06 + 0.10 * weak) * gain;
    col = col + vec3<f32>(grain);
    // Snow: replaces the picture (not adds) — quadratic in weakness so it slams in at the end.
    let snow_amt = gain * weak * weak * (0.25 + 0.75 * step(0.35, n));
    col = mix(col, vec3<f32>(n), clamp(snow_amt, 0.0, 1.0));

    // --- Scanlines + slow hum bar ----------------------------------------------------------
    col = col * (1.0 - 0.10 * gain * (0.5 + 0.5 * sin(uv.y * rows * 6.2832)));
    col = col * (1.0 + 0.025 * gain * sin(uv.y * 2.5 + t * 1.7));

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(4.0)), 1.0);
}
