// eft::autoexposure — eye adaptation for the grade chain.
//
// WHY: the display chain ran at a FIXED exposure (GradeParams.exposure, default 0.18). Its own comment
// said "eye-adaptation may override", and nothing ever did. A single constant cannot serve both a
// sunlit exterior and an unlit interior on the same map, so interiors read flat-dark or exteriors blow
// out — and adapting as you move between them is one of the strongest perceptual cues the game has.
//
// METHOD: ONE workgroup of 64 invocations, each taking a strided grid of texel loads over the whole
// HDR scene, averaging LOG luminance, then a shared-memory reduction to a single value. Log-average
// (geometric mean) rather than arithmetic: a few specular pixels at 50.0 would drag an arithmetic mean
// far above what the frame actually looks like, which is exactly how auto-exposure gets a reputation
// for pumping.
//
// One dispatch, no mip chain, no multi-pass, and NO CPU READBACK — the adapted value lives in a
// storage buffer that the grade fragment reads the same frame. A readback would either stall the
// frame or adapt a frame or more late.
//
// The smoothing is asymmetric on purpose: eyes darken faster than they brighten. Both rates are
// per-second and integrated with the frame delta, so adaptation speed does not change with framerate
// (it would otherwise adapt ~3x faster at 180 fps than at 60).

struct ExposureState {
    // Smoothed log2 luminance carried between frames. Sentinel 1e9 = "never initialised": the first
    // frame snaps to the measured value instead of easing up from an arbitrary start, so a map load
    // does not open with a visible fade-in.
    log_lum: f32,
    // The exposure the grade fragment actually uses. 0 until the reduction has run once.
    exposure: f32,
    // Log-luminance of the first measured frame: the REFERENCE adaptation is relative to, so the
    // authored exposure is reproduced exactly at load and no absolute key has to be guessed.
    ref_log: f32,
    _pad1: f32,
};

// Byte-identical to the Rust GradeParamsGpu (48 bytes). Bound here so the compute stage can read the
// frame delta and the base exposure without a second uniform.
struct GradeParams {
    exposure: f32,
    sharpen: f32,
    aa: f32,
    // Frame delta in seconds (was a pad lane). 0 => snap, which is what a paused/first frame wants.
    dt: f32,
    vig: vec4<f32>,
    // x = vignette strength, y = ARMED (see arm_auto_exposure in main.rs).
    vig_strength: vec4<f32>,
};

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> state: ExposureState;
@group(0) @binding(2) var<uniform> grade: GradeParams;

const THREADS: u32 = 64u;
// Taps per invocation. 64 x 64 = 4096 samples of the frame: plenty for a single scalar, and cheap
// enough to be unmeasurable next to the pass that follows it.
const TAPS: u32 = 64u;

// Adaptation rates, log2 units per second. Darkening (stepping into sun) is faster than brightening
// (stepping into a basement), matching how eyes actually behave.
const RATE_DOWN: f32 = 3.0;
const RATE_UP: f32 = 1.2;
// Clamp the measured average so a pitch-black or blown frame cannot drive exposure to absurdity.
const MIN_LOG_LUM: f32 = -8.0;
const MAX_LOG_LUM: f32 = 6.0;
// Adaptation is RELATIVE to the authored exposure, not an absolute key.
//
// The first version aimed the frame at an absolute middle grey of 0.18 and clamped the result to
// [0.04, 0.60] — numbers taken from grade.wgsl's comment "exposure (default 0.18)". That comment is
// STALE: the real default is DEFAULT_GRADE_EXPOSURE = 1.35, so the whole calibration was anchored
// 7.5x off and the clamp ceiling sat BELOW the authored value. Measured, it moved a woods exterior by
// 2.05x, which is not an adaptation, it is a regrade.
//
// So: remember the log-luminance of the FIRST measured frame as the reference, and thereafter apply
// only the EV difference from it. Two properties fall out for free — at load the exposure is exactly
// the authored one (the game's grade is preserved, which is the whole point of shipping its LUT), and
// there is no absolute constant to get wrong, because the reference is measured rather than assumed.
const MAX_EV: f32 = 2.0;   // +-2 stops of adaptation authority, no more
const SENTINEL: f32 = 1.0e9;

var<workgroup> partial: array<f32, THREADS>;

@compute @workgroup_size(64)
fn cs_autoexposure(@builtin(local_invocation_id) lid: vec3<u32>) {
    let dims = textureDimensions(scene);
    let n = dims.x * dims.y;
    let t = lid.x;

    var sum = 0.0;
    var cnt = 0.0;
    if (n > 0u) {
        // Stride across the WHOLE image rather than sampling a central block: a centre-weighted meter
        // on a map viewer would swing wildly as the camera crosses a bright roof or a dark treeline.
        let stride = max(n / (THREADS * TAPS), 1u);
        var i = t * stride;
        for (var k = 0u; k < TAPS; k++) {
            let idx = i % n;
            let c = vec2<u32>(idx % dims.x, idx / dims.x);
            let rgb = textureLoad(scene, vec2<i32>(c), 0).rgb;
            let lum = max(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1.0e-5);
            sum += log2(lum);
            cnt += 1.0;
            i += stride * THREADS;
        }
    }
    partial[t] = select(0.0, sum / max(cnt, 1.0), cnt > 0.0);
    workgroupBarrier();

    // Tree reduction over the 64 partials.
    var step = THREADS / 2u;
    loop {
        if (step == 0u) { break; }
        if (t < step) {
            partial[t] = partial[t] + partial[t + step];
        }
        workgroupBarrier();
        step = step / 2u;
    }

    if (t != 0u) {
        return;
    }
    let measured = clamp(partial[0] / f32(THREADS), MIN_LOG_LUM, MAX_LOG_LUM);
    // NOT ARMED (still loading, or just loaded): publish the authored exposure verbatim and latch
    // nothing. A reference taken from a partially-streamed frame is not a reference, and the visible
    // result was the image brightening the moment the camera first moved.
    if (grade.vig_strength.y < 0.5) {
        state.log_lum = SENTINEL;
        state.ref_log = SENTINEL;
        state.exposure = grade.exposure;
        return;
    }
    var cur = state.log_lum;
    if (cur > SENTINEL * 0.5 || cur != cur) {
        cur = measured; // first frame (or a NaN got in): snap, do not fade
    } else {
        // Exponential approach at a per-SECOND rate, so framerate does not change the feel.
        let rate = select(RATE_UP, RATE_DOWN, measured > cur);
        let a = 1.0 - exp(-max(grade.dt, 0.0) * rate);
        cur = mix(cur, measured, clamp(a, 0.0, 1.0));
    }
    state.log_lum = cur;
    // Latch the reference on the first frame that produced a value.
    var refl = state.ref_log;
    if (refl > SENTINEL * 0.5 || refl != refl) {
        refl = cur;
        state.ref_log = refl;
    }
    // Darker than the reference -> positive EV -> brighter exposure, and vice versa. exp2 of a clamped
    // stop count, times the AUTHORED exposure: at the reference this is exactly 1.0x.
    let ev = clamp(refl - cur, -MAX_EV, MAX_EV);
    state.exposure = max(grade.exposure * exp2(ev), 1.0e-5);
}
