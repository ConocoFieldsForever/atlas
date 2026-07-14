// eft::splat — EFT "Custom/Vert Paint SoftCutout Decal" 3-layer height splat.
// Ported MATH from tarkmap/out/_vpsplat.js (which itself is the RE'd DX11 fragment:
//   w_i = pow(Heights_i * vColor_i, BlendStrength), normalized; albedo = sum(w_i * layer_i)).
//
// This module is BINDING-FREE on purpose: the caller (instanced.wgsl) does the bindless
// texture sampling at the per-layer tiled UVs, then hands the sampled values to these pure
// functions. That keeps the splat math portable and lets Bevy's own sRGB texture import do
// the color decode (the web pow(2.2) is a NO-OP here — see note in vp_albedo below).
//
// CORRECTNESS FIXES CARRIED FROM THE WEB (do not drop):
//   * The Heights control mask (R/G/B = layer0/1/2 weight) is sampled ONCE at the RAW mesh
//     uv, NOT at the tiled per-layer uvs (tiled sampling speckles the weights). Caller must
//     sample it at the raw uv and pass rgb here.
//   * vColor is the per-LAYER BLEND WEIGHT, never a tint. NEVER multiply diffuse by vColor
//     (the stock three.js color_fragment is killed in the web port for exactly this reason).
//   * Two fallbacks: (a) zero height-coverage -> base layer (1,0,0); (b) blended albedo that
//     resolves near-black -> layer0.
//   * Roughness: smoothness is packed in each layer's albedo ALPHA; influence compressed x0.30
//     and FLOORED at 0.72 so asphalt can't mirror the sky.

#define_import_path eft::splat

const VP_LUMA: vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);

// Per-layer height/vColor -> normalized 3-way blend weights.
//   heights: Heights control mask .rgb sampled ONCE at the raw mesh uv (R->L0, G->L1, B->L2).
//   vcolor:  COLOR_0.rgb vertex attribute (the per-layer paint weights).
//   blend:   BlendStrength (vp.b, default 1.0).
fn vp_weights(heights: vec3<f32>, vcolor: vec3<f32>, blend: f32) -> vec3<f32> {
    let hw = heights * vcolor;
    let hs = hw.x + hw.y + hw.z;
    var w: vec3<f32>;
    if (hs <= 1e-5) {
        // codex fix: unpainted / "Solid" variant / missing vcol -> base layer, not a 3-way wash.
        w = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        w = pow(max(hw, vec3<f32>(1e-4)), vec3<f32>(blend));
    }
    return w / max(w.x + w.y + w.z, 1e-4);
}

// Blend the 3 layer albedos by weight, with the near-black -> layer0 fallback.
//   a0/a1/a2 are LINEAR layer colors already multiplied by their per-layer tint (uCol*).
//   NOTE on sRGB: the web sampled RAW and did pow(2.2) here. In Bevy the layer albedo
//   textures are imported as sRGB (BC7), so sampling already yields LINEAR — pass linear and
//   do NOT pow again. Apply the per-layer tint (uCol0..2) at the call site before passing.
fn vp_albedo(w: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>) -> vec3<f32> {
    var spl = w.x * a0 + w.y * a1 + w.z * a2;
    if (dot(spl, VP_LUMA) < 0.02) {
        spl = a0;
    }
    return spl;
}

// Ground-matte roughness: smoothness packed in the layer albedo alpha (s0/s1/s2).
// roughness = clamp(1 - 0.30*sum(w*smoothness), 0.72, 1.0).
fn vp_roughness(w: vec3<f32>, s0: f32, s1: f32, s2: f32) -> f32 {
    return clamp(1.0 - 0.30 * (w.x * s0 + w.y * s1 + w.z * s2), 0.72, 1.0);
}

// SoftCutout feather alpha (roads painted onto terrain). coverage = COLOR_0.a.
//   a_str = _AlphaStrength (vp.a[0]), a_cut = _Cutoff (vp.a[1]), a_hgt = _AlphaHeight (vp.a[2]).
// Only applied when the material is a SoftCutout variant (uASoft>0.5) AND drawn transparent
// with depth-write off, after terrain — the caller owns that render state.
fn vp_softcutout_alpha(coverage: f32, a_str: f32, a_cut: f32, a_hgt: f32) -> f32 {
    return clamp(coverage * a_str - (a_cut - a_hgt), 0.0, 1.0) * coverage;
}

// Convenience aggregate: everything the caller needs after it has sampled the 4 maps.
struct VpResult {
    albedo: vec3<f32>,
    roughness: f32,
    weights: vec3<f32>,
};

fn vp_splat(
    heights: vec3<f32>, vcolor: vec3<f32>, blend: f32,
    a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>,   // per-layer LINEAR albedo * uCol
    s0: f32, s1: f32, s2: f32,                      // per-layer smoothness (albedo alpha)
) -> VpResult {
    let w = vp_weights(heights, vcolor, blend);
    var out: VpResult;
    out.weights = w;
    out.albedo = vp_albedo(w, a0, a1, a2);
    out.roughness = vp_roughness(w, s0, s1, s2);
    return out;
}
