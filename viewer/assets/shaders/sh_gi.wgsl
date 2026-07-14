// eft::sh_gi — baked SH-L1 irradiance volume, sampled by WORLD POSITION.
// Ported MATH from the web modules tarkmap/out/_gi.js (+ _gigl.js DC parity) and the
// bake's own layout contract in tarkmap/out/interchange/volume.json.
//
// KEY DIVERGENCE FROM THE WEB VIEWERS (intentional, per the locked design):
//   The web viewers ship DC-ONLY (the c0 ambient term) because GLSL1 has no sampler3D
//   (_gigl.js) and texture3D() hangs WebGPU BatchedMesh compilation (_gi.js). Those are
//   WEB WORKAROUNDS. Native Bevy has neither constraint, so we do the FULL L1 directional
//   reconstruction from a REAL 3D texture. The DC-only path is kept as sh_dc() for A/B.
//
// VOLUME FORMAT (volume.json): float16 LE, probe-major, probe index = ((z*ny)+y)*nx + x.
//   12 halfs/probe = c0.rgb, c1.rgb, c2.rgb, c3.rgb. Coeffs are RADIANCE SH, L1 real basis:
//     c0 = Y00 (0.282095)          -> ambient / DC
//     c1 = Y1-1 (0.488603 * n.y)   -> NOTE axis map: c1 <- y
//     c2 = Y10  (0.488603 * n.z)   -> c2 <- z
//     c3 = Y11  (0.488603 * n.x)   -> c3 <- x
//   IRRADIANCE via cosine convolution A0=pi, A1=2pi/3:
//     E(n) = A0*c0*Y00 + A1*(c1*Y1-1 + c2*Y10 + c3*Y11)
//          = c0*0.8862269 + 1.0233267*(c1*n.y + c2*n.z + c3*n.x)     (then max 0)
//   The RADIANCE coeffs are stored; this convolution turns them into IRRADIANCE.
//
// The Rust loader uploads each coefficient as its own Rgba16Float 3D texture (only .rgb
// used) with the correct axis permutation so texel order is (x,y,z). HW trilinear does the
// probe interpolation for free on the fast path.

#define_import_path eft::sh_gi

// Cosine-convolution reconstruction constants (see volume.json layout string).
const SH_A0: f32 = 0.8862269;   // pi * Y00      = pi * 0.282095
const SH_A1: f32 = 1.0233267;   // (2pi/3) * 0.488603

struct GiParams {
    // xyz = volume world-space min; w = giStr (~0.3183 = 1/pi baseline).
    vol_min: vec4<f32>,
    // xyz = 1/(max-min) per axis; w = leak-gate distance in metres (<=0 => gate off / full path everywhere).
    vol_inv: vec4<f32>,
    // xyz = probe spacing (metres); w = vis dmax (uint8 moment scale).
    spacing: vec4<f32>,
    // x=nx y=ny z=nz (probe dims); w flags: bit0 = vis moments bound (leak gate available).
    dims: vec4<u32>,
};

// --- SH volume: one Rgba16Float 3D texture per L1 coefficient (.rgb used). group(1). ---
@group(1) @binding(0) var sh_c0: texture_3d<f32>;
@group(1) @binding(1) var sh_c1: texture_3d<f32>;
@group(1) @binding(2) var sh_c2: texture_3d<f32>;
@group(1) @binding(3) var sh_c3: texture_3d<f32>;
@group(1) @binding(4) var sh_samp: sampler;                 // Linear, ClampToEdge, no mips
@group(1) @binding(5) var<uniform> gi: GiParams;
// Optional DDGI visibility moments (leak gate). 4 bands: band j = octants {2j, 2j+1}
// packed [meanA, sqA, meanB, sqB] normalized 0..1 (scale by dmax / dmax^2). NearestFilter.
@group(1) @binding(6) var sh_vis0: texture_3d<f32>;
@group(1) @binding(7) var sh_vis1: texture_3d<f32>;
@group(1) @binding(8) var sh_vis2: texture_3d<f32>;
@group(1) @binding(9) var sh_vis3: texture_3d<f32>;

fn gi_dims_f() -> vec3<f32> {
    return vec3<f32>(f32(gi.dims.x), f32(gi.dims.y), f32(gi.dims.z));
}

// world -> normalized [0,1] grid (clamped, matches _fastE()'s clamp(0,1)).
fn gi_grid(world_pos: vec3<f32>) -> vec3<f32> {
    return clamp((world_pos - gi.vol_min.xyz) * gi.vol_inv.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
}

// grid[0,1] -> probe-center-aligned normalized texture coord for HW trilinear.
// probe index i maps to normalized (i+0.5)/dim; grid 0..1 spans probe 0..dim-1.
fn gi_texcoord(g: vec3<f32>) -> vec3<f32> {
    let d = gi_dims_f();
    return (g * (d - vec3<f32>(1.0)) + vec3<f32>(0.5)) / d;
}

// FAST PATH: HW-trilinear sample of all 4 coeffs, full L1 irradiance. No leak gate.
// This is the M1 target (replaces the web DC-only fast path with real directional GI).
fn sh_irradiance_fast(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let tc = gi_texcoord(gi_grid(world_pos));
    let c0 = textureSampleLevel(sh_c0, sh_samp, tc, 0.0).rgb;
    let c1 = textureSampleLevel(sh_c1, sh_samp, tc, 0.0).rgb;
    let c2 = textureSampleLevel(sh_c2, sh_samp, tc, 0.0).rgb;
    let c3 = textureSampleLevel(sh_c3, sh_samp, tc, 0.0).rgb;
    let E = c0 * SH_A0 + SH_A1 * (c1 * n.y + c2 * n.z + c3 * n.x);
    return max(E, vec3<f32>(0.0));
}

// DC-ONLY (web parity / cheapest). Direction-independent ambient term.
fn sh_dc(world_pos: vec3<f32>) -> vec3<f32> {
    let tc = gi_texcoord(gi_grid(world_pos));
    return max(textureSampleLevel(sh_c0, sh_samp, tc, 0.0).rgb * SH_A0, vec3<f32>(0.0));
}

// --- Leak-gate helpers (optional M2 path; ports _gi.js _fullE Chebyshev 8-tap) ---

// integer-voxel fetch of a coefficient (manual trilinear needs corner taps).
fn sh_load(tex: texture_3d<f32>, v: vec3<i32>) -> vec3<f32> {
    return textureLoad(tex, v, 0).rgb;
}

// fetch the vis band texel for a given octant band at voxel v.
fn vis_band(band: i32, v: vec3<i32>) -> vec4<f32> {
    // dynamic texture indexing avoided (stable across wgpu backends) -> select.
    if (band == 0) { return textureLoad(sh_vis0, v, 0); }
    if (band == 1) { return textureLoad(sh_vis1, v, 0); }
    if (band == 2) { return textureLoad(sh_vis2, v, 0); }
    return textureLoad(sh_vis3, v, 0);
}

// FULL PATH: normal-biased, Chebyshev-visibility-weighted 8-tap trilinear (leak resistant).
// Mirrors _gi.js _fullE(): Pb = P + normalize(N)*spacing*0.5; per corner tap the probe->point
// octant picks a (mean, meanSq) moment; cheb = var/(var + max(r-mean,0)^2); w = wtri*cheb^3 (floor 0.02).
fn sh_irradiance_gated(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let sp = gi.spacing.xyz;
    let dmax = gi.spacing.w;
    let Pb = world_pos + normalize(n) * sp * 0.5;
    let g = clamp((Pb - gi.vol_min.xyz) * gi.vol_inv.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let dimf = gi_dims_f();
    let f = g * (dimf - vec3<f32>(1.0));
    let base = floor(f);
    let t = f - base;
    let maxv = vec3<i32>(gi.dims.xyz) - vec3<i32>(1);

    var sumC0 = vec3<f32>(0.0); var sumC1 = vec3<f32>(0.0);
    var sumC2 = vec3<f32>(0.0); var sumC3 = vec3<f32>(0.0);
    var sumW = 0.0;

    for (var k: i32 = 0; k < 8; k = k + 1) {
        let ox = f32(k & 1); let oy = f32((k >> 1) & 1); let oz = f32((k >> 2) & 1);
        let vi = clamp(vec3<i32>(base) + vec3<i32>(i32(ox), i32(oy), i32(oz)), vec3<i32>(0), maxv);
        let wtri = mix(1.0 - t.x, t.x, ox) * mix(1.0 - t.y, t.y, oy) * mix(1.0 - t.z, t.z, oz);

        // probe world position -> biased-point direction -> octant + Chebyshev.
        let probe_w = gi.vol_min.xyz + vec3<f32>(vi) * sp;
        let d = Pb - probe_w;
        let r = length(d);
        let oct = i32(step(0.0, d.x)) + i32(step(0.0, d.y)) * 2 + i32(step(0.0, d.z)) * 4;
        let band = oct >> 1;
        let hi = oct & 1;
        let m4 = vis_band(band, vi);
        let mean = mix(m4.r, m4.b, f32(hi)) * dmax;
        let msq = mix(m4.g, m4.a, f32(hi)) * dmax * dmax;
        let var_ = max(msq - mean * mean, 1e-4);
        let dd = max(r - mean, 0.0);
        let cheb = var_ / (var_ + dd * dd);
        let w = wtri * max(cheb * cheb * cheb, 0.02);

        sumC0 = sumC0 + sh_load(sh_c0, vi) * w;
        sumC1 = sumC1 + sh_load(sh_c1, vi) * w;
        sumC2 = sumC2 + sh_load(sh_c2, vi) * w;
        sumC3 = sumC3 + sh_load(sh_c3, vi) * w;
        sumW = sumW + w;
    }

    let inv = 1.0 / max(sumW, 1e-4);
    let c0 = sumC0 * inv; let c1 = sumC1 * inv; let c2 = sumC2 * inv; let c3 = sumC3 * inv;
    let E = c0 * SH_A0 + SH_A1 * (c1 * n.y + c2 * n.z + c3 * n.x);
    return max(E, vec3<f32>(0.0));
}

// PUBLIC entry: distance-gated leak gate (ports the _gi.js ?gidist branch).
// Within gi.vol_inv.w metres of the camera use the Chebyshev 8-tap; beyond it (or when
// vis moments aren't bound) use the cheap HW-trilinear full-L1 path. Returns irradiance
// PRE-strength; multiply by gi.vol_min.w (giStr) at the call site (see instanced.wgsl).
fn sample_sh_irradiance(world_pos: vec3<f32>, n: vec3<f32>, cam_pos: vec3<f32>) -> vec3<f32> {
    let gate = gi.vol_inv.w;
    let has_vis = (gi.dims.w & 1u) != 0u;
    if (has_vis && gate > 0.0 && distance(world_pos, cam_pos) < gate) {
        return sh_irradiance_gated(world_pos, n);
    }
    return sh_irradiance_fast(world_pos, n);
}
