// SSR (Phase 6, docs/GRAPHICS_PLAN.md) — screen-space reflections over the prepass substrate.
// Runs AFTER the SSAO composite and BEFORE TAA (TAA then smooths the trace noise — the plan's
// ordering), on the HDR ping-pong. Opt-in via EFT_SSR.
//
// Per-pixel gates, all from the prepass targets:
//   class == 0        -> untouched (sky/blend/grass: no surface data).
//   roughness > 0.35  -> untouched (rough surfaces keep the analytic/SH reflection already in the
//                        color; SSR on rough surfaces needs a filtered color pyramid we don't
//                        build in v1 — GRAPHICS_PLAN.md lists it under the full hierarchical SSR).
//   else              -> march the reflection ray in screen space against the prepass depth and
//                        BLEND toward the hit color by a fresnel-and-gloss weight.
//
// The march is fixed-step + binary refinement against the 1x prepass depth (mip 0 of the pyramid
// equals it; v1 does not walk the mip chain). Misses keep the existing color — which already
// contains the analytic sky/SH reflection, so a miss degrades to exactly today's look, never to
// black. That is the composition rule that makes post-hoc SSR safe: it can only REPLACE reflection
// where it found real scene geometry to reflect.
//
// WATER (class in [2,3]): the prepass normal is the FLAT plane normal (the animated chop lives in
// the forward shader only — Codex review §1), so water traces as a calm mirror. That reads well on
// a lake at distance; the chop's own glint stays in the base color. Sharper integration (tracing
// the animated normal) requires the prepass water path from Phase 5's displacement work.

struct SsrParams {
    clip_from_world: mat4x4<f32>,   // current, unjittered
    inv_proj: mat4x4<f32>,          // view-from-clip
    world_from_view: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    // x = max distance (m), y = stride px, z = intensity, w = viewport height px.
    p: vec4<f32>,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var depth_tex: texture_depth_2d;       // prepass 1x depth
@group(0) @binding(3) var normal_tex: texture_2d<f32>;       // prepass normal + class/roughness (w)
@group(0) @binding(4) var<uniform> ssr: SsrParams;

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

fn view_z_at(uv: vec2<f32>, dims: vec2<i32>) -> f32 {
    let px = clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0), dims - 1);
    let d = textureLoad(depth_tex, px, 0);
    if (d <= 1e-7) {
        return -1.0e9; // sky: infinitely far
    }
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, d);
    let v = ssr.inv_proj * vec4<f32>(ndc, 1.0);
    return v.z / v.w; // negative (view looks down -Z)
}

@fragment
fn fs_ssr(in: FsIn) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(scene_tex, scene_samp, in.uv, 0.0);
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let px = clamp(vec2<i32>(in.clip.xy), vec2<i32>(0), dims - 1);
    let nr = textureLoad(normal_tex, px, 0);
    let class_w = nr.w;
    if (class_w <= 0.0) {
        return color;
    }
    let is_water = class_w >= 2.0;
    let rough = select(class_w, class_w - 2.0, is_water);
    if (rough > 0.35) {
        return color;
    }
    let d = textureLoad(depth_tex, px, 0);
    if (d <= 1e-7) {
        return color;
    }

    // Reconstruct world position + reflection ray.
    let ndc = vec3<f32>(in.uv.x * 2.0 - 1.0, 1.0 - 2.0 * in.uv.y, d);
    let vp4 = ssr.inv_proj * vec4<f32>(ndc, 1.0);
    let vpos = vp4.xyz / vp4.w;
    let wpos = (ssr.world_from_view * vec4<f32>(vpos, 1.0)).xyz;
    let cam = (ssr.world_from_view * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let V = normalize(cam - wpos);
    let N = normalize(nr.xyz);
    let NdV = max(dot(N, V), 1e-3);
    let R = reflect(-V, N);
    if (dot(R, N) <= 0.0) {
        return color;
    }

    // March in WORLD space, project each sample: robust at grazing angles where screen-space DDA
    // steps collapse. Step grows geometrically so nearby detail is precise and range is covered.
    var t = 0.15;
    var hit_uv = vec2<f32>(-1.0);
    var prev_t = 0.0;
    let max_t = ssr.p.x;
    for (var i = 0; i < 28; i++) {
        let p = wpos + R * t;
        let clip = ssr.clip_from_world * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {
            break;
        }
        let pn = clip.xyz / clip.w;
        let uv = vec2<f32>(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
        if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
            break;
        }
        let ray_vz = (ssr.view_from_world * vec4<f32>(p, 1.0)).z;
        let sur_vz = view_z_at(uv, dims);
        // Behind the surface within a thickness bound = hit. Thickness scales with distance so
        // thin railings at range don't swallow the ray.
        let thick = 0.35 + 0.03 * t;
        if (sur_vz > ray_vz && sur_vz - ray_vz < thick) {
            // Binary refine between prev_t and t.
            var lo = prev_t;
            var hi = t;
            for (var r = 0; r < 5; r++) {
                let mid = 0.5 * (lo + hi);
                let pm = wpos + R * mid;
                let cm = ssr.clip_from_world * vec4<f32>(pm, 1.0);
                let um = vec2<f32>(cm.x / cm.w * 0.5 + 0.5, 0.5 - cm.y / cm.w * 0.5);
                let rz = (ssr.view_from_world * vec4<f32>(pm, 1.0)).z;
                if (view_z_at(um, dims) > rz) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let ph = wpos + R * hi;
            let ch = ssr.clip_from_world * vec4<f32>(ph, 1.0);
            hit_uv = vec2<f32>(ch.x / ch.w * 0.5 + 0.5, 0.5 - ch.y / ch.w * 0.5);
            break;
        }
        prev_t = t;
        t = t * 1.22 + 0.05;
        if (t > max_t) {
            break;
        }
    }
    if (hit_uv.x < 0.0) {
        return color; // miss: the analytic reflection already in `color` stands
    }
    // Edge fade: hits near the screen border pop as the camera turns; fade them out.
    let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_w = smoothstep(0.0, 0.08, edge);
    let refl = textureSampleLevel(scene_tex, scene_samp, hit_uv, 0.0).rgb;

    // Fresnel (Schlick, F0=0.02 dielectric) x gloss x class weight. Water gets a floor so a calm
    // lake actually mirrors; everything else stays subtle and only fully replaces at grazing.
    var f = 0.02 + 0.98 * pow(1.0 - NdV, 5.0);
    let gloss = 1.0 - rough / 0.35;
    var w = f * gloss * edge_w * ssr.p.z;
    if (is_water) {
        w = max(w, 0.35 * gloss * edge_w * ssr.p.z);
    }
    return vec4<f32>(mix(color.rgb, refl, clamp(w, 0.0, 0.9)), color.a);
}
