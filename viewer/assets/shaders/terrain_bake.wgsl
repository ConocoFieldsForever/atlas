// terrain_bake — vendor-neutral wgpu-compute port of eft_extract_v2.py::_terrain_bake_composite.
// One thread PER OUTPUT TEXEL of an R×R MicroSplat terrain-albedo bake: for every layer, weight its
// (tiled, supersampled) diffuse by the layer's splat-control channel and accumulate. Writes the raw
// (albedo_sum, weight_sum) per texel; the CPU side normalises + neutral-fills uncovered texels + PNGs.
//
// Bilinear is implemented BY HAND from storage buffers (not a hardware sampler) to match the numpy
// `_bilinear` EXACTLY: ALIGN-CORNERS (py = fy*(h-1)), clamp for the single control map (wrap=false),
// wrap for the tiling diffuse (wrap=true). Control weight is sampled ONCE at the texel center; only
// the fine-tiled diffuse gather is jittered across the ss×ss sub-samples (matches the CPU perf path).

struct Params { dims: vec4<u32> };        // x=R y=ss z=n_layers w=_
struct Tex   { off: u32, w: u32, h: u32, pad: u32 };            // off in vec4 (texel) units into `pixels`
struct Layer { ctrl: u32, ch: u32, diffuse: i32, pad: u32, rep: vec2<f32>, pad2: vec2<f32> }; // rep=(repX,repZ)

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pixels: array<vec4<f32>>;   // all textures concatenated, RGBA (diffuse: A unused)
@group(0) @binding(2) var<storage, read> texs: array<Tex>;
@group(0) @binding(3) var<storage, read> layers: array<Layer>;
@group(0) @binding(4) var<storage, read_write> out_buf: array<vec4<f32>>; // (albedo_sum.rgb, weight_sum) per texel

fn texel(t: Tex, x: i32, y: i32) -> vec4<f32> {
    return pixels[t.off + u32(y) * t.w + u32(x)];
}

// numpy `_bilinear`: py=fy*(h-1), px=fx*(w-1); floor + frac; wrap (mod) or clamp; bilinear blend.
fn bilinear(t: Tex, fy: f32, fx: f32, wrap: bool) -> vec4<f32> {
    let h = i32(t.h);
    let w = i32(t.w);
    let py = fy * f32(h - 1);
    let px = fx * f32(w - 1);
    var y0 = i32(floor(py));
    var x0 = i32(floor(px));
    var y1 = y0 + 1;
    var x1 = x0 + 1;
    let wy = py - f32(y0);
    let wx = px - f32(x0);
    if (wrap) {
        y0 = ((y0 % h) + h) % h; y1 = ((y1 % h) + h) % h;
        x0 = ((x0 % w) + w) % w; x1 = ((x1 % w) + w) % w;
    } else {
        y0 = clamp(y0, 0, h - 1); y1 = clamp(y1, 0, h - 1);
        x0 = clamp(x0, 0, w - 1); x1 = clamp(x1, 0, w - 1);
    }
    let top = mix(texel(t, x0, y0), texel(t, x1, y0), wx);
    let bot = mix(texel(t, x0, y1), texel(t, x1, y1), wx);
    return mix(top, bot, wy);
}

@compute @workgroup_size(8, 8)
fn cs_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let R = P.dims.x;
    if (gid.x >= R || gid.y >= R) { return; }
    let j = gid.x;              // column: across terrain (u)
    let i = gid.y;              // row:    down terrain  (v)
    let ss = P.dims.y;
    let n_layers = P.dims.z;
    let rf = f32(R);
    let ssf = f32(ss);
    let cy = (f32(i) + 0.5) / rf; // control center (v)
    let cx = (f32(j) + 0.5) / rf; // control center (u)

    var alb = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var L = 0u; L < n_layers; L = L + 1u) {
        let layer = layers[L];
        let cw = bilinear(texs[layer.ctrl], cy, cx, false);
        var ch4 = array<f32, 4>(cw.x, cw.y, cw.z, cw.w);
        let w = ch4[layer.ch];
        if (w <= 0.0) { continue; }
        for (var a = 0u; a < ss; a = a + 1u) {
            for (var b = 0u; b < ss; b = b + 1u) {
                var sample = vec3<f32>(0.4); // untextured layer -> neutral grey (matches CPU)
                if (layer.diffuse >= 0) {
                    let v = (f32(i) + (f32(b) + 0.5) / ssf) / rf;
                    let u = (f32(j) + (f32(a) + 0.5) / ssf) / rf;
                    sample = bilinear(texs[u32(layer.diffuse)], fract(v * layer.rep.y), fract(u * layer.rep.x), true).xyz;
                }
                alb += w * sample;
                wsum += w;
            }
        }
    }
    out_buf[i * R + j] = vec4<f32>(alb, wsum);
}
