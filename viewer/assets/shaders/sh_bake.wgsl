// sh_bake — vendor-neutral wgpu-compute PASS A of the SH irradiance bake (sky-visibility + shadow-
// tested practicals -> L1 radiance SH), one thread PER PROBE. A faithful port of the rayon CPU pass
// in sh_bake.rs (fib_dir / sh_basis / sky / the Moller-Trumbore any-hit BVH occlusion + the M2 light
// attenuation), so `--backend gpu` and `--backend cpu` produce matching volumes. The occluder BVH is
// `nav_bake`'s (shared). To keep GIANT maps (interchange/streets, >4 GiB of tris) ON the GPU rather
// than deferring to CPU, the tri + node arrays are CHUNKED across up to 3 storage bindings each
// (a single wgpu storage binding is capped at max_storage_buffer_binding_size, a u32 => <=4 GiB);
// tri_at()/node_at() index the global element across chunks. Maps that would need >3 chunks, or that
// exhaust VRAM, are the only ones that fall back to CPU.

const MAX_CHUNKS: u32 = 3u;   // keep in lockstep with sh_bake_gpu.rs (bindings tris0..2 / nodes0..2)

struct Node { lo: vec4<f32>, hi: vec4<f32> };   // lo.xyz=aabb min, lo.w=bitcast(start); hi.xyz=max, hi.w=bitcast(count)
struct Tri  { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };          // .xyz = world verts (all tris occlude, incl doors)
struct Light { l0: vec4<f32>, l1: vec4<f32>, l2: vec4<f32> };      // l0=pos+range, l1=color+cos_outer, l2=dir+cos_inner

struct Params {
    gmin: vec4<f32>,      // xyz = grid min corner, w = sky_scale
    spacing: vec4<f32>,   // xyz = probe spacing (m), w = light_scale
    dims: vec4<u32>,      // x=nx y=ny z=nz w=n_dir
    counts: vec4<u32>,    // x=n_light y=n_node z=indirect_only(0/1) w=n_probe
    consts: vec4<f32>,    // x=range_floor y=min_d2 z=golden_angle w=norm(4pi/n_dir)
    chunk: vec4<u32>,     // x=tris_per_chunk y=nodes_per_chunk z,w=unused
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tris0: array<Tri>;
@group(0) @binding(2) var<storage, read> tris1: array<Tri>;
@group(0) @binding(3) var<storage, read> tris2: array<Tri>;
@group(0) @binding(4) var<storage, read> nodes0: array<Node>;
@group(0) @binding(5) var<storage, read> nodes1: array<Node>;
@group(0) @binding(6) var<storage, read> nodes2: array<Node>;
@group(0) @binding(7) var<storage, read> lights: array<Light>;
@group(0) @binding(8) var<storage, read_write> out_sh: array<f32>;   // 12 floats (4 coeffs x rgb) per probe

const RAY_EPS: f32 = 0.02;   // matches sh_bake.rs RAY_EPS (origin push-off / slab entry clamp)

// Global element access across the (up to 3) chunk bindings — chunk = idx / per_chunk, local = idx % per_chunk.
// Small maps use one chunk, so idx < per_chunk => c==0 always (the extra bindings are bound to dummies).
fn tri_at(i: u32) -> Tri {
    let c = i / P.chunk.x;
    let li = i % P.chunk.x;
    if (c == 0u) { return tris0[li]; }
    if (c == 1u) { return tris1[li]; }
    return tris2[li];
}
fn node_at(i: u32) -> Node {
    let c = i / P.chunk.y;
    let li = i % P.chunk.y;
    if (c == 0u) { return nodes0[li]; }
    if (c == 1u) { return nodes1[li]; }
    return nodes2[li];
}

// Moller-Trumbore any-hit: does o + t*d cross the tri for t in (RAY_EPS, t_max)?
fn ray_tri(o: vec3<f32>, d: vec3<f32>, t: Tri, t_max: f32) -> bool {
    let e1 = t.b.xyz - t.a.xyz;
    let e2 = t.c.xyz - t.a.xyz;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1.0e-8) { return false; }
    let inv = 1.0 / det;
    let tv = o - t.a.xyz;
    let u = dot(tv, p) * inv;
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(tv, e1);
    let v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) { return false; }
    let tt = dot(e2, q) * inv;
    return tt > RAY_EPS && tt < t_max;
}

// Ray-AABB slab test, MATCHING sh_bake.rs: entry clamped to RAY_EPS, accept when entry<=exit && entry<=t_max.
fn slab_hit(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv_d: vec3<f32>, t_max: f32) -> bool {
    let t0 = (lo - o) * inv_d;
    let t1 = (hi - o) * inv_d;
    let entry = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), RAY_EPS);
    let exit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return entry <= exit && entry <= t_max;
}

// Any-hit occlusion: is o + t*d (t in (RAY_EPS, t_max)) blocked by ANY triangle? Iterative BVH walk
// with a fixed stack (maps that need >64 deep aren't produced by nav_bake's median-split builder).
fn ray_occluded(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> bool {
    if (P.counts.y == 0u) { return false; }
    let inv_d = 1.0 / d;                 // component may be +-inf when d==0; slab_hit handles it
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = node_at(stack[sp]);
        if (!slab_hit(node.lo.xyz, node.hi.xyz, o, inv_d, t_max)) { continue; }
        let count = bitcast<u32>(node.hi.w);
        let start = bitcast<u32>(node.lo.w);
        if (count > 0u) {
            for (var i = 0u; i < count; i = i + 1u) {
                if (ray_tri(o, d, tri_at(start + i), t_max)) { return true; }
            }
        } else {
            if (sp < 63u) { stack[sp] = start;       sp = sp + 1u; }
            if (sp < 63u) { stack[sp] = start + 1u;  sp = sp + 1u; }
        }
    }
    return false;
}

fn fib_dir(i: u32, n: u32, ga: f32) -> vec3<f32> {
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = ga * f32(i);
    return vec3<f32>(r * cos(phi), z, r * sin(phi));  // Y-up
}

fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

fn sky(d: vec3<f32>, scale: f32) -> f32 {
    return (0.35 + 0.75 * max(clamp(d.y, -1.0, 1.0), 0.0)) * scale;
}

@compute @workgroup_size(64)
fn cs_bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pi = P.chunk.z + gid.x; // chunk.z = probe batch offset (TDR-avoiding batched dispatch)
    if (pi >= P.counts.w) { return; }

    // probe origin (idx = ((z*ny)+y)*nx + x) — matches sh_bake.rs::probe_o
    let nx = P.dims.x; let ny = P.dims.y;
    let x = pi % nx;
    let y = (pi / nx) % ny;
    let z = pi / (nx * ny);
    let o = P.gmin.xyz + vec3<f32>(f32(x), f32(y), f32(z)) * P.spacing.xyz;

    let n_dir = P.dims.w;
    let ga = P.consts.z;
    // per-coeff rgb accumulators (radiance SH)
    var c0 = vec3<f32>(0.0); var c1 = vec3<f32>(0.0); var c2 = vec3<f32>(0.0); var c3 = vec3<f32>(0.0);

    // --- M1: sky-visibility ---
    for (var i = 0u; i < n_dir; i = i + 1u) {
        let d = fib_dir(i, n_dir, ga);
        if (ray_occluded(o, d, 1.0e30)) { continue; }
        let l = vec3<f32>(sky(d, P.gmin.w));          // neutral grayscale sky -> rgb
        let b = sh_basis(d);
        c0 += l * b.x; c1 += l * b.y; c2 += l * b.z; c3 += l * b.w;
    }
    let norm = P.consts.w;                            // 4pi / n_dir (solid-angle weight)
    c0 *= norm; c1 *= norm; c2 *= norm; c3 *= norm;

    // --- M2: shadow-tested practicals (delta lights, NOT weighted by norm), unless indirect-only ---
    if (P.counts.z == 0u) {
        let light_scale = P.spacing.w;
        let range_floor = P.consts.x;
        let min_d2 = P.consts.y;
        let n_light = P.counts.x;
        for (var li = 0u; li < n_light; li = li + 1u) {
            let L = lights[li];
            let tol = L.l0.xyz - o;
            let dist = length(tol);
            let r = max(L.l0.w, range_floor);
            if (dist <= 0.05 || dist >= r) { continue; }
            let dl = tol / dist;
            var spot = 1.0;
            if (L.l1.w > -1.5) {                       // cos_outer sentinel -2 => point light
                let cosang = -dot(dl, L.l2.xyz);
                spot = clamp((cosang - L.l1.w) / (L.l2.w - L.l1.w + 1.0e-4), 0.0, 1.0);
            }
            if (spot <= 0.0) { continue; }
            let xx = dist / r;
            let win = clamp(1.0 - xx * xx * xx * xx, 0.0, 1.0);
            let at = win * win / max(dist * dist, min_d2);
            if (ray_occluded(o, dl, dist - 0.1)) { continue; }   // shadowed
            let rad = L.l1.xyz * (at * spot * light_scale);
            let b = sh_basis(dl);
            c0 += rad * b.x; c1 += rad * b.y; c2 += rad * b.z; c3 += rad * b.w;
        }
    }

    let base = pi * 12u;
    out_sh[base + 0u] = c0.x; out_sh[base + 1u]  = c0.y; out_sh[base + 2u]  = c0.z;
    out_sh[base + 3u] = c1.x; out_sh[base + 4u]  = c1.y; out_sh[base + 5u]  = c1.z;
    out_sh[base + 6u] = c2.x; out_sh[base + 7u]  = c2.y; out_sh[base + 8u]  = c2.z;
    out_sh[base + 9u] = c3.x; out_sh[base + 10u] = c3.y; out_sh[base + 11u] = c3.z;
}
