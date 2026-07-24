// sh_bounce — vendor-neutral wgpu-compute PASS B of the SH bake (M3: one diffuse bounce), one thread
// PER PROBE. A port of the rayon pass in sh_bake.rs: re-cast a Fibonacci sphere; at each NEAREST
// surface hit, gather irradiance E(hit,n) from the pass-A radiance-SH grid (trilinear + cosine
// convolution) and re-emit albedo/pi * E + emissive back into the SH, then combine with pass A
// (out = passA + bounce*bnorm). Albedo/emissive come from per-material LUTs (mean of the pack's own
// PNGs, computed CPU-side). The occluder BVH is nav_bake's, CHUNKED across up to 3 storage bindings
// (tris carry their material id in a.w) exactly like sh_bake.wgsl, so giants stay on-GPU.

// start/count/mat as REAL u32 fields packed into the vec3 padding (byte layout unchanged) — loaded as
// u32 directly, never round-tripped through an f32 load (which some GPUs may denorm-flush). AMD-safe.
struct Node { min: vec3<f32>, start: u32, max: vec3<f32>, count: u32 };
struct Tri  { a: vec3<f32>, mat: u32, b: vec3<f32>, p1: u32, c: vec3<f32>, p2: u32 };

struct Params {
    gmin: vec4<f32>,    // xyz grid min, w = inv_pi_boost (albedo/pi * boost)
    spacing: vec4<f32>, // xyz probe spacing, w = emis_gain
    inv_sp: vec4<f32>,  // xyz 1/spacing, w = bnorm (4pi/bounce_rays)
    dims: vec4<u32>,    // x=nx y=ny z=nz w=bounce_rays
    counts: vec4<u32>,  // x=n_node y=n_probe z=n_mat w=unused
    fconst: vec4<f32>,  // x=max_dist y=golden_angle z,w=unused
    chunk: vec4<u32>,   // x=tris_per_chunk y=nodes_per_chunk z,w=unused
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tris0: array<Tri>;
@group(0) @binding(2) var<storage, read> tris1: array<Tri>;
@group(0) @binding(3) var<storage, read> tris2: array<Tri>;
@group(0) @binding(4) var<storage, read> nodes0: array<Node>;
@group(0) @binding(5) var<storage, read> nodes1: array<Node>;
@group(0) @binding(6) var<storage, read> nodes2: array<Node>;
@group(0) @binding(7) var<storage, read> sh_a: array<f32>;        // pass-A grid, 12 f32/probe
@group(0) @binding(8) var<storage, read> mats: array<vec4<f32>>;  // 2 vec4/material: [albedo.xyz|_ , emissive.xyz|_]
@group(0) @binding(9) var<storage, read_write> out_sh: array<f32>; // 12 f32/probe (combined)

const RAY_EPS: f32 = 0.02;

fn tri_at(i: u32) -> Tri {
    let c = i / P.chunk.x; let li = i % P.chunk.x;
    if (c == 0u) { return tris0[li]; }
    if (c == 1u) { return tris1[li]; }
    return tris2[li];
}
fn node_at(i: u32) -> Node {
    let c = i / P.chunk.y; let li = i % P.chunk.y;
    if (c == 0u) { return nodes0[li]; }
    if (c == 1u) { return nodes1[li]; }
    return nodes2[li];
}

// Moller-Trumbore: t of the crossing in (RAY_EPS, t_max), or -1.0 for a miss.
fn ray_tri_t(o: vec3<f32>, d: vec3<f32>, t: Tri, t_max: f32) -> f32 {
    let e1 = t.b.xyz - t.a.xyz;
    let e2 = t.c.xyz - t.a.xyz;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1.0e-8) { return -1.0; }
    let inv = 1.0 / det;
    let tv = o - t.a.xyz;
    let u = dot(tv, p) * inv;
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let q = cross(tv, e1);
    let v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    let tt = dot(e2, q) * inv;
    if (tt > RAY_EPS && tt < t_max) { return tt; }
    return -1.0;
}

struct Hit { t: f32, face: u32 };

// NEAREST hit, front-to-back BVH walk (child entered sooner processed first; ordering only affects
// speed, never which hit is nearest). face = 0xffffffff on miss. Matches sh_bake.rs::ray_hit.
fn ray_hit(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> Hit {
    var best_t = t_max;
    var best_face = 0xffffffffu;
    if (P.counts.x == 0u) { return Hit(best_t, best_face); }
    let inv_d = 1.0 / d;
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = node_at(stack[sp]);
        let t0 = (node.min - o) * inv_d;
        let t1 = (node.max - o) * inv_d;
        let enter = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), RAY_EPS);
        let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
        if (enter > exit || enter > best_t) { continue; }
        let count = node.count;
        let start = node.start;
        if (count > 0u) {
            for (var i = 0u; i < count; i = i + 1u) {
                let tt = ray_tri_t(o, d, tri_at(start + i), best_t);
                if (tt > 0.0 && tt < best_t) { best_t = tt; best_face = start + i; }
            }
        } else {
            let na = node_at(start);
            let nb = node_at(start + 1u);
            let pa = min((na.min - o) * inv_d, (na.max - o) * inv_d);
            let pb = min((nb.min - o) * inv_d, (nb.max - o) * inv_d);
            let ea = max(max(pa.x, pa.y), pa.z);
            let eb = max(max(pb.x, pb.y), pb.z);
            if (ea <= eb) {
                if (sp < 63u) { stack[sp] = start + 1u; sp = sp + 1u; } // far pushed first
                if (sp < 63u) { stack[sp] = start;      sp = sp + 1u; } // near popped first
            } else {
                if (sp < 63u) { stack[sp] = start;      sp = sp + 1u; }
                if (sp < 63u) { stack[sp] = start + 1u; sp = sp + 1u; }
            }
        }
    }
    return Hit(best_t, best_face);
}

fn coeff(idx: u32, c: u32) -> vec3<f32> {
    let b = idx * 12u + c * 3u;
    return vec3<f32>(sh_a[b], sh_a[b + 1u], sh_a[b + 2u]);
}

// Trilinear irradiance E(p,n) from the pass-A radiance-SH grid via cosine convolution (A0=pi, A1=2pi/3),
// clamped per channel — the SAME reconstruction as sh_bake.rs::irr_at / the viewer / bake_volume2.py.
fn irr_at(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let nx = P.dims.x; let ny = P.dims.y; let nz = P.dims.z;
    let gx = clamp((p.x - P.gmin.x) * P.inv_sp.x, 0.0, f32(nx - 1u));
    let gy = clamp((p.y - P.gmin.y) * P.inv_sp.y, 0.0, f32(ny - 1u));
    let gz = clamp((p.z - P.gmin.z) * P.inv_sp.z, 0.0, f32(nz - 1u));
    let x0 = u32(floor(gx)); let y0 = u32(floor(gy)); let z0 = u32(floor(gz));
    let x1 = min(x0 + 1u, nx - 1u); let y1 = min(y0 + 1u, ny - 1u); let z1 = min(z0 + 1u, nz - 1u);
    let fx = gx - f32(x0); let fy = gy - f32(y0); let fz = gz - f32(z0);
    var a0 = vec3<f32>(0.0); var a1 = vec3<f32>(0.0); var a2 = vec3<f32>(0.0); var a3 = vec3<f32>(0.0);
    for (var k = 0u; k < 8u; k = k + 1u) {
        var xi = x0; var wx = 1.0 - fx; if ((k & 1u) != 0u) { xi = x1; wx = fx; }
        var yi = y0; var wy = 1.0 - fy; if ((k & 2u) != 0u) { yi = y1; wy = fy; }
        var zi = z0; var wz = 1.0 - fz; if ((k & 4u) != 0u) { zi = z1; wz = fz; }
        let w = wx * wy * wz;
        let idx = (zi * ny + yi) * nx + xi;
        a0 += coeff(idx, 0u) * w;
        a1 += coeff(idx, 1u) * w;
        a2 += coeff(idx, 2u) * w;
        a3 += coeff(idx, 3u) * w;
    }
    let e = a0 * 0.8862269 + (a1 * n.y + a2 * n.z + a3 * n.x) * 1.0233267;
    return max(e, vec3<f32>(0.0));
}

fn fib_dir(i: u32, n: u32, ga: f32) -> vec3<f32> {
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = ga * f32(i);
    return vec3<f32>(r * cos(phi), z, r * sin(phi));
}
fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

@compute @workgroup_size(64)
fn cs_bounce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pi = P.chunk.z + gid.x; // chunk.z = probe batch offset (TDR-avoiding batched dispatch)
    if (pi >= P.counts.y) { return; }
    let nx = P.dims.x; let ny = P.dims.y;
    let x = pi % nx;
    let y = (pi / nx) % ny;
    let z = pi / (nx * ny);
    let o = P.gmin.xyz + vec3<f32>(f32(x), f32(y), f32(z)) * P.spacing.xyz;

    let brays = P.dims.w;
    let ga = P.fconst.y;
    let max_dist = P.fconst.x;
    let inv_pi_boost = P.gmin.w;
    let emis_gain = P.spacing.w;
    let n_mat = P.counts.z;

    var b0 = vec3<f32>(0.0); var b1 = vec3<f32>(0.0); var b2 = vec3<f32>(0.0); var b3 = vec3<f32>(0.0);
    for (var i = 0u; i < brays; i = i + 1u) {
        let d = fib_dir(i, brays, ga);
        let hit = ray_hit(o, d, max_dist);
        if (hit.face == 0xffffffffu) { continue; }
        let tri = tri_at(hit.face);
        let cr = cross(tri.b.xyz - tri.a.xyz, tri.c.xyz - tri.a.xyz);
        let clen = length(cr);
        var n = select(vec3<f32>(0.0), cr / clen, clen > 1.0e-12); // normalize_or_zero
        if (dot(n, d) > 0.0) { n = -n; }                            // orient toward the incoming ray
        let h = o + d * hit.t + n * 0.05;                           // hit point, nudged off surface
        let e = irr_at(h, n);
        let mat = tri.mat;
        var alb = vec3<f32>(0.3); var emi = vec3<f32>(0.0);         // untextured/oob fallback (matches CPU)
        if (mat < n_mat) { alb = mats[2u * mat].xyz; emi = mats[2u * mat + 1u].xyz; }
        let rad = e * alb * inv_pi_boost + emi * emis_gain;         // albedo/pi * E + emissive
        let bs = sh_basis(d);
        b0 += rad * bs.x; b1 += rad * bs.y; b2 += rad * bs.z; b3 += rad * bs.w;
    }

    let bnorm = P.inv_sp.w;
    let base = pi * 12u;
    // combined = passA + bounce hemisphere integral
    let c0 = coeff(pi, 0u) + b0 * bnorm;
    let c1 = coeff(pi, 1u) + b1 * bnorm;
    let c2 = coeff(pi, 2u) + b2 * bnorm;
    let c3 = coeff(pi, 3u) + b3 * bnorm;
    out_sh[base + 0u] = c0.x; out_sh[base + 1u]  = c0.y; out_sh[base + 2u]  = c0.z;
    out_sh[base + 3u] = c1.x; out_sh[base + 4u]  = c1.y; out_sh[base + 5u]  = c1.z;
    out_sh[base + 6u] = c2.x; out_sh[base + 7u]  = c2.y; out_sh[base + 8u]  = c2.z;
    out_sh[base + 9u] = c3.x; out_sh[base + 10u] = c3.y; out_sh[base + 11u] = c3.z;
}
