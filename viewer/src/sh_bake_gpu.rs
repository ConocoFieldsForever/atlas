//! sh_bake_gpu — vendor-neutral wgpu-compute backend for the SH irradiance bake. Ports BOTH passes of
//! the rayon baker in `sh_bake.rs` to headless `wgpu` compute (NVIDIA + AMD, no CUDA):
//!   * `pass_a_gpu`  — M1 sky-visibility + M2 shadow-tested practicals -> L1 radiance SH.
//!   * `pass_b_gpu`  — M3 one diffuse bounce: nearest-hit re-cast, trilinear irradiance gather from the
//!                     pass-A grid, per-material colored re-emission, combined with pass A.
//! Runs HEADLESS (bake-sh is a CLI branch that exits before any Bevy app exists): own Instance/Adapter/
//! Device (Vulkan on Win/Linux via `render::allowed_backends()`), driven by `bevy::tasks::block_on`.
//!
//! BATCHING: a single wgpu storage binding is capped at `max_storage_buffer_binding_size` (a u32 =>
//! <=4 GiB, and 2 GiB on the 5090/Vulkan), which interchange (~5.8 GiB tris) / Streets exceed. So the
//! tri + node BVH are CHUNKED across up to `MAX_CHUNKS` storage bindings each; `tri_at()`/`node_at()` in
//! the WGSL index the global element. Only a map needing > MAX_CHUNKS chunks, an adapter with too few
//! storage buffers/stage, no adapter, or a VRAM OOM falls back to the CPU pass (identical output).

use crate::eftpack::Light;
use crate::nav_bake::Bvh;
use glam::Vec3;

const MAX_CHUNKS: usize = 3; // MUST match sh_bake.wgsl / sh_bounce.wgsl (tris0..2 / nodes0..2)
const TRI_STRIDE: u64 = 48; // 3 x vec4<f32> (a,b,c; a.w carries the material id for pass B)
const NODE_STRIDE: u64 = 32; // 2 x vec4<f32> (min|start, max|count)
const LIGHT_STRIDE: u64 = 48; // 3 x vec4<f32>
const MAT_STRIDE: u64 = 32; // 2 x vec4<f32> (albedo|_, emissive|_)

// Constants that MUST equal sh_bake.rs's (so gpu/cpu backends agree numerically).
const GOLDEN_ANGLE: f32 = 2.399_963_2;
const RANGE_FLOOR: f32 = 4.0;
const MIN_D2: f32 = 0.25;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsA {
    gmin: [f32; 4],    // xyz grid min, w sky_scale
    spacing: [f32; 4], // xyz probe spacing, w light_scale
    dims: [u32; 4],    // nx ny nz n_dir
    counts: [u32; 4],  // n_light n_node indirect_only n_probe
    consts: [f32; 4],  // range_floor min_d2 golden_angle norm(4pi/n_dir)
    chunk: [u32; 4],   // tris_per_chunk nodes_per_chunk 0 0
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsB {
    gmin: [f32; 4],    // xyz grid min, w inv_pi_boost
    spacing: [f32; 4], // xyz probe spacing, w emis_gain
    inv_sp: [f32; 4],  // xyz 1/spacing, w bnorm
    dims: [u32; 4],    // nx ny nz bounce_rays
    counts: [u32; 4],  // n_node n_probe n_mat 0
    fconst: [f32; 4],  // max_dist golden_angle 0 0
    chunk: [u32; 4],   // tris_per_chunk nodes_per_chunk 0 0
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    name: String,
    backend: wgpu::Backend,
}

struct ChunkPlan {
    tpc: u64,
    npc: u64,
    tri_chunks: usize,
    node_chunks: usize,
    cap: u64,
}

// ---- shared setup ------------------------------------------------------------------------------

fn init_gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: crate::render::allowed_backends(),
        ..Default::default()
    });
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let (device, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sh_bake_gpu"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    Some(Gpu { device, queue, limits, name: info.name, backend: info.backend })
}

fn plan_chunks(n_tris: usize, n_nodes: usize, limits: &wgpu::Limits) -> Option<ChunkPlan> {
    let cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    // Test hook: force a smaller per-binding cap to exercise the multi-chunk path on a SMALL map.
    let cap = std::env::var("EFT_SH_GPU_CAP_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|mb| (mb << 20).clamp(TRI_STRIDE, cap))
        .unwrap_or(cap);
    let tpc = (cap / TRI_STRIDE).max(1);
    let npc = (cap / NODE_STRIDE).max(1);
    let tri_chunks = ((n_tris as u64 + tpc - 1) / tpc) as usize;
    let node_chunks = ((n_nodes as u64 + npc - 1) / npc) as usize;
    if tri_chunks > MAX_CHUNKS || node_chunks > MAX_CHUNKS {
        eprintln!(
            "  sh-bake/gpu: needs {tri_chunks} tri + {node_chunks} node chunks (> {MAX_CHUNKS}) — deferring to CPU"
        );
        return None;
    }
    Some(ChunkPlan { tpc, npc, tri_chunks, node_chunks, cap })
}

/// The tri chunk buffers (MAX_CHUNKS of them; unused slots are 1-element dummies). `with_mat` stores
/// each tri's material id in `a.w` (bitcast) for the bounce pass; pass A leaves it 0.
fn tri_bufs(device: &wgpu::Device, bvh: &Bvh, plan: &ChunkPlan, with_mat: bool) -> Vec<wgpu::Buffer> {
    let n = bvh.tris.len();
    (0..MAX_CHUNKS)
        .map(|c| {
            if c >= plan.tri_chunks {
                return dummy(device, TRI_STRIDE, "sh_bake tris(dummy)");
            }
            let start = c * plan.tpc as usize;
            let cnt = (n - start).min(plan.tpc as usize);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sh_bake tris"),
                size: cnt as u64 * TRI_STRIDE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..).get_mapped_range_mut();
                let f: &mut [f32] = bytemuck::cast_slice_mut(&mut view[..]);
                for (k, t) in bvh.tris[start..start + cnt].iter().enumerate() {
                    let o = k * 12;
                    f[o] = t.a.x; f[o + 1] = t.a.y; f[o + 2] = t.a.z;
                    f[o + 3] = if with_mat { f32::from_bits(t.mat) } else { 0.0 };
                    f[o + 4] = t.b.x; f[o + 5] = t.b.y; f[o + 6] = t.b.z; f[o + 7] = 0.0;
                    f[o + 8] = t.c.x; f[o + 9] = t.c.y; f[o + 10] = t.c.z; f[o + 11] = 0.0;
                }
            }
            buf.unmap();
            buf
        })
        .collect()
}

fn node_bufs(device: &wgpu::Device, bvh: &Bvh, plan: &ChunkPlan) -> Vec<wgpu::Buffer> {
    let n = bvh.nodes.len();
    (0..MAX_CHUNKS)
        .map(|c| {
            if c >= plan.node_chunks {
                return dummy(device, NODE_STRIDE, "sh_bake nodes(dummy)");
            }
            let start = c * plan.npc as usize;
            let cnt = (n - start).min(plan.npc as usize);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sh_bake nodes"),
                size: cnt as u64 * NODE_STRIDE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..).get_mapped_range_mut();
                let f: &mut [f32] = bytemuck::cast_slice_mut(&mut view[..]);
                for (k, nd) in bvh.nodes[start..start + cnt].iter().enumerate() {
                    let o = k * 8;
                    f[o] = nd.min.x; f[o + 1] = nd.min.y; f[o + 2] = nd.min.z;
                    f[o + 3] = f32::from_bits(nd.start);
                    f[o + 4] = nd.max.x; f[o + 5] = nd.max.y; f[o + 6] = nd.max.z;
                    f[o + 7] = f32::from_bits(nd.count);
                }
            }
            buf.unmap();
            buf
        })
        .collect()
}

fn dummy(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn storage_buf_mapped(device: &wgpu::Device, bytes: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    })
}

fn ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    bgl_entry(binding, wgpu::BufferBindingType::Storage { read_only: true })
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

/// Submit, map the readback buffer, block until the GPU finishes, then pop the OOM/validation error
/// scopes pushed by the caller. Returns the raw f32 payload, or None on any GPU error.
fn finish_read(g: &Gpu, enc: wgpu::CommandEncoder, read_buf: &wgpu::Buffer, out_bytes: u64) -> Option<Vec<f32>> {
    g.queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    read_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    if g.device.poll(wgpu::PollType::Wait).is_err() {
        eprintln!("  sh-bake/gpu: device.poll failed — deferring to CPU");
        return None;
    }
    if !matches!(rx.recv(), Ok(Ok(()))) {
        eprintln!("  sh-bake/gpu: buffer map failed — deferring to CPU");
        return None;
    }
    let oom = bevy::tasks::block_on(g.device.pop_error_scope());
    let val = bevy::tasks::block_on(g.device.pop_error_scope());
    if let Some(e) = oom.or(val) {
        eprintln!("  sh-bake/gpu: GPU error ({e}) — deferring to CPU");
        return None;
    }
    let data = read_buf.slice(..).get_mapped_range();
    let out = bytemuck::cast_slice::<u8, f32>(&data[..out_bytes as usize]).to_vec();
    drop(data);
    read_buf.unmap();
    Some(out)
}

fn to_sh4(f: &[f32], n_probe: usize) -> Vec<[Vec3; 4]> {
    let mut out = vec![[Vec3::ZERO; 4]; n_probe];
    for (pi, o) in out.iter_mut().enumerate() {
        let b = pi * 12;
        *o = [
            Vec3::new(f[b], f[b + 1], f[b + 2]),
            Vec3::new(f[b + 3], f[b + 4], f[b + 5]),
            Vec3::new(f[b + 6], f[b + 7], f[b + 8]),
            Vec3::new(f[b + 9], f[b + 10], f[b + 11]),
        ];
    }
    out
}

// ---- PASS A: sky-visibility + shadow-tested practicals -----------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn pass_a_gpu(
    bvh: &Bvh,
    lights: &[Light],
    gmin: Vec3,
    spacing: [f32; 3],
    dims: [usize; 3],
    n_dir: usize,
    sky_scale: f32,
    light_scale: f32,
    indirect_only: bool,
) -> Option<Vec<[Vec3; 4]>> {
    let n_tris = bvh.tris.len();
    let n_nodes = bvh.nodes.len();
    let [nx, ny, nz] = dims;
    let n_probe = nx * ny * nz;
    if n_tris == 0 || n_nodes == 0 || n_probe == 0 {
        return None;
    }
    let g = init_gpu()?;
    let plan = plan_chunks(n_tris, n_nodes, &g.limits)?;
    // storage buffers used = 3 tri + 3 node + lights + out = 8
    if g.limits.max_storage_buffers_per_shader_stage < 8 {
        eprintln!("  sh-bake/gpu: adapter has {} storage buffers/stage (< 8) — CPU", g.limits.max_storage_buffers_per_shader_stage);
        return None;
    }
    let out_bytes = n_probe as u64 * 12 * 4;
    let wg = ((n_probe as u64 + 63) / 64) as u32;
    if wg > g.limits.max_compute_workgroups_per_dimension {
        return None;
    }
    eprintln!(
        "  sh-bake/gpu: {} ({:?}) pass A — {n_tris} tris/{} chunk(s), {n_nodes} nodes/{} chunk(s), {} lights, {n_probe} probes, {:.1} GiB/binding",
        g.name, g.backend, plan.tri_chunks, plan.node_chunks, lights.len(), plan.cap as f64 / (1u64 << 30) as f64,
    );

    g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    g.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let tbufs = tri_bufs(&g.device, bvh, &plan, false);
    let nbufs = node_bufs(&g.device, bvh, &plan);

    // lights (>=1 element so the binding is valid; the shader loops n_light so a dummy is unread)
    let light_buf = storage_buf_mapped(&g.device, lights.len().max(1) as u64 * LIGHT_STRIDE, "sh_bake lights");
    {
        let mut view = light_buf.slice(..).get_mapped_range_mut();
        let f: &mut [f32] = bytemuck::cast_slice_mut(&mut view[..]);
        for (k, l) in lights.iter().enumerate() {
            let o = k * 12;
            f[o] = l.pos.x; f[o + 1] = l.pos.y; f[o + 2] = l.pos.z; f[o + 3] = l.range;
            f[o + 4] = l.color.x; f[o + 5] = l.color.y; f[o + 6] = l.color.z; f[o + 7] = l.cos_outer;
            f[o + 8] = l.dir.x; f[o + 9] = l.dir.y; f[o + 10] = l.dir.z; f[o + 11] = l.cos_inner;
        }
    }
    light_buf.unmap();

    let out_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bake out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bake readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = ParamsA {
        gmin: [gmin.x, gmin.y, gmin.z, sky_scale],
        spacing: [spacing[0], spacing[1], spacing[2], light_scale],
        dims: [nx as u32, ny as u32, nz as u32, n_dir as u32],
        counts: [lights.len() as u32, n_nodes as u32, indirect_only as u32, n_probe as u32],
        consts: [RANGE_FLOOR, MIN_D2, GOLDEN_ANGLE, 4.0 * std::f32::consts::PI / n_dir as f32],
        chunk: [plan.tpc as u32, plan.npc as u32, 0, 0],
    };
    let param_buf = uniform_buf(&g, bytemuck::bytes_of(&params));

    let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sh_bake.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/sh_bake.wgsl").into()),
    });
    let mut entries = vec![bgl_entry(0, wgpu::BufferBindingType::Uniform)];
    for b in 1..=7 {
        entries.push(ro(b));
    }
    entries.push(bgl_entry(8, wgpu::BufferBindingType::Storage { read_only: false }));
    let (pipeline, bgl) = compute_pipeline(&g, &shader, "cs_bake", &entries);
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sh_bake bg"),
        layout: &bgl,
        entries: &[
            be(0, &param_buf), be(1, &tbufs[0]), be(2, &tbufs[1]), be(3, &tbufs[2]),
            be(4, &nbufs[0]), be(5, &nbufs[1]), be(6, &nbufs[2]), be(7, &light_buf), be(8, &out_buf),
        ],
    });

    let t = std::time::Instant::now();
    let off = std::mem::offset_of!(ParamsA, chunk) + 8; // chunk.z
    let payload = run_batched(
        &g, &pipeline, &bind, &param_buf, bytemuck::bytes_of(&params).to_vec(), off,
        n_probe, &out_buf, &read_buf, out_bytes,
    )?;
    eprintln!("  sh-bake/gpu: pass A done in {:.2}s (GPU)", t.elapsed().as_secs_f32());
    Some(to_sh4(&payload, n_probe))
}

// ---- PASS B: one diffuse bounce ----------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn pass_b_gpu(
    bvh: &Bvh,
    sh_a: &[[Vec3; 4]],
    gmin: Vec3,
    spacing: [f32; 3],
    dims: [usize; 3],
    bounce_rays: usize,
    albedo_lut: &[Vec3],
    emis_lut: &[Vec3],
    inv_pi_boost: f32,
    emis_gain: f32,
) -> Option<Vec<[Vec3; 4]>> {
    let n_tris = bvh.tris.len();
    let n_nodes = bvh.nodes.len();
    let [nx, ny, nz] = dims;
    let n_probe = nx * ny * nz;
    if n_tris == 0 || n_nodes == 0 || n_probe == 0 || sh_a.len() != n_probe {
        return None;
    }
    let g = init_gpu()?;
    let plan = plan_chunks(n_tris, n_nodes, &g.limits)?;
    // storage buffers used = 3 tri + 3 node + sh_a + mats + out = 9
    if g.limits.max_storage_buffers_per_shader_stage < 9 {
        eprintln!("  sh-bake/gpu: adapter has {} storage buffers/stage (< 9 needed for bounce) — CPU pass B", g.limits.max_storage_buffers_per_shader_stage);
        return None;
    }
    let out_bytes = n_probe as u64 * 12 * 4;
    let wg = ((n_probe as u64 + 63) / 64) as u32;
    if wg > g.limits.max_compute_workgroups_per_dimension {
        return None;
    }
    let n_mat = albedo_lut.len().min(emis_lut.len());
    let root = bvh.nodes[0];
    let max_dist = (root.max - root.min).length() * 1.2;
    eprintln!(
        "  sh-bake/gpu: {} ({:?}) pass B — {} bounce rays, {n_mat} materials, {n_probe} probes",
        g.name, g.backend, bounce_rays,
    );

    g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    g.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let tbufs = tri_bufs(&g.device, bvh, &plan, true); // bounce needs material ids in a.w
    let nbufs = node_bufs(&g.device, bvh, &plan);

    // pass-A grid (12 f32/probe)
    let sha_buf = storage_buf_mapped(&g.device, out_bytes, "sh_bake sh_a");
    {
        let mut view = sha_buf.slice(..).get_mapped_range_mut();
        let f: &mut [f32] = bytemuck::cast_slice_mut(&mut view[..]);
        for (pi, s) in sh_a.iter().enumerate() {
            let b = pi * 12;
            for c in 0..4 {
                f[b + c * 3] = s[c].x; f[b + c * 3 + 1] = s[c].y; f[b + c * 3 + 2] = s[c].z;
            }
        }
    }
    sha_buf.unmap();

    // per-material albedo|emissive (2 vec4 each); >=1 element for a valid binding
    let mat_buf = storage_buf_mapped(&g.device, n_mat.max(1) as u64 * MAT_STRIDE, "sh_bake mats");
    {
        let mut view = mat_buf.slice(..).get_mapped_range_mut();
        let f: &mut [f32] = bytemuck::cast_slice_mut(&mut view[..]);
        for m in 0..n_mat {
            let o = m * 8;
            let a = albedo_lut[m];
            let e = emis_lut[m];
            f[o] = a.x; f[o + 1] = a.y; f[o + 2] = a.z; f[o + 3] = 0.0;
            f[o + 4] = e.x; f[o + 5] = e.y; f[o + 6] = e.z; f[o + 7] = 0.0;
        }
    }
    mat_buf.unmap();

    let out_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bounce out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bounce readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let inv_sp = [1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]];
    let bnorm = 4.0 * std::f32::consts::PI / bounce_rays as f32;
    let params = ParamsB {
        gmin: [gmin.x, gmin.y, gmin.z, inv_pi_boost],
        spacing: [spacing[0], spacing[1], spacing[2], emis_gain],
        inv_sp: [inv_sp[0], inv_sp[1], inv_sp[2], bnorm],
        dims: [nx as u32, ny as u32, nz as u32, bounce_rays as u32],
        counts: [n_nodes as u32, n_probe as u32, n_mat as u32, 0],
        fconst: [max_dist, GOLDEN_ANGLE, 0.0, 0.0],
        chunk: [plan.tpc as u32, plan.npc as u32, 0, 0],
    };
    let param_buf = uniform_buf(&g, bytemuck::bytes_of(&params));

    let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sh_bounce.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/sh_bounce.wgsl").into()),
    });
    let mut entries = vec![bgl_entry(0, wgpu::BufferBindingType::Uniform)];
    for b in 1..=8 {
        entries.push(ro(b));
    }
    entries.push(bgl_entry(9, wgpu::BufferBindingType::Storage { read_only: false }));
    let (pipeline, bgl) = compute_pipeline(&g, &shader, "cs_bounce", &entries);
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sh_bounce bg"),
        layout: &bgl,
        entries: &[
            be(0, &param_buf), be(1, &tbufs[0]), be(2, &tbufs[1]), be(3, &tbufs[2]),
            be(4, &nbufs[0]), be(5, &nbufs[1]), be(6, &nbufs[2]),
            be(7, &sha_buf), be(8, &mat_buf), be(9, &out_buf),
        ],
    });

    let t = std::time::Instant::now();
    let off = std::mem::offset_of!(ParamsB, chunk) + 8; // chunk.z
    let payload = run_batched(
        &g, &pipeline, &bind, &param_buf, bytemuck::bytes_of(&params).to_vec(), off,
        n_probe, &out_buf, &read_buf, out_bytes,
    )?;
    eprintln!("  sh-bake/gpu: pass B done in {:.2}s (GPU)", t.elapsed().as_secs_f32());
    Some(to_sh4(&payload, n_probe))
}

// ---- small wgpu helpers ------------------------------------------------------------------------

fn uniform_buf(g: &Gpu, bytes: &[u8]) -> wgpu::Buffer {
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bake params"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    g.queue.write_buffer(&buf, 0, bytes);
    buf
}

fn compute_pipeline(
    g: &Gpu,
    shader: &wgpu::ShaderModule,
    entry: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sh_bake bgl"),
        entries,
    });
    let pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sh_bake pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sh_bake pipeline"),
        layout: Some(&pl),
        module: shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (pipeline, bgl)
}

fn be<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
}

/// Adaptive probe-BATCHED dispatch: cover a subset of probes per submission and yield between them so
/// no single submit trips the OS GPU watchdog (~2 s TDR on Windows — a full-map dispatch of the costly
/// nearest-hit bounce or a giant-BVH pass A blows past it and the driver resets, discarding the output
/// as zeros). `chunk.z` carries the probe offset; batch size auto-tunes toward ~0.4 s/dispatch, so a
/// tiny map runs in a few big batches and interchange in many small ones. Returns the f32 payload.
#[allow(clippy::too_many_arguments)]
fn run_batched(
    g: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    param_buf: &wgpu::Buffer,
    mut param_bytes: Vec<u8>,
    off_pos: usize, // byte offset of the probe_offset u32 (chunk.z) within the params struct
    n_probe: usize,
    out_buf: &wgpu::Buffer,
    read_buf: &wgpu::Buffer,
    out_bytes: u64,
) -> Option<Vec<f32>> {
    // TDR-safe ADAPTIVE batching. A single dispatch of the whole probe set is FASTEST (the GPU hides
    // the ~3x regional cost variation across all probes), but a full-map bounce / giant-BVH pass
    // exceeds the ~2 s Windows GPU watchdog and is reset to zeros. So dispatch in probe batches, sizing
    // each from the PREVIOUS batch's measured rate to target ~`budget` s (encode/submit are ~free; the
    // poll wall time IS the batch's GPU time). Start small and cap growth so an early hot region can't
    // overshoot into the watchdog; small maps converge to a couple of big (efficient) batches.
    // Batch-size control by DISPATCH WALL TIME (not per-probe rate — small batches under-utilise the
    // GPU and report an inflated rate, which traps rate-based sizing at tiny batches). Double while a
    // batch runs comfortably under the ~2 s watchdog, hold in the sweet spot, shrink if it ran long.
    // Doubling is at most 2x/step from a batch already < `hi`, so the next batch (even in a ~2x hotter
    // region) stays under the watchdog. A batch that still TDRs loses the device -> poll errors -> CPU
    // fallback (+ the all-zero net below), so correctness holds regardless.
    let hi = std::env::var("EFT_SH_GPU_BATCH_S")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.9)
        .clamp(0.3, 1.4);
    let mut bsz = 4096usize;
    let mut start = 0usize;
    let mut nbatch = 0u32;
    while start < n_probe {
        let cnt = bsz.min(n_probe - start);
        param_bytes[off_pos..off_pos + 4].copy_from_slice(&(start as u32).to_le_bytes());
        g.queue.write_buffer(param_buf, 0, &param_bytes);
        let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sh_bake enc") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sh_bake pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(pipeline);
            cp.set_bind_group(0, bind, &[]);
            cp.dispatch_workgroups(((cnt as u64 + 63) / 64) as u32, 1, 1);
        }
        let t0 = std::time::Instant::now();
        g.queue.submit(Some(enc.finish()));
        if g.device.poll(wgpu::PollType::Wait).is_err() {
            eprintln!("  sh-bake/gpu: device lost mid-batch (likely watchdog) — deferring to CPU");
            return None;
        }
        let dt = t0.elapsed().as_secs_f32();
        start += cnt;
        nbatch += 1;
        bsz = if dt < hi {
            bsz * 2 // comfortably under the watchdog -> grow toward the efficient regime
        } else if dt > hi * 1.6 {
            (bsz / 2).max(2048) // ran long -> back off
        } else {
            bsz // sweet spot
        }
        .clamp(2048, n_probe);
    }
    if nbatch > 1 {
        eprintln!("  sh-bake/gpu: dispatched in {nbatch} adaptive probe-batches (TDR-safe)");
    }
    let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sh_bake copy") });
    enc.copy_buffer_to_buffer(out_buf, 0, read_buf, 0, out_bytes);
    let payload = finish_read(g, enc, read_buf, out_bytes)?;
    // TDR safety net: a watchdog reset silently zeros the whole buffer (poll still returns Ok). A real
    // bake always has structure, so a (near-)all-zero result means a batch TDR'd — fall back to CPU.
    let zeros = payload.iter().filter(|&&v| v == 0.0).count();
    if zeros as f32 > payload.len() as f32 * 0.98 {
        eprintln!("  sh-bake/gpu: result is {:.0}% zero — suspected GPU watchdog reset, deferring to CPU", 100.0 * zeros as f32 / payload.len() as f32);
        return None;
    }
    Some(payload)
}
