//! sh_bake_gpu — vendor-neutral wgpu-compute backend for PASS A of the SH irradiance bake
//! (sky-visibility M1 + shadow-tested practicals M2). A GPU port of the rayon pass in `sh_bake.rs`,
//! sharing `nav_bake`'s occluder BVH. Runs HEADLESS: `bake-sh` is a CLI branch that exits before any
//! Bevy app exists, so this creates its own `wgpu` Instance/Adapter/Device (Vulkan on Win/Linux via
//! the same `render::allowed_backends()` the viewer pins) and drives the async wgpu calls with
//! `bevy::tasks::block_on`.
//!
//! BATCHING (the whole point): a single wgpu storage binding is capped at
//! `max_storage_buffer_binding_size` — a **u32**, so <= 4 GiB. Interchange (~120 M tris ≈ 5.8 GiB) and
//! Streets exceed that in ONE buffer. Rather than defer those giants to CPU, the tri + node arrays are
//! CHUNKED across up to `MAX_CHUNKS` storage bindings each (`tris0..2`, `nodes0..2` in the WGSL);
//! `tri_at()`/`node_at()` index the global element across chunks. Only a map that would need > MAX_CHUNKS
//! chunks, an adapter with < 8 storage buffers/stage, no adapter at all, or a genuine VRAM OOM falls
//! back to CPU (the caller uses the rayon pass — identical output format).

use crate::eftpack::Light;
use crate::nav_bake::Bvh;
use glam::Vec3;

const MAX_CHUNKS: usize = 3; // MUST match sh_bake.wgsl (tris0..2 / nodes0..2)
const TRI_STRIDE: u64 = 48; // 3 x vec4<f32> (only a,b,c used; .w padding)
const NODE_STRIDE: u64 = 32; // 2 x vec4<f32> (min|start, max|count)
const LIGHT_STRIDE: u64 = 48; // 3 x vec4<f32> (pos|range, color|cos_outer, dir|cos_inner)

// Constants that MUST equal sh_bake.rs's (so gpu/cpu backends agree numerically).
const GOLDEN_ANGLE: f32 = 2.399_963_2; // sh_bake::fib_dir GA
const RANGE_FLOOR: f32 = 4.0; // LIGHT_RANGE_FLOOR
const MIN_D2: f32 = 0.25; // MIN_D2

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    gmin: [f32; 4],    // xyz grid min, w sky_scale
    spacing: [f32; 4], // xyz probe spacing, w light_scale
    dims: [u32; 4],    // nx ny nz n_dir
    counts: [u32; 4],  // n_light n_node indirect_only n_probe
    consts: [f32; 4],  // range_floor min_d2 golden_angle norm(4pi/n_dir)
    chunk: [u32; 4],   // tris_per_chunk nodes_per_chunk 0 0
}

/// GPU PASS A. Returns one `[Vec3;4]` L1 radiance-SH per probe (probe idx = ((z*ny)+y)*nx + x), or
/// `None` if the GPU can't/shouldn't take this bake (caller then runs the CPU pass). Never panics.
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

    // ---- headless device (Vulkan on Win/Linux, Metal/all elsewhere — same as the viewer) ----
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

    // ---- size-gate + chunk plan (bounded by per-binding u32 AND per-buffer limits) ----
    let cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    // Test hook: EFT_SH_GPU_CAP_MB forces a smaller per-binding cap so the multi-chunk batching path
    // can be exercised on a SMALL map (no 5.8 GiB giant needed). Ignored in normal builds.
    let cap = std::env::var("EFT_SH_GPU_CAP_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|mb| (mb << 20).clamp(TRI_STRIDE, cap))
        .unwrap_or(cap);
    let tpc = (cap / TRI_STRIDE).max(1); // tris per chunk
    let npc = (cap / NODE_STRIDE).max(1); // nodes per chunk
    let tri_chunks = ((n_tris as u64 + tpc - 1) / tpc) as usize;
    let node_chunks = ((n_nodes as u64 + npc - 1) / npc) as usize;
    if tri_chunks > MAX_CHUNKS || node_chunks > MAX_CHUNKS {
        eprintln!(
            "  sh-bake/gpu: needs {tri_chunks} tri + {node_chunks} node chunks (> {MAX_CHUNKS}) — deferring to CPU"
        );
        return None;
    }
    // storage buffers used = MAX_CHUNKS tris + MAX_CHUNKS nodes + lights + out = 3+3+1+1 = 8.
    if limits.max_storage_buffers_per_shader_stage < 8 {
        eprintln!(
            "  sh-bake/gpu: adapter exposes {} storage buffers/stage (< 8) — deferring to CPU",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }
    let out_bytes = n_probe as u64 * 12 * 4;
    if out_bytes > cap {
        return None; // 124 MB vs 4 GiB cap — never; guard anyway
    }
    let wg = ((n_probe as u64 + 63) / 64) as u32;
    if wg > limits.max_compute_workgroups_per_dimension {
        return None; // 2.6 M probes / 64 ≈ 41 k < 65535 — never; guard anyway
    }

    let (device, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sh_bake_gpu"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;

    eprintln!(
        "  sh-bake/gpu: {} ({:?}) — {n_tris} tris/{tri_chunks} chunk(s), {n_nodes} nodes/{node_chunks} chunk(s), {} lights, {n_probe} probes, {:.1} GiB/binding",
        info.name,
        info.backend,
        lights.len(),
        cap as f64 / (1u64 << 30) as f64,
    );

    // Catch a genuine VRAM exhaustion (buffers within limits but no memory) and the odd validation
    // slip, converting either into a clean CPU fallback instead of a panic or silent zeros.
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    // ---- upload BVH, chunked. Real chunks are filled via a creation-time mapping (no host copy);
    //      unused chunk slots get a 1-element dummy so every declared binding is bound. ----
    let mut tri_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(MAX_CHUNKS);
    for c in 0..MAX_CHUNKS {
        if c < tri_chunks {
            let start = c * tpc as usize;
            let cnt = (n_tris - start).min(tpc as usize);
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
                    f[o] = t.a.x; f[o + 1] = t.a.y; f[o + 2] = t.a.z; f[o + 3] = 0.0;
                    f[o + 4] = t.b.x; f[o + 5] = t.b.y; f[o + 6] = t.b.z; f[o + 7] = 0.0;
                    f[o + 8] = t.c.x; f[o + 9] = t.c.y; f[o + 10] = t.c.z; f[o + 11] = 0.0;
                }
            }
            buf.unmap();
            tri_bufs.push(buf);
        } else {
            tri_bufs.push(dummy(&device, TRI_STRIDE, "sh_bake tris(dummy)"));
        }
    }

    let mut node_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(MAX_CHUNKS);
    for c in 0..MAX_CHUNKS {
        if c < node_chunks {
            let start = c * npc as usize;
            let cnt = (n_nodes - start).min(npc as usize);
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
            node_bufs.push(buf);
        } else {
            node_bufs.push(dummy(&device, NODE_STRIDE, "sh_bake nodes(dummy)"));
        }
    }

    // ---- lights (>=1 element so the binding is valid; the shader loops n_light, so a dummy is unread) ----
    let n_light = lights.len();
    let light_buf = {
        let elems = n_light.max(1);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sh_bake lights"),
            size: elems as u64 * LIGHT_STRIDE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        {
            let mut view = buf.slice(..).get_mapped_range_mut();
            let f: &mut [f32] = bytemuck::cast_slice_mut(&mut view[..]);
            for (k, l) in lights.iter().enumerate() {
                let o = k * 12;
                f[o] = l.pos.x; f[o + 1] = l.pos.y; f[o + 2] = l.pos.z; f[o + 3] = l.range;
                f[o + 4] = l.color.x; f[o + 5] = l.color.y; f[o + 6] = l.color.z; f[o + 7] = l.cos_outer;
                f[o + 8] = l.dir.x; f[o + 9] = l.dir.y; f[o + 10] = l.dir.z; f[o + 11] = l.cos_inner;
            }
        }
        buf.unmap();
        buf
    };

    // ---- output + readback staging ----
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bake out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bake readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ---- params ----
    let norm = 4.0 * std::f32::consts::PI / n_dir as f32;
    let params = Params {
        gmin: [gmin.x, gmin.y, gmin.z, sky_scale],
        spacing: [spacing[0], spacing[1], spacing[2], light_scale],
        dims: [nx as u32, ny as u32, nz as u32, n_dir as u32],
        counts: [n_light as u32, n_nodes as u32, indirect_only as u32, n_probe as u32],
        consts: [RANGE_FLOOR, MIN_D2, GOLDEN_ANGLE, norm],
        chunk: [tpc as u32, npc as u32, 0, 0],
    };
    let param_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sh_bake params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));

    // ---- pipeline ----
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sh_bake.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/sh_bake.wgsl").into()),
    });
    // bindings: 0 uniform, 1..3 tris, 4..6 nodes, 7 lights, 8 out(rw)
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::with_capacity(9);
    entries.push(bgl_entry(0, buffer_ty(wgpu::BufferBindingType::Uniform)));
    for b in 1..=7 {
        entries.push(bgl_entry(b, buffer_ty(wgpu::BufferBindingType::Storage { read_only: true })));
    }
    entries.push(bgl_entry(8, buffer_ty(wgpu::BufferBindingType::Storage { read_only: false })));
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sh_bake bgl"),
        entries: &entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sh_bake pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sh_bake pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("cs_bake"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sh_bake bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tri_bufs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tri_bufs[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: tri_bufs[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: node_bufs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: node_bufs[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: node_bufs[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: out_buf.as_entire_binding() },
        ],
    });

    // ---- dispatch + copy to readback ----
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sh_bake enc"),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sh_bake pass"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bind, &[]);
        cp.dispatch_workgroups(wg, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
    let t = std::time::Instant::now();
    queue.submit(Some(enc.finish()));

    // ---- map + block until the GPU is done ----
    let (tx, rx) = std::sync::mpsc::channel();
    read_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    if device.poll(wgpu::PollType::Wait).is_err() {
        eprintln!("  sh-bake/gpu: device.poll failed — deferring to CPU");
        return None;
    }
    match rx.recv() {
        Ok(Ok(())) => {}
        _ => {
            eprintln!("  sh-bake/gpu: buffer map failed — deferring to CPU");
            return None;
        }
    }

    // ---- OOM / validation backstop (pop inner OOM first, then outer Validation) ----
    let oom = bevy::tasks::block_on(device.pop_error_scope());
    let val = bevy::tasks::block_on(device.pop_error_scope());
    if let Some(e) = oom.or(val) {
        eprintln!("  sh-bake/gpu: GPU error ({e}) — deferring to CPU");
        return None;
    }

    let data = read_buf.slice(..).get_mapped_range();
    let f: &[f32] = bytemuck::cast_slice(&data);
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
    drop(data);
    read_buf.unmap();
    eprintln!("  sh-bake/gpu: pass A done in {:.2}s (GPU)", t.elapsed().as_secs_f32());
    Some(out)
}

fn dummy(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn buffer_ty(ty: wgpu::BufferBindingType) -> wgpu::BindingType {
    wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }
}

fn bgl_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
