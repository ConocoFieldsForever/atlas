//! terrain_bake — headless vendor-neutral wgpu-compute MicroSplat terrain-albedo baker, a GPU port of
//! `extraction/unity/eft_extract_v2.py::_terrain_bake_composite` (the pure-numpy composite that on
//! Reserve took ~961 s). The Python side still does the UnityPy reads (`_terrain_bake_prepare`) and
//! writes a tiny manifest + a concatenated `pixels.bin` (RGBA f32, all control + diffuse textures);
//! this subcommand uploads them, runs `terrain_bake.wgsl` (one thread per output texel), reads back the
//! raw (albedo_sum, weight_sum), normalises + neutral-fills uncovered texels, and writes the PNG.
//! Invoked `atlas bake-terrain <manifest.json>`; exits non-zero on any GPU failure so the caller falls
//! back to the numpy composite. Runs on NVIDIA + AMD (Vulkan on Win/Linux via render::allowed_backends).

use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct LayerJson {
    ctrl: u32,
    ch: u32,
    diffuse: i32,
    #[serde(rename = "repX")]
    rep_x: f32,
    #[serde(rename = "repZ")]
    rep_z: f32,
}

#[derive(serde::Deserialize)]
struct Manifest {
    #[serde(rename = "R")]
    r: u32,
    ss: u32,
    out: String,
    pixels: String,
    texs: Vec<[u32; 3]>, // [off (in RGBA texels), w, h]
    layers: Vec<LayerJson>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dims: [u32; 4], // R, ss, n_layers, 0
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TexG {
    off: u32,
    w: u32,
    h: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerG {
    ctrl: u32,
    ch: u32,
    diffuse: i32,
    pad: u32,
    rep: [f32; 2], // repX, repZ
    pad2: [f32; 2],
}

pub fn run_cli(args: &[String]) -> i32 {
    let Some(mpath) = args.first() else {
        eprintln!("usage: atlas bake-terrain <manifest.json>");
        return 2;
    };
    match bake(Path::new(mpath)) {
        Ok(secs) => {
            eprintln!("bake-terrain: OK in {secs:.2}s (GPU)");
            0
        }
        Err(e) => {
            eprintln!("bake-terrain: {e:#}");
            1
        }
    }
}

fn bake(mpath: &Path) -> Result<f32> {
    let t0 = Instant::now();
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(mpath).with_context(|| format!("reading {}", mpath.display()))?,
    )
    .context("parsing manifest")?;
    let base = mpath.parent().unwrap_or(Path::new("."));
    let resolve = |p: &str| {
        let pp = Path::new(p);
        if pp.is_absolute() { pp.to_path_buf() } else { base.join(pp) }
    };

    let r = m.r as usize;
    let n_texel = r * r;
    if r == 0 || m.layers.is_empty() || m.texs.is_empty() {
        return Err(anyhow!("empty bake (R={}, {} layers, {} texs)", m.r, m.layers.len(), m.texs.len()));
    }

    // pixels.bin: all textures concatenated as RGBA f32 (4 floats/texel).
    let pix_bytes = std::fs::read(resolve(&m.pixels)).with_context(|| format!("reading {}", m.pixels))?;
    if pix_bytes.len() % 16 != 0 {
        return Err(anyhow!("pixels.bin not a multiple of 16 bytes ({} )", pix_bytes.len()));
    }
    let texs: Vec<TexG> = m.texs.iter().map(|t| TexG { off: t[0], w: t[1], h: t[2], pad: 0 }).collect();
    let layers: Vec<LayerG> = m
        .layers
        .iter()
        .map(|l| LayerG {
            ctrl: l.ctrl,
            ch: l.ch,
            diffuse: l.diffuse,
            pad: 0,
            rep: [l.rep_x, l.rep_z],
            pad2: [0.0, 0.0],
        })
        .collect();
    let params = Params { dims: [m.r, m.ss.max(1), m.layers.len() as u32, 0] };

    // ---- headless device (Vulkan on Win/Linux via the same backends the viewer pins) ----
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: crate::render::allowed_backends(),
        ..Default::default()
    });
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|e| anyhow!("no GPU adapter ({e})"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let out_bytes = (n_texel as u64) * 16;
    if pix_bytes.len() as u64 > limits.max_storage_buffer_binding_size as u64
        || out_bytes > limits.max_storage_buffer_binding_size as u64
    {
        return Err(anyhow!("textures/output exceed the storage-binding limit"));
    }
    let (device, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("terrain_bake"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| anyhow!("request_device ({e})"))?;
    eprintln!(
        "bake-terrain: {} ({:?}) — R={} ss={} {} layers, {} textures, {} MiB pixels",
        info.name, info.backend, m.r, m.ss, m.layers.len(), m.texs.len(), pix_bytes.len() / (1 << 20)
    );

    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let mk_storage = |data: &[u8], label: &str| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len().max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut()[..data.len()].copy_from_slice(data);
        buf.unmap();
        buf
    };
    let pixels_buf = mk_storage(&pix_bytes, "pixels");
    let texs_buf = mk_storage(bytemuck::cast_slice(&texs), "texs");
    let layers_buf = mk_storage(bytemuck::cast_slice(&layers), "layers");
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let param_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain_bake.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/terrain_bake.wgsl").into()),
    });
    let ro = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: b != 4 },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    entries.extend((1..=4).map(ro));
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain_bake"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("cs_terrain"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pixels_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: texs_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: layers_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
        ],
    });

    let wg = m.r.div_ceil(8);
    if wg > limits.max_compute_workgroups_per_dimension {
        return Err(anyhow!("R={} too large for a single dispatch", m.r));
    }
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bind, &[]);
        cp.dispatch_workgroups(wg, wg, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
    queue.submit(Some(enc.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    read_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::Wait).map_err(|e| anyhow!("device.poll ({e})"))?;
    if !matches!(rx.recv(), Ok(Ok(()))) {
        return Err(anyhow!("buffer map failed"));
    }
    if let Some(e) = bevy::tasks::block_on(device.pop_error_scope()) {
        return Err(anyhow!("GPU error ({e})"));
    }

    // ---- CPU normalize + neutral fill (matches _terrain_bake_composite tail) ----
    let data = read_buf.slice(..).get_mapped_range();
    let f: &[f32] = bytemuck::cast_slice(&data);
    // fill = sum(albedo over covered) / sum(wsum over covered), else 0.4 grey
    let (mut sa, mut sw) = ([0f64; 3], 0f64);
    for k in 0..n_texel {
        let wsum = f[k * 4 + 3];
        if wsum > 1e-3 {
            sa[0] += f[k * 4] as f64;
            sa[1] += f[k * 4 + 1] as f64;
            sa[2] += f[k * 4 + 2] as f64;
            sw += wsum as f64;
        }
    }
    let fill = if sw > 1e-6 {
        [(sa[0] / sw) as f32, (sa[1] / sw) as f32, (sa[2] / sw) as f32]
    } else {
        [0.4, 0.4, 0.4]
    };
    let mut rgb = vec![0u8; n_texel * 3];
    for k in 0..n_texel {
        let wsum = f[k * 4 + 3];
        let c = if wsum > 1e-3 {
            [f[k * 4] / wsum, f[k * 4 + 1] / wsum, f[k * 4 + 2] / wsum]
        } else {
            fill
        };
        for ci in 0..3 {
            // truncate (not round) to match numpy `(out*255).astype(uint8)`
            rgb[k * 3 + ci] = (c[ci].clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    drop(data);
    read_buf.unmap();

    // Force PNG (don't infer from the extension — the caller writes to a `.gpu.tmp` for atomic replace).
    let out_path = resolve(&m.out);
    image::save_buffer_with_format(&out_path, &rgb, m.r, m.r, image::ColorType::Rgb8, image::ImageFormat::Png)
        .with_context(|| format!("writing {}", out_path.display()))?;
    Ok(t0.elapsed().as_secs_f32())
}
