//! eft::grade — the REAL EFT display chain as a post pass (shaders/grade.wgsl).
//!
//! The game's grade was fitted offline (Hejl-Dawson tonemap with EFT's constants → per-channel
//! film curves → fitted "Fahrenheit" stage) and baked into a 64³ LUT (make_grade_lut.py →
//! eft_grade_lut.bin, a 512×512 RGBA8 atlas of 8×8 64×64 tiles). This module wires that LUT
//! into Bevy's render graph between Bloom and Tonemapping, replacing the hand-tuned
//! TonyMcMapface + ColorGrading approximation with the ACTUAL game look — identical on every
//! map, because it is derived from the game, not tuned per screenshot.
//!
//! Color management (the one subtle bit): the web viewer wrote the LUT's display-encoded bytes
//! RAW to the canvas. Bevy's upscaling node blits through an sRGB swapchain view, which would
//! encode a second time. So at LOAD we invert the display encode per texel (sRGB EOTF → linear):
//! the pass then outputs LINEAR, the swapchain encode re-applies the transfer function once,
//! and the presented bytes match the reference. The LUT is stored Rgba16Float (8-bit linear
//! would band in the toe).
//!
//! LUT resolution order: EFT_GRADE_LUT env → <pack>/grade_lut.bin → tarkmap/out default.
//! EFT_GRADE=0 disables the pass entirely (camera falls back to TonyMcMapface + hand grade).
//! EFT_GRADE_EXPOSURE overrides the pre-LUT exposure (the default is shared with
//! `GfxSettings`; the web viewer's 0.18 was tuned for a different radiance scale).
//! EFT_VIGNETTE=0 zeroes the PRISM vignette strength.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_graph::{NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner},
    render_resource::{
        binding_types::{sampler, texture_2d, texture_3d, uniform_buffer_sized},
        AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
        BufferInitDescriptor, BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites,
        Extent3d, FilterMode, FragmentState, LoadOp, MultisampleState, Operations, PipelineCache,
        PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, StoreOp, Texture,
        TextureDataOrder, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
        TextureUsages, TextureView, TextureViewDescriptor, VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::ViewTarget,
    RenderApp, RenderStartup,
};
use bytemuck::{Pod, Zeroable};

/// CPU-side LUT + params, resolved in the MAIN world (so main.rs can also branch the camera's
/// Tonemapping on whether the grade is active) and extracted to the render world for upload.
#[derive(Resource, Clone, ExtractResource)]
pub struct GradeLutCpu {
    /// 64³ texels, Rgba16Float LE bytes, x=R fastest → y=G → z=B (texture_3d upload order).
    /// Texels are LINEAR (display encode inverted at load — see module docs).
    pub texels: Vec<u8>,
    /// Pre-LUT exposure multiplier.
    pub exposure: f32,
    /// PRISM vignette strength (0 = off).
    pub vignette: f32,
}

/// Resolve + load + repack the LUT. `None` when disabled (EFT_GRADE=0) or unresolvable.
pub fn load_grade_lut(pack_root: Option<&std::path::Path>) -> Option<GradeLutCpu> {
    if std::env::var("EFT_GRADE").map(|v| v.trim() == "0").unwrap_or(false) {
        info!("grade: EFT_GRADE=0 — game grade LUT disabled (TonyMcMapface fallback)");
        return None;
    }
    // Resolution order mirrors loot.json: env override → pack-local → repo default.
    let candidates: Vec<std::path::PathBuf> = [
        std::env::var("EFT_GRADE_LUT").ok().map(std::path::PathBuf::from),
        pack_root.map(|r| r.join("grade_lut.bin")),
        pack_root.and_then(|r| r.parent()).map(|p| p.join("shared").join("grade_lut.bin")),
        Some(crate::paths::shared_dir().join("grade_lut.bin")),
    ]
    .into_iter()
    .flatten()
    .collect();
    let path = candidates.iter().find(|p| p.is_file())?;
    let atlas = match std::fs::read(path) {
        Ok(b) if b.len() == 512 * 512 * 4 => b,
        Ok(b) => {
            warn!("grade: {} is {} bytes (want 1048576) — ignoring", path.display(), b.len());
            return None;
        }
        Err(e) => {
            warn!("grade: {} unreadable ({e}) — TonyMcMapface fallback", path.display());
            return None;
        }
    };

    // Repack the 8×8-tiled 512² atlas into a 64³ volume. Baker layout (make_grade_lut.py):
    // slice b → tile (b%8, b/8); within a tile x=R index, y=G index. Texel (r,g,b) lives at
    // atlas[row = (b/8)*64 + g][col = (b%8)*64 + r]. Invert the display encode (sRGB EOTF) so
    // the LUT emits LINEAR, and pack as f16.
    let srgb_to_linear = |s: f32| -> f32 {
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut texels = Vec::with_capacity(64 * 64 * 64 * 8);
    for bi in 0..64usize {
        let (tx, ty) = (bi % 8, bi / 8);
        for gi in 0..64usize {
            for ri in 0..64usize {
                let src = ((ty * 64 + gi) * 512 + tx * 64 + ri) * 4;
                for ch in 0..3 {
                    let lin = srgb_to_linear(atlas[src + ch] as f32 / 255.0);
                    texels.extend_from_slice(&crate::f32_to_f16_bits(lin).to_le_bytes());
                }
                texels.extend_from_slice(&crate::f32_to_f16_bits(1.0).to_le_bytes());
            }
        }
    }

    let exposure = std::env::var("EFT_GRADE_EXPOSURE")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(super::DEFAULT_GRADE_EXPOSURE);
    let vignette = if std::env::var("EFT_VIGNETTE").map(|v| v.trim() == "0").unwrap_or(false) {
        0.0
    } else {
        0.488
    };
    info!(
        "grade: game grade LUT active from {} (exposure {exposure}, vignette {vignette})",
        path.display()
    );
    Some(GradeLutCpu { texels, exposure, vignette })
}

/// GradeParams uniform — byte-identical to grade.wgsl's struct (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradeParamsGpu {
    exposure: f32,
    /// EFT-style unsharp-mask strength (rides the old pad lane; 0 = off).
    sharpen: f32,
    _pad: [f32; 2],
    vig: [f32; 4],          // xy = aspect divisors, zw = smoothstep edges
    vig_strength: [f32; 4], // x = strength
}

/// Render-world GPU state, built once at RenderStartup.
#[derive(Resource)]
struct GradePipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    scene_sampler: Sampler,
    lut_sampler: Sampler,
    lut_view: TextureView,
    #[allow(dead_code)]
    lut_tex: Texture,
    params: Buffer,
}

fn init_grade_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    lut: Option<Res<GradeLutCpu>>,
    existing: Option<Res<GradePipeline>>,
    asset_server: Res<AssetServer>,
) {
    // Run-once in the Render schedule, NOT RenderStartup: RenderStartup executes BEFORE the very
    // first extract (bevy_render set_extract), so GradeLutCpu — which only reaches the render
    // world via ExtractResourcePlugin — was still absent and the pipeline was silently never
    // built (the viewer rendered un-tonemapped linear; caught by a Codex review + an exposure
    // A/B: 10x exposure change produced byte-identical output).
    if existing.is_some() {
        return;
    }
    let Some(lut) = lut else { return }; // grade disabled — node stays a no-op
    let lut_tex = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("eft_grade_lut"),
            size: Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 64,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &lut.texels,
    );
    let lut_view = lut_tex.create_view(&TextureViewDescriptor::default());
    let lut_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("eft_grade_lut_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let scene_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("eft_grade_scene_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let params = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_grade_params"),
        contents: bytemuck::bytes_of(&GradeParamsGpu {
            exposure: lut.exposure,
            sharpen: 0.0, // live value comes from update_grade_params each frame
            _pad: [0.0; 2],
            vig: [1.15, 0.95, 0.55, 1.25], // PRISM defaults (see grade.wgsl header)
            vig_strength: [lut.vignette, 0.0, 0.0, 0.0],
        }),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let layout = device.create_bind_group_layout(
        "eft_grade_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }), // scene
                sampler(SamplerBindingType::Filtering),
                texture_3d(TextureSampleType::Float { filterable: true }), // lut
                sampler(SamplerBindingType::Filtering),
                uniform_buffer_sized(false, Some(std::num::NonZeroU64::new(48).unwrap())),
            ),
        ),
    );
    let shader = asset_server.load("shaders/grade.wgsl");
    let pipeline_id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("eft_grade_pipeline".into()),
        layout: vec![layout.clone()],
        push_constant_ranges: vec![],
        // grade.wgsl ships its own fullscreen-triangle vertex stage (vs_fullscreen, Bevy's
        // y-flip convention) — no dependency on Bevy's FullscreenShader resource.
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_fullscreen".into()),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(), // post-process runs on the resolved target
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_grade".into()),
            targets: vec![Some(ColorTargetState {
                // The camera is HDR (Hdr marker in main.rs); post_process_write ping-pongs on
                // the Rgba16Float main textures. Output is LINEAR (LUT inverted at load).
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(GradePipeline {
        layout,
        pipeline_id,
        scene_sampler,
        lut_sampler,
        lut_view,
        lut_tex,
        params,
    });
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct GradeLabel;

/// Bind-group cache keyed on the post-process SOURCE view id (Bevy's own custom_post_processing
/// example pattern): post_process_write ping-pongs between two textures, so at most two entries
/// ever exist — re-creating the bind group every frame is pure churn.
#[derive(Default)]
struct GradeNode {
    cached_bg: std::sync::Mutex<Option<(bevy::render::render_resource::TextureViewId, bevy::render::render_resource::BindGroup)>>,
}

impl ViewNode for GradeNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        target: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(gp) = world.get_resource::<GradePipeline>() else {
            return Ok(()); // grade disabled
        };
        // Step aside when the grade is off OR the current pack has no grade LUT (grade_available):
        // in both cases the main world runs TonyMcMapface on the camera (apply_gfx_camera). Checking
        // grade_available too keeps the "grade OWNS the chain XOR TonyMcMapface" invariant across an
        // in-place swap to a LUT-less pack — otherwise this node keeps applying the OLD pack's LUT
        // on top of TonyMcMapface (double tonemapping).
        if let Some(s) = world.get_resource::<crate::render::GfxSettings>() {
            if !s.grade || !s.grade_available {
                return Ok(());
            }
        }
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(gp.pipeline_id) else {
            return Ok(()); // still compiling
        };
        // Skip non-HDR views (pipeline targets Rgba16Float; the main camera is always Hdr).
        if target.main_texture_format() != ViewTarget::TEXTURE_FORMAT_HDR {
            return Ok(());
        }
        let post = target.post_process_write();
        let mut cache = self.cached_bg.lock().unwrap();
        let bind = match cache.as_ref() {
            Some((id, bg)) if *id == post.source.id() => bg.clone(),
            _ => {
                let bg = render_context.render_device().create_bind_group(
                    "eft_grade_bg",
                    &gp.layout,
                    &BindGroupEntries::sequential((
                        post.source,
                        &gp.scene_sampler,
                        &gp.lut_view,
                        &gp.lut_sampler,
                        gp.params.as_entire_binding(),
                    )),
                );
                *cache = Some((post.source.id(), bg.clone()));
                bg
            }
        };
        drop(cache);
        let mut pass = render_context.command_encoder().begin_render_pass(&RenderPassDescriptor {
            label: Some("eft_grade_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Plugin: extracts the CPU LUT, uploads once, and runs the grade pass between Bloom and
/// Tonemapping (the camera uses Tonemapping::None when the grade is active, so this pass IS
/// the tonemap — see main.rs).
pub struct GradePlugin;

/// Live-tune the 48-byte GradeParams uniform from the UI settings (exposure / vignette).
/// A per-frame 48 B write is free; skipping change-detection keeps it robust across worlds.
fn update_grade_params(
    queue: Res<bevy::render::renderer::RenderQueue>,
    gp: Option<Res<GradePipeline>>,
    settings: Option<Res<crate::render::GfxSettings>>,
) {
    let (Some(gp), Some(s)) = (gp, settings) else { return };
    queue.write_buffer(
        &gp.params,
        0,
        bytemuck::bytes_of(&GradeParamsGpu {
            exposure: s.grade_exposure,
            sharpen: s.sharpen,
            _pad: [0.0; 2],
            vig: [1.15, 0.95, 0.55, 1.25],
            vig_strength: [if s.vignette { 0.488 } else { 0.0 }, 0.0, 0.0, 0.0],
        }),
    );
}

impl Plugin for GradePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<GradeLutCpu>::default());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                bevy::render::Render,
                (
                    // Run-once pipeline init (guarded internally) — MUST be in Render, not
                    // RenderStartup, because the extracted GradeLutCpu doesn't exist yet when
                    // RenderStartup runs (see init_grade_pipeline).
                    init_grade_pipeline.in_set(bevy::render::RenderSystems::PrepareResources),
                    update_grade_params
                        .in_set(bevy::render::RenderSystems::PrepareResources)
                        .after(init_grade_pipeline),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<GradeNode>>(Core3d, GradeLabel)
            .add_render_graph_edges(Core3d, (Node3d::Bloom, GradeLabel, Node3d::Tonemapping));
    }
}
