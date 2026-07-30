//! eft::taa — temporal anti-aliasing (Phase 2 of docs/GRAPHICS_PLAN.md; shaders/taa.wgsl).
//!
//! A ViewNode between the SSAO composite and Bloom, same family as render::ssao. Per frame it
//! resolves current color against a reprojected, variance-clipped history and writes BOTH the
//! post-process destination and its own history for next frame.
//!
//! Motion model is CAMERA-ONLY reprojection from the prepass depth + the prepass resource's
//! previous clip_from_world; everything without valid prepass data (sky, blend, grass) and every
//! reactive class (water) blends toward current instead of trusting history — the plan's explicit
//! answer to "the world is not static". History lives HERE (two ping-pong Rgba16Float textures),
//! and is invalidated whenever the prepass invalidates its matrices (resize/map swap/toggle) —
//! `has_history` rides the params uniform so the shader passes through on invalid frames.
//!
//! The resolve renders INTO the history-write texture, then a trivial blit copies it to the
//! post-process destination: the history must outlive the frame, and the ViewTarget's internal
//! texture usage flags are not ours to rely on.
//!
//! V1 runs WITHOUT projection jitter, i.e. the plan's "4x MSAA + TAA" A/B arm: MSAA keeps owning
//! geometric edges, TAA adds shading-domain convergence (specular glint, splat blend, FXAA-residual
//! shimmer). The 1x + jitter arm (TemporalJitter exists in the vendored bevy_render) is the
//! Phase-2 decision experiment and must be measured, not assumed — see GRAPHICS_PLAN.md Phase 2.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::{
    render_graph::{
        NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
    },
    render_resource::{
        binding_types::{sampler, texture_2d, texture_depth_2d, uniform_buffer_sized},
        BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer, BufferDescriptor,
        BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FilterMode,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, StoreOp, Texture, TextureDescriptor,
        TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
        TextureViewDescriptor, VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::{ExtractedView, ViewTarget},
    RenderApp, RenderStartup,
};
use bytemuck::{Pod, Zeroable};

/// Byte-identical to taa.wgsl's `TaaParams` (208 bytes: 3 mat4 + vec4).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaParamsGpu {
    prev_clip_from_world: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    world_from_view: [[f32; 4]; 4],
    /// x = static blend alpha, y = has_history, z,w = viewport px.
    p: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<TaaParamsGpu>() == 208);

#[derive(Resource)]
struct TaaPipeline {
    layout: BindGroupLayout,
    blit_layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    blit_pipeline_id: CachedRenderPipelineId,
    sampler: Sampler,
    params: Buffer,
}

/// The ping-pong history. `write_idx` flips AFTER a successful resolve; `valid` mirrors whether
/// last frame's resolve actually happened (any skip — disabled, no prepass, resize — clears it).
#[derive(Resource, Default)]
struct TaaHistory {
    textures: Vec<Texture>,
    views: Vec<TextureView>,
    write_idx: usize,
    valid: bool,
    size: UVec2,
}

fn init_taa_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let layout = device.create_bind_group_layout(
        "eft_taa_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }), // scene
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }), // history read
                texture_depth_2d(),                                        // prepass depth
                texture_2d(TextureSampleType::Float { filterable: false }), // prepass normal+class
                uniform_buffer_sized(false, Some(std::num::NonZeroU64::new(208).unwrap())),
            ),
        ),
    );
    let blit_layout = device.create_bind_group_layout(
        "eft_taa_blit_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("eft_taa_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("eft_taa_params"),
        size: 208,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = asset_server.load("shaders/taa.wgsl");
    let pipeline_id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("eft_taa_pipeline".into()),
        layout: vec![layout.clone()],
        push_constant_ranges: vec![],
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_fullscreen".into()),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_taa".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float, // renders into the HISTORY texture
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    let blit_pipeline_id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("eft_taa_blit_pipeline".into()),
        layout: vec![blit_layout.clone()],
        push_constant_ranges: vec![],
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_fullscreen".into()),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_blit".into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(TaaPipeline {
        layout,
        blit_layout,
        pipeline_id,
        blit_pipeline_id,
        sampler,
        params,
    });
    commands.insert_resource(TaaHistory::default());
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct TaaLabel;

#[derive(Default)]
struct TaaNode;

impl ViewNode for TaaNode {
    type ViewQuery = (&'static ViewTarget, &'static ExtractedView);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (target, view): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(tp), Some(settings)) = (
            world.get_resource::<TaaPipeline>(),
            world.get_resource::<crate::render::GfxSettings>(),
        ) else {
            return Ok(());
        };
        // History validity is managed through a cell on the resource; any early-out below must
        // clear it so a stale history is never blended after a gap frame.
        let invalidate = || {
            if let Some(mut h) = unsafe { world.as_unsafe_world_cell_readonly().get_resource_mut::<TaaHistory>() } {
                h.valid = false;
            }
        };
        if !settings.taa {
            invalidate();
            return Ok(());
        }
        let Some(pre) = world.get_resource::<super::gpu_driven::EftPrepassResources>() else {
            invalidate();
            return Ok(());
        };
        if !pre.active {
            invalidate();
            return Ok(());
        }
        let (Some(depth_view), Some(normal_view)) = (&pre.depth_view, &pre.normal_view) else {
            invalidate();
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let (Some(pipeline), Some(blit)) = (
            cache.get_render_pipeline(tp.pipeline_id),
            cache.get_render_pipeline(tp.blit_pipeline_id),
        ) else {
            invalidate();
            return Ok(());
        };
        if target.main_texture_format() != ViewTarget::TEXTURE_FORMAT_HDR {
            return Ok(());
        }
        let vp = view.viewport;
        let size = UVec2::new(vp.z, vp.w);
        if size.x == 0 || size.y == 0 {
            invalidate();
            return Ok(());
        }

        // History (re)allocation + the post-view state. SAFETY of the unsafe cell: this node is
        // the only writer of TaaHistory, and the render graph runs nodes serially per view.
        let Some(mut hist) = (unsafe {
            world
                .as_unsafe_world_cell_readonly()
                .get_resource_mut::<TaaHistory>()
        }) else {
            return Ok(());
        };
        if hist.size != size || hist.textures.len() != 2 {
            let mk = || {
                render_context
                    .render_device()
                    .create_texture(&TextureDescriptor {
                        label: Some("eft_taa_history"),
                        size: Extent3d {
                            width: size.x,
                            height: size.y,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::Rgba16Float,
                        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
            };
            let t0 = mk();
            let t1 = mk();
            hist.views = vec![
                t0.create_view(&TextureViewDescriptor::default()),
                t1.create_view(&TextureViewDescriptor::default()),
            ];
            hist.textures = vec![t0, t1];
            hist.write_idx = 0;
            hist.valid = false;
            hist.size = size;
            info!("taa: history {}x{} Rgba16Float x2", size.x, size.y);
        }
        let has_history = hist.valid && pre.prev_clip_from_world.is_some();

        // Params.
        let world_from_view = view.world_from_view.to_matrix();
        let params = TaaParamsGpu {
            prev_clip_from_world: pre
                .prev_clip_from_world
                .unwrap_or(pre.clip_from_world),
            inv_proj: view.clip_from_view.inverse().to_cols_array_2d(),
            world_from_view: world_from_view.to_cols_array_2d(),
            p: [
                0.1, // static convergence alpha: ~10 frames to steady state
                if has_history { 1.0 } else { 0.0 },
                size.x as f32,
                size.y as f32,
            ],
        };
        world
            .resource::<RenderQueue>()
            .write_buffer(&tp.params, 0, bytemuck::bytes_of(&params));

        let post = target.post_process_write();
        let read_idx = 1 - hist.write_idx;
        let resolve_bg = render_context.render_device().create_bind_group(
            "eft_taa_bg",
            &tp.layout,
            &BindGroupEntries::sequential((
                post.source,
                &tp.sampler,
                &hist.views[read_idx],
                depth_view,
                normal_view,
                tp.params.as_entire_binding(),
            )),
        );
        let blit_bg = render_context.render_device().create_bind_group(
            "eft_taa_blit_bg",
            &tp.blit_layout,
            &BindGroupEntries::sequential((&hist.views[hist.write_idx], &tp.sampler)),
        );

        // Pass 1: resolve into history[write].
        {
            let mut pass = render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("eft_taa_resolve"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &hist.views[hist.write_idx],
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
            pass.set_bind_group(0, &resolve_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        // Pass 2: blit history[write] -> post destination.
        {
            let mut pass = render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("eft_taa_blit"),
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
            pass.set_pipeline(blit);
            pass.set_bind_group(0, &blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        hist.write_idx = read_idx;
        hist.valid = true;
        Ok(())
    }
}

/// TAA between SSAO and Bloom: (EndMainPass, SsaoLabel?, TaaLabel, Bloom) — SSAO's node label is
/// private to ssao.rs, so we order against the stable Core3d anchors on both sides and rely on
/// SSAO's own (EndMainPass, Ssao, Bloom) edges for the relative order (graph edges are a partial
/// order; both constraints hold simultaneously).
pub struct TaaPlugin;

impl Plugin for TaaPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_taa_pipeline)
            .add_render_graph_node::<ViewNodeRunner<TaaNode>>(Core3d, TaaLabel)
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, TaaLabel, Node3d::Bloom))
            // The plan places TAA AFTER the SSAO composite; without this edge the two would only
            // share the (EndMainPass .. Bloom) window and their relative order would be unspecified.
            .add_render_graph_edges(Core3d, (super::ssao::SsaoLabel, TaaLabel));
    }
}
