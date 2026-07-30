//! eft::ssr — screen-space reflections (Phase 6 of docs/GRAPHICS_PLAN.md; shaders/ssr.wgsl).
//!
//! ViewNode ordered SSAO -> SSR -> TAA -> Bloom, opt-in via GfxSettings.ssr (EFT_SSR=1). One
//! fullscreen pass on the HDR ping-pong: world-space march against the prepass depth, blend toward
//! the hit color by fresnel x gloss; misses keep the existing analytic reflection, so degradation
//! is exactly today's look. See the shader header for the per-pixel gates and the water rule.

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
        BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, StoreOp, TextureSampleType,
        VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::{ExtractedView, ViewTarget},
    RenderApp, RenderStartup,
};
use bytemuck::{Pod, Zeroable};

/// Byte-identical to ssr.wgsl's `SsrParams` (272 bytes: 4 mat4 + vec4).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsrParamsGpu {
    clip_from_world: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    world_from_view: [[f32; 4]; 4],
    view_from_world: [[f32; 4]; 4],
    /// x = max trace distance (m), y = reserved, z = intensity, w = viewport height px.
    p: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<SsrParamsGpu>() == 272);

#[derive(Resource)]
struct SsrPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    sampler: Sampler,
    params: Buffer,
}

fn init_ssr_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let layout = device.create_bind_group_layout(
        "eft_ssr_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer_sized(false, Some(std::num::NonZeroU64::new(272).unwrap())),
            ),
        ),
    );
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("eft_ssr_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("eft_ssr_params"),
        size: 272,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = asset_server.load("shaders/ssr.wgsl");
    let pipeline_id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("eft_ssr_pipeline".into()),
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
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_ssr".into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(SsrPipeline {
        layout,
        pipeline_id,
        sampler,
        params,
    });
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct SsrLabel;

#[derive(Default)]
struct SsrNode;

impl ViewNode for SsrNode {
    type ViewQuery = (&'static ViewTarget, &'static ExtractedView);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (target, view): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(sp), Some(settings)) = (
            world.get_resource::<SsrPipeline>(),
            world.get_resource::<crate::render::GfxSettings>(),
        ) else {
            return Ok(());
        };
        if !settings.ssr {
            return Ok(());
        }
        let Some(pre) = world.get_resource::<super::gpu_driven::EftPrepassResources>() else {
            return Ok(());
        };
        if !pre.active {
            return Ok(());
        }
        let (Some(depth_view), Some(normal_view)) = (&pre.depth_view, &pre.normal_view) else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(sp.pipeline_id) else {
            return Ok(());
        };
        if target.main_texture_format() != ViewTarget::TEXTURE_FORMAT_HDR {
            return Ok(());
        }
        let world_from_view = view.world_from_view.to_matrix();
        let params = SsrParamsGpu {
            clip_from_world: pre.clip_from_world,
            inv_proj: view.clip_from_view.inverse().to_cols_array_2d(),
            world_from_view: world_from_view.to_cols_array_2d(),
            view_from_world: world_from_view.inverse().to_cols_array_2d(),
            p: [60.0, 0.0, 1.0, view.viewport.w as f32],
        };
        world
            .resource::<RenderQueue>()
            .write_buffer(&sp.params, 0, bytemuck::bytes_of(&params));
        let post = target.post_process_write();
        let bg = render_context.render_device().create_bind_group(
            "eft_ssr_bg",
            &sp.layout,
            &BindGroupEntries::sequential((
                post.source,
                &sp.sampler,
                depth_view,
                normal_view,
                sp.params.as_entire_binding(),
            )),
        );
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("eft_ssr_pass"),
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
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

/// SSR between SSAO and TAA (SSAO -> SSR -> TAA -> Bloom), per the plan's composition order:
/// TAA must smooth the trace, not precede it.
pub struct SsrPlugin;

impl Plugin for SsrPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_ssr_pipeline)
            .add_render_graph_node::<ViewNodeRunner<SsrNode>>(Core3d, SsrLabel)
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, SsrLabel, Node3d::Bloom))
            .add_render_graph_edges(
                Core3d,
                (super::ssao::SsaoLabel, SsrLabel, super::taa::TaaLabel),
            );
    }
}
