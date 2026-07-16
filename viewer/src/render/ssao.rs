//! eft::ssao — depth-only SSAO post pass (Graphics (experimental) toggle; shaders/ssao.wgsl).
//!
//! The custom GPU-driven path writes no normal/motion prepass, so Bevy's built-in SSAO/TAA can't
//! see our geometry. This is a self-contained ViewNode (same pattern as render::grade): it binds
//! the resolved scene color + the MULTISAMPLED reverse-z depth (sample 0), reconstructs view-space
//! positions/normals, and multiplies a spiral-tap occlusion term onto the color. It runs between
//! the main pass and Bloom so the grade LUT sees the occluded color. Off by default (GfxSettings.
//! ssao / EFT_SSAO=1); radius/intensity are live UI sliders — the tiny uniform is rewritten in
//! the node each frame from the extracted settings.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::{
    render_graph::{
        NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
    },
    render_resource::{
        binding_types::{sampler, texture_2d, texture_depth_2d_multisampled, uniform_buffer_sized},
        BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer, BufferDescriptor,
        BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, StoreOp, TextureSampleType,
        VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::{ExtractedView, ViewDepthTexture, ViewTarget},
    RenderApp, RenderStartup,
};
use bytemuck::{Pod, Zeroable};

/// Byte-identical to ssao.wgsl's `SsaoParams` (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsaoParamsGpu {
    inv_proj: [[f32; 4]; 4],
    /// x = world radius (m), y = intensity, z = power, w = fade-end view distance (m).
    p: [f32; 4],
    /// x,y = viewport px, z = proj11, w = pad.
    vp: [f32; 4],
}

#[derive(Resource)]
struct SsaoPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    scene_sampler: Sampler,
    params: Buffer,
}

fn init_ssao_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let layout = device.create_bind_group_layout(
        "eft_ssao_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }), // scene color
                sampler(SamplerBindingType::Filtering),
                texture_depth_2d_multisampled(), // main-pass depth (MSAA; sample 0 read)
                uniform_buffer_sized(false, Some(std::num::NonZeroU64::new(96).unwrap())),
            ),
        ),
    );
    let scene_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("eft_ssao_scene_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("eft_ssao_params"),
        size: 96,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = asset_server.load("shaders/ssao.wgsl");
    let pipeline_id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("eft_ssao_pipeline".into()),
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
        multisample: MultisampleState::default(), // post-process on the resolved color target
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_ssao".into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(SsaoPipeline {
        layout,
        pipeline_id,
        scene_sampler,
        params,
    });
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct SsaoLabel;

/// Bind-group cache keyed on (source view id, depth view id) — same pattern as render::grade
/// (the depth view is also swapped when the window resizes).
#[derive(Default)]
struct SsaoNode {
    cached_bg: std::sync::Mutex<
        Option<(
            bevy::render::render_resource::TextureViewId,
            bevy::render::render_resource::TextureViewId,
            bevy::render::render_resource::BindGroup,
        )>,
    >,
}

impl ViewNode for SsaoNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ExtractedView,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (target, depth, view): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(sp) = world.get_resource::<SsaoPipeline>() else {
            return Ok(());
        };
        let Some(settings) = world.get_resource::<crate::render::GfxSettings>() else {
            return Ok(());
        };
        if !settings.ssao {
            return Ok(());
        }
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(sp.pipeline_id) else {
            return Ok(());
        };
        if target.main_texture_format() != ViewTarget::TEXTURE_FORMAT_HDR {
            return Ok(());
        }
        // The pipeline binds texture_depth_2d_multisampled — a 1x-MSAA view would fail bind-group
        // validation (latent: this app always runs MSAA 4x; guard anyway per the Codex review).
        if depth.texture.sample_count() <= 1 {
            return Ok(());
        }
        // Live params from the UI (96 B write per frame while enabled — negligible).
        let vp = view.viewport;
        let params = SsaoParamsGpu {
            inv_proj: view.clip_from_view.inverse().to_cols_array_2d(),
            p: [settings.ssao_radius, settings.ssao_intensity, 1.5, 80.0],
            vp: [
                vp.z as f32,
                vp.w as f32,
                view.clip_from_view.y_axis.y,
                0.0,
            ],
        };
        world
            .resource::<RenderQueue>()
            .write_buffer(&sp.params, 0, bytemuck::bytes_of(&params));

        let post = target.post_process_write();
        let mut cache = self.cached_bg.lock().unwrap();
        let bind = match cache.as_ref() {
            Some((sid, did, bg)) if *sid == post.source.id() && *did == depth.view().id() => {
                bg.clone()
            }
            _ => {
                let bg = render_context.render_device().create_bind_group(
                    "eft_ssao_bg",
                    &sp.layout,
                    &BindGroupEntries::sequential((
                        post.source,
                        &sp.scene_sampler,
                        depth.view(),
                        sp.params.as_entire_binding(),
                    )),
                );
                *cache = Some((post.source.id(), depth.view().id(), bg.clone()));
                bg
            }
        };
        drop(cache);
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("eft_ssao_pass"),
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

/// SSAO between the main pass and Bloom (the grade LUT then tonemaps the occluded color).
pub struct SsaoPlugin;

impl Plugin for SsaoPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_ssao_pipeline)
            .add_render_graph_node::<ViewNodeRunner<SsaoNode>>(Core3d, SsaoLabel)
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, SsaoLabel, Node3d::Bloom));
    }
}
