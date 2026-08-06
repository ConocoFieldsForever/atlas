//! eft::ssao — SSAO as a PRE-MAIN AO LANE (Graphics (experimental) toggle; shaders/ssao.wgsl).
//!
//! Ordered prepass -> SSAO -> main pass: reconstructs occlusion from the PREPASS depth + normals
//! and writes an R8 factor the main pass samples during OPAQUE shading (group(1) binding(3) in
//! gpu_draw.wgsl). This replaced the old post-multiply over the finished frame, which darkened
//! glass panes by the occlusion of the interior BEHIND them (glass never writes the prepass, so
//! its pixels carried the background's AO). As a lane, BLEND surfaces read ao = 1 by material
//! flag and the term scales only ambient + lamp diffuse — sun and emissive stay untouched.
//!
//! The target is (re)created to the view size by `prepare_ao_target` and initialized WHITE, so a
//! frame where the node skips (toggle off, prepass idle) shades exactly as ao = 1. gpu_driven's
//! `sync_draw_bg_ao` swaps the draw bind group between this target and the 1x1 white fallback.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::{
    render_graph::{
        NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
    },
    render_resource::{
        binding_types::{texture_2d, texture_depth_2d, uniform_buffer_sized},
        BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer, BufferDescriptor,
        BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
        StoreOp, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
        TextureUsages, TextureView, TextureViewDescriptor, VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::ExtractedView,
    RenderApp, RenderStartup,
};
use bytemuck::{Pod, Zeroable};

/// Byte-identical to ssao.wgsl's `SsaoParams` (160 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsaoParamsGpu {
    inv_proj: [[f32; 4]; 4],
    /// world -> view (the prepass stores WORLD normals; this shader works in view space).
    view_from_world: [[f32; 4]; 4],
    /// x = world radius (m), y = intensity, z = power, w = fade-end view distance (m).
    p: [f32; 4],
    /// x,y = viewport px, z = proj11, w = reserved.
    vp: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<SsaoParamsGpu>() == 160);

#[derive(Resource)]
pub(crate) struct SsaoPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    params: Buffer,
    /// 1x1 WHITE R8 texture the draw pass binds while SSAO is off / has no target yet: every
    /// opaque fragment then shades with ao = 1, byte-identical to no-SSAO.
    pub(crate) fallback_ao_view: TextureView,
}

/// The viewport-sized AO lane. `view` is `None` until the first `prepare_ao_target` run.
#[derive(Resource, Default)]
pub(crate) struct EftAoTarget {
    pub(crate) view: Option<TextureView>,
    size: (u32, u32),
}

fn init_ssao_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let layout = device.create_bind_group_layout(
        "eft_ssao_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_depth_2d(), // prepass depth (1x, reverse-z)
                // Prepass world-normal target (Rgba16Float, textureLoad only — no sampler).
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer_sized(false, Some(std::num::NonZeroU64::new(160).unwrap())),
            ),
        ),
    );
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("eft_ssao_params"),
        size: 160,
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
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_ssao".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::R8Unorm,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    let fallback = device.create_texture(&TextureDescriptor {
        label: Some("eft_ssao_fallback_white"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        fallback.as_image_copy(),
        &[255u8],
        bevy::render::render_resource::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: None, // single row: alignment rules don't apply
            rows_per_image: None,
        },
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let fallback_ao_view = fallback.create_view(&TextureViewDescriptor::default());
    commands.insert_resource(SsaoPipeline {
        layout,
        pipeline_id,
        params,
        fallback_ao_view,
    });
    commands.insert_resource(EftAoTarget::default());
}

/// (Re)create the AO lane at the view size, initialized WHITE so frames where the SSAO node
/// skips (toggle off, prepass idle, pipeline compiling) shade exactly as ao = 1.
pub(crate) fn prepare_ao_target(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<&ExtractedView>,
    target: Option<ResMut<EftAoTarget>>,
) {
    let Some(mut target) = target else { return };
    let Some(view) = views.iter().next() else {
        return;
    };
    let (w, h) = (view.viewport.z.max(1), view.viewport.w.max(1));
    if target.size == (w, h) && target.view.is_some() {
        return;
    }
    let tex = device.create_texture(&TextureDescriptor {
        label: Some("eft_ssao_ao_lane"),
        size: Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // wgpu zero-inits textures; zero would mean "fully occluded" and black out ambient on any
    // frame the node skips. Stamp WHITE once at creation.
    let row = ((w + 255) / 256) * 256; // 256-byte row alignment for write_texture
    queue.write_texture(
        tex.as_image_copy(),
        &vec![255u8; (row * h) as usize],
        bevy::render::render_resource::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: None,
        },
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    target.view = Some(tex.create_view(&TextureViewDescriptor::default()));
    target.size = (w, h);
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct SsaoLabel;

/// Bind-group cache keyed on (prepass depth id, prepass normal id).
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
    type ViewQuery = &'static ExtractedView;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(sp), Some(settings), Some(target)) = (
            world.get_resource::<SsaoPipeline>(),
            world.get_resource::<crate::render::GfxSettings>(),
            world.get_resource::<EftAoTarget>(),
        ) else {
            return Ok(());
        };
        if !settings.ssao {
            return Ok(());
        }
        let Some(ao_view) = target.view.as_ref() else {
            return Ok(());
        };
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
        // Live params from the UI (160 B write per frame while enabled — negligible).
        let vp = view.viewport;
        let params = SsaoParamsGpu {
            inv_proj: view.clip_from_view.inverse().to_cols_array_2d(),
            view_from_world: view.world_from_view.to_matrix().inverse().to_cols_array_2d(),
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

        let mut cached = self.cached_bg.lock().unwrap();
        let bind = match cached.as_ref() {
            Some((did, nid, bg)) if *did == depth_view.id() && *nid == normal_view.id() => {
                bg.clone()
            }
            _ => {
                let bg = render_context.render_device().create_bind_group(
                    "eft_ssao_bg",
                    &sp.layout,
                    &BindGroupEntries::sequential((
                        depth_view,
                        normal_view,
                        sp.params.as_entire_binding(),
                    )),
                );
                *cached = Some((depth_view.id(), normal_view.id(), bg.clone()));
                bg
            }
        };
        drop(cached);
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("eft_ssao_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: ao_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(bevy::color::LinearRgba::WHITE.into()),
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

/// SSAO between the prepass and the main pass: the AO lane must be written before the opaque
/// shading samples it. `sync_draw_bg_ao` (gpu_driven) swaps the sampled view per the toggle.
pub struct SsaoPlugin {
    /// Whether the GPU-driven path is the one being installed, i.e. whether `EftPrepassLabel`
    /// will exist in the render graph. It is passed in rather than read from the world because
    /// `RenderPath` is not inserted as a resource until AFTER this plugin is added
    /// (`main.rs`), and it cannot be inferred at build() time by any other means.
    pub gpu_driven: bool,
}

impl Plugin for SsaoPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // NOTE: `prepare_ao_target` + gpu_driven's `sync_draw_bg_ao` are registered by
        // EftGpuDrivenPlugin (they order against its private `prepare_gpu_buffers`).
        // The prepass edge is CONDITIONAL, because `EftPrepassLabel` exists on exactly one render
        // path and `add_render_graph_edges` panics on a label that is not in the graph. This
        // plugin is installed on every path, so naming it unconditionally killed the process at
        // startup with "node EftPrepassLabel does not exist" -- on BOTH fallbacks, which is the
        // entire population they exist for: they are what an under-featured GPU gets, and the
        // LLPC probe routes AMD users straight onto Standard (issue #9, three reporters). It
        // reproduces on any GPU with EFT_RENDER=std or =m0.
        //
        // It stays HERE rather than moving to the plugin that owns the node, which was the first
        // thing tried: `EftGpuDrivenPlugin` builds well before this one, so at its build() the
        // SsaoLabel node does not exist yet and the panic simply changes direction. An edge has
        // to be declared where BOTH endpoints already exist, which is the later of the two.
        //
        // SsaoNode itself needs no guard: it early-returns without `EftPrepassResources`, which
        // only the GPU-driven path inserts, so on the other paths it is an inert no-op. The node
        // is still registered on every path because `taa.rs` orders against `SsaoLabel` and would
        // panic in turn if it vanished.
        render_app
            .add_systems(RenderStartup, init_ssao_pipeline)
            .add_render_graph_node::<ViewNodeRunner<SsaoNode>>(Core3d, SsaoLabel)
            .add_render_graph_edges(Core3d, (SsaoLabel, Node3d::StartMainPass));
        if self.gpu_driven {
            render_app
                .add_render_graph_edges(Core3d, (super::gpu_driven::EftPrepassLabel, SsaoLabel));
        }
    }
}
