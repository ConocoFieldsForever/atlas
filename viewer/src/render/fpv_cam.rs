//! eft::render::fpv_cam — analog FPV video-link post pass (shaders/fpv_cam.wgsl).
//!
//! Drone mode looks through a simulated 5.8 GHz analog VTX: ever-present grain, scanline tear
//! bursts, chroma fringing, and full snow breakup as the link dies. The LINK QUALITY is a real
//! RF model computed on the CPU once per frame: free-space range falloff from the PILOT position
//! (the rig's spawn point — where you "stood" when you armed; the agent session's reset pose) plus
//! per-obstruction attenuation from `GroundData::segment_crossings` (every wall/floor/ceiling
//! triangle crossed on the pilot→drone line eats ~2.5 dB-ish). The GPU pass is uniform-driven and
//! costs one fullscreen triangle; it no-ops entirely outside drone mode.
//!
//! Graph position: after Tonemapping (the grade LUT or TonyMcMapface has produced display-referred
//! values) and before Upscaling — analog noise rides the video signal, not the scene light.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_graph::{NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner},
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer_sized},
        BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer, BufferInitDescriptor,
        BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, StoreOp, TextureSampleType,
        VertexState,
    },
    renderer::{RenderContext, RenderDevice},
    view::ViewTarget,
    RenderApp,
};
use bytemuck::{Pod, Zeroable};

/// Main-world state, extracted to the render world each frame. `update_fpv_fx` (main world)
/// owns the RF model; the render side just uploads these four floats.
#[derive(Resource, Clone, ExtractResource)]
pub struct FpvCamFx {
    /// Pass active this frame (drone mode + user toggle).
    pub active: bool,
    /// Link quality 0..1 (1 = full RSSI). Smoothed — real RSSI meters don't snap.
    pub signal: f32,
    /// Effect master gain (camera-tab slider).
    pub intensity: f32,
    /// Accumulated effect time (s).
    pub time: f32,
    /// Window aspect for noise texel shaping.
    pub aspect: f32,
}

impl Default for FpvCamFx {
    fn default() -> Self {
        Self { active: false, signal: 1.0, intensity: 0.8, time: 0.0, aspect: 16.0 / 9.0 }
    }
}

/// Per-frame RF + housekeeping (main world). Pilot position = the manual rig's spawn point, or
/// the agent drone's reset home when a session is live and spectated.
pub fn update_fpv_fx(
    time: Res<Time>,
    settings: Res<crate::CameraSettings>,
    grid: Option<Res<crate::walk_ground::GroundGrid>>,
    rig: Query<&crate::drone::DroneRig, With<crate::render::CullCamera>>,
    shared: Option<Res<crate::agent_link::AgentShared>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut fx: ResMut<FpvCamFx>,
) {
    fx.time = (fx.time + time.delta_secs()) % 3600.0;
    fx.intensity = settings.fpv_noise;
    if let Ok(w) = windows.single() {
        fx.aspect = (w.width() / w.height().max(1.0)).max(0.1);
    }
    fx.active = settings.mode == crate::CamMode::Drone && settings.fpv_noise > 0.001;
    if !fx.active {
        fx.signal = 1.0;
        return;
    }
    // Whose airframe? An active agent session (its home = reset spawn), else the manual rig.
    let mut pose: Option<(Vec3, Vec3)> = None; // (pilot/home, drone)
    if let Some(sh) = &shared {
        let w = sh.0.lock().unwrap();
        if w.active {
            pose = Some((w.home, w.drone.pos));
        }
    }
    if pose.is_none() {
        if let Ok(r) = rig.single() {
            if r.live {
                pose = Some((r.spawn_pos, r.state.pos));
            }
        }
    }
    let Some((home, pos)) = pose else {
        fx.signal = 1.0;
        return;
    };
    // Free-space falloff + per-obstruction attenuation (walls eat analog video fast).
    let d = home.distance(pos);
    let range = settings.fpv_range.max(10.0);
    let mut s = (1.0 - (d / range).powf(1.6)).clamp(0.0, 1.0);
    if let Some(g) = &grid {
        let crossings = g.segment_crossings(home, pos, 8);
        s *= 0.72f32.powi(crossings as i32);
    }
    // Smooth like a real RSSI needle (fast drop, slower recover).
    let tau = if s < fx.signal { 0.12 } else { 0.5 };
    let a = 1.0 - (-time.delta_secs() / tau).exp();
    fx.signal += (s - fx.signal) * a;
}

/// FpvParams uniform — byte-identical to fpv_cam.wgsl (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FpvParamsGpu {
    time: f32,
    signal: f32,
    intensity: f32,
    aspect: f32,
    enabled: f32,
    _pad: [f32; 3],
}

#[derive(Resource)]
struct FpvPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    scene_sampler: Sampler,
    params: Buffer,
}

fn init_fpv_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    existing: Option<Res<FpvPipeline>>,
    asset_server: Res<AssetServer>,
) {
    // Run-once in Render (same guard pattern as grade.rs).
    if existing.is_some() {
        return;
    }
    let scene_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("fpv_cam_scene_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let params = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("fpv_cam_params"),
        contents: bytemuck::bytes_of(&FpvParamsGpu {
            time: 0.0,
            signal: 1.0,
            intensity: 0.0,
            aspect: 16.0 / 9.0,
            enabled: 0.0,
            _pad: [0.0; 3],
        }),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    let layout = device.create_bind_group_layout(
        "fpv_cam_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer_sized(false, Some(std::num::NonZeroU64::new(32).unwrap())),
            ),
        ),
    );
    let shader = asset_server.load("shaders/fpv_cam.wgsl");
    let pipeline_id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("fpv_cam_pipeline".into()),
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
            entry_point: Some("fs_fpv".into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(FpvPipeline { layout, pipeline_id, scene_sampler, params });
}

/// Upload the 32-byte uniform every frame from the extracted main-world state.
fn update_fpv_params(
    queue: Res<bevy::render::renderer::RenderQueue>,
    gp: Option<Res<FpvPipeline>>,
    fx: Option<Res<FpvCamFx>>,
) {
    let (Some(gp), Some(fx)) = (gp, fx) else { return };
    queue.write_buffer(
        &gp.params,
        0,
        bytemuck::bytes_of(&FpvParamsGpu {
            time: fx.time,
            signal: fx.signal,
            intensity: fx.intensity,
            aspect: fx.aspect,
            enabled: if fx.active { 1.0 } else { 0.0 },
            _pad: [0.0; 3],
        }),
    );
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct FpvLabel;

/// Bind-group cached on the ping-pong source id (same pattern as GradeNode).
#[derive(Default)]
struct FpvNode {
    cached_bg: std::sync::Mutex<
        Option<(bevy::render::render_resource::TextureViewId, bevy::render::render_resource::BindGroup)>,
    >,
}

impl ViewNode for FpvNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        target: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(gp) = world.get_resource::<FpvPipeline>() else {
            return Ok(());
        };
        // Fully step aside unless drone mode wants the effect this frame — zero cost otherwise.
        match world.get_resource::<FpvCamFx>() {
            Some(fx) if fx.active => {}
            _ => return Ok(()),
        }
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(gp.pipeline_id) else {
            return Ok(());
        };
        if target.main_texture_format() != ViewTarget::TEXTURE_FORMAT_HDR {
            return Ok(());
        }
        let post = target.post_process_write();
        let mut cached = self.cached_bg.lock().unwrap();
        let bind = match cached.as_ref() {
            Some((id, bg)) if *id == post.source.id() => bg.clone(),
            _ => {
                let bg = render_context.render_device().create_bind_group(
                    "fpv_cam_bg",
                    &gp.layout,
                    &BindGroupEntries::sequential((
                        post.source,
                        &gp.scene_sampler,
                        gp.params.as_entire_binding(),
                    )),
                );
                *cached = Some((post.source.id(), bg.clone()));
                bg
            }
        };
        drop(cached);
        let mut pass = render_context.command_encoder().begin_render_pass(&RenderPassDescriptor {
            label: Some("fpv_cam_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations { load: LoadOp::Clear(Default::default()), store: StoreOp::Store },
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

pub struct FpvCamPlugin;

impl Plugin for FpvCamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FpvCamFx>()
            .add_plugins(ExtractResourcePlugin::<FpvCamFx>::default())
            .add_systems(Update, update_fpv_fx);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                bevy::render::Render,
                (
                    init_fpv_pipeline.in_set(bevy::render::RenderSystems::PrepareResources),
                    update_fpv_params
                        .in_set(bevy::render::RenderSystems::PrepareResources)
                        .after(init_fpv_pipeline),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<FpvNode>>(Core3d, FpvLabel)
            // After the tonemap (grade LUT or TonyMcMapface → display-referred), before the
            // swapchain blit: analog noise rides the video signal, not the scene light.
            .add_render_graph_edges(Core3d, (Node3d::Tonemapping, FpvLabel, Node3d::Upscaling));
    }
}
