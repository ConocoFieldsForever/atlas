//! M0 custom instanced render path — the WORKING first-pixel draw.
//!
//! This is the low-level custom-instancing pattern from Bevy 0.17's
//! `examples/shader_advanced/custom_shader_instancing.rs`, adapted to the
//! .eftpack instance stream. For every unique eftpack mesh we spawn ONE entity
//! carrying a Bevy `Mesh3d` (built once from meshes.bin) plus an
//! `InstanceMaterialData` holding that mesh's per-instance FULL 3x4 affine. A
//! custom pipeline draws all instances of a mesh in a single instanced draw.
//!
//! THE #1 RULE (tarkov-unity-extraction skill): the vertex shader applies the
//! FULL ROW-MAJOR 3x4 (incl shear) to RAW verts and transforms normals by the
//! COFACTOR matrix (det·inverse-transpose) — NEVER TRS-decompose. Mirrors
//! (det<0) stay correct with ZERO baking because (a) we draw double-sided
//! (`cull_mode = None`) and (b) the cofactor matrix flips the normal sign for
//! det<0 automatically. No per-instance front-face pipeline is needed for M0.
//!
//! M0 SCOPE: flat sun+ambient lambert, all submeshes drawn with one shader, no
//! textures/materials yet (bindless materials, SH-GI, vert-paint, grade LUT are
//! the M2–M4 upgrades in `assets/shaders/instanced.wgsl` + friends). This gets
//! honest geometry on screen. The GPU-driven compute-cull + indirect-multidraw
//! design lives in `gpu_driven.rs` (M1).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, VertexBufferLayout};
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup,
    SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup,
};
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::{
    camera::visibility::NoFrustumCulling,
    core_pipeline::core_3d::Transparent3d,
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::ExtractedView,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};

use crate::eftpack::Pack;

/// The wired M0 shader (single source of truth — the advanced per-effect shaders
/// in this dir are the documented M2–M4 targets, not yet wired).
const SHADER_ASSET_PATH: &str = "shaders/instancing_m0.wgsl";

/// Bevy resource wrapping the loaded pack (inserted by `main` before app build).
/// Holds the pack behind an `Arc` so the off-thread `build_cpu_data` task can share it
/// (cheap clone) without copying the ~hundreds-of-MiB `meshes.bin` blob.
#[derive(Resource)]
pub struct LoadedPack(pub std::sync::Arc<Pack>);

/// Plugin: builds the pack's meshes into entities and installs the custom
/// instanced draw in the render app.
pub struct EftInstancingPlugin;

impl Plugin for EftInstancingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<InstanceMaterialData>::default())
            .add_systems(Startup, spawn_pack_entities);

        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems(RenderStartup, init_custom_pipeline)
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSystems::QueueMeshes),
                    prepare_instance_buffers.in_set(RenderSystems::PrepareResources),
                ),
            );
    }
}

// ---------------------------------------------------------------------------
// Per-instance GPU record fed as an instance-rate vertex buffer.
// 3 affine rows (row-major world 3x4) + an id/flags uvec4. 64 bytes.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct InstanceData {
    m0: [f32; 4], // affine row 0 = (m00 m01 m02 m03)
    m1: [f32; 4], // affine row 1
    m2: [f32; 4], // affine row 2
    ids: [u32; 4], // x=flags y=materialId(reserved) z,w=reserved
}

#[derive(Component, Deref)]
struct InstanceMaterialData(Vec<InstanceData>);

impl ExtractComponent for InstanceMaterialData {
    type QueryData = &'static InstanceMaterialData;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self> {
        Some(InstanceMaterialData(item.0.clone()))
    }
}

// ---------------------------------------------------------------------------
// Main-world startup: build one Bevy Mesh + one instanced entity per pack mesh.
// ---------------------------------------------------------------------------
fn spawn_pack_entities(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    pack: Option<Res<LoadedPack>>,
) {
    let Some(pack) = pack else {
        return;
    };
    let pack = &pack.0;
    let by_mesh = pack.instances_by_mesh();

    let mut spawned = 0usize;
    let mut total_inst = 0usize;
    for (mi, m) in pack.manifest.meshes.iter().enumerate() {
        let inst_ids = &by_mesh[mi];
        if inst_ids.is_empty() {
            continue; // orphan/degenerate-only mesh: nothing references it
        }
        let geom = match pack.mesh_geom(m) {
            Ok(g) => g,
            Err(e) => {
                warn!("mesh {} '{}' skipped: {:#}", m.id, m.name, e);
                continue;
            }
        };
        if geom.positions.is_empty() || geom.indices.is_empty() {
            continue;
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, geom.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, geom.normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, geom.uvs);
        mesh.insert_indices(Indices::U32(geom.indices));
        let handle = meshes.add(mesh);

        let data: Vec<InstanceData> = inst_ids
            .iter()
            // All-LOD pack: the M0 fallback draws the default shell only (else all shells stack).
            .filter(|&&i| pack.is_default_lod(i as usize))
            .map(|&i| {
                let inst = &pack.instances[i as usize];
                let a = &inst.affine;
                InstanceData {
                    m0: [a[0], a[1], a[2], a[3]],
                    m1: [a[4], a[5], a[6], a[7]],
                    m2: [a[8], a[9], a[10], a[11]],
                    ids: [inst.flags, 0, 0, 0],
                }
            })
            .collect();
        total_inst += data.len();

        commands.spawn((
            Mesh3d(handle),
            InstanceMaterialData(data),
            // Instances are scattered map-wide; the mesh's local AABB at origin
            // would frustum-cull the whole batch wrongly. Our own cull is M1.
            NoFrustumCulling,
        ));
        spawned += 1;
    }
    info!(
        "spawned {} instanced mesh entities ({} instances total)",
        spawned, total_inst
    );
}

// ---------------------------------------------------------------------------
// Queue custom draws into the Transparent3d phase (matches the example; the
// pipeline itself is opaque — depth test+write on, no blend — so geometry is
// depth-correct; the phase only adds a back-to-front sort we don't strictly
// need. Opaque3d/binned is the M1 upgrade).
// ---------------------------------------------------------------------------
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    material_meshes: Query<(Entity, &MainEntity), With<InstanceMaterialData>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    for (view, msaa) in &views {
        let Some(transparent_phase) =
            transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();

        for (entity, main_entity) in &material_meshes {
            let Some(mesh_instance) =
                render_mesh_instances.render_mesh_queue_data(*main_entity)
            else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let key =
                view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());
            let pipeline = pipelines
                .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                .unwrap();
            transparent_phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_custom,
                distance: rangefinder.distance_translation(&mesh_instance.translation),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

#[derive(Component)]
struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    // Instance data is static for M0, so build each entity's buffer ONCE. In the
    // retained render world the inserted InstanceBuffer persists across frames, so
    // filtering `Without<InstanceBuffer>` skips already-built entities and avoids
    // thousands of per-frame buffer reallocations/uploads at full-map scale (Codex P1).
    query: Query<(Entity, &InstanceMaterialData), Without<InstanceBuffer>>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("eft instance data buffer"),
            contents: bytemuck::cast_slice(instance_data.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instance_data.len(),
        });
    }
}

#[derive(Resource)]
struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

fn init_custom_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(CustomPipeline {
        shader: asset_server.load(SHADER_ASSET_PATH),
        mesh_pipeline: mesh_pipeline.clone(),
    });
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        descriptor.vertex.shader = self.shader.clone();
        // instance-rate buffer: 3 affine rows (loc 3,4,5) + id/flags uvec4 (loc 6).
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 4,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 5,
                },
                VertexAttribute {
                    format: VertexFormat::Uint32x4,
                    offset: 48,
                    shader_location: 6,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        // EFT deferred draws building shells solid from both sides; mirrors
        // (det<0) also rely on this so winding never matters. Cofactor normals
        // + double-sided = mirror-correct with no bake.
        descriptor.primitive.cull_mode = None;
        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    DrawMeshInstanced,
);

struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        (meshes, render_mesh_instances, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_allocator = mesh_allocator.into_inner();

        let Some(mesh_instance) =
            render_mesh_instances.render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.into_inner().get(mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };
        let Some(instance_buffer) = instance_buffer else {
            return RenderCommandResult::Skip;
        };
        let Some(vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id)
        else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };
                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, *index_format);
                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..instance_buffer.length as u32,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertex_buffer_slice.range, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}
