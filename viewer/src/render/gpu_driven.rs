//! M2 GPU-driven render path: GPU-resident buffers built ONCE, a compute frustum
//! cull that compacts survivors per-mesh + fills `DrawIndexedIndirectArgs`, and a
//! per-mesh `draw_indexed_indirect` loop. Selectable against the M0 path
//! (`instancing.rs`) via `EFT_RENDER=m0|gpu` â€” see `main.rs`.
//!
//! DATA FLOW (locked M2 design â€” do not redesign):
//!   * ONE-TIME build (CPU, main world): from the `Pack` assemble, GROUPED-BY-MESH
//!     and CONTIGUOUS, a global vertex buffer + index buffer (deterministic
//!     firstIndex/baseVertex we own, NOT MeshAllocator's dynamic packing), an
//!     instances SSBO ({row-major 3x4 affine, meshId, flags, worldSphere}), a
//!     meshMeta SSBO, and the per-mesh instanceBase offsets. The worldSphere radius
//!     is a CONSERVATIVE upper bound under the affine's 3x3 (Frobenius norm â€–Lâ€–_F,
//!     a guaranteed â‰¥ operator-norm bound), NOT max-column-norm (a LOWER bound that
//!     underestimates under shear and wrongly culls visible geometry). All
//!     computed on the CPU once. The heavy CPU blob is shipped to the render world
//!     as an `Arc` (cheap per-frame extract), and uploaded to the GPU exactly once.
//!   * PER FRAME (render world): upload the 6 Gribb-Hartmann frustum planes (tiny
//!     uniform); a compute node runs `cs_reset` (rewrite indirect args, zero
//!     instance_count) then `cs_cull` (one thread/instance: sphere-vs-frustum â†’
//!     atomicAdd instance_count, write visible[instanceBase+slot]=i). The draw is a
//!     Transparent3d phase item whose render command loops
//!     `draw_indexed_indirect` per mesh; the vertex shader fetches the affine from
//!     the instances SSBO via `visible[instance_index]`.
//!
//! THE #1 RULE (tarkov-unity-extraction): apply the raw 3x4 to verts, cofactor
//! normals, mirrors via double-sided â€” NEVER TRS-decompose.
#![allow(dead_code)] // POD layouts + frustum helper are shared / reference surface.

use core::num::NonZeroU32;
use std::collections::HashMap;
use std::sync::Arc;

use bevy::core_pipeline::core_3d::{
    graph::{Core3d, Node3d},
    Transparent3d, CORE_3D_DEPTH_FORMAT,
};
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::{lifetimeless::SRes, SystemParamItem};
use bevy::image::BevyDefault;
use bevy::mesh::VertexBufferLayout;
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey, SetMeshViewBindGroup,
};
use bevy::prelude::*;
use bevy::render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel},
    render_phase::{
        AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
        RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
    },
    render_resource::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
            uniform_buffer_sized,
        },
        AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
        BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
        ColorTargetState, ColorWrites, CompareFunction,
        ComputePassDescriptor, ComputePipelineDescriptor, DepthBiasState, DepthStencilState,
        Extent3d, FilterMode, FragmentState, IndexFormat, MultisampleState, PipelineCache,
        PrimitiveState, PrimitiveTopology, RenderPipelineDescriptor, Sampler, SamplerBindingType,
        SamplerDescriptor, ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines,
        StencilState, Texture, TextureDataOrder, TextureDescriptor, TextureDimension,
        TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
        VertexAttribute, VertexFormat, VertexState, VertexStepMode,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::MainEntity,
    view::{ExtractedView, ViewTarget},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};

pub use crate::eftpack::{BoundingSphere, GpuInstance};
use crate::render::LoadedPack;

// ===========================================================================
// POD GPU layouts (must match gpu_cull.wgsl / gpu_draw.wgsl exactly).
// ===========================================================================

/// Per-instance storage record. 80 bytes (16-aligned). Three ROW-MAJOR affine rows,
/// an id/flags uvec4, and the PRECOMPUTED conservative world bounding sphere.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct InstanceGpuRecord {
    pub m0: [f32; 4],
    pub m1: [f32; 4],
    pub m2: [f32; 4],
    /// x = mesh_id, y = flags, z,w = pad.
    pub ids: [u32; 4],
    /// xyz = world center, w = conservative world radius (Frobenius-norm scaled).
    pub sphere: [f32; 4],
}

/// Per-mesh static metadata. 32 bytes (16-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshMeta {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub instance_base: u32,
    pub instance_count: u32,
    pub _pad: [u32; 3],
}

/// wgpu `DrawIndexedIndirect` layout (20 bytes). Kept for reference / size checks;
/// the buffer is GPU-written so we never upload this from the CPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// Tiny per-frame cull uniform: 6 normalized inward frustum planes + counts. 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CullUniform {
    pub frustum: [[f32; 4]; 6],
    /// x = instance_count, y = mesh_count, z,w = pad.
    pub counts: [u32; 4],
}

/// Stride of one indirect draw record, in bytes.
pub const DRAW_ARG_STRIDE: u64 = 20;
/// Interleaved draw vertex stride (M3): pos f32x3 @0 + normal f32x3 @12 + uv f32x2 @24
/// + material_index u32 @32 = 36 bytes. The u32 material index is written as
/// `f32::from_bits(material_id)` so vertex_data stays a single `Vec<f32>`; the GPU reads
/// slot @32 as `Uint32` and recovers the id bit-exact (a pure reinterpretation, NOT a
/// numeric cast which would corrupt large ids).
pub const DRAW_VERTEX_STRIDE: u64 = 36;

/// Per-material GPU record (M3). 48 bytes, 16-aligned. Indexed DIRECTLY by the global
/// materialId (SubMesh.material_id == materials.json array index for this pack), which the
/// per-vertex `material_index` carries into the fragment shader.
///
/// `albedo_index` = index into the bindless albedo `binding_array`, or `NO_ALBEDO`
/// (0xFFFFFFFF) for the 93 materials with no albedo -> shade with tint/white.
/// `flags` bit0 = cutout (role=cutout / alphaMode=MASK -> discard albedo.a < alpha_cutoff).
/// `uv_xform` is REFERENCE ONLY (uvTilingBaked=true: tiling already in the vertex UVs;
/// the shader must NOT re-apply it). `tint` multiplies albedo.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuMaterial {
    pub albedo_index: u32,
    pub flags: u32,
    pub alpha_cutoff: f32,
    pub _pad: u32,
    pub uv_xform: [f32; 4],
    pub tint: [f32; 4],
}

/// `GpuMaterial::albedo_index` sentinel: material has no albedo texture.
pub const NO_ALBEDO: u32 = 0xFFFF_FFFF;
/// `GpuMaterial::flags` bit: cutout (alpha-test discard).
pub const MAT_FLAG_CUTOUT: u32 = 1 << 0;

// ===========================================================================
// Frustum plane extraction (Gribbâ€“Hartmann). Planes point INWARD; a sphere is
// visible when dot(plane.xyz, center) + plane.w >= -radius for all six.
//
// Feed `clip_from_world` (projection * view). wgpu clip space has z in [0,1].
// NOTE: Bevy's default camera is REVERSE-Z + infinite-far. Under that projection r2
// (clip.z = 0) is the FAR plane at infinity â€” a degenerate zero-normal plane that the
// length guard below turns into a harmless always-true test â€” and r3 - r2 is the valid
// active NEAR plane that actually culls. The plane SET is identical to Bevy's `Frustum`
// extraction, so the cull is correct regardless of these nominal labels.
// ===========================================================================
pub fn build_frustum_planes(clip_from_world: Mat4) -> [Vec4; 6] {
    let r0 = clip_from_world.row(0);
    let r1 = clip_from_world.row(1);
    let r2 = clip_from_world.row(2);
    let r3 = clip_from_world.row(3);

    let planes = [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // far (z=0; degenerate/always-true under infinite reverse-z)
        r3 - r2, // near (active culling plane)
    ];
    let mut out = [Vec4::ZERO; 6];
    for (i, p) in planes.into_iter().enumerate() {
        let n = Vec3::new(p.x, p.y, p.z).length();
        out[i] = if n > 0.0 { p / n } else { p };
    }
    out
}

/// GUARANTEED-CONSERVATIVE radius scale for a local sphere under the affine's linear
/// 3x3 `L`: the Frobenius norm â€–Lâ€–_F = sqrt(|c0|Â² + |c1|Â² + |c2|Â²).
///
/// Why Frobenius and NOT a power-iteration Ïƒ_max estimate (verify major finding): the
/// operator norm Ïƒ_max(L) is what we WANT, but a finite Rayleigh-quotient power
/// iteration converges to Ïƒ_max FROM BELOW and can start (near-)orthogonal to the
/// dominant eigenvector â€” so it UNDER-estimates, and an under-estimated radius wrongly
/// culls visible sheared/rotated instances (pop-out). Frobenius is a hard upper bound:
///     Ïƒ_max(L) <= â€–Lâ€–_F <= sqrt(3)Â·Ïƒ_max(L),
/// so the world sphere is NEVER too small (correctness) and at most ~1.73x too large
/// (a negligible loosening of the broad-phase cull). Max-column-norm â€” the original
/// bug â€” is a LOWER bound and must never be used. No decompose; matches the WGSL
/// `world_sphere_from_affine` fallback in gpu_cull.wgsl.
fn conservative_radius_scale(l: Mat3) -> f32 {
    let c0 = l.col(0);
    let c1 = l.col(1);
    let c2 = l.col(2);
    (c0.dot(c0) + c1.dot(c1) + c2.dot(c2)).sqrt()
}

// ===========================================================================
// CPU-assembled blob, built once in the main world, shipped to the render world by
// Arc (cheap per-frame extract), uploaded to the GPU exactly once.
// ===========================================================================
pub struct CpuData {
    /// Interleaved draw vertices (M3): [px,py,pz, nx,ny,nz, u,v, material_bits] per vertex,
    /// where `material_bits = f32::from_bits(material_id)` (read as Uint32 on the GPU).
    vertex_data: Vec<f32>,
    /// Global u32 indices (LOCAL to each mesh; base_vertex offsets them).
    index_data: Vec<u32>,
    instances: Vec<InstanceGpuRecord>,
    mesh_meta: Vec<MeshMeta>,
    /// Per-material GPU table, indexed by global materialId (== materials.json order).
    materials: Vec<GpuMaterial>,
    /// Unique albedo texture paths in bindless-array-index order. `GpuMaterial.albedo_index`
    /// indexes THIS list. Built in the SAME single pass as `materials` so indices can't drift.
    albedo_paths: Vec<String>,
    instance_total: u32,
    mesh_count: u32,
}

#[derive(Resource, Clone)]
pub struct ExtractedCpuData(Arc<CpuData>);

impl ExtractResource for ExtractedCpuData {
    type Source = ExtractedCpuData;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Marker for the camera whose frustum drives the GPU cull. Extracted so the render
/// world can pick THE player view out of Bevy's multiple ExtractedViews â€” otherwise
/// `views.iter().next()` grabs a prepass/default view nondeterministically and the cull
/// runs against a static wrong frustum (half the map wrongly culled, no camera tracking).
#[derive(Component, Clone, Default)]
pub struct CullCamera;

impl ExtractComponent for CullCamera {
    type QueryData = &'static CullCamera;
    type QueryFilter = ();
    type Out = CullCamera;
    fn extract_component(_: QueryItem<'_, '_, Self::QueryData>) -> Option<Self> {
        Some(CullCamera)
    }
}

/// Marker for the single render-world entity that carries the GPU-driven draw phase
/// item. Extracted so it has a `MainEntity` in the render world.
#[derive(Component, Clone, Default)]
pub struct GpuDrivenTag;

impl ExtractComponent for GpuDrivenTag {
    type QueryData = &'static GpuDrivenTag;
    type QueryFilter = ();
    type Out = GpuDrivenTag;
    fn extract_component(_: QueryItem<'_, '_, Self::QueryData>) -> Option<Self> {
        Some(GpuDrivenTag)
    }
}

// ===========================================================================
// Plugin.
// ===========================================================================
pub struct EftGpuDrivenPlugin;

impl Plugin for EftGpuDrivenPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<GpuDrivenTag>::default(),
            ExtractComponentPlugin::<CullCamera>::default(),
            ExtractResourcePlugin::<ExtractedCpuData>::default(),
        ))
        .add_systems(Startup, build_cpu_data)
        .add_systems(Update, free_cpu_staging);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_render_command::<Transparent3d, DrawGpuDriven>()
            .init_resource::<SpecializedRenderPipelines<EftDrawPipeline>>()
            .add_systems(RenderStartup, init_gpu_pipelines)
            .add_systems(
                Render,
                (
                    prepare_gpu_buffers.in_set(RenderSystems::PrepareResources),
                    upload_frustum
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    queue_gpu_driven.in_set(RenderSystems::QueueMeshes),
                ),
            )
            .add_render_graph_node::<EftCullNode>(Core3d, EftCullLabel)
            .add_render_graph_edges(Core3d, (EftCullLabel, Node3d::StartMainPass));
    }
}

// ===========================================================================
// Main-world one-time CPU assembly.
// ===========================================================================

/// The CPU staging blob (~650 MiB of repacked geometry) is only needed for the
/// one-time GPU upload. Drop the main-world source a few frames in â€” by then the
/// render world has extracted + uploaded it, and prepare_gpu_buffers frees the
/// render-world copy â€” so the whole Arc is released (Codex P1).
fn free_cpu_staging(
    mut commands: Commands,
    mut frames: Local<u32>,
    cpu: Option<Res<ExtractedCpuData>>,
) {
    if cpu.is_none() {
        return;
    }
    *frames += 1;
    if *frames >= 4 {
        commands.remove_resource::<ExtractedCpuData>();
    }
}

fn build_cpu_data(mut commands: Commands, pack: Option<Res<LoadedPack>>) {
    let Some(pack) = pack else {
        return;
    };
    let pack = &pack.0;
    let by_mesh = pack.instances_by_mesh();
    let local_spheres = match pack.bounding_spheres() {
        Ok(s) => s,
        Err(e) => {
            error!("gpu-driven: bounding_spheres failed: {e:#}");
            return;
        }
    };

    // --- material table + unique albedo list, ONE ordered pass (index consistency) ---
    // materials.json is authored so material.id == array index; the per-vertex material_index
    // (a global materialId from SubMesh.material_id) indexes this Vec directly. Dedup albedo
    // paths first-seen: the unique list IS the bindless-array order, and each material's
    // albedo_index is assigned from the SAME pass so the two can never disagree.
    let mut materials_gpu: Vec<GpuMaterial> = Vec::with_capacity(pack.materials.len());
    let mut albedo_paths: Vec<String> = Vec::new();
    let mut path_to_index: HashMap<String, u32> = HashMap::new();
    for mat in &pack.materials {
        let albedo_index = match mat.albedo.as_deref() {
            Some(p) if !p.is_empty() => *path_to_index.entry(p.to_string()).or_insert_with(|| {
                let idx = albedo_paths.len() as u32;
                albedo_paths.push(p.to_string());
                idx
            }),
            _ => NO_ALBEDO,
        };
        // M3 v1 scope: opaque + cutout. role=cutout / alphaMode=MASK -> alpha-test discard.
        // BLEND (glass/water/decal) deferred to M3b: rendered opaque here (no cutout bit).
        let mut flags = 0u32;
        if mat.role == "cutout" || mat.alpha_mode == "MASK" {
            flags |= MAT_FLAG_CUTOUT;
        }
        materials_gpu.push(GpuMaterial {
            albedo_index,
            flags,
            alpha_cutoff: mat.alpha_cutoff,
            _pad: 0,
            uv_xform: mat.uv_xform, // reference only (uvTilingBaked=true); shader must NOT apply
            tint: mat.tint,
        });
    }
    info!(
        "gpu-driven M3: {} materials, {} unique albedo textures ({} untextured)",
        materials_gpu.len(),
        albedo_paths.len(),
        materials_gpu
            .iter()
            .filter(|m| m.albedo_index == NO_ALBEDO)
            .count(),
    );

    let mut vertex_data: Vec<f32> = Vec::new();
    let mut index_data: Vec<u32> = Vec::new();
    let mut instances: Vec<InstanceGpuRecord> = Vec::new();
    let mut mesh_meta: Vec<MeshMeta> = Vec::new();

    let mut vtx_cursor: u32 = 0;
    let mut idx_cursor: u32 = 0;
    let mut inst_cursor: u32 = 0;

    for (mi, m) in pack.manifest.meshes.iter().enumerate() {
        let inst_ids = &by_mesh[mi];
        if inst_ids.is_empty() {
            continue; // orphan mesh â€” nothing references it
        }
        let geom = match pack.mesh_geom(m) {
            Ok(g) => g,
            Err(e) => {
                warn!("gpu-driven: mesh {} '{}' skipped: {:#}", m.id, m.name, e);
                continue;
            }
        };
        if geom.positions.is_empty() || geom.indices.is_empty() {
            continue;
        }

        // --- geometry into the global vertex/index buffers (offsets we own) ---
        let base_vertex = vtx_cursor as i32;
        let n = geom.positions.len();

        // M3: per-vertex material index. Each submesh is a contiguous index range into this
        // mesh's single vertex array; across ALL multi-submesh meshes in this pack the
        // submeshes reference DISJOINT vertex sets (measured: zero cross-submesh sharing),
        // so tagging each referenced vertex with its submesh's materialId needs NO vertex
        // duplication. Verts not referenced by any submesh are never rasterized (they are
        // absent from the drawn index run), so the fallback material is irrelevant; we seed
        // it to the first submesh's id for safety.
        let default_mat = m.submeshes.first().map(|s| s.material_id).unwrap_or(0);
        let mut vert_mat: Vec<u32> = vec![default_mat; n];
        for sm in &m.submeshes {
            let start = sm.idx_start as usize;
            let end = start + sm.idx_count as usize;
            for &vi in &geom.indices[start..end.min(geom.indices.len())] {
                if (vi as usize) < n {
                    vert_mat[vi as usize] = sm.material_id;
                }
            }
        }

        for k in 0..n {
            let p = geom.positions[k];
            let nrm = *geom.normals.get(k).unwrap_or(&[0.0, 1.0, 0.0]);
            let uv = *geom.uvs.get(k).unwrap_or(&[0.0, 0.0]);
            vertex_data.extend_from_slice(&[
                p[0], p[1], p[2],
                nrm[0], nrm[1], nrm[2],
                uv[0], uv[1],
                f32::from_bits(vert_mat[k]), // material_index (read as Uint32 on the GPU)
            ]);
        }
        vtx_cursor += n as u32;

        let first_index = idx_cursor;
        index_data.extend_from_slice(&geom.indices); // indices are mesh-local
        let index_count = geom.indices.len() as u32;
        idx_cursor += index_count;

        // --- instances (grouped-by-mesh, contiguous) with conservative world sphere ---
        let instance_base = inst_cursor;
        let bs = local_spheres[mi];
        let local_center = Vec3::new(bs[0], bs[1], bs[2]);
        let local_r = bs[3];
        for &i in inst_ids {
            let inst = &pack.instances[i as usize];
            let a = &inst.affine;
            let aff = inst.affine3a();
            let lin = Mat3::from(aff.matrix3);
            let center = aff.transform_point3(local_center);
            let radius = local_r * conservative_radius_scale(lin);
            instances.push(InstanceGpuRecord {
                m0: [a[0], a[1], a[2], a[3]],
                m1: [a[4], a[5], a[6], a[7]],
                m2: [a[8], a[9], a[10], a[11]],
                ids: [mesh_meta.len() as u32, inst.flags, 0, 0],
                sphere: [center.x, center.y, center.z, radius],
            });
        }
        let instance_count = inst_ids.len() as u32;
        inst_cursor += instance_count;

        mesh_meta.push(MeshMeta {
            index_count,
            first_index,
            base_vertex,
            instance_base,
            instance_count,
            _pad: [0; 3],
        });
    }

    let mesh_count = mesh_meta.len() as u32;
    let instance_total = inst_cursor;
    if mesh_count == 0 || instance_total == 0 {
        warn!("gpu-driven: nothing to draw (0 meshes / 0 instances)");
        return;
    }

    info!(
        "gpu-driven: assembled {} meshes, {} instances, {} verts, {} indices",
        mesh_count,
        instance_total,
        vtx_cursor,
        index_data.len()
    );

    commands.insert_resource(ExtractedCpuData(Arc::new(CpuData {
        vertex_data,
        index_data,
        instances,
        mesh_meta,
        materials: materials_gpu,
        albedo_paths,
        instance_total,
        mesh_count,
    })));
    // one entity to hang the draw phase item on (ignored by the draw command)
    commands.spawn((GpuDrivenTag, Name::new("eft_gpu_driven_draw")));
}

// ===========================================================================
// Render-world persistent resources.
// ===========================================================================
#[derive(Resource)]
struct EftComputePipelines {
    reset_id: CachedComputePipelineId,
    cull_id: CachedComputePipelineId,
    cull_layout: BindGroupLayout,
}

#[derive(Resource, Clone)]
struct EftDrawPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
    ssbo_layout: BindGroupLayout,
    /// group(2) bindless material layout: material-table SSBO + albedo `binding_array` +
    /// sampler. Built in `prepare_gpu_buffers` (needs the unique-albedo count for the
    /// `binding_array` size) and the pipeline is re-inserted with it set. `None` until then;
    /// `queue_gpu_driven` gates specialization on it being `Some` (M3).
    material_layout: Option<BindGroupLayout>,
}

#[derive(Resource)]
struct EftGpuBuffers {
    vertex: Buffer,
    index: Buffer,
    indirect: Buffer,
    cull_uniform: Buffer,
    mesh_count: u32,
    instance_total: u32,
}

#[derive(Resource)]
struct EftCullBindGroup(BindGroup);

#[derive(Resource)]
struct EftDrawBindGroup(BindGroup);

/// Owns the bindless material GPU resources so the `TextureView`s (and the material SSBO)
/// outlive `EftMaterialBindGroup`. Built once in `prepare_gpu_buffers`.
#[derive(Resource)]
struct EftMaterialResources {
    material_buf: Buffer,
    #[allow(dead_code)] // kept alive so the views/bind group stay valid
    textures: Vec<Texture>,
    views: Vec<TextureView>,
    #[allow(dead_code)]
    sampler: Sampler,
}

#[derive(Resource)]
struct EftMaterialBindGroup(BindGroup);

// ---- RenderStartup: bind group layouts, shaders, compute pipelines ----------
fn init_gpu_pipelines(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mesh_pipeline: Res<MeshPipeline>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // HARD GUARD (verify finding): every mesh but the first bakes a nonzero
    // first_instance (= instance_base) into the GPU-written indirect args. Without
    // INDIRECT_FIRST_INSTANCE the driver silently ignores it, @builtin(instance_index)
    // restarts at 0 per mesh, and visible[instance_index] reads mesh 0's region â†’ the
    // whole scene draws the wrong instances with no validation error. On native Vulkan
    // with Bevy's default (Functionality priority) the feature is auto-enabled; if it
    // is genuinely absent we DISABLE the GPU path entirely (skip inserting the pipeline
    // resources so queue/prepare/node all no-op â†’ empty view, not scrambled geometry)
    // and tell the user to fall back to the M0 path. We do NOT force-request it via
    // WgpuSettings because that would hard-panic device creation on adapters lacking it;
    // graceful disable is safer given GpuDriven is the default path.
    use bevy::render::settings::WgpuFeatures;
    let need = WgpuFeatures::INDIRECT_FIRST_INSTANCE | WgpuFeatures::MULTI_DRAW_INDIRECT;
    if !render_device.features().contains(need) {
        error!(
            "gpu-driven: adapter lacks INDIRECT_FIRST_INSTANCE | MULTI_DRAW_INDIRECT â€” the \
             GPU-driven path is DISABLED (view will be empty). Re-run with EFT_RENDER=m0 for \
             the instanced path."
        );
        return; // no pipeline resources inserted â†’ entire gpu-driven path no-ops
    }
    // M3 bindless guard (graceful-disable, same as MULTI_DRAW above). TEXTURE_BINDING_ARRAY:
    // the albedo binding_array itself. SAMPLED_..._NON_UNIFORM_INDEXING: adjacent fragments in
    // one draw sample DIFFERENT albedo_tex[idx] (index is non-uniform) â€” without it sampling is
    // undefined/garbage even though the shader compiles. PARTIALLY_BOUND_BINDING_ARRAY: lets the
    // array be under-filled without padding. All three auto-enable on native Vulkan/RTX 5090
    // under Bevy's default (Functionality) priority; if absent we disable the whole path (empty
    // view) exactly like the MULTI_DRAW guard rather than force-request + hard-panic.
    // Every array slot is supplied (count == texture count), so PARTIALLY_BOUND is NOT needed;
    // requiring it would needlessly disable adapters that support the rest but not it (Codex P2).
    let need_bindless = WgpuFeatures::TEXTURE_BINDING_ARRAY
        | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    if !render_device.features().contains(need_bindless) {
        error!(
            "gpu-driven M3: adapter lacks TEXTURE_BINDING_ARRAY | \
             SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING â€” the textured \
             GPU-driven path is DISABLED (view will be empty). Re-run with EFT_RENDER=m0 for \
             the instanced path."
        );
        return; // no pipeline resources inserted â†’ entire gpu-driven path no-ops
    }

    let cull_layout = render_device.create_bind_group_layout(
        "eft_cull_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),          // 0: CullGlobals
                storage_buffer_read_only_sized(false, None), // 1: instances
                storage_buffer_read_only_sized(false, None), // 2: mesh_meta
                storage_buffer_sized(false, None),           // 3: visible (rw)
                storage_buffer_sized(false, None),           // 4: indirect (rw)
            ),
        ),
    );

    let ssbo_layout = render_device.create_bind_group_layout(
        "eft_draw_ssbo_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, None), // 0: instances
                storage_buffer_read_only_sized(false, None), // 1: visible
            ),
        ),
    );

    let cull_shader = asset_server.load("shaders/gpu_cull.wgsl");
    let draw_shader = asset_server.load("shaders/gpu_draw.wgsl");

    let reset_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("eft_cull_reset".into()),
        layout: vec![cull_layout.clone()],
        push_constant_ranges: vec![],
        shader: cull_shader.clone(),
        shader_defs: vec![],
        entry_point: Some("cs_reset".into()),
        zero_initialize_workgroup_memory: false,
    });
    let cull_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("eft_cull".into()),
        layout: vec![cull_layout.clone()],
        push_constant_ranges: vec![],
        shader: cull_shader,
        shader_defs: vec![],
        entry_point: Some("cs_cull".into()),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(EftComputePipelines {
        reset_id,
        cull_id,
        cull_layout,
    });
    commands.insert_resource(EftDrawPipeline {
        shader: draw_shader,
        mesh_pipeline: mesh_pipeline.clone(),
        ssbo_layout,
        material_layout: None, // filled in prepare_gpu_buffers once the albedo count is known
    });
}

// ---- PrepareResources: build all GPU buffers + bind groups ONCE -------------
#[allow(clippy::too_many_arguments)]
fn prepare_gpu_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    cpu: Option<Res<ExtractedCpuData>>,
    already: Option<Res<EftGpuBuffers>>,
    compute: Option<Res<EftComputePipelines>>,
    draw: Option<Res<EftDrawPipeline>>,
) {
    if already.is_some() {
        // Buffers are built. Drop any render-world copy of the ~650 MiB CPU staging
        // blob that got re-extracted before free_cpu_staging drops the main-world
        // source, so the whole Arc is released (Codex P1).
        if cpu.is_some() {
            commands.remove_resource::<ExtractedCpuData>();
        }
        return;
    }
    let (Some(cpu), Some(compute), Some(draw)) = (cpu, compute, draw) else {
        return; // wait for the extracted blob + layouts (also skipped if feature-disabled)
    };
    let cpu = &cpu.0;

    let vertex = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_vertex"),
        contents: bytemuck::cast_slice(&cpu.vertex_data),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    });
    let index = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_index"),
        contents: bytemuck::cast_slice(&cpu.index_data),
        usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
    });
    let instances = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_instances"),
        contents: bytemuck::cast_slice(&cpu.instances),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let mesh_meta = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_mesh_meta"),
        contents: bytemuck::cast_slice(&cpu.mesh_meta),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let visible = render_device.create_buffer(&BufferDescriptor {
        label: Some("eft_gpu_visible"),
        size: cpu.instance_total as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indirect = render_device.create_buffer(&BufferDescriptor {
        label: Some("eft_gpu_indirect"),
        size: cpu.mesh_count as u64 * DRAW_ARG_STRIDE,
        usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // seed the cull uniform to all-zero planes (= everything visible) so frame 0,
    // before the first frustum upload, draws rather than randomly culling.
    let seed = CullUniform {
        frustum: [[0.0; 4]; 6],
        counts: [cpu.instance_total, cpu.mesh_count, 0, 0],
    };
    let cull_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_cull_uniform"),
        contents: bytemuck::bytes_of(&seed),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let cull_bg = render_device.create_bind_group(
        "eft_cull_bg",
        &compute.cull_layout,
        &BindGroupEntries::sequential((
            cull_uniform.as_entire_binding(),
            instances.as_entire_binding(),
            mesh_meta.as_entire_binding(),
            visible.as_entire_binding(),
            indirect.as_entire_binding(),
        )),
    );
    let draw_bg = render_device.create_bind_group(
        "eft_draw_bg",
        &draw.ssbo_layout,
        &BindGroupEntries::sequential((
            instances.as_entire_binding(),
            visible.as_entire_binding(),
        )),
    );

    // ---- M3: bindless material table + albedo texture array (built ONCE) -----------
    // material-table SSBO (indexed by the per-vertex global materialId in the fragment).
    let material_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_material_table"),
        contents: bytemuck::cast_slice(&cpu.materials),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Decode + upload every UNIQUE albedo (image crate -> Rgba8UnormSrgb). One texture per
    // entry, IN THE SAME order as cpu.albedo_paths, so GpuMaterial.albedo_index stays aligned;
    // a failed decode still pushes a placeholder at its slot to preserve that alignment.
    let mut textures: Vec<Texture> = Vec::with_capacity(cpu.albedo_paths.len());
    let mut views: Vec<TextureView> = Vec::with_capacity(cpu.albedo_paths.len());
    for path in &cpu.albedo_paths {
        let (tex, view) = load_albedo_texture(&render_device, &render_queue, path);
        textures.push(tex);
        views.push(view);
    }
    // A binding_array needs >= 1 element; if this pack referenced no albedo at all, synth a
    // 1x1 white so the layout/bind group stay valid (all materials then hit the sentinel).
    if views.is_empty() {
        let (tex, view) = make_dummy_texture(&render_device, &render_queue);
        textures.push(tex);
        views.push(view);
    }
    let tex_count = views.len() as u32;

    let albedo_sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("eft_albedo_sampler"),
        // Tiling is baked into the vertex UVs (uvTilingBaked=true) so UVs can exceed [0,1] ->
        // Repeat is the correct wrap for the baked tiling.
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..default()
    });

    // group(2): material-table SSBO (0) + albedo binding_array of size tex_count (1) + sampler (2).
    let material_layout = render_device.create_bind_group_layout(
        "eft_material_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, storage_buffer_read_only_sized(false, None)),
                (
                    1,
                    texture_2d(TextureSampleType::Float { filterable: true })
                        .count(NonZeroU32::new(tex_count).unwrap()),
                ),
                (2, sampler(SamplerBindingType::Filtering)),
            ),
        ),
    );

    // TextureViewArray wants raw &[&wgpu::TextureView]; Bevy's TextureView derefs to it.
    let view_refs: Vec<_> = views.iter().map(|v| &**v).collect();
    let material_bg = render_device.create_bind_group(
        "eft_material_bg",
        &material_layout,
        &BindGroupEntries::with_indices((
            (0, material_buf.as_entire_binding()),
            (1, &view_refs[..]),
            (2, &albedo_sampler),
        )),
    );

    // Re-insert the draw pipeline WITH the material layout now that tex_count is known, so
    // specialize() can build the 3-group pipeline layout (view / ssbo / material).
    commands.insert_resource(EftDrawPipeline {
        shader: draw.shader.clone(),
        mesh_pipeline: draw.mesh_pipeline.clone(),
        ssbo_layout: draw.ssbo_layout.clone(),
        material_layout: Some(material_layout),
    });
    commands.insert_resource(EftMaterialResources {
        material_buf,
        textures,
        views,
        sampler: albedo_sampler,
    });
    commands.insert_resource(EftMaterialBindGroup(material_bg));
    info!(
        "gpu-driven M3: {} albedo textures uploaded, material table + bindless bind group built",
        tex_count
    );

    commands.insert_resource(EftGpuBuffers {
        vertex,
        index,
        indirect,
        cull_uniform,
        mesh_count: cpu.mesh_count,
        instance_total: cpu.instance_total,
    });
    commands.insert_resource(EftCullBindGroup(cull_bg));
    commands.insert_resource(EftDrawBindGroup(draw_bg));
    info!("gpu-driven: GPU buffers + bind groups built (once)");
}

// ---- M3 texture upload helpers ---------------------------------------------
/// Decode one albedo PNG (full-res, `image` crate) and upload it as an Rgba8UnormSrgb GPU
/// texture (+ view). Albedo is sRGB (conventions.colorSpace.albedo='srgb') so the srgb
/// format makes the sampler return linear. On ANY read/decode failure returns a 1x1 magenta
/// placeholder so the bindless-array index stays aligned with materials.json â€” a shifted
/// index would texture the whole map wrong with no error.
fn load_albedo_texture(
    device: &RenderDevice,
    queue: &RenderQueue,
    path: &str,
) -> (Texture, TextureView) {
    match image::open(path) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            upload_rgba8_srgb(device, queue, w.max(1), h.max(1), &rgba, "eft_albedo")
        }
        Err(e) => {
            warn!("gpu-driven M3: albedo '{path}' failed to load ({e}); using placeholder");
            upload_rgba8_srgb(device, queue, 1, 1, &[255u8, 0, 255, 255], "eft_albedo_missing")
        }
    }
}

/// 1x1 white placeholder for a pack that referenced no albedo at all (keeps the
/// binding_array non-empty).
fn make_dummy_texture(device: &RenderDevice, queue: &RenderQueue) -> (Texture, TextureView) {
    upload_rgba8_srgb(device, queue, 1, 1, &[255u8, 255, 255, 255], "eft_albedo_dummy")
}

fn upload_rgba8_srgb(
    device: &RenderDevice,
    queue: &RenderQueue,
    width: u32,
    height: u32,
    rgba: &[u8],
    label: &'static str,
) -> (Texture, TextureView) {
    // create_texture_with_data handles the 256-byte row-padding for the staging copy.
    let tex = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1, // full mip chain / BC7 compression deferred to M3b
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        rgba,
    );
    let view = tex.create_view(&TextureViewDescriptor::default());
    (tex, view)
}

// ---- PrepareResources: upload the 6 frustum planes (tiny) each frame --------
fn upload_frustum(
    render_queue: Res<RenderQueue>,
    buffers: Option<Res<EftGpuBuffers>>,
    views: Query<&ExtractedView, With<CullCamera>>,
) {
    let Some(buffers) = buffers else {
        return;
    };
    // Only the tagged player camera's view (Bevy has multiple ExtractedViews).
    let Some(view) = views.iter().next() else {
        return;
    };
    let clip_from_world = view.clip_from_world.unwrap_or_else(|| {
        view.clip_from_view * view.world_from_view.to_matrix().inverse()
    });
    let planes = build_frustum_planes(clip_from_world);
    let uniform = CullUniform {
        frustum: [
            planes[0].to_array(),
            planes[1].to_array(),
            planes[2].to_array(),
            planes[3].to_array(),
            planes[4].to_array(),
            planes[5].to_array(),
        ],
        counts: [buffers.instance_total, buffers.mesh_count, 0, 0],
    };
    render_queue.write_buffer(&buffers.cull_uniform, 0, bytemuck::bytes_of(&uniform));
}

// ---- QueueMeshes: specialize the draw pipeline + add ONE phase item ----------
fn queue_gpu_driven(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<EftDrawPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    draw_pipeline: Option<Res<EftDrawPipeline>>,
    // Gate on the GPU buffers + bind groups actually existing before adding the phase
    // item: the DrawGpuDriven render command fetches EftGpuBuffers/EftDrawBindGroup via
    // SRes (which PANICS if missing). EftDrawPipeline is inserted at RenderStartup but
    // the buffers are only built once the extracted CPU blob has arrived + prepared, so
    // pipeline-ready does NOT imply buffers-ready (verify finding).
    buffers: Option<Res<EftGpuBuffers>>,
    markers: Query<(Entity, &MainEntity), With<GpuDrivenTag>>,
    mut transparent_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let (Some(draw_pipeline), Some(_buffers)) = (draw_pipeline, buffers) else {
        return;
    };
    // M3: don't specialize until the material layout exists (built in prepare_gpu_buffers once
    // the albedo count is known). specialize() needs it for the group(2) pipeline layout, and
    // DrawGpuDrivenInner needs the matching EftMaterialBindGroup â€” both land in the same prepare
    // that builds the (already-gated) buffers, so this is a belt-and-suspenders skip, never a
    // panic on a None layout.
    if draw_pipeline.material_layout.is_none() {
        return;
    }
    let draw_fn = draw_functions.read().id::<DrawGpuDriven>();

    for (view, msaa) in &views {
        let Some(phase) = transparent_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let key = EftDrawKey {
            samples: msaa.samples(),
            hdr: view.hdr,
        };
        let pipeline = pipelines.specialize(&pipeline_cache, &draw_pipeline, key);

        for (entity, main_entity) in &markers {
            phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_fn,
                distance: 0.0,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct EftDrawKey {
    samples: u32,
    hdr: bool,
}

impl SpecializedRenderPipeline for EftDrawPipeline {
    type Key = EftDrawKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mesh_key =
            MeshPipelineKey::from_msaa_samples(key.samples) | MeshPipelineKey::from_hdr(key.hdr);
        // group(0): reuse Bevy's mesh view bind-group layout so SetMeshViewBindGroup<0>
        // + position_world_to_clip resolve. group(1): our storage buffers.
        let view_layout = self
            .mesh_pipeline
            .get_view_layout(MeshPipelineViewLayoutKey::from(mesh_key))
            .main_layout
            .clone();
        let format = if key.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };
        // group(2): bindless material layout. queue_gpu_driven gates specialization on this
        // being Some, so the pipeline is never built without it.
        let material_layout = self
            .material_layout
            .clone()
            .expect("EftDrawPipeline.material_layout must be set before specialize (M3)");

        RenderPipelineDescriptor {
            label: Some("eft_gpu_draw".into()),
            layout: vec![view_layout, self.ssbo_layout.clone(), material_layout],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![VertexBufferLayout {
                    array_stride: DRAW_VERTEX_STRIDE,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                        // M3: per-vertex material index (read bit-exact as Uint32 @32).
                        VertexAttribute {
                            format: VertexFormat::Uint32,
                            offset: 32,
                            shader_location: 3,
                        },
                    ],
                }],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                // EFT shells + mirrors are double-sided; winding never matters.
                cull_mode: None,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                // Bevy uses reverse-z; opaque geometry compares GreaterEqual.
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        }
    }
}

// ===========================================================================
// Compute node: cs_reset then cs_cull, before the main pass.
// ===========================================================================
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct EftCullLabel;

struct EftCullNode;

impl FromWorld for EftCullNode {
    fn from_world(_: &mut World) -> Self {
        Self
    }
}

impl Node for EftCullNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(buffers), Some(bind), Some(pipelines)) = (
            world.get_resource::<EftGpuBuffers>(),
            world.get_resource::<EftCullBindGroup>(),
            world.get_resource::<EftComputePipelines>(),
        ) else {
            return Ok(()); // buffers not built yet (or feature-disabled)
        };
        let cache = world.resource::<PipelineCache>();
        let (Some(reset), Some(cull)) = (
            cache.get_compute_pipeline(pipelines.reset_id),
            cache.get_compute_pipeline(pipelines.cull_id),
        ) else {
            return Ok(()); // pipelines still compiling
        };

        let bg = &bind.0;
        let reset_groups = buffers.mesh_count.div_ceil(64);
        let cull_groups = buffers.instance_total.div_ceil(64);
        let encoder = render_context.command_encoder();

        // Separate passes â†’ wgpu inserts a barrier so cs_reset is fully visible to cs_cull.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eft_cull_reset"),
                timestamp_writes: None,
            });
            pass.set_pipeline(reset);
            pass.set_bind_group(0, &**bg, &[]);
            pass.dispatch_workgroups(reset_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eft_cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cull);
            pass.set_bind_group(0, &**bg, &[]);
            pass.dispatch_workgroups(cull_groups, 1, 1);
        }
        Ok(())
    }
}

// ===========================================================================
// Draw: per-mesh draw_indexed_indirect loop (view bind group set by the chain).
// ===========================================================================
type DrawGpuDriven = (SetItemPipeline, SetMeshViewBindGroup<0>, DrawGpuDrivenInner);

struct DrawGpuDrivenInner;

impl<P: PhaseItem> RenderCommand<P> for DrawGpuDrivenInner {
    // Optional fetch so a missing resource returns Skip instead of panicking â€” belt &
    // suspenders on top of queue_gpu_driven's buffers gate (verify finding). group(2) is the
    // M3 bindless material bind group (built in the same prepare as the buffers).
    type Param = (
        Option<SRes<EftGpuBuffers>>,
        Option<SRes<EftDrawBindGroup>>,
        Option<SRes<EftMaterialBindGroup>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        _entity: Option<()>,
        (buffers, draw_bg, material_bg): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (Some(buffers), Some(draw_bg), Some(material_bg)) = (buffers, draw_bg, material_bg)
        else {
            return RenderCommandResult::Skip;
        };
        let buffers = buffers.into_inner();
        let draw_bg = draw_bg.into_inner();
        let material_bg = material_bg.into_inner();

        pass.set_bind_group(1, &draw_bg.0, &[]);
        pass.set_bind_group(2, &material_bg.0, &[]); // M3: bindless materials/textures/sampler
        pass.set_vertex_buffer(0, buffers.vertex.slice(..));
        pass.set_index_buffer(buffers.index.slice(..), 0, IndexFormat::Uint32);

        // ONE multi-draw for ALL meshes: the GPU reads every mesh's DrawIndexedIndirectArgs
        // (index_count / first_index / base_vertex / instance_base + the cull-filled
        // instance_count) straight from the indirect buffer â€” near-zero CPU submission
        // (replaces a 6.5k-call loop). Fully-culled meshes have instance_count 0 â†’ nothing
        // drawn. Requires MULTI_DRAW_INDIRECT (guarded at pipeline init).
        pass.multi_draw_indexed_indirect(&buffers.indirect, 0, buffers.mesh_count);
        RenderCommandResult::Success
    }
}
