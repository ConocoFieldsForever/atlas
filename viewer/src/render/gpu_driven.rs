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
            texture_2d_array, texture_3d, uniform_buffer_sized,
        },
        AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
        BlendState, Buffer,
        BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction,
        ComputePassDescriptor, ComputePipelineDescriptor, DepthBiasState, DepthStencilState,
        Extent3d, FilterMode, FragmentState, IndexFormat, LoadOp, MultisampleState, Operations,
        PipelineCache, PrimitiveState, PrimitiveTopology, RenderPassDepthStencilAttachment,
        RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
        SamplerDescriptor, ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines,
        StencilState, StoreOp, Texture, TextureDataOrder, TextureDescriptor, TextureDimension,
        TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
        TextureViewDimension, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::MainEntity,
    view::{ExtractedView, ViewTarget},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use serde::Deserialize;

pub use crate::eftpack::{BoundingSphere, GpuInstance};
use crate::eftpack::Pack;
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
/// Interleaved draw vertex stride (M3/M3b2): pos f32x3 @0 + normal f32x3 @12 + uv f32x2 @24
/// + material_index u32 @32 + color f32x4 @36 = 52 bytes. The u32 material index is written
/// as `f32::from_bits(material_id)` so vertex_data stays a single `Vec<f32>`; the GPU reads
/// slot @32 as `Uint32` and recovers the id bit-exact (a pure reinterpretation, NOT a numeric
/// cast which would corrupt large ids). The trailing f32x4 @36 is the per-vertex COLOR_0
/// vert-paint weight (interpolated); the SoftCutout road/track feather rides on color.a.
pub const DRAW_VERTEX_STRIDE: u64 = 52;

/// Per-material GPU record (M3; 80 bytes after Phase 2b normal mapping, 160 bytes after #6 detail maps), 16-aligned. Indexed DIRECTLY by the global
/// materialId (SubMesh.material_id == materials.json array index for this pack), which the
/// per-vertex `material_index` carries into the fragment shader.
///
/// `albedo_index` = index into the bindless albedo `binding_array`, or `NO_ALBEDO`
/// (0xFFFFFFFF) for the 93 materials with no albedo -> shade with tint/white.
/// `flags` bit0 = cutout (role=cutout / alphaMode=MASK -> discard albedo.a < alpha_cutoff).
/// `uv_xform` is REFERENCE ONLY (uvTilingBaked=true: tiling already in the vertex UVs;
/// the shader must NOT re-apply it). `tint` multiplies albedo.
///
/// M3b2: `vp` = `[_AlphaStrength, _Cutoff, _AlphaHeight, 0]` (from `Material.vp.softCutout`;
/// zeros for non-SoftCutout materials). In the BLEND pass a SoftCutout material's coverage is
/// `clamp(color.a * vp.x - (vp.y - vp.z), 0, 1)` (feathers roads/tire-tracks into the ground),
/// NOT tex.a (tex.a is smoothness for that shader family).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuMaterial {
    pub albedo_index: u32,
    pub flags: u32,
    pub alpha_cutoff: f32,
    /// Phase 1.6 GGX spec: repurposed from `_pad` (offset 12, NO size change) — per-material
    /// roughness for the dielectric spec lobe, clamped to [0.03, 1.0]. Glass carries ~0.05 so
    /// it comes through sharp; default 0.55 for materials with no authored roughness.
    pub roughness: f32,
    pub uv_xform: [f32; 4],
    pub tint: [f32; 4],
    /// SoftCutout params [_AlphaStrength, _Cutoff, _AlphaHeight, 0]. @48 (16-aligned).
    pub vp: [f32; 4],
    /// Phase 2b normal mapping: 4th 16-byte block @64 (size 64 -> 80).
    /// `normal_index` = index into the bindless `normal_tex` array, or `NO_NORMAL`
    /// (0xFFFFFFFF) for materials with no normal map -> shade with the geometric normal.
    pub normal_index: u32,
    /// bit0 = green-flip (DirectX-convention Y down; negate sampled n.y). Set from
    /// Material.normalGreenFlip OR the pack Conventions.normalMapGreenFlip.
    pub normal_flags: u32,
    /// Material.normalScale (tangent xy multiplier; default 1.0).
    pub normal_scale: f32,
    pub _pad2: u32,
    // ---- #6 Detail maps: adds 80 bytes (80 -> 160). All zero for the 4436 non-detail materials
    //      (detail_flags==0 AND flags lacks MAT_FLAG_DETAIL -> the shader's detail path is fully
    //      skipped -> those materials render byte-identical). The detail albedo/normal textures are
    //      appended to the SAME bindless `albedo_tex` / `normal_tex` arrays the base textures use;
    //      these indices point into them. ----
    /// bindless `albedo_tex` index of the detail albedo PNG, or 0 when absent (bit0 gates use). @80
    pub detail_albedo_index: u32,
    /// bindless `normal_tex` index of the detail normal PNG, or 0 when absent (bit1 gates use). @84
    pub detail_normal_index: u32,
    /// detail sub-flags: bit0 = has detail albedo, bit1 = has detail normal. @88
    pub detail_flags: u32,
    pub _detpad: u32, // @92
    /// RAW _DetailAlbedoMap_ST (sx,sy,ox,oy). Shader derives the relative transform vs `uv_xform`. @96
    pub detail_albedo_uv: [f32; 4],
    /// RAW _DetailNormalMap_ST (sx,sy,ox,oy). @112
    pub detail_normal_uv: [f32; 4],
    /// x = albedo blend strength, y = detail normal scale, z = fade start (8 m), w = fade end (15 m). @128
    pub detail_params: [f32; 4],
    /// xyz = offline albedoMeanGain = mean(sample_linear × 4.5948); w = 1. Divisor for neutralize. @144
    pub detail_mean_gain: [f32; 4], // -> total 160 bytes (16-aligned)
}

// #6: compile-time guard that GpuMaterial stays byte-matched to the WGSL `MaterialGpu` (160 B, all
// vec4 lanes 16-aligned). A silent mismatch here would corrupt EVERY material's GPU record, so this
// is checked at `cargo check` time (const eval) rather than trusted by eye.
const _: () = assert!(std::mem::size_of::<GpuMaterial>() == 160);
const _: () = assert!(std::mem::align_of::<GpuMaterial>() == 4);

/// `GpuMaterial::albedo_index` sentinel: material has no albedo texture.
pub const NO_ALBEDO: u32 = 0xFFFF_FFFF;
/// `GpuMaterial::normal_index` sentinel: material has no normal map (Phase 2b).
pub const NO_NORMAL: u32 = 0xFFFF_FFFF;
/// `GpuMaterial::normal_flags` bit0: DirectX-convention normal (green points down) -> negate n.y.
pub const MAT_NORMAL_FLAG_GREEN_FLIP: u32 = 1 << 0;
/// `GpuMaterial::flags` bit: cutout (alpha-test discard).
pub const MAT_FLAG_CUTOUT: u32 = 1 << 0;
/// `GpuMaterial::flags` bit: BLEND transparency (role decal/glass/water or alphaMode=BLEND).
/// Drawn in the P2 blend specialization (alpha blending, depth-write off); DISCARDED by the
/// P1 opaque specialization. Disjoint from CUTOUT (cutout stays opaque-pass). See M3b1.
pub const MAT_FLAG_BLEND: u32 = 1 << 1;
/// `GpuMaterial::flags` bit (M3b2): Vert-Paint SoftCutout road/track decal (Custom/Vert Paint
/// SoftCutout Decal — identified by the `vp.softCutout` param triple). BLEND-pass coverage =
/// COLOR_0.a modulated by `vp`, NOT tex.a. Feathers the decal into the terrain. Implies BLEND.
pub const MAT_FLAG_SOFTCUTOUT: u32 = 1 << 2;
/// `GpuMaterial::flags` bit (M3b2): water/mirror surface (role=="water"). BLEND-pass outputs a
/// translucent dark wet sheen instead of the white tint fallback (untextured water was WHITE).
/// Implies BLEND.
pub const MAT_FLAG_WATER: u32 = 1 << 3;
/// `GpuMaterial::flags` bit (#1 MicroSplat): terrain tile. The fragment ignores `albedo_index`
/// and instead splat-blends the 12 MicroSplat layers by the slice's 3 control maps. The slice
/// index (0..3) rides in `_pad2`.
pub const MAT_FLAG_TERRAIN: u32 = 1 << 4;
/// `GpuMaterial::flags` bit (#6 Detail maps): material carries a detail albedo and/or normal.
/// The fragment samples the detail texture(s) from the SAME bindless arrays, mean-neutralizes the
/// albedo, RNM-blends the normal, and distance-fades both. NEVER set together with MAT_FLAG_TERRAIN
/// (the terrain splat branch owns albedo/normal and must never enter the detail path).
pub const MAT_FLAG_DETAIL: u32 = 1 << 5;
/// `GpuMaterial::detail_flags` bit0: this material has a detail ALBEDO texture.
pub const DETAIL_FLAG_ALBEDO: u32 = 1 << 0;
/// `GpuMaterial::detail_flags` bit1: this material has a detail NORMAL texture.
pub const DETAIL_FLAG_NORMAL: u32 = 1 << 1;

/// MicroSplat splat table (group(2) binding(4), storage). All indices are into the SAME bindless
/// `albedo_tex` array as normal materials (the terrain textures are appended to `albedo_paths`).
/// Layer `i` weight = control map `i/4`, channel `i%4`. `layer_uv = terrainUV01 * rep` (the value
/// recovered from the MicroSplat material; NEVER `m_TileSize`). 144 bytes (12·4·3), 16-aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TerrainSplatGpu {
    /// bindless albedo index of each of the 12 layers.
    pub layer_albedo: [u32; 12],
    /// per-layer UV repeat (`terrainUV01 * rep`).
    pub layer_rep: [f32; 12],
    /// 4 slices × 3 control-map bindless indices: slice `s` map `k` at `[s*3 + k]`.
    pub ctrl_idx: [u32; 12],
}

// ---------------------------------------------------------------------------
// Phase 1 SH-GI: baked spherical-harmonics irradiance volume.
// ---------------------------------------------------------------------------

/// group(3) @binding(0) uniform. 64 bytes (16-aligned, four vec4s). Maps a world position
/// into the probe grid, carries the GI intensity + normal-bias, and (for the manual 8-tap
/// leak fix) the probe grid dims + spacing. Byte-identical to the WGSL `ShVolume`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ShVolumeUniform {
    /// xyz = world-space min corner of the probe AABB, w = gi_intensity (default 1.0).
    pub vol_min: [f32; 4],
    /// xyz = 1/(max-min) (world -> [0,1] uvw, hardware-trilinear fallback path),
    /// w = normal_bias in meters (default 0.75) for the manual 8-tap.
    pub vol_inv_extent: [f32; 4],
    /// xyz = (nx, ny, nz) probe grid dims (as f32), w unused.
    pub dims: [f32; 4],
    /// xyz = (sx, sy, sz) probe spacing in meters, w unused.
    pub spacing: [f32; 4],
}

/// Default normal-bias (meters) written to `ShVolumeUniform::vol_inv_extent.w`: the shading
/// point is pushed this far along the surface normal before sampling the probe grid, so a
/// point sitting on a slab doesn't sample the dark "inside-solid" probe directly beneath it.
const SH_NORMAL_BIAS: f32 = 0.75;

// ---------------------------------------------------------------------------
// #5 Dynamic sun shadows — 2-cascade near-field contact CSM.
// ---------------------------------------------------------------------------
// A near-field, sun-aligned contact shadow map. The SH volume already bakes the BROAD sun shadow,
// so this only adds the missing high-frequency contact edge and is combined in the shader under a
// hard cap (anti double-darkening). Rendered into a 2-layer Depth32Float array by reusing the
// camera-culled `visible[]`/`indirect` stream READ-ONLY (never re-culls it). All shadow work is a
// strict no-op when the feature is disabled (sun_dir missing or not EFT_SHADOWS=1): `enabled=0` in the
// uniform, and the depth array — always allocated so the group(3) layout stays stable — is ignored.

/// Shadow-map resolution per cascade (square). 2048² * 2 layers * 4 bytes = 32 MiB.
const SHADOW_MAP_SIZE: u32 = 2048;
/// Cascade count (2 near cascades). The depth array has this many layers.
const SHADOW_CASCADES: usize = 2;
/// Practical/log split distances (metres): cascade i covers [SHADOW_SPLITS[i], SHADOW_SPLITS[i+1]].
const SHADOW_SPLITS: [f32; SHADOW_CASCADES + 1] = [0.5, 15.0, 80.0];
/// Cascade overlap fraction (reported in the uniform; the shader blends 13.5..15 m).
const SHADOW_CASCADE_OVERLAP: f32 = 0.10;
/// How far a caster may sit toward the sun and still project into the slice (light-space Z fit).
const SHADOW_CASTER_EXTRUDE: f32 = 80.0;
/// Receiver-side margin pulled away from the sun in the light-space Z fit.
const SHADOW_RECEIVER_MARGIN: f32 = 10.0;
/// Max fraction of REMOVABLE (above-floor) baked diffuse the contact term may subtract. Hard-capped.
const SHADOW_DIFFUSE_CAP: f32 = 0.12;
/// Far contact fade band (metres): the whole shadow effect fades to fully lit across this range.
const SHADOW_FADE_START: f32 = 65.0;
const SHADOW_FADE_END: f32 = 80.0;

/// group(1) per-cascade uniform for the shadow depth pass. Byte-identical to the WGSL
/// `ShadowCascadeUniform` (80 bytes, 16-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ShadowCascadeUniform {
    /// world -> sun light clip (conventional 0..1-depth ortho). Column-major Mat4 upload.
    view_proj: [[f32; 4]; 4],
    /// xyz = Lsun (toward the sun), w = 1/SHADOW_MAP_SIZE (PCF texel).
    dir_texel: [f32; 4],
}

/// group(3) binding(5) main sun-shadow uniform read by gpu_draw.wgsl. Byte-identical to the WGSL
/// `SunShadowUniform` (192 bytes: 2×64 + 4×16).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SunShadowUniform {
    /// Per-cascade world->light-clip matrices (column-major).
    view_proj: [[[f32; 4]; 4]; SHADOW_CASCADES],
    /// x = far0 (15), y = far1 (80), z = overlap (0.10), w = enabled (1/0).
    split_depths: [f32; 4],
    /// xyz = Lsun (toward the sun), w = 1/SHADOW_MAP_SIZE (PCF texel).
    sun_dir_texel: [f32; 4],
    /// x = cascade0 world texel, y = cascade1 world texel (world-space bias units), z,w reserved.
    texel_world: [f32; 4],
    /// x = diffuse cap (0.12), y = fade start (65), z = fade end (80), w = debug mode (1 = spec-only).
    combine: [f32; 4],
}

/// Runtime shadow feature switch + the pack's sun direction (already X-flipped into pack space).
/// `enabled=false` (missing sun_dir or `not EFT_SHADOWS=1`) makes the whole pass a no-op.
#[derive(Resource)]
struct EftShadowConfig {
    /// Lsun: points TOWARD the sun (light travels along -Lsun). Unit. Y-up sentinel when disabled.
    lsun: Vec3,
    enabled: bool,
    /// `EFT_SHADOW_DEBUG=1`: specular-only diagnostic (diffuse cap forced to 0 in the shader).
    debug: bool,
}

/// The queued shadow depth pipeline + its group(1) cascade-uniform layout.
#[derive(Resource)]
struct EftShadowPipeline {
    pipeline_id: CachedRenderPipelineId,
    #[allow(dead_code)] // kept for symmetry / potential rebuilds; the bind groups already own it
    cascade_layout: BindGroupLayout,
}

/// Owns the shadow GPU resources so the depth views + uniforms outlive their bind groups.
#[derive(Resource)]
struct EftShadowResources {
    #[allow(dead_code)] // kept alive so all the views stay valid
    depth_texture: Texture,
    #[allow(dead_code)] // D2Array sampling view — bound in the main draw's group(3) binding(6)
    array_view: TextureView,
    /// One D2 render view per cascade layer (the shadow node's depth attachment).
    layer_views: [TextureView; SHADOW_CASCADES],
    /// Per-cascade group(1) uniform buffers (world->light-clip), rewritten each frame.
    cascade_uniforms: [Buffer; SHADOW_CASCADES],
    /// Per-cascade group(1) bind groups over `cascade_uniforms`.
    cascade_bind_groups: [BindGroup; SHADOW_CASCADES],
    /// The main SunShadowUniform (bound in the main draw's group(3) binding(5)), rewritten each frame.
    main_uniform: Buffer,
    #[allow(dead_code)] // comparison sampler — bound in the main draw's group(3) binding(7)
    comparison_sampler: Sampler,
}

/// volume.json layout descriptor (read at load; NEVER hardcoded — the emitter is authority).
#[derive(Debug, Clone, Deserialize)]
struct VolumeMeta {
    min: [f32; 3],
    max: [f32; 3],
    /// [nx, ny, nz] probe grid dims.
    dims: [u32; 3],
    /// [sx, sy, sz] probe spacing (meters). Emitter authority; if the sidecar omits it we
    /// derive it from (max-min)/(dims-1) so the manual 8-tap still has a valid grid step.
    #[serde(default)]
    spacing: Option<[f32; 3]>,
    coeffs: u32,
    channels: u32,
}

/// CPU-staged SH irradiance volume, ready for a ONE-TIME GPU upload as three RGBA16Float 3D
/// textures (one per color channel). `tex_{r,g,b}` are the raw f16 LE bytes already shuffled
/// into per-channel texel order (c0,c1,c2,c3), so the render world just `write_texture`s them.
/// Rides in `CpuData` (Arc-extracted, then freed with the rest of the staging blob).
struct ShVolumeCpu {
    /// [nx, ny, nz].
    dims: [u32; 3],
    min: [f32; 3],
    max: [f32; 3],
    /// [sx, sy, sz] probe spacing (meters) — for the manual 8-tap leak-fix grid step.
    spacing: [f32; 3],
    tex_r: Vec<u8>,
    tex_g: Vec<u8>,
    tex_b: Vec<u8>,
}

impl ShVolumeCpu {
    /// 1x1x1 fallback used when the pack ships no volume sidecar: c0 = 1.0 (half), c1..c3 = 0,
    /// so E/π = 0.282095 -> a flat ~0.28 gray ambient (roughly the old `ambient` constant),
    /// keeping group(3) valid rather than crashing the draw on a missing bind group.
    fn dummy() -> Self {
        // half(1.0) = 0x3C00, half(0.0) = 0x0000 (LE bytes). texel = (c0=1, c1=0, c2=0, c3=0).
        let texel: [u8; 8] = [0x00, 0x3C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        Self {
            dims: [1, 1, 1],
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
            spacing: [1.0, 1.0, 1.0], // single probe: grid clamps to 0, any nonzero step is inert
            tex_r: texel.to_vec(),
            tex_g: texel.to_vec(),
            tex_b: texel.to_vec(),
        }
    }
}

/// Load + repack the SH irradiance volume from the pack's `volume`/`volumeMeta` sidecars.
/// Returns `None` (caller falls back to `ShVolumeCpu::dummy`) on any missing/invalid input.
///
/// volume.bin is float16 LE, probe-major: probe index pi = ((z*ny)+y)*nx + x, each probe = 12
/// halfs [c0.r,c0.g,c0.b, c1.r..c3.b]. We shuffle into 3 per-channel buffers whose texel is
/// (c0,c1,c2,c3) for that channel — hardware trilinear then interpolates each SH coeff across
/// probes for free (correct: SH interpolates linearly). No float conversion: just move the
/// 2-byte halfs. Probe order (x-fastest -> y -> z) == wgpu 3D texel order, so pi -> texel copies.
fn load_sh_volume(pack: &Pack) -> Option<ShVolumeCpu> {
    let meta_path = pack.manifest.sidecars.volume_meta.as_deref()?;
    let bin_path = pack.manifest.sidecars.volume.as_deref()?;

    let meta_str = match std::fs::read_to_string(meta_path) {
        Ok(s) => s,
        Err(e) => {
            warn!("SH-GI: volume.json '{meta_path}' unreadable ({e}); flat-ambient fallback");
            return None;
        }
    };
    let meta: VolumeMeta = match serde_json::from_str(&meta_str) {
        Ok(m) => m,
        Err(e) => {
            warn!("SH-GI: volume.json '{meta_path}' parse failed ({e}); flat-ambient fallback");
            return None;
        }
    };
    if meta.coeffs != 4 || meta.channels != 3 {
        warn!(
            "SH-GI: unsupported volume (coeffs={}, channels={}; expected 4/3); fallback",
            meta.coeffs, meta.channels
        );
        return None;
    }
    let [nx, ny, nz] = meta.dims;
    let n_probes = nx as usize * ny as usize * nz as usize;
    if n_probes == 0 {
        warn!("SH-GI: volume dims {:?} degenerate; fallback", meta.dims);
        return None;
    }

    let bin = match std::fs::read(bin_path) {
        Ok(b) => b,
        Err(e) => {
            warn!("SH-GI: volume.bin '{bin_path}' unreadable ({e}); flat-ambient fallback");
            return None;
        }
    };
    // 12 halfs * 2 bytes = 24 bytes/probe.
    let need = n_probes * 24;
    if bin.len() < need {
        warn!(
            "SH-GI: volume.bin '{bin_path}' too short ({} bytes, need {}); fallback",
            bin.len(),
            need
        );
        return None;
    }

    // Per-channel texel = (c0,c1,c2,c3); each coeff is one f16 (2 bytes). Source half indices:
    //   R: 0,3,6,9   G: 1,4,7,10   B: 2,5,8,11
    let mut tex_r = Vec::with_capacity(n_probes * 8);
    let mut tex_g = Vec::with_capacity(n_probes * 8);
    let mut tex_b = Vec::with_capacity(n_probes * 8);
    let copy_half = |dst: &mut Vec<u8>, base: usize, h: usize| {
        let o = base + h * 2;
        dst.extend_from_slice(&bin[o..o + 2]);
    };
    for pi in 0..n_probes {
        let base = pi * 24;
        for &h in &[0usize, 3, 6, 9] {
            copy_half(&mut tex_r, base, h);
        }
        for &h in &[1usize, 4, 7, 10] {
            copy_half(&mut tex_g, base, h);
        }
        for &h in &[2usize, 5, 8, 11] {
            copy_half(&mut tex_b, base, h);
        }
    }

    // Probe spacing (meters) for the manual 8-tap leak fix. Prefer the emitter's authored
    // `spacing`; if the sidecar omits it, derive it from (max-min)/(dims-1) (probe i sits at
    // min + i*spacing, so a dim of 1 falls back to the full extent to avoid a divide-by-zero).
    let derive_spacing = |axis: usize| -> f32 {
        let extent = meta.max[axis] - meta.min[axis];
        let d = meta.dims[axis];
        if d > 1 {
            extent / (d - 1) as f32
        } else {
            extent.max(1e-6)
        }
    };
    let spacing = match meta.spacing {
        Some(s) => s,
        None => [derive_spacing(0), derive_spacing(1), derive_spacing(2)],
    };

    info!(
        "SH-GI: loaded irradiance volume {}x{}x{} ({} probes, {:.1} MB) min={:?} max={:?} spacing={:?}",
        nx,
        ny,
        nz,
        n_probes,
        need as f32 / (1024.0 * 1024.0),
        meta.min,
        meta.max,
        spacing
    );
    Some(ShVolumeCpu {
        dims: meta.dims,
        min: meta.min,
        max: meta.max,
        spacing,
        tex_r,
        tex_g,
        tex_b,
    })
}

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
    /// Phase 2b: unique normal-map texture paths in bindless-array-index order.
    /// `GpuMaterial.normal_index` indexes THIS list. Built in the SAME pass as `materials`.
    normal_paths: Vec<String>,
    /// Phase 1 SH-GI: the baked irradiance volume, repacked into per-channel f16 texel buffers.
    /// `None` if the pack shipped no volume sidecar (render world synthesizes a flat-ambient
    /// dummy so group(3) stays valid).
    sh_volume: Option<ShVolumeCpu>,
    /// #1 MicroSplat: the terrain splat table (layer/control bindless indices + per-layer rep).
    terrain: TerrainSplatGpu,
    /// #5 shadows: sun direction (points TOWARD the sun) X-flipped into pack space, or `None` when
    /// the volume sidecar has no valid `sun_dir` (the shadow feature then disables itself; no
    /// invented fallback direction). Mirrors standard.rs's exact access + flip.
    sun_dir: Option<Vec3>,
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
                    // #5 shadows: fit + upload the cascade matrices AFTER the buffers exist (the
                    // shadow resources are built in prepare_gpu_buffers).
                    prepare_shadow_uniforms
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    queue_gpu_driven.in_set(RenderSystems::QueueMeshes),
                ),
            )
            // #5: EftCull (writes visible/indirect) -> EftShadow (reads them, writes the depth
            // atlas) -> StartMainPass (main draw samples the atlas). The shadow node NEVER re-culls
            // or resets the shared stream.
            .add_render_graph_node::<EftCullNode>(Core3d, EftCullLabel)
            .add_render_graph_node::<EftShadowNode>(Core3d, EftShadowLabel)
            .add_render_graph_edges(
                Core3d,
                (EftCullLabel, EftShadowLabel, Node3d::StartMainPass),
            );
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

/// Extract the Vert-Paint SoftCutout params `[_AlphaStrength, _Cutoff, _AlphaHeight, 0]` from a
/// material's `vp` block. Returns `Some` ONLY for the Custom/Vert Paint SoftCutout Decal family
/// — identified by the `vp.softCutout` triple being present (there is no separate shader-name
/// field; this param IS the shader signature). Returns `None` for plain vert-paint-solid (vp
/// with NO softCutout), for water, and for every non-vp material.
fn softcutout_params(vp: &Option<crate::eftpack::VertPaint>) -> Option<[f32; 4]> {
    let arr = vp.as_ref()?.get("softCutout")?.as_array()?;
    if arr.len() < 3 {
        return None;
    }
    Some([
        arr[0].as_f64()? as f32,
        arr[1].as_f64()? as f32,
        arr[2].as_f64()? as f32,
        0.0,
    ])
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
    // Phase 2b: dedup normal-map paths in the SAME pass (bindless index consistency, like albedo).
    let mut normal_paths: Vec<String> = Vec::new();
    let mut normal_path_to_index: HashMap<String, u32> = HashMap::new();
    // Pack-wide green-flip convention (DirectX Y-down): OR'd with each material's own flag.
    let conv_green_flip = pack.manifest.conventions.normal_map_green_flip;
    for mat in &pack.materials {
        let albedo_index = match mat.albedo.as_deref() {
            Some(p) if !p.is_empty() => *path_to_index.entry(p.to_string()).or_insert_with(|| {
                let idx = albedo_paths.len() as u32;
                albedo_paths.push(p.to_string());
                idx
            }),
            _ => NO_ALBEDO,
        };
        // Phase 2b: bindless normal-map index (dedup first-seen, mirrors albedo). null -> sentinel.
        let normal_index = match mat.normal.as_deref() {
            Some(p) if !p.is_empty() => {
                *normal_path_to_index.entry(p.to_string()).or_insert_with(|| {
                    let idx = normal_paths.len() as u32;
                    normal_paths.push(p.to_string());
                    idx
                })
            }
            _ => NO_NORMAL,
        };
        // normal_flags bit0 = green-flip: the material's own flag OR the pack-wide convention.
        let mut normal_flags = 0u32;
        if mat.normal_green_flip || conv_green_flip {
            normal_flags |= MAT_NORMAL_FLAG_GREEN_FLIP;
        }
        // Material class flags. CUTOUT (role=cutout / alphaMode=MASK) -> alpha-test discard,
        // stays in the OPAQUE (P1) pass. BLEND (M3b1: role decal/glass/water OR alphaMode=BLEND)
        // -> the P2 alpha-blended pass (depth-write off). The two bits are disjoint: the P1
        // opaque specialization discards BLEND, the P2 blend specialization discards non-BLEND,
        // so a material authored as both cutout+blend would only ever draw in P2.
        let mut flags = 0u32;
        if mat.role == "cutout" || mat.alpha_mode == "MASK" {
            flags |= MAT_FLAG_CUTOUT;
        }
        if mat.role == "decal"
            || mat.role == "glass"
            || mat.role == "water"
            || mat.alpha_mode == "BLEND"
        {
            flags |= MAT_FLAG_BLEND;
        }
        // M3b2 SoftCutout / water classification. The Vert-Paint SoftCutout family (Custom/Vert
        // Paint SoftCutout Decal) is identified by the `vp.softCutout` param triple — its BLEND
        // coverage is COLOR_0.a modulated by these params, NOT tex.a (which is smoothness here).
        // Water/mirror surfaces (role=="water") had (mostly) no usable albedo and fell back to a
        // flat WHITE tint; they get a dark wet sheen instead. Both classes ALSO blend (force
        // MAT_FLAG_BLEND even for the 16 SoftCutout materials the extractor marked OPAQUE, so
        // they feather in the P2 pass instead of hard-slabbing in P1).
        let vp_params = softcutout_params(&mat.vp);
        if vp_params.is_some() {
            flags |= MAT_FLAG_SOFTCUTOUT | MAT_FLAG_BLEND;
        }
        if mat.role == "water" {
            flags |= MAT_FLAG_WATER | MAT_FLAG_BLEND;
        }
        // #6 Detail maps: resolve the (optional) detail albedo + normal into the SAME bindless
        // arrays the base textures use — dedup by path via the SAME first-seen maps as the base
        // textures, so the 2 shared detail textures (one albedo, one normal, reused across all 23
        // rock materials) append only 2 entries total and their indices can never drift. Albedo and
        // normal are independent (either may be present); detail_flags gates each half. Terrain
        // materials are excluded (they're tagged AFTER this loop, and we clear detail there too).
        let mut detail_albedo_index = 0u32;
        let mut detail_normal_index = 0u32;
        let mut detail_flags = 0u32;
        let mut detail_albedo_uv = [0.0f32; 4];
        let mut detail_normal_uv = [0.0f32; 4];
        let mut detail_params = [0.0f32; 4];
        let mut detail_mean_gain = [0.0f32; 4];
        if let Some(det) = &mat.detail {
            if let Some(p) = det.albedo.as_deref().filter(|p| !p.is_empty()) {
                detail_albedo_index = *path_to_index.entry(p.to_string()).or_insert_with(|| {
                    let idx = albedo_paths.len() as u32;
                    albedo_paths.push(p.to_string());
                    idx
                });
                detail_flags |= DETAIL_FLAG_ALBEDO;
                detail_albedo_uv = det.albedo_uv;
            }
            if let Some(p) = det.normal.as_deref().filter(|p| !p.is_empty()) {
                detail_normal_index =
                    *normal_path_to_index.entry(p.to_string()).or_insert_with(|| {
                        let idx = normal_paths.len() as u32;
                        normal_paths.push(p.to_string());
                        idx
                    });
                detail_flags |= DETAIL_FLAG_NORMAL;
                detail_normal_uv = det.normal_uv;
            }
            if detail_flags != 0 {
                flags |= MAT_FLAG_DETAIL;
                // detail_params: [albedoStrength, normalScale, fade_start, fade_end]. The fade window
                // is env-tunable (EFT_DETAIL_FADE="near,far", default 8,15 m) so the detail range can
                // be verified/tuned without a rebuild — the default camera sits ~15 m out, at which
                // the shipping 8-15 m window has already faded detail to ~0.
                let (fnear, ffar) = std::env::var("EFT_DETAIL_FADE")
                    .ok()
                    .and_then(|s| {
                        let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                        (v.len() == 2).then(|| (v[0], v[1]))
                    })
                    .unwrap_or((8.0, 25.0)); // wider than the web's 8-15 m: this viewer orbits farther out
                detail_params = [det.albedo_strength, det.normal_scale, fnear, ffar];
                // mean-neutralize divisor (offline mean of linear×4.5948); w=1 (unused lane).
                detail_mean_gain = [
                    det.albedo_mean_gain[0],
                    det.albedo_mean_gain[1],
                    det.albedo_mean_gain[2],
                    1.0,
                ];
            }
        }
        materials_gpu.push(GpuMaterial {
            albedo_index,
            flags,
            alpha_cutoff: mat.alpha_cutoff,
            // Phase 1.6 GGX spec: per-material roughness (was _pad). Glass ships ~0.05 (sharp);
            // default 0.55 for unspecified. Clamp [0.03,1.0] so the NDF can't blow up / go mirror-hard.
            roughness: mat.roughness.unwrap_or(0.55).clamp(0.03, 1.0),
            uv_xform: mat.uv_xform, // reference only (uvTilingBaked=true); shader must NOT apply
            tint: mat.tint,
            vp: vp_params.unwrap_or([0.0; 4]),
            // Phase 2b normal mapping.
            normal_index,
            normal_flags,
            normal_scale: mat.normal_scale,
            _pad2: 0,
            // #6 Detail maps (zeros unless MAT_FLAG_DETAIL was set above).
            detail_albedo_index,
            detail_normal_index,
            detail_flags,
            _detpad: 0,
            detail_albedo_uv,
            detail_normal_uv,
            detail_params,
            detail_mean_gain,
        });
    }

    // ---- #1 MicroSplat terrain: append the 12 layer + 12 control textures to the SAME bindless
    //      albedo set, build the splat table, and tag the 4 terrain materials (FLAG_TERRAIN +
    //      slice index in _pad2, matte roughness). Layer i weight = control(i/4).chan(i%4);
    //      layer_uv = terrainUV01*rep (the recovered MicroSplat tiling; NEVER m_TileSize). ----
    let mut terrain = TerrainSplatGpu {
        layer_albedo: [0; 12],
        layer_rep: [1.0; 12],
        ctrl_idx: [0; 12],
    };
    'terrain: {
        let Some(tl_path) = pack.manifest.sidecars.terrain_layers.as_deref() else {
            warn!("gpu-driven terrain: no terrainLayers sidecar — terrain stays single-layer");
            break 'terrain;
        };
        let dir = std::path::Path::new(tl_path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let tl: serde_json::Value = match std::fs::read_to_string(tl_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
        {
            Some(v) => v,
            None => {
                warn!("gpu-driven terrain: could not read/parse {tl_path}");
                break 'terrain;
            }
        };
        let Some(tiles) = tl.get("tiles").and_then(|v| v.as_object()) else {
            break 'terrain;
        };
        // append a terrain texture (filename relative to the sidecar dir) to the bindless set.
        let mut add_tex = |name: &str| -> u32 {
            let full = dir.join(name).to_string_lossy().replace('\\', "/");
            *path_to_index.entry(full.clone()).or_insert_with(|| {
                let idx = albedo_paths.len() as u32;
                albedo_paths.push(full);
                idx
            })
        };
        const SLICES: [&str; 4] = ["Slice_1_1", "Slice_1_2", "Slice_2_1", "Slice_2_2"];
        let mut layers_done = false;
        for (si, sname) in SLICES.iter().enumerate() {
            let Some(tile) = tiles.get(*sname) else { continue };
            if let Some(cm) = tile.get("ctrl_maps").and_then(|v| v.as_array()) {
                for (k, c) in cm.iter().take(3).enumerate() {
                    if let Some(cn) = c.as_str() {
                        terrain.ctrl_idx[si * 3 + k] = add_tex(cn);
                    }
                }
            }
            // The 12 layers are shared across slices (same MicroSplat material); capture once.
            if !layers_done {
                if let Some(layers) = tile.get("layers").and_then(|v| v.as_array()) {
                    for l in layers {
                        let idx = l.get("idx").and_then(|v| v.as_u64()).unwrap_or(99) as usize;
                        if idx >= 12 {
                            continue;
                        }
                        let name = l.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let rep = l.get("rep").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                        terrain.layer_albedo[idx] = add_tex(&format!("layer_{name}.png"));
                        terrain.layer_rep[idx] = rep;
                    }
                    layers_done = true;
                }
            }
        }
        // Tag the 4 terrain materials: FLAG_TERRAIN + slice index (0..3) in _pad2, matte roughness.
        let mut tagged = 0u32;
        for inst in &pack.instances {
            if inst.flags & crate::eftpack::flags::TERRAIN == 0 {
                continue;
            }
            let me = &pack.manifest.meshes[inst.mesh_id as usize];
            let Some(slice) = SLICES.iter().position(|s| me.name.contains(s)) else {
                continue;
            };
            let Some(sub) = me.submeshes.first() else { continue };
            let mid = sub.material_id as usize;
            if mid < materials_gpu.len() {
                materials_gpu[mid].flags |= MAT_FLAG_TERRAIN;
                // #6: terrain owns albedo/normal via the splat branch — it must NEVER enter the
                // detail path. Clear any detail a terrain material might have carried (defensive;
                // no known terrain material has a `detail` object).
                materials_gpu[mid].flags &= !MAT_FLAG_DETAIL;
                materials_gpu[mid].detail_flags = 0;
                materials_gpu[mid]._pad2 = slice as u32;
                materials_gpu[mid].roughness = 0.95; // matte ground, no shiny slab
                tagged += 1;
            }
        }
        info!(
            "gpu-driven #1 terrain: MicroSplat table built (12 layers × 4 slices, {tagged} tiles tagged)"
        );
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
    info!(
        "gpu-driven Phase2b: {} unique normal-map textures ({} materials with no normal map)",
        normal_paths.len(),
        materials_gpu
            .iter()
            .filter(|m| m.normal_index == NO_NORMAL)
            .count(),
    );
    info!(
        "gpu-driven M3b2: {} SoftCutout (feathered road/track) + {} water materials",
        materials_gpu
            .iter()
            .filter(|m| m.flags & MAT_FLAG_SOFTCUTOUT != 0)
            .count(),
        materials_gpu
            .iter()
            .filter(|m| m.flags & MAT_FLAG_WATER != 0)
            .count(),
    );
    info!(
        "gpu-driven #6 detail: {} materials tagged ({} with detail albedo, {} with detail normal)",
        materials_gpu
            .iter()
            .filter(|m| m.flags & MAT_FLAG_DETAIL != 0)
            .count(),
        materials_gpu
            .iter()
            .filter(|m| m.detail_flags & DETAIL_FLAG_ALBEDO != 0)
            .count(),
        materials_gpu
            .iter()
            .filter(|m| m.detail_flags & DETAIL_FLAG_NORMAL != 0)
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
            // M3b2: per-vertex COLOR_0 vert-paint weight. Every mesh in this pack carries a
            // color attr (unorm8x4 @32) so geom.colors is populated; default opaque-white for
            // any mesh that lacks it (color.a=1 -> SoftCutout coverage stays fully covered).
            let col = *geom.colors.get(k).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
            vertex_data.extend_from_slice(&[
                p[0], p[1], p[2],
                nrm[0], nrm[1], nrm[2],
                uv[0], uv[1],
                f32::from_bits(vert_mat[k]), // material_index (read as Uint32 on the GPU)
                col[0], col[1], col[2], col[3], // color f32x4 @36 (interpolated in the shader)
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

    // ---- #4 GRASS: append the density-placed grass clumps as a cross-quad mesh + N instances,
    //      rendered by the SAME cull + multidraw + alpha-cutout path. grass.bin = N×[x,y,z,rotY,
    //      scale] f32 from build_grass.py (deterministic, road-excluding GPU-Instancer density). ----
    'grass: {
        let bin = match std::fs::read(pack.root.join("grass.bin")) {
            Ok(b) if !b.is_empty() => b,
            _ => {
                info!("gpu-driven grass: no grass.bin (run build_grass.py) — skipping grass");
                break 'grass;
            }
        };
        // grass albedo + tint from the sidecar.
        let side = std::fs::read_to_string(pack.root.join("grass_sidecar.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let grass_albedo = side
            .as_ref()
            .and_then(|v| v.get("albedo").and_then(|a| a.as_str()))
            .unwrap_or("")
            .to_string();
        if grass_albedo.is_empty() {
            warn!("gpu-driven grass: no grass albedo in sidecar — skipping");
            break 'grass;
        }
        let grass_tint = side
            .as_ref()
            .and_then(|v| v.get("tint").and_then(|a| a.as_array()))
            .map(|a| {
                let g = |i: usize, d: f32| a.get(i).and_then(|x| x.as_f64()).unwrap_or(d as f64) as f32;
                [g(0, 0.7), g(1, 0.75), g(2, 0.55), 1.0]
            })
            .unwrap_or([0.7, 0.75, 0.55, 1.0]);

        // Grass material: alpha-cutout (blade coverage in the texture alpha), matte.
        let grass_albedo_idx = *path_to_index.entry(grass_albedo.clone()).or_insert_with(|| {
            let idx = albedo_paths.len() as u32;
            albedo_paths.push(grass_albedo.clone());
            idx
        });
        let grass_mat_id = materials_gpu.len() as u32;
        materials_gpu.push(GpuMaterial {
            albedo_index: grass_albedo_idx,
            flags: MAT_FLAG_CUTOUT,
            alpha_cutoff: 0.35,
            roughness: 0.9,
            uv_xform: [1.0, 1.0, 0.0, 0.0],
            tint: grass_tint,
            vp: [0.0; 4],
            normal_index: NO_NORMAL,
            normal_flags: 0,
            normal_scale: 1.0,
            _pad2: 0,
            // #6: grass carries no detail map.
            ..GpuMaterial::default()
        });

        // Cross-quad clump mesh: 3 quads at 0/60/120° around Y, base at y=0.
        let base_vertex = vtx_cursor as i32;
        let first_index = idx_cursor;
        let mbits = f32::from_bits(grass_mat_id);
        let (hw, gh) = (0.42f32, 0.9f32);
        let (mut nverts, mut nidx) = (0u32, 0u32);
        for q in 0..3u32 {
            let ang = q as f32 * std::f32::consts::PI / 3.0;
            let (s, c) = ang.sin_cos();
            let (dx, dz) = (c * hw, s * hw);
            let b = nverts;
            let mk = |x: f32, y: f32, z: f32, u: f32, v: f32| {
                [x, y, z, 0.0, 1.0, 0.0, u, v, mbits, 1.0, 1.0, 1.0, 1.0]
            };
            for vtx in [
                mk(-dx, 0.0, -dz, 0.0, 1.0),
                mk(dx, 0.0, dz, 1.0, 1.0),
                mk(dx, gh, dz, 1.0, 0.0),
                mk(-dx, gh, -dz, 0.0, 0.0),
            ] {
                vertex_data.extend_from_slice(&vtx);
            }
            index_data.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
            nverts += 4;
            nidx += 6;
        }
        vtx_cursor += nverts;
        idx_cursor += nidx;

        // One instance per grass clump (deterministic transform from grass.bin).
        let instance_base = inst_cursor;
        let mut count = 0u32;
        for ch in bin.chunks_exact(20) {
            let f = |o: usize| f32::from_le_bytes([ch[o], ch[o + 1], ch[o + 2], ch[o + 3]]);
            let (x, y, z, rot, sc) = (f(0), f(4), f(8), f(12), f(16));
            let (s, c) = rot.sin_cos();
            instances.push(InstanceGpuRecord {
                m0: [c * sc, 0.0, s * sc, x],
                m1: [0.0, sc, 0.0, y],
                m2: [-s * sc, 0.0, c * sc, z],
                ids: [mesh_meta.len() as u32, 0, 0, 0],
                sphere: [x, y + gh * sc * 0.5, z, 1.3 * sc],
            });
            count += 1;
        }
        inst_cursor += count;
        mesh_meta.push(MeshMeta {
            index_count: nidx,
            first_index,
            base_vertex,
            instance_base,
            instance_count: count,
            _pad: [0; 3],
        });
        info!("gpu-driven #4 grass: {count} clumps appended (cross-quad, alpha-cutout)");
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

    // Phase 1 SH-GI: load + repack the baked irradiance volume (volume.bin + volume.json).
    let sh_volume = load_sh_volume(pack);

    // #5 shadows: source the sun direction from the SAME volume.json sidecar the SH bake used, with
    // the SAME X-flip standard.rs applies (Lsun = normalize(-raw.x, raw.y, raw.z), pointing TOWARD
    // the sun). `None` (missing/degenerate) => the shadow feature disables itself downstream.
    let sun_dir = pack
        .manifest
        .sidecars
        .volume_meta
        .as_deref()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|txt| serde_json::from_str::<serde_json::Value>(&txt).ok())
        .and_then(|v| {
            v.get("sun_dir").and_then(|s| s.as_array()).and_then(|a| {
                let raw = Vec3::new(
                    -(a.first()?.as_f64()? as f32),
                    a.get(1)?.as_f64()? as f32,
                    a.get(2)?.as_f64()? as f32,
                );
                (raw.length_squared() > 1e-6).then(|| raw.normalize())
            })
        });
    match sun_dir {
        Some(d) => info!("gpu-driven #5 shadows: sun_dir (pack space, X-flipped) = {d:?}"),
        None => info!("gpu-driven #5 shadows: no valid sun_dir in volume.json — shadows disabled"),
    }

    commands.insert_resource(ExtractedCpuData(Arc::new(CpuData {
        vertex_data,
        index_data,
        instances,
        mesh_meta,
        materials: materials_gpu,
        albedo_paths,
        normal_paths,
        sh_volume,
        terrain,
        sun_dir,
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
    /// #5 shadows: the depth-only shadow-caster shader (`gpu_shadow.wgsl`). Loaded at RenderStartup;
    /// the shadow render pipeline (which also needs the material_layout) is queued in
    /// `prepare_gpu_buffers` once that layout exists.
    shadow_shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
    ssbo_layout: BindGroupLayout,
    /// group(2) bindless material layout: material-table SSBO + albedo `binding_array` +
    /// sampler. Built in `prepare_gpu_buffers` (needs the unique-albedo count for the
    /// `binding_array` size) and the pipeline is re-inserted with it set. `None` until then;
    /// `queue_gpu_driven` gates specialization on it being `Some` (M3).
    material_layout: Option<BindGroupLayout>,
    /// group(3) SH-GI layout: ShVolume uniform + 3 SH 3D textures + sampler (Phase 1). Shared by
    /// BOTH the opaque and BLEND specializations. Built in `prepare_gpu_buffers` alongside the
    /// material layout; `queue_gpu_driven` gates specialization on it being `Some`.
    sh_layout: Option<BindGroupLayout>,
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
    /// Phase 2b: bindless normal-map textures + views, kept alive alongside the albedo set.
    #[allow(dead_code)]
    normal_textures: Vec<Texture>,
    #[allow(dead_code)]
    normal_views: Vec<TextureView>,
    #[allow(dead_code)]
    sampler: Sampler,
}

#[derive(Resource)]
struct EftMaterialBindGroup(BindGroup);

/// Owns the Phase-1 SH-GI GPU resources so the 3D texture views + uniform outlive
/// `EftShBindGroup`. Built once in `prepare_gpu_buffers`.
#[derive(Resource)]
struct EftShResources {
    #[allow(dead_code)] // kept alive so the views/bind group stay valid
    uniform: Buffer,
    #[allow(dead_code)]
    textures: Vec<Texture>,
    #[allow(dead_code)]
    views: Vec<TextureView>,
    #[allow(dead_code)]
    sampler: Sampler,
}

#[derive(Resource)]
struct EftShBindGroup(BindGroup);

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
    let shadow_shader = asset_server.load("shaders/gpu_shadow.wgsl"); // #5 depth-only caster

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
        shadow_shader,
        mesh_pipeline: mesh_pipeline.clone(),
        ssbo_layout,
        material_layout: None, // filled in prepare_gpu_buffers once the albedo count is known
        sh_layout: None,       // filled in prepare_gpu_buffers alongside the material layout
    });
}

// ---- PrepareResources: build all GPU buffers + bind groups ONCE -------------
#[allow(clippy::too_many_arguments)]
fn prepare_gpu_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>, // #5 shadows: queue the shadow depth pipeline once here
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
    // #1 MicroSplat terrain splat table (group(2) binding(4)).
    let terrain_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_terrain_splat"),
        contents: bytemuck::bytes_of(&cpu.terrain),
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

    // Phase 2b: decode + upload every UNIQUE normal map, MIRRORING the albedo load but with a
    // LINEAR format (Rgba8Unorm) — normal maps are LINEAR data, NOT sRGB; the sRGB format would
    // gamma-wash the tangent vectors and flatten the perturbation. Same order as cpu.normal_paths
    // so GpuMaterial.normal_index stays aligned; a failed decode pushes a flat-normal placeholder.
    let mut normal_textures: Vec<Texture> = Vec::with_capacity(cpu.normal_paths.len());
    let mut normal_views: Vec<TextureView> = Vec::with_capacity(cpu.normal_paths.len());
    for path in &cpu.normal_paths {
        let (tex, view) = load_normal_texture(&render_device, &render_queue, path);
        normal_textures.push(tex);
        normal_views.push(view);
    }
    // binding_array needs >= 1 element; synth a 1x1 flat normal if this pack has no normal maps.
    if normal_views.is_empty() {
        let (tex, view) = make_dummy_normal_texture(&render_device, &render_queue);
        normal_textures.push(tex);
        normal_views.push(view);
    }
    let normal_count = normal_views.len() as u32;

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

    // group(2): material-table SSBO (0) + albedo binding_array size tex_count (1) + sampler (2)
    // + Phase 2b normal-map binding_array size normal_count (3). The normal array reuses the
    // albedo sampler and the same non-uniform-indexing device feature.
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
                (
                    3,
                    texture_2d(TextureSampleType::Float { filterable: true })
                        .count(NonZeroU32::new(normal_count).unwrap()),
                ),
                (4, storage_buffer_read_only_sized(false, None)), // #1 terrain splat table
            ),
        ),
    );

    // TextureViewArray wants raw &[&wgpu::TextureView]; Bevy's TextureView derefs to it.
    let view_refs: Vec<_> = views.iter().map(|v| &**v).collect();
    let normal_view_refs: Vec<_> = normal_views.iter().map(|v| &**v).collect();
    let material_bg = render_device.create_bind_group(
        "eft_material_bg",
        &material_layout,
        &BindGroupEntries::with_indices((
            (0, material_buf.as_entire_binding()),
            (1, &view_refs[..]),
            (2, &albedo_sampler),
            (3, &normal_view_refs[..]),
            (4, terrain_buf.as_entire_binding()),
        )),
    );

    // ---- Phase 1 SH-GI: 3 RGBA16Float 3D textures (one per color channel) + uniform ----------
    // Each texel = (c0,c1,c2,c3) for that channel; hardware trilinear interpolates each SH coeff
    // across probes for free. The fragment reconstructs diffuse irradiance per fragment. If the
    // pack shipped no volume sidecar, synthesize a 1x1x1 flat-ambient dummy so group(3) stays
    // valid (a missing bind group would fail the draw at validation).
    let dummy_sh;
    let sh: &ShVolumeCpu = match &cpu.sh_volume {
        Some(v) => v,
        None => {
            warn!("gpu-driven SH-GI: no volume sidecar; using 1x1x1 flat-ambient fallback");
            dummy_sh = ShVolumeCpu::dummy();
            &dummy_sh
        }
    };
    let [sh_nx, sh_ny, sh_nz] = sh.dims;
    let sh_extent = Extent3d {
        width: sh_nx,
        height: sh_ny,
        depth_or_array_layers: sh_nz,
    };
    // create_texture_with_data handles staging + row-padding; probe order (x-fastest -> y -> z)
    // is exactly wgpu 3D texel order, so the shuffled bytes upload as a direct copy.
    let make_sh_tex = |bytes: &[u8], label: &'static str| -> (Texture, TextureView) {
        let tex = render_device.create_texture_with_data(
            &render_queue,
            &TextureDescriptor {
                label: Some(label),
                size: sh_extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::default(),
            bytes,
        );
        let view = tex.create_view(&TextureViewDescriptor::default()); // infers D3 from the texture
        (tex, view)
    };
    let (sh_r_tex, sh_r_view) = make_sh_tex(&sh.tex_r, "eft_sh_r");
    let (sh_g_tex, sh_g_view) = make_sh_tex(&sh.tex_g, "eft_sh_g");
    let (sh_b_tex, sh_b_view) = make_sh_tex(&sh.tex_b, "eft_sh_b");

    let sh_sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("eft_sh_sampler"),
        // ClampToEdge: a fragment just outside the probe AABB reuses the boundary probe rather
        // than wrapping to the far side of the map.
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Nearest, // single-level (no mips)
        ..default()
    });

    let sh_inv_extent = [
        1.0 / (sh.max[0] - sh.min[0]).max(1e-6),
        1.0 / (sh.max[1] - sh.min[1]).max(1e-6),
        1.0 / (sh.max[2] - sh.min[2]).max(1e-6),
    ];
    let sh_uniform_data = ShVolumeUniform {
        vol_min: [sh.min[0], sh.min[1], sh.min[2], 1.0], // w = gi_intensity (default 1.0)
        // w = normal_bias (meters) for the manual 8-tap leak fix.
        vol_inv_extent: [sh_inv_extent[0], sh_inv_extent[1], sh_inv_extent[2], SH_NORMAL_BIAS],
        // xyz = probe grid dims (as f32), for the manual 8-tap corner enumeration.
        dims: [sh_nx as f32, sh_ny as f32, sh_nz as f32, 0.0],
        // xyz = probe spacing (meters); probe i sits at vol_min + i*spacing.
        spacing: [sh.spacing[0], sh.spacing[1], sh.spacing[2], 0.0],
    };
    let sh_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_sh_uniform"),
        contents: bytemuck::bytes_of(&sh_uniform_data),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    // ---- #5 Dynamic sun shadows: depth array + sampler + per-cascade uniforms + pipeline --------
    // Built BEFORE the group(3) layout/bind-group below because the main draw's group(3) samples the
    // shadow depth array (binding 6) + comparison sampler (binding 7) and reads the SunShadowUniform
    // (binding 5). Everything here is allocated unconditionally so the group(3) LAYOUT is stable
    // whether or not shadows are enabled; the runtime switch lives in the SunShadowUniform (enabled)
    // and `EftShadowConfig`.
    // #5 sun shadows are OPT-IN, default OFF: the baked SH volume already contains the sun's
    // static shadows, so the real-time contact term is a marginal add that still needs bias/gate
    // tuning. Enable explicitly with EFT_SHADOWS=1; otherwise the pass is a strict no-op.
    let shadows_env_on = std::env::var("EFT_SHADOWS")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let shadow_debug = std::env::var("EFT_SHADOW_DEBUG")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let (lsun, shadows_enabled) = match cpu.sun_dir {
        Some(d) if shadows_env_on => (d, true), // opt-in via EFT_SHADOWS=1
        Some(d) => (d, false),                  // sun present but not requested -> default OFF
        None => (Vec3::Y, false),               // no sun_dir -> disabled (Y-up sentinel; never sampled)
    };
    info!(
        "gpu-driven #5 shadows: enabled={shadows_enabled} debug={shadow_debug} Lsun={lsun:?} \
         (2 cascades, {sz}²×{n} Depth32Float; opt-in EFT_SHADOWS=1, diag EFT_SHADOW_DEBUG=1)",
        sz = SHADOW_MAP_SIZE,
        n = SHADOW_CASCADES,
    );

    // The 2-layer depth atlas. RENDER_ATTACHMENT (the shadow pass writes it) | TEXTURE_BINDING (the
    // main pass samples it). One D2Array sampling view + one D2 render view per layer.
    let shadow_depth = render_device.create_texture(&TextureDescriptor {
        label: Some("eft_shadow_depth"),
        size: Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: SHADOW_CASCADES as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_array_view = shadow_depth.create_view(&TextureViewDescriptor {
        label: Some("eft_shadow_array_view"),
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let shadow_layer_view = |layer: u32| {
        shadow_depth.create_view(&TextureViewDescriptor {
            label: Some("eft_shadow_layer_view"),
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..default()
        })
    };
    let shadow_layer_views: [TextureView; SHADOW_CASCADES] =
        [shadow_layer_view(0), shadow_layer_view(1)];

    // Comparison sampler: LessEqual (fragment lit when its light-space depth <= stored occluder).
    let shadow_cmp_sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("eft_shadow_cmp"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Nearest,
        compare: Some(CompareFunction::LessEqual),
        ..default()
    });

    // group(1) cascade-uniform layout for the shadow pipeline (vertex-stage world->light-clip).
    let cascade_layout = render_device.create_bind_group_layout(
        "eft_shadow_cascade_layout",
        &BindGroupLayoutEntries::single(ShaderStages::VERTEX, uniform_buffer_sized(false, None)),
    );
    // Two per-cascade uniform buffers (+ bind groups). Filled per frame by prepare_shadow_uniforms;
    // sized to the POD so the initial (zeroed) content is a valid, inert matrix until then.
    let make_cascade_uniform = || {
        render_device.create_buffer(&BufferDescriptor {
            label: Some("eft_shadow_cascade_uniform"),
            size: std::mem::size_of::<ShadowCascadeUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let cascade_uniforms: [Buffer; SHADOW_CASCADES] =
        [make_cascade_uniform(), make_cascade_uniform()];
    let cascade_bind_groups: [BindGroup; SHADOW_CASCADES] = [
        render_device.create_bind_group(
            "eft_shadow_cascade_bg0",
            &cascade_layout,
            &BindGroupEntries::single(cascade_uniforms[0].as_entire_binding()),
        ),
        render_device.create_bind_group(
            "eft_shadow_cascade_bg1",
            &cascade_layout,
            &BindGroupEntries::single(cascade_uniforms[1].as_entire_binding()),
        ),
    ];

    // The main SunShadowUniform (group(3) binding(5)). Initialize enabled=0 so the very first frame
    // — before prepare_shadow_uniforms runs — is a strict no-op; per-frame fill flips it on.
    let shadow_main_seed = SunShadowUniform {
        combine: [
            SHADOW_DIFFUSE_CAP,
            SHADOW_FADE_START,
            SHADOW_FADE_END,
            if shadow_debug { 1.0 } else { 0.0 },
        ],
        sun_dir_texel: [lsun.x, lsun.y, lsun.z, 1.0 / SHADOW_MAP_SIZE as f32],
        ..default()
    };
    let shadow_main_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_sun_shadow_uniform"),
        contents: bytemuck::bytes_of(&shadow_main_seed),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    // Queue the shadow depth pipeline: groups [ssbo(0), cascade(1), material(2)]; empty color target;
    // Depth32Float write + LessEqual; cull None (double-sided); raster bias 2 / slope 2.0.
    let shadow_pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("eft_shadow_depth".into()),
        layout: vec![
            draw.ssbo_layout.clone(),
            cascade_layout.clone(),
            material_layout.clone(),
        ],
        push_constant_ranges: vec![],
        vertex: VertexState {
            shader: draw.shadow_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            buffers: vec![VertexBufferLayout {
                array_stride: DRAW_VERTEX_STRIDE,
                step_mode: VertexStepMode::Vertex,
                // pos @0 (loc0), uv @24 (loc2), material @32 (loc3). normal/color are skipped.
                attributes: vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 24,
                        shader_location: 2,
                    },
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
            cull_mode: None, // double-sided casters, like the main pass
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            // Conventional 0..1 shadow depth (NOT the main pass's reverse-z GreaterEqual).
            depth_compare: CompareFunction::LessEqual,
            stencil: StencilState::default(),
            // Constant + slope-scaled raster bias to fight shadow acne (tuned by the human next).
            bias: DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: MultisampleState {
            count: 1, // the depth atlas is single-sampled regardless of the main view's MSAA
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // Fragment with NO color target: it only discards (BLEND) / alpha-tests (CUTOUT) casters.
        fragment: Some(FragmentState {
            shader: draw.shadow_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![],
        }),
        zero_initialize_workgroup_memory: false,
    });

    // group(3): ShVolume uniform (0) + 3 SH 3D textures (1,2,3) + filtering sampler (4) + #5 shadow
    // additions: SunShadowUniform (5) + depth-2d-array (6) + comparison sampler (7). SHARED by both
    // the opaque and BLEND pipeline specializations (like the group(2) material layout).
    let sh_layout = render_device.create_bind_group_layout(
        "eft_sh_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, uniform_buffer_sized(false, None)),
                (1, texture_3d(TextureSampleType::Float { filterable: true })),
                (2, texture_3d(TextureSampleType::Float { filterable: true })),
                (3, texture_3d(TextureSampleType::Float { filterable: true })),
                (4, sampler(SamplerBindingType::Filtering)),
                (5, uniform_buffer_sized(false, None)),          // #5 SunShadowUniform
                (6, texture_2d_array(TextureSampleType::Depth)), // #5 texture_depth_2d_array
                (7, sampler(SamplerBindingType::Comparison)),    // #5 sampler_comparison
            ),
        ),
    );
    let sh_bg = render_device.create_bind_group(
        "eft_sh_bg",
        &sh_layout,
        &BindGroupEntries::with_indices((
            (0, sh_uniform.as_entire_binding()),
            (1, &sh_r_view),
            (2, &sh_g_view),
            (3, &sh_b_view),
            (4, &sh_sampler),
            (5, shadow_main_uniform.as_entire_binding()),
            (6, &shadow_array_view),
            (7, &shadow_cmp_sampler),
        )),
    );

    // Re-insert the draw pipeline WITH the material + SH layouts now known, so specialize() can
    // build the 4-group pipeline layout (view / ssbo / material / sh-gi).
    commands.insert_resource(EftDrawPipeline {
        shader: draw.shader.clone(),
        shadow_shader: draw.shadow_shader.clone(),
        mesh_pipeline: draw.mesh_pipeline.clone(),
        ssbo_layout: draw.ssbo_layout.clone(),
        material_layout: Some(material_layout),
        sh_layout: Some(sh_layout),
    });
    commands.insert_resource(EftMaterialResources {
        material_buf,
        textures,
        views,
        normal_textures,
        normal_views,
        sampler: albedo_sampler,
    });
    commands.insert_resource(EftMaterialBindGroup(material_bg));
    commands.insert_resource(EftShResources {
        uniform: sh_uniform,
        textures: vec![sh_r_tex, sh_g_tex, sh_b_tex],
        views: vec![sh_r_view, sh_g_view, sh_b_view],
        sampler: sh_sampler,
    });
    commands.insert_resource(EftShBindGroup(sh_bg));
    // #5 shadows: the runtime switch, the queued pipeline + cascade layout, and the GPU resources.
    commands.insert_resource(EftShadowConfig {
        lsun,
        enabled: shadows_enabled,
        debug: shadow_debug,
    });
    commands.insert_resource(EftShadowPipeline {
        pipeline_id: shadow_pipeline_id,
        cascade_layout,
    });
    commands.insert_resource(EftShadowResources {
        depth_texture: shadow_depth,
        array_view: shadow_array_view,
        layer_views: shadow_layer_views,
        cascade_uniforms,
        cascade_bind_groups,
        main_uniform: shadow_main_uniform,
        comparison_sampler: shadow_cmp_sampler,
    });
    info!(
        "gpu-driven M3: {} albedo textures uploaded, material table + bindless bind group built",
        tex_count
    );
    info!(
        "gpu-driven Phase2b: {} normal-map textures uploaded (LINEAR Rgba8Unorm), normal_tex @group(2) binding(3)",
        normal_count
    );
    info!(
        "gpu-driven SH-GI: irradiance volume uploaded ({}x{}x{}), group(3) bind group built",
        sh_nx, sh_ny, sh_nz
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

/// Phase 2b: decode one normal-map PNG and upload it as a LINEAR Rgba8Unorm GPU texture (+ view).
/// Normal maps encode tangent-space vectors, NOT color — they are LINEAR data, so we must use the
/// non-sRGB format (an sRGB view would gamma-decode the vectors and wash out the perturbation).
/// On any read/decode failure returns a 1x1 flat tangent normal (128,128,255 -> +Z) so the
/// bindless index stays aligned with materials.json (a shifted index would normal-map the map wrong).
fn load_normal_texture(
    device: &RenderDevice,
    queue: &RenderQueue,
    path: &str,
) -> (Texture, TextureView) {
    match image::open(path) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            upload_rgba8_linear(device, queue, w.max(1), h.max(1), &rgba, "eft_normal")
        }
        Err(e) => {
            warn!("gpu-driven Phase2b: normal '{path}' failed to load ({e}); using flat placeholder");
            upload_rgba8_linear(device, queue, 1, 1, &[128u8, 128, 255, 255], "eft_normal_missing")
        }
    }
}

/// 1x1 flat tangent normal (128,128,255 -> +Z) for a pack that referenced no normal maps at all
/// (keeps the `normal_tex` binding_array non-empty).
fn make_dummy_normal_texture(device: &RenderDevice, queue: &RenderQueue) -> (Texture, TextureView) {
    upload_rgba8_linear(device, queue, 1, 1, &[128u8, 128, 255, 255], "eft_normal_dummy")
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

/// Phase 2b: upload RGBA8 bytes as a LINEAR (Rgba8Unorm) texture — for normal maps, whose texels
/// are tangent-space vectors, not sRGB color. Identical to `upload_rgba8_srgb` but for the format.
fn upload_rgba8_linear(
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
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm, // LINEAR — NOT sRGB (normal vectors, not color)
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
    // #5 shadows: when enabled, extrude the frustum toward the sun so off-screen casters survive
    // the SHARED cull and appear in the shadow map. `None`/disabled -> the cull is byte-identical
    // to before (perfect A/B against not EFT_SHADOWS=1).
    shadow: Option<Res<EftShadowConfig>>,
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
    let mut planes = build_frustum_planes(clip_from_world);
    // Conservatively extrude toward the sun: a possible caster sits at `receiver + Lsun*t`, so push
    // only the planes it could cross by `t*max(0, -n·Lsun)`. This ONLY loosens the frustum (never
    // wrongly culls a visible instance); the main pass then processes some extra off-screen
    // instances but its image is unchanged (they clip). See the plan's Visibility/indirect reuse.
    if let Some(shadow) = shadow.as_ref() {
        if shadow.enabled {
            let lsun = shadow.lsun;
            for p in planes.iter_mut() {
                let n = Vec3::new(p.x, p.y, p.z);
                p.w += SHADOW_CASTER_EXTRUDE * (-n.dot(lsun)).max(0.0);
            }
        }
    }
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

// ---- PrepareResources: #5 fit + upload the 2 cascade matrices each frame ----
// For each cascade slice [n_i, f_i] this reconstructs the camera sub-frustum's 8 world corners,
// fits a rotation-invariant (bounding-sphere) SQUARE in the sun's light space, texel-snaps its
// centre (kills shimmer), fits the light-space Z over the caster-extruded + receiver-margin corner
// set, and builds a conventional 0..1-depth orthographic `view_proj = ortho * light_view`. Uploads
// the per-cascade uniforms (shadow pass) + the combined SunShadowUniform (main pass). No-op cost
// when disabled is trivial (still uploads, but the main shader gates everything on enabled).
fn prepare_shadow_uniforms(
    render_queue: Res<RenderQueue>,
    config: Option<Res<EftShadowConfig>>,
    resources: Option<Res<EftShadowResources>>,
    views: Query<&ExtractedView, With<CullCamera>>,
) {
    let (Some(config), Some(res)) = (config, resources) else {
        return;
    };
    let Some(view) = views.iter().next() else {
        return;
    };
    let lsun = config.lsun;
    let clip_from_view = view.clip_from_view;
    let world_from_view = view.world_from_view.to_matrix();
    let world_from_clip = world_from_view * clip_from_view.inverse();

    // NDC z for a point at positive view-space distance `d` in front of the camera (view looks down
    // -Z). Works for any projection (incl. Bevy reverse-z) since it re-projects through the camera.
    let ndc_z_at = |d: f32| -> f32 {
        let clip = clip_from_view * Vec4::new(0.0, 0.0, -d, 1.0);
        clip.z / clip.w
    };
    // Stable up axis: Y unless Lsun is nearly vertical, then Z.
    let up = if lsun.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };

    let mut main = SunShadowUniform {
        split_depths: [
            SHADOW_SPLITS[1],
            SHADOW_SPLITS[2],
            SHADOW_CASCADE_OVERLAP,
            if config.enabled { 1.0 } else { 0.0 },
        ],
        sun_dir_texel: [lsun.x, lsun.y, lsun.z, 1.0 / SHADOW_MAP_SIZE as f32],
        combine: [
            SHADOW_DIFFUSE_CAP,
            SHADOW_FADE_START,
            SHADOW_FADE_END,
            if config.debug { 1.0 } else { 0.0 },
        ],
        ..default()
    };

    for c in 0..SHADOW_CASCADES {
        let near = SHADOW_SPLITS[c];
        let far = SHADOW_SPLITS[c + 1];
        let zn = ndc_z_at(near);
        let zf = ndc_z_at(far);

        // 8 world-space corners of this slice.
        let mut corners = [Vec3::ZERO; 8];
        let mut k = 0usize;
        for &z in &[zn, zf] {
            for &y in &[-1.0f32, 1.0] {
                for &x in &[-1.0f32, 1.0] {
                    let p = world_from_clip * Vec4::new(x, y, z, 1.0);
                    corners[k] = p.truncate() / p.w;
                    k += 1;
                }
            }
        }

        // Centroid + rotation-invariant bounding-sphere radius (constant cascade size under camera
        // rotation -> no size shimmer). SQUARE fit uses this radius on both axes.
        let mut center = Vec3::ZERO;
        for cc in &corners {
            center += *cc;
        }
        center /= 8.0;
        let mut radius = 0.0f32;
        for cc in &corners {
            radius = radius.max((*cc - center).length());
        }
        radius = radius.max(0.05);

        // Light view: eye on the sun side looking at the slice centre.
        let eye = center + lsun * (radius + SHADOW_CASTER_EXTRUDE);
        let light_view = Mat4::look_at_rh(eye, center, up);

        // Texel-snap the light-space XY centre so the cascade doesn't crawl as the camera moves.
        let world_texel = (2.0 * radius) / SHADOW_MAP_SIZE as f32;
        let center_ls = light_view.transform_point3(center);
        let snapped_x = (center_ls.x / world_texel).round() * world_texel;
        let snapped_y = (center_ls.y / world_texel).round() * world_texel;

        // Light-space Z fit over the receiver corners + caster extrusion (toward the sun) + receiver
        // margin (away from the sun). In RH light space, in-front points have negative z.
        let mut zmin = f32::MAX;
        let mut zmax = f32::MIN;
        for cc in &corners {
            for p in &[
                *cc,
                *cc + lsun * SHADOW_CASTER_EXTRUDE,
                *cc - lsun * SHADOW_RECEIVER_MARGIN,
            ] {
                let z = light_view.transform_point3(*p).z;
                zmin = zmin.min(z);
                zmax = zmax.max(z);
            }
        }
        let ortho_near = (-zmax).max(0.0);
        let ortho_far = (-zmin).max(ortho_near + 0.1);

        let proj = Mat4::orthographic_rh(
            snapped_x - radius,
            snapped_x + radius,
            snapped_y - radius,
            snapped_y + radius,
            ortho_near,
            ortho_far,
        );
        let view_proj = proj * light_view;
        let vp_cols = view_proj.to_cols_array_2d();

        main.view_proj[c] = vp_cols;
        main.texel_world[c] = world_texel;

        let cascade = ShadowCascadeUniform {
            view_proj: vp_cols,
            dir_texel: [lsun.x, lsun.y, lsun.z, 1.0 / SHADOW_MAP_SIZE as f32],
        };
        render_queue.write_buffer(
            &res.cascade_uniforms[c],
            0,
            bytemuck::bytes_of(&cascade),
        );
    }

    render_queue.write_buffer(&res.main_uniform, 0, bytemuck::bytes_of(&main));
}

// ---- QueueMeshes: specialize both passes + add the TWO phase items ----------
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
    // panic on a None layout. Phase 1: the group(3) SH-GI layout lands in the SAME prepare, so
    // gate on it too (specialize() builds the 4-group layout; the draw sets the SH bind group).
    if draw_pipeline.material_layout.is_none() || draw_pipeline.sh_layout.is_none() {
        return;
    }
    let draw_fn = draw_functions.read().id::<DrawGpuDriven>();

    for (view, msaa) in &views {
        let Some(phase) = transparent_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        // M3b1: TWO specializations of the same shader/mesh, selected by `blend_pass`.
        // They must be distinct keys so the cache yields two distinct pipeline ids.
        let opaque_pipeline = pipelines.specialize(
            &pipeline_cache,
            &draw_pipeline,
            EftDrawKey {
                samples: msaa.samples(),
                hdr: view.hdr,
                blend_pass: false,
            },
        );
        let blend_pipeline = pipelines.specialize(
            &pipeline_cache,
            &draw_pipeline,
            EftDrawKey {
                samples: msaa.samples(),
                hdr: view.hdr,
                blend_pass: true,
            },
        );

        for (entity, main_entity) in &markers {
            // Transparent3d sorts ASCENDING by distance (values increase toward the camera), so
            // the OPAQUE item at a large NEGATIVE distance runs FIRST and writes depth; the BLEND
            // item at ~0.0 runs after and depth-tests against that. Both share the same draw_fn /
            // multi-draw command and differ ONLY by pipeline (P1 discards BLEND mats, P2 discards
            // the rest), so each pass draws exactly its material class in one multi_draw.
            phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline: opaque_pipeline,
                draw_function: draw_fn,
                distance: -1.0e30, // sort FIRST (writes depth)
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
            phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline: blend_pipeline,
                draw_function: draw_fn,
                distance: 0.0, // sort AFTER opaque (depth-tests against P1's writes)
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
    /// M3b1 pass selector. `false` = P1 OPAQUE specialization (blend None, depth-write on,
    /// default bias, discards BLEND materials). `true` = P2 BLEND specialization (alpha
    /// blending, depth-write OFF, toward-camera depth bias, `BLEND_PASS` shader_def, discards
    /// non-BLEND materials). MUST be part of Hash/Eq so P1 and P2 cache as SEPARATE pipelines.
    blend_pass: bool,
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
        // group(3): SH-GI irradiance-volume layout (Phase 1). Same gate as material_layout, and
        // SHARED by both the opaque and BLEND specializations.
        let sh_layout = self
            .sh_layout
            .clone()
            .expect("EftDrawPipeline.sh_layout must be set before specialize (SH-GI)");

        // --- M3b1 pass-dependent state -----------------------------------------------
        // P2 (blend_pass) uses non-premultiplied alpha blending (matches Unity _Color*_MainTex),
        // turns OFF depth-write (transparents must not occlude each other or later opaques), and
        // nudges decals TOWARD the camera under reverse-z so they win the coplanar z-test against
        // the ground they lie on. P1 (opaque) keeps the original opaque state exactly.
        let (blend, depth_write_enabled, bias, frag_defs): (
            Option<BlendState>,
            bool,
            DepthBiasState,
            Vec<bevy::shader::ShaderDefVal>,
        ) = if key.blend_pass {
            (
                Some(BlendState::ALPHA_BLENDING),
                false,
                // Depth bias for coplanar decals under Bevy REVERSE-Z (near=1.0, far=0.0,
                // depth_compare GreaterEqual). The rasterizer bias is ADDED to window-space depth
                // [0,1]; a POSITIVE bias INCREASES depth = pulls the fragment TOWARD the camera
                // (larger reverse-z value), so the decal beats the coplanar ground P1 wrote and
                // passes GreaterEqual. (This matches Bevy StandardMaterial: positive depth bias
                // renders "closer to the camera".) A negative bias would push decals BEHIND the
                // ground and drop them.
                // TODO(M3b1 depth-bias magnitude): CORE_3D_DEPTH_FORMAT is Depth32Float, so the
                // `constant` unit scales with the polygon's depth exponent and huge Tarkov map
                // distances can make constant:2 too weak. If road markings still z-fight after the
                // first visual test, RAISE magnitude (constant: 4..16 and/or slope_scale: 2.0..4.0),
                // keeping BOTH positive. Do NOT flip to negative â€” that hides decals entirely.
                DepthBiasState {
                    constant: 2,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
                vec!["BLEND_PASS".into()],
            )
        } else {
            (None, true, DepthBiasState::default(), vec![])
        };

        RenderPipelineDescriptor {
            label: Some(if key.blend_pass {
                "eft_gpu_draw_blend".into()
            } else {
                "eft_gpu_draw_opaque".into()
            }),
            layout: vec![
                view_layout,
                self.ssbo_layout.clone(),
                material_layout,
                sh_layout,
            ],
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
                        // M3b2: per-vertex COLOR_0 vert-paint weight @36 (SoftCutout coverage
                        // rides on color.a). Interpolated (NOT flat) in the fragment shader.
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 36,
                            shader_location: 4,
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
                // P1 opaque writes depth; P2 blend reads it but does NOT write (see above).
                depth_write_enabled,
                // Bevy uses reverse-z; both passes compare GreaterEqual (blend still depth-TESTS
                // against the depth P1 wrote â€” both ride the one transparent pass that LOADS depth).
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState::default(),
                bias,
            }),
            multisample: MultisampleState {
                count: key.samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                // P2 pushes "BLEND_PASS" so the fragment discards NON-blend materials and outputs
                // the real computed alpha; P1 has no def and discards BLEND materials, alpha 1.0.
                shader_defs: frag_defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend,
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
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // Only run for the tagged player view (Core3d may run for several views); the cull writes
        // GLOBAL buffers from that view's frustum, so running it for other views is redundant work.
        if world.get::<CullCamera>(graph.view_entity()).is_none() {
            return Ok(());
        }
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
// #5 Shadow node: render the 2 cascade depth layers, reusing the camera-culled
// visible[]/indirect stream READ-ONLY. Runs after EftCull, before StartMainPass.
// ===========================================================================
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct EftShadowLabel;

struct EftShadowNode;

impl FromWorld for EftShadowNode {
    fn from_world(_: &mut World) -> Self {
        Self
    }
}

impl Node for EftShadowNode {
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // Only run for the tagged player view (avoids duplicate atlas clears/draws on other views).
        if world.get::<CullCamera>(graph.view_entity()).is_none() {
            return Ok(());
        }
        let (Some(config), Some(buffers), Some(draw_bg), Some(material_bg), Some(res), Some(pipe)) = (
            world.get_resource::<EftShadowConfig>(),
            world.get_resource::<EftGpuBuffers>(),
            world.get_resource::<EftDrawBindGroup>(),
            world.get_resource::<EftMaterialBindGroup>(),
            world.get_resource::<EftShadowResources>(),
            world.get_resource::<EftShadowPipeline>(),
        ) else {
            return Ok(()); // resources not built yet (or feature-disabled path)
        };
        // Disabled (no sun_dir or not EFT_SHADOWS=1): skip entirely. The main shader has enabled=0 and
        // never samples the (then-undefined) depth atlas, so this is a strict no-op.
        if !config.enabled {
            return Ok(());
        }
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(pipe.pipeline_id) else {
            return Ok(()); // shadow pipeline still compiling
        };

        // One depth-only render pass per cascade layer: clear to 1.0, then the SAME multidraw the
        // main pass uses (indirect buffer READ-ONLY — never reset/reculled here).
        for c in 0..SHADOW_CASCADES {
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("eft_shadow_cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &res.layer_views[c],
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &draw_bg.0, &[]); // instances + visible (shared)
            pass.set_bind_group(1, &res.cascade_bind_groups[c], &[]); // this cascade's view_proj
            pass.set_bind_group(2, &material_bg.0, &[]); // material table + albedo (alpha test)
            pass.set_vertex_buffer(0, buffers.vertex.slice(..));
            pass.set_index_buffer(buffers.index.slice(..), 0, IndexFormat::Uint32);
            pass.multi_draw_indexed_indirect(&buffers.indirect, 0, buffers.mesh_count);
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
        Option<SRes<EftShBindGroup>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        _entity: Option<()>,
        (buffers, draw_bg, material_bg, sh_bg): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (Some(buffers), Some(draw_bg), Some(material_bg), Some(sh_bg)) =
            (buffers, draw_bg, material_bg, sh_bg)
        else {
            return RenderCommandResult::Skip;
        };
        let buffers = buffers.into_inner();
        let draw_bg = draw_bg.into_inner();
        let material_bg = material_bg.into_inner();
        let sh_bg = sh_bg.into_inner();

        pass.set_bind_group(1, &draw_bg.0, &[]);
        pass.set_bind_group(2, &material_bg.0, &[]); // M3: bindless materials/textures/sampler
        pass.set_bind_group(3, &sh_bg.0, &[]); // Phase 1: SH-GI irradiance volume + uniform
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
