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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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
    diagnostic::RecordDiagnostics,
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
        PipelineCache, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment,
        RenderPassDepthStencilAttachment,
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
// #6: LOCK the byte layout — matches `InstanceGpu` in gpu_cull.wgsl / gpu_draw.wgsl / gpu_shadow.wgsl
// (5×vec4 = 80). A silent drift corrupts every instance's transform on the GPU.
const _: () = assert!(std::mem::size_of::<InstanceGpuRecord>() == 80);

/// Per-mesh static metadata. 32 bytes (16-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshMeta {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub instance_base: u32,
    pub instance_count: u32,
    /// Blend-pass class carried to the GPU: 0 = opaque-only, 1 = blend-only, 2 = mixed (draws in
    /// both passes). Read by gpu_cull.wgsl `cs_reset` (field `blend_class`) to zero the opaque vs
    /// blend indirect index_count per class. NOT padding — do not zero it.
    pub blend_class: u32,
    pub _pad: [u32; 2],
}
// #6: LOCK the byte layout — matches `MeshMeta` in gpu_cull.wgsl (8×u32 = 32, blend_class @20).
const _: () = assert!(std::mem::size_of::<MeshMeta>() == 32);

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

/// Tiny per-frame cull uniform: 6 normalized inward frustum planes + counts + the screen-size
/// cull anchor + the distance-LOD params. 144 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CullUniform {
    pub frustum: [[f32; 4]; 6],
    /// x = instance_count, y = mesh_count, z = bitcast f32 k_grass (grass screen-size cull
    /// threshold — larger than the general one so 100k sub-pixel clumps drop early),
    /// w = bitcast f32 grass MAX DISTANCE in metres (0 = off). The z threshold is already a distance
    /// test, but in px/(0.5·viewport_h·proj11) units, so its world horizon moves with resolution and
    /// FOV and differs per grass kind; w is an absolute clamp that does none of that. Was a pad lane,
    /// so adding it does not change the struct size.
    pub counts: [u32; 4],
    /// Screen-size cull: xyz = camera world pos, w = k_general where
    /// k = min_px / (0.5 * viewport_h * proj11). An instance is culled when its bounding-sphere
    /// radius subtends fewer than min_px pixels: sphere.w < k * distance(cam, sphere.center).
    /// Zeros = cull nothing (build-time seed before the first upload_frustum).
    pub cam_k: [f32; 4],
    /// Distance-LOD (LOD_DISTANCE_PLAN.md): x = proj11 (1/tan(fovY/2)), y = lod_bias (>1 holds finer
    /// shells longer), z = mode (0 = max detail / default shell only, 1 = distance-based, 2 = force
    /// shell w), w = forced shell index (mode 2). Instances with ids.w == 0 (sentinel) ignore all of
    /// this and always draw (lean packs, ungrouped, single-shell groups).
    pub lod_params: [f32; 4],
}
// #6: LOCK the byte layout — matches `CullGlobals` in gpu_cull.wgsl (array<vec4,6> + 3×vec4 = 144).
const _: () = assert!(std::mem::size_of::<CullUniform>() == 144);

/// Stride of one indirect draw record, in bytes.
pub const DRAW_ARG_STRIDE: u64 = 20;
/// The u32 material index is written as `f32::from_bits(material_id)` so vertex_data stays a single
/// `Vec<f32>`; the GPU reads that slot as `Uint32` and recovers the id bit-exact (a pure
/// reinterpretation, NOT a numeric cast, which would corrupt large ids). The colour slot is
/// smuggled the same way. The SoftCutout road/track feather rides on color.a.
/// GPU vertex stride: pos f32x3 @0, normal **oct-encoded Snorm16x2** @12, uv f32x2 @16,
/// material Uint32 @24, color **Unorm8x4** @28. The pack stores colour as unorm8x4 already; this used to inflate it to
/// Float32x4 (stride 52) for no reason, costing 12 B on every one of the map's tens of millions of
/// vertices (653 MiB on streets). Unorm8x4 still arrives in the shader as a normalised
/// `vec4<f32>`, so the WGSL is unchanged and the values are bit-identical to the source data.
pub const DRAW_VERTEX_STRIDE: u64 = 32;

/// Octahedral-encode a unit normal to two snorm components in [-1,1] (Meyer et al. / Cigolle et al.).
/// 16 bits per component is far past what a normal needs — it is what Bevy uses for meshlet normals —
/// and it costs 4 B instead of 12 B, i.e. 457 MiB on streets' 57.1M vertices. The shader decodes with
/// a few add/mul and one normalize (`oct_decode` in gpu_draw.wgsl).
fn oct_encode(n: Vec3) -> [i16; 2] {
    let n = n.normalize_or_zero();
    let d = n.x.abs() + n.y.abs() + n.z.abs();
    let (mut x, mut y) = if d > 1e-20 { (n.x / d, n.y / d) } else { (0.0, 0.0) };
    if n.z < 0.0 {
        // Fold the lower hemisphere out onto the octahedron's outer diamond.
        let (ax, ay) = (x.abs(), y.abs());
        let (sx, sy) = (if x >= 0.0 { 1.0 } else { -1.0 }, if y >= 0.0 { 1.0 } else { -1.0 });
        let (nx, ny) = ((1.0 - ay) * sx, (1.0 - ax) * sy);
        x = nx;
        y = ny;
    }
    // Round-to-nearest into snorm16; clamp keeps -1.0 exactly representable at -32767.
    let q = |v: f32| -> i16 { (v.clamp(-1.0, 1.0) * 32767.0).round() as i16 };
    [q(x), q(y)]
}

/// Pack an oct-encoded normal into one u32 (low half = x, high = y) for the f32 staging vec.
fn oct_bits(n: Vec3) -> f32 {
    let e = oct_encode(n);
    f32::from_bits((e[0] as u16 as u32) | ((e[1] as u16 as u32) << 16))
}

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
    /// GLASS_TRS: `_ReflectColor` packed RGBA8 (was padding; 0 on every non-TRS material). @92
    pub glass_refl: u32,
    /// RAW _DetailAlbedoMap_ST (sx,sy,ox,oy). Shader derives the relative transform vs `uv_xform`. @96
    pub detail_albedo_uv: [f32; 4],
    /// RAW _DetailNormalMap_ST (sx,sy,ox,oy). @112
    pub detail_normal_uv: [f32; 4],
    /// x = albedo blend strength, y = detail normal scale, z = fade start (8 m), w = fade end (15 m). @128
    pub detail_params: [f32; 4],
    /// xyz = offline albedoMeanGain = mean(sample_linear × 4.5948); w = 1. Divisor for neutralize. @144
    pub detail_mean_gain: [f32; 4],
    // ---- Emissive: adds 16 bytes (160 -> 176). Windows / monitors / signs / lamps glow —
    //      with the HDR view target + Bloom they read like the game's lit interiors. ----
    /// bindless `albedo_tex` index of the emissive texture (sRGB, matching the pack's
    /// conventions.colorSpace.emissive), or `NO_EMISSIVE`. @160
    pub emissive_index: u32,
    /// linear rgb emissive = factor × hdr, precomputed on CPU. Declared as 3 scalars (not a
    /// vec3) in WGSL too, so the struct stays vec4-aligned with no implicit vec3 16-padding. @164
    pub emissive_rgb: [f32; 3], // @164
    // ---- Parallax: adds 16 bytes (176 -> 192). Zero for the ~all non-parallax materials
    //      (parallax_index == NO_ALBEDO -> the shader's steep-parallax path is fully skipped ->
    //      those materials render byte-identical). The height map is appended to the SAME bindless
    //      `albedo_tex` array as the base textures (uploaded LINEAR — it is height DATA). ----
    /// bindless `albedo_tex` index of the grayscale height map, or `NO_ALBEDO` when absent. @176
    pub parallax_index: u32,
    /// Unity `_Parallax` amount (max tangent-space UV offset; typical 0.02-0.08). @180
    pub parallax_scale: f32,
    /// GLASS_TRS: `_SpecColor` packed RGB8 (was padding; 0 on every non-TRS material). @184
    pub glass_spec: u32,
    /// GLASS_TRS: `_Shininess` (Blinn-Phong gloss 0..1; was padding). @188 -> total 192 bytes
    pub glass_shin: f32,
}

// #6: compile-time guard that GpuMaterial stays byte-matched to the WGSL `MaterialGpu` (192 B, all
// vec4 lanes 16-aligned). A silent mismatch here would corrupt EVERY material's GPU record, so this
// is checked at `cargo check` time (const eval) rather than trusted by eye.
const _: () = assert!(std::mem::size_of::<GpuMaterial>() == 192);
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
/// `GpuMaterial::flags` bit: roughness-from-albedo-alpha (Unity Standard smoothness-in-alpha).
/// The fragment derives per-pixel roughness = 1 - tex.a (raw alpha, NOT tint-multiplied) instead
/// of the constant `roughness`. Only set for role=opaque (glass keeps its authored 0.05; cutout
/// alpha is coverage, not smoothness); cleared again for terrain-tagged materials.
pub const MAT_FLAG_RFA: u32 = 1 << 6;
/// `GpuMaterial::flags` bit: Vert-Paint 3-layer splat (Custom/Vert Paint SoftCutout Decal AND
/// the opaque "Vert Paint Shader Solid" variant — any material whose `vp.layers` has 3 entries).
/// The fragment replaces the single-albedo sample with the game's height-splat blend
/// (`w_i = pow(Heights_i(raw_uv) * COLOR_0_i, blend)`, normalized), reading `VpGpu` at index
/// `_pad2` (disjoint with terrain's `_pad2` slice — a material is never both). Without this the
/// viewer rendered ONLY layer 0 at full strength: parking lots whose layer 0 is `road_sand`
/// tiled a loud rust-orange blotch grid instead of the game's asphalt/gravel/sand mix.
pub const MAT_FLAG_VP: u32 = 1 << 7;
/// `GpuMaterial::flags` bit: puddle whose shape MASK is in the LUMA (rgb) channel, not alpha.
/// City_puddle_atlas ships alpha≡1.0 with the coverage in red (the game's Decal/Water Deferred
/// Decal samples `.r`); without this the puddle feathers on a constant-1 alpha and the whole quad
/// renders as a solid slab. Detected at load by `puddle_alpha_is_constant`.
pub const MAT_FLAG_PUDDLE_LUMA: u32 = 1 << 8;
/// `GpuMaterial::flags` bit: a STRETCHED floor water-decal — the `Water Deferred Decal` shader also
/// serves large wet-ground / tire-mark trails whose texture is mapped at tens-to-hundreds of meters
/// per repeat (vs a puddle's few). Those are matte, NOT reflective puddles, so the shader drops the
/// mirror + sun glint for them. Set at load from the per-material world-meters-per-uv-repeat.
pub const MAT_FLAG_WATER_MATTE: u32 = 1 << 9;
/// `GpuMaterial::flags` bit: plain surface decal. Transparent texels mask every lighting term;
/// glass intentionally keeps its reflection outside that coverage mask.
pub const MAT_FLAG_DECAL: u32 = 1 << 10;
/// `GpuMaterial::flags` bit: parallax (steep/occlusion) mapping — offset the base albedo/normal UV
/// along the tangent-space view vector using `parallax_index`'s height map. Set only when a valid
/// height map is present; skipped entirely otherwise (byte-identical to the pre-parallax path).
pub const MAT_FLAG_PARALLAX: u32 = 1 << 11;
/// `GpuMaterial::flags` bit: glass whose albedo ALPHA is a COVERAGE MASK (broken-shard atlases),
/// not packed smoothness. Alpha 0 there means NO SURFACE — so the glass branch masks EVERY term
/// by it, including the additive specular/reflection that clear panes deliberately keep outside
/// their alpha. Without this the empty atlas area still rendered as a ghost pane of sky
/// reflection. Detected at load by `glass_alpha_is_mask`; mutually exclusive with MAT_FLAG_RFA.
pub const MAT_FLAG_GLASS_MASK: u32 = 1 << 12;
/// `GpuMaterial::flags` bit: legacy Transparent/Reflective/Specular glass (the game's car and
/// storefront glass family). tex.a = TRANSPARENCY x gloss (legacy Unity convention, never
/// smoothness); reflection is tinted by the material's own `_ReflectColor` (`glass_refl`) and
/// specular by `_SpecColor`/`_Shininess` (`glass_spec`/`glass_shin`) — the authored values whose
/// absence made crumpled windshields mirror the full-strength analytic sky as WHITE FOIL and
/// painted bullet holes as dark smoothness spots. Only set on packs whose extraction captured
/// the family (glassTRS in materials.json); older packs keep the probe/RFA path bit-exact.
pub const MAT_FLAG_GLASS_TRS: u32 = 1 << 13;
/// Per-mesh transparent-pass membership. A mixed-material mesh may set more than one bit and is
/// then submitted to each relevant specialization; the fragment material flag keeps only its class.
const BLEND_MESH_SOFTCUTOUT: u32 = 1 << 0;
const BLEND_MESH_OVERLAY: u32 = 1 << 1;
const BLEND_MESH_TRANSPARENT: u32 = 1 << 2;
/// `GpuMaterial::detail_flags` bit0: this material has a detail ALBEDO texture.
pub const DETAIL_FLAG_ALBEDO: u32 = 1 << 0;
/// `GpuMaterial::detail_flags` bit1: this material has a detail NORMAL texture.
pub const DETAIL_FLAG_NORMAL: u32 = 1 << 1;
/// `GpuMaterial::emissive_index` sentinel: material has no emissive texture.
pub const NO_EMISSIVE: u32 = 0xFFFF_FFFF;

/// MicroSplat splat table (group(2) binding(4), storage). All indices are into the SAME bindless
/// `albedo_tex` array as normal materials (the terrain textures are appended to `albedo_paths`).
/// Layer `i` weight = control map `i/4`, channel `i%4`. `layer_uv = terrainUV01 * rep` (the value
/// recovered from the MicroSplat material; NEVER `m_TileSize`). Slice names come from the
/// terrainLayers sidecar itself (Interchange = 4 slices, Lighthouse = 6; capacity 16). 288 bytes,
/// 16-aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TerrainSplatGpu {
    /// bindless albedo index of each of the 12 layers.
    pub layer_albedo: [u32; 12],
    /// per-layer UV repeat (`terrainUV01 * rep`).
    pub layer_rep: [f32; 12],
    /// up to 16 slices × 3 control-map bindless indices: slice `s` map `k` at `[s*3 + k]`.
    /// (Streets-scale maps can carry more slices than Interchange's 4 / Lighthouse's 6.)
    pub ctrl_idx: [u32; 48],
}
// #6: LOCK the byte layout — matches `TerrainSplat` in gpu_draw.wgsl (12+12+48 u32/f32 = 288).
const _: () = assert!(std::mem::size_of::<TerrainSplatGpu>() == 288);

/// Vert-Paint 3-layer splat table entry (group(2) binding(5), storage; one per MAT_FLAG_VP
/// material, indexed by `GpuMaterial::_pad2`). The EXACT game blend was RE'd from the DX11
/// fragment and validated in the web viewer (`tarkmap/out/_vpsplat.js`):
///   `w_i = pow(Heights_i(raw_uv) * COLOR_0_i, blend)` normalized; albedo = Σ w_i·layer_i·tint_i.
/// Layer 0's ST is baked into the mesh UVs (uvTilingBaked), so the shader un-bakes with `uv0`
/// to recover the raw UV that the heights mask and layers 1/2 sample from. 112 bytes, 16-aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct VpGpu {
    /// x,y,z = bindless `albedo_tex` indices of layers 0..2; w = heights control-mask index
    /// (uploaded LINEAR — it's blend weights, not color) or NO_ALBEDO when absent.
    pub tex: [u32; 4],
    /// RAW per-layer `_MainTex_ST` (sx,sy,ox,oy). uv0 is also the baked-in transform.
    pub uv0: [f32; 4],
    pub uv1: [f32; 4],
    pub uv2: [f32; 4],
    /// rgb = layer tint. tint0.w = heights blend sharpness (`vp.blend`); other w lanes unused.
    pub tint0: [f32; 4],
    pub tint1: [f32; 4],
    pub tint2: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<VpGpu>() == 112);

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
    /// xyz = (nx, ny, nz) probe grid dims (as f32); w = mean ground-adjacent / top-layer sky
    /// luma ratio (scales out-of-volume redirected samples down to ground-equivalent).
    pub dims: [f32; 4],
    /// xyz = (sx, sy, sz) probe spacing in meters, w unused.
    pub spacing: [f32; 4],
}
// #6: LOCK the byte layout — matches `ShVolume` in gpu_draw.wgsl (4×vec4 = 64).
const _: () = assert!(std::mem::size_of::<ShVolumeUniform>() == 64);

/// Default normal-bias (meters) written to `ShVolumeUniform::vol_inv_extent.w`: the shading
/// point is pushed this far along the surface normal before sampling the probe grid, so a
/// point sitting on a slab doesn't sample the dark "inside-solid" probe directly beneath it.
const SH_NORMAL_BIAS: f32 = 0.75;

// ---------------------------------------------------------------------------
// REALTIME point/spot lighting (no CUDA SH bake needed). EFT lights its maps with
// realtime lights; the pack carries the raw set (eftpack::Light). We build a static
// world CSR light grid on the CPU once per map and a fragment loop shades each pixel
// from the few lights whose range-sphere covers its cell. Auto-selected against the
// baked SH volume to avoid double-counting: a REAL volume already integrates the
// practicals (realtime OFF); a dummy volume (no CUDA) -> realtime ON.
// ---------------------------------------------------------------------------

/// Default realtime light-intensity multiplier (EFT_LIGHT_SCALE overrides). Folded into
/// `LightGridUniform::params.x`. Anchored to the CUDA bake's scale (6.0) — headless A/B on
/// factory_rework showed 6.0 reads a touch more present than 4.0 with no extra blow-out
/// (near-light pixels already saturate at both; broad interior fill is what lifts).
const DEFAULT_LIGHT_SCALE: f32 = 6.0;

/// Max cells in the world light grid before the cell size is grown to fit (keeps the grid
/// buffer small on kilometre-scale maps).
const LIGHT_GRID_MAX_CELLS: u64 = 4_000_000;

/// group(3) @binding(8) uniform. 48 bytes. Byte-identical to the WGSL `LightGrid`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct LightGridUniform {
    /// xyz = grid world-min corner, w = cell size (meters).
    grid_min: [f32; 4],
    /// xyz = grid dims, w = n_lights (0 => the shader skips the whole loop).
    grid_dims: [u32; 4],
    /// x = light_scale, y = ambient_scale, z = rt_enabled (1/0), w = B4-M sun-diffuse strength
    /// (EFT_SUN_DIFFUSE, 0 on a full/direct bake so it never double-counts the baked sun).
    params: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<LightGridUniform>() == 48);

/// CPU-staged realtime light set + CSR world grid, uploaded once in `prepare_gpu_buffers`.
/// Rides in `CpuData` (Arc-extracted, then freed with the rest of the staging blob).
struct LightGridCpu {
    uniform: LightGridUniform,
    /// 3 vec4 per light: v0=(pos.xyz,range) v1=(color.rgb,cos_outer) v2=(dir.xyz,cos_inner).
    /// Always >= 1 element (a dummy light when the pack has none) so the storage binding is valid.
    lights: Vec<[f32; 4]>,
    /// Per-light power-switch group index (-1 = always on), parallel to the LIGHTS (not the vec4s):
    /// `light_group[i]` is the group of the light packed at `lights[3*i..3*i+3]`. Used by the live
    /// power toggle to zero a group's colors without touching the CSR grid.
    light_group: Vec<i32>,
    /// CSR: `[0..=nCells]` prefix-sum offsets (each already includes the base = nCells+1) then the
    /// concatenated per-cell light-index lists. cell i's lights = `grid[grid[i]..grid[i+1]]`.
    grid: Vec<u32>,
}

/// Read an f32 env knob with a default (best-effort; unparseable -> default).
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

/// Build the static world light grid from the pack's reduced lights + the viewer-world AABB.
/// `rt_enabled` (auto-selected against the SH volume upstream) gates whether the grid is actually
/// populated: when off, a tiny 1-cell/0-light grid is emitted so the GPU binding stays valid but the
/// shader skips the loop (no wasted memory on maps that use the baked SH path).
fn build_light_grid(lights: &[crate::eftpack::Light], bounds: &[f32; 6], rt_enabled: bool) -> LightGridCpu {
    let light_scale = env_f32("EFT_LIGHT_SCALE", DEFAULT_LIGHT_SCALE);
    let ambient_scale = env_f32("EFT_AMBIENT_SCALE", 1.0);
    // B4-M: additive direct-sun diffuse for indirect-only bakes (the SH carries sky+bounce only, so
    // sunlit exteriors read flat). Strength lives in params.w; ONLY when rt_enabled (indirect-only /
    // no-volume) so a FULL bake — which already integrates the sun — leaves it 0 and never
    // double-counts. Live-tunable via EFT_SUN_DIFFUSE (0 = off).
    let sun_diffuse = if rt_enabled { env_f32("EFT_SUN_DIFFUSE", 0.8) } else { 0.0 };

    // Pack the light records (>=1 element so the storage buffer is never zero-sized).
    let mut lbuf: Vec<[f32; 4]> = Vec::with_capacity(lights.len().max(1) * 3);
    let mut light_group: Vec<i32> = Vec::with_capacity(lights.len().max(1));
    for l in lights {
        lbuf.push([l.pos.x, l.pos.y, l.pos.z, l.range]);
        lbuf.push([l.color.x, l.color.y, l.color.z, l.cos_outer]);
        lbuf.push([l.dir.x, l.dir.y, l.dir.z, l.cos_inner]);
        light_group.push(l.group_idx);
    }
    if lbuf.is_empty() {
        lbuf.extend_from_slice(&[[0.0; 4], [0.0; 4], [0.0; 4]]); // dummy light 0 (n_lights stays 0)
        light_group.push(-1);
    }

    // Grid extents from the LIGHTS' own AABB (± range), NOT the pack mesh bounds. Backdrop/skybox
    // meshes inflate pack bounds far past where lights live (Terminal: ±4.4 km vs a ~1 km harbor);
    // with the per-axis 256-cell clamp below, a bounds-sized grid stopped COVERING the playable
    // area — every light and every shaded pixel clamped into the same edge cells, so each pixel
    // looped over ~all lights (225 ms/frame). Lights bound their own influence exactly.
    let (min, max) = if lights.is_empty() {
        (
            Vec3::new(bounds[0], bounds[1], bounds[2]),
            Vec3::new(bounds[3], bounds[4], bounds[5]),
        )
    } else {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for l in lights {
            lo = lo.min(l.pos - Vec3::splat(l.range));
            hi = hi.max(l.pos + Vec3::splat(l.range));
        }
        (lo, hi)
    };

    let active = rt_enabled && !lights.is_empty();
    if !active {
        // 1-cell / 0-light grid: valid bindings, shader skips (grid_dims.w == 0).
        // offsets for the single cell: base = nCells+1 = 2, empty range [2,2).
        return LightGridCpu {
            uniform: LightGridUniform {
                grid_min: [min.x, min.y, min.z, 8.0],
                grid_dims: [1, 1, 1, 0],
                params: [light_scale, ambient_scale, 0.0, sun_diffuse],
            },
            lights: lbuf,
            light_group,
            grid: vec![2u32, 2u32],
        };
    }

    let extent = (max - min).max(Vec3::splat(1e-3));
    // Cell size = median light range clamped [4,12] m (small avg lights/cell, cheap fragment loop),
    // raised as needed so 256 cells/axis always COVER the extent — a grid that stops short of the
    // lights silently degenerates to the all-lights-in-edge-cells worst case (see bounds note above).
    let mut cell = {
        let mut ranges: Vec<f32> = lights.iter().map(|l| l.range).collect();
        ranges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ranges[ranges.len() / 2].clamp(4.0, 12.0)
    };
    cell = cell
        .max(extent.x / 256.0)
        .max(extent.y / 256.0)
        .max(extent.z / 256.0);
    let dims_for = |cell: f32| -> [u32; 3] {
        [
            ((extent.x / cell).ceil() as i64).clamp(1, 256) as u32,
            ((extent.y / cell).ceil() as i64).clamp(1, 256) as u32,
            ((extent.z / cell).ceil() as i64).clamp(1, 256) as u32,
        ]
    };
    let mut dims = dims_for(cell);
    let mut guard = 0;
    while (dims[0] as u64 * dims[1] as u64 * dims[2] as u64) > LIGHT_GRID_MAX_CELLS && guard < 64 {
        cell *= 1.5;
        dims = dims_for(cell);
        guard += 1;
    }
    let [nx, ny, nz] = dims;
    let n_cells = nx as usize * ny as usize * nz as usize;

    // Range of cells a light's range-sphere AABB overlaps, clamped to the grid.
    let cell_range = |l: &crate::eftpack::Light| -> ([u32; 3], [u32; 3]) {
        let idx = |v: f32, axis_min: f32, dim: u32| -> u32 {
            (((v - axis_min) / cell).floor() as i64).clamp(0, dim as i64 - 1) as u32
        };
        let lo = l.pos - Vec3::splat(l.range);
        let hi = l.pos + Vec3::splat(l.range);
        (
            [idx(lo.x, min.x, nx), idx(lo.y, min.y, ny), idx(lo.z, min.z, nz)],
            [idx(hi.x, min.x, nx), idx(hi.y, min.y, ny), idx(hi.z, min.z, nz)],
        )
    };

    // Two-pass CSR build: count per cell, prefix-sum (base-included), then scatter light indices.
    let mut counts = vec![0u32; n_cells];
    for l in lights {
        let (c0, c1) = cell_range(l);
        for z in c0[2]..=c1[2] {
            for y in c0[1]..=c1[1] {
                let row = (z as usize * ny as usize + y as usize) * nx as usize;
                for x in c0[0]..=c1[0] {
                    counts[row + x as usize] += 1;
                }
            }
        }
    }
    let base = (n_cells + 1) as u32;
    let mut offsets = vec![0u32; n_cells + 1];
    let mut acc = base;
    for i in 0..n_cells {
        offsets[i] = acc;
        acc += counts[i];
    }
    offsets[n_cells] = acc;
    let total_ins = (acc - base) as usize;
    let mut grid = vec![0u32; (n_cells + 1) + total_ins];
    grid[..n_cells + 1].copy_from_slice(&offsets);
    let mut cursor = offsets; // reuse as write cursors
    for (li, l) in lights.iter().enumerate() {
        let (c0, c1) = cell_range(l);
        for z in c0[2]..=c1[2] {
            for y in c0[1]..=c1[1] {
                let row = (z as usize * ny as usize + y as usize) * nx as usize;
                for x in c0[0]..=c1[0] {
                    let ci = row + x as usize;
                    grid[cursor[ci] as usize] = li as u32;
                    cursor[ci] += 1;
                }
            }
        }
    }

    info!(
        "gpu-driven realtime lights: {} lights, grid {}x{}x{} ({} cells, cell {:.1} m), {} index entries, \
         scale={:.2} ambient={:.2}",
        lights.len(),
        nx,
        ny,
        nz,
        n_cells,
        cell,
        total_ins,
        light_scale,
        ambient_scale,
    );

    LightGridCpu {
        uniform: LightGridUniform {
            grid_min: [min.x, min.y, min.z, cell],
            grid_dims: [nx, ny, nz, lights.len() as u32],
            params: [light_scale, ambient_scale, 1.0, sun_diffuse],
        },
        lights: lbuf,
        light_group,
        grid,
    }
}

// ---------------------------------------------------------------------------
// #5 Dynamic sun shadows — 2-cascade near-field contact CSM.
// ---------------------------------------------------------------------------
// A near-field, sun-aligned contact shadow map. The SH volume already bakes the BROAD sun shadow,
// so this only adds the missing high-frequency contact edge and is combined in the shader under a
// hard cap (anti double-darkening). Rendered into a 2-layer Depth32Float array by reusing the
// camera-culled `visible[]`/`indirect` stream READ-ONLY (never re-culls it). All shadow work is a
// strict no-op when the feature is disabled (sun_dir missing or not EFT_SHADOWS=1): `enabled=0` in the
// uniform, and the depth array — always allocated so the group(3) layout stays stable — is ignored.

/// Shadow-map resolution per cascade (square). 3072² * 2 layers * 4 bytes = 72 MiB.
///
/// Raised from 2048 alongside the rotation-invariant cascade fit. That fit centres each cascade on
/// the CAMERA instead of the frustum slice, which is what lets the #5b cache survive a pan (the
/// shadow pass went from 6.16 ms to ~0 while rotating), but an eye-centred square must reach the
/// slice's far corners: cascade 0's radius goes ~15 -> 20 m and cascade 1's ~78 -> 107 m, so texels
/// would be ~1.35x coarser at the same resolution. 3072 more than absorbs that (1.5x the samples
/// for 1.35x the area = ~1.1x FINER than the old default), so shadows come out slightly sharper
/// rather than softer. Resolution is nearly free on this path — 512² -> 4096² measured +0.32 ms,
/// because the cost is geometry resubmission, not fill. VRAM goes 32 -> 72 MiB; `EFT_SHADOW_SIZE`
/// still overrides either way (2048 restores the old 32 MiB, 4096 costs 128 MiB).
const SHADOW_MAP_SIZE_DEFAULT: u32 = 3072;
/// Live-tunable shadow-map resolution (`EFT_SHADOW_SIZE`, default 2048). Exposed so the shadow
/// pass cost can be bisected: if frame time scales with this, the cascades are FILL-bound; if it
/// barely moves, the cost is geometry resubmission (the cascades replay the main camera's
/// indirect buffer, so every visible instance is rasterized 3x per frame).
fn shadow_map_size() -> u32 {
    use std::sync::OnceLock;
    static S: OnceLock<u32> = OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("EFT_SHADOW_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|v| v.clamp(256, 8192))
            .unwrap_or(SHADOW_MAP_SIZE_DEFAULT)
    })
}
/// Does grass cast sun shadows? Off by default: the alpha-tested cross-quads dominated the shadow
/// pass for micro-shadows that read as noise at map scale. `EFT_GRASS_SHADOWS=1` restores them.
/// Shared by the cascade-uniform fit and the shadow multidraw so the two cannot disagree about
/// whether the grass range is drawn.
fn shadow_debug() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("EFT_SHADOW_DEBUG").is_ok_and(|v| v.trim() == "1"))
}

fn grass_shadows() -> bool {
    use std::sync::OnceLock;
    static G: OnceLock<bool> = OnceLock::new();
    *G.get_or_init(|| std::env::var("EFT_GRASS_SHADOWS").is_ok_and(|v| v.trim() == "1"))
}

/// Cascade count. The depth array has this many layers, and `SunShadowUniform` carries this many
/// matrices — bumping it changes that struct's size, so the WGSL twin and its assert move too.
///
/// Was 2, reaching only 80 m. On a map spanning ~1.2 km that left EVERYTHING past 80 m unshadowed:
/// no tree shadows on distant terrain, no self-shadowing on a treeline, and a hard fade band at
/// 65-80 m where the effect simply stopped. Two far cascades extend the reach to 700 m. They are
/// affordable specifically because the #5b cache now survives camera motion: a far cascade's snap
/// quantum is metres wide (`SHADOW_SNAP_TEXELS` × its texel), so it re-renders rarely even while
/// moving, where an unquantised fit would have re-rendered all four every frame.
const SHADOW_CASCADES: usize = 4;
/// Practical/log split distances (metres): cascade i covers [SHADOW_SPLITS[i], SHADOW_SPLITS[i+1]].
/// Roughly logarithmic so near-field texel density is preserved: the 0.5-15 m and 15-80 m bands are
/// unchanged from the 2-cascade fit, so close-up shadow quality is bit-identical to before.
const SHADOW_SPLITS: [f32; SHADOW_CASCADES + 1] = [0.5, 15.0, 80.0, 250.0, 700.0];
/// Cascade overlap fraction (reported in the uniform; the shader blends 13.5..15 m).
const SHADOW_CASCADE_OVERLAP: f32 = 0.10;
/// How far a caster may sit toward the sun and still project into the slice (light-space Z fit).
const SHADOW_CASTER_EXTRUDE: f32 = 80.0;
/// Cascade centre snap quantum, in shadow-map texels. 1 would snap to the finest grid (no crawl,
/// but the fit changes every 1.3 cm of walking, so the #5b cache never survives movement). Larger
/// values make translation lazy at the price of the shadow grid stepping in blocks when it does
/// move. 16 is the measured compromise; the fit radius is padded by one quantum to compensate.
const SHADOW_SNAP_TEXELS: f32 = 16.0;
/// Cascade radius quantum (metres). The raw max-corner distance jitters in f32's low bits as the
/// view matrix changes, and `view_proj` is cache-compared for exact equality, so the radius is
/// rounded UP to this step to make the fit bit-reproducible. 1 m costs <5% extra cascade extent.
const SHADOW_RADIUS_STEP: f32 = 1.0;
/// Receiver-side margin pulled away from the sun in the light-space Z fit.
const SHADOW_RECEIVER_MARGIN: f32 = 10.0;
/// Max fraction of REMOVABLE (above-floor) baked diffuse the contact term may subtract. Hard-capped.
const SHADOW_DIFFUSE_CAP: f32 = 0.12;
/// Far contact fade band (metres): the whole shadow effect fades to fully lit across this range.
/// MUST track the last cascade's far plane — these were 65/80 against the old 80 m reach, and left
/// unchanged they would have thrown away both new cascades by fading the effect out at 80 m.
const SHADOW_FADE_START: f32 = SHADOW_SPLITS[SHADOW_CASCADES - 1] + 350.0; // 600
const SHADOW_FADE_END: f32 = SHADOW_SPLITS[SHADOW_CASCADES]; // 700

/// group(1) per-cascade uniform for the shadow depth pass. Byte-identical to the WGSL
/// `ShadowCascadeUniform` (80 bytes, 16-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ShadowCascadeUniform {
    /// world -> sun light clip (conventional 0..1-depth ortho). Column-major Mat4 upload.
    view_proj: [[f32; 4]; 4],
    /// xyz = Lsun (toward the sun), w = 1/shadow_map_size() (PCF texel).
    dir_texel: [f32; 4],
    /// x = grass casts shadows (1/0). B2: the 109k grass cross-quads were ~the whole shadow-pass
    /// fragment cost (alpha-tested albedo sample × 2 cascades) for micro-shadows that read as
    /// noise at map scale — skipped by default; EFT_GRASS_SHADOWS=1 restores.
    /// y = bindless `albedo_tex` array length, so the shadow fragment can CLAMP its descriptor
    /// index (WGSL has no `arrayLength` for a `binding_array`). An out-of-range binding_array
    /// index faults AMD hardware outright, so the bound must reach the shader. zw pad.
    params: [f32; 4],
}
// #6: LOCK the byte layout — matches `ShadowCascadeUniform` in gpu_shadow.wgsl (mat4 + 2×vec4 = 96).
const _: () = assert!(std::mem::size_of::<ShadowCascadeUniform>() == 96);

/// group(3) binding(5) main sun-shadow uniform read by gpu_draw.wgsl. Byte-identical to the WGSL
/// `SunShadowUniform` (352 bytes: 4×64 + 6×16).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SunShadowUniform {
    /// Per-cascade world->light-clip matrices (column-major).
    view_proj: [[[f32; 4]; 4]; SHADOW_CASCADES],
    /// Per-cascade FAR plane in metres: x = far0 (15), y = far1 (80), z = far2 (250), w = far3 (700).
    /// One lane per cascade, so this is exactly full at SHADOW_CASCADES == 4. Overlap and the enable
    /// flag moved to `casc_params` when the two far cascades took the z/w lanes.
    split_far: [f32; 4],
    /// xyz = Lsun (toward the sun), w = 1/shadow_map_size() (PCF texel).
    sun_dir_texel: [f32; 4],
    /// Per-cascade world texel size (world-space bias units), one lane per cascade.
    texel_world: [f32; 4],
    /// x = diffuse cap (0.12), y = fade start, z = fade end, w = debug mode (1 = spec-only).
    combine: [f32; 4],
    /// Runtime graphics scales from the UI (GfxSettings): x = fog density scale (0 = off),
    /// y = sky-reflection gain scale, z = emissive scale.
    /// w = APP TIME for the grass wind phase — NOT a reserved lane. It said "reserved" here for a
    /// while, which is exactly the sort of wrong comment that gets a live lane overwritten.
    gfx: [f32; 4],
    /// x = cascade overlap fraction (0.10), y = enabled (1/0), z = cascade count,
    /// w = volumetric shaft strength (0 = off).
    /// The shader reads the COUNT rather than hardcoding 4, so a future count change needs no WGSL
    /// edit beyond the array length and this struct's assert.
    casc_params: [f32; 4],
}
// #6: LOCK the byte layout — matches `SunShadowUniform` in gpu_draw.wgsl (4×mat4 = 256 + 6×vec4 = 96
// => 352). Changing SHADOW_CASCADES changes this size: update the WGSL twin's array length AND its
// own size assert in the same commit. A silent mismatch here is how the RX6800 device loss happened.
const _: () = assert!(std::mem::size_of::<SunShadowUniform>() == 352);
const _: () = assert!(SHADOW_CASCADES <= 4, "split_far/texel_world carry one lane per cascade");

/// Runtime shadow feature switch + the pack's sun direction (already X-flipped into pack space).
/// Default ON; `enabled=false` (missing sun_dir, EFT_SHADOWS=0, or the UI toggle off) makes the
/// whole pass a strict no-op.
#[derive(Resource)]
struct EftShadowConfig {
    /// Lsun: points TOWARD the sun (light travels along -Lsun). Unit. Y-up sentinel when disabled.
    lsun: Vec3,
    /// EFFECTIVE switch consulted by the extrusion / uniform / shadow node — refreshed every
    /// frame by `sync_gfx_shadow_toggle` = env_enabled AND the UI toggle, gated on sun_ok.
    enabled: bool,
    /// Env ALLOW flag captured at startup: true by default, false only if EFT_SHADOWS=0/false (a
    /// hard dev/perf veto). ANDed with the UI toggle, so BOTH must permit shadows.
    env_enabled: bool,
    /// The pack HAS a valid sun_dir (lsun is real, not the sentinel) — the runtime UI toggle may
    /// enable shadows even when the EFT_SHADOWS env opt-in was off.
    sun_ok: bool,
    /// `EFT_SHADOW_DEBUG=1`: specular-only diagnostic (diffuse cap forced to 0 in the shader).
    debug: bool,
}

/// Refresh the effective shadow switch from the UI settings (extracted GfxSettings) once per
/// frame, BEFORE the frustum extrusion, the uniform upload, and the shadow node consult it.
fn sync_gfx_shadow_toggle(
    config: Option<ResMut<EftShadowConfig>>,
    settings: Option<Res<crate::render::GfxSettings>>,
) {
    if let (Some(mut c), Some(s)) = (config, settings) {
        // Default ON: shadows show whenever the env doesn't veto (env_enabled), the UI toggle is on,
        // and the pack has a sun_dir. Either the env veto (EFT_SHADOWS=0) or the UI toggle disables.
        let eff = c.env_enabled && s.shadows && c.sun_ok;
        if c.enabled != eff {
            c.enabled = eff;
        }
    }
}

/// Bumped whenever GPU instance records are rewritten mid-session (door swings) so the shadow
/// cascade cache re-renders that frame — see `EftShadowCache`.
#[derive(Resource, Default)]
struct EftDynamicNonce(u64);

/// #5b cascade CACHE. The sun is static and the world is static, so a cascade whose fitted
/// view-proj is BIT-IDENTICAL to what its atlas layer already holds does not need re-rendering.
/// Texel snapping makes any real camera motion change the fit (the snap quantum is a shadow
/// texel, ~1–6 cm), so this is precisely a "camera at rest" cache — the dominant state of a
/// map viewer — and it is artifact-free: at rest the camera cull the shadow pass reuses is
/// unchanged too. Measured before the cache: the two 2048² cascades were 9.9 ms of
/// interchange's 18.5 ms overview frame (docs/PERF_BENCHMARKS.md).
///
/// `vp[c]` = the fit the CURRENT atlas content was rendered with (None = layer invalid, e.g.
/// while shadows are disabled); `render[c]` = this frame's instruction to `EftShadowNode`.
/// Invalidation beyond a vp change: door swings (`EftDynamicNonce`), geometry/pack rebuilds
/// (`EftGpuBuffers` change), and any GfxSettings change (LOD sliders alter the draw set).
#[derive(Resource)]
struct EftShadowCache {
    vp: [Option<[[f32; 4]; 4]>; SHADOW_CASCADES],
    render: [bool; SHADOW_CASCADES],
}
impl Default for EftShadowCache {
    fn default() -> Self {
        Self { vp: [None; SHADOW_CASCADES], render: [false; SHADOW_CASCADES] }
    }
}

/// The queued shadow depth pipeline + its group(1) cascade-uniform layout.
#[derive(Resource)]
struct EftShadowPipeline {
    pipeline_id: CachedRenderPipelineId,
    #[allow(dead_code)] // kept for symmetry / potential rebuilds; the bind groups already own it
    cascade_layout: BindGroupLayout,
}

/// group(1) uniform for the NORMAL PREPASS (gpu_prepass.wgsl). Byte-identical to the WGSL
/// `PrepassUniform` (80 bytes: mat4 + vec4) — pinned like every other cross-shader struct here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PrepassUniform {
    /// world -> camera clip (Bevy reverse-z). Column-major Mat4 upload.
    view_proj: [[f32; 4]; 4],
    /// x = bindless albedo array length (descriptor-index clamp in the cutout alpha test — same
    /// upload the shadow pass makes, and for the same AMD-fault reason). yzw pad.
    params: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<PrepassUniform>() == 80);

/// The normal prepass: camera-view geometric normals + roughness (Rgba16Float) over its own 1x
/// Depth32Float, drawn from the SAME culled indirect buffers as the main pass. The enabler for
/// normal-aware SSAO now and SSR later — the forward main pass writes only color, so before this
/// every screen-space effect had to reconstruct normals from depth derivatives.
///
/// Textures are (re)created by `prepare_prepass` whenever the view size changes; consumers (ssao)
/// key their bind-group caches on the view ids, so recreation Just Works on resize.
#[derive(Resource)]
pub(crate) struct EftPrepassResources {
    pipeline_id: CachedRenderPipelineId,
    uniform: Buffer,
    /// group(1) — just the uniform; the targets are attachments, not bindings.
    bind_group: BindGroup,
    #[allow(dead_code)] // keeps the views below valid
    normal_texture: Option<Texture>,
    /// pub(crate): ssao binds this as its normal source.
    pub(crate) normal_view: Option<TextureView>,
    #[allow(dead_code)]
    depth_texture: Option<Texture>,
    /// pub(crate): TAA reprojects from this depth; the pyramid reduces it.
    pub(crate) depth_view: Option<TextureView>,
    size: UVec2,
    /// Set by `prepare_prepass` when this frame actually has a valid camera + targets AND the ssao
    /// consumer is on. The node and ssao both read it, so "prepass off" degrades cleanly to the
    /// old derivative-normal path instead of binding stale textures.
    pub(crate) active: bool,
    /// Phase 1 history: this frame's and the previous frame's clip_from_world (UNJITTERED — TAA
    /// reprojection must not chase the jitter). `prev` is None whenever history is invalid: first
    /// frame, resize, map swap, or the prepass toggling off — consumers reject history on None.
    pub(crate) clip_from_world: [[f32; 4]; 4],
    pub(crate) prev_clip_from_world: Option<[[f32; 4]; 4]>,
}

/// Phase 1 substrate: ONE reverse-z max-reduction depth pyramid over the prepass depth, shared by
/// every hierarchical-depth consumer (SSR Phase 6, Hi-Z Phase 3) — the plan is explicit that they
/// must not each build their own. Gated on those consumers, so it is exactly absent otherwise.
#[derive(Resource)]
pub(crate) struct EftPyramidResources {
    layout: BindGroupLayout,
    copy_pipeline: CachedComputePipelineId,
    reduce_pipeline: CachedComputePipelineId,
    #[allow(dead_code)] // keeps the views valid
    tex: Option<Texture>,
    /// One view per mip (base_mip_level=i, count=1). mip_views[0] is also the SSR/Hi-Z sample view
    /// for the finest level; a whole-chain sampling view is created alongside.
    mip_views: Vec<TextureView>,
    /// Full-chain sampling view (all mips) for consumers.
    pub(crate) sample_view: Option<TextureView>,
    /// Per-mip bind groups: [0] = copy (prepass depth -> mip0), [i>=1] = reduce (mip i-1 -> mip i).
    bind_groups: Vec<BindGroup>,
    size: UVec2,
    mips: u32,
    pub(crate) active: bool,
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
    /// Optional per-map SH GI intensity multiplier (the shader multiplies the sampled
    /// irradiance/env by it via `vol_min.w`). Data-driven so a dark bake (e.g. Interchange) can be
    /// lifted a couple of stops without a Rust rebuild. Absent -> 1.0 (unchanged behaviour).
    #[serde(default)]
    gi_intensity: Option<f32>,
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
    /// Per-map SH GI intensity (shader `vol_min.w`); from the sidecar's `gi_intensity`, else 1.0.
    gi_intensity: f32,
    /// mean(layer-1 c0 luma) / mean(top-layer c0 luma) over sky-lit probes — the out-of-volume
    /// redirect samples the TOP layer (clean sky) but ground sees a slightly dimmer dome
    /// (horizon occlusion), so redirected samples are scaled by this (shader `dims.w`).
    ground_over_top: f32,
    tex_r: Vec<u8>,
    tex_g: Vec<u8>,
    tex_b: Vec<u8>,
    /// PER-PROBE VALIDITY (Unity APV): one u8 per probe, probe-major, 255 = open space, 0 = buried
    /// in geometry. Uploaded as an R8Unorm 3D texture and used to weight the shader's trilinear
    /// taps, so a probe sealed inside a wall cannot bleed onto the surfaces around it. All-255
    /// (= "everything valid") when the pack predates `volume_valid.bin`, which reproduces the old
    /// behaviour exactly.
    valid: Vec<u8>,
}

impl ShVolumeCpu {
    /// 1x1x1 fallback used when the pack ships no volume sidecar: c0 = 1.0 (half), c1..c3 = 0,
    /// so E/π = 0.282095 -> a flat ~0.28 gray ambient (roughly the old `ambient` constant),
    /// keeping group(3) valid rather than crashing the draw on a missing bind group.
    fn dummy() -> Self {
        // half(1.0) = 0x3C00, half(0.0) = 0x0000 (LE bytes). texel = (c0=1, c1=0, c2=0, c3=0).
        let texel: [u8; 8] = [0x00, 0x3C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        Self {
            ground_over_top: 1.0,
            dims: [1, 1, 1],
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
            spacing: [1.0, 1.0, 1.0], // single probe: grid clamps to 0, any nonzero step is inert
            gi_intensity: 1.0,
            tex_r: texel.to_vec(),
            tex_g: texel.to_vec(),
            tex_b: texel.to_vec(),
            valid: vec![255u8], // the single fallback probe is "open"
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
    let meta_path = &pack.resolve_path(pack.manifest.sidecars.volume_meta.as_deref()?);
    let bin_path = &pack.resolve_path(pack.manifest.sidecars.volume.as_deref()?);

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
    let read_half = |base: usize, h: usize| -> f32 {
        let o = base + h * 2;
        half::f16::from_le_bytes([bin[o], bin[o + 1]]).to_f32()
    };
    let (mut g_sum, mut g_n, mut t_sum, mut t_n) = (0f64, 0u32, 0f64, 0u32);
    for pi in 0..n_probes {
        let base = pi * 24;
        // ground-adjacent (layer 1) vs top-layer mean c0 luma of sky-lit probes, for the
        // out-of-volume redirect scale (see ShVolumeCpu::ground_over_top).
        let yl = (pi / nx as usize) % ny as usize;
        if yl == 1.min(ny as usize - 1) || yl == ny as usize - 1 {
            let l = 0.2126 * read_half(base, 0) + 0.7152 * read_half(base, 1) + 0.0722 * read_half(base, 2);
            if l.is_finite() && l > 0.05 {
                if yl == ny as usize - 1 { t_sum += l as f64; t_n += 1; } else { g_sum += l as f64; g_n += 1; }
            }
        }
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
    // Per-map GI intensity (shader vol_min.w). Reject a non-finite / negative sidecar value so a
    // bad bake can't NaN the whole GI term; absent -> 1.0 (behaviour unchanged for older packs).
    let gi_intensity = meta
        .gi_intensity
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(1.0);

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
    let ground_over_top = if g_n > 0 && t_n > 0 {
        (((g_sum / g_n as f64) / (t_sum / t_n as f64)) as f32).clamp(0.5, 1.5)
    } else {
        1.0
    };
    // Per-probe VALIDITY (Unity APV). Optional sidecar: packs baked before it simply get all-255,
    // which makes every tap fully weighted — i.e. byte-identical behaviour to before.
    let valid = std::fs::read(pack.resolve_path("volume_valid.bin"))
        .ok()
        .filter(|v: &Vec<u8>| {
            if v.len() == n_probes {
                true
            } else {
                warn!(
                    "SH-GI: volume_valid.bin is {} bytes, expected {n_probes} (one per probe) —                      ignoring it and treating every probe as valid",
                    v.len()
                );
                false
            }
        })
        .unwrap_or_else(|| vec![255u8; n_probes]);
    let n_invalid = valid.iter().filter(|&&v| v < 128).count();
    info!(
        "SH-GI: probe validity {} ({} of {} probes invalid/in-geometry)",
        if n_invalid > 0 || valid.iter().any(|&v| v != 255) { "loaded" } else { "absent (all valid)" },
        n_invalid,
        n_probes
    );
    Some(ShVolumeCpu {
        ground_over_top,
        dims: meta.dims,
        min: meta.min,
        max: meta.max,
        spacing,
        gi_intensity,
        tex_r,
        tex_g,
        tex_b,
        valid,
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

/// FNV-1a 64-bit over a byte slice — the geometry byte-identity gate (EFT_GEOM_SHA=1). Not crypto;
/// a mismatch between the old and the fused-encoder paths on the same pack is all we need to catch.
#[inline]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// True when EFT_GEOM_SHA=1 — logs FNV hashes of the final vertex/index byte streams so an old-vs-new
/// build on the same pack can be compared for byte-exactness. Cheap check, read once.
#[inline]
fn geom_hash_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("EFT_GEOM_SHA").map(|v| v.trim() == "1").unwrap_or(false))
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
    /// Index bytes ready for upload, in `index_u16`'s width. Indices are LOCAL to each mesh
    /// (base_vertex offsets them), so when every mesh fits under 65,536 vertices the whole buffer
    /// can be u16 and halves in size (341 MiB on streets). Built from the u32 staging vec, which
    /// is dropped before upload so the narrow copy never coexists with the wide one for long.
    index_bytes: Vec<u8>,
    /// True when `index_bytes` is u16 (else u32). Drives `IndexFormat` at draw time.
    index_u16: bool,
    index_count: usize,
    instances: Vec<InstanceGpuRecord>,
    /// The contiguous mesh-SLOT range the grass kinds occupy, `[start, end)`. Recorded because the
    /// shadow pass must skip it, and it cannot be derived as `mesh_count - n_kinds`: the SEA quad is
    /// appended AFTER grass on coastal maps, so a tail-relative guess would skip the sea instead.
    grass_mesh_range: Option<(u32, u32)>,
    /// How many of `instances` are GRASS. Grass is the only source that can push the instance
    /// array past a storage BINDING limit on its own, so the render world reports it by name when
    /// the buffer is oversized rather than making the reader guess.
    grass_instances: usize,
    /// First grass record in `instances`. Grass is built as one contiguous run; synthetic sea
    /// instances may follow it. The render upload can therefore omit the run without rebuilding
    /// the pack, then shift only later mesh bases. This is the seam that makes Low quality avoid
    /// Woods' ~883 MiB grass SSBO instead of merely culling it after upload.
    grass_instance_base: usize,
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
    /// Vert-Paint 3-layer splat entries (MAT_FLAG_VP materials; `GpuMaterial._pad2` indexes this).
    vp_table: Vec<VpGpu>,
    /// Bindless albedo indices that are terrain CONTROL maps (blend weights = data, not color):
    /// uploaded LINEAR instead of sRGB so the weights aren't gamma-warped toward one layer.
    ctrl_tex_linear: std::collections::HashSet<u32>,
    /// Bindless albedo indices whose RESOLUTION is authoritative and must never be reduced by the
    /// texture-quality setting (terrain blend weights: one texel drives one patch of splat).
    /// SEPARATE from `ctrl_tex_linear`, which only means "upload linear, don't BC-compress":
    /// vert-paint coverage masks and parallax heights are linear DATA too, but they are smooth
    /// masks that downscale fine, and lumping them in here kept them at full res on every quality
    /// setting (the same class of bug as the Standard path ignoring TEX_MIP_SKIP).
    no_downscale: std::collections::HashSet<u32>,
    /// Meshes with >=1 BLEND submesh: (mesh index, first-instance world center, pass mask).
    /// The mask separates depth-writing SoftCutout coverage, surface overlays, and true
    /// transparency so coplanar roads never share glass's render state.
    blend_meshes: Vec<(u32, Vec<[f32; 3]>, u32)>,
    /// #5 shadows: sun direction (points TOWARD the sun) X-flipped into pack space, or `None` when
    /// the volume sidecar has no valid `sun_dir` (the shadow feature then disables itself; no
    /// invented fallback direction). Mirrors standard.rs's exact access + flip.
    sun_dir: Option<Vec3>,
    /// Realtime point/spot light set + static world CSR grid, built once per map. Uploaded to
    /// group(3) bindings 8/9/10 in `prepare_gpu_buffers`. Always present (a 1-cell/0-light dummy
    /// when the pack uses the baked SH path or ships no lights).
    light_grid: LightGridCpu,
    instance_total: u32,
    mesh_count: u32,
    /// Swing doors (gamedata) the viewer can open on click. Matched to their GPU instance +
    /// animated in `prepare_gpu_buffers` / `animate_doors`.
    doors: Vec<crate::eftpack::LevelDoor>,
    /// Pack mesh NAME per `mesh_meta` slot (an instance's `ids[0]`). Only doors use it — to
    /// resolve their part list by the game's own mesh names. Synthetic meshes (grass, sea)
    /// carry an empty name.
    mesh_names: Vec<String>,
    /// LODGroup id per GPU instance (-1 = ungrouped), parallel to `instances`. Doors use it to
    /// animate EVERY shell of a leaf, not just the one that happens to be resident (AUDIT #4).
    inst_lod_group: Vec<i32>,
    /// B1 distance-LOD: per-lod_group reference center (already conjugated by the emitter), indexed
    /// by `lod_group` (== manifest.lod_groups order). `cs_cull` mode 1 measures the shell-switch
    /// distance from THIS shared point (via the group id packed in `ids.z` bits 13+), not each
    /// shell's own bounding-sphere centroid, so every renderer/shell in a group switches together
    /// (no per-boundary double-draw/hole from mismatched centroids). >=1 element (never zero-sized).
    lod_centers: Vec<[f32; 4]>,
    /// Loot-glow model match, AUTHORITATIVE: (gamedata container index, model-center world pos,
    /// GPU instances). Joined by PREFAB ANCESTRY — the container's folded transform chain
    /// (gamedata `tf`) intersected with each instance's shipped `par`/`par2` at the same level —
    /// never by name or radius (which lit decorative same-mesh neighbours and missed
    /// offset-pivot parts). Cloned into the persistent main-world `loot::LootModelIndex` at
    /// insert time — this blob itself is dropped after upload. Empty when either side predates
    /// the ancestry capture: no guess, no glow, markers stay boxes.
    loot_models: Vec<(u32, [f32; 3], Vec<u32>)>,
}

/// The repacked CPU geometry blob + the `MapEpoch` it was built for. `prepare_gpu_buffers` builds
/// GPU buffers ONLY when `.1 == MapEpoch`, so a fast double-swap can't rebuild from the previous
/// map's still-resident blob (the epoch reaches the render world one frame before the matching blob).
#[derive(Resource, Clone)]
pub struct ExtractedCpuData(Arc<CpuData>, u64);

impl ExtractResource for ExtractedCpuData {
    type Source = ExtractedCpuData;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Cross-world GPU map-build state. The progress flag is set the moment a new build begins
/// (main world: `build_cpu_data` / `poll_map_load`), cleared FALSE when `prepare_gpu_buffers`
/// finishes uploading every texture and inserts `EftGpuBuffers` (render world). The SAME `Arc` is
/// inserted into BOTH the main app and the render sub-app, so the render world can signal the main
/// world's `map_loading_indicator` to keep showing the "Loading…" toast until the map is actually
/// on-screen — not just until the .eftpack FILE finished loading (which is all `PendingMapLoad`
/// tracks). Allocation preflight errors use the same state to reach the main-world error panel.
struct GpuLoadState {
    in_progress: AtomicBool,
    error: Mutex<Option<String>>,
}

#[derive(Resource, Clone)]
pub struct GpuLoadSignal(Arc<GpuLoadState>);

impl Default for GpuLoadSignal {
    fn default() -> Self {
        Self(Arc::new(GpuLoadState {
            in_progress: AtomicBool::new(false),
            error: Mutex::new(None),
        }))
    }
}

impl GpuLoadSignal {
    /// True while a map's GPU build is still running (textures uploading / buffers not yet built).
    pub fn in_progress(&self) -> bool {
        self.0.in_progress.load(Ordering::Relaxed)
    }
    /// Latch the flag TRUE — a new map's GPU build is starting. Called by `poll_map_load` the moment
    /// it applies a finished file load (closing the 1-frame gap before `build_cpu_data` runs) and by
    /// `build_cpu_data` itself.
    pub fn begin(&self) {
        self.clear_error();
        self.0.in_progress.store(true, Ordering::Relaxed);
    }
    fn set(&self, v: bool) {
        self.0.in_progress.store(v, Ordering::Relaxed);
    }
    /// Surface a render-device allocation failure to the main-world error panel. Continuing after
    /// an invalid wgpu buffer/binding request only poisons the encoder and produces unrelated
    /// validation cascades, so allocation preflight terminates this map build here.
    fn fail(&self, message: String) {
        *self.0.error.lock().unwrap_or_else(|p| p.into_inner()) = Some(message);
        self.set(false);
    }
    pub fn error(&self) -> Option<String> {
        self.0.error.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    pub fn clear_error(&self) {
        *self.0.error.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// Shared "the real render device can't do GPU-driven" flag (finding 6). The preflight probe in
/// `render::gpu_driven_supported` is surface-less and can disagree with the device Bevy actually
/// creates (hybrid-adapter mismatch): if `init_gpu_pipelines` then finds the required indirect /
/// bindless features missing it disables the whole path -> EMPTY view. Instead of leaving that empty
/// view, the render world sets this flag; the SAME `Arc` lives in the main world where
/// `gpu_fallback_relaunch` reads it and relaunches into the M0 instanced path (honest geometry).
#[derive(Resource, Clone)]
pub struct GpuFallback(pub Arc<AtomicBool>);

impl Default for GpuFallback {
    fn default() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
}

/// Main-world system: when the render world signals the GPU-driven path is unsupported on the real
/// device (`GpuFallback`), relaunch the process into the M0 instanced path instead of sitting on a
/// blank view. Only fires when the path was AUTO-selected (no explicit `EFT_RENDER` override — an
/// explicit `EFT_RENDER=gpu` is the user's choice, so we log + leave it). One-shot via the Local.
pub fn gpu_fallback_relaunch(
    fallback: Option<Res<GpuFallback>>,
    mut exit: MessageWriter<AppExit>,
    mut fired: Local<bool>,
) {
    if *fired {
        return;
    }
    let Some(fallback) = fallback else { return };
    if !fallback.0.load(Ordering::SeqCst) {
        return;
    }
    *fired = true; // whatever we decide, don't re-evaluate every frame
    // Respect an explicit override: if the user forced EFT_RENDER we don't second-guess them.
    if std::env::var("EFT_RENDER").map(|v| !v.trim().is_empty()).unwrap_or(false) {
        error!(
            "gpu-driven: the render device lacks the required features, but EFT_RENDER is set \
             explicitly - leaving the view as-is. Re-run with EFT_RENDER=m0 for the instanced path."
        );
        return;
    }
    match std::env::current_exe() {
        Ok(exe) => {
            let mut cmd = std::process::Command::new(exe);
            for a in std::env::args().skip(1) {
                cmd.arg(a); // preserve the pack argv; EFT_RENDER=m0 below overrides any render token
            }
            cmd.env("EFT_RENDER", "m0");
            match cmd.spawn() {
                Ok(_) => {
                    eprintln!(
                        "gpu-driven unsupported on the real device - relaunching into the M0 \
                         instanced path (honest geometry instead of an empty view)"
                    );
                    exit.write(AppExit::Success);
                }
                Err(e) => error!("gpu fallback: relaunch into M0 failed: {e}"),
            }
        }
        Err(e) => error!("gpu fallback: current_exe failed: {e}"),
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
        // Shared load-progress flag: the SAME Arc lives in the main app (read by the loading
        // indicator + written when a build starts) and the render sub-app (cleared when the GPU
        // build finishes). Insert into the MAIN app here; the render sub-app clone is inserted below.
        let load_signal = GpuLoadSignal::default();
        app.insert_resource(load_signal.clone());
        // Shared GPU-unsupported flag (finding 6): render world sets it, main world relaunches M0.
        let fallback = GpuFallback::default();
        app.insert_resource(fallback.clone());
        app.add_systems(Update, gpu_fallback_relaunch);
        app.add_plugins((
            ExtractComponentPlugin::<GpuDrivenTag>::default(),
            ExtractComponentPlugin::<CullCamera>::default(),
            ExtractResourcePlugin::<ExtractedCpuData>::default(),
            // The map epoch reaches the render world so `reset_gpu_map_if_epoch_changed` can tear
            // down the old pack's GPU state on a swap.
            ExtractResourcePlugin::<super::MapEpoch>::default(),
            // Door click-to-open: the pick's world point crosses into the render world.
            ExtractResourcePlugin::<DoorClick>::default(),
            // Loot glow: the overlay's visible container->instance set crosses per frame (slim;
            // cloned only when its generation bumps).
            ExtractResourcePlugin::<crate::loot::LootGlowState>::default(),
        ))
        .init_resource::<DoorClick>()
        .init_resource::<crate::loot::LootGlowState>()
        // The CPU staging build re-runs on every map epoch (the initial insert included) so an
        // in-place .eftpack swap rebuilds the blob; the render-world reset then rebuilds the GPU
        // side. Step 3: `kick_cpu_build` spawns the heavy build onto the AsyncComputeTaskPool (so
        // the ~0.6–1.3 s work no longer freezes the main thread); `poll_cpu_build` applies the
        // result when it lands, dropping any stale (superseded-epoch) blob. (Was one synchronous
        // `build_cpu_data` system; was `Startup` before that, which only ran for the first pack.)
        .add_systems(Update, kick_cpu_build.run_if(resource_changed::<super::MapEpoch>))
        .add_systems(Update, poll_cpu_build.run_if(resource_exists::<PendingCpuBuild>))
        .add_systems(Update, free_cpu_staging);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(load_signal)
            .insert_resource(fallback) // render world raises it in init_gpu_pipelines on a guard miss
            .add_render_command::<Transparent3d, DrawGpuDriven>()
            .init_resource::<SpecializedRenderPipelines<EftDrawPipeline>>()
            .init_resource::<EftShadowCache>()
            .init_resource::<EftDynamicNonce>()
            .add_systems(RenderStartup, init_gpu_pipelines)
            .add_systems(
                Render,
                (
                    // Before prepare: on a NEW MapEpoch, drop the previous pack's per-map GPU
                    // resources + null the bindless layouts + invalidate the pipeline cache, so
                    // prepare_gpu_buffers rebuilds everything for the new pack.
                    reset_gpu_map_if_epoch_changed
                        .in_set(RenderSystems::PrepareResources)
                        .before(prepare_gpu_buffers),
                    prepare_gpu_buffers.in_set(RenderSystems::PrepareResources),
                    // Loot glow: rewrite the per-instance highlight lane when the overlay's
                    // visible set changed (toggle flip, min-value change, marker respawn).
                    prepare_loot_glow
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    // SSAO AO lane: (re)create the target, then swap the draw bind group's AO
                    // binding between it and the white fallback. Registered HERE (not in
                    // SsaoPlugin) so the ordering can name the private `prepare_gpu_buffers`.
                    (super::ssao::prepare_ao_target, sync_draw_bg_ao)
                        .chain()
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    // Runtime UI shadow toggle: refresh the effective switch BEFORE the frustum
                    // extrusion + uniform upload + shadow node read it this frame.
                    sync_gfx_shadow_toggle
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers)
                        .before(upload_frustum)
                        .before(prepare_shadow_uniforms),
                    upload_frustum
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    // #5 shadows: fit + upload the cascade matrices AFTER the buffers exist (the
                    // shadow resources are built in prepare_gpu_buffers).
                    prepare_shadow_uniforms
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    // Normal prepass: camera matrix + (re)size the normal/depth targets. After the
                    // buffers exist for the same reason the shadow prepare is.
                    prepare_prepass
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    prepare_pyramid
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_prepass),
                    // Live lighting sliders: base x GfxSettings multipliers into the LightGrid
                    // uniform (48 B/frame; byte-identical at the default multipliers).
                    update_light_uniform
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    // Live POWER SWITCH toggle: re-upload the light buffer with unpowered groups
                    // zeroed when GfxSettings.light_groups changes (no-op until a switch is flipped).
                    update_light_power
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers),
                    // Door click-to-open: toggle the nearest door + ease in-flight swings, mutating
                    // the matched instance record (no-op until a door is clicked).
                    // Ordered before the shadow prepare so a swing invalidates the cascade
                    // cache the SAME frame (EftDynamicNonce).
                    animate_doors
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_gpu_buffers)
                        .before(prepare_shadow_uniforms),
                    queue_gpu_driven.in_set(RenderSystems::QueueMeshes),
                ),
            )
            // #5: EftCull (writes visible/indirect) -> EftShadow (reads them, writes the depth
            // atlas) -> StartMainPass (main draw samples the atlas). The shadow node NEVER re-culls
            // or resets the shared stream.
            .add_render_graph_node::<EftCullNode>(Core3d, EftCullLabel)
            .add_render_graph_node::<EftShadowNode>(Core3d, EftShadowLabel)
            .add_render_graph_node::<EftPrepassNode>(Core3d, EftPrepassLabel)
            .add_render_graph_node::<EftPyramidNode>(Core3d, EftPyramidLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    EftCullLabel,
                    EftShadowLabel,
                    EftPrepassLabel,
                    EftPyramidLabel,
                    Node3d::StartMainPass,
                ),
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
    load_signal: Option<Res<GpuLoadSignal>>,
) {
    if cpu.is_none() {
        return;
    }
    // The GPU build now streams textures across MANY frames (async), reading the extracted staging
    // blob every frame. Do NOT drop it while a build is in progress — hold the countdown until the
    // render world signals the map is on-screen, else prepare_gpu_buffers loses `cpu` mid-build and
    // the load stalls forever. (Originally the whole build fit in one frame, so 4 frames sufficed.)
    if load_signal.as_ref().map(|s| s.in_progress()).unwrap_or(false) {
        *frames = 0;
        return;
    }
    // A NEW blob (in-place map swap re-inserts it → "added" again) restarts the countdown, so the
    // new map's staging survives its ~4-frame upload window instead of being dropped next frame by
    // the counter left stuck at 4 from the previous map.
    if cpu.as_ref().is_some_and(|c| c.is_added()) {
        *frames = 0;
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

/// Pure CPU staging build (NO ECS/Bevy access) so it can run on the `AsyncComputeTaskPool` off
/// the main thread (Step 3). Parses the pack + selected LOD into the GPU-ready `CpuData` blob.
/// Returns `None` when there is nothing to draw (empty pack, or a failed bounding-sphere pass);
/// the caller (`poll_cpu_build`) clears the loading flag in that case. The heavy work here — the
/// fused geometry encode, the material table, grass, SH/light-grid load — is exactly what used to
/// stall the main thread for ~0.6–1.3 s per map load.
fn compute_cpu_blob(pack: &Pack, lod: i32) -> Option<CpuData> {
    let build_t0 = std::time::Instant::now(); // STALL INSTRUMENTATION (off-thread now)
    // DISTANCE-LOD: a pack that ships more than one shell per group ("multi-LOD") packs EVERY shell
    // and lets the GPU cull select per-frame (ids.w window + ids.z bits); a lean pack keeps the old
    // single-shell CPU selection (byte-identical). See LOD_DISTANCE_PLAN.md.
    let multi_lod = pack.default_lod_mask.iter().any(|&d| !d);
    let by_mesh = if multi_lod {
        pack.instances_by_mesh() // all shells; GPU selects
    } else {
        pack.instances_by_mesh_for_lod(lod) // lean: one shell per group (unchanged)
    };
    let t_bymesh = build_t0.elapsed(); // phase: instance-by-mesh grouping
    // Per-instance LOD encode (multi-LOD only): (ids.z extra bits, ids.w f16 window). `ids.z` bit8 =
    // is-default-shell, bits9..12 = lod_index (clamped 0..15); `ids.w` = pack_f16(near', far') where the
    // runtime distance window is (near'*proj11*bias, far'*proj11*bias] (0 = sentinel, always drawn).
    // `lod_present` = per-group SORTED present lod_index set: a shell's near boundary meets the
    // *previous present* shell's far, so an internal gap (present {0,2}) can't leave a coverage hole.
    let lod_present = if multi_lod {
        pack.group_present_lods()
    } else {
        std::collections::HashMap::new()
    };
    let lod_encode = |i: u32| -> (u32, u32) {
        let inst = &pack.instances[i as usize];
        let idx = inst.lod_index.max(0);
        // B6: lod_index rides a 4-bit field; a group with >15 LODs would wrap (LOD16 -> shell 0).
        // EFT LODGroups are ~4-8 levels so this never fires, but CLAMP (don't wrap) as a backstop and
        // keep bits 13+ free for the group id (finding #1's lod_centers lookup).
        debug_assert!(idx <= 15, "lod_index {idx} exceeds the 4-bit ids.z field");
        let z = ((pack.is_default_lod(i as usize) as u32) << 8) | (((idx as u32).min(15)) << 9);
        if inst.lod_group < 0 {
            return (z, 0); // ungrouped: always drawn
        }
        // B1: the group id rides ids.z bits 13-31 (19 bits) so cs_cull can look up the group's
        // shared reference center. A pack with >=2^19 groups can't encode it — fall back to the
        // always-draw sentinel (never happens; interchange, the largest, has ~77k groups).
        if (inst.lod_group as u32) >= (1u32 << 19) {
            return (z, 0);
        }
        let Some(present) = lod_present.get(&inst.lod_group) else {
            return (z, 0);
        };
        if present.len() <= 1 {
            return (z, 0); // single present shell: always drawn (reserve's LOD2 windows land here)
        }
        let Some(g) = pack.manifest.lod_groups.get(inst.lod_group as usize) else {
            return (z, 0);
        };
        // far distance boundary of shell `lvl` (÷proj11): d beyond which the shell is too small.
        let far = |lvl: i32| -> f32 {
            let h = g.srh.get(lvl as usize).copied().unwrap_or(0.0);
            if h > 1.0e-6 {
                g.size / (2.0 * h)
            } else {
                f32::INFINITY
            }
        };
        // near = far of the previous PRESENT shell (not idx-1): with an internal gap the next present
        // shell picks up exactly where the previous one ended, so no distance band is left undrawn.
        let prev_present = present.iter().rev().find(|&&p| p < idx).copied();
        let near = prev_present.map(far).unwrap_or(0.0);
        let is_coarsest = idx >= *present.last().unwrap();
        // Force-shell coverage window (cull mode 2), ids.z bits 1..4 = lo, bits 5..7 = hi: the
        // instance draws when the forced shell F lands in [lo, hi] — its own level, widened down
        // to 0 on the group's finest-present shell and up to 7 on its coarsest, ending where the
        // next PRESENT shell begins. Forcing a level a group doesn't ship thus falls back to its
        // nearest present shell instead of vanishing — the CPU selector's clamp rule
        // (eftpack.rs::keep_lod) that the shader's plain index-equality test lost. Bit 0 stays
        // clear: the grass checks are exact `ids.z == 1` and must never alias.
        let min_present = *present.first().unwrap();
        let f_lo = if idx <= min_present { 0u32 } else { (idx as u32).min(15) };
        let f_hi = if is_coarsest {
            7u32
        } else {
            let next = present.iter().find(|&&p| p > idx).copied().unwrap_or(idx + 1);
            ((next.max(idx + 1) - 1) as u32).min(7)
        };
        let z = z | (f_lo << 1) | (f_hi << 5);
        let far_b = if is_coarsest {
            if g.last_is_billboard && g.cull_h > 1.0e-6 {
                g.size / (2.0 * g.cull_h) // billboard groups cull past their last threshold (no billboard geom ships)
            } else {
                f32::INFINITY
            }
        } else {
            far(idx)
        };
        if !(near < far_b) || !near.is_finite() {
            return (z, 0); // degenerate/inverted (bad srh) -> always draw
        }
        let a16 = half::f16::from_f32(near.max(0.0)).to_bits() as u32;
        let b16 = half::f16::from_f32(far_b.min(65504.0)).to_bits() as u32; // f16 max ~= +inf at any real d
        let w = (b16 << 16) | a16;
        // B5: a genuine window must never quantize to the 0 sentinel (would make a real shell always
        // draw). Only possible if both halves underflow f16 (sub-nanometer geometry) — bump to the
        // smallest representable far so ids.w stays non-zero. No effect on any realistic pack.
        // B1: OR the group id into ids.z bits 13+ so cs_cull mode-1 measures distance from the
        // group's shared reference center (lod_centers[group]) instead of this shell's own centroid.
        let z = z | ((inst.lod_group as u32) << 13);
        (z, if w == 0 { 1 } else { w })
    };
    let local_spheres = match pack.bounding_spheres() {
        Ok(s) => s,
        Err(e) => {
            error!("gpu-driven: bounding_spheres failed: {e:#}");
            return None; // poll_cpu_build clears the loading flag on a None result
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
    // DATA textures (terrain control maps, vp heights masks): uploaded LINEAR, not sRGB —
    // the sRGB decode would gamma-warp blend weights. Declared here (not in the terrain block)
    // because the vp material loop below also registers its heights masks into it.
    let mut ctrl_tex_linear: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut no_downscale: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // Vert-Paint 3-layer splat table (one entry per MAT_FLAG_VP material; GpuMaterial._pad2 indexes it).
    let mut vp_table: Vec<VpGpu> = Vec::new();
    // Pack-wide green-flip convention (DirectX Y-down): OR'd with each material's own flag.
    let conv_green_flip = pack.manifest.conventions.normal_map_green_flip;

    // Pre-pass: which TEXTURED-water materials are STRETCHED floor decals (matte wet-ground / tire
    // marks) vs real reflective puddles. The `Water Deferred Decal` shader serves both; the ONLY
    // discriminator is world-meters-per-texture-repeat — a puddle maps its texture at a few m/repeat,
    // a facility-floor / wet-asphalt / tire-trail decal at tens-to-hundreds. Measured once from the
    // geometry (submesh local vertex-span / uv-span), map-agnostically. `MAT_FLAG_WATER_MATTE` on the
    // stretched ones tells the shader to drop the mirror + sun glint.
    const WATER_MATTE_MPR: f32 = 40.0; // meters/texture-repeat; puddles <=~22, floor decals >=~60 on lighthouse
    let max_mat_id = pack.materials.iter().map(|m| m.id).max().map_or(0usize, |m| m as usize);
    let mut water_tex = vec![false; max_mat_id + 1];
    for m in &pack.materials {
        if m.role == "water" && m.albedo.is_some() {
            water_tex[m.id as usize] = true;
        }
    }
    let mut stretched_water = vec![false; max_mat_id + 1];
    if water_tex.iter().any(|&b| b) {
        for me in &pack.manifest.meshes {
            if !me.submeshes.iter().any(|sm| water_tex.get(sm.material_id as usize).copied().unwrap_or(false)) {
                continue;
            }
            let geom = match pack.mesh_geom(me) {
                Ok(g) => g,
                Err(_) => continue,
            };
            for sm in &me.submeshes {
                if !water_tex.get(sm.material_id as usize).copied().unwrap_or(false) {
                    continue;
                }
                let (mut pmin, mut pmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                let (mut umin, mut umax) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
                let s0 = sm.idx_start as usize;
                let s1 = (s0 + sm.idx_count as usize).min(geom.indices.len());
                for &vi in &geom.indices[s0..s1] {
                    let vi = vi as usize;
                    if let Some(p) = geom.positions.get(vi) {
                        let v = Vec3::from(*p);
                        pmin = pmin.min(v);
                        pmax = pmax.max(v);
                    }
                    if let Some(uv) = geom.uvs.get(vi) {
                        umin[0] = umin[0].min(uv[0]);
                        umin[1] = umin[1].min(uv[1]);
                        umax[0] = umax[0].max(uv[0]);
                        umax[1] = umax[1].max(uv[1]);
                    }
                }
                if !pmin.is_finite() {
                    continue;
                }
                let span = (pmax - pmin).length();
                let uv_rep = (umax[0] - umin[0]).max(umax[1] - umin[1]).max(1.0e-3);
                if span / uv_rep > WATER_MATTE_MPR {
                    stretched_water[sm.material_id as usize] = true;
                }
            }
        }
    }

    // Per-albedo memo for the glass alpha-semantics probe below: glass shares atlas textures
    // heavily (streets: 489 glass materials over a handful of atlases), so each PNG is probed once.
    let mut glass_mask_memo: std::collections::HashMap<String, bool> = Default::default();
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
        // NOTE: water is EXCLUDED here (its alphaMode is BLEND in the pack) — the water block
        // below decides blend vs opaque by whether it's a textured puddle or deep water.
        if mat.role == "decal"
            || mat.role == "glass"
            || (mat.alpha_mode == "BLEND" && mat.role != "water")
        {
            flags |= MAT_FLAG_BLEND;
        }
        if mat.role == "decal" {
            flags |= MAT_FLAG_DECAL;
        }
        // Legacy Transparent/Reflective/Specular glass: the game's OWN semantics, captured at
        // extraction (glassTRS + _ReflectColor/_SpecColor/_Shininess). Supersedes both the RFA
        // rule and the per-texture mask probe below — tex.a is transparency for the whole family.
        let glass_trs = mat.glass_trs && mat.role == "glass";
        if glass_trs {
            flags |= MAT_FLAG_GLASS_TRS;
        }
        // Glass alpha semantics, decided per TEXTURE (LEGACY PACKS ONLY — a glassTRS capture is
        // authoritative and skips the probe): shard atlases pack COVERAGE in tex.a, nearly all
        // other glass packs SMOOTHNESS there (streets: 479/489). Both live on the same game
        // shader (Transparent Reflective Specular), so only the texture can tell them apart —
        // ground_zero's broken-glass atlas is 67% fully-transparent texels, while smoothness panes
        // are nowhere near zero anywhere. Probe once per texture, memoized.
        let glass_alpha_mask = !glass_trs
            && mat.role == "glass"
            && mat
                .albedo
                .as_deref()
                .map(|p| {
                    *glass_mask_memo
                        .entry(p.to_string())
                        .or_insert_with(|| glass_alpha_is_mask(p))
                })
                .unwrap_or(false);
        if glass_alpha_mask {
            // Coverage-mask glass: tex.a gates EVERY lighting term, including the additive
            // reflection clear panes keep outside their alpha — the empty atlas area must render
            // as nothing, not a ghost pane (nor RFA's constant-tint.a solid dark pane).
            flags |= MAT_FLAG_GLASS_MASK;
        }
        // Per-pixel roughness from the albedo alpha (Unity Standard smoothness-in-alpha).
        // Opaque AND glass (cutout alpha is coverage). 82% of materials carry this — without it
        // everything specular-shades at one constant roughness. For GLASS the flag additionally
        // switches the shader's coverage source to tint.a alone: multiplying smoothness into
        // opacity painted the pattern as opacity blotches (the "shattered" dusty retail panes).
        if mat.roughness_from_albedo_alpha
            && (mat.role == "opaque" || mat.role == "glass")
            && !glass_alpha_mask
        {
            flags |= MAT_FLAG_RFA;
        }
        // M3b2 SoftCutout / water classification. The Vert-Paint SoftCutout family (Custom/Vert
        // Paint SoftCutout Decal) is identified by the `vp.softCutout` param triple — its BLEND
        // coverage is COLOR_0.a modulated by these params, NOT tex.a (which is smoothness here).
        // Water/mirror surfaces (role=="water") had (mostly) no usable albedo and fell back to a
        // flat WHITE tint; they get a dark wet sheen instead. Both classes ALSO blend (force
        // MAT_FLAG_BLEND even for the 16 SoftCutout materials the extractor marked OPAQUE, so
        // they feather in the P2 pass instead of hard-slabbing in P1).
        // SoftCutout is the "Custom/Vert Paint SoftCutout DECAL" shader ONLY — role=decal
        // (RenderType Transparent; _AlphaStrength 1.3/1.7). It feathers into terrain via COLOR_0
        // coverage in the BLEND pass. The "Vert Paint Shader SOLID" variant shares the softCutout
        // PARAM triple but is an OPAQUE 3-layer splat with NO alpha gate: force-blending it made
        // coverage clamp to 0 (astr=0) -> whole courtyard/ground slabs rendered INVISIBLE (the
        // ground_zero "yellow cube" was sky/bloom through the hole). Gate on role=="decal"; the
        // COLOR_0 coverage path owns a real decal's visibility, so clear its hard cutout too.
        let vp_params = softcutout_params(&mat.vp);
        if vp_params.is_some() && mat.role == "decal" {
            flags |= MAT_FLAG_SOFTCUTOUT | MAT_FLAG_BLEND;
            flags &= !MAT_FLAG_CUTOUT;
        } else if vp_params.is_some() {
            // Vert-Paint SOLID splat that the assembler classified cutout/MASK (streets ships 160
            // ground/courtyard slabs like that): its tex.a is SMOOTHNESS, so the opaque-pass
            // alpha-test discarded most of each slab — see-through rectangular holes in the park
            // ground with only the high-smoothness chunks surviving. Same principle as the decal
            // gate above: a vp SOLID splat has NO alpha gate; render it fully opaque.
            flags &= !(MAT_FLAG_CUTOUT | MAT_FLAG_BLEND);
        }
        // Vert-Paint 3-layer splat (BOTH the SoftCutout decal AND the opaque "Solid" variant):
        // build the VpGpu entry so the fragment blends the game's 3 layers by COLOR_0.rgb ×
        // heights-mask instead of tiling layer 0 alone (layer0=road_sand parking lots rendered
        // as a rust-orange blotch grid). All layer albedos must resolve or we skip (fall back
        // to the old single-layer look rather than splat with a placeholder).
        let mut vp_index = 0u32;
        if let Some(vpv) = &mat.vp {
            let layers = vpv
                .get("layers")
                .and_then(|v| v.as_array())
                .filter(|l| l.len() == 3);
            if let Some(layers) = layers {
                let f4 = |v: Option<&serde_json::Value>, d: [f32; 4]| -> [f32; 4] {
                    v.and_then(|a| a.as_array()).map_or(d, |a| {
                        let mut out = d;
                        for (i, x) in a.iter().take(4).enumerate() {
                            out[i] = x.as_f64().unwrap_or(d[i] as f64) as f32;
                        }
                        out
                    })
                };
                let f3w = |v: Option<&serde_json::Value>, w: f32| -> [f32; 4] {
                    let mut out = [1.0, 1.0, 1.0, w];
                    if let Some(a) = v.and_then(|a| a.as_array()) {
                        for (i, x) in a.iter().take(3).enumerate() {
                            out[i] = x.as_f64().unwrap_or(1.0) as f32;
                        }
                    }
                    out
                };
                let mut tex_idx = [NO_ALBEDO; 4];
                let mut ok = true;
                for (i, l) in layers.iter().enumerate() {
                    match l.get("albedo").and_then(|v| v.as_str()).filter(|p| !p.is_empty()) {
                        Some(p) => {
                            tex_idx[i] =
                                *path_to_index.entry(p.to_string()).or_insert_with(|| {
                                    let idx = albedo_paths.len() as u32;
                                    albedo_paths.push(p.to_string());
                                    idx
                                });
                        }
                        None => ok = false,
                    }
                }
                if ok {
                    // Heights control mask: R/G/B = per-layer coverage, sampled at the RAW uv.
                    // DATA, not color -> linear upload (same rule as the terrain control maps).
                    tex_idx[3] = vpv
                        .get("heights")
                        .and_then(|v| v.as_str())
                        .filter(|p| !p.is_empty())
                        .map(|p| {
                            let idx =
                                *path_to_index.entry(p.to_string()).or_insert_with(|| {
                                    let idx = albedo_paths.len() as u32;
                                    albedo_paths.push(p.to_string());
                                    idx
                                });
                            ctrl_tex_linear.insert(idx);
                            idx
                        })
                        .unwrap_or(NO_ALBEDO);
                    let blend = vpv.get("blend").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    flags |= MAT_FLAG_VP;
                    vp_index = vp_table.len() as u32;
                    vp_table.push(VpGpu {
                        tex: tex_idx,
                        uv0: f4(layers[0].get("uv"), [1.0, 1.0, 0.0, 0.0]),
                        uv1: f4(layers[1].get("uv"), [1.0, 1.0, 0.0, 0.0]),
                        uv2: f4(layers[2].get("uv"), [1.0, 1.0, 0.0, 0.0]),
                        tint0: f3w(layers[0].get("tint"), blend.max(1.0)),
                        tint1: f3w(layers[1].get("tint"), 0.0),
                        tint2: f3w(layers[2].get("tint"), 0.0),
                    });
                }
            }
        }
        // Vert-Paint SOLID splats have NO alpha test — they render OPAQUE with their 3-layer
        // splat. The Otsu alpha-coverage detector mis-tags some as role=cutout with an impossible
        // _Cutoff (1.3) because their albedo alpha is SMOOTHNESS, not hole-coverage; left set, the
        // cutout discard (alpha < 0.5*1.3) would nuke every fragment. Clear it for any non-decal
        // vp material (genuine SoftCutout decals kept their softcutout path above).
        // Gate on the vp BLOCK, not on MAT_FLAG_VP. That flag is only set when all of the splat
        // layers' albedos resolved; a vp material with one null layer falls back to single-layer
        // tiling and never gets the flag, so this guard silently skipped the exact materials it
        // exists to protect. "tex.a is smoothness, not coverage" is a property of the MATERIAL,
        // not of whether we managed to build its layer table. woods material 725 (concrete
        // platform, 3 layers with the third missing) was the one material in all five packs that
        // fell through: alpha mean 0.043 against cutoff 0.5 discarded 98.3% of its fragments, so
        // it rendered as nothing while still being pickable. The other 270 cutout+vp materials
        // already cleared the flag and are unaffected.
        if mat.vp.is_some() && mat.role != "decal" {
            flags &= !MAT_FLAG_CUTOUT;
        }
        if mat.role == "water" {
            flags |= MAT_FLAG_WATER;
            // Textured water = a thin PUDDLE film: alpha-blended over the ground (P2).
            // Untextured water = DEEP water (sea / basins): OPAQUE pass — depth-write on, so
            // glass composites over it correctly and no pale clear-color bleeds through, and
            // the surface can't z-fight with the unsorted blend pass.
            if albedo_index != NO_ALBEDO {
                flags |= MAT_FLAG_BLEND;
                // Route the puddle shape mask to luma when its alpha is constant (atlas puddles).
                if mat
                    .albedo
                    .as_deref()
                    .is_some_and(|p| puddle_alpha_is_constant(&pack.resolve_path(p)))
                {
                    flags |= MAT_FLAG_PUDDLE_LUMA;
                }
                // Stretched floor decal (tire marks / wet-ground) -> matte, no mirror (pre-pass above).
                if stretched_water.get(mat.id as usize).copied().unwrap_or(false) {
                    flags |= MAT_FLAG_WATER_MATTE;
                }
            }
        }
        // Emissive (windows / monitors / signs / lamps): resolve the texture into the SAME
        // bindless sRGB albedo array (conventions.colorSpace.emissive == "srgb"); rgb = factor×hdr
        // precomputed. Both packs' emissive materials all carry textures — no factor-only path.
        let mut emissive_index = NO_EMISSIVE;
        let mut emissive_rgb = [0.0f32; 3];
        if let Some(em) = &mat.emissive {
            if let Some(p) = em.texture.as_deref().filter(|p| !p.is_empty()) {
                emissive_index = *path_to_index.entry(p.to_string()).or_insert_with(|| {
                    let idx = albedo_paths.len() as u32;
                    albedo_paths.push(p.to_string());
                    idx
                });
                emissive_rgb = [
                    em.factor[0] * em.hdr,
                    em.factor[1] * em.hdr,
                    em.factor[2] * em.hdr,
                ];
            }
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
                // is env-tunable (EFT_DETAIL_FADE="near,far") so the detail range can be verified/tuned
                // without a rebuild. Default raised to 40..120 m: this viewer's cold-load framing sits
                // tens-to-hundreds of metres out, so the old 8..25 m window faded detail to ~0 before
                // it was ever seen ("detail maps don't work"). 40..120 keeps detail subtle-but-visible
                // at normal viewing distance and still fades tiling out in the far field.
                let (fnear, ffar) = std::env::var("EFT_DETAIL_FADE")
                    .ok()
                    .and_then(|s| {
                        let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                        (v.len() == 2).then(|| (v[0], v[1]))
                    })
                    .unwrap_or((40.0, 120.0)); // was 8..25 m — detail faded out before the camera reached it
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
        // PARALLAX height map: resolve into the SAME bindless albedo array (dedup by path, first-seen),
        // mark it LINEAR (height is data, not sRGB color), set the flag + amount. Absent -> NO_ALBEDO
        // + no flag -> the shader skips the whole steep-parallax path (byte-identical render). VP/terrain
        // own their own UV blend so they never carry parallax (assembler already omits it for VP).
        // EFT_PARALLAX=0 kill switch: masks the flag for EVERY material so the shader's steep-parallax
        // path never runs — the byte-identical A/B lever for "texture swimming" reports (parallax is
        // the only per-pixel view-dependent UV math in the pipeline).
        let mut parallax_index = NO_ALBEDO;
        let mut parallax_scale = 0.0f32;
        let parallax_enabled = std::env::var("EFT_PARALLAX").map(|v| v.trim() != "0").unwrap_or(true);
        if let Some(par) = mat.parallax.as_ref().filter(|_| parallax_enabled) {
            if let Some(p) = par.map.as_deref().filter(|p| !p.is_empty()) {
                parallax_index = *path_to_index.entry(p.to_string()).or_insert_with(|| {
                    let idx = albedo_paths.len() as u32;
                    albedo_paths.push(p.to_string());
                    idx
                });
                ctrl_tex_linear.insert(parallax_index); // height is linear DATA, not sRGB
                parallax_scale = par.scale.clamp(0.0, 0.5);
                flags |= MAT_FLAG_PARALLAX;
            }
        }
        // GLASS_TRS response lanes (zeros on every other material): packed _ReflectColor /
        // _SpecColor + _Shininess, with Unity's legacy defaults where the material didn't author
        // one (grey 0.5 reflection/specular, gloss 0.078 — the legacy shader's UI defaults).
        let (glass_refl, glass_spec, glass_shin) = if glass_trs {
            let rc = mat.reflect_color.unwrap_or([0.5, 0.5, 0.5, 0.5]);
            let sc = mat.spec_color.unwrap_or([0.5, 0.5, 0.5]);
            let pk = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            // glass_refl's top byte carries the family's opacity PRE-SCALE (0..8 quantized) —
            // the dithered glass blocks multiply tex.a by _OpacityScale before their dither
            // (streets ships 4.0 over a 0.24-mean alpha); the reflective family packs 1.0.
            // _ReflectColor.a itself is unused by both game shaders.
            let opac = ((mat.opacity_scale.unwrap_or(1.0).clamp(0.0, 8.0) / 8.0) * 255.0).round() as u32;
            (
                pk(rc[0]) << 16 | pk(rc[1]) << 8 | pk(rc[2]) | opac << 24,
                pk(sc[0]) << 16 | pk(sc[1]) << 8 | pk(sc[2]),
                mat.shininess.unwrap_or(0.078).clamp(0.01, 1.0),
            )
        } else {
            (0u32, 0u32, 0.0f32)
        };
        materials_gpu.push(GpuMaterial {
            albedo_index,
            flags,
            alpha_cutoff: mat.alpha_cutoff,
            // Phase 1.6 GGX spec: per-material roughness (was _pad). Glass ships ~0.05 (sharp);
            // default 0.55 for unspecified. Clamp [0.03,1.0] so the NDF can't blow up / go
            // mirror-hard. TRS glass derives it from the authored Blinn-Phong _Shininess
            // (power = shin x 128; GGX rough = sqrt(2/(power+2))) instead of the flat 0.05.
            roughness: if glass_trs {
                (2.0 / (glass_shin * 128.0 + 2.0)).sqrt().clamp(0.03, 1.0)
            } else {
                mat.roughness.unwrap_or(0.55).clamp(0.03, 1.0)
            },
            uv_xform: mat.uv_xform, // reference only (uvTilingBaked=true); shader must NOT apply
            tint: mat.tint,
            vp: vp_params.unwrap_or([0.0; 4]),
            // Phase 2b normal mapping.
            normal_index,
            normal_flags,
            normal_scale: mat.normal_scale,
            // MAT_FLAG_VP: index into vp_table. MAT_FLAG_TERRAIN: slice index (tagged after this
            // loop, overwrites). The two classes are disjoint so the lane can't collide.
            _pad2: vp_index,
            // #6 Detail maps (zeros unless MAT_FLAG_DETAIL was set above).
            detail_albedo_index,
            detail_normal_index,
            detail_flags,
            glass_refl,
            detail_albedo_uv,
            detail_normal_uv,
            detail_params,
            detail_mean_gain,
            emissive_index,
            emissive_rgb,
            parallax_index,
            parallax_scale,
            glass_spec,
            glass_shin,
        });
    }

    // ---- #1 MicroSplat terrain: append the 12 layer + 12 control textures to the SAME bindless
    //      albedo set, build the splat table, and tag the 4 terrain materials (FLAG_TERRAIN +
    //      slice index in _pad2, matte roughness). Layer i weight = control(i/4).chan(i%4);
    //      layer_uv = terrainUV01*rep (the recovered MicroSplat tiling; NEVER m_TileSize). ----
    let mut terrain = TerrainSplatGpu {
        layer_albedo: [0; 12],
        layer_rep: [1.0; 12],
        ctrl_idx: [0; 48],
    };
    // (ctrl_tex_linear is declared above the material loop — vp heights masks share it.)
    'terrain: {
        let tl_path_owned = pack
            .manifest
            .sidecars
            .terrain_layers
            .as_deref()
            .map(|p| pack.resolve_path(p));
        let Some(tl_path) = tl_path_owned.as_deref() else {
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
        // Slice names come from the sidecar itself (NOT a hardcoded list — that silently
        // disabled MicroSplat on every non-Interchange map). Sorted for a stable slice->index
        // mapping; for Interchange the sorted order is identical to the old hardcoded const.
        let mut slice_names: Vec<String> = tiles.keys().cloned().collect();
        slice_names.sort();
        if slice_names.len() > 16 {
            warn!(
                "gpu-driven terrain: {} slices exceeds the 16-slice ctrl table — truncating",
                slice_names.len()
            );
            slice_names.truncate(16);
        }
        let mut layers_done = false;
        for (si, sname) in slice_names.iter().enumerate() {
            let Some(tile) = tiles.get(sname) else { continue };
            if let Some(cm) = tile.get("ctrl_maps").and_then(|v| v.as_array()) {
                for (k, c) in cm.iter().take(3).enumerate() {
                    if let Some(cn) = c.as_str() {
                        let idx = add_tex(cn);
                        terrain.ctrl_idx[si * 3 + k] = idx;
                        ctrl_tex_linear.insert(idx); // blend weights -> linear upload
                        no_downscale.insert(idx); // ...and full res: the texel IS the weight
                    }
                }
            }
            // The 12 layers are shared across slices (same MicroSplat material); capture once.
            if !layers_done {
                if let Some(layers) = tile.get("layers").and_then(|v| v.as_array()) {
                    // Layer albedos missing from the pack (pre-B8 extractor gated export on MEAN
                    // coverage and silently dropped locally-dominant layers, e.g. Sand/Pebbles)
                    // would bind the 1x1 MAGENTA load-failure placeholder -> magenta ground
                    // blotches wherever that layer's control weight dominates. Fall back to the
                    // first PRESENT layer instead: visually plausible ground + a loud warn telling
                    // the pack builder to re-extract, never magenta terrain.
                    let mut missing: Vec<(usize, String)> = Vec::new();
                    let mut first_present: Option<u32> = None;
                    for l in layers {
                        let idx = l.get("idx").and_then(|v| v.as_u64()).unwrap_or(99) as usize;
                        if idx >= 12 {
                            continue;
                        }
                        let name = l.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let rep = l.get("rep").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                        let fname = format!("layer_{name}.png");
                        if dir.join(&fname).exists() {
                            let ti = add_tex(&fname);
                            terrain.layer_albedo[idx] = ti;
                            first_present.get_or_insert(ti);
                        } else {
                            missing.push((idx, fname));
                        }
                        terrain.layer_rep[idx] = rep;
                    }
                    if !missing.is_empty() {
                        let fb = first_present.unwrap_or(0);
                        for (idx, fname) in &missing {
                            warn!(
                                "gpu-driven terrain: layer albedo '{fname}' (idx {idx}) missing from \
                                 the pack (pre-B8 export?) — substituting a present layer; re-extract \
                                 this map's terrain to restore the real texture"
                            );
                            terrain.layer_albedo[*idx] = fb;
                        }
                    }
                    layers_done = true;
                }
            }
        }
        // Tag the terrain materials: FLAG_TERRAIN + slice index in _pad2, matte roughness.
        // Cross-map correctness (audit): match the slice name as a WHOLE token (substring
        // matching mis-assigned Slice_1_1's control maps to Slice_1_11 on >9-slice maps), and
        // tag EVERY submesh's material, not just the first (multi-submesh terrain slices left
        // their remaining submeshes un-splatted).
        let mut tagged = 0u32;
        let token_match = |name: &str, s: &str| {
            name.match_indices(s).any(|(i, _)| {
                !name[i + s.len()..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit())
            })
        };
        for inst in &pack.instances {
            if inst.flags & crate::eftpack::flags::TERRAIN == 0 {
                continue;
            }
            let me = &pack.manifest.meshes[inst.mesh_id as usize];
            let Some(slice) = slice_names.iter().position(|s| token_match(&me.name, s))
            else {
                continue;
            };
            for sub in &me.submeshes {
                let mid = sub.material_id as usize;
                if mid < materials_gpu.len() {
                    materials_gpu[mid].flags |= MAT_FLAG_TERRAIN;
                    // #6: terrain owns albedo/normal via the splat branch — it must NEVER enter
                    // the detail path. Clear any detail a terrain material might have carried
                    // (defensive). Same for RFA (terrain forces matte roughness below — the base
                    // albedo alpha is meaningless in the splat path) and emissive.
                    materials_gpu[mid].flags &= !(MAT_FLAG_DETAIL | MAT_FLAG_RFA);
                    materials_gpu[mid].detail_flags = 0;
                    materials_gpu[mid].emissive_index = NO_EMISSIVE;
                    materials_gpu[mid]._pad2 = slice as u32;
                    materials_gpu[mid].roughness = 0.95; // matte ground, no shiny slab
                    tagged += 1;
                }
            }
        }
        info!(
            "gpu-driven #1 terrain: MicroSplat table built (12 layers × {} slices, {tagged} tiles tagged)",
            slice_names.len()
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

    let t_mats = build_t0.elapsed(); // phase mark: spheres + material/albedo/normal table done
    let mut vertex_data: Vec<f32> = Vec::new();
    let mut index_data: Vec<u32> = Vec::new();
    let mut instances: Vec<InstanceGpuRecord> = Vec::new();
    let mut mesh_meta: Vec<MeshMeta> = Vec::new();
    let mut mesh_names: Vec<String> = Vec::new();
    let mut inst_lod_group: Vec<i32> = Vec::new();
    // (par, par2, lv) per GPU instance — the loot-glow ancestry join key, parallel to `instances`.
    let mut inst_ancestry: Vec<(u32, u32, u32)> = Vec::new();
    // Blend-pass restructure (Codex review): per-mesh material class + a representative center
    // for back-to-front sorting of the per-mesh blend draws. class: 0=opaque-only, 1=blend-only,
    // 2=mixed (drawn in both passes; fragment class-discard splits it).
    let mut blend_meshes: Vec<(u32, Vec<[f32; 3]>, u32)> = Vec::new();

    let mut vtx_cursor: u32 = 0;
    let mut idx_cursor: u32 = 0;
    let mut inst_cursor: u32 = 0;

    // --- Fused raw->GPU geometry encoder (PERF: was pack.mesh_geom() per mesh = 5 temp Vecs +
    // a UV clone + interleave-append; now reads the pack's interleaved bytes ONCE per vertex and
    // writes final GPU records directly into pre-reserved buffers. Byte-identical output is gated
    // by EFT_GEOM_SHA=1 against the old path.) The vertex layout is pack-wide (one manifest.vertex
    // for every mesh), so hoist the attribute offsets out of the loop. ----
    let vlayout = &pack.manifest.vertex;
    let vstride = vlayout.stride as usize;
    let pos_off = vlayout
        .attr("position")
        .map(|a| a.offset as usize)
        .expect("vertex layout must define a 'position' attribute");
    let nrm_off = vlayout.attr("normal").map(|a| a.offset as usize);
    let uv_off = vlayout.attr("uv").map(|a| a.offset as usize);
    let col_off = vlayout.attr("color").map(|a| a.offset as usize);
    let mbin: &[u8] = &pack.meshes_bin;
    let blen = mbin.len();
    // Exact-size the destination buffers up front (over-reserves only for the rare mesh later
    // skipped by an OOB/empty guard, which is harmless). 13 f32 per vertex; 1 u32 per index.
    // The grass block later appends a fixed 12-vertex / 18-index cross-quad; include that headroom
    // so a single trailing append can't force a full-buffer doubling realloc (~340ms on a 937MiB
    // vertex buffer). instances/mesh_meta are sized to the non-grass counts (grass adds a small,
    // cheap-to-grow tail).
    {
        let (mut tot_v, mut tot_i, mut tot_inst, mut tot_mesh) = (0usize, 0usize, 0usize, 0usize);
        for (mi, m) in pack.manifest.meshes.iter().enumerate() {
            if by_mesh[mi].is_empty() {
                continue;
            }
            tot_v += m.vtx_count as usize;
            tot_i += m.idx_count as usize;
            // Shard-glass side walls (mesh loop below) append geometry; reserve their worst case
            // (4 verts / 6 indices per boundary edge, boundary edges <= submesh index count) so
            // the append can't force a doubling realloc of these near-exact-sized buffers.
            for sm in &m.submeshes {
                let is_mask = materials_gpu
                    .get(sm.material_id as usize)
                    .map_or(false, |mt| {
                        mt.flags & (MAT_FLAG_GLASS_MASK | MAT_FLAG_GLASS_TRS) != 0
                    });
                if is_mask {
                    tot_v += 4 * sm.idx_count as usize;
                    tot_i += 6 * sm.idx_count as usize;
                }
            }
            tot_inst += by_mesh[mi].len();
            tot_mesh += 1;
        }
        const GRASS_VERTS: usize = 12; // 3 cross-quads * 4 verts
        const GRASS_IDX: usize = 18; // 3 quads * 6 indices
        // f32 slots per vertex, DERIVED from the stride — a hardcoded count here silently
        // over-reserved by 12 B/vertex (1.14 GiB on streets) after the stride shrank, and a
        // Vec never gives that back.
        const VF: usize = DRAW_VERTEX_STRIDE as usize / 4;
        vertex_data.reserve((tot_v + GRASS_VERTS) * VF);
        index_data.reserve(tot_i + GRASS_IDX);
        instances.reserve(tot_inst);
        mesh_meta.reserve(tot_mesh + 1);
    }
    // Reused per-mesh scratch (cleared+resized each mesh; avoids the per-mesh Vec allocations).
    let mut vert_mat: Vec<u32> = Vec::new();
    let mut vert_uv: Vec<[f32; 2]> = Vec::new();
    // Shard-glass wall emission (logged after the loop so a silent no-op is visible).
    let mut glass_wall_quads = 0usize;
    let mut glass_wall_meshes = 0usize;

    for (mi, m) in pack.manifest.meshes.iter().enumerate() {
        let inst_ids = &by_mesh[mi];
        if inst_ids.is_empty() {
            continue; // orphan mesh â€” nothing references it
        }
        // --- fused raw->GPU encode: slice this mesh's interleaved vertex + index bytes directly
        // out of meshes.bin (== the old vertex_bytes(m)/index_bytes(m)) and validate the byte
        // ranges, replicating mesh_geom()'s OOB guard (skip+warn) exactly. ---
        let n = m.vtx_count as usize;
        let ni = m.idx_count as usize;
        let vtx_end = m.vtx_offset as usize + n * vstride;
        let idx_end = m.idx_offset as usize + ni * 4;
        if vtx_end > blen || idx_end > blen {
            warn!(
                "gpu-driven: mesh {} '{}' skipped: byte range out of bounds",
                m.id, m.name
            );
            continue;
        }
        if n == 0 || ni == 0 {
            continue;
        }
        let vb = &mbin[m.vtx_offset as usize..vtx_end];
        let ib = &mbin[m.idx_offset as usize..idx_end];

        // --- geometry into the global vertex/index buffers (offsets we own) ---
        let base_vertex = vtx_cursor as i32;

        // Append this mesh's (mesh-local) indices straight from bytes; borrow them back for the
        // per-vertex material scatter + puddle detection (identical to reading geom.indices).
        let idx_data_start = index_data.len();
        index_data.extend((0..ni).map(|i| crate::eftpack::read_u32(ib, i * 4)));
        let local_idx = &index_data[idx_data_start..];

        // M3: per-vertex material index. Each submesh is a contiguous index range into this
        // mesh's single vertex array; across ALL multi-submesh meshes in this pack the
        // submeshes reference DISJOINT vertex sets (measured: zero cross-submesh sharing),
        // so tagging each referenced vertex with its submesh's materialId needs NO vertex
        // duplication. Verts not referenced by any submesh are never rasterized (they are
        // absent from the drawn index run), so the fallback material is irrelevant; we seed
        // it to the first submesh's id for safety.
        let default_mat = m.submeshes.first().map(|s| s.material_id).unwrap_or(0);
        vert_mat.clear();
        vert_mat.resize(n, default_mat);
        for sm in &m.submeshes {
            let start = sm.idx_start as usize;
            let end = start + sm.idx_count as usize;
            for &vi in &local_idx[start..end.min(local_idx.len())] {
                if (vi as usize) < n {
                    vert_mat[vi as usize] = sm.material_id;
                }
            }
        }

        // --- Puddle re-UV (load-time; fixes hard-edged water/puddle decals) ---------------------
        // EFT's real puddles are small `decal_plane` quads (~5 m) whose [0,1] UVs map the WHOLE
        // City_puddle_big soft-blob stamp -> soft feathered edges. But some puddle materials are baked
        // onto huge ROAD-strip submeshes (e.g. 77x318 m) with the puddle texture mapped ~[0,1] across
        // the ENTIRE strip. Every visible fragment then samples a <3% UV window deep in the blob's
        // OPAQUE CORE (alpha ~1 there, at ANY mip), so the strip renders as a uniform HARD-edged slab.
        // Fix: for such STRETCHED water strips, ignore the unreliable baked UVs and planar-project the
        // LOCAL position onto the strip's plane at a fixed PUDDLE-sized metric scale, so the blob (soft
        // rim included) repeats every few metres -> a field of soft-edged puddles like the decal_planes.
        // Data-driven + map-agnostic (geometry only, no texture/mesh names):
        //   * textured water only (untextured deep water is the opaque path);
        //   * STRETCHED: local extent / baked-UV-span >> a puddle. The 5 m decal_planes are ~5 m/tile and
        //     stay untouched; the 300 m road strips are ~200 m/tile -> re-projected.
        //   * genuinely 2D: the strip's SECOND-widest axis must exceed a puddle, else it is a 1D
        //     tire-mark / water-trail streak authored to tile along one axis -> leave it alone.
        // (The matte flag is deliberately NOT a gate: the stretched-water heuristic mis-tags these big
        // road puddles as matte; matte only kills reflection in the shader, not edge softness.)
        const PUDDLE_TARGET_M: f32 = 6.0; // one soft blob per ~6 m (decal_plane puddles are ~5 m)
        const PUDDLE_STRETCH_MIN: f32 = 15.0; // m-per-UV-tile above which a water decal is "stretched"
        const PUDDLE_MIN_WIDTH_M: f32 = 3.0; // 2nd-widest local axis must exceed this (else a 1D streak)
        // Base UVs straight from the interleaved bytes (== mesh_geom's geom.uvs: the raw uv attr,
        // or [0,0] when the layout has no uv). Puddle re-UV may overwrite entries below.
        vert_uv.clear();
        match uv_off {
            Some(uo) => vert_uv.extend((0..n).map(|k| {
                let b = k * vstride + uo;
                [crate::eftpack::read_f32(vb, b), crate::eftpack::read_f32(vb, b + 4)]
            })),
            None => vert_uv.resize(n, [0.0, 0.0]),
        }
        for sm in &m.submeshes {
            let mid = sm.material_id as usize;
            let Some(mt) = materials_gpu.get(mid) else { continue };
            if mt.flags & MAT_FLAG_WATER == 0 || mt.albedo_index == NO_ALBEDO {
                continue;
            }
            let s0 = sm.idx_start as usize;
            let s1 = (s0 + sm.idx_count as usize).min(local_idx.len());
            if s1 <= s0 {
                continue;
            }
            let idx = &local_idx[s0..s1];
            // Local position bounds + baked-UV span (span only used to detect the stretch ratio).
            // Positions/UVs read from bytes; guard vi<n replicates geom.positions.get(vi)==Some.
            let (mut pmin, mut pmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            let (mut umin, mut umax) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
            for &vi in idx {
                let vi = vi as usize;
                if vi < n {
                    let v = crate::eftpack::read_vec3(vb, vi * vstride + pos_off);
                    pmin = pmin.min(v);
                    pmax = pmax.max(v);
                    if let Some(uo) = uv_off {
                        let b = vi * vstride + uo;
                        let u0 = crate::eftpack::read_f32(vb, b);
                        let u1 = crate::eftpack::read_f32(vb, b + 4);
                        umin[0] = umin[0].min(u0);
                        umax[0] = umax[0].max(u0);
                        umin[1] = umin[1].min(u1);
                        umax[1] = umax[1].max(u1);
                    }
                }
            }
            if !pmin.is_finite() {
                continue;
            }
            let psz = pmax - pmin;
            let uv_span = (umax[0] - umin[0]).max(umax[1] - umin[1]).max(1.0e-3);
            let m_per_tile = psz.length() / uv_span;
            // Sort local axes by extent: widest two are the plane, smallest is the surface normal.
            let mut ax = [(psz.x, 0usize), (psz.y, 1usize), (psz.z, 2usize)];
            ax.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let (a_u, a_v, second_width) = (ax[0].1, ax[1].1, ax[1].0);
            if m_per_tile < PUDDLE_STRETCH_MIN || second_width < PUDDLE_MIN_WIDTH_M {
                continue;
            }
            // Planar projection: LOCAL position -> UV at a fixed puddle scale, centred so the tiling
            // straddles the strip symmetrically. Repeat addressing tiles the blob down the strip.
            let cu = 0.5 * (pmin[a_u] + pmax[a_u]);
            let cv = 0.5 * (pmin[a_v] + pmax[a_v]);
            for &vi in idx {
                let vi = vi as usize;
                if vi < n && vert_mat[vi] == sm.material_id {
                    let p = crate::eftpack::read_vec3(vb, vi * vstride + pos_off).to_array();
                    vert_uv[vi] = [
                        (p[a_u] - cu) / PUDDLE_TARGET_M,
                        (p[a_v] - cv) / PUDDLE_TARGET_M,
                    ];
                }
            }
        }

        for k in 0..n {
            let base = k * vstride;
            let p = crate::eftpack::read_vec3(vb, base + pos_off).to_array();
            let nrm = match nrm_off {
                Some(o) => crate::eftpack::read_vec3(vb, base + o).to_array(),
                None => [0.0, 1.0, 0.0],
            };
            let uv = *vert_uv.get(k).unwrap_or(&[0.0, 0.0]);
            // M3b2: per-vertex COLOR_0 vert-paint weight. Every mesh in this pack carries a
            // color attr (unorm8x4 @32) so geom.colors is populated; default opaque-white for
            // any mesh that lacks it (color.a=1 -> SoftCutout coverage stays fully covered).
            // Colour rides through as the pack's own unorm8x4 bytes: pack them into one u32
            // (little-endian => memory order r,g,b,a, which is what Unorm8x4 reads) and smuggle it
            // through the f32 staging vec bit-exactly, the same trick material_index uses.
            let col_bits = match col_off {
                Some(o) => {
                    let b = base + o;
                    u32::from(vb[b])
                        | (u32::from(vb[b + 1]) << 8)
                        | (u32::from(vb[b + 2]) << 16)
                        | (u32::from(vb[b + 3]) << 24)
                }
                None => 0xFFFF_FFFF, // opaque white (color.a=1 -> SoftCutout stays fully covered)
            };
            vertex_data.extend_from_slice(&[
                p[0], p[1], p[2],
                oct_bits(Vec3::new(nrm[0], nrm[1], nrm[2])), // normal Snorm16x2 @12 (octahedral)
                uv[0], uv[1],
                f32::from_bits(vert_mat[k]), // material_index (read as Uint32 on the GPU)
                f32::from_bits(col_bits),    // color Unorm8x4 @28 (interpolated in the shader)
            ]);
        }
        // --- Shard-glass thickness: the game ships broken panes as TWO coincident shard layers
        // (front + back — 23 mm apart on ground_zero's Window_plastic_02) with no side walls, so
        // shards vanish into dark films seen side-on. Bridge each layer's boundary edges toward
        // its twin: every vertex's wall depth is HALF its distance to the nearest neighbour that
        // lies behind its normal (the twin layer), so the wall spans the pane's own authored
        // thickness — derived per vertex, nothing hardcoded. A single-layer pane has no twin and
        // gets no walls: the game gave it no thickness to draw. Side walls only — no new sheets —
        // so the face-on look (and its calibrated blend depth) is unchanged.
        // Boundary edge = undirected edge referenced by exactly one triangle.
        let mut side_edges: Vec<(u32, u32, u32, Vec3)> = Vec::new(); // (a, b, material, face normal)
        let mut twin_half: std::collections::HashMap<u32, f32> = Default::default();
        {
            let local_idx = &index_data[idx_data_start..];
            for sm in &m.submeshes {
                // Mask glass always; TRS glass too — its alpha is coverage as well, and the
                // twin-layer requirement below keeps single-sheet panes wall-free for free.
                let is_mask = materials_gpu
                    .get(sm.material_id as usize)
                    .map_or(false, |mt| {
                        mt.flags & (MAT_FLAG_GLASS_MASK | MAT_FLAG_GLASS_TRS) != 0
                    });
                if !is_mask {
                    continue;
                }
                let s0 = sm.idx_start as usize;
                let s1 = (s0 + sm.idx_count as usize).min(local_idx.len());
                // Referenced vertex set (positions + normals) for the twin search.
                let mut verts: Vec<u32> = local_idx[s0..s1]
                    .iter()
                    .copied()
                    .filter(|&v| (v as usize) < n)
                    .collect();
                verts.sort_unstable();
                verts.dedup();
                if verts.len() > 4096 {
                    continue; // perf guard: shard meshes are small; a giant sheet is not one
                }
                let pn: Vec<(Vec3, Vec3)> = verts
                    .iter()
                    .map(|&v| {
                        let base = v as usize * vstride;
                        let p = crate::eftpack::read_vec3(vb, base + pos_off);
                        let nrm = match nrm_off {
                            Some(o) => crate::eftpack::read_vec3(vb, base + o).normalize_or_zero(),
                            None => Vec3::Y,
                        };
                        (p, nrm)
                    })
                    .collect();
                // Twin distance: nearest vertex within a ~45° cone BEHIND this vertex's normal.
                // Both layers find each other symmetrically (each looks behind its own face).
                for (i, &vi) in verts.iter().enumerate() {
                    let (p, nrm) = pn[i];
                    let mut best = f32::INFINITY;
                    for (j, &(q, _)) in pn.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        let d = q - p;
                        let len = d.length();
                        if len < 1.0e-5 || len >= best {
                            continue;
                        }
                        if d.dot(nrm) < -0.7 * len {
                            best = len;
                        }
                    }
                    if best.is_finite() {
                        twin_half.insert(vi, 0.5 * best);
                    }
                }
                // undirected edge -> (directed a, b, owning-face normal, refcount)
                let mut edges: std::collections::HashMap<(u32, u32), (u32, u32, Vec3, u32)> =
                    Default::default();
                for tri in local_idx[s0..s1].chunks_exact(3) {
                    let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
                    if i0 as usize >= n || i1 as usize >= n || i2 as usize >= n {
                        continue;
                    }
                    let p0 = crate::eftpack::read_vec3(vb, i0 as usize * vstride + pos_off);
                    let p1 = crate::eftpack::read_vec3(vb, i1 as usize * vstride + pos_off);
                    let p2 = crate::eftpack::read_vec3(vb, i2 as usize * vstride + pos_off);
                    let fnrm = (p1 - p0).cross(p2 - p0);
                    if fnrm.length_squared() < 1.0e-12 {
                        continue; // degenerate sliver: no reliable outward direction
                    }
                    let fnrm = fnrm.normalize();
                    for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                        edges
                            .entry((a.min(b), a.max(b)))
                            .and_modify(|e| e.3 += 1)
                            .or_insert((a, b, fnrm, 1));
                    }
                }
                for (_, (a, b, fnrm, cnt)) in edges {
                    if cnt == 1 {
                        side_edges.push((a, b, sm.material_id, fnrm));
                    }
                }
            }
        }
        let mut extra_v = 0u32;
        let mut extra_i = 0u32;
        for (a, b, mat_id, fnrm) in side_edges {
            // No measured twin layer on either end -> no authored thickness -> no wall.
            let (Some(&ta), Some(&tb)) = (twin_half.get(&a), twin_half.get(&b)) else {
                continue;
            };
            let read_vert = |vi: u32| -> (Vec3, Vec3, [f32; 2], u32) {
                let base = vi as usize * vstride;
                let p = crate::eftpack::read_vec3(vb, base + pos_off);
                let nrm = match nrm_off {
                    Some(o) => crate::eftpack::read_vec3(vb, base + o),
                    None => Vec3::Y,
                };
                let uv = *vert_uv.get(vi as usize).unwrap_or(&[0.0, 0.0]);
                let col = match col_off {
                    Some(o) => {
                        u32::from(vb[base + o])
                            | (u32::from(vb[base + o + 1]) << 8)
                            | (u32::from(vb[base + o + 2]) << 16)
                            | (u32::from(vb[base + o + 3]) << 24)
                    }
                    None => 0xFFFF_FFFF,
                };
                (p, nrm, uv, col)
            };
            let (pa, na, uva, ca) = read_vert(a);
            let (pb, nb, uvb, cb) = read_vert(b);
            let dir = pb - pa;
            if dir.length_squared() < 1.0e-12 {
                continue;
            }
            // Outward wall normal: for a CCW front face the interior lies LEFT of a->b, so
            // edge x face-normal points away from the shard.
            let ns = dir.normalize().cross(fnrm).normalize();
            let ns_bits = oct_bits(ns);
            // Quad [a-front, b-front, b-back, a-back]; back rides the VERTEX normal toward the
            // twin layer so both layers' walls meet mid-gap. UV/color copied from the edge
            // vertex: the wall samples the shard's own edge texel (inside the coverage mask).
            let quad = [
                (pa, uva, ca),
                (pb, uvb, cb),
                (pb - nb.normalize_or_zero() * tb, uvb, cb),
                (pa - na.normalize_or_zero() * ta, uva, ca),
            ];
            let lbase = n as u32 + extra_v;
            for (p, uv, col) in quad {
                vertex_data.extend_from_slice(&[
                    p.x, p.y, p.z,
                    ns_bits,
                    uv[0], uv[1],
                    f32::from_bits(mat_id),
                    f32::from_bits(col),
                ]);
            }
            // Outward winding (verified against the CCW-interior rule above).
            index_data.extend_from_slice(&[
                lbase, lbase + 2, lbase + 1,
                lbase, lbase + 3, lbase + 2,
            ]);
            extra_v += 4;
            extra_i += 6;
        }
        glass_wall_quads += (extra_i / 6) as usize;
        if extra_i > 0 {
            glass_wall_meshes += 1;
        }
        vtx_cursor += n as u32 + extra_v;

        // indices were appended (from bytes) at the top of the loop; record the run.
        let first_index = idx_cursor;
        let index_count = ni as u32 + extra_i;
        idx_cursor += index_count;

        // --- instances (grouped-by-mesh, contiguous) with conservative world sphere ---
        let instance_base = inst_cursor;
        let bs = local_spheres[mi];
        let local_center = Vec3::new(bs[0], bs[1], bs[2]);
        let local_r = bs[3];
        // ALL instance centres for this mesh: the blend sort key is the distance to the NEAREST
        // one (see prepare/queue). Keeping only the first instance's centre made a mesh sort by an
        // arbitrary far-away copy of itself.
        let mut inst_centers: Vec<[f32; 3]> = Vec::new();
        for &i in inst_ids {
            let inst = &pack.instances[i as usize];
            let a = &inst.affine;
            let aff = inst.affine3a();
            let lin = Mat3::from(aff.matrix3);
            let center = aff.transform_point3(local_center);
            let radius = local_r * conservative_radius_scale(lin);
            // Distance-LOD encode into ids.z/ids.w on multi-LOD packs; (0,0) on lean = unchanged.
            let (lz, lw) = if multi_lod { lod_encode(i) } else { (0, 0) };
            inst_lod_group.push(inst.lod_group);
            inst_ancestry.push((inst.par, inst.par2, inst.lv));
            instances.push(InstanceGpuRecord {
                m0: [a[0], a[1], a[2], a[3]],
                m1: [a[4], a[5], a[6], a[7]],
                m2: [a[8], a[9], a[10], a[11]],
                ids: [mesh_meta.len() as u32, inst.flags, lz, lw],
                sphere: [center.x, center.y, center.z, radius],
            });
            // Collected for every mesh (the blend class is only known after this loop); the Vec is
            // moved into `blend_meshes` for blend meshes and dropped for the rest.
            inst_centers.push(center.to_array());
        }
        let instance_count = inst_ids.len() as u32;
        inst_cursor += instance_count;

        // Blend class from the submeshes' FINAL material flags (terrain tagging ran earlier).
        let (mut has_blend, mut has_opaque) = (false, false);
        let mut blend_passes = 0u32;
        for sm in &m.submeshes {
            let f = materials_gpu
                .get(sm.material_id as usize)
                .map(|mt| mt.flags)
                .unwrap_or(0);
            if f & MAT_FLAG_BLEND != 0 {
                has_blend = true;
                if f & MAT_FLAG_SOFTCUTOUT != 0 {
                    blend_passes |= BLEND_MESH_SOFTCUTOUT;
                } else if f & (MAT_FLAG_DECAL | MAT_FLAG_WATER) != 0 {
                    blend_passes |= BLEND_MESH_OVERLAY;
                } else {
                    blend_passes |= BLEND_MESH_TRANSPARENT;
                }
            } else {
                has_opaque = true;
            }
        }
        let blend_class: u32 = match (has_opaque, has_blend) {
            (_, false) => 0,
            (false, true) => 1,
            (true, true) => 2,
        };
        if blend_class != 0 {
            blend_meshes.push((
                mesh_meta.len() as u32,
                core::mem::take(&mut inst_centers),
                blend_passes,
            ));
        }

        mesh_names.push(m.name.clone());
        mesh_meta.push(MeshMeta {
            index_count,
            first_index,
            base_vertex,
            instance_base,
            instance_count,
            blend_class,
            _pad: [0, 0],
        });
    }
    info!(
        "shard-glass walls: {glass_wall_quads} edge quads across {glass_wall_meshes} meshes \
         (depth = each pane's own twin-layer gap; 0 quads = no mask-glass or single-layer panes)"
    );
    let t_geo = build_t0.elapsed(); // phase: the mesh geometry loop (parse + repack + append)
    let grass_instance_base = instances.len();
    let mut grass_instances = 0usize;
    let mut grass_mesh_range: Option<(u32, u32)> = None;

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
        // Grass KINDS from the sidecar. EFT scatters 9-30 different detail prototypes per map
        // (grass11, T_WhitGrass_A, Grass4_D, Field_grass_D, nettles...), each its own billboard
        // card with its own density grid. `format:2` sidecars carry that list; older sidecars
        // carry a single `albedo`+`tint`, which we read as a one-kind list so old packs still
        // render. One kind == one cross-quad mesh + one material + one indirect draw.
        let side = std::fs::read_to_string(pack.root.join("grass_sidecar.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let rec_stride: usize = match side.as_ref().and_then(|v| v.get("format")).and_then(|f| f.as_u64()) {
            Some(2) => 24, // [x,y,z,rotY,scale] f32 + kind u32
            _ => 20,       // legacy: [x,y,z,rotY,scale] f32
        };
        // Resolve a sidecar albedo PORTABLY. A correct sidecar carries a pack-relative name. A
        // broken build can bake an ABSOLUTE build-time path (a personal-path leak) that may even
        // point at ANOTHER map's texture — observed on customs, whose sidecar referenced
        // `.../eft_assets/interchange_v2/terrain_layers/grass_Grass3_D.png`. Trust an absolute
        // path only if that exact file exists on THIS machine; otherwise look for a pack-local
        // file of the same basename. Unresolvable => that kind is dropped (never the magenta
        // placeholder — that placeholder is the "pink grass all over customs" bug).
        let resolve_albedo = |raw_s: &str| -> Option<String> {
            let raw = std::path::Path::new(raw_s);
            let base = raw.file_name().unwrap_or(raw.as_os_str());
            let mut cands: Vec<std::path::PathBuf> = Vec::new();
            if raw.is_absolute() {
                cands.push(raw.to_path_buf());
            } else {
                cands.push(pack.root.join(raw));
            }
            cands.push(pack.root.join(base));
            cands.push(pack.root.join("terrain_layers").join(base));
            cands.iter().find(|c| c.is_file()).map(|c| c.to_string_lossy().into_owned())
        };
        let read_tint = |v: Option<&serde_json::Value>| -> [f32; 4] {
            v.and_then(|a| a.as_array())
                .map(|a| {
                    let g = |i: usize, d: f32| {
                        a.get(i).and_then(|x| x.as_f64()).unwrap_or(d as f64) as f32
                    };
                    [g(0, 0.7), g(1, 0.75), g(2, 0.55), 1.0]
                })
                .unwrap_or([0.7, 0.75, 0.55, 1.0])
        };
        // (resolved albedo path, tint) per kind, in SIDECAR ORDER — grass.bin's kind indices
        // address this list, so a dropped kind must keep its slot (None) rather than shift.
        let kinds: Vec<Option<(String, [f32; 4])>> = match side
            .as_ref()
            .and_then(|v| v.get("kinds"))
            .and_then(|k| k.as_array())
        {
            Some(arr) => arr
                .iter()
                .map(|k| {
                    let a = k.get("albedo").and_then(|a| a.as_str()).unwrap_or("");
                    resolve_albedo(a).map(|p| (p, read_tint(k.get("tint"))))
                })
                .collect(),
            None => {
                let a = side
                    .as_ref()
                    .and_then(|v| v.get("albedo").and_then(|a| a.as_str()))
                    .unwrap_or("");
                vec![if a.is_empty() { None } else { resolve_albedo(a).map(|p| (p, read_tint(side.as_ref().and_then(|v| v.get("tint"))))) }]
            }
        };
        if kinds.iter().all(|k| k.is_none()) {
            warn!("gpu-driven grass: no usable grass texture in the sidecar — skipping grass");
            break 'grass;
        }
        // WavingGrass params authored on the terrain (extractor -> grass.json -> sidecar). Fed to
        // the shader through the material's otherwise-unused `vp` lane; the vertex stage sways
        // each blade's TOP verts so the base stays planted. Absent/zero => static grass.
        let wind = side.as_ref().and_then(|v| v.get("wind"));
        let wf = |k: &str, d: f32| {
            wind.and_then(|w| w.get(k)).and_then(|x| x.as_f64()).unwrap_or(d as f64) as f32
        };
        let (w_strength, w_amount, w_speed) = (wf("strength", 0.0), wf("amount", 0.0), wf("speed", 0.0));

        // Cross-quad clump mesh + material, ONE PER KIND. The geometry is identical (3 quads at
        // 0/60/120 deg around Y, base at y=0) but the material id is baked into the vertex data,
        // so each kind needs its own 12-vertex copy — 11 kinds is ~132 verts, free.
        let (hw, gh) = (0.42f32, 0.9f32);
        let mut kind_mesh: Vec<Option<usize>> = Vec::with_capacity(kinds.len()); // kind -> mesh slot
        let mut kind_bins: Vec<Vec<[f32; 5]>> = vec![Vec::new(); kinds.len()];
        for k in &kinds {
            let Some((albedo, tint)) = k else {
                kind_mesh.push(None);
                continue;
            };
            let albedo_idx = *path_to_index.entry(albedo.clone()).or_insert_with(|| {
                let idx = albedo_paths.len() as u32;
                albedo_paths.push(albedo.clone());
                idx
            });
            let mat_id = materials_gpu.len() as u32;
            materials_gpu.push(GpuMaterial {
                albedo_index: albedo_idx,
                flags: MAT_FLAG_CUTOUT,
                alpha_cutoff: 0.35,
                roughness: 0.9,
                uv_xform: [1.0, 1.0, 0.0, 0.0],
                tint: *tint,
                // vp is unused by cutout foliage -> carry the wind params (see gpu_draw.wgsl).
                vp: [w_strength, w_amount, w_speed, 0.0],
                normal_index: NO_NORMAL,
                normal_flags: 0,
                normal_scale: 1.0,
                _pad2: 0,
                // No emissive: the all-zero Default would alias bindless slot 0 as an emissive map.
                emissive_index: NO_EMISSIVE,
                // #6: grass carries no detail map.
                ..GpuMaterial::default()
            });
            let base_vertex = vtx_cursor as i32;
            let first_index = idx_cursor;
            let mbits = f32::from_bits(mat_id);
            let (mut nverts, mut nidx) = (0u32, 0u32);
            for q in 0..3u32 {
                let ang = q as f32 * std::f32::consts::PI / 3.0;
                let (sa, ca) = ang.sin_cos();
                let (dx, dz) = (ca * hw, sa * hw);
                let b = nverts;
                let white = f32::from_bits(0xFFFF_FFFF); // color Unorm8x4 @28
                let up_oct = oct_bits(Vec3::Y);
                let mk = |x: f32, y: f32, z: f32, u: f32, v: f32| {
                    [x, y, z, up_oct, u, v, mbits, white]
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
            if grass_mesh_range.is_none() {
                grass_mesh_range = Some((mesh_meta.len() as u32, mesh_meta.len() as u32));
            }
            kind_mesh.push(Some(mesh_meta.len()));
            mesh_names.push(String::new()); // grass (synthetic)
            mesh_meta.push(MeshMeta {
                index_count: nidx,
                first_index,
                base_vertex,
                instance_base: 0, // filled once the instances are bucketed below
                instance_count: 0,
                blend_class: 0,
                _pad: [0; 2],
            });
        }

        // Bucket the records by kind: a mesh's instances must be CONTIGUOUS for the indirect
        // multidraw (instance_base + instance_count), and grass.bin is written per prototype
        // layer per slice, so several runs map to the same kind.
        let mut dropped = 0u32;
        for ch in bin.chunks_exact(rec_stride) {
            let f = |o: usize| f32::from_le_bytes([ch[o], ch[o + 1], ch[o + 2], ch[o + 3]]);
            let kind = if rec_stride == 24 {
                u32::from_le_bytes([ch[20], ch[21], ch[22], ch[23]]) as usize
            } else {
                0
            };
            match kind_bins.get_mut(kind) {
                Some(b) if kind_mesh.get(kind).copied().flatten().is_some() => {
                    b.push([f(0), f(4), f(8), f(12), f(16)])
                }
                _ => dropped += 1,
            }
        }
        if dropped > 0 {
            warn!("gpu-driven grass: {dropped} instances referenced a missing grass kind — dropped");
        }
        // Every grass kind's mesh slot has been pushed by now, and they are contiguous, so close the
        // range here — before the SEA quad can append past it.
        if let Some((start, _)) = grass_mesh_range {
            grass_mesh_range = Some((start, mesh_meta.len() as u32));
        }
        let mut count = 0u32;
        for (kind, recs) in kind_bins.iter().enumerate() {
            let Some(slot) = kind_mesh[kind] else { continue };
            let instance_base = inst_cursor;
            for r in recs {
                let (x, y, z, rot, sc) = (r[0], r[1], r[2], r[3], r[4]);
                let (s, c) = rot.sin_cos();
                inst_lod_group.push(-1); // grass (synthetic, ungrouped)
                instances.push(InstanceGpuRecord {
                    m0: [c * sc, 0.0, s * sc, x],
                    m1: [0.0, sc, 0.0, y],
                    m2: [-s * sc, 0.0, c * sc, z],
                    // ids.z = 1 tags GRASS for the cull's screen-size test (a clump's ~1.3 m
                    // sphere is sub-pixel long before the frustum far plane — cull it by
                    // projected size) AND for the wind sway in the vertex stage.
                    ids: [slot as u32, 0, 1, 0],
                    sphere: [x, y + gh * sc * 0.5, z, 1.3 * sc],
                });
            }
            let n = recs.len() as u32;
            mesh_meta[slot].instance_base = instance_base;
            mesh_meta[slot].instance_count = n;
            inst_cursor += n;
            count += n;
        }
        grass_instances = count as usize;
        info!(
            "gpu-driven #4 grass: {count} clumps across {} kind(s) appended (cross-quad, \
             alpha-cutout; wind {w_strength}/{w_amount}/{w_speed})",
            kind_mesh.iter().filter(|m| m.is_some()).count()
        );
    }

    // ---- SEA: horizon fill for coastal maps. The game DOES ship its ocean surface as geometry
    //      (shoreline: tiled `*_Sea_Water_*` role-water planes, all at one height) and those render
    //      through the deep-water path below — but they stop a couple km out, so the horizon past
    //      them is a VOID. One big untextured role-water quad 5 cm above the shipped tiles rides
    //      the same deep-water shading (dark teal body + band-limited ripple + fresnel sky mirror +
    //      sun glint), covering the tiles and running to the horizon. Height: EFT_SEA_LEVEL env
    //      (live tuning) > manifest.seaLevel (DERIVED from the scene's water planes by build_map's
    //      `derive_sea_level` — game truth, never authored). Absent -> inland map, no sea,
    //      byte-identical render. ----
    'sea: {
        let sea_level = std::env::var("EFT_SEA_LEVEL")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .or(pack.manifest.sea_level);
        let Some(sl) = sea_level else { break 'sea };
        let b = &pack.manifest.bounds;
        let (cx, cz) = ((b[0] + b[3]) * 0.5, (b[2] + b[5]) * 0.5);
        // The sea reaches well past the playable AABB so the horizon never shows the quad edge.
        let (hx, hz) = ((b[3] - b[0]) * 0.5 + 1200.0, (b[5] - b[2]) * 0.5 + 1200.0);
        let sea_mat_id = materials_gpu.len() as u32;
        materials_gpu.push(GpuMaterial {
            albedo_index: NO_ALBEDO, // untextured => the shader's DEEP-water branch (dark teal body)
            flags: MAT_FLAG_WATER,   // opaque pass: depth-write, no z-fight with the blend pass
            alpha_cutoff: 0.0,
            roughness: 0.08, // near-mirror: crisp fresnel sky reflection + tight sun glint
            uv_xform: [1.0, 1.0, 0.0, 0.0],
            tint: [1.0, 1.0, 1.0, 1.0], // ignored by the deep-water branch (dark teal body)
            vp: [0.0; 4],
            normal_index: NO_NORMAL,
            normal_flags: 0,
            normal_scale: 1.0,
            _pad2: 0,
            emissive_index: NO_EMISSIVE,
            ..GpuMaterial::default()
        });
        let base_vertex = vtx_cursor as i32;
        let first_index = idx_cursor;
        let mbits = f32::from_bits(sea_mat_id);
        // One quad, +Y normal, local origin at the center (height baked into the instance row).
        let white = f32::from_bits(0xFFFF_FFFF); // color Unorm8x4 @28
        let up_oct = oct_bits(Vec3::Y);
        let mk = |x: f32, z: f32, u: f32, v: f32| [x, 0.0, z, up_oct, u, v, mbits, white];
        for vtx in [
            mk(-hx, -hz, 0.0, 0.0),
            mk(hx, -hz, 1.0, 0.0),
            mk(hx, hz, 1.0, 1.0),
            mk(-hx, hz, 0.0, 1.0),
        ] {
            vertex_data.extend_from_slice(&vtx);
        }
        index_data.extend_from_slice(&[0, 1, 2, 0, 2, 3]); // shader flips N on back faces, winding-forgiving
        let instance_base = inst_cursor;
        inst_lod_group.push(-1); // sea (synthetic, ungrouped)
        instances.push(InstanceGpuRecord {
            m0: [1.0, 0.0, 0.0, cx],
            m1: [0.0, 1.0, 0.0, sl],
            m2: [0.0, 0.0, 1.0, cz],
            ids: [mesh_meta.len() as u32, 0, 0, 0],
            sphere: [cx, sl, cz, (hx * hx + hz * hz).sqrt() + 1.0],
        });
        inst_cursor += 1;
        vtx_cursor += 4;
        idx_cursor += 6;
        mesh_names.push(String::new()); // sea (synthetic)
        mesh_meta.push(MeshMeta {
            index_count: 6,
            first_index,
            base_vertex,
            instance_base,
            instance_count: 1,
            blend_class: 0, // deep water is OPAQUE (depth-write) — see the material-flag comment
            _pad: [0; 2],
        });
        info!("gpu-driven sea: synthesized ocean quad at y={sl:.1} ({:.0}x{:.0} m)", hx * 2.0, hz * 2.0);
    }

    let mesh_count = mesh_meta.len() as u32;
    let instance_total = inst_cursor;
    if mesh_count == 0 || instance_total == 0 {
        warn!("gpu-driven: nothing to draw (0 meshes / 0 instances)");
        return None; // poll_cpu_build clears the loading flag on a None result
    }

    info!(
        "gpu-driven: assembled {} meshes, {} instances, {} verts, {} indices",
        mesh_count,
        instance_total,
        vtx_cursor,
        index_data.len()
    );

    // Phase 1 SH-GI: load + repack the baked irradiance volume (volume.bin + volume.json).
    let t_grass = build_t0.elapsed(); // phase: grass append (done)
    let sh_volume = load_sh_volume(pack);

    // REALTIME lights: auto-select against the baked SH volume to avoid DOUBLE-COUNTING. Three cases:
    //  * no volume (no-CUDA fallback) -> FLAT under SH alone -> realtime ON.
    //  * legacy FULL volume (practicals baked in) -> realtime OFF (they'd double-count).
    //  * INDIRECT-only volume (bake-sh --indirect-only; volume.json "direct": false) -> practicals
    //    were EXCLUDED from the bake, so realtime ON supplies the crisp direct lighting (the
    //    direct/indirect split — baked soft indirect GI + real-time direct practicals).
    // `EFT_LIGHTS` overrides: `auto` (rule above), `rt` (force on), `sh` (force off).
    let has_real_volume = sh_volume.is_some();
    let indirect_volume = pack
        .manifest
        .sidecars
        .volume_meta
        .as_deref()
        .map(|m| pack.resolve_path(m))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("direct").and_then(|d| d.as_bool()))
        .map(|direct| !direct) // "direct": false -> indirect-only
        .unwrap_or(false); // absent/true -> legacy full bake
    let rt_mode = std::env::var("EFT_LIGHTS")
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_else(|_| "auto".to_string());
    let rt_enabled = match rt_mode.as_str() {
        "rt" | "on" | "1" => true,
        "sh" | "off" | "0" => false,
        _ => !has_real_volume || indirect_volume, // auto
    };
    info!(
        "gpu-driven realtime lights: EFT_LIGHTS={rt_mode} real_volume={has_real_volume} \
         indirect={indirect_volume} -> realtime {}",
        if rt_enabled { "ON" } else { "OFF" }
    );
    let light_grid = build_light_grid(&pack.lights, &pack.manifest.bounds, rt_enabled);

    // #5 shadows: source the sun direction from the SAME volume.json sidecar the SH bake used, with
    // the SAME X-flip standard.rs applies (Lsun = normalize(-raw.x, raw.y, raw.z), pointing TOWARD
    // the sun). `None` (missing/degenerate) => the shadow feature disables itself downstream.
    let sun_dir = pack
        .manifest
        .sidecars
        .volume_meta
        .as_deref()
        .map(|m| pack.resolve_path(m)) // self-contained packs: pack-relative sidecars
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|txt| serde_json::from_str::<serde_json::Value>(&txt).ok())
        .and_then(|v| {
            v.get("sun_dir").and_then(|s| s.as_array()).and_then(|a| {
                let raw = Vec3::new(
                    a.first()?.as_f64()? as f32, // volume.json sun_dir is ALREADY viewer-space (bake conjugates); flipping again mirrored sun/shadows vs the SH radiance (audit C1)
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

    // SELF-CONTAINED packs (PR3): every texture path collected above (albedo/normal/emissive/
    // detail/vp/heights/terrain/grass) may be pack-relative - resolve once here against the
    // pack dir. Absolute (legacy) paths pass through untouched; dedup already happened on the
    // raw strings, which stays consistent within one pack.
    for v in [&mut albedo_paths, &mut normal_paths] {
        for s in v.iter_mut() {
            if !std::path::Path::new(s.as_str()).is_absolute() {
                *s = pack.resolve_path(s);
            }
        }
    }
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    let vbytes = std::mem::size_of_val(vertex_data.as_slice());
    let ibytes = std::mem::size_of_val(index_data.as_slice());
    eprintln!(
        "[stall] build_cpu_data (off main thread unless EFT_SYNC_LOAD): {:.1} ms  ({} meshes, {} instances, {} albedo, {} normal)\n\
         [stall]   phases ms: bymesh={:.1} spheres+materials={:.1} geometry={:.1} grass={:.1} | \
         vtx_buf={:.1}MiB idx_buf={:.1}MiB",
        ms(build_t0.elapsed()),
        mesh_count,
        instance_total,
        albedo_paths.len(),
        normal_paths.len(),
        ms(t_bymesh),
        ms(t_mats - t_bymesh),
        ms(t_geo - t_mats),
        ms(t_grass - t_geo),
        vbytes as f64 / 1048576.0,
        ibytes as f64 / 1048576.0,
    );
    if geom_hash_enabled() {
        eprintln!(
            "[EFT_GEOM_SHA] vtx=0x{:016x} ({} f32)  idx=0x{:016x} ({} u32)",
            fnv1a64(bytemuck::cast_slice(&vertex_data)),
            vertex_data.len(),
            fnv1a64(bytemuck::cast_slice(&index_data)),
            index_data.len(),
        );
    }
    // Narrow the index buffer when every mesh is u16-eligible. Indices are mesh-LOCAL, so the
    // deciding value is the largest index actually stored, not the global vertex count. The guard
    // is a real check (not an assumption about the pack): any pack with a >64Ki-vertex mesh simply
    // stays on u32.
    let index_count = index_data.len();
    let index_u16 = index_data.iter().copied().max().is_none_or(|m| m < 65_536);
    let mut index_bytes: Vec<u8> = if index_u16 {
        let mut b = Vec::with_capacity(index_count * 2 + 2);
        for &i in &index_data {
            b.extend_from_slice(&(i as u16).to_le_bytes());
        }
        b
    } else {
        bytemuck::cast_slice(&index_data).to_vec()
    };
    // wgpu requires buffer writes to be 4-byte aligned in BOTH offset and size. An odd number of
    // u16 triangles lands on a 2-byte boundary, which would trip validation on the final streaming
    // chunk — pad with unused indices (draws are bounded by the indirect args, so they're inert).
    while index_bytes.len() % 4 != 0 {
        index_bytes.push(0);
    }
    eprintln!(
        "[stall]   index buffer: {} indices as {} -> {:.1} MiB{}",
        index_count,
        if index_u16 { "u16" } else { "u32" },
        index_bytes.len() as f64 / 1048576.0,
        if index_u16 { " (halved; all meshes < 64Ki verts)" } else { " (a mesh exceeds 64Ki verts)" },
    );
    drop(index_data); // the wide staging copy is dead now — free it before the upload
    // Loot-glow model match (loot.rs overlay): join the game's own LootableContainer records to
    // the GPU instances built above by PREFAB ANCESTRY (see match_loot_models). LOD shells join
    // through the same parent chain, so whichever shell the cull draws still glows.
    let loot_models = match_loot_models(&pack.root, &instances, &inst_ancestry);
    Some(CpuData {
        vertex_data,
        index_bytes,
        index_u16,
        index_count,
        instances,
        grass_mesh_range,
        grass_instances,
        grass_instance_base,
        mesh_meta,
        mesh_names,
        inst_lod_group,
        materials: materials_gpu,
        albedo_paths,
        normal_paths,
        sh_volume,
        terrain,
        vp_table,
        ctrl_tex_linear,
        no_downscale,
        blend_meshes,
        sun_dir,
        light_grid,
        instance_total,
        mesh_count,
        doors: pack.doors.clone(),
        loot_models,
        // B1: per-group reference center for the mode-1 distance metric, indexed by lod_group.
        // Padded to >=1 so the storage buffer is never zero-sized (wgpu rejects that); on a lean
        // pack it's bound but never read (every instance is a sentinel, mode-1 is skipped).
        lod_centers: if pack.manifest.lod_groups.is_empty() {
            vec![[0.0; 4]]
        } else {
            pack.manifest
                .lod_groups
                .iter()
                .map(|g| {
                    // `w` was a spare lane; it now carries the group's LOD-TRANSITION BAND as a
                    // FRACTION of each shell's far distance, derived from Unity's own authoring.
                    //
                    // The game ships `ftw` (fadeTransitionWidth) and `srh` (screenRelativeHeight)
                    // per level, and until now the viewer parsed BOTH and used NEITHER — every shell
                    // swap was a hard pop at one exact distance, which is the visible cost of the
                    // distance-LOD we just enabled. woods ships fadeMode=1 (cross-fade) on 35,252 of
                    // 42,928 groups and a non-zero width on 7,318, so this is the game telling us how
                    // wide each transition should be.
                    //
                    // ftw is in screen-relative-height units and far = size/(2*srh), so d(far)/far =
                    // -d(srh)/srh: a width of `ftw` in srh is a band of `ftw/srh` in DISTANCE. Taken
                    // as the max over the group's levels (one band per group is all `w` can hold) and
                    // clamped to 40%: an unclamped ratio on a tiny srh would swallow the whole shell.
                    let band = g
                        .srh
                        .iter()
                        .zip(g.ftw.iter())
                        .filter(|(s, _)| **s > 1.0e-6)
                        .map(|(s, w)| (w / s).abs())
                        .fold(0.0f32, f32::max)
                        .clamp(0.0, 0.40);
                    [g.center[0], g.center[1], g.center[2], band]
                })
                .collect()
        },
    })
}

/// In-flight off-thread CPU build. Keyed by the `MapEpoch` it was kicked for so a stale result
/// (an older map, superseded by a fast swap) is dropped instead of applied. Replacing this
/// resource cancels any previous in-flight task (a superseded build is wasted work anyway).
#[derive(Resource)]
struct PendingCpuBuild {
    task: bevy::tasks::Task<Option<CpuData>>,
    epoch: u64,
}

/// KICK (main world, on every `MapEpoch` change incl. the initial insert + LOD swaps): latch the
/// loading flag and spawn `compute_cpu_blob` onto the AsyncComputeTaskPool so the ~0.6–1.3 s build
/// no longer freezes the main thread — the loading indicator keeps animating while it runs.
/// `EFT_SYNC_LOAD=1` keeps the old in-one-frame behavior (build inline, apply immediately) as an
/// escape hatch for deterministic capture.
fn kick_cpu_build(
    mut commands: Commands,
    pack: Option<Res<LoadedPack>>,
    epoch: Res<super::MapEpoch>,
    tags: Query<(), With<GpuDrivenTag>>,
    lod: Res<crate::ForcedLod>,
    load_signal: Option<Res<GpuLoadSignal>>,
) {
    let Some(pack) = pack else {
        return;
    };
    // A new map's GPU build starts NOW: latch the loading flag so the indicator stays up through
    // the whole (off-thread) build + the (multi-frame) texture upload, not just the file load.
    // Cleared by `prepare_gpu_buffers` once the map is on-screen, or by `poll_cpu_build` on a
    // build that produced nothing.
    if let Some(s) = &load_signal {
        s.set(true);
    }
    let pack_arc = pack.0.clone(); // Arc clone (cheap); shares meshes.bin with the worker
    let lod = lod.0;
    let ep = epoch.0;

    // Escape hatch: build synchronously in this frame and apply immediately (old behavior).
    let sync_load = std::env::var("EFT_SYNC_LOAD")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    if sync_load {
        commands.remove_resource::<PendingCpuBuild>();
        match compute_cpu_blob(&pack_arc, lod) {
            Some(cpu) => {
                // Slim persistent copy for loot.rs: the blob itself is dropped after upload.
                commands.insert_resource(crate::loot::LootModelIndex {
                    models: cpu.loot_models.clone(),
                });
                commands.insert_resource(ExtractedCpuData(Arc::new(cpu), ep));
                if tags.is_empty() {
                    commands.spawn((GpuDrivenTag, Name::new("eft_gpu_driven_draw")));
                }
            }
            None => {
                if let Some(s) = &load_signal {
                    s.set(false);
                }
            }
        }
        return;
    }

    let task = bevy::tasks::AsyncComputeTaskPool::get()
        .spawn(async move { compute_cpu_blob(&pack_arc, lod) });
    // Inserting replaces (and thus cancels) any previous in-flight build for a superseded epoch.
    commands.insert_resource(PendingCpuBuild { task, epoch: ep });
}

/// POLL (main world, whenever a build is in flight): when the off-thread `compute_cpu_blob`
/// finishes, apply its result IFF it still matches the current `MapEpoch` (a fast map swap bumps
/// the epoch and re-kicks; the stale blob is dropped). Mirrors the drop-stale-results discipline
/// of `PendingMapLoad`.
fn poll_cpu_build(
    mut commands: Commands,
    pending: Option<ResMut<PendingCpuBuild>>,
    epoch: Res<super::MapEpoch>,
    tags: Query<(), With<GpuDrivenTag>>,
    load_signal: Option<Res<GpuLoadSignal>>,
) {
    let Some(mut pending) = pending else {
        return;
    };
    let Some(result) = bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(
        &mut pending.task,
    )) else {
        return; // still building
    };
    let built_epoch = pending.epoch;
    commands.remove_resource::<PendingCpuBuild>();

    if built_epoch != epoch.0 {
        // Superseded by a newer map/LOD; the newer kick's build is (or will be) in flight, and it
        // owns the loading flag. Drop this blob silently.
        return;
    }
    match result {
        Some(cpu) => {
            // Slim persistent copy for loot.rs: the blob itself is dropped after upload.
            commands.insert_resource(crate::loot::LootModelIndex {
                models: cpu.loot_models.clone(),
            });
            commands.insert_resource(ExtractedCpuData(Arc::new(cpu), built_epoch));
            // one entity to hang the draw phase item on (ignored by the draw command). Idempotent:
            // a SECOND GpuDrivenTag would make queue_gpu_driven emit every phase item twice (the
            // whole scene drawn 2×). The tag carries no per-map data, so keep the single one.
            if tags.is_empty() {
                commands.spawn((GpuDrivenTag, Name::new("eft_gpu_driven_draw")));
            }
        }
        None => {
            // Nothing to draw / build failed: clear the flag so the loading toast doesn't spin.
            if let Some(s) = &load_signal {
                s.set(false);
            }
        }
    }
}

// ===========================================================================
// Render-world persistent resources.
// ===========================================================================
#[derive(Resource)]
struct EftComputePipelines {
    reset_id: CachedComputePipelineId,
    cull_id: CachedComputePipelineId,
    sort_blend_id: CachedComputePipelineId,
    cull_layout: BindGroupLayout,
}

#[derive(Resource, Clone)]
struct EftDrawPipeline {
    shader: Handle<Shader>,
    /// #5 shadows: the depth-only shadow-caster shader (`gpu_shadow.wgsl`). Loaded at RenderStartup;
    /// the shadow render pipeline (which also needs the material_layout) is queued in
    /// `prepare_gpu_buffers` once that layout exists.
    shadow_shader: Handle<Shader>,
    prepass_shader: Handle<Shader>,
    pyramid_shader: Handle<Shader>,
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
    /// Grass mesh-slot range `[start, end)` — excluded from the SHADOW multidraw. Grass was already
    /// "skipped" by emitting a degenerate triangle inside the shadow vertex shader, which saves
    /// fragments but still runs a vertex invocation per clump: measured identical cost whether the
    /// quads were really rasterized or not (18.606 vs 18.609 ms), i.e. fill was never the cost.
    /// Dropping the draw range removes the invocations too (~1.75 ms of a 6.3 ms shadow pass).
    grass_mesh_range: Option<(u32, u32)>,
    vertex: Buffer,
    index: Buffer,
    /// Width the index buffer was uploaded in (u16 when every mesh fits under 64Ki vertices).
    index_format: IndexFormat,
    /// P1 OPAQUE indirect args (multidraw over all meshes; blend-only records zeroed by cs_reset).
    /// Also drives the shadow casters (blend never casts).
    indirect: Buffer,
    /// P2 BLEND indirect args (opaque-only records zeroed). Drawn as ONE record per blend mesh
    /// from depth-sorted Transparent3d items — no whole-scene re-raster, stable back-to-front.
    indirect_blend: Buffer,
    cull_uniform: Buffer,
    /// (mesh index, first-instance world center, transparent-pass mask) for every mesh with a
    /// BLEND submesh — the per-frame sort key and render-state classification source.
    blend_meshes: Vec<(u32, Vec<[f32; 3]>, u32)>,
    mesh_count: u32,
    instance_total: u32,
    /// Workgroups for `cs_sort_blend` (one invocation per BLEND mesh; 64 per group).
    blend_sort_groups: u32,
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
    /// Realtime lighting group(3) additions (bindings 8/9/10): the LightGrid uniform, the packed
    /// light records storage buffer, and the CSR grid storage buffer. Kept alive so `EftShBindGroup`
    /// stays valid; torn down with the rest of the per-map group(3) on an epoch swap.
    light_uniform: Buffer,
    #[allow(dead_code)]
    lights_buf: Buffer,
    #[allow(dead_code)]
    light_grid_buf: Buffer,
    /// The as-built LightGridUniform (params = light_scale/ambient/rt/sun_diffuse BASE values).
    /// `update_light_uniform` rewrites the GPU copy per frame as base x GfxSettings multipliers,
    /// so the UI lighting sliders are live with no rebuild (identical bytes at multiplier 1).
    light_base: LightGridUniform,
    /// BASE packed light records (with real colors) + the per-light group index (parallel to the
    /// lights, not the vec4s). `update_light_power` rewrites `lights_buf` from these whenever the
    /// per-group power state (GfxSettings.light_groups bitmask) changes: a light whose group is
    /// unpowered gets its color lane zeroed (contributes nothing) without touching positions/grid.
    light_records_base: Vec<[f32; 4]>,
    light_group: Vec<i32>,
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
    // Raised when a guard below disables the path, so the main world relaunches into M0 (finding 6).
    fallback: Option<Res<GpuFallback>>,
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
             GPU-driven path is DISABLED. Falling back to the M0 instanced path."
        );
        if let Some(f) = &fallback {
            f.0.store(true, Ordering::SeqCst); // main world relaunches into M0 (finding 6)
        }
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
             GPU-driven path is DISABLED. Falling back to the M0 instanced path."
        );
        if let Some(f) = &fallback {
            f.0.store(true, Ordering::SeqCst); // main world relaunches into M0 (finding 6)
        }
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
                storage_buffer_sized(false, None),           // 4: indirect OPAQUE (rw)
                storage_buffer_sized(false, None),           // 5: indirect BLEND (rw)
                storage_buffer_read_only_sized(false, None), // 6: lod_centers (B1 group-center metric)
                storage_buffer_read_only_sized(false, None), // 7: blend mesh ids (cs_sort_blend)
            ),
        ),
    );

    let ssbo_layout = render_device.create_bind_group_layout(
        "eft_draw_ssbo_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::VERTEX,
            (
                (0, storage_buffer_read_only_sized(false, None)), // instances
                (1, storage_buffer_read_only_sized(false, None)), // visible
                (2, storage_buffer_read_only_sized(false, None)), // loot-glow (u32 per instance)
                // 3: the SSAO AO lane (R8, prepass-derived), FRAGMENT-sampled by the opaque
                // shading; the white fallback binds here while SSAO is off. The shadow/prepass
                // pipelines share this layout and simply don't declare the binding (allowed).
                (
                    3,
                    texture_2d(TextureSampleType::Float { filterable: true })
                        .visibility(ShaderStages::FRAGMENT),
                ),
            ),
        ),
    );

    let cull_shader = asset_server.load("shaders/gpu_cull.wgsl");
    let cull_shader_sort = cull_shader.clone();
    let draw_shader = asset_server.load("shaders/gpu_draw.wgsl");
    let shadow_shader = asset_server.load("shaders/gpu_shadow.wgsl"); // #5 depth-only caster
    let prepass_shader = asset_server.load("shaders/gpu_prepass.wgsl"); // normal+roughness prepass
    let pyramid_shader = asset_server.load("shaders/depth_pyramid.wgsl"); // Phase-1 shared depth pyramid

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

    // Per-instance back-to-front ordering for transparent draws (see cs_sort_blend). Runs after
    // cs_cull, before the main pass.
    let sort_blend_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("eft_cull_sort_blend".into()),
        layout: vec![cull_layout.clone()],
        push_constant_ranges: vec![],
        shader: cull_shader_sort,
        shader_defs: vec![],
        entry_point: Some("cs_sort_blend".into()),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(EftComputePipelines {
        reset_id,
        cull_id,
        sort_blend_id,
        cull_layout,
    });
    commands.insert_resource(EftDrawPipeline {
        shader: draw_shader,
        shadow_shader,
        prepass_shader,
        pyramid_shader,
        mesh_pipeline: mesh_pipeline.clone(),
        ssbo_layout,
        material_layout: None, // filled in prepare_gpu_buffers once the albedo count is known
        sh_layout: None,       // filled in prepare_gpu_buffers alongside the material layout
    });
}

// ---- PrepareResources: build all GPU buffers + bind groups ONCE -------------
#[allow(clippy::too_many_arguments)]
/// On a NEW `MapEpoch` (an in-place `.eftpack` swap), drop the previous pack's per-map GPU
/// resources, null the two bindless layouts on `EftDrawPipeline`, and invalidate the specialized
/// pipeline cache — so `prepare_gpu_buffers` rebuilds everything for the new pack and no draw ever
/// binds a fresh material bind group against a pipeline compiled for the OLD pack's bindless array
/// size (a wgpu layout-incompatibility error). Map-INVARIANT state (`EftComputePipelines`, the
/// shaders, `ssbo_layout`, `mesh_pipeline`) is preserved; `ExtractedCpuData` is left alone (the
/// fresh blob is exactly what prepare needs). Runs before `prepare_gpu_buffers`.
fn reset_gpu_map_if_epoch_changed(
    mut commands: Commands,
    epoch: Option<Res<super::MapEpoch>>,
    draw: Option<Res<EftDrawPipeline>>,
    mut last: Local<Option<u64>>,
) {
    let Some(epoch) = epoch else {
        return;
    };
    let cur = epoch.0;
    if *last == Some(cur) {
        return; // unchanged since we last looked
    }
    let first = last.is_none();
    *last = Some(cur);
    if first {
        return; // first observation: let the initial map build normally — nothing to tear down yet
    }
    // Remove every per-map GPU resource (no-op if absent). Removing EftGpuBuffers clears the
    // build-once guard at the top of prepare_gpu_buffers; removing the bind groups drops the
    // instance/mesh_meta/visible buffers they solely own.
    commands.remove_resource::<EftGpuBuffers>();
    commands.remove_resource::<EftCullBindGroup>();
    commands.remove_resource::<EftDrawBindGroup>();
    // Abandon any in-flight async texture build for the OLD map (its tasks + partial uploads are
    // for stale geometry); prepare_gpu_buffers re-kicks a fresh build for the new epoch.
    commands.remove_resource::<GpuBuildState>();
    commands.remove_resource::<EftMaterialResources>();
    commands.remove_resource::<EftMaterialBindGroup>();
    commands.remove_resource::<EftShResources>();
    commands.remove_resource::<EftShBindGroup>();
    commands.remove_resource::<EftShadowConfig>();
    commands.remove_resource::<EftShadowPipeline>();
    commands.remove_resource::<EftShadowResources>();
    // EftDoors holds its own clone of the OLD pack's instance buffer (and door records keyed to
    // that pack's instance indices). Without this it survives the swap, stranding the buffer
    // (13 MiB on streets) and pointing at indices that no longer mean anything — the door swing
    // would animate whatever instance now sits at that slot. Doorless packs never rebuild it.
    commands.remove_resource::<EftDoors>();
    // Null the per-pack bindless layouts (keep the invariant fields) so `queue_gpu_driven`'s
    // `material_layout/sh_layout.is_none()` gate blocks specialization until prepare rebuilds them.
    if let Some(d) = draw {
        commands.insert_resource(EftDrawPipeline {
            shader: d.shader.clone(),
            shadow_shader: d.shadow_shader.clone(),
            prepass_shader: d.prepass_shader.clone(),
            pyramid_shader: d.pyramid_shader.clone(),
            mesh_pipeline: d.mesh_pipeline.clone(),
            ssbo_layout: d.ssbo_layout.clone(),
            material_layout: None,
            sh_layout: None,
        });
    }
    // Invalidate the specialized-pipeline cache: its entries reference the OLD pack's material
    // layout; re-init drops them so the next queue_gpu_driven re-specializes against the new one.
    // (PipelineCache itself has no removal API in Bevy 0.17 — a few leaked pipelines per swap is
    // acceptable for a viewer.)
    commands.insert_resource(SpecializedRenderPipelines::<EftDrawPipeline>::default());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuUploadPlan {
    omit_grass: bool,
    instance_count: usize,
    instance_bytes: u64,
}

/// Decide whether this map can be represented by the adapter before creating any large buffers.
/// If grass alone pushes the instance SSBO over the binding limit, omitting that contiguous run is
/// a compatible fallback: the rest of the map remains exact and the grass mesh indirect counts are
/// zeroed. Vertex/index geometry cannot be split without changing the draw architecture, so those
/// produce one clear, user-visible failure instead of an invalid device/encoder cascade.
fn gpu_upload_plan(
    vertex_bytes: u64,
    index_bytes: u64,
    instance_count: usize,
    grass_instances: usize,
    max_buffer_size: u64,
    max_storage_binding_size: u64,
    grass_requested: bool,
) -> Result<GpuUploadPlan, String> {
    for (label, bytes) in [("vertex", vertex_bytes), ("index", index_bytes)] {
        if bytes > max_buffer_size {
            return Err(format!(
                "{label} buffer needs {:.0} MiB, but this GPU supports at most {:.0} MiB per \
                 buffer. Try a smaller map or rebuild it with lean geometry.",
                bytes as f64 / 1048576.0,
                max_buffer_size as f64 / 1048576.0,
            ));
        }
    }
    if grass_instances > instance_count {
        return Err("pack has more grass records than total instances".to_string());
    }
    let instance_stride = std::mem::size_of::<InstanceGpuRecord>() as u64;
    let full_bytes = (instance_count as u64)
        .checked_mul(instance_stride)
        .ok_or_else(|| "instance buffer byte size overflowed u64".to_string())?;
    let binding_limit = max_buffer_size.min(max_storage_binding_size);
    let omit_grass = grass_instances > 0 && (!grass_requested || full_bytes > binding_limit);
    let selected_count =
        if omit_grass { instance_count - grass_instances } else { instance_count };
    let selected_bytes = (selected_count as u64)
        .checked_mul(instance_stride)
        .ok_or_else(|| "instance buffer byte size overflowed u64".to_string())?;
    if selected_bytes > binding_limit {
        return Err(format!(
            "instance buffer needs {:.0} MiB even without optional grass, but this GPU supports \
             at most {:.0} MiB for one storage binding. Try a smaller map or a leaner pack.",
            selected_bytes as f64 / 1048576.0,
            binding_limit as f64 / 1048576.0,
        ));
    }
    if selected_count > u32::MAX as usize {
        return Err(format!(
            "map has {selected_count} drawable instances, exceeding the renderer's {}-instance limit",
            u32::MAX
        ));
    }
    Ok(GpuUploadPlan {
        omit_grass,
        instance_count: selected_count,
        instance_bytes: selected_bytes,
    })
}

/// Remove the contiguous grass instance range while preserving mesh IDs. Grass mesh metadata is
/// retained with zero instances so every material/mesh index remains stable; synthetic meshes
/// appended after grass (currently the sea horizon) have only their instance base shifted.
fn compact_without_grass(
    instances: &[InstanceGpuRecord],
    mesh_meta: &[MeshMeta],
    grass_start: usize,
    grass_count: usize,
) -> (Vec<InstanceGpuRecord>, Vec<MeshMeta>) {
    let grass_end = grass_start
        .checked_add(grass_count)
        .expect("grass instance range overflow");
    assert!(grass_end <= instances.len(), "grass instance range outside instance data");
    let mut compact_instances = Vec::with_capacity(instances.len() - grass_count);
    compact_instances.extend_from_slice(&instances[..grass_start]);
    compact_instances.extend_from_slice(&instances[grass_end..]);
    let mut compact_mesh_meta = mesh_meta.to_vec();
    let removed = grass_count as u32;
    for meta in &mut compact_mesh_meta {
        let base = meta.instance_base as usize;
        if base >= grass_start && base < grass_end {
            meta.instance_count = 0;
        } else if base >= grass_end {
            meta.instance_base = meta.instance_base.saturating_sub(removed);
        }
    }
    (compact_instances, compact_mesh_meta)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_gpu_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>, // #5 shadows: queue the shadow depth pipeline once here
    cpu: Option<Res<ExtractedCpuData>>,
    already: Option<Res<EftGpuBuffers>>,
    ssao_pipe: Option<Res<super::ssao::SsaoPipeline>>,
    compute: Option<Res<EftComputePipelines>>,
    draw: Option<Res<EftDrawPipeline>>,
    map_epoch: Option<Res<super::MapEpoch>>,
    settings: Option<Res<crate::render::GfxSettings>>,
    // Async streaming build state (present only DURING a build) + the cross-world loading flag.
    mut build: Option<ResMut<GpuBuildState>>,
    load_signal: Option<Res<GpuLoadSignal>>,
) {
    if already.is_some() {
        // Buffers are built. The map is on-screen: the load is fully done, so drop the loading flag
        // (belt-and-suspenders — finalize already cleared it) and drop any render-world copy of the
        // ~650 MiB CPU staging blob that got re-extracted before free_cpu_staging drops the
        // main-world source, so the whole Arc is released (Codex P1).
        if let Some(s) = &load_signal {
            s.set(false);
        }
        if cpu.is_some() {
            commands.remove_resource::<ExtractedCpuData>();
        }
        return;
    }
    // Pipelines are created in RenderStartup, which runs BEFORE the first extract — so if the
    // extracted blob exists but the pipelines don't, the bindless feature guard disabled the
    // path permanently: drop the ~650 MiB render-world copy instead of retaining it forever
    // (Codex review). Capture the flag before the destructuring move below.
    let pipelines_missing = compute.is_none() || draw.is_none();
    let (Some(cpu), Some(compute), Some(draw)) = (cpu, compute, draw) else {
        if pipelines_missing {
            // The GPU-driven path is permanently disabled (feature guard); the map will never build,
            // so clear the loading flag or the indicator would spin forever.
            if let Some(s) = &load_signal {
                s.set(false);
            }
            commands.remove_resource::<ExtractedCpuData>();
        }
        return;
    };
    // Epoch gate: build ONLY from the blob that matches the CURRENT map epoch. The MapEpoch reaches
    // the render world a frame before build_cpu_data emits the matching blob, so on a fast swap the
    // previous map's still-resident blob would otherwise be rebuilt here and then locked in by the
    // `already.is_some()` guard above — rendering the wrong map forever.
    if let Some(ep) = &map_epoch {
        if cpu.1 != ep.0 {
            return;
        }
    }
    let epoch = cpu.1;
    let cpu = &cpu.0;
    let limits = render_device.limits();
    let grass_requested = settings.as_ref().map(|s| s.grass).unwrap_or(true);
    let upload_plan = match gpu_upload_plan(
        std::mem::size_of_val(cpu.vertex_data.as_slice()) as u64,
        cpu.index_bytes.len() as u64,
        cpu.instances.len(),
        cpu.grass_instances,
        limits.max_buffer_size,
        limits.max_storage_buffer_binding_size as u64,
        grass_requested,
    ) {
        Ok(plan) => plan,
        Err(reason) => {
            let message = format!("This map is too large for the selected GPU: {reason}");
            error!("gpu-driven allocation preflight: {message}");
            if let Some(s) = &load_signal {
                s.fail(message);
            }
            commands.remove_resource::<GpuBuildState>();
            commands.remove_resource::<ExtractedCpuData>();
            return;
        }
    };

    // ===== ASYNC STREAMING TEXTURE BUILD (fixes the "Not Responding" load freeze) ==============
    // Instead of decoding+BC-encoding+uploading all ~700 albedo + ~540 normal textures in ONE
    // render-thread pass (the multi-second stall that froze the winit pump), stream them in:
    //   * KICKOFF  — spawn the CPU-heavy prep (fs::read/decode/mip/BC, or a warm cache read) for
    //                every texture on the AsyncComputeTaskPool (parallel across cores), then RETURN.
    //   * PROGRESS — each frame, poll finished payloads + upload a TIME-BUDGETED batch, then RETURN.
    //                The map stays gated off (EftGpuBuffers not yet inserted) and the loading
    //                indicator keeps animating because every frame is short.
    //   * FINALIZE — once ALL textures are uploaded, fall through to the geometry/material/SH build
    //                below (one ~30 ms frame) which inserts EftGpuBuffers and the map appears.
    // `EFT_SYNC_LOAD=1` bypasses this and builds synchronously in one frame (escape hatch): the two
    // texture loops below then load inline exactly as before.
    let sync_load = std::env::var("EFT_SYNC_LOAD")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let (async_albedo, async_normal, async_geo): (
        Option<Vec<(Texture, TextureView)>>,
        Option<Vec<(Texture, TextureView)>>,
        Option<(Buffer, Buffer)>,
    ) = if sync_load {
        (None, None, None) // escape hatch: the synchronous path below produces textures + geometry
    } else {
        // -- KICKOFF: spawn off-thread prep for every texture (once per map epoch) --
        let need_kickoff = build.as_ref().map(|b| b.epoch != epoch).unwrap_or(true);
        if need_kickoff {
            if upload_plan.omit_grass {
                let why = if grass_requested {
                    "the full instance SSBO exceeds this adapter's binding limit"
                } else {
                    "foliage is disabled by the active quality settings"
                };
                warn!(
                    "gpu-driven grass: omitting {} clumps from GPU upload/dispatch because {why} \
                     ({:.0} MiB avoided)",
                    cpu.grass_instances,
                    cpu.grass_instances as f64
                        * std::mem::size_of::<InstanceGpuRecord>() as f64
                        / 1048576.0,
                );
            }
            let pool = bevy::tasks::AsyncComputeTaskPool::get();
            let bc = bc_enabled(&render_device);
            // Texture-quality mip skip, captured ONCE per map build so every texture of one map
            // agrees (a mid-build settings change applies to the next load, not half of this one).
            let mip_skip = TEX_MIP_SKIP.load(std::sync::atomic::Ordering::Relaxed) as u32;
            let albedo_tasks: Vec<Option<bevy::tasks::Task<TexCpu>>> = cpu
                .albedo_paths
                .iter()
                .enumerate()
                .map(|(i, path)| {
                    let path = path.clone();
                    // Terrain CONTROL maps are blend weights: load LINEAR + never BC (data_linear).
                    let data_linear = cpu.ctrl_tex_linear.contains(&(i as u32));
                    let no_downscale = cpu.no_downscale.contains(&(i as u32));
                    Some(pool.spawn(async move {
                        prepare_tex_cpu(path, bc, data_linear, no_downscale, false, [255, 0, 255, 255], mip_skip) // magenta placeholder
                    }))
                })
                .collect();
            let normal_tasks: Vec<Option<bevy::tasks::Task<TexCpu>>> = cpu
                .normal_paths
                .iter()
                .map(|path| {
                    let path = path.clone();
                    Some(pool.spawn(async move {
                        prepare_tex_cpu(path, bc, false, false, true, [128, 128, 255, 255], mip_skip) // flat-normal placeholder (is_normal: raw linear, never BC)
                    }))
                })
                .collect();
            let n_a = albedo_tasks.len();
            let n_n = normal_tasks.len();
            // Step 4: create the vertex+index buffers EMPTY now (COPY_DST) so the geometry can be
            // streamed in across the following frames rather than memcpy'd in one finalize frame.
            let vtx_total = std::mem::size_of_val(cpu.vertex_data.as_slice());
            let idx_total = cpu.index_bytes.len();
            let vertex = render_device.create_buffer(&BufferDescriptor {
                label: Some("eft_gpu_vertex"),
                size: vtx_total as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let index = render_device.create_buffer(&BufferDescriptor {
                label: Some("eft_gpu_index"),
                size: idx_total as u64,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            commands.insert_resource(GpuBuildState {
                epoch,
                albedo_tasks,
                normal_tasks,
                albedo_tex: (0..n_a).map(|_| None).collect(),
                normal_tex: (0..n_n).map(|_| None).collect(),
                started: std::time::Instant::now(),
                frames: 0,
                peak_ms: 0.0,
                geo: GeoStream {
                    vertex,
                    index,
                    vtx_total,
                    idx_total,
                    vtx_cursor: 0,
                    idx_cursor: 0,
                },
            });
            if let Some(s) = &load_signal {
                s.set(true);
            }
            eprintln!(
                "[stall] prepare_gpu_buffers: spawned {n_a} albedo + {n_n} normal off-thread prep \
                 tasks (async streaming build; EFT_SYNC_LOAD=1 to force synchronous)"
            );
            return;
        }

        // -- PROGRESS: poll finished tasks + upload a time-budgeted batch this frame --
        let bs = build.as_mut().unwrap();
        let frame_t0 = std::time::Instant::now();
        let budget = std::time::Duration::from_secs_f64(upload_budget_ms() / 1000.0);
        for i in 0..bs.albedo_tasks.len() {
            if bs.albedo_tex[i].is_some() {
                continue;
            }
            if frame_t0.elapsed() > budget {
                break;
            }
            // Poll into a temporary FIRST (ends the &mut borrow of the task slot), then upload +
            // clear — avoids a borrow conflict between `task` and the two slot writes.
            let ready = match bs.albedo_tasks[i].as_mut() {
                Some(task) => bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(task)),
                None => None,
            };
            if let Some(tc) = ready {
                // sRGB unless this is a terrain control map (data_linear was used to prep it).
                let srgb = !cpu.ctrl_tex_linear.contains(&(i as u32));
                bs.albedo_tex[i] =
                    Some(upload_prepared(&render_device, &render_queue, &tc, srgb, "eft_albedo"));
                bs.albedo_tasks[i] = None;
            }
        }
        for i in 0..bs.normal_tasks.len() {
            if bs.normal_tex[i].is_some() {
                continue;
            }
            if frame_t0.elapsed() > budget {
                break;
            }
            let ready = match bs.normal_tasks[i].as_mut() {
                Some(task) => bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(task)),
                None => None,
            };
            if let Some(tc) = ready {
                bs.normal_tex[i] = Some(upload_prepared(
                    &render_device,
                    &render_queue,
                    &tc,
                    false, // normals are LINEAR
                    "eft_normal",
                ));
                bs.normal_tasks[i] = None;
            }
        }
        // Step 4: stream a budgeted slice of the ~1.1 GiB geometry into the pre-created buffers
        // this frame (write_buffer chunks are 4-byte aligned — whole f32/u32 records). One vtx +
        // one idx chunk per frame overlaps the texture window, so the finalize frame no longer does
        // a big one-shot memcpy. GEO_CHUNK is per-buffer per-frame.
        const GEO_CHUNK: usize = 16 * 1024 * 1024;
        {
            let g = &mut bs.geo;
            if g.vtx_cursor < g.vtx_total {
                let end = (g.vtx_cursor + GEO_CHUNK).min(g.vtx_total);
                let bytes: &[u8] = bytemuck::cast_slice(&cpu.vertex_data);
                render_queue.write_buffer(&g.vertex, g.vtx_cursor as u64, &bytes[g.vtx_cursor..end]);
                g.vtx_cursor = end;
            }
            if g.idx_cursor < g.idx_total {
                let end = (g.idx_cursor + GEO_CHUNK).min(g.idx_total);
                let bytes: &[u8] = &cpu.index_bytes;
                render_queue.write_buffer(&g.index, g.idx_cursor as u64, &bytes[g.idx_cursor..end]);
                g.idx_cursor = end;
            }
        }
        let geo_done = bs.geo.vtx_cursor >= bs.geo.vtx_total && bs.geo.idx_cursor >= bs.geo.idx_total;
        let frame_ms = frame_t0.elapsed().as_secs_f64() * 1000.0;
        bs.peak_ms = bs.peak_ms.max(frame_ms);
        bs.frames += 1;
        let a_done = bs.albedo_tex.iter().all(|o| o.is_some());
        let n_done = bs.normal_tex.iter().all(|o| o.is_some());
        if !(a_done && n_done && geo_done) {
            return; // more frames needed — map stays gated off, indicator keeps animating
        }

        // -- DONE: drain the uploaded textures (order preserved) for the finalize block --
        let mut a: Vec<(Texture, TextureView)> = Vec::with_capacity(bs.albedo_tex.len());
        for slot in std::mem::take(&mut bs.albedo_tex) {
            a.push(slot.expect("albedo slot filled once a_done"));
        }
        let mut n: Vec<(Texture, TextureView)> = Vec::with_capacity(bs.normal_tex.len());
        for slot in std::mem::take(&mut bs.normal_tex) {
            n.push(slot.expect("normal slot filled once n_done"));
        }
        // Step 4: hand the fully-streamed geometry buffers to the finalize block (Buffer clones
        // share the same GPU allocation — no copy).
        let geo = (bs.geo.vertex.clone(), bs.geo.index.clone());
        eprintln!(
            "[stall] prepare_gpu_buffers ASYNC build DONE: {} albedo + {} normal textures + \
             {:.0} MiB geometry over {} frames, {:.0} ms wall — LONGEST single render-thread stall \
             {:.1} ms (budget {:.0} ms)",
            a.len(),
            n.len(),
            (bs.geo.vtx_total + bs.geo.idx_total) as f64 / 1048576.0,
            bs.frames,
            bs.started.elapsed().as_secs_f64() * 1000.0,
            bs.peak_ms,
            upload_budget_ms(),
        );
        commands.remove_resource::<GpuBuildState>();
        (Some(a), Some(n), Some(geo))
    };

    let prep_t0 = std::time::Instant::now(); // STALL: the finalize frame (geometry + SH + shadows)

    // Step 4: in the async path the vertex+index buffers were already created + streamed full over
    // the loading window, so the finalize frame just adopts them (no ~1.1 GiB memcpy here). The
    // sync (EFT_SYNC_LOAD) path still builds them one-shot in this frame.
    let geo_streamed = async_geo.is_some();
    let (vertex, index) = match async_geo {
        Some((v, i)) => (v, i),
        None => {
            let v = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("eft_gpu_vertex"),
                contents: bytemuck::cast_slice(&cpu.vertex_data),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            });
            let i = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("eft_gpu_index"),
                contents: &cpu.index_bytes,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            });
            (v, i)
        }
    };
    // The instance array is ONE storage binding. `gpu_upload_plan` has already checked the adapter
    // limit and selected the no-grass fallback when needed; build that compact upload view now.
    // Woods' 11.5 M clumps are 883 MiB at 80 B each, so this is a material residency reduction,
    // not merely a shader-side visibility toggle.
    let compact = if upload_plan.omit_grass {
        Some(compact_without_grass(
            &cpu.instances,
            &cpu.mesh_meta,
            cpu.grass_instance_base,
            cpu.grass_instances,
        ))
    } else {
        None
    };
    let (instance_records, mesh_records): (&[InstanceGpuRecord], &[MeshMeta]) = compact
        .as_ref()
        .map(|(instances, meta)| (instances.as_slice(), meta.as_slice()))
        .unwrap_or((cpu.instances.as_slice(), cpu.mesh_meta.as_slice()));
    debug_assert_eq!(instance_records.len(), upload_plan.instance_count);
    let instance_total = upload_plan.instance_count as u32;
    info!(
        "gpu-driven limits: max_buffer_size {:.0} MiB, max_storage_buffer_binding_size {:.0} MiB, \
         max_compute_workgroups_per_dimension {} (= {} instances at 64/group) \
         | vtx {:.0} MiB, idx {:.0} MiB, inst {:.0} MiB",
        limits.max_buffer_size as f64 / 1048576.0,
        limits.max_storage_buffer_binding_size as f64 / 1048576.0,
        limits.max_compute_workgroups_per_dimension,
        limits.max_compute_workgroups_per_dimension as u64 * 64,
        cpu.vertex_data.len() as f64 * 4.0 / 1048576.0,
        cpu.index_bytes.len() as f64 / 1048576.0,
        upload_plan.instance_bytes as f64 / 1048576.0,
    );
    if upload_plan.omit_grass {
        info!(
            "gpu-driven: {} grass instances omitted; uploading {:.0} MiB / {} instances",
            cpu.grass_instances,
            upload_plan.instance_bytes as f64 / 1048576.0,
            upload_plan.instance_count,
        );
    }
    let instances = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_instances"),
        contents: bytemuck::cast_slice(instance_records),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    // DOORS: match each swing door to its GPU instance (the panel sits at the door pivot) so
    // `animate_doors` can rotate it on click. Nearest instance by translation within 1.5 m.
    // MULTI-SHELL DOORS (was AUDIT #4, now fixed). A door leaf in a multi-shell LODGroup ships one
    // instance per shell; animating only the resident one made the door SNAP SHUT the moment
    // distance-LOD handed off. This is live data, not a hypothetical: factory_rework is an all-LOD
    // pack (17.1% non-default shells, 4,119 multi-shell groups) and distance-LOD now defaults on.
    // The part list can't cover it — parts match by MESH NAME and the extractor records the leaf's
    // `..._LOD0`, so a coarse sibling never matches. So after resolving parts by name, pull in every
    // instance sharing a matched part's LODGroup: shells of one renderer share a group, while the
    // static frame is its own group, so this adds exactly the door's other shells. Lean packs have
    // one shell per group and collapse to today's behaviour.
    if !cpu.doors.is_empty() {
        // Instance lookup by mesh id so a door's PARTS resolve by the game's own mesh names.
        let mut by_mesh: std::collections::HashMap<&str, Vec<usize>> = std::collections::HashMap::new();
        for (i, r) in cpu.instances.iter().enumerate() {
            if let Some(name) = cpu.mesh_names.get(r.ids[0] as usize).filter(|n| !n.is_empty()) {
                by_mesh.entry(name.as_str()).or_default().push(i);
            }
        }
        // LODGroup -> instances, built ONCE. The naive per-door scan is O(parts x instances)
        // (~275M iterations on streets) and would show up as load-time stall.
        let mut by_group: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();
        for (i, g) in cpu.inst_lod_group.iter().enumerate() {
            if *g >= 0 {
                by_group.entry(*g).or_default().push(i);
            }
        }
        let mut door_insts: Vec<DoorInst> = Vec::new();
        let mut n_parts = 0usize;
        for d in &cpu.doors {
            // The PANEL: nearest instance to the hinge. It defines the swing axis (and is the
            // fallback part set on a pack built before the extractor shipped `parts`).
            let mut best: Option<(usize, f32)> = None;
            for (i, r) in cpu.instances.iter().enumerate() {
                let t = Vec3::new(r.m0[3], r.m1[3], r.m2[3]);
                let dist = t.distance_squared(d.pivot);
                if dist < 2.25 && best.map(|(_, b)| dist < b).unwrap_or(true) {
                    best = Some((i, dist));
                }
            }
            let Some((panel, _)) = best else { continue };
            let base = cpu.instances[panel];
            // swing axis = door local-Z in viewer world (= instance affine column 2), normalized.
            // Game-verified: an OPEN door's Unity local rotation is exactly `open_angle` about its
            // local +Z (streets Inside_Door_Wood_23: +94.00 deg vs payload 94.0), and every shut
            // door sits at local identity — so the pack's baked matrix IS the closed pose.
            let axis = Vec3::new(base.m0[2], base.m1[2], base.m2[2]).normalize_or_zero();
            if axis.length_squared() < 0.5 {
                continue;
            }
            // SIGN: the viewer world is an X-MIRROR of Unity's (G = diag(-1,1,1), det<0), and
            // conjugation maps a rotation to R(G.a, -theta) — a mirror reverses rotational sense.
            // So the authored Unity angle must be NEGATED here or every door swings the wrong way.
            let open_rad = -d.open_angle.to_radians();
            // Collect the parts that swing with the panel. `parts` is the door GameObject's
            // renderer subtree (game truth); match each to the nearest instance of that MESH so
            // repeated door prefabs can't cross-match. Always include the panel.
            let mut idxs = vec![panel];
            for (mesh, pos) in &d.parts {
                let Some(cands) = by_mesh.get(mesh.as_str()) else { continue };
                let mut bi: Option<(usize, f32)> = None;
                for &i in cands {
                    let r = &cpu.instances[i];
                    let t = Vec3::new(r.m0[3], r.m1[3], r.m2[3]);
                    let dd = t.distance_squared(*pos);
                    if dd < 1.0 && bi.map(|(_, b)| dd < b).unwrap_or(true) {
                        bi = Some((i, dd));
                    }
                }
                if let Some((i, _)) = bi {
                    if !idxs.contains(&i) {
                        idxs.push(i);
                    }
                }
            }
            // Sibling SHELLS of everything matched so far (see the multi-shell note above).
            let mut shells: Vec<usize> = Vec::new();
            for &i in &idxs {
                let g = cpu.inst_lod_group.get(i).copied().unwrap_or(-1);
                if g < 0 {
                    continue;
                }
                if let Some(sib) = by_group.get(&g) {
                    for &j in sib {
                        if !idxs.contains(&j) && !shells.contains(&j) {
                            shells.push(j);
                        }
                    }
                }
            }
            idxs.extend(shells);
            let locked = d.state.eq_ignore_ascii_case("locked");
            let shipped_open = d.state.eq_ignore_ascii_case("open");
            // EFT_DOORS_OPEN=1 opens every unlocked door INSTANTLY at spawn (debug / screenshots).
            let dbg_open = std::env::var("EFT_DOORS_OPEN").map(|v| v.trim() == "1").unwrap_or(false);
            let p = if dbg_open && !locked { 1.0 } else { shipped_open as u8 as f32 };
            // A door authored OPEN ships its OPEN pose baked into the instance matrix, so the
            // animation base must be that pose rotated BACK to closed — otherwise progress=1
            // rotated it a second time and it rendered at double the open angle.
            let parts = idxs
                .into_iter()
                .map(|i| DoorPart {
                    gpu_idx: i as u32,
                    closed: if shipped_open {
                        door_record(&cpu.instances[i], d.pivot, axis, -open_rad)
                    } else {
                        cpu.instances[i]
                    },
                })
                .collect::<Vec<_>>();
            n_parts += parts.len();
            door_insts.push(DoorInst {
                parts,
                pivot: d.pivot,
                axis,
                open_rad,
                locked,
                progress: p,
                target: p,
            });
        }
        eprintln!(
            "[doors] matched {} of {} swing doors ({n_parts} parts)",
            door_insts.len(),
            cpu.doors.len()
        );
        commands.insert_resource(EftDoors {
            doors: door_insts,
            instances_buf: instances.clone(),
        });
    }
    let mesh_meta = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_mesh_meta"),
        contents: bytemuck::cast_slice(mesh_records),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let visible = render_device.create_buffer(&BufferDescriptor {
        label: Some("eft_gpu_visible"),
        size: instance_total as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indirect = render_device.create_buffer(&BufferDescriptor {
        label: Some("eft_gpu_indirect"),
        size: cpu.mesh_count as u64 * DRAW_ARG_STRIDE,
        usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indirect_blend = render_device.create_buffer(&BufferDescriptor {
        label: Some("eft_gpu_indirect_blend"),
        size: cpu.mesh_count as u64 * DRAW_ARG_STRIDE,
        usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // seed the cull uniform to all-zero planes (= everything visible) and zero screen-size
    // thresholds (= cull nothing) so frame 0, before the first frustum upload, draws rather
    // than randomly culling.
    let seed = CullUniform {
        frustum: [[0.0; 4]; 6],
        counts: [instance_total, cpu.mesh_count, 0, 0],
        cam_k: [0.0; 4],
        lod_params: [1.0, 1.0, 0.0, 0.0], // proj11=1, bias=1, mode=0 (max detail) until upload_frustum
    };
    let cull_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_cull_uniform"),
        contents: bytemuck::bytes_of(&seed),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    // B1: per-group reference centers for the mode-1 distance metric (indexed by the group id in
    // ids.z bits 13+). Read-only; never read on lean packs (mode-1 unreachable) but always bound.
    let lod_centers = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_lod_centers"),
        contents: bytemuck::cast_slice(&cpu.lod_centers),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    // Mesh indices that have a BLEND submesh — the work list for cs_sort_blend. Never empty:
    // an empty storage buffer is invalid, so a map with no transparent geometry gets one dummy
    // entry and the shader's `count < 2` guard makes it a no-op.
    let mut blend_ids: Vec<u32> = cpu.blend_meshes.iter().map(|(m, _, _)| *m).collect();
    if blend_ids.is_empty() {
        blend_ids.push(0);
    }
    let blend_ids_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_gpu_blend_mesh_ids"),
        contents: bytemuck::cast_slice(&blend_ids),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let blend_sort_groups = (blend_ids.len() as u32).div_ceil(64);
    let cull_bg = render_device.create_bind_group(
        "eft_cull_bg",
        &compute.cull_layout,
        &BindGroupEntries::sequential((
            cull_uniform.as_entire_binding(),
            instances.as_entire_binding(),
            mesh_meta.as_entire_binding(),
            visible.as_entire_binding(),
            indirect.as_entire_binding(),
            indirect_blend.as_entire_binding(),
            lod_centers.as_entire_binding(),
            blend_ids_buf.as_entire_binding(),
        )),
    );
    // Loot-glow highlight lane: one u32 per instance (0 = no glow), zero-initialized; the
    // per-frame `prepare_loot_glow` rewrites it whenever the loot overlay's visible set changes.
    let loot_glow_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("eft_loot_glow"),
        size: (instance_total.max(1) as u64) * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(EftLootGlow {
        buffer: loot_glow_buf.clone(),
        len: instance_total.max(1),
        last_gen: u64::MAX,
    });
    // Draw bind group starts on the WHITE AO fallback; `sync_draw_bg_ao` swaps in the live AO
    // lane (and back) as the SSAO toggle / window size change. The inputs resource is what lets
    // it rebuild this group without re-running the whole map build.
    let ao_fallback = ssao_pipe.as_ref().map(|s| s.fallback_ao_view.clone());
    let draw_bg = render_device.create_bind_group(
        "eft_draw_bg",
        &draw.ssbo_layout,
        &BindGroupEntries::with_indices((
            (0, instances.as_entire_binding()),
            (1, visible.as_entire_binding()),
            (2, loot_glow_buf.as_entire_binding()),
            (
                3,
                bevy::render::render_resource::BindingResource::TextureView(
                    ao_fallback.as_ref().expect(
                        "SsaoPipeline initializes in RenderStartup, before any map build",
                    ),
                ),
            ),
        )),
    );
    commands.insert_resource(EftDrawBgInputs {
        instances: instances.clone(),
        visible: visible.clone(),
        loot_glow: loot_glow_buf.clone(),
        bound_ao: None, // fallback bound; sync swaps in the live lane when SSAO is on
    });

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
    // Vert-Paint 3-layer splat table (group(2) binding(5)); a zeroed sentinel entry keeps the
    // binding valid when the pack has no vp materials (the shader never reads it then).
    let vp_entries: &[VpGpu] = if cpu.vp_table.is_empty() {
        &[VpGpu {
            tex: [NO_ALBEDO; 4],
            ..VpGpu::default()
        }]
    } else {
        &cpu.vp_table
    };
    let vp_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_vp_splat"),
        contents: bytemuck::cast_slice(vp_entries),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Decode + upload every UNIQUE albedo (image crate -> Rgba8UnormSrgb). One texture per
    // entry, IN THE SAME order as cpu.albedo_paths, so GpuMaterial.albedo_index stays aligned;
    // a failed decode still pushes a placeholder at its slot to preserve that alignment.
    let geo_ms = prep_t0.elapsed().as_secs_f64() * 1000.0; // STALL: geometry+SSBO buffers phase
    let tex_t0 = std::time::Instant::now();
    let mut textures: Vec<Texture> = Vec::with_capacity(cpu.albedo_paths.len());
    let mut views: Vec<TextureView> = Vec::with_capacity(cpu.albedo_paths.len());
    if let Some(prepared) = async_albedo {
        // ASYNC path: textures were decoded off-thread + uploaded across frames already — just
        // collect them here (same order as cpu.albedo_paths, so albedo_index stays aligned).
        for (tex, view) in prepared {
            textures.push(tex);
            views.push(view);
        }
    } else {
        // EFT_SYNC_LOAD escape hatch: decode + upload every UNIQUE albedo inline (image crate ->
        // Rgba8UnormSrgb). IN THE SAME order as cpu.albedo_paths, so GpuMaterial.albedo_index stays
        // aligned; a failed decode still pushes a placeholder at its slot to preserve that alignment.
        for (i, path) in cpu.albedo_paths.iter().enumerate() {
            // Terrain CONTROL maps are blend weights (data, not color): load them LINEAR — the sRGB
            // decode would gamma-warp the weights toward the dominant layer (visible splat banding).
            let (tex, view) = if cpu.ctrl_tex_linear.contains(&(i as u32)) {
                load_data_texture(&render_device, &render_queue, path) // linear, never BC (weights)
            } else {
                load_albedo_texture(&render_device, &render_queue, path)
            };
            textures.push(tex);
            views.push(view);
        }
    }
    // A binding_array needs >= 1 element; if this pack referenced no albedo at all, synth a
    // 1x1 white so the layout/bind group stay valid (all materials then hit the sentinel).
    if views.is_empty() {
        let (tex, view) = make_dummy_texture(&render_device, &render_queue);
        textures.push(tex);
        views.push(view);
    }
    let tex_count = views.len() as u32;
    let albedo_ms = tex_t0.elapsed().as_secs_f64() * 1000.0; // STALL: albedo decode+BC+upload loop
    let norm_t0 = std::time::Instant::now();

    // Phase 2b: decode + upload every UNIQUE normal map, MIRRORING the albedo load but with a
    // LINEAR format (Rgba8Unorm) — normal maps are LINEAR data, NOT sRGB; the sRGB format would
    // gamma-wash the tangent vectors and flatten the perturbation. Same order as cpu.normal_paths
    // so GpuMaterial.normal_index stays aligned; a failed decode pushes a flat-normal placeholder.
    let mut normal_textures: Vec<Texture> = Vec::with_capacity(cpu.normal_paths.len());
    let mut normal_views: Vec<TextureView> = Vec::with_capacity(cpu.normal_paths.len());
    if let Some(prepared) = async_normal {
        for (tex, view) in prepared {
            normal_textures.push(tex);
            normal_views.push(view);
        }
    } else {
        for path in &cpu.normal_paths {
            let (tex, view) = load_normal_texture(&render_device, &render_queue, path);
            normal_textures.push(tex);
            normal_views.push(view);
        }
    }
    // binding_array needs >= 1 element; synth a 1x1 flat normal if this pack has no normal maps.
    if normal_views.is_empty() {
        let (tex, view) = make_dummy_normal_texture(&render_device, &render_queue);
        normal_textures.push(tex);
        normal_views.push(view);
    }
    let normal_count = normal_views.len() as u32;
    let normal_ms = norm_t0.elapsed().as_secs_f64() * 1000.0; // STALL: normal decode+BC+upload loop

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
        // 8x anisotropy: keeps ground/road textures sharp at grazing angles now that the full
        // mip chain exists (valid because mag/min/mipmap are all Linear, a wgpu requirement).
        anisotropy_clamp: 8,
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
                // The material TABLE is also read in the VERTEX stage (the grass WavingGrass sway
                // reads its wind params from the material's `vp` lane); without VERTEX here the
                // draw pipelines fail to create with "Shader global ResourceBinding { group: 2,
                // binding: 0 } is not available in the pipeline layout". Only binding 0 is
                // widened — the bindless texture ARRAY stays fragment-only.
                (
                    0,
                    storage_buffer_read_only_sized(false, None)
                        .visibility(ShaderStages::VERTEX_FRAGMENT),
                ),
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
                (5, storage_buffer_read_only_sized(false, None)), // vert-paint 3-layer splat table
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
            (5, vp_buf.as_entire_binding()),
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
    // Probe validity: R8Unorm so the shader reads it straight as 0..1. Sampled with the SAME
    // linear sampler and uvw as the SH textures, so a tap's validity always matches its probe.
    let sh_valid_tex = render_device.create_texture_with_data(
        &render_queue,
        &TextureDescriptor {
            label: Some("eft_sh_valid"),
            size: sh_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &sh.valid,
    );
    let sh_valid_view = sh_valid_tex.create_view(&TextureViewDescriptor::default());

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
    // #3 GI intensity (shader multiplies GI/env by vol_min.w). Priority: EFT_GI env override >
    // the pack's data-driven per-map `gi_intensity` (volume-meta sidecar) > 1.0. Lets a dark bake
    // (Interchange reads ~2 stops dark) be lifted without a rebuild; NOT hardcoded per-map in Rust.
    let gi_intensity = std::env::var("EFT_GI")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(sh.gi_intensity);
    let sh_uniform_data = ShVolumeUniform {
        vol_min: [sh.min[0], sh.min[1], sh.min[2], gi_intensity], // w = gi_intensity (EFT_GI / sidecar / 1.0)
        // w = normal_bias (meters) for the manual 8-tap leak fix.
        vol_inv_extent: [sh_inv_extent[0], sh_inv_extent[1], sh_inv_extent[2], SH_NORMAL_BIAS],
        // xyz = probe grid dims (as f32); w = ground/top sky ratio (out-of-volume redirect scale).
        dims: [sh_nx as f32, sh_ny as f32, sh_nz as f32, sh.ground_over_top],
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
    // #5 sun shadows now default ON for every map that has a sun_dir (the baked SH volume carries
    // soft static GI; the real-time cascade adds the crisp directional contact term daytime maps
    // need). EFT_SHADOWS=0 (or =false) is a HARD VETO (dev/perf); the in-app graphics toggle
    // (GfxSettings.shadows, default on) is the user control — sync_gfx_shadow_toggle ANDs the two.
    let shadows_env_allow = std::env::var("EFT_SHADOWS")
        .map(|v| {
            let t = v.trim();
            t != "0" && !t.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true);
    let shadow_debug = std::env::var("EFT_SHADOW_DEBUG")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let (lsun, shadows_enabled) = match cpu.sun_dir {
        Some(d) => (d, shadows_env_allow), // sun present -> on unless EFT_SHADOWS vetoes (UI refines/frame)
        None => (Vec3::Y, false),          // no sun_dir -> disabled (Y-up sentinel; never sampled)
    };
    info!(
        "gpu-driven #5 shadows: enabled={shadows_enabled} debug={shadow_debug} Lsun={lsun:?} \
         ({n} cascades to {reach:.0} m, {sz}²×{n} Depth32Float; default ON, EFT_SHADOWS=0 to disable, \
          diag EFT_SHADOW_DEBUG=1)",
        sz = shadow_map_size(),
        n = SHADOW_CASCADES,
        reach = SHADOW_SPLITS[SHADOW_CASCADES],
    );

    // The depth atlas, one layer per cascade. RENDER_ATTACHMENT (the shadow pass writes it) | TEXTURE_BINDING (the
    // main pass samples it). One D2Array sampling view + one D2 render view per layer.
    let shadow_depth = render_device.create_texture(&TextureDescriptor {
        label: Some("eft_shadow_depth"),
        size: Extent3d {
            width: shadow_map_size(),
            height: shadow_map_size(),
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
        std::array::from_fn(|c| shadow_layer_view(c as u32));

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

    // group(1) cascade-uniform layout for the shadow pipeline. VERTEX reads the world->light-clip
    // matrix; FRAGMENT reads `params.y` (the bindless albedo count) for its descriptor-index
    // clamp — so the binding must be visible to BOTH stages or pipeline creation fails with
    // "Shader global ResourceBinding { group: 1, binding: 0 } is not available in the pipeline
    // layout / Visibility flags don't include the shader stage" and the shadow pass silently
    // stops existing.
    let cascade_layout = render_device.create_bind_group_layout(
        "eft_shadow_cascade_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer_sized(false, None),
        ),
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
    // Built with from_fn rather than a hand-written array literal: the literal had to grow by hand
    // for every cascade added, and a count/literal mismatch is a compile error at best and a
    // silently unbound cascade at worst. Labels stay per-index for capture tooling.
    let cascade_uniforms: [Buffer; SHADOW_CASCADES] =
        std::array::from_fn(|_| make_cascade_uniform());
    let cascade_bind_groups: [BindGroup; SHADOW_CASCADES] = std::array::from_fn(|c| {
        render_device.create_bind_group(
            match c {
                0 => "eft_shadow_cascade_bg0",
                1 => "eft_shadow_cascade_bg1",
                2 => "eft_shadow_cascade_bg2",
                _ => "eft_shadow_cascade_bg3",
            },
            &cascade_layout,
            &BindGroupEntries::single(cascade_uniforms[c].as_entire_binding()),
        )
    });

    // The main SunShadowUniform (group(3) binding(5)). Initialize enabled=0 so the very first frame
    // — before prepare_shadow_uniforms runs — is a strict no-op; per-frame fill flips it on.
    let shadow_main_seed = SunShadowUniform {
        combine: [
            SHADOW_DIFFUSE_CAP,
            SHADOW_FADE_START,
            SHADOW_FADE_END,
            if shadow_debug { 1.0 } else { 0.0 },
        ],
        sun_dir_texel: [lsun.x, lsun.y, lsun.z, 1.0 / shadow_map_size() as f32],
        gfx: [1.0, 1.0, 1.0, 0.0], // neutral scales — a zeroed lane would kill fog on frame 0
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
                // pos @0 (loc0), uv @16 (loc2), material @24 (loc3). normal/color are skipped.
                attributes: vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 16,
                        shader_location: 2,
                    },
                    VertexAttribute {
                        format: VertexFormat::Uint32,
                        offset: 24,
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

    // ---- NORMAL PREPASS (gpu_prepass.wgsl) ------------------------------------------------------
    // Same "shared culled buffers, different pipeline" shape as the shadow pass, but through the
    // CAMERA into an Rgba16Float normal+roughness target with its own 1x depth. Targets are created
    // by `prepare_prepass` once the view size is known (and recreated on resize); only the uniform,
    // its bind group and the queued pipeline are made here.
    let prepass_layout = render_device.create_bind_group_layout(
        "eft_prepass_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (uniform_buffer_sized(
                false,
                Some(std::num::NonZeroU64::new(80).unwrap()),
            ),),
        ),
    );
    let prepass_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_prepass_uniform"),
        contents: bytemuck::bytes_of(&PrepassUniform::default()),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    let prepass_bg = render_device.create_bind_group(
        "eft_prepass_bg",
        &prepass_layout,
        &BindGroupEntries::single(prepass_uniform.as_entire_binding()),
    );
    let prepass_pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("eft_normal_prepass".into()),
        layout: vec![
            draw.ssbo_layout.clone(),
            prepass_layout,
            material_layout.clone(),
        ],
        push_constant_ranges: vec![],
        vertex: VertexState {
            shader: draw.prepass_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            buffers: vec![VertexBufferLayout {
                array_stride: DRAW_VERTEX_STRIDE,
                step_mode: VertexStepMode::Vertex,
                // pos @0 (loc0), oct normal @12 (loc1), uv @16 (loc2), material @24 (loc3).
                // color (@28) is skipped — the prepass has no vert-paint use.
                attributes: vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Snorm16x2,
                        offset: 12,
                        shader_location: 1,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 16,
                        shader_location: 2,
                    },
                    VertexAttribute {
                        format: VertexFormat::Uint32,
                        offset: 24,
                        shader_location: 3,
                    },
                ],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            cull_mode: None, // double-sided, like the main pass; front_facing flip in the fragment
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            // CAMERA depth: Bevy reverse-z (clear 0.0, GreaterEqual) — NOT the shadow pass's
            // conventional LessEqual. Getting this backwards renders exactly nothing.
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: 1, // consumers (ssao/ssr) want single-sample data; no A2C at 1x
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(FragmentState {
            shader: draw.prepass_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    // Depth pyramid: layout (depth src, mip src, storage dst) + the two compute pipelines.
    let pyramid_layout = render_device.create_bind_group_layout(
        "eft_pyramid_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                bevy::render::render_resource::binding_types::texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                bevy::render::render_resource::binding_types::texture_storage_2d(
                    TextureFormat::R32Float,
                    bevy::render::render_resource::StorageTextureAccess::WriteOnly,
                ),
            ),
        ),
    );
    let pyramid_shader = draw.pyramid_shader.clone();
    let pyramid_copy = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("eft_pyramid_copy".into()),
        layout: vec![pyramid_layout.clone()],
        push_constant_ranges: vec![],
        shader: pyramid_shader.clone(),
        shader_defs: vec![],
        entry_point: Some("cs_copy".into()),
        zero_initialize_workgroup_memory: false,
    });
    let pyramid_reduce = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("eft_pyramid_reduce".into()),
        layout: vec![pyramid_layout.clone()],
        push_constant_ranges: vec![],
        shader: pyramid_shader,
        shader_defs: vec![],
        entry_point: Some("cs_reduce".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(EftPyramidResources {
        layout: pyramid_layout,
        copy_pipeline: pyramid_copy,
        reduce_pipeline: pyramid_reduce,
        tex: None,
        mip_views: Vec::new(),
        sample_view: None,
        bind_groups: Vec::new(),
        size: UVec2::ZERO,
        mips: 0,
        active: false,
    });
    commands.insert_resource(EftPrepassResources {
        pipeline_id: prepass_pipeline_id,
        uniform: prepass_uniform,
        bind_group: prepass_bg,
        normal_texture: None,
        normal_view: None,
        depth_texture: None,
        depth_view: None,
        size: UVec2::ZERO,
        active: false,
        clip_from_world: [[0.0; 4]; 4],
        prev_clip_from_world: None,
    });

    // ---- REALTIME lights (group(3) bindings 8/9/10) --------------------------------------------
    // Tiny CPU-built buffers (a few KB of light records + a few 100 KB grid) — no streaming needed;
    // build them here on the render thread in the same finalize as the SH/shadow group(3) resources.
    // Torn down with the rest of group(3) on an epoch swap (EftShResources/EftShBindGroup removed).
    let lg = &cpu.light_grid;
    let light_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_light_grid_uniform"),
        contents: bytemuck::bytes_of(&lg.uniform),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    // Build the records in the DEFAULT power state (all groups OFF = mask 0): a switch-controlled
    // light ships dark until its lever is flipped, so zero its color lane now. `update_light_power`
    // rewrites this buffer when a switch toggles. Ungrouped lights are unchanged.
    let mut light_records = lg.lights.clone();
    apply_light_power_records(&mut light_records, &lg.light_group, 0);
    let lights_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_lights"),
        contents: bytemuck::cast_slice(&light_records),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let light_grid_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("eft_light_grid"),
        contents: bytemuck::cast_slice(&lg.grid),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // group(3): ShVolume uniform (0) + 3 SH 3D textures (1,2,3) + filtering sampler (4) + #5 shadow
    // additions: SunShadowUniform (5) + depth-2d-array (6) + comparison sampler (7) + realtime-light
    // additions: LightGrid uniform (8) + lights storage (9) + CSR grid storage (10). SHARED by both
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
                // #5 SunShadowUniform. VERTEX too: gfx.w carries the app time that phases the
                // grass WavingGrass sway in the vertex stage (gpu_draw.wgsl).
                (
                    5,
                    uniform_buffer_sized(false, None).visibility(ShaderStages::VERTEX_FRAGMENT),
                ),
                (6, texture_2d_array(TextureSampleType::Depth)), // #5 texture_depth_2d_array
                (7, sampler(SamplerBindingType::Comparison)),    // #5 sampler_comparison
                (8, uniform_buffer_sized(false, None)),          // realtime LightGrid uniform
                (9, storage_buffer_read_only_sized(false, None)), // realtime packed light records
                (10, storage_buffer_read_only_sized(false, None)), // realtime CSR light grid
                // 11: per-probe validity (Unity APV leak reduction), R8Unorm 3D, same grid as sh_r/g/b
                (11, texture_3d(TextureSampleType::Float { filterable: true })),
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
            (8, light_uniform.as_entire_binding()),
            (9, lights_buf.as_entire_binding()),
            (10, light_grid_buf.as_entire_binding()),
            (11, &sh_valid_view),
        )),
    );

    // Re-insert the draw pipeline WITH the material + SH layouts now known, so specialize() can
    // build the 4-group pipeline layout (view / ssbo / material / sh-gi).
    commands.insert_resource(EftDrawPipeline {
        shader: draw.shader.clone(),
        shadow_shader: draw.shadow_shader.clone(),
        prepass_shader: draw.prepass_shader.clone(),
        pyramid_shader: draw.pyramid_shader.clone(),
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
        light_uniform,
        lights_buf,
        light_grid_buf,
        light_base: lg.uniform,
        light_records_base: lg.lights.clone(),
        light_group: lg.light_group.clone(),
    });
    commands.insert_resource(EftShBindGroup(sh_bg));
    // #5 shadows: the runtime switch, the queued pipeline + cascade layout, and the GPU resources.
    commands.insert_resource(EftShadowConfig {
        lsun,
        enabled: shadows_enabled,
        env_enabled: shadows_env_allow,
        sun_ok: cpu.sun_dir.is_some(),
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
        "gpu-driven Phase2b: {} normal-map textures uploaded (LINEAR BC5 Rg where BC is supported, else raw Rgba8Unorm), normal_tex @group(2) binding(3)",
        normal_count
    );
    info!(
        "gpu-driven SH-GI: irradiance volume uploaded ({}x{}x{}), group(3) bind group built",
        sh_nx, sh_ny, sh_nz
    );

    commands.insert_resource(EftGpuBuffers {
        grass_mesh_range: cpu.grass_mesh_range,
        vertex,
        index,
        indirect,
        indirect_blend,
        cull_uniform,
        blend_meshes: cpu.blend_meshes.clone(),
        mesh_count: cpu.mesh_count,
        instance_total,
        blend_sort_groups,
        index_format: if cpu.index_u16 { IndexFormat::Uint16 } else { IndexFormat::Uint32 },
    });
    commands.insert_resource(EftCullBindGroup(cull_bg));
    commands.insert_resource(EftDrawBindGroup(draw_bg));
    // The map is now fully built + about to draw: clear the cross-world loading flag so the
    // main-world `map_loading_indicator` toast dismisses.
    if let Some(s) = &load_signal {
        s.set(false);
    }
    eprintln!(
        "[stall] prepare_gpu_buffers FINALIZE frame (render thread): {:.1} ms  \
         [geo-finalize {:.1} ({:.0}MiB vtx + {:.0}MiB idx {}) | albedo {} tex {:.1} | normal {} tex {:.1} | SH+shadows {:.1}]{}",
        prep_t0.elapsed().as_secs_f64() * 1000.0,
        geo_ms,
        std::mem::size_of_val(cpu.vertex_data.as_slice()) as f64 / 1048576.0,
        cpu.index_bytes.len() as f64 / 1048576.0,
        if geo_streamed { "STREAMED over load window" } else { "one-shot" },
        tex_count,
        albedo_ms,
        normal_count,
        normal_ms,
        prep_t0.elapsed().as_secs_f64() * 1000.0 - geo_ms - albedo_ms - normal_ms,
        if sync_load { "  (EFT_SYNC_LOAD: whole build in this one frame)" } else { "" },
    );
    info!("gpu-driven: GPU buffers + bind groups built (once)");
}

// ---- M3 texture upload helpers ---------------------------------------------
/// Decode one albedo PNG (full-res, `image` crate) and upload it as an Rgba8UnormSrgb GPU
/// texture (+ view). Albedo is sRGB (conventions.colorSpace.albedo='srgb') so the srgb
/// format makes the sampler return linear. On ANY read/decode failure returns a 1x1 magenta
/// placeholder so the bindless-array index stays aligned with materials.json â€” a shifted
/// index would texture the whole map wrong with no error.
/// True when a puddle albedo's ALPHA channel is (near) constant — so the puddle shape mask lives
/// in the RGB/luma channel instead (City_puddle_atlas ships alpha≡1.0). Sampled on a big stride
/// (only ~38 water textures per map, at load). Undecodable -> false (assume the alpha mask).
fn puddle_alpha_is_constant(path: &str) -> bool {
    let Ok(img) = image::open(path) else {
        return false;
    };
    let rgba = img.to_rgba8();
    let (mut lo, mut hi) = (255u8, 0u8);
    for px in rgba.pixels().step_by(101) {
        let a = px.0[3];
        lo = lo.min(a);
        hi = hi.max(a);
    }
    (hi - lo) < 13 // < ~0.05 of full range
}

/// Loot-glow model match, AUTHORITATIVE: every ACTIVE gamedata LootableContainer -> the GPU
/// instances that share its PREFAB ANCESTRY. The container record carries its folded transform
/// chain (`tf` = self/parent/grandparent, from the game's own scene hierarchy); every shipped
/// instance carries its renderer's folded `par`/`par2` + source level. An instance belongs to a
/// container iff the two chains INTERSECT at the same level. No names, no radius: a decorative
/// crate stacked beside a lootable one shares neither ancestor (the streets false-positive), and
/// a prefab part whose pivot sits meters away still joins (the suitcase false-negative).
///
/// Scene-ORGANIZATION nodes (one parent holding dozens of containers) would over-join, so any
/// ancestor id claimed by more than 3 containers on its level is dropped from every chain — a
/// real multi-container prop prefab (a stacked-crates set) stays under the cap; "Design_Stuff"
/// roots do not. `tf[0]` (the container's own transform) is always kept: it is unique.
///
/// Returns (gamedata container index, model-center world pos, instances). A container with no
/// model isn't listed — loot.rs keeps its box marker so the overlay never silently loses an
/// item. Packs or gamedata from before the ancestry capture yield an EMPTY result: authoritative
/// or nothing, per the project's derive-don't-author rule.
fn match_loot_models(
    root: &std::path::Path,
    instances: &[InstanceGpuRecord],
    inst_ancestry: &[(u32, u32, u32)],
) -> Vec<(u32, [f32; 3], Vec<u32>)> {
    let Ok(txt) = std::fs::read_to_string(root.join("gamedata.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return Vec::new();
    };
    let none = Vec::new();
    let containers = v
        .get("containers")
        .and_then(|c| c.as_array())
        .unwrap_or(&none);
    if containers.is_empty() {
        return Vec::new();
    }
    // Container chains: (index, lv, kept tf ids). Absent `tf` (pre-capture gamedata) -> skipped.
    let mut chains: Vec<(u32, u32, Vec<u32>)> = Vec::new();
    let mut ancestor_claims: std::collections::HashMap<(u32, u32), u32> = Default::default();
    for (ci, c) in containers.iter().enumerate() {
        if !c.get("active").and_then(|a| a.as_bool()).unwrap_or(true) {
            continue; // not present at raid start — no marker, no glow
        }
        let lv = c.get("lv").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        let tf: Vec<u32> = c
            .get("tf")
            .and_then(|t| t.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect())
            .unwrap_or_default();
        if tf.is_empty() {
            continue;
        }
        for (k, &id) in tf.iter().enumerate() {
            if k > 0 && id != 0 {
                *ancestor_claims.entry((lv, id)).or_insert(0) += 1;
            }
        }
        chains.push((ci as u32, lv, tf));
    }
    if chains.is_empty() {
        return Vec::new();
    }
    // (lv, folded id) -> container-chain slots that keep it (organization nodes dropped).
    let mut key_to_chain: std::collections::HashMap<(u32, u32), Vec<u32>> = Default::default();
    for (slot, (_, lv, tf)) in chains.iter().enumerate() {
        for (k, &id) in tf.iter().enumerate() {
            if id == 0 || (k > 0 && ancestor_claims.get(&(*lv, id)).copied().unwrap_or(0) > 3) {
                continue;
            }
            key_to_chain.entry((*lv, id)).or_default().push(slot as u32);
        }
    }
    // One pass over the instances: an instance whose (lv, par) or (lv, par2) is a kept key
    // belongs to those containers.
    let mut hits: Vec<Vec<u32>> = vec![Vec::new(); chains.len()];
    for (i, &(par, par2, lv)) in inst_ancestry.iter().enumerate() {
        for id in [par, par2] {
            if id == 0 {
                continue;
            }
            if let Some(slots) = key_to_chain.get(&(lv, id)) {
                for &s in slots {
                    hits[s as usize].push(i as u32);
                }
            }
        }
    }
    let mut out: Vec<(u32, [f32; 3], Vec<u32>)> = Vec::new();
    let mut missed = 0usize;
    for (slot, (ci, _, _)) in chains.iter().enumerate() {
        let mut idxs = std::mem::take(&mut hits[slot]);
        idxs.sort_unstable();
        idxs.dedup();
        if idxs.is_empty() {
            missed += 1;
            continue;
        }
        // Marker anchor = the MODEL's world center (mean of instance bounding-sphere centers) —
        // the container's own pivot can sit hundreds of meters from the visible prop (DesignStuff
        // scenes author verts in scene space with near-origin pivots).
        let mut c = Vec3::ZERO;
        for &i in &idxs {
            let s = instances[i as usize].sphere;
            c += Vec3::new(s[0], s[1], s[2]);
        }
        c /= idxs.len() as f32;
        out.push((*ci, [c.x, c.y, c.z], idxs));
    }
    info!(
        "loot-glow: {}/{} active containers ancestry-matched to scene models ({} instance refs); \
         {missed} without a model keep their marker box",
        out.len(),
        chains.len(),
        out.iter().map(|(_, _, v)| v.len()).sum::<usize>(),
    );
    out
}

/// True when a glass albedo's alpha channel is a COVERAGE mask (mostly fully-transparent texels)
/// rather than packed smoothness. Strided sampling like `puddle_alpha_is_constant`; undecodable
/// counts as smoothness (keep the streets-calibrated RFA behavior).
fn glass_alpha_is_mask(path: &str) -> bool {
    let Ok(img) = image::open(path) else {
        return false;
    };
    let rgba = img.to_rgba8();
    let (mut zero, mut n) = (0u32, 0u32);
    for px in rgba.pixels().step_by(101) {
        n += 1;
        if px.0[3] < 26 {
            zero += 1;
        }
    }
    n > 0 && (zero as f32 / n as f32) > 0.4
}

fn load_albedo_texture(
    device: &RenderDevice,
    queue: &RenderQueue,
    path: &str,
) -> (Texture, TextureView) {
    match std::fs::read(path) {
        Ok(bytes) => {
            // Content-hash first: a shared-cache hit skips PNG decode AND BC encode entirely.
            let hash = fnv64(&bytes);
            if bc_enabled(device) {
                if let Some((w, h, mips, payload)) = texcache_read(hash, "bc3c") {
                    return upload_bc3(device, queue, w, h, mips, &payload, true, "eft_albedo");
                }
            }
            let Ok(img) = image::load_from_memory(&bytes) else {
                warn!("gpu-driven M3: albedo '{path}' failed to decode; using placeholder");
                return upload_rgba8_srgb(device, queue, 1, 1, &[255u8, 0, 255, 255], "eft_albedo_missing");
            };
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            if bc_wanted(device, w, h) {
                let (mips, chain) = build_mip_chain(w, h, &rgba);
                let payload = bc3_compress_chain(w, h, mips, &chain);
                texcache_write(hash, "bc3c", w, h, mips, &payload);
                return upload_bc3(device, queue, w, h, mips, &payload, true, "eft_albedo");
            }
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
    // Normal maps compress to BC5 (Rg, LINEAR): tangent XY in two dedicated interpolated channels
    // (Z reconstructed in the shader). Unlike BC3 — whose BC1-quality RGB565 block crushes the small
    // X/Y relief to flat — BC5 preserves the relief, at 8 bpp (4x smaller than the raw Rgba8 normals
    // used to upload as: ~6.3 GB -> ~1.6 GB on lighthouse). Cached under .bc5c (the .bc3c albedo cache
    // stores a DIFFERENT format, so the extensions must not be shared). Sync mirror of prepare_tex_cpu.
    let flat = |d: &RenderDevice, q: &RenderQueue| {
        upload_rgba8_linear(d, q, 1, 1, &[128u8, 128, 255, 255], "eft_normal_missing")
    };
    match std::fs::read(path) {
        Ok(bytes) => {
            let hash = fnv64(&bytes);
            if bc_enabled(device) {
                if let Some((w, h, mips, payload)) = texcache_read(hash, "bc5c") {
                    return upload_bc5(device, queue, w, h, mips, &payload, "eft_normal");
                }
            }
            let Ok(img) = image::load_from_memory(&bytes) else {
                warn!("gpu-driven Phase2b: normal '{path}' failed to decode; flat placeholder");
                return flat(device, queue);
            };
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            if bc_wanted(device, w, h) {
                let (mips, chain) = build_mip_chain(w, h, &rgba);
                let payload = bc5_compress_chain(w, h, mips, &chain);
                texcache_write(hash, "bc5c", w, h, mips, &payload);
                return upload_bc5(device, queue, w, h, mips, &payload, "eft_normal");
            }
            upload_rgba8_linear(device, queue, w.max(1), h.max(1), &rgba, "eft_normal")
        }
        Err(e) => {
            warn!("gpu-driven Phase2b: normal '{path}' failed to load ({e}); using flat placeholder");
            flat(device, queue)
        }
    }
}

/// DATA textures (terrain control maps, vp heights masks): LINEAR and NEVER block-compressed —
/// they are exact blend weights, and BC3's palette interpolation would warp them (visible splat
/// banding). Small population (~35 textures), negligible VRAM.
fn load_data_texture(
    device: &RenderDevice,
    queue: &RenderQueue,
    path: &str,
) -> (Texture, TextureView) {
    match image::open(path) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            upload_rgba8_linear(device, queue, w.max(1), h.max(1), &rgba, "eft_data")
        }
        Err(e) => {
            warn!("gpu-driven: data map '{path}' failed to load ({e}); using placeholder");
            upload_rgba8_linear(device, queue, 1, 1, &[0u8, 0, 0, 255], "eft_data_missing")
        }
    }
}

/// 1x1 flat tangent normal (128,128,255 -> +Z) for a pack that referenced no normal maps at all
/// (keeps the `normal_tex` binding_array non-empty).
fn make_dummy_normal_texture(device: &RenderDevice, queue: &RenderQueue) -> (Texture, TextureView) {
    upload_rgba8_linear(device, queue, 1, 1, &[128u8, 128, 255, 255], "eft_normal_dummy")
}

/// Build a full mip chain from mip0 RGBA8 bytes. Each level is Triangle-resampled from the
/// PREVIOUS level ((w>>l).max(1) — the .max(1) matters for non-square textures whose short axis
/// hits 1 early). Returns (mip_count, concatenated level bytes). Without mips every distant
/// surface point-samples mip0 -> the severe far-field shimmer (opposite of EFT's soft look) and
/// texture-cache thrash. 1x1 placeholders return (1, mip0) untouched.
fn build_mip_chain(width: u32, height: u32, rgba: &[u8]) -> (u32, Vec<u8>) {
    let mips = 32 - width.max(height).leading_zeros(); // floor(log2)+1
    if mips <= 1 || rgba.len() != (width * height * 4) as usize {
        return (1, rgba.to_vec());
    }
    let mut data = Vec::with_capacity(rgba.len() * 4 / 3 + 64);
    data.extend_from_slice(rgba);
    let mut prev = match image::RgbaImage::from_raw(width, height, rgba.to_vec()) {
        Some(img) => img,
        None => return (1, rgba.to_vec()),
    };
    for l in 1..mips {
        let (mw, mh) = ((width >> l).max(1), (height >> l).max(1));
        let next = image::imageops::resize(&prev, mw, mh, image::imageops::FilterType::Triangle);
        data.extend_from_slice(&next);
        prev = next;
    }
    (mips, data)
}

/// BC3-compress a full RGBA8 mip chain (texpresso RangeFit — fast; the source PNGs were
/// decoded FROM the game's own BC textures, so re-encoding is quality-parity with the game).
/// Returns the concatenated per-mip BC3 payload. Каждый mip padded to 4x4 blocks by texpresso;
/// create_texture_with_data expects exactly ceil(w/4)*ceil(h/4)*16 per level, which matches.
fn bc3_compress_chain(width: u32, height: u32, mips: u32, chain: &[u8]) -> Vec<u8> {
    let fmt = texpresso::Format::Bc3;
    let params = texpresso::Params {
        algorithm: texpresso::Algorithm::RangeFit,
        ..Default::default()
    };
    let mut out = Vec::new();
    let mut off = 0usize;
    for l in 0..mips {
        let (mw, mh) = ((width >> l).max(1) as usize, (height >> l).max(1) as usize);
        let n = mw * mh * 4;
        let size = fmt.compressed_size(mw, mh);
        let base = out.len();
        out.resize(base + size, 0);
        fmt.compress(&chain[off..off + n], mw, mh, params, &mut out[base..]);
        off += n;
    }
    out
}

/// Encode one 4x4 block of a single 8-bit channel to a BC4 (8-byte) block: endpoints = block
/// max/min (r0 >= r1 -> the 8-value interpolation mode), each texel gets the nearest 3-bit index.
/// Pure Rust (no ISPC/C dep) — quality is ample for smooth tangent-space normals.
fn bc4_block(vals: &[u8; 16]) -> [u8; 8] {
    let (mut lo, mut hi) = (255u8, 0u8);
    for &v in vals {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let (r0, r1) = (hi, lo); // r0 >= r1 -> code0=r0, code1=r1, code2..7 = interpolated
    let mut refv = [r0; 8];
    refv[1] = r1;
    if r0 > r1 {
        // code k (k in 2..=7) = ((8-k)*r0 + (k-1)*r1)/7, rounded.
        for k in 2..8u32 {
            refv[k as usize] = (((8 - k) * r0 as u32 + (k - 1) * r1 as u32 + 3) / 7) as u8;
        }
    }
    let mut bits: u64 = 0;
    for (i, &v) in vals.iter().enumerate() {
        let mut best = 0u64;
        let mut bestd = i32::MAX;
        for (k, &rk) in refv.iter().enumerate() {
            let d = (v as i32 - rk as i32).abs();
            if d < bestd {
                bestd = d;
                best = k as u64;
            }
        }
        bits |= best << (3 * i);
    }
    let mut out = [0u8; 8];
    out[0] = r0;
    out[1] = r1;
    for (b, o) in out[2..8].iter_mut().enumerate() {
        *o = ((bits >> (8 * b)) & 0xFF) as u8;
    }
    out
}

/// BC5-compress an RGBA8 mip chain (tangent-space NORMAL maps): per 4x4 block, BC4(R) then BC4(G) =
/// 16 bytes (RG = tangent XY; the shader reconstructs Z). Same 16-byte-per-block layout as BC3, so
/// `create_texture_with_data` / `bc3_payload_len` accept it unchanged. 4x smaller than raw Rgba8 and,
/// unlike BC3, does NOT crush the small X/Y relief (each channel gets its own interpolated endpoints).
fn bc5_compress_chain(width: u32, height: u32, mips: u32, chain: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut off = 0usize;
    for l in 0..mips {
        let (mw, mh) = ((width >> l).max(1) as usize, (height >> l).max(1) as usize);
        let (bw, bh) = (mw.div_ceil(4), mh.div_ceil(4));
        for by in 0..bh {
            for bx in 0..bw {
                let (mut rr, mut gg) = ([0u8; 16], [0u8; 16]);
                for ty in 0..4 {
                    for tx in 0..4 {
                        let px = (bx * 4 + tx).min(mw - 1);
                        let py = (by * 4 + ty).min(mh - 1);
                        let idx = off + (py * mw + px) * 4;
                        rr[ty * 4 + tx] = chain[idx];
                        gg[ty * 4 + tx] = chain[idx + 1];
                    }
                }
                out.extend_from_slice(&bc4_block(&rr));
                out.extend_from_slice(&bc4_block(&gg));
            }
        }
        off += mw * mh * 4;
    }
    out
}

/// Cross-map BC3 texture cache, keyed by CONTENT HASH of the source PNG bytes — the same game
/// texture extracted into several map datasets (different filenames, identical bytes) encodes
/// ONCE and every map reuses it. Lives in packs/shared/texcache/<fnv64>.bc3c =
/// [w,h,mips: u32 LE] + concatenated BC3 mips. Content addressing self-invalidates.
fn texcache_path(hash: u64, ext: &str) -> std::path::PathBuf {
    crate::paths::shared_dir()
        .join("texcache")
        .join(format!("{hash:016x}.{ext}"))
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1_0000_0001_B3);
    }
    h
}

/// Cache read: (w, h, mips, payload) when present.
/// Exact byte length `bc3_compress_chain` produces for a (w,h,mips) BC3 payload (same per-mip
/// `compressed_size` accumulation) — lets `texcache_read` reject a truncated/corrupt entry.
fn bc3_payload_len(width: u32, height: u32, mips: u32) -> usize {
    let fmt = texpresso::Format::Bc3;
    (0..mips)
        .map(|l| fmt.compressed_size((width >> l).max(1) as usize, (height >> l).max(1) as usize))
        .sum()
}

fn texcache_read(hash: u64, ext: &str) -> Option<(u32, u32, u32, Vec<u8>)> {
    let bytes = std::fs::read(texcache_path(hash, ext)).ok()?;
    if bytes.len() <= 12 {
        return None;
    }
    let w = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let h = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let m = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let payload = &bytes[12..];
    // Reject an implausible header or a wrong-length payload (e.g. a cache write interrupted by a
    // crash): treat as a MISS so the caller re-decodes from the source PNG rather than feeding a
    // short buffer into `create_texture_with_data`, which would panic/abort the process.
    if w == 0 || h == 0 || m == 0 || w > 16384 || h > 16384 || m > 16
        || payload.len() != bc3_payload_len(w, h, m)
    {
        return None;
    }
    Some((w, h, m, payload.to_vec()))
}

fn texcache_write(hash: u64, ext: &str, width: u32, height: u32, mips: u32, payload: &[u8]) {
    let p = texcache_path(hash, ext);
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut file = Vec::with_capacity(12 + payload.len());
    file.extend_from_slice(&width.to_le_bytes());
    file.extend_from_slice(&height.to_le_bytes());
    file.extend_from_slice(&mips.to_le_bytes());
    file.extend_from_slice(payload);
    // Atomic write: unique temp beside the target then rename, so a crash mid-write can never leave a
    // truncated entry. The unique suffix avoids a collision when two workers encode the same content
    // hash concurrently. Best-effort (a read-only fs just re-encodes next launch).
    static TMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = TMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = p.with_extension(format!("tmp{uniq:x}"));
    if std::fs::write(&tmp, &file).is_ok() && std::fs::rename(&tmp, &p).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Feature+env gate alone (no dims) — used to probe the shared cache BEFORE decoding.
fn bc_enabled(device: &RenderDevice) -> bool {
    use bevy::render::settings::WgpuFeatures;
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *DISABLED
        .get_or_init(|| std::env::var("EFT_TEX_BC").map(|v| v.trim() == "0").unwrap_or(false));
    !off && device.features().contains(WgpuFeatures::TEXTURE_COMPRESSION_BC)
}

/// True when BC compression should be used for this texture (feature present, not disabled,
/// large enough to matter — tiny placeholders/dummies stay RGBA8).
fn bc_wanted(device: &RenderDevice, width: u32, height: u32) -> bool {
    use bevy::render::settings::WgpuFeatures;
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *DISABLED
        .get_or_init(|| std::env::var("EFT_TEX_BC").map(|v| v.trim() == "0").unwrap_or(false));
    !off && width >= 64
        && height >= 64
        && device.features().contains(WgpuFeatures::TEXTURE_COMPRESSION_BC)
}

/// Upload a pre-built BC3 mip payload as a texture (sRGB or linear view of the same bits).
fn upload_bc3(
    device: &RenderDevice,
    queue: &RenderQueue,
    width: u32,
    height: u32,
    mips: u32,
    payload: &[u8],
    srgb: bool,
    label: &'static str,
) -> (Texture, TextureView) {
    let tex = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: if srgb {
                TextureFormat::Bc3RgbaUnormSrgb
            } else {
                TextureFormat::Bc3RgbaUnorm
            },
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        payload,
    );
    let view = tex.create_view(&TextureViewDescriptor::default());
    (tex, view)
}

/// Upload a pre-built BC5 (Rg) mip payload as a LINEAR normal-map texture — tangent XY (Z is
/// reconstructed in the shader). 8 bpp = 4x smaller than the raw Rgba8 the normals used to upload as.
fn upload_bc5(
    device: &RenderDevice,
    queue: &RenderQueue,
    width: u32,
    height: u32,
    mips: u32,
    payload: &[u8],
    label: &'static str,
) -> (Texture, TextureView) {
    let tex = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bc5RgUnorm, // LINEAR two-channel; normals are vectors, not color
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        payload,
    );
    let view = tex.create_view(&TextureViewDescriptor::default());
    (tex, view)
}

/// Upload a PRE-BUILT RGBA8 mip chain as an sRGB or linear texture. `create_texture_with_data`
/// handles the 256-byte row-padding for the staging copy (per mip). Shared by the sync uploaders
/// (which build the chain inline) and the async `upload_prepared` (chain built off-thread).
fn upload_rgba8_chain(
    device: &RenderDevice,
    queue: &RenderQueue,
    width: u32,
    height: u32,
    mips: u32,
    chain: &[u8],
    srgb: bool,
    label: &'static str,
) -> (Texture, TextureView) {
    let tex = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: if srgb {
                TextureFormat::Rgba8UnormSrgb
            } else {
                TextureFormat::Rgba8Unorm // LINEAR — normal vectors / data maps, not color
            },
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        chain,
    );
    let view = tex.create_view(&TextureViewDescriptor::default());
    (tex, view)
}

fn upload_rgba8_srgb(
    device: &RenderDevice,
    queue: &RenderQueue,
    width: u32,
    height: u32,
    rgba: &[u8],
    label: &'static str,
) -> (Texture, TextureView) {
    let (mips, chain) = build_mip_chain(width, height, rgba);
    upload_rgba8_chain(device, queue, width, height, mips, &chain, true, label)
}

/// Phase 2b: upload RGBA8 bytes as a LINEAR (Rgba8Unorm) texture — for normal maps, whose texels
/// are tangent-space vectors, not sRGB color. Identical to `upload_rgba8_srgb` but for the format.
/// (Mipping normals by box filter denormalizes them slightly; the shader renormalizes after
/// perturbation, and shortened far-mip normals actually soften distant spec — desirable here.)
fn upload_rgba8_linear(
    device: &RenderDevice,
    queue: &RenderQueue,
    width: u32,
    height: u32,
    rgba: &[u8],
    label: &'static str,
) -> (Texture, TextureView) {
    let (mips, chain) = build_mip_chain(width, height, rgba);
    upload_rgba8_chain(device, queue, width, height, mips, &chain, false, label)
}

// ---- Off-thread texture preparation (fixes the "Not Responding" load freeze) ----------------
// The freeze was `prepare_gpu_buffers` decoding+BC-encoding+uploading ALL ~700 albedo + ~540 normal
// textures SYNCHRONOUSLY in one render-thread pass (cold: >40 s; even warm-cache: ~3.9 s of
// disk-read + upload). That blocked the render thread, which stalls the main thread's next extract,
// which freezes the winit message pump -> Windows "Not Responding". Fix (hybrid A+B): the CPU-heavy
// half (fs::read + PNG decode + mip + BC3 encode, or a warm shared-cache read) runs OFF-THREAD on
// the AsyncComputeTaskPool (parallel across cores); the render thread only polls finished payloads
// and does the fast `create_texture_with_data` uploads, TIME-BUDGETED across frames so no single
// frame stalls. See `prepare_gpu_buffers` for the per-frame state machine.

/// CPU-side texture payload produced OFF-THREAD by `prepare_tex_cpu` — the expensive work with NO
/// GPU handles, so it parallelizes across cores while the render thread only does the fast upload.
enum TexCpu {
    /// A finished BC3 mip payload (`upload_bc3`); `srgb` is decided per-array at upload time.
    Bc3 { w: u32, h: u32, mips: u32, payload: Vec<u8> },
    /// A finished BC5 (Rg) mip payload for tangent-space NORMAL maps (`upload_bc5`, always LINEAR).
    Bc5 { w: u32, h: u32, mips: u32, payload: Vec<u8> },
    /// A finished RGBA8 mip chain (small textures / data maps / decode failures): uploaded sRGB or
    /// linear per-array. The mip chain is built OFF-THREAD here (not in `upload_*`) so the render
    /// thread only does the GPU copy — otherwise a large data map's mip build stalls a frame.
    Raw { w: u32, h: u32, mips: u32, chain: Vec<u8> },
}

/// TEXTURE QUALITY (menu setting): how many TOP mip levels to drop at upload. 0 = full,
/// 1 = half resolution, 2 = quarter. The VRAM audit (docs/VRAM_AUDIT.md) measured textures at
/// 59% of streets' 8.7 GiB residency, all uploaded full-res; dropping one level reclaims ~3.8 GiB
/// on that map. Static atomic (the game_watch flag pattern): the prepare tasks run off-thread and
/// the kickoff captures the value once per map build — changing it live applies on the NEXT map
/// (re)load, which the menu tooltip says out loud.
pub static TEX_MIP_SKIP: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_tex_mip_skip(n: u8) {
    TEX_MIP_SKIP.store(n.min(2), std::sync::atomic::Ordering::Relaxed);
}

/// Drop the first `skip` mip levels of an already-prepared chain — pure byte slicing, no
/// re-encode (both the texcache and fresh encodes store full concatenated chains, so "half
/// resolution" is literally "start the upload at level 1").
///
/// Two guards keep it safe and worth it:
///   * stop shrinking once the longest side would fall below 128 px — small textures are cheap,
///     and crushing them buys nothing;
///   * for BC formats never produce a base level that isn't a multiple of the 4px block (wgpu
///     validates base-level dimensions against the block size).
fn slice_mips(tex: TexCpu, skip: u32) -> TexCpu {
    if skip == 0 {
        return tex;
    }
    fn effective(w: u32, h: u32, mips: u32, skip: u32, block4: bool) -> u32 {
        let mut e = skip.min(mips.saturating_sub(1));
        while e > 0 {
            let (nw, nh) = ((w >> e).max(1), (h >> e).max(1));
            let too_small = nw.max(nh) < 128;
            let misaligned = block4 && (nw % 4 != 0 || nh % 4 != 0);
            if too_small || misaligned {
                e -= 1;
            } else {
                break;
            }
        }
        e
    }
    // Byte offset of level `e` in a BC chain (16-byte blocks for both BC3 and BC5).
    fn bc_offset(w: u32, h: u32, e: u32) -> usize {
        (0..e)
            .map(|l| {
                let (mw, mh) = (((w >> l).max(1) as usize), ((h >> l).max(1) as usize));
                mw.div_ceil(4) * mh.div_ceil(4) * 16
            })
            .sum()
    }
    match tex {
        TexCpu::Bc3 { w, h, mips, payload } => {
            let e = effective(w, h, mips, skip, true);
            if e == 0 {
                return TexCpu::Bc3 { w, h, mips, payload };
            }
            let off = bc_offset(w, h, e);
            TexCpu::Bc3 {
                w: (w >> e).max(1),
                h: (h >> e).max(1),
                mips: mips - e,
                payload: payload[off..].to_vec(),
            }
        }
        TexCpu::Bc5 { w, h, mips, payload } => {
            let e = effective(w, h, mips, skip, true);
            if e == 0 {
                return TexCpu::Bc5 { w, h, mips, payload };
            }
            let off = bc_offset(w, h, e);
            TexCpu::Bc5 {
                w: (w >> e).max(1),
                h: (h >> e).max(1),
                mips: mips - e,
                payload: payload[off..].to_vec(),
            }
        }
        TexCpu::Raw { w, h, mips, chain } => {
            let e = effective(w, h, mips, skip, false);
            if e == 0 {
                return TexCpu::Raw { w, h, mips, chain };
            }
            let off: usize = (0..e)
                .map(|l| ((w >> l).max(1) as usize) * ((h >> l).max(1) as usize) * 4)
                .sum();
            TexCpu::Raw {
                w: (w >> e).max(1),
                h: (h >> e).max(1),
                mips: mips - e,
                chain: chain[off..].to_vec(),
            }
        }
    }
}

/// OFF-THREAD texture preparation: exactly the CPU half of `load_albedo_texture` /
/// `load_normal_texture` / `load_data_texture` (fs::read -> content hash -> warm shared-cache read
/// OR PNG decode + mip chain + BC3 encode + cache write), returning a `TexCpu` the render thread
/// uploads. NO `RenderDevice`/`RenderQueue`, so N of these run in parallel on the task pool.
/// `bc` = BC compression enabled on this device (captured before spawn, folds in the feature + env
/// gate). `data_linear` = terrain control/data map (raw RGBA, NEVER BC — BC's palette warps blend
/// weights). `is_normal` = tangent-space normal map -> BC5 (Rg, tangent XY; Z reconstructed in the
/// shader): 4x smaller than raw Rgba8, and unlike BC3 no X/Y-relief crush. `placeholder` = the 1x1
/// fill on any load/decode failure (magenta for albedo, flat normal for normals) so the bindless
/// array index stays aligned with materials.json.
fn prepare_tex_cpu(
    path: String,
    bc: bool,
    data_linear: bool,
    no_downscale: bool,
    is_normal: bool,
    placeholder: [u8; 4],
    mip_skip: u32,
) -> TexCpu {
    let tex = prepare_tex_cpu_inner(path, bc, data_linear, is_normal, placeholder);
    if no_downscale {
        return tex; // terrain blend weights: the resolution IS the data — never degrade
    }
    slice_mips(tex, mip_skip)
}

fn prepare_tex_cpu_inner(path: String, bc: bool, data_linear: bool, is_normal: bool, placeholder: [u8; 4]) -> TexCpu {
    // Cache is keyed by format extension so BC5 (normal) and BC3 (albedo) entries never collide.
    let cache_ext = if is_normal { "bc5c" } else { "bc3c" };
    // Build the RGBA8 mip chain OFF-THREAD (the render thread just copies it) — mirrors what
    // `upload_rgba8_srgb/linear` did inline, moved here so a big data map can't stall a frame.
    let raw = |w: u32, h: u32, rgba: &[u8]| {
        let (mips, chain) = build_mip_chain(w.max(1), h.max(1), rgba);
        TexCpu::Raw { w: w.max(1), h: h.max(1), mips, chain }
    };
    if data_linear {
        // Control/data map: raw linear, never block-compressed (mirrors `load_data_texture`).
        return match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                raw(w, h, &rgba)
            }
            Err(e) => {
                warn!("gpu-driven: data map '{path}' failed to load ({e}); using placeholder");
                raw(1, 1, &[0, 0, 0, 255])
            }
        };
    }
    match std::fs::read(&path) {
        Ok(bytes) => {
            // Content-hash first: a shared-cache hit skips PNG decode AND BC encode entirely.
            let hash = fnv64(&bytes);
            if bc {
                if let Some((w, h, mips, payload)) = texcache_read(hash, cache_ext) {
                    return if is_normal {
                        TexCpu::Bc5 { w, h, mips, payload }
                    } else {
                        TexCpu::Bc3 { w, h, mips, payload }
                    };
                }
            }
            let Ok(img) = image::load_from_memory(&bytes) else {
                warn!("gpu-driven: texture '{path}' failed to decode; using placeholder");
                return raw(1, 1, &placeholder);
            };
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            // Same threshold as `bc_wanted` (feature already folded into `bc`): >= 64px each axis.
            if bc && w >= 64 && h >= 64 {
                let (mips, chain) = build_mip_chain(w, h, &rgba);
                if is_normal {
                    let payload = bc5_compress_chain(w, h, mips, &chain);
                    texcache_write(hash, cache_ext, w, h, mips, &payload);
                    return TexCpu::Bc5 { w, h, mips, payload };
                }
                let payload = bc3_compress_chain(w, h, mips, &chain);
                texcache_write(hash, cache_ext, w, h, mips, &payload);
                return TexCpu::Bc3 { w, h, mips, payload };
            }
            raw(w, h, &rgba)
        }
        Err(e) => {
            warn!("gpu-driven: texture '{path}' failed to load ({e}); using placeholder");
            raw(1, 1, &placeholder)
        }
    }
}

/// Upload a `TexCpu` (produced off-thread) to the GPU — the fast half that MUST stay on the render
/// thread. `srgb` selects the format (albedo = sRGB; normals + data maps = linear). Byte-identical
/// to what the old inline `load_*_texture` path produced.
fn upload_prepared(
    device: &RenderDevice,
    queue: &RenderQueue,
    tex: &TexCpu,
    srgb: bool,
    label: &'static str,
) -> (Texture, TextureView) {
    match tex {
        TexCpu::Bc3 { w, h, mips, payload } => {
            upload_bc3(device, queue, *w, *h, *mips, payload, srgb, label)
        }
        TexCpu::Bc5 { w, h, mips, payload } => {
            upload_bc5(device, queue, *w, *h, *mips, payload, label) // normals: always LINEAR
        }
        TexCpu::Raw { w, h, mips, chain } => {
            // Mip chain already built off-thread — the render thread only does the GPU copy.
            upload_rgba8_chain(device, queue, *w, *h, *mips, chain, srgb, label)
        }
    }
}

/// Per-map GPU texture-build progress, held across frames in the RENDER world while
/// `prepare_gpu_buffers` streams textures in. Present only DURING a build (kickoff -> finalize);
/// removed when the build completes or the map swaps (`reset_gpu_map_if_epoch_changed`). Each frame
/// polls the finished off-thread tasks and uploads a time-budgeted batch, so the render thread never
/// stalls. `EFT_SYNC_LOAD=1` bypasses all of this (the whole build runs in one synchronous frame).
#[derive(Resource)]
struct GpuBuildState {
    /// The `MapEpoch` this build is for; a newer epoch (map swap) discards it and re-kicks.
    epoch: u64,
    /// Off-thread CPU-prep tasks, in `albedo_paths` order. `Some` until polled+uploaded, then `None`.
    albedo_tasks: Vec<Option<bevy::tasks::Task<TexCpu>>>,
    normal_tasks: Vec<Option<bevy::tasks::Task<TexCpu>>>,
    /// Uploaded `(Texture, View)` in the same order; `Some` once its task finished + uploaded.
    albedo_tex: Vec<Option<(Texture, TextureView)>>,
    normal_tex: Vec<Option<(Texture, TextureView)>>,
    /// Instrumentation: wall-clock start, frames spent, and the longest single render-thread stall.
    started: std::time::Instant,
    frames: u32,
    peak_ms: f64,
    /// Step 4: the ~1.1 GiB vertex+index upload, streamed in budgeted chunks across these same
    /// frames (via `write_buffer`) instead of one big `create_buffer_with_data` memcpy in the
    /// finalize frame. The buffers are created empty at kickoff and filled progressively.
    geo: GeoStream,
}

/// Streamed geometry upload state (Step 4). Buffers are created empty (COPY_DST) at kickoff and
/// filled a chunk at a time each PROGRESS frame; the finalize block reuses them once full.
struct GeoStream {
    vertex: Buffer,
    index: Buffer,
    vtx_total: usize, // total bytes
    idx_total: usize,
    vtx_cursor: usize, // bytes written so far (4-byte aligned; f32/u32 records)
    idx_cursor: usize,
}

/// Per-frame render-thread upload budget (ms). Uploads run until this is exceeded, then yield to
/// the next frame — keeps every frame well under a frame budget so the message pump + egui stay
/// live. Tunable via `EFT_LOAD_BUDGET_MS`; default 6 ms (fast reveal, no perceptible hitch).
fn upload_budget_ms() -> f64 {
    static MS: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("EFT_LOAD_BUDGET_MS")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0) // reject inf/NaN: Duration::from_secs_f64 panics on them
            .unwrap_or(6.0)
    })
}

// ---- PrepareResources: upload the 6 frustum planes (tiny) each frame --------
fn upload_frustum(
    render_queue: Res<RenderQueue>,
    buffers: Option<Res<EftGpuBuffers>>,
    // #5 shadows: when enabled, extrude the frustum toward the sun so off-screen casters survive
    // the SHARED cull and appear in the shadow map. `None`/disabled -> the cull is byte-identical
    // to before (perfect A/B against not EFT_SHADOWS=1).
    shadow: Option<Res<EftShadowConfig>>,
    settings: Option<Res<crate::render::GfxSettings>>,
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
    // Screen-size cull constants: an instance is culled when its bounding sphere subtends fewer
    // than `min_px` pixels — sphere.w < k * distance, with k = min_px / (0.5 * viewport_h * proj11).
    // Grass gets a larger threshold (a ~1.3 m clump is invisible long before the far plane; this
    // is where the 100k-clump draw cost goes). Values come from GfxSettings (UI-tunable; defaults
    // seeded from EFT_CULL_PX). Grass OFF = an enormous k so every clump culls at any distance.
    let (px_gen, px_grass) = match settings.as_ref() {
        Some(s) => (s.cull_px, if s.grass { s.cull_px_grass } else { f32::MAX }),
        None => (1.5, 4.0),
    };
    let proj11 = view.clip_from_view.y_axis.y; // 1/tan(fov_y/2)
    let vh = view.viewport.w.max(1) as f32;
    let denom = (0.5 * vh * proj11).max(1e-4);
    let cam_pos = view.world_from_view.translation();
    let uniform = CullUniform {
        frustum: [
            planes[0].to_array(),
            planes[1].to_array(),
            planes[2].to_array(),
            planes[3].to_array(),
            planes[4].to_array(),
            planes[5].to_array(),
        ],
        counts: [
            buffers.instance_total,
            buffers.mesh_count,
            (px_grass / denom).to_bits(),
            // Grass distance clamp in metres. Gated on `grass` so the master toggle still wins:
            // grass OFF already sets an enormous k above, and a finite limit here must not read as
            // "grass is on out to N metres".
            settings
                .as_ref()
                .filter(|s| s.grass)
                .map(|s| s.grass_dist_m.max(0.0))
                .unwrap_or(0.0)
                .to_bits(),
        ],
        cam_k: [cam_pos.x, cam_pos.y, cam_pos.z, px_gen / denom],
        // distance-LOD: proj11 for the screen-height metric; mode/bias from the graphics panel
        // (default mode 0 = max detail = today's look). Sentinel-window instances ignore it.
        lod_params: {
            let (mode, bias, forced) = settings
                .as_ref()
                .map(|g| {
                    if g.lod_force >= 0 {
                        (2.0, 1.0, g.lod_force as f32) // force shell N (debug)
                    } else if g.lod_distance {
                        // B7: sanitize the bias. The UI slider is 0.25..=4.0 but EFT_LOD_BIAS is raw
                        // env input; an inf/NaN bias makes proj11*bias overflow and every window
                        // comparison NaN out (total blackout). f32::clamp keeps NaN as NaN, so map
                        // non-finite -> 1.0 first, then clamp the finite range.
                        let bias = if g.lod_bias.is_finite() { g.lod_bias.clamp(0.05, 64.0) } else { 1.0 };
                        (1.0, bias, 0.0) // distance-based
                    } else {
                        (0.0, 1.0, 0.0) // max detail (default)
                    }
                })
                .unwrap_or((0.0, 1.0, 0.0));
            [proj11, bias, mode, forced]
        },
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
/// Live lighting controls: rewrite the 48-byte LightGrid uniform each frame as the as-built BASE
/// values x the UI multipliers (GfxSettings.lights / light_intensity / gi_intensity / sun_diffuse).
/// At the default multipliers this writes bytes identical to the build, so the shipped look is
/// untouched; a slider change lands the same frame with no rebuild. params.w stays 0 on full bakes
/// (base is 0), so the sun-diffuse slider can never double-count a baked sun.
fn update_light_uniform(
    render_queue: Res<RenderQueue>,
    res: Option<Res<EftShResources>>,
    settings: Option<Res<crate::render::GfxSettings>>,
) {
    let (Some(res), Some(g)) = (res, settings) else {
        return;
    };
    let mut u = res.light_base;
    u.params[0] *= if g.lights { g.light_intensity.max(0.0) } else { 0.0 };
    u.params[1] *= g.gi_intensity.max(0.0);
    u.params[3] *= g.sun_diffuse.max(0.0);
    render_queue.write_buffer(&res.light_uniform, 0, bytemuck::bytes_of(&u));
}

/// Zero the color lane of every light whose power-group bit is not set in `mask` (group -1 = always
/// lit). records = 3 vec4/light: [3i]=(pos,range) [3i+1]=(color,cos_outer) [3i+2]=(dir,cos_inner).
fn apply_light_power_records(records: &mut [[f32; 4]], group: &[i32], mask: u32) {
    for (i, &grp) in group.iter().enumerate() {
        let on = grp < 0 || (grp < 32 && (mask >> grp) & 1 == 1);
        if !on {
            if let Some(c) = records.get_mut(3 * i + 1) {
                c[0] = 0.0;
                c[1] = 0.0;
                c[2] = 0.0; // keep cos_outer in .w
            }
        }
    }
}

/// Live POWER SWITCH toggle: when the per-group power bitmask (`GfxSettings.light_groups`) changes,
/// rewrite the packed light buffer from the base records with the unpowered groups' colors zeroed.
/// The light buffer is small (a few hundred lights), the CSR grid + positions never change, and the
/// default state (mask 0) matches the as-built buffer, so this is a no-op until a switch is flipped.
fn update_light_power(
    render_queue: Res<RenderQueue>,
    res: Option<Res<EftShResources>>,
    settings: Option<Res<crate::render::GfxSettings>>,
    mut last_mask: Local<Option<u32>>,
) {
    let (Some(res), Some(g)) = (res, settings) else {
        return;
    };
    let mask = g.light_groups;
    if *last_mask == Some(mask) {
        return; // unchanged since last frame — nothing to re-upload
    }
    *last_mask = Some(mask);
    let mut records = res.light_records_base.clone();
    apply_light_power_records(&mut records, &res.light_group, mask);
    render_queue.write_buffer(&res.lights_buf, 0, bytemuck::cast_slice(&records));
}

// ============================================================================
// DOORS: click-to-open swing. A swing door's panel instance sits at its hinge origin (Codex audit:
// panel is a direct child at ~identity, its local X=0 edge is the hinge), so opening it is a
// rotation of the instance's linear part about the door's local-Z axis, keeping translation. We
// match each gamedata door to its GPU instance by pivot proximity at finalize, then animate the
// matched instance record + re-upload it (like update_light_power, but per instance).
// ============================================================================

/// One renderer that swings with a door (panel, its glass, inlays).
struct DoorPart {
    gpu_idx: u32,            // index into the instance storage buffer
    closed: InstanceGpuRecord, // the CLOSED-pose record — the animation base
}

/// One matched, animatable door (render world). A door is several renderers rotating together
/// about ONE hinge (see `DoorPart`), so every part shares this record's axis/pivot/angle.
struct DoorInst {
    parts: Vec<DoorPart>,
    pivot: Vec3,                  // hinge point (viewer world) — parts rotate ABOUT it, not their own origin
    axis: Vec3,                   // swing axis (door local-Z in viewer world), normalized
    open_rad: f32,                // signed open angle in radians (progress 0->1 sweeps 0->open_rad)
    locked: bool,                 // locked+keyed doors don't swing on a plain click
    progress: f32,                // current 0=closed .. 1=open
    target: f32,                  // where it's heading
}

#[derive(Resource)]
struct EftDoors {
    doors: Vec<DoorInst>,
    instances_buf: Buffer,
}

/// Render-world half of the loot glow: the per-instance highlight lane bound at
/// group(1) binding(2) of the draw pass, rewritten only when the main world's
/// [`crate::loot::LootGlowState`] generation moves.
#[derive(Resource)]
struct EftLootGlow {
    buffer: Buffer,
    len: u32,
    last_gen: u64,
}

/// Everything needed to REBUILD the draw bind group when the AO lane view changes (SSAO toggle,
/// window resize) without re-running the map build. `bound_ao` tracks which live AO view is
/// currently bound (`None` = the white fallback).
#[derive(Resource)]
struct EftDrawBgInputs {
    instances: Buffer,
    visible: Buffer,
    loot_glow: Buffer,
    bound_ao: Option<bevy::render::render_resource::TextureViewId>,
}

/// Swap the draw bind group's AO binding between the live SSAO lane and the white fallback as
/// the toggle / target change. Runs after `prepare_ao_target` (ssao.rs chains it) so a fresh
/// resize target is bound the same frame it appears.
pub(crate) fn sync_draw_bg_ao(
    mut commands: Commands,
    device: Res<RenderDevice>,
    settings: Option<Res<super::GfxSettings>>,
    pipe: Option<Res<EftDrawPipeline>>,
    ssao_pipe: Option<Res<super::ssao::SsaoPipeline>>,
    ao: Option<Res<super::ssao::EftAoTarget>>,
    inputs: Option<ResMut<EftDrawBgInputs>>,
) {
    let (Some(pipe), Some(ssao_pipe), Some(mut inputs)) = (pipe, ssao_pipe, inputs) else {
        return;
    };
    let ssao_on = settings.map(|s| s.ssao).unwrap_or(false);
    let live = ao.as_ref().and_then(|t| t.view.as_ref()).filter(|_| ssao_on);
    let want = live.map(|v| v.id());
    if want == inputs.bound_ao {
        return;
    }
    let view = live.unwrap_or(&ssao_pipe.fallback_ao_view);
    let bg = device.create_bind_group(
        "eft_draw_bg",
        &pipe.ssbo_layout,
        &BindGroupEntries::with_indices((
            (0, inputs.instances.as_entire_binding()),
            (1, inputs.visible.as_entire_binding()),
            (2, inputs.loot_glow.as_entire_binding()),
            (
                3,
                bevy::render::render_resource::BindingResource::TextureView(view),
            ),
        )),
    );
    inputs.bound_ao = want;
    commands.insert_resource(EftDrawBindGroup(bg));
}

/// Scatter the visible loot set into the highlight lane. Full-buffer rewrite (u32 per instance,
/// ~370 KiB on ground_zero) on a GENERATION change only — toggle flips are user-rate events.
fn prepare_loot_glow(
    glow: Option<Res<crate::loot::LootGlowState>>,
    res: Option<ResMut<EftLootGlow>>,
    queue: Res<RenderQueue>,
) {
    let (Some(glow), Some(mut res)) = (glow, res) else {
        return;
    };
    if glow.gen == res.last_gen {
        return;
    }
    res.last_gen = glow.gen;
    let mut lane = vec![0u32; res.len as usize];
    for &(idx, packed) in &glow.entries {
        if let Some(slot) = lane.get_mut(idx as usize) {
            *slot = packed;
        }
    }
    queue.write_buffer(&res.buffer, 0, bytemuck::cast_slice(&lane));
}

/// Main-world one-shot: the last world point a non-switch left click landed on (pick.rs writes it).
/// A generation counter makes it a one-shot across the world boundary without a clear-back channel.
#[derive(Resource, Clone, Default)]
pub struct DoorClick {
    pub point: Option<Vec3>,
    pub gen: u64,
}
impl ExtractResource for DoorClick {
    type Source = DoorClick;
    fn extract_resource(s: &Self) -> Self {
        s.clone()
    }
}

/// Rotate one door part by `rad` about `axis` THROUGH THE HINGE `pivot`. The panel's own origin
/// is the hinge, but a part (glass, inlay) can sit anywhere on the leaf, so the translation is
/// carried around the pivot too — rotating each part about its own origin would tear the door
/// apart. Returns the mutated 80-byte record.
fn door_record(base: &InstanceGpuRecord, pivot: Vec3, axis: Vec3, rad: f32) -> InstanceGpuRecord {
    let r = Mat3::from_axis_angle(axis, rad);
    let base_lin = Mat3::from_cols(
        Vec3::new(base.m0[0], base.m1[0], base.m2[0]),
        Vec3::new(base.m0[1], base.m1[1], base.m2[1]),
        Vec3::new(base.m0[2], base.m1[2], base.m2[2]),
    );
    let l = r * base_lin;
    let t = Vec3::new(base.m0[3], base.m1[3], base.m2[3]);
    let t2 = pivot + r * (t - pivot);
    let mut rec = *base;
    rec.m0[0] = l.x_axis.x; rec.m0[1] = l.y_axis.x; rec.m0[2] = l.z_axis.x; rec.m0[3] = t2.x;
    rec.m1[0] = l.x_axis.y; rec.m1[1] = l.y_axis.y; rec.m1[2] = l.z_axis.y; rec.m1[3] = t2.y;
    rec.m2[0] = l.x_axis.z; rec.m2[1] = l.y_axis.z; rec.m2[2] = l.z_axis.z; rec.m2[3] = t2.z;
    // Carry the cull sphere with the part (same rigid motion) — its authored centre would
    // otherwise stay at the closed pose and cull a wide-open door at grazing angles. Rotation
    // preserves distances, so the radius is unchanged.
    let c = pivot + r * (Vec3::new(base.sphere[0], base.sphere[1], base.sphere[2]) - pivot);
    rec.sphere[0] = c.x;
    rec.sphere[1] = c.y;
    rec.sphere[2] = c.z;
    rec
}

/// Toggle the nearest door on a click + ease all in-flight doors, re-uploading changed instances.
fn animate_doors(
    time: Res<Time>,
    render_queue: Res<RenderQueue>,
    doors: Option<ResMut<EftDoors>>,
    click: Option<Res<DoorClick>>,
    mut last_gen: Local<u64>,
    mut nonce: ResMut<EftDynamicNonce>,
) {
    let Some(mut res) = doors else { return };
    // First frame for this map's doors: write every door's initial-pose record so already-open
    // doors (or EFT_DOORS_OPEN) show open from the start (the buffer was built with closed bases).
    let force_all = res.is_added();
    // ---- process a fresh click: toggle the nearest openable door ----
    if let Some(c) = click.as_ref() {
        if c.gen != *last_gen {
            *last_gen = c.gen;
            if let Some(p) = c.point {
                let mut best: Option<(usize, f32)> = None;
                for (i, d) in res.doors.iter().enumerate() {
                    let dist = d.pivot.distance_squared(p);
                    if dist < 6.25 && best.map(|(_, b)| dist < b).unwrap_or(true) {
                        best = Some((i, dist));
                    }
                }
                if let Some((i, _)) = best {
                    let d = &mut res.doors[i];
                    if !d.locked {
                        d.target = if d.target > 0.5 { 0.0 } else { 1.0 };
                    }
                }
            }
        }
    }
    // ---- ease in-flight doors + upload the ones that moved ----
    let step = (time.delta_secs() / 0.35).clamp(0.0, 1.0); // ~0.35 s open/close
    // Split the borrow: read buffer handle, then mutate doors.
    let buf = res.instances_buf.clone();
    let mut wrote_any = false;
    for d in res.doors.iter_mut() {
        let moving = (d.progress - d.target).abs() >= 1.0e-4;
        if !moving && !force_all {
            continue;
        }
        wrote_any = true;
        if moving {
            d.progress += (d.target - d.progress).signum() * step;
            d.progress = d.progress.clamp(0.0, 1.0);
        }
        // ease (smoothstep) for a nicer swing — every part of the leaf moves together
        let e = d.progress * d.progress * (3.0 - 2.0 * d.progress);
        for part in &d.parts {
            let rec = door_record(&part.closed, d.pivot, d.axis, d.open_rad * e);
            render_queue.write_buffer(&buf, part.gpu_idx as u64 * 80, bytemuck::bytes_of(&rec));
        }
    }
    // A moving door rewrote instance records — the cached shadow cascades are stale.
    if wrote_any {
        nonce.0 = nonce.0.wrapping_add(1);
    }
}

fn prepare_shadow_uniforms(
    render_queue: Res<RenderQueue>,
    config: Option<Res<EftShadowConfig>>,
    resources: Option<Res<EftShadowResources>>,
    settings: Option<Res<crate::render::GfxSettings>>,
    views: Query<&ExtractedView, With<CullCamera>>,
    mut cache: ResMut<EftShadowCache>,
    nonce: Res<EftDynamicNonce>,
    buffers: Option<Res<EftGpuBuffers>>,
    // Source of the bindless albedo count uploaded in `params.y` (the shadow fragment's
    // descriptor-index clamp). Same `views` vec that builds the material bind group.
    mat_res: Option<Res<EftMaterialResources>>,
    // gfx.w = phase for the grass sway (see the WavingGrass block in gpu_draw.wgsl's vertex stage).
    time: Res<bevy::time::Time>,
) {
    let (Some(config), Some(res)) = (config, resources) else {
        return;
    };
    // Cascade-cache force conditions beyond a view-proj change: door swings rewrote instance
    // records, the pack/geometry rebuilt, or any GfxSettings change (LOD sliders and cull
    // thresholds alter the indirect draw set the shadow pass replays).
    let force = nonce.is_changed()
        || buffers.as_ref().is_some_and(|b| b.is_changed())
        || settings.as_ref().is_some_and(|s| s.is_changed());
    let Some(view) = views.iter().next() else {
        return;
    };
    // Runtime UI scales (GfxSettings, extracted) ride a spare uniform lane; the shadow switch
    // itself was already synced into config.enabled by sync_gfx_shadow_toggle this frame.
    let shadows_on = config.enabled;
    // gfx.w = app time in seconds, the phase for the grass WavingGrass sway (gpu_draw.wgsl
    // vertex stage). Wrapped to a long period so f32 keeps sub-millisecond precision over a
    // multi-hour session instead of visibly quantising the animation.
    let t_wind = (time.elapsed_secs_wrapped() % 3600.0) as f32;
    let gfx = match settings.as_ref() {
        Some(s) => [s.fog, s.sky_refl, s.emissive, t_wind],
        None => [1.0, 1.0, 1.0, t_wind],
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

    // B2: grass shadow casters off by default (the 109k alpha-tested cross-quads dominated the
    // shadow pass for micro-shadows invisible at map scale). EFT_GRASS_SHADOWS=1 restores.
    let grass_casters = grass_shadows();

    let mut main = SunShadowUniform {
        // One far plane per cascade (SHADOW_SPLITS[1..=N]); overlap/enabled live in casc_params now.
        split_far: std::array::from_fn(|c| {
            SHADOW_SPLITS
                .get(c + 1)
                .copied()
                .unwrap_or(SHADOW_SPLITS[SHADOW_CASCADES])
        }),
        casc_params: [
            SHADOW_CASCADE_OVERLAP,
            if shadows_on { 1.0 } else { 0.0 },
            SHADOW_CASCADES as f32,
            // Volumetric shafts need the cascades to march, so they are forced off whenever shadows
            // are off — otherwise the march finds "lit" everywhere and paints a flat glow.
            match settings.as_ref() {
                Some(s) if shadows_on && s.volumetric => s.volumetric_strength.max(0.0),
                _ => 0.0,
            },
        ],
        sun_dir_texel: [lsun.x, lsun.y, lsun.z, 1.0 / shadow_map_size() as f32],
        combine: [
            SHADOW_DIFFUSE_CAP,
            SHADOW_FADE_START,
            SHADOW_FADE_END,
            if config.debug { 1.0 } else { 0.0 },
        ],
        gfx,
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

        // ROTATION-INVARIANT FIT, centred on the CAMERA rather than the slice.
        //
        // This used to centre on the slice centroid. The radius was rotation-invariant, but the
        // CENTRE was not: the frustum slice swings around the eye as you look around, so its
        // centroid orbits, every yaw/pitch produced a different `view_proj`, and the #5b cache
        // below missed on every frame of a pan — both cascades fully re-rendered. Measured on
        // interchange's mall at 2560x1440: the shadow pass costs 0.57 ms with the camera at rest
        // and 6.16 ms while panning, 11x more, ~36% of the frame. Static benchmarks could never
        // see it, which is why "sun shadows off" had only ever measured ~5%.
        //
        // The eye is fixed under rotation and |corner - eye| is preserved by rotation, so centring
        // here makes the whole fit depend on camera POSITION alone. Panning is then bit-identical
        // and free. The price is a larger square: the eye-centred bound must reach the slice's far
        // corners, so the radius grows ~1.3x per cascade (coarser texels). That is worth paying
        // because shadow-map resolution is nearly free on this path — 512^2 -> 4096^2 measured
        // +0.32 ms — so SHADOW_MAP_SIZE_DEFAULT absorbs it and comes out ahead of the old texel
        // density at a fraction of the old cost.
        let center = world_from_view.w_axis.truncate();
        let mut radius = 0.0f32;
        for cc in &corners {
            radius = radius.max((*cc - center).length());
        }
        radius = radius.max(0.05);

        // QUANTISE THE WHOLE FIT, not just the ortho offsets.
        //
        // Snapping the ortho centre alone did nothing (measured: 15.007 ms vs 14.976 unsnapped),
        // because `light_view` was built from the UNSNAPPED centre and the Z range was fitted to the
        // moving corners — so `view_proj` changed every frame no matter what the ortho offsets did.
        // Everything that feeds view_proj has to be quantised or constant:
        //   * the light BASIS depends only on lsun + up, so it is already constant;
        //   * the centre is snapped in that basis to a block of SHADOW_SNAP_TEXELS texels, and the
        //     radius is padded by one quantum so the slice can never escape the square between
        //     snaps;
        //   * the Z range is derived from `radius` and the extrude/margin constants instead of the
        //     corner set, which makes it invariant too (slightly more conservative depth, same
        //     precision in a 0..1 depth buffer).
        // Rotation is then free and translation only re-renders once per quantum crossed. The cost
        // of a coarse quantum is that the texel grid steps in blocks when it does move, so
        // shadow-edge aliasing changes in one frame rather than crawling — which is why the constant
        // stays modest.
        // QUANTISE THE RADIUS FIRST. `radius` is the max corner distance, which is invariant under
        // rotation in exact arithmetic but wobbles in the low bits of f32 as the view matrix changes.
        // `view_proj` is compared for EXACT equality, so that wobble alone re-rendered ~89% of frames
        // even after the centre was snapped. Rounding up to a fixed step makes the whole fit — texel
        // size, snap quantum and ortho extent — bit-reproducible while the camera stays inside a step.
        let radius = (radius / SHADOW_RADIUS_STEP).ceil() * SHADOW_RADIUS_STEP;
        let base_texel = (2.0 * radius) / shadow_map_size() as f32;
        let snap = base_texel * SHADOW_SNAP_TEXELS;
        let radius = radius + snap;
        let world_texel = (2.0 * radius) / shadow_map_size() as f32;

        // Constant light basis (rotation only, about the world origin).
        let basis = Mat4::look_at_rh(lsun, Vec3::ZERO, up);
        let c_ls = basis.transform_point3(center);
        // ALL THREE axes, not just the texel plane. Leaving z unquantised kept the target sliding
        // along the light's view direction every frame, so `light_view` — and therefore `view_proj` —
        // still changed on every frame of movement even though the ortho box was stable. Measured
        // exactly that: `vp changed 600/600` with `forced 1`. Sliding along z is harmless to cover
        // because ortho_near/far already span 2*radius + extrude + margin around the target.
        let snapped_ls = Vec3::new(
            (c_ls.x / snap).round() * snap,
            (c_ls.y / snap).round() * snap,
            (c_ls.z / snap).round() * snap,
        );
        let target = basis.inverse().transform_point3(snapped_ls);
        let eye = target + lsun * (radius + SHADOW_CASTER_EXTRUDE);
        let light_view = Mat4::look_at_rh(eye, target, up);

        // Z range from the geometry of the fit rather than the corners: every slice point lies
        // within `radius` of `target`, and the eye sits (radius + EXTRUDE) toward the sun, so the
        // near plane can be EXTRUDE and the far plane 2*radius + EXTRUDE + MARGIN.
        let ortho_near = SHADOW_CASTER_EXTRUDE.max(0.0);
        let ortho_far = ortho_near + 2.0 * radius + SHADOW_RECEIVER_MARGIN;

        let proj = Mat4::orthographic_rh(
            -radius,
            radius,
            -radius,
            radius,
            ortho_near,
            ortho_far,
        );
        let view_proj = proj * light_view;
        let vp_cols = view_proj.to_cols_array_2d();

        // #5b cascade cache: identical fit == identical content (static sun, static world,
        // unchanged camera cull) -> tell the node to skip this cascade's render pass. While
        // shadows are OFF the atlas content goes stale/undefined, so the cache is voided and
        // re-enabling re-renders both layers.
        if shadows_on {
            let vp_changed = cache.vp[c] != Some(vp_cols);
            let dirty = force || vp_changed;
            cache.render[c] = dirty;
            if dirty {
                cache.vp[c] = Some(vp_cols);
            }
            // EFT_SHADOW_DEBUG=1: report how often each cascade actually re-renders. Frame time
            // alone cannot separate "the cascade re-rendered" from "the main pass sampled more
            // shadow texels", and inferring one from the other is how a bad conclusion gets shipped.
            if shadow_debug() {
                use std::sync::atomic::{AtomicU64, Ordering};
                // `[const { .. }; N]` rather than a literal per cascade: AtomicU64 is not Copy, so a
                // static array cannot use the `[expr; N]` repeat form, and the hand-written literal
                // had to be edited for every cascade added.
                static FRAMES: [AtomicU64; SHADOW_CASCADES] =
                    [const { AtomicU64::new(0) }; SHADOW_CASCADES];
                static RENDERS: [AtomicU64; SHADOW_CASCADES] =
                    [const { AtomicU64::new(0) }; SHADOW_CASCADES];
                static FORCED: [AtomicU64; SHADOW_CASCADES] =
                    [const { AtomicU64::new(0) }; SHADOW_CASCADES];
                static VPDIFF: [AtomicU64; SHADOW_CASCADES] =
                    [const { AtomicU64::new(0) }; SHADOW_CASCADES];
                let f = FRAMES[c].fetch_add(1, Ordering::Relaxed) + 1;
                if dirty {
                    RENDERS[c].fetch_add(1, Ordering::Relaxed);
                }
                if force {
                    FORCED[c].fetch_add(1, Ordering::Relaxed);
                }
                if vp_changed {
                    VPDIFF[c].fetch_add(1, Ordering::Relaxed);
                }
                if f % 300 == 0 {
                    let r = RENDERS[c].load(Ordering::Relaxed);
                    eprintln!(
                        "  [shadow] cascade {c}: rendered {r}/{f} ({:.0}%) | forced {} | vp changed {}                          | radius {:.1} m snap {:.3} m",
                        100.0 * r as f32 / f as f32,
                        FORCED[c].load(Ordering::Relaxed),
                        VPDIFF[c].load(Ordering::Relaxed),
                        radius,
                        snap
                    );
                }
            }
        } else {
            cache.render[c] = false;
            cache.vp[c] = None;
        }

        main.view_proj[c] = vp_cols;
        main.texel_world[c] = world_texel;

        let cascade = ShadowCascadeUniform {
            view_proj: vp_cols,
            dir_texel: [lsun.x, lsun.y, lsun.z, 1.0 / shadow_map_size() as f32],
            params: [
                if grass_casters { 1.0 } else { 0.0 },
                mat_res.as_ref().map_or(0.0, |m| m.views.len() as f32),
                0.0,
                0.0,
            ],
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
        // Five specializations of the same shader/mesh. Surface overlays and SoftCutout roads need
        // stronger coplanar handling than glass, while SoftCutout uses a depth-only coverage pass
        // followed by a non-depth-writing color pass to avoid road-on-road z-fighting.
        let opaque_pipeline = pipelines.specialize(
            &pipeline_cache,
            &draw_pipeline,
            EftDrawKey {
                samples: msaa.samples(),
                hdr: view.hdr,
                pass: DrawPass::Opaque,
            },
        );
        let blend_pipeline = pipelines.specialize(
            &pipeline_cache,
            &draw_pipeline,
            EftDrawKey {
                samples: msaa.samples(),
                hdr: view.hdr,
                pass: DrawPass::Blend,
            },
        );
        let overlay_pipeline = pipelines.specialize(
            &pipeline_cache,
            &draw_pipeline,
            EftDrawKey {
                samples: msaa.samples(),
                hdr: view.hdr,
                pass: DrawPass::Overlay,
            },
        );
        let decal_depth_pipeline = pipelines.specialize(
            &pipeline_cache,
            &draw_pipeline,
            EftDrawKey {
                samples: msaa.samples(),
                hdr: view.hdr,
                pass: DrawPass::DecalDepth,
            },
        );
        let decal_color_pipeline = pipelines.specialize(
            &pipeline_cache,
            &draw_pipeline,
            EftDrawKey {
                samples: msaa.samples(),
                hdr: view.hdr,
                pass: DrawPass::DecalColor,
            },
        );

        let cam_pos = view.world_from_view.translation();
        for (entity, main_entity) in &markers {
            // Transparent3d sorts ASCENDING by distance (values increase toward the camera), so
            // the OPAQUE item at a large NEGATIVE distance runs FIRST and writes depth. Blend
            // meshes then draw as ONE ITEM EACH, depth-sorted back-to-front (farthest = most
            // negative = first), each issuing a single indirect record from indirect_blend —
            // this replaced the whole-scene P2 re-raster AND gave transparency a stable order
            // (Codex review). Mixed-class meshes draw in both passes; the fragment class-discard
            // splits them.
            phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline: opaque_pipeline,
                draw_function: draw_fn,
                distance: -1.0e30, // sort FIRST (writes depth)
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
            for (mesh_idx, centers, pass_mask) in &_buffers.blend_meshes {
                // Sort key = distance to this mesh's NEAREST instance. It used to be the distance
                // to its FIRST instance, which is an arbitrary copy that can sit anywhere on the
                // map, so a mesh sorted as if it were somewhere it is not: on interchange the
                // pane you look THROUGH (Nikitskaya_2_Outdoor_Glass_04, nearest instance 7.2 m)
                // carried a 32.2 m key while the pane BEHIND it (…Glass_02, 8.0 m) carried 16.6 m
                // — so the near glass composited behind the far glass. Worse, both keys track
                // far-away instances, so ordinary camera movement made them CROSS and the pair
                // swapped: windows seen through windows flashed between two shadings. Taking the
                // minimum is O(blend instances) per frame (6,235 on interchange — microseconds).
                // NOTE this orders MESHES correctly, not instances WITHIN one mesh: two panes of
                // the SAME mesh still share one indirect record and blend in arbitrary order.
                let d = centers
                    .iter()
                    .map(|c| (cam_pos - Vec3::from_array(*c)).length())
                    .fold(f32::INFINITY, f32::min);
                let d = if d.is_finite() { d } else { 0.0 };
                let item = |pipeline, distance| Transparent3d {
                    entity: (entity, *main_entity),
                    pipeline,
                    draw_function: draw_fn,
                    distance,
                    batch_range: 0..1,
                    extra_index: PhaseItemExtraIndex::IndirectParametersIndex {
                        range: *mesh_idx..(*mesh_idx + 1),
                        batch_set_index: None,
                    },
                    indexed: true,
                };
                if pass_mask & BLEND_MESH_SOFTCUTOUT != 0 {
                    // THE road-decal flicker fix. Two coplanar SoftCutout roads at the bus stop
                    // (Bus_stop_road_01 mat 776 + _02 mat 724) flickered on ANY camera rotate/zoom.
                    // Root cause was a two-part depth+order interaction, not a simple z-fight:
                    // the coverage-only depth PREPASS used to draw FIRST and wrote BOTH decals' depth,
                    // so each decal's COLOR was then GreaterEqual-tested against the OTHER decal's
                    // prepass depth. Rotating (even in place — view-space z changes) flipped that test
                    // per-pixel so a decal dropped in/out, and their `-d` distance sort also swapped
                    // which composited on top. No depth bias / NDC-push could fix it: the interaction
                    // was decal-vs-decal in the depth buffer.
                    //
                    // Fix mirrors Unity's fixed decal render-queue: composite the COLORS FIRST, tested
                    // ONLY against real opaque scene depth (the prepass has not run yet), in a stable
                    // camera-INDEPENDENT order — so decals never cull or reorder against each other,
                    // only against solid geometry. `mesh_idx` is the unique, deterministic, view-
                    // invariant build index -> a strict total order (base 2e6 keeps idx increments
                    // f32-distinct; a 1e28 base collapses all to one tie). The +1e-3*w clip push
                    // (gpu_draw.wgsl) still lifts the color over the coplanar OPAQUE ground.
                    phase.add(item(decal_color_pipeline, -2.0e6 - (*mesh_idx as f32)));
                    // Coverage-only depth prepass drawn AFTER the colors: re-asserts the road's raw
                    // depth so it still occludes the underground ceiling + POIs drawn later (0d95be1),
                    // but can no longer gate the decal colors above. No NDC push (a depth writer would
                    // peter-pan). Fixed key, less-negative than the colors (draws after them) and far
                    // more negative than the -d Overlay/Blend bands (draws before them).
                    phase.add(item(decal_depth_pipeline, -1.5e6));
                }
                if pass_mask & BLEND_MESH_OVERLAY != 0 {
                    phase.add(item(overlay_pipeline, -d - 0.001));
                }
                if pass_mask & BLEND_MESH_TRANSPARENT != 0 {
                    phase.add(item(blend_pipeline, -d));
                }
            }
        }
    }
}

/// Which GPU-driven draw specialization a pipeline is. Part of `EftDrawKey`'s
/// Hash/Eq so each caches as a SEPARATE pipeline.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum DrawPass {
    /// P1 OPAQUE: blend None, depth-write ON, no bias, A2C for cutout edges. Discards BLEND frags.
    Opaque,
    /// True transparency (currently glass): alpha blend, depth-write off, no coplanar bias.
    Blend,
    /// Plain decal and textured-water surface overlays: alpha blend, depth-write off, strong bias.
    Overlay,
    /// SoftCutout coverage-only depth prepass (A2C); keeps road occlusion without color fighting.
    DecalDepth,
    /// SoftCutout premultiplied color, blended after its depth prepass with a slightly larger bias.
    DecalColor,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct EftDrawKey {
    samples: u32,
    hdr: bool,
    pass: DrawPass,
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

        // --- pass-dependent state ----------------------------------------------------
        // Coplanar road/water decals are separated from the ground in CLIP space (the DECAL_NDC_PUSH
        // vertex offset in gpu_draw.wgsl), NOT with a rasterizer DepthBiasState. Under Bevy REVERSE-Z
        // (near=1.0, far=0.0, GreaterEqual) on a Depth32Float target, the rasterizer bias `constant`
        // is `constant * 2^(exponent(z) - 23)` (D3D spec) — it rides the fragment's depth EXPONENT,
        // which drifts as the camera zooms/rotates, so NO constant value is stable (8 -> 256 -> 512
        // all still flickered). A `clip.z += eps*clip.w` push is exactly +eps on z_ndc after the
        // perspective divide, exponent-INDEPENDENT, so the decal wins GreaterEqual at every
        // distance/angle. So these passes run with ZERO rasterizer bias; DecalDepth still writes
        // depth, so an open road over a void still occludes the underground (keeps 0d95be1).
        let (blend, depth_write_enabled, bias, frag_defs, write_mask): (
            Option<BlendState>,
            bool,
            DepthBiasState,
            Vec<bevy::shader::ShaderDefVal>,
            ColorWrites,
        ) = match key.pass {
            DrawPass::Opaque => (
                None,
                true,
                DepthBiasState::default(),
                vec![],
                ColorWrites::ALL,
            ),
            DrawPass::Blend => (
                // PREMULTIPLIED (src=One, dst=OneMinusSrcAlpha): the fragment premultiplies its
                // DIFFUSE by the transmission alpha but ADDS specular/reflection/emissive at full
                // strength — the Unity Standard transparent convention. Under the old straight
                // ALPHA_BLENDING every term was scaled by alpha, so clear glass showed only ~10-30%
                // of its sky reflection and read as a dark tinted slab (render-audit finding #18).
                Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                false,
                DepthBiasState::default(),
                vec!["BLEND_PASS".into()],
                ColorWrites::ALL,
            ),
            DrawPass::Overlay => (
                Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                false,
                DepthBiasState::default(),
                vec!["BLEND_PASS".into(), "OVERLAY_PASS".into(), "DECAL_NDC_PUSH".into()],
                ColorWrites::ALL,
            ),
            DrawPass::DecalDepth => (
                None,
                true,
                DepthBiasState::default(),
                // NO DECAL_NDC_PUSH: this prepass writes the road's RAW depth purely to occlude the
                // underground over voids; pushing a depth-WRITER would peter-pan. The COLOR passes
                // (DecalColor/Overlay) carry the push and clear this prepass + coplanar road decals.
                vec!["DECAL_DEPTH_PASS".into()],
                ColorWrites::empty(),
            ),
            DrawPass::DecalColor => (
                Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                false,
                DepthBiasState::default(),
                vec!["BLEND_PASS".into(), "DECAL_COLOR_PASS".into(), "DECAL_NDC_PUSH".into()],
                ColorWrites::ALL,
            ),
        };

        RenderPipelineDescriptor {
            label: Some(match key.pass {
                DrawPass::Opaque => "eft_gpu_draw_opaque".into(),
                DrawPass::Blend => "eft_gpu_draw_blend".into(),
                DrawPass::Overlay => "eft_gpu_draw_overlay".into(),
                DrawPass::DecalDepth => "eft_gpu_draw_decal_depth".into(),
                DrawPass::DecalColor => "eft_gpu_draw_decal_color".into(),
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
                        // Octahedral normal: 2 snorm16 in [-1,1], decoded in the vertex shader.
                        VertexAttribute {
                            format: VertexFormat::Snorm16x2,
                            offset: 12,
                            shader_location: 1,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 2,
                        },
                        // M3: per-vertex material index (read bit-exact as Uint32 @24).
                        VertexAttribute {
                            format: VertexFormat::Uint32,
                            offset: 24,
                            shader_location: 3,
                        },
                        // M3b2: per-vertex COLOR_0 vert-paint weight @36 (SoftCutout coverage
                        // rides on color.a). Interpolated (NOT flat) in the fragment shader.
                        // Unorm8x4: the pack's native format, expanded to vec4<f32> by the fetch.
                        VertexAttribute {
                            format: VertexFormat::Unorm8x4,
                            offset: 28,
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
                // Only opaque cutouts and the coverage-only road depth pass use A2C. Every color
                // overlay uses real alpha blending for a continuous, non-quantized edge.
                alpha_to_coverage_enabled: matches!(key.pass, DrawPass::Opaque | DrawPass::DecalDepth)
                    && key.samples > 1,
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
                    write_mask,
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
        let reset_groups = dispatch_2d(buffers.mesh_count.div_ceil(64));
        let cull_groups = dispatch_2d(buffers.instance_total.div_ceil(64));
        let blend_groups = dispatch_2d(buffers.blend_sort_groups.max(1));
        let diag = render_context.diagnostic_recorder();
        let encoder = render_context.command_encoder();
        let span = diag.time_span(encoder, "eft cull");

        // Separate passes â†’ wgpu inserts a barrier so cs_reset is fully visible to cs_cull.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eft_cull_reset"),
                timestamp_writes: None,
            });
            pass.set_pipeline(reset);
            pass.set_bind_group(0, &**bg, &[]);
            pass.dispatch_workgroups(reset_groups.0, reset_groups.1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eft_cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cull);
            pass.set_bind_group(0, &**bg, &[]);
            pass.dispatch_workgroups(cull_groups.0, cull_groups.1, 1);
        }
        // Third pass (its own barrier): order each BLEND mesh's survivors back-to-front. cs_cull
        // compacts with atomics, so without this the per-instance draw order inside a transparent
        // mesh reshuffles every frame and overlapping glass flickers with a STILL camera.
        if let Some(sort_blend) = cache.get_compute_pipeline(pipelines.sort_blend_id) {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eft_cull_sort_blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(sort_blend);
            pass.set_bind_group(0, &**bg, &[]);
            pass.dispatch_workgroups(blend_groups.0, blend_groups.1, 1);
        }
        span.end(render_context.command_encoder());
        Ok(())
    }
}

/// Split a workgroup count over X and Y so no dimension exceeds the adapter limit.
///
/// `max_compute_workgroups_per_dimension` is 65,535 on essentially every adapter — a hard Vulkan /
/// D3D limit rather than a soft wgpu default that a good card raises. At `@workgroup_size(64)` a
/// 1-D dispatch therefore covers at most 4,194,240 invocations, and a map with grass blows past
/// that on its own: woods ships 11,572,828 instances, needing 180,826 groups. Requesting them in X
/// is a validation error, and its failure mode is silent — wgpu invalidates the command encoder and
/// every subsequent pass reports "Encoder is invalid", so the symptom is 2 fps and a screen of
/// cascade errors that never name the dispatch. Nothing in the log pointed at the cause.
///
/// The Y rows are exact, not padded: `linear_index` in gpu_cull.wgsl reconstructs the index from
/// both dimensions using `num_workgroups`, and every entry point already bounds-checks its index,
/// so the tail invocations of the last row simply return.
fn dispatch_2d(groups: u32) -> (u32, u32) {
    // The floor is the guaranteed minimum from the WebGPU spec; every real adapter reports exactly
    // this, and using the constant keeps the split independent of which device we ended up on.
    const MAX_DIM: u32 = 65_535;
    if groups <= MAX_DIM {
        return (groups.max(1), 1);
    }
    let rows = groups.div_ceil(MAX_DIM);
    (MAX_DIM, rows)
}

// ===========================================================================
// #5 Shadow node: render the 2 cascade depth layers, reusing the camera-culled
// visible[]/indirect stream READ-ONLY. Runs after EftCull, before StartMainPass.
// ===========================================================================
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct EftShadowLabel;

// ===========================================================================
// Normal prepass: camera matrix + target management (PrepareResources) and the
// draw node (cull -> shadow -> PREPASS -> main). Shape cloned from the shadow
// pass — same shared buffers, different camera and targets.
// ===========================================================================
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct EftPrepassLabel;

/// Per-frame: write the camera clip_from_world into the prepass uniform and (re)create the
/// normal/depth targets when the view size changes. Sets `active` false whenever there is no
/// consumer (ssao off), no camera, or no size — the node and ssao both key off it, so the whole
/// feature degrades to the old derivative-normal path instead of half-running.
fn prepare_prepass(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    settings: Option<Res<crate::render::GfxSettings>>,
    mats: Option<Res<EftMaterialResources>>,
    views: Query<&ExtractedView, With<CullCamera>>,
    res: Option<ResMut<EftPrepassResources>>,
) {
    let Some(mut res) = res else { return };
    // Phase 1 (docs/GRAPHICS_PLAN.md): consumer MASK, not an SSAO boolean. The prepass runs when
    // ANY screen-space consumer wants it and is exactly absent otherwise (the acceptance criterion:
    // "no consumer -> prepass and pyramid GPU timestamps are exactly absent"). Consumer-driven, not
    // always-on: a Low-preset user with everything off pays zero.
    let want = settings
        .as_ref()
        .map(|s| s.ssao || s.ssr || s.taa || s.hiz || s.depth_prime || s.pcss)
        .unwrap_or(false);
    if !want {
        res.active = false;
        res.prev_clip_from_world = None; // toggling off invalidates history
        return;
    }
    let Ok(view) = views.single() else {
        res.active = false;
        res.prev_clip_from_world = None;
        return;
    };
    let vp = view.viewport; // (x, y, w, h)
    let size = UVec2::new(vp.z, vp.w);
    if size.x == 0 || size.y == 0 {
        res.active = false;
        return;
    }
    if res.size != size || res.normal_view.is_none() {
        res.prev_clip_from_world = None; // resize invalidates history
        let normal = render_device.create_texture(&TextureDescriptor {
            label: Some("eft_prepass_normal"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // consumers read single-sample; MSAA here would only cost
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = render_device.create_texture(&TextureDescriptor {
            label: Some("eft_prepass_depth"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        res.normal_view = Some(normal.create_view(&TextureViewDescriptor::default()));
        res.depth_view = Some(depth.create_view(&TextureViewDescriptor::default()));
        res.normal_texture = Some(normal);
        res.depth_texture = Some(depth);
        res.size = size;
        info!(
            "prepass: normal+depth targets {}x{} (Rgba16Float + Depth32Float, ~{} MiB)",
            size.x,
            size.y,
            (size.x as u64 * size.y as u64 * 12) >> 20
        );
    }
    // clip_from_world, exactly the transform the vertex needs (Bevy reverse-z projection included).
    let world_from_view = view.world_from_view.to_matrix();
    let clip_from_world = view.clip_from_view * world_from_view.inverse();
    // History shift: last frame's matrix becomes prev. Done AFTER the invalidation paths above so
    // a rebuilt frame never offers a stale matrix as history.
    if res.active {
        res.prev_clip_from_world = Some(res.clip_from_world);
    }
    res.clip_from_world = clip_from_world.to_cols_array_2d();
    let n_tex = mats.map(|m| m.views.len()).unwrap_or(0) as f32;
    render_queue.write_buffer(
        &res.uniform,
        0,
        bytemuck::bytes_of(&PrepassUniform {
            view_proj: clip_from_world.to_cols_array_2d(),
            params: [n_tex, 0.0, 0.0, 0.0],
        }),
    );
    res.active = true;
}

/// (Re)build the pyramid chain when its consumers are on and the prepass target resized. Bind
/// groups are rebuilt with the textures — group[0] copies prepass depth into mip 0 (its unused
/// src_mip slot binds mip 1 to avoid a same-subresource storage/sampled conflict), group[i>0]
/// reduces mip i-1 into mip i.
fn prepare_pyramid(
    render_device: Res<RenderDevice>,
    settings: Option<Res<crate::render::GfxSettings>>,
    prepass: Option<Res<EftPrepassResources>>,
    res: Option<ResMut<EftPyramidResources>>,
) {
    let (Some(mut res), Some(pre)) = (res, prepass) else { return };
    let want = settings
        .as_ref()
        .map(|s| s.ssr || s.hiz || s.depth_prime)
        .unwrap_or(false);
    if !want || !pre.active || pre.depth_view.is_none() {
        res.active = false;
        return;
    }
    let size = pre.size;
    if res.size != size || res.sample_view.is_none() {
        let mips = 32 - size.x.max(size.y).leading_zeros();
        let tex = render_device.create_texture(&TextureDescriptor {
            label: Some("eft_depth_pyramid"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mip_views: Vec<TextureView> = (0..mips)
            .map(|i| {
                tex.create_view(&TextureViewDescriptor {
                    label: Some("eft_depth_pyramid_mip"),
                    base_mip_level: i,
                    mip_level_count: Some(1),
                    ..default()
                })
            })
            .collect();
        let sample_view = tex.create_view(&TextureViewDescriptor::default());
        let depth_view = pre.depth_view.as_ref().unwrap();
        let mut bind_groups = Vec::with_capacity(mips as usize);
        for i in 0..mips as usize {
            let (src_mip, dst) = if i == 0 {
                // copy: src_mip unused; bind mip 1 (or mip 0 on a 1-mip chain, where reduce never runs)
                (&mip_views[1.min(mips as usize - 1)], &mip_views[0])
            } else {
                (&mip_views[i - 1], &mip_views[i])
            };
            bind_groups.push(render_device.create_bind_group(
                "eft_pyramid_bg",
                &res.layout,
                &BindGroupEntries::sequential((depth_view, src_mip, dst)),
            ));
        }
        info!("pyramid: {}x{} R32Float, {mips} mips", size.x, size.y);
        res.tex = Some(tex);
        res.mip_views = mip_views;
        res.sample_view = Some(sample_view);
        res.bind_groups = bind_groups;
        res.size = size;
        res.mips = mips;
    }
    res.active = true;
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct EftPyramidLabel;

struct EftPyramidNode;

impl FromWorld for EftPyramidNode {
    fn from_world(_: &mut World) -> Self {
        Self
    }
}

impl Node for EftPyramidNode {
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if world.get::<CullCamera>(graph.view_entity()).is_none() {
            return Ok(());
        }
        let Some(res) = world.get_resource::<EftPyramidResources>() else {
            return Ok(());
        };
        if !res.active || res.bind_groups.is_empty() {
            return Ok(());
        }
        let cache = world.resource::<PipelineCache>();
        let (Some(copy), Some(reduce)) = (
            cache.get_compute_pipeline(res.copy_pipeline),
            cache.get_compute_pipeline(res.reduce_pipeline),
        ) else {
            return Ok(());
        };
        let diag = render_context.diagnostic_recorder();
        let encoder = render_context.command_encoder();
        let span = diag.time_span(encoder, "eft pyramid");
        // One pass per mip: wgpu inserts the barrier that makes mip i-1 visible to mip i.
        for i in 0..res.mips as usize {
            let (w, h) = (
                (res.size.x >> i).max(1).div_ceil(8),
                (res.size.y >> i).max(1).div_ceil(8),
            );
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eft_pyramid_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(if i == 0 { copy } else { reduce });
            pass.set_bind_group(0, &res.bind_groups[i], &[]);
            pass.dispatch_workgroups(w, h, 1);
        }
        span.end(render_context.command_encoder());
        Ok(())
    }
}

struct EftPrepassNode;

impl FromWorld for EftPrepassNode {
    fn from_world(_: &mut World) -> Self {
        Self
    }
}

impl Node for EftPrepassNode {
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if world.get::<CullCamera>(graph.view_entity()).is_none() {
            return Ok(());
        }
        let (Some(res), Some(buffers), Some(draw_bg), Some(material_bg)) = (
            world.get_resource::<EftPrepassResources>(),
            world.get_resource::<EftGpuBuffers>(),
            world.get_resource::<EftDrawBindGroup>(),
            world.get_resource::<EftMaterialBindGroup>(),
        ) else {
            return Ok(());
        };
        if !res.active {
            return Ok(());
        }
        let (Some(normal_view), Some(depth_view)) = (&res.normal_view, &res.depth_view) else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(res.pipeline_id) else {
            return Ok(()); // still compiling
        };
        let diag = render_context.diagnostic_recorder();
        let span = diag.time_span(render_context.command_encoder(), "eft prepass");
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("eft_normal_prepass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: normal_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    // Clear to ZERO: a zero normal is the ssao shader's "no prepass data here"
                    // sentinel (sky, blend surfaces, the excluded grass), which falls back to the
                    // derivative reconstruction for that pixel.
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0), // reverse-z far, NOT the shadow pass's 1.0
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &draw_bg.0, &[]);
        pass.set_bind_group(1, &res.bind_group, &[]);
        pass.set_bind_group(2, &material_bg.0, &[]);
        pass.set_vertex_buffer(0, buffers.vertex.slice(..));
        pass.set_index_buffer(buffers.index.slice(..), 0, buffers.index_format);
        // Grass never enters the prepass (AO at blade scale is noise; the fragment bill is not) —
        // same two-range skip as the shadow pass, same stored range, same sea-quad caveat.
        match buffers.grass_mesh_range {
            Some((gs, ge)) if ge > gs && ge <= buffers.mesh_count => {
                if gs > 0 {
                    pass.multi_draw_indexed_indirect(&buffers.indirect, 0, gs);
                }
                if buffers.mesh_count > ge {
                    pass.multi_draw_indexed_indirect(
                        &buffers.indirect,
                        ge as u64 * DRAW_ARG_STRIDE,
                        buffers.mesh_count - ge,
                    );
                }
            }
            _ => pass.multi_draw_indexed_indirect(&buffers.indirect, 0, buffers.mesh_count),
        }
        drop(pass);
        span.end(render_context.command_encoder());
        Ok(())
    }
}

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

        let diag = render_context.diagnostic_recorder();
        let span = diag.time_span(render_context.command_encoder(), "eft shadow");
        // #5b cascade cache: prepare_shadow_uniforms marked which layers actually need
        // re-rendering this frame (camera at rest + static world = none; see EftShadowCache).
        let cache = world.get_resource::<EftShadowCache>();
        // One depth-only render pass per cascade layer: clear to 1.0, then the SAME multidraw the
        // main pass uses (indirect buffer READ-ONLY — never reset/reculled here).
        for c in 0..SHADOW_CASCADES {
            if let Some(cache) = cache {
                if !cache.render[c] {
                    continue; // atlas layer already holds this exact fit
                }
            }
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
            pass.set_index_buffer(buffers.index.slice(..), 0, buffers.index_format);
            // Skip the GRASS mesh range unless grass is meant to cast. The shadow vertex shader
            // already collapsed grass to a degenerate triangle, but that still costs one vertex
            // invocation per clump — 3.26 M of them on interchange — and measured the same whether
            // the quads were really rasterized or not. Not drawing the range at all is what removes
            // the work. Two ranges around the hole, because the SEA quad is appended AFTER grass and
            // must still cast.
            const IND_STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
            match buffers.grass_mesh_range.filter(|_| !grass_shadows()) {
                Some((gs, ge)) if ge > gs && ge <= buffers.mesh_count => {
                    if gs > 0 {
                        pass.multi_draw_indexed_indirect(&buffers.indirect, 0, gs);
                    }
                    if buffers.mesh_count > ge {
                        pass.multi_draw_indexed_indirect(
                            &buffers.indirect,
                            ge as u64 * IND_STRIDE,
                            buffers.mesh_count - ge,
                        );
                    }
                }
                _ => pass.multi_draw_indexed_indirect(&buffers.indirect, 0, buffers.mesh_count),
            }
        }
        span.end(render_context.command_encoder());
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
        item: &P,
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
        pass.set_index_buffer(buffers.index.slice(..), 0, buffers.index_format);

        // OPAQUE item (extra_index None): ONE multi-draw for ALL meshes from the opaque indirect
        // buffer (blend-only records are zeroed by cs_reset). BLEND items carry their mesh index
        // in IndirectParametersIndex and draw exactly ONE record from indirect_blend, already
        // depth-sorted by the phase. Requires MULTI_DRAW_INDIRECT (guarded at pipeline init).
        match item.extra_index() {
            PhaseItemExtraIndex::IndirectParametersIndex { range, .. } => {
                pass.multi_draw_indexed_indirect(
                    &buffers.indirect_blend,
                    range.start as u64 * DRAW_ARG_STRIDE,
                    1,
                );
            }
            _ => {
                pass.multi_draw_indexed_indirect(&buffers.indirect, 0, buffers.mesh_count);
            }
        }
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod material_stride_tests {
    use super::*;

    #[test]
    fn upload_plan_omits_disabled_grass_before_sizing_the_ssbo() {
        let plan = gpu_upload_plan(
            64,
            64,
            1_100,
            1_000,
            1_000_000,
            1_000_000,
            false,
        )
        .unwrap();
        assert!(plan.omit_grass);
        assert_eq!(plan.instance_count, 100);
        assert_eq!(plan.instance_bytes, 100 * 80);
    }

    #[test]
    fn upload_plan_uses_grass_as_a_binding_limit_fallback() {
        // Full = 88,000 B and does not fit. Base geometry = 8,000 B and does.
        let plan =
            gpu_upload_plan(64, 64, 1_100, 1_000, 1_000_000, 16_000, true).unwrap();
        assert!(plan.omit_grass);
        assert_eq!(plan.instance_count, 100);
    }

    #[test]
    fn upload_plan_rejects_unsplittable_or_still_oversized_buffers() {
        assert!(gpu_upload_plan(65, 1, 1, 0, 64, 64, true)
            .unwrap_err()
            .contains("vertex buffer"));
        assert!(gpu_upload_plan(1, 1, 1_000, 100, 1_000_000, 16_000, true)
            .unwrap_err()
            .contains("even without optional grass"));
    }

    #[test]
    fn grass_compaction_zeroes_its_draw_and_shifts_a_later_sea_instance() {
        let instances = vec![InstanceGpuRecord::default(); 6];
        let meta = vec![
            MeshMeta {
                instance_base: 0,
                instance_count: 2,
                ..default()
            },
            MeshMeta {
                instance_base: 2,
                instance_count: 3,
                ..default()
            },
            MeshMeta {
                instance_base: 5,
                instance_count: 1,
                ..default()
            },
        ];
        let (instances, meta) = compact_without_grass(&instances, &meta, 2, 3);
        assert_eq!(instances.len(), 3);
        assert_eq!((meta[0].instance_base, meta[0].instance_count), (0, 2));
        assert_eq!((meta[1].instance_base, meta[1].instance_count), (2, 0));
        assert_eq!((meta[2].instance_base, meta[2].instance_count), (2, 1));
    }

    /// No dispatch may exceed `max_compute_workgroups_per_dimension` (65,535 on every real
    /// adapter). Exceeding it is a validation error whose only symptom is a cascade of
    /// "Encoder is invalid" — woods rendered at 2 fps this way, needing 180,826 groups for its
    /// 11,572,828 instances. The split must also COVER the work: rows * MAX_DIM * 64 >= instances.
    #[test]
    fn dispatch_2d_never_exceeds_a_dimension_and_covers_the_work() {
        const MAX_DIM: u32 = 65_535;
        // 11_572_828 = woods with grass; the rest bracket the boundary and the degenerate ends.
        for &instances in &[0u32, 1, 64, 65_535, 4_194_240, 4_194_241, 11_572_828, u32::MAX / 2] {
            let groups = instances.div_ceil(64);
            let (x, y) = dispatch_2d(groups);
            assert!(x <= MAX_DIM, "{instances}: x={x} exceeds {MAX_DIM}");
            assert!(y <= MAX_DIM, "{instances}: y={y} exceeds {MAX_DIM}");
            assert!(x >= 1 && y >= 1, "{instances}: degenerate dispatch {x}x{y}");
            let covered = (x as u64) * (y as u64) * 64;
            assert!(
                covered >= instances as u64,
                "{instances}: dispatch {x}x{y} covers only {covered} invocations"
            );
        }
    }

    /// The shader must reconstruct the index the same way the host lays the grid out. Mirrors
    /// `linear_index` in gpu_cull.wgsl: every instance is hit exactly once, in order.
    #[test]
    fn linear_index_matches_the_shader_for_a_multi_row_dispatch() {
        let instances: u32 = 11_572_828;
        let (x, y) = dispatch_2d(instances.div_ceil(64));
        assert!(y > 1, "this case must actually span rows, got {x}x{y}");
        let stride = (x as u64) * 64; // == ng.x * 64u in the shader
        // First, last and a row boundary — enough to catch an off-by-one in the stride.
        assert_eq!(0u64 * stride + 0, 0);
        assert_eq!(1u64 * stride, stride);
        let last = (y as u64 - 1) * stride + (stride - 1);
        assert!(last >= instances as u64 - 1, "tail {last} misses instance {}", instances - 1);
    }

    use super::GpuMaterial;

    /// Byte size of one WGSL `MaterialGpu` declaration, computed from the shader source with
    /// std430 rules (scalar 4/4, vec2 8/8, vec3 12/16, vec4 16/16; struct rounded to its
    /// strictest member alignment). Only the field forms these shaders actually use.
    fn wgsl_material_size(src: &str) -> usize {
        let body = src
            .split_once("struct MaterialGpu {")
            .expect("no `struct MaterialGpu` in shader")
            .1
            .split_once("};")
            .expect("unterminated struct MaterialGpu")
            .0;
        let (mut size, mut struct_align) = (0usize, 4usize);
        for line in body.lines() {
            // strip comments, then take the `name: type,` pair
            let line = line.split("//").next().unwrap_or("").trim();
            let Some((_, ty)) = line.split_once(':') else { continue };
            let ty = ty.trim().trim_end_matches(',').trim();
            let (fsize, falign) = match ty {
                "u32" | "i32" | "f32" => (4, 4),
                t if t.starts_with("vec2<") => (8, 8),
                t if t.starts_with("vec3<") => (12, 16),
                t if t.starts_with("vec4<") => (16, 16),
                t if t.starts_with("mat4x4<") => (64, 16),
                other => panic!("material_stride_tests: unhandled WGSL type `{other}`"),
            };
            size = size.div_ceil(falign) * falign + fsize; // pad up to this field's alignment
            struct_align = struct_align.max(falign);
        }
        size.div_ceil(struct_align) * struct_align
    }

    /// REGRESSION (field device-loss, RX 7800 XT + RX 6800): the material table is ONE buffer read
    /// by BOTH gpu_draw.wgsl and gpu_shadow.wgsl. When parallax mapping grew the record 176 -> 192,
    /// the Rust POD and gpu_draw.wgsl were updated but gpu_shadow.wgsl was left at 176 — so the
    /// shadow pass indexed a 192-byte table with a 176-byte stride and every material after the
    /// first decoded garbage. The misread `albedo_index` lane became an out-of-range bindless
    /// descriptor index, which NVIDIA answers with zeros and AMD answers with a DEVICE FAULT:
    /// "Parent device is lost" a random 1..1980 frames into any map. Nothing in the build caught
    /// it — the Rust-side size assert only pins the Rust side. This pins all three together.
    #[test]
    fn wgsl_material_structs_match_the_rust_pod() {
        let shaders = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/shaders");
        let rust = std::mem::size_of::<GpuMaterial>();
        // gpu_prepass.wgsl added by the Phase-0 audit hardening: it declares the same 192-byte
        // record and shipped UNPINNED for a night — exactly the stride-mismatch shape that faulted
        // two Radeons. Every shader that declares MaterialGpu belongs in this list, no exceptions.
        for name in ["gpu_draw.wgsl", "gpu_shadow.wgsl", "gpu_prepass.wgsl"] {
            let src = std::fs::read_to_string(shaders.join(name))
                .unwrap_or_else(|e| panic!("cannot read {name}: {e}"));
            assert_eq!(
                wgsl_material_size(&src),
                rust,
                "{name}: WGSL `MaterialGpu` is {} B but the Rust `GpuMaterial` is {rust} B. \
                 These index the SAME storage buffer — a stride mismatch makes every material \
                 index > 0 decode garbage and turns the bindless texture index into an \
                 out-of-range descriptor access (device fault on AMD). Add/remove the padding \
                 block in {name} to match.",
                wgsl_material_size(&src),
            );
        }
    }
}
