//! M1 GPU-driven design: the buffer/uniform POD structs + frustum math for the
//! compute-cull → indirect-multidraw path that will supersede the M0 per-mesh
//! instanced draw in `instancing.rs`.
//!
//! WHY (locked design): best performance for this already-instanced low-poly data
//! (p50 ~384 tris, ~10.5k unique meshes stored once) is LOW-OVERHEAD GPU-DRIVEN
//! INSTANCING, NOT virtual geometry. A compute pass frustum(+occlusion)-culls each
//! instance's shear-correct world bounding sphere, applies screen-height LOD, and
//! COMPACTS survivors + builds `DrawIndexedIndirectArgs`, so the CPU issues a
//! handful of `multi_draw_indexed_indirect` calls regardless of instance count.
//!
//! The actual cull/draw WGSL is a SINGLE SOURCE OF TRUTH in
//! `assets/shaders/cull.wgsl` (+ `instanced.wgsl`) — deliberately NOT duplicated
//! as embedded strings here (that drift was flagged in review). This module owns
//! only the Rust-side layouts those shaders bind and the CPU frustum extraction.
//!
//! Wiring this needs a RenderApp compute node + storage-buffer bind groups; those
//! sites are marked `TODO(gpu-wiring)` in `GpuDrivenPlan`.
#![allow(dead_code)] // M1 design layouts — referenced once the compute-cull node lands.

use bevy::prelude::*;

pub use crate::eftpack::{BoundingSphere, GpuInstance};

// ---------------------------------------------------------------------------
// Indirect draw args — one per (mesh, submesh/material) batch. Matches the
// wgpu/DX/VK `DrawIndexedIndirect` layout for `multi_draw_indexed_indirect`.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    /// submesh idxCount.
    pub index_count: u32,
    /// number of VISIBLE instances for this batch (written by the cull pass).
    pub instance_count: u32,
    /// global first index into the shared index buffer.
    pub first_index: u32,
    /// global base vertex into the shared vertex buffer.
    pub base_vertex: i32,
    /// offset into the compacted visible-instance index buffer for this batch.
    pub first_instance: u32,
}

// ---------------------------------------------------------------------------
// Per-frame cull uniform.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullGlobals {
    pub view_proj: [[f32; 4]; 4],
    /// 6 normalized frustum planes (xyz=normal, w=distance): L,R,B,T,N,F.
    pub frustum_planes: [[f32; 4]; 6],
    /// world camera position (xyz) + lodBias (w).
    pub camera_pos_lodbias: [f32; 4],
    /// 1 / (2 * tan(fovV/2)) for screen-height LOD (x); clamp-coarsest flag
    /// (y != 0 → hold coarsest LOD at distance); (zw) pad.
    pub lod_params: [f32; 4],
    pub instance_count: u32,
    pub _pad: [u32; 3],
}

/// Screen-height LOD group table entry (ported from `_lod.js`, one per lodGroup).
/// On the M0 pack (LOD-shell-deduped to one instance per group) this collapses to
/// a distance predicate; the full math is kept for future multi-LOD packs.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LodGroup {
    /// world-space group center (conjugated G3·center when !GID) + size (w).
    pub center_size: [f32; 4],
    /// screen-relative-height thresholds (descending). Up to 4 LOD levels.
    pub srh: [f32; 4],
    /// number of renderable LOD levels (srh.len - lastIsBillboard).
    pub rcnt: u32,
    pub _pad: [u32; 3],
}

// ---------------------------------------------------------------------------
// Frustum plane extraction (Gribb–Hartmann). Planes point INWARD; a sphere is
// visible when dot(plane.xyz, center) + plane.w >= -radius for all six.
// wgpu/Bevy clip space has z in [0,1], so near uses r2 (no r3+ combination).
// ---------------------------------------------------------------------------
pub fn build_frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let r0 = view_proj.row(0);
    let r1 = view_proj.row(1);
    let r2 = view_proj.row(2);
    let r3 = view_proj.row(3);

    let planes = [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // near   (z in [0,1])
        r3 - r2, // far
    ];
    let mut out = [Vec4::ZERO; 6];
    for (i, p) in planes.into_iter().enumerate() {
        let n = Vec3::new(p.x, p.y, p.z).length();
        out[i] = if n > 0.0 { p / n } else { p };
    }
    out
}

// ---------------------------------------------------------------------------
// The GPU-driven plan resource (M1). Holds handles to the buffers created in the
// render app once the compute node lands.
// ---------------------------------------------------------------------------
#[derive(Resource, Default)]
pub struct GpuDrivenPlan {
    pub ready: bool,
    // TODO(gpu-wiring): store bevy::render::render_resource::Buffer handles once
    // the RenderApp compute Node + bind groups are added:
    //   vertex/index/instance/bounding_spheres/lod_groups (storage, read)
    //   visible_indices/draw_args/draw_count (storage, read_write + INDIRECT)
    //   cull_globals (uniform); material_table + BC7/BC5 binding arrays (bindless)
    // The compute shader is assets/shaders/cull.wgsl; the draw is instanced.wgsl.
    // Needs: (1) a prefix-sum pass so each batch's first_instance points at its
    // slice of visible[]; (2) a mesh_id→batch_base map; (3) HZB occlusion.
}

/// Sizes other systems need without re-deriving them.
pub const INSTANCE_GPU_STRIDE: u64 = std::mem::size_of::<GpuInstance>() as u64; // 80
pub const DRAW_ARG_STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64; // 20
