//! Render subsystem for the native EFT viewer.
//!
//! Two layers:
//!   * `instancing` â€” the WORKING M0 custom instanced draw (first-pixel). One
//!     entity + one instanced draw per unique eftpack mesh; the full 3x4 affine
//!     (incl shear/mirror) is applied in the vertex shader, cofactor normals +
//!     double-sided keep mirrors correct with zero baking.
//!   * `gpu_driven` â€” the M2 GPU-driven path: GPU-resident buffers built once,
//!     compute frustum cull â†’ per-mesh contiguous compaction â†’ per-mesh indirect
//!     draw. Rust-side POD layouts + frustum math + the full plugin; the WGSL lives
//!     in `assets/shaders/gpu_cull.wgsl` (cull) + `gpu_draw.wgsl` (draw).
//!
//! Design center (locked): low-overhead GPU-driven instancing, NOT meshlets â€”
//! the data is already instanced low-poly (p50 ~384 tris, ~10.5k unique meshes
//! stored once).

pub mod gpu_driven;
pub mod instancing;
pub mod standard;

pub use gpu_driven::{CullCamera, EftGpuDrivenPlugin};
pub use instancing::{EftInstancingPlugin, LoadedPack};
pub use standard::EftStandardPlugin;

/// A/B render-path selector. `EFT_RENDER=m0` picks the working M0 custom instanced
/// path (`instancing.rs`, zero culling); anything else (default) picks the M2
/// GPU-driven compute-cull + indirect-draw path (`gpu_driven.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPath {
    /// M0: one instanced draw per unique mesh, no culling (A/B baseline).
    M0Instanced,
    /// M2: GPU-resident buffers + compute frustum cull + indirect multidraw (default).
    GpuDriven,
    /// Bevy STANDARD PBR mesh path (Mesh3d + StandardMaterial per instance×submesh).
    /// Slower, but unlocks Bevy's full lighting stack (shadows/SSAO/SSR/Solari RTX).
    Standard,
}

impl RenderPath {
    /// Resolve from the `EFT_RENDER` env var (`m0` | `gpu`) or an optional CLI token
    /// (e.g. the 2nd argv). Defaults to GPU-driven.
    pub fn from_env_or(cli: Option<&str>) -> Self {
        let pick = std::env::var("EFT_RENDER")
            .ok()
            .or_else(|| cli.map(str::to_string));
        match pick.as_deref().map(str::trim).map(str::to_ascii_lowercase) {
            Some(ref s) if s == "m0" || s == "instanced" => RenderPath::M0Instanced,
            Some(ref s) if s == "std" || s == "standard" || s == "pbr" => RenderPath::Standard,
            _ => RenderPath::GpuDriven,
        }
    }
}
