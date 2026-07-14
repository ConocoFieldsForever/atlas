//! Render subsystem for the native EFT viewer.
//!
//! Two layers:
//!   * `instancing` — the WORKING M0 custom instanced draw (first-pixel). One
//!     entity + one instanced draw per unique eftpack mesh; the full 3x4 affine
//!     (incl shear/mirror) is applied in the vertex shader, cofactor normals +
//!     double-sided keep mirrors correct with zero baking.
//!   * `gpu_driven` — the M1 design (compute frustum/occlusion cull → compaction
//!     → indirect multidraw + screen-height LOD). Rust-side POD layouts + frustum
//!     math; the WGSL lives in `assets/shaders/cull.wgsl` (single source).
//!
//! Design center (locked): low-overhead GPU-driven instancing, NOT meshlets —
//! the data is already instanced low-poly (p50 ~384 tris, ~10.5k unique meshes
//! stored once).

pub mod gpu_driven;
pub mod instancing;

pub use instancing::{EftInstancingPlugin, LoadedPack};
