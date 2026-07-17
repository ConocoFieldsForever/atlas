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

use bevy::prelude::Resource;

pub mod gpu_driven;
pub mod grade;
pub mod instancing;
pub mod ssao;
pub mod standard;

pub use gpu_driven::{CullCamera, EftGpuDrivenPlugin};
pub use grade::{load_grade_lut, GradeLutCpu, GradePlugin};
pub use instancing::{EftInstancingPlugin, LoadedPack};
pub use ssao::SsaoPlugin;
pub use standard::EftStandardPlugin;

/// Runtime graphics settings, driven by the UI's "Graphics (experimental)" section and extracted
/// into the render world every frame. Every default reproduces the shipped look EXACTLY (scales
/// at 1.0, toggles matching their env-var startup defaults) so the panel is opt-in tweaking, not
/// a second source of truth. Scales ride spare uniform lanes (SunShadowUniform.gfx) — a slider
/// change is visible the same frame with no rebuild.
#[derive(Resource, Clone, PartialEq, bevy::render::extract_resource::ExtractResource)]
pub struct GfxSettings {
    /// Distance-fog density scale (0 = fog off, 1 = shipped look, 2 = pea soup).
    pub fog: f32,
    /// Analytic sky-reflection gain scale on glossy surfaces (0 = SH-probe only).
    pub sky_refl: f32,
    /// Emissive strength scale (monitors / signs / lamps).
    pub emissive: f32,
    /// Experimental real-time sun shadows (needs a valid sun_dir; marginal on baked-SH scenes).
    pub shadows: bool,
    /// Whether the pack has a usable sun_dir at all (set at startup; greys the toggle out).
    pub shadows_available: bool,
    /// Bloom on/off + intensity (Bevy camera component; applied in the main world).
    pub bloom: bool,
    pub bloom_intensity: f32,
    /// The game grade LUT (off = TonyMcMapface + hand-grade fallback).
    pub grade: bool,
    pub grade_available: bool,
    /// Pre-LUT exposure (web viewer default 0.18).
    pub grade_exposure: f32,
    /// PRISM vignette on/off.
    pub vignette: bool,
    /// Grass rendering (off = all clumps screen-size-culled).
    pub grass: bool,
    /// Screen-size cull thresholds in pixels (general, grass). 0 disables that cull.
    pub cull_px: f32,
    pub cull_px_grass: f32,
    /// Depth-only SSAO post pass (experimental; off = shipped look).
    pub ssao: bool,
    pub ssao_intensity: f32,
    /// SSAO sampling radius in meters.
    pub ssao_radius: f32,
    /// EFT-style unsharp-mask strength in the grade pass (0 = off; the game ships ~0.5).
    pub sharpen: f32,
}

impl Default for GfxSettings {
    fn default() -> Self {
        let (cull_px, cull_px_grass) = std::env::var("EFT_CULL_PX")
            .ok()
            .and_then(|s| {
                let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (v.len() == 2).then(|| (v[0], v[1]))
            })
            .unwrap_or((1.5, 4.0));
        Self {
            // EFT_FOG=0 for haze-free debug screenshots (A/B comparisons against game captures).
            fog: std::env::var("EFT_FOG")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1.0),
            sky_refl: 1.0,
            emissive: 1.0,
            shadows: std::env::var("EFT_SHADOWS").map(|v| v.trim() == "1").unwrap_or(false),
            shadows_available: false, // set at startup when sun_dir resolves
            bloom: true,
            bloom_intensity: 0.06,
            grade: true,             // no-op unless grade_available
            grade_available: false,  // set at startup when the LUT loads
            grade_exposure: std::env::var("EFT_GRADE_EXPOSURE")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1.7), // recalibrated: 0.18 was the web viewer's scale (and was tuned
                                 // while the LUT pass was silently dead — see render/grade.rs)
            vignette: !std::env::var("EFT_VIGNETTE").map(|v| v.trim() == "0").unwrap_or(false),
            grass: true,
            cull_px,
            cull_px_grass,
            ssao: std::env::var("EFT_SSAO").map(|v| v.trim() == "1").unwrap_or(false),
            ssao_intensity: 1.0,
            ssao_radius: 0.7,
            sharpen: std::env::var("EFT_SHARPEN")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0.0),
        }
    }
}

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
