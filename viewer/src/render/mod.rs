//! Render subsystem for Atlas (native GPU-driven EFT map viewer).
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

/// Pre-LUT exposure calibrated for the native renderer's SH radiance scale.  Keep this in one
/// place: both the startup LUT resource and the live graphics settings must begin at the same value.
/// 1.7 clipped too much of Lighthouse's pale road/rock range; 1.35 is roughly one third stop lower
/// while retaining the game's extracted LUT rather than replacing it with a hand grade.
pub const DEFAULT_GRADE_EXPOSURE: f32 = 1.35;

pub use gpu_driven::{CullCamera, EftGpuDrivenPlugin, GpuLoadSignal};
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
    /// Real-time sun shadows (default ON; needs a valid sun_dir). EFT_SHADOWS=0 force-disables.
    pub shadows: bool,
    /// Whether the pack has a usable sun_dir at all (set at startup; greys the toggle out).
    pub shadows_available: bool,
    /// Bloom on/off + intensity (Bevy camera component; applied in the main world).
    pub bloom: bool,
    pub bloom_intensity: f32,
    /// The game grade LUT (off = TonyMcMapface + hand-grade fallback).
    pub grade: bool,
    pub grade_available: bool,
    /// Pre-LUT exposure (native renderer default [`DEFAULT_GRADE_EXPOSURE`]).
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
            // Sun shadows default ON for every map with a sun_dir; EFT_SHADOWS=0 (or =false) opts OUT.
            shadows: std::env::var("EFT_SHADOWS")
                .map(|v| {
                    let t = v.trim();
                    t != "0" && !t.eq_ignore_ascii_case("false")
                })
                .unwrap_or(true),
            shadows_available: false, // set at startup when sun_dir resolves
            // EFT_BLOOM=0 disables (debug A/B: bloom's downsample grid can checker bright haze).
            bloom: !std::env::var("EFT_BLOOM").map(|v| v.trim() == "0").unwrap_or(false),
            bloom_intensity: 0.06,
            grade: true,             // no-op unless grade_available
            grade_available: false,  // set at startup when the LUT loads
            grade_exposure: std::env::var("EFT_GRADE_EXPOSURE")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(DEFAULT_GRADE_EXPOSURE),
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

/// Bumped by the in-place map loader (`main::load_map`) on every `.eftpack` swap. Extracted to the
/// render world so the epoch-aware GPU reset (`gpu_driven::reset_gpu_map_if_epoch_changed`) can tear
/// down the old map's buffers/bind-groups/pipelines and rebuild for the new pack. Also gates the
/// main-world per-map rebuild systems (`build_cpu_data`, `spawn_pois`, `spawn_loot`, camera reset,
/// teardown) via `run_if(resource_changed::<MapEpoch>)`. Starts at 0 (fires once on the first frame
/// so the initial map builds); each swap does `.0 += 1`.
#[derive(Resource, Clone, Copy, PartialEq, Eq, bevy::render::extract_resource::ExtractResource)]
pub struct MapEpoch(pub u64);

/// A/B render-path selector. `EFT_RENDER=m0` picks the working M0 custom instanced
/// path (`instancing.rs`, zero culling); anything else (default) picks the M2
/// GPU-driven compute-cull + indirect-draw path (`gpu_driven.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource)]
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
    /// Resolve from the `EFT_RENDER` env var (`m0` | `gpu` | `std`) or an optional CLI token
    /// (e.g. the 2nd argv). With NO override the path is chosen by GPU capability: the
    /// GPU-driven path if the adapter supports it (any modern AMD/NVIDIA discrete card via
    /// DX12/Vulkan), else the M0 instanced path — so an under-featured GPU renders honest
    /// geometry instead of the empty view the render-world feature guards would otherwise
    /// leave. An explicit `EFT_RENDER=gpu` still forces GPU-driven (skips the probe).
    pub fn from_env_or(cli: Option<&str>) -> Self {
        let pick = std::env::var("EFT_RENDER")
            .ok()
            .or_else(|| cli.map(str::to_string));
        match pick.as_deref().map(str::trim).map(str::to_ascii_lowercase) {
            Some(ref s) if s == "m0" || s == "instanced" => RenderPath::M0Instanced,
            Some(ref s) if s == "std" || s == "standard" || s == "pbr" => RenderPath::Standard,
            Some(ref s) if s == "gpu" || s == "gpu-driven" => RenderPath::GpuDriven,
            _ => {
                if gpu_driven_supported() {
                    RenderPath::GpuDriven
                } else {
                    eprintln!(
                        "render path: GPU lacks MULTI_DRAW_INDIRECT / bindless features - \
                         auto-selecting the M0 instanced path (override with EFT_RENDER=gpu)"
                    );
                    RenderPath::M0Instanced
                }
            }
        }
    }
}

/// Probe a throwaway wgpu adapter for the features the GPU-driven path hard-requires
/// (`init_gpu_pipelines` disables that path — empty view — without them). Uses the same
/// `HighPerformance` preference Bevy defaults to, so on a single-GPU AMD/NVIDIA box we inspect
/// the very adapter Bevy will pick. The instance/adapter are dropped immediately.
///
/// Finding 6: a probe ERROR now returns `false` (UNSUPPORTED -> M0). The M0 instanced path renders
/// honest geometry, so falling back on a probe hiccup is strictly safer than optimistically choosing
/// GPU-driven and risking the empty-view guard. If the probe SUCCEEDS but the real Bevy device still
/// lacks the features (hybrid-adapter mismatch), the render-world guard relaunches into M0 via
/// `GpuFallback` — so there is no reachable blank-view path either way.
/// Backends Atlas permits. DX12 PANICS at pipeline creation on Bevy's own `downsample_depth.wgsl`
/// (a scalar `push_constant`, wgpu#5683) — BEFORE any render path runs, so neither the GPU-driven
/// guard nor the M0 fallback can rescue it (both share the device). Atlas also hard-requires
/// Vulkan-class features regardless. So on Windows we restrict to Vulkan: a Vulkan-capable machine
/// runs, and a Vulkan-less one is caught by `main`'s pre-flight with an actionable message instead
/// of a confusing mid-pipeline wgpu panic. Non-Windows keeps wgpu's default (all) backends.
pub fn allowed_backends() -> wgpu::Backends {
    #[cfg(target_os = "windows")]
    {
        wgpu::Backends::VULKAN
    }
    #[cfg(not(target_os = "windows"))]
    {
        wgpu::Backends::all()
    }
}

/// True if wgpu finds ANY adapter within [`allowed_backends`]. `main` pre-flights this: a false here
/// means Bevy would otherwise panic deep in device init (no Vulkan adapter on a DX12-only machine),
/// so we exit early with a clear message instead.
pub fn has_usable_adapter() -> bool {
    let instance =
        wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: allowed_backends(), ..Default::default() });
    bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .is_ok()
}

fn gpu_driven_supported() -> bool {
    use bevy::render::settings::WgpuFeatures;
    let need = WgpuFeatures::MULTI_DRAW_INDIRECT
        | WgpuFeatures::INDIRECT_FIRST_INSTANCE
        | WgpuFeatures::TEXTURE_BINDING_ARRAY
        | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    // Probe within the SAME backends Bevy will use (allowed_backends), so on a multi-backend box we
    // inspect the very adapter Bevy picks — not a DX12 one it will never select.
    let instance =
        wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: allowed_backends(), ..Default::default() });
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }));
    match adapter {
        Ok(a) => {
            let ok = a.features().contains(need);
            let info = a.get_info();
            eprintln!(
                "gpu probe: {} ({:?}/{:?}) gpu-driven={}",
                info.name, info.device_type, info.backend, ok
            );
            ok
        }
        Err(e) => {
            eprintln!("gpu probe: adapter request failed ({e}) - treating GPU-driven as UNSUPPORTED, using M0");
            false
        }
    }
}
