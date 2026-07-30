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

pub mod fpv_cam;
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

pub use fpv_cam::FpvCamPlugin;
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
    /// Hard grass draw distance in METRES (0 = no clamp; `cull_px_grass` alone decides).
    /// A screen-size threshold's world horizon scales with viewport height and 1/tan(fov/2), so the
    /// same `cull_px_grass` draws grass 33% further at 1440p than at 1080p and further again when
    /// zoomed — and it lands at a different distance for each of woods' 15 grass kinds, because the
    /// distance is radius/k. This is the resolution- and kind-independent control. EFT_GRASS_DIST.
    pub grass_dist_m: f32,
    /// Depth-only SSAO post pass (experimental; off = shipped look).
    pub ssao: bool,
    pub ssao_intensity: f32,
    /// SSAO sampling radius in meters.
    pub ssao_radius: f32,
    /// EFT-style unsharp-mask strength in the grade pass (0 = off; the game ships ~0.5).
    pub sharpen: f32,
    /// FXAA in the grade pass. Default ON: every pipeline is single-sampled with alpha-to-coverage
    /// off, so without this there is no anti-aliasing anywhere, and alpha-cutout foliage against sky
    /// crawls badly in motion. EFT_AA=0 opts out (A/B against the game's own edges).
    pub aa: bool,
    /// FXAA blend strength, 0..1. 0.75 keeps foliage edges soft without smearing the grade's
    /// unsharp pass, which runs on the same tap set.
    pub aa_strength: f32,
    /// Realtime practical lights (lamps/spots from the light grid) master toggle. Only affects maps
    /// whose grid is populated (indirect-only bakes / EFT_LIGHTS-forced); a no-op on full bakes.
    pub lights: bool,
    /// Practical-light intensity MULTIPLIER on the baked-in scale (1 = shipped look).
    pub light_intensity: f32,
    /// Direct-sun diffuse MULTIPLIER on the B4-M base (EFT_SUN_DIFFUSE; the uniform slot is 0 on
    /// full bakes — where the baked SH already integrates the sun — so the slider is a no-op there).
    pub sun_diffuse: f32,
    /// Baked-GI (SH ambient) intensity MULTIPLIER (render-audit "gi_intensity": lifts/dims the
    /// whole indirect term without touching practicals or the sun).
    pub gi_intensity: f32,
    /// Volumetric sun shafts (god rays) — marches the shadow cascades through the fog medium.
    /// Default OFF: it is the most expensive photoreal extra here (a per-fragment march in a forward
    /// pass with no depth prepass, so overdraw multiplies it) and it changes the look, so it is opt-in
    /// rather than a silent change to every map. Requires shadows; forced off without them.
    pub volumetric: bool,
    /// Shaft brightness. 1.0 is a visible-but-plausible overcast shaft; the phase function already
    /// concentrates it toward the sun, so this does not need to be large.
    pub volumetric_strength: f32,
    /// Depth of field (bokeh) — photoreal extra, default off.
    pub dof: bool,
    pub dof_focal_m: f32,
    pub dof_fstop: f32,
    /// Chromatic aberration intensity (0 = off).
    pub chroma: f32,
    /// Power-switch state: bit i = light group i is POWERED (its switch flipped on). Default 0 =
    /// every group off (mall dark at raid spawn). Flipped by the Level-controls UI + clicking a
    /// switch mesh; `update_light_power` re-uploads the light buffer when it changes.
    pub light_groups: u32,
    /// Distance-based LOD on/off. Default ON (`EFT_LOD=0` opts out) — this comment used to claim
    /// "default off = shipped look", which was wrong and actively misleading once packs started
    /// shipping multiple shells: on an --alllod pack it is worth 3.83 ms of a 15.68 ms frame on
    /// woods (measured, 2560x1440), so a reader who believed it was off would mis-attribute that.
    /// Only meaningful on an all-LOD pack; a lean pack has one shell per group so it's a no-op.
    /// A live cull-uniform switch — no rebuild.
    pub lod_distance: bool,
    /// LOD bias (>1 holds finer shells to a greater distance; <1 switches to coarse sooner).
    pub lod_bias: f32,
    /// Debug: force a single LOD shell index (>=0); -1 = off (respect `lod_distance`).
    pub lod_force: i32,
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
            // Distance fog default (B4-S): the old 1.0 added a warm-grey haze that flattened mid/far
            // contrast and paled the sky — fog-off reads markedly more game-accurate (crisper, greener
            // distance). 0.4 keeps a hint of atmospheric depth without the flattening. EFT_FOG overrides
            // (0 = off for A/B captures against in-game shots).
            fog: std::env::var("EFT_FOG")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0.4),
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
            // Default 0 = OFF, so the shipped look is unchanged and this is purely opt-in. A default
            // horizon here would silently shorten grass on every existing map.
            grass_dist_m: std::env::var("EFT_GRASS_DIST")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0.0),
            ssao: std::env::var("EFT_SSAO").map(|v| v.trim() == "1").unwrap_or(false),
            ssao_intensity: 1.0,
            ssao_radius: 0.7,
            sharpen: std::env::var("EFT_SHARPEN")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0.0),
            aa: !std::env::var("EFT_AA").map(|v| v.trim() == "0").unwrap_or(false),
            aa_strength: std::env::var("EFT_AA_STRENGTH")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0.75),
            lights: true,
            light_intensity: 1.0,
            sun_diffuse: 1.0,
            gi_intensity: 1.0,
            volumetric: std::env::var("EFT_VOLUMETRIC")
                .map(|v| v.trim() == "1")
                .unwrap_or(false),
            volumetric_strength: std::env::var("EFT_VOLUMETRIC_STRENGTH")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1.0),
            dof: false,
            dof_focal_m: 15.0,
            dof_fstop: 2.8,
            chroma: 0.0,
            // All power groups OFF at spawn (mall dark until a switch is flipped). EFT_POWER=1
            // spawns fully powered (every group on) for screenshots / a lit walkthrough.
            light_groups: if std::env::var("EFT_POWER").map(|v| v.trim() == "1").unwrap_or(false) {
                u32::MAX
            } else {
                0
            },
            // Distance-LOD default ON: when a pack ships more than one shell per LODGroup the GPU
            // picks the cheapest one that still covers its screen size, which is the single biggest
            // lever on draw cost (and matters most when Atlas shares the GPU with the game). It is a
            // NO-OP on the lean LOD0-only packs the pipeline builds today -- those carry one shell
            // per group, so there is nothing to switch to -- but costs nothing to leave on and takes
            // effect the moment an `--alllod` pack is loaded. EFT_LOD=0 forces it off.
            lod_distance: std::env::var("EFT_LOD").map(|v| v.trim() != "0").unwrap_or(true),
            lod_bias: std::env::var("EFT_LOD_BIAS").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(1.0),
            lod_force: std::env::var("EFT_LOD_FORCE").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(-1),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// QUALITY PRESETS — every number below is MEASURED, not guessed.
// ---------------------------------------------------------------------------------------------
/// `tools/bench_gfx.py` runs one headless-ish bench per knob against a fixed Interchange camera at
/// two resolutions, recording frame time and VRAM (board delta, since Windows/WDDM reports no
/// per-process figure). Raw data: `docs/GFX_BENCH_1600x1000.json`, `docs/GFX_BENCH_2560x1440.json`.
///
/// Baseline reproduced within 0.35% across runs, so treat anything under ~1.5% as noise.
///
///   knob                     Δfps @1600x1000   Δfps @2560x1440   ΔVRAM
///   foliage off                   +13.1%            +16.8%          0
///   bloom off                      +7.8%             +6.1%          0
///   sun shadows off                +5.3%             +5.5%          0
///   aggressive prop cull           +2.3%             +0.9%          0
///   SSAO on (cost)                 -2.0%             -3.0%          0
///   textures Full (from Half)      +0.3%             -0.2%      +2177 MiB
///   textures Quarter (from Half)   +1.7%             +0.7%       -577 MiB
///   lights off / GI off / fog off / vignette off / parallax off / LOD bias / shadow-map size
///                                  all within noise at both resolutions
///
/// Two conclusions drive the presets:
///   1. VRAM is ENTIRELY a texture-quality story. Nothing else moves it by more than ~20 MiB.
///   2. Speed is foliage > bloom > shadows, and then nothing. The remaining toggles are visual
///      choices, and the UI must not advertise them as performance levers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QualityPreset {
    Low,
    Medium,
    High,
    Ultra,
    Custom,
}

impl QualityPreset {
    pub const ALL: [QualityPreset; 5] = [
        QualityPreset::Low,
        QualityPreset::Medium,
        QualityPreset::High,
        QualityPreset::Ultra,
        QualityPreset::Custom,
    ];

    /// Stable index for persistence (`qualityPreset` in atlas.config.json). The preset is chosen in
    /// the MAIN MENU because texture quality is applied when textures are uploaded — picking it
    /// after a map is already resident cannot change what was uploaded.
    pub fn index(self) -> u8 {
        match self {
            QualityPreset::Low => 0,
            QualityPreset::Medium => 1,
            QualityPreset::High => 2,
            QualityPreset::Ultra => 3,
            QualityPreset::Custom => 4,
        }
    }

    pub fn from_index(i: u8) -> QualityPreset {
        match i {
            0 => QualityPreset::Low,
            1 => QualityPreset::Medium,
            3 => QualityPreset::Ultra,
            4 => QualityPreset::Custom,
            _ => QualityPreset::High,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            QualityPreset::Low => "Low",
            QualityPreset::Medium => "Medium",
            QualityPreset::High => "High",
            QualityPreset::Ultra => "Ultra",
            QualityPreset::Custom => "Custom",
        }
    }

    /// Headline the user can act on: measured speed vs High, and measured VRAM.
    pub fn summary(self) -> &'static str {
        match self {
            QualityPreset::Low => {
                "~30% faster \u{2022} ~1.6 GB VRAM \u{2014} no foliage, shadows or bloom"
            }
            QualityPreset::Medium => {
                "~25% faster \u{2022} ~2.2 GB VRAM \u{2014} thinned foliage to 150 m, no shadows"
            }
            QualityPreset::High => "baseline \u{2022} ~2.3 GB VRAM \u{2014} the shipped look",
            // Ultra used to read "~2% slower". Volumetric shafts cost a measured +5.4 ms of a ~12 ms
            // frame, so that number would now be a lie by a factor of ~15.
            QualityPreset::Ultra => {
                "~30% slower \u{2022} ~4.5 GB VRAM \u{2014} full-res textures, SSAO + volumetric sun shafts"
            }
            QualityPreset::Custom => "your own mix \u{2014} see the per-option costs below",
        }
    }

    /// Texture quality this preset wants: 0 = Full, 1 = Half, 2 = Quarter. `None` for Custom
    /// (leave whatever the user set).
    pub fn tex_quality(self) -> Option<u8> {
        match self {
            QualityPreset::Low => Some(2),
            QualityPreset::Medium | QualityPreset::High => Some(1),
            QualityPreset::Ultra => Some(0),
            QualityPreset::Custom => None,
        }
    }

    /// Apply the preset's render-side choices in place. Texture quality is handled separately by
    /// the caller because it is a persisted config value that only takes effect on the next map
    /// load (mip levels are dropped at upload time).
    /// Every named preset must set EVERY field it means to control. `detect` applies a preset to a
    /// probe and compares, so a field left unset silently inherits the user's value and the preset
    /// stops being a description of what is running.
    ///
    /// Placement of the three newest options, from measurements on the woods flythrough at
    /// 2560x1440 (docs/GFX_BENCH_woods_*.json):
    ///   * FXAA (`aa`): +0.04 ms — inside the run-to-run noise floor, so it is ON everywhere,
    ///     including Low. Nothing else in this renderer anti-aliases (every pipeline is
    ///     sample_count 1), and turning it off buys a weak GPU almost nothing.
    ///   * Volumetric shafts: +5.40 ms, ~45% of the frame — ULTRA ONLY. It is the most expensive
    ///     option here by a wide margin, more than the whole distance-LOD win.
    ///   * Grass distance clamp: -3.24 ms at 150 m, -4.65 ms at 80 m. Left OFF for High/Ultra
    ///     (those are the "shipped look" presets and a horizon would visibly shorten grass), and
    ///     used by Medium/Low, which already thin foliage via `cull_px_grass`.
    pub fn apply(self, g: &mut GfxSettings) {
        let d = GfxSettings::default();
        match self {
            QualityPreset::Custom => {}
            QualityPreset::Ultra => {
                g.grass = true;
                g.shadows = true;
                g.bloom = true;
                g.ssao = true;
                g.lights = true;
                g.cull_px = d.cull_px;
                g.cull_px_grass = d.cull_px_grass;
                g.aa = true;
                g.volumetric = true;
                g.volumetric_strength = d.volumetric_strength;
                g.grass_dist_m = 0.0;
            }
            QualityPreset::High => {
                g.grass = true;
                g.shadows = true;
                g.bloom = true;
                g.ssao = false;
                g.lights = true;
                g.cull_px = d.cull_px;
                g.cull_px_grass = d.cull_px_grass;
                g.aa = true;
                g.volumetric = false;
                g.volumetric_strength = d.volumetric_strength;
                g.grass_dist_m = 0.0;
            }
            QualityPreset::Medium => {
                // Measured stack: shadows off + thinned foliage = +17% / +22%.
                g.grass = true;
                g.shadows = false;
                g.bloom = true;
                g.ssao = false;
                g.lights = true;
                g.cull_px = 2.0;
                g.cull_px_grass = 600.0;
                g.aa = true;
                // Shafts need the cascades; shadows are off here, so this would be forced off at the
                // uniform anyway. Set it explicitly so the preset says what it means.
                g.volumetric = false;
                g.volumetric_strength = d.volumetric_strength;
                g.grass_dist_m = 150.0;
            }
            QualityPreset::Low => {
                // Measured stack: +29% / +32%, and the texture drop takes VRAM to ~1.6 GB.
                g.grass = false;
                g.shadows = false;
                g.bloom = false;
                g.ssao = false;
                g.lights = false;
                g.cull_px = 4.0;
                g.cull_px_grass = 1000.0;
                g.aa = true;
                g.volumetric = false;
                g.volumetric_strength = d.volumetric_strength;
                // Grass is off entirely here, so this is belt-and-braces rather than a saving.
                g.grass_dist_m = 80.0;
            }
        }
    }

    /// Which preset (if any) the current settings correspond to. Returns `Custom` as soon as the
    /// user deviates, so the UI never claims a preset the scene isn't actually running.
    pub fn detect(g: &GfxSettings, tex_quality: u8) -> QualityPreset {
        for p in [
            QualityPreset::Ultra,
            QualityPreset::High,
            QualityPreset::Medium,
            QualityPreset::Low,
        ] {
            let mut probe = g.clone();
            p.apply(&mut probe);
            if probe == *g && p.tex_quality() == Some(tex_quality) {
                return p;
            }
        }
        QualityPreset::Custom
    }
}

#[cfg(test)]
mod quality_preset_tests {
    use super::{GfxSettings, QualityPreset};

    #[test]
    fn every_named_preset_round_trips_through_detection() {
        for preset in [
            QualityPreset::Low,
            QualityPreset::Medium,
            QualityPreset::High,
            QualityPreset::Ultra,
        ] {
            let mut settings = GfxSettings::default();
            preset.apply(&mut settings);
            assert_eq!(
                QualityPreset::detect(&settings, preset.tex_quality().unwrap()),
                preset,
                "{preset:?} must be the state the menu says it selected"
            );
        }
    }

    #[test]
    fn presets_apply_the_advertised_cost_drivers_and_texture_tiers() {
        let mut low = GfxSettings::default();
        QualityPreset::Low.apply(&mut low);
        assert_eq!(QualityPreset::Low.tex_quality(), Some(2));
        assert!(!low.grass && !low.shadows && !low.bloom && !low.ssao && !low.lights);
        assert_eq!((low.cull_px, low.cull_px_grass), (4.0, 1000.0));

        let mut medium = GfxSettings::default();
        QualityPreset::Medium.apply(&mut medium);
        assert_eq!(QualityPreset::Medium.tex_quality(), Some(1));
        assert!(medium.grass && medium.bloom && medium.lights);
        assert!(!medium.shadows && !medium.ssao);
        assert_eq!((medium.cull_px, medium.cull_px_grass), (2.0, 600.0));

        let mut high = GfxSettings::default();
        QualityPreset::High.apply(&mut high);
        assert_eq!(QualityPreset::High.tex_quality(), Some(1));
        assert!(high.grass && high.shadows && high.bloom && high.lights);
        assert!(!high.ssao);

        let mut ultra = GfxSettings::default();
        QualityPreset::Ultra.apply(&mut ultra);
        assert_eq!(QualityPreset::Ultra.tex_quality(), Some(0));
        assert!(ultra.grass && ultra.shadows && ultra.bloom && ultra.ssao && ultra.lights);
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
            _ => probe_render_path(),
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
/// Vulkan-class features regardless. So on **Windows and Linux** we restrict to Vulkan: a
/// Vulkan-capable machine runs, and a Vulkan-less one is caught by `main`'s pre-flight with an
/// actionable message instead of a confusing panic (or, on Linux, an unsupported GL fallback).
/// Modern AMD/NVIDIA/Intel drivers ship Vulkan on both OSes; the GL path in `all()` can't drive the
/// GPU-driven features anyway. macOS/other keep wgpu's default (all -> Metal) backends.
pub fn allowed_backends() -> wgpu::Backends {
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        wgpu::Backends::VULKAN
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
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

/// Auto-select the render path from the adapter (the `_ =>` arm of `RenderPath::from_env_or`;
/// an explicit `EFT_RENDER=` skips this entirely):
///  * features present, sane driver → GpuDriven (the full renderer);
///  * LLPC driver quirk → **Standard** — textured via Bevy's stock PBR pipelines, which that
///    compiler handles fine (field report: RX 7800 XT on an AMDVLK-lineage ICD device-loses on
///    the first frame of the gpu-driven bindless/indirect pipelines, and the M0 fallback's
///    untextured look was the immediate next complaint — Standard gives them real textures);
///  * features missing / probe error → M0 (the safest, lightest path for weak adapters).
fn probe_render_path() -> RenderPath {
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
            let info = a.get_info();
            // Driver quirk (field report: RX 7800 XT, driver_info "26.2.2 (LLPC)"): the LLPC
            // shader-compiler stack (AMDVLK-lineage Vulkan ICD, not the standard Adrenalin
            // compiler) device-loses on the FIRST FRAME of the gpu-driven path's bindless/
            // indirect pipelines — the app dies at swapchain acquire right after "GPU buffers +
            // bind groups built". The feature bits all report supported, so only the driver
            // string identifies it. Bevy's standard PBR pipelines (the menu, and the Standard
            // path) run fine there → fall back to Standard, which keeps textures.
            if info.driver_info.contains("LLPC") {
                eprintln!(
                    "gpu probe: {} uses an LLPC Vulkan compiler ({}) — known to crash the \
                     gpu-driven pipelines; auto-selecting the Standard (Bevy PBR) path. Install \
                     the standard AMD Adrenalin driver for the full renderer, or force with \
                     EFT_RENDER=gpu / EFT_RENDER=m0.",
                    info.name, info.driver_info
                );
                return RenderPath::Standard;
            }
            let ok = a.features().contains(need);
            eprintln!(
                "gpu probe: {} ({:?}/{:?}) gpu-driven={}",
                info.name, info.device_type, info.backend, ok
            );
            if ok {
                RenderPath::GpuDriven
            } else {
                eprintln!(
                    "render path: GPU lacks MULTI_DRAW_INDIRECT / bindless features - \
                     auto-selecting the M0 instanced path (override with EFT_RENDER=gpu)"
                );
                RenderPath::M0Instanced
            }
        }
        Err(e) => {
            eprintln!("gpu probe: adapter request failed ({e}) - treating GPU-driven as UNSUPPORTED, using M0");
            RenderPath::M0Instanced
        }
    }
}
