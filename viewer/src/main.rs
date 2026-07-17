//! eft_viewer — native GPU-driven EFT map viewer (Bevy 0.17).
//!
//! Usage:  eft_viewer <path-to-.eftpack-dir>
//!
//! M0 target dataset is the "interchange" pack. This binary opens a window, sets
//! up a fly camera, loads the pack (reading its layout FROM manifest.json), and
//! draws it via the M0 custom instanced path (`render::instancing`): one instanced
//! draw per unique mesh, the FULL 3x4 affine (incl shear/mirror) applied in the
//! vertex shader — NEVER TRS-decomposed. The GPU-driven compute-cull upgrade is
//! designed in `render::gpu_driven` (M1).

mod eftpack;
mod inspect;
mod loot;
mod menu;
mod menu_fx;
mod pathfind;
mod paths;
mod pick;
mod poi;
mod render;
mod ui;

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::core_pipeline::Skybox;
use bevy::post_process::bloom::Bloom;
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection, Hdr, NoIndirectDrawing};
use bevy::window::{CursorGrabMode, CursorOptions, PresentMode, PrimaryWindow};

use eftpack::Pack;
use render::{
    CullCamera, EftGpuDrivenPlugin, EftInstancingPlugin, GradeLutCpu, GradePlugin, LoadedPack,
    RenderPath,
};

/// Fly camera state (WASD + mouse-look while RMB held; QE up/down; Shift = fast).
#[derive(Component)]
struct FlyCam {
    speed: f32,
    boost: f32,
    sensitivity: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for FlyCam {
    fn default() -> Self {
        Self {
            speed: 40.0,
            boost: 6.0,
            sensitivity: 0.0025,
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

/// UI-driven camera command: set `fly_to` from any egui panel (marker search, quest jump, route
/// start) and `apply_camera_command` frames the camera on that world point next frame. This keeps
/// the panels (ui.rs) decoupled from the private `FlyCam` — they only touch this resource, mirroring
/// the `LayerToggles` -> reactive-apply pattern.
#[derive(Resource, Default)]
pub struct CameraCommand {
    pub fly_to: Option<Vec3>,
}

/// UI map dropdown target: when set, `apply_map_switch` restarts the viewer into that pack.
/// (The GPU-driven path builds its buffers/bind-groups exactly once by design — a process swap
/// is the honest, robust map switch; the new instance inherits the current env settings.)
#[derive(Resource, Default)]
pub struct MapSwitch(pub Option<String>);

/// Spawn a fresh viewer on the selected pack, then exit this one. The viewer-OWNED pathfind
/// server child is stopped first so the switch doesn't orphan it (Codex review); an externally
/// started server is untouched.
fn apply_map_switch(
    mut sw: ResMut<MapSwitch>,
    mut server: ResMut<pathfind::PathfindServer>,
    mut exit: MessageWriter<bevy::app::AppExit>,
) {
    if sw.0.is_none() {
        return; // immutable-ish fast path: don't dirty change detection via take() every frame
    }
    let Some(pack) = sw.0.take() else { return };
    match std::env::current_exe() {
        Ok(exe) => {
            let mut cmd = std::process::Command::new(exe);
            cmd.arg(&pack);
            if let Some(rp) = std::env::args().nth(2) {
                cmd.arg(rp); // forward an argv render-path token (EFT_RENDER env inherits anyway)
            }
            match cmd.spawn() {
            Ok(_) => {
                info!("map switch: relaunching into {pack}");
                server.stop_owned_child();
                exit.write(bevy::app::AppExit::Success);
            }
                Err(e) => error!("map switch: failed to spawn viewer for {pack}: {e}"),
            }
        }
        Err(e) => error!("map switch: current_exe failed: {e}"),
    }
}

/// Apply the main-world halves of the runtime graphics settings: Bloom (component add/remove +
/// intensity) and the grade-LUT toggle (Tonemapping::None + LUT pass vs TonyMcMapface + hand
/// grade). Runs only when the settings actually changed.
fn apply_gfx_camera(
    mut commands: Commands,
    gfx: Res<render::GfxSettings>,
    cam: Query<Entity, With<FlyCam>>,
) {
    if !gfx.is_changed() {
        return;
    }
    let Ok(e) = cam.single() else { return };
    let mut ec = commands.entity(e);
    if gfx.bloom {
        ec.insert(Bloom {
            intensity: gfx.bloom_intensity,
            ..Bloom::NATURAL
        });
    } else {
        ec.remove::<Bloom>();
    }
    if gfx.grade && gfx.grade_available {
        // Game grade LUT owns the display chain (the render node applies it after Bloom).
        ec.insert(Tonemapping::None);
        ec.remove::<ColorGrading>();
    } else {
        // Fallback approximation (same values as the EFT_GRADE=0 path in setup()).
        ec.insert((
            Tonemapping::TonyMcMapface,
            ColorGrading {
                global: ColorGradingGlobal {
                    exposure: 0.4,
                    temperature: -0.02,
                    tint: -0.005,
                    post_saturation: 0.95,
                    ..default()
                },
                shadows: ColorGradingSection {
                    lift: 0.02,
                    ..default()
                },
                midtones: ColorGradingSection {
                    saturation: 0.98,
                    contrast: 1.16,
                    ..default()
                },
                ..default()
            },
        ));
    }
}

/// Consume a pending `CameraCommand::fly_to`: place the fly-cam at a framing offset above the target,
/// looking at it, and sync `FlyCam.yaw/pitch` so subsequent mouse-look continues smoothly.
fn apply_camera_command(mut cmd: ResMut<CameraCommand>, mut q: Query<(&mut Transform, &mut FlyCam)>) {
    if cmd.fly_to.is_none() {
        return; // read-only fast path: a take() through DerefMut would dirty change detection
    }
    let Some(target) = cmd.fly_to.take() else {
        return;
    };
    let Ok((mut tf, mut cam)) = q.single_mut() else {
        return;
    };
    let cam_pos = target + Vec3::new(6.0, 11.0, 18.0); // pulled back + up for context
    let dir = (target - cam_pos).normalize_or_zero();
    cam.yaw = dir.x.atan2(-dir.z); // same convention as `setup` (main.rs) / flycam_look
    cam.pitch = dir.y.asin();
    tf.translation = cam_pos;
    tf.rotation = Quat::from_axis_angle(Vec3::Y, cam.yaw) * Quat::from_axis_angle(Vec3::X, cam.pitch);
}

fn main() {
    // --version/--help fast path BEFORE any Bevy/GPU init: CI runners have no usable GPU, so
    // this is the only smoke test a workflow can run (redistribution PR5).
    if let Some(flag) = std::env::args().nth(1) {
        if matches!(flag.as_str(), "--version" | "-V") {
            println!("eft_viewer {} ({})", env!("CARGO_PKG_VERSION"), env!("CARGO_PKG_NAME"));
            return;
        }
        if matches!(flag.as_str(), "--help" | "-h") {
            println!(
                "eft_viewer [<pack-dir>] [m0|gpu]\n\
                 no args: start menu (scans <exe>/packs).  env: EFT_PACK, EFT_RENDER, EFT_SHADOWS,\n\
                 EFT_GRADE/EFT_GRADE_EXPOSURE, EFT_FOG, EFT_UNCAPPED, EFT_HIDDEN, EFT_SHOT,\n\
                 EFT_GAME_DATA, EFT_LOOT_JSON, EFT_TEX_BC=0. Docs: README_DIST.md"
            );
            return;
        }
    }
    // ---- parse argv: pack dir + optional render-path token ----
    // Pack selection order: explicit argv[1] > EFT_PACK env > first existing default pack.
    // Default map is LIGHTHOUSE (falls back to interchange if its pack isn't built), so a
    // bare `eft_viewer` with no arguments opens a map instead of an empty window.
    // Bare launch (no argv pack, no EFT_PACK) opens the START MENU (menu.rs) instead of a
    // default map — the menu's PLAY relaunches with the chosen pack as argv[1].
    let pack_dir = std::env::args().nth(1)
        .filter(|a| !a.starts_with('-'))
        .or_else(|| std::env::var("EFT_PACK").ok().filter(|s| !s.is_empty()));
    // A/B selector: `EFT_RENDER=m0|gpu` env, or a 2nd argv token; default = GPU-driven.
    let render_path = RenderPath::from_env_or(std::env::args().nth(2).as_deref());
    eprintln!("render path: {render_path:?}  (override with EFT_RENDER=m0|gpu)");
    // NOTE: this runs BEFORE DefaultPlugins installs Bevy's log subscriber, so use
    // eprintln! (not info!/error!) or the diagnostics are silently dropped and a
    // bad pack opens an empty window with no message (Codex P2).
    let pack = match &pack_dir {
        Some(dir) => match Pack::load(dir) {
            Ok(p) => {
                eprintln!(
                    "loaded .eftpack '{}': {} unique meshes, {} instances, {} materials",
                    p.manifest.dataset,
                    p.manifest.meshes.len(),
                    p.instances.len(),
                    p.materials.len(),
                );
                let mirrors = p.instances.iter().filter(|i| i.is_mirror()).count();
                eprintln!(
                    "  bounds center {:?} extent {:.1}m; {} mirrored instances (winding-flip, NOT baked)",
                    p.bounds_center(),
                    p.bounds_extent(),
                    mirrors
                );
                Some(p)
            }
            Err(e) => {
                eprintln!("failed to load pack '{}': {:#}", dir, e);
                None
            }
        },
        None => {
            eprintln!("no pack given — opening the start menu.  direct: eft_viewer <pack-dir>");
            None
        }
    };
    let menu_mode = pack.is_none();

    // Play-alongside-a-game friendliness: by DEFAULT cap to vsync (don't render faster than the
    // monitor) and idle when the window loses focus (see WinitSettings below) — so with the game in
    // the foreground the viewer stops churning the GPU. EFT_UNCAPPED=1 restores the old uncapped /
    // always-render behaviour for FPS A/B benchmarking.
    let uncapped = std::env::var("EFT_UNCAPPED").map(|v| v.trim() == "1").unwrap_or(false);
    let present_mode = if uncapped {
        PresentMode::AutoNoVsync // Immediate/Mailbox — uncapped, lowest latency (benchmark)
    } else {
        PresentMode::AutoVsync // capped to refresh — far less GPU when it IS in the foreground
    };

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "EFT Native Viewer".into(),
                    resolution: (1600u32, 900u32).into(),
                    present_mode,
                    // EFT_HIDDEN=1: render without showing a window (headless EFT_SHOT
                    // verification runs — GPU screenshot capture works on an invisible
                    // window; pair with EFT_UNCAPPED so the focus-idle gate doesn't stall).
                    // Bevy re-shows the window after the first present, so belt-and-braces:
                    // also park it far off-screen and skip the taskbar.
                    visible: !std::env::var("EFT_HIDDEN")
                        .map(|v| v.trim() == "1")
                        .unwrap_or(false),
                    position: if std::env::var("EFT_HIDDEN")
                        .map(|v| v.trim() == "1")
                        .unwrap_or(false)
                    {
                        WindowPosition::At(IVec2::new(-20000, -20000))
                    } else {
                        WindowPosition::Automatic
                    },
                    skip_taskbar: std::env::var("EFT_HIDDEN")
                        .map(|v| v.trim() == "1")
                        .unwrap_or(false),
                    ..default()
                }),
                ..default()
            })
            .set(AssetPlugin {
                // Shipped bundle: assets/ sits beside the exe (portability PR1). Dev
                // `cargo run` (exe in target/release, no assets/ there) falls back to
                // the compile-time crate dir so shader hot-editing keeps working.
                file_path: {
                    let exe_assets = paths::exe_dir().join("assets");
                    if exe_assets.is_dir() {
                        exe_assets.to_string_lossy().into_owned()
                    } else {
                        concat!(env!("CARGO_MANIFEST_DIR"), "/assets").to_string()
                    }
                },
                ..default()
            }),
    )
    // FPS readout for the before/after A/B measurement (prints to the console).
    .add_plugins((
        FrameTimeDiagnosticsPlugin::default(),
        LogDiagnosticsPlugin::default(),
    ));

    // Install exactly ONE render path so they can be FPS-compared cleanly.
    match render_path {
        RenderPath::M0Instanced => {
            app.add_plugins(EftInstancingPlugin);
        }
        RenderPath::GpuDriven => {
            app.add_plugins(EftGpuDrivenPlugin);
        }
        RenderPath::Standard => {
            app.add_plugins(render::EftStandardPlugin);
        }
    }

    // The REAL EFT display chain (grade LUT): resolved from the pack (or env/repo default) and
    // active by default — EFT_GRADE=0 falls back to the TonyMcMapface + hand-grade approximation.
    // Loaded BEFORE the pack moves into its resource so we can use its root for pack-local LUTs.
    let grade_lut = render::load_grade_lut(pack.as_ref().map(|p| p.root.as_path()));
    // Runtime graphics settings (UI "Graphics (experimental)"). Defaults reproduce the shipped
    // look; availability flags gate the toggles that need pack data.
    let mut gfx = render::GfxSettings::default();
    gfx.grade_available = grade_lut.is_some();
    app.insert_resource(gfx);
    if let Some(g) = grade_lut {
        app.insert_resource(g);
    }
    app.add_plugins((GradePlugin, render::SsaoPlugin));
    // Runtime graphics settings reach the render world on EVERY render path (grade/SSAO install
    // unconditionally, so the extraction can't live inside EftGpuDrivenPlugin — under EFT_RENDER=
    // m0/std the toggles would silently stop reaching the GPU).
    app.add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<render::GfxSettings>::default());

    if let Some(p) = pack {
        app.insert_resource(LoadedPack(p));
    }

    // Foreground-gated redraw: full-rate when the window is focused, near-idle (only user/window
    // events, ~2 Hz) when it's not — so alt-tabbing to your game frees the GPU. Skipped under
    // EFT_UNCAPPED so the benchmark keeps rendering continuously.
    if !uncapped {
        app.insert_resource(bevy::winit::WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            unfocused_mode: bevy::winit::UpdateMode::reactive_low_power(
                std::time::Duration::from_millis(500),
            ),
        });
    }

    // In-raid: overcast horizon stand-in. Menu mode: the egui menu's near-black #090909 —
    // the CentralPanel goes transparent when the real-asset 3D CCTV decor is active
    // (menu_fx), so the 3D clear IS the menu field; setup() also skips the Skybox then.
    app.insert_resource(if menu_mode {
        ClearColor(Color::srgb_u8(9, 9, 9))
    } else {
        ClearColor(Color::srgb(0.55, 0.58, 0.58))
    })
        .add_plugins(pick::PickPlugin) // double-LEFT-click raycast-vs-pack-data debug pick
        .add_plugins(loot::LootPlugin) // 823 loot containers from tarkmap out/loot.json
        .add_plugins(poi::PoiPlugin) // PMC/scav/boss spawns + extracts/doors/interactables
        .add_plugins(inspect::InspectPlugin) // left-click a marker -> floating info card (\u{2715} to close)
        .add_plugins(ui::UiPlugin) // right-hand layer-toggle panel
        .add_plugins(pathfind::PathfindPlugin) // on-demand routing via the :8091 GPU pathfind server
        .init_resource::<CameraCommand>() // UI-driven "fly the camera to X" (search / quest jump / route)
        .init_resource::<MapSwitch>() // UI map dropdown -> restart into the selected pack
        .add_systems(Startup, setup)
        .add_systems(Update, (cursor_grab, flycam_look, flycam_move).chain())
        .add_systems(Update, (apply_camera_command, auto_screenshot))
        .add_systems(Update, (apply_gfx_camera, apply_map_switch));

    #[cfg(feature = "egui")]
    {
        // NOTE: bevy_egui's plugin ctor / context-access API drift between point
        // releases; adjust these two lines if they don't match your bevy_egui 0.37.x.
        app.add_plugins(bevy_egui::EguiPlugin::default())
            // egui UI runs in EguiPrimaryContextPass, not Update (else ctx_mut() panics: no fonts).
            .add_systems(bevy_egui::EguiPrimaryContextPass, stats_ui);
        if menu_mode {
            app.add_systems(bevy_egui::EguiPrimaryContextPass, menu::menu_ui);
        }
    }
    // Start menu (bare launch): scan packs/, fingerprint the game install, present the map
    // manager. The in-raid panels check for this resource and stand down while it exists.
    // build_state() also runs the ONE-TIME local extraction of the real CCTV menu prop
    // (menu::ensure_menu_prop) before anything draws; spawn_menu_prop then loads it (or
    // silently leaves the vector camera in charge when the files are absent/corrupt).
    if menu_mode {
        app.insert_resource(menu::build_state());
        app.add_systems(Startup, menu_fx::spawn_menu_prop.after(setup));
        app.add_systems(Update, menu_fx::menu_prop_update);
    }

    app.run();
}

/// f32 -> IEEE 754 half bits (round-to-nearest-even). Shared by the sky cubemap and the grade
/// LUT (render::grade) so Rgba16Float textures need no `half` dependency (Rgba32Float is NOT
/// filterable — filtering samplers on it fail wgpu validation).
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let mut man = (x >> 13) & 0x3ff;
    if exp <= 0 {
        return sign; // flush denormals/underflow to signed zero (sky values never need them)
    }
    if exp >= 31 {
        exp = 30; // clamp to max finite half (65504) instead of inf
        man = 0x3ff;
    }
    sign | ((exp as u16) << 10) | man as u16
}

/// Procedural overcast-sky cubemap: the same horizon/zenith gradient family the shader's
/// `sky_reflect` uses (so reflections agree with the visible sky) plus a soft warm sun disk +
/// wide glow at the bake's sun_dir. HDR (disk peaks ~4.0) so Bloom picks it up. 6x128x128
/// Rgba16Float; Skybox.brightness rescales it against the camera's physical Exposure.
fn build_sky_cubemap(images: &mut Assets<Image>, sun: Vec3) -> Handle<Image> {
    const N: usize = 128;
    let mut data = Vec::with_capacity(N * N * 6 * 8);
    for face in 0..6 {
        for y in 0..N {
            for x in 0..N {
                let u = 2.0 * (x as f32 + 0.5) / N as f32 - 1.0;
                let v = 2.0 * (y as f32 + 0.5) / N as f32 - 1.0;
                // Standard wgpu/Vulkan cubemap texel->direction mapping, face order +X..-Z.
                let dir = match face {
                    0 => Vec3::new(1.0, -v, -u),
                    1 => Vec3::new(-1.0, -v, u),
                    2 => Vec3::new(u, 1.0, v),
                    3 => Vec3::new(u, -1.0, -v),
                    4 => Vec3::new(u, -v, 1.0),
                    _ => Vec3::new(-u, -v, -1.0),
                }
                .normalize();
                let up = (dir.y * 0.5 + 0.5).clamp(0.0, 1.0);
                let t = up * up;
                let horizon = Vec3::new(0.66, 0.72, 0.82);
                let zenith = Vec3::new(0.92, 0.98, 1.10);
                let mut sky = horizon.lerp(zenith, t);
                if dir.y < 0.0 {
                    // Below the horizon: fade to a darker sea/ground haze so coastline edges
                    // and downward reflections don't read as bright sky.
                    sky *= 1.0 - 0.55 * (-dir.y * 3.0).min(1.0);
                }
                let s = dir.dot(sun).max(0.0);
                // Overcast sun: a soft disk (not a hard point) + a broad warm glow behind cloud.
                let sun_col = Vec3::new(1.05, 1.0, 0.9);
                sky += sun_col * (s.powf(350.0) * 3.0 + s.powf(8.0) * 0.3);
                for c in [sky.x, sky.y, sky.z, 1.0] {
                    data.extend_from_slice(&f32_to_f16_bits(c).to_le_bytes());
                }
            }
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: N as u32,
            height: N as u32,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    images.add(image)
}

/// Spawn camera + a key light, framed on the pack bounds if one is loaded.
fn setup(
    mut commands: Commands,
    pack: Option<Res<LoadedPack>>,
    mut images: ResMut<Assets<Image>>,
    grade: Option<Res<GradeLutCpu>>,
    mut gfx: ResMut<render::GfxSettings>,
) {
    // Frame the world AABB: stand back along +Y/+Z by the half-diagonal.
    // Debug: EFT_LOOK="x,y,z" frames the camera CLOSE on that world point (e.g. a coordinate from
    // the double-click pick) instead of the whole-map overview — to confirm a specific mesh renders
    // where the data says it is.
    let look_override = std::env::var("EFT_LOOK").ok().and_then(|s| {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        (p.len() == 3).then(|| Vec3::new(p[0], p[1], p[2]))
    });
    let (target, cam_pos, far) = if let Some(t) = look_override {
        (t, t + Vec3::new(4.0, 6.0, 14.0), 4000.0)
    } else {
        match pack.as_ref() {
            Some(p) => {
                let c = p.0.bounds_center();
                let ext = p.0.bounds_extent().max(1.0);
                // Stand back ~1.34*ext; the far plane must clear the cam->far-corner
                // distance (~3*ext) with margin or the map center clips (Codex P1).
                (
                    c,
                    c + Vec3::new(0.0, ext * 0.6, ext * 1.2),
                    (ext * 6.0).max(2000.0),
                )
            }
            None => (Vec3::ZERO, Vec3::new(0.0, 20.0, 60.0), 2000.0),
        }
    };

    let dir = (target - cam_pos).normalize_or_zero();
    let yaw = dir.x.atan2(-dir.z);
    let pitch = dir.y.asin();

    // Sky sun direction: same volume.json sidecar + X-flip the SH/shadow path uses
    // (gpu_driven::extract_pack_to_render_world), so the skybox sun disk, the baked GI, and the
    // shader's reflected sun all agree. Fallback = a plausible high overcast sun.
    let sun_from_pack = pack
        .as_ref()
        .and_then(|p| {
            p.0.manifest
                .sidecars
                .volume_meta
                .as_deref()
                .map(|m| p.0.resolve_path(m)) // self-contained packs: pack-relative sidecars
        })
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
    // The experimental shadow toggle needs a real sun (matches the render side's sun_ok gate).
    gfx.shadows_available = sun_from_pack.is_some();
    let sun_dir = sun_from_pack.unwrap_or_else(|| Vec3::new(-0.45, 0.8, -0.4).normalize());
    // Menu mode (no pack — same test main() uses): NO skybox — the menu's ClearColor
    // (#090909, set in main) must be the backdrop behind the transparent egui panel /
    // the 3D CCTV decor (menu_fx).
    let menu_mode = pack.is_none();
    let sky = (!menu_mode).then(|| build_sky_cubemap(&mut images, sun_dir));

    let mut cam = commands.spawn((
        Camera3d {
            // SSAO (render::ssao) samples the main depth buffer — without TEXTURE_BINDING the
            // depth view is attachment-only and the SSAO bind group fails wgpu validation.
            depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING)
                .into(),
            ..default()
        },
        // HDR view target: the custom draw shader outputs LINEAR HDR radiance (sun glints,
        // sky reflections >1.0). Without this marker the pipeline specialized to an 8-bit sRGB
        // target and everything above 1.0 flat-clipped BEFORE tonemapping — and Bloom (which
        // #[require(Hdr)]s) was impossible.
        Hdr,
        Bloom {
            intensity: 0.06, // subtle: sun disk / glints / emissive bleed, not a haze filter
            ..Bloom::NATURAL
        },
        // Tonemapping is decided below: the REAL game grade LUT (render::grade) replaces the
        // whole tonemap+grade chain when active; the TonyMcMapface + hand ColorGrading
        // approximation is only the EFT_GRADE=0 fallback.
        // Far plane derived from pack bounds so the whole map is visible; the
        // default 1000 m clipped Interchange (extent >745 m) — Codex P1.
        Projection::Perspective(PerspectiveProjection {
            far,
            ..default()
        }),
        Transform::from_translation(cam_pos).looking_at(target, Vec3::Y),
        // The custom instancing path is incompatible with Bevy's GPU indirect
        // draw preprocessing; opt this view out (matches the bevy example).
        NoIndirectDrawing,
        // Tag THIS camera as the cull-frustum source (Bevy has multiple ExtractedViews;
        // the GPU cull must use the player view, not a prepass/default one).
        CullCamera,
        FlyCam {
            yaw,
            pitch,
            ..default()
        },
    ));
    // Display chain: the REAL game grade LUT (render::grade — Hejl + film curves + Fahrenheit
    // fit, baked FROM THE GAME and identical on every map) replaces Bevy's tonemapping when
    // active: Tonemapping::None keeps the scene linear for the LUT pass, which runs after Bloom.
    // Fallback (EFT_GRADE=0 / LUT missing): TonyMcMapface + a hand-grade approximation.
    if grade.is_some() {
        cam.insert(Tonemapping::None);
    } else {
        cam.insert((
            Tonemapping::TonyMcMapface,
            ColorGrading {
                global: ColorGradingGlobal {
                    exposure: 0.4,
                    temperature: -0.02,
                    tint: -0.005,
                    post_saturation: 0.95, // EFT palette is DEsaturated
                    ..default()
                },
                shadows: ColorGradingSection {
                    lift: 0.02, // milky (not crushed) blacks
                    ..default()
                },
                midtones: ColorGradingSection {
                    saturation: 0.98,
                    contrast: 1.16, // midtone contrast carries the look instead of saturation
                    ..default()
                },
                ..default()
            },
        ));
    }

    // Analytic-sky key light (real sun_dir comes from the SH volume sidecar later).
    // M0 lighting is a fixed key baked into the shader; this light is for when the
    // material path (M3) uses Bevy's lighting for non-instanced helpers.
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(1.0, 3.0, 1.5).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    info!("camera at {cam_pos:?} looking at {target:?}");
    info!("RMB: look  |  WASD: move  |  QE: down/up  |  Shift: fast");
}

/// Hold RMB to capture the cursor for mouse-look; release to free it.
fn cursor_grab(
    mouse: Res<ButtonInput<MouseButton>>,
    pointer_on_ui: Res<inspect::PointerOnUi>,
    // Bevy 0.17 split cursor state out of `Window` into a `CursorOptions` component
    // on the same window entity.
    mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut cursor) = cursors.single_mut() else {
        return;
    };
    // Don't lock the cursor when the RMB press lands on an egui panel (Codex review: UI
    // right-clicks were hijacking the camera).
    if mouse.just_pressed(MouseButton::Right) && !pointer_on_ui.0 {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
    if mouse.just_released(MouseButton::Right) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

/// Mouse-look (only while RMB held). Uses the `AccumulatedMouseMotion` resource
/// (version-stable) instead of a buffered-event reader whose type name churns
/// across Bevy releases.
fn flycam_look(
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    pointer_on_ui: Res<inspect::PointerOnUi>,
    mut q: Query<(&mut Transform, &mut FlyCam)>,
) {
    if !mouse.pressed(MouseButton::Right) || pointer_on_ui.0 {
        return;
    }
    let delta = motion.delta;
    if delta == Vec2::ZERO {
        return;
    }
    for (mut tf, mut cam) in &mut q {
        cam.yaw -= delta.x * cam.sensitivity;
        cam.pitch = (cam.pitch - delta.y * cam.sensitivity)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);
        tf.rotation =
            Quat::from_axis_angle(Vec3::Y, cam.yaw) * Quat::from_axis_angle(Vec3::X, cam.pitch);
    }
}

/// WASD/QE movement in camera space.
fn flycam_move(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    ui_kb: Res<inspect::UiWantsKeyboard>,
    mut q: Query<(&mut Transform, &FlyCam)>,
) {
    // Typing 'wasd' into the marker-search box must not fly the camera (Codex review).
    if ui_kb.0 {
        return;
    }
    let dt = time.delta_secs();
    for (mut tf, cam) in &mut q {
        let mut v = Vec3::ZERO;
        let fwd = *tf.forward();
        let right = *tf.right();
        if keys.pressed(KeyCode::KeyW) {
            v += fwd;
        }
        if keys.pressed(KeyCode::KeyS) {
            v -= fwd;
        }
        if keys.pressed(KeyCode::KeyD) {
            v += right;
        }
        if keys.pressed(KeyCode::KeyA) {
            v -= right;
        }
        if keys.pressed(KeyCode::KeyE) {
            v += Vec3::Y;
        }
        if keys.pressed(KeyCode::KeyQ) {
            v -= Vec3::Y;
        }
        if v != Vec3::ZERO {
            let mut speed = cam.speed;
            if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                speed *= cam.boost;
            }
            tf.translation += v.normalize() * speed * dt;
        }
    }
}

/// Reliable frame capture: with `EFT_SHOT=<path>` set, save ONE screenshot of the primary
/// window ~90 Update-frames in (a beat after the heavy Startup finishes) via Bevy's own GPU
/// screenshot — this bypasses the DWM/flip-model capture that makes external CopyFromScreen
/// grab a blank white frame.
fn auto_screenshot(mut commands: Commands, mut frames: Local<u32>) {
    *frames += 1;
    if *frames != 90 {
        return;
    }
    if let Ok(path) = std::env::var("EFT_SHOT") {
        use bevy::render::view::screenshot::{save_to_disk, Screenshot};
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        info!("auto-screenshot -> {path}");
    }
}

#[cfg(feature = "egui")]
fn stats_ui(
    mut contexts: bevy_egui::EguiContexts,
    pack: Option<Res<LoadedPack>>,
    menu: Option<Res<menu::MenuState>>,
) {
    if menu.is_some() {
        return; // start menu owns the screen
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    bevy_egui::egui::Window::new("pack").show(ctx, |ui| match pack.as_ref() {
        Some(p) => {
            ui.label(format!("dataset: {}", p.0.manifest.dataset));
            ui.label(format!("meshes:  {}", p.0.manifest.meshes.len()));
            ui.label(format!("instances: {}", p.0.instances.len()));
            ui.label(format!("materials: {}", p.0.materials.len()));
            let mirrors = p.0.instances.iter().filter(|i| i.is_mirror()).count();
            ui.label(format!("mirrored: {mirrors}"));
        }
        None => {
            ui.label("no pack loaded");
        }
    });
}
