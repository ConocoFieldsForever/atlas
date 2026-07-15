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
mod loot;
mod pick;
mod render;
mod ui;

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection, NoIndirectDrawing};
use bevy::window::{CursorGrabMode, CursorOptions, PresentMode, PrimaryWindow};

use eftpack::Pack;
use render::{CullCamera, EftGpuDrivenPlugin, EftInstancingPlugin, LoadedPack, RenderPath};

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

fn main() {
    // ---- parse argv: pack dir + optional render-path token ----
    let pack_dir = std::env::args().nth(1);
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
            eprintln!("no .eftpack path given — opening empty viewer.  usage: eft_viewer <pack-dir>");
            None
        }
    };

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "EFT Native Viewer".into(),
                    resolution: (1600u32, 900u32).into(),
                    // Lowest-latency present: no vsync wait — uncaps FPS and minimizes
                    // input-to-photon latency. AutoNoVsync picks Immediate/Mailbox.
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            })
            .set(AssetPlugin {
                // Assets live in <viewer-crate>/assets; anchor to the crate dir so
                // `cargo run` from the workspace root (the documented launch dir)
                // resolves shaders regardless of cwd (Codex P1 — shader not found).
                file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").to_string(),
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

    if let Some(p) = pack {
        app.insert_resource(LoadedPack(p));
    }

    app.insert_resource(ClearColor(Color::srgb(0.55, 0.58, 0.58))) // overcast horizon stand-in
        .add_plugins(pick::PickPlugin) // double-LEFT-click raycast-vs-pack-data debug pick
        .add_plugins(loot::LootPlugin) // 823 loot containers from tarkmap out/loot.json
        .add_plugins(ui::UiPlugin) // right-hand layer-toggle panel
        .add_systems(Startup, setup)
        .add_systems(Update, (cursor_grab, flycam_look, flycam_move).chain());

    #[cfg(feature = "egui")]
    {
        // NOTE: bevy_egui's plugin ctor / context-access API drift between point
        // releases; adjust these two lines if they don't match your bevy_egui 0.37.x.
        app.add_plugins(bevy_egui::EguiPlugin::default())
            // egui UI runs in EguiPrimaryContextPass, not Update (else ctx_mut() panics: no fonts).
            .add_systems(bevy_egui::EguiPrimaryContextPass, stats_ui);
    }

    app.run();
}

/// Spawn camera + a key light, framed on the pack bounds if one is loaded.
fn setup(mut commands: Commands, pack: Option<Res<LoadedPack>>) {
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

    commands.spawn((
        Camera3d::default(),
        // Phase 3 — EFT display grade. AgX filmic base (muted, desaturates highlights toward
        // the game's washed palette) + a subtle grade: cool shadows, slight green tint, reduced
        // saturation, and a small shadow lift for EFT's milky (not crushed) blacks. Exposure is
        // the shared brightness knob — it pairs with gi_intensity in the SH volume uniform.
        Tonemapping::TonyMcMapface,
        ColorGrading {
            global: ColorGradingGlobal {
                exposure: 0.4,             // brighten — prior grade came out too dark
                temperature: -0.03,        // barely cool
                tint: -0.02,
                post_saturation: 1.08,     // keep color (0.90 was washed; ~1.08 reads right)
                ..default()
            },
            shadows: ColorGradingSection {
                lift: 0.03,                // raise blacks so shadows aren't crushed
                ..default()
            },
            midtones: ColorGradingSection {
                saturation: 1.06,          // color without the contrast that darkened it
                ..default()
            },
            ..default()
        },
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
    // Bevy 0.17 split cursor state out of `Window` into a `CursorOptions` component
    // on the same window entity.
    mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut cursor) = cursors.single_mut() else {
        return;
    };
    if mouse.just_pressed(MouseButton::Right) {
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
    mut q: Query<(&mut Transform, &mut FlyCam)>,
) {
    if !mouse.pressed(MouseButton::Right) {
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
    mut q: Query<(&mut Transform, &FlyCam)>,
) {
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

#[cfg(feature = "egui")]
fn stats_ui(mut contexts: bevy_egui::EguiContexts, pack: Option<Res<LoadedPack>>) {
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
