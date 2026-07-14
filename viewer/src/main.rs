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
mod render;

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::view::NoIndirectDrawing;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use eftpack::Pack;
use render::{EftInstancingPlugin, LoadedPack};

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
    // ---- parse argv: pack dir ----
    let pack_dir = std::env::args().nth(1);
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
                    resolution: (1600.0, 900.0).into(),
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
    .add_plugins(EftInstancingPlugin);

    if let Some(p) = pack {
        app.insert_resource(LoadedPack(p));
    }

    app.insert_resource(ClearColor(Color::srgb(0.55, 0.58, 0.58))) // overcast horizon stand-in
        .add_systems(Startup, setup)
        .add_systems(Update, (cursor_grab, flycam_look, flycam_move).chain());

    #[cfg(feature = "egui")]
    {
        // NOTE: bevy_egui's plugin ctor / context-access API drift between point
        // releases; adjust these two lines if they don't match your bevy_egui 0.37.x.
        app.add_plugins(bevy_egui::EguiPlugin::default())
            .add_systems(Update, stats_ui);
    }

    app.run();
}

/// Spawn camera + a key light, framed on the pack bounds if one is loaded.
fn setup(mut commands: Commands, pack: Option<Res<LoadedPack>>) {
    // Frame the world AABB: stand back along +Y/+Z by the half-diagonal.
    let (target, cam_pos, far) = match pack.as_ref() {
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
    };

    let dir = (target - cam_pos).normalize_or_zero();
    let yaw = dir.x.atan2(-dir.z);
    let pitch = dir.y.asin();

    commands.spawn((
        Camera3d::default(),
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
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    if mouse.just_pressed(MouseButton::Right) {
        window.cursor_options.grab_mode = CursorGrabMode::Locked;
        window.cursor_options.visible = false;
    }
    if mouse.just_released(MouseButton::Right) {
        window.cursor_options.grab_mode = CursorGrabMode::None;
        window.cursor_options.visible = true;
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
