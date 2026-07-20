//! atlas — native GPU-driven EFT map viewer (Bevy 0.17).
//!
//! Usage:  atlas <path-to-.eftpack-dir>
//!
//! M0 target dataset is the "interchange" pack. This binary opens a window, sets
//! up a fly camera, loads the pack (reading its layout FROM manifest.json), and
//! draws it via the M0 custom instanced path (`render::instancing`): one instanced
//! draw per unique mesh, the FULL 3x4 affine (incl shear/mirror) applied in the
//! vertex shader — NEVER TRS-decomposed. The GPU-driven compute-cull upgrade is
//! designed in `render::gpu_driven` (M1).

mod eftpack;
mod i18n;
mod inspect;
mod jobs;
mod loot;
mod maps;
mod menu;
mod menu_fx;
mod nav;
mod nav_bake;
mod sh_bake;
mod navigate_panel;
mod pathfind;
mod paths;
mod pick;
mod planner;
mod poi;
mod progress;
mod render;
mod tasks_panel;
mod ui;
mod ui_theme;
mod update;
mod walk_ground;

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

/// Camera-tab settings (the toolbar's camera panel edits these; the flycam systems read them).
/// Decoupled from the private `FlyCam` like `CameraCommand`.
#[derive(Resource)]
pub struct CameraSettings {
    /// Vertical FOV in degrees (applied to the perspective projection).
    pub fov_deg: f32,
    /// Base fly-move speed (m/s); the scroll wheel scales this live.
    pub fly_speed: f32,
    /// Base WALK speed (m/s); the scroll wheel scales this in walk mode, and jump height rides
    /// off it (scroll faster -> move faster + jump higher).
    pub walk_speed: f32,
    /// Walk mode (ground-follow + jump) vs free-fly.
    pub walk_mode: bool,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            fov_deg: 60.0,   // Bevy's default PerspectiveProjection fov (0.25π ≈ 45°? no — ~60)
            fly_speed: 40.0, // matches the old FlyCam::default speed
            walk_speed: 5.0, // human-ish
            walk_mode: std::env::var("EFT_WALK").map(|v| v.trim() == "1").unwrap_or(false),
        }
    }
}

/// Scroll wheel scales the fly speed live (up = faster), clamped to a sane band. Ignored while
/// the pointer is over the UI (scrolling a panel must not change speed).
fn flycam_scroll(
    scroll: Res<bevy::input::mouse::AccumulatedMouseScroll>,
    pointer_on_ui: Res<inspect::PointerOnUi>,
    mut settings: ResMut<CameraSettings>,
) {
    if pointer_on_ui.0 || scroll.delta.y == 0.0 {
        return;
    }
    // ~1.15x per notch; clamp so it never crawls or teleports.
    let factor = 1.15f32.powf(scroll.delta.y);
    if settings.walk_mode {
        // In walk mode the wheel juices walk speed (and jump height rides off it) into a
        // human-ish band, so a fast scroll makes the walk cam quicker AND jump higher.
        settings.walk_speed = (settings.walk_speed * factor).clamp(1.5, 12.0);
    } else {
        settings.fly_speed = (settings.fly_speed * factor).clamp(2.0, 4000.0);
    }
}

/// Apply the camera-tab FOV to the perspective projection when it changes.
fn apply_camera_fov(
    settings: Res<CameraSettings>,
    mut q: Query<&mut Projection, With<CullCamera>>,
) {
    if !settings.is_changed() {
        return;
    }
    for mut proj in &mut q {
        if let Projection::Perspective(p) = &mut *proj {
            p.fov = settings.fov_deg.clamp(20.0, 120.0).to_radians();
        }
    }
}

/// UI map dropdown / menu PLAY target: when set, `load_map` swaps to that pack IN-PLACE (replace
/// `LoadedPack` + bump `MapEpoch`; the epoch-gated teardown/rebuild systems do the rest — no process
/// relaunch, so a background build keeps running across the switch). `EFT_RELAUNCH_ON_SWITCH=1`
/// restores the old process-swap behavior as a fallback until the in-place path is fully trusted.
#[derive(Resource, Default)]
pub struct MapSwitch(pub Option<String>);

/// Forced LOD level for the GPU-driven viewer (the graphics-panel LOD selector). 0 = finest LOD
/// (default / best detail); a higher value forces a coarser LOD per LODGroup (clamped to each
/// group's max available level). Only meaningful on `--alllod` packs that carry multiple LODs; a
/// no-op on lean LOD0-only packs. Changing it bumps `MapEpoch` so `build_cpu_data` rebuilds the
/// instance set for the new level.
#[derive(Resource)]
pub struct ForcedLod(pub i32);
impl Default for ForcedLod {
    fn default() -> Self {
        ForcedLod(0)
    }
}

/// Re-trigger the per-map GPU rebuild when the LOD selector changes: bump `MapEpoch`, which the
/// teardown/rebuild systems (incl. `build_cpu_data`, now LOD-aware) already gate on. Skips the
/// initial add so it doesn't double-fire on startup.
///
/// Finding 7: a `MapEpoch` bump is DESTRUCTIVE (it reframes the camera and clears nav/pins/routes/
/// plans/quests). On a LOD0-only pack — which every SHIPPED pack is — changing the LOD selector
/// yields the IDENTICAL instance set (`instances_by_mesh_for_lod` collapses to the full set), so the
/// bump would nuke all that state for no visual change. Only bump when the pack ACTUALLY carries
/// multiple LODs (an `--alllod` pack), so the selector is a true no-op on standard packs and never
/// touches camera/nav/POI/plan state there. (Reset-to-defaults also resets `ForcedLod`, in ui.rs.)
fn bump_epoch_on_lod_change(
    lod: Res<ForcedLod>,
    pack: Option<Res<LoadedPack>>,
    mut epoch: ResMut<render::MapEpoch>,
) {
    if !lod.is_changed() || lod.is_added() {
        return;
    }
    // Effective LOD set is unchanged unless the pack has any grouped instance beyond LOD0.
    let has_multi_lod = pack
        .as_ref()
        .map(|p| p.0.instances.iter().any(|i| i.lod_group >= 0 && i.lod_index > 0))
        .unwrap_or(false);
    if has_multi_lod {
        epoch.0 = epoch.0.wrapping_add(1);
    }
}

/// Set by the toolbar's "back to menu" button: relaunch the process with NO pack so the start menu
/// (map manager) opens. The menu<->raid transition still relaunches (the in-place path is raid->raid
/// only for now); a background build DOES die on this relaunch — full in-place menu is a follow-up.
#[derive(Resource, Default)]
pub struct ReturnToMenu(pub bool);

/// Relaunch into the start menu when `ReturnToMenu` is set (a fresh process with no pack argv AND
/// EFT_PACK stripped, so `main()` opens the menu instead of re-opening the current pack).
fn return_to_menu(
    mut req: ResMut<ReturnToMenu>,
    mut server: ResMut<pathfind::PathfindServer>,
    mut exit: MessageWriter<bevy::app::AppExit>,
) {
    if !req.0 {
        return;
    }
    req.0 = false;
    match std::env::current_exe() {
        Ok(exe) => {
            // No pack arg + EFT_PACK removed -> menu mode (main() pack-selection order).
            match std::process::Command::new(exe).env_remove("EFT_PACK").spawn() {
                Ok(_) => {
                    info!("returning to the start menu (relaunch, no pack)");
                    server.stop_owned_child();
                    exit.write(bevy::app::AppExit::Success);
                }
                Err(e) => error!("return to menu: spawn failed: {e}"),
            }
        }
        Err(e) => error!("return to menu: current_exe failed: {e}"),
    }
}

/// Load the selected pack in-place: swap `LoadedPack`, reload pack-local grade/gfx flags, drop the
/// per-map ground grid, and bump `MapEpoch` (which drives every per-map rebuild + the render-world
/// GPU reset). On `EFT_RELAUNCH_ON_SWITCH=1`, falls back to spawning a fresh process + exiting.
fn load_map(
    mut sw: ResMut<MapSwitch>,
    mut server: ResMut<pathfind::PathfindServer>,
    mut exit: MessageWriter<bevy::app::AppExit>,
    menu: Option<Res<menu::MenuState>>,
    render_path: Option<Res<RenderPath>>,
    mut pending: ResMut<PendingMapLoad>,
) {
    if sw.0.is_none() {
        return; // fast path: don't dirty change detection via take() every frame
    }
    let Some(dir) = sw.0.take() else { return };

    // RELAUNCH (not in-place) when:
    //  - menu PLAY (the menu->raid transition also needs MenuState torn down + the menu UI stood
    //    down, which the in-place path doesn't yet do), OR
    //  - the render path is NOT GPU-driven (only EftGpuDrivenPlugin has an epoch-aware rebuild; the
    //    m0/std paths spawn geometry once at Startup, so in-place would leave stale map geometry), OR
    //  - EFT_RELAUNCH_ON_SWITCH=1 (explicit fallback).
    let not_gpu_driven = render_path.map(|r| *r != RenderPath::GpuDriven).unwrap_or(false);
    let relaunch = menu.is_some()
        || not_gpu_driven
        || std::env::var("EFT_RELAUNCH_ON_SWITCH").map(|v| v.trim() == "1").unwrap_or(false);
    if relaunch {
        match std::env::current_exe() {
            Ok(exe) => {
                let mut cmd = std::process::Command::new(exe);
                cmd.arg(&dir);
                if let Some(rp) = std::env::args().nth(2) {
                    cmd.arg(rp);
                }
                match cmd.spawn() {
                    Ok(_) => {
                        info!("map switch: relaunching into {dir}");
                        server.stop_owned_child();
                        exit.write(bevy::app::AppExit::Success);
                    }
                    Err(e) => error!("map switch: failed to spawn viewer for {dir}: {e}"),
                }
            }
            Err(e) => error!("map switch: current_exe failed: {e}"),
        }
        return;
    }

    // In-place swap: load the pack OFF-THREAD (AsyncComputeTaskPool) so the current map keeps
    // rendering — no ~1-2s freeze while ~650 MB is repacked. `poll_map_load` applies the result when
    // it's ready. A second switch REPLACES the pending load (drops the old task) — latest wins.
    let name = dir
        .rsplit(['/', '\\'])
        .next()
        .and_then(|n| n.strip_suffix(".eftpack"))
        .unwrap_or(&dir)
        .to_string();
    info!("map switch: loading '{name}' in place (async)\u{2026}");
    let task = bevy::tasks::AsyncComputeTaskPool::get()
        .spawn(async move { Pack::load(&dir).map_err(|e| format!("{e:#}")) });
    pending.0 = Some((name, task));
}

/// Clear a stale `MapLoadError` the moment a NEW async load is kicked off, so the error toast from a
/// previous failed attempt doesn't linger over a fresh (possibly succeeding) load.
fn clear_map_error_on_new_load(pending: Res<PendingMapLoad>, mut err: ResMut<MapLoadError>) {
    if pending.is_changed() && pending.loading().is_some() && err.0.is_some() {
        err.0 = None;
    }
}

/// Last async map-load FAILURE (finding 4): a corrupt/partial pack whose off-thread `Pack::load`
/// returned Err. `poll_map_load` sets it; the UI (`ui::map_load_error_panel`) shows a clear error
/// with a "Back to menu" action instead of leaving a blank window. Cleared when a new load starts or
/// one succeeds.
#[derive(Resource, Default)]
pub struct MapLoadError(pub Option<String>);

/// A pack being loaded off-thread for an in-place swap: (display name, load task). The current map
/// keeps rendering until `poll_map_load` applies the result — so a switch never freezes the frame.
#[derive(Resource, Default)]
pub struct PendingMapLoad(Option<(String, bevy::tasks::Task<Result<Pack, String>>)>);

impl PendingMapLoad {
    /// The name of the map currently loading (drives the loading indicator), or None.
    pub fn loading(&self) -> Option<&str> {
        self.0.as_ref().map(|(n, _)| n.as_str())
    }
}

/// Apply a finished async pack load: reload the pack-local grade/gfx flags, drop the ground grid,
/// then swap LoadedPack + bump MapEpoch (both via commands → one sync point). Same tail the old
/// synchronous `load_map` ran, now off the background task's completion.
fn poll_map_load(
    mut pending: ResMut<PendingMapLoad>,
    mut commands: Commands,
    epoch: Res<render::MapEpoch>,
    mut gfx: ResMut<render::GfxSettings>,
    mut load_err: ResMut<MapLoadError>,
    // Latch the "GPU build in progress" flag the instant the file load is applied, so the loading
    // indicator stays visible with no 1-frame gap between PendingMapLoad clearing and the render
    // world starting the (multi-frame) GPU build. GPU-driven path only (Option = absent under m0/std).
    gpu_load: Option<Res<render::GpuLoadSignal>>,
) {
    let Some((_, task)) = pending.0.as_mut() else {
        return;
    };
    let Some(result) = bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(task))
    else {
        return; // still loading — the current map keeps rendering
    };
    let name = pending.0.take().map(|(n, _)| n).unwrap_or_default();
    match result {
        Ok(p) => {
            if let Some(s) = &gpu_load {
                s.begin(); // GPU build starts next frame (build_cpu_data); keep the toast up
            }
            info!(
                "map switch: '{}' loaded in place ({} meshes, {} instances)",
                p.manifest.dataset,
                p.manifest.meshes.len(),
                p.instances.len()
            );
            // Reload the pack-local grade LUT + availability flags (gfx change re-runs
            // apply_gfx_camera: Bloom + Tonemapping selection).
            let grade_lut = render::load_grade_lut(Some(p.root.as_path()));
            gfx.grade_available = grade_lut.is_some();
            let (_, sun_ok) = pack_sun_dir(Some(&p));
            gfx.shadows_available = sun_ok;
            match grade_lut {
                Some(g) => commands.insert_resource(g),
                None => commands.remove_resource::<GradeLutCpu>(),
            }
            commands.remove_resource::<walk_ground::GroundGrid>();
            commands.insert_resource(LoadedPack(p));
            commands.insert_resource(render::MapEpoch(epoch.0.wrapping_add(1)));
            load_err.0 = None; // a successful load clears any prior failure toast
        }
        Err(e) => {
            error!("map switch: failed to load pack '{name}': {e}");
            // Surface the failure (finding 4): `pending` is now cleared (loading toast gone) and no
            // MapEpoch bump means the GPU build never starts, so the window would otherwise sit blank
            // with no message. The MapLoadError panel shows the error + a "Back to menu" action.
            load_err.0 = Some(format!("Could not load {name}: {e}"));
        }
    }
}

/// On an in-place map swap (`MapEpoch` bump), re-frame the single reused camera on the new pack and
/// rebuild its skybox. The first observation is skipped ONLY when `setup` already framed this pack in
/// the SYNCHRONOUS load path (detected by the camera having a skybox). On the default ASYNC cold-load
/// path `setup` runs pack-less (menu pose, no skybox), so the first pack observation here must
/// actually frame the map + INSERT the skybox — otherwise the map opens at the menu vantage over a
/// flat grey backdrop. Menu mode (no pack) is skipped entirely.
fn reset_map_view(
    mut commands: Commands,
    pack: Option<Res<LoadedPack>>,
    epoch: Res<render::MapEpoch>,
    mut images: ResMut<Assets<Image>>,
    mut cam: Query<
        (
            Entity,
            &mut Transform,
            &mut Projection,
            &mut FlyCam,
            &mut walk_ground::WalkState,
            Option<&mut Skybox>,
        ),
        With<CullCamera>,
    >,
    mut last: Local<Option<u64>>,
) {
    let Some(pack) = pack else { return };
    let cur = epoch.0;
    if *last == Some(cur) {
        return;
    }
    let was_first = last.is_none();
    *last = Some(cur);
    let Ok((cam_entity, mut tf, mut proj, mut fly, mut walk, skybox)) = cam.single_mut() else {
        return;
    };
    // Skip only if `setup` already framed this pack AND built its skybox (sync path). On the async
    // cold-load path the camera has NO skybox yet, so fall through to frame + insert it.
    if was_first && skybox.is_some() {
        return;
    }
    // Debug overrides (EFT_POSE / EFT_LOOK) pin the camera in `setup`; don't clobber them with the
    // content-anchor reframe here (the skybox insert below still runs so the sky is correct).
    if std::env::var("EFT_POSE").is_err() && std::env::var("EFT_LOOK").is_err() {
        let (cam_pos, _target, far, yaw, pitch) = frame_for_pack(Some(&pack.0));
        tf.translation = cam_pos;
        tf.rotation = Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, pitch);
        fly.yaw = yaw;
        fly.pitch = pitch;
        if let Projection::Perspective(pp) = &mut *proj {
            pp.far = far;
        }
    }
    // Drop stale ground/velocity from the old map (else the fell-through-world backstop can teleport
    // the player to a nonexistent old-map Y, and a mid-jump vy carries over).
    *walk = walk_ground::WalkState::default();
    // Rebuild the skybox for the new sun. SWAP an existing cubemap (in-place swap / sync path, frees
    // the old image so it doesn't leak each swap) or INSERT one when the camera has none yet (the
    // async cold-load first frame — same params as `setup`'s insert).
    let (sun_dir, _) = pack_sun_dir(Some(&pack.0));
    let new_sky = build_sky_cubemap(&mut images, sun_dir);
    match skybox {
        Some(mut sb) => {
            let old = sb.image.clone();
            sb.image = new_sky;
            images.remove(&old);
        }
        None => {
            commands.entity(cam_entity).insert(Skybox {
                image: new_sky,
                brightness: 900.0,
                rotation: Quat::IDENTITY,
            });
        }
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
                    exposure: 0.0,
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
    // Invert `Ry(yaw)·Rx(pitch)` (which builds forward = (-cos p·sin yaw, sin p, -cos p·cos yaw)):
    // yaw = atan2(-dir.x, -dir.z). The old atan2(dir.x, -dir.z) was the negated yaw, so `fly_to`
    // looked at the X-mirror of the target. Now it faces the target (and matches EFT_POSE/pos_hud).
    cam.yaw = (-dir.x).atan2(-dir.z);
    cam.pitch = dir.y.asin();
    tf.translation = cam_pos;
    tf.rotation = Quat::from_axis_angle(Vec3::Y, cam.yaw) * Quat::from_axis_angle(Vec3::X, cam.pitch);
}

fn main() {
    // Headless nav baker BEFORE any Bevy/GPU init: `atlas bake-nav <pack_dir> [--res R] [--layers K]`
    // bakes the routing grid on the CPU (portable — AMD/NVIDIA/no-GPU) and exits, so the map-build
    // pipeline can produce routing on any machine without CUDA. No window, no adapter.
    {
        let argv: Vec<String> = std::env::args().collect();
        if argv.get(1).map(String::as_str) == Some("bake-nav") {
            std::process::exit(nav_bake::run_cli(&argv[2..]));
        }
        // Headless PORTABLE lighting bake (CPU rayon, any GPU / none) — `atlas bake-sh <pack_dir>`
        // bakes the SH irradiance volume without CUDA/warp, so AMD/Intel builds get real baked
        // lighting instead of the flat realtime fallback. Replaces bake_volume2.py in the pipeline.
        if argv.get(1).map(String::as_str) == Some("bake-sh") {
            std::process::exit(sh_bake::run_cli(&argv[2..]));
        }
    }
    // --version/--help fast path BEFORE any Bevy/GPU init: CI runners have no usable GPU, so
    // this is the only smoke test a workflow can run (redistribution PR5).
    if let Some(flag) = std::env::args().nth(1) {
        if matches!(flag.as_str(), "--version" | "-V") {
            println!("atlas {} ({})", env!("CARGO_PKG_VERSION"), env!("CARGO_PKG_NAME"));
            return;
        }
        if matches!(flag.as_str(), "--help" | "-h") {
            println!(
                "atlas [<pack-dir>] [m0|gpu]\n\
                 atlas bake-nav <pack-dir> [--res 1.0] [--layers 8]  (headless CPU nav baker)\n\
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
    // bare `atlas` with no arguments opens a map instead of an empty window.
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
    // COLD-LOAD LOADING SCREEN: when a pack is given, DON'T load it synchronously here — that blocks
    // in main() before the window ever paints, so a big map (~60k instances) shows a FROZEN window
    // for the whole load. Instead start in a "loading" mode (no pack yet, but NOT the menu) and hand
    // the pack to the SAME async MapSwitch -> load_map -> PendingMapLoad -> poll_map_load path that
    // in-place swaps use: the window opens immediately, the loading indicator animates, and
    // Pack::load runs off-thread. Only the GPU-driven path has that epoch-aware async rebuild; the
    // m0/std paths spawn geometry once at Startup, so they keep loading synchronously.
    // EFT_SYNC_LOAD=1 forces the old blocking load.
    let async_cold_load = pack_dir.is_some()
        && render_path == RenderPath::GpuDriven
        && !std::env::var("EFT_SYNC_LOAD").map(|v| v.trim() == "1").unwrap_or(false);
    let pack = if async_cold_load {
        eprintln!(
            "cold load: '{}' loads async behind a loading screen (EFT_SYNC_LOAD=1 to disable)",
            pack_dir.as_deref().unwrap_or("")
        );
        None
    } else if let Some(dir) = &pack_dir {
        match Pack::load(dir) {
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
        }
    } else {
        eprintln!("no pack given — opening the start menu.  direct: atlas <pack-dir>");
        None
    };
    // Menu = a bare launch (no pack arg) or a failed synchronous load. A cold-loading map is NOT the
    // menu — it renders the loading screen while the pack streams in via the async path.
    let menu_mode = pack.is_none() && !async_cold_load;

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
                    title: "Atlas".into(),
                    resolution: (1600u32, 1000u32).into(),
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
                // Shipped bundle: assets/ sits beside the exe (portability PR1). If it's not there,
                // an ATLAS_ASSETS_DIR env var wins (escape hatch for running a release build out of
                // the cargo target dir). Only a DEBUG build falls back to the compile-time crate dir
                // for shader hot-editing — a RELEASE build must NOT: env!("CARGO_MANIFEST_DIR") bakes
                // the build machine's home path (leaking the builder's username) into the exe and
                // never exists on a user's PC. In release we point at the expected <exe>/assets so a
                // missing-shader error makes the "keep assets next to atlas.exe" rule obvious.
                file_path: {
                    let exe_assets = paths::exe_dir().join("assets");
                    if exe_assets.is_dir() {
                        exe_assets.to_string_lossy().into_owned()
                    } else if let Ok(dir) = std::env::var("ATLAS_ASSETS_DIR") {
                        dir
                    } else {
                        #[cfg(debug_assertions)]
                        {
                            concat!(env!("CARGO_MANIFEST_DIR"), "/assets").to_string()
                        }
                        #[cfg(not(debug_assertions))]
                        {
                            exe_assets.to_string_lossy().into_owned()
                        }
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
    // Menu backdrop: crank Bloom so the neon globe reads as a hazy VOLUMETRIC glow (in-raid keeps
    // the subtle 0.06). apply_gfx_camera pushes this to the camera.
    if menu_mode {
        gfx.bloom_intensity = 0.32;
    }
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
    // In-place map-swap epoch: bumped by `load_map` on each .eftpack swap; extracted to the render
    // world and used as the run_if gate for every per-map (re)build system. Inserted always (menu
    // mode too) so `build_cpu_data`'s `run_if(resource_changed::<MapEpoch>)` fires on the first frame.
    app.insert_resource(render::MapEpoch(0));
    // The active render path: load_map only swaps IN-PLACE under GPU-driven (the only path with an
    // epoch-aware rebuild); m0/std spawn geometry once at Startup, so they must relaunch on a switch.
    app.insert_resource(render_path);
    // UI language (EN/RU): saved override in atlas.config.json > system locale > English. The menu
    // language toggle flips + persists it; egui re-renders the whole UI next frame.
    app.insert_resource(i18n::detect_lang(menu::config_lang().as_deref()));

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
        .add_plugins(progress::ProgressPlugin) // persistent tracked tasks, objectives, and owned keys
        .add_plugins(tasks_panel::TasksPanelPlugin) // revamped Tasks tab: catalog + icon cache (router calls tasks_panel_ui)
        .add_plugins(pathfind::PathfindPlugin) // in-process CPU routing over the baked nav grid (nav.rs)
        .add_plugins(planner::PlannerPlugin) // loot-run orienteering planner (Navigation tab)
        .add_plugins(jobs::JobsPlugin) // background job worker: build/sync maps while a map is open
        .init_resource::<CameraCommand>() // UI-driven "fly the camera to X" (search / quest jump / route)
        .init_resource::<CameraSettings>() // camera-tab: FOV / fly speed / walk mode
        .init_resource::<MapSwitch>() // UI map dropdown -> switch to the selected pack (in place)
        .init_resource::<ReturnToMenu>() // toolbar "back to menu" button -> relaunch into the menu
        .init_resource::<PendingMapLoad>() // async in-place pack load (no frame freeze on switch)
        .init_resource::<MapLoadError>() // async load failure -> UI error + back-to-menu (finding 4)
        .init_resource::<ForcedLod>() // graphics-panel LOD selector (meaningful on --alllod packs)
        .add_systems(Startup, setup)
        // walk_move runs AFTER flycam_look (orientation resolved) and flycam_move (mutually
        // exclusive by walk_mode) so they can't race the shared Transform. Disabled in the MENU
        // (MenuState present): the backdrop camera stays locked to its composed pose — no WASD /
        // RMB-look / cursor-grab — while the scene itself drifts under the cursor (menu_city_update).
        .add_systems(
            Update,
            (cursor_grab, flycam_look, flycam_move, walk_move)
                .chain()
                .run_if(not(resource_exists::<menu::MenuState>)),
        )
        .add_systems(Update, (apply_camera_command, auto_screenshot, debug_switch, return_to_menu, bump_epoch_on_lod_change))
        .add_systems(
            Update,
            (apply_gfx_camera, load_map, poll_map_load, clear_map_error_on_new_load, flycam_scroll, apply_camera_fov, build_walk_ground),
        )
        // In-place map swap: re-frame the reused camera + rebuild the skybox on a MapEpoch bump.
        .add_systems(Update, reset_map_view.run_if(resource_changed::<render::MapEpoch>));

    #[cfg(feature = "egui")]
    {
        // NOTE: bevy_egui's plugin ctor / context-access API drift between point
        // releases; adjust these two lines if they don't match your bevy_egui 0.37.x.
        // egui UI runs in EguiPrimaryContextPass, not Update (else ctx_mut() panics: no fonts).
        app.add_plugins(bevy_egui::EguiPlugin::default());
        if menu_mode {
            app.add_systems(bevy_egui::EguiPrimaryContextPass, menu::menu_ui);
        }
    }
    // Start menu (bare launch): scan packs/, fingerprint the game install, present the map
    // manager. The in-raid panels check for this resource and stand down while it exists.
    // Menu backdrop = the INTERCHANGE-INSPIRED NEON WIREFRAME EXFIL: a stylized, derivative low-poly
    // schematic (razor-wire overpass + pillars, rail line with boxcars, receding power pylons,
    // containers, gantry crane) in the 3D world that the camera's Bloom halos into a glowing hologram,
    // with idle drift + cursor parallax (menu_fx::spawn_menu_scene / update). Fully synthetic (no game
    // geometry) so it ships with the app. The CentralPanel goes transparent (menu.rs) so it shows
    // behind the UI. EFT_MENU_TERRAIN=1 falls back to the old rippling triangle terrain.
    if menu_mode {
        app.insert_resource(menu::build_state());
        // Menu-only GitHub update check: fires one token-less GET off-thread on startup and, if a
        // newer release exists, drives the top-right version indicator + the themed update modal in
        // menu::menu_ui. Offline-safe (folds to Unknown) and never blocks the first frame.
        app.add_plugins(update::UpdatePlugin);
        // Default backdrop is the 2D reactive triangle field, painted in egui (menu::menu_ui ->
        // menu_fx::triangle_field) — no 3D world needed. The 3D backdrops are opt-in fallbacks:
        // EFT_MENU_EXFIL=1 = neon wireframe exfil, EFT_MENU_TERRAIN=1 = rippling triangle terrain.
        if std::env::var("EFT_MENU_TERRAIN").map(|v| v.trim() == "1").unwrap_or(false) {
            app.add_systems(Startup, menu_fx::spawn_menu_terrain.after(setup));
            app.add_systems(Update, menu_fx::menu_terrain_update);
        } else if std::env::var("EFT_MENU_EXFIL").map(|v| v.trim() == "1").unwrap_or(false) {
            app.add_systems(Startup, menu_fx::spawn_menu_scene.after(setup));
            app.add_systems(Update, menu_fx::menu_scene_update);
        }
    }

    // Cold-load kick-off: seed MapSwitch so load_map (frame 1) starts the async pack load down the
    // same path as an in-place swap — the window is already up rendering the loading screen while
    // Pack::load runs off-thread, instead of main() blocking before the first frame.
    if async_cold_load {
        app.insert_resource(MapSwitch(pack_dir));
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

/// Framing for a pack (or a sensible default when none): `(cam_pos, target, far, yaw, pitch)`.
/// Honors the EFT_LOOK (frame close on a world point) and EFT_POSE (exact pose) debug overrides.
/// Shared by the initial `setup` spawn and the in-place `reset_map_view` swap path.
fn frame_for_pack(pack: Option<&crate::eftpack::Pack>) -> (Vec3, Vec3, f32, f32, f32) {
    // EFT_LOOK="x,y,z" frames the camera CLOSE on that world point (a picked coordinate) instead of
    // the whole-map overview — to confirm a specific mesh renders where the data says it is.
    let look_override = std::env::var("EFT_LOOK").ok().and_then(|s| {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        (p.len() == 3).then(|| Vec3::new(p[0], p[1], p[2]))
    });
    let (target, mut cam_pos, far) = if let Some(t) = look_override {
        (t, t + Vec3::new(4.0, 6.0, 14.0), 4000.0)
    } else {
        match pack {
            Some(p) => {
                // Open NEAR the map's content (median instance position), not a whole-map overview:
                // consistent across ALL maps — small maps open close, big maps pull back — always
                // looking at populated geometry, never the empty AABB center out over the sea.
                let anchor = p.content_anchor();
                let ext = p.bounds_extent().max(1.0);
                let d = (ext * 0.10).clamp(30.0, 90.0);
                (anchor, anchor + Vec3::new(0.0, d * 0.5, d), (ext * 6.0).max(2000.0))
            }
            // Menu: pose is set explicitly just below (the target here only feeds `far`).
            None => (Vec3::ZERO, Vec3::new(140.0, 56.0, 150.0), 4000.0),
        }
    };
    let dir = (target - cam_pos).normalize_or_zero();
    let mut yaw = dir.x.atan2(-dir.z);
    let mut pitch = dir.y.asin();
    // Menu backdrop: a hand-picked elevated 3/4 vantage over the neon wireframe exfil scene
    // (menu_fx::spawn_menu_scene), ~-17deg pitch to echo the in-game Interchange railway/overpass pose
    // it's derived from. yaw/pitch are set DIRECTLY because the target->yaw derivation above only aims
    // correctly when dir.x==0 (every in-raid framing offsets in the YZ-plane); an off-axis menu camera
    // needs explicit angles. EFT_POSE below still overrides, for live tuning.
    if pack.is_none() {
        cam_pos = Vec3::new(60.0, 64.0, 155.0);
        yaw = 22.0_f32.to_radians();
        pitch = (-17.0_f32).to_radians();
    }
    // EFT_POSE="x,y,z,yaw_deg,pitch_deg" reproduces an EXACT camera pose (the POS HUD's copy button).
    if let Ok(s) = std::env::var("EFT_POSE") {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() == 5 {
            cam_pos = Vec3::new(p[0], p[1], p[2]);
            yaw = p[3].to_radians();
            pitch = p[4].to_radians();
        }
    }
    (cam_pos, target, far, yaw, pitch)
}

/// The pack's baked sun direction (viewer-space; the bake already conjugates it) + whether it was
/// found (gates real-time shadows). Falls back to a plausible high overcast sun.
fn pack_sun_dir(pack: Option<&crate::eftpack::Pack>) -> (Vec3, bool) {
    let from_pack = pack
        .and_then(|p| p.manifest.sidecars.volume_meta.as_deref().map(|m| p.resolve_path(m)))
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|txt| serde_json::from_str::<serde_json::Value>(&txt).ok())
        .and_then(|v| {
            v.get("sun_dir").and_then(|s| s.as_array()).and_then(|a| {
                let raw = Vec3::new(
                    a.first()?.as_f64()? as f32,
                    a.get(1)?.as_f64()? as f32,
                    a.get(2)?.as_f64()? as f32,
                );
                (raw.length_squared() > 1e-6).then(|| raw.normalize())
            })
        });
    let ok = from_pack.is_some();
    (from_pack.unwrap_or_else(|| Vec3::new(-0.45, 0.8, -0.4).normalize()), ok)
}

/// Spawn camera + a key light, framed on the pack bounds if one is loaded.
fn setup(
    mut commands: Commands,
    pack: Option<Res<LoadedPack>>,
    mut images: ResMut<Assets<Image>>,
    grade: Option<Res<GradeLutCpu>>,
    mut gfx: ResMut<render::GfxSettings>,
) {
    let (cam_pos, target, far, yaw, pitch) = frame_for_pack(pack.as_ref().map(|p| &p.0));

    // Sky sun direction from the pack's volume sidecar (same one the SH/shadow path uses, so the
    // skybox sun disk, baked GI and reflected sun agree). The experimental shadow toggle needs a
    // real sun (matches the render side's sun_ok gate).
    let (sun_dir, sun_ok) = pack_sun_dir(pack.as_ref().map(|p| &p.0));
    gfx.shadows_available = sun_ok;
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
        // Build rotation from yaw/pitch (the FlyCam convention) so it matches FlyCam.{yaw,pitch}
        // exactly — for the normal path these derive from `target`, and EFT_POSE overrides them.
        Transform::from_translation(cam_pos)
            .with_rotation(Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, pitch)),
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
        walk_ground::WalkState::default(), // per-camera walk locomotion state (inert until walk mode)
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
                    exposure: 0.0,
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

    // SKY: attach the procedural overcast cubemap (horizon->zenith gradient + soft HDR sun disk,
    // matching sky_reflect() so reflections agree with the visible sky). REGRESSION FIX — the
    // menu-CCTV change (commit 51c5cea) dropped this insert entirely, leaving `sky` a dead binding
    // and every outdoor map rendering the flat ClearColor as "sky" (no gradient, no sun for Bloom,
    // fog/horizon mismatch). Skipped in menu mode (sky is None there). brightness 900 nits maps a
    // cubemap value of 1.0 to ~0.9 render radiance under the default camera Exposure — the grade LUT
    // then remaps sky + scene identically, so relative brightness is preserved.
    if let Some(image) = sky {
        cam.insert(Skybox {
            image,
            brightness: 900.0,
            rotation: Quat::IDENTITY,
        });
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
    settings: Res<CameraSettings>,
    mut q: Query<(&mut Transform, &FlyCam)>,
) {
    // Typing 'wasd' into the marker-search box must not fly the camera (Codex review).
    // In walk mode, walk_move owns locomotion — fly is inert.
    if ui_kb.0 || settings.walk_mode {
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
            // Base speed comes from the camera-tab setting (scroll-wheel adjustable), not the
            // fixed FlyCam::speed; shift still boosts.
            let mut speed = settings.fly_speed;
            if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                speed *= cam.boost;
            }
            tf.translation += v.normalize() * speed * dt;
        }
    }
}

/// Lazily build the walk-mode ground grid the first time walk mode is enabled (fly-only users
/// never pay the ~250-400 MB + build cost). One-shot: skips once the resource exists.
fn build_walk_ground(
    mut commands: Commands,
    settings: Res<CameraSettings>,
    grid: Option<Res<walk_ground::GroundGrid>>,
    pack: Option<Res<LoadedPack>>,
) {
    if !settings.walk_mode || grid.is_some() {
        return;
    }
    let Some(pack) = pack else { return };
    info!("walk_ground: building walkable-surface grid (first walk-mode activation)…");
    commands.insert_resource(walk_ground::GroundGrid::build(&pack.0));
}

/// Walk locomotion: yaw-only WASD on the ground plane at `walk_speed`, ground-follow (stairs glide),
/// gravity + Space jump (jump height rides off walk_speed). Gated on `walk_mode`; fly is inert then.
fn walk_move(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    ui_kb: Res<inspect::UiWantsKeyboard>,
    settings: Res<CameraSettings>,
    grid: Option<Res<walk_ground::GroundGrid>>,
    mut q: Query<(&mut Transform, &FlyCam, &mut walk_ground::WalkState), With<CullCamera>>,
) {
    use walk_ground::{EYE_HEIGHT, GRAVITY, KILL_DROP, STEP_UP};
    if !settings.walk_mode {
        return;
    }
    let Some(grid) = grid else { return }; // still building
    let dt = time.delta_secs().min(0.05); // clamp big frame gaps so jumps don't over-integrate
    let typing = ui_kb.0;
    for (mut tf, cam, mut ws) in &mut q {
        // Undo last frame's cosmetic head-bob so every physics query below runs on the TRUE eye
        // height (the bob must never feed back into ground/step selection).
        tf.translation.y -= ws.last_bob;
        // Horizontal: yaw-only (looking up/down must not change ground speed). Forward/right from
        // the FlyCam yaw, flattened onto XZ (matches Quat::from_axis_angle(Y, yaw)).
        let (s, c) = cam.yaw.sin_cos();
        let fwd = Vec3::new(-s, 0.0, -c);
        let right = Vec3::new(c, 0.0, -s);
        let mut h = Vec3::ZERO;
        if !typing {
            if keys.pressed(KeyCode::KeyW) { h += fwd; }
            if keys.pressed(KeyCode::KeyS) { h -= fwd; }
            if keys.pressed(KeyCode::KeyD) { h += right; }
            if keys.pressed(KeyCode::KeyA) { h -= right; }
        }
        let mut moved = 0.0f32; // horizontal distance walked this frame (drives the head-bob)
        if h != Vec3::ZERO {
            let mut spd = settings.walk_speed;
            if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                spd *= 1.8; // sprint
            }
            moved = spd * dt;
            tf.translation += h.normalize() * spd * dt;
            // Player-sized collision: push the body capsule back out of any wall it entered so you
            // can't run through walls/fences. Purely horizontal (feet height is unchanged here).
            let feet_y = tf.translation.y - EYE_HEIGHT;
            let fixed = grid.resolve_walls(tf.translation, feet_y);
            tf.translation.x = fixed.x;
            tf.translation.z = fixed.y;
        }

        // Jump (behind the typing guard so Space in a text field doesn't launch).
        if !typing && keys.just_pressed(KeyCode::Space) && ws.grounded {
            ws.vy = walk_ground::jump_velocity(settings.walk_speed);
            ws.grounded = false;
        }

        // Vertical integration + ground resolve.
        let (x, z) = (tf.translation.x, tf.translation.z);
        let feet_y = tf.translation.y - EYE_HEIGHT;
        let ground = grid.ground_height(x, z, feet_y, STEP_UP);
        ws.vy -= GRAVITY * dt;
        let mut new_y = tf.translation.y + ws.vy * dt;
        match ground {
            Some(g) => {
                let target = g + EYE_HEIGHT;
                ws.last_ground_y = g;
                ws.has_ground = true;
                if ws.vy <= 0.0 && new_y <= target {
                    // Land / stand: settle exactly, and while grounded exp-smooth toward the
                    // surface so stepping up curbs/treads glides instead of snapping.
                    let follow = 1.0 - (-20.0 * dt).exp();
                    new_y = tf.translation.y + (target - tf.translation.y) * follow;
                    // Snap the last little bit to avoid perpetual approach.
                    if (new_y - target).abs() < 0.01 {
                        new_y = target;
                    }
                    ws.vy = 0.0;
                    ws.grounded = true;
                } else {
                    ws.grounded = false; // airborne (rising, or above target)
                }
            }
            None => {
                // Void under the feet: keep falling. Fell-through-world backstop -> snap back.
                ws.grounded = false;
                if ws.has_ground && new_y < ws.last_ground_y - KILL_DROP {
                    new_y = ws.last_ground_y + EYE_HEIGHT;
                    ws.vy = 0.0;
                    ws.grounded = true;
                }
            }
        }
        tf.translation.y = new_y;

        // Head bob: a subtle vertical sine advanced by distance walked while grounded; eases back to
        // zero when you stop. Applied ON TOP of the settled eye height and removed at the top of the
        // next frame, so it never perturbs the ground/step physics.
        let new_bob = if ws.grounded && moved > 0.0 {
            ws.bob_phase += moved * walk_ground::BOB_RATE;
            walk_ground::BOB_AMP * ws.bob_phase.sin()
        } else {
            ws.last_bob * (-8.0 * dt).exp()
        };
        tf.translation.y += new_bob;
        ws.last_bob = new_bob;
    }
}

/// Reliable frame capture: with `EFT_SHOT=<path>` set, save ONE screenshot of the primary
/// window ~90 Update-frames in (a beat after the heavy Startup finishes) via Bevy's own GPU
/// screenshot — this bypasses the DWM/flip-model capture that makes external CopyFromScreen
/// grab a blank white frame.
fn auto_screenshot(mut commands: Commands, mut frames: Local<u32>) {
    *frames += 1;
    // EFT_SHOT_FRAME overrides the default frame-90 capture (later frames let a scripted in-place
    // map swap settle before the shot — see debug_switch).
    let target = std::env::var("EFT_SHOT_FRAME").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(90);
    if *frames != target {
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

/// Headless soak-test hook for the in-place map swap: `EFT_SWITCH="dir@frame;dir@frame;..."` fires
/// each `MapSwitch` at its frame (relative to Update start), so an A->B->A swap can be exercised +
/// screenshot without clicking. e.g. `EFT_SWITCH="packs/factory.eftpack@150"`.
fn debug_switch(mut sw: ResMut<MapSwitch>, mut frames: Local<u32>) {
    let Ok(spec) = std::env::var("EFT_SWITCH") else {
        return;
    };
    *frames += 1;
    for step in spec.split(';') {
        if let Some((dir, at)) = step.rsplit_once('@') {
            if at.trim().parse::<u32>().ok() == Some(*frames) {
                info!("debug_switch: frame {} -> {dir}", *frames);
                sw.0 = Some(dir.trim().to_string());
            }
        }
    }
}
