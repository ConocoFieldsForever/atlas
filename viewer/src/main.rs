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

mod agent_link;
mod assets;
mod character;
mod drone;
mod esp_labels;
mod eftpack;
mod game_watch;
mod gpu_lease;
mod insights;
mod overlay;
mod i18n;
mod inspect;
mod jobs;
mod fx;
mod loot;
mod loot_volume;
mod npc;
mod maps;
mod menu;
mod menu_fx;
mod nav;
mod nav_bake;
mod sh_bake;
mod sh_bake_gpu;
mod terrain_bake;
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
    /// Put the camera EXACTLY here, looking exactly this way -- (eye position, forward), both in
    /// viewer world space. Unlike `fly_to` (which pulls back for context) this is a 1:1 pose, used
    /// by the screenshot position fix to stand in the player's eyes. Takes priority over `fly_to`.
    pub eye: Option<(Vec3, Vec3)>,
}

/// Exact camera pose handed across the menu -> map relaunch by the screenshot trigger.
///
/// `OverlayPlugin` deliberately removes `EFT_POSE` in PostStartup so it cannot leak into a later
/// PLAY relaunch. An async cold load, however, finishes several frames after that cleanup. Keep one
/// parsed copy here until `reset_map_view` observes the first real pack, then consume it exactly once.
#[derive(Clone, Copy, Debug)]
struct ExactCameraPose {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

#[derive(Resource, Default)]
struct PendingStartupPose(Option<ExactCameraPose>);

fn parse_eft_pose(value: &str) -> Option<ExactCameraPose> {
    let parts = value
        .split(',')
        .map(|v| v.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if parts.len() != 5 || parts.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(ExactCameraPose {
        position: Vec3::new(parts[0], parts[1], parts[2]),
        yaw: parts[3].to_radians(),
        pitch: parts[4].to_radians(),
    })
}

/// Which locomotion model owns the camera. `Fly` = free-fly (WASD+QE), `Walk` = FPV on foot
/// (ground-follow + jump + collision), `Drone` = FPV quadcopter (drone.rs physics; also the body
/// the agent link flies).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CamMode {
    #[default]
    Fly,
    Walk,
    Drone,
}

/// Camera-tab settings (the toolbar's camera panel edits these; the flycam systems read them).
/// Decoupled from the private `FlyCam` like `CameraCommand`.
#[derive(Resource)]
pub struct CameraSettings {
    /// Vertical FOV in degrees (applied to the perspective projection).
    pub fov_deg: f32,
    /// Base fly-move speed (m/s); the scroll wheel scales this live.
    pub fly_speed: f32,
    /// Selected WALK speed (m/s); the scroll wheel changes EFT's normalized walking-speed dial.
    /// Sprint and jump remain independent of that dial.
    pub walk_speed: f32,
    /// Camera locomotion mode (Fly / Walk / Drone).
    pub mode: CamMode,
    /// FPV camera uptilt in drone mode (deg) — real FPV quads mount the camera tilted up so the
    /// horizon is visible while pitched forward at speed. Scroll adjusts it live in drone mode.
    pub drone_cam_tilt_deg: f32,
    /// Drone manual flight: true = ACRO (rates + positional throttle — the real FPV deal),
    /// false = ANGLE (self-leveling + altitude assist — trainer wheels).
    pub drone_acro: bool,
    /// Betaflight-style rate profile for manual acro (RC rate / expo / super rate).
    pub drone_rc_rate: f32,
    pub drone_expo: f32,
    pub drone_super_rate: f32,
    /// Analog FPV video-link effect master gain (0 = clean digital picture).
    pub fpv_noise: f32,
    /// Video-link usable range (m): RSSI hits zero around here in open air; walls shorten it.
    pub fpv_range: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        // EFT_CAM=fly|walk|drone picks the start mode; EFT_WALK=1 kept as the legacy walk alias.
        let mode = match std::env::var("EFT_CAM").as_deref().map(str::trim) {
            Ok("walk") => CamMode::Walk,
            Ok("drone") => CamMode::Drone,
            Ok(_) => CamMode::Fly,
            Err(_) if std::env::var("EFT_WALK").map(|v| v.trim() == "1").unwrap_or(false) => {
                CamMode::Walk
            }
            Err(_) => CamMode::Fly,
        };
        Self {
            // A screenshot-triggered menu -> map relaunch carries the FOV that the menu process
            // already parsed from Tarkov's settings log. This makes the very first loaded frame use
            // the game's projection instead of waiting for the child watcher to re-tail the log.
            fov_deg: std::env::var("EFT_GAME_FOV")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| (20.0..=120.0).contains(v))
                .unwrap_or(60.0),
            fly_speed: 40.0, // matches the old FlyCam::default speed
            walk_speed: 5.0, // human-ish
            mode,
            drone_cam_tilt_deg: 18.0,
            drone_acro: true,
            drone_rc_rate: 1.0,
            drone_expo: 0.25,
            drone_super_rate: 0.7,
            fpv_noise: 0.7,
            fpv_range: 350.0,
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
    match settings.mode {
        CamMode::Walk => {
            // EFT's wheel controls the normalized walking-speed dial. It does not alter sprint
            // top speed or ballistic jump height.
            settings.walk_speed = (settings.walk_speed * factor).clamp(
                walk_ground::MIN_WALK_SPEED,
                walk_ground::MAX_WALK_SPEED,
            );
        }
        CamMode::Drone => {
            // In drone mode the wheel adjusts the FPV camera uptilt (like re-mounting the cam).
            settings.drone_cam_tilt_deg =
                (settings.drone_cam_tilt_deg + scroll.delta.y * 2.0).clamp(0.0, 45.0);
        }
        CamMode::Fly => {
            settings.fly_speed = (settings.fly_speed * factor).clamp(2.0, 4000.0);
        }
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
/// Draw the geometry Unity has SWITCHED OFF (`eftpack::flags::INACTIVE`) — parked scenery and the
/// interiors of unreleased rooms, which the build ships flagged but the renderer hides so the view
/// matches what the game draws. OFF by default. Entirely separate from `LayerToggles::hide_inactive`,
/// which filters gamedata MARKERS; this one is about geometry.
#[derive(Resource)]
pub struct ShowDisabledGeom(pub bool);
impl Default for ShowDisabledGeom {
    /// `EFT_SHOW_DISABLED=1` starts with it on, matching the `EFT_LOD` / `EFT_LAYERS=showinactive`
    /// debug-override style. Without this the toggle is only reachable by clicking, so a headless
    /// capture cannot exercise it and the flag's end-to-end behaviour could not be verified.
    fn default() -> Self {
        ShowDisabledGeom(std::env::var("EFT_SHOW_DISABLED").map(|v| v.trim() == "1").unwrap_or(false))
    }
}

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
/// Same re-trigger as `bump_epoch_on_lod_change`, for the disabled-geometry toggle. Guarded the
/// same way: a `MapEpoch` bump is destructive (reframes the camera, clears nav/pins/routes), so only
/// bump when the pack ACTUALLY ships flagged geometry — on every pack built before this flag existed
/// the toggle is a true no-op and must not nuke that state.
///
/// Compares the VALUE, never `is_changed()`. egui draws the checkbox with
/// `ui.checkbox(&mut res.0, ..)`, and `ResMut`'s `deref_mut` sets the changed flag on every frame
/// the panel is open whether or not the bool moved — so an `is_changed()` guard here bumped the
/// epoch every frame and rebuilt the 4.6 s CPU blob in a loop (39 rebuilds in one session; the map
/// never finished loading). A `Local` copy of the last applied value is the only reliable edge.
fn bump_epoch_on_disabled_geom_change(
    show: Res<ShowDisabledGeom>,
    pack: Option<Res<LoadedPack>>,
    mut epoch: ResMut<render::MapEpoch>,
    mut last: Local<Option<bool>>,
) {
    let cur = show.0;
    if last.is_none() {
        *last = Some(cur); // first observation: adopt, never bump
        return;
    }
    if *last == Some(cur) {
        return; // no real edge — the flag was just touched by the UI's &mut
    }
    *last = Some(cur);
    let has_any = pack
        .as_ref()
        .map(|p| p.0.instances.iter().any(|i| i.is_inactive()))
        .unwrap_or(false);
    if has_any {
        epoch.0 = epoch.0.wrapping_add(1);
    }
}

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
/// ESP / overlay-only mode: draw the markers, not the map.
///
/// It is a LOADER decision, not a render toggle. `Pack::load_tier(_, Markers)` skips meshes.bin,
/// instances.bin and materials.json, and every downstream stage already bails on an empty mesh
/// table -- so the 708 MB that is not read and the world that is not drawn are the same change,
/// rather than a render path that has to be kept in agreement with a load path.
#[derive(Resource, Clone, Copy)]
pub struct EspMode(pub bool);

/// Whether THIS window was created transparent. A launch fact, not a setting: the overlay's
/// dismiss/exit paths consult it because re-decorating a transparent-created window breaks the
/// DWM conjunction it can never get back (see OverlayPresentation) -- those paths minimize or
/// hide instead when this is set.
#[derive(Resource, Clone, Copy)]
pub struct TransparentWindow(pub bool);

fn load_map(
    mut sw: ResMut<MapSwitch>,
    esp: Res<EspMode>,
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
    // We are becoming an interactive MAP view now, so claim the GPU from here on: a bake started
    // after this point must see us and stay off the adapter. Startup deliberately does NOT take the
    // lease in menu mode (see main()), and PLAY is an in-place switch rather than a relaunch — so
    // without this the process would render a map while still advertising the GPU as free.
    // Idempotent, so a second switch is a no-op.
    crate::gpu_lease::hold("map loaded");
    let want_tier = if esp.0 { eftpack::PackTier::Markers } else { eftpack::PackTier::Full };
    let task = bevy::tasks::AsyncComputeTaskPool::get()
        .spawn(async move { Pack::load_tier(&dir, want_tier).map_err(|e| format!("{e:#}")) });
    pending.0 = Some((name, task));
}

/// Clear a stale `MapLoadError` the moment a NEW async load is kicked off, so the error toast from a
/// previous failed attempt doesn't linger over a fresh (possibly succeeding) load.
fn clear_map_error_on_new_load(
    pending: Res<PendingMapLoad>,
    mut err: ResMut<MapLoadError>,
    gpu_load: Option<Res<render::GpuLoadSignal>>,
) {
    if pending.is_changed() && pending.loading().is_some() {
        err.0 = None;
        if let Some(signal) = gpu_load {
            signal.clear_error();
        }
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
            // A Markers pack has no world, so every loot container that would have GLOWED ITS
            // MODEL now has no model to glow: LootModelIndex is insert-only (nothing in the tree
            // removes it), so toggling ESP on a session that already loaded a Full pack leaves the
            // index resident, every container takes the glow branch, gets no Mesh3d, and is drawn
            // by a world pass that no longer exists. The loot layer would be silently empty --
            // which in ESP mode is indistinguishable from "this map has no loot".
            if p.tier == eftpack::PackTier::Markers {
                commands.remove_resource::<crate::loot::LootModelIndex>();
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
            commands.insert_resource(LoadedPack(std::sync::Arc::new(p)));
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
    esp: Res<EspMode>,
    pack: Option<Res<LoadedPack>>,
    epoch: Res<render::MapEpoch>,
    mut images: ResMut<Assets<Image>>,
    mut startup_pose: ResMut<PendingStartupPose>,
    mut cam: Query<
        (
            Entity,
            &mut Transform,
            &mut Projection,
            &mut FlyCam,
            &mut walk_ground::WalkState,
            &mut drone::DroneRig,
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
    let Ok((cam_entity, mut tf, mut proj, mut fly, mut walk, mut rig, skybox)) = cam.single_mut()
    else {
        return;
    };
    // Skip only if `setup` already framed this pack AND built its skybox (sync path). On the async
    // cold-load path the camera has NO skybox yet, so fall through to frame + insert it.
    // Take the relaunch pose on the first real pack observation even on the synchronous path, where
    // `setup` already applied it and the skybox early-return below is correct. That keeps it one-shot
    // and prevents a later in-place map switch from inheriting the old screenshot location.
    let startup_pose = startup_pose.0.take();
    if was_first && skybox.is_some() {
        return;
    }
    // The screenshot handoff must win AFTER an async cold load becomes active. PostStartup has
    // already removed EFT_POSE by now, so consulting only the environment here used to reframe the
    // camera to the map overview and lose the user's position.
    if let Some(pose) = startup_pose {
        tf.translation = pose.position;
        tf.rotation =
            Quat::from_axis_angle(Vec3::Y, pose.yaw) * Quat::from_axis_angle(Vec3::X, pose.pitch);
        fly.yaw = pose.yaw;
        fly.pitch = pose.pitch;
    // EFT_LOOK remains a persistent debug override. A direct EFT_POSE launch also keeps its env var
    // (only an overlay summon removes it), preserving the existing "pin across map resets" behavior.
    } else if std::env::var("EFT_POSE").is_err() && std::env::var("EFT_LOOK").is_err() {
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
    // the player to a nonexistent old-map Y, and a mid-jump vy carries over). Same for the drone —
    // it respawns at the new camera pose on the next drone-mode frame.
    *walk = walk_ground::WalkState::default();
    *rig = drone::DroneRig::default();
    // Rebuild the skybox for the new sun. SWAP an existing cubemap (in-place swap / sync path, frees
    // the old image so it doesn't leak each swap) or INSERT one when the camera has none yet (the
    // async cold-load first frame — same params as `setup`'s insert).
    let (sun_dir, _) = pack_sun_dir(Some(&pack.0));
    // Toggling ESP on a session that already loaded a Full map leaves the old skybox on the
    // camera; nothing else removes it, and it would keep painting the frame.
    if esp.0 {
        if let Some(sb) = skybox {
            let old = sb.image.clone();
            images.remove(&old);
        }
        commands.entity(cam_entity).remove::<Skybox>();
        return;
    }
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
    // Photoreal extras (Graphics panel): plain Bevy camera post components, so they work on every
    // render path. DoF reads the main-pass depth (the GPU-driven pass writes the standard
    // ViewDepthTexture); chromatic aberration is a pure color post.
    if gfx.dof {
        ec.insert(bevy::post_process::dof::DepthOfField {
            mode: bevy::post_process::dof::DepthOfFieldMode::Bokeh,
            focal_distance: gfx.dof_focal_m,
            aperture_f_stops: gfx.dof_fstop,
            ..default()
        });
    } else {
        ec.remove::<bevy::post_process::dof::DepthOfField>();
    }
    if gfx.chroma > 0.0005 {
        ec.insert(bevy::post_process::effect_stack::ChromaticAberration {
            intensity: gfx.chroma,
            ..default()
        });
    } else {
        ec.remove::<bevy::post_process::effect_stack::ChromaticAberration>();
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
    // Read-only fast path: a take() through DerefMut would dirty change detection every frame.
    // BOTH commands must be checked — gating on `fly_to` alone made the screenshot-to-eyes pose
    // (which sets only `eye`) unreachable.
    if cmd.fly_to.is_none() && cmd.eye.is_none() {
        return;
    }
    // EXACT pose first (screenshot fix): stand in the player's eyes rather than framing them.
    if let Some((eye, fwd)) = cmd.eye.take() {
        if let Ok((mut tf, mut cam)) = q.single_mut() {
            let dir = fwd.normalize_or_zero();
            if dir.length_squared() > 0.5 {
                // Same yaw/pitch inversion as below, so the flycam keeps flying from this pose.
                cam.yaw = (-dir.x).atan2(-dir.z);
                cam.pitch = dir.y.clamp(-1.0, 1.0).asin();
                tf.rotation = Quat::from_axis_angle(Vec3::Y, cam.yaw)
                    * Quat::from_axis_angle(Vec3::X, cam.pitch);
            }
            tf.translation = eye;
        }
        cmd.fly_to = None; // an exact pose wins over any queued framing request
        return;
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
    // Texture-quality (0 full / 1 half / 2 quarter) is read by the render world at map build;
    // seed it from the persisted setting before anything can load a pack.
    // DEFAULT = 1 (Half). Full-resolution textures are ~5 GB on streets alone, which is more
    // than the whole budget of a mid-range card; Half is visually near-identical (one mip down)
    // and cuts that to ~1.3 GB. Users who have explicitly chosen a quality keep their choice --
    // this only changes the default for someone who has never touched the setting.
    // A non-Custom preset OWNS texture quality, so the two can never drift apart (a stale
    // `textureQuality` from before a preset was picked would otherwise load the wrong mips).
    let startup_preset =
        render::QualityPreset::from_index(menu::config_f32_pub("qualityPreset").unwrap_or(2.0) as u8);
    let configured_tex = menu::config_f32_pub("textureQuality").unwrap_or(1.0) as u8;
    let startup_tex = startup_preset.tex_quality().unwrap_or(configured_tex).min(2);
    render::gpu_driven::set_tex_mip_skip(startup_tex);
    // Heal config pairs written by older builds or competing old viewer processes. Without this,
    // rendering correctly obeys the named preset but the menu/in-map label reads the stale texture
    // value and claims the scene is Custom.
    if startup_preset.tex_quality().is_some() && configured_tex != startup_tex {
        let _ = menu::save_quality_preset_pub(startup_preset);
    }
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
        // Headless vendor-neutral GPU MicroSplat terrain-albedo bake — `atlas bake-terrain <manifest>`.
        // Ports the numpy `_terrain_bake_composite` (a ~961 s Reserve tail) to wgpu compute; the Python
        // extractor writes the manifest + pixels and falls back to numpy if this exits non-zero.
        if argv.get(1).map(String::as_str) == Some("bake-terrain") {
            std::process::exit(terrain_bake::run_cli(&argv[2..]));
        }
        // Headless routing QA — `atlas check-nav <pack_dir> --to "<exfil>" [--side pmc|scav|all]`
        // routes EVERY spawn point in the map to an extract and reports which ones cannot get
        // there. This is the acceptance test for a nav bake: a spawn that cannot reach an extract
        // is a map you can be stranded on.
        if argv.get(1).map(String::as_str) == Some("check-nav") {
            std::process::exit(nav_bake::run_check_cli(&argv[2..]));
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
                 EFT_GAME_DATA, EFT_LOOT_JSON, EFT_TEX_BC=0. Docs: README.md"
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
    let render_forced = std::env::var("EFT_RENDER").is_ok()
        || std::env::args().nth(2).is_some_and(|a| !a.trim().is_empty());
    let render_path = RenderPath::from_env_or(std::env::args().nth(2).as_deref());
    // A path that aborted the LAST run is not chosen again: see render/path_guard.rs. The marker
    // brackets device creation -- written before Bevy touches the GPU, cleared only once this path
    // has drawn real frames -- because `panic = "abort"` plus wgpu's `handle_error_fatal` (`-> !`)
    // mean a lost device can never be caught in-process.
    let render_path = render::path_guard::resolve_after_crash(render_path, render_forced);
    render::path_guard::mark_attempt(render_path);
    eprintln!("render path: {render_path:?}  (override with EFT_RENDER=m0|gpu)");
    // Standard-path VRAM guard: Standard decodes textures as UNCOMPRESSED RGBA8 (no BC), so a
    // persisted textureQuality=Full that is fine on the GPU-driven path (~2.2 GB BC on
    // interchange) is a ~17 GB peak there (measured, 707e5d2) — overcommit + driver paging on
    // any 16 GB card, and the first resize's reallocation dies under it. Clamp Standard to
    // Half; EFT_TEX_FULL=1 is the explicit escape hatch. Other paths keep the user's choice.
    if matches!(render_path, RenderPath::Standard)
        && render::gpu_driven::TEX_MIP_SKIP.load(std::sync::atomic::Ordering::Relaxed) == 0
        && !std::env::var("EFT_TEX_FULL").map(|v| v.trim() == "1").unwrap_or(false)
    {
        render::gpu_driven::set_tex_mip_skip(1);
        eprintln!(
            "texture quality: Full clamped to Half on the Standard render path \
             (uncompressed decode; Full peaks ~17 GB on big maps). EFT_TEX_FULL=1 overrides."
        );
    }
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
    // ESP / overlay-only: read once here so both the sync and async load paths agree, and so the
    // decision is made before anything touches the GPU.
    let esp_mode = std::env::var("EFT_ESP")
        .ok()
        .map(|v| v.trim() == "1")
        .unwrap_or_else(|| menu::config_bool_pub("espMode").unwrap_or(false));
    // Transparent overlay: resolved HERE, before the window exists, because the window's creation
    // attributes are the one chance to honour it (see the WindowPlugin block). Menu sessions are
    // always opaque -- the menu is a normal desktop app and its egui backdrop fills every pixel
    // anyway; the mode is for a MAP session summoned over the game. EFT_TRANSPARENT=0/1 overrides
    // for A/B and for scripted runs, same convention as EFT_ESP.
    let transparent_launch = pack_dir.is_some()
        && std::env::var("EFT_TRANSPARENT")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or_else(|| {
                menu::config_str_pub("overlayPresentation").as_deref() == Some("transparent")
            });
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
        match Pack::load_tier(dir, if esp_mode { eftpack::PackTier::Markers } else { eftpack::PackTier::Full }) {
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

    // Headless mode is only valid for a finite automated capture/benchmark. A leaked EFT_HIDDEN
    // used to create a perfectly healthy Atlas process at (-20000,-20000), absent from the taskbar,
    // which is indistinguishable from "the app did not open". EFT_HIDDEN_ALLOW=1 is an explicit
    // escape hatch for a custom finite harness.
    let hidden_requested =
        std::env::var("EFT_HIDDEN").map(|v| v.trim() == "1").unwrap_or(false);
    let finite_hidden_job = automated_finite_job();
    let hidden = hidden_requested && finite_hidden_job;
    if hidden_requested && !finite_hidden_job {
        eprintln!(
            "Atlas: ignoring EFT_HIDDEN=1 because no finite EFT_SHOT/EFT_BENCH job was supplied. \
             Set EFT_HIDDEN_ALLOW=1 only for a harness that guarantees process cleanup."
        );
    }

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

    // Pre-flight (P0/B1): DX12 is disabled below (it panics at pipeline creation on Bevy's own
    // downsample_depth.wgsl — a scalar push-constant, wgpu#5683 — BEFORE any render path runs, so
    // neither the GPU-driven guard nor M0 can catch it). On a machine with no Vulkan adapter, Bevy
    // would then panic deep in device init; detect it here and exit with an actionable message.
    if !render::has_usable_adapter() {
        eprintln!(
            "Atlas: no Vulkan-capable GPU adapter found.\n\
             Atlas renders through Vulkan (DirectX 12 is disabled due to an upstream driver bug,\n\
             wgpu#5683). Update your GPU drivers to a version with Vulkan support, then relaunch."
        );
        std::process::exit(1);
    }

    // Take the INTERACTIVE-GPU lease, but ONLY when we are actually rendering a map. A bake worker
    // started while we render sees it and picks the CPU backend instead of fighting us for the
    // adapter — a TDR resets the DEVICE, not just the offending process, so a runaway compute
    // dispatch would take this viewer down with it (that is the 0xC0000409 abort we hit).
    //
    // NOT IN MENU MODE. The menu and the viewer are one process, and this used to be taken for the
    // whole lifetime unconditionally — so a build launched from the menu always found the GPU
    // "busy", held by an idle settings screen, and EVERY bake started from the UI silently took the
    // CPU backend. Measured: interchange's SH bake spent 6m34s on the CPU with the GPU path sitting
    // right there. A menu is not the interactive map view this protects. The in-place PLAY switch
    // calls `gpu_lease::hold` as the map loads, so the moment we really are rendering, we claim it.
    if !menu_mode {
        let lease_held = gpu_lease::hold("map on the command line");
        // Hidden captures are easy to strand and historically accumulated until several viewers
        // fought over one Vulkan device. Interactive launches remain permissive, but an automated
        // hidden job must be the only map renderer.
        if hidden && !lease_held {
            eprintln!(
                "Atlas: refusing hidden capture because another map viewer owns the GPU lease. \
                 Close it or capture from that visible viewer."
            );
            std::process::exit(73);
        }
    }

    let mut app = App::new();
    // Capture the screenshot handoff before OverlayPlugin's PostStartup cleanup removes EFT_POSE.
    // The resource survives the async file/GPU load and is consumed by the first map reset.
    app.insert_resource(PendingStartupPose(
        std::env::var("EFT_POSE")
            .ok()
            .and_then(|value| parse_eft_pose(&value)),
    ));
    app.add_plugins(
        DefaultPlugins
            // Persistent file log (packs/logs/atlas_viewer.log): double-click launches have no
            // console, so without this a GPU crash on a user's machine (wgpu validation error,
            // device-lost, TDR) leaves zero evidence. The file layer tees everything the console
            // gets; a panic hook below routes panic messages through it too.
            .set(bevy::log::LogPlugin {
                custom_layer: viewer_file_log_layer,
                ..default()
            })
            .set(bevy::render::RenderPlugin {
                // P0/B1: pin to Vulkan on Windows (see render::allowed_backends). DX12 never worked
                // for EITHER the GPU-driven or M0 path (it crashes on a Bevy shader regardless of
                // features), so restricting to Vulkan loses no working configuration and removes a
                // confusing mid-pipeline panic on the AMD/Intel machines that would route to it.
                render_creation: bevy::render::settings::WgpuSettings {
                    backends: Some(render::allowed_backends()),
                    ..default()
                }
                .into(),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Atlas".into(),
                    // EFT_WIN=WxH overrides for benches (resolution scaling splits
                    // fragment-bound from CPU/fixed cost); default 1600x1000.
                    resolution: std::env::var("EFT_WIN")
                        .ok()
                        .and_then(|s| {
                            let (w, h) = s.trim().split_once('x')?;
                            Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))
                        })
                        .unwrap_or((1600u32, 1000u32))
                        .into(),
                    present_mode,
                    // Frame-queue depth 1 (wgpu default: 2). On a GPU-bound machine under
                    // FIFO/vsync the deeper queue keeps the swapchain backlog permanently
                    // full, which is what pushed acquire toward its 1 s budget during window
                    // moves/resizes (RX 6800 field crash — see the vendored bevy_render
                    // prepare_windows patch, the other half of this fix). Depth 1 trades a
                    // little CPU/GPU overlap for backlog headroom + lower input latency;
                    // EFT_FRAME_LATENCY=2 restores the wgpu default for A/B.
                    desired_maximum_frame_latency: std::num::NonZeroU32::new(
                        std::env::var("EFT_FRAME_LATENCY")
                            .ok()
                            .and_then(|v| v.trim().parse::<u32>().ok())
                            .unwrap_or(1),
                    ),
                    // EFT_HIDDEN=1: render without showing a window (headless EFT_SHOT
                    // verification runs — GPU screenshot capture works on an invisible
                    // window; pair with EFT_UNCAPPED so the focus-idle gate doesn't stall).
                    // Bevy re-shows the window after the first present, so belt-and-braces:
                    // also park it far off-screen and skip the taskbar.
                    visible: !hidden,
                    position: if hidden {
                        WindowPosition::At(IVec2::new(-20000, -20000))
                    } else {
                        WindowPosition::Automatic
                    },
                    skip_taskbar: hidden,
                    // TRANSPARENT MODE: every one of these must be set at CREATION. DWM decides
                    // whether a window composites per-pixel at CreateWindowEx, and measurement
                    // says it only honours alpha for the undecorated + non-resizable +
                    // always-on-top conjunction — created decorated it blends to white FOREVER,
                    // and no runtime mutation (which is what the overlay's summon path does for
                    // the other modes) can repair it. Menu sessions never take this branch: the
                    // menu is its own launch, and PLAY relaunches into a fresh process, which is
                    // what makes a launch-gated mode usable from a settings screen at all.
                    // `composite_alpha_mode` stays Auto->Opaque on adapters with no blending mode
                    // (the vendored bevy_render guard), so the worst case is an opaque window,
                    // never an abort.
                    transparent: transparent_launch,
                    decorations: !transparent_launch,
                    resizable: !transparent_launch,
                    window_level: if transparent_launch {
                        bevy::window::WindowLevel::AlwaysOnTop
                    } else {
                        bevy::window::WindowLevel::Normal
                    },
                    composite_alpha_mode: if transparent_launch {
                        // PreMultiplied matches what Vulkan advertises on the measured NVIDIA
                        // driver; Intel advertises Inherit, which measured numerically identical.
                        // The bevy_render guard downgrades whichever is absent to Opaque with a
                        // warning instead of the 0xC0000409 abort wgpu would otherwise raise.
                        bevy::window::CompositeAlphaMode::PreMultiplied
                    } else {
                        bevy::window::CompositeAlphaMode::Auto
                    },
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
    // CARRY THE MENU'S QUALITY PRESET INTO THE SCENE. The preset is picked in the main menu (it has
    // to be: texture quality is applied when textures are UPLOADED, so choosing it after a map is
    // resident cannot change what was uploaded). Apply it here, before the pack builds, so the
    // render-side choices match the textures that are about to be loaded. `Custom` deliberately
    // applies nothing — it means "the user's own mix", which is whatever the other settings say.
    let preset = render::QualityPreset::from_index(
        menu::config_f32_pub("qualityPreset").unwrap_or(2.0) as u8,
    );
    {
        // An EXPLICIT env override outranks the preset. `GfxSettings::default()` reads
        // EFT_SHADOWS/EFT_BLOOM/EFT_SSAO/EFT_LIGHTS, and a blanket `apply` would overwrite them --
        // silently, and on a fresh install too, since a missing `qualityPreset` defaults to High.
        // That would make the A/B capture and benchmark harnesses (tools/bench_gfx.py) measure
        // something other than what they asked for, which is exactly how a measurement lies.
        let (env_shadows, env_bloom, env_ssao, env_lights) = (
            std::env::var("EFT_SHADOWS").is_ok(),
            std::env::var("EFT_BLOOM").is_ok(),
            std::env::var("EFT_SSAO").is_ok(),
            std::env::var("EFT_LIGHTS").is_ok(),
        );
        // Same rule for the three newest options. Ultra now turns volumetric shafts ON, so without
        // this an `EFT_VOLUMETRIC=0` A/B run on a machine with Ultra persisted would silently measure
        // shafts ON — the precise failure the comment above describes, and one that would have
        // corrupted the +5.40 ms figure had the harness not forced the Custom preset.
        let (env_vol, env_aa, env_grass_dist, env_ae) = (
            std::env::var("EFT_VOLUMETRIC").is_ok(),
            std::env::var("EFT_AA").is_ok(),
            std::env::var("EFT_GRASS_DIST").is_ok(),
            std::env::var("EFT_AUTO_EXPOSURE").is_ok(),
        );
        let before = gfx.clone();
        preset.apply(&mut gfx);
        if env_shadows {
            gfx.shadows = before.shadows;
        }
        if env_bloom {
            gfx.bloom = before.bloom;
        }
        if env_ssao {
            gfx.ssao = before.ssao;
        }
        if env_lights {
            gfx.lights = before.lights;
        }
        if env_vol {
            gfx.volumetric = before.volumetric;
        }
        if env_aa {
            gfx.aa = before.aa;
        }
        if env_grass_dist {
            gfx.grass_dist_m = before.grass_dist_m;
        }
        if env_ae {
            gfx.auto_exposure = before.auto_exposure;
        }
    }
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
    // SsaoPlugin is told which path is installed: it orders against a render-graph node that only
    // the GPU-driven path creates, and `render_path` is not a resource yet at this point.
    if transparent_launch {
        // Registered ONLY on transparent launches: both its checks are meaningless (and the
        // readback not free) when the window is opaque by design.
        app.add_plugins(render::TransparencyCheckPlugin);
    }
    app.add_plugins((
        render::RenderPathGuardPlugin,
        GradePlugin,
        render::SsaoPlugin { gpu_driven: render_path == RenderPath::GpuDriven },
        render::TaaPlugin,
        render::SsrPlugin,
        render::FpvCamPlugin,
    ));
    // Phase 0 (docs/GRAPHICS_PLAN.md): per-pass GPU timestamps. Frame averages cannot resolve the
    // plan's sub-millisecond costs from the 0.3 ms noise floor; the phases' acceptance criteria are
    // written against these spans ("eft cull/shadow/prepass..."). Env-gated because timestamp
    // queries are not free and nothing reads them in normal play.
    if std::env::var("EFT_GPU_TIMING").map(|v| v.trim() == "1").unwrap_or(false) {
        app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
        app.add_systems(Update, print_gpu_timing);
    }
    // Runtime graphics settings reach the render world on EVERY render path (grade/SSAO install
    // unconditionally, so the extraction can't live inside EftGpuDrivenPlugin — under EFT_RENDER=
    // m0/std the toggles would silently stop reaching the GPU).
    app.add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<render::GfxSettings>::default());

    if let Some(p) = pack {
        app.insert_resource(LoadedPack(std::sync::Arc::new(p)));
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
    app.insert_resource(if transparent_launch {
        // Transparent window: the composite is PREMULTIPLIED, so "no coverage" must be (0,0,0,0)
        // exactly -- a premultiplied pixel with rgb > 0 at alpha 0 is additive glow over the game,
        // not absence. This is the one clear colour where the rgb values are load-bearing.
        ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0))
    } else if menu_mode {
        ClearColor(Color::srgb_u8(9, 9, 9))
    } else if esp_mode {
        // ESP draws no world, so the overcast horizon stand-in would just be a grey wall over the
        // game. Near-black rather than pure black: an opaque pure-black panel over a raid reads as
        // "the app crashed", and the whole point is that the player can trust what they are seeing.
        ClearColor(Color::srgb_u8(11, 13, 16))
    } else {
        ClearColor(Color::srgb(0.55, 0.58, 0.58))
    })
        .add_plugins(pick::PickPlugin) // double-LEFT-click raycast-vs-pack-data debug pick
        .add_plugins(loot::LootPlugin) // 823 loot containers from tarkmap out/loot.json
        .add_plugins(fx::FxPlugin) // looping fires/smoke/steam from the game's ParticleSystems
        .add_plugins(npc::NpcPlugin) // scavs walking the game's own patrol_ways
        .add_plugins(overlay::OverlayPlugin) // in-game screenshot summons the map over the game (same window)
        .add_plugins(insights::InsightsPlugin) // netcode position breadcrumbs mined from the logs
        .add_plugins(assets::AssetsPlugin) // browse the game's Unity bundles, joined to the pick
        .add_plugins(poi::PoiPlugin) // PMC/scav/boss spawns + extracts/doors/interactables
        .add_plugins(inspect::InspectPlugin) // left-click a marker -> floating info card (\u{2715} to close)
        .add_plugins(ui::UiPlugin) // right-hand layer-toggle panel
        .add_plugins(progress::ProgressPlugin) // persistent tracked tasks, objectives, and owned keys
        .add_plugins(tasks_panel::TasksPanelPlugin) // revamped Tasks tab: catalog + icon cache (router calls tasks_panel_ui)
        .add_plugins(pathfind::PathfindPlugin) // in-process CPU routing over the baked nav grid (nav.rs)
        .add_plugins(game_watch::GameWatchPlugin) // passive game link: auto map swap, live player fix, task sync
        .add_plugins(planner::PlannerPlugin) // loot-run orienteering planner (Navigation tab)
        .add_plugins(jobs::JobsPlugin) // background job worker: build/sync maps while a map is open
        .add_plugins(character::CharacterPlugin) // EFT_CHARACTER=<id|dir> -> skinned body on the walk camera
        .init_resource::<CameraCommand>() // UI-driven "fly the camera to X" (search / quest jump / route)
        .init_resource::<CameraSettings>() // camera-tab: FOV / fly speed / walk mode
        .init_resource::<MapSwitch>() // UI map dropdown -> switch to the selected pack (in place)
        .init_resource::<ReturnToMenu>() // toolbar "back to menu" button -> relaunch into the menu
        .init_resource::<PendingMapLoad>() // async in-place pack load (no frame freeze on switch)
        .init_resource::<MapLoadError>() // async load failure -> UI error + back-to-menu (finding 4)
        .init_resource::<ForcedLod>() // graphics-panel LOD selector (meaningful on --alllod packs)
        .init_resource::<ShowDisabledGeom>() // "show disabled geometry" toggle (Unity-inactive scenery)
        .add_plugins(loot_volume::LootVolumePlugin) // Analysis tab: loot-VALUE grid over the pack bounds
        .init_resource::<agent_link::AgentLinkCtl>() // drone agent link (TCP lockstep sim control)
        // DoorClick is CONSUMED only by the gpu-driven render path, but pick_system (always-on)
        // WRITES it unconditionally — on the M0/std paths EftGpuDrivenPlugin (its only init site)
        // never runs and Bevy 0.17 panics at param validation the first Update tick (field report:
        // the LLPC auto-fallback exposed this instantly on an RX 7800 XT). Init it here for every
        // path; on M0 it's a dead-letter box, which is harmless.
        .init_resource::<render::gpu_driven::DoorClick>()
        .add_systems(Startup, (setup, log_render_path, install_gpu_error_handler))
        // Clean, logged exit on a fatal GPU error (device lost / OOM): wgpu's default handler
        // panics, and release panic=abort made that a silent death — see the handler below.
        .add_systems(Update, gpu_fatal_watchdog)
        // walk_move/drone_move run AFTER flycam_look (orientation resolved) and flycam_move
        // (mutually exclusive by CamMode) so they can't race the shared Transform; the agent
        // spectate cam runs last and wins while a session drives the drone. Disabled in the MENU
        // (MenuState present): the backdrop camera stays locked to its composed pose — no WASD /
        // RMB-look / cursor-grab — while the scene itself drifts under the cursor (menu_city_update).
        .add_systems(
            Update,
            (
                cursor_grab,
                flycam_look,
                flycam_move,
                walk_move,
                drone_move,
            )
                .chain()
                .run_if(not(resource_exists::<menu::MenuState>)),
        )
        .init_resource::<BenchSampleStart>()
        .add_systems(Update, (apply_camera_command, auto_screenshot, debug_switch, return_to_menu, bump_epoch_on_lod_change, bump_epoch_on_disabled_geom_change, bench_stats, arm_auto_exposure))
        // Bench cameras override the fly-cam AFTER Update, before transforms propagate.
        .add_systems(
            PostUpdate,
            debug_bench_camera.before(bevy::transform::TransformSystems::Propagate),
        )
        .insert_resource(TransparentWindow(transparent_launch))
        .insert_resource(EspMode(menu::config_bool_pub("espMode").unwrap_or(false)))
        .insert_resource(EspMode(esp_mode))
        .init_resource::<esp_labels::EspLabels>()
        .init_resource::<render::PanelLensShift>()
        .init_resource::<render::OverlaySlice>()
        // The SOLE writer of Camera::sub_camera_view, after egui has finished laying out (so the
        // panel width it derives from is this frame's) and before Bevy consumes the camera.
        .add_systems(
            PostUpdate,
            render::apply_view_slice
                .after(bevy_egui::EguiPostUpdateSet::ProcessOutput)
                .before(bevy::camera::CameraUpdateSystems),
        )
        .add_systems(
            Update,
            (apply_gfx_camera, load_map, poll_map_load, clear_map_error_on_new_load, flycam_scroll, apply_camera_fov, build_walk_ground),
        )
        .add_systems(
            Update,
            (agent_link::agent_sync, agent_link::agent_gizmos)
                .run_if(not(resource_exists::<menu::MenuState>)),
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

    // Panics → atlas_viewer.log (the subscriber exists now that LogPlugin is installed).
    install_panic_log_hook();

    app.run();
}

/// True when this process is a finite automated job (an `EFT_SHOT` capture, an `EFT_BENCH` run, or
/// a custom `EFT_HIDDEN_ALLOW` harness) rather than an interactive desk session. Automated runs
/// must be deterministic, so the desk-tool conveniences stand down: the overlay is forced off (its
/// hidden-idle throttle is a 500 ms reactive frame clock — a wall-clock bench that inherits it
/// measures exactly 2 fps regardless of the map) and the game link never starts (it consumes +
/// deletes the player's screenshots and commits a deferred map swap, either of which corrupts a
/// scripted run when the game happens to be live).
pub fn automated_finite_job() -> bool {
    std::env::var_os("EFT_SHOT").is_some()
        || std::env::var_os("EFT_BENCH").is_some()
        || std::env::var("EFT_HIDDEN_ALLOW").map(|v| v.trim() == "1").unwrap_or(false)
}

/// Startup echo of the resolved render path INTO the file log: the capability probe (incl. the
/// LLPC driver quirk) runs before the log subscriber exists, so its eprintln is invisible on a
/// double-click launch — this line is how a user's atlas_viewer.log tells us which path ran.
fn log_render_path(rp: Res<RenderPath>) {
    info!("render path: {:?} (auto-probed unless EFT_RENDER was set)", *rp);
}

/// Set by the uncaptured-error handler (render/wgpu internal thread) when an error class the
/// device cannot survive comes through; drained by `gpu_fatal_watchdog` on the main schedule.
static GPU_FATAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// wgpu's DEFAULT uncaptured-error handler panics, and release builds are panic=abort — so any
/// validation error or device loss at runtime was a silent process death with no field
/// evidence (the sh-bake CLI installs a non-panicking handler for exactly this reason,
/// sh_bake_gpu.rs). Log every uncaptured error through the file-log layer instead; flag the
/// fatal classes (OutOfMemory, Internal = device-lost family) for a clean exit. Validation
/// errors are rate-limited: a lost device error-floods every frame, and the first few lines
/// carry all the signal.
fn install_gpu_error_handler(device: Res<bevy::render::renderer::RenderDevice>) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    device.wgpu_device().on_uncaptured_error(Box::new(|e| {
        let fatal = matches!(
            e,
            wgpu::Error::OutOfMemory { .. } | wgpu::Error::Internal { .. }
        );
        let n = SEEN.fetch_add(1, Ordering::Relaxed);
        if fatal || n < 24 || n % 256 == 0 {
            error!(
                "wgpu uncaptured error #{n}{}: {e}",
                if fatal { " (FATAL class)" } else { "" }
            );
        }
        if fatal {
            GPU_FATAL.store(true, Ordering::Relaxed);
        }
    }));
    // wgpu deliberately does NOT send DeviceLost through the uncaptured-error callback. Without
    // this separate callback, the next mapped staging-buffer access can panic first and make a
    // device reset look like an unrelated UniformBuffer unwrap (the July 28/29 field logs).
    device.wgpu_device().set_device_lost_callback(|reason, message| {
        error!("wgpu DEVICE LOST ({reason:?}): {message}");
        GPU_FATAL.store(true, Ordering::Release);
    });
}

/// A lost device can't be rebuilt inside a running Bevy 0.17 app — every later submit fails
/// forever. Freezing (skip-frame loop) or aborting both strand the user; one loud log line +
/// a clean coded exit is actionable and shows up in packs/logs/atlas_viewer.log.
fn gpu_fatal_watchdog(mut exit: MessageWriter<bevy::app::AppExit>) {
    if GPU_FATAL.swap(false, std::sync::atomic::Ordering::Relaxed) {
        error!(
            "fatal GPU error (device lost or out of GPU memory) — exiting cleanly. Please \
             report this with packs/logs/atlas_viewer.log attached. If it happened while \
             moving/resizing the window or alt-tabbing, mention that too."
        );
        exit.write(bevy::app::AppExit::Error(
            std::num::NonZeroU8::new(86).unwrap(),
        ));
    }
}

/// File layer for the LogPlugin: tee all tracing output (incl. wgpu validation errors and
/// device-lost messages) to `packs/logs/atlas_viewer.log`, appending across sessions with a
/// separator line, rotated once past ~5 MB. Returns None (console-only) if the packs dir isn't
/// writable — logging must never keep the app from starting.
fn viewer_file_log_layer(_app: &mut App) -> Option<bevy::log::BoxedLayer> {
    let dir = paths::packs_root().join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join("atlas_viewer.log");
    if std::fs::metadata(&path).map(|m| m.len() > 5_000_000).unwrap_or(false) {
        let _ = std::fs::rename(&path, dir.join("atlas_viewer.old.log"));
    }
    let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok()?;
    {
        use std::io::Write as _;
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut f = &file;
        let _ = writeln!(f, "\n==== atlas session start (epoch {epoch}, v{}) ====", env!("CARGO_PKG_VERSION"));
    }
    Some(Box::new(
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(file)),
    ))
}

/// Route panic messages through tracing so they land in atlas_viewer.log before the process dies
/// (release builds are panic=abort — the hook still runs first). Chains the default hook so the
/// console backtrace behavior is unchanged. Installed AFTER LogPlugin so the subscriber exists.
fn install_panic_log_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        error!("PANIC: {info}");
        default_hook(info);
    }));
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
/// Phase 4 (docs/GRAPHICS_PLAN.md): the game's OWN sky when extracted, procedural fallback
/// otherwise. eft_extract_sky.py exports the cubemaps EFT ships in StreamingAssets — for the
/// visible sky we load `rain_1k_sharp` (the 1024px overcast raid sky; EFT raids are overcast) from
/// packs/shared/sky/. Faces are already in wgpu order (+X,-X,+Y,-Y,+Z,-Z — Unity's order matches).
/// PNGs are sRGB; the skybox samples linearly, so decode here (×2.2 approx, same as the sidecar's
/// derived colors). Any failure falls through to the procedural gradient, clearly logged, so a
/// pack-less install renders exactly as before — the fallback is legacy, and is never claimed
/// derived.
fn build_sky_cubemap(images: &mut Assets<Image>, sun: Vec3) -> Handle<Image> {
    // Procedural overcast dome, and ONLY that. The Phase-4 attempt to use the game's extracted
    // cubemaps as the visible sky is REMOVED (not an option): those assets are environment
    // CAPTURES — photo-spheres with treelines baked into the horizon — and as a sky dome they put
    // photographic trees behind the map's real geometry. The extraction survives as
    // packs/shared/sky's DERIVED zenith/horizon colors, which are the right feed for reflections
    // and fog (colors, not photographs); the dome itself stays synthesized.
    build_procedural_sky(sun)
        .map(|img| images.add(img))
        .expect("procedural sky is infallible")
}

fn build_procedural_sky(sun: Vec3) -> Option<Image> {
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
    Some(image)
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
        if let Some(pose) = parse_eft_pose(&s) {
            cam_pos = pose.position;
            yaw = pose.yaw;
            pitch = pose.pitch;
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
    esp: Res<EspMode>,
    pack: Option<Res<LoadedPack>>,
    mut images: ResMut<Assets<Image>>,
    grade: Option<Res<GradeLutCpu>>,
    mut gfx: ResMut<render::GfxSettings>,
) {
    let (cam_pos, target, far, yaw, pitch) = frame_for_pack(pack.as_ref().map(|p| &*p.0));

    // Sky sun direction from the pack's volume sidecar (same one the SH/shadow path uses, so the
    // skybox sun disk, baked GI and reflected sun agree). The experimental shadow toggle needs a
    // real sun (matches the render side's sun_ok gate).
    let (sun_dir, sun_ok) = pack_sun_dir(pack.as_ref().map(|p| &*p.0));
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
        drone::DroneRig::default(), // per-camera drone airframe state (inert until drone mode)
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
    // ESP: no skybox. It is the one thing left that paints the WHOLE frame, so with it in place
    // the near-black clear colour is never seen and the overlay is a tan gradient over the game
    // rather than a cutout of markers.
    if let Some(image) = sky.filter(|_| !esp.0) {
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
    settings: Res<CameraSettings>,
    mut q: Query<(&mut Transform, &mut FlyCam)>,
) {
    // Drone mode: the camera is bolted to the airframe — mouse X feeds the drone's yaw stick
    // (drone::drone_move), never free-look.
    if !mouse.pressed(MouseButton::Right) || pointer_on_ui.0 || settings.mode == CamMode::Drone {
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
    // In walk/drone mode, walk_move / drone_move owns locomotion — fly is inert.
    if ui_kb.0 || settings.mode != CamMode::Fly {
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

/// FPV drone locomotion (CamMode::Drone). Two sources can own the airframe:
///  - an ACTIVE agent-link session (external trainer stepping the sim over TCP): the camera
///    spectates that drone's pose (toggleable) and manual input stands down;
///  - otherwise the manual rig. ACRO (default — the real FPV deal): sticks command body rates
///    through a Betaflight rate curve, throttle is POSITIONAL (Space/Ctrl ramp it, a gamepad /
///    USB RC transmitter's left stick sets it directly — Mode 2: left = throttle+yaw, right =
///    pitch+roll). ANGLE: self-leveling tilt + altitude assist (trainer wheels). R (or gamepad
///    South) respawns. Physics = drone::step at 1 ms substeps; crash cuts thrust until reset.
fn drone_move(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    pads: Query<&Gamepad>,
    time: Res<Time>,
    ui_kb: Res<inspect::UiWantsKeyboard>,
    settings: Res<CameraSettings>,
    shared: Option<Res<agent_link::AgentShared>>,
    grid: Option<Res<walk_ground::GroundGrid>>,
    mut q: Query<(&mut Transform, &mut FlyCam, &mut drone::DroneRig), With<CullCamera>>,
) {
    if settings.mode != CamMode::Drone {
        // Leaving drone mode: the rig respawns fresh at the camera pose next activation.
        for (_, _, mut rig) in &mut q {
            if rig.live {
                rig.live = false;
            }
        }
        return;
    }
    let cam_tilt = settings.drone_cam_tilt_deg.to_radians();
    let dt = time.delta_secs().min(0.05);
    let typing = ui_kb.0;
    for (mut tf, mut cam, mut rig) in &mut q {
        // Agent session active → spectate its airframe instead of flying the manual rig.
        if let Some(sh) = &shared {
            let w = sh.0.lock().unwrap();
            if w.active {
                if w.spectate {
                    tf.translation = w.drone.pos;
                    tf.rotation = w.drone.quat * Quat::from_rotation_x(w.params.cam_tilt);
                    let fwd = *tf.forward();
                    cam.yaw = (-fwd.x).atan2(-fwd.z);
                    cam.pitch = fwd.y.clamp(-1.0, 1.0).asin();
                }
                rig.live = false;
                continue;
            }
        }
        let p = drone::DroneParams::default();
        // (Re)spawn the manual rig at the current camera pose.
        if !rig.live {
            rig.state = drone::DroneState::spawn(tf.translation, cam.yaw);
            rig.spawn_pos = tf.translation;
            rig.spawn_yaw = cam.yaw;
            rig.kb_stick = Vec3::ZERO;
            rig.throttle = p.hover_throttle();
            rig.live = true;
        }
        let pad = pads.iter().next();
        let respawn = (!typing && keys.just_pressed(KeyCode::KeyR))
            || pad.map(|g| g.just_pressed(GamepadButton::South)).unwrap_or(false)
            // Fell out of the world (void map edge) → respawn rather than fall forever.
            || rig.state.pos.y < rig.spawn_pos.y - 400.0;
        if respawn {
            rig.state = drone::DroneState::spawn(rig.spawn_pos, rig.spawn_yaw);
            rig.throttle = p.hover_throttle();
        }

        // --- Virtual sticks: smoothed keyboard + mouse yaw + raw gamepad (Mode 2) -------------
        let mut kb = Vec3::ZERO; // (roll, pitch, yaw) targets, ±1
        if !typing {
            if keys.pressed(KeyCode::KeyW) {
                kb.y += 1.0;
            }
            if keys.pressed(KeyCode::KeyS) {
                kb.y -= 1.0;
            }
            if keys.pressed(KeyCode::KeyD) {
                kb.x += 1.0;
            }
            if keys.pressed(KeyCode::KeyA) {
                kb.x -= 1.0;
            }
            if keys.pressed(KeyCode::KeyE) {
                kb.z += 1.0;
            }
            if keys.pressed(KeyCode::KeyQ) {
                kb.z -= 1.0;
            }
        }
        // Keys are square waves; a ~90 ms low-pass turns them into flyable stick ramps.
        let rise = 1.0 - (-dt / 0.09).exp();
        let ramp = (kb - rig.kb_stick) * rise;
        rig.kb_stick += ramp;
        let mut roll = rig.kb_stick.x;
        let mut pitch = rig.kb_stick.y;
        let mut yaw = rig.kb_stick.z;
        if mouse.pressed(MouseButton::Right) {
            // Mouse-X is a yaw-rate stick: pixels/frame → normalized, so flick speed = turn rate.
            yaw += (motion.delta.x * 0.012).clamp(-1.0, 1.0);
        }
        // Gamepad / USB RC transmitter (first one found), standard Mode-2 layout with a small
        // deadzone. Axes ADD to keyboard so either works without a toggle.
        let mut pad_throttle: Option<f32> = None;
        if let Some(g) = pad {
            let dz = |v: f32| if v.abs() < 0.04 { 0.0 } else { v };
            let rx = dz(g.get(GamepadAxis::RightStickX).unwrap_or(0.0));
            let ry = dz(g.get(GamepadAxis::RightStickY).unwrap_or(0.0));
            let lx = dz(g.get(GamepadAxis::LeftStickX).unwrap_or(0.0));
            let ly = g.get(GamepadAxis::LeftStickY).unwrap_or(0.0);
            roll += rx;
            pitch += ry; // stick up = nose forward
            yaw += lx;
            if ly.abs() > 0.02 || settings.drone_acro {
                pad_throttle = Some(((ly + 1.0) * 0.5).clamp(0.0, 1.0));
            }
        }
        roll = roll.clamp(-1.0, 1.0);
        pitch = pitch.clamp(-1.0, 1.0);
        yaw = yaw.clamp(-1.0, 1.0);

        let mut act = drone::DroneAction::default();
        let mode = if settings.drone_acro {
            // ACRO: Betaflight rate curve → normalized rate command (physical cap = max_rate).
            let (rc, ex, sr) = (settings.drone_rc_rate, settings.drone_expo, settings.drone_super_rate);
            act.roll = (drone::bf_rate(roll, rc, ex, sr) / p.max_rate.z).clamp(-1.0, 1.0);
            act.pitch = (drone::bf_rate(pitch, rc, ex, sr) / p.max_rate.x).clamp(-1.0, 1.0);
            act.yaw = (drone::bf_rate(yaw, rc, ex, sr) / p.max_rate.y).clamp(-1.0, 1.0);
            // POSITIONAL throttle: gamepad left-Y sets it, keyboard Space/Ctrl ramp it and it
            // STAYS where you leave it (that's how a real throttle stick works).
            if let Some(t) = pad_throttle {
                rig.throttle = t;
            } else if !typing {
                let mut d = 0.0;
                if keys.pressed(KeyCode::Space) {
                    d += 1.5;
                }
                if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
                    d -= 1.5;
                }
                rig.throttle = (rig.throttle + d * dt).clamp(0.0, 1.0);
            }
            act.throttle = rig.throttle;
            drone::ControlMode::Rates
        } else {
            // ANGLE (trainer wheels): sticks = tilt, altitude-assist throttle. Tilt compensation
            // keeps height in banked turns; a vertical-speed P-loop does the rest.
            act.roll = roll;
            act.pitch = pitch;
            act.yaw = yaw;
            let up_y = (rig.state.quat * Vec3::Y).y.max(0.35);
            let mut climb = 0.0;
            if let Some(t) = pad_throttle {
                climb = (t * 2.0 - 1.0) * 5.0;
            }
            if !typing {
                if keys.pressed(KeyCode::Space) {
                    climb += 4.0;
                }
                if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
                    climb -= 4.0;
                }
            }
            act.throttle =
                (p.hover_throttle() / up_y + 0.09 * (climb - rig.state.vel.y)).clamp(0.0, 1.0);
            rig.throttle = act.throttle; // keep the HUD bar honest in angle mode too
            drone::ControlMode::Angle
        };

        // --- Physics: real-FPV-sim rate — 1 ms substeps (1 kHz), contact + attitude stable ---
        let g: Option<&walk_ground::GroundData> = grid.as_ref().map(|r| &*r.0);
        let n = ((dt / 0.001).ceil() as u32).clamp(1, 64);
        let sub = dt / n as f32;
        for _ in 0..n {
            drone::step(&mut rig.state, &p, act, mode, Vec3::ZERO, g, sub);
        }

        // --- Camera = airframe + FPV uptilt ---
        tf.translation = rig.state.pos;
        tf.rotation = rig.state.quat * Quat::from_rotation_x(cam_tilt);
        let fwd = *tf.forward();
        cam.yaw = (-fwd.x).atan2(-fwd.z);
        cam.pitch = fwd.y.clamp(-1.0, 1.0).asin();
    }
}

/// Lazily build the walk/drone collision grid the first time it's needed (fly-only users never
/// pay the ~250-400 MB + build cost). Walk needs ground+walls; drone mode and the agent link also
/// need ceilings — if the grid was first built walk-only and a flying consumer appears later, it
/// is rebuilt once with ceilings.
fn build_walk_ground(
    mut commands: Commands,
    settings: Res<CameraSettings>,
    agent: Res<agent_link::AgentLinkCtl>,
    grid: Option<Res<walk_ground::GroundGrid>>,
    pack: Option<Res<LoadedPack>>,
) {
    let flying = settings.mode == CamMode::Drone || agent.enabled;
    let needed = flying || settings.mode == CamMode::Walk;
    if !needed {
        return;
    }
    if let Some(g) = &grid {
        if g.has_ceilings || !flying {
            return; // already sufficient
        }
    }
    let Some(pack) = pack else { return };
    info!(
        "walk_ground: building collision grid (ceilings: {}) …",
        flying
    );
    commands.insert_resource(walk_ground::GroundGrid::build(&pack.0, flying));
}

/// Walk locomotion: yaw-only WASD drives a persistent horizontal velocity using the extracted EFT
/// acceleration/deceleration and sprint-transition settings. Ground-follow, capsule collision, and
/// fixed ballistic jumping remain viewer-side because the original animation/root-motion rig is
/// not part of an eftpack. Gated on walk mode; fly is inert then.
fn walk_move(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    ui_kb: Res<inspect::UiWantsKeyboard>,
    settings: Res<CameraSettings>,
    grid: Option<Res<walk_ground::GroundGrid>>,
    mut q: Query<(&mut Transform, &FlyCam, &mut walk_ground::WalkState), With<CullCamera>>,
) {
    use walk_ground::{EYE_HEIGHT, GRAVITY, KILL_DROP, STEP_UP};
    if settings.mode != CamMode::Walk {
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
        let wish = Vec2::new(h.x, h.z);
        let sprint = !typing
            && (keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight));
        let start_xz = Vec2::new(tf.translation.x, tf.translation.z);
        let grounded = ws.grounded;
        let delta = walk_ground::advance_horizontal(
            &mut ws,
            wish,
            settings.walk_speed,
            sprint,
            grounded,
            dt,
        );
        tf.translation.x += delta.x;
        tf.translation.z += delta.y;

        // Player-sized collision: push the body capsule back out of any wall it entered. Feed the
        // correction back into velocity by cancelling only the inward component, which preserves
        // momentum parallel to the wall and produces a natural slide.
        let feet_y = tf.translation.y - EYE_HEIGHT;
        let proposed_xz = Vec2::new(tf.translation.x, tf.translation.z);
        let fixed = grid.resolve_walls(tf.translation, feet_y);
        let fixed_xz = Vec2::new(fixed.x, fixed.y);
        walk_ground::cancel_velocity_into_wall(&mut ws, fixed_xz - proposed_xz);
        tf.translation.x = fixed.x;
        tf.translation.z = fixed.y;
        // Actual resolved displacement drives footsteps/head bob; pushing into a wall no longer
        // makes the camera bob in place.
        let moved = (fixed_xz - start_xz).length();

        // Jump (behind the typing guard so Space in a text field doesn't launch).
        if !typing && keys.just_pressed(KeyCode::Space) && ws.grounded {
            ws.vy = walk_ground::jump_velocity();
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
                // A step you can walk UP is one you can walk DOWN. `ground_height` already probes
                // `STEP_UP` for the surface underfoot, so a drop within that same tolerance is a
                // curb, not a ledge — stay grounded and glide down it. Requiring `new_y <= target`
                // instead made every downward step read as airborne until gravity caught up, which
                // is ~0.17 s of the FALLING animation for a 15 cm kerb.
                let drop = tf.translation.y - target;
                let stepping_down = ws.vy <= 0.0 && drop > 0.0 && drop <= STEP_UP;
                if ws.vy <= 0.0 && (new_y <= target || stepping_down) {
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
                    ws.grounded = false; // airborne (rising, or off a real ledge)
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

/// Reliable frame capture: with `EFT_SHOT=<path>` set, save ONE screenshot of the primary window
/// via Bevy's own GPU screenshot (bypasses the DWM/flip-model capture that grabs a blank white
/// frame). By default it is LOAD-AWARE: it waits until the pack has loaded AND the GPU build has
/// finished (the texcache/geometry stream across many frames after the file load), then settles a
/// beat — so a bench screenshot is never a blank/loading frame, regardless of a slow first load.
/// Hidden captures exit after the file is written so they cannot become invisible orphan viewers.
/// A visible capture remains interactive unless `EFT_SHOT_EXIT=1` is explicitly set.
/// `EFT_SHOT_FRAME=<n>` forces the legacy ABSOLUTE-frame capture instead (for `EFT_SWITCH` soak
/// tests that need a precise frame); `EFT_SHOT_SETTLE=<n>` tunes the post-load settle (default 30).
fn auto_screenshot(
    mut commands: Commands,
    mut frames: Local<u32>,
    mut settle: Local<i32>,
    mut done: Local<bool>,
    pending: Res<PendingMapLoad>,
    gpu_load: Option<Res<render::GpuLoadSignal>>,
    pack: Option<Res<LoadedPack>>,
) {
    if *done {
        return;
    }
    let Ok(path) = std::env::var("EFT_SHOT") else {
        return;
    };
    *frames += 1;

    if let Some(target) = std::env::var("EFT_SHOT_FRAME").ok().and_then(|s| s.trim().parse::<u32>().ok()) {
        // Absolute-frame mode (unchanged): a scripted in-place swap settles by a fixed frame.
        if *frames < target {
            return;
        }
    } else {
        // Load-aware mode (default): pack present, file load done, and the GPU build finished
        // (GpuLoadSignal stays in_progress across the whole texcache+geometry build). Absent under
        // the M0/std path -> fall back to "pack present + not loading".
        let loaded = pack.is_some()
            && pending.loading().is_none()
            && gpu_load.as_ref().map(|s| !s.in_progress()).unwrap_or(true);
        if !loaded {
            *settle = 0;
            return;
        }
        *settle += 1;
        let settle_target: i32 =
            std::env::var("EFT_SHOT_SETTLE").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(30);
        if *settle < settle_target {
            return;
        }
    }

    use bevy::render::view::screenshot::{save_to_disk, Screenshot};
    let exit_after = std::env::var("EFT_HIDDEN").map(|v| v.trim() == "1").unwrap_or(false)
        || std::env::var("EFT_SHOT_EXIT").map(|v| v.trim() == "1").unwrap_or(false);
    if exit_after {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_screenshot_then_exit(path.clone()));
    } else {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
    }
    info!(
        "auto-screenshot -> {path} (frame {}, exit_after={exit_after})",
        *frames
    );
    *done = true;
}

fn save_screenshot_then_exit(
    path: String,
) -> impl FnMut(
    On<bevy::render::view::screenshot::ScreenshotCaptured>,
    MessageWriter<bevy::app::AppExit>,
) {
    use bevy::render::view::screenshot::save_to_disk;
    let mut save = save_to_disk(path);
    move |captured, mut exit| {
        save(captured);
        exit.write(bevy::app::AppExit::Success);
    }
}

/// The instant benchmark sampling began (seconds since app start), or `None` while still settling.
///
/// SHARED with `debug_bench_camera` so a scripted camera path starts at the same moment sampling
/// does. Driving the path from `time.elapsed_secs()` instead makes the phase depend on how long the
/// map took to load, and load time varies per run (a cold pipeline cache cost woods 32.3 s against
/// 17.6 s warm). Two configs then sample different stretches of the path over terrain of different
/// density, which is not a comparison at all: it once put "SSAO off" 1.6 ms SLOWER than SSAO on.
#[derive(Resource, Default)]
struct BenchSampleStart(Option<f32>);

/// Steady state is not a frame COUNT. 90 frames is ~1 s at 90 fps but ~4 s at 22 fps, so the slower
/// the config the longer it waited — the gate itself was biased. And `GpuLoadSignal` clearing only
/// means the texcache+geometry build finished; render pipelines still compile on each pass's first
/// draw and the frame time is visibly unsettled after it. Require a wall-clock floor AND a rolling
/// window whose spread has collapsed, capped so a genuinely unstable scene still reports.
const BENCH_SETTLE_MIN_S: f32 = 3.0;
const BENCH_SETTLE_MAX_S: f32 = 30.0;
const BENCH_STABLE_FRAMES: usize = 60;
/// p95/p50 - 1 within the window. Compilation hitches and streaming spikes blow the tail out well
/// past this; a settled scene on a moving camera sits comfortably under it.
const BENCH_STABLE_SPREAD: f32 = 0.25;

/// EFT_BENCH=<seconds>: benchmark mode. Once the pack load has finished AND the frame time has
/// actually settled, record EVERY frame's CPU delta for the given window, print a one-line stats
/// dump (avg/p50/p95/p99/max ms + fps) and exit 0 so scripted runs end cleanly. Pair with
/// EFT_UNCAPPED=1 (vsync off) and EFT_POSE / EFT_ORBIT / EFT_FLY for repeatable scenarios.
///
/// With a moving camera, make the sample window a whole multiple of the path period, or the window
/// covers an arbitrary slice of the route and configs stop being comparable.
fn bench_stats(
    time: Res<Time>,
    pending: Res<PendingMapLoad>,
    gpu_load: Option<Res<render::GpuLoadSignal>>,
    pack: Option<Res<LoadedPack>>,
    mut samples: Local<Vec<f32>>,
    mut warm: Local<Vec<f32>>,
    mut start: ResMut<BenchSampleStart>,
) {
    let Some(secs) = std::env::var("EFT_BENCH").ok().and_then(|s| s.trim().parse::<f32>().ok())
    else {
        return;
    };
    let loaded = pack.is_some()
        && pending.loading().is_none()
        && gpu_load.as_ref().map(|s| !s.in_progress()).unwrap_or(true);
    if !loaded {
        warm.clear();
        start.0 = None;
        return;
    }
    let dt_ms = time.delta_secs() * 1000.0;
    if start.0.is_none() {
        warm.push(dt_ms);
        let elapsed: f32 = warm.iter().sum::<f32>() / 1000.0;
        if elapsed < BENCH_SETTLE_MIN_S {
            return;
        }
        // Spread over the most recent window only — early compilation hitches must not keep the
        // gate shut forever once the scene has actually calmed down.
        let stable = if warm.len() >= BENCH_STABLE_FRAMES {
            let mut w: Vec<f32> = warm[warm.len() - BENCH_STABLE_FRAMES..].to_vec();
            w.sort_by(|a, b| a.total_cmp(b));
            let p50 = w[w.len() / 2];
            let p95 = w[((w.len() - 1) as f32 * 0.95) as usize];
            p50 > 0.0 && (p95 / p50 - 1.0) < BENCH_STABLE_SPREAD
        } else {
            false
        };
        if !stable && elapsed < BENCH_SETTLE_MAX_S {
            return;
        }
        start.0 = Some(time.elapsed_secs());
        eprintln!(
            "[bench] settled after {elapsed:.1}s of warm-up ({} frames, stable={stable}) — \
             sampling {secs}s from here",
            warm.len()
        );
        // Sample from the NEXT frame: this one still contains the warm-up's last delta, and the
        // camera path is only now being anchored to t=0.
        return;
    }
    samples.push(dt_ms);
    let total: f32 = samples.iter().sum::<f32>() / 1000.0;
    if total >= secs {
        let mut s = samples.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let n = s.len();
        let pct = |p: f32| s[(((n - 1) as f32) * p) as usize];
        let avg = s.iter().sum::<f32>() / n as f32;
        let line = format!(
            "[bench] frames={n} secs={total:.1} avg={avg:.3}ms fps={:.1} p50={:.3} p95={:.3} p99={:.3} max={:.3}",
            1000.0 / avg,
            pct(0.50),
            pct(0.95),
            pct(0.99),
            pct(1.0)
        );
        info!("{line}");
        eprintln!("{line}"); // bypass the subscriber too — the run exits immediately after
        std::process::exit(0);
    }
}

/// Arm auto-exposure once the scene on screen actually REPRESENTS the map.
///
/// Adaptation is relative to a reference log-luminance, and the reference is worthless if it is
/// latched from a load-time frame: during streaming the framebuffer holds a partial scene, so the
/// reference encodes that instead of the map. Symptom, reported from a real session: the image looked
/// correct on arrival and then brightened the instant the camera first moved, because that was the
/// first frame whose luminance differed from the bogus reference.
///
/// The gate is the SAME one `bench_stats` settles on (pack resident, no pending load, GPU build
/// finished), plus a short frame hold so the first post-load frames -- which still upload textures --
/// cannot be the reference either. Disarms on any new load so a map switch re-latches.
fn arm_auto_exposure(
    pending: Res<PendingMapLoad>,
    gpu_load: Option<Res<render::GpuLoadSignal>>,
    pack: Option<Res<LoadedPack>>,
    settings: Option<ResMut<render::GfxSettings>>,
    mut held: Local<u32>,
) {
    let Some(mut s) = settings else { return };
    let loaded = pack.is_some()
        && pending.loading().is_none()
        && gpu_load.as_ref().map(|sig| !sig.in_progress()).unwrap_or(true);
    if !loaded {
        *held = 0;
        if s.exposure_armed {
            s.exposure_armed = false; // a new load invalidates the reference
        }
        return;
    }
    // ~0.5 s at 60 fps. Long enough for the texcache to stop landing, short enough that the user
    // never waits on it.
    if *held < 30 {
        *held += 1;
        return;
    }
    if !s.exposure_armed {
        s.exposure_armed = true;
    }
}

/// EFT_GPU_TIMING=1: print smoothed per-pass GPU times once a second. The names are the
/// diagnostic spans recorded in the render nodes; the bench harness greps `[gpu]` lines.
fn print_gpu_timing(
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    time: Res<Time>,
    mut acc: Local<f32>,
) {
    *acc += time.delta_secs();
    if *acc < 1.0 {
        return;
    }
    *acc = 0.0;
    let mut parts: Vec<String> = Vec::new();
    for d in diagnostics.iter() {
        let path = d.path().as_str();
        let Some(v) = d.smoothed() else { continue };
        // Our own nodes always print; Bevy's pass spans (main_opaque_pass_3d & co) print when
        // they cost real time — without them the report has a hole exactly where the frame goes.
        // CPU spans matter as much as GPU: a pass encoding thousands of items burns main-thread
        // time that no elapsed_gpu number shows.
        if !path.contains("eft") && v < 0.25 {
            continue;
        }
        parts.push(format!("{}={:.3}ms", path.trim_start_matches("render/"), v));
    }
    if !parts.is_empty() {
        parts.sort();
        eprintln!("[gpu] {}", parts.join("  "));
    }
}

/// Scripted benchmark cameras (moving-camera load: culling churn / LOD swaps / uploads).
///   EFT_ORBIT="cx,cy,cz,radius,height,degps" — circle the target point, looking at it.
///   EFT_FLY="x1,y1,z1>x2,y2,z2@secs"         — ping-pong a straight path, looking forward.
/// Runs in PostUpdate before transform propagation so it deterministically overrides the
/// fly-cam's Update-stage writes.
fn debug_bench_camera(
    time: Res<Time>,
    start: Res<BenchSampleStart>,
    mut q: Query<&mut Transform, With<render::CullCamera>>,
) {
    let Ok(mut tf) = q.single_mut() else { return };
    // Under EFT_BENCH the path clock is anchored to the start of SAMPLING, so every config flies the
    // identical route over the identical ground. Before that instant the camera is parked at the
    // path's t=0 pose, so the warm-up settles where sampling begins and there is no teleport (and
    // therefore no culling-churn spike) on the first measured frame. Without EFT_BENCH — an
    // interactive EFT_FLY/EFT_ORBIT — the clock is just app time, as before.
    let benching = std::env::var_os("EFT_BENCH").is_some();
    let clock = if benching {
        match start.0 {
            Some(t0) => time.elapsed_secs() - t0,
            None => 0.0,
        }
    } else {
        time.elapsed_secs()
    };
    if let Ok(spec) = std::env::var("EFT_ORBIT") {
        let v: Vec<f32> = spec.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 6 {
            let ang = (clock * v[5]).to_radians();
            let target = Vec3::new(v[0], v[1], v[2]);
            let pos = target + Vec3::new(v[3] * ang.cos(), v[4], v[3] * ang.sin());
            *tf = Transform::from_translation(pos).looking_at(target, Vec3::Y);
            return;
        }
    }
    if let Ok(spec) = std::env::var("EFT_FLY") {
        if let Some((ab, secs)) = spec.rsplit_once('@') {
            if let Some((a, b)) = ab.split_once('>') {
                let pa: Vec<f32> = a.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                let pb: Vec<f32> = b.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                let dur: f32 = secs.trim().parse().unwrap_or(10.0);
                if pa.len() == 3 && pb.len() == 3 && dur > 0.0 {
                    let (a, b) = (Vec3::from_slice(&pa), Vec3::from_slice(&pb));
                    let t = clock / dur;
                    let (from, to) = if (t as i32) % 2 == 0 { (a, b) } else { (b, a) };
                    let p = from.lerp(to, t.fract());
                    *tf = Transform::from_translation(p).looking_at(to, Vec3::Y);
                }
            }
        }
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

#[cfg(test)]
mod startup_pose_tests {
    use super::*;

    #[test]
    fn parses_exact_screenshot_pose() {
        let pose = parse_eft_pose(" 12.5, -4, 91.25, 135, -17.5 ").expect("valid pose");
        assert_eq!(pose.position, Vec3::new(12.5, -4.0, 91.25));
        assert!((pose.yaw - 135.0_f32.to_radians()).abs() < 1e-6);
        assert!((pose.pitch - (-17.5_f32).to_radians()).abs() < 1e-6);
    }

    #[test]
    fn rejects_malformed_or_non_finite_pose() {
        assert!(parse_eft_pose("1,2,3,4").is_none());
        assert!(parse_eft_pose("1,bad,3,4,5").is_none());
        assert!(parse_eft_pose("1,2,3,NaN,5").is_none());
    }
}
