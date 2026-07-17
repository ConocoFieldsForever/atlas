//! pathfind.rs — in-process routing over the baked nav grid (see `nav.rs`).
//!
//! REPLACES the old external NVIDIA-Warp GPU server (`:8091` GraphQL). Routing now runs on the CPU,
//! in-process, via `crate::nav::NavGrid` — so it works on EVERY GPU (or none), ships inside the exe,
//! and needs no Python/CUDA/server process. A `RouteRequest` (from a UI button, marker click, or the
//! Tasks panel) kicks off ONE async task on the compute pool (so a big chain never hitches the render
//! frame); the returned floor-snapped polyline is drawn with immediate-mode `Gizmos` until a new or
//! empty request replaces it.
//!
//! `PathfindServer` / `ServerCmd` / `ServerStatus` are kept for UI compatibility but no longer mean an
//! external process — `Running` now just means "the nav grid for this map is loaded" (routing ready).

use crate::nav::{NavGrid, Scratch};
use crate::render::{CullCamera, LoadedPack};
use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};
use std::sync::Arc;

/// Ask to route from `start` (or the placed pin / camera if `None`) through `dests`. EMPTY `dests`
/// clears the route. Sent by the UI via `MessageWriter<RouteRequest>` (Bevy 0.17 messages).
#[derive(Message, Clone, Default)]
pub struct RouteRequest {
    pub start: Option<Vec3>,
    pub dests: Vec<Vec3>,
    /// true -> cheapest visiting order (`chain`); false -> keep the given order (`tour`).
    pub optimize_order: bool,
    /// true -> the dests are ALTERNATIVES: route to ONLY the one cheapest to reach by foot
    /// (one A* per candidate, keep the shortest) instead of visiting all of them.
    pub nearest_of: bool,
    /// Display labels aligned with `dests` (may be empty = unlabeled). The label of the dest the
    /// route actually ends at is echoed back in [`RouteResult::dest_label`] for the UI.
    pub labels: Vec<String>,
}

#[derive(Clone, PartialEq)]
pub enum RouteStatus {
    Idle,
    Pending,
    Ok,
    Error(String),
}

/// The current route (flattened, floor-snapped polyline) + its status, for drawing + the UI readout.
#[derive(Resource)]
pub struct RouteResult {
    pub points: Vec<Vec3>,
    pub dist: f32,
    pub status: RouteStatus,
    /// Display label of the destination this route ends at (from `RouteRequest::labels`), so the
    /// UI can say WHERE the route goes and highlight the matching row. None = unlabeled request.
    pub dest_label: Option<String>,
}
impl Default for RouteResult {
    fn default() -> Self {
        Self { points: Vec::new(), dist: 0.0, status: RouteStatus::Idle, dest_label: None }
    }
}

/// The loaded nav grid for the CURRENT map. `None` = this pack has no baked nav -> routing off.
#[derive(Resource, Default)]
pub struct Nav(pub Option<Arc<NavGrid>>);

#[derive(Resource, Default)]
struct PathfindTask(Option<Task<Result<(Vec<Vec3>, f32, Option<String>), String>>>);

/// The player's placed "you are here" start. `None` -> routes fall back to the camera (which, in
/// walk mode, IS the player). Set by clicking the map while [`PlaceMode`] is armed (the Navigation
/// tab's PLACE ON MAP button; pick.rs does the raycast). Drawn as a gold pin.
#[derive(Resource, Default)]
pub struct StartPoint(pub Option<Vec3>);

/// Armed = the next left-click on the map places [`StartPoint`] (single-shot; Esc cancels). Set by
/// the Navigation tab's PLACE ON MAP button, consumed by pick.rs.
#[derive(Resource, Default)]
pub struct PlaceMode(pub bool);

/// Kept for UI compatibility. `Running` = nav grid loaded (routing available); `Stopped` = none.
/// (`Starting` is unused now — there is no external process to warm up.)
#[derive(Clone, PartialEq, Default)]
pub enum ServerStatus {
    #[default]
    Stopped,
    Starting,
    Running,
}

/// UI -> nav control. `Start` (re)loads the pack's nav grid; `Stop` unloads it. No process anymore.
#[derive(Message)]
pub enum ServerCmd {
    Start,
    Stop,
}

#[derive(Resource, Default)]
pub struct PathfindServer {
    pub status: ServerStatus,
}
impl PathfindServer {
    /// No external process to reap anymore (routing is in-process) — kept so existing callers (the
    /// map-switch teardown) compile unchanged.
    pub fn stop_owned_child(&mut self) {}
}

pub struct PathfindPlugin;
impl Plugin for PathfindPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<RouteRequest>()
            .add_message::<ServerCmd>()
            .init_resource::<RouteResult>()
            .init_resource::<PathfindTask>()
            .init_resource::<PathfindServer>()
            .init_resource::<Nav>()
            .init_resource::<StartPoint>()
            .init_resource::<PlaceMode>()
            .add_systems(
                Update,
                // chained: nav-load -> stale-clear -> scripted-route -> dispatch -> poll -> draw.
                (
                    manage_nav,
                    clear_route_on_start_move,
                    debug_route,
                    dispatch_route,
                    poll_route,
                    draw_route,
                    draw_start,
                )
                    .chain(),
            );
    }
}

/// Load the pack's nav grid once it appears; `ServerCmd` re-loads / unloads it (UI Start/Stop).
fn manage_nav(
    mut ev: MessageReader<ServerCmd>,
    pack: Option<Res<LoadedPack>>,
    mut nav: ResMut<Nav>,
    mut server: ResMut<PathfindServer>,
    mut tried: Local<bool>,
) {
    if !*tried {
        if let Some(p) = pack.as_ref() {
            *tried = true;
            load_nav(&p.0.root, &mut nav, &mut server);
        }
    }
    for cmd in ev.read() {
        match cmd {
            ServerCmd::Start => {
                if let Some(p) = pack.as_ref() {
                    load_nav(&p.0.root, &mut nav, &mut server);
                }
            }
            ServerCmd::Stop => {
                nav.0 = None;
                server.status = ServerStatus::Stopped;
            }
        }
    }
}

fn load_nav(root: &std::path::Path, nav: &mut Nav, server: &mut PathfindServer) {
    match NavGrid::load(root) {
        Some(g) => {
            nav.0 = Some(Arc::new(g));
            server.status = ServerStatus::Running;
        }
        None => {
            nav.0 = None;
            server.status = ServerStatus::Stopped;
            info!("nav: no nav grid in this pack — routing unavailable (bake_nav runs in the map build)");
        }
    }
}

/// Headless-QA aid: `EFT_ROUTE="x,y,z;x,y,z;..."` fires ONE route request a few frames in (first
/// point = start, rest = dests) so a screenshot can show a real drawn route without clicking.
fn debug_route(
    mut frame: Local<u32>,
    mut done: Local<bool>,
    mut w: MessageWriter<RouteRequest>,
) {
    if *done {
        return;
    }
    *frame += 1;
    if *frame < 20 {
        return;
    }
    *done = true;
    let Ok(spec) = std::env::var("EFT_ROUTE") else {
        return;
    };
    let pts: Vec<Vec3> = spec
        .split(';')
        .filter_map(|p| {
            let c: Vec<f32> = p.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
        })
        .collect();
    if pts.len() < 2 {
        return;
    }
    w.write(RouteRequest {
        start: Some(pts[0]),
        dests: pts[1..].to_vec(),
        optimize_order: true,
        ..Default::default()
    });
    info!("pathfind: EFT_ROUTE debug route requested ({} points)", pts.len());
}

/// On a request, kick off ONE async CPU route (replacing any in-flight one). Empty dests = clear.
fn dispatch_route(
    mut ev: MessageReader<RouteRequest>,
    nav: Res<Nav>,
    start_pt: Res<StartPoint>,
    cam: Query<&GlobalTransform, With<CullCamera>>,
    mut task: ResMut<PathfindTask>,
    mut result: ResMut<RouteResult>,
) {
    let Some(req) = ev.read().last().cloned() else {
        return;
    };
    if req.dests.is_empty() {
        task.0 = None;
        result.points.clear();
        result.dist = 0.0;
        result.dest_label = None;
        result.status = RouteStatus::Idle;
        return;
    }
    let Some(grid) = nav.0.clone() else {
        result.points.clear();
        result.dist = 0.0;
        result.status =
            RouteStatus::Error("no route data for this map (nav is baked during the map build)".to_string());
        return;
    };
    // Start = the explicit request start, else the placed "you are here" pin, else the camera
    // (which IS the player in walk mode). snap_start then drops it onto the nearest walkable floor.
    let start = req
        .start
        .or(start_pt.0)
        .or_else(|| cam.single().ok().map(|t| t.translation()))
        .unwrap_or(Vec3::ZERO);
    let dests = req.dests.clone();
    let labels = req.labels.clone();
    let optimize = req.optimize_order;
    let nearest_of = req.nearest_of;
    result.status = RouteStatus::Pending;
    // Route on a compute-pool thread — off the render loop; dropping the old task drops its result.
    let t = AsyncComputeTaskPool::get().spawn(async move {
        let mut s = Scratch::new(grid.nodes());
        let lbl = |i: usize| labels.get(i).cloned();
        let routed: Option<((Vec<Vec3>, f32), Option<String>)> = if dests.len() == 1 {
            grid.path(start, dests[0], &mut s).map(|r| (r, lbl(0)))
        } else if nearest_of {
            // The dests are ALTERNATIVES: one A* per candidate, keep the shortest walkable.
            let mut best: Option<((Vec<Vec3>, f32), Option<String>)> = None;
            for (i, d) in dests.iter().enumerate() {
                if let Some(r) = grid.path(start, *d, &mut s) {
                    if best.as_ref().is_none_or(|(b, _)| r.1 < b.1) {
                        best = Some((r, lbl(i)));
                    }
                }
            }
            best
        } else if optimize {
            grid.chain(start, &dests, &mut s).map(|r| (r, None))
        } else {
            let mut pts = Vec::with_capacity(dests.len() + 1);
            pts.push(start);
            pts.extend_from_slice(&dests);
            grid.tour(&pts, &mut s).map(|r| (r, None))
        };
        match routed {
            Some(((pts, dist), label)) => {
                Ok((crate::nav::simplify(&pts, grid.res * 0.4), dist, label))
            }
            None => Err("no walkable path found".to_string()),
        }
    });
    task.0 = Some(t);
}

/// Moving/placing/removing the "you are here" pin invalidates any drawn route (it started from the
/// OLD position) — clear it instead of leaving a stale, now-wrong polyline + distance on screen.
/// Runs before dispatch so a same-frame new request still goes through.
fn clear_route_on_start_move(
    start_pt: Res<StartPoint>,
    mut task: ResMut<PathfindTask>,
    mut result: ResMut<RouteResult>,
) {
    if !start_pt.is_changed() || start_pt.is_added() {
        return;
    }
    if result.status != RouteStatus::Idle || task.0.is_some() {
        task.0 = None;
        result.points.clear();
        result.dist = 0.0;
        result.dest_label = None;
        result.status = RouteStatus::Idle;
    }
}

/// Poll the in-flight task; when it finishes, publish the polyline (or the error) to `RouteResult`.
fn poll_route(mut task: ResMut<PathfindTask>, mut result: ResMut<RouteResult>) {
    let Some(t) = task.0.as_mut() else {
        return;
    };
    if let Some(res) = block_on(future::poll_once(t)) {
        task.0 = None;
        match res {
            Ok((points, dist, label)) => {
                info!("pathfind: route ok — {} pts, {:.1} m", points.len(), dist);
                result.points = points;
                result.dist = dist;
                result.dest_label = label;
                result.status = RouteStatus::Ok;
            }
            Err(e) => {
                warn!("pathfind: {e}");
                result.points.clear();
                result.dist = 0.0;
                result.dest_label = None;
                result.status = RouteStatus::Error(e);
            }
        }
    }
}

/// Draw the placed start as a gold "you are here" pin (stem + head + base), distinct from the cyan
/// route. No-op when unplaced (routes then start at the camera).
fn draw_start(mut gizmos: Gizmos, start_pt: Res<StartPoint>) {
    let Some(p) = start_pt.0 else {
        return;
    };
    let gold = Color::srgb(1.0, 0.82, 0.2);
    let top = p + Vec3::Y * 2.4;
    gizmos.line(p, top, gold);
    gizmos.sphere(Isometry3d::from_translation(top), 0.45, gold);
    gizmos.sphere(Isometry3d::from_translation(p), 0.2, gold);
}

/// Draw the current route each frame (immediate-mode; the polyline is static once computed):
/// a bright cyan line lifted just off the floor + a node dot at each snapped waypoint.
fn draw_route(mut gizmos: Gizmos, result: Res<RouteResult>) {
    if result.points.len() < 2 {
        return;
    }
    const LIFT: Vec3 = Vec3::new(0.0, 0.3, 0.0);
    gizmos.linestrip(
        result.points.iter().map(|p| *p + LIFT),
        Color::srgb(0.25, 0.9, 1.0),
    );
    for p in &result.points {
        gizmos.sphere(
            Isometry3d::from_translation(*p + LIFT),
            0.35,
            Color::srgb(0.1, 1.0, 0.6),
        );
    }
}
