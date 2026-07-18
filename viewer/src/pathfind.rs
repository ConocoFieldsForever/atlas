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

/// One computed route VARIANT ("Direct" / "Cautious" / "Wide berth") — same start + destination,
/// different avoidance weight. `dist` is real walked metres (penalties shape the path, not the number).
#[derive(Clone)]
pub struct RouteOption {
    pub name: &'static str,
    pub points: Vec<Vec3>,
    pub dist: f32,
}

/// The current route + its status, for drawing + the UI readout. `options` holds every computed
/// variant; `selected` picks the one drawn bright (the others draw as dim alternates).
/// `points`/`dist` MIRROR the selected option so simple readouts (tasks tab) stay one-field.
#[derive(Resource)]
pub struct RouteResult {
    pub points: Vec<Vec3>,
    pub dist: f32,
    pub status: RouteStatus,
    /// Display label of the destination this route ends at (from `RouteRequest::labels`), so the
    /// UI can say WHERE the route goes and highlight the matching row. None = unlabeled request.
    pub dest_label: Option<String>,
    pub options: Vec<RouteOption>,
    pub selected: usize,
}
impl Default for RouteResult {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            dist: 0.0,
            status: RouteStatus::Idle,
            dest_label: None,
            options: Vec::new(),
            selected: 0,
        }
    }
}
impl RouteResult {
    /// Select option `i` and mirror its polyline/length into `points`/`dist`.
    pub fn select(&mut self, i: usize) {
        if let Some(o) = self.options.get(i) {
            self.selected = i;
            self.points = o.points.clone();
            self.dist = o.dist;
        }
    }
    fn clear(&mut self) {
        self.points.clear();
        self.dist = 0.0;
        self.dest_label = None;
        self.options.clear();
        self.selected = 0;
        self.status = RouteStatus::Idle;
    }
}

/// Route-planning options: soft-avoid danger areas (boss/PMC/scav spawn zones). When any is on,
/// routing computes Direct + Cautious + Wide-berth variants so the cost of caution is visible.
/// Persisted for the session; `EFT_AVOID=boss,pmc,scav` seeds it (screenshots / power users).
#[derive(Resource)]
pub struct RouteOpts {
    pub avoid_boss: bool,
    pub avoid_pmc: bool,
    pub avoid_scav: bool,
    /// Animate the A* search wavefront converging on the next single-destination route.
    pub visualize: bool,
}
impl Default for RouteOpts {
    fn default() -> Self {
        let s = std::env::var("EFT_AVOID").unwrap_or_default();
        let has = |k: &str| s.split(',').any(|t| t.trim().eq_ignore_ascii_case(k));
        Self {
            avoid_boss: has("boss"),
            avoid_pmc: has("pmc"),
            avoid_scav: has("scav"),
            visualize: std::env::var("EFT_VIZ").map(|v| v.trim() == "1").unwrap_or(false),
        }
    }
}

/// The recorded A* wavefront for the live search visualization: each closed node's position + its
/// g-distance (ascending). `draw_trace` reveals them over ~1.6 s coloured by distance so you watch
/// the flood expand and converge on the destination.
#[derive(Resource, Default)]
pub struct NavTrace {
    nodes: Vec<(Vec3, f32)>,
    max_g: f32,
    start_t: f32,
    playing: bool,
}

/// The loaded nav grid for the CURRENT map. `None` = this pack has no baked nav -> routing off.
#[derive(Resource, Default)]
pub struct Nav(pub Option<Arc<NavGrid>>);

#[derive(Resource, Default)]
struct PathfindTask(Option<Task<Result<(Vec<RouteOption>, Option<String>, Vec<(Vec3, f32)>), String>>>);

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
            .init_resource::<RouteOpts>()
            .init_resource::<NavTrace>()
            .add_systems(
                Update,
                // chained: nav-load -> stale-clear -> scripted-route -> dispatch -> poll -> draw.
                (
                    manage_nav,
                    clear_route_on_start_move,
                    debug_route,
                    dispatch_route,
                    poll_route,
                    draw_trace,
                    draw_route,
                    draw_start,
                )
                    .chain(),
            )
            // In-place map swap: drop the old pack's routing state (manage_nav reloads the new nav).
            .add_systems(
                Update,
                teardown_nav.run_if(resource_changed::<crate::render::MapEpoch>),
            );
    }
}

/// Load the pack's nav grid whenever the map epoch advances (initial load + every in-place swap);
/// `ServerCmd` re-loads / unloads it (UI Start/Stop). Epoch-tracked instead of a one-shot latch so
/// an in-place `.eftpack` swap reloads the NEW pack's nav grid.
fn manage_nav(
    mut ev: MessageReader<ServerCmd>,
    pack: Option<Res<LoadedPack>>,
    epoch: Res<crate::render::MapEpoch>,
    mut nav: ResMut<Nav>,
    mut server: ResMut<PathfindServer>,
    mut loaded_epoch: Local<Option<u64>>,
) {
    if *loaded_epoch != Some(epoch.0) {
        if let Some(p) = pack.as_ref() {
            *loaded_epoch = Some(epoch.0);
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

/// In-place map swap: drop the previous pack's routing state so a stale route/pin/plan can't linger
/// on the new map (or an in-flight A* publish an old-map polyline after the swap). `manage_nav`
/// reloads the new pack's nav grid separately (epoch-tracked).
fn teardown_nav(
    mut task: ResMut<PathfindTask>,
    mut result: ResMut<RouteResult>,
    mut trace: ResMut<NavTrace>,
    mut start: ResMut<StartPoint>,
    mut place: ResMut<PlaceMode>,
) {
    task.0 = None; // cancel the in-flight async A* (captured the OLD grid)
    result.clear();
    trace.nodes.clear();
    trace.playing = false;
    start.0 = None;
    place.0 = false;
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
        // A single destination needs no ordering (and staying un-optimized lets EFT_VIZ exercise the
        // live search wavefront); multi-stop still optimizes the visiting order.
        optimize_order: pts.len() > 2,
        ..Default::default()
    });
    info!("pathfind: EFT_ROUTE debug route requested ({} points)", pts.len());
}

/// Danger radius (m) per avoided spawn kind — the soft-penalty zone around each marker.
const AVOID_R_BOSS: f32 = 45.0;
const AVOID_R_PMC: f32 = 32.0;
const AVOID_R_SCAV: f32 = 24.0;
/// Avoidance strengths for the two cautious variants (extra metres-equivalent per metre walked at
/// a zone centre): "Cautious" takes modest detours; "Wide berth" strongly refuses danger zones.
const AVOID_W_CAUTIOUS: f32 = 4.0;
const AVOID_W_WIDE: f32 = 12.0;

/// On a request, kick off ONE async CPU route (replacing any in-flight one). Empty dests = clear.
/// When any avoid option is on, computes Direct + Cautious + Wide-berth VARIANTS (the avoid points
/// come from the live boss/PMC/scav spawn markers) so the cost of caution is visible per route.
#[allow(clippy::too_many_arguments)]
fn dispatch_route(
    mut ev: MessageReader<RouteRequest>,
    nav: Res<Nav>,
    start_pt: Res<StartPoint>,
    opts: Res<RouteOpts>,
    cam: Query<&GlobalTransform, With<CullCamera>>,
    spawns: Query<(&crate::poi::PoiLayer, &GlobalTransform), Without<crate::poi::ZoneWall>>,
    mut task: ResMut<PathfindTask>,
    mut result: ResMut<RouteResult>,
) {
    let Some(req) = ev.read().last().cloned() else {
        return;
    };
    if req.dests.is_empty() {
        task.0 = None;
        result.clear();
        return;
    }
    let Some(grid) = nav.0.clone() else {
        result.clear();
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
    // Danger points for the enabled avoid options, from the live spawn markers (positions exist
    // whether or not their overlay layers are shown).
    let mut avoid_pts: Vec<(Vec3, f32)> = Vec::new();
    if opts.avoid_boss || opts.avoid_pmc || opts.avoid_scav {
        use crate::poi::PoiLayer;
        for (l, gt) in &spawns {
            let r = match l {
                PoiLayer::Boss if opts.avoid_boss => AVOID_R_BOSS,
                PoiLayer::PmcSpawn if opts.avoid_pmc => AVOID_R_PMC,
                PoiLayer::ScavSpawn if opts.avoid_scav => AVOID_R_SCAV,
                _ => continue,
            };
            avoid_pts.push((gt.translation(), r));
        }
    }
    let dests = req.dests.clone();
    let labels = req.labels.clone();
    let optimize = req.optimize_order;
    let nearest_of = req.nearest_of;
    // Only the single-destination Direct leg records a wavefront — the flood is meaningless for a
    // multi-stop tour and adds cost to every leg.
    let visualize = opts.visualize && dests.len() == 1 && nearest_of == false && !optimize;
    result.status = RouteStatus::Pending;
    // Route on a compute-pool thread — off the render loop; dropping the old task drops its result.
    let t = AsyncComputeTaskPool::get().spawn(async move {
        let mut s = Scratch::new(grid.nodes());
        let lbl = |i: usize| labels.get(i).cloned();
        // One variant under a given avoid field. Multi-stop queries (chain/tour) stay single-plan;
        // single-dest + nearest-of get the full Direct/Cautious/Wide-berth comparison.
        let mut run = |s: &mut Scratch, avoid: Option<&crate::nav::AvoidMap>| -> Option<((Vec<Vec3>, f32), Option<String>)> {
            if dests.len() == 1 {
                grid.path(start, dests[0], s, avoid).map(|r| (r, lbl(0)))
            } else if nearest_of {
                let mut best: Option<((Vec<Vec3>, f32), Option<String>)> = None;
                for (i, d) in dests.iter().enumerate() {
                    if let Some(r) = grid.path(start, *d, s, avoid) {
                        if best.as_ref().is_none_or(|(b, _)| r.1 < b.1) {
                            best = Some((r, lbl(i)));
                        }
                    }
                }
                best
            } else if optimize {
                grid.chain(start, &dests, s, avoid).map(|r| (r, None))
            } else {
                let mut pts = Vec::with_capacity(dests.len() + 1);
                pts.push(start);
                pts.extend_from_slice(&dests);
                grid.tour(&pts, s, avoid).map(|r| (r, None))
            }
        };
        let mut options: Vec<RouteOption> = Vec::new();
        let mut label: Option<String> = None;
        let mut push = |options: &mut Vec<RouteOption>, name: &'static str,
                        r: Option<((Vec<Vec3>, f32), Option<String>)>, label: &mut Option<String>| {
            if let Some(((pts, dist), l)) = r {
                // Drop a variant that found no better detour (same length as an existing one).
                if !options.iter().any(|o| (o.dist - dist).abs() < 1.0) {
                    options.push(RouteOption {
                        name,
                        points: crate::nav::simplify(&pts, grid.res * 0.4),
                        dist,
                    });
                    if label.is_none() {
                        *label = l;
                    }
                }
            }
        };
        let mut trace: Vec<(Vec3, f32)> = Vec::new();
        if visualize {
            // Instrumented A*: same result as run(), plus the recorded flood for the live viz.
            if let Some((pts, dist, tr)) = grid.path_traced(start, dests[0], &mut s, None, 4000) {
                push(&mut options, "Direct", Some(((pts, dist), lbl(0))), &mut label);
                trace = tr;
            }
        } else {
            let direct = run(&mut s, None);
            push(&mut options, "Direct", direct, &mut label);
        }
        if !avoid_pts.is_empty() {
            let cautious = grid.build_avoid(&avoid_pts, AVOID_W_CAUTIOUS);
            let r = run(&mut s, Some(&cautious));
            push(&mut options, "Cautious", r, &mut label);
            let wide = grid.build_avoid(&avoid_pts, AVOID_W_WIDE);
            let r = run(&mut s, Some(&wide));
            push(&mut options, "Wide berth", r, &mut label);
        }
        if options.is_empty() {
            Err("no walkable path found".to_string())
        } else {
            Ok((options, label, trace))
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
        result.clear();
    }
}

/// Poll the in-flight task; when it finishes, publish the polyline (or the error) to `RouteResult`.
fn poll_route(
    mut task: ResMut<PathfindTask>,
    mut result: ResMut<RouteResult>,
    mut trace: ResMut<NavTrace>,
    time: Res<Time>,
) {
    let Some(t) = task.0.as_mut() else {
        return;
    };
    if let Some(res) = block_on(future::poll_once(t)) {
        task.0 = None;
        match res {
            Ok((options, label, nodes)) => {
                info!(
                    "pathfind: route ok — {} option(s): {}",
                    options.len(),
                    options
                        .iter()
                        .map(|o| format!("{} {:.0}m", o.name, o.dist))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                result.options = options;
                result.dest_label = label;
                result.status = RouteStatus::Ok;
                result.select(0);
                // Kick off the wavefront animation (or clear a stale one when viz is off).
                if nodes.len() >= 2 {
                    trace.max_g = nodes.iter().map(|(_, g)| *g).fold(0.0_f32, f32::max);
                    trace.nodes = nodes;
                    trace.start_t = time.elapsed_secs();
                    trace.playing = true;
                } else {
                    trace.playing = false;
                    trace.nodes.clear();
                }
            }
            Err(e) => {
                warn!("pathfind: {e}");
                result.clear();
                result.status = RouteStatus::Error(e);
                trace.playing = false;
                trace.nodes.clear();
            }
        }
    }
}

/// Animate the recorded A* wavefront: reveal nodes in g-distance order over ~1.6 s, each drawn as a
/// short vertical tick coloured cyan (near the start) -> magenta (the frontier), with a brighter
/// band right at the advancing edge. Purely cosmetic and immediate-mode; stops itself once the
/// flood has fully revealed so the finished route line stands alone.
fn draw_trace(mut gizmos: Gizmos, mut trace: ResMut<NavTrace>, time: Res<Time>) {
    if !trace.playing || trace.nodes.len() < 2 || trace.max_g <= 0.0 {
        return;
    }
    const REVEAL_SECS: f32 = 1.6;
    let elapsed = time.elapsed_secs() - trace.start_t;
    // g-distance revealed so far (whole flood in REVEAL_SECS, then held one extra beat before stop).
    let reveal_g = (elapsed / REVEAL_SECS).clamp(0.0, 1.0) * trace.max_g;
    let edge_band = trace.max_g * 0.06; // width of the bright advancing-front band
    for (p, g) in &trace.nodes {
        if *g > reveal_g {
            continue;
        }
        let f = (*g / trace.max_g).clamp(0.0, 1.0); // 0 near start .. 1 at frontier
        // cyan (0.2,0.9,1.0) -> magenta (1.0,0.25,0.95)
        let mut col = Color::srgba(0.2 + f * 0.8, 0.9 - f * 0.65, 1.0 - f * 0.05, 0.42);
        let mut h = 0.35;
        if reveal_g - *g < edge_band {
            // advancing front: brighter + taller
            col = Color::srgba(0.85, 1.0, 1.0, 0.9);
            h = 0.9;
        }
        gizmos.line(*p + Vec3::Y * 0.08, *p + Vec3::Y * (0.08 + h), col);
    }
    if elapsed > REVEAL_SECS + 0.4 {
        trace.playing = false; // done — let the route line take over cleanly
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

/// Draw the current route each frame (immediate-mode; the polylines are static once computed):
///  - the SELECTED option as a marching-dash line whose colour sweeps cyan -> spring green toward
///    the destination (the dash flow gives an unmistakable direction-of-travel read),
///  - every other option as a thin, dim, static dashed alternate (click one in the panel to swap),
///  - a destination BEACON: vertical light column + two counter-phased pulsing ground rings,
///  - a small cyan start orb where the route begins.
fn draw_route(mut gizmos: Gizmos, result: Res<RouteResult>, time: Res<Time>) {
    const LIFT: Vec3 = Vec3::new(0.0, 0.35, 0.0);
    if result.options.is_empty() {
        return;
    }
    let t = time.elapsed_secs();

    // ---- dim alternates first (under the selected line) ----
    for (i, o) in result.options.iter().enumerate() {
        if i == result.selected || o.points.len() < 2 {
            continue;
        }
        draw_dashed(&mut gizmos, &o.points, LIFT, 1.2, 1.8, 0.0, |_| {
            Color::srgba(0.35, 0.65, 0.75, 0.5)
        });
    }

    // ---- selected: marching gradient dashes ----
    let sel = &result.options[result.selected];
    if sel.points.len() < 2 {
        return;
    }
    let total: f32 = sel.points.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f32>().max(1.0);
    let phase = (t * 6.0) % 3.6; // dash+gap cycle: 2.4 m dash / 1.2 m gap, flowing toward the dest
    draw_dashed(&mut gizmos, &sel.points, LIFT, 2.4, 1.2, -phase, |frac| {
        // cyan (start) -> spring green (destination)
        let a = Vec3::new(0.25, 0.90, 1.00);
        let b = Vec3::new(0.25, 1.00, 0.45);
        let c = a.lerp(b, frac);
        Color::srgb(c.x, c.y, c.z)
    });

    // ---- start orb ----
    if let Some(p0) = sel.points.first() {
        gizmos.sphere(Isometry3d::from_translation(*p0 + LIFT), 0.4, Color::srgb(0.25, 0.9, 1.0));
    }

    // ---- destination beacon ----
    if let Some(pe) = sel.points.last() {
        let green = Color::srgb(0.25, 1.0, 0.45);
        gizmos.line(*pe, *pe + Vec3::Y * 26.0, Color::srgba(0.25, 1.0, 0.45, 0.55));
        // Two counter-phased expanding rings hugging the ground.
        for k in 0..2 {
            let ph = ((t * 0.9 + k as f32 * 0.5) % 1.0).clamp(0.0, 1.0);
            let r = 0.6 + ph * 2.6;
            let alpha = (1.0 - ph) * 0.9;
            gizmos.circle(
                Isometry3d::new(*pe + Vec3::Y * 0.25, Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
                r,
                Color::srgba(0.25, 1.0, 0.45, alpha),
            );
        }
        gizmos.sphere(Isometry3d::from_translation(*pe + Vec3::Y * 0.6), 0.45, green);
    }
}

/// Walk `pts` drawing alternating dash/gap segments (`phase` shifts the pattern along the arc —
/// animate it for a marching-ants flow). `color(frac)` is sampled at the dash's arc fraction.
fn draw_dashed(
    gizmos: &mut Gizmos,
    pts: &[Vec3],
    lift: Vec3,
    dash: f32,
    gap: f32,
    phase: f32,
    color: impl Fn(f32) -> Color,
) {
    let cycle = dash + gap;
    if cycle <= 1e-3 {
        return;
    }
    // Minimum advance per iteration: at f32, `u += tiny` is a NO-OP once `tiny` drops below u's
    // ULP (~2.4e-7 at u≈2), which froze the loop (and the render thread) whenever the animated
    // phase landed within an ulp of a cycle boundary. Clamping the step kills that class of hang;
    // a 5 cm quantisation is invisible at dash scale.
    const MIN_STEP: f32 = 0.05;
    let total: f32 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f32>().max(1e-3);
    let mut s = 0.0; // arc length at segment start
    for w in pts.windows(2) {
        let seg = w[1] - w[0];
        let len = seg.length();
        if len < 1e-4 {
            continue;
        }
        let dir = seg / len;
        let mut u = 0.0;
        while u < len {
            // Position within the dash cycle at arc (s + u + phase).
            let m = (s + u + phase).rem_euclid(cycle);
            if m < dash {
                let e = (u + (dash - m).max(MIN_STEP)).min(len);
                let frac = ((s + u) / total).clamp(0.0, 1.0);
                gizmos.line(w[0] + dir * u + lift, w[0] + dir * e + lift, color(frac));
                u = e + gap.max(MIN_STEP);
            } else {
                u += (cycle - m).max(MIN_STEP); // jump to the next dash start
            }
        }
        s += len;
    }
}
