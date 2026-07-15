//! pathfind.rs — ON-DEMAND routing via the local `:8091` GPU pathfind server.
//!
//! The viewer never pathfinds itself. It POSTs a start + destination(s) to the resident NVIDIA-Warp
//! MapWorker (`pathfind_server.py`, GraphQL) ONLY when asked — a `RouteRequest` event from the UI
//! (a button / hotkey), never per-frame — so the GPU churns only per query, not while idle. The
//! request runs on an async task thread (blocking HTTP, off the render thread); the returned
//! floor-snapped polyline is drawn with immediate-mode `Gizmos` until a new/empty request replaces
//! it. Server URL from `EFT_PATHFIND_URL` (default `http://127.0.0.1:8091/graphql`); if the server
//! is down the status carries the error so the UI can say so instead of hanging.
//!
//! Query selection: 1 dest -> `path`; N dests + optimize -> `chain` (server picks the visiting
//! order); N dests keep-order -> `tour`. All map to one flattened polyline for drawing.

use crate::render::{CullCamera, LoadedPack};
use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};

/// Ask the server to route from `start` (or the camera if `None`) through `dests`. An EMPTY
/// `dests` clears the current route. Sent by the UI via `MessageWriter<RouteRequest>` (Bevy 0.17
/// renamed buffered events to "messages").
#[derive(Message, Clone)]
pub struct RouteRequest {
    pub start: Option<Vec3>,
    pub dests: Vec<Vec3>,
    /// true -> `chain` (server re-orders for shortest tour); false -> `tour` (keep the given order).
    pub optimize_order: bool,
}

#[derive(Clone, PartialEq)]
pub enum RouteStatus {
    Idle,
    Pending,
    Ok,
    Error(String),
}

/// The current route (a flattened, floor-snapped polyline) + its status, for drawing + UI readout.
#[derive(Resource)]
pub struct RouteResult {
    pub points: Vec<Vec3>,
    pub dist: f32,
    pub status: RouteStatus,
}
impl Default for RouteResult {
    fn default() -> Self {
        Self { points: Vec::new(), dist: 0.0, status: RouteStatus::Idle }
    }
}

#[derive(Resource)]
struct PathfindConfig {
    url: String,
}

#[derive(Resource, Default)]
struct PathfindTask(Option<Task<Result<(Vec<Vec3>, f32), String>>>);

pub struct PathfindPlugin;
impl Plugin for PathfindPlugin {
    fn build(&self, app: &mut App) {
        let url = std::env::var("EFT_PATHFIND_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8091/graphql".to_string());
        app.add_message::<RouteRequest>()
            .insert_resource(PathfindConfig { url })
            .init_resource::<RouteResult>()
            .init_resource::<PathfindTask>()
            .add_systems(Update, (debug_route, dispatch_route, poll_route, draw_route));
    }
}

/// dataset "interchange_v2" -> pathfind map id "interchange" (strip a `_vN` suffix).
fn map_key(dataset: &str) -> String {
    if let Some((base, ver)) = dataset.rsplit_once("_v") {
        if !ver.is_empty() && ver.chars().all(|c| c.is_ascii_digit()) {
            return base.to_string();
        }
    }
    dataset.to_string()
}

/// Headless-QA aid: `EFT_ROUTE="x,y,z;x,y,z;..."` fires ONE route request a few frames in (first
/// point = start, rest = dests) so a screenshot can show a real drawn route without clicking. Runs
/// once, then disables itself. No-op unless the env is set.
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
    });
    info!("pathfind: EFT_ROUTE debug route requested ({} points)", pts.len());
}

/// On a request, kick off ONE async HTTP query (replacing any in-flight one). Empty dests = clear.
fn dispatch_route(
    mut ev: MessageReader<RouteRequest>,
    cfg: Res<PathfindConfig>,
    pack: Option<Res<LoadedPack>>,
    cam: Query<&GlobalTransform, With<CullCamera>>,
    mut task: ResMut<PathfindTask>,
    mut result: ResMut<RouteResult>,
) {
    // Only the most recent request in the frame matters.
    let Some(req) = ev.read().last().cloned() else {
        return;
    };
    if req.dests.is_empty() {
        // clear
        task.0 = None;
        result.points.clear();
        result.dist = 0.0;
        result.status = RouteStatus::Idle;
        return;
    }
    let Some(pack) = pack else {
        return;
    };
    let start = req
        .start
        .or_else(|| cam.single().ok().map(|t| t.translation()))
        .unwrap_or(Vec3::ZERO);
    let map = map_key(&pack.0.manifest.dataset);
    let url = cfg.url.clone();
    let dests = req.dests.clone();
    let optimize = req.optimize_order;
    result.status = RouteStatus::Pending;
    // Blocking HTTP on a task thread — off the render loop; dropping the old task cancels it.
    let t = AsyncComputeTaskPool::get()
        .spawn(async move { run_query(&url, &map, start, &dests, optimize) });
    task.0 = Some(t);
}

/// Poll the in-flight task; when it finishes, publish the polyline (or the error) to `RouteResult`.
fn poll_route(mut task: ResMut<PathfindTask>, mut result: ResMut<RouteResult>) {
    let Some(t) = task.0.as_mut() else {
        return;
    };
    if let Some(res) = block_on(future::poll_once(t)) {
        task.0 = None;
        match res {
            Ok((points, dist)) => {
                result.points = points;
                result.dist = dist;
                result.status = RouteStatus::Ok;
            }
            Err(e) => {
                result.points.clear();
                result.dist = 0.0;
                result.status = RouteStatus::Error(e);
            }
        }
    }
}

/// Blocking GraphQL POST to the pathfind server. Returns the flattened polyline + walkable length.
fn run_query(
    url: &str,
    map: &str,
    start: Vec3,
    dests: &[Vec3],
    optimize: bool,
) -> Result<(Vec<Vec3>, f32), String> {
    let arr = |p: Vec3| serde_json::json!([p.x, p.y, p.z]);
    let (query, field, vars) = if dests.len() == 1 {
        (
            "query P($m:String!,$s:[Float!]!,$d:[Float!]!){ path(map:$m,start:$s,dest:$d){ ok points dist } }",
            "path",
            serde_json::json!({"m": map, "s": arr(start), "d": arr(dests[0])}),
        )
    } else if optimize {
        (
            "query C($m:String!,$s:[Float!]!,$d:[[Float!]!]!){ chain(map:$m,start:$s,dests:$d){ ok legs{points dist} dist } }",
            "chain",
            serde_json::json!({"m": map, "s": arr(start), "d": dests.iter().map(|p| arr(*p)).collect::<Vec<_>>()}),
        )
    } else {
        let mut pts: Vec<serde_json::Value> = vec![arr(start)];
        pts.extend(dests.iter().map(|p| arr(*p)));
        (
            "query T($m:String!,$p:[[Float!]!]!){ tour(map:$m,points:$p){ ok legs{points dist} dist } }",
            "tour",
            serde_json::json!({"m": map, "p": pts}),
        )
    };
    let body = serde_json::json!({"query": query, "variables": vars}).to_string();
    let resp = ureq::post(url)
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(60))
        .send_string(&body)
        .map_err(|e| format!("pathfind server unreachable at {url} — is it running? ({e})"))?;
    let text = resp
        .into_string()
        .map_err(|e| format!("read failed: {e}"))?;
    let j: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("bad JSON: {e}"))?;
    if let Some(errs) = j.get("errors") {
        return Err(format!("server error: {errs}"));
    }
    let node = &j["data"][field];
    if !node["ok"].as_bool().unwrap_or(false) {
        return Err("no walkable path found".to_string());
    }
    let dist = node["dist"].as_f64().unwrap_or(0.0) as f32;
    let mut points = Vec::new();
    if let Some(pts) = node["points"].as_array() {
        points.extend(pts.iter().map(json_vec3)); // single `path`
    } else if let Some(legs) = node["legs"].as_array() {
        for lg in legs {
            if let Some(pts) = lg["points"].as_array() {
                points.extend(pts.iter().map(json_vec3)); // `chain`/`tour` legs
            }
        }
    }
    Ok((points, dist))
}

fn json_vec3(v: &serde_json::Value) -> Vec3 {
    let a = v.as_array();
    let g = |i: usize| {
        a.and_then(|a| a.get(i))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as f32
    };
    Vec3::new(g(0), g(1), g(2))
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
