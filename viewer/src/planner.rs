//! planner.rs — the LOOT-RUN planner: a budgeted, extract-terminated loot tour optimizer.
//!
//! The problem is ORIENTEERING (prize-collecting TSP): from your position, visit the loot stops
//! that maximize expected value under a walking-distance budget, and END at an extract. Exact
//! solutions are NP-hard; this uses the classic two-phase heuristic the old web loot planner
//! validated (mapworker distance-matrix + 2-opt), adapted to run fully in-process:
//!
//!   1. CHEAPEST-INSERTION by value density (straight-line): start with [you -> best extract];
//!      repeatedly insert the candidate with the best value / marginal-detour ratio at its best
//!      slot, while the detour-corrected estimate fits the budget and the stop cap.
//!   2. 2-OPT (straight-line) to untangle crossings — cheap and removes most of the insertion
//!      artifacts before any A* runs.
//!   3. REAL LEGS: thread the chosen order through the nav grid (one A* per leg, continuing from
//!      each snapped endpoint), honoring the avoid options. If the real total blows the budget by
//!      >15%, drop the worst value/detour stop and re-thread (up to 3 repairs).
//!
//! Candidates come from the live loot markers — container `ev` estimates + priced loose loot
//! (tarkov.dev value model, min-value filtered, top-N capped). The result feeds `RouteResult`
//! (so the tour draws with the same marching-dash + variant machinery) plus a `PlanResult` stop
//! list for the panel; stop orbs draw as gold gizmos.

use crate::nav::{AvoidMap, Scratch};
use crate::pathfind::{Nav, RouteOption, RouteResult, RouteStatus};
use crate::render::CullCamera;
use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};

/// Ask for a loot-run plan. Sent by the Navigation tab's PLAN button.
#[derive(Message, Clone)]
pub struct PlanRequest {
    /// Ignore loot below this estimated rouble value.
    pub min_value: i64,
    /// Stop cap (the tour visits at most this many loot points).
    pub max_stops: usize,
    /// Total raid-time budget in seconds, including search time and extract reserve.
    pub budget_s: f32,
}

#[derive(Clone, PartialEq, Default)]
pub enum PlanStatus {
    #[default]
    Idle,
    Pending,
    Ok,
    Error(String),
}

/// One ordered stop of the planned run.
#[derive(Clone)]
pub struct PlanStop {
    pub name: String,
    pub value: i64,
    pub pos: Vec3,
    /// Real walkable metres from the previous stop (leg INTO this stop).
    pub leg: f32,
    pub loot_s: f32,
}

/// The current plan (ordered stops + totals) for the panel list; the tour polyline itself lives
/// in `RouteResult` (option "Loot run") so all route drawing/UI is reused.
#[derive(Resource, Default)]
pub struct PlanResult {
    pub status: PlanStatus,
    pub stops: Vec<PlanStop>,
    pub total_value: i64,
    pub total_dist: f32,
    pub total_time: f32,
    /// Name of the extract the run ends at.
    pub extract: String,
}

#[derive(Resource, Default)]
struct PlanTask(Option<Task<Result<Plan, String>>>);

struct Plan {
    stops: Vec<PlanStop>,
    extract: String,
    polyline: Vec<Vec3>,
    total_dist: f32,
    total_time: f32,
    total_value: i64,
}

pub struct PlannerPlugin;
impl Plugin for PlannerPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<PlanRequest>()
            .init_resource::<PlanResult>()
            .init_resource::<PlanTask>()
            .add_systems(Update, (debug_plan, dispatch_plan, poll_plan, draw_stops).chain())
            // In-place map swap: cancel the in-flight solve + clear the plan. BEFORE poll_plan so a
            // solve completing on the swap frame can't republish an old-map route/PlanResult (it
            // sees PlanTask=None). RouteResult is also cleared for order-independence.
            .add_systems(
                Update,
                teardown_plan
                    .run_if(resource_changed::<crate::render::MapEpoch>)
                    .before(poll_plan),
            );
    }
}

/// In-place map swap: cancel the in-flight orienteering solve (it captured a clone of the OLD nav
/// grid; if it completed, `poll_plan` would re-populate `PlanResult` AND overwrite `RouteResult`
/// with an old-map "Loot run" route after teardown) and clear the stale plan list.
fn teardown_plan(
    mut task: ResMut<PlanTask>,
    mut result: ResMut<PlanResult>,
    mut route: ResMut<RouteResult>,
) {
    task.0 = None;
    *result = PlanResult::default();
    // Belt-and-braces: the plan tour shares RouteResult with pathfind; clear it here too so a stale
    // "Loot run" polyline can't survive regardless of teardown_nav vs poll_plan ordering.
    route.clear();
}

/// Headless-QA aid: `EFT_PLAN="min_value,max_stops,budget_minutes"` (or `1` for
/// defaults) fires ONE plan request a few frames in so a screenshot shows a real loot run.
fn debug_plan(mut frame: Local<u32>, mut done: Local<bool>, mut w: MessageWriter<PlanRequest>) {
    if *done {
        return;
    }
    *frame += 1;
    if *frame < 25 {
        return;
    }
    *done = true;
    let Ok(spec) = std::env::var("EFT_PLAN") else {
        return;
    };
    let nums: Vec<f32> = spec.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let req = PlanRequest {
        min_value: nums.first().map(|v| *v as i64).filter(|&v| v > 1).unwrap_or(100_000),
        max_stops: nums.get(1).map(|v| *v as usize).unwrap_or(10),
        budget_s: nums.get(2).copied().unwrap_or(25.0) * 60.0,
    };
    info!(
        "planner: EFT_PLAN debug plan requested (min {}k, {} stops, {:.0} min)",
        req.min_value / 1000,
        req.max_stops,
        req.budget_s / 60.0
    );
    w.write(req);
}

/// Candidate loot point fed to the async optimizer.
#[derive(Clone)]
struct Cand {
    name: String,
    value: i64,
    score_value: f32,
    pos: Vec3,
    loot_s: f32,
}

/// Gather candidates + extracts, then solve on the compute pool.
#[allow(clippy::too_many_arguments)]
fn dispatch_plan(
    mut ev: MessageReader<PlanRequest>,
    nav: Res<Nav>,
    start_pt: Res<crate::pathfind::StartPoint>,
    opts: Res<crate::pathfind::RouteOpts>,
    cam: Query<&GlobalTransform, With<CullCamera>>,
    loot: Query<(
        &GlobalTransform,
        &crate::inspect::MarkerInfo,
        &crate::poi::MarkerValue,
        Option<&crate::loot::LootClass>,
        Option<&crate::poi::PoiLayer>,
        Option<&crate::loot::LootTime>,
        Option<&crate::poi::LootJackpot>,
    )>,
    locks: Query<(&GlobalTransform, &crate::poi::LockKeys)>,
    progress: Res<crate::progress::PlayerProgress>,
    all_marks: Query<
        (&crate::poi::PoiLayer, &GlobalTransform, &crate::inspect::MarkerInfo, Option<&crate::poi::SceneInactive>),
        Without<crate::poi::ZoneWall>,
    >,
    mut task: ResMut<PlanTask>,
    mut plan: ResMut<PlanResult>,
    mut route_result: ResMut<RouteResult>,
) {
    let Some(req) = ev.read().last().cloned() else {
        return;
    };
    if req.max_stops == 0 {
        // clear
        task.0 = None;
        *plan = PlanResult::default();
        return;
    }
    let Some(grid) = nav.0.clone() else {
        plan.status = PlanStatus::Error("no route data for this map".into());
        return;
    };
    let start = start_pt
        .0
        .or_else(|| cam.single().ok().map(|t| t.translation()))
        .unwrap_or(Vec3::ZERO);

    // ---- candidates: value-tagged loot markers (container ev + priced loose), min-filtered,
    // top-120 by value so the optimizer stays bounded on loot-dense maps (streets: 2k+ points).
    let mut cands: Vec<Cand> = loot
        .iter()
        .filter(|(_, _, v, cls, layer, _, _)| {
            v.0 >= req.min_value
                && (cls.is_some() // loot.rs container
                    || matches!(layer, Some(crate::poi::PoiLayer::LooseLoot))) // priced loose
        })
        .filter(|(gt, _, _, _, _, _, _)| {
            !locks.iter().any(|(lock_gt, keys)| {
                lock_gt.translation().distance(gt.translation()) <= 14.0
                    && !keys.0.is_empty()
                    && !keys.0.iter().any(|key| progress.owns_key(key))
            })
        })
        .map(|(gt, info, v, _, _, loot_time, jackpot)| Cand {
            name: info.title.clone(),
            value: v.0,
            score_value: v.0 as f32 * if jackpot.is_some() { 0.18 } else { 1.0 },
            pos: gt.translation(),
            loot_s: loot_time.map(|t| t.0).unwrap_or(5.0),
        })
        .collect();
    cands.sort_by(|a, b| b.value.cmp(&a.value));
    cands.truncate(120);
    if cands.is_empty() {
        plan.status = PlanStatus::Error("no loot above the value filter on this map".into());
        return;
    }

    // ---- extract candidates (active only) — the run must END somewhere safe.
    let extracts: Vec<(String, Vec3)> = all_marks
        .iter()
        .filter(|(l, _, _, inactive)| **l == crate::poi::PoiLayer::Extract && inactive.is_none())
        .map(|(_, gt, info, _)| (info.title.clone(), gt.translation()))
        .collect();
    if extracts.is_empty() {
        plan.status = PlanStatus::Error("no active extracts on this map".into());
        return;
    }

    // ---- avoid field (same options as normal routing) ----
    let mut avoid_pts: Vec<(Vec3, f32)> = Vec::new();
    if opts.avoid_boss || opts.avoid_pmc || opts.avoid_scav {
        for (l, gt, _, _) in &all_marks {
            let r = match l {
                crate::poi::PoiLayer::Boss if opts.avoid_boss => 45.0,
                crate::poi::PoiLayer::PmcSpawn if opts.avoid_pmc => 32.0,
                crate::poi::PoiLayer::ScavSpawn if opts.avoid_scav => 24.0,
                _ => continue,
            };
            avoid_pts.push((gt.translation(), r));
        }
    }

    plan.status = PlanStatus::Pending;
    route_result.status = RouteStatus::Pending;
    let (max_stops, budget) = (req.max_stops, req.budget_s.max(300.0));
    let t = AsyncComputeTaskPool::get().spawn(async move {
        let avoid = (!avoid_pts.is_empty()).then(|| grid.build_avoid(&avoid_pts, 4.0));
        solve(&grid, start, cands, extracts, max_stops, budget, avoid.as_ref())
    });
    task.0 = Some(t);
}

/// The two/three-phase orienteering heuristic (see module doc).
fn solve(
    grid: &crate::nav::NavGrid,
    start: Vec3,
    cands: Vec<Cand>,
    extracts: Vec<(String, Vec3)>,
    max_stops: usize,
    budget: f32,
    avoid: Option<&AvoidMap>,
) -> Result<Plan, String> {
    // Straight-line with a detour factor approximates walkable distance for the FAST phases;
    // ~1.35 matches sampled A*/straight ratios (open lot ~1.1, indoor ~1.7).
    const DETOUR: f32 = 1.35;
    const WALK_MPS: f32 = 1.65;
    const EXTRACT_BUFFER_S: f32 = 120.0;
    let est = |a: Vec3, b: Vec3| a.distance(b) * DETOUR;

    // ---- phase 0: ONE bounded Dijkstra flood from the start prunes unreachable candidates
    // up front. Without this, every shelf/roof loot point that isn't on the nav mesh cost a
    // full EXHAUSTIVE failed A* during threading (seconds each — the planner looked hung).
    let mut field_s = Scratch::new(grid.nodes());
    let walk_budget_m = ((budget - EXTRACT_BUFFER_S).max(60.0) * WALK_MPS).max(200.0);
    if !grid.dijkstra_field(start, walk_budget_m * 1.4, &mut field_s) {
        return Err("start is off the walkable mesh".into());
    }
    let cands: Vec<Cand> = cands
        .into_iter()
        .filter(|c| grid.field_dist(&field_s, c.pos).is_some())
        .collect();
    if cands.is_empty() {
        return Err("no reachable loot above the value filter within the budget".into());
    }
    let extracts: Vec<(String, Vec3)> = extracts
        .into_iter()
        .filter(|e| grid.field_dist(&field_s, e.1).is_some())
        .collect();
    if extracts.is_empty() {
        return Err("no reachable extract within the budget".into());
    }

    // Initial extract anchor: the one nearest the start (the final extract is re-picked after
    // the stops are chosen — a run drifting across the map should end at the FAR side's exit).
    let ex0 = extracts
        .iter()
        .min_by(|a, b| est(start, a.1).total_cmp(&est(start, b.1)))
        .cloned()
        .unwrap();

    // ---- phase 1: cheapest insertion by value density (estimates) ----
    // Node index space: 0 = start, 1..=len = stops (tour[i-1]), len+1 = extract.
    let mut tour: Vec<usize> = Vec::new(); // indices into cands
    let mut used = vec![false; cands.len()];
    let node_pos = |i: usize, tour: &[usize], ex: Vec3| -> Vec3 {
        if i == 0 {
            start
        } else if i <= tour.len() {
            cands[tour[i - 1]].pos
        } else {
            ex
        }
    };
    let mut est_total = est(start, ex0.1) / WALK_MPS + EXTRACT_BUFFER_S;
    while tour.len() < max_stops {
        let mut best: Option<(usize, usize, f32, f32)> = None; // (cand, slot, delta, score)
        for (ci, c) in cands.iter().enumerate() {
            if used[ci] {
                continue;
            }
            for slot in 0..=tour.len() {
                let a = node_pos(slot, &tour, ex0.1);
                let b = node_pos(slot + 1, &tour, ex0.1);
                let delta = (est(a, c.pos) + est(c.pos, b) - est(a, b)) / WALK_MPS + c.loot_s;
                if est_total + delta > budget {
                    continue;
                }
                // Expected value per marginal second; the floor keeps "free" on-path stops from
                // swallowing the whole cap before anything valuable gets a slot.
                let score = c.score_value / delta.max(5.0);
                if best.map_or(true, |(_, _, _, s)| score > s) {
                    best = Some((ci, slot, delta, score));
                }
            }
        }
        match best {
            Some((ci, slot, delta, _)) => {
                used[ci] = true;
                tour.insert(slot, ci);
                est_total += delta;
            }
            None => break, // nothing else fits the budget
        }
    }
    if tour.is_empty() {
        return Err("budget too small \u{2014} no loot fits before the extract".into());
    }

    // ---- phase 2: 2-opt on estimates (fixed endpoints; untangles insertion crossings) ----
    let mut improved = true;
    while improved {
        improved = false;
        let n = tour.len();
        for i in 1..n {
            for j in (i + 1)..=n {
                let old = est(node_pos(i - 1, &tour, ex0.1), node_pos(i, &tour, ex0.1))
                    + est(node_pos(j, &tour, ex0.1), node_pos(j + 1, &tour, ex0.1));
                let new = est(node_pos(i - 1, &tour, ex0.1), node_pos(j, &tour, ex0.1))
                    + est(node_pos(i, &tour, ex0.1), node_pos(j + 1, &tour, ex0.1));
                if new + 0.01 < old {
                    tour[i - 1..j].reverse();
                    improved = true;
                }
            }
        }
    }

    // ---- phase 3: real legs (A* threading) + budget repair ----
    let mut s = Scratch::new(grid.nodes());
    for _repair in 0..6 {
        if tour.is_empty() {
            return Err("no reachable loot within the budget".into());
        }
        // End extract: nearest (estimate) to the LAST stop — the run exits where it ends up.
        let last_pos = cands[*tour.last().unwrap()].pos;
        let ex = extracts
            .iter()
            .min_by(|a, b| est(last_pos, a.1).total_cmp(&est(last_pos, b.1)))
            .cloned()
            .unwrap();

        let mut cur = start;
        let mut poly: Vec<Vec3> = Vec::new();
        let mut legs: Vec<f32> = Vec::new();
        let mut total = 0.0f32;
        let mut unreachable: Option<usize> = None;
        for (k, &ci) in tour.iter().enumerate() {
            match grid.path(cur, cands[ci].pos, &mut s, avoid) {
                Some((p, d)) => {
                    if poly.is_empty() {
                        poly.extend_from_slice(&p);
                    } else {
                        poly.extend_from_slice(&p[1..]);
                    }
                    cur = *poly.last().unwrap();
                    legs.push(d);
                    total += d;
                }
                None => {
                    unreachable = Some(k);
                    break;
                }
            }
        }
        if let Some(k) = unreachable {
            tour.remove(k); // off-mesh stop (shelf/roof glitch) — drop and re-thread
            continue;
        }
        let Some((exp, exd)) = grid.path(cur, ex.1, &mut s, avoid) else {
            return Err("no walkable path to any extract".into());
        };
        total += exd;
        let loot_time: f32 = tour.iter().map(|&ci| cands[ci].loot_s).sum();
        let total_time = total / WALK_MPS + loot_time + EXTRACT_BUFFER_S;
        if total_time > budget * 1.15 && tour.len() > 1 {
            // Over budget in the real world: drop the worst expected-value-per-second stop.
            let worst = (0..tour.len())
                .min_by(|&a, &b| {
                    (cands[tour[a]].score_value / (legs[a] / WALK_MPS + cands[tour[a]].loot_s).max(1.0))
                        .total_cmp(&(cands[tour[b]].score_value / (legs[b] / WALK_MPS + cands[tour[b]].loot_s).max(1.0)))
                })
                .unwrap();
            tour.remove(worst);
            continue;
        }
        poly.extend_from_slice(&exp[1..]);
        let stops: Vec<PlanStop> = tour
            .iter()
            .zip(legs.iter())
            .map(|(&ci, &l)| PlanStop {
                name: cands[ci].name.clone(),
                value: cands[ci].value,
                pos: cands[ci].pos,
                leg: l,
                loot_s: cands[ci].loot_s,
            })
            .collect();
        let total_value = stops.iter().map(|st| st.value).sum();
        return Ok(Plan {
            stops,
            extract: ex.0,
            // `poly` is a concatenation of per-leg grid.path polylines, each ALREADY
            // wall-aware-simplified with its endpoints pinned; a second plain Douglas–Peucker over
            // the stitched line would corner-cut across the seams, so keep it verbatim.
            polyline: poly,
            total_dist: total,
            total_time,
            total_value,
        });
    }
    Err("couldn't fit a run into the budget (try fewer stops / larger budget)".into())
}

/// Publish the finished plan: the stop list into `PlanResult`, the tour polyline into
/// `RouteResult` (as the single "Loot run" option) so the marching-dash drawing + ROUTE card
/// machinery is reused verbatim.
fn poll_plan(
    mut task: ResMut<PlanTask>,
    mut plan: ResMut<PlanResult>,
    mut route_result: ResMut<RouteResult>,
) {
    let Some(t) = task.0.as_mut() else {
        return;
    };
    if let Some(res) = block_on(future::poll_once(t)) {
        task.0 = None;
        match res {
            Ok(p) => {
                info!(
                    "planner: {} stops, ~{}k value, {:.0} m / {:.1} min, exit {}",
                    p.stops.len(),
                    p.total_value / 1000,
                    p.total_dist,
                    p.total_time / 60.0,
                    p.extract
                );
                plan.stops = p.stops;
                plan.total_value = p.total_value;
                plan.total_dist = p.total_dist;
                plan.total_time = p.total_time;
                plan.extract = p.extract.clone();
                plan.status = PlanStatus::Ok;
                route_result.options = vec![RouteOption {
                    name: "Loot run",
                    points: p.polyline,
                    dist: p.total_dist,
                }];
                route_result.dest_label = Some(format!("Loot run \u{2192} {}", p.extract));
                route_result.stop_count = plan.stops.len() + 1;
                route_result.status = RouteStatus::Ok;
                route_result.select(0);
            }
            Err(e) => {
                warn!("planner: {e}");
                plan.status = PlanStatus::Error(e.clone());
                if route_result.status == RouteStatus::Pending {
                    route_result.status = RouteStatus::Error(e);
                }
            }
        }
    }
}

/// Gold orbs + a short tick over each planned stop (the ordered list lives in the panel).
fn draw_stops(mut gizmos: Gizmos, plan: Res<PlanResult>) {
    if plan.status != PlanStatus::Ok {
        return;
    }
    let gold = Color::srgb(1.0, 0.82, 0.2);
    for st in &plan.stops {
        gizmos.sphere(Isometry3d::from_translation(st.pos + Vec3::Y * 0.5), 0.5, gold);
        gizmos.line(st.pos + Vec3::Y * 0.9, st.pos + Vec3::Y * 2.2, Color::srgba(1.0, 0.82, 0.2, 0.6));
    }
}
