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

/// Walking speed the plan budgets at, m/s.
pub(crate) const WALK_MPS: f32 = 1.65;
/// Seconds held back from the walking budget for the extract itself. The WALK to the exit is a
/// routed leg and is counted separately; this is the time standing at it. Module-scope so the
/// panel's caption is the solver's own number rather than a "2" free to drift from it.
pub(crate) const EXTRACT_BUFFER_S: f32 = 120.0;

/// Ask for a loot-run plan. Sent by the Navigation tab's PLAN button.
#[derive(Message, Clone)]
pub struct PlanRequest {
    /// Ignore loot below this estimated rouble value.
    pub min_value: i64,
    /// Stop cap (the tour visits at most this many loot points).
    pub max_stops: usize,
    /// Total raid-time budget in seconds, including search time and extract reserve.
    pub budget_s: f32,
    /// Priced loose loot (`PoiLayer::LooseLoot`) counts as a candidate stop alongside containers.
    /// false = containers only.
    pub include_loose: bool,
    /// Extract names the run is allowed to END at. EMPTY = no filter, i.e. every active extract.
    ///
    /// A loot run is only useful if it finishes somewhere you can actually leave from, and which
    /// extracts are open depends on side, time, keys and the raid's random selection - none of
    /// which the viewer can know. So the player says which ones count and the plan honours exactly
    /// those, instead of the optimizer quietly picking one that will not be available.
    ///
    /// The Navigation panel NEVER sends this empty: it disables PLAN LOOT RUN until at least one
    /// extract is ticked, because "no choice made" and "every extract is fine" are different
    /// statements and only the player knows which one is true. The sole empty sender is the
    /// `EFT_PLAN` headless harness, which has no user to tick boxes.
    pub extracts: Vec<String>,
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

pub(crate) struct Plan {
    pub(crate) stops: Vec<PlanStop>,
    pub(crate) extract: String,
    pub(crate) polyline: Vec<Vec3>,
    pub(crate) total_dist: f32,
    pub(crate) total_time: f32,
    pub(crate) total_value: i64,
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

/// Headless-QA aid: `EFT_PLAN="min_value,max_stops,budget_minutes[,include_loose]"` (or `1` for
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
        include_loose: nums.get(3).map(|v| *v != 0.0).unwrap_or(true),
            // debug harness: any active extract is acceptable
        extracts: Vec::new(),
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
pub(crate) struct Cand {
    pub(crate) name: String,
    pub(crate) value: i64,
    pub(crate) score_value: f32,
    pub(crate) pos: Vec3,
    pub(crate) loot_s: f32,
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
        Option<&crate::loot::SpawnChance>,
    )>,
    locks: Query<(&GlobalTransform, &crate::poi::LockKeys)>,
    progress: Res<crate::progress::PlayerProgress>,
    all_marks: Query<
        (
            &crate::poi::PoiLayer,
            &GlobalTransform,
            &crate::inspect::MarkerInfo,
            Option<&crate::poi::SceneInactive>,
            Option<&crate::poi::PlayerStart>,
            Option<&crate::poi::ExtractFaction>,
        ),
        Without<crate::poi::ZoneWall>,
    >,
    zones: Res<crate::poi::GameDataZones>,
    game_link: Option<Res<crate::game_watch::GameLink>>,
    side_choice: Option<Res<crate::game_watch::SideChoice>>,
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
        .filter(|(_, _, v, cls, layer, _, _, _)| {
            // The filter stays on RAW worth: "min value" means "worth this much if it is there".
            v.0 >= req.min_value
                && (cls.is_some() // loot.rs container
                    || (req.include_loose
                        && matches!(layer, Some(crate::poi::PoiLayer::LooseLoot)))) // priced loose
        })
        .filter(|(gt, _, _, _, _, _, _, _)| {
            !locks.iter().any(|(lock_gt, keys)| {
                lock_gt.translation().distance(gt.translation()) <= 14.0
                    && !keys.0.is_empty()
                    && !keys.0.iter().any(|key| progress.owns_key(key))
            })
        })
        .map(|(gt, info, v, _, _, loot_time, jackpot, chance)| Cand {
            name: info.title.clone(),
            value: v.0,
            // RANK by expected value: worth x probability it is actually there. The odds come from
            // the game's own LootableContainersGroup counts where available (19% for the mall
            // stashes, 83% at Kiba Arms), else loot.json's location-blind per-type average. Ranking
            // on raw worth scored those identically, which is a 4x error on the same container type.
            score_value: v.0 as f32
                * chance.map(|c| c.0).unwrap_or(1.0)
                * if jackpot.is_some() { 0.18 } else { 1.0 },
            pos: gt.translation(),
            loot_s: loot_time.map(|t| t.0).unwrap_or(5.0),
        })
        .collect();
    cands.sort_by(|a, b| b.value.cmp(&a.value));
    cands.truncate(120);
    if cands.is_empty() {
        plan.status = PlanStatus::Error(if req.include_loose {
            "no loot above the value filter on this map".to_string()
        } else {
            "no containers above the value filter on this map \u{2014} lower it, or tick \
             'include loose loot'"
                .to_string()
        });
        return;
    }

    // ---- extract candidates (active only) — the run must END somewhere safe.
    let want: std::collections::HashSet<&str> = req.extracts.iter().map(String::as_str).collect();
    let raid_side = crate::game_watch::effective_side(game_link.as_deref(), side_choice.as_deref());
    let extracts: Vec<(String, Vec3)> = all_marks
        .iter()
        .filter(|(l, _, _, inactive, _, faction)| {
            **l == crate::poi::PoiLayer::Extract
                && inactive.is_none()
                && !raid_side.is_some_and(|side| {
                    faction.is_some_and(|faction| !side.allows_extract(&faction.0))
                })
        })
        .filter(|(_, _, info, _, _, _)| want.is_empty() || want.contains(info.title.as_str()))
        .map(|(_, gt, info, _, _, _)| (info.title.clone(), gt.translation()))
        .collect();
    if extracts.is_empty() {
        // Distinguish "this map has none" from "you deselected them all" - otherwise the user
        // reads a data problem into their own filter.
        plan.status = PlanStatus::Error(if want.is_empty() {
            "no active extracts on this map".to_string()
        } else {
            "the extracts you ticked are not active on this map".to_string()
        });
        return;
    }

    // ---- avoid field (same options as normal routing) ----
    let mut avoid_pts: Vec<(Vec3, f32)> = Vec::new();
    if opts.avoid_boss || opts.avoid_pmc || opts.avoid_scav {
        for (l, gt, _, _, player_start, _) in &all_marks {
            let r = match l {
                crate::poi::PoiLayer::Boss if opts.avoid_boss => 45.0,
                // Player raid starts excluded (players scatter within minutes) — only AI-PMC
                // bot anchors on the PMC layer repel. Same rule as pathfind's avoid field.
                crate::poi::PoiLayer::PmcSpawn if opts.avoid_pmc && player_start.is_none() => 32.0,
                crate::poi::PoiLayer::ScavSpawn if opts.avoid_scav => 24.0,
                _ => continue,
            };
            avoid_pts.push((gt.translation(), r));
        }
    }

    // "Avoid combat": the game's patrol lines + ground an AI-PMC anchor can actually see.
    // Gathered here because it needs the patrol polylines rather than a per-marker radius, and
    // merged into the same avoid field so the tour and any route drawn to one of its stops agree
    // about what is dangerous.
    let (combat_patrols, combat_eyes): (Vec<Vec<Vec3>>, Vec<Vec3>) = if opts.avoid_combat {
        (
            zones.patrols.clone(),
            all_marks
                .iter()
                .filter(|(l, _, _, _, ps, _)| {
                    **l == crate::poi::PoiLayer::PmcSpawn && ps.is_none()
                })
                .map(|(_, gt, _, _, _, _)| gt.translation())
                .collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    plan.status = PlanStatus::Pending;
    route_result.status = RouteStatus::Pending;
    let (max_stops, budget) = (req.max_stops, req.budget_s.max(300.0));
    let t = AsyncComputeTaskPool::get().spawn(async move {
        let mut avoid = (!avoid_pts.is_empty()).then(|| grid.build_avoid(&avoid_pts, 4.0));
        if !combat_patrols.is_empty() || !combat_eyes.is_empty() {
            if let Some(c) =
                crate::pathfind::build_combat_avoid(&grid, &combat_patrols, &combat_eyes, 4.0)
            {
                match avoid.as_mut() {
                    Some(a) => crate::nav::NavGrid::merge_avoid(a, c),
                    None => avoid = Some(c),
                }
            }
        }
        solve(&grid, start, cands, extracts, max_stops, budget, avoid.as_ref())
    });
    task.0 = Some(t);
}

/// The two/three-phase orienteering heuristic (see module doc).
pub(crate) fn solve(
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
    let est = |a: Vec3, b: Vec3| a.distance(b) * DETOUR;

    // ---- phase 0: ONE bounded Dijkstra flood from the start prunes unreachable candidates
    // up front. Without this, every shelf/roof loot point that isn't on the nav mesh cost a
    // full EXHAUSTIVE failed A* during threading (seconds each — the planner looked hung).
    let mut field_s = crate::nav::pooled_scratch(grid.nodes());
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

    // ---- phase 2b: re-order on REAL walked distance -------------------------------------------
    //
    // Phase 2 untangles the tour in STRAIGHT-LINE space, which is the wrong space. Two containers
    // 20 m apart with a building between them are 150 m of walking, so an ordering that looks
    // clean on a map can double back repeatedly on foot — measured on streets: 4 self-crossing leg
    // pairs and a tour 5.6x the straight line through its own stops.
    //
    // The stop set is already chosen and small (<= max_stops), so the real distances are affordable
    // here in a way they were not during insertion: one bounded Dijkstra flood per stop fills a row
    // of the matrix, and `field_dist` reads every other stop out of that single flood. 2-opt on
    // those numbers reorders by how far the player actually walks.
    //
    // The flood bound is a deliberate trade. Measured over 6 streets runs: 5078 m of tour with no
    // real-distance pass, 4809 m if every row floods the whole walk budget, 5052 m with the bound
    // below. The unbounded version wins by 5% and sweeps the map fourteen times per plan, which is
    // seconds of stall on a button the player is waiting on; the bounded one keeps most of the
    // ordering wins for a fraction of that. Far pairs it cannot reach stay INFINITY and are simply
    // not swapped, which is why the gain is smaller rather than wrong.
    {
        let mut pts: Vec<Vec3> = Vec::with_capacity(tour.len() + 2);
        pts.push(start);
        pts.extend(tour.iter().map(|&ci| cands[ci].pos));
        pts.push(ex0.1);
        let np = pts.len();
        let mut dm = vec![f32::INFINITY; np * np];
        let mut fs = crate::nav::pooled_scratch(grid.nodes());
        for i in 0..np {
            // Bound each flood by what THIS row needs: the farthest stop from `pts[i]`, with the
            // estimator's own detour factor and some slack. Flooding to the whole walk budget
            // instead would sweep the entire map 14 times for a set of stops that often sit within
            // one building, which is far too slow for a button a player presses and waits on.
            let reach = pts
                .iter()
                .map(|q| est(pts[i], *q))
                .fold(0.0f32, f32::max)
                * 1.6;
            let limit = reach.clamp(50.0, walk_budget_m * 1.4);
            if !grid.dijkstra_field(pts[i], limit, &mut fs) {
                continue;
            }
            for j in 0..np {
                if i == j {
                    dm[i * np + j] = 0.0;
                } else if let Some(d) = grid.field_dist(&fs, pts[j]) {
                    dm[i * np + j] = d;
                }
            }
        }
        // Symmetrise: a flood that ran out of budget one way may still have reached the other.
        for i in 0..np {
            for j in (i + 1)..np {
                let m = dm[i * np + j].min(dm[j * np + i]);
                dm[i * np + j] = m;
                dm[j * np + i] = m;
            }
        }
        let n = tour.len();
        let orig = tour.clone();
        let mut order: Vec<usize> = (1..=n).collect(); // indices into `pts`
        let dist = |a: usize, b: usize| dm[a * np + b];
        let mut improved = true;
        let mut guard = 0usize;
        while improved && guard < 64 {
            guard += 1;
            improved = false;
            for i in 0..n {
                for j in (i + 1)..n {
                    let a = if i == 0 { 0 } else { order[i - 1] };
                    let b = order[i];
                    let c = order[j];
                    let e = if j + 1 == n { np - 1 } else { order[j + 1] };
                    let old = dist(a, b) + dist(c, e);
                    let new = dist(a, c) + dist(b, e);
                    // Only act on numbers we actually have. An unreachable pair is INFINITY, and
                    // swapping on it would "improve" the tour by hiding a leg that cannot be walked.
                    if old.is_finite() && new.is_finite() && new + 0.01 < old {
                        order[i..=j].reverse();
                        improved = true;
                    }
                }
            }
        }
        tour = order.iter().map(|&k| orig[k - 1]).collect();
    }

    // ---- phase 3: real legs (A* threading) + budget repair ----
    let mut s = crate::nav::pooled_scratch(grid.nodes());
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
        // A routed leg does not necessarily BEGIN where the previous one ended: `NavGrid::path`
        // snaps the start with a 16-cell search that can land on a different storey when the
        // column holds several floors. Dropping the leg's first vertex (`&p[1..]`) then splices
        // the previous endpoint straight onto the relocated one, and the plan line jumps through
        // whatever is between — on interchange's stacked floors, through a ceiling. If the start
        // moved, we did not actually find a walk from the last stop to this one, so treat it as
        // unreachable and let the existing re-thread drop the stop.
        const JOIN_TOL: f32 = 1.5; // metres; grid res is 1 m, so a real join is ~0
        for (k, &ci) in tour.iter().enumerate() {
            match grid.path(cur, cands[ci].pos, &mut s, avoid) {
                Some((p, _)) if !poly.is_empty() && p[0].distance(cur) > JOIN_TOL => {
                    unreachable = Some(k);
                    break;
                }
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
        // Same guard on the final leg. Rather than failing the whole plan, drop the last stop and
        // re-thread — the tour is what stranded us, and a shorter plan beats a plan whose exit leg
        // is a line through a building.
        if exp[0].distance(cur) > JOIN_TOL {
            if tour.len() > 1 {
                tour.pop();
                continue;
            }
            return Err("no walkable path to any extract".into());
        }
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
        // INVARIANT: the drawn plan never leaves the walkable floor. The failure this guards is
        // silent by nature — a spliced leg join draws a clean straight line through a storey slab
        // and looks like a route — so assert it on the finished polyline rather than trusting the
        // construction. `on_floor` samples the 3x3 cell neighbourhood, so a diagonal step across a
        // corner notch does not trip it; only a line with no floor at that height anywhere near it.
        {
            let mut off = 0usize;
            let (mut worst, mut at) = (0.0f32, Vec3::ZERO);
            for w in poly.windows(2) {
                let n = (w[0].distance(w[1]) / 1.0).ceil().max(1.0) as usize;
                for i in 0..=n {
                    let p = w[0].lerp(w[1], i as f32 / n as f32);
                    if !grid.on_floor(p.x, p.z, p.y, 1.5) {
                        off += 1;
                        let d = grid.floor_near(p.x, p.z, p.y).map_or(99.0, |f| (p.y - f).abs());
                        if d > worst {
                            worst = d;
                            at = p;
                        }
                    }
                }
            }
            if off > 0 {
                warn!(
                    "loot plan route leaves the floor at {off} sampled metre(s), worst {worst:.1} m                      at ({:.1},{:.1},{:.1}) — the drawn line is not a walkable route there",
                    at.x, at.y, at.z
                );
            }
        }
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
                route_result.dest_label = Some(format!("Loot run \u{203a} {}", p.extract));
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
