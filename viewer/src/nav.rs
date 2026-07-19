//! eft::nav — in-process CPU pathfinding over the baked layered-2.5D nav grid.
//!
//! This REPLACES the old external GPU pathfind server (NVIDIA-Warp/CUDA + Python GraphQL on :8091):
//! routing now runs entirely on the CPU, in-process, so it works on EVERY GPU (indeed with none) and
//! ships inside the exe with no Python/CUDA/server dependency. It is a faithful port of the web
//! viewer's proven `_route.js` A* — the same algorithm that already served as that viewer's fallback
//! whenever the GPU server was offline.
//!
//! DATA (baked once at build time by `bake_nav.py` on the author's GPU; shipped in the .eftpack):
//!   nav.json      — { min_x, min_z, res, nx, nz, n_layers(K), miss, climb, drop_max, ... }
//!   nav.bin       — f32[nx*nz*K]: cell (iz*nx+ix) layer l height at (iz*nx+ix)*K + l, ascending,
//!                    `miss` for empty layers. A layered 2.5-D heightfield (mall floors / floor-under-
//!                    canopy each get a layer).
//!   nav_door.bin  — u8[nx*nz]: 1 = door cell (forced passable; paths cross closed doors).
//!   nav_blk.bin   — u8[nx*nz*K]: per (cell,layer) 8-bit mask; bit d set = the edge to neighbour d is
//!                    blocked by a thin wall/fence (caught by a body-height ray at bake time).
//!
//! ALGORITHM: A* over nodes = cell*K + layer. A neighbour connects if step-up <= climb and
//! descent <= drop_max (doors bypass), the edge isn't in the block mask, and (for diagonals) it isn't
//! a corner-cut. Edge cost is true 3-D surface distance with a vertical penalty (`VERT`) so routes
//! prefer staying on one floor. `path` snaps the start onto the nearest real floor (spiral) and tries
//! the destination layers nearest the requested Y. `chain` visits every dest in the cheapest order
//! (exact TSP <= 7 stops, nearest-neighbour above); `tour` keeps a given order.
//!
//! Scratch uses per-query "generation" stamps instead of clearing M-sized arrays each call, so a
//! single A* costs O(nodes visited), not O(grid) — important for big maps + N^2 chain matrices.

use bevy::prelude::*;
use std::path::Path;

/// 8-neighbour offsets — SAME order/semantics as `_route.js` NB and the bake's block-mask bit `d`.
const NB: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
/// Vertical-movement cost multiplier: strongly prefers the flat floor (no roof/ceiling detours).
const VERT: f32 = 6.0;
/// Extra cost (× res) for entering a wall-adjacent cell — an agent-clearance nudge that biases
/// routes off walls and away from the sub-cell gaps a thin edge-ray can thread. Soft + uniform,
/// so it steers toward a clearer route when one exists but never blocks a genuinely narrow passage.
const WALL_CLEARANCE: f32 = 0.35;

/// Per-CELL soft-avoidance cost (extra metres-equivalent added when the path enters the cell).
/// XZ-only (a danger zone spans all floors above it). Built by [`NavGrid::build_avoid`] from
/// danger points (boss/PMC/scav spawns); the A* takes it as an optional penalty layer, so paths
/// "avoid if possible" — they still cross a zone when no reasonable detour exists.
pub type AvoidMap = std::collections::HashMap<u32, f32>;

/// A loaded, immutable nav grid for one map. Shared read-only across async query tasks.
pub struct NavGrid {
    pub min_x: f32,
    pub min_z: f32,
    pub res: f32,
    pub nx: usize,
    pub nz: usize,
    pub k: usize,
    miss: f32,
    /// Ceiling for an up-move: rises above this are never scaled (players vault ~1.2 m at most).
    climb: f32,
    /// A drop larger than this is routed around (fall damage), not stepped off.
    drop_max: f32,
    /// Free step-up height (stairs / curbs auto-stepped without a slope check) — Unity stepHeight.
    step_up: f32,
    /// tan(max walkable incline). An up-move's rise/run above this is too STEEP to scale — the
    /// player would slide (Unity NavMesh maxSlope). Separates "walk a hill" from "scale a wall": a
    /// 1.2 m rise over a 1 m cell is 50° and now rejected, while a curb (<= step_up) still passes.
    slope_tan: f32,
    /// nx*nz*K ascending floor heights (`miss` = empty).
    h: Vec<f32>,
    /// nx*nz door bits.
    door: Vec<u8>,
    /// nx*nz*K 8-dir edge-block masks.
    blk: Vec<u8>,
    /// nx*nz: true when the cell is within ~1 cell of a blocked edge. A small enter-penalty on
    /// these keeps the route centred in corridors instead of hugging walls / threading the
    /// sub-cell gaps the thin edge-ray can miss — an agent-radius clearance nudge (Unity erode).
    near_wall: Vec<bool>,
    /// nx*nz: 1 = a wall triangle occupies this cell's body column (baked `nav_wallcell.bin`). The
    /// wall-aware simplifier refuses to straighten a chord THROUGH such a cell, so the drawn line
    /// never cuts a sub-cell wall that blocks no cell-edge (invisible to `blk`). Absent = all-zero.
    wall_cell: Vec<u8>,
}

/// Reusable per-query A* scratch (generation-stamped so no full clears). One `Scratch` is reused
/// across all legs of a chain/tour. Sized to the grid on first use.
pub struct Scratch {
    gen: u32,
    g: Vec<f32>,
    came: Vec<i32>,
    open_gen: Vec<u32>,
    closed_gen: Vec<u32>,
    heap: Vec<u32>,
}

impl Scratch {
    pub fn new(m: usize) -> Self {
        Self {
            gen: 0,
            g: vec![0.0; m],
            came: vec![-1; m],
            open_gen: vec![0; m],
            closed_gen: vec![0; m],
            heap: Vec::with_capacity(1024),
        }
    }
}

impl NavGrid {
    /// Load the nav grid from a directory holding nav.json + nav.bin (+ optional door/blk). Returns
    /// None (with a log) if no grid is present — the caller then reports "no route data for this map".
    pub fn load(dir: &Path) -> Option<NavGrid> {
        let meta_txt = std::fs::read_to_string(dir.join("nav.json")).ok()?;
        let meta: serde_json::Value = serde_json::from_str(&meta_txt).ok()?;
        let f = |k: &str| meta.get(k).and_then(|v| v.as_f64());
        let i = |k: &str| meta.get(k).and_then(|v| v.as_u64());
        let (min_x, min_z, res) = (f("min_x")? as f32, f("min_z")? as f32, f("res")? as f32);
        let (nx, nz, k) = (i("nx")? as usize, i("nz")? as usize, i("n_layers")? as usize);
        let miss = f("miss").unwrap_or(-1.0e9) as f32;
        let climb = f("climb").unwrap_or(1.2) as f32;
        let drop_max = f("drop_max").unwrap_or(2.0) as f32;
        // Walkability tuning (runtime — no re-bake). step_up = freely-walked curb/stair height;
        // walk_slope_deg = max incline you can scale (Unity maxSlope), distinct from the bake's
        // surface-recording slope (nav.json slope_max_deg, ~60). Env-overridable for A/B tuning.
        let step_up = env_f32("EFT_NAV_STEP").or_else(|| f("step_up").map(|v| v as f32)).unwrap_or(0.45);
        let slope_deg = env_f32("EFT_NAV_SLOPE").or_else(|| f("walk_slope_deg").map(|v| v as f32)).unwrap_or(45.0);
        let slope_tan = slope_deg.clamp(20.0, 70.0).to_radians().tan();
        let m = nx * nz * k;

        let h = read_f32(&dir.join("nav.bin"), m)?;
        // Door / block / wall-cell masks are optional; absent -> no doors / no blocked edges / no
        // wall cells (graceful — old packs baked before these existed still load & route).
        let door = read_u8(&dir.join("nav_door.bin"), nx * nz).unwrap_or_else(|| vec![0; nx * nz]);
        let blk = read_u8(&dir.join("nav_blk.bin"), m).unwrap_or_else(|| vec![0; m]);
        let wall_cell = read_u8(&dir.join("nav_wallcell.bin"), nx * nz).unwrap_or_else(|| vec![0; nx * nz]);

        // Agent-clearance field: a cell is "near a wall" if it (or a neighbour) has any blocked
        // edge. Dilated by one cell so the penalty biases routes ~1 cell (res) off walls — the
        // grid's coarse stand-in for eroding the walkable area by the agent radius.
        let mut near_wall = vec![false; nx * nz];
        for cell in 0..nx * nz {
            if (0..k).any(|l| blk[cell * k + l] != 0) {
                near_wall[cell] = true;
            }
        }
        let seed = near_wall.clone();
        for cz in 0..nz as i64 {
            for cx in 0..nx as i64 {
                let c = (cz * nx as i64 + cx) as usize;
                if near_wall[c] {
                    continue;
                }
                'ring: for dz in -1..=1i64 {
                    for dx in -1..=1i64 {
                        let (jx, jz) = (cx + dx, cz + dz);
                        if jx < 0 || jz < 0 || jx >= nx as i64 || jz >= nz as i64 {
                            continue;
                        }
                        if seed[(jz * nx as i64 + jx) as usize] {
                            near_wall[c] = true;
                            break 'ring;
                        }
                    }
                }
            }
        }
        info!(
            "nav: loaded grid {}x{}x{} @ {}m ({:.0} MB); step_up {:.2}m slope {:.0}deg; from {}",
            nx, nz, k, res,
            (h.len() * 4 + door.len() + blk.len() + wall_cell.len()) as f32 / 1e6,
            step_up, slope_deg,
            dir.display()
        );
        Some(NavGrid {
            min_x, min_z, res, nx, nz, k, miss, climb, drop_max, step_up, slope_tan, h, door, blk,
            near_wall, wall_cell,
        })
    }

    /// Can you move onto a neighbour whose floor is `up` metres above (negative = below) yours,
    /// `run` metres away horizontally? Doors always pass. UP: a free step (<= step_up), else a
    /// walkable incline capped by both the vault ceiling AND the max slope. DOWN: any survivable
    /// drop. This is what keeps routes on terrain the player can actually scale.
    #[inline]
    fn walkable_step(&self, up: f32, run: f32, forced: bool) -> bool {
        if forced {
            // Doors bypass the step/slope/thin-wall rule for UP moves and small downs, but a forced
            // edge still must not authorize a lethal multi-storey fall — cap the drop at drop_max.
            return up >= 0.0 || -up <= self.drop_max;
        }
        if up > 0.0 {
            up <= self.step_up || (up <= self.climb && up <= run * self.slope_tan)
        } else {
            -up <= self.drop_max
        }
    }

    /// One shared-ortho side of a diagonal is passable iff its edge isn't wall-blocked, the ortho
    /// cell has a floor near `h_ref`, and that step is walkable from `h_ref`. `blk_c` is the block
    /// mask of the diagonal's SOURCE (cell,layer). `o` is the ortho neighbour dir index (0..4).
    #[inline]
    fn ortho_ok(&self, ix: i64, iz: i64, h_ref: f32, blk_c: u8, o: usize) -> bool {
        if (blk_c >> o) & 1 != 0 {
            return false; // the ortho edge itself is a wall
        }
        let (dx, dz) = (NB[o].0 as i64, NB[o].1 as i64);
        let (jx, jz) = (ix + dx, iz + dz);
        if jx < 0 || jz < 0 || jx >= self.nx as i64 || jz >= self.nz as i64 {
            return false;
        }
        let oc = (jz * self.nx as i64 + jx) as usize;
        let nl = self.best_layer(oc, h_ref);
        if nl < 0 {
            return false;
        }
        let up = self.h_lay(oc, nl as usize) - h_ref;
        let run = ((dx * dx + dz * dz) as f32).sqrt() * self.res;
        self.walkable_step(up, run, false)
    }

    /// Strict diagonal corner-cut test: diagonal `d` (4..8) is allowed only when BOTH shared ortho
    /// sides are floored near `h_ref`, walkable from `h_ref`, AND blk-unblocked in `blk_c`. A
    /// 0.64 m-wide capsule can't squeeze a corner where either side is a wall or a missing floor,
    /// so both must pass (combining the floor AND the block mask, since a wall reads as unblocked
    /// in blk when it only removes the far cell's floor).
    #[inline]
    fn diag_ok(&self, ix: i64, iz: i64, h_ref: f32, blk_c: u8, d: usize) -> bool {
        // ortho pair composing diagonal d: o1 = (dx,0) step, o2 = (0,dz) step.
        let (o1, o2) = match d {
            4 => (0usize, 2usize), // (1,1)  -> +x, +z
            5 => (0, 3),           // (1,-1) -> +x, -z
            6 => (1, 2),           // (-1,1) -> -x, +z
            7 => (1, 3),           // (-1,-1)-> -x, -z
            _ => return true,
        };
        self.ortho_ok(ix, iz, h_ref, blk_c, o1) && self.ortho_ok(ix, iz, h_ref, blk_c, o2)
    }

    /// Node count (nx*nz*K) — the size a `Scratch` must match.
    pub fn nodes(&self) -> usize {
        self.nx * self.nz * self.k
    }

    /// Build a soft-avoidance cost field from danger points `(pos, radius_m)`. Cost per entered
    /// cell falls off linearly from `strength * res` at the centre to 0 at the radius edge —
    /// `strength` is roughly "extra metres of detour a path accepts per metre walked at the
    /// centre". Overlapping zones keep the max (not sum), so stacked spawns don't explode.
    pub fn build_avoid(&self, pts: &[(Vec3, f32)], strength: f32) -> AvoidMap {
        let mut m = AvoidMap::new();
        for (p, r) in pts {
            let r = r.max(self.res);
            let cr = (r / self.res).ceil() as i64;
            let cx = ((p.x - self.min_x) / self.res).round() as i64;
            let cz = ((p.z - self.min_z) / self.res).round() as i64;
            for dz in -cr..=cr {
                for dx in -cr..=cr {
                    let (jx, jz) = (cx + dx, cz + dz);
                    if jx < 0 || jz < 0 || jx >= self.nx as i64 || jz >= self.nz as i64 {
                        continue;
                    }
                    let d = (((dx * dx + dz * dz) as f32).sqrt()) * self.res;
                    if d > r {
                        continue;
                    }
                    let w = strength * (1.0 - d / r) * self.res;
                    let cell = (jz * self.nx as i64 + jx) as u32;
                    let e = m.entry(cell).or_insert(0.0);
                    if w > *e {
                        *e = w;
                    }
                }
            }
        }
        m
    }

    #[inline]
    fn cell_of(&self, x: f32, z: f32) -> i64 {
        let ix = ((x - self.min_x) / self.res).round() as i64;
        let iz = ((z - self.min_z) / self.res).round() as i64;
        if ix < 0 || iz < 0 || ix >= self.nx as i64 || iz >= self.nz as i64 {
            -1
        } else {
            iz * self.nx as i64 + ix
        }
    }

    #[inline]
    fn h_lay(&self, c: usize, l: usize) -> f32 {
        self.h[c * self.k + l]
    }

    /// Layer in cell `c` whose height is nearest `ref_y` (-1 if the cell has no floor).
    fn best_layer(&self, c: usize, ref_y: f32) -> i32 {
        let (mut b, mut bd) = (-1i32, f32::MAX);
        for l in 0..self.k {
            let hh = self.h[c * self.k + l];
            if hh <= self.miss * 0.5 {
                break; // layers are ascending; `miss` sinks to the end
            }
            let d = (hh - ref_y).abs();
            if d < bd {
                bd = d;
                b = l as i32;
            }
        }
        b
    }

    /// Layers in `cell` ordered by |height - y| (nearest first) — the dest tries these in order.
    fn layers_by_height(&self, cell: usize, y: f32) -> Vec<usize> {
        let mut out: Vec<(usize, f32)> = Vec::new();
        for l in 0..self.k {
            let hh = self.h[cell * self.k + l];
            if hh > self.miss * 0.5 {
                out.push((l, (hh - y).abs()));
            }
        }
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out.into_iter().map(|x| x.0).collect()
    }

    /// Snap a start onto the nearest cell+layer with a walkable floor near y (spiral; clamps
    /// off-grid). Mirrors `_route.js` snapStart so a start on a shelf/roof/off-grid still routes.
    fn snap_start(&self, x: f32, y: f32, z: f32, max_cells: i64) -> Option<(usize, usize)> {
        let mut cix = ((x - self.min_x) / self.res).round() as i64;
        let mut ciz = ((z - self.min_z) / self.res).round() as i64;
        cix = cix.clamp(0, self.nx as i64 - 1);
        ciz = ciz.clamp(0, self.nz as i64 - 1);
        for rad in 0..=max_cells {
            let (mut bc, mut bl, mut bd) = (-1i64, -1i64, f64::MAX);
            for dz in -rad..=rad {
                for dx in -rad..=rad {
                    if rad > 0 && dx.abs().max(dz.abs()) != rad {
                        continue; // only the ring at this radius
                    }
                    let (jx, jz) = (cix + dx, ciz + dz);
                    if jx < 0 || jz < 0 || jx >= self.nx as i64 || jz >= self.nz as i64 {
                        continue;
                    }
                    let c = (jz * self.nx as i64 + jx) as usize;
                    for l in 0..self.k {
                        let hh = self.h[c * self.k + l];
                        if hh <= self.miss * 0.5 {
                            break;
                        }
                        let d = (hh - y).abs() as f64 + rad as f64 * 0.5;
                        if d < bd {
                            bd = d;
                            bc = c as i64;
                            bl = l as i64;
                        }
                    }
                }
            }
            if bc >= 0 {
                return Some((bc as usize, bl as usize));
            }
        }
        None
    }

    #[inline]
    fn node_pos(&self, node: usize) -> Vec3 {
        let c = node / self.k;
        let l = node % self.k;
        Vec3::new(
            self.min_x + (c % self.nx) as f32 * self.res,
            self.h_lay(c, l),
            self.min_z + (c / self.nx) as f32 * self.res,
        )
    }

    /// A* from (start cell,layer) to (dest cell,layer). Returns the node polyline, or None if
    /// unreachable. Uses generation-stamped scratch (no O(grid) clears). `avoid` adds a per-cell
    /// soft penalty (danger zones) — heuristic stays the plain distance, which remains admissible
    /// (penalties only ADD cost), so the path is still optimal under the penalised metric.
    fn astar(
        &self,
        sc: usize,
        sl: usize,
        dc: usize,
        dl: usize,
        s: &mut Scratch,
        avoid: Option<&AvoidMap>,
        trace: &mut Option<Vec<(Vec3, f32)>>,
    ) -> Option<Vec<Vec3>> {
        let k = self.k;
        let nx = self.nx as i64;
        let nz = self.nz as i64;
        s.gen = s.gen.wrapping_add(1);
        let gen = s.gen;
        s.heap.clear();

        let (dix, diz) = ((dc % self.nx) as i64, (dc / self.nx) as i64);
        let heur = |c: usize| -> f32 {
            let (ix, iz) = ((c % self.nx) as i64, (c / self.nx) as i64);
            (((ix - dix) * (ix - dix) + (iz - diz) * (iz - diz)) as f32).sqrt() * self.res
        };

        // binary min-heap keyed by f = g + heur (stored implicitly: compare via s.g + heur cache).
        // We store f in a parallel value alongside the node via `fs`. To avoid another M-array we
        // recompute f lazily is wrong (g changes); instead keep f in the node's g slot is not enough.
        // Simplest faithful port: keep an f array stamped by gen.
        // (Reuse closed_gen's companion via a small local map is overkill; use a dedicated fs vec.)
        let start = sc * k + sl;
        let goal = dc * k + dl;
        s.g[start] = 0.0;
        s.open_gen[start] = gen;
        s.came[start] = -1;
        // heap holds node ids; ordering by f computed from g + heur (heur is cheap, cache per push).
        // Push helper:
        heap_push(&mut s.heap, start, &s.g, gen, &s.open_gen, &heur, k, self.nx);

        let mut expanded: u64 = 0;
        let mut found = false;
        while let Some(cur) = heap_pop(&mut s.heap, &s.g, &s.open_gen, gen, &heur, k, self.nx) {
            if cur == goal {
                found = true;
                break;
            }
            if s.closed_gen[cur] == gen {
                continue;
            }
            s.closed_gen[cur] = gen;
            if let Some(tr) = trace.as_mut() {
                tr.push((self.node_pos(cur), s.g[cur])); // record the wavefront for the live search viz
            }
            expanded += 1;
            if expanded > 2_000_000 {
                warn!("nav: A* expansion cap hit");
                break;
            }
            let c = cur / k;
            let l = cur % k;
            let (ix, iz) = ((c % self.nx) as i64, (c / self.nx) as i64);
            let h_cur = self.h_lay(c, l);
            let blk_c = self.blk[cur];
            for d in 0..8 {
                let (dx, dz) = (NB[d].0 as i64, NB[d].1 as i64);
                let (jx, jz) = (ix + dx, iz + dz);
                if jx < 0 || jz < 0 || jx >= nx || jz >= nz {
                    continue;
                }
                let nc = (jz * nx + jx) as usize;
                let nl = self.best_layer(nc, h_cur);
                if nl < 0 {
                    continue;
                }
                let nl = nl as usize;
                let h_n = self.h_lay(nc, nl);
                let up = h_n - h_cur;
                // A door on EITHER side forces the seam passable (symmetric — fixes the
                // leaving-a-door asymmetry). Gate the block mask AFTER nc/forced exist so a stray
                // blk bit on a door edge can never break a door.
                let forced = self.door[c] == 1 || self.door[nc] == 1;
                if !forced && (blk_c >> d) & 1 != 0 {
                    continue; // thin wall/fence blocks this edge
                }
                let horiz = ((dx * dx + dz * dz) as f32).sqrt() * self.res;
                if !self.walkable_step(up, horiz, forced) {
                    continue; // too steep/tall to scale, or too deep a drop (doors bypass)
                }
                if dx != 0 && dz != 0 && !forced && !self.diag_ok(ix, iz, h_cur, blk_c, d) {
                    continue; // strict corner-cut: both ortho sides must be clear + unblocked
                }
                let nn = nc * k + nl;
                let mut step = (horiz * horiz + (up * VERT) * (up * VERT)).sqrt();
                if self.near_wall[nc] {
                    step += self.res * WALL_CLEARANCE; // clearance nudge (soft; never blocks a corridor)
                }
                if let Some(av) = avoid {
                    if let Some(&p) = av.get(&(nc as u32)) {
                        step += p; // danger-zone soft penalty (extra metres-equivalent)
                    }
                }
                let ng = s.g[cur] + step;
                let known = s.open_gen[nn] == gen;
                if !known || ng < s.g[nn] {
                    s.g[nn] = ng;
                    s.came[nn] = cur as i32;
                    s.open_gen[nn] = gen;
                    heap_push(&mut s.heap, nn, &s.g, gen, &s.open_gen, &heur, k, self.nx);
                }
            }
        }

        if !found && goal != start {
            return None;
        }
        // Reconstruct.
        let mut path: Vec<Vec3> = Vec::new();
        let mut n = goal as i64;
        while n >= 0 {
            path.push(self.node_pos(n as usize));
            if n as usize == start {
                break;
            }
            n = s.came[n as usize] as i64;
        }
        path.reverse();
        Some(path)
    }

    /// Single-source Dijkstra COST FIELD from `from`, expanded until `g_limit` metres (no dest, no
    /// heuristic). The field lives in `s` (generation-stamped `g`); query it with [`Self::field_dist`].
    /// One bounded flood lets a planner test reachability/distance of MANY points without ever
    /// paying an exhaustive failed A* per unreachable point. Returns false if the start won't snap.
    pub fn dijkstra_field(&self, from: Vec3, g_limit: f32, s: &mut Scratch) -> bool {
        let Some((sc, sl)) = self.snap_start(from.x, from.y, from.z, 16) else {
            return false;
        };
        let k = self.k;
        let nx = self.nx as i64;
        let nz = self.nz as i64;
        s.gen = s.gen.wrapping_add(1);
        let gen = s.gen;
        s.heap.clear();
        let start = sc * k + sl;
        s.g[start] = 0.0;
        s.open_gen[start] = gen;
        // Plain g-ordered heap (heuristic = 0): reuse the shared helpers with a zero heur.
        let zero = |_c: usize| 0.0f32;
        heap_push(&mut s.heap, start, &s.g, gen, &s.open_gen, &zero, k, self.nx);
        while let Some(cur) = heap_pop(&mut s.heap, &s.g, &s.open_gen, gen, &zero, k, self.nx) {
            if s.closed_gen[cur] == gen {
                continue;
            }
            s.closed_gen[cur] = gen;
            if s.g[cur] > g_limit {
                continue; // beyond the budget ring — don't expand further
            }
            let c = cur / k;
            let l = cur % k;
            let (ix, iz) = ((c % self.nx) as i64, (c / self.nx) as i64);
            let h_cur = self.h_lay(c, l);
            let blk_c = self.blk[cur];
            for d in 0..8 {
                let (dx, dz) = (NB[d].0 as i64, NB[d].1 as i64);
                let (jx, jz) = (ix + dx, iz + dz);
                if jx < 0 || jz < 0 || jx >= nx || jz >= nz {
                    continue;
                }
                let nc = (jz * nx + jx) as usize;
                let nl = self.best_layer(nc, h_cur);
                if nl < 0 {
                    continue;
                }
                let nl = nl as usize;
                let h_n = self.h_lay(nc, nl);
                let up = h_n - h_cur;
                // Same door/blk ordering + strict corner-cut as astar — reachability MUST agree.
                let forced = self.door[c] == 1 || self.door[nc] == 1;
                if !forced && (blk_c >> d) & 1 != 0 {
                    continue;
                }
                let horiz = ((dx * dx + dz * dz) as f32).sqrt() * self.res;
                if !self.walkable_step(up, horiz, forced) {
                    continue; // same connectivity as astar (reachability must agree)
                }
                if dx != 0 && dz != 0 && !forced && !self.diag_ok(ix, iz, h_cur, blk_c, d) {
                    continue;
                }
                let nn = nc * k + nl;
                // Pure geometric step (NO clearance/avoid penalty) — the field is a distance
                // estimate for the planner, so its g must stay real metres.
                let step = (horiz * horiz + (up * VERT) * (up * VERT)).sqrt();
                let ng = s.g[cur] + step;
                let known = s.open_gen[nn] == gen;
                if !known || ng < s.g[nn] {
                    s.g[nn] = ng;
                    s.open_gen[nn] = gen;
                    heap_push(&mut s.heap, nn, &s.g, gen, &s.open_gen, &zero, k, self.nx);
                }
            }
        }
        true
    }

    /// Walkable distance to `pos` read from a [`Self::dijkstra_field`] in `s` — None if `pos`
    /// wasn't reached (unreachable, or beyond the flood's g_limit). Checks every layer of the
    /// cell and its 3x3 ring (seam/shelf tolerance), keeping the best stamped g.
    pub fn field_dist(&self, s: &Scratch, pos: Vec3) -> Option<f32> {
        let gen = s.gen;
        let (cx, cz) = self.cell_of_xz(pos.x, pos.z);
        let mut best: Option<f32> = None;
        for dz in -1..=1i64 {
            for dx in -1..=1i64 {
                let (jx, jz) = (cx + dx, cz + dz);
                if jx < 0 || jz < 0 || jx >= self.nx as i64 || jz >= self.nz as i64 {
                    continue;
                }
                let c = (jz * self.nx as i64 + jx) as usize;
                for l in 0..self.k {
                    let n = c * self.k + l;
                    if s.open_gen[n] == gen && best.is_none_or(|b| s.g[n] < b) {
                        best = Some(s.g[n]);
                    }
                }
            }
        }
        best
    }

    #[inline]
    fn cell_of_xz(&self, x: f32, z: f32) -> (i64, i64) {
        (
            ((x - self.min_x) / self.res).round() as i64,
            ((z - self.min_z) / self.res).round() as i64,
        )
    }

    /// Line-of-sight for the wall-aware simplifier: true when the straight XZ segment `a`->`b` can
    /// be WALKED end-to-end — every cell it enters has a floor, no cell-edge it crosses is a
    /// blocked (thin-wall) edge, every step is walkable, and diagonal corner crossings satisfy the
    /// strict corner-cut rule. Traverses the grid with a supercover DDA (Amanatides–Woo; a corner
    /// hit is a diagonal step) enumerating EVERY entered cell + EVERY crossed edge, so a grazing
    /// blocked edge a point-sampler would skip is still caught. Layers are resolved by the running
    /// floor (A*-consistent). Doors keep their seam passable. A chord that would float off the
    /// floor (tunnel over a pit / through a mezzanine) is rejected.
    fn segment_clear(&self, a: Vec3, b: Vec3) -> bool {
        let res = self.res;
        let (nxi, nzi) = (self.nx as i64, self.nz as i64);
        // Cell-space with cell CENTRES at integers -> shift +0.5 so index = floor(u) and cell
        // boundaries land on integers (clean AW).
        let u0 = (a.x - self.min_x) / res + 0.5;
        let w0 = (a.z - self.min_z) / res + 0.5;
        let u1 = (b.x - self.min_x) / res + 0.5;
        let w1 = (b.z - self.min_z) / res + 0.5;
        let du = u1 - u0;
        let dw = w1 - w0;
        let mut ix = u0.floor() as i64;
        let mut iz = w0.floor() as i64;
        let ixe = u1.floor() as i64;
        let ize = w1.floor() as i64;
        if ix < 0 || iz < 0 || ix >= nxi || iz >= nzi {
            return false;
        }
        let mut cur_cell = (iz * nxi + ix) as usize;
        let l0 = self.best_layer(cur_cell, a.y);
        if l0 < 0 {
            return false;
        }
        // A multi-cell chord LEAVING a wall cell can clip that cell's wall at an angle the raw edge
        // avoids, so never straighten OUT of a wall cell either (only the trivial same-cell span is
        // allowed). Combined with the per-step wall_cell reject below, EVERY cell the chord touches
        // is wall-free — the drawn line is provably wall-clear.
        if (ix != ixe || iz != ize) && self.wall_cell.get(cur_cell).is_some_and(|&w| w != 0) {
            return false;
        }
        let mut cur_layer = l0 as usize;
        let mut cur_floor = self.h_lay(cur_cell, cur_layer);
        let step_x: i64 = if du > 0.0 { 1 } else { -1 };
        let step_z: i64 = if dw > 0.0 { 1 } else { -1 };
        let tdelta_x = if du != 0.0 { 1.0 / du.abs() } else { f32::INFINITY };
        let tdelta_z = if dw != 0.0 { 1.0 / dw.abs() } else { f32::INFINITY };
        let mut tmax_x = if du > 0.0 {
            ((ix + 1) as f32 - u0) / du
        } else if du < 0.0 {
            (ix as f32 - u0) / du
        } else {
            f32::INFINITY
        };
        let mut tmax_z = if dw > 0.0 {
            ((iz + 1) as f32 - w0) / dw
        } else if dw < 0.0 {
            (iz as f32 - w0) / dw
        } else {
            f32::INFINITY
        };
        let dy = b.y - a.y;
        // How far the drawn chord may sit off the real floor before it's "floating". Kept TIGHT
        // (a free step) so the chord hugs the ground: a straight span is only accepted where it
        // tracks the floor within a step, which also keeps stairs from flattening into a ramp that
        // floats up past the wall_cell body band and clips a ledge/rail the mask can't see.
        let float_tol = self.step_up;
        let max_steps = (du.abs() + dw.abs()) as usize + 4;
        let mut guard = 0usize;
        loop {
            if ix == ixe && iz == ize {
                break;
            }
            guard += 1;
            if guard > max_steps + 8 {
                return false; // safety: never spin (panic=abort) — treat as not-clear
            }
            let (odx, odz, t_at);
            if (tmax_x - tmax_z).abs() < 1.0e-6 {
                // exact corner: cross both boundaries at once (diagonal step)
                odx = step_x;
                odz = step_z;
                t_at = tmax_x;
                ix += step_x;
                iz += step_z;
                tmax_x += tdelta_x;
                tmax_z += tdelta_z;
            } else if tmax_x < tmax_z {
                odx = step_x;
                odz = 0;
                t_at = tmax_x;
                ix += step_x;
                tmax_x += tdelta_x;
            } else {
                odx = 0;
                odz = step_z;
                t_at = tmax_z;
                iz += step_z;
                tmax_z += tdelta_z;
            }
            if ix < 0 || iz < 0 || ix >= nxi || iz >= nzi {
                return false;
            }
            let d = match (odx, odz) {
                (1, 0) => 0,
                (-1, 0) => 1,
                (0, 1) => 2,
                (0, -1) => 3,
                (1, 1) => 4,
                (1, -1) => 5,
                (-1, 1) => 6,
                (-1, -1) => 7,
                _ => return false,
            };
            let new_cell = (iz * nxi + ix) as usize;
            let forced = self.door[cur_cell] == 1 || self.door[new_cell] == 1;
            let blk_c = self.blk[cur_cell * self.k + cur_layer];
            if !forced && (blk_c >> d) & 1 != 0 {
                return false; // crosses a thin-wall edge
            }
            // Strict corner-cut on the diagonal micro-step (source = the cell we're leaving).
            if odx != 0 && odz != 0 && !forced && !self.diag_ok(ix - odx, iz - odz, cur_floor, blk_c, d)
            {
                return false;
            }
            let nnl = self.best_layer(new_cell, cur_floor);
            if nnl < 0 {
                return false; // ran off the walkable floor
            }
            let nnl = nnl as usize;
            // Refuse to STRAIGHTEN into/through ANY cell whose body column holds a wall (including
            // the destination cell — a chord's final approach must not clip a wall right at the
            // endpoint). The per-edge blk mask can't see a sub-cell wall that blocks no cell-centre
            // edge, but a long chord cutting at an angle CAN clip it. `wall_cell` is baked from the
            // SAME wall geometry as the acceptance test, so rejecting straightening through every
            // wall cell provably keeps the drawn chord clear of walls. Only the START (anchor) cell
            // is exempt (it's a pinned route point, checked as `cur_cell`, never as `new_cell`).
            // Falls back to the raw staircase near walls; open spans still straighten.
            if self.wall_cell.get(new_cell).is_some_and(|&w| w != 0) {
                return false;
            }
            let new_floor = self.h_lay(new_cell, nnl);
            let up = new_floor - cur_floor;
            let run = if odx != 0 && odz != 0 {
                res * std::f32::consts::SQRT_2
            } else {
                res
            };
            if !self.walkable_step(up, run, forced) {
                return false; // a true riser/drop across this edge — can't straighten through it
            }
            let y_interp = a.y + dy * t_at.clamp(0.0, 1.0);
            if (new_floor - y_interp).abs() > float_tol {
                return false; // the chord would float off the floor here
            }
            cur_cell = new_cell;
            cur_layer = nnl;
            cur_floor = new_floor;
        }
        true
    }

    /// Wall-aware simplification of a raw 8-connected node path: an LOS-constrained greedy
    /// string-pull. Keeps a vertex only when the straight line from the last kept anchor to the
    /// NEXT vertex is not [`Self::segment_clear`], so every kept segment is guaranteed wall-clear.
    /// Falls back to the raw staircase for any span that can't be straightened (correctness over
    /// cosmetics) — never emits a chord that cuts a wall/fence corner the way plain Douglas–Peucker
    /// could. Endpoints are always pinned.
    pub fn simplify_route(&self, pts: &[Vec3]) -> Vec<Vec3> {
        let n = pts.len();
        if n < 3 {
            return pts.to_vec();
        }
        let mut keep = vec![false; n];
        keep[0] = true;
        let mut anchor = 0usize;
        for i in 1..n - 1 {
            // anchor..pts[i] is known clear; if extending to pts[i+1] would cross a wall, plant a
            // kept vertex at i and restart the pull from there.
            if !self.segment_clear(pts[anchor], pts[i + 1]) {
                keep[i] = true;
                anchor = i;
            }
        }
        keep[n - 1] = true;
        (0..n).filter(|&i| keep[i]).map(|i| pts[i]).collect()
    }

    /// Bake self-check helper: return the RAW 8-connected node path AND its wall-aware
    /// simplification for a->b (or None). Lets the machine proof attribute wall-crossings to the
    /// A* connectivity (raw) vs the simplifier (simplified).
    pub fn route_debug(&self, a: Vec3, b: Vec3, s: &mut Scratch) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
        let (sc, sl) = self.snap_start(a.x, a.y, a.z, 16)?;
        let mut dc = self.cell_of(b.x, b.z);
        if dc < 0 {
            let cix = (((b.x - self.min_x) / self.res).round() as i64).clamp(0, self.nx as i64 - 1);
            let ciz = (((b.z - self.min_z) / self.res).round() as i64).clamp(0, self.nz as i64 - 1);
            dc = ciz * self.nx as i64 + cix;
        }
        let dc = dc as usize;
        for dl in self.layers_by_height(dc, b.y) {
            if let Some(path) = self.astar(sc, sl, dc, dl, s, None, &mut None) {
                let simp = self.simplify_route(&path);
                return Some((path, simp));
            }
        }
        None
    }

    /// Test/bake-only: zero the wall block mask + clearance + wall-cell fields to reproduce OLD
    /// routing (packs with no nav_blk.bin/nav_wallcell.bin) for the self-check before/after contrast.
    pub fn clear_wall_data(&mut self) {
        for b in self.blk.iter_mut() {
            *b = 0;
        }
        for w in self.near_wall.iter_mut() {
            *w = false;
        }
        for w in self.wall_cell.iter_mut() {
            *w = 0;
        }
    }

    /// Route a->b: snap the start, try dest layers nearest b.y. Returns (polyline, walkable length).
    /// The reported length is the REAL walked metres (avoid penalties shape the path, not the number).
    pub fn path(
        &self,
        a: Vec3,
        b: Vec3,
        s: &mut Scratch,
        avoid: Option<&AvoidMap>,
    ) -> Option<(Vec<Vec3>, f32)> {
        let (sc, sl) = self.snap_start(a.x, a.y, a.z, 16)?;
        let mut dc = self.cell_of(b.x, b.z);
        if dc < 0 {
            // dest off-grid: clamp XZ into the grid
            let cix = (((b.x - self.min_x) / self.res).round() as i64).clamp(0, self.nx as i64 - 1);
            let ciz = (((b.z - self.min_z) / self.res).round() as i64).clamp(0, self.nz as i64 - 1);
            dc = ciz * self.nx as i64 + cix;
        }
        let dc = dc as usize;
        for dl in self.layers_by_height(dc, b.y) {
            if let Some(path) = self.astar(sc, sl, dc, dl, s, avoid, &mut None) {
                // Report the length of the SIMPLIFIED polyline (what actually gets drawn + walked),
                // not the raw 8-connected staircase: the staircase over-measures a diagonal-ish leg
                // by the grid metrication error (~up to 8%). Simplifying first makes the displayed
                // metres match the drawn line and the true walked distance. The wall-aware pull
                // guarantees the drawn chords never cut through a wall the cell path avoided.
                let simp = self.simplify_route(&path);
                let dist = polyline_len(&simp);
                return Some((simp, dist));
            }
        }
        None
    }

    /// Like `path`, but ALSO records the A* wavefront (every closed node's position + g-distance),
    /// down-sampled to `max_trace` points, so the UI can animate the search converging. Only used
    /// for single-destination routes with the "visualize search" toggle on.
    pub fn path_traced(
        &self,
        a: Vec3,
        b: Vec3,
        s: &mut Scratch,
        avoid: Option<&AvoidMap>,
        max_trace: usize,
    ) -> Option<(Vec<Vec3>, f32, Vec<(Vec3, f32)>)> {
        let (sc, sl) = self.snap_start(a.x, a.y, a.z, 16)?;
        let mut dc = self.cell_of(b.x, b.z);
        if dc < 0 {
            let cix = (((b.x - self.min_x) / self.res).round() as i64).clamp(0, self.nx as i64 - 1);
            let ciz = (((b.z - self.min_z) / self.res).round() as i64).clamp(0, self.nz as i64 - 1);
            dc = ciz * self.nx as i64 + cix;
        }
        let dc = dc as usize;
        for dl in self.layers_by_height(dc, b.y) {
            let mut trace: Option<Vec<(Vec3, f32)>> = Some(Vec::new());
            if let Some(path) = self.astar(sc, sl, dc, dl, s, avoid, &mut trace) {
                let simp = self.simplify_route(&path);
                let dist = polyline_len(&simp);
                let mut tr = trace.unwrap_or_default();
                // Down-sample the wavefront (keeps g-order) so the animated draw stays cheap.
                if max_trace > 0 && tr.len() > max_trace {
                    let stride = tr.len() / max_trace + 1;
                    tr = tr.into_iter().step_by(stride).collect();
                }
                return Some((simp, dist, tr));
            }
        }
        None
    }

    /// Chain: visit every dest from `start` in the cheapest order (exact TSP <= 7 stops, else
    /// nearest-neighbour). Returns one flattened polyline + total length + the visiting order (into
    /// `dests`). Legs from unreachable dests are skipped.
    pub fn chain(
        &self,
        start: Vec3,
        dests: &[Vec3],
        s: &mut Scratch,
        avoid: Option<&AvoidMap>,
    ) -> Option<(Vec<Vec3>, f32)> {
        if dests.is_empty() {
            return None;
        }
        if dests.len() == 1 {
            return self.path(start, dests[0], s, avoid);
        }
        let n = dests.len() + 1;
        // nodes[0] = start, nodes[1..] = dests
        let node = |i: usize| if i == 0 { start } else { dests[i - 1] };
        // pairwise legs P[(i,j)] for i in 0..n, j in 1..n, i!=j
        let mut legs: std::collections::HashMap<(usize, usize), (Vec<Vec3>, f32)> = std::collections::HashMap::new();
        for i in 0..n {
            for j in 1..n {
                if i != j {
                    if let Some(r) = self.path(node(i), node(j), s, avoid) {
                        legs.insert((i, j), r);
                    }
                }
            }
        }
        let leg_dist = |i: usize, j: usize| legs.get(&(i, j)).map(|r| r.1);
        let dests_idx: Vec<usize> = (1..n).collect();
        let mut best_order: Option<Vec<usize>> = None;
        let mut best_total = f32::MAX;
        if dests.len() <= 7 {
            // exact TSP over a fixed start
            permute(&dests_idx, &mut |perm: &[usize]| {
                let mut tot = 0.0;
                let mut prev = 0usize;
                for &kk in perm {
                    match leg_dist(prev, kk) {
                        Some(dd) => {
                            tot += dd;
                            prev = kk;
                        }
                        None => return,
                    }
                }
                if tot < best_total {
                    best_total = tot;
                    best_order = Some(perm.to_vec());
                }
            });
        }
        if best_order.is_none() {
            // greedy nearest-neighbour over the reachable subset
            let mut rem: std::collections::BTreeSet<usize> = dests_idx.iter().copied().collect();
            let mut prev = 0usize;
            let mut order = Vec::new();
            while !rem.is_empty() {
                let nxt = rem
                    .iter()
                    .copied()
                    .filter(|&kk| leg_dist(prev, kk).is_some())
                    .min_by(|&x, &y| leg_dist(prev, x).unwrap().total_cmp(&leg_dist(prev, y).unwrap()));
                match nxt {
                    Some(kk) => {
                        order.push(kk);
                        prev = kk;
                        rem.remove(&kk);
                    }
                    None => break,
                }
            }
            best_order = Some(order);
        }
        // stitch legs (skip the duplicated shared endpoint between legs)
        let order = best_order?;
        let mut full: Vec<Vec3> = Vec::new();
        let mut total = 0.0;
        let mut prev = 0usize;
        for kk in order {
            let Some((pts, d)) = legs.get(&(prev, kk)) else { break };
            if full.is_empty() {
                full.extend_from_slice(pts);
            } else {
                full.extend_from_slice(&pts[1.min(pts.len())..]);
            }
            total += d;
            prev = kk;
        }
        (full.len() > 1).then_some((full, total))
    }

    /// Tour: route an ORDERED sequence of waypoints as one continuous polyline (each leg continues
    /// from the previous SNAPPED endpoint so shared elevated waypoints don't jump floors).
    pub fn tour(
        &self,
        points: &[Vec3],
        s: &mut Scratch,
        avoid: Option<&AvoidMap>,
    ) -> Option<(Vec<Vec3>, f32)> {
        if points.len() < 2 {
            return None;
        }
        let mut full: Vec<Vec3> = Vec::new();
        let mut total = 0.0;
        let mut prev: Option<Vec3> = None;
        for i in 1..points.len() {
            let a = prev.unwrap_or(points[i - 1]);
            if let Some((pts, d)) = self.path(a, points[i], s, avoid) {
                if pts.len() > 1 {
                    if full.is_empty() {
                        full.extend_from_slice(&pts);
                    } else {
                        full.extend_from_slice(&pts[1..]);
                    }
                    prev = pts.last().copied();
                    total += d;
                }
            }
        }
        (full.len() > 1).then_some((full, total))
    }
}

fn polyline_len(p: &[Vec3]) -> f32 {
    p.windows(2).map(|w| (w[1] - w[0]).length()).sum()
}

// ---- binary min-heap over node ids, keyed by f = g + heur (ported from _route.js) --------------

#[inline]
fn f_of(node: usize, g: &[f32], open_gen: &[u32], gen: u32, heur: &impl Fn(usize) -> f32, k: usize, nx: usize) -> f32 {
    let gv = if open_gen[node] == gen { g[node] } else { f32::INFINITY };
    let c = node / k;
    let _ = nx;
    gv + heur(c)
}

fn heap_push(
    heap: &mut Vec<u32>,
    node: usize,
    g: &[f32],
    gen: u32,
    open_gen: &[u32],
    heur: &impl Fn(usize) -> f32,
    k: usize,
    nx: usize,
) {
    heap.push(node as u32);
    let mut i = heap.len() - 1;
    while i > 0 {
        let p = (i - 1) >> 1;
        if f_of(heap[p] as usize, g, open_gen, gen, heur, k, nx)
            <= f_of(heap[i] as usize, g, open_gen, gen, heur, k, nx)
        {
            break;
        }
        heap.swap(p, i);
        i = p;
    }
}

fn heap_pop(
    heap: &mut Vec<u32>,
    g: &[f32],
    open_gen: &[u32],
    gen: u32,
    heur: &impl Fn(usize) -> f32,
    k: usize,
    nx: usize,
) -> Option<usize> {
    if heap.is_empty() {
        return None;
    }
    let top = heap[0];
    let last = heap.pop().unwrap();
    if !heap.is_empty() {
        heap[0] = last;
        let mut i = 0usize;
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut sm = i;
            if l < heap.len()
                && f_of(heap[l] as usize, g, open_gen, gen, heur, k, nx)
                    < f_of(heap[sm] as usize, g, open_gen, gen, heur, k, nx)
            {
                sm = l;
            }
            if r < heap.len()
                && f_of(heap[r] as usize, g, open_gen, gen, heur, k, nx)
                    < f_of(heap[sm] as usize, g, open_gen, gen, heur, k, nx)
            {
                sm = r;
            }
            if sm == i {
                break;
            }
            heap.swap(sm, i);
            i = sm;
        }
    }
    Some(top as usize)
}

// ---- small helpers ---------------------------------------------------------------------------

/// Heap's-permutations of `items`, calling `f` on each ordering (Heap's algorithm, iterative-ish).
fn permute(items: &[usize], f: &mut impl FnMut(&[usize])) {
    fn go(arr: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
        if k == arr.len() {
            f(arr);
            return;
        }
        for i in k..arr.len() {
            arr.swap(k, i);
            go(arr, k + 1, f);
            arr.swap(k, i);
        }
    }
    let mut a = items.to_vec();
    go(&mut a, 0, f);
}

fn read_f32(path: &Path, n: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < n * 4 {
        warn!("nav: {} too small ({} bytes, need {})", path.display(), bytes.len(), n * 4);
        return None;
    }
    let mut out = vec![0.0f32; n];
    for (i, o) in out.iter_mut().enumerate() {
        let b = [bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]];
        *o = f32::from_le_bytes(b);
    }
    Some(out)
}

/// Parse an f32 from an env var (trimmed); None if unset or unparseable.
fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|s| s.trim().parse().ok())
}

fn read_u8(path: &Path, n: usize) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < n {
        return None;
    }
    Some(bytes[..n].to_vec())
}
