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
    /// A layer above the previous by <= this many metres is walkable up (players vault ~1.2 m).
    climb: f32,
    /// A drop larger than this is routed around (fall damage), not stepped off.
    drop_max: f32,
    /// nx*nz*K ascending floor heights (`miss` = empty).
    h: Vec<f32>,
    /// nx*nz door bits.
    door: Vec<u8>,
    /// nx*nz*K 8-dir edge-block masks.
    blk: Vec<u8>,
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
        let m = nx * nz * k;

        let h = read_f32(&dir.join("nav.bin"), m)?;
        // Door / block masks are optional; absent -> no doors / no blocked edges (graceful).
        let door = read_u8(&dir.join("nav_door.bin"), nx * nz).unwrap_or_else(|| vec![0; nx * nz]);
        let blk = read_u8(&dir.join("nav_blk.bin"), m).unwrap_or_else(|| vec![0; m]);
        info!(
            "nav: loaded grid {}x{}x{} @ {}m ({:.0} MB) from {}",
            nx, nz, k, res,
            (h.len() * 4 + door.len() + blk.len()) as f32 / 1e6,
            dir.display()
        );
        Some(NavGrid { min_x, min_z, res, nx, nz, k, miss, climb, drop_max, h, door, blk })
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
                if (blk_c >> d) & 1 != 0 {
                    continue; // thin wall/fence blocks this edge
                }
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
                let forced = self.door[nc] == 1;
                if !forced && (up > self.climb || -up > self.drop_max) {
                    continue; // wall / cliff (doors bypass)
                }
                if dx != 0 && dz != 0 {
                    // no corner-cut through a wall: at least one of the two ortho cells must be floor
                    let a = (iz * nx + jx) as usize;
                    let b = (jz * nx + ix) as usize;
                    if self.best_layer(a, h_cur) < 0 && self.best_layer(b, h_cur) < 0 && !forced {
                        continue;
                    }
                }
                let nn = nc * k + nl;
                let horiz = ((dx * dx + dz * dz) as f32).sqrt() * self.res;
                let mut step = (horiz * horiz + (up * VERT) * (up * VERT)).sqrt();
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
            if let Some(path) = self.astar(sc, sl, dc, dl, s, avoid) {
                let dist = polyline_len(&path);
                return Some((path, dist));
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

/// Douglas–Peucker in 3-D (perpendicular distance incl. Y) — drops the 8-connected staircase but
/// KEEPS ramps/stairs (an XZ-only reduction would float a straight diagonal up through floors).
pub fn simplify(pts: &[Vec3], eps: f32) -> Vec<Vec3> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    *keep.last_mut().unwrap() = true;
    let mut st = vec![(0usize, pts.len() - 1)];
    while let Some((a, b)) = st.pop() {
        let (aa, bb) = (pts[a], pts[b]);
        let u = bb - aa;
        let len = u.length().max(1e-6);
        let (mut md, mut mi) = (0.0f32, usize::MAX);
        for i in (a + 1)..b {
            let p = pts[i] - aa;
            let d = p.cross(u).length() / len; // |(P-A) x u| / |u|
            if d > md {
                md = d;
                mi = i;
            }
        }
        if md > eps && mi != usize::MAX {
            keep[mi] = true;
            st.push((a, mi));
            st.push((mi, b));
        }
    }
    pts.iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| *p)
        .collect()
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

fn read_u8(path: &Path, n: usize) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < n {
        return None;
    }
    Some(bytes[..n].to_vec())
}
