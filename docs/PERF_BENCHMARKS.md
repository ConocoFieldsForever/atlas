# Viewer performance benchmarks — 2026-07-26

Release build, RTX 5090, 1600×1000 window, `EFT_UNCAPPED=1`, visible window (a HIDDEN window's
swapchain is DWM-paced to ~54 fps regardless of vsync mode — never benchmark hidden). Each row =
20 s of per-frame CPU deltas via `EFT_BENCH` after load settle. Camera scripts: `EFT_POSE`
(static), `EFT_ORBIT="cx,cy,cz,r,h,degps"` (moving), `EFT_FLY="a>b@secs"` (ping-pong path).
Harness lives in `main.rs` (`bench_stats`, `debug_bench_camera`); runner scripts under the
session scratchpad (`bench.ps1` pattern — trivially recreatable from this table's env columns).

## Baseline matrix (pre-fix)

| scenario | avg ms | fps | p95 | p99 | notes |
|---|---:|---:|---:|---:|---|
| interchange overview, static, min layers | 18.82 | 53 | 19.5 | 19.8 | the slow view |
| interchange overview, static, ALL layers | 18.86 | 53 | 19.5 | 19.9 | layers ≈ +0.04 ms |
| interchange orbit (moving), min | 18.52 | 54 | 21.1 | 21.6 | |
| interchange orbit, ALL layers | 18.54 | 54 | 21.1 | 21.6 | layers ≈ free |
| interchange ground fly | 5.04 | 198 | 12.8 | 14.9 | fast but SPIKY (streaming) |
| streets overview, static, min | 5.36 | 187 | 6.0 | 6.9 | streets ≈ 3.5× faster than interchange?! |
| streets overview, static, ALL | 5.62 | 178 | 6.2 | 6.9 | layers ≈ +0.26 ms |
| streets orbit, min | 8.52 | 117 | 9.4 | 9.7 | |
| streets orbit, ALL | 8.66 | 116 | 9.6 | 10.0 | |
| streets street fly | 5.82 | 172 | 7.6 | 8.2 | |
| factory close-up | 3.45 | 290 | 3.9 | 4.3 | sanity floor |

## Feature ablations (orbit scenarios, min layers)

| ablation | interchange | Δ | streets | Δ |
|---|---:|---:|---:|---:|
| baseline | 18.52 ms | — | 8.52 ms | — |
| `EFT_SHADOWS=0` | **8.63 ms** | **−9.9 ms (−53%)** | 8.79 ms | 0 (no sun_dir → already off) |
| `EFT_CULL_PX=4,4` | **15.34 ms** | **−3.2 ms (−17%)** | **6.55 ms** | **−2.0 ms (−23%)** |
| `EFT_LIGHTS=off` | 17.86 ms | −0.65 ms | **6.54 ms** | **−2.0 ms (−23%)** |
| `EFT_LIGHTS=rt` (forced) | 18.43 ms | ~0 | 8.45 ms | ~0 (auto already rt) |
| `EFT_PARALLAX=0` | 18.43 ms | ~0 | 8.53 ms | ~0 |
| grass culled (`1.5,10000`) | 18.44 ms | ~0 | 8.79 ms | ~0 |
| `EFT_FOG=0` | 18.45 ms | ~0 | — | ~0 |

## Ranked improvement opportunities (quantified)

1. **Sun-shadow cascade caching — up to 9.9 ms on interchange-class maps (54→116 fps).**
   The sun is STATIC and the world is STATIC; the 2×2048² cascades still re-render every frame.
   Cache each cascade and re-render only when its snapped origin moves (camera crossing a snap
   cell) — static views pay ~0, moving views amortize to a fraction. This is the single biggest
   win in the entire viewer. Fallback option: drop to 1 cascade / 1024² when the camera is high
   (map-overview mode) — overview shadows don't need contact detail.
2. **Adaptive screen-size cull — 2.0–3.2 ms everywhere (−17..23%).** `cull_px` 1.5→4 px is
   visually invisible from overview height (those instances subtend <4 px) but pays on every
   map. Make the threshold scale with camera altitude instead of a global constant.
3. **Realtime-light distance gating — 2.0 ms on streets-class maps.** Thousands of indoor
   practicals iterate per-fragment through the light grid even when the camera is 400 m above
   the roofline. Skip/fade the RT light loop above a camera-height threshold (the SH volume
   already carries the ambient look from that distance).
4. **Streaming hitches while flying — p95 12.8 ms vs p50 4.1 ms on the interchange ground fly.**
   Steady-state is fine; the spikes are texcache/geometry residency work. Tune
   `EFT_LOAD_BUDGET_MS` down for in-flight smoothness / spread finalize work across more frames.
5. **Marker/overlay systems — ≤0.26 ms even with every layer on (streets, ~6.5 k markers).**
   Fixed anyway (this session): visibility components were rewritten unconditionally every
   frame (change-detection churn over every marker + a String alloc per loot marker per frame in
   the clustering key); both now write-on-change with a static-camera early-out. The earlier
   subjective "laggy with everything visible" was the immediate-mode gizmo tessellation
   (waypoint dot circles at 32 segments etc.), distance-LOD'd in the previous commit.

## After the marker-churn fixes (same scenarios, same day)

| scenario | before | after | Δ |
|---|---:|---:|---:|
| streets overview static, ALL layers | 5.62 ms | **5.40 ms** | −0.23 ms — layers now ≈ FREE on a static camera (min was 5.36) |
| streets orbit, ALL layers | 8.66 ms | 8.57 ms | −0.09 ms (moving camera still re-clusters, write-on-change still helps) |
| interchange static/orbit ALL | 18.86 / 18.54 | 18.86 / 18.52 | 0 — buried under the render cost |

## Resolution discriminator (interchange orbit, min layers)

| window | avg ms |
|---|---:|
| 800×500 (0.4 MP) | 14.32 |
| 1600×1000 (1.6 MP) | 18.52 |
| 3200×2000 (6.4 MP) | 25.13 |

Fragment-dependent slope ≈ **1.8 ms/MP** → at the default window only ~3 ms of the 18.5 ms is
pixel work. The other ~15 ms is resolution-INDEPENDENT: shadow-cascade rendering (fixed 2048²
targets, full-map caster set) + instance/vertex processing. This independently confirms the
ablation ranking — the wins are geometry-side, not shading-side.

## Combined candidate ("what's on the table")

`EFT_SHADOWS=0` + `EFT_CULL_PX=4,4` on interchange orbit: **18.52 → 7.71 ms (54 → 130 fps,
−58% frame time)** — and that's the settings-only version. Shadow *caching* (opportunity 1)
keeps the shadows and captures most of the same win.

## Dead folklore (measured ≈ 0)

Grass rendering, parallax mapping, distance fog, rt-vs-sh light PATH choice (given the volume
exists), marker entity count at current scale. Do not spend effort there.

## Harness reference

- `EFT_BENCH=20` — record 20 s of frames after settle, print `[bench] …` line, exit 0.
- `EFT_ORBIT` / `EFT_FLY` — deterministic moving-camera load (PostUpdate override).
- `EFT_WIN=WxH` — resolution scaling (fragment-bound vs fixed-cost discrimination).
- Benchmark VISIBLE and UNCAPPED, one instance at a time (the gpu-lease warns otherwise).
