# Atlas field test — RTX 4060 / RX 6800

Thanks for helping test. This build exists because of two field reports on AMD hardware, and
because everything below the "baseline numbers" section has only ever been measured on one
machine (an RTX 5090). You two are the calibration points for mid-range NVIDIA and AMD.

**Time needed:** ~20 minutes of viewer time (plus a one-time map build from your own game
install — see README.md; the biggest maps take a while and ~2–10 GB disk each).

---

## 1. What changed in this build (why we think your crash is fixed)

**If you're the RX 6800 tester:** your "loads fine, dies when I move/resize the window" crash
was traced to a chain that has nothing to do with your card being broken:

- Windows freezes an app's presentation while you drag a window (the "modal move/size loop").
- Your GPU was already saturated (85% util at 45 fps), so the render queue was full.
- Under vsync, acquiring the next frame then times out after 1 second — and the engine we build
  on (Bevy) treats that timeout as fatal **on Windows only** (on Linux it shrugs it off,
  explicitly for AMD cards). Our release builds abort instantly on any fatal error, so it died
  with no error message.

This build patches that engine code: a missed frame during a window drag is now **skipped**, not
fatal. Three more layers of hardening shipped alongside:

- Frame queue depth lowered 2 → 1 (less backlog when the GPU is the bottleneck; also lower
  input latency). `EFT_FRAME_LATENCY=2` restores the old value if we ask you to A/B it.
- GPU errors no longer abort silently: they are logged to `packs\logs\atlas_viewer.log`, and a
  genuinely fatal one (device lost / out of GPU memory) now exits cleanly with a message
  instead of vanishing.
- The AMD-fallback "Standard" render path now refuses to decode full-resolution textures
  (that configuration peaked at 17 GB on a 16 GB card and caused death-by-paging).

**Everything crash-shaped now leaves a trace in `packs\logs\atlas_viewer.log`. That file is
the single most valuable thing you can send back.**

---

## 2. The 20-minute test protocol

Do these in order. If anything crashes or freezes, note **what step**, grab
`packs\logs\atlas_viewer.log`, and keep going if you can.

### A. Setup (once)
1. Unzip anywhere. Keep `atlas.exe`, `assets\`, `packs\` side by side.
2. Build at least one map from your own game install (README.md "Building your own packs" —
   the in-app BUILD button does everything). If you can, build **the biggest map you play**
   (Streets is the stress test; Interchange is the GPU-heavy one; Factory is quick and small).
3. Launch `atlas.exe`, open a map, fly around for a minute.

### B. The window-abuse gauntlet (this is the crash repro — especially RX 6800)
With the biggest map loaded and the camera looking at a busy area (fps at its lowest):
1. Grab the title bar and **drag the window around continuously for 10+ seconds**. Wiggle it.
2. Grab a corner and **drag-resize continuously for 10+ seconds** — small, huge, small.
3. Maximize, restore, maximize, restore.
4. Minimize to taskbar, wait 5 s, restore. Repeat 3×.
5. Alt-tab away to another app, wait 10 s, alt-tab back.
6. If you have two monitors: drag the window slowly from one to the other and back
   (especially if they have different Windows scale factors — 100%/125%/150%).
7. Windows key + arrow snapping: snap left, right, corners.

**Pass:** the app survives all seven with no crash (stutter during drags is expected and fine).
**Before this build, step 1 or 2 killed the RX 6800 in seconds.** If it still dies: send
`atlas_viewer.log` — the last lines now say exactly which mechanism fired.

### C. Benchmark numbers (copy-paste, one PowerShell window)
From the folder containing `atlas.exe`, with the window **visible** (don't minimize it while a
bench runs) and nothing heavy running on the GPU:

```powershell
# 20-second orbit benchmark, interchange (swap the pack path for the map you built).
# Each prints one "[bench] frames= ... fps= p50= p95= p99=" line and exits.
$env:EFT_UNCAPPED="1"; $env:EFT_BENCH="20"

$env:EFT_ORBIT="-55,140,-134,60,40,12"
.\atlas.exe packs\interchange.eftpack          # run 1: orbit (moving camera)

Remove-Item Env:\EFT_ORBIT
$env:EFT_POSE="-55,140,-134,0,-35"
.\atlas.exe packs\interchange.eftpack          # run 2: static overview

$env:EFT_SHADOWS="0"
.\atlas.exe packs\interchange.eftpack          # run 3: static, shadows off (the big lever)

Remove-Item Env:\EFT_SHADOWS, Env:\EFT_POSE, Env:\EFT_UNCAPPED, Env:\EFT_BENCH
```

If you built a different map, run the same three configs on it — any fixed pose is fine
(fly somewhere representative, press nothing, and use `EFT_POSE`-less runs 1–3 without the
pose vars; the `[bench]` line is what matters). The `[bench]` lines also land in
`atlas_viewer.log`, so you don't need to copy the console.

### D. Two A/B one-liners (only if B and C went fine)
```powershell
# 1) old frame-queue depth — checks the new default didn't cost you fps
$env:EFT_FRAME_LATENCY="2"; $env:EFT_UNCAPPED="1"; $env:EFT_BENCH="20"
.\atlas.exe packs\interchange.eftpack
Remove-Item Env:\EFT_FRAME_LATENCY

# 2) vsync mode ON (the default users run) — then repeat gauntlet steps 1-2 for 10 s
Remove-Item Env:\EFT_UNCAPPED, Env:\EFT_BENCH
.\atlas.exe packs\interchange.eftpack
```

### E. What to send back
1. `packs\logs\atlas_viewer.log` (contains adapter/driver line, render path, `[bench]` lines,
   and any errors — it rotates, so send it right after testing).
2. Which gauntlet steps crashed/froze, if any.
3. GPU driver version (NVIDIA: GeForce app; AMD: Adrenalin → Settings → System) and whether
   your driver is a stock install.
4. Monitor setup: how many, resolutions, Windows scale factor(s), refresh rate.
5. System RAM, and which drive the packs live on (SSD/HDD).
6. Rough subjective notes: load time, how it feels flying around, anything ugly.

---

## 3. What performance to expect (calibrated guesses — you're the real data)

All first-party numbers are from the dev RTX 5090 at 1600×1000, uncapped, release build
(`docs/bench_2026-07-26_*.csv`). Scaling estimates use published relative-performance data;
Atlas is unusually **geometry/shadow-bound** (~15 ms of the interchange frame on the 5090 is
resolution-independent), so it scales *worse* than typical games on smaller GPUs — treat the
estimates as ±30%.

| Scenario (1600×1000) | RTX 5090 measured | RTX 4060 expected | RX 6800 expected |
|---|---:|---:|---:|
| Interchange overview, camera at rest | 8.9 ms / 113 fps | ~25–35 ms / 30–40 fps | ~20–28 ms / 36–50 fps |
| Interchange orbit (camera moving) | 18.5 ms / 54 fps | ~50–70 ms / 15–20 fps | ~40–55 ms / 18–25 fps |
| Interchange, shadows off (`EFT_SHADOWS=0`) | 8.6 ms / 116 fps | ~24–33 ms | ~19–27 ms |
| Streets overview, static | 5.4 ms / 187 fps | ~15–20 ms / 50–65 fps | ~12–17 ms / 60–80 fps |
| Streets orbit | 8.5 ms / 117 fps | ~24–33 ms / 30–42 fps | ~19–27 ms / 37–52 fps |
| Factory close-up | 3.5 ms / 290 fps | ~10 ms / ~100 fps | ~8 ms / ~125 fps |

Context for the RX 6800 tester's original "45 fps": that is in the plausible range for
interchange-class scenes on this card, i.e. the app was likely working correctly and simply
GPU-bound — which is exactly the state that made the window-drag crash fire.

**The two settings that matter if it feels slow:**
- Graphics panel → shadows off: biggest single lever (halved the interchange frame on dev).
- Smaller window = faster: fragment cost is ~1.8 ms per megapixel on the 5090, proportionally
  more on your cards. A 4K maximized window is 4× the pixels of the default 1600×1000.

## 4. VRAM budget per map (why texture quality defaults to Half)

Resident GPU memory by texture-quality setting (GPU-driven path, BC-compressed, measured from
pack contents; add ~0.1–0.6 GB for window targets + whatever Windows/other apps hold):

| Map | Full | **Half (default)** | Quarter |
|---|---:|---:|---:|
| Streets | 8.5 GB | **3.9 GB** | 2.8 GB |
| Woods | 3.7 GB | **1.2 GB** | 0.6 GB |
| Terminal | 3.7 GB | **1.5 GB** | 1.0 GB |
| Interchange | 2.9 GB | **1.3 GB** | 0.9 GB |
| Factory | 2.0 GB | **0.7 GB** | 0.4 GB |
| Icebreaker | 0.8 GB | **0.3 GB** | 0.2 GB |

- **RTX 4060 (8 GB): leave texture quality on Half.** Full on Streets alone exceeds the whole
  card and there is no eviction — it will page, crawl, and can die on a resize. Half is one
  mip level down and visually near-identical.
- **RX 6800 (16 GB): Full fits every map** on the normal (GPU-driven) path, but Half is still
  the sensible default while we're chasing stability.
- System RAM: the viewer also keeps the map's geometry in RAM (~2.8 GB for Streets, less for
  others) plus load-time staging; 16 GB total RAM machines may page during load.

## 5. Known rough edges (not bugs you need to report)

- **Streaming hitches while flying fast** — frame spikes to ~13 ms p95 on dev while textures/
  geometry finalize; proportionally bigger on your cards. Known, being worked on.
- **Moving the camera on Interchange is ~2× the cost of sitting still** — the sun-shadow
  cascade cache only helps a stationary camera. Known top perf item.
- **First map open after a build is slower** — the BC texture cache warms on first render.
- **Map switching on the AMD-fallback (Standard) path relaunches the app** — only the main
  GPU-driven path swaps maps in place.
- **Alt-tabbed viewer idles to ~2 Hz on purpose** (frees the GPU for your game).
  `EFT_UNCAPPED=1` disables that for benching.
- **DX12 is deliberately disabled** (a driver-stack bug crashes it before we get control);
  the viewer requires working Vulkan. A missing/ancient Vulkan runtime exits with a clear
  message rather than rendering.
- Stutter while actively dragging/resizing the window is expected (Windows stalls the app's
  presentation during drags) — it should *never* crash, that's what changed in this build.
- **Occasional tearing when fps drops below your monitor's refresh rate** — the default vsync
  mode resolves to "relaxed vsync" (FifoRelaxed) on both of your cards, which trades a tear
  for not halving the frame rate. Benign; mention it only if it's constant.

## 6. Library/driver support notes (what we depend on your card providing)

Atlas renders through **Vulkan only** via wgpu 26 / Bevy 0.17. The main (GPU-driven) path
auto-probes at startup and falls back gracefully if anything is missing; the probe result is
the `render path:` line in `atlas_viewer.log`.

| Requirement | RTX 4060 | RX 6800 |
|---|---|---|
| Vulkan multi-draw-indirect + first-instance | ✅ (count limit 2³²−1) | ✅ (count limit 2³²−1) |
| Bindless texture arrays (~6,200 descriptors bound in one stage) | ✅ (limit 1,048,576 → ~170× headroom) | ✅ (limit 2³²−1 → effectively unlimited; non-uniform indexing is compiler-emulated on RDNA2 — works, small perf cost) |
| BC texture compression | ✅ (all 16 formats) | ✅ (all 16 formats) |
| One 1.7 GB vertex buffer (Streets) | ✅ (4 GiB storage-binding range, 1 TiB buffer cap → ~2.3× headroom) | ✅ but tight: **hard 2.147 GB per-allocation cap** (verified identical across Adrenalin 25.6.1 → 25.12.1) → ~15% headroom; a bigger-than-Streets map breaks AMD first |
| MSAA 4× HDR + Depth32Float targets, workgroup-64 compute | ✅ (subgroup 32 → 2 full warps) | ✅ (native wave32/wave64 → ideal fit) |
| Present modes (Windows) | Fifo/FifoRelaxed/Mailbox/Immediate | Fifo/FifoRelaxed/Immediate — **no Mailbox**; uncapped mode = Immediate (tearing is expected, benign) |
| Usable VRAM before overcommit | 7.77 GiB of the 8 GB exposed to apps | ~16 GiB |
| Known driver quirk | — | AMDVLK/LLPC-lineage drivers (not stock Adrenalin) crash the bindless path; auto-detected → falls back to the Standard path. **Stock Adrenalin recommended.** |

*(Sources: vulkan.gpuinfo.org device reports — RX 6800 Windows ids 45082/41296/40630 spanning
Adrenalin 25.6.1→25.12.1, RTX 4060 Windows ids 49613/49066 — cross-checked against the exact
wgpu 26 feature/limit derivation in its sources. The `AdapterInfo` line in your
`atlas_viewer.log` is the ground truth for YOUR machine.)*

## 7. Env-var cheat sheet (only the ones this doc uses)

| Var | Effect |
|---|---|
| `EFT_BENCH=<secs>` | benchmark mode: prints one `[bench]` stats line after load, exits |
| `EFT_UNCAPPED=1` | disable vsync + focus-idle (required for meaningful bench numbers) |
| `EFT_POSE="x,y,z,yaw,pitch"` | pin the camera to an exact pose |
| `EFT_ORBIT="cx,cy,cz,r,h,degps"` | scripted orbiting camera (moving-camera load) |
| `EFT_SHADOWS=0` | kill sun-shadow cascades (the biggest perf lever) |
| `EFT_FRAME_LATENCY=2` | restore the old frame-queue depth (A/B only) |
| `EFT_RENDER=gpu\|std\|m0` | force a render path (skip the auto-probe) — only if we ask |
| `EFT_TEX_FULL=1` | override the Standard-path Full-texture clamp — only if we ask |

---

*Build: see the zip name (`atlas-<version>-<git sha>-win64-full`). Doc source:
`docs/FIELD_TEST_GPU.md`; dev-side benchmark detail: `docs/PERF_BENCHMARKS.md`,
`docs/VRAM_AUDIT.md`; crash investigation record: session notes + `PERF_FINDINGS.md`.*
