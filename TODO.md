# EFT Native Viewer — TODO / deferred

Tracking issues deferred during M0–M2. Updated 2026-07-14.

## Performance / latency
- [ ] **Opaque3d early-z** (HIGH — the big one). The GPU-driven draw rides the `Transparent3d`
  sorted phase with **no early-z**, so occluded fragments are still shaded → overdraw-bound at
  dense/overhead views (~58 FPS top-down on Interchange; the whole multi-floor mall overdraws).
  The pipeline is already opaque (depth write + GreaterEqual, no blend), so pixels are identical —
  moving the phase item to a binned `Opaque3d` (front-to-back + depth pre-rejection) is a pure win.
  (M2 synth punch #4; single-item design needs a per-item distance for the binned phase.)
- [ ] **HZB / occlusion culling**. Cull is frustum-only. Add hierarchical-Z occlusion in the compute
  pass — huge for interiors where most geometry is behind walls/floors.
- [ ] **Exact σ_max cull radius** (LOW). World-sphere radius uses the conservative Frobenius norm
  (≤ √3× over-estimate). A closed-form symmetric-3×3 eigenvalue σ_max tightens culling under shear.
  Perf-only; Frobenius is already correct (never under-culls).
- [ ] **Release LTO**. `lto="thin"` crashed rustc (STATUS_ACCESS_VIOLATION) on Bevy/Windows; profile
  is `lto=false` for now. Revisit with a newer toolchain to reclaim the LTO gain.
- [ ] **BC7/BC5 texture compression** on load (M3 will start with RGBA8 uploads → more VRAM).

## Correctness / robustness
- [ ] **ExtractedCpuData warning**. The P1 memory-free churns the resource for a few frames → Bevy
  logs "Removing resource … not expected, may decrease performance". Benign (settles in ~4 frames),
  but a clean extract-once pattern (custom extract gated on buffers-built) would silence it.
- [ ] **Single cull view**. `upload_frustum` uses the first `CullCamera` view — fine for one camera;
  revisit for splitscreen / multiple cull cameras.

## Features (roadmap)
- [ ] **M3 textures/materials** (IN PROGRESS): bindless albedo + cutout first; then normal maps,
  emissive, glass/water, MicroSplat vert-paint terrain, SH-GI volume, EFT grade LUT.
- [ ] **M3b2 transparency polish** (Codex-flagged during M3b1, deferred — not among the 4 reported bugs):
  (a) back-to-front **sorting / OIT** for overlapping transparents (glass-behind-glass, stacked decals) — the blend pass is one unsorted multi-draw, so overlapping blend order is undefined;
  (b) **restrict the decal depth-bias to decals only** — glass/water currently share the positive bias and can bleed through geometry near opaque intersections (split decal vs non-decal blend pipeline, or a MAT_FLAG_DECAL sub-pass);
  (c) full **`vp` 3-layer splat** for tire-track/road detail; (d) **animated water** (flow/normal — needs new material params).
- [ ] **Volumetric lighting** (EFT haze + god-ray shafts — high wow-factor for interiors). Froxel
  (frustum-voxel) fog: a camera-aligned 3D texture (~160×90×64), compute-filled per froxel with
  in-scattered light = **ambient from the SH-GI volume** (already spatially shadowed by the bake →
  soft shafts near skylights/windows for FREE, no shadow map) + **sun HG forward-scatter** (g≈0.7,
  `sun_dir` from volume.json) × height-falloff fog density; raymarch front-to-back → (scatter,
  transmittance); composite over the scene by depth. Reuses the SH volume + the opaque pass's depth
  buffer (already written). CRISP shafts (hard sun occlusion) need a sun shadow map or the RT track
  — deferred; the SH-scatter gives soft shafts first. SEQUENCE: after surface shading (GI/spec/
  normals/grade), since it composites on top of the lit frame.
- [ ] **On-screen FPS / HUD**. FPS is console-only (`LogDiagnosticsPlugin`). Add an on-screen counter
  (the `egui` overlay is behind the non-default `egui` feature).
- [ ] **All 38 maps**. Only the Interchange `.eftpack` is built; wire per-map pack builds (one at a
  time — disk).
- [ ] **M4 raid planner**. Semantic overlays (`semantics.json`), ruler/measure, markers, routes via
  the `:8091` GPU pathfind server.

## Portability / distribution (TABLED — after the renderer "look" work)
- [ ] **In-GUI EFT extraction library + status UI**: fold the EFT extraction into the GUI as a separate library, with an informative in-app UI showing extract status/progress — so friends extract from their OWN game files without touching the CLI pipeline (portability model A: operate off game files).
- [ ] **Portable relative paths**: the `.eftpack` bakes ABSOLUTE machine-specific paths (`datasetPath` + every sidecar = `C:/Users/user/...`). Make `assemble_bevy.py` write paths RELATIVE to a configurable `EFT_ASSETS` root and resolve them at viewer load, so pack + `eft_assets` relocate cleanly. NOTE: all three viewers already share the `eft_assets` source-of-truth extraction from the game files — this is only about relocatable paths, not a data-source change.

## Cleanup
- [ ] 13 dead-code warnings in `eftpack.rs` (unused Manifest/Material fields, TERRAIN/BAKED flags,
  the `bounding_spheres` alt path). Most get used as M3+ lands; prune or `#[allow]` the truly dead.
