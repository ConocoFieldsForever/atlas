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
- [ ] **On-screen FPS / HUD**. FPS is console-only (`LogDiagnosticsPlugin`). Add an on-screen counter
  (the `egui` overlay is behind the non-default `egui` feature).
- [ ] **All 38 maps**. Only the Interchange `.eftpack` is built; wire per-map pack builds (one at a
  time — disk).
- [ ] **M4 raid planner**. Semantic overlays (`semantics.json`), ruler/measure, markers, routes via
  the `:8091` GPU pathfind server.

## Cleanup
- [ ] 13 dead-code warnings in `eftpack.rs` (unused Manifest/Material fields, TERRAIN/BAKED flags,
  the `bounding_spheres` alt path). Most get used as M3+ lands; prune or `#[allow]` the truly dead.
