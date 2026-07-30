# Graphics Implementation Plan

> Produced by a Codex (gpt-5.6-sol) architectural review of docs/GRAPHICS.md $3 against the
> actual renderer source, 2026-07-30. Two of its headline corrections were independently
> verified before adoption: the 192-byte material stride test omits gpu_prepass.wgsl, and the
> prepass stores constant material roughness while 82% of materials derive it per-pixel (RFA).
> Read WITH docs/GRAPHICS.md — this document supersedes $3's ordering and cost claims.

## Executive assessment

All eleven techniques are implementable in some form, but the section-3 ranking and dependency graph should not be used as written.

The largest corrections are:

- The current 1× prepass cannot directly prime the 4× MSAA main depth. The prepass uses single-sample attachments (`gpu_driven.rs:6973-6991`), while the main pipeline uses the view’s MSAA sample count (`gpu_driven.rs:6792-6798`). Depth priming therefore depends on either a compatible MSAA depth-only pass or a validated TAA-driven transition to 1× rendering.
- Hi-Z cannot consume the current frame’s prepass without a second culling/compaction stage. The graph is presently `cull → shadow → prepass → main` (`gpu_driven.rs:1594-1603`).
- Halving SH spacing in X, Y, and Z creates approximately 8× as many probes, not 3×. The existing grid is capped at 2.6 million probes (`sh_bake.rs:41-50`) and explicitly widens spacing until it fits (`sh_bake.rs:502-517`).
- The sea is not discarded from the prepass. Untextured deep water is explicitly opaque (`gpu_driven.rs:2015-2023`), and the synthetic sea carries `MAT_FLAG_WATER` without `MAT_FLAG_BLEND` (`gpu_driven.rs:2953-2967`). It currently writes a flat normal and constant roughness, however, so water SSR still needs better prepass data.
- “Zero runtime cost” is too strong for denser SH and upscaled textures. Shader instruction count may remain constant, but larger working sets can reduce texture-cache residency. Runtime cost must be measured.
- Several sub-0.3 ms estimates are below the stated benchmark noise floor and should be treated as budgets, not claims.

The correct strategic order is: measurement and extracted data → prepass/history substrate → TAA/MSAA decision → depth/occlusion work → shared atmosphere → froxels/water → shadow and reflection consumers → offline quality improvements.

## Technique-by-technique review

| # | Technique | Verdict | Code-based correction and realistic cost position |
|---|---|---|---|
| 1 | SSR | Feasible, but understated | The graph and HDR ping-pong pattern exist in SSAO (`ssao.rs:159-270`, `ssao.rs:275-286`). But the prepass roughness is only `m.roughness` (`gpu_prepass.wgsl:165-170`), while the forward shader derives roughness per pixel from albedo alpha for RFA materials (`gpu_draw.wgsl:1399-1408`). SSR would therefore trace with wrong roughness across much of the scene. Deep water already enters the prepass, contrary to the document, but it receives the flat geometric normal rather than the animated `Nw` used by water shading (`gpu_draw.wgsl:1602-1635`). Hierarchical SSR also needs a depth pyramid and preferably a color pyramid, temporal stabilization, and miss fallback to `sky_reflect`; it cannot simply “replace” analytic reflection. Treat 0.5–1.5 ms as an aggressive half-resolution target; budget 0.8–2.0 ms until measured. |
| 2 | Depth priming | Conditional; proposal is incompatible as written | Main rendering is reverse-Z `GreaterEqual` and writes MSAA depth (`gpu_driven.rs:6782-6798`). The prepass writes separate 1× depth (`gpu_driven.rs:6973-6991`). That depth cannot be loaded as the main 4× attachment. `Equal` is also risky for alpha-tested/A2C geometry because the main opaque pipeline uses A2C (`gpu_driven.rs:6795-6798`) while the prepass uses hard discard (`gpu_prepass.wgsl:154-162`). The claimed 1–3 ms saving is unmeasured and applies mostly when the optional 5.4 ms in-fragment volumetrics are enabled; once froxels replace them, the saving will be smaller. Current woods data shows SSAO plus its prepass costs only about 0.20 ms end-to-end (`GFX_BENCH_woods_fly_2560x1440.json:25,53`), so the source comment’s “~1 ms” at `gpu_driven.rs:6951-6953` is not supported by the available run. |
| 3 | Froxel volumetrics | Strongest structural proposal | The current cost really is multiplied by forward-fragment execution: every surviving fragment calls `apply_fog`, which calls the 12-step march (`gpu_draw.wgsl:623-633`, `gpu_draw.wgsl:706-764`). A 160×90×64 grid is 921,600 froxels; evaluating CSR lights at every froxel and temporal reprojection are not automatically “cheap.” Temporal froxels can ghost too, especially around doors and moving camera edges. A target of 0.6–1.5 ms is credible on the 5090; acceptance should require ≤1.2 ms GPU time and at least 3.0 ms end-to-end savings against current volumetrics-on. |
| 4 | Physical sky | Blocked on game-data extraction | Sun direction alone is insufficient for Hosek/Preetham: turbidity/aerosol state, ground albedo, solar elevation/intensity, cloud attenuation and exposure are needed. Today even sun direction is authored by the baker (`sh_bake.rs:847-852`) and falls back again in the viewer (`main.rs:1496-1515`). The visible sky (`main.rs:1407-1420`), reflection sky (`gpu_draw.wgsl:608-614`), bake sky (`sh_bake.wgsl:122-123`) and fog (`gpu_draw.wgsl:143-147`) are indeed inconsistent. Runtime generation can be effectively free, but it must use one extracted atmosphere model in both the baker and viewer. Horizon color alone does not derive fog density. |
| 5 | Gerstner water | Feasible after extraction; current mesh is insufficient | Repository search confirms the Water4 Gerstner properties are not extracted. The synthetic sea is literally four vertices and six indices (`gpu_driven.rs:2969-3006`), so vertex displacement requires a camera-centred projected grid/clipmap or a substantially subdivided mesh. Wave displacement must be identical in main, prepass, and any caster path; otherwise depth, SSR and shadows disagree. Time is available to the main shader, but not the current 80-byte prepass uniform (`gpu_driven.rs:876-887`). Do not grow the pinned 192-byte material record; add a separate pinned water-parameter table. The 0.1–0.3 ms estimate is below the harness noise and ignores added tessellation. Target ≤0.3 ms GPU time, report end-to-end as “within noise” unless the delta exceeds 0.3 ms. |
| 6 | PCSS | Feasible but optional and potentially redundant | The shadow texture and comparison sampler exist (`gpu_driven.rs:4964-4965`), and current PCF performs nine comparisons per selected cascade (`gpu_draw.wgsl:771-810`). A blocker search can use depth loads from the same depth texture; no material-layout change is needed. But 9 blocker reads plus 12–24 filter taps is roughly 2–4× the current receiver sampling, so 0.3–0.8 ms is optimistic over a foliage-heavy screen. Penumbra scale also requires an extracted game shadow-softness/light-angular-size value; it cannot become another hand-tuned constant. PCSS should be evaluated after contact shadows, and the two should not necessarily ship stacked. |
| 7 | Denser, multi-bounce SH | Valid goal; implementation description is wrong | Uniform half spacing is ~8× probes and exceeds the current cap on maps already near 2.6 million (`sh_bake.rs:502-524`). Three RGBA16F volumes plus R8 validity consume 25 bytes/probe before overhead (`gpu_driven.rs:4512-4556`), so 2.6 million probes are about 62 MiB and an 8× version about 496 MiB—not “a 96 MiB volume.” Pass B is not a simple iterable function today: it samples pass A and emits `passA + bounce(passA)` (`sh_bounce.wgsl:194-200`). Later iterations require ping-ponging the previous converged field while still anchoring output to pass A. GPU bounce is experimental and off by default because of TDR risk (`sh_bake.rs:748-760`); CPU is the reliable default. Use adaptive refinement derived from geometry/validity/radiance gradients, not a uniform 2× grid. |
| 8 | TAA | Feasible, but not because “the world is static” | Doors are dynamic, grass moves from `sun.gfx.w` (`gpu_draw.wgsl:941-961`), water shading moves (`gpu_draw.wgsl:1587-1625`), and blends are excluded from the prepass. Camera-only reprojection covers only opaque static pixels. TAA therefore needs a reactive/class mask, explicit history invalidation on resize/map swap/camera cuts, rejection for pixels without valid prepass data, and either previous transforms for doors or rejection while the dynamic nonce changes. It belongs after SSAO/SSR composition and before Bloom. The 0.3–0.6 ms estimate is plausible but optimistic for full-resolution HDR neighborhood clipping; budget 0.4–0.9 ms. Its most important decision is whether it can replace 4× MSAA/A2C, which would unlock unified 1× depth priming. |
| 9 | Hi-Z culling | Feasible only after redesigning the cull sequence | `cs_cull` currently runs one thread per instance and directly compacts into per-mesh regions (`gpu_cull.wgsl:144-164`, `gpu_cull.wgsl:239-249`). Adding texture reads to every one of Woods’ millions of instances could cost more than the claimed 0.1 ms. Current-frame Hi-Z requires: coarse cull → prepass → pyramid → occlusion cull into a second visible/indirect set → main. A global coarse-survivor list is needed so stage two does not rescan every instance. Last-frame Hi-Z is simpler but needs conservative camera-motion dilation and uncertainty drawing; “two-phase draw-late correction” is not present. The 0.1 ms claim covers, at best, pyramid construction and omits occlusion testing and recompaction. |
| 10 | Offline texture upscaling | Technically feasible; conflicts with project policy unless constrained | The runtime texture preparation/cache path is at `gpu_driven.rs:5705-5781`, but that is the wrong insertion point for an offline pack transformation. An ESRGAN model invents learned detail rather than extracting game data, which conflicts with derive-don’t-author unless the project explicitly permits synthetic enhancement. A hand-authored hero-material list also violates the rule. Priority must instead derive from mesh world-area/UV-area, projected occupancy and source texel density. “Sampling cost unchanged” is too absolute: higher-resolution resident mips increase cache and bandwidth pressure. Full textures already add about 2.2 GiB in the existing bench. Keep this an optional derived-pack product with model/version provenance and a strict VRAM cap. |
| 11 | Screen-space contact shadows | Feasible; wiring is understated | Depth and sun direction exist globally, but they are not “already bound” to a new post pass. More importantly, post-darkening final HDR would bypass the renderer’s SH anti-double-darkening logic. The contact mask should be generated after the prepass and bound into main shading, where it modifies the existing shadow visibility before the gate at `gpu_draw.wgsl:1342-1380`. Eight to sixteen full-resolution steps may exceed 0.4 ms; half-resolution with depth-aware upsample is the correct first target. It will miss off-screen casters by construction. |

## Corrected dependency graph

```text
Extracted weather/TOD + Water4 data
              │
              ├── physical atmosphere ──> sky/reflections/fog ──> SH rebake
              └── Gerstner parameters ──> water grid/displacement

Prepass consumer manager
              │
              ├── accurate normal/roughness/class payload
              ├── depth pyramid + history matrices
              │       ├── Hi-Z
              │       ├── SSR
              │       └── contact shadows
              └── TAA/reactive mask
                      └── MSAA-off decision
                              └── unified depth priming

TAA history infrastructure ──> froxel temporal reuse

Contact shadows ──> PCSS incremental-value decision
```

## Phased implementation plan

### Phase 0 — Truth, measurement, and layout safety

Deliverables:

- Add independent feature switches for prepass, depth prime, froxels, TAA, Hi-Z, SSR, PCSS, contact shadows and water displacement.
- Add Vulkan timestamp queries around cull, shadow, prepass, pyramid, occlusion cull, froxels and each fullscreen pass. The existing harness only parses aggregate frame statistics (`bench_gfx.py:38-41`).
- Extend the 192-byte material stride test to include `gpu_prepass.wgsl`. It currently checks only `gpu_draw.wgsl` and `gpu_shadow.wgsl` (`gpu_driven.rs:7416-7432`), despite the prepass declaring the same record.
- Extract and serialize weather/TOD/atmosphere values and Water4 Gerstner properties. Missing values remain “unavailable”; do not silently author defaults.
- Record shader/model/cache schema versions in derived artifacts.

Acceptance:

- Five alternating A/B runs, 12 seconds each, at 2560×1440 using Woods all-LOD fly, Interchange mall fly, Streets canyon fly, and one coastal path.
- Report median average, p95 and per-pass GPU time. A delta under 0.3 ms is “within noise”; claim a frame-time improvement only at ≥0.6 ms.
- No Vulkan validation errors, NaNs, out-of-bounds bindless indices, or device loss on NVIDIA plus at least one AMD target.
- Pinned-layout tests cover Rust and every shader declaring `MaterialGpu`.

The two existing Woods files should not be combined as one baseline: nominally similar fly paths report 11.852 ms and 14.718 ms (`GFX_BENCH_woods_alllod_2560x1440.json:25`; `GFX_BENCH_woods_fly_2560x1440.json:25`), indicating different code/data states.

### Phase 1 — First-class prepass and screen-space substrate

Deliverables:

- Replace the SSAO boolean gate at `gpu_driven.rs:6951-6953` with a consumer mask: SSAO, SSR, TAA, Hi-Z, contact shadows, and diagnostic depth.
- Keep it consumer-driven, not unconditionally always-on.
- Make prepass roughness match forward RFA roughness.
- Add a compact class/reactive target or equivalent mask distinguishing static opaque, deep water, cutout/grass-invalid, dynamic, and no-data pixels.
- Build a reverse-Z R32Float depth pyramid after the prepass.
- Store current/previous unjittered view-projection matrices and invalidate history on resize, map swap, camera cut, or resource rebuild.

Acceptance:

- With only SSAO enabled, rendered output matches current SSAO within one 8-bit LSB outside known roughness corrections.
- No consumer: prepass and pyramid GPU timestamps are exactly absent.
- All enabled consumers share one prepass and one pyramid.
- Prepass plus pyramid ≤0.6 ms GPU on Woods and Interchange.
- Deep-water pixels contain valid depth/class and forward-matching roughness; blend glass/puddles remain invalid.

### Phase 2 — TAA and the MSAA decision

Deliverables:

- TAA after SSAO/SSR and before Bloom.
- Camera reprojection from depth, neighborhood clipping, exposure-aware history, reactive rejection, and explicit dynamic invalidation.
- Temporally stable cutout policy. If testing 1× main rendering, replace A2C with derived stochastic/hashed coverage rather than simply dropping coverage AA.
- A/B modes: 4× MSAA+TAA, 1×+TAA, and current 4×+FXAA.

Acceptance:

- TAA GPU time ≤0.8 ms at 1440p.
- Static-camera output converges without visible pumping.
- Fly paths show no trails longer than two frames around doors, grass silhouettes, water edges, or disocclusions.
- 1×+TAA may replace 4× only if treeline/cutout shimmer is no worse in captured fly sequences and end-to-end time improves by ≥0.6 ms.
- If 1× fails, retain 4× and treat depth priming as a separate MSAA depth-only experiment.

### Phase 3 — Depth priming and Hi-Z

Deliverables:

- If Phase 2 accepts 1×: render the prepass directly into the main depth attachment; use `Equal` only for genuinely identical opaque geometry and `GreaterEqual` for cutouts/exception classes.
- If 4× remains: prototype a depth-only 4× prime separately and retain it only on measured net benefit.
- Add a coarse-survivor list during existing culling.
- Execute same-frame occlusion flow: coarse cull → shadow → prepass → pyramid → occlusion cull into secondary indirect/visible buffers → main.
- Use conservative projected sphere bounds, reverse-Z max reduction, mip selection from screen bounds, strict self-occlusion bias, and bypass for uncertain near-plane/large/dynamic objects.

Acceptance:

- Zero false occlusion in scripted camera cuts and fast lateral fly paths; automated conservative reference can compare Hi-Z-off visibility.
- Interchange and Streets main-pass visible instance/triangle counts fall by at least 30% in occluded paths.
- Pyramid plus occlusion cull ≤0.5 ms GPU.
- Hi-Z must save ≥0.6 ms end-to-end on at least Interchange or Streets and regress Woods by no more than 0.3 ms.
- Depth priming is retained only if net-positive with both volumetrics off and on. Do not justify it solely with the soon-to-be-replaced in-fragment march.

### Phase 4 — Shared physical atmosphere

Deliverables:

- One atmosphere evaluator, parameterized only by extracted game data, used by:
  - CPU cubemap generation;
  - `sky_reflect`;
  - fog color/aerial perspective;
  - SH bake sky radiance;
  - volumetric scattering.
- Preserve a clearly marked legacy fallback for packs without extracted data; do not claim it is derived.

Acceptance:

- Sky disc direction, shadow direction and baked SH direction agree within 0.1°.
- Cubemap and shader evaluator agree within 1% linear RGB on a fixed direction set.
- Rebaked outdoor probe irradiance agrees with the runtime sky integral within 3%.
- Runtime sky cost remains within noise.
- No authored per-map constants.

### Phase 5 — Froxels and Gerstner water

Froxel deliverables:

- Start at 160×90×64, logarithmic Z.
- Compute sun visibility and single scattering once per froxel; integrate front-to-back.
- Add CSR local lights only after sun-only performance is proven.
- Reproject history using Phase-2 matrices with depth/camera-cut rejection.
- Replace the per-fragment `volumetric_inscatter`; do not stack both.

Froxel acceptance:

- ≤1.2 ms GPU including integration and composite.
- ≥3.0 ms average improvement versus current volumetrics-on Woods fly.
- No visible history lag longer than two frames around doors or window shafts.
- Adding local lights remains ≤0.4 ms and produces no light beyond extracted range/cone bounds.

Water deliverables:

- Extract Water4 properties into a new versioned sidecar.
- Add a camera-centred projected grid/clipmap for synthetic ocean only.
- Share identical displacement code and time inputs across main and prepass; update conservative bounds from extracted maximum amplitude.
- Add water motion/reactive information for TAA.
- Do not modify the 192-byte `GpuMaterial`.

Water acceptance:

- Main and prepass depth differ by less than one Depth32Float ULP at tested water pixels.
- No horizon cracks, shoreline gaps, grid swimming, or frustum culling at wave crests.
- GPU regression ≤0.3 ms.
- Missing Gerstner data produces current flat geometry exactly.

### Phase 6 — Contact shadows, PCSS, and SSR

Implement contact shadows first:

- Half-resolution ray mask between pyramid and main.
- Compose with existing sun visibility before SH/diffuse/specular shadow gating.
- Reject invalid/no-prepass pixels and cap distance from extracted shadow settings.

Acceptance: ≤0.4 ms GPU, no post-process double-darkening, and visibly closes the bias gap on the small-prop test set.

Then PCSS:

- Add non-comparison blocker reads, derived penumbra scale, rotated Poisson filtering, and cascade-safe radius clamps.
- Compare PCSS alone, contact alone, and both.

Acceptance: monotonically increasing penumbra with receiver/blocker separation; no cascade seam regression; ≤0.8 ms GPU. Ship both together only if blind captures show incremental value.

Then SSR:

- Reuse the pyramid, add a scene-color mip chain, half-resolution hierarchical trace, confidence mask, temporal resolve and depth-aware upsample.
- Preserve SH/physical-sky reflection on misses and rough surfaces.
- Use animated water normals from the prepass water path; do not apply flat-normal SSR to the sea.

Acceptance:

- ≤1.5 ms GPU at 1440p.
- No persistent trails longer than two frames.
- No reflection across depth discontinuities larger than one depth-aware tolerance.
- Water/building reflection tests show at least 80% valid hit coverage where the reflected geometry is on-screen.
- Analytic fallback remains stable when the reflector’s source moves off-screen.

### Phase 7 — Adaptive SH and optional texture enhancement

SH deliverables:

- Add ping-pong iterative bounce: `L[n+1] = Ldirect + K(L[n])`.
- Stop from a derived convergence criterion, not a hand-selected visual bounce count.
- Replace uniform half-spacing with adaptive bricks or refinement selected from geometry proximity, validity discontinuity, and first-pass radiance gradient.
- Enforce a serialized size/VRAM budget.

Acceptance:

- Successive-bounce RMS energy change below 2% before stopping.
- No channel gains violating the diffuse energy bound.
- ≤256 MiB runtime SH allocation per pack.
- Runtime frame delta versus current SH ≤0.3 ms.
- Interior probe validation set improves irradiance error by at least 20% against a high-ray reference.

Texture deliverables:

- Keep source PNGs immutable.
- Compute candidates automatically from source texel density and projected world coverage.
- Generate a separate, versioned derived cache; store model/hash/provenance.
- Require an explicit project decision that learned hallucination is allowed. Otherwise use deterministic reconstruction and do not describe it as recovered game detail.

Acceptance:

- Fixed VRAM growth cap, recommended ≤1 GiB at Full quality.
- Runtime frame delta ≤0.3 ms on all paths.
- No increase in temporal shimmer or BC artifacts in fly captures.
- No hand-authored per-material priority list.

## What section 3 is missing

1. A TAA/MSAA transition plan. This is the key dependency for making the existing single-sample prepass a true depth prime.

2. Accurate prepass semantics. It currently stores geometric normal plus constant material roughness, not the forward shading normal or per-pixel roughness. That is sufficient for SSAO but insufficient for high-quality SSR.

3. A shared depth-pyramid resource and graph stage. SSR, contact shadows and Hi-Z should not each build their own hierarchy.

4. A reactive/dynamic classification channel. Camera-only reprojection is not enough for doors, grass, water or transparent surfaces.

5. Resource-lifetime and invalidation rules: resize, map swaps, camera cuts, pipeline rebuilds, door changes and history warm-up.

6. Feature interaction budgeting. SSR, PCSS, contact shadows, TAA and froxels cannot all consume their top-end estimates while holding a 12 ms baseline. Structural savings from froxels/Hi-Z/MSAA reduction must fund optional effects.

7. Per-pass GPU timing. End-to-end frame averages cannot distinguish several proposed costs from the 0.3 ms noise floor.

8. AMD validation as an acceptance gate. The material stride test currently omits the prepass shader even though stride mismatch has already caused device losses.

9. A clear derive-don’t-author policy for physical sky, PCSS softness, Gerstner waves and ML upscaling. Each currently invites new hand-tuned or invented values.

10. A distinction between “implemented” and “default enabled.” All eleven can be implemented, but SSR, PCSS and learned upscaling should remain opt-in unless the full shipped stack stays at or below 12.0 ms average on the Woods Ultra reference, with p95 no worse than baseline by more than 0.3 ms.
