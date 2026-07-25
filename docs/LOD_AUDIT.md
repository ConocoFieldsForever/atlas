# LOD audit — what shipping multi-LOD packs by default would actually cost and buy

Question: what is required to make the **build pipeline** ship multi-LOD packs **by default**, so the
viewer stays performant when it runs as an overlay next to the game and shares the GPU?

Everything below is measured on this machine against real packs and real datasets. Every claim
carries a `file:line` or a number you can re-derive. Where a number is a *model* rather than a
measurement it is labelled as such, and the model's error against ground truth is stated.

**Verdict up front:** the viewer's distance-LOD path is finished and correct, but making the
pipeline ship all-LOD packs by default is **the wrong first lever**. On interchange, multi-LOD cuts
drawn triangles to ~45% while growing resident geometry from 715 MiB to ~997 MiB; on streets it
would take a 2.64 GiB vertex/index buffer to ~3.9 GiB. For an overlay that shares VRAM with the
game, that is the wrong side of the trade. Meanwhile a threshold the pipeline **already ships in
every manifest** — the LODGroup's own cull height — is unused, and honouring it cuts drawn
triangles to **2-9%** on interchange with **zero** extra bytes, zero re-extraction, and no format
change. Do that first. Details in §3 and §7.

---

## 0. Starting facts, confirmed and one correction

| Claim | Status |
|---|---|
| Viewer distance-LOD is implemented (`ids.z`/`ids.w` window, `lod_centers`, `cs_cull` modes 0/1/2) | **Confirmed**, and it is more complete than `LOD_DISTANCE_PLAN.md` describes |
| `GfxSettings::lod_distance` now defaults ON, `EFT_LOD=0` forces off | **Confirmed**, `viewer/src/render/mod.rs:176` |
| Every shipped pack is lean (one shell per group) | **FALSE for `factory_rework`** — see below |
| `tools/build_map.py --alllod` / `EFT_ALLLOD=1` plumbs `--alllod` + `--keep-lods` | **Confirmed**, `tools/build_map.py:349-356,430-432,526` |
| `assemble_bevy.py` gates its dedup on `KEEP_LODS` | **Confirmed**, `eft_pipeline/assemble_bevy.py:554,640-661` |

### The correction: `factory_rework` is already a multi-LOD pack

`packs/factory_rework.eftpack` ships **6,786 non-default shells out of 39,709 instances (17.1%)**,
across 4,119 multi-shell LOD groups (2 shells: 2,584; 3 shells: 1,070; 4 shells: 465). The shells
are genuine LOD siblings, not id collisions: within a multi-shell group the pivot spread is
0.000 m at the 90th percentile, and 3,985 of 4,119 groups have all shells within 1 cm. Mesh names
confirm it (`folders_04_LOD0/1/2/3`, 483/246/100/30 triangles).

Its dataset is a genuine `--alllod` extraction —
`target/release/eft_assets/factory_rework/scene.json` (49.0 MB, 73,214 instances) has
`lod.i` histogram `{0: 58818, 1: 5959, 2: 2212, 3: 617}`.

Two consequences:

1. **The `--alllod` path is not theoretical — it has been run end to end and produces a pack the
   viewer loads.** That retires most of the "never exercised" risk in `LOD_DISTANCE_PLAN.md:27-36`.
2. **With `lod_distance` now defaulting ON, distance-LOD is live *today* on `factory_rework`** —
   including the unfixed door blocker in §4.1. This is the only pack where a regression can appear
   without any pipeline change.

Note there are **two dataset roots** and they disagree:
`<EFT_ASSETS_ROOT>` (stale factory_rework, LOD0-only, 2026-07-13)
and `<exe dir>\eft_assets` (what recent `packs/logs` builds
actually used). Any future measurement must state which root it used.

---

## 1. Extraction

### 1.1 LOD0 is resolved at extraction time, and coarse meshes are never written

The entire decision is one predicate, `keep_renderer` — `extraction/unity/eft_extract_v2.py:1196-1211`:

```python
def keep_renderer(rpid):
    if rpid in billboard_only_rids: return False   # impostors: no billboard mesh ships, ever
    if args.alllod:                 return True    # keep every level
    if rpid not in all_lod_rids:    return True    # ungrouped renderer
    gi, li = rid2lod.get(rpid, (None, None))
    if gi is None:                  return rpid in lod0_rids
    return li == group_min_lod.get(gi, 0)          # finest PRESENT level, not literally 0
```

It is applied at `:1221` (`if not keep_renderer(o.path_id): continue`) **before** `export_mesh()`
at `:1245`. So with `--alllod` off a coarse shell's OBJ **is never written to disk**. This is the
single most important fact for migration: coarse geometry is absent, not merely unreferenced.

`--alllod` is declared at `:651`; `extraction/unity/extract_parallel.py:261,275-276` only passes it
through to each chunk subprocess.

The per-instance tag is written at `:1260-1261`: `inst["lod"] = {"g": <global group index>, "i":
<unity lod index>}`. Terrain instances are never tagged.

**Measured: in a default extraction the tag carries no information.** Across nine datasets,
`lod.i > 0` occurs **twice in total** (ground_zero — the empty-LOD0-slot fallback at `:1187-1192`
firing). streets: 278,495 tagged, all `i == 0`. icebreaker: 154,099, all 0. Every referenced group
has exactly one distinct `lod.i`.

### 1.2 The screen-height thresholds are already extracted, and they are complete

`eft_extract_v2.py:1128-1194`, one pass over `LODGroup` objects:

```python
srh = [round(_fin(L.get("screenRelativeHeight")), 5) for L in mlods]
ftw = [round(_fin(L.get("fadeTransitionWidth")),  5) for L in mlods]
grp = {"size": round(_fin(float(d.get("m_Size",1.0) or 1.0) * (wscale or 1.0), 1.0), 4),
       "center": [...], "fadeMode": ..., "lastIsBillboard": last_bb,
       "srh": srh, "ftw": ftw, "n": len(mlods)}
if last_bb: grp["cullH"] = srh[-1]
```

* **Complete by construction** — `srh`/`ftw` are comprehensions over `m_LODs` and `n == len(mlods)`.
  Verified: `len(srh) != n` in **0** groups across all maps measured.
* **`size` is the only derived field**: `m_Size x max world-axis scale` (Unity lossyScale rule).
* **Sanitization is NaN/Inf-only** (`_fin`, `:1151-1156`) — motivated by a real Reserve group
  shipping `fadeTransitionWidth = NaN`.
* **Not sanitized: monotonicity.** Measured non-monotonic (ascending) `srh`: factory_rework **9**,
  streets **3**, shoreline **18**; zero elsewhere. Example (factory_rework)
  `g=13603 n=3 srh=[0.08619, 0.20301, 0.10175]` — LOD1's threshold exceeds LOD0's, so a naive band
  selector yields an inverted window. The viewer already degrades these safely (§4.5).
* **Lossy NaN coercion**: `_fin` maps NaN to `0.0`, so a genuine `srh == 0.0` is indistinguishable
  from a sanitized NaN. Occurrences: icebreaker **104**, streets **7**. Which were originally NaN is
  unrecoverable from the datasets.

**Answer: yes, the thresholds are correct and complete enough to drive selection.** They already do,
on factory_rework.

### 1.3 Re-extraction is required, and it cannot be done incrementally in practice

Because coarse OBJs were never written, going multi-LOD **requires a re-extract**. A re-assemble is
not enough.

Worse, `--alllod` alone is silently ignored on an existing dataset: the stage-1 gate is
`tools/build_map.py:420` `if force or not os.path.isfile(<dataset>/scene.json)`. The source
acknowledges this at `:351-353`. You must pass `--alllod --force`.

And `--force`'s documented promise ("never deletes the big mesh/texture exports",
`build_map.py:346-348`) is **false in effect**: deleting `scene.json` re-arms stage 1, and stage 1
on the default multi-job path runs `extract_parallel.py:371-373` `shutil.rmtree(out)` — wiping the
whole dataset dir, meshes and textures included. Measured: today's streets force-rebuild printed
"no dataset yet - running the ONE-TIME full extraction" and re-exported all 80,743 OBJs in 602 s.

The only incremental path is `EFT_JOBS=1` (`extract_parallel.py:289-292` runs the extractor straight
into `out`, where the skip-if-exists at `eft_extract_v2.py:1039` fires) — but it is fully serial:

| map | levels | parallel wall (measured) | serial-equivalent (sum of per-level times) |
|---|---|---|---|
| streets | 245 | **602 s** (22 jobs) | 9,103 s (~152 min) |
| reserve | 44 | 1,024 s | 1,879 s |
| customs | 24 | 522 s | 930 s |
| shoreline | 17 | 133 s | 332 s |

So for streets, "incremental" costs ~2.5 h serial versus ~10 min for a full parallel re-extract.
**Incremental upgrade is not worth it.** Textures survive either way — `.texcache` is
content-addressed (blake2b over raw source bytes, `eft_extract_v2.py:69-93`) and lives outside the
dataset dir; the streets rebuild scored 36,982 hits / 6,275 misses.

**No `--alllod` timing exists for any map** — factory_rework's build predates the current log. I will
not guess a multiplier; the honest floor for a streets all-LOD extract is the measured 602 s plus
mesh-export work proportional to the extra shells.

### 1.4 Disk cost, measured

The only real lean-vs-all-LOD dataset pair on this machine (factory_rework, two roots):

| | OBJ bytes | OBJ files | dataset total |
|---|---|---|---|
| LEAN (beamng root, 60,973 instances) | 864.0 MB | 5,768 | 4,636.7 MB |
| ALL-LOD (live root, 73,214 instances) | **1,065.1 MB** | **7,610** | 3,615.4 MB |

**+23% OBJ bytes, +32% OBJ files.** Caveat: the two extractions are from different dates and
differ in instance count (60,973 vs 73,214), so this is indicative, not a controlled A/B. For
scale, streets' OBJ set alone is **11.2 GB** today across 80,743 files, in a 20.2 GB dataset.

---

## 2. Assembly

With `KEEP_LODS` off, `eft_pipeline/assemble_bevy.py:644-659` groups by `(lv, lod.g)` and keeps only
`lod.i == group-min`. On an already-LOD0-resolved dataset this is a **no-op** — confirmed in three
build logs, e.g. `packs/logs/build_streets.log:18` "LOD-shell dedup: 227,241/227,241 instances kept
(0 coarser LOD shells removed)". With `--keep-lods` it prints the `:661` message instead.

Per-instance `lodGroup`/`lodIndex` are written into `instances.bin` at `:862-864` in the 80-byte
record declared at `:98-105` (`lodGroup i32 @52`, `lodIndex i32 @56`).

The `lodGroups` manifest table is emitted at `:877-883,941` — the extractor's dict verbatim, with
only `center` conjugated into viewer world by `G3`. Every shipped pack has it:

| pack | `lodGroups` entries | referenced by >=1 instance | orphaned |
|---|---|---|---|
| streets | 137,311 | 132,066 | 5,245 |
| interchange | 76,723 | 58,125 | 18,598 |
| icebreaker | 81,489 | 81,019 | 470 |
| factory_rework | 25,659 | 24,019 | 1,640 |

Orphans come from the structural culls dropping every renderer of a group; they are harmless. **No
pack has a single out-of-range `lodGroup` index** (measured: 0 in all four), so the positional join
`instance.lodGroup -> lodGroups[i]` is sound.

**Shells per group, as Unity authored them** (`n = len(m_LODs)`, from the shipped manifests — this
is what the extractor saw *before* dropping):

| n | streets | interchange | icebreaker | factory_rework |
|---|---|---|---|---|
| 1 | 49,344 | 27,854 | 79,534 | 19,885 |
| 2 | 30,514 | 8,837 | 873 | 2,197 |
| 3 | 24,357 | 16,716 | 146 | 1,382 |
| 4 | 27,802 | 4,718 | 466 | 555 |
| 5 | 49 | 0 | 0 | 0 |
| groups with `lastIsBillboard`/`cullH` | 442 | 1 | 0 | 0 |

So EFT LODGroups are overwhelmingly 1-4 levels, and **63% of streets' groups declare more than
one** — but only 442 of them ship a billboard cull height.

---

## 3. The real cost and benefit — measured

### 3.1 Cost (a): pack bytes

`n`-based projection overestimates, because a declared level does not always have a distinct
renderer. Ground truth from factory_rework: declared `n` predicts **8,476** extra shells, actual is
**6,786** — a **realisation factor of 0.801**, applied below.

Byte cost per shell, measured across factory_rework's 460 multi-level mesh families
(mesh bytes = `vtxCount x stride + idxCount x 4`, vs the family's LOD0):

| level | bytes vs LOD0 (median) | triangles vs LOD0 (median) | n |
|---|---|---|---|
| LOD1 | **0.509** | 0.460 | 449 |
| LOD2 | **0.284** | 0.242 | 201 |
| LOD3 | **0.105** | 0.081 | 107 |

**Model validation** — applying this model to factory_rework's own lean subset predicts 263.3 MiB
against an actual 244.8 MiB: **+7.6% high**. The projections below are therefore mild
over-estimates.

| pack | `meshes.bin` today | all-LOD (calibrated) | instances today | all-LOD instances | `instances.bin` |
|---|---|---|---|---|---|
| **streets** | **2,642 MiB** | **~3,871 MiB (x1.47)** | 173,260 | ~341,000 (x1.97) | 13.2 -> 26.0 MiB |
| **interchange** | 715 MiB | ~997 MiB (x1.39) | 70,147 | ~122,000 (x1.74) | 5.4 -> 9.3 MiB |
| icebreaker | 96 MiB | ~110 MiB (x1.14) | 93,970 | ~96,000 (x1.02) | 7.2 -> 7.3 MiB |
| factory_rework (**actual**) | 201.0 MiB lean-equiv | **244.8 MiB (x1.218)** | 32,923 default | **39,709 (x1.21)** | 3.2 MiB |

The repo's "~47% bigger" claim is about **instances**, and it checks out: streets +49.2%,
interchange +42.6% of the all-LOD total. **Bytes grow less than instances** (x1.39-1.47), because
coarse shells are cheap. But note the asymmetry: icebreaker gains almost nothing (x1.02 instances)
while streets nearly doubles.

### 3.2 Cost (b): VRAM — this is the decisive one

`compute_cpu_blob` packs **every shell** on a multi-LOD pack
(`viewer/src/render/gpu_driven.rs:1378-1383`, `multi_lod -> pack.instances_by_mesh()`), and the
geometry ends up in single **fully resident** vertex/index buffers. The async path streams them over
the loading window rather than one-shot memcpy'ing them on the finalize frame
(`gpu_driven.rs:3428-3431`, which puts the current cost at "~1.1 GiB"), but that is a *load-time*
optimisation: nothing is evicted afterwards, and there is **no residency budget**. So the
`meshes.bin` growth in §3.1 is VRAM growth, one for one:

* streets: **2.64 GiB -> ~3.87 GiB resident** vertex+index.
* interchange: 715 MiB -> ~997 MiB.

That is before textures. For a viewer whose stated future is running **as an overlay sharing the
GPU with EFT itself**, adding ~1.2 GiB of resident geometry on the biggest map to save raster time
is the wrong direction — the game will be contending for exactly that memory.

**State this plainly: multi-LOD trades memory for draw cost. It does not save memory. It costs
~40% more VRAM to draw ~55% fewer triangles.**

### 3.3 Cost (c): build time

Extraction dominates and is unmeasured for `--alllod` (§1.3). Assemble grows roughly with instance
and mesh count; the assemble hotspot is OBJ text parsing (~69% of the loop, per `PERF_FINDINGS.md`),
so +32% OBJ files is close to a +32% assemble cost. The SH and nav bakes are **not** affected —
both already filter to default shells (§4.7).

### 3.4 Benefit: drawn triangles

Simulated by applying Unity's own rule (`relativeHeight = size x proj11 / 2d`, shell `i` active
while `srh[i] <= h < srh[i-1]`) to real pack data — group centres and `srh` from the manifest,
LOD0 triangle counts from `meshes.bin` metadata, coarse triangle counts from the measured ratios in
§3.1. fov 60 deg (`viewer/src/main.rs:114`), 1440p, frustum-culled, averaged over 8 yaw directions
at three poses. **This is a model, not a capture — no GPU work was run.**

| pack / pose | triangles today | distance-LOD | **cull-past-coarsest only** |
|---|---|---|---|
| interchange / center | 30,184,125 | 13,463,053 (**44.6%**) | 2,697,392 (**8.9%**) |
| interchange / quarter | 34,420,303 | 15,976,936 (46.4%) | 1,300,553 (3.8%) |
| interchange / overview | 40,723,885 | 17,331,036 (42.6%) | 937,920 (2.3%) |
| factory / center | 5,304,291 | 4,302,342 (81.1%) | 1,849,880 (34.9%) |
| factory / quarter | 5,115,315 | 4,102,333 (80.2%) | 556,539 (10.9%) |
| factory / overview | 7,031,976 | 5,734,555 (81.5%) | 597,164 (8.5%) |

Two readings, and the second is the important one:

1. **Distance-LOD is a real ~2.2x raster win on interchange** (45% of today's triangles) and a weak
   one on factory (81%) — factory is a small indoor map where most objects are close, so most
   instances stay on their finest shell.
2. **The last column needs no coarse shells at all.** Unity culls a LODGroup entirely below its last
   `srh`. The viewer does not: `gpu_driven.rs:1435-1440` applies a far bound only when
   `last_is_billboard && cull_h > 0` — which is **442 groups on streets, 1 on interchange, 0 on
   icebreaker and factory**. Honouring the last `srh` as a cull for *all* groups, on the packs we
   already have, cuts interchange to **2.3-8.9%** of today's triangles.

For calibration, the existing screen-size cull is far weaker: sweeping `cull_px` on interchange/center
gives 96.9% (1.5 px, the default), 88.4% (4), 67.0% (8), 31.8% (16) of today's triangles. The
LODGroup cull is more aggressive *and* it is the game's own authored intent rather than a tuned
constant.

**Caveat, and it is a big one.** EFT applies `QualitySettings.lodBias` (its "Object LOD quality"
slider) as a multiplier on these thresholds; the sim assumes bias 1.0. A higher in-game bias means
the game culls *later* than this model, so the 2-9% figure is the aggressive end. The viewer already
has `lod_bias` (`GfxSettings::lod_bias`, `EFT_LOD_BIAS`) wired into the same `proj11 * bias` product
(`gpu_cull.wgsl:162`), so this is tunable rather than a cliff — but the correct default must be found
by looking at the result, not by argument. Culling 91-97% of triangles will visibly change the image,
and "visibly changed" is only acceptable if it matches what the game itself draws.

---

## 4. Blockers and correctness risks

Six of the eight CPU consumers that `LOD_DISTANCE_PLAN.md:144-169` listed are **already fixed**.
`Pack::is_default_lod` exists (`viewer/src/eftpack.rs:1373-1375`), backed by a mask built in
`Pack::load` (`:993-1013`) whose rule is *ungrouped -> true, else `lod_index == min_present(group)`*
— correctly the finest **present** shell, not literally 0, so reserve's LOD2-only window groups
survive. Filtered sites: `nav_bake.rs:288-293`, `sh_bake.rs:392` (transitively, via `build_tris`),
`pick.rs:237-242`, `walk_ground.rs:167-172`, `render/standard.rs:416-420`,
`render/instancing.rs:158-162`. `poi.rs` and `loot.rs` no longer touch `pack.instances` at all
(`poi.rs:2217-2222` documents the removal of the geometry-mining tier).

### 4.1 B1 — doors animate one shell. **Severity: HIGH.** Live today on factory_rework.

`gpu_driven.rs:3454-3546`. The panel is matched as the nearest instance to the hinge within 1.5 m
(`:3475-3481`). On a multi-LOD pack every shell of a door sits at the *same pivot*, so all shells tie
at identical distance and the strict `dist < b` keeps the **first encountered** — and `cpu.instances`
is ordered grouped-by-mesh (`:2313-2340`), not by pack index, so the winner is whichever LOD mesh
appears first in `manifest.meshes`. That can be a coarse shell.

**Does the new `parts` rework fix it? No — it is orthogonal, and the data proves why.** `parts`
matching (`:3500-3516`) resolves each part **by mesh name** against `by_mesh`. The names the
extractor records are explicitly the LOD0 renderers:

```json
"parts": [["Outside_Door_Metal_08_L_210-140_door_LOD0", [-67.88, 0.89, -74.61]],
          ["Outside_Door_Metal_08_L_210-140_glass_01_LOD0", [...]], ...]
```

A coarse sibling is a *different mesh name* (`..._LOD1`), so it can never match. The parts rework
correctly fixes multi-**renderer** doors (leaf + glass + hardware swing together) and it makes part
matching deterministic — a genuine improvement — but it does nothing for multi-**shell** doors. The
symptom in the `:3456-3460` comment stands unchanged: open a door, walk away, and when `cs_cull`
mode 1 hands off to the next shell the door renders **closed**.

Measured exposure — door leaves whose nearest instance sits in a group declaring more than one LOD
level:

| map | doors matched | leaf in a multi-level group |
|---|---|---|
| interchange | 479 | **224 (47%)** |
| streets | 1,196 | **397 (33%)** |
| icebreaker | 94 | 0 (0%) |
| factory_rework (**actual multi-shell**, not projected) | 85 | 5 (6%) |

**The fix is small and needs no new plumbing.** `ids.z` bits 13+ already carry `lod_group` for every
instance with a genuine window (`gpu_driven.rs:1455`), and instances with `ids.w == 0` are
single-shell by construction (`:1415-1416`). So after `idxs` is built at `:3499-3516`, expand each
index to every instance sharing its `lod_group`, and animate all of them. `DoorPart.gpu_idx`
(`:3528`) is already a GPU-order index, which is the same space, so no pack-index map is needed.

### 4.2 B2 — extractor collapses a renderer's LOD span. **Severity: MEDIUM (extraction correctness).**

`eft_extract_v2.py:1177-1178` collapses a renderer listed in *several* LOD levels to its **min**
level; the in-file `AUDIT #3` note at `:1179-1186` flags it as deferred. On an all-LOD pack such a
renderer is emitted once, gets a window ending at its finest level's far bound, and then **vanishes
in the coarser bands where Unity keeps it visible** — a hole that only appears at distance.

Incidence is **unmeasurable from any dataset**: the full span is discarded at `:1177` and never
written. This must be fixed *before* an all-LOD build is trusted, and it is the one blocker that
requires an extractor change plus a re-extract to validate.

### 4.3 B3 — `content_anchor` double-counts. **Severity: LOW.**

`eftpack.rs:1264-1276` takes the per-axis median of **every** instance translation, unfiltered, and
feeds initial camera framing (`main.rs:1041-1044`). Heavily-LOD'd objects get 3-4 votes each, so the
opening camera lands somewhere different on an all-LOD pack than on the lean build of the same map.
No correctness break, but it silently breaks lean-vs-all-LOD screenshot parity — which is exactly
the gate you would use to validate the change. Fix with `is_default_lod`.

### 4.4 B4 — `ForcedLod` epoch bump is now inverted. **Severity: LOW (dormant).**

`main.rs:188-204` bumps `MapEpoch` (destructive: reframes camera, clears nav/pins/routes) when
`instances.iter().any(|i| i.lod_group >= 0 && i.lod_index > 0)` — a predicate that is **true exactly
on all-LOD packs**. But `compute_cpu_blob` ignores `lod` entirely when `multi_lod`
(`gpu_driven.rs:1378-1383`), so the rebuild is a byte-identical no-op that wipes user state.
Currently dormant: `crate::ForcedLod` is never written anywhere in the tree (the UI moved to
`g.lod_force`/`g.lod_distance`, `ui.rs:1329-1351`). Delete the path rather than fix it.

### 4.5 B5 — bad `srh` data. **Severity: LOW, already handled defensively.**

Non-monotonic `srh` exists (factory 9, streets 3, shoreline 18 groups). The encoder's guard at
`gpu_driven.rs:1444-1446` (`if !(near < far_b) || !near.is_finite() -> sentinel`) degrades these to
"always draw", which is the safe direction. Same for `srh == 0.0` (icebreaker 104, streets 7): `far`
returns `INFINITY` at `:1423-1427`, so the shell simply never switches out. No action needed, but do
not "fix" `srh` handling without preserving these fallbacks.

### 4.6 B6 — build gating traps. **Severity: MEDIUM (build correctness).**

`--alllod` without `--force` is silently ignored (§1.3) and produces a lean pack that *looks* like a
successful all-LOD build. And `--force` wipes the entire dataset via `extract_parallel.py:371-373`
despite `build_map.py:346-348` claiming otherwise. If all-LOD becomes the default, both must be
fixed or every rebuild is a 20 GB re-download-equivalent.

### 4.7 Verified indifferent (no change needed)

Nav bake, SH bake, pick, walk_ground, M0/standard fallbacks — all already filtered (above). Grass
and the synthetic sea append after the pack loop with `ids.w = 0`, so the LOD branch is skipped
entirely (`gpu_cull.wgsl:149`) and grass's `ids.z == 1` marker is untouched. Terrain is ungrouped:
measured, interchange's 4 TERRAIN-flagged instances all carry `lod_group < 0` — this confirms the
assumption every filter rests on. Shadows reuse the main cull's visibility buffer
(`gpu_shadow.wgsl:29,93-94`), so shell selection is automatically consistent. Blend/water
classification is per-mesh, so coarse shells classify independently and correctly. Load-time stats
(`gpu_driven.rs:2623-2629`) double-count — cosmetic, but it makes `[stall]` numbers incomparable
across pack types.

---

## 5. What "by default" would actually require

1. **Extraction**: fix B2 (`AUDIT #3` span collapse) first — it is the only silent-visual-hole bug.
   Then flip the `--alllod` default in `eft_extract_v2.py:1196-1211` / `build_map.py:354`.
2. **Build gating**: fix B6 so `--alllod` cannot be silently ignored, and make `--force` mean what
   it says (or rename it). Add an incremental-and-parallel path, or accept full re-extracts.
3. **Assembler**: flip `KEEP_LODS` default at `assemble_bevy.py:554`. No format change needed.
4. **Manifest/format version**: **none required.** `lodGroup`/`lodIndex` are already mandatory
   instance fields and `lodGroups` already ships in every pack. `manifest.version` stays 1.
5. **Migration for the 5 built packs**: **full re-extract, not a re-assemble** — coarse OBJs were
   never written (§1.1). Only `factory_rework` is already all-LOD. Budget the measured extraction
   walls (§1.3) plus unmeasured all-LOD overhead, and ~+23% dataset OBJ bytes.
6. **Coexistence — verified working.** The viewer branches on
   `pack.default_lod_mask.iter().any(|&d| !d)` (`gpu_driven.rs:1378`): a lean pack takes the old
   single-shell CPU path unchanged, and every instance on it gets `ids.w == 0` because
   `present.len() <= 1` (`:1415-1416`). `cs_cull` skips the whole LOD block when `ids.w == 0`
   (`gpu_cull.wgsl:149`), so **a lean pack renders bit-identically regardless of LOD mode**. Mixed
   fleets are safe; this is the one part of the design that needs no work.
7. **Doors (B1)** must land before all-LOD ships, and should land now regardless, because
   factory_rework is already multi-LOD with LOD mode defaulting on.

---

## 6. Validation

The instrumentation that exists:

* `[stall] build_cpu_data ... vtx_buf=<N>MiB idx_buf=<N>MiB` — `gpu_driven.rs:2707-2724`. This is
  the **VRAM proxy**; capture it for every before/after.
* `gpu-driven: assembled {meshes}, {instances}, {verts}, {indices}` — `:2623-2629`.
* `[doors] matched N of M swing doors (P parts)` — `:3547-3551`.
* `EFT_GEOM_SHA` — byte-identity of the geometry stream. Valid **only** when the packed set is
  supposed to be identical (lean-pack regression); it will and should differ lean-vs-all-LOD.
* Frame time via `FrameTimeDiagnosticsPlugin` + `LogDiagnosticsPlugin` (`main.rs:788`).
* Headless capture: `EFT_HIDDEN=1 EFT_POSE="x,y,z,yaw,pitch" EFT_SHOT=out.png`, plus `EFT_UNCAPPED`.

**The gap: nothing reports drawn triangles.** `cs_cull` writes draw counts into the indirect buffer
and no one reads them back. Any honest "we drew N% fewer triangles" claim needs either a readback of
the indirect buffer behind an env flag, or an external capture (RenderDoc / PIX / Nsight). **Add the
readback** — it is small, and without it §3.4 stays a model forever.

### Proposed before/after, on interchange

Interchange is the right map: 47% of its door leaves are multi-level (worst-case B1 exposure), it has
the largest raster win in the model, and at 715 MiB it is affordable to build twice.

1. Pick 3 poses from the in-viewer POS HUD: interior mall, mid parking, far overview. Record them.
2. **Baseline** (today's lean pack, LOD off): for each pose capture frame time (median of 300
   uncapped frames), `[stall]` vtx/idx MiB, and a screenshot.
   `EFT_LOD=0 EFT_HIDDEN=1 EFT_POSE=... EFT_SHOT=base_<pose>.png`
3. **Lever A — cull-past-coarsest on the lean pack** (the §7 phase 1 change): same poses, same
   captures. Expect a large frame-time drop, **identical** vtx/idx MiB, and screenshots that differ
   only by distant small props disappearing. Diff against the baseline and *look at what vanished* —
   this is the acceptance test, and it is a judgement call, not a threshold.
4. **Lever B — all-LOD pack**: `python tools/build_map.py interchange --alllod --force` (10-25 min;
   do not run it while the user is on the machine). Then the same three captures at `EFT_LOD=1` and
   `EFT_LOD=0`.
   * Gate 1: `EFT_LOD=0` on the all-LOD pack must be visually identical to the baseline (mode 0
     draws default shells only). Grass sway is the known nondeterminism floor (~30-70 px cluster,
     <=28/channel) — prove any diff is jitter with a same-binary double-run control.
   * Gate 2: `[stall]` vtx/idx MiB confirms the §3.1 projection (predict ~997 MiB; the model runs
     +7.6% high, so ~920-1000 MiB).
   * Gate 3: open a door, back away past a shell boundary, confirm it stays open (B1 fix).
   * Gate 4: frame time at each pose, LOD on vs off.
5. **Decide on the ratio that matters**: frame-time gain per MiB of added resident geometry, for
   lever A versus lever B.

Also re-run the lean-pack regression: `EFT_GEOM_SHA` byte-identical and screenshots mean=0 on
streets/icebreaker, proving the sentinel path is untouched.

---

## 7. Recommendation, smallest first

**Phase 1 — honour the LODGroup cull height on the packs we already have.** No re-extract, no
format change, no extra byte. Extend `gpu_driven.rs:1435-1440` so the coarsest present shell gets
`far = size / (2 * srh_last)` for *every* group, not only `last_is_billboard` ones — and put it
behind its own setting (`EFT_LOD_CULL`, default off until validated) so it can be A/B'd against the
current look. Modelled effect on interchange: **2.3-8.9%** of today's drawn triangles. This is by far
the best ratio of win to risk in this audit, and it is a viewer-only change. Validate per §6 step 3;
the risk is entirely visual (over-aggressive pop-out), and `lod_bias` is the mitigation.

**Phase 2 — fix the doors (B1).** Small, self-contained, and already live on factory_rework. Expand
each door's part set to all instances sharing its `lod_group`, read from `ids.z >> 13`.

**Phase 3 — fix B3/B4/B6** (content_anchor filter, delete the dead epoch bump, make `--alllod`
gating honest). Cheap hygiene that makes phase 4 measurable.

**Phase 4 — fix the extractor span collapse (B2), then build ONE all-LOD map (interchange) and
measure it** per §6. Decide with the frame-time-per-MiB number in hand.

**Phase 5 — default, only if phase 4 justifies it.** My expectation, from §3: it will justify itself
on interchange and **not** on streets, where +1.2 GiB of resident geometry is a poor trade for an
overlay. A per-map policy ("all-LOD for maps under N MiB") is more defensible than a global default.

### Is LOD-by-default the right lever?

**No — not first, and possibly not at all for the biggest maps.** Ranked by win-per-risk:

1. **LODGroup cull height** (phase 1) — largest modelled win, zero memory cost, zero pipeline
   change. The pipeline already ships the data; the viewer just ignores it.
2. **Pause/throttle rendering when hidden.** Not measured here, but for an overlay that is idle most
   of the time this dominates everything else, and it is nearly free. Worth measuring before any
   geometry work.
3. **`cull_px`** — already wired, instantly tunable, no memory cost. Weaker than the LODGroup cull
   (67% of triangles at 8 px vs 8.9%) and it is a tuned constant rather than authored intent, but it
   is the zero-effort dial.
4. **Texture budget** — not audited here. Note only that streets ships 6.2 GB of source textures
   against a 2.64 GiB geometry buffer, so texture residency is plausibly the larger VRAM lever. This
   is the most important thing this audit did **not** measure.
5. **Multi-LOD by default** — real raster win (~2.2x on interchange), but it is the only option on
   this list that *increases* VRAM, and it is the only one that needs a re-extract of every map plus
   an extractor bug fix.

### What I could not verify

* No GPU work was run and no viewer was launched, per the audit constraints. **Every triangle count
  in §3.4 is a model**, using measured inputs (real group centres, real `srh`, real LOD0 triangle
  counts, coarse counts from measured factory ratios) — not a capture.
* No `--alllod` build timing exists for any map, so the build-time cost of the default flip is
  unquantified. factory_rework proves the path works, not what it costs.
* The incidence of B2 (extractor span collapse) is unmeasurable from any existing dataset.
* Whether EFT's in-game `lodBias` makes the phase-1 cull too aggressive. This needs eyes on a
  screenshot, and it is the single judgement that decides whether phase 1 ships.
* Texture/VRAM residency behaviour was out of scope and is likely the bigger lever (see above).
* The two dataset roots disagree; all dataset numbers here name their root.
