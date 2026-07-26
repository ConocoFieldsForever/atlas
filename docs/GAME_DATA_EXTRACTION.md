# Extracting data from the EFT game files — working notes

A handoff document for anyone (human or agent) extending Atlas's extraction pipeline. It records
what has been **verified against the real assets**, the conventions you must not get wrong, and the
dead ends, so nobody re-derives them.

Everything below is marked **CONFIRMED** (someone read the bytes), **PLAUSIBLE** (inferred but
consistent), or **DEAD END** (looked for it, it isn't there).

---

## 0. The governing rule

**Derive from the game; never author a constant.** If a number can be read out of the game's own
data, read it — do not hand-tune it and do not copy it from a wiki. Two bugs this project shipped
came from breaking that rule (a hand-picked sea-level heuristic that flooded Woods; nav climb/slope
limits ~3× more permissive than the game's own). When you must approximate, say so in the code
comment and record what the real source would be.

Corollary: prefer a *structural* test over a *magnitude* test. "Is this water body large?" was
wrong; "does this water reach the map boundary?" was right. "Is this triangle big?" was wrong for
fences; "does it span height?" was right.

---

## 1. Environment

| | |
|---|---|
| Game install | `C:\Battlestate Games\Escape from Tarkov` (assets under `EscapeFromTarkov_Data`) |
| Game logs | `<install>\Logs\log_<date>_<version>\` — note logs sit **beside the exe**, not under `..._Data` |
| Extracted datasets | `<workspace>\beamng_blender_pipeline\eft_assets\<dataset>` |
| tarkmap dir | `<workspace>\beamng_blender_pipeline\tarkmap` (holds `maps/` + `out/`) |
| Python with UnityPy | the repo venv: `.\venv\Scripts\python.exe` |
| tarkov.dev data | **json.tarkov.dev static catalogs ONLY** via `extraction/intel/tarkov_static.py` (ETag cache; offline = last snapshot). Never `api.tarkov.dev/graphql` — it 503s for hours; the static dumps are the feed tarkov.dev's own apps consume. Bonus: `maps_en`/`maps_ru` carry `Zone*`/`BotZone*` translations ("ZoneWoodCutter" → "Lumber Mill") — the exact table boss `spawnLocations` names render from. |

`EFT_ASSETS_ROOT` / `EFT_TARKMAP_ROOT` are **not set** in the shell by default — export them or pass
them explicitly. `tools/build_map.py` derives sane defaults if they are absent.

Read `ARCHITECTURE.md` for the `.eftpack` format and the extraction skill
(`tarkov-unity-extraction`) for the geometry/placement rules before touching the pipeline.

---

## 2. What is already extracted (don't redo)

- **Geometry / materials / terrain layers / lights / grass density / SH volume inputs** —
  `extraction/unity/eft_extract_v2.py`, consumed by `eft_pipeline/assemble_bevy.py`.
- **`gamedata.json`** — `extraction/intel/extract_gamedata.py`: exfils, typed doors (with `key_id`
  and `state`), minefields, sniper zones, quest triggers, loose-loot points, containers, transit
  points, switches/interactables, and (2026-07, this audit) **the full spawn system**: every
  `SpawnPointMarker` with GUID / zone / collider radius / categories (player scenes + AI scenes),
  `patrol_ways` (ordered PatrolWay polylines) and `bot_zones` (centroid + draped convex hull +
  friendly `en` name). The viewer routes markers to the PMC/Scav/Boss layers by side+mask
  (`poi.rs::gd_spawn_layer`), suppresses tarkov.dev's clustered spawn nodes when first-party
  markers exist, and snaps boss nodes onto zone centroids (exact `en` join, substring+nearest
  fallback).
- **`NavMeshProjectSettings`** — the agent climb/slope/cellSize values now used by the nav baker.

  **Adoption trap:** the `gamedata.json` inside a PACK is not the raw extractor output — build_map
  stage 6 MERGES the dataset's `interact_<lv>.json` into a `switches` array, tags power-gated
  extracts and wires switch→door links before copying. Never copy a fresh extraction straight
  into a pack; run `build_map.merge_gamedata_interactables(gd_path, dataset_dir)` first (the
  merge was lifted into that function precisely so adoption can reuse it).
  A raw extraction therefore reports `switches: 0` — that is the missing merge, NOT a regression.
  Never diff raw extractor output against a pack's gamedata without merging first.

### 2.1 SCAN-SCOPE TRAP: the level list is per-stage, and one stage had the wrong one (fixed 2026-07)

`extract_gamedata.py` defaults `LEVELS` to the **hand-curated `config.source.levels`**, while
geometry/interactables/grass all use `build_map.dataset_levels()` — the list DERIVED live from
BuildSettings. Those two disagree on almost every map, because the config omits the
`*_DesignStuff` scene and **DesignStuff is where the loot lives**:

| map | DesignStuff lv | in derived? | in config? | containers there |
|---|---:|---|---|---:|
| interchange | 52 | yes | **no** | 902 |
| reserve | 116 | yes | **no** | 992 |
| streets | 384-391, 454-457 | yes | **no** | 1278 |
| customs | 8 | yes | **no** | 552 |
| shoreline | 31 | yes | **no** | 761 |
| labs | 115 | yes | **no** | 319 |
| lighthouse | 189 | yes | **no** | 534 |
| woods | 166 | yes | yes | 431 |

Woods and factory_rework only worked by accident of their hand-written configs. Because geometry
DID scan those levels, the loot props were **rendered on screen while their typed records were
missing** — interchange shipped 5 of 907 containers, reserve 0 of 992, and customs 0 loose-loot
points. Fixed by passing `--levels=` at the stage-6 call site (`build_map.py`, "6: typed gameplay
zones"); interchange re-extracted to 907 containers / 112 loose points, confirmed.

**The general rule: if you add a new extractor, take the level list from `dataset_levels()`, never
from `config.source.levels`.** The config is authored, so by §0 it is the wrong source; it exists
only as a union fallback. Any per-stage level list is a place this bug can recur silently — the
symptom is plausible-but-small counts, never an error.
- **Scene-preset bundle → map id** — `gen_maps.derive_bundles()` reads each
  `StreamingAssets/Windows/maps/*.bundle` and joins its dominant location folder to the roster;
  `manifest.json` ships the stems and `maps.rs::bundle_to_id` replaced the hardcoded
  `bundle_to_map()` table (which had silently omitted `icebreaker.bundle`).

The viewer reads these via `viewer/src/eftpack.rs` and surfaces them in `viewer/src/poi.rs`.

---

## 3. Coordinate convention — get this right first

Unity world space → viewer world space is an **X flip** (`diag(-1, 1, 1)` conjugation). The
extractor's `bridge()` helper does it; **always** route positions through the same helper rather
than negating by hand.

For instance transforms the rule is absolute (from the extraction skill): **apply the raw 3×4
affine to the vertices. NEVER decompose to translation/rotation/scale** — EFT ships sheared and
mirrored instances and decomposition silently corrupts them.

Areas and Y are preserved by the flip, so map-scale/height tests can be done in raw Unity space
without bridging (see `derive_sea_level`).

---

## 4. The AI scenes — the richest untapped source (CONFIRMED)

Every playable location ships an `*_AI.unity` scene that the pipeline currently **skips**.
Interchange's is level 66 (`Shopping_Mall_AI.unity`); Woods' is level 42.

### 4.1 Do NOT simply un-skip it

`tools/gen_maps.py` `SERVICE_TOKENS` (~line 128) excludes `ai` / `culling` / `levelborders` scenes.
That exclusion is **correct for geometry**: level 66 alone holds 554 MeshRenderers of placeholder
cubes, spheres and cultist-sign quads that would pollute every pack. Instead add a separate
`ai_levels()` pass that scans those scenes **in addition to** the geometry levels. Measured cost:
**0.1–0.4 s per AI scene** — negligible.

### 4.2 `EFT.Game.Spawning.SpawnPointMarker`

The payload is **exactly the layout `dec_spawn` in `extraction/intel/extract_gamedata.py` (~line
227) already reads**; everything from `Infiltration` onward is new:

```
+0    Id            string (GUID)
      Position      float3      <- USE THIS (identical to the Transform; bridge() it)
+12   Rotation      quat4
+28   Sides         u32 mask
+32   Categories    u32 mask
+36   Infiltration  string
+40   Name          string      (== GameObject name)
      then: PPtr BotZone (12B) | float (40.0 in AI scenes, 4.0 in lv520)
            | int CorePointId | PPtr SphereCollider (12B) | 16B constant
```

**Category mask vocabulary (complete, small):** `Player=1`, `Bot=2`, `Boss=4`, `BotPmc=64`.
Observed masks: 0, 2, 3, 4, 6, 7, 64, 65, 67, 70, 71. Level 520 additionally uses bits 8/16/32
(`spawns_coop`=24, `spawns_op`=40) — **masks CONFIRMED, the names for 8/16/32 are PLAUSIBLE only, so
ship the raw mask alongside any decoded label.**

There is **no** Scav / Cursed / Marksman category — those are server-side bot *types*, not scene
categories.

**Sides:** AI-scene markers are `4` (Savage) on 1625/1628 sampled; level-520 player spawns are `3`
(PMC). So the AI scene is the **scav + bot + boss** set and *complements* the PMC spawns already
extracted — it does not duplicate them. Some maps also ship side `7` (All) player markers
(ground_zero ×100, labs ×75 — any-faction raid starts).

**Tail decode CONFIRMED at scale (2026-07 implementation):** the layout above validated on
interchange 102/102 AI markers (collider radius == every `rad:` token) AND 177/177 lv520 player
markers (null BotZone PPtr, default 50 m collider, `Infiltration` set, float 4.0). The trailing
16 B constant is `0,1,0,1` floats everywhere. Both PPtrs are always in-scene (fid 0). lv520 cats
histogram: `{1:130, 40:24, 24:22, 8:1}` — a **bare bit-8 exists**, so 8/16/32 are independent
mode bits. One lv520 pair (`spawns_coop (21)`/`(22)`) shares a GUID at the same position —
authoring copy-paste; Id-dedupe correctly collapses it (177 → 176).

**Two traps:**

1. **Radius comes from the `SphereCollider` on the same GameObject** (via the payload PPtr), not
   from the `rad:` token in the name. They disagree on 11 of 1032 markers; the collider is
   authoritative.
2. **Zone comes from the BotZone PPtr** → that MonoBehaviour's GameObject name (`ZoneTagilla`,
   `ZoneCenterBot`). Do **not** parse it out of the marker name: the prefix is `BP.Zone…` on
   Interchange/Woods/Customs but `BP.BotZone…` on Labs/Lighthouse/Factory, and free-form elsewhere.

**Name grammar** `BP.<Zone> <Roles> rad:<R> <N>` — `<N>` is a **0-based serial within the zone**
(e.g. ZoneBearCamp runs 0..14 for 15 markers), *not* a count or a priority.

Sample decoded record:

```json
{"go":"BP.ZoneTagilla Bot, Boss rad:70 3","id":"71971bcf-…","pos":[183.057,21.373,-130.534],
 "sides":4,"cats":6,"zone":"ZoneTagilla","radius":70.0,"core":5}
```

### 4.3 `PatrolWay` / `PatrolWayWithName` / `PatrolWayWithConditions`

```
+0  u32 type
+4  u32 N
    N × PPtr PatrolPoint (12B each)   <- IN ROUTE ORDER (CONFIRMED)
    0xFFFFFFFF | 1.0f | name string   (name only on PatrolWayWithName)
```

Order is genuine: the trailing index in each point's GameObject name matches its array index.
`PatrolPoint` payloads are all zero — **take the position from the Transform.**

Real decoded route, `Boss_Killa_way` (level 66, zone `ZoneCenterBot`, viewer space):

```
[-68.67,27.09,-76.22] [-62.08,27.09,-36.56] [-29.25,27.09,-3.40] [-40.59,27.09,-75.22]
[ -9.58,27.09,-71.76] [ 32.59,27.09, -2.92] [-11.03,27.09,-8.77] [-12.55,27.09,-31.65]
```

**Caveat:** the path zig-zags spatially, which suggests bots treat a way as a **point set**, not a
strict loop. Render as ordered polyline *and* dots, and label it "patrol area" rather than implying
a fixed circuit.

Only ~8% of `PatrolPoint`s are referenced by a way (Interchange 261/3088, Streets 317/3946, Customs
390/4213); the remainder are `SubPoint_N` children used as cover/look slots. **Ship only the
referenced points.**

Implementation findings (2026-07):

- **1-point ways are real data**: `Patrol_Killa_alarm1..6` (WithName, n=1) are alarm-response
  POSTS, not routes — keep them, render as a dot.
- **Read the trailing name with the STRICT printable reader** (`read_cstr_strict`), not the
  lenient one: `PatrolWayWithConditions` serializes condition state after the sentinel, and the
  lenient read shipped a `"\x12"` name on labs. With the strict read all 28 labs conditional
  ways decode cleanly via the same sentinel anchor.
- **Serialized route id ≠ GameObject name**: the six alarm posts all serialize
  `KILLA_PATROL_ALT` while their GO names stay distinct (`Patrol_Killa_alarm1..6`) — ship both
  (`name` + `go`).
- Per-map way counts: interchange 30, customs 49 (1 unzoned), woods 27, labs 34 (28
  conditional), ground_zero 3.

### 4.4 Zones

- **`BotZone`** has **no collider** (verified on all 12 Interchange zones). Derive a footprint as the
  convex hull of its spawn markers + patrol points. Payload layout (CONFIRMED on all 12
  interchange zones): `f32 1.0 @0 | i32 zone-id @4 | 0xFFFFFFFF @8 | i32 | i32`, then **@20 the
  PatrolWay PPtr array (`u32 N` + N×12 B) immediately followed by the SpawnPointMarker array**,
  then tuning floats. The extractor still LOCATES the arrays by walking
  (`locate_pptr_arrays`) so a per-map layout wobble degrades instead of shipping garbage.
- **Friendly zone names come from json.tarkov.dev's `maps_en`** (`Zone*`/`BotZone*` keys:
  `ZoneCenterBot` → "Center", `ZoneWoodCutter` → "Lumber Mill"). This is the table tarkov.dev's
  boss `spawnLocations` names render from, so the boss→zone join is an EXACT match on it —
  substring matching both missed ZoneWoodCutter and was ambiguous on interchange's
  ZoneCenter/ZoneCenterBot pair (only the bot zone carries a translation, which resolves it).
  Only boss-relevant zones are translated (interchange 5/12) — exactly the ones that need it.
- **`AIPlaceInfo`** (×106) *does* carry a BoxCollider — 100×12×100 "Home_zone" volumes with an
  integer id. Usable directly via the existing `footprint()` helper.
- **`AICorePoint`** records carry a `CG:` **connectivity group** id — the game's own reachability
  partition. Potentially valuable for validating `nav_bake.rs` island detection.

### 4.5 Other AI-scene data (CONFIRMED present, decode unverified)

`AIVoxelesData` (38,016 records / 2.2 MB) and `AICoversData` (12,256 records, position + facing
normal) are the game's own walkability and cover sets — the most valuable un-decoded thing left,
since nav is where this project has had the most bugs. `NavMeshDoorLink` ×219 is keyed by **the
same `door_…` ids already in `gamedata.json`** — free traversal edges for the nav graph, and
already extracted (§5).

**AIVoxelesData (2026-07 probe): `u32 count = 38016` at payload+0 is CONFIRMED; a fixed stride is
REFUTED — do not start from one.**

What holds up: the data really is **vertical columns**. The first records share an XZ and step Y
by exactly 5.0 m — `(-393.43, 15.71, -415.49) → (…, 20.71, …) → (…, 25.71, …)` — and within clean
runs the only Y deltas seen are 0.0 and 5.0. So the sample lattice is 5 m vertical.

What does NOT hold up: **stride 56**. It looks right on the first ~300 records and on a 2000-record
spot check (79% plausible), which is exactly the trap. Tested against the full 38,016 it gives
29,560/38,016 plausible and **breaks first at record #333**, after which clean runs come in
irregular lengths (333, 371, 386, 1, 391, …). A uniform stride cannot produce a run of length 1.
The dead giveaway that the alignment is drifting rather than the data being odd: 2,359 of the
in-column Y deltas are **0.0** — consecutive "samples" at an identical XZ *and* Y, i.e. the reader
re-reading the same bytes at a wrong offset.

So the record is **variable-length or block-structured** (4 + 38016×56 also leaves 88,796 trailing
bytes, whose head decodes as another plausible float3 — likely a second section, not padding).
Anyone picking this up should start by finding the block header at byte 18652 (record #333),
**not** by fitting a stride. Budget it as a real reverse-engineering task.

---

## 5. Other scenes worth opening — **all EXTRACTED 2026-07 (second audit)** except where noted

- **`*_Scripts.unity`** (Interchange level 53): `AirdropPoint` ×186 (position from Transform,
  payload empty → `airdrop_points` sink + viewer Airdrops layer), `IndoorTrigger` ×35
  (BoxCollider volumes → `indoor_volumes` sink; roof culling / floor picking later),
  `TOD_Sky` → **decoded**: `[5 ints][f Hour @20][i Day][i Month][i Year][f Lat][f Lon]`
  (interchange: 6.4 h, 1/8/2018, 46.0 N 84.0 E) → top-level `sun` — drive the viewer sun with
  the log's `hourOfDay`. `WeatherController` = 2.9 kB of config curves, skipped.
- **`*_Culling.unity`** (Interchange level 521): `LevelBorder` → **decoded**: `u32 N + N×float3`
  (37 verts at fixed Y) → top-level `level_border` (terrain-draped; the viewer draws it as an
  always-on dim boundary ring).

  **PerfectCulling PVS — container format DECODED (2026-07). Reader: `tools/pvs_probe.py`.**

  The bytes are not in the scene. `PerfectCullingAdaptiveGrid` is only a 48 B stub —
  `{u32 5, u32 2, u32 0, 32-char GUID}` — and the bake ships separately as
  `StreamingAssets/Culling_Data/<guid>_packed_cull.bytes`: **15 files, 4.6 GB**, one per
  location. The GUID in the stub matches the filename exactly, **15/15 with no unmatched
  files**, so the scene→file join needs no heuristic (`pvs_probe.py list`).

  ```
  u32   nScenes
  nScenes x { u32 16 ; byte[16] sceneGuid }
  <cell records, contiguous>
  u32   cellOffset[nCells]        # absolute byte offset of each cell
  u32   nCells                    # LAST dword in the file

  cell:   float3 centre ; float3 size ; float4 rotation ; u32 clen ; byte[clen] zlib
  payload (inflated):
          u8 nBlocks                                   # <= nScenes
          nBlocks x { u8 sceneIdx ; u16 nVisible ; u16 dataLen ; byte[dataLen] }
  ```

  Read the index table first (EOF−4 gives `nCells`, then back `4*nCells`) and **random-access**
  cells. Do not walk sequentially — it is slower and one bad block costs the rest of the file.

  Validated on **all 15 files** (`pvs_probe.py verify`): offsets strictly increasing,
  `offsets[0] == header_end`, the last cell ending exactly at the index table, every sampled
  cell's sub-block walk landing exactly on the payload end, every rotation a unit quaternion,
  every `sceneIdx` in range. Cell counts run 8,431 (sandbox_sl) to 300,044 (woods); interchange
  has 296,708 in 440 MB. Cells are genuinely adaptive — 3.0 m at the coarsest, subdividing to
  ~0.7 m. Decoded bounds match the maps (interchange x −434..649, z −462..448).

  **Producer quirk:** ~0.7% of cells in factory_rework day/night carry a zlib stream missing its
  final marker/Adler-32. Strict `zlib.decompress` throws "incomplete or truncated stream" on data
  that is fine; `decompressobj` returns it. The offset table proves our framing is right there
  (`next_offset − offset − 44 == clen` exactly), so it is their bug, not our misparse.

  **NOT decoded: the innermost `data`.** The vendor documents it as *"variable bit length
  encoding"*, which matches measurement — 3.8–5.5 bits per entry, and the size depends on the
  VALUES not just the count (two blocks with `nVisible=17` encode to 8 and 10 bytes). So it is an
  entropy/varint code over renderer indices; finishing it needs the library's bit reader, **not
  another stride guess**.

  **Second blocker, independent of the codec:** even fully decoded, the indices are positions in
  a scene's renderer list, and the pack does not preserve renderer identity through assembly —
  the same gap `EXTRACTABLES_AUDIT` flagged for door animation. Consuming the PVS means solving
  that first.

  Also decodable now: `CullingGridPreProcess` (112 kB) — `u32 count = 4679` at +12, then float3
  position + float3 extents per cell, stride 24, uniform 9.95 m cubes. That is the coarse query
  grid, not the adaptive one. Left unextracted deliberately: with all cells identical, the whole
  grid is described by {bounds, cell size, count}, and nothing consumes it.

  Why it matters: the viewer has no occlusion culling (Hi-Z was scoped and deferred), and this is
  the game's own baked solution for the exact geometry Atlas renders — it would cut frame time
  and resident VRAM together. The container is no longer the obstacle; the codec and the renderer
  join are.
- **`maps/*_preset.bundle`** → DONE (dominant-location-folder join; see §2).
- **Room semantics:** the `*_Sound.unity` mirror is extracted: `SpatialAudioRoom` ×369 →
  `rooms` (name + BoxCollider footprint), `SpatialAudioPortal` ×1101 → `room_portals` with the
  edge PARSED FROM THE GO NAME (`AudioPortal_FROM_<room>_TO_<room>` — no payload decode
  needed). File-only for now: room labels + portal-aware pathfinding are viewer follow-ups.
  (`BaseSpatialRoom`/`ServerSpatialPortal` live in geometry scenes and stay untapped — the
  Sound mirror is a superset.)
- **`NavMeshDoorLink`** ×219 (AI scene) → **decoded**: `u32 id + door_… id string + float3 A/B`
  → `door_links` sink — traversal edges keyed to gamedata's own door ids, for the nav graph.
- **`AICorePoint`** ×28 (AI scene) → `core_points` {id, cg, pos}: id/CG match the GO name
  (`AICore ID:14 CG:27`); interchange has 8 connectivity groups, main island 19 points —
  nav-island ground truth for `nav_bake.rs` validation.
- **`AIPlaceInfo`** ×106 → `ai_places` {string id @4, name (`Home_zone1`), BoxCollider
  footprint} — bot anchor volumes, file-only.

### Cultists (2026-07 hunt)

- **`CultistSignEffect`** — a TYPED component in the AI scenes (interchange ×92, woods ×27; GOs
  `HalloweenCultisSign` / `EventSectants`): the event ritual-sign spots. Extracted →
  `cultist_signs` sink + viewer "Cultist signs" layer. These GOs are the same placeholder
  quads §4.1 warns would pollute geometry — take the transforms, never the meshes.
- **Cultist SPAWNS are the boss system**, nothing separate: the static catalog's
  `sectantPriest` spawnLocations are internal zone ids we already extract with hulls —
  customs `ZoneScavBase`, woods `ZoneMiniHouse`/`ZoneBrokenVill`, shoreline
  `ZoneSanatorium1/2`/`ZoneForestSpawn`, night-factory `BotZone`, ground-zero-21
  `ZoneSandbox`. No Cursed/Sectant category bit exists in SpawnPointMarker masks (§4.2) —
  cultist placement is server-side bot typing over these zones.
- NOTE: the static dump's boss `spawnLocations.name` are INTERNAL zone ids ("ZoneScavBase"),
  while the old GraphQL loot.json carried friendly names ("Stronghold"). The viewer's
  two-pass boss join (exact `en`, then substring on the internal id) handles both vintages.

---

## 6. DEAD ENDS — verified absent, do not chase

- **No NavMeshData.** Every `NavMeshSettings` has a null data PPtr; zero NavMesh tiles; a byte scan
  of ~37 GB found no `NAVMESHSET`. Surfaces are built at runtime. (`NavMeshProjectSettings`,
  `NavMeshModifierVolume` ×176 and `NavMeshModifier`/`IgnoreFromBuild` ×5,798 **do** exist and are
  worth consuming.)
- **No localization / item templates in the client.** Item, key and exfil **display names must come
  from tarkov.dev** — nothing under `StreamingAssets` maps a template id to a name.
- **No spawn chance, escorts, or boss→zone assignment.** Server-side. Positions are ours; the
  probabilities are not.
- **No exfil timers or item requirements.** `ExfiltrationPoint`'s payload holds only the name string.
- **No time-of-day gating of spawns or patrols.** All 17 AI scenes searched: zero day/night fields,
  class names, or GameObject names, across 106 distinct AI MonoBehaviour classes. Day/night splits
  exist only for lighting/culling scenes, and those maps have exactly **one** AI scene — so
  spawns/patrols are TOD-invariant. TOD-dependent spawn logic is folklore w.r.t. the client assets.
- **No lightmaps or light probes shipped.**
- **Some window geometry is NOT in the scene files (streets, verified 2026-07).** Zmeiskiy 1's
  upper-floor courtyard windows: the pack was suspected of dropping them (see-through facade,
  un-raycastable openings, the cell tower visible through the building). Full audit chain —
  pack meshes == dataset meshes == **Unity level 304's own GameObjects** (only air conditioners
  + window fences exist in those openings; ground-floor `Window_plastic/wood_*` units ARE
  serialized, upper floors are NOT). All four City scenes the geometry pipeline skips were then
  probed directly: `City_Scripts` 211 (DryPlane rain quads; its `WindowBreakerManager` points
  only at `BrokenWindowPieceTemplate` shatter-VFX pieces), `City_AI` 212 (zero window-named GOs
  in 16k), `City_Grass` 394 (empty), `City_Quests` 395 (triggers). The scene-preset bundle's
  249 referenced scenes ALL resolve to BuildSettings levels — no streamed scene bundles exist.
  So the client necessarily instantiates those windows at RUNTIME from prefab bundles
  (`StreamingAssets/Windows/assets/content/…`), placements baked in the bundle-side prefabs,
  not in any scene. Any fix means a NEW extraction source (bundle prefab transforms diffed
  against the scene instance), not a pipeline change — the viewer already matches the client's
  scene data exactly. TRAP for future audits: `gen_maps --levels-for <folder>` lists GEOMETRY
  levels only; "every level extracted" claims must separately account for the service scenes.

---

## 7. Game logs

Logs live in `<install>\Logs\log_<date>_<version>\`. `viewer/src/game_watch.rs` already consumes:
`scene preset path:maps/<bundle>.bundle`, `"FieldOfView":`, `UserMatchOver`, `ChatMessageReceived`
types 10/11/12, and the player pose parsed out of **EFT screenshot filenames**.

Untapped, all CONFIRMED present:

| Source | Content | Use |
|---|---|---|
| `push-notifications_000.log` → `groupMatchRaidSettings` | `"side":"Pmc"\|"Savage"`, `location`, `timeVariant`, `hourOfDay`, weather | auto-select PMC vs Scav layers; set the viewer's sun |
| `application_000.log` | `MatchingCompleted` → `LocationLoaded` → `GameStarted:<t>` → `SessionEndUIScene` | raid T0 → own countdown; second map-id source |
| `output_000.log` | `Reason:Speed, Position:(…), CurrentState:Sprint` | free live position pings (~7/raid) without a screenshot |
| `groupMatchRaidReady` | per-member `Nickname`, `Side`, `Level` | squad display |
| `network-connection_000.log` | `rtt`, `Sid: US-DEN01G008` | server region + ping |

**Two traps when reading `side`:**

1. `groupMatchRaidSettings` is **not in every session** — it was present in May logs and absent from
   the six most recent. Any consumer must degrade gracefully (show all factions) rather than guess.
2. There is a *separate* capital-S `"Side": "Bear"/"Usec"/"Savage"` field with 11,000+ occurrences —
   those are **other players'** profiles in network messages. Only the lowercase
   `groupMatchRaidSettings.side` describes you.

**Absent from logs** (searched): raid countdown value, exfil availability/activation, survived/MIA
outcome, kill feed, loot pickups, XP, measured FPS. `backend_000.log` has empty `responseText` on
every request.

`Logging.config` pins several categories (`player`, `quests`, `exfiltration`, `spawn-system`) to
`Error`. Raising them would likely expose live exfil status — **but it edits a client file under
BattlEye. Not recommended.**

---

## 8. Method: decoding an unknown MonoBehaviour

1. Enumerate first. Load the level with UnityPy and histogram `obj.type` / the MonoBehaviour script
   class names before reading any bytes — it tells you what exists and how many.
2. Read the payload as raw bytes for several instances of the same class and diff them. Fields that
   vary are data; fields identical everywhere are usually constants or padding.
3. Anchor on strings. UnityPy string fields are length-prefixed and 4-byte aligned; they make
   reliable landmarks for locating adjacent numeric fields.
4. Cross-validate against something independent — a GameObject name, a Transform, a sibling
   collider. The `rad:` name token vs the SphereCollider disagreement was only caught this way.
5. Resolve PPtrs (12 bytes: file id + path id) to reach siblings; that is how zone names and radii
   are obtained.
6. Verify on **more than one map**. Naming schemes differ per map (see §4.2), and Ground Zero ships
   two AI scenes while Terminal's is duplicated in BuildSettings.

---

## 9. Known per-map gotchas

- **Ground Zero** has two AI scenes: `Sandbox_AI` (508) and `Sandbox_AI_high` (512, the level-25+
  variant). They are a REAL fork: 83 markers shared (byte-identical), 9 only in 508, 10 only in
  512 — union by Id, `lv` keeps the variant. **Trap:** the hand-curated ground_zero config lists
  512 as a *geometry* level, so its markers arrive without the `ai` flag — the flag is
  informational only; the viewer routes by side+mask, never by `ai`.
- **Labs** has `Laboratory_AI` (113) AND `Laboratory_dark_AI` (710, event variant) in the same
  folder — both scan, Id-dedupe unions them.
- **Terminal**'s `Terminal_AI` appears **twice** in BuildSettings (635 and 687 — same scene
  path). `ai_levels()` dedupes by path; spawn `Id` dedupe (now implemented) covers the rest.
- **Customs** has legacy markers named `p1 (N)` / `Sandbox_BP (N)` with `sides` 0 or 3 and empty
  masks — the viewer's side+mask router sends them nowhere (correct: they're authoring residue).
- Every playable location has an AI scene; none are missing. `Sandbox_SL_AI` (596) belongs to the
  tutorial folder `Sandbox_StartLocation` and is excluded by the folder scope automatically.

---

## 10. Suggested order of work

1. ~~`ScenesPreset` bundles → delete the hardcoded `bundle_to_map()` table.~~ **DONE 2026-07**
   (via dominant location folder, not ServerName — no editorial join needed; found the missing
   icebreaker bundle).
2. ~~AI-scene spawn markers → replaces the **online** PMC/Scav spawn layers with game truth.~~
   **DONE 2026-07** (`spawn_points` extended, `patrol_ways` + `bot_zones` sinks, viewer layers
   `BotZone`/`Patrol`, side+mask routing, boss snap on the `en` zone join).
3. `groupMatchRaidSettings` side/hour → auto-select faction layers, match the sun. (Remember
   both §7 traps: session-absent field, capital-S `Side` decoy.)
4. ~~Patrol ways + bot zones~~ — folded into 2.
5. ~~Room graph extraction~~ **DONE** (Sound-scene mirror → `rooms`/`room_portals`); the viewer
   follow-up (room labels, portal-aware pathfinding) remains.
6. ~~`LevelBorder`, `AirdropPoint`, `IndoorTrigger`~~ **DONE** (+ `TOD_Sky` sun, `door_links`,
   `core_points`, `ai_places`, `cultist_signs`).
7. ~~Scan-scope fix~~ **DONE 2026-07** (§2.1) — stage 6 now takes the derived level list, so the
   `*_DesignStuff` loot population is extracted. **Only interchange has been re-extracted**;
   reserve / streets / customs / shoreline / labs / lighthouse still carry stale gamedata and
   need a stage-6 re-run each.
8. ~~PerfectCulling PVS container~~ **DECODED 2026-07** (§5, `tools/pvs_probe.py`, all 15 files
   verified). Two things still stand between it and a working occlusion cull, in this order:
   **(a)** preserve renderer identity through pack assembly — needed by door animation too, so it
   pays for itself twice; **(b)** the variable-bit-length index codec. Do (a) first: without it a
   decoded codec has nothing to point at.
9. `AIVoxelesData` decode (§4.5 — a fixed stride is refuted; start at the record-#333 boundary);
   consume `core_points` CGs in nav_bake island validation; drive the viewer sun from `sun` ×
   log `hourOfDay`.

### 10.1 Extracted but NOT consumed by the viewer

These sinks ship in every pack and nothing reads them. That is deliberate ("file-only"), but it
means the data is unvalidated — a decode bug in any of them would be invisible today, so treat
their contents as unproven until something renders or asserts on them.

| sink | interchange | streets | intended use |
|---|---:|---:|---|
| `rooms` / `room_portals` | 369 / 1101 | 1224 / 3458 | room labels, portal-aware pathfinding |
| `door_links` | 219 | 723 | nav traversal edges (keyed to existing door ids) |
| `ai_places` | 106 | 220 | bot anchor volumes |
| `indoor_volumes` | 35 | 329 | roof culling / floor picking |
| `core_points` | 28 | 26 | nav-island ground truth |
| `sun` | 1 | 1 | drive the viewer sun with the log's `hourOfDay` |

Cost is ~192 kB of interchange's 898 kB `gamedata.json` and ~618 kB of streets' 1.9 MB — a third
of the file. Not enough to act on for size alone, but worth knowing before adding more sinks.
