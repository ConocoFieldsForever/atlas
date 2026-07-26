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
| Extracted datasets | `C:\Users\nhaum\beamng_blender_pipeline\eft_assets\<dataset>` |
| tarkmap dir | `C:\Users\nhaum\beamng_blender_pipeline\tarkmap` (holds `maps/` + `out/`) |
| Python with UnityPy | the repo venv: `.\venv\Scripts\python.exe` |

`EFT_ASSETS_ROOT` / `EFT_TARKMAP_ROOT` are **not set** in the shell by default — export them or pass
them explicitly. `tools/build_map.py` derives sane defaults if they are absent.

Read `ARCHITECTURE.md` for the `.eftpack` format and the extraction skill
(`tarkov-unity-extraction`) for the geometry/placement rules before touching the pipeline.

---

## 2. What is already extracted (don't redo)

- **Geometry / materials / terrain layers / lights / grass density / SH volume inputs** —
  `extraction/unity/eft_extract_v2.py`, consumed by `eft_pipeline/assemble_bevy.py`.
- **`gamedata.json`** — `extraction/intel/extract_gamedata.py`: exfils, typed doors (with `key_id`
  and `state`), minefields, sniper zones, quest triggers, loose-loot points, containers, **player
  spawn points** (177 on Interchange, from level 520), transit points, switches/interactables.
- **`NavMeshProjectSettings`** — the agent climb/slope/cellSize values now used by the nav baker.

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
extracted — it does not duplicate them.

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

### 4.4 Zones

- **`BotZone`** has **no collider** (verified on all 12 Interchange zones). Derive a footprint as the
  convex hull of its spawn markers + patrol points. Its payload holds two PPtr arrays (its
  PatrolWays, then its SpawnPointMarkers) plus tuning floats.
- **`AIPlaceInfo`** (×106) *does* carry a BoxCollider — 100×12×100 "Home_zone" volumes with an
  integer id. Usable directly via the existing `footprint()` helper.
- **`AICorePoint`** records carry a `CG:` **connectivity group** id — the game's own reachability
  partition. Potentially valuable for validating `nav_bake.rs` island detection.

### 4.5 Other AI-scene data (CONFIRMED present, decode unverified)

`AIVoxelesData` (38,016 records / 2.2 MB, xyz + column index) and `AICoversData` (12,256 records,
position + facing normal) are the game's own walkability and cover sets. The record stride is
**non-uniform** (2,217,692 B / 38,016 is not an integer), so decoding needs column-statistics work.
`NavMeshDoorLink` ×219 is keyed by **the same `door_…` ids already in `gamedata.json`** — free
traversal edges for the nav graph.

---

## 5. Other scenes worth opening

- **`*_Scripts.unity`** (Interchange level 53): `AirdropPoint` ×186 (position from Transform,
  payload empty), `IndoorTrigger` ×35 (indoor/outdoor volumes — useful for roof culling and floor
  picking), `TOD_Sky` (carries latitude/longitude/date → a real sun model), `WeatherController`.
- **`*_Culling.unity`** (Interchange level 521): `LevelBorder` — the real playable-area polygon
  (37 vertices at fixed Y). `PerfectCullingAdaptiveGrid` references
  `StreamingAssets/Culling_Data/<guid>_packed_cull.bytes` (461 MB across 15 files) — the game's own
  PVS. Format undocumented; **unverified**.
- **`maps/*_preset.bundle`** ×21 → `ScenesPreset.ServerName` (`Interchange`, `bigmap`,
  `factory4_day`, `RezervBase`, …). This is the derived replacement for the hardcoded
  `bundle_to_map()` table in `viewer/src/game_watch.rs` (~line 46), whose comment admits it was
  copied from TarkovMonitor. **Trivial win.**
- **Room semantics:** `BaseSpatialRoom` ×346 (names like `ServerRoom_LW_1st_KibaStore`),
  `ServerSpatialPortal` ×977, mirrored in the `*_Sound.unity` scene as `SpatialAudioRoom` ×369 /
  `SpatialAudioPortal` ×1101. A named room-and-doorway graph: room labels, indoor tests, and a
  coarse portal graph for pathfinding.

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
  variant).
- **Terminal**'s `Terminal_AI` appears **twice** in BuildSettings (635 and 687). Dedupe by spawn
  `Id` — the existing `dedupe()` keys on name+position and should be switched to `Id`.
- **Customs** has 5 legacy markers named `p1 (N)` with `sides` 0 or 3; fall back to the BotZone PPtr
  for their zone.
- Every playable location has an AI scene; none are missing.

---

## 10. Suggested order of work

1. `ScenesPreset.ServerName` → delete the hardcoded `bundle_to_map()` table. Trivial, pure principle.
2. AI-scene spawn markers → replaces the **online** PMC/Scav spawn layers with game truth.
3. `groupMatchRaidSettings` side/hour → auto-select faction layers, match the sun.
4. Patrol ways + bot zones → new layers nothing online offers.
5. Room graph (`BaseSpatialRoom` + portals) → room labels and portal-aware pathfinding.
6. `LevelBorder`, `AirdropPoint`, `IndoorTrigger` → three cheap layers.
7. `AIVoxelesData` / `AICorePoint CG:` → validate nav reachability (needs decode work).

Estimated effort for items 1–2 with schema and viewer changes: **~2.5–3 days.**
