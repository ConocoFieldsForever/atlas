## Contents

- [1. Scope and inputs](#1-scope-and-inputs)
- [2. Reading IL2CPP MonoBehaviours without typetrees](#2-reading-il2cpp-monobehaviours-without-typetrees)
- [3. Coordinate bridge, handedness, units](#3-coordinate-bridge-handedness-units)
- [4. Transforms, colliders, footprints](#4-transforms-colliders-footprints)
- [5. Per-class payload layouts](#5-per-class-payload-layouts)
- [6. Bit masks and enums](#6-bit-masks-and-enums)
- [7. The active/enabled-chain verdict](#7-the-activeenabled-chain-verdict)
- [8. Scene discovery: AI, service and sibling levels](#8-scene-discovery-ai-service-and-sibling-levels)
- [9. Bot zones, patrol ways, convex-hull synthesis](#9-bot-zones-patrol-ways-convex-hull-synthesis)
- [10. Loot: containers, groups, LootPoint pools](#10-loot-containers-groups-lootpoint-pools)
- [11. Terrain draping](#11-terrain-draping)
- [12. Cross-level dedupe](#12-cross-level-dedupe)
- [13. gamedata.json schema](#13-gamedatajson-schema)
- [14. External data sources and joins](#14-external-data-sources-and-joins)
- [15. Invariants and failure signatures](#15-invariants-and-failure-signatures)
- [16. Old patterns](#16-old-patterns)

---

## 1. Scope and inputs

The gameplay-intel extractor turns Unity serialized scene files (`<GameData>/levelN`, plus `globalgamemanagers` and `resources.assets`) into one `gamedata.json` per map. It is *typed*: every record comes from a named MonoBehaviour class, never from a GameObject-name heuristic.

Inputs:
- `<GameData>/levelN` - serialized scene files. `N` is the **build index** into `BuildSettings.scenes`.
- `<GameData>/globalgamemanagers` - holds the `BuildSettings` object whose `scenes` array maps build index to scene asset path.
- `<GameData>/resources.assets` - TextAssets: `TestBackendLocaleEn`, `TestBackendLocaleRu`, `TestItemTemplates`, and baked `/client/location` snapshots.
- `maps/<map>/config.json` - `source.levels` (geometry scenes), `source.unity_location` (the `Assets/Content/Locations/<folder>` name), optional `coordinates.global_matrix`.

Authorities: `extraction/intel/extract_gamedata.py` (scene decode), `extraction/intel/extract_client_intel.py` (resources.assets intel), `extraction/intel/tarkov_static.py` (json.tarkov.dev catalogs), `extraction/intel/build_loot.py` (the join into the loot model). The pack-side enrichment pass lives in `tools/build_map.py` (§13).

---

## 2. Reading IL2CPP MonoBehaviours without typetrees

The game is IL2CPP with an **encrypted `global-metadata.dat`**, so script typetrees cannot be generated. (That premise is asserted by the extractor's module docstring, `extract_gamedata.py:11-14`; it is **unverified** in this repo - no code path attempts metadata extraction, so nothing here proves or disproves it.) Consequence: a MonoBehaviour object's serialized bytes can only be parsed for the **engine-defined header**; everything after it is script fields that must be decoded by hand-recovered layout.

### 2.1 The 32-byte header

All values little-endian. Unity aligns to 4 bytes after a `bool` and after a string's payload.

```
off  size  field
  0     4  m_GameObject.m_FileID   int32
  4     8  m_GameObject.m_PathID   int64
 12     1  m_Enabled               uint8   (+3 bytes pad)
 16     4  m_Script.m_FileID       int32
 20     8  m_Script.m_PathID       int64
 28     4  m_Name length           int32
 32     L  m_Name                  utf8, no terminator
          pad to 4-byte boundary
```

Payload start (`extraction/intel/extract_gamedata.py:211`):

```
hsize = (12 + 4 + 12 + 4 + len(utf8(m_Name)) + 3) & ~3
payload = raw[hsize:]
```

`12 + 4 + 12 + 4 = 32`. **A component with an empty `m_Name` has `hsize == 32`; a named one is 32 + ceil4(len(name)).** Every field offset quoted in §5 is relative to `payload`, i.e. *after* the name.

### 2.2 PPtr

A `PPtr<T>` is exactly **12 bytes, unaligned**: `int32 m_FileID`, `int64 m_PathID` (`struct.unpack_from("<iq", …)`). `m_FileID == 0` means "object lives in this file"; `m_FileID == k > 0` means `externals[k-1]` of the serialized file, resolved by basename against `<GameData>/<basename>` (`extract_gamedata.py:687`). A PPtr with `m_PathID == 0` is null.

### 2.3 Length-prefixed strings

Two readers exist and they are **not interchangeable**.

Lenient (`extract_gamedata.py:197`) - used at *known* offsets:
```
ln = u32(buf[off:off+4])
reject if ln < 0 or ln > 4096 or off+4+ln > len(buf)
s   = utf8(buf[off+4 : off+4+ln])          # UnicodeDecodeError -> reject
end = (off + 4 + ln + 3) & ~3              # 4-align AFTER the bytes
```

Strict/printable (`extract_gamedata.py:448`) - mandatory for blind **walks**:
```
reject if ln <= 0 or ln > 256 or off+4+ln > len(buf)
reject unless every char c satisfies 31 < ord(c) < 127
```

`walk_strings(pl, off)` (`:467`) steps 4 bytes at a time, emitting `(offset, string)` for every strict hit of length ≥ 3 and jumping to the string's aligned end on a hit. `hex24_strings` (`:482`) filters that walk to strings of **exactly 24 lowercase hex characters** - the shape of every MongoDB-style item/container/category template id in this game.

### 2.4 Class identification

`MonoScript` is an *engine* type, so its typetree is intact. Build `{path_id: m_ClassName}` per file (`extract_gamedata.py:646`, `:675`), then a MonoBehaviour's class = `resolve(m_Script.m_FileID, m_Script.m_PathID)`. Only classes on the accept list (`extract_gamedata.py:975-987`) are decoded; everything else is skipped before any payload read.

---

## 3. Coordinate bridge, handedness, units

Unity world space is **left-handed, Y-up, metres**. Every position, collider corner and route vertex is mapped to viewer/pack space by the top-left 3×3 of the map config's `coordinates.global_matrix`, defaulting to `diag(-1, 1, 1)` (`extract_gamedata.py:89`, `:638`):

```
p_viewer = G3 @ p_unity          # default: (x, y, z) -> (-x, y, z)
```

Rounded to 2 decimals (centimetre precision) on output.

Third-party positions are flipped before any spatial join, but by **two different mechanisms**:
- Hardcoded literal `[-x, y, z]`: the tarkov.dev loose-loot join (`extract_gamedata.py:1618`, `[-p["x"], p["y"], p["z"]]`) and `build_loot.py:123-124`.
- Configured `G3`: the stationary-weapon join calls `bridge(w["pos"])` (`extract_gamedata.py:1731`).

Under the default matrix the two agree exactly. Under a non-default `coordinates.global_matrix` they diverge - the two hardcoded sites silently keep the X-flip while `:1731` follows the config. Any map that needs a custom matrix must fix the hardcoded sites first.

`diag(-1,1,1)` is a **mirror** - it reverses handedness (Unity LH → viewer RH) and therefore reverses polygon winding. Two places compensate explicitly:
- Collider footprints emit corners in reversed order after the flip (`extract_gamedata.py:925`).
- `LevelBorder` vertices are reversed after the flip (`:431`).

Units everywhere: **metres** for positions/extents/radii/distances, **degrees** for door open angles and stationary-weapon arcs (Unity world degrees), **seconds** for search/fight times, **minutes** for raid timers, **roubles** for values.

---

## 4. Transforms, colliders, footprints

**Local TRS** (`extract_gamedata.py:627`). Quaternion `(x,y,z,w)` → rotation matrix `R`, then `M[:3,:3] = R * s` (column-wise scale, equivalent to `R @ diag(s)`), `M[:3,3] = t`. **World matrix** = father-chain product `world(f) @ L`, memoized per Transform (`:749`). Never decompose; multiply the 4×4s.

**BoxCollider footprint** (`extract_gamedata.py:915`). Fields `m_Center` (c) and `m_Size` (s), both local. Colliders are frequently unit boxes scaled by the transform, so the TRS chain is mandatory.

```
hx = s.x/2 ; hz = s.z/2 ; y = c.y - s.y/2          # BOTTOM face
local corners = [(c.x-hx, y, c.z-hz), (c.x+hx, y, c.z-hz),
                 (c.x+hx, y, c.z+hz), (c.x-hx, y, c.z+hz)]
world_i = bridge( (M @ [corner_i, 1])[:3] )
outline = [world_0, world_3, world_2, world_1]      # order reversed for the mirror
```

`col_center` (`:927`) bridges `M @ [c,1]` - this is the marker `pos` for every zone that has a collider.

`poly_area_xz` (`:590`) is the shoelace `|Σ(x_i·z_{i+1} − x_{i+1}·z_i)| / 2`, used to pick the largest child collider.

`largest_child_box` (`:946`) exists because `MineDirectional` carries **no collider on its own GameObject**: its blast volume is on child GameObjects (`MON-50_MineTrigger` ×3 plus a small body collider). Take the child BoxCollider footprint of maximum XZ area; the mine `kind` is the child name split on `_MineTrigger`.

`door_parts` (`extract_gamedata.py:880`) walks the door leaf's Transform subtree to depth 6 and emits `[mesh_name, bridged_world_pos]` for every GameObject that has a MeshFilter **and** actually draws: `MeshRenderer.m_Enabled` truthy and `m_CastShadows != 3` (ShadowsOnly) - `:862`. This is the game's own grouping of panel + glass + inlays; a proximity guess leaves glass behind when the panel swings, and 3ds-max-default names (`Box001`) on non-rendering ballistic proxies false-match unrelated instances.

---

## 5. Per-class payload layouts

All offsets are into `payload` (§2.1). `str@k` means a length-prefixed 4-aligned string starting at k. Every decoder is **defensive**: a field that fails its range check becomes `null`, never garbage.

### 5.1 SpawnPointMarker - `extract_gamedata.py:293`

```
  0        str   Id            -> off
 off       str   Name          -> off  (reject if None)
 off+ 0   3f32   position (Unity world)
 off+12   4f32   rotation quaternion
 off+28    u32   Sides mask       (EPlayerSideMask, §6.1)
 off+32    u32   Categories mask  (§6.2)
 off+36    str   Infiltration  -> end
 end + 0  PPtr   BotZone          (12 B; i32 fid + i64 pid)
 end +12   f32   40.0 on AI scenes / 4.0 on player scenes
 end +16   i32   CorePointId
 end +20  PPtr   SphereCollider   (spawn radius)
 end +32   16 B  constant
```

Requires `off+36 <= len` for the mask block. The **only** whole-tail degrade conditions are `inf is None` or `end + 32 > len(pl)` (`:311`); when either holds, `zone`, `core` and `radius` are all `null`.

Inside the tail the fields are independent (`:310-318`):
- `core` is read **unconditionally** at `end+16` (`:313`).
- `bz_pid = p0` only when `f0 == 0 and p0` (`:315`); `sph_pid = p1` only when `f1 == 0 and p1` (`:317`).

So an external fid on the BotZone PPtr nulls `zone` alone and leaves `radius` and `core` intact, and vice versa - the two PPtrs are validated separately and neither affects the rest of the tail.

Position must satisfy `isfinite(v) and |v| < 1e5` on all three components. The `SphereCollider.m_Radius` is read raw (marker transforms carry unit scale) and accepted when `0 < r < 1e4`. The `BotZone` PPtr resolves to that MonoBehaviour's GameObject name (`ZoneTagilla`, `ZoneCenterBot`) - that string is the zone key used everywhere downstream.

### 5.2 PatrolWay / PatrolWayWithName / PatrolWayWithConditions - `:322`

```
  0   u32   type
  4   u32   N                      (reject unless 0 < N <= 4096 and 8+12N+8 <= len)
  8   N x 12B PPtr PatrolPoint     IN ROUTE ORDER (fid must be 0, pid non-zero)
 off  4B    0xFFFFFFFF sentinel    (reject if absent)
 off+4 f32  1.0
 off+8 str  route name             (STRICT reader; WithName only)
```

`PatrolPoint` payloads are all zero - a point's **position is its Transform's world translation**, resolved through its GameObject. One-point ways are real data (alarm-response posts), not errors.

**Unverified:** the claim that the trailing integer in each point's GameObject name matches its array index is a comment-only assertion (`:324-325`). No code enforces it, so a drift in serialization order would be silent.

### 5.3 BotZone - `:350`

BotZone has no serialized layout worth hardcoding and **no collider**. Instead `locate_pptr_arrays` walks the (<400 B) payload in 4-byte steps looking for any `[u32 N][N × 12B in-scene PPtr]` run with `0 < N <= 2000` whose pids **all** belong to a known target set. An array that fails validation is simply not found and the caller degrades.

**Unverified:** the comment-only claim (`:352-353`) that in practice the PatrolWay array sits at offset 20 immediately followed by the SpawnPointMarker array. Nothing depends on it - the walk is what runs, and it survives per-map layout drift.

### 5.4 Exfiltration points - `:220`

All four classes share: **48 fixed bytes, then `str@48` = `Settings.Name`**, which is a *locale key* (`"NW Exfil"`, `"E1"`, `"factory gate"`), not display text. Faction comes from the component **class**, not the payload (§6.4). The zone footprint is `cols[0]`, the first BoxCollider on the same GameObject.

`CarExtraction` is deliberately excluded: it derives from `ExfiltrationSubscriber` and only animates a car subscribed to a real `ExfiltrationPoint`; accepting it produces duplicate/stray markers.

### 5.5 Doors (WorldInteractiveObject) - `:226`

Two layouts, distinguished by whether `str@28` is a valid KeyId. A **KeyId is always `""` or exactly 24 lowercase hex characters** - a trigger name (`Open_01_722179887`) is neither, and that is the entire discriminator (`:260`).

Classic (every pre-trigger map):
```
  0   20 B  zeros
 20   u32   N = interaction-trigger count (== 0)
 24   u32   0x0F layer
 28   str   KeyId               -> kend
```
Trigger-block:
```
 20   u32   N                   (reject unless 0 < N <= 8)
 24   N x [ u32 kind (4=open, 2=close) , str trigger name ]
 off  u32   0x0F                (REQUIRED anchor; mismatch -> all None)
 off+4 str  KeyId               -> kend
```
Common tail:
```
 kend+12  str  Id ("door_…")    -> iend
 iend+56  f32  open angle       accept when finite and 0 < |a| <= 180
 iend+92  u32  EDoorState       lookup in §6.3; unknown -> None
```

Trigger names carry a trailing digit hash (`Open_01_722179887`). `links` = sorted unique trailing hashes with ≥ 6 digits; the Switch interactable that drives the door serializes the *same* hash, giving a zero-name-matching switch→door edge (consumed by the pack-side merge, §13).

### 5.6 StationaryWeapon - `:488`

```
  0   20 B   float block
 20   str    Name                -> nend
 nend u32    N_mounts            (accept 0..64)
 nend+4      N x 12B mount PPtrs
             + 3 x 12B fixed PPtrs        => p = nend + 4 + 12*N + 36
 p    7 x f32: [default yaw, 0, pitch_min, pitch_max, yaw_min, yaw_max, 0]
 ...  24-hex weapon template id (found by hex24 walk from nend)
```
Arc accepted only when all seven `|f| <= 720`, `f[2] < f[3]`, `f[4] < f[5]`, and `f[4]-1 <= f[0] <= f[5]+1`. Exactly one 24-hex id must be present or `weapon_id` is dropped.

### 5.7 LootableContainer / LootableContainersGroup / LootPoint / CardReader / others

| class | layout | anchor |
|---|---|---|
| `LootableContainer` | 44 B fixed, `str@44` = Id (`container_<zone>_00001`), then first 24-hex string = container **template** id | `:551` |
| `LootableContainersGroup` | `u32 len@0`, id bytes, pad to 4, `i32 min`, `i32 max` | `:515` |
| `LootPoint` | `u32 flags(=1)@0`, `str@4` = GUID Id, 28-byte fixed block, `u32 N`, N × 24-hex template ids, `u32` tail | `:601` |
| `CardReader` | id string, then PAIRS of (24-hex accepted-card id, event name), then fallback event | `:562` |
| `RaidDialogEntryPoint` | strict walk; localization key = first string containing `/`, dialog id = last string without `/` | `:575` |
| quest triggers | zone id is the **first** script field, `str@0` | `:584` |
| `WindowBreaker` | scene id, `str@0` | `:1076` |
| `AIPlaceInfo` | id, `str@4` | `:1220` |
| `AICorePoint` | `u32 id@0`, `u32 connectivity-group@4` (same values as the `AICore ID:14 CG:27` GameObject name) | `:1216` |
| `NavMeshDoorLink` | `u32 link id@0`, `str@4` = door id, 12 B zeros, `3f32 A`, `3f32 B` (then B repeated) | `:434` |
| `TOD_Sky` | 5 ints, `f32 Hour@20`, `3i32 day/month/year@24`, `2f32 lat/lon@36` | `:403` |
| `LevelBorder` | `u32 N@0`, N × float3 Unity verts | `:419` |
| `FlameDamageTrigger`, `AirdropPoint` | payload is 0 bytes - the Transform/BoxCollider **is** the data | `:194`, `:1192` |

`LootableContainersGroup` worked example (bytes shown from `raw[28:]`, i.e. the zero `m_Name` length dword first):
```
Goshan        : 00000000 06000000 "Goshan" pad 11000000 15000000 -> min 17, max 21
ClothingShops : 00000000 0d000000 "ClothingShops"  0b000000 0f000000 -> min 11, max 15
```
Accept only when `0 <= min <= max <= 4096`; otherwise ship the id with no odds.

`LootPoint` does **not** hardcode the array offset: it scans `range(guid_end, guid_end+64, 4)` for a `u32 N` with `1 <= N <= 64` followed by exactly N valid 24-hex strings. A fixed-block size change degrades to `(guid, [])`. **No weights are serialized** - the payload is the pool, not the distribution.

`TOD_Sky` acceptance: `0 <= hour < 24`, `1 <= day <= 31`, `1 <= month <= 12`, `2000 <= year <= 2100`, `|lat| <= 90`, `|lon| <= 180`.

`LevelBorder` acceptance: `3 <= N <= 4096`, `4 + 12N <= len`, all components finite with `|v| < 1e5`.

---

## 6. Bit masks and enums

### 6.1 SIDE_MASK - EPlayerSideMask (`extract_gamedata.py:168`)

Bit 0 = Usec, bit 1 = Bear, bit 2 = Savage (scav). The extractor ships a **string label** via a full-value lookup, and unknown values stringify the raw integer:

| value | label | meaning |
|---|---|---|
| 1 | `usec` | USEC only |
| 2 | `bear` | BEAR only |
| 3 | `pmc` | either PMC faction |
| 4 | `savage` | scav |
| 5 | `usec+savage` | |
| 6 | `bear+savage` | |
| 7 | `all` | |

### 6.2 CATEGORY mask - SpawnPointMarker Categories (`:172`)

| bit | token | status |
|---|---|---|
| 1 | `player` | confirmed |
| 2 | `bot` | confirmed |
| 4 | `boss` | confirmed |
| 8 | `bit8` | appears only on player-scene markers; unproven |
| 16 | `bit16` | unproven |
| 32 | `bit32` | unproven |
| 64 | `botpmc` | confirmed |

`cat_names(mask)` (`:175`) tests bits in the order `1,2,4,8,16,32,64` and emits `bitN` for unnamed bits. **The raw integer always ships as `categories_mask` alongside the token list** - a consumer must be able to re-derive the truth when a bit is later named.

### 6.3 DOOR_STATE - EDoorState (`:166`)

`0 none`, `1 locked`, `2 shut`, `4 open`, `8 interacting`, `16 breach`. Declared as flags but scenes serialize a single initial state; an unrecognised value yields `null`, never a guess.

### 6.4 Class → semantic tables

- Exfil faction from class (`:94`): `ExfiltrationPoint→pmc`, `ScavExfiltrationPoint→scav`, `SharedExfiltrationPoint→shared`, `SecretExfiltrationPoint→secret`.
- Door kind (`:160`): `Door/KeycardDoor/SlidingDoor/DoorSwitch→door`, `Trunk→trunk`, `ExfiltrationDoor→exfil_door`. **Swing set** (`:164`) = `{Door, KeycardDoor, DoorSwitch}` - only these get `swing:true`, `open_angle` and `parts`; trunks/sliding/exfil doors move differently and must not be rotated about the pivot.
- Quest-trigger kind (`:180`): `PlaceItemTrigger→place_item`, `ExperienceTrigger→visit`, `FlareShootDetectorZone→flare`, `QuestTrigger→quest`; `ShootableQuestLocationObject` is folded into the same array with `kind:"shoot"` and an empty outline.
- Buffer-zone kind (`:182`): `BufferGates/BufferGate→buffer_gate`, `BufferZone→buffer`, `IgnorePlayerInputZone→input_lock`, `LighthouseKeeperZone→lightkeeper`, `EventObjectInteractive→event_interactive`, `InteractiveObjectCutsceneTrigger→cutscene`, `GuardedZone→guarded`.
- Damage-zone kind (`:194`): `FlameDamageTrigger→flame`.

---

## 7. The active/enabled-chain verdict

`active` is a single boolean produced by `active_chain(transform_pid, go_pid) and bool(m_Enabled)` (`extract_gamedata.py:995`, definition at `:766`):

1. The owning GameObject's `m_IsActive` must be true.
2. **Every ancestor** GameObject via the Transform `m_Father` chain must also have `m_IsActive` true (recursive, memoized on `(transform_pid, go_pid)`).
3. The MonoBehaviour's own header `m_Enabled` must be non-zero.

Inactive content **still ships** - it is authored-but-off, useful for map history and event content - but the flag must be honoured: a spawn under a disabled parent is not a live raid start. Missing typetree data defaults to `True` (unreadable ≠ disabled). Dedupe (§12) prefers an active row over an inactive twin of the same key.

Service-scene arrays (`airdrop_points`, `indoor_volumes`, `door_links`, `core_points`, `ai_places`, `rooms`, `room_portals`) intentionally carry **no** `active` field.

---

## 8. Scene discovery: AI, service and sibling levels

The map config lists *geometry* levels only. Gameplay data lives in scenes the geometry list excludes on purpose (placeholder cubes, cultist-sign quads).

**Rule** (`extract_gamedata.py:1770`). Load `globalgamemanagers`, find the `BuildSettings` object, take `scenes` (or `m_Scenes`) - an array of asset paths where **the array index is the `levelN` number**. Then for each scene path `p`:

1. Normalize `\` → `/`.
2. Find the marker `Assets/Content/Locations/`; skip the path if absent. The first segment after it is the **location folder**.
3. Accept only when `folder.lower() == config.source.unity_location.lower()`.
4. `base = basename(p) without extension`; split on `_`; the token set must intersect the wanted token set, case-insensitively.
5. Reject indices already in the geometry `LEVELS`; dedupe by scene **path** (a scene can appear at two build indices - keep the first).

Token sets used, in scan order (`extract_gamedata.py:1841`, `:1851`):

| tokens | scene | what it contributes |
|---|---|---|
| `ai` | `*_AI` | SpawnPointMarker, PatrolWay*, BotZone, AICorePoint, AIPlaceInfo, NavMeshDoorLink |
| `scripts` | `*_Scripts` | AirdropPoint, IndoorTrigger, TOD_Sky |
| `culling`, `levelborders` | `*_Culling` | LevelBorder |
| `sound` | `*_Sound` | SpatialAudioRoom, SpatialAudioPortal |

AI scans set `ai=True`, which stamps `"ai": true` on the resulting spawn records. **Genuine variant scenes with different paths all scan** (`Sandbox_AI` + `Sandbox_AI_high`, `Laboratory_dark_AI`); the spawn-Id dedupe unions them and each record's `lv` keeps its origin.

**Sibling fallback** (`:1744`), used only when zero exfils were found after all of the above: candidates are every build index whose scene *dirname* equals the dirname of any already-scanned scene and which has not been scanned. This is how a logic scene like `Factory_DesignStuff` (level 68, holding the exfils, absent from the geometry levels 2/69/70/177) is discovered without a per-map constant.

---

## 9. Bot zones, patrol ways, convex-hull synthesis

BotZone components carry **no collider** - a zone's footprint has to be synthesized.

Per-scene pass (`extract_gamedata.py:1330`):
1. Decode every PatrolWay (§5.2); resolve each PatrolPoint PPtr to its GameObject's Transform world translation, bridge it. Drop ways that fail the decode or resolve to zero points.
2. Record `kind` = `patrol` / `named` / `conditional` by class.
3. When the serialized route name differs from the GameObject name, ship both (`name` = serialized, `go` = GameObject) - the serialized id repeats across alarm posts while the GameObject name stays distinct.
4. For each BotZone, run `locate_pptr_arrays` against the set of PatrolWay path_ids; every hit assigns `way_zone[way_pid] = botzone_gameobject_name`.
5. Register `{name, lv}` into an internal `_zones_reg` sink.

Global pass, **after** the spawn-Id dedupe (`:1893`) so a zone spanning two variant AI scenes gets the union of its members:
```
members  = spawn_points where zone == name
ways     = patrol_ways  where zone == name
pts      = [s.pos for s in members] + [p for w in ways for p in w.points]
skip the zone entirely if pts is empty
pos      = componentwise mean of pts, rounded 2dp   (3-component centroid, includes Y)
hull     = hull_xz(pts)
```

`hull_xz` (`:379`) is a monotone-chain convex hull on the XZ plane:
- Deduplicate to `sorted(set((round(x,2), round(z,2), round(y,2))))` - sorted lexicographically by x then z then y. Two points sharing XZ but differing in Y count as distinct.
- Cross product `cross(o,a,b) = (a.x-o.x)(b.z-o.z) − (a.z-o.z)(b.x-o.x)`; pop while `cross <= 0` (collinear points are removed).
- Lower chain over the sorted list, upper chain over the reversed list, concatenate dropping each chain's last element.
- Return `[]` when fewer than 3 distinct entries or the final ring has < 3 vertices - a one-marker zone has no footprint, only its `pos`.
- **Each hull vertex keeps its own Y** (member points already sit at ground height); the hull is not planar.

Patrol-way `points` are the game's own ordered route vertices and are deliberately **never** draped or subdivided - synthetic vertices would masquerade as authored ones. Bot-zone hulls *are* draped (§11).

---

## 10. Loot: containers, groups, LootPoint pools

### 10.1 Group membership is hierarchical

A `LootableContainersGroup` sits on a **parent GameObject**; its containers are descendants. Attribution walks the Transform `m_Father` chain from the container up to the nearest registered group Transform (`extract_gamedata.py:786`), memoized **per node**, not per leaf - the per-leaf memo is the quadratic pattern that costs hours on deep hierarchies. Guard depth 256.

Because a container can be scanned before its group, unresolved containers stash `_tpid` and are re-walked in a post-pass with the memo cleared first (`:1296`).

### 10.2 Group spawn probability

`min`/`max` are how many of the group's containers actually spawn in a raid. The member count is **recounted from the attribution**, never trusted from the payload, so it always matches the containers actually shipped:

```
members = count of containers with grp == gid
p       = round(min(1.0, ((min + max) / 2) / members), 4)      # omitted when members == 0
```
Every container then receives `grp_p = p` of its group. Observed ratios vary enormously by area (a 3-container arms shop at 2–3 of 3 → `grp_p` 0.8333, mall stashes at 18–21 of 104 → 0.1875) - this is the game's own per-location spawn odds, which a single type-average fill rate cannot express. The Goshan group of §5.7 lands at `(17+21)/2/42 = 0.4524`.

### 10.3 Container → renderer join key

Each container ships `tf`: up to 3 entries walking self → parent → grandparent, each a folded u32 of the int64 Transform path id (`:1117`):
```
tf_i = int((pathID ^ (pathID >> 32)) & 0xFFFFFFFF)
```
The assembler attaches prefab renderer instances by **ancestry intersection** against the same fold - `match_loot_models` in `viewer/src/render/gpu_driven.rs:5884-5967` reads gamedata's `tf` against each instance's `(par, par2, lv)` ancestry. **Unverified:** that the Rust side computes the byte-identical fold; only the shape of the match was corroborated. Name matching and radius matching both fail regardless: offset prefab pivots miss parts, and radius lights decorative same-mesh neighbours on shelves.

### 10.4 LootPoint (loose loot, first-party)

`LootPoint` MonoBehaviours are the **only** loose-loot positions the client ships - a small curated set (gun racks, gun safes, food piles, car trunks; some maps have none). The bulk of loose loot is server data.

`finalize_loose` (`:1642`):
1. Dedupe on the serialized GUID (scene variants re-serialize the same rack); an active row beats an inactive twin.
2. Merge points with the **same `name` within 0.5 m** into one map point: `n` = slot count, `templates` = union in first-seen order, `active` = OR, `pos` = first point's.
3. Resolve template ids (§14.2) and sort `items` by `(cat asc, -price)` so priced real items lead and category slots trail - the viewer titles the card off `items[0]`.

---

## 11. Terrain draping

Outline vertices sit at the collider's **bottom face**, which floats or sinks on undulating ground.

**Heightfield** (`extract_gamedata.py:1390`). Built from the map's `.eftpack` (`EFT_PACK_DIR`, else `packs/<map>.eftpack`). Pack space == viewer space, so no further bridging.
- Read `manifest.json` for `vertex.stride`, the `position` attribute `offset`, `instance.stride`, and the instance field offsets `flags`, `affine`, `meshId`.
- Keep only instances where `flags & 2` - `FLAG_TERRAIN = 1 << 1`.
- `affine` is 12 floats read as a 3×4 row-major matrix: `world = local @ A[:, :3].T + A[:, 3]`.
- Bin all world vertices to a **2.0 m** XZ grid by `floor(coord / 2.0)`; each cell stores the **mean Y** (terrain slices overlap at seams).
- `sample(x, z)`: bilinear over the 4 surrounding cell *centres* - `fx = x/2.0 - 0.5 - x0`, weights `(1-tx)(1-tz)`, `tx(1-tz)`, `(1-tx)tz`, `tx·tz`; renormalize by the summed weight of the cells that exist; return `None` when the total weight is `< 1e-6`.

**Drape** (`:1475`), applied to `exfils`, `transit_points`, `quest_triggers`, `trader_zones`, `buffer_zones`, `loot_groups`, `damage_zones`, plus bot-zone hulls and the `level_border` ring:
```
for each closed-outline edge a->b:
    k = max(1, ceil(hypot(b.x-a.x, b.z-a.z) / 4.0))     # ~4 m steps
    for j in 0..k-1:
        t  = j/k ; x,z linear in t ; y0 linear in t     # collider base Y interpolated
        ty = field.sample(x, z)
        y  = max(ty + 0.3, y0) if ty is not None else y0
```
Vertices off the grid keep the collider Y. Lines follow the ground and never sink below the collider.

**Elevated zones** - `minefields`, `sniper_zones`, `mines_directional` (`:1518`) - are **never** draped. Their trigger boxes are tall volumes whose bottom face can reach base terrain far below a raised platform (one observed minefield: collider centre Y = 15.65 on a train platform, footprint Y ≤ −0.41). Instead the whole outline is **flattened to the collider centre height**, which is exactly `pos.y`, with no subdivision.

Before either transform, `extent = [w, d]` is stamped from the **pre-subdivision** rectangle: `w = dist(p0,p1)`, `d = dist(p1,p2)`, kept only when both exceed 0.05 m. The document's top-level `draped` boolean tells the consumer whether to apply its own lift.

---

## 12. Cross-level dedupe

`dedupe(rows, keyf)` (`extract_gamedata.py:1377`) keeps one row per key; **an active row replaces an inactive one with the same key**, otherwise first wins.

| array | key |
|---|---|
| `exfils` | `(faction, name)` |
| `doors` | `id` or `(name, pos)` |
| `containers` | `id` or `(name, pos)` |
| `spawn_points` | serialized `id` (GUID) or `(name, pos)` |
| `patrol_ways` | `(name, zone, all points)` |
| `door_links` | `(door, a)` |
| `core_points` | `(id, cg, pos)` |
| `room_portals` | `(from, to, name, pos)` |
| everything else | `(name, pos)` |

Loose points dedupe on GUID inside `finalize_loose` instead (§10.4). Bot zones are rebuilt after dedupe, not deduped.

---

## 13. gamedata.json schema

Single JSON object, compact separators. Top level **as written by the extractor**:

```
map                str
generated_levels   [int]   every scene index actually scanned, in scan order
logic_levels       [int]   sorted distinct lv of the exfil records
draped             bool    a terrain heightfield was available
counts             {array name: int, exfils_by_faction: {faction:int},
                    doors_with_key: int, level_border?: int}
sun?               {hour, day, month, year, lat, lon}   from TOD_Sky
level_border?      [[x,y,z]]  longest ring across variant scenes, draped
<arrays…>
```

**The shipped file has one more key.** Every pack's `gamedata.json` also carries a top-level `switches` array, merged in place by `merge_gamedata_interactables` (`tools/build_map.py:515-563`), which additionally tags power-gated exfils and wires the switch→door edges from the trigger hashes of §5.5. A freshly extracted `gamedata.json` has none of the three (`build_map.py:517-519`), so a tool adopting a re-extraction into an already-built pack must re-run the merge - a raw copy silently drops the Level Controls data.

Switch record keys: `controlled_lamp_gos`, `controlled_light_gos`, `count`, `go_name`, `group`, `id`, `kind` (`power` | `switch` | `card_reader` | `dialog`), `label`, `level`, `path`, `switch_go`, `targets`, `trigger`, `world_pos`. Two conventions invert relative to every extractor array: the scene index is `level`, not `lv`, and `world_pos` is **raw Unity** - `build_map.py:550` undoes the X-flip when folding gamedata's own point interactables in.

Every record carries `lv` (source scene index). Every record carries `pos` `[x,y,z]` in viewer space **except**:
- `patrol_ways` - `{name, kind, points, lv, zone, go?}`; the route lives in `points` (`extract_gamedata.py:1348-1358`).
- `door_links` - `{door, a, b, lv}`; the geometry is the two endpoints (`:1211`).

`outline`/`hull` are `[[x,y,z]]` rings, CCW after the mirror, 4 vertices before draping. Fields marked `?` are omitted when absent. Arrays in the "new sinks" set (`containers`, `damage_zones`, `card_readers`, `dialogs`, `barbed_wire`, `windows`, `patrol_ways`, `bot_zones`, `airdrop_points`, `indoor_volumes`, `door_links`, `core_points`, `ai_places`, `cultist_signs`, `rooms`, `room_portals`) are **dropped entirely - data and count - when empty**, so a map without them keeps a byte-identical document.

| array | fields |
|---|---|
| `exfils` | `name` (locale key), `faction`, `pos`, `outline`, `go`, `active`, `lv`, `extent?`, `display_name_en?`, `display_name_ru?` |
| `minefields` | `pos`, `outline`, `name`, `active`, `lv`, `extent?` - one record per BoxCollider on the GameObject |
| `sniper_zones` | `pos`, `outline`, `name`, `active`, `lv`, `extent?` |
| `mines_directional` | `pos` (largest child box centre), `name`, `kind` (`MON-50`…), `outline`, `active`, `lv`, `extent?` |
| `doors` | `pos`, `key_id` (24-hex or null), `state`, `kind`, `id`, `name`, `active`, `lv`, `swing?`, `open_angle?` (deg), `parts?` `[[mesh,pos]]`, `links?` `[hash]` |
| `transit_points` | `pos`, `name`, `outline`, `active`, `lv`, `extent?` |
| `stationary` | `pos`, `name`, `active`, `lv`, `weapon_id?`, `yaw?`, `yaw_range?`, `pitch_range?`, `weapon_name?` |
| `spawn_points` | `pos`, `name`, `side`, `categories_mask`, `infiltration`, `active`, `lv`, `id?`, `categories?`, `zone?`, `radius?`, `core?`, `ai?` |
| `patrol_ways` | `name`, `kind`, `points` `[[x,y,z]]`, `zone`, `lv`, `go?` - **no `pos`** |
| `bot_zones` | `name`, `pos` (centroid), `hull`, `n_spawns`, `n_ways`, `lv`, `en?` |
| `quest_triggers` | `pos`, `name` (zone id), `kind`, `outline`, `active`, `lv`, `extent?` |
| `trader_zones` | `pos`, `name`, `outline`, `active`, `lv`, `extent?` |
| `buffer_zones` | `pos`, `name`, `kind`, `outline`, `active`, `lv`, `extent?` |
| `buffer_switches` | `pos`, `name`, `kind`, `active`, `lv` |
| `damage_zones` | `pos`, `name`, `kind`, `outline`, `active`, `lv`, `extent?` |
| `containers` | `pos`, `name`, `active`, `lv`, `id?`, `template?`, `tf?`, `grp?`, `grp_p?`, `tpl_name?` |
| `loot_groups` | container groups: `pos`, `name`, `active`, `lv`, `gid?`, `min?`, `max?`, `members?`, `p?`. `LootPointsGroup` rows instead carry `outline`/`extent?` |
| `loose_points` | `pos`, `name`, `n` (slots), `active`, `lv`, `items?` `[{tpl,n,s,pr,cat}]`, `items_src?`, `dev_d?` |
| `card_readers` | `pos`, `name`, `active`, `lv`, `items?` `[{id,n?}]` (or `item_ids` when unresolved) |
| `dialogs` | `pos`, `name`, `active`, `lv`, `id?`, `loc_key?` |
| `barbed_wire` | `pos`, `name`, `active`, `lv` |
| `windows` | `pos`, `name`, `active`, `lv`, `id?` |
| `cultist_signs` | `pos`, `name`, `active`, `lv` |
| `airdrop_points` | `pos`, `name`, `lv` |
| `indoor_volumes` | `pos`, `name`, `outline`, `lv` |
| `door_links` | `door` (matches `doors[].id`), `a`, `b`, `lv` - **no `pos`** |
| `core_points` | `pos`, `id`, `cg` (connectivity group = the game's reachability partition), `lv` |
| `ai_places` | `pos`, `id`, `name`, `outline`, `lv` |
| `rooms` | `pos`, `name`, `outline`, `lv` |
| `room_portals` | `pos`, `lv`, `from?`, `to?` (parsed from `AudioPortal_FROM_<room>_TO_<room>`), `name?` |

---

## 14. External data sources and joins

Everything in this section is **optional and degrades**. `EFT_GAMEDATA_OFFLINE` disables all network joins; raw ids ship instead.

### 14.1 First-party locale (resources.assets)

`game_locale_tables` (`extract_gamedata.py:103`) loads TextAssets `TestBackendLocaleEn` / `TestBackendLocaleRu`, parses `m_Script` as JSON, takes `root["data"]`, and builds `{key.casefold(): value}` for non-empty string values. `localize_exfils` (`:140`) looks up `exfil.name.casefold()` and attaches `display_name_en` / `display_name_ru` **without changing the serialized identity key**. Case folding is required because one scene serializes `factory gate` while the table holds `Factory gate`.

### 14.2 json.tarkov.dev static catalogs

`tarkov_static.py` fetches `https://json.tarkov.dev/regular/<name>` with ETag / Last-Modified revalidation into a disk cache; 304 keeps the cached document, a network failure falls back to the cached snapshot, and only a total miss raises. **GraphQL is not used.** Three schema quirks the adapter bridges (`tarkov_static.py:9-16`):
1. Each file is `{"data": {...}, "translations": [...]}`.
2. Nested objects are **id strings** resolved through bundled tables (`data.lootContainers`, `data.mobs`, `data.maps`, the items dump).
3. `data.*` name fields are the literal key `"<id> Name"`; the real English string lives in the separate `<name>_en` dump, a flat `{"<id> Name": "English"}` map. Numeric fields (prices, chances) are real.

Joins performed:

| target | source | rule |
|---|---|---|
| `containers[].tpl_name` | `load_static_containers` - `maps.data.lootContainers`, names via `maps_en`/`items_en` | exact template-id lookup |
| `loose_points[].items` | `load_static_item_index` - `items` + `itemCategories` | id lookup; `cat:1` marks a **category** template (a pool slot such as "Food and drink", no price) |
| `loose_points[].dev_d`, fallback `items` | `load_static_loose(display_name)` | nearest-neighbour in 3D after flipping dev positions with the hardcoded `[-x,y,z]`; distance always recorded; items copied only when `dist <= 2.5 m`, top 4 by `avg24hPrice`, `items_src:"tarkov.dev"` |
| `card_readers[].items[].n` | `load_static_items(ids=…)` | exact id lookup |
| `stationary[].weapon_name` | `load_static_stationary` | id lookup first; if unresolved, nearest map mount within **3.0 m** after `bridge()` |
| `bot_zones[].en` | `load_static_zone_names` - every `maps_en` key starting with `Zone` or `BotZone` | exact key match on the BotZone GameObject name (`ZoneCenterBot`→`Center`, `ZoneWoodCutter`→`Lumber Mill`) |

**Map selection is per-join, and the two paths do not agree.**

- `join_dev_loose` (`extract_gamedata.py:1607`) builds `DEV_NAME.get(MAP, MAP.replace("_", " ").title())` - the title-cased fallback exists **only here** - and passes it to `load_static_loose`, which picks a **single** map by exact `normalizedName == map_slug(name)` or exact case-insensitive EN display-name match (`tarkov_static.py:279-282`). **No prefix folding**: a variant that is not in `DEV_NAME` and does not match exactly returns zero rows.
- The stationary-weapon join (`extract_gamedata.py:1725`) uses `map_slug(DEV_NAME.get(MAP, MAP))` - the fallback is the **raw map id**, so an unlisted map slugs as `factory-rework`, not `Factory Rework`. Variant slugs (`ground-zero-21`) fold into their base map's row set by prefix **only** on this path (`tarkov_static.py:236-241`, accepting `slug.startswith(known + "-")` in either direction).

`map_slug` (`tarkov_static.py:169`): lowercase, runs of non-alphanumeric collapse to a single `-`, trimmed (`Streets of Tarkov` → `streets-of-tarkov`). `DEV_NAME` is at `extract_gamedata.py:1580`.

### 14.3 client_intel.json (resources.assets)

`extract_client_intel.py` reads every TextAsset in `resources.assets`. A location config is any TextAsset whose text starts with `{` and contains `"Location"` in the first 400 characters; the **richest** (longest raw) snapshot per `Location.Id` wins. `LOC_TO_ID` (`extract_client_intel.py:55`) maps game location ids to map ids (`bigmap→customs`, `factory4_day`/`factory4_night→factory_rework`, `Sandbox`/`Sandbox_high→ground_zero`, …); unmapped ids are **reported, never silently dropped**. When two ids map to one map, the record with more `exits + bosses` wins.

Emitted per location: 21 scalars from `LOC_SCALARS` (camel-cased with a lowercase first letter), plus `locationId`, `src`, `bosses[]` (`name` = the game's bot **role id**, `chance`, `zones` = `BossZone` split on `,`, `difficulty`, `escort`, `escortAmount`, `forced`) and `exits[]` (`name`, `chance`, `time`, `minTime`, `maxTime`, `type`, `requirement`, `requiredSlot`, `entryPoints`, `playersCount`).

Item templates come from `TestItemTemplates` → `["data"]`: `ITEM_PROPS` physical facts plus `id_name` from `_name`, plus `cells = Σ over Grids of (cellsH × cellsV)` (`:119`) - 0 means "not a container".

Trust boundary, per the extractor's own field-by-field comparison (`extract_client_intel.py:25-34`) - **the individual per-map numbers quoted there are unverified**, only the surrounding structure was checked: these are **staging snapshots** (`BackendUrl: stage-01`), but on raid timers the client is *newer* than the community catalog. Player counts are the one field to stay cautious about. **Rouble prices and per-item spawn weights are not present** and remain tarkov.dev's.

### 14.4 The boss-odds join

Three hops (`build_loot.py:226-254`, `tarkov_static.py:587`):
1. The client's `BossLocationSpawn.BossName` is a **bot role id** (`bossKojaniy`, `bossBully`, `bossBoar`, `sectantPriest`).
2. `load_static_mob_names` keys the `maps.data.mobs` catalog - whose ids *are* those role ids - through `maps_en`, giving `bossKojaniy → Shturman`, `bossBully → Reshala`, `bossBoar → Kaban`. **A prefix strip cannot do this job.**
3. That display name, lowercased and stripped, matches the positioned tarkov.dev boss node's name.

`chance_game = round(BossChance / 100.0, 3)` - the client stores a **percentage**, the node stores a 0..1 fraction. Where several client rows share a boss (one map ships ten Rogue rows at different chances/zones), the **highest** chance wins, which is the probability the boss appears anywhere on the map. The client's `BossChance` is the **base, event-blind** rate; event state is server-driven, so during an event upstream may read 100 % while the client still reads 30 % - both correct, about different things. The client gives zone **names** but no positions, so the positioned nodes stay and the client value attaches as `chance_game` / `zones_game`.

`zones_game` strings are the same internal keys as first-party BotZone GameObject names, so they are *expected* to join `gamedata.bot_zones[].name` exactly, and `bot_zones[].en` joins the other direction to tarkov.dev `spawnLocations[].name` (`tarkov_static.py:299-307`). **Unverified:** the code paths exist and are shaped for it and shipped packs do carry `en` on `bot_zones`, but the exactness of the string join has not been tested.

Two more client-intel folds in `build_loot.py`: `globalLootChanceModifier` multiplies every container `ev` (a map at 0.17 versus another at 0.9 is a ~5× difference the value model could not otherwise express) while `spawn` is left alone because per-area odds already ride on `gamedata.containers[].grp_p`; and search time becomes capacity-derived, `t = 2.5 + 0.26 × cells` (`build_loot.py:172-177`), with the cell count reached by `tpl_name → template id` (learned from shipped packs) `→ client_intel.items[tid].cells`.

---

## 15. Invariants and failure signatures

| # | Invariant | Failure signature |
|---|---|---|
| 1 | Payload offset is `(32 + len(utf8(m_Name)) + 3) & ~3`, not a constant 32 or 28 | Every field shifts by `4 + len(name)`. Decoders return `None` but records are still emitted, so arrays ship **full of null-field rows** rather than being empty. Reading probe offsets (`raw[28:]`, id length at 4) straight into a decoder made every loot group decode to nothing while still emitting a record. |
| 2 | Blind walks must use the **strict printable** string reader | Arbitrary binary bytes < 0x80 are valid UTF-8, so a garbage dword masquerades as a long string and carries the walk past the real field. Observed: a `PatrolWayWithName` name decoding as `"\x12"`. |
| 3 | Door `KeyId` is `""` or exactly 24 lowercase hex | On trigger-block doors the fixed read returns the **trigger name as `key_id`** and loses `state` and `open_angle` - doors that never open in the viewer. |
| 4 | The `0x0F` layer dword must land where the trigger walk predicts | Misaligned trigger block; all five door fields degrade to `None` rather than shipping a wrong key. |
| 5 | Every PPtr in a decoded **array** has `m_FileID == 0`; standalone per-field PPtrs are validated **independently** | An external fid inside an array means the walk found a false array - the array is not found and the caller degrades. On a per-field PPtr only that field nulls: an external BotZone fid on a SpawnPointMarker drops `zone` while `core` and `radius` still ship (`extract_gamedata.py:310-318`). |
| 6 | Positions satisfy `isfinite(v) and abs(v) < 1e5` | Misaligned float reads produce ±1e38 spawn markers that blow up the map bounding box. |
| 7 | Door state must be a known `EDoorState` value | Unknown → `None`; without the check, arbitrary payload dwords label doors with invented states. |
| 8 | Door open angle finite and `0 < abs(a) <= 180` | Doors swing by absurd amounts, or a zero-angle "open" pose is asserted. |
| 9 | Stationary arc: all seven `abs(f) <= 720`, `pitch_min < pitch_max`, `yaw_min < yaw_max`, `yaw_min-1 <= yaw <= yaw_max+1` | Inverted or nonsense firing arcs drawn on the map. |
| 10 | Loot group `0 <= min <= max <= 4096` | Group ships with no odds instead of a negative or absurd spawn count. |
| 11 | Group member count is recounted from attribution, never trusted from payload | `grp_p` exceeds 1.0 or under-reports; the probability stops matching the containers actually shipped. |
| 12 | Group attribution is memoized **per node**, not per leaf | Quadratic ancestor walks on deep hierarchies - minutes-to-hours extraction times. |
| 13 | The group post-pass clears the memo before re-walking | Containers scanned before their group keep a cached "no group" verdict and lose `grp`/`grp_p` permanently. |
| 14 | Footprint corners are reversed after the mirror; LevelBorder verts likewise | Rings wind CW in the mirrored space; backface-cull, fill rules and any signed-area test invert. |
| 15 | Colliders are resolved through the full parent TRS chain | Unit-box colliders (very common) render as 1 m squares at the wrong place. |
| 16 | `MineDirectional` blast zone is the largest **child** BoxCollider | The mine GameObject has none; a self-only lookup yields no outline at all. |
| 17 | Elevated zones keep collider-centre Y; only ground-hugging zones drape | A tall trigger volume on a raised platform snaps its whole ring to base terrain, tens of metres below where the mines actually are. |
| 18 | Bot-zone hulls are built **after** the spawn-Id dedupe | A zone spanning two variant AI scenes gets a first-scene-wins subset instead of the union - wrong `n_spawns`, truncated hull. |
| 19 | Hull needs ≥ 3 distinct XZ-rounded points and non-collinearity | Returns `[]`; a one-marker zone is a point, not a degenerate polygon. |
| 20 | Service scenes are scanned **in addition to** the geometry levels, never merged into them | Geometry configs exclude them on purpose (placeholder cubes, sign quads); merging pulls junk meshes into the pack. |
| 21 | Duplicate BuildSettings rows collapse by scene **path**; genuine variants all scan | Double-scanning a repeated scene inflates counts; skipping variants loses a large share of Ground Zero's markers (the source comments claim ~17 %; **unverified**). |
| 22 | The `active` verdict includes the whole ancestor chain and `m_Enabled` | Authored-but-disabled content is presented as live raid data. |
| 23 | Third-party positions are flipped before any spatial join - hardcoded at `extract_gamedata.py:1618` and `build_loot.py:124`, via configured `G3` at `:1731` | Unbridged: every nearest-neighbour distance is wrong by `2·|x|` and the 2.5 m / 3.0 m thresholds never match. With a non-default `global_matrix`: the hardcoded sites and the `bridge()` site land in different spaces. |
| 24 | `categories_mask` ships raw alongside the token list | Unnamed bits (8/16/32) become unrecoverable once tokenized. |
| 25 | Boss chance from the client is a percentage; node chance is a fraction | 100× overstated boss odds. |
| 26 | A re-extracted `gamedata.json` is re-run through `merge_gamedata_interactables` before it replaces a pack's copy | The pack loses `switches`, power tags and switch→door links - the Level Controls panel goes empty with no error. |

---

## 16. Old patterns

- **Name-classifying GameObjects.** The predecessor heuristic classified extracts by GameObject name; the module docstring (`extract_gamedata.py:5`) puts its false-positive rate at 71 % (**unverified** - quoted from that comment, not re-measured). The typed-MonoBehaviour approach replaced it: faction comes from the component **class**, never from a string.
- **Renaming exfils by proximity.** An earlier viewer renamed each exfil to the nearest community-catalog extract within 60 m, which silently attached wrong names. Superseded by the first-party locale tables (§14.1), which keep the serialized key and add display names beside it.
- **GraphQL (`api.tarkov.dev`).** Replaced throughout by the pre-generated `json.tarkov.dev/regular/*` catalogs, which is what tarkov.dev's own applications consume. The GraphQL endpoint returns 503 routinely; no code path should reintroduce it.
- **`boss_clusters[i % len]`.** Bosses were once scattered across spawn clusters by index modulo. Replaced by matching `bosses.spawnLocations[].name` against a boss-category spawn's `zoneName`, with the largest unused cluster as fallback and no marker invented when there is no geometry (`build_loot.py:375-414`).
- **Fixed-offset door reads.** Correct only for the classic layout; kept as the first branch because it is validated on every older map, but now guarded by the KeyId shape test and followed by the trigger-block path.
- **Hand-picked per-type search times.** Superseded by capacity-derived time (`2.5 + 0.26 × cells`) fitted to the old table's range, so budgets stay comparable but are monotonic in real grid capacity.
- **Packs predating the spawn `active` flag.** Older shipped `gamedata.json` files carry no `active` key on `spawn_points`. That is pack vintage, not a schema difference: a consumer validating §13 against them sees a mismatch the current extractor does not produce.