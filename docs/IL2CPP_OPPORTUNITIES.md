# IL2CPP & adjacent-source opportunities (metadata-adjacent inventory)

Companion to `docs/IL2CPP.md`. That file established EFT ships an **encrypted**
`global-metadata.dat`, so no declared type/field tables. This file answers the follow-up:
*is there ANY unencrypted source that gives authoritative MonoBehaviour field names/offsets,
and what else adjacent to the metadata are we leaving on the table?*

Every claim below is backed by a command that was actually run on the live install.
Paths use `<EFT_INSTALL>` = `C:\Battlestate Games\Escape from Tarkov` and
`<REPO>` = this repository. `<DATA>` = `<EFT_INSTALL>\EscapeFromTarkov_Data`.
Game version at time of writing: **1.0.6.5 (build 46221)** — from
`(Get-Item '<EFT_INSTALL>\EscapeFromTarkov.exe').VersionInfo` → `ProductVersion 1.0.6.5-46221`.

---

## 1. Encryption status on the current version — STILL ENCRYPTED

The 1.0.6.5 update did **not** change the packaging. The metadata is still obfuscated.

```
<REPO>\venv\Scripts\python.exe tools/il2cpp_explore.py header
```
Output:
```
[il2cpp] file    : <DATA>\il2cpp_data\Metadata\global-metadata.dat (27,257,812 bytes)
[il2cpp] magic   : 0x63153607  (il2cpp sanity is 0xfab11baf)
[il2cpp] VERDICT : ENCRYPTED / not plain il2cpp metadata -- cannot parse.
```

First dword is still `0x63153607`, not il2cpp's `0xFAB11BAF`. Same verdict as the prior finding;
nothing to enumerate. `strings`/`types`/`fields` correctly decline; only `scan` runs and (per
the prior 52,058-noise-run result) yields nothing. **No change — move on.**

---

## 2. Adjacent-source inventory — can ANY unencrypted source give field names? NO.

Bottom line up front: **class NAMES are fully recoverable; field NAMES/offsets are not, from any
shipped source.** In IL2CPP the field-name strings live only in the encrypted metadata; they are
referenced by index from `GameAssembly.dll` and never appear as plaintext, and the level/bundle
serialized files ship with type trees stripped. Details:

### 2a. `GameAssembly.dll` — field names are NOT in the binary

103,161,168 bytes. A raw byte scan for the field names we decode empirically:
```
<REPO>\venv\Scripts\python.exe -c "d=open(r'<EFT_INSTALL>\GameAssembly.dll','rb').read();
 [print(n, d.count(n.encode())) for n in ['openAngle','KeyId','doorState','LampController','EDoorState','_keyId','KeycardDoor','StationaryWeapon','LootPoint','MineDirectional']]"
```
→ **every one returns 0** (ASCII and UTF-16). The handful of near-name hits that do exist are
C++ RTTI / diagnostic / network-packet strings, not serialization layout, e.g.:
- `DoorState` (1 hit) → inside `"...field 'Door' of type 'DoorStateChange': Reference type field must be..."` (an Odin serializer error-message template)
- `ExfiltrationPoint` (1 hit) → inside `"UpdateExfiltrationPointPacket"` (a netcode type name)
- `FieldOfView` (3 hits) → inside method signature `HorizontalToVerticalFieldOfView(Single,Single)`

None gives a byte offset or an ordered field list. GameAssembly.dll is a **dead end** for named
fields — consistent with IL2CPP putting all identifier strings in the encrypted metadata.

### 2b. Odin / Sirenix serialization — present in the game, but NOT on our components

`RuntimeInitializeOnLoads.json` and `ScriptingAssemblies.json` show EFT bundles Odin
(`Sirenix.Serialization.dll`, `Sirenix.OdinInspector.Attributes.dll`). Odin's node format
**does** embed field-name strings in the payload — which would be a goldmine *if* our
interactables used it. They don't:
```
<REPO>\venv\Scripts\python.exe -c "import UnityPy;
 env=UnityPy.load(r'<DATA>\level42');
 mb=[o for o in env.objects if o.type.name=='MonoBehaviour'];
 print(sum(b'SerializationData' in o.get_raw_data() or b'$type' in o.get_raw_data() for o in mb),'of',len(mb))"
```
→ **0 of 6656** MonoBehaviour payloads carry an Odin marker. Door/Exfil/Switch/LootPoint use
plain Unity binary serialization (which is exactly why fixed offsets are stable across dumps).
The readable tokens that do appear in payloads are field *values* (`Boss`, `Player`,
`server_0`, `woods_design_stuff_00071`) and float bytes that happen to be printable — never
field names. **Odin path closed** for the components we care about.

### 2c. Config files under `<DATA>` — no gameplay config

Only four text/JSON files exist at the data root; none carries gameplay layout:
| file | content |
|---|---|
| `RuntimeInitializeOnLoads.json` | startup init hooks (reveals Odin usage; no gameplay data) |
| `ScriptingAssemblies.json` | 295 assembly names (`Assembly-CSharp`, `Comfort`, `Sirenix.*`, `bsg.*`) |
| `boot.config`, `app.info` | engine boot flags / company+product only |
No `.txt`/`.bytes` gameplay config; loot/spawn/quest tables are **server-side**, not shipped
(consistent with `docs/GAME_DATA_SOURCES.md`).

### 2d. `StreamingAssets` — real data, but not field names

```
find "<DATA>\StreamingAssets" -maxdepth 2 -type f | sed 's/.*\.//' | sort | uniq -c
```
| dir | files | format | what it is | usable? |
|---|---|---|---|---|
| `Windows\assets\...` | **7,385** `.bundle` | Unity AssetBundle | all game content (weapons, chars, audio, **Time-of-Day sky/cloud textures**) | yes for meshes/textures; MB fields still stripped (see §3) |
| `Acoustics\**` | 754 `.xrageo` + 14 `.xramap` | Steam Audio geometry | per-room baked acoustic scene geometry (room/occlusion volumes) | opaque binary; would need format RE |
| `AudioBakeData` | 19 `.audiobakedata` | BSG audio bake | per-scene audio bake | opaque binary |
| `Culling_Data` | 15 `.bytes` (`*_packed_cull`) | occlusion/portal bake | one per map — portal/cell visibility | opaque binary |
| `Grass` | 7 `.pcl` | grass point clouds | (already used by `build_grass.py`) | yes |

### 2e. `globalgamemanagers.assets` — class names, in full, for everything

This is the one genuinely useful adjacent source and we already use it. It ships **8,220
MonoScript** objects with intact engine type trees, each giving `m_ClassName` + `m_NameSpace`
+ `m_AssemblyName`:
```
<REPO>\venv\Scripts\python.exe -c "import UnityPy;
 g=UnityPy.load(r'<DATA>\globalgamemanagers.assets');
 ms=[o for o in g.objects if o.type.name=='MonoScript'];
 print(len(ms),'MonoScripts')"
```
→ `8220 MonoScripts`. Every MonoBehaviour in a level resolves to its authoritative C# class
name through its `m_Script` PPtr into this file. That is the naming backbone the extractor
already rides — but it names the *class*, never the *fields*.

---

## 3. UnityPy type-tree verdict — NOT SHIPPED (levels or bundles)

Type trees are disabled game-wide, so UnityPy cannot hand us structured fields for any script.

```
<REPO>\venv\Scripts\python.exe -c "import UnityPy;
 env=UnityPy.load(r'<DATA>\level600');
 f=next(f for f in env.files.values() if hasattr(f,'objects'));
 print('enable_type_tree=',f.enable_type_tree);
 o=[o for o in env.objects if o.type.name=='MonoBehaviour'][0];
 print('MB typetree nodes=', len(o.serialized_type.nodes or []))"
```
→ `enable_type_tree= False` and `MB typetree nodes= 0`. Same result on content AssetBundles
sampled from `StreamingAssets\Windows` (`enable_type_tree` false, MB node count 0). The
serialized `SerializedType` records are stripped (`is_stripped_type`), so even the class name
is blank locally — it is recovered only via the external MonoScript (§2e).

### Empirically-decoded classes vs type-tree availability

For every class we decode in `extraction/intel/extract_gamedata.py`, the situation is uniform:
class name is authoritative, field layout is empirical, and **no type tree exists to confirm or
extend it**. There are no NEW fields exposed anywhere — the "extra" fields are simply the bytes
we already walk past.

| decoder / class | class name source | type tree? | field names? | how fields are got today |
|---|---|---|---|---|
| `Door` / `Trunk` / `KeycardDoor` / `SlidingDoor` / `ExfiltrationDoor` / `DoorSwitch` | MonoScript (ext) ✓ | ✗ | ✗ | offsets (KeyId, Id, state @IdEnd+92, angle @IdEnd+56, trigger block) |
| `ExfiltrationPoint` / `Scav` / `Shared` / `Secret` / `CarExtraction` | MonoScript ✓ | ✗ | ✗ | settings Name @48 |
| `StationaryWeapon` | MonoScript ✓ | ✗ | ✗ | float-block + 24-hex weapon id + aim arcs |
| `LootableContainer` | MonoScript ✓ | ✗ | ✗ | Id @44 + 24-hex template id |
| `LootPoint` | MonoScript ✓ | ✗ | ✗ | GUID + scanned template-id array |
| `CardReader` | MonoScript ✓ | ✗ | ✗ | 24-hex accepted-card id pairs |
| `SpawnPointMarker` | MonoScript ✓ | ✗ | ✗ | Id/Name/pos/sides/cats/infil walk |
| quest/buffer/damage triggers (`PlaceItemTrigger`, `QuestTrigger`, `FlameDamageTrigger`, …) | MonoScript ✓ | ✗ | ✗ | zone id @0 + BoxCollider |

**Verdict: no type tree is available for any script, in any shipped file. The empirical
decoders remain the only path, and there is no unencrypted source to name-confirm them.**
The only defence against a patch moving an offset stays what `docs/IL2CPP.md` already
prescribes: re-run `tools/audit_doors.py` and the extractor validations after each update.

---

## 4. Ranked improvement scan — what IS accessible and worth doing

Since named fields are unavailable, the leverage is in (a) making the class-name backbone do
more, and (b) mining the *values* we can already reach. Ranked by user value / effort:

### 1. Per-map component census to find un-decoded interactables — **HIGH value, LOW effort**
The MonoScript index (§2e) already resolves every MonoBehaviour class in a level. Emit a
per-map histogram of class → count and diff it against the classes the extractor decodes, so
new/renamed interactables surface immediately after a patch instead of being silently dropped.
- Source: `<DATA>\level*` + `globalgamemanagers.assets`; access: UnityPy MonoScript resolution (verified — the resolve loop in this report printed `DryPlane 1739, IndoorTrigger 64, WeatherController 1, …`).
- Effort: ~1 hr (a `--census` flag on `extract_gamedata.py`). Unlocks: a maintenance safety net + a to-do list of decodable components. This is the highest-ROI item because it is pure reuse of proven code.

### 2. Weather / lighting / time-of-day presets — **MEDIUM value, MEDIUM effort**
`level600` is a shared environment scene whose MonoBehaviours are `TOD_Sky`, `TOD_Time`,
`WeatherController`, `RainController`, `WindController`, `SnowFlakes`, `SceneLights` (verified —
class resolution above). These carry the map's sky/weather/time defaults.
- Source: environment level (e.g. `<DATA>\level600`) + Time-of-Day bundles in `StreamingAssets\Windows\...\time of day\`; access: UnityPy raw-payload decode, **empirical offsets** (same method as the light controller).
- Effort: 1–2 days (per-class column-stat recovery, defensive). Unlocks: authored ambient light colour / fog / rain-intensity presets to seed the viewer's lighting instead of guessing — complements the existing SH bake.

### 3. Card-reader / keycard-door requirement enrichment — **MEDIUM value, LOW effort**
`dec_card_reader` and keyed doors already recover 24-hex template ids; join them to tarkov.dev
(as loot points already do) to name the required keycard/keys in the overlay.
- Source: existing extractor output; access: existing tarkov.dev item query. Effort: a few hrs. Unlocks: "needs Red keycard" labels on doors/readers with no new format work.

### 4. Occlusion/portal cell data (`Culling_Data\*_packed_cull.bytes`) — **MEDIUM value, HIGH effort**
15 files, one per map — a baked visibility/portal graph. If reversed, it yields room/cell
adjacency (indoor vs outdoor, portal connectivity) useful for a floor/room overlay.
- Source: `<DATA>\StreamingAssets\Culling_Data`; access: **unknown binary format — needs RE** (likely Umbra/BSG custom). Effort: high (format reversing, uncertain payoff). Lower priority.

### 5. Acoustic room geometry (`Acoustics\*.xrageo`) — **LOW value, HIGH effort**
Steam Audio baked geometry per room could delimit interior volumes ("audio zones"), but it is
an opaque third-party binary and the same room information is more cheaply approximated from
`IndoorTrigger` MonoBehaviours (already present, class-resolved). Skip unless a room-volume
feature is specifically requested.

### Explicitly NOT accessible (don't spend effort)
- **Named door/interactable fields** — only in the encrypted metadata; no unencrypted mirror (§2).
- **Loot spawn tables / boss spawn configs / extract timers & requirements** — server-side; the
  client ships only mock `LootData`/`Test*` objects in `resources.assets` (`docs/GAME_DATA_SOURCES.md`).
  Get these from tarkov.dev/SPT, not from the client files.

---

## Summary

Encryption unchanged on 1.0.6.5. No unencrypted source (GameAssembly.dll, Odin payloads,
bundles, type trees, config JSON) yields MonoBehaviour field names or offsets — those live only
in the encrypted metadata. What *is* authoritative and already exploited is the **class name**
of every component (8,220 MonoScripts in `globalgamemanagers.assets`). The empirical decoders
stay the only route to fields. The best next steps are pure reuse of that backbone — a per-map
component census to catch patch drift and find un-decoded interactables (highest ROI), then
empirical weather/lighting-preset extraction and tarkov.dev keycard enrichment.
