## Contents

- [Scope and vocabulary](#scope-and-vocabulary)
- [Environment contract](#environment-contract)
- [Coordinate, unit and handedness conventions](#coordinate-unit-and-handedness-conventions)
- [Build stages 1-9](#build-stages-1-9)
- [Cache keys and forced invalidation](#cache-keys-and-forced-invalidation)
- [.eftpack directory layout](#eftpack-directory-layout)
- [manifest.json schema](#manifestjson-schema)
- [meshes.bin](#meshesbin)
- [instances.bin](#instancesbin)
- [colliders.bin and collider_meshes.bin](#collidersbin-and-collider_meshesbin)
- [grass.bin](#grassbin)
- [nav.bin and siblings](#navbin-and-siblings)
- [volume.bin](#volumebin)
- [Self-contained / hardlink shipping mode](#self-contained--hardlink-shipping-mode)
- [Pack tiers and the file probe](#pack-tiers-and-the-file-probe)
- [Invariants and failure signatures](#invariants-and-failure-signatures)
- [Old patterns](#old-patterns)

---

## Scope and vocabulary

Three artifacts, three owners:

| Term | Meaning | Producer |
|---|---|---|
| DATASET | `<EFT_ASSETS_ROOT>/<dataset>/` - `scene.json`, `meshes/*.obj`, `tex/*.png`, `terrain_layers/`, `lights_*.json`, `colliders.json`, `decals.json`, `interact_*.json`, `lodmode.json` | the Unity extractors |
| BUILD TREE | `<EFT_TARKMAP_ROOT>/out/<map id>/` - `gamedata.json`, `volume*.bin`, `eft_grade_lut.bin`, `loot.json`, `tasks.json` | intel + bake steps |
| PACK | `packs/<map id>.eftpack/` - the shipping unit, self-describing | `assemble_bevy.py` + post-assemble bakes |

`dataset` and `map id` are DIFFERENT keys and must not be conflated: the dataset directory name is `basename(config.source.root)` (`tools/build_map.py:171`), e.g. map id `interchange` → dataset `interchange_v2`. The pack name, the `out/` directory and every intel join key use the MAP ID (`tools/build_map.py:673`); the dataset name is only the geometry input.

**Figure provenance.** The measured anecdotes in this document - misplaced colliders, mis-masked probes, dropped clumps, bytes and seconds saved - are quoted from the source comments that record them. They are historical measurements of a specific failure, not invariants, and they are not re-measured per build. Trust the mechanism; treat the number as an illustration of scale.

---

## Environment contract

Read once at import (`tools/build_map.py:54`):

- **`EFT_TARKMAP_ROOT` (TK)** - the directory that ITSELF contains `maps/` and `out/`, not the parent workspace. Unset → `normpath(<EFT_ASSETS_ROOT>/../tarkmap)`; if that is also unset, a legacy dev-machine absolute path (dead on any other machine). Wrong TK is not fatal but silently guts stage 6: `extract_gamedata` exits 1 and the pack ships without doors/exfil/zones.
- **`EFT_ASSETS_ROOT` (ASSETS)** - the datasets dir. Unset → `normpath(<TK>/../eft_assets)` (`tools/build_map.py:59`). The canonical layout is `workspace/tarkmap` + `workspace/eft_assets` as siblings; `extraction/check_env.py:138` warns when they are not.
- **`EFT_GAME_DATA`** - the game's `EscapeFromTarkov_Data` directory ITSELF (the one holding `globalgamemanagers`, `level0`, `level1`, `sharedassets*.assets`), not the install root. Never derived from TK/ASSETS; every consumer falls back to the vendor default install path independently (`tools/stamp_fingerprint.py:20`, `extraction/intel/extract_gamedata.py:62`, `extraction/grade/make_grade_lut_game.py:51`). `extraction/check_env.py:89` reads the variable and `:95` validates it by probing for `globalgamemanagers`.
- **`EFT_PY_UNITY` / `EFT_PY_BAKE`** - interpreters for the UnityPy and CUDA-warp stages. Resolution order: env var > a legacy anaconda path if that file exists > `sys.executable` (`tools/build_map.py:64`).
- **`EFT_ATLAS_EXE`** - the built viewer binary hosting `bake-sh` / `bake-nav` / `bake-terrain`. Probe order: env > `<repo>/target/release` > `target/debug` > repo root > `dist/` > `tools/` (`tools/build_map.py:625`). Missing → stages 3 and 8 are SKIPPED, not failed.
- **`EFT_BAKE=warp`** - switches the SH bake to the pre-assemble CUDA baker; anything else uses the portable post-assemble Rust baker (`tools/build_map.py:921`).
- **`EFT_FORCE_REBUILD=1`** (== `--force`), **`EFT_ALLLOD=0`** (== `--lod0`), **`EFT_BAKE_CPU=1`** (CPU SH retry), **`EFT_ASM_VEC=0`** (legacy `np.unique` path in the assembler, `eft_pipeline/assemble_bevy.py:455`), **`EFT_SEA_LEVEL`** (runtime override of `manifest.seaLevel`).
- **`EFT_JOBS=1`** (serial extraction) - UNVERIFIED. `tools/build_map.py` never reads it; the name appears only in a comment at `:788`. The presumed consumer is `extraction/unity/extract_parallel.py`, which was not read.

Every child is spawned with `PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8` plus `EFT_TARKMAP_ROOT`/`EFT_ASSETS_ROOT` set via `setdefault` (`tools/build_map.py:145`), stdout+stderr merged and streamed line-by-line, and on Windows `CREATE_NO_WINDOW` (`tools/procflags.py:26`) so a detached background build cannot steal focus.

---

## Coordinate, unit and handedness conventions

Units are METERS throughout (Unity native); no scaling anywhere in the pipeline.

The Unity→viewer transform is a constant handedness fix `G = diag(-1, 1, 1, 1)` for every EFT map, overridable per map via `coordinates.global_matrix` (`eft_pipeline/tarkmap_core/config.py:121`). Instance matrices are CONJUGATED, never composed:

```
M_world = G4 · M_unity · G4⁻¹        (instmath.py:21)
```

Conjugation is what keeps `det > 0` instances positive-determinant and preserves the local-space parameterisation of the mesh. `apply_global` returns a row-major 16; the pack takes `[:12]` - the row-major 3x4. **Never use the column-major glTF transpose here** (`eft_pipeline/assemble_bevy.py:70`).

Point/vector data that is NOT a matrix takes the bare 3x3 instead:
- lights, switch `world_pos`: `p' = (-x, y, z)` (`viewer/src/eftpack.rs:686`, `:964`); spot forward = `G3 · (q · +Z)` (`viewer/src/eftpack.rs:696`).
- collider `m_Center`: `ctr' = G3 · ctr` (`eft_pipeline/assemble_bevy.py:1238`). Mesh-collider verts get this free because UnityPy's `mesh.export()` already X-negates; primitives do NOT, and omitting it mirrors each primitive about its own pivot (source comment: 2,704 misplaced colliders, up to 4.02 m).
- LODGroup centers: `G3 · center` (`eft_pipeline/assemble_bevy.py:1263`).
- `gamedata.json` `pos` on exfils/doors is ALREADY viewer-space (bridged by the extractor) - flipping it again is a real bug the loader documents at `viewer/src/eftpack.rs:979`.

Collider primitives only survive a global matrix that is a **signed permutation**; `eft_pipeline/assemble_bevy.py:1154` asserts `|G3|` is doubly stochastic with unit entries and refuses to emit otherwise.

---

## Build stages 1-9

`total = 9` (`tools/build_map.py:681`). Markers are `[STAGE i/9] <name>` … `[STAGE i/9] <name>: done (Ns)` plus a machine-readable `[TIMING] <name>=<sec>` line per stage (`tools/build_map.py:160`). A stage marked `optional=True` prints `FAILED rc=… - optional, continuing` and returns False; a non-optional failure prints `[BUILD FAILED]` and `sys.exit(rc)` (`tools/build_map.py:162`).

The `1b` / `2a` / `2b` / `4a` labels below are a DOCUMENTATION convention for sub-steps that share a stage counter. The build emits integers only: stage 1b prints `[STAGE 1/9] extract nav agent settings (global)` (`:838`), 2a prints `[STAGE 2/9] scan interactables` (`:860`), 4a prints `[STAGE 4/9] extract projected decals` (`:958`). A log parser keying on `1b`/`4a` finds nothing.

The extractor scripts themselves (`extract_parallel.py`, `eft_extract_grass.py`, `eft_extract_colliders.py`, `eft_extract_nav.py`, `extract_interact.py`, `eft_extract_lights.py`, `make_grade_lut.py`, `fetch_icons.py`) are not read here; their stated outputs are taken from the invoking command lines and cache gates in `build_map.py`.

| # | Name | Runs | Produces | Cache gate | Fatal? |
|---|---|---|---|---|---|
| 1 | check dataset | `extraction/unity/extract_parallel.py --levels <derived> --name <dataset> [--alllod]` | dataset `scene.json`, `meshes/`, `tex/` | `<dataset>/scene.json` exists | **FATAL** (exit 3 if no levels, exit 3 if extraction leaves no `scene.json`) |
| 1 | extract grass density | `eft_extract_grass.py` | `terrain_layers/grass_density_*.bin`, `grass_protos_*.bin`, `grass.json` | same gate as above | optional |
| 1 | extract physics colliders | `eft_extract_colliders.py` | `<dataset>/colliders.json` | same gate | optional, but a missing `colliders.json` triggers a loud `[BUILD WARNING]` (`tools/build_map.py:818`) |
| 1b | extract nav agent settings (global) | `eft_extract_nav.py --out packs/shared` | `packs/shared/nav_agents.json` = `{agents, areas, layers}` from Unity `NavMeshProjectSettings` | **none** - re-run every build (engine-global, one `globalgamemanagers` read) | optional |
| 2a | scan interactables | `extract_interact.py --levels <missing>` | `<dataset>/interact_<lv>.json` | per-level file presence | optional |
| 2 | extract lights (level N) | `eft_extract_lights.py --level N` | `<dataset>/lights_<lv>.json` | per-level file presence | optional; empty level list → sky-only bake |
| 2b | extract grade LUT | `make_grade_lut_game.py` → fallback `make_grade_lut.py` | `<TK>/out/eft_grade_lut.bin` | file presence | optional (two-tier fallback so a build never fails for want of a grade) |
| 3 | bake lighting | **warp mode**: `bake_volume2.py <map>` pre-assemble → `out/volume2.*`, copied to `volume.*`. **default**: `atlas bake-sh <pack> --indirect-only` POST-assemble | `volume.bin`, `volume.json`, `volume_valid.bin` in the pack | `out/volume2.bin` (warp) / `<pack>/volume.bin` mtime ≥ stage start (portable) | optional; on a stale volume it retries once with `EFT_BAKE_CPU=1` (`tools/build_map.py:1000`) |
| 4a | extract projected decals | `extraction/intel/extract_decals.py <map> --dataset --levels` | `<dataset>/decals.json` | `decals.json` presence | optional |
| 4 | assemble pack | `python -m eft_pipeline.assemble_bevy <map> [--self-contained] [--keep-lods]` | the pack (see layout) | **none** - always runs | **FATAL** |
| 5 | grass | `eft_extract_grass.py` (only if no grids) then `python -m eft_pipeline.build_grass --pack <pack>` | `<pack>/grass.bin`, `grass_sidecar.json`, `grass_<Tex>.png` | any `terrain_layers/grass_density_*.bin` | optional; no grids → stage skipped entirely (indoor maps have no hardcoded list) |
| 6 | gameplay zones | `extract_gamedata.py <map> --levels=<derived>` then in-process `merge_gamedata_interactables`, then copy `out/gamedata.json` → pack | `<pack>/gamedata.json` | **none** | optional + mtime freshness warning |
| 7 | item icons | `fetch_icons.py <map>` (network) | `packs/shared/icons/<slug>.png`, `packs/shared/task_images/` | **none at the call site** (`tools/build_map.py:1114` passes no gate); per-file caching is presumed to live inside `fetch_icons.py` - UNVERIFIED, script not read | optional |
| 8 | bake nav grid (CPU) | `atlas bake-nav <pack>` | `nav.bin`, `nav.json`, `nav_door.bin`, `nav_blk.bin`, `nav_wallcell.bin` | `<pack>/nav.bin` mtime ≥ stage start | optional; skipped with a message when no atlas exe |
| 9 | stamp fingerprint | `tools/stamp_fingerprint.py <pack>` | `manifest.sourceFingerprint`, `manifest.sourceStampedAt` | **none** | **FATAL** |

**Level list derivation.** `dataset_levels` (`tools/build_map.py:201`) shells `gen_maps.py --levels-for <unity_location>` to read the LIVE BuildSettings, then UNIONS with `config.source.levels` - never fewer than the config. It returns a COMMA-JOINED STRING; iterating that string character-by-character is a real historical bug (`"5,2,,,5,4"`) guarded against at `tools/build_map.py:851` and `:952`. Light levels come from `extraction/maps/manifest.json` `light_levels` (a LIST, so multi-scene maps get full lighting) with a BuildSettings fallback (`tools/build_map.py:112`); switch-bearing levels are unioned in (`tools/build_map.py:873`) so default-off, switch-controlled banks are extracted too.

**Post-assemble, pre-stamp fixups** (order matters, all in `main`):
1. SH bake + freshness check (`:978`–`:1021`).
2. `seaLevel` patch into `manifest.json` (`:1026`) - see `derive_sea_level` below.
3. grass, gamedata, icons, nav.
4. `finalize_pack_manifest(pack, dataset)` (`:361`) - reconciles the sidecar table against files that actually exist. The pack's OWN `volume.bin`/`volume.json`/`volume_valid.bin` ALWAYS win over any outside reference (`:412`–`:418`, unconditional), compared on the WHOLE value (a basename test keeps a stale absolute path pointing at a different bake). `lightsAll` becomes the union of manifest-listed + dataset `lights_*.json` + pack-local `lights_*.json`, keyed by basename, pack-local superseding. A pack assembled by a DIRECT `eft_pipeline.assemble_bevy` run never gets this pass: the shipped interchange pack carries `volume.bin`/`volume.json`/`volume_valid.bin` while `sidecars.volume`, `volumeMeta` and `volumeVis` are all null and `sourceFingerprint`/`sourceStampedAt` are null - exactly the state step 5 warns about.
5. `verify_pack_lighting(pack)` (`:450`) - warn-only coherence audit: `len(volume_valid.bin) == dims[0]*dims[1]*dims[2]`, `volume.json.direct is false`, sidecars pointing at the pack's own files.
6. Stage 9 stamp, then a DETACHED `tools/dedup_textures.py` (`:1192`) that hardlinks byte-identical `tex/`+`terrain_layers/` files across datasets. It runs AFTER `[BUILD OK]` because sha1-hashing GBs kept the UI on "BUILDING" long after the pack was usable.

**`derive_sea_level(dataset)`** (`tools/build_map.py:228`) - game-truth ocean height, no authored constants. Over instances with a water-role, non-decal submesh (`role == "water"` AND (`sh` is None OR `"water" in sh.lower() and "decal" not in sh.lower()`)): transform the mesh's local OBJ AABB corners by the raw Unity matrix, reject anything whose world-Y span > 2.0 m (cascades), bin surfaces by `round(y*10)` (0.1 m), accumulate a per-bin XZ footprint. A bin qualifies as SEA iff its area ≥ 0.10 × the scene translation-AABB area **and** it touches the scene AABB on at least one side within `EDGE_FRAC = 0.02` of that span. Returns `round(y + 0.05, 3)`. Containment, not size, is the discriminator: woods' lake clears any area bar and produced a bogus 7.454 m sea.

**Fingerprint** (`tools/stamp_fingerprint.py:24`): FNV-1a 64 (offset `0xCBF29CE484222325`, prime `0x100000001B3`) over `"<name>|<size>|<mtime_seconds>;"` UTF-8 bytes for every top-level file in `EFT_GAME_DATA` whose name starts with `level` or ends `.assets`/`.resS`/`.resource`, sorted by name; formatted `{h:016x}`. Stat-only. `viewer/src/menu.rs` reimplements this digit-for-digit; any divergence makes every pack read as stale.

---

## Cache keys and forced invalidation

`--force` / `EFT_FORCE_REBUILD=1` deletes only the CACHE GATES (`tools/build_map.py:697`): `<dataset>/scene.json`, `out/volume2.bin`, `out/volume.bin`, `<pack>/nav.bin`, `out/instanced_raw.glb`, and every `<dataset>/lights_*.json`. It never touches the big mesh/texture exports or the existing pack, so a failed re-extract leaves the old pack playable until stage 4 swaps.

`<dataset>/lodmode.json` = `{"alllod": bool}` records the mode the dataset was extracted in (`tools/build_map.py:794`). An absent file means a pre-marker dataset = LOD0-only. A mismatch against the requested mode WARNS and downgrades the request rather than shipping a pack silently missing shells (`tools/build_map.py:770`).

---

## .eftpack directory layout

Every file a pack can carry, and who writes it:

| File | Written by | Contents |
|---|---|---|
| `manifest.json` | assemble stage 4, patched by stages 3/9 and `finalize_pack_manifest` | the self-describing layout table (below) |
| `meshes.bin` | assemble | all interleaved vertices, then all u32 indices |
| `instances.bin` | assemble | fixed-stride instance records |
| `materials.json` | assemble | JSON ARRAY of material records, index == `id` |
| `colliders.bin` | assemble (if `colliders.json` present) | fixed-stride physics colliders |
| `collider_meshes.bin` | assemble | positions f32x3, then u32 indices - POSITIONS ONLY, never rendered |
| `lod_integrity.json` | assemble; the record is first populated in the LOD-dedup block whenever `n_dead_mesh > 0` (`eft_pipeline/assemble_bevy.py:894`), and the drew-nothing block (`:1084`) only creates it if still None, else `update()`s it; the file is emitted whenever the record is non-None (`:1379`) | `{map, dataset, probedMeshes, deadMeshes[], keptDeadFinerShell, keptNotCovered, deadMeshNote, shellsKeptByFallback, drewNothing*}` - a re-extraction work list, NOT referenced by the manifest |
| `volume.bin` / `volume.json` / `volume_valid.bin` | `atlas bake-sh` or promoted from the warp bake | SH irradiance volume triple |
| `nav.bin` / `nav.json` / `nav_door.bin` / `nav_blk.bin` / `nav_wallcell.bin` | `atlas bake-nav` | layered 2.5-D nav grid |
| `grass.bin` / `grass_sidecar.json` / `grass_<Tex>.png` | `build_grass` | procedural grass field + one billboard card per kind |
| `gamedata.json` | stage 6 copy | exfils, doors, zones, loot points, merged `switches` |
| `lights_<lv>.json` | dataset copy (self-contained) or referenced in place | raw Unity realtime lights |
| `terrain_layers/` | self-contained ship_dir | ctrl/layer PNGs, density bins, `grass.json`, `manifest.json` |
| `tex/` | self-contained `ship_tex` | flat, basename-keyed texture copies |
| `particles.json`, `tex_fx/`, `semantics.json` | other tools; MIGRATED across the atomic swap | - |

`packs/shared/` is the map-agnostic tier, resolved pack-local → shared → cwd: `loot.json`, `tasks.json`, `grade_lut.bin` (copied by assemble when newer, `eft_pipeline/assemble_bevy.py:1390`), `nav_agents.json` (stage 1b), `icons/`, `task_images/`, plus other global catalogs.

**Atomic emission** (`eft_pipeline/assemble_bevy.py:682`): assemble writes into `<pack>.building`, then at the end moves every file the staging dir LACKS out of the live pack (sidecar migration), renames the live pack to `<pack>.old`, renames staging into place, and deletes `.old` (`:1414`). The migration is unfiltered, which is exactly why stages 3/6/8 each verify FRESHNESS by mtime instead of file presence - a carried-across `nav.bin`/`gamedata.json`/`volume.bin` from the previous build is otherwise indistinguishable from a fresh one.

---

## manifest.json schema

Written with `indent=1, allow_nan=False`. Non-finite floats (the game ships e.g. a LODGroup `fadeTransitionWidth = NaN`) are sanitized to 0.0 and each path is reported (`eft_pipeline/assemble_bevy.py:1360`).

```
version        u32 == 1                 (loader rejects anything else: eftpack.rs:1124)
dataset        dataset DIRECTORY basename
datasetPath    absolute - provenance ONLY, never resolved through
map            canonical map id - the STABLE intel join key
bounds         [minX,minY,minZ,maxX,maxY,maxZ] world AABB from transformed local corners
vertex         { stride: 36, attrs: [{name, fmt, offset}] }
instance       { stride: 80, fields: [{name, fmt, offset, note?}], align16: true }
meshes         [{ id, name, vtxOffset, vtxCount, idxOffset, idxCount, submeshes:[{materialId, idxStart, idxCount}] }]
instanceCount  u32   - MUST equal len(instances.bin)/stride (eftpack.rs:1144)
materialCount  u32
roots          [str] - root GameObject names; instance.rootId indexes this; slot 0 is ""
lodGroups      [{ size, center[3], srh[], ftw[], fadeMode, lastIsBillboard, cullH?, n }]
flagsLegend    {"0x1": "MIRROR (det<0: flip front-face/winding)",
                "0x2": "TERRAIN (MicroSplat splat shader)",
                "0x4": "BAKED_WORLD (identity affine, geometry pre-baked)",
                "0x8": "INACTIVE (Unity-disabled scenery/rooms; viewer hides unless 'show disabled geometry' is on)"}
                                        (assemble_bevy.py:1324-1327)
conventions    { uvVFlipBaked, uvOrigin, uvTilingBaked, uvXformNote, normalMapGreenFlip, normalMapConvention,
                 colorSpace{albedo,normal,emissive}, textureImport, affine, normals }   - 10 keys (:1332)
sidecars       { terrainLayers, lights, lightsAll[], volume, volumeMeta, volumeVis, grassJson, semantics }
seaLevel       f32 | absent   (patched by build_map after assemble)
collider       { stride: 96, fields:[...], flagsLegend{...} }   - absent when no colliders.json;
                 the collider flagsLegend (:1345-1348) has the same full-sentence shape
colliderCount  u32
colliderMeshes [{ id, name, vtxOffset, vtxCount, idxOffset, idxCount }]
layerNames     { "<layer index as string>": "<Unity layer name>" }
selfContained  true | absent
sourceFingerprint / sourceStampedAt   (stage 9)
```

`flagsLegend` values are full descriptive SENTENCES, not bare tokens. Match on the KEY (`"0x1"` … `"0x8"`) or on the leading word; a consumer string-comparing against `"MIRROR"` or `"TERRAIN"` misses every entry.

Of the ten `conventions` keys, THREE are load-bearing and seven are documentation. The consumer struct deserializes only `uvVFlipBaked` → `uv_v_flip_baked`, `uvTilingBaked` → `uv_tiling_baked` and `normalMapGreenFlip` → `normal_map_green_flip` (`viewer/src/eftpack.rs:218-231`), each defaulting to `true`, so an absent block means "already baked". Both UV flags are true in shipped packs, so `materials.json.uvXform` is REFERENCE ONLY - the manifest says as much in `uvXformNote` ("materials.json.uvXform is REFERENCE ONLY; tiling already baked into vertex UV") - and re-applying either in a shader double-applies it.

The **instance layout descriptor** is the anti-drift mechanism. The consumer resolves every field BY NAME (`viewer/src/eftpack.rs:1741`), validates `offset + size ≤ stride` per field before the record loop (`:1766`), and treats `par`/`par2`/`lv` as optional lanes that default to 0 on packs that predate them. `fmt_byte_size` (`:1910`) parses `<base>[x<count>]` with per-component sizes {f64:8; f32/u32/i32/unorm32/snorm32:4; f16/u16/i16/unorm16/snorm16:2; u8/i8/unorm8/snorm8:1} and returns None for an unknown base so `validate` rejects rather than guesses.

`materials.json` records carry: `id, role ∈ {opaque,cutout,glass,decal,water}, albedo, normal, uvXform[4], alphaMode ∈ {OPAQUE,MASK,BLEND}, alphaCutoff, tint[4] (sRGB→linear rgb, linear a), metallic, roughness, normalScale, normalGreenFlip, doubleSided, emissive{texture,factor[3],hdr}|null, roughnessFromAlbedoAlpha, specMap, vp{layers[],heights,blend,softCutout?}, detail{albedo,albedoUv[4],albedoStrength,albedoMeanGain[3],normal,normalUv[4],normalScale}, parallax{map,scale}, glassTRS+reflectColor/specColor/shininess/opacityScale/reflectCube`. `emissive` is an OBJECT or null - modelling it as a string aborts the whole parse (`viewer/src/eftpack.rs:350`). `albedoMeanGain` is `mean(linear(sample) × 4.5948)` per channel, computed offline on a ≤256 px thumbnail (`eft_pipeline/assemble_bevy.py:413`); the shader DIVIDES by it so the Unity-Standard ×2 detail blend is mean-neutral.

---

## meshes.bin

Two concatenated sections, no header: `[all interleaved vertices][all u32 indices]`. `idxOffset` is patched to `len(vertex_section) + local_index_offset` after the vertex section is final (`eft_pipeline/assemble_bevy.py:1121`; the section length `vlen` is taken at `:1119`).

Vertex record, **stride 36 B, little-endian** (`eft_pipeline/assemble_bevy.py:88`, asserted):

| offset | fmt | field |
|---|---|---|
| 0 | f32x3 | position - MESH-LOCAL |
| 12 | f32x3 | normal - MESH-LOCAL smooth normal |
| 24 | f32x2 | uv - tiling AND V-flip already baked |
| 32 | unorm8x4 | colour - vert-paint weights; `255,255,255,255` on non-vp submeshes |

Indices are `<u4`, LOCAL to the mesh (`submesh.idxStart` is an offset within the mesh's own index run, not a global one).

Per-submesh construction (`eft_pipeline/assemble_bevy.py:988`): take faces `F[f0 : f0+n]`; `f0 += n` MUST advance for skipped submeshes too, or every later submesh reads a face range shifted earlier by `n` and the last one loses its final `n` faces. UVs: `uv = uv_raw * (sx, sy) + (ox, oy)` from `_MainTex_ST`, then `uv.y = 1 - uv.y` (Unity bottom-left → PNG/wgpu top-left), in that order. Vertex dedup keys on `concat(round(pos,3), round(uv,3))` rows; smooth normals accumulate un-normalized face normals `cross(p1-p0, p2-p0)` into the deduped bins, then normalize.

Bounding spheres are derived by the consumer, not shipped: center = mean of positions, radius = max distance from center, both in mesh-local space (`viewer/src/eftpack.rs:1681`).

---

## instances.bin

Headerless array of fixed records. **Stride 80 B** (multiple of 16 so a WGSL storage read maps to 3×vec4 + 2×vec4 with no straddling), little-endian (`eft_pipeline/assemble_bevy.py:102`, asserted):

| offset | size | fmt | field |
|---|---|---|---|
| 0 | 48 | f32x12 | `affine` - ROW-MAJOR world 3x4, FULL 3x3 including shear and mirror |
| 48 | 4 | u32 | `meshId` - index into `manifest.meshes` |
| 52 | 4 | i32 | `lodGroup` - scene `lod.g`, or −1 |
| 56 | 4 | i32 | `lodIndex` - scene `lod.i`, or −1 |
| 60 | 4 | u32 | `rootId` - index into `manifest.roots` |
| 64 | 4 | u32 | `flags` |
| 68 | 4 | u32 | `par` - folded parent Transform id, 0 = none |
| 72 | 4 | u32 | `par2` - folded grandparent Transform id |
| 76 | 4 | u32 | `lv` - source scene level (folded ids are LEVEL-LOCAL, so joins must match on it) |

Affine reconstruction, **without decomposition** (`viewer/src/eftpack.rs:542`): rows are `r0 = a[0..4]`, `r1 = a[4..8]`, `r2 = a[8..12]`; linear column *i* = `(a[i], a[4+i], a[8+i])`; translation = `(a[3], a[7], a[11])`. Normals transform by the COFACTOR / inverse-transpose of that 3x3 - this is what makes shear and det<0 correct without baking.

`_fold32(x) = (x ^ (x >> 32)) & 0xFFFFFFFF` on the signed 64-bit Unity `path_id`, 0 stays 0 (`eft_pipeline/assemble_bevy.py:118`). The gamedata side folds identically, which makes `(lv, par, par2)` the authoritative prefab-ancestry join key for loot-glow - replacing name/radius guessing.

Flag bits (`eft_pipeline/assemble_bevy.py:125` ↔ `viewer/src/eftpack.rs:35`, must stay in lockstep):

- `1<<0 MIRROR` - `det3(conjugated affine) < 0`. The renderer flips winding / draws double-sided and relies on the cofactor normal matrix. NOT baked.
- `1<<1 TERRAIN` - MicroSplat terrain tile; drive the splat shader.
- `1<<2 BAKED` - identity affine, geometry PRE-BAKED into world space. Emitted ONLY for a rank-deficient 3x3.
- `1<<3 INACTIVE` - Unity `activeInHierarchy == false` geometry the oversized-inactive gate kept. Shipped, hidden by default.

The single degenerate case - `_degenerate(M3)`, DEFINED at `eft_pipeline/assemble_bevy.py:491-499`, called at `:1063`. It returns true when `max|M3| ≤ 1e-12`, or when `|det| ≤ (max|M3|)³ × 1e-9` AND the SVD agrees:

```
return bool(s[0] <= 0 or s[-1] < s[0] * 1e-6)
```

- note the disjunction: a zero-or-negative largest singular value is degenerate on its own, independently of the `σ_min/σ_max` ratio. Those instances are baked into ONE `baked_world` mesh with a single identity-affine instance carrying `FLAG_BAKED` (`:1090`). A small-but-uniform scale is explicitly NOT degenerate.

---

## colliders.bin and collider_meshes.bin

The physics world is NOT a subset of the render world - it is built from MeshRenderers' absence. The source comment (`eft_pipeline/assemble_bevy.py:135`, `viewer/src/nav_bake.rs:19`) records 131,945 of 141,347 interchange colliders as having no renderer at all; the shipped interchange pack now reports `colliderCount = 145,179` (`colliders.bin` 13,937,184 / 96), so treat those exact numbers as a stale anecdote and the ratio as indicative only. This is what Unity's own navmesh bakes from (`NavMeshSurface.m_UseGeometry = PhysicsColliders`).

Record **stride 96 B** (`eft_pipeline/assemble_bevy.py:139`, asserted; last 8 B are pad):

| offset | fmt | field |
|---|---|---|
| 0 | f32x12 | `affine` - same conjugation as a render instance, applied exactly once |
| 48 | u32 | `kind` - 0 box, 1 sphere, 2 capsule, 3 mesh |
| 52 | i32 | `meshId` - index into `manifest.colliderMeshes`, else −1 |
| 56 | f32x3 | `center` - Unity `m_Center`, collider-local, `G3`-mapped |
| 68 | f32x3 | `shape` - box: `m_Size` xyz \| sphere: `(r,0,0)` \| capsule: `(r, h, direction)` |
| 80 | u32 | `layer` - Unity `m_Layer`; name it via `manifest.layerNames` |
| 84 | u32 | `flags` |
| 88 | 8 B | pad |

Collider flags: `1<<0 TRIGGER` (no contact response, never blocks), `1<<1 NAV_IGNORE` (`NavMeshModifier.m_IgnoreFromBuild`), `1<<2 VISIBLE` (also a render instance), `1<<3 MIRROR`.

`collider_meshes.bin` = `[positions f32x3][u32 indices]`, same two-section pattern as `meshes.bin`; `vtxOffset` is `vertex_index * 12`, `idxOffset` is patched to `len(position_section) + local` (`eft_pipeline/assemble_bevy.py:1249`). No normals, no UVs, no colours - this geometry must never reach the renderer. A `kind == 3` collider whose OBJ is missing or degenerate is DROPPED and counted (`:1205`).

EFT separates MOVEMENT collision (`LowPolyCollider`) from HIT collision (`HighPolyCollider`); select by layer NAME via `layerNames`, never by a hardcoded index.

---

## grass.bin

Headerless array, **stride 24 B**, `format: 2` (`eft_pipeline/build_grass.py:23`, `:501`; consumer stride switch at `viewer/src/render/gpu_driven.rs:3136`):

| offset | fmt | field |
|---|---|---|
| 0 | f32x3 | x, y, z - PACK (viewer) space, exact terrain height |
| 12 | f32 | rotY - radians, `[0, 2π)` |
| 16 | f32 | scale - `[0.75, 1.35]` |
| 20 | u32 | kind - index into `grass_sidecar.kinds[]`, **written as the u32 bit pattern viewed as f32** |

Legacy `format` absent/1 → stride 20 (no kind lane), one implicit kind.

Placement is fully deterministic (`eft_pipeline/build_grass.py:459`), never client-random. Per prototype layer, per nonzero density cell `(cx, cy)` holding count `c`, for `k` in `0..c`:

```
h  = (cx*73856093) ^ (cy*19349663) ^ (k*83492791) ^ seed         seed = (proto_index+1) * 2654435761
h ^= h >> 13;  h *= 0x9E3779B97F4A7C15;  h ^= h >> 29           (uint64)
du = (h & 0xFFFF)/65535;  dv = ((h>>16) & 0xFFFF)/65535
u  = (cx + du)/side;      v  = 1 - (cy + dv)/side
rotY  = ((h>>32) & 0xFFFF)/65535 * 6.2831855
scale = 0.75 + ((h>>48) & 0xFF)/255 * 0.6
```

`u,v` are bilinearly sampled against a UV→world grid built from the PACK's own terrain meshes, so XZ and height are exact. The **v-flip is mandatory**: grids are dumped in Unity detail row order `[row = z][col = x]` while the terrain meshes carry image-frame UVs (`v = 1 - z_frac`); the un-flipped mapping drops ~75% of one lighthouse slice into the sea.

`grass_sidecar.json` = `{count, format: 2, kinds: [{albedo, tint[3]}], wind: {strength, amount, speed}, albedo, tint}` - the last two are legacy single-card fields for older consumers. Kind slots are POSITIONAL: a kind whose texture cannot be resolved is dropped AND `grass.bin`'s indices are remapped (`eft_pipeline/build_grass.py:577`), never left to shift. Emitting zero clumps or resolving zero textures is a hard `SystemExit` (`:547`, `:574`) because both silently disable grass - reserve once shipped 197,599 clumps that never drew a pixel.

---

## nav.bin and siblings

Layered 2.5-D heightfield (`viewer/src/nav.rs:10`, written at `viewer/src/nav_bake.rs:1914`). Cell index `ci = iz*nx + ix`; node = `ci*K + layer`.

- **`nav.bin`** - `f32[nx*nz*K]` LE. Layer `l` of cell `ci` at element `ci*K + l`. **Invariant: floor heights ASCENDING within a cell, `miss` sentinel in trailing slots only.** `miss` is `-1e9`.
- **`nav_door.bin`** - `u8[nx*nz]`, 1 = door cell (forced passable; routes cross closed doors). Stamped as a disc of radius 1.1 m around each typed door pivot.
- **`nav_blk.bin`** - `u8[nx*nz*K]`, 8-bit edge mask; bit `d` set = the edge to neighbour `NB[d]` is blocked by a thin wall/fence. `NB = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]` (`viewer/src/nav.rs:32`) - the bit order is part of the format.
- **`nav_wallcell.bin`** - `u8[nx*nz]`, 1 = a wall occupies this cell's body column (path-simplify guard).
- **`nav.json`** - `{map, min_x, min_z, res, nx, nz, n_layers, y_high, miss, climb, drop_max, step_up, ledge_drop_height, vault, slope_max_deg, walk_slope_deg, agent_radius, agent_height, min_region_area, agent_source, baker, baker_version, index, layout, ...}`.

World from grid: `x = min_x + ix*res`, `z = min_z + iz*res`, `y = nav.bin[ci*K + l]`. A shipped interchange grid is `res 0.5, nx 2634, nz 2060, n_layers 8` → `2634*2060*8*4 = 173,633,280` B, which is exactly the file size (a usable integrity check).

`baker_version` (`viewer/src/nav.rs:212`) is bumped ONLY when a baker change alters the CONTENT of `nav.bin`/`nav_blk.bin`/`nav_wallcell.bin` for the same input. A mismatch loads but is reported at error level; a pack claiming the current version while omitting `step_up` is flagged as older-baker output. Agent parameters come from Unity `NavMeshProjectSettings` via `packs/shared/nav_agents.json` - `ledgeDropHeight` and `maxJumpAcrossDistance` are 0 on every EFT agent, so the game's navmesh has NO drop or jump links, and descents are bounded by the same continuous-surface rule as ascents. `drop_max`/`step_up` are `free_step(res)` ≈ `res·tan(55°)` clamped to `[climb, VAULT]` - the grid's ALIASING limit, not a physical stride; without it every stair-only upper floor becomes a sealed island.

---

## volume.bin

SH irradiance volume (`viewer/src/sh_bake.rs:916`, consumed at `viewer/src/render/gpu_driven.rs:1065`).

- **`volume.bin`** - **float16 LE**, probe-major. Probe index `pi = ((z*ny) + y)*nx + x` (x fastest). Each probe is **12 halfs = 24 B**, ordered `c0.r, c0.g, c0.b, c1.r, c1.g, c1.b, c2.r, c2.g, c2.b, c3.r, c3.g, c3.b`. Total = `nx*ny*nz*24` B.
- Coefficients are **RADIANCE** SH in the L1 real basis: `0 = Y00 = 0.282095`, `1 = Y1-1 = 0.488603·y`, `2 = Y10 = 0.488603·z`, `3 = Y11 = 0.488603·x`. The consumer reconstructs IRRADIANCE by cosine convolution `A0 = π`, `A1 = 2π/3`. Probe order equals wgpu 3D texel order, so `pi` → texel is a straight copy.
- **`volume.json`** - `{min[3], max[3], dims[nx,ny,nz], spacing[3], coeffs: 4, channels: 3, layout, sun_dir[3], bounces, direct, validity, validity_layout, baker, gi_intensity?}`. `spacing` is emitter authority; absent → derive `(max-min)/(dims-1)`. `direct: false` marks an INDIRECT-ONLY bake, which is what enables the viewer's realtime practicals; a full bake (`EFT_BAKE=warp`) must disable them.
- **`sun_dir` is authored directly in VIEWER space and must NOT be X-flipped again** - flipping mirrors sun and shadows against the SH radiance. Nothing conjugates it: `viewer/src/sh_bake.rs:887` is `fn pack_sun_dir(_pack: &Pack) -> [f32;3] { [0.449, 0.799, -0.400] }`, a hardcoded constant that ignores the pack ("a fixed neutral default matching interchange's baked sun_dir"), and the warp baker authors the same vector - `extraction/bake/bake_volume2.py:360` sets `sun = normalize([0.45, 0.80, -0.40])` and logs "no Directional light exists in EFT scenes -> FALLBACK warm sun". The comment at `viewer/src/render/gpu_driven.rs:3485` saying the bake conjugates it is WRONG about the mechanism; the rule it guards is right.
- **`volume_valid.bin`** - `u8` per probe, probe-major, SAME index as `volume.bin`. `255` = open space, `0` = buried in geometry (backface ratio ≥ 0.25). Read by FIXED NAME. Length ≠ `nx*ny*nz` → ignored, every probe treated valid.

**The mismatch this format cannot self-detect**: probe COUNT can match while `min` and `spacing` differ, so a `volume.bin` from one bake plus a `volume_valid.bin` from another passes every size check while masking probes against geometry that is not there. Recorded case: two bakes sharing a `401×13×302` grid, origin off 7.6 m in Z, spacing off 0.05 m/cell, ~20 m of drift at the far edge, 677,882 probes mis-masked, mall interiors rendering flat. Those dims are historical - today's interchange bake is `402×13×301`, a different probe count - which is the point: compare `min` and `spacing`, never counts. The three files must always travel together, which is why `finalize_pack_manifest` unconditionally repoints the manifest at the PACK's own copies.

---

## Self-contained / hardlink shipping mode

`--self-contained` (`eft_pipeline/assemble_bevy.py:512`) makes every referenced file live INSIDE the pack with pack-RELATIVE references. Default OFF: a dev pack references textures and sidecars by ABSOLUTE path and copies nothing.

- `_PackShipper.ship(src, rel)` (`:533`) tries **`os.link` first** and only falls back to a copy. A self-contained streets pack references ~6.4 GB of textures that already exist byte-identical and read-only in the dataset; a hardlink is the same inode - no bytes moved, no extra space, ~56 s saved per build (`:536-539`). Hardlinks require the same volume; cross-volume/SMB falls back to `copyfile` into `<dst>.copying-<pid>-<id>` + `os.replace`, with 4 attempts and exponential backoff (0.25/0.5/1.0 s), so a transient share failure never leaves a plausible-looking partial texture in the `.building` pack.
- **Immutability contract**: anything that later rewrites a texture MUST write-temp-then-`os.replace`, never edit in place, or it mutates every pack sharing the inode.
- `ship_tex(src)` (`:592`) flattens into `tex/<basename>`. Two DIFFERENT sources with the same basename: if sha1 matches, they share one copy; otherwise the second gets `<stem>.<sha1[:8]>.png`. A MISSING source still returns `tex/<basename>` and is tallied - the loader's missing-texture fallback is identical for relative and absolute paths.
- `ship_dir(srcdir, reldir)` copies `terrain_layers/` flat, skipping `*.bak`. `_relativize_tl_manifest` (`:654`) rewrites any absolute `*.png` inside the copied terrain manifest to its basename, since that sidecar resolves names against its OWN directory.
- `_self_contain_materials` (`:633`) rewrites EVERY texture-bearing field: `albedo`, `normal`, `specMap`, `emissive.texture`, `detail.albedo/.normal`, `vp.layers[].albedo/.normal`, `vp.heights`.
- `manifest.datasetPath` stays ABSOLUTE deliberately (build provenance only; no consumer resolves through it) and `manifest.selfContained = true` marks the mode.
- Consumer rule (`viewer/src/eftpack.rs:1067`): relative → join against the pack dir; absolute → pass through untouched. Light sidecars additionally prefer a pack-local file of the same BASENAME even when the manifest names an absolute path (`:1187`), so a moved pack still resolves.
- Post-build, `tools/dedup_textures.py` hardlinks duplicate textures across `<assets>/*/{tex,terrain_layers}/` down to one master, detached, after `[BUILD OK]`. UNVERIFIED - the script was not read: its stated grouping key `(size, sha1)`, its idempotence and its per-file best-effort behaviour are inferred from the invocation, not from the source.

---

## Pack tiers and the file probe

```
Markers < Routes < Full                     (viewer/src/eftpack.rs:1018)
```

`available_tier(dir)` (`viewer/src/eftpack.rs:1081`) probes FILES ON DISK - `metadata().len() > 0` - and deliberately reads nothing the pack says about itself, because a half-swapped or crashed-mid-assemble pack will happily claim a tier it cannot serve:

1. no `manifest.json` (or zero-length) → `None`
2. `meshes.bin` AND `instances.bin` AND `materials.json` all non-empty → `Full`
3. else `nav.json` AND `nav.bin` → `Routes`
4. else `gamedata.json` → `Markers`
5. else `None`

Loading below `Full` (`:1152`) skips `materials.json`, `meshes.bin` and `instances.bin` entirely - on interchange that is 708 MB of the 725 MB otherwise read - and then FORCES the manifest self-consistent: `meshes` cleared, `instanceCount = 0`, `materialCount = 0`, `seaLevel = None`, and `sidecars.{grassJson, terrainLayers, volume, volumeMeta, volumeVis}` nulled. Both halves are load-bearing: `seaLevel` synthesizes an ocean quad + material + instance BEFORE the mesh-count bail, and grass/terrain build their OWN instance streams straight off disk, so leaving them set gives foliage and ground floating over an empty world. The grass path re-checks `pack.tier != Full` independently (`viewer/src/render/gpu_driven.rs:3113`).

`Pack.tier` records what was ACTUALLY loaded; consumers must branch on that, not on what the directory could have supported.

---

## Invariants and failure signatures

| Invariant | Broken → |
|---|---|
| `len(instances.bin) % instance.stride == 0` | load error "not a multiple of stride" |
| `instanceCount == len(instances.bin)/stride` | load error (`eftpack.rs:1144`) |
| every vertex attr `[offset, offset+size) ⊆ [0, stride)` | load error naming the attr; unchecked it reads past the last vertex |
| every instance/collider field `offset + size ≤ stride` | load error naming the field, instead of a panic deep in the record loop |
| `mesh.vtxOffset + vtxCount*stride ≤ len(meshes.bin)` and `idxOffset + idxCount*4 ≤ len` | load error; unchecked it panic-aborts at a GPU slice (release is `panic = abort`) |
| `submesh.idxStart + idxCount ≤ mesh.idxCount` | load error; unchecked the GPU submesh slice panics |
| `submesh.materialId < len(materials)` | load error |
| `instance.meshId < len(manifest.meshes)` | load error; `instances_by_mesh` would drop it silently while the GPU path indexes with it |
| `f0` advances for EVERY submesh including skipped ones | later submeshes read face ranges shifted earlier by `n`; the last submesh loses its final `n` faces - a see-through hole in otherwise-correct ground |
| a dropped LOD shell has a KEPT shell that both renders and encloses it | see-through holes; a crashed extraction left NUL-filled OBJs whose group-minimum shell drew nothing after its replacement was deleted |
| a LOD level is dropped ALL-OR-NOTHING per group | that level's distance band draws a PARTIAL object (source comment: trailer body vanishing between ~7.4 and ~24.7 m, `assemble_bevy.py:864`) |
| `len(volume_valid.bin) == dims.x*dims.y*dims.z` | validity ignored, every probe treated valid → light leaks through walls |
| `volume.bin`, `volume.json`, `volume_valid.bin` from the SAME bake | mis-masked probes; NOT detectable by any size check |
| every `lights_*.json` in the pack appears in `sidecars.lightsAll` | that bank never loads; a switch controlling it resolves to zero groups ("Power (no lights)") |
| `nav.json.baker_version` matches the router's | routes pass through walls; reported at error level, not silently |
| `nav.bin` floors ascending, `miss` trailing | neighbour resolution picks the wrong floor |
| `grass_sidecar.kinds` slots positional w.r.t. `grass.bin` kind ids | wrong billboard per clump, or dropped kinds shifting every later index |
| cull keeps > 0.5% of raw instances | hard `SystemExit` (guard `assemble_bevy.py:764`, raise `:765`) |
| not ALL probed LOD meshes are unreadable | hard `SystemExit` (guard `:883`, raise `:884`) - a total probe failure means a wrong `mesh_dir`, and every drop it authorised would be a guess |
| `G3` is a signed permutation when colliders are emitted | hard `SystemExit` (`:1158`) - box/capsule parameterisation cannot survive a rotational global matrix |
| `manifest.json` contains no non-finite floats | `serde_json` rejects the file and the pack is unloadable; sanitized to 0.0 with a report instead |

Freshness, not presence, decides whether a stage succeeded. Stages 3, 6 and 8 each capture a start timestamp and compare `os.path.getmtime` of their output against it (`tools/build_map.py:999`, `:1099`, `:1143`), because the atomic swap migrates any file the staging dir lacks - so a previous build's `volume.bin`/`gamedata.json`/`nav.bin` survives a failed re-bake and looks exactly like a fresh one.

---

## Old patterns

- **Hardcoded `*_Light` scene indices per map** - replaced by `extraction/maps/manifest.json` `light_levels` (a LIST, so multi-scene maps get full lighting) with a live-BuildSettings fallback.
- **Hardcoded `INDOOR_NO_GRASS` map set** - replaced by the data-driven test "does this dataset yield density grids".
- **`config.source.levels` as the level list** - replaced by live BuildSettings derivation UNIONED with the config. The config drifts as the game adds scenes; a missing `*_DesignStuff` level cost reserve 992 loot containers and left crates floating on un-extracted geometry.
- **Keep-min-LOD dedup** - replaced by the prove-the-premise rule (a shell goes only when kept shells in its group render AND enclose it).
- **The web three-way instance/bake gate** (`det<0` → bake, shear ≥ 0.02 → bake, else TRS) - replaced by emitting the full conjugated 3x4 for every instance plus a `MIRROR` flag. TRS decomposition is never correct here.
- **`volume.vis.bin`** - a legacy web-viewer artifact; not promoted from the warp bake and not read by the native viewer. The live file is `volume_valid.bin`.
- **A single `lights` sidecar field** - superseded by `lightsAll[]`; `lights` is now just the primary entry.
- **Inline volume-only manifest patching after the SH bake** - superseded by `finalize_pack_manifest`, which runs LAST and covers volume AND lights in one place; the inline version is why a light sidecar appearing after assemble stayed invisible.
- **`api.tarkov.dev` GraphQL** for item names/icons - replaced by the cached static JSON dump (offline-safe) through `tarkov_static.load_static_items`.