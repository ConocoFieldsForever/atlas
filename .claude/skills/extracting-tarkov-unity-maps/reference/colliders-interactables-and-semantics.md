## Contents

1. [Scope, stages, and what each tier keeps](#1-scope-stages-and-what-each-tier-keeps)
2. [Dataset tier: `colliders.json`](#2-dataset-tier-collidersjson)
3. [Pack tier: `colliders.bin` record layout](#3-pack-tier-collidersbin-record-layout)
4. [The collider flag legend, bit by bit](#4-the-collider-flag-legend-bit-by-bit)
5. [`collider_meshes.bin`](#5-collider_meshesbin)
6. [Convex vs concave](#6-convex-vs-concave)
7. [Primitive tessellation: the math a consumer must generate](#7-primitive-tessellation-the-math-a-consumer-must-generate)
8. [Consumer selection rules (the nav bake)](#8-consumer-selection-rules-the-nav-bake)
9. [`interact_<lv>.json`](#9-interact_lvjson)
10. [Folding interactables into `gamedata.json`](#10-folding-interactables-into-gamedatajson)
11. [The GameObject-name semantic layer](#11-the-gameobject-name-semantic-layer)
12. [Invariants and failure signatures](#12-invariants-and-failure-signatures)
13. [Old patterns](#13-old-patterns)

---

## 1. Scope, stages, and what each tier keeps

Three separate data paths, three separate files. No collider geometry reaches the renderer. The interactables path does reach the viewer, as `PoiLayer::Switch` markers and the info card's "Requires switch" line (`viewer/src/poi.rs:3109, :1598`).

| Stage | Producer | Output | Consumer |
|---|---|---|---|
| Physics colliders | `extraction/unity/eft_extract_colliders.py` | `<dataset>/colliders.json` + OBJs in `<dataset>/meshes/` | `eft_pipeline/assemble_bevy.py` |
| Pack | `eft_pipeline/assemble_bevy.py:1143-1257` | `<pack>/colliders.bin`, `<pack>/collider_meshes.bin`, manifest keys | `viewer/src/eftpack.rs`, `viewer/src/nav_bake.rs` |
| Interactables | `extraction/unity/extract_interact.py` | `<dataset>/interact_<lv>.json` | `tools/build_map.py:515` → `gamedata.json` → `viewer/src/eftpack.rs:922` |

Figures tagged **source-comment** in this document are quoted from the cited in-repo comment: a sample taken on the named map when that code was written, not a re-measurement.

The render pack is built from MeshRenderers, so it holds only geometry you can see. The physics world is mostly invisible: measured on the shipped interchange dataset and pack, 135,818 of 145,179 colliders have `vis == false` - no renderer on the GameObject at all. The comments at `eft_pipeline/assemble_bevy.py:135` and `viewer/src/nav_bake.rs:19` still quote an earlier sample of the same ratio (131,945 of 141,347); the shape of the claim holds, the counts move build to build. Source-comment, interchange level63 alone: 31,015 SphereCollider, 8,253 BoxCollider, 7,732 MeshCollider, 6,360 CapsuleCollider, 4 TerrainCollider (`extraction/unity/eft_extract_colliders.py:12-13`).

**TerrainCollider is not extracted.** `COLLIDER_TYPES` at `extraction/unity/eft_extract_colliders.py:82` is exactly `("BoxCollider", "SphereCollider", "CapsuleCollider", "MeshCollider")`. Terrain collision must be taken from the render terrain tiles. WheelCollider and every other Unity collider type is likewise absent.

Fields drop between tiers. `colliders.json` carries `go`, `root`, `lv`, `nav_area` and `convex`; the pack record tuple at `eft_pipeline/assemble_bevy.py:1239` is `(aff, kind, mid, ctr, shape, lyr, flags)` - **all five of those fields are discarded at pack time**. A consumer that needs per-collider names, source level, nav-area override or convexity must read the dataset JSON, not the pack.

---

## 2. Dataset tier: `colliders.json`

UTF-8 JSON, written with `separators=(",", ":")` (`extraction/unity/eft_extract_colliders.py:404-406`). Top level:

```
{"colliders": [ <record>, ... ],
 "counts":    {"BoxCollider": n, "SphereCollider": n, "CapsuleCollider": n, "MeshCollider": n},
 "layers":    {"<layer index as string>": "<TagManager layer name>"}}
```

`layers` comes from `globalgamemanagers`'s TagManager, an engine type with a hardcoded type tree, so it reads despite encrypted IL2CPP metadata (`extraction/unity/eft_extract_colliders.py:423-438`). Unnamed layers are omitted entirely. Known EFT indices (`extraction/unity/eft_extract_colliders.py:38-39`): 9 `DoorLowPolyCollider`, 11 `Terrain`, 12 `HighPolyCollider`, 13 `Triggers`, 18 `LowPolyCollider`, 26 `Foliage`, 29 `LevelBorder`, 30 `TransparentCollider`, 31 `Grass`. Do not hardcode these; resolve by name from `layers`.

### Record fields

Every record (`extraction/unity/eft_extract_colliders.py:344-383`):

| Key | Type | Present | Meaning |
|---|---|---|---|
| `m` | 16 floats | always | RAW Unity world 4×4, row-major flatten, each value rounded to 5 dp |
| `lv` | int | always | source level index |
| `t` | string | always | `"box"` \| `"sphere"` \| `"capsule"` \| `"mesh"` |
| `c` | [3] float | box/sphere/capsule | Unity `m_Center`, collider-local, metres |
| `s` | [3] float | box only | Unity `m_Size` - FULL extent, not half; default `[1,1,1]` |
| `r` | float | sphere/capsule | Unity `m_Radius`, metres; default `0.5` |
| `h` | float | capsule only | Unity `m_Height`, TOTAL including both hemispherical caps; default `2.0` |
| `d` | int | capsule only | Unity `m_Direction`: 0 = X, 1 = Y, 2 = Z; default `1` |
| `mesh` | string | mesh only | OBJ filename in `<dataset>/meshes/` |
| `convex` | bool | mesh only | Unity `m_Convex` |
| `go` | string | always | the collider GameObject's own `m_Name` |
| `root` | string | always | name of the TOPMOST named ancestor (`root_of`, `extraction/unity/eft_extract_colliders.py:211-222`) |
| `lyr` | int | always | GameObject `m_Layer` |
| `vis` | bool | always | the GameObject also owns a MeshRenderer/SkinnedMeshRenderer |
| `trig` | true | only when true | `m_IsTrigger` |
| `nav_ignore` | true | only when true | `NavMeshModifier.m_IgnoreFromBuild` |
| `nav_area` | int | only when overridden | `NavMeshModifier.m_Area` when `m_OverrideArea` is set |

Absent `trig`/`nav_ignore` means false. Absent `nav_area` means "no override" - it is **not** area 0. Area indices: 0 Walkable (cost 1.0), 1 Not Walkable, 2 Jump (2.0), 3 Sitdown (1.0), 4 Danger (2.0), 5 Terrain (1.0) (`extraction/unity/eft_extract_nav.py:26-27`).

### The world matrix `m`

Built by the memoised father-chain walk at `extraction/unity/eft_extract_colliders.py:164-178`. Each local matrix is `trs(t)` from `extraction/unity/eft_scene_extract.py:35-40`: `M[:3,:3] = quat_to_mat(m_LocalRotation) @ diag(m_LocalScale)`, `M[:3,3] = m_LocalPosition`. Composition is `W = M_root @ … @ M_leaf`, i.e. full 4×4 father-chain multiplication. It is **raw Unity** - left-handed, Y-up, metres. No handedness flip is applied here; the assembler owns that (`extraction/unity/eft_extract_colliders.py:24-27`).

Row-major flatten means element `(r, c)` is at index `r*4 + c`; translation lands at indices 3, 7, 11.

### Skips

Six drop paths, four of them counted. The `skipped` dict (`extraction/unity/eft_extract_colliders.py:318`) counts `disabled` (`m_Enabled == 0`, `:332-334`), `inactive` (any ancestor with `m_IsActive` false, `:335-337`), `no_transform` (no Transform on the GameObject, `:338-341`) and `no_mesh` (a MeshCollider whose mesh export failed, `:365-367`), and is printed per level at `:388`. Two further drops are silent and appear in no counter: a collider whose `o.read_typetree()` raises (`:323-326`), and one whose `m_GameObject.m_PathID` is 0 or absent (`:327-329`). A level whose emitted count undershoots its `COLLIDER_TYPES` object count by more than the printed `skipped` totals lost the difference to those two.

**Triggers are never dropped** - they are flagged and shipped so a consumer can use them (slow volumes, water, splash).

### MeshCollider OBJ naming

`f"{san(name)}__{lv}_{fileID}_{pathID}.obj"`, where `san` is `"".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))[:96]` (`extraction/unity/eft_extract_colliders.py:97-98, 301`): every other character becomes `_`, then the result is truncated to 96 characters. `str.isalnum()` is Unicode-aware, so the surviving set is every Unicode letter and digit plus `._-`, NOT ASCII `[A-Za-z0-9._-]`. Non-ASCII names pass through verbatim: `manifest.colliderMeshes` on interchange contains `Сupboard3_Collider_set_3__56_8_1870.obj`, whose leading `С` is Cyrillic U+0421, not ASCII `C`. A consumer that re-derives filenames with an ASCII-only sanitiser will fail to join that mesh.

Interning key is `(lv, file_id, path_id)`. Files land in the SAME `meshes/` directory as render meshes, so the assembler resolves both from one place.

`_obj_complete` from `extraction/unity/fileguards.py` is the completeness guard, and writes go through `_atomic_write`. A size check alone cannot see NTFS preallocation zeros: a killed collider pass leaves NUL-filled OBJs that parse without raising, yield zero vertices, and silently remove the wall from the nav bake (`extraction/unity/eft_extract_colliders.py:61-69`).

### NavMeshModifier payload decode

The MonoBehaviour raw payload layout used at `extraction/unity/eft_extract_colliders.py:266-288`, all little-endian:

```
byte  0  int32   m_GameObject.m_FileID
byte  4  int64   m_GameObject.m_PathID          <- struct.unpack_from("<q", raw, 4)
byte 12  int32   m_Enabled                      [inferred; not read here]
byte 16  int32   m_Script.m_FileID              <- script_class: unpack_from("<iq", raw, 16)
byte 20  int64   m_Script.m_PathID
byte 28  int32   m_Name length (0 here)         [inferred; not read here]
byte 32  int32[8] body                          <- NAVMOD_FIELD_OFF = 32
```

Only three offsets are exercised by code: `<q` at 4 (`:287`), `<iq` at 16 (`:244`), `<8i` at 32 (`:285-286`). The rows at 12 and 28 follow from the 28-byte-header comment at `:83-85` but are read and asserted nowhere in this repo - treat them as unverified and re-derive them from bytes before relying on them.

Of the eight int32s: `f[0]` = `m_OverrideArea`, `f[1]` = `m_Area`, `f[4]` = `m_IgnoreFromBuild`. `f[2]`, `f[3]`, `f[5..7]` are not read. The class identity comes from the `m_Script` PPtr resolved to a `MonoScript`, whose `m_Namespace`/`m_ClassName` are readable engine fields. Source-comment (`:83-85`): the layout was checked against all 5,764 NavMeshModifier instances in interchange level63.

---

## 3. Pack tier: `colliders.bin` record layout

Defined once, at `eft_pipeline/assemble_bevy.py:139-153`, and asserted to 96 bytes:

```python
CDT = np.dtype([('affine','<f4',(12,)), ('kind','<u4'), ('meshId','<i4'),
                ('center','<f4',(3,)),  ('shape','<f4',(3,)),
                ('layer','<u4'), ('flags','<u4'), ('_pad','<u4',(2,))])
assert CDT.itemsize == 96
```

**Endianness: little-endian, every field, no exceptions** (`<` on every numpy dtype; `f32::from_le_bytes` / `u32::from_le_bytes` on the read side, `viewer/src/eftpack.rs:1889-1904`).

| Offset | Size | Type | Field | Meaning |
|---:|---:|---|---|---|
| 0 | 48 | f32×12 | `affine` | ROW-MAJOR world 3×4, already handedness-conjugated |
| 48 | 4 | u32 | `kind` | 0 box, 1 sphere, 2 capsule, 3 mesh |
| 52 | 4 | i32 | `meshId` | index into `manifest.colliderMeshes` when `kind == 3`, else `-1` |
| 56 | 12 | f32×3 | `center` | Unity `m_Center`, collider-local, **G3-applied** |
| 68 | 12 | f32×3 | `shape` | per-kind, see below |
| 80 | 4 | u32 | `layer` | Unity `m_Layer`; name via `manifest.layerNames` |
| 84 | 4 | u32 | `flags` | see §4 |
| 88 | 8 | - | `_pad` | zero-filled trailing padding |

**Stride is 96 including the 8 trailing pad bytes.** There is no file header and no per-record count: `parse_colliders` derives it as `count = bin.len() / stride` (`viewer/src/eftpack.rs:1863`), behind the single `bin.len() % stride != 0` guard (`:1819-1825`). `manifest.colliderCount` is written by the assembler (`eft_pipeline/assemble_bevy.py:1349`) and deserialised into `Manifest::collider_count` (`viewer/src/eftpack.rs:133-134`), but nothing reads it: it is never compared against the parsed count, and `Pack::validate` (`viewer/src/eftpack.rs:1335`) does not touch it. The field is informational; the file length is the count. `manifest.collider.stride` and `manifest.collider.fields` govern the layout - the viewer resolves offsets by field NAME and bounds-checks `offset + size <= stride` before parsing (`viewer/src/eftpack.rs:1826-1862`). Do not read `_pad`; it is reserved.

### `shape` semantics, per kind

Written at `eft_pipeline/assemble_bevy.py:1217-1224`, read at `viewer/src/nav_bake.rs:628-638`:

| `kind` | `shape.x` | `shape.y` | `shape.z` | `center` |
|---|---|---|---|---|
| 0 box | `m_Size.x` (full extent) | `m_Size.y` | `m_Size.z` | `m_Center`, G3-applied |
| 1 sphere | radius | 0.0 | 0.0 | `m_Center`, G3-applied |
| 2 capsule | radius | height (total, caps included) | direction as f32: 0=X, 1=Y, 2=Z | `m_Center`, G3-applied |
| 3 mesh | 0.0 | 0.0 | 0.0 | **always (0,0,0) - ignore it** |

For `kind == 3` the extractor writes no `c`, so `c.get('c') or [0,0,0]` yields zeros; mesh vertices carry their own origin.

### The affine and the handedness conjugation

`mg = apply_global(c['m'])` = `G4 @ M @ G4⁻¹`, then `aff = mg.reshape(4,4)[:3,:].flatten()` (`eft_pipeline/assemble_bevy.py:1210-1211`, `eft_pipeline/tarkmap_core/instmath.py:21-22`). `G3 = diag(-1, 1, 1)`. This is the SAME conjugation a render instance gets, applied exactly ONCE.

Reconstruct in the pipeline's row-vector convention:

```
M3 = [[a0, a1, a2],
      [a4, a5, a6],
      [a8, a9, a10]]          T = (a3, a7, a11)

p_world = p_local @ M3.T + T           n_world = n_local @ inv(M3)
```

`viewer/src/eftpack.rs:198-208` builds the same thing column-wise for `glam::Affine3A`. `M3` may contain shear and may be mirrored. Never TRS-decompose it.

### Why `center` is conjugated but `shape` is not

`ctr = G3 @ c` (`eft_pipeline/assemble_bevy.py:1237-1238`). Primitive geometry is *generated* from `center`/`shape` at bake time in the viewer frame, so its local space is not automatically G-applied - unlike a MeshCollider OBJ, whose vertices UnityPy's `mesh.export()` has already X-negated. That flip is asserted only by the in-repo comments at `eft_pipeline/assemble_bevy.py:1226-1228` and `extraction/unity/eft_extract_colliders.py:25-27`; UnityPy is third-party and the flip is not independently verified here.

Applying the conjugated affine `G·M·G⁻¹` to an un-flipped centre mirrors the primitive about its own pivot. The world error is `2·c.x` times the affine's X column, since `G·M·G⁻¹·c − G·M·c = G·M·(2c.x, 0, 0)`. Source-comment, interchange: 2,704 misplaced nav colliders, up to 4.02 m out.

`shape` needs no transform under a signed-permutation `G`: box size is symmetric about the centre, a sphere is isotropic, and a capsule's axis maps onto itself up to sign, so the direction index is invariant.

### The signed-permutation guard

Before writing anything, `eft_pipeline/assemble_bevy.py:1154-1162` verifies `|G3|` has exactly one 1 per row and per column, and exits with:

```
[bevy] global_matrix is not a signed permutation; collider box/capsule parameterisation
cannot be expressed in the viewer frame. Refusing to emit silently-wrong colliders.
```

A rotational global matrix breaks the axis-aligned parameterisation entirely, so the assembler exits rather than write a file whose box and capsule parameters cannot express the transform.

---

## 4. The collider flag legend, bit by bit

`eft_pipeline/assemble_bevy.py:154-158`, mirrored in `viewer/src/eftpack.rs:167-177` and published as `manifest.collider.flagsLegend` (`eft_pipeline/assemble_bevy.py:1345-1348`):

| Bit | Mask | Name | Source | Meaning |
|---|---|---|---|---|
| 0 | `0x1` | `TRIGGER` | `colliders.json` `trig` | Unity `m_IsTrigger`. No contact response - never blocks movement |
| 1 | `0x2` | `NAV_IGNORE` | `nav_ignore` | `NavMeshModifier.m_IgnoreFromBuild`; the GAME excludes it from its bot navmesh |
| 2 | `0x4` | `VISIBLE` | `vis` | the GameObject also has a MeshRenderer, so it is already a render instance |
| 3 | `0x8` | `MIRROR` | computed | `det3(apply_global(m)) < 0`; triangle winding is reversed for this collider |

Bits 4-31 are unused and written zero.

**These bits are a SEPARATE numbering space from the instance flags.** The instance legend (`eft_pipeline/assemble_bevy.py:124-130`) is: `0x1` MIRROR, `0x2` TERRAIN, `0x4` BAKED_WORLD, `0x8` INACTIVE. The two spaces collide on every bit, and MIRROR in particular sits at bit 0 for instances and bit 3 for colliders.

Failure signature for confusing them: every trigger in the map reads as mirrored, so its `ny` sign is negated, every box's top face is classified as a ceiling and its underside as a floor, and the nav bake grows a walkable surface under each trigger volume - on interchange, a phantom floor inside every swamp and foliage volume.

---

## 5. `collider_meshes.bin`

Deliberately a separate file from `meshes.bin`: this geometry is invisible and must never reach the renderer (`eft_pipeline/assemble_bevy.py:1167-1168`, `viewer/src/eftpack.rs:135-138`).

Layout is two concatenated regions, in this order:

```
[ vertex region : all meshes' positions, f32x3 LE, 12 B per vertex ]
[ index region  : all meshes' indices,   u32   LE,  4 B per index  ]
```

Positions only. No normals, no UVs, no colours.

Per-mesh directory in `manifest.colliderMeshes` (built at `eft_pipeline/assemble_bevy.py:1183-1192`, patched at `:1247-1249`, deserialised at `viewer/src/eftpack.rs:153-165`):

| Key | Type | Meaning |
|---|---|---|
| `id` | u32 | index into the array; equals `Collider.meshId` |
| `name` | string | source OBJ filename |
| `vtxOffset` | u64 | BYTE offset from file start into the vertex region |
| `vtxCount` | u32 | vertex COUNT (bytes = `vtxCount * 12`) |
| `idxOffset` | u64 | BYTE offset from file start = `len(vertex region) + local index offset` |
| `idxCount` | u32 | index COUNT (bytes = `idxCount * 4`); always a multiple of 3 |

Indices are **mesh-local, 0-based** into that mesh's own vertex block, not global into the file. They come from `load_obj`'s `F[:, :, 0]` (`eft_pipeline/assemble_bevy.py:1179`), which is the OBJ's 1-based face indices minus one (`eft_pipeline/tarkmap_core/objio.py:19-21`). Triangle list, no strips, no primitive restart.

Interning is by OBJ filename (`cmesh_id`), so a mesh shared by N colliders is stored once. A filename that fails to load, yields zero vertices, or fewer than 3 indices caches `-1`, and **the collider record is dropped entirely** with a count reported as "mesh colliders dropped (OBJ missing)" (`eft_pipeline/assemble_bevy.py:1173-1182, 1202-1207, 1257`).

Bounds checking on read: `vtxOffset + vtxCount*12` and `idxOffset + idxCount*4` must both be `<= len(file)`, else the entry returns nothing rather than panicking (`viewer/src/eftpack.rs:1448-1460`).

Mesh vertices are already in the G-applied local frame (X-negated by `mesh.export()`, per the in-repo comments cited in §3 and unverified against UnityPy itself), so applying `G·M·G⁻¹` to them is correct and a second flip is not.

---

## 6. Convex vs concave

`m_Convex` is read and recorded per MeshCollider at `extraction/unity/eft_extract_colliders.py:370` as `rec["convex"]`, and it goes no further. `assemble_bevy` never reads the key; the pack record has no field for it; `colliders.bin` has no bit for it; `viewer/src/eftpack.rs` does not deserialise it.

**The pack treats every MeshCollider as a raw triangle soup.** Both convex and concave meshes are tessellated identically and consumed identically (`viewer/src/nav_bake.rs:639-656`). Unity's own semantics - a convex MeshCollider is replaced by its hull (≤255 faces) and is the only kind that can be a trigger or a rigidbody - are not modelled anywhere in this pipeline.

A reimplementer who needs the distinction (a physics engine that requires convex decomposition, for example) must read `convex` from `colliders.json` and re-join to the pack by OBJ filename via `manifest.colliderMeshes[i].name`. There is no other join key: `go`, `root` and `lv` are gone by then.

---

## 7. Primitive tessellation: the math a consumer must generate

The pack stores parameters, not geometry. This is the reference tessellation (`viewer/src/nav_bake.rs:976-1064`), and every primitive **must be wound OUTWARD**.

**Box** (`shape_box`, 8 vertices / 12 triangles). Half-extent `h = size * 0.5`. Vertex order is the sign lattice with `corner index = (z<<2) | (y<<1) | x`, each sign from `[-1, +1]`, position `center + (h.x*sx, h.y*sy, h.z*sz)`. Faces:

```
-z: [0,2,1] [1,2,3]   +z: [4,5,6] [5,7,6]
-y: [0,1,4] [1,5,4]   +y: [2,6,3] [3,6,7]
-x: [0,4,2] [2,4,6]   +x: [1,3,5] [3,7,5]
```

**Sphere** (`shape_sphere`, RINGS = 6 latitude bands, SEGS = 10 longitude segments). `phi = pi*i/RINGS` for `i` in `0..=RINGS`, `theta = tau*j/SEGS` for `j` in `0..SEGS`, position `center + (r*sin(phi)*cos(theta), r*cos(phi), r*sin(phi)*sin(theta))`. Quad `(a,b,c,d) = (i*SEGS+j, i*SEGS+(j+1)%SEGS, (i+1)*SEGS+j, (i+1)*SEGS+(j+1)%SEGS)` → `[a,b,c]`, `[b,d,c]`. The polar bands collapse to zero-area triangles; discard any triangle whose cross-product length is below `1e-12`.

**Capsule** (`shape_capsule`, SEGS = 10). `half = max(height*0.5 - r, 0)` is the cylindrical half-length between cap centres - note `height` is the TOTAL including caps. `axis` = `X` for `dir==0`, `Z` for `dir==2`, `Y` otherwise (so an out-of-range direction falls back to Y). Radial basis: `u = normalize(X × axis)` when `|axis.x| < 0.9` else `normalize(Y × axis)`; `w = axis × u`. Four rings at `(offset, radius) = (-(half+r), 0), (-half, r), (half, r), (half+r, 0)`, each `center + axis*off + (u*cos(theta) + w*sin(theta))*rad`. Same quad → two-triangle rule across the 3 bands. This is a capped cylinder, not a true hemispherical capsule: exact enough at ~1 m nav resolution, and the caps only affect headroom.

Failure signature for inward winding: `resolve_column` classifies a surface purely on the sign of `ny`, so an inward-wound primitive's top reads as a ceiling and its underside reads as a floor - a walkable surface invented in mid-air. `collider_primitives_are_wound_outward` at `viewer/src/nav_bake.rs:3928-3945` asserts `n · (centroid − centre) > 0` for all three primitives and all three capsule directions.

---

## 8. Consumer selection rules (the nav bake)

`add_collider_tris` (`viewer/src/nav_bake.rs:552-715`) applies four filters, in this order, and prints the tally of each:

1. **Trigger** - `flags & TRIGGER` → skip. A Unity trigger has no contact response. Source-comment, interchange (`viewer/src/nav_bake.rs:549-551`): 5,763 `Swamp_collider` boxes on the `Triggers` layer and 26,450 `Foliage` bush volumes are triggers.
2. **`NAV_IGNORE`** - skip. This is BSG's own authored answer about what is not navigation geometry. Source-comment, streets (`viewer/src/nav_bake.rs:593-596`): 2,329 objects.
3. **Layer allowlist** - `NAV_COLLIDER_LAYERS` (`viewer/src/nav_bake.rs:538-545`) is exactly `["LowPolyCollider", "DoorLowPolyCollider", "Terrain", "LevelBorder", "TransparentCollider", "Default"]`, matched by NAME through `manifest.layerNames` so no index is hardcoded. `HighPolyCollider` is deliberately excluded: it is the fine ballistics/hit shell on the same objects, so including it doubles every surface for no navigational gain.
4. **World-backstop gate** - a box whose world extent has `max(ext.x, ext.z) >= SLAB_MIN_SPAN` (500.0) and `ext.y <= SLAB_MAX_THICK` (5.0) is a fall-out-of-the-world catcher, not a floor (`viewer/src/nav_bake.rs:582-583`). Extent is computed shear-correctly as `|m3.x_axis*s.x| + |m3.y_axis*s.y| + |m3.z_axis*s.z|`, never from `manifest.bounds` (which includes the distant skyline backdrop). Source-comment, streets (`viewer/src/nav_bake.rs:567-583`): an 1874 × 1.0 × 1898 m `TEMP_GROUND_COLIDER` on `LowPolyCollider` at y ≈ −16; without the gate 97.3% of streets' floored cells sat on it and routes ran 18 m underground. ground_zero ships one too; interchange does not.

Doors need no name matching at the collider tier: `pack.layer_name(c.layer) == "DoorLowPolyCollider"` is the whole rule (`viewer/src/nav_bake.rs:666`). A door-tagged triangle stays out of the wall set and stamps a door cell the router may force an edge through.

Wall/floor split per triangle, identical to the render path: `ny = normal.y / |normal|`, negated when `MIRROR` is set; a wall is `|ny| < WALL_MAX_NY` (0.38) AND (`area >= WALL_MIN_AREA` (0.04 m²) OR vertical span `>= WALL_MIN_SPAN_Y` (0.40 m)); a floor candidate additionally needs XZ-projected parallelogram area `>= MIN_XZ_AREA2` (1.0e-6). Split at `viewer/src/nav_bake.rs:660-701`; constants at `:190` (`MIN_XZ_AREA2`), `:198` (`WALL_MAX_NY`), `:203` (`WALL_MIN_AREA`), `:207` (`WALL_MIN_SPAN_Y`). Do not confuse these with `free_step(res)` at `:236`, which is the router/baker step-up allowance and has no part in the wall/floor split.

`EFT_NAV_COLLIDERS=0` bakes from render geometry only, for A/B (`viewer/src/nav_bake.rs:153-155`).

Loader degradation: a `colliders.bin` whose length is not a multiple of stride, or a layout missing a named field, logs `pack: colliders.bin unusable (…); nav will bake from render geometry only` and loads zero colliders rather than failing the map (`viewer/src/eftpack.rs:1280-1300`).

---

## 9. `interact_<lv>.json`

Produced by `extraction/unity/extract_interact.py`; one file per level, written only when the level has at least one interactable, and a stale file is DELETED when it has none (`extraction/unity/extract_interact.py:266-279`). Top level is a flat JSON ARRAY, not an object.

Classification is purely typed - zero name matching. A record is a MonoBehaviour whose `m_Script` resolves to `EFT.Interactive.Switch`. It is `kind: "power"` when its trailing PPtr array resolves ENTIRELY to `EFT.Interactive.LampController` (that array IS the light bank it owns); otherwise `kind: "switch"` (`extraction/unity/extract_interact.py:144-147, 191-240`).

### Record fields

| Key | Type | Kind | Meaning |
|---|---|---|---|
| `id` | string | both | `"unity:<level>:mb:<mono path_id>"` |
| `level` | int | both | source level index |
| `switch_go` | int64 | both | the Switch's GameObject `path_id` |
| `group` | string | both | `"<level>:<switch_go>"` - the JOIN KEY the light extractor stamps on each controlled light |
| `world_pos` | [3] float | both | **RAW Unity** world translation, rounded to 4 dp; `null` if unresolvable |
| `label` | string | both | display label; power = raw GO name, switch = built by `build_labels` |
| `kind` | string | both | `"power"` \| `"switch"` |
| `count` | int | both | number of lamp fixtures; 0 for `switch` |
| `controlled_lamp_gos` | [int64] | both | sorted LampController GameObject ids; `[]` for `switch` |
| `controlled_light_gos` | [int64] | both | sorted Unity Light GameObject ids under those lamps; `[]` for `switch` |
| `targets` | [object] | both | class-validated PPtr edges, see below |
| `go_name` | string | switch | raw GameObject name, never overwritten |
| `path` | [string] | switch | GO name path, root → … → own GO |
| `trigger` | string | switch, optional | first payload string, e.g. `"Open_01_722179887"` |
| `link` | string | switch, optional | trailing digit run of `trigger` - the door join key |
| `item_id` | string | switch, optional | 24-char lowercase-hex item template id |
| `verb` | string | switch, optional | in-game interaction verb (`"Use"`, `"Open"`, `"Place"`) |

Each `targets` entry is `{type, target_go, name, world_pos}` - `type` is the full class name, `world_pos` is RAW Unity. The internal `offset` from `decode_scalar_targets` is not emitted.

### Payload decoding

All little-endian; every scan starts at byte 32, past the MonoBehaviour header laid out in §2. Step sizes differ by scanner: `decode_scalar_targets` and `decode_lamp_array` advance a fixed 4 bytes per probe (`extraction/unity/eft_extract_switches.py:138-151, :97-112`), while `payload_strings` jumps to the 4-aligned END of an accepted string (`off = e`, `extraction/unity/extract_interact.py:70`) and steps +4 only on a rejected read (`:72`).

- **Length-prefixed strings** (`read_cstr`, `extraction/unity/extract_interact.py:41-54`): int32 length, then that many UTF-8 bytes, end offset rounded up with `(off + 4 + len + 3) & ~3`. Rejected unless `0 < len <= 256`, the bytes decode, and every character satisfies `31 < ord(c) < 127`. `payload_strings` keeps accepted strings of length ≥ 3, in serialized order.
- **`dissect_strings`** (`:76-87`): `trigger` = `strs[0]`; `link` = the substring after the last `_` when it is all digits and ≥ 6 chars; `item_id` = the first string of exactly 24 chars all in `[0-9a-f]`; `verb` = `strs[-1]` when there are ≥ 2 strings, it is ≤ 12 chars and contains no `_`.
- **Scalar PPtr targets** (`decode_scalar_targets`, `extraction/unity/eft_extract_switches.py:130-152`): at each 4-byte-aligned offset unpack `"<iq"` (int32 FileID at +0, int64 PathID at +4, 12 bytes total, no struct padding under `<`). Accept when `FileID == 0`, the PathID is a MonoBehaviour in this file, and its class is in `SWITCH_TARGET_TYPES`: `ExfiltrationPoint`, `ScavExfiltrationPoint`, `SharedExfiltrationPoint`, `SecretExfiltrations.SecretExfiltrationPoint`, `Door`, `KeycardDoor`, `TransitPoint` - all under namespace `EFT.Interactive`. Deduped by target GameObject; first hit wins.
- **Lamp array** (`decode_lamp_array`, `:90-113`): at each offset read int32 `K`; require `1 <= K <= 400` and `off + 4 + K*12 <= len`; read `K` consecutive `"<iq"` PPtrs; accept only if every FileID is 0 and every target is a `LampController`. Keep the LARGEST such array.

### Label construction

`build_labels` (`extraction/unity/extract_interact.py:90-133`) applies to `switch` records only. `STRIP_PREFIXES = ("INTERACTIVE_", "SBG_", "Node_")`; `GENERIC_SEGMENTS = {"logic", "oo", "interactive", "switch", "node"}` (a segment reducing to one of these is a pure organizer node and carries no meaning). Action = own name with a trailing `_?[Ll]ogic$` stripped, else the trigger stem with trailing `(_\d+)+` removed, else the verb, else `"switch"`. Context = nearest informative ancestors, joined `" · "`, first character upper-cased. On a label collision, colliding records extend context upward one ancestor at a time, up to 3 (loop `range(4)`) - that is what separates three identically-named CPU-panel repairs by their `Room_01`/`Room_02`/`Room_03` organizers. The first `_`-token of the dataset name, lower-cased, is stripped from display tokens. This is display only; `go_name` keeps the raw name.

---

## 10. Folding interactables into `gamedata.json`

`merge_gamedata_interactables(gd_path, dataset_dir, switch_levels)` at `tools/build_map.py:515-622`, called from stage 6 at `:1087`, immediately before the file is copied into the pack. It mutates `gamedata.json` in place, and only when there is something to merge. The whole body is wrapped in `try/except` that prints a note - a failure is silent in the build's exit status.

Steps, in order:

1. **Level selection.** `switch_levels`, or when `None`, every filename matching `interact_(\d+)\.json` in the dataset dir.
2. **Concatenate.** All records from all `interact_<lv>.json` into one list `sw`.
3. **Append gamedata's own point interactables.** For `("card_readers", "card_reader")` and `("dialogs", "dialog")`, each record becomes `{id: "gd:<lv>:<kind>:<i>", level, kind, world_pos, label, count: 0, targets: []}` plus `item_id`/`item_ids` (from `items[].id` or `item_ids`) and `item_name`/`item_names` (from `items[].n`). **`world_pos = [-p[0], p[1], p[2]]`**: gamedata `pos` is already viewer-bridged (`tpos = bridge(M[:3,3])`, `extraction/intel/extract_gamedata.py:994`, `bridge` = `G3 @ p` at `:639-641`), while the `switches` contract is RAW Unity, so the X-flip is UNDONE here to match. Labels use `_disp`: strip `^(INTERACTIVE_|SBG_|Node_)`, insert a space at each lower→upper boundary, `_` → space, capitalise the first character.
4. **`data["switches"] = sw`.**
5. **Tag power-gated extracts.** For every target whose `type` contains `"Exfil"` and whose `name` matches an exfil's `go`, set `exfil["requires_power"] = True`. Surfaces as the viewer's "Requires switch" line.
6. **Wire switch→door edges.** Index `doors[].links` → door. A door's `links` are the trailing digit hashes of its interaction trigger names (`extraction/intel/extract_gamedata.py:1040-1047`); the Switch serializes the same hash in ITS trigger string, so the join is byte-derived on both sides with zero name matching. Each match appends `{"type": "EFT.Interactive.Door", "name": <door id or name>, "world_pos": <door pos>, "via": "trigger-link"}` to the switch's targets and sets `door["controlled_by"] = switch.label`.
7. **Resolve requirement item names** via `tarkov_static.load_static_items(ids=…)`, offline-safe (disk cache or skip), writing `s["item_name"]`.

### The coordinate rule the consumer must honour

Two different spaces now live in one array. `viewer/src/eftpack.rs:945-964`:

- switch `world_pos` → X-negated on load (raw Unity → viewer).
- target `world_pos` → X-negated UNLESS `via == "trigger-link"`, because those carry an already-bridged gamedata door position.
- `exfils[].pos` and `doors[].pos` → **no** flip; they were bridged by `extract_gamedata` (`viewer/src/eftpack.rs:979-991`).

Failure signature: negate a trigger-link target and the door marker lands mirrored across the map's x = 0 plane, hundreds of metres from its switch; skip the negation on a PPtr target and the same thing happens in the other direction. The same double-flip was latent on `exfils` because nothing read the field.

---

## 11. The GameObject-name semantic layer

EFT ships no component field that says "this is Room 3 of Floor 2". That meaning is carried by GameObject NAMES and by the Transform hierarchy - the organizer nodes a level designer used to group things. `extraction/intel/extract_semantics.py:7-8` states the contract: semantics live in names + transforms + colliders, not MonoBehaviour fields; gameplay VALUES (loot tables, spawn percentages, keycards, quests) are external and are not extracted from names at all.

### The actual patterns

Classifier at `extraction/intel/extract_semantics.py:43-53`. First match wins; order is priority. All case-insensitive.

| Category | Regex alternatives |
|---|---|
| `extract` | `exfil`, `extract`, `Saferoom`, `Gates_Rollets`, `Terminal_Entrance`, `Fire_Exit`, `Road_Gate`, `Rollete?_Gate`, `EXIT_`, `ZoneRoad` |
| `loot` | `lootable`, `LootPoint`, `_showcase`, `GunsafeSpawn`, `Weapon_box`, `Weapon_crate`, `scontainer`, `Cashbox`, `cash_register`, `jacket`, `_drawer`, `safe_\d`, `_wallet`, `medbag`, `toolbox`, `ammo_box` |
| `spawn` | `SpawnPoint`, `BotZone`, `PlayerSpawn`, `ScavSpawn`, `Triger_.*Out` |
| `door` | `Inside_Door`, `Door_Metal`, `Door_Wood`, `Keycard`, `_Door_R`, `_Door_L`, `LockBox`, `padlock` |
| `zone` (rooms/floors/corridors) | `Floor[123]_`, `_Corridor`, `_Hall\b`, `_Office\b`, `_Room\b`, `_Stairs\b`, `Parking_Zone`, `Basement`, `Atrium` |

`EXCLUDE = (_COLLIDER|_LOD[1-9]|_SHADOW|decal|Particle|VFX|_proxy)` is applied FIRST and vetoes everything. `FLOOR_RE = Floor\s?([0-3])` pulls a floor number out of any matched name into the record's `floor` field.

Output records are `{name, p: [x,y,z], floor, lv}` with `p = G3 @ world_translation` rounded to 2 dp, deduped on `(category, round(x,1), round(y,1), round(z,1))`.

Door-panel names, hand-rolled without a regex dependency at `viewer/src/nav_bake.rs:283-346`: match on substring `inside_door`, `door_metal`, `door_wood`, `_door_left`, `_door_right`, `glass_door`, `rollet`, `shutter`; plus `_door_l`/`_door_r` at a word boundary; plus `gate` as a whole word. Vetoed by `trailer`, `truck`, `van`, `lovlo`, `tarcola`, `transformator`, `locker`, `fridge`, `microwave`, `oven`, `cabinet`, `lockbox`, `padlock`, `wagon`, `gaz`, `kamaz`, `ural`.

Organizer-node names at `extraction/unity/extract_interact.py:27-38`: prefixes `INTERACTIVE_`, `SBG_`, `Node_` are decoration; segments reducing to `logic`, `oo`, `interactive`, `switch`, `node` are pure grouping nodes with no human meaning.

Collider-side names: `go` (own name) and `root` (topmost named ancestor, `extraction/unity/eft_extract_colliders.py:211-222`) in `colliders.json`. On the instance side the pack keeps a root-name table in `manifest.roots` indexed by `Instance.rootId`, plus the folded ancestry `par`/`par2`/`lv` (`eft_pipeline/assemble_bevy.py:106-122`) - a stable id join, not a name join. Colliders have no equivalent; their names do not survive into the pack.

### Names are recycled - a name search alone is not a classifier

Three defences in this repo exist precisely because the name space is ambiguous, and each is evidence of a specific collision:

- **`door_skip`** exists because `gate` and `door` appear inside vehicle and appliance names. Without it a fridge door and a truck's rear gate become passable holes in the nav grid.
- **`DOOR_FOOTPRINT_MAX = 1.5 m`** (`viewer/src/nav_bake.rs:242-246`): a door tag only opens a passable hole when the INSTANCE footprint, measured as the smaller horizontal span of the 8 transformed AABB corners, is door-panel sized. A `gate`-named wall-wide mesh keeps blocking. Without the cap a name on a large shutter opens a wall-wide gap the player cannot actually pass.
- **The `zone` + `has_mesh` veto** (`extraction/intel/extract_semantics.py:164-165`): a `zone`-classified GameObject that owns a MeshFilter/MeshRenderer is discarded - it is a prop NAMED after an area, not the area group itself.
- **`build_labels`' collision loop** exists because identically-named interactables recur; the only thing that distinguishes them is the `Room_01`/`Room_02`/`Room_03` organizer above them.
- **`extract_semantics`' positional dedup** exists because the same name recurs at many positions in a level.

The reliable discriminators, in order of preference: the Unity LAYER (`DoorLowPolyCollider` needs no name matching at all - `viewer/src/nav_bake.rs:661-666`), the typed component class resolved through `MonoScript.m_ClassName`, and the class-validated PPtr edge. Use names for DISPLAY and as a last-resort heuristic, and always pair a name rule with a structural test (footprint, mesh ownership, layer).

---

## 12. Invariants and failure signatures

| Invariant | Failure signature |
|---|---|
| `CDT.itemsize == 96`, stride read from `manifest.collider.stride`, fields resolved by NAME | Reading a hardcoded 88-byte stride walks the buffer: colliders progressively drift, then the tail is garbage. The loader instead errors on `len % stride != 0` and bakes without colliders |
| Record count comes from the file length, not `manifest.colliderCount` | Trusting the manifest field: nothing validates it, so a truncated or over-long `colliders.bin` is read to the field's count and either drops the tail or reads past the parsed records |
| `apply_global` runs exactly ONCE per collider | Twice: the map mirrors back to Unity handedness and every collider sits on the wrong side of x = 0, while the render geometry does not |
| `center` is G3-multiplied; `shape` is not | Skip the centre flip and every primitive mirrors about its own pivot, displaced by `2·c.x` times the affine's X column. Flip `shape` too and box sizes stay right by symmetry but capsule direction indices scramble |
| `G3` is a signed permutation | A rotational global matrix makes box/capsule parameterisation inexpressible; the assembler exits rather than emit it |
| Collider flag bits are a distinct space from instance flag bits | Reading the instance legend against colliders: every trigger looks mirrored, `ny` inverts, ceilings become floors, phantom walkable surfaces appear above every trigger volume |
| Primitives wound outward | Top face reads as ceiling, underside as floor: a walkable surface invented in mid-air |
| `collider_meshes.bin` indices are mesh-LOCAL | Treating them as file-global indexes into another mesh's vertices: triangles stretch across the map, and `min_y`/`max_y` blow out so the nav grid's height range covers kilometres |
| MeshCollider OBJ verts are already G-applied | X-negating them again mirrors each collider shell locally: doorways on the wrong side of their frames, staircases running backwards |
| OBJ filenames are Unicode, not ASCII | Re-deriving a filename with an ASCII-only sanitiser rewrites e.g. Cyrillic `С` to `_`, the join against `manifest.colliderMeshes[i].name` misses, and that mesh collider silently has no geometry |
| `interact` `world_pos` is RAW Unity; gamedata `pos` is bridged | Markers land mirrored across x = 0; the `via: "trigger-link"` exception makes it happen to some targets and not others in the same list |
| `_obj_complete` gates OBJ reuse | A killed collider pass leaves NUL-filled OBJs; `load_obj` parses them without raising, `len(V) == 0`, the MeshCollider never reaches `colliders.bin`, and the nav bake never learns the wall is there |
| Absent `nav_area` ≠ area 0 | Treating it as Walkable overrides authored Not-Walkable/Danger regions with default cost |

---

## 13. Old patterns

- **Pre-collider nav bake.** The nav grid was built from render geometry alone. `EFT_NAV_COLLIDERS=0` reproduces that input for A/B comparison (`viewer/src/nav_bake.rs:153-155`). A pack without `manifest.collider` degrades to it automatically.
- **`switches_<lv>.json`.** `extraction/unity/eft_extract_switches.py` writes a power-lever-only sidecar. `interact_<lv>.json` is its superset and its power records are format-compatible - same `group` join key `"<lv>:<GO>"`, so the light extractor's group tagging works against either.
- **Packs without `kind`.** A `switches` record with no `kind` is treated as `"power"` (`viewer/src/eftpack.rs:965`).
- **Packs with scalar `item_name`.** Folded into `item_names` at parse when the array is empty (`viewer/src/eftpack.rs:957-960`).
- **Packs without `layerNames`.** `Pack::layer_name` returns `""`, which matches nothing in `NAV_COLLIDER_LAYERS`, so every collider is skipped as off-layer. The bake still completes on render geometry.
- **The exporter's own completeness check.** This exporter once carried a private `not exists or getsize == 0` test that the mesh exporter had already been fixed away from; the shared `fileguards.obj_complete` replaced it (`extraction/unity/eft_extract_colliders.py:61-69`).