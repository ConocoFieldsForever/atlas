## Contents

1. [Scope and module map](#1-scope-and-module-map)
2. [Locating the rig, the parts and the clips in the bundles](#2-locating-the-rig-the-parts-and-the-clips-in-the-bundles)
3. [Coordinate frame, units, and the handedness rule](#3-coordinate-frame-units-and-the-handedness-rule)
4. [The skeleton](#4-the-skeleton)
5. [Skinning: bone remap, weights, bindposes](#5-skinning-bone-remap-weights-bindposes)
6. [Materials and textures](#6-materials-and-textures)
7. [Animation clips: Unity curve decode](#7-animation-clips-unity-curve-decode)
8. [Root motion and the derived forward axis](#8-root-motion-and-the-derived-forward-axis)
9. [The animator controller graph](#9-the-animator-controller-graph)
10. [Equipment and attachment binding](#10-equipment-and-attachment-binding)
11. [The .eftchar container: exact byte layout](#11-the-eftchar-container-exact-byte-layout)
12. [What is dropped](#12-what-is-dropped)
13. [Consumer-side math: pose, blend, skin](#13-consumer-side-math-pose-blend-skin)
14. [Invariants and their failure signatures](#14-invariants-and-their-failure-signatures)
15. [Old patterns](#15-old-patterns)

---

## 1. Scope and module map

The character subsystem turns EFT's one shared biped, a character's skinned part prefabs, and its Mecanim animation set into a `.eftchar` pack. The pack is self-describing: `manifest.json` declares every stride, byte offset, count and convention, and the consumer reads the layout from the manifest rather than hardcoding it.

| file | role |
|---|---|
| `extraction/characters/coords.py` | the `G3` conjugation and the UV flip, and nothing else |
| `extraction/characters/unity_bind.py` | transform-path CRC-32, curve-index ↔ binding walk, digest self-check |
| `extraction/characters/skeleton.py` | `skeleton.bundle` → canonical bone table |
| `extraction/characters/skin.py` | part bundle → meshes, bone remap, inverse bindposes, materials, textures, rigid attachments |
| `extraction/characters/clips.py` | `AnimationClip` → resampled per-bone tracks |
| `extraction/characters/controller.py` | `AnimatorController` + `PlayerStateContainer` → state table |
| `extraction/characters/validate.py` | anatomical validation of composed poses |
| `extraction/characters/pack.py` | writes `manifest.json` + `skin.bin` + `anim.bin` + `textures/` |
| `extraction/characters/appearance.py` | rolls WHICH prefabs a bot wears, from the game's own weighted tables |
| `extraction/characters/build_character.py` | CLI and ordering |
| `extraction/characters/unity_deps.py` | CAB → bundle index for cross-bundle PPtr resolution |
| `viewer/src/character/pack.rs` | `.eftchar` loader (Rust) |
| `viewer/src/character/rig.rs` | pack → entity hierarchy + skinned draws |
| `viewer/src/character/anim.rs` | clip sampling, blend-tree evaluation, pose accumulation |
| `viewer/src/character/drive.rs`, `viewer/src/npc.rs` | parameter synthesis and state selection |

Build ordering is load-bearing (`extraction/characters/build_character.py:12-15`): skeleton first because every later join keys off its path strings; controller before clips because the controller's `m_TOS` self-validates the path digest in one second instead of after thousands of unbound tracks.

---

## 2. Locating the rig, the parts and the clips in the bundles

Everything lives under `<game>/EscapeFromTarkov_Data/StreamingAssets/Windows/`. The character root is `assets/content/characters/`.

- **Rig**: `character/skeleton.bundle`. One `Transform` hierarchy, exactly one root. Every character in the game binds this rig (`extraction/characters/skeleton.py:3`).
- **Body parts**: `character/prefabs/<part>.bundle`, e.g. `top_boss_tagilla.bundle`. Each is a `SkinnedMeshRenderer` + `LODGroup` with a `Skin` MonoBehaviour carrying `_bonePaths`.
- **Controllers**: `controllers/animationcontrollers/<stem>botanimcontroller.bundle` and `controllers/player_anim_controller.bundle`. `extraction/characters/appearance.py:15-16` names the stems `base`, `boar`, `tagilla`; `controller_for` discovers stems by directory listing at runtime, so the shipped set is unverified.
- **Shared clips**: `animations/character_animations.bundle`.
- **Root-motion tables**: `rootmotiontable/<stem>botrootmotiontable.bundle` (read as a bundle path in the registry; the extractor does not currently parse it).

Two bundle-loading rules that must be respected or half the data silently reads as empty:

1. **The controller and the shared animation bundles must be loaded into ONE `UnityPy` environment** (`extraction/characters/build_character.py:121-131`). `m_AnimationClips` entries with non-zero `m_FileID` are external PPtrs; UnityPy can only follow those into a file it already holds. Measured on `TagillaBotAnimController`: 847 declared clip slots. How those split between the controller's own bundle and external ones (430 in its own bundle, per the `_animationsComment` in `extraction/characters/characters.json`) is unverified. A missing dependency does not error, it yields an empty clip name.
2. **A bundle is recognised by its header, never its filename.** `extraction/characters/unity_deps.py:35` accepts `UnityFS`, `UnityWeb`, `UnityRaw`, `UnityArchive`. The counts recorded at `unity_deps.py:14-17` - 1,160 bundles shipping with no extension at all, and 2,295 CABs missed by an index that globbed `*.bundle` - are unverified without a game install.

Cross-bundle dependency resolution for equipment prefabs goes through a CAB→bundle map (`extraction/characters/unity_deps.py:118`): the container's `AssetBundle.m_Dependencies` lists CAB names, the index maps each to a file, and those files are loaded into the same environment. Only the objects the container itself introduced are baked - dependency bundles carry unrelated assets.

**Who a character IS is derived, not authored.** `extraction/characters/appearance.py:111` rolls the bot type's weighted `appearance` table (slots `body`, `feet`, `hands`, `head`) with `random.Random(f"appearance:{bot_type}:{seed}")`, resolves each id through `customization.json` to a prefab path, and returns a build spec. `hands` resolves to the first-person prefabs and is tagged `view: "first"`; the other three are `"third"` (`extraction/characters/appearance.py:48`).

---

## 3. Coordinate frame, units, and the handedness rule

Unity world → viewer world is a single reflection, identical to the map pipeline's:

```
G3 = diag(-1, 1, 1)        det(G3) = -1        G3⁻¹ = G3
```

Units are metres, up axis is +Y, quaternions are stored **xyzw**. All binary data is **little-endian** (`struct` format strings use `<`; numpy `tobytes()` on x86; the Rust loader uses `f32::from_le_bytes` / `u32::from_le_bytes` / `u16::from_le_bytes`).

Per-datum rules, all in `extraction/characters/coords.py`:

| datum | transform | anchor |
|---|---|---|
| point / position key | `(x,y,z) → (-x, y, z)` | `coords.py:34` |
| normal | same as point (`(G⁻¹)ᵀ = G` for diagonal `G`), stays unit | `coords.py:46` |
| tangent `(x,y,z,w)` | `(-x, y, z, -w)` - the handedness sign flips | `coords.py:51` |
| rotation `(x,y,z,w)` | `(x, -y, -z, w)` | `coords.py:74` |
| scale | unchanged (diagonal ∘ diagonal) | `coords.py:18` |
| 4×4 affine (inverse bindpose) | `M' = G M G⁻¹` | `coords.py:87` |
| triangle winding | `(a,b,c) → (a,c,b)` | `coords.py:97` |
| UV | `v → 1 − v` (Unity origin bottom-left) | `coords.py:60` |

The quaternion rule is the pseudovector rule, not a typo. A rotation conjugated by a reflection is still a rotation (`det(GRG⁻¹) = det(R) = 1`), but the rotation axis is an axial vector so it picks up `det(G)`: `a → det(G)·(G a) = −(G a)`, which for `diag(-1,1,1)` is `(x, −y, −z)`; the angle is unchanged, hence `w` is untouched.

**The conjugation is applied once, per datum, and it telescopes through the hierarchy.** Each bone's LOCAL matrix is conjugated independently. Because `(G L₁ G⁻¹)(G L₂ G⁻¹)…(G Lₙ G⁻¹) = G (L₁L₂…Lₙ) G⁻¹`, composing conjugated locals gives exactly the conjugated world matrix. There is no separate "apply handedness to the whole rig" step, and adding one double-conjugates back to Unity space.

Skinning survives the same way. With `p' = G p`, `W' = G W G⁻¹`, `B' = G B G⁻¹`: `W' B' p' = G W G⁻¹ · G B G⁻¹ · G p = G (W B p)`.

**Matrix convention is COLUMN-vector**, in the character pipeline and the map pipeline alike. `extraction/characters/skeleton.py:82` builds `M` with the rotation in `M[:3,:3]`, per-axis scale multiplied into the COLUMNS (`r * s[None, :]`), and translation in `M[:3, 3]`; composition is `parent @ local` (`skeleton.py:67`). A point transforms as `p' = M p`. Inverse bindposes are flattened **row-major** into the manifest (element order `m00 m01 m02 m03 m10 …`; measured: elements 12..15 of every emitted row are `0,0,0,1`). The only difference on the map side is storage packing, not math: `viewer/src/eftpack.rs:539-550` reconstructs an instance affine from a 3×4 row-major block as `Mat3::from_cols(Vec3::new(a[0],a[4],a[8]), Vec3::new(a[1],a[5],a[9]), Vec3::new(a[2],a[6],a[10]))` with translation `(a[3],a[7],a[11])` - the same column-vector matrix, stored 3×4 instead of 4×4.

The character path draws through the ordinary PBR pipeline with back-face culling ON, so the mirror is absorbed by **reversing the index buffer**. The map's `gpu_driven` path instead draws double-sided with a cofactor normal matrix. Both are correct; do not mix them on one asset.

---

## 4. The skeleton

`extraction/characters/skeleton.py:107` reads every `Transform` and `GameObject` typetree in `skeleton.bundle`. A bone's name is its owning GameObject's `m_Name`.

- **Root**: the single transform whose `m_Father.m_PathID == 0`. More or fewer than one is a hard error (`skeleton.py:132`).
- **Order**: depth-first from the root, children in `m_Children` order. This is stable across runs and guarantees `parents[i] < i`, so a consumer computes world matrices in one forward pass with no sort (`skeleton.py:191`, re-asserted in `viewer/src/character/pack.rs:506`).
- **Paths** are relative to the rig ROOT and exclude it: the root's own path is `""`, its first child's path is just that child's name (`skeleton.py:144-153`). This matters because `Skin._bonePaths` and Mecanim clip bindings are both relative to the GameObject carrying the Animator, and because Unity denotes "my root transform" with `CRC32("") == 0`.
- **Per-bone data**: `m_LocalPosition` through `coords.point`, `m_LocalRotation` (xyzw) through `coords.quat`, `m_LocalScale` verbatim. Stored as `float32` arrays of shape `(N,3)`, `(N,4)`, `(N,3)`.

Build-time assertions: every transform must be reached by the walk (`skeleton.py:183`); bone names must be unique (`skeleton.py:188`); `parents[i] < i` (`skeleton.py:192`); path hashes must not collide (`skeleton.py:197` → `unity_bind.build_hash_map`).

**Measured rig (Tagilla and player packs, EFT live build): 79 bones.** Index 0 `Skeleton` (root, parent −1, path `""`), 1 `Root_Joint`, 2 `Base HumanPelvis`, 24 `Base HumanHead`, 68 `Weapon_root`, 78 `Camera_animated_3rd`.

Indices 3..67 hold the biped chains (`Base HumanL/RThigh1/2`, `Calf`, `Foot`, `Toe`, `Spine1..3`, `Neck`, `Head`, `Ribcage`, `L/RCollarbone`, `Upperarm`, `Forearm1..3`, `Palm`, `Digit11..53`) plus SIX gear nodes - `Base HumanGear1` 14, `Gear2` 15, `Gear3` 16, `Gear4` 17, `Gear4_1` 18, `Gear5` 19 - and `Base HumanBackpack` 22, and nothing else. Every rig extra sits ABOVE `Weapon_root`: `Bend_Goal_Left` 69, `Bend_Goal_Right` 70, `IK_S_LPalm` 71, `IK_S_RPalm` 72, `LCollarbone_anim` 73, `RCollarbone_anim` 74, `weapon_holster` 75, `weapon_holster1` 76, `Weapon_root_3rd_anim` 77, `Camera_animated_3rd` 78.

Measured on the Tagilla pack, maximum path depth is **15 segments** and the digit chains occupy depths 13–15. The longest path is:

```
Root_Joint/Base HumanPelvis/Base HumanSpine1/Base HumanSpine2/Base HumanSpine3/Base HumanRibcage/Base HumanLCollarbone/Base HumanLUpperarm/Base HumanLForearm1/Base HumanLForearm2/Base HumanLForearm3/Base HumanLPalm/Base HumanLDigit11/Base HumanLDigit12/Base HumanLDigit13
```

The suffix index at `skin.py:171` is sized from a rough per-path segment estimate, not from this depth figure.

`Skeleton.by_hash` (`skeleton.py:54`) is `{CRC32(path): index}` and is THE join key for clip bindings.

---

## 5. Skinning: bone remap, weights, bindposes

`extraction/characters/skin.py:346` reads one part bundle. Geometry comes out of `UnityPy.helpers.MeshHelper.MeshHandler` after `process()`.

### 5.1 Mesh-bone slot → rig bone

A part binds only the bones it needs (measured on Tagilla: the top binds 48 of 79, the pants 12). Two independent sources describe the mapping and both are read (`skin.py:210`):

- `Skin._bonePaths[i]` - path strings, per renderer component.
- `Mesh.m_BoneNameHashes[i]` - `CRC32(path)`, parallel to `Mesh.m_BindPose`.

Resolution order for a path: exact match against `Skeleton.paths`, then unique **root-relative suffix**. First-person hand prefabs root their paths one level down (`Base HumanPelvis/…` where the canonical rig says `Root_Joint/Base HumanPelvis/…`); `Wild_Body_1_firstHands` binds exactly 40 bones in the built `player_0`, `pmcusec_0` and `bosskilla_0` packs, and the stronger claim that all 40 of its paths are exact suffixes of canonical paths is unverified without the hands bundle. A suffix matching two different bones is a hard error, never a silent pick (`skin.py:236`, marker constant `AMBIGUOUS = -1` at `skin.py:162`). Hashes get the same treatment through a suffix-hash index (`skin.py:189`).

If NONE of the paths resolve, the mesh binds a genuinely different skeleton and is SKIPPED (`ForeignRigError`, `skin.py:205`). If some resolve and some do not, that is corruption and the build fails.

**When the two sources disagree, the mesh's own `m_BoneNameHashes` wins** (`skin.py:280-299`). The vertex bone indices address the mesh's bind-pose array, which is parallel to the hash list, so the hashes are authoritative by construction. Disagreements in shipped assets take both forms: `usec_upper_commando` is the same length but shifted two slots; `Top_BOSS_Killa_base` is a different length outright (49 vs 48), which is why a naive `zip()` diff reported nothing. The built packs are consistent with those figures (`Tshirt_usec_Commando` binds 49 bones, `Top_BOSS_Killa_base_lod0` binds 48), but the `_bonePaths`-versus-hash delta itself is unverified without the source bundles.

### 5.2 Influences and weights

Fixed **4 influences per vertex**. `handler.m_BoneIndices` and `handler.m_BoneWeights` are reshaped to `(V, ·)` and truncated to the first 4 columns (`skin.py:472-473`).

```
global_joints = remap[clip(local_joints, 0, n_slots-1)]      # local slot -> rig index
global_joints = where(weight > 0, global_joints, 0)          # zero-weight influence -> root
wsum          = weights.sum(axis=1, keepdims=True)
weights       = weights / wsum   where wsum > 1e-8           # partition of unity
weights[wsum <= 1e-8, 0] = 1.0                               # degenerate -> pinned to bone 0
```

(`skin.py:491-504`.) Any vertex referencing a slot ≥ `len(remap)` is a hard error (`skin.py:486`). Joint indices are stored as `u16`, so the rig may not exceed 65,536 bones.

### 5.3 Inverse bindposes

`Mesh.m_BindPose` is a list of `Matrix4x4f` typetrees with fields `e{row}{col}`, read into a 4×4 with translation in column 3 (`skin.py:131`). The count must equal the number of bound slots or the part is rejected (`skin.py:508`).

Each mesh gets a **rig-sized** table: `(boneCount, 4, 4) float32`, identity everywhere the mesh does not bind, and `coords.matrix4(B_slot)` at `remap[slot]` (`skin.py:512-514`). Cost is ~5 KB per mesh (79 × 64 bytes). The payoff is that every mesh of every part shares ONE joint list in the consumer, so assembling a character is "spawn the rig once, attach N meshes" with no per-part bone mapping at runtime.

### 5.4 Submeshes and indices

`Mesh.m_IndexFormat == 0` means 16-bit indices, anything else 32-bit; the per-submesh `firstByte` is divided by that size to get an index offset (`skin.py:534`). Per submesh:

```
seg = all_indices[firstByte/index_size : firstByte/index_size + indexCount] + baseVertex
seg = flip_winding(seg)                # (a,b,c) -> (a,c,b)
```

Submesh ranges are concatenated in order and recorded as `(material, indexStart, indexCount)` relative to the mesh's own index block.

**LOD** is parsed from the mesh name with `re.compile(r"_lod(\d+)")` searched ANYWHERE in the lowered name, not anchored at the end (`skin.py:140`): character parts are `Top_..._lod0` but equipment is `item_..._lod1_base`, and a suffix-only match let every equipment LOD through and drew the item twice.

---

## 6. Materials and textures

Materials come from `Material.m_SavedProperties` (`skin.py:390-421`). `m_TexEnvs` entries are `(slotName, {m_Texture: PPtr, …})`; the texture's `m_Name` becomes `textures/<name>.png` and the slot name is kept verbatim as the shader names it (`skin.py:393-409`). Measured across the built `tagilla`, `player_0`, `pmcusec_0` and `bosskilla_0` packs, the only slots emitted are `_MainTex`, `_BumpMap`, `_SpecMap`, and `_SpecTex` - the last appearing only on `item_equipment_facecover_welding_glass`. `_MarkTex`, `_EnvTex` and `_Cube` belong to the weapon and kit pipelines (`viewer/src/character/weapon.rs:218-228`, `out/weapons/*`, `out/kits/_parts/*`) and appear in no `.eftchar` pack. `m_Floats` and `m_Colors` are carried through raw. Pack-wide material indices are assigned with a `material_base` offset per part so submesh references stay global.

**Normal maps are repacked at write time** (`extraction/characters/pack.py:266`). Unity ships DXT5nm / BC5: X in ALPHA, Y in green, red pinned to ~1.0, Z reconstructed. The repack triggers on measurement, not filename - `if r.std() > 0.02: return img` passes an already-standard map through unchanged. Otherwise:

```
x = a*2 - 1;  y = g*2 - 1;  z = sqrt(clamp(1 - x² - y², 0, 1))
out = ((x,y,z) + 1) * 0.5
```

The filename check that selects candidates is `name.lower().endswith(("_n", "_normal", "_nrm"))` (`pack.py:89`).

`_SpecMap` is emitted but deliberately NOT bound by the consumer (`viewer/src/character/rig.rs:174-187`): it is a GLOSS map (high = shiny), the inverse of a roughness map, and binding it to occlusion crushed character ambient to a fifth with hard seams. It needs a `1 − x` pass at extraction before it is usable.

---

## 7. Animation clips: Unity curve decode

EFT's character clips are **generic**, not humanoid: bindings are `typeID 4` (Transform) with attribute 1/2/3/4. There is no muscle-space retargeting to undo.

### 7.1 Bindings and the curve index space

`m_ClipBindingConstant.genericBindings[]` entries carry `path` (a `uint32` CRC-32 of the transform path), `typeID`, and `attribute`. Curves live in ONE flat index space; each binding consumes a contiguous run whose width depends on the attribute (`extraction/characters/unity_bind.py:37`):

| attribute | meaning | curves |
|---|---|---|
| 1 | `ATTR_POSITION` | 3 |
| 2 | `ATTR_ROTATION` (quaternion) | 4 |
| 3 | `ATTR_SCALE` | 3 |
| 4 | `ATTR_EULER` (degrees) | 3 |

Any non-Transform binding is width 1 (`unity_bind.py:121`). `walk_bindings` (`unity_bind.py:128`) walks the list accumulating widths, assigning each binding `curve_start` and `curve_count` - the exact inverse of Unity's own `ClipBindingConstant::FindBinding`.

`path_hash` is `zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF`, with `""` mapping to `0` (`unity_bind.py:52`). The digest is ASSERTED, not assumed: `validate_hash_fn` (`unity_bind.py:74`) checks it against the controller's `m_TOS` (a hash→string debug table Unity ships) and raises unless at least 8 entries agree. A hit rate of 1.0 is not expected - `m_TOS` mixes true transform paths with state-machine labels like `Base Layer.JUMP.Fall`, which use a different digest.

### 7.2 The three concurrent encodings

The clip container is `m_MuscleClip.m_Clip` (unwrap UnityPy's extra nested `"data"` key, `clips.py:328`). Its curves concatenate in this exact order:

```
[ m_StreamedClip curves ][ m_DenseClip curves ][ m_ConstantClip curves ]
```

**STREAMED** (`clips.py:100`). `m_StreamedClip.data` is a list of `uint32` words; reinterpret as a little-endian byte stream of variable-length frames:

```
offset  size  type   field
0       4     f32    time
4       4     i32    keyCount
8       20*keyCount  keys
  per key (stride 20):
    +0    4   i32    curveIndex
    +4   16   f32[4] c0, c1, c2, c3
```

Unity brackets the real frames with sentinels at non-finite `time`; those frames are parsed but their keys are discarded (`clips.py:123`). A `keyCount` outside `[0, 2²⁰]` or a short read terminates the walk. Keys are grouped by `curveIndex` and sorted by time. The stored coefficients are the CUBIC of the segment STARTING at that key:

```
v(dt) = c0·dt³ + c1·dt² + c2·dt + c3        dt = t − keyTime
```

Evaluated directly (`clips.py:133`), with the segment chosen as the last key at or before `t`, clamped to the first key. Converting to Hermite tangents and back only loses precision.

**DENSE** (`clips.py:147`). `m_SampleArray` is frame-major: `sample[frame * m_CurveCount + curve]`. Frame `f` sits at `m_BeginTime + f / m_SampleRate`. Reconstruction is linear between neighbouring frames, clamped at both ends. A single-frame dense clip is constant.

**CONSTANT**. `m_ConstantClip.data[i]` is one `f32` for all time.

`_CurveSet.evaluate` (`clips.py:185`) resolves an absolute curve index by subtracting the streamed and then the dense counts. `curveCount` from `m_StreamedClip`, `m_CurveCount` from `m_DenseClip`, and `len(m_ConstantClip.data)` must sum to the width claimed by the binding list, or the build fails (`clips.py:334`). That assertion is the whole reason to trust the output: a mis-sliced curve space produces animation that looks plausible while being wrong.

### 7.3 Resampling to a uniform grid

```
start       = m_MuscleClip.m_StartTime
stop        = m_MuscleClip.m_StopTime
duration    = max(0, stop − start)
rate        = m_DenseClip.m_SampleRate  or  AnimationClip.m_SampleRate  or  30.0
frameCount  = clamp(round(duration * rate) + 1, 1, 8192)
times[f]    = clamp(start + f / rate, start, stop)
loop        = m_MuscleClip.m_LoopTime
```

(`clips.py:344-357`, `MAX_FRAMES = 8192` at `clips.py:50`, `DEFAULT_SAMPLE_RATE = 30.0` at `clips.py:48`.) All three encodings are baked onto this one grid, so the consumer ships exactly one sampler and adding a character can never introduce a new decode path. Because `frameCount` is `round(duration·rate) + 1`, the LAST frame sits exactly at `duration` - the consumer's index math depends on this.

### 7.4 Rotation channels

Quaternion curves (attribute 2) are `Transform.localRotation` values: `coords.quats` and nothing else (`clips.py:396`). They are then made **sign-continuous** - walk frames forward, negate the whole quaternion whenever `dot(q[i], q[i-1]) < 0` (`clips.py:289`). Component-wise curve storage legally produces `q` and `−q` on adjacent frames, identical rotations that a naive nlerp walks the long way between.

Euler curves (attribute 4) are degrees, composed as `Rz(z) @ Ry(y) @ Rx(x)` - the intrinsic sequence X, then Y, then Z, i.e. **Maya's default XYZ rotate order**, matching a Maya-authored rig with `Base Human*` bone names (`clips.py:214-240`). This is NOT `Quaternion.Euler`'s ZXY order. The result is converted to a quaternion with a branch-per-largest-diagonal matrix→quaternion (`clips.py:243`), returned in RAW Unity space, and only then passed through `coords.quats` - so exactly one place knows about the mirror. **Euler only wins where the clip supplied no quaternion curve for that bone** (`clips.py:405-408`). How many shipped clips actually reach that branch is unverified; treat it as the less-exercised path.

### 7.5 Output shape

A `BoneTrack` (`clips.py:57`) holds `position (F,3) f32`, `rotation (F,4) f32 xyzw`, `scale (F,3) f32`, each independently `None` when the clip does not drive that channel - in which case the bind-pose value stands. Tracks are emitted sorted by bone index. Bindings that do not resolve to a rig bone (Animator float parameters, unbound properties) are counted and reported, never silently dropped (`clips.py:76`). Measured on the Tagilla pack: 117 clips, 78 distinct bones animated (every non-root bone), `unresolvedBindings` between 7 and 185 per clip, 1,496 in total.

---

## 8. Root motion and the derived forward axis

Root motion is **found, not assumed** (`clips.py:412-431`). Walk the tracks in bone order; the first (hence lowest-indexed, nearest the rig root) bone whose animated position spans more than `ROOT_MOTION_EPS = 0.02 m` on any axis is the carrier. Every other bone's position curve merely restates its constant bind offset, so the threshold separates them cleanly.

```
root_motion[f] = position[f] − position[0]          # (F,3), viewer space
position[f]    = position[0]  for all f             # the bone is pinned
average_speed  = root_motion[-1] / duration         # m/s, viewer space
```

The travel is STRIPPED off the bone track into its own channel so no consumer can double-apply it: walk physics already moves the character through the world, and leaving 4 m of forward travel in the skeleton slides the body out from under the camera.

Measured on the Tagilla pack: 109 of 117 clips carry root motion; the carrier is bone 1 (`Root_Joint`) in 100 clips, bone 2 (`Base HumanPelvis`) in 6, bone 14 (`Base HumanGear1`) in 3. `walk_aim_0` travels 2.502 m/s over 1.600 s at 30 Hz (49 frames).

**Facing is derived, never authored** (`extraction/characters/build_character.py:455-472`). Take the forward-walk clip (default name `walk_aim_0`, overridable per character), zero its Y component, normalise, and write it as `characterForward`. Measured for Tagilla: `[0, 0, 1]`, flagged `characterForwardDerived: true`. The consumer aligns character-forward to movement direction with no magic 180° (`viewer/src/character/drive.rs:326`). `characterForwardDerived: false` means the value fell back to `+Z` and must not be trusted.

---

## 9. The animator controller graph

`extraction/characters/controller.py:238` reads an `AnimatorController` typetree. This is a DATA extraction, not a reimplementation of Unity's Animator. UnityPy nests serialized structs under an extra `"data"` key, so every access goes through `_d` (`controller.py:32`).

**Parameters** come from `m_Controller.m_Values.m_ValueArray`; names resolve through `m_TOS` by `m_ID`; defaults come from `m_DefaultValues.m_Float/Int/BoolValues` indexed by `m_Index`. Type ids are `{1: float, 3: int, 4: bool, 5: trigger, 9: bool}` (`controller.py:29`). Measured on `TagillaBotAnimController`: 100 parameters, including `Direct_X`, `Direct_Y`, `Speed`, `Level`, `Sprint`, `Tilt`, `Aim_angle`.

**Layers** come from `m_LayerArray`. Blending is `{0: override, 1: additive}`. Critically, several layers legitimately share one state machine, and **the FIRST layer to reference a machine owns it**; later ones are synced views (`controller.py:288-295`). Building the map with a comprehension instead attributes every base-layer state to the LAST synced layer. Measured (Tagilla, 13 layers): layers 0 (`Base Layer`), 3 (`Sync_SprintHands`) and 10 (`TagillaSyncLayerForRegularOperations`) all point at state machine 0; only layer 0 owns it.

**States** come from each `m_StateMachineArray[i].m_StateConstantArray[]`. `m_FullPathID` and `m_NameID` resolve through `m_TOS` to `"Base Layer.StateMachine_Move.MOVE"` and `"MOVE"`. Preserved: `m_Speed`, `m_Loop`, `m_Mirror`, `m_CycleOffset`, `m_SpeedParamID`.

**Blend trees NEST, and they hang off the STATE.** `m_BlendTreeConstantArray` is on the state; `m_BlendTreeConstantIndexArray` has one entry per synchronized-layer slot indexing into it, with `−1` meaning "this state contributes nothing on that layer" (`controller.py:383`). Measured: Tagilla's `MOVE` has `trees` of length 3.

Node parsing (`controller.py:408`) is recursive with a `seen` cycle guard. A node with no `m_ChildIndices` is a LEAF regardless of its `m_BlendType` byte; `m_ClipID == 0xFFFFFFFF` means "no clip" and is emitted as `−1`. Blend types: `{0: 1d, 1: 2d_simple_directional, 2: 2d_freeform_directional, 3: 2d_freeform_cartesian, 4: direct}` (`controller.py:48`). A child's position in its PARENT's blend space is stored ON the child: `threshold` from `m_Blend1dData.m_ChildThresholdArray[n]`, `position` from `m_Blend2dData.m_ChildPositionArray[n]`.

Measured on the Tagilla pack: `trees[0]`, `Base Layer.StateMachine_Move.MOVE`, is a `2d_freeform_cartesian` on `(Direct_X, Direct_Y)` with 9 children. Eight of them - at positions `(0,1)`, `(1,1)`, `(1,0)`, `(1,−1)`, `(0,−1)`, `(−1,−1)`, `(−1,0)`, `(−1,1)` - are themselves `2d_freeform_cartesian` nodes on `(Speed, Level)` with 6 children each; child index 8 is a `1d` node on parameter `Direct` with 4 children. The whole tree is 86 nodes and 72 clip leaves. `trees[2]` is a `2d_simple_directional` on `(Direct_X, Direct_Y)` with 1 child. A flat "root's children" read returns nine nodes with no clips at all.

**Transitions** come from `m_TransitionConstantArray`: destination (via `m_DestinationState` through `m_TOS`, reduced to its LEAF name), `m_TransitionDuration`, `m_TransitionOffset`, `m_ExitTime`, `m_HasExitTime`, and `(parameterName, m_ConditionMode, m_EventThreshold)` triples from `m_ConditionConstantArray`. `m_ConditionMode` is passed through as Unity's raw enum.

**Gameplay metadata** is joined by name from the bundle's `PlayerStateContainer` MonoBehaviours (`controller.py:309-329`), keeping `Type`, `IsDefaultState`, `AdditionalDirectionInfo`, `RotationSpeedClamp`, `StateSensitivity`, `CanInteract`, `DisableRootMotion`, `CreateUniqueMovementStateObject`, `AnimationAuthority`. Measured on the Tagilla pack: 0 states matched - that bundle ships no `PlayerStateContainer`, so every state's `gameplay` is `{}`. It is populated for controllers that do ship them.

**Clip ids, not clip names.** `clipNames` is for humans and is NOT unique: measured on the Tagilla pack, `crouch_run_aim_0` appears twice in `clipNames`, backed by two distinct assets. That only one of the two is an absolute-pose clip (the other being an additive delta) is stated at `extraction/characters/pack.py:45-47` and `viewer/src/character/pack.rs:239-240` and is unverified. `clipIndexById[controller_clip_id] → index into the pack's clips[]`, or `−1` when not extracted, is the authoritative resolution (`extraction/characters/pack.py:40-53`). Deduplication at build time is by asset `path_id`, never by name (`build_character.py:368-380`). Measured: 847 clip slots, 117 extracted for the `locomotion` set.

Clip sets name STATES, and the build expands each to every clip its blend trees reach (`build_character.py:336-366`). Use FULL paths - leaf names recur across layers. A set that resolves to zero clips is a hard error, not an empty pack.

---

## 10. Equipment and attachment binding

EFT's headwear and facecover items are **not skinned**: the prefab is `MeshFilter` + `MeshRenderer` with zero bindposes and no bone hashes, so the item rides a bone rather than deforming (`extraction/characters/skin.py:92-101`). `load_attachment` (`skin.py:572`) reads them.

The mesh's local transform is composed up to the prefab root by walking `m_Father` (guard depth 64), multiplying `parent @ local` with every level already conjugated, then decomposed back to `(pos, xyzw quat, scale)` for storage (`skin.py:640-671`). Scale is the per-column norm of the basis; the rotation is that basis divided by the scale, converted with the same matrix→quaternion routine the clip decoder uses. Measured: the Tagilla welding-mask prefab carries `localRot = (−0.7071, 0, 0, 0.7071)` - the −90° X fixup these prefabs use - with unit scale and zero translation.

Every attachment vertex is pinned to the target bone with full weight (`jointIndex = (0,0,0,0)`, `jointWeight = (1,0,0,0)`) so the same vertex layout and the same shader path serve skinned and rigid geometry (`skin.py:717-719`). Prefabs shipping both a `_base` and a `_custom` variant of one mesh keep only `_base`; taking both draws the item twice (`skin.py:690`).

**The slot→bone mapping is NOT in the asset.** The `Dress` component lists only renderers and a decal type; the real mapping lives in the runtime's `PlayerBody.SlotView`. So the target bone comes from the registry (`extraction/characters/characters.json`, e.g. `"bone": "Base HumanHead"` → index 24) and is an explicit authoring choice, flagged as such. This is the one place in the character pipeline where a value is authored rather than derived.

The weapon uses a different mechanism: no attachment record at all. The consumer looks up the rig bone named `Weapon_root` (measured index 68) and parents the `.eftweap` mesh under an identity offset node (`viewer/src/character/weapon.rs:21`, `viewer/src/character/mod.rs:171-226`). The rig also ships `weapon_holster` (75) and `weapon_holster1` (76) for the slung pose.

---

## 11. The `.eftchar` container: exact byte layout

```
<pack>/manifest.json
<pack>/skin.bin
<pack>/anim.bin
<pack>/textures/*.png
```

`version` is `1` (`extraction/characters/pack.py:27`). All binary is little-endian.

### 11.1 `skin.bin`

Two blocks, in this order:

```
[ every mesh's vertex block, then every attachment's vertex block ]   <- vertexBlockByteLength
[ every mesh's index block,  then every attachment's index block  ]
```

Emission order within each block is `meshes[]` in manifest order followed by `attachments[]` in manifest order (`pack.py:102-169`). `vertexByteOffset` and `indexByteOffset` are ABSOLUTE within the file - the index offsets are computed as a local cursor and then shifted by `vertexBlockByteLength` once the vertex block is closed (`pack.py:165-169`). A consumer can memory-map and slice with no walk.

**Vertex layout - interleaved, stride 72 bytes** (`extraction/characters/skin.py:34`, declared in the manifest as `vertexLayout`):

| offset | size | format | attribute |
|---:|---:|---|---|
| 0 | 12 | `f32x3` | `position` |
| 12 | 12 | `f32x3` | `normal` |
| 24 | 16 | `f32x4` | `tangent` (w = handedness) |
| 40 | 8 | `f32x2` | `uv0` (V already flipped) |
| 48 | 8 | `u16x4` | `jointIndex` (GLOBAL rig indices) |
| 56 | 16 | `f32x4` | `jointWeight` (sums to 1) |

`vertexByteLength == vertexCount * stride` is asserted by the loader (`viewer/src/character/pack.rs:550`).

**Indices**: `u32`, `indexFormat: "u32"`, mesh-local (0-based within that mesh's own vertex block), winding already reversed. `indexByteLength == indexCount * 4`.

Per-mesh manifest fields: `name`, `part`, `view` (`"third"` | `"first"`, defaulting to `"third"` for packs predating the field), `lod`, `vertexCount`, `vertexByteOffset`, `vertexByteLength`, `indexCount`, `indexByteOffset`, `indexByteLength`, `boundBones` (sorted rig indices, debug/validation), `inverseBindposes` (a `boneCount`-long array of 16-float **row-major** rows), and `submeshes[]` of `{material, indexStart, indexCount}` where `indexStart` is in INDICES relative to this mesh's block.

Measured on the Tagilla pack (LOD 0): `Top_Boss_Tagilla_lod0` 8,560 verts at offset 0 (616,320 bytes), 2 submeshes (24,654 + 15,834 indices); `Pants_BOSS_Tagilla_lod0` 2,558 verts at 616,320; `vertexBlockByteLength` 916,920; total 1,147,296 bytes.

Attachment entries add `bone`, `localPos [3]`, `localRot [4] xyzw`, `localScale [3]` and otherwise share the same fields and the same vertex layout, so the loader has one parser.

### 11.2 `anim.bin`

Purely sequential, no header. For each clip in `clips[]` order, for each track in the clip's track order (bone-index ascending), the PRESENT channels are appended in the fixed order **position, rotation, scale**; then, if the clip has root motion, its `(F,3)` array (`pack.py:180-213`).

```
position : F * 3 * 4 bytes,  f32 xyz
rotation : F * 4 * 4 bytes,  f32 xyzw, sign-continuous within the clip
scale    : F * 3 * 4 bytes,  f32 xyz
rootMotion: F * 3 * 4 bytes, f32 xyz, displacement relative to frame 0
```

Each is described in the manifest by `{byteOffset, byteLength, components}` with `byteOffset` absolute in the file. Absent channels have no entry at all, and the consumer then uses the bind-pose value. Measured on the Tagilla pack: clip `Fall`, `frameCount = 2`, track bone 1 → position `{0, 24, 3}`, rotation `{24, 32, 4}`, scale `{56, 24, 3}`.

Per-clip manifest fields: `name`, `duration` (s), `sampleRate` (Hz), `frameCount`, `loop`, `averageSpeed [3]` or `null`, `unresolvedBindings` (count), `rootMotion` (`{bone, byteOffset, byteLength, components}` or `null`), `tracks[]`.

### 11.3 `manifest.json` top level

`version`, `id`, `displayName`, `source {gameBuild, bundles[]}`, `conventions`, `skeleton`, `vertexLayout`, `indexFormat`, `defaultLod`, `meshes[]`, `attachments[]`, `materials[]`, `textures[]`, `clips[]`, `controller`, `blobs {skin {file, vertexBlockByteLength, totalByteLength}, anim {file, totalByteLength}}`, plus `characterForward [3]` and `characterForwardDerived` merged in by the builder.

`conventions` (`extraction/characters/coords.py:109`) is the block a loader asserts against:

```json
{"coordSystem":"viewer","g3":[-1,1,1],"quatOrder":"xyzw","windingFlipped":true,
 "tangentHandednessFlipped":true,"uvVFlipBaked":true,"upAxis":"y","unit":"meter"}
```

`skeleton` carries `boneCount`, `names[]`, `paths[]`, `parents[]` (`−1` for the root), `localPos[][3]`, `localRot[][4]`, `localScale[][3]`.

The Rust loader hard-fails on: `version != 1`; `g3 != [-1,1,1]`; `uvVFlipBaked == false` (the field is `#[serde(default)]`, so a pack predating it is rejected); `windingFlipped == false`; any skeleton array whose length differs from `boneCount`; `parents[i] >= i`; `vertexByteLength != vertexCount * stride`; a missing required vertex attribute; a bindpose table not `boneCount` long or a row not 16 floats; a decoded index count differing from `indexCount`; a channel range past the blob; a channel `byteLength` not divisible by 4; a channel float count differing from `frameCount * components`; a track or attachment targeting a bone ≥ `boneCount` (`viewer/src/character/pack.rs:453-484, 489-517, 546-629, 656-700, 736-774`).

---

## 12. What is dropped

Stated plainly, because a reimplementer will look for these and they are not there.

- **Blend shapes / morph targets.** `Mesh.m_Shapes` is never read anywhere in the package. There is no blend-shape channel in the vertex layout, the manifest, or `anim.bin`.
- **IK.** `Layer.m_IKPass` is recorded as a boolean (`controller.py:303`) and nothing consumes it. The rig's IK helper bones (`IK_S_LPalm` 71, `IK_S_RPalm` 72, `Bend_Goal_Left` 69, `Bend_Goal_Right` 70) are extracted as ordinary bones, animated if a clip drives them, and never solved. No foot planting, no hand-to-weapon IK.
- **Avatar / body masks.** `m_SkeletonMask` and any `AvatarMask` asset are not read. Every layer, if evaluated, would affect every bone.
- **Layer stack.** The controller's layers are extracted as metadata (index, name, blending mode, default weight, owning state machine, synchronized index) but the pose evaluator runs ONE override state at a time plus a single hand-picked additive state. Measured: `TagillaBotAnimController` has 13 layers; the consumer evaluates the base layer's tree plus `Additive_ISaim.idle_aim_AimIn` (`viewer/src/character/drive.rs:55-56, 434-450`).
- **State-machine execution.** Transitions, conditions, exit times and `m_ConditionMode` are all extracted, but the consumer runs its own small explicit machine and uses the graph only for cross-fade DURATIONS (`viewer/src/character/pack.rs:407`). Sub-state-machines, any-state transitions and entry/exit nodes are not modelled.
- **`m_AdditiveReferencePose`.** Not read by anything in the package. The additive reference is the clip's own frame 0 (`viewer/src/character/anim.rs:149-154`); that this matches the shipped reference pose for these clips rests on that source comment and is unverified.
- **Legacy clips.** `m_Legacy` clips raise immediately - their curves live in `m_RotationCurves`, which is unhandled (`clips.py:319`).
- **Humanoid / muscle clips.** Not handled. The absence of `m_MuscleClip` is a hard error (`clips.py:324`); the presence of humanoid muscle bindings is not decoded.
- **Root-motion tables.** `rootMotion` bundle paths are recorded in the registry and resolved by `appearance.controller_for`, but nothing parses them. Per-clip average speed from stripped root motion is used instead.
- **`m_Events`, curve tangent modes, per-clip wrap modes** other than `m_LoopTime`.
- **Second UV set, vertex colours.** Only `m_UV0` is read (`skin.py:462`).

---

## 13. Consumer-side math: pose, blend, skin

Reproduce this and the pack plays correctly in any engine.

**Sampling** (`viewer/src/character/anim.rs:99`). Because `frameCount = round(duration·rate) + 1`, the last frame is exactly at `duration`:

```
t  = looping ? rem_euclid(time, duration) : clamp(time, 0, duration)
f  = (t / duration) * (frameCount − 1)
i0 = min(floor(f), frameCount − 1);  i1 = min(i0 + 1, frameCount − 1);  a = f − i0
```

Position and scale lerp; rotation slerps (within a clip the emitter already made the sequence sign-continuous, so the short arc is guaranteed).

**Blending is weighted accumulation per bone, not a chain of pairwise lerps** (`anim.rs:21-96`). A 2D blend can have nine active clips. Accumulate `pos += p·w`, `scale += s·w`, and for rotation add the quaternion as a `Vec4` after aligning it to the first contribution's hemisphere (`dot < 0 → negate`), then normalise (nlerp). **Weight is tracked PER BONE, PER CHANNEL**: clips in a blend legitimately drive different bone subsets, and normalising by the blend's total weight instead shrinks the untouched bones toward zero. Where a channel's accumulated weight is `≤ 1e-5`, fall back to the bind pose value.

**Blend-tree evaluation** (`anim.rs:203`). Recursive, multiplying the parent's weight into each child, pruning below `1e-5`, then renormalising the leaf weights (gaps from unextracted clips and float error both leave the sum off; an unnormalised pose reads as the character sinking or shrinking).

- `1d`: linear between the two bracketing `threshold`s, clamped outside the range.
- `direct`: uniform `1/n`.
- every 2D flavour: **gradient band interpolation** - for child `i`, `wᵢ = minⱼ≠ᵢ clamp(1 − ((sample − posᵢ)·(posⱼ − posᵢ)) / |posⱼ − posᵢ|², 0, 1)`, then normalise; degenerate case snaps to the nearest child. This IS Unity's algorithm for Freeform Cartesian; for the two Directional flavours Unity uses a polar variant, so this is an APPROXIMATION there.

**Additive layers** (`anim.rs:155`). Reference is the clip's frame 0. Per bone: `out.rot = slerp(identity, q_sampled · q_frame0⁻¹, w) · out.rot`, and `out.pos += (p_sampled − p_frame0) · w`. Time is CLAMPED, never wrapped - these are aim-in transitions being scrubbed, and wrapping snaps back to neutral.

**Skinning.** Standard linear blend skinning with the pack's own bindposes:

```
skin(v) = Σᵢ wᵢ · World(joint[i]) · InverseBindpose[joint[i]] · v
```

where `World(b)` is the forward-pass product of local TRS matrices from the root, `parent @ local`. Because the tables are rig-sized and joint indices are global, one joint palette serves every mesh and every attachment of the character.

**Root motion is NOT re-applied** by the consumer. It is stripped from the bone tracks; the driver instead uses `|averageSpeed.xz|` to rate-match playback so feet do not skate (`viewer/src/character/drive.rs:383-391`, `viewer/src/npc.rs`). A consumer that also wants the clip's travel must read the `rootMotion` channel explicitly.

---

## 14. Invariants and their failure signatures

The "what you SEE" column records observations from the source notes taken when each invariant was broken; those signatures are not reproducible from the repo as it stands.

| invariant | how it breaks | what you SEE |
|---|---|---|
| `G3` applied exactly once, per datum | applied twice, or applied to the composed world matrix on top of already-conjugated locals | the character is mirrored back into Unity space; he stands correctly but faces the wrong way relative to the map, and left/right equipment swaps |
| winding reversed for an `X`-mirrored mesh | forgotten | every triangle faces away; the body renders inside-out or vanishes under back-face culling. The loader refuses a pack with `windingFlipped: false` |
| tangent `w` negated | forgotten | normal-mapped detail lights from the mirrored side; a face shades convex where it should be concave, and the error is invisible on flat surfaces |
| quaternion conjugated as `(x,−y,−z,w)` | using `(−x,y,z,w)` (naively "the same as a point") | the rig is connected, every quaternion is unit, positions match the bind pose exactly - and every pose composes to a body tilted 60–130° from vertical with legs folded to half reach and feet floating. This is what `validate.py` exists to catch |
| euler order `Rz@Ry@Rx` (Maya XYZ) | using Unity's `Quaternion.Euler` ZXY | same signature as above, but only on euler-encoded clips, so a character animates correctly in one state and folds in another |
| UV V flipped exactly once | skipped, or done again in the shader | textures sample upside down. On a face this reads as a subtle "UV mapping issue", not an obvious break. The loader refuses a pack with `uvVFlipBaked: false` |
| bindpose ≠ rest pose | using the skeleton's bind-pose world matrix as the inverse bindpose instead of the mesh's own `m_BindPose` | the mesh explodes or collapses toward the origin at spawn, before any clip plays. Diagnostic: freeze the rig in its bind pose (`EFT_CHARACTER_BIND=1`) - if it is still wrong, skinning is wrong, not the pose pipeline |
| bone remap from the authoritative source | trusting `Skin._bonePaths` over `Mesh.m_BoneNameHashes` | limbs animate to the wrong joints - an arm follows the leg. Recorded shipped disagreements: `usec_upper_commando` shifted two slots, `Top_BOSS_Killa_base` a different length (49 vs 48) |
| `parents[i] < i` | unsorted hierarchy | a forward-pass world-matrix computation reads a parent that has not been written yet: children lag one frame or jitter. Both emitter and loader assert it |
| binding curve widths sum to the clip's curve count | assuming a width, or ignoring the euler attribute | curves are read one slot off; the animation is smooth, connected, unit-quaternion clean, and completely wrong. The build fails on the sum mismatch |
| root motion stripped once | leaving it in the track AND moving the character | the body slides out from under the camera along the clip's own axis |
| clip resolved by controller id, not name | resolving by name | you get the additive-DELTA twin of a clip and play its deltas as absolute poses; the character folds only in the states that happen to hit the duplicate |
| joint weights sum to 1 | Unity's stored weights used raw | vertices drift slightly toward the origin - a soft shrink most visible at the extremities |
| zero-weight influence clamped | left as whatever slot byte the exporter wrote | an out-of-range joint index reads garbage from the palette; stray vertices shoot to infinity as long spikes |
| `_SpecMap` not bound to occlusion | bound anyway | every character surface multiplies down to a fifth of its ambient light, with hard seams where the atlas's gloss regions change (the "black skullcap with a seam across the scalp") |
| normal map repacked from DXT5nm | written raw | tangent normal ≈ `(1, y, z)`, pointing along the tangent instead of out of the surface; shading flips between lit and black as the head turns |

Anatomical validation (`extraction/characters/validate.py`) is the last line of defence, because a wrong rotation basis produces output that no structural check can detect. It composes the pose and measures, on reference clips resolved BY CLIP ID from standing-locomotion states (never by name). Bounds: lowest foot `y ∈ [−0.20, 0.45] m`; head `y ∈ [0.15, 2.05] m`; pelvis→head tilt from `+Y` `≤ 50°` (skipped for clips whose name contains `prone`); pelvis→foot reach `∈ [0.35, 1.15] m`; collarbone→palm reach `∈ [0.15, 0.80] m`; palm world `y ∈ [−0.10, 2.20] m`. Frames `0`, `F/3`, `2F/3` of each reference clip are measured. Arm reach is checked because a validator measuring only pelvis/head/feet POSITIONS passed a pose whose head was folded onto the chest and whose arms were inside the torso - three positions do not constrain a skeleton.

---

## 15. Old patterns

- **Hand-authored part lists.** `characters.json` used to carry hand-picked prefabs per character. Measured against the game, the authored scav was wrong: it wore `head_civilian_1`, which does not appear in the scav appearance table at all (the game rolls `wild_head_1/2/3/drozd/misha`), and it had no hands slot. Appearance is now rolled from the game's own weighted tables; `characters.json` survives for named one-offs and for the few facts the tables do not carry (the equipment slot→bone choice, controller overrides, clip sets).
- **A per-clip rotation basis chosen by scoring.** An earlier revision decoded each clip twice - with and without an extra X flip on the rotation curves - and selected the better-scoring decode with a `validate.choose_basis` helper. Neither `choose_basis` nor `clips._curve_quat_to_transform` exists in the current source; the names appear only in prose (`README.md:164`, `validate.py:10`, `validate.py:168`). The present decoder applies `coords.quats` uniformly to quaternion curves and to euler-derived quaternions, with no per-clip choice; the apparent need for one was an artefact of a broken euler conversion scoring the two encodings against each other (`clips.py:393-395`).
- **Zeroing bone 0's translation to remove root motion.** The carrier is not bone 0; measured on the Tagilla pack, it is bone 1 (`Root_Joint`) in 100 of 109 clips, bone 2 in 6 and bone 14 in 3. Root motion is now found by displacement threshold and stripped into its own channel (`viewer/src/character/drive.rs:453-457`).
- **A CAB index built by globbing `*.bundle`.** Every item whose geometry an extensionless bundle provides assembled to nothing. Bundles are now recognised by their UnityFS-family header. The scale of the miss recorded at `unity_deps.py:14-17` - 2,295 CABs across 1,160 extensionless bundles, taking with it Killa's 6B13 and rig, the 6B5 Flora, and eleven submeshes of the M4A1 - is unverified without a game install.