# `.eftchar` — EFT playable-character extraction

Turns EFT's shared 78-bone character rig, a character's skinned body parts, and its Mecanim
animation set into a **self-describing** `.eftchar` pack the native viewer consumes.

Same contract as `.eftpack`: `manifest.json` declares every stride, byte offset and convention,
and the Rust loader reads the layout **from the manifest** — the emitter and the consumer cannot
drift.

## Why this shape

EFT does not ship a character as one asset. It ships:

- **one canonical rig** — `characters/character/skeleton.bundle`, 79 GameObjects, a named biped
  (`Root_Joint/Base HumanPelvis/Base HumanSpine1/...`) plus rig extras (`Weapon_root_3rd_anim`,
  `Camera_animated_3rd`, `IK_S_LPalm`, `Bend_Goal_Right`, `weapon_holster`). **Every** character
  — PMC, scav, Tagilla, Killa — binds to this one rig.
- **body parts** — separate prefab bundles (`top_boss_tagilla`, `pants_boss_tagilla`, …), each a
  `SkinnedMeshRenderer` + `LODGroup`, each binding only the bones it needs (top = 48, pants = 12).
- **animation** — Mecanim `AnimationClip`s whose curves are bound by **CRC32 of the bone path**,
  not by bone index.

So the extractor's whole job is resolving three joins, all game-derived:

1. `skeleton.bundle` transform hierarchy → canonical bone order + path strings.
2. Each part's `Skin._bonePaths` (and the mesh's own `m_BoneNameHashes`) → mesh-bone → rig-bone.
3. Clip `genericBindings[].path` (a CRC32) → rig-bone, cross-validated against the
   AnimatorController's `m_TOS` hash→string table.

Every join is **asserted**, not assumed. A mismatch fails the build loudly rather than emitting a
subtly wrong character (see `--strict`, on by default).

## Who a bot IS — derived, not listed

A character is **not** a hand-picked set of prefabs. Each bot type carries a weighted `appearance`
table (`body` / `feet` / `hands` / `head`) whose ids resolve through `customization.json` to
prefab bundles, and the game rolls it. `appearance.py` owns that question and nothing else:

```
appearance.resolve(bot_type, seed) -> {parts[], controller, rootMotion, appearance{}}
```

Every consumer calls it — the character builder, the kit baker (`loadout.py`), the viewer's
roster — so a kit and the body it is worn on can never disagree about which meshes a bot has.
Seeded from `(bot_type, seed)`, so a given NPC keeps its identity across frames, reloads and
machines.

```
build_character.py --bot bosskilla --seed 0
```

Why this replaced the old hand-written part lists: measured against the game, the authored scav
was **wrong**. It wore `head_civilian_1`, which does not appear in the scav appearance table at
all (the game rolls `wild_head_1/2/3/drozd/misha`), and it had no hands slot. `characters.json`
now survives only for the few facts the tables genuinely do not carry.

Three details the tables force:

- **`hands` is resolved but not built.** The game's hands slot points at first-person prefabs
  binding the FPV rig; a third-person body already includes its arms. It is reported, because a
  first-person view would need exactly it.
- **A bot type with an empty appearance table is an error, not a default.** Six of the 63 types
  (BSG's own `*test` placeholders, a few followers that never spawn standalone) define no body.
  The resolver refuses rather than inventing one.
- **The animator is a bounded lookup, not a guess.** The game ships exactly three bot controllers
  (base / boar / tagilla); the stems present on disk are the whole candidate set.

Tagilla needs no `head` part: his head mesh and its `Head_BOSS_Tagilla` material are the **second
submesh of `top_boss_tagilla`** — which is why, alone among the bosses, he has no
`head_boss_tagilla.bundle`. The extractor discovers submesh→material pairing from the asset, so
this needs no special case.

## Conventions baked in at extraction

The viewer's world is the map pack's world: **`G3 = diag(-1, 1, 1)`** (an X-flip) applied to Unity
world. `coords.py` is the *only* place that conjugation lives. It is applied to:

| data | rule |
|---|---|
| bone local position, clip position keys | `p → (-x, y, z)` |
| bone local rotation, clip rotation keys | `q → (x, -y, -z, w)` (conjugation by a reflection) |
| bone local scale | unchanged (diagonal ∘ diagonal) |
| mesh positions / normals | `v → (-x, y, z)` |
| mesh tangents | xyz flipped, handedness `w` negated |
| inverse bindpose `B` | `G B G⁻¹` |
| triangle winding | **reversed** (an X-flip mirrors every triangle) |

Winding is flipped rather than drawn double-sided because a character goes through Bevy's ordinary
PBR mesh path, where back-face culling is on. That differs from the map's `gpu_driven` path, which
keeps mirrors correct with a cofactor normal matrix + double-sided draw instead. Both are correct;
only one is available per path.

## Things that bite

**A bundle is not a `*.bundle`.** 1,160 of the game's ~7,400 bundles ship with **no extension** —
`assets/content/characters/character/vest/ar_6b13_mesh` is one, and it is where Killa's armour
mesh lives. A CAB index that walked `*.bundle` therefore lacked 2,295 CABs, and every item whose
geometry those bundles provide assembled to *nothing*: Killa's 6B13 and his rig, the 6B5 Flora,
and eleven submeshes of the M4A1. `unity_deps.py` now recognises a bundle by its **UnityFS
signature**, never by its name — the same rule that catches DXT5nm normal maps by measurement
rather than filename.

**When two bone tables disagree, the mesh's own wins.** A part carries `Skin._bonePaths` and the
mesh carries `m_BoneNameHashes`. The vertex bone *indices* address the mesh's bind-pose array,
which is parallel to the hash list — so the hashes are authoritative **by construction**, and a
remap of any other length would index out of range. They do disagree in shipped assets, in both
possible ways: `usec_upper_commando` is the same length but shifted two slots, `Top_BOSS_Killa_base`
is a different length outright (49 vs 48). The length case is why a naive `zip()` comparison
reported an empty diff and looked like a phantom.

**Missing geometry is a regression, not a footnote.** A rolled slot that assembles to nothing means
a bot spawns in game wearing armour the viewer does not draw. `build_loadouts.py` reports every
drop with its cause and exits non-zero. The single exception is Unity's built-in library
(`unity default resources`) — the editor primitives a prefab uses as a gizmo or collider proxy,
part of the engine rather than the game, and absent from StreamingAssets by design.



**Clip references cross bundles.** `TagillaBotAnimController` declares **847** clip slots; its own
bundle holds **430**. The rest are external PPtrs into `character_animations.bundle`. The controller
and the shared animation bundles must be loaded into ONE `UnityPy` environment (`_shared.animations`
in the registry) or roughly half of every blend tree reads back nameless — silently, since a missing
clip just looks like an empty string. The build prints `847 clip slots, all resolved`; if that says
`N unresolved`, a bundle is missing from the list.

**Blend trees nest, and layers share state machines.** `Base Layer.StateMachine_Move.MOVE` is a
9-way 2D directional blend on (`Direct_X`, `Direct_Y`) where *every direction is itself* a 2D blend
on (`Speed`, `Level`) — 86 nodes, 76 leaf clips. A flat "root's children" read returns nine nodes
with no clips. Separately, `Base Layer`, `Sync_SprintHands` and
`TagillaSyncLayerForRegularOperations` all point at state machine 0; the FIRST layer to reference a
machine owns it and the rest are synced views, so a naive `{sm: layer}` map attributes every
base-layer state to the last synced layer.

## Curve decode

Unity stores a generic clip's curves in three encodings at once — `m_StreamedClip` (cubic
polynomial keys at arbitrary times), `m_DenseClip` (baked frame-major samples), `m_ConstantClip`
(one value forever) — concatenated into a single curve-index space in that order.

The extractor **resamples all three onto one uniform grid** at the clip's own sample rate, then
groups curves into per-bone position / rotation / scale tracks. The viewer therefore ships exactly
one sampler and knows nothing about Unity's encodings. Cost is a modest size increase; the win is
that adding a character or a clip can never introduce a new decode path.

Streamed keys are evaluated as the cubic they are — `v(dt) = c₀dt³ + c₁dt² + c₂dt + c₃` — not
converted to Hermite tangents and back.

### The rotation basis is per-clip, and it is MEASURED

The hardest bug in this pipeline, and the reason `validate.py` exists.

A clip's rotation curves do not always sit in the same basis as the `m_LocalRotation` values in
`skeleton.bundle`; some are mirrored on X. Get it wrong and nothing looks broken *structurally* —
positions still match the bind pose exactly, every scale is 1.0, every quaternion is unit to 1e-5 —
but every pose composes to a body tilted 60–70° from vertical with the legs folded to half their
reach and both feet floating 0.5–0.9 m off the floor.

Worse, **it is not a global property.** Tagilla's `MOVE` tree references two distinct assets both
named `crouch_run_aim_0` (ids 149 and 203) which need *opposite* choices. They differ in encoding —
149 drives 58 of its 78 bones with euler curves and is 373-curve streamed; 203 is pure quaternion and
mostly dense — so no single rule serves both.

So the basis is not assumed, it is **derived**: each clip is decoded both ways and
`validate.choose_basis` scores the composed poses against the skeleton's own geometry (does a foot
ever reach the floor, does the head sit in a human band, is the spine roughly up), integrated over
several frames. The better-scoring decode wins, and the choice is recorded per clip. Clips where the
two score alike — prone, vault, death, where "standing body" is meaningless — inherit the character's
majority basis instead of a coin flip, and the build reports the split.

For Tagilla's locomotion set: **108 flipped / 9 unflipped, 0 undecided**, and all 56 clips reachable
from `Idle_Aim` and `MOVE` then pass anatomical validation across 168 pose measurements.

Honest limitation: this is empirical. I can demonstrate which decode yields a real human but not why
BSG's clips disagree, so the selector measures rather than claiming a rule.

## Layout

```
out/characters/<id>/
  manifest.json     # version, conventions, skeleton, vertexLayout, meshes[], materials[], clips[], states[]
  skin.bin          # interleaved vertices per vertexLayout, then u32 indices
  anim.bin          # per-clip per-bone f32 tracks; offsets/counts in the manifest
  textures/*.png
```

## Modules

| file | role |
|---|---|
| `coords.py` | the G3 conjugation, and nothing else |
| `unity_bind.py` | path CRC32, curve-index ↔ binding walk, hash-function self-validation |
| `skeleton.py` | `skeleton.bundle` → canonical bone table |
| `skin.py` | part bundle → meshes, bone remap, bindposes, materials, textures |
| `clips.py` | `AnimationClip` → resampled per-bone tracks |
| `controller.py` | `AnimatorController` + `PlayerStateContainer` → locomotion state table |
| `pack.py` | writes the manifest + blobs |
| `build_character.py` | CLI |

## Facing is derived, never authored

The forward-walk clip's own root motion **is** the character's forward axis, so the manifest carries
a measured `characterForward` and the viewer aligns character-forward to movement-direction with no
magic 180° offset. For Tagilla, `walk_aim_0` travels 4.003 m in 1.600 s → `[0, 0, 1]` at 2.50 m/s.
`characterForwardDerived: false` means it fell back to +Z and should not be trusted.

## Usage

```
python extraction/characters/build_character.py --list
python extraction/characters/build_character.py --character tagilla --dump-states
python extraction/characters/build_character.py --character tagilla --dump-states --grep JUMP.
python extraction/characters/build_character.py --character tagilla --lod 0
python extraction/characters/build_character.py --character tagilla --clips all
python extraction/characters/build_character.py --character tagilla --skip-clips   # geometry only
```

`--dump-states` before authoring a clip set: state paths are graph-specific, and a set that resolves
to zero clips is a hard error rather than an empty pack.

Verified output for `tagilla --lod 0`: 79 bones, 2 meshes (8560 + 2558 verts), 3 materials,
9 textures, 117 clips, **78 distinct bones animated** (= every non-root bone of the rig, which is
what tells you the CRC32 join landed), 1496 non-bone bindings ignored (Animator float parameters).
`skin.bin` 1.0 MB, `anim.bin` 10.2 MB, ~10 s.

Paths come from the environment, as elsewhere in `extraction/`:
`EFT_GAME_DATA` (default `C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data`),
`EFT_CHAR_OUT` (default `<repo>/out/characters`).
