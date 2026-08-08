# Extracting Tarkov Unity maps

Turning a Unity EFT installation into a renderable, self-describing map pack. Everything here is
engine-agnostic data and math: byte layouts, formulas, invariants, and what breaks when each one is
violated. A reimplementation (Blender importer, Unreal importer, custom renderer) needs only this
document plus the reference file for the subsystem it touches.

The single most expensive mistake in this pipeline's history is decomposing a world matrix into
translation/rotation/scale. The second is applying the handedness flip twice. Both are covered in
the rules below and both have a distinctive visual signature.

## Pipeline shape

```
GAME FILES                    DATASET                        PACK                     VIEWER
<EFT_GAME_DATA>/              <EFT_ASSETS_ROOT>/<dataset>/   <pack>.eftpack/          native wgpu
  globalgamemanagers            scene.json                     manifest.json          renderer
  level0..level<N>              meshes/*.obj (+ .vcol.npy)     meshes.bin             reads layout
  sharedassets*.assets          tex/*.png                      instances.bin          from the
  *.bundle (terrain, chars)     lights_<lv>.json               materials.json         manifest and
                                interact_<lv>.json             colliders.bin          hardcodes
                                decals.json                    collider_meshes.bin    nothing
                                terrain_layers/                volume.json/.bin
                                                               nav.bin/.json
                                                               gamedata.json
                                                               grass.bin, tex/
```

One command drives all of it: `python tools/build_map.py <map>` (`--force`, `--self-contained`,
`--dry-run`). It prints `[STAGE i/N] name` markers and exits 0 only when the pack is stamped.

| Stage | Does | Authority |
|---|---|---|
| 1 | Full game extraction to the dataset when `scene.json` is absent (long; game must be CLOSED) | `extraction/unity/eft_extract_v2.py` |
| 2 | Interactables, then every `*_Light` scene, then the colour-grade LUT | `extraction/unity/extract_interact.py`, `eft_extract_lights.py`, `extraction/grade/` |
| 3 | SH irradiance volume - portable baker runs POST-assemble; `EFT_BAKE=warp` runs a CUDA baker PRE-assemble | `atlas bake-sh`, `extraction/bake/bake_volume2.py` |
| 4 | Projected decals, then assemble the pack (atomic: staged into `<pack>.building`, then swapped) | `extraction/intel/extract_decals.py`, `eft_pipeline/assemble_bevy.py` |
| 5 | Grass density grids to `grass.bin` (skipped when the dataset yields no grids) | `extraction/unity/eft_extract_grass.py`, `eft_pipeline/build_grass.py` |
| 6 | Typed gameplay zones: exfils, doors, containers, loot points, mines, snipers | `extraction/intel/extract_gamedata.py` |
| 7 | Item icons (network, cached into the pack) | `extraction/intel/fetch_icons.py` |
| 8 | Nav grid from the pack's collider world | `atlas bake-nav` |
| 9 | Game fingerprint stamp, manifest reconciliation, lighting verification | `tools/stamp_fingerprint.py`, `build_map.finalize_pack_manifest` |

Every stage after 1 is cached by output presence and re-runnable in isolation. Stages 2, 3, 5, 6, 7,
8 are OPTIONAL: they never fail the build, which is exactly why each has a freshness guard (see
"stale sidecar" in the failure table).

## THE RULES

These six are non-negotiable. A reimplementation that gets any of them wrong produces output that
looks plausible and is wrong in a way that is hard to attribute.

### Rule 1 - Apply the raw 3x3. Never TRS-decompose.

`scene.json` `instances[].m` is 16 floats, **row-major**, the FULL baked world transform (the whole
parent chain is already applied), in **raw Unity space**.

```
M3 = [[m0, m1, m2],        T = (m3, m7, m11)
      [m4, m5, m6],
      [m8, m9, m10]]

world = V_local @ M3.T + T          # normals: n_world = normalize(n @ inv(M3))
```

The rest of the record is what the assembler consumes: `mesh` (OBJ basename), `subs[]` (one material
dict per submesh: `tex`, `nrm`, `col`, `sh`, `uv` = `[sx, sy, ox, oy]`, `cut`, `role`, `vp`, `n` =
face count), `kind` (`mesh` or `terrain`), `root`, `lv` (level), `lod` `{g, i}`, `drop` (Unity-hidden),
and the ancestry `path_id`s `par` / `par2`.

Shear is common and legal: a non-uniformly scaled parent times a rotated child bakes a world 3x3 of
`R·S` that is not a similarity *(measured on Interchange: about 4% of instances, column-orthogonality
error up to 0.33; the code's own threshold is `> 0.02` = sheared, `instmath.py:37`)*. `scale = norm(columns); quat = from(columns/scale)` silently discards the off-diagonal
terms. There is no threshold at which this is acceptable.

The **only** sanctioned exception is a rank-deficient 3x3 (a mesh flattened to a plane: billboards,
baked decal planes), where `inv(M3)` raises. Then bake the instance to world geometry with a
pseudo-inverse normal transform and emit an identity affine - `eft_pipeline/tarkmap_core/instmath.py:55`
(`bake_into`), flagged `BAKED_WORLD` (`0x4`) in the pack.

### Rule 2 - Handedness is a CONJUGATION, applied exactly once.

Unity is left-handed; glTF, Blender, and this repo's viewer world are right-handed. Y-up matches;
handedness does not. The fix is a single-axis flip `G = diag(-1, 1, 1, 1)` - constant for every EFT
map, overridable per map via `coordinates.global_matrix`
(`eft_pipeline/tarkmap_core/config.py:121`).

```
M' = G · M · G⁻¹        applied to the RAW, unreflected mesh vertices
det(M') = det(M)        so instances stay det>0 and nothing turns inside-out
```

`eft_pipeline/tarkmap_core/instmath.py:13` (`make_conjugator`) is the authority. The pack stores
`apply_global(m)[:12]` = the row-major 3x4, **not** the glTF column-major transpose.

Three wrong ways, each with its own signature:

- **No flip at all** - the entire map is X-mirrored. "Y-up so no swap needed" is false.
- **Premultiply** `M' = G·M` - every instance has det<0; instanced renderers cannot fix normals or
  winding, so undersides face the light and the world reads inside-out.
- **Reflect the vertices AND conjugate** (`V' = V @ G3.T` plus `G·M·G⁻¹`) - net `G·M·v`, a true
  mirror. Each piece flips about its own pivot while the pivot stays put; textures read
  right-to-left. Users describe it as *"the position is fine, but slices of the map are mirrored."*
  A global transform can neither cause nor fix per-piece mirroring - if pieces are wrong relative to
  each other, stop editing the global matrix.

Everything else in the world uses the SAME `G3`, reduced: a point `p → G3 @ p`, a direction
`d → G3 @ d`, a rotation `R → G3 @ R @ G3`. Lights, spot forwards, decal projectors, colliders,
LODGroup centres, gameplay zones. Apply it in exactly one stage. Extractors that write raw Unity
matrices (decals, colliders) must NOT conjugate; the assembler owns it. Conjugating in both places
is a double flip and lands geometry on the wrong side of the map.

`det(M) < 0` genuinely occurs *(measured: roughly one instance in a 137k-instance scene)*. Do not "fix" it -
flag it and let the renderer flip front-face/winding (`FLAG_MIRROR = 1<<0`), or bake it with
reversed winding.

### Rule 3 - Vertex X-negation is the LOCAL frame, not the world fix.

Mesh OBJs come out of UnityPy's OBJ exporter, which writes:

```
v  -x  y  z          vn  -x  y  z          vt  u  v   (unchanged)
f   c   b  a         (triangle indices emitted in REVERSED order)
```

Terrain OBJs are written by this pipeline to match that frame **exactly** -
`extraction/unity/eft_extract_v2.py:717` (`write_terrain_obj`) emits
`v {-col*step*sx} {height} {row*step*sz}` with reversed winding - so terrain and meshes live in ONE
local frame and one conjugated instance places both.

The height is **not** the raw sample:

```
height = (raw_u16 / 65535) * 2 * m_Scale.y      # the *2 is Unity storing 15 bits in a 16-bit field
uv     = (col / (cols-1), row / (rows-1))       # decimation step defaults to 2
```

Omitting the `* 2` halves every terrain and sinks it tens of metres below the rest of the map. Holes
come from `m_Heightmap.m_Holes`, a `(res-1)^2` u8 grid where a value **< 128 cuts the quad**; skip
them or tunnels and bunker mouths fill in solid.

There is **no per-terrain flip and no per-terrain shift**. Any such compensation is a hack for a broken conjugation and desynchronises
terrain once the conjugation is correct. The generated decal quad is X-negated for the same reason.

X-negation fixes each mesh's local shape. It does NOT fix world handedness - that is Rule 2. Doing
both to the same vertices is the double-flip bug above.

### Rule 4 - Texture V-flip: baked into the UVs, after tiling, once.

Unity's texture origin is bottom-left; PNG rows and the wgpu sampler are top-left.

```
uv' = uv * [sx, sy] + [ox, oy]      # bake the Unity _ST tiling FIRST
uv'.v = 1.0 - uv'.v                 # then flip V in texture space
```

`eft_pipeline/assemble_bevy.py:1005-1006`. The manifest records
`conventions.uvVFlipBaked = true`, `uvOrigin = "top-left"`, `uvTilingBaked = true`, so the loader
must not redo either. `materials.json.uvXform` is reference only - the tiling is already in the
vertex UVs.

There is never a U-flip. Wanting one means the geometry is still mirrored (Rule 2); fix the
geometry, do not compensate on the texture.

Inverse direction, and it bites: a baker that SAMPLES a source PNG (SH bake albedo, grade) must
sample with Unity's bottom-left V origin. The pipeline flips at consumption, never at the source.

Normal maps are DirectX convention (green points down): flip G on import or negate `n.y` in the
shader (`materials.json.normalGreenFlip`, `manifest.conventions.normalMapGreenFlip`). Colour spaces:
albedo and emissive sRGB, normal linear.

### Rule 5 - Decal projectors have their own axis and UV conventions.

`StaticDeferredDecal` is an IL2CPP MonoBehaviour with no type tree; read the raw payload after the
serialized header:

```
hsize = (12 + 4 + 12 + 4 + len(m_Name utf8) + 3) & ~3      # PPtr GO, m_Enabled, PPtr script, name len
payload[ 0..16) : 4x f32   x1, y1, x2, y2   atlas-PIXEL rect selecting the cell
payload[16..20) : i32 fileID                PPtr<Material>; externals[fileID-1] names the file
payload[20..28) : i64 pathID                unpacked as `<iq` - STANDARD size, NO native alignment
```

The `pathID` is at offset **20, not 24**. `<iq` disables alignment padding; a reader assuming natural
8-byte alignment decodes four bytes late and resolves the wrong material, or none.

`extraction/intel/extract_decals.py:351,358,359`.

- **Axes**: the image spans the box's local **X and Z** and projects along local **Y**. Derived from
  the data, not assumed *(measured across 1,269 Interchange projectors: the atlas cell's aspect
  matches `|colX|/|colZ|` about three times better than `|colX|/|colY|`, mean |log| error 0.43 vs
  1.34)*. Building the quad in XY lays every decal on its side.
- **UVs**: `su=(x2-x1)/W, sv=(y2-y1)/H, ou=x1/W, ov=y1/H`, where `W`,`H` are the **pixel dimensions
  of the resolved `_MainTex` PNG**, not of the projector. The rect is in that texture's pixel space,
  so a resized export must have its rect scaled by the same factor or every cell shifts. The rect's Y
  origin is the texture's bottom-left, so **V passes through unflipped** - flipping it selects the
  mirrored atlas row.
- Rects legitimately carry small negative insets (-2, -8 px are common) and can overrun the texture
  by a pixel. That is authored bleed: clamp into range, do not reject.
- The emitted matrix is **raw Unity space**; the assembler conjugates.
- A decal is a PROJECTION, not a quad. Clip receiving triangles to the box: facing cutoff
  `cos >= 0.50`, depth reach `2.5x` the authored box half-depth, surface offset `0.012 m` along the
  receiving normal, box size sanity `<= 200 m`, `<= 4000` triangles per decal
  (`extraction/intel/decal_project.py:25-42`). A flat quad loses the half of a sign that sits behind
  a staggered plate.

### Rule 6 - Derive from the game data; never author a constant.

Sea level comes from the scene's water planes, LOD distances from `LODGroup`s, light scene indices
from BuildSettings, culls from structural properties. Per-map hardcoded values and per-mesh-name
special cases are rejected: every map must build identically from the same code. Enrichments are
acceptable only when programmatic.

## Environment contract

| Variable | Meaning |
|---|---|
| `EFT_GAME_DATA` | The `EscapeFromTarkov_Data` dir **itself** - the one holding `globalgamemanagers`, `level0..N`, `sharedassets*.assets`. Not the install root. |
| `EFT_TARKMAP_ROOT` | The tarkmap dir itself: holds `maps/<id>/config.json` and `out/`. |
| `EFT_ASSETS_ROOT` | Datasets dir. Default `<EFT_TARKMAP_ROOT>/../eft_assets`; keep the standard sibling layout or config path resolution breaks. |
| `EFT_PY_UNITY` / `EFT_PY_BAKE` | Interpreters for the UnityPy stages and the CUDA bake stage. |
| `EFT_ATLAS_EXE` | Built viewer binary, used for `bake-sh` and `bake-nav`. Absent means no baked lighting and no routing. |
| `EFT_BAKE=warp` | Use the CUDA/warp baker (pre-assemble) instead of the portable Rust baker (post-assemble). |
| `EFT_BAKE_CPU=1` | Force the SH bake onto the CPU (device-loss retry path). |

Python 3.10+. Hard deps: `numpy`, `Pillow`, and `UnityPy` pinned to exactly **1.25.0**
(`extraction/requirements.txt`; the API shifts between minors, so 1.25.3 is not a substitute). Soft: `scipy` (grade LUT rebuild), `warp-lang` + an NVIDIA GPU (CUDA SH bake).
`python extraction/check_env.py` is the gate; `--init <dir>` scaffolds a workspace. Stage 1 needs
the game **closed**.

No game asset is ever redistributed: the colour-grade LUT is rebuilt from the local install, with a
parameter-fitted reconstruction as the no-game-files fallback.

## Which reference do I need

| I am working on | Read |
|---|---|
| Instance placement, the conjugation, shear/mirror handling, `scene.json` schema, mesh and terrain vertex frames, winding | [geometry and placement](geometry-and-placement.md) |
| Texture export and naming, UV tiling and the V-flip, alpha roles and cutoff, PBR fields, detail/parallax/emissive, normal-map convention | [textures and materials](textures-and-materials.md) |
| Terrain heightmaps and holes, MicroSplat layer scales and control-map channel packing, albedo slice tiling, the colour-grade LUT format | [terrain and the colour grade](terrain-and-colour-grade.md) |
| `StaticDeferredDecal` discovery without type trees, payload byte layout, material and atlas resolution, box axes, projection clipping | [decals](decals.md) |
| Reading IL2CPP MonoBehaviours from raw bytes, per-class payload layouts, exfils/doors/loot/zones, bit masks | [game data](game-data.md) |
| Physics colliders and their record layout, `interact_<lv>.json`, and the GameObject-name semantic layer | [colliders, interactables and semantics](colliders-interactables-and-semantics.md) |
| Skeleton and skinning, animation clip decode, the animator graph, equipment binding, the `.eftchar` container | [characters and animation](characters-and-animation.md) |
| Sky cubemap faces and derived colours, particle systems and flipbook atlases, the Water4 parameter set | [sky, particles and water](sky-particles-and-water.md) |
| Light extraction, controller-driven lamps, what EFT does and does not ship, the SH irradiance volume format and bake math | [lighting and the SH bake](lighting-and-sh-bake.md) |
| Build stages, caching and forced invalidation, the `.eftpack` directory layout, `manifest.json` schema, every binary stride and offset, self-contained packs | [build pipeline and pack format](build-pipeline-and-pack-format.md) |

## Pack layout at a glance

Strides are declared in `manifest.json` and generated from the emitter's dtypes, so emitter and
loader cannot drift. Read them from the manifest; hardcode nothing.

**Every binary in the pack is little-endian** - `meshes.bin`, `instances.bin`, `colliders.bin`,
`collider_meshes.bin`, `grass.bin`, `nav.bin` and `volume.bin` alike (the emitter dtypes are explicit
`<f4` / `<u4` / `<i4`, `assemble_bevy.py:88,102,139`).

- **vertex** stride 36: `position f32x3 @0`, `normal f32x3 @12`, `uv f32x2 @24`, `color unorm8x4 @32`.
  `meshes.bin` is the whole vertex section followed by the whole u32 index section. Watch the units,
  which differ between adjacent fields:
  - `vtxOffset` is a **byte** offset into the file; `vtxCount` is a **vertex count**. Divide by the
    36-byte stride to convert.
  - `idxOffset` is a **byte** offset (vertex-section length plus the mesh's local index offset), not
    an element index.
  - Each mesh's indices are **local, 0-based within its own vertex block**. An absolute vertex id is
    `index + vtxOffset/36`.
  - `submeshes[].idxStart` / `idxCount` are in **index elements**, local to that mesh's index block.
- **instance** stride 80 (16-byte aligned for storage-buffer reads): `affine f32x12 @0` (row-major
  world 3x4, shear included), `meshId u32 @48`, `lodGroup i32 @52`, `lodIndex i32 @56`,
  `rootId u32 @60`, `flags u32 @64`, `par u32 @68`, `par2 u32 @72`, `lv u32 @76`.
  Flags: `0x1` MIRROR, `0x2` TERRAIN, `0x4` BAKED_WORLD, `0x8` INACTIVE.
- **collider** stride 96: `affine f32x12 @0`, `kind u32 @48` (0 box, 1 sphere, 2 capsule, 3 mesh),
  `meshId i32 @52` (index into `manifest.colliderMeshes`, else -1), `center f32x3 @56` (Unity
  `m_Center`, collider-local), `shape f32x3 @68`, `layer u32 @80`, `flags u32 @84`, then **8 bytes of
  trailing pad @88**. The pad is what makes the stride 96; emitting 88-byte records desynchronises the
  whole buffer. `shape` is per-kind: box = `m_Size` xyz, sphere = `(radius, 0, 0)`, capsule =
  `(radius, height, direction)`. Shapes stay in Unity's local parameterisation, so a primitive only
  survives a global matrix that is a signed permutation.
  The physics world is mostly invisible - on Interchange, 131,945 of 141,347 colliders have no
  renderer - and it, not the render set, is what the nav bake consumes.
- **SH volume**: `volume.bin` is float16 LE, probe-major, probe index `((z*ny)+y)*nx + x`, 12 halfs
  per probe ordered `c0.r,c0.g,c0.b, c1.r,c1.g,c1.b, c2.r,c2.g,c2.b, c3.r,c3.g,c3.b`. `volume.json`
  carries `min`, `max`, `dims` = `[nx, ny, nz]`, `spacing`, and the layout string; a world point maps
  to a probe by `(p - min) / spacing`.

  The coefficients are RADIANCE in the L1 real basis, and **the coefficient order is not xyz**:

  ```
  c0 = Y00  (0.282095)       c1 = Y1-1 (0.488603 * y)
  c2 = Y10  (0.488603 * z)   c3 = Y11  (0.488603 * x)
  ```

  Irradiance comes from cosine convolution (`A0 = pi`, `A1 = 2pi/3`), which the viewer folds into
  `E/pi = 0.282095*c0 + 0.325735*(c1*n.y + c2*n.z + c3*n.x)`. Mapping c1/c2/c3 onto x/y/z instead
  tilts all indirect lighting by a fixed rotation.

  `volume_valid.bin` is u8 per probe on the same index: `255` = valid, `0` = inside geometry (backface
  ratio >= 0.25). Treating a missing validity file as all-valid leaks light through walls.

Ancestry ids (`par`, `par2`) are Unity `path_id`s folded to u32 by `(x ^ (x >> 32)) & 0xFFFFFFFF`,
level-local. `x` is a **signed int64** and the shift is **arithmetic** (`assemble_bevy.py:118`).
Negative `path_id`s are common, so a reimplementation that folds with a logical shift on a u64
produces different keys and the instance-to-gamedata join silently returns nothing.

## Failure signatures

| Symptom | Diagnosis |
|---|---|
| Whole map mirrored about X | No handedness flip applied (Rule 2). |
| Map inside-out, undersides lit | Flip applied as premultiply `G·M` instead of conjugation. |
| "Position is fine but slices are mirrored"; textures read right-to-left | Mesh vertices reflected AND the matrix conjugated - net `G·M·v`. Remove the vertex reflection. |
| Buildings skewed, floors misplaced, ~4% of instances wrong | A sheared world matrix was TRS-decomposed. |
| Terrain hundreds of metres off; terrain bbox does not envelope the buildings | Terrain vertex frame differs from mesh frame, or a legacy terrain-only flip/shift is still applied on top of the conjugation. |
| Decals on the wrong side of the map, roughly mirrored about the map centre | `G` applied in both the extractor and the assembler. |
| Every texture upside-down (not mirrored) | Missing V-flip, or the loader re-flipped what was already baked. |
| Lighting reads from the wrong side on bumpy surfaces | Normal-map green channel not flipped (DirectX convention). |
| Chain-link, foliage, grates render solid with black holes | Alpha channel dropped on texture export, or the material ignored `role`/`cut` and shipped OPAQUE instead of MASK. |
| Decals lying flat on the ground / rotated 90 degrees | Decal quad built in local XY; the projector spans X and Z and projects along Y. |
| Decal shows the wrong row of the atlas (wrong word) | The atlas pixel rect's V was flipped; its origin is already bottom-left. |
| Half a decal missing behind a plate | Flat quad emitted instead of projecting and clipping against receiving geometry. |
| Extraction "hangs" for hours; `py-spy dump` always inside `read_typetree`; CPU idle, no disk writes | A parent-chain walk memoized per leaf instead of per node - quadratic on deep hierarchies. |
| All lights import as Point, or all intensity 0 | Read via object attribute access (silently defaults) instead of `read_typetree()`; newer maps serialize `m_Intensity=0` and drive lamps from a controller MonoBehaviour. |
| Interiors black or flat | No `*_Light` scene extracted and no SH volume baked. EFT ships no lightmaps, no probes, no reflection probes, and no directional sun - if you did not bake it, it does not exist. |
| GPU device loss / driver reset during a bake or a frame | A shader read a GPU struct at the wrong stride. Pin struct sizes in EVERY shader and clamp bindless indices; get the field log before theorizing. |
| Rebuild does not fix a corrupt mesh; the OBJ is full of NUL bytes | A size check cannot see NTFS preallocation zeros. Completeness must be content-checked, and the damaged file dropped before re-export. |
| A MonoBehaviour payload decodes to garbage | The header size was computed without 4-byte-aligning the name length. |
| Material lookups all return None while the assets exist | A local variable named `env` shadowed by closure at call time, so lookups searched the wrong file. |
| Pack loads without error and renders garbage | The manifest was written before the blobs, or a mid-build failure left new blobs under an old manifest. Assembly must stage and swap atomically. |
| `[BUILD OK]` but lighting/routing/intel describe older geometry | An optional stage failed and the previous build's `volume.bin` / `nav.bin` / `gamedata.json` migrated across the atomic swap. Compare each sidecar's mtime against that stage's start time. |
| "Fixed it but still broken" | The viewer loaded a stale cached artifact with the same filename. Verify mtime and structure of the file actually loaded. |
| A whole named group of props vanished | A cull denylist over-matched a real content root that also passed the allowlist. The allowlist must short-circuit the denylist. Find the blast radius by diffing kept scene entries against emitted node names, bucketed by root. |
| An offline verifier disagrees with the renderer | Trust the real renderer. Re-implementations of a compressed or quantized format read garbage and burn hours. |
| A known-answer test passes on visibly wrong output | The test's ground truth was computed with the same convention as the build. Truth must be derived independently. |

## Invariants worth asserting

- `det(M')` equals `det(M)` for every instance after the global transform.
- Terrain's world bbox envelopes the building footprint on every outdoor map.
- Unique texture names used by the geometry exporter and the texture exporter come from the SAME
  filtered instance set; a shared id-to-name table is the only safe way to keep them in sync.
- Every submesh's face span is consumed (`f0 += n`) even when the submesh is skipped, or all
  subsequent submeshes read shifted triangle ranges.
- Non-finite floats occur in the game data (a `LODGroup` with `fadeTransitionWidth = NaN`);
  sanitize to 0.0 and report, do not let them reach the manifest.
- Mesh names include Cyrillic. Write every text file as UTF-8; a cp1252 encoder truncates the file
  to zero bytes and silently drops the geometry.

## Old patterns

The web/glTF delivery tail this pipeline used to end with is gone: no 512 px texture downscale, no
KTX2/ETC1S/UASTC transcode, no `gltf-transform quantize`/`meshopt`, no glb splitting, no
`EXT_mesh_gpu_instancing` TRS split. Textures are referenced full-resolution and imported as BC7
(albedo, emissive; sRGB) and BC5 (normal; linear) on load.

If you are targeting glTF anyway, two constraints from that era still hold: `EXT_mesh_gpu_instancing`
stores only translation/rotation/scale and therefore **cannot represent shear**, and mesh
quantization strips `node.matrix`, collapsing matrix-placed instances to the origin. Sheared and
mirrored instances must be baked to world geometry for that target. The `.eftpack` instance buffer
has neither limitation, which is why it stores the full 3x4 affine.