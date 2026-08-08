## Contents

- [1. What the subsystem does](#1-what-the-subsystem-does)
- [2. Mesh readout: Unity level → OBJ](#2-mesh-readout-unity-level--obj)
- [3. The X-negation vertex convention, and why](#3-the-x-negation-vertex-convention-and-why)
- [4. scene.json - exact instance schema](#4-scenejson--exact-instance-schema)
- [5. THE PLACEMENT FORMULA](#5-the-placement-formula)
- [6. Handedness: the conjugation, and the two wrong variants](#6-handedness-the-conjugation-and-the-two-wrong-variants)
- [7. Shear, non-uniform scale, mirrors, normals](#7-shear-non-uniform-scale-mirrors-normals)
- [8. Terrain extraction](#8-terrain-extraction)
- [9. LOD extraction, `--alllod`, `--keep-lods`, shell dedup](#9-lod-extraction---alllod---keep-lods-shell-dedup)
- [10. Structural culls](#10-structural-culls)
- [11. Parallel extraction and merge invariants](#11-parallel-extraction-and-merge-invariants)
- [12. Consumer-side binary layouts](#12-consumer-side-binary-layouts)
- [13. Invariant → failure-signature table](#13-invariant--failure-signature-table)
- [14. Environment knobs](#14-environment-knobs)
- [15. Old patterns](#15-old-patterns)

**Annotations.** *(measured)* marks a number recorded in a source comment - traceable to that comment, not re-derivable from the code. *(runtime-observed)* marks an on-screen symptom: the code path is confirmed in source, the rendered outcome is not statically checkable.

---

## 1. What the subsystem does

Three stages, three on-disk contracts:

| Stage | Script | Output |
|---|---|---|
| extract | `extraction/unity/eft_extract_v2.py` | `<dataset>/meshes/*.obj`, `<dataset>/tex/*.png`, `<dataset>/terrain_layers/`, `<dataset>/scene.json` |
| parallelize | `extraction/unity/extract_parallel.py` | same dataset, produced by N chunk processes + a merge |
| assemble | `eft_pipeline/assemble_bevy.py` (+ `tarkmap_core/{instmath,objio,culls,matsig}.py`) | `<pack>.eftpack/{manifest.json,meshes.bin,instances.bin,materials.json}` |

Units are **metres** throughout (Unity scene units; EFT authors 1 unit = 1 m). Angles never appear - rotation only ever exists as a 3×3 block. Textures are PNG, referenced by *stem* (no extension) in `scene.json` and by full path in `materials.json`.

Invocation: `--levels 52,54,55,...` (Unity scene files `<Game>_Data/level<N>`), `--name <dataset>`, plus `--alllod`, `--terrain-only`, `--terrain-step N` (default **2**, `eft_extract_v2.py:786`), `--data-root <Game>_Data`.

---

## 2. Mesh readout: Unity level → OBJ

**Selection.** Every `MeshRenderer` / `SkinnedMeshRenderer` object in the level (`eft_extract_v2.py:1606-1608`). The mesh PPtr is `smr.m_Mesh` for skinned renderers, else the sibling `MeshFilter.m_Mesh` found by walking `GameObject.m_Component` (`:1621-1628`). A renderer with an **empty `m_Materials`** is skipped (`:1647-1650`) - Unity runs no shader for it; 1,624 of 207,713 on Interchange *(measured)*, all invisible placeholders (AreaLight quads, lantern proxies).

**Identity / dedup key.** `(level, pptr.file_id, pptr.path_id)` (`:1216`). One OBJ per unique mesh; every instance references it by filename:

```
meshes/<san(m_Name)>__<lv>_<fid>_<pid>.obj        # san(): non-alnum outside "._-" -> '_'  (:328-329,1220)
meshes/terrain_<lv>_<san(TerrainData.m_Name)>.obj # terrain tiles                          (:1713)
tex/<san(m_Name)>__<sourceFileStem>_<sourcePathId>.png                                     (:930)
```

`san()` is `"".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))` (`:329`). Python's `str.isalnum()` is **Unicode-aware**, so Cyrillic, Greek and CJK characters are *preserved* in the filename - only non-alphanumerics outside `._-` become `_`.

Textures key on **source identity** (`srcid`, `:914-924`): the assets file the object actually lives in plus its own `path_id`, because `PPtr.file_id == 0` means "same file as the referrer" and collides across materials (~7 % wrong-texture rate if keyed naively *(measured)*).

**OBJ writer (UnityPy `UnityPy/export/MeshExporter.py`), byte-level:**

- `g <mesh name>` first line, then one `g <name>_<submeshIndex>` before each submesh's faces (`:25,:51`).
- `v {-pos.x:.9G} {pos.y:.9G} {pos.z:.9G}` - **X negated** (`:33`).
- `vt {uv0.x:.9G} {uv0.y:.9G}` - raw Unity UV, origin **bottom-left**, never flipped here (`:39`).
- `vn {-n.x:.9G} {n.y:.9G} {n.z:.9G}` - X negated (`:45`).
- `f {c+1}/{c+1}/{c+1} {b+1}/... {a+1}/...` - 1-based, all three indices equal, **triangle order reversed** (a,b,c → c,b,a) (`:54`).
- Returns Python `False` (not a string) when `m_VertexCount <= 0` or vertices are absent (`:22-23,:29-30`); the extractor treats that as "genuinely empty mesh" and writes nothing (`eft_extract_v2.py:1240-1244`).

Files are written **atomically** (temp + `os.replace`) with `encoding='utf-8'` (`:1239`). UTF-8 is load-bearing precisely because `san()` passes Cyrillic through: EFT ships mesh names like `Сontainer_hospital`, and a cp1252 encoder raises *after* `open('w')` truncated the file, leaving a 0-byte stub that a naive `exists()` guard reuses forever.

**Reuse guard.** `_obj_complete()` (`:177-201`) accepts a file only if its first 2 bytes are `g ` or `v ` **and** its last byte is `\r` or `\n`. A size check is not sufficient: a killed run leaves NTFS preallocation - the directory entry carries the final size while the data was never flushed, so the file is megabytes of NUL and every later run "reuses" it (streets carried 8,962 such OBJs / 968 MiB, referenced by 47,235 instances *(measured)*). Same idea for PNG (`_png_complete`, IEND in the last 16 bytes, `:159-174`) and for the vertex-colour sidecar (`_vcol_usable`, rejects all-zero, `:203-214`).

**Vertex colours.** OBJ drops colours, so when vertex-data channel 3 has non-zero dimension the mesh is re-decoded through `MeshHandler` and `m_Colors` is written as `<mesh>.vcol.npy`, shape `(-1,4)` float32 (`:1260-1272`). Consumer normalizes 0..255 → 0..1 if `max > 1.5` (`objio.py:159`). These are Vert-Paint blend weights, not tint.

**Submesh table.** Per-submesh triangle counts come from `m_SubMeshes[i].triangleCount`, falling back to `indexCount // 3` (`:1289`), and the material list is `mp = mats[min(i, len(mats) - 1)] if mats else None` (`:1303`). Because the OBJ emits faces in submesh order, submesh *i* owns face rows `[f0, f0+n)` of the OBJ, with `f0` a running cursor. **Invariant:** `sum(n_i) == number of f lines`. Broken → `assemble_bevy.py:989-992` prints `submesh span overruns OBJ tris` and silently drops the tail of that mesh.

**OBJ parsing (consumer).** `objio.load_obj` returns `(V[nv,3] f32, VT[nt,2] f32, F[nf,3,2] i32)`, where `F[...,0]` is the 0-based vertex index and `F[...,1]` the 0-based uv index (`-1` if absent). The bulk parser bails to the line-by-line parser on any non-`a/b/c` face, `//`, or a `v`/`vt` line without exactly 3/2 tokens (`objio.py:30-78`). Empty `vt` yields `VT = zeros((1,2))` so face uv indices can be clamped.

A binary sidecar `<mesh>.obj.msh` caches the parse: 8-byte magic `EMSH1\0\0\0`, `int64 mtime_ns`, `int64 size`, `uint32 nv, nt, nf` (36-byte header), then `V` f32×3, `VT` f32×2, `F` i32×3×2, native endianness, invalidated by the OBJ's `(mtime_ns, size)` (`objio.py:88-124`). It is a local cache - do not ship it.

---

## 3. The X-negation vertex convention, and why

Unity's basis (+X right, +Y up, +Z forward) is **left-handed**: `x̂ × ŷ = −ẑ`. glTF / Blender / wgpu-class renderers are **right-handed**, Y-up. The minimal change of basis is a reflection in one axis; this pipeline uses

```
G  = diag(-1, 1, 1, 1)        # 4x4, config default, eft_pipeline/tarkmap_core/config.py:121-126
G3 = diag(-1, 1, 1)           # G^-1 = G, det(G3) = -1
```

Negating X (rather than Z) is not arbitrary - it is the convention **UnityPy's own OBJ exporter already applies**, so mesh vertices arrive pre-multiplied by `G3` for free. Everything else in the dataset must match that choice exactly once:

| data | who applies `G3` | anchor |
|---|---|---|
| mesh vertices / normals | UnityPy `mesh.export()` | `MeshExporter.py:33,45` |
| mesh triangle winding | UnityPy reverses `a,b,c → c,b,a` | `MeshExporter.py:54` |
| terrain grid vertices | the extractor writes `-x` by hand | `eft_extract_v2.py:757` |
| instance matrices | the assembler conjugates | `instmath.py:22` |
| LODGroup centers (points) | `G3 @ c`, single application | `assemble_bevy.py:1263` |
| collider `m_Center` (generated primitive) | `G3 @ center`, single application | `assemble_bevy.py:1237-1238` |
| collider box `size`, sphere `radius`, capsule axis | **nothing** - invariant under a signed permutation | `assemble_bevy.py:1232-1236` |

The reflection is *not* an artistic mirror. It is the LH→RH basis change: a right-handed renderer fed `G3`-transformed data produces the same image the game does. Reversing the winding alongside the reflection is what keeps front faces front.

**Collider case is the cleanest statement of the rule:** mesh colliders get the flip free (their OBJ verts are already `G3`-applied), while box/sphere/capsule geometry is *generated at load time from `center`/`shape`*, so `center` must be flipped explicitly. Omitting it mirrors each primitive about its own pivot - 2,704 misplaced nav colliders on Interchange, up to 4.02 m out *(measured)*. The assembler refuses to emit colliders at all if `G3` is not a signed permutation, because `shape`/`direction` cannot be re-expressed under a rotational global matrix (`assemble_bevy.py:1154-1162`).

---

## 4. scene.json - exact instance schema

Top level (`eft_extract_v2.py:1872-1876`):

```jsonc
{
  "instances":  [ ... ],          // array, order = extraction order (level by level)
  "up":         "unity",          // string literal; Y-up, LEFT-handed source frame
  "levels":     [52, 54, ...],    // int level ids that produced this file
  "lodGroups":  [ ... ],          // global table; instance.lod.g indexes it
  "lod_schema": 1,                // int
  "waterBodies":[ {"lv":int,"y":float,"mesh":str} ],  // absent in parallel-merged datasets
  "note":       "OBJ verts are UnityPy X-flipped+winding-reversed; builder must un-flip"
}
```

### instance object

| field | type | meaning |
|---|---|---|
| `mesh` | str | filename in `meshes/`, no directory part |
| `m` | float[16] | **row-major 4×4** Unity world matrix, `numpy.flatten()` order, each element `round(v,5)`. Row 3 is always `[0,0,0,1]` |
| `subs` | list[sub] | one per submesh, in submesh order |
| `lv` | int | source level |
| `kind` | `"mesh"` \| `"terrain"` | terrain instances have no `root`/`cast`/`renON`/`aih` |
| `root` | str | top-most ancestor GameObject name (`root_of`, `:890-900`) - the scene-tree grouping the culls key on |
| `cast` | int | Unity `m_CastShadows`: 0 Off, 1 On, 2 TwoSided (**drawn**), 3 ShadowsOnly |
| `renON` | bool | `Renderer.m_Enabled` |
| `aih` | bool | activeInHierarchy = own `m_IsActive` AND every ancestor's (`:903-912`); serialized `m_IsActive` is **local only**, so the parent walk is mandatory - it changes the verdict for >50 % of instances |
| `drop` | bool | `cast==3 or not renON or not aih` (precomputed hint) |
| `par` | int64 | immediate parent Transform `path_id` (level-local), optional |
| `par2` | int64 | grandparent Transform `path_id`, optional |
| `lod` | `{"g":int,"i":int}` | LODGroup index into `lodGroups` + LOD level index, optional |
| `wlayer` | 1 | water-surface tag; overrides the root denylist. Written by **two** passes - see below. Optional |

`oversize_inactive` is **not** written by the extractor - the cull stage adds it in memory (`culls.py:123`).

**`wlayer` is not a layer test.** Two passes write it:

- **Pass 1 - layer 4** (`:1477-1522`): MeshFilters on Unity layer 4 (Water); the extractor synthesises a water instance from the mesh.
- **Pass 2 - SIBLING SURFACES** (`:1535-1592`): **layer-0**, material-less `MeshRenderer`s whose immediate parent Transform *also* parents a layer-4 `MeshCollider`/`MeshFilter` (`:1542,1562,1591`). This is how woods' ponds, river segments and swamp pools ship: `Shoreline_Lake_Water_02_LOD0` is layer 0 and only its `_BALLISTIC_water` sibling is layer 4, so a layer-only test tags the collider and misses the visible surface. The second class is layer-0 geometry inferred **structurally** from a layer-4 collider sibling.

### sub object

Always: `n` (int triangle count, or `-1` = "all remaining faces"), `tex` (albedo PNG stem or null), `nrm`, `sh` (shader name string), `uv` `[scaleX,scaleY,offsetX,offsetY]` from the albedo slot's `m_Scale/m_Offset`, `col`, `role`, `cut` (`_Cutoff`, default 0.5).

`col` is **3 or 4 elements**. The initialiser and both failure paths emit `[1.0, 1.0, 1.0]` (`:972,1005,1213`); the 4-element `[r,g,b,a]` form is written only when the material actually carries `_Color`/`_BaseColor`/`_TintColor`/`_MainColor` (`:1010-1016`), so any consumer reading `col[3]` must guard. No colour space is declared for `col` anywhere in the extractor or assembler - `manifest.conventions.colorSpace` (`assemble_bevy.py:1334`) labels **textures**, not the material tint.

`role` is decided from Unity's authoritative `RenderType` subshader tag plus render queue (`:1029-1047`): `transparentcutout → "cutout"`; `transparent → "glass"` if `m_CustomRenderQueue >= 2900` else `"decal"`; otherwise `"opaque"`; a shader name containing `dithered`+`transparent` is forced to `"glass"`; any shader naming `water` is forced to `"water"`.

Optional, presence-gated (absent unless the material authors them): `vp` (3-layer Vert-Paint splat: `layers[].{tex,nrm,uv,col}`, `heights`, `blend`, plus soft-cutout `astr/acut/ahgt`), `emis`, `emisCol` (HDR, only when Unity emission is actually enabled - `_EMISSION` keyword or an `Emissive` shader variant, `:952-968`), `gloss`, `metal`, `bumpScale`, `spec`, `smA`, `glassTRS`, `reflCube`, `reflCol`, `specCol`, `shin`, `opacS`, `detA/detAuv/detAI`, `detN/detNuv/detNS`, `detMask`, `par`/`parS` (parallax height + amount).

Water surfaces synthesised by either pass carry `subs = [{"role":"water","sh":null,"tex":null}, ...]` with **no `n`** (`:1503-1504`; the sibling pass builds the same shape), which the consumer's `sb.get('n',-1)` resolves to "all remaining faces" for the first entry and 0 for the rest.

### lodGroups entry (`:1376-1383`)

`{"size": m_Size × max‖world column‖ (Unity lossyScale rule), "center": [x,y,z] in **Unity world**, "fadeMode": int, "lastIsBillboard": bool, "srh": [screenRelativeHeight per level], "ftw": [fadeTransitionWidth per level], "n": level count, "cullH": srh[-1] (billboard groups only)}`.

Three values are **derived**, not verbatim Unity: `size = m_Size × wscale`; `center = M3 @ m_LocalReferencePoint + T`, i.e. the world transform of Unity's *local* reference point (`:1354-1356`); and `lastIsBillboard`, which is Unity's `m_LastLODIsBillboard` **overridden to False** whenever the last level resolves to any Mesh/SkinnedMeshRenderer (`:1363-1367`, §9.1). Everything is rounded on write - `size`/`center` to 4 decimals, `srh`/`ftw` to 5 (`:1376-1381`) - and passed through `_fin` NaN coercion (`:1370-1375`): Reserve ships a `fadeTransitionWidth = NaN`, and `x or 0.0` does not catch NaN (NaN is truthy).

---

## 5. THE PLACEMENT FORMULA

The Unity world matrix is built by chaining local TRS 4×4s up the `Transform.m_Father` chain, `W(node) = W(father) @ trs(node)`, memoized per node, 256-deep guard (`eft_extract_v2.py:836-849`); `trs()` is `M[:3,:3] = quat_to_mat(m_LocalRotation) @ diag(m_LocalScale)`, `M[:3,3] = m_LocalPosition` (`eft_scene_extract.py:33-40`).

Given a scene.json `m` (row-major 16) and its conjugated form `mg = apply_global(m)`:

```
M3 = [[mg0, mg1, mg2],
      [mg4, mg5, mg6],
      [mg8, mg9, mg10]]          # rows
T  = [mg3, mg7, mg11]

world_position = V @ M3.T + T          # V = OBJ vertices, (nv,3)
world_normal   = normalize( N @ inv(M3) )      # == normalize( inv(M3).T @ n_col )
```

`_M3T` at `assemble_bevy.py:484-488`. The world bake is the same pair: `instmath.py:61` sets `M3iT = inv(M3).T`, `:66` computes `wn = nrm @ M3iT.T` - i.e. `N @ inv(M3)` - and `:58` builds `M3`/`T` from the same indices. The row-vector convention is what forces the pairing: `p' = p @ M3.T` goes with `n' = n @ inv(M3)`. Writing `N @ inv(M3).T` is the *transpose* of the correct expression and, for a pure rotation, applies the **inverse** rotation to normals. Row-major storage with the translation in column 3 is why the row indices are 3/7/11 and why the pack keeps only the first 12 floats.

**TRS decomposition is forbidden.** `instmath.trs` exists only for the legacy glTF `EXT_mesh_gpu_instancing` path and it *measures its own invalidity*: `S = ‖column‖`, `R = M3 / S`, and `ortho = max|Rᵀ R − I|` (`instmath.py:36-39`). `ortho ≥ 0.02` means the columns are not orthogonal - the matrix carries **shear**, which no (T,R,S) triple can represent. EFT produces this routinely (a parent with non-uniform scale composed with a child rotation; Interchange's mall floors are the canonical case). Decomposing drops the off-diagonal term silently: the object un-skews back to square, so a sheared floor slab or ramp stops meeting its walls and leaves a wedge-shaped gap that grows with the slab's length *(runtime-observed)*. There is no error, no warning, and the geometry looks individually plausible - which is why the rule is absolute: **apply the raw 3×3; never decompose.**

Two escape hatches, in order of preference:

1. Emit the full row-major 3×4 per instance and let the renderer multiply (`assemble_bevy.py:1072`). Correct for shear *and* mirror.
2. Only when the 3×3 is genuinely **rank-deficient** - a mesh flattened to a plane, e.g. a baked billboard/decal - bake the geometry to world with `instmath.bake_into` (`:55-69`), which falls back to `pinv` for the normal matrix. `_degenerate` (`assemble_bevy.py:491-499`) gates this: `|det| > (max|M3|)³ · 1e-9` ⇒ invertible; otherwise SVD and singular iff `s[0] <= 0 or s[-1] < s[0]·1e-6`. Baked geometry is emitted once under an identity affine.

---

## 6. Handedness: the conjugation, and the two wrong variants

Let `v_u` be a Unity mesh-local vertex, `M` its Unity world matrix, and `V = G3 v_u` the OBJ vertex (already reflected by the exporter).

**Correct - similarity conjugation** (`instmath.make_conjugator`, `instmath.py:13-26`, applied at `assemble_bevy.py:1061`):

```
M' = G · M · G⁻¹          (G⁻¹ = G for a diagonal reflection)
p  = V @ M3'.T + T'  =  G3·M3·G3 · (G3 v_u) + G3·T  =  G3 · (M v_u)
```

The inner `G3·G3 = I` cancels: the mesh is un-reflected into Unity local space, transformed by Unity's own matrix, and the **whole scene is reflected exactly once, globally**. Consequences that must hold:

- `det(M') = det(G)·det(M)·det(G)⁻¹ = det(M)` - instance determinants are **preserved**, so nothing becomes inside-out.
- No surface is locally mirrored - textures, decals and lettering read correctly.
- `apply_global` is a no-op when `G == I` (`GID`, `:16,22`), so an identity global matrix costs nothing.

The *on screen* paragraphs below are *(runtime-observed)*: the algebra and the code paths are confirmed in source, the rendered outcomes are recorded observations.

**Wrong variant A - premultiply only (`M' = G·M`).** `p = G3·M3·(G3 v_u) + G3·T`: an extra `G3` survives *inside*, mirroring every mesh about its own local X=0 plane. `det(M') = −det(M) < 0` for every instance.
*On screen:* the map layout, pivots and terrain are all in the right places, but every object is mirrored on the spot - signage and license plates read backwards, door handles and stairs are handed wrong, modular wall kits no longer tile at their seams - and with backface culling on you see through the front of every shell into its interior (or, if the renderer also honours `det<0` by flipping front-face, you get solid but uniformly mirrored geometry).

**Wrong variant B - conjugate *and* reflect the mesh verts (`V' = V @ G3.T`).** Algebraically identical to A: `G3 V = G3 G3 v_u = v_u`, so `p = G3·M3·G3·v_u + G3·T`. This is the historically recurring regression (`instmath.py:5-8`). It is the more dangerous of the two because the global layout is *perfect*, so it survives a wide screenshot review and only shows up on lettering and asymmetric props.

**Wrong variant C - no global fix (raw `M` on flipped verts).** `p = M3·(G3 v_u) + T`: the world stays in Unity's left-handed layout while every mesh is locally mirrored. Modular kits stop tiling, and the synthesized terrain (whose vertices *are* `G3`-applied, §8) is now in a different frame from the meshes - terrain and buildings separate by up to the terrain's own width (hundreds of metres). This is exactly the failure the extractor's terrain comment warns about (`eft_extract_v2.py:748-752`): writing raw `+X` terrain vertices makes the global flip move terrain and meshes in opposite directions.

Diagnostic that separates all four in one shot: pick one asymmetric, text-bearing asset (a shop sign, a road marking) and one modular corridor. Correct → text reads, corridor tiles. A/B → text mirrored, corridor still tiles (it was mirrored as a unit). C → text mirrored *and* corridor seams gap.

---

## 7. Shear, non-uniform scale, mirrors, normals

- **Non-uniform scale** is common and legal; the only requirement is transforming normals by the normal matrix `inv(M3).T` - as a row vector, `n @ inv(M3)` (§5) - and re-normalizing (`instmath.py:61,66`, and `manifest.conventions.normals` at `assemble_bevy.py:1330`). Using `M3` directly on normals tilts them proportionally to the anisotropy - surfaces light as if bent, worst on stretched pipes/railings *(runtime-observed)*.
- **Shear** is detected but never repaired: `ortho = max|RᵀR − I| ≥ 0.02` (`instmath.py:39`). Any consumer that stores a per-instance affine (glam `Affine3A`, a 4×4 node matrix, a Blender `matrix_world`) handles it natively. Anything that stores loc/rot/scale does not - bake to world instead.
- **Negative determinant.** BSG mirrors props with a negative `m_LocalScale` component. `det3(mg) < 0` survives the conjugation, and the assembler sets `FLAG_MIRROR = 1<<0` (`assemble_bevy.py:1068,125`) rather than baking. The renderer must respond by **flipping front-face/winding for that instance only**. If it does not, those instances render inside-out: you see the interior/back faces, lighting comes from behind the surface, and thin objects appear to vanish at grazing angles *(runtime-observed)*. The baked path does the equivalent by reversing the index order: `ind[:, [0,2,1]]` when `det(M3) < 0` (`instmath.py:59,67`). The same helper exists standalone as `revwind(F) = F[:, [0,2,1], :]` (`instmath.py:29`).
- **Do not** absorb a reflection into scale (`S[0] *= -1; R[:,0] *= -1`, `instmath.py:38`) unless you also flip winding - that line is a decomposition artifact, not a placement strategy.

---

## 8. Terrain extraction

Unity terrain has **no mesh asset** - it is a heightmap plus splat weights, so the pipeline synthesizes both geometry and an albedo.

### 8.1 Heightmap → vertex grid (`write_terrain_obj`, `eft_extract_v2.py:717-775`)

```
res    = TerrainData.m_Heightmap.m_Resolution           # e.g. 1025 (2^k + 1)
scale  = m_Heightmap.m_Scale                            # metres per heightmap cell in x/z; y = FULL height range
H      = asarray(m_Heights, float64).reshape(res, res)  # H[row, col];  row -> +Z, col -> +X   (:727)
Hw     = (H / 65535.0) * 2.0 * scale.y                  # metres                               (:728)
Hs     = Hw[::step, ::step];  rr, cc = Hs.shape         # step = --terrain-step (default 2)
sizeX  = (res - 1) * scale.x ;  sizeZ = (res - 1) * scale.z
```

The read casts straight to float64, so the on-disk element dtype is not visible in this code path; the `/65535.0` divisor is consistent with an unsigned 16-bit field *(unverified)*. The **`* 2.0`** is confirmed (`:720-723,728`) and is Unity's "16-bit field, 15 bits used" quirk: a stored 32767 means full `scale.y`. Omit it and every terrain is exactly half height - the map sinks tens of metres below its own buildings, with props floating over the ground *(runtime-observed)*.

Emitted lines, in this exact order (`:754-771`):

```
v  {-(c*step*scale.x):.4f}  {Hs[r,c]:.4f}  {(r*step*scale.z):.4f}     for r in 0..rr-1, c in 0..cc-1
vt {c/(cc-1):.5f} {r/(rr-1):.5f}                                       same traversal order
f  b/b d/d a/a          and          f e/e d/d b/b
        a = r*cc + c + 1        b = r*cc + (c+1) + 1
        d = (r+1)*cc + c + 1    e = (r+1)*cc + (c+1) + 1
```

No `vn`. Written atomically. Note the hand-applied **X negation** - this is what puts the terrain in the same frame as the UnityPy-exported meshes (§3). The face order `(b,d,a)` / `(e,d,b)` yields +Y face normals under `cross(p1−p0, p2−p0)` in that frame, i.e. terrain faces up.

**Terrain vertex frame vs mesh frame.** A mesh OBJ's local frame is the Unity mesh pivot with X negated; the instance matrix carries it to world. A terrain OBJ's local frame is the *Terrain component's* space: origin at heightmap sample (0,0), `x = −col·step·scale.x` (runs toward −X), `y` = metres above the tile origin, `z = +row·step·scale.z`. Its instance matrix is `world_of_go(Terrain.m_GameObject)` - in practice a pure translation to the tile origin - so under the conjugation only the X translation flips, and the tiles reassemble into a continuous sheet.

**Holes** (`:736-747,764-769`). `m_Heightmap.m_Holes` is a `(res-1)²` byte grid; a value `< 128` is a cut-out quad (tunnel mouths, bunker entries, pits). Decimation is **conservative**: a decimated quad is dropped if *any* full-resolution hole cell it spans is holed, so a tunnel is never re-filled (the cut edge is at most `step-1` cells wide). Skipping this fills tunnels with solid ground and objects inside bunkers read as floating over terrain *(runtime-observed)*. `EFT_TERRAIN_HOLES=0` disables the cut (`:738`) - a geometry change, not a speed knob (§14).

### 8.2 MicroSplat layers and the real tiling (`microsplat_uv_scales`, `:400-464`)

`TerrainLayer.m_TileSize` is **garbage** for MicroSplat terrains (grass `x=137.25`, `y=inf` *(measured)*). The real tiling lives in the game's MicroSplat material, in `sharedassets17.assets`, named `MicroSplat_<quality>_<season>`:

- `m_SavedProperties.m_Colors["_UVScale"].r` - one global float (**233.33** for EFT *(measured)*).
- `_PerTexProps` texture, **row 0, R channel**, indexed by texture-array slot (== TerrainLayer order). It must be reinterpreted with the dtype matching `m_TextureFormat`: **20 = RGBAFloat** (float32, 16 B/texel) or **17 = RGBAHalf** (float16, 8 B/texel). Any other format ⇒ reject and fall back to `m_TileSize`. Values are validated finite, `> 0`, `max < 1e6`.

```
rep_i     = _UVScale * perTex[i]        # repeats across the 0..1 terrain UV, SAME on u and v
tiledUV   = uv01 * rep_i
tile_size = terrainSize / rep_i          # metres; grass lands at ~1.76 m (measured)
```

Fallback when no MicroSplat material resolves: `repX = sizeX / tileSize.x`, `repZ = sizeZ / tileSize.x` - **axis-separate**, because a fixed metre tile repeats a different number of times on each axis when `sizeX != sizeZ`. Using `sizeX` for both is the historical "V-tiling wrong on non-square terrains" bug.

Season is voted from the layers' diffuse texture names (`_terrain_season`, `:368-397`), matching compound tokens first: `spring_early, autumn_late, summer, winter, spring, autumn`. A near-tie or a winner covering `< half` the layers logs `terrain season UNCERTAIN`.

### 8.3 Splat control maps and manifest (`export_terrain_splat`, `:666-714`)

Layer *i* reads control texture `i // 4`, channel `i % 4` (RGBA). Written per tile:

```
terrain_layers/ctrl_<tile>_<k>.png     # RGBA control maps, one per 4 layers
terrain_layers/layer_<diffuseName>.png # RGB layer diffuse, deduped by name across tiles
terrain_layers/manifest.json
  tiles[<tile>] = {ctrl_maps:[...], sizeX, season, uvscale,
                   layers:[{idx, name, ctrl, chan, cov, tileX, rep}]}
  layers = [sorted diffuse names]
```

A layer's diffuse is exported when `mean(alpha channel) >= 0.005` **or** `max(alpha) >= 0.5`. Mean-only pruning drops layers that are locally dominant in small patches (Reserve's Sand/Pebbles: 0.4 % mean, ~100 % inside their patches *(measured)*) while every tile still references them - the consumer then renders a placeholder exactly where those layers *are* the ground.

### 8.4 Baked albedo (fallback path, `:490-604`)

`R = 4096`, `ss = 2` supersampling. Control weights are sampled **once at pixel centres**, bilinear with clamp; the fine-tiled diffuse is gathered per supersample pass, bilinear with **wrap** (so the `mod 1.0` seam interpolates). Per pass: `u = mod(jj·repX, 1)`, `v = mod(ii·repZ, 1)`, `albedo += w·bilinear(diffuse, v, u)`, `wsum += w`. Texels with `wsum <= 1e-3` (no control coverage) are **not divided** - they are filled with the covered-area mean (fallback `[0.4,0.4,0.4]`), otherwise they come out black. Output PNG `tex/terrain_<lv>_<tile>_albedo.png`; the terrain instance's single sub is `{"n":-1, "tex":<albedo stem>, "nrm":null, "sh":"terrain", "uv":[1,1,0,0]}` (`:1742`). A GPU compositor (`atlas bake-terrain`) can replace the numpy pass - the dispatch is at `:607+` and the per-tile fallback at `:1756-1760`; whether the wgpu and numpy composites are pixel-identical is *(unverified - the two have not been diffed)*.

---

## 9. LOD extraction, `--alllod`, `--keep-lods`, shell dedup

### 9.1 Building the table (`:1339-1413`)

Iterate `LODGroup` objects via `read_typetree()`. For each: world matrix of its GameObject, `wscale = max(‖M3 column‖)`, `center = M3 @ m_LocalReferencePoint + T` (Unity world), `size = m_Size * wscale`. Then for level `li` and each `m_LODs[li].renderers[j].renderer.m_PathID`:

- `all_lod_rids` - every grouped renderer.
- `rid2lod[rid] = (globalGroupIdx, min li)` - a renderer listed at several levels collapses to its **finest**.
- `rid2levels[rid] = {(g, li)...}` - the **full span**, used only by `--alllod`.
- `group_min_lod[g] = min li that actually has renderers` - some groups ship an **empty LOD0 slot** (vehicles whose only geometry starts at LOD1).
- `billboard_only_rids` - renderers that appear *only* at a synthesized billboard level.

`m_LastLODIsBillboard` is **not** trusted on its own (`:1363-1367`): EFT sets it on groups whose last level is real `MeshRenderer`s, so the extractor overrides it to False when the last level resolves to any Mesh/SkinnedMeshRenderer. Believe the flag only in the negative case; otherwise those shells are erased past LOD0's ~18 m band.

### 9.2 Selection (`keep_renderer`, `:1415-1430`)

```
if rid in billboard_only_rids:  drop          # no billboard geometry is shipped
if --alllod:                    keep          # every level
if rid not in all_lod_rids:     keep          # ungrouped renderer
otherwise:                      keep iff li == group_min_lod[g]
```

Selecting LOD0 by the LODGroup PPtrs is authoritative; the old `_LOD0` **name substring** heuristic both dropped 16 generically-named LOD0 objects *(measured)* and kept lowercase `model_lod` shells stacked on top of each other. A mesh *named* `..._LOD1` can legitimately be a LOD0 renderer.

Under `--alllod`, a renderer spanning several levels is emitted **once per level**, each copy carrying its own `lod.i` and sharing `subs`/`m` by reference (`:1686-1695`). Without this, the renderer appears only at its finest level and disappears in the coarser distance bands, because the consumer derives each shell's distance window from the *present* set.

### 9.3 Distance windows (consumer side)

`far(level) = size / (2 · srh[level])`, `near(level) = far(previous **present** level)` (`viewer/src/render/gpu_driven.rs:1823-1834`). "Previous present", not `level-1`, so an internal gap in the present set cannot leave an undrawn band *(the band artefact itself is runtime-observed)*. Billboard-last groups cull past `size / (2 · cullH)`. `srh <= 1e-6` ⇒ infinite far.

### 9.4 Shell dedup (`assemble_bevy.py:805-899`; skipped by `--keep-lods`)

Bucket kept instances by `(lv, lod.g)`; groups with one distinct `lod.i` are untouched, as are all untagged instances (terrain, ungrouped). Then per group, ascending level:

1. `cover` starts as the world AABBs of the finest present level's instances.
2. An instance's world box is `corners(localAABB) @ M3.T + T`, and is `None` (draws nothing) if its mesh fails `load_obj` / has 0 faces, **or** if every submesh is rejected by `Culls.keep_submesh` - the same two calls the geometry loop makes, so "renders nothing" here and "silently skipped there" are one verdict.
3. A coarser level may be dropped **only if every instance on it** is enclosed by `cover`, with a **per-axis** tolerance: `eps = 1e-5 · (1 + maximum(|w[0]|, |w[1]|))` - an elementwise 3-vector built from the box's own lo/hi corners - applied componentwise in `(w[0] < clo − eps).any() or (w[1] > chi + eps).any()` (`assemble_bevy.py:856-857`). A scalar `max|corner|` would be a looser, uniform tolerance. The `1e-5` floor is scene.json's own 5-decimal rounding, deliberately far below real sub-metre overhangs.
4. If the finer levels draw nothing, the current level **is** the group's geometry: keep it and let it join `cover`.
5. **All-or-nothing per level**: if any instance on a level is uncovered, the whole level stays (its level-mates too). Keeping one instance of a level while dropping its mates leaves that distance band drawing a partial object - streets' trailer group kept a 92-tri door and dropped the 5,402-tri body, so the trailer vanished between ~7.4 m and ~24.7 m *(measured)*.

Fail-loud: if *every* probed mesh is unreadable, the build aborts rather than shipping a pack whose holes are indistinguishable from correct culling (`:883-886`). Dead meshes are listed in `lod_integrity.json`.

The naive rule ("keep `min(lod.i)`") assumes the finest shell exists and draws. On a dataset damaged by a crashed extraction that assumption is false, and the visible result is a see-through hole where the coarse shell used to be.

---

## 10. Structural culls (`eft_pipeline/tarkmap_core/culls.py`)

Config-driven per map (`cull` block), all fields optional.

**Root denylist** (`DEFAULT_DROP_ROOT_RE`, `:27-30`), matched against the instance's `root`:

```
(?i)^(decals?|.*_decals?|.*(?:day|night)_?light|.*audio.*|trig?ger.*|blocker.*|justplane.*|
 .*event.*|.*volume|reflectionprobe.*|lightprobe.*|skybox.*|spatialaudio.*)$
```

The light term is `(day|night)_?light`, **not** `.*light`: `*_Light` scenes hold real geometry (lamp fixtures, the vehicles whose headlights are the sources), so a bare `.*light` deletes thousands of instances.

**Allowlist `keep_root_prefix` is PROTECTION, not EXCLUSION** (`:143-145`). Its only job is shielding a map's declared-content roots (e.g. `SBG` on Interchange) from the generic denylist. Non-allowlisted roots still flow through the same denylist, exactly as on maps with no allowlist. The old "keep ONLY allowlisted roots" semantics silently deleted every real root lacking the prefix (Interchange: `New_mechanics` 828 instances *(measured)*, `STATIONARY`, `Power Plant`, ...).

**Submesh drops** (`keep_submesh`, `:147-166`):

| test | why |
|---|---|
| `sh.lower().startswith("shadow")` | shadow-caster proxies; invisible in game |
| `sh == "hidden/tree billboard lod"` | SpeedTree impostor → untextured white cross if kept |
| `"fogsheet"` or `"billboard_fog"` in `sh` | camera-facing fog planes → translucent "walls" |
| no `tex` **and** `sh ∈ {"", "?", "standard"}` | Unity occlusion/collision/stencil proxy boxes → solid grey boxes occluding real geometry. **Exception:** `role == "water"`, which is legitimately untextured and material-less |

**Unity-hidden instances** (`:102-123`): `cast == 3` (ShadowsOnly) or `renON is False` are never drawn by Unity → always dropped. `aih is False` is **kept** by default, because EFT activates much of that geometry at raid load (checkout counters, registers, crates); dropping it leaves baskets floating over vanished counters. Exceptions: purely-numeric mesh names (duplicate lane labels that z-fight), and instances whose world bounding diameter exceeds `inactive_keep_max_m` (default **10 m**) - parked scenery such as a disabled 140 m mountain copy - which are kept but tagged `oversize_inactive` for the consumer to hide.

**Alternate-state pairs** (`:216-265`). Destructible props ship as coincident state pairs (`..._glass` intact vs `..._glass_broken`) with `m_IsActive` selecting the raid-start state. Two structural rules, no name matching: (a) drop an inactive instance whose translation, rounded to 0.1 m in x/y/z, coincides with any active instance's; (b) drop an inactive instance sharing `par` or `par2` with an active one within **6 m** (`36.0` squared) - needed because the broken mesh is authored at the frame pivot and each intact pane at its own, ~3 m apart.

**Off-map backdrops** (`:168-192`). Terrain tile translations define the footprint; pad it by `offmap_pad_m` (**700 m**) and drop non-terrain instances whose translation lies more than `offmap_margin_m` (**300 m**) beyond that. On Interchange: real geometry is <100 u outside the padded footprint, the 10 skyline backdrops are ~780 u outside *(measured)*. Requires ≥4 terrain tiles, otherwise inert.

The assembler aborts if the cull keeps 0 or `< 0.5 %` of instances (`assemble_bevy.py:764-765`) - a mis-specified `cull` block is a build failure, not a quiet empty map.

---

## 11. Parallel extraction and merge invariants

`extract_parallel.py` splits the level list by greedy longest-processing-time bin packing on **level file size**, into up to `3 × jobs` chunks so a fixed-size pool work-steals (`:338-351`). Each chunk runs the unmodified single-process extractor into `<name>__p<idx>`. Correctness rests on exactly three properties (`:9-16`):

1. Mesh OBJ filenames are **level-scoped** and chunks hold disjoint levels ⇒ no collisions.
2. Texture and splat-layer PNGs are **source-identity scoped** ⇒ same content, same name ⇒ first writer wins.
3. `lod.g` is a per-chunk cumulative index ⇒ the merge offsets each chunk's instance `lod.g` by the running LODGroup count (`:241-247`). A dangling `lod.g >= len(lodGroups)` aborts the build (`:263-269`).

The merge is therefore **partition-invariant**: any chunk count produces the dataset a `jobs=1` run would. Two divergences to know: the merged `scene.json` is written without a `waterBodies` key (`:274-279`), and the single-process extractor additionally *merges* into an existing `scene.json` - a partial run owns only the levels it was asked for, keeps every instance from untouched levels, and offsets the new `lod.g` past the old table (`eft_extract_v2.py:1835-1867`). An unreadable existing `scene.json` is fatal, never overwritten.

---

## 12. Consumer-side binary layouts

All little-endian. Declared in the manifest so the reader hardcodes nothing.

**Vertex, stride 36** (`assemble_bevy.py:88-95`): `position f32x3 @0`, `normal f32x3 @12`, `uv f32x2 @24`, `color unorm8x4 @32`.

**Instance, stride 80** (`:102-116`): `affine f32x12 @0` (row-major 3×4, shear included), `meshId u32 @48`, `lodGroup i32 @52`, `lodIndex i32 @56`, `rootId u32 @60`, `flags u32 @64`, `par u32 @68`, `par2 u32 @72`, `lv u32 @76`. Flags: `0x1` MIRROR, `0x2` TERRAIN, `0x4` BAKED (identity affine, world-baked geometry), `0x8` INACTIVE. `par`/`par2` are Unity `path_id`s folded to u32 by `int((x ^ (x >> 32)) & 0xFFFFFFFF)` (`:118-122`). The parentheses are load-bearing: Python binds `&` tighter than `^`, so an unparenthesised `x ^ (x >> 32) & 0xFFFFFFFF` masks the *shifted term* instead of the result, leaves the value wider than 32 bits, and does not reproduce these ids.

**Collider, stride 96** (`:139-153`): `affine f32x12 @0`, `kind u32 @48` (0 box, 1 sphere, 2 capsule, 3 mesh), `meshId i32 @52`, `center f32x3 @56` (Unity-local, `G3`-applied), `shape f32x3 @68` (box `m_Size`; sphere `(r,0,0)`; capsule `(r,h,direction)`), `layer u32 @80`, `flags u32 @84`.

`meshes.bin` = the whole vertex section, then the whole index section (u32); each mesh's `idxOffset` is absolute, patched as `len(vertexSection) + localOffset` (`:1119-1123`).

**Geometry build, per submesh** (`:998-1024`): slice the OBJ face rows `[f0, f0+n)`; gather `pos = V[vertIdx]` and `uv = VT[uvIdx]` (uv index `< 0` → `[0,0]`); bake the material tiling `uv = uv*[sx,sy] + [ox,oy]`; then **V-flip `uv.y = 1 − uv.y`** because Unity's UV origin is bottom-left while PNG rows and the sampler are top-left (`manifest.conventions.uvVFlipBaked = true`, `uvTilingBaked = true` - `materials.json.uvXform` is reference only, do **not** apply it again). Face normals `cross(p1−p0, p2−p0)` are accumulated onto vertices deduped by the key `[round(pos,3), round(uv,3)]` (1 mm / 1e-3 UV weld) and normalized - that is where smooth shading comes from, since the OBJ's `vn` values are discarded. `COLOR_0` is the vert-paint weight for `vp` materials, opaque white `(255,255,255,255)` otherwise.

The `manifest["conventions"]` block (`:1328-1336`) holds exactly: `affine`, `normals`, `uvVFlipBaked`, `uvOrigin`, `uvTilingBaked`, `uvXformNote`, `normalMapGreenFlip`, `normalMapConvention`, `colorSpace`, `textureImport`. Normal maps are **DirectX** convention (green down) and must be G-flipped on import or negated in the shader; albedo/emissive are sRGB, normal/height linear (`:1333-1334`).

`doubleSided` is **not** in that block. It is a **per-material** field in `materials.json`, written at `assemble_bevy.py:348` and consumed by `viewer/src/eftpack.rs:438` and `viewer/src/render/standard.rs:383-384` - EFT's deferred pass draws building shells solid from both sides.

---

## 13. Invariant → failure-signature table

The right-hand column is *(runtime-observed)* throughout: the invariants and their code paths are confirmed in source, the rendered symptoms are recorded observations.

| Invariant | Broken → what you see |
|---|---|
| `world = V @ M3.T + T` with `M' = G·M·G⁻¹`, verts left raw | see §6: mirrored-in-place objects (A/B) or terrain/mesh separation (C) |
| Never TRS-decompose a matrix with `ortho ≥ 0.02` | sheared floors/ramps un-skew to square; wedge gaps at walls that grow with slab length |
| `det(M') < 0` ⇒ flip winding or reverse indices | mirrored props render inside-out; back faces visible, lighting from behind |
| Normals via the normal matrix (`n @ inv(M3)`) | non-uniformly-scaled props light as if bent |
| Terrain vertices pre-negated in X | terrain mirrored/offset from the buildings, roads not meeting the ground |
| Terrain height `× 2.0` | every terrain exactly half height; whole map sunk, props floating |
| Terrain holes cut | tunnels/bunkers filled with solid ground, interiors floating over it |
| `_UVScale × _PerTexProps`, never `m_TileSize` | grass tiles every ~137 m instead of ~1.8 m ("massive grass"), blurry ground |
| `_PerTexProps` dtype matches `m_TextureFormat` (20=f32, 17=f16) | garbage tiling values → silent fallback to the 137 m case |
| Uncovered terrain texels filled, not divided | black patches on the ground where no control layer has weight |
| `sum(sub.n) == OBJ face count` | `submesh span overruns OBJ tris`; the last submesh loses geometry |
| Never delete a sub from `subs` - mark it | every later sub reads a face range shifted early; the last sub's faces disappear (a see-through hole with the wrong material on the survivor) |
| OBJ completeness by head+tail bytes, not size | NUL-filled OBJs are reused forever; LOD dedup then deletes the coarse shell that still had geometry → hole in the world |
| `lod.g` offset on every merge/partial run | instances index the wrong LODGroup; shells swap or vanish at distance |
| Whole-LOD-level all-or-nothing dedup | partially-populated distance band: half an object appears at one range |
| `wlayer` (both passes) outranks the root denylist | the lake extracts perfectly and is discarded at assembly (BSG parks it under a `BLOCKERS` root); miss pass 2 and the pond surface never ships at all |
| `activeInHierarchy` walked to the root | ~50 % of hidden geometry misclassified; drawn or dropped wrongly |
| Emissive gated on Unity's `_EMISSION` keyword / `Emissive` shader variant | stale `_EmissionColor` glows: "yellow bonfire stones", `*_OFF` lamps, glowing produce |
| Untextured + default shader = proxy, **except** `role == "water"` | ponds/lakes silently merged out of the pack |
| UTF-8 when writing OBJs | Cyrillic-named meshes become 0-byte stubs; whole walls invisible |

---

## 14. Environment knobs

There is no single default rule - read each knob's truth test before setting it, because the tests differ.

**Extraction:** `EFT_GAME_DATA`, `EFT_ASSETS_ROOT`, `EFT_TARKMAP_ROOT`, `EFT_TEXCACHE`/`EFT_TEXCACHE_DIR` (content-addressed PNG cache keyed on blake2b of the resolved source bytes + `WxH|format|N/A|pil<ver>`), `EFT_TEX_WORKERS`, `EFT_TERRAIN_GPU`/`EFT_ATLAS_EXE`, `EFT_TERRAIN_TILE_JOBS`, `EFT_JOBS`, `EFT_KEEP_STAGING`.

- `EFT_PNG_FAST` - **opt-in, off by default** (`:133-137`, "OPT-IN speed knob, NOT default"). It is read as `bool(os.environ.get("EFT_PNG_FAST"))`, so *any non-empty value enables it*, including the string `0`. Enabled it uses zlib level 1: still lossless, ~15-40 % larger output. To disable, unset the variable.
- `EFT_TERRAIN_HOLES` - set to `0` to stop cutting terrain hole quads (`:738`). This changes geometry (§8.1: tunnels fill with solid ground), it is not a speed knob.

**Assembly:** `EFT_OBJ_FASTPARSE`, `EFT_MESH_BINARY`, `EFT_ASM_VEC` - the three genuine speed knobs; disabling one selects a slower but byte-identical path. `TARKMAP_KEEP_HIDDEN` and `TARKMAP_DROP_INACTIVE` are `== "1"` **opt-ins** that change the kept instance set (`culls.py:61,67`); any other value, `0` included, does nothing.

---

## 15. Old patterns

Each entry is corroborated by a comment at the anchor given; the removed code is gone, so the before/after behaviour is recorded history, not re-checkable.

- **LOD selection by mesh-name substring (`_LOD0`, `model_lod`).** Replaced by `LODGroup.m_LODs[i].renderers` PPtrs. The name heuristic dropped generically-named LOD0 objects and stacked lowercase LOD shells; a LOD0 renderer whose *mesh* is named `..._LOD1` is legal.
- **Texture/material keying on `path_id` alone.** Replaced by `(source file stem, source path_id)`; `file_id == 0` is relative to the referrer, which collapsed distinct textures (~7 % wrong-texture rate - a dumpster texture on a wall).
- **Allowlist as exclusion (`keep only roots starting with SBG`).** Replaced by allowlist-as-protection + one shared structural denylist (`culls.py:133-142`). The old form silently deleted every non-prefixed root and worked on exactly one map.
- **Emitting a synthetic white submesh for material-less renderers.** Replaced by dropping them, plus the two-pass water tagging of §4 (layer-4 MeshFilters, and layer-0 material-less renderers with a layer-4 collider sibling). See the "flat white sheets" class of artifacts.
- **`m_TileSize` for terrain tiling.** Garbage under MicroSplat; use `_UVScale × _PerTexProps`.
- **`np.fromstring` in the OBJ fast parser** (`objio.py:42-50`). Deprecated sep-mode C parser; replaced by `np.array(tokens, dtype)` after a heap-overrun hunt that manifested as an access violation in unrelated code.
- **Keep-min LOD dedup** (`assemble_bevy.py:796-804`). Replaced by the prove-the-premise dedup of §9.4.
- **`EXT_mesh_gpu_instancing` TRS split with a three-way bake gate** (`assemble_bevy.py:8-19`). The glTF-era compromise; a raw affine instance buffer removes the need for it entirely, leaving the rank-deficient bake as the only special case.