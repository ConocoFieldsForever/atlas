## Contents

- [1. Scope and file map](#1-scope-and-file-map)
- [2. Source objects: Unity `Terrain` → `TerrainData`](#2-source-objects-unity-terrain--terraindata)
- [3. The heightmap: storage, the 15-bit convention, the height formula](#3-the-heightmap-storage-the-15-bit-convention-the-height-formula)
- [4. The terrain OBJ: vertex frame, X-negation, UVs, winding, holes](#4-the-terrain-obj-vertex-frame-x-negation-uvs-winding-holes)
- [5. Terrain placement: one frame with every other mesh](#5-terrain-placement-one-frame-with-every-other-mesh)
- [6. MicroSplat tiling: where the real layer scales come from](#6-microsplat-tiling-where-the-real-layer-scales-come-from)
- [7. Splat control maps and the layer→(texture, channel) rule](#7-splat-control-maps-and-the-layertexture-channel-rule)
- [8. `terrain_layers/`: contents, naming, manifest schema](#8-terrain_layers-contents-naming-manifest-schema)
- [9. Blending N layers at runtime](#9-blending-n-layers-at-runtime)
- [10. The baked albedo slice (fallback path)](#10-the-baked-albedo-slice-fallback-path)
- [11. Terrain knobs](#11-terrain-knobs)
- [12. Terrain invariants and failure signatures](#12-terrain-invariants-and-failure-signatures)
- [13. Colour grade LUT: where it comes from](#13-colour-grade-lut-where-it-comes-from)
- [14. The game strip: dimensions, axis attribution, cube extraction](#14-the-game-strip-dimensions-axis-attribution-cube-extraction)
- [15. The shipped LUT file: exact format and packing](#15-the-shipped-lut-file-exact-format-and-packing)
- [16. Applying the LUT: shaper, sampling, slice interpolation](#16-applying-the-lut-shaper-sampling-slice-interpolation)
- [17. Grade invariants and failure signatures](#17-grade-invariants-and-failure-signatures)
- [18. Old patterns](#18-old-patterns)

---

## 1. Scope and file map

Terrain producer: `extraction/unity/eft_extract_v2.py`
- `write_terrain_obj` (:717-775) - heightmap → OBJ
- `export_terrain_splat` (:666-714) - control maps, layer diffuse PNGs, manifest
- `microsplat_uv_scales` (:400-464), `_terrain_season` (:368-397) - tiling recovery
- `_terrain_bake_prepare` / `_terrain_bake_composite` / `bake_terrain_albedo` (:490-604) - flat albedo bake
- `_terrain_bake_gpu` (:616-663) + `viewer/src/terrain_bake.rs` + `viewer/assets/shaders/terrain_bake.wgsl` - GPU port of the same composite
- driver loop (:1699-1782), manifest write (:1795-1799)

Terrain consumers: `eft_pipeline/assemble_bevy.py` (`FLAG_TERRAIN` :126, tagging :970/:1069, sidecar shipping :1286-1307), `viewer/src/render/gpu_driven.rs` (`TerrainSplatGpu` :388-406, table build :2373-2527), `viewer/assets/shaders/gpu_draw.wgsl` (:185-193, :1240-1261).

Grade producer: `extraction/grade/make_grade_lut_game.py` (authentic, from the game texture), `extraction/grade/make_grade_lut.py` (legacy reconstruction). Consumers: `viewer/src/render/grade.rs`, `viewer/assets/shaders/grade.wgsl`.

All Unity asset payloads are **little-endian** (PC build). All intermediate binaries this pipeline writes are little-endian (`'<f4'`, `eft_extract_v2.py:644`).

---

## 2. Source objects: Unity `Terrain` → `TerrainData`

Iterate objects of type `Terrain` in each `level<N>` file; each has `m_TerrainData` (PPtr) and `m_GameObject` (`eft_extract_v2.py:1708-1712`). One `Terrain` object == one **slice** (called a "tile" in the manifest). `TerrainData.m_Name` is the slice name, sanitized (`san()`); it is the ONLY identity - never assume a naming scheme.

Measured on Interchange level 63: 4 slices named `Slice_1_1`, `Slice_1_2`, `Slice_2_1`, `Slice_2_2`. A source comment at `gpu_driven.rs:392` records Lighthouse at 6 slices and larger maps at more; no Lighthouse or Streets terrain sidecar is present in this repo, so that count is unverified here. The 16-slice cap in the GPU table IS confirmed (`gpu_driven.rs:2420-2426`).

Fields consumed:
- `m_Heightmap.m_Heights` - height samples
- `m_Heightmap.m_Resolution` - `res`. Only `res = 1025` is observable here (§3), and only indirectly; the general `2^k + 1` form is a Unity convention that nothing in this repo checks - unverified.
- `m_Heightmap.m_Scale` - `Vector3(x, y, z)`, metres per grid step in X/Z and full height in Y
- `m_Heightmap.m_Holes` - per-cell coverage bytes
- `m_SplatDatabase.m_AlphaTextures` - control (splat) textures
- `m_SplatDatabase.m_TerrainLayers` (older names `m_Splats` / `m_SplatPrototypes`) - layer list, each with `m_DiffuseTexture` (aliases `m_Texture`, `m_Diffuse`) and `m_TileSize`

---

## 3. The heightmap: storage, the 15-bit convention, the height formula

`m_Heights` is read as `np.asarray(g(hm, "m_Heights"), dtype=np.float64).reshape(res, res)` (`eft_extract_v2.py:727`) and indexed `H[row, col]`. The reshape confirms a flat, **row-major** array of exactly `res*res` samples - it would raise otherwise. The extractor never asserts a source dtype, and the sample width/signedness is a UnityPy/Unity typetree property not observable anywhere in this repo: the widely-held "signed 16-bit" reading is consistent with the max-32767 convention below but is **unverified here**. Row index advances along **+Z**, column index along **+X** in Unity's frame.

The height formula (`eft_extract_v2.py:728`):

```
y_metres = (raw / 65535.0) * 2.0 * m_Scale.y
```

The `* 2.0` is Unity's "16-bit field, 15 bits used" convention: the maximum stored sample is **32767**, and 32767 must map to full `m_Scale.y`, not half of it. Equivalently `y = raw / 32767.5 * m_Scale.y`. This is general Unity behaviour, not an EFT quirk.

**Failure signature if you omit the ×2:** every terrain sits at exactly half its real elevation - the ground plunges tens of metres below the buildings, props and colliders float in the air, and the map looks like it was cut along a horizontal plane.

Grid geometry:
- vertex count per axis at native resolution = `res`
- **quad** count per axis = `res - 1`
- world extent `sizeX = (res - 1) * m_Scale.x`, `sizeZ = (res - 1) * m_Scale.z` (`eft_extract_v2.py:503`, `:677`, `:775`)
- native step in metres = `m_Scale.x` (X) and `m_Scale.z` (Z)

Measured (Interchange level 63): `res = 1025`, `sizeX = sizeZ = 700.0 m`, therefore `m_Scale.x = m_Scale.z = 700/1024 = 0.68359375 m` per native step. `res` here is **not** read back from `m_Resolution`; it is forced by the shipped step-2 OBJ, whose numbers were confirmed byte-for-byte: 263169 verts (= 513²), second vertex `v -1.3672 117.0007 0.0000`, x span `[-700, 0]`, z span `[0, 700]` - which give `res - 1 = 1024`.

Decimation: `write_terrain_obj(td, path, step)` takes every `step`-th sample, `Hs = Hw[::step, ::step]` (`:729`). CLI `--terrain-step`, default **2** (`:786-787`): `1` = native (1025² verts/slice, ~0.68 m/quad), `2` = 513² verts (~1.37 m/quad), `4` = coarse. Decimated spacing = `step * m_Scale.x` metres. Measured Interchange at step 2: 263169 verts, spacing 1.3672 m.

---

## 4. The terrain OBJ: vertex frame, X-negation, UVs, winding, holes

For decimated grid `(rr, cc) = Hs.shape`, the writer emits, in this order (`eft_extract_v2.py:754-771`):

```
v  (-c*step*m_Scale.x)   Hs[r,c]   (+r*step*m_Scale.z)      # rr*cc lines, r outer, c inner
vt (c/(cc-1))            (r/(rr-1))                          # rr*cc lines, same order
f  b/b d/d a/a
f  e/e d/d b/b                                                # per quad (r,c), 1-based indices
```
with `a = r*cc + c + 1`, `b = r*cc + (c+1) + 1`, `d = (r+1)*cc + c + 1`, `e = (r+1)*cc + (c+1) + 1`.

**X is negated.** This is deliberate and matches what UnityPy's `Mesh.export()` emits for every other mesh in the dataset (`eft_extract_v2.py:748-750`; regular meshes go through `mesh.export()` at `:1232`). Terrain therefore lands in the *same* vertex frame as everything else, and the assembler's single global handedness conjugation covers both. Z is **not** negated; Y is the raw metric height.

No `vn` lines are written. The assembler derives normals from face winding (`assemble_bevy.py` face-normal accumulation after `_unique_rows`). With the X-negation in place, the emitted winding `(b, d, a)` / `(e, d, b)` yields face normals of `+Y`:
`cross(p_d - p_b, p_a - p_b) = (0, +step²·sx·sz, 0)`, and `(e, d, b)` likewise.

**Failure signature if you keep +X (or re-flip):** the terrain is mirrored about the map's X axis relative to every building - roads run off the edge of their embankments, the slice offsets look "almost right but shifted", and the error grows linearly with distance from x = 0.

**Failure signature if you reverse the winding:** the ground is backface-culled - you see through it into the skybox from above, and shadow-casting inverts.

UVs are the normalized grid position, `u = c/(cc-1)`, `v = r/(rr-1)`, both inclusive of 1.0 at the far edge. The assembler bakes the submesh `_ST` (here `[1,1,0,0]`, i.e. identity) and then applies **one V flip**, `v ← 1 - v` (`assemble_bevy.py:1006`), because Unity's texture origin is bottom-left and PNG rows / wgpu samplers are top-left. `manifest.conventions.uvVFlipBaked` records that the flip is already in the vertex data, so a loader must **not** flip again.

**Failure signature if the V flip is applied twice or zero times:** the splat control maps are mirrored top-to-bottom against the geometry - roads and gravel patches appear on the wrong side of the slice, and the mismatch is a mirror, not an offset, so it is worst at the slice edges and vanishes at the centre line.

**Terrain holes** (`eft_extract_v2.py:730-747, 762-769`). `m_Heightmap.m_Holes` is a flat `uint8` array of `(res-1)²` coverage cells (side length recovered as `hres = round(sqrt(len))`, accepted only when `hres*hres == len` and `hres >= res-1`). A cell value `< 128` means **hole** - the game cuts that quad out (tunnel mouths, bunker entrances, pits). A decimated quad `(r, c)` spans full-res cells `[r*step : (r+1)*step] × [c*step : (c+1)*step]`; if **any** spanned cell is holed, the quad is dropped entirely (conservative: a tunnel is never re-filled; the cut edge is at most `step-1` cells too wide). The `(r,c)` topology is identical to the face loop, so no index remapping is needed. Measured Interchange `Slice_1_1` at step 2: 524268 faces vs `2*(513-1)² = 524288` - 10 quads cut.

**Failure signature if holes are ignored:** tunnels and bunker mouths are paved over with solid ground; every prop, light and loot point placed *inside* those spaces reads as floating above intact terrain.

Writes are atomic - temp file + `os.replace` (`:753`, `:772`) - because the resume path trusts `os.path.exists(fp)` and would otherwise reuse a truncated OBJ forever.

`write_terrain_obj` returns `(sizeX, sizeZ, sizeY)` where `sizeY = Hw.max() - Hw.min()` - the *realized* elevation range, not `m_Scale.y`.

---

## 5. Terrain placement: one frame with every other mesh

The terrain instance record (`eft_extract_v2.py:1739-1743`):

```json
{"mesh": "terrain_<lv>_<tname>.obj",
 "m": [16 floats, ROW-MAJOR 4x4],
 "subs": [{"n": -1, "tex": "terrain_<lv>_<tname>_albedo", "nrm": null, "sh": "terrain", "uv": [1,1,0,0]}],
 "lv": <level>, "kind": "terrain"}
```

`m` is the **raw Unity world matrix** of the terrain's GameObject, obtained by the same father-chain walk used for every mesh: `W(node) = W(father) @ trs(node)`, full 4×4 multiplication, no TRS decomposition anywhere (`eft_extract_v2.py:836-866`; `trs()` in `extraction/unity/eft_scene_extract.py:35-40` builds `M[:3,:3] = quat_to_mat(r) @ diag(s)`, `M[:3,3] = p`). Unreadable terrains fall back to identity (`:1740`).

The assembler applies the global handedness conjugation **once**, `M' = G·M·G⁻¹` with `G3 = diag(-1, 1, 1)` (`assemble_bevy.py:901-905`, `:1061`), and emits the row-major 3×4 of `M'`. Because the OBJ vertices were already emitted as `v' = G3 · v_unity` (§4), the composition is exactly

```
M' · v' = (G·M·G⁻¹)·(G·v_unity) = G·(M · v_unity)
```

- a single mirror of the whole map. That is *why* terrain shares one frame with meshes: the X-negation in the OBJ writer is not a terrain fudge, it is the price of entry into the same conjugation.

**Never** conjugate in the extractor. Extractors emit raw Unity matrices; the assembler owns `G`. Conjugating twice yields `M·(G·v)` where `G·(M·v)` is correct, which for a pure translation `t` displaces the mesh by exactly `2·t_x` in x - twice its distance from the origin.

Measured Interchange (values as stored in `scene.json`, i.e. raw Unity, pre-conjugation): the four slices carry pure-translation matrices at `x ∈ {-647.19995, 52.80005}`, `z ∈ {-810.12134, -110.12134}`, `y = -90.67603` - a 2×2 grid with exactly 700 m spacing on both axes, matching `sizeX = sizeZ = 700`. Each slice's OBJ spans `x ∈ [-700, 0]`, `z ∈ [0, 700]` in local space.

The assembler tags the instance `FLAG_TERRAIN = 1 << 1` (`assemble_bevy.py:126`, `:970`, `:1069`; mirrored as `eftpack::flags::TERRAIN` in `viewer/src/eftpack.rs:42`) when any submesh has `sh == "terrain"` or the instance's `kind == "terrain"`.

---

## 6. MicroSplat tiling: where the real layer scales come from

EFT terrains use MicroSplat, not Unity's stock terrain shader. **`TerrainLayer.m_TileSize` is garbage on these terrains** - measured Interchange values are 129.6 … 333.3 m. A source comment at `eft_extract_v2.py:406` also records a layer with `y = inf`; the manifest persists only `tileX`, so the `y` component cannot be checked from shipped data and that value is unverified. (The same comment writes grass `x = 137.25` while the shipped Interchange manifest records Grass `tileX = 137.3` - consistent after rounding, but the two numbers come from different runs.) Using `m_TileSize` tiles grass every ~137 m.

The real tiling lives in the MicroSplat *material*, resolved by `microsplat_uv_scales(season)` (`eft_extract_v2.py:400-464`):

1. **Season token** from the layers' diffuse texture names (`_terrain_season`, :368-397). Candidate tokens, compound-first so `spring_early` is not double-counted as `spring`: `("spring_early", "autumn_late", "summer", "winter", "spring", "autumn")`. First matching token per layer casts one vote; the plurality wins. A near-tie (`winner - runner_up <= 1`) or a winner covering fewer than half the layers prints an UNCERTAIN warning.
2. Load `sharedassets17.assets` and find the `Material` whose `m_Name` starts with `MicroSplat_` and ends with the season token (`:419-424`). The source describes the full name pattern as `MicroSplat_<Q>_<season>`; the game install was not read here, so the middle token is **unverified**.
3. `_UVScale` - a **Color** property in `m_SavedProperties.m_Colors`; take `.r` (`:430-432`). The EFT value **233.333** is corroborated indirectly by the shipped Interchange manifest (`uvscale: 233.333`, `season: "summer"` on all four slices), not by reading the game asset.
4. `_PerTexProps` - a texture in `m_SavedProperties.m_TexEnvs` whose key contains `PerTex`. Reinterpret `image_data` with the dtype that matches `m_TextureFormat`: `RGBAFloat = 20` → `float32`, 16 bytes/texel; `RGBAHalf = 17` → `float16`, 8 bytes/texel. Reshape `(H, W, 4)`. **Row 0, channel R**, indexed by layer index (== `TerrainLayer` order == texture-array slot), is the per-texture scale (`:440-451`). EFT ships `RGBAFloat` (20).
5. Validate: all finite, all `> 0`, `max < 1e6`. Otherwise reject and fall back to `m_TileSize`.

The MicroSplat UV law:

```
tiledUV_i = terrainUV01 * _UVScale * perTexScale[i]
rep_i     = _UVScale * perTexScale[i]              # repeats across the 0..1 terrain UV
tile_metres_i = sizeX / rep_i
```

`rep` is **isotropic in UV space** - the same on U and V regardless of the slice's metre aspect ratio. Only the `m_TileSize` fallback is per-axis: `repX = sizeX/tile_m`, `repZ = sizeZ/tile_m` (`eft_extract_v2.py:522-528`).

Measured Interchange, from the shipped manifest (`_UVScale = 233.333`, `sizeX = 700`):

| layer | name | `rep` | `perTexScale` | tile size |
|---|---|---|---|---|
| 0 | `Grass_summer_D` | 396.667 | 1.70 | 1.765 m |
| 1 | `Ground_summer_D` | 175.000 | 0.75 | 4.000 m |
| 2 | `Gravel_Road_A_summer_D` | 303.333 | 1.30 | 2.308 m |
| 7 | `Gravel_summer_D` | 420.000 | 1.80 | 1.667 m |
| 8 | `Grassy_Ground_summer_D` | 233.333 | 1.00 | 3.000 m |

**The cache is keyed on the season string.** `microsplat_uv_scales` opens with `if season in _MS_UV_CACHE: return _MS_UV_CACHE[season]` (`:413-414`) and closes with `_MS_UV_CACHE[season] = res` (`:463`). The season early-return short-circuits the entire function *before* any `Material` is scanned, so the second same-season terrain in a run unconditionally reuses the first's `(uvscale, pertex)` regardless of which MicroSplat material it would have resolved to. The secondary `mat_key = ("ms_uv", o.path_id)` entries (`:426-428`, `:459`) live inside the material scan and are therefore unreachable on that second call. The docstring at `:408-409` asserts per-material cache identity; the code at `:413-414` and `:463` contradicts it. Treat per-material identity as *not* implemented.

**Failure signature if you use `m_TileSize`:** "massive grass" - a single grass blade stretched over ~137 m, ground that looks like a low-frequency watercolour, and roads whose gravel is smeared into bands.

---

## 7. Splat control maps and the layer→(texture, channel) rule

`m_SplatDatabase.m_AlphaTextures` is a list of RGBA textures. Each control texture carries **four** layers, one per channel, in `TerrainLayer` order:

```
control_texture_index = layer // 4
channel_index         = layer % 4      # 0=R, 1=G, 2=B, 3=A
```

(`eft_extract_v2.py:518`, `:690`; `gpu_draw.wgsl:1248`.)

Layer count is `len(m_SplatDatabase.m_TerrainLayers)`. `4 * len(m_AlphaTextures)` is only an **upper bound**, enforced by early exit from the layer loops: `if tex_i >= len(ctrl): break` (`:517-519`) and `if ti >= len(alphas): break` (`:689-691`). Layers past that bound are dropped. Interchange is the coincident case where the two numbers agree (12 layers, 3 control textures).

Number of control textures = `ceil(n_layers / 4)`. Measured Interchange: **3** control textures per slice, 12 layers, control texture resolution **1024 × 1024 RGBA8**. Layer diffuse textures are also 1024 × 1024 (RGB after conversion).

Control maps are exported as RGBA PNG per slice, `ctrl_<tname>_<i>.png` for `i` in `0..len(alphas)` (`:685-687`).

Control-map texels are **data, not colour**. They must be uploaded as a **linear/UNORM** format and must **not** be downscaled by texture-quality settings (`gpu_driven.rs:2435-2436`, `:1894-1895`): one control texel drives one patch of ground.

**Failure signature if control maps go through an sRGB decode:** the blend weights are gamma-warped - dominant layers over-dominate, minor layers vanish, and the transitions between ground types get a hard, contrasty edge instead of a soft ramp.

**Failure signature if control maps are downscaled:** roads and paths wander and blur by tens of metres; small gravel patches disappear entirely.

The weights are **not** guaranteed to sum to 1. Both the bake and the runtime normalize by the accumulated weight sum (`eft_extract_v2.py:590`, `gpu_draw.wgsl:1260`).

---

## 8. `terrain_layers/`: contents, naming, manifest schema

One directory per dataset, `<dataset>/terrain_layers/`, accumulated across all slices and all levels (`eft_extract_v2.py:807`, `:1795-1799`):

| file | producer | content |
|---|---|---|
| `manifest.json` | `eft_extract_v2.py:1798` | the schema below |
| `ctrl_<slice>_<i>.png` | `:686-687` | RGBA8 control texture `i` of `<slice>` |
| `layer_<diffuseName>.png` | `:708` | RGB layer diffuse, **deduplicated by name across every slice** |
| `grass_<Tex>.png`, `grass_density_<slice>.bin`, `grass_protos_<slice>.bin`, `grass.json` | `extraction/unity/eft_extract_grass.py:3, 409-411, 495` | grass sidecars - same directory, different subsystem |

Layer diffuse export is gated on coverage: a layer is written when its mean control coverage `>= thresh` (default 0.005) **or** its peak `>= 0.5` (`:701-706`). The peak clause exists because layers that are locally dominant in small patches (Sand/Pebbles on Reserve: ~0.4 % mean, ~100 % inside their patches) would otherwise be referenced by the manifest but absent from disk.

`manifest.json` schema:

```json
{"tiles": {"<slice>": {"ctrl_maps": ["ctrl_<slice>_0.png", ...],
                        "sizeX": <float metres>,
                        "season": "<token>|null",
                        "uvscale": <float|null>,
                        "layers": [{"idx": <int>, "name": "<diffuse m_Name>",
                                    "ctrl": <idx//4>, "chan": <idx%4>,
                                    "cov": <mean coverage 0..1>,
                                    "tileX": <m_TileSize.x, reference only>,
                                    "rep": <repeats across terrain UV01>}]}},
 "layers": ["<sorted list of every diffuse name written>"]}
```

The layer diffuse file for a record is `layer_<name>.png` - the consumer reconstructs the filename from `name` (`gpu_driven.rs:2458`). `tileX` is retained for provenance only; **use `rep`**.

The assembler ships the whole directory into the pack verbatim (`assemble_bevy.py:1290`), then calls `_relativize_tl_manifest` (def at `:654`, call site `:1293`) on the copied manifest. That function's recursive `walk()` (`:663-668`) replaces any string that is both an absolute path and ends in `.png` with its basename (test at `:666-667`) and rewrites the file only if something changed (`:669-671`), so a basename always resolves as `terrain_layers/<name>.png` relative to the manifest. `manifest.sidecars.terrainLayers` points at `terrain_layers/manifest.json` (`:1298`).

---

## 9. Blending N layers at runtime

The reference implementation is `gpu_draw.wgsl:1240-1261`, driven by the table in `gpu_driven.rs:388-406`:

```
TerrainSplatGpu {          // 288 bytes, 16-aligned, matches WGSL `TerrainSplat` byte-for-byte
    layer_albedo: [u32; 12],   // bindless texture index per layer
    layer_rep:    [f32; 12],   // rep_i from the manifest
    ctrl_idx:     [u32; 48],   // slice s, control map k at [s*3 + k]; capacity 16 slices
}
```

Per fragment, with `uv` = the terrain's V-flipped grid UV and `slice` = the material's slice index:

```
base = slice * 3
c0 = sampleGrad(ctrl[base+0], uv, duv_dx, duv_dy)
c1 = sampleGrad(ctrl[base+1], uv, duv_dx, duv_dy)
c2 = sampleGrad(ctrl[base+2], uv, duv_dx, duv_dy)
w  = [c0.r,c0.g,c0.b,c0.a, c1.r,c1.g,c1.b,c1.a, c2.r,c2.g,c2.b,c2.a]

acc = 0; wsum = 0
for i in 0..12:
    if w[i] <= 0.002: continue
    rep = layer_rep[i]
    la  = sampleGrad(layer_albedo[i], uv*rep, duv_dx*rep, duv_dy*rep)
    acc  += w[i] * la.rgb
    wsum += w[i]
albedo = vec4(acc / max(wsum, 0.002), 1.0) * tint
```

Generalizing to N layers: the loop bound is `4 * n_control_textures`, and the weight vector is the concatenation of the control textures' RGBA channels in texture order.

Implementation requirements:
- **Explicit gradients.** `let duv_dx = dpdx(o.uv);` / `let duv_dy = dpdy(o.uv);` are evaluated in *uniform* control flow, before the terrain branch (`gpu_draw.wgsl:1075-1076`; the rationale comment is `:1072-1074`), then scaled by `rep` per layer. Implicit derivatives inside a non-uniform branch are undefined; hardware-computed derivatives on `uv*rep` would also be wrong at slice-edge quads.
- **The layer sampler wraps.** `uv*rep` runs to hundreds of repeats.
- Terrain materials must be excluded from every other albedo path - the splat branch owns albedo and normal. The loader clears `MAT_FLAG_DETAIL`, `MAT_FLAG_RFA` and any emissive index on terrain materials, and forces `roughness = 0.95` (`gpu_driven.rs:2514-2518`).
- The 12 layers are shared across every slice of a map (same MicroSplat material) - capture them once from the first slice (`gpu_driven.rs:2440-2441`). Only `ctrl_idx` varies per slice.
- Slice names come from the sidecar and are **sorted** for a stable slice→index mapping (`:2418-2419`). Matching a mesh name to a slice must be a **whole-token** match - plain substring matching assigns `Slice_1_1`'s control maps to `Slice_1_11` on maps with more than 9 slices (`:2489-2496`).

**Failure signature if a slice→control mapping is wrong:** one slice wears another slice's roads - a perfectly plausible-looking ground that does not line up with the geometry, worst at map scale, invisible in a close-up screenshot.

**Failure signature if a layer PNG is missing:** the bindless load-failure placeholder (1×1 magenta) binds and the ground gets magenta blotches exactly where that layer dominates. The loader substitutes the first present layer and warns instead (`:2443-2478`).

---

## 10. The baked albedo slice (fallback path)

Independently of the splat path, the extractor bakes one flat albedo PNG per slice: `<dataset>/tex/terrain_<lv>_<slice>_albedo.png`, referenced by the instance's submesh `tex` field. It is the fallback for consumers with no splat support, and the *only* ground colour on packs whose `terrainLayers` sidecar is absent (`eft_extract_v2.py:1720-1734`).

Bake parameters: `res_out = 4096`, supersampling `ss = 2` (`:490`, `:597`). Measured Interchange output: **4096 × 4096 RGB PNG per slice**, 38.4-39.3 MB each - one map at one bake setting, i.e. sample data, not a specification.

The composite (`_terrain_bake_composite`, `:554-594`; GPU port `terrain_bake.wgsl`):

```
for each output texel (i=row/v, j=col/u) of an R×R image:
    for each layer L (control tex t, channel ch, diffuse D, repX, repZ):
        w = bilinear(ctrl[t][..,ch], (i+0.5)/R, (j+0.5)/R, clamp)     # sampled ONCE at pixel centre
        for a in 0..ss, b in 0..ss:
            v = (i + (b+0.5)/ss) / R ; u = (j + (a+0.5)/ss) / R       # jittered
            s = bilinear(D, frac(v*repZ), frac(u*repX), wrap)          # or (0.4,0.4,0.4) if D is absent
            acc += w*s ; wsum += w
covered = wsum > 1e-3
out[covered]  = acc[covered] / wsum[covered]
out[!covered] = mean colour of the covered area (fallback 0.4 grey)
```

Details that matter for bit-parity:
- `_bilinear` (`:467-487`) is **align-corners**: `py = fy*(h-1)`, `px = fx*(w-1)`. `wrap=True` uses modulo (so the `frac()` seam interpolates); `wrap=False` clamps.
- Control weights are sampled once at the texel centre, not once per sub-sample - the control maps are low-frequency; the jitter exists to anti-alias the fine-tiled *diffuse* gather.
- Layers whose global control max is `<= 0.001` are pruned before any pass (`:542-548`).
- Uncovered texels (all weights ~0) are **filled with the covered-area mean**, never divided ~0/~0. Skipping this paints black ground wherever the control maps have no coverage.

GPU path: `_terrain_bake_gpu` (`:616-663`) writes a temp dir with `pixels.bin` - every control and diffuse texture concatenated as **RGBA float32 little-endian**, `w*h` texels each, `off` recorded in texel units - plus `m.json` (`{R, ss, out, pixels, texs:[[off,w,h]...], layers:[{ctrl,ch,diffuse,repX,repZ}]}`, mirrored in `viewer/src/terrain_bake.rs:14-33`). Control textures occupy indices `0..len(ctrl)` so `layer.ctrl == tex_i` directly. `atlas bake-terrain <m.json>` runs `terrain_bake.wgsl` (one thread per output texel, `@workgroup_size(8,8)`), writes `(albedo_sum.rgb, weight_sum)` per texel, and the Rust side normalizes and PNGs. Non-zero exit → numpy fallback.

**Sharpness budget.** A slice's baked texel covers `sizeX / res_out` metres. Measured Interchange: `700 / 4096 = 0.1709 m/texel`, while the real grass tile is `1.765 m` - about **10 texels per tile**. Halving to 2048 gives ~5 texels per grass tile, at which point the tiled detail becomes indistinguishable noise. Treat `res_out >= sizeX / 0.171`, rounded up to the next power of two, as the floor: 4096 for a 700 m slice (`700/0.171 = 4093.6`), 8192 for a 1400 m one (`1400/0.171 = 8187.1`). The splat path (§9) has no such cap and is the reason it exists.

Robustness: the albedo is regenerated whenever `_png_complete` fails, not on a size test - NTFS preallocation makes a killed run leave a full-size, NUL-filled PNG that a size check happily reuses forever (`:1720-1729`).

---

## 11. Terrain knobs

| variable / flag | default | effect |
|---|---|---|
| `--terrain-step` | `2` | heightmap decimation (1 = native) |
| `--terrain-only` | off | re-bake/re-export terrain into an existing dataset; **merges** `scene.json` instead of overwriting (`:1806-1821`) |
| `EFT_TERRAIN_HOLES` | `1` | `0` disables hole cutting |
| `EFT_TERRAIN_GPU` | `1` | `0` forces the numpy composite |
| `EFT_ATLAS_EXE` | unset | path to the `atlas` binary providing `bake-terrain` |
| `EFT_TERRAIN_TILE_JOBS` | `4` | CPU composite thread count (`1` = sequential) |
| `EFT_BAKE_CPU` | `0` | `1` makes `atlas bake-terrain` decline (exit 3) so the extractor uses numpy |

`--terrain-only` without the merge guard replaces the whole scene graph with the run's ~4 terrain records - a 3-second touch-up destroying a 17-minute extraction.

---

## 12. Terrain invariants and failure signatures

The failure descriptions in §4, §5 and this table are **analytic** - derived from each invariant's math, not observed as shipped regressions - except the counts and dimensions explicitly marked measured. The winding and double-conjugation cases were confirmed by independent derivation (§4, §5).

| invariant | violated → you see |
|---|---|
| `y = raw/65535 * 2 * m_Scale.y` | terrain at half elevation; everything else floats |
| OBJ X negated, Z and Y not | terrain mirrored against buildings, error grows with \|x\| |
| winding `(b,d,a)`,`(e,d,b)` → +Y normals | ground backface-culled; sky visible through the floor |
| exactly one V flip on UVs | control maps mirrored top-to-bottom against geometry |
| conjugation `G·M·G⁻¹` applied once, in the assembler | terrain offset from meshes by twice its distance from the origin |
| no TRS decomposition anywhere | sheared/non-uniformly-scaled parents drift; terrain usually survives (near-identity) while props do not |
| `layer//4`, `layer%4` channel packing | wrong ground type everywhere; roads made of grass |
| control maps uploaded linear, not downscaled | washed-out or hard-edged blends; roads that wander |
| `rep = _UVScale * perTexScale[i]`, never `m_TileSize` | ~137 m grass tiles - smeared watercolour ground |
| holed quads dropped | tunnels paved over; interior props floating |
| slice matched as a whole token | one slice wearing another's roads on >9-slice maps |
| uncovered bake texels neutral-filled | black patches in the baked albedo |

---

## 13. Colour grade LUT: where it comes from

The game's grading LUT is a `Texture2D` in `<EFT_GAME_DATA>/resources.assets` named `LUT-amidgenofbluegreen2lighterblack`. `make_grade_lut_game.py:43-75` locates it **by name substring** (`EFT_GRADE_LUT_NAME`, default `amidgen`), never by path id, so an update that renumbers assets still resolves. A source docstring (`make_grade_lut_game.py:4`) records path id 524 in one observed build; `resources.assets` was not read here, so that id is unverified. `EFT_GAME_DATA` overrides the install path. No game texture ships with the tool: absent a local strip, the baker extracts one. (The script's own docstring at `:23` claims the source strip ships next to it; it does not - `extraction/grade/` holds only `eft_grade_fit.json`, `eft_grade_lut.bin` and the two scripts.)

This is a **PostProcessing v2 LDR LUT strip**. It operates in **display-referred sRGB space**: the game clamps linear scene colour to 0..1, sRGB-encodes it (EFT hard-clips highlights - that hard clip is authentic), and uses the encoded triple as the LUT input (`make_grade_lut_game.py:11-13`).

`tools/build_map.py:905-919` runs the extraction as build stage 2 into `<tarkmap>/out/eft_grade_lut.bin`; if the game is absent it falls back to `make_grade_lut.py` + `eft_grade_fit.json` (the legacy reconstruction). `assemble_bevy.py:1381-1390` promotes that file into `packs/shared/grade_lut.bin`.

---

## 14. The game strip: dimensions, axis attribution, cube extraction

The script requires the strip to be **32 × 1024 RGB** - `assert strip.shape == (32, 1024, 3)` at `:82` - i.e. 32 tiles of 32 × 32 laid out horizontally. No strip ships in `extraction/grade/`, so that assert is unexercised in this repo and the dimension is a code-stated expectation, not a measurement made here.

Axis attribution was established by per-axis output-gradient measurement rather than by inspecting greys (greys cannot disambiguate a symmetric packing). The recorded gradients - `dR/dx = 0.024`, `dB/dtile = 0.017`, `dG/drow = 0.022` (`make_grade_lut_game.py:6-9`) - come from the script's header and are **unreproducible in this repo** without the strip:

```
strip[row, tile*32 + x]      x = R input, tile = B input, row = G input
```

in UnityPy's export orientation, **no vertical flip**. Extraction to a `[r][g][b]` cube (`:81-84`):

```python
strip = asarray(Image.open(SRC).convert('RGB'), float32) / 255.0   # (32, 1024, 3)
game  = strip.reshape(32, 32, 32, 3).transpose(2, 0, 1, 3)         # [g,b,x=r] -> [r,g,b]
```

Sampling `game` is straightforward trilinear on `f = clamp(x,0,1) * 31`, `i0 = floor(f)`, `i1 = min(i0+1, 31)`, `t = f - i0`, lerping R then G then B (`:91-107`).

**Failure signature if the axis attribution is permuted:** the image gains a consistent, plausible-looking but wrong colour cast (typically a red/blue swap producing a uniform cool cast), and greys stay grey - which is exactly why greys cannot be used to check it.

---

## 15. The shipped LUT file: exact format and packing

The baker resamples the 32³ game LUT onto a 64³ grid in the *viewer's shaper space* and writes `eft_grade_lut.bin`:

- **512 × 512 RGBA8**, raw bytes, **no header, no PNG container**. Exact size **1048576 bytes**; the loader rejects any other length (`grade.rs:76-79`).
- Row-major, row 0 = data row 0 (raw `DataTexture` byte order - no flipY, no colour-space tag).
- Byte offset of texel `(row, col)` = `(row*512 + col) * 4`; channel order **R, G, B, A**; alpha is a constant 255.
- **8 × 8 flipbook of 64 × 64 tiles.** Blue slice `b` → tile `(tx, ty) = (b % 8, b // 8)`. Inside a tile, **x = R index, y = G index**.
- Full address of LUT entry `(ri, gi, bi)`, all in `0..63`:

```
row = (bi // 8) * 64 + gi
col = (bi %  8) * 64 + ri
offset = (row * 512 + col) * 4
```

(`make_grade_lut_game.py:116-121`; the identical packing in `make_grade_lut.py:65-70`; the inverse read in `grade.rs:99-111`.)

- The stored triples are **display-encoded** (sRGB), matching what the game's LUT emits.
- **Input shaper**: axis index `i` in `0..63` corresponds to linear channel value

```
u     = i / 63
c_lin = 4 * u²                     # so the LUT domain covers linear 0 .. 4.0
```

The baker evaluates `disp = srgb_encode(clamp(c_lin, 0, 1))` - sRGB encode with the standard piecewise form, `c <= 0.0031308 ? 12.92c : 1.055·c^(1/2.4) - 0.055` (`:87-89`) - then samples the game cube at `disp` (`:109-114`). Note the clamp: linear inputs above 1.0 all map to the same LUT output, which is the hard highlight clip EFT actually performs.

---

## 16. Applying the LUT: shaper, sampling, slice interpolation

### Load-time repack (recommended)

`grade.rs:87-111` repacks the atlas into a real **64×64×64 `Rgba16Float` 3D texture**, upload order x=R fastest → y=G → z=B, with **linear filtering and ClampToEdge on all three axes** (`:189-217`). Two things happen at load:

1. The **display encode is inverted per texel** (sRGB EOTF → linear), so the sampled LUT emits **linear**. The pass then outputs linear into an `Rgba16Float` target and the swapchain applies the sRGB encode exactly once.
2. Storage is f16, not 8-bit - 8-bit linear bands visibly in the toe.

**Failure signature if you skip the inversion but still present through an sRGB swapchain:** the image is encoded twice - milky, washed-out, lifted blacks. Skip the *encode* instead and the image is crushed and oversaturated.

### Sampling

```wgsl
p   = sqrt(clamp(c_linear / 4.0, 0, 1))          // inverse of the 4u² shaper
uvw = p * (63.0/64.0) + (0.5/64.0)               // index p*(N-1) -> texel CENTRE
out = textureSampleLevel(lut3d, lut_samp, uvw, 0).rgb
```

(`grade.wgsl:72-76`, the `lut_sample` function; its rationale comment is `:67-71`.) The shaper is `sqrt(lin/4)`, **not** sRGB and **not** log. The `(N-1)/N` + `0.5/N` remap is mandatory: the LUT stores the value *for* `p` at integer index `p*(N-1)`, while a normalized 3D-texture coordinate addresses `coord*N - 0.5`. Feeding raw `p` skews the entire transfer curve by up to half a texel.

**Failure signature if the shaper is wrong:** the LUT reads the wrong blue slice - a strong, uniform colour cast that changes with exposure.
**Failure signature if the half-texel remap is missing:** shadows read too dark and highlights clamp roughly one texel early; the error is subtle and uniform, easy to mistake for "the grade is just contrasty".

### Sampling the 2D atlas directly (no 3D texture)

Manual blue-slice trilinear, preserved as a comment block at `grade.wgsl:78-83`:

```
u  = sqrt(clamp(c/4, 0, 1)) * 63.0            // vec3, in 0..63
b0 = floor(u.b); b1 = min(b0 + 1, 63); f = u.b - b0
xy = vec2(u.r, u.g) + 0.5                      // half-texel centre within the tile
uv0 = (vec2(b0 % 8, floor(b0/8)) * 64 + xy) / 512
uv1 = (vec2(b1 % 8, floor(b1/8)) * 64 + xy) / 512
rgb = mix(sample(uv0), sample(uv1), f)         // hardware bilinear covers R and G
```

The two taps **must** come from different tiles when `b0 != b1`, and hardware filtering must never be allowed to interpolate *across* a tile boundary in x or y - clamp `xy` to `[0.5, 63.5]` inside the tile.

**Failure signature if tiles bleed:** thin, hard-edged colour seams at regular intervals along one channel's ramp - most visible on smooth gradients like sky and fog.

### Pass order

`grade.wgsl:152-189`, running between Bloom and Tonemapping with the camera's own tonemapper set to None (this pass *is* the tonemap):

1. optional FXAA on the linear HDR scene, **before** sharpen
2. optional 4-tap unsharp mask on the pre-LUT linear scene, `scene + (scene - n*0.25)*sharpen`, clamped at 0
3. `lin = scene * exposure` - adapted exposure when auto-exposure has produced one (`ae.exposure > 0`), else the authored constant. Native default `DEFAULT_GRADE_EXPOSURE = 1.35` (`viewer/src/render/mod.rs:34`), overridable via `EFT_GRADE_EXPOSURE`
4. `g = lut_sample(lin)`
5. PRISM vignette: `e = (uv - 0.5) * 2 / (1.15, 0.95)`, `vig = 1 - smoothstep(0.55, 1.25, length(e)) * 0.488`. The reference multiplied the vignette onto **display-encoded** pixels; here `g` is linear, so the shader outputs `g * pow(vig, 2.4)` - this makes the post-encode result equal `encode(g) * vig`.

**Failure signature if the vignette is applied linearly:** corners darken roughly half as much as the game's.

`EFT_GRADE=0` disables the whole pass (camera falls back to TonyMcMapface); `EFT_VIGNETTE=0` zeroes the strength. LUT resolution order: `EFT_GRADE_LUT` → `<pack>/grade_lut.bin` → `<pack>/../shared/grade_lut.bin` → the shared dir default (`grade.rs:65-74`). The grade pass and the fallback tonemapper are mutually exclusive - running both double-tonemaps.

---

## 17. Grade invariants and failure signatures

| invariant | violated → you see |
|---|---|
| file is exactly 1048576 bytes, RGBA8, 512×512, no header | loader rejects, or garbage colours from a misparsed PNG |
| slice `b` → tile `(b%8, b//8)`, in-tile x=R y=G | LUT reads the wrong slice; strong uniform cast |
| shaper `p = sqrt(lin/4)` | as above, cast varies with exposure |
| `uvw = p*(63/64) + 0.5/64` | shadows too dark, highlights clip early |
| display encode inverted exactly once across load + swapchain | washed-out (twice encoded) or crushed (never encoded) |
| strip axes x=R, tile=B, row=G | plausible but wrong cast; greys unchanged |
| vignette raised to the 2.4 power in linear | corners half as dark as the game's |
| grade pass XOR camera tonemapper | double tonemapping - blown, low-contrast highlights |

---

## 18. Old patterns

- `TerrainLayer.m_TileSize` was once the tiling source. It produced ~137 m grass tiles. It survives in the manifest as `tileX` for provenance and as the fallback when no MicroSplat material resolves; new consumers must read `rep`.
- The `_PerTexProps` format constant was once mislabelled (`17` named "RGBAFloat"), which rejected the real `RGBAFloat = 20` data and silently fell back to `m_TileSize`. `RGBAHalf = 17`, `RGBAFloat = 20`.
- The control map was once sampled nearest-neighbour in the bake, giving blocky layer transitions; it is bilinear now (`eft_extract_v2.py:546`).
- Layer-diffuse export was once gated on **mean** coverage alone, which dropped locally dominant layers that every manifest still referenced - hence the magenta-blotch fallback path in the loader. The gate is now mean OR peak.
- Slice names were once a hardcoded list, which silently disabled MicroSplat on every map but Interchange. They come from the sidecar and are sorted.
- The MicroSplat UV cache was intended to key on the resolved material's `path_id`; the season-string early-return added in front of it makes that keying dead (§6). The docstring at `eft_extract_v2.py:408-409` still describes the intent, not the behaviour.
- `make_grade_lut.py` bakes the **legacy reconstructed** look (Hejl-Dawson with EFT-fitted constants 6.2/0.05/0.8, 0.004/0.06 → per-channel PCHIP film curves → a fitted 3×3 "Fahrenheit" matrix plus 16-point curves, mixed at `fit.mix = 0.498`). Its output format is byte-compatible with the game LUT so consumers need no branch, but it is a different look. The shipped `eft_grade_lut.bin` is the authentic one from `make_grade_lut_game.py`.
- The 2D-tiled in-shader trilinear (§16) was the original runtime path; the 3D-texture repack replaced it so hardware filtering handles the blue-slice lerp with no tile-seam math.