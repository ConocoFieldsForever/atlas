# EFT sky, particles and water4 extraction

## Contents

- [1. Scope and file map](#1-scope-and-file-map)
- [2. Sky](#2-sky)
  - [2.1 Source bundle and asset selection](#21-source-bundle-and-asset-selection)
  - [2.2 Cubemap byte layout: DXT1, face-major mip chains](#22-cubemap-byte-layout-dxt1-face-major-mip-chains)
  - [2.3 Face order, orientation, and the texel to direction map](#23-face-order-orientation-and-the-texel-to-direction-map)
  - [2.4 Structural sky classifier](#24-structural-sky-classifier)
  - [2.5 Derived colours](#25-derived-colours)
  - [2.6 sky.json schema](#26-skyjson-schema)
  - [2.7 What the viewer actually renders](#27-what-the-viewer-actually-renders)
- [3. Particles](#3-particles)
  - [3.1 Invocation, inputs and outputs](#31-invocation-inputs-and-outputs)
  - [3.2 The exact keep/drop predicate](#32-the-exact-keepdrop-predicate)
  - [3.3 World position and uniform scale](#33-world-position-and-uniform-scale)
  - [3.4 MinMaxCurve / MinMaxGradient reduction](#34-minmaxcurve--minmaxgradient-reduction)
  - [3.5 particles.json schema](#35-particlesjson-schema)
  - [3.6 tex_fx/*.png](#36-tex_fxpng)
  - [3.7 Flipbook atlas and the UVModule grid](#37-flipbook-atlas-and-the-uvmodule-grid)
  - [3.8 Emission rate, cluster sizing, billboard convention](#38-emission-rate-cluster-sizing-billboard-convention)
  - [3.9 Measured figures](#39-measured-figures)
- [4. Water](#4-water)
  - [4.1 water4.json schema](#41-water4json-schema)
  - [4.2 The authored Water4 parameter set](#42-the-authored-water4-parameter-set)
  - [4.3 The Gerstner math those parameters drive](#43-the-gerstner-math-those-parameters-drive)
  - [4.4 What the renderer actually does with water](#44-what-the-renderer-actually-does-with-water)
  - [4.5 Sea-level derivation](#45-sea-level-derivation)
  - [4.6 The synthetic sea quad](#46-the-synthetic-sea-quad)
- [5. Invariants and failure signatures](#5-invariants-and-failure-signatures)
- [6. Old patterns](#6-old-patterns)

---

## 1. Scope and file map

Three independent sidecar producers. None of them participates in pack reassembly; each writes JSON (plus PNGs) that a built pack or a shared directory absorbs in place.

| Subsystem | Extractor | Output | Consumer |
|---|---|---|---|
| Sky | `extraction/unity/eft_extract_sky.py` | `packs/shared/sky/*.png` + `packs/shared/sky/sky.json` | none in this repo (see §2.7) |
| Particles | `extraction/unity/eft_extract_particles.py` | `<pack>/particles.json` + `<pack>/tex_fx/*.png` | `viewer/src/fx.rs` |
| Water params | `extraction/unity/eft_extract_water4.py` | `packs/shared/water4.json` | none in this repo; one value hand-copied into `viewer/assets/shaders/gpu_draw.wgsl:1765` |
| Sea level | `tools/build_map.py:228` `derive_sea_level` | `manifest.seaLevel` (float, metres) | `viewer/src/render/gpu_driven.rs:3351` |

All three read the game install through `EFT_GAME_DATA` (`eft_extract_sky.py:27`, `eft_extract_particles.py:24`, `eft_extract_water4.py:29`). Sky and water honour `EFT_INTEL_OUT_DIR` and otherwise write under `packs/shared` (`eft_extract_sky.py:25`, `eft_extract_water4.py:27`).

Units are metres and seconds throughout. Colour components are unbounded non-negative floats unless stated.

The failure-signature notes throughout are derived from the code paths cited beside them; none of them is reproduced from an artifact present in this repo.

---

## 2. Sky

### 2.1 Source bundle and asset selection

`eft_extract_sky.py:29` builds the bundle path constant:

```
BUNDLE = os.path.join(GAME, "StreamingAssets", "Windows", "cubemaps")
```

The load itself is `env = UnityPy.load(BUNDLE)` at `:35`. It iterates `env.objects` and keeps only `o.type.name == "Cubemap"` (`:40-41`). Pixel data is obtained through `o.read()` then `tex.get_image_data()` (`:49-50`). The comment at `:46-47` states the load-bearing fact: the typetree path returns an **empty inline blob** because the pixels live in the bundle's `.resS`; only `get_image_data()` resolves `m_StreamData`. That comment's specific claim - that reading via `read_typetree()["image data"]` yields zero bytes - is unverified here, since no cubemaps bundle is present in this repo to reproduce it.

`UnityPy`'s `.image` accessor returns **one** face for a Cubemap (`:43`), which is why the decode is done by hand.

### 2.2 Cubemap byte layout: DXT1, face-major mip chains

Only `m_TextureFormat == 10` (DXT1 / BC1) is handled; anything else is skipped with a log line (`:55-58`).

Faces are square: `face = m_Width` (`:59`), mip count `mips = m_MipCount` (`:60`).

Block-compressed size, from `:64-65`:

```
mip_bytes(w, h) = max(1, w // 4) * max(1, h // 4) * 8      # 8 bytes per 4x4 BC1 block
```

One face's full mip chain (`:66`):

```
chain = sum( mip_bytes(max(1, face >> m), max(1, face >> m)) for m in 0..mips-1 )
```

The buffer is **face-major**: face 0's entire mip chain, then face 1's, and so on. Face `i`'s mip 0 therefore starts at byte offset

```
offset_i = i * chain          # stride between faces = chain
length   = mip_bytes(face, face)
```

(`:73`). A sanity gate rejects buffers shorter than `6 * chain` (`:67-69`).

Decode is `texture2ddecoder.decode_bc1(raw, face, face)` returning a **BGRA8** byte buffer, wrapped as `Image.frombytes("RGBA", (face, face), bgra, "raw", "BGRA")` (`:74-75`). The channel-swizzle string is mandatory: without it red and blue swap and every sky reads brown.

Each face becomes a float array `numpy.float32` of shape `(face, face, 3)` scaled to `[0,1]` by `/255.0` (`:76`). Those are still **sRGB-encoded** values.

Failure signature for a wrong `chain`: face 0 is correct and faces 1..5 decode as garbage diagonal streaks, because every subsequent face starts mid-mip.

### 2.3 Face order, orientation, and the texel to direction map

Faces are emitted in index order 0..5 straight from the buffer (`:72-76`, `:89-93`), which is Unity's storage order and is identical to the wgpu/Vulkan/D3D cubemap array-layer order:

```
0 = +X   1 = -X   2 = +Y   3 = -Y   4 = +Z   5 = -Z
```

`:78` asserts the consequence used downstream: **index 2 is up**.

**No per-face rotation and no flip of any kind is applied.** The numpy array produced by the BC1 decode is saved verbatim: `Image.fromarray((f * 255).astype("uint8"))` (`:92`). Row 0 of the PNG is row 0 of the decoded surface.

A consumer that uploads these six PNGs as cube array layers 0..5 must use the standard face parameterisation. The repo's own reference implementation of that mapping, for `u, v ∈ [-1, 1]` with `u = 2(x+0.5)/N - 1`, `v = 2(y+0.5)/N - 1` and `(x, y)` the texel column/row, is `viewer/src/main.rs:1656-1667`:

```
face 0 (+X): dir = ( 1, -v, -u)
face 1 (-X): dir = (-1, -v,  u)
face 2 (+Y): dir = ( u,  1,  v)
face 3 (-Y): dir = ( u, -1, -v)
face 4 (+Z): dir = ( u, -v,  1)
face 5 (-Z): dir = (-u, -v, -1)
```

then normalised. Note `-v` on the four side faces and on ±Z: increasing row index moves **downward** in world terms. This is the invariant the horizon strip in §2.5 relies on.

Failure signature for an unwanted V flip: the ground appears at the top of the four side faces, and the derived `horizon` colour samples sky instead of ground - on `rain_1k_sharp_DXT1` that turns a `[0.022, 0.019, 0.015]` horizon into something near the `[0.434, 0.433, 0.434]` zenith.

Failure signature for a face-order permutation (a common Unity-to-wgpu mistake is swapping ±Y): the sky renders sideways and `is_sky` inverts, because the classifier reads index 2 and 3.

### 2.4 Structural sky classifier

The bundle carries three kinds of Cubemap: map-scale skies, interior reflection probes, and material captures (`patron_*` bullet-brass swatches). The classifier is deliberately **name-free** (`:10-15`).

Per-face mean luma, Rec.709 weights, computed on the **sRGB-encoded** values (`:77`):

```
lum[i] = mean over all texels of ( 0.2126*R + 0.7152*G + 0.0722*B )
```

Decision (`:79`):

```
is_sky = lum[2] > 1.3 * max(lum[3], 1e-4)
```

i.e. the +Y face must be more than 1.3x brighter than the -Y face. There is no horizon-continuity test despite the docstring at `:11-12` mentioning one - the shipped rule is the top/bottom luma ratio alone (`:13-14` says so explicitly).

The classifier over-accepts. Measured on the shipped bundle, **18 of 24** cubemaps flag `is_sky=true`, including every `patron_cubemap_*` material capture (brass swatch: `face_luma = [0.203, 0.260, 0.232, 0.057, 0.170, 0.115]`, ratio 4.07). Exactly six flag false: `factory_dush_DXT1`, `factory_ceh_DXT1`, `factory_ceh_sharp_DXT1`, `morning clouds_DXT1`, `site_cubemapa_DXT1`, `moscow_2_DXT1`. One of those six is a genuine sky that is dark on top: `morning clouds_DXT1` has `face_luma[2]=0.585` vs `face_luma[3]=0.516`, ratio 1.13. The extractor's stated contract is that consumers **pick a cubemap by name** and treat `is_sky` as a hint (`:14-15`).

### 2.5 Derived colours

All three derived colours are computed on an approximate linearisation, `lin[i] = faces[i] ** 2.2` (`:82`), applied per channel. This is a pure gamma decode with no sRGB linear-segment near black; `:80-81` states the consumer may re-derive exactly.

**zenith** (`:83`) - mean of the +Y face over all texels:

```
zenith = mean_{x,y} lin[2][y][x]        # per channel, shape (3,)
```

**horizon** (`:84-86`) - the bottom `max(4, face // 8)` rows of the four side faces, concatenated along the row axis and averaged:

```
rows   = max(4, face // 8)
strip  = concat( lin[0][-rows:], lin[1][-rows:], lin[4][-rows:], lin[5][-rows:] )
horizon = mean over axes (0,1) of strip
```

Side faces are indices `(0, 1, 4, 5)` = `+X, -X, +Z, -Z`. For a 1024 px face this is 128 rows per side; for 128 px it is 16 rows (the `max(4, …)` floor never binds at shipped sizes).

**mean** (`:87`) - flat mean over all six faces:

```
mean = mean over all 6*face*face texels of lin
```

This is **texel-uniform, not solid-angle weighted**. A true irradiance mean requires the cube-face Jacobian `dΩ = du dv / (1 + u² + v²)^{3/2}`, which down-weights corners by up to `3^{-3/2} ≈ 0.192`. A reimplementer wanting a physically correct average must add that weight; the shipped value biases toward face corners. The size of that bias on real probes is unverified - no per-probe solid-angle-weighted comparison exists here.

All three are rounded to 5 decimals in the record (`:98-100`).

### 2.6 sky.json schema

Written to `<out>/sky.json` with `indent=1` (`:109-112`):

```json
{
  "schema": 1,
  "source": "StreamingAssets/Windows/cubemaps (Cubemap assets, verbatim)",
  "built": 1700000000,
  "cubemaps": {
    "<m_Name>": {
      "faces":     ["<safe>_face0.png", ... 6 entries, index == face index],
      "size":      1024,
      "is_sky":    true,
      "zenith":    [r, g, b],
      "horizon":   [r, g, b],
      "mean":      [r, g, b],
      "face_luma": [l0, l1, l2, l3, l4, l5]
    }
  }
}
```

`built` is a Unix epoch second count (`:110`). The record key is `m_Name` verbatim; the PNG basename uses `safe`, the name with every character failing the predicate `c.isalnum() or c in "-_"` replaced by `_` (`:70`). Python's `str.isalnum()` is Unicode-aware, so non-ASCII letters and digits survive unchanged - this is **not** an ASCII `[A-Za-z0-9_-]` filter. Two cubemaps with names differing only in a rejected character collide on disk - the second overwrites the first, and `faces` in both records then points at the same six PNGs.

`zenith`/`horizon`/`mean` are in the 2.2-decoded space; `face_luma` is in sRGB-encoded space. They are not comparable to each other.

Measured on the shipped bundle (24 cubemaps): `rain_1k_sharp_DXT1`, 1024 px, `is_sky=true`, `zenith=[0.43372, 0.43318, 0.43372]`, `horizon=[0.02218, 0.01880, 0.01549]`, `mean=[0.21019, 0.20864, 0.20448]`. The near-perfect grey of the zenith and the ~20x zenith/horizon ratio are what an overcast raid sky looks like in this data.

### 2.7 What the viewer actually renders

**Nothing in this repo reads `sky.json` or the exported face PNGs.** `viewer/src/main.rs:1638-1648` `build_sky_cubemap` calls `build_procedural_sky` unconditionally and nothing else. The comment at `:1639-1644` records why: the exported assets are environment **captures** - photo-spheres with treelines baked into the horizon - so using them as a sky dome puts photographic trees behind real map geometry.

The dome that ships is 6 × 128 × 128 `Rgba16Float` (`main.rs:1651`, `:1696`), with `TextureViewDimension::Cube` (`:1700`), built from:

```
up     = clamp(dir.y * 0.5 + 0.5, 0, 1)
sky    = lerp( (0.66,0.72,0.82), (0.92,0.98,1.10), up*up )          # main.rs:1670-1672
if dir.y < 0:  sky *= 1 - 0.55 * min(-dir.y * 3, 1)                  # main.rs:1676
s      = max(dot(dir, sun), 0)
sky   += (1.05,1.00,0.90) * ( s^350 * 3.0 + s^8 * 0.3 )              # main.rs:1680-1681
```

The same horizon/zenith pair is duplicated in `viewer/assets/shaders/gpu_draw.wgsl:620-621` for glossy reflections, scaled by local SH luma and `SKY_REFL_GAIN = 1.45` (`gpu_draw.wgsl:142`, `:618-623`). Fog uses a third, independent constant `FOG_COLOR = (0.44, 0.49, 0.58)` with `FOG_DENSITY = 0.00075` per metre, exp-squared (`gpu_draw.wgsl:150`, `:153`, `:636`). The SH bake uses a fourth, neutral grey gradient `g = (0.35 + 0.75 * max(dir.y, 0)) * scale` (`viewer/src/sh_bake.rs:116-121`), which ramps horizon 0.35 → zenith 1.10 (`sh_bake.rs:118`).

The Python bake's sky gradient is **inverted relative to that one**: `extraction/bake/bake_volume2.py:358-359` sets `SKY_ZENITH = 0.436 * SKY_SCALE` and `SKY_HORIZON = 0.743 * SKY_SCALE`, and `sky_radiance` lerps `v = hor*(1-grad) + zen*grad`, so its zenith is **darker** than its horizon. The only property the two share is being neutral grey with no tint.

That is four separately-authored sky descriptions in the viewer plus a fifth in the Python bake that disagrees with the Rust one on gradient direction. The `zenith`/`horizon`/`mean` triples are the intended single feed for the reflection and fog consumers; the wiring does not exist.

---

## 3. Particles

### 3.1 Invocation, inputs and outputs

```
python extraction/unity/eft_extract_particles.py --pack <pack.eftpack> --levels 466,467,...
```

(`eft_extract_particles.py:15`, `:121-124`). Level files are `<EFT_GAME_DATA>/level<N>`, silently skipped when absent (`:133-136`). Outputs are `<pack>/particles.json` and `<pack>/tex_fx/` (`:126-127`, `:339-341`). No pack rebuild is needed; the sidecar heals an already-built pack (`:8-9`).

Per level, three lookup tables are built by a full `env.objects` sweep (`:141-154`):

- `tf[transform_pathID] = (father_pathID, local_TRS_dict, gameObject_pathID)`
- `go_act[gameObject_pathID] = m_IsActive`
- `go2tf[gameObject_pathID] = transform_pathID`

and a fourth over `ParticleSystemRenderer` (`:200-209`): `rend[gameObject_pathID] = (object, m_Materials[0], m_RenderMode)`.

### 3.2 The exact keep/drop predicate

A `ParticleSystem` is emitted **iff all five** hold:

1. `o.read_typetree()` succeeds (`:216-218`).
2. `bool(d.get("looping"))` is true (test at `:223`, `continue` at `:224`).
3. `active_chain(gameObject)` is true (`:225`, implementation `:188-197`, 64-hop cap). The walk starts at the emitter's **own** transform (`t = go2tf.get(go_pid, 0)`, `:189`) and tests that transform's GameObject before climbing (`:192-194`), so the emitter's own `m_IsActive` is part of the predicate: this is `activeInHierarchy`, not an ancestors-only test.
4. A `ParticleSystemRenderer` exists on the same GameObject and its `m_Materials` list is **non-empty** (`:227-229`). The table stores `mats[0] if mats else None` (`:208`) - the raw PPtr dict, so a null reference `{m_FileID: 0, m_PathID: 0}` is not `None` and passes this gate; such an emitter is only dropped by gate 5.
5. That material resolves a texture in `_MainTex`, `_BaseMap` or `_Tex` (first hit wins) and the texture saves as a PNG; otherwise `continue` (`:244-261`, `:269-270`).

The predicate is **looping AND active-in-hierarchy**, plus two data-availability gates: a renderer with a material slot, and an atlas texture that saves.

`playOnAwake` is **not** tested. The module docstring at `:11` claims it is; the code at `:221-223` explicitly overrules it - "playOnAwake is NOT required, fire prefabs trigger some of their looping children (sparks, embers) from scripts".

`n_seen` counts all ParticleSystems (`:219`), `n_loop` counts those passing gates 2–4 (`:230`); the difference against the emitted count is the texture gate (`:342-343` prints both).

Failure signature for adding `playOnAwake`: burning buildings lose their sparks and ember columns while the base smoke survives.

### 3.3 World position and uniform scale

`world_pos_scale` (`:156-186`) walks the father chain, at most 64 hops, accumulating a full 4×4:

```
R = quaternion(m_LocalRotation) -> 3x3          # :172-176, column-vector form
S = diag(m_LocalScale.xyz)                      # :177
L[:3,:3] = R @ S ;  L[:3,3] = m_LocalPosition   # :178-180
M = L @ M                                       # :181, child accumulates on the right
```

Then:

```
pos   = M[:3, 3]
scale = cbrt( abs( max(det(M[:3,:3]), 1e-9) ) )      # :184-185
```

`scale` is a volume-preserving uniform magnitude; non-uniform and sheared parents collapse to one number. Rotation is discarded on purpose (`:157-158`) because the consumer re-billboards every quad.

The X-negation handedness conjugation is applied **here**, at write time, not by the assembler (`:316-317`):

```
"pos": [ -pos.x, pos.y, pos.z ]
```

This is the pack's `G3 = diag(-1, 1, 1)`. Because the record carries no rotation and no direction vector, negating the point alone is complete - there is nothing else to conjugate. Applying the conjugation a second time downstream mirrors every effect to the wrong side of the map.

Note this is a chained TRS product, not the raw-3×3 path used for mesh instances, because the ParticleSystem transform data is only available as local TRS in the typetree.

### 3.4 MinMaxCurve / MinMaxGradient reduction

`curve_scalar` (`:32-42`): take `mm["scalar"]` (Unity mode 0, constant); else `mm["maxScalar"]` (mode 3, random between two constants - the **max** branch is taken); else the caller's default.

`gradient_rgba` (`:45-52`): `mg["maxColor"]` else `mg["minColor"]`, read as `r,g,b,a` and rounded to 4 decimals.

`gradient_keys` (`:55-103`) reduces `maxGradient` (else `minGradient`) to at most 6 samples. Unity serialises colour keys and alpha keys **separately** but shares the `key{i}` RGBA slots:

- colour key `i`: RGB from `grad["key{i}"]`, time from `grad["ctime{i}"]`, count `m_NumColorKeys`
- alpha key `i`: A from `grad["key{i}"]`, time from `grad["atime{i}"]`, count `m_NumAlphaKeys`
- both loops are capped at 8 (`:68-77`)

**Times are uint16 ticks; the conversion is `t = ticks / 65535.0` into `[0,1]`** (`:72`, `:77`). Getting this divisor wrong (65536, or treating the field as seconds) puts every key at t≈0 and the gradient collapses to its first colour.

The output times are `sorted({0.0, 1.0} ∪ colour_times ∪ alpha_times)[:6]` (`:101`). The slice takes the **first six sorted** values, so a gradient with six or more early keys silently loses its tail including `t = 1.0`. The consumer then holds the last emitted key for all larger `t` (`viewer/src/fx.rs:231`) - a fire that should fade out stays lit.

Each emitted key is `[t, r, g, b, a]`, with rgb from `sample_c(t)` and a from `sample_a(t)`, both piecewise linear with constant extrapolation outside the key range (`:81-103`).

`curve_keys` (`:106-117`): the first 4 entries of `maxCurve.m_Curve`, each `[time, value]`; returns `None` under 2 keys. Tangents are discarded - the consumer resamples linearly.

### 3.5 particles.json schema

```json
{
  "emitters": [ { ... } ],
  "note": "looping ParticleSystems (game data; viewer flipbook billboards)"
}
```

Per-emitter fields, with source line and extractor default. Optional fields are **absent**, never null.

| Field | Type | Source | Meaning / units |
|---|---|---|---|
| `pos` | `[f32;3]` | `:317` | world metres, X already negated |
| `lv` | int | `:318` | source level index |
| `tex` | string | `:319` | pack-relative, always `tex_fx/…png` |
| `shader` | string | `:320` | `m_ParsedForm.m_Name` else `m_Name`, else `""` |
| `tint` | `[f32;4]` | `:321` | `_TintColor` else `_Color` from `m_Colors`, default `[1,1,1,1]` |
| `renderMode` | int | `:322` | `ParticleSystemRenderer.m_RenderMode` (0 = Billboard, 1 = Stretch on the shipped data) |
| `lifetime` | f32 | `:323` | `InitialModule.startLifetime`, seconds, default 3.0 |
| `speed` | f32 | `:324` | `startSpeed`, m/s, default 0.5 |
| `size` | f32 | `:325` | `startSize × world_scale`, metres, default 1.0 before scaling |
| `color` | `[f32;4]` | `:326` | `startColor` maxColor |
| `gravity` | f32 | `:327` | `gravityModifier`, dimensionless multiplier of g, default 0.0 |
| `rate` | f32 | `:328` | `EmissionModule.rateOverTime`, particles/s, default 4.0 |
| `maxParticles` | int | `:329` | `InitialModule.maxNumParticles`, default 64 |
| `tiles` | `[u32;2]` | `:330-331` | `UVModule.tilesX/tilesY` (fallback keys `xTile`/`yTile`), default `[1,1]` |
| `uvEnabled` | bool | `:332` | `UVModule.enabled` |
| `uvFps` | f32 | `:333` | `UVModule.fps`, default 30.0 |
| `uvCycles` | f32 | `:334` | `UVModule.cycles`, default 1.0 |
| `shapeRadius` | f32 | `:335-336` | `ShapeModule.radius`, metres, default 0.3 |
| `colorOverLife` | `[[t,r,g,b,a]]` opt | `:282-285` | present only when `ColorModule.enabled` and keys exist |
| `sizeOverLife` | `[[t,v]]` opt | `:286-289` | present only when `SizeModule.enabled` and ≥2 curve keys |
| `light` | object opt | `:290-311` | `{ratio, intensity, range, color?}` |
| `uvFpsMode` | int opt | `:312-313` | present and `1` only when `UVModule.timeMode == 1` |

Two extractor behaviours alter the data:

- `shapeRadius` ends in `... or 0.3` (`:336`), so a genuine radius of exactly **0.0 becomes 0.3**. A point emitter therefore scatters over a 0.3 m disc.
- The `light` record multiplies `LightsModule.intensityCurve` by the referenced `Light.m_Intensity` and takes `max(rangeCurve, Light.m_Range)` (`:304-307`). The `Light` lookup is a linear scan of `env.objects` per emitter.

### 3.6 tex_fx/*.png

Filename (`:252-255`):

```
fx_<sanitised m_Name>_<path_id & 0xFFFFFFFF>.png
```

with the same `c.isalnum() or c in "-_"` predicate as §2.6 (`:253`), Unicode-aware and therefore not an ASCII-only filter. The low 32 bits of the path_id disambiguate same-named atlases; on a collision in those 32 bits the second texture is written to the same path and **overwrites** the first (`:255-257`). `tex_cache` is keyed on the full path_id, so both emitters keep their own record and both point at the last-written image. Saved through `timg.image.save(...)` (`:256`), i.e. UnityPy's decoded PIL image: an RGBA PNG in sRGB (RGBA confirmed on the shipped `tex_fx` PNGs; the row origin depends on UnityPy's `.image` flip and is unverified here). `tex_cache` maps path_id to the relative path or `None` on failure so a broken texture is attempted once (`:130`, `:249-259`).

The consumer loads them with `is_srgb = true` (`viewer/src/fx.rs:186-190`).

### 3.7 Flipbook atlas and the UVModule grid

Grid is `tiles = [tilesX, tilesY]`, frame count `frames = tilesX * tilesY` (`fx.rs:261`).

Frame rate (`fx.rs:266-275`):

```
if !uvEnabled || frames <= 1:      fps = 0            # static, frame 0 forever
else if uvFpsMode == 1:            fps = max(uvFps, 0.1)
else:                              fps = frames * max(uvCycles, 0.1) / max(lifetime, 0.05)
```

The `uvFpsMode == 1` branch is plain fps playback, as `fx.rs:82` documents ("UVModule timeMode 1 = plain fps playback"). The third (`else`) branch is Unity's "Lifetime" time mode: the whole sheet plays `uvCycles` times over one particle lifetime (`fx.rs:271`).

Frame selection and UV window (`fx.rs:363-371`):

```
f  = floor(now * fps) mod frames
fx = f mod tilesX
fy = f div tilesX                          # row-major
uv_scale       = (1/tilesX, 1/tilesY)      # fx.rs:249-253
uv_translation = (fx/tilesX, fy/tilesY)
```

so sampled UV is `uv_local * uv_scale + uv_translation` with `uv_local ∈ [0,1]²` from the unit quad. `fy` therefore indexes atlas rows **downward from row 0 of the PNG on disk**. Which end of the sheet a given game atlas puts frame 0 at is unverified - nothing in this repo establishes it.

Failure signature for using `tiles` as `[rows, cols]` instead of `[x, y]`: a non-square grid such as the shipped `8×6` sheets samples half-frames diagonally - visible as a torn seam scrolling through the sprite.

### 3.8 Emission rate, cluster sizing, billboard convention

`rate` does **not** drive spawning. The consumer converts it into a fixed cluster size once (`fx.rs:209-210`):

```
n    = clamp( ceil(rate * lifetime), 1, 10 )     # quads per emitter
norm = min( 2.0 / n, 1.0 )                       # alpha normalisation
```

`norm` exists because n overlapping additive quads accumulate into the grade LUT's clip plateau; it holds total cluster energy at roughly 2× one quad regardless of n.

Per-quad deterministic scatter, golden-ratio, no RNG (`fx.rs:283-285`):

```
h      = (i * 0.618034) mod 1
ang    = h * 2π
jitter = (cos ang, 0, sin ang) * shapeRadius * sqrt(h)
phase  = i / n
roll   = h * 2π
```

The `sqrt(h)` gives area-uniform coverage of the disc.

Motion, per frame (`fx.rs:323-326`) - a loop, not an integrator:

```
t = ((now / lifetime + phase) mod 1) * lifetime          # seconds into this quad's life
y = speed * t - 0.5 * gravity * 9.81 * t²                # metres, +Y up
position = base + (0, y, 0)
```

`9.81` m/s² is hard-coded, `gravity` being Unity's dimensionless `gravityModifier`.

Size, with `lf = t / lifetime` (`fx.rs:327-343`): piecewise-linear sample of `sizeOverLife` when it has ≥2 keys, else the fallback ramp `0.75 + 0.45 * lf`; final scale is `max(size * s, 1e-3)`.

Colour is not per-frame: the `colorOverLife` gradient is discretised into `BINS = 5` materials sampled at `t = 0.1, 0.3, 0.5, 0.7, 0.9` (`fx.rs:233-237`), and an ageing quad swaps its material handle at `bin = min(floor(lf * 5), 4)` (`fx.rs:344-348`). Base colour per bin (`fx.rs:240-245`):

```
base_color = linear_rgba( color.rgb * tint.rgb * grad.rgb ,
                          clamp( clamp(color.a * tint.a, 0, 1) * grad.a * norm, 0, 1 ) )
```

The alpha is clamped **twice**: `fx.rs:203` clamps `color.a * tint.a`, and `fx.rs:244` clamps the whole product. Dropping the outer clamp changes output whenever `grad.a > 1`, or when `norm = 1` with an unclamped gradient alpha.

Blend family is decided from the game's own shader **name**, lowercased (`fx.rs:197-198`):

```
additive = name.contains("additive") || name.contains("hdrfire")
```

additive → `AlphaMode::Add`, otherwise `AlphaMode::Blend` (`fx.rs:246`). Materials are `unlit: true` (`fx.rs:247`), `cull_mode: None` (`fx.rs:248`).

**Billboard convention** (`fx.rs:377-387`): the quad is a unit `Rectangle` in the XY plane (`fx.rs:176`); each frame its rotation is set to

```
quad_rotation = camera_world_rotation * Quat::from_rotation_z(roll)
```

taken from the cull camera's `GlobalTransform`. This is a full view-aligned billboard (the quad plane is parallel to the near plane), **not** a cylindrical/axis-locked one, and the per-quad `roll` is a constant spin about the view axis so cluster members do not read as clones. `renderMode` is recorded in the sidecar but ignored: stretched-billboard emitters are drawn as plain billboards too (`fx.rs:377-379`).

`EFT_FX=0` disables the whole overlay (`fx.rs:159-161`); ESP mode spawns nothing but still runs the teardown (`fx.rs:153-158`).

### 3.9 Measured figures

Measured on the built Interchange pack (`packs/interchange.eftpack/particles.json`, 53 emitters: 52 from level 64 and 1 from level 520).

- shaders: `Particles/SuperAdditive` ×15, `Particles/Smoke Distorted` ×13, `CustomFX/HDRFireParticle` ×12, `Legacy Shaders/Particles/Additive` ×12, `Legacy Shaders/Particles/Alpha Blended` ×1 - so 39 of 53 take the additive branch.
- `renderMode`: 0 ×41, 1 ×12.
- `tiles`: `[8,6]` ×24, `[1,1]` ×15, `[2,2]` ×13, `[8,8]` ×1.
- `colorOverLife` present on all 53; `sizeOverLife` on 38; `uvFpsMode` and `light` on none.
- The single level-520 record is also the single `Legacy Shaders/Particles/Alpha Blended` and `tiles [8,8]` emitter: `pos [302.318, 21.692, -368.031]`, `tex tex_fx/fx_Cycle_Smoke32_gray_1599.png`, `lifetime 1.0 s`, `rate 15.0/s`, `maxParticles 100`.
- Largest emitter: `pos [144.11, 28.68, -9.87]`, `tex tex_fx/fx_smokeTiles2_tga_50.png`, `lifetime 50.0 s`, `speed 2.0 m/s`, `size 30.0 m`, `rate 10.0/s`, `maxParticles 120`, `shapeRadius 0.79 m`, `tint [0.6029, 0.5565, 0.5143, 0.341]`. With `n = clamp(ceil(10*50), 1, 10) = 10`, that emitter draws 10 quads at `norm = 0.2`.

---

## 4. Water

### 4.1 water4.json schema

`eft_extract_water4.py:51-52` scans `resources.assets` plus every `sharedassets*.assets` in `EFT_GAME_DATA`, reading `Material` typetrees (`:64-70`).

Shader membership is decided by **property signature**, not by resolving the shader PPtr (`:71-75`):

```
keep iff  "WaveSpeed" in m_Colors  or  "_GAmplitude" in m_Colors
```

`:72` states the justification: those two names exist only on the Water4 family. This is why the scan needs no shader resolution and survives stripped PPtrs.

`m_SavedProperties` sub-tables are lists of `{first, second}` pairs (or 2-tuples), flattened by `kv` (`:33-40`). **`m_Colors` holds every `float4` property, not just colours** - Unity serialises `Vector` shader properties there. `_GAmplitude`, `_BumpTiling`, `_Extinction` and the `_ST` entries all live under the JSON key `"colors"` and are `[x, y, z, w]`, not RGBA.

```json
{
  "schema": 1,
  "source": "EscapeFromTarkov_Data resources.assets + sharedassets* (Material typetrees)",
  "note": "Authored Water4 parameters, verbatim. Gerstner: _GAmplitude/_GSteepness/_GSpeed/_GDirectionAB/_GDirectionCD. Absent keys were absent in the game.",
  "built": 1700000000,
  "materials": {
    "<m_Name>": {
      "asset":   "sharedassets489.assets",
      "colors":  { "<prop>": [x, y, z, w] },
      "floats":  { "<prop>": f },
      "textures":{ "<slot>": { "fileID": i, "pathID": i, "scale": [...], "offset": [...] } }
    }
  }
}
```

Absent properties are **omitted**, never defaulted (`:14`). Duplicate material names across asset files are resolved by keeping the record with the larger `len(colors) + len(floats)` (`:92-96`), so the surviving `asset` field names the richest copy, not necessarily the one a given map loads.

**Defect to work around**: `color4` (`:43-46`) reads keys `r,g,b,a`, but `m_Scale` and `m_Offset` are `Vector2f` with keys `x,y`. Every `textures[*].scale` and `textures[*].offset` in the file is therefore `[0,0,0,0]`, confirmed in the shipped data. Failure signature: a reimplementer honouring those values tiles the bump map at scale zero and the whole surface samples one texel - a flat, single-coloured sheet. Use the `_<Slot>_ST` entry inside `colors` instead (for example `Sandbox_Water4Advanced.colors._BumpMap_ST = [1, 1, 0, 0]`).

### 4.2 The authored Water4 parameter set

Measured, from `packs/shared/water4.json` (16 materials). Two distinct authored families exist.

**Family A - `FX/Water4Advanced` style.** Members `Sandbox_Water4Advanced`, `Water4Advanced`, `Water4Advanced_Cardinal`, `Water4Advanced_Indoor`, `Wastewater`, `Laboratory_Water_FX`, `Laboratory_FX_OpacityZero`. These carry `WaveSpeed` and `_Extinction` and share identical values for both.

**Family B - Gerstner-authored map water.** Members `Lighthouse_Water`, `Lighthouse_Water_Facility_01`, `City_Dirt_Water`, `City_Water_2`, `Reserve_Water_bunker`, `Reserve_Water_bunker 2`. These carry no `WaveSpeed` and no `_Extinction` but do carry non-zero `_G*`.

| Property | Type | Family A (`Sandbox_Water4Advanced`) | Family B (`Lighthouse_Water`) | Units / range |
|---|---|---|---|---|
| `_GAmplitude` | float4 | `[0.05, 0.20, 0.0, 0.0]` | `[0.2, 0.1, 0.25, 0.25]` | wave amplitude A,B,C,D - metres of vertical displacement |
| `_GFrequency` | float4 | `[15.10, 13.33, 0.0, 0.0]` | `[0.5, 0.25, 0.6, 0.245]` | angular spatial frequency, rad/m; wavelength = 2π/f |
| `_GSteepness` | float4 | `[-2.06, 2.0, 6.0, 2.0]` | `[3.0301, 1.0, 1.0, 1.0]` | Gerstner Q, dimensionless; >1/(f·A) self-intersects |
| `_GSpeed` | float4 | `[0.31, 0.09, -0.09, 3.0]` | `[2.0, 2.0, 1.0, 1.0]` | temporal angular speed, rad/s; sign = travel direction |
| `_GDirectionAB` | float4 | `[0.4691, 0.3541, -0.2, 0.1]` | `[0.85, 0.3, 0.25, 0.25]` | `(dirA.xz, dirB.xz)`, **not normalised** - magnitude scales the effective frequency |
| `_GDirectionCD` | float4 | `[0.7034, -0.6800, 0.7176, -0.2]` | `[-0.3, -0.9, 0.5, 0.5]` | `(dirC.xz, dirD.xz)`, same |
| `_GerstnerIntensity` | float | `-2.5` | `1.0` | global displacement multiplier; the negative value inverts the crest |
| `WaveSpeed` | float4 | `[1.0862, 4.4406, 12.1051, -3.0207]` | absent | bump-layer scroll speeds, UV units/s, `(layer1.xy, layer2.xy)` |
| `_BumpTiling` | float4 | `[0.1, 0.1, 0.02, 0.02]` | `[0.8, 0.8, 0.8, 0.8]` | bump UV frequency per layer, cycles/world-unit |
| `_BumpDirection` | float4 | `[0.2, 0.0, 3.0, 1.0]` | `[1.0, 1.5, 1.5, -0.5]` | per-layer scroll direction multipliers |
| `_BaseColor` | float4 | `[0.1307, 0.1547, 0.1633, 0.653]` | `[0.0, 0.0, 0.0, 0.5529]` | shallow/base tint RGBA, 0..1 |
| `_DepthColor` | float4 | `[0.02751, 0.14179, 0.13227, 0.4627]` | `[0.2981, 0.3661, 0.3955, 0.3451]` | deep-water extinction colour RGBA, 0..1 |
| `_Extinction` | float4 | `[4.5, 75.0, 300.0, 1.0]` | absent | per-channel extinction distance in metres (R fastest, B slowest) |
| `_HorizonColor` | float4 | `[0.0529, 0.2449, 0.2836, 1.0]` | absent | grazing-angle tint |
| `_ReflectionColor` | float4 | `[0.0598, 0.0709, 0.0970, 0.394]` | `[0.0094, 0.0090, 0.0086, 0.9529]` | reflection tint |
| `_SpecularColor` | float4 | `[0.70754719, 0.68102771, 0.63745999, 1.0]` | `[1.0, 1.0, 1.0, 0.6353]` | specular tint RGBA; carried alongside `_Shininess`, not instead of it |
| `_Foam` | float4 | `[0.144, 0, 0, 0]` | `[0.3276, 0.7471, 0, 0]` | `(intensity, cutoff, …)` |
| `_DistortParams` | float4 | `[0.49, 0.055, 3.17, -0.6]` | `[0.055, 0.1, 15.0, 0.12]` | refraction/reflection distortion |
| `_InvFadeParemeter` | float4 | `[1.63, 0.25, 0.0431, 1.63]` | `[3.0, 0.0856, 0.0941, 0.4803]` | soft-edge fade (Unity's spelling, preserved verbatim) |
| `_FresnelScale` | float | `0.81` | `4.0` | |
| `_HeightDisplacement` | float | `3.2723` | `2.337` | metres |
| `_NormalsDisplacement` | float | `0.0` | `72.7273` | |
| `_WaterDepth` | float | `40.9808` | absent | metres |
| `_Shininess` | float | `191.0` | `409.0` | Blinn exponent |

Texture slots present: `_BumpMap`, `_MainTex`, `_ReflectionTex` (family B and most of A). `_Extinction` `[4.5, 75, 300]` reads as metres-to-1/e: red is gone at 4.5 m, blue survives 300 m - the standard reason deep water goes blue-green.

Two materials qualify on the signature test but are not water surfaces at all: `Old_Bunker_Light_Glass_Materials` and `shopping_mall_med_camp_set_tarpaulin_transparent` both carry `_GAmplitude [0.3, 0.35, 0.25, 0.25]` on a glass/tarpaulin shader. The signature filter cannot exclude them; consumers must select by name.

`City_Water` is an all-zero Gerstner record (`_GAmplitude`, `_GFrequency`, `_GSteepness`, `_GSpeed`, `_GDirectionAB`, `_GDirectionCD` all `[0,0,0,0]`, `_GerstnerIntensity 0.0`) - flat water by authoring, not by extraction failure.

### 4.3 The Gerstner math those parameters drive

**This repo contains no Gerstner implementation.** `docs/GRAPHICS_PLAN.md:32` records the state: the properties were confirmed unextracted at plan time, the synthetic sea is four vertices and six indices, and vertex displacement would need a projected grid or clipmap plus identical displacement in the main pass, the depth prepass and any shadow caster. `viewer/src/render/mod.rs:167-168` declares a `water_disp` flag gated on `EFT_WATER_DISP=1` (`mod.rs:269`); nothing reads it.

The property names are Unity's Water4 contract. The formulation below is the reading those names imply, with `p = (x, z)` the horizontal world position and `t` seconds; **the operand layout and the sign conventions are unverified** - no implementation exists here to check them against.

```
AB   = _GSteepness.xxyy * _GAmplitude.xxyy * _GDirectionAB.xyzw
CD   = _GSteepness.zzww * _GAmplitude.zzww * _GDirectionCD.xyzw
dot4 = _GFrequency * vec4( dot(_GDirectionAB.xy, p), dot(_GDirectionAB.zw, p),
                           dot(_GDirectionCD.xy, p), dot(_GDirectionCD.zw, p) )
phase = dot4 + t * _GSpeed
C = cos(phase) ; S = sin(phase)

offset.x = dot( C, vec4(AB.x, AB.z, CD.x, CD.z) )
offset.z = dot( C, vec4(AB.y, AB.w, CD.y, CD.w) )
offset.y = dot( S, _GAmplitude )

displaced = vertex + offset * _GerstnerIntensity
```

Four independent waves A..D. The vertical term is a plain sum of sines weighted by amplitude; the horizontal terms are the Gerstner crest-sharpening, cosine-phased and scaled by steepness×amplitude×direction. Because `_GDirectionAB/CD` are not unit vectors in the shipped data, the effective spatial frequency of a wave is `_GFrequency.c * |dir_c|` and its effective wavelength is `2π / (f·|dir|)` - for `Lighthouse_Water` wave A, `2π / (0.5 × 0.901) ≈ 13.9 m`.

Failure signature for normalising the direction vectors: every wave's frequency changes by `1/|dir|`. Shipped `Lighthouse_Water` magnitudes are `|dirA| = |(0.85, 0.30)| = 0.9014` (11% shift), `|dirB| = |(0.25, 0.25)| = 0.3536` (2.83×, i.e. +183%), `|dirC| = |(-0.3, -0.9)| = 0.9487` (5%), `|dirD| = |(0.5, 0.5)| = 0.7071` (41%). Wave B's wavelength shortens by a factor of 2.83, and the four-wave interference pattern that makes the surface look non-repeating turns into visible beating.

Failure signature for treating `_GSpeed` as m/s rather than rad/s: `Sandbox_Water4Advanced` wave D at `_GSpeed.w = 3.0` with `_GAmplitude.w = 0.0` is invisible either way, but `Lighthouse_Water` at `[2, 2, 1, 1]` runs 2π× too fast and strobes.

Normals follow from the analytic derivative of the same expression (or a finite difference at ±ε in world XZ), then `n @ inv(M3)` under the row-vector convention if the water plane is not axis-aligned.

### 4.4 What the renderer actually does with water

Material classification is by role, in `viewer/src/render/gpu_driven.rs:2172-2192`:

- `role == "water"` sets `MAT_FLAG_WATER` (`gpu_driven.rs:317`, bit 3).
- Textured water (`albedo_index != NO_ALBEDO`) is a thin puddle film: also `MAT_FLAG_BLEND`, drawn in the blend pass. A constant-alpha atlas additionally gets `MAT_FLAG_PUDDLE_LUMA` (`gpu_driven.rs:344`, `1 << 8`) so the shape mask comes from luma.
- Untextured water is **deep** water and stays in the **opaque** pass so depth-write sorts it under glass.
- Stretched floor water-decals get `MAT_FLAG_WATER_MATTE` (`gpu_driven.rs:349`, bit 9), classified geometrically by metres-per-texture-repeat with threshold `WATER_MATTE_MPR = 40.0` (`gpu_driven.rs:1910`). The threshold's authoring comment records puddles measuring ≲22 and floor decals ≳60 on the lighthouse dataset; no lighthouse pack is present in this repo to re-measure that.

The deep-water shading branch is `viewer/assets/shaders/gpu_draw.wgsl:1671-1790`:

```
ripple_amp = (0.06 / (1 + d*0.004)) * (1 - smoothstep(500, 1400, d))     # :1680
wp  = world_pos.xz * 0.35                                                # :1681
cyc = 0.35 * max(footprint.x, footprint.y) / 2π                          # :1688
w1  = 1 - smoothstep(0.10, 0.22, cyc)          # base octave, ~18 m       # :1692
w2  = 1 - smoothstep(0.10, 0.22, cyc * 3.4)    # detail octave, ~5 m      # :1693
drift1 = t * (0.0200, 0.0126) ; drift2 = t * (-0.0160, 0.0230)           # :1704-1705
```

`w1`/`w2` are a per-octave Nyquist band-limit driven by the screen-space derivative footprint - a shader-side mip for procedural sines. `rsin(phase_cycles) = sin(fract(phase_cycles) * 2π)` (`gpu_draw.wgsl:614-616`) exists because raw `sin()` of a world-scaled argument reaches ±1200 rad at map edges where GPU fast-sin collapses into structured precision noise.

When the material carries a normal map (the game's `WaterBasicNormals`), it **replaces** the procedural chop (`gpu_draw.wgsl:1721-1733`): two layers over world XZ at `wfreq = 0.15` cycles/m and `0.15 × 1.73`, scrolling at `(0.050, 0.0315)` and `(-0.040, 0.0575)` UV/s, mixed 50/50 at 0.85 amplitude, sampled with `textureSampleGrad` because the branch is non-uniform control flow.

The surface normal is built from **world up**, never the mesh normal (`gpu_draw.wgsl:1740-1741`), because the sea mesh's per-vertex normals crosshatch at the quad-grid period. Fresnel is Schlick with `F0 = 0.02` (`gpu_draw.wgsl:1744`). SH is sampled one probe layer **above** the surface (`gpu_draw.wgsl:1753-1755`) because sea-level probes straddle the water plane and their trilinear alternation checkerboards the whole sea.

The deep body colour is a hand-copied constant:

```
let deep = vec3<f32>(0.0275, 0.1418, 0.1323);      // gpu_draw.wgsl:1765
```

That is `Sandbox_Water4Advanced._DepthColor.rgb` rounded to 4 decimals (measured `[0.02751, 0.14179, 0.13227]`). It is the one authored Water4 value in the renderer, and it is hard-coded rather than read from `water4.json`. Final radiance is capped hue-preservingly at `mix(0.18, 0.28, fresnel)` (`gpu_draw.wgsl:1785-1786`) so the pack's LDR grade LUT cannot clip the water into its highlight plateau.

### 4.5 Sea-level derivation

`tools/build_map.py:228-358` `derive_sea_level(dataset)` reads the dataset `scene.json` and returns a world-Y float or `None`.

Candidate selection, per instance (`:277-291`): keep the instance if **any** submesh satisfies `is_water_sub`:

```
sub.role == "water"  AND  ( sub.sh is None  OR  ("water" in sh.lower() AND "decal" not in sh.lower()) )
```

`sh is None` admits the untextured shoreline sea tiles; the `decal` exclusion drops puddles (they ride `Decal/Water …` shaders); the `"water" in sh` test drops role-water collision proxies tagged with a `Standard` shader, such as the map-wide `TEMP_GROUND_COLIDER` cube on streets.

For each candidate the local OBJ AABB is read from `v ` lines only (`:250-271`) and its 8 corners are transformed by the instance's raw row-major 3×4:

```
x = m[0]*cx + m[1]*cy + m[2]*cz + m[3]
y = m[4]*cx + m[5]*cy + m[6]*cz + m[7]
z = m[8]*cx + m[9]*cy + m[10]*cz + m[11]
```

(`:302-304`) - the raw matrix applied directly, no TRS decomposition.

Rejection: `wy_max - wy_min > 2.0` m drops sloped/cascade water (`:308`).

Binning: key `round(y_surf * 10)`, i.e. **0.1 m bins**, where `y_surf = (wy_min + wy_max) / 2` (`:310-311`). Each bin accumulates a union XZ footprint and the maximum `y_surf` (`:312-317`).

Qualification is **containment**, not size (`:318-353`). Scene extents come from instance translations `m[3]` and `m[11]` (`:273-275`):

```
EDGE_FRAC     = 0.02      # "touching" = within 2% of the scene span on that side
MIN_AREA_FRAC = 0.10      # footprint must be >= 10% of the scene translation AABB area

qualifies iff  area >= 0.10 * scene_area
          AND  ( b.xmin <= sx_min + 0.02*span_x  or  b.xmax >= sx_max - 0.02*span_x
              or b.zmin <= sz_min + 0.02*span_z  or  b.zmax >= sz_max - 0.02*span_z )
```

The rationale at `:322-333`: an ocean is unbounded and reaches the scene boundary; a lake is enclosed by terrain and stops short on every side. Size alone cannot separate them - the woods lake clears any sane area bar and once produced a spurious 7.454 m "sea level" that flooded the map.

Return value (`:354-358`): among qualifying bins, the largest by area, then

```
seaLevel = round(bin.y_surf + 0.05, 3)
```

The +5 cm lifts the synthesised horizon quad just above the shipped sea tiles so the two cannot z-fight visibly; both draw with the same deep-water shading, so the overlap is invisible (`:238-239`).

Coordinate frame: extents are measured in **raw Unity world space** with no handedness conjugation, because `G3 = diag(-1, 1, 1)` preserves both Y and area (`:236-237`). Conjugating here changes nothing and is therefore omitted, not forgotten.

The result is patched straight into `manifest.json` after assembly (`build_map.py:1023-1039`); when no bin qualifies, an existing `seaLevel` key is **deleted** (`:1034-1036`) - an inland map must not inherit a stale ocean.

Measured: none of the packs present in this repo (interchange, woods, ground_zero, factory_rework, icebreaker, streets_nav) carries a `seaLevel`; they are all inland by this test.

### 4.6 The synthetic sea quad

`viewer/src/render/gpu_driven.rs:3351-3416`. Height source order (`:3352-3356`): `EFT_SEA_LEVEL` env override, then `manifest.seaLevel`; absent → the whole block is skipped and the render is byte-identical to a no-sea build.

Geometry: one quad centred on the manifest bounds centre, half-extents `(bounds_span/2 + 1200 m)` in each of X and Z (`:3358-3360`), 4 vertices and 6 indices, local origin at the centre with the height carried in the instance row (`:3380-3392`). The instance matrix is an identity 3×3 with translation `(cx, seaLevel, cz)` (`:3396-3398`). Vertex normal is `+Y` octahedral-encoded; UVs are `[0,1]²` across the whole quad.

Material (`:3362-3376`): `albedo_index = NO_ALBEDO` - this is what routes it to the shader's deep-water branch - `flags = MAT_FLAG_WATER` only (no `MAT_FLAG_BLEND`, so opaque with depth-write), `roughness = 0.08` for a near-mirror fresnel and a tight sun glint, `blend_class = 0` (`:3412`). The `tint` is `[1,1,1,1]` and is ignored by the deep-water branch.

The `Markers` pack tier force-clears `sea_level` (`viewer/src/eftpack.rs:1159`) precisely because the sea block runs *before* the empty-mesh bail and would otherwise build a full buffer and bind group around one giant water plane (`eftpack.rs:1107-1109`).

---

## 5. Invariants and failure signatures

| Invariant | Where | Failure signature |
|---|---|---|
| Cubemap data is face-major, stride = full mip chain, not mip 0 | `eft_extract_sky.py:66`, `:73` | face 0 correct, faces 1..5 diagonal garbage |
| BC1 decode output is BGRA and must be swizzled | `eft_extract_sky.py:74-75` | every sky is brown/orange; red and blue swapped |
| Faces are written in wgpu order 0..5 = +X,-X,+Y,-Y,+Z,-Z with **no** flip or rotation | `eft_extract_sky.py:72-93` | sky renders sideways; `horizon` samples zenith; `is_sky` inverts |
| `is_sky` uses face 2 vs face 3, ratio 1.3 | `eft_extract_sky.py:79` | reading face 0/1 flags every interior probe as sky |
| Derived colours are in 2.2-decoded space; `face_luma` is not | `eft_extract_sky.py:77`, `:82` | comparing them produces a ~2.2-gamma-off "sky is too dark" tune |
| `mean` is texel-uniform, not solid-angle weighted | `eft_extract_sky.py:87` | corner-biased irradiance on high-contrast probes |
| Particle keep = looping AND active-in-hierarchy (not playOnAwake) | `eft_extract_particles.py:223`, `:225` | adding playOnAwake removes fire sparks and ember columns |
| Gradient key times are uint16 ticks / 65535 | `eft_extract_particles.py:72`, `:77` | all keys land at t≈0; the gradient collapses to its first colour |
| Only the first 6 sorted key times survive | `eft_extract_particles.py:101` | dense gradients lose their tail; the fire never fades out |
| `pos` is already X-negated by the extractor | `eft_extract_particles.py:317` | a second conjugation mirrors every effect across the map |
| `pos` carries no rotation; the consumer must billboard | `eft_extract_particles.py:157-158` | quads render edge-on and disappear at most camera angles |
| `shapeRadius == 0` is rewritten to 0.3 | `eft_extract_particles.py:336` | point emitters scatter over a 0.3 m disc |
| A same-name atlas colliding in the low 32 path_id bits overwrites the earlier PNG | `eft_extract_particles.py:255-257` | both emitters render the last-written atlas |
| Particle alpha is clamped twice, inner and outer | `viewer/src/fx.rs:203`, `:244` | dropping the outer clamp blows cluster alpha past 1 whenever `grad.a > 1` |
| `tiles` is `[x, y]`, frames are row-major `f%tx, f/tx` | `viewer/src/fx.rs:365` | non-square sheets sample half-frames; a seam scrolls through the sprite |
| `rate` sizes the cluster, it does not spawn | `viewer/src/fx.rs:209` | reimplementing as a real emitter at rate 100 spawns 5000 quads |
| `m_Colors` holds all float4 properties, not just colours | `eft_extract_water4.py:74`, measured | `_GAmplitude` looked up under `floats` returns nothing; waves stay flat |
| `textures[*].scale/offset` are always `[0,0,0,0]` | `eft_extract_water4.py:44-45`, measured | bump map tiles at zero; the surface is a single flat colour |
| `_GDirectionAB/CD` are **not** unit vectors | measured, `Lighthouse_Water` `\|dir\|` = 0.901 / 0.354 / 0.949 / 0.707 | normalising shifts the four wavelengths by 11% / 183% / 5% / 41%; the waves beat visibly |
| Absent water properties are absent, never defaulted | `eft_extract_water4.py:14` | substituting Unity defaults invents water the game does not ship |
| Sea level is derived by boundary contact, not by area | `tools/build_map.py:344-353` | an inland lake qualifies and floods the map with a horizon quad |
| Sea level is measured in raw Unity space; no conjugation | `tools/build_map.py:236-237` | conjugating is harmless for Y but signals a misunderstanding downstream |
| Deep water is untextured → opaque pass | `gpu_driven.rs:2175-2178`, `:3363-3364` | routing it to blend makes glass composite wrong and the clear colour bleed through |
| Water normal is world-up, not the mesh normal | `gpu_draw.wgsl:1740` | screen-period crosshatch plus kilometre-scale dark fresnel bands |
| Procedural wave phase must go through `rsin` (fract first) | `gpu_draw.wgsl:614-616` | checkerboard and "shadow streak" beat bands at map edges |

---

## 6. Old patterns

- The sky extractor's docstring at `eft_extract_sky.py:11-12` describes a horizon-continuity term in the classifier. No such term was ever implemented; `:13-14` supersedes it and the shipped rule is the top/bottom luma ratio alone.
- The particle extractor's docstring at `eft_extract_particles.py:11` lists `playOnAwake` as a keep condition. The implementation at `:221-223` deliberately dropped it.
- The extracted sky cubemaps were once wired in as the visible sky dome. That was removed (`viewer/src/main.rs:1639-1644`): the assets are environment captures with treelines baked into the horizon, and as a dome they draw photographic trees behind real geometry. The exports survive only as a colour source.
- The deep-water ripple amplitude once had a hard cutoff band of `(150, 350)` m, which froze an entire lake from any flyover view. It was widened to `(500, 1400)` once the per-octave Nyquist gates took over anti-aliasing (`gpu_draw.wgsl:1676-1679`).
- The water drift speeds were first converted at 0.0075 cycles/s (0.13 m/s), 2.7× too slow; on an 18 m wavelength that reads as completely static (`gpu_draw.wgsl:1700-1702`).