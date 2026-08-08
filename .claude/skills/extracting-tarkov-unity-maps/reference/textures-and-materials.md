## Contents

1. [Stage boundaries and where each value is decided](#1-stage-boundaries)
2. [Texture export and naming](#2-texture-export-and-naming)
3. [The persistent texture cache (extraction side)](#3-the-persistent-texture-cache-extraction-side)
4. [Normal maps: DXT5nm unswizzle and the green flip](#4-normal-maps)
5. [UV: the `uv` = [sx, sy, ox, oy] tiling convention and the V-flip](#5-uv-tiling-and-the-v-flip)
6. [Material roles and the alphaMode / cutoff mapping](#6-material-roles)
7. [Alpha preservation: the RGBA-vs-RGB trap](#7-alpha-preservation)
8. [Alpha-coverage recovery (Otsu) and the normal-map-decal drop](#8-alpha-coverage-recovery)
9. [tint, PBR scalars, roughness sources](#9-tint-and-pbr-scalars)
10. [Vertex-paint `vp` splat materials](#10-vertex-paint-vp-splat-materials)
11. [Detail maps](#11-detail-maps)
12. [Parallax / height](#12-parallax--height)
13. [Emissive](#13-emissive)
14. [Legacy Transparent/Reflective/Specular glass (`glassTRS`)](#14-legacy-glass-glasstrs)
15. [materials.json schema](#15-materialsjson-schema)
16. [The material dedup key](#16-the-material-dedup-key)
17. [manifest.conventions - the self-describing contract](#17-manifestconventions)
18. [GPU-side consumption: flags, BC formats, mip skip](#18-gpu-side-consumption)
19. [Captured vs DROPPED material fields](#19-captured-vs-dropped)
20. [Invariants and their failure signatures](#20-invariants-and-failure-signatures)

> Claims marked **[unverified]** come from source-code comments, dataset measurements, or external engine conventions. They are not derivable from code in this repository. Everything else is anchored to a `file:line` that was read.

---

## 1. Stage boundaries

Three stages, three different responsibilities. Reimplementers must not move work across these lines, because each stage records a claim the next one trusts.

| Stage | Authority file | Produces |
|---|---|---|
| Unity extraction | `extraction/unity/eft_extract_v2.py` | `<dataset>/tex/*.png` (lossless, full-res), `<dataset>/meshes/*.obj` + `.vcol.npy`, and per-submesh material dicts inside `scene.json` |
| Pack assembly | `eft_pipeline/assemble_bevy.py` | `materials.json`, `meshes.bin` (UV tiling + V-flip **baked into vertex UVs**), `manifest.json` conventions block |
| Runtime import | `viewer/src/render/gpu_driven.rs`, `viewer/src/render/standard.rs` | GPU textures (BC3/BC5/raw), `GpuMaterial` records, flags |

Textures are **referenced in place by path** and never rewritten by the assembler (`eft_pipeline/assemble_bevy.py:23-25`). This is why every texture-space convention (V origin, green flip) has to be either baked into geometry or *recorded* in the manifest - the PNG on disk cannot be pre-corrected.

---

## 2. Texture export and naming

`exp_tex(tx, is_normal=False)` at `extraction/unity/eft_extract_v2.py:926`.

Output filename:

```
san(m_Name) + "__" + <asset_file_stem> + "_" + <path_id> + ".png"
```

- `san(s)` (`extraction/unity/eft_extract_v2.py:328-329`) is exactly `"".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))`. Python's `str.isalnum()` is **Unicode-aware**, so non-ASCII letters and digits (Cyrillic, accented Latin) survive unchanged - only punctuation, whitespace and symbols become `_`. This is load-bearing here: EFT ships Cyrillic-named Unity assets (`Сontainer_hospital`, with a Cyrillic С - see the UTF-8 OBJ-write comment at `eft_extract_v2.py:1234-1238`), so exported PNG filenames can and do contain non-ASCII characters. A reimplementation using an ASCII-only `[A-Za-z0-9._-]` filter produces different filenames and will not match an existing `tex/` directory.
- `<asset_file_stem>` and `<path_id>` come from `srcid()` (`:914`): the PPtr is **read** and the identity is taken from the *resolved* object's `assets_file.name` (basename, extension stripped, sanitized) plus `object_reader.path_id` (signed int64, printed in decimal). Using the PPtr's own `file_id` is wrong: `file_id == 0` means "same file as the referencing material", so distinct physical textures collapse onto one key and later ones are silently skipped.
- Dedup set `tex_done` is keyed on `(stem, path_id)`, not on the filename.
- Files land in `<EFT_ASSETS_ROOT>/<dataset>/tex/` (`:795`).

Encoding: PIL PNG. `_save_png` (`:140-144`) calls `img.save(path, format="PNG")` with **no explicit compression level** on the default path - Pillow's own default applies; `EFT_PNG_FAST` set → `compress_level=1`. PNG is lossless at every level, so this only trades size for speed.

Writes are atomic: `_atomic_write` (`:103`) writes `<final>.tmp<pid>_<tid>` and `os.replace`s it. Completion is tested by `_png_complete` (`:159`) - file ≥ 60 bytes **and** `IEND` present in the final 16 bytes. A plain `os.path.exists` check is not sufficient: a killed run leaves a valid IHDR followed by megabytes of NTFS preallocation zeros, which PIL's lazy `open()` accepts and which then survives every rebuild.

PNG encoding runs on a small thread pool (`_TexPool`, `:238`); default worker count `max(1, min(8, cpu_count // 3))` (`:67-73`), `EFT_TEX_WORKERS` overrides, `1` = the exact serial path. Only `img.save()` on a private `img.copy()` runs off-thread - every UnityPy/`texture2ddecoder` call stays on the calling thread.

Terrain layer and control textures follow a separate path into `<dataset>/terrain_layers/` (`splat_root`, `extraction/unity/eft_extract_v2.py:807`) and are resolved at runtime from the terrain sidecar via `add_tex` (`viewer/src/render/gpu_driven.rs:2428-2440`), not from `materials.json` records. **[unverified]** that no terrain layer PNG can ever *also* appear in some material's `albedo` field.

---

## 3. The persistent texture cache (extraction side)

`_texcache_key_path(to, is_normal)` at `extraction/unity/eft_extract_v2.py:76`.

Key = `blake2b(digest_size=16)` over:

1. `to.get_image_data()` - the **resolved** raw bytes. Most EFT textures are streamed, so `m_StreamData` must be followed into the `.resS`; the inline `image_data` is empty for them and is only the fallback.
2. then `update(f"|{w}x{h}|{fmt}|{'N' if is_normal else 'A'}|pil{PIL_VERSION}")`, where `fmt` is the integer `m_TextureFormat` (or `.value` if it is an enum).

Path: `<EFT_ASSETS_ROOT>/.texcache/<32-hex-digest>.png`. `EFT_TEXCACHE=0` disables, `EFT_TEXCACHE_DIR` relocates (`:62-63`).

Hit path (`:938-941`) `_link_or_copy`s the cached PNG into `tex/` (`os.link` first - same inode, zero bytes; atomic copy fallback across volumes, `:147`) and skips decode + encode entirely. Miss path decodes, saves, then `_publish_cache` (`:225`) hardlinks the new file into the cache, replacing an incomplete entry.

Content addressing means the cache self-invalidates on a game update, and two maps sharing a texture pay the decode once. The PIL version is in the key because a PIL upgrade changes the encoded PNG bytes.

**Consequence for consumers:** files in `tex/` may be hardlinks shared by several packs. Anything that rewrites a texture must write-temp-then-replace; editing in place mutates every pack sharing the inode.

---

## 4. Normal maps

### Unswizzle at export

`unswizzle_normal(img)` at `extraction/unity/eft_extract_v2.py:332`.

**[unverified, external Unity convention]** DXT5nm packs normal **X in ALPHA**, **Y in GREEN**, and leaves R at a near-constant value. The code at `:332-344` and its comments are written on that assumption; nothing in this repository proves it independently.

Detection (data-driven, no name rules): with `a = RGBA/255.0`,

```
is_dxt5nm  ⟺  mean(a[...,0]) > 0.95  AND  std(a[...,0]) < 0.06
```

Reconstruction:

```
X = a[...,3]*2 - 1
Y = a[...,1]*2 - 1
Z = sqrt(clamp(1 - X² - Y², 0, 1))
out = (XYZ * 0.5 + 0.5) * 255  →  PNG mode "RGB"
```

Non-DXT5nm inputs pass through untouched. The result is a standard 3-channel tangent normal; **alpha is intentionally dropped** here because X was moved out of it.

### Green flip is recorded, not applied

The PNG stays in **DirectX convention (green points down)**. `materials.json.normalGreenFlip` is set to `true` whenever a normal texture exists (`eft_pipeline/assemble_bevy.py:347`), and `manifest.conventions.normalMapGreenFlip = true` / `normalMapConvention = "directx"` (`:1333`) states it pack-wide.

Two valid implementations, exactly one of which may run:

- **At import**: invert G in the decoded image - `px[1] = 255 - px[1]` (`viewer/src/render/standard.rs:272-279`). Required when the target material system has no runtime flip.
- **In the shader**: `viewer/assets/shaders/gpu_draw.wgsl:1173-1177`

```
base_xy = tex.xy * 2 - 1
if (normal_flags & 1) { base_xy.y = -base_xy.y }
base_xy *= normalScale
base_ts  = vec3(base_xy, sqrt(max(1 - dot(base_xy,base_xy), 1e-4)))
```

Note `.xy` only: normals are uploaded BC5 (two-channel), so `.z` reads 0 and Z must be reconstructed. The same reconstruction is correct for legacy raw-RGB normals, so it is a drop-in.

The flag is OR'd from the per-material field and the pack convention: `mat.normal_green_flip || conv_green_flip` (`viewer/src/render/gpu_driven.rs:1986-1989`).

`normalScale` = Unity `_BumpScale`, captured only when it differs from 1.0 by more than 1e-3 (`extraction/unity/eft_extract_v2.py:1104`); default 1.0 in `materials.json` (`eft_pipeline/assemble_bevy.py:343`).

Detail normals decode with the **same** green-flip bit and are combined with the base in **tangent space** via `blend_rnm` before a single TBN transform (`viewer/assets/shaders/gpu_draw.wgsl:1188-1207`). Applying the TBN twice is the classic double-transform bug.

---

## 5. UV tiling and the V-flip

### What `uv` holds

`sub['uv'] = [scale_x, scale_y, offset_x, offset_y]` - Unity's `_MainTex_ST`, read from the **TexEnv of the first matching albedo slot** and rounded to 4 decimals (`extraction/unity/eft_extract_v2.py:994-1001`):

```
tile = [tenv.m_Scale.x, tenv.m_Scale.y, tenv.m_Offset.x, tenv.m_Offset.y]
```

Albedo slot priority (first hit wins), `extraction/unity/eft_extract_v2.py:41-42`:
`_MainTex`, `_Diffuse`, `_BaseMap`, `_AlbedoMap`, `_MainTex0`, `_BaseAlbedoASmoothness`, `_TopAlbedoASmoothness`, `_Albedo`, `_Aldebo` *(the game shader misspells it)*, `_Tex`, `_BaseColorMap`, `_MainTexture`.

Normal slots, `:44`: `_BumpMap`, `_NormalMap`, `_Normalmap`, `_Normal`, `_BaseNormalMap`, `_BumpMap0`, `_TopNormalMap`.

Default when no slot binds a texture: `[1, 1, 0, 0]`.

### The bake - exact formula

`eft_pipeline/assemble_bevy.py:1002-1006`, per vertex, applied **in this order**:

```
u_out = u_obj * sx + ox
v_out = 1.0 - (v_obj * sy + oy)
```

where `(u_obj, v_obj)` is the OBJ `vt` (Unity mesh UV, **bottom-left origin**). The V-flip converts to the top-left origin used by PNG row order and by wgpu/glTF samplers. It is applied **after** tiling, which makes it a texture-space flip that matches Unity's `tex2D` fetch - flipping before tiling shifts by the offset and is wrong.

Both facts are declared in the manifest so no consumer double-applies them:
`uvVFlipBaked: true`, `uvOrigin: "top-left"`, `uvTilingBaked: true`, and `uvXformNote: "materials.json.uvXform is REFERENCE ONLY"` (`eft_pipeline/assemble_bevy.py:1331-1332`).

`materials.json.uvXform` still carries `[sx, sy, ox, oy]` - **not to be applied**, but needed to *un-bake* when a secondary map has its own ST.

### Un-baking / re-basing a secondary ST

`detail_xform(base_st b, target_st d)` - `viewer/assets/shaders/gpu_draw.wgsl:906-917`:

```
bsx = |b.x| > 1e-6 ? b.x : 1.0
bsy = |b.y| > 1e-6 ? b.y : 1.0
rx  = d.x / bsx
ry  = d.y / bsy
out = ( rx, ry, d.z - b.z*rx, 1 - d.w - ry*(1 - b.w) )
uv_target = uv_baked * out.xy + out.zw
```

Derivation of the w lane (this is the part naive code gets wrong): `v_baked = 1 - (v_raw*bsy + b.w)` ⇒ `v_raw = ((1 - v_baked) - b.w)/bsy`; the target's own baked V is `1 - (v_raw*d.y + d.w)`, which expands to `ry*v_baked + (1 - d.w - ry*(1 - b.w))`. A naive `(uv - zw)/xy` un-bake ignores the flip and shifts by up to half a tile.

Sampler addressing **must be Repeat** in U and V: tiling is baked, so UVs routinely run far outside `[0,1]`. `viewer/src/render/gpu_driven.rs:5129-5137` (Repeat, aniso 8) and `viewer/src/render/standard.rs:286-294` (Repeat, trilinear, aniso 16). ClampToEdge smears the last texel row into long streaks.

Vertex dedup after the bake keys on `round(pos, 3)` concatenated with `round(uv, 3)` (`eft_pipeline/assemble_bevy.py:1008`), so UV seams are preserved as duplicate positions.

### Vertex layout the UVs live in

`eft_pipeline/assemble_bevy.py:88-95` - stride **36 bytes**, little-endian:

| offset | field | format |
|---|---|---|
| 0 | position | f32x3 |
| 12 | normal | f32x3 |
| 24 | uv | f32x2 |
| 32 | colour | unorm8x4 |

One UV set only. There is no lightmap/secondary UV channel.

---

## 6. Material roles

`ROLES = ('opaque', 'cutout', 'glass', 'decal', 'water')` (`eft_pipeline/assemble_bevy.py:160`).

### Primary classification (extraction, authoritative)

`extraction/unity/eft_extract_v2.py:1019-1047`. Inputs: the shader's `m_ParsedForm.m_SubShaders[0].m_Tags["RenderType"]`, `m_CustomRenderQueue`, `_Cutoff`, and the shader name.

```
rt = RenderType.lower()
rt == "transparentcutout"          -> cutout
rt == "transparent"                -> glass if renderQueue >= 2900 else decal
otherwise                          -> opaque

shader contains "transparent" AND "dithered"   -> glass    (overrides the tag)
shader contains "water"                        -> water    (overrides everything)
```

`cut` = `_Cutoff` rounded to 3 decimals, default **0.5**.

The dithered override exists because EFT's depth-writing dithered glass tags itself `TransparentCutout` (it discards on a dither pattern so it can write depth); alpha-testing it turns glass-block walls into a pegboard. Classify by the shader, never by the tag, for that family.

### Assembly-side corrections

`eft_pipeline/assemble_bevy.py:960-969`: a mesh whose name contains `water`/`lake`/`pond`/`river`/`ocean` and whose sub has an empty shader **and** no albedo → `role = 'water'`, `sh = 'water'`. Any sub whose shader names water → water.

`:983-987`: opaque + not `smA` + the albedo alpha passes the Otsu coverage test → `role = 'cutout'`, with `cut` = the **authored** `_Cutoff` if present, else the Otsu split (see §8).

### Role → alphaMode / cutoff

`eft_pipeline/assemble_bevy.py:307-311`:

| role | alphaMode | alphaCutoff |
|---|---|---|
| opaque | `OPAQUE` | 0.0 |
| cutout | `MASK` | `round(cut or 0.5, 4)` |
| glass | `BLEND` | 0.0 |
| decal | `BLEND` | 0.0 |
| water | `BLEND` | 0.0 |

Cutout is an **opaque-pass** material (alpha test, depth write on). Blend is a separate, depth-write-off pass. The two are disjoint in the GPU flags (`MAT_FLAG_CUTOUT` vs `MAT_FLAG_BLEND`, `viewer/src/render/gpu_driven.rs:305-309`).

Alpha-test is evaluated on the **computed** albedo alpha (`tex.a * tint.a`, or `tint.a` when untextured), so an untextured cutout with `tint.a < cutoff` still discards (`viewer/assets/shaders/gpu_draw.wgsl:1341-1343`). The hard discard sits at `0.5 * alphaCutoff` because the remaining half of the ramp is handled by alpha-to-coverage.

Glass additionally forces `roughness = 0.05`, `metallic = 0.0`, and `tint.a` = the authored `_Color.a`, or **0.28** when the material authored no 4-component colour (`eft_pipeline/assemble_bevy.py:312-316`).

Decals are coplanar overlays: the Standard path biases them toward the camera with `depth_bias = 4.0` (`viewer/src/render/standard.rs:386`); the GPU-driven path uses a per-material NDC push in the unified surface pass.

---

## 7. Alpha preservation

**The trap:** the albedo alpha channel is load-bearing in three mutually exclusive ways depending on the shader family - coverage (cutout holes), smoothness (Unity Standard packing), or transparency×gloss (legacy TRS glass). Converting an albedo to RGB anywhere in the chain destroys it irrecoverably, and nothing downstream errors.

Rules that must hold:

1. **Export writes whatever mode the decode produced.** `img.save(path, "PNG")` (`extraction/unity/eft_extract_v2.py:140-144`). Never `.convert("RGB")` an albedo.
2. `unswizzle_normal` **does** return mode `"RGB"` (`:341`) - correct and only correct for normals, because X was lifted out of alpha into the reconstructed RGB.
3. The coverage detector requires an alpha channel: `if im is not None and im.mode == 'RGBA'` (`eft_pipeline/assemble_bevy.py:228`). An RGB-saved foliage atlas can never be promoted to `cutout`.
4. `_detail_mean` converts to RGB (`eft_pipeline/assemble_bevy.py:424`) **only** to compute a mean; the shipped texture is untouched.
5. The viewer always calls `img.to_rgba8()` (`viewer/src/render/gpu_driven.rs:6048`, `:6678`). An RGB PNG silently becomes alpha ≡ 1.0. So a stripped alpha does **not** crash - it turns leaf cards into solid rectangles and pins roughness-from-alpha at the constant `1 - 1 = 0` (clamped to 0.06, i.e. mirror-smooth).
6. Albedo compresses to **BC3**, not BC1, specifically because BC3 keeps a full interpolated 8-bit alpha block (`viewer/src/render/gpu_driven.rs:6174-6192`).
7. Two runtime probes read alpha straight off the PNG and both invert their decision if alpha is missing: `glass_alpha_is_mask` (`:6015`) and `puddle_alpha_is_constant` (`:5855`).

Alpha semantics resolution order, per material:

- `glassTRS` set → tex.a is **transparency × gloss**; `roughnessFromAlbedoAlpha` is force-cleared (`eft_pipeline/assemble_bevy.py:370-372`).
- `role == cutout` → tex.a is **coverage**.
- `smA` set (shader name contains `specular`/`smap`, or the bound albedo slot name contains `asmoothness`) → tex.a is **smoothness** (`extraction/unity/eft_extract_v2.py:1118-1122`), emitted as `roughnessFromAlbedoAlpha: true`. Shader: `rough = clamp(1 - tex.a, 0.06, 1.0)` on the **raw** `tex.a`, never `albedo.a`, since `tint.a` would bias it (`viewer/assets/shaders/gpu_draw.wgsl:1430-1432`).
- vertex-paint materials → tex.a is smoothness; the alpha gate is cleared entirely (§10).

---

## 8. Alpha-coverage recovery

`_TexTest.alpha_coverage(name)` - `eft_pipeline/assemble_bevy.py:209-265`. Purpose: EFT's custom shaders (SpeedTree-derived foliage, Cloth ground overlays) do not tag `TransparentCutout`, so they arrive as `opaque` and render as solid black cards. The texture's own alpha histogram decides, with no shader lists and no fixed cutoff.

Procedure: 256-bin histogram of `A/255` over the full-res image; standard Otsu between-class variance; `t` = argmax bin; `lv[i] = (i + 0.5)/256`.

Accept as coverage iff **all five** hold:

| test | threshold | meaning |
|---|---|---|
| `between[t] / total_var` | ≥ 0.5 | alpha is clearly bimodal |
| `m0[t]` (low-mode mean) | ≤ 0.1 | the low mode is true holes (AO/height alpha rarely reaches zero) |
| `m1[t]` (high-mode mean) | ≥ 0.3 | the kept class is meaningfully opaque |
| `sum(w[243:])` (alpha ≥ 0.95) | ≥ 0.005 | some texels are *fully* opaque |
| `w_lo` (cumulative weight at `t`) | in `[0.005, 0.995]` | both classes non-trivial |

Returned cutoff = `lv[t]`. The fourth test separates real masks from smooth masks like fire/AO. **[unverified]** the source comment quotes the measured separation as 1.06 %–99.6 % of texels above alpha 0.95 for real masks versus 0.01 % for smooth ones - a hundred-fold gap - and reports that without this test a fire mask took `cutout` at Otsu 0.22, the material's authored `_Cutoff` of 0.5 then discarded 98.6 % of the surface, and a burning prop rendered see-through. These are dataset measurements, not code.

Priority: the material's authored `_Cutoff` **wins** over the Otsu split (`eft_pipeline/assemble_bevy.py:987`).

### Normal-map-albedo decals

`albedo_is_normalmap(name)` - `eft_pipeline/assemble_bevy.py:196-207`. Downsample to 8×8 then 1×1, read the single RGB pixel; true when

```
b > 200 AND |r-128| < 45 AND |g-128| < 45 AND b > r+55 AND b > g+55
```

A `decal` matching this is a deferred bevel decal whose "albedo" is a normal map; it is dropped (`:778-781`). The drop must still **consume its face range** (`f0 += n`, `:997`) or every later submesh in the mesh reads a shifted triangle span - the failure signature is a see-through hole plus a neighbouring material drawing the decal's faces.

---

## 9. tint and PBR scalars

`_col4(col)` - `eft_pipeline/assemble_bevy.py:166-174`. Unity `_Color` (or `_BaseColor`/`_TintColor`/`_MainColor`, `extraction/unity/eft_extract_v2.py:1010`) is **sRGB**; RGB is converted to linear, alpha is passed through unchanged (it is coverage/opacity, not colour):

```
lin(c) = c/12.92                      if c <= 0.04045
       = ((c + 0.055)/1.055)^2.4      otherwise
tint = [lin(r), lin(g), lin(b), round(a, 4)]
```

Final albedo is `texture_sample * tint` (`viewer/assets/shaders/gpu_draw.wgsl:1237-1240`), i.e. `_MainTex * _Color`. Untextured materials shade with `tint` over implicit white.

`_pbr(shader, role)` - `eft_pipeline/assemble_bevy.py:177-183`, shader-string only (map-agnostic):

| condition | roughness | metallic |
|---|---|---|
| role ∈ {water, glass} | 0.05 | 0.0 |
| shader contains `chrome` or `metal` | 0.4 | 0.85 |
| shader contains `specular`, `reflective`, `smap` | 0.55 | 0.0 |
| else | 0.9 | 0.0 |

Authored overrides (`:294-304`): `_Glossiness` → `roughness = clamp(1 - gloss, 0.02, 1.0)` **only** when the shader name contains one of `specular, reflective, smap, chrome, metal, standard`. The gate matters: the extractor records a fabricated default `gloss = 0.5` for shaders with no `_Glossiness` property, and letting that through made pure-Diffuse fabrics (camo nets) glossy. `_Metallic` → `metallic = clamp(metal, 0, 1)` unconditionally.

Runtime clamps `roughness` to `[0.03, 1.0]`, default 0.55 when absent (`viewer/src/render/gpu_driven.rs:2330-2334`).

`specMap` carries the `_SpecMap`/`_SpecTex`/`_GlossMap` path (legacy Unity specular convention: RGB = specular colour, A = gloss) for consumers that want per-pixel roughness where the albedo alpha is unusable (`extraction/unity/eft_extract_v2.py:1113-1117`).

`doubleSided` is hardcoded `true` for every material (`eft_pipeline/assemble_bevy.py:348`): EFT's deferred renderer draws building shells solid from both sides. Back faces flip the shading normal (`viewer/assets/shaders/gpu_draw.wgsl:1162-1164`).

---

## 10. Vertex-paint `vp` splat materials

### Detection and capture

Triggered by the presence of `_MainTex1` or `_MainTex2` in the material's TexEnvs (`extraction/unity/eft_extract_v2.py:1053`). Exactly **3 layers** are captured, indices 0/1/2:

- `tex` ← `_MainTex{i}`, `nrm` ← `_BumpMap{i}` (exported with `is_normal=True`)
- `uv` ← that layer's own TexEnv `[m_Scale.x, m_Scale.y, m_Offset.x, m_Offset.y]`, default `[1,1,0,0]`
- `col` ← `_Color{i}` as RGB (3 components), default `[1,1,1]`
- `heights` ← the `_Heights` slot (a control mask; R/G/B = per-layer coverage)
- `blend` ← `_BlendStrength`, default 1.0, rounded to 3

SoftCutout triple, written **only** when `_AlphaStrength` is authored (`:1073-1076`) - absent (engine-default feathering) and explicit `0` (gate off) are different render paths and must not be conflated:

```
softCutout = [ _AlphaStrength, _Cutoff (default 0), _AlphaHeight (default 0) ]
```

`materials.json` emission at `eft_pipeline/assemble_bevy.py:435-449`.

### COLOR_0

The blend weights live in the mesh's vertex colours. `eft_pipeline/assemble_bevy.py:1013-1021`: vp submeshes read the `.vcol` sidecar (float RGBA per source vertex), take `vcol[vi][idx0]`, clamp to `[0,1]`, and quantize `round(x * 255)` into `unorm8x4`. Non-vp submeshes get a constant `(255,255,255,255)`; baked degenerate geometry likewise (`:1098`). Unpainted/missing vcol falls back to `(1,0,0,1)` in float, i.e. full layer 0.

Do **not** collapse white vertex colours - for vp materials white is a meaningful weight.

### The blend (reverse-engineered from the DX11 fragment)

`viewer/assets/shaders/gpu_draw.wgsl:1265-1310`, table layout `viewer/src/render/gpu_driven.rs:408-429` (`VpGpu`, 112 bytes):

```
x1 = detail_xform(uv0, uv1)              # layers 1,2 re-based off the baked layer-0 ST
x2 = detail_xform(uv0, uv2)
xh = detail_xform(uv0, [1,1,0,0])        # heights mask in the RAW frame

h  = heights.rgb sampled at uv*xh.xy + xh.zw   (identity vec3(1) if no mask)
hw = h * COLOR_0.rgb
if (hw.x+hw.y+hw.z > 1e-5):
    w = pow(max(hw, 1e-4), max(blend, 1.0));  w /= sum(w)
else:
    w = (1, 0, 0)                        # unpainted / "Solid" variant -> base layer

albedo.rgb = ( w0*a0.rgb*tint0 + w1*a1.rgb*tint1 + w2*a2.rgb*tint2 ) * material.tint
vp_smooth  = w0*a0.a + w1*a1.a + w2*a2.a
roughness  = clamp(1 - 0.30*vp_smooth, 0.72, 1.0)
```

Layer 0 samples at the baked `uv` directly (its ST is what was baked). If the resolved colour's luma `dot(spl, (0.299, 0.587, 0.114))` falls below 0.02, fall back to layer 0 alone.

The heights mask is uploaded **LINEAR**, not sRGB - it is blend weights, not colour (`viewer/src/render/gpu_driven.rs:2123-2136`).

### Two variants, one param triple

`Custom/Vert Paint SoftCutout **Decal**` (role `decal`) is a blended, feathered road/track overlay. Coverage comes from **COLOR_0.a**, never `tex.a` (tex.a is smoothness for this family):

```
coverage = clamp(COLOR_0.a * astr - (acut - ahgt), 0, 1) * COLOR_0.a
```

The trailing `* COLOR_0.a` is **not** redundant - the shader's own comment states its purpose: it keeps feather tails soft where `_AlphaStrength` would re-saturate them. Shipped at `viewer/assets/shaders/gpu_draw.wgsl:1516` (DECAL_DEPTH_PASS) and `:1523` (DECAL_COLOR_PASS). The Rust doc-comment at `viewer/src/render/gpu_driven.rs:221-224` writes the formula **without** that factor; the shader is authority, and copying the Rust comment gives visibly harder feather edges.

SoftCutout does **not** run in the BLEND pass: `gpu_draw.wgsl:1228` discards softcutout from BLEND_PASS. It is drawn by two dedicated pipelines - DECAL_DEPTH (alpha-to-coverage into a depth-only prepass, colour writes off) and DECAL_COLOR (premultiplied `vec4(col * coverage, coverage)`, stronger depth bias, depth-test but no depth-write, so overlapping roads blend in stable phase order).

`Vert Paint Shader **Solid**` shares the same param triple but is an opaque splat with **no** alpha gate.

Gate on `role == "decal"` (`:2066-2076`). Force-blending the Solid variant clamps coverage to 0 when `astr == 0` and whole courtyard slabs vanish. Conversely, any vp material that is not a decal must have `MAT_FLAG_CUTOUT` cleared (`:2169-2171`) - its tex.a is smoothness, and the Otsu detector mis-tags some of them as `cutout` with an impossible `_Cutoff` of 1.3. Gate that clear on the **vp block**, not on the successfully-built splat table: a material with one null layer never gets `MAT_FLAG_VP` and would otherwise slip through.

vp materials are excluded from detail maps (`eft_pipeline/assemble_bevy.py:398`), parallax (`:389`), and emissive (`:321`).

---

## 11. Detail maps

Two authoring conventions coexist; both are handled by slot/float name only (`extraction/unity/eft_extract_v2.py:1162-1197`):

- **Unity Standard**: `_DetailAlbedoMap` / `_DetailNormalMap`, tiling in the TexEnv `_ST`, strength in `_DetailNormalMapScale`.
- **ANGRYMESH PBR Rocks**: `_DetailAlbedo` / `_DetailNormalMap` with `ST == identity` and the real tiling in the floats `_DetailUVScale` / `_DetailNormalUVScale`, intensities in `_DetailAlbedoIntensity` / `_DetailNormalMapIntensity`.

`_det_uv(tenv, scale_float)` (`:1170-1179`): take the TexEnv ST; **if** it equals `[1,1,0,0]` and the named float exists and is `> 0`, substitute `[s, s, 0, 0]`.

`materials.json.detail` (`eft_pipeline/assemble_bevy.py:394-410`):

```
{ "albedo": <path>, "albedoUv": [sx,sy,ox,oy],  "albedoStrength": float (default 1.0),
  "albedoMeanGain": [r,g,b],
  "normal": <path>, "normalUv": [sx,sy,ox,oy], "normalScale": float (default 1.0) }
```

`albedoUv`/`normalUv` are the **RAW** Unity `_Detail*Map_ST` - they must be re-based through `detail_xform` against `uvXform` before use, because the base ST is already baked into the vertex UVs.

`albedoMeanGain` (`eft_pipeline/assemble_bevy.py:412-433`): open the detail albedo, `convert('RGB')`, `thumbnail((256,256))`, `/255`, sRGB→linear with the same piecewise curve as `_col4`, per-channel mean, **× 4.5948** (Unity Standard's detail ×2 expressed in linear space), rounded to 5. Fallback `[1,1,1]` on any read failure.

Shader application (`viewer/assets/shaders/gpu_draw.wgsl:1318-1337`):

```
uv_d    = uv * da.xy + da.zw                    (da = detail_xform(uvXform, albedoUv))
neutral = (detail_lin * 4.5948) / max(albedoMeanGain, 1e-3)
weight  = clamp(albedoStrength * fade, 0, 1)
albedo.rgb *= mix(1, clamp(neutral, 0.25, 4.0), weight)      # alpha untouched
```

`fade = 1 - smoothstep(near, far, distance(camera, world_pos))`, default window **40 m → 120 m** (`viewer/src/render/gpu_driven.rs:2254-2261`, `EFT_DETAIL_FADE="near,far"`). Detail is mutually exclusive with terrain splat (`:2514` clears `MAT_FLAG_DETAIL | MAT_FLAG_RFA` on terrain materials).

Without mean-neutralization a dark detail map darkens the surface roughly 2× under Unity's ×2 blend.

Texture sampling uses `textureSampleGrad` with gradients scaled by the **relative** UV scale (`duv_dx * da.xy`), because the sample sits in non-uniform control flow.

---

## 12. Parallax / height

`extraction/unity/eft_extract_v2.py:1198-1209`: the `_ParallaxMap` slot plus the `_Parallax` float (default 0.02, 5 decimals). **[unverified]** a source comment puts the population at roughly 44 materials game-wide.

`materials.json.parallax = { "map": <path>, "scale": float }` (`eft_pipeline/assemble_bevy.py:386-392`), `null` for vp materials or when no map is bound.

Height is **DATA**, uploaded LINEAR: the runtime inserts the height map into the same bindless albedo array but registers it in `ctrl_tex_linear` (`viewer/src/render/gpu_driven.rs:2288`), which forces a raw linear upload and blocks BC. It does **not** set `no_downscale`, so the height map *is* mip-skipped at reduced texture quality (§18). Scale is clamped to `[0, 0.5]` (`:2289`). `EFT_PARALLAX=0` masks the flag for every material, giving a byte-identical A/B against the non-parallax render.

The shader marches a tangent-space view ray (steep parallax/occlusion) and produces `puv`, which then feeds the **base albedo and base normal** samples - note `viewer/assets/shaders/gpu_draw.wgsl:1173` samples the normal at `puv`, not `o.uv`. `puv == o.uv` when the flag is clear.

---

## 13. Emissive

Unity's emission-**enable** rule is reproduced, because materials whose emission is off still serialize a stale `_EmissionColor` (`_emission_enabled`, `extraction/unity/eft_extract_v2.py:952-968`):

1. shader name contains `emissive` → **on** (EFT bakes emission into the shader variant with no keyword);
2. `_EMISSION` in `m_ValidKeywords` → on; in `m_InvalidKeywords` → off;
3. legacy space-joined `m_ShaderKeywords` string contains `_EMISSION` → on;
4. no keyword info and a non-emissive shader → **off**.

Capture (`:1084-1101`): texture from the first of `_EmissionMap`, `_EmissiveMap`, `_Emission`; colour from `_EmissionColor` as RGB (HDR, values above 1 are normal - lamps ship e.g. `[1.5, 1.82, 2.34]`). A standalone colour is honoured **only** when emission is enabled; when an emission *map* is present the colour is merely its HDR factor and is kept regardless.

Assembly (`eft_pipeline/assemble_bevy.py:318-329`), skipped for roles `decal` and `glass` (BLEND shaders repurpose `_EmissionColor` as a tint) and for vp materials:

```
mx = max(emisCol) or 1.0
if map and colour and mx > 1:  factor = [min(c/mx, 1) for c in colour];  hdr = round(mx, 3)
elif colour and max > 0:       factor = [min(c, 1) for c in colour];     hdr = 1.0
else:                          factor = [1,1,1];                          hdr = 1.0
emissive = { "texture": <path|null>, "factor": [r,g,b], "hdr": float }
```

Runtime: `emissive_rgb = factor * hdr` precomputed on the CPU, texture placed in the **sRGB** albedo array (`conventions.colorSpace.emissive == "srgb"`), final contribution `em_tex * emissive_rgb * ui_scale` added after lighting (`viewer/src/render/gpu_driven.rs:2199-2211`, `viewer/assets/shaders/gpu_draw.wgsl:1131-1137`).

`emissive` is a JSON **object or null** - never a bare string. Modelling it as a string aborts the whole `materials.json` parse and the pack fails to load (`viewer/src/eftpack.rs:346-350`).

---

## 14. Legacy glass (`glassTRS`)

Family detection: shader name (lowercased) contains `transparent` **and** (`reflective` **or** `dithered`) - `extraction/unity/eft_extract_v2.py:1133`. This is EFT's car and storefront glass shader; in it **tex.a is transparency × gloss**, never smoothness.

Captured, all presence-gated so non-glass materials re-extract identically:

| key | source | note |
|---|---|---|
| `glassTRS` | - | `1` marks the family semantics |
| `reflCol` | `_ReflectColor` RGBA | authored cubemap-reflection tint (dark on car glass) |
| `specCol` | `_SpecColor` RGB | Blinn-Phong response |
| `shin` | `_Shininess` | 0..1 |
| `opacS` | `_OpacityScale * _AlphaMult` | dithered family only; scales tex.a **before** the dither |
| `reflCube` | mean linear RGB of `_Cube` | see below |

`cube_mean_rgb(pptr)` - `extraction/unity/eft_extract_v2.py:270-318`. **[unverified, external Unity convention]** cubemaps are stored **face-major** (face 0's full mip chain, then face 1's, …); the byte-offset arithmetic below is written on that assumption and nothing in this repository proves it.

Bytes come through `get_image_data()` (resolves `m_StreamData`). Supported `m_TextureFormat` values and block sizes: `10 → BC1 (8 B/4×4)`, `12 → BC3 (16 B)`, `25 → BC7 (16 B)`, `4 → RGBA32 (w*h*4)`. Per-face chain length = `Σ_{m<mips} mip_bytes(face>>m, face>>m)`; face *i*'s top mip starts at `i * chain`. The BC decoders return **BGRA**, so RGB is taken as `[..., 2::-1]`. Each face contributes `mean((px/255)^2.2)`; the result is the sum divided by 6, rounded to 5. Any failure → `None` → the consumer keeps its analytic environment.

Assembly (`eft_pipeline/assemble_bevy.py:365-384`) sets `glassTRS: true`, force-clears `roughnessFromAlbedoAlpha`, and emits `opacityScale`, `reflectCube`, `reflectColor`, `specColor`, `shininess` when present.

Runtime packing (`viewer/src/render/gpu_driven.rs:2296-2331`):

```
rc = reflectColor  (default [0.5,0.5,0.5,0.5])
if reflectCube:  rc[k] *= cube[k]  for k in 0..3;  set MAT_FLAG_GLASS_CUBE
glass_refl = (opac<<24) | (R<<16) | (G<<8) | B      # 8-bit each; opac = round(clamp(opacityScale,0,8)/8 * 255)
glass_spec = (R<<16) | (G<<8) | B                   # specColor, default 0.5 grey
glass_shin = clamp(shininess, 0.01, 1.0)            # default 0.078 (legacy shader UI default)
roughness  = clamp( sqrt(2 / (glass_shin*128 + 2)), 0.03, 1.0 )     # Blinn-Phong power -> GGX
```

For packs **without** this capture, glass alpha semantics are probed per texture: `glass_alpha_is_mask` (`viewer/src/render/gpu_driven.rs:6015-6028`) samples every 101st pixel and returns true when more than 40 % have `alpha < 26`. True → coverage-mask glass (shard atlases), which masks every lighting term including the additive reflection; false → smoothness-in-alpha. A `glassTRS` capture is authoritative and skips the probe entirely.

---

## 15. materials.json schema

One JSON array; array index == `id` == the `materialId` referenced by every `manifest.meshes[].submeshes[]` entry. Written with `separators=(',', ':')` (`eft_pipeline/assemble_bevy.py:1141`). Schema built at `:332-384`, consumed at `viewer/src/eftpack.rs:405-482`.

```jsonc
{
  "id": 0,                       // == array index
  "role": "opaque",              // opaque | cutout | glass | decal | water
  "albedo": "…/tex/<name>.png",  // or null
  "normal": "…/tex/<name>.png",  // or null
  "uvXform": [sx, sy, ox, oy],   // REFERENCE ONLY - already baked into vertex UVs
  "alphaMode": "OPAQUE",         // OPAQUE | MASK | BLEND
  "alphaCutoff": 0.0,            // meaningful only for MASK
  "tint": [r, g, b, a],          // rgb LINEAR, a linear/coverage
  "metallic": 0.0,
  "roughness": 0.9,
  "normalScale": 1.0,            // Unity _BumpScale
  "normalGreenFlip": true,       // DirectX green-down; flip on import OR negate n.y
  "doubleSided": true,           // always true
  "emissive": null,              // or { texture, factor[3], hdr }
  "roughnessFromAlbedoAlpha": false,   // roughness = 1 - tex.a
  "specMap": null,               // _SpecMap path (RGB spec colour, A gloss)
  "vp": null,                    // or { layers[3], heights, blend, softCutout? }
  "detail": null,                // or { albedo, albedoUv, albedoStrength, albedoMeanGain,
                                 //      normal, normalUv, normalScale }
  "parallax": null,              // or { map, scale }

  // present only on the legacy glass family:
  "glassTRS": true,
  "opacityScale": 4.0,
  "reflectCube": [r, g, b],
  "reflectColor": [r, g, b, a],
  "specColor": [r, g, b],
  "shininess": 0.078
}
```

Numeric rounding: `uvXform`/`tint`/detail UVs to 6 decimals; `alphaCutoff`, `metallic`, `roughness`, `normalScale`, `opacityScale`, `shininess`, detail strengths to 4; `parallax.scale` to 5; `emissive.factor` to 4 and `hdr` to 3; `reflectCube` to 5; `albedoMeanGain` to 5.

Texture paths are absolute POSIX-slashed paths into `<dataset>/tex/` by default (`eft_pipeline/assemble_bevy.py:276-277`). With `--self-contained` every texture-bearing field is rewritten to pack-relative `tex/<basename>` and the file is hardlinked/copied in (`_self_contain_materials`, `:633-651`; covers `albedo`, `normal`, `specMap`, `emissive.texture`, `detail.albedo/.normal`, `vp.layers[].albedo/.normal`, `vp.heights`). Basename collisions between different files get a deterministic `<stem>.<sha1[:8]>.png` suffix; identical content shares one copy (`ship_tex`, `:592-618`). A consumer must resolve relative paths against the pack directory and pass absolute paths through unchanged.

---

## 16. The material dedup key

`sub_sig(subs)` - `eft_pipeline/tarkmap_core/matsig.py`. `MaterialFactory.get()` caches on this key (`eft_pipeline/assemble_bevy.py:279-286`), and geometry grouping uses the *same* key space, so a mesh reused with different textures splits into per-material groups instead of taking the first instance's material.

The tuple, per submesh:

```
(tex, nrm, tuple(col), sh, role, round(cut, 3), tuple(uv), vp_sig(vp), n,
 emis, tuple(emisCol), gloss, metal, bumpScale,
 spec, smA, detA, detN, tuple(detAuv), tuple(detNuv), detAI, detNS)
```

`vp_sig` (same file) folds the 3 layers' `(tex, nrm, uv, col)` plus `heights` and `blend`.

`n` (the submesh face count) is included so a differing geometry split cannot collapse two materials.

**Known limitation:** the `glassTRS` response fields (`reflCol`, `specCol`, `shin`, `opacS`, `reflCube`) and the parallax fields (`par`, `parS`) are **not** in the key. Two materials that share textures, colour, shader and role but authored different `_SpecColor` will collapse into one record. In practice the shader name (`sh`) is in the key and the family is shader-determined, so the collapse is confined to same-shader variants.

---

## 17. manifest.conventions

`eft_pipeline/assemble_bevy.py:1328-1336`. This block is the contract; nothing downstream may hardcode the opposite.

```jsonc
"conventions": {
  "affine": "ROW-MAJOR world 3x4 incl shear",
  "normals": "LOCAL smooth normals; renderer applies per-instance inverse-transpose of the 3x3",
  "uvVFlipBaked": true,  "uvOrigin": "top-left",
  "uvTilingBaked": true,
  "uvXformNote": "materials.json.uvXform is REFERENCE ONLY; tiling already baked into vertex UV",
  "normalMapGreenFlip": true, "normalMapConvention": "directx",
  "colorSpace": { "albedo": "srgb", "normal": "linear", "emissive": "srgb" },
  "textureImport": "BC7 (albedo/emissive sRGB), BC5 (normal, linear); referenced in place"
}
```

Loader defaults for all three booleans are `true` (`viewer/src/eftpack.rs:217-242`), so a pack that omits the block still behaves correctly.

`textureImport` is **advisory prose**, not a directive: the shipped importer uses **BC3** for albedo/emissive and **BC5** for normals (§18). Treat the string as documentation of intent and the loader as authority on format.

Also emitted per pack: `vertex.stride` + `attrs` and `instance.stride` + `fields`, so a reimplementer reads the binary layout from the manifest rather than hardcoding it.

---

## 18. GPU-side consumption

### Material flags

`GpuMaterial` (defined `viewer/src/render/gpu_driven.rs:225-290`) is **192 bytes** with Rust `align_of == 4`. Both are statically asserted at `:295-296`:

```rust
const _: () = assert!(std::mem::size_of::<GpuMaterial>() == 192);
const _: () = assert!(std::mem::align_of::<GpuMaterial>() == 4);
```

Do **not** pin `align_of == 16`. The "16-aligned" phrasing in the neighbouring comment describes the **WGSL vec4 lanes** of the matching `MaterialGpu`, not the Rust struct. The record is indexed directly by `materialId`.

| bit | constant | set when |
|---|---|---|
| 0 | `MAT_FLAG_CUTOUT` | `role == cutout` or `alphaMode == MASK` |
| 1 | `MAT_FLAG_BLEND` | `role ∈ {decal, glass}` or (`alphaMode == BLEND` and `role != water`) |
| 2 | `MAT_FLAG_SOFTCUTOUT` | `vp.softCutout` present **and** `role == decal` (implies BLEND, clears CUTOUT) |
| 3 | `MAT_FLAG_WATER` | `role == water` |
| 4 | `MAT_FLAG_TERRAIN` | tagged after the material loop; clears DETAIL and RFA |
| 5 | `MAT_FLAG_DETAIL` | `detail` block resolved at least one texture |
| 6 | `MAT_FLAG_RFA` | `roughnessFromAlbedoAlpha` and `role ∈ {opaque, glass}` and not glass-mask |
| 7 | `MAT_FLAG_VP` | all 3 vp layer albedos resolved |
| 8 | `MAT_FLAG_PUDDLE_LUMA` | textured water whose alpha is near-constant → mask is in luma |
| 9 | `MAT_FLAG_WATER_MATTE` | textured water with > 40 world-m per UV repeat |
| 10 | `MAT_FLAG_DECAL` | `role == decal` |
| 11 | `MAT_FLAG_PARALLAX` | parallax map resolved |
| 12 | `MAT_FLAG_GLASS_MASK` | non-TRS glass whose alpha probes as coverage |
| 13 | `MAT_FLAG_GLASS_TRS` | `glassTRS` and `role == glass` |
| 14 | `MAT_FLAG_GLASS_CUBE` | `reflectCube` present |

Water routing (`:2172-2192`): **textured** water is a thin puddle film → BLEND pass. **Untextured** water is deep water → opaque pass with depth write, so glass composites over it and it cannot z-fight the unsorted blend pass. The matte discriminator is world-metres-per-texture-repeat, measured per submesh as `‖pmax - pmin‖ / max(u_span, v_span)`, threshold `WATER_MATTE_MPR = 40.0` (`:1910`, `:1953-1957`). **[unverified]** the source comment reports measured puddles at ≤ ~22 and floor/tire decals at ≥ ~60.

### Formats

| content | format | condition |
|---|---|---|
| albedo, emissive, detail albedo | `Bc3RgbaUnormSrgb` | BC feature present, `w >= 64 && h >= 64` |
| albedo fallback | `Rgba8UnormSrgb` | otherwise |
| normal, detail normal | `Bc5RgUnorm` (LINEAR) | same size gate |
| normal fallback | `Rgba8Unorm` | otherwise |
| terrain control maps, vp heights, parallax height | `Rgba8Unorm`, **never BC** | always (membership in `ctrl_tex_linear`) |

`bc_wanted` at `viewer/src/render/gpu_driven.rs:6345-6353`; `EFT_TEX_BC=0` disables. Control/data maps are exempt because BC's palette interpolation warps exact blend weights into visible splat banding (`:6117-6119`).

Mip chain (`build_mip_chain`, `:6150-6168`): `mips = 32 - leading_zeros(max(w,h))` = `floor(log2(max)) + 1`; level *l* is `((w>>l).max(1), (h>>l).max(1))`, Triangle-resampled from level *l-1*; levels are concatenated with no padding between them.

BC3 encode: `texpresso` `RangeFit`, per-mip `compressed_size` = `ceil(w/4)*ceil(h/4)*16` (`:6174-6192`).

BC5 encode (`:6238-6263`): per 4×4 block, `BC4(R)` then `BC4(G)` = 16 bytes, same block stride as BC3. `bc4_block` (`:6197-6232`): `r0 = max`, `r1 = min` (the `r0 >= r1` 8-value mode); reference values `k ∈ 2..7` are `((8-k)*r0 + (k-1)*r1 + 3) / 7`; each texel takes the nearest 3-bit index, packed LSB-first into a 48-bit field stored in bytes 2..7. Edge blocks clamp their sample coordinates to `mw-1` / `mh-1`.

Rationale for BC5 on normals: BC3's BC1-quality RGB565 colour block crushes small X/Y relief flat. BC5 gives each channel its own interpolated endpoints at 8 bpp.

### The runtime BC cache

`<shared_dir>/texcache/<fnv64:016x>.<ext>`, `ext ∈ {bc3c, bc5c}` - built by `texcache_path` (`viewer/src/render/gpu_driven.rs:6269-6272`) as `crate::paths::shared_dir().join("texcache").join(format!("{hash:016x}.{ext}"))`; read/write at `:6264-6332`. `shared_dir() = packs_root().join("shared")` (`viewer/src/paths.rs:71-73`); **`packs_root()` resolves dynamically** (exe dir or app-data) - call it, do not hardcode a `packs/` prefix.

File = `[w:u32 LE][h:u32 LE][mips:u32 LE]` then the concatenated payload. Hash is FNV-1a 64 over the **source PNG bytes**: offset basis `0xCBF29CE484222325`, prime `0x100000001B3`. A read is rejected (treated as a miss) when `w == 0 || h == 0 || m == 0 || w > 16384 || h > 16384 || m > 16` or the payload length differs from the recomputed sum - feeding a short buffer to `create_texture_with_data` aborts the process. Writes are temp + rename. Content addressing self-invalidates and lets the same game texture, extracted into several datasets under different filenames, encode once.

### Texture quality tiers (`TEX_MIP_SKIP`)

`viewer/src/render/gpu_driven.rs:6518-6528`. A `u8`, clamped by `set_tex_mip_skip(n.min(2))`: **0 = full, 1 = half, 2 = quarter**. It is captured once per map build; changing it live applies on the next map load.

`slice_mips(tex, skip)` (`:6540-6610`) is pure byte slicing - no re-encode, because both the cache and fresh encodes store full chains, so "half resolution" is literally "start the upload at level 1". Effective skip:

```
e = min(skip, mips - 1)
while e > 0:
    nw, nh = (w>>e).max(1), (h>>e).max(1)
    if max(nw, nh) < 128:              e -= 1     # small textures are cheap; don't crush them
    elif block4 and (nw%4 or nh%4):    e -= 1     # wgpu validates BC base-level dims against the block
    else: break
```

Byte offset of level `e`: BC → `Σ_{l<e} ceil((w>>l)/4) * ceil((h>>l)/4) * 16`; raw → `Σ_{l<e} (w>>l)*(h>>l)*4`. Result carries `w>>e`, `h>>e`, `mips - e`.

`no_downscale` returns the texture before `slice_mips` runs (`prepare_tex_cpu`, `:6632-6633`), and **only terrain control maps set it** - `:2436` is the sole `no_downscale.insert` in the file besides the declaration, because for splat weights the resolution *is* the data. The other two linear/data categories are **not** exempt: vp `heights` masks (`:2136`) and the parallax height map (`:2288`) go into `ctrl_tex_linear` **only**, so they upload raw and linear but are still sliced and downscaled at TEX_MIP_SKIP 1/2.

The Standard path mirrors the slicing with `resize_exact` by `2^e` and the identical 128 px floor (`viewer/src/render/standard.rs:252-271`). Without the mirror the setting silently does nothing on that path and VRAM is identical at every quality tier.

### Index-stability invariant

Every texture-load failure returns a **1×1 placeholder** rather than skipping the slot - magenta `(255,0,255,255)` for albedo, flat `(128,128,255,255)` for normals. Off-thread: `viewer/src/render/gpu_driven.rs:4427` (magenta) and `:4437` (flat normal). Sync mirrors: `:6060` (magenta) and `:6087` (flat normal). A skipped slot shifts every later bindless index and textures the entire map wrong with no error.

---

## 19. Captured vs DROPPED

**Captured and shipped:** albedo, normal (+ `_BumpScale`), `_MainTex_ST`, `_Color` (RGB **and** A), `_Cutoff`, RenderType/renderQueue role, `_Glossiness`, `_Metallic`, `_EmissionMap` + `_EmissionColor` (keyword-gated), `_SpecMap`, the smoothness-in-alpha flag, the 3-layer vert-paint block + `_Heights` + `_BlendStrength` + the SoftCutout triple, `_DetailAlbedoMap`/`_DetailNormalMap` (+ ST or UV-scale floats, + intensities), `_ParallaxMap` + `_Parallax`, and the legacy glass response set (`_ReflectColor`, `_SpecColor`, `_Shininess`, `_OpacityScale × _AlphaMult`, `_Cube` mean).

**Captured by the extractor, DROPPED by the assembler:**

- `_DetailMask` - written as `extra["detMask"]` (`extraction/unity/eft_extract_v2.py:1195-1197`) but `MaterialFactory._detail()` never emits it (`eft_pipeline/assemble_bevy.py:394-410`). Detail is applied unmasked.
- `emis`/`emisCol` on `role ∈ {decal, glass}` and on any vp material (`eft_pipeline/assemble_bevy.py:321`).
- `gloss` on shader families with no specular term (`:300-303`).

**Never captured at all:**

- **Occlusion / AO maps** - no `_OcclusionMap` slot appears in any slot list, and `_OcclusionStrength` is never read. Ambient occlusion is entirely screen-space at runtime.
- **Metallic-gloss / spec-gloss *maps*** - only the scalar floats `_Metallic` and `_Glossiness` are read. `_MetallicGlossMap` and `_SpecGlossMap` are not in any slot list. Per-pixel roughness comes from albedo alpha (`roughnessFromAlbedoAlpha`) or `_SpecMap` luma.
- **Texture sampler state** - `m_WrapMode`, `m_FilterMode`, `m_Aniso` are never read; the runtime hardcodes Repeat + trilinear + anisotropy.
- **Secondary UV sets / lightmap UVs** - the vertex format carries exactly one UV (`eft_pipeline/assemble_bevy.py:88-95`). **[unverified]** whether EFT ships baked lightmaps or light probes at all; no code in this repository settles that, so treat "nothing is lost" as an assumption, not a fact.
- **Tangents** - not exported; generated at import from UV + normal (`viewer/src/render/standard.rs:443`) or derived from a screen-space cotangent frame in the shader.
- **Vertex colour on non-vp submeshes** - force-set to `(255,255,255,255)` (`eft_pipeline/assemble_bevy.py:1021`).
- **Render-queue ordering** beyond the glass-vs-decal split at 2900.

---

## 20. Invariants and failure signatures

| Invariant | Break it and you see |
|---|---|
| `uvVFlipBaked` honoured - the consumer must **not** apply `1 - v` again | every texture upside-down; text and signage mirrored vertically |
| `uvTilingBaked` honoured - `uvXform` must **not** be multiplied into UVs | double-tiling: brick/asphalt at the square of the intended repeat |
| Tiling applied **before** the V-flip at bake time | textures offset by `oy` along V; tiled atlases sample the wrong cell |
| Sampler address mode = Repeat in U and V | the last texel row smeared into long streaks across large surfaces |
| Green flip applied exactly once (import **or** shader, never both) | lighting on bumps inverted - dents read as bumps; the SH bake fights the sun |
| Normal maps uploaded LINEAR, albedo/emissive sRGB | washed-out, over-flat normal perturbation; emissive too dark |
| Only `.xy` read from a normal map, Z reconstructed | BC5 gives `z = 0`; normals collapse to the tangent plane and shading goes black at grazing angles |
| `san()` left Unicode-aware (`str.isalnum`, not `[A-Za-z0-9]`) | Cyrillic-named textures get different filenames; every material path misses an existing `tex/` directory |
| Albedo alpha preserved through export (never `.convert("RGB")`) | foliage/fence cutouts render as solid rectangles; RFA roughness pins to 0.06 mirror-smooth |
| Albedo compressed to BC3, not BC1 | alpha quantized to 1 bit - cutout silhouettes turn jagged, blend surfaces turn binary |
| `f0 += n` executes for **every** skipped submesh | later submeshes read shifted triangle ranges: holes in the world plus a neighbouring material drawing another's faces |
| Failed texture loads return a placeholder, never a skipped slot | every bindless index after the failure shifts; the whole map is textured wrong, silently |
| Terrain/vp/parallax control maps stay LINEAR and uncompressed | splat banding; gamma-warped blend weights; height-map stepping |
| `no_downscale` set on terrain control maps only | set it too widely and the mip-skip quality tiers stop reclaiming VRAM; omit it and splat weights blur across their boundaries |
| `MAT_FLAG_CUTOUT` cleared on non-decal vp materials | tex.a is smoothness; the alpha test discards nearly the whole slab and ground renders as see-through rectangles |
| SoftCutout coverage keeps its trailing `* COLOR_0.a` | feather tails re-saturate; road and tire-track decals get hard, stamped edges |
| SoftCutout blend forced only for `role == decal` | the opaque "Solid" variant clamps coverage to 0 and whole courtyards vanish |
| `_Cutoff` (authored) wins over the Otsu split | a smooth mask promoted to cutout at a low Otsu split then cut at 0.5 discards nearly the whole surface |
| `emissive` parsed as object-or-null | serde aborts the whole `materials.json` parse; the pack fails to load entirely |
| `GpuMaterial` pinned by static assert (size 192 B, `align_of` **4**) | a silent CPU/WGSL layout mismatch corrupts every material record - the same class of bug as a mis-strided shader table read |
| Texture files treated as immutable (write-temp-then-replace) | hardlinked packs mutate each other's textures in place |
| `_png_complete` (IEND tail) used as the reuse guard, not `exists()` | NTFS-preallocation zero-filled PNGs survive every rebuild and render as the magenta placeholder |
