# Atlas — Rendering Fidelity: SOURCE OF TRUTH

Working doc for getting the native viewer to EFT-quality rendering. Every issue below
has a known root cause + a solution recipe distilled from the **web-viewer battles**
(flir-app project memory, `tarkmap-*.md`). Work top-to-bottom; update STATUS as we go.

## DECISION (2026-07-15): build on the CUSTOM `gpu_driven` path

EFT renders terrain/decals/roads/grass with **custom Unity shaders** (MicroSplat,
`Custom/Vert Paint SoftCutout Decal`, GPU-Instancer) that plain Bevy `StandardMaterial`
cannot reproduce. The custom path already renders decals (SoftCutout, 133 mats), SH-GI
in-game lighting, and materials correctly — and it's fast (68k instances, one indirect
multidraw). The `render/standard.rs` migration (EFT_RENDER=std) got dynamic shadows but
**broke terrain/decals/roads and never had grass**, and every per-frame lighting effect
tanked its 102k-entity scene:

| tried on standard path | FPS | verdict |
|---|---|---|
| shadows + textures | 91 | good, but terrain/decals/roads broken |
| Solari RTX | 15 | noisy, no transparent lighting — dropped |
| SSR (deferred) | 28 | no visible reflections, can't hit blend surfaces — dropped |
| 1285 in-game lights (clustered fwd) | 0.4 | froxel overflow — dropped |
| Atmosphere env-map | 27 | cyan wash, clear-sky ≠ EFT overcast — dropped |

→ **Target = custom path.** Keep `standard.rs` only as an A/B / shadow reference.

---

## ISSUES

### 1. Terrain + asphalt: low-res + repeating noise  —  STATUS: ✅ DONE on custom path (2026-07-15)
Implemented: 12 layer albedos + 12 control maps appended to the bindless set; `TerrainSplatGpu`
SSBO at group(2) binding(4); the 4 terrain materials tagged `MAT_FLAG_TERRAIN` + slice in `_pad2`
+ matte roughness; `gpu_draw.wgsl` splat branch blends 12 layers by control weights at
`uv*rep` via `textureSampleGrad`. 521 FPS. (Follow-ups if needed: per-layer normal maps;
load control maps LINEAR not sRGB; detail maps #6.)

- **Symptom:** terrain/asphalt look low-res with obvious tiling/repeating noise.
- **Root cause:** rendered as a single tiled albedo. Unity uses **MicroSplat**: N layers
  (grass/ground/gravel/forest/asphalt…) blended through control maps, each at its own UV
  tiling. We do none of that.
- **Data (already extracted):** `eft_assets/interchange_v2/terrain_layers/manifest.json`
  — per tile: `ctrl_maps` (`ctrl_Slice_*_{0,1,2}.png`, RGBA = up to 12 layer weights),
  `layers[]` each `{idx, name, ctrl, chan, cov, tileX, rep}`. **`tileX` = the real
  per-layer UV tiling** (the value that is NOT in `m_TileSize`, whose `y=inf` is garbage —
  recovered by scanning the shared bundles). e.g. `Grass_summer_D tileX=137.3`,
  `Gravel_Road_A_summer_D tileX=179.5`.
- **Solution:** MicroSplat splat shader for `FLAG_TERRAIN` tiles: sample the 3 control maps →
  12 weights; sample each layer albedo (+normal) at its tiling; weight-blend + normalize.
- **CONFIRMED DATA (2026-07-15), all present under `terrain_layers/`:**
  - 4 tiles: `Slice_1_1/1_2/2_1/2_2`, each 700×700 m, each with 3 control maps
    (`ctrl_<slice>_{0,1,2}.png`, RGBA = 12 layer weights).
  - 12 layers, each `{idx, name, ctrl(0..2), chan(0..3=RGBA), tileX, rep}`; all 12 albedos
    `layer_<name>.png` + most normals `layer_<name>_N.png` on disk.
  - **TILING (verified via `microsplat_uv_scales`):** MicroSplat = `tiledUV = terrainUV01 ×
    _UVScale × perTexScale[i]`; manifest `rep = _UVScale·perTexScale` ⇒ **`layer_uv =
    terrainUV01 × rep`** (Repeat wrap). NEVER use `tileX`/`m_TileSize` (garbage `y=inf`).
    Grass rep≈397 → ~1.8 m repeat ✓; tileX=137 → 137 m repeat = the "massive grass" bug.
- **IMPLEMENTATION PLAN (custom path):**
  1. Pipeline (assemble_bevy): for each `FLAG_TERRAIN` tile, tag which Slice it is + emit a
     terrain-material record (3 ctrl-map paths, 12× {albedo path, ctrl, chan, rep}). Need to
     confirm the terrain mesh's UV convention (raw uv01 vs baked ×uvscale).
  2. Rust (gpu_driven): load the 3 ctrl maps (per slice) + 12 layer albedos into the bindless
     set; add a terrain uniform (rep[12], ctrl/chan indices) + a terrain flag path.
  3. Shader (gpu_draw.wgsl): terrain branch — sample 3 ctrl maps → w[12]; `acc = Σ w[i]·
     texture(layer[i], uv·rep[i])`; normalize by Σw. Matte roughness.
- **Refs:** `tarkmap-terrain-microsplat-tiling`, `tarkmap-road-terrain-matte-and-hole-bake`

### 2. Decals render as "big sheets"  —  STATUS: ✅ VERIFIED fixed on custom path (2026-07-15)
- **Symptom:** tire tracks / drips / posters render as full semi-transparent quads.
- **Root cause:** decal coverage is either **vertex paint `COLOR_0.a`** (SoftCutout family)
  or **albedo alpha** (`decal_surface`); the standard path ignores both and just blends the
  whole quad.
- **Solution:** SoftCutout feather — `alpha = clamp(COLOR_0.a·_AlphaStrength − (_Cutoff −
  _AlphaHeight), 0, 1)`, with `vp.a = [_AlphaStrength, _Cutoff, _AlphaHeight]`.
  `decal_surface`: `alpha = albedo alpha` (coverage), EXCEPT SoftCutout (there albedo alpha
  is smoothness). Matte roughness.
- **Status:** custom path already implements this (log: "133 SoftCutout + M3b2"). → fixed by
  targeting the custom path. VERIFY on it.
- **2026-07-20 follow-up:** plain `role=decal` atlases still shared the glass response in the
  premultiplied blend pass. Diffuse respected `albedo.a`, but uncovered specular/environment
  terms remained nonzero at alpha 0, painting transparent tire-track/pothole texels as pale
  rectangles. `MAT_FLAG_DECAL` now coverage-masks every lighting term; glass keeps the original
  full-strength reflection behavior.
- **2026-07-20 follow-up 2:** water-decal mask detection was opening the pack-relative texture
  path from the process CWD, so constant-alpha puddle atlases never switched to their RGB mask and
  rendered full quads. Resolve before inspecting; smoothly suppress the BC/mip near-zero tail.
- **Refs:** `tarkmap-decal-alpha-coverage`, `tarkmap-vp-shader-variants`

### 3. Road splines torn / floating tire-track sheets  —  STATUS: ✅ VERIFIED fixed on custom path (2026-07-15)
- **Symptom:** `RoadSplineGenerator_GenerateLods` decals render as torn, floating sheets.
- **Root cause:** same SoftCutout coverage not applied (standard path).
- **Solution:** same as #2. The road is a **decal that FEATHERS into the terrain** via
  `COLOR_0.a`, NOT a terrain-hole cut. **Matte the ground** (road roughness floor ~0.72, not
  a near-mirror) or the road pops off as a separate slab.
- **REJECTED — do NOT re-do:** cutting the paved footprint out of the terrain geometry
  (`terrain_cut.py` / `bake_roads_into_terrain`). User rejected it; terrain must stay whole.
- **Refs:** `tarkmap-road-terrain-matte-and-hole-bake`
- **2026-07-20 z-fight follow-up:** two overlapping SoftCutout road meshes both wrote color+depth
  with the same weak bias. Split them into an A2C coverage-only depth prepass and a premultiplied,
  non-depth-writing color pass with a stronger bias. Plain surface overlays get their own biased
  pass; glass stays unbiased.

### 4. No grass  —  STATUS: ✅ DONE on custom path (2026-07-15)
Implemented: `build_grass.py` reads the 4 density grids + terrain meshes → `grass.bin`
(N×[x,y,z,rotY,scale], deterministic per-cell hash, placed on the surface via a UV→world
bilinear lookup, road-excluding). `gpu_driven` appends a cross-quad grass mesh + cutout
material + the ~109k instances to the same cull+multidraw buffers. 400 FPS.

- **Symptom:** terrain has no grass/foliage.
- **Root cause:** EFT grass = **GPU-Instancer baked density grids**, NOT Unity terrain
  detail (that DB is a zeroed decoy). 88 `GPUInstancerDetailPrototype` MonoBehaviours
  (4 slices × 22) in `sharedassets63.assets`.
- **Data:** each prototype MB (~1,049,268 B) → **last 1,048,576 B = 1024×1024 uint8 density
  grid**, row-major, `density[y*1024+x]`. Sparse (~6.4% nonzero), **road/building-excluding,
  hand-authored**. Grass Texture2D PPtr + min/max scale in the ~692 B header. (The pack's
  `grass.json` currently only has per-slice tint/amount/strength — the density grids still
  need extracting.)
- **CONFIRMED (2026-07-15): density grids ALREADY extracted** — `terrain_layers/
  grass_density_Slice_{1_1,1_2,2_1,2_2}.bin`, each 1,048,576 B = 1024² uint8. Grass albedos
  present (`grass_Grass3_D.png`, `grass_Grass5_512_D.png` + `_N`). → render-only task now.
- **Solution:** place instances from the density grids; per nonzero cell scatter grass (fixed
  per-cell hash) at the cell's world XZ via terrain UV space; render GPU-instanced billboards/
  cross-quads; per-slice tint. **DETERMINISTIC** — never client-scatter from splat weight
  (competitive-shooter visibility surface; also fixes grass-through-concrete for free).
- **Refs:** `eft-grass-gpuinstancer`, `competitive-grass-deterministic`

### 5. Dynamic sun shadows on the custom path  —  STATUS: TODO
- The custom path has no shadows (the one thing the standard path did well; user liked them).
- **Solution:** sun shadow-map pass — render scene depth from `sun_dir` (in `volume.json`,
  X-flip to pack space) into a shadow atlas; sample with PCF in `gpu_draw.wgsl`. Cascades for
  the ~1.2 km map.

### 6. Close-up surface detail (detail maps)  —  STATUS: TODO (optional, after 1–5)
- Web viewer adds name-keyed detail KTX2 (×2 Unity Standard blend ×4.5948, mean-neutralized,
  distance fade 8–15 m) for close-up local contrast (+52% rock, +11% floor).
- **Ref:** `tarkmap-spec-detail-maps`

---

## WORK ORDER
1. Verify decals + road splines feather correctly on the custom path (#2, #3) — should already work.
2. **Terrain MicroSplat splat shader (#1)** — the #1 visual complaint.
3. **Grass extraction + GPU-instanced render (#4).**
4. **Sun shadow-map pass (#5).**
5. Detail maps (#6) — polish.

## GLOBAL RULES / DON'T-RE-BREAK
- Placement (grass, loot, lights) comes from **Unity game data**, never invented/scattered.
- Game texture **alpha is DATA** (smoothness), not opacity — coverage only via the decal/
  SoftCutout rules above. (`CHANNEL_PACKED` in Blender; don't associate alpha into color.)
- `_EmissionMap` with no serialized `_EmissionColor` = emission OFF (Unity default black).
- Never re-enable the terrain-hole cut. Never put per-shader gloss on ground without matting.
