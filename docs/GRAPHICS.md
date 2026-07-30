# Atlas Graphics Audit

Audience: this project's developers. Everything below was verified against source on 2026-07-30;
where a code comment and the code disagreed, the code wins and the discrepancy is flagged.
Measured numbers come from `docs/GFX_BENCH_*.json` (bench_gfx.py runs) or from measurement notes
embedded at the cited line — nothing is estimated unless marked as such.

Baseline for all costs: woods `--alllod` pack, Ultra preset, 2560x1440, avg **11.85 ms**
(`docs/GFX_BENCH_woods_alllod_2560x1440.json` row `u_base`).

Render-path context: three paths exist (`viewer/src/render/mod.rs:536-545`) — M0 instanced
fallback (`instancing.rs:1-22`, flat lambert, no cull), Bevy Standard PBR fallback
(`standard.rs:1-13`, one entity per instance×submesh, used on LLPC-driver AMD ICDs,
`mod.rs:618-652`), and the default M2 GPU-driven path (`gpu_driven.rs`) that everything below
describes. Backends are restricted to Vulkan on Windows/Linux (`mod.rs:585-594`).

---

## 1. Techniques in use

### Frame anatomy (order of operations)

1. Compute cull: `cs_reset` → `cs_cull` → `cs_sort_blend` (`gpu_cull.wgsl`), before the main pass
   (`gpu_driven.rs:6817-6821`).
2. Normal prepass (only when SSAO is on — sole consumer today, `gpu_driven.rs:6951-6953`).
3. Shadow pass: up to 4 cascades, each skippable via the fit cache (`gpu_driven.rs:7109+`).
4. Main pass: one `Transparent3d` phase carrying five specialized pipelines in queue order —
   Opaque (enqueued at large negative distance so it runs first), DecalDepth, DecalColor,
   Overlay, Blend (`gpu_driven.rs:6607-6618`, `gpu_draw.wgsl:29-39`).
5. Post: SSAO → Bloom → auto-exposure reduction + grade (FXAA/sharpen/LUT/vignette) →
   Tonemapping slot (None when grade active) → DOF/chroma (Bevy components) → FPV cam pass →
   upscale/sRGB encode. Graph edges: `ssao.rs:286`, `grade.rs:516-517`, `fpv_cam.rs:11-12`.

### GPU-driven culling + multidraw

- One storage-buffer scene: 80 B `InstanceGpuRecord` (full row-major 3x4 affine + ids +
  precomputed conservative world sphere, `gpu_driven.rs:94-105`), 32 B `MeshMeta`
  (`gpu_driven.rs:110-123`), interleaved 32 B vertex (pos + octahedral Snorm16x2 normal + uv +
  u32 material id + Unorm8x4 COLOR_0, `gpu_driven.rs:6739-6774`). Normals are oct-encoded to save
  8 B/vertex (~457 MiB on streets, `gpu_draw.wgsl:908-910`).
- `cs_reset` regenerates every `DrawIndexedIndirectArgs` from static MeshMeta each frame (no
  stale-data hazard), splitting index counts between an opaque and a blend indirect buffer by
  material class (`gpu_cull.wgsl:92-113`).
- `cs_cull` (one thread/instance): 6-plane frustum test on the CPU-precomputed world sphere
  (Frobenius-norm conservative radius — never max-column-norm, `gpu_cull.wgsl:22-26`,
  `gpu_driven.rs:1221`), screen-size cull in pixels (defaults 1.5 px general / 4.0 px grass,
  `mod.rs:166-172`), optional metric grass distance clamp (`gpu_cull.wgsl:157-176`), LOD window
  test, then compaction: `slot = atomicAdd(instance_count)` into a per-mesh contiguous
  `visible[]` region — no prefix sum needed because instances are stored grouped-by-mesh
  (`gpu_cull.wgsl:17-21, 235-249`). All indices clamped (B5) because OOB reads return garbage on
  AMD (`gpu_cull.wgsl:235-248`).
- Dispatches are 2-D (`dispatch_2d`, `gpu_driven.rs:6912`): woods ships 11,572,828 instances
  (883 MiB), past the 65,535×64 1-D limit (`gpu_cull.wgsl:75-90`).
- `cs_sort_blend`: per-blend-mesh GPU insertion sort of the visible run, farthest first, so
  overlapping same-mesh glass panes composite deterministically (fixed a still-camera flicker on
  interchange; ~6,235 blend instances / 1,377 meshes, `gpu_cull.wgsl:252-308`).
- Draw: `multi_draw_indexed_indirect` per pass; the vertex shader fetches
  `instances[visible[instance_index]]` and applies the raw 3x4 affine + cofactor normal
  transform, double-sided, never TRS-decomposed (`gpu_draw.wgsl:1-17, 922-989`).
- Measured: foliage is the dominant draw cost (grass off = 11.85→7.19 ms alllod;
  `GFX_BENCH_woods_alllod` `u_no_grass`).

### Distance LOD with fade-band stagger

- Multi-shell packs (`--alllod`) upload every LOD shell; the GPU picks per frame. Per instance:
  `ids.z` bit8 = default shell, bits9-12 = lod_index, bits13+ = group id; `ids.w` =
  pack2x16float(near', far') distance window (0 = always draw) (`gpu_driven.rs:1680-1752`,
  `gpu_cull.wgsl:32, 180-233`).
- Windows are derived from the game's own LODGroup data: far = size/(2·srh)/proj11; near = far of
  the previous *present* shell so internal gaps leave no hole; billboard-tail groups cull past
  their last threshold (`gpu_driven.rs:1716-1743`).
- Shell switch distances are staggered per GROUP by a stable hash scaled by the game's own fade
  band w = clamp(max(ftw/srh), 0, 0.40) carried in `lod_centers[gid].w`
  (`gpu_driven.rs:3186-3218`, `gpu_cull.wgsl:196-227`). The hash keys on the group id, not the
  instance index — the instance-keyed version left an undrawn band between shells that blinked
  during zooms (bug fix documented at `gpu_cull.wgsl:211-226`). This is a stagger, NOT Unity's
  dithered cross-fade (deliberately: a true fade needs a per-instance fade weight buffer,
  `gpu_cull.wgsl:203-208`).
- Distance from the group's shared reference center (`lod_centers`), not each shell's centroid,
  so a group switches as a unit (`gpu_cull.wgsl:186-192`).
- Default ON (`EFT_LOD=0` opts out). The `mod.rs` comment used to claim "default off" — corrected
  in place (`mod.rs:151-157`). Measured worth **3.83 ms**: LOD off = 15.68 ms vs 11.85 ms
  (`GFX_BENCH_woods_alllod` `u_lod_off`).

### Bindless materials

- 192 B `GpuMaterial` SSBO (asserted, `gpu_driven.rs:220-286`) + `binding_array` bindless albedo
  and normal arrays indexed non-uniformly (`gpu_draw.wgsl:168-173`). The 192-byte stride is
  pinned in ALL FOUR shaders that bind the table (`gpu_draw`, `gpu_shadow.wgsl:48-74`,
  `gpu_prepass.wgsl:50-70`, plus the Rust POD) — a 176-vs-192 stride mismatch mis-decoded
  `albedo_index` into an OOB bindless descriptor and device-lost two Radeons
  (`gpu_shadow.wgsl:59-74`). Descriptor indices are clamped against an uploaded array length
  everywhere (WGSL has no `arrayLength` for binding_array, `gpu_shadow.wgsl:148-155`).
- 12 material flag bits: cutout, blend, softcutout, water, terrain, detail,
  roughness-from-alpha, vert-paint, puddle-luma, water-matte, decal, parallax
  (`gpu_draw.wgsl:149-163`, `gpu_driven.rs:295-346`).
- Per-vertex u32 material id (materials vary per submesh; wgpu 0.17.3 has no draw_id; submesh
  vertex sets verified disjoint, `gpu_draw.wgsl:19-26`).
- Textures: mips built at load (`gpu_driven.rs:5241`), BC3/BC5-compressed on CPU with an on-disk
  cache (`gpu_driven.rs:5265-5427`), texture-quality tiers drop mips at upload
  (`TEX_MIP_SKIP`, `gpu_driven.rs:5616-5629`). Measured: VRAM is entirely a texture story
  (Full = +2177 MiB vs Half; everything else <20 MiB, `mod.rs:271-285`).

### MSAA + alpha-to-coverage, FXAA

- Main pass runs at the view's MSAA count (Bevy default 4x; `count: key.samples`,
  `gpu_driven.rs:6792-6798`). A2C enabled only on Opaque and DecalDepth (`gpu_driven.rs:6795-6798`);
  cutouts output an fwidth-remapped coverage ramp with the hard discard at half the cutoff
  (`gpu_draw.wgsl:1110-1114, 1316-1318, 1687-1693`).
- FXAA (console variant) runs first in the grade pass on perceptual (sqrt) luma, ±2-texel search
  clamp, flat-area early-out returns the untouched center sample (`grade.wgsl:80-144`). It exists
  for what MSAA can't fix: shading aliasing and A2C's 4-level coverage quantization
  (`mod.rs:104-121`). Cost +0.043 ms — noise floor. Default on, strength 0.75.
- STALE COMMENT WARNING: `mod.rs:377-378` still claims "Nothing else in this renderer
  anti-aliases (every pipeline is sample_count 1)" inside the preset-placement note. That is the
  old, corrected-elsewhere mistake (`sample_count: 1` hits were the shadow atlas); the pipelines
  are MSAA. Trust `mod.rs:104-121` and `gpu_driven.rs:6792`, not that line.

### Sun shadows: 4-cascade CSM with a quantized-fit cache

- 4 cascades (was 2; the "2-cascade" header comment at `gpu_driven.rs:660-661` is stale — the
  constant is 4 at `gpu_driven.rs:722`), splits [0.5, 15, 80, 250, 700] m
  (`gpu_driven.rs:726`), 3072² Depth32Float array (72 MiB, `gpu_driven.rs:670-681`,
  `EFT_SHADOW_SIZE` overrides). 352 B `SunShadowUniform`, size-asserted against the WGSL twin
  (`gpu_driven.rs:774-801`; a silent mismatch here caused the RX 6800 device loss).
- Fit (`prepare_shadow_uniforms`, `gpu_driven.rs:6165-6435`): rotation-invariant, camera-centred
  (panning is bit-identical → free; slice-centroid fit cost 6.16 ms while panning vs 0.57 ms at
  rest, `gpu_driven.rs:6274-6290`). Everything feeding `view_proj` is quantized: radius rounded
  up to 1 m steps, centre snapped to 16-texel blocks on ALL THREE light-space axes, Z range
  derived from radius+constants, not corners (`gpu_driven.rs:6298-6347`). The #5b cache then
  skips a cascade's render pass when its `view_proj` is bit-identical
  (`gpu_driven.rs:6360-6412`); dirtied by door animation, buffer rebuilds, or any GfxSettings
  change (`gpu_driven.rs:6183-6188`).
- Caster pass (`gpu_shadow.wgsl`) replays the camera-culled `visible[]`/indirect stream (no
  re-cull) with a minimal fragment: BLEND materials discard, CUTOUT alpha-tests, grass casters
  degenerate-skipped by default (`gpu_shadow.wgsl:104-112`; grass casters measured **+7.4 ms**,
  `GFX_BENCH_woods_fly` `u_grass_shadows`).
- Receiver (`gpu_draw.wgsl:767-840`): 3x3 [1,2,1]² tent PCF, receiver-plane offset on the
  geometric normal (1.5 texels + 0.25 toward sun), cascade select generalized over the count
  with a 10% overlap blend, far fade 600→700 m.
- Anti-double-darkening combine (`gpu_draw.wgsl:1339-1383`): the baked SH already contains the
  broad sun shadow, so the realtime term is gated by SH directionality × dom·Lsun alignment ×
  N·Lsun and may remove at most 12% (`SHADOW_DIFFUSE_CAP`) of above-floor diffuse; the GGX lobe
  takes the full shadow (it is not baked). Measured cost of shadows: ~2.0 ms
  (`GFX_BENCH_woods_alllod` `u_no_shadows`).

### Normal prepass (NEW) — `gpu_prepass.wgsl`

- Re-draws the opaque scene through the SAME culled `visible[]`/indirect buffers (nothing is
  re-culled) into Rgba16Float `vec4(world_normal, roughness)` + its own 1x Depth32Float
  (`gpu_prepass.wgsl:1-23`, targets at `gpu_driven.rs:6965-6998`). Geometric normal on purpose
  (not normal-mapped — SSAO over mapped normals turns detail into noise;
  `gpu_prepass.wgsl:9-13`). BLEND surfaces discard (their normals belong to what's behind them);
  cutouts alpha-test; grass excluded twice (node skips the mesh range + vertex degenerate guard,
  `gpu_prepass.wgsl:20-23, 114-121`).
- Currently gated on SSAO being enabled ("sole consumer today; don't pay ~1 ms for an unread
  buffer", `gpu_driven.rs:6951-6953`). Explicitly built as the SSR enabler
  (`gpu_prepass.wgsl:3-7`).

### SSAO (normal-aware)

- Fullscreen post between main pass and Bloom (`ssao.rs:286`): 10-tap golden-angle spiral,
  per-pixel hash rotation, range-checked horizon term, distance fade to 80 m
  (`ssao.wgsl:61, 97-115`; params radius 0.7 m / intensity 1.0 / power 1.5 set at
  `ssao.rs:207`, radius/intensity live UI sliders).
- Per-pixel normal source selection: the prepass world normal when it ran and wrote the pixel,
  else depth-derivative face normal (prepass clears to zero; one code path,
  `ssao.wgsl:74-89`, fallback 1x1 zero texture `ssao.rs:53-56, 117-132`). Reads sample 0 of the
  MSAA depth. Multiplies scene color (classic non-physical SSAO; darkens all light).
- Off by default / Ultra-only. Measured: ~0.2 ms at 1440p (`GFX_BENCH_woods_fly` `u_base` vs
  `u_no_ssao`; older interchange bench: −2..−3% fps, `mod.rs:276`).

### Baked SH irradiance volume (+ bounce, + validity)

- Build-time bake, reachable from `tools/build_map.py` stage 3 (`build_map.py:818-875`):
  default = portable Rust baker `atlas bake-sh <pack> --indirect-only` (`viewer/src/sh_bake.rs`),
  GPU (wgpu compute, vendor-neutral) with CPU rayon fallback; `EFT_BAKE=warp` selects the legacy
  CUDA baker (`extraction/bake/bake_volume2.py`).
- Pass A (`sh_bake.wgsl`): per probe, Fibonacci-sphere sky-visibility rays against the shared
  nav-bake BVH (Möller-Trumbore any-hit), projected into L1 radiance SH; plus shadow-tested
  practicals unless `--indirect-only` (`sh_bake.wgsl:126-188`). Giant maps stay on GPU by
  chunking tris/nodes across up to 3 storage bindings (`sh_bake.wgsl:1-11`), probe batches
  dispatched to dodge TDR (`chunk.z`, `sh_bake.wgsl:128`).
- Pass B (`sh_bounce.wgsl`): one diffuse bounce — nearest-hit recast, gather E(hit,n) from pass A
  (trilinear + cosine convolution), re-emit `albedo/π·E + emissive` where albedo/emissive are
  per-material means of the pack's own PNGs (`sh_bounce.wgsl:1-7, 175-205`).
- Grid: ~3 m XZ / 4 m Y spacing, ≤2.6 M probes (`sh_bake.rs:42-50`); output = three RGBA16Float
  3D textures (one per channel, texel = c0..c3 — SH interpolates linearly so HW trilinear is
  correct, `gpu_driven.rs:1005-1097`) + an R8Unorm per-probe VALIDITY volume (Unity APV
  "backface ratio" analog, `gpu_driven.rs:975-980, 1140-1160`).
- Runtime sampling (`gpu_draw.wgsl:343-438`): manual 8-tap trilinear with two independent probe
  rejections — validity and normal-hemisphere — plus a 0.75 m normal bias
  (`gpu_driven.rs:429`), a smooth handover to hardware trilinear for fully-enclosed points, and
  an out-of-volume redirect that slides samples to the volume's top (open-sky) layer scaled by a
  measured ground/top luma ratio (`gpu_draw.wgsl:283-296`, `gpu_driven.rs:1076-1137`). Multiple
  fixed leak/seam bugs are documented inline (window "cream rectangle" partition-of-unity fix at
  `gpu_draw.wgsl:402-411`).
- The SH also supplies the DOMINANT LIGHT (luminance-weighted L1 direction + radiance +
  directionality, `gpu_draw.wgsl:440-505`) that drives the GGX sun glint and the shadow gates —
  no separate sun light data needed.

### Realtime light grid

- The pack's extracted practical lights (`extraction/unity/eft_extract_lights.py`) in a static
  CPU-built CSR world grid (cell size grown to cap 4 M cells, `gpu_driven.rs:444-478, 493`);
  fragment loops only its cell's lights: smooth-windowed 1/d² falloff, spot cones, Lambert + the
  same dielectric GGX as the sun lobe (`gpu_draw.wgsl:507-588`).
- Auto-selected vs the baked volume so they never double-count: an indirect-only bake
  (`volume.json "direct": false`) enables realtime practicals; a full bake disables them
  (`sh_bake.rs:549-556`, `gpu_draw.wgsl:250-254`).
- Per-light SH directional occlusion: gate each light by SH radiance(toward light)/ambient — soft
  wall attenuation with no per-light shadow maps (`gpu_draw.wgsl:534-571`).
- Power groups: bitmask uniform, groups toggled by clickable switch meshes / UI, default all-off
  at spawn (`mod.rs:147-150, 242-248`). Measured: within noise (`GFX_BENCH_woods_fly`
  `u_no_lights`).

### Volumetric sun shafts

- 12-step ray march of the shadow cascades through a scattering medium, inside `apply_fog` in the
  forward fragment (deliberately not a froxel volume — everything needed is already bound,
  `gpu_draw.wgsl:636-765`). Henyey-Greenstein g=0.40, shaft medium density 0.00145/m decoupled
  from the fog slider, scatter gain 8.0 calibrated for a level view, 1-tap shadow samples,
  world-position jitter (no TAA to resolve a screen dither), march capped 25–300 m with optical
  depth over the full view distance.
- Strength rides `casc_params.w` — NOT `gfx.w`, which carries wind time (the Rust comment that
  called gfx.w "reserved" was wrong and has been corrected; `gpu_draw.wgsl:707-711`,
  `gpu_driven.rs:788-791`).
- Ultra-only, default off. Measured **+5.40 ms** at 1440p woods (~45% of the frame) — the most
  expensive option in the renderer (`mod.rs:377-382`; the VOL_MIN_DIST skip bought only ~0.5 ms,
  honestly recorded at `gpu_draw.wgsl:652-660`). Forced off when shadows are off
  (`gpu_driven.rs:6237-6242`).

### Water (Water4-derived branch)

Three material classes (`gpu_draw.wgsl:1471-1684`):
- DEEP WATER (untextured role=water; sea + basins) draws OPAQUE for correct depth under glass:
  energy-balanced fresnel between a dark teal body and a Reinhard-compressed sky mirror, over
  either (a) two procedural sine octaves (18 m / 5 m), per-octave Nyquist band-limited by the
  world-XZ derivative footprint, f32-safe `rsin` (fract-first — raw sin at ±1200 rad was the
  sea's checkerboard/"shadow streak" bug, `gpu_draw.wgsl:599-606`), or (b) the game's own
  `WaterBasicNormals` bump sampled exactly as the web `_water.js` port does when the map ships a
  real FX/SimpleWater4 material (`gpu_draw.wgsl:1605-1626`). Drift speeds are ported from
  `tarkmap/out/_water.js` (0.05 uv/s calibration; the first 2.7x-slower conversion read as
  static and was corrected, `gpu_draw.wgsl:1587-1598`). Water normal is based on world up, not
  the mesh normal (crosshatch fix, `gpu_draw.wgsl:1627-1634`); SH sampled one probe layer above
  the surface (sea-level probe alternation fix, `gpu_draw.wgsl:1638-1648`). Body color =
  `_DepthColor` (0.0275, 0.1418, 0.1323) extracted from the game's own `Sandbox_Water4Advanced`
  material (`gpu_draw.wgsl:1651-1658`); final radiance hue-preservingly capped 0.18–0.28 to stay
  below the LDR grade LUT's clip plateau (`gpu_draw.wgsl:1669-1679`).
- PUDDLES (textured water decals): the game's `Decal/Water Deferred Decal` RE'd from its DX11
  fragment — coverage = saturate((mask + COLOR_0.a)·1.52) with COLOR_0.a forced 0 (assembler
  writes opaque white for non-vp meshes; Unity's decal default is 0 — hard-slab fix), game
  `_Fresnel` 0.354, reflection gated to coverage interior (`gpu_draw.wgsl:1498-1536`).
- WATER_MATTE: stretched floor decals (tire marks / wet ground) detected at build by
  meters-per-texture-repeat and stripped of mirror + glint (`gpu_driven.rs:1782-1787`,
  `gpu_draw.wgsl:1523-1531`). Sea plane itself is synthesized at `manifest.seaLevel`, which is
  DERIVED from the game's scene water planes — never authored (`build_map.py:877-889`,
  `eft_extract_v2.py:1230-1259`).

### MicroSplat terrain

- 12-layer splat blend in the fragment: weights from up to 3 RGBA control maps per terrain slice,
  layer UV = terrainUV01 × per-layer rep, explicit-gradient samples
  (`gpu_draw.wgsl:176-183, 1216-1234`; `TerrainSplatGpu` `gpu_driven.rs:367-377`).
- Tiling comes FROM THE GAME: MicroSplat `_UVScale × _PerTexProps` read out of the MicroSplat
  material assets — never TerrainLayer.m_TileSize, which is garbage on these terrains
  (`extraction/unity/eft_extract_v2.py:267-329`).
- A build-time composite bake also exists (`terrain_bake.wgsl`, a wgpu port of the numpy
  `_terrain_bake_composite`, hand-rolled bilinear to match numpy exactly).

### Vert-paint splat (3-layer)

- The game's `Custom/Vert Paint` height-splat blend, RE'd from the DX11 fragment and validated in
  the web viewer: w_i = pow(Heights_i(raw_uv)·COLOR_0_i, sharpness), normalized; layer STs
  un-baked via the same V-flip-aware `detail_xform` the detail maps use (naive un-bake shifted
  136 materials by half a tile, `gpu_draw.wgsl:186-201, 1236-1283`). Matte roughness override
  (compress ×0.30, floor 0.72 — web-validated constants, `gpu_draw.wgsl:1410-1415`).

### Detail maps

- Secondary albedo/normal from Unity Standard `_Detail*` + ANGRYMESH variants, extracted with
  their STs and intensities (`eft_extract_v2.py:976-1010`). Albedo is mean-neutralized (divide by
  offline-measured mean × Unity's ×2-in-linear gain 4.5948) so it adds only local contrast
  (`gpu_draw.wgsl:1285-1306`); normal is RNM-blended in tangent space before the single
  cotangent-frame transform (`gpu_draw.wgsl:880-888, 1153-1179`); both fade over 8–15 m.

### Parallax (steep/occlusion) mapping

- 8–32 layer height-map march in the Mikkelsen cotangent frame with occlusion interpolation,
  distance-faded 25–50 m against derivative shimmer, degenerate-UV NaN guard
  (`gpu_draw.wgsl:991-1035, 1050-1063`). Height maps + `_Parallax` amounts extracted
  (`eft_extract_v2.py:1012-1022`). Measured within noise (`mod.rs:279`).

### Normal mapping

- No stored tangents: screen-space cotangent-frame TBN per fragment (`gpu_draw.wgsl:866-878`).
  BC5 two-channel normals with reconstructed Z; per-material green-flip (DirectX Y-down) OR'd
  with a pack-wide convention (`gpu_draw.wgsl:1140-1151`, `gpu_driven.rs:1779-1780`).

### Specular / environment reflection

- Dielectric GGX/Cook-Torrance (F0=0.04) lit by the SH dominant light; roughness per material,
  82% of the pack uses smoothness-in-alpha (RFA) per-pixel roughness
  (`gpu_draw.wgsl:1387-1453, 1399-1406`).
- Environment term: mix of the SH probe (evaluated toward R but probe-biased along N — passing R
  for both slid across the indoor/outdoor probe cliff on windows, `gpu_draw.wgsl:353-359, 1455-1469`)
  and an analytic sky gradient anchored to local SH luma so interiors can't blow out
  (`gpu_draw.wgsl:590-614`).

### Emissive

- `_EmissionMap`/variant-name detection ('Emissive' in the shader name enables emission with no
  `_EMISSION` keyword — EFT custom-shader quirk, `eft_extract_v2.py:818-830, 932-946`);
  factor×HDR precomputed CPU-side, texture rides the sRGB bindless array
  (`gpu_driven.rs:264-269, 2038-2049`; sampled `gpu_draw.wgsl:1101-1109`). Feeds Bloom.

### Grade LUT chain + auto-exposure

- The game's grade fitted offline (Hejl-Dawson tonemap with EFT constants → per-channel film
  curves → fitted "Fahrenheit" stage) into a 64³ LUT; at load the display encode is inverted per
  texel so the LUT emits LINEAR and the swapchain encodes exactly once (`grade.rs:1-21, 87-111`).
  Shaper domain p = sqrt(clamp(lin/4,0,1)) covering HDR to 4.0, half-texel-correct coordinates
  (`grade.wgsl:16-17, 62-71`). Camera runs `Tonemapping::None` when grade is active; fallback is
  TonyMcMapface + a hand ColorGrading (`main.rs:631-658`).
- Pass order inside `fs_grade`: FXAA → 4-tap unsharp sharpen (game ships ~0.5; ours defaults 0,
  `mod.rs:214-217`) → exposure (×1.35 default, `mod.rs:26-30`) → LUT → PRISM vignette raised to
  the 2.4 power because it's applied pre-encode (`grade.wgsl:146-180`).
- Auto-exposure (`autoexposure.wgsl`): one 64-thread workgroup, 4096 strided log-luminance taps,
  asymmetric per-second rates (down 3.0 / up 1.2), ±2 EV authority RELATIVE to a latched
  first-frame reference (an absolute 0.18 middle-grey target was anchored 7.5x off by a stale
  comment and regraded a woods exterior 2.05x — `autoexposure.wgsl:64-76`). No CPU readback.
- PARKED OPT-IN (`EFT_AUTO_EXPOSURE=1`; forced off in every preset, `mod.rs:397-398`): the
  original failure was latching the reference from a partially-streamed load frame (measured
  1.92x where 1.00x is correct). The armed-gate fix IS now implemented —
  `arm_auto_exposure` (main.rs:2278) holds arming until the pack is resident and
  `GpuLoadSignal` settles, and the shader publishes the authored exposure verbatim until armed
  (`autoexposure.wgsl:123-130`, `mod.rs:77-85`) — but the feature remains opt-in pending
  validation (`mod.rs:69-75`).

### DOF / chromatic aberration / vignette / sharpen

- DOF: Bevy's built-in Bokeh `DepthOfField` component (reads the standard ViewDepthTexture the
  GPU-driven pass writes); default off, focal 15 m, f/2.8 (`main.rs:613-622`, `mod.rs:238-240`).
- Chromatic aberration: Bevy `ChromaticAberration`, intensity slider, off by default
  (`main.rs:623-629`).
- Vignette: PRISM parameters in the grade pass (divisors 1.15/0.95, smoothstep 0.55→1.25,
  strength 0.488, `grade.rs:231, 117-121`).
- Sharpen: EFT-style unsharp mask pre-LUT (`grade.wgsl:156-163`), default 0.

### Fog / aerial perspective

- Exp² distance haze toward a fixed haze color, gated down indoors by SH directionality with a
  0.2 floor, rgb-only (correct under non-premultiplied blending)
  (`gpu_draw.wgsl:137-147, 616-634`). Default density scale 0.4 — 1.0 measured as flattening
  mid/far contrast vs the game (`mod.rs:174-181`). Measured within noise (`mod.rs:279`).

### Procedural sky cubemap

- `build_sky_cubemap` (`main.rs:1389-1443`): 6×128² Rgba16Float, horizon→zenith gradient (same
  color family as the shader's `sky_reflect` so reflections agree with the visible sky),
  below-horizon darkening, soft sun disk (s^350·3.0) + warm glow (s^8·0.3) at the bake's
  `sun_dir`, HDR so Bloom picks the disk up. `Skybox.brightness` 900 (`main.rs:580-585`).
  Rebuilt per map swap (`main.rs:568-586`).

### Bloom

- Bevy `Bloom::NATURAL` at intensity 0.06 (`main.rs:602-606`, `mod.rs:192-194`). Measured
  −6..−8% fps (`mod.rs:274`).

### FPV camera pass (drone mode)

- Analog 5.8 GHz VTX emulation post pass after tonemapping: grain, per-scanline tear bursts,
  chroma fringing, scanlines + hum bar, snow breakup; driven by a real CPU RF model (free-space
  range from the pilot position + ~2.5 dB per wall/floor crossing via
  `GroundData::segment_crossings`) (`fpv_cam.rs:1-12`, `fpv_cam.wgsl:1-27`). No-op outside drone
  mode.

### Grass / wind

- `grass.bin` (deterministic density placement from the game's GPU-Instancer detail prototypes,
  `eft_pipeline/build_grass.py`) appended as cross-quad meshes + instances through the SAME
  cull/multidraw path, tagged `ids.z==1` (`gpu_driven.rs:2708-2928`). WavingGrass
  strength/amount/speed are the terrain's authored values, extracted via the grass sidecar and
  fed through the material `vp` lane; blade tops sway by two decorrelated phases on
  `sun.gfx.w` app time (`gpu_driven.rs:2787-2789`, `gpu_draw.wgsl:941-963`).

---

## 2. Provenance: derived vs authored

Classification of each technique's inputs. EXTRACTED = read from game files; DERIVED = computed
from extracted data at build time; PORTED = math/constants ported from the game's shaders or the
calibrated web reference; AUTHORED = hand-tuned constants in our code (the debt list).

| Technique | EXTRACTED | DERIVED | PORTED | AUTHORED |
|---|---|---|---|---|
| Instancing/transforms | scene instances, full 3x4 affines (eft_extract_v2.py / assemble_bevy) | conservative spheres (gpu_driven.rs:1221), oct normals | — | — |
| Distance LOD | LODGroup size/srh/ftw/center (eft_extract_v2.py:1179) | windows + fade bands (gpu_driven.rs:1690-1752, 3186-3218) | — | 0.40 band clamp (3214), stagger hash, lod_bias 1.0 |
| Materials | albedo/normal/tint/cutoff/smoothness-in-alpha/uv STs (eft_extract_v2.py:34, 962-975) | bindless index assignment, BC compress | — | roughness clamp [0.03,1], glass 0.05, water floor 0.10 (gpu_draw.wgsl:1403-1409) |
| Terrain | MicroSplat layers, control maps, _UVScale×_PerTexProps (eft_extract_v2.py:267-329) | composite bake (terrain_bake.wgsl) | splat blend math | 0.002 weight epsilon |
| Vert-paint | heights masks, layer STs, tints, COLOR_0 | — | RE'd DX11 blend (gpu_draw.wgsl:186-191) | matte compress 0.30 / floor 0.72 (1413-1415) |
| Detail maps | maps, STs, intensities (eft_extract_v2.py:976-1010) | offline means for neutralize | Unity ×2 gain = 4.5948 | 8–15 m fade band |
| Parallax | _ParallaxMap + _Parallax (eft_extract_v2.py:1012-1022) | — | — | 25–50 m fade, layer counts 8–32, vz clamp 0.15 (gpu_draw.wgsl:999-1017) |
| Emissive | maps + factor×HDR, shader-variant detection (eft_extract_v2.py:818-830) | — | — | UI scale only |
| SH volume | world tris (pack), practicals (eft_extract_lights.py) | grid, bake, validity, ground/top ratio (sh_bake.rs, gpu_driven.rs:1076-1137) | Unity APV validity concept | **sky model 0.35+0.75·y (sh_bake.wgsl:122-124); sky_scale; LIGHT_SCALE 6.0 (gpu_driven.rs:444); bounce boost/emis_gain; SH_NORMAL_BIAS 0.75; ambient_floor 0.03 (gpu_draw.wgsl:1332)** |
| **Sun direction** | — | — | — | **FALLBACK CONSTANT [0.449, 0.799, -0.400] — EFT scenes ship no Directional light (bake_volume2.py:334-338 skips authored sun/moon entities; sh_bake.rs:847-852). The viewer's own fallback: main.rs:1514. The single most consequential authored value: shadows, shafts, sky disk, and the baked sun all hang off it.** |
| Realtime lights | positions/ranges/colors/cones, switch groups (eft_extract_lights.py, eft_extract_switches.py) | CSR grid (gpu_driven.rs:493) | Unity falloff window | scale 6.0, SH-occlusion smoothstep(0.12, 0.85) (gpu_draw.wgsl:570) |
| Sun shadows | — (direction is authored, above) | quantized fit from view (6165-6435) | — | splits [0.5,15,80,250,700], 3072², snap 16 texels, radius step 1 m, extrude 80 m, cap 0.12, fade 600–700, overlap 0.10, PCF offsets (gpu_driven.rs:722-748), gate smoothsteps (gpu_draw.wgsl:1354-1357) |
| Volumetrics | — | cascades reused | HG phase (physics) | **all VOL_*: STEPS 12, 25–300 m, g 0.40, density 0.00145, scatter 8.0, sun color (gpu_draw.wgsl:649-683)** |
| Water: deep body | **_DepthColor from Sandbox_Water4Advanced (gpu_draw.wgsl:1651-1658)**; WaterBasicNormals map | sea level from scene water planes (build_map.py:877-889); matte-vs-puddle by m/repeat (gpu_driven.rs:1782-1787) | drift speeds + layer scheme from _water.js / FX-SimpleWater4 (1587-1598, 1605-1626) | ripple amp 0.06 + falloff, octave wavelengths 18/5 m, Nyquist gate edges 0.10–0.22, refl weight 0.08–0.24, radiance cap 0.18–0.28, fresnel F0 0.02 (1559-1684) |
| Water: puddles | mask textures | luma-vs-alpha mask flag (gpu_driven.rs:5107) | _FadeStrength 1.52, _Fresnel 0.354, refl 0.88 (RE'd DXBC, 1498-1536) | tail suppression smoothstep(0.015, 0.10) |
| Grade chain | LUT bytes (pack sidecar) | LUT linearization/repack (grade.rs:87-111) | grade fit (Hejl-Dawson + film + Fahrenheit → make_grade_lut.py, grade.rs:1-8); PRISM vignette params (grade.rs:231); sharpen concept (~0.5 in game) | **exposure 1.35 (mod.rs:26-30)**; TonyMcMapface fallback grade (main.rs:637-658) |
| Auto-exposure | — | measured reference | photographic log-average | rates 3.0/1.2, ±2 EV, luma clamps (autoexposure.wgsl:56-77) |
| FXAA | — | — | console-FXAA algorithm | threshold max(0.05, 0.15·lmax), ±2 texel clamp, strength 0.75 (grade.wgsl:118-129, mod.rs:223-226) |
| SSAO | — | — | standard horizon SSAO | taps 10, radius 0.7 m, power 1.5, fade 80 m, bias 0.08, falloff (ssao.rs:207, ssao.wgsl:99-115) |
| Fog | — | indoor gate from SH directionality | — | **density 0.00075, color (0.44,0.49,0.58), indoor floor 0.2, default scale 0.4 (gpu_draw.wgsl:143-147, mod.rs:174-181)** |
| Sky cubemap + sky_reflect | — | sun position = the authored sun_dir | — | **horizon (0.66,0.72,0.82), zenith (0.92,0.98,1.10), disk powers 350/8 and gains 3.0/0.3, below-horizon 0.55, brightness 900 (main.rs:1407-1420, 582); SKY_REFL_GAIN 1.45 (gpu_draw.wgsl:135)** |
| Spec/env | roughness from textures | dominant light from SH | GGX/Smith/Schlick (physics) | SPEC_STRENGTH 1.5, ENV_REFL_STRENGTH 1.6 (gpu_draw.wgsl:126-130) |
| Bloom / DOF / chroma | — | — | Bevy implementations | intensity 0.06; focal 15 m / f2.8 defaults (mod.rs:194, 238-240) |
| Grass | prototypes/densities/textures; WavingGrass params (gpu_driven.rs:2787) | placement (build_grass.py); cross-quad geometry | Unity WavingGrass semantics | sway phase constants 2.13/0.87/1.61 etc. (gpu_draw.wgsl:958-961), cull px 1.5/4.0 (mod.rs:172) |
| FPV cam | — | RF model from pack geometry crossings | analog-video lore | every effect constant (fpv_cam.wgsl:60-97) |

**The AUTHORED debt list, ranked by visual consequence:**

1. `sun_dir` — a constant masquerading as data. Everything directional derives from it.
2. The sky: bake-side gradient (`sh_bake.wgsl:122-124`), viewer cubemap colors
   (`main.rs:1407-1420`), `sky_reflect` gradient (`gpu_draw.wgsl:608-614`), and `FOG_COLOR`
   (`gpu_draw.wgsl:146`) are four separately-authored descriptions of the same atmosphere.
3. Fog density/scale (`gpu_draw.wgsl:143`, `mod.rs:174-181`).
4. The whole `VOL_*` block (`gpu_draw.wgsl:649-683`).
5. Shadow cascade splits/fades/cap (`gpu_driven.rs:722-748`).
6. `DEFAULT_GRADE_EXPOSURE` 1.35 (`mod.rs:26-30`) and light/GI scales (`gpu_driven.rs:444`).
7. Water look constants beyond the extracted `_DepthColor` (`gpu_draw.wgsl:1559-1684`).
8. `SPEC_STRENGTH`/`ENV_REFL_STRENGTH`/`SKY_REFL_GAIN`, SSAO parameters, bloom intensity,
   FXAA strength, ambient floor 0.03.

The game's actual weather/TOD system (its sun position, sky state, fog color/density per
weather preset) lives in GameAssembly-side config the extraction does not yet reach; every
constant in items 1–3 is in principle derivable from it.

---

## 3. Groundbreaking possibilities

Ranked. Frame costs are engineering estimates for an RTX 5090 at 1440p against the current
~11.9 ms woods-Ultra frame (`GFX_BENCH_woods_alllod` `u_base`); only the baseline and the quoted
comparison numbers are measured. Size: S ≲ 1 day, M ≈ days, L ≈ week+.

1. **SSR from the new prepass** (M). The inputs now exist: per-pixel world normal + roughness
   (Rgba16Float) and a dedicated Depth32Float (`gpu_prepass.wgsl:1-7` — written explicitly as
   the SSR enabler). Add a ViewNode between main pass and SSAO (`ssao.rs` is the exact template:
   same bind-group pattern, same HDR ping-pong) that hierarchically ray-marches the prepass
   depth and blends mirror hits into glossy pixels by the stored roughness. Two prerequisites
   visible in code: (a) unlink the prepass from SSAO's toggle (`gpu_driven.rs:6951-6953` gates
   it on `settings.ssao`); (b) water/glass discard from the prepass
   (`gpu_prepass.wgsl:151-154`), so SSR on water needs the v2 noted in the prepass header (write
   the water branch's ripple normal) or a composite in the water branch itself. Payoff: replaces
   the analytic-gradient reflections (`sky_reflect`) with real scene reflections — the sea
   mirroring the actual treeline and buildings would exceed the game's own SSR-less overcast
   water. Est. 0.5–1.5 ms (half-res trace + upsample).
2. **Depth-prime the main pass from the prepass** (S/M). The volumetric march's cost is
   explicitly overdraw-multiplied ("a forward pass with no depth prepass, so overdraw multiplies
   it", `gpu_draw.wgsl:646-648`), and the prepass already rasterizes the opaque scene to its own
   depth. Either render the prepass into the main-pass depth buffer (then main pass uses
   `Equal`-style testing), or run it always-on and let early-Z kill occluded fragments.
   Directly attacks the 5.4 ms volumetric bill and every heavy fragment (water, terrain splat,
   parallax). Est. saves 1–3 ms on Ultra+volumetrics; costs the ~1 ms prepass when it wasn't
   already on.
3. **Froxel volumetrics + local light scattering** (L). The in-fragment march was chosen for
   plumbing convenience, not physics (`gpu_draw.wgsl:642-648`). A camera-frustum 3D froxel LUT
   (e.g. 160×90×64) filled by one compute dispatch marching the SAME cascade array, sampled in
   `apply_fog`, makes cost resolution-independent and overdraw-immune (fixed ~0.5–1 ms vs the
   measured 5.4 ms) — and the CSR light grid (`gpu_draw.wgsl:250-264`) can be evaluated per
   froxel, giving lamp glow in mall interiors, which EFT itself only fakes with sprites. Add
   temporal reuse per froxel (cheap; no screen-space ghosting).
4. **Physical sky + sun disc replacing the procedural gradient** (M). `build_sky_cubemap`
   (`main.rs:1389-1443`) is a two-color lerp; the same `sun_dir` could drive a Hosek-Wilkie or
   Preetham evaluation at load (CPU, 6×128² texels — free at runtime). Feed the SAME evaluation
   into `sky_reflect` and derive `FOG_COLOR` from the horizon integral: one change deletes three
   entries of the authored-debt list (sky cubemap colors, sky_reflect gradient, fog color) and
   keeps them mutually consistent by construction. Pair with extracting the game's real per-map
   sun/TOD (the roadmap's sun_dir item) for the full win. Est. 0 ms runtime.
5. **Gerstner-displaced water** (M). The extraction already reads the FX/SimpleWater4 material
   family (it is where `_DepthColor` came from, `gpu_draw.wgsl:1651-1658`); the same materials
   carry Unity Water4's `_GAmplitude/_GFrequency/_GSteepness/_GDirectionAB/_GDirectionCD` —
   currently extracted nowhere (`grep` confirms no reference in the repo). The sea is a synthetic
   quad (`gpu_driven.rs:1268-1277`); emit it as a subdivided grid near the camera and displace
   verts in the water branch of the vertex stage — precedent for per-class vertex animation
   already exists (grass sway, `gpu_draw.wgsl:947-963`; time already on `sun.gfx.w`). Actual 3-D
   swell + moving silhouettes against the horizon would exceed the game's mostly-flat normal-map
   sea. Est. 0.1–0.3 ms.
6. **PCSS contact-hardening shadows** (M). All inputs are bound: 4-cascade depth array +
   comparison sampler + per-cascade world texel sizes (`gpu_draw.wgsl:246-248, 771-810`).
   Replace the fixed 3x3 tent with blocker search (plain `texture_depth_2d_array` view, ~9
   taps) → penumbra estimate → variable-radius Poisson PCF. Sun shadows currently have one
   softness at every range; contact-hardening reads dramatically better on tree shadows, and the
   quantized-fit cache (`gpu_driven.rs:6298-6347`) is unaffected since only receiver-side
   filtering changes. Est. 0.3–0.8 ms on shadowed pixels.
7. **Higher-density multi-bounce SH rebake** (M, offline-only cost). The GPU baker already
   batches dispatches to dodge TDR and chunks >4 GiB maps (`sh_bake.wgsl:1-11, 128`), and pass B
   is a pure function of pass A (`sh_bounce.wgsl:194-205`) — iterating it N times converges to
   multi-bounce. Density: XZ_TARGET 3 m / Y 4 m and the 2.6 M probe cap (`sh_bake.rs:42-50`)
   could go to 1.5 m/2 m within VRAM (volume triples; a 96 MiB volume is still small next to the
   6.4 GiB texture set). Sharper interior gradients directly improve the dominant-light term,
   the shadow gates, and the SH-occluded realtime lights — three consumers per probe. Runtime
   cost: zero (same sampling).
8. **Temporal accumulation / TAA** (L). No motion vectors exist (the GPU-driven path bypasses
   Bevy's prepasses), but the world is static: camera-only reprojection from the prepass depth +
   previous `view_proj` covers everything except doors/grass/water. Wins waiting for it, all
   noted in code: A2C's 4-step treeline quantization (`mod.rs:113-117`), the volumetric jitter
   that is world-anchored precisely "because there is no TAA to resolve it"
   (`gpu_draw.wgsl:699-704` — with TAA the march could drop to ~6 jittered steps), specular
   shimmer FXAA only blurs. Est. 0.3–0.6 ms + history memory; the real cost is engineering
   (disocclusion, the blend pipeline placement before the grade).
9. **Hi-Z occlusion culling** (M/L). `cs_cull` is frustum + screen-size only
   (`gpu_cull.wgsl:144-176`); interchange's mall and streets' canyons draw the whole depth
   complexity. Build a depth pyramid from last frame's (or the prepass') depth, test instance
   spheres against it in `cs_cull` — the buffer layout already supports rejection at that one
   choke point. Two-phase (draw-late) correction needed for disocclusion. Biggest structural win
   for the maps where LOD helps least. Est. saves multiple ms indoors; ~0.1 ms pyramid cost.
10. **Offline texture upscaling** (M, offline). The BC texcache pipeline
    (`gpu_driven.rs:5241-5427`) is a clean insertion point: an ESRGAN-class 2x pass over source
    PNGs before mip build would exceed the game's own texel density (EFT ships many 512²/1024²
    surfaces). The constraint is measured, not guessed: full-res already costs +2.2 GiB VRAM
    (`mod.rs:277`), so upscaling wants a per-material priority list (hero surfaces: terrain
    layers, roads, large walls) rather than a blanket 2x. Runtime cost ~0; sampling cost
    unchanged (same mip footprint at distance).
11. **Screen-space contact shadows** (S). One more consumer of the prepass depth: a short (8–16
    step) screen-space ray toward `sun_dir_texel.xyz` filling the gap below the PCF bias where
    small props float. Cheap, and the sun direction + depth are already bound in the main pass.
    Est. 0.2–0.4 ms.

Cross-cutting note for 1, 2, 8, 9, 11: they all feed on the prepass. It currently runs only
under SSAO (`gpu_driven.rs:6951`); the first project of any of these is promoting it to a
first-class always-on (or consumer-refcounted) pass — after which each additional consumer is
nearly free.
