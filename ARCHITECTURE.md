# Atlas — Architecture

A native desktop EFT map viewer (**Rust + Bevy**, GPU-driven) replacing the web three.js
`tarkmap` viewer. This repo is **code only**; it references the extracted assets *in place*
under `C:\Users\user\beamng_blender_pipeline\eft_assets\<dataset>` and never copies them
(disk is tight). The old web viewer under `beamng_blender_pipeline` stays untouched.

## Locked decisions

- **UnityPy extractor is the source of truth.** The engine is a pure *consumer*. We do not
  rewrite extraction, reimplement Unity parsing in-engine, or use AssetRipper.
- **Feed = `eft_pipeline/assemble_bevy.py`**, a fork of `tarkmap/assemble_instanced.py` that
  **reuses the correctness core verbatim** and emits the `.eftpack` below. That core is
  **VENDORED** into `eft_pipeline/tarkmap_core/`: `instmath.py`, `culls.py`, `objio.py`,
  `matsig.py` are copied byte-for-byte from `<beamng>/tarkmap/tarkmap/`; `config.py` is copied
  with **one edit** — its `ROOT`/`MAPS_DIR` anchors are repointed at the in-place beamng tree
  (so `maps/<id>/config.json` and the `eft_assets` datasets resolve without copying anything;
  override with `EFT_TARKMAP_ROOT`). The emitter imports the vendored package first; the old
  hardcoded upstream-path fallback is now dead code. The entire web-lossy tail is dropped.
- **Never TRS-decompose.** Instances carry the **full row-major 3x4 affine including shear and
  mirror**. `glam::Affine3A` / the raw instance buffer is shear- and det<0-correct.
  `bake_into` is a **degenerate-only** fallback (rank-deficient 3x3 → the `pinv` path).
- **LOD-shell dedup** (added, missing from the web/UE paths): group kept instances by
  `(lv, lod.g)` and keep only `lod.i == group-min`. (No-op on an already-LOD0-resolved
  `scene.json`; ~47% fewer instances on an `--alllod` extraction.)
- **Full-res textures**, referenced in place; Bevy imports BC7 (albedo/emissive sRGB) / BC5
  (normal, linear) on load. The SH volume is consumed as a real 3D texture.
- **Best performance here is low-overhead GPU-driven instancing**, not meshlets. The data is
  low-poly (p50 ~384 tri), already instanced (10.5k unique meshes stored once). Center the
  renderer on: bindless textures, indirect multidraw, a compute frustum/occlusion cull that
  compacts visible instances, and screen-height LOD. Meshlet/virtual-geometry is largely moot.

---

## The `.eftpack` v1 format

A pack is a directory `<map>.eftpack/` with four files. It is **self-describing**: the
manifest declares every stride and byte offset, and the Rust loader reads layout from the
manifest and hardcodes nothing — so the Python emitter and the Rust loader cannot drift.

### `manifest.json`
```
{ version:1, dataset, datasetPath, map,
  bounds:[minx,miny,minz, maxx,maxy,maxz],           // world AABB, computed from placed verts
  vertex:   { stride:36, attrs:[ position f32x3 @0, normal f32x3 @12, uv f32x2 @24, color unorm8x4 @32 ] },
  instance: { stride:80, align16:true, fields:[
                affine   f32x12 @0   (ROW-MAJOR world 3x4 incl shear),
                meshId   u32    @48,
                lodGroup i32    @52  (scene lod.g or -1),
                lodIndex i32    @56  (scene lod.i or -1),
                rootId   u32    @60  (index into manifest.roots),
                flags    u32    @64 ] },              // + 12B pad → 80 (16B-aligned for storage buffers)
  meshes:[ { id, name, vtxOffset(bytes), vtxCount, idxOffset(bytes), idxCount,
             submeshes:[ { materialId, idxStart(local index), idxCount } ] } ],
  instanceCount, materialCount,
  roots:[name,…],                                     // rootId → GameObject root name (semantic layer)
  lodGroups:[ { size, center(conjugated), srh[], ftw[], n, lastIsBillboard,… } ],
  flagsLegend, conventions, sidecars, note }
```

**Flag bits** (`instance.flags`): `0x1` MIRROR (det<0 → renderer flips front-face/winding),
`0x2` TERRAIN (MicroSplat splat shader), `0x4` BAKED_WORLD (identity affine, geometry
pre-baked to world; no per-instance normal matrix).

**Conventions** (recorded so the loader never double-applies):
- `affine` is row-major 3x4 incl shear; the renderer builds the per-instance **normal matrix as
  the inverse-transpose of the 3x3** (shear-correct — no decomposition).
- **UV V-flip is baked** into vertex UV (`uvVFlipBaked:true`, top-left origin) — Unity's
  bottom-left UVs flipped to the wgpu/PNG top-left convention. `uvXform` in `materials.json`
  is **reference only**; tiling is already baked into the vertex UV.
- **Normal maps are DirectX-convention** (`normalMapGreenFlip:true`): the loader flips G on
  BC5 import (or the shader negates `n.y`). Textures are referenced in place and can't be
  pre-flipped.

### `meshes.bin`
All meshes' interleaved vertices (36-byte records) concatenated first, then all u32 indices.
Per-mesh `vtxOffset`/`idxOffset` are **byte** offsets; `submesh.idxStart` is a **local index**
offset within that mesh. A draw call is `baseVertex = vtxOffset/36`,
`firstIndex = idxOffset/4 + submesh.idxStart`, `indexCount = submesh.idxCount`.

### `instances.bin`
Fixed-stride 80-byte records (layout above). `affine` is the full conjugated 3x4
(`apply_global(m)[:12]`) — **not** `mat4_colmajor` (that is the glTF column-major transpose,
wrong here). One geometry group `(mesh, material-signature)` = one `meshId`; mirror and
non-mirror instances share it (winding handled per-instance via the MIRROR flag).

### `materials.json`
One record per unique submesh material signature (`matsig.sub_sig`):
```
{ id, role:'opaque|cutout|glass|decal|water',
  albedo:<abs path|null>, normal:<abs path|null>, uvXform:[sx,sy,ox,oy] (reference),
  alphaMode:'OPAQUE|MASK|BLEND', alphaCutoff, tint:[linR,linG,linB, a],
  metallic, roughness, normalScale, normalGreenFlip, doubleSided,
  emissive:{ texture, factor[3], hdr }|null,
  roughnessFromAlbedoAlpha:bool,   // smA family: roughness = 1 - albedo.a
  specMap:<abs path|null>,         // _SpecMap luma → roughness
  vp:{ layers:[{albedo,normal,uv,tint}], heights, blend, softCutout? }|null }
```
`tint` linearizes Unity's sRGB `_Color` (alpha stays linear/coverage); it is applied even for
textured materials (EFT albedo = `_MainTex × _Color`). `doubleSided` defaults true (EFT
deferred draws building shells solid from both sides).

### Sidecars — referenced in place, never copied
`terrainLayers` (MicroSplat `terrain_layers/manifest.json`), `lights` (`lights_<lv>.json`),
`volume` (`out/<map>/volume.bin`) + `volumeMeta`/`volumeVis`, `grassJson`. The SH layout is
read from `volume.json`.

---

## What `assemble_bevy.py` keeps vs. drops

**Kept verbatim:** `culls.Culls.filter` (Unity-hidden drop, junk-root denylist with the `SBG`
*protection* allowlist, off-map backdrop cull on raw translation) and `keep_submesh`
(shadow/billboard-LOD/fog/untextured-proxy); `make_conjugator` conjugation `G·M·G⁻¹` on **raw**
verts; the objio dedup + smooth-normal build; `matsig.sub_sig` grouping (per-variant colour);
`bake_into` (degenerate pinv path only). Correctness fixes ported: decal-normal-map drop,
SpeedTree cutout promotion, water recovery, cutout `alphaCutoff`, glass keeps its dirt film.

**Divergences:** the web three-way instance/bake gate → **emit full 3x4 for every instance**
(mirror via flag, bake only rank-deficient); web payload split → **LOD-shell dedup**; UV
V-flip and normal green-flip recorded as conventions; `_Color`/gloss/metal/smA/spec carried as
real material fields.

**Dropped wholesale (the web-lossy tail):** `build_textures` 512 downscale + ETC1S/UASTC KTX2;
the 6-step gltf-transform chain (quantize / etc1s / uastc / meshopt / fix-texcoords /
deinstance); `split_glb`; the `EXT_mesh_gpu_instancing` TRS split; the `~ra~`/`~mr~`
real-specular synth textures + `texgen`/`texmap` id indirection; sidecar **copies**.
Cosmetic z-fight hacks (anti-coplanar stagger, `_detect_overlays`) are dropped for M0 — Bevy
uses depth bias instead.

Run: `python -m eft_pipeline.assemble_bevy interchange [--out <dir.eftpack>] [--limit N]`.
Verified against `interchange_v2`: 65,167 kept instances / 6,151 unique meshes / 341 materials
in the first 300 groups, binary layout round-trips (unit normals, in-range local indices,
resolved sidecars).

---

## Milestone plan (Bevy path)

**M0 — Pack + static render (target: Interchange). — IMPLEMENTED (needs a `cargo build` pass).**
`assemble_bevy.py` emits the `.eftpack`. The Bevy loader (`eftpack.rs`) parses the manifest
(strides/offsets/conventions read, never hardcoded) and unpacks `meshes.bin` per-mesh. The
render path (`render/instancing.rs`) is the **Bevy-0.17 low-level custom-instancing pattern**:
one Bevy `Mesh` + one instanced draw per unique meshId, its instances uploaded as an
instance-rate vertex buffer of the **full row-major 3x4 affine**. The vertex shader
(`instancing_m0.wgsl`) applies that affine to **raw** verts and transforms normals by the
**cofactor matrix** (det·inverse-transpose — shear-correct, no decompose). Mirrors (det<0) are
correct with **zero baking**: the pipeline is **double-sided** (`cull_mode = None`) and the
cofactor flips the normal sign for det<0, so no per-instance front-face pipeline is needed.
M0 lighting is a flat sun+ambient lambert — **textures/materials/GI are M2+**. Goal met: the
map on screen with correct placement/shear/mirror + a free-fly camera. No culling yet.

> The crate resolves (bevy 0.17.3) but has **not been compiled** here (disk/time). First
> action for M1 is a `cargo build` to confirm the 0.17.3 render-API surface used by the custom
> pipeline (`MeshPipeline::specialize`, `MeshAllocator` slices, the `DrawCustom` render command).

**M1 — GPU-driven instancing (the performance core).**
Compute **frustum cull + compaction**: per instance, world sphere center =
`affine · meshBoundingSphere.center`, radius = `meshBoundingSphere.radius · max column-norm of
the 3x3` (shear/mirror-correct, no decompose); test 6 Gribb-Hartmann planes; atomically append
survivors per meshId and fill an **indirect multidraw** arg buffer. Add per-mesh bounding
spheres at load (manifest `meshes[]` has none). Bindless textures + indirect multidraw =
few, large draws. Port `_lod.js` **screen-height LOD** (`H = size/(dist·2·tan(fovV/2))·bias`,
first `srh` threshold; `clampCoarsest` on for a viewer) into the cull pass using `lodGroups`.

> **Distance-LOD SHIPPED** (lod 1/3–3/3). Packs built `--alllod` carry every shell; the CPU bakes
> each instance's screen-height distance window (`near'=size/2·srh[i-1]`, `far'=size/2·srh[i]`) as an
> f16 pair in `ids.w`, and `cs_cull` selects one shell per group by camera distance — a **live
> cull-uniform toggle** (`Graphics ▸ Distance LOD` + bias + force-shell), no rebuild. Default packs
> resolve to the finest shell only (`ids.w==0` sentinel) → byte-identical to the pre-LOD path.
> Producer→consumer contract verified end-to-end; correctness proven on a synthetic 3-shell inject
> (force-0 ≡ max-detail byte-identical, distance grounds near / raises far). A post-ship Codex +
> multi-agent audit found 6 LATENT defects (all `--alllod`-only; lean packs are byte-identical
> sentinels) — 4 hardened (present-adjacency holes, group-center shell-switch metric via a
> `lod_centers` buffer, f16/lod_index/bias guards), 2 deferred to when the alllod pipeline is first
> built (producer shared-renderer span, door multi-shell binding). The real per-frame perf delta
> awaits a decimated `--alllod` rebuild (coarse shells = fewer triangles).

**M2 — Lighting / GI.**
SH-L1 irradiance volume as a real `Rgba16Float` **3D texture** (probe order `((z·ny)+y)·nx+x`;
axis map `c1←y, c2←z, c3←x`); full L1 reconstruction
`E(n) = c0·0.8862269 + 1.0233267·(c1·n.y + c2·n.z + c3·n.x)`, `max(0)`, `×albedo`. Load the
real practical lights (`lights_<lv>.json`). Optional DDGI leak-gate (`volume.vis.bin`,
Chebyshev) as a quality lever.

**M3 — EFT material shaders (WGSL ports).**
Vert-Paint 3-layer height splat (`_vpsplat.js`: weights `pow(Heights·vColor, blend)`
normalized, Heights sampled once at raw UV, roughness floored 0.72, SoftCutout alpha, **kill
the stock `×vColor` tint**); glass (dirt film × tint, fresnel); water (animated); emissive
(HDR strength). Occlusion cull (Hi-Z) folded into the M1 compute pass.

**M4 — Display chain + sky.**
EFT grade **3D LUT** as a `64³` texture (shaper `sqrt(clamp(lin/4,0,1))`, HW trilinear) + PRISM
vignette; the LUT bakes the sRGB encode, so **`Tonemapping::None` + non-sRGB (Unorm) target**
(no double gamma). Optional eye-adaptation (compute average-luma → exposure). Analytic sky
(`_sky.js`) baked once to a cubemap for skybox + IBL.

**M5 — Terrain + polish.**
MicroSplat terrain from `terrain_layers/` (per-tile control maps → up to 12 layer weights);
grass from the baked density grids (GPU Instancer detail, deterministic — never client-scatter,
it's a visibility-cheat surface). Multi-map on-demand pack builds; the semantic
GameObject-name layer (`manifest.roots`) for picking/labels; depth-bias z-fight handling in
lieu of the dropped stagger/overlay hacks.
