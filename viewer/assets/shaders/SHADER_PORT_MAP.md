# EFT native viewer — WGSL shader port map

Ports the pure MATH from the web three.js/WebGL viewer modules (under
`beamng_blender_pipeline/tarkmap/out/`) into Bevy 0.17 / wgpu 26 WGSL. The GLSL/WebGL
modules are the cleaner reference (the TSL ones fight WebGPU-batch-compile constraints that
do not exist natively). Every file below is grounded in the real web module noted.

Shaders assume Bevy's `naga_oil` preprocessor: importable modules declare
`#define_import_path`, consumers `#import`. Global bindings declared in an imported module are
shared into the consumer, so `sh_gi.wgsl` owns the `@group(1)` SH-volume bindings and
`instanced.wgsl` reuses them by importing the module.

## Wiring status (M0)

- **`instancing_m0.wgsl` is the ONE shader wired today** (loaded by `render/instancing.rs`).
  It is the minimal first-pixel surface shader: applies the full ROW-MAJOR 3x4 affine to raw
  verts, cofactor normal transform (shear/mirror-correct, no decompose), double-sided
  front-face normal flip, flat sun+ambient lambert. No `#import` — it uses only
  `bevy_pbr::view_transformations::position_world_to_clip`.
- The five modules below (`instanced/sh_gi/splat/grade/cull`) are the **M1–M4 port targets**,
  NOT yet wired. They use `naga_oil` `#import` modules and need Bevy shader-registration /
  compute-node wiring. `instanced.wgsl` supersedes `instancing_m0.wgsl` at M2/M3.
- The old duplicate embedded WGSL strings that lived in `render/instancing.rs` are **removed**;
  `cull.wgsl` here is the single source for the compute cull.
- **Normal green-flip fix applied:** `instanced.wgsl` `perturb_normal` now negates sampled
  `n.y` when the material's DirectX-normal flag (`MF_NORMAL_DX`, set from
  `materials.json.normalGreenFlip` / `manifest.conventions.normalMapGreenFlip`) is set — it
  HONORS the declared convention instead of hardcoding a flip.

## Web module -> WGSL file

| WGSL file | Web source | What was ported |
|---|---|---|
| `instancing_m0.wgsl` | (new — the wired M0 first-pixel path) | Full 3x4 affine -> mat3+translation (NO decompose); cofactor normals; double-sided; flat lambert. The only shader wired today. |
| `instanced.wgsl` | (new — the GPU-driven surface path) + `_vpsplat.js`, `_gi.js` call sites | Per-instance FULL 3x4 affine -> mat3+translation (NO decompose); cofactor normal transform; bindless albedo/normal sample; `_MainTex*_Color`; cutout discard; SH-GI + one sun term; per-pixel cotangent-frame normal mapping (no tangent attribute in the eftpack vertex layout) |
| `sh_gi.wgsl` | `_gi.js` (`_fastE`/`_fullE`), `_gigl.js` (DC parity), `volume.json` layout | L1 SH irradiance sampled by WORLD POS from a REAL 3D texture; cosine-convolution reconstruction (A0=pi, A1=2pi/3); Chebyshev-visibility leak gate (DDGI moments), distance-gated |
| `splat.wgsl` | `_vpsplat.js` | EFT "Custom/Vert Paint SoftCutout Decal" 3-layer height splat: `w_i = pow(Heights*vColor, Blend)` normalized; near-black/zero-coverage fallbacks; ground-matte roughness (x0.30, floor 0.72); SoftCutout feather alpha |
| `grade.wgsl` | `_gradegl.js` | Exposure -> LUT shaper `sqrt(clamp(c/4,0,1))` -> 3D LUT -> PRISM vignette. Output is DISPLAY-ENCODED (Hejl baked into the LUT); tonemap OFF, non-sRGB target |
| `cull.wgsl` | `_cullgl.js` + `_lod.js` | Per-instance world bounding sphere (max column-norm radius, no decompose); sphere-vs-6-plane frustum; per-mesh contiguous compaction -> DrawIndexedIndirect; `_lod.js` screen-height gate |

## Binding scaffold (Rust side must match)

Applies to the M1–M4 modules (not `instancing_m0.wgsl`, which reuses Bevy's mesh view bind
group at group(0) only). The M0 render code is live in `render/instancing.rs`; the bindless
material + SH-GI + compute-cull bind groups below are the M1+ target layout.

- **group(0)** — global view/frame uniform (`View` in `instanced.wgsl`): `view_proj/view/proj`,
  `camera_pos`, `exposure`, `sun_dir`, `sun_color`, `gi_str`, `time`.
- **group(1)** — SH-GI volume (`sh_gi.wgsl`): `sh_c0..c3` (4 × `texture_3d<f32>` Rgba16Float,
  one per L1 coeff, `.rgb` used), `sh_samp` (Linear/Clamp/no-mips), `gi: GiParams` uniform,
  `sh_vis0..3` (optional DDGI moment bands, Nearest). Loader must upload with the correct axis
  permutation so texel order is (x,y,z) — the bin is probe-major `((z*ny)+y)*nx+x`.
- **group(2)** — bindless materials (`instanced.wgsl`): `textures: binding_array<texture_2d<f32>>`
  (full-res BC7 sRGB albedo / BC5 normal, imported in place from `eft_assets/<ds>/tex`),
  `tex_samp`, `materials: array<Material>` storage.
- **cull.wgsl group(0)** — its own compute layout: `CullParams` (6 frustum planes, camera+`tan(fovV/2)`,
  lodBias/clampCoarsest/enable, counts), `instances_in` (raw = `instances.bin` + loader material_id),
  `mesh_bounds` (per-mesh local sphere + contiguous region), `instances_out` (compacted, = the
  instance-rate vertex buffer for `instanced.wgsl`), `vis_count` (atomic per mesh), `draws`
  (DrawIndexedIndirect), `submeshes` (static draw templates), optional LOD SoA + HZB.
- **grade.wgsl group(0)** — `scene_tex` (linear HDR RT) + sampler, `lut3d` (64³) + sampler,
  `GradeParams`.

### Instance record layouts
- `instances.bin` (contract, 72 B): affine f32x12 + meshId u32 + lodGroup i32 + lodIndex i32 +
  rootId u32 + flags u32. Loader repacks into the 16-B-aligned `InstanceRaw` (adds `material_id`).
- Compacted `InstanceGpu` (cull output = `instanced.wgsl` instance attrs loc4..7): m0,m1,m2
  (vec4 f32, the ROW-MAJOR affine rows) + ids (mesh_id, material_id, flags, lod_packed).

### Vertex layout (manifest, matches `instanced.wgsl` VertexIn loc0..3)
position f32x3 @0, normal f32x3 @12, uv f32x2 @24, color unorm8x4 @32 (stride 36; loader may pad
to 16-B for the storage-buffer path). UV tiling is BAKED into vertex uv by assemble_bevy;
`uv_xform` in `Material` is reference-only (do not double-apply — except the vert-paint per-layer
tiling `vp_til0..2`, which IS applied in-shader to the raw mesh uv).

## Correctness rules honored (tarkov-unity-extraction skill)

1. **NEVER TRS-decompose.** The affine is applied whole. Positions use `lin*p+T`; normals use the
   COFACTOR matrix `mat3(cross(c1,c2),cross(c2,c0),cross(c0,c1))` = det·inverse-transpose —
   shear/non-uniform correct without inverting. Cull radius uses `max(|col0|,|col1|,|col2|)`.
2. **Mirror (det<0)** is handled by front-face/winding state (an instance flag bit the Rust
   pipeline sets), NOT by baking. Shaders normalize the transformed normal and let winding own
   the sign; opaque is two-sided by default (`MF_TWO_SIDED`), single-sided only for shell faces.
3. **`_MainTex*_Color`.** Albedo is always tinted by the linearized `_Color` (`Material.tint`),
   even for textured mats — flat near-white base textures are tinted dark/rust purely via tint.
4. **Vert-paint.** vColor is the per-LAYER blend WEIGHT, never a tint. Heights mask sampled ONCE
   at the raw uv (R/G/B = layer0/1/2). Zero-coverage -> base layer; near-black blend -> layer0.
5. **Grade output is display-encoded** — Bevy runs `Tonemapping::None` and the grade target is a
   non-sRGB (Unorm) surface; no OutputPass/second sRGB after it.
6. **SH coeffs are RADIANCE** (not irradiance); the cosine convolution turns them into irradiance.
   Axis map is non-obvious: `c1<-n.y, c2<-n.z, c3<-n.x`.

## Divergences from the web (intentional, per the locked design)

- **Full L1 GI, not DC-only.** The web viewers ship DC-only (`c0` term) because GLSL1 has no
  `sampler3D` and `texture3D()` hangs WebGPU BatchedMesh compile. Native Bevy has neither
  constraint -> `sh_gi.wgsl` does the full directional L1 from a real 3D texture. `sh_dc()` kept
  for A/B.
- **GPU compaction, not CPU.** `cull.wgsl` replaces `_cullgl.js`'s CPU compaction + camera-motion
  throttle with a compute pass -> indirect multidraw.
- **Real 3D LUT.** `grade.wgsl` samples a 64³ 3D texture (HW trilinear does the blue-slice lerp);
  the web's 512×512 8×8-tiled manual lerp is preserved only as a commented fallback.

## Open items / TODO for the Rust integration

1. **Per-submesh material routing.** WGSL/wgpu exposes no draw-index builtin, so multi-submesh
   meshIds cannot pick their material from the draw alone. Current scaffold carries a single
   `material_id` in the compacted instance record (exact for single-material meshIds — the common
   case since assemble groups by `(mesh, sub_sig)`). For genuinely multi-material meshIds, either
   (a) emit one compacted instance copy per submesh (cull writes per-submesh regions), or
   (b) adopt `DrawIndirectCount` + a per-draw material buffer indexed by a base-instance offset.
   Pick one before multi-submesh meshes render with the wrong material.
2. **Mesh bounding spheres.** The manifest `meshes[]` has no bounding sphere; the loader must
   compute per-mesh local center/radius (feeds `MeshBounds` for `cull.wgsl`) or the manifest must
   be extended.
3. **HZB occlusion.** `cull.wgsl::occlusion_visible` is a conservative stub (returns true). Wire a
   depth-pyramid pass (project world sphere -> screen extent -> sample coarsest covering HZB mip ->
   keep if front depth <= sampled max depth) and set `counts.w` bit0 to enable.
4. **Eye adaptation.** `grade.wgsl` takes a fixed `exposure` (0.18). Port `_gradegl.js measure()`
   as a compute log-luma reduction (`exposure = clamp(0.259/logmean, 0.02, 8.0)`) if desired; M1 can
   ship fixed.
5. **GI calibration.** `gi_str` defaults to ~1/pi. The web rescales the volume to the current
   analytic ambient (`openDC`/hemiLuma); a naive fixed str will over/under-light. Acceptable for M1.
6. **Leak gate data.** `sh_irradiance_gated` needs the `.vis.bin` moments uploaded as `sh_vis0..3`
   and `GiParams.dims.w` bit0 set; otherwise `sample_sh_irradiance` uses the fast HW-trilinear path.
7. **Glass / water / decal render state.** `Material` roles are tagged, but blend/depth state
   (glass BLEND + dirt-film albedo kept, water role, decal polygon-offset) lives in the Rust
   pipeline descriptors, not the shader. Wire per-role pipelines.
8. **Sky / IBL.** `_sky.js` (analytic overcast equirect) is not yet ported; bake once to a cubemap
   for skybox + a cheap specular IBL, or add a `sky.wgsl` skybox pass. Not in this initial set.
