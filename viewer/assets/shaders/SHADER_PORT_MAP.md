# EFT native viewer — WGSL shader port map

Ports the pure MATH from the web three.js/WebGL viewer modules (under
`beamng_blender_pipeline/tarkmap/out/`) into Bevy 0.17 / wgpu 26 WGSL. The GLSL/WebGL
modules are the cleaner reference (the TSL ones fight WebGPU-batch-compile constraints that
do not exist natively). Every file below is grounded in the real web module noted.

Shaders assume Bevy's `naga_oil` preprocessor: importable modules declare
`#define_import_path`, consumers `#import`. Global bindings declared in an imported module are
shared into the consumer, so `sh_gi.wgsl` owns the `@group(1)` SH-volume bindings and
`instanced.wgsl` reuses them by importing the module.

## Wiring status (M0 / M2)

- **`instancing_m0.wgsl`** is the M0 CPU-fed path (loaded by `render/instancing.rs`), kept as the
  A/B fallback (`EFT_RENDER=m0`). It is the minimal first-pixel surface shader: applies the full
  ROW-MAJOR 3x4 affine to raw verts, cofactor normal transform (shear/mirror-correct, no decompose),
  double-sided front-face normal flip, flat sun+ambient lambert. Per-instance data arrives as
  instance-rate VERTEX attributes (loc3..6). No `#import` beyond
  `bevy_pbr::view_transformations::position_world_to_clip`.
- **`gpu_cull.wgsl` + `gpu_draw.wgsl` are the LIVE M2 GPU-driven path** (`EFT_RENDER=gpu`, the default,
  loaded by `render/gpu_driven.rs`). `gpu_cull.wgsl` compute-culls the GPU-resident instances and
  compacts survivors + fills DrawIndexedIndirect (`cs_reset` + `cs_cull`); `gpu_draw.wgsl` is the
  indirect-draw fork of `instancing_m0.wgsl` — same vertex-transform math, but FETCHES the instance from
  the `instances` SSBO via `visible[instance_index]` instead of instance-rate attrs. See "M2 GPU-driven
  binding scaffold". (Earlier draft duplicates `cull.wgsl` + `instancing_gpu.wgsl` were DELETED — they
  were never referenced by the Rust and had a swapped group(1) binding order; do not resurrect them.)
- The four remaining modules (`instanced/sh_gi/splat/grade`) are the **M3–M4 port targets**, NOT yet
  wired. They use `naga_oil` `#import` modules. `instanced.wgsl` (textured/GI) supersedes
  `gpu_draw.wgsl` at M3.
- The old duplicate embedded WGSL strings that lived in `render/instancing.rs` are **removed**;
  `gpu_cull.wgsl` here is the single source for the compute cull.
- **Normal green-flip fix applied:** `instanced.wgsl` `perturb_normal` now negates sampled
  `n.y` when the material's DirectX-normal flag (`MF_NORMAL_DX`, set from
  `materials.json.normalGreenFlip` / `manifest.conventions.normalMapGreenFlip`) is set — it
  HONORS the declared convention instead of hardcoding a flip.

## Web module -> WGSL file

| WGSL file | Web source | What was ported |
|---|---|---|
| `instancing_m0.wgsl` | (new — the M0 CPU-fed first-pixel path) | Full 3x4 affine -> mat3+translation (NO decompose); cofactor normals; double-sided; flat lambert. Instance-rate vertex attrs. A/B fallback (`EFT_RENDER=m0`). |
| `gpu_draw.wgsl` | (new — the M2 GPU-driven draw, fork of `instancing_m0.wgsl`) | Same 3x4 affine + cofactor-normal math, but fetches the instance from the `instances` SSBO via `visible[instance_index]` (indirect draw); flat lambert. Default path (`EFT_RENDER=gpu`). |
| `instanced.wgsl` | (new — the GPU-driven surface path) + `_vpsplat.js`, `_gi.js` call sites | Per-instance FULL 3x4 affine -> mat3+translation (NO decompose); cofactor normal transform; bindless albedo/normal sample; `_MainTex*_Color`; cutout discard; SH-GI + one sun term; per-pixel cotangent-frame normal mapping (no tangent attribute in the eftpack vertex layout) |
| `sh_gi.wgsl` | `_gi.js` (`_fastE`/`_fullE`), `_gigl.js` (DC parity), `volume.json` layout | L1 SH irradiance sampled by WORLD POS from a REAL 3D texture; cosine-convolution reconstruction (A0=pi, A1=2pi/3); Chebyshev-visibility leak gate (DDGI moments), distance-gated |
| `splat.wgsl` | `_vpsplat.js` | EFT "Custom/Vert Paint SoftCutout Decal" 3-layer height splat: `w_i = pow(Heights*vColor, Blend)` normalized; near-black/zero-coverage fallbacks; ground-matte roughness (x0.30, floor 0.72); SoftCutout feather alpha |
| `grade.wgsl` | `_gradegl.js` | Exposure -> LUT shaper `sqrt(clamp(c/4,0,1))` -> 3D LUT -> PRISM vignette. Output is DISPLAY-ENCODED (Hejl baked into the LUT); tonemap OFF, non-sRGB target |
| `gpu_cull.wgsl` | `_cullgl.js` + `_lod.js` | **M2 rewrite:** per-instance world bounding sphere (CONSERVATIVE Frobenius-norm radius, CPU-precomputed, no decompose — NOT max-column, see rule 1); sphere-vs-6-plane frustum; per-mesh contiguous compaction via atomicAdd on the indirect `instance_count`, scattering `visible[]` indices -> DrawIndexedIndirect. Fused to 2 passes (`cs_reset`, `cs_cull`). The `_lod.js` screen-height gate is deferred (was the M1 3-pass stub). |

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
- **grade.wgsl group(0)** — `scene_tex` (linear HDR RT) + sampler, `lut3d` (64³) + sampler,
  `GradeParams`.

## M2 GPU-driven binding scaffold (Rust POD structs must match `gpu_cull.wgsl` + `gpu_draw.wgsl`)

The M2 buffers are built ONCE on the CPU (main world) then uploaded to the GPU exactly once in the
render world at `PrepareResources` (GPU-resident) grouped-by-mesh and contiguous; only the tiny
`CullGlobals` uniform is re-uploaded per frame. Live field names below match `render/gpu_driven.rs`.

- **`gpu_cull.wgsl` group(0)** — the compute cull/compaction layout (`ShaderStages::COMPUTE`,
  `BindGroupLayoutEntries::sequential`):
  - `@binding(0)` `var<uniform> G: CullGlobals` — `frustum: array<vec4<f32>,6>` normalized INWARD frustum
    planes (Gribb-Hartmann from `clip_from_world`, `gpu_driven::build_frustum_planes`) + `counts: vec4<u32>`
    (x=instance_count, y=mesh_count, z,w=pad). std140, 112 B. Re-uploaded per frame (`upload_frustum`).
  - `@binding(1)` `var<storage, read> instances: array<InstanceGpu>` — the global per-instance SSBO,
    80 B/record: `m0,m1,m2` (ROW-MAJOR 3x4 affine rows) + `ids` (vec4 u32: x=mesh_id, y=flags, z,w=pad) +
    `sphere` (vec4: xyz CPU world center, w CONSERVATIVE world radius = `r_local * ‖L‖_F` Frobenius norm).
    Built grouped-by-mesh + contiguous (iterate `manifest.meshes` in order, append `instances_by_mesh[mi]`).
    (Rust `InstanceGpuRecord` field order is m0,m1,m2,ids,sphere — WGSL struct matches.)
  - `@binding(2)` `var<storage, read> mesh_meta: array<MeshMeta>` — 32 B/mesh: `index_count, first_index,
    base_vertex(i32), instance_base, instance_count(region length)` + 3 pad. `instance_base` is the
    running cumulative count = the indirect `first_instance`.
  - `@binding(3)` `var<storage, read_write> visible: array<u32>` — length == total instances; each mesh
    compacts its survivors' indices into `[instance_base, instance_base+instance_count)`. usage `STORAGE | COPY_DST`.
  - `@binding(4)` `var<storage, read_write> indirect: array<DrawArgs>` — one 20 B entry/mesh;
    `instance_count` is `atomic<u32>` (scattered by `cs_cull`, zeroed by `cs_reset`); the other fields are
    REWRITTEN each frame by `cs_reset` from `mesh_meta` (first_instance = instance_base). usage
    `INDIRECT | STORAGE | COPY_DST`.
  - Entry points: `cs_reset` (dispatch ceil(mesh_count/64)) then `cs_cull` (dispatch ceil(instance_count/64)),
    both `@workgroup_size(64)`, SAME bind group, recorded as SEPARATE compute passes (wgpu barrier between).
    Optional `CULL_COMPUTE_SPHERE` shader-def switches to a GPU-side Frobenius radius (reinterprets
    `sphere` as the LOCAL center/radius and transforms it in-shader) if the CPU precompute is skipped.
- **`gpu_draw.wgsl` group(0)** — Bevy's mesh VIEW bind group (reused via `SetMeshViewBindGroup<0>`)
  so `position_world_to_clip` resolves. The live M2 draw command is `(SetItemPipeline,
  SetMeshViewBindGroup<0>, DrawGpuDrivenInner)` where `DrawGpuDrivenInner` sets group(1) + the shared
  vertex/index buffers and loops `draw_indexed_indirect` per mesh — do NOT include M0's
  `SetMeshViewBindingArrayBindGroup<1>` or `SetMeshBindGroup<2>`; group(1) is OUR storage.
- **`gpu_draw.wgsl` group(1)** — the M2 draw storage (`ShaderStages::VERTEX`), order MATCHES the Rust
  `ssbo_layout`: `@binding(0)` `var<storage, read> instances: array<InstanceGpu>` (SAME buffer as cull
  binding 1), `@binding(1)` `var<storage, read> visible: array<u32>` (SAME buffer as cull binding 3).
  Vertex attrs are the shared mesh vertex buffer only: position@0, normal@1, uv@2 (step_mode Vertex,
  stride 32). The draw is `draw_indexed_indirect(indirect_buf, mesh*20)` per mesh.
  Requires `Features::INDIRECT_FIRST_INSTANCE` (nonzero `first_instance`) — `init_gpu_pipelines` HARD-
  disables the GPU path (empty view + error, use `EFT_RENDER=m0`) if the adapter lacks it; auto-on with
  Bevy default (Functionality priority) on native Vulkan.

### Instance record layouts
- `instances.bin` (contract, 72 B): affine f32x12 + meshId u32 + lodGroup i32 + lodIndex i32 +
  rootId u32 + flags u32.
- **M2 GPU-resident `InstanceGpu` (80 B, `gpu_cull.wgsl` + `gpu_draw.wgsl`; Rust `InstanceGpuRecord`):**
  m0,m1,m2 (vec4 f32, ROW-MAJOR affine rows incl shear) + ids (vec4 u32: x=mesh_id, y=flags, z,w=pad) +
  sphere (vec4 f32: xyz CPU world center, w conservative Frobenius-scaled world radius). Built once,
  grouped-by-mesh + contiguous. This replaces the M1 `InstanceRaw`/`InstanceGpu` split: M2 does NOT copy
  compacted records — the cull scatters u32 INDICES into `visible[]` and the draw fetches
  `instances[visible[instance_index]]`.
- Compacted `InstanceGpu` (M3 `instanced.wgsl` instance attrs loc4..7): m0,m1,m2 + ids (mesh_id,
  material_id, flags, lod_packed) — the textured M3 path may revive an instance-rate copy for materials.

### Vertex layout (manifest, matches `instanced.wgsl` VertexIn loc0..3)
position f32x3 @0, normal f32x3 @12, uv f32x2 @24, color unorm8x4 @32 (stride 36; loader may pad
to 16-B for the storage-buffer path). UV tiling is BAKED into vertex uv by assemble_bevy;
`uv_xform` in `Material` is reference-only (do not double-apply — except the vert-paint per-layer
tiling `vp_til0..2`, which IS applied in-shader to the raw mesh uv).

## Correctness rules honored (tarkov-unity-extraction skill)

1. **NEVER TRS-decompose.** The affine is applied whole. Positions use `lin*p+T`; normals use the
   COFACTOR matrix `mat3(cross(c1,c2),cross(c2,c0),cross(c0,c1))` = det·inverse-transpose —
   shear/non-uniform correct without inverting. **Cull radius uses `r_local * ‖L‖_F`** — the Frobenius
   norm of the linear 3x3, `sqrt(|c0|²+|c1|²+|c2|²)`, computed ONCE on the CPU
   (`gpu_driven::conservative_radius_scale`) and stored in `InstanceGpu.sphere.w`. Frobenius is a
   GUARANTEED upper bound on the operator norm: `σ_max(L) <= ‖L‖_F <= sqrt(3)·σ_max(L)`, so the world
   sphere is never too small (correctness) and at most ~1.73x too large. Do NOT use
   `max(|col0|,|col1|,|col2|)` (max-column) — a LOWER bound on σ_max that UNDER-estimates under shear and
   wrongly culls visible geometry (Codex P2 #8). Do NOT use a finite power-iteration σ_max estimate
   either — it converges FROM BELOW and can seed orthogonal to the dominant eigenvector, so it also
   under-estimates (verify major finding). The GPU fallback (shader-def `CULL_COMPUTE_SPHERE`) uses the
   same Frobenius bound. (The old M1 `cull.wgsl` `max_col_scale` was the bug; removed in the M2 rewrite.)
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
2. **Mesh bounding spheres.** DONE for M2: `eftpack::Pack::bounding_spheres()` supplies per-mesh local
   center/radius, consumed once on the CPU in `gpu_driven::build_cpu_data` to precompute each instance's
   conservative world sphere. (No manifest change needed.)
3. **HZB occlusion.** Deferred for M2 (frustum-only). Wire a
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
