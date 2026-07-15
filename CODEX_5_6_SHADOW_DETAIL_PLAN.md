## #5 Dynamic Sun Shadows

### Approach & justification

Implement a near-field, sun-aligned contact CSM: 2 cascades covering approximately `0.5–15 m` and `15–80 m`, with a hard cap on how much baked diffuse energy it may remove.

This is preferable to a full-map four-cascade solution:

- Broad static sun shadows are already represented by the SH volume; the main shader explicitly treats it as the complete baked lighting solution at [gpu_draw.wgsl:423](/C:/Users/user/eft_native_viewer/viewer/assets/shaders/gpu_draw.wgsl:423).
- Four cascades to 500 m—the abandoned standard path’s configuration at [standard.rs:204](/C:/Users/user/eft_native_viewer/viewer/src/render/standard.rs:204)—would redraw the scene four times to reproduce lighting already baked into SH. It also maximizes double-darkening risk.
- SSAO/SSGI-lite attenuates ambient/indirect light indiscriminately, exactly the component that is already baked. It would darken interiors, add halos, and fight the existing `0.03` ambient floor.
- Screen-space sun ray marching would be directionally correct but view-dependent, miss off-screen casters, require camera depth before lighting, and force either a prepass or post-lighting composite.
- Two near cascades add the missing high-frequency/contact edge information, including off-screen casters within a bounded extrusion, without attempting to replace the baked broad shadows.

Recommended shipping defaults:

```text
cascade splits:        [0.5, 15.0, 80.0] metres
cascade count:         2
resolution:            2048² per cascade
depth format:          Depth32Float
cascade overlap:       10%
PCF:                   weighted 3×3 tent
far contact fade:      65–80 m
diffuse removal cap:   0.12 initially, never more than 0.15
specular shadowing:    full, but only when SH dominant light aligns with sun
```

A single-cascade fallback should cover `0.5–25 m` at 2048². Do not use one map for all 1.2 km: even 4096² gives roughly 0.29 m/texel before fitting margins, which cannot supply crisp contact shadows.

### Render-graph & buffers

The current graph only installs `EftCullLabel -> Node3d::StartMainPass` at [gpu_driven.rs:545](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:545). Also, `DrawGpuDriven` is a render command inside `Transparent3d`, not a render-graph node; see [gpu_driven.rs:1784](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:1784) and [gpu_driven.rs:2103](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:2103).

Add:

```rust
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct EftShadowLabel;

struct EftShadowNode;
```

Replace the existing graph edge with the exact chain:

```rust
.add_render_graph_node::<EftCullNode>(Core3d, EftCullLabel)
.add_render_graph_node::<EftShadowNode>(Core3d, EftShadowLabel)
.add_render_graph_edges(
    Core3d,
    (EftCullLabel, EftShadowLabel, Node3d::StartMainPass),
);
```

Both nodes should return immediately unless `graph.view_entity()` has the extracted `CullCamera`. This prevents duplicate atlas clears/draws if another `Core3d` view exists.

#### Visibility/indirect reuse

The existing allocations are sufficient:

- `visible`: one `u32` per instance.
- `indirect`: one 20-byte `DrawIndexedIndirectArgs` per mesh.
- `EftDrawBindGroup`: `instances + visible`.
- `EftGpuBuffers`: vertex/index/indirect handles.

They are created at [gpu_driven.rs:1303](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:1303) and bound for the main multidraw at [gpu_driven.rs:2138](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:2138).

`EftShadowNode` must only read them:

```text
EftCullNode:
    writes visible + indirect

EftShadowNode:
    binds the same EftDrawBindGroup
    binds the same vertex/index buffers
    multi_draw_indexed_indirect(same indirect buffer)

StartMainPass / DrawGpuDriven:
    reads the same buffers unchanged
```

This preserves the existing indirect layout and `INDIRECT_FIRST_INSTANCE` behavior.

Camera-only culling would miss a caster just outside the view. Conservatively extrude the camera frustum toward the sun before uploading it:

```rust
// Lsun points toward the sun; max_cast_length = 80 m.
plane.w += max_cast_length * (-plane.xyz().dot(Lsun)).max(0.0);
```

This follows from a possible caster position `caster = receiver + Lsun * t`. It expands only planes from which a caster could project a shadow into the camera frustum. It makes the main pass process some extra off-screen instances, but does not change its image.

If that expansion measurably harms main-pass vertex cost, the second-stage optimization is a separate `shadow_visible`/`shadow_indirect` stream using the same `MeshMeta`, compute shader, and compaction scheme. Do not overwrite the camera stream between the shadow and main passes.

#### New resources

Add proposed resources:

```rust
struct ShadowCascadeUniform {
    view_proj: [[f32; 4]; 4], // column-major Mat4 upload
    dir_texel: [f32; 4],      // xyz=Lsun, w=1/2048
}

struct SunShadowUniform {
    view_proj: [[[f32; 4]; 4]; 2],
    split_depths: [f32; 4],   // 15, 80, overlap=0.10, enabled
    sun_dir_texel: [f32; 4],  // xyz=Lsun, w=1/2048
    texel_world: [f32; 4],    // cascade world texel sizes, normal/depth bias params
    combine: [f32; 4],        // diffuse cap=0.12, fade start=65, end=80, debug mode
}

struct EftShadowPipeline {
    pipeline_id: CachedRenderPipelineId,
    cascade_layout: BindGroupLayout,
}

struct EftShadowResources {
    depth_texture: Texture,
    array_view: TextureView,
    layer_views: [TextureView; 2],
    cascade_uniforms: [Buffer; 2],
    cascade_bind_groups: [BindGroup; 2],
    main_uniform: Buffer,
    comparison_sampler: Sampler,
}
```

Allocate a two-layer texture rather than a side-by-side atlas:

```text
dimension:                  D2
size:                       2048 × 2048 × 2 layers
format:                     Depth32Float
usage:                      RENDER_ATTACHMENT | TEXTURE_BINDING
mips/samples:               1 / 1
memory:                     32 MiB
sampling view dimension:    D2Array
rendering views:            one D2 view per layer
```

This is an atlas allocation logically, but avoids tile gutters and PCF bleeding.

Do not add a fifth main-pipeline bind group. The existing pipeline already uses groups 0–3 at [gpu_driven.rs:1946](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:1946). Extend the existing SH/lighting group 3:

```text
group(3) binding(0): ShVolume uniform
group(3) binding(1): sh_r
group(3) binding(2): sh_g
group(3) binding(3): sh_b
group(3) binding(4): sh_samp
group(3) binding(5): SunShadowUniform
group(3) binding(6): texture_depth_2d_array
group(3) binding(7): sampler_comparison
```

Create a cleared/dummy depth array and set `enabled=0` when shadows or `sun_dir` are unavailable. That keeps both main specializations’ layouts stable.

#### Shadow pipeline

The scaffold currently contains only position input, `instances/visible`, and a single `SunUniform` at [gpu_shadow.wgsl:7](/C:/Users/user/eft_native_viewer/viewer/assets/shaders/gpu_shadow.wgsl:7).

A fragment stage is technically unnecessary for solid opaque depth, but it is required for this scene:

- Grass is appended as 109k alpha-cutout cross quads at [gpu_driven.rs:942](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:942).
- Without discard, grass and foliage cast rectangular card shadows.
- BLEND decals/water/glass must not cast their full geometry.

Extend the shadow vertex output with `uv` and flat `material_index`. Use vertex attributes:

```text
location 0: Float32x3, offset 0
location 2: Float32x2, offset 24
location 3: Uint32,    offset 32
stride:     52
```

Add a minimal fragment entry with no color output:

```wgsl
@fragment
fn fragment(o: ShadowVOut) {
    let m = materials[o.material_index];

    let duv_dx = dpdx(o.uv);
    let duv_dy = dpdy(o.uv);

    if ((m.flags & MAT_FLAG_BLEND) != 0u) {
        discard;
    }

    if ((m.flags & MAT_FLAG_CUTOUT) != 0u) {
        let idx = select(0u, m.albedo_index, m.albedo_index != MAT_ALBEDO_NONE);
        let a = textureSampleGrad(
            albedo_tex[idx], albedo_samp, o.uv, duv_dx, duv_dy
        ).a * m.tint.a;
        if (a < m.alpha_cutoff) {
            discard;
        }
    }
}
```

The shadow pipeline layout is:

```text
group 0: existing ssbo_layout / EftDrawBindGroup
group 1: ShadowCascadeUniform
group 2: existing material_layout / EftMaterialBindGroup
```

Pipeline state:

```text
fragment targets:       empty
depth format:           Depth32Float
depth write:            true
depth compare:          LessEqual
clear depth:            1.0
cull mode:              None
multisample count:      1
initial raster bias:    constant=2, slope_scale=2.0, clamp=0
```

`EftShadowNode` loops over the two layer views, clears each to `1.0`, sets groups 0–2, and calls the same `multi_draw_indexed_indirect`.

### WGSL & math

#### Sun sourcing

`VolumeMeta` currently parses only bounds/dims/spacing/coefficients at [gpu_driven.rs:250](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:250). Add optional `sun_dir`, validate it, and convert once:

```rust
let Lsun = Vec3::new(-raw[0], raw[1], raw[2]).normalize();
```

`Lsun` points toward the sun. Light travels along `-Lsun`. This matches the existing standard-path conversion at [standard.rs:175](/C:/Users/user/eft_native_viewer/viewer/src/render/standard.rs:175).

If missing or degenerate, disable the feature; do not invent a fallback direction.

#### Cascade fitting

Use the confirmed perspective camera and finite far plane at [main.rs:216](/C:/Users/user/eft_native_viewer/viewer/src/main.rs:216).

For two cascades, practical/log splits with `λ=0.75` give:

```text
split(i) = λ * near*(far/near)^(i/N)
         + (1-λ) * (near + (far-near)*i/N)

near=0.5, far=80, N=2
→ approximately [0.5, 14.8, 80]
→ round to [0.5, 15, 80]
```

For each slice:

1. Reconstruct its eight world-space corners from `clip_from_view` and `world_from_view`.
2. Compute the corner centroid and a square XY radius in light space.
3. Use a stable up axis: `Y` unless `abs(dot(Lsun,Y)) > 0.99`, then `Z`.
4. Build the light view from `eye = center + Lsun * eye_distance`, looking at `center`.
5. Include the receiver corners, `corner + Lsun*80 m` caster extrusion, and `corner - Lsun*10 m` receiver margin in the light-space Z fit.
6. Snap the light-space XY center:

```text
world_texel = (2 * cascade_radius) / 2048
snapped_x = round(center_x / world_texel) * world_texel
snapped_y = round(center_y / world_texel) * world_texel
```

7. Build a conventional 0–1 depth orthographic projection and use `LessEqual`; do not reuse the main pass’s reverse-Z `GreaterEqual`.

#### PCF

```wgsl
fn sample_cascade(p: vec3<f32>, Ng: vec3<f32>, c: u32) -> f32 {
    let world_texel = sun.texel_world[c];

    // Geometric normal only: normal maps must not wobble the receiver offset.
    let offset_p =
        p
        + Ng * (1.5 * world_texel)
        + sun.sun_dir_texel.xyz * (0.25 * world_texel);

    let q = sun.view_proj[c] * vec4<f32>(offset_p, 1.0);
    let ndc = q.xyz / q.w;

    if (any(ndc.xy < vec2(-1.0)) || any(ndc.xy > vec2(1.0))
        || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let dt = sun.sun_dir_texel.w;

    var sum = 0.0;
    // Tent weights [1,2,1]², total 16.
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let wx = select(1.0, 2.0, x == 0);
            let wy = select(1.0, 2.0, y == 0);
            sum += wx * wy * textureSampleCompareLevel(
                shadow_map,
                shadow_cmp,
                uv + vec2<f32>(f32(x), f32(y)) * dt,
                i32(c),
                ndc.z
            );
        }
    }
    return sum / 16.0;
}
```

Select using view-space depth:

```wgsl
let view_depth = -(view.view_from_world * vec4(o.world_pos, 1.0)).z;
```

Blend from cascade 0 to 1 over `13.5–15 m`, then fade the entire effect to fully lit over `65–80 m`.

#### Double-darkening-safe combination

Extend `DomLight` at [gpu_draw.wgsl:218](/C:/Users/user/eft_native_viewer/viewer/assets/shaders/gpu_draw.wgsl:218) with a normalized directionality value:

```wgsl
let l0 = max(dot(vec3(cr.x, cg.x, cb.x), lw), 1e-4);
let directionality = clamp(dmag / (1.73205 * l0), 0.0, 1.0);
```

For an ideal directional source, the L1/L0 ratio approaches `sqrt(3)`; diffuse/isotropic lighting approaches zero.

Construct a baked-lit gate:

```wgsl
let align = dot(dom.dir, sun.sun_dir_texel.xyz);
let NdotSun = dot(N, sun.sun_dir_texel.xyz);

let sun_lit_gate =
      smoothstep(0.10, 0.35, dom.directionality)
    * smoothstep(0.75, 0.95, align)
    * smoothstep(0.05, 0.35, NdotSun);

let contact_fade = 1.0 - smoothstep(65.0, 80.0, view_depth);
let shadow_event =
    sun_lit_gate * contact_fade * (1.0 - shadow_visibility);
```

Then preserve the baked floor and remove only a bounded fraction of lighting above it:

```wgsl
let ambient_floor = vec3<f32>(0.03);
let gi_baked = max(sh_irradiance(o.world_pos, N), ambient_floor);

let removable = max(gi_baked - ambient_floor, vec3<f32>(0.0));
let gi_shadowed =
    gi_baked - removable * (0.12 * shadow_event);

let lit =
    albedo.rgb * gi_shadowed * sh.vol_min.w;

// The GGX lobe is the only real-time directional-looking term.
// Full removal is safe because sun_lit_gate includes SH/sun alignment.
spec_rgb *= 1.0 - shadow_event;
```

Properties:

- `gi_shadowed >= ambient_floor` component-wise.
- A location whose SH already says “baked shadow” has low directionality/alignment and receives little or no further attenuation.
- Diffuse contact attenuation cannot exceed 12%.
- Set the `0.12` coefficient to `0.0` for a specular-only diagnostic mode.

### Step-by-step

1. Parse and log `sun_dir`; add the X-flip and a runtime `enabled` switch. No shader change.
2. Allocate the two-layer depth texture and clear it to `1.0`; bind it in the expanded group 3 with `enabled=0`.
3. Queue the opaque depth pipeline and wire `EftCull -> EftShadow -> StartMainPass`. Initially visualize the atlas/debug cascades without sampling it in lighting.
4. Add the alpha-aware shadow fragment and compare grass/foliage silhouettes against the vertex-only variant.
5. Reuse `visible`/`indirect`, then add the 80 m directional frustum extrusion and measure added main-pass instance/vertex count.
6. Sample cascade 0 only, specular-only combination.
7. Enable the 12% bounded diffuse formula and A/B `sun_lit_gate`, `shadow_event`, and final luminance.
8. Add cascade 1, 10% seam blending, 65–80 m fade, and texel snapping.
9. Only if performance demands it: cache the static shadow render while the snapped matrices are unchanged, or add a separate shadow cull stream.

### Regression risks

- **Double-darkening:** preserve the `0.03` floor, cap diffuse removal at 12–15%, and gate by SH directionality plus sun alignment. Validate that already-dark SH regions change by less than 1%.
- **Acne/peter-panning:** use geometric normals for bias, begin at `1.5` world texels normal bias plus `0.25` texel toward-sun bias, and tune alongside raster `2/2` bias. Expose a debug mode showing raw compare results.
- **Cascade seams/shimmer:** square stable fits, light-space texel snapping, 10% overlap, and far fade.
- **Cutout card shadows:** retain the minimal shadow fragment; do not ship the position-only scaffold for grass.
- **Off-screen casters:** apply the directional frustum extrusion. Compare against a no-cull shadow capture at several camera positions.
- **Second full-scene pass cost:** two multidraws, not per-mesh CPU submission. Offer a single 25 m cascade fallback and measure the expansion overhead separately.
- **Indirect-buffer breakage:** shadow pass is read-only; never reset or recull the shared stream between `EftCullNode` and `StartMainPass`.

---

## #6 Detail Maps

### Approach & justification

Extract and use the actual game detail textures, with the web viewer’s name-keyed mapping as the authoritative eligibility list. Do not synthesize procedural noise as the default.

Why:

- Actual detail maps preserve material identity and the measured contrast gains recorded in issue #6 at [RENDERING_ISSUES.md:122](/C:/Users/user/eft_native_viewer/RENDERING_ISSUES.md:122).
- Procedural detail would put similar noise on unrelated rock, concrete, metal, and floor surfaces and risks becoming more visible than the underlying art.
- The scene and materials are static, so extraction and preprocessing are one-time costs.
- Procedural generation remains useful only as an offline fallback when an authored source is genuinely missing; the runtime fallback should be neutral, not invented.

There is a confirmed pipeline gap:

- Material signatures already preserve `detA`, `detN`, UVs and strengths at [matsig.py:25](/C:/Users/user/eft_native_viewer/eft_pipeline/tarkmap_core/matsig.py:25).
- `assemble_bevy.py` currently emits normal/spec fields but drops all detail fields at [assemble_bevy.py:228](/C:/Users/user/eft_native_viewer/eft_pipeline/assemble_bevy.py:228).
- Rust `Material` likewise ends without detail fields at [eftpack.rs:229](/C:/Users/user/eft_native_viewer/viewer/src/eftpack.rs:229).

The first work item is therefore data propagation, not shader invention.

### Render-graph & buffers

No render-graph node is required. Detail sampling stays in `gpu_draw.wgsl` before SH/GGX evaluation.

#### Pack-side data

Emit a versioned `detailMaps` sidecar or equivalent material fields:

```json
{
  "key": "normalized_material_or_albedo_name",
  "albedo": "..._detail_albedo.ktx2",
  "normal": "..._detail_normal.ktx2",
  "albedoUv": [sx, sy, ox, oy],
  "normalUv": [sx, sy, ox, oy],
  "albedoStrength": 1.0,
  "normalStrength": 1.0,
  "meanGain": [1.0, 1.0, 1.0],
  "allowOnTiled": false
}
```

Matching policy:

1. Exact source texture GUID/path if present.
2. Exact normalized material name.
3. Exact normalized base-albedo basename.
4. No substring/wildcard fallback.

Only an explicit match receives `MAT_FLAG_DETAIL`. Hard-exclude:

- `MAT_FLAG_TERRAIN`.
- Water, glass, decals, SoftCutout roads.
- Grass/cutout foliage.
- Materials considered already tiled unless their mapping explicitly sets `allowOnTiled=true`.

#### GPU material layout

The current `GpuMaterial` is 80 bytes at [gpu_driven.rs:144](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:144). Extend it to 160 bytes:

```text
offset  80: detail_albedo_index u32
offset  84: detail_normal_index u32
offset  88: detail_flags u32
offset  92: pad u32

offset  96: detail_albedo_uv vec4
offset 112: detail_normal_uv vec4

offset 128: detail_params vec4
            x = albedo strength
            y = normal strength
            z = fade start, default 8
            w = fade end, default 15

offset 144: detail_mean_gain vec4
            xyz = mean(sample_linear.rgb * 4.5948), offline
```

Add:

```text
Material.flags bit5: MAT_FLAG_DETAIL
detail_flags bit0:   has detail albedo
detail_flags bit1:   has detail normal
detail_flags bit2:   detail normal green flip
0xFFFFFFFF:          missing texture sentinel
```

#### Bindings and resources

Extend group 2, whose present bindings are defined at [gpu_driven.rs:1432](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:1432):

```text
binding 0: material SSBO
binding 1: base albedo binding_array
binding 2: base sampler
binding 3: base normal binding_array
binding 4: terrain splat SSBO
binding 5: detail albedo binding_array
binding 6: detail normal binding_array
binding 7: detail sampler
```

Add `detail_albedo_paths` and `detail_normal_paths` to `CpuData`, and retain their textures/views in `EftMaterialResources`.

Detail sampler:

```text
address U/V/W:       Repeat
mag/min/mip filter:  Linear
anisotropy:          8
```

Use GPU-ready KTX2 with complete mip chains:

- Detail albedo: BC7 sRGB.
- Detail normal: BC5 linear, reconstruct Z in WGSL.
- Neutral fallbacks: 0.5-sRGB albedo and `(0.5,0.5,1)` normal.

Bevy’s enabled default features can parse KTX2, but Basis ETC1S/UASTC transcoding is not enabled by default. Prefer BCn KTX2 output or explicitly enable/test `basis-universal`.

#### UV transformation

The pack stores base `_MainTex_ST` baked into vertex UVs and flips V afterward at [assemble_bevy.py:422](/C:/Users/user/eft_native_viewer/eft_pipeline/assemble_bevy.py:422). Do not apply detail ST directly to `o.uv`.

Precompute a transform from the baked base UV to the baked detail UV.

For base ST `(bsx,bsy,box,boy)` and detail ST `(dsx,dsy,dox,doy)`:

```text
rx = dsx / bsx
ry = dsy / bsy

relative_scale  = (rx, ry)
relative_offset = (
    dox - rx*box,
    1 - doy - ry*(1 - boy)
)

detail_uv = baked_base_uv * relative_scale + relative_offset
```

Reject or neutralize materials whose base scale is effectively zero.

### WGSL & math

#### Distance fade

```wgsl
let detail_distance = distance(view.world_position.xyz, o.world_pos);
let detail_fade = 1.0 - smoothstep(
    m.detail_params.z, // 8 m
    m.detail_params.w, // 15 m
    detail_distance
);
```

#### Detail albedo

Sample the KTX2 albedo through an sRGB view, so WGSL receives linear values:

```wgsl
let detail_sample = textureSampleGrad(
    detail_albedo_tex[m.detail_albedo_index],
    detail_samp,
    detail_uv,
    duv_dx * m.detail_albedo_uv.xy,
    duv_dy * m.detail_albedo_uv.xy
).rgb;

// Unity Standard ×2 in linear space.
let unity_gain = detail_sample * 4.5948;

// Offline mean is also computed after the 4.5948 factor.
let neutral_gain =
    unity_gain / max(m.detail_mean_gain.xyz, vec3<f32>(1e-3));

let detail_weight =
    clamp(m.detail_params.x * detail_fade, 0.0, 1.0);

albedo.rgb *= mix(
    vec3<f32>(1.0),
    clamp(neutral_gain, vec3<f32>(0.25), vec3<f32>(4.0)),
    detail_weight
);
```

Apply this after ordinary albedo construction but before lighting. The terrain branch at [gpu_draw.wgsl:391](/C:/Users/user/eft_native_viewer/viewer/assets/shaders/gpu_draw.wgsl:391) must never enter it.

Never modify `albedo.a`. Texture alpha remains smoothness/data except in the existing explicitly classified coverage paths, per [RENDERING_ISSUES.md:136](/C:/Users/user/eft_native_viewer/RENDERING_ISSUES.md:136).

#### Detail normal

Refactor the existing normal block at [gpu_draw.wgsl:356](/C:/Users/user/eft_native_viewer/viewer/assets/shaders/gpu_draw.wgsl:356) so base and detail normals are combined in tangent space, then transformed to world space once.

RNM:

```wgsl
fn blend_rnm(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    let t = base + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * (dot(t, u) / max(t.z, 1e-4)) - u);
}
```

Decode and fade the detail normal:

```wgsl
var dxy = detail_normal_sample.xy * 2.0 - vec2<f32>(1.0);
if ((m.detail_flags & DETAIL_NORMAL_GREEN_FLIP) != 0u) {
    dxy.y = -dxy.y;
}

dxy *= m.detail_params.y * detail_fade;

// Keep a valid hemisphere after authored strength.
let d2 = dot(dxy, dxy);
if (d2 > 0.99) {
    dxy *= sqrt(0.99 / d2);
}
let detail_ts = vec3<f32>(
    dxy,
    sqrt(max(1.0 - dot(dxy, dxy), 1e-4))
);

let combined_ts = blend_rnm(base_ts, detail_ts);
```

Then call the derivative/TBN transform exactly once:

```wgsl
let has_any_normal = has_base_normal || has_detail_normal;
let N_mapped = perturb_normal(
    N_geometric,
    o.world_pos,
    o.uv,
    combined_ts
);
N = select(N_geometric, N_mapped, has_any_normal);
```

For best control-flow safety, compute the cotangent frame/derivatives before any discard, then use `textureSampleGrad` for optional detail samples. Do not call the derivative-based `perturb_normal` separately for base and detail.

### Step-by-step

1. Recover the authoritative web-viewer mapping and actual game detail sources. Emit an inventory with exact matches, unmatched entries, duplicate normalized names, UVs, intensities, and means.
2. Extend the pack sidecar/material schema and Rust `Material`; verify materials still deserialize when detail fields are absent.
3. Extend `GpuMaterial` and CPU dedup tables. Log eligible/excluded/terrain/tiled counts; shader remains disabled.
4. Add KTX2 loading, full mip retention, dedicated binding arrays, neutral placeholders, sampler, and runtime binding-limit checks.
5. Implement albedo detail only. A/B average luminance over representative rock/floor patches; require under 1% mean shift.
6. Add the 8–15 m fade and verify no visible transition band.
7. Refactor normal handling to one cotangent frame and RNM; enable detail normals separately.
8. Add explicit-gradient sampling and anisotropy. If shimmer remains, multiply detail gradients by `1.414` for a `+0.5` mip bias.
9. Enable only the exact allowlist. MicroSplat remains excluded until detail is deliberately integrated per terrain layer inside its 12-layer loop.

### Regression risks

- **Brightness/color shift:** store per-channel mean after the `4.5948` conversion, normalize to mean gain 1, and compare average linear luminance before/after.
- **Aliasing/shimmer:** full mip chains, explicit gradients, anisotropy 8, 8–15 m fade, and optional `+0.5` detail LOD bias.
- **Fighting MicroSplat:** hard-disable `MAT_FLAG_DETAIL` whenever `MAT_FLAG_TERRAIN` is set. A future terrain implementation belongs per layer, not after the splat blend.
- **Already-tiled materials:** exact allowlist only; default-exclude tiled surfaces unless the mapping explicitly overrides.
- **Alpha/smoothness corruption:** sample detail `.rgb` only. Never multiply base alpha or derive opacity from detail alpha.
- **Normal double-application:** RNM in tangent space, one TBN transform, one final normalized world normal.
- **Ground gloss regression:** do not touch roughness/specular. The existing terrain matte override at [gpu_driven.rs:785](/C:/Users/user/eft_native_viewer/viewer/src/render/gpu_driven.rs:785) remains authoritative.
- **Bindless/device limits:** check total sampled-texture counts and adapter limits before building group 2; disable detail alone if unsupported rather than disabling the GPU path.

### Must-verify against code

- **FLAG:** The actual `volume.json` is not present in this workspace. Verify its exact `sun_dir` value, units, and whether its SH coefficient directions require the same X-flip as the metadata vector.
- **FLAG:** Verify the two proposed 80 m caster extrusions against actual building heights and the sun’s elevation; near-horizon light may need a larger caster range.
- **FLAG:** The actual web-viewer detail mapping/KTX2 files are not in this repository. Confirm keys, `detAI`/`detNS` ranges, channel encoding, green convention, and whether KTX2 is BCn or Basis-supercompressed.
- **FLAG:** Confirm detail UV fields use Unity `_DetailAlbedoMap_ST`/`_DetailNormalMap_ST` before adopting the relative-transform formula.
- **FLAG:** Query target adapters’ `max_sampled_textures_per_shader_stage` and BC compression support; the repository does not currently log these limits.
- **FLAG:** Validate the proposed SH directionality thresholds (`0.10–0.35`) and diffuse cap (`0.12`) against captured lit, baked-shadowed, interior, rock, and floor samples before making them defaults.
