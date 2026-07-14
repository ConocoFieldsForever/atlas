// eft::instanced — main GPU-driven instanced surface shader for the native EFT viewer.
// Consumes the .eftpack v1 instance stream (compacted by cull.wgsl) + the interleaved mesh
// vertex layout (position f32x3 @0, normal f32x3 @12, uv f32x2 @24, color unorm8x4 @32).
//
// THE #1 RULE (tarkov-unity-extraction skill): the per-instance transform is the FULL 3x4
// affine INCLUDING SHEAR, applied to RAW verts. We NEVER TRS-decompose. The affine is
// row-major world 3x4 (12 f32) exactly as apply_global(m)[:12] wrote it into instances.bin.
//   * Position:  world = lin3x3 * pos + translation.
//   * Normal:    transformed by the COFACTOR matrix (cross-products of the linear columns) =
//                det * inverse-transpose — shear/non-uniform-scale correct WITHOUT inverting
//                or decomposing. Mirror (det<0) is handled by front-face state (flag bit), so
//                we normalize and let winding own the sign.
//
// Imports the SH-GI volume sample (eft::sh_gi, group 1) and the vert-paint splat math
// (eft::splat, binding-free). Outputs LINEAR HDR — tonemap/grade happens in grade.wgsl.

#import eft::sh_gi::{sample_sh_irradiance, gi}
#import eft::splat as splat

// -------------------------------------------------------------------------------------------
// group(0): global view / frame uniforms (shared with grade.wgsl / cull.wgsl intent).
// -------------------------------------------------------------------------------------------
struct View {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    exposure: f32,             // used by grade.wgsl; here for a cheap fallback if grade off
    sun_dir: vec3<f32>,        // world-space, normalized, points TOWARD the sun
    time: f32,
    sun_color: vec3<f32>,
    gi_str: f32,               // giStr (~0.3183 = 1/pi); mirrors GiParams for convenience
};
@group(0) @binding(0) var<uniform> view: View;

// -------------------------------------------------------------------------------------------
// group(2): bindless material textures + material table.
// -------------------------------------------------------------------------------------------
const NO_TEX: u32 = 0xffffffffu;

// Material flags (bit field). Role is the low nibble; behavior bits above.
const MF_ROLE_MASK: u32   = 0x7u;
const MF_ROLE_OPAQUE: u32 = 0u;
const MF_ROLE_CUTOUT: u32 = 1u;
const MF_ROLE_GLASS: u32  = 2u;
const MF_ROLE_DECAL: u32  = 3u;
const MF_ROLE_WATER: u32  = 4u;
const MF_TWO_SIDED: u32    = 0x08u;
const MF_VERT_PAINT: u32   = 0x10u;
const MF_VP_SOFTCUT: u32    = 0x20u;
const MF_EMISSIVE: u32      = 0x40u;
const MF_HAS_NORMAL: u32    = 0x80u;
// normal map is DirectX-convention (green points down) -> negate sampled n.y.
// Set from materials.json.normalGreenFlip / manifest.conventions.normalMapGreenFlip
// so the shader HONORS the declared convention instead of hardcoding a flip.
const MF_NORMAL_DX: u32     = 0x100u;

struct Material {
    tint: vec4<f32>,          // linear rgb + alpha (coverage / glass); _col4(_Color)
    uv_xform: vec4<f32>,      // sx,sy,ox,oy — ALREADY baked into vertex uv; kept for reference
    emissive: vec4<f32>,      // linear rgb * gain, a unused
    albedo_idx: u32,
    normal_idx: u32,
    emissive_idx: u32,
    flags: u32,
    roughness: f32,
    metallic: f32,
    alpha_cutoff: f32,
    _pad0: f32,
    // vert-paint block (valid only when MF_VERT_PAINT set):
    vp_l0_idx: u32, vp_l1_idx: u32, vp_l2_idx: u32, vp_heights_idx: u32,
    vp_til0: vec4<f32>, vp_til1: vec4<f32>, vp_til2: vec4<f32>,
    vp_col0: vec4<f32>, vp_col1: vec4<f32>, vp_col2: vec4<f32>,
    vp_params: vec4<f32>,     // blend, aStr, aCut, aHgt
};

@group(2) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(2) @binding(1) var tex_samp: sampler;
@group(2) @binding(2) var<storage, read> materials: array<Material>;

// -------------------------------------------------------------------------------------------
// Vertex / instance IO. Instance attributes come from the CULL-COMPACTED instance buffer
// (one instance-rate vertex buffer). @location 0..3 = mesh vertex; 4..7 = instance record.
// -------------------------------------------------------------------------------------------
struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,      // unorm8x4 -> [0,1]; vert-paint COLOR_0 weights + coverage
    // per-instance (step mode: Instance):
    @location(4) m0: vec4<f32>,         // affine row0 [m00 m01 m02 m03]
    @location(5) m1: vec4<f32>,         // affine row1
    @location(6) m2: vec4<f32>,         // affine row2
    @location(7) ids: vec4<u32>,        // x=mesh_id y=material_id z=flags w=(lodGroup<<16|lodIndex)
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) material_id: u32,
    @location(5) @interpolate(flat) inst_flags: u32,
};

// cofactor(linear 3x3) = det * inverse-transpose. Columns = cross products of the linear
// columns. Correct normal transform under shear/non-uniform scale WITHOUT decomposing.
fn cofactor(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(cross(c1, c2), cross(c2, c0), cross(c0, c1));
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    // Rebuild the linear 3x3 (columns) + translation from the ROW-MAJOR 3x4 affine.
    let col0 = vec3<f32>(in.m0.x, in.m1.x, in.m2.x);
    let col1 = vec3<f32>(in.m0.y, in.m1.y, in.m2.y);
    let col2 = vec3<f32>(in.m0.z, in.m1.z, in.m2.z);
    let translation = vec3<f32>(in.m0.w, in.m1.w, in.m2.w);
    let lin = mat3x3<f32>(col0, col1, col2);

    let world_pos = lin * in.position + translation;
    let nrm_mat = cofactor(col0, col1, col2);

    var out: VertexOut;
    out.clip = view.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_nrm = normalize(nrm_mat * in.normal);
    out.uv = in.uv;                         // UV tiling already baked in by assemble_bevy
    out.color = in.color;
    out.material_id = in.ids.y;
    out.inst_flags = in.ids.z;
    return out;
}

// --- bindless helpers ---
fn sample_tex(idx: u32, uv: vec2<f32>) -> vec4<f32> {
    return textureSample(textures[idx], tex_samp, uv);
}

// Reconstruct a world-space normal from a BC5 (RG) tangent-space normal map using a
// per-pixel cotangent frame (we carry no tangent attribute in the eftpack vertex layout).
fn perturb_normal(N: vec3<f32>, world_pos: vec3<f32>, uv: vec2<f32>, nidx: u32, dx_flip: bool) -> vec3<f32> {
    var rg = sample_tex(nidx, uv).rg * 2.0 - 1.0;
    // DirectX-convention (green down): negate G so tangent-space +Y is up (OpenGL).
    // Honors the manifest/material convention rather than hardcoding either way.
    if (dx_flip) { rg.y = -rg.y; }
    let nz = sqrt(max(1.0 - dot(rg, rg), 0.0));
    let map_n = vec3<f32>(rg, nz);

    // Cotangent frame (Mikkelsen) from screen-space derivatives.
    let dp1 = dpdx(world_pos); let dp2 = dpdy(world_pos);
    let duv1 = dpdx(uv);       let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, N);
    let dp1perp = cross(N, dp1);
    let T = dp2perp * duv1.x + dp1perp * duv2.x;
    let B = dp2perp * duv1.y + dp1perp * duv2.y;
    let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
    let TBN = mat3x3<f32>(T * invmax, B * invmax, N);
    return normalize(TBN * map_n);
}

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let mat = materials[in.material_id];
    let role = mat.flags & MF_ROLE_MASK;

    // --- base albedo + alpha ---
    var albedo = mat.tint.rgb;
    var alpha = mat.tint.a;
    var roughness = mat.roughness;

    if ((mat.flags & MF_VERT_PAINT) != 0u) {
        // EFT 3-layer vert-paint height splat (eft::splat).
        // Heights control mask sampled ONCE at the RAW mesh uv (R/G/B = layer weights).
        let heights = sample_tex(mat.vp_heights_idx, in.uv).rgb;
        let u0 = in.uv * mat.vp_til0.xy + mat.vp_til0.zw;
        let u1 = in.uv * mat.vp_til1.xy + mat.vp_til1.zw;
        let u2 = in.uv * mat.vp_til2.xy + mat.vp_til2.zw;
        let t0 = sample_tex(mat.vp_l0_idx, u0);
        let t1 = sample_tex(mat.vp_l1_idx, u1);
        let t2 = sample_tex(mat.vp_l2_idx, u2);
        // Bevy imports layer albedo as sRGB -> already LINEAR; just apply per-layer tint.
        let r = splat::vp_splat(
            heights, in.color.rgb, mat.vp_params.x,
            t0.rgb * mat.vp_col0.rgb, t1.rgb * mat.vp_col1.rgb, t2.rgb * mat.vp_col2.rgb,
            t0.a, t1.a, t2.a,
        );
        albedo = r.albedo;
        roughness = r.roughness;
        if ((mat.flags & MF_VP_SOFTCUT) != 0u) {
            alpha = splat::vp_softcutout_alpha(in.color.a, mat.vp_params.y, mat.vp_params.z, mat.vp_params.w);
        }
    } else if (mat.albedo_idx != NO_TEX) {
        // EFT albedo = _MainTex * _Color. Bevy imports color textures sRGB -> linear sample.
        let tex = sample_tex(mat.albedo_idx, in.uv);
        albedo = tex.rgb * mat.tint.rgb;
        alpha = tex.a * mat.tint.a;
    }

    // --- cutout discard ---
    if (role == MF_ROLE_CUTOUT && alpha < mat.alpha_cutoff) {
        discard;
    }

    // --- normal ---
    var N = normalize(in.world_nrm);
    // Double-sided default for opaque (EFT deferred draws building shells solid both sides);
    // flip the normal on back faces unless the material is single-sided (no MF_TWO_SIDED here
    // means single-sided shell -> keep as authored).
    if ((mat.flags & MF_TWO_SIDED) != 0u && !front) {
        N = -N;
    }
    if ((mat.flags & MF_HAS_NORMAL) != 0u && mat.normal_idx != NO_TEX) {
        N = perturb_normal(N, in.world_pos, in.uv, mat.normal_idx, (mat.flags & MF_NORMAL_DX) != 0u);
    }

    // --- lighting: baked SH-L1 irradiance (world-pos) + one analytic sun term ---
    let irr = sample_sh_irradiance(in.world_pos, N, view.camera_pos) * gi.vol_min.w;
    let ndl = max(dot(N, view.sun_dir), 0.0);
    var color = albedo * (irr + view.sun_color * ndl);

    if ((mat.flags & MF_EMISSIVE) != 0u && mat.emissive_idx != NO_TEX) {
        color = color + sample_tex(mat.emissive_idx, in.uv).rgb * mat.emissive.rgb;
    } else if ((mat.flags & MF_EMISSIVE) != 0u) {
        color = color + mat.emissive.rgb;
    }

    // Output LINEAR HDR. grade.wgsl applies exposure + LUT + display encode downstream.
    return vec4<f32>(color, alpha);
}
