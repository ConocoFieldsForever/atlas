//! .eftpack v1 loader.
//!
//! The pack is SELF-DESCRIBING: `manifest.json` declares every stride and byte
//! offset, and this loader reads the layout FROM the manifest — it never
//! hardcodes strides or field positions, so the python emitter (assemble_bevy.py)
//! and this consumer cannot drift.
//!
//! Layout (v1 contract):
//!   <pack>/manifest.json   — version, bounds, vertex layout, instance layout,
//!                            meshes[] (per-submesh material + index ranges),
//!                            counts, conventions, in-place sidecar abs paths.
//!   <pack>/meshes.bin      — interleaved vertices (per vertex layout) followed
//!                            by u32 indices; byte offsets/counts in manifest.
//!   <pack>/instances.bin   — fixed-stride instance records per the instance
//!                            layout. `affine` is the FULL conjugated ROW-MAJOR
//!                            world 3x4 (12 f32) INCLUDING shear + mirror.
//!   <pack>/materials.json  — per-submesh material records.
//!
//! THE #1 RULE (tarkov-unity-extraction skill): the affine is applied to RAW
//! verts. NEVER TRS-decompose. glam `Affine3A` carries the full 3x3 (shear and
//! negative-determinant / mirror) losslessly; the renderer flips winding via the
//! MIRROR flag bit instead of baking.

use anyhow::{anyhow, Context, Result};
use glam::{Affine3A, Mat3, Quat, Vec3};
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Instance flag bits (consumer side). MUST stay in lockstep with the emitter
// in assemble_bevy.py (FLAG_MIRROR/FLAG_TERRAIN/FLAG_BAKED). These replace the
// web path's bake-everything gate: a mirror is INSTANCED with a winding flip,
// not baked.
// ---------------------------------------------------------------------------
pub mod flags {
    /// det3(conjugated affine) < 0. The emitter sets this; the renderer keeps
    /// mirrors correct by (a) drawing double-sided (cull off) and (b) using the
    /// COFACTOR normal matrix (which flips normal sign for det<0) — no baking,
    /// no per-instance front-face pipeline needed for the M0 double-sided path.
    pub const MIRROR: u32 = 1 << 0;
    /// MicroSplat terrain tile (drive with the terrain splat shader).
    pub const TERRAIN: u32 = 1 << 1;
    /// Identity affine; geometry PRE-BAKED to world (degenerate/rank-deficient
    /// fallback). No per-instance normal matrix.
    pub const BAKED: u32 = 1 << 2;
}

// ---------------------------------------------------------------------------
// manifest.json
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub dataset: String,
    /// Canonical map id (== the roster / `config.json` `id` and the `.eftpack` basename), written by
    /// `assemble_bevy` as `map`. This is the STABLE join key for loot / POI / tasks — unlike `dataset`,
    /// which is the dataset DIRECTORY basename and can be `interchange_v2` / mismatch the intel key
    /// (e.g. `factory_rework` vs a stale `factory`). Empty on an older pack that predates the field, in
    /// which case the `dataset` + `_vN`-strip candidates still resolve.
    #[serde(default)]
    pub map: String,
    /// [minX,minY,minZ, maxX,maxY,maxZ] world AABB (computed by the emitter from
    /// world-space verts). Used for the initial camera framing.
    pub bounds: [f32; 6],
    pub vertex: VertexLayout,
    pub instance: InstanceLayout,
    pub meshes: Vec<MeshEntry>,
    #[serde(rename = "instanceCount")]
    pub instance_count: u32,
    #[serde(rename = "materialCount")]
    pub material_count: u32,
    /// Root-GameObject names; instance.rootId indexes this table.
    #[serde(default)]
    pub roots: Vec<String>,
    /// Correctness conventions the emitter baked in. The renderer READS these so
    /// it cannot double-apply a flip (SKILL: the historical whole-map-mirror bug).
    #[serde(default)]
    pub conventions: Conventions,
    #[serde(default)]
    pub sidecars: Sidecars,
}

/// Emitter-declared conventions. Every flag here changes how a shader must treat
/// the data; hardcoding the opposite double-applies it (upside-down textures /
/// inverted normal Y). Defaults match assemble_bevy.py's current output.
#[derive(Debug, Clone, Deserialize)]
pub struct Conventions {
    /// UV V was already flipped into the vertex UVs → the shader must NOT re-flip.
    #[serde(rename = "uvVFlipBaked", default = "yes")]
    pub uv_v_flip_baked: bool,
    /// Unity _ST tiling already baked into vertex UVs → uvXform is reference-only.
    #[serde(rename = "uvTilingBaked", default = "yes")]
    pub uv_tiling_baked: bool,
    /// Normal maps are DirectX-convention (green points down). The BC5 import
    /// must flip G, OR the shader must negate sampled n.y.
    #[serde(rename = "normalMapGreenFlip", default = "yes")]
    pub normal_map_green_flip: bool,
}

impl Default for Conventions {
    fn default() -> Self {
        Self {
            uv_v_flip_baked: true,
            uv_tiling_baked: true,
            normal_map_green_flip: true,
        }
    }
}
fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct VertexLayout {
    /// Interleaved vertex stride in bytes (contract default: 36 =
    /// pos f32x3 @0 + normal f32x3 @12 + uv f32x2 @24 + color unorm8x4 @32).
    pub stride: u32,
    pub attrs: Vec<Attr>,
}

impl VertexLayout {
    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.iter().find(|a| a.name == name)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Attr {
    pub name: String,
    /// "f32x3" | "f32x2" | "unorm8x4" | ...
    pub fmt: String,
    /// byte offset within the vertex stride.
    pub offset: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InstanceLayout {
    /// Fixed record stride in bytes (padded to 4B; contract pads to 80 for the
    /// 16B-aligned storage-buffer path).
    pub stride: u32,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Field {
    pub name: String,
    /// "f32x12" | "u32" | "i32"
    pub fmt: String,
    pub offset: u32,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeshEntry {
    pub id: u32,
    pub name: String,
    /// BYTE offset into meshes.bin where this mesh's interleaved vertices begin.
    #[serde(rename = "vtxOffset")]
    pub vtx_offset: u64,
    /// number of vertices (each `vertex.stride` bytes).
    #[serde(rename = "vtxCount")]
    pub vtx_count: u32,
    /// BYTE offset into meshes.bin where this mesh's u32 index run begins.
    #[serde(rename = "idxOffset")]
    pub idx_offset: u64,
    /// number of u32 indices.
    #[serde(rename = "idxCount")]
    pub idx_count: u32,
    pub submeshes: Vec<SubMesh>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SubMesh {
    #[serde(rename = "materialId")]
    pub material_id: u32,
    /// start index WITHIN this mesh's index run (not a global offset).
    #[serde(rename = "idxStart")]
    pub idx_start: u32,
    #[serde(rename = "idxCount")]
    pub idx_count: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Sidecars {
    /// Absolute paths INTO eft_assets — referenced in place, never copied.
    #[serde(rename = "terrainLayers", default)]
    pub terrain_layers: Option<String>,
    #[serde(default)]
    pub lights: Option<String>,
    /// EVERY light sidecar the assembler emitted (`lightsAll`). A multi-scene map (or any pack with
    /// several `lights_*.json`) writes them all here; the single `lights` field is just the primary
    /// one. The loader parses+merges ALL of these so multi-file light maps aren't reduced to one file
    /// (finding 3b). Empty/absent -> fall back to the single `lights` field.
    #[serde(rename = "lightsAll", default)]
    pub lights_all: Vec<String>,
    #[serde(default)]
    pub volume: Option<String>,
    #[serde(default)]
    pub semantics: Option<String>,
    /// SH volume layout descriptor (volume.json) — the loader reads the 3D-texture
    /// dims/layout from here rather than hardcoding.
    #[serde(rename = "volumeMeta", default)]
    pub volume_meta: Option<String>,
    #[serde(rename = "volumeVis", default)]
    pub volume_vis: Option<String>,
    #[serde(rename = "grassJson", default)]
    pub grass_json: Option<String>,
}

// ---------------------------------------------------------------------------
// materials.json
// ---------------------------------------------------------------------------

/// Emissive block. NOTE: the emitter writes this as a JSON OBJECT (or null), not
/// a bare string — modelling it as `Option<String>` (the previous bug) made
/// serde abort the whole materials.json parse on any lit material (2,526 subs on
/// the full interchange map), so the pack would fail to load entirely.
#[derive(Debug, Clone, Deserialize)]
pub struct Emissive {
    #[serde(default)]
    pub texture: Option<String>,
    /// linear rgb emissive factor (HDR normalized into [0,1]).
    #[serde(default = "one3")]
    pub factor: [f32; 3],
    /// HDR overdrive (>1 means factor was normalized by this).
    #[serde(default = "one")]
    pub hdr: f32,
}
fn one3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn one() -> f32 {
    1.0
}

/// Vert-paint (Custom/Vert Paint SoftCutout Decal) 3-layer splat block.
/// Kept as a loose value tree for M0; the M3 shader path reads it into uniforms.
pub type VertPaint = serde_json::Value;

/// #6 Detail maps. Optional per-material SECONDARY albedo/normal that adds high-frequency surface
/// contrast on top of the base textures. Present on ~23 rock materials; `null`/absent otherwise.
/// Albedo and normal are INDEPENDENT (either may be present). The `*Uv` fields are the RAW Unity
/// `_DetailAlbedoMap_ST` / `_DetailNormalMap_ST` (sx,sy,ox,oy) — NOT applied directly: the shader
/// derives a RELATIVE transform against the base `_MainTex_ST` (`Material.uv_xform`), because that
/// base ST is already baked into the vertex UVs. `albedoMeanGain` is the offline mean of the detail
/// sample (linear × 4.5948) per channel; the shader divides by it so the blend adds contrast
/// without shifting overall brightness/color (mean-neutralize).
#[derive(Debug, Clone, Deserialize)]
pub struct DetailMap {
    /// sRGB detail albedo PNG (same format as base albedo), or absent.
    #[serde(default)]
    pub albedo: Option<String>,
    /// RAW `_DetailAlbedoMap_ST` (sx,sy,ox,oy).
    #[serde(rename = "albedoUv", default)]
    pub albedo_uv: [f32; 4],
    /// Detail albedo blend strength 0..1.
    #[serde(rename = "albedoStrength", default)]
    pub albedo_strength: f32,
    /// mean(sample_linear.rgb × 4.5948) per channel — for the mean-neutralize divide.
    #[serde(rename = "albedoMeanGain", default)]
    pub albedo_mean_gain: [f32; 3],
    /// linear detail normal PNG (same format as base normal), or absent.
    #[serde(default)]
    pub normal: Option<String>,
    /// RAW `_DetailNormalMap_ST` (sx,sy,ox,oy).
    #[serde(rename = "normalUv", default)]
    pub normal_uv: [f32; 4],
    /// Detail normal strength (tangent xy multiplier).
    #[serde(rename = "normalScale", default)]
    pub normal_scale: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Material {
    pub id: u32,
    /// "opaque" | "cutout" | "glass" | "decal" | "water"
    pub role: String,
    /// Full-res albedo path into eft_assets/<ds>/tex (BC7/sRGB on import).
    #[serde(default)]
    pub albedo: Option<String>,
    /// Full-res normal path (BC5 on import), or null.
    #[serde(default)]
    pub normal: Option<String>,
    /// [sx,sy,ox,oy] — recorded for reference; geometry already has it baked
    /// into UVs (do NOT double-apply in the shader; see manifest.conventions).
    #[serde(rename = "uvXform", default = "default_uv")]
    pub uv_xform: [f32; 4],
    /// "OPAQUE" | "MASK" | "BLEND"
    #[serde(rename = "alphaMode", default = "opaque_mode")]
    pub alpha_mode: String,
    #[serde(rename = "alphaCutoff", default)]
    pub alpha_cutoff: f32,
    /// _col4(_Color): sRGB->linear RGB, alpha linear. Albedo = tex * tint.
    #[serde(default = "default_tint")]
    pub tint: [f32; 4],
    #[serde(default)]
    pub metallic: Option<f32>,
    #[serde(default)]
    pub roughness: Option<f32>,
    #[serde(rename = "normalScale", default = "one")]
    pub normal_scale: f32,
    /// DirectX-convention normal (green down) → flip G on import or negate n.y.
    #[serde(rename = "normalGreenFlip", default)]
    pub normal_green_flip: bool,
    /// EFT deferred draws building shells solid from both sides → default true.
    #[serde(rename = "doubleSided", default = "yes")]
    pub double_sided: bool,
    /// Emissive object {texture,factor,hdr} or null (see `Emissive`).
    #[serde(default)]
    pub emissive: Option<Emissive>,
    /// roughness = 1 - albedo.a (smoothness packed in albedo alpha).
    #[serde(rename = "roughnessFromAlbedoAlpha", default)]
    pub roughness_from_albedo_alpha: bool,
    /// _SpecMap path; roughness derived from its luma.
    #[serde(rename = "specMap", default)]
    pub spec_map: Option<String>,
    /// Vert-paint block (null unless this material is a Vert Paint variant).
    #[serde(default)]
    pub vp: Option<VertPaint>,
    /// #6 Detail map block (rock secondary albedo/normal), or `null`/absent. See `DetailMap`.
    /// Deserializes cleanly to `None` when the field is missing or JSON null.
    #[serde(default)]
    pub detail: Option<DetailMap>,
}

fn default_uv() -> [f32; 4] {
    [1.0, 1.0, 0.0, 0.0]
}
fn default_tint() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}
fn opaque_mode() -> String {
    "OPAQUE".to_string()
}

// ---------------------------------------------------------------------------
// GPU instance record (repacked for the instance-rate vertex buffer / storage
// buffer). On disk the record is `instance.stride` bytes (padded to 4B). We
// re-read it by manifest offsets and repack into this 16-byte-aligned struct.
// 80 bytes = multiple of 16.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstance {
    /// ROW-MAJOR world 3x4 incl shear+mirror. Rows: [0..4]=r0, [4..8]=r1, [8..12]=r2.
    pub affine: [f32; 12],
    pub mesh_id: u32,
    pub lod_group: i32,
    pub lod_index: i32,
    pub root_id: u32,
    pub flags: u32,
    pub _pad: [u32; 3],
}

impl GpuInstance {
    /// Build a glam `Affine3A` from the row-major 3x4 WITHOUT decomposing.
    /// Column i of the linear part = (r0[i], r1[i], r2[i]); translation = col 3.
    pub fn affine3a(&self) -> Affine3A {
        let a = &self.affine;
        let m = Mat3::from_cols(
            Vec3::new(a[0], a[4], a[8]),
            Vec3::new(a[1], a[5], a[9]),
            Vec3::new(a[2], a[6], a[10]),
        );
        Affine3A::from_mat3_translation(m, Vec3::new(a[3], a[7], a[11]))
    }

    #[inline]
    pub fn is_mirror(&self) -> bool {
        self.flags & flags::MIRROR != 0
    }
}

/// Per-mesh bounding sphere for the frustum-cull compute pass.
/// [cx,cy,cz,radius] in MESH-LOCAL space (transform to world in the shader with
/// the shear-correct max-column-norm radius scale).
pub type BoundingSphere = [f32; 4];

/// CPU-side unpacked mesh geometry ready to build a Bevy `Mesh`.
pub struct MeshGeom {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// COLOR_0 vert-paint weights (linear [0,1]); empty when not a vp mesh.
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Realtime lights (point/spot). EFT lights its maps with REALTIME lights (no baked
// lightmaps/probes ship in the bundles). The pack carries the raw light set in the
// `lights` sidecar; the viewer lights the scene entirely at runtime from it (see the
// GPU light grid in render/gpu_driven.rs). We REDUCE each raw Unity light to a
// viewer-world point/vector here — the ONLY transform is the EFT handedness flip
// G3 = diag(-1,1,1) (X sign) applied to both position and the spot forward vector
// (tarkov-unity-extraction skill §9); no TRS decompose.
// ---------------------------------------------------------------------------

/// A realtime point/spot light reduced to viewer world space and packed for the GPU
/// light loop. Light intensity is folded into `color` at parse (linear_rgb * intensity).
#[derive(Debug, Clone, Copy)]
pub struct Light {
    /// Viewer-world position (G3 X-flip applied to the raw Unity position).
    pub pos: Vec3,
    /// Linear RGB × intensity — the radiant color the shader multiplies by attenuation.
    pub color: Vec3,
    /// Range in meters (hard cutoff).
    pub range: f32,
    /// Spot forward direction, viewer-world, normalized. (0,0,0) for point lights.
    pub dir: Vec3,
    /// cos(spotAngle/2): -2.0 for point lights (so the spot factor is 1 everywhere).
    pub cos_outer: f32,
    /// cos(innerSpotAngle/2): -1.0 for point lights. Always > `cos_outer` (smoothstep edges).
    pub cos_inner: f32,
}

/// Raw light record as authored in the `lights_*.json` sidecar (Unity world space).
#[derive(Debug, Clone, Deserialize)]
struct LightRaw {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    position: [f32; 3],
    #[serde(default = "ident_quat")]
    rotation: [f32; 4],
    #[serde(default = "default_tint")]
    color: [f32; 4],
    #[serde(default = "one")]
    intensity: f32,
    #[serde(default)]
    range: f32,
    #[serde(rename = "spotAngle", default)]
    spot_angle: f32,
    #[serde(rename = "innerSpotAngle", default)]
    inner_spot_angle: f32,
    #[serde(default = "yes")]
    on: bool,
}
fn ident_quat() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// The sidecar may be a bare array OR `{ "lights": [...] }`. Tolerate both.
#[derive(Debug, Clone, Deserialize)]
struct LightsWrapper {
    #[serde(default)]
    lights: Vec<LightRaw>,
}

/// Outcome of reducing one raw light: a usable point/spot `Light`, a deliberate skip because the
/// light is off / contributes nothing, or a skip because its Unity type isn't a point/spot the
/// realtime loop supports (Directional/Rectangle/Disc/Area). The last variant is counted + warned
/// so an unsupported type is never silently mis-coerced into a point light (finding 3c).
enum Reduced {
    Light(Light),
    /// off / zero-intensity / zero-range — expected, not warned.
    Inactive,
    /// unsupported Light type (kind names it) — skipped, warned in aggregate.
    Unsupported,
}

/// Reduce a raw Unity light to a viewer-world point/spot `Light`. Directional/area (Rectangle/Disc)
/// and any other non-point/spot type is reported `Unsupported` (skipped, not mis-coerced): the
/// viewer lights interiors from the baked SH + sky sun, so a Directional (sun) is NOT a local point
/// light and must not be dropped into the point loop at the sun's position.
fn reduce_light(r: &LightRaw) -> Reduced {
    // Classify the Unity light type. Empty kind is legacy "point" (older sidecars omitted it).
    let k = r.kind.trim().to_ascii_lowercase();
    let is_spot = k == "spot";
    let is_point = k.is_empty() || k == "point";
    if !is_spot && !is_point {
        return Reduced::Unsupported; // Directional / Rectangle / Disc / Area / unknown
    }
    if !r.on || r.intensity <= 0.0 || r.range <= 0.0 {
        return Reduced::Inactive;
    }
    // G3 = diag(-1,1,1): sign-flip X on the raw Unity world position.
    let pos = Vec3::new(-r.position[0], r.position[1], r.position[2]);
    let color = Vec3::new(r.color[0], r.color[1], r.color[2]) * r.intensity;
    let (dir, cos_outer, cos_inner) = if is_spot {
        // Unity forward is +Z; rotate by the light's quat, then apply the same G3 X-flip.
        let q = Quat::from_xyzw(r.rotation[0], r.rotation[1], r.rotation[2], r.rotation[3]);
        let q = if q.length_squared() > 1e-8 {
            q.normalize()
        } else {
            Quat::IDENTITY
        };
        let f = q * Vec3::Z;
        let fwd = Vec3::new(-f.x, f.y, f.z).normalize_or_zero();
        let co = (r.spot_angle.to_radians() * 0.5).cos();
        let mut ci = (r.inner_spot_angle.to_radians() * 0.5).cos();
        // smoothstep(cos_outer, cos_inner, cosang) needs cos_inner > cos_outer.
        if ci <= co {
            ci = co + 1e-3;
        }
        (fwd, co, ci)
    } else {
        (Vec3::ZERO, -2.0, -1.0) // point: spot factor == 1 everywhere
    };
    Reduced::Light(Light {
        pos,
        color,
        range: r.range,
        dir,
        cos_outer,
        cos_inner,
    })
}

/// Parse ONE lights sidecar into usable point/spot lights, appending to `out`. Returns the count of
/// UNSUPPORTED-type lights skipped (Directional/Rectangle/Disc/…). A missing / unreadable /
/// unparseable file logs a clear WARNING and contributes nothing — realtime lighting is optional and
/// must NEVER fail the load, but a parse failure is no longer SILENT (finding 3c).
fn parse_lights_into(root: &Path, sidecar: &str, out: &mut Vec<Light>) -> usize {
    let path = if Path::new(sidecar).is_absolute() {
        PathBuf::from(sidecar)
    } else {
        root.join(sidecar)
    };
    let s = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  lights: WARNING could not read '{}': {e}", path.display());
            return 0;
        }
    };
    let raws: Vec<LightRaw> = match serde_json::from_str::<Vec<LightRaw>>(&s) {
        Ok(v) => v,
        Err(_) => match serde_json::from_str::<LightsWrapper>(&s) {
            Ok(w) => w.lights,
            Err(e) => {
                eprintln!("  lights: WARNING could not parse '{}': {e}", path.display());
                return 0;
            }
        },
    };
    let mut unsupported = 0usize;
    for r in &raws {
        match reduce_light(r) {
            Reduced::Light(l) => out.push(l),
            Reduced::Inactive => {}
            Reduced::Unsupported => unsupported += 1,
        }
    }
    unsupported
}

// ---------------------------------------------------------------------------
// Loaded pack (CPU side).
// ---------------------------------------------------------------------------
pub struct Pack {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub materials: Vec<Material>,
    /// Raw meshes.bin: interleaved verts then u32 indices. Sliced via manifest
    /// byte offsets.
    pub meshes_bin: Vec<u8>,
    /// Repacked, 16B-aligned instances.
    pub instances: Vec<GpuInstance>,
    /// Realtime point/spot lights reduced to viewer world space (empty if the pack
    /// ships no lights sidecar or it failed to parse — never fatal).
    pub lights: Vec<Light>,
}

impl Pack {
    /// Resolve a manifest/materials path: SELF-CONTAINED packs (redistribution PR3) write
    /// pack-relative paths ("tex/foo.png"); legacy packs wrote absolute workspace paths.
    /// Relative resolves against the pack dir; absolute passes through untouched.
    pub fn resolve_path(&self, p: &str) -> String {
        let pb = std::path::Path::new(p);
        if pb.is_absolute() {
            p.to_string()
        } else {
            self.root.join(pb).to_string_lossy().into_owned()
        }
    }

    pub fn load(dir: impl AsRef<Path>) -> Result<Pack> {
        // Keep the pack root absolute from the moment it enters the loader.  Several rendering
        // sidecars are resolved in more than one stage (main world -> render world); a relative
        // root made an already-resolved path still look relative and produced paths such as
        // `packs/lighthouse.eftpack/packs/lighthouse.eftpack/terrain_layers/...`.
        let requested_root = dir.as_ref();
        let root = std::fs::canonicalize(requested_root)
            .unwrap_or_else(|_| requested_root.to_path_buf());
        if !root.is_dir() {
            return Err(anyhow!("pack dir does not exist: {}", root.display()));
        }

        let manifest: Manifest =
            read_json(&root.join("manifest.json")).context("reading manifest.json")?;
        if manifest.version != 1 {
            return Err(anyhow!(
                "unsupported .eftpack version {} (loader speaks v1)",
                manifest.version
            ));
        }

        let materials: Vec<Material> =
            read_json(&root.join("materials.json")).context("reading materials.json")?;

        let meshes_bin = std::fs::read(root.join("meshes.bin"))
            .with_context(|| format!("reading {}", root.join("meshes.bin").display()))?;
        let inst_bin = std::fs::read(root.join("instances.bin"))
            .with_context(|| format!("reading {}", root.join("instances.bin").display()))?;

        let instances = parse_instances(&manifest.instance, &inst_bin)
            .context("parsing instances.bin by manifest layout")?;

        if instances.len() != manifest.instance_count as usize {
            return Err(anyhow!(
                "instanceCount {} disagrees with parsed {} records",
                manifest.instance_count,
                instances.len()
            ));
        }

        // Realtime lights (best-effort; empty vec on any failure — never fatal). Parse+merge EVERY
        // sidecar in `lightsAll` (finding 3b); fall back to the single `lights` field when the list
        // is empty (older packs). De-dup identical entries so `lights` + `lightsAll` overlap once.
        let mut light_files: Vec<&str> = manifest
            .sidecars
            .lights_all
            .iter()
            .map(String::as_str)
            .collect();
        if light_files.is_empty() {
            if let Some(s) = manifest.sidecars.lights.as_deref() {
                light_files.push(s);
            }
        }
        light_files.sort_unstable();
        light_files.dedup();
        let mut lights = Vec::new();
        let mut unsupported = 0usize;
        for f in &light_files {
            unsupported += parse_lights_into(&root, f, &mut lights);
        }
        eprintln!(
            "  realtime lights: {} usable from {} sidecar(s){}",
            lights.len(),
            light_files.len(),
            if unsupported > 0 {
                format!(" ({unsupported} unsupported-type light(s) skipped: Directional/area handled by SH+sky, not the point loop)")
            } else {
                String::new()
            }
        );

        let pack = Pack {
            root,
            manifest,
            materials,
            meshes_bin,
            instances,
            lights,
        };
        // Complete checked structural validation BEFORE the pack is exposed (finding 5): a malformed
        // manifest that parsed fine but whose offsets/ranges/ids are out of bounds would otherwise
        // panic-abort later at a GPU slice (release builds are panic=abort). Returning Err here lets
        // the loader's existing error path handle it gracefully.
        pack.validate().context("validating pack structure")?;
        Ok(pack)
    }

    /// Complete checked structural validation of a parsed pack (finding 5). Verifies every
    /// manifest-declared offset / range / id lands inside the buffer or table it indexes, so no
    /// later unchecked read (`mesh_geom`, the GPU submesh slice in `gpu_driven`, an instance's
    /// mesh_id lookup) can panic on user-supplied data. Returns a descriptive Err on the first
    /// violation. Called once at the end of `load`.
    fn validate(&self) -> Result<()> {
        let vl = &self.manifest.vertex;
        let stride = vl.stride as usize;
        if stride == 0 {
            return Err(anyhow!("vertex stride is 0"));
        }
        // Every vertex attribute must lie fully within one stride record. `mesh_geom` reads each
        // attribute at `base + offset`, so an attr whose [offset, offset+size] overruns the stride
        // reads past the last vertex's bytes (e.g. stride=12 with position.offset=8 reads byte 19).
        for a in &vl.attrs {
            let size = fmt_byte_size(&a.fmt).ok_or_else(|| {
                anyhow!("vertex attr '{}' has unknown format '{}'", a.name, a.fmt)
            })?;
            let end = a.offset as usize + size;
            if end > stride {
                return Err(anyhow!(
                    "vertex attr '{}' [{}, {}) overruns stride {}",
                    a.name,
                    a.offset,
                    end,
                    stride
                ));
            }
        }
        let mat_count = self.materials.len();
        let mesh_count = self.manifest.meshes.len();
        let blen = self.meshes_bin.len();
        for m in &self.manifest.meshes {
            // Mesh vertex + index byte ranges must fit meshes.bin.
            let vtx_end = m
                .vtx_offset
                .checked_add(m.vtx_count as u64 * stride as u64)
                .ok_or_else(|| anyhow!("mesh {} vtx range overflow", m.id))? as usize;
            let idx_end = m
                .idx_offset
                .checked_add(m.idx_count as u64 * 4)
                .ok_or_else(|| anyhow!("mesh {} idx range overflow", m.id))? as usize;
            if vtx_end > blen || idx_end > blen {
                return Err(anyhow!(
                    "mesh {} '{}' byte range out of bounds (vtx_end {}, idx_end {}, meshes.bin {})",
                    m.id,
                    m.name,
                    vtx_end,
                    idx_end,
                    blen
                ));
            }
            // Each submesh index window must fit WITHIN this mesh's index run, and its material id
            // must exist. A submesh with idx_start > idx_count panics the GPU slice in gpu_driven.
            for (si, sm) in m.submeshes.iter().enumerate() {
                let sub_end = sm
                    .idx_start
                    .checked_add(sm.idx_count)
                    .ok_or_else(|| anyhow!("mesh {} submesh {} idx overflow", m.id, si))?;
                if sub_end > m.idx_count {
                    return Err(anyhow!(
                        "mesh {} '{}' submesh {} index window [{}, {}) exceeds mesh idxCount {}",
                        m.id,
                        m.name,
                        si,
                        sm.idx_start,
                        sub_end,
                        m.idx_count
                    ));
                }
                if mat_count > 0 && sm.material_id as usize >= mat_count {
                    return Err(anyhow!(
                        "mesh {} '{}' submesh {} materialId {} out of range (materials {})",
                        m.id,
                        m.name,
                        si,
                        sm.material_id,
                        mat_count
                    ));
                }
            }
        }
        // Every instance must reference a real mesh (mesh_id indexes manifest.meshes / the GPU
        // mesh-meta buffer). `instances_by_mesh` silently drops out-of-range ids, but the GPU path
        // indexes with them directly.
        for (i, inst) in self.instances.iter().enumerate() {
            if inst.mesh_id as usize >= mesh_count {
                return Err(anyhow!(
                    "instance {} references mesh_id {} but only {} meshes exist",
                    i,
                    inst.mesh_id,
                    mesh_count
                ));
            }
        }
        Ok(())
    }

    /// Interleaved vertex bytes for a mesh (length = vtxCount * vertex.stride).
    pub fn vertex_bytes(&self, m: &MeshEntry) -> &[u8] {
        let stride = self.manifest.vertex.stride as u64;
        let start = m.vtx_offset as usize;
        let end = (m.vtx_offset + m.vtx_count as u64 * stride) as usize;
        &self.meshes_bin[start..end]
    }

    /// u32 index bytes for a mesh (length = idxCount * 4).
    pub fn index_bytes(&self, m: &MeshEntry) -> &[u8] {
        let start = m.idx_offset as usize;
        let end = (m.idx_offset + m.idx_count as u64 * 4) as usize;
        &self.meshes_bin[start..end]
    }

    /// Unpack a mesh's interleaved bytes into typed attribute vectors for a Bevy
    /// `Mesh`. Reads attribute offsets/formats FROM the manifest vertex layout.
    pub fn mesh_geom(&self, m: &MeshEntry) -> Result<MeshGeom> {
        let vl = &self.manifest.vertex;
        let stride = vl.stride as usize;
        // Validate byte ranges before slicing so a truncated / mismatched pack
        // returns an error the caller can skip, not a panic (Codex P2).
        let vtx_end = m.vtx_offset as usize + m.vtx_count as usize * stride;
        let idx_end = m.idx_offset as usize + m.idx_count as usize * 4;
        let blen = self.meshes_bin.len();
        if vtx_end > blen || idx_end > blen {
            return Err(anyhow!(
                "mesh {} '{}' byte range out of bounds (vtx_end {}, idx_end {}, meshes.bin {})",
                m.id,
                m.name,
                vtx_end,
                idx_end,
                blen
            ));
        }
        let vb = self.vertex_bytes(m);
        let n = m.vtx_count as usize;

        let pos = vl
            .attr("position")
            .ok_or_else(|| anyhow!("vertex layout missing 'position'"))?;
        let nrm = vl.attr("normal");
        let uv = vl.attr("uv");
        let col = vl.attr("color");

        let mut positions = Vec::with_capacity(n);
        let mut normals = Vec::with_capacity(n);
        let mut uvs = Vec::with_capacity(n);
        let mut colors = Vec::new();
        if col.is_some() {
            colors.reserve(n);
        }

        for i in 0..n {
            let base = i * stride;
            let p = read_vec3(vb, base + pos.offset as usize);
            positions.push([p.x, p.y, p.z]);
            if let Some(a) = nrm {
                let v = read_vec3(vb, base + a.offset as usize);
                normals.push([v.x, v.y, v.z]);
            } else {
                normals.push([0.0, 1.0, 0.0]);
            }
            if let Some(a) = uv {
                let o = base + a.offset as usize;
                uvs.push([read_f32(vb, o), read_f32(vb, o + 4)]);
            } else {
                uvs.push([0.0, 0.0]);
            }
            if let Some(a) = col {
                let o = base + a.offset as usize;
                colors.push([
                    vb[o] as f32 / 255.0,
                    vb[o + 1] as f32 / 255.0,
                    vb[o + 2] as f32 / 255.0,
                    vb[o + 3] as f32 / 255.0,
                ]);
            }
        }

        let ib = self.index_bytes(m);
        let ni = m.idx_count as usize;
        let mut indices = Vec::with_capacity(ni);
        for i in 0..ni {
            indices.push(read_u32(ib, i * 4));
        }

        Ok(MeshGeom {
            positions,
            normals,
            uvs,
            colors,
            indices,
        })
    }

    /// Center of the world AABB (initial camera target).
    pub fn bounds_center(&self) -> Vec3 {
        let b = &self.manifest.bounds;
        Vec3::new(
            0.5 * (b[0] + b[3]),
            0.5 * (b[1] + b[4]),
            0.5 * (b[2] + b[5]),
        )
    }

    /// "Where the map's content actually is" — the per-axis MEDIAN of every instance's world
    /// translation. Used for the initial camera framing so it opens NEAR populated geometry on
    /// EVERY map, not at the AABB center (which for many maps sits in ocean / dead space off the
    /// playable area). Median (not mean) rejects far backdrop / skybox / sea-plane outliers that
    /// would otherwise drag the anchor into emptiness. Falls back to the AABB center when the pack
    /// has no instances.
    pub fn content_anchor(&self) -> Vec3 {
        if self.instances.is_empty() {
            return self.bounds_center();
        }
        let mut xs: Vec<f32> = self.instances.iter().map(|i| i.affine[3]).collect();
        let mut ys: Vec<f32> = self.instances.iter().map(|i| i.affine[7]).collect();
        let mut zs: Vec<f32> = self.instances.iter().map(|i| i.affine[11]).collect();
        let med = |v: &mut [f32]| -> f32 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        Vec3::new(med(&mut xs), med(&mut ys), med(&mut zs))
    }

    /// Half-diagonal of the world AABB (initial camera standoff).
    pub fn bounds_extent(&self) -> f32 {
        let b = &self.manifest.bounds;
        Vec3::new(b[3] - b[0], b[4] - b[1], b[5] - b[2]).length() * 0.5
    }

    /// Group instance indices by their meshId (skips the pack's baked-world
    /// mesh entries whose sole instance carries FLAG_BAKED — those still render,
    /// but grouping is by meshId regardless). Result[k] = instance indices whose
    /// mesh_id == k; length == manifest.meshes.len().
    pub fn instances_by_mesh(&self) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); self.manifest.meshes.len()];
        for (i, inst) in self.instances.iter().enumerate() {
            let mid = inst.mesh_id as usize;
            if mid < out.len() {
                out[mid].push(i as u32);
            }
        }
        out
    }

    /// Like `instances_by_mesh` but keeps only ONE LOD level per LODGroup, for the viewer's forced-
    /// LOD selector on `--alllod` packs. `forced_lod` = 0 (finest / default) .. N (coarser); each
    /// group renders `min(forced_lod, that group's max available lod)`, so forcing a level a group
    /// doesn't have falls back to its coarsest instead of vanishing. Ungrouped instances
    /// (`lod_group < 0` — terrain, non-LOD meshes) are always kept. A LOD0-only pack has every
    /// instance at `lod_index == 0`, so any `forced_lod` yields the identical (full) set — the
    /// selector is a no-op there.
    pub fn instances_by_mesh_for_lod(&self, forced_lod: i32) -> Vec<Vec<u32>> {
        let mut group_max: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
        for inst in &self.instances {
            if inst.lod_group >= 0 {
                let e = group_max.entry(inst.lod_group).or_insert(0);
                if inst.lod_index > *e {
                    *e = inst.lod_index;
                }
            }
        }
        let want = forced_lod.max(0);
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); self.manifest.meshes.len()];
        for (i, inst) in self.instances.iter().enumerate() {
            let keep = if inst.lod_group < 0 {
                true
            } else {
                inst.lod_index == want.min(*group_max.get(&inst.lod_group).unwrap_or(&0))
            };
            if keep {
                let mid = inst.mesh_id as usize;
                if mid < out.len() {
                    out[mid].push(i as u32);
                }
            }
        }
        out
    }

    /// Per-mesh LOCAL bounding spheres, indexed to match `manifest.meshes`.
    /// Feeds the cull compute pass. center = mean of positions, radius = max
    /// distance from center (cheap, tight enough for frustum rejection).
    pub fn bounding_spheres(&self) -> Result<Vec<BoundingSphere>> {
        let pos = self
            .manifest
            .vertex
            .attr("position")
            .ok_or_else(|| anyhow!("vertex layout has no 'position' attr"))?;
        if !pos.fmt.starts_with("f32x3") {
            return Err(anyhow!("position attr is {}, expected f32x3", pos.fmt));
        }
        let stride = self.manifest.vertex.stride as usize;
        let poff = pos.offset as usize;

        let mut out = Vec::with_capacity(self.manifest.meshes.len());
        for m in &self.manifest.meshes {
            // Validate the vertex byte range before slicing so a truncated/corrupt
            // meshes.bin returns an error the caller can handle, not a panic (Codex P2).
            let vtx_end = m.vtx_offset as usize + m.vtx_count as usize * stride;
            if vtx_end > self.meshes_bin.len() {
                return Err(anyhow!(
                    "mesh {} '{}' vertex range out of bounds (end {}, meshes.bin {})",
                    m.id,
                    m.name,
                    vtx_end,
                    self.meshes_bin.len()
                ));
            }
            let vb = self.vertex_bytes(m);
            let n = m.vtx_count as usize;
            let mut c = Vec3::ZERO;
            for i in 0..n {
                c += read_vec3(vb, i * stride + poff);
            }
            let center = if n > 0 { c / n as f32 } else { Vec3::ZERO };
            let mut r2 = 0.0f32;
            for i in 0..n {
                let p = read_vec3(vb, i * stride + poff);
                r2 = r2.max((p - center).length_squared());
            }
            out.push([center.x, center.y, center.z, r2.sqrt()]);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Instance record parsing driven by the manifest field table (no hardcoding).
// ---------------------------------------------------------------------------
fn parse_instances(layout: &InstanceLayout, bin: &[u8]) -> Result<Vec<GpuInstance>> {
    let stride = layout.stride as usize;
    if stride == 0 {
        return Err(anyhow!("instance.stride is 0"));
    }
    if bin.len() % stride != 0 {
        return Err(anyhow!(
            "instances.bin length {} is not a multiple of stride {}",
            bin.len(),
            stride
        ));
    }

    // Resolve field offsets by name so we tolerate any emitter field ordering.
    let find = |name: &str| -> Result<&Field> {
        layout
            .fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| anyhow!("instance layout missing field '{}'", name))
    };
    let f_affine = find("affine")?;
    if !f_affine.fmt.starts_with("f32x12") {
        return Err(anyhow!("affine field is {}, expected f32x12", f_affine.fmt));
    }
    let o_affine = f_affine.offset as usize;
    let o_mesh = find("meshId")?.offset as usize;
    let o_lg = find("lodGroup")?.offset as usize;
    let o_li = find("lodIndex")?.offset as usize;
    let o_root = find("rootId")?.offset as usize;
    let o_flags = find("flags")?.offset as usize;
    // Validate every field fits inside one stride record: a malformed manifest must Err at load
    // (message names the field), not panic on a slice index deep in the record loop (Codex review).
    for (name, off, sz) in [
        ("affine", o_affine, 48usize),
        ("meshId", o_mesh, 4),
        ("lodGroup", o_lg, 4),
        ("lodIndex", o_li, 4),
        ("rootId", o_root, 4),
        ("flags", o_flags, 4),
    ] {
        if off + sz > stride {
            return Err(anyhow!(
                "instance field '{}' (offset {} + {} bytes) overruns stride {}",
                name,
                off,
                sz,
                stride
            ));
        }
    }

    let count = bin.len() / stride;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * stride;
        let mut affine = [0.0f32; 12];
        for k in 0..12 {
            affine[k] = read_f32(bin, base + o_affine + k * 4);
        }
        out.push(GpuInstance {
            affine,
            mesh_id: read_u32(bin, base + o_mesh),
            lod_group: read_i32(bin, base + o_lg),
            lod_index: read_i32(bin, base + o_li),
            root_id: read_u32(bin, base + o_root),
            flags: read_u32(bin, base + o_flags),
            _pad: [0; 3],
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// little-endian primitive readers (alignment-safe)
// ---------------------------------------------------------------------------
#[inline]
pub(crate) fn read_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
pub(crate) fn read_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn read_i32(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
pub(crate) fn read_vec3(b: &[u8], o: usize) -> Vec3 {
    Vec3::new(read_f32(b, o), read_f32(b, o + 4), read_f32(b, o + 8))
}

/// Byte size of a vertex-attribute format string ("f32x3"=12, "unorm8x4"=4, "u32"=4, "f16x2"=4…).
/// `None` for an unrecognized format so `validate` rejects a manifest it can't bounds-check rather
/// than guessing. Grammar: `<base>[x<count>]`; per-component bytes by base type, times an optional
/// `x<count>` lane count.
fn fmt_byte_size(fmt: &str) -> Option<usize> {
    let (base, count) = match fmt.split_once('x') {
        Some((b, c)) => (b, c.parse::<usize>().ok()?),
        None => (fmt, 1),
    };
    let per = match base {
        "f32" | "u32" | "i32" | "unorm32" | "snorm32" => 4,
        "f16" | "u16" | "i16" | "unorm16" | "snorm16" => 2,
        "u8" | "i8" | "unorm8" | "snorm8" => 1,
        "f64" => 8,
        _ => return None,
    };
    Some(per * count)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))
}
