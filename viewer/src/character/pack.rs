//! `.eftchar` v1 loader.
//!
//! Same contract as [`crate::eftpack`]: the pack is SELF-DESCRIBING. `manifest.json` declares every
//! stride and byte offset and this loader reads the layout FROM the manifest, so the python emitter
//! (`extraction/characters/build_character.py`) and this consumer cannot drift.
//!
//! Layout (v1):
//!   <pack>/manifest.json  — conventions, skeleton, vertexLayout, meshes[], materials[], clips[],
//!                           controller state table, blob offsets.
//!   <pack>/skin.bin       — all meshes' interleaved vertices, then all meshes' u32 indices.
//!   <pack>/anim.bin       — per clip, per bone track, f32 position/rotation/scale arrays.
//!   <pack>/textures/*.png
//!
//! Everything in the pack is ALREADY in viewer space: the emitter applied the map pipeline's
//! `G3 = diag(-1,1,1)` conjugation to positions, rotations and bindposes and reversed triangle
//! winding to match (see `extraction/characters/coords.py`). This loader therefore performs NO
//! coordinate fixups — if geometry looks mirrored the emitter is wrong, not the consumer.

// The manifest structs mirror the v1 contract IN FULL, including fields no consumer reads yet
// (material scalar properties, per-mesh `boundBones`, state `layer`, `sampleRate`). They are the
// format's documentation, and declaring them is what turns an emitter/consumer mismatch into a
// parse error instead of a silent misread — so they stay even while unused.
#![allow(dead_code)]

use anyhow::{anyhow, bail, Context, Result};
use glam::{Mat4, Quat, Vec3};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const SUPPORTED_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// manifest.json
// ---------------------------------------------------------------------------
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Conventions {
    #[serde(default)]
    pub g3: Vec<f32>,
    #[serde(default)]
    pub winding_flipped: bool,
    #[serde(default)]
    pub quat_order: String,
    /// Same flag `.eftpack` carries: V is already flipped in the vertex UVs, so nothing downstream
    /// re-flips. Unity's UV origin is bottom-left, Bevy/wgpu's is top-left.
    #[serde(default, rename = "uvVFlipBaked")]
    pub uv_v_flip_baked: bool,
}

#[derive(Debug, Deserialize)]
pub struct SkeletonManifest {
    #[serde(rename = "boneCount")]
    pub bone_count: usize,
    pub names: Vec<String>,
    pub paths: Vec<String>,
    pub parents: Vec<i32>,
    #[serde(rename = "localPos")]
    pub local_pos: Vec<[f32; 3]>,
    #[serde(rename = "localRot")]
    pub local_rot: Vec<[f32; 4]>,
    #[serde(rename = "localScale")]
    pub local_scale: Vec<[f32; 3]>,
}

#[derive(Debug, Deserialize)]
pub struct VertexAttr {
    pub name: String,
    pub format: String,
    pub offset: usize,
}

#[derive(Debug, Deserialize)]
pub struct VertexLayout {
    pub stride: usize,
    pub attributes: Vec<VertexAttr>,
}

impl VertexLayout {
    pub fn attr(&self, name: &str) -> Option<&VertexAttr> {
        self.attributes.iter().find(|a| a.name == name)
    }
}

fn view_third() -> String {
    "third".into()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubMeshManifest {
    pub material: usize,
    pub index_start: usize,
    pub index_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshManifest {
    pub name: String,
    pub part: String,
    /// Which view this geometry belongs to: `"third"` (the body) or `"first"` (the FPV hands).
    /// Packs built before the hands existed carry no field and are all third-person.
    #[serde(default = "view_third")]
    pub view: String,
    pub lod: u32,
    pub vertex_count: usize,
    pub vertex_byte_offset: usize,
    pub vertex_byte_length: usize,
    pub index_count: usize,
    pub index_byte_offset: usize,
    pub index_byte_length: usize,
    pub bound_bones: Vec<usize>,
    /// Rig-sized (boneCount x 16) row-major inverse bindposes. Identity where this mesh does not
    /// bind that bone, which is what lets every mesh share one joint entity list.
    pub inverse_bindposes: Vec<Vec<f32>>,
    pub submeshes: Vec<SubMeshManifest>,
}

/// A RIGID equipment mesh (helmet, facecover, cap) parented to one rig bone. Unskinned: EFT's
/// headwear prefabs are `MeshFilter`/`MeshRenderer` with no bindposes, so they ride a bone.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentManifest {
    pub name: String,
    pub bone: usize,
    pub lod: u32,
    pub local_pos: [f32; 3],
    pub local_rot: [f32; 4],
    pub local_scale: [f32; 3],
    pub vertex_count: usize,
    pub vertex_byte_offset: usize,
    pub vertex_byte_length: usize,
    pub index_count: usize,
    pub index_byte_offset: usize,
    pub index_byte_length: usize,
    pub submeshes: Vec<SubMeshManifest>,
}

#[derive(Debug, Deserialize)]
pub struct MaterialManifest {
    pub name: String,
    #[serde(default)]
    pub textures: HashMap<String, String>,
    #[serde(default)]
    pub floats: HashMap<String, f32>,
    #[serde(default)]
    pub colors: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelManifest {
    pub byte_offset: usize,
    pub byte_length: usize,
    pub components: usize,
}

#[derive(Debug, Deserialize)]
pub struct TrackManifest {
    pub bone: usize,
    #[serde(default)]
    pub position: Option<ChannelManifest>,
    #[serde(default)]
    pub rotation: Option<ChannelManifest>,
    #[serde(default)]
    pub scale: Option<ChannelManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipManifest {
    pub name: String,
    pub duration: f32,
    pub sample_rate: f32,
    pub frame_count: usize,
    /// `rename_all = "camelCase"` cannot express this one: the JSON key is the Rust keyword `loop`.
    #[serde(rename = "loop", default)]
    pub looping: bool,
    #[serde(default)]
    pub average_speed: Option<Vec<f32>>,
    pub tracks: Vec<TrackManifest>,
}

/// One node of an extracted Unity blend tree. `kind` is "clip" for a leaf.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BlendNodeManifest {
    pub kind: String,
    #[serde(default)]
    pub clip: Option<i64>,
    #[serde(default)]
    pub param_x: String,
    #[serde(default)]
    pub param_y: String,
    #[serde(default)]
    pub threshold: f32,
    #[serde(default)]
    pub position: Option<[f32; 2]>,
    #[serde(default)]
    pub children: Vec<BlendNodeManifest>,
}

/// One outgoing transition. Carries the graph's OWN cross-fade duration, which is what stops a
/// state change from snapping — notably at a jump apex, where `vy` crossing zero swaps states.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TransitionManifest {
    /// Destination state's leaf name.
    pub target: String,
    #[serde(default)]
    pub duration: f32,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct StateManifest {
    pub name: String,
    pub full_path: String,
    pub layer: usize,
    #[serde(default = "one")]
    pub speed: f32,
    #[serde(default)]
    pub looping: bool,
    #[serde(default)]
    pub trees: Vec<Option<BlendNodeManifest>>,
    #[serde(default)]
    pub transitions: Vec<TransitionManifest>,
}

fn one() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ControllerManifest {
    pub name: String,
    /// For humans. NOT unique — Tagilla's graph has two distinct assets both called
    /// `crouch_run_aim_0`, and only one is an absolute-pose clip.
    #[serde(default)]
    pub clip_names: Vec<String>,
    /// THE authoritative blend-tree-leaf -> clip resolution: index into `clips[]`, or -1 when that
    /// clip was not extracted. Indexed by controller clip id.
    #[serde(default)]
    pub clip_index_by_id: Vec<i64>,
    #[serde(default)]
    pub states: Vec<StateManifest>,
    /// The controller's LAYERS. EFT stacks eleven of them: a base locomotion layer plus additive
    /// ones that add aim, breathing and body turn on top. Aiming lives on `Additive_Aiming`.
    #[serde(default)]
    pub layers: Vec<LayerManifest>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LayerManifest {
    pub index: usize,
    pub name: String,
    /// `"override"` or `"additive"` — additive layers contribute a DELTA against their clip's
    /// first frame rather than an absolute pose.
    #[serde(default)]
    pub blending: String,
    #[serde(default)]
    pub default_weight: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub version: u32,
    pub id: String,
    pub display_name: String,
    pub conventions: Conventions,
    pub skeleton: SkeletonManifest,
    pub vertex_layout: VertexLayout,
    pub default_lod: u32,
    pub meshes: Vec<MeshManifest>,
    #[serde(default)]
    pub attachments: Vec<AttachmentManifest>,
    pub materials: Vec<MaterialManifest>,
    #[serde(default)]
    pub textures: Vec<String>,
    #[serde(default)]
    pub clips: Vec<ClipManifest>,
    #[serde(default)]
    pub controller: Option<ControllerManifest>,
    /// Measured from the forward-walk clip's root motion by the emitter. NOT a hand-authored facing
    /// offset — see `characterForwardDerived`.
    #[serde(default = "fwd_z")]
    pub character_forward: [f32; 3],
    #[serde(default)]
    pub character_forward_derived: bool,
}

fn fwd_z() -> [f32; 3] {
    [0.0, 0.0, 1.0]
}

// ---------------------------------------------------------------------------
// decoded, engine-agnostic pack
// ---------------------------------------------------------------------------
/// One vertex, de-interleaved. Kept as parallel arrays per mesh because that is what
/// `Mesh::insert_attribute` wants.
pub struct MeshData {
    pub name: String,
    /// `"third"` or `"first"` — see [`MeshManifest::view`].
    pub view: String,
    pub lod: u32,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub joint_indices: Vec<[u16; 4]>,
    pub joint_weights: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<SubMeshManifest>,
    pub inverse_bindposes: Vec<Mat4>,
}

/// A resampled per-bone track. Channel vectors are empty when the clip does not drive that channel,
/// in which case the bind-pose value stands.
pub struct BoneTrack {
    pub bone: usize,
    pub position: Vec<Vec3>,
    pub rotation: Vec<Quat>,
    pub scale: Vec<Vec3>,
}

pub struct ClipData {
    pub name: String,
    pub duration: f32,
    pub frame_count: usize,
    pub looping: bool,
    /// Horizontal magnitude of the clip's own root motion (m/s), measured by the emitter. Driving
    /// playback rate from this is what keeps feet planted instead of ice-skating.
    pub root_speed: f32,
    pub tracks: Vec<BoneTrack>,
}

pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    pub local_pos: Vec3,
    pub local_rot: Quat,
    pub local_scale: Vec3,
}

/// Decoded rigid attachment: geometry plus its prefab-local transform under the target bone.
pub struct AttachData {
    pub name: String,
    pub bone: usize,
    pub lod: u32,
    pub local: (Vec3, Quat, Vec3),
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<SubMeshManifest>,
}

pub struct CharacterPack {
    pub root: PathBuf,
    pub id: String,
    pub display_name: String,
    pub bones: Vec<Bone>,
    pub bone_by_name: HashMap<String, usize>,
    pub meshes: Vec<MeshData>,
    pub attachments: Vec<AttachData>,
    pub materials: Vec<MaterialManifest>,
    pub clips: Vec<ClipData>,
    pub clip_by_name: HashMap<String, usize>,
    pub controller: Option<ControllerManifest>,
    pub default_lod: u32,
    pub forward: Vec3,
    pub forward_derived: bool,
}

impl CharacterPack {
    pub fn clip(&self, name: &str) -> Option<&ClipData> {
        self.clip_by_name.get(name).map(|i| &self.clips[*i])
    }

    /// Resolve a controller clip id (blend-tree leaf) to a loaded clip, if that clip was extracted.
    /// A clip set narrower than the controller's full graph legitimately leaves gaps.
    ///
    /// Goes through `clipIndexById`, never through the name: clip names are NOT unique, and the
    /// duplicate is sometimes an additive-delta twin whose poses are wrong to play absolutely.
    pub fn clip_by_controller_id(&self, id: i64) -> Option<&ClipData> {
        let ctrl = self.controller.as_ref()?;
        let idx = *ctrl.clip_index_by_id.get(usize::try_from(id).ok()?)?;
        self.clips.get(usize::try_from(idx).ok()?)
    }

    pub fn state(&self, full_path: &str) -> Option<&StateManifest> {
        self.controller
            .as_ref()?
            .states
            .iter()
            .find(|s| s.full_path == full_path)
    }

    /// The graph's own cross-fade duration for `from -> to`, or `None` if the graph has no such
    /// transition (our state machine takes shortcuts EFT's does not, so that is expected).
    /// `to` is matched on the destination's LEAF name, which is how the emitter records targets.
    pub fn transition_time(&self, from: &str, to: &str) -> Option<f32> {
        let to_leaf = to.rsplit('.').next().unwrap_or(to);
        let s = self.state(from)?;
        s.transitions
            .iter()
            .find(|t| t.target == to_leaf)
            .map(|t| t.duration)
    }
}

fn read_f32s(blob: &[u8], ch: &ChannelManifest, expect_components: usize) -> Result<Vec<f32>> {
    if ch.components != expect_components {
        bail!(
            "channel declares {} components, expected {}",
            ch.components,
            expect_components
        );
    }
    let end = ch.byte_offset + ch.byte_length;
    if end > blob.len() {
        bail!(
            "channel range {}..{} exceeds blob length {}",
            ch.byte_offset,
            end,
            blob.len()
        );
    }
    if ch.byte_length % 4 != 0 {
        bail!("channel byte length {} is not f32-aligned", ch.byte_length);
    }
    let slice = &blob[ch.byte_offset..end];
    Ok(slice
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Load a `.eftchar` directory.
pub fn load(dir: impl AsRef<Path>) -> Result<CharacterPack> {
    let dir = dir.as_ref();
    let mpath = dir.join("manifest.json");
    let text = std::fs::read_to_string(&mpath)
        .with_context(|| format!("reading {}", mpath.display()))?;
    let m: Manifest =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", mpath.display()))?;

    if m.version != SUPPORTED_VERSION {
        bail!(
            "{}: .eftchar version {} but this build supports {}",
            dir.display(),
            m.version,
            SUPPORTED_VERSION
        );
    }
    // Assert the convention the emitter claims, rather than silently trusting it. A pack built
    // without the X-flip would render mirrored and inside-out and this is the cheapest place to say
    // so.
    if m.conventions.g3.as_slice() != [-1.0, 1.0, 1.0] {
        bail!(
            "{}: unexpected coordinate convention g3={:?}; the viewer's world is diag(-1,1,1)",
            dir.display(),
            m.conventions.g3
        );
    }
    if !m.conventions.uv_v_flip_baked {
        bail!(
            "{}: pack reports uvVFlipBaked=false; Unity UVs are bottom-left and this consumer does \
             not re-flip, so textures would sample upside down",
            dir.display()
        );
    }
    if !m.conventions.winding_flipped {
        bail!(
            "{}: pack reports windingFlipped=false; an X-flipped pack must have reversed winding \
             or every triangle faces away",
            dir.display()
        );
    }

    // ---- skeleton ----
    let sk = &m.skeleton;
    let n = sk.bone_count;
    for (label, len) in [
        ("names", sk.names.len()),
        ("paths", sk.paths.len()),
        ("parents", sk.parents.len()),
        ("localPos", sk.local_pos.len()),
        ("localRot", sk.local_rot.len()),
        ("localScale", sk.local_scale.len()),
    ] {
        if len != n {
            bail!("skeleton.{label} has {len} entries, boneCount is {n}");
        }
    }
    let mut bones = Vec::with_capacity(n);
    for i in 0..n {
        let p = sk.parents[i];
        // The emitter guarantees parents[i] < i so a single forward pass suffices; re-check, because
        // the whole pose pipeline depends on it.
        if p >= i as i32 {
            bail!("bone {i} ({}) has parent {p} >= {i}", sk.names[i]);
        }
        let q = sk.local_rot[i];
        bones.push(Bone {
            name: sk.names[i].clone(),
            parent: (p >= 0).then(|| p as usize),
            local_pos: Vec3::from(sk.local_pos[i]),
            local_rot: Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize(),
            local_scale: Vec3::from(sk.local_scale[i]),
        });
    }
    let bone_by_name = bones
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.clone(), i))
        .collect();

    // ---- skin.bin ----
    let skin = std::fs::read(dir.join("skin.bin"))
        .with_context(|| format!("reading {}", dir.join("skin.bin").display()))?;
    let layout = &m.vertex_layout;
    let need = ["position", "normal", "tangent", "uv0", "jointIndex", "jointWeight"];
    for a in need {
        if layout.attr(a).is_none() {
            bail!("vertexLayout is missing attribute {a:?}");
        }
    }
    let off = |name: &str| layout.attr(name).unwrap().offset;
    let (o_pos, o_nrm, o_tan, o_uv, o_ji, o_jw) = (
        off("position"),
        off("normal"),
        off("tangent"),
        off("uv0"),
        off("jointIndex"),
        off("jointWeight"),
    );

    let mut meshes = Vec::with_capacity(m.meshes.len());
    for mm in &m.meshes {
        let vend = mm.vertex_byte_offset + mm.vertex_byte_length;
        if vend > skin.len() {
            bail!("{}: vertex range exceeds skin.bin", mm.name);
        }
        if mm.vertex_byte_length != mm.vertex_count * layout.stride {
            bail!(
                "{}: {} vertices x stride {} != declared {} bytes",
                mm.name,
                mm.vertex_count,
                layout.stride,
                mm.vertex_byte_length
            );
        }
        let vb = &skin[mm.vertex_byte_offset..vend];
        let v = mm.vertex_count;
        let mut positions = Vec::with_capacity(v);
        let mut normals = Vec::with_capacity(v);
        let mut tangents = Vec::with_capacity(v);
        let mut uvs = Vec::with_capacity(v);
        let mut joint_indices = Vec::with_capacity(v);
        let mut joint_weights = Vec::with_capacity(v);
        let f32_at = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let u16_at = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        for i in 0..v {
            let r = &vb[i * layout.stride..(i + 1) * layout.stride];
            positions.push([f32_at(r, o_pos), f32_at(r, o_pos + 4), f32_at(r, o_pos + 8)]);
            normals.push([f32_at(r, o_nrm), f32_at(r, o_nrm + 4), f32_at(r, o_nrm + 8)]);
            tangents.push([
                f32_at(r, o_tan),
                f32_at(r, o_tan + 4),
                f32_at(r, o_tan + 8),
                f32_at(r, o_tan + 12),
            ]);
            uvs.push([f32_at(r, o_uv), f32_at(r, o_uv + 4)]);
            joint_indices.push([
                u16_at(r, o_ji),
                u16_at(r, o_ji + 2),
                u16_at(r, o_ji + 4),
                u16_at(r, o_ji + 6),
            ]);
            joint_weights.push([
                f32_at(r, o_jw),
                f32_at(r, o_jw + 4),
                f32_at(r, o_jw + 8),
                f32_at(r, o_jw + 12),
            ]);
        }

        let iend = mm.index_byte_offset + mm.index_byte_length;
        if iend > skin.len() {
            bail!("{}: index range exceeds skin.bin", mm.name);
        }
        let indices: Vec<u32> = skin[mm.index_byte_offset..iend]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if indices.len() != mm.index_count {
            bail!(
                "{}: decoded {} indices, manifest says {}",
                mm.name,
                indices.len(),
                mm.index_count
            );
        }
        if mm.inverse_bindposes.len() != n {
            bail!(
                "{}: {} inverse bindposes for a {}-bone rig",
                mm.name,
                mm.inverse_bindposes.len(),
                n
            );
        }
        let inverse_bindposes = mm
            .inverse_bindposes
            .iter()
            .map(|row| {
                if row.len() != 16 {
                    return Err(anyhow!("{}: bindpose row has {} floats", mm.name, row.len()));
                }
                // Emitted ROW-major; glam is column-major.
                Ok(Mat4::from_cols_array(&<[f32; 16]>::try_from(row.as_slice()).unwrap())
                    .transpose())
            })
            .collect::<Result<Vec<_>>>()?;

        meshes.push(MeshData {
            name: mm.name.clone(),
            view: mm.view.clone(),
            lod: mm.lod,
            positions,
            normals,
            tangents,
            uvs,
            joint_indices,
            joint_weights,
            indices,
            submeshes: mm
                .submeshes
                .iter()
                .map(|s| SubMeshManifest {
                    material: s.material,
                    index_start: s.index_start,
                    index_count: s.index_count,
                })
                .collect(),
            inverse_bindposes,
        });
    }

    // ---- attachments (same blob, same vertex layout) ----
    let mut attachments: Vec<AttachData> = Vec::with_capacity(m.attachments.len());
    for am in &m.attachments {
        let vend = am.vertex_byte_offset + am.vertex_byte_length;
        if vend > skin.len() {
            bail!("attachment {}: vertex range exceeds skin.bin", am.name);
        }
        if am.vertex_byte_length != am.vertex_count * layout.stride {
            bail!(
                "attachment {}: {} vertices x stride {} != declared {} bytes",
                am.name,
                am.vertex_count,
                layout.stride,
                am.vertex_byte_length
            );
        }
        if am.bone >= n {
            bail!("attachment {}: targets bone {} of {}", am.name, am.bone, n);
        }
        let vb = &skin[am.vertex_byte_offset..vend];
        let v = am.vertex_count;
        let f32_at = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let mut positions = Vec::with_capacity(v);
        let mut normals = Vec::with_capacity(v);
        let mut tangents = Vec::with_capacity(v);
        let mut uvs = Vec::with_capacity(v);
        for i in 0..v {
            let r = &vb[i * layout.stride..(i + 1) * layout.stride];
            positions.push([f32_at(r, o_pos), f32_at(r, o_pos + 4), f32_at(r, o_pos + 8)]);
            normals.push([f32_at(r, o_nrm), f32_at(r, o_nrm + 4), f32_at(r, o_nrm + 8)]);
            tangents.push([
                f32_at(r, o_tan),
                f32_at(r, o_tan + 4),
                f32_at(r, o_tan + 8),
                f32_at(r, o_tan + 12),
            ]);
            uvs.push([f32_at(r, o_uv), f32_at(r, o_uv + 4)]);
        }
        let iend = am.index_byte_offset + am.index_byte_length;
        if iend > skin.len() {
            bail!("attachment {}: index range exceeds skin.bin", am.name);
        }
        let indices: Vec<u32> = skin[am.index_byte_offset..iend]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let q = am.local_rot;
        attachments.push(AttachData {
            name: am.name.clone(),
            bone: am.bone,
            lod: am.lod,
            local: (
                Vec3::from(am.local_pos),
                Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize(),
                Vec3::from(am.local_scale),
            ),
            positions,
            normals,
            tangents,
            uvs,
            indices,
            submeshes: am
                .submeshes
                .iter()
                .map(|s| SubMeshManifest {
                    material: s.material,
                    index_start: s.index_start,
                    index_count: s.index_count,
                })
                .collect(),
        });
    }

    // ---- anim.bin ----
    let anim_path = dir.join("anim.bin");
    let anim = if m.clips.is_empty() {
        Vec::new()
    } else {
        std::fs::read(&anim_path).with_context(|| format!("reading {}", anim_path.display()))?
    };
    let mut clips = Vec::with_capacity(m.clips.len());
    for cm in &m.clips {
        let frames = cm.frame_count;
        let mut tracks = Vec::with_capacity(cm.tracks.len());
        for tm in &cm.tracks {
            if tm.bone >= n {
                bail!("clip {}: track targets bone {} of {}", cm.name, tm.bone, n);
            }
            let mut position = Vec::new();
            let mut rotation = Vec::new();
            let mut scale = Vec::new();
            if let Some(ch) = &tm.position {
                let f = read_f32s(&anim, ch, 3)
                    .with_context(|| format!("clip {} bone {} position", cm.name, tm.bone))?;
                if f.len() != frames * 3 {
                    bail!("clip {}: position has {} floats, want {}", cm.name, f.len(), frames * 3);
                }
                position = f.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
            }
            if let Some(ch) = &tm.rotation {
                let f = read_f32s(&anim, ch, 4)
                    .with_context(|| format!("clip {} bone {} rotation", cm.name, tm.bone))?;
                if f.len() != frames * 4 {
                    bail!("clip {}: rotation has {} floats, want {}", cm.name, f.len(), frames * 4);
                }
                rotation = f
                    .chunks_exact(4)
                    .map(|c| Quat::from_xyzw(c[0], c[1], c[2], c[3]).normalize())
                    .collect();
            }
            if let Some(ch) = &tm.scale {
                let f = read_f32s(&anim, ch, 3)
                    .with_context(|| format!("clip {} bone {} scale", cm.name, tm.bone))?;
                if f.len() != frames * 3 {
                    bail!("clip {}: scale has {} floats, want {}", cm.name, f.len(), frames * 3);
                }
                scale = f.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
            }
            tracks.push(BoneTrack { bone: tm.bone, position, rotation, scale });
        }
        let root_speed = cm
            .average_speed
            .as_ref()
            .filter(|v| v.len() >= 3)
            .map(|v| Vec3::new(v[0], 0.0, v[2]).length())
            .unwrap_or(0.0);
        clips.push(ClipData {
            name: cm.name.clone(),
            duration: cm.duration,
            frame_count: frames,
            looping: cm.looping,
            root_speed,
            tracks,
        });
    }
    let clip_by_name = clips
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.clone(), i))
        .collect();

    let fwd = Vec3::from(m.character_forward);
    Ok(CharacterPack {
        attachments,
        root: dir.to_path_buf(),
        id: m.id,
        display_name: m.display_name,
        bones,
        bone_by_name,
        meshes,
        materials: m.materials,
        clips,
        clip_by_name,
        controller: m.controller,
        default_lod: m.default_lod,
        forward: if fwd.length_squared() > 1e-6 { fwd.normalize() } else { Vec3::Z },
        forward_derived: m.character_forward_derived,
    })
}
