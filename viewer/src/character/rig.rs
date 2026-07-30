//! Turning a loaded [`CharacterPack`] into live Bevy entities.
//!
//! Shape of what gets spawned:
//!
//! ```text
//! CharacterRoot (Transform: feet position + facing)
//!  ├── bone 0 "Skeleton" ── bone 1 "Root_Joint" ── ... 79 CharacterBone entities
//!  └── one entity per (mesh, submesh): Mesh3d + MeshMaterial3d + SkinnedMesh
//! ```
//!
//! Every skinned entity shares the SAME `joints` list — the whole rig in bone order — because the
//! emitter rewrote joint indices into global rig indices and padded each mesh's inverse-bindpose
//! table to rig size. That is the reason assembling a character is "spawn the rig once, attach N
//! meshes" rather than a per-part bone mapping at runtime.
//!
//! Bevy has no submesh concept, so each Unity submesh becomes its own `Mesh` asset carrying the
//! part's full vertex buffer and only its own index range. The duplicated vertices cost ~600 KB for
//! Tagilla, which is not worth avoiding.

use super::pack::{CharacterPack, MeshData};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::image::{Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use std::collections::HashMap;

/// Marker + runtime state on the character's root entity.
#[derive(Component)]
pub struct CharacterRoot {
    /// Bone entities in rig order; index i is bone i of the pack's skeleton.
    pub bones: Vec<Entity>,
    /// Currently playing state's full path, and how long it has been playing.
    pub state: String,
    pub state_time: f32,
    /// The state being cross-faded OUT of, its clock, and the fade's progress/length in seconds.
    /// Without this every state change is a hard cut — most visible at a jump apex, where `vy`
    /// crossing zero swaps states and the body snapped to the incoming clip's first frame.
    pub prev_state: String,
    pub prev_time: f32,
    pub fade: f32,
    pub fade_len: f32,
    /// The BODY's facing yaw (radians), independent of the camera. Keeping this separate is what
    /// makes right-drag orbit the camera around the character instead of spinning the character:
    /// previously body yaw was simply read off the camera, so the two could never differ.
    pub heading: f32,
    /// False until the first frame has seeded `heading` from the camera.
    pub heading_init: bool,
    /// Scratch: resolved local transforms, reused every frame to avoid per-frame allocation.
    pub locals: Vec<(Vec3, Quat, Vec3)>,
}

/// Marker on each bone entity, carrying its rig index.
///
/// The pose writer walks `CharacterRoot::bones` by position so it does not need to read this, but
/// keeping the index ON the entity is what lets anything else (an inspector, a future IK pass, an
/// attachment socket) identify a bone without the root's lookup table.
#[derive(Component)]
pub struct CharacterBone(#[allow(dead_code)] pub usize);

/// Marker on the skinned mesh entities, so a teardown can find them.
#[derive(Component)]
pub struct CharacterMesh;

fn load_texture(
    dir: &std::path::Path,
    rel: &str,
    srgb: bool,
    images: &mut Assets<Image>,
    cache: &mut HashMap<(String, bool), Option<Handle<Image>>>,
) -> Option<Handle<Image>> {
    let key = (rel.to_string(), srgb);
    if let Some(h) = cache.get(&key) {
        return h.clone();
    }
    let path = dir.join(rel);
    let handle = match image::open(&path) {
        Ok(img) => {
            let mut image = Image::from_dynamic(img, srgb, RenderAssetUsages::default());
            // Character UVs stay inside 0..1, but Repeat costs nothing and matches the map path's
            // sampler so the two never disagree on a shared texture.
            image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                mag_filter: ImageFilterMode::Linear,
                min_filter: ImageFilterMode::Linear,
                mipmap_filter: ImageFilterMode::Linear,
                anisotropy_clamp: 16,
                ..default()
            });
            Some(images.add(image))
        }
        Err(e) => {
            warn!("character texture {} failed to load: {e}", path.display());
            None
        }
    };
    cache.insert(key, handle.clone());
    handle
}

fn build_mesh(md: &MeshData, index_start: usize, index_count: usize) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, md.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, md.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, md.uvs.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, md.tangents.clone());
    // ATTRIBUTE_JOINT_INDEX is Uint16x4, which has no blanket `From<Vec<[u16; 4]>>` — name the
    // variant explicitly or it silently will not compile against the right format.
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(md.joint_indices.clone()),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, md.joint_weights.clone());
    let end = (index_start + index_count).min(md.indices.len());
    mesh.insert_indices(Indices::U32(md.indices[index_start..end].to_vec()));
    mesh
}

/// Spawn the character. Returns the root entity.
pub fn spawn(
    pack: &CharacterPack,
    lod: u32,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    ibms: &mut Assets<SkinnedMeshInverseBindposes>,
) -> Entity {
    // ---- materials ----
    let mut tex_cache: HashMap<(String, bool), Option<Handle<Image>>> = HashMap::new();
    let mat_handles: Vec<Handle<StandardMaterial>> = pack
        .materials
        .iter()
        .map(|m| {
            let base = m
                .textures
                .get("_MainTex")
                .and_then(|p| load_texture(&pack.root, p, true, images, &mut tex_cache));
            // Normal and spec are DATA, not colour: they must be sampled linearly.
            let normal = m
                .textures
                .get("_BumpMap")
                .and_then(|p| load_texture(&pack.root, p, false, images, &mut tex_cache));
            let spec = m
                .textures
                .get("_SpecMap")
                .and_then(|p| load_texture(&pack.root, p, false, images, &mut tex_cache));
            materials.add(StandardMaterial {
                base_color_texture: base,
                normal_map_texture: normal,
                // BSG's character shader is not PBR metal/rough; the spec map drives a Blinn-ish
                // highlight. Feeding it to occlusion is a deliberate stand-in that keeps cloth from
                // reading as wet plastic, not a claim of parity with the game's shading.
                occlusion_texture: spec,
                perceptual_roughness: 0.75,
                metallic: 0.0,
                ..default()
            })
        })
        .collect();

    // ---- root ----
    let root = commands
        .spawn((
            Transform::IDENTITY,
            Visibility::default(),
            Name::new(format!("character:{}", pack.id)),
        ))
        .id();

    // ---- bones ----
    // Reserve ids first so a child can name its parent regardless of spawn order (parents[i] < i is
    // asserted by the loader, but reserving keeps this independent of that).
    let bone_entities: Vec<Entity> = (0..pack.bones.len())
        .map(|_| commands.spawn_empty().id())
        .collect();
    for (i, bone) in pack.bones.iter().enumerate() {
        let e = bone_entities[i];
        let parent = bone.parent.map(|p| bone_entities[p]).unwrap_or(root);
        commands.entity(e).insert((
            Transform {
                translation: bone.local_pos,
                rotation: bone.local_rot,
                scale: bone.local_scale,
            },
            Visibility::default(),
            CharacterBone(i),
            Name::new(bone.name.clone()),
            ChildOf(parent),
        ));
    }

    // ---- skinned meshes ----
    let mut spawned_meshes = 0usize;
    for md in &pack.meshes {
        if md.lod != lod {
            continue;
        }
        let ibm = ibms.add(SkinnedMeshInverseBindposes::from(md.inverse_bindposes.clone()));
        for sub in &md.submeshes {
            if sub.index_count == 0 {
                continue;
            }
            let mesh = meshes.add(build_mesh(md, sub.index_start, sub.index_count));
            let material = mat_handles
                .get(sub.material)
                .cloned()
                .unwrap_or_else(|| materials.add(StandardMaterial::default()));
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                // Identity: the bindposes already live in rig space and the joints are this root's
                // descendants, so any transform here would double-apply.
                Transform::IDENTITY,
                Visibility::default(),
                SkinnedMesh { inverse_bindposes: ibm.clone(), joints: bone_entities.clone() },
                // A skinned mesh's Aabb is computed from the BIND pose; an animated character
                // reaching outside it would pop out of view. One character is not worth culling.
                NoFrustumCulling,
                CharacterMesh,
                Name::new(md.name.clone()),
                ChildOf(root),
            ));
            spawned_meshes += 1;
        }
    }

    // ---- rigid attachments (helmet / facecover / cap) ----
    // Unskinned geometry parented to a bone entity, carrying its prefab-local transform. Bevy's
    // hierarchy does the rest, so an attachment follows the head through every animation with no
    // per-frame work here.
    let mut spawned_attachments = 0usize;
    for att in &pack.attachments {
        if att.lod != lod {
            continue;
        }
        let Some(parent) = bone_entities.get(att.bone).copied() else { continue };
        for sub in &att.submeshes {
            if sub.index_count == 0 {
                continue;
            }
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, att.positions.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, att.normals.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, att.uvs.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, att.tangents.clone());
            let end = (sub.index_start + sub.index_count).min(att.indices.len());
            mesh.insert_indices(Indices::U32(att.indices[sub.index_start..end].to_vec()));
            let material = mat_handles
                .get(sub.material)
                .cloned()
                .unwrap_or_else(|| materials.add(StandardMaterial::default()));
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material),
                Transform {
                    translation: att.local.0,
                    rotation: att.local.1,
                    scale: att.local.2,
                },
                Visibility::default(),
                NoFrustumCulling,
                CharacterMesh,
                Name::new(att.name.clone()),
                ChildOf(parent),
            ));
            spawned_attachments += 1;
        }
    }

    let bone_count = pack.bones.len();
    commands.entity(root).insert(CharacterRoot {
        bones: bone_entities,
        state: String::new(),
        state_time: 0.0,
        prev_state: String::new(),
        prev_time: 0.0,
        fade: 1.0,
        fade_len: 0.0,
        heading: 0.0,
        heading_init: false,
        locals: vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); bone_count],
    });

    info!(
        "character '{}' spawned: {} bones, {} skinned draws, {} attachment draws, {} clips, forward={:?}{}",
        pack.display_name,
        bone_count,
        spawned_meshes,
        spawned_attachments,
        pack.clips.len(),
        pack.forward,
        if pack.forward_derived { "" } else { " (NOT derived — fell back to +Z)" }
    );
    root
}
