//! character::weapon — load an `.eftweap` pack and hang it on the rig's weapon socket.
//!
//! `.eftweap` is what `extraction/characters/build_weapon.py` emits: the item's prefab tree
//! (BSG's own factory preset, or a bot's rolled mod tree) baked into ONE merged mesh at the
//! game's own attachment nodes, plus per-submesh materials and their textures. Loading it is a
//! straight read of the manifest's declared layout — same self-describing contract as `.eftpack`
//! and `.eftchar`, so emitter and consumer cannot drift.
//!
//! The mesh is parented to the rig bone named `Weapon_root` (the game's own socket — the rig also
//! ships `weapon_holster` for the slung pose), so it follows the animation with no extra work.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The rig's weapon socket (verified present in the extracted skeleton at index 68).
pub const WEAPON_BONE: &str = "Weapon_root";

#[derive(Debug, Deserialize)]
struct WeapManifest {
    #[serde(default)]
    name: String,
    #[serde(rename = "vertexCount")]
    vertex_count: usize,
    #[serde(rename = "indexCount")]
    index_count: usize,
    vertex: VertexLayout,
    submeshes: Vec<SubMesh>,
    materials: Vec<String>,
    #[serde(rename = "materialTextures", default)]
    material_textures: HashMap<String, HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct VertexLayout {
    stride: usize,
}

#[derive(Debug, Deserialize)]
struct SubMesh {
    material: usize,
    #[serde(rename = "idxStart")]
    idx_start: usize,
    #[serde(rename = "idxCount")]
    idx_count: usize,
}

/// One loaded weapon: a mesh + material per submesh, ready to spawn as children of a bone.
pub struct WeaponPack {
    pub name: String,
    pub parts: Vec<(Handle<Mesh>, Handle<StandardMaterial>)>,
}

/// Read `<dir>/manifest.json` + `mesh.bin` into Bevy assets.
pub fn load(
    dir: &Path,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
) -> Option<WeaponPack> {
    let man: WeapManifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).ok()?).ok()?;
    let blob = std::fs::read(dir.join("mesh.bin")).ok()?;
    let stride = man.vertex.stride;
    let vbytes = man.vertex_count * stride;
    if blob.len() < vbytes + man.index_count * 4 {
        warn!("weapon {}: mesh.bin shorter than the manifest declares", dir.display());
        return None;
    }
    // pos f32x3 @0, nrm f32x3 @12, uv f32x2 @24 (the emitter's declared layout).
    let mut pos = Vec::with_capacity(man.vertex_count);
    let mut nrm = Vec::with_capacity(man.vertex_count);
    let mut uv = Vec::with_capacity(man.vertex_count);
    let f = |o: usize| f32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]);
    for i in 0..man.vertex_count {
        let b = i * stride;
        pos.push([f(b), f(b + 4), f(b + 8)]);
        nrm.push([f(b + 12), f(b + 16), f(b + 20)]);
        uv.push([f(b + 24), f(b + 28)]);
    }
    let idx_base = vbytes;
    let all_idx: Vec<u32> = (0..man.index_count)
        .map(|i| {
            let o = idx_base + i * 4;
            u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]])
        })
        .collect();

    let mut tex_cache: HashMap<String, Option<Handle<Image>>> = HashMap::new();
    let mut load_tex = |rel: &str, srgb: bool, images: &mut Assets<Image>| -> Option<Handle<Image>> {
        tex_cache
            .entry(format!("{rel}:{srgb}"))
            .or_insert_with(|| {
                image::open(dir.join(rel))
                    .ok()
                    .map(|img| images.add(Image::from_dynamic(img, srgb, RenderAssetUsages::RENDER_WORLD)))
            })
            .clone()
    };

    let mut parts = Vec::new();
    for sm in &man.submeshes {
        if sm.idx_count == 0 || sm.idx_start + sm.idx_count > all_idx.len() {
            continue;
        }
        // Re-index this submesh into its own compact mesh (Bevy draws one material per mesh).
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let (mut p, mut n, mut t, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &gi in &all_idx[sm.idx_start..sm.idx_start + sm.idx_count] {
            let gi_u = gi as usize;
            if gi_u >= pos.len() {
                continue;
            }
            let local = *remap.entry(gi).or_insert_with(|| {
                p.push(pos[gi_u]);
                n.push(nrm[gi_u]);
                t.push(uv[gi_u]);
                (p.len() - 1) as u32
            });
            idx.push(local);
        }
        if idx.is_empty() {
            continue;
        }
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, p);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, n);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, t);
        mesh.insert_indices(Indices::U32(idx));
        let mat_name = man.materials.get(sm.material).cloned().unwrap_or_default();
        let slots = man.material_textures.get(&mat_name);
        let base = slots
            .and_then(|s| s.get("_MainTex"))
            .and_then(|r| load_tex(r, true, images));
        let normal = slots
            .and_then(|s| s.get("_BumpMap"))
            .and_then(|r| load_tex(r, false, images));
        let mat = materials.add(StandardMaterial {
            base_color_texture: base,
            normal_map_texture: normal,
            // The _SpecMap is BSG GLOSS (high = shiny), the inverse of Bevy roughness, so it is
            // deliberately NOT bound (binding a gloss map to occlusion once crushed all
            // character ambient — see character/rig.rs). Constant until the extractor inverts it.
            perceptual_roughness: 0.55,
            metallic: 0.0,
            ..default()
        });
        parts.push((meshes.add(mesh), mat));
    }
    if parts.is_empty() {
        return None;
    }
    info!("weapon '{}': {} part(s) from {}", man.name, parts.len(), dir.display());
    Some(WeaponPack { name: man.name, parts })
}

/// `out/weapons/<id>` unless overridden.
pub fn weapon_dir(id: &str) -> PathBuf {
    PathBuf::from("out").join("weapons").join(id.trim())
}
