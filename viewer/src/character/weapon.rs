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
    /// Per-material scalars/colours as the game's shader names them. See [`MatProps`].
    #[serde(rename = "materialProps", default)]
    material_props: HashMap<String, MatProps>,
    /// Where the eye goes when aiming. See [`AimAnchor`].
    #[serde(default)]
    aim: Option<AimAnchor>,
}

/// The sight's own eye anchor, in the assembled weapon's space.
///
/// Taken from `OpticSight.ScopeTransform` in the sight prefab — the node the game aligns the eye
/// to when the sight comes up — so aiming needs no authored offsets. `fov` is that optic's
/// magnification as a field of view in degrees (`ScopeCameraData.FieldOfView`); it describes the
/// image rendered THROUGH the lens, not the whole screen.
#[derive(Debug, Deserialize, Clone)]
// The manifest is camelCase like every other block in it. Without this the multi-word fields
// silently deserialize to their defaults -- which is how the optic's own surface names went
// missing and every lens fell back to being half-opaque.
#[serde(rename_all = "camelCase")]
pub struct AimAnchor {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    #[serde(default)]
    pub fov: Option<f32>,
    /// `OpticSight.DistanceToCamera` — the game's own eye relief, metres.
    #[serde(default)]
    pub eye_relief: Option<f32>,
    /// `OpticSight.LensRenderer` / `DecorLensRenderer`, by material name.
    #[serde(default)]
    pub lens_material: Option<String>,
    #[serde(default)]
    pub decor_material: Option<String>,
    /// Every material inside the optic's own mode subtree.
    #[serde(default)]
    pub optic_materials: Vec<String>,
}

/// A material's raw properties, straight from the game.
///
/// EFT's transparent-reflective family (the scope lenses, vehicle glass) carries its OPACITY in
/// `_Color.a` — the EOTech's is 0.128 — with `_ReflectColor` / `_SpecColor` / `_Shininess`
/// describing the reflection. A pack built before these were exported simply has none, and every
/// material stays opaque, which is the old behaviour.
#[derive(Debug, Deserialize, Default)]
struct MatProps {
    #[serde(default)]
    floats: HashMap<String, f32>,
    #[serde(default)]
    colors: HashMap<String, [f32; 4]>,
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
    /// Present when the build carries an optic that declares a scope transform.
    pub aim: Option<AimAnchor>,
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
        // ---- the game's own material description ----------------------------------------
        // Everything below is keyed off PROPERTIES, never off the material's name: BSG names the
        // same shader `_glass`, `_linza` and `mag_glass` interchangeably, and `mag_glass` is
        // actually opaque. What a surface IS shows up in which properties it carries.
        let props = man.material_props.get(&mat_name);
        let f = |k: &str| props.and_then(|p| p.floats.get(k)).copied();
        let c = |k: &str| props.and_then(|p| p.colors.get(k)).copied();
        let color = c("_Color").unwrap_or([1.0, 1.0, 1.0, 1.0]);

        // A RETICLE lens carries `_MarkTex` — the reticle image, whose alpha is its shape — with
        // `_Color` as its tint and `_HDR` as its overdrive. It is a projected light, not a lit
        // surface, so it is drawn unlit and emissive; lighting it would let shadow darken a
        // holographic sight.
        let mark = slots.and_then(|s| s.get("_MarkTex")).and_then(|r| load_tex(r, true, images));
        let is_reticle = mark.is_some();
        // GLASS: either the game states the opacity outright (`_Color.a < 1`, the EXPS3 window at
        // 0.128), or the material has no `_MainTex` at all and shades itself from a reflection
        // probe (`_EnvTex`/`_Cube`) — the fresnel lens family, whose opacity is the reflection
        // blend in `_ReflectColor.a`.
        let has_env = slots.is_some_and(|s| s.contains_key("_EnvTex") || s.contains_key("_Cube"));
        // WHAT YOU LOOK THROUGH. The optic's own components name its surfaces: `LensRenderer`
        // carries the sight picture and `DecorLensRenderer` its glass. The third one is the
        // BACKING (`back_linza`) — authored opaque black with no textures at all, because in the
        // game it is the surface the optic's camera renders the magnified image onto. Drawn as
        // authored it is simply a black disc that blocks the sight completely, which is why the
        // magnifier could not be seen through. A material inside the optic that carries no
        // texture is a render target, not paint.
        let is_optic_glass = man.aim.as_ref().is_some_and(|a| {
            Some(&mat_name) == a.lens_material.as_ref()
                || Some(&mat_name) == a.decor_material.as_ref()
                || (a.optic_materials.iter().any(|m| *m == mat_name)
                    && slots.is_none_or(|s| s.is_empty()))
        });
        let refl = c("_ReflectColor").unwrap_or([1.0, 1.0, 1.0, 1.0]);
        let stated_alpha = color[3];
        let is_glass = !is_reticle && (is_optic_glass || stated_alpha < 0.999 || (base.is_none() && has_env));
        // An optic surface you look through keeps only a faint tint; its authored alpha describes
        // a lens with the scope image behind it, which we do not render yet.
        let alpha = if is_optic_glass {
            0.12
        } else if stated_alpha < 0.999 {
            stated_alpha
        } else {
            refl[3]
        };

        // The lens TINT is `_MainColor` where the fresnel family provides it (the G33's blue-grey),
        // otherwise `_Color`. Falling back to white here is what made every untextured lens — and
        // the black `back_linza` backing disc — render as a solid white pane.
        let tint = c("_MainColor").unwrap_or(color);
        let base_color = if is_reticle {
            Color::srgba(color[0], color[1], color[2], 1.0)
        } else if is_glass {
            Color::srgba(tint[0], tint[1], tint[2], alpha.clamp(0.05, 0.95))
        } else {
            Color::srgba(color[0], color[1], color[2], 1.0)
        };
        // `_Shininess`/`_Glossiness` are BSG GLOSS: high = shiny, the inverse of Bevy roughness.
        let gloss = f("_Shininess").or_else(|| f("_Glossiness")).unwrap_or(0.45);
        let roughness = (1.0 - gloss.clamp(0.0, 1.0)).clamp(0.03, 1.0);
        let hdr = f("_HDR").unwrap_or(1.0).max(1.0);
        let mat = materials.add(StandardMaterial {
            base_color,
            base_color_texture: if is_reticle { mark } else { base },
            normal_map_texture: if is_reticle { None } else { normal },
            // A reticle emits; `_HDR` is how hard. Bevy wants linear emissive, and the tint is
            // already in `_Color`.
            emissive: if is_reticle {
                LinearRgba::rgb(color[0] * hdr, color[1] * hdr, color[2] * hdr)
            } else {
                LinearRgba::BLACK
            },
            unlit: is_reticle,
            // The _SpecMap is BSG GLOSS (high = shiny), the inverse of Bevy roughness, so it is
            // deliberately NOT bound (binding a gloss map to occlusion once crushed all
            // character ambient — see character/rig.rs).
            perceptual_roughness: if is_glass { 0.05 } else { roughness },
            metallic: 0.0,
            // Glass reflects hard at grazing angles; without this a lens reads as a flat tinted
            // hole rather than a curved piece of glass.
            reflectance: if is_glass { 0.9 } else { 0.5 },
            alpha_mode: if is_reticle || is_glass { AlphaMode::Blend } else { AlphaMode::Opaque },
            // A lens is a thin double-sided shell; culling its back face shows the inside of the
            // housing through it.
            double_sided: is_reticle || is_glass,
            cull_mode: if is_reticle || is_glass {
                None
            } else {
                Some(bevy::render::render_resource::Face::Back)
            },
            ..default()
        });
        parts.push((meshes.add(mesh), mat));
    }
    if parts.is_empty() {
        return None;
    }
    info!("weapon '{}': {} part(s) from {}", man.name, parts.len(), dir.display());
    Some(WeaponPack { aim: man.aim.clone(), name: man.name, parts })
}

/// `out/weapons/<id>` unless overridden.
pub fn weapon_dir(id: &str) -> PathBuf {
    PathBuf::from("out").join("weapons").join(id.trim())
}
