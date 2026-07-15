//! standard.rs — BEVY STANDARD MESH-PATH renderer for a `.eftpack`.
//!
//! Renders the pack as ordinary `Mesh3d` + `MeshMaterial3d(StandardMaterial)`
//! entities, one per (instance × submesh), placing each with the instance's FULL
//! affine via `Transform::from_matrix`. Unlike the custom `gpu_driven` path (which
//! bypasses Bevy's mesh/material/prepass systems entirely), this path flows through
//! Bevy's PBR pipeline — so the full built-in lighting stack (cascaded shadow maps,
//! SSAO, SSR, volumetric fog, and the experimental Solari RTX GI) applies to the
//! map geometry. Selected with `EFT_RENDER=std`.
//!
//! Tradeoff: ~110k entities / ~11k mesh assets vs. the custom path's single
//! GPU-driven multidraw. That's the deliberate cost of getting Bevy's lighting.

use crate::eftpack::{Material as PackMaterial, Pack};
use crate::render::LoadedPack;
use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::light::CascadeShadowConfigBuilder;
use bevy::mesh::Indices;
use bevy::pbr::ScreenSpaceAmbientOcclusion;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::view::NoIndirectDrawing;
use glam::Mat4;
use std::collections::HashMap;

/// Marker for every spawned map-geometry entity (so lighting/debug systems can
/// query them, and so a future teardown can despawn the whole map).
#[derive(Component)]
pub struct EftGeom;

pub struct EftStandardPlugin;
impl Plugin for EftStandardPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_standard)
            // PostStartup so the camera + sun (spawned in main's Startup `setup`)
            // already exist when we turn on shadows / attach effects.
            .add_systems(PostStartup, configure_lighting);
    }
}

/// Phase 2: enable Bevy's real-time lighting on the migrated map — cascaded sun
/// shadows + a sky ambient fill so shadowed areas read (the custom path's SH-GI is
/// gone on this path; Phase 3 Solari brings dynamic GI back). Only runs on the
/// Standard path, so the custom paths are untouched.
fn configure_lighting(
    mut commands: Commands,
    cam: Query<Entity, With<Camera3d>>,
    mut lights: Query<(Entity, &mut DirectionalLight)>,
    mut ambient: ResMut<AmbientLight>,
) {
    // Cool sky-fill ambient so shadows aren't crushed to black.
    ambient.color = Color::srgb(0.62, 0.70, 0.90);
    ambient.brightness = 500.0;

    for (e, mut dl) in &mut lights {
        dl.shadows_enabled = true;
        dl.illuminance = 11_000.0;
        // Cascades sized for a large map: tight near cascade for crisp contact
        // shadows, out to 500 m so mid-range geometry still shadows.
        commands.entity(e).insert(
            CascadeShadowConfigBuilder {
                num_cascades: 4,
                minimum_distance: 0.5,
                maximum_distance: 500.0,
                first_cascade_far_bound: 24.0,
                overlap_proportion: 0.2,
            }
            .build(),
        );
    }

    // The camera was tagged NoIndirectDrawing for the custom path; the standard
    // path wants Bevy's GPU preprocessing (faster for 100k+ entities), so drop it.
    // SSAO adds contact/crevice shadows for a grounded look (it auto-pulls in the
    // depth + normal prepass it needs).
    for e in &cam {
        commands
            .entity(e)
            .remove::<NoIndirectDrawing>()
            .insert(ScreenSpaceAmbientOcclusion::default());
    }
}

/// Decode one texture (`image` crate) into a Bevy `Image`, cached by path. Normal
/// maps load LINEAR (`is_srgb=false`) and, when the material declares DirectX
/// convention (`normalGreenFlip`), get their green channel inverted here — Bevy's
/// StandardMaterial has no runtime flip-Y, so we bake it at load (matches the
/// gpu_driven path's in-shader negate).
fn load_texture(
    path: &str,
    srgb: bool,
    flip_green: bool,
    images: &mut Assets<Image>,
    cache: &mut HashMap<(String, bool), Option<Handle<Image>>>,
) -> Option<Handle<Image>> {
    let key = (path.to_string(), flip_green);
    if let Some(h) = cache.get(&key) {
        return h.clone();
    }
    let handle = match image::open(path) {
        Ok(img) => {
            let dyn_img = if flip_green {
                let mut rgba = img.to_rgba8();
                for px in rgba.pixels_mut() {
                    px[1] = 255 - px[1];
                }
                image::DynamicImage::ImageRgba8(rgba)
            } else {
                img
            };
            let mut image = Image::from_dynamic(dyn_img, srgb, RenderAssetUsages::default());
            // EFT bakes UV TILING into the vertex UVs (values well past 0..1 on large
            // surfaces), so textures MUST wrap — Bevy's default ClampToEdge smears the
            // last texel row into long streaks. Repeat + trilinear + anisotropy fixes
            // the stretching and softens grazing-angle aliasing.
            image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                address_mode_w: ImageAddressMode::Repeat,
                mag_filter: ImageFilterMode::Linear,
                min_filter: ImageFilterMode::Linear,
                mipmap_filter: ImageFilterMode::Linear,
                anisotropy_clamp: 16,
                ..default()
            });
            Some(images.add(image))
        }
        Err(e) => {
            warn!("standard: texture load failed {path}: {e}");
            None
        }
    };
    cache.insert(key, handle.clone());
    handle
}

/// Map one pack material → a Bevy `StandardMaterial`.
fn build_material(
    m: &PackMaterial,
    images: &mut Assets<Image>,
    tex_cache: &mut HashMap<(String, bool), Option<Handle<Image>>>,
) -> StandardMaterial {
    let base_color_texture = m
        .albedo
        .as_deref()
        .and_then(|p| load_texture(p, true, false, images, tex_cache));
    let normal_map_texture = m
        .normal
        .as_deref()
        .and_then(|p| load_texture(p, false, m.normal_green_flip, images, tex_cache));

    let (emissive, emissive_texture) = match &m.emissive {
        Some(e) => (
            LinearRgba::new(e.factor[0], e.factor[1], e.factor[2], 1.0),
            e.texture
                .as_deref()
                .and_then(|p| load_texture(p, true, false, images, tex_cache)),
        ),
        None => (LinearRgba::BLACK, None),
    };

    // Alpha is driven by the material ROLE (authoritative), not the raw alphaMode:
    //   cutout -> Mask (foliage/fences; the albedo alpha IS the coverage mask),
    //   glass/decal/water -> Blend, everything else Opaque.
    let alpha_mode = match m.role.as_str() {
        "cutout" => AlphaMode::Mask(if m.alpha_cutoff > 0.0 { m.alpha_cutoff } else { 0.5 }),
        "glass" | "decal" | "water" => AlphaMode::Blend,
        _ => AlphaMode::Opaque,
    };

    // WATER: the "albedo" is a noise texture, not colour — sampled under flat
    // lighting + a smooth specular it reads WHITE. Render it as a dark, glossy,
    // translucent sheet instead (matches the gpu_driven path's wet-sheen), dropping
    // the noise albedo so nothing blows out.
    let is_water = m.role == "water";
    let base_color = if is_water {
        Color::srgba(0.015, 0.035, 0.045, m.tint[3].clamp(0.4, 0.9))
    } else {
        Color::linear_rgba(m.tint[0], m.tint[1], m.tint[2], m.tint[3])
    };

    StandardMaterial {
        base_color,
        base_color_texture: if is_water { None } else { base_color_texture },
        normal_map_texture,
        emissive,
        emissive_texture,
        perceptual_roughness: if is_water {
            0.08
        } else {
            m.roughness.unwrap_or(0.7).clamp(0.05, 1.0)
        },
        metallic: m.metallic.unwrap_or(0.0),
        alpha_mode,
        double_sided: m.double_sided,
        cull_mode: if m.double_sided { None } else { Some(bevy::render::render_resource::Face::Back) },
        // Decals are coplanar overlays; bias them toward the camera to stop z-fighting.
        depth_bias: if m.role == "decal" { 4.0 } else { 0.0 },
        ..default()
    }
}

fn spawn_standard(
    mut commands: Commands,
    pack: Option<Res<LoadedPack>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(lp) = pack else {
        warn!("standard: no pack loaded — nothing to render");
        return;
    };
    let pack: &Pack = &lp.0;
    let t0 = std::time::Instant::now();

    // 1. Materials → StandardMaterial handles, keyed by material id.
    let mut tex_cache: HashMap<(String, bool), Option<Handle<Image>>> = HashMap::new();
    let mut mat_handles: HashMap<u32, Handle<StandardMaterial>> = HashMap::new();
    for m in &pack.materials {
        let sm = build_material(m, &mut images, &mut tex_cache);
        mat_handles.insert(m.id, materials.add(sm));
    }
    let n_tex = tex_cache.values().filter(|v| v.is_some()).count();

    // 2. Per-mesh: decode geometry ONCE, build one Bevy mesh per submesh (Bevy is
    //    one-material-per-mesh, so a multi-material eftpack mesh becomes N assets).
    //    submesh_assets[mesh_id] = Vec<(mesh handle, material handle)>.
    let mut submesh_assets: Vec<Vec<(Handle<Mesh>, Handle<StandardMaterial>)>> =
        vec![Vec::new(); pack.manifest.meshes.len()];
    let mut n_submeshes = 0usize;
    for me in &pack.manifest.meshes {
        let geom = match pack.mesh_geom(me) {
            Ok(g) => g,
            Err(e) => {
                warn!("standard: mesh {} '{}' decode failed: {e}", me.id, me.name);
                continue;
            }
        };
        for sub in &me.submeshes {
            let s = sub.idx_start as usize;
            let e = s + sub.idx_count as usize;
            if e > geom.indices.len() {
                continue;
            }
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, geom.positions.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, geom.normals.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, geom.uvs.clone());
            mesh.insert_indices(Indices::U32(geom.indices[s..e].to_vec()));
            // Tangents are required for normal mapping; generate from UV+normal.
            let _ = mesh.generate_tangents();
            let mesh_h = meshes.add(mesh);
            let mat_h = mat_handles
                .get(&sub.material_id)
                .cloned()
                .unwrap_or_default();
            submesh_assets[me.id as usize].push((mesh_h, mat_h));
            n_submeshes += 1;
        }
    }

    // 3. Spawn one entity per (instance × submesh) with the instance's full affine.
    let mut n_entities = 0usize;
    for inst in &pack.instances {
        let mid = inst.mesh_id as usize;
        if mid >= submesh_assets.len() {
            continue;
        }
        let xform = Transform::from_matrix(Mat4::from(inst.affine3a()));
        for (mesh_h, mat_h) in &submesh_assets[mid] {
            commands.spawn((
                Mesh3d(mesh_h.clone()),
                MeshMaterial3d(mat_h.clone()),
                xform,
                EftGeom,
            ));
            n_entities += 1;
        }
    }

    info!(
        "standard: spawned {} entities from {} instances ({} submesh assets, {} materials, {} textures) in {:.1}s",
        n_entities,
        pack.instances.len(),
        n_submeshes,
        pack.materials.len(),
        n_tex,
        t0.elapsed().as_secs_f32()
    );
}
