//! loot.rs — LOOT-CONTAINER OVERLAY.
//!
//! Loads `loot.json` (loot containers mined from tarkov.dev by `build_loot.py`).
//! The positions are ALREADY in our viewer/pack space: build_loot.py bridges every
//! container with `[-x, y, z]` — the exact same `diag(-1,1,1)` X-mirror the .eftpack
//! geometry uses — so a container's `pos` drops straight onto the rendered map.
//!
//! PORTABILITY (SKILL: the three viewers share ONE source of truth and must run on
//! a friend's machine): NOTHING here is a hardcoded absolute path. `loot.json` is
//! resolved relative to the loaded pack (drop it next to the .eftpack and the pack
//! is self-contained), with an `EFT_LOOT_JSON` override; the map key comes from the
//! pack manifest's `dataset`, never a baked-in literal.
//!
//! Each container is drawn as a class-colored marker cuboid via Bevy's STANDARD PBR
//! mesh path, alongside the custom GPU-driven .eftpack draw. Every marker is emissive
//! so it reads even in a dark interior.

use crate::render::LoadedPack;
use bevy::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct LootPlugin;
impl Plugin for LootPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_loot);
    }
}

#[derive(Deserialize)]
struct LootFile {
    maps: HashMap<String, MapLoot>,
}
#[derive(Deserialize)]
struct MapLoot {
    #[serde(default)]
    containers: Vec<Container>,
}
#[derive(Deserialize)]
struct Container {
    pos: [f32; 3],
    cls: String,
}

/// Container class -> (base color, half-extents in metres). Weapon boxes are dark
/// (the "black weapon crate") but never pure black, and every class gets an emissive
/// term so it's visible in shadow. Sizes are rough per-type so markers read as boxes.
fn class_look(cls: &str) -> (Color, Vec3) {
    match cls {
        "weapon" => (Color::srgb(0.14, 0.15, 0.18), Vec3::new(0.60, 0.28, 0.42)),
        "medical" => (Color::srgb(0.15, 0.80, 0.30), Vec3::new(0.35, 0.30, 0.30)),
        "safe" => (Color::srgb(0.95, 0.72, 0.12), Vec3::new(0.32, 0.45, 0.28)),
        "register" => (Color::srgb(0.15, 0.60, 0.95), Vec3::new(0.35, 0.28, 0.28)),
        "bag" => (Color::srgb(0.62, 0.40, 0.20), Vec3::new(0.30, 0.24, 0.30)),
        "crate" => (Color::srgb(0.74, 0.58, 0.35), Vec3::new(0.45, 0.35, 0.45)),
        "tech" => (Color::srgb(0.62, 0.28, 0.88), Vec3::new(0.35, 0.30, 0.30)),
        "stash" => (Color::srgb(0.45, 0.45, 0.45), Vec3::new(0.35, 0.20, 0.35)),
        "furniture" => (Color::srgb(0.60, 0.48, 0.40), Vec3::new(0.35, 0.30, 0.30)),
        "body" => (Color::srgb(0.90, 0.12, 0.12), Vec3::new(0.35, 0.30, 0.60)),
        _ => (Color::srgb(0.85, 0.85, 0.85), Vec3::new(0.30, 0.28, 0.30)),
    }
}

/// Resolve `loot.json` WITHOUT a hardcoded absolute path. Order:
///   1. `EFT_LOOT_JSON` env override (explicit path),
///   2. `<pack-dir>/loot.json` — co-located with the pack, so the pack is a
///      self-contained bundle you can hand to a friend,
///   3. `./loot.json` (cwd).
fn resolve_loot_json(pack: &Option<Res<LoadedPack>>) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("EFT_LOOT_JSON") {
        let pb = PathBuf::from(&p);
        if pb.is_file() {
            return Some(pb);
        }
        warn!("loot: EFT_LOOT_JSON='{p}' is not a file — ignoring");
    }
    if let Some(lp) = pack {
        let pb = lp.0.root.join("loot.json");
        if pb.is_file() {
            return Some(pb);
        }
    }
    let cwd = PathBuf::from("loot.json");
    if cwd.is_file() {
        return Some(cwd);
    }
    None
}

fn spawn_loot(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pack: Option<Res<LoadedPack>>,
) {
    let Some(path) = resolve_loot_json(&pack) else {
        warn!(
            "loot: no loot.json found (set EFT_LOOT_JSON, or drop loot.json next to the .eftpack) — no loot overlay"
        );
        return;
    };
    let txt = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            warn!("loot: {} unreadable ({e}) — no loot overlay", path.display());
            return;
        }
    };
    let lf: LootFile = match serde_json::from_str(&txt) {
        Ok(v) => v,
        Err(e) => {
            warn!("loot: parse failed: {e}");
            return;
        }
    };

    // Candidate map keys derived from the pack (never a hardcoded literal): the
    // exact `dataset`, then its version-suffix-stripped base name — the pack dir is
    // named e.g. "interchange_v2" while build_loot.py keys by the canonical
    // tarkov.dev map "interchange". First matching candidate wins; with no pack (or
    // no match) fall back to the sole map if the file is unambiguous.
    let mut keys: Vec<String> = Vec::new();
    if let Some(p) = pack.as_ref() {
        let ds = &p.0.manifest.dataset;
        keys.push(ds.clone());
        if let Some((base, ver)) = ds.rsplit_once("_v") {
            if !ver.is_empty() && ver.chars().all(|c| c.is_ascii_digit()) {
                keys.push(base.to_string());
            }
        }
    }
    let resolved = keys
        .iter()
        .find_map(|k| lf.maps.get(k).map(|m| (k.clone(), m)))
        .or_else(|| {
            (lf.maps.len() == 1)
                .then(|| lf.maps.iter().next().map(|(k, m)| (k.clone(), m)))
                .flatten()
        });
    let Some((map_key, ml)) = resolved else {
        warn!(
            "loot: pack dataset {:?} matched no map in {} (have: {:?})",
            pack.as_ref().map(|p| p.0.manifest.dataset.as_str()),
            path.display(),
            lf.maps.keys().collect::<Vec<_>>()
        );
        return;
    };

    let unit_cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let mut mats: HashMap<String, Handle<StandardMaterial>> = HashMap::new();
    for c in &ml.containers {
        let (color, half) = class_look(&c.cls);
        let mat = mats
            .entry(c.cls.clone())
            .or_insert_with(|| {
                let l = color.to_linear();
                materials.add(StandardMaterial {
                    base_color: color,
                    // self-lit so the container never vanishes in a dark aisle
                    emissive: LinearRgba::new(l.red * 0.7, l.green * 0.7, l.blue * 0.7, 1.0),
                    perceptual_roughness: 0.85,
                    ..default()
                })
            })
            .clone();
        // pos is the container's floor point; lift by half-height so the box sits ON the floor.
        commands.spawn((
            Mesh3d(unit_cube.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(c.pos[0], c.pos[1] + half.y, c.pos[2]).with_scale(half * 2.0),
            LootMarker,
            LootClass(c.cls.clone()),
        ));
    }
    info!(
        "loot: {} container markers spawned from {} (map '{}', {} classes)",
        ml.containers.len(),
        path.display(),
        map_key,
        mats.len()
    );
}

#[derive(Component)]
struct LootMarker;

/// The loot class of a marker ("weapon"/"medical"/…), so the layer panel can filter by class.
#[derive(Component)]
pub struct LootClass(pub String);
