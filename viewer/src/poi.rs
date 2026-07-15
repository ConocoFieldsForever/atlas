//! poi.rs — POINT-OF-INTEREST overlays for the layer panel.
//!
//! PMC / scav / boss spawns come from `loot.json` (tarkov.dev `maps.spawns`, clustered by
//! build_loot.py); extracts / doors / interactables come from `semantics.json`
//! (extract_semantics.py, name-classified GameObjects). Every marker carries a `PoiLayer`
//! component; the layer panel (`ui::LayerToggles`) drives its visibility. All positions are
//! already pack space (the same diag(-1,1,1) X-flip as the geometry). Both sidecars resolve
//! next to the pack (portable).

use crate::inspect::{money, prettify, titlecase, MarkerInfo, PickRadius};
use crate::render::LoadedPack;
use crate::ui::LayerToggles;
use bevy::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum PoiLayer {
    PmcSpawn,
    ScavSpawn,
    Boss,
    Extract,
    Door,
    Interactable,
}

pub struct PoiPlugin;
impl Plugin for PoiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_pois)
            .add_systems(Update, apply_poi_visibility);
    }
}

/// (colour, marker radius m, y-lift m). Colours match the panel swatches.
pub fn poi_look(l: PoiLayer) -> (Color, f32, f32) {
    match l {
        PoiLayer::PmcSpawn => (Color::srgb(0.90, 0.24, 0.22), 1.1, 1.2),
        PoiLayer::ScavSpawn => (Color::srgb(0.93, 0.80, 0.24), 0.95, 1.0),
        PoiLayer::Boss => (Color::srgb(0.86, 0.24, 0.86), 1.7, 1.8),
        PoiLayer::Extract => (Color::srgb(0.26, 0.90, 0.38), 1.3, 1.4),
        PoiLayer::Door => (Color::srgb(0.95, 0.56, 0.16), 0.35, 0.6),
        PoiLayer::Interactable => (Color::srgb(0.32, 0.85, 0.92), 0.32, 0.5),
    }
}

#[derive(Deserialize)]
struct LootFile {
    maps: HashMap<String, MapNodes>,
}
#[derive(Deserialize)]
struct MapNodes {
    #[serde(default)]
    pmc_nodes: Vec<Node>,
    #[serde(default)]
    scav_nodes: Vec<Node>,
    #[serde(default)]
    boss_nodes: Vec<Node>,
}
#[derive(Deserialize)]
struct Node {
    pos: [f32; 3],
    /// Spawn group size / count.
    #[serde(default)]
    n: Option<u32>,
    /// Expected ruble value of loot reachable from this spawn.
    #[serde(default)]
    ev: Option<i64>,
    /// Boss name (boss_nodes only).
    #[serde(default)]
    boss: Option<String>,
    /// Boss spawn chance 0..1 (boss_nodes only).
    #[serde(default)]
    chance: Option<f32>,
}

#[derive(Deserialize)]
struct SemFile {
    layers: HashMap<String, Vec<Poi>>,
}
#[derive(Deserialize)]
struct Poi {
    p: [f32; 3],
    /// Raw GameObject name (prettified for the card title).
    #[serde(default)]
    name: String,
}

/// Card contents for a loot.json spawn node (pmc/scav/boss).
fn node_info(l: PoiLayer, nd: &Node) -> MarkerInfo {
    let accent = poi_look(l).0;
    match l {
        PoiLayer::PmcSpawn => {
            let mut detail = Vec::new();
            if let Some(n) = nd.n {
                detail.push(format!("Group \u{00D7}{n}"));
            }
            if let Some(ev) = nd.ev {
                detail.push(format!("Est. loot  {}", money(ev)));
            }
            MarkerInfo {
                title: "PMC spawn".into(),
                subtitle: "Spawn point".into(),
                detail,
                accent,
            }
        }
        PoiLayer::ScavSpawn => {
            let mut detail = Vec::new();
            if let Some(n) = nd.n {
                detail.push(format!("Group \u{00D7}{n}"));
            }
            MarkerInfo {
                title: "Scav spawn".into(),
                subtitle: "Spawn point".into(),
                detail,
                accent,
            }
        }
        PoiLayer::Boss => {
            let title = nd
                .boss
                .as_deref()
                .map(titlecase)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "Boss".into());
            let mut detail = Vec::new();
            if let Some(ch) = nd.chance {
                detail.push(format!("Chance {:.0}%", ch * 100.0));
            }
            if let Some(ev) = nd.ev {
                detail.push(format!("Est. loot  {}", money(ev)));
            }
            MarkerInfo {
                title,
                subtitle: "Boss spawn".into(),
                detail,
                accent,
            }
        }
        _ => MarkerInfo {
            title: "Spawn".into(),
            subtitle: "Spawn point".into(),
            detail: Vec::new(),
            accent,
        },
    }
}

/// Card contents for a semantics.json POI (extract/door/interactable).
fn sem_info(l: PoiLayer, poi: &Poi) -> MarkerInfo {
    let accent = poi_look(l).0;
    let pretty = prettify(&poi.name);
    let (fallback, subtitle) = match l {
        PoiLayer::Extract => ("Extract", "Extract"),
        PoiLayer::Door => ("Door", "Door"),
        _ => ("Interactable", "Interactable"),
    };
    let title = if pretty.is_empty() {
        fallback.to_string()
    } else {
        pretty
    };
    MarkerInfo {
        title,
        subtitle: subtitle.to_string(),
        detail: Vec::new(),
        accent,
    }
}

/// dataset "interchange_v2" -> loot/semantics map key "interchange" (strip a `_vN` suffix).
fn map_key(dataset: &str) -> String {
    if let Some((base, ver)) = dataset.rsplit_once("_v") {
        if !ver.is_empty() && ver.chars().all(|c| c.is_ascii_digit()) {
            return base.to_string();
        }
    }
    dataset.to_string()
}

fn spawn_pois(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pack: Option<Res<LoadedPack>>,
) {
    let Some(lp) = pack else { return };
    let root = &lp.0.root;

    let sphere = meshes.add(Sphere::new(1.0));
    // one emissive material per layer.
    let all = [
        PoiLayer::PmcSpawn,
        PoiLayer::ScavSpawn,
        PoiLayer::Boss,
        PoiLayer::Extract,
        PoiLayer::Door,
        PoiLayer::Interactable,
    ];
    let mut mats: HashMap<u8, Handle<StandardMaterial>> = HashMap::new();
    for &l in &all {
        let (c, _, _) = poi_look(l);
        let lin = c.to_linear();
        mats.insert(
            l as u8,
            materials.add(StandardMaterial {
                base_color: c,
                emissive: LinearRgba::new(lin.red * 0.85, lin.green * 0.85, lin.blue * 0.85, 1.0),
                perceptual_roughness: 0.9,
                ..default()
            }),
        );
    }

    // helper closure captured by the spawn loop below (all shared handles are Clone).
    let mut n = 0u32;
    let mut spawn = |commands: &mut Commands, l: PoiLayer, p: [f32; 3], info: MarkerInfo| {
        let (_, r, lift) = poi_look(l);
        // Clamp the click radius up so tiny door/interactable markers stay hittable.
        let pick_r = r.max(0.9);
        commands.spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(mats[&(l as u8)].clone()),
            Transform::from_xyz(p[0], p[1] + lift, p[2]).with_scale(Vec3::splat(r)),
            l,
            PickRadius(pick_r),
            info,
            Visibility::Hidden, // POI layers default OFF; the panel toggles them on
        ));
        n += 1;
    };

    // ---- spawns from loot.json (pmc/scav/boss nodes) ----
    let key = map_key(&lp.0.manifest.dataset);
    if let Some(mn) = std::fs::read_to_string(root.join("loot.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<LootFile>(&s).ok())
        .and_then(|mut f| f.maps.remove(&key))
    {
        for nd in &mn.pmc_nodes {
            spawn(&mut commands, PoiLayer::PmcSpawn, nd.pos, node_info(PoiLayer::PmcSpawn, nd));
        }
        for nd in &mn.scav_nodes {
            spawn(&mut commands, PoiLayer::ScavSpawn, nd.pos, node_info(PoiLayer::ScavSpawn, nd));
        }
        for nd in &mn.boss_nodes {
            spawn(&mut commands, PoiLayer::Boss, nd.pos, node_info(PoiLayer::Boss, nd));
        }
    }

    // ---- extracts / doors / interactables from semantics.json ----
    if let Some(layers) = std::fs::read_to_string(root.join("semantics.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<SemFile>(&s).ok())
        .map(|f| f.layers)
    {
        let map = [
            ("extract", PoiLayer::Extract),
            ("door", PoiLayer::Door),
            ("loot", PoiLayer::Interactable),
        ];
        for (lname, ly) in map {
            if let Some(v) = layers.get(lname) {
                for poi in v {
                    spawn(&mut commands, ly, poi.p, sem_info(ly, poi));
                }
            }
        }
    }

    info!("poi: {n} POI markers spawned (spawns/extracts/doors/interactables)");
}

fn apply_poi_visibility(toggles: Res<LayerToggles>, mut q: Query<(&PoiLayer, &mut Visibility)>) {
    if !toggles.is_changed() {
        return;
    }
    for (l, mut vis) in &mut q {
        let show = match l {
            PoiLayer::PmcSpawn => toggles.pmc_spawns,
            PoiLayer::ScavSpawn => toggles.scav_spawns,
            PoiLayer::Boss => toggles.bosses,
            PoiLayer::Extract => toggles.extracts,
            PoiLayer::Door => toggles.doors,
            PoiLayer::Interactable => toggles.interactables,
        };
        *vis = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}
