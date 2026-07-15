//! poi.rs — POINT-OF-INTEREST overlays for the layer panel.
//!
//! PMC / scav / boss spawns come from `loot.json` (tarkov.dev `maps.spawns`, clustered by
//! build_loot.py); extracts / doors / interactables come from `semantics.json`
//! (extract_semantics.py, name-classified GameObjects). loot.json v2 also carries a MAP-INTEL
//! set — locks & keys, hazards, switches, transits, stationary weapons, loose loot, plus a
//! clean faction-tagged extract list (`extracts_dev`) that supersedes the semantics extracts
//! when present. Every marker carries a `PoiLayer` component; the layer panel
//! (`ui::LayerToggles`) drives its visibility. All positions are already pack space (the same
//! diag(-1,1,1) X-flip as the geometry). Both sidecars resolve next to the pack (portable).

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
    // ---- MAP INTEL (loot.json v2) ----
    Lock,
    Hazard,
    Switch,
    Transit,
    Stationary,
    LooseLoot,
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
        PoiLayer::Lock => (Color::srgb(0.93, 0.78, 0.30), 0.45, 0.7),
        PoiLayer::Hazard => (Color::srgb(0.95, 0.26, 0.20), 1.0, 1.0),
        PoiLayer::Switch => (Color::srgb(0.30, 0.85, 0.92), 0.45, 0.7),
        PoiLayer::Transit => (Color::srgb(0.20, 0.85, 0.62), 1.2, 1.3),
        PoiLayer::Stationary => (Color::srgb(0.66, 0.62, 0.35), 0.6, 0.8),
        PoiLayer::LooseLoot => (Color::srgb(0.86, 0.80, 0.55), 0.32, 0.5),
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
    // ---- MAP INTEL (loot.json v2) — every array `default` so old packs are fine ----
    #[serde(default)]
    locks: Vec<Lock>,
    #[serde(default)]
    switches: Vec<Switch>,
    #[serde(default)]
    transits: Vec<Transit>,
    #[serde(default)]
    hazards: Vec<Hazard>,
    #[serde(default)]
    stationary: Vec<Stationary>,
    #[serde(default)]
    extracts_dev: Vec<ExtractDev>,
    #[serde(default)]
    loose: Vec<Loose>,
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

/// A locked door/container/trunk and the key(s) that open it.
#[derive(Deserialize)]
struct Lock {
    pos: [f32; 3],
    /// Lock type: "door" / "container" / "trunk".
    #[serde(default)]
    lt: String,
    /// Needs power (0/1).
    #[serde(default)]
    pw: i64,
    /// Usually one key. `{n, s, card, pr}`.
    #[serde(default)]
    keys: Vec<Key>,
}
#[derive(Deserialize)]
struct Key {
    /// Full item name.
    #[serde(default)]
    n: String,
    /// shortName (unused in the card, kept for schema fidelity).
    #[serde(default)]
    #[allow(dead_code)]
    s: String,
    /// 0/1 — 1 means a keycard.
    #[serde(default)]
    card: i64,
    /// Roubles, or null.
    #[serde(default)]
    pr: Option<i64>,
}
/// A power/lever switch.
#[derive(Deserialize)]
struct Switch {
    pos: [f32; 3],
    #[serde(default)]
    name: String,
    /// Switch state / type label (e.g. "Close").
    #[serde(default)]
    st: String,
}
/// A transit exit to another map.
#[derive(Deserialize)]
struct Transit {
    pos: [f32; 3],
    /// Destination map key.
    #[serde(default)]
    to: String,
    #[serde(default)]
    desc: String,
    #[serde(default)]
    cond: String,
}
/// An environmental hazard (minefield, radiation, ...).
#[derive(Deserialize)]
struct Hazard {
    pos: [f32; 3],
    /// Hazard type.
    #[serde(default)]
    ht: String,
    #[serde(default)]
    name: String,
}
/// A mounted / stationary weapon.
#[derive(Deserialize)]
struct Stationary {
    pos: [f32; 3],
    #[serde(default)]
    name: String,
}
/// A clean, faction-tagged extract (supersedes the semantics extracts when present).
#[derive(Deserialize)]
struct ExtractDev {
    pos: [f32; 3],
    #[serde(default)]
    name: String,
    /// Faction: "pmc" / "scav" / "shared".
    #[serde(default)]
    fac: String,
}
/// A single valuable loose-loot point (already price-filtered upstream).
#[derive(Deserialize)]
struct Loose {
    pos: [f32; 3],
    /// Top item shortName.
    #[serde(default)]
    s: String,
    /// Top item full name.
    #[serde(default)]
    n: String,
    /// Its price in roubles, or null.
    #[serde(default)]
    pr: Option<i64>,
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

/// Human faction label for an extract card ("pmc"->"PMC", "scav"->"Scav", "shared"->"All").
fn faction_label(fac: &str) -> String {
    match fac {
        "pmc" => "PMC".into(),
        "scav" => "Scav".into(),
        "shared" => "All".into(),
        _ => titlecase(fac),
    }
}

/// Card for a clean faction-tagged extract (loot.json `extracts_dev`).
fn extract_dev_info(ex: &ExtractDev) -> MarkerInfo {
    let accent = poi_look(PoiLayer::Extract).0;
    let title = if ex.name.is_empty() {
        "Extract".into()
    } else {
        ex.name.clone()
    };
    MarkerInfo {
        title,
        subtitle: "Extract".into(),
        detail: vec![format!("Faction: {}", faction_label(&ex.fac))],
        accent,
    }
}

/// Card for a locked door/container/trunk (loot.json `locks`).
fn lock_info(lk: &Lock) -> MarkerInfo {
    // Keycard locks read violet; ordinary locks keep the gold layer colour.
    let keycard = lk.keys.first().is_some_and(|k| k.card == 1);
    let accent = if keycard {
        Color::srgb(0.72, 0.45, 0.92)
    } else {
        poi_look(PoiLayer::Lock).0
    };
    let detail = if let Some(k) = lk.keys.first() {
        let mut d = Vec::new();
        let mut needs = format!("Needs: {}", k.n);
        if k.card == 1 {
            needs.push_str("  [keycard]");
        }
        d.push(needs);
        if let Some(pr) = k.pr {
            if pr > 0 {
                d.push(format!("Value  {}", money(pr)));
            }
        }
        if lk.pw == 1 {
            d.push("Power required".into());
        }
        d
    } else {
        vec!["No key listed".into()]
    };
    MarkerInfo {
        title: "Locked".into(),
        subtitle: titlecase(&lk.lt),
        detail,
        accent,
    }
}

/// Card for a switch (loot.json `switches`).
fn switch_info(sw: &Switch) -> MarkerInfo {
    let mut detail = Vec::new();
    if !sw.st.is_empty() {
        detail.push(format!("Type: {}", sw.st));
    }
    MarkerInfo {
        title: sw.name.clone(),
        subtitle: "Switch".into(),
        detail,
        accent: poi_look(PoiLayer::Switch).0,
    }
}

/// Card for a transit to another map (loot.json `transits`).
fn transit_info(tr: &Transit) -> MarkerInfo {
    let mut detail = Vec::new();
    if !tr.desc.is_empty() {
        detail.push(tr.desc.clone());
    }
    if !tr.cond.is_empty() {
        detail.push(tr.cond.clone());
    }
    MarkerInfo {
        title: format!("Transit \u{2192} {}", titlecase(&tr.to)),
        subtitle: "Transit".into(),
        detail,
        accent: poi_look(PoiLayer::Transit).0,
    }
}

/// Card for an environmental hazard (loot.json `hazards`).
fn hazard_info(hz: &Hazard) -> MarkerInfo {
    let ht_pretty = titlecase(&hz.ht);
    let title = if hz.name.is_empty() {
        ht_pretty.clone()
    } else {
        hz.name.clone()
    };
    let mut detail = Vec::new();
    if ht_pretty != title {
        detail.push(ht_pretty);
    }
    MarkerInfo {
        title,
        subtitle: "Hazard".into(),
        detail,
        accent: poi_look(PoiLayer::Hazard).0,
    }
}

/// Card for a mounted/stationary weapon (loot.json `stationary`).
fn stationary_info(st: &Stationary) -> MarkerInfo {
    MarkerInfo {
        title: st.name.clone(),
        subtitle: "Stationary weapon".into(),
        detail: Vec::new(),
        accent: poi_look(PoiLayer::Stationary).0,
    }
}

/// Card for a valuable loose-loot point (loot.json `loose`).
fn loose_info(lo: &Loose) -> MarkerInfo {
    let title = if lo.n.is_empty() { lo.s.clone() } else { lo.n.clone() };
    let mut detail = Vec::new();
    if let Some(pr) = lo.pr {
        if pr > 0 {
            detail.push(format!("Value  {}", money(pr)));
        }
    }
    MarkerInfo {
        title,
        subtitle: "Loose loot".into(),
        detail,
        accent: poi_look(PoiLayer::LooseLoot).0,
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
        PoiLayer::Lock,
        PoiLayer::Hazard,
        PoiLayer::Switch,
        PoiLayer::Transit,
        PoiLayer::Stationary,
        PoiLayer::LooseLoot,
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

    // ---- spawns + map intel from loot.json (pmc/scav/boss nodes, locks, hazards, ...) ----
    // Set when loot.json ships clean faction-tagged extracts; those supersede the semantics
    // `extract` layer below.
    let mut have_dev_extracts = false;
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
        // ---- map intel ----
        for lk in &mn.locks {
            spawn(&mut commands, PoiLayer::Lock, lk.pos, lock_info(lk));
        }
        for sw in &mn.switches {
            spawn(&mut commands, PoiLayer::Switch, sw.pos, switch_info(sw));
        }
        for tr in &mn.transits {
            spawn(&mut commands, PoiLayer::Transit, tr.pos, transit_info(tr));
        }
        for hz in &mn.hazards {
            spawn(&mut commands, PoiLayer::Hazard, hz.pos, hazard_info(hz));
        }
        for st in &mn.stationary {
            spawn(&mut commands, PoiLayer::Stationary, st.pos, stationary_info(st));
        }
        for lo in &mn.loose {
            spawn(&mut commands, PoiLayer::LooseLoot, lo.pos, loose_info(lo));
        }
        // Prefer the clean faction-tagged extract list when it's present.
        if !mn.extracts_dev.is_empty() {
            have_dev_extracts = true;
            for ex in &mn.extracts_dev {
                spawn(&mut commands, PoiLayer::Extract, ex.pos, extract_dev_info(ex));
            }
        }
    }

    // ---- extracts / doors / interactables from semantics.json ----
    // Skip the semantics `extract` layer if loot.json already gave us clean extracts.
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
            if lname == "extract" && have_dev_extracts {
                continue;
            }
            if let Some(v) = layers.get(lname) {
                for poi in v {
                    spawn(&mut commands, ly, poi.p, sem_info(ly, poi));
                }
            }
        }
    }

    info!("poi: {n} POI markers spawned (spawns/extracts/doors/interactables + map intel)");
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
            PoiLayer::Lock => toggles.locks,
            PoiLayer::Hazard => toggles.hazards,
            PoiLayer::Switch => toggles.switches,
            PoiLayer::Transit => toggles.transits,
            PoiLayer::Stationary => toggles.stationary,
            PoiLayer::LooseLoot => toggles.loose,
        };
        *vis = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}
