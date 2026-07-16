//! poi.rs — POINT-OF-INTEREST overlays for the layer panel.
//!
//! PMC / scav / boss spawns come from `loot.json` (tarkov.dev `maps.spawns`, clustered by
//! build_loot.py); extracts / doors / interactables come from `semantics.json`
//! (extract_semantics.py, name-classified GameObjects). loot.json v2 also carries a MAP-INTEL
//! set — locks & keys, hazards, switches, transits, stationary weapons, loose loot, plus a
//! clean faction-tagged extract list (`extracts_dev`) that supersedes the semantics extracts
//! when present. A QUEST layer comes from `tasks.json` (build_tasks.py) — the global task
//! catalog, filtered to the current map's objective zones. Every marker carries a `PoiLayer`
//! component; the layer panel
//! (`ui::LayerToggles`) drives its visibility. All positions are already pack space (the same
//! diag(-1,1,1) X-flip as the geometry). Both sidecars resolve next to the pack (portable).

use crate::inspect::{money, prettify, titlecase, MarkerInfo, PickRadius};
use crate::render::LoadedPack;
use crate::ui::{LayerToggles, QuestTracker};
use bevy::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    // ---- QUESTS (tasks.json) ----
    Quest,
}

/// The owning task id carried by every Quest marker, so the tracker (`ui::QuestTracker`) can
/// focus marker visibility to the selected task(s) — see `apply_quest_visibility`.
#[derive(Component)]
pub struct QuestMarkerTask(pub String);

/// The extract's faction ("pmc"/"scav"/"shared") carried by every `extracts_dev` marker, so the
/// UI can reason about who can use it (nearest-extract routing; the string itself is kept for
/// future per-faction filtering).
#[derive(Component)]
pub struct ExtractFaction(#[allow(dead_code)] pub String);

/// A marker's estimated ruble value — the container's `ev` on loot.rs container markers, the item
/// price `pr` on loose-loot markers (0 when unpriced). Read by the panel's "min value" filter
/// (`ui::LayerToggles::min_value`); markers WITHOUT this component are never value-filtered.
#[derive(Component)]
pub struct MarkerValue(pub i64);

/// true iff an (optionally value-tagged) marker passes the min-value filter. `min == 0` means the
/// filter is off; untagged markers always pass; tagged-but-unpriced (value 0) markers hide under
/// an active filter — an unknown-value marker is exactly the clutter the filter exists to cut.
pub fn value_passes(min: i64, v: Option<&MarkerValue>) -> bool {
    min <= 0 || v.map_or(true, |mv| mv.0 >= min)
}

/// Startup-built catalog of THIS map's tasks (tasks.json filtered to zones on this map), read by
/// the quest tracker UI (ui.rs) and the outline gizmo. Empty by default so the resource always
/// exists even without a `tasks.json`.
#[derive(Resource, Default)]
pub struct QuestData {
    pub tasks: Vec<QuestEntry>,
}
/// One task with the objective zones that fall on this map.
pub struct QuestEntry {
    pub id: String,
    pub name: String,
    pub trader: String,
    pub min_level: Option<u32>,
    pub kappa: bool,
    /// Lightkeeper-chain task.
    pub lk: bool,
    /// Prerequisite task ids (kept for the UI to reason about ordering).
    #[allow(dead_code)]
    pub requires: Vec<String>,
    pub objectives: Vec<QuestObj>,
}
/// One objective and its map-local zones.
pub struct QuestObj {
    /// Objective text (schema fidelity; the marker card sources its own copy from tasks.json).
    #[allow(dead_code)]
    pub desc: String,
    pub zones: Vec<QuestZoneW>,
}
/// A single objective zone bridged to viewer space: a point + an optional footprint polygon.
pub struct QuestZoneW {
    pub pos: Vec3,
    /// Closed polygon footprint (already world-space at zone height); empty for point-only zones.
    pub outline: Vec<Vec3>,
    /// Upper bound of the zone volume (schema fidelity; the outline verts carry their own height).
    #[allow(dead_code)]
    pub top: f32,
}

/// Startup-built aggregate of every key used by THIS map's locks (loot.json `locks`), read by the
/// layer panel's "Keys for this map" list. One entry per distinct key (most expensive first),
/// carrying every lock position it opens so a click can fly to one. Empty by default so the
/// resource always exists even without a `loot.json`.
#[derive(Resource, Default)]
pub struct KeyCatalog {
    pub keys: Vec<KeyUse>,
}
/// One key and the locks it opens on this map. Only the egui panel reads it (hence the gated
/// allow, same as ui.rs `UiSearch`).
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct KeyUse {
    pub name: String,
    pub price: Option<i64>,
    /// true = keycard (reads violet in the panel, matching the marker/card accent).
    pub card: bool,
    pub lock_positions: Vec<Vec3>,
}

pub struct PoiPlugin;
impl Plugin for PoiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<QuestData>()
            .init_resource::<KeyCatalog>()
            .add_systems(Startup, spawn_pois)
            // Quest markers get their own visibility pass (toggle AND tracker selection); the other
            // POI layers stay on the plain toggle-driven `apply_poi_visibility`.
            .add_systems(Update, (apply_poi_visibility, apply_quest_visibility))
            .add_systems(Update, draw_quest_outlines);
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
        PoiLayer::Quest => (Color::srgb(0.52, 0.48, 0.96), 0.85, 1.0),
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
    /// shortName (titles the lock marker when present).
    #[serde(default)]
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

/// tasks.json (build_tasks.py) — the global quest catalog. One entry per task; each task's
/// objectives carry map-located `zones` (positions already bridged to viewer space).
#[derive(Deserialize)]
struct QuestFile {
    tasks: Vec<QuestTask>,
}
#[derive(Deserialize)]
struct QuestTask {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    trader: String,
    #[serde(default, rename = "minLevel")]
    min_level: Option<u32>,
    #[serde(default)]
    kappa: bool,
    /// Lightkeeper-chain task.
    #[serde(default)]
    lk: bool,
    /// Prerequisite task ids.
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    objectives: Vec<QuestObjective>,
}
#[derive(Deserialize)]
struct QuestObjective {
    #[serde(default)]
    desc: String,
    #[serde(default)]
    zones: Vec<QuestZone>,
}
#[derive(Deserialize)]
struct QuestZone {
    /// Map id this zone belongs to (a task can span maps).
    #[serde(default)]
    map: String,
    /// Bridged viewer-space position, or absent (outline-only zone — skipped).
    #[serde(default)]
    pos: Option<[f32; 3]>,
    /// Footprint polygon (bridged viewer-space verts); empty for point-only zones.
    #[serde(default)]
    outline: Vec<[f32; 3]>,
    /// Upper bound of the zone volume.
    #[serde(default)]
    top: f32,
}

/// Card for a quest objective located on this map (tasks.json).
fn quest_info(t: &QuestTask, o: &QuestObjective) -> MarkerInfo {
    let mut detail = Vec::new();
    if !o.desc.is_empty() {
        detail.push(o.desc.clone());
    }
    let mut tags = Vec::new();
    if let Some(lv) = t.min_level.filter(|&l| l > 0) {
        tags.push(format!("Lvl {lv}"));
    }
    if t.kappa {
        tags.push("Kappa".into());
    }
    if !tags.is_empty() {
        detail.push(tags.join("  \u{00B7}  "));
    }
    MarkerInfo {
        title: if t.name.is_empty() { "Task".into() } else { t.name.clone() },
        subtitle: if t.trader.is_empty() { "Task".into() } else { format!("Task \u{00B7} {}", t.trader) },
        detail,
        accent: poi_look(PoiLayer::Quest).0,
    }
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

/// Marker/accent colour per extract faction. PMC keeps the layer's extract green; scav-only
/// reads amber; shared reads a pale white-green.
fn extract_faction_color(fac: &str) -> Color {
    match fac {
        "scav" => Color::srgb(0.95, 0.45, 0.10),
        "shared" => Color::srgb(0.78, 0.95, 0.82),
        _ => poi_look(PoiLayer::Extract).0,
    }
}

/// Card for a clean faction-tagged extract (loot.json `extracts_dev`).
fn extract_dev_info(ex: &ExtractDev) -> MarkerInfo {
    let accent = extract_faction_color(&ex.fac);
    let name = if ex.name.is_empty() {
        "Extract".to_string()
    } else {
        ex.name.clone()
    };
    // Faction in the title (e.g. "Armored Train  [All]") so search hits "pmc"/"scav".
    let title = if ex.fac.is_empty() {
        name
    } else {
        format!("{}  [{}]", name, faction_label(&ex.fac))
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
    // EVERY key (not just the first) goes into the detail lines so secondary keys are searchable.
    let detail = if lk.keys.is_empty() {
        vec!["No key listed".into()]
    } else {
        let mut d = Vec::new();
        for (i, k) in lk.keys.iter().enumerate() {
            let mut needs = format!("{}: {}", if i == 0 { "Needs" } else { "Or" }, k.n);
            if k.card == 1 {
                needs.push_str("  [keycard]");
            }
            d.push(needs);
            if let Some(pr) = k.pr {
                if pr > 0 {
                    d.push(format!("Value  {}", money(pr)));
                }
            }
        }
        if lk.pw == 1 {
            d.push("Power required".into());
        }
        d
    };
    // Title with the key's shortName when we have one, so the marker/search read as the key.
    let title = lk
        .keys
        .first()
        .filter(|k| !k.s.is_empty())
        .map(|k| k.s.clone())
        .unwrap_or_else(|| "Locked".into());
    MarkerInfo {
        title,
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

/// Resolve a POI sidecar (`tasks.json` / `semantics.json`) WITHOUT a hardcoded path, mirroring
/// loot.rs `resolve_loot_json`: the env override, then next to the pack, then the cwd. Warns on a
/// total miss naming the UI feature that stays empty (`lost`) — the sidecars are optional, so
/// this is the only breadcrumb.
fn resolve_sidecar(env_key: &str, file: &str, root: &Path, lost: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_key) {
        let pb = PathBuf::from(&p);
        if pb.is_file() {
            return Some(pb);
        }
        warn!("poi: {env_key}='{p}' is not a file — ignoring");
    }
    let pb = root.join(file);
    if pb.is_file() {
        return Some(pb);
    }
    let cwd = PathBuf::from(file);
    if cwd.is_file() {
        return Some(cwd);
    }
    warn!("poi: no {file} found (set {env_key}, or drop {file} next to the .eftpack) — {lost}");
    None
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
        PoiLayer::Quest,
    ];
    let mk = |materials: &mut Assets<StandardMaterial>, c: Color| {
        let lin = c.to_linear();
        materials.add(StandardMaterial {
            base_color: c,
            emissive: LinearRgba::new(lin.red * 0.85, lin.green * 0.85, lin.blue * 0.85, 1.0),
            perceptual_roughness: 0.9,
            ..default()
        })
    };
    let mut mats: HashMap<u8, Handle<StandardMaterial>> = HashMap::new();
    for &l in &all {
        let (c, _, _) = poi_look(l);
        mats.insert(l as u8, mk(&mut materials, c));
    }
    // Off-palette marker materials (same emissive formula as the layer mats): faction-tinted
    // extracts + violet keycard locks. PMC extracts keep the layer's default extract green.
    let ex_scav = mk(&mut materials, extract_faction_color("scav"));
    let ex_shared = mk(&mut materials, extract_faction_color("shared"));
    let keycard_mat = mk(&mut materials, Color::srgb(0.72, 0.45, 0.92)); // lock_info's violet

    // helper closure captured by the spawn loop below (all shared handles are Clone). `mat`
    // overrides the per-layer material (faction extracts / keycard locks); None = layer default.
    // Returns the spawned entity so the quest loop can attach a `QuestMarkerTask` after the fact.
    let mut n = 0u32;
    let mut spawn = |commands: &mut Commands,
                     l: PoiLayer,
                     p: [f32; 3],
                     info: MarkerInfo,
                     mat: Option<Handle<StandardMaterial>>|
     -> Entity {
        let (_, r, lift) = poi_look(l);
        // Clamp the click radius up so tiny door/interactable markers stay hittable.
        let pick_r = r.max(0.9);
        let e = commands
            .spawn((
                Mesh3d(sphere.clone()),
                MeshMaterial3d(mat.unwrap_or_else(|| mats[&(l as u8)].clone())),
                Transform::from_xyz(p[0], p[1] + lift, p[2]).with_scale(Vec3::splat(r)),
                l,
                PickRadius(pick_r),
                info,
                Visibility::Hidden, // POI layers default OFF; the panel toggles them on
            ))
            .id();
        n += 1;
        e
    };

    // ---- spawns + map intel from loot.json (pmc/scav/boss nodes, locks, hazards, ...) ----
    // Set when loot.json ships clean faction-tagged extracts; those supersede the semantics
    // `extract` layer below.
    let mut have_dev_extracts = false;
    let mut have_dev_stationary = false;
    // Aggregated per-key lock list, folded into `KeyCatalog` below (panel's "Keys for this map").
    let mut key_uses: Vec<KeyUse> = Vec::new();
    let key = map_key(&lp.0.manifest.dataset);
    if let Some(mn) = std::fs::read_to_string(root.join("loot.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<LootFile>(&s).ok())
        .and_then(|mut f| f.maps.remove(&key))
    {
        for nd in &mn.pmc_nodes {
            spawn(&mut commands, PoiLayer::PmcSpawn, nd.pos, node_info(PoiLayer::PmcSpawn, nd), None);
        }
        for nd in &mn.scav_nodes {
            spawn(&mut commands, PoiLayer::ScavSpawn, nd.pos, node_info(PoiLayer::ScavSpawn, nd), None);
        }
        for nd in &mn.boss_nodes {
            spawn(&mut commands, PoiLayer::Boss, nd.pos, node_info(PoiLayer::Boss, nd), None);
        }
        // ---- map intel ----
        for lk in &mn.locks {
            // Keycard locks get the violet marker material (matches lock_info's accent).
            let keycard = lk.keys.first().is_some_and(|k| k.card == 1);
            spawn(
                &mut commands,
                PoiLayer::Lock,
                lk.pos,
                lock_info(lk),
                keycard.then(|| keycard_mat.clone()),
            );
            // Fold the lock's (first) key into the catalog, grouped by key name.
            if let Some(k) = lk.keys.first().filter(|k| !k.n.is_empty()) {
                let pos = Vec3::new(lk.pos[0], lk.pos[1], lk.pos[2]);
                if let Some(u) = key_uses.iter_mut().find(|u| u.name == k.n) {
                    u.lock_positions.push(pos);
                } else {
                    key_uses.push(KeyUse {
                        name: k.n.clone(),
                        price: k.pr,
                        card: k.card == 1,
                        lock_positions: vec![pos],
                    });
                }
            }
        }
        for sw in &mn.switches {
            spawn(&mut commands, PoiLayer::Switch, sw.pos, switch_info(sw), None);
        }
        for tr in &mn.transits {
            spawn(&mut commands, PoiLayer::Transit, tr.pos, transit_info(tr), None);
        }
        for hz in &mn.hazards {
            spawn(&mut commands, PoiLayer::Hazard, hz.pos, hazard_info(hz), None);
        }
        have_dev_stationary = !mn.stationary.is_empty();
        for st in &mn.stationary {
            spawn(&mut commands, PoiLayer::Stationary, st.pos, stationary_info(st), None);
        }
        for lo in &mn.loose {
            // Tagged with the item price so the panel's min-value filter applies (0 = unpriced).
            let e = spawn(&mut commands, PoiLayer::LooseLoot, lo.pos, loose_info(lo), None);
            commands.entity(e).insert(MarkerValue(lo.pr.unwrap_or(0)));
        }
        // Prefer the clean faction-tagged extract list when it's present.
        if !mn.extracts_dev.is_empty() {
            have_dev_extracts = true;
            for ex in &mn.extracts_dev {
                let mat = match ex.fac.as_str() {
                    "scav" => Some(ex_scav.clone()),
                    "shared" => Some(ex_shared.clone()),
                    _ => None, // pmc / unknown keep the layer's extract green
                };
                let e = spawn(&mut commands, PoiLayer::Extract, ex.pos, extract_dev_info(ex), mat);
                commands.entity(e).insert(ExtractFaction(ex.fac.clone()));
            }
        }
    }

    // ---- extracts / doors / interactables from semantics.json ----
    // Skip the semantics `extract` layer if loot.json already gave us clean extracts.
    if let Some(layers) =
        resolve_sidecar("EFT_SEMANTICS_JSON", "semantics.json", root, "doors/interactables missing")
            .and_then(|p| std::fs::read_to_string(p).ok())
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
                    spawn(&mut commands, ly, poi.p, sem_info(ly, poi), None);
                }
            }
        }
    }

    // ---- quest objectives from tasks.json (global catalog, filtered to THIS map's zones) ----
    // As we spawn each Quest marker we also fold the task into `QuestData` (the tracker's catalog):
    // one entry per task that has at least one objective zone on THIS map, carrying every such
    // zone's point + footprint outline. Each marker gets a `QuestMarkerTask` so the tracker can
    // focus visibility to selected tasks.
    let mut quest_tasks: Vec<QuestEntry> = Vec::new();
    if let Some(qf) = resolve_sidecar("EFT_TASKS_JSON", "tasks.json", root, "quest tracker empty")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<QuestFile>(&s).ok())
    {
        for t in &qf.tasks {
            let mut objs: Vec<QuestObj> = Vec::new();
            for o in &t.objectives {
                let mut zones: Vec<QuestZoneW> = Vec::new();
                for z in &o.zones {
                    if z.map != key {
                        continue;
                    }
                    let Some(p) = z.pos else { continue };
                    let e = spawn(&mut commands, PoiLayer::Quest, p, quest_info(t, o), None);
                    commands.entity(e).insert(QuestMarkerTask(t.id.clone()));
                    zones.push(QuestZoneW {
                        pos: Vec3::new(p[0], p[1], p[2]),
                        outline: z.outline.iter().map(|a| Vec3::new(a[0], a[1], a[2])).collect(),
                        top: z.top,
                    });
                }
                if !zones.is_empty() {
                    objs.push(QuestObj { desc: o.desc.clone(), zones });
                }
            }
            if !objs.is_empty() {
                quest_tasks.push(QuestEntry {
                    id: t.id.clone(),
                    name: t.name.clone(),
                    trader: t.trader.clone(),
                    min_level: t.min_level,
                    kappa: t.kappa,
                    lk: t.lk,
                    requires: t.requires.clone(),
                    objectives: objs,
                });
            }
        }
    }
    let quest_count = quest_tasks.len();
    // Most expensive keys first — the panel renders the catalog in this order.
    key_uses.sort_by_key(|k| std::cmp::Reverse(k.price.unwrap_or(0)));
    commands.insert_resource(KeyCatalog { keys: key_uses });
    commands.insert_resource(QuestData { tasks: quest_tasks });
    if quest_count > 0 {
        info!("poi: {quest_count} tasks tracked on this map");
    }

    // ---- STATIONARY weapons from GAME GEOMETRY (fills the tarkov.dev gap) ----
    // tarkov.dev's `stationaryWeapons` is empty for many maps (incl. Interchange), yet the map
    // HAS mounted MGs — in the pack they're `reciever_*` (and NSV/Kord/DShK/Utes) meshes. When
    // loot.json gave us no stationary data, scan the pack instances and dedupe co-located
    // receiver parts (one gun = several meshes at ~one spot) into one marker per gun.
    if !have_dev_stationary {
        let pack = &lp.0;
        let is_mg = |name: &str| {
            let s = name.to_ascii_lowercase();
            !s.contains("sandbag")
                && (s.contains("reciever")
                    || s.contains("receiver")
                    || s.contains("dshk")
                    || s.contains("nsv")
                    || s.contains("kord")
                    || s.contains("utes")
                    || s.contains("pkm")
                    || s.contains("pulemet"))
        };
        let mut seen: Vec<Vec3> = Vec::new();
        for inst in &pack.instances {
            let mid = inst.mesh_id as usize;
            let Some(m) = pack.manifest.meshes.get(mid) else {
                continue;
            };
            if !is_mg(&m.name) {
                continue;
            }
            let t = inst.affine3a().translation;
            let p = Vec3::new(t.x, t.y, t.z);
            // one gun spans several receiver meshes at ~the same spot — collapse them.
            if seen.iter().any(|q| (q.x - p.x).abs() < 3.0 && (q.z - p.z).abs() < 3.0) {
                continue;
            }
            seen.push(p);
            spawn(
                &mut commands,
                PoiLayer::Stationary,
                [p.x, p.y, p.z],
                MarkerInfo {
                    title: "Stationary gun".into(),
                    subtitle: "Mounted MG".into(),
                    detail: vec!["From map geometry".into()],
                    accent: poi_look(PoiLayer::Stationary).0,
                },
                None,
            );
        }
        if !seen.is_empty() {
            info!("poi: +{} stationary guns mined from geometry", seen.len());
        }
    }

    info!("poi: {n} POI markers spawned (spawns/extracts/doors/interactables + map intel + quests)");
}

// Quest markers are handled by `apply_quest_visibility` (they carry a `QuestMarkerTask`); this pass
// owns every OTHER POI layer, so exclude them here to avoid the two systems fighting over the same
// `Visibility`.
fn apply_poi_visibility(
    toggles: Res<LayerToggles>,
    mut q: Query<(&PoiLayer, Option<&MarkerValue>, &mut Visibility), Without<QuestMarkerTask>>,
) {
    if !toggles.is_changed() {
        return;
    }
    for (l, val, mut vis) in &mut q {
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
            PoiLayer::Quest => toggles.quests, // unreachable here (filtered out), kept exhaustive
        };
        // Value-tagged markers (loose loot) additionally pass the panel's min-value filter.
        let show = show && value_passes(toggles.min_value, val);
        *vis = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// Quest-marker visibility: shown iff the master `quests` toggle is on AND either no task is
/// selected (all quest markers behave as today) or this marker's task is in the tracker's active
/// set (selecting tasks focuses the map to them). Cheap; runs only when a toggle or the tracker
/// changes (~30 quest markers).
fn apply_quest_visibility(
    toggles: Res<LayerToggles>,
    tracker: Res<QuestTracker>,
    mut q: Query<(&QuestMarkerTask, &mut Visibility)>,
) {
    if !toggles.is_changed() && !tracker.is_changed() {
        return;
    }
    for (task, mut vis) in &mut q {
        let show = toggles.quests
            && (tracker.active.is_empty() || tracker.active.contains(&task.0));
        *vis = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// Draw each ACTIVE task's objective-zone footprints as closed quest-purple polygons (immediate
/// mode, mirrors pathfind's `draw_route`). Point-only zones (empty outline) are skipped.
fn draw_quest_outlines(mut gizmos: Gizmos, quest_data: Res<QuestData>, tracker: Res<QuestTracker>) {
    if tracker.active.is_empty() {
        return;
    }
    let color = Color::srgb(0.60, 0.50, 0.98); // quest purple (matches the Quest marker hue)
    for t in &quest_data.tasks {
        if !tracker.active.contains(&t.id) {
            continue;
        }
        for o in &t.objectives {
            for z in &o.zones {
                if z.outline.len() < 2 {
                    continue;
                }
                // Close the loop by chaining the first vertex onto the end.
                gizmos.linestrip(
                    z.outline.iter().copied().chain(z.outline.first().copied()),
                    color,
                );
            }
        }
    }
}
