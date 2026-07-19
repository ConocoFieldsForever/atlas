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

use crate::inspect::{icon_slug, money, prettify, titlecase, MarkerIcon, MarkerInfo, PickRadius};
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
    // ---- TYPED GAME DATA (gamedata.json — extract_gamedata.py) ----
    Minefield,
    SniperZone,
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

/// Tag on markers whose gamedata.json record is INACTIVE in the game scene (`active: false` —
/// disabled exfils like factory's Gate 2, low-power minefields, off sniper zones, disabled
/// doors/loot points). The panel's global "hide inactive" filter
/// (`ui::LayerToggles::hide_inactive`) hides them; their cards already say "Inactive in scene".
#[derive(Component)]
pub struct SceneInactive;

/// Tag on the translucent zone-wall ribbon entities so the panel's per-layer marker COUNTS
/// and the "route: nearest extract" destination query can exclude them (a wall shares its
/// zone's `PoiLayer` for visibility, but it is scenery, not a marker: no `MarkerInfo`, never
/// pickable, its transform is identity with world-space verts).
#[derive(Component)]
pub struct ZoneWall;

/// User-specified zone-wall height: "extrude the boundary line upwards … only go up about
/// 1.5 m", fading to fully transparent at the top.
const WALL_HEIGHT: f32 = 1.5;
/// Wall opacity at the BASE of the fade ramp (alpha 0 at the top).
const WALL_BASE_ALPHA: f32 = 0.35;

/// Triangle-strip-style ribbon around a closed outline: base ring = the (terrain-draped)
/// outline verts, top ring = the same verts +`height` Y; vertex colours carry the layer
/// colour with an alpha ramp (base_alpha -> 0), so one shared unlit
/// `AlphaMode::Blend`/`cull_mode: None` white material renders every wall (Bevy multiplies
/// vertex colours into the material). Blend alpha also disables depth WRITE, so walls never
/// occlude markers.
fn zone_wall_mesh(outline: &[Vec3], color: Color, base_alpha: f32) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::Indices;
    use bevy::render::render_resource::PrimitiveTopology;
    let n = outline.len();
    let lin = color.to_linear();
    let mut pos = Vec::with_capacity(n * 2);
    let mut col = Vec::with_capacity(n * 2);
    let mut nrm = Vec::with_capacity(n * 2);
    for p in outline {
        pos.push([p.x, p.y, p.z]);
        pos.push([p.x, p.y + WALL_HEIGHT, p.z]);
        col.push([lin.red, lin.green, lin.blue, base_alpha]);
        col.push([lin.red, lin.green, lin.blue, 0.0]);
        // unlit material ignores normals; present only to satisfy the mesh vertex layout.
        nrm.push([0.0, 1.0, 0.0]);
        nrm.push([0.0, 1.0, 0.0]);
    }
    let mut idx = Vec::with_capacity(n * 6);
    for i in 0..n {
        let a = (i * 2) as u32;
        let b = (((i + 1) % n) * 2) as u32;
        idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, nrm);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, col);
    mesh.insert_indices(Indices::U32(idx));
    mesh
}

/// Even-odd point-in-polygon test on the XZ plane (zone footprints are XZ polygons; Y is
/// terrain drape). Used to prune tarkov.dev hazard POINTS that a typed zone already covers.
fn point_in_poly_xz(p: Vec3, poly: &[Vec3]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.z > p.z) != (b.z > p.z) && p.x < a.x + (p.z - a.z) / (b.z - a.z) * (b.x - a.x) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

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

/// Startup-built zone outlines from `gamedata.json` (TYPED game-file data —
/// extract_gamedata.py). `live` flips the UI footer to credit the game files and marks that the
/// typed exfils replaced the tarkov.dev `extracts_dev` layer. Outline verts are already viewer
/// space; with `draped` set they were terrain-draped at EXTRACTION time (subdivided ~4 m,
/// Y = max(terrain+0.3, collider base)) so `draw_gamedata_outlines` adds only a token lift —
/// undraped (old files / indoor maps) outlines sit at the collider's bottom face and keep the
/// bigger lift. Each zone carries its scene-active flag so the outlines can follow the panel's
/// "hide inactive" filter exactly like the markers.
#[derive(Resource, Default)]
pub struct GameDataZones {
    pub live: bool,
    pub draped: bool,
    /// (faction, footprint, scene-active) per typed exfil — drawn with the Extracts toggle.
    pub exfils: Vec<(String, Vec<Vec3>, bool)>,
    pub minefields: Vec<(Vec<Vec3>, bool)>,
    pub sniper_zones: Vec<(Vec<Vec3>, bool)>,
    /// Directional-mine blast boxes + special compound zones — Hazards toggle.
    pub hazard_zones: Vec<(Vec<Vec3>, bool)>,
    /// Typed transit footprints — Transits toggle.
    pub transits: Vec<(Vec<Vec3>, bool)>,
}

pub struct PoiPlugin;
impl Plugin for PoiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<QuestData>()
            .init_resource::<KeyCatalog>()
            .init_resource::<GameDataZones>()
            // Build ALL per-map POI markers + zone walls (and the QuestData/KeyCatalog/GameDataZones
            // resources) on each MapEpoch — the initial epoch-0 insert included — despawning the old
            // map's first. (Command ordering: teardown's despawns are queued before spawn's inserts.)
            .add_systems(
                Update,
                (teardown_pois, spawn_pois)
                    .chain()
                    .run_if(resource_changed::<crate::render::MapEpoch>),
            )
            // Quest markers get their own visibility pass (toggle AND tracker selection); the other
            // POI layers stay on the plain toggle-driven `apply_poi_visibility`. Ordered AFTER
            // spawn_pois so the auto-inserted sync point makes the fresh markers visible on a swap.
            .add_systems(Update, (apply_poi_visibility, apply_quest_visibility).after(spawn_pois))
            .add_systems(Update, (draw_quest_outlines, draw_gamedata_outlines));
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
        PoiLayer::Minefield => (Color::srgb(0.95, 0.26, 0.20), 0.9, 1.0),
        PoiLayer::SniperZone => (Color::srgb(0.95, 0.60, 0.15), 0.9, 1.1),
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
    /// Boss slug, e.g. "cultist-priest" (boss_nodes only).
    #[serde(default)]
    boss: Option<String>,
    /// Boss spawn chance 0..1 (boss_nodes only).
    #[serde(default)]
    chance: Option<f32>,
    /// Boss DISPLAY name from tarkov.dev, e.g. "Cultist Priest" (boss_nodes only).
    #[serde(default)]
    name: Option<String>,
    /// Spawn delay in seconds after raid start; -1 = spawns at start (boss_nodes only).
    #[serde(default)]
    st: Option<i64>,
    /// Escort/guard groups (boss_nodes only).
    #[serde(default)]
    escorts: Vec<Escort>,
    /// Named spawn zones with per-zone chance (boss_nodes only).
    #[serde(default)]
    locs: Vec<BossLoc>,
}
/// One escort/guard group on a boss node, e.g. 4x reshala-guard.
#[derive(Deserialize)]
struct Escort {
    #[serde(default)]
    boss: String,
    #[serde(default)]
    count: Option<u32>,
    /// Presence chance 0..1 (schema fidelity; near-always 1.0, so not surfaced).
    #[serde(default)]
    #[allow(dead_code)]
    chance: Option<f32>,
}
/// One named boss spawn zone and its share of the spawn chance.
#[derive(Deserialize)]
struct BossLoc {
    #[serde(default)]
    name: String,
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

/// gamedata.json (extract_gamedata.py) — TYPED gameplay data read from the map's Unity logic
/// scene (ExfiltrationPoint/Minefield/SniperFiringZone/Door MonoBehaviours — ground truth, not
/// name classification). Positions/outlines are already viewer space. Only the sections the
/// viewer consumes are declared; extra sections (spawn_points, transit_points, ...) stay in the
/// file for other tools.
#[derive(Deserialize)]
struct GameDataFile {
    /// Outlines were terrain-draped at extraction time (see `GameDataZones::draped`).
    #[serde(default)]
    draped: bool,
    #[serde(default)]
    exfils: Vec<GdExfil>,
    #[serde(default)]
    minefields: Vec<GdZone>,
    #[serde(default)]
    sniper_zones: Vec<GdZone>,
    #[serde(default)]
    doors: Vec<GdDoor>,
    #[serde(default)]
    stationary: Vec<GdStationary>,
    /// First-party LOOSE-LOOT points (LootPoint MonoBehaviours) with their client-side item
    /// POOLS — the small curated set the client actually ships (gun racks / safes / piles).
    #[serde(default)]
    loose_points: Vec<GdLoose>,
    /// Directional-mine (claymore) blast zones — largest child BoxCollider per mine.
    #[serde(default)]
    mines_directional: Vec<GdZone>,
    /// Typed quest/visit trigger zones (PlaceItemTrigger / ExperienceTrigger / Flare…).
    #[serde(default)]
    quest_triggers: Vec<GdZone>,
    /// Special compound zones (LighthouseTraderZone) — shown on the Hazard layer.
    #[serde(default)]
    trader_zones: Vec<GdZone>,
    /// Transit outlines already ship inside `transit_points`.
    #[serde(default)]
    transit_points: Vec<GdZone>,
}
fn default_true() -> bool {
    true
}
/// A typed extract: faction comes from the COMPONENT TYPE (ExfiltrationPoint = pmc,
/// ScavExfiltrationPoint = scav, Shared… = shared, Secret… = secret).
#[derive(Deserialize)]
struct GdExfil {
    pos: [f32; 3],
    #[serde(default)]
    name: String,
    #[serde(default)]
    faction: String,
    /// BoxCollider footprint corners (viewer space, ground height); empty when none.
    #[serde(default)]
    outline: Vec<[f32; 3]>,
    #[serde(default = "default_true")]
    active: bool,
}
/// A typed zone (Minefield / SniperFiringZone / MineDirectional / quest trigger / trader
/// zone) with its collider footprint.
#[derive(Deserialize)]
struct GdZone {
    pos: [f32; 3],
    #[serde(default)]
    name: Option<String>,
    /// Zone flavour: the mine model ("MON-50") on `mines_directional`, the trigger class
    /// ("place_item"/"visit"/"flare"/"quest") on `quest_triggers`; absent elsewhere.
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    outline: Vec<[f32; 3]>,
    /// Pre-subdivision footprint [w, d] m — the outline itself may be terrain-subdivided, so
    /// the card's "~W x D m" line can no longer be derived from its first three verts.
    #[serde(default)]
    extent: Option<[f32; 2]>,
    #[serde(default = "default_true")]
    active: bool,
}
/// A typed Door/Trunk MonoBehaviour: `key_id` is the game's item id for the key; `state` the
/// serialized initial EDoorState ("locked"/"shut"/"open"/"breach").
#[derive(Deserialize)]
struct GdDoor {
    pos: [f32; 3],
    #[serde(default)]
    key_id: Option<String>,
    #[serde(default)]
    state: Option<String>,
    /// "door" | "trunk".
    #[serde(default)]
    kind: String,
    /// Raw GameObject name (prettified for the card).
    #[serde(default)]
    name: Option<String>,
    #[serde(default = "default_true")]
    active: bool,
}
#[derive(Deserialize)]
struct GdStationary {
    pos: [f32; 3],
    #[serde(default)]
    name: String,
    #[serde(default = "default_true")]
    active: bool,
}

/// One first-party loose-loot point (gamedata.json `loose_points`): a LootPoint transform +
/// its serialized item pool. `items` is sorted by the extractor — priced real items first,
/// category slots ("Food and drink") last — so `items[0]` is the card's best item.
#[derive(Deserialize)]
struct GdLoose {
    pos: [f32; 3],
    #[serde(default)]
    name: Option<String>,
    /// Spawn slots merged into this point (several LootPoints at ~one spot).
    #[serde(default)]
    n: u32,
    #[serde(default = "default_true")]
    active: bool,
    #[serde(default)]
    items: Vec<GdLooseItem>,
    /// "game files" (pool from the LootPoint payload) or "tarkov.dev" (template-less point
    /// filled from the snapshot's co-located lootLoose entry). Absent = no items known.
    #[serde(default)]
    items_src: Option<String>,
}
/// One pool entry. `cat == 1` marks a CATEGORY template ("Meds") — unpriced, no icon.
#[derive(Deserialize)]
struct GdLooseItem {
    /// Raw 24-hex template id (the fallback label when the name lookup was offline).
    #[serde(default)]
    tpl: Option<String>,
    #[serde(default)]
    n: Option<String>,
    #[serde(default)]
    s: Option<String>,
    #[serde(default)]
    pr: Option<i64>,
    #[serde(default)]
    cat: i64,
}

/// Approx "W x D m" of a rectangular footprint, for zone cards; None for degenerate outlines.
fn outline_extent(outline: &[[f32; 3]]) -> Option<String> {
    if outline.len() < 3 {
        return None;
    }
    let d = |a: &[f32; 3], b: &[f32; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let (w, h) = (d(&outline[0], &outline[1]), d(&outline[1], &outline[2]));
    (w > 1.0 && h > 1.0).then(|| format!("~{:.0} x {:.0} m", w, h))
}

/// Card for a typed extract (gamedata.json `exfils`). Same look as the tarkov.dev card it
/// replaces, plus the game-file provenance and an inactive tag. `friendly` is the community name
/// resolved from the tarkov.dev extract list ("Railway Exfil" for scene id "NW Exfil") — it
/// becomes the display name, with the raw scene id kept as a detail line.
fn gd_exfil_info(e: &GdExfil, friendly: Option<&str>) -> MarkerInfo {
    let display = friendly.unwrap_or(e.name.as_str());
    let mut info = extract_dev_info(display, &e.faction);
    if friendly.is_some_and(|f| f != e.name) && !e.name.is_empty() {
        info.detail.push(format!("Scene id: {}", e.name));
    }
    if !e.active {
        info.detail.push("Inactive in scene".into());
    }
    info.detail.push("Source: game files".into());
    info
}

/// Card for a typed minefield / sniper zone (gamedata.json).
fn gd_zone_info(l: PoiLayer, z: &GdZone) -> MarkerInfo {
    let (title, subtitle) = match l {
        PoiLayer::SniperZone => ("Sniper zone", "Hazard \u{00B7} game files"),
        _ => ("Minefield", "Hazard \u{00B7} game files"),
    };
    let mut detail = Vec::new();
    // Pre-subdivision extent preferred — on a terrain-draped outline the first three verts
    // lie along ONE edge, so the derived heuristic would read ~4 x 4 m.
    if let Some(ext) = zone_extent_line(z) {
        detail.push(ext);
    }
    // The raw GameObject name distinguishes zones ("Minefield LowPower25", "SniperFiringZone
    // Right") when several sit close together.
    if let Some(n) = z.name.as_deref().map(prettify).filter(|n| !n.is_empty()) {
        detail.push(n);
    }
    if !z.active {
        detail.push("Inactive in scene".into());
    }
    MarkerInfo {
        title: title.into(),
        subtitle: subtitle.into(),
        detail,
        accent: poi_look(l).0,
    }
}

/// Extent line for a zone card: prefer the extractor's pre-subdivision extent, fall back to
/// the first-three-verts heuristic (correct only on undraped 4-corner outlines).
fn zone_extent_line(z: &GdZone) -> Option<String> {
    if let Some([w, d]) = z.extent.filter(|[w, d]| *w > 1.0 && *d > 1.0) {
        return Some(format!("~{:.0} x {:.0} m", w, d));
    }
    outline_extent(&z.outline)
}

/// Card for a directional mine (gamedata.json `mines_directional`) — the zone is the mine's
/// blast/trigger box, so the card leads with the mine model (MON-50).
fn gd_mine_info(z: &GdZone) -> MarkerInfo {
    let title = z
        .kind
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|k| format!("{k} directional mine"))
        .unwrap_or_else(|| "Directional mine".into());
    let mut detail = Vec::new();
    if let Some(ext) = zone_extent_line(z) {
        detail.push(ext);
    }
    if !z.active {
        detail.push("Inactive in scene".into());
    }
    detail.push("game files".into());
    MarkerInfo {
        title,
        subtitle: "Hazard \u{00B7} game files".into(),
        detail,
        accent: poi_look(PoiLayer::Hazard).0,
    }
}

/// Card for a typed quest/visit trigger zone (gamedata.json `quest_triggers`). The title is
/// the RAW zone id ("qlight_pc1_ucot_kill") — it's the game's quest-zone key, so search hits
/// it directly.
fn gd_trigger_info(z: &GdZone) -> MarkerInfo {
    let kind_line = match z.kind.as_deref() {
        Some("place_item") => "Place-item zone",
        Some("visit") => "Visit / exploration zone",
        Some("flare") => "Flare-signal zone",
        _ => "Quest trigger",
    };
    let mut detail = vec![kind_line.to_string()];
    if let Some(ext) = zone_extent_line(z) {
        detail.push(ext);
    }
    if !z.active {
        detail.push("Inactive in scene".into());
    }
    MarkerInfo {
        title: z.name.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "Quest zone".into()),
        subtitle: "Quest zone \u{00B7} game files".into(),
        detail,
        accent: poi_look(PoiLayer::Quest).0,
    }
}

/// Card for a special compound zone (gamedata.json `trader_zones` — LighthouseTraderZone).
fn gd_trader_info(z: &GdZone) -> MarkerInfo {
    let mut detail = Vec::new();
    if let Some(ext) = zone_extent_line(z) {
        detail.push(ext);
    }
    if !z.active {
        detail.push("Inactive in scene".into());
    }
    MarkerInfo {
        title: z
            .name
            .as_deref()
            .map(prettify)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Zone".into()),
        subtitle: "Zone \u{00B7} game files".into(),
        detail,
        accent: poi_look(PoiLayer::Hazard).0,
    }
}

/// Card for a typed Door/Trunk (gamedata.json). `dev_key` is the tarkov.dev key NAME when a
/// loot.json lock sits within 2 m (proximity cross-check) — the human-readable answer to the
/// raw `key_id`.
fn gd_door_info(d: &GdDoor, dev_key: Option<&str>) -> MarkerInfo {
    let kind = if d.kind == "trunk" { "Trunk" } else { "Door" };
    let title = dev_key
        .map(|s| s.to_string())
        .or_else(|| d.name.as_deref().map(prettify).filter(|s| !s.is_empty()))
        .unwrap_or_else(|| kind.to_string());
    let mut detail = Vec::new();
    if let Some(k) = d.key_id.as_deref().filter(|k| !k.is_empty()) {
        detail.push(format!("Key: {k}"));
        if let Some(n) = dev_key {
            detail.push(format!("= {n} (tarkov.dev)"));
        }
    }
    if let Some(st) = d.state.as_deref() {
        detail.push(format!("State: {}", titlecase(st)));
    }
    if !d.active {
        detail.push("Inactive in scene".into());
    }
    // Locked doors read in the locks-gold so the state is visible at card level; the rest keep
    // the door orange.
    let accent = if d.state.as_deref() == Some("locked") {
        poi_look(PoiLayer::Lock).0
    } else {
        poi_look(PoiLayer::Door).0
    };
    MarkerInfo {
        title,
        subtitle: format!("{kind} \u{00B7} game files"),
        detail,
        accent,
    }
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
            // Prefer the tarkov.dev display name ("Cultist Priest") — titlecasing the slug
            // gives "Cultist-priest".
            let title = nd
                .name
                .clone()
                .filter(|s| !s.is_empty())
                .or_else(|| nd.boss.as_deref().map(titlecase).filter(|s| !s.is_empty()))
                .unwrap_or_else(|| "Boss".into());
            let mut detail = Vec::new();
            if let Some(ch) = nd.chance {
                detail.push(format!("Chance {:.0}%", ch * 100.0));
            }
            // Delayed spawns (e.g. Partisan at 900 s) are raid-planning gold; -1/small
            // values mean "at raid start" and stay silent.
            if let Some(st) = nd.st.filter(|&s| s >= 60) {
                detail.push(format!("Spawns ~{} min in", (st + 30) / 60));
            }
            if !nd.escorts.is_empty() {
                let parts: Vec<String> = nd
                    .escorts
                    .iter()
                    .map(|e| {
                        let who = titlecase(&e.boss.replace('-', " "));
                        match e.count {
                            Some(c) if c > 0 => format!("{c}\u{00D7} {who}"),
                            _ => who,
                        }
                    })
                    .collect();
                detail.push(format!("Escorts: {}", parts.join(", ")));
            }
            if !nd.locs.is_empty() {
                let parts: Vec<String> = nd
                    .locs
                    .iter()
                    .map(|l| match l.chance {
                        Some(c) if c > 0.0 && c < 1.0 => {
                            format!("{} {:.0}%", l.name, c * 100.0)
                        }
                        _ => l.name.clone(),
                    })
                    .collect();
                detail.push(format!("Zones: {}", parts.join("  \u{00B7}  ")));
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

/// Card contents for a semantics.json POI (extract/door/loot prop). The "loot" layer is
/// NAME-classified props (jackets/weapon boxes/safes) — ground truth showed it mixes real
/// lootables with decorative variants, so the copy says "Loot prop", not "Interactable".
fn sem_info(l: PoiLayer, poi: &Poi) -> MarkerInfo {
    let accent = poi_look(l).0;
    let pretty = prettify(&poi.name);
    let (fallback, subtitle) = match l {
        PoiLayer::Extract => ("Extract", "Extract"),
        PoiLayer::Door => ("Door", "Door"),
        _ => ("Loot prop", "Loot prop"),
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
/// A merged multi-faction extract ("pmc+scav", see the merge in `spawn_pois`) is usable by
/// everyone on the map, so it reads "All" too.
fn faction_label(fac: &str) -> String {
    match fac {
        "pmc" => "PMC".into(),
        "scav" => "Scav".into(),
        "shared" => "All".into(),
        "secret" => "Secret".into(),
        _ if fac.contains('+') => "All".into(),
        _ => titlecase(fac),
    }
}

/// Marker/accent colour per extract faction. PMC keeps the layer's extract green; scav-only
/// reads amber; shared / merged multi-faction reads a pale white-green.
fn extract_faction_color(fac: &str) -> Color {
    match fac {
        "scav" => Color::srgb(0.95, 0.45, 0.10),
        "shared" => Color::srgb(0.78, 0.95, 0.82),
        "secret" => Color::srgb(0.85, 0.30, 0.90),
        _ if fac.contains('+') => Color::srgb(0.78, 0.95, 0.82),
        _ => poi_look(PoiLayer::Extract).0,
    }
}

/// Card for a clean faction-tagged extract (loot.json `extracts_dev`). `fac` may be a merged
/// "pmc+scav" (one physical extract listed once per faction by tarkov.dev).
fn extract_dev_info(name: &str, fac: &str) -> MarkerInfo {
    let accent = extract_faction_color(fac);
    let name = if name.is_empty() { "Extract" } else { name };
    // Faction in the title (e.g. "Armored Train  [All]") so search hits "pmc"/"scav".
    let title = if fac.is_empty() {
        name.to_string()
    } else {
        format!("{}  [{}]", name, faction_label(fac))
    };
    // Spell out the per-faction breakdown for merged extracts ("Faction: PMC + Scav").
    let fac_line = if fac.contains('+') {
        fac.split('+').map(faction_label).collect::<Vec<_>>().join(" + ")
    } else {
        faction_label(fac)
    };
    MarkerInfo {
        title,
        subtitle: "Extract".into(),
        detail: vec![format!("Faction: {fac_line}")],
        accent,
    }
}

/// Card for a locked door/container/trunk (loot.json `locks`).
fn lock_info(lk: &Lock) -> MarkerInfo {
    // Keycard locks read violet; ordinary locks keep the gold layer colour. ANY key being a
    // card counts (build_loot.py ships exactly one key per lock today — audited 318 locks,
    // all single-key — but the rule must not silently miss a card at a later position).
    let keycard = lk.keys.iter().any(|k| k.card == 1);
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
    // Destination for the title: `to` is a slug ("the-lab-dark" titlecases to "The-lab-dark"),
    // while the desc usually carries the clean display name ("Transit to The Lab") — prefer
    // that; otherwise de-hyphenate the slug. When the desc IS the title's source it's fully
    // redundant, so it's skipped from the detail lines.
    let (dest, desc_line) = match tr.desc.strip_prefix("Transit to ") {
        Some(d) if !d.is_empty() => (d.to_string(), None),
        _ => (
            titlecase(&tr.to.replace('-', " ")),
            (!tr.desc.is_empty()).then(|| tr.desc.clone()),
        ),
    };
    let mut detail = Vec::new();
    if let Some(d) = desc_line {
        detail.push(d);
    }
    if !tr.cond.is_empty() {
        detail.push(tr.cond.clone());
    }
    MarkerInfo {
        title: format!("Transit \u{2192} {dest}"),
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

/// Card for a FIRST-PARTY loose-loot point (gamedata.json `loose_points`): titled by the
/// spawn's GameObject name, detailing the best pool item + price, the pool size, and the
/// provenance split the honesty check settled on — 'game files' when the item pool came out
/// of the LootPoint payload itself, 'pos: game files · items: tarkov.dev' when only the
/// position is first-party and the items were joined from the snapshot.
fn gd_loose_info(lo: &GdLoose) -> MarkerInfo {
    let title = lo
        .name
        .as_deref()
        .map(prettify)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Loose loot".into());
    let mut detail = Vec::new();
    if let Some(best) = lo.items.first() {
        let label = best
            .n
            .clone()
            .or_else(|| best.s.clone())
            .or_else(|| best.tpl.clone())
            .unwrap_or_default();
        if !label.is_empty() {
            detail.push(if best.cat == 1 { format!("{label} (category)") } else { label });
        }
        if let Some(pr) = best.pr.filter(|&p| p > 0) {
            detail.push(format!("Value  {}", money(pr)));
        }
        if lo.items.len() > 1 {
            detail.push(format!("Pool: {} possible", lo.items.len()));
        }
    }
    if lo.n > 1 {
        detail.push(format!("\u{00D7}{} spawn slots", lo.n));
    }
    if !lo.active {
        detail.push("Inactive in scene".into());
    }
    detail.push(match lo.items_src.as_deref() {
        Some("tarkov.dev") => "pos: game files \u{00B7} items: tarkov.dev".into(),
        _ => "game files".into(),
    });
    MarkerInfo {
        title,
        subtitle: "Loose loot \u{00B7} game files".into(),
        detail,
        accent: poi_look(PoiLayer::LooseLoot).0,
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

/// Canonical map key for intel joins: the manifest's `map` (canonical id) if present, else the
/// dataset dir basename with a `_vN` suffix stripped (e.g. `interchange_v2` -> `interchange`).
fn map_key(manifest: &crate::eftpack::Manifest) -> String {
    if !manifest.map.is_empty() {
        return manifest.map.clone();
    }
    let dataset = &manifest.dataset;
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
    // Shared tier (packs/shared/): map-agnostic sidecars (tasks.json) live above the packs.
    if let Some(shared) = root.parent().map(|p| p.join("shared").join(file)) {
        if shared.is_file() {
            return Some(shared);
        }
    }
    let shared = crate::paths::shared_dir().join(file);
    if shared.is_file() {
        return Some(shared);
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
        PoiLayer::Minefield,
        PoiLayer::SniperZone,
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
    let ex_secret = mk(&mut materials, extract_faction_color("secret"));
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

    // ---- ZONE WALLS: one translucent ribbon per zone outline (user spec: the boundary line
    // extruded 1.5 m up, fading transparent with height). ONE shared unlit white
    // blend/double-sided material; the layer colour + alpha ramp live in vertex colours.
    // Walls carry their zone's PoiLayer (+SceneInactive) so the SAME visibility systems that
    // drive the markers drive them; `ZoneWall` keeps them out of panel counts and routing.
    let wall_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        double_sided: true,
        ..default()
    });
    let wall = |commands: &mut Commands,
                meshes: &mut Assets<Mesh>,
                l: PoiLayer,
                outline: &[Vec3],
                color: Color,
                active: bool|
     -> Option<Entity> {
        if outline.len() < 3 {
            return None;
        }
        let e = commands
            .spawn((
                Mesh3d(meshes.add(zone_wall_mesh(outline, color, WALL_BASE_ALPHA))),
                MeshMaterial3d(wall_mat.clone()),
                Transform::IDENTITY, // verts are world space already
                l,
                ZoneWall,
                bevy::light::NotShadowCaster,
                Visibility::Hidden,
            ))
            .id();
        if !active {
            commands.entity(e).insert(SceneInactive);
        }
        Some(e)
    };

    // ---- TYPED game data (gamedata.json, extract_gamedata.py) is loaded FIRST because its
    // typed exfils/doors supersede both the tarkov.dev `extracts_dev` list and the
    // name-classified semantics layers below (the classifier's extract layer measured a 71%
    // false-positive rate; the typed MonoBehaviours are ground truth).
    let gamedata: Option<GameDataFile> = resolve_sidecar(
        "EFT_GAMEDATA_JSON",
        "gamedata.json",
        root,
        "typed exfils/minefields/sniper zones unavailable (tarkov.dev extracts still shown)",
    )
    .and_then(|p| std::fs::read_to_string(p).ok())
    .and_then(|s| serde_json::from_str::<GameDataFile>(&s).ok());
    let have_gd_extracts = gamedata.as_ref().is_some_and(|g| !g.exfils.is_empty());
    let have_gd_doors = gamedata.as_ref().is_some_and(|g| !g.doors.is_empty());
    // First-party loose-loot positions, known BEFORE the loot.json pass: a tarkov.dev loose
    // point within 2.5 m of one duplicates it (same physical spawn seen by the snapshot) and
    // is skipped — the typed point wins (exact transform + the game's own pool + provenance).
    let gd_loose_pos: Vec<Vec3> = gamedata
        .as_ref()
        .map(|g| g.loose_points.iter().map(|l| Vec3::from(l.pos)).collect())
        .unwrap_or_default();
    // Typed hazard-carrying footprints (minefields + sniper zones + mine blast boxes + special
    // compound zones), known BEFORE the loot.json pass: tarkov.dev "hazards" is a POINT GRID
    // sampled over these same areas (lighthouse: 344 points, ~5 m spacing over the mine
    // belts), so any dev point INSIDE a typed footprint is the same hazard already shown as a
    // zone — skip its marker. Points no typed zone covers always ship (never an empty layer).
    let hazard_polys: Vec<Vec<Vec3>> = gamedata
        .as_ref()
        .map(|g| {
            g.minefields
                .iter()
                .chain(g.sniper_zones.iter())
                .chain(g.mines_directional.iter())
                .chain(g.trader_zones.iter())
                .filter(|z| z.outline.len() >= 3)
                .map(|z| z.outline.iter().map(|a| Vec3::from(*a)).collect())
                .collect()
        })
        .unwrap_or_default();

    // ---- spawns + map intel from loot.json (pmc/scav/boss nodes, locks, hazards, ...) ----
    // Set when loot.json ships clean faction-tagged extracts; those supersede the semantics
    // `extract` layer below.
    let mut have_dev_extracts = false;
    let mut have_dev_stationary = false;
    // Aggregated per-key lock list, folded into `KeyCatalog` below (panel's "Keys for this map").
    let mut key_uses: Vec<KeyUse> = Vec::new();
    // tarkov.dev loose points skipped because a first-party LootPoint owns that spot.
    let mut pruned_dev_loose = 0usize;
    // Every loot.json lock position + its key's display name, for the typed-door proximity
    // cross-check (a tarkov.dev lock within 2 m names the door's `key_id`).
    let mut lock_keys: Vec<(Vec3, String)> = Vec::new();
    // tarkov.dev extract names + positions: the COMMUNITY names players actually know ("Railway
    // Exfil", "Emercom Checkpoint"). The typed gamedata exfils carry raw scene ids ("NW Exfil",
    // "SE Exfil"), so each is renamed to the nearest dev extract within 60 m (XZ) below.
    let mut dev_extract_names: Vec<(String, Vec3)> = Vec::new();
    let key = map_key(&lp.0.manifest);
    // ONE loot.json resolver for the whole app (loot.rs: env > pack > pack-parent shared >
    // shared_dir > cwd) - poi.rs used to re-implement a subset (audit A6).
    let loot_path = crate::loot::resolve_loot_json(Some(lp.0.root.as_path()));
    if let Some(mn) = loot_path
        .and_then(|pb| std::fs::read_to_string(pb).ok())
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
            // Keycard locks get the violet marker material (matches lock_info's accent —
            // ANY key being a card counts, same rule as there).
            let keycard = lk.keys.iter().any(|k| k.card == 1);
            let e = spawn(
                &mut commands,
                PoiLayer::Lock,
                lk.pos,
                lock_info(lk),
                keycard.then(|| keycard_mat.clone()),
            );
            // The key's icon on the lock card (first named key = the card's title/needs line).
            if let Some(k) = lk.keys.iter().find(|k| !k.n.is_empty()) {
                commands.entity(e).insert(MarkerIcon(icon_slug(&k.n)));
            }
            // Fold EVERY key into the catalog, grouped by key name (today each lock ships
            // exactly one key, but an alternate key must not vanish from the list).
            if let Some(k) = lk.keys.iter().find(|k| !k.n.is_empty()) {
                lock_keys.push((Vec3::new(lk.pos[0], lk.pos[1], lk.pos[2]), k.n.clone()));
            }
            for k in lk.keys.iter().filter(|k| !k.n.is_empty()) {
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
        let mut pruned_dev_hazards = 0usize;
        for hz in &mn.hazards {
            // A typed zone footprint covering this point owns the hazard — the zone (outline
            // + wall) shows there; the point marker would only stack on top of it.
            let p = Vec3::from(hz.pos);
            if hazard_polys.iter().any(|poly| point_in_poly_xz(p, poly)) {
                pruned_dev_hazards += 1;
                continue;
            }
            spawn(&mut commands, PoiLayer::Hazard, hz.pos, hazard_info(hz), None);
        }
        if pruned_dev_hazards > 0 {
            info!(
                "poi: {pruned_dev_hazards}/{} tarkov.dev hazard points covered by typed zone footprints (pruned)",
                mn.hazards.len()
            );
        }
        have_dev_stationary = !mn.stationary.is_empty();
        for st in &mn.stationary {
            spawn(&mut commands, PoiLayer::Stationary, st.pos, stationary_info(st), None);
        }
        for lo in &mn.loose {
            // A typed first-party point within 2.5 m owns this spawn — skip the snapshot twin.
            let p = Vec3::from(lo.pos);
            if gd_loose_pos.iter().any(|q| q.distance_squared(p) < 2.5 * 2.5) {
                pruned_dev_loose += 1;
                continue;
            }
            // Tagged with the item price so the panel's min-value filter applies (0 = unpriced).
            let e = spawn(&mut commands, PoiLayer::LooseLoot, lo.pos, loose_info(lo), None);
            commands.entity(e).insert(MarkerValue(lo.pr.unwrap_or(0)));
            // Item icon (cached per map by fetch_icons.py; missing file = no icon).
            if !lo.n.is_empty() {
                commands.entity(e).insert(MarkerIcon(icon_slug(&lo.n)));
            }
        }
        // Community extract names for the typed-exfil rename (kept whether or not the dev
        // markers themselves spawn — the names matter either way).
        dev_extract_names = mn
            .extracts_dev
            .iter()
            .filter(|ex| !ex.name.is_empty())
            .map(|ex| (ex.name.clone(), Vec3::from(ex.pos)))
            .collect();
        // Prefer the clean faction-tagged extract list when it's present — unless the TYPED
        // exfils from gamedata.json already own the Extract layer (they include secret
        // extracts and exact collider footprints; tarkov.dev stays the fallback).
        if !mn.extracts_dev.is_empty() && !have_gd_extracts {
            have_dev_extracts = true;
            // tarkov.dev lists one entry PER FACTION, so the same physical extract can appear
            // twice — pmc + scav at (nearly) the same spot (6 such pairs across the 10 maps,
            // 2 at IDENTICAL coordinates, which z-fight as stacked spheres and hide the second
            // faction). Merge same-name entries within 3 m into ONE marker carrying every
            // faction ("pmc+scav" -> "[All]").
            let mut merged: Vec<(&ExtractDev, String)> = Vec::new();
            for ex in &mn.extracts_dev {
                let close = |a: &[f32; 3], b: &[f32; 3]| {
                    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2) < 9.0
                };
                if let Some((_, facs)) = merged
                    .iter_mut()
                    .find(|(f, _)| f.name == ex.name && close(&f.pos, &ex.pos))
                {
                    if !facs.split('+').any(|f| f == ex.fac) {
                        facs.push('+');
                        facs.push_str(&ex.fac);
                    }
                } else {
                    merged.push((ex, ex.fac.clone()));
                }
            }
            for (ex, fac) in &merged {
                let mat = match fac.as_str() {
                    "scav" => Some(ex_scav.clone()),
                    "shared" => Some(ex_shared.clone()),
                    f if f.contains('+') => Some(ex_shared.clone()),
                    _ => None, // pmc / unknown keep the layer's extract green
                };
                let e = spawn(
                    &mut commands,
                    PoiLayer::Extract,
                    ex.pos,
                    extract_dev_info(&ex.name, fac),
                    mat,
                );
                commands.entity(e).insert(ExtractFaction(fac.clone()));
            }
        }
    }

    // ---- TYPED game data markers + zone outlines (gamedata.json) ----
    // Exfil markers REPLACE extracts_dev (gated above); minefields / sniper zones are their
    // own layers; typed doors replace the semantics door layer and carry KeyId/DoorState.
    let mut gd_zones = GameDataZones::default();
    let mut have_gd_stationary = false;
    if let Some(gd) = &gamedata {
        gd_zones.live = have_gd_extracts || !gd.minefields.is_empty() || !gd.sniper_zones.is_empty();
        gd_zones.draped = gd.draped;
        for e in &gd.exfils {
            let mat = match e.faction.as_str() {
                "scav" => Some(ex_scav.clone()),
                "shared" => Some(ex_shared.clone()),
                "secret" => Some(ex_secret.clone()),
                _ => None, // pmc keeps the layer's extract green
            };
            // Rename to the community name: nearest tarkov.dev extract within 60 m (XZ — the Y
            // conventions differ: collider bottom vs surface point). NON-exclusive on purpose: a
            // physical extract listed per-faction in the scene ("NW Exfil" pmc + scav) maps both
            // entries to the one dev extract ("Railway Exfil").
            let friendly = dev_extract_names
                .iter()
                .map(|(n, p)| {
                    let d = ((p.x - e.pos[0]).powi(2) + (p.z - e.pos[2]).powi(2)).sqrt();
                    (n, d)
                })
                .filter(|(_, d)| *d <= 60.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(n, _)| n.clone());
            let ent = spawn(
                &mut commands,
                PoiLayer::Extract,
                e.pos,
                gd_exfil_info(e, friendly.as_deref()),
                mat,
            );
            commands.entity(ent).insert(ExtractFaction(e.faction.clone()));
            if !e.active {
                commands.entity(ent).insert(SceneInactive);
            }
            if e.outline.len() >= 3 {
                let pts: Vec<Vec3> = e.outline.iter().map(|a| Vec3::from(*a)).collect();
                wall(&mut commands, &mut meshes, PoiLayer::Extract, &pts,
                     extract_faction_color(&e.faction), e.active);
                gd_zones.exfils.push((e.faction.clone(), pts, e.active));
            }
        }
        for (zs, layer, sink) in [
            (&gd.minefields, PoiLayer::Minefield, &mut gd_zones.minefields),
            (&gd.sniper_zones, PoiLayer::SniperZone, &mut gd_zones.sniper_zones),
        ] {
            for z in zs {
                let ent = spawn(&mut commands, layer, z.pos, gd_zone_info(layer, z), None);
                if !z.active {
                    commands.entity(ent).insert(SceneInactive);
                }
                if z.outline.len() >= 3 {
                    let pts: Vec<Vec3> = z.outline.iter().map(|a| Vec3::from(*a)).collect();
                    wall(&mut commands, &mut meshes, layer, &pts, poi_look(layer).0, z.active);
                    sink.push((pts, z.active));
                }
            }
        }
        // ---- "anything with a zone" (directional mines / quest triggers / special zones /
        // transit footprints): marker + wall + baseline on their existing layers.
        for z in &gd.mines_directional {
            let ent = spawn(&mut commands, PoiLayer::Hazard, z.pos, gd_mine_info(z), None);
            if !z.active {
                commands.entity(ent).insert(SceneInactive);
            }
            if z.outline.len() >= 3 {
                let pts: Vec<Vec3> = z.outline.iter().map(|a| Vec3::from(*a)).collect();
                wall(&mut commands, &mut meshes, PoiLayer::Hazard, &pts,
                     poi_look(PoiLayer::Hazard).0, z.active);
                gd_zones.hazard_zones.push((pts, z.active));
            }
        }
        for z in &gd.trader_zones {
            let ent = spawn(&mut commands, PoiLayer::Hazard, z.pos, gd_trader_info(z), None);
            if !z.active {
                commands.entity(ent).insert(SceneInactive);
            }
            if z.outline.len() >= 3 {
                let pts: Vec<Vec3> = z.outline.iter().map(|a| Vec3::from(*a)).collect();
                wall(&mut commands, &mut meshes, PoiLayer::Hazard, &pts,
                     poi_look(PoiLayer::Hazard).0, z.active);
                gd_zones.hazard_zones.push((pts, z.active));
            }
        }
        // Quest TRIGGER markers only. The zone FOOTPRINTS (walls + outlines) are intentionally NOT
        // built here: gamedata triggers carry no task id, so they can't be limited to the currently
        // tracked quest. Tracked-quest zones come from the tasks.json path below (`QuestMarkerTask`
        // walls + `draw_quest_outlines`), which the tracker focuses. (User: only show tracked
        // quests' zones — a raw scene-truth footprint for every quest is exactly the clutter to avoid.)
        for z in &gd.quest_triggers {
            let ent = spawn(&mut commands, PoiLayer::Quest, z.pos, gd_trigger_info(z), None);
            if !z.active {
                commands.entity(ent).insert(SceneInactive);
            }
        }
        // Transit MARKERS stay tarkov.dev (loot.json `transits`, richer copy: destination +
        // conditions); the typed footprints add the zone outline + wall those markers lack.
        for z in &gd.transit_points {
            if z.outline.len() >= 3 {
                let pts: Vec<Vec3> = z.outline.iter().map(|a| Vec3::from(*a)).collect();
                wall(&mut commands, &mut meshes, PoiLayer::Transit, &pts,
                     poi_look(PoiLayer::Transit).0, z.active);
                gd_zones.transits.push((pts, z.active));
            }
        }
        // First-party LOOSE-LOOT points. Priced points feed the min-value filter; unpriced
        // ones (category pools / offline ids) stay UNTAGGED so an active filter never hides
        // them — an unknown pool is not "low value". Real best items also get their icon.
        for lo in &gd.loose_points {
            let ent = spawn(&mut commands, PoiLayer::LooseLoot, lo.pos, gd_loose_info(lo), None);
            let best = lo.items.first();
            if let Some(pr) = best.and_then(|b| b.pr).filter(|&p| p > 0) {
                commands.entity(ent).insert(MarkerValue(pr));
            }
            if let Some(n) = best.filter(|b| b.cat == 0).and_then(|b| b.n.as_deref()) {
                commands.entity(ent).insert(MarkerIcon(icon_slug(n)));
            }
            if !lo.active {
                commands.entity(ent).insert(SceneInactive);
            }
        }
        if !gd.loose_points.is_empty() {
            info!(
                "poi: {} first-party loose-loot points (game-file LootPoints; {} tarkov.dev twins pruned within 2.5 m)",
                gd.loose_points.len(),
                pruned_dev_loose
            );
        }
        // Typed doors: cross-check each keyed door against the tarkov.dev locks — a lock
        // within 2 m names the raw key_id on the card.
        let (mut keyed, mut matched) = (0usize, 0usize);
        for d in &gd.doors {
            let p = Vec3::new(d.pos[0], d.pos[1], d.pos[2]);
            let dev_key = d
                .key_id
                .as_deref()
                .filter(|k| !k.is_empty())
                .and_then(|_| {
                    lock_keys
                        .iter()
                        .filter(|(lp, _)| lp.distance_squared(p) < 4.0)
                        .min_by(|(a, _), (b, _)| {
                            a.distance_squared(p).total_cmp(&b.distance_squared(p))
                        })
                        .map(|(_, n)| n.as_str())
                });
            if d.key_id.as_deref().is_some_and(|k| !k.is_empty()) {
                keyed += 1;
                matched += dev_key.is_some() as usize;
            }
            let ent = spawn(&mut commands, PoiLayer::Door, d.pos, gd_door_info(d, dev_key), None);
            // The key ITEM's icon on keyed-door cards, keyed by the tarkov.dev name the card
            // shows (fetch_icons.py also caches these by the door's raw key_id template).
            if let Some(n) = dev_key {
                commands.entity(ent).insert(MarkerIcon(icon_slug(n)));
            }
            if !d.active {
                commands.entity(ent).insert(SceneInactive);
            }
        }
        if !gd.doors.is_empty() {
            info!(
                "poi: gamedata {} typed doors ({} keyed, {} matched a tarkov.dev lock within 2 m)",
                gd.doors.len(),
                keyed,
                matched
            );
        }
        // Typed stationary weapons fill the gap when tarkov.dev shipped none (positions are
        // the game's own StationaryWeapon components — better than mining the geometry).
        if !have_dev_stationary && !gd.stationary.is_empty() {
            have_gd_stationary = true;
            for s in &gd.stationary {
                let ent = spawn(
                    &mut commands,
                    PoiLayer::Stationary,
                    s.pos,
                    MarkerInfo {
                        title: prettify(&s.name),
                        subtitle: "Stationary weapon \u{00B7} game files".into(),
                        detail: Vec::new(),
                        accent: poi_look(PoiLayer::Stationary).0,
                    },
                    None,
                );
                if !s.active {
                    commands.entity(ent).insert(SceneInactive);
                }
            }
        }
        info!(
            "poi: gamedata live — {} exfils, {} minefields, {} sniper zones, {} mine zones, \
             {} quest triggers, {} special zones, {} transit footprints",
            gd.exfils.len(),
            gd.minefields.len(),
            gd.sniper_zones.len(),
            gd.mines_directional.len(),
            gd.quest_triggers.len(),
            gd.trader_zones.len(),
            gd.transit_points.len()
        );
    }
    commands.insert_resource(gd_zones);

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
        // The name classifier overcounts: one physical door splits into leaf/handle/ballistic
        // sub-objects each spawning a marker (lighthouse level198 ground truth: 36 Door
        // components vs 137 name-matched markers), and the "extract" regex catches kitchen
        // "Extractor_Fan/Hood" props and "emergency_exit_light" lamps (45 of lighthouse's 63
        // semantics extracts). Viewer-side guards: drop the known-junk extract names and
        // collapse same-layer markers within 1.2 m (lighthouse doors: 1390 -> ~670).
        let junk_extract = |n: &str| {
            let s = n.to_ascii_lowercase();
            s.contains("extractor_") || s.contains("exit_light")
        };
        for (lname, ly) in map {
            // typed gamedata exfils/doors own these layers when present (ground truth beats
            // the name classifier); tarkov.dev extracts_dev is the next-best extract source.
            if lname == "extract" && (have_dev_extracts || have_gd_extracts) {
                continue;
            }
            if lname == "door" && have_gd_doors {
                continue;
            }
            if let Some(v) = layers.get(lname) {
                let mut kept: Vec<Vec3> = Vec::new();
                for poi in v {
                    if lname == "extract" && junk_extract(&poi.name) {
                        continue;
                    }
                    let p = Vec3::new(poi.p[0], poi.p[1], poi.p[2]);
                    if kept.iter().any(|q| q.distance_squared(p) < 1.2 * 1.2) {
                        continue; // a sub-part of an already-marked object
                    }
                    kept.push(p);
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
                    let outline: Vec<Vec3> =
                        z.outline.iter().map(|a| Vec3::from(*a)).collect();
                    // Wall on the objective footprint too; the `QuestMarkerTask` hands its
                    // visibility to the tracker-aware pass, exactly like the marker's.
                    if let Some(w) = wall(&mut commands, &mut meshes, PoiLayer::Quest, &outline,
                                          poi_look(PoiLayer::Quest).0, true) {
                        commands.entity(w).insert(QuestMarkerTask(t.id.clone()));
                    }
                    zones.push(QuestZoneW {
                        pos: Vec3::new(p[0], p[1], p[2]),
                        outline,
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
    // neither loot.json nor gamedata.json gave us stationary data, scan the pack instances and
    // dedupe co-located receiver parts (one gun = several meshes at ~one spot) into one marker
    // per gun.
    if !have_dev_stationary && !have_gd_stationary {
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
#[allow(clippy::type_complexity)]
/// In-place map swap: despawn every POI marker + zone wall (all carry `PoiLayer`) so `spawn_pois`
/// can rebuild for the new pack. Runs (chained) immediately before `spawn_pois` on a MapEpoch bump.
fn teardown_pois(mut commands: Commands, q: Query<Entity, With<PoiLayer>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn apply_poi_visibility(
    toggles: Res<LayerToggles>,
    epoch: Res<crate::render::MapEpoch>,
    mut q: Query<
        (&PoiLayer, Option<&MarkerValue>, Option<&SceneInactive>, &mut Visibility),
        Without<QuestMarkerTask>,
    >,
) {
    // Re-apply on a toggle change OR a map swap (fresh markers spawn Hidden; the swap didn't touch
    // the toggles, so without the epoch trigger they'd stay invisible).
    if !toggles.is_changed() && !epoch.is_changed() {
        return;
    }
    for (l, val, inactive, mut vis) in &mut q {
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
            PoiLayer::Minefield => toggles.minefields,
            PoiLayer::SniperZone => toggles.sniper_zones,
        };
        // Value-tagged markers (loose loot) additionally pass the panel's min-value filter,
        // and scene-inactive markers the global "hide inactive" filter — both COMPOSE with
        // the layer toggle rather than replacing it.
        let show = show
            && value_passes(toggles.min_value, val)
            && !(toggles.hide_inactive && inactive.is_some());
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
    epoch: Res<crate::render::MapEpoch>,
    mut q: Query<(&QuestMarkerTask, Option<&ZoneWall>, &mut Visibility)>,
) {
    if !toggles.is_changed() && !tracker.is_changed() && !epoch.is_changed() {
        return;
    }
    for (task, is_zone, mut vis) in &mut q {
        let tracked = tracker.active.contains(&task.0);
        // Zone FOOTPRINTS (walls) show ONLY for a tracked quest — no "show all when nothing tracked"
        // fallback, so the map stays clean until you track something (user: don't display quest
        // zones unless they're from a tracked quest). Point MARKERS keep the fallback: with nothing
        // tracked, all quest markers show so you can discover quests; tracking then focuses to them.
        let show = toggles.quests
            && if is_zone.is_some() { tracked } else { tracker.active.is_empty() || tracked };
        *vis = if show { Visibility::Visible } else { Visibility::Hidden };
    }
}

/// Draw the TYPED zone footprints from gamedata.json as closed polygons (immediate mode, the
/// same idiom as `draw_quest_outlines`): exfil collider footprints in their faction colour with
/// the Extracts toggle, minefields in hazard red, sniper zones in orange with their own
/// toggles. GROUND-hugging outlines (exfils / transits / quest / trader zones) are terrain-DRAPED
/// at extraction (subdivided ~4 m + lifted to terrain+0.3) and get only a token lift here; the
/// elevated collider zones (minefields / sniper zones / directional mines) instead sit at their
/// collider CENTER height — exactly where their marker sphere is — so the ring/wall/marker all read
/// at the platform, not draped to the ground far below (DEFECT 2 fix). The `draped` flag stays true
/// (the ground group is still draped), so the token 0.05 lift is right for both. Zones inactive in
/// the scene follow the panel's "hide inactive" filter like their markers.
fn draw_gamedata_outlines(mut gizmos: Gizmos, zones: Res<GameDataZones>, toggles: Res<LayerToggles>) {
    let lift = Vec3::new(0.0, if zones.draped { 0.05 } else { 0.4 }, 0.0);
    let hide_inactive = toggles.hide_inactive;
    let mut ring = |outline: &Vec<Vec3>, color: Color| {
        gizmos.linestrip(
            outline
                .iter()
                .map(|p| *p + lift)
                .chain(outline.first().map(|p| *p + lift)),
            color,
        );
    };
    if toggles.extracts {
        for (fac, outline, active) in &zones.exfils {
            if hide_inactive && !active {
                continue;
            }
            ring(outline, extract_faction_color(fac));
        }
    }
    // Toggle-driven baseline groups (crisp line under each zone's translucent wall).
    for (on, list, color) in [
        (toggles.minefields, &zones.minefields, poi_look(PoiLayer::Minefield).0),
        (toggles.sniper_zones, &zones.sniper_zones, poi_look(PoiLayer::SniperZone).0),
        (toggles.hazards, &zones.hazard_zones, poi_look(PoiLayer::Hazard).0),
        (toggles.transits, &zones.transits, poi_look(PoiLayer::Transit).0),
    ] {
        if !on {
            continue;
        }
        for (outline, active) in list {
            if hide_inactive && !active {
                continue;
            }
            ring(outline, color);
        }
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
