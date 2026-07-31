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

use crate::inspect::{money, titlecase, MarkerInfo, PickRadius};
use crate::poi::MarkerValue;
use crate::render::LoadedPack;
use bevy::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct LootPlugin;
impl Plugin for LootPlugin {
    fn build(&self, app: &mut App) {
        // Rebuild the loot overlay on each MapEpoch (initial epoch-0 insert included), despawning
        // the old map's markers first. Despawn is UNCONDITIONAL (chained before spawn_loot, which
        // has early-returns): a new pack may have no loot.json, so its markers must clear regardless.
        // Also re-run when the container->model match lands (LootModelIndex arrives AFTER the async
        // geometry build), so markers upgrade from box to model-glow without a map swap.
        app.init_resource::<LootGlowState>().add_systems(
            Update,
            (teardown_loot, spawn_loot)
                .chain()
                .run_if(loot_needs_rebuild),
        );
        // Mirror the overlay's effective per-marker visibility into the glow lane AFTER the panel
        // logic ran — every rule (master toggle, class filters, min-value, dense clustering)
        // transfers to the model glow without a second implementation.
        app.add_systems(
            Update,
            update_loot_glow.after(crate::ui::apply_loot_visibility),
        );
    }
}

fn loot_needs_rebuild(
    epoch: Res<crate::render::MapEpoch>,
    pack: Option<Res<LoadedPack>>,
    index: Option<Res<LootModelIndex>>,
) -> bool {
    epoch.is_changed()
        || pack.is_some_and(|p| p.is_added())
        || index.is_some_and(|i| i.is_changed())
}

/// Container -> GPU-instance match, built by the geometry blob build (`match_loot_models` in
/// gpu_driven.rs, prefab-ancestry join) and inserted as a slim persistent copy — the blob itself
/// is dropped after upload. Entries: (gamedata container index, model-center world pos, GPU
/// instances of every part + LOD shell).
#[derive(Resource, Default)]
pub struct LootModelIndex {
    pub models: Vec<(u32, [f32; 3], Vec<u32>)>,
}

/// The GPU instances a loot marker's MODEL occupies (all parts + LOD shells). Present only on
/// markers that matched a scene model — those spawn without the cuboid and glow instead.
#[derive(Component)]
pub struct GlowInstances(pub Vec<u32>);

/// The marker's class colour packed for the glow lane (bits 0..23 = RGB8).
#[derive(Component)]
pub struct GlowColor(pub u32);

/// Cross-world glow state: (gpu instance, packed colour+phase+enable) for every VISIBLE matched
/// marker. `gen` bumps only on real change, so the render world rewrites its lane at user rate.
#[derive(Resource, Default, Clone)]
pub struct LootGlowState {
    pub entries: Vec<(u32, u32)>,
    pub gen: u64,
}

impl bevy::render::extract_resource::ExtractResource for LootGlowState {
    type Source = LootGlowState;
    fn extract_resource(s: &Self) -> Self {
        s.clone()
    }
}

/// Compose the glow lane from the markers the panel decided to SHOW. Runs after
/// `apply_loot_visibility`, so its Visibility verdict is this frame's truth. Gated on actual
/// Visibility flips (that system writes only on a real change) + marker spawns, so the compose
/// runs at user rate, not per frame.
pub(crate) fn update_loot_glow(
    mut state: ResMut<LootGlowState>,
    changed: Query<
        (),
        (
            With<LootMarker>,
            Or<(Changed<Visibility>, Added<GlowInstances>)>,
        ),
    >,
    q: Query<(&GlowInstances, &GlowColor, &Visibility), With<LootMarker>>,
) {
    if changed.is_empty() {
        return;
    }
    let mut entries: Vec<(u32, u32)> = Vec::new();
    for (gi, col, vis) in &q {
        if *vis == Visibility::Hidden {
            continue;
        }
        for &idx in &gi.0 {
            // col.0 already carries the phase nibble (per CONTAINER, so body+lid pulse together).
            entries.push((idx, 0x8000_0000 | col.0));
        }
    }
    entries.sort_unstable();
    entries.dedup();
    if entries != state.entries {
        state.entries = entries;
        state.gen = state.gen.wrapping_add(1);
    }
}

/// In-place map swap: despawn every loot marker so `spawn_loot` rebuilds for the new pack (freeing
/// the per-class materials + cube mesh once the last handle drops).
fn teardown_loot(mut commands: Commands, q: Query<Entity, With<LootMarker>>) {
    for e in &q {
        commands.entity(e).despawn();
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
    /// Human-readable container type (JSON key `type`), e.g. "Weapon box (5x2)".
    #[serde(default, rename = "type")]
    type_: String,
    /// Expected ruble value of the container's loot.
    #[serde(default)]
    ev: i64,
    /// Spawn probability 0..1.
    #[serde(default)]
    spawn: f32,
    /// Estimated seconds spent opening/searching this container.
    #[serde(default)]
    t: Option<f32>,
}

/// Estimated seconds spent at a loot stop. Shared with the raid-time planner.
#[derive(Component, Clone, Copy)]
pub struct LootTime(pub f32);

/// Probability this container is actually PRESENT in a given raid (0..1).
///
/// Kept separate from [`crate::poi::MarkerValue`] deliberately. MarkerValue stays the raw worth of
/// the contents, so the panel's "min value" filter keeps meaning "worth this much IF it is there";
/// folding the odds into it would silently hide most of the map behind the existing 100k default.
/// The planner multiplies the two to rank by EXPECTED value, which is the number that should decide
/// where a run goes.
///
/// Preferred source is the game's own `LootableContainersGroup` odds (`grp_p` in gamedata.json:
/// how many of a group's containers spawn, over its member count - 19% for the mall stashes, 83%
/// at Kiba Arms). Falls back to loot.json's per-TYPE average fill rate, which is location-blind.
#[derive(Component, Clone, Copy)]
pub struct SpawnChance(pub f32);

/// One LootableContainer as the GAME ships it (gamedata.json — the authoritative overlay
/// driver). `idx` is its position in the file's containers array: the same index
/// `LootModelIndex` keys its ancestry-matched model instances on.
struct GdContainer {
    idx: u32,
    pos: Vec3,
    /// The game's own container template name ("Weapon box", "Drawer", ...).
    tpl_name: String,
    /// The game's per-area spawn odds (LootableContainersGroup), when grouped.
    grp_p: Option<f32>,
}

/// Every ACTIVE LootableContainer from the pack's gamedata.json — the authoritative set the
/// overlay spawns from. tarkov.dev's loot.json only ENRICHES these (prices/classes); it can no
/// longer add or subtract a marker (its stale entries were both missing real containers and
/// placing ghosts).
fn load_gamedata_containers(pack_root: Option<&std::path::Path>) -> Vec<GdContainer> {
    let Some(root) = pack_root else { return Vec::new() };
    let Ok(txt) = std::fs::read_to_string(root.join("gamedata.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (idx, c) in v
        .get("containers")
        .and_then(|c| c.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .enumerate()
    {
        if !c.get("active").and_then(|a| a.as_bool()).unwrap_or(true) {
            continue;
        }
        let Some(pos) = c.get("pos").and_then(|x| x.as_array()).filter(|p| p.len() >= 3) else {
            continue;
        };
        out.push(GdContainer {
            idx: idx as u32,
            pos: Vec3::new(
                pos[0].as_f64().unwrap_or(0.0) as f32,
                pos[1].as_f64().unwrap_or(0.0) as f32,
                pos[2].as_f64().unwrap_or(0.0) as f32,
            ),
            tpl_name: c
                .get("tpl_name")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            grp_p: c.get("grp_p").and_then(|x| x.as_f64()).map(|p| p as f32),
        });
    }
    out
}

/// Container class -> (base color, half-extents in metres). Weapon boxes are dark
/// (the "black weapon crate") but never pure black, and every class gets an emissive
/// term so it's visible in shadow. Sizes are rough per-type so markers read as boxes.
fn class_look(cls: &str) -> (Color, Vec3) {
    // Colours MATCH the panel's swatch legend (ui.rs `class_color`) so the on-map markers and
    // the panel are one consistent key.
    match cls {
        "weapon" => (Color::srgb(0.839, 0.361, 0.282), Vec3::new(0.60, 0.28, 0.42)),
        "medical" => (Color::srgb(0.361, 0.784, 0.478), Vec3::new(0.35, 0.30, 0.30)),
        "safe" => (Color::srgb(0.922, 0.745, 0.290), Vec3::new(0.32, 0.45, 0.28)),
        "register" => (Color::srgb(0.329, 0.635, 0.922), Vec3::new(0.35, 0.28, 0.28)),
        "bag" => (Color::srgb(0.804, 0.588, 0.361), Vec3::new(0.30, 0.24, 0.30)),
        "crate" => (Color::srgb(0.769, 0.635, 0.424), Vec3::new(0.45, 0.35, 0.45)),
        "tech" => (Color::srgb(0.690, 0.439, 0.886), Vec3::new(0.35, 0.30, 0.30)),
        "stash" => (Color::srgb(0.588, 0.588, 0.588), Vec3::new(0.35, 0.20, 0.35)),
        "furniture" => (Color::srgb(0.635, 0.541, 0.455), Vec3::new(0.35, 0.30, 0.30)),
        "body" => (Color::srgb(0.871, 0.290, 0.290), Vec3::new(0.35, 0.30, 0.60)),
        _ => (Color::srgb(0.85, 0.85, 0.85), Vec3::new(0.30, 0.28, 0.30)),
    }
}

/// Resolve `loot.json` WITHOUT a hardcoded absolute path. Order:
///   1. `EFT_LOOT_JSON` env override (explicit path),
///   2. `<pack-dir>/loot.json` — co-located with the pack, so the pack is a
///      self-contained bundle you can hand to a friend,
///   3. `./loot.json` (cwd).
pub(crate) fn resolve_loot_json(pack_root: Option<&std::path::Path>) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("EFT_LOOT_JSON") {
        let pb = PathBuf::from(&p);
        if pb.is_file() {
            return Some(pb);
        }
        warn!("loot: EFT_LOOT_JSON='{p}' is not a file — ignoring");
    }
    if let Some(root) = pack_root {
        let pb = root.join("loot.json");
        if pb.is_file() {
            return Some(pb);
        }
        // Shared tier: tarkov.dev data is map-agnostic (all-maps files) — it lives ABOVE the
        // packs in packs/shared/ so it isn't duplicated per map. Pack-local still wins (override).
        if let Some(shared) = root.parent().map(|p| p.join("shared").join("loot.json")) {
            if shared.is_file() {
                return Some(shared);
            }
        }
    }
    let shared = crate::paths::shared_dir().join("loot.json");
    if shared.is_file() {
        return Some(shared);
    }
    let cwd = PathBuf::from("loot.json");
    if cwd.is_file() {
        return Some(cwd);
    }
    None
}

pub(crate) fn spawn_loot(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pack: Option<Res<LoadedPack>>,
    model_index: Option<Res<LootModelIndex>>,
) {
    // AUTHORITATIVE set: the game's own LootableContainers from the pack's gamedata. tarkov.dev
    // (loot.json) is loaded below as optional ENRICHMENT only — it prices and classifies, it
    // never adds or removes a marker (its stale entries both missed real containers and placed
    // ghosts, e.g. the streets weapon-box stack).
    let gd = load_gamedata_containers(pack.as_ref().map(|lp| lp.0.root.as_path()));
    if gd.is_empty() {
        warn!("loot: pack has no gamedata containers — no loot overlay");
        return;
    }
    let lf: Option<LootFile> = resolve_loot_json(pack.as_ref().map(|lp| lp.0.root.as_path()))
        .and_then(|path| match std::fs::read_to_string(&path) {
            Ok(t) => match serde_json::from_str::<LootFile>(&t) {
                Ok(v) => Some(v),
                Err(e) => {
                    warn!("loot: {} parse failed ({e}) — markers unpriced", path.display());
                    None
                }
            },
            Err(e) => {
                warn!("loot: {} unreadable ({e}) — markers unpriced", path.display());
                None
            }
        });

    // Enrichment map key: canonical map id first, dataset dir basename + `_vN` strip as
    // fallbacks for older packs (the pack dir is "interchange_v2", tarkov.dev keys "interchange").
    let mut keys: Vec<String> = Vec::new();
    if let Some(p) = pack.as_ref() {
        let m = &p.0.manifest.map;
        if !m.is_empty() {
            keys.push(m.clone());
        }
        let ds = &p.0.manifest.dataset;
        keys.push(ds.clone());
        if let Some((base, ver)) = ds.rsplit_once("_v") {
            if !ver.is_empty() && ver.chars().all(|c| c.is_ascii_digit()) {
                keys.push(base.to_string());
            }
        }
    }
    let ml: Option<&MapLoot> = lf.as_ref().and_then(|f| {
        keys.iter()
            .find_map(|k| f.maps.get(k))
            .or_else(|| (f.maps.len() == 1).then(|| f.maps.values().next()).flatten())
    });

    // Per-TYPE stats from the enrichment set: a container whose position has no tarkov.dev twin
    // (jittered or missing entry) still gets its type's class/value/search figures by joining on
    // the game's OWN template name ("Weapon box" matches "Weapon box (5x2)").
    struct TypeAgg {
        cls: String,
        evs: Vec<i64>,
        spawn: (f32, u32),
        t: (f32, u32),
    }
    let mut by_type: HashMap<String, TypeAgg> = HashMap::new();
    if let Some(ml) = ml {
        for c in &ml.containers {
            let key = c.type_.to_ascii_lowercase();
            let a = by_type.entry(key).or_insert_with(|| TypeAgg {
                cls: c.cls.clone(),
                evs: Vec::new(),
                spawn: (0.0, 0),
                t: (0.0, 0),
            });
            if c.ev > 0 {
                a.evs.push(c.ev);
            }
            if c.spawn > 0.0 {
                a.spawn.0 += c.spawn;
                a.spawn.1 += 1;
            }
            if let Some(t) = c.t {
                a.t.0 += t;
                a.t.1 += 1;
            }
        }
        for a in by_type.values_mut() {
            a.evs.sort_unstable();
        }
    }

    // Model matches keyed by gamedata container index (the authoritative ancestry join).
    let models: HashMap<u32, (&[f32; 3], &Vec<u32>)> = model_index
        .as_ref()
        .map(|ix| {
            ix.models
                .iter()
                .map(|(ci, p, ids)| (*ci, (p, ids)))
                .collect()
        })
        .unwrap_or_default();

    let unit_cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let mut mats: HashMap<String, Handle<StandardMaterial>> = HashMap::new();
    let mut claimed = ml.map(|m| vec![false; m.containers.len()]).unwrap_or_default();
    let (mut n_grouped, mut n_model, mut n_priced) = (0usize, 0usize, 0usize);
    for gc in &gd {
        // Positional twin ≤ 2 m for THIS container's pivot (both sources record the component
        // pivot, so they agree even when the visible model sits meters away).
        let twin = ml.and_then(|m| {
            m.containers
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    (i, Vec3::new(c.pos[0], c.pos[1], c.pos[2]).distance_squared(gc.pos), c)
                })
                .filter(|(_, d, _)| *d <= 4.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
        });
        let (cls, title, ev, spawn_avg, t_est) = match twin {
            Some((i, _, c)) => {
                if let Some(slot) = claimed.get_mut(i) {
                    *slot = true;
                }
                n_priced += 1;
                let title = if c.type_.is_empty() {
                    titlecase(&c.cls)
                } else {
                    c.type_.clone()
                };
                (c.cls.clone(), title, c.ev, c.spawn, c.t)
            }
            None => match by_type
                .iter()
                .find(|(ty, _)| ty.starts_with(&gc.tpl_name.to_ascii_lowercase()) && !gc.tpl_name.is_empty())
                .map(|(_, a)| a)
            {
                Some(a) => {
                    n_priced += 1;
                    let ev = a.evs.get(a.evs.len() / 2).copied().unwrap_or(0);
                    let spawn = if a.spawn.1 > 0 { a.spawn.0 / a.spawn.1 as f32 } else { 0.0 };
                    let t = (a.t.1 > 0).then(|| a.t.0 / a.t.1 as f32);
                    (a.cls.clone(), gc.tpl_name.clone(), ev, spawn, t)
                }
                // Unpriced: the game says it's lootable; show it honestly with type only.
                None => ("crate".to_string(), gc.tpl_name.clone(), 0, 0.0, None),
            },
        };
        let (color, half) = class_look(&cls);
        // The game's own per-area odds beat every estimate; type average is the fallback.
        let spawn_p = gc.grp_p.unwrap_or(if spawn_avg > 0.0 { spawn_avg } else { 1.0 }).clamp(0.0, 1.0);
        if gc.grp_p.is_some() {
            n_grouped += 1;
        }
        let mut detail = Vec::new();
        if ev > 0 {
            detail.push(format!("Value  {}", money(ev)));
            // Expected value is what actually decides a route; show it next to the raw worth.
            detail.push(format!("Expected  {}", money((ev as f32 * spawn_p) as i64)));
        }
        if spawn_p > 0.0 {
            detail.push(format!(
                "Spawn {:.0}%{}",
                spawn_p * 100.0,
                if gc.grp_p.is_some() { " (this area)" } else { "" }
            ));
        }
        let search_s = t_est.unwrap_or(7.0).max(0.0);
        detail.push(format!("Search ~{search_s:.0}s"));
        // Marker anchor: the MODEL's center when matched (the container pivot can sit far from
        // the visible prop in DesignStuff scenes), else the container pivot.
        let glow = models.get(&gc.idx);
        let anchor = glow
            .map(|(p, _)| Vec3::from(**p))
            .unwrap_or(gc.pos + Vec3::Y * half.y);
        if glow.is_some() {
            n_model += 1;
        }
        let pick_r = ((half * 2.0).length() * 0.5).max(0.9);
        let mut e = commands.spawn((
            Transform::from_translation(anchor).with_scale(half * 2.0),
            Visibility::default(),
            LootMarker,
            LootClass(cls.clone()),
            // The ev estimate feeds the panel's min-value filter (0 = no estimate, hides under
            // an active filter).
            MarkerValue(ev),
            SpawnChance(spawn_p),
            LootTime(search_s),
            crate::poi::DenseMarker,
            PickRadius(pick_r),
            MarkerInfo {
                title,
                subtitle: format!("Loot \u{00B7} {cls}"),
                detail,
                accent: color,
            },
        ));
        match glow {
            Some((_, ids)) => {
                // Phase rides the CONTAINER index so every part of one prop (body + lid + LOD
                // shells) breathes in unison while neighbours stay out of step.
                let l = color.to_linear();
                let packed = ((gc.idx % 16) << 24)
                    | ((l.red * 255.0) as u32) << 16
                    | ((l.green * 255.0) as u32) << 8
                    | (l.blue * 255.0) as u32;
                e.insert((GlowInstances((*ids).clone()), GlowColor(packed)));
            }
            None => {
                // No ancestry-matched model (pre-capture pack, or a model-less stash): honest box.
                let mat = mats
                    .entry(cls.clone())
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
                e.insert((Mesh3d(unit_cube.clone()), MeshMaterial3d(mat)));
            }
        }
    }
    let orphans = claimed.iter().filter(|c| !**c).count();
    info!(
        "loot: {} markers from the pack's OWN containers; {} glow their model, {} boxed; \
         {} priced ({} via positional twin), {} area-odds; {} stale tarkov.dev entries dropped{}",
        gd.len(),
        n_model,
        gd.len().saturating_sub(n_model),
        n_priced,
        claimed.iter().filter(|c| **c).count(),
        n_grouped,
        orphans,
        if model_index.is_none() {
            " — model index not built yet; markers respawn when it lands"
        } else {
            ""
        },
    );
}

#[derive(Component)]
pub(crate) struct LootMarker;

/// The loot class of a marker ("weapon"/"medical"/…), so the layer panel can filter by class.
#[derive(Component)]
pub struct LootClass(pub String);
