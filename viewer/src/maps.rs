//! Game-derived map roster.
//!
//! The set of playable maps, their order, ids, and EN/RU display names is NOT hardcoded in the
//! viewer. It is generated from the game's own `BuildSettings` scene list by `tools/gen_maps.py`
//! into `extraction/maps/manifest.json` (regenerate after a game update) and embedded here at
//! compile time. This replaces the old `menu::KNOWN_MAPS` table and is the single source the menu,
//! the jobs panel, and i18n's Russian map names all read from — so adding/renaming a map is a
//! data-only change and the EN/RU keys can never drift apart (they share one `id`).
use serde::Deserialize;
use std::sync::OnceLock;

/// One roster entry: the community `id` (== `extraction/maps/<id>/config.json` + pack name) and the
/// curated English + Russian display names. `bundles` holds the map's scene-preset bundle stems
/// (derived by gen_maps from the game's `maps/*.bundle` files) — the raid-detection join for
/// `bundle_to_id`. Extra manifest fields (folder, derived_levels) are provenance for the
/// generator/drift report and are ignored here.
#[derive(Deserialize)]
pub struct MapMeta {
    pub id: String,
    pub en: String,
    pub ru: String,
    /// Scene-preset bundle stems ("shopping_mall", "factory_day_preset", …). Empty on a
    /// manifest generated before the bundle join existed.
    #[serde(default)]
    pub bundles: Vec<String>,
}

#[derive(Deserialize)]
struct Manifest {
    maps: Vec<MapMeta>,
}

// Embedded at compile time so the shipped binary needs no game install / external file to know the
// roster. Path is relative to this source file (viewer/src/ -> repo/extraction/maps/).
const MANIFEST_JSON: &str = include_str!("../../extraction/maps/manifest.json");

/// The ordered roster of playable raid maps, parsed once from the embedded game-derived manifest.
/// An unparseable manifest logs and yields an empty roster (the on-disk "extra packs" scan in
/// `menu::scan` still surfaces any built packs, so the menu degrades rather than vanishing).
pub fn roster() -> &'static [MapMeta] {
    static ROSTER: OnceLock<Vec<MapMeta>> = OnceLock::new();
    ROSTER.get_or_init(|| match serde_json::from_str::<Manifest>(MANIFEST_JSON) {
        Ok(m) => m.maps,
        Err(e) => {
            bevy::log::error!("maps manifest parse failed: {e}");
            Vec::new()
        }
    })
}

/// `(id, english_title)` pairs, in roster order — the drop-in shape the old `KNOWN_MAPS` const had.
pub fn known_pairs() -> &'static [(&'static str, &'static str)] {
    static PAIRS: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    PAIRS.get_or_init(|| {
        roster()
            .iter()
            .map(|m| (m.id.as_str(), m.en.as_str()))
            .collect()
    })
}

/// Russian display name for a map id, or `None` if the id isn't in the roster.
pub fn ru_title(id: &str) -> Option<&'static str> {
    roster().iter().find(|m| m.id == id).map(|m| m.ru.as_str())
}

/// Map id for a scene-preset bundle stem (`scene preset path:maps/<stem>.bundle` in the game
/// log): "shopping_mall" -> "interchange". GAME-DERIVED (gen_maps reads the bundles' own scene
/// lists) — replaces game_watch's hardcoded TarkovMonitor table, which had silently omitted
/// the `icebreaker` bundle so Icebreaker raids were never detected. Bundles for locations we
/// don't ship (terminal, arena) simply aren't in the roster.
pub fn bundle_to_id(stem: &str) -> Option<&'static str> {
    roster()
        .iter()
        .find(|m| m.bundles.iter().any(|b| b == stem))
        .map(|m| m.id.as_str())
}
