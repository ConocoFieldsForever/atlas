"""tarkov.dev `normalizedName` -> this repo's pack id. ONE table, one miss rule.

There were two byte-identical copies of this table with OPPOSITE miss handling: `build_tasks`
passed an unmapped slug straight through (`DEV_TO_ID.get(nn, nn)`), `build_loot` dropped the map
(`if not mid: continue`). Neither listed `icebreaker` or `terminal`, both of which ship with a pack
id equal to their dev slug, so tasks.json gained `icebreaker: 4` while loot.json had no icebreaker
key at all. `poi.rs` does `f.maps.remove(&key)` with no fallback, so on icebreaker the entire
tarkov.dev intel block was skipped: no map-intel card, no exit configs, and 40 boss markers that the
cached dump does carry never spawned. Routing consequence: the avoid-boss field is built from live
`PoiLayer::Boss` markers, so "avoid boss" was a silent no-op on that map.

The passthrough was wrong in the other direction too, emitting keys like `the-lab-dark` that no
pack's `map_key` matches, so those zones were invisible rather than folded into the right map.
"""

DEV_TO_ID = {
    'interchange': 'interchange',
    'ground-zero': 'ground_zero', 'ground-zero-21': 'ground_zero',
    # Shipped "Factory" is the 1.0 rework (id factory_rework); tarkov.dev still names it
    # factory / night-factory, so map both or the quest/POI layers are empty there.
    'factory': 'factory_rework', 'night-factory': 'factory_rework',
    'woods': 'woods',
    'customs': 'customs',
    'shoreline': 'shoreline',
    'streets-of-tarkov': 'streets',
    'reserve': 'reserve',
    'the-lab': 'labs', 'the-labs': 'labs', 'the-lab-dark': 'labs',
    'lighthouse': 'lighthouse',
    'the-labyrinth': 'labyrinth', 'labyrinth': 'labyrinth',
    'icebreaker': 'icebreaker',
    'terminal': 'terminal',
}


def map_id(normalized_name):
    """Pack id for a tarkov.dev map, or None if we do not ship it.

    None, never a passthrough: emitting an unmapped slug as if it were a pack id produces a key
    nothing reads, which looks like data and behaves like a silent drop.
    """
    return DEV_TO_ID.get(normalized_name)
