"""Generate the playable-map ROSTER from the game's own asset list + tarkov.dev names.

The roster (which maps exist, in what order, under which id) is DERIVED from Escape from Tarkov's
`globalgamemanagers` -> BuildSettings.scenes list (index == levelN) rather than hardcoded in the
viewer. For each map we:
  * take the game's location FOLDER (Assets/Content/Locations/<Folder>/) as the authoritative group,
  * classify each scene in the folder as GEOMETRY vs a service scene (AI/Scripts/Culling/Light/...),
  * derive the LIGHT levels (the `*_Light` scenes) token-aware, applying the day/night pick,
  * pull the EN + RU display names from tarkov.dev (the same API the loot/intel pipeline uses),
  * emit id + names + derived geometry levels + derived light levels.

Why an editorial table still exists (and it is now TINY): EFT ships NO TypeTree for its IL2CPP
MonoBehaviours, so there is no readable object that maps a location folder to a community id, marks
which folders are actually playable raid maps, or picks Factory's day-vs-night lighting. Those three
bits -- the folder->id alias, a `playable` flag, and the day/night bit -- are the ONLY hand-authored
data. Names come from tarkov.dev; geometry + light levels come straight from the game.

Outputs (under extraction/maps/):
  * manifest.json  -- the roster the viewer embeds: [{id, en, ru, folder, dev, light_levels,
    derived_levels}]. The viewer reads only id/en/ru; the rest is provenance for the drift report.
  * prints a DRIFT REPORT: derived geometry levels vs each extraction/maps/<id>/config.json
    source.levels, AND derived light_levels vs the legacy hardcoded LIGHT_LEVELS table, so a game
    update that moves scenes is visible.

  python tools/gen_maps.py                  # write manifest.json + print drift
  python tools/gen_maps.py --check          # drift report only, do not write
  python tools/gen_maps.py --lights-for F   # print JSON light levels for location folder F (used by
                                            #   build_map.py's manifest-miss fallback); add --night
                                            #   for the night variant of a day/night-split location

Env: EFT_GAME_DATA overrides the game data dir.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from collections import OrderedDict, namedtuple

import UnityPy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MAPS_DIR = os.path.join(REPO, "extraction", "maps")
GAME = os.environ.get(
    "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data"
)
API = "https://api.tarkov.dev/graphql"

# --- Editorial residue: the ONLY hand-authored per-map data (see module docstring for why). --------
# One row per KNOWN game location folder.
#   id       : community id (== extraction/maps/<id>/config.json + pack name); "" for excluded rows
#   folder   : Assets/Content/Locations/<folder>/ group in the game (join key to BuildSettings)
#   dev      : tarkov.dev normalizedName (join key for the EN/RU display names); "" when n/a
#   playable : emit into the shipped viewer roster (False = legacy/arena/upcoming, kept only to
#              DOCUMENT the exclusion so a newly-added game folder is flagged, not silently dropped)
#   night    : for a day/night-split location, bake the NIGHT lights instead of DAY (default DAY).
#              This is the single lighting editorial bit (Factory ships day).
Map = namedtuple("Map", "id folder dev playable night")


def _M(id, folder, dev, playable=True, night=False):
    return Map(id, folder, dev, playable, night)


ROSTER = [
    # --- playable raid maps (menu order) ---
    _M("lighthouse",     "Lighthouse",     "lighthouse"),
    _M("interchange",    "Shopping_Mall",  "interchange"),
    _M("factory_rework", "Factory_Rework", "factory"),            # ships the 1.0 rework; DAY lights
    _M("customs",        "Custom",         "customs"),
    _M("woods",          "Woods",          "woods"),
    _M("shoreline",      "shorline",       "shoreline"),
    _M("reserve",        "Reserve_Base",   "reserve"),
    _M("labs",           "Laboratory",     "the-lab"),
    _M("ground_zero",    "Sandbox",        "ground-zero"),
    _M("streets",        "City",           "streets-of-tarkov"),
    _M("labyrinth",      "Labyrinth",      "the-labyrinth"),
    # --- known NON-playable location folders (excluded from the menu ON PURPOSE) ---
    _M("", "Factory",               "", playable=False),   # legacy pre-rework Factory (rework ships)
    _M("", "Arena",                 "", playable=False),   # Arena mode, not a raid map
    _M("", "bunker",                "", playable=False),   # shared sub-scene, not a standalone map
    _M("", "Sandbox_StartLocation", "", playable=False),   # Ground Zero tutorial start, not the raid
    _M("", "Terminal",              "", playable=False),   # upcoming, not shipped
    _M("", "Venders",               "", playable=False),   # hideout/vendor scenes, not a raid map
    _M("", "Icebreaker",            "", playable=False),   # upcoming, not shipped
]

# Offline fallback display names (used ONLY when tarkov.dev is unreachable, so a no-network regen
# still produces a valid manifest instead of blank names). Online, tarkov.dev is authoritative and
# EN/RU can never drift apart (they share one id). Keyed by roster id.
FALLBACK_NAMES = {
    "lighthouse":     ("Lighthouse",        "Маяк"),
    "interchange":    ("Interchange",       "Развязка"),
    "factory_rework": ("Factory",           "Завод"),
    "customs":        ("Customs",           "Таможня"),
    "woods":          ("Woods",             "Лес"),
    "shoreline":      ("Shoreline",         "Берег"),
    "reserve":        ("Reserve",           "Резерв"),
    "labs":           ("The Lab",           "Лаборатория"),
    "ground_zero":    ("Ground Zero",       "Эпицентр"),
    "streets":        ("Streets of Tarkov", "Улицы Таркова"),
    "labyrinth":      ("The Labyrinth",     "Лабиринт"),
}

# Legacy hardcoded scalar LIGHT_LEVELS (the pre-manifest tools/build_map.py table), kept HERE only as
# the baseline the drift report diffs the derived light_levels against. Not used for generation.
_LEGACY_LIGHT_LEVELS = {
    "interchange": 64, "lighthouse": 191, "factory_rework": 526, "customs": 13, "woods": 167,
    "shoreline": 41, "reserve": 146, "labs": 114, "labyrinth": 551,
    # streets / ground_zero were omitted (multi-scene lighting was fleet-handled).
}

# A scene is a SERVICE scene (no renderable geometry) if any underscore-token of its basename is here.
# _Sound and _Design* are intentionally NOT excluded -- the hand-tuned configs keep them near-
# universally and the extractor tolerates their low/zero mesh yield (keeps derived == config closer).
SERVICE_TOKENS = {
    "ai", "scripts", "culling", "light", "quests", "portals", "stencil",
    "develop", "levelborders", "navmesh", "cutscene", "cutscenes",
}


def load_scenes():
    ggm = os.path.join(GAME, "globalgamemanagers")
    if not os.path.isfile(ggm):
        sys.exit(f"[gen_maps] globalgamemanagers not found at {ggm} (set EFT_GAME_DATA)")
    env = UnityPy.load(ggm)
    for obj in env.objects:
        if obj.type.name == "BuildSettings":
            d = obj.read_typetree()
            return d.get("scenes") or d.get("m_Scenes") or []
    sys.exit("[gen_maps] no BuildSettings.scenes in globalgamemanagers")


def folder_of(path):
    """First path component after Assets/Content/Locations/ (the location group), or None."""
    marker = "Assets/Content/Locations/"
    p = path.replace("\\", "/")
    i = p.find(marker)
    if i < 0:
        return None
    rest = p[i + len(marker):]
    return rest.split("/", 1)[0] if "/" in rest else rest.rsplit(".", 1)[0]


def _basename_tokens(path):
    base = os.path.basename(path.replace("\\", "/")).rsplit(".", 1)[0]  # drop .unity
    return [t.lower() for t in base.split("_")]


def is_geometry(path):
    return not (set(_basename_tokens(path)) & SERVICE_TOKENS)


def derive_levels(scenes, folder):
    """Geometry level indices (== scene index) whose location folder == folder."""
    out = []
    for i, s in enumerate(scenes):
        if folder_of(s) == folder and is_geometry(s):
            out.append(i)
    return out


def derive_light_scenes(scenes, folder):
    """[(level_index, [tokens])] for every `*_Light` scene in the folder. TOKEN-aware: a scene whose
    basename has a `light`/`lights` underscore-token -- NOT a substring, so the `Lighthouse` folder's
    non-light scenes (e.g. Lighthouse_Main) don't false-match. Folder match is case-insensitive so a
    config's `source.unity_location` case doesn't matter for the build-time fallback."""
    fl = folder.lower()
    out = []
    for i, s in enumerate(scenes):
        f = folder_of(s)
        if f and f.lower() == fl:
            toks = _basename_tokens(s)
            if "light" in toks or "lights" in toks:
                out.append((i, toks))
    return out


def select_light_levels(light_scenes, night=False):
    """The light level indices to BAKE for a location. If the location has a DAY/NIGHT split (both a
    `*_Day_*Light` and a `*_Night_*Light` scene exist) pick ONE variant -- DAY by default (the single
    editorial bit; Factory). Otherwise bake EVERY `*_Light` scene (streets/ground_zero split their
    lighting across many district scenes -- all are needed for full lighting)."""
    day = [i for i, toks in light_scenes if "day" in toks]
    ni = [i for i, toks in light_scenes if "night" in toks]
    if day and ni:
        return sorted(ni if night else day)
    return sorted(i for i, _ in light_scenes)


def config_levels(map_id):
    p = os.path.join(MAPS_DIR, map_id, "config.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return d.get("source", {}).get("levels")


def fetch_names():
    """{normalizedName: (en, ru)} from tarkov.dev, or None if unreachable (caller uses the offline
    fallback). RU comes from the `maps(lang: ru)` locale; EN from the default query."""
    def query(lang=None):
        arg = f"(lang: {lang})" if lang else ""
        body = json.dumps({"query": "{ maps%s { name normalizedName } }" % arg}).encode()
        req = urllib.request.Request(
            API, data=body,
            headers={"Content-Type": "application/json", "User-Agent": "gen_maps/1.0"})
        r = json.load(urllib.request.urlopen(req, timeout=30))
        if r.get("errors"):
            raise RuntimeError(f"tarkov.dev errors: {r['errors'][:2]}")
        return {m["normalizedName"]: m["name"] for m in (r.get("data", {}).get("maps") or [])}

    try:
        en = query()
        ru = query("ru")
        return {nn: (en[nn], ru.get(nn, en[nn])) for nn in en}
    except (urllib.error.URLError, OSError, RuntimeError, KeyError, ValueError) as e:
        print(f"[gen_maps] tarkov.dev unreachable ({e}) -- using offline fallback names")
        return None


def name_for(entry, api_names):
    """(en, ru) for a roster entry: tarkov.dev when reachable and the id is present, else the offline
    fallback. Guarantees EN/RU stay paired to the one id."""
    if api_names and entry.dev in api_names:
        return api_names[entry.dev]
    return FALLBACK_NAMES[entry.id]


def main():
    argv = sys.argv[1:]

    # Sub-mode used by tools/build_map.py's manifest-miss fallback: derive + print the light levels
    # for a raw location folder (e.g. a brand-new map not yet in the manifest). JSON on stdout.
    if "--lights-for" in argv:
        folder = argv[argv.index("--lights-for") + 1]
        night = "--night" in argv
        scenes = load_scenes()
        print(json.dumps(select_light_levels(derive_light_scenes(scenes, folder), night=night)))
        return

    # Sub-mode for build_map.dataset_levels: the AUTHORITATIVE geometry level list for a location
    # folder, derived live from BuildSettings (every non-service scene in the folder). Replaces the
    # hand-curated per-map config.source.levels, which drifts as the game adds scenes -- e.g. reserve
    # was missing level116 (Reserve_Base_DesignStuff: vehicles/loot/props) so those objects never
    # extracted and anything resting on them floated. JSON list on stdout.
    if "--levels-for" in argv:
        folder = argv[argv.index("--levels-for") + 1]
        print(json.dumps(derive_levels(load_scenes(), folder)))
        return

    write = "--check" not in argv
    scenes = load_scenes()
    print(f"[gen_maps] BuildSettings: {len(scenes)} scenes")

    api_names = fetch_names()
    print(f"[gen_maps] names: {'tarkov.dev' if api_names else 'OFFLINE fallback'}\n")

    # --- discovery: every BuildSettings location folder must be classified in ROSTER ---
    known = {e.folder.lower() for e in ROSTER}
    seen = OrderedDict()
    for s in scenes:
        f = folder_of(s)
        if f:
            seen.setdefault(f.lower(), f)
    unknown = [orig for low, orig in seen.items() if low not in known]
    if unknown:
        print(f"[gen_maps] WARNING: {len(unknown)} BuildSettings location folder(s) not classified "
              f"in ROSTER -- add them as playable or excluded: {unknown}\n")

    maps = []
    # --- geometry-level drift (derived vs config.json source.levels) ---
    print("GEOMETRY levels (derived vs config.json source.levels):")
    print(f"  {'id':16s} {'folder':16s} {'derived':>8s} {'config':>8s}  drift")
    print("  " + "-" * 70)
    light_rows = []
    for e in ROSTER:
        if not e.playable:
            continue
        en, ru = name_for(e, api_names)
        derived = derive_levels(scenes, e.folder)
        light = select_light_levels(derive_light_scenes(scenes, e.folder), night=e.night)
        maps.append({"id": e.id, "en": en, "ru": ru, "folder": e.folder, "dev": e.dev,
                     "light_levels": light, "derived_levels": derived})

        cfg = config_levels(e.id)
        if cfg is None:
            drift = "no config.json"
        else:
            ds, cs = set(derived), set(cfg)
            extra, missing = sorted(ds - cs), sorted(cs - ds)
            if not extra and not missing:
                drift = "exact match"
            else:
                parts = []
                if extra:
                    parts.append(f"+{extra} (derived-only)")
                if missing:
                    parts.append(f"-{missing} (config keeps: aux/service)")
                drift = "; ".join(parts)
        print(f"  {e.id:16s} {e.folder:16s} {len(derived):8d} "
              f"{('-' if cfg is None else len(cfg)):>8}  {drift}")

        # collect the light-drift row (derived light_levels vs the legacy hardcoded scalar)
        full = [i for i, _ in derive_light_scenes(scenes, e.folder)]
        light_rows.append((e, light, full))

    # --- LIGHT-level drift (derived light_levels vs the legacy hardcoded LIGHT_LEVELS table) ---
    print("\nLIGHT levels (derived vs legacy hardcoded LIGHT_LEVELS):")
    print(f"  {'id':16s} {'emitted light_levels':32s} {'legacy':>7s}  verdict")
    print("  " + "-" * 78)
    for e, light, full in light_rows:
        legacy = _LEGACY_LIGHT_LEVELS.get(e.id)
        if legacy is None:
            verdict = "now populated (was: menu skipped -> SKY-ONLY)" if light else "none"
        elif light == [legacy]:
            if len(full) > 1:
                verdict = f"DAY-pick (candidates {full}, chose {'NIGHT' if e.night else 'DAY'})"
            else:
                verdict = "exact"
        else:
            verdict = f"*** DRIFT: legacy {legacy} not the sole derived light (full={full}) ***"
        lstr = str(light if len(light) != 1 else light[0])
        print(f"  {e.id:16s} {lstr:32s} {str(legacy):>7s}  {verdict}")

    manifest = {
        "_comment": "GENERATED by tools/gen_maps.py from the game's BuildSettings scene list + "
                    "tarkov.dev names. Roster of playable raid maps derived from the location "
                    "folders; the viewer embeds this instead of a hardcoded list. Regenerate after "
                    "a game update: python tools/gen_maps.py",
        "maps": maps,
    }
    out = os.path.join(MAPS_DIR, "manifest.json")
    if write:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"\n[gen_maps] wrote {out} ({len(maps)} maps)")
    else:
        print("\n[gen_maps] --check: manifest NOT written")


if __name__ == "__main__":
    main()
