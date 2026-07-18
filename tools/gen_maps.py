"""Generate the playable-map ROSTER from the game's own asset list.

The roster (which maps exist, in what order, under which id) is DERIVED from Escape from Tarkov's
`globalgamemanagers` -> BuildSettings.scenes list (index == levelN) rather than hardcoded in the
viewer. For each map we:
  * take the game's location FOLDER (Assets/Content/Locations/<Folder>/) as the authoritative group,
  * classify each scene in the folder as GEOMETRY vs a service scene (AI/Scripts/Culling/Light/...),
  * emit the id + curated display names + the derived geometry levels.

Why a curation table still exists: EFT ships NO TypeTree for its IL2CPP MonoBehaviours, so there is
no readable object that maps a location to its scene set or its community name. A handful of fields
are genuinely not derivable from the scene list and are curated here per map: the community id/dir
(labs vs folder "Laboratory"), the English + Russian display names, and the legacy-vs-rework choice
(Factory ships the 1.0 rework). Everything else -- the roster membership, ordering source, and the
geometry level indices -- comes straight from the game.

Outputs (under extraction/maps/):
  * manifest.json  -- the roster the viewer embeds: [{id, en, ru, folder, derived_levels}]
  * prints a DRIFT REPORT comparing derived geometry levels to each extraction/maps/<id>/config.json
    source.levels, so a game update that moves scenes is visible.

  python tools/gen_maps.py            # write manifest.json + print drift
  python tools/gen_maps.py --check    # drift report only, do not write

Env: EFT_GAME_DATA overrides the game data dir.
"""
import json
import os
import sys

import UnityPy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MAPS_DIR = os.path.join(REPO, "extraction", "maps")
GAME = os.environ.get(
    "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data"
)

# --- Curation table: the ONLY hand-authored per-map data (see module docstring for why). ----------
# (id  matches extraction/maps/<id>/config.json + the pack name,
#  folder  = Assets/Content/Locations/<folder>/ group in the game,
#  en / ru = curated display names). Order == menu roster order.
ROSTER = [
    ("lighthouse",     "Lighthouse",     "Lighthouse",         "Маяк"),
    ("interchange",    "Shopping_Mall",  "Interchange",        "Развязка"),
    ("factory_rework", "Factory_Rework", "Factory",            "Завод"),
    ("customs",        "Custom",         "Customs",            "Таможня"),
    ("woods",          "Woods",          "Woods",              "Лес"),
    ("shoreline",      "shorline",       "Shoreline",          "Берег"),
    ("reserve",        "Reserve_Base",   "Reserve",            "Резерв"),
    ("labs",           "Laboratory",     "The Lab",            "Лаборатория"),
    ("ground_zero",    "Sandbox",        "Ground Zero",        "Эпицентр"),
    ("streets",        "City",           "Streets of Tarkov",  "Улицы Таркова"),
    ("labyrinth",      "Labyrinth",      "The Labyrinth",      "Лабиринт"),
]

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


def is_geometry(path):
    base = os.path.basename(path).replace("\\", "/")
    base = base.rsplit(".", 1)[0]  # drop .unity
    tokens = {t.lower() for t in base.split("_")}
    return not (tokens & SERVICE_TOKENS)


def derive_levels(scenes, folder):
    """Geometry level indices (== scene index) whose location folder == folder."""
    out = []
    for i, s in enumerate(scenes):
        if folder_of(s) == folder and is_geometry(s):
            out.append(i)
    return out


def config_levels(map_id):
    p = os.path.join(MAPS_DIR, map_id, "config.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return d.get("source", {}).get("levels")


def main():
    write = "--check" not in sys.argv
    scenes = load_scenes()
    print(f"[gen_maps] BuildSettings: {len(scenes)} scenes\n")

    maps = []
    print(f"{'id':16s} {'folder':16s} {'derived':>8s} {'config':>8s}  drift")
    print("-" * 72)
    for map_id, folder, en, ru in ROSTER:
        derived = derive_levels(scenes, folder)
        cfg = config_levels(map_id)
        maps.append({"id": map_id, "en": en, "ru": ru, "folder": folder,
                     "derived_levels": derived})
        if cfg is None:
            drift = "no config.json"
        else:
            ds, cs = set(derived), set(cfg)
            extra = sorted(ds - cs)          # derived geometry the config omits
            missing = sorted(cs - ds)        # config keeps but derivation classes as service/other
            if not extra and not missing:
                drift = "exact match"
            else:
                parts = []
                if extra:
                    parts.append(f"+{extra} (derived-only)")
                if missing:
                    parts.append(f"-{missing} (config keeps: aux/service)")
                drift = "; ".join(parts)
        print(f"{map_id:16s} {folder:16s} {len(derived):8d} "
              f"{('-' if cfg is None else len(cfg)):>8}  {drift}")

    manifest = {
        "_comment": "GENERATED by tools/gen_maps.py from the game's BuildSettings scene list. "
                    "Roster of playable raid maps derived from the location folders; the viewer "
                    "embeds this instead of a hardcoded list. Regenerate after a game update: "
                    "python tools/gen_maps.py",
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
