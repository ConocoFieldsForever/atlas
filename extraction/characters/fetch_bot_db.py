#!/usr/bin/env python3
"""Bank the game's bot/customization tables into packs/shared.

Nothing here enumerates bots by hand: the roster is DISCOVERED by listing the database's own
`bots/types` directory, so a game update that adds a boss is picked up by re-running this.

Sources (BSG's data, mirrored):
  bots/types/*.json          per-bot generation tables (equipment weights, mods, chances, appearance)
  templates/customization.json   appearance id -> body/head prefab bundle
  templates/items.json       item templates (prefab path + slots)   [large, LFS]
  globals.json               ItemPresets: BSG's factory weapon builds

Config: extraction/characters/bot_db.json (created on first run) sets the base URL, which slots
count as wearable, and an optional roster filter. Everything else is derived.
"""
import argparse
import gzip
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SHARED = os.path.join(REPO, "packs", "shared")
CONFIG = os.path.join(HERE, "bot_db.json")

DEFAULT_CONFIG = {
    "_comment": "Sources + policy for the bot database bank. No bot names are hardcoded: the "
                "roster is listed from the database itself.",
    "api": "https://api.github.com/repos/sp-tarkov/server/contents/project/assets/database",
    "raw": "https://raw.githubusercontent.com/sp-tarkov/server/master/project/assets/database",
    "ref": "master",
    # Which bot types to bank. Empty = ALL discovered. Prefixes are honoured, so "boss" takes
    # every boss without naming them.
    "roster": [],
    # Equipment slots that carry geometry we can render. Extend as more slots are supported.
    "wearableSlots": ["ArmorVest", "TacticalVest", "Headwear", "Backpack", "FaceCover",
                      "Eyewear", "Earpiece"],
    # Appearance slots resolved through customization.json.
    "appearanceSlots": ["body", "head", "hands", "feet"],
}


def load_config(path=CONFIG):
    if not os.path.exists(path):
        json.dump(DEFAULT_CONFIG, open(path, "w"), indent=1)
        print(f"[cfg] wrote default {path}")
        return dict(DEFAULT_CONFIG)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(json.load(open(path, encoding="utf-8")))
    return cfg


def _get(url, timeout=240):
    req = urllib.request.Request(url, headers={"User-Agent": "atlas/1.0",
                                               "Accept-Encoding": "gzip"})
    r = urllib.request.urlopen(req, timeout=timeout)
    raw = r.read()
    if r.headers.get("Content-Encoding") == "gzip":
        raw = gzip.decompress(raw)
    return raw


def discover_roster(cfg):
    """Every bot type the database defines — listed, never hardcoded."""
    lst = json.loads(_get(f"{cfg['api']}/bots/types?ref={cfg['ref']}"))
    names = sorted(e["name"][:-5] for e in lst if e.get("name", "").endswith(".json"))
    want = cfg.get("roster") or []
    if want:
        names = [n for n in names if any(n == w or n.startswith(w) for w in want)]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--skip-items", action="store_true",
                    help="don't refresh the big item template file (LFS; use git clone for it)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(SHARED, exist_ok=True)

    roster = discover_roster(cfg)
    print(f"[bots] {len(roster)} type(s) discovered")
    bots = {}
    for i, name in enumerate(roster, 1):
        try:
            b = json.loads(_get(f"{cfg['raw']}/bots/types/{name}.json"))
        except Exception as e:
            print(f"  [warn] {name}: {str(e)[:60]}")
            continue
        # Keep only what the kit builder consumes; these files are large.
        bots[name] = {
            "inventory": b.get("inventory") or {},
            "appearance": b.get("appearance") or {},
            "chances": b.get("chances") or {},
        }
        if i % 10 == 0:
            print(f"  [bots] {i}/{len(roster)}", flush=True)
    out = os.path.join(SHARED, "bot_loadouts.json")
    json.dump(bots, open(out, "w"), separators=(",", ":"))
    print(f"[bots] {len(bots)} types -> {out} ({os.path.getsize(out)//1024} KiB)")

    for rel, dst in (("templates/customization.json", "customization.json"),
                     ("globals.json", "globals.json")):
        try:
            raw = _get(f"{cfg['raw']}/{rel}")
            open(os.path.join(SHARED, dst), "wb").write(raw)
            print(f"[db] {dst}: {len(raw)//1024} KiB")
        except Exception as e:
            print(f"  [warn] {dst}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
