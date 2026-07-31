#!/usr/bin/env python3
"""Bake rolled kits into `.eftkit` bundles the viewer can spawn.

One kit = one rolled bot: its appearance parts (body/head/hands/feet), its worn equipment
(armour, rig, helmet, ...), and its assembled weapon — all resolved from the GAME'S OWN tables by
`loadout.py`, then baked to geometry by `build_weapon.py`'s assembler.

ROUTING (structural, never by item name): a part is SKINNED when its renderer is a
SkinnedMeshRenderer (it binds the shared 79-bone rig, so the viewer skins it with the body) and
RIGID otherwise (a MeshFilter/MeshRenderer prefab, which hangs off a bone like the weapon does).
The test is a property of the asset, so it cannot drift as items are added.

usage:
  build_loadouts.py --bot assault --count 4
  build_loadouts.py --all --count 2        # every bot type the roster knows
"""
import argparse
import json
import os
import sys

import UnityPy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_weapon
import fetch_bot_db
import loadout as loadout_mod
import unity_deps

REPO = os.path.dirname(os.path.dirname(HERE))
OUT_ROOT = os.path.join(REPO, "out", "kits")


def classify(prefab_rel, cabs):
    """('skinned'|'rigid'|None, bone_hint) for an item prefab — decided by what the asset IS."""
    p = os.path.join(unity_deps.SA_WIN, (prefab_rel or "").replace("/", os.sep))
    if not prefab_rel or not os.path.exists(p):
        return None, None
    try:
        env = UnityPy.Environment()
        own, _ = unity_deps.resolve_into(env, p, cabs)
    except Exception:
        return None, None
    kinds = set()
    for o in own:
        if o.type.name in ("SkinnedMeshRenderer", "MeshRenderer"):
            kinds.add(o.type.name)
    if "SkinnedMeshRenderer" in kinds:
        return "skinned", None
    if "MeshRenderer" in kinds:
        return "rigid", None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", action="append", help="bot type (repeatable); default from config")
    ap.add_argument("--all", action="store_true", help="every bot type in the banked roster")
    ap.add_argument("--count", type=int, default=2, help="kits per bot type")
    ap.add_argument("--out", default=OUT_ROOT)
    args = ap.parse_args()

    cfg = fetch_bot_db.load_config()
    items, bots, presets, cust = loadout_mod.load_tables()
    cabs = unity_deps.load(verbose=False)

    types = args.bot or (sorted(bots) if args.all else ["assault"])
    os.makedirs(args.out, exist_ok=True)
    index = {}
    for bt in types:
        if bt not in bots:
            print(f"[skip] unknown bot type {bt!r}")
            continue
        for seed in range(args.count):
            kit = loadout_mod.roll(bt, seed, items, bots, presets, cust, cfg)
            kid = f"{bt}_{seed}"
            kdir = os.path.join(args.out, kid)
            os.makedirs(kdir, exist_ok=True)
            built = {"bot": bt, "seed": seed, "parts": [], "weapon": None,
                     "weaponName": kit.get("weaponName")}

            # --- weapon (its own mod tree, anchored on the game's Weapon_root) ---
            if kit.get("weapon"):
                wdir = os.path.join(build_weapon.OUT_ROOT, kit["weaponName"] or kit["weapon"])
                if not os.path.exists(os.path.join(wdir, "manifest.json")):
                    try:
                        build_weapon.build(kit["weapon"], items, wdir,
                                           install=kit.get("weaponTree") or {}, cabs=cabs)
                    except SystemExit as e:
                        print(f"  [warn] weapon {kit['weaponName']}: {e}")
                if os.path.exists(os.path.join(wdir, "manifest.json")):
                    built["weapon"] = os.path.relpath(wdir, REPO).replace(os.sep, "/")

            # --- appearance + worn equipment ---
            sources = [("appearance", s, v) for s, v in (kit.get("appearance") or {}).items()]
            sources += [("worn", s, v) for s, v in (kit.get("worn") or {}).items()]
            for group, slot, v in sources:
                rel = v.get("prefab")
                kind, _bone = classify(rel, cabs)
                if kind is None:
                    continue
                name = v.get("name") or v.get("id")
                pdir = os.path.join(args.out, "_parts", str(name))
                if not os.path.exists(os.path.join(pdir, "manifest.json")):
                    try:
                        build_weapon.build(v["id"], items, pdir, install={}, cabs=cabs)
                    except SystemExit:
                        continue
                    except Exception as e:
                        print(f"  [warn] {name}: {str(e)[:60]}")
                        continue
                if os.path.exists(os.path.join(pdir, "manifest.json")):
                    built["parts"].append({
                        "group": group, "slot": slot, "name": name, "kind": kind,
                        "dir": os.path.relpath(pdir, REPO).replace(os.sep, "/"),
                    })
            json.dump(built, open(os.path.join(kdir, "kit.json"), "w"), indent=1)
            index[kid] = {"bot": bt, "seed": seed,
                          "dir": os.path.relpath(kdir, REPO).replace(os.sep, "/")}
            print(f"[kit] {kid}: weapon={bool(built['weapon'])} parts={len(built['parts'])}")
    json.dump({"kits": index}, open(os.path.join(args.out, "index.json"), "w"), indent=1)
    print(f"[kits] {len(index)} kit(s) -> {args.out}")


if __name__ == "__main__":
    main()
