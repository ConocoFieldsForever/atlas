#!/usr/bin/env python
"""Cache tarkov.dev item/task images for one or more built map packs.

Only items referenced by the supplied packs are resolved. All packs are handled in one process so
shared references are deduplicated and the cached items catalog is parsed once. Existing PNGs are
kept; only missing images hit the asset CDN.

  python extraction/intel/fetch_icons.py <map> [<map> ...] [--pack DIR] [--tasks-all]

The <slug>.png contract is shared with viewer/src/inspect.rs `icon_slug`: lowercase ASCII
alphanumerics pass through and every other run becomes one dash.
"""
import argparse
import functools
import json
import os
import sys
import urllib.request
from io import BytesIO

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="*", default=["lighthouse"])
    parser.add_argument("--pack", help="explicit pack dir (valid with exactly one map)")
    parser.add_argument("--tasks-all", action="store_true")
    args = parser.parse_args()
    if not args.maps:
        args.maps = ["lighthouse"]
    if args.pack and len(args.maps) != 1:
        parser.error("--pack requires exactly one map")
    return args


def slug(value):
    out, dash = [], False
    for ch in value.lower():
        if ch.isascii() and ch.isalnum():
            out.append(ch)
            dash = False
        elif not dash:
            out.append("-")
            dash = True
    return "".join(out).strip("-")


def jload(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def collect_refs(map_name, pack, tasks_all):
    names, ids, task_images = set(), set(), {}

    gd = jload(os.path.join(pack, "gamedata.json"))
    if gd:
        for ref in gd.get("loose_points") or []:
            for item in ref.get("items") or []:
                if item.get("cat"):
                    continue
                if item.get("n"):
                    names.add(item["n"])
                elif item.get("tpl"):
                    ids.add(item["tpl"])
        for door in gd.get("doors") or []:
            if door.get("key_id"):
                ids.add(door["key_id"])

    loot = jload(os.path.join(pack, "loot.json")) or jload(
        os.path.join(os.path.dirname(pack), "shared", "loot.json"))
    if loot:
        maps = loot.get("maps") or {}
        current = maps.get(map_name) or (list(maps.values())[0] if len(maps) == 1 else None) or {}
        for lock in current.get("locks") or []:
            for key in lock.get("keys") or []:
                if key.get("n"):
                    names.add(key["n"])
        for loose in current.get("loose") or []:
            if loose.get("n"):
                names.add(loose["n"])

    tasks = jload(os.path.join(pack, "tasks.json")) or jload(
        os.path.join(os.path.dirname(pack), "shared", "tasks.json"))
    n_task = 0
    if tasks:
        for task in tasks.get("tasks") or []:
            if not tasks_all and not (task.get("map") == map_name or map_name in (task.get("maps") or [])):
                continue
            if task.get("id") and task.get("image"):
                task_images[task["id"]] = task["image"]
            for objective in task.get("objectives") or []:
                for item in objective.get("items") or []:
                    names.add(item)
                    n_task += 1
                for key in ("questItem", "markerItem"):
                    if objective.get(key):
                        names.add(objective[key])
                        n_task += 1
                for key in ("weapons", "weaponMods", "wearing", "notWearing", "useAny"):
                    for item_name in objective.get(key) or []:
                        names.add(item_name)
                        n_task += 1
                for group in objective.get("requiredKeys") or []:
                    for item in group or []:
                        if item.get("n"):
                            names.add(item["n"])
                            n_task += 1
            for reward in (task.get("rewards") or {}).get("items") or []:
                if reward.get("n"):
                    names.add(reward["n"])
                    n_task += 1
            for offer in (task.get("rewards") or {}).get("offers") or []:
                if offer.get("item"):
                    names.add(offer["item"])
                    n_task += 1
    print(f"[icons] {map_name}: {n_task} task item refs; {len(names)} names, "
          f"{len(ids)} template ids, {len(task_images)} task images")
    return names, ids, task_images


def main():
    args = parse_args()
    packs = [(name, args.pack or os.path.join(REPO, "packs", f"{name}.eftpack"))
             for name in args.maps]
    for _, pack in packs:
        if not os.path.isdir(pack):
            raise SystemExit(f"[icons] no pack dir {pack}")
    roots = {os.path.normcase(os.path.abspath(os.path.dirname(pack))) for _, pack in packs}
    if len(roots) != 1:
        raise SystemExit("[icons] all pack dirs must share one parent")

    names, ids, task_images = set(), set(), {}
    for map_name, pack in packs:
        map_names, map_ids, map_task_images = collect_refs(map_name, pack, args.tasks_all)
        names.update(map_names)
        ids.update(map_ids)
        task_images.update(map_task_images)
    label = args.maps[0] if len(args.maps) == 1 else f"{len(args.maps)} maps"

    if not names and not ids and not task_images:
        print(f"[icons] {label}: no referenced images - nothing to do")
        return

    # Pillow is an extraction dependency, not part of the bare embeddable Python. Loot/task sync is
    # still required, so this optional image stage exits immediately when Pillow is unavailable.
    try:
        from PIL import Image
    except ImportError:
        print("[icons] Pillow not installed - optional icon refresh skipped (INSTALL DEPS enables it)")
        return

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import tarkov_static
    items = {}
    for item in tarkov_static.load_static_items(names, ids).get("items") or []:
        if item.get("name"):
            items[item["name"]] = item

    packs_parent = os.path.dirname(packs[0][1])
    out_dir = os.path.join(packs_parent, "shared", "icons")
    os.makedirs(out_dir, exist_ok=True)
    n_new = n_have = n_fail = 0
    for name, item in sorted(items.items()):
        item_slug = slug(name)
        if not item_slug:
            continue
        dst = os.path.join(out_dir, item_slug + ".png")
        if os.path.exists(dst):
            n_have += 1
            continue
        url = item.get("iconLink") or item.get("gridImageLink")
        if not url:
            n_fail += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eft-native-viewer-icons/1.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            image = Image.open(BytesIO(raw)).convert("RGBA")
            if max(image.size) > 128:
                image.thumbnail((128, 128), Image.LANCZOS)
            image.save(dst, "PNG")
            n_new += 1
        except Exception as ex:
            print(f"[icons]   FAIL {name}: {type(ex).__name__}: {ex}")
            n_fail += 1
    n_missing = len(names - set(items))
    print(f"[icons] {label}: {len(items)} resolved, {n_new} fetched, {n_have} cached, "
          f"{n_fail} failed, {n_missing} names unresolved -> {out_dir}")

    task_dir = os.path.join(packs_parent, "shared", "task_images")
    os.makedirs(task_dir, exist_ok=True)
    ti_new = ti_have = ti_fail = 0
    for task_id, url in sorted(task_images.items()):
        dst = os.path.join(task_dir, task_id + ".png")
        if os.path.exists(dst):
            ti_have += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eft-native-viewer-icons/1.0"})
            image = Image.open(BytesIO(urllib.request.urlopen(req, timeout=60).read())).convert("RGBA")
            if max(image.size) > 320:
                image.thumbnail((320, 320), Image.LANCZOS)
            image.save(dst, "PNG")
            ti_new += 1
        except Exception as ex:
            print(f"[icons]   FAIL task {task_id}: {type(ex).__name__}: {ex}")
            ti_fail += 1
    print(f"[icons] {label}: task images {ti_new} fetched, {ti_have} cached, "
          f"{ti_fail} failed -> {task_dir}")


if __name__ == "__main__":
    main()
