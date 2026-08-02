"""Enumerate what is actually inside the game's Unity bundles.

The extraction pipeline reads a deliberate SUBSET (renderers, colliders, lights, LODGroups, the
typed gameplay records). This walks the same bundles with no filter at all, so you can see the
whole population before deciding what is worth extracting next.

EFT bundles ship their TYPE TREES, so custom MonoBehaviours read back with real field names rather
than opaque blobs -- which is what makes `dump` useful instead of a hex view. Watch for the extra
nested "data" key on MonoBehaviour typetrees.

  python tools/explore_bundles.py types  <level>              type histogram for one level
  python tools/explore_bundles.py types  all                  histogram across EVERY level (slow)
  python tools/explore_bundles.py scripts <level>             MonoBehaviour classes + counts
  python tools/explore_bundles.py find   <level> <substr>     objects whose name matches
  python tools/explore_bundles.py dump   <level> <path_id>    one object's full typetree as JSON

<level> is a level number (e.g. 421) or a bundle filename under the game's *_Data dir.
EFT_GAME_DATA overrides the install path.
"""
import collections
import json
import os
import sys

EFTDATA = os.environ.get(
    "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")


def _bundle(level):
    p = os.path.join(EFTDATA, f"level{level}" if str(level).isdigit() else str(level))
    if not os.path.exists(p):
        raise SystemExit(f"no such bundle: {p}")
    return p


def _levels():
    for fn in sorted(os.listdir(EFTDATA)):
        if fn.startswith("level") and fn[5:].isdigit():
            yield int(fn[5:])


def _name(o):
    """Object name without reading the whole thing when we can avoid it."""
    try:
        d = o.read_typetree(check_read=False)
        return d.get("m_Name") or (d.get("data") or {}).get("m_Name") or ""
    except Exception:
        return ""


def cmd_types(arg):
    import UnityPy
    levels = list(_levels()) if arg == "all" else [arg]
    total = collections.Counter()
    for lv in levels:
        env = UnityPy.load(_bundle(lv))
        c = collections.Counter(o.type.name for o in env.objects)
        total.update(c)
        if arg != "all":
            print(f"level{lv}: {sum(c.values()):,} objects, {len(c)} types")
            for t, n in c.most_common():
                print(f"  {n:8,}  {t}")
            return
        print(f"  level{lv}: {sum(c.values()):,}", flush=True)
    print(f"\nALL LEVELS: {sum(total.values()):,} objects, {len(total)} types")
    for t, n in total.most_common():
        print(f"  {n:9,}  {t}")


def cmd_scripts(lv):
    """MonoBehaviour classes by name — where the GAMEPLAY data lives (zones, spawns, loot points)."""
    import UnityPy
    env = UnityPy.load(_bundle(lv))
    names = {o.path_id: _name(o) for o in env.objects if o.type.name == "MonoScript"}
    c = collections.Counter()
    for o in env.objects:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            d = o.read_typetree(check_read=False)
            sid = (d.get("m_Script") or {}).get("m_PathID", 0)
            c[names.get(sid) or f"<script {sid}>"] += 1
        except Exception:
            c["<unreadable>"] += 1
    print(f"level{lv}: {sum(c.values()):,} MonoBehaviours, {len(c)} distinct scripts")
    for n, k in c.most_common():
        print(f"  {k:7,}  {n}")


def cmd_find(lv, sub):
    import UnityPy
    env = UnityPy.load(_bundle(lv))
    sub = sub.lower()
    hits = 0
    for o in env.objects:
        n = _name(o)
        if n and sub in n.lower():
            print(f"  {o.type.name:22} path_id={o.path_id:<22} {n}")
            hits += 1
    print(f"{hits} match(es) for {sub!r} in level{lv}")


def cmd_dump(lv, pid):
    import UnityPy
    env = UnityPy.load(_bundle(lv))
    pid = int(pid)
    for o in env.objects:
        if o.path_id == pid:
            print(json.dumps(o.read_typetree(), indent=1, default=str)[:200000])
            return
    raise SystemExit(f"path_id {pid} not found in level{lv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"types": cmd_types, "scripts": cmd_scripts, "find": cmd_find, "dump": cmd_dump}[cmd](*rest)
