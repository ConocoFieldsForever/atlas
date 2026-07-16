#!/usr/bin/env python
"""Cache tarkov.dev item ICONS for one map's cards -> packs/<map>.eftpack/icons/<slug>.png.

The client ships NO inventory-icon sprites: item bundles are 3D mesh/texture assets and the
game RENDERS grid icons at runtime into a local cache keyed by opaque numeric hashes
(%TEMP%/Battlestate Games/EscapeFromTarkov/Icon Cache/live/<n>.png — no usable template-id
index). So icons come from the tarkov.dev asset CDN at BUILD time and the viewer stays
offline: only items actually referenced by THIS map's cards are fetched — gamedata.json
loose_points item pools + keyed doors (key_id template ids) and loot.json lock keys + loose
"jackpot" items. webp -> PNG via PIL, capped at 128 px. Existing files are kept (re-run is
incremental); a fetch failure skips that icon (the viewer renders the card without it).

  python extraction/intel/fetch_icons.py <map> [--pack DIR]

The <slug>.png name contract (shared with viewer/src/inspect.rs `icon_slug`): lowercase the
item's display name; ASCII alphanumerics pass through; every other run of chars becomes one
'-'; leading/trailing '-' stripped.
"""
import os, sys, json, time, functools, urllib.request
from io import BytesIO

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
API = "https://api.tarkov.dev/graphql"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAP = args[0] if args else "lighthouse"
PACK = os.path.join(REPO, "packs", f"{MAP}.eftpack")
for a in sys.argv[1:]:
    if a.startswith("--pack="):
        PACK = a.split("=", 1)[1]


def slug(s):
    out, dash = [], False
    for ch in s.lower():
        if ch.isascii() and ch.isalnum():
            out.append(ch)
            dash = False
        elif not dash:
            out.append("-")
            dash = True
    return "".join(out).strip("-")


def gql(q, tries=3):
    req = urllib.request.Request(API, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "eft-native-viewer-icons/1.0"})
    last = None
    for i in range(tries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            if "errors" in r:
                raise RuntimeError(json.dumps(r["errors"][:2])[:300])
            return r["data"]
        except Exception as ex:
            last = ex
            time.sleep(1.5 * (i + 1))
    raise SystemExit(f"[icons] tarkov.dev unreachable: {last}")


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def main():
    if not os.path.isdir(PACK):
        raise SystemExit(f"[icons] no pack dir {PACK}")
    names, ids = set(), set()

    gd = jload(os.path.join(PACK, "gamedata.json"))
    if gd:
        for r in gd.get("loose_points") or []:
            for it in r.get("items") or []:
                if it.get("cat"):
                    continue                      # category pool slots have no icon
                if it.get("n"):
                    names.add(it["n"])
                elif it.get("tpl"):
                    ids.add(it["tpl"])            # unresolved pool entry — resolve by id
        for d in gd.get("doors") or []:
            k = d.get("key_id")
            if k:
                ids.add(k)                        # key template id -> key item icon

    lj = jload(os.path.join(PACK, "loot.json"))
    mkey = MAP
    if lj:
        maps = lj.get("maps") or {}
        mm = maps.get(mkey) or (list(maps.values())[0] if len(maps) == 1 else None) or {}
        for lk in mm.get("locks") or []:
            for k in lk.get("keys") or []:
                if k.get("n"):
                    names.add(k["n"])
        for lo in mm.get("loose") or []:
            if lo.get("n"):
                names.add(lo["n"])

    if not names and not ids:
        print(f"[icons] {MAP}: no items referenced — nothing to do")
        return
    print(f"[icons] {MAP}: {len(names)} item names + {len(ids)} template ids referenced")

    items = {}
    Q = "{ items(%s: [%s]) { id name iconLink gridImageLink } }"
    if names:
        lst = ",".join(json.dumps(n) for n in sorted(names))
        for it in gql(Q % ("names", lst)).get("items") or []:
            # items(names:) can match loosely — keep ONLY exact referenced names so the pack
            # never carries icons no card shows (our names round-trip from tarkov.dev data).
            if it["name"] in names:
                items[it["name"]] = it
    if ids:
        lst = ",".join(json.dumps(i) for i in sorted(ids))
        for it in gql(Q % ("ids", lst)).get("items") or []:
            items.setdefault(it["name"], it)
    from PIL import Image
    out_dir = os.path.join(PACK, "icons")
    os.makedirs(out_dir, exist_ok=True)
    n_new = n_have = n_fail = 0
    for name, it in sorted(items.items()):
        sl = slug(name)
        if not sl:
            continue
        dst = os.path.join(out_dir, sl + ".png")
        if os.path.exists(dst):
            n_have += 1
            continue
        url = it.get("iconLink") or it.get("gridImageLink")
        if not url:
            n_fail += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eft-native-viewer-icons/1.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            img = Image.open(BytesIO(raw)).convert("RGBA")
            if max(img.size) > 128:
                img.thumbnail((128, 128), Image.LANCZOS)
            img.save(dst, "PNG")
            n_new += 1
        except Exception as ex:
            print(f"[icons]   FAIL {name}: {type(ex).__name__}: {ex}")
            n_fail += 1
    n_missing = len(names - set(items.keys()))
    print(f"[icons] {MAP}: {len(items)} resolved, {n_new} fetched, {n_have} cached, "
          f"{n_fail} failed, {n_missing} names unresolved -> {out_dir}")


if __name__ == "__main__":
    main()
