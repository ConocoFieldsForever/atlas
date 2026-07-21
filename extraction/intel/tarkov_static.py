#!/usr/bin/env python
"""Static-dump FALLBACK for the tarkov.dev intel sync.

tarkov.dev serves two independent things:
  - api.tarkov.dev/graphql : the live GraphQL API (dynamic; goes 503 when their backend/DB is down).
  - json.tarkov.dev/regular/* : pre-generated static JSON snapshots on a CDN (what tarkov.dev's own
    site loads). These stay UP during API outages because there is no live backend behind them.

build_loot.py / build_tasks.py prefer the GraphQL API (freshest, fully name-resolved). When it is
unreachable this module rebuilds the SAME data from the static dumps so a tarkov.dev API incident
never bricks the sync (and, downstream, never greys out the menu's PLAY button, which gates on
loot.json existing). It returns objects shaped EXACTLY like the GraphQL query responses, so the
builders' main() logic is unchanged.

The static schema differs from GraphQL in three ways this adapter bridges:
  1. wrapping: each file is { "data": {...}, "translations": [...] }.
  2. id-references: nested objects are ID strings (a container/mob/item/map id) that resolve via
     bundled tables (data.lootContainers, data.mobs, the items dump, data.maps) instead of being
     inlined.
  3. placeholder names: data.* name fields are the literal key "<id> Name"; the real localized
     string lives in the SEPARATE `<name>_en` dump, a flat { "<id> Name": "English" } dict. So we
     fetch maps_en / items_en purely as name lookups. Numeric fields (prices, chances) are real.
"""
import json, time, urllib.request, urllib.error

BASE = "https://json.tarkov.dev/regular/"
UA = "tarkmap-static/1.0"


def _get(name, tries=4):
    """GET json.tarkov.dev/regular/<name> with the same back-off discipline as the gql() path."""
    url = BASE + name
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
    last = None
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.URLError as e:
            last = e
            print(f"  [static] {name}: {getattr(e, 'code', e)} - retry {i + 1}/{tries}", flush=True)
            time.sleep(2 * (i + 1))
    raise SystemExit(f"json.tarkov.dev/{name} unreachable after {tries} tries: {last}")


def _list(x):
    """The dumps key collections either by id (dict) or as a plain list - normalise to a list."""
    return x if isinstance(x, list) else list(x.values())


def load_static_maps():
    """Rebuild the build_loot.py QUERY response ({'maps': [...]}) from the static dumps.

    Includes lootLoose INLINE per map (the base maps dump already carries it), so the caller does not
    need the per-map LOOSE_QUERY round-trips in fallback mode.
    """
    print("  [static] fetching maps + items catalogs from json.tarkov.dev ...", flush=True)
    D = _get("maps")["data"]
    I = _get("items")["data"]
    men = _get("maps_en").get("data", {})   # { "<id> Name": "English", "bossTagilla": "Tagilla", ... }
    ien = _get("items_en").get("data", {})

    # ---- resolver tables ------------------------------------------------------------------------
    cont = {c["id"]: (men.get(c.get("name")) or c.get("normalizedName"), c.get("normalizedName"))
            for c in _list(D["lootContainers"])}                       # container id -> (display, norm)
    mob = {m["id"]: (men.get(m.get("name")) or m.get("normalizedName"), m.get("normalizedName"))
           for m in _list(D["mobs"])}                                  # mob id -> (display, norm)
    items = _list(I["items"])
    item_by = {it["id"]: it for it in items}
    # itemCategory display names are placeholders too, but normalizedName is real ('keycard').
    keycard_ids = {c["id"] for c in _list(I.get("itemCategories") or [])
                   if str(c.get("normalizedName") or "").lower() == "keycard"}
    map_nn = {m["id"]: m.get("normalizedName") for m in _list(D["maps"])}  # map id -> normalizedName

    def item_obj(iid):
        """Resolve an item ID to the GraphQL item shape the builders read (prices are real; names via _en)."""
        if not iid:
            return None
        it = item_by.get(iid)
        if not it:
            return None
        cats = it.get("categories") or []
        return {
            "name": ien.get(it.get("name")) or it.get("normalizedName"),
            "shortName": ien.get(it.get("shortName")),
            "avg24hPrice": it.get("avg24hPrice"), "low24hPrice": it.get("low24hPrice"),
            "high24hPrice": it.get("high24hPrice"), "changeLast48hPercent": it.get("changeLast48hPercent"),
            # main() only asks whether category.name == 'keycard'; membership test covers that.
            "category": {"name": "Keycard" if any(c in keycard_ids for c in cats) else ""},
            "sellFor": None,   # the static items dump carries no per-vendor sellFor (loose 'sell' degrades to None)
        }

    out = []
    for m in _list(D["maps"]):
        # Bosses live in a dedicated `bosses` field (mob id + spawnLocations WITH positions); they are
        # NOT tagged into `spawns` the way GraphQL exposes them, and main() places bosses by matching a
        # boss-category spawn's zoneName. So synthesise boss-category spawns from spawnLocations.positions
        # -> main()'s existing zone-match placement then works unchanged.
        spawns = list(m.get("spawns") or [])
        gbosses = []
        for b in (m.get("bosses") or []):
            dn, nn = mob.get(b.get("mob"), (None, None))
            for sl in (b.get("spawnLocations") or []):
                for p in (sl.get("positions") or []):
                    spawns.append({"zoneName": sl.get("name"), "categories": ["boss"], "sides": [], "position": p})
            gbosses.append({
                "boss": {"name": dn, "normalizedName": nn},
                "spawnChance": b.get("spawnChance"), "spawnTime": b.get("spawnTime"),
                "spawnTrigger": b.get("spawnTrigger"),
                "spawnLocations": [{"name": sl.get("name"), "chance": sl.get("chance")}
                                   for sl in (b.get("spawnLocations") or [])],
                "escorts": [{"boss": {"normalizedName": mob.get(e.get("mob"), (None, None))[1]},
                             "amount": e.get("amount") or []} for e in (b.get("escorts") or [])],
            })

        def switch_obj(s):
            av = s.get("activatedBy")
            return {"id": s.get("id"), "name": s.get("name"), "switchType": s.get("switchType"),
                    "position": s.get("position"),
                    "activatedBy": ({"id": av, "name": None} if isinstance(av, str) else None),
                    "activates": [{"operation": op.get("operation"),
                                   "target": {"id": op.get("switch") or op.get("extract"), "name": None}}
                                  for op in (s.get("activates") or [])]}

        out.append({
            "normalizedName": m.get("normalizedName"),
            "name": men.get(m.get("name")) or m.get("normalizedName"),
            "wiki": m.get("wiki"), "description": men.get(m.get("description")) or m.get("description"),
            "enemies": [men.get(e, e) for e in (m.get("enemies") or [])],
            "raidDuration": m.get("raidDuration"), "players": m.get("players"),
            "minPlayerLevel": m.get("minPlayerLevel"), "maxPlayerLevel": m.get("maxPlayerLevel"),
            "spawns": spawns,
            "bosses": gbosses,
            "lootContainers": [{"lootContainer": {"name": cont.get(c.get("lootContainer"), (None, None))[0],
                                                  "normalizedName": cont.get(c.get("lootContainer"), (None, None))[1]},
                                "position": c.get("position")} for c in (m.get("lootContainers") or [])],
            "locks": [{"lockType": lk.get("lockType"), "needsPower": lk.get("needsPower"),
                       "key": item_obj(lk.get("key")), "position": lk.get("position")}
                      for lk in (m.get("locks") or [])],
            "switches": [switch_obj(s) for s in (m.get("switches") or [])],
            "transits": [{"description": men.get(t.get("description")) or t.get("description"),
                          "conditions": t.get("conditions"),
                          "map": {"normalizedName": map_nn.get(t.get("map"), t.get("map"))},
                          "position": t.get("position")} for t in (m.get("transits") or [])],
            "hazards": m.get("hazards") or [],   # {hazardType,name,position} already matches
            "stationaryWeapons": [{"stationaryWeapon": {"name": (item_obj(w.get("stationaryWeapon")) or {}).get("name")},
                                   "position": w.get("position")} for w in (m.get("stationaryWeapons") or [])],
            "extracts": [{"id": e.get("id"), "name": men.get(e.get("name")) or e.get("name"),
                          "faction": e.get("faction"), "position": e.get("position"),
                          "outline": e.get("outline") or [], "top": e.get("top"), "bottom": e.get("bottom"),
                          "switches": [], "transferItem": None}  # extract fees absent in static
                         for e in (m.get("extracts") or [])],
            "accessKeys": [],                    # not carried in the static maps dump
            "btrStops": m.get("btrStops") or [],
            "artillery": m.get("artillery") or {},
            "lootLoose": [{"position": p.get("position"),
                           "items": [io for io in (item_obj(i) for i in (p.get("items") or [])) if io]}
                          for p in (m.get("lootLoose") or [])],
        })
    print(f"  [static] rebuilt {len(out)} maps from CDN dumps", flush=True)
    return {"maps": out}


def load_static_tasks():
    """Rebuild the build_tasks.py QUERY response ({'tasks': [...]}) from the static dumps.

    The static task objectives carry every type-specific field GraphQL exposes, just FLATTENED (no
    union wrapper) and id-referenced (item / questItem / trader / task / map ids + a translation-key
    description). This resolves all of them back to the inlined GraphQL shape main() reads.
    """
    print("  [static] fetching tasks + traders + items catalogs from json.tarkov.dev ...", flush=True)
    T = _get("tasks")["data"]
    I = _get("items")["data"]
    TRD = _get("traders")["data"]
    ten = _get("tasks_en").get("data", {})        # task names, objective descriptions, quest-item names
    ien = _get("items_en").get("data", {})
    tren = _get("traders_en").get("data", {})     # trader "<id> Nickname" -> "Prapor"
    map_nn = {m["id"]: m.get("normalizedName") for m in _list(_get("maps")["data"]["maps"])}

    item_by = {it["id"]: it for it in _list(I["items"])}
    qitem = {q["id"]: (ten.get(q.get("name")) or q.get("normalizedName")) for q in _list(T.get("questItems") or [])}
    trader = {t["id"]: (tren.get(t.get("name")) or t.get("normalizedName"))
              for t in _list(TRD)}                 # traders dump is keyed by id
    task_name = {t["id"]: (ten.get(t.get("name")) or t.get("normalizedName")) for t in _list(T["tasks"])}

    def item_obj(iid):
        it = item_by.get(iid)
        if not it:
            return None
        return {"name": ien.get(it.get("name")) or it.get("normalizedName"),
                "shortName": ien.get(it.get("shortName")),
                "avg24hPrice": it.get("avg24hPrice")}

    def resolve_items(val):
        """A field that is an item-id, a list of ids, or nested lists of ids -> same nesting of item objects.
        Covers items / requiredKeys (flat or alternative-key groups) / usingWeapon / wearing / useAny."""
        out = []
        for x in (val or []):
            if isinstance(x, list):
                out.append(resolve_items(x))
            elif isinstance(x, str):
                io = item_obj(x)
                if io:
                    out.append(io)
            elif isinstance(x, dict):
                out.append(x)
        return out

    def mapref(mid):
        return {"normalizedName": map_nn.get(mid, mid)} if mid else None

    def conv_obj(o):
        oo = {"id": o.get("id"), "type": o.get("type"),
              "description": ten.get(o.get("description")) or o.get("description"),
              "optional": o.get("optional", False),
              "maps": [mapref(mid) for mid in (o.get("maps") or [])],
              "zones": [{"map": mapref(z.get("map")), "position": z.get("position"),
                         "outline": z.get("outline"), "top": z.get("top"), "bottom": z.get("bottom")}
                        for z in (o.get("zones") or [])]}
        if o.get("items"):        oo["items"] = resolve_items(o["items"])
        if o.get("markerItem"):   oo["markerItem"] = item_obj(o["markerItem"])
        if o.get("questItem"):    oo["questItem"] = {"name": qitem.get(o["questItem"])}
        if o.get("requiredKeys"): oo["requiredKeys"] = resolve_items(o["requiredKeys"])
        for k in ("usingWeapon", "usingWeaponMods", "wearing", "notWearing", "useAny"):
            if o.get(k):          oo[k] = resolve_items(o[k])
        # scalars main() reads straight through
        for k in ("targetNames", "count", "foundInRaid", "exitName", "distance", "bodyParts", "shotType",
                  "timeFromHour", "timeUntilHour", "minDurability", "maxDurability"):
            if o.get(k) is not None:
                oo[k] = o[k]
        if o.get("possibleLocations"):
            oo["possibleLocations"] = [{"map": mapref(pl.get("map")), "positions": pl.get("positions")}
                                       for pl in o["possibleLocations"]]
        return oo

    def conv_rewards(r):
        r = r or {}
        return {
            "items": [{"item": item_obj(x.get("item")), "count": x.get("count"), "quantity": x.get("quantity")}
                      for x in (r.get("items") or []) if x.get("item")],
            "traderStanding": [{"trader": {"name": trader.get(x.get("trader"))}, "standing": x.get("standing")}
                               for x in (r.get("traderStanding") or []) if x.get("trader")],
            "offerUnlock": [{"trader": {"name": trader.get(x.get("trader"))} if x.get("trader") else None,
                             "level": x.get("level"), "item": item_obj(x.get("item"))}
                            for x in (r.get("offerUnlock") or [])],
            "skillLevelReward": [{"name": ten.get(x.get("name")) or x.get("name"), "level": x.get("level")}
                                 for x in (r.get("skillLevelReward") or [])],
            # traderUnlock is a bare trader id in the static dump -> wrap as {name} so conv_rewards() reads it
            "traderUnlock": [{"name": trader.get(x)} for x in (r.get("traderUnlock") or []) if trader.get(x)],
        }

    out = []
    for t in _list(T["tasks"]):
        out.append({
            "id": t.get("id"), "name": ten.get(t.get("name")) or t.get("normalizedName"),
            "normalizedName": t.get("normalizedName"),
            "kappaRequired": t.get("kappaRequired"), "lightkeeperRequired": t.get("lightkeeperRequired"),
            "experience": t.get("experience"), "wikiLink": t.get("wikiLink"), "taskImageLink": t.get("taskImageLink"),
            "minPlayerLevel": t.get("minPlayerLevel"), "factionName": t.get("factionName"),
            "restartable": t.get("restartable"),
            "availableDelaySecondsMin": t.get("availableDelaySecondsMin"),
            "availableDelaySecondsMax": t.get("availableDelaySecondsMax"),
            "trader": {"name": trader.get(t.get("trader"))} if t.get("trader") else None,
            "map": mapref(t.get("map")),
            "taskRequirements": [{"task": {"name": task_name.get(r.get("task"))}, "status": r.get("status")}
                                 for r in (t.get("taskRequirements") or []) if r.get("task")],
            "traderRequirements": [{"trader": {"name": trader.get(r.get("trader"))},
                                    "requirementType": r.get("requirementType"),
                                    "compareMethod": r.get("compareMethod"), "value": r.get("value")}
                                   for r in (t.get("traderRequirements") or []) if r.get("trader")],
            "finishRewards": conv_rewards(t.get("finishRewards")),
            "objectives": [conv_obj(o) for o in (t.get("objectives") or [])],
        })
    print(f"  [static] rebuilt {len(out)} tasks from CDN dumps", flush=True)
    return {"tasks": out}
