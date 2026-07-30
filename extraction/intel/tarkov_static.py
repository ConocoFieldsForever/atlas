#!/usr/bin/env python
"""Adapter for tarkov.dev's supported JSON API (https://json.tarkov.dev/endpoints).

tarkov.dev's own applications consume the pre-generated ``json.tarkov.dev/regular/*`` catalogs.
Atlas does the same.  This module resolves their compact/id-referenced schema into the shapes the
existing loot/task builders consume, and maintains a persistent conditional-HTTP cache so a manual
SYNC does not download unchanged multi-megabyte catalogs again.

The static schema differs from GraphQL in three ways this adapter bridges:
  1. wrapping: each file is { "data": {...}, "translations": [...] }.
  2. id-references: nested objects are ID strings (a container/mob/item/map id) that resolve via
     bundled tables (data.lootContainers, data.mobs, the items dump, data.maps) instead of being
     inlined.
  3. placeholder names: data.* name fields are the literal key "<id> Name"; the real localized
     string lives in the SEPARATE `<name>_en` dump, a flat { "<id> Name": "English" } dict. So we
     fetch maps_en / items_en purely as name lookups. Numeric fields (prices, chances) are real.
"""
import gzip, json, os, time, urllib.request, urllib.error

BASE = "https://json.tarkov.dev/regular/"
UA = "atlas-json-sync/1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CACHE = os.path.join(REPO, "packs", "shared", ".tarkov-json-cache")


def _cache_dir():
    return (os.environ.get("EFT_TARKOV_JSON_CACHE")
            or os.environ.get("EFT_TARKOV_STATIC_CACHE")  # compatibility with the first fallback build
            or DEFAULT_CACHE)


def _read_json(path):
    try:
        with open(path, "rb") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _atomic_write(path, raw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _fetch(name, tries=3):
    """Return (document, changed, downloaded_bytes) using ETag/Last-Modified revalidation."""
    cache_dir = _cache_dir()
    data_path = os.path.join(cache_dir, name + ".json")
    meta_path = os.path.join(cache_dir, name + ".http.json")
    cached_raw = None
    try:
        with open(data_path, "rb") as f:
            cached_raw = f.read()
        cached = json.loads(cached_raw)
    except (OSError, ValueError):
        cached = None
        cached_raw = None

    # sync_intel.py prefetches every required dataset once, then marks the cache ready for its child
    # builders/icon pass. Those children must do zero duplicate HTTP requests.
    if cached is not None and os.environ.get("EFT_TARKOV_JSON_CACHE_READY") == "1":
        return cached, False, 0

    meta = _read_json(meta_path) or {}
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "User-Agent": UA}
    if cached is not None:
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]
    req = urllib.request.Request(BASE + name, headers=headers)
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                wire = resp.read()
                raw = gzip.decompress(wire) if resp.headers.get("Content-Encoding") == "gzip" else wire
                doc = json.loads(raw)
                new_meta = {
                    "etag": resp.headers.get("ETag"),
                    "last_modified": resp.headers.get("Last-Modified"),
                    "checked": int(time.time()),
                }
            content_changed = cached_raw != raw
            if content_changed:
                _atomic_write(data_path, raw)
            _atomic_write(meta_path, json.dumps(new_meta, separators=(",", ":")).encode("ascii"))
            status = "updated" if content_changed else "content unchanged"
            print(f"  [json] {name}: {status} ({len(wire) / 1e6:.2f} MB downloaded)", flush=True)
            return doc, content_changed, len(wire)
        except urllib.error.HTTPError as e:
            if e.code == 304 and cached is not None:
                meta["checked"] = int(time.time())
                _atomic_write(meta_path, json.dumps(meta, separators=(",", ":")).encode("ascii"))
                print(f"  [json] {name}: unchanged (304)", flush=True)
                return cached, False, 0
            last = e
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            last = e
        if i + 1 < tries:
            print(f"  [json] {name}: {getattr(last, 'code', last)} - retry {i + 1}/{tries}", flush=True)
            time.sleep(2 * (i + 1))

    # An existing validated snapshot is more useful than failing an otherwise-offline viewer. Do
    # not call it fresh: the unchanged output mtimes/state still expose its real age in the menu.
    if cached is not None:
        print(f"  [json] {name}: network unavailable - using cached snapshot ({last})", flush=True)
        return cached, False, 0
    raise SystemExit(f"json.tarkov.dev/{name} unreachable and no cache exists: {last}")


def _get(name):
    return _fetch(name)[0]


def prefetch(names):
    """Revalidate exactly the named catalogs once; return (changed_names, downloaded_bytes)."""
    changed, downloaded = set(), 0
    for name in names:
        _, did_change, nbytes = _fetch(name)
        if did_change:
            changed.add(name)
        downloaded += nbytes
    return changed, downloaded


def _list(x):
    """The dumps key collections either by id (dict) or as a plain list - normalise to a list."""
    return x if isinstance(x, list) else list(x.values())


def load_static_items(names=(), ids=()):
    """Resolve item display names/template IDs for fetch_icons.py without GraphQL.

    The static item dump already carries the asset-CDN URLs; only its display-name fields are
    translation keys, resolved through items_en.  Return the same small item shape as the GraphQL
    query so the icon downloader stays source-agnostic.
    """
    wanted_names = {str(x) for x in names if x}
    wanted_ids = {str(x) for x in ids if x}
    I = _get("items")["data"]
    ien = _get("items_en").get("data", {})
    out = []
    for it in _list(I["items"]):
        iid = str(it.get("id") or "")
        name = ien.get(it.get("name")) or it.get("normalizedName")
        if iid not in wanted_ids and name not in wanted_names:
            continue
        out.append({
            "id": iid,
            "name": name,
            "iconLink": it.get("iconLink"),
            "gridImageLink": it.get("gridImageLink"),
        })
    return {"items": out}


def map_slug(display):
    """tarkov.dev map normalizedName from a display name ('Streets of Tarkov' ->
    'streets-of-tarkov') — the same slug rule as the asset CDN uses."""
    out, dash = [], False
    for ch in str(display).lower():
        if ch.isascii() and ch.isalnum():
            out.append(ch)
            dash = False
        elif not dash:
            out.append("-")
            dash = True
    return "".join(out).strip("-")


def _display_name(entry, en_tables):
    """localized name for a data.* entry: the '<id> Name' translation when any table has it,
    else the normalizedName prettified ('duffle-bag' -> 'Duffle bag', short leading token
    upper-cased so 'nsv-...' reads 'NSV ...'). Display only — never a join key."""
    for en in en_tables:
        nm = en.get(entry.get("name") or "")
        if nm:
            return nm
    toks = (entry.get("normalizedName") or "").split("-")
    toks = [t for t in toks if t]
    if not toks:
        return None
    if len(toks[0]) <= 4 and toks[0].isalpha():
        toks[0] = toks[0].upper()
    else:
        toks[0] = toks[0].capitalize()
    return " ".join(toks)


def load_static_containers():
    """lootContainer template id -> display name ('Jacket' / 'Duffle bag' / 'Dead Scav')
    from the maps dump's data.lootContainers table."""
    d = _get("maps").get("data", {})
    en = [_get("maps_en").get("data", {}) or {}, _get("items_en").get("data", {}) or {}]
    out = {}
    for c in _list(d.get("lootContainers") or {}):
        nm = _display_name(c, en)
        if c.get("id") and nm:
            out[str(c["id"])] = nm
    return out


def load_static_stationary():
    """{'weapons': {id -> display name}, 'maps': {map normalizedName -> [{'id', 'pos'}]}}
    from the maps dump: the stationaryWeapons TYPE table + every map's mount positions
    (raw Unity coordinates, exactly as the game serializes them)."""
    d = _get("maps").get("data", {})
    en = [_get("maps_en").get("data", {}) or {}, _get("items_en").get("data", {}) or {}]
    weapons = {}
    for w in _list(d.get("stationaryWeapons") or {}):
        nm = _display_name(w, en)
        if w.get("id") and nm:
            weapons[str(w["id"])] = nm
    per_map = {}
    for m in _list(d.get("maps") or {}):
        slug = m.get("normalizedName") or ""
        rows = []
        for w in m.get("stationaryWeapons") or []:
            p = w.get("position") or {}
            if all(k in p for k in "xyz"):
                rows.append({"id": str(w.get("stationaryWeapon") or ""),
                             "pos": [p["x"], p["y"], p["z"]]})
        if slug and rows:
            # variant scenes ('ground-zero-21') fold into their base map's row set.
            base = slug
            for known in per_map:
                if slug.startswith(known + "-") or known.startswith(slug + "-"):
                    base = known
                    break
            per_map.setdefault(base, []).extend(rows)
    return {"weapons": weapons, "maps": per_map}


def load_static_item_index(ids):
    """{template id: {'n','s','pr','cat'}} for extract_gamedata's loose-loot pools. Real items
    resolve from the items dump (names via items_en, prices real); ids that are CATEGORY
    templates ('Food and drink' pool slots) resolve from itemCategories with cat=1 (display
    name prettified from normalizedName — the static dump carries no category translations)."""
    wanted = {str(i) for i in ids if i}
    if not wanted:
        return {}
    I = _get("items")["data"]
    ien = _get("items_en").get("data", {})
    out = {}
    for it in _list(I["items"]):
        iid = str(it.get("id") or "")
        if iid in wanted:
            out[iid] = {"n": ien.get(it.get("name")) or it.get("normalizedName"),
                        "s": ien.get(it.get("shortName")), "pr": it.get("avg24hPrice"), "cat": 0}
    for c in _list(I.get("itemCategories") or {}):
        cid = str(c.get("id") or "")
        if cid in wanted and cid not in out:
            nm = _display_name(c, [ien])
            if nm:
                out[cid] = {"n": nm, "s": None, "pr": None, "cat": 1}
    return out


def load_static_loose(display_name):
    """One map's lootLoose rows ([{'position', 'items': [{'name','shortName','avg24hPrice'}]}])
    from the static maps dump — the JSON replacement for extract_gamedata's per-map GraphQL
    lootLoose query. Selected by display name (via the EN table) or by normalizedName slug."""
    D = _get("maps")["data"]
    I = _get("items")["data"]
    men = _get("maps_en").get("data", {})
    ien = _get("items_en").get("data", {})
    slug = map_slug(display_name)
    m = next((m for m in _list(D["maps"])
              if m.get("normalizedName") == slug
              or (men.get(m.get("name")) or "").lower() == str(display_name).lower()), None)
    if not m:
        return []
    item_by = {it["id"]: it for it in _list(I["items"])}
    out = []
    for p in (m.get("lootLoose") or []):
        items = []
        for iid in (p.get("items") or []):
            it = item_by.get(iid)
            if it:
                items.append({"name": ien.get(it.get("name")) or it.get("normalizedName"),
                              "shortName": ien.get(it.get("shortName")),
                              "avg24hPrice": it.get("avg24hPrice")})
        out.append({"position": p.get("position"), "items": items})
    return out


def load_static_zone_names():
    """{internal zone key: EN display} — the maps_en table's Zone*/BotZone* rows
    ('ZoneCenterBot' -> 'Center', 'ZoneWoodCutter' -> 'Lumber Mill'). This is the SAME table
    tarkov.dev renders its bosses' spawnLocations names from, so a boss loc name joins a
    first-party BotZone EXACTLY through it (substring matching misses ZoneWoodCutter)."""
    men = _get("maps_en").get("data", {})
    return {k: v for k, v in men.items()
            if isinstance(v, str) and (k.startswith("Zone") or k.startswith("BotZone"))}


def load_static_map_names():
    """{normalizedName: (en, ru)} for gen_maps' roster display names — maps + maps_en +
    maps_ru catalogs (ru degrades to en when its table lacks the key)."""
    D = _get("maps")["data"]
    men = _get("maps_en").get("data", {})
    mru = _get("maps_ru").get("data", {})
    out = {}
    for m in _list(D["maps"]):
        nn = m.get("normalizedName")
        en = men.get(m.get("name")) or nn
        if nn and en:
            out[nn] = (en, mru.get(m.get("name")) or en)
    return out


def load_static_maps():
    """Rebuild the build_loot.py QUERY response ({'maps': [...]}) from the static dumps.

    Includes lootLoose INLINE per map (the base maps dump already carries it), so the caller does not
    need any per-map round-trips.
    """
    print("  [json] resolving maps + items catalogs ...", flush=True)
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
    print(f"  [json] resolved {len(out)} maps", flush=True)
    return {"maps": out}


def load_static_tasks():
    """Rebuild the build_tasks.py QUERY response ({'tasks': [...]}) from the static dumps.

    The static task objectives carry every type-specific field GraphQL exposes, just FLATTENED (no
    union wrapper) and id-referenced (item / questItem / trader / task / map ids + a translation-key
    description). This resolves all of them back to the inlined GraphQL shape main() reads.
    """
    print("  [json] resolving tasks + traders + items catalogs ...", flush=True)
    T = _get("tasks")["data"]
    I = _get("items")["data"]
    TRD = _get("traders")["data"]
    ten = _get("tasks_en").get("data", {})        # task names, objective descriptions, quest-item names
    ien = _get("items_en").get("data", {})
    tren = _get("traders_en").get("data", {})     # trader "<id> Nickname" -> "Prapor"
    MAPS = _get("maps")["data"]
    men = _get("maps_en").get("data", {})         # mob role id -> display name ("bossKojaniy" -> "Shturman")
    map_nn = {m["id"]: m.get("normalizedName") for m in _list(MAPS["maps"])}

    # Kill-objective targets arrive as the GAME's internal bot role ids. The maps dump ships a
    # `mobs` catalog whose ids ARE those roles, and maps_en translates them, so named bosses and
    # followers resolve straight from data ("bossKojaniy" -> "Shturman", "PmcBot" -> "Raider").
    # Nine role ids have no mob row because they are generic CATEGORIES rather than named mobs;
    # for those alone the label below is a display alias, and unresolved ids are passed through
    # verbatim rather than invented.
    GENERIC_ROLES = {"Any": "Anyone", "AnyPmc": "PMC", "Savage": "Scav", "Usec": "USEC",
                     "Bear": "BEAR", "Marksman": "Scav sniper"}
    mob_name = {m: men.get(m) for m in _list(MAPS.get("mobs") or {}) if isinstance(m, str)}

    def target_name(role):
        return men.get(role) or GENERIC_ROLES.get(role) or mob_name.get(role) or role

    # Body parts arrive as full localization KEYS
    # ("QuestCondition/Elimination/Kill/BodyPart/Stomach"); the leaf is the value.
    def body_part(key):
        return (key or "").rsplit("/", 1)[-1]

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
              # z["id"] IS THE GAME'S TRIGGER NAME ("place_SALE_03_KOSTIN",
              # "Sandbox_1_MedicalArea_exploration") — the same string the scene's
              # PlaceItemTrigger/ExperienceTrigger serializes as its zone id and that
              # extract_gamedata.py stores as quest_triggers[].name. Carrying it through turns
              # build_tasks.py's first-party join from a 12 m nearest-neighbour GUESS into an
              # EXACT key match. Dropping it here is what forced the positional fallback before.
              "zones": [{"id": z.get("id"), "map": mapref(z.get("map")), "position": z.get("position"),
                         "outline": z.get("outline"), "top": z.get("top"), "bottom": z.get("bottom")}
                        for z in (o.get("zones") or [])]}
        if o.get("items"):        oo["items"] = resolve_items(o["items"])
        if o.get("markerItem"):   oo["markerItem"] = item_obj(o["markerItem"])
        if o.get("questItem"):    oo["questItem"] = {"name": qitem.get(o["questItem"])}
        if o.get("requiredKeys"): oo["requiredKeys"] = resolve_items(o["requiredKeys"])
        for k in ("usingWeapon", "usingWeaponMods", "wearing", "notWearing", "useAny"):
            if o.get(k):          oo[k] = resolve_items(o[k])
        # scalars main() reads straight through
        for k in ("count", "foundInRaid", "exitName", "distance", "shotType",
                  "timeFromHour", "timeUntilHour", "minDurability", "maxDurability"):
            if o.get(k) is not None:
                oo[k] = o[k]
        if o.get("targetNames"):
            oo["targetNames"] = [target_name(r) for r in o["targetNames"]]
        if o.get("bodyParts"):
            oo["bodyParts"] = [body_part(b) for b in o["bodyParts"]]
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
            # The static dump names this field "skill", not "name" — reading "name" yielded null on
            # all 79 skill rewards across 30 tasks, so "Consolation Prize" shipped 15 rows of
            # "Skill: skill +1". Prefer "skill", keep "name" as the fallback in case the schema
            # gains it later, and translate through tasks_en like every other display string.
            "skillLevelReward": [{"name": (ten.get(x.get("skill") or x.get("name"))
                                           or x.get("skill") or x.get("name")),
                                  "level": x.get("level")}
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
    print(f"  [json] resolved {len(out)} tasks", flush=True)
    return {"tasks": out}


def load_static_mob_names():
    """{game bot-role id: EN display name} — e.g. bossKojaniy -> "Shturman", bossBully -> "Reshala".

    The `mobs` catalog in the maps dump is keyed by the SAME role ids the client uses in a location's
    BossLocationSpawn, and maps_en translates them. That makes it the join between the client's boss
    entries (which carry the real spawn CHANCE) and tarkov.dev's boss nodes (which carry the world
    POSITION). A prefix-strip cannot do this job: bossKojaniy is "Shturman", bossBully is "Reshala",
    bossBoar is "Kaban", sectantPriest is "Cultist Priest".
    """
    men = _get("maps_en").get("data", {}) or {}
    mobs = _get("maps")["data"].get("mobs") or {}
    ids = list(mobs) if isinstance(mobs, dict) else [m.get("id") for m in _list(mobs)]
    return {i: men.get(i) for i in ids if i and men.get(i)}
