#!/usr/bin/env python
"""Extract the CLIENT-SHIPPED intel that lives outside the map scenes -> packs/shared/client_intel.json.

Everything the other extractors read comes from a level scene. Two of the most useful datasets do
not: they are TextAssets baked into `EscapeFromTarkov_Data/resources.assets`.

  1. LOCATION CONFIGS. Snapshots of the server's `/client/location` payload, 88 fields each, keyed
     by the game's own location Id (Interchange / RezervBase / TarkovStreets / Sandbox / ...). They
     carry things NO other source we have does:
       * GlobalLootChanceModifier + GlobalContainerChanceModifier — per-map loot rate. Lighthouse
         ships 0.17 against Interchange's 0.64: a ~4x difference in expected loot that the loot
         planner could not previously model at all.
       * BossLocationSpawn — boss name, spawn CHANCE and the zones it can spawn in, first-party
         (the viewer's boss odds came from tarkov.dev).
       * exits — per-exfil chance, ExfiltrationTime, Min/MaxTime, PassageRequirement, RequiredSlot.
       * EscapeTimeLimit, player counts, bot cadence + difficulty, spawn-distance rules.

  2. ITEM TEMPLATES (`TestItemTemplates`, ~24 MB, 5,381 entries) — the baked `/client/items` payload.
     Not prices (rouble value is server economy data and is NOT in here), but the PHYSICAL facts the
     value model needs: slot footprint (Width x Height), Weight, StackMaxSize, the game's own rarity
     tint (BackgroundColor), LootExperience, QuestItem, CanSellOnRagfair. And for a CONTAINER
     template, its `Grids` give real capacity in cells — a 25-cell supply crate is not a 4-cell
     jacket, in value or in search time.

HOW MUCH TO TRUST IT — measured against what we already ship, not assumed:
  * The configs carry `BackendUrl: stage-01...`, so they are snapshots rather than a live feed. That
    does NOT make them stale: on raid timers the client is the CURRENT one and tarkov.dev's copy in
    our loot.json is behind — customs 40 vs 35, woods 40 vs 35, streets 50 vs 40, labs 35 vs 30,
    ground_zero 50 vs 30. Over raid time + player count on 10 maps, 11 fields agree and 9 differ,
    and the differences are the client being newer.
  * Player counts are the one place to stay cautious: 6 of 10 maps agree exactly and the rest differ
    by one or two (Interchange 10-14 here vs 11-15 upstream) with no clear winner.
  * Rouble prices and per-item spawn weights are NOT here. They stay tarkov.dev.
  Every location record carries `src` so a consumer can attribute it honestly.

  python extraction/intel/extract_client_intel.py            -> packs/shared/client_intel.json
Re-run per game patch (a wipe does not change it; a client update can).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
_OUT_DIR = os.environ.get("EFT_INTEL_OUT_DIR") or os.path.join(REPO, "packs", "shared")
OUT = os.path.join(_OUT_DIR, "client_intel.json")

GAME = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")

# The game's location Id -> our map id (matches extraction/maps/<id>). The client ships several
# variants per location (day/night, difficulty tiers); they collapse onto one map because we render
# one geometry per map. Unmapped ids are REPORTED, never silently dropped.
LOC_TO_ID = {
    "Interchange": "interchange",
    "RezervBase": "reserve",
    "Shoreline": "shoreline",
    "TarkovStreets": "streets",
    "Lighthouse": "lighthouse",
    "Woods": "woods",
    "bigmap": "customs",
    "factory4_day": "factory_rework",
    "factory4_night": "factory_rework",
    "laboratory": "labs",
    "Sandbox": "ground_zero",
    "Sandbox_high": "ground_zero",
    "Labyrinth": "labyrinth",
}

# Location scalars worth carrying. Kept explicit rather than copying all 88 fields: the rest are
# matchmaking/queue plumbing that means nothing without a server.
LOC_SCALARS = (
    "EscapeTimeLimit", "EscapeTimeLimitCoop", "MinPlayers", "MaxPlayers",
    "GlobalLootChanceModifier", "GlobalContainerChanceModifier",
    "AveragePlayTime", "AveragePlayerLevel",
    "MaxBotPerZone", "BotMax", "BotStart", "BotStop", "BotSpawnTimeOnMin", "BotSpawnTimeOnMax",
    "MinDistToExitPoint", "MinDistToFreePoint", "MaxDistToFreePoint",
    "RequiredPlayerLevelMin", "RequiredPlayerLevelMax", "Insurance", "SafeLocation",
)

# Item _props worth carrying (physical facts; NOT prices — those are not in this payload).
ITEM_PROPS = ("Width", "Height", "Weight", "StackMaxSize", "BackgroundColor",
              "LootExperience", "ExamineExperience", "QuestItem", "CanSellOnRagfair",
              "InsuranceDisabled", "DiscardLimit")


def _load_textassets(path):
    """{name: decoded text} for every TextAsset in an assets file. Requires UnityPy."""
    try:
        import UnityPy
    except ImportError:
        print("[client-intel] UnityPy not installed - run INSTALL DEPS (or pip install UnityPy)",
              flush=True)
        return None
    env = UnityPy.load(path)
    out = {}
    for o in env.objects:
        if o.type.name != "TextAsset":
            continue
        try:
            d = o.read()
        except Exception:
            continue
        name = getattr(d, "m_Name", "") or ""
        sc = getattr(d, "m_Script", None)
        if sc is None:
            continue
        try:
            txt = sc if isinstance(sc, str) else bytes(sc).decode("utf-8", "replace")
        except Exception:
            continue
        # Several assets share a name across variants; keep the LARGEST (the richest snapshot).
        if name not in out or len(txt) > len(out[name]):
            out[name] = txt
    return out


def _grid_cells(props):
    """A container template's capacity in cells, summed over its grids (0 = not a container)."""
    total = 0
    for g in (props.get("Grids") or []):
        gp = g.get("_props") or {}
        total += int(gp.get("cellsH") or 0) * int(gp.get("cellsV") or 0)
    return total


def collect_items(assets):
    """{template id: physical facts} from TestItemTemplates, plus container capacity."""
    raw = assets.get("TestItemTemplates")
    if not raw:
        print("[client-intel] no TestItemTemplates - item facts skipped", flush=True)
        return {}
    try:
        D = json.loads(raw)["data"]
    except Exception as e:
        print(f"[client-intel] TestItemTemplates unreadable ({e})", flush=True)
        return {}
    items = {}
    n_cap = 0
    for tid, e in D.items():
        p = e.get("_props") or {}
        rec = {}
        for k in ITEM_PROPS:
            if p.get(k) is not None:
                rec[k[0].lower() + k[1:]] = p[k]
        cells = _grid_cells(p)
        if cells:
            rec["cells"] = cells
            n_cap += 1
        if e.get("_name"):
            rec["id_name"] = e["_name"]
        if rec:
            items[tid] = rec
    print(f"[client-intel] items: {len(items)} templates with physical facts "
          f"({n_cap} are containers with grid capacity)", flush=True)
    return items


def collect_locations(assets):
    """{our map id: location intel} from the baked /client/location snapshots."""
    best = {}
    for name, raw in assets.items():
        s = raw.lstrip()
        if not s.startswith("{") or '"Location"' not in raw[:400]:
            continue
        try:
            L = json.loads(raw)["Location"]
        except Exception:
            continue
        lid = L.get("Id") or L.get("_Id") or name
        # Richest snapshot per location id wins (variants differ in how much they populate).
        if lid not in best or len(raw) > best[lid][0]:
            best[lid] = (len(raw), L)
    out, unmapped = {}, []
    for lid, (_, L) in sorted(best.items()):
        mid = LOC_TO_ID.get(lid)
        if not mid:
            unmapped.append(lid)
            continue
        rec = {k[0].lower() + k[1:]: L[k] for k in LOC_SCALARS if L.get(k) is not None}
        rec["locationId"] = lid
        rec["src"] = "game files (staging snapshot)"
        # BOSSES: name, chance, and the zones it can use. Escort tells you what comes with it.
        bosses = []
        for b in (L.get("BossLocationSpawn") or []):
            if not b.get("BossName"):
                continue
            bosses.append({k: v for k, v in (
                ("name", b.get("BossName")),
                ("chance", b.get("BossChance")),
                ("zones", [z for z in (b.get("BossZone") or "").split(",") if z]),
                ("difficulty", b.get("BossDifficult")),
                ("escort", b.get("BossEscortType")),
                ("escortAmount", b.get("BossEscortAmount")),
                ("forced", b.get("ForceSpawn") or None),
            ) if v not in (None, [], "")})
        if bosses:
            rec["bosses"] = bosses
        # EXITS: the timing + gating facts tarkov.dev does not carry per-exit.
        exits = []
        for x in (L.get("exits") or []):
            if not x.get("Name"):
                continue
            exits.append({k: v for k, v in (
                ("name", x.get("Name")),
                ("chance", x.get("Chance")),
                ("time", x.get("ExfiltrationTime")),
                ("minTime", x.get("MinTime")),
                ("maxTime", x.get("MaxTime")),
                ("type", x.get("ExfiltrationType")),
                ("requirement", x.get("PassageRequirement")),
                ("requiredSlot", x.get("RequiredSlot")),
                ("entryPoints", x.get("EntryPoints")),
                ("playersCount", x.get("PlayersCount") or None),
            ) if v not in (None, "", "None")})
        if exits:
            rec["exits"] = exits
        # A location can map from several ids (factory day/night); merge rather than clobber, and
        # keep the one with the most content so a sparse night variant cannot win.
        prev = out.get(mid)
        if prev is None or (len(rec.get("exits") or []) + len(rec.get("bosses") or [])) > (
                len(prev.get("exits") or []) + len(prev.get("bosses") or [])):
            out[mid] = rec
    if unmapped:
        print(f"[client-intel] {len(unmapped)} location id(s) not mapped to a map "
              f"(extend LOC_TO_ID if one of these is a map we render): {sorted(unmapped)}",
              flush=True)
    for mid, r in sorted(out.items()):
        print(f"[client-intel]   {mid:16s} raid={r.get('escapeTimeLimit')}min "
              f"players={r.get('minPlayers')}-{r.get('maxPlayers')} "
              f"lootMod={r.get('globalLootChanceModifier')} "
              f"bosses={len(r.get('bosses') or [])} exits={len(r.get('exits') or [])}", flush=True)
    return out


def main():
    res = os.path.join(GAME, "resources.assets")
    if not os.path.isfile(res):
        print(f"[client-intel] no resources.assets at {res} - set EFT_GAME_DATA. Nothing written; "
              f"consumers degrade to tarkov.dev only.", flush=True)
        return 1
    print(f"[client-intel] reading {res}", flush=True)
    assets = _load_textassets(res)
    if assets is None:
        return 1
    print(f"[client-intel] {len(assets)} TextAssets", flush=True)
    locations = collect_locations(assets)
    items = collect_items(assets)
    if not locations and not items:
        print("[client-intel] nothing extracted - not writing", flush=True)
        return 1
    doc = {
        "version": 1,
        "source": "EscapeFromTarkov_Data/resources.assets (client-baked)",
        "note": ("Location configs are STAGING snapshots (BackendUrl stage-01), so cross-check "
                 "anything tarkov.dev also provides. Rouble prices and item spawn weights are "
                 "server data and are NOT in this file."),
        "built": int(time.time()),
        "locations": locations,
        "items": items,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"[client-intel] {len(locations)} location(s), {len(items)} item template(s) -> {OUT} "
          f"({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
