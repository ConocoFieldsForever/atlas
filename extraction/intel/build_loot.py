#!/usr/bin/env python
"""Build per-map LOOT intel for the loot-planner overlay (loot.html / _loot.js) -> out/loot.json.

Mines the loot SYSTEM from tarkov game data (tarkov.dev, which is community-extracted from the same Unity assets we
render) and COORDINATE-BRIDGES every container to viewer world with the same G3 = diag(-1,1,1) the geometry uses
(viewer = (-x, y, z)) -- so containers land exactly on the rendered map.

What we pull:
  - maps.lootContainers  -> every STATIC loot container's TYPE + world position (823 on Interchange).
  - maps.spawns          -> PMC/player spawn points -> "combat value" nodes (where you fight/kill PMCs for their gear).

Value model: each container TYPE gets an EXPECTED value in roubles (`ev`) and a loot TIME in seconds (`t`). These are
community-grounded averages of the container's loot-table EV (the real per-item pool*price*fillrate; refine from an SPT
`looseLoot`/`staticLoot` dump if you want exact numbers). The planner uses ev/t (value density) + walk distance to solve
the time-limited raid as an ORIENTEERING problem. Everything here is data-driven + tunable; no per-container hand placement.

  python extraction/intel/build_loot.py            -> <EFT_TARKMAP_ROOT>/out/loot.json
Re-run per wipe (prices/containers shift). No game files needed (tarkov.dev only)."""
import os, json, urllib.request, time

HERE = os.path.dirname(os.path.abspath(__file__))
# portable kit: output goes to the workspace tarkmap/out so assemble_bevy.py auto-ships it into packs.
_TK = os.environ.get("EFT_TARKMAP_ROOT")
if not _TK:
    raise SystemExit("build_loot: EFT_TARKMAP_ROOT is not set. Point it at your workspace tarkmap dir "
                     "(the one holding maps/ and out/), e.g.  setx EFT_TARKMAP_ROOT D:\\eft_work\\tarkmap")
OUT = os.path.join(_TK, 'out', 'loot.json')
API = "https://api.tarkov.dev/graphql"
DEV_TO_ID = {
    'interchange': 'interchange', 'ground-zero': 'ground_zero', 'ground-zero-21': 'ground_zero',
    # The shipped roster's "Factory" is the 1.0 rework (id factory_rework). tarkov.dev still calls
    # it factory / night-factory, so both map to factory_rework or the pack loads with no POI.
    'factory': 'factory_rework', 'night-factory': 'factory_rework', 'woods': 'woods', 'customs': 'customs',
    'shoreline': 'shoreline', 'streets-of-tarkov': 'streets', 'reserve': 'reserve',
    'the-lab': 'labs', 'the-labs': 'labs', 'lighthouse': 'lighthouse',
    'the-labyrinth': 'labyrinth', 'labyrinth': 'labyrinth',
}

# Per container TYPE: ev = roubles WHEN it spawns loot; spawn = P(worthwhile loot this raid = "fill rate"); t = seconds
# to open+search; cls = class (for include/exclude filtering). EFFECTIVE value the optimiser uses = ev * spawn -- so a
# hidden STASH (Ground/Buried cache) with a ~0.35 fill rate ranks far below a weapon box even at similar filled value,
# and you can drop the whole 'stash' class in the planner. All community-grounded + tunable.
CONTAINER = {
    'Safe':               {'ev': 62000, 't': 6, 'spawn': 0.70, 'cls': 'safe'},
    'Bank safe':          {'ev': 82000, 't': 6, 'spawn': 0.70, 'cls': 'safe'},
    'Weapon box':         {'ev': 41000, 't': 8, 'spawn': 0.88, 'cls': 'weapon'},
    'Weapon box (5x5)':   {'ev': 45000, 't': 8, 'spawn': 0.90, 'cls': 'weapon'},
    'Weapon box (6x3)':   {'ev': 41000, 't': 8, 'spawn': 0.88, 'cls': 'weapon'},
    'Weapon box (5x2)':   {'ev': 30000, 't': 7, 'spawn': 0.85, 'cls': 'weapon'},
    'Weapon box (4x4)':   {'ev': 34000, 't': 7, 'spawn': 0.86, 'cls': 'weapon'},
    'Weapon box (4x2)':   {'ev': 24000, 't': 6, 'spawn': 0.84, 'cls': 'weapon'},
    'Wooden ammo box':    {'ev': 5000,  't': 4, 'spawn': 0.80, 'cls': 'weapon'},
    'Ammo box':           {'ev': 5000,  't': 4, 'spawn': 0.80, 'cls': 'weapon'},
    'PC block':           {'ev': 28000, 't': 5, 'spawn': 0.78, 'cls': 'tech'},
    'Toolbox':            {'ev': 14000, 't': 5, 'spawn': 0.82, 'cls': 'tech'},
    'Ground cache':       {'ev': 22000, 't': 6, 'spawn': 0.35, 'cls': 'stash'},   # HIDDEN STASH — low fill rate
    'Buried barrel cache':{'ev': 22000, 't': 6, 'spawn': 0.35, 'cls': 'stash'},   # HIDDEN STASH — low fill rate
    'Duffle bag':         {'ev': 12000, 't': 4, 'spawn': 0.70, 'cls': 'bag'},
    'Jacket':             {'ev': 8000,  't': 4, 'spawn': 0.65, 'cls': 'bag'},
    'Plastic suitcase':   {'ev': 9000,  't': 4, 'spawn': 0.68, 'cls': 'bag'},
    'Wooden crate':       {'ev': 9000,  't': 5, 'spawn': 0.72, 'cls': 'crate'},
    'Grenade box':        {'ev': 8500,  't': 4, 'spawn': 0.75, 'cls': 'crate'},
    'Technical supply crate': {'ev': 16000, 't': 6, 'spawn': 0.78, 'cls': 'crate'},
    'Ration supply crate':{'ev': 6000,  't': 5, 'spawn': 0.75, 'cls': 'crate'},
    'Medical supply crate':{'ev': 9000, 't': 5, 'spawn': 0.75, 'cls': 'medical'},
    'Medbag':             {'ev': 7000,  't': 4, 'spawn': 0.72, 'cls': 'medical'},
    'Medbag SMU06':       {'ev': 7000,  't': 4, 'spawn': 0.72, 'cls': 'medical'},
    'Medcase':            {'ev': 9000,  't': 4, 'spawn': 0.72, 'cls': 'medical'},
    'Cash register':      {'ev': 7500,  't': 3, 'spawn': 0.85, 'cls': 'register'},
    'Cash register TAR2-2':{'ev': 7500, 't': 3, 'spawn': 0.85, 'cls': 'register'},
    'Bank cash register': {'ev': 12000, 't': 3, 'spawn': 0.85, 'cls': 'register'},
    'Drawer':             {'ev': 4500,  't': 3, 'spawn': 0.60, 'cls': 'furniture'},
    'Dead Scav':          {'ev': 11000, 't': 5, 'spawn': 0.90, 'cls': 'body'},
    'PMC body':           {'ev': 60000, 't': 6, 'spawn': 1.00, 'cls': 'body'},
    'Scav body':          {'ev': 9000,  't': 5, 'spawn': 1.00, 'cls': 'body'},
    'Civilian body':      {'ev': 6000,  't': 5, 'spawn': 1.00, 'cls': 'body'},
    'Lab technician body':{'ev': 12000, 't': 5, 'spawn': 1.00, 'cls': 'body'},
    'Airdrop':            {'ev': 150000,'t': 10,'spawn': 1.00, 'cls': 'special'},
    "Shturman's Stash":   {'ev': 45000, 't': 6, 'spawn': 0.55, 'cls': 'stash'},
}
DEFAULT_C = {'ev': 6000, 't': 4, 'spawn': 0.60, 'cls': 'misc'}

# PMC kill value: a fought-and-looted PMC ~ their gear. Nodes are spawn CLUSTERS (fight zones), value = per-kill EV.
PMC_KILL_EV = 90000     # mean value of a killed PMC's lootable kit (rough; tune)
PMC_CLUSTER_R = 22.0    # merge spawn points within this many metres into one fight node

# BOSS kit value (roubles): a killed boss drops unique gear/weapons + a good kit. Multiplied by the map's REAL spawn
# CHANCE (from tarkov.dev, e.g. Killa 0.75 on Interchange) so a boss node's effective value already bakes in "how
# often is it actually there" — the spawn-rate factoring the planner ranks on, exactly like container fill-rate.
BOSS_EV = {
    'killa': 350000, 'tagilla': 260000, 'reshala': 200000, 'shturman': 220000, 'sanitar': 210000,
    'gluhar': 300000, 'kaban': 380000, 'kolontay': 300000, 'zryachiy': 230000, 'partisan': 150000,
    'knight': 400000, 'bigpipe': 320000, 'birdeye': 320000, 'cultist-priest': 180000, 'legion': 300000,
}
BOSS_EV_DEFAULT = 200000
BOSS_FIGHT_T = 120      # seconds to find + fight + loot a boss (+ its guards) — longer/riskier than a lone PMC


def bridge(p):
    return None if not p else [round(-p['x'], 2), round(p['y'], 2), round(p['z'], 2)]


# Loose-loot points are ~600/map (every jacket/table item pool). A loose point's `items` is the
# whole POOL that CAN spawn there, so its MAX price is high almost everywhere — to make this a
# useful "jackpot spots" layer (GPU ~120k / LEDX ~150k / elite keys) rather than clutter, keep
# only points whose best possible item clears this bar. Tunable.
LOOSE_MIN_EV = 120000

QUERY = """{ maps {
  normalizedName
  name
  spawns { zoneName sides categories position { x y z } }
  bosses { boss { name normalizedName } spawnChance spawnTime spawnTrigger
           escorts { boss { normalizedName } amount { count chance } }
           spawnLocations { name chance } }
  lootContainers { lootContainer { name normalizedName } position { x y z } }
  locks { lockType needsPower key { name shortName avg24hPrice category { name } } position { x y z } }
  switches { name switchType position { x y z } }
  transits { description conditions map { normalizedName } position { x y z } }
  hazards { hazardType name position { x y z } }
  stationaryWeapons { stationaryWeapon { name } position { x y z } }
  extracts { name faction position { x y z } }
  accessKeys { name shortName }
} }"""

# lootLoose is ~6.4k points across all maps — folding it into the big query (or even one all-maps
# loose query) 503s / drops the 4 MB reply, so it's fetched PER MAP (small, robust) by display name.
LOOSE_QUERY = '{ maps(name:"%s"){ lootLoose { position { x y z } items { shortName name avg24hPrice } } } }'


def fetch_loose(display_name):
    """Per-map loose-loot points; returns [] on any failure so the pipeline never dies on it."""
    try:
        ms = gql(LOOSE_QUERY % display_name.replace('"', ''))['maps']
        return (ms[0].get('lootLoose') or []) if ms else []
    except SystemExit:
        print(f"  [loose] {display_name}: fetch failed — skipping loose layer")
        return []


def gql(q, tries=4):
    req = urllib.request.Request(API, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "tarkmap-loot/1.0"})
    last = None
    for i in range(tries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            if 'errors' in r:
                raise SystemExit("tarkov.dev errors: " + json.dumps(r['errors'][:3]))
            return r['data']
        except urllib.error.URLError as e:  # 503/429/transient — back off and retry
            last = e
            print(f"  [gql] {getattr(e, 'code', e)} — retry {i + 1}/{tries}")
            time.sleep(2 * (i + 1))
    raise SystemExit(f"tarkov.dev unreachable after {tries} tries: {last}")


def cluster(pts, radius):
    """greedy-merge nearby [x,y,z] points -> list of (centroid, count)."""
    out = []
    for p in pts:
        for c in out:
            if (p[0] - c['s'][0]) ** 2 + (p[2] - c['s'][2]) ** 2 <= radius * radius:
                c['s'] = [c['s'][0] + p[0], c['s'][1] + p[1], c['s'][2] + p[2]]; c['n'] += 1; break
        else:
            out.append({'s': list(p), 'n': 1})
    return [([round(c['s'][0] / c['n'], 2), round(c['s'][1] / c['n'], 2), round(c['s'][2] / c['n'], 2)], c['n']) for c in out]


def main():
    print("[tarkov.dev] fetching loot + spawns + intel ...")
    data = gql(QUERY)
    out = {}
    for m in data['maps']:
        mid = DEV_TO_ID.get(m['normalizedName'])
        if not mid:
            continue
        # ---- static loot containers ----
        containers = []
        unknown = {}
        for lc in (m['lootContainers'] or []):
            p = bridge(lc.get('position'))
            if not p:
                continue
            name = (lc['lootContainer'] or {}).get('name') or '?'
            cv = CONTAINER.get(name)
            if cv is None:
                unknown[name] = unknown.get(name, 0) + 1
                cv = DEFAULT_C
            containers.append({'pos': p, 'type': name, 'cls': cv['cls'], 'ev': cv['ev'], 'spawn': cv['spawn'], 't': cv['t']})
        # ---- normalize each spawn's category/side sets (lowercase) for DEFENSIBLE filtering ----
        # (Codex review: `sides:all` is NOT player-PMC, and `categories:bot` alone sweeps in raiders/
        #  rogues/cultists/boss-guards. Split them explicitly.)
        def sets(s):
            return ({str(c).lower() for c in (s.get('categories') or [])},
                    {str(x).lower() for x in (s.get('sides') or [])})
        SPECIAL = {'boss', 'raider', 'rogue', 'cultist', 'sectant', 'bossfollower', 'follower', 'exusec'}
        def is_pmc(s):
            c, sd = sets(s)
            return ('botpmc' in c) or ('pmc' in sd)                          # true player-PMC spawns only
        def is_boss(s):
            c, _ = sets(s)
            return 'boss' in c
        def is_scav(s):
            c, sd = sets(s)
            if is_pmc(s) or (c & SPECIAL):
                return False
            return ('bot' in c) or ('assault' in c) or ('scav' in sd) or ('savage' in sd)

        # ---- PMC fight nodes (clustered player-PMC spawns) ----
        pmc_pts = [p for p in (bridge(s.get('position')) for s in (m['spawns'] or []) if is_pmc(s)) if p]
        pmc_nodes = [{'pos': c, 'n': n, 'ev': PMC_KILL_EV} for c, n in cluster(pmc_pts, PMC_CLUSTER_R)]

        # ---- SCAV spawn nodes (regular scavs only) ----
        scav_pts = [p for p in (bridge(s.get('position')) for s in (m['spawns'] or []) if is_scav(s)) if p]
        scav_nodes = [{'pos': c, 'n': n} for c, n in cluster(scav_pts, PMC_CLUSTER_R)]

        # ---- BOSS fight nodes: place each boss at its NAMED spawn zone by matching
        # bosses.spawnLocations[].name to a boss-category spawn's zoneName (Codex fix — the old
        # `boss_clusters[i % len]` scattered bosses across clusters arbitrarily). Fall back to the
        # largest unused boss cluster if no name matches; never invent a marker with no geometry.
        # ev already bakes in spawnChance so the planner ranks by EXPECTED value.
        from collections import defaultdict as _dd
        zone_pts = _dd(list)
        for s in (m['spawns'] or []):
            if is_boss(s):
                p = bridge(s.get('position'))
                if p:
                    zone_pts[(s.get('zoneName') or '').lower()].append(p)
        zone_cen = {z: [round(sum(c[i] for c in pts) / len(pts), 2) for i in range(3)]
                    for z, pts in zone_pts.items() if pts}
        fallback = sorted(cluster([p for pts in zone_pts.values() for p in pts], PMC_CLUSTER_R), key=lambda cc: -cc[1])

        def match_zone(spawn_locs):
            for sl in (spawn_locs or []):
                nm = (sl.get('name') or '').lower()
                if nm:
                    for z, cen in zone_cen.items():
                        if z and (nm in z or z in nm):
                            return cen
            return None

        ranked_bosses = sorted(
            [b for b in (m['bosses'] or []) if float(b.get('spawnChance') or 0) > 0],
            key=lambda b: -(BOSS_EV.get((b['boss'] or {}).get('normalizedName', ''), BOSS_EV_DEFAULT) * float(b.get('spawnChance') or 0)))
        boss_nodes, fb_i = [], 0
        for b in ranked_bosses:
            nm = (b['boss'] or {}).get('normalizedName') or 'boss'
            ch = round(float(b.get('spawnChance') or 0), 3)
            cen = match_zone(b.get('spawnLocations'))
            if cen is None:
                if fb_i < len(fallback):
                    cen = fallback[fb_i][0]; fb_i += 1
                elif fallback:
                    cen = fallback[-1][0]
                else:
                    continue
            locs = [{'name': sl.get('name'), 'chance': round(float(sl.get('chance') or 0), 3)}
                    for sl in (b.get('spawnLocations') or []) if sl.get('name')]
            escorts = []
            for e in (b.get('escorts') or []):
                amt = e.get('amount') or []
                a0 = amt[0] if isinstance(amt, list) and amt else (amt if isinstance(amt, dict) else {})
                escorts.append({'boss': (e.get('boss') or {}).get('normalizedName') or 'guard',
                                'count': (a0 or {}).get('count'), 'chance': round(float((a0 or {}).get('chance') or 0), 3)})
            boss_nodes.append({'pos': cen, 'boss': nm, 'name': (b['boss'] or {}).get('name') or nm, 'chance': ch,
                               'ev': round(BOSS_EV.get(nm, BOSS_EV_DEFAULT) * ch), 't': BOSS_FIGHT_T,
                               'st': b.get('spawnTime'), 'locs': locs, 'escorts': escorts})

        # ---- LOCKS + keys/keycards (headline intel: every locked door/container/trunk + the key that opens it) ----
        locks = []
        for lk in (m['locks'] or []):
            p = bridge(lk.get('position'))
            if not p:
                continue
            k = lk.get('key') or {}
            keys = []
            if k:
                cat = ((k.get('category') or {}).get('name') or '').lower()
                keys.append({'n': k.get('name'), 's': k.get('shortName'),
                             'card': 1 if cat == 'keycard' else 0, 'pr': k.get('avg24hPrice')})
            locks.append({'pos': p, 'lt': lk.get('lockType') or 'lock',
                          'pw': 1 if lk.get('needsPower') else 0, 'keys': keys})

        # ---- switches / transits / hazards / stationary weapons / faction-tagged extracts ----
        switches = [x for x in ({'pos': bridge(s.get('position')), 'name': s.get('name') or 'Switch',
                                 'st': s.get('switchType') or ''} for s in (m['switches'] or [])) if x['pos']]
        transits = [x for x in ({'pos': bridge(t.get('position')), 'to': (t.get('map') or {}).get('normalizedName') or '?',
                                 'desc': t.get('description') or '', 'cond': t.get('conditions') or ''}
                                for t in (m['transits'] or [])) if x['pos']]
        hazards = [x for x in ({'pos': bridge(h.get('position')), 'ht': h.get('hazardType') or 'hazard',
                                'name': h.get('name') or ''} for h in (m['hazards'] or [])) if x['pos']]
        stationary = [x for x in ({'pos': bridge(w.get('position')),
                                   'name': (w.get('stationaryWeapon') or {}).get('name') or 'Stationary weapon'}
                                  for w in (m['stationaryWeapons'] or [])) if x['pos']]
        extracts_dev = [x for x in ({'pos': bridge(e.get('position')), 'name': e.get('name') or 'Extract',
                                     'fac': e.get('faction') or 'shared'} for e in (m['extracts'] or [])) if x['pos']]

        # ---- valuable LOOSE loot points (filtered to GPU/LEDX/keycard-tier so the layer isn't clutter) ----
        loose = []
        for ll in fetch_loose(m.get('name') or m['normalizedName']):
            best = None
            for it in (ll.get('items') or []):
                if best is None or (it.get('avg24hPrice') or 0) > (best.get('avg24hPrice') or 0):
                    best = it
            if best and (best.get('avg24hPrice') or 0) >= LOOSE_MIN_EV:
                p = bridge(ll.get('position'))
                if p:
                    loose.append({'pos': p, 's': best.get('shortName'), 'n': best.get('name'), 'pr': best.get('avg24hPrice')})

        out[mid] = {'containers': containers, 'pmc_nodes': pmc_nodes, 'scav_nodes': scav_nodes, 'boss_nodes': boss_nodes,
                    'locks': locks, 'switches': switches, 'transits': transits, 'hazards': hazards,
                    'stationary': stationary, 'extracts_dev': extracts_dev, 'loose': loose,
                    'access_keys': [{'n': k.get('name'), 's': k.get('shortName')} for k in (m.get('accessKeys') or [])]}
        eff = sum(c['ev'] * c['spawn'] for c in containers)
        from collections import Counter as _C
        by_cls = _C(c['cls'] for c in containers)
        bstr = ','.join(f"{b['boss']}({b['chance']})" for b in boss_nodes) or '-'
        nkc = sum(1 for l in locks if l['keys'] and l['keys'][0]['card'])
        print(f"  {mid:12s} containers={len(containers):4d} (eff.EV {eff/1e6:.1f}M R)  pmc={len(pmc_nodes)} scav={len(scav_nodes)}  bosses={bstr}"
              f"  locks={len(locks)}(kc {nkc}) sw={len(switches)} tr={len(transits)} hz={len(hazards)} ext={len(extracts_dev)} loose={len(loose)}"
              + (f"  [unmapped: {dict(list(unknown.items())[:5])}]" if unknown else ""))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({'version': 2, 'source': 'tarkov.dev', 'built': int(time.time()),
               'coord_bridge': 'viewer = diag(-1,1,1) * unity',
               'value_model': {'pmc_kill_ev': PMC_KILL_EV, 'boss_fight_t': BOSS_FIGHT_T, 'loose_min_ev': LOOSE_MIN_EV,
                               'note': 'container ev = type-average filled value; effective value = ev*spawn (fill rate). '
                                       'boss_nodes.ev already = kit_value * real spawnChance. PMC nodes carry n (spawn-point '
                                       'density) — the planner weights PMC value by n/mean. v2 adds locks(+keys/keycards), '
                                       'switches, transits, hazards, stationary, extracts_dev(faction), loose(valuable). '
                                       'tune ev tables in build_loot.py'},
               'maps': out}, open(OUT, 'w'), separators=(',', ':'))
    print(f"[loot] -> {OUT} ({os.path.getsize(OUT)/1e3:.0f} KB, {len(out)} maps)")


if __name__ == '__main__':
    main()
