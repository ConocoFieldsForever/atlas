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
import os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
_TK = os.environ.get("EFT_TARKMAP_ROOT")
_OUT_DIR = os.environ.get("EFT_INTEL_OUT_DIR") or (
    os.path.join(_TK, "out") if _TK else os.path.join(REPO, "packs", "shared")
)
OUT = os.path.join(_OUT_DIR, 'loot.json')
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


# Populated in main() from tarkov.dev's mobs catalog: {bot role id: EN display name}. Used to join
# the client's boss entries (which carry the real spawn chance) to tarkov.dev's positioned nodes.
MOB_NAMES = {}


def load_client_intel():
    """packs/shared/client_intel.json (extract_client_intel.py), or None when it hasn't been run.

    Absent -> every consumer below falls back to today's behaviour, so this never becomes a hard
    dependency on having the game installed.
    """
    p = os.path.join(REPO, 'packs', 'shared', 'client_intel.json')
    if not os.path.isfile(p):
        print('[loot] no client_intel.json - loot rates / boss odds / exit timings stay tarkov.dev '
              '(run extraction/intel/extract_client_intel.py to add them)')
        return None
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception as e:
        print(f'[loot] client_intel.json unreadable ({e}) - continuing tarkov.dev-only')
        return None


# Search time from CAPACITY, not from a hand-picked constant. The container's own template gives its
# grid size, and cells are what you actually click through: a 25-cell supply crate is not a 4-cell
# jacket. Fitted to keep the old hand-tuned table's range (jacket ~4 s .. big crate ~9 s) so the
# planner's budgets stay comparable, but now MONOTONIC in real capacity instead of per-type opinion.
T_BASE = 2.5        # seconds to reach + open anything
T_PER_CELL = 0.26   # ~1 s per 4 cells of grid to scan


def search_time_from_cells(cells):
    return round(T_BASE + T_PER_CELL * cells, 1)


def apply_client_intel(mid, rec, intel, containers):
    """Fold first-party location facts into one map's record. Returns a short report string."""
    if not intel:
        return ''
    loc = (intel.get('locations') or {}).get(mid)
    if not loc:
        return ''
    notes = []

    # 1. RAID TIMER + PLAYERS. The client is CURRENT here and upstream is behind (customs 40 vs 35,
    #    woods 40 vs 35, streets 50 vs 40, labs 35 vs 30). Take the game's, keep upstream's as
    #    `raid_minutes_dev` so the disagreement stays visible rather than being silently overwritten.
    meta = rec['meta']
    if loc.get('escapeTimeLimit'):
        if meta.get('raid_minutes') and meta['raid_minutes'] != loc['escapeTimeLimit']:
            meta['raid_minutes_dev'] = meta['raid_minutes']
            notes.append(f"raid {meta['raid_minutes']}->{loc['escapeTimeLimit']}min")
        meta['raid_minutes'] = loc['escapeTimeLimit']
        meta['raid_src'] = 'game files'
    for src, dst in (('minPlayers', 'min_players'), ('maxPlayers', 'max_players')):
        if loc.get(src) is not None:
            meta[dst] = loc[src]

    # 2. PER-MAP LOOT RATE — first-party and previously modelled NOWHERE. Lighthouse ships 0.17
    #    against woods' 0.9, so an unscaled `ev` overstates a lighthouse run by ~5x relative to
    #    woods. Scale the expected value; leave `spawn` alone (that is per-container, and the game's
    #    own per-area odds already ride on gamedata's grp_p).
    lm = loc.get('globalLootChanceModifier')
    cm = loc.get('globalContainerChanceModifier')
    if lm:
        for c in containers:
            c['ev'] = int(round(c['ev'] * lm))
        meta['loot_modifier'] = lm
        notes.append(f'ev x{lm}')
    if cm and cm != 1:
        meta['container_modifier'] = cm

    # 3. BOSS BASE RATE + ZONES, first-party — attached to, never replacing, tarkov.dev's chance.
    #    The client's BossChance is the BASE rate and is event-blind: all six location variants per
    #    map ship an identical boss-chance set, no exit is flagged EventAvailable, and the only
    #    Triggers present are quest-gated. Event state is server-driven, so during a "all bosses
    #    100%" event upstream reads 100% while the client still reads 30% — both correct, about
    #    different things. The client also gives the zone NAMES but no world position; the position
    #    is what puts a marker on the map, so the nodes stay. Joined on display name through the mobs
    #    catalog (bossKojaniy is "Shturman", bossBoar is "Kaban" — a prefix-strip cannot do this).
    #    BossChance is a percentage in the client payload; boss_nodes.chance is a 0..1 fraction.
    if loc.get('bosses'):
        game_bosses = []
        for b in loc['bosses']:
            disp = MOB_NAMES.get(b['name'])
            game_bosses.append({'role': b['name'], 'name': disp or b['name'],
                                'chance': round(float(b.get('chance') or 0) / 100.0, 3),
                                'zones': b.get('zones') or [],
                                'escort': MOB_NAMES.get(b.get('escort')) or b.get('escort'),
                                'escort_amount': b.get('escortAmount'),
                                'difficulty': b.get('difficulty')})
        rec['bosses_game'] = game_bosses
        # Attach to the positioned nodes. Several client entries can share a boss (lighthouse ships
        # ten Rogue rows at different chances/zones); take the HIGHEST, which is the chance that it
        # appears at all somewhere on the map.
        best = {}
        for g in game_bosses:
            k = (g['name'] or '').strip().lower()
            if k and g['chance'] > best.get(k, {'chance': -1})['chance']:
                best[k] = g
        n_join = 0
        for nd in rec['boss_nodes']:
            g = best.get((nd.get('name') or '').strip().lower())
            if not g:
                continue
            n_join += 1
            nd['chance_game'] = g['chance']
            if g['zones']:
                nd['zones_game'] = g['zones']
        notes.append(f"{len(game_bosses)} boss entr(ies), {n_join}/{len(rec['boss_nodes'])} joined")

    # 4. EXFIL TIMINGS + GATING. Upstream gives us extract positions and factions; the client gives
    #    how long standing in one takes, its spawn chance, and what it demands of you.
    if loc.get('exits'):
        rec['exits_game'] = loc['exits']
        notes.append(f"{len(loc['exits'])} exit config(s)")

    rec['meta']['intel_src'] = loc.get('src') or 'game files'
    return '; '.join(notes)


def container_capacity_by_type(intel):
    """{container type name: grid cells} from the game's item templates.

    The loot.json container `type` is tarkov.dev's display name for the container ("Jacket",
    "Weapon box (5x5)"). The game's own template for that container carries `Grids`, and the item
    record we extracted keeps its `_name` plus the cell count — so match on the TEMPLATE NAME the
    packs already record against each placed container (`tpl_name`), which is the same display
    string. Names that do not resolve simply keep the hand-tuned time.
    """
    if not intel:
        return {}
    import glob
    # tpl_name -> template id, learned from any pack's gamedata (the packs already did this join).
    tpl_of = {}
    for gp in glob.glob(os.path.join(REPO, 'packs', '*.eftpack', 'gamedata.json')):
        try:
            gd = json.load(open(gp, encoding='utf-8'))
        except Exception:
            continue
        for c in (gd.get('containers') or []):
            if c.get('tpl_name') and c.get('template'):
                tpl_of.setdefault(c['tpl_name'], c['template'])
    items = intel.get('items') or {}
    out = {}
    for nm, tid in tpl_of.items():
        cells = (items.get(tid) or {}).get('cells')
        if cells:
            out[nm] = cells
    return out


def main():
    print("[tarkov.dev/json] building loot + spawns + intel ...")
    # The bundled embeddable Python pins sys.path via python311._pth and does not add this directory.
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import tarkov_static
    data = tarkov_static.load_static_maps()
    source = 'tarkov.dev/json'
    # FIRST-PARTY intel from the client (extract_client_intel.py). Absent -> everything below keeps
    # today's tarkov.dev-only behaviour, so this is never a hard dependency on a game install.
    intel = load_client_intel()
    global MOB_NAMES
    try:
        MOB_NAMES = tarkov_static.load_static_mob_names()
    except Exception as e:
        print(f'[loot] mob-name table unavailable ({e}) - boss odds will not be joined')
        MOB_NAMES = {}
    cap_by_type = container_capacity_by_type(intel)
    if cap_by_type:
        print(f'[loot] container capacity known for {len(cap_by_type)} type(s) from game templates '
              f'- search time derived from real grid cells')
    out = {}
    intel_used = 0
    for m in data['maps']:
        mid = DEV_TO_ID.get(m['normalizedName'])
        if not mid:
            continue
        # ---- static loot containers ----
        containers = []
        unknown = {}
        n_cap_t = 0
        for lc in (m['lootContainers'] or []):
            p = bridge(lc.get('position'))
            if not p:
                continue
            name = (lc['lootContainer'] or {}).get('name') or '?'
            cv = CONTAINER.get(name)
            if cv is None:
                unknown[name] = unknown.get(name, 0) + 1
                cv = DEFAULT_C
            # `t` (search time) prefers the container's REAL capacity from its own game template
            # over the hand-picked constant; falls back to the table when the name has no cells.
            cells = cap_by_type.get(name)
            t = search_time_from_cells(cells) if cells else cv['t']
            rec_c = {'pos': p, 'type': name, 'cls': cv['cls'], 'ev': cv['ev'],
                     'spawn': cv['spawn'], 't': t}
            if cells:
                rec_c['cells'] = cells
                n_cap_t += 1
            containers.append(rec_c)
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
                             'card': 1 if cat == 'keycard' else 0, 'pr': k.get('avg24hPrice'),
                             'low': k.get('low24hPrice'), 'high': k.get('high24hPrice'),
                             'trend': k.get('changeLast48hPercent')})
            locks.append({'pos': p, 'lt': lk.get('lockType') or 'lock',
                          'pw': 1 if lk.get('needsPower') else 0, 'keys': keys})

        # ---- switches / transits / hazards / stationary weapons / faction-tagged extracts ----
        switches = []
        for s in (m['switches'] or []):
            p = bridge(s.get('position'))
            if not p:
                continue
            activates = []
            for op in (s.get('activates') or []):
                target = op.get('target') or {}
                activates.append({'op': op.get('operation') or 'activates',
                                  'id': target.get('id'), 'name': target.get('name')})
            switches.append({'id': s.get('id'), 'pos': p, 'name': s.get('name') or 'Switch',
                             'st': s.get('switchType') or '',
                             'activated_by': (s.get('activatedBy') or {}).get('name'),
                             'activates': activates})
        transits = [x for x in ({'pos': bridge(t.get('position')), 'to': (t.get('map') or {}).get('normalizedName') or '?',
                                 'desc': t.get('description') or '', 'cond': t.get('conditions') or ''}
                                for t in (m['transits'] or [])) if x['pos']]
        hazards = [x for x in ({'pos': bridge(h.get('position')), 'ht': h.get('hazardType') or 'hazard',
                                'name': h.get('name') or ''} for h in (m['hazards'] or [])) if x['pos']]
        stationary = [x for x in ({'pos': bridge(w.get('position')),
                                   'name': (w.get('stationaryWeapon') or {}).get('name') or 'Stationary weapon'}
                                  for w in (m['stationaryWeapons'] or [])) if x['pos']]
        extracts_dev = []
        for e in (m['extracts'] or []):
            p = bridge(e.get('position'))
            if not p:
                continue
            transfer = e.get('transferItem') or {}
            ti = transfer.get('item') or {}
            extracts_dev.append({
                'id': e.get('id'), 'pos': p, 'name': e.get('name') or 'Extract',
                'fac': e.get('faction') or 'shared',
                'outline': [bridge(x) for x in (e.get('outline') or [])],
                'top': e.get('top'), 'bottom': e.get('bottom'),
                'switches': [x.get('name') for x in (e.get('switches') or []) if x.get('name')],
                'transfer': ({'n': ti.get('name'), 's': ti.get('shortName'),
                              'count': transfer.get('count') if transfer.get('count') is not None else transfer.get('quantity'),
                              'pr': ti.get('avg24hPrice')} if ti else None),
            })

        btr = [{'name': x.get('name') or 'BTR stop', 'pos': bridge(x)} for x in (m.get('btrStops') or [])]
        btr = [x for x in btr if x['pos']]
        artillery = []
        for z in ((m.get('artillery') or {}).get('zones') or []):
            p = bridge(z.get('position'))
            if p:
                artillery.append({'pos': p, 'outline': [bridge(x) for x in (z.get('outline') or [])],
                                  'top': z.get('top'), 'bottom': z.get('bottom')})

        # ---- valuable LOOSE loot points (filtered to GPU/LEDX/keycard-tier so the layer isn't clutter) ----
        loose = []
        for ll in (m.get('lootLoose') or []):
            best = None
            for it in (ll.get('items') or []):
                if best is None or (it.get('avg24hPrice') or 0) > (best.get('avg24hPrice') or 0):
                    best = it
            if best and (best.get('avg24hPrice') or 0) >= LOOSE_MIN_EV:
                p = bridge(ll.get('position'))
                if p:
                    vendors = sorted((best.get('sellFor') or []), key=lambda x: x.get('priceRUB') or 0, reverse=True)
                    sell = vendors[0] if vendors else {}
                    loose.append({'pos': p, 's': best.get('shortName'), 'n': best.get('name'),
                                  'pr': best.get('avg24hPrice'), 'low': best.get('low24hPrice'),
                                  'high': best.get('high24hPrice'), 'trend': best.get('changeLast48hPercent'),
                                  'vendor': (sell.get('vendor') or {}).get('name'), 'sell': sell.get('priceRUB'),
                                  't': 2, 'jackpot': 1})

        out[mid] = {'containers': containers, 'pmc_nodes': pmc_nodes, 'scav_nodes': scav_nodes, 'boss_nodes': boss_nodes,
                    'locks': locks, 'switches': switches, 'transits': transits, 'hazards': hazards,
                    'stationary': stationary, 'extracts_dev': extracts_dev, 'loose': loose,
                    'btr': btr, 'artillery': artillery,
                    'meta': {'name': m.get('name'), 'wiki': m.get('wiki'), 'description': m.get('description'),
                             'enemies': m.get('enemies') or [], 'raid_minutes': m.get('raidDuration'),
                             'players': m.get('players'), 'min_level': m.get('minPlayerLevel'),
                             'max_level': m.get('maxPlayerLevel')},
                    'access_keys': [{'n': k.get('name'), 's': k.get('shortName'), 'pr': k.get('avg24hPrice')}
                                    for k in (m.get('accessKeys') or [])]}
        note = apply_client_intel(mid, out[mid], intel, containers)
        if note:
            intel_used += 1
            print(f'[loot]   {mid}: first-party {note}')
        eff = sum(c['ev'] * c['spawn'] for c in containers)
        from collections import Counter as _C
        by_cls = _C(c['cls'] for c in containers)
        bstr = ','.join(f"{b['boss']}({b['chance']})" for b in boss_nodes) or '-'
        nkc = sum(1 for l in locks if l['keys'] and l['keys'][0]['card'])
        print(f"  {mid:12s} containers={len(containers):4d} (eff.EV {eff/1e6:.1f}M R)  pmc={len(pmc_nodes)} scav={len(scav_nodes)}  bosses={bstr}"
              f"  locks={len(locks)}(kc {nkc}) sw={len(switches)} tr={len(transits)} hz={len(hazards)} ext={len(extracts_dev)} loose={len(loose)}"
              + (f"  [unmapped: {dict(list(unknown.items())[:5])}]" if unknown else ""))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({'version': 3, 'source': source, 'built': int(time.time()),
               'coord_bridge': 'viewer = diag(-1,1,1) * unity',
               'value_model': {'pmc_kill_ev': PMC_KILL_EV, 'boss_fight_t': BOSS_FIGHT_T, 'loose_min_ev': LOOSE_MIN_EV,
                               'note': 'container ev = type-average filled value; effective value = ev*spawn (fill rate). '
                                       'boss_nodes.ev already = kit_value * real spawnChance. PMC nodes carry n (spawn-point '
                                       'density) — the planner weights PMC value by n/mean. v2 adds locks(+keys/keycards), '
                                       'v3 adds raid metadata, switch/extract dependencies, extract footprints and fees, '
                                       'BTR stops, artillery zones, and loose-loot price ranges/trends/vendor values. '
                                       'tune ev tables in build_loot.py'},
               'maps': out}, open(OUT, 'w'), separators=(',', ':'))
    print(f"[loot] -> {OUT} ({os.path.getsize(OUT)/1e3:.0f} KB, {len(out)} maps)")


if __name__ == '__main__':
    main()
