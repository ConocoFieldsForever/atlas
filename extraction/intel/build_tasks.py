#!/usr/bin/env python
"""Build the TASK catalog for the tarkmap task-tracker (out/tasks.json).

Pulls the full quest/task list from tarkov.dev's pre-generated JSON API, keeps the
fields the tracker needs (name/trader/map/level/kappa/prereqs + every objective with its description and, where it has a
map location, its zone position + outline), and COORDINATE-BRIDGES every position/outline into viewer-world space with the
SAME G3 = diag(-1,1,1) conjugation the geometry pipeline uses (viewer = (-x, y, z)). So a task zone drops straight onto the
tacmap. Tasks span all maps, so this is ONE global catalog the viewer filters by the current map — supports every map.

  python extraction/intel/build_tasks.py                 -> <EFT_TARKMAP_ROOT>/out/tasks.json  (all tasks, all maps)
Re-run per wipe (task data changes per wipe, not per session).

ZONE GEOMETRY IS FIRST-PARTY where the game has it: tarkov.dev supplies task IDENTITY (name,
trader, prereqs, rewards - none of which the client ships), but every objective zone also exists
in the scene as a typed trigger with a real polygon footprint, and that wins. See
apply_first_party_zones. Falls back cleanly to tarkov.dev-only when no map has been built."""
import os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# Standalone viewers may not have the old tarkmap source tree. Prefer an explicit build-output
# directory, retain the legacy EFT_TARKMAP_ROOT contract, then fall back to the viewer's shared
# pack data so the in-app SYNC button works on a clean checkout.
_TK = os.environ.get("EFT_TARKMAP_ROOT")
_OUT_DIR = os.environ.get("EFT_INTEL_OUT_DIR") or (
    os.path.join(_TK, "out") if _TK else os.path.join(REPO, "packs", "shared")
)
OUT = os.path.join(_OUT_DIR, 'tasks.json')
# tarkov.dev map normalizedName -> our map id (matches tarkmap/maps/<id>). Extend as maps are added.
DEV_TO_ID = {
    'interchange': 'interchange', 'ground-zero': 'ground_zero', 'ground-zero-21': 'ground_zero',
    # Shipped "Factory" is the 1.0 rework (id factory_rework); tarkov.dev still names it
    # factory / night-factory, so map both to factory_rework or the quest layer is empty there.
    'factory': 'factory_rework', 'night-factory': 'factory_rework', 'woods': 'woods', 'customs': 'customs',
    'shoreline': 'shoreline', 'streets-of-tarkov': 'streets', 'reserve': 'reserve',
    'the-lab': 'labs', 'the-labs': 'labs', 'lighthouse': 'lighthouse',
    'the-labyrinth': 'labyrinth', 'labyrinth': 'labyrinth',
}
G3 = (-1.0, 1.0, 1.0)   # Unity world -> viewer world (X-flip), read logically from coordinates.global_matrix

def bridge(p):
    return None if p is None else [round(G3[0] * p['x'], 2), round(G3[1] * p['y'], 2), round(G3[2] * p['z'], 2)]

def map_id(nn):
    return DEV_TO_ID.get(nn, nn)


def zclean(z):
    """Drop null-valued keys from a zone.

    Not cosmetic. A consumer declaring `top: f32` with serde's `default` accepts a MISSING key but
    rejects an explicit `null`, and one such key fails the WHOLE document. Exactly that happened:
    a single "outline": null (from the old supplemental zone patch) made poi.rs's parse of all 501
    tasks fail, and because the result was swallowed with `.ok()` the map simply had no
    tracked-quest zones and said nothing about why. Emitting absence as absence removes the trap
    at the source; poi.rs was also made null-tolerant and now logs the error.
    """
    return {k: v for k, v in z.items() if v is not None}


def conv_zone(z):
    zid = map_id(z['map']['normalizedName']) if z.get('map') else None
    # 'zid' is upstream's zone id, which IS the game's trigger name — the exact join key.
    return zclean({'map': zid, 'zid': z.get('id') or None, 'pos': bridge(z.get('position')),
                   'outline': [bridge(p) for p in z['outline']] if z.get('outline') else None,
                   'top': z.get('top'), 'bottom': z.get('bottom')})


# ---------------------------------------------------------------------------------------------
# FIRST-PARTY ZONE GEOMETRY
# ---------------------------------------------------------------------------------------------
# tarkov.dev is the only source for task IDENTITY (name, trader, prereqs, rewards) - the client
# ships none of that. It is NOT the source for task GEOMETRY: every objective zone exists in the
# scene as a typed trigger component (PlaceItemTrigger / ExperienceTrigger / FlareShootDetectorZone
# / QuestTrigger) with a true polygon footprint, and that wins outright.
#
# THE JOIN IS BY NAME, NOT BY POSITION. Upstream's zone `id` IS the game's serialized trigger name
# ("place_SALE_03_KOSTIN", "Sandbox_1_MedicalArea_exploration") - the exact string
# extract_gamedata.py stores as quest_triggers[].name. Measured over the built packs
# (factory_rework / interchange / woods): 189 zones join by name, 5 do not (97.4%). The earlier
# implementation of this function joined by NEAREST POSITION within 12 m because tarkov_static.py
# dropped that id while resolving; a radius join silently mis-assigns when two triggers sit close
# together, and cannot resolve a zone whose upstream position is simply wrong. The name join has
# neither failure mode - and it fixes the one zone that radius-matching flagged suspect on
# interchange ("The Blood of War - Part 1", 75.8 m from any trigger: its id is place_WARBLOOD_04_2,
# which the game has exactly).
#
# Position matching is kept ONLY as a fallback for zones upstream ships without an id. A zone that
# neither joins by name nor lands near a trigger keeps upstream geometry and is MARKED, so a stale
# position is visible as such instead of being drawn as fact.
MATCH_R = 12.0      # metres; used only by the id-less fallback path
SUSPECT_R = 30.0    # beyond this from ANY trigger, an unjoined dev position is probably wrong


def _load_game_zones():
    """{map id: {'by_name': {name: rec}, 'all': [rec]}} from every pack's gamedata.json.

    Absent packs simply yield nothing and the build degrades to the old tarkov.dev-only behaviour,
    so this never becomes a hard dependency on having built a map.
    """
    import glob
    out = {}
    roots = [os.path.join(REPO, 'packs', '*.eftpack', 'gamedata.json')]
    if _TK:
        roots.append(os.path.join(_TK, 'out', '*', 'gamedata.json'))
    for pat in roots:
        for p in glob.glob(pat):
            mid = os.path.basename(os.path.dirname(p)).replace('.eftpack', '')
            if mid in out:
                continue
            try:
                gd = json.load(open(p, encoding='utf-8'))
            except Exception:
                continue
            zs = []
            for q in (gd.get('quest_triggers') or []):
                if not q.get('pos'):
                    continue
                zs.append({'pos': q['pos'], 'outline': q.get('outline') or [],
                           'name': q.get('name'), 'kind': q.get('kind'),
                           'active': q.get('active', True)})
            if zs:
                # A trigger name can repeat within a map, and when it does the boxes are PARTS OF
                # ONE LOGICAL ZONE, not duplicates: woods ships 8 'kill_in_forest_woods' boxes and
                # factory 4 'nf2024_4_zone_kill1'. The objective is satisfied in any of them, so the
                # index maps name -> LIST and the join adopts every part. Keeping only the first
                # would draw one box and hide the rest of the objective's real footprint.
                by_name = {}
                for z in zs:
                    if z['name']:
                        by_name.setdefault(z['name'], []).append(z)
                out[mid] = {'by_name': by_name, 'all': zs}
    return out


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _from_game(z, best, join, d=None, part=None):
    """A task zone rebuilt on a game trigger's geometry, recording HOW it was matched."""
    # top/bottom describe a vertical band measured at UPSTREAM's position. When the game puts the
    # zone somewhere else the band no longer belongs to it — carrying it over left
    # interchange/place_WARBLOOD_04_2 with a ceiling 2.97 m BELOW its own floor after a 190.3 m
    # correction. Keep the band only while the position it was measured at still stands.
    band_ok = d is None or d <= MATCH_R
    out = {'map': z['map'], 'zid': z.get('zid'), 'pos': best['pos'],
           'outline': best['outline'] or (z.get('outline') if join == 'pos' else None),
           'top': z.get('top') if band_ok else None,
           'bottom': z.get('bottom') if band_ok else None,
           'src': 'game', 'join': join, 'game': best['name']}
    if best.get('kind'):
        out['gkind'] = best['kind']
    if not best.get('active', True):
        out['inactive'] = True
    if d is not None:
        out['d'] = round(d, 1)
    if part is not None:
        out['part'] = part          # 1-based index within a multi-box zone
    return zclean(out)


def apply_first_party_zones(out_tasks, game):
    """Make the game authoritative for objective geometry; return the triggers CLAIMED.

    Rewrites each objective's zone list rather than patching in place, because one upstream zone
    can expand to SEVERAL game boxes (a repeated trigger id) and several upstream zones can
    collapse to ONE (day/night map aliases both resolve to the same map id, so tarkov.dev lists
    the identical trigger twice - factory + night-factory, ground-zero + ground-zero-21 - and the
    viewer used to draw a duplicate marker and a duplicate wall on top of itself for every such
    objective).

    Returns {map id: {trigger name}} so the caller can report which first-party quest zones no
    upstream task accounts for.
    """
    claimed = {m: set() for m in game}
    if not game:
        print('[tasks] no gamedata.json found - zones stay tarkov.dev-only')
        return claimed
    n_name = n_pos = n_dev = n_suspect = n_offmap = n_parts = n_dupe = 0
    by_map_offmap = {}          # map id -> zones we could not verify against the game
    for t in out_tasks:
        for o in t['objectives']:
            if not o.get('zones'):
                continue
            rebuilt, seen = [], set()
            for z in o['zones']:
                # Collapse the day/night + tutorial map aliases: the SAME trigger on the SAME
                # resolved map is one zone however many upstream rows name it. Keyed on the zone
                # ID ALONE when there is one — upstream lists a multi-box zone once PER BOX with
                # each box's own position, and since the name join adopts every box, keeping
                # position in the key would expand that zone N times over (factory's
                # 'nf2024_4_zone_kill1' has 4 boxes and 4 upstream rows: 16 markers for 4 boxes).
                # The map's gamedata must be resolved BEFORE the dedupe key is chosen. Collapsing
                # on the zone id alone is only sound because the name join re-expands the id to
                # every box the scene has; with no gamedata for this map there is nothing to
                # re-expand it, so the id-only key would silently delete real boxes. Measured
                # cost of getting this wrong: customs 'exit777' lost a box 556.8 m from the one
                # kept, and 'place_flyers1' lost 4 of its 5 plant spots.
                g = game.get(z.get('map') or '')
                key = ((z.get('map'), z['zid']) if z.get('zid') and g
                       else (z.get('map'), z.get('zid'),
                             None if not z.get('pos') else tuple(z['pos'])))
                if key in seen:
                    n_dupe += 1
                    continue
                seen.add(key)
                if not g:
                    n_offmap += 1          # a map we have not built; nothing to say about it
                    mid = z.get('map') or '(no map)'
                    by_map_offmap[mid] = by_map_offmap.get(mid, 0) + 1
                    rebuilt.append(z)
                    continue
                zid = z.get('zid')
                # 1. EXACT: upstream's zone id IS the game's trigger name. Adopt every box.
                if zid and zid in g['by_name']:
                    boxes = g['by_name'][zid]
                    multi = len(boxes) > 1
                    for i, best in enumerate(boxes):
                        d = _dist(best['pos'], z['pos']) if z.get('pos') else None
                        rebuilt.append(_from_game(z, best, 'name', d, i + 1 if multi else None))
                    claimed[z['map']].add(zid)
                    n_name += 1
                    n_parts += len(boxes) - 1
                    continue
                # 2. FALLBACK: no id (or an id this map's scene does not carry) - nearest trigger.
                if z.get('pos'):
                    best = min(g['all'], key=lambda c: _dist(c['pos'], z['pos']))
                    d = _dist(best['pos'], z['pos'])
                    if d <= MATCH_R:
                        rebuilt.append(_from_game(z, best, 'pos', d))
                        if best['name']:
                            claimed[z['map']].add(best['name'])
                        n_pos += 1
                        continue
                    z['d'] = round(d, 1)
                    if d > SUSPECT_R:
                        z['suspect'] = True
                        n_suspect += 1
                z['src'] = 'dev'
                n_dev += 1
                rebuilt.append(z)
            o['zones'] = rebuilt
    print(f'[tasks] first-party zones: {n_name} joined BY NAME (upstream zone id == game trigger '
          f'id, +{n_parts} extra box(es) from multi-part zones), {n_pos} by position fallback, '
          f'{n_dev} kept tarkov.dev geometry ({n_suspect} flagged suspect, >{SUSPECT_R:.0f} m from '
          f'any trigger); {n_dupe} duplicate map-alias zone(s) collapsed')
    # "N on maps not built here" was one quiet number at the end of a busy line, and it hid a
    # whole-catalog regression: a build run without EFT_TARKMAP_ROOT sees only packs/*.eftpack, so
    # 8 of 11 maps silently kept tarkov.dev geometry and first-party coverage fell 583 -> 136 with
    # no error. Name the maps and how many zones each one lost, so a starved build is impossible
    # to mistake for a clean one.
    if by_map_offmap:
        total = sum(by_map_offmap.values())
        print(f'[tasks] *** {total} zone(s) on {len(by_map_offmap)} map(s) have NO game data - '
              f'they keep tarkov.dev geometry, unverified ***')
        for mid, n in sorted(by_map_offmap.items(), key=lambda kv: -kv[1]):
            print(f'[tasks]     {mid:20s} {n:4d} zone(s)')
        print('[tasks]     set EFT_TARKMAP_ROOT (or build these maps into packs/) and re-run to '
              'give them first-party geometry')
    return claimed


# A trigger id ending in an index — "place_WARBLOOD_04_4", "Shootable_Duck_2 (3)".
_INDEXED = __import__('re').compile(r'^(.*?)[ _]*(?:\((\d+)\)|_(\d+))$')


def _stem(name):
    m = _INDEXED.match(name or '')
    return m.group(1) if m else None


def link_sibling_zones(out_tasks, game, claimed):
    """Attach an unclaimed trigger to a task when its INDEXED SIBLINGS all belong to that task.

    "place_WARBLOOD_04_4" is a real, positioned, shipped zone for The Blood of War - Part 1: the
    game has it, tarkov.dev does not, and _1/_2/_3 are all joined to that one task. So the fourth
    is derivable rather than guessable, and this rule retires the hand-authored entry
    tasks_zone_patch.json carried for exactly this objective (whose position, derived by matching
    the cistern mesh, turns out to be 0.4 m from the game's own trigger - right, but no longer
    something a human has to work out).

    The rule is deliberately narrow, because a loose prefix match is not derivation: the stem must
    be the full name minus a trailing numeric index, and every claimed sibling sharing that stem
    must belong to the SAME single task. Across the built maps it fires exactly once. A wider rule
    was measured and rejected - matching on any '_'-separated prefix resolved 3 zones and one of
    them ('event' -> "Needle in a Haystack") was plainly wrong.
    """
    if not game:
        return 0
    # stem -> {task name: (task, objective)} for zones already joined to a game trigger
    stems = {}
    for t in out_tasks:
        for o in t['objectives']:
            for z in (o.get('zones') or []):
                if z.get('src') != 'game' or not z.get('game'):
                    continue
                s = _stem(z['game'])
                if s:
                    stems.setdefault((z['map'], s), {}).setdefault(t['name'], (t, o))
    n = 0
    for mid, g in game.items():
        for name, boxes in g['by_name'].items():
            if name in claimed.get(mid, ()):
                continue
            s = _stem(name)
            owners = stems.get((mid, s)) if s else None
            if not owners or len(owners) != 1:
                continue
            t, o = next(iter(owners.values()))
            # The stem tells us the TASK; it does not say which objective. A task whose siblings
            # are one-per-objective ("mark the first/second/third fuel tank") leaves exactly one
            # objective with no zone on this map, and that is where the unclaimed one belongs —
            # otherwise the fourth tank's marker would hang off the first tank's objective. When
            # that is ambiguous (zero or several zoneless objectives) fall back to the stem owner.
            bare = [x for x in t['objectives']
                    if not any(zz.get('map') == mid for zz in (x.get('zones') or []))]
            if len(bare) == 1:
                o = bare[0]
            # A sibling family can share ONE physical spot: customs ships Q019_1/_2/_3 at the
            # identical position, a trigger per item stashed in the same bin. Appending a marker
            # where this objective already has one adds a second pin on the same pixel and nothing
            # else, so drop the geometry but still count the trigger as accounted for — it belongs
            # to this task either way, and listing it as "no task claims this" would be wrong.
            here = [tuple(zz['pos']) for zz in (o.get('zones') or [])
                    if zz.get('map') == mid and zz.get('pos')]
            fresh = [b for b in boxes
                     if not any(_dist(b['pos'], p) < 0.5 for p in here)]
            multi = len(fresh) > 1
            for i, best in enumerate(fresh):
                o.setdefault('zones', []).append(
                    _from_game({'map': mid, 'zid': name, 'top': None, 'bottom': None},
                               best, 'sibling', part=i + 1 if multi else None))
            claimed[mid].add(name)
            if fresh and mid not in t['maps']:
                t['maps'].append(mid)
            n += 1
            note = '' if fresh else '  (co-located with an existing zone - no marker added)'
            print(f"[tasks]   sibling link: {mid}/{name} -> {t['name']}{note}")
    if n:
        print(f'[tasks] {n} unclaimed trigger(s) linked to a task by indexed sibling')
    return n


# Seasonal/event and achievement trigger families. These are NAME-DERIVED presentation hints only
# - never geometry, never a task link. They exist because ~40% of the unclaimed first-party zones
# are holiday-event content that upstream has no task row for at all, and lumping those in with
# real unlinked quest zones makes the list unreadable. Ordered: first match wins.
ZONE_FAMILIES = (
    ('event', ('ny25_', 'ny24_', 'new_year_', 'q_ny_', 'nosquests_', 'halloween_', 'nf2024_',
               'event_', 'shootable_duck', 'flarefirecollector', 'rshg_event')),
    ('achievement', ('achiv_',)),
    ('transit', ('mark_transits_', '_place_transit', 'transition_from_')),
    ('extract', ('exit', 'caseextr', 'case_extraction', 'secret_extraction')),
)


def zone_family(name):
    n = (name or '').lower()
    for fam, pats in ZONE_FAMILIES:
        if any(p in n for p in pats):
            return fam
    return 'quest'


def unlinked_game_zones(game, claimed):
    """First-party quest triggers that NO upstream task accounts for, per map.

    These are real, shipped, positioned quest geometry - the game knows about them and
    tarkov.dev does not. They are emitted so the viewer can show them as what they are
    (first-party zones of unknown task) rather than silently dropping them.
    """
    out = {}
    for mid, g in game.items():
        rows = []
        # Grouped by trigger id for the same reason the join expands it: repeated boxes are ONE
        # zone with several parts, and a list that repeats 'kill_in_forest_woods' eight times
        # reads as eight findings when it is one.
        for name, boxes in g['by_name'].items():
            if name in claimed.get(mid, ()):
                continue
            rows.append({'name': name, 'kind': boxes[0].get('kind'), 'pos': boxes[0]['pos'],
                         'family': zone_family(name),
                         **({'parts': len(boxes)} if len(boxes) > 1 else {}),
                         **({'outline': boxes[0]['outline']} if boxes[0]['outline'] else {}),
                         **({} if boxes[0].get('active', True) else {'inactive': True})})
        if rows:
            rows.sort(key=lambda r: (r['family'], r['name']))
            out[mid] = rows
    total = sum(len(v) for v in out.values())
    if total:
        fam = {}
        for rows in out.values():
            for r in rows:
                fam[r['family']] = fam.get(r['family'], 0) + 1
        print(f'[tasks] {total} first-party quest zone(s) no upstream task claims, '
              f'across {len(out)} map(s): {dict(sorted(fam.items(), key=lambda kv: -kv[1]))}')
    return out


def item_ref(i):
    """Small, stable item shape used by rewards/keys without shipping the full API object."""
    return {'n': i.get('name'), 's': i.get('shortName'), 'pr': i.get('avg24hPrice')}


def flat_items(value):
    """Yield Item objects from API fields that may be Item[], Item[][], or null."""
    for x in value or []:
        if isinstance(x, list):
            yield from flat_items(x)
        elif isinstance(x, dict):
            yield x


def conv_rewards(r):
    r = r or {}
    out = {}
    items = []
    for x in r.get('items') or []:
        if not x.get('item'):
            continue
        v = item_ref(x['item'])
        v['count'] = x.get('count') if x.get('count') is not None else x.get('quantity')
        items.append(v)
    if items:
        out['items'] = items
    standing = [{'trader': x['trader']['name'], 'value': x.get('standing')}
                for x in (r.get('traderStanding') or []) if x.get('trader')]
    if standing:
        out['standing'] = standing
    offers = [{'trader': x['trader']['name'], 'level': x.get('level'),
               'item': (x.get('item') or {}).get('name')}
              for x in (r.get('offerUnlock') or []) if x.get('trader')]
    if offers:
        out['offers'] = offers
    skills = [{'name': x.get('name'), 'level': x.get('level')}
              for x in (r.get('skillLevelReward') or [])]
    if skills:
        out['skills'] = skills
    for src, dst, key in (('traderUnlock', 'traders', 'name'),
                          ('achievement', 'achievements', 'name'),
                          ('customization', 'customization', 'name')):
        vals = [x.get(key) for x in (r.get(src) or []) if x.get(key)]
        if vals:
            out[dst] = vals
    return out


def main():
    print("[tarkov.dev/json] building tasks...")
    # The bundled embeddable Python pins sys.path via python311._pth and does not add this directory.
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import tarkov_static
    data = tarkov_static.load_static_tasks()
    source = 'tarkov.dev/json'
    tasks_in = data['tasks']
    out_tasks = []
    map_task_count = {}
    for t in tasks_in:
        objs = []
        task_maps = set()
        if t.get('map'): task_maps.add(map_id(t['map']['normalizedName']))
        for o in t['objectives'] or []:
            zones = [conv_zone(z) for z in (o.get('zones') or [])]
            for z in zones:
                if z['map']: task_maps.add(z['map'])
            for m in (o.get('maps') or []):
                task_maps.add(map_id(m['normalizedName']))
            oo = {'id': o['id'], 'type': o['type'], 'desc': o['description'], 'optional': o.get('optional', False)}
            if zones: oo['zones'] = zones
            # type-specific "what to do" detail
            if o.get('items'): oo['items'] = [i['name'] for i in o['items']]
            if o.get('markerItem'): oo['markerItem'] = o['markerItem']['name']
            if o.get('questItem'): oo['questItem'] = o['questItem']['name']
            if o.get('targetNames'): oo['targets'] = o['targetNames']
            if o.get('count'): oo['count'] = o['count']
            if o.get('foundInRaid'): oo['fir'] = True
            if o.get('exitName'): oo['exit'] = o['exitName']
            if o.get('requiredKeys'):
                # Upstream currently returns a flat list, but tolerate nested alternative-key groups.
                raw_keys = o['requiredKeys']
                if raw_keys and isinstance(raw_keys[0], list):
                    oo['requiredKeys'] = [[item_ref(k) for k in flat_items(group)] for group in raw_keys]
                else:
                    oo['requiredKeys'] = [[item_ref(k) for k in flat_items(raw_keys)]]
            for src, dst in (('usingWeapon', 'weapons'), ('usingWeaponMods', 'weaponMods'),
                             ('wearing', 'wearing'), ('notWearing', 'notWearing'),
                             ('useAny', 'useAny')):
                vals = [i.get('name') for i in flat_items(o.get(src)) if i.get('name')]
                if vals: oo[dst] = vals
            if o.get('distance'):
                oo['distance'] = o['distance']
            if o.get('bodyParts'): oo['bodyParts'] = o['bodyParts']
            if o.get('shotType'): oo['shotType'] = o['shotType']
            if o.get('timeFromHour') or o.get('timeUntilHour'):
                oo['timeWindow'] = [o.get('timeFromHour') or 0, o.get('timeUntilHour') or 0]
            # 0-100% is the full range - the objective has no durability requirement at all, and
            # 551 of 1463 objectives carry it. Emitting it put a meaningless "durability 0-100%"
            # chip on a third of the list. Only a real bound survives.
            lo, hi = o.get('minDurability'), o.get('maxDurability')
            lo = None if lo is None else float(lo)
            hi = None if hi is None else float(hi)
            if not (lo in (None, 0) and hi in (None, 100)):
                if lo is not None: oo['minDurability'] = lo
                if hi is not None: oo['maxDurability'] = hi
            if o.get('possibleLocations'):
                oo['itemLocations'] = [{'map': map_id(pl['map']['normalizedName']),
                                        'pts': [bridge(p) for p in (pl['positions'] or [])]}
                                       for pl in o['possibleLocations'] if pl.get('map')]
                for pl in oo['itemLocations']:
                    if pl['map']: task_maps.add(pl['map'])
            objs.append(oo)
        out = {
            'id': t['id'], 'name': t['name'], 'norm': t['normalizedName'],
            'trader': t['trader']['name'] if t.get('trader') else None,
            'map': map_id(t['map']['normalizedName']) if t.get('map') else None,
            'minLevel': t.get('minPlayerLevel') or 0, 'kappa': bool(t.get('kappaRequired')),
            'lk': bool(t.get('lightkeeperRequired')), 'wiki': t.get('wikiLink'),
            'image': t.get('taskImageLink'), 'xp': t.get('experience') or 0,
            # "Any" is the default on 489 of 501 tasks - a faction chip that says nothing. Only a
            # real BEAR/USEC restriction is worth carrying.
            'faction': (t.get('factionName') if t.get('factionName') not in (None, '', 'Any') else None),
            'restartable': bool(t.get('restartable')),
            'delayMin': t.get('availableDelaySecondsMin') or 0,
            'delayMax': t.get('availableDelaySecondsMax') or 0,
            'requires': [r['task']['name'] for r in (t.get('taskRequirements') or []) if r.get('task')],
            'traderReqs': [{'trader': r['trader']['name'], 'type': r.get('requirementType'),
                            'compare': r.get('compareMethod'), 'value': r.get('value')}
                           for r in (t.get('traderRequirements') or []) if r.get('trader')],
            'rewards': conv_rewards(t.get('finishRewards')),
            'maps': sorted(m for m in task_maps if m),
            'objectives': objs,
        }
        out_tasks.append(out)
        for m in out['maps']: map_task_count[m] = map_task_count.get(m, 0) + 1

    # FIRST-PARTY GEOMETRY: run BEFORE the supplemental patch so the patch only ever fills gaps the
    # game itself cannot fill. Most of what that file was written for is now answered upstream of it.
    game_zones = _load_game_zones()
    claimed = apply_first_party_zones(out_tasks, game_zones)
    link_sibling_zones(out_tasks, game_zones, claimed)
    unlinked = unlinked_game_zones(game_zones, claimed)

    # SUPPLEMENTAL ZONES (tasks_zone_patch.json): the escape hatch for a gap NEITHER source can
    # fill, applied by (task name, desc substring). It is EMPTY, and should stay that way — its one
    # entry (The Blood of War - Part 1's fourth fuel tank, a position a human derived by matching
    # the cistern mesh against the trailer cluster) is now supplied by the game itself via
    # link_sibling_zones, which lands 0.4 m from that hand-derived point. The mechanism is kept
    # because a genuinely underivable gap could reappear; every entry must carry a `derivation`.
    # Zones added here are marked src='patch' so the viewer never presents them as first-party.
    _patch_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks_zone_patch.json')
    if os.path.exists(_patch_p):
        applied = 0
        for pe in json.load(open(_patch_p, encoding='utf-8')):
            for t in out_tasks:
                if t['name'] != pe['task']: continue
                for o in t['objectives']:
                    if pe['desc_contains'].lower() in (o.get('desc') or '').lower():
                        o.setdefault('zones', []).append(zclean(dict(pe['zone'], src='patch'))); applied += 1
                        if pe['zone'].get('map') and pe['zone']['map'] not in t['maps']:
                            t['maps'].append(pe['zone']['map'])
        if applied:
            print(f"[tasks] zone patch: {applied} supplemental zone(s) applied")

    # Per-map OBJECTIVE counts (how many located objectives a map actually carries) alongside the
    # task counts, so the viewer's map filter can say what switching to "all maps" would show
    # without walking the whole 1.3 MB catalog every frame.
    map_obj_count = {}
    for t in out_tasks:
        for o in t['objectives']:
            for z in (o.get('zones') or []):
                if z.get('map'):
                    map_obj_count[z['map']] = map_obj_count.get(z['map'], 0) + 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = {'version': 4, 'source': source, 'built': int(time.time()),
           'coord_bridge': 'viewer = diag(-1,1,1) * unity', 'count': len(out_tasks),
           'map_task_count': map_task_count, 'map_obj_count': map_obj_count,
           # First-party quest geometry no upstream task claims — real zones the game ships and
           # tarkov.dev has no row for. Keyed by map id.
           'unlinkedZones': unlinked,
           'tasks': out_tasks}
    json.dump(doc, open(OUT, 'w'), separators=(',', ':'))
    zoned = sum(1 for t in out_tasks if any('zones' in o for o in t['objectives']))
    print(f"[tasks] {len(out_tasks)} tasks -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB); {zoned} have map zones")
    print(f"[tasks] per map: {dict(sorted(map_task_count.items(), key=lambda kv: -kv[1]))}")


if __name__ == '__main__':
    main()
