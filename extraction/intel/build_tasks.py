#!/usr/bin/env python
"""Build the TASK catalog for the tarkmap task-tracker (out/tasks.json).

Pulls the full quest/task list from tarkov.dev's pre-generated JSON API, keeps the
fields the tracker needs (name/trader/map/level/kappa/prereqs + every objective with its description and, where it has a
map location, its zone position + outline), and COORDINATE-BRIDGES every position/outline into viewer-world space with the
SAME G3 = diag(-1,1,1) conjugation the geometry pipeline uses (viewer = (-x, y, z)). So a task zone drops straight onto the
tacmap. Tasks span all maps, so this is ONE global catalog the viewer filters by the current map — supports every map.

  python extraction/intel/build_tasks.py                 -> <EFT_TARKMAP_ROOT>/out/tasks.json  (all tasks, all maps)
Re-run per wipe (task data changes per wipe, not per session). No game files needed (tarkov.dev only)."""
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


def conv_zone(z):
    zid = map_id(z['map']['normalizedName']) if z.get('map') else None
    return {'map': zid, 'pos': bridge(z.get('position')),
            'outline': [bridge(p) for p in z['outline']] if z.get('outline') else None,
            'top': z.get('top'), 'bottom': z.get('bottom')}


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
            if o.get('minDurability') is not None: oo['minDurability'] = o['minDurability']
            if o.get('maxDurability') is not None: oo['maxDurability'] = o['maxDurability']
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
            'faction': t.get('factionName'), 'restartable': bool(t.get('restartable')),
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

    # SUPPLEMENTAL ZONES (tasks_zone_patch.json): upstream data gaps, positions derived from OUR map
    # geometry (each entry documents its derivation). Applied by (task name, desc substring) so a
    # rebuilt/refreshed upstream keeps the patch until tarkov.dev fills the gap (then it double-zones,
    # which the tracker renders fine).
    _patch_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks_zone_patch.json')
    if os.path.exists(_patch_p):
        applied = 0
        for pe in json.load(open(_patch_p, encoding='utf-8')):
            for t in out_tasks:
                if t['name'] != pe['task']: continue
                for o in t['objectives']:
                    if pe['desc_contains'].lower() in (o.get('desc') or '').lower():
                        o.setdefault('zones', []).append(pe['zone']); applied += 1
                        if pe['zone'].get('map') and pe['zone']['map'] not in t['maps']:
                            t['maps'].append(pe['zone']['map'])
        print(f"[tasks] zone patch: {applied} supplemental zone(s) applied")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = {'version': 2, 'source': source, 'built': int(time.time()),
           'coord_bridge': 'viewer = diag(-1,1,1) * unity', 'count': len(out_tasks),
           'map_task_count': map_task_count, 'tasks': out_tasks}
    json.dump(doc, open(OUT, 'w'), separators=(',', ':'))
    zoned = sum(1 for t in out_tasks if any('zones' in o for o in t['objectives']))
    print(f"[tasks] {len(out_tasks)} tasks -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB); {zoned} have map zones")
    print(f"[tasks] per map: {dict(sorted(map_task_count.items(), key=lambda kv: -kv[1]))}")


if __name__ == '__main__':
    main()
