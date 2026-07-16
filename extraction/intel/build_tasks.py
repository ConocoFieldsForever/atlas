#!/usr/bin/env python
"""Build the TASK catalog for the tarkmap task-tracker (out/tasks.json).

Pulls the full quest/task list from the public tarkov.dev GraphQL API (needs a User-Agent header or it 403s), keeps the
fields the tracker needs (name/trader/map/level/kappa/prereqs + every objective with its description and, where it has a
map location, its zone position + outline), and COORDINATE-BRIDGES every position/outline into viewer-world space with the
SAME G3 = diag(-1,1,1) conjugation the geometry pipeline uses (viewer = (-x, y, z)). So a task zone drops straight onto the
tacmap. Tasks span all maps, so this is ONE global catalog the viewer filters by the current map — supports every map.

  python extraction/intel/build_tasks.py                 -> <EFT_TARKMAP_ROOT>/out/tasks.json  (all tasks, all maps)
Re-run per wipe (task data changes per wipe, not per session). No game files needed (tarkov.dev only)."""
import os, json, urllib.request, time

HERE = os.path.dirname(os.path.abspath(__file__))
# portable kit: output goes to the workspace tarkmap/out so assemble_bevy.py auto-ships it into packs.
_TK = os.environ.get("EFT_TARKMAP_ROOT")
if not _TK:
    raise SystemExit("build_tasks: EFT_TARKMAP_ROOT is not set. Point it at your workspace tarkmap dir "
                     "(the one holding maps/ and out/), e.g.  setx EFT_TARKMAP_ROOT D:\\eft_work\\tarkmap")
OUT = os.path.join(_TK, 'out', 'tasks.json')
API = "https://api.tarkov.dev/graphql"
# tarkov.dev map normalizedName -> our map id (matches tarkmap/maps/<id>). Extend as maps are added.
DEV_TO_ID = {
    'interchange': 'interchange', 'ground-zero': 'ground_zero', 'ground-zero-21': 'ground_zero',
    'factory': 'factory', 'night-factory': 'factory', 'woods': 'woods', 'customs': 'customs',
    'shoreline': 'shoreline', 'streets-of-tarkov': 'streets', 'reserve': 'reserve',
    'the-lab': 'labs', 'the-labs': 'labs', 'lighthouse': 'lighthouse',
    'the-labyrinth': 'labyrinth', 'labyrinth': 'labyrinth',
}
G3 = (-1.0, 1.0, 1.0)   # Unity world -> viewer world (X-flip), read logically from coordinates.global_matrix

def bridge(p):
    return None if p is None else [round(G3[0] * p['x'], 2), round(G3[1] * p['y'], 2), round(G3[2] * p['z'], 2)]

QUERY = """
{ tasks {
    id name normalizedName kappaRequired lightkeeperRequired experience wikiLink minPlayerLevel
    trader { name } map { normalizedName }
    taskRequirements { task { name } status }
    objectives {
      id type description optional maps { normalizedName }
      ... on TaskObjectiveBasic { zones { map { normalizedName } position { x y z } outline { x y z } top bottom } }
      ... on TaskObjectiveExtract { exitStatus exitName zoneNames count }
      ... on TaskObjectiveItem { items { name } count foundInRaid zones { map { normalizedName } position { x y z } outline { x y z } top bottom } }
      ... on TaskObjectiveMark { markerItem { name } zones { map { normalizedName } position { x y z } outline { x y z } top bottom } }
      ... on TaskObjectiveQuestItem { questItem { name } count possibleLocations { map { normalizedName } positions { x y z } } zones { map { normalizedName } position { x y z } outline { x y z } top bottom } }
      ... on TaskObjectiveShoot { targetNames count shotType bodyParts distance { compareMethod value } zones { map { normalizedName } position { x y z } outline { x y z } top bottom } }
      ... on TaskObjectiveUseItem { count zoneNames zones { map { normalizedName } position { x y z } outline { x y z } top bottom } }
      ... on TaskObjectiveBuildItem { item { name } }
      ... on TaskObjectivePlayerLevel { playerLevel }
      ... on TaskObjectiveTraderLevel { trader { name } level }
      ... on TaskObjectiveTraderStanding { trader { name } value }
      ... on TaskObjectiveSkill { skillLevel { name level } }
      ... on TaskObjectiveTaskStatus { task { name } status }
      ... on TaskObjectiveExperience { count }
    }
} }
"""


def gql(q):
    req = urllib.request.Request(API, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "tarkmap/1.0"})
    r = json.load(urllib.request.urlopen(req, timeout=90))
    if 'errors' in r: raise SystemExit("tarkov.dev errors: " + json.dumps(r['errors'][:3]))
    return r['data']


def map_id(nn):
    return DEV_TO_ID.get(nn, nn)


def conv_zone(z):
    zid = map_id(z['map']['normalizedName']) if z.get('map') else None
    return {'map': zid, 'pos': bridge(z.get('position')),
            'outline': [bridge(p) for p in z['outline']] if z.get('outline') else None,
            'top': z.get('top'), 'bottom': z.get('bottom')}


def main():
    print("[tarkov.dev] fetching tasks...")
    data = gql(QUERY)
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
            'requires': [r['task']['name'] for r in (t.get('taskRequirements') or []) if r.get('task')],
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
    doc = {'version': 1, 'source': 'tarkov.dev', 'built': int(time.time()),
           'coord_bridge': 'viewer = diag(-1,1,1) * unity', 'count': len(out_tasks),
           'map_task_count': map_task_count, 'tasks': out_tasks}
    json.dump(doc, open(OUT, 'w'), separators=(',', ':'))
    zoned = sum(1 for t in out_tasks if any('zones' in o for o in t['objectives']))
    print(f"[tasks] {len(out_tasks)} tasks -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB); {zoned} have map zones")
    print(f"[tasks] per map: {dict(sorted(map_task_count.items(), key=lambda kv: -kv[1]))}")


if __name__ == '__main__':
    main()
