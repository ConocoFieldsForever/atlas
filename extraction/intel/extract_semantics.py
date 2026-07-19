#!/usr/bin/env python
"""Extract the DISPLAYABLE semantic + lighting layers from an EFT Unity map -> out/<map>/semantics.json,
consumed by the Carbon tactical-map overlay. MAP-AGNOSTIC: the Unity->glb transform is the map's
coordinates.global_matrix conjugation, reduced to a point/vector (G3 @ p) -- identical rule as the geometry
pipeline (tarkov-unity-extraction skill SS1/SS3/SS9).

Semantics live in GameObject NAMES + transforms + colliders, NOT MonoBehaviour fields. Gameplay VALUES
(loot tables, spawn %, keycards, quests) are external (Windows.json) and NOT extracted here.

  python extraction/intel/extract_semantics.py [map] [level,level,...]
      (requires EFT_TARKMAP_ROOT; levels default to the map config's source.levels)
"""
import os, sys, json, re, math, functools, time, glob
import numpy as np
print = functools.partial(print, flush=True)
import UnityPy

# portable kit: paths come from the environment (see extraction/README.md)
HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/extraction/intel
KIT = os.path.dirname(HERE)                                # <repo>/extraction
DATA = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
TK = os.environ.get("EFT_TARKMAP_ROOT")                    # <workspace>/tarkmap (maps/ + out/)
if not TK:
    raise SystemExit("extract_semantics: EFT_TARKMAP_ROOT is not set. Point it at your workspace tarkmap "
                     "dir (the one holding maps/ and out/), e.g.  setx EFT_TARKMAP_ROOT D:\\eft_work\\tarkmap")
MAP = sys.argv[1] if len(sys.argv) > 1 else "interchange"
# map config: workspace copy first, kit copy as fallback (they are the same files)
_cfg_p = os.path.join(TK, 'maps', MAP, 'config.json')
if not os.path.exists(_cfg_p):
    _cfg_p = os.path.join(KIT, 'maps', MAP, 'config.json')
_cfg = json.load(open(_cfg_p, encoding='utf-8'))
LEVELS = ([int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2
          else [int(x) for x in (_cfg['source'].get('levels') or [52, 54, 55, 56, 57, 58, 59, 60, 62])])
# dataset root (for the lights sidecar): config source.root resolves against the workspace root,
# exactly like eft_pipeline/tarkmap_core/config.py does.
_ds = _cfg['source']['root']
DS_ROOT = _ds if os.path.isabs(_ds) else os.path.normpath(os.path.join(TK, '..', _ds))
_lj = sorted(p for p in glob.glob(os.path.join(DS_ROOT, 'lights_*.json')) if not p.endswith('_all.json'))
LIGHTS_JSON = _lj[0] if _lj else os.path.join(DS_ROOT, 'lights_none.json')
OUT = os.path.join(TK, 'out', MAP, 'semantics.json')

# category -> regex (FIRST match wins; order = priority). Heuristic, name-driven (see skill SS9).
CATS = [
    ("extract", re.compile(r"(exfil|extract|Saferoom|Gates_Rollets|Terminal_Entrance|Fire_Exit|Road_Gate|Rollete?_Gate|EXIT_|ZoneRoad)", re.I)),
    ("loot",    re.compile(r"(lootable|LootPoint|_showcase|GunsafeSpawn|Weapon_box|Weapon_crate|scontainer|Cashbox|cash_register|jacket|_drawer|safe_\d|_wallet|medbag|toolbox|ammo_box)", re.I)),
    ("spawn",   re.compile(r"(SpawnPoint|BotZone|PlayerSpawn|ScavSpawn|Triger_.*Out)", re.I)),
    ("door",    re.compile(r"(Inside_Door|Door_Metal|Door_Wood|Keycard|_Door_R|_Door_L|LockBox|padlock)", re.I)),
    ("zone",    re.compile(r"(Floor[123]_|_Corridor|_Hall\b|_Office\b|_Room\b|_Stairs\b|Parking_Zone|Basement|Atrium)", re.I)),
]
# things that are never POIs even if a regex grazes them
EXCLUDE = re.compile(r"(_COLLIDER|_LOD[1-9]|_SHADOW|decal|Particle|VFX|_proxy)", re.I)
FLOOR_RE = re.compile(r"Floor\s?([0-3])", re.I)


def get_G3(map_id):
    # global_matrix is a constant X-flip and is no longer stored per-config; default to it when
    # absent (config already loaded above, workspace/kit maps dir).
    gm = (_cfg.get('coordinates') or {}).get('global_matrix')
    if not gm:
        return np.diag([-1.0, 1.0, 1.0, 1.0])[:3, :3]
    return np.array(gm, np.float64).reshape(4, 4)[:3, :3]


def trs(t, q, s):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)
    M = np.eye(4); M[:3, :3] = R * np.array(s, np.float64); M[:3, 3] = t
    return M


def vec(d, k, default=(0, 0, 0)):
    v = d.get(k) or {}
    return [v.get('x', default[0]), v.get('y', default[1]), v.get('z', default[2])]


def quat(d, k):
    v = d.get(k) or {}
    return [v.get('x', 0), v.get('y', 0), v.get('z', 0), v.get('w', 1)]


def classify(name):
    if EXCLUDE.search(name):
        return None
    for cat, rx in CATS:
        if rx.search(name):
            return cat
    return None


def main():
    G3 = get_G3(MAP)
    print(f"[cfg] map={MAP} levels={LEVELS} G3={G3.round(2).tolist()}")
    layers = {c: [] for c, _ in CATS}
    seen = set()
    t_all = time.time()
    for lv in LEVELS:
        p = os.path.join(DATA, f"level{lv}")
        if not os.path.exists(p):
            continue
        t0 = time.time()
        env = UnityPy.load(p)
        # index transforms + gameobjects by path_id for father-chain + name resolution
        tr_by_id = {}   # path_id -> transform typetree (lazy)
        go_obj = {}     # path_id -> gameobject obj
        tr_obj = {}
        mr_ids = set()  # MeshFilter/MeshRenderer path_ids -> tells "this GO is a PROP, not a grouping zone"
        for o in env.objects:
            tn = o.type.name
            if tn in ('Transform', 'RectTransform'):
                tr_obj[o.path_id] = o
            elif tn == 'GameObject':
                go_obj[o.path_id] = o
            elif tn in ('MeshFilter', 'MeshRenderer'):
                mr_ids.add(o.path_id)

        def tr_tt(pid):
            if pid not in tr_by_id:
                try:
                    tr_by_id[pid] = tr_obj[pid].read_typetree() if pid in tr_obj else None
                except Exception:
                    tr_by_id[pid] = None
            return tr_by_id[pid]

        def world_pos(tpid):
            chain = []; pid = tpid; guard = 0
            while pid and guard < 64:
                tt = tr_tt(pid)
                if not tt:
                    break
                chain.append(tt)
                f = tt.get('m_Father') or {}
                pid = f.get('m_PathID', 0); guard += 1
            M = np.eye(4)
            for tt in reversed(chain):
                M = M @ trs(vec(tt, 'm_LocalPosition'), quat(tt, 'm_LocalRotation'), vec(tt, 'm_LocalScale', (1, 1, 1)))
            return M[:3, 3]

        nmatch = 0
        for gpid, go in go_obj.items():
            try:
                gd = go.read_typetree()
            except Exception:
                continue
            name = gd.get('m_Name', '')
            cat = classify(name)
            if not cat:
                continue
            # find its Transform component + detect whether it owns a mesh (=> a prop, not a grouping zone)
            tpid = None; has_mesh = False
            for c in gd.get('m_Component', []):
                pp = (c.get('component') if isinstance(c, dict) else None) or c
                cid = pp.get('m_PathID') if isinstance(pp, dict) else None
                if cid is None:
                    continue
                if cid in tr_obj and tpid is None:
                    tpid = cid
                elif cid in mr_ids:
                    has_mesh = True
            if not tpid:
                continue
            if cat == 'zone' and has_mesh:
                continue                                  # a prop named after an area, not the area group itself
            wp = world_pos(tpid)
            gp = (G3 @ wp).tolist()                       # Unity world -> glb world (X-flip conjugation)
            key = (cat, round(gp[0], 1), round(gp[1], 1), round(gp[2], 1))
            if key in seen:
                continue
            seen.add(key)
            fm = FLOOR_RE.search(name)
            layers[cat].append({
                "name": name, "p": [round(v, 2) for v in gp],
                "floor": int(fm.group(1)) if fm else None, "lv": lv,
            })
            nmatch += 1
        print(f"[level{lv}] {len(go_obj):,} GameObjects -> {nmatch} POIs  ({time.time()-t0:.0f}s)")

    # lights -> their own layer (already in lights_level64.json; X-flip into glb space)
    lights = []
    try:
        L = json.load(open(LIGHTS_JSON))
        for l in L:
            if (l.get('intensity', 0) or 0) <= 0:
                continue
            gp = (G3 @ np.array(l['position'], np.float64)).tolist()
            c = (l.get('color') or [1, 1, 1, 1])[:3]
            lights.append({"p": [round(v, 2) for v in gp], "type": l.get('type', 'Point'),
                           "color": [round(v, 3) for v in c], "intensity": round(float(l.get('intensity', 0)), 2),
                           "range": round(float(l.get('range', 0) or 0), 1)})
    except Exception as e:
        print(f"[lights] {e}")

    out = {
        "map": MAP,
        "generated_levels": LEVELS,
        "counts": {**{c: len(layers[c]) for c, _ in CATS}, "lights": len(lights)},
        "layers": {**layers, "lights": lights},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), separators=(',', ':'))
    print(f"\n[out] {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB, {time.time()-t_all:.0f}s)")
    print("[counts]", json.dumps(out["counts"]))
    # quick sanity: do POI positions land in the glb bbox?
    allp = np.array([q["p"] for c in layers for q in layers[c]] + [l["p"] for l in lights]) if (any(layers.values()) or lights) else np.zeros((1, 3))
    print(f"[sanity] POI+light bbox min={allp.min(0).round(0).tolist()} max={allp.max(0).round(0).tolist()}  (glb grid ~ [-718,14,-457]..[502,56,463])")
    for c, _ in CATS:
        if layers[c]:
            print(f"  {c:8s} e.g. {[q['name'] for q in layers[c][:4]]}")


if __name__ == "__main__":
    main()
