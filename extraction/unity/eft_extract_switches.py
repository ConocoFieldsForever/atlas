"""Extract a map's POWER SWITCHES and the light banks they control, purely from typed Unity
components -- no map/mesh name rules, so it works identically on every map.

Mechanism (verified byte-identical on Reserve level518 and Interchange level520):
  A GameObject's switch mesh carries a MonoBehaviour whose m_Script resolves (via the MonoScript's
  readable m_ClassName) to EFT.Interactive.Switch. The real power lever serializes a trailing
  `count K` + K x PPtr{FileID=0, PathID} array; when EVERY element resolves to
  EFT.Interactive.LampController, that array IS the exact bank the switch powers. Every other Switch
  (alarm/keycard/exfil-node/gate) has no such all-LampController array, so this cleanly selects the
  power lever with zero name matching. Each LampController owns/parents the Unity Light components,
  so the controlled Lights are the ones under those GameObjects.

  python extraction/unity/eft_extract_switches.py --levels 520 --name interchange_v2
  python extraction/unity/eft_extract_switches.py --levels 518 --name reserve

Writes <dataset>/switches_<level>.json: a flat array of
  {id, level, switch_go, world_pos:[x,y,z], label, count, controlled_lamp_gos:[...],
   controlled_light_gos:[...]}
The light group tag is applied in eft_extract_lights.py by joining a Light to a switch when the
Light's owner-or-ancestor GameObject is in controlled_lamp_gos.
"""
import os, sys, json, struct, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eft_scene_extract import trs

EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else
    os.path.join(os.getcwd(), "eft_assets"))

SWITCH_CLASS = "EFT.Interactive.Switch"
LAMP_CLASS = "EFT.Interactive.LampController"


def load_level(level):
    import UnityPy
    paths = [os.path.join(EFTDATA, f"level{level}")]
    ggm = os.path.join(EFTDATA, "globalgamemanagers.assets")
    if os.path.isfile(ggm):
        paths.append(ggm)
    return UnityPy.load(*paths)


def build_maps(env):
    objs = list(env.objects)
    tfm = {o.path_id: o for o in objs if o.type.name == "Transform"}
    gos = {o.path_id: o for o in objs if o.type.name == "GameObject"}
    monos = {o.path_id: o for o in objs if o.type.name == "MonoBehaviour"}
    # MonoScript path_id -> "Namespace.Class"
    sc = {}
    for o in objs:
        if o.type.name == "MonoScript":
            try:
                d = o.read_typetree()
                ns = d.get("m_Namespace", "") or ""
                cn = d.get("m_ClassName", "") or ""
                sc[o.path_id] = f"{ns}.{cn}" if ns else cn
            except Exception:
                pass
    return objs, tfm, gos, monos, sc


def mono_class(mb, sc):
    try:
        r = mb.read(check_read=False)          # base fields; IL2CPP body is unreadable
    except Exception:
        try:
            r = mb.read()
        except Exception:
            return None
    sp = getattr(r, "m_Script", None)
    if sp is None:
        return None
    return sc.get(getattr(sp, "m_PathID", None) or getattr(sp, "path_id", None))


def mono_go(mb):
    try:
        return mb.read(check_read=False).m_GameObject.path_id
    except Exception:
        try:
            return mb.read().m_GameObject.path_id
        except Exception:
            return None


def decode_lamp_array(raw, monos, sc):
    """Scan for `count K` + K PPtr{FileID=0(int32), PathID(int64)} where every PathID resolves to a
    LampController MonoBehaviour in this file. Return the largest such [pathids] or None."""
    if not raw:
        return None
    best = None
    n = len(raw)
    off = 32
    while off + 4 <= n:
        K = struct.unpack_from("<i", raw, off)[0]
        if 1 <= K <= 400 and off + 4 + K * 12 <= n:
            pids = []
            ok = True
            for i in range(K):
                fid, pid = struct.unpack_from("<iq", raw, off + 4 + i * 12)
                if fid != 0 or pid not in monos:
                    ok = False
                    break
                pids.append(pid)
            if ok and all(mono_class(monos[p], sc) == LAMP_CLASS for p in pids):
                if best is None or K > len(best):
                    best = pids
        off += 4
    return best


def world_pos(go_pid, gos, tfm):
    """World-space translation of a GameObject (walk the parent Transform chain)."""
    go2tf = {}
    # find the GO's transform
    try:
        for comp in gos[go_pid].read_typetree().get("m_Component", []):
            cp = comp.get("component") or comp.get("second") or {}
            if cp.get("m_PathID") in tfm:
                tp = cp.get("m_PathID")
                break
        else:
            return None
    except Exception:
        return None
    chain = []
    cur = tfm.get(tp)
    depth = 0
    while cur is not None and depth < 256:
        depth += 1
        t = cur.read()
        chain.append(t)
        fp = getattr(t, "m_Father", None)
        if fp is None or getattr(fp, "path_id", 0) == 0:
            break
        cur = tfm.get(fp.path_id)
    W = np.eye(4)
    for t in reversed(chain):
        W = W @ trs(t)
    p = W[:3, 3]
    return [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]


def find_power_switches(level, objs, tfm, gos, monos, sc):
    """Core (reusable): return the power-switch records in an already-loaded level. A power switch is
    an EFT.Interactive.Switch whose trailing PPtr array resolves ENTIRELY to LampController; that
    array is the exact controlled bank. Also resolves the Unity Lights under those lamp GOs so the
    light extractor can tag them. Callable from eft_extract_lights (shares its loaded env)."""
    # ancestor-walk helper (GO -> its transform, then up via m_Father)
    def go_tf(go_pid):
        try:
            for comp in gos[go_pid].read_typetree().get("m_Component", []):
                cp = comp.get("component") or comp.get("second") or {}
                if cp.get("m_PathID") in tfm:
                    return cp.get("m_PathID")
        except Exception:
            pass
        return None

    switches = []
    for pid, mb in monos.items():
        if mono_class(mb, sc) != SWITCH_CLASS:
            continue
        try:
            raw = mb.get_raw_data()
        except Exception:
            raw = None
        lamp_pids = decode_lamp_array(raw, monos, sc)
        if not lamp_pids:
            continue                                   # not a power lever (no all-LampController bank)
        sgo = mono_go(mb)
        lamp_set = {mono_go(monos[p]) for p in lamp_pids} - {None}
        # Lights whose owner-or-ancestor GO is one of the controlled lamp GOs
        light_gos = []
        for o in objs:
            if o.type.name != "Light":
                continue
            try:
                lg = o.read(check_read=False).m_GameObject.path_id
            except Exception:
                continue
            hit = lg in lamp_set
            tp = go_tf(lg)
            depth = 0
            while not hit and tp is not None and depth < 64:
                t = tfm.get(tp)
                if t is None:
                    break
                try:
                    td = t.read_typetree()
                except Exception:
                    break
                if (td.get("m_GameObject") or {}).get("m_PathID") in lamp_set:
                    hit = True
                    break
                tp = (td.get("m_Father") or {}).get("m_PathID")
                depth += 1
            if hit:
                light_gos.append(lg)
        try:
            label = gos[sgo].read_typetree().get("m_Name", "switch")
        except Exception:
            label = "switch"
        switches.append({
            "id": f"unity:{level}:mb:{pid}",
            "level": level,
            "switch_go": sgo,
            # JOIN KEY: the same "<level>:<switchGO>" string eft_extract_lights writes onto each
            # controlled light's `group`, so the viewer maps this switch to its light group.
            "group": f"{level}:{sgo}",
            "world_pos": world_pos(sgo, gos, tfm),
            "label": label,
            "count": len(lamp_pids),
            "controlled_lamp_gos": sorted(lamp_set),
            "controlled_light_gos": sorted(set(light_gos)),
        })
    return switches


def extract_level(level):
    env = load_level(level)
    objs, tfm, gos, monos, sc = build_maps(env)
    return find_power_switches(level, objs, tfm, gos, monos, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True, help="comma-separated level indices to scan (e.g. 520)")
    ap.add_argument("--name", required=True, help="output dataset folder name")
    args = ap.parse_args()
    out = os.path.join(OUTROOT, args.name)
    os.makedirs(out, exist_ok=True)
    found = 0
    for lv in [int(x) for x in args.levels.split(",") if x.strip()]:
        try:
            sw = extract_level(lv)
        except Exception as e:
            print(f"level{lv}: scan failed ({e})", flush=True)
            continue
        fp = os.path.join(out, f"switches_{lv}.json")
        if sw:                                    # only write levels that actually hold a power lever
            json.dump(sw, open(fp, "w"))
            found += len(sw)
            for s in sw:
                print(f"  level{lv} switch '{s['label']}' GO {s['switch_go']} @ {s['world_pos']}: "
                      f"{s['count']} lamps -> {len(s['controlled_light_gos'])} lights -> {fp}", flush=True)
        elif os.path.isfile(fp):
            os.remove(fp)                         # stale (a game update moved the switch)
    print(f"switch scan: {found} power switch(es) across {len(args.levels.split(','))} level(s)", flush=True)


if __name__ == "__main__":
    main()
