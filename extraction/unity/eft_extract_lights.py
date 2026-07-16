"""Extract a map's REAL in-game lights from its Unity *_light scene -> lights_<level>.json, which
eft_build_gn.py auto-imports. Interchange's light scene is level64 (Shopping_Mall_light).

    python extraction/unity/eft_extract_lights.py --level 64 --name interchange_v2

Schema (bmpq-compatible): a flat array of
  {name, type:"Point"|"Spot"|"Directional", position:[x,y,z] (Unity world),
   rotation:[x,y,z,w] (Unity world quat), color:[r,g,b,a] (linear), intensity, range,
   spotAngle, innerSpotAngle, shadowType}
The builder maps these to Blender lamps: location=(-x,-z,y), a rotation fixup, energy=intensity*MULT,
color linear, spot_size=rad(spotAngle), spot_blend=1-inner/outer, cutoff_distance=range.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eft_scene_extract import trs               # local TRS helper (quat+pos+scale -> 4x4)
from eft_extract_v2 import g

# portable kit: paths come from the environment (see extraction/README.md)
#   EFT_GAME_DATA   = the game's EscapeFromTarkov_Data dir (default: standard install path)
#   EFT_ASSETS_ROOT = where extracted datasets are written (default: <EFT_TARKMAP_ROOT>/../eft_assets, else ./eft_assets)
EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else
    os.path.join(os.getcwd(), "eft_assets"))
TYPE = {0: "Spot", 1: "Directional", 2: "Point", 3: "Rectangle", 4: "Disc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True, help="the *_light level index (Interchange=64)")
    ap.add_argument("--name", required=True, help="output dataset folder name (e.g. interchange_v2)")
    ap.add_argument("--all", action="store_true", help="ALSO keep disabled/inactive lights, tagged on:false "
                    "(night-only banks, controller-driven fixtures) -> lights_<level>_all.json")
    args = ap.parse_args()
    import UnityPy

    env = UnityPy.load(os.path.join(EFTDATA, f"level{args.level}"))
    tfm = {o.path_id: o for o in env.objects if o.type.name == "Transform"}
    gos_all = {o.path_id: o for o in env.objects if o.type.name == "GameObject"}
    monos_all = {o.path_id: o for o in env.objects if o.type.name == "MonoBehaviour"}

    def controller_state(go_pid):
        """Newer maps (Icebreaker+) serialize m_Intensity=0 and drive lamps from a 220-byte
        controller MonoBehaviour on the same GameObject. Field layout recovered by column
        statistics over all 1564 Icebreaker controllers: f[8]=spotAngle(15-55),
        f[9]=intensity(~1-4), f[28]=range(10-40). Returns (intensity, range, angle) or None."""
        import struct as _st
        go = gos_all.get(go_pid)
        if go is None:
            return None
        try:
            gtt = go.read_typetree()
        except Exception:
            return None
        for comp in gtt.get("m_Component", []):
            pid = comp.get("component", {}).get("m_PathID")
            mb = monos_all.get(pid)
            if mb is None:
                continue
            try:
                raw = mb.get_raw_data()
            except Exception:
                continue
            if len(raw) == 220:
                f = _st.unpack("<55f", raw)
                # intensity ONLY: f[28]/f[8] looked like range/angle but their distribution
                # (10-40 m) contradicts the authored range population (3-7 m) - overriding
                # range made every lamp span whole decks (uniform white wash, no contrast).
                # The zeroed Lights keep AUTHORED range/angle/color; only intensity is
                # controller-driven.
                if 0.05 < f[9] < 60:
                    return float(f[9]), 0.0, 0.0
            elif len(raw) == 92:
                # smaller lamp-controller class (flicker bulbs etc.): f[9]=intensity,
                # f[20]=range (column stats over factory_rework + icebreaker corpora).
                # f[9] validated against the directly-authored population on factory.
                f = _st.unpack("<23f", raw)
                if 0.03 < f[9] < 60:
                    return float(f[9]), 0.0, 0.0
        return None
    go2tf = {}
    for pid, o in tfm.items():
        try: go2tf[o.read().m_GameObject.path_id] = pid
        except Exception: pass

    _go_active = {}
    def go_active(go_ref):
        pid = getattr(go_ref, "path_id", 0)
        if pid in _go_active: return _go_active[pid]
        try: act = bool(g(go_ref.read(), "m_IsActive", default=True))
        except Exception: act = True
        _go_active[pid] = act
        return act

    def world(go_pid):
        """World matrix AND Unity activeInHierarchy (every ancestor GO must be active)."""
        tp = go2tf.get(go_pid)
        if tp is None: return np.eye(4), True
        chain = []; cur = tfm.get(tp); gd = 0; active = True
        while cur is not None and gd < 256:
            gd += 1; t = cur.read(); chain.append(t)
            go = getattr(t, "m_GameObject", None)
            if go is not None and not go_active(go):
                active = False
            fp = getattr(t, "m_Father", None)
            if fp is None or getattr(fp, "path_id", 0) == 0: break
            cur = tfm.get(fp.path_id)
        W = np.eye(4)
        for t in reversed(chain): W = W @ trs(t)
        return W, active

    lights = []
    n_total = n_disabled = 0
    for o in env.objects:
        if o.type.name != "Light": continue
        n_total += 1
        try:
            L = o.read(); intensity = float(g(L, "m_Intensity", default=0.0) or 0.0)
            ctrl = None
            if intensity <= 0:
                ctrl = controller_state(L.m_GameObject.path_id)
                if ctrl is None:
                    continue                                 # blank/placeholder
                intensity = ctrl[0]
            # faithful default: only lights that are LIVE in-game (component enabled AND every
            # ancestor GameObject active) - Factory_Rework's Day_Light scene carries whole
            # disabled banks that made the import 5x too dense/bright. --all keeps them, tagged
            # on:false, so a viewer can opt into the night-only/controller-driven banks.
            on = bool(g(L, "m_Enabled", default=True))
            W, active = world(L.m_GameObject.path_id)
            if not active:
                on = False
            if not on:
                n_disabled += 1
                if not args.all:
                    continue
            pos = W[:3, 3]
            if abs(pos[0]) < 0.01 and abs(pos[2]) < 0.01: continue   # pooled-at-origin
            # world quaternion from the rotation part (orthonormal R -> quat, xyzw)
            R = W[:3, :3]
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s = 0.5 / np.sqrt(tr + 1.0); qw = 0.25 / s
                qx = (R[2, 1] - R[1, 2]) * s; qy = (R[0, 2] - R[2, 0]) * s; qz = (R[1, 0] - R[0, 1]) * s
            else:
                i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]])); j = (i + 1) % 3; k = (i + 2) % 3
                s = np.sqrt(max(1e-9, R[i, i] - R[j, j] - R[k, k] + 1.0)) * 2
                q = [0, 0, 0]; q[i] = 0.25 * s
                q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
                qw = (R[k, j] - R[j, k]) / s; qx, qy, qz = q
            # the read() object's attribute access silently misses several Light fields
            # (m_Type read as absent -> everything defaulted to Point); the TYPETREE is
            # the reliable source for scalars
            tt = o.read_typetree()
            colr = tt.get("m_Color") or {}
            color = [float(colr.get("r", 1)), float(colr.get("g", 1)),
                     float(colr.get("b", 1)), float(colr.get("a", 1))]
            shv = tt.get("m_Shadows") or {}
            lights.append({
                "name": str(g(L.m_GameObject.read(), "m_Name", default="light")),
                "type": TYPE.get(int(tt.get("m_Type", 2)), "Point"),
                "position": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
                "rotation": [round(float(qx), 5), round(float(qy), 5), round(float(qz), 5), round(float(qw), 5)],
                "color": [round(c, 4) for c in color],
                "intensity": round(intensity, 4),
                "range": round(ctrl[1] if (ctrl and ctrl[1] > 0) else float(tt.get("m_Range", 0.0) or 0.0), 3),
                "spotAngle": round(ctrl[2] if (ctrl and ctrl[2] > 0) else float(tt.get("m_SpotAngle", 0.0) or 0.0), 2),
                "innerSpotAngle": round(float(tt.get("m_InnerSpotAngle", 0.0) or 0.0), 2),
                "shadowType": int((shv.get("m_Type", 0) if isinstance(shv, dict) else 0) or 0),
                "on": on,
            })
        except Exception:
            continue

    out = os.path.join(OUTROOT, args.name); os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, f"lights_{args.level}_all.json" if args.all else f"lights_{args.level}.json")
    json.dump(lights, open(fp, "w"))
    import collections
    byt = collections.Counter(l["type"] for l in lights)
    n_off = sum(1 for l in lights if not l.get("on", True))
    print(f"level{args.level}: {n_total} Light components -> {len(lights)} kept "
          f"({n_disabled} disabled/inactive{' INCLUDED as on:false' if args.all else ' filtered'}; "
          f"{n_off} off in output) {dict(byt)} -> {fp}")


if __name__ == "__main__":
    main()
