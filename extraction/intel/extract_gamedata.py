#!/usr/bin/env python
"""Extract TYPED gameplay data from an EFT map's Unity scenes -> out/<map>/gamedata.json.

Ground truth for the tactical overlay: instead of name-classifying GameObjects (the
extract_semantics.py heuristic — 71% false-positive rate on extracts), this reads the TYPED
MonoBehaviours the game itself uses: ExfiltrationPoint / ScavExfiltrationPoint /
SharedExfiltrationPoint / SecretExfiltrationPoint (faction = component TYPE), Minefield,
SniperFiringZone, Door / Trunk (KeyId + DoorState), TransitPoint, StationaryWeapon,
SpawnPointMarker.

EFT is IL2CPP with an ENCRYPTED global-metadata.dat, so script typetrees CANNOT be generated;
each MonoBehaviour parses only its 32-byte header (m_GameObject/m_Enabled/m_Script/m_Name).
The script class comes from the MonoScript (an engine type, typetree intact); the script
FIELDS are decoded from the raw payload with layouts recovered empirically on lighthouse
level524 + all-level door dumps (column statistics; same method as the light-controller
decode). Layouts are validated defensively — a field that doesn't look right degrades to null
instead of shipping garbage.

Zone footprints come from the BoxCollider on the same GameObject: 4 bottom-face corners
through the full world TRS chain (colliders are often unit boxes scaled by the transform).
The Unity->viewer bridge is the map config's coordinates.global_matrix conjugation reduced to
points (G3 @ p, the diag(-1,1,1) X-flip) — identical rule as the geometry pipeline; corner
order is reversed after the flip so outlines stay CCW.

  python extraction/intel/extract_gamedata.py <map> [--levels a,b,c] [--out FILE]
      (requires EFT_TARKMAP_ROOT; levels default to the map config's source.levels;
       default output <EFT_TARKMAP_ROOT>/out/<map>/gamedata.json)
"""
import os, sys, json, gc, time, math, struct, functools
from collections import Counter

import numpy as np
print = functools.partial(print, flush=True)
import UnityPy

# portable kit: paths come from the environment (see extraction/README.md)
HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/extraction/intel
KIT = os.path.dirname(HERE)                                # <repo>/extraction
DATA = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
TK = os.environ.get("EFT_TARKMAP_ROOT")
if not TK:
    raise SystemExit("extract_gamedata: EFT_TARKMAP_ROOT is not set. Point it at your workspace "
                     "tarkmap dir (the one holding maps/ and out/), "
                     "e.g.  setx EFT_TARKMAP_ROOT D:\\eft_work\\tarkmap")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAP = args[0] if args else "lighthouse"
LEVELS = None
OUT = None
for a in sys.argv[1:]:
    if a.startswith("--levels="):
        LEVELS = [int(x) for x in a.split("=", 1)[1].split(",")]
    elif a.startswith("--out="):
        OUT = a.split("=", 1)[1]

_cfg_p = os.path.join(TK, "maps", MAP, "config.json")
if not os.path.exists(_cfg_p):
    _cfg_p = os.path.join(KIT, "maps", MAP, "config.json")
_cfg = json.load(open(_cfg_p, encoding="utf-8"))
if LEVELS is None:
    LEVELS = [int(x) for x in (_cfg["source"].get("levels") or [])]
if OUT is None:
    OUT = os.path.join(TK, "out", MAP, "gamedata.json")
G3 = np.array(_cfg["coordinates"]["global_matrix"], np.float64).reshape(4, 4)[:3, :3]

# faction from the component TYPE — the whole point of this extractor.
EXFIL_CLASSES = {
    "ExfiltrationPoint": "pmc",
    "ScavExfiltrationPoint": "scav",
    "SharedExfiltrationPoint": "shared",
    "SecretExfiltrationPoint": "secret",
}
DOOR_CLASSES = {"Door": "door", "Trunk": "trunk", "KeycardDoor": "door", "SlidingDoor": "door"}
# EDoorState (EFT.Interactive) — flags; scenes serialize a single initial state.
DOOR_STATE = {0: "none", 1: "locked", 2: "shut", 4: "open", 8: "interacting", 16: "breach"}
# EPlayerSideMask
SIDE_MASK = {1: "usec", 2: "bear", 3: "pmc", 4: "savage", 5: "usec+savage", 6: "bear+savage", 7: "all"}


def read_cstr(buf, off):
    """length-prefixed utf8 string + 4-aligned end offset; (None, off) when implausible."""
    if off + 4 > len(buf):
        return None, off
    ln = int.from_bytes(buf[off:off + 4], "little")
    if ln < 0 or ln > 4096 or off + 4 + ln > len(buf):
        return None, off
    try:
        s = buf[off + 4:off + 4 + ln].decode("utf8")
    except UnicodeDecodeError:
        return None, off
    return s, (off + 4 + ln + 3) & ~3


def payload_of(o, hdr):
    """raw script fields after the 32(+name)-byte MonoBehaviour header."""
    raw = o.get_raw_data()
    nm = hdr.get("m_Name") or ""
    hsize = (12 + 4 + 12 + 4 + len(nm.encode("utf8")) + 3) & ~3
    return raw[hsize:]


# ---- payload decoders (layouts: see module docstring) ----
def dec_exfil_name(pl):
    """all four exfil classes: 48 fixed bytes, then the settings Name string."""
    s, _ = read_cstr(pl, 48)
    return (s or "").strip() or None


def dec_door(pl):
    """WorldInteractiveObject prefix: 28 bytes, KeyId str, 12 bytes, Id str, ... state at +92.
    Validated on 299 lighthouse doors: state column reads only {1,2,4,16}; keyed doors all 1."""
    key, kend = read_cstr(pl, 28)
    if key is None:
        return None, None, None
    did, iend = read_cstr(pl, kend + 12)
    st = None
    if iend + 96 <= len(pl):
        v = int.from_bytes(pl[iend + 92:iend + 96], "little")
        st = DOOR_STATE.get(v)                              # unknown value -> None, not garbage
    return (key or None), (did or None), st


def dec_spawn(pl):
    """Id str, Name str, Vector3 pos, Quaternion rot, Sides mask, Categories mask, Infil str."""
    sid, off = read_cstr(pl, 0)
    name, off = read_cstr(pl, off)
    if name is None or off + 36 > len(pl):
        return None
    pos = struct.unpack_from("<3f", pl, off)
    sides = int.from_bytes(pl[off + 28:off + 32], "little")
    cats = int.from_bytes(pl[off + 32:off + 36], "little")
    inf, _ = read_cstr(pl, off + 36)
    if not all(math.isfinite(v) and abs(v) < 1e5 for v in pos):
        return None
    return sid, name, pos, sides, cats, inf


def dec_stationary_name(pl):
    s, _ = read_cstr(pl, 20)
    return (s or "").strip() or None


def trs_mat(t, q, s):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)
    M = np.eye(4)
    M[:3, :3] = R * np.array(s, np.float64)
    M[:3, 3] = t
    return M


def bridge(p):
    """Unity world point -> viewer/pack space (global_matrix conjugation reduced to a point)."""
    return [round(float(v), 2) for v in (G3 @ np.asarray(p, np.float64))]


# MonoScript name cache is global across levels (externals repeat).
_ms_idx = {}

def monoscript_index(path):
    if path not in _ms_idx:
        idx = {}
        if os.path.exists(path):
            e = UnityPy.load(path)
            for o in e.objects:
                if o.type.name == "MonoScript":
                    try:
                        idx[o.path_id] = o.read_typetree().get("m_ClassName")
                    except Exception:
                        pass
            del e
        _ms_idx[path] = idx
    return _ms_idx[path]


def scan_level(lv, sink):
    p = os.path.join(DATA, f"level{lv}")
    if not os.path.exists(p):
        print(f"[level{lv}] missing - skip")
        return
    t0 = time.time()
    env = UnityPy.load(p)
    sf = next((f for f in env.files.values() if hasattr(f, "objects")), None)
    externals = list(getattr(sf, "externals", []) or [])
    objs = env.objects

    local_scripts = None

    def resolve(fid, pid):
        nonlocal local_scripts
        if fid == 0:
            if local_scripts is None:
                local_scripts = {}
                for o in objs:
                    if o.type.name == "MonoScript":
                        try:
                            local_scripts[o.path_id] = o.read_typetree().get("m_ClassName")
                        except Exception:
                            pass
            return local_scripts.get(pid)
        base = os.path.basename(getattr(externals[fid - 1], "path", "").replace("\\", "/"))
        return monoscript_index(os.path.join(DATA, base)).get(pid)

    go_obj, tr_obj, col_obj = {}, {}, {}
    mbs = []
    for o in objs:
        tn = o.type.name
        if tn == "GameObject":
            go_obj[o.path_id] = o
        elif tn in ("Transform", "RectTransform"):
            tr_obj[o.path_id] = o
        elif tn == "BoxCollider":
            col_obj[o.path_id] = o
        elif tn == "MonoBehaviour":
            mbs.append(o)

    # lazy per-object typetree caches (engine types — typetrees intact)
    tt_cache = {}

    def tt(pid, table):
        if pid not in tt_cache:
            o = table.get(pid)
            try:
                tt_cache[pid] = o.read_typetree() if o else None
            except Exception:
                tt_cache[pid] = None
        return tt_cache[pid]

    def vec(d, k, dft):
        v = d.get(k) or {}
        return [v.get("x", dft[0]), v.get("y", dft[1]), v.get("z", dft[2])]

    go_tt_cache = {}

    def go_tt(pid):
        if pid not in go_tt_cache:
            o = go_obj.get(pid)
            try:
                go_tt_cache[pid] = o.read_typetree() if o else None
            except Exception:
                go_tt_cache[pid] = None
        return go_tt_cache[pid]

    wm_cache = {}

    def world_mat(tpid):
        """full TRS world matrix of a Transform (father-chain product), memoized per node."""
        if tpid in wm_cache:
            return wm_cache[tpid]
        d = tt(tpid, tr_obj)
        if not d:
            wm_cache[tpid] = np.eye(4)
            return wm_cache[tpid]
        q = [d.get("m_LocalRotation", {}).get(a, b) for a, b in zip("xyzw", (0, 0, 0, 1))]
        L = trs_mat(vec(d, "m_LocalPosition", (0, 0, 0)), q, vec(d, "m_LocalScale", (1, 1, 1)))
        f = (d.get("m_Father") or {}).get("m_PathID", 0)
        M = world_mat(f) @ L if f else L
        wm_cache[tpid] = M
        return M

    act_cache = {}

    def active_chain(tpid, go_pid):
        """GO m_IsActive AND every ancestor's — inactive content still ships, flag it."""
        key = (tpid, go_pid)
        if key in act_cache:
            return act_cache[key]
        gd = go_tt(go_pid)
        ok = bool(gd.get("m_IsActive", True)) if gd else True
        if ok and tpid:
            d = tt(tpid, tr_obj)
            f = (d.get("m_Father") or {}).get("m_PathID", 0) if d else 0
            if f:
                fd = tt(f, tr_obj)
                fgo = (fd.get("m_GameObject") or {}).get("m_PathID", 0) if fd else 0
                ok = active_chain(f, fgo)
        act_cache[key] = ok
        return ok

    def go_info(go_pid):
        """(name, transform pid, [BoxCollider tt, ...]) of a GameObject."""
        gd = go_tt(go_pid)
        if not gd:
            return None, None, []
        tpid, cols = None, []
        for c in gd.get("m_Component", []):
            pp = (c.get("component") if isinstance(c, dict) else None) or c
            cid = pp.get("m_PathID") if isinstance(pp, dict) else None
            if cid is None:
                continue
            if cid in tr_obj and tpid is None:
                tpid = cid
            elif cid in col_obj:
                d = tt(cid, col_obj)
                if d:
                    cols.append(d)
        return gd.get("m_Name"), tpid, cols

    def footprint(M, col):
        """4 world bottom-face corners of a BoxCollider under world matrix M, bridged.
        Corner order reversed after the X-flip so the outline stays CCW."""
        c = vec(col, "m_Center", (0, 0, 0))
        s = vec(col, "m_Size", (1, 1, 1))
        hx, hz = s[0] / 2.0, s[2] / 2.0
        y = c[1] - s[1] / 2.0
        loc = [(c[0] - hx, y, c[2] - hz), (c[0] + hx, y, c[2] - hz),
               (c[0] + hx, y, c[2] + hz), (c[0] - hx, y, c[2] + hz)]
        out = [bridge((M @ np.array([*l, 1.0]))[:3]) for l in loc]
        return [out[0], out[3], out[2], out[1]]

    def col_center(M, col):
        c = vec(col, "m_Center", (0, 0, 0))
        return bridge((M @ np.array([*c, 1.0]))[:3])

    n_hit = 0
    for o in mbs:
        try:
            hdr = o.read_typetree(check_read=False)
        except Exception:
            continue
        s = hdr.get("m_Script") or {}
        try:
            cls = resolve(s.get("m_FileID", 0), s.get("m_PathID", 0))
        except Exception:
            cls = None
        if cls not in EXFIL_CLASSES and cls not in DOOR_CLASSES and cls not in (
                "Minefield", "SniperFiringZone", "TransitPoint", "StationaryWeapon",
                "SpawnPointMarker", "MineDirectional"):
            continue
        go_pid = (hdr.get("m_GameObject") or {}).get("m_PathID")
        if not go_pid:
            continue
        name, tpid, cols = go_info(go_pid)
        M = world_mat(tpid) if tpid else np.eye(4)
        tpos = bridge(M[:3, 3])
        active = active_chain(tpid, go_pid) and bool(hdr.get("m_Enabled", 1))
        pl = payload_of(o, hdr)
        n_hit += 1

        if cls in EXFIL_CLASSES:
            box = cols[0] if cols else None
            sink["exfils"].append({
                "name": dec_exfil_name(pl) or name or "Extract",
                "faction": EXFIL_CLASSES[cls],
                "pos": col_center(M, box) if box else tpos,
                "outline": footprint(M, box) if box else [],
                "go": name, "active": active, "lv": lv,
            })
        elif cls == "Minefield":
            for box in (cols or [None]):
                sink["minefields"].append({
                    "pos": col_center(M, box) if box else tpos,
                    "outline": footprint(M, box) if box else [],
                    "name": name, "active": active, "lv": lv,
                })
        elif cls == "SniperFiringZone":
            box = cols[0] if cols else None
            sink["sniper_zones"].append({
                "pos": col_center(M, box) if box else tpos,
                "outline": footprint(M, box) if box else [],
                "name": name, "active": active, "lv": lv,
            })
        elif cls in DOOR_CLASSES:
            key, did, st = dec_door(pl)
            sink["doors"].append({
                "pos": tpos, "key_id": key, "state": st, "kind": DOOR_CLASSES[cls],
                "id": did, "name": name, "active": active, "lv": lv,
            })
        elif cls == "TransitPoint":
            box = cols[0] if cols else None
            sink["transit_points"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "StationaryWeapon":
            sink["stationary"].append({
                "pos": tpos, "name": dec_stationary_name(pl) or name or "Stationary weapon",
                "active": active, "lv": lv,
            })
        elif cls == "SpawnPointMarker":
            d = dec_spawn(pl)
            if d:
                sid, sname, pos, sides, cats, inf = d
                sink["spawn_points"].append({
                    "pos": bridge(pos), "name": sname, "side": SIDE_MASK.get(sides, str(sides)),
                    "categories_mask": cats, "infiltration": inf or None, "lv": lv,
                })
            else:
                sink["spawn_points"].append({"pos": tpos, "name": name, "side": None,
                                             "categories_mask": None, "infiltration": None, "lv": lv})
        elif cls == "MineDirectional":
            sink["mines_directional"].append({"pos": tpos, "lv": lv})

    print(f"[level{lv}] {len(objs)} objs, {len(mbs)} MBs -> {n_hit} typed hits ({time.time()-t0:.0f}s)")
    del env, objs, go_obj, tr_obj, col_obj, mbs, tt_cache, go_tt_cache, wm_cache
    gc.collect()


def dedupe(rows, keyf):
    """cross-level dedupe; an ACTIVE row wins over an inactive twin."""
    best = {}
    for r in rows:
        k = keyf(r)
        if k not in best or (r.get("active") and not best[k].get("active")):
            best[k] = r
    return list(best.values())


def sibling_levels(scanned):
    """AUTO-PROBE for the gameplay-logic scene: the map config's level list is the GEOMETRY
    set and may not include it (factory: exfils live in Factory_DesignStuff = level 68, not in
    levels 2/69/70/177). Candidates = every BuildSettings scene in the SAME directory as the
    already-scanned levels' scenes (data-driven, no per-map constants)."""
    try:
        env = UnityPy.load(os.path.join(DATA, "globalgamemanagers"))
        scenes = None
        for o in env.objects:
            if o.type.name == "BuildSettings":
                d = o.read_typetree()
                scenes = d.get("scenes") or d.get("m_Scenes") or []
                break
        if not scenes:
            return []
        dirs = {os.path.dirname(scenes[lv]) for lv in scanned if 0 <= lv < len(scenes)}
        cand = [i for i, s in enumerate(scenes)
                if os.path.dirname(s) in dirs and i not in scanned]
        print(f"[auto-probe] no exfils in the config levels; probing {len(cand)} sibling "
              f"scenes: {cand}")
        return cand
    except Exception as ex:
        print(f"[auto-probe] failed: {type(ex).__name__}: {ex}")
        return []


def main():
    print(f"[cfg] map={MAP} levels={LEVELS} G3={G3.round(2).tolist()}")
    sink = {k: [] for k in ("exfils", "minefields", "sniper_zones", "doors",
                            "transit_points", "stationary", "spawn_points", "mines_directional")}
    t0 = time.time()
    scanned = list(LEVELS)
    for lv in LEVELS:
        scan_level(lv, sink)
    # the logic scene (the one carrying the exfil MBs) may sit outside the config's
    # geometry-level list — probe the sibling scenes for it.
    if not sink["exfils"]:
        extra = sibling_levels(scanned)
        for lv in extra:
            scan_level(lv, sink)
        scanned += extra

    sink["exfils"] = dedupe(sink["exfils"], lambda r: (r["faction"], r["name"]))
    sink["doors"] = dedupe(sink["doors"], lambda r: r["id"] or (r["name"], tuple(r["pos"])))
    for k in ("minefields", "sniper_zones", "transit_points", "stationary", "mines_directional"):
        sink[k] = dedupe(sink[k], lambda r: (r.get("name"), tuple(r["pos"])))
    sink["spawn_points"] = dedupe(sink["spawn_points"], lambda r: (r.get("name"), tuple(r["pos"])))

    logic_levels = sorted({e["lv"] for e in sink["exfils"]})
    counts = {k: len(v) for k, v in sink.items()}
    counts["exfils_by_faction"] = dict(Counter(e["faction"] for e in sink["exfils"]))
    counts["doors_with_key"] = sum(1 for d in sink["doors"] if d.get("key_id"))
    out = {"map": MAP, "generated_levels": scanned, "logic_levels": logic_levels,
           "counts": counts, **sink}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"\n[out] {OUT}  ({os.path.getsize(OUT)/1e3:.0f} kB, {time.time()-t0:.0f}s)")
    print("[counts]", json.dumps(counts))
    for e in sink["exfils"]:
        print(f"  exfil [{e['faction']:6s}] {e['name']:34s} pos={e['pos']} outline_pts={len(e['outline'])} active={e['active']}")
    for t in sink["transit_points"]:
        print(f"  transit {t['name']:24s} pos={t['pos']}")
    for s in sink["stationary"]:
        print(f"  stationary {s['name']:12s} pos={s['pos']}")
    st = Counter(d["state"] for d in sink["doors"])
    print(f"  doors: {len(sink['doors'])} states={dict(st)} with_key={counts['doors_with_key']}")
    print("  copy next to the pack:  copy \"%s\" packs\\%s.eftpack\\" % (OUT, MAP))


if __name__ == "__main__":
    main()
