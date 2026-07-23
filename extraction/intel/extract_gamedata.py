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
order is reversed after the flip so outlines stay CCW. "Anything with a zone" ships one:
MineDirectional blast boxes (largest CHILD BoxCollider — the mine GO itself has none),
quest/visit trigger zones (PlaceItemTrigger / ExperienceTrigger / FlareShootDetectorZone /
QuestTrigger, zone id = first script string), the LighthouseTraderZone compound, plus the
original minefields / sniper zones / exfils / transits.

LOOSE LOOT (first-party): LootPoint MonoBehaviours are the ONLY loose-loot positions the
client ships — a small curated set (lighthouse: gun racks / gun safes / food piles / car
trunks; factory: none). The bulk of loose loot is SERVER data (resources.assets carries only
"err"-wrapped Test*/LootData mocks of that exchange). A LootPoint payload DOES carry its item
pool: dword flags(=1); Id GUID string @4; 28-byte fixed block; dword N; N length-prefixed
24-hex item/category TEMPLATE ids; dword tail (validated on all 4 lighthouse variants). No
weights are serialized. Template ids resolve to names/prices via tarkov.dev (items +
itemCategories); each point is also nearest-neighbor-joined to tarkov.dev lootLoose for a
match-distance report. Both net steps degrade gracefully offline (ids ship un-named).

TERRAIN DRAPING: outline verts sit at the collider's BOTTOM face, which floats/sinks on
hills. When the map's .eftpack is present (EFT_PACK_DIR override, default <repo>/packs/
<map>.eftpack), a world heightfield is built from the pack's FLAG_TERRAIN instances (same
uv->world idea as eft_pipeline/build_grass.py, binned to a 2 m world-XZ grid), every outline
edge is subdivided ~4 m and each vert lifted to max(terrain+0.3, collider_base_y) — lines
follow the ground and never sink below the collider. Verts off the terrain grid keep the
collider Y. Zones keep a pre-subdivision "extent" [w, d] for the cards; the file gets a
top-level "draped" flag so the viewer can drop its own lift.

  python extraction/intel/extract_gamedata.py <map> [--levels a,b,c] [--out FILE]
      (requires EFT_TARKMAP_ROOT; levels default to the map config's source.levels;
       default output <EFT_TARKMAP_ROOT>/out/<map>/gamedata.json)
"""
import os, sys, json, gc, time, math, struct, functools, urllib.request
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
# global_matrix is a constant X-flip and is no longer stored per-config; default to it when absent.
_gm = (_cfg.get("coordinates") or {}).get("global_matrix")
G3 = (np.array(_gm, np.float64).reshape(4, 4) if _gm
      else np.diag([-1.0, 1.0, 1.0, 1.0]))[:3, :3]

# faction from the component TYPE — the whole point of this extractor.
EXFIL_CLASSES = {
    "ExfiltrationPoint": "pmc",
    "ScavExfiltrationPoint": "scav",
    "SharedExfiltrationPoint": "shared",
    "SecretExfiltrationPoint": "secret",
    # Vehicle extracts are separate typed components in newer scenes.
    "CarExtraction": "shared",
}
DOOR_CLASSES = {"Door": "door", "Trunk": "trunk", "KeycardDoor": "door", "SlidingDoor": "door",
                "ExfiltrationDoor": "exfil_door", "DoorSwitch": "door"}
# Swing doors we can open by rotating about the owner's local Z (Codex audit): Trunk / sliding /
# exfil doors need different motion and are marked non-swing so the viewer doesn't rotate them.
SWING_DOOR_CLASSES = {"Door", "KeycardDoor", "DoorSwitch"}
# EDoorState (EFT.Interactive) — flags; scenes serialize a single initial state.
DOOR_STATE = {0: "none", 1: "locked", 2: "shut", 4: "open", 8: "interacting", 16: "breach"}
# EPlayerSideMask
SIDE_MASK = {1: "usec", 2: "bear", 3: "pmc", 4: "savage", 5: "usec+savage", 6: "bear+savage", 7: "all"}
# quest/visit trigger MonoBehaviours ("anything with a zone gets extracted"): each carries its
# BoxCollider on the SAME GameObject and serializes the quest ZONE ID as the first script
# field (validated: lighthouse level524 x110, factory level68 x42).
QUEST_TRIGGER_CLASSES = {"PlaceItemTrigger": "place_item", "ExperienceTrigger": "visit",
                         "FlareShootDetectorZone": "flare", "QuestTrigger": "quest"}
BUFFER_ZONE_CLASSES = {
    "BufferGates": "buffer_gate", "BufferGate": "buffer_gate", "BufferZone": "buffer",
    "IgnorePlayerInputZone": "input_lock", "LighthouseKeeperZone": "lightkeeper",
    "EventObjectInteractive": "event_interactive",
    "InteractiveObjectCutsceneTrigger": "cutscene",
}


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
    Validated on 299 lighthouse doors: state column reads only {1,2,4,16}; keyed doors all 1.
    Also recovers the signed OPEN ANGLE (degrees) at IdEnd+56 (Codex audit: matched the authored
    open transform within 0.15deg on all 97 open Interchange doors). Returns (key, id, state, angle)."""
    import struct as _st
    key, kend = read_cstr(pl, 28)
    if key is None:
        return None, None, None, None
    did, iend = read_cstr(pl, kend + 12)
    st = None
    if iend + 96 <= len(pl):
        v = int.from_bytes(pl[iend + 92:iend + 96], "little")
        st = DOOR_STATE.get(v)                              # unknown value -> None, not garbage
    ang = None
    if iend + 60 <= len(pl):
        a = _st.unpack_from("<f", pl, iend + 56)[0]
        if a == a and 0.0 < abs(a) <= 180.0:                # finite, door-scale
            ang = round(float(a), 2)
    return (key or None), (did or None), st, ang


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


def dec_zone_id(pl):
    """quest-trigger classes serialize the zone id as the FIRST script field (string @0)."""
    s, _ = read_cstr(pl, 0)
    return (s or "").strip() or None


def poly_area_xz(pts):
    """shoelace area of a polygon projected to XZ (picks the mine's real blast box)."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, z1 = pts[i][0], pts[i][2]
        x2, z2 = pts[(i + 1) % n][0], pts[(i + 1) % n][2]
        a += x1 * z2 - x2 * z1
    return abs(a) / 2.0


def dec_lootpoint(pl):
    """LootPoint: dword flags(=1); Id GUID str @4; 28-byte fixed block (two variant dwords +
    zeros); dword N; N x length-prefixed 24-hex item/category template ids; dword tail.
    Recovered empirically on lighthouse levels 185-207 (all 4 GameObject variants agree). The
    array offset is SCANNED over a small window past the GUID instead of hardcoded, so a
    fixed-block size change degrades to (guid, []) rather than garbage."""
    guid, e = read_cstr(pl, 4)
    if guid is None or len(guid) < 8:
        return None, []
    for off in range(e, min(len(pl) - 4, e + 64), 4):
        n = int.from_bytes(pl[off:off + 4], "little")
        if not 1 <= n <= 64:
            continue
        tps, p, ok = [], off + 4, True
        for _ in range(n):
            s, p2 = read_cstr(pl, p)
            if s is None or len(s) != 24 or not all(c in "0123456789abcdef" for c in s):
                ok = False
                break
            tps.append(s)
            p = p2
        if ok:
            return guid, tps
    return guid, []


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

    # father -> [child transform pids], built LAZILY (reads every Transform typetree, so only
    # levels that actually hold child-collider zones — MineDirectional — pay for it).
    kid_map = None

    def child_transforms(tpid):
        nonlocal kid_map
        if kid_map is None:
            kid_map = {}
            for pid in tr_obj:
                d = tt(pid, tr_obj)
                f = (d.get("m_Father") or {}).get("m_PathID", 0) if d else 0
                if f:
                    kid_map.setdefault(f, []).append(pid)
        return kid_map.get(tpid, [])

    def largest_child_box(tpid):
        """(outline, center, child GO name) of the LARGEST child BoxCollider footprint — a
        MineDirectional hangs its blast/trigger boxes on child GOs (MON-50_MineTrigger x3 + a
        small body collider); the largest box IS the danger zone."""
        best = None
        for cpid in child_transforms(tpid):
            cd = tt(cpid, tr_obj)
            cgo = (cd.get("m_GameObject") or {}).get("m_PathID", 0) if cd else 0
            cname, _, ccols = go_info(cgo)
            M2 = world_mat(cpid)
            for col in ccols:
                fp = footprint(M2, col)
                area = poly_area_xz(fp)
                if best is None or area > best[0]:
                    best = (area, fp, col_center(M2, col), cname)
        return (best[1], best[2], best[3]) if best else (None, None, None)

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
        if cls not in EXFIL_CLASSES and cls not in DOOR_CLASSES \
                and cls not in QUEST_TRIGGER_CLASSES and cls not in (
                "Minefield", "SniperFiringZone", "TransitPoint", "StationaryWeapon",
                "SpawnPointMarker", "MineDirectional", "LootPoint", "LootPointsGroup",
                "LighthouseTraderZone", "BufferGateSwitcher") and cls not in BUFFER_ZONE_CLASSES:
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
            key, did, st, ang = dec_door(pl)
            rec = {
                "pos": tpos, "key_id": key, "state": st, "kind": DOOR_CLASSES[cls],
                "id": did, "name": name, "active": active, "lv": lv,
            }
            # Swing doors (Door/KeycardDoor/DoorSwitch) carry the open angle so the viewer can
            # animate them about their pivot; trunks/sliding/exfil doors move differently (no swing).
            if cls in SWING_DOOR_CLASSES:
                rec["swing"] = True
                if ang is not None:
                    rec["open_angle"] = ang
            sink["doors"].append(rec)
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
            # blast/trigger zone = the largest CHILD BoxCollider footprint (the mine GO itself
            # has none). Kind from the child name ("MON-50_MineTrigger" -> "MON-50").
            ol, cen, cname = largest_child_box(tpid) if tpid else (None, None, None)
            kind = (cname or "").split("_MineTrigger")[0] if cname and "_MineTrigger" in cname else None
            sink["mines_directional"].append({
                "pos": cen or tpos, "name": name, "kind": kind,
                "outline": ol or [], "active": active, "lv": lv,
            })
        elif cls in QUEST_TRIGGER_CLASSES:
            box = cols[0] if cols else None
            sink["quest_triggers"].append({
                "pos": col_center(M, box) if box else tpos,
                "name": dec_zone_id(pl) or name,
                "kind": QUEST_TRIGGER_CLASSES[cls],
                "outline": footprint(M, box) if box else [],
                "active": active, "lv": lv,
            })
        elif cls == "LighthouseTraderZone":
            box = cols[0] if cols else None
            sink["trader_zones"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "BufferGateSwitcher":
            sink["buffer_switches"].append({
                "pos": tpos, "name": name or "Buffer gate switch", "kind": cls,
                "active": active, "lv": lv,
            })
        elif cls in BUFFER_ZONE_CLASSES:
            box = cols[0] if cols else None
            sink["buffer_zones"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "kind": BUFFER_ZONE_CLASSES[cls],
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "LootPointsGroup":
            box = cols[0] if cols else None
            sink["loot_groups"].append({
                "pos": col_center(M, box) if box else tpos, "name": name or "Loot points group",
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "LootPoint":
            guid, tps = dec_lootpoint(pl)
            if guid:
                sink["loose_points"].append({
                    "pos": tpos, "name": name, "guid": guid, "templates": tps,
                    "active": active, "lv": lv,
                })

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


# ---------------------------------------------------------------------------
# TERRAIN HEIGHTFIELD (Task: drape zone outlines on the ground)
# ---------------------------------------------------------------------------
class TerrainField:
    """World-XZ heightfield from the map's .eftpack FLAG_TERRAIN instances (pack space =
    viewer space, so outline verts sample it directly). Vertices are binned to a CELL-metre
    grid (mean Y per cell — terrain slices overlap at seams); sampling is a bilinear blend of
    the 4 surrounding cell centres, degrading to the plain cell mean where neighbours are
    missing and to None off the grid. Same data as eft_pipeline/build_grass.py's uv->world
    grids, but keyed by world XZ (we need height AT a point, not point AT a uv)."""
    CELL = 2.0  # instances are filtered on FLAG_TERRAIN = 1<<1 (build_grass.py's contract)

    def __init__(self, pack_dir):
        mani = json.load(open(os.path.join(pack_dir, "manifest.json")))
        mb = open(os.path.join(pack_dir, "meshes.bin"), "rb").read()
        ib = open(os.path.join(pack_dir, "instances.bin"), "rb").read()
        vl = mani["vertex"]
        vs = vl["stride"]
        poff = next(a for a in vl["attrs"] if a["name"] == "position")["offset"]
        inst = mani["instance"]
        istride = inst["stride"]
        fo = {f["name"]: f["offset"] for f in inst["fields"]}
        id2mesh = {m["id"]: m for m in mani["meshes"]}
        pts = []
        for i in range(len(ib) // istride):
            b = i * istride
            if not struct.unpack_from("<I", ib, b + fo["flags"])[0] & 2:
                continue
            a = np.array(struct.unpack_from("<12f", ib, b + fo["affine"]),
                         np.float64).reshape(3, 4)
            me = id2mesh[struct.unpack_from("<I", ib, b + fo["meshId"])[0]]
            n, off = me["vtxCount"], me["vtxOffset"]
            vb = np.frombuffer(mb, np.uint8, count=n * vs, offset=off).reshape(n, vs)
            loc = vb[:, poff:poff + 12].copy().view("<f4").astype(np.float64)
            pts.append(loc @ a[:, :3].T + a[:, 3])
        if not pts:
            raise ValueError("no FLAG_TERRAIN instances")
        w = np.concatenate(pts)
        c = self.CELL
        ix = np.floor(w[:, 0] / c).astype(np.int64)
        iz = np.floor(w[:, 2] / c).astype(np.int64)
        self.x0, self.z0 = int(ix.min()), int(iz.min())
        nx, nz = int(ix.max()) - self.x0 + 1, int(iz.max()) - self.z0 + 1
        s = np.zeros((nx, nz), np.float64)
        n = np.zeros((nx, nz), np.int64)
        np.add.at(s, (ix - self.x0, iz - self.z0), w[:, 1])
        np.add.at(n, (ix - self.x0, iz - self.z0), 1)
        self.h = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        self.n_verts, self.n_cells = len(w), int((n > 0).sum())

    def sample(self, x, z):
        """terrain height at world (x, z) or None when off-grid."""
        c = self.CELL
        fx, fz = x / c - 0.5 - self.x0, z / c - 0.5 - self.z0
        x0, z0 = int(np.floor(fx)), int(np.floor(fz))
        tx, tz = fx - x0, fz - z0
        acc = wsum = 0.0
        for dx, dz, wgt in ((0, 0, (1 - tx) * (1 - tz)), (1, 0, tx * (1 - tz)),
                            (0, 1, (1 - tx) * tz), (1, 1, tx * tz)):
            xi, zi = x0 + dx, z0 + dz
            if 0 <= xi < self.h.shape[0] and 0 <= zi < self.h.shape[1]:
                v = self.h[xi, zi]
                if np.isfinite(v) and wgt > 0.0:
                    acc += v * wgt
                    wsum += wgt
        return acc / wsum if wsum > 1e-6 else None


def load_terrain_field():
    """the map's pack heightfield, or None (indoor maps / pack absent) — draping is optional."""
    pack = os.environ.get("EFT_PACK_DIR") or os.path.join(
        os.path.dirname(KIT), "packs", f"{MAP}.eftpack")
    if not os.path.exists(os.path.join(pack, "manifest.json")):
        print(f"[drape] no pack at {pack} - outlines stay at collider Y")
        return None
    try:
        tf = TerrainField(pack)
        print(f"[drape] heightfield from {pack}: {tf.n_verts} terrain verts -> "
              f"{tf.n_cells} cells @ {tf.CELL} m")
        return tf
    except ValueError:
        print(f"[drape] pack has no terrain instances (indoor map) - outlines stay at collider Y")
        return None
    except Exception as ex:
        print(f"[drape] heightfield failed ({type(ex).__name__}: {ex}) - outlines stay at collider Y")
        return None


def drape_outline(outline, field, step=4.0, lift=0.3):
    """subdivide each closed-outline edge every ~`step` m and set vert Y to
    max(terrain + lift, collider_base_y); base Y interpolates along the edge and is kept
    wherever the terrain grid has no data. Returns the new vert list."""
    n = len(outline)
    if n < 3 or field is None:
        return outline
    out = []
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        seg = math.hypot(b[0] - a[0], b[2] - a[2])
        k = max(1, int(math.ceil(seg / step)))
        for j in range(k):
            t = j / k
            x = a[0] + (b[0] - a[0]) * t
            y0 = a[1] + (b[1] - a[1]) * t
            z = a[2] + (b[2] - a[2]) * t
            ty = field.sample(x, z)
            y = max(ty + lift, y0) if ty is not None else y0
            out.append([round(x, 2), round(y, 2), round(z, 2)])
    return out


def outline_extent(outline):
    """[w, d] metres of the (pre-subdivision) rectangular footprint, for the viewer cards."""
    if len(outline) < 3:
        return None
    d = lambda a, b: math.dist(a, b)
    w, h = d(outline[0], outline[1]), d(outline[1], outline[2])
    return [round(w, 1), round(h, 1)] if w > 0.05 and h > 0.05 else None


# Ground-hugging zones (colliders that genuinely sit on the terrain): drape to the ground so
# the outline follows undulating terrain instead of floating/sinking at the flat collider face.
DRAPE_KEYS = ("exfils", "transit_points", "quest_triggers", "trader_zones",
              "buffer_zones", "loot_groups")
# Elevated collider zones (minefields, sniper zones, directional mines): keep the collider's own
# world height. Their trigger boxes are frequently TALL volumes whose bottom face reaches the base
# terrain far below a raised platform (e.g. ground_zero Minefield_LowPower: collider center Y=15.65
# on a train platform, but bottom face Y<=-0.41 at ground). Draping to terrain snapped the whole
# zone to the ground far below where the mines actually are. USER PREFERENCE: use the collider's
# actual height (its center, == the marker `pos`), NOT the terrain drape.
COLLIDER_HEIGHT_KEYS = ("minefields", "sniper_zones", "mines_directional")


def drape_zones(sink):
    """Place every zone outline (and stamp its pre-subdivision extent):
      * DRAPE_KEYS          -> subdivide ~4 m and lift to the terrain (ground-hugging zones).
      * COLLIDER_HEIGHT_KEYS -> keep the collider's OWN height (the marker `pos` Y), never terrain.
    Returns (terrain_field_loaded?, verts_before, verts_after) for the drape group's logging."""
    field = load_terrain_field()
    before = after = 0
    for k in DRAPE_KEYS:
        for r in sink[k]:
            ol = r.get("outline") or []
            if len(ol) < 3:
                continue
            ext = outline_extent(ol)
            if ext:
                r["extent"] = ext
            before += len(ol)
            r["outline"] = drape_outline(ol, field)
            after += len(r["outline"])
    # Elevated collider zones: FLATTEN each outline to the collider center height (`pos` Y), which
    # is exactly where the marker sphere sits, so the ring/wall/marker all read at the platform
    # level. NO terrain sampling, NO subdivision (the footprint is already a horizontal rectangle).
    for k in COLLIDER_HEIGHT_KEYS:
        for r in sink[k]:
            ol = r.get("outline") or []
            if len(ol) < 3:
                continue
            ext = outline_extent(ol)
            if ext:
                r["extent"] = ext
            raw_ys = [p[1] for p in ol]
            y = round(r["pos"][1], 2)  # collider center height == the marker position
            print(f"[collider-height] {k} {r.get('name')!r:36s} pos_y={r['pos'][1]:.2f} "
                  f"footprint_y=[{min(raw_ys):.2f},{max(raw_ys):.2f}] -> outline_y={y}")
            r["outline"] = [[round(p[0], 2), y, round(p[2], 2)] for p in ol]
    return field is not None, before, after


# ---------------------------------------------------------------------------
# tarkov.dev RESOLUTION + lootLoose JOIN (loose_points) — all failures degrade to offline.
# ---------------------------------------------------------------------------
DEV_API = "https://api.tarkov.dev/graphql"
# tarkov.dev display name for the per-map lootLoose query (same per-map fetch pattern as
# build_loot.py — the all-maps query 503s). Unlisted maps fall back to a title-cased key.
DEV_NAME = {"lighthouse": "Lighthouse", "factory": "Factory", "factory_rework": "Factory",
            "labs": "The Lab",
            "streets": "Streets of Tarkov", "ground_zero": "Ground Zero", "labyrinth": "The Labyrinth"}


def gql(q, tries=3):
    req = urllib.request.Request(DEV_API, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "eft-native-viewer-gamedata/1.0"})
    last = None
    for i in range(tries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            if "errors" in r:
                raise RuntimeError("tarkov.dev errors: " + json.dumps(r["errors"][:2])[:300])
            return r["data"]
        except Exception as ex:                                  # 503/timeout/offline
            last = ex
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"tarkov.dev unreachable: {last}")


def resolve_templates(loose):
    """template id -> {'n','s','pr','cat'} via tarkov.dev items(ids) + itemCategories.
    cat=1 marks a CATEGORY template ('Food and drink' pool slot, no price/icon)."""
    ids = sorted({t for r in loose for t in r["templates"]})
    if not ids or os.environ.get("EFT_GAMEDATA_OFFLINE"):
        return {}
    idx = {}
    try:
        lst = ",".join('"%s"' % i for i in ids)
        d = gql("{ items(ids: [%s]) { id name shortName avg24hPrice } }" % lst)
        for it in d.get("items") or []:
            idx[it["id"]] = {"n": it.get("name"), "s": it.get("shortName"),
                             "pr": it.get("avg24hPrice"), "cat": 0}
        left = [i for i in ids if i not in idx]
        if left:
            cd = gql("{ itemCategories { id name } }")
            cidx = {c["id"]: c["name"] for c in cd.get("itemCategories") or []}
            for i in left:
                if i in cidx:
                    idx[i] = {"n": cidx[i], "s": None, "pr": None, "cat": 1}
        print(f"[loose] resolved {len(idx)}/{len(ids)} template ids via tarkov.dev "
              f"({sum(1 for v in idx.values() if v['cat'])} categories)")
    except RuntimeError as ex:
        print(f"[loose] template resolution OFFLINE ({ex}) - shipping raw template ids")
    return idx


def join_dev_loose(loose):
    """nearest tarkov.dev lootLoose point per first-party point -> r['dev_d'] (m) + items for
    template-less points within 2.5 m. Prints the match-distance distribution."""
    if not loose or os.environ.get("EFT_GAMEDATA_OFFLINE"):
        return
    name = DEV_NAME.get(MAP, MAP.replace("_", " ").title())
    try:
        d = gql('{ maps(name:"%s"){ lootLoose { position { x y z } '
                'items { name shortName avg24hPrice } } } }' % name)
        ms = d.get("maps") or []
        rows = (ms[0].get("lootLoose") or []) if ms else []
    except RuntimeError as ex:
        print(f"[loose] lootLoose join OFFLINE ({ex})")
        return
    pts = []
    for ll in rows:
        p = ll.get("position") or {}
        if all(k in p for k in "xyz"):
            pts.append(([-p["x"], p["y"], p["z"]], ll.get("items") or []))  # dev -> viewer bridge
    if not pts:
        print(f"[loose] tarkov.dev lootLoose('{name}') returned 0 points - no join")
        return
    P = np.array([p for p, _ in pts])
    ds = []
    for r in loose:
        q = np.array(r["pos"])
        i = int(np.argmin(((P - q) ** 2).sum(axis=1)))
        dist = float(np.linalg.norm(P[i] - q))
        r["dev_d"] = round(dist, 2)
        ds.append(dist)
        # a point whose payload had NO pool still gets the snapshot's items when co-located
        if not r.get("items") and dist <= 2.5:
            best = sorted(pts[i][1], key=lambda it: -(it.get("avg24hPrice") or 0))[:4]
            r["items"] = [{"n": it.get("name"), "s": it.get("shortName"),
                           "pr": it.get("avg24hPrice"), "cat": 0} for it in best]
            r["items_src"] = "tarkov.dev"
    ds = np.array(ds)
    print(f"[loose] join vs {len(pts)} tarkov.dev lootLoose points: "
          f"median {np.median(ds):.1f} m, p90 {np.percentile(ds, 90):.1f} m, max {ds.max():.1f} m; "
          f"<=1m {(ds <= 1).sum()}, <=2m {(ds <= 2).sum()}, <=5m {(ds <= 5).sum()} of {len(ds)}")


def finalize_loose(sink):
    """guid-dedupe (scene variants re-serialize the same rack), then merge same-name points
    within 0.5 m into one map point (a rack has several spawn SLOTS at ~one spot): n = slot
    count, templates = union. Then resolve templates + join tarkov.dev."""
    best = {}
    for r in sink["loose_points"]:
        k = r["guid"]
        if k not in best or (r.get("active") and not best[k].get("active")):
            best[k] = r
    merged = []
    for r in best.values():
        hit = None
        for m in merged:
            if m["name"] == r["name"] and math.dist(m["pos"], r["pos"]) <= 0.5:
                hit = m
                break
        if hit is None:
            merged.append({"pos": r["pos"], "name": r["name"], "n": 1,
                           "templates": list(r["templates"]),
                           "active": r["active"], "lv": r["lv"]})
        else:
            hit["n"] += 1
            hit["active"] = hit["active"] or r["active"]
            for t in r["templates"]:
                if t not in hit["templates"]:
                    hit["templates"].append(t)
    idx = resolve_templates(merged)
    for r in merged:
        items = []
        for t in r["templates"]:
            e = idx.get(t)
            items.append({"tpl": t, **e} if e else {"tpl": t})
        # priced real items first, categories last — the viewer titles the card off items[0]
        items.sort(key=lambda it: (it.get("cat", 0), -(it.get("pr") or 0)))
        if items:
            r["items"] = items
            r["items_src"] = "game files"  # the POOL is client data; names/prices are lookups
        del r["templates"]
    join_dev_loose(merged)
    sink["loose_points"] = merged


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
                            "transit_points", "stationary", "spawn_points",
                            "mines_directional", "loose_points",
                            "quest_triggers", "trader_zones", "buffer_switches",
                            "buffer_zones", "loot_groups")}
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
    for k in ("minefields", "sniper_zones", "transit_points", "stationary", "mines_directional",
              "quest_triggers", "trader_zones", "buffer_switches", "buffer_zones", "loot_groups"):
        sink[k] = dedupe(sink[k], lambda r: (r.get("name"), tuple(r["pos"])))
    sink["spawn_points"] = dedupe(sink["spawn_points"], lambda r: (r.get("name"), tuple(r["pos"])))

    # first-party loose loot: guid-dedupe + slot-merge, then tarkov.dev resolution + join.
    finalize_loose(sink)
    # terrain-drape every zone outline (subdivide ~4 m; Y = max(terrain+0.3, collider base)).
    draped, ol_before, ol_after = drape_zones(sink)
    if draped:
        print(f"[drape] outline verts {ol_before} -> {ol_after}")

    logic_levels = sorted({e["lv"] for e in sink["exfils"]})
    counts = {k: len(v) for k, v in sink.items()}
    counts["exfils_by_faction"] = dict(Counter(e["faction"] for e in sink["exfils"]))
    counts["doors_with_key"] = sum(1 for d in sink["doors"] if d.get("key_id"))
    out = {"map": MAP, "generated_levels": scanned, "logic_levels": logic_levels,
           "draped": draped, "counts": counts, **sink}
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
    mk = Counter(m.get("kind") for m in sink["mines_directional"])
    qk = Counter(q.get("kind") for q in sink["quest_triggers"])
    print(f"  mines_directional: {len(sink['mines_directional'])} kinds={dict(mk)} "
          f"with_outline={sum(1 for m in sink['mines_directional'] if m.get('outline'))}")
    print(f"  quest_triggers: {len(sink['quest_triggers'])} kinds={dict(qk)}")
    for z in sink["trader_zones"]:
        print(f"  trader_zone {z['name']} pos={z['pos']} outline_pts={len(z['outline'])}")
    for r in sink["loose_points"]:
        top = (r.get("items") or [{}])[0]
        print(f"  loose {str(r['name'])[:28]:28s} pos={r['pos']} slots={r['n']} "
              f"pool={len(r.get('items') or [])} top={top.get('s') or top.get('n') or top.get('tpl')} "
              f"dev_d={r.get('dev_d')}")
    print("  copy next to the pack:  copy \"%s\" packs\\%s.eftpack\\" % (OUT, MAP))


if __name__ == "__main__":
    main()
