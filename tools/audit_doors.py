#!/usr/bin/env python
"""Audit the door pipeline against the GAME's own truth -- run after touching dec_door or the
door animation. Nothing here is authored: every expected value is re-derived from the Unity
scenes, then compared with what the built pack actually ships.

Per door it re-reads, straight from levelN:
  * the payload's signed OPEN ANGLE + EDoorState             (extract_gamedata.dec_door)
  * the Transform's LOCAL rotation (axis + angle)            -- ground truth for the swing:
      a door authored OPEN is rotated by exactly its open angle about its local +Z;
      a SHUT door sits at local identity, which is what makes the pack's baked matrix
      the CLOSED pose.
  * the renderer subtree                                     -- the parts that must swing
Then, against <repo>/packs/<map>.eftpack:
  * does the pack carry the door, with an angle?
  * does every shipped part resolve to a pack instance at that position?

Exit code 0 = clean, 1 = at least one MISMATCH (so CI/a build script can gate on it).

  python tools/audit_doors.py <map> [--levels a,b,c] [--limit N] [--pack DIR]

Env: EFT_GAME_DATA (game dir), EFT_TARKMAP_ROOT (maps/ + out/) -- same contract as the
extraction kit; see README.md. ASCII output only.
"""
import argparse
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "extraction", "intel"))

DATA = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
DOOR_STATE = {0: "none", 1: "locked", 2: "shut", 4: "open", 8: "interacting", 16: "breach"}
SWING = ("Door", "KeycardDoor", "DoorSwitch")
# A door's hinge is its local +Z (see docs/DOORS.md); tolerance for calling an axis "Z".
AXIS_TOL = 0.02
ANGLE_TOL = 0.5      # degrees: payload vs authored local rotation
POS_TOL = 1.0        # metres: a shipped part must land on a pack instance this close


def read_cstr(buf, off):
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


def dec_door(pl):
    """both payload layouts -- kept in step with extract_gamedata.dec_door (docs/DOORS.md S3)."""
    def u32(o):
        return int.from_bytes(pl[o:o + 4], "little") if o + 4 <= len(pl) else None

    def tail(kend):
        did, iend = read_cstr(pl, kend + 12)
        st = DOOR_STATE.get(int.from_bytes(pl[iend + 92:iend + 96], "little")) \
            if iend + 96 <= len(pl) else None
        ang = None
        if iend + 60 <= len(pl):
            a = struct.unpack_from("<f", pl, iend + 56)[0]
            if a == a and 0.0 < abs(a) <= 180.0:
                ang = round(float(a), 2)
        return did, st, ang

    def is_key(s):
        return s == "" or (len(s) == 24 and all(c in "0123456789abcdef" for c in s))

    key, kend = read_cstr(pl, 28)
    if key is not None and is_key(key):
        return (key, *tail(kend))
    n = u32(20)
    if n is None or not 0 < n <= 8:
        return None, None, None, None
    off = 24
    for _ in range(n):
        off += 4
        s, off = read_cstr(pl, off)
        if s is None:
            return None, None, None, None
    if u32(off) != 0x0F:
        return None, None, None, None
    key, kend = read_cstr(pl, off + 4)
    if key is None or not is_key(key):
        return None, None, None, None
    return (key, *tail(kend))


def quat_axis_angle(q):
    """(unit axis, signed degrees) of a Unity quaternion; (None, 0) for ~identity."""
    x, y, z, w = q
    w = max(-1.0, min(1.0, w))
    ang = math.degrees(2.0 * math.acos(w))
    if ang > 180.0:
        ang -= 360.0
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-6:
        return None, 0.0
    return (x / s, y / s, z / s), ang


def load_pack_doors(pack):
    gd = os.path.join(pack, "gamedata.json")
    if not os.path.isfile(gd):
        return None
    return json.load(open(gd, encoding="utf-8"))


def load_pack_instances(pack):
    """[(mesh name, (x,y,z))] for every instance -- to confirm parts resolve."""
    man = json.load(open(os.path.join(pack, "manifest.json"), encoding="utf-8"))
    stride = man["instance"]["stride"]
    names = [m["name"] for m in man["meshes"]]
    out = {}
    with open(os.path.join(pack, "instances.bin"), "rb") as f:
        buf = f.read()
    for i in range(len(buf) // stride):
        aff = struct.unpack_from("<12f", buf, i * stride)
        mid = struct.unpack_from("<I", buf, i * stride + 48)[0]
        if mid < len(names):
            out.setdefault(names[mid], []).append((aff[3], aff[7], aff[11]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--levels", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pack", default=None)
    args = ap.parse_args()

    import UnityPy

    tk = os.environ.get("EFT_TARKMAP_ROOT")
    cfg_p = None
    for c in ([os.path.join(tk, "maps", args.map, "config.json")] if tk else []) + \
             [os.path.join(REPO, "extraction", "maps", args.map, "config.json")]:
        if os.path.isfile(c):
            cfg_p = c
            break
    if not cfg_p:
        raise SystemExit(f"audit_doors: no config for map '{args.map}'")
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    levels = [int(x) for x in args.levels.split(",")] if args.levels else \
        [int(x) for x in (cfg["source"].get("levels") or [])]

    pack = args.pack or os.path.join(REPO, "packs", f"{args.map}.eftpack")
    gd = load_pack_doors(pack)
    pack_inst = load_pack_instances(pack) if gd else {}
    pack_doors = {}
    if gd:
        for d in gd.get("doors", []):
            pack_doors[(round(d["pos"][0], 1), round(d["pos"][1], 1), round(d["pos"][2], 1))] = d
    print(f"[audit] map={args.map} levels={len(levels)} pack={'yes' if gd else 'MISSING'}")

    n = bad = 0
    stats = {"open_authored": 0, "axis_z": 0, "angle_match": 0, "parts_resolved": 0,
             "parts_total": 0, "parts_culled": 0, "parts_lost": 0}
    for lv in levels:
        p = os.path.join(DATA, f"level{lv}")
        if not os.path.exists(p):
            continue
        env = UnityPy.load(p)
        objs = list(env.objects)
        gos = {o.path_id: o for o in objs if o.type.name == "GameObject"}
        tfm = {o.path_id: o for o in objs if o.type.name in ("Transform", "RectTransform")}
        local_scripts = {}
        for o in objs:
            if o.type.name == "MonoScript":
                try:
                    local_scripts[o.path_id] = o.read_typetree().get("m_ClassName")
                except Exception:
                    pass
        sf = next((f for f in env.files.values() if hasattr(f, "objects")), None)
        externals = list(getattr(sf, "externals", []) or [])
        ext_cache = {}

        def cls_of(hdr):
            s = hdr.get("m_Script") or {}
            fid, pid = s.get("m_FileID", 0), s.get("m_PathID", 0)
            if fid == 0:
                return local_scripts.get(pid)
            base = os.path.basename(getattr(externals[fid - 1], "path", "").replace("\\", "/"))
            if base not in ext_cache:
                idx = {}
                fp = os.path.join(DATA, base)
                if os.path.exists(fp):
                    for oo in UnityPy.load(fp).objects:
                        if oo.type.name == "MonoScript":
                            try:
                                idx[oo.path_id] = oo.read_typetree().get("m_ClassName")
                            except Exception:
                                pass
                ext_cache[base] = idx
            return ext_cache[base].get(pid)

        def go_tf(gpid):
            try:
                for c in gos[gpid].read_typetree().get("m_Component", []):
                    cp = c.get("component") or c.get("second") or {}
                    if cp.get("m_PathID") in tfm:
                        return cp.get("m_PathID")
            except Exception:
                pass
            return None

        for o in objs:
            if o.type.name != "MonoBehaviour":
                continue
            try:
                hdr = o.read_typetree(check_read=False)
            except Exception:
                continue
            if cls_of(hdr) not in SWING:
                continue
            gpid = (hdr.get("m_GameObject") or {}).get("m_PathID")
            tp = go_tf(gpid) if gpid else None
            if tp is None:
                continue
            raw = o.get_raw_data()
            nm = hdr.get("m_Name") or ""
            pl = raw[(12 + 4 + 12 + 4 + len(nm.encode("utf8")) + 3) & ~3:]
            _key, _id, st, ang = dec_door(pl)
            td = tfm[tp].read_typetree()
            q = td.get("m_LocalRotation") or {}
            axis, la = quat_axis_angle((q.get("x", 0.0), q.get("y", 0.0),
                                        q.get("z", 0.0), q.get("w", 1.0)))
            gname = gos[gpid].read_typetree().get("m_Name", "?")
            n += 1
            msgs = []
            if st == "open":
                stats["open_authored"] += 1
                # GROUND TRUTH: an authored-open door is rotated by its open angle about local Z
                if axis is None or abs(abs(axis[2]) - 1.0) > AXIS_TOL:
                    msgs.append(f"MISMATCH open door hinge is not local Z (axis={axis})")
                else:
                    stats["axis_z"] += 1
                    expect = la * (1.0 if axis[2] > 0 else -1.0)
                    if ang is None:
                        msgs.append(f"MISMATCH no payload angle, authored rotation is {expect:+.2f}")
                    elif abs(expect - ang) > ANGLE_TOL:
                        msgs.append(f"MISMATCH payload angle {ang:+.2f} vs authored {expect:+.2f}")
                    else:
                        stats["angle_match"] += 1
            elif st == "shut" and axis is not None and abs(la) > ANGLE_TOL:
                msgs.append(f"MISMATCH shut door is not at local identity ({la:+.2f} deg) "
                            f"-- the pack's baked matrix is NOT the closed pose")
            # pack side
            if gd is not None and ang is not None:
                pd = None
                for k, cand in pack_doors.items():
                    if cand.get("name") == gname and cand.get("open_angle") == ang:
                        pd = cand
                        break
                if pd is None:
                    msgs.append("MISMATCH not found in the pack's gamedata")
                else:
                    parts = pd.get("parts") or []
                    stats["parts_total"] += len(parts)
                    for mesh, pos in parts:
                        cands = pack_inst.get(mesh)
                        if not cands:
                            # the mesh isn't in the pack AT ALL -> a culled proxy (ballistic /
                            # collision panel). Expected, not a failure.
                            stats["parts_culled"] += 1
                            continue
                        if any((c[0] - pos[0]) ** 2 + (c[1] - pos[1]) ** 2 +
                               (c[2] - pos[2]) ** 2 < POS_TOL ** 2 for c in cands):
                            stats["parts_resolved"] += 1
                        else:
                            # the mesh IS shipped but nothing sits where the game says -> real
                            stats["parts_lost"] += 1
                            msgs.append(f"MISMATCH part '{mesh}' has no pack instance at "
                                        f"{[round(v, 2) for v in pos]}")
            if msgs:
                bad += 1
                print(f"  [{gname[:44]:44s}] state={st} ang={ang}")
                for m in msgs:
                    print(f"      {m}")
            if args.limit and n >= args.limit:
                break
        del env, objs
        if args.limit and n >= args.limit:
            break

    print(f"\n[audit] {n} swing door(s); {bad} with problems")
    print(f"  authored-OPEN doors (the ground truth set): {stats['open_authored']}")
    print(f"    hinge on local Z: {stats['axis_z']}   payload angle == authored: {stats['angle_match']}")
    print(f"  shipped parts: {stats['parts_total']} = {stats['parts_resolved']} resolved to a pack "
          f"instance + {stats['parts_culled']} culled proxies (not in the pack at all) + "
          f"{stats['parts_lost']} LOST")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
