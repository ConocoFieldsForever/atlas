"""extract_decals — StaticDeferredDecal projectors -> dataset decal quads.

The spray paint ("STOP", "UNTAR GO HOME"), wall graffiti, ground stains and painted markings in
EFT are NOT meshes, material variants, or textures on the receiving surface: they are
`StaticDeferredDecal` MonoBehaviour PROJECTORS (1,737 on Interchange across its levels), invisible
to any MeshRenderer walk — which is why the geometry extractor never saw them and no name search
could find them (the projector GameObjects have recycled names; the one painting the writings
atlas at Interchange's checkpoint shields is literally called "decal_carshadow (70)").

PAYLOAD (verified against live decals, level62):
    [f32 x1, f32 y1, f32 x2, f32 y2]      atlas-PIXEL rect selecting the cell (which word/stain)
    [i32 fileID, i64 pathID]              PPtr<Material> through the level's externals
The GameObject's world transform is the projection box; the decal image spans the box's local XY
and projects along local Z.

OUTPUT (into the DATASET, so the normal assemble pipeline ships everything untouched):
    meshes/decal_quad__gen.obj   one shared two-sided unit quad (extractor vertex convention:
                                 X-negated locals, both windings emitted so face culling can
                                 never hide a decal)
    tex/<name>__<src>_<pid>.png  every referenced atlas, exported once, dataset naming scheme
    decals.json                  {"instances": [...]} in scene.json's instance schema, one quad
                                 per projector: mesh=the shared quad, m=the projector's RAW
                                 Unity-space world matrix (the assembler owns handedness),
                                 subs=[{tex, role: "decal", uv: [su, sv, ou, ov], ...}]

Usage: extract_decals.py <map> --dataset=<dataset_dir> [--levels=a,b,c]
Env:   EFT_GAME_DATA (default Battlestate install), EFT_TARKMAP_ROOT (map configs, like
       extract_gamedata).
"""
import json
import math
import os
import struct
import sys

import numpy as np
import UnityPy

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
DATA = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
TK = os.environ.get("EFT_TARKMAP_ROOT")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAP = args[0] if args else "interchange"
LEVELS = None
DS = None
for a in sys.argv[1:]:
    if a.startswith("--levels="):
        LEVELS = [int(x) for x in a.split("=", 1)[1].split(",")]
    elif a.startswith("--dataset="):
        DS = a.split("=", 1)[1]
if not DS:
    raise SystemExit("extract_decals: --dataset=<dataset_dir> is required")
if LEVELS is None:
    for root in (TK, KIT):
        if not root:
            continue
        p = os.path.join(root, "maps", MAP, "config.json")
        if os.path.exists(p):
            cfg = json.load(open(p, encoding="utf-8"))
            LEVELS = [int(x) for x in (cfg["source"].get("levels") or [])]
            break
if not LEVELS:
    raise SystemExit("extract_decals: no levels (pass --levels= or provide the map config)")

QUAD_NAME = "decal_quad__gen.obj"

# The map's global_matrix (constant X-flip; same default as extract_gamedata).
G3 = np.diag([-1.0, 1.0, 1.0])


def write_quad(path):
    """Two-sided unit quad in the XZ PLANE, facing +/-Y.

    The projector's box spans local X and Z and projects along local Y, not the XY/along-Z that
    "decal projector" naively suggests. Measured, not assumed: across all 1,269 Interchange
    projectors the atlas cell's aspect matches |colX|/|colZ| about three times better than
    |colX|/|colY| (mean |log| error 0.43 vs 1.34), and the checkpoint's own sprays sit in a box
    scaled (3.53, 0.63, 1.13) against a 469x149 cell whose 3.15 aspect matches 3.53/1.13 = 3.12.
    Building the quad in XY laid every decal on its side.

    Verts lift +0.01 along Y against z-fighting; X-NEGATED like every extractor OBJ, and both
    windings are written so the decal shows regardless of the receiving pipeline's cull mode."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("g decal_quad__gen\n")
        for x, z in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            f.write("v %g 0.01 %g\n" % (-x, z))  # X-negated local frame, XZ plane
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            f.write("vt %g %g\n" % (u, v))
        # front (as-negated winding) + back, both referencing the same vt
        f.write("f 1/1 3/3 2/2\nf 1/1 4/4 3/3\n")
        f.write("f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")


def quat_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def main():
    os.makedirs(os.path.join(DS, "tex"), exist_ok=True)
    os.makedirs(os.path.join(DS, "meshes"), exist_ok=True)
    write_quad(os.path.join(DS, "meshes", QUAD_NAME))

    script_cache = {}

    def scripts_for(fname):
        if fname not in script_cache:
            m = {}
            fp = os.path.join(DATA, fname)
            if os.path.isfile(fp):
                try:
                    for o in UnityPy.load(fp).objects:
                        if o.type.name == "MonoScript":
                            try:
                                m[o.path_id] = o.read_typetree().get("m_ClassName", "?")
                            except Exception:
                                pass
                except Exception:
                    pass
            script_cache[fname] = m
        return script_cache[fname]

    # (source file, material pid) -> {"tex": dataset tex name or None, "nrm": ..., "w": px, "h": px}
    mat_cache = {}
    mat_envs = {}

    def material_info(fname, pid):
        # NOTE the local name `menv`. This used to be `env`, which SHADOWED nothing at define
        # time but read the enclosing per-level `env` at call time under Python's closure rules
        # once the level loop rebound it -- so material lookups searched the LEVEL file instead
        # of the material's own file and silently returned None. That single letter hid the
        # writings atlas (the STOP / UNTAR sprays) behind a "602 unresolved materials" counter.
        key = (fname, pid)
        if key in mat_cache:
            return mat_cache[key]
        info = None
        try:
            if fname not in mat_envs:
                mat_envs[fname] = UnityPy.load(os.path.join(DATA, fname))
            menv = mat_envs[fname]
            byid = {o.path_id: o for o in menv.objects}
            mo = byid.get(pid)
            if mo is not None and mo.type.name == "Material":
                md = mo.read_typetree()
                out = {"name": md.get("m_Name"), "tex": None, "nrm": None, "w": 0, "h": 0}
                mat_ext = None
                for f in menv.files.values():
                    try:
                        mat_ext = f.externals
                        break
                    except Exception:
                        pass
                for te in md.get("m_SavedProperties", {}).get("m_TexEnvs", []):
                    k = te[0] if isinstance(te, (list, tuple)) else te.get("first")
                    v = te[1] if isinstance(te, (list, tuple)) else te.get("second")
                    tref = v.get("m_Texture", {})
                    tp, tf = tref.get("m_PathID"), tref.get("m_FileID")
                    if not tp:
                        continue
                    if k not in ("_MainTex", "_BumpMap"):
                        continue  # array/aux slots share pathIDs and confuse the export naming
                    if tf == 0:
                        to = byid.get(tp)
                        tex_src = fname
                    else:
                        # ONE-HOP external: the material's own externals table names the file
                        # holding the texture. Most decal materials resolve only this way.
                        fname2 = (getattr(mat_ext[tf - 1], "name", None)
                                  if mat_ext and tf <= len(mat_ext) else None)
                        if not fname2:
                            continue
                        if fname2 not in mat_envs:
                            try:
                                mat_envs[fname2] = UnityPy.load(os.path.join(DATA, fname2))
                            except Exception:
                                continue
                        to = {o2.path_id: o2 for o2 in mat_envs[fname2].objects}.get(tp)
                        tex_src = fname2
                    if to is None or to.type.name != "Texture2D":
                        continue
                    tname = to.read_typetree().get("m_Name", "tex")
                    src = os.path.basename(tex_src).replace(".assets", "")
                    ds_name = "%s__%s_%d" % (tname, src, tp)
                    png = os.path.join(DS, "tex", ds_name + ".png")
                    if not os.path.isfile(png):
                        try:
                            img = to.read().image
                            img.save(png)
                        except Exception:
                            continue
                    if k == "_MainTex":
                        from PIL import Image
                        with Image.open(png) as im:
                            out["w"], out["h"] = im.size
                        out["tex"] = ds_name
                    elif k == "_BumpMap":
                        out["nrm"] = ds_name
                info = info or out
        except Exception as e:
            import traceback
            print("  [decals] material %s:%s unreadable (%s: %s)" % (fname, pid, type(e).__name__, e))
            if os.environ.get("EFT_DECAL_DEBUG"):
                traceback.print_exc()
            info = None
        mat_cache[key] = info
        return info

    instances = []
    n_seen = n_bad_payload = n_no_mat = n_inactive = 0

    for lv in LEVELS:
        lp = os.path.join(DATA, "level%d" % lv)
        if not os.path.isfile(lp):
            continue
        try:
            env = UnityPy.load(lp)
        except Exception as e:
            print("  [decals] level%d load failed: %s" % (lv, e))
            continue
        byid = {o.path_id: o for o in env.objects}
        ext = None
        for f in env.files.values():
            try:
                ext = f.externals
                break
            except Exception:
                pass
        local_scripts = {}
        tf = {}
        gos = {}
        for o in env.objects:
            t = o.type.name
            if t == "MonoScript":
                try:
                    local_scripts[o.path_id] = o.read_typetree().get("m_ClassName", "?")
                except Exception:
                    pass
            elif t == "Transform":
                try:
                    d = o.read_typetree()
                    tf[o.path_id] = (
                        d["m_GameObject"]["m_PathID"], d["m_Father"]["m_PathID"],
                        (d["m_LocalPosition"]["x"], d["m_LocalPosition"]["y"], d["m_LocalPosition"]["z"]),
                        (d["m_LocalRotation"]["x"], d["m_LocalRotation"]["y"], d["m_LocalRotation"]["z"], d["m_LocalRotation"]["w"]),
                        (d["m_LocalScale"]["x"], d["m_LocalScale"]["y"], d["m_LocalScale"]["z"]),
                    )
                except Exception:
                    pass
            elif t == "GameObject":
                try:
                    d = o.read_typetree()
                    gos[o.path_id] = bool(d.get("m_IsActive", True))
                except Exception:
                    pass
        go_tf = {e[0]: pid for pid, e in tf.items()}
        gonames = {}
        for _o in env.objects:
            if _o.type.name == "GameObject":
                try:
                    gonames[_o.path_id] = _o.read_typetree().get("m_Name", "")
                except Exception:
                    pass

        world_cache = {}

        def world(pid):
            """Parent-composed (pos, quat, scale) in RAW Unity space."""
            if pid in world_cache:
                return world_cache[pid]
            e = tf.get(pid)
            if e is None:
                return None
            go, fa, lp_, lr, ls = e
            if fa == 0 or fa not in tf:
                w = (np.array(lp_, np.float64), lr, np.array(ls, np.float64))
            else:
                pw = world(fa)
                if pw is None:
                    w = (np.array(lp_, np.float64), lr, np.array(ls, np.float64))
                else:
                    pp, pr, ps = pw
                    R = quat_mat(pr)
                    pos = pp + R @ (np.array(lp_, np.float64) * ps)
                    x1, y1, z1, w1 = pr
                    x2, y2, z2, w2 = lr
                    rot = (
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    )
                    w = (pos, rot, ps * np.array(ls, np.float64))
            world_cache[pid] = w
            return w

        def active_chain(pid):
            seen = 0
            while pid and pid in tf and seen < 64:
                go = tf[pid][0]
                if not gos.get(go, True):
                    return False
                pid = tf[pid][1]
                seen += 1
            return True

        lv_count = 0
        for o in env.objects:
            if o.type.name != "MonoBehaviour":
                continue
            try:
                h = o.read_typetree(check_read=False)
            except Exception:
                continue
            sc = h.get("m_Script", {})
            fid, spid = sc.get("m_FileID", 0), sc.get("m_PathID", 0)
            if fid == 0:
                cls = local_scripts.get(spid)
            else:
                fname = getattr(ext[fid - 1], "name", None) if ext and fid <= len(ext) else None
                cls = scripts_for(fname).get(spid) if fname else None
            if cls != "StaticDeferredDecal":
                continue
            n_seen += 1
            go_pid = h.get("m_GameObject", {}).get("m_PathID")
            # EFT_DECAL_TRACE=<GameObject name substring> follows ONE projector through
            # every gate. This is what finally located the checkpoint sprays after three wrong
            # theories; keep it.
            _trace = os.environ.get("EFT_DECAL_TRACE")
            _tname = None
            if _trace:
                _tname = gonames.get(go_pid, "")
                if _trace not in _tname:
                    _trace = None
                else:
                    print("  [trace] %s: seen, go=%s" % (_tname, go_pid))
            tpid = go_tf.get(go_pid)
            w = world(tpid) if tpid else None
            if _trace:
                print("  [trace] %s: tpid=%s world=%s" % (_tname, tpid, "ok" if w else "NONE"))
            if w is None:
                continue
            raw = o.get_raw_data()
            nm = h.get("m_Name") or ""
            hsize = (12 + 4 + 12 + 4 + len(nm.encode("utf8")) + 3) & ~3
            pl = raw[hsize:]
            if _trace:
                print("  [trace] %s: payload %dB" % (_tname, len(pl)))
            if len(pl) < 28:
                n_bad_payload += 1
                continue
            x1, y1, x2, y2 = struct.unpack_from("<4f", pl, 0)
            mfid, mpid = struct.unpack_from("<iq", pl, 16)
            # Rects legitimately carry small NEGATIVE insets (-2, -8 px are common) and can
            # overrun the texture by a pixel: they are authored bleed, not corruption. Rejecting
            # on `0 <= v` threw away whole atlas families, the writings sprays among them.
            # Accept a modest out-of-range margin and let the UV clamp below handle the rest.
            bad_rect = (not all(math.isfinite(v) and -64 <= v <= 16384 for v in (x1, y1, x2, y2))
                        or x2 <= x1 or y2 <= y1)
            bad_ptr = mfid <= 0 or not ext or mfid > len(ext)
            if _trace and (bad_rect or bad_ptr):
                print("  [trace] %s: REJECTED (%s)" % (_tname, "rect" if bad_rect else "pptr"))
            if bad_rect or bad_ptr:
                n_bad_payload += 1
                if os.environ.get("EFT_DECAL_DEBUG"):
                    print("  [decals] reject %-24s rect=%s fid=%s pid=%s (%s)" % (
                        (h.get("m_Name") or "")[:24], (x1, y1, x2, y2), mfid, mpid,
                        "rect" if bad_rect else "pptr"))
                continue
            _mfile = getattr(ext[mfid - 1], "name", "?")
            if _trace:
                print("  [trace] %s: rect=%s mfid=%s(%s) mpid=%s" % (_tname, (x1, y1, x2, y2), mfid, _mfile, mpid))
            mat = material_info(_mfile, mpid)
            if _trace:
                print("  [trace] %s: material -> %s" % (_tname, mat))
            if not mat or not mat.get("tex") or not mat.get("w"):
                n_no_mat += 1
                if os.environ.get("EFT_DECAL_DEBUG"):
                    print("  [decals] no-mat %s:%s -> %r" % (_mfile, mpid, mat))
                continue
            active = active_chain(tpid) and bool(h.get("m_Enabled", 1))
            if not active:
                n_inactive += 1
            pos, rot, scl = w
            R = quat_mat(rot)
            M3 = R @ np.diag(scl)
            # RAW UNITY SPACE, exactly like scene.json's instance matrices: `assemble_bevy`
            # owns the handedness conjugation for every instance it ships, decals included, so
            # conjugating here would apply the flip TWICE. Verified against the checkpoint whose
            # photo started this: the writings projector reads x=+130.9 raw, its shields read
            # +131.2 raw, and the viewer's pick HUD shows +131.2 -- same side, same frame.
            m = [
                M3[0, 0], M3[0, 1], M3[0, 2], pos[0],
                M3[1, 0], M3[1, 1], M3[1, 2], pos[1],
                M3[2, 0], M3[2, 1], M3[2, 2], pos[2],
                0.0, 0.0, 0.0, 1.0,
            ]
            W, H = mat["w"], mat["h"]
            # Clamp the authored bleed into the texture before deriving UVs.
            cx1, cy1 = max(0.0, x1), max(0.0, y1)
            cx2, cy2 = min(float(W), x2), min(float(H), y2)
            if cx2 <= cx1 or cy2 <= cy1:
                n_bad_payload += 1
                continue
            su, sv = (cx2 - cx1) / W, (cy2 - cy1) / H
            # The rect's Y origin is the texture's BOTTOM-left -- Unity's own texture convention,
            # NOT the top-left image convention -- so V passes through unflipped. Flipping it
            # mirrored the row selection: the checkpoint plates rendered the atlas's top rows
            # ("НЕ ПРОЕХАТЬ", "ЖОПА ОБЪЕЗД") where the game paints its bottom rows ("СТОП STOP",
            # "UNTAR GO HOME"). Confirmed against a photograph of the real checkpoint.
            ou, ov = cx1 / W, cy1 / H
            if _trace:
                print("  [trace] %s: EMITTED at (%.1f, %.1f, %.1f)" % (_tname, pos[0], pos[1], pos[2]))
            instances.append({
                "mesh": QUAD_NAME,
                "m": [round(float(v), 6) for v in m],
                "kind": "mesh",
                "root": "DECALS_PROJECTED",
                "lv": lv,
                "drop": not active,
                "subs": [{
                    "tex": mat["tex"],
                    "nrm": mat.get("nrm"),
                    "col": None,
                    "sh": "p0/DeferredDecal",
                    "uv": [round(su, 6), round(sv, 6), round(ou, 6), round(ov, 6)],
                    "cut": None,
                    "n": 4,
                    "role": "decal",
                }],
            })
            lv_count += 1
        if lv_count:
            print("  [decals] level%d: %d projector(s)" % (lv, lv_count))

    # PROJECT: clip each decal against the geometry inside its box, the way Unity's deferred pass
    # does per-frame. These are Static decals on static geometry, so once is enough -- and it is
    # the only way a decal spanning surfaces at different depths (the checkpoint's two staggered
    # plates) paints all of them instead of just the nearest. EFT_DECAL_FLAT=1 keeps the old flat
    # quads for comparison.
    if os.environ.get("EFT_DECAL_FLAT") != "1":
        try:
            from decal_project import project_decals
        except ImportError:
            sys.path.insert(0, HERE)
            from decal_project import project_decals
        instances = project_decals(DS, instances)

    out = os.path.join(DS, "decals.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"instances": instances}, f)
    print("[decals] %d emitted (%d seen, %d bad payloads, %d unresolved materials, %d inactive "
          "kept with drop=true) -> %s" % (len(instances), n_seen, n_bad_payload, n_no_mat,
                                          n_inactive, out))


if __name__ == "__main__":
    main()
