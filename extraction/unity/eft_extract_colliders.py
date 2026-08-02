"""EFT physics colliders -> `colliders.json` (+ collider-only meshes) for the nav bake.

WHY THIS EXISTS
---------------
`eft_extract_v2.py` walks MeshRenderers, so the dataset only ever contains geometry you can SEE.
The world the player actually collides with is the PHYSICS world, and a large part of it has no
renderer at all. Interchange's `Swamp_collider` is the canonical case: its GameObject is
`Transform + MeshFilter + BoxCollider + MonoBehaviour x2` -- a MeshFilter with NO MeshRenderer, so
it is invisible, never extracted, and every route we bake walks straight through 5,763 of them.

Level63 (one of interchange's fourteen levels) alone holds:
    31,015 SphereCollider   8,253 BoxCollider   7,732 MeshCollider   6,360 CapsuleCollider
    4 TerrainCollider
None of it reaches the nav bake today.

This is also what Unity itself does: `NavMeshSurface.m_UseGeometry` selects RenderMeshes or
PhysicsColliders as the bake input, and the agent descriptors that drive the bake
(`NavMeshProjectSettings`, extracted by `eft_extract_nav.py`) are defined against that world.

WHAT IT EMITS
-------------
`colliders.json` next to `scene.json`:
    {"colliders": [ {t, m[16], ...shape..., lv, root, go, vis, nav_ignore, nav_area}, ... ],
     "counts": {...}}
`m` is the RAW Unity row-major world matrix, exactly like `scene.json` instances -- the same
downstream handedness conjugation places both (see the tarkov-unity-extraction skill S1/S3). Do NOT
bake any coordinate flip in here.

Shapes carry Unity's LOCAL shape params; the world matrix supplies position/rotation/scale:
    box     c[3] centre, s[3] size          sphere  c[3] centre, r radius
    capsule c[3] centre, r radius, h height, d direction (0=X,1=Y,2=Z)
    mesh    mesh <obj filename>, convex

FLAGS (facts, not policy -- the consumer decides)
    lyr         GameObject `m_Layer`. THE key discriminator: EFT separates MOVEMENT collision from
                HIT collision, so the layer says what a collider is actually for. From TagManager
                (an engine type, so readable):
                    9 DoorLowPolyCollider  11 Terrain  12 HighPolyCollider  18 LowPolyCollider
                    13 Triggers  26 Foliage  29 LevelBorder  30 TransparentCollider  31 Grass
                `LowPolyCollider` (+ Terrain + doors + LevelBorder) is the world you walk against;
                `HighPolyCollider` is the finer shell used for ballistics/hit detection.
    trig        `m_IsTrigger`. A Unity trigger has NO contact response -- it fires OnTriggerEnter
                and nothing more -- so a trigger never blocks movement. Interchange's 5,763
                `Swamp_collider` boxes are `m_IsTrigger=true` on layer 13 `Triggers`: swamp
                splash/sound/slow volumes, not solids. Recorded rather than dropped so a consumer
                that wants them (e.g. to mark "slow" terrain cost) still can.
    vis         the GameObject also has a MeshRenderer (i.e. already in the render pack)
    nav_ignore  `Unity.AI.Navigation.NavMeshModifier.m_IgnoreFromBuild` -- the GAME excludes this
                object from its BOT navmesh.
    nav_area    `m_Area` when `m_OverrideArea` is set; indexes NavMeshProjectSettings.areas
                (0 Walkable, 1 Not Walkable, 2 Jump, 3 Sitdown, 4 Danger, 5 Terrain).

Nothing is dropped for being a trigger -- every collider is emitted with its `lyr`/`trig` facts and
the nav bake decides. Only disabled colliders (`m_Enabled=0`) and objects inactive in the hierarchy
are skipped: those are not in the physics world at all.

    python extraction/unity/eft_extract_colliders.py --levels 54,63 --name interchange_v2
"""
import os, sys, json, argparse, time, struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eft_scene_extract import trs

EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else
    os.path.join(os.getcwd(), "eft_assets"))

COLLIDER_TYPES = ("BoxCollider", "SphereCollider", "CapsuleCollider", "MeshCollider")
# NavMeshModifier payload: 28 B MonoBehaviour header, then m_Name (int32 length, always 0 here),
# then eight int32s. Verified against all 5,764 instances in level63 -- see the module docstring of
# eft_extract_nav.py for the derivation and the `terrain -> area 5 == "Terrain"` cross-check.
NAVMOD_FIELD_OFF = 32


def g(o, *names, default=None):
    for n in names:
        v = getattr(o, n, None)
        if v is not None:
            return v
    return default


def san(s):
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))[:96]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None, help="dataset dir (default <OUTROOT>/<name>)")
    args = ap.parse_args()

    import UnityPy

    out = args.out or os.path.join(OUTROOT, args.name)
    md = os.path.join(out, "meshes")
    os.makedirs(md, exist_ok=True)

    colliders = []
    counts = {}
    exported = {}
    t_all = time.time()

    # [SUBPROGRESS] denominators, byte-weighted like the parallel extractor's: streets' heavy
    # levels take 30-60s each while most take <1s, so a level COUNT would misrepresent the pass
    # exactly the way it did for extraction. Without this the whole (up to ~1h on streets) pass
    # was silent to the loader bar, which parked on the stage-1 "no sub-signal" fallback — the
    # frozen "28%" a first-time builder reads as a hang.
    _levels = [int(x) for x in args.levels.split(",")]
    _lv_w = {}
    for _lv in _levels:
        try:
            _lv_w[_lv] = os.path.getsize(os.path.join(EFTDATA, f"level{_lv}")) + 1
        except OSError:
            _lv_w[_lv] = 1
    _w_total, _w_done, _n_done = sum(_lv_w.values()), 0, 0

    for lv in _levels:
        path = os.path.join(EFTDATA, f"level{lv}")
        if not os.path.exists(path):
            print(f"level{lv} missing", flush=True)
            continue
        t0 = time.time()
        env = UnityPy.load(path)
        sfile = list(env.files.values())[0]
        externals = [e.path for e in sfile.externals]
        objs = {o.path_id: o for o in env.objects}
        tfm = {pid: o for pid, o in objs.items() if o.type.name == "Transform"}

        # ---- transform machinery: identical semantics to eft_extract_v2 (memoised PER NODE, never
        # per leaf -- a per-leaf memo is quadratic on deep hierarchies, see the skill's S7 entry).
        _I4 = np.eye(4)
        _tf = {}
        wcache = {}
        go2tf = {}
        for pid, o in tfm.items():
            try:
                t = o.read()
                fp = getattr(t, "m_Father", None)
                fpid = getattr(fp, "path_id", 0) if fp is not None else 0
                goptr = getattr(t, "m_GameObject", None)
                _tf[pid] = (fpid if fpid in tfm else 0, trs(t), goptr)
            except Exception:
                continue
            gp = getattr(goptr, "path_id", None) if goptr is not None else None
            if gp is not None:
                go2tf[gp] = pid

        def world(tf_pid):
            W = wcache.get(tf_pid)
            if W is not None:
                return W
            stack, cur = [], tf_pid
            while cur and cur not in wcache and len(stack) < 256:
                stack.append(cur)
                cur = _tf.get(cur, (0, _I4, None))[0]
            W = wcache.get(cur) if cur else None
            if W is None:
                W = np.eye(4)
            for p in reversed(stack):
                W = W @ _tf.get(p, (0, _I4, None))[1]
                wcache[p] = W
            return W

        _goc = {}

        def _go_entry(tp):
            """(name, activeSelf, layer) for Transform `tp`'s own GameObject, read at most once."""
            v = _goc.get(tp)
            if v is None:
                nm, act, lyr = "", True, 0
                goptr = _tf.get(tp, (0, _I4, None))[2]
                if goptr is not None:
                    try:
                        go = goptr.read()
                    except Exception:
                        go = None
                    if go is not None:
                        try:
                            nm = go.m_Name
                        except Exception:
                            nm = ""
                        try:
                            act = bool(g(go, "m_IsActive", default=True))
                        except Exception:
                            act = True
                        try:
                            lyr = int(g(go, "m_Layer", default=0) or 0)
                        except Exception:
                            lyr = 0
                v = _goc[tp] = (nm, act, lyr)
            return v

        rcache, ahcache = {}, {}

        def root_of(go_pid):
            if go_pid in rcache:
                return rcache[go_pid]
            tp, root, gd = go2tf.get(go_pid), "", 0
            while tp and gd < 256:
                gd += 1
                nm = _go_entry(tp)[0]
                if nm:
                    root = nm
                tp = _tf.get(tp, (0, _I4, None))[0] or None
            rcache[go_pid] = root
            return root

        def active_in_hierarchy(go_pid):
            if go_pid in ahcache:
                return ahcache[go_pid]
            tp, ok, gd = go2tf.get(go_pid), True, 0
            while tp and gd < 256:
                gd += 1
                if not _go_entry(tp)[1]:
                    ok = False
                    break
                tp = _tf.get(tp, (0, _I4, None))[0] or None
            ahcache[go_pid] = ok
            return ok

        # ---- pass 1: per-GameObject component index (renderer presence + NavMeshModifier flags).
        # MonoScript is an ENGINE type with a hardcoded type tree, so the real C# class name is
        # readable even though global-metadata.dat is encrypted -- that is how NavMeshModifier is
        # identified without guessing at MonoBehaviour payload shapes.
        _cls = {}

        def script_class(raw):
            fid, pid = struct.unpack_from("<iq", raw, 16)
            key = (fid, pid)
            if key in _cls:
                return _cls[key]
            n = None
            try:
                if fid == 0:
                    tbl = objs
                else:
                    p = os.path.join(EFTDATA, os.path.basename(externals[fid - 1]))
                    if p not in _cls:
                        _cls[p] = {x.path_id: x for x in UnityPy.load(p).objects}
                    tbl = _cls[p]
                so = tbl.get(pid)
                if so is not None and so.type.name == "MonoScript":
                    d = so.read_typetree()
                    n = f"{d.get('m_Namespace') or ''}.{d.get('m_ClassName')}".lstrip(".")
            except Exception:
                pass
            _cls[key] = n
            return n

        has_renderer = set()
        navmod = {}  # go_pid -> (ignore_from_build, area or None)
        for o in env.objects:
            tn = o.type.name
            if tn in ("MeshRenderer", "SkinnedMeshRenderer"):
                try:
                    r = o.read()
                    has_renderer.add(r.m_GameObject.path_id)
                except Exception:
                    pass
            elif tn == "MonoBehaviour":
                try:
                    raw = bytes(o.get_raw_data())
                except Exception:
                    continue
                if script_class(raw) != "Unity.AI.Navigation.NavMeshModifier":
                    continue
                if len(raw) < NAVMOD_FIELD_OFF + 32:
                    continue
                f = struct.unpack_from("<8i", raw, NAVMOD_FIELD_OFF)
                override_area, area, ignore = f[0], f[1], f[4]
                gp = struct.unpack_from("<q", raw, 4)[0]
                navmod[gp] = (ignore == 1, area if override_area == 1 else None)

        def export_collider_mesh(mesh_pptr):
            """Export a MeshCollider's mesh to OBJ (same dir/naming as the render meshes, so the
            assembler resolves both from one place). Returns the filename or None."""
            fid = g(mesh_pptr, "file_id", "m_FileID", default=0) or 0
            key = (lv, fid, mesh_pptr.path_id)
            if key in exported:
                return exported[key]
            fn = None
            try:
                mesh = mesh_pptr.read()
                nm = g(mesh, "m_Name", default=f"m{mesh_pptr.path_id}") or f"m{mesh_pptr.path_id}"
                fn = f"{san(nm)}__{key[0]}_{key[1]}_{key[2]}.obj"
                fp = os.path.join(md, fn)
                if (not os.path.exists(fp)) or os.path.getsize(fp) == 0:
                    data = mesh.export()
                    if isinstance(data, str) and data:
                        # utf-8: EFT ships Cyrillic mesh names; cp1252 would truncate the file to 0 B.
                        with open(fp, "w", encoding="utf-8") as fh:
                            fh.write(data)
                    else:
                        fn = None
            except Exception:
                fn = None
            exported[key] = fn
            return fn

        n_lv = 0
        skipped = {"disabled": 0, "inactive": 0, "no_transform": 0, "no_mesh": 0}
        for o in env.objects:
            tn = o.type.name
            if tn not in COLLIDER_TYPES:
                continue
            try:
                d = o.read_typetree()
            except Exception:
                continue
            gp = (d.get("m_GameObject") or {}).get("m_PathID", 0)
            if not gp:
                continue
            # Triggers are KEPT (flagged `trig`) -- see the module docstring. A disabled or
            # inactive collider genuinely is not in the physics world, so those are skipped.
            if not d.get("m_Enabled", 1):
                skipped["disabled"] += 1
                continue
            if not active_in_hierarchy(gp):
                skipped["inactive"] += 1
                continue
            tp = go2tf.get(gp)
            if tp is None:
                skipped["no_transform"] += 1
                continue
            W = world(tp)

            rec = {"m": [round(float(v), 5) for v in np.asarray(W).flatten()], "lv": lv}
            if tn == "BoxCollider":
                c, s = d.get("m_Center") or {}, d.get("m_Size") or {}
                rec["t"] = "box"
                rec["c"] = [float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 0))]
                rec["s"] = [float(s.get("x", 1)), float(s.get("y", 1)), float(s.get("z", 1))]
            elif tn == "SphereCollider":
                c = d.get("m_Center") or {}
                rec["t"] = "sphere"
                rec["c"] = [float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 0))]
                rec["r"] = float(d.get("m_Radius", 0.5))
            elif tn == "CapsuleCollider":
                c = d.get("m_Center") or {}
                rec["t"] = "capsule"
                rec["c"] = [float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 0))]
                rec["r"] = float(d.get("m_Radius", 0.5))
                rec["h"] = float(d.get("m_Height", 2.0))
                rec["d"] = int(d.get("m_Direction", 1))
            else:  # MeshCollider
                mp = getattr(o.read(), "m_Mesh", None)
                fn = export_collider_mesh(mp) if mp is not None else None
                if not fn:
                    skipped["no_mesh"] += 1
                    continue
                rec["t"] = "mesh"
                rec["mesh"] = fn
                rec["convex"] = bool(d.get("m_Convex", False))

            ign, area = navmod.get(gp, (False, None))
            nm, _act, lyr = _go_entry(tp)
            rec["go"] = nm
            rec["root"] = root_of(gp)
            rec["lyr"] = lyr
            rec["vis"] = gp in has_renderer
            if d.get("m_IsTrigger"):
                rec["trig"] = True
            if ign:
                rec["nav_ignore"] = True
            if area is not None:
                rec["nav_area"] = area
            colliders.append(rec)
            counts[tn] = counts.get(tn, 0) + 1
            n_lv += 1

        print(f"  level{lv}: {n_lv} colliders in {time.time()-t0:.1f}s  skipped={skipped}", flush=True)
        # Machine-readable ratio LAST on the line (the viewer parses the final whitespace token).
        # RAW bytes — a rounded MB collapses small totals to a degenerate 0.0/0.0.
        _w_done += _lv_w.get(lv, 1)
        _n_done += 1
        print(f"[SUBPROGRESS] colliders levels {_n_done}/{len(_levels)} "
              f"bytes {_w_done}/{_w_total}", flush=True)

    # Unity layer names come from TagManager (an engine type -> readable despite the encrypted
    # il2cpp metadata). Shipped alongside the colliders so the consumer never hardcodes indices.
    layers = read_layer_names()

    fp = os.path.join(out, "colliders.json")
    invisible = sum(1 for c in colliders if not c.get("vis"))
    ignored = sum(1 for c in colliders if c.get("nav_ignore"))
    triggers = sum(1 for c in colliders if c.get("trig"))
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump({"colliders": colliders, "counts": counts, "layers": layers},
                  fh, separators=(",", ":"))
    print(f"\nwrote {fp}: {len(colliders)} colliders {counts}", flush=True)
    print(f"  {invisible} have NO renderer (invisible collision the render pack never saw)", flush=True)
    print(f"  {triggers} are triggers (no contact response -- never block movement)", flush=True)
    print(f"  {ignored} are NavMeshModifier m_IgnoreFromBuild (excluded from the GAME's bot navmesh)",
          flush=True)
    by_layer = {}
    for c in colliders:
        k = (c.get("lyr", 0), bool(c.get("trig")))
        by_layer[k] = by_layer.get(k, 0) + 1
    print("\n  by layer (solid / trigger):", flush=True)
    for (l, t), n in sorted(by_layer.items(), key=lambda kv: -kv[1]):
        nm = layers.get(str(l)) or layers.get(l) or f"<{l}>"
        print(f"    {n:7d}  layer {l:2d} {nm:<22} {'TRIGGER' if t else 'solid'}", flush=True)
    print(f"\n  total {time.time()-t_all:.1f}s", flush=True)


def read_layer_names():
    """Unity layer index -> name, from globalgamemanagers' TagManager."""
    import UnityPy
    out = {}
    try:
        env = UnityPy.load(os.path.join(EFTDATA, "globalgamemanagers"))
        for o in env.objects:
            if o.type.name != "TagManager":
                continue
            for i, n in enumerate(o.read_typetree().get("layers") or []):
                if n:
                    out[str(i)] = n
            break
    except Exception as e:
        print(f"  (layer names unavailable: {e})", flush=True)
    return out


if __name__ == "__main__":
    main()
