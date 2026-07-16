"""Extract a baked EFT Unity SCENE (level<N>) -> unique meshes (OBJ) + an instance placement JSON
(world transforms) + per-material diffuse texture, for assembling the map in Blender.

EFT bakes maps as scenes in EscapeFromTarkov_Data\\level<N> (see BuildSettings); the level file holds
the GameObject/Transform graph + MeshFilter/MeshRenderer, meshes/textures live in sharedassets and
UnityPy resolves them on deref. Interchange = level52,54-60,62,63(terrain),65.

    python extraction/unity/eft_scene_extract.py --level 62 --name interchange_outdoor [--max 4000] [--alllod]
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# portable kit: paths come from the environment (see extraction/README.md)
#   EFT_GAME_DATA   = the game's EscapeFromTarkov_Data dir (default: standard install path)
#   EFT_ASSETS_ROOT = where extracted datasets are written (default: <EFT_TARKMAP_ROOT>/../eft_assets, else ./eft_assets)
EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else
    os.path.join(os.getcwd(), "eft_assets"))


def quat_to_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    n = (x*x+y*y+z*z+w*w) ** 0.5 or 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def trs(t):
    p = t.m_LocalPosition; r = t.m_LocalRotation; s = t.m_LocalScale
    M = np.eye(4)
    R = quat_to_mat(r) @ np.diag([s.x, s.y, s.z])
    M[:3, :3] = R; M[:3, 3] = [p.x, p.y, p.z]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max", type=int, default=0, help="cap instances (0=all) for a test run")
    ap.add_argument("--alllod", action="store_true", help="keep all LODs (default: LOD0/no-LOD only)")
    ap.add_argument("--tex", action="store_true", help="also export diffuse textures")
    args = ap.parse_args()
    import UnityPy

    t0 = time.time()
    env = UnityPy.load(os.path.join(EFTDATA, f"level{args.level}"))
    print(f"loaded level{args.level} in {time.time()-t0:.1f}s")
    out = os.path.join(OUTROOT, args.name); md = os.path.join(out, "meshes")
    os.makedirs(md, exist_ok=True)

    # index transforms + gameobjects by path_id for parent walking
    tfm = {}; go_tfm = {}
    for o in env.objects:
        if o.type.name == "Transform":
            tfm[o.path_id] = o
    print(f"{len(tfm)} transforms indexed ({time.time()-t0:.0f}s)")

    wcache = {}
    def world(tf_obj):
        pid = tf_obj.path_id
        if pid in wcache: return wcache[pid]
        chain = []; cur = tf_obj
        guard = 0
        while cur is not None and guard < 256:
            guard += 1
            t = cur.read(); chain.append(t)
            fp = getattr(t, "m_Father", None)
            if fp is None or getattr(fp, "path_id", 0) == 0: break
            nxt = tfm.get(fp.path_id)
            if nxt is None: break
            cur = nxt
        W = np.eye(4)
        for t in reversed(chain): W = W @ trs(t)
        wcache[pid] = W
        return W

    # material -> diffuse texture name
    mat_diff = {}
    def diffuse_of(mr):
        try:
            mats = mr.m_Materials
            if not mats or getattr(mats[0], "path_id", 0) == 0: return None
            mid = mats[0].path_id
            if mid in mat_diff: return mat_diff[mid]
            mat = mats[0].read(); name = None
            tenvs = mat.m_SavedProperties.m_TexEnvs
            items = tenvs.items() if hasattr(tenvs, "items") else tenvs
            for k, tenv in items:
                if str(k) in ("_MainTex", "_Diffuse", "_BaseMap", "_AlbedoMap"):
                    tx = getattr(tenv, "m_Texture", None)
                    if tx is not None and getattr(tx, "path_id", 0):
                        name = tx.read().m_Name; break
            mat_diff[mid] = name; return name
        except Exception:
            return None

    # gameobject path_id -> its transform path_id  (via Transform.m_GameObject)
    go2tf = {}
    for pid, o in tfm.items():
        try:
            t = o.read(); go2tf[t.m_GameObject.path_id] = pid
        except Exception: pass

    exported = {}                      # mesh path_id -> obj filename
    instances = []
    tex_want = set()
    n = 0; t1 = time.time()
    for o in env.objects:
        if o.type.name != "MeshRenderer": continue
        try:
            mr = o.read(); go_pid = mr.m_GameObject.path_id
            # sibling MeshFilter mesh: read GameObject components
            go = mr.m_GameObject.read()
            mesh_pptr = None
            for comp in go.m_Component:
                cp = comp[1] if isinstance(comp, (list, tuple)) else comp.component
                co = cp.read()
                if co.__class__.__name__ == "MeshFilter":
                    mesh_pptr = co.m_Mesh; break
            if mesh_pptr is None or getattr(mesh_pptr, "path_id", 0) == 0: continue
            mid = mesh_pptr.path_id
            if mid not in exported:
                mesh = mesh_pptr.read()
                nm = getattr(mesh, "m_Name", f"m{mid}") or f"m{mid}"
                if (not args.alllod) and ("_LOD" in nm) and ("_LOD0" not in nm):
                    exported[mid] = None                      # skip non-LOD0
                else:
                    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in nm) + f"_{abs(mid)%100000}"
                    try:
                        open(os.path.join(md, safe + ".obj"), "w").write(mesh.export())
                        exported[mid] = safe + ".obj"
                    except Exception:
                        exported[mid] = None
            fn = exported.get(mid)
            if not fn: continue
            tfpid = go2tf.get(go_pid)
            if tfpid is None: continue
            W = world(tfm[tfpid])
            dt = diffuse_of(mr)
            if dt: tex_want.add(dt)
            instances.append({"mesh": fn, "m": [round(float(v), 5) for v in W.flatten()], "tex": dt})
            n += 1
            if n % 2000 == 0: print(f"  {n} instances, {len([v for v in exported.values() if v])} unique meshes ({time.time()-t1:.0f}s)")
            if args.max and n >= args.max: break
        except Exception as e:
            continue
    uniq = len([v for v in exported.values() if v])
    json.dump({"instances": instances, "up": "unity"}, open(os.path.join(out, "scene.json"), "w"))
    print(f"\nDONE: {n} instances, {uniq} unique meshes, {len(tex_want)} diffuse textures -> {out}")

    if args.tex:
        td = os.path.join(out, "tex"); os.makedirs(td, exist_ok=True); got = 0
        for o in env.objects:
            if o.type.name != "Texture2D": continue
            try:
                tx = o.read()
                if tx.m_Name in tex_want:
                    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tx.m_Name)
                    img = tx.image
                    if img: img.save(os.path.join(td, safe + ".png")); got += 1
            except Exception: pass
        print(f"  exported {got} textures")


if __name__ == "__main__":
    main()
