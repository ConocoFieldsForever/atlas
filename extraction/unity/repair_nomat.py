"""Repair an ALREADY-EXTRACTED dataset by dropping instances whose renderer has NO material.

A Unity MeshRenderer with an empty `m_Materials` array draws nothing -- there is no shader to run.
`eft_extract_v2.py` used to emit one anyway, synthesising a fake untextured white submesh
(`sh: null, tex: null, col: [1,1,1]`), so the viewer drew flat sheets the game never shows:
interchange's four `Shoreline_Lake_Water_02_LOD0` planes, plus ~1,620 `AreaLight`/`AreaLightGI` and
per-vehicle `*_Car_light_sourse_lanterns_*_Area` placeholder quads.

The extractor now skips them at the source. This script applies the same fix to a dataset that was
already built, so a full re-extract is not needed. It re-reads the GAME to find which renderers have
no material and matches them to `scene.json` instances by (level, mesh file, world translation) --
structural, derived from the game, never a name rule. Objects whose material EXISTS but whose shader
failed to resolve keep their material and are untouched.

    python extraction/unity/repair_nomat.py --levels 54,63,64,520 --dataset <dir> [--apply]

Without --apply it reports what it would drop and changes nothing.
"""
import os, sys, json, argparse, shutil, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eft_scene_extract import trs

EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
# Match tolerance on the world translation (m). Instance matrices are rounded to 5 dp on write.
EPS = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--apply", action="store_true", help="write scene.json (default: dry run)")
    args = ap.parse_args()
    import UnityPy

    scene_fp = os.path.join(args.dataset, "scene.json")
    scene = json.load(open(scene_fp, encoding="utf-8"))
    instances = scene["instances"]
    print(f"dataset {args.dataset}: {len(instances):,} instances")

    # (lv, rounded translation) -> count of material-less renderers there
    drop_keys = {}
    for lv in [int(x) for x in args.levels.split(",")]:
        p = os.path.join(EFTDATA, f"level{lv}")
        if not os.path.exists(p):
            continue
        t0 = time.time()
        env = UnityPy.load(p)
        objs = {o.path_id: o for o in env.objects}
        tfm = {pid: o for pid, o in objs.items() if o.type.name == "Transform"}
        _I4 = np.eye(4)
        _tf, wcache, go2tf = {}, {}, {}
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
            for q in reversed(stack):
                W = W @ _tf.get(q, (0, _I4, None))[1]
                wcache[q] = W
            return W

        n = 0
        for o in env.objects:
            if o.type.name not in ("MeshRenderer", "SkinnedMeshRenderer"):
                continue
            d = o.read_typetree()
            if d.get("m_Materials"):
                continue  # has a material -> renders -> keep
            gp = (d.get("m_GameObject") or {}).get("m_PathID", 0)
            tp = go2tf.get(gp)
            if tp is None:
                continue
            W = np.asarray(world(tp))
            key = (lv, round(float(W[0, 3]), 3), round(float(W[1, 3]), 3), round(float(W[2, 3]), 3))
            drop_keys[key] = drop_keys.get(key, 0) + 1
            n += 1
        print(f"  level{lv}: {n} material-less renderer(s) in {time.time()-t0:.0f}s", flush=True)

    # Match scene.json instances. `m` is the row-major world matrix, translation at 3/7/11.
    kept, dropped = [], []
    for it in instances:
        m = it.get("m")
        key = (it.get("lv"), round(float(m[3]), 3), round(float(m[7]), 3), round(float(m[11]), 3)) \
            if m and len(m) >= 12 else None
        if key is not None and key in drop_keys:
            dropped.append(it)
        else:
            kept.append(it)

    by_mesh = {}
    for it in dropped:
        by_mesh[it["mesh"]] = by_mesh.get(it["mesh"], 0) + 1
    print(f"\nmatched {len(dropped):,} instance(s) to drop "
          f"({len(drop_keys):,} distinct material-less renderer positions found in the game):")
    for mname, c in sorted(by_mesh.items(), key=lambda kv: -kv[1])[:20]:
        print(f"   {c:6d}  {mname}")
    if len(by_mesh) > 20:
        print(f"   ... and {len(by_mesh)-20} more mesh(es)")

    if not args.apply:
        print("\n(dry run — pass --apply to write scene.json)")
        return 0
    bak = scene_fp + ".nomat.bak"
    if not os.path.exists(bak):
        shutil.copy2(scene_fp, bak)
        print(f"\nbacked up -> {bak}")
    scene["instances"] = kept
    with open(scene_fp, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, separators=(",", ":"))
    print(f"wrote {scene_fp}: {len(kept):,} instances ({len(dropped):,} removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
