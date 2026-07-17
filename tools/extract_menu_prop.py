"""Extract the REAL game CCTV prop (Street_Camera_01) for the start menu's 3D decor.

Writes packs/shared/menu/{camera.bin, camera.png, camera.json}. MACHINE-LOCAL ONLY:
packs/ is gitignored -- this asset must never be committed or shipped. The viewer
(menu_fx.rs) loads camera.bin/camera.png at menu startup and falls back to the
vector-drawn camera when they are absent or corrupt.

Sources, tried in order:
  1. DATASET (fast copy): <EFT_ASSETS_ROOT>/lighthouse/meshes/Street_Camera_01_LOD0__*.obj
     + tex/CameraStreet_d__*.png. The mesh->texture pairing is confirmed by the lighthouse
     scene.json instance: subs[0].tex == "CameraStreet_d__sharedassets2_432" (single submesh).
  2. GAME FILES (fresh machine, no dataset): UnityPy name-scan of the game's
     sharedassetsN.assets for Mesh "Street_Camera_01_LOD0" + Texture2D "CameraStreet_d"
     (sharedassets2.assets first -- the known texture home -- then ascending), bounded by
     --budget seconds. The mesh is exported through UnityPy's OBJ exporter, i.e. the SAME
     convention as dataset OBJs (X-flipped, reversed winding), so one parser serves both.

The OBJ is parsed with the vendored tarkmap_core.objio.load_obj, welded to unified
vertices, CENTERED (bounds-center subtracted -- per tarkov-unity-extraction: never
decompose/re-transform, raw verts only), and the UV V-flip that assemble_bevy.py bakes for
in-place PNG textures (manifest conventions.uvVFlipBaked) is applied here too.

camera.bin layout (little-endian):
  [u32 vert_count][u32 index_count]
  [pos f32x3 * n][normal f32x3 * n][uv f32x2 * n][indices u32 * m]

Called by the viewer at menu startup (menu.rs ensure_menu_prop) when the files are
missing; also runnable by hand:  python tools/extract_menu_prop.py
ASCII output only (cp1252 console).
"""
import argparse
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from eft_pipeline.tarkmap_core.objio import load_obj  # noqa: E402

MESH_NAME = "Street_Camera_01_LOD0"
TEX_NAME = "CameraStreet_d"
# World scale of this mesh's lighthouse placements (scene.json instance matrix, uniform
# 1.503): raw-mesh max extent * this = the size the prop has in game, recorded as a hint.
INSTANCE_SCALE = 1.503

# Same env-driven anchors as tools/build_map.py (legacy dev-machine defaults).
TK = os.environ.get("EFT_TARKMAP_ROOT", r"C:\Users\user\beamng_blender_pipeline\tarkmap")
ASSETS = os.environ.get("EFT_ASSETS_ROOT") or os.path.normpath(
    os.path.join(TK, os.pardir, "eft_assets"))
GAME_DEFAULT = os.environ.get(
    "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")


def log(msg):
    print("[menu-prop] " + msg, flush=True)


def read_vn(path):
    """Per-vertex normals from the OBJ's vn lines (UnityPy exports f v/vt/vn with all
    three indices IDENTICAL, so vn aligns 1:1 with v -- verified on the dataset OBJs)."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("vn "):
                out.append(line[3:].split()[:3])
    return np.array(out, np.float32).reshape(-1, 3) if out else np.zeros((0, 3), np.float32)


def weld(V, VT, VN, F):
    """(v,uv) index pairs -> unified vertex streams + flat index list."""
    aligned = len(VN) == len(V)
    key2new, P, U, N, I = {}, [], [], [], []
    for tri in F:
        for vi, ti in tri:
            k = (int(vi), int(ti))
            j = key2new.get(k)
            if j is None:
                j = len(P)
                key2new[k] = j
                P.append(V[vi])
                U.append(VT[ti] if 0 <= ti < len(VT) else (0.0, 0.0))
                N.append(VN[vi] if aligned else (0.0, 0.0, 0.0))
            I.append(j)
    P = np.array(P, np.float32).reshape(-1, 3)
    U = np.array(U, np.float32).reshape(-1, 2)
    N = np.array(N, np.float32).reshape(-1, 3)
    I = np.array(I, np.uint32)
    if not aligned:  # defensive: OBJ without vn -> area-weighted face-normal accumulation
        tri = I.reshape(-1, 3)
        fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
        for c in range(3):
            np.add.at(N, tri[:, c], fn)
        L = np.linalg.norm(N, axis=1, keepdims=True)
        N = np.where(L > 1e-12, N / np.maximum(L, 1e-12), [0.0, 1.0, 0.0]).astype(np.float32)
    return P, U, N, I


def parse_obj(ds_root, fn):
    """Dataset-convention OBJ -> welded, CENTERED streams. Returns (P,N,U,I,size,center)."""
    r = load_obj(ds_root, fn)
    if r is None:
        return None
    V, VT, F = r
    if not len(V) or not len(F):
        return None
    VN = read_vn(os.path.join(ds_root, "meshes", fn))
    P, U, N, I = weld(V, VT, VN, F)
    lo, hi = P.min(0), P.max(0)
    center = (lo + hi) * 0.5
    P = P - center
    # assemble_bevy.py bakes a V-flip for in-place PNG textures (conventions.uvVFlipBaked);
    # this material's _ST tiling is [1,1,0,0] (scene.json), so the flip is the whole xform.
    U = U.copy()
    U[:, 1] = 1.0 - U[:, 1]
    return P, N, U, I, (hi - lo), center


def write_pack(out_dir, P, N, U, I, size, center, png_src, source, mode, tex_desc):
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "camera.bin")
    png_path = os.path.join(out_dir, "camera.png")
    json_path = os.path.join(out_dir, "camera.json")

    data = struct.pack("<II", len(P), len(I))
    data += P.astype("<f4").tobytes() + N.astype("<f4").tobytes()
    data += U.astype("<f4").tobytes() + I.astype("<u4").tobytes()
    tmp = bin_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, bin_path)

    if png_src != png_path:
        tmp = png_path + ".tmp"
        shutil.copyfile(png_src, tmp)
        os.replace(tmp, png_path)

    meta = {
        "scale_hint": round(float(size.max()) * INSTANCE_SCALE, 4),
        "source": source,
        "mode": mode,
        "mesh": MESH_NAME,
        "texture": tex_desc,
        "verts": int(len(P)),
        "indices": int(len(I)),
        "raw_size": [round(float(v), 5) for v in size],
        "raw_center": [round(float(v), 5) for v in center],
        "uv_v_flipped": True,
        "confirmed_by": "lighthouse scene.json: Street_Camera_01_LOD0 subs[0].tex == "
                        "CameraStreet_d__sharedassets2_432 (single submesh)",
        "note": "machine-local extraction; packs/ is gitignored - never commit or ship",
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    lo, hi = -size * 0.5, size * 0.5
    log("wrote %s (%d bytes): %d verts, %d indices (%d tris)"
        % (bin_path, len(data), len(P), len(I), len(I) // 3))
    log("centered bounds lo [%.3f %.3f %.3f] hi [%.3f %.3f %.3f] (raw units)"
        % (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]))
    log("wrote %s (%d bytes)" % (png_path, os.path.getsize(png_path)))
    log("scale_hint %.3f m (in-game size at instance scale %.3f)"
        % (meta["scale_hint"], INSTANCE_SCALE))


def try_dataset(ds_root, out_dir):
    """Fast path: the already-extracted lighthouse dataset (OBJ + PNG copy)."""
    objs = sorted(glob.glob(os.path.join(ds_root, "meshes", MESH_NAME + "__*.obj")))
    pngs = sorted(glob.glob(os.path.join(ds_root, "tex", TEX_NAME + "__*.png")))
    if not objs or not pngs:
        log("dataset not usable at %s (obj:%d tex:%d)" % (ds_root, len(objs), len(pngs)))
        return False
    parsed = parse_obj(ds_root, os.path.basename(objs[0]))
    if parsed is None:
        log("dataset OBJ parse failed: " + objs[0])
        return False
    P, N, U, I, size, center = parsed
    write_pack(out_dir, P, N, U, I, size, center, pngs[0], objs[0], "dataset",
               os.path.basename(pngs[0]))
    log("source: dataset (" + ds_root + ")")
    return True


def try_game(game, out_dir, budget):
    """Fresh-machine path: UnityPy name-scan of the game's sharedassetsN.assets."""
    if not os.path.isfile(os.path.join(game, "sharedassets2.assets")):
        log("game install not usable at " + game)
        return False
    try:
        import UnityPy
    except ImportError:
        # Same interpreter contract as tools/build_map.py: EFT_PY_UNITY > legacy anaconda.
        alt = os.environ.get("EFT_PY_UNITY") or r"C:\Users\user\anaconda3\python.exe"
        if os.path.isfile(alt) and not os.environ.get("_MENU_PROP_REEXEC"):
            log("UnityPy missing in this python; retrying via " + alt)
            env = dict(os.environ, _MENU_PROP_REEXEC="1")
            r = subprocess.run(
                [alt, os.path.abspath(__file__), "--game", game, "--out", out_dir,
                 "--game-only", "--budget", str(budget)], env=env)
            return r.returncode == 0
        log("UnityPy not available; cannot extract from game files")
        return False

    deadline = time.time() + budget
    # sharedassets2 first (the known CameraStreet_d home on the reference install),
    # then ascending -- scene-shared assets usually keep mesh+texture in one file.
    nums = []
    for fn in os.listdir(game):
        if fn.startswith("sharedassets") and fn.endswith(".assets"):
            try:
                nums.append(int(fn[len("sharedassets"):-len(".assets")]))
            except ValueError:
                pass
    order = [2] + [n for n in sorted(nums) if n != 2] if 2 in nums else sorted(nums)

    mesh_obj = tex_obj = None
    mesh_src = tex_src = None
    for n in order:
        if mesh_obj is not None and tex_obj is not None:
            break
        if time.time() > deadline:
            log("budget (%.0fs) exhausted at sharedassets%d; giving up" % (budget, n))
            break
        path = os.path.join(game, "sharedassets%d.assets" % n)
        try:
            env = UnityPy.load(path)
        except Exception as e:
            log("skip sharedassets%d: %s" % (n, e))
            continue
        checked = 0
        for o in env.objects:
            tn = o.type.name
            if tn not in ("Mesh", "Texture2D"):
                continue
            checked += 1
            if checked % 2000 == 0 and time.time() > deadline:
                break
            try:
                nm = o.peek_name()
            except Exception:
                continue
            if tn == "Mesh" and mesh_obj is None and nm == MESH_NAME:
                mesh_obj = o.read()
                mesh_src = "%s (path_id %d)" % (os.path.basename(path), o.path_id)
                log("found Mesh %s in %s" % (MESH_NAME, mesh_src))
            elif tn == "Texture2D" and tex_obj is None and nm == TEX_NAME:
                tex_obj = o.read()
                tex_src = "%s (path_id %d)" % (os.path.basename(path), o.path_id)
                log("found Texture2D %s in %s" % (TEX_NAME, tex_src))
            if mesh_obj is not None and tex_obj is not None:
                break
    if mesh_obj is None or tex_obj is None:
        log("game scan incomplete (mesh:%s tex:%s)"
            % ("ok" if mesh_obj is not None else "missing",
               "ok" if tex_obj is not None else "missing"))
        return False

    obj_text = mesh_obj.export()  # UnityPy OBJ: same X-flip/winding convention as the dataset
    if not isinstance(obj_text, str) or not obj_text:
        log("mesh export produced no OBJ data")
        return False
    tmp_root = os.path.join(out_dir, "_tmp")
    os.makedirs(os.path.join(tmp_root, "meshes"), exist_ok=True)
    tmp_obj = os.path.join(tmp_root, "meshes", "cam.obj")
    with open(tmp_obj, "w", encoding="utf-8") as fh:
        fh.write(obj_text)
    parsed = parse_obj(tmp_root, "cam.obj")
    if parsed is None:
        log("game OBJ parse failed")
        return False
    P, N, U, I, size, center = parsed

    png_path = os.path.join(out_dir, "camera.png")
    img = tex_obj.image
    if img is None:
        log("texture decode failed")
        return False
    tmp_png = png_path + ".tmp"
    img.save(tmp_png, format="PNG")  # explicit: PIL can't infer a format from ".tmp"
    os.replace(tmp_png, png_path)

    write_pack(out_dir, P, N, U, I, size, center, png_path,
               os.path.join(game, mesh_src.split(" ")[0]), "gamefiles",
               "%s from %s" % (TEX_NAME, tex_src))
    shutil.rmtree(tmp_root, ignore_errors=True)
    log("source: gamefiles (" + game + ")")
    return True


def main():
    ap = argparse.ArgumentParser(description="extract the menu CCTV prop (local-only)")
    ap.add_argument("--out", default=os.path.join(ROOT, "packs", "shared", "menu"))
    ap.add_argument("--dataset", default=os.path.join(ASSETS, "lighthouse"))
    ap.add_argument("--game", default=GAME_DEFAULT)
    ap.add_argument("--budget", type=float, default=20.0,
                    help="max seconds for the game-file scan fallback")
    ap.add_argument("--force", action="store_true", help="re-extract even if files exist")
    ap.add_argument("--game-only", action="store_true",
                    help="skip the dataset path (used by the UnityPy re-exec)")
    a = ap.parse_args()

    have = all(os.path.isfile(os.path.join(a.out, f)) for f in ("camera.bin", "camera.png"))
    if have and not a.force:
        log("camera.bin + camera.png already present in %s (use --force to redo)" % a.out)
        return 0

    if not a.game_only and try_dataset(a.dataset, a.out):
        return 0
    if try_game(a.game, a.out, a.budget):
        return 0
    log("FAILED: no usable source (dataset %s, game %s) - viewer keeps the vector camera"
        % (a.dataset, a.game))
    return 1


if __name__ == "__main__":
    sys.exit(main())
