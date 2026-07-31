#!/usr/bin/env python3
"""CAB dependency index for the game's asset bundles.

A container bundle (a weapon, an armor, a helmet) holds only the PREFAB. Its meshes and textures
live in OTHER bundles, referenced by PPtrs whose `m_FileID` is an index into that serialized
file's EXTERNALS list — each external naming a CAB (`cab-d5a3ee3e...`). To read the geometry you
must have the bundle that PROVIDES that CAB loaded in the same UnityPy Environment.

UnityPy will fall back to a recursive scan of the whole environment path on a miss — which walks
tens of gigabytes per lookup. So we build the CAB -> bundle map ONCE and cache it, stamped with
the game root, file count and newest mtime so it self-invalidates when the game updates.

Nothing here is authored: the mapping is read out of the bundles themselves.
"""
import json
import os
import time

import UnityPy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SA = os.environ.get("EFT_GAME_DATA",
                    r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
SA_WIN = os.path.join(SA, "StreamingAssets", "Windows")
CACHE = os.path.join(REPO, "packs", "shared", "unity_cabs.json")


def _stamp(root):
    """Cheap fingerprint of the bundle tree: count, total size, newest mtime."""
    n = 0
    total = 0
    newest = 0.0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".bundle"):
                try:
                    st = os.stat(os.path.join(dirpath, f))
                except OSError:
                    continue
                n += 1
                total += st.st_size
                newest = max(newest, st.st_mtime)
    return {"root": root, "bundles": n, "bytes": total, "newest": round(newest, 3)}


def build(root=SA_WIN, out=CACHE, verbose=True):
    """Walk every bundle and record which CABs it provides."""
    index = {}
    t0 = time.time()
    n = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith(".bundle"):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            try:
                env = UnityPy.Environment()
                env.load_file(p)
                for cab in list(getattr(env, "cabs", {}) or {}):
                    index.setdefault(str(cab).lower(), rel)
            except Exception:
                continue
            n += 1
            if verbose and n % 500 == 0:
                print(f"  [cabs] {n} bundles, {len(index)} CABs ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"stamp": _stamp(root), "cabs": index}, open(out, "w"), separators=(",", ":"))
    if verbose:
        print(f"[cabs] {len(index)} CABs from {n} bundles -> {out} ({time.time()-t0:.0f}s)")
    return index


def load(root=SA_WIN, path=CACHE, rebuild_if_stale=True, verbose=True):
    """The cached CAB->bundle map, rebuilt when the game tree changed."""
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            if not rebuild_if_stale or d.get("stamp") == _stamp(root):
                return d.get("cabs") or {}
            if verbose:
                print("[cabs] game files changed — rebuilding the CAB index")
        except Exception:
            pass
    return build(root, path, verbose=verbose)


def resolve_into(env, container_path, cabs, root=SA_WIN, limit=64):
    """Load `container_path` plus the bundles providing its CAB dependencies into `env`.

    Returns the objects that came from the CONTAINER itself — the caller must bake only those,
    since the dependency bundles carry unrelated assets (loading a weapon's neighbours pulled
    every other gun in the shared bundle into the merge).
    """
    before = {id(o) for o in env.objects}
    env.load_file(container_path)
    own = [o for o in env.objects if id(o) not in before]
    # The container's AssetBundle object lists its CAB dependencies by name.
    deps = []
    for o in own:
        if o.type.name == "AssetBundle":
            try:
                deps = list(o.read_typetree().get("m_Dependencies") or [])
            except Exception:
                deps = []
            break
    loaded = 0
    for cab in deps[:limit]:
        key = str(cab).lower()
        if getattr(env, "cabs", None) and key in {str(k).lower() for k in env.cabs}:
            continue  # already resolved
        rel = cabs.get(key)
        if not rel:
            continue
        try:
            env.load_file(os.path.join(root, rel.replace("/", os.sep)))
            loaded += 1
        except Exception:
            continue
    return own, loaded


if __name__ == "__main__":
    build()
