#!/usr/bin/env python3
"""Assemble a weapon (or any equipment item) exactly the way the game does -> `.eftweap` pack.

THE CONTRACT, all game-derived, nothing guessed:

- Each item TEMPLATE (`packs/shared/item_templates.json`, BSG's own data) declares
  `_props.Prefab.path` — the bundle holding that item's model — and `_props.Slots[]`, each named
  `mod_muzzle`, `mod_magazine`, `mod_reciever`, ... with a Filter of allowed child items.
- Each PREFAB contains transform GameObjects with EXACTLY those slot names (verified on
  weapon_izhmash_ak74n_545x39_container: mod_muzzle / mod_gas_block / mod_handguard /
  mod_magazine / mod_pistol_grip / mod_reciever / mod_sight_rear / mod_stock / mod_charge).
- So the game builds a gun by parenting each installed mod's prefab AT the parent's node of the
  same name, recursively. This script replays that: walk the build tree, load each bundle, bake
  every renderer's mesh by its local-to-root matrix, and emit ONE merged mesh + materials.

WHICH mods are installed comes from the item catalog's own default preset (BSG's factory build),
so an AK-74N gets the receiver/stock/magazine BSG ships it with.

usage:
  build_weapon.py --preset ak74n_default          # a registered build
  build_weapon.py --item 5644bd2b4bdc2d3b4c8b4572 # bare item, default preset if any
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import UnityPy

import unity_deps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SA = os.environ.get("EFT_GAME_DATA",
                    r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
SA_WIN = os.path.join(SA, "StreamingAssets", "Windows")
TEMPLATES = os.path.join(REPO, "packs", "shared", "item_templates.json")
OUT_ROOT = os.path.join(REPO, "out", "weapons")

# Unity -> viewer world: the pack's X-flip (same conjugation as every other extractor).
G3 = np.diag([-1.0, 1.0, 1.0])


def g(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def trs(d):
    """Transform typetree -> 4x4 local matrix."""
    lp = g(d, "m_LocalPosition") or {}
    lr = g(d, "m_LocalRotation") or {}
    ls = g(d, "m_LocalScale") or {}
    x, y, z, w = (float(lr.get(k, 0.0)) for k in ("x", "y", "z", "w"))
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)
    S = np.diag([float(ls.get(k, 1.0)) for k in ("x", "y", "z")])
    M = np.eye(4)
    M[:3, :3] = R @ S
    M[:3, 3] = [float(lp.get(k, 0.0)) for k in ("x", "y", "z")]
    return M


class Bundle:
    """One item prefab bundle: its transform tree, renderers, and named slot nodes.

    A container bundle holds the PREFAB (GameObjects/Transforms/MeshFilters) but its meshes live
    in sibling asset bundles — the MeshFilter PPtrs carry m_FileID > 0 and the AssetBundle lists
    CAB dependencies. Loading the folder's sibling bundles into the SAME UnityPy Environment
    resolves them; the vertex/index data then needs MeshHandler (the streams are not in the
    typetree), exactly like skin.py reads skinned meshes.
    """

    def __init__(self, path, cabs=None):
        self.env = UnityPy.Environment()
        # Resolve the container's DECLARED CAB dependencies through the prebuilt index rather
        # than hoping a sibling happens to provide them: `m_FileID` is an index into the
        # serialized file's externals list, and UnityPy's fallback for a miss is a recursive
        # scan of the whole 37 GB game tree. `own` is the container's own objects — the
        # dependency bundles carry unrelated assets and must not be baked.
        own, _n = unity_deps.resolve_into(self.env, path, cabs if cabs is not None else {})
        self.tf = {}        # transform pid -> (typetree, go pid)
        self.children = {}  # transform pid -> [child transform pid]
        self.go_name = {}
        self.go2tf = {}
        self.renderers = []  # (go pid, MeshRenderer/SkinnedMeshRenderer object)
        for o in own:
            t = o.type.name
            if t == "Transform":
                d = o.read_typetree()
                gp = (d.get("m_GameObject") or {}).get("m_PathID", 0)
                self.tf[o.path_id] = (d, gp)
                self.go2tf[gp] = o.path_id
                for c in (d.get("m_Children") or []):
                    self.children.setdefault(o.path_id, []).append(c.get("m_PathID", 0))
            elif t == "GameObject":
                self.go_name[o.path_id] = o.read_typetree().get("m_Name", "")
            elif t in ("MeshRenderer", "SkinnedMeshRenderer"):
                d = o.read_typetree()
                self.renderers.append(((d.get("m_GameObject") or {}).get("m_PathID", 0), o, t))

    def roots(self):
        parented = {c for kids in self.children.values() for c in kids}
        return [p for p in self.tf if p not in parented]

    def world_of(self, tpid, upto=None):
        """Local-to-bundle-root matrix of a transform."""
        M = np.eye(4)
        seen = 0
        t = tpid
        while t and seen < 64:
            d, _ = self.tf.get(t, (None, 0))
            if d is None:
                break
            M = trs(d) @ M
            f = (d.get("m_Father") or {}).get("m_PathID", 0)
            if f == upto:
                break
            t = f
            seen += 1
        return M

    def slot_node(self, name):
        """Transform pid of the GameObject named `name` (a `mod_*` socket), or None."""
        for gp, nm in self.go_name.items():
            if nm == name:
                return self.go2tf.get(gp)
        return None


def bake(bundle, out_v, out_i, out_sub, base_M, mat_names, lod=0):
    """Append every renderer's mesh, transformed by base_M x its local matrix.

    LOD: item prefabs ship several detail shells (ak74_..._LOD0/_LOD1/...). Baking them all
    merged 3 copies of every part. Keep only the requested level, by the GameObject's own name
    suffix — the game's own naming, not a guess; a part with no suffix is kept (single-shell).
    """
    keep = f"_lod{lod}"
    for gp, obj, kind in bundle.renderers:
        tpid = bundle.go2tf.get(gp)
        if tpid is None:
            continue
        nm = (bundle.go_name.get(gp) or "").lower()
        if "_lod" in nm and keep not in nm:
            continue
        M = base_M @ bundle.world_of(tpid)
        try:
            r = obj.read()
            if kind == "SkinnedMeshRenderer":
                mesh_pptr = getattr(r, "m_Mesh", None)
            else:
                mesh_pptr = None
                go = r.m_GameObject.read()
                for comp in go.m_Component:
                    cp = comp[1] if isinstance(comp, (list, tuple)) else comp.component
                    co = cp.read()
                    if co.__class__.__name__ == "MeshFilter":
                        mesh_pptr = co.m_Mesh
                        break
            if mesh_pptr is None or not getattr(mesh_pptr, "path_id", 0):
                continue
            mesh = mesh_pptr.read()
            mats = list(getattr(r, "m_Materials", []) or [])
        except Exception:
            continue
        # MeshHandler decodes the vertex/index STREAMS; the typetree attributes are empty for
        # these bundles (m_Vertices == []), which is why a raw read assembled nothing.
        try:
            from UnityPy.helpers.MeshHelper import MeshHandler
            h = MeshHandler(mesh)
            h.process()
        except Exception as e:
            print(f"  [skip] mesh decode failed: {str(e)[:60]}")
            continue
        verts = h.m_Vertices
        if not verts:
            continue
        # MeshHandler returns either a FLAT float list or a list of 3-vectors depending on the
        # source layout — normalise to (N, 3) rather than assuming one shape.
        P = np.asarray(verts, np.float64)
        P = P.reshape(-1, 3) if P.ndim == 1 else P[:, :3]
        n = P.shape[0]
        P = (M[:3, :3] @ P.T).T + M[:3, 3]
        P = (G3 @ P.T).T                              # viewer conjugation
        nrm = h.m_Normals
        if nrm:
            N = np.asarray(nrm, np.float64)
            N = N.reshape(-1, 3) if N.ndim == 1 else N[:, :3]
            N = N[:n] if N.shape[0] >= n else np.tile([0.0, 1.0, 0.0], (n, 1))
        else:
            N = np.tile([0.0, 1.0, 0.0], (n, 1))
        N = (G3 @ (M[:3, :3] @ N.T)).T
        uv = getattr(h, "m_UV0", None) or getattr(h, "m_UV1", None)
        if uv:
            UV = np.asarray(uv, np.float64)
            UV = UV.reshape(-1, 2) if UV.ndim == 1 else UV[:, :2]
            UV = UV[:n] if UV.shape[0] >= n else np.zeros((n, 2))
        else:
            UV = np.zeros((n, 2))
        base = len(out_v)
        for i in range(n):
            out_v.append((P[i], N[i], UV[i]))
        # submeshes: one per material, so per-part materials survive the merge
        idx = list(h.m_IndexBuffer or [])
        subs = getattr(mesh, "m_SubMeshes", []) or []
        # Index width: `firstByte` is a BYTE offset, so converting it with a hardcoded /2
        # garbles every 32-bit-indexed mesh. MeshHandler knows which width this mesh uses.
        isz = 2 if getattr(h, "m_Use16BitIndices", True) else 4
        for si, sm in enumerate(subs):
            cnt = int(getattr(sm, "indexCount", 0) or 0)
            first = int(getattr(sm, "firstByte", 0) or 0) // isz
            if cnt <= 0:
                continue
            mat_name = ""
            try:
                if si < len(mats):
                    mat_name = mats[si].read().m_Name
            except Exception:
                pass
            if mat_name not in mat_names:
                mat_names.append(mat_name)
            start = len(out_i)
            # X-flip mirrors the winding — swap to keep faces outward.
            tri = idx[first:first + cnt]
            for k in range(0, len(tri) - 2, 3):
                out_i.extend((base + tri[k], base + tri[k + 2], base + tri[k + 1]))
            out_sub.append((mat_names.index(mat_name), start, len(out_i) - start))


def build(item_id, templates, out_dir, install=None, depth=0, bundle_cache=None, cabs=None):
    """Recursively assemble `item_id` and its installed mods into one merged mesh."""
    verts, idxs, subs, mat_names = [], [], [], []
    bundle_cache = bundle_cache if bundle_cache is not None else {}

    def rec(iid, M, slot_path):
        t = templates.get(iid)
        if not t:
            return
        rel = ((t.get("_props") or {}).get("Prefab") or {}).get("path") or ""
        if not rel:
            return
        p = os.path.join(SA_WIN, rel.replace("/", os.sep))
        if not os.path.exists(p):
            print(f"  [skip] {t.get('_name')}: bundle missing ({rel})")
            return
        b = bundle_cache.get(p)
        if b is None:
            b = bundle_cache[p] = Bundle(p, cabs)
        bake(b, verts, idxs, subs, M, mat_names)
        # children: for each installed mod, find the slot node with that name and recurse.
        for slot_name, child_id in (install or {}).get(iid, {}).items():
            node = b.slot_node(slot_name)
            if node is None:
                print(f"  [warn] {t.get('_name')}: no node '{slot_name}' in prefab")
                continue
            rec(child_id, M @ b.world_of(node), slot_path + "/" + slot_name)

    rec(item_id, np.eye(4), "")
    if not verts:
        raise SystemExit(f"no geometry assembled for {item_id}")
    os.makedirs(out_dir, exist_ok=True)
    # mesh.bin: pos f32x3, nrm f32x3, uv f32x2 = 32 B/vertex, then u32 indices.
    with open(os.path.join(out_dir, "mesh.bin"), "wb") as fh:
        for P, N, UV in verts:
            fh.write(struct.pack("<8f", *P, *N, *UV))
        for i in idxs:
            fh.write(struct.pack("<I", i))
    man = {
        "item": item_id,
        "name": (templates.get(item_id) or {}).get("_name", ""),
        "vertexCount": len(verts),
        "indexCount": len(idxs),
        "vertex": {"stride": 32, "fields": [
            {"name": "pos", "fmt": "f32x3", "offset": 0},
            {"name": "nrm", "fmt": "f32x3", "offset": 12},
            {"name": "uv", "fmt": "f32x2", "offset": 24}]},
        "submeshes": [{"material": m, "idxStart": s, "idxCount": c} for m, s, c in subs],
        "materials": mat_names,
        "conventions": {"world": "viewer (X-flipped from Unity)", "windingFlipped": True},
    }
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[done] {out_dir}\n  {len(verts)} verts, {len(idxs)//3} tris, "
          f"{len(subs)} submeshes, {len(mat_names)} materials")


def default_install(item_id, templates, presets):
    """{parent_id: {slot_name: child_id}} from BSG's own default preset for this weapon."""
    pre = presets.get(item_id)
    if not pre:
        return {}
    install = {}
    by_id = {}
    for it in pre:
        by_id[it["_id"]] = it
    for it in pre:
        par = it.get("parentId")
        slot = it.get("slotId")
        if par and slot and par in by_id:
            install.setdefault(by_id[par]["_tpl"], {})[slot] = it["_tpl"]
    return install


def load_presets(path):
    """globals.json ItemPresets -> {weapon tpl: [items]} (BSG's factory builds)."""
    if not os.path.exists(path):
        return {}
    g = json.load(open(path, encoding="utf-8"))
    out = {}
    for p in (g.get("ItemPresets") or {}).values():
        items = p.get("_items") or []
        if not items:
            continue
        root_tpl = items[0].get("_tpl")
        if p.get("_encyclopedia") or root_tpl not in out:
            out[root_tpl] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True, help="item template id (or _name)")
    ap.add_argument("--out", help="output dir (default out/weapons/<name>)")
    ap.add_argument("--globals", default=os.path.join(REPO, "packs", "shared", "globals.json"))
    args = ap.parse_args()
    templates = json.load(open(TEMPLATES, encoding="utf-8"))
    cabs = unity_deps.load(verbose=False)
    iid = args.item
    if iid not in templates:
        hit = next((k for k, v in templates.items()
                    if v.get("_name", "").lower() == iid.lower()), None)
        if not hit:
            raise SystemExit(f"no template {iid!r}")
        iid = hit
    presets = load_presets(args.globals)
    install = default_install(iid, templates, presets)
    name = templates[iid].get("_name") or iid
    out = args.out or os.path.join(OUT_ROOT, name)
    print(f"[build] {name} ({iid}); {sum(len(v) for v in install.values())} installed mod(s)")
    build(iid, templates, out, install=install, cabs=cabs)


if __name__ == "__main__":
    main()
