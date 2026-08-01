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
        self.path = path
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
        self.monos = []     # MonoBehaviour objects (scope modes, optic sights, ...)
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
            elif t == "MonoBehaviour":
                self.monos.append(o)
        # Objects are addressed within their own serialized file, so a PPtr with m_FileID == 0
        # must be resolved against the file that referenced it -- not by path id alone, which
        # collides once dependency bundles are in the same environment.
        self._by_file_pid = {(id(o.assets_file), o.path_id): o for o in self.env.objects}

    def deref(self, owner, pptr):
        """A same-file PPtr -> object, or None."""
        if not isinstance(pptr, dict) or pptr.get("m_FileID", 0) != 0 or not pptr.get("m_PathID"):
            return None
        return self._by_file_pid.get((id(owner.assets_file), pptr["m_PathID"]))

    def scope_modes(self):
        """The game's own list of a sight's mutually exclusive MODES.

        `ScopePrefabCache._scopeModeInfos` is BSG's table of what a sight can be at any moment:
        each entry names a `ModeGameObject` (the subtree to show) and either a `CollimatorSight`
        (a red dot) or an `OpticSight` (a magnified optic, carrying the `ScopeTransform` the eye
        aligns to when aiming). The HHS-1 has two: the G33 magnifier INLINE behind the holo sight,
        and the same magnifier FLIPPED ASIDE. Baking every renderer put both on the rail at once,
        which is why the optic looked like two scopes stacked on each other.

        Returns [{"tf": <mode root transform pid>, "nodes": {pids}, "optic": bool,
                  "aim": <ScopeTransform pid or None>, "fov": float or None}], in the game's order.
        """
        if getattr(self, "_scope_modes", None) is None:
            modes = []
            for o in self.monos:
                try:
                    if o.read().m_Script.read().m_Name != "ScopePrefabCache":
                        continue
                    d = o.read_typetree()
                except Exception:
                    continue
                for m in d.get("_scopeModeInfos") or []:
                    go = self.deref(o, m.get("ModeGameObject") or {})
                    if go is None:
                        continue
                    tfp = self.go2tf.get(go.path_id)
                    if tfp is None:
                        continue
                    optic = self.deref(o, m.get("OpticSight") or {})
                    aim = fov = lens = None
                    if optic is not None:
                        try:
                            od = optic.read_typetree()
                            st = self.deref(optic, od.get("ScopeTransform") or {})
                            aim = st.path_id if st is not None else None
                            # The LENS the eye looks through. Its position disambiguates which way
                            # the aim node faces -- see the sign derivation in bake().
                            lr = self.deref(optic, od.get("LensRenderer") or {})
                            if lr is not None:
                                lg = (lr.read_typetree().get("m_GameObject") or {}).get("m_PathID")
                                lens = self.go2tf.get(lg)
                            sd = self.deref(optic, od.get("ScopeData") or {})
                            # The optic's camera data is what magnification MEANS: a field of
                            # view in degrees, straight from the game.
                            for cand in (sd, optic):
                                if cand is None:
                                    continue
                                cd = cand.read_typetree()
                                if "FieldOfView" in cd:
                                    fov = float(cd["FieldOfView"])
                                    break
                        except Exception:
                            pass
                    if fov is None:
                        fov = self._camera_fov_near(go)
                    modes.append({"tf": tfp, "nodes": self.subtree(tfp),
                                  "optic": optic is not None, "aim": aim, "fov": fov,
                                  "lens": lens})
            self._scope_modes = modes
        return self._scope_modes

    def _camera_fov_near(self, mode_go):
        """`ScopeCameraData.FieldOfView` on any node inside this mode (the optic camera)."""
        want = self.subtree(self.go2tf.get(mode_go.path_id))
        for o in self.monos:
            try:
                if o.read().m_Script.read().m_Name != "ScopeCameraData":
                    continue
                d = o.read_typetree()
            except Exception:
                continue
            gp = (d.get("m_GameObject") or {}).get("m_PathID", 0)
            if self.go2tf.get(gp) in want and "FieldOfView" in d:
                return float(d["FieldOfView"])
        return None

    def roots(self):
        parented = {c for kids in self.children.values() for c in kids}
        return [p for p in self.tf if p not in parented]

    def subtree(self, root):
        """Transform pids under `root` (inclusive)."""
        out, stack = set(), [root]
        while stack:
            t = stack.pop()
            if t in out:
                continue
            out.add(t)
            stack.extend(self.children.get(t, []))
        return out

    def variant(self):
        """The ONE model variant to read.

        A weapon container ships several roots: `<name>_container`, `<name>_model.generated` and
        `<name>_model_simple.generated`. Both `.generated` trees carry their own `Weapon_root`
        and `mod_*` sockets, in DIFFERENT frames — so resolving a socket by name across the whole
        bundle mixed the two and threw individual parts onto the wrong axis. The game loads one
        variant; prefer the full `_model.generated` over `_model_simple`.
        """
        if getattr(self, "_variant", None) is None:
            best, best_score = None, -1
            for r in self.roots():
                name = (self.go_name.get(self.tf[r][1], "") or "").lower()
                sub = self.subtree(r)
                has_anchor = any(
                    (self.go_name.get(self.tf[t][1], "") or "") == "Weapon_root" for t in sub
                )
                # full model > simple model > anything else; ties by size
                score = len(sub)
                if has_anchor:
                    score += 100_000
                if "simple" in name:
                    score -= 50_000
                if score > best_score:
                    best, best_score = r, score
            self._variant = self.subtree(best) if best is not None else set(self.tf)
        return self._variant

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

    def root_inv(self):
        """Inverse of the ANCHOR's world matrix — what makes baked geometry land correctly when
        the result is parented to the character rig.

        THE GAME'S OWN CONTRACT (read out of the bundle, not guessed): a weapon container ships
        a `*.generated` root carrying a partial arm rig — `Base HumanLCollarbone`,
        `Base HumanRCollarbone`, `Camera_animated` — and the weapon itself under
        `Weapon_root / Weapon_root_anim / weapon`. That `Weapon_root` node is the SAME NAME as
        the socket bone in the 79-bone character rig (index 68): the game aligns the two, which
        is precisely how a gun ends up in the hands with the arms posed around it.

        So the anchor is that node when present. Equipment prefabs (armor, helmets) have no such
        node and fall back to the prefab root, which is their own authored origin.
        """
        if getattr(self, "_root_inv", None) is None:
            anchor = self.slot_node("Weapon_root")
            if anchor is None:
                # MOD PREFABS: the mount frame is the MESH node's, not the prefab root's.
                # Measured: a mod's LOD node carries a -90 deg X rotation relative to its root
                # (mesh-local Y -> -Z). Anchoring on the root therefore baked that rotation in and
                # every mod came out turned 90 deg from the weapon body, whose own chain is
                # rotation-free relative to Weapon_root. Anchoring on the mesh node cancels it:
                # the receiver then runs ALONG the barrel (Y) like the body, and the magazine sits
                # PERPENDICULAR (Z) hanging below it — an AK. Falls back to the root when a prefab
                # has no renderer to anchor on.
                #
                # The anchor must be a node that actually CONTRIBUTES BAKED GEOMETRY. Matching
                # any name without "_lod" also matched the Unity primitive gizmos a prefab uses
                # as markers -- the LA-5's laser emitters are `Sphere` nodes carrying a ~1/252
                # scale, and anchoring on one inverted that scale into every vertex, baking the
                # 8 cm device as a 51 m object. Preferring a renderer whose mesh actually
                # resolves excludes gizmos structurally: they point at Unity's built-in library,
                # which the game does not ship.
                lod0 = [t for t in (self._anchor_candidates("_lod0"))]
                anchor = next(iter(lod0), None) or next(iter(self._anchor_candidates(None)), None)
            if anchor is None:
                roots = [r for r in self.roots() if r in self.variant()] or self.roots()
                anchor = None
                best_n = -1
                for r in roots:
                    n, stack = 0, [r]
                    while stack:
                        cur = stack.pop()
                        kids = self.children.get(cur, [])
                        n += len(kids)
                        stack.extend(kids)
                    if n > best_n:
                        anchor, best_n = r, n
            M = self.world_of(anchor) if anchor is not None else np.eye(4)
            # A mount is a FRAME: it says where the part sits and which way it faces, not how
            # big it is. Scale on the anchor is an authoring artifact of that one node, and
            # inverting it rescales the geometry -- so the basis is orthonormalised first,
            # keeping orientation and handedness and dropping scale. This is safe because the
            # meshes are already authored in metres (measured: LA-5 8 cm, receiver 21 cm,
            # PMAG 19 cm), so nothing needs a compensating scale.
            R = M[:3, :3].astype(np.float64).copy()
            norms = np.linalg.norm(R, axis=0)
            scaled = np.any(np.abs(norms - 1.0) > 1e-3)
            R[:, norms > 1e-9] /= norms[norms > 1e-9]
            F = np.eye(4)
            F[:3, :3], F[:3, 3] = R, M[:3, 3]
            if scaled:
                print(f"  [anchor] {os.path.basename(self.path)}: mount frame carried scale "
                      f"{np.round(norms, 4)} -- using orientation only")
            try:
                self._root_inv = np.linalg.inv(F)
            except np.linalg.LinAlgError:
                self._root_inv = np.eye(4)
        return self._root_inv

    def _anchor_candidates(self, name_contains):
        """Transform pids of renderers that will actually bake, in bundle order.

        `name_contains=None` accepts any renderer that has no explicit `_lod` suffix. A renderer
        whose mesh cannot be resolved (Unity's built-in primitives — the gizmo spheres) is never
        a candidate: it contributes no geometry, so it cannot define the part's frame.
        """
        for gp, obj, kind in self.renderers:
            t = self.go2tf.get(gp)
            nm = (self.go_name.get(gp) or "").lower()
            if t is None:
                continue
            if name_contains is None:
                if "_lod" in nm:
                    continue
            elif name_contains not in nm:
                continue
            try:
                r = obj.read()
                mp = getattr(r, "m_Mesh", None)
                if mp is None:
                    for c in r.m_GameObject.read().m_Component:
                        cp = c[1] if isinstance(c, (list, tuple)) else c.component
                        co = cp.read()
                        if co.__class__.__name__ == "MeshFilter":
                            mp = co.m_Mesh
                            break
                if mp is None or not getattr(mp, "path_id", 0):
                    continue
                mp.read()          # resolves, or raises for a built-in / missing dependency
            except Exception:
                continue
            yield t

    def slot_node(self, name):
        """Transform pid of the GameObject named `name` (a `mod_*` socket) WITHIN the chosen
        model variant — searching the whole bundle mixed the full and simple frames."""
        var = self.variant()
        for t in var:
            if (self.go_name.get(self.tf[t][1], "") or "") == name:
                return t
        return None


def bake(bundle, out_v, out_i, out_sub, base_M, mat_names, tex_by_mat, lod=0, props_by_mat=None,
         out_aim=None):
    """Append every renderer's mesh, transformed by base_M x its local matrix.

    LOD: item prefabs ship several detail shells (ak74_..._LOD0/_LOD1/...). Baking them all
    merged 3 copies of every part. Keep only the requested level, by the GameObject's own name
    suffix — the game's own naming, not a guess; a part with no suffix is kept (single-shell).
    """
    keep = f"_lod{lod}"
    # A sight's MODES are mutually exclusive states of one item, listed by the game itself in
    # `ScopePrefabCache._scopeModeInfos`. Only the selected mode's geometry exists at any moment;
    # baking them all stacked the HHS-1's magnifier onto its own flipped-aside copy. `EFT_SCOPE_MODE`
    # picks which, defaulting to the game's first entry.
    seen_state_nodes = {}
    dropped_states = []
    modes = bundle.scope_modes()
    hidden_modes = set()
    if modes:
        want = int(os.environ.get("EFT_SCOPE_MODE", "0"))
        want = want if 0 <= want < len(modes) else 0
        for i, m in enumerate(modes):
            if i != want:
                hidden_modes |= m["nodes"]
        hidden_modes -= modes[want]["nodes"]
        # THE AIM ANCHOR. `OpticSight.ScopeTransform` is the node the game aligns the eye to when
        # you bring the sight up, and `ScopeCameraData.FieldOfView` is that optic's magnification
        # expressed as a field of view in degrees. Both come straight from the sight's own prefab,
        # so aiming needs no authored offsets. Recorded in the ASSEMBLED weapon's space -- the same
        # space as the baked vertices -- because that is what the viewer parents to.
        sel = modes[want]
        if out_aim is not None and sel.get("aim") is not None:
            A = base_M @ bundle.root_inv() @ bundle.world_of(sel["aim"])
            R = A[:3, :3].astype(np.float64)
            norms = np.linalg.norm(R, axis=0)
            R[:, norms > 1e-9] /= norms[norms > 1e-9]
            # Same X-flip conjugation the vertices get: a point maps by G3, a basis by G3 R.
            pos = (G3 @ A[:3, 3])
            basis = G3 @ R
            fwd = basis[:, 2]
            # WHICH WAY THE EYE FACES. A Unity camera looks down +Z, but this node's +Z points
            # back toward the shoulder -- measured on the HHS-1, where the anchor sits at Y -0.146,
            # its lens at -0.26 and the muzzle at -0.74, so downrange is -Y while the node's +Z
            # reads +Y. Rather than hardcode a flip, the sign is taken from the optic's OWN
            # `LensRenderer`: you look THROUGH the lens, so forward is whichever of +/-Z agrees
            # with the direction from the anchor to it.
            if sel.get("lens") is not None:
                lp = G3 @ (base_M @ bundle.root_inv() @ bundle.world_of(sel["lens"]))[:3, 3]
                to_lens = lp - pos
                if float(np.dot(fwd, to_lens)) < 0.0:
                    fwd = -fwd
            out_aim.append({
                "position": [float(v) for v in pos],
                "forward": [float(v) for v in fwd],
                "up": [float(v) for v in basis[:, 1]],
                "fov": float(sel["fov"]) if sel.get("fov") else None,
                "source": "OpticSight.ScopeTransform",
            })
    for gp, obj, kind in bundle.renderers:
        tpid = bundle.go2tf.get(gp)
        if tpid is None:
            continue
        if tpid not in bundle.variant():
            continue  # the other model variant's copy of this part
        if tpid in hidden_modes:
            continue  # belongs to a scope mode that is not the active one
        nm = (bundle.go_name.get(gp) or "").lower()
        if "_lod" in nm and keep not in nm:
            continue
        # Relative to the prefab ROOT, not the bundle's absolute space: a container's root
        # transform can sit far from the origin, and baking that in put the assembled weapon
        # metres away from the hand once parented to the rig's socket.
        M = base_M @ bundle.root_inv() @ bundle.world_of(tpid)
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
                print(f"  [skip] {nm}: renderer has no mesh pointer")
                continue
            # Cross-bundle: m_FileID > 0 means the mesh lives in a DEPENDENCY, which must have
            # been resolved into this environment. A miss here reads as "the item has no
            # geometry", so it is reported with the CAB that is missing — swallowing it is how
            # Killa's armour silently assembled to nothing.
            mesh = mesh_pptr.read()
            mats = list(getattr(r, "m_Materials", []) or [])
        except Exception as e:
            # Unity's BUILT-IN library ("unity default resources") holds the editor primitives —
            # the sphere/cube a prefab uses as a gizmo or collider proxy. It is part of the
            # engine, not of the game, so it is absent from StreamingAssets by design and
            # carries no geometry we want. Everything else is a genuine broken reference.
            if "unity default resources" in str(e) or "unity_builtin" in str(e):
                continue
            print(f"  [skip] {nm}: mesh unreadable ({type(e).__name__}: {str(e)[:70]})")
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
        if os.environ.get("EFT_WEAPON_DEBUG"):
            sz = P.max(0) - P.min(0)
            ctr = (P.max(0) + P.min(0)) / 2
            print(f"  [part] {nm[:38]:38s} mesh={getattr(mesh,'m_Name','?')[:28]:28s} "
                  f"verts={n:6d} size={np.round(sz,2)} ctr={np.round(ctr,2)}")
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
                    mobj = mats[si].read()
                    mat_name = mobj.m_Name
                    if mat_name not in tex_by_mat:
                        slots = {}
                        te = mobj.m_SavedProperties.m_TexEnvs
                        for k, v in (te.items() if hasattr(te, "items") else te):
                            tp = getattr(v, "m_Texture", None)
                            if getattr(tp, "path_id", 0):
                                try:
                                    timg = tp.read()
                                    slots[str(k)] = (timg.m_Name, timg)
                                except Exception:
                                    pass
                        tex_by_mat[mat_name] = slots
                        # The material's own SCALARS and COLOURS, kept raw and named as the
                        # shader names them. This is how glass survives the trip: EFT's
                        # transparent-reflective family carries its opacity in `_Color.a`
                        # (the EOTech's is 0.128) alongside `_ReflectColor` / `_SpecColor` /
                        # `_FresPow`. Dropping them left the viewer with a default material,
                        # which is opaque WHITE -- so every scope lens rendered as a solid pane.
                        if props_by_mat is not None:
                            sp = mobj.m_SavedProperties
                            fl, co = {}, {}
                            src = sp.m_Floats
                            for k, v in (src.items() if hasattr(src, "items") else src):
                                try:
                                    fl[str(k)] = float(v)
                                except Exception:
                                    pass
                            src = sp.m_Colors
                            for k, v in (src.items() if hasattr(src, "items") else src):
                                try:
                                    fl_c = [float(v.r), float(v.g), float(v.b), float(v.a)]
                                except Exception:
                                    continue
                                co[str(k)] = fl_c
                            props_by_mat[mat_name] = {"floats": fl, "colors": co}
            except Exception:
                pass
            if mat_name not in mat_names:
                mat_names.append(mat_name)
            # MUTUALLY EXCLUSIVE STATE NODES. A mod prefab ships every state of a moving part and
            # the game shows one: the LA-5 carries `switch_000..switch_004`, its selector drawn in
            # all five positions at once (its template says `ModesCount: 4`), and the MBUS front
            # sight ships folded AND deployed, tagged by `AutoFoldableSight.Mode`. Baking them all
            # stacked identical geometry on itself.
            #
            # The test is MEASURED, not named: two pieces sharing a material whose bounds centre on
            # the same point are one part drawn twice. Genuinely different parts that merely overlap
            # -- the barrel inside its handguard, the G33's lens against its backing disc -- carry
            # different materials and are untouched.
            tri = idx[first:first + cnt]
            if len(tri):
                u = np.unique(np.asarray(tri, np.int64))
                u = u[u < P.shape[0]]
                if u.size:
                    pts = P[u]
                    ctr = (pts.max(0) + pts.min(0)) / 2
                    # Same material AND the same triangle count AND centred within 2 cm: the same
                    # part drawn in another of its states. The switch positions differ by only a
                    # few millimetres, so this is a distance test, not an exact key.
                    key = (mat_name, int(cnt))
                    prev = seen_state_nodes.setdefault(key, [])
                    if any(float(np.linalg.norm(ctr - c)) < 0.02 for c in prev):
                        dropped_states.append(nm)
                        continue
                    prev.append(ctr)
            start = len(out_i)
            # X-flip mirrors the winding — swap to keep faces outward.
            for k in range(0, len(tri) - 2, 3):
                out_i.extend((base + tri[k], base + tri[k + 2], base + tri[k + 1]))
            out_sub.append((mat_names.index(mat_name), start, len(out_i) - start))
    if dropped_states:
        from collections import Counter
        for nm_, k in Counter(dropped_states).most_common():
            print(f"  [state] {os.path.basename(bundle.path)}: {nm_} drawn {k + 1}x at one place "
                  f"-- kept 1 (alternate states of a moving part)")


def build(item_id, templates, out_dir, install=None, depth=0, bundle_cache=None, cabs=None):
    """Recursively assemble `item_id` and its installed mods into one merged mesh."""
    verts, idxs, subs, mat_names = [], [], [], []
    aim = []
    tex_by_mat = {}
    props_by_mat = {}
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
        bake(b, verts, idxs, subs, M, mat_names, tex_by_mat, props_by_mat=props_by_mat,
             out_aim=aim)
        # children: for each installed mod, find the slot node with that name and recurse.
        for slot_name, child_id in (install or {}).get(iid, {}).items():
            node = b.slot_node(slot_name)
            if node is None:
                print(f"  [warn] {t.get('_name')}: no node '{slot_name}' in prefab")
                continue
            # The socket matrix must be root-RELATIVE too, or each child inherits the parent
            # bundle's absolute offset (which stretched the assembled gun to 1.75 m).
            rec(child_id, M @ b.root_inv() @ b.world_of(node), slot_path + "/" + slot_name)

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
    # ---- textures: same conventions as the character packer ----
    tex_dir = os.path.join(out_dir, "textures")
    written = {}
    for mname, slots in tex_by_mat.items():
        for slot, (tname, timg) in slots.items():
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(tname))
            rel = f"textures/{safe}.png"
            if rel not in written.values():
                try:
                    os.makedirs(tex_dir, exist_ok=True)
                    img = timg.image
                    # Unity DXT5nm normal maps carry X in ALPHA; repack to standard RGB or every
                    # PBR consumer reads a tangent normal pointing along the surface.
                    if slot in ("_BumpMap", "_NormalMap") or safe.lower().endswith(("_n", "_normal")):
                        img = _repack_normal(img, safe)
                    img.save(os.path.join(out_dir, rel))
                except Exception as e:
                    print(f"  [warn] texture {safe}: {str(e)[:50]}")
                    continue
            written.setdefault(mname, {})[slot] = rel
    man = {
        "item": item_id,
        "materialTextures": written,
        "name": (templates.get(item_id) or {}).get("_name", ""),
        "vertexCount": len(verts),
        "indexCount": len(idxs),
        "vertex": {"stride": 32, "fields": [
            {"name": "pos", "fmt": "f32x3", "offset": 0},
            {"name": "nrm", "fmt": "f32x3", "offset": 12},
            {"name": "uv", "fmt": "f32x2", "offset": 24}]},
        "submeshes": [{"material": m, "idxStart": s, "idxCount": c} for m, s, c in subs],
        "materials": mat_names,
        # Per-material scalars/colours exactly as the shader names them. The viewer maps what it
        # understands (`_Color.a` -> opacity, `_SpecColor`/`_Shininess` -> reflectance) and
        # ignores the rest, so a new property never breaks the contract.
        "materialProps": props_by_mat,
        # Where the eye goes when aiming, from the sight's own OpticSight component. Absent when
        # the build carries no magnified optic (iron sights and red dots have no ScopeTransform).
        "aim": (aim[0] if aim else None),
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


def _repack_normal(img, name):
    """Unity DXT5nm -> standard RGB normal (X from ALPHA, Z reconstructed). Detected by
    MEASUREMENT (red pinned near-constant), so a standard map passes through untouched. Same
    rule as extraction/characters/pack.py and eft_extract_v2.unswizzle_normal."""
    try:
        from PIL import Image
    except Exception:
        return img
    a = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    r, g, al = a[..., 0], a[..., 1], a[..., 3]
    if r.std() > 0.02:
        return img
    x = al * 2.0 - 1.0
    y = g * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    out = np.stack([(x + 1.0) * 0.5, (y + 1.0) * 0.5, (z + 1.0) * 0.5], axis=-1)
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype("uint8"), "RGB")


if __name__ == "__main__":
    main()

