"""A character part bundle -> skinned meshes, bone remap, bindposes, materials, textures.

EFT assembles a character from independent part prefabs (`top_boss_tagilla`, `pants_boss_tagilla`,
`bear_body`, ...). Each part is a `SkinnedMeshRenderer` + `LODGroup` and binds only the bones it
needs -- Tagilla's top binds 48 of the rig's 79, his pants bind 12. The part's `Skin` MonoBehaviour
carries `_bonePaths`, the ordered path strings for its own bone slots, which is the join into the
canonical rig.

TWO INDEPENDENT SOURCES agree on that join and both are checked:
  * `Skin._bonePaths[i]`      -> path string  -> rig index
  * `Mesh.m_BoneNameHashes[i]` -> CRC32(path) -> rig index
They must produce the same remap. If they disagree the part is rejected; a wrong remap is the one
failure mode that produces a character that looks *almost* right, which is far worse than a crash.

Joint indices are rewritten to GLOBAL rig indices here, and each mesh gets a rig-sized inverse
bindpose table (79 entries, identity where the mesh does not bind that bone). Cost is ~5 KB per
mesh; the payoff is that every part of every character shares one joint entity list in the viewer,
so assembling a character is "spawn the rig once, attach N meshes".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coords
from .skeleton import Skeleton
from .unity_bind import path_hash

#: Interleaved vertex layout written to skin.bin. The manifest declares it; the loader reads it
#: from there. Keep `format` strings in sync with the Rust side's parser.
VERTEX_LAYOUT = [
    ("position", "f32x3", 12),
    ("normal", "f32x3", 12),
    ("tangent", "f32x4", 16),
    ("uv0", "f32x2", 8),
    ("jointIndex", "u16x4", 8),
    ("jointWeight", "f32x4", 16),
]
VERTEX_STRIDE = sum(sz for _, _, sz in VERTEX_LAYOUT)  # 72


def vertex_layout_manifest() -> dict:
    attrs = []
    off = 0
    for name, fmt, size in VERTEX_LAYOUT:
        attrs.append({"name": name, "format": fmt, "offset": off})
        off += size
    return {"stride": VERTEX_STRIDE, "attributes": attrs}


@dataclass
class SubMesh:
    material: int  #: index into the pack's materials[]
    index_start: int  #: in indices, relative to this mesh's index block
    index_count: int


@dataclass
class SkinMesh:
    name: str
    part: str
    lod: int
    vertices: np.ndarray  #: (V, VERTEX_STRIDE) uint8 -- already interleaved
    indices: np.ndarray  #: (I,) uint32, mesh-local, winding already flipped
    submeshes: List[SubMesh]
    #: (rig_bone_count, 4, 4) float32 inverse bindposes, viewer space, identity where unbound.
    inverse_bindposes: np.ndarray
    bound_bones: List[int]  #: rig indices this mesh actually skins to (for debug/validation)
    vertex_count: int = 0

    def __post_init__(self) -> None:
        self.vertex_count = int(self.vertices.shape[0])


@dataclass
class Material:
    name: str
    textures: Dict[str, str] = field(default_factory=dict)  #: slot -> "textures/<file>.png"
    #: Scalar/colour properties worth carrying (glossiness, tint, ...). Kept raw and named as the
    #: shader names them; the viewer maps what it understands and ignores the rest.
    floats: Dict[str, float] = field(default_factory=dict)
    colors: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class Attachment:
    """A RIGID equipment mesh parented to one rig bone.

    EFT's headwear/facecover items are not skinned: the welding-mask prefab is `MeshFilter` +
    `MeshRenderer` with zero bindposes and no bone hashes, so it rides a bone rather than deforming.
    Its `Dress` component lists only renderers and a decal type -- the slot->bone mapping lives in
    the runtime's `PlayerBody.SlotView`, not in the prefab -- so the target bone comes from the
    registry and is an explicit authoring choice, flagged as such in the manifest.
    """

    name: str
    bone: int
    lod: int
    #: Local transform of the mesh within the prefab, composed down from the prefab root. Carries the
    #: -90 deg X fixup these prefabs use.
    local_pos: List[float]
    local_rot: List[float]
    local_scale: List[float]
    vertices: np.ndarray
    indices: np.ndarray
    submeshes: List[SubMesh]
    vertex_count: int = 0

    def __post_init__(self) -> None:
        self.vertex_count = int(self.vertices.shape[0])


@dataclass
class PartResult:
    meshes: List[SkinMesh] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    #: texture m_Name -> PIL image, deduplicated by the caller across parts.
    images: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _matrix4_from_typetree(m: dict) -> np.ndarray:
    """Unity `Matrix4x4f` typetree (e00..e33, e{row}{col}) -> 4x4 with translation in column 3."""
    out = np.empty((4, 4), np.float64)
    for r in range(4):
        for c in range(4):
            out[r, c] = float(m[f"e{r}{c}"])
    return out


_LOD_RE = re.compile(r"_lod(\d+)")


def _lod_from_name(name: str) -> int:
    """Extract the LOD index from a mesh name.

    Searches ANYWHERE in the name, not just the end: character parts are `Top_..._lod0` but equipment
    is `item_..._lod1_base`, and matching only a suffix let every equipment LOD through the filter, so
    the item drew twice overlapping itself.
    """
    m = _LOD_RE.search(name.lower())
    return int(m.group(1)) if m else 0


def _script_name(obj) -> str:
    try:
        return obj.read().m_Script.read().m_Name
    except Exception:
        return ""


def _resolve_remap(
    skel: Skeleton,
    bone_paths: Optional[Sequence[str]],
    bone_hashes: Optional[Sequence[int]],
    mesh_name: str,
    strict: bool,
) -> List[int]:
    """mesh bone slot -> rig bone index, from both sources, cross-checked."""
    by_path = skel.by_path
    by_hash = skel.by_hash

    from_paths: Optional[List[int]] = None
    if bone_paths:
        from_paths = []
        for p in bone_paths:
            idx = by_path.get(p)
            if idx is None:
                raise RuntimeError(
                    f"{mesh_name}: Skin._bonePaths entry {p!r} is not a bone of the canonical rig"
                )
            from_paths.append(idx)

    from_hashes: Optional[List[int]] = None
    if bone_hashes:
        from_hashes = []
        for h in bone_hashes:
            idx = by_hash.get(int(h) & 0xFFFFFFFF)
            if idx is None:
                raise RuntimeError(
                    f"{mesh_name}: m_BoneNameHashes entry {int(h):#010x} matches no rig bone path"
                )
            from_hashes.append(idx)

    if from_paths is not None and from_hashes is not None:
        if from_paths != from_hashes:
            bad = [
                (i, skel.names[a], skel.names[b])
                for i, (a, b) in enumerate(zip(from_paths, from_hashes))
                if a != b
            ]
            msg = f"{mesh_name}: bone remap disagreement between _bonePaths and m_BoneNameHashes: {bad[:6]}"
            if strict:
                raise RuntimeError(msg)
            print(f"  [warn] {msg} -- trusting _bonePaths")
        return from_paths

    remap = from_paths if from_paths is not None else from_hashes
    if remap is None:
        raise RuntimeError(f"{mesh_name}: no bone binding source (neither _bonePaths nor hashes)")
    return remap


def _pack_vertices(
    positions: np.ndarray,
    normals: Optional[np.ndarray],
    tangents: Optional[np.ndarray],
    uv0: Optional[np.ndarray],
    joint_index: np.ndarray,
    joint_weight: np.ndarray,
) -> np.ndarray:
    """Interleave into VERTEX_LAYOUT. Returns (V, VERTEX_STRIDE) uint8."""
    v = positions.shape[0]
    buf = np.zeros((v, VERTEX_STRIDE), np.uint8)

    def put(off: int, arr: np.ndarray, dtype) -> None:
        raw = np.ascontiguousarray(arr.astype(dtype)).view(np.uint8).reshape(v, -1)
        buf[:, off : off + raw.shape[1]] = raw

    off = 0
    put(off, positions[:, :3], np.float32)
    off += 12
    n = normals if normals is not None else np.tile(np.array([0, 1, 0], np.float32), (v, 1))
    put(off, n[:, :3], np.float32)
    off += 12
    t = tangents if tangents is not None else np.tile(np.array([1, 0, 0, 1], np.float32), (v, 1))
    if t.shape[1] == 3:
        t = np.column_stack([t, np.ones(v, np.float32)])
    put(off, t[:, :4], np.float32)
    off += 16
    u = uv0 if uv0 is not None else np.zeros((v, 2), np.float32)
    put(off, u[:, :2], np.float32)
    off += 8
    put(off, joint_index[:, :4], np.uint16)
    off += 8
    put(off, joint_weight[:, :4], np.float32)
    return buf


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def load_part(
    bundle_path: str,
    part_name: str,
    skel: Skeleton,
    material_base: int,
    strict: bool = True,
    lods: Optional[Sequence[int]] = None,
) -> PartResult:
    """Read one part bundle. `material_base` is the pack-wide index the first emitted material takes.

    `lods=None` keeps every LOD found; `lods=(0,)` keeps only LOD0.
    """
    import UnityPy
    from UnityPy.helpers.MeshHelper import MeshHandler

    env = UnityPy.load(bundle_path)
    result = PartResult()

    # ---- pass 1: index the bundle -------------------------------------------------
    meshes: List[Tuple[object, dict]] = []  # (object_reader, typetree)
    smrs: List[dict] = []
    mats_by_pathid: Dict[int, dict] = {}
    texs_by_pathid: Dict[int, object] = {}
    skins: List[dict] = []

    for obj in env.objects:
        tname = obj.type.name
        if tname == "Mesh":
            meshes.append((obj, obj.read_typetree()))
        elif tname == "SkinnedMeshRenderer":
            smrs.append(obj.read_typetree())
        elif tname == "Material":
            mats_by_pathid[obj.path_id] = obj.read_typetree()
        elif tname == "Texture2D":
            texs_by_pathid[obj.path_id] = obj
        elif tname == "MonoBehaviour" and _script_name(obj) == "Skin":
            skins.append(obj.read_typetree())

    if not meshes:
        raise RuntimeError(f"{bundle_path}: no Mesh objects")

    # ---- materials + textures ----------------------------------------------------
    #: Material path_id -> pack material index. SMRs reference materials by PPtr.
    mat_index: Dict[int, int] = {}
    for pid, mt in mats_by_pathid.items():
        mat = Material(name=str(mt.get("m_Name", f"material_{pid}")))
        saved = mt.get("m_SavedProperties", {}) or {}
        for entry in saved.get("m_TexEnvs", []) or []:
            slot, val = entry[0], entry[1]
            tex_pid = int((val.get("m_Texture") or {}).get("m_PathID", 0))
            if not tex_pid or tex_pid not in texs_by_pathid:
                continue
            tex_obj = texs_by_pathid[tex_pid]
            try:
                tex = tex_obj.read()
                img = tex.image
                if img is None:
                    continue
                tex_name = str(tex.m_Name)
            except Exception as exc:  # streamed texture missing its .resS, unsupported format, ...
                print(f"  [warn] {bundle_path}: texture for {slot} unreadable ({exc})")
                continue
            result.images[tex_name] = img
            mat.textures[str(slot)] = f"textures/{tex_name}.png"
        for entry in saved.get("m_Floats", []) or []:
            mat.floats[str(entry[0])] = float(entry[1])
        for entry in saved.get("m_Colors", []) or []:
            c = entry[1]
            mat.colors[str(entry[0])] = [
                float(c.get("r", 1.0)),
                float(c.get("g", 1.0)),
                float(c.get("b", 1.0)),
                float(c.get("a", 1.0)),
            ]
        mat_index[pid] = material_base + len(result.materials)
        result.materials.append(mat)

    # ---- SMR lookup: mesh path_id -> its renderer (for the material list) --------
    smr_by_mesh: Dict[int, dict] = {}
    for smr in smrs:
        mpid = int((smr.get("m_Mesh") or {}).get("m_PathID", 0))
        if mpid:
            smr_by_mesh[mpid] = smr

    # `Skin` bone paths are per renderer; in practice a part has one binding set shared by its
    # LOD meshes, so take the first non-empty and validate per mesh against the hashes.
    skin_bone_paths: Optional[List[str]] = None
    for sk in skins:
        bp = sk.get("_bonePaths") or []
        if bp:
            skin_bone_paths = [str(p) for p in bp]
            break

    # ---- pass 2: geometry --------------------------------------------------------
    for obj, tt in meshes:
        name = str(tt.get("m_Name", "mesh"))
        lod = _lod_from_name(name)
        if lods is not None and lod not in lods:
            continue

        mesh = obj.read()
        handler = MeshHandler(mesh)
        handler.process()

        if not handler.m_Vertices:
            raise RuntimeError(f"{name}: MeshHandler produced no vertices")
        positions = np.asarray(handler.m_Vertices, np.float32).reshape(-1, 3)
        v = positions.shape[0]
        normals = (
            np.asarray(handler.m_Normals, np.float32).reshape(v, -1)[:, :3]
            if handler.m_Normals
            else None
        )
        tangents = (
            np.asarray(handler.m_Tangents, np.float32).reshape(v, -1) if handler.m_Tangents else None
        )
        uv0 = (
            coords.uvs(np.asarray(handler.m_UV0, np.float32).reshape(v, -1)[:, :2])
            if handler.m_UV0
            else None
        )

        if not handler.m_BoneIndices or not handler.m_BoneWeights:
            raise RuntimeError(
                f"{name}: no skin weights in the vertex data -- this is not a skinned mesh"
            )
        local_joints = np.asarray(handler.m_BoneIndices, np.uint32).reshape(v, -1)[:, :4]
        weights = np.asarray(handler.m_BoneWeights, np.float32).reshape(v, -1)[:, :4]

        # ---- bone remap, cross-validated ----
        bone_hashes = tt.get("m_BoneNameHashes") or []
        remap = _resolve_remap(skel, skin_bone_paths, bone_hashes, name, strict)
        n_slots = len(remap)
        if int(local_joints.max(initial=0)) >= n_slots:
            raise RuntimeError(
                f"{name}: vertex references bone slot {int(local_joints.max())} but only "
                f"{n_slots} slots are bound"
            )
        remap_arr = np.asarray(remap, np.uint32)
        # A zero-weight influence keeps whatever slot byte the exporter left there; clamp it to the
        # rig root so it can never index out of the joint palette.
        global_joints = remap_arr[np.clip(local_joints, 0, n_slots - 1)]
        global_joints = np.where(weights > 0.0, global_joints, 0).astype(np.uint16)

        # Renormalise: Unity's stored weights are close to 1 but not exactly, and Bevy expects a
        # partition of unity.
        wsum = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, wsum, out=np.zeros_like(weights), where=wsum > 1e-8)
        degenerate = int((wsum <= 1e-8).sum())
        if degenerate:
            weights[(wsum <= 1e-8).ravel(), 0] = 1.0
            print(f"  [warn] {name}: {degenerate} vertices had zero total weight -> pinned to root")

        # ---- inverse bindposes, rig-sized ----
        bindposes = tt.get("m_BindPose") or []
        if len(bindposes) != n_slots:
            raise RuntimeError(
                f"{name}: {len(bindposes)} bindposes for {n_slots} bound bones -- mismatched part"
            )
        ibm = np.tile(np.eye(4, dtype=np.float32), (len(skel), 1, 1))
        for slot, rig_idx in enumerate(remap):
            ibm[rig_idx] = coords.matrix4(_matrix4_from_typetree(bindposes[slot]))

        # ---- geometry into viewer space ----
        positions = coords.points(positions)
        if normals is not None:
            normals = coords.normals(normals)
        if tangents is not None:
            tangents = coords.tangents(tangents)

        if not handler.m_IndexBuffer:
            raise RuntimeError(f"{name}: no index buffer")
        all_indices = np.asarray(handler.m_IndexBuffer, np.uint32)

        # ---- submeshes ----
        smr = smr_by_mesh.get(obj.path_id, {})
        smr_mats = [int((m or {}).get("m_PathID", 0)) for m in (smr.get("m_Materials") or [])]
        sub_tt = tt.get("m_SubMeshes") or []
        submeshes: List[SubMesh] = []
        kept_indices: List[np.ndarray] = []
        cursor = 0
        index_size = 2 if tt.get("m_IndexFormat", 0) == 0 else 4
        for si, sm in enumerate(sub_tt):
            first = int(sm.get("firstByte", 0)) // index_size
            count = int(sm.get("indexCount", 0))
            base = int(sm.get("baseVertex", 0) or 0)
            seg = all_indices[first : first + count] + base
            seg = coords.flip_winding(seg)
            kept_indices.append(seg)
            mat_pid = smr_mats[si] if si < len(smr_mats) else (smr_mats[0] if smr_mats else 0)
            submeshes.append(
                SubMesh(
                    material=mat_index.get(mat_pid, material_base),
                    index_start=cursor,
                    index_count=int(seg.size),
                )
            )
            cursor += int(seg.size)

        result.meshes.append(
            SkinMesh(
                name=name,
                part=part_name,
                lod=lod,
                vertices=_pack_vertices(positions, normals, tangents, uv0, global_joints, weights),
                indices=(
                    np.concatenate(kept_indices) if kept_indices else np.zeros(0, np.uint32)
                ),
                submeshes=submeshes,
                inverse_bindposes=ibm,
                bound_bones=sorted(set(remap)),
            )
        )

    if not result.meshes:
        raise RuntimeError(f"{bundle_path}: no meshes survived the LOD filter {lods}")
    return result


def load_attachment(
    bundle_path: str,
    bone: int,
    material_base: int,
    lods: Optional[Sequence[int]] = None,
) -> Tuple[List[Attachment], List[Material], Dict[str, object]]:
    """Read a rigid equipment prefab (helmet, facecover, cap) -> attachments + materials.

    Same texture/material handling as `load_part`; the difference is that geometry is unskinned and
    keeps its prefab-local transform, which the viewer applies under the target bone entity.
    """
    import UnityPy
    from UnityPy.helpers.MeshHelper import MeshHandler

    env = UnityPy.load(bundle_path)
    meshes: List[Tuple[object, dict]] = []
    renderers: List[dict] = []
    filters: Dict[int, dict] = {}
    mats_by_pathid: Dict[int, dict] = {}
    texs_by_pathid: Dict[int, object] = {}
    tfs: Dict[int, dict] = {}
    gos: Dict[int, dict] = {}

    for obj in env.objects:
        t = obj.type.name
        if t == "Mesh":
            meshes.append((obj, obj.read_typetree()))
        elif t == "MeshRenderer":
            renderers.append(obj.read_typetree())
        elif t == "MeshFilter":
            filters[obj.path_id] = obj.read_typetree()
        elif t == "Material":
            mats_by_pathid[obj.path_id] = obj.read_typetree()
        elif t == "Texture2D":
            texs_by_pathid[obj.path_id] = obj
        elif t == "Transform":
            tfs[obj.path_id] = obj.read_typetree()
        elif t == "GameObject":
            gos[obj.path_id] = obj.read_typetree()

    materials: List[Material] = []
    images: Dict[str, object] = {}
    mat_index: Dict[int, int] = {}
    for pid, mt in mats_by_pathid.items():
        mat = Material(name=str(mt.get("m_Name", f"material_{pid}")))
        saved = mt.get("m_SavedProperties", {}) or {}
        for entry in saved.get("m_TexEnvs", []) or []:
            slot, val = entry[0], entry[1]
            tex_pid = int((val.get("m_Texture") or {}).get("m_PathID", 0))
            if not tex_pid or tex_pid not in texs_by_pathid:
                continue
            try:
                tex = texs_by_pathid[tex_pid].read()
                img = tex.image
                if img is None:
                    continue
                images[str(tex.m_Name)] = img
                mat.textures[str(slot)] = f"textures/{tex.m_Name}.png"
            except Exception as exc:
                print(f"  [warn] {bundle_path}: texture for {slot} unreadable ({exc})")
        mat_index[pid] = material_base + len(materials)
        materials.append(mat)

    # GameObject -> its Transform, so a mesh can be located within the prefab.
    tf_of_go: Dict[int, dict] = {}
    for tt in tfs.values():
        tf_of_go[int(tt.get("m_GameObject", {}).get("m_PathID", 0))] = tt

    def world_in_prefab(tf: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compose this transform up to the prefab root -> (pos, xyzw quat, scale), viewer space."""
        chain: List[dict] = []
        cur: Optional[dict] = tf
        guard = 0
        while cur is not None and guard < 64:
            chain.append(cur)
            fid = int(cur.get("m_Father", {}).get("m_PathID", 0))
            cur = tfs.get(fid)
            guard += 1
        m = np.eye(4)
        for node in reversed(chain):
            lp = node.get("m_LocalPosition", {})
            lr = node.get("m_LocalRotation", {})
            ls = node.get("m_LocalScale", {})
            from .skeleton import _trs

            m = m @ _trs(
                coords.point((lp.get("x", 0.0), lp.get("y", 0.0), lp.get("z", 0.0))),
                coords.quat((lr.get("x", 0.0), lr.get("y", 0.0), lr.get("z", 0.0), lr.get("w", 1.0))),
                (ls.get("x", 1.0), ls.get("y", 1.0), ls.get("z", 1.0)),
            )
        pos = m[:3, 3].copy()
        basis = m[:3, :3]
        scale = np.linalg.norm(basis, axis=0)
        scale[scale < 1e-8] = 1.0
        rot = basis / scale[None, :]
        # rotation matrix -> xyzw
        from .clips import _matrix_to_quat

        q = _matrix_to_quat(rot[None, :, :])[0]
        return pos, q, scale

    mesh_to_go: Dict[int, int] = {}
    for f in filters.values():
        mpid = int((f.get("m_Mesh") or {}).get("m_PathID", 0))
        if mpid:
            mesh_to_go[mpid] = int(f.get("m_GameObject", {}).get("m_PathID", 0))
    rend_by_go: Dict[int, dict] = {
        int(r.get("m_GameObject", {}).get("m_PathID", 0)): r for r in renderers
    }

    out: List[Attachment] = []
    for obj, tt in meshes:
        name = str(tt.get("m_Name", "mesh"))
        lod = _lod_from_name(name)
        if lods is not None and lod not in lods:
            continue
        # These prefabs ship both a `_base` and a `_custom` variant of the same mesh; taking both
        # would draw the item twice.
        if name.lower().endswith("_custom"):
            continue

        mesh = obj.read()
        handler = MeshHandler(mesh)
        handler.process()
        if not handler.m_Vertices or not handler.m_IndexBuffer:
            continue
        positions = coords.points(np.asarray(handler.m_Vertices, np.float32).reshape(-1, 3))
        v = positions.shape[0]
        normals = (
            coords.normals(np.asarray(handler.m_Normals, np.float32).reshape(v, -1)[:, :3])
            if handler.m_Normals
            else None
        )
        tangents = (
            coords.tangents(np.asarray(handler.m_Tangents, np.float32).reshape(v, -1))
            if handler.m_Tangents
            else None
        )
        uv0 = (
            coords.uvs(np.asarray(handler.m_UV0, np.float32).reshape(v, -1)[:, :2])
            if handler.m_UV0
            else None
        )
        # Rigid: pin every vertex to the target bone with full weight so the same shader path serves
        # skinned and rigid geometry.
        ji = np.zeros((v, 4), np.uint16)
        jw = np.zeros((v, 4), np.float32)
        jw[:, 0] = 1.0

        go = mesh_to_go.get(obj.path_id, 0)
        tf = tf_of_go.get(go)
        if tf is None:
            continue
        pos, rot, scale = world_in_prefab(tf)

        all_indices = np.asarray(handler.m_IndexBuffer, np.uint32)
        rend = rend_by_go.get(go, {})
        rmats = [int((m or {}).get("m_PathID", 0)) for m in (rend.get("m_Materials") or [])]
        index_size = 2 if tt.get("m_IndexFormat", 0) == 0 else 4
        subs: List[SubMesh] = []
        segs: List[np.ndarray] = []
        cursor = 0
        for si, sm in enumerate(tt.get("m_SubMeshes") or []):
            first = int(sm.get("firstByte", 0)) // index_size
            count = int(sm.get("indexCount", 0))
            base = int(sm.get("baseVertex", 0) or 0)
            seg = coords.flip_winding(all_indices[first : first + count] + base)
            segs.append(seg)
            mp = rmats[si] if si < len(rmats) else (rmats[0] if rmats else 0)
            subs.append(
                SubMesh(
                    material=mat_index.get(mp, material_base),
                    index_start=cursor,
                    index_count=int(seg.size),
                )
            )
            cursor += int(seg.size)

        out.append(
            Attachment(
                name=name,
                bone=bone,
                lod=lod,
                local_pos=[float(x) for x in pos],
                local_rot=[float(x) for x in rot],
                local_scale=[float(x) for x in scale],
                vertices=_pack_vertices(positions, normals, tangents, uv0, ji, jw),
                indices=np.concatenate(segs) if segs else np.zeros(0, np.uint32),
                submeshes=subs,
            )
        )
    return out, materials, images
