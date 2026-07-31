"""`characters/character/skeleton.bundle` -> the canonical EFT character rig.

One rig serves every character in the game: PMC, scav, and every boss bind to the same 79-node
`Base Human*` biped. Extracting it once and keying everything else off it is what makes the pack
format character-agnostic.

Bone ORDER is the depth-first order of the transform hierarchy from the single root. That is stable
across runs (children come from `m_Children`, which is ordered in the asset) and it guarantees
`parents[i] < i`, so a consumer can compute world matrices in one forward pass with no sorting --
which is exactly what the viewer's skeleton update does.

Everything is emitted in VIEWER space (see coords.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import coords
from .unity_bind import build_hash_map, path_hash


@dataclass
class Skeleton:
    """The canonical rig, in viewer space, depth-first ordered."""

    names: List[str] = field(default_factory=list)
    #: Transform path of each bone RELATIVE TO THE RIG ROOT, e.g. "Root_Joint/Base HumanPelvis".
    #: The root itself has the empty path. This is not cosmetic: `Skin._bonePaths` and Mecanim clip
    #: bindings are both relative to the GameObject carrying the Animator (the rig root), so
    #: including the root's own name here would make every join miss. The empty path also matches
    #: Unity's own convention -- CRC32("") == 0 is how a clip denotes "my root transform".
    paths: List[str] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)  #: -1 for the root
    local_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    local_rot: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    local_scale: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))

    def __len__(self) -> int:
        return len(self.names)

    @property
    def by_path(self) -> Dict[str, int]:
        return {p: i for i, p in enumerate(self.paths)}

    @property
    def by_name(self) -> Dict[str, int]:
        """name -> index. Names are unique in this rig; asserted at build time."""
        return {n: i for i, n in enumerate(self.names)}

    @property
    def by_hash(self) -> Dict[int, int]:
        """CRC32(path) -> bone index. The join key for clip bindings."""
        return {path_hash(p): i for i, p in enumerate(self.paths)}

    def index_of_path(self, path: str) -> Optional[int]:
        return self.by_path.get(path)

    def world_matrices(self) -> np.ndarray:
        """(N, 4, 4) bind-pose world matrices, forward pass. Useful for validation/debug."""
        out = np.zeros((len(self), 4, 4), np.float64)
        for i in range(len(self)):
            local = _trs(self.local_pos[i], self.local_rot[i], self.local_scale[i])
            p = self.parents[i]
            out[i] = local if p < 0 else out[p] @ local
        return out

    def to_manifest(self) -> dict:
        return {
            "boneCount": len(self),
            "names": self.names,
            "paths": self.paths,
            "parents": self.parents,
            "localPos": self.local_pos.tolist(),
            "localRot": self.local_rot.tolist(),
            "localScale": self.local_scale.tolist(),
        }


def _trs(p: Sequence[float], q: Sequence[float], s: Sequence[float]) -> np.ndarray:
    """position + xyzw quaternion + scale -> 4x4."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n > 0.0:
        k = 2.0 / n
    else:
        k = 0.0
    xx, yy, zz = x * x * k, y * y * k, z * z * k
    xy, xz, yz = x * y * k, x * z * k, y * z * k
    wx, wy, wz = w * x * k, w * y * k, w * z * k
    r = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r * np.asarray(s, np.float64)[None, :]
    m[:3, 3] = np.asarray(p, np.float64)
    return m


def load_skeleton(bundle_path: str) -> Skeleton:
    """Read skeleton.bundle -> `Skeleton`, conjugated into viewer space.

    Raises if the bundle does not contain exactly one root transform, or if bone names collide --
    both would break the path joins that everything else depends on.
    """
    import UnityPy

    env = UnityPy.load(bundle_path)

    # path_id -> (transform typetree, owning GameObject name)
    tfs: Dict[int, dict] = {}
    go_names: Dict[int, str] = {}
    for obj in env.objects:
        if obj.type.name == "Transform":
            tfs[obj.path_id] = obj.read_typetree()
        elif obj.type.name == "GameObject":
            go_names[obj.path_id] = obj.read_typetree().get("m_Name", "")

    if not tfs:
        raise RuntimeError(f"no Transform objects in {bundle_path}")

    def name_of(tf: dict) -> str:
        return go_names.get(int(tf.get("m_GameObject", {}).get("m_PathID", 0)), "")

    roots = [pid for pid, tf in tfs.items() if int(tf.get("m_Father", {}).get("m_PathID", 0)) == 0]
    if len(roots) != 1:
        raise RuntimeError(
            f"{bundle_path}: expected exactly 1 root transform, found {len(roots)} "
            f"({[name_of(tfs[r]) for r in roots]})"
        )

    skel = Skeleton()
    pos: List[List[float]] = []
    rot: List[List[float]] = []
    scl: List[List[float]] = []

    def visit(pid: int, parent_index: int, parent_path: Optional[str]) -> None:
        tf = tfs[pid]
        nm = name_of(tf)
        # parent_path is None only for the rig root, whose own path is "" (see Skeleton.paths).
        if parent_path is None:
            path = ""
        elif parent_path == "":
            path = nm
        else:
            path = f"{parent_path}/{nm}"
        idx = len(skel.names)
        skel.names.append(nm)
        skel.paths.append(path)
        skel.parents.append(parent_index)

        lp = tf.get("m_LocalPosition", {})
        lr = tf.get("m_LocalRotation", {})
        ls = tf.get("m_LocalScale", {})
        pos.append(list(coords.point((lp.get("x", 0.0), lp.get("y", 0.0), lp.get("z", 0.0)))))
        rot.append(
            list(
                coords.quat(
                    (lr.get("x", 0.0), lr.get("y", 0.0), lr.get("z", 0.0), lr.get("w", 1.0))
                )
            )
        )
        scl.append([float(ls.get("x", 1.0)), float(ls.get("y", 1.0)), float(ls.get("z", 1.0))])

        for child in tf.get("m_Children", []) or []:
            cid = int(child.get("m_PathID", 0))
            if cid in tfs:
                visit(cid, idx, path)

    visit(roots[0], -1, None)

    skel.local_pos = np.asarray(pos, np.float32).reshape(-1, 3)
    skel.local_rot = np.asarray(rot, np.float32).reshape(-1, 4)
    skel.local_scale = np.asarray(scl, np.float32).reshape(-1, 3)

    if len(skel) != len(tfs):
        raise RuntimeError(
            f"{bundle_path}: walked {len(skel)} of {len(tfs)} transforms -- the hierarchy is "
            f"disjoint, which would silently drop bones"
        )
    dupes = {n for n in skel.names if skel.names.count(n) > 1}
    if dupes:
        raise RuntimeError(f"{bundle_path}: duplicate bone names {sorted(dupes)}")
    # Guarantee the forward-pass invariant the viewer relies on.
    for i, p in enumerate(skel.parents):
        if p >= i:
            raise RuntimeError(f"bone {i} ({skel.names[i]}) has parent {p} >= {i}")

    # Path hashes must be unique or clip bindings cannot be resolved.
    build_hash_map(skel.paths)
    return skel
