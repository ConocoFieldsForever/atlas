"""Unity world -> viewer world conjugation for character data. THE ONLY PLACE THIS LIVES.

The map pipeline bridges Unity to viewer space with G3 = diag(-1, 1, 1) (see
extraction/intel/build_tasks.py, extract_gamedata.py). A character has to land in the SAME space or
it will not stand on the floor the walk camera stands on, so the identical G3 is applied here.

G3 is a REFLECTION (det = -1). That has three consequences people get wrong:

  * A rotation conjugated by a reflection is still a rotation: R' = G R G⁻¹, det(R') = det(R) = 1.
    In quaternion terms the axis is a pseudovector, so it picks up the reflection's sign:
    a -> det(G)·(G a) = -(G a). For G = diag(-1,1,1) that is (x, -y, -z), angle unchanged.
  * Every triangle's winding is mirrored. The character goes through Bevy's ordinary PBR path where
    back-face culling is ON, so the winding must be REVERSED in the index buffer. (The map's
    gpu_driven path instead draws double-sided with a cofactor normal matrix — a different valid
    answer to the same problem. Do not mix the two.)
  * A tangent's handedness (w) flips.

Scale is untouched: conjugating a diagonal scale by a diagonal reflection returns it unchanged.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Unity world -> viewer world. Read logically from the map configs' coordinates.global_matrix.
G3 = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
G3_DET = -1.0

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def point(p: Sequence[float]) -> Vec3:
    """Unity position -> viewer position. (x, y, z) -> (-x, y, z)."""
    return (-float(p[0]), float(p[1]), float(p[2]))


def points(arr: np.ndarray) -> np.ndarray:
    """Vectorised `point` over an (N, 3) array. Also correct for normals under this G3."""
    out = np.asarray(arr, np.float32).copy()
    out[:, 0] *= -1.0
    return out


def normals(arr: np.ndarray) -> np.ndarray:
    """Normals transform by (G⁻¹)ᵀ = G for this diagonal G, then stay unit length."""
    return points(arr)


def tangents(arr: np.ndarray) -> np.ndarray:
    """(N, 4) tangents: reflect xyz, negate the handedness sign in w."""
    out = np.asarray(arr, np.float32).copy()
    out[:, 0] *= -1.0
    if out.shape[1] >= 4:
        out[:, 3] *= -1.0
    return out


def uvs(arr: np.ndarray) -> np.ndarray:
    """(N, 2) UVs: flip V. Unity's origin is bottom-left, Bevy/wgpu's is top-left.

    This matches the map pipeline's contract exactly — `.eftpack` declares `uvVFlipBaked: true`
    ("UV V was already flipped into the vertex UVs -> the shader must NOT re-flip"), so the character
    pack bakes it the same way and neither consumer has to know which asset it is looking at.
    Skipping it does not garble the mesh, it just samples the texture upside down, which on a face
    reads as a subtle-but-wrong "UV mapping issue" rather than an obvious break.
    """
    out = np.asarray(arr, np.float32).copy()
    out[:, 1] = 1.0 - out[:, 1]
    return out


def quat(q: Sequence[float]) -> Quat:
    """Unity rotation (x, y, z, w) -> viewer rotation. Conjugation by diag(-1,1,1)."""
    return (float(q[0]), -float(q[1]), -float(q[2]), float(q[3]))


def quats(arr: np.ndarray) -> np.ndarray:
    """Vectorised `quat` over an (N, 4) xyzw array."""
    out = np.asarray(arr, np.float32).copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def matrix4(m: np.ndarray) -> np.ndarray:
    """Conjugate a 4x4 affine: M' = G M G⁻¹ (G⁻¹ == G for this reflection).

    Used for inverse bindposes, which must live in the same space as the bones that drive them.
    """
    g = np.eye(4, dtype=np.float64)
    g[:3, :3] = G3
    return (g @ np.asarray(m, np.float64) @ g).astype(np.float32)


def flip_winding(indices: np.ndarray) -> np.ndarray:
    """Reverse triangle winding: (a, b, c) -> (a, c, b).

    Required because G3 mirrors the mesh. Input length must be a multiple of 3.
    """
    idx = np.asarray(indices, np.uint32)
    if idx.size % 3:
        raise ValueError(f"index count {idx.size} is not a multiple of 3")
    tris = idx.reshape(-1, 3)
    return tris[:, [0, 2, 1]].reshape(-1).copy()


def conventions() -> dict:
    """The block written into manifest.json so the Rust loader can assert what it was handed."""
    return {
        "coordSystem": "viewer",
        "g3": [-1.0, 1.0, 1.0],
        "quatOrder": "xyzw",
        "windingFlipped": True,
        "tangentHandednessFlipped": True,
        #: Same meaning as .eftpack's flag: V is already flipped in the vertex UVs, do NOT re-flip.
        "uvVFlipBaked": True,
        "upAxis": "y",
        "unit": "meter",
    }
