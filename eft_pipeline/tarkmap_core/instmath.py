"""Instance-placement math for the assembler: the handedness conjugation, TRS decomposition (shear detection) and
world-baking of non-TRS instances. Split out of assemble_instanced.py (2026-07-01, verbatim).

THE RULES (see the tarkov-unity-extraction skill §1/§3 — regressions here mirrored the whole map before):
- Global orientation fix = the PROVEN similarity-conjugation: place each instance with M' = G @ M @ G^-1 applied to the
  RAW (unreflected) mesh verts. det(M')=det(M) so instances stay >0 (no inside-out), textures are NOT locally mirrored.
  DO NOT also reflect the mesh verts (V@G3.T) — that double-applies G and mirrors every surface locally (the old bug).
- NEVER TRS-decompose a sheared matrix (silently drops the off-diagonal shear) — bake it to world geometry instead.
"""
import numpy as np


def make_conjugator(G4):
    """Returns (apply_global, det3, GID, GDET) bound to the map's global coordinate matrix."""
    G3 = G4[:3, :3].astype(np.float64); G4I = np.linalg.inv(G4)
    GID = bool(np.allclose(G4, np.eye(4))); GDET = float(np.linalg.det(G3))

    def _m(m): return np.array(m, np.float64).reshape(4, 4)
    def _rm(M): return [float(M[r, c]) for r in range(4) for c in range(4)]

    def apply_global(m):
        return m if GID else _rm(G4 @ _m(m) @ G4I)                          # conjugation keeps instances det>0

    def det3(m): return float(np.linalg.det(_m(m)[:3, :3]))

    return apply_global, det3, GID, GDET


def revwind(F): return F[:, [0, 2, 1], :] if len(F) else F


def trs(m):
    """row-major 16 -> (T, quat xyzw, S) for EXT_mesh_gpu_instancing. M3 = R@diag(S); winding flip if det<0."""
    M = np.array([[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]], np.float64)
    T = np.array([m[3], m[7], m[11]], np.float64)
    S = np.linalg.norm(M, axis=0); S[S == 0] = 1e-8
    R = M / S
    if np.linalg.det(R) < 0: S[0] *= -1; R[:, 0] *= -1                  # reflection -> absorb (P2: flip winding)
    ortho = float(np.abs(R.T @ R - np.eye(3)).max())                    # >.02 => sheared, TRS can't represent (mall floors)
    q = np.empty(4); tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1); q[3] = 0.25 / s; q[0] = (R[2, 1] - R[1, 2]) * s; q[1] = (R[0, 2] - R[2, 0]) * s; q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]]); j = (i + 1) % 3; k = (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s; q[3] = (R[k, j] - R[j, k]) / s
    return T.astype(np.float32), (q / np.linalg.norm(q)).astype(np.float32), S.astype(np.float32), ortho


def mat4_colmajor(m):
    """row-major scene.json 16 -> glTF node.matrix (column-major); exact for sheared instances."""
    return [m[0], m[4], m[8], 0.0, m[1], m[5], m[9], 0.0, m[2], m[6], m[10], 0.0, m[3], m[7], m[11], 1.0]


def bake_into(baked, prim_raw, m):
    """Bake a SHEARED instance to WORLD geometry (world = V@M3.T + T, exactly like soup_build). Node matrices get
    stripped by gltf-transform meshopt quantization, so sheared/non-similarity instances must live in world coords."""
    M3 = np.array([[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]], np.float64); T = np.array([m[3], m[7], m[11]], np.float64)
    flip = np.linalg.det(M3) < 0.0                                          # reflected matrix -> reverse winding so faces aren't inside-out
    try:
        M3iT = np.linalg.inv(M3).T
    except np.linalg.LinAlgError:
        M3iT = np.linalg.pinv(M3).T                                        # DEGENERATE instance (a mesh flattened to a plane -> rank-deficient 3x3, e.g. Streets billboards/decals baked as sheared): the pseudo-inverse gives a best-effort normal transform instead of crashing the whole map build
    for texid, pos, nrm, uv, ind in prim_raw:
        wpos = (pos @ M3.T + T).astype(np.float32)
        wn = nrm @ M3iT.T; wn = (wn / np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-9)).astype(np.float32)
        oi = ind[:, [0, 2, 1]] if flip else ind
        b = baked.setdefault(texid, {'pos': [], 'nrm': [], 'uv': [], 'idx': [], 'voff': 0})
        b['idx'].append(oi + b['voff']); b['pos'].append(wpos); b['nrm'].append(wn); b['uv'].append(uv); b['voff'] += len(wpos)
