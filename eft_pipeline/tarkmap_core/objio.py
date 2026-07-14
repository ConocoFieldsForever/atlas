"""OBJ + vert-paint sidecar loaders for the assembler. Split out of assemble_instanced.py (2026-07-01, verbatim)."""
import os
import numpy as np


def load_obj(ds, fn):
    """Parse a dataset OBJ -> (V[nv,3] f32, VT[nt,2] f32, F[nf,3,2] i32 of (vert,uv) index pairs), or None if missing."""
    p = os.path.join(ds, 'meshes', fn); V, VT, F = [], [], []
    if not os.path.exists(p): return None
    with open(p) as fh:
        for line in fh:
            if line[:2] == 'v ': V.append(line[2:].split()[:3])
            elif line[:3] == 'vt ': VT.append(line[3:].split()[:2])
            elif line[:2] == 'f ':
                idx = [(int(a[0]) - 1, (int(a[1]) - 1) if len(a) > 1 and a[1] else -1)
                       for a in (tok.split('/') for tok in line[2:].split())]
                for k in range(1, len(idx) - 1): F.append((idx[0], idx[k], idx[k + 1]))
    Va = np.array(V, np.float32).reshape(-1, 3) if V else np.zeros((0, 3), np.float32)
    VTa = np.array(VT, np.float32).reshape(-1, 2) if VT else np.zeros((1, 2), np.float32)
    Fa = np.array(F, np.int32).reshape(-1, 3, 2) if F else np.zeros((0, 3, 2), np.int32)
    return Va, VTa, Fa


def load_vcol(ds, fn):
    """Per-vertex Vert Paint blend weights (extractor sidecar), aligned with the OBJ vertices; or None."""
    p = os.path.join(ds, 'meshes', fn[:-4] + '.vcol.npy')
    if not os.path.exists(p): return None
    try:
        c = np.load(p).astype(np.float32)
        return (c / 255.0) if (c.size and c.max() > 1.5) else c            # normalize 0..255 -> 0..1
    except Exception: return None
