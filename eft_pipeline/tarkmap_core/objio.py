"""OBJ + vert-paint sidecar loaders for the assembler. Split out of assemble_instanced.py (2026-07-01, verbatim)."""
import os
import numpy as np

# F3: fast binary OBJ parser. EFT meshes are all-triangle v/vt/vn with faces `a/b/c a/b/c a/b/c`, so the
# per-line Python loop + per-token int/float parse below is replaced by a bulk numpy parse. Any face that is
# not a clean 3-vertex a/b/c triangle (>3 tokens, `a//c`, `a/b`, `a`) makes _parse_obj_fast bail to None and
# _load_obj_slow (the ORIGINAL parser, verbatim) runs -> byte-identical fallback. EFT_OBJ_FASTPARSE=0 forces slow.
_OBJ_FASTPARSE = os.environ.get('EFT_OBJ_FASTPARSE', '1') != '0'


def _load_obj_slow(p):
    """The original pure-Python OBJ parser (verbatim). Used directly when EFT_OBJ_FASTPARSE=0 and as the
    correctness fallback for any OBJ the fast path declines (non-triangle / non-`a/b/c` faces, irregular v/vt)."""
    V, VT, F = [], [], []
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


def _parse_obj_fast(data):
    """Bulk-parse OBJ bytes -> (V,VT,F) IFF every face is a clean 3-vertex `a/b/c` triangle and every v/vt line
    has exactly 3/2 tokens; else None (caller falls back). Byte-identical to _load_obj_slow: v/vt go through the
    SAME numpy string->f32 conversion; face vert/uv indices are exact integers. Rebuilding joins from split()
    tokens strips any CR so CRLF files parse identically without touching the numeric conversion."""
    lines = data.split(b'\n')
    vrem = []; vtrem = []; frem = []
    for ln in lines:                                    # one cheap classify pass (no per-token parse here)
        h2 = ln[:2]
        if h2 == b'v ': vrem.append(ln[2:])
        elif h2 == b'f ': frem.append(ln[2:])
        elif ln[:3] == b'vt ': vtrem.append(ln[3:])
    # --- vertices: require exactly 3 tokens/line (matches the slow path's [:3] on all-3-coord EFT meshes) ---
    if vrem:
        vtoks = b' '.join(vrem).split()
        if len(vtoks) != 3 * len(vrem): return None
        Va = np.fromstring(b' '.join(vtoks), sep=' ', dtype=np.float32).reshape(-1, 3)
    else:
        Va = np.zeros((0, 3), np.float32)
    # --- texcoords: exactly 2 tokens/line; empty -> (1,2) zeros exactly like the slow path ---
    if vtrem:
        ttoks = b' '.join(vtrem).split()
        if len(ttoks) != 2 * len(vtrem): return None
        VTa = np.fromstring(b' '.join(ttoks), sep=' ', dtype=np.float32).reshape(-1, 2)
    else:
        VTa = np.zeros((1, 2), np.float32)
    # --- faces: all-triangle `v/t/n`. Bail on `//` (missing uv) or any non-3-token line -> slow path ---
    if frem:
        nf = len(frem)
        fjoin = b' '.join(frem)
        if b'//' in fjoin: return None
        ftoks = fjoin.split()
        if len(ftoks) != 3 * nf: return None                        # quad/ngon/degenerate -> slow (fan triangulation)
        fi = np.fromstring(b' '.join(ftoks).replace(b'/', b' '), dtype=np.int64, sep=' ')
        if fi.size != 9 * nf: return None                           # not exactly v/t/n per corner -> slow
        Fa = (fi.reshape(nf, 3, 3)[:, :, :2] - 1).astype(np.int32)  # keep (vert,uv), drop normal; -1 like slow path
    else:
        Fa = np.zeros((0, 3, 2), np.int32)
    return Va, VTa, Fa


def load_obj(ds, fn):
    """Parse a dataset OBJ -> (V[nv,3] f32, VT[nt,2] f32, F[nf,3,2] i32 of (vert,uv) index pairs), or None if missing."""
    p = os.path.join(ds, 'meshes', fn)
    if not os.path.exists(p): return None
    if _OBJ_FASTPARSE:
        try:
            with open(p, 'rb') as fh:
                fast = _parse_obj_fast(fh.read())
            if fast is not None: return fast
        except Exception:
            pass                                        # any surprise -> exact original parser below
    return _load_obj_slow(p)


def load_vcol(ds, fn):
    """Per-vertex Vert Paint blend weights (extractor sidecar), aligned with the OBJ vertices; or None."""
    p = os.path.join(ds, 'meshes', fn[:-4] + '.vcol.npy')
    if not os.path.exists(p): return None
    try:
        c = np.load(p).astype(np.float32)
        return (c / 255.0) if (c.size and c.max() > 1.5) else c            # normalize 0..255 -> 0..1
    except Exception: return None
