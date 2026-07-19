#!/usr/bin/env python
"""RTX Phase 2 -- re-bake the world-space L1 SH irradiance volume with the FILTERED live light set,
physical falloff, UV-mapped surface albedo and ONE diffuse bounce (+ emissive-surface gather).

  python extraction/bake/bake_volume2.py <map> [--rebuild-mesh]     (requires EFT_TARKMAP_ROOT; NVIDIA CUDA GPU)

What's new vs bake_irradiance_volume.py (v1 "RTX Phase 1"):
  * LIGHTS: reads the FILTERED live-set export (eft_assets/<dataset>/lights_*.json, e.g. lights_64.json =
    1,285 LIVE lights on Interchange; the old lights_level64.json 3,715 set was UNFILTERED -- lights that
    are off in-game). Spot cone = FULL angle from Unity (halved here), color LINEAR, physical inverse-square
    falloff with a smooth range window (matches the Blender import: real point/spot lamps + cutoff_distance).
  * BAKE MESH: built from scene.json with the SAME culls + LOD0 filter the viewer geometry uses, with
    PER-VERTEX UV-MAPPED ALBEDO: every vertex samples its sub's albedo texture at the vertex UV (per-sub
    [sx,sy,ox,oy] tiling applied; SOURCE PNGs sampled with Unity's bottom-left V origin -- the pipeline
    flips the TEXTURE for glTF, never the UVs), x material col tint, sRGB->linear. A red container bounces
    red; painted asphalt bounces dark; terrain bounces its baked-albedo slice. Emissive subs additionally
    carry per-vertex emissive = emissive_map(uv) * emisCol, so a fixture whose emissive map lights only a
    strip only GLOWS on that strip (coverage-weighted by construction). glass/decal subs are excluded from
    the occluder (skylights must not block sky; decals hug their base surface).
  * BOUNCE: one diffuse bounce -- pass A bakes direct (neutral sky + shadow-tested practicals) into the
    probe grid; pass B re-traces the same Fibonacci dirs and, at each hit, gathers albedo/pi * E(hit,normal)
    (E = irradiance from the flood-filled pass-A grid, trilinear) + per-vertex emissive. Sky stays NEUTRAL
    (grayscale gradient, luma-matched to v1) -- the Cycles reference is neutral Nishita; no green/blue tint.
  * RESOLUTION: ~4x the XZ probe density of v1 (~3.2m vs 13m; Y 4m), auto-widened to keep ny*nz<=8192
    (WebGL 2D texture height) and the probe count sane. The WebGPU viewer packs z-slices as a TILED
    flipbook (multiple rows) when nx*nz exceeds 8192 (_gi.js/_wao.js).

Outputs (format-compatible with volume.json/bin -- same fields, same probe order, same 12-half layout):
  out/<map>/volume2.json / volume2.bin      (validated first; promote over volume.json/bin after)
  out/<map>/volume2_debug_slice.png         mid-Y + ground-Y DC luminance slices
  out/<map>/_bakemesh2/{V,F,ALB,EMI}.npy    cached albedo bake mesh (delete or --rebuild-mesh to rebuild)

Everything is map-agnostic: culls/coordinates from maps/<id>/config.json, lights discovered per-dataset,
grid from geometry percentiles. No mesh/map-name special cases.
"""
import os, sys, json, time, math, glob, functools
import numpy as np
print = functools.partial(print, flush=True)
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass

# portable kit: the correctness core (MapConfig/Culls/matsig/objio) is the copy VENDORED in this
# repo at eft_pipeline/tarkmap_core (same code the pack emitter uses). config.py there honors
# EFT_TARKMAP_ROOT for maps/<id>/config.json + dataset resolution -- set it (see extraction/README.md).
HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/extraction/bake
REPO = os.path.dirname(os.path.dirname(HERE))              # <repo> (eft_native_viewer)
sys.path.insert(0, REPO)
from eft_pipeline.tarkmap_core.config import MapConfig
from eft_pipeline.tarkmap_core.culls import Culls
from eft_pipeline.tarkmap_core.matsig import sub_sig
from eft_pipeline.tarkmap_core.objio import load_obj

TK = os.environ.get("EFT_TARKMAP_ROOT")                    # <workspace>/tarkmap (maps/ + out/)
if not TK:
    raise SystemExit("bake_volume2: EFT_TARKMAP_ROOT is not set. Point it at your workspace tarkmap dir "
                     "(the one holding maps/ and out/), e.g.  setx EFT_TARKMAP_ROOT D:\\eft_work\\tarkmap  "
                     "-- see extraction/README.md")
ROOT = os.path.dirname(TK)                                 # workspace root (holds eft_assets/)

MAP = next((a for a in sys.argv[1:] if not a.startswith('-')), 'interchange')
REBUILD = '--rebuild-mesh' in sys.argv
OUTDIR = os.path.join(TK, 'out', MAP)
CACHE = os.path.join(OUTDIR, '_bakemesh2')

# ---- tunables (env-overridable calibration knobs, same convention as v1) ----
SKY_SCALE = float(os.environ.get('SKY_SCALE', '1.0'))
SUN_SCALE = float(os.environ.get('SUN_SCALE', '1.0'))
LIGHT_SCALE = float(os.environ.get('LIGHT_SCALE', '6.0'))   # Unity (color*intensity) -> SH radiance units, physical 1/d^2 falloff
EMIS_GAIN = float(os.environ.get('EMIS_GAIN', '1.0'))       # emissive-surface radiance gain in the bounce gather
ALBEDO_BOOST = float(os.environ.get('ALBEDO_BOOST', '1.0'))
N_DIR = int(os.environ.get('N_DIR', '256'))                 # Fibonacci dirs per probe, both passes
XZ_TARGET = float(os.environ.get('XZ_TARGET', '3.0'))       # target XZ probe spacing floor (m); v1 was 13m
Y_SPACING = float(os.environ.get('Y_SPACING', '4.0'))
EMIS_MAX = 8.0                                              # per-vertex emissive u8 quantization ceiling (linear)
MIN_D2 = 0.25                                               # clamp 1/d^2 within 0.5m of a bulb (probe-on-lamp)
TEX_DIM = 160                                               # albedo sample resolution (bounce-grade)


# ================================================================================================
# PART 1 -- UV-mapped per-vertex-albedo bake mesh (viewer/glb world space)
# ================================================================================================
def _srgb_lut():
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip((x ** 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
_LUT = _srgb_lut()


class TexCache:
    def __init__(self, texdir):
        self.dir = texdir; self.c = {}
    def get(self, name):
        """-> (HxWx3 uint8 LINEAR albedo array, mean linear rgb float3) or None."""
        if name in self.c: return self.c[name]
        res = None
        try:
            from PIL import Image
            p = os.path.join(self.dir, name + '.png')
            im = Image.open(p).convert('RGB')
            w, h = im.size; s = max(w, h) / float(TEX_DIM)
            if s > 1.0: im = im.resize((max(1, int(w / s)), max(1, int(h / s))), Image.BILINEAR)
            arr = _LUT[np.asarray(im, np.uint8)]                       # sRGB -> linear, u8
            res = (arr, arr.reshape(-1, 3).mean(0) / 255.0)
        except Exception:
            res = None
        self.c[name] = res
        return res


def _sample(arr, uv):
    """Nearest sample of HxWx3 u8 with WRAP, Unity bottom-left V origin (the pipeline flips the
    TEXTURE for glTF and leaves UVs alone, so the SOURCE png row = (1 - frac(v)) * H)."""
    H, W = arr.shape[:2]
    fu = uv[:, 0] - np.floor(uv[:, 0]); fv = uv[:, 1] - np.floor(uv[:, 1])
    ix = np.minimum((fu * W).astype(np.int32), W - 1)
    iy = np.minimum(((1.0 - fv) * H).astype(np.int32), H - 1)
    return arr[iy, ix]


def build_bake_mesh(cfg):
    t0 = time.time()
    DS = cfg.dataset
    if not os.path.isabs(DS): DS = os.path.join(ROOT, DS)
    sc = json.load(open(os.path.join(DS, 'scene.json')))['instances']
    culls = Culls(cfg.get('cull'))
    kept, rep = culls.filter(sc)
    print(f"[mesh] culls kept {rep['kept']:,}/{rep['raw']:,} instances")
    # LOD0 filter -- scene.json is ALL-LOD; the shipped viewer geometry is finest + untagged (assemble --payload lod0)
    def _lod0(it):
        L = it.get('lod'); i = L['i'] if (L and L.get('i', -1) >= 0) else None
        return i is None or i == 0
    n0 = len(kept); kept = [it for it in kept if _lod0(it)]
    print(f"[mesh] LOD0 filter -> {len(kept):,}/{n0:,} instances")

    G4 = np.array(cfg.coord_matrix(), np.float64).reshape(4, 4)
    G4i = np.linalg.inv(G4)

    tex = TexCache(os.path.join(DS, 'tex'))
    groups = {}
    for it in kept: groups.setdefault((it['mesh'], sub_sig(it['subs'])), []).append(it)
    print(f"[mesh] {len(groups):,} (mesh, material-sig) groups")

    Vl, Fl, Al, El = [], [], [], []
    voff = 0; ntex_miss = 0; nuv_miss = 0
    spot = {}                                                  # texture name -> (sum rgb, count) for the albedo spot-check
    obj_cache = {}
    t_rep = time.time()
    for gi_, ((mname, _sig), insts) in enumerate(groups.items()):
        if mname not in obj_cache:
            if len(obj_cache) > 4096: obj_cache.clear()        # bound RAM; groups of one mesh are adjacent-ish anyway
            obj_cache[mname] = load_obj(DS, mname)
        lo = obj_cache[mname]
        if lo is None: continue
        V, VT, F = lo
        if not len(F): continue
        subs = insts[0]['subs']
        # ---- per-sub: faces are CONSECUTIVE runs of sb['n'] (exactly assemble_instanced's walk) ----
        gp, gf, ga, ge = [], [], [], []
        gvo = 0; f0 = 0
        for sb in subs:
            n = sb.get('n', -1); n = (len(F) - f0) if n < 0 else n
            if n <= 0 or f0 + n > len(F): f0 += max(n, 0); continue
            role = sb.get('role', 'opaque')
            if not culls.keep_submesh(sb) or role in ('glass', 'decal'):     # shadow proxies / skylight glass / hugging decals
                f0 += n; continue
            cor = F[f0:f0 + n]; f0 += n
            vi = cor[:, :, 0].reshape(-1); ti = cor[:, :, 1].reshape(-1)
            key = vi.astype(np.int64) * (len(VT) + 1) + (ti.astype(np.int64) + 1)
            uk, idx0, inv = np.unique(key, return_index=True, return_inverse=True)
            pos = V[vi[idx0]].astype(np.float32)
            has_uv = ti[idx0] >= 0
            uv = np.where(has_uv[:, None], VT[np.clip(ti[idx0], 0, len(VT) - 1)], 0.0).astype(np.float32)
            sx, sy, ox, oy = sb.get('uv', [1, 1, 0, 0]); uv = uv * [sx, sy] + [ox, oy]
            col = np.array((sb.get('col') or [1, 1, 1, 1])[:3], np.float32)
            # ---- UV-mapped albedo (linear u8) ----
            tx = tex.get(sb['tex']) if sb.get('tex') else None
            if tx is not None:
                arr, mean = tx
                a = _sample(arr, uv).astype(np.float32) / 255.0
                if not has_uv.all():                                        # no UV channel -> texture mean
                    a[~has_uv] = mean; nuv_miss += int((~has_uv).sum())
                a *= col
                s = spot.setdefault(sb['tex'], [np.zeros(3), 0]); s[0] += a.sum(0); s[1] += len(a)
            else:
                if sb.get('tex'): ntex_miss += 1
                a = np.tile(np.clip(col, 0, 1), (len(pos), 1)).astype(np.float32) * 0.5   # untextured: tinted mid albedo
            ga.append(np.clip(a * 255.0, 0, 255).astype(np.uint8))
            # ---- UV-mapped emissive (linear u8 / EMIS_MAX): coverage-weighted by sampling the map itself ----
            if sb.get('emis'):
                et = tex.get(sb['emis'])
                ec = np.array((sb.get('emisCol') or [1.0, 1.0, 1.0])[:3], np.float32)
                if et is not None:
                    e = _sample(et[0], uv).astype(np.float32) / 255.0 * ec
                else:
                    e = np.tile(ec, (len(pos), 1))
                ge.append(np.clip(e / EMIS_MAX * 255.0, 0, 255).astype(np.uint8))
            else:
                ge.append(np.zeros((len(pos), 3), np.uint8))
            gp.append(pos); gf.append(inv.reshape(-1, 3).astype(np.int32) + gvo); gvo += len(pos)
        if not gp: continue
        gp = np.concatenate(gp); gf = np.concatenate(gf)
        ga = np.concatenate(ga); ge = np.concatenate(ge)
        # ---- place every instance: conjugation world = (G M G^-1) v_obj (identical to the viewer/PLY export) ----
        for it in insts:
            m = it['m']
            M4 = np.array([[m[0], m[1], m[2], m[3]], [m[4], m[5], m[6], m[7]],
                           [m[8], m[9], m[10], m[11]], [0, 0, 0, 1]], np.float64)
            Mp = G4 @ M4 @ G4i
            W = (gp.astype(np.float64) @ Mp[:3, :3].T + Mp[:3, 3]).astype(np.float32)
            Vl.append(W); Fl.append(gf + voff); Al.append(ga); El.append(ge); voff += len(gp)
        if gi_ % 2000 == 0 and time.time() - t_rep > 20:
            print(f"  [mesh] group {gi_}/{len(groups)}  verts so far {voff:,} ({time.time()-t0:.0f}s)"); t_rep = time.time()
    V = np.concatenate(Vl); F = np.concatenate(Fl); A = np.concatenate(Al); E = np.concatenate(El)
    del Vl, Fl, Al, El
    print(f"[mesh] soup: {len(V):,} verts {len(F):,} tris  (missing tex files: {ntex_miss}, uv-less verts: {nuv_miss:,})  {time.time()-t0:.0f}s")
    # ---- albedo spot-check: known-color textures must come out that color (UV mapping validation) ----
    print("[mesh] albedo spot-check (mean LINEAR rgb of sampled verts):")
    names = sorted(spot.keys(), key=lambda k: -spot[k][1])
    picks = [n for n in names if 'container' in n.lower()][:4] + names[:4]
    for n in dict.fromkeys(picks):
        s, c = spot[n]; print(f"    {n:56s} n={c:9,d}  rgb={np.round(s/max(c,1),3).tolist()}")
    return V, F, A, E


def get_bake_mesh(cfg):
    os.makedirs(CACHE, exist_ok=True)
    paths = {k: os.path.join(CACHE, k + '.npy') for k in ('V', 'F', 'ALB', 'EMI')}
    if not REBUILD and all(os.path.exists(p) for p in paths.values()):
        print(f"[mesh] loading cached bake mesh from {CACHE}")
        return (np.load(paths['V']), np.load(paths['F']), np.load(paths['ALB']), np.load(paths['EMI']))
    V, F, A, E = build_bake_mesh(cfg)
    for k, arr in (('V', V), ('F', F), ('ALB', A), ('EMI', E)): np.save(paths[k], arr)
    print(f"[mesh] cached -> {CACHE}")
    return V, F, A, E


# ================================================================================================
# PART 2 -- Warp bake: pass A direct (neutral sky + live practicals), pass B one diffuse bounce
# ================================================================================================
def load_lights(G3, DS):
    """FILTERED live lights (the extractor already applies m_Enabled + ancestor m_IsActive).
    Unity world -> glb world = G3 @ p; spot forward = G3 @ (R_quat @ (0,0,1)); spotAngle is the FULL cone."""
    cand = sorted(glob.glob(os.path.join(DS, 'lights_*.json')))
    # Skip .bak and the *_all.json superset (that variant includes off/disabled lights tagged
    # on:false, which are not for baking).
    cand = [c for c in cand if not c.endswith('.bak') and not c.endswith('_all.json')]
    if not cand:
        print("[lights] no lights_*.json for this dataset -> SKY-ONLY bake"); L = []
    else:
        # MERGE every per-scene light sidecar. Single-light maps have exactly one file (behaviour
        # unchanged); maps that split their lighting across many district scenes (streets/ground_zero)
        # need ALL of them for full lighting -- the previous code baked only the first file, leaving
        # most districts dark.
        L = []
        for c in cand:
            part = json.load(open(c))
            L.extend(part)
            print(f"[lights] {os.path.basename(c)} -> {len(part)} lights")
        if len(cand) > 1:
            print(f"[lights] merged {len(cand)} sidecars -> {len(L)} lights total")
    pos, ci, rng_, sf, cono, coni = [], [], [], [], [], []
    n_spot = n_dir_skipped = 0
    for l in L:
        if l.get('type') == 'Directional':
            n_dir_skipped += 1; continue   # authored sun/moon (older maps ship one, e.g. Factory_Day):
                                           # not a positional practical -- the bake supplies its own sky+sun
        inten = float(l.get('intensity', 0) or 0)
        if inten <= 0: continue
        p = G3 @ np.array(l['position'], np.float64)
        col = np.array((l.get('color') or [1, 1, 1, 1])[:3], np.float64)
        r = float(l.get('range', 6.0) or 6.0)
        pos.append(p); ci.append(col * inten); rng_.append(max(r, 4.0))
        if l.get('type') == 'Spot':
            x, y, z, w = np.array(l['rotation'], np.float64)
            fwd = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], np.float64)
            fwd = G3 @ fwd; fwd /= (np.linalg.norm(fwd) + 1e-9)
            ho = math.radians(float(l.get('spotAngle', 90.0)) * 0.5)                      # FULL -> half
            hi = math.radians(float(l.get('innerSpotAngle') or l.get('spotAngle', 90.0) * 0.8) * 0.5)
            sf.append(fwd); cono.append(math.cos(ho)); coni.append(math.cos(min(hi, ho * 0.999))); n_spot += 1
        else:
            sf.append([0.0, 0.0, 0.0]); cono.append(-1.0); coni.append(-1.0)
    n = len(pos)
    print(f"[lights] {n} LIVE lights ({n_spot} spot, {n-n_spot} point"
          f"{f', {n_dir_skipped} Directional skipped' if n_dir_skipped else ''}); "
          f"inverse-square + range window, X-flipped to glb")
    if n == 0:
        z = np.zeros
        return (z((1, 3), np.float32), z((1, 3), np.float32), np.ones(1, np.float32),
                z((1, 3), np.float32), -np.ones(1, np.float32), -np.ones(1, np.float32), 0)
    return (np.asarray(pos, np.float32), np.asarray(ci, np.float32), np.asarray(rng_, np.float32),
            np.asarray(sf, np.float32), np.asarray(cono, np.float32), np.asarray(coni, np.float32), n)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], 1).astype(np.float32)


def flood_fill(sh, valid, dims):
    """Vectorized 6-neighbour flood fill of invalid probes (v1 semantics, numpy-shift implementation)."""
    nx, ny, nz = dims
    sh = sh.reshape(nz, ny, nx, 12).copy(); valid = valid.reshape(nz, ny, nx).copy()
    n_filled = 0
    for _ in range(256):
        inv_ = ~valid
        if not inv_.any(): break
        acc = np.zeros(sh.shape, np.float32); cnt = np.zeros(valid.shape, np.float32)
        for ax, d in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            vs = np.roll(valid, d, axis=ax); ss = np.roll(sh, d, axis=ax)
            edge = [slice(None)] * 3; edge[ax] = 0 if d == 1 else -1
            vs[tuple(edge)] = False
            acc += ss * vs[..., None]; cnt += vs
        fill = inv_ & (cnt > 0)
        if not fill.any():
            amb = np.zeros(12, np.float32); amb[0:3] = [0.04, 0.05, 0.06]
            sh[inv_] = amb; n_filled += int(inv_.sum()); break
        sh[fill] = acc[fill] / cnt[fill][:, None]
        valid |= fill; n_filled += int(fill.sum())
    return sh.reshape(-1, 12), n_filled


def main():
    cfg = MapConfig.load(MAP)
    DS = cfg.dataset
    if not os.path.isabs(DS): DS = os.path.join(ROOT, DS)
    V, F, ALB, EMI = get_bake_mesh(cfg)
    gmin_full = V.min(0); gmax_full = V.max(0)
    print(f"[geo] bbox min={gmin_full.round(1).tolist()} max={gmax_full.round(1).tolist()}  tris={len(F):,}")

    import warp as wp
    wp.init()

    # ---- neutral sky (grayscale, luma-matched to v1's tinted gradient) + warm fallback sun ----
    SKY_ZENITH = 0.436 * SKY_SCALE
    SKY_HORIZON = 0.743 * SKY_SCALE
    sun = np.array([0.45, 0.80, -0.40], np.float64); sun /= np.linalg.norm(sun)
    print(f"[sun] no Directional light exists in EFT scenes -> FALLBACK warm sun dir={sun.round(3).tolist()} (matches v1)")

    G3 = np.array(cfg.coord_matrix(), np.float64).reshape(4, 4)[:3, :3]
    lpos, lci, lrng, lsf, lcono, lconi, n_light = load_lights(G3, DS)

    # ---- probe grid: 1/99-percentile XZ core + data-driven Y band; ~XZ_TARGET spacing, auto-widened ----
    rng = np.random.default_rng(0)
    samp = V[rng.choice(len(V), size=min(3_000_000, len(V)), replace=False)]
    lo = np.percentile(samp, 1, axis=0); hi = np.percentile(samp, 99, axis=0)
    pad = 8.0
    ylo = float(np.percentile(samp[:, 1], 0.5)) - 2.0
    yhi = float(np.percentile(samp[:, 1], 99.7)) + 4.0
    gmin = np.array([lo[0] - pad, ylo, lo[2] - pad], np.float64)
    gmax = np.array([hi[0] + pad, yhi, hi[2] + pad], np.float64)
    sxz = max(XZ_TARGET, float(((gmax[0] - gmin[0]) * (gmax[2] - gmin[2]) / 120000.0)) ** 0.5)
    for _ in range(64):
        dims = np.maximum(np.ceil((gmax - gmin) / np.array([sxz, Y_SPACING, sxz])).astype(int) + 1, 2)
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        if ny * nz <= 8192 and nx <= 4096 and nx * ny * nz <= 2_600_000: break   # WebGL tex height cap + memory
        sxz *= 1.05
    n_probe = nx * ny * nz
    spacing = (gmax - gmin) / np.maximum(dims - 1, 1)
    print(f"[grid] min={gmin.round(1).tolist()} max={gmax.round(1).tolist()}")
    print(f"[grid] dims={[nx,ny,nz]} spacing={spacing.round(2).tolist()} -> {n_probe:,} probes (v1: 95x8x72=54,720 @13m)")

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    origins = np.stack([gmin[0] + xx.reshape(-1) * spacing[0],
                        gmin[1] + yy.reshape(-1) * spacing[1],
                        gmin[2] + zz.reshape(-1) * spacing[2]], 1).astype(np.float32)
    dirs = fibonacci_sphere(N_DIR)

    # ---- Warp kernels ----
    SUN_COS = float(math.cos(math.radians(4.0))); SUN_SOFT = float(math.cos(math.radians(9.0)))

    @wp.func
    def sky_radiance(d: wp.vec3, sun_d: wp.vec3, zen: float, hor: float, sun_s: float):
        t = wp.clamp(d[1] * 0.5 + 0.5, 0.0, 1.0)
        grad = t * t
        v = hor * (1.0 - grad) + zen * grad
        col = wp.vec3(v, v, v)                                  # NEUTRAL gray sky (no tint)
        if d[1] < 0.0:
            col = col * (0.30 + 0.70 * (d[1] + 1.0))
        cd = wp.dot(d, sun_d)
        if cd > SUN_SOFT:
            s = wp.clamp((cd - SUN_SOFT) / (SUN_COS - SUN_SOFT), 0.0, 1.0)
            col = col + wp.vec3(8.0 * sun_s, 6.5 * sun_s, 4.5 * sun_s) * (s * s)
        return col

    @wp.kernel
    def bake_direct(mesh: wp.uint64,
                    origins_: wp.array(dtype=wp.vec3), dirs_: wp.array(dtype=wp.vec3),
                    sun_d: wp.vec3, n_dir: int, max_dist: float, zen: float, hor: float, sun_s: float,
                    lp: wp.array(dtype=wp.vec3), lc: wp.array(dtype=wp.vec3), lr: wp.array(dtype=wp.float32),
                    lsf: wp.array(dtype=wp.vec3), lcono: wp.array(dtype=wp.float32), lconi: wp.array(dtype=wp.float32),
                    n_light: int, light_scale: float, vis_dmax: float,
                    sh_out: wp.array2d(dtype=wp.vec3), inside_out: wp.array(dtype=wp.int32),
                    vis_r: wp.array2d(dtype=wp.float32), vis_r2: wp.array2d(dtype=wp.float32),
                    vis_c: wp.array2d(dtype=wp.float32)):
        pid = wp.tid()
        o = origins_[pid]
        Y0 = 0.282095
        Y1 = 0.488603
        c0 = wp.vec3(0.0, 0.0, 0.0); c1 = wp.vec3(0.0, 0.0, 0.0)
        c2 = wp.vec3(0.0, 0.0, 0.0); c3 = wp.vec3(0.0, 0.0, 0.0)
        dw = 4.0 * 3.14159265358979 / float(n_dir)
        n_short = int(0)
        for i in range(n_dir):
            d = dirs_[i]
            q = wp.mesh_query_ray(mesh, o, d, max_dist)
            rad = wp.vec3(0.0, 0.0, 0.0)
            tc = vis_dmax                                       # DDGI-style visibility moment: distance to the nearest
            if q.result:                                        # occluder, clamped to dmax (a miss counts as dmax = open)
                tc = wp.min(q.t, vis_dmax)
                if q.t < 0.75:
                    n_short += 1
            else:
                rad = sky_radiance(d, sun_d, zen, hor, sun_s)
            oct_ = int(0)                                       # octant sign bits of the ray dir (probe-local)
            if d[0] > 0.0:
                oct_ += 1
            if d[1] > 0.0:
                oct_ += 2
            if d[2] > 0.0:
                oct_ += 4
            vis_r[pid, oct_] = vis_r[pid, oct_] + tc
            vis_r2[pid, oct_] = vis_r2[pid, oct_] + tc * tc
            vis_c[pid, oct_] = vis_c[pid, oct_] + 1.0
            c0 = c0 + rad * (Y0 * dw)
            c1 = c1 + rad * (Y1 * d[1] * dw)
            c2 = c2 + rad * (Y1 * d[2] * dw)
            c3 = c3 + rad * (Y1 * d[0] * dw)
        # ---- LIVE practicals: physical 1/d^2 with a smooth range window, shadow-tested; FULL-angle spot cone ----
        for j in range(n_light):
            tol = lp[j] - o
            dist = wp.length(tol)
            R = lr[j]
            if dist > 0.05 and dist < R:
                dl = tol / dist
                spot = float(1.0)
                co = lcono[j]
                if co > -0.999:
                    cosang = -wp.dot(dl, lsf[j])
                    spot = wp.clamp((cosang - co) / (lconi[j] - co + 0.0001), 0.0, 1.0)
                if spot > 0.0:
                    x = dist / R
                    win = wp.clamp(1.0 - x * x * x * x, 0.0, 1.0)
                    at = win * win / wp.max(dist * dist, MIN_D2)
                    qs = wp.mesh_query_ray(mesh, o, dl, dist - 0.1)
                    if not qs.result:
                        rad = lc[j] * (at * spot * light_scale)
                        c0 = c0 + rad * Y0
                        c1 = c1 + rad * (Y1 * dl[1])
                        c2 = c2 + rad * (Y1 * dl[2])
                        c3 = c3 + rad * (Y1 * dl[0])
        sh_out[pid, 0] = c0; sh_out[pid, 1] = c1; sh_out[pid, 2] = c2; sh_out[pid, 3] = c3
        if n_short > (n_dir * 7) / 10:
            inside_out[pid] = 1
        else:
            inside_out[pid] = 0

    @wp.func
    def irr_at(sh: wp.array2d(dtype=wp.vec3), gmn: wp.vec3, inv_sp: wp.vec3,
               nx_: int, ny_: int, nz_: int, p: wp.vec3, n: wp.vec3):
        """Trilinear irradiance E(p, n) from the flood-filled direct grid (radiance SH -> cosine conv)."""
        gx = wp.clamp((p[0] - gmn[0]) * inv_sp[0], 0.0, float(nx_ - 1))
        gy = wp.clamp((p[1] - gmn[1]) * inv_sp[1], 0.0, float(ny_ - 1))
        gz = wp.clamp((p[2] - gmn[2]) * inv_sp[2], 0.0, float(nz_ - 1))
        x0 = int(wp.floor(gx)); y0 = int(wp.floor(gy)); z0 = int(wp.floor(gz))
        x1 = wp.min(x0 + 1, nx_ - 1); y1 = wp.min(y0 + 1, ny_ - 1); z1 = wp.min(z0 + 1, nz_ - 1)
        fx = gx - float(x0); fy = gy - float(y0); fz = gz - float(z0)
        a0 = wp.vec3(0.0, 0.0, 0.0); a1 = wp.vec3(0.0, 0.0, 0.0)
        a2 = wp.vec3(0.0, 0.0, 0.0); a3 = wp.vec3(0.0, 0.0, 0.0)
        for k in range(8):
            xi = x0; yi = y0; zi = z0
            wx = 1.0 - fx; wy = 1.0 - fy; wz = 1.0 - fz
            if k & 1 != 0:
                xi = x1; wx = fx
            if k & 2 != 0:
                yi = y1; wy = fy
            if k & 4 != 0:
                zi = z1; wz = fz
            w = wx * wy * wz
            pi_ = (zi * ny_ + yi) * nx_ + xi
            a0 = a0 + sh[pi_, 0] * w; a1 = a1 + sh[pi_, 1] * w
            a2 = a2 + sh[pi_, 2] * w; a3 = a3 + sh[pi_, 3] * w
        # E(n) = pi*Y00*c0 + (2pi/3)*Y1*(c1*ny + c2*nz + c3*nx)
        E = a0 * 0.8862269 + (a1 * n[1] + a2 * n[2] + a3 * n[0]) * 1.0233267
        return wp.vec3(wp.max(E[0], 0.0), wp.max(E[1], 0.0), wp.max(E[2], 0.0))

    @wp.kernel
    def bake_bounce(mesh: wp.uint64,
                    origins_: wp.array(dtype=wp.vec3), dirs_: wp.array(dtype=wp.vec3),
                    n_dir: int, max_dist: float,
                    tri: wp.array(dtype=wp.int32), alb: wp.array(dtype=wp.uint8), emi: wp.array(dtype=wp.uint8),
                    sh_src: wp.array2d(dtype=wp.vec3), gmn: wp.vec3, inv_sp: wp.vec3,
                    nx_: int, ny_: int, nz_: int, emis_scale: float, alb_boost: float,
                    sh_out: wp.array2d(dtype=wp.vec3)):
        pid = wp.tid()
        o = origins_[pid]
        Y0 = 0.282095
        Y1 = 0.488603
        c0 = wp.vec3(0.0, 0.0, 0.0); c1 = wp.vec3(0.0, 0.0, 0.0)
        c2 = wp.vec3(0.0, 0.0, 0.0); c3 = wp.vec3(0.0, 0.0, 0.0)
        dw = 4.0 * 3.14159265358979 / float(n_dir)
        for i in range(n_dir):
            d = dirs_[i]
            q = wp.mesh_query_ray(mesh, o, d, max_dist)
            if q.result:
                f = q.face
                i0 = tri[f * 3 + 0]; i1 = tri[f * 3 + 1]; i2 = tri[f * 3 + 2]
                w0 = q.u; w1 = q.v; w2 = 1.0 - q.u - q.v
                ca = (wp.vec3(float(alb[i0 * 3 + 0]), float(alb[i0 * 3 + 1]), float(alb[i0 * 3 + 2])) * w0 +
                      wp.vec3(float(alb[i1 * 3 + 0]), float(alb[i1 * 3 + 1]), float(alb[i1 * 3 + 2])) * w1 +
                      wp.vec3(float(alb[i2 * 3 + 0]), float(alb[i2 * 3 + 1]), float(alb[i2 * 3 + 2])) * w2) * (1.0 / 255.0)
                ce = (wp.vec3(float(emi[i0 * 3 + 0]), float(emi[i0 * 3 + 1]), float(emi[i0 * 3 + 2])) * w0 +
                      wp.vec3(float(emi[i1 * 3 + 0]), float(emi[i1 * 3 + 1]), float(emi[i1 * 3 + 2])) * w1 +
                      wp.vec3(float(emi[i2 * 3 + 0]), float(emi[i2 * 3 + 1]), float(emi[i2 * 3 + 2])) * w2) * emis_scale
                n = q.normal
                if wp.dot(n, d) > 0.0:
                    n = -n
                h = o + d * q.t + n * 0.05
                Eh = irr_at(sh_src, gmn, inv_sp, nx_, ny_, nz_, h, n)
                rad = wp.cw_mul(ca, Eh) * (0.3183099 * alb_boost) + ce   # albedo/pi * E + emissive
                c0 = c0 + rad * (Y0 * dw)
                c1 = c1 + rad * (Y1 * d[1] * dw)
                c2 = c2 + rad * (Y1 * d[2] * dw)
                c3 = c3 + rad * (Y1 * d[0] * dw)
        sh_out[pid, 0] = c0; sh_out[pid, 1] = c1; sh_out[pid, 2] = c2; sh_out[pid, 3] = c3

    # ---- BVH ----
    t0 = time.time()
    wp_v = wp.array(V, dtype=wp.vec3)
    wp_i = wp.array(F.reshape(-1), dtype=wp.int32)
    mesh = wp.Mesh(points=wp_v, indices=wp_i)
    print(f"[bvh] wp.Mesh over {len(F):,} tris in {time.time()-t0:.1f}s")

    wp_o = wp.array(origins, dtype=wp.vec3)
    wp_d = wp.array(dirs, dtype=wp.vec3)
    wp_lp = wp.array(lpos, dtype=wp.vec3); wp_lc = wp.array(lci, dtype=wp.vec3)
    wp_lr = wp.array(lrng, dtype=wp.float32); wp_lsf = wp.array(lsf, dtype=wp.vec3)
    wp_lco = wp.array(lcono, dtype=wp.float32); wp_lci = wp.array(lconi, dtype=wp.float32)
    max_dist = float(np.linalg.norm(gmax_full - gmin_full) * 1.2)

    # ---- PASS A: direct (+ DDGI visibility moments per octant) ----
    VIS_DMAX = float(2.0 * max(spacing))                     # moment clamp: leaks matter within ~a cell
    t0 = time.time()
    sh_A = wp.zeros((n_probe, 4), dtype=wp.vec3)
    inside_out = wp.zeros(n_probe, dtype=wp.int32)
    vis_r = wp.zeros((n_probe, 8), dtype=wp.float32)
    vis_r2 = wp.zeros((n_probe, 8), dtype=wp.float32)
    vis_c = wp.zeros((n_probe, 8), dtype=wp.float32)
    wp.launch(bake_direct, dim=n_probe,
              inputs=[mesh.id, wp_o, wp_d, wp.vec3(*sun.tolist()), N_DIR, max_dist,
                      float(SKY_ZENITH), float(SKY_HORIZON), float(SUN_SCALE),
                      wp_lp, wp_lc, wp_lr, wp_lsf, wp_lco, wp_lci, n_light, float(LIGHT_SCALE),
                      VIS_DMAX, sh_A, inside_out, vis_r, vis_r2, vis_c])
    wp.synchronize()
    shA = sh_A.numpy().astype(np.float32)
    inside = inside_out.numpy().astype(bool)
    vr = vis_r.numpy(); vr2 = vis_r2.numpy(); vc = np.maximum(vis_c.numpy(), 1.0)
    vr /= vc; vr2 /= vc                                      # per-octant mean dist + mean sq dist to nearest occluder
    print(f"[passA] direct: {n_probe:,} probes x {N_DIR} rays + {n_light} lights in {time.time()-t0:.1f}s; inside-solid={int(inside.sum()):,}")
    print(f"[vis] octant moments: dmax={VIS_DMAX:.1f}m  mean r={vr.mean():.2f}m  frac fully-open octants={(vr > VIS_DMAX*0.98).mean()*100:.0f}%")

    # ---- flood-fill the DIRECT grid (bounce gather source must not read inside-wall zeros) ----
    src_filled, nf = flood_fill(shA.reshape(n_probe, 12), ~inside, (nx, ny, nz))
    print(f"[fill] pass-A source: {nf:,} probes filled")
    sh_src = wp.array(np.ascontiguousarray(src_filled.reshape(n_probe, 4, 3)), dtype=wp.vec3)

    # ---- PASS B: one diffuse bounce + emissive gather ----
    t0 = time.time()
    wp_alb = wp.array(ALB.reshape(-1), dtype=wp.uint8)
    wp_emi = wp.array(EMI.reshape(-1), dtype=wp.uint8)
    sh_B = wp.zeros((n_probe, 4), dtype=wp.vec3)
    inv_sp = wp.vec3(float(1.0 / spacing[0]), float(1.0 / spacing[1]), float(1.0 / spacing[2]))
    wp.launch(bake_bounce, dim=n_probe,
              inputs=[mesh.id, wp_o, wp_d, N_DIR, max_dist,
                      wp_i, wp_alb, wp_emi, sh_src, wp.vec3(*gmin.tolist()), inv_sp,
                      nx, ny, nz, float(EMIS_MAX / 255.0 * EMIS_GAIN), float(ALBEDO_BOOST),
                      sh_B])
    wp.synchronize()
    shB = sh_B.numpy().astype(np.float32)
    print(f"[passB] bounce+emissive in {time.time()-t0:.1f}s")

    sh = (shA + shB).reshape(n_probe, 12)
    dcA = 0.2126 * shA[:, 0, 0] + 0.7152 * shA[:, 0, 1] + 0.0722 * shA[:, 0, 2]
    dcB = 0.2126 * shB[:, 0, 0] + 0.7152 * shB[:, 0, 1] + 0.0722 * shB[:, 0, 2]
    ok_ = ~inside
    print(f"[passB] bounce DC adds {dcB[ok_].mean():.4f} mean luma on top of direct {dcA[ok_].mean():.4f} "
          f"(+{100*dcB[ok_].mean()/max(dcA[ok_].mean(),1e-6):.0f}%)")

    sh, n_filled = flood_fill(sh, ~inside, (nx, ny, nz))
    print(f"[fill] final: {n_filled:,} probes filled")
    sh = sh.reshape(n_probe, 4, 3)

    # ---- stats / validation ----
    def dc_lum(c): return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
    dcl = dc_lum(sh[:, 0]).reshape(nz, ny, nx)
    flat = dcl.reshape(-1)
    p5, p50, p95 = np.percentile(flat, [5, 50, 95])
    zfrac = float((flat < 1e-3).mean())
    bmask = np.zeros((nz, ny, nx), bool)
    bmask[0, :, :] = bmask[-1, :, :] = True; bmask[:, 0, :] = bmask[:, -1, :] = True; bmask[:, :, 0] = bmask[:, :, -1] = True
    bvals = dcl[bmask]
    print(f"[stats] DC luma p5={p5:.4f} p50={p50:.4f} p95={p95:.4f}  zero(<1e-3) frac={zfrac*100:.2f}%")
    print(f"[stats] boundary probes: mean={bvals.mean():.4f} median={np.median(bvals):.4f} zero-frac={(bvals<1e-3).mean()*100:.2f}% (v1 boundary read ~0)")

    def probe_at(wx, wy, wz):
        i = np.clip(np.round(([wx, wy, wz] - gmin) / spacing).astype(int), 0, [nx - 1, ny - 1, nz - 1])
        return (i[2] * ny + i[1]) * nx + i[0], tuple(i)
    out_idx, out_ijk = probe_at(gmin[0] + 0.25 * (gmax[0] - gmin[0]), gmax[1] - spacing[1], gmin[2] + 0.75 * (gmax[2] - gmin[2]))
    gy = 1
    ground = dcl[:, gy, :]
    k = 3; best = (1e9, 0, 0)
    st = max(1, nz // 128)
    for iz in range(k, nz - k, st):
        for ix in range(k, nx - k, st):
            v = float(ground[iz - k:iz + k + 1, ix - k:ix + k + 1].mean())
            if v < best[0]: best = (v, iz, ix)
    _, mz, mx_ = best
    in_idx = (mz * ny + gy) * nx + mx_
    out_dc = float(dc_lum(sh[out_idx, 0])); in_dc = float(dc_lum(sh[in_idx, 0]))
    print(f"[sanity] OUTDOOR probe {out_ijk} DC={out_dc:.4f}   most-occluded ground cell {(mx_,gy,mz)} DC={in_dc:.4f}")
    print(f"[sanity] top-layer mean={dcl[:, -1, :].mean():.4f} ground-layer mean={dcl[:, gy, :].mean():.4f} global={dcl.mean():.4f}")
    litfrac = 0.0
    if n_light:
        ijk = np.clip(np.round((lpos - gmin) / spacing).astype(int), [0, 0, 0], [nx - 1, ny - 1, nz - 1])
        pids = (ijk[:, 2] * ny + ijk[:, 1]) * nx + ijk[:, 0]
        nearlum = dc_lum(sh[pids, 0])
        litfrac = float((nearlum > 0.05).mean())
        print(f"[VERIFY] probes AT the {n_light} live lights: {litfrac*100:.0f}% lit (DC>0.05), mean DC={nearlum.mean():.3f}")

    # ---- write volume2 (format-compatible layout) ----
    layout = ("float16 LE, probe-major; probe index = ((z*ny)+y)*nx + x; within each probe 12 halfs "
              "ordered coeff0.r,coeff0.g,coeff0.b, coeff1.r,coeff1.g,coeff1.b, coeff2.r,coeff2.g,coeff2.b, "
              "coeff3.r,coeff3.g,coeff3.b. Coeffs are RADIANCE SH (L1 real basis): 0=Y00(0.282095), "
              "1=Y1-1(0.488603*y), 2=Y10(0.488603*z), 3=Y11(0.488603*x). Viewer reconstructs IRRADIANCE "
              "via cosine convolution A0=pi, A1=2pi/3.")
    meta = {
        "min": gmin.round(4).tolist(), "max": gmax.round(4).tolist(),
        "dims": [nx, ny, nz], "spacing": [round(float(s), 4) for s in spacing],
        "coeffs": 4, "channels": 3, "layout": layout,
        "sun_dir": [round(float(x), 5) for x in sun.tolist()],
        "bounces": 1,
        "notes": (f"RTX Phase 2 bake (bake_volume2.py). {N_DIR} Fibonacci sky rays/probe, NEUTRAL gray sky "
                  f"(no tint) + warm fallback sun; {n_light} LIVE practicals (m_Enabled+activeInHierarchy-filtered "
                  f"lights json; shadow-tested; 1/d^2 * smooth range window; FULL-angle spot cones; scale={LIGHT_SCALE}) "
                  f"+ ONE diffuse bounce gathered from hit surfaces with UV-MAPPED per-vertex albedo (sRGB->linear, "
                  f"tex*col, terrain splat slices included) and UV-coverage-weighted emissive maps (gain={EMIS_GAIN}). "
                  f"Bake mesh = scene.json culls+LOD0, glass/decal excluded from occluder. "
                  f"{int(inside.sum())} inside-solid probes flood-filled."),
        # DDGI-style leak gate: per-probe per-OCTANT visibility moments in <name>.vis.bin (sidecar; the 12-half
        # SH block above is unchanged so old consumers still parse). Samplers Chebyshev-weight each trilinear tap.
        "vis": {
            "octants": 8, "moments": 2, "encoding": "uint8", "dmax": round(VIS_DMAX, 3),
            "layout": ("uint8, probe-major (same probe order as the SH bin); 16 bytes/probe = 8 octants x "
                       "(mean_dist, mean_sq_dist), interleaved [o0.r, o0.r2, o1.r, o1.r2, ...]. octant index = "
                       "(dx>0) + 2*(dy>0) + 4*(dz>0) of the PROBE->POINT direction. r stored as r/dmax*255, "
                       "r2 as r2/dmax^2*255; ray distances clamped to dmax at bake (miss = dmax)."),
        },
    }
    binarr = sh.reshape(n_probe, 12).astype(np.float16)
    binarr.tofile(os.path.join(OUTDIR, 'volume2.bin'))
    visb = np.empty((n_probe, 16), np.uint8)
    visb[:, 0::2] = np.clip(vr / VIS_DMAX * 255.0 + 0.5, 0, 255).astype(np.uint8)
    visb[:, 1::2] = np.clip(vr2 / (VIS_DMAX * VIS_DMAX) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    visb.tofile(os.path.join(OUTDIR, 'volume2.vis.bin'))
    json.dump(meta, open(os.path.join(OUTDIR, 'volume2.json'), 'w'), indent=2)
    print(f"[out] volume2.bin = {os.path.getsize(os.path.join(OUTDIR,'volume2.bin'))/1e6:.2f} MB + "
          f"volume2.vis.bin = {os.path.getsize(os.path.join(OUTDIR,'volume2.vis.bin'))/1e6:.2f} MB, volume2.json written")

    # ---- debug slices ----
    try:
        from PIL import Image
        for tag, yy_ in (('midY', ny // 2), ('ground', gy)):
            sl = dcl[:, yy_, :]
            t = np.clip(sl / max(np.percentile(sl, 99), 1e-6), 0, 1)
            r = np.clip(1.5 * t - 0.5, 0, 1); g = np.clip(1.5 * t, 0, 1)
            b = np.clip(1.0 - 1.5 * (t - 0.33), 0, 1) * 0.5 + 0.3 * (1 - t)
            rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
            Image.fromarray(rgb, 'RGB').resize((nx * 2, nz * 2), Image.NEAREST).save(
                os.path.join(OUTDIR, f'volume2_debug_{tag}.png'))
        print(f"[debug] slices -> {OUTDIR}/volume2_debug_*.png")
    except Exception as e:
        print("[debug] PNG failed:", e)

    ok = litfrac > 0.6 and zfrac < 0.5
    print(f"[VERIFY] {'PASS' if ok else 'FAIL'}  (litfrac={litfrac:.2f}, zerofrac={zfrac:.3f})")
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 2)
