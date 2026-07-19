"""EFT scene -> Blender dataset, v2. Fixes the wrong-texture & missing-mesh root causes found by
the codex/agent review:
  * TEXTURES/MATERIALS keyed by (file_id, path_id)  -> kills the ~7% path_id collisions that put a
    'dumpster' texture on a wall (textures resolve into many external sharedassets files).
  * LOD selection via LODGroup.m_LODs[0].renderers PPtrs (NOT the '_LOD0' name substring) -> stops
    dropping the 16 generically-named LOD0 objects and stops stacking lowercase 'model_lod' shells.
  * Unity TERRAIN (level63) heightmap -> mesh + a numpy-baked MicroSplat albedo (the missing ground).
  * per-SUBMESH materials kept (verified correct: EFT meshes are all Triangle-topology).
  * SkinnedMeshRenderer handled (mesh via smr.m_Mesh) defensively.
  * Meshes are exported as UnityPy OBJ (which X-flips + reverses winding); the builder UN-flips so the
    final scene is NOT mirrored.

  python extraction/unity/eft_extract_v2.py --levels 52,54,55,56,57,58,59,60,62,63,65 --name interchange_v2
  python extraction/unity/eft_extract_v2.py --levels 54 --name ix_test          # quick core test
  python extraction/unity/eft_extract_v2.py --levels 63 --name ix_terrain --terrain-only
"""
import os, sys, json, argparse, time, gc
import shutil, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eft_scene_extract import quat_to_mat, trs

# portable kit: paths come from the environment (see extraction/README.md)
#   EFT_GAME_DATA   = the game's EscapeFromTarkov_Data dir (default: standard install path)
#   EFT_ASSETS_ROOT = where extracted datasets are written (default: <EFT_TARKMAP_ROOT>/../eft_assets, else ./eft_assets)
EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else
    os.path.join(os.getcwd(), "eft_assets"))

ALB = ("_MainTex", "_Diffuse", "_BaseMap", "_AlbedoMap", "_MainTex0", "_BaseAlbedoASmoothness",
       "_TopAlbedoASmoothness", "_Albedo", "_Aldebo", "_Tex", "_BaseColorMap", "_MainTexture")   # _Aldebo: MK4/Rock's
#      albedo slot is literally misspelled in the game shader -> Woods rocks were untextured white without it (map-agnostic add)
NRM = ("_BumpMap", "_NormalMap", "_Normalmap", "_Normal", "_BaseNormalMap", "_BumpMap0", "_TopNormalMap")   # _Normalmap: MK4/Rock

# ---------------------------------------------------------------- perf: persistent tex cache + parallel PNG write
# Textures are CONTENT-ADDRESSED: the same source texture always decodes to the same PNG (this is the very
# property the parallel-extract merge already relies on). So:
#  (1) PERSISTENT CACHE keyed on the source BYTES (+ format/dims/normal-flag/PIL version) lets any rebuild -- or a
#      second map that shares a texture -- SKIP the ~95ms decode+PNG-encode by hardlinking the already-made PNG.
#  (2) The PNG encode (zlib; releases the GIL; ~90% of texture cost) is handed to a small thread pool so the
#      biggest level's ~700 textures aren't written one-at-a-time.
# Both are correctness-preserving: a cache MISS or ANY error falls straight back to the exact current decode+save,
# and the pool only ever runs PIL img.save() on a private img.copy() -- every UnityPy/texture2ddecoder call stays
# on the main thread, so output PNGs are byte-identical (verified with a full md5 diff of the produced set).
try:
    import PIL as _PILmod
    _PIL_VER = getattr(_PILmod, "__version__", "?")
except Exception:
    _PIL_VER = "?"

_TEXCACHE_ENABLED = os.environ.get("EFT_TEXCACHE", "1") != "0"          # EFT_TEXCACHE=0 -> disable persistent cache
_TEXCACHE_DIR = os.environ.get("EFT_TEXCACHE_DIR") or os.path.join(OUTROOT, ".texcache")
_texstats = {"hit": 0, "miss": 0}


def _tex_workers():
    """EFT_TEX_WORKERS: PNG-write threads. A digit sets it exactly (1 = serial, the exact legacy path);
    anything else = auto (a modest fraction of the cores so N concurrent chunk-processes don't thrash)."""
    v = os.environ.get("EFT_TEX_WORKERS", "").strip()
    if v.isdigit():
        return max(1, int(v))
    return max(1, min(8, (os.cpu_count() or 4) // 3))


def _texcache_key_path(to, is_normal):
    """Content-addressed cache path for a Texture2D read `to`: blake2b of the RESOLVED raw source bytes
    (get_image_data() reads the .resS stream for streamed textures -- most EFT textures stream, so the plain
    inline `image_data` is empty for them), plus width/height/format + the normal-swizzle flag + the PIL
    version (so a PIL upgrade that changes the encoded PNG bytes auto-invalidates rather than serving stale
    bytes). ~4ms per texture; immune to game updates (keyed on the actual bytes, not a path-id). Returns None
    if anything is off -> caller just decodes normally (correctness-preserving fallback)."""
    try:
        try: raw = to.get_image_data()          # resolves m_StreamData (.resS) -> real bytes
        except Exception: raw = None
        if not raw:
            raw = getattr(to, "image_data", None)  # inline fallback
        if not raw:
            if os.environ.get("EFT_TEXCACHE_DEBUG"): print(f"  [texcache-debug] raw empty for {getattr(to,'m_Name','?')}", flush=True)
            return None
        w = int(g(to, "m_Width", default=0) or 0); h = int(g(to, "m_Height", default=0) or 0)
        fmt = g(to, "m_TextureFormat", default=0)
        try: fmt = int(fmt)
        except (TypeError, ValueError): fmt = int(getattr(fmt, "value", -1))
        d = hashlib.blake2b(bytes(raw), digest_size=16)
        d.update(f"|{w}x{h}|{fmt}|{'N' if is_normal else 'A'}|pil{_PIL_VER}".encode())
        return os.path.join(_TEXCACHE_DIR, d.hexdigest() + ".png")
    except Exception as e:
        if os.environ.get("EFT_TEXCACHE_DEBUG"): print(f"  [texcache-debug] EXC {type(e).__name__}: {e}", flush=True)
        return None


def _link_or_copy(src, dst):
    """Materialize dst from a fully-written PNG src as cheaply as possible: hardlink (zero extra bytes,
    instant, same volume) else fall back to a byte copy. src is always complete, so dst is never partial."""
    try:
        os.link(src, dst)
    except FileExistsError:
        pass
    except OSError:
        try: shutil.copyfile(src, dst)
        except Exception: pass


def _publish_cache(out_fp, cache_fp):
    """After writing out_fp, hardlink it into the cache (first writer wins; identical dup is a no-op)."""
    if not cache_fp or os.path.exists(cache_fp):
        return
    _link_or_copy(out_fp, cache_fp)


class _TexPool:
    """Background PNG writer. Decode + normal-unswizzle stay on the CALLING (main) thread; only the pure
    PIL img.save() (+ cache publish) runs on workers, and only on a private img.copy() so no UnityPy/
    texture2ddecoder state is ever touched off the main thread. workers<=1 -> the exact serial legacy path."""
    def __init__(self, workers):
        self.workers = max(1, workers)
        self.pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        self.sem = threading.Semaphore(self.workers + 2) if self.pool else None   # bound in-flight images (memory)
        self.errors = 0

    def save(self, img, out_fp, cache_fp):
        if self.pool is None:
            img.save(out_fp); _publish_cache(out_fp, cache_fp); return
        img = img.copy()                                    # detach from UnityPy's lazy/reused decode buffer
        self.sem.acquire()
        def _job():
            try:
                img.save(out_fp); _publish_cache(out_fp, cache_fp)
            except Exception:
                self.errors += 1
            finally:
                self.sem.release()
        self.pool.submit(_job)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


def g(o, *names, default=None):
    for n in names:
        if hasattr(o, n):
            return getattr(o, n)
    return default


def san(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def unswizzle_normal(img):
    """Unity DXT5nm packs normal.X in ALPHA, Y in GREEN, R~const. Rebuild standard RGB normal."""
    try:
        from PIL import Image
        a = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        if a[..., 0].mean() > 0.95 and a[..., 0].std() < 0.06:      # red ~constant => DXT5nm
            X = a[..., 3] * 2 - 1; Y = a[..., 1] * 2 - 1
            Z = np.sqrt(np.clip(1 - X * X - Y * Y, 0, 1))
            out = np.stack([X * .5 + .5, Y * .5 + .5, Z * .5 + .5], -1)
            return Image.fromarray((out * 255).astype(np.uint8), "RGB")
    except Exception:
        pass
    return img


# ----------------------------------------------------------------------------- terrain bake
def _img_np(texpp):
    try:
        t = texpp.read(); im = t.image
        if im is None: return None
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None


_MS_UV_CACHE = {}
def _ms_items(x):
    try: return list(x.items())
    except Exception:
        out = []
        for e in (x or []):
            try: out.append((e[0], e[1]))
            except Exception: out.append((getattr(e, 'first', e), getattr(e, 'second', None)))
        return out


def _terrain_season(layers):
    """Detect the EFT season token from the terrain layers' diffuse names (e.g. 'Grass_summer_D' -> 'summer').
    L16: vote by per-layer token frequency instead of a fixed priority order, so a mixed-season terrain picks the
    DOMINANT season rather than whichever token happens to sort first; flag an uncertain/degraded pick (close vote
    or token present in <half the layers). Compound tokens (spring_early/autumn_late) are matched before their bare
    base so they aren't double-counted. Map-agnostic: pure token statistics, no per-map constants."""
    # Compound seasons first so 'spring_early' isn't also counted as 'spring'.
    SEASONS = ("spring_early", "autumn_late", "summer", "winter", "spring", "autumn")
    votes = {}
    nlayers = 0
    for lpp in layers:
        try:
            tl = lpp.read(); dpp = g(tl, "m_DiffuseTexture", "m_Texture", "m_Diffuse")
            nm = str(getattr(dpp.read(), 'm_Name', '')).lower()
        except Exception:
            continue
        nlayers += 1
        for s in SEASONS:                       # first (most specific) matching token wins for THIS layer
            if s in nm:
                votes[s] = votes.get(s, 0) + 1
                break
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    best, n = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    # Degraded/uncertain: a near-tie with the runner-up, or the winner only covers a minority of layers.
    if (n - runner) <= 1 or (nlayers and n < nlayers / 2.0):
        print(f"  terrain season UNCERTAIN: picked '{best}' from votes={dict(ranked)} over {nlayers} layers")
    return best


def microsplat_uv_scales(season):
    """REAL per-texture terrain tiling FROM THE GAME's MicroSplat material (sharedassets17.assets). Returns
    (uvscale, [perTexScale...]) or (None,None). MicroSplat computes tiledUV = terrainUV01 * _UVScale * perTexScale[i],
    so a layer's world tile size = terrainSize / (uvscale * perTexScale[i]). _UVScale is one global colour-property
    (=233.33 for EFT); perTexScale is row-0 (R) of the _PerTexProps RGBAFloat texture, indexed by texture-array slot
    (== TerrainLayer order). General/data-driven — NO per-map fudge. NB: TerrainLayer.m_TileSize is GARBAGE for
    MicroSplat terrains (e.g. grass x=137.25, y=inf) — never use it. Materials are named MicroSplat_<Q>_<season>.

    L17: cache is keyed on the RESOLVED MicroSplat material identity (path_id), not on the season string, so two
    same-season terrains that resolve to different MicroSplat materials in one run don't reuse the first's scales.
    M6: only reinterpret _PerTexProps as raw float32 when its m_TextureFormat is actually RGBAFloat (==17); any
    other format would be garbage under np.frombuffer — fall back to TileSize. Parsed pertex values are validated
    (finite + sane positive range) before being accepted."""
    if season in _MS_UV_CACHE:
        return _MS_UV_CACHE[season]
    import UnityPy
    RGBAFloat = 20; RGBAHalf = 17                                        # Unity TextureFormat enum (RGBAHalf=17, RGBAFloat=20)
    res = (None, None)
    try:
        env = UnityPy.load(os.path.join(EFTDATA, "sharedassets17.assets"))
        for o in env.objects:
            if o.type.name != "Material":
                continue
            m = o.read(); nm = str(getattr(m, 'm_Name', ''))
            if not (nm.startswith("MicroSplat_") and nm.endswith(season)):
                continue
            mat_key = ("ms_uv", getattr(o, "path_id", None))            # L17: per-material cache identity
            if mat_key in _MS_UV_CACHE:
                res = _MS_UV_CACHE[mat_key]; break
            sp = g(m, "m_SavedProperties"); uvs = None; pertex = None
            for k, v in _ms_items(g(sp, "m_Colors")):
                if str(k) == "_UVScale":
                    uvs = float(v.r)
            for k, v in _ms_items(g(sp, "m_TexEnvs")):
                if 'PerTex' in str(k) and v.m_Texture and v.m_Texture.m_PathID:
                    t = v.m_Texture.read(); W = int(g(t, "m_Width")); H = int(g(t, "m_Height"))
                    # M6: reinterpret _PerTexProps with the dtype that MATCHES its format, else np.frombuffer is garbage.
                    # RGBAFloat(20)=float32 (16B/texel); RGBAHalf(17)=float16 (8B/texel). The constant was previously
                    # mislabeled (17 called "RGBAFloat"), so the real fmt=20 data was REJECTED and the tiling silently
                    # fell back to the garbage m_TileSize (grass ~137m instead of ~1.76m). EFT ships RGBAFloat=20.
                    fmt = g(t, "m_TextureFormat")
                    try: fmt = int(fmt)
                    except (TypeError, ValueError): fmt = int(getattr(fmt, "value", -1))
                    raw = bytes(t.image_data)
                    if fmt == RGBAFloat and len(raw) >= W * H * 16:
                        a = np.frombuffer(raw[:W * H * 16], dtype=np.float32).reshape(H, W, 4)
                    elif fmt == RGBAHalf and len(raw) >= W * H * 8:
                        a = np.frombuffer(raw[:W * H * 8], dtype=np.float16).reshape(H, W, 4).astype(np.float32)
                    else:
                        print(f"  microsplat _PerTexProps fmt={fmt} unsupported (bytes={len(raw)}) -> TileSize fallback")
                        continue
                    cand = a[0, :, 0].astype(float)                      # row 0, R channel = per-tex UV scale
                    # Validate: finite and in a sane positive tiling range; else reject (TileSize fallback).
                    if np.all(np.isfinite(cand)) and np.all(cand > 0) and float(cand.max()) < 1e6:
                        pertex = cand.tolist()
                    else:
                        print(f"  microsplat pertex out-of-range (max={float(np.nan_to_num(cand).max()):.3g}) -> TileSize fallback")
            if uvs and pertex:
                res = (uvs, pertex)
                _MS_UV_CACHE[mat_key] = res                              # L17: remember by material identity
                break
    except Exception as e:
        print(f"  microsplat uv-scale read failed ({e}) -> TileSize fallback")
    _MS_UV_CACHE[season] = res
    return res


def _bilinear(arr, fy, fx, wrap=False):
    """Bilinearly sample arr (HxW or HxWxC) at normalized fractional coords fy,fx (both in [0,1], same shape).
    wrap=True tiles at the edges (for the tiling diffuse, so the mod-1.0 seam interpolates instead of clamping);
    wrap=False clamps (for the single non-tiling control map). Returns arr's per-texel value with arr's channel
    layout. General/data-driven — no per-map assumptions."""
    h, w = arr.shape[:2]
    py = fy * (h - 1); px = fx * (w - 1)
    y0 = np.floor(py).astype(np.int32); x0 = np.floor(px).astype(np.int32)
    y1 = y0 + 1; x1 = x0 + 1
    wy = (py - y0); wx = (px - x0)
    if wrap:
        y0 %= h; y1 %= h; x0 %= w; x1 %= w
    else:
        y0 = np.clip(y0, 0, h - 1); y1 = np.clip(y1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1); x1 = np.clip(x1, 0, w - 1)
    if arr.ndim == 3:
        wy = wy[..., None]; wx = wx[..., None]
    c00 = arr[y0, x0]; c01 = arr[y0, x1]; c10 = arr[y1, x0]; c11 = arr[y1, x1]
    top = c00 * (1 - wx) + c01 * wx
    bot = c10 * (1 - wx) + c11 * wx
    return top * (1 - wy) + bot * wy


def bake_terrain_albedo(td, res_out=4096, ss=2):
    """Composite a MicroSplat terrain's layers by its splat-alpha control maps into ONE albedo image.
    Tiling comes FROM THE GAME (MicroSplat _UVScale x _PerTexProps), NOT the garbage TerrainLayer.m_TileSize that
    was tiling grass every 137m ("massive grass"). Supersampled (ss x ss) so the now-fine real tiling (~1.8m grass)
    doesn't alias in the fixed-res bake. (A live tiling splat is sharper at extreme zoom; this is the from-game bake.)"""
    sdb = g(td, "m_SplatDatabase")
    if sdb is None: return None
    layers = g(sdb, "m_TerrainLayers", "m_Splats", "m_SplatPrototypes") or []
    alphas = g(sdb, "m_AlphaTextures") or []
    if not layers or not alphas: return None
    hm = g(td, "m_Heightmap"); scale = g(hm, "m_Scale"); res = int(g(hm, "m_Resolution"))
    sizeX = (res - 1) * float(scale.x); sizeZ = (res - 1) * float(scale.z)
    season = _terrain_season(layers); uvscale, pertex = microsplat_uv_scales(season) if season else (None, None)
    if uvscale:
        print(f"  terrain tiling FROM GAME: season={season} _UVScale={uvscale:.1f}  (grass-class tile ~{sizeX/(uvscale*max(pertex[0],1e-3)):.2f}m)")
    else:
        print(f"  terrain tiling: no MicroSplat material for season={season} -> TileSize fallback")
    ctrl = [np.asarray(a.read().image.convert("RGBA"), dtype=np.float32) / 255.0 for a in alphas]
    R = res_out
    # per-layer: diffuse image + REAL tiling as separate U/V repetition counts across the terrain.
    # H5: tiling must be axis-correct. MicroSplat tiles in normalized terrain-UV01: tiledUV = uv01 * uvscale*pertex,
    # so the layer repeats uvscale*pertex times in BOTH u and v (square in UV-space, independent of metre aspect).
    # For the m_TileSize fallback the tile is a fixed metre size, so repetitions differ per axis when sizeX!=sizeZ:
    # repX=sizeX/tile, repZ=sizeZ/tile. Storing rep (not a single metre "tile_m") removes the sizeX-only-for-V bug.
    LD = []
    for li, lpp in enumerate(layers):
        tex_i = li // 4
        if tex_i >= len(ctrl): break
        try:
            tl = lpp.read(); dimg = _img_np(g(tl, "m_DiffuseTexture", "m_Texture", "m_Diffuse"))
            if uvscale and li < len(pertex) and pertex[li] > 0:
                repX = repZ = uvscale * pertex[li]                      # FROM-GAME tiling (UV-space, same on both axes)
            else:
                t = g(tl, "m_TileSize"); _tx = float(t.x) if t is not None else 0.0
                tile_m = _tx if (np.isfinite(_tx) and _tx > 0) else sizeX
                tile_m = max(tile_m, 1e-3)
                repX = sizeX / tile_m; repZ = sizeZ / tile_m            # axis-separate metre-tile fallback
        except Exception:
            dimg, repX, repZ = None, 1.0, 1.0
        LD.append((tex_i, li % 4, dimg, max(repX, 1e-6), max(repZ, 1e-6)))
    albedo = np.zeros((R, R, 3), np.float32); wsum = np.zeros((R, R), np.float32)
    # ss x ss jittered supersample -> clean mip0 despite the fine real tiling
    for a in range(ss):
        for b in range(ss):
            ii = (((np.arange(R) + (b + 0.5) / ss) / R)[:, None] * np.ones((1, R)))   # normalized v (0..1) down terrain
            jj = (np.ones((R, 1)) * ((np.arange(R) + (a + 0.5) / ss) / R)[None, :])   # normalized u (0..1) across terrain
            for tex_i, ch, dimg, repX, repZ in LD:
                # L19: bilinearly sample the control map (was nearest-neighbour -> blocky layer transitions).
                cm = ctrl[tex_i]; w = _bilinear(cm[..., ch], ii, jj, wrap=False)
                if w.max() <= 0.001: continue
                if dimg is None:
                    albedo += w[..., None] * np.array([0.4, 0.4, 0.4], np.float32); wsum += w; continue
                # H5: tile in normalized terrain-UV space; per-axis rep => correct V tiling at any aspect ratio.
                u = np.mod(jj * repX, 1.0); v = np.mod(ii * repZ, 1.0)
                albedo += w[..., None] * _bilinear(dimg, v, u, wrap=True); wsum += w
    # M7: a per-LAYER `w.max()<=0.001 continue` only skipped layers that are globally empty; texels where ALL
    # layers' weight ~0 (no control coverage) still divided ~0/~0 -> BLACK ground. Mask those uncovered texels
    # and fill with a neutral terrain colour (the covered-area mean, fallback 0.4 grey) instead of black.
    covered = wsum > 1e-3
    if covered.any():
        fill = albedo[covered].sum(0) / max(float(wsum[covered].sum()), 1e-6)   # dominant covered-area mean
    else:
        fill = np.array([0.4, 0.4, 0.4], np.float32)
    out = np.empty((R, R, 3), np.float32)
    out[covered] = albedo[covered] / wsum[covered][..., None]
    out[~covered] = fill
    out = np.clip(out, 0, 1)
    from PIL import Image
    return Image.fromarray((out * 255).astype(np.uint8), "RGB")


def export_terrain_splat(td, tname, splat_root, manifest, layer_saved, thresh=0.005):
    """Export a MicroSplat terrain's RAW layer diffuse textures + this tile's splat control maps + a manifest entry.
    The builder renders a TILING splatmap-blended material from these (sharp at any zoom), instead of one flat baked
    albedo whose resolution is capped by the bake. General: works for any Unity MicroSplat terrain.
    manifest/layer_saved are accumulators across all tiles (layer textures are shared, so dedup by name)."""
    sdb = g(td, "m_SplatDatabase")
    if sdb is None: return
    layers = g(sdb, "m_TerrainLayers", "m_Splats", "m_SplatPrototypes") or []
    alphas = g(sdb, "m_AlphaTextures") or []
    if not layers or not alphas: return
    hm = g(td, "m_Heightmap"); scale = g(hm, "m_Scale"); res = int(g(hm, "m_Resolution"))
    sizeX = (res - 1) * float(scale.x)
    # REAL from-game tiling (MicroSplat _UVScale x _PerTexProps), the SAME source bake_terrain_albedo uses. The manifest
    # previously only carried the GARBAGE m_TileSize (grass tiled every ~137m instead of ~1.8m) which made the runtime
    # splat material as blurry as the bake. We now write `rep` = repeats across the 0..1 terrain UV (tiledUV = uv01*rep),
    # so the viewer can tile each layer at game-correct density. Falls back to a m_TileSize-derived rep if no MicroSplat material.
    season = _terrain_season(layers); uvscale, pertex = microsplat_uv_scales(season) if season else (None, None)
    os.makedirs(splat_root, exist_ok=True)
    cmaps = []
    for ci, a in enumerate(alphas):
        fn = f"ctrl_{tname}_{ci}.png"
        a.read().image.convert("RGBA").save(os.path.join(splat_root, fn)); cmaps.append(fn)
    lcov = []
    for li, lpp in enumerate(layers):
        ti, ch = li // 4, li % 4
        if ti >= len(alphas): break
        arr = np.asarray(alphas[ti].read().image.convert("RGBA"), dtype=np.float32) / 255.0
        cov = float(arr[:, :, ch].mean())
        tl = lpp.read(); dn = g(tl, "m_DiffuseTexture", "m_Texture", "m_Diffuse")
        nm = str(g(dn.read(), "m_Name", "")) if dn is not None else f"L{li}"
        tile = g(tl, "m_TileSize"); tx = float(tile.x) if tile is not None else 0.0
        if uvscale and li < len(pertex) and pertex[li] > 0:
            rep = uvscale * pertex[li]                                  # FROM-GAME (UV-space repeats, the real ~1.8m grass tiling)
        else:
            rep = sizeX / tx if (np.isfinite(tx) and tx > 0) else 1.0   # m_TileSize fallback (only if no MicroSplat material)
        if nm not in layer_saved and dn is not None and cov >= thresh:
            try:
                dn.read().image.convert("RGB").save(os.path.join(splat_root, f"layer_{nm}.png")); layer_saved[nm] = 1
            except Exception:
                pass
        lcov.append({"idx": li, "name": nm, "ctrl": ti, "chan": ch, "cov": round(cov, 4),
                     "tileX": round(tx, 1), "rep": round(float(rep), 3)})   # rep = real per-layer tiling (uv01*rep); tileX kept for reference
    manifest["tiles"][tname] = {"ctrl_maps": cmaps, "sizeX": round(sizeX, 2),
                                "season": season, "uvscale": round(float(uvscale), 3) if uvscale else None, "layers": lcov}


def write_terrain_obj(td, path, step=4):
    """Heightmap -> decimated vertex-grid OBJ WITH uvs (uv=(col_frac,row_frac)). Returns (sizeX,sizeZ,sizeY).

    Unity world height = (raw / 65535) * 2 * m_Scale.y. The *2 is Unity's "16-bit field but only 15 bits used"
    heightmap quirk (a stored value of 32767 == full size.y, not 65535). Omitting it halves every terrain's
    absolute height, sinking it ~tens of metres below the rest of the map. This is the general Unity formula
    (applies to ANY Unity terrain/map), not a per-map fudge.
    """
    hm = g(td, "m_Heightmap"); res = int(g(hm, "m_Resolution")); scale = g(hm, "m_Scale")
    sx, sy, sz = float(scale.x), float(scale.y), float(scale.z)
    H = np.asarray(g(hm, "m_Heights"), dtype=np.float64).reshape(res, res)
    Hw = (H / 65535.0) * 2.0 * sy
    Hs = Hw[::step, ::step]; rr, cc = Hs.shape
    # Match UnityPy's mesh.export() convention so the builder's uniform FLIPX/coord pipeline handles terrain
    # IDENTICALLY to every other mesh: negate X and reverse triangle winding. (Writing raw +X here would make
    # FLIPX wrongly flip the terrain, offsetting it from the rest of the map.) General, not a per-map fudge.
    with open(path, "w", encoding="utf-8") as f:
        for r in range(rr):
            for c in range(cc):
                f.write(f"v {-c*step*sx:.4f} {Hs[r,c]:.4f} {r*step*sz:.4f}\n")
        for r in range(rr):
            for c in range(cc):
                f.write(f"vt {c/(cc-1):.5f} {r/(rr-1):.5f}\n")
        for r in range(rr - 1):
            for c in range(cc - 1):
                a = r*cc + c + 1; b = r*cc + (c+1) + 1; d = (r+1)*cc + c + 1; e = (r+1)*cc + (c+1) + 1
                f.write(f"f {b}/{b} {d}/{d} {a}/{a}\n"); f.write(f"f {e}/{e} {d}/{d} {b}/{b}\n")
    return (res - 1) * sx, (res - 1) * sz, float(Hw.max() - Hw.min())


# ----------------------------------------------------------------------------- main
def main():
    global EFTDATA                                                       # reassigned from --data-root below; must precede any use
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--alllod", action="store_true")
    ap.add_argument("--terrain-only", action="store_true")
    ap.add_argument("--terrain-step", type=int, default=2,
                    help="heightmap decimation: 1=native (1025^2/tile, ~0.7m/quad, heavy), 2=default (513^2, ~1.4m/quad), 4=coarse")
    ap.add_argument("--data-root", default=EFTDATA,
                    help="Unity <Game>_Data dir to read levels/sharedassets from; defaults to the EFT install "
                         "(pass the Arena *_Data dir to extract Arena maps). Back-compat: EFT invocations omit it.")
    args = ap.parse_args()
    EFTDATA = args.data_root                                             # every level{lv}/sharedassets read uses this module global
    import UnityPy
    levels = [int(x) for x in args.levels.split(",")]
    out = os.path.join(OUTROOT, args.name); md = os.path.join(out, "meshes"); td = os.path.join(out, "tex")
    os.makedirs(md, exist_ok=True); os.makedirs(td, exist_ok=True)

    exported = {}        # (lv,fid,pid) mesh -> obj filename
    tex_done = set()     # (fid,pid) textures already written
    mat_cache = {}       # (fid,pid) material -> (alb,nrm,sh,tile)
    instances = []
    lodgroups = []       # Stage A (LOD_PLAN.md): SSoT Unity LODGroup table; global index across levels. values verbatim from Unity
    terrain_manifest = {"tiles": {}, "layers": []}; _layer_saved = {}   # accumulate splat layers/maps across tiles
    splat_root = os.path.join(out, "terrain_layers")
    if _TEXCACHE_ENABLED:
        os.makedirs(_TEXCACHE_DIR, exist_ok=True)
    texpool = _TexPool(_tex_workers())                                  # background PNG writer (exp_tex closes over it)
    print(f"  tex: cache={'on -> ' + _TEXCACHE_DIR if _TEXCACHE_ENABLED else 'off'}  write-threads={texpool.workers}",
          flush=True)
    T0 = time.time()

    def fidpid(pptr):
        fid = g(pptr, "file_id", "m_FileID", default=0) or 0
        return (fid, pptr.path_id)

    for lv in levels:
        t0 = time.time()
        path = os.path.join(EFTDATA, f"level{lv}")
        if not os.path.exists(path):
            print(f"level{lv} missing"); continue
        env = UnityPy.load(path)
        tfm = {o.path_id: o for o in env.objects if o.type.name == "Transform"}
        wcache = {}

        _tfw = {}                     # tf_pid -> (father_tf_pid, local TRS 4x4)
        def _tf_local(pid):
            """Parse each Transform once (third leaf-only-memo quadratic walker, after
            active_in_hierarchy and root_of - Icebreaker Indoor_02's 79k renderers)."""
            v = _tfw.get(pid)
            if v is None:
                t = tfm[pid].read()
                fp = getattr(t, "m_Father", None)
                fpid = getattr(fp, "path_id", 0) if fp is not None else 0
                v = _tfw[pid] = (fpid if fpid in tfm else 0, trs(t))
            return v

        def world(tf_obj):
            pid = tf_obj.path_id
            W = wcache.get(pid)
            if W is not None: return W
            stack = []; cur = pid
            while cur and cur not in wcache and len(stack) < 256:
                stack.append(cur)
                cur = _tf_local(cur)[0]
            W = wcache.get(cur) if cur else None
            if W is None: W = np.eye(4)
            for p in reversed(stack):     # W(node) = W(father) @ trs(node), cached per node
                W = W @ _tf_local(p)[1]
                wcache[p] = W
            return W

        go2tf = {}
        for pid, o in tfm.items():
            try: go2tf[o.read().m_GameObject.path_id] = pid
            except Exception: pass

        def world_of_go(go_pid):
            tp = go2tf.get(go_pid)
            return world(tfm[tp]) if tp is not None else None

        rcache = {}
        _tfroot = {}                      # tf_pid -> (father_tf_pid, own GO name)
        def _tf_root_entry(tp):
            """Read each Transform/GameObject once (leaf-only memo made this quadratic on
            deep hierarchies - same disease as active_in_hierarchy, see _tf_entry)."""
            v = _tfroot.get(tp)
            if v is None:
                t = tfm[tp].read()
                nm = ""
                try: nm = t.m_GameObject.read().m_Name
                except Exception: pass
                fp = getattr(t, "m_Father", None)
                fpid = getattr(fp, "path_id", 0) if fp is not None else 0
                v = _tfroot[tp] = (fpid if fpid in tfm else 0, nm)
            return v

        def root_of(go_pid):
            """Top-level scene-root GameObject name (the game's own hierarchy grouping: 'SBG_*' = real content,
            'NewYear_Event' = event overlay, 'BLOCKER'/'JUSTPLANE' = proxies). Lets us cull by structure, not names."""
            if go_pid in rcache: return rcache[go_pid]
            tp = go2tf.get(go_pid); root = ""; gd = 0
            while tp and gd < 256:
                gd += 1
                fpid, nm = _tf_root_entry(tp)
                if nm: root = nm
                tp = fpid or None
            rcache[go_pid] = root; return root

        ahcache = {}
        _tfread = {}                      # tf_pid -> (father_tf_pid, own GO activeSelf)
        def _tf_entry(tp):
            """Read each Transform (and its GameObject's m_IsActive) exactly ONCE. The naive
            chain walk re-parsed every ancestor's typetree per leaf renderer - quadratic on
            deep hierarchies (Icebreaker Indoor_02: 79k renderers ground for an hour)."""
            v = _tfread.get(tp)
            if v is None:
                t = tfm[tp].read()
                act = True
                try:
                    act = bool(g(t.m_GameObject.read(), "m_IsActive", default=True))
                except Exception: pass
                fp = getattr(t, "m_Father", None)
                fpid = getattr(fp, "path_id", 0) if fp is not None else 0
                v = _tfread[tp] = (fpid if fpid in tfm else 0, act)
            return v

        def active_in_hierarchy(go_pid):
            """activeInHierarchy = activeSelf AND every ancestor's activeSelf (serialized m_IsActive is LOCAL only).
            Walk Transform.m_Father to a root (same chain as world()); any inactive ancestor hides the whole subtree."""
            if go_pid in ahcache: return ahcache[go_pid]
            tp = go2tf.get(go_pid); ok = True; gd = 0
            while tp and gd < 256:
                gd += 1
                fpid, act = _tf_entry(tp)
                if not act: ok = False; break
                tp = fpid or None
            ahcache[go_pid] = ok; return ok

        def srcid(pptr):
            """Resolve a PPtr to its STABLE GLOBAL identity (source sharedassets file + path_id).
            The PPtr's file_id is ambiguous: file_id=0 means 'same file as the referencing object', which differs
            per material -> different physical textures collapse to the same (lv,fid,pid) key and get skipped."""
            to = pptr.read()
            af = getattr(to, "assets_file", None)
            stem = san(os.path.splitext(os.path.basename(getattr(af, "name", "") or "x"))[0])
            orr = getattr(to, "object_reader", None)
            pid = getattr(orr, "path_id", None)
            if pid is None: pid = pptr.path_id
            return stem, int(pid), to

        def exp_tex(tx, is_normal=False):
            try:
                stem, pid, to = srcid(tx)
                key = (stem, pid)
                nm = san(g(to, "m_Name", default="t")) + f"__{stem}_{pid}"
                if key not in tex_done:
                    tex_done.add(key)
                    out_fp = os.path.join(td, nm + ".png")
                    if not os.path.exists(out_fp):          # skip if already on disk (this dataset, prior run)
                        cache_fp = _texcache_key_path(to, is_normal) if _TEXCACHE_ENABLED else None
                        if cache_fp and os.path.exists(cache_fp):
                            _link_or_copy(cache_fp, out_fp)  # cache HIT: skip decode + PNG encode entirely
                            _texstats["hit"] += 1
                        else:
                            img = to.image
                            if img:
                                if is_normal: img = unswizzle_normal(img)
                                texpool.save(img, out_fp, cache_fp)   # serial or threaded save; publishes to cache
                                _texstats["miss"] += 1
                return nm
            except Exception:
                return None

        def _emission_enabled(mat, shader_name):
            """Unity's emission-ENABLE rule (why a nonzero _EmissionColor may still NOT glow in-game).
            - Standard-family shaders gate emission on the `_EMISSION` keyword (m_ValidKeywords / legacy
              m_ShaderKeywords string). Disabled -> it lands in m_InvalidKeywords or is simply absent.
            - EFT's custom shaders bake emission into the shader VARIANT: the name carries 'Emissive'
              (e.g. 'p0/Reflective/Bumped Emissive Specular SMap') and they emit with NO `_EMISSION` keyword.
            Materials whose emission is OFF still serialize a stale _EmissionColor (e.g. stones_pack_rebake_noheat
            -> a bright HDR orange, the '*_OFF' / '*_noheat' lamp variants, produce/vegetables). Capturing it
            unconditionally made those objects glow. Map-agnostic: reads only the material's own keyword/shader state."""
            if 'emissive' in (shader_name or '').lower(): return True
            vk = getattr(mat, 'm_ValidKeywords', None) or []
            ik = getattr(mat, 'm_InvalidKeywords', None) or []
            sk = getattr(mat, 'm_ShaderKeywords', None)
            if '_EMISSION' in vk: return True
            if '_EMISSION' in ik: return False
            if sk: return '_EMISSION' in sk          # legacy single space-joined keyword string
            return False                              # no keyword info + non-emissive shader -> emission OFF

        def capture_mat(mp):
            try:
                if mp is None or getattr(mp, "path_id", 0) == 0: return (None, None, None, [1, 1, 0, 0], [1, 1, 1], "opaque", 0.5, None, None)
                mstem, mpid, mat = srcid(mp)   # stable (source file, path_id) — file_id=0 collisions otherwise
                key = (mstem, mpid)
                if key in mat_cache: return mat_cache[key]
                sh = "?"; rtype = None
                # read the shader ONCE for both its name and its authoritative RenderType tag
                try:
                    sho = mat.m_Shader.read(); pf = getattr(sho, "m_ParsedForm", None)
                    sh = (g(pf, "m_Name") if pf is not None else None) or g(sho, "m_Name") or "?"
                    if pf is not None:
                        subsh = g(pf, "m_SubShaders") or []
                        if subsh:
                            tags = g(subsh[0], "m_Tags")
                            tagd = getattr(tags, "tags", None) if tags is not None else (tags if isinstance(tags, dict) else None)
                            if tagd: rtype = dict(tagd).get("RenderType")
                except Exception: pass
                tenvs = mat.m_SavedProperties.m_TexEnvs
                items = tenvs.items() if hasattr(tenvs, "items") else tenvs
                slots = {}
                for k, tenv in items:
                    tx = getattr(tenv, "m_Texture", None)
                    if tx is not None and getattr(tx, "path_id", 0): slots[str(k)] = (tx, tenv)
                tile = [1.0, 1.0, 0.0, 0.0]; alb = nrm = None; alb_slot = None
                for p in ALB:
                    if p in slots:
                        alb, tenv, alb_slot = slots[p][0], slots[p][1], p
                        try: tile = [round(float(tenv.m_Scale.x), 4), round(float(tenv.m_Scale.y), 4),
                                     round(float(tenv.m_Offset.x), 4), round(float(tenv.m_Offset.y), 4)]
                        except Exception: pass
                        break
                for p in NRM:
                    if p in slots: nrm = slots[p][0]; break
                # real Unity _Color (game asset value) for the flat-shading fallback when no albedo texture
                col = [1.0, 1.0, 1.0]; cd = {}
                try:
                    cols = mat.m_SavedProperties.m_Colors
                    citems = cols.items() if hasattr(cols, "items") else cols
                    cd = {str(k): v for k, v in citems}
                    cv = cd.get("_Color") or cd.get("_BaseColor") or cd.get("_TintColor") or cd.get("_MainColor")
                    if cv is not None:
                        # 4 components: _Color.a is the AUTHORED transparency of Unity's Transparent shader family
                        # (per-pane glass opacity — how dirty/opaque BSG made each pane). Consumers using col[:3]
                        # are unaffected; gltfbuild's glass branch reads [3] for the blend factor.
                        col = [round(float(cv.r), 4), round(float(cv.g), 4), round(float(cv.b), 4),
                               round(float(getattr(cv, "a", 1.0)), 4)]
                except Exception:
                    pass
                # authoritative transparency ROLE from Unity's RenderType tag + render queue + _Cutoff
                rq = -1; cut = 0.5; fd = {}
                try: rq = int(g(mat, "m_CustomRenderQueue", default=-1))
                except Exception: pass
                try:
                    floats = mat.m_SavedProperties.m_Floats
                    fitems = floats.items() if hasattr(floats, "items") else floats
                    fd = {str(k): float(v if not isinstance(v, (list, tuple)) else v[1]) for k, v in fitems}
                    if "_Cutoff" in fd: cut = round(fd["_Cutoff"], 3)
                except Exception: pass
                rt = (rtype or "").lower()
                if rt == "transparentcutout":  role = "cutout"
                elif rt == "transparent":      role = "glass" if rq >= 2900 else "decal"
                else:                          role = "opaque"
                # WATER (map-agnostic, material-level): any shader that NAMES water is genuine water — puddles, wet-ground
                # decals, lakes/seas on every map (woods `Decal/Water Deferred Decal`, shoreline ocean, etc.). Unambiguous
                # (never a false glass/floor). The big untextured water PLANES whose shader does NOT name water (the woods
                # lake uses a reflective-no-_MainTex shader) are tagged geometrically per-instance in the renderer loop below
                # (they need the mesh bounds), so this material-level rule stays name-only and can't mislabel a reflective floor.
                if "water" in (sh or "").lower(): role = "water"
                # Vert Paint multi-layer SPLAT: capture all 3 layers (albedo + normal + per-layer tiling + colour) +
                # the _Heights mask + _BlendStrength. Blend RE'd statically from the DX11 fragment (UnityPy blob ->
                # LZ4 -> DXBC -> fxc): per layer i, w_i = pow(Heights(uv_i)*vertexColor_i, _BlendStrength), normalised;
                # albedo/normal/smoothness = sum(w_i * layer_i). The viewer reproduces this exact blend.
                vp = None
                if "_MainTex1" in slots or "_MainTex2" in slots:
                    cdict = {}
                    try:
                        _cs = mat.m_SavedProperties.m_Colors; _ci = _cs.items() if hasattr(_cs, "items") else _cs
                        cdict = {str(k): v for k, v in _ci}
                    except Exception: pass
                    def _layer(i):
                        la = slots.get(f"_MainTex{i}"); ln = slots.get(f"_BumpMap{i}"); til = [1.0, 1.0, 0.0, 0.0]
                        if la:
                            try: t = la[1]; til = [round(float(t.m_Scale.x), 4), round(float(t.m_Scale.y), 4), round(float(t.m_Offset.x), 4), round(float(t.m_Offset.y), 4)]
                            except Exception: pass
                        lc = cdict.get(f"_Color{i}")
                        lcol = [round(float(lc.r), 4), round(float(lc.g), 4), round(float(lc.b), 4)] if lc is not None else [1.0, 1.0, 1.0]
                        return {"tex": exp_tex(la[0]) if la else None, "nrm": exp_tex(ln[0], True) if ln else None, "uv": til, "col": lcol}
                    _hm = slots.get("_Heights")
                    vp = {"layers": [_layer(0), _layer(1), _layer(2)], "heights": exp_tex(_hm[0]) if _hm else None, "blend": round(fd.get("_BlendStrength", 1.0), 3)}
                    # SoftCutout Decal soft-alpha params (feather roads into the terrain): alpha = clamp(COLOR_0.a*astr - (acut-ahgt), 0, 1).
                    # Written ONLY when the material AUTHORS _AlphaStrength: ABSENT (engine-default feathering) vs
                    # EXPLICIT 0 (soft gate off -> opaque) are different render paths in the viewer, and an
                    # unconditional 0.0 default conflated them (the invisible-parking-lot / hard-dirt-road pair).
                    if "_AlphaStrength" in fd:
                        vp["astr"] = round(fd["_AlphaStrength"], 4)
                        vp["acut"] = round(fd.get("_Cutoff", 0.0), 4)
                        vp["ahgt"] = round(fd.get("_AlphaHeight", 0.0), 4)
                # FAITHFULNESS capture (UnityPy audit 2026-07-01): fields the pipeline previously DROPPED. All optional —
                # only written when meaningful, so scene.json stays compact and old maps re-extract identically otherwise.
                #  - _EmissionMap/_EmissionColor: illuminated store signage / lamps / exit signs (79 emissive mats on
                #    interchange; emColor is HDR — e.g. lamps [1.5,1.82,2.34]).
                #  - _Glossiness/_Metallic: real PBR scalars (~1100 mats carry them) replacing the shader-name heuristic.
                #  - _BumpScale: authored normal-map intensity (we hardcoded scale=1.0 before).
                extra = {}
                for p in ("_EmissionMap", "_EmissiveMap", "_Emission"):
                    if p in slots:
                        _em = exp_tex(slots[p][0])
                        if _em: extra["emis"] = _em
                        break
                try:
                    ev = cd.get("_EmissionColor")
                    if ev is not None:
                        ec = [round(float(ev.r), 4), round(float(ev.g), 4), round(float(ev.b), 4)]
                        # Honor a STANDALONE color-emissive ONLY when emission is actually enabled in Unity. When an
                        # emission MAP is present the color is merely its HDR factor -> keep it regardless (the map is
                        # the glow, and gating the map would darken custom screen shaders like Custom/TextureGlitch).
                        # This is the fix for the spurious "yellow bonfire stones" (stones_pack_rebake_noheat carried a
                        # stale _EmissionColor [2.738,0.963,0] with `_EMISSION` DISABLED) and the whole map-less
                        # emission-off class (the *_OFF lamps, produce, concrete decals).
                        if max(ec) > 0.0 and ("emis" in extra or _emission_enabled(mat, sh)):
                            extra["emisCol"] = ec
                except Exception: pass
                if "_Glossiness" in fd: extra["gloss"] = round(fd["_Glossiness"], 4)
                if "_Metallic" in fd: extra["metal"] = round(fd["_Metallic"], 4)
                if "_BumpScale" in fd and abs(fd["_BumpScale"] - 1.0) > 1e-3: extra["bumpScale"] = round(fd["_BumpScale"], 4)
                # REAL SPECULAR RESPONSE (UnityPy audit 2026-07: the per-pixel gloss data EFT actually ships).
                #  - spec: the _SpecMap slot of the p0/* Bumped-Specular shader family (legacy Unity specular
                #    convention: RGB=specular colour, A=gloss). ~606 Interchange materials bind it. Dumped like any
                #    other texture; the builder derives a roughness texture from it when the albedo alpha is unusable.
                #  - smA=1: the shader VARIANT ('Specular'/'SMap' family) packs SMOOTHNESS IN THE ALBEDO ALPHA
                #    (the ~858-texture family the Blender build wired via smoothness_alpha.json). Whether the alpha
                #    actually carries data (vs constant 1.0) is decided downstream from the texture itself —
                #    the flag only records the shader-family semantics. Presence-gated: absent on other families.
                for p in ("_SpecMap", "_SpecTex", "_GlossMap"):
                    if p in slots:
                        _sp = exp_tex(slots[p][0])
                        if _sp: extra["spec"] = _sp
                        break
                _shl = (sh or "").lower()
                #    (also triggered when the bound albedo SLOT NAME itself declares the packing — e.g. ANGRYMESH
                #    rocks bind '_BaseAlbedoASmoothness'; the slot name IS the semantic, no shader-name matching)
                if alb is not None and ("specular" in _shl or "smap" in _shl
                                        or "asmoothness" in (alb_slot or "").lower()): extra["smA"] = 1
                # DETAIL MAPS (the up-close micro-texture layer the game blends over the albedo/normal). Presence-
                # gated: keys written ONLY when a slot is bound (like astr), so other maps re-extract identically.
                # TWO authoring conventions observed (census probe 2026-07-12, map-agnostic slot/float names only):
                #  - Unity Standard: _DetailAlbedoMap/_DetailNormalMap tiled via the tenv _ST (+_DetailNormalMapScale).
                #  - ANGRYMESH PBR Rocks (Interchange's actual detail users): _DetailAlbedo/_DetailNormalMap with
                #    ST=identity and the REAL tiling in floats _DetailUVScale/_DetailNormalUVScale, intensities in
                #    _DetailAlbedoIntensity/_DetailNormalMapIntensity.
                # Captured per channel: name + [sx,sy,ox,oy] uv + intensity. detMask for faithfulness (viewer may ignore).
                def _det_uv(tenv, scale_float):
                    try:
                        st = [round(float(tenv.m_Scale.x), 4), round(float(tenv.m_Scale.y), 4),
                              round(float(tenv.m_Offset.x), 4), round(float(tenv.m_Offset.y), 4)]
                    except Exception:
                        st = [1.0, 1.0, 0.0, 0.0]
                    if st == [1.0, 1.0, 0.0, 0.0] and scale_float and fd.get(scale_float):
                        s = round(fd[scale_float], 4)
                        if s > 0: st = [s, s, 0.0, 0.0]                  # uniform float tiling (ANGRYMESH convention)
                    return st
                _dA = slots.get("_DetailAlbedoMap") or slots.get("_DetailAlbedo")
                _dN = slots.get("_DetailNormalMap"); _dM = slots.get("_DetailMask")
                if _dA:
                    _t = exp_tex(_dA[0])
                    if _t:
                        extra["detA"] = _t
                        extra["detAuv"] = _det_uv(_dA[1], "_DetailUVScale")
                        if "_DetailAlbedoIntensity" in fd: extra["detAI"] = round(fd["_DetailAlbedoIntensity"], 4)
                if _dN:
                    _t = exp_tex(_dN[0], True)
                    if _t:
                        extra["detN"] = _t
                        extra["detNuv"] = _det_uv(_dN[1], "_DetailNormalUVScale")
                        _ni = fd.get("_DetailNormalMapScale", fd.get("_DetailNormalMapIntensity"))
                        if _ni is not None: extra["detNS"] = round(_ni, 4)
                if (_dA or _dN) and _dM:
                    _t = exp_tex(_dM[0])
                    if _t: extra["detMask"] = _t
                res = (exp_tex(alb) if alb else None, exp_tex(nrm, True) if nrm else None, sh, tile, col, role, cut, vp, extra or None)
                mat_cache[key] = res; return res
            except Exception:
                return (None, None, None, [1, 1, 0, 0], [1, 1, 1], "opaque", 0.5, None, None)

        def export_mesh(mesh_pptr):
            key = (lv, *fidpid(mesh_pptr))
            if key in exported: return exported[key]
            try:
                mesh = mesh_pptr.read(); nm = g(mesh, "m_Name", default=f"m{mesh_pptr.path_id}") or f"m{mesh_pptr.path_id}"
                obj_fn = f"{san(nm)}__{key[0]}_{key[1]}_{key[2]}.obj"
                fp = os.path.join(md, obj_fn)
                # re-export if missing OR a 0-byte STUB from a prior failed run: mesh.export() returns the bool False
                # on VertexCount<=0, and the old `open(fp,"w").write(mesh.export())` truncated the file to 0 bytes
                # BEFORE the write(False) raised -> a permanent empty stub that the exists() check never repaired
                # (dropped real geometry like hair_shop / upboard3 whose streamed .resS wasn't resolved that run).
                if (not os.path.exists(fp)) or os.path.getsize(fp) == 0:
                    data = mesh.export()
                    if isinstance(data, str) and data:
                        # encoding='utf-8' is CRITICAL: OBJ data embeds the mesh's `g <name>` line, and EFT has many
                        # CYRILLIC-named meshes (e.g. 'Сontainer_hospital', with a Cyrillic С). The Windows-default
                        # cp1252 encoder throws UnicodeEncodeError on those -> open('w') truncates the file to 0 bytes,
                        # export_mesh returns None, and the whole renderer is dropped (the invisible container-hospital
                        # WALLS). Writing UTF-8 fixes every non-ASCII-named mesh across all maps.
                        open(fp, "w", encoding="utf-8").write(data); exported[key] = obj_fn
                    else:
                        exported[key] = None                # genuinely empty mesh (e.g. 0-vert road-decal stub)
                        if os.path.exists(fp) and os.path.getsize(fp) == 0:
                            try: os.remove(fp)
                            except OSError: pass
                else:
                    exported[key] = obj_fn
                # vertex-colour sidecar (the vert-paint blend weights): OBJ export drops colours, so when the mesh
                # carries a Colour channel (vertex-data channel 3) decode it via MeshHandler -> <mesh>.vcol.npy.
                if exported.get(key):
                    vc_fp = fp[:-4] + ".vcol.npy"
                    if not os.path.exists(vc_fp):
                        try:
                            chs = mesh.m_VertexData.m_Channels
                            if len(chs) > 3 and getattr(chs[3], "dimension", 0):
                                from UnityPy.helpers.MeshHelper import MeshHandler
                                mh = MeshHandler(mesh); mh.process(); _c = getattr(mh, "m_Colors", None)
                                if _c is not None: np.save(vc_fp, np.asarray(_c, np.float32).reshape(-1, 4))
                        except Exception: pass
            except Exception:
                exported[key] = None
            return exported[key]

        def submesh_mats(mesh, mats):
            try: tri_per = [int(g(s, "triangleCount", default=None) or s.indexCount // 3) for s in mesh.m_SubMeshes]
            except Exception: tri_per = []
            subs = []
            for i, ntri in enumerate(tri_per):
                mp = mats[min(i, len(mats) - 1)] if mats else None
                alb, nrm, sh, tile, col, role, cut, vp, ex = capture_mat(mp)
                _s = {"n": int(ntri), "tex": alb, "nrm": nrm, "sh": sh, "uv": tile, "col": col, "role": role, "cut": cut}
                if vp: _s["vp"] = vp
                if ex: _s.update(ex)                                     # faithfulness extras: emis/emisCol/gloss/metal/bumpScale
                subs.append(_s)
            if not subs:
                alb, nrm, sh, tile, col, role, cut, vp, ex = capture_mat(mats[0] if mats else None)
                _s = {"n": -1, "tex": alb, "nrm": nrm, "sh": sh, "uv": tile, "col": col, "role": role, "cut": cut}
                if vp: _s["vp"] = vp
                if ex: _s.update(ex)
                subs = [_s]
            return subs

        def _flat_water_plane(mesh, W):
            """True iff this instance is a large, DEAD-FLAT, horizontal plane — a water/mirror sheet. World-space AABB from
            the mesh LOCAL AABB transformed by the FULL world matrix (robust to a plane authored in XY then rotated flat).
            Reflective props (wires/pipes/toilets) share the no-albedo reflective shader but have a real vertical extent, so
            the flatness gate (world Y span < 1.5m) excludes them. No map/mesh names, no hardcoded water level — map-agnostic."""
            try:
                aabb = g(mesh, "m_LocalAABB"); c = g(aabb, "m_Center"); e = g(aabb, "m_Extent")
                cx, cy, cz = float(c.x), float(c.y), float(c.z)
                ex, ey, ez = abs(float(e.x)), abs(float(e.y)), abs(float(e.z))
                M3 = np.asarray(W[:3, :3], np.float64); T3 = np.asarray(W[:3, 3], np.float64)
                corners = np.array([[cx + sx * ex, cy + sy * ey, cz + sz * ez]
                                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
                wsize = np.ptp(corners @ M3.T + T3, axis=0)
                return wsize[1] < 1.5 and wsize[0] > 60.0 and wsize[2] > 60.0 and (wsize[0] * wsize[2]) > 10000.0
            except Exception:
                return False

        # ---- Unity LODGroups -> SSoT switching table + renderer->LOD map (Stage A of the data-driven LOD architecture) ----
        # Every value is copied VERBATIM from Unity; the ONLY derived value is sizeWorld = m_Size x max world-axis scale
        # (Unity's lossyScale rule). center stays in UNITY world here -- assemble conjugates it by the map global_matrix when
        # it writes lod.json (same as instance placement). No hardcoded thresholds; keyed on path_ids present in every scene.
        lod0_rids = set(); all_lod_rids = set(); billboard_only_rids = set()
        rid2lod = {}                                              # renderer path_id -> (GLOBAL groupIdx, lodIndex)  [min index if shared]
        for o in env.objects:
            if o.type.name != "LODGroup": continue
            try:
                d = o.read_typetree(); mlods = d.get("m_LODs") or []
                if not mlods: continue
                gpid = (d.get("m_GameObject") or {}).get("m_PathID", 0)
                Wlg = world_of_go(gpid) if gpid else None
                if Wlg is None: continue
                Wlg = np.asarray(Wlg, np.float64); M3 = Wlg[:3, :3]; T3 = Wlg[:3, 3]
                wscale = float(max(np.linalg.norm(M3[:, 0]), np.linalg.norm(M3[:, 1]), np.linalg.norm(M3[:, 2])))
                rp = d.get("m_LocalReferencePoint") or {}
                rpl = np.array([rp.get("x", 0.0), rp.get("y", 0.0), rp.get("z", 0.0)], np.float64)
                center = (M3 @ rpl + T3).tolist()
                last_bb = bool(d.get("m_LastLODIsBillboard", False))
                srh = [round(float(L.get("screenRelativeHeight", 0.0) or 0.0), 5) for L in mlods]
                ftw = [round(float(L.get("fadeTransitionWidth", 0.0) or 0.0), 5) for L in mlods]
                gidx = len(lodgroups)
                grp = {"size": round(float(d.get("m_Size", 1.0) or 1.0) * (wscale or 1.0), 4),
                       "center": [round(c, 4) for c in center], "fadeMode": int(d.get("m_FadeMode", 0) or 0),
                       "lastIsBillboard": last_bb, "srh": srh, "ftw": ftw, "n": len(mlods)}
                if last_bb: grp["cullH"] = srh[-1]
                lodgroups.append(grp)
                nlast = len(mlods) - 1
                for li, lod in enumerate(mlods):
                    is_bb = last_bb and li == nlast                # the synthesized billboard impostor (we ship no billboard geometry)
                    for rpp in (lod.get("renderers") or []):
                        rid = (rpp.get("renderer") or {}).get("m_PathID", 0)
                        if not rid: continue
                        all_lod_rids.add(rid)
                        if li == 0: lod0_rids.add(rid)
                        if is_bb and rid not in rid2lod:
                            billboard_only_rids.add(rid)
                        else:
                            billboard_only_rids.discard(rid)
                            if rid not in rid2lod or li < rid2lod[rid][1]:   # shared renderer -> highest-detail (min) lodIndex
                                rid2lod[rid] = (gidx, li)
            except Exception:
                continue

        def keep_renderer(rpid):
            if rpid in billboard_only_rids:
                return False                                          # only ever the billboard impostor -> cull (no billboard mesh)
            if args.alllod:
                return True                                           # keep every LOD level (Stage C: runtime swaps among them)
            return (rpid in lod0_rids) or (rpid not in all_lod_rids)  # default / --lod0only: today's behavior (LOD0 + un-grouped)
        print(f"  [lv{lv}] LODGroups: +{sum(1 for grp in lodgroups)} cumulative={len(lodgroups)}, renderers tagged this level={len(rid2lod)}", flush=True)

        # ---- mesh renderers + skinned mesh renderers ----
        cnt = 0
        if not args.terrain_only:
            for o in env.objects:
                tn = o.type.name
                if tn not in ("MeshRenderer", "SkinnedMeshRenderer"): continue
                try:
                    if not keep_renderer(o.path_id): continue
                    mr = o.read(); go = mr.m_GameObject.read()
                    # Unity in-game-camera VISIBILITY flags (recorded; soft-culled in tarkmap/culls.py). A renderer is
                    # never drawn to the colour buffer if it is ShadowsOnly, disabled, or in an inactive hierarchy --
                    # the no-shadows/no-script viewer must drop these to match Unity. ShadowCastingMode: 0 Off,1 On,
                    # 2 TwoSided(DRAWN),3 ShadowsOnly. activeInHierarchy is parent-walked (local m_IsActive misses 50%+).
                    cast = int(g(mr, "m_CastShadows", default=1) or 0)
                    ren_on = bool(g(mr, "m_Enabled", default=True))
                    aih = active_in_hierarchy(mr.m_GameObject.path_id)
                    hidden = (cast == 3) or (not ren_on) or (not aih)
                    mesh_pptr = None
                    if tn == "SkinnedMeshRenderer":
                        mesh_pptr = g(mr, "m_Mesh")
                    else:
                        for comp in go.m_Component:
                            cp = comp[1] if isinstance(comp, (list, tuple)) else comp.component
                            co = cp.read()
                            if co.__class__.__name__ == "MeshFilter": mesh_pptr = co.m_Mesh; break
                    if mesh_pptr is None or getattr(mesh_pptr, "path_id", 0) == 0: continue
                    mesh = mesh_pptr.read()
                    nm = g(mesh, "m_Name", default="") or ""
                    # (removed the mesh-NAME LOD skip: keep_renderer already selects LOD0 via the LODGroup PPtrs, which
                    # is authoritative. A LOD0 renderer whose MESH is named "_LOD1" — e.g. the canopy metal-sheet covers
                    # 'metal_sheet_standing_LOD1' over the car — was wrongly dropped by the name heuristic. Only genuine
                    # non-LOD0 LODGroup members reach here already excluded, so the name check only over-culled.)
                    fn = export_mesh(mesh_pptr)
                    if not fn: continue
                    W = world_of_go(mr.m_GameObject.path_id)
                    if W is None: continue
                    subs = submesh_mats(mesh, mr.m_Materials or [])
                    # WATER (geometric): a big dead-flat horizontal plane whose material has NO albedo texture is a water/
                    # mirror sheet (its flat `col` was the "white water"). Reflective props share the shader but aren't flat.
                    if _flat_water_plane(mesh, W):
                        for _s in subs:
                            if _s.get("role") not in ("water", "cutout") and not _s.get("tex") \
                                    and not str(_s.get("sh") or "").lower().startswith("shadow"):
                                _s["role"] = "water"
                    inst = {"mesh": fn, "m": [round(float(v), 5) for v in W.flatten()], "subs": subs,
                            "lv": lv, "kind": "mesh", "root": root_of(mr.m_GameObject.path_id),
                            "cast": cast, "renON": ren_on, "aih": aih, "drop": hidden}
                    _lt = rid2lod.get(o.path_id)                  # Stage A: tag LODGroup membership (g=global group idx, i=lod index)
                    if _lt: inst["lod"] = {"g": _lt[0], "i": _lt[1]}
                    instances.append(inst)
                    cnt += 1
                except Exception:
                    continue

        # ---- terrain ----
        tcnt = 0
        for o in env.objects:
            if o.type.name != "Terrain": continue
            try:
                t = o.read(); tdpp = g(t, "m_TerrainData")
                tdata = tdpp.read(); tname = san(g(tdata, "m_Name", default=f"terr{o.path_id}"))
                obj_fn = f"terrain_{lv}_{tname}.obj"
                fp = os.path.join(md, obj_fn)
                if not os.path.exists(fp):
                    write_terrain_obj(tdata, fp, step=args.terrain_step)
                # bake albedo whenever the PNG is missing (decoupled from OBJ existence)
                alb_name = f"terrain_{lv}_{tname}_albedo"
                alb_path = os.path.join(td, alb_name + ".png")
                # L18: re-bake if the PNG is missing OR a 0-byte stub from a truncated/failed prior run (mirrors the
                # mesh-export size guard) so a pre-fix or partially-written albedo is never silently reused.
                if (not os.path.exists(alb_path)) or os.path.getsize(alb_path) == 0:
                    alb = bake_terrain_albedo(tdata)
                    if alb is not None: alb.save(alb_path)
                    else:
                        alb_name = None
                        if os.path.exists(alb_path) and os.path.getsize(alb_path) == 0:
                            try: os.remove(alb_path)
                            except OSError: pass
                # export raw splat layers + control maps so the builder can render a SHARP tiling material
                # (the flat baked albedo above stays as a fallback). Keyed by tile name (san of m_Name).
                try: export_terrain_splat(tdata, tname, splat_root, terrain_manifest, _layer_saved)
                except Exception as e: print(f"  splat export err: {e}")
                W = world_of_go(t.m_GameObject.path_id)
                if W is None: W = np.eye(4)
                instances.append({"mesh": obj_fn, "m": [round(float(v), 5) for v in W.flatten()],
                                  "subs": [{"n": -1, "tex": alb_name, "nrm": None, "sh": "terrain", "uv": [1, 1, 0, 0]}],
                                  "lv": lv, "kind": "terrain"})
                tcnt += 1
            except Exception as e:
                print(f"  terrain err: {e}")
        print(f"level{lv}: +{cnt} mesh +{tcnt} terrain  total={len(instances)} meshes={len([v for v in exported.values() if v])} tex={len(tex_done)} ({time.time()-t0:.0f}s)", flush=True)
        del env, tfm, wcache, go2tf; gc.collect()

    texpool.close()                                                     # flush all background PNG writes to disk
    if texpool.errors:
        print(f"  WARNING: {texpool.errors} async PNG write(s) failed", flush=True)
    if _TEXCACHE_ENABLED:
        print(f"  texcache: {_texstats['hit']} hits, {_texstats['miss']} misses", flush=True)
    terrain_manifest["layers"] = sorted(_layer_saved.keys())
    if terrain_manifest["tiles"]:
        os.makedirs(splat_root, exist_ok=True)
        json.dump(terrain_manifest, open(os.path.join(splat_root, "manifest.json"), "w"), indent=1)
        print(f"terrain splat: {len(terrain_manifest['tiles'])} tiles, {len(_layer_saved)} layer textures -> {splat_root}")
    json.dump({"instances": instances, "up": "unity", "levels": levels, "lodGroups": lodgroups, "lod_schema": 1,
               "note": "OBJ verts are UnityPy X-flipped+winding-reversed; builder must un-flip"},
              open(os.path.join(out, "scene.json"), "w"))
    print(f"  LOD: {len(lodgroups)} LODGroups, {sum(1 for it in instances if it.get('lod'))} tagged instances", flush=True)
    print(f"\nDONE {len(instances)} instances, {len([v for v in exported.values() if v])} meshes, "
          f"{len(tex_done)} textures in {time.time()-T0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
