#!/usr/bin/env python
"""assemble_bevy.py  --  .eftpack emitter for the native Bevy EFT map viewer.

A fork of tarkmap/assemble_instanced.py. It REUSES the correctness code verbatim
(instmath.make_conjugator / instmath.bake_into, culls.Culls.filter, objio.load_obj/
load_vcol, matsig.sub_sig) and DIVERGES only where the target engine differs:

  * the web three-way instance/bake GATE (det<0 -> bake, ortho>=0.02 shear -> bake,
    else EXT_mesh_gpu_instancing TRS) is REPLACED by: emit the FULL conjugated 3x4
    affine (INCLUDING shear + mirror) for every kept instance into instances.bin.
    Bevy's raw instance buffer (glam Affine3A) is shear- and det<0-correct; a MIRROR
    flag bit tells the renderer to flip winding/front-face instead of baking normals.
    instmath.bake_into is used ONLY for the rank-deficient / degenerate 3x3 case
    (flattened billboard/decal planes) -- the pinv fallback -- exactly per the
    tarkov-unity-extraction skill's #1 rule: NEVER TRS-decompose.

  * a NEW (lv, lod.g) -> keep-min(lod.i) LOD-shell dedup replaces the web payload
    split. (No-op on an already-LOD0-resolved scene; ~47% cut on an --alllod scene.)

  * the ENTIRE web-lossy tail is DROPPED: no build_textures 512 downscale, no
    gltf-transform quantize/etc1s/uastc/meshopt/fix-texcoords/deinstance, no
    split_glb, no EXT_mesh_gpu_instancing TRS split, no ../slice/tex id indirection.
    Full-res textures are REFERENCED IN PLACE by absolute path; Bevy imports BC7/BC5
    on load. Sidecars (terrain/lights/SH volume) are referenced, never copied.

Output is the self-describing .eftpack v1 contract:
  <pack>/manifest.json   -- declares every stride/offset; the loader reads layout
                            from here and hardcodes nothing (emitter & loader can't
                            drift).
  <pack>/meshes.bin      -- interleaved vertices (all meshes) then u32 indices.
  <pack>/instances.bin   -- fixed-stride instance records, full row-major 3x4 affine.
  <pack>/materials.json  -- one record per unique (submesh) material signature.

Usage:
  python -m eft_pipeline.assemble_bevy [map=interchange] [--out <dir.eftpack>] [--limit N] [--self-contained]

--self-contained (redistribution PR3, default OFF): copy every referenced texture into
<pack>/tex/ and every sidecar file (volume.bin/volume.json/volume.vis.bin, terrain_layers/,
lights_*.json) INTO the pack, and write pack-RELATIVE paths everywhere. The Rust loader
(Pack::resolve_path) resolves relative manifest/materials/sidecar paths against the pack
dir; absolute (legacy dev) paths pass through untouched, so default builds are unchanged.
manifest.datasetPath stays ABSOLUTE for provenance; "selfContained": true marks the mode.
"""
import sys, os, time, json, glob, shutil, functools, math
import numpy as np

print = functools.partial(print, flush=True)
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass

# --- reuse the tarkmap correctness core VERBATIM (vendored into the new repo) --------------------------------
# Primary: the vendored package. Dev fallback: the upstream tarkmap in place, so this
# script is runnable against the real interchange_v2 dataset today.
try:
    from eft_pipeline.tarkmap_core import instmath, culls, objio, matsig
    from eft_pipeline.tarkmap_core.config import MapConfig
except Exception:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    try:
        from eft_pipeline.tarkmap_core import instmath, culls, objio, matsig
        from eft_pipeline.tarkmap_core.config import MapConfig
    except Exception:
        _UP = r"C:\Users\user\beamng_blender_pipeline\tarkmap"
        sys.path.insert(0, _UP)
        from tarkmap import instmath, culls, objio, matsig            # type: ignore
        from tarkmap.config import MapConfig                          # type: ignore

# make_conjugator / mat4_colmajor live in instmath. We import the module (both are
# reused) but DELIBERATELY use apply_global(m)[:12] (ROW-MAJOR 3x4) for the instance
# buffer and NEVER instmath.mat4_colmajor -- that is the glTF COLUMN-MAJOR transpose,
# wrong for the eftpack affine contract.
make_conjugator = instmath.make_conjugator
bake_into       = instmath.bake_into
Culls           = culls.Culls
load_obj        = objio.load_obj
load_vcol       = objio.load_vcol
sub_sig         = matsig.sub_sig

try:
    from PIL import Image as _PILImage
except Exception:
    _PILImage = None

# =============================================================================================================
# .eftpack v1 fixed binary layouts (kept in ONE place; the manifest is generated from these so it can't drift)
# =============================================================================================================
VDT = np.dtype([('pos', '<f4', (3,)), ('nrm', '<f4', (3,)), ('uv', '<f4', (2,)), ('col', 'u1', (4,))])
assert VDT.itemsize == 36
VERTEX_ATTRS = [
    {"name": "position", "fmt": "f32x3",    "offset": 0},
    {"name": "normal",   "fmt": "f32x3",    "offset": 12},
    {"name": "uv",       "fmt": "f32x2",    "offset": 24},
    {"name": "color",    "fmt": "unorm8x4", "offset": 32},
]

# instance stride padded to 80 (multiple of 16) so a WGSL storage-buffer read maps to
# 3x vec4 (affine) + 2x vec4 (ids+flags+pad) with no straddling. The 3 trailing u32 are pad.
IDT = np.dtype([('affine', '<f4', (12,)), ('meshId', '<u4'), ('lodGroup', '<i4'),
                ('lodIndex', '<i4'), ('rootId', '<u4'), ('flags', '<u4'), ('_pad', '<u4', (3,))])
assert IDT.itemsize == 80
INSTANCE_FIELDS = [
    {"name": "affine",   "fmt": "f32x12", "offset": 0,  "note": "ROW-MAJOR world 3x4 incl shear = apply_global(m)[:12]"},
    {"name": "meshId",   "fmt": "u32",    "offset": 48},
    {"name": "lodGroup", "fmt": "i32",    "offset": 52, "note": "scene lod.g, or -1"},
    {"name": "lodIndex", "fmt": "i32",    "offset": 56, "note": "scene lod.i, or -1"},
    {"name": "rootId",   "fmt": "u32",    "offset": 60, "note": "index into manifest.roots"},
    {"name": "flags",    "fmt": "u32",    "offset": 64},
]

# instance flag bits
FLAG_MIRROR  = 1 << 0   # det3(affine) < 0 -> renderer flips front-face / winding for this instance
FLAG_TERRAIN = 1 << 1   # MicroSplat terrain tile (drive with the terrain splat shader)
FLAG_BAKED   = 1 << 2   # identity-affine, geometry PRE-BAKED to world (degenerate fallback); no normal-matrix

ROLES = ('opaque', 'cutout', 'glass', 'decal', 'water')


# =============================================================================================================
# small ported helpers (material math + content tests) -- verbatim math from gltfbuild / assemble_instanced
# =============================================================================================================
def _srgb2lin(c):
    c = min(max(float(c), 0.0), 1.0)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _col4(col):
    """Unity _Color (sRGB) -> LINEAR rgb; alpha stays linear (coverage). == materials.json tint[4]."""
    c = (list(col or []) + [1, 1, 1, 1])[:4]
    return [_srgb2lin(c[0]), _srgb2lin(c[1]), _srgb2lin(c[2]), round(float(c[3]), 4)]


def _pbr(sh, role):
    """(roughness, metallic) from shader-string + role only (no map/mesh names -> map-agnostic)."""
    if role in ('water', 'glass'): return 0.05, 0.0
    s = (sh or '').lower()
    if any(h in s for h in ('chrome', 'metal')):                return 0.4, 0.85
    if any(h in s for h in ('specular', 'reflective', 'smap')): return 0.55, 0.0
    return 0.9, 0.0


class _TexTest:
    """Full-res content tests (need PIL). Cached. Degrade to False when PIL is absent."""
    def __init__(self, ds):
        self.ds = ds; self._nm = {}; self._cov = {}

    def _open(self, name):
        if _PILImage is None or not name: return None
        try: return _PILImage.open(os.path.join(self.ds, 'tex', name + '.png'))
        except Exception: return None

    def albedo_is_normalmap(self, name):
        """A 'decal' whose albedo is really a bevel NORMAL map (avg ~[128,128,255]) -> drop it
        (deferred bevel decals would paint every edge blue). Map-agnostic, no name hardcoding."""
        if name in self._nm: return self._nm[name]
        res = False
        im = self._open(name)
        if im is not None:
            try:
                r, g, b = im.convert('RGB').resize((8, 8)).resize((1, 1)).getpixel((0, 0))
                res = (b > 200 and abs(r - 128) < 45 and abs(g - 128) < 45 and b > r + 55 and b > g + 55)
            except Exception: res = False
        self._nm[name] = res; return res

    def alpha_coverage(self, name):
        """Universal DATA-DRIVEN coverage detection: returns the Otsu-split alpha cutoff when the
        texture's own alpha histogram says it is authored hole-coverage, else None. No shader
        names, no per-asset rules, and no fixed cutoff — the histogram supplies its own split.
        Three criteria, each physically motivated (validated across foliage atlases, ground
        overlays, camo nets vs. AO/height/smoothness alpha on floors and props):
          * Otsu separability >= 0.5      — the alpha is clearly BIMODAL (two populations);
          * transparent-mode mean <= 0.1  — the low mode is actual HOLES (data-alpha lows sit
                                            higher: AO/height rarely reaches true zero);
          * solid-mode mean >= 0.3        — the stuff you KEEP is meaningfully opaque (alpha-as-
                                            data clusters far below: measured 0.12-0.22 on the
                                            false-positive floors vs 0.36-0.97 on real coverage).
        The old fixed-number test ((A<80)>10% AND (A>200)>2%) missed real foliage whose leaves
        are semi-soft (brush_dry: 95% holes but few texels above 200) — exactly the class of
        hardcoded-threshold bug this replaces."""
        if not name: return None
        if name in self._cov: return self._cov[name]
        res = None
        im = self._open(name)
        if im is not None and im.mode == 'RGBA':
            try:
                a = np.asarray(im.getchannel('A'), np.float64) / 255.0
                hist, _ = np.histogram(a, bins=256, range=(0.0, 1.0))
                w = hist / max(hist.sum(), 1)
                lv = (np.arange(256) + 0.5) / 256.0
                mean_all = (w * lv).sum()
                total_var = ((lv - mean_all) ** 2 * w).sum()
                if total_var >= 1e-6:
                    wc = np.cumsum(w); mc = np.cumsum(w * lv); mt = mc[-1]
                    w0 = wc; w1 = 1.0 - wc
                    ok = (w0 > 0) & (w1 > 0)
                    m0 = np.where(ok, mc / np.maximum(w0, 1e-12), 0.0)
                    m1 = np.where(ok, (mt - mc) / np.maximum(w1, 1e-12), 0.0)
                    between = w0 * w1 * (m0 - m1) ** 2
                    t = int(np.argmax(between))
                    w_lo = float(wc[t])
                    if (between[t] / total_var >= 0.5     # bimodal
                            and m0[t] <= 0.1              # low mode = true holes
                            and m1[t] >= 0.3              # solid mode = meaningfully opaque
                            and 0.005 <= w_lo <= 0.995):  # both classes non-trivial (Codex: one
                        res = float(lv[t])                # stray texel must not flip a texture)
            except Exception:
                res = None
        self._cov[name] = res; return res


# =============================================================================================================
# material factory -- dedups on the sub's material signature, emits materials.json records (retargeted from
# gltfbuild.material_for). Textures are referenced IN PLACE by absolute path (loader imports BC7/BC5).
# =============================================================================================================
class MaterialFactory:
    def __init__(self, ds):
        self.ds = ds; self.cache = {}; self.records = []

    def _tex(self, name):
        return os.path.join(self.ds, 'tex', name + '.png').replace('\\', '/') if name else None

    def get(self, sb):
        key = sub_sig([sb])                                   # exact same key space as the geometry grouping
        hit = self.cache.get(key)
        if hit is not None: return hit
        mid = len(self.records)
        self.records.append(self._build(mid, sb))
        self.cache[key] = mid
        return mid

    def _build(self, mid, sb):
        role = sb.get('role') or 'opaque'
        if role not in ROLES: role = 'opaque'
        sh   = sb.get('sh') or ''
        col  = sb.get('col')
        rough, metal = _pbr(sh, role)
        gloss = sb.get('gloss'); metalf = sb.get('metal')
        if gloss is not None:  rough = round(max(0.02, min(1.0, 1.0 - float(gloss))), 4)   # real smoothness wins
        if metalf is not None: metal = round(max(0.0, min(1.0, float(metalf))), 4)

        tint = _col4(col)
        alpha_mode, alpha_cutoff = 'OPAQUE', 0.0
        if role == 'cutout':
            alpha_mode, alpha_cutoff = 'MASK', round(float(sb.get('cut', 0.5) or 0.5), 4)
        elif role in ('glass', 'decal', 'water'):
            alpha_mode = 'BLEND'
        if role == 'glass':
            # glass KEEPS its dirt-film albedo; tint.a = authored _Color.a (or 0.28 stand-in); glossy dielectric.
            rough, metal = 0.05, 0.0
            ga = tint[3] if (col and len(col) >= 4) else 0.28
            tint = [tint[0], tint[1], tint[2], ga]

        # emissive (illuminated signage/lamps). HDR emColor normalized into factor; overdrive kept in .hdr.
        # honored on non-decal/non-glass (BLEND shaders repurpose _EmissionColor as a tint) and non-vp.
        emissive = None
        if role not in ('decal', 'glass') and not sb.get('vp'):
            et, ec = sb.get('emis'), sb.get('emisCol')
            if et or (ec and max(ec) > 0):
                mx = max(ec) if ec else 1.0
                if et and ec and mx > 1.0:   factor = [min(c / mx, 1.0) for c in ec]
                elif ec and max(ec) > 0:     factor = [min(c, 1.0) for c in ec]
                else:                        factor = [1.0, 1.0, 1.0]
                emissive = {"texture": self._tex(et), "factor": [round(x, 4) for x in factor],
                            "hdr": round(mx, 3) if mx > 1.0 else 1.0}

        normal = self._tex(sb.get('nrm'))
        rec = {
            "id": mid,
            "role": role,
            "albedo": self._tex(sb.get('tex')),
            "normal": normal,
            "uvXform": [round(float(x), 6) for x in (sb.get('uv') or [1, 1, 0, 0])],
            "alphaMode": alpha_mode,
            "alphaCutoff": alpha_cutoff,
            "tint": [round(float(x), 6) for x in tint],
            "metallic": round(float(metal), 4),
            "roughness": round(float(rough), 4),
            "normalScale": round(float(sb['bumpScale']), 4) if sb.get('bumpScale') is not None else 1.0,
            # normal maps are DirectX-convention (green points down). The loader must flip G on import
            # (BC5), OR the shader must negate n.y. Recorded here because textures are referenced in place
            # and cannot be pre-flipped.
            "normalGreenFlip": normal is not None,
            "doubleSided": True,   # EFT deferred draws building shells solid from both sides (see gotcha)
            "emissive": emissive,
            # roughness sources kept as REAL fields (the web ~ra~/~mr~ synth textures are DROPPED):
            "roughnessFromAlbedoAlpha": bool(sb.get('smA')),          # roughness = 1 - albedo.a
            "specMap": self._tex(sb.get('spec')),                     # roughness from _SpecMap luma
            "vp": self._vp(sb.get('vp')),
            # #6 DETAIL MAPS: name-keyed up-close detail albedo/normal (ANGRYMESH rocks etc.).
            # RAW Unity _Detail*Map_ST is emitted here; the shader re-expresses it RELATIVE to the
            # baked+V-flipped base UV (uvXform) and applies the Unity-Standard x2 (x4.5948) mean-
            # neutralized albedo blend + whiteout normal blend + 8-15 m distance fade. See
            # CODEX_5_6_SHADOW_DETAIL_PLAN.md #6. Textures referenced in place from tex/ like normals.
            "detail": self._detail(sb),
        }
        return rec

    def _detail(self, sb):
        """Detail-map block {albedo, albedoUv, albedoStrength, normal, normalUv, normalScale} or None.
        vp (Vert-Paint carrier-slot) subs are skipped (the Bevy vp path doesn't consume detail). UVs are
        the RAW Unity _Detail*Map_ST; the shader makes them relative to the baked base UV."""
        if sb.get('vp') or not (sb.get('detA') or sb.get('detN')):
            return None
        rec = {}
        if sb.get('detA'):
            rec["albedo"] = self._tex(sb['detA'])
            rec["albedoUv"] = [round(float(x), 6) for x in (sb.get('detAuv') or [1, 1, 0, 0])]
            rec["albedoStrength"] = round(float(sb['detAI']), 4) if sb.get('detAI') is not None else 1.0
            rec["albedoMeanGain"] = self._detail_mean(sb['detA'])
        if sb.get('detN'):
            rec["normal"] = self._tex(sb['detN'])
            rec["normalUv"] = [round(float(x), 6) for x in (sb.get('detNuv') or [1, 1, 0, 0])]
            rec["normalScale"] = round(float(sb['detNS']), 4) if sb.get('detNS') is not None else 1.0
        return rec

    _DET_MEAN: dict = {}
    def _detail_mean(self, name):
        """Mean of the detail albedo in LINEAR space x 4.5948 (Unity Standard x2), for the shader's
        mean-neutralization (dark ANGRYMESH detail maps would otherwise darken surfaces ~2x under the
        Standard blend). Cached per texture; falls back to neutral [1,1,1] if the file is unreadable."""
        if name in self._DET_MEAN:
            return self._DET_MEAN[name]
        try:
            # NOTE: MaterialFactory has no _open (that's _TexTest) — calling self._open here was an
            # AttributeError swallowed by this except, silently neutralizing EVERY pack's detail
            # mean (dark ANGRYMESH detail maps then darken surfaces ~2x — the exact bug this code
            # exists to fix). Open the texture directly.
            im = _PILImage.open(os.path.join(self.ds, 'tex', name + '.png')).convert('RGB')
            im.thumbnail((256, 256))                       # mean is ~scale-invariant; keep it cheap
            a = np.asarray(im, np.float32) / 255.0
            lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
            m = [round(float(x), 5) for x in (lin.reshape(-1, 3).mean(0) * 4.5948)]
        except Exception as e:
            print(f"[bevy] detail mean fallback for {name}: {e}")
            m = [1.0, 1.0, 1.0]
        self._DET_MEAN[name] = m
        return m

    def _vp(self, vp):
        if not vp: return None
        layers = []
        for ly in (vp.get('layers') or []):
            layers.append({
                "albedo": self._tex(ly.get('tex')),
                "normal": self._tex(ly.get('nrm')),
                "uv":  [round(float(x), 6) for x in (ly.get('uv') or [1, 1, 0, 0])],
                "tint": [round(float(x), 6) for x in (ly.get('col') or [1, 1, 1])],
            })
        rec = {"layers": layers, "heights": self._tex(vp.get('heights')),
               "blend": float(vp.get('blend', 1.0))}
        if any(k in vp for k in ('astr', 'acut', 'ahgt')):
            rec["softCutout"] = [float(vp.get('astr', 0.0)), float(vp.get('acut', 0.0)), float(vp.get('ahgt', 0.0))]
        return rec


# =============================================================================================================
def _M3T(mg):
    """3x3 (rows) of a row-major 3x4/4x4 flat list, and translation T."""
    M3 = np.array([[mg[0], mg[1], mg[2]], [mg[4], mg[5], mg[6]], [mg[8], mg[9], mg[10]]], np.float64)
    T  = np.array([mg[3], mg[7], mg[11]], np.float64)
    return M3, T


def _degenerate(M3):
    """True only for a genuinely rank-deficient 3x3 (a mesh flattened to a plane -> no invertible
    normal transform). NOT true for a small-but-uniform scale. Cheap det gate first, SVD to confirm."""
    det = float(np.linalg.det(M3))
    scale = float(np.abs(M3).max())
    if scale <= 1e-12: return True
    if abs(det) > (scale ** 3) * 1e-9: return False           # clearly invertible
    s = np.linalg.svd(M3, compute_uv=False)
    return bool(s[0] <= 0 or s[-1] < s[0] * 1e-6)


def _corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float64)


# =============================================================================================================
# SELF-CONTAINED PACK mode (redistribution PR3, --self-contained). Everything a shipped pack
# needs is COPIED into the pack dir and referenced pack-RELATIVE; the Rust loader
# (Pack::resolve_path) resolves relative paths against the pack dir and passes absolute
# (legacy dev) paths through, so default builds stay byte-identical.
# =============================================================================================================
class _PackShipper:
    """Copies files into the staging pack dir and tallies count/bytes for the summary line."""

    def __init__(self, out_dir):
        self.out = out_dir; self.files = 0; self.bytes = 0; self.missing = []
        self._by_src = {}    # normalized source path -> pack-relative path (copy dedup)
        self._by_base = {}   # claimed tex/ basename  -> owning source path (collision check)
        self._sha = {}       # source path -> short content hash (lazy, for collisions only)

    def _sha8(self, path):
        h = self._sha.get(path)
        if h is None:
            import hashlib
            hh = hashlib.sha1()
            with open(path, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    hh.update(chunk)
            h = self._sha[path] = hh.hexdigest()[:8]
        return h

    def ship(self, src, rel):
        """Copy src -> <pack>/<rel> (rel = pack-relative, posix slashes). None if src missing."""
        if not src or not os.path.exists(src):
            return None
        dst = os.path.join(self.out, rel.replace('/', os.sep))
        d = os.path.dirname(dst)
        if d:
            os.makedirs(d, exist_ok=True)
        shutil.copy2(src, dst)
        self.files += 1; self.bytes += os.path.getsize(src)
        return rel

    def ship_tex(self, src):
        """Copy a referenced texture into <pack>/tex/ FLAT (basenames kept). Two DIFFERENT
        source files sharing a basename get a deterministic short-content-hash suffix
        (<stem>.<sha1[:8]>.png); identical content shares one copy. A MISSING source still
        returns tex/<basename> (the loader falls back on a missing texture exactly as it
        did for a missing absolute path) and is tallied for the summary."""
        src = os.path.normpath(src)
        hit = self._by_src.get(src)
        if hit is not None:
            return hit
        base = os.path.basename(src)
        if not os.path.exists(src):
            self.missing.append(src)
            rel = self._by_src[src] = 'tex/' + base
            return rel
        owner = self._by_base.get(base)
        if owner is not None and owner != src:
            if self._sha8(owner) == self._sha8(src):        # same bytes -> share the copy
                rel = self._by_src[src] = self._by_src[owner]
                return rel
            stem, ext = os.path.splitext(base)
            base = f"{stem}.{self._sha8(src)}{ext}"         # different bytes -> deterministic suffix
        else:
            self._by_base[base] = src
        rel = self.ship(src, 'tex/' + base)
        self._by_src[src] = rel
        return rel

    def ship_dir(self, srcdir, reldir, skip_suffixes=('.bak',)):
        """Copy every regular file of srcdir into <pack>/<reldir>/ (flat, sorted, backups skipped)."""
        n = 0
        if srcdir and os.path.isdir(srcdir):
            for fn in sorted(os.listdir(srcdir)):
                sp = os.path.join(srcdir, fn)
                if not os.path.isfile(sp) or fn.endswith(tuple(skip_suffixes)):
                    continue
                if self.ship(sp, f"{reldir}/{fn}"):
                    n += 1
        return n


def _self_contain_materials(records, shipper):
    """Rewrite EVERY texture path in the materials.json records to pack-relative tex/<name>,
    copying the files via shipper.ship_tex. Covers all texture-bearing fields: albedo, normal,
    specMap, emissive.texture, detail.albedo/.normal, vp.layers[].albedo/.normal, vp.heights."""
    for m in records:
        for k in ('albedo', 'normal', 'specMap'):
            if m.get(k): m[k] = shipper.ship_tex(m[k])
        em = m.get('emissive')
        if em and em.get('texture'): em['texture'] = shipper.ship_tex(em['texture'])
        det = m.get('detail')
        if det:
            for k in ('albedo', 'normal'):
                if det.get(k): det[k] = shipper.ship_tex(det[k])
        vp = m.get('vp')
        if vp:
            if vp.get('heights'): vp['heights'] = shipper.ship_tex(vp['heights'])
            for ly in vp.get('layers') or []:
                for k in ('albedo', 'normal'):
                    if ly.get(k): ly[k] = shipper.ship_tex(ly[k])


def _relativize_tl_manifest(path):
    """Defensive: rewrite any ABSOLUTE *.png path inside the COPIED terrain_layers manifest to
    its basename. The loader resolves those names relative to the sidecar's own dir (i.e.
    <pack>/terrain_layers/), so a basename IS the pack-relative terrain_layers/<name>.png.
    Current extractors already emit bare basenames -> normally a no-op."""
    try:
        d0 = json.load(open(path, encoding='utf-8'))
    except Exception:
        return
    def walk(o):
        if isinstance(o, dict): return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list): return [walk(v) for v in o]
        if isinstance(o, str) and os.path.isabs(o) and o.lower().endswith('.png'):
            return os.path.basename(o)
        return o
    d1 = walk(d0)
    if d1 != d0:
        json.dump(d1, open(path, 'w'), separators=(',', ':'))


def main():
    argv = sys.argv[1:]
    MAP = argv[0] if argv and not argv[0].startswith('-') else 'interchange'
    LIMIT = int(argv[argv.index('--limit') + 1]) if '--limit' in argv else 0
    SELF_CONTAINED = '--self-contained' in argv     # redistribution PR3; default OFF (dev builds unchanged)
    KEEP_LODS = '--keep-lods' in argv               # --alllod builds: keep every LOD shell for the viewer LOD selector
    OUT = (argv[argv.index('--out') + 1] if '--out' in argv
           else os.path.join(os.getcwd(), 'packs', f'{MAP}.eftpack'))
    # ATOMIC EMISSION (Codex review): write into a staging sibling and swap at the end. Writing
    # blobs in place with the manifest last meant a mid-build failure left new meshes.bin under
    # the OLD manifest — a pack that loads without error and renders garbage.
    FINAL_OUT = OUT
    OUT = OUT + '.building'
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    cfg = MapConfig.load(MAP)
    DS = cfg.dataset
    scene = json.load(open(os.path.join(DS, 'scene.json'), encoding='utf-8'))
    tex = _TexTest(DS)

    # ---- STEP 1: structural culls (culls.Culls -- verbatim) --------------------------------------------------
    CULLS = Culls(cfg.get('cull'))
    inst, rep = CULLS.filter(scene['instances'])
    print(f"[bevy] cull: kept {rep['kept']:,}/{rep['raw']:,} (dropped {rep['dropped']:,}; "
          f"Unity-hidden {rep.get('hidden_unity', 0):,}); top dropped roots "
          f"{[r for r, _ in rep['top_dropped_roots'][:5]]}")
    if rep.get('offmap_backdrop'):
        print(f"[bevy] off-map backdrop cull: dropped {rep['offmap_backdrop']} distant-skyline instances")
    if rep['kept'] == 0 or rep['kept'] < rep['raw'] * 0.005:
        raise SystemExit(f"[bevy] FATAL: cull kept only {rep['kept']}/{rep['raw']} for '{MAP}'. Fix cull config.")

    # ---- STEP 2: DECAL normal-map albedo drop (correctness fix -- port) ---------------------------------------
    ndrop = 0
    for it in inst:
        keep = []
        for sb in it['subs']:
            if sb.get('role') == 'decal' and sb.get('tex') and tex.albedo_is_normalmap(sb['tex']):
                ndrop += 1; continue
            keep.append(sb)
        it['subs'] = keep
    inst = [it for it in inst if it.get('subs')]
    if ndrop: print(f"[bevy] dropped {ndrop} normal-map-albedo decal submeshes (would paint edges blue)")

    # ---- STEP 3: LOD-SHELL DEDUP -- group by (lv, lod.g), keep only lod.i == group-min -----------------------
    # (Replaces the web payload split. Untagged instances -- terrain, ungrouped meshes -- are ALWAYS kept.
    #  lod.g is a global/cumulative index so (lv,g) == g, but keying on (lv,g) is redundant-but-safe. This is a
    #  NO-OP on an already-LOD0-resolved scene.json and yields the ~47% cut only on an --alllod extraction.)
    if not KEEP_LODS:
        gmin = {}
        for it in inst:
            L = it.get('lod')
            if not L: continue
            k = (it['lv'], L['g'])
            gmin[k] = min(gmin.get(k, 1 << 30), L['i'])
        n0 = len(inst)
        kept = []
        for it in inst:
            L = it.get('lod')
            if not L or L['i'] == gmin[(it['lv'], L['g'])]:
                kept.append(it)
        inst = kept
        print(f"[bevy] LOD-shell dedup: {len(inst):,}/{n0:,} instances kept "
              f"({n0 - len(inst):,} coarser LOD shells removed)")
    else:
        print(f"[bevy] --keep-lods: kept all {len(inst):,} LOD shells for the viewer LOD selector")

    # ---- STEP 4: global orientation (make_conjugator -- verbatim) --------------------------------------------
    G4 = cfg.coord_matrix()
    apply_global, det3, GID, GDET = make_conjugator(G4)
    G3 = G4[:3, :3].astype(np.float64)
    print(f"[bevy] global orientation: det={GDET:+.2f} mode={'identity' if GID else 'conjugate'}")

    # ---- STEP 5: group kept instances by (mesh, material-signature) (matsig.sub_sig -- verbatim) --------------
    by_mesh = {}
    for it in inst:
        by_mesh.setdefault((it['mesh'], sub_sig(it['subs'])), []).append(it)
    groups = list(by_mesh.keys())
    if LIMIT: groups = groups[:LIMIT]
    print(f"[bevy] {len(inst):,} instances, {len(by_mesh):,} unique (mesh,material) groups, "
          f"{len({k[0] for k in by_mesh}):,} unique meshes ({time.time()-t0:.0f}s)")

    # ---- STEP 6: build geometry + instances ------------------------------------------------------------------
    MF = MaterialFactory(DS)
    obj_cache = {}
    vbuf = bytearray(); ibuf = bytearray()                 # meshes.bin = all verts, then all u32 indices
    meshes_meta = []                                       # per-mesh manifest records (idxOffset patched later)
    inst_records = []                                      # (affine12, meshId, lodGroup, lodIndex, rootId, flags)
    baked = {}                                             # degenerate fallback: matId -> world geom (bake_into)
    n_baked = 0
    root_names = [""]; root_index = {"": 0}
    def rid(name):
        i = root_index.get(name)
        if i is None:
            i = len(root_names); root_index[name] = i; root_names.append(name)
        return i
    wmin = np.array([np.inf] * 3); wmax = np.array([-np.inf] * 3)
    def upd_bounds(pts):
        nonlocal wmin, wmax
        if len(pts):
            wmin = np.minimum(wmin, pts.min(0)); wmax = np.maximum(wmax, pts.max(0))

    utris = 0
    for gi, mkey in enumerate(groups):
        mname = mkey[0]
        if mname not in obj_cache:
            obj_cache[mname] = (load_obj(DS, mname), load_vcol(DS, mname))
        lo, vcol = obj_cache[mname]
        if not lo: continue
        V, VT, F = lo
        if len(F) == 0: continue
        subs = by_mesh[mkey][0]['subs']                    # consistent across the group (same material signature)

        # WATER recovery (correctness, map-agnostic): material-less+untextured lake/pond/river/ocean meshes -> water;
        # any sub whose shader names water -> water (drainage pools / puddle sheets the cull restored under Water).
        mnl = (mname or '').lower()
        if any(w in mnl for w in ('water', 'lake', 'pond', 'river', 'ocean')):
            for sb in subs:
                if not (sb.get('sh') or '').strip() and not sb.get('tex'):
                    sb['role'] = 'water'; sb['sh'] = 'water'
        for sb in subs:
            if 'water' in (sb.get('sh') or '').lower() and sb.get('role') != 'water':
                sb['role'] = 'water'
        is_terrain = any((s.get('sh') or '') == 'terrain' for s in subs) or by_mesh[mkey][0].get('kind') == 'terrain'

        # ---- per-submesh dedup / smooth-normal build (objio + the assemble geometry loop -- verbatim math) ----
        pending = []; f0 = 0
        for sb in subs:
            # UNIVERSAL alpha-coverage recovery — no shader lists, the texture data decides.
            # Unity's RenderType tag gives the extractor an authoritative role, but CUSTOM EFT
            # shaders (SpeedTreeEFT foliage, Cloth ground overlays, deferred one-offs) don't tag
            # TransparentCutout and fell through to 'opaque' -> solid black cards/sheets. For any
            # opaque textured sub whose alpha is NOT smoothness (smA — the game's own flag), ask
            # the albedo's alpha histogram whether it is authored hole-coverage (alpha_coverage:
            # Otsu bimodality + true-zero holes + opaque solid mode). Cutoff priority: the
            # material's own authored _Cutoff (game data) over the histogram's Otsu split.
            if (sb.get('role', 'opaque') == 'opaque' and not sb.get('smA')):
                _otsu = tex.alpha_coverage(sb.get('tex'))
                if _otsu is not None:
                    sb['role'] = 'cutout'
                    sb['cut'] = float(sb.get('cut') or _otsu)
            n = sb.get('n', -1); n = (len(F) - f0) if n < 0 else n
            if n <= 0 or f0 + n > len(F):
                if f0 + n > len(F):
                    print(f"[bevy] WARNING: submesh span overruns OBJ tris "
                          f"({f0}+{n} > {len(F)}) - geometry silently dropped for this sub")
                f0 += max(n, 0); continue
            if not CULLS.keep_submesh(sb): f0 += n; continue          # shadow / billboard-LOD / fog / proxy
            cor = F[f0:f0 + n]; f0 += n
            vi = cor[:, :, 0].reshape(-1); ti = cor[:, :, 1].reshape(-1)
            pos = V[vi]
            uvr = np.where(ti[:, None] >= 0, VT[np.clip(ti, 0, len(VT) - 1)], 0).astype(np.float32)
            sx, sy, ox, oy = sb.get('uv', [1, 1, 0, 0]); uvr = uvr * [sx, sy] + [ox, oy]   # BAKE Unity _ST tiling
            # V-FLIP: Unity UV origin is bottom-left; PNG rows + wgpu sampler are top-left. Baked here (textures
            # are referenced in place and can't be pre-flipped). manifest.conventions.uvVFlipBaked records it so
            # the loader does NOT re-flip. Applied AFTER tiling (texture-space flip, matches Unity tex2D fetch).
            uvr[:, 1] = 1.0 - uvr[:, 1]
            fn = np.cross(pos[1::3] - pos[0::3], pos[2::3] - pos[0::3]); fnr = np.repeat(fn, 3, 0)
            key = np.concatenate([np.round(pos, 3), np.round(uvr, 3)], 1)
            _, idx0, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
            nv = int(inv.max()) + 1
            nrm = np.zeros((nv, 3)); np.add.at(nrm, inv, fnr)
            ln = np.linalg.norm(nrm, axis=1, keepdims=True); nrm = (nrm / np.where(ln > 0, ln, 1)).astype(np.float32)
            # COLOR_0 = vert-paint blend weights (do NOT collapse white/unpainted). Non-vp -> opaque white.
            if sb.get('vp'):
                if vcol is not None and len(vcol) == len(V):
                    cc = vcol[vi][idx0].astype(np.float32)
                else:
                    cc = np.zeros((len(idx0), 4), np.float32); cc[:, 0] = 1.0; cc[:, 3] = 1.0
                col8 = np.clip(np.rint(np.clip(cc, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
            else:
                col8 = np.full((len(idx0), 4), 255, np.uint8)
            matId = MF.get(sb)
            pending.append({"mat": matId, "pos": pos[idx0].astype(np.float32), "nrm": nrm,
                            "uv": uvr[idx0].astype(np.float32), "inv": inv.astype(np.uint32), "col": col8})
        if not pending: continue

        # pack this mesh's vertices + local indices; assign a meshId
        va_parts, idx_parts, submeshes = [], [], []
        base = 0; iloc = 0
        for p in pending:
            nverts = len(p["pos"])
            va = np.empty(nverts, VDT)
            va["pos"] = p["pos"]; va["nrm"] = p["nrm"]; va["uv"] = p["uv"]; va["col"] = p["col"]
            va_parts.append(va)
            idx = p["inv"] + base
            idx_parts.append(idx)
            submeshes.append({"materialId": int(p["mat"]), "idxStart": int(iloc), "idxCount": int(len(idx))})
            base += nverts; iloc += len(idx)
        mesh_va = np.concatenate(va_parts)
        mesh_idx = np.concatenate(idx_parts).astype('<u4')
        meshId = len(meshes_meta)
        vtx_off = len(vbuf); vbuf += mesh_va.tobytes()
        idx_off_local = len(ibuf); ibuf += mesh_idx.tobytes()
        meshes_meta.append({"id": meshId, "name": mname.split('__')[0],
                            "vtxOffset": vtx_off, "vtxCount": int(base),
                            "_idxLocal": idx_off_local, "idxCount": int(len(mesh_idx)),
                            "submeshes": submeshes})
        utris += len(mesh_idx) // 3

        # local bbox corners for conservative world bounds
        allpos = np.concatenate([p["pos"] for p in pending])
        corners = _corners(allpos.min(0), allpos.max(0))
        # prim_raw for the degenerate bake fallback (matId, pos, nrm, uv, tri-index Nx3)
        prim_raw = [(p["mat"], p["pos"], p["nrm"], p["uv"], p["inv"].reshape(-1, 3)) for p in pending]

        # ---- STEP 7: per-instance emit (the CENTRAL divergence) ----------------------------------------------
        for it in by_mesh[mkey]:
            mg = apply_global(it['m'])                     # conjugated row-major 16 (verbatim). NO TRS-decompose.
            M3, T = _M3T(mg)
            if _degenerate(M3):
                # rank-deficient 3x3 (flattened plane) -> no invertible normal transform -> bake to world
                # (instmath.bake_into, pinv branch). This is the ONLY case that bakes.
                bake_into(baked, prim_raw, mg); n_baked += 1; continue
            flags = 0
            if det3(mg) < 0.0: flags |= FLAG_MIRROR         # renderer flips winding; we do NOT bake
            if is_terrain: flags |= FLAG_TERRAIN
            L = it.get('lod'); lg, li = (L['g'], L['i']) if L else (-1, -1)
            inst_records.append((list(mg[:12]), meshId, int(lg), int(li), rid(it.get('root') or ''), flags))
            upd_bounds(corners @ M3.T + T)

        if gi % 2000 == 0:
            print(f"[bevy]   {gi}/{len(groups)} groups  utris={utris/1e6:.1f}M  "
                  f"vbuf={len(vbuf)/1e6:.0f}MB ({time.time()-t0:.0f}s)")

    # ---- STEP 8: degenerate baked-world geometry -> one mesh + one identity instance -------------------------
    if baked:
        va_parts, idx_parts, submeshes = [], [], []
        base = 0; iloc = 0
        for matId, b in baked.items():
            pos = np.concatenate(b['pos']); nrm = np.concatenate(b['nrm'])
            uv = np.concatenate(b['uv']); idx = np.concatenate(b['idx']).reshape(-1)
            va = np.empty(len(pos), VDT)
            va["pos"] = pos.astype(np.float32); va["nrm"] = nrm.astype(np.float32)
            va["uv"] = uv.astype(np.float32); va["col"] = 255            # baked decals/billboards carry no vert-paint
            va_parts.append(va)
            idx_parts.append(idx.astype('<u4') + base)
            submeshes.append({"materialId": int(matId), "idxStart": int(iloc), "idxCount": int(len(idx))})
            base += len(pos); iloc += len(idx)
            upd_bounds(pos)
        mesh_va = np.concatenate(va_parts); mesh_idx = np.concatenate(idx_parts).astype('<u4')
        meshId = len(meshes_meta)
        vtx_off = len(vbuf); vbuf += mesh_va.tobytes()
        idx_off_local = len(ibuf); ibuf += mesh_idx.tobytes()
        meshes_meta.append({"id": meshId, "name": "baked_world",
                            "vtxOffset": vtx_off, "vtxCount": int(base),
                            "_idxLocal": idx_off_local, "idxCount": int(len(mesh_idx)),
                            "submeshes": submeshes})
        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        inst_records.append((identity, meshId, -1, -1, 0, FLAG_BAKED))
        utris += len(mesh_idx) // 3
        print(f"[bevy] degenerate fallback: baked {n_baked} rank-deficient instances -> 1 world mesh "
              f"({len(submeshes)} submeshes)")

    # ---- patch idxOffset (absolute into meshes.bin = after the whole vertex section) and write meshes.bin ----
    vlen = len(vbuf)
    for m in meshes_meta:
        m["idxOffset"] = vlen + m.pop("_idxLocal")
    with open(os.path.join(OUT, 'meshes.bin'), 'wb') as fh:
        fh.write(vbuf); fh.write(ibuf)

    # ---- write instances.bin ---------------------------------------------------------------------------------
    ia = np.zeros(len(inst_records), IDT)
    for i, (aff, mid, lg, li, rt, fl) in enumerate(inst_records):
        ia['affine'][i] = aff; ia['meshId'][i] = mid; ia['lodGroup'][i] = lg
        ia['lodIndex'][i] = li; ia['rootId'][i] = rt; ia['flags'][i] = fl
    with open(os.path.join(OUT, 'instances.bin'), 'wb') as fh:
        fh.write(ia.tobytes())

    # ---- materials.json --------------------------------------------------------------------------------------
    shipper = _PackShipper(OUT) if SELF_CONTAINED else None
    if shipper:
        _self_contain_materials(MF.records, shipper)
        if shipper.missing:
            print(f"[bevy] self-contained: {len(shipper.missing)} referenced textures MISSING on disk "
                  f"(kept as tex/<name>; loader falls back same as for a missing absolute path)")
    json.dump(MF.records, open(os.path.join(OUT, 'materials.json'), 'w'), separators=(',', ':'))

    # ---- LOD groups (conjugated centers) for runtime screen-height LOD ---------------------------------------
    lod_groups = []
    for grp in scene.get('lodGroups', []):
        c = np.array(grp.get('center', [0, 0, 0]), np.float64)
        if not GID: c = G3 @ c
        g2 = dict(grp); g2['center'] = [round(float(v), 4) for v in c]
        lod_groups.append(g2)

    # ---- sidecars: referenced IN PLACE by default; COPIED INTO THE PACK with --self-contained ----------------
    beamng = os.path.dirname(os.path.dirname(DS))           # .../beamng_blender_pipeline
    vol_dir = os.path.join(beamng, 'tarkmap', 'out', MAP)
    def _abs(p): return p.replace('\\', '/') if p and os.path.exists(p) else None
    lights = sorted(g for g in glob.glob(os.path.join(DS, 'lights_*.json')) if not g.endswith('_all.json'))
    lights_primary = next((l for l in lights if os.path.basename(l) == 'lights_64.json'), (lights[0] if lights else None))
    if not shipper:
        sidecars = {
            "terrainLayers": _abs(os.path.join(DS, 'terrain_layers', 'manifest.json')),
            "lights":        _abs(lights_primary or ''),
            "volume":        _abs(os.path.join(vol_dir, 'volume.bin')),
            "semantics":     None,                          # roots table embedded in manifest.roots instead
            # extras (self-describing; the loader reads the SH layout from volume.json):
            "volumeMeta":    _abs(os.path.join(vol_dir, 'volume.json')),
            "volumeVis":     _abs(os.path.join(vol_dir, 'volume.vis.bin')),
            "lightsAll":     [p.replace('\\', '/') for p in lights],
            "grassJson":     _abs(os.path.join(DS, 'terrain_layers', 'grass.json')),
        }
    else:
        # SELF-CONTAINED: ship the whole terrain_layers dir (ctrl/layer PNGs, density bins,
        # grass.json, its manifest -- build_grass reads density from the sidecar's dir, so a
        # shipped pack can rebuild grass), the volume triple and the lights jsons; reference
        # everything pack-relative. Missing sources -> null, same as the legacy _abs contract.
        shipper.ship_dir(os.path.join(DS, 'terrain_layers'), 'terrain_layers')
        tl_rel = 'terrain_layers/manifest.json'
        if os.path.exists(os.path.join(OUT, 'terrain_layers', 'manifest.json')):
            _relativize_tl_manifest(os.path.join(OUT, 'terrain_layers', 'manifest.json'))
        else:
            tl_rel = None
        lights_rel = [r for r in (shipper.ship(p, os.path.basename(p)) for p in lights) if r]
        sidecars = {
            "terrainLayers": tl_rel,
            "lights":        (os.path.basename(lights_primary)
                              if lights_primary and os.path.exists(lights_primary) else None),
            "volume":        shipper.ship(os.path.join(vol_dir, 'volume.bin'), 'volume.bin'),
            "semantics":     None,                          # roots table embedded in manifest.roots instead
            "volumeMeta":    shipper.ship(os.path.join(vol_dir, 'volume.json'), 'volume.json'),
            "volumeVis":     shipper.ship(os.path.join(vol_dir, 'volume.vis.bin'), 'volume.vis.bin'),
            "lightsAll":     lights_rel,
            "grassJson":     ('terrain_layers/grass.json'
                              if os.path.exists(os.path.join(OUT, 'terrain_layers', 'grass.json')) else None),
        }

    manifest = {
        "version": 1,
        "dataset": os.path.basename(DS),
        "datasetPath": DS.replace('\\', '/'),
        "map": MAP,
        "bounds": [round(float(x), 4) for x in (list(wmin) + list(wmax))],
        "vertex": {"stride": VDT.itemsize, "attrs": VERTEX_ATTRS},
        "instance": {"stride": IDT.itemsize, "fields": INSTANCE_FIELDS,
                     "align16": True, "note": "stride padded to 16B for the storage-buffer cull/draw path"},
        "meshes": meshes_meta,
        "instanceCount": len(inst_records),
        "materialCount": len(MF.records),
        "roots": root_names,
        "lodGroups": lod_groups,
        "flagsLegend": {"0x1": "MIRROR (det<0: flip front-face/winding)",
                        "0x2": "TERRAIN (MicroSplat splat shader)",
                        "0x4": "BAKED_WORLD (identity affine, geometry pre-baked)"},
        "conventions": {
            "affine": "ROW-MAJOR world 3x4 incl shear (glam Affine3A / raw instance buffer is shear+mirror correct)",
            "normals": "LOCAL smooth normals; renderer applies per-instance inverse-transpose of the 3x3 (shear-correct)",
            "uvVFlipBaked": True,     "uvOrigin": "top-left",
            "uvTilingBaked": True,    "uvXformNote": "materials.json.uvXform is REFERENCE ONLY; tiling already baked into vertex UV",
            "normalMapGreenFlip": True, "normalMapConvention": "directx",
            "colorSpace": {"albedo": "srgb", "normal": "linear", "emissive": "srgb"},
            "textureImport": "BC7 (albedo/emissive sRGB), BC5 (normal, linear); referenced in place, imported on load",
        },
        "sidecars": sidecars,
        "note": "web-lossy tail dropped (no 512 downscale / KTX2 / meshopt / quantize / split_glb / TRS split)",
    }
    if SELF_CONTAINED:
        # datasetPath above stays ABSOLUTE deliberately (build provenance only): the loader
        # never resolves textures/sidecars through it -- every consumer path is pack-relative.
        manifest["selfContained"] = True
    # Non-finite floats (the game data itself ships some - e.g. a Reserve LODGroup with
    # fadeTransitionWidth=NaN) must not kill a 20-minute build at the very last step: sanitize to
    # 0.0 and REPORT each path loudly so a real data bug stays visible. allow_nan=False stays as
    # the final backstop (serde_json rejects non-finite numbers, so a miss would brick the pack).
    _nonfinite = []
    def _sane(v, path):
        if isinstance(v, float) and not math.isfinite(v):
            _nonfinite.append(path)
            return 0.0
        if isinstance(v, dict):
            return {k: _sane(x, f"{path}.{k}") for k, x in v.items()}
        if isinstance(v, list):
            return [_sane(x, f"{path}[{i}]") for i, x in enumerate(v)]
        return v
    manifest = _sane(manifest, "manifest")
    if _nonfinite:
        print(f"[bevy] WARNING: sanitized {len(_nonfinite)} non-finite float(s) in the manifest -> 0.0: "
              + ", ".join(_nonfinite[:8]) + (" ..." if len(_nonfinite) > 8 else ""), flush=True)
    json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1, allow_nan=False)

    # ---- GLOBAL sidecars: the all-maps catalogs (tarkov.dev loot/tasks) + the game grade LUT are
    #      map-AGNOSTIC, so they live ONCE in packs/shared/ (above the packs; the viewer resolves
    #      pack-local -> shared -> cwd). Refreshed here when the upstream copy is newer. Per-map
    #      sidecars (gamedata/semantics/grass/volume) still have their own steps.
    tk_out = os.path.join(beamng, 'tarkmap', 'out')
    shared = os.path.join(os.path.dirname(FINAL_OUT), 'shared')
    os.makedirs(shared, exist_ok=True)
    for src, dst in ((os.path.join(tk_out, 'loot.json'), 'loot.json'),
                     (os.path.join(tk_out, 'tasks.json'), 'tasks.json'),
                     (os.path.join(tk_out, 'eft_grade_lut.bin'), 'grade_lut.bin')):
        tgt = os.path.join(shared, dst)
        if os.path.exists(src) and (
            not os.path.exists(tgt) or os.path.getmtime(src) > os.path.getmtime(tgt)
        ):
            shutil.copy2(src, tgt)
            print(f"[bevy] shared sidecar: {dst} <- {src}")
        elif not os.path.exists(tgt):
            print(f"[bevy] shared sidecar MISSING: {dst} (no {src}) — the viewer loses that layer")
    print("[bevy] remaining per-map steps: extract_semantics.py -> semantics.json; SH bake -> volume; build_grass")

    mb = lambda f: os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
    print(f"\n[EFTPACK] {OUT}")
    print(f"  meshes.bin    = {mb(os.path.join(OUT,'meshes.bin')):.0f} MB  "
          f"({len(meshes_meta):,} meshes, {utris/1e6:.1f}M unique tris)")
    print(f"  instances.bin = {mb(os.path.join(OUT,'instances.bin')):.1f} MB  ({len(inst_records):,} instances)")
    print(f"  materials.json= {len(MF.records):,} materials   roots={len(root_names):,}   "
          f"bounds={manifest['bounds']}")
    if shipper:
        print(f"[bevy] SELF-CONTAINED: copied {shipper.files} files (+{shipper.bytes/1e6:.1f} MB) into the pack "
              f"(tex/ + sidecars); {len(shipper.missing)} referenced textures missing")
    # ---- atomic swap: migrate per-map sidecars the build doesn't regenerate (semantics.json,
    #      grass.bin/grass_sidecar.json, and any loot/tasks/grade already in the live pack), then
    #      retire the old dir and move the staging dir into place. ----
    if os.path.abspath(FINAL_OUT) != os.path.abspath(OUT):
        old_dir = FINAL_OUT + '.old'
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
        if os.path.exists(FINAL_OUT):
            for fn in os.listdir(FINAL_OUT):
                if not os.path.exists(os.path.join(OUT, fn)):
                    shutil.move(os.path.join(FINAL_OUT, fn), os.path.join(OUT, fn))
            os.rename(FINAL_OUT, old_dir)
        os.rename(OUT, FINAL_OUT)
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
        print(f"[bevy] pack swapped into place: {FINAL_OUT}")
    print(f"[bevy] done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
