"""Extract EFT's terrain grass: the DETERMINISTIC per-slice DENSITY grids (the real, authoritative,
road-excluding placement), the grass billboard textures, and the authored WavingGrass params
-> eft_assets/<name>/terrain_layers/{grass_density_<slice>.bin, grass_<Tex>.png, grass.json}.
Consumed by the pack grass step (eft_pipeline/build_grass.py), which places grass FROM the grids.

EFT's grass is NOT Unity's native terrain detail (that DB is a deliberately-ZEROED decoy:
m_DetailPrototypes=[], 16384 empty patches, m_DrawTreesAndFoliage=False). It uses the GPU Instancer
plugin: each terrain slice's "GPUI Detail Manager (Slice_X_Y)" GameObject (in level<lv>) references
30-50 GPUInstancerDetailPrototype MonoBehaviours (in sharedassets<lv>), each a different plant.

IDENTIFICATION IS BY CLASS, NOT BY SHAPE. Components are found by resolving m_Script -> MonoScript
-> "GPUInstancer.GPUInstancerDetailPrototype" / "...DetailManager". MonoScript is an ENGINE type
with a hardcoded type tree, so this works even though il2cpp global-metadata.dat is encrypted
(docs/IL2CPP.md) — no decryption, no process contact. Do NOT go back to sniffing payloads.

DENSITY LAYOUT: each prototype MB serializes an int32[side*side] instance-count grid (values
0..16 per cell). We locate it by its aligned int32 count field, NOT by a fixed byte window - the
array offset varies with the prototype's name length, and the file also carries ~136 tail bytes of
float params after the grid. The SIDE IS DERIVED from the count (any perfect square); it is
map-specific - 320 and 512 on interchange, 384 and 576 on woods, 448/640 on customs. A previous
version tested a hardcoded whitelist of "plausible" sides and so extracted ZERO grass from woods,
which carries 186 real prototypes. (An older version still read "the last 1MiB as 1024x1024 uint8",
which sheared every row across two byte-rows and x-stretched the placement - do not regress.)

MANAGER -> PROTOTYPE references are parsed as a strict serialized PPtr array (int32 count, then
count x {int32 fileID, int64 pathID}) with fileID validated against the level file's externals
table (must point at sharedassets<lv>.assets) and pathID validated against the prototype set.
(A previous version substring-searched the raw bytes for the 8-byte pathID, which false-positived
on small path_ids like 62/256 and summed OTHER slices' grids into each slice.)

Output grids are dumped in GAME ROW ORDER: row = terrain-local Z cell index, col = terrain-local X
cell index (Unity detail [z][x] convention), as side*side uint8 (sum of all prototypes, clipped).
The consumer maps cells onto the pack terrain mesh UVs, whose v axis runs OPPOSITE the grid rows
(our terrain mesher writes image-frame v = 1 - z_frac), i.e. it samples v = 1-(row+.5)/side.
Type trees are stripped, so we raw-parse. See the tarkov-unity-extraction skill.

    python extraction/unity/eft_extract_grass.py --level 63 --name interchange_v2   (or --levels a,b,c to auto-detect)
"""
import os, json, argparse, struct, re
import numpy as np
import UnityPy

# portable kit: paths come from the environment (see README.md)
#   EFT_GAME_DATA   = the game's EscapeFromTarkov_Data dir (default: standard install path)
#   EFT_ASSETS_ROOT = where extracted datasets are written (default: <EFT_TARKMAP_ROOT>/../eft_assets, else ./eft_assets)
EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else
    os.path.join(os.getcwd(), "eft_assets"))

_MAX_TAIL = 4096          # bytes of trailing fields allowed after the density array
_MAX_CELL_VALUE = 65535   # sanity ceiling for per-cell instance counts
_MIN_SIDE = 32            # below this a "square array" is more likely a coincidence than a grid


def _find_density_grid(raw):
    """Locate the serialized int32 density array in a GPUInstancerDetailPrototype's raw MB bytes.

    Returns (side, int32 ndarray[side,side]) or None. Anchors on an aligned int32 COUNT field
    whose array fills the rest of the payload bar a small tail of trailing fields, whose count is
    a PERFECT SQUARE, and whose values are sane non-negative instance counts.

    NO whitelist of resolutions. This used to test a hardcoded `_SIDES` tuple of "plausible"
    sides (256/448/512/640/1024/2048) — so Woods, whose grids are 384^2 and 576^2, matched
    nothing and silently extracted ZERO grass while carrying 186 real detail prototypes. The side
    is DERIVED from the array length instead; callers must not assume any particular resolution.
    """
    good, L = [], len(raw)
    for off in range(0, max(0, L - 8), 4):
        cnt = struct.unpack_from("<i", raw, off)[0]
        if cnt < _MIN_SIDE * _MIN_SIDE or off + 4 + cnt * 4 > L:
            continue
        if L - (off + 4 + cnt * 4) > _MAX_TAIL:
            continue
        side = int(round(cnt ** 0.5))
        if side * side != cnt:
            continue
        arr = np.frombuffer(raw, "<i4", count=cnt, offset=off + 4)
        if arr.min() >= 0 and arr.max() <= _MAX_CELL_VALUE:
            good.append((side, arr))
    if len(good) != 1:
        if len(good) > 1:
            print(f"  density grid: {len(good)} candidate arrays in one MB - AMBIGUOUS, skipped")
        return None
    side, arr = good[0]
    return side, arr.reshape(side, side)


# The component classes we care about, by their REAL C# names. Resolved from each
# MonoBehaviour's m_Script -> MonoScript, which is an ENGINE type with a hardcoded type tree —
# so this works with the il2cpp metadata fully encrypted (docs/IL2CPP.md). Identifying components
# by class instead of by "payload looks like X" removes the last discovery guess: a prototype is
# a prototype because the game says so, not because its bytes resembled a grid.
CLS_PROTOTYPE = "GPUInstancer.GPUInstancerDetailPrototype"
CLS_MANAGER = "GPUInstancer.GPUInstancerDetailManager"


class _ScriptClasses:
    """MonoBehaviour raw bytes -> "Namespace.ClassName" (or None)."""

    def __init__(self, data_root, env):
        self.data_root = data_root
        sf = list(env.files.values())[0]
        self.externals = [e.path for e in sf.externals]
        self.local = {o.path_id: o for o in env.objects}
        self._files = {}
        self._names = {}

    def _table(self, fid):
        if fid == 0:
            return self.local
        if not (1 <= fid <= len(self.externals)):
            return {}
        key = self.externals[fid - 1]
        if key not in self._files:
            p = os.path.join(self.data_root, os.path.basename(key))
            try:
                self._files[key] = {o.path_id: o for o in UnityPy.load(p).objects} \
                    if os.path.exists(p) else {}
            except Exception:
                self._files[key] = {}
        return self._files[key]

    def of(self, raw):
        # m_Script PPtr @16: m_GameObject(12) + m_Enabled(1 + 3 pad)
        if len(raw) < 28:
            return None
        fid, pid = struct.unpack_from("<iq", raw, 16)
        if (fid, pid) in self._names:
            return self._names[(fid, pid)]
        name = None
        try:
            o = self._table(fid).get(pid)
            if o is not None and o.type.name == "MonoScript":
                d = o.read_typetree()
                ns = d.get("m_Namespace") or ""
                name = f"{ns + '.' if ns else ''}{d.get('m_ClassName') or ''}"
        except Exception:
            name = None
        self._names[(fid, pid)] = name
        return name


def _sharedassets_fids(level_env, lv):
    """fileID values that, from level<lv>, reference sharedassets<lv>.assets (externals are 1-based;
    fileID 0 is the level file itself). Returns a set. A READABLE table with no sharedassets<lv>
    entry is a hard error (the detail managers MUST reference it - accepting any fileID would
    re-open the path_id-collision hole). Only an unreadable table (UnityPy API drift) fails open:
    returns None and the caller accepts any fileID >= 1 with a warning."""
    want = f"sharedassets{lv}.assets"
    try:
        sf = next(iter(level_env.objects)).assets_file
        ext = getattr(sf, "externals", None) or getattr(sf, "m_Externals", None)
        names = [re.split(r"[\\/]", str(getattr(e, "path", getattr(e, "name", ""))))[-1].lower()
                 for e in ext]
    except Exception:
        return None
    fids = {i + 1 for i, nm in enumerate(names) if nm == want}
    if not fids:
        raise SystemExit(f"grass density: level{lv} externals table has no {want} entry "
                         f"(externals: {names}) - refusing to guess prototype references")
    return fids


def _pptr_list(raw, valid_fids, pids):
    """Strict parse of the manager's prototype list: int32 count N (4..64), then N x
    {int32 fileID, int64 pathID} where every fileID references sharedassets<lv> and every pathID
    is a known prototype. Returns the longest such run (the prototypeList field)."""
    best = []
    L = len(raw)
    for off in range(0, L - 16, 4):
        n = struct.unpack_from("<i", raw, off)[0]
        # >=1: the caller now identifies the manager by CLASS, so a slice legitimately referencing
        # only a couple of prototypes must still be accepted (the old >=4 floor was a proxy for
        # "this payload is probably a detail manager").
        if not (1 <= n <= 4096) or off + 4 + n * 12 > L:
            continue
        got = []
        for k in range(n):
            fid, pid = struct.unpack_from("<iq", raw, off + 4 + k * 12)
            if (valid_fids is not None and fid not in valid_fids) or \
               (valid_fids is None and fid < 1) or pid not in pids:
                got = None
                break
            got.append(pid)
        if got and len(got) > len(best):
            best = got
    return best


_PROTO_NAME_RE = re.compile(r"^Detail_\d+_(.+?)_[0-9a-fA-F]{6}$")


def _mb_name(raw):
    """MonoBehaviour m_Name: after m_GameObject(12) + m_Enabled(4) + m_Script(12) = @28."""
    try:
        n = struct.unpack_from("<i", raw, 28)[0]
        if 0 < n <= 200 and 32 + n <= len(raw):
            return raw[32:32 + n].decode("utf-8", "replace")
    except Exception:
        pass
    return ""


class _ExtResolver:
    """Resolves a prototype's texture PPtrs, which point into OTHER assets files.

    The grass billboard textures do NOT live in the terrain level's sharedassets bundle — on
    reserve they sit in sharedassets17/25, referenced through the level's externals table. The
    old exporter only scanned the terrain bundle for Texture2D objects whose NAME contained
    'grass', so every map whose textures live elsewhere (reserve, lighthouse) exported ZERO
    cards and silently lost its grass. Follow the actual reference instead of guessing names.
    """

    def __init__(self, data_root, env):
        self.data_root = data_root
        self.sf = list(env.files.values())[0]
        self.externals = [e.path for e in self.sf.externals]
        self.local = {o.path_id: o for o in env.objects}
        self._cache = {}

    def _ext(self, fid):
        """{path_id: obj} for external fileID (1-based into the externals table)."""
        if not (1 <= fid <= len(self.externals)):
            return None
        key = self.externals[fid - 1]
        if key not in self._cache:
            p = os.path.join(self.data_root, os.path.basename(key))
            try:
                self._cache[key] = {o.path_id: o for o in UnityPy.load(p).objects} \
                    if os.path.exists(p) else {}
            except Exception:
                self._cache[key] = {}
        return self._cache[key]

    def textures(self, raw, grid_off, proto_name):
        """Texture2D objects referenced by this prototype, best match first.

        Prefers the texture whose m_Name matches the one embedded in the prototype's own name
        ('Detail_7_Grass4_D_bf0a23' -> 'Grass4_D') — the game's own authored link, so it is a
        strict validation rather than a heuristic. Falls back to any referenced Texture2D that
        carries a real alpha cutout (a billboard card).
        """
        want = (_PROTO_NAME_RE.match(proto_name).group(1).lower()
                if _PROTO_NAME_RE.match(proto_name) else None)
        head, seen, hits = raw[:grid_off], set(), []
        for off in range(0, max(0, len(head) - 12), 4):
            fid, pid = struct.unpack_from("<iq", head, off)
            if pid <= 0 or pid > 10 ** 9 or (fid, pid) in seen:
                continue
            seen.add((fid, pid))
            tbl = self.local if fid == 0 else self._ext(fid)
            obj = (tbl or {}).get(pid)
            if obj is None or obj.type.name != "Texture2D":
                continue
            try:
                nm = obj.read().m_Name or ""
            except Exception:
                continue
            hits.append((nm, obj))
        if want:
            hits.sort(key=lambda h: h[0].lower() != want)
        return hits


def extract_grass_density(data_root, lv, out_dir):
    """Per-slice grass density from GPU Instancer.

    Writes, per slice: the COMBINED grid `grass_density_<Slice>.bin` (uint8, back-compat) AND
    the PER-PROTOTYPE stack `grass_protos_<Slice>.bin` (uint8[nproto][side][side]). The per-
    prototype grids are what carry the map's plant VARIETY: each detail type is a different
    plant with its own card (grass11, T_WhitGrass_A, Grass4_D, Grass_new_1_D...) and its own
    footprint. Summing them into one grid (the old behaviour) collapsed 12-30 species into a
    single repeated texture.

    Returns ({slice_name: {dims, nonzero, prototypes:[...]}} , exported_texture_count).
    """
    sa = UnityPy.load(os.path.join(data_root, f"sharedassets{lv}.assets"))
    res = _ExtResolver(data_root, sa)
    cls = _ScriptClasses(data_root, sa)
    proto = {}
    proto_side = {}
    proto_name = {}
    proto_tex = {}
    n_class, n_nogrid = 0, 0
    for o in sa.objects:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            raw = bytes(o.get_raw_data())
        except Exception:
            continue
        if cls.of(raw) != CLS_PROTOTYPE:             # identified by CLASS, not by payload shape
            continue
        n_class += 1
        found = _find_density_grid(raw)
        if found is None:
            n_nogrid += 1
            continue
        proto_side[o.path_id], proto[o.path_id] = found
        nm = _mb_name(raw)
        proto_name[o.path_id] = nm
        # grid offset: _find_density_grid anchored on the count field; recover it for the head slice
        side = found[0]
        goff = raw.find(struct.pack("<i", side * side))
        proto_tex[o.path_id] = res.textures(raw, goff if goff > 0 else 0, nm)
    if n_class and not proto:
        # LOUD: the game says these ARE detail prototypes, so a grid we cannot parse is a decoder
        # gap, not an absent feature. Silently returning {} here is what made Woods look grassless.
        print(f"grass density: {n_class} {CLS_PROTOTYPE} in sharedassets{lv} but NO density grid "
              f"could be parsed in any of them — decoder gap, NOT a grassless map")
        return {}
    if not proto:
        print(f"grass density: no {CLS_PROTOTYPE} components in sharedassets{lv} — skip")
        return {}
    sides = sorted(set(proto_side.values()))
    print(f"grass density: {len(proto)}/{n_class} detail prototypes carry a grid in "
          f"sharedassets{lv} (sides {sides}"
          + (f"; {n_nogrid} with no parsable grid" if n_nogrid else "") + ")")

    lvl = UnityPy.load(os.path.join(data_root, f"level{lv}"))
    lvl_cls = _ScriptClasses(data_root, lvl)
    objmap = {o.path_id: o for o in lvl.objects}
    valid_fids = _sharedassets_fids(lvl, lv)
    if valid_fids is None:
        print("  WARNING: could not resolve the level's externals table - "
              "accepting any external fileID for prototype PPtrs")
    # (prototype pathIDs are validated inside _pptr_list against `proto`)
    slice_pids = {}                                  # slice_name -> set of prototype path_ids
    for o in lvl.objects:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            raw = bytes(o.get_raw_data())
        except Exception:
            continue
        # Identify the manager by CLASS (was: "small payload that mentions >=3 prototype pathIDs
        # and parses as a >=4-entry PPtr run"). The strict PPtr parse below still validates the
        # list itself; this just stops the search from depending on how many prototypes a manager
        # happens to reference.
        if lvl_cls.of(raw) != CLS_MANAGER:
            continue
        pl = _pptr_list(raw, valid_fids, proto)
        if not pl:
            continue
        go_pid = struct.unpack("<q", raw[4:12])[0]   # MonoBehaviour.m_GameObject pathID
        go = objmap.get(go_pid)
        nm = ""
        try:
            nm = go.read().m_Name if go else ""
        except Exception:
            pass
        m = re.search(r"Slice_\d+_\d+", nm or "")
        if not m:
            continue
        # union across managers (e.g. -OPTIC duplicates) so no prototype is summed twice
        slice_pids.setdefault(m.group(0), set()).update(pl)

    result = {}
    exported = {}                                    # texture m_Name -> written filename
    for slice_name, pids in sorted(slice_pids.items()):
        ss = {proto_side[p] for p in pids}
        side = max(ss)
        if len(ss) != 1:
            # customs: one slice mixes 448^2 and 640^2 prototype grids. All grids of a slice share
            # the SAME normalized UV footprint (the consumer samples u=(cx+.5)/side, v=1-(cy+.5)/side),
            # so nearest-resample the coarser grids UP to the finest side and sum there. Upsampling
            # replicates counts (footprint-exact, deterministic); never downsample — that would smear
            # the road/building-excluding boundaries the grids exist to preserve.
            print(f"  grass density {slice_name}: MIXED grid sides {sorted(ss)} - nearest-resampled to {side}")
        acc = np.zeros((side, side), np.uint32)
        stack, protos = [], []
        for p in sorted(pids):                       # SUM instance counts across this slice's detail types
            g = proto[p].astype(np.uint32)
            if g.shape[0] != side:                   # nearest neighbour in the shared normalized UV space
                idx = (np.arange(side, dtype=np.int64) * g.shape[0]) // side
                g = g[np.ix_(idx, idx)]
            acc += g
            if not g.any():                          # a prototype with an all-zero grid places nothing
                continue
            # export this prototype's billboard card (following its OWN reference, not a name scan)
            texfile = ""
            for nm, obj in proto_tex.get(p, []):
                if nm in exported:
                    texfile = exported[nm]
                    break
                try:
                    img = obj.read().image
                    if "A" not in img.getbands():
                        continue
                    lo, hi = img.getchannel("A").getextrema()
                    if hi - lo < 32:                 # no real cutout -> not a billboard card
                        continue
                    texfile = "grass_" + nm + ".png"
                    img.save(os.path.join(out_dir, texfile))
                    exported[nm] = texfile
                    break
                except Exception:
                    continue
            if not texfile:
                print(f"    prototype {proto_name.get(p, p)!r}: no usable billboard texture — skipped")
                continue
            stack.append(np.clip(g, 0, 255).astype(np.uint8))
            protos.append({"name": proto_name.get(p, str(p)), "tex": texfile,
                           "cells": int((g > 0).sum()), "max": int(g.max())})
        grid = np.clip(acc, 0, 255).astype(np.uint8)
        grid.tofile(os.path.join(out_dir, f"grass_density_{slice_name}.bin"))
        if stack:
            np.stack(stack).tofile(os.path.join(out_dir, f"grass_protos_{slice_name}.bin"))
        result[slice_name] = {"dims": [side, side],
                              "nonzero": round(float((grid > 0).mean()), 4),
                              "prototypes": protos}
        print(f"  grass density {slice_name}: {len(pids)} prototypes "
              f"({len(protos)} with geometry+texture), "
              f"{result[slice_name]['nonzero']*100:.1f}% cells, max {int(acc.max())}/cell")
    print(f"grass billboard textures exported: {len(exported)} "
          f"({', '.join(sorted(exported)) if exported else 'none'})")
    return result


def _find_terrain_level(data_root, levels):
    """Return the first level whose sharedassets<level>.assets contains a TerrainData (the grass/terrain bundle)."""
    for lv in levels:
        pth = os.path.join(data_root, f"sharedassets{lv}.assets")
        if not os.path.exists(pth):
            continue
        try:
            env = UnityPy.load(pth)
            if any(o.type.name == "TerrainData" for o in env.objects):
                return lv
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, help="terrain level index (Interchange=63, Lighthouse=200); omit with --levels to auto-detect")
    ap.add_argument("--levels", help="comma list of levels to scan for the TerrainData bundle (pipeline pass-through)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-root", default=EFTDATA)
    args = ap.parse_args()

    if args.level is None:
        lvls = [int(x) for x in (args.levels or "").split(",") if x.strip()]
        args.level = _find_terrain_level(args.data_root, lvls)
        if args.level is None:
            print(f"grass: no TerrainData bundle among levels {lvls} — skip (interior/arena map)")
            return

    env = UnityPy.load(os.path.join(args.data_root, f"sharedassets{args.level}.assets"))
    out = os.path.join(OUTROOT, args.name, "terrain_layers")
    os.makedirs(out, exist_ok=True)
    # GRASS BILLBOARD TEXTURES are exported by extract_grass_density() below, which follows each
    # GPU-Instancer prototype's OWN texture reference (across the externals table into whatever
    # sharedassets bundle actually holds the card). The previous implementation scanned only this
    # terrain bundle for Texture2D objects whose NAME contained "grass" — a name guess that
    # exported 0 textures on every map storing them elsewhere (reserve, lighthouse), which then
    # emitted an empty sidecar albedo and silently disabled grass in the viewer.
    slices = {}
    for obj in env.objects:
        if obj.type.name != "TerrainData":
            continue
        d = obj.read_typetree()
        det = d.get("m_DetailDatabase", {})
        t = det.get("WavingGrassTint", {}) or {}
        slices[d.get("m_Name", f"td_{obj.path_id}")] = {
            "tint": [round(float(t.get(k, 1.0)), 4) for k in ("r", "g", "b")],
            "strength": round(float(det.get("m_WavingGrassStrength", 0.5)), 4),
            "amount": round(float(det.get("m_WavingGrassAmount", 0.15)), 4),
            "speed": round(float(det.get("m_WavingGrassSpeed", 0.5)), 4),
            "detail_prototypes": len(det.get("m_DetailPrototypes", []) or []),
        }
    if not slices:
        print(f"level{args.level}: no TerrainData in sharedassets{args.level}.assets — nothing written")
        return
    # DETERMINISTIC per-slice grass DENSITY grids (GPU Instancer) — the authoritative placement.
    # The detail prototypes do NOT always live in the same bundle as the TerrainData, so try the
    # terrain level first and then every other level the caller listed. (Woods really has none
    # anywhere — its ground cover is placed mesh geometry, not GPU-Instancer detail — but a map
    # that merely SPLITS them would otherwise silently extract zero grass.)
    density = extract_grass_density(args.data_root, args.level, out)
    if not density:
        others = [int(x) for x in (args.levels or "").split(",") if x.strip()]
        for lv in [l for l in others if l != args.level]:
            if not os.path.exists(os.path.join(args.data_root, f"sharedassets{lv}.assets")):
                continue
            print(f"grass density: retrying in sharedassets{lv} (prototypes may not share the "
                  f"TerrainData bundle)")
            density = extract_grass_density(args.data_root, lv, out)
            if density:
                break
    fp = os.path.join(out, "grass.json")
    json.dump({"slices": slices, "density": density}, open(fp, "w"), indent=1)
    print(f"wrote {fp}: {len(slices)} slice(s), {len(density)} density grid(s)")


if __name__ == "__main__":
    main()
