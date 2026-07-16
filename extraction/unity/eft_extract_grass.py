"""Extract EFT's terrain grass: the DETERMINISTIC per-slice DENSITY grids (the real, authoritative,
road-excluding placement), the grass billboard textures, and the authored WavingGrass params
-> eft_assets/<name>/terrain_layers/{grass_density_<slice>.bin, grass_<Tex>.png, grass.json}.
Consumed by the pack grass step (eft_pipeline/build_grass.py), which places grass FROM the grids.

EFT's grass is NOT Unity's native terrain detail (that DB is a deliberately-ZEROED decoy:
m_DetailPrototypes=[], 16384 empty patches, m_DrawTreesAndFoliage=False). It uses the GPU Instancer
plugin: each terrain slice's "GPUI Detail Manager (Slice_X_Y)" GameObject (in level<lv>) references
~12-22 GPUInstancerDetailPrototype MonoBehaviours (in sharedassets<lv>).

DENSITY LAYOUT (verified on interchange lv63 + lighthouse lv200): each prototype MB serializes an
int32[side*side] instance-count grid (side=512 on both; values ~0..16 per cell). We locate it by its
aligned int32 count field (side*side for a plausible side), NOT by a fixed byte window - the array
offset varies with the prototype's name length, and the file also carries ~136 tail bytes of float
params after the grid. (A previous version read "the last 1MiB as 1024x1024 uint8", which sheared
every row across two byte-rows and x-stretched the placement - do not regress to that.)

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

# portable kit: paths come from the environment (see extraction/README.md)
#   EFT_GAME_DATA   = the game's EscapeFromTarkov_Data dir (default: standard install path)
#   EFT_ASSETS_ROOT = where extracted datasets are written (default: <EFT_TARKMAP_ROOT>/../eft_assets, else ./eft_assets)
EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else
    os.path.join(os.getcwd(), "eft_assets"))

# plausible detail-grid resolutions (side of the square int32 array). Both extracted maps use 512;
# other resolutions are accepted so a future map with a denser grid extracts instead of vanishing.
_SIDES = (256, 512, 1024, 2048)
_MAX_TAIL = 4096          # bytes of trailing fields allowed after the density array
_MAX_CELL_VALUE = 65535   # sanity ceiling for per-cell instance counts


def _find_density_grid(raw):
    """Locate the serialized int32 density array in a GPUInstancerDetailPrototype's raw MB bytes.
    Returns (side, int32 ndarray[side,side]) or None. Anchors on the aligned int32 COUNT field
    (side*side) whose array fits the remaining bytes with only a small tail, and whose values are
    sane non-negative instance counts. Ambiguity (2+ candidates) is rejected loudly."""
    good = []
    for side in _SIDES:
        cnt = side * side
        if len(raw) < 4 + cnt * 4:
            continue
        pat = struct.pack("<i", cnt)
        off = raw.find(pat)
        while off != -1:
            if off % 4 == 0 and off + 4 + cnt * 4 <= len(raw):
                tail = len(raw) - (off + 4 + cnt * 4)
                if tail <= _MAX_TAIL:
                    arr = np.frombuffer(raw, "<i4", count=cnt, offset=off + 4)
                    if arr.min() >= 0 and arr.max() <= _MAX_CELL_VALUE:
                        good.append((side, arr))
            off = raw.find(pat, off + 4)
    if len(good) != 1:
        if len(good) > 1:
            print(f"  density grid: {len(good)} candidate arrays in one MB - AMBIGUOUS, skipped")
        return None
    side, arr = good[0]
    return side, arr.reshape(side, side)


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
        if not (4 <= n <= 64) or off + 4 + n * 12 > L:
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


def extract_grass_density(data_root, lv, out_dir):
    """Per-slice combined grass density grids from GPU Instancer. Returns {slice_name: {dims, nonzero}}."""
    sa = UnityPy.load(os.path.join(data_root, f"sharedassets{lv}.assets"))
    proto = {}
    proto_side = {}
    for o in sa.objects:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            raw = o.get_raw_data()
        except Exception:
            continue
        if len(raw) < 4 + min(_SIDES) ** 2 * 4:      # too small to hold any density grid
            continue
        found = _find_density_grid(bytes(raw))
        if found is None:
            continue
        proto_side[o.path_id], proto[o.path_id] = found
    if not proto:
        print(f"grass density: no GPU Instancer detail prototypes in sharedassets{lv} — skip")
        return {}
    sides = sorted(set(proto_side.values()))
    print(f"grass density: {len(proto)} detail prototypes in sharedassets{lv} (grid side {sides})")

    lvl = UnityPy.load(os.path.join(data_root, f"level{lv}"))
    objmap = {o.path_id: o for o in lvl.objects}
    valid_fids = _sharedassets_fids(lvl, lv)
    if valid_fids is None:
        print("  WARNING: could not resolve the level's externals table - "
              "accepting any external fileID for prototype PPtrs")
    quick_pats = [struct.pack("<q", p) for p in proto]
    slice_pids = {}                                  # slice_name -> set of prototype path_ids
    for o in lvl.objects:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            raw = bytes(o.get_raw_data())
        except Exception:
            continue
        if len(raw) > 200000:                        # managers are small
            continue
        if sum(p in raw for p in quick_pats) < 3:    # cheap prefilter before the strict scan
            continue
        pl = _pptr_list(raw, valid_fids, proto)
        if len(pl) < 4:                              # a detail manager references >=~12 prototypes
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
    for slice_name, pids in sorted(slice_pids.items()):
        ss = {proto_side[p] for p in pids}
        if len(ss) != 1:
            print(f"  grass density {slice_name}: MIXED grid sides {sorted(ss)} - skipped")
            continue
        side = ss.pop()
        acc = np.zeros((side, side), np.uint32)
        for p in sorted(pids):                       # SUM instance counts across this slice's detail types
            acc += proto[p].astype(np.uint32)
        grid = np.clip(acc, 0, 255).astype(np.uint8)
        grid.tofile(os.path.join(out_dir, f"grass_density_{slice_name}.bin"))
        result[slice_name] = {"dims": [side, side], "nonzero": round(float((grid > 0).mean()), 4)}
        print(f"  grass density {slice_name}: {len(pids)} prototypes, "
              f"{result[slice_name]['nonzero']*100:.1f}% cells, max {int(acc.max())}")
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
    # GRASS BILLBOARD TEXTURES: EFT's terrain grass cards (Grass3_D = tall rye clump, Grass5_512_D = seed-head
    # row). Real alpha-cutout atlases the viewer scatters as billboards. Export every grass-named Texture2D that
    # HAS a real alpha channel (the *_D albedo cards; skip *_N normals). Sidecar-published like the splat layers.
    ntex = 0
    for obj in env.objects:
        if obj.type.name != "Texture2D":
            continue
        try:
            d = obj.read(); nm = d.m_Name or ""
            if "grass" not in nm.lower() or nm.lower().endswith("_n"):
                continue
            img = d.image
            if "A" not in img.getbands():
                continue
            lo, hi = img.getchannel("A").getextrema()
            if hi - lo < 32:                                   # no real cutout -> not a billboard card
                continue
            img.save(os.path.join(out, "grass_" + nm + ".png")); ntex += 1
        except Exception:
            pass
    print(f"grass billboard textures exported: {ntex}")
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
    density = extract_grass_density(args.data_root, args.level, out)
    fp = os.path.join(out, "grass.json")
    json.dump({"slices": slices, "density": density}, open(fp, "w"), indent=1)
    print(f"wrote {fp}: {len(slices)} slice(s), {len(density)} density grid(s)")


if __name__ == "__main__":
    main()
