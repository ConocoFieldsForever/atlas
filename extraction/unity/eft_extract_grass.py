"""Extract EFT's terrain grass: the DETERMINISTIC per-slice DENSITY grids (the real, authoritative,
road-excluding placement), the grass billboard textures, and the authored WavingGrass params
-> eft_assets/<name>/terrain_layers/{grass_density_<slice>.bin, grass_<Tex>.png, grass.json}.
Consumed by the viewer (out/_grassblades.js), which places grass FROM the density grids.

EFT's grass is NOT Unity's native terrain detail (that DB is a deliberately-ZEROED decoy:
m_DetailPrototypes=[], 16384 empty patches, m_DrawTreesAndFoliage=False). It uses the GPU Instancer
plugin: each terrain slice's "GPUI Detail Manager (Slice_X_Y)" GameObject (in level<lv>) references
~22 GPUInstancerDetailPrototype MonoBehaviours (in sharedassets<lv>) whose LAST 1,048,576 raw bytes
are a 1024x1024 uint8 density grid (instance count per terrain cell; ~6-15% nonzero, roads/buildings
zeroed). Type trees are stripped, so we raw-parse. See the tarkov-unity-extraction skill.

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
_G = 1024 * 1024


def extract_grass_density(data_root, lv, out_dir):
    """Per-slice combined grass density grids from GPU Instancer. Returns {slice_name: {dims, nonzero}}."""
    sa = UnityPy.load(os.path.join(data_root, f"sharedassets{lv}.assets"))
    proto = {}
    for o in sa.objects:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            raw = o.get_raw_data()
        except Exception:
            continue
        if 1049000 <= len(raw) <= 1049600:                       # GPUInstancerDetailPrototype: header + 1024^2 uint8 density
            proto[o.path_id] = np.frombuffer(raw[-_G:], dtype=np.uint8).reshape(1024, 1024)
    if not proto:
        print(f"grass density: no GPU Instancer detail prototypes in sharedassets{lv} — skip")
        return {}
    lvl = UnityPy.load(os.path.join(data_root, f"level{lv}"))
    objmap = {o.path_id: o for o in lvl.objects}
    result = {}
    combined = {}                                                # slice_name -> uint16 accumulator (dedup managers, e.g. -OPTIC)
    for o in lvl.objects:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            raw = o.get_raw_data()
        except Exception:
            continue
        if len(raw) > 200000:                                    # managers are small
            continue
        hits = [p for p in proto if struct.pack("<q", p) in raw]
        if len(hits) < 8:                                        # a detail manager references ~22 prototypes
            continue
        go_pid = struct.unpack("<q", raw[4:12])[0]               # MonoBehaviour.m_GameObject pathID
        go = objmap.get(go_pid)
        nm = ""
        try:
            nm = go.read().m_Name if go else ""
        except Exception:
            pass
        m = re.search(r"Slice_\d+_\d+", nm or "")
        if not m:
            continue
        acc = combined.setdefault(m.group(0), np.zeros((1024, 1024), np.uint16))
        for p in hits:
            acc += proto[p]                                      # SUM instance counts across this slice's detail types
    for slice_name, acc in combined.items():
        grid = np.clip(acc, 0, 255).astype(np.uint8)
        grid.tofile(os.path.join(out_dir, f"grass_density_{slice_name}.bin"))
        result[slice_name] = {"dims": [1024, 1024], "nonzero": round(float((grid > 0).mean()), 4)}
        print(f"  grass density {slice_name}: {result[slice_name]['nonzero']*100:.1f}% cells, max {int(acc.max())}")
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
    ap.add_argument("--level", type=int, help="terrain level index (Interchange=63); omit with --levels to auto-detect")
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
