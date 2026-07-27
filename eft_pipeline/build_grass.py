#!/usr/bin/env python
"""#4 GRASS — deterministic placement from the GPU-Instancer density grids.

EFT grass is baked GPU-Instancer density (NOT Unity terrain detail — that DB is a zeroed
decoy). The extractor (extraction/unity/eft_extract_grass.py) dumps per-slice combined grids
(grass_density_Slice_*.bin, side^2 uint8; side=512 on interchange/lighthouse — the game's
int32[512*512] detail arrays summed over ~12-22 prototypes, road/building-excluding,
hand-authored). This emits one grass instance per non-empty cell (deterministic per-cell hash
for rotation/scale — NEVER client-random), placed on the terrain surface via a UV->world
bilinear lookup built from the pack's terrain meshes (so XZ AND height are exact and in pack
space).

GRID ORIENTATION: grids are dumped in GAME row order (Unity detail [row=z][col=x], terrain-
local). Our terrain meshes carry UVs in the Unity heightmap/splat IMAGE frame — u = x_frac,
but v = 1 - z_frac (validated: splat ctrl textures sampled with these UVs render correctly) —
so grid cell (col cx, row cy) samples the mesh at u=(cx+.5)/side, v=1-(cy+.5)/side.
Verified against road footprints + sea level on lighthouse AND interchange (the un-flipped
mapping drops 75% of lighthouse Slice_5_4's clumps into the sea).

Output: <pack>/grass.bin  = N records of [x,y,z, rotY, scale] f32 (20 B), pack space.
        <pack>/grass_sidecar.json = {count, albedo, tint}

  python -m eft_pipeline.build_grass --pack packs/interchange.eftpack [--self-contained]

SELF-CONTAINED packs (redistribution PR3): when the resolved albedo already lives INSIDE the
pack (terrain_layers/ was copied in by assemble_bevy --self-contained) the sidecar gets the
PACK-RELATIVE path; with --self-contained an outside albedo is COPIED into the pack as
grass_albedo.png. Otherwise the legacy absolute-path contract is unchanged (the viewer
resolves relative sidecar paths against the pack dir, absolute passes through).
"""
import os, sys, json, struct, argparse, re, glob, shutil
import numpy as np

FLAG_TERRAIN = 1 << 1


def _fallback_albedo(TL):
    """Cross-map grass albedo fallback for packs whose own terrain_layers ship no grass
    texture. The sidecar contract stays an ABSOLUTE path (the viewer reads it verbatim),
    but it is derived portably at build time instead of a hardcoded dev-machine literal:
    datasets root = EFT_ASSETS_ROOT, else <EFT_TARKMAP_ROOT>/../eft_assets, else the
    grandparent of TL (<assets>/<dataset>/terrain_layers). Any sibling dataset's grass
    albedo qualifies; interchange_v2 (the validated source) is preferred when present."""
    assets = os.environ.get("EFT_ASSETS_ROOT")
    if not assets:
        tk = os.environ.get("EFT_TARKMAP_ROOT")
        assets = os.path.join(os.path.dirname(tk), "eft_assets") if tk else None
    if not assets or not os.path.isdir(assets):
        assets = os.path.dirname(os.path.dirname(os.path.abspath(TL)))
    for cand in ("grass_Grass3_D.png", "grass_Grass5_512_D.png"):
        hits = sorted(glob.glob(os.path.join(assets, "*", "terrain_layers", cand)))
        pref = [h for h in hits
                if os.path.basename(os.path.dirname(os.path.dirname(h))) == "interchange_v2"]
        if hits:
            return os.path.abspath((pref or hits)[0])
    # nothing found anywhere: keep the legacy sidecar semantics (a best-guess absolute
    # path that may not exist) so the schema and viewer behavior stay unchanged
    return os.path.abspath(os.path.join(assets, "interchange_v2", "terrain_layers",
                                        "grass_Grass3_D.png"))
# Road/asphalt SURFACE meshes (laid ON the grass terrain; the game grids are road-excluding but
# leave some road-EDGE cells nonzero, and decal-role roads sit millimetres above the terrain, so
# a safety mask is kept). Their XZ footprint masks grass. Exclude non-surface props that happen
# to match (signs, lamps, fences, barriers, WIRE splines — wires cross grass fields).
# NB 'light(?!house)': plain 'light' matched 'Lighthouse_main_road_*', silently disabling the
# whole mask on Lighthouse (dry tufts through the main road).
ROAD_RE = re.compile(r"road|asphalt|spline|tarcola|parking|sidewalk|\bcurb", re.I)
ROAD_NOT_RE = re.compile(r"sign|lamp|light(?!house)|fence|barrier|pole|rail|wall|cone|bollard|wire", re.I)


def _rasterize_tri_xz(cells, cell, ax, az, bx, bz, cx, cz):
    """Add every grid cell whose center falls inside triangle (a,b,c) projected to XZ, PLUS the
    cells under the triangle's edges. Cell index k under the round(w/cell) convention has its
    center at world k*cell, so we test the barycentric half-plane sign at PX=gx*cell (matches
    the clump lookup exactly). The edge pass is the conservative half: a thin road strip can
    contain no cell center at all and would otherwise contribute nothing (dilation cannot grow
    an empty seed)."""
    minx = int(np.floor(min(ax, bx, cx) / cell)); maxx = int(np.ceil(max(ax, bx, cx) / cell))
    minz = int(np.floor(min(az, bz, cz) / cell)); maxz = int(np.ceil(max(az, bz, cz) / cell))
    if maxx - minx > 1400 or maxz - minz > 1400:   # guard vs a mismatched map-spanning mesh
        return
    gx = np.arange(minx, maxx + 1); gz = np.arange(minz, maxz + 1)
    if not len(gx) or not len(gz):
        return
    PX, PZ = np.meshgrid(gx * cell, gz * cell)
    d1 = (PX - bx) * (az - bz) - (ax - bx) * (PZ - bz)
    d2 = (PX - cx) * (bz - cz) - (bx - cx) * (PZ - cz)
    d3 = (PX - ax) * (cz - az) - (cx - ax) * (PZ - az)
    inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
    zz, xx = np.where(inside)
    cells.update(zip(gx[xx].tolist(), gz[zz].tolist()))
    # conservative edge sampling at half-cell spacing (bounded by the same span guard above)
    for (x0, z0, x1, z1) in ((ax, az, bx, bz), (bx, bz, cx, cz), (cx, cz, ax, az)):
        npt = int(max(abs(x1 - x0), abs(z1 - z0)) / (cell * 0.5)) + 1
        t = np.linspace(0.0, 1.0, npt + 1)
        ex = np.round((x0 + (x1 - x0) * t) / cell).astype(np.int64)
        ez = np.round((z0 + (z1 - z0) * t) / cell).astype(np.int64)
        cells.update(zip(ex.tolist(), ez.tolist()))


def build_road_mask(mani, mb, ib, cell=1.0, dilate=1):
    """World-XZ coverage set (cell-quantized) of road/asphalt mesh footprints. Grass clumps that
    land inside are skipped. Footprints are filled by rasterizing the road TRIANGLES (not just
    the vertices): large flat surfaces (parking_floor_LOD0 = 298x519m from only 433 verts) have
    almost no interior verts, so vertex-only sampling missed their middle and grass poked through
    the whole slab. Triangle fill covers the interior; a small dilation closes seams + edges."""
    id2mesh = {m["id"]: m for m in mani["meshes"]}
    road_ids = {mid for mid, m in id2mesh.items()
                if ROAD_RE.search(m["name"]) and not ROAD_NOT_RE.search(m["name"])}
    vl = mani["vertex"]; vs = vl["stride"]
    poff = next(a for a in vl["attrs"] if a["name"] == "position")["offset"]
    inst = mani["instance"]; istride = inst["stride"]
    fo = {f["name"]: f["offset"] for f in inst["fields"]}
    mbnp = np.frombuffer(mb, np.uint8)
    n = len(ib) // istride
    cells = set()
    ninst = 0; ntri = 0
    for i in range(n):
        b = i * istride
        mid = struct.unpack_from("<I", ib, b + fo["meshId"])[0]
        if mid not in road_ids:
            continue
        ninst += 1
        a = struct.unpack_from("<12f", ib, b + fo["affine"])
        me = id2mesh[mid]; voff = me["vtxOffset"]
        ioff = me["idxOffset"]; ic = me["idxCount"]
        idx = mbnp[ioff:ioff + ic * 4].view("<u4").reshape(-1, 3)  # explicit LE (pack contract)
        # NO triangle subsampling: index order gives no coverage guarantee, so a cap can drop
        # arbitrary (even large) triangles and leave grass-through-road holes. Full raster is
        # only ~14% more triangles than the old cap on interchange.
        for t in idx:
            wx = [0.0, 0.0, 0.0]; wz = [0.0, 0.0, 0.0]
            for j in range(3):
                o = voff + int(t[j]) * vs
                lx, ly, lz = struct.unpack_from("<3f", mb, o + poff)
                wx[j] = a[0] * lx + a[1] * ly + a[2] * lz + a[3]
                wz[j] = a[8] * lx + a[9] * ly + a[10] * lz + a[11]
            _rasterize_tri_xz(cells, cell, wx[0], wz[0], wx[1], wz[1], wx[2], wz[2])
            ntri += 1
    if dilate:
        d = set()
        for (cx, cz) in cells:
            for dx in range(-dilate, dilate + 1):
                for dz in range(-dilate, dilate + 1):
                    d.add((cx + dx, cz + dz))
        cells = d
    print(f"[grass] road mask: {len(road_ids)} road meshes / {ninst} instances, {ntri} tris "
          f"-> {len(cells)} masked {cell}m cells")
    return cells, cell
# 1 clump per non-empty density cell keeps it dense but bounded (~435k on Lighthouse,
# ~217k on Interchange at 512-res grids / 1.37m cells).


def load_pack(pack):
    mani = json.load(open(os.path.join(pack, "manifest.json")))
    mb = open(os.path.join(pack, "meshes.bin"), "rb").read()
    ib = open(os.path.join(pack, "instances.bin"), "rb").read()
    return mani, mb, ib


def terrain_uv_world_grid(mani, mb, aff, mesh):
    """GxG grid of world XYZ indexed [gy, gx], from the terrain mesh verts (uv->world).
    G is derived from the mesh's vertex count (513x513 heightmap grid on all current maps);
    a non-square terrain mesh falls back to 513 with a warning. Nodes no vertex maps to stay
    NaN — the caller counts them and clump placement skips non-finite lookups."""
    vl = mani["vertex"]; vs = vl["stride"]
    poff = next(a for a in vl["attrs"] if a["name"] == "position")["offset"]
    uoff = next(a for a in vl["attrs"] if a["name"] == "uv")["offset"]
    N = mesh["vtxCount"]; off = mesh["vtxOffset"]
    G = int(round(N ** 0.5))
    if G * G != N or G < 2:
        print(f"[grass] WARNING: terrain mesh {mesh['name']} has {N} verts (not a square grid) "
              f"- assuming 513x513 uv nodes")
        G = 513
    grid = np.full((G, G, 3), np.nan, np.float32)
    a = aff  # row-major 3x4
    for k in range(N):
        o = off + k * vs
        lx, ly, lz = struct.unpack_from("<3f", mb, o + poff)
        u, v = struct.unpack_from("<2f", mb, o + uoff)
        wx = a[0]*lx + a[1]*ly + a[2]*lz + a[3]
        wy = a[4]*lx + a[5]*ly + a[6]*lz + a[7]
        wz = a[8]*lx + a[9]*ly + a[10]*lz + a[11]
        gx = int(round(u * (G - 1))); gy = int(round(v * (G - 1)))
        if 0 <= gx < G and 0 <= gy < G:
            grid[gy, gx] = (wx, wy, wz)
    nan = int((~np.isfinite(grid[..., 0])).sum())
    if nan:
        print(f"[grass] WARNING: {mesh['name']}: {nan}/{G*G} uv grid nodes unfilled "
              f"({100.0*nan/(G*G):.2f}%) - clumps hitting them are skipped")
    return grid


def bilinear(grid, u, v):
    G = grid.shape[0]
    fx = u * (G - 1); fy = v * (G - 1)
    x0 = int(np.floor(fx)); y0 = int(np.floor(fy))
    x0 = max(0, min(G - 2, x0)); y0 = max(0, min(G - 2, y0))
    tx = fx - x0; ty = fy - y0
    c = (grid[y0, x0] * (1 - tx) * (1 - ty) + grid[y0, x0 + 1] * tx * (1 - ty)
         + grid[y0 + 1, x0] * (1 - tx) * ty + grid[y0 + 1, x0 + 1] * tx * ty)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep ~1/N^2 of the nonzero density cells (deterministic hash-selected, "
                         "no parity bias). Default 1 = every nonzero cell: the 512-res grids "
                         "(1.37m cells) already bound the count (~435k lighthouse, ~217k interchange)")
    ap.add_argument("--density", type=float, default=1.0,
                    help="scale on the game's per-cell instance COUNT. 1.0 (default) = faithful: "
                         "every instance the game's grids ask for (reserve ~1.24M). Lower it "
                         "(0.5, 0.25) to thin the meadow on weaker GPUs; relative variation and "
                         "placement are preserved, and every nonzero cell keeps >=1 instance")
    ap.add_argument("--self-contained", action="store_true",
                    help="redistribution PR3: keep every path written into grass_sidecar.json "
                         "pack-relative, copying the grass albedo into the pack when it does not "
                         "already live there")
    a = ap.parse_args()
    pack = a.pack
    mani, mb, ib = load_pack(pack)
    # per-pack terrain_layers dir (manifest sidecar is the path to its manifest.json; legacy
    # packs wrote it ABSOLUTE, self-contained packs write it PACK-RELATIVE -> resolve against
    # the pack dir, mirroring the viewer's Pack::resolve_path).
    # Maps with no terrain sidecar or no density grids (indoor maps: Factory, Labs) simply have
    # no grass — SKIP cleanly (write nothing, exit 0) so the all-maps build fleet doesn't abort.
    # A map that HAS density grids but yields zero clumps still hard-fails below (that's a bug).
    tl_side = (mani.get("sidecars") or {}).get("terrainLayers")
    if not tl_side:
        print(f"[grass] {pack}: no terrainLayers sidecar — grassless map, skipping"); return
    if not os.path.isabs(tl_side):
        tl_side = os.path.join(pack, tl_side)
    TL = os.path.dirname(tl_side)

    def _density_names(d):
        return sorted({m.group(1)
                       for f in glob.glob(os.path.join(d, "grass_density_Slice_*.bin"))
                       if (m := re.search(r"(Slice_\d+_\d+)", os.path.basename(f)))})

    # discover slice names from the density files (interchange: 4, lighthouse: 6, ...)
    names = _density_names(TL)
    # SELF-CONTAINED gap: build_map runs the density EXTRACTOR after assemble copied
    # terrain_layers into the pack, so a first-ever build finds no density grids in the pack
    # copy. Fall back to the dataset's terrain_layers (manifest.datasetPath provenance) and,
    # under --self-contained, mirror the density files into the pack so it stays complete.
    ds_tl = os.path.join(mani.get("datasetPath") or "", "terrain_layers")
    if not names and os.path.normcase(os.path.normpath(TL)) != os.path.normcase(os.path.normpath(ds_tl)):
        ds_names = _density_names(ds_tl) if os.path.isdir(ds_tl) else []
        if ds_names:
            if a.self_contained:
                os.makedirs(TL, exist_ok=True)
                for f in glob.glob(os.path.join(ds_tl, "grass_density_Slice_*.bin")):
                    shutil.copy2(f, os.path.join(TL, os.path.basename(f)))
                print(f"[grass] self-contained: mirrored {len(ds_names)} dataset density slices "
                      f"into {TL}")
                names = _density_names(TL)
            else:
                print(f"[grass] density grids not in {TL} - reading from dataset {ds_tl}")
                TL = ds_tl
                names = ds_names
    # The pack's terrain_layers copy can be STALE relative to the dataset (assemble copies it
    # before the grass extractor runs, and a re-extraction that adds per-prototype grids lands
    # only in the dataset). Per-prototype grids are what carry the plant variety, so prefer
    # whichever source actually has them — otherwise a pack silently rebuilds from the old
    # single-summed-grid data and loses every species but one.
    if os.path.isdir(ds_tl) and os.path.normcase(os.path.normpath(TL)) != os.path.normcase(os.path.normpath(ds_tl)):
        has_pack_protos = bool(glob.glob(os.path.join(TL, "grass_protos_Slice_*.bin")))
        ds_protos = glob.glob(os.path.join(ds_tl, "grass_protos_Slice_*.bin"))
        if ds_protos and not has_pack_protos:
            if a.self_contained:
                os.makedirs(TL, exist_ok=True)
                for pat in ("grass_protos_Slice_*.bin", "grass_density_Slice_*.bin",
                            "grass_*.png", "grass.json"):
                    for f in glob.glob(os.path.join(ds_tl, pat)):
                        shutil.copy2(f, os.path.join(TL, os.path.basename(f)))
                print(f"[grass] self-contained: mirrored {len(ds_protos)} per-prototype slice(s) "
                      f"+ cards from the dataset into {TL}")
                names = _density_names(TL)
            else:
                print(f"[grass] per-prototype grids only in the dataset — reading from {ds_tl}")
                TL = ds_tl
                names = _density_names(TL)
    if not names:
        print(f"[grass] {pack}: no grass_density_Slice_*.bin under {TL} — grassless map, skipping")
        return
    print(f"[grass] density slices in {TL}: {names}")
    id2mesh = {m["id"]: m for m in mani["meshes"]}
    inst = mani["instance"]; istride = inst["stride"]
    fo = {f["name"]: f["offset"] for f in inst["fields"]}
    n = len(ib) // istride

    # slice name -> (affine, mesh)
    slices = {}
    for i in range(n):
        b = i * istride
        flags = struct.unpack_from("<I", ib, b + fo["flags"])[0]
        if not (flags & FLAG_TERRAIN):
            continue
        mid = struct.unpack_from("<I", ib, b + fo["meshId"])[0]
        aff = struct.unpack_from("<12f", ib, b + fo["affine"])
        me = id2mesh[mid]
        for s in names:
            if s in me["name"]:
                if s in slices:
                    print(f"[grass] WARNING: slice {s} matches MULTIPLE terrain instances - "
                          f"keeping the last; split/multipart terrain (Streets-style) loses parts")
                slices[s] = (aff, me)
    print(f"[grass] {len(slices)} terrain slices")
    if not slices:
        raise SystemExit(f"[grass] FATAL: no FLAG_TERRAIN instance matched any of {names} — "
                         f"refusing to emit an empty grass.bin (it would silently disable grass)")

    # road/asphalt footprint mask (grass under road SURFACE meshes -> pokes through, skip it).
    road_cells, rcell = build_road_mask(mani, mb, ib)

    # ---- authored grass params from the extractor's grass.json --------------------------------
    # PER-SLICE tint + the WavingGrass wind params. Both were extracted all along and thrown
    # away: the old build read `next(iter(slices.values()))` — the FIRST slice's tint applied to
    # the whole map (customs' slices genuinely differ: [.499,.484,.275] vs [.7,.6,.5]) — and the
    # wind params never reached the pack at all, so our grass stood perfectly still.
    DEF_TINT = [0.7, 0.75, 0.55]
    grass_json, slice_tints, wind = {}, {}, {"strength": 0.0, "amount": 0.0, "speed": 0.0}
    try:
        # Read grass.json from the SAME source TL resolved to above. The manifest sidecar points
        # at the pack's copy, which can be stale (different slices, no prototype list) — trusting
        # it silently reverted this build to the single-summed-grid path.
        gj = os.path.join(TL, "grass.json")
        if not os.path.exists(gj):
            gj = (mani.get("sidecars") or {}).get("grassJson") or ""
            if gj and not os.path.isabs(gj):
                gj = os.path.join(pack, gj)
        grass_json = json.load(open(gj))
        print(f"[grass] params from {gj}")
        # NOTE the two key spaces do NOT always agree: `slices` is keyed by TerrainData m_Name
        # while `density` is keyed by the GPU-Instancer detail MANAGER's GameObject name (reserve:
        # TerrainData "Slice_3_4" vs managers "Slice_4_3"/"Slice_4_4"). Requiring an exact match
        # silently dropped BOTH the authored tint and the wind params to defaults, which is how
        # reserve ended up with wind 0 and the generic green. Exact match wins; otherwise fall
        # back to the first authored slice (the params are per-terrain and near-uniform per map).
        sl_all = grass_json.get("slices") or {}
        first = next(iter(sl_all.values()), None)
        for sn, sv in sl_all.items():
            slice_tints[sn] = sv.get("tint", DEF_TINT)
        for sn in (grass_json.get("density") or {}):
            if sn not in slice_tints and first:
                slice_tints[sn] = first.get("tint", DEF_TINT)
        sv = next((sl_all[s] for s in (grass_json.get("density") or {}) if s in sl_all), first)
        if sv:
            wind = {"strength": float(sv.get("strength", 0.5)),
                    "amount": float(sv.get("amount", 0.15)),
                    "speed": float(sv.get("speed", 0.5))}
        print(f"[grass] wind: {wind}; per-slice tints: "
              f"{ {k: [round(c,3) for c in v] for k, v in slice_tints.items()} }")
    except Exception as e:
        print(f"[grass] no grass.json params ({e}) — default tint, no wind")

    # legacy fallback card, used only for datasets extracted before per-prototype grids existed
    legacy_albedo = ""
    for cand in ("grass_Grass3_D.png", "grass_Grass5_512_D.png"):
        if os.path.exists(os.path.join(TL, cand)):
            legacy_albedo = cand
            break
    if not legacy_albedo:
        hits = sorted(glob.glob(os.path.join(TL, "grass_*.png")))
        hits = [h for h in hits if not os.path.basename(h).lower().endswith(("_n.png", "_normal.png"))]
        if hits:
            legacy_albedo = os.path.basename(hits[0])

    def bilinear_many(grid, u, v):
        """Vectorised twin of `bilinear` — the faithful density now places ~1.2M instances, far
        too many for the old per-instance Python call."""
        G = grid.shape[0]
        fx = np.clip(np.asarray(u, np.float64), 0.0, 1.0) * (G - 1)
        fy = np.clip(np.asarray(v, np.float64), 0.0, 1.0) * (G - 1)
        x0 = np.clip(np.floor(fx).astype(np.int64), 0, G - 2)
        y0 = np.clip(np.floor(fy).astype(np.int64), 0, G - 2)
        tx = (fx - x0)[:, None]
        ty = (fy - y0)[:, None]
        return (grid[y0, x0] * (1 - tx) * (1 - ty) + grid[y0, x0 + 1] * tx * (1 - ty)
                + grid[y0 + 1, x0] * (1 - tx) * ty + grid[y0 + 1, x0 + 1] * tx * ty)

    # ---- PLACEMENT -------------------------------------------------------------------------
    # Per PROTOTYPE (plant species) and per CELL COUNT. EFT's grids store an instance COUNT per
    # cell (0..16 per prototype here, up to 50 summed) for EACH of the map's 9-30 detail
    # prototypes, every one a different plant with its own billboard card. The old build
    # collapsed BOTH axes: it summed every prototype into one grid, then treated that grid as a
    # BOOLEAN (np.nonzero), emitting exactly one clump per nonzero cell of a single repeated
    # texture. That is why our grass read as a uniform carpet of one plant instead of the game's
    # mixed, clumpy meadow. Records are 24 B: [x,y,z,rotY,scale] f32 + kind u32.
    kinds, kind_list = {}, []

    def kind_id(tex, tint):
        """One render kind per (billboard texture, tint). Several prototypes share a card
        (Grass_new_1_D appears 4x on reserve Slice_4_3), so dedupe — each kind becomes one
        mesh + material + draw in the viewer."""
        key = (tex, tuple(round(float(c), 4) for c in tint))
        if key not in kinds:
            kinds[key] = len(kind_list)
            kind_list.append({"albedo": tex, "tint": list(key[1])})
        return kinds[key]

    def place_layer(layer, side, grid, kid, seed):
        """Vectorised placement of ONE prototype layer -> (bytes, emitted, road-skipped)."""
        ys, xs = np.nonzero(layer)
        if len(xs) == 0:
            return b"", 0, 0
        cnt = layer[ys, xs].astype(np.int64)
        if a.density != 1.0:
            cnt = np.maximum(1, np.rint(cnt * a.density).astype(np.int64))
        rep = np.repeat(np.arange(len(xs), dtype=np.int64), cnt)
        k = np.arange(len(rep), dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
        cx, cy = xs[rep].astype(np.int64), ys[rep].astype(np.int64)
        # deterministic 64-bit hash per (cell, prototype, k) — NEVER client-random
        h = ((cx * np.int64(73856093)) ^ (cy * np.int64(19349663))
             ^ (k * np.int64(83492791)) ^ np.int64(seed)).astype(np.uint64)
        h ^= h >> np.uint64(13)
        h *= np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(29)
        # sub-cell jitter: the game scatters instances INSIDE the detail cell. Placing several
        # per cell at the cell CENTRE would rebuild a visible 512-lattice.
        du = (h & np.uint64(0xFFFF)).astype(np.float64) / 65535.0
        dv = ((h >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.float64) / 65535.0
        u = (cx + du) / float(side)
        v = 1.0 - (cy + dv) / float(side)
        w = np.asarray(bilinear_many(grid, u, v), np.float32)
        ok = np.all(np.isfinite(w), axis=1)
        onroad = np.zeros(len(w), bool)
        if road_cells and ok.any():
            # only the finite rows: an unfilled terrain UV node yields NaN, and casting NaN to
            # int64 is undefined (and warns)
            wf = w[ok]
            kx = np.rint(wf[:, 0] / rcell).astype(np.int64)
            kz = np.rint(wf[:, 2] / rcell).astype(np.int64)
            onroad[ok] = np.fromiter((p in road_cells for p in zip(kx.tolist(), kz.tolist())),
                                     bool, len(kx))
        nroad = int((ok & onroad).sum())
        sel = ok & ~onroad
        w, hs = w[sel], h[sel]
        if len(w) == 0:
            return b"", 0, nroad
        rot = ((hs >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.float32) / 65535.0 * 6.2831855
        scale = 0.75 + ((hs >> np.uint64(48)) & np.uint64(0xFF)).astype(np.float32) / 255.0 * 0.6
        out = np.empty((len(w), 6), np.float32)
        out[:, 0:3] = w[:, 0:3]
        out[:, 3] = rot
        out[:, 4] = scale
        out[:, 5] = np.full(len(w), kid, np.uint32).view(np.float32)
        return out.tobytes(), len(w), nroad

    recs = bytearray()
    total = 0
    skipped_road = 0
    for sname, (aff, me) in slices.items():
        dpath = os.path.join(TL, f"grass_density_{sname}.bin")
        if not os.path.exists(dpath):
            print(f"[grass] {sname}: no density {dpath}"); continue
        raw = np.fromfile(dpath, np.uint8)
        side = int(round(len(raw) ** 0.5))
        if side * side != len(raw):
            print(f"[grass] {sname}: density file is {len(raw)} bytes (not square) - skipping"); continue
        grid = terrain_uv_world_grid(mani, mb, aff, me)
        stint = slice_tints.get(sname, DEF_TINT)          # PER-SLICE tint (was: first slice for all)
        ppath = os.path.join(TL, f"grass_protos_{sname}.bin")
        pmeta = ((grass_json.get("density", {}) or {}).get(sname, {}) or {}).get("prototypes") or []
        layers = []
        if os.path.exists(ppath) and pmeta:
            stack = np.fromfile(ppath, np.uint8)
            if len(stack) == len(pmeta) * side * side:
                stack = stack.reshape(len(pmeta), side, side)
                layers = [(stack[i], pmeta[i].get("tex", "")) for i in range(len(pmeta))]
            else:
                print(f"[grass] {sname}: grass_protos is {len(stack)} B, expected "
                      f"{len(pmeta)*side*side} — falling back to the summed grid")
        if not layers:                                     # datasets extracted before per-proto grids
            layers = [(raw.reshape(side, side), legacy_albedo)]
        scnt = 0
        for i, (layer, tex) in enumerate(layers):
            if not tex:
                continue
            b, n, nr = place_layer(layer, side, grid, kind_id(tex, stint), (i + 1) * 2654435761)
            recs += b
            scnt += n
            skipped_road += nr
        total += scnt
        print(f"[grass] {sname}: {scnt} instances from {len(layers)} prototype layer(s), "
              f"tint {[round(c, 3) for c in stint]}")

    if total == 0:
        raise SystemExit(f"[grass] FATAL: 0 clumps emitted for {pack} — "
                         f"refusing to write an empty grass.bin (it would silently disable grass)")
    open(os.path.join(pack, "grass.bin"), "wb").write(recs)

    # ---- SIDECAR ---------------------------------------------------------------------------
    # Every kind's card is copied INTO the pack (packs must be self-contained + portable: an
    # absolute build path breaks on the user's machine and leaks a personal path). A kind whose
    # texture cannot be resolved is DROPPED loudly rather than silently emptying the sidecar —
    # an empty albedo makes the viewer skip grass entirely, which is how reserve shipped 197,599
    # clumps that never drew a single pixel.
    out_kinds, dropped = [], []
    for ki in kind_list:
        srcpng = ki["albedo"]
        cand = srcpng if os.path.isabs(srcpng) else os.path.join(TL, srcpng)
        base = os.path.basename(srcpng)
        if not os.path.exists(cand):
            dropped.append(base)
            out_kinds.append(None)
            continue
        dst = os.path.join(os.path.abspath(pack), base)
        if os.path.abspath(cand) != os.path.abspath(dst):
            shutil.copy2(cand, dst)
        out_kinds.append({"albedo": base, "tint": ki["tint"]})
    if dropped:
        print(f"[grass] WARNING: {len(dropped)} kind(s) had no texture on disk and were dropped: "
              f"{sorted(set(dropped))}")
    if not any(out_kinds):
        raise SystemExit(f"[grass] FATAL: no grass texture resolved for {pack} — refusing to "
                         f"write a sidecar the viewer would silently skip (run the grass "
                         f"extractor for this dataset)")
    # Remap kind indices past any dropped kind so grass.bin stays consistent with the sidecar.
    remap = {}
    kept = []
    for i, k in enumerate(out_kinds):
        if k is not None:
            remap[i] = len(kept)
            kept.append(k)
    if len(kept) != len(out_kinds):
        arr = np.frombuffer(bytes(recs), np.float32).reshape(-1, 6).copy()
        kid = arr[:, 5].view(np.uint32)
        keep = np.fromiter((int(v) in remap for v in kid), bool, len(kid))
        arr = arr[keep]
        kid = arr[:, 5].view(np.uint32)
        arr[:, 5] = np.fromiter((remap[int(v)] for v in kid), np.uint32, len(kid)).view(np.float32)
        recs = bytearray(arr.tobytes())
        total = len(arr)
        open(os.path.join(pack, "grass.bin"), "wb").write(recs)
    json.dump({"count": total, "format": 2, "kinds": kept, "wind": wind,
               # legacy single-texture fields: a viewer older than the multi-kind format still
               # renders (one card, one tint) instead of skipping grass outright.
               "albedo": kept[0]["albedo"], "tint": kept[0]["tint"]},
              open(os.path.join(pack, "grass_sidecar.json"), "w"), indent=1)
    print(f"[grass] TOTAL {total} instances across {len(kept)} kind(s) "
          f"({skipped_road} skipped under roads) -> {pack}/grass.bin "
          f"({len(recs)//24} recs, {len(recs)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
