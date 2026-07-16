#!/usr/bin/env python
"""#4 GRASS — deterministic placement from the GPU-Instancer density grids.

EFT grass is baked GPU-Instancer density (NOT Unity terrain detail — that DB is a zeroed
decoy). We already extracted the 4 slice density grids (grass_density_Slice_*.bin, 1024^2
uint8, road/building-excluding, hand-authored). This emits one grass instance per non-empty
cell (deterministic per-cell hash for rotation/scale — NEVER client-random), placed on the
terrain surface via a UV->world bilinear lookup built from the pack's terrain meshes (so XZ
AND height are exact and in pack space).

Output: <pack>/grass.bin  = N records of [x,y,z, rotY, scale] f32 (20 B), pack space.
        <pack>/grass_sidecar.json = {count, albedo, tint}

  python -m eft_pipeline.build_grass --pack packs/interchange.eftpack
"""
import os, sys, json, struct, argparse, re, glob
import numpy as np

# Cross-map albedo fallback for packs whose terrain_layers ship no grass texture.
FALLBACK_ALBEDO = r"C:/Users/user/beamng_blender_pipeline/eft_assets/interchange_v2/terrain_layers/grass_Grass3_D.png"
FLAG_TERRAIN = 1 << 1
# Road/asphalt SURFACE meshes (laid ON the grass terrain, so the density grid still has grass
# under them -> grass pokes through). Their XZ footprint masks grass. Exclude non-surface props
# that happen to be named road_* (signs, lamps, fences, barriers).
ROAD_RE = re.compile(r"road|asphalt|spline|tarcola|parking|sidewalk|\bcurb", re.I)
ROAD_NOT_RE = re.compile(r"sign|lamp|light|fence|barrier|pole|rail|wall|cone|bollard", re.I)


def _rasterize_tri_xz(cells, cell, ax, az, bx, bz, cx, cz):
    """Add every grid cell whose center falls inside triangle (a,b,c) projected to XZ.
    Cell index k under the round(w/cell) convention has its center at world k*cell, so we
    test the barycentric half-plane sign at PX=gx*cell (matches the clump lookup exactly)."""
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
        idx = mbnp[ioff:ioff + ic * 4].view(np.uint32).reshape(-1, 3)
        step = max(1, len(idx) // 4000)          # cap tris/instance on dense splines
        for t in idx[::step]:
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
# density cell -> #instances by value band (sparse map is already road-excluding).
# 1 clump per non-empty cell keeps it dense but bounded (~300k total on Interchange).


def load_pack(pack):
    mani = json.load(open(os.path.join(pack, "manifest.json")))
    mb = open(os.path.join(pack, "meshes.bin"), "rb").read()
    ib = open(os.path.join(pack, "instances.bin"), "rb").read()
    return mani, mb, ib


def terrain_uv_world_grid(mani, mb, aff, mesh):
    """513x513 grid of world XYZ indexed [gy, gx], from the terrain mesh verts (uv->world)."""
    vl = mani["vertex"]; vs = vl["stride"]
    poff = next(a for a in vl["attrs"] if a["name"] == "position")["offset"]
    uoff = next(a for a in vl["attrs"] if a["name"] == "uv")["offset"]
    N = mesh["vtxCount"]; off = mesh["vtxOffset"]
    G = 513  # 513x513 heightmap grid
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
    # fill any missing cells (rare) by nearest along rows
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
    ap.add_argument("--stride", type=int, default=2, help="place 1 clump per N density cells (2 = 1/4 density)")
    a = ap.parse_args()
    pack = a.pack
    mani, mb, ib = load_pack(pack)
    # per-pack terrain_layers dir (manifest sidecar is an absolute path to its manifest.json).
    # Maps with no terrain sidecar or no density grids (indoor maps: Factory, Labs) simply have
    # no grass — SKIP cleanly (write nothing, exit 0) so the all-maps build fleet doesn't abort.
    # A map that HAS density grids but yields zero clumps still hard-fails below (that's a bug).
    tl_side = (mani.get("sidecars") or {}).get("terrainLayers")
    if not tl_side:
        print(f"[grass] {pack}: no terrainLayers sidecar — grassless map, skipping"); return
    TL = os.path.dirname(tl_side)
    # discover slice names from the density files (interchange: 4, lighthouse: 6, ...)
    names = sorted({re.search(r"(Slice_\d+_\d+)", os.path.basename(f)).group(1)
                    for f in glob.glob(os.path.join(TL, "grass_density_Slice_*.bin"))})
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
                slices[s] = (aff, me)
    print(f"[grass] {len(slices)} terrain slices")
    if not slices:
        raise SystemExit(f"[grass] FATAL: no FLAG_TERRAIN instance matched any of {names} — "
                         f"refusing to emit an empty grass.bin (it would silently disable grass)")

    # road/asphalt footprint mask (grass under road SURFACE meshes -> pokes through, skip it).
    road_cells, rcell = build_road_mask(mani, mb, ib)

    recs = bytearray()
    total = 0
    skipped_road = 0
    for sname, (aff, me) in slices.items():
        dpath = os.path.join(TL, f"grass_density_{sname}.bin")
        if not os.path.exists(dpath):
            print(f"[grass] {sname}: no density {dpath}"); continue
        dens = np.fromfile(dpath, np.uint8).reshape(1024, 1024)  # [row=y, col=x]
        grid = terrain_uv_world_grid(mani, mb, aff, me)
        ys, xs = np.nonzero(dens)
        # subsample by stride for a bounded count
        sel = (xs % a.stride == 0) & (ys % a.stride == 0)
        xs, ys = xs[sel], ys[sel]
        cnt = 0
        for cx, cy in zip(xs, ys):
            u = (cx + 0.5) / 1024.0
            v = (cy + 0.5) / 1024.0
            w = bilinear(grid, u, v)
            if not np.all(np.isfinite(w)):
                continue
            # skip clumps that land under a road/asphalt surface mesh
            if (int(round(w[0] / rcell)), int(round(w[2] / rcell))) in road_cells:
                skipped_road += 1
                continue
            # deterministic per-cell hash -> rotation + scale (never client-random)
            h = (int(cx) * 73856093) ^ (int(cy) * 19349663)
            rot = (h & 0xFFFF) / 0xFFFF * 6.28318
            scale = 0.75 + ((h >> 16) & 0xFF) / 255.0 * 0.6
            recs += struct.pack("<5f", float(w[0]), float(w[1]), float(w[2]), rot, scale)
            cnt += 1
        total += cnt
        print(f"[grass] {sname}: {cnt} clumps")

    if total == 0:
        raise SystemExit(f"[grass] FATAL: 0 clumps emitted for {pack} — "
                         f"refusing to write an empty grass.bin (it would silently disable grass)")
    open(os.path.join(pack, "grass.bin"), "wb").write(recs)
    # grass albedo: prefer the denser Grass3_D, else Grass5, else cross-map fallback
    alb = None
    for cand in ("grass_Grass3_D.png", "grass_Grass5_512_D.png"):
        p = os.path.join(TL, cand)
        if os.path.exists(p):
            alb = p
            break
    if alb is None:
        alb = FALLBACK_ALBEDO
        print(f"[grass] no grass albedo in {TL}, using cross-map fallback {alb}")
    tint = [0.7, 0.75, 0.55]
    try:
        g = json.load(open(mani["sidecars"]["grassJson"]))
        sl = next(iter(g.get("slices", {}).values()), {})
        tint = sl.get("tint", tint)
    except Exception:
        pass
    json.dump({"count": total, "albedo": alb.replace("\\", "/"), "tint": tint},
              open(os.path.join(pack, "grass_sidecar.json"), "w"), indent=1)
    print(f"[grass] TOTAL {total} clumps ({skipped_road} skipped under roads) -> {pack}/grass.bin ({len(recs)//20} recs, {len(recs)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
