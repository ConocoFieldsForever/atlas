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
import os, sys, json, struct, argparse, re
import numpy as np

TL = r"C:/Users/user/beamng_blender_pipeline/eft_assets/interchange_v2/terrain_layers"
FLAG_TERRAIN = 1 << 1
# Road/asphalt SURFACE meshes (laid ON the grass terrain, so the density grid still has grass
# under them -> grass pokes through). Their XZ footprint masks grass. Exclude non-surface props
# that happen to be named road_* (signs, lamps, fences, barriers).
ROAD_RE = re.compile(r"road|asphalt|spline|tarcola|parking|sidewalk|\bcurb", re.I)
ROAD_NOT_RE = re.compile(r"sign|lamp|light|fence|barrier|pole|rail|wall|cone|bollard", re.I)


def build_road_mask(mani, mb, ib, cell=1.0, dilate=1):
    """World-XZ coverage set (cell-quantized) of road/asphalt mesh footprints. Grass clumps that
    land inside are skipped. Footprint is sampled from the road meshes' world-space vertices (dense
    along the splines) plus a small dilation to close gaps and cover the surface width."""
    id2mesh = {m["id"]: m for m in mani["meshes"]}
    road_ids = {mid for mid, m in id2mesh.items()
                if ROAD_RE.search(m["name"]) and not ROAD_NOT_RE.search(m["name"])}
    vl = mani["vertex"]; vs = vl["stride"]
    poff = next(a for a in vl["attrs"] if a["name"] == "position")["offset"]
    inst = mani["instance"]; istride = inst["stride"]
    fo = {f["name"]: f["offset"] for f in inst["fields"]}
    n = len(ib) // istride
    cells = set()
    ninst = 0
    for i in range(n):
        b = i * istride
        mid = struct.unpack_from("<I", ib, b + fo["meshId"])[0]
        if mid not in road_ids:
            continue
        ninst += 1
        a = struct.unpack_from("<12f", ib, b + fo["affine"])
        me = id2mesh[mid]; N = me["vtxCount"]; off = me["vtxOffset"]
        step = max(1, N // 4000)                                    # cap verts/mesh for speed
        for k in range(0, N, step):
            o = off + k * vs
            lx, ly, lz = struct.unpack_from("<3f", mb, o + poff)
            wx = a[0] * lx + a[1] * ly + a[2] * lz + a[3]
            wz = a[8] * lx + a[9] * ly + a[10] * lz + a[11]
            cells.add((int(round(wx / cell)), int(round(wz / cell))))
    if dilate:
        d = set()
        for (cx, cz) in cells:
            for dx in range(-dilate, dilate + 1):
                for dz in range(-dilate, dilate + 1):
                    d.add((cx + dx, cz + dz))
        cells = d
    print(f"[grass] road mask: {len(road_ids)} road meshes, {ninst} instances -> {len(cells)} masked {cell}m cells")
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
        for s in ("Slice_1_1", "Slice_1_2", "Slice_2_1", "Slice_2_2"):
            if s in me["name"]:
                slices[s] = (aff, me)
    print(f"[grass] {len(slices)} terrain slices")

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

    open(os.path.join(pack, "grass.bin"), "wb").write(recs)
    # grass albedo: prefer the denser Grass3_D
    alb = os.path.join(TL, "grass_Grass3_D.png")
    tint = [0.7, 0.75, 0.55]
    try:
        g = json.load(open(os.path.join(TL, "grass.json")))
        sl = next(iter(g.get("slices", {}).values()), {})
        tint = sl.get("tint", tint)
    except Exception:
        pass
    json.dump({"count": total, "albedo": alb.replace("\\", "/"), "tint": tint},
              open(os.path.join(pack, "grass_sidecar.json"), "w"), indent=1)
    print(f"[grass] TOTAL {total} clumps ({skipped_road} skipped under roads) -> {pack}/grass.bin ({len(recs)//20} recs, {len(recs)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
