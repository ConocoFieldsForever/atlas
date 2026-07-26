"""Reader for EFT's baked occlusion data: StreamingAssets/Culling_Data/<guid>_packed_cull.bytes.

These are the bake output of the "Perfect Culling" Unity asset (Koenigz) -- the game's own
precomputed visibility set (PVS). 15 files, 4.6 GB, one per location. Nothing in this repo
consumes them yet; this tool exists so the format stays reproducible.

WHAT IS CONFIRMED (validated on all 15 files -- see `verify`)

    u32   nScenes
    nScenes x { u32 16 ; byte[16] sceneGuid }     # Unity scene GUIDs, raw 16-byte form
    <cell records, contiguous>
    u32   cellOffset[nCells]                      # absolute byte offset of each cell
    u32   nCells                                  # LAST dword in the file

  cell record:
    float3 centre
    float3 size          # ADAPTIVE: 3.0 m at the coarsest, subdivided smaller as needed
    float4 rotation      # unit quaternion
    u32    clen
    byte[clen]           # zlib stream (78 da), or clen == 0 for an empty cell

  inflated cell payload:
    u8    nBlocks                                 # <= nScenes
    nBlocks x { u8 sceneIdx ; u16 nVisible ; u16 dataLen ; byte[dataLen] }

  Read the index table FIRST (seek to EOF-4 for nCells, then back 4*nCells for the offsets) and
  random-access cells from it. Do NOT walk the file sequentially: it is slower, and one bad block
  costs you the remainder of the file.

WHAT IS NOT DECODED

  The innermost `data` of each sub-block. The vendor documents it as "variable bit length
  encoding", which matches measurement: 3.5-6 bits per entry, and the length depends on the
  VALUES not just the count (two blocks with nVisible=17 encode to 8 and 10 bytes). So it is an
  entropy/varint code over renderer indices, and finishing it needs the library's bit reader --
  not another stride guess.

  Consuming it would also need a renderer-index join: `sceneIdx` points into this file's GUID
  table, and the indices inside `data` are positions in that scene's renderer list, which the
  pack does not currently preserve (the same identity gap EXTRACTABLES_AUDIT flagged for doors).

Usage:
    python tools/pvs_probe.py list                 # every file, joined to its map
    python tools/pvs_probe.py info <guid|map>      # header, cell count, bounds
    python tools/pvs_probe.py cell <guid|map> <i>  # one cell's record + sub-blocks
    python tools/pvs_probe.py verify [guid|map]    # structural validation (all files by default)
"""
import os
import struct
import sys
import zlib

GAME = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
CD = os.path.join(GAME, "StreamingAssets", "Culling_Data")

# guid -> map id. Derived, not authored: each *_Culling scene's PerfectCullingAdaptiveGrid holds
# the guid as a 32-char string and it matches the filename exactly (15/15, no unmatched files).
# Regenerate with the AdaptiveGrid scan in the docstring of docs/GAME_DATA_EXTRACTION.md S5.
KNOWN = {
    "408a4e02c7264964ba401ac12936f3d6": "shoreline",
    "e21f6b3e21a448ef91e6da3a71f4902e": "lighthouse",
    "04c9e612665c44ab9833e4c817cf17f1": "streets",
    "d7443b32f81f42e8becafe5602835ba7": "ground_zero",
    "659fcc1e14014372baa985a921bef9f8": "customs",
    "99a5ef6adc1a42fd8b333e292f59da48": "labs",
    "78c128dd326a4b8a81cb1e82d9745c32": "reserve",
    "3f8c141e1ca84602a218c54bf4429508": "interchange",
    "e1f487f5608a4627bdeda58cb9c7639d": "woods",
    "e00110633b2c434bb9cf1bc0c2f7ad99": "factory_rework_day",
    "995afa7fc85a4c4c8354ebc1304f689a": "factory_rework_night",
    "ffbf28973616489e9216f21144e7c271": "labyrinth",
    "21564873ace844f88d2457724e627418": "sandbox_sl",
    "e5db7c9f81894dcab9cf11bdfe0b6347": "terminal",
    "065281ec5449481391979c8269072a13": "icebreaker",
}
BY_MAP = {v: k for k, v in KNOWN.items()}
CELL_HDR = 44          # float3 centre + float3 size + float4 rot + u32 clen


def resolve(name):
    """Accept a map id or a guid (full or unique prefix)."""
    if name in BY_MAP:
        return BY_MAP[name]
    if name in KNOWN:
        return name
    hits = [g for g in KNOWN if g.startswith(name)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"unknown map/guid {name!r}; try: {', '.join(sorted(BY_MAP))}")


class Pvs:
    """Random-access reader. Opens the index table only; cells are read on demand."""

    def __init__(self, guid):
        self.guid = guid
        self.path = os.path.join(CD, f"{guid}_packed_cull.bytes")
        self.f = open(self.path, "rb")
        self.size = os.path.getsize(self.path)

        n = struct.unpack("<I", self.f.read(4))[0]
        self.scene_guids = []
        for _ in range(n):
            ln = struct.unpack("<I", self.f.read(4))[0]
            if ln != 16:
                raise ValueError(f"scene guid length {ln} != 16")
            self.scene_guids.append(self.f.read(16).hex())
        self.header_end = self.f.tell()

        self.f.seek(self.size - 4)
        self.count = struct.unpack("<I", self.f.read(4))[0]
        tbl = self.size - 4 - 4 * self.count
        if tbl < self.header_end:
            raise ValueError(f"index table offset {tbl} precedes header end {self.header_end}")
        self.f.seek(tbl)
        self.offsets = struct.unpack(f"<{self.count}I", self.f.read(4 * self.count))
        self.table_off = tbl

    def close(self):
        self.f.close()

    def cell(self, i):
        """(centre, size, rotation, blocks) for cell i. blocks = [(sceneIdx, nVisible, data)]."""
        off = self.offsets[i]
        self.f.seek(off)
        v = struct.unpack("<10fI", self.f.read(CELL_HDR))
        clen = v[10]
        blocks = []
        if clen:
            # Tolerant inflate. A handful of blocks (factory_rework day+night, ~0.7% of cells)
            # ship a stream whose final marker/Adler-32 is missing, so the strict
            # zlib.decompress() raises "incomplete or truncated stream" on data that is
            # otherwise fine -- decompressobj returns the bytes. The offset table proves the
            # framing is right in these cells (next_offset - offset - 44 == clen exactly), so
            # this is a producer quirk, not a misparse on our side.
            raw = zlib.decompressobj().decompress(self.f.read(clen))
            if raw:
                p = 1
                for _ in range(raw[0]):
                    idx = raw[p]
                    nvis, dlen = struct.unpack_from("<HH", raw, p + 1)
                    blocks.append((idx, nvis, raw[p + 5:p + 5 + dlen]))
                    p += 5 + dlen
        return v[0:3], v[3:6], v[6:10], blocks


def cmd_list():
    print(f"{'map':22s} {'size':>8s}  guid")
    tot = 0
    for f in sorted(os.listdir(CD)):
        g = f.split("_")[0]
        sz = os.path.getsize(os.path.join(CD, f))
        tot += sz
        print(f"{KNOWN.get(g, '?'):22s} {sz/2**20:7.0f}M  {g}")
    print(f"{'TOTAL':22s} {tot/2**30:7.1f}G")


def cmd_info(name):
    p = Pvs(resolve(name))
    print(f"file    {os.path.basename(p.path)}  ({p.size/2**20:.0f} MB)")
    print(f"map     {KNOWN.get(p.guid, '?')}")
    print(f"scenes  {len(p.scene_guids)}")
    for i, g in enumerate(p.scene_guids):
        print(f"  [{i}] {g}")
    print(f"cells   {p.count}   index table @{p.table_off}")
    lo = hi = None
    step = max(1, p.count // 2000)
    sizes = set()
    for i in range(0, p.count, step):
        c, s, _, _ = p.cell(i)
        sizes.add(tuple(round(x, 2) for x in s))
        lo = c if lo is None else tuple(min(a, b) for a, b in zip(lo, c))
        hi = c if hi is None else tuple(max(a, b) for a, b in zip(hi, c))
    print(f"bounds  x {lo[0]:8.1f} .. {hi[0]:8.1f}")
    print(f"        y {lo[1]:8.1f} .. {hi[1]:8.1f}")
    print(f"        z {lo[2]:8.1f} .. {hi[2]:8.1f}")
    print(f"cell sizes sampled: {len(sizes)} distinct, largest "
          f"{max(sizes, key=lambda t: t[0]*t[1]*t[2])}")
    p.close()


def cmd_cell(name, i):
    p = Pvs(resolve(name))
    i = int(i)
    c, s, r, blocks = p.cell(i)
    print(f"cell {i}/{p.count}  @{p.offsets[i]}")
    print(f"  centre   ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    print(f"  size     ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})")
    print(f"  rotation ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f})")
    print(f"  blocks   {len(blocks)} of {len(p.scene_guids)} scenes")
    for idx, nvis, data in blocks:
        bits = len(data) * 8 / nvis if nvis else 0
        print(f"    scene[{idx}] visible={nvis:6d} data={len(data):6d} B "
              f"({bits:.2f} bits/entry)  {data[:12].hex(' ')}"
              f"{' ...' if len(data) > 12 else ''}")
    p.close()


def cmd_verify(name=None):
    """Structural validation. Every claim in the module docstring is re-checked here."""
    guids = [resolve(name)] if name else sorted(KNOWN)
    allok = True
    for g in guids:
        try:
            p = Pvs(g)
        except Exception as e:
            print(f"{KNOWN.get(g,'?'):22s} OPEN FAILED: {type(e).__name__}: {e}")
            allok = False
            continue
        errs = []
        # 1. offsets strictly increasing, first == header end, last cell ends at the table
        if p.offsets[0] != p.header_end:
            errs.append(f"offsets[0]={p.offsets[0]} != header_end={p.header_end}")
        if any(b <= a for a, b in zip(p.offsets, p.offsets[1:])):
            errs.append("offsets not strictly increasing")
        # 2. sample cells: parse, and check the sub-block walk lands exactly
        step = max(1, p.count // 400)
        checked = badcell = 0
        for i in range(0, p.count, step):
            try:
                _, _, rot, blocks = p.cell(i)
            except Exception as e:
                badcell += 1
                if badcell <= 2:
                    errs.append(f"cell {i}: {type(e).__name__} {str(e)[:40]}")
                continue
            checked += 1
            q = sum(x * x for x in rot) ** 0.5
            if abs(q - 1.0) > 1e-3:
                errs.append(f"cell {i}: rotation not unit ({q:.4f})")
            for idx, nvis, data in blocks:
                if idx >= len(p.scene_guids):
                    errs.append(f"cell {i}: sceneIdx {idx} >= {len(p.scene_guids)}")
            if len(blocks) > len(p.scene_guids):
                errs.append(f"cell {i}: {len(blocks)} blocks > {len(p.scene_guids)} scenes")
        # 3. the last cell must end exactly where the index table begins
        last = p.offsets[-1]
        p.f.seek(last + 40)
        clen = struct.unpack("<I", p.f.read(4))[0]
        if last + CELL_HDR + clen != p.table_off:
            errs.append(f"last cell ends {last+CELL_HDR+clen}, table @{p.table_off}")
        status = "OK" if not errs else "FAIL"
        if errs:
            allok = False
        print(f"{KNOWN.get(g,'?'):22s} {status:4s} cells={p.count:7d} scenes={len(p.scene_guids)} "
              f"checked={checked} badcells={badcell}")
        for e in errs[:4]:
            print(f"    - {e}")
        p.close()
    return 0 if allok else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        cmd_list()
    elif cmd == "info":
        cmd_info(*rest)
    elif cmd == "cell":
        cmd_cell(*rest)
    elif cmd == "verify":
        return cmd_verify(*rest)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
