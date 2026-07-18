"""Parallel wrapper around eft_extract_v2.py.

Split the level list into N chunks and run the (UNCHANGED, tested) single-process extractor on each
chunk concurrently into a private staging dataset, then MERGE the chunk outputs into the real
dataset. The per-level extraction logic is reused verbatim; the only new code is chunk scheduling +
an output merge. Big maps go from single-core to N-core (Streets = 217 levels).

Correctness of the merge relies on three properties of the extractor's output (verified against
eft_extract_v2.py):
  * Mesh OBJ filenames are LEVEL-scoped ("<name>__<lv>_<fid>_<pid>.obj", "terrain_<lv>_<name>.obj").
    Chunks hold DISJOINT levels, so mesh files never collide between chunks.
  * Texture PNGs + terrain splat-layer PNGs are SOURCE-identity scoped (same content -> same name),
    so a texture referenced from two chunks is byte-identical -> first writer wins, dedup is safe.
  * scene.json instances reference meshes by FILENAME and LODGroups by a GLOBAL index; each chunk
    numbers its LODGroups 0..K locally, so the merge offsets each chunk's instance `lod.g` by the
    running LODGroup count. LODGroups are per-level Unity objects -> disjoint across chunks, no dedup.

  python extraction/unity/extract_parallel.py --levels a,b,c,... --name <dataset> [--jobs N]
    [--data-root DIR] [--terrain-step N] [--alllod] [--terrain-only]

Env: EFT_JOBS overrides --jobs (EFT_JOBS=1 forces the plain serial single extractor). ASCII output,
[STAGE i/N]-style markers so the menu's loader still reads progress.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Global level-completion counter -> a single [SUBPROGRESS] extract <done>/<total> stream the viewer
# reads to move the loader bar DURING the (long) extraction, across all chunk processes.
_prog_lock = threading.Lock()
_prog = {"done": 0, "total": 0}

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, "eft_extract_v2.py")
PY = sys.executable or "python"

# Same OUTROOT resolution as eft_extract_v2.py (datasets dir).
_TK = os.environ.get("EFT_TARKMAP_ROOT")
OUTROOT = os.environ.get("EFT_ASSETS_ROOT") or (
    os.path.join(os.path.dirname(_TK), "eft_assets") if _TK else os.path.join(os.getcwd(), "eft_assets")
)


def _level_size(data_root, lv):
    """Bytes of level<lv> (schedule the biggest first to shrink the long tail)."""
    try:
        return os.path.getsize(os.path.join(data_root, f"level{lv}"))
    except OSError:
        return 0


def _chunk(levels, n):
    """Greedy longest-processing-time bin packing into n balanced chunks (by level file size)."""
    bins = [[] for _ in range(n)]
    load = [0] * n
    for lv, sz in levels:
        i = min(range(n), key=lambda k: load[k])
        bins[i].append(lv)
        load[i] += sz + 1  # +1 so zero-size levels still spread out
    return [b for b in bins if b]


def _run_chunk(idx, chunk_levels, name, passthrough):
    """Run the single-process extractor on one chunk into <name>__p<idx>. Returns (idx, rc)."""
    cname = f"{name}__p{idx}"
    cmd = [PY, EXTRACT, "--levels", ",".join(str(x) for x in chunk_levels), "--name", cname] + passthrough
    print(f"[CHUNK {idx}] {len(chunk_levels)} levels -> {cname}", flush=True)
    # Stream the child's stdout with a per-chunk prefix so progress is legible when interleaved.
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="ascii", errors="replace",
    )
    for line in p.stdout:
        print(f"  [p{idx}] {line.rstrip()}", flush=True)
        # eft_extract_v2 prints "level<lv>: +N mesh ... (<t>s)" once per finished level -> global bar.
        s = line.strip()
        if s.startswith("level") and " mesh" in s and s.endswith("s)"):
            with _prog_lock:
                _prog["done"] += 1
                d, t = _prog["done"], _prog["total"]
            if t:
                print(f"[SUBPROGRESS] extract {d}/{t}", flush=True)
    rc = p.wait()
    print(f"[CHUNK {idx}] done rc={rc}", flush=True)
    return idx, rc


def _move_into(src_dir, dst_dir, overwrite, skip=()):
    """Move regular files src_dir/* -> dst_dir/* (MOVE, not copy, so peak disk stays ~1x). With
    overwrite=False an existing target is kept (dedup for content-identical texture/layer files)."""
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for fn in os.listdir(src_dir):
        if fn in skip:
            continue
        sp = os.path.join(src_dir, fn)
        if not os.path.isfile(sp):
            continue
        dp = os.path.join(dst_dir, fn)
        if os.path.exists(dp):
            if not overwrite:
                continue  # identical content already present (dedup)
            os.remove(dp)
        shutil.move(sp, dp)
        n += 1
    return n


def _merge(name, n_chunks, out, levels_order):
    """Merge <name>__p0..p{n-1} into <name>/. Offsets per-chunk LODGroup indices; dedups tex/layers."""
    md, td, tl = (os.path.join(out, d) for d in ("meshes", "tex", "terrain_layers"))
    os.makedirs(md, exist_ok=True)
    os.makedirs(td, exist_ok=True)
    all_inst, all_lod, all_levels = [], [], []
    terrain = {"tiles": {}, "layers": []}
    for idx in range(n_chunks):
        cout = os.path.join(OUTROOT, f"{name}__p{idx}")
        scene_fp = os.path.join(cout, "scene.json")
        if not os.path.isfile(scene_fp):
            raise SystemExit(f"[MERGE FAILED] chunk {idx} produced no scene.json ({cout})")
        sc = json.load(open(scene_fp, encoding="utf-8"))
        base = len(all_lod)  # this chunk's LODGroups land at [base, base+len)
        for it in sc.get("instances", []):
            lod = it.get("lod")
            if lod is not None and "g" in lod:
                lod["g"] = int(lod["g"]) + base
            all_inst.append(it)
        all_lod.extend(sc.get("lodGroups", []))
        all_levels.extend(sc.get("levels", []))
        # meshes: level-scoped -> disjoint across chunks (overwrite is a no-op safety net).
        _move_into(os.path.join(cout, "meshes"), md, overwrite=True)
        # textures + terrain layers: source-identity scoped -> dedup (keep first, drop identical dup).
        _move_into(os.path.join(cout, "tex"), td, overwrite=False)
        ctl = os.path.join(cout, "terrain_layers")
        if os.path.isdir(ctl):
            _move_into(ctl, tl, overwrite=False, skip=("manifest.json",))
            cm = os.path.join(ctl, "manifest.json")
            if os.path.isfile(cm):
                m = json.load(open(cm, encoding="utf-8"))
                terrain["tiles"].update(m.get("tiles", {}))
                for layer in m.get("layers", []):
                    if layer not in terrain["layers"]:
                        terrain["layers"].append(layer)
    # Guard: a mis-offset lod.g would silently render a WRONG pack (LODs swapped/dropped). A dangling
    # index is impossible in the correct merge, so treat it as a merge bug and fail the build loudly.
    nlod = len(all_lod)
    bad = sum(1 for it in all_inst if it.get("lod") and int(it["lod"].get("g", -1)) >= nlod)
    if bad:
        raise SystemExit(f"[BUILD FAILED] merge: {bad} instances reference a LODGroup index >= {nlod} "
                         f"(offset bug) - refusing to write a corrupt scene.json")
    if terrain["tiles"]:
        os.makedirs(tl, exist_ok=True)
        json.dump(terrain, open(os.path.join(tl, "manifest.json"), "w"), indent=1)
    # Emit scene.json in the CONFIGURED level order (provenance only; instances already carry lv).
    json.dump(
        {"instances": all_inst, "up": "unity", "levels": levels_order, "lodGroups": all_lod,
         "lod_schema": 1,
         "note": "OBJ verts are UnityPy X-flipped+winding-reversed; builder must un-flip"},
        open(os.path.join(out, "scene.json"), "w"),
    )
    print(f"[MERGE] {len(all_inst)} instances, {len(all_lod)} LODGroups, {len(terrain['tiles'])} "
          f"terrain tiles -> {out}", flush=True)
    return len(all_inst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs", type=int, default=0, help="parallel extractor processes (0=auto)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--terrain-step", type=int, default=None)
    ap.add_argument("--alllod", action="store_true")
    ap.add_argument("--terrain-only", action="store_true")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    data_root = args.data_root or os.environ.get(
        "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")

    # passthrough args forwarded to every chunk extractor
    passthrough = []
    if args.data_root:
        passthrough += ["--data-root", args.data_root]
    if args.terrain_step is not None:
        passthrough += ["--terrain-step", str(args.terrain_step)]
    if args.alllod:
        passthrough.append("--alllod")
    if args.terrain_only:
        passthrough.append("--terrain-only")

    env_jobs = os.environ.get("EFT_JOBS")
    jobs = int(env_jobs) if env_jobs and env_jobs.strip().isdigit() else args.jobs
    if jobs <= 0:
        jobs = max(1, (os.cpu_count() or 4) - 2)
    jobs = min(jobs, len(levels))

    out = os.path.join(OUTROOT, args.name)

    # 1 job (or 1 level) -> just run the plain extractor into the dataset directly (no merge risk).
    if jobs <= 1:
        print(f"[PARALLEL] jobs=1 -> single-process extraction into {args.name}", flush=True)
        rc = subprocess.call([PY, EXTRACT, "--levels", args.levels, "--name", args.name] + passthrough)
        sys.exit(rc)

    sized = sorted(((lv, _level_size(data_root, lv)) for lv in levels), key=lambda t: -t[1])
    chunks = _chunk(sized, jobs)
    n = len(chunks)
    print(f"[PARALLEL] {len(levels)} levels across {n} chunks (jobs={jobs})", flush=True)
    _prog["total"] = len(levels)  # denominator for the [SUBPROGRESS] extraction bar
    T0 = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(_run_chunk, i, ch, args.name, passthrough) for i, ch in enumerate(chunks)]
        for f in futs:
            results.append(f.result())
    failed = [i for i, rc in results if rc != 0]
    if failed:
        print(f"[BUILD FAILED] extractor chunk(s) {failed} failed - see the [pN] log above", flush=True)
        sys.exit(1)
    print(f"[PARALLEL] all {n} chunks done in {time.time()-T0:.0f}s - merging", flush=True)

    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    _merge(args.name, n, out, levels)

    # cleanup the (now-emptied) chunk staging dirs
    for i in range(n):
        cout = os.path.join(OUTROOT, f"{args.name}__p{i}")
        if os.path.isdir(cout):
            shutil.rmtree(cout, ignore_errors=True)
    if not os.path.isfile(os.path.join(out, "scene.json")):
        print(f"[BUILD FAILED] merge produced no scene.json at {out}", flush=True)
        sys.exit(1)
    print(f"[PARALLEL] done in {time.time()-T0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
