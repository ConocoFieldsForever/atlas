"""Benchmark every graphics knob: frame time AND VRAM, one run per configuration.

WHY: the settings menu should tell a user what a toggle actually buys them. That means measuring
each knob in isolation against a fixed baseline, on the same scene and camera, rather than guessing
which ones are expensive.

METHOD
  * One run per config in a REAL window at a fixed size (`EFT_WIN`), fixed camera (`EFT_POSE`),
    vsync off
    (`EFT_UNCAPPED=1`), `EFT_BENCH=<secs>`. The viewer settles 90 frames after the load completes,
    samples every frame for the window, prints one `[bench] ... avg=..ms fps=..` line and exits 0.
  * VRAM: `nvidia-smi` reports NO per-process figure on Windows (WDDM), so we sample TOTAL board
    usage. Baseline is sampled immediately before launch (median of several reads) and peak is the
    max seen while the run is alive; the difference is this process's residency. Anything else on
    the GPU (the game, browsers) only matters if it moves DURING a run, so each config re-samples
    its own baseline instead of trusting one global number.
  * Configs are deltas from a single baseline so each row is attributable to one knob.

    python tools/bench_gfx.py --pack packs/interchange.eftpack --secs 8
    python tools/bench_gfx.py --only shadows_off,tex_quarter        # re-run a subset
"""
import os, sys, io, json, time, argparse, subprocess, statistics, re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(HERE, "target", "release", "atlas.exe")
CONFIG = os.path.join(os.environ.get("APPDATA", ""), "atlas", "atlas.config.json")

# A view with heavy mixed load: mall shell + power station + foliage + many lights.
POSE = "239.2,32.7,-335.3,50.0,-10.1"
WINDOW = "1600x1000"   # fixed across configs so fragment cost is comparable
LOGDIR = os.path.join(HERE, "docs", "_bench_logs")

BENCH_RE = re.compile(
    r"\[bench\]\s+frames=(\d+)\s+secs=([\d.]+)\s+avg=([\d.]+)ms\s+fps=([\d.]+)"
    r"\s+p50=([\d.]+)\s+p95=([\d.]+)\s+p99=([\d.]+)\s+max=([\d.]+)"
)

# name -> (label, env delta, textureQuality override or None)
# `tex_quality` is a persisted config value, not an env var, so it is applied to the config file.
CONFIGS = [
    ("baseline",        "Baseline (defaults, textures Half)",      {},                              1),
    # --- textures: the single biggest VRAM lever (docs/VRAM_AUDIT.md) ---
    ("tex_full",        "Textures: Full",                          {},                              0),
    ("tex_quarter",     "Textures: Quarter",                       {},                              2),
    # --- shadows ---
    ("shadows_off",     "Sun shadows OFF",                         {"EFT_SHADOWS": "0"},            1),
    ("shadow_1024",     "Shadow map 1024 (from 2048)",             {"EFT_SHADOW_SIZE": "1024"},     1),
    ("shadow_4096",     "Shadow map 4096 (from 2048)",             {"EFT_SHADOW_SIZE": "4096"},     1),
    ("grass_shadows",   "Grass casts shadows ON",                  {"EFT_GRASS_SHADOWS": "1"},      1),
    # --- lighting ---
    ("lights_off",      "Realtime point/spot lights OFF",          {"EFT_LIGHTS": "0"},             1),
    ("gi_off",          "Baked GI (SH volume) OFF",                {"EFT_GI": "0"},                 1),
    # --- post ---
    ("bloom_off",       "Bloom OFF",                               {"EFT_BLOOM": "0"},              1),
    ("ssao_on",         "SSAO ON (default off)",                   {"EFT_SSAO": "1"},               1),
    ("fog_off",         "Distance fog OFF",                        {"EFT_FOG": "0"},                1),
    ("vignette_off",    "Vignette OFF",                            {"EFT_VIGNETTE": "0"},           1),
    # --- geometry / material ---
    ("parallax_off",    "Parallax mapping OFF",                    {"EFT_PARALLAX": "0"},           1),
    ("grass_off",       "Foliage/grass culled",                    {"EFT_CULL_PX": "1.5,1000"},     1),
    ("cull_aggressive", "Aggressive small-object cull (4px/8px)",  {"EFT_CULL_PX": "4,8"},          1),
    ("lod_bias2",       "LOD bias 2.0 (coarser shells sooner)",    {"EFT_LOD_BIAS": "2.0"},         1),
    # --- stacked presets, to check the knobs compose ---
    ("stack_low",       "STACK: quarter tex + no shadows/bloom/GI/lights + grass culled",
     {"EFT_SHADOWS": "0", "EFT_BLOOM": "0", "EFT_GI": "0", "EFT_LIGHTS": "0",
      "EFT_CULL_PX": "4,1000", "EFT_PARALLAX": "0", "EFT_VIGNETTE": "0", "EFT_FOG": "0"}, 2),
    ("stack_medium",    "STACK: half tex + no shadows + grass culled",
     {"EFT_SHADOWS": "0", "EFT_CULL_PX": "2,600"},                                                  1),
]


def vram_used():
    """Total board MiB in use, or None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def vram_baseline(n=5):
    vals = [v for v in (vram_used() for _ in range(n)) if v is not None]
    return statistics.median(vals) if vals else None


def set_tex_quality(q):
    try:
        cfg = json.load(open(CONFIG, encoding="utf-8-sig")) if os.path.exists(CONFIG) else {}
    except Exception:
        cfg = {}
    cfg["textureQuality"] = float(q)
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    with open(CONFIG, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)


def run_one(name, label, delta, texq, pack, secs):
    set_tex_quality(texq)
    env = dict(os.environ)
    env.update({
        "EFT_BENCH": str(secs),
        "EFT_UNCAPPED": "1",
        # NOT EFT_HIDDEN: with no window the GPU-driven upload never finishes, so the bench gate
        # (`gpu_load.in_progress() == false`) never opens and the run hangs until timeout. A real
        # window at a FIXED size is also the honest thing to measure -- fill cost is part of it.
        "EFT_WIN": WINDOW,
        "EFT_POSE": POSE,
    })
    env.update(delta)
    base = vram_baseline()
    t0 = time.time()
    # Child output goes to a FILE, never a pipe. The viewer logs heavily at INFO; with a pipe that
    # nothing drains until exit, the 64 KB buffer fills, the child blocks on write and never reaches
    # the bench -- every run then dies on the timeout with no [bench] line. (That was the bug, not
    # anything about headless mode.)
    log_path = os.path.join(LOGDIR, f"{name}.log")
    os.makedirs(LOGDIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen([EXE, pack], env=env, stdout=lf, stderr=subprocess.STDOUT)
        peak = base or 0
        while p.poll() is None:
            v = vram_used()
            if v is not None and v > peak:
                peak = v
            time.sleep(0.5)
            if time.time() - t0 > 300:
                p.kill()
                break
    try:
        out_lines = io.open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        out_lines = []
    m = None
    for ln in reversed(out_lines):
        m = BENCH_RE.search(ln)
        if m:
            break
    rec = {"name": name, "label": label, "texQuality": texq, "env": delta,
           "wall_s": round(time.time() - t0, 1),
           "vram_base": base, "vram_peak": peak,
           "vram_mib": (peak - base) if (base is not None) else None}
    if m:
        rec.update(frames=int(m.group(1)), avg_ms=float(m.group(3)), fps=float(m.group(4)),
                   p50=float(m.group(5)), p95=float(m.group(6)), p99=float(m.group(7)),
                   max_ms=float(m.group(8)))
    else:
        rec["error"] = "no [bench] line — run failed or never settled"
        rec["tail"] = out_lines[-6:]
    return rec


def main():
    global WINDOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="packs/interchange.eftpack")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--only", default="", help="comma-separated config names")
    ap.add_argument("--out", default=None)
    ap.add_argument("--win", default=WINDOW,
                    help="window WxH. A 5090 is CPU-bound at 1600x1000 on this scene, which hides "
                         "every fragment-bound knob; sweep a higher resolution too.")
    args = ap.parse_args()

    WINDOW = args.win
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    todo = [c for c in CONFIGS if not want or c[0] in want]
    out_path = args.out or os.path.join(HERE, "docs", f"GFX_BENCH_{WINDOW}.json")

    print(f"benchmarking {len(todo)} configs x {args.secs}s on {args.pack} @ {WINDOW}", flush=True)
    results = []
    for i, (name, label, delta, texq) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name:16} {label}", flush=True)
        rec = run_one(name, label, delta, texq, args.pack, args.secs)
        if "fps" in rec:
            print(f"        fps={rec['fps']:.1f} avg={rec['avg_ms']:.2f}ms p95={rec['p95']:.2f}ms "
                  f"vram={rec['vram_mib']} MiB ({rec['wall_s']}s)", flush=True)
        else:
            print(f"        FAILED: {rec.get('error')}", flush=True)
        results.append(rec)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)

    # Summary table, deltas relative to baseline.
    base = next((r for r in results if r["name"] == "baseline" and "fps" in r), None)
    print("\n" + "=" * 108)
    print(f"{'config':18} {'fps':>7} {'avg ms':>8} {'p95 ms':>8} {'VRAM MiB':>9} "
          f"{'d fps':>8} {'d VRAM':>8}  label")
    print("-" * 108)
    for r in results:
        if "fps" not in r:
            print(f"{r['name']:18} {'FAILED':>7}")
            continue
        dfps = f"{r['fps'] - base['fps']:+.1f}" if base else "-"
        dv = (f"{r['vram_mib'] - base['vram_mib']:+d}"
              if base and r.get("vram_mib") is not None and base.get("vram_mib") is not None else "-")
        print(f"{r['name']:18} {r['fps']:7.1f} {r['avg_ms']:8.2f} {r['p95']:8.2f} "
              f"{str(r['vram_mib']):>9} {dfps:>8} {dv:>8}  {r['label']}")
    print("=" * 108)
    print(f"\nwrote {out_path}")
    set_tex_quality(1)  # leave the user's config on the shipped default
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
