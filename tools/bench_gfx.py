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
# Camera env for the runs. A STATIC pose cannot see any cost that only appears under motion -- the
# sun cascades refit while the camera turns, and that was measured at 11x the resting shadow cost.
# So a moving scenario (--fly / --orbit) is not a nicety, it is the only way to price shadows
# honestly. Overridden by --pose/--fly/--orbit; whatever is set here is passed through verbatim.
CAMERA = {"EFT_POSE": POSE}
WINDOW = "1600x1000"   # fixed across configs so fragment cost is comparable
LOGDIR = os.path.join(HERE, "docs", "_bench_logs")

BENCH_RE = re.compile(
    r"\[bench\]\s+frames=(\d+)\s+secs=([\d.]+)\s+avg=([\d.]+)ms\s+fps=([\d.]+)"
    r"\s+p50=([\d.]+)\s+p95=([\d.]+)\s+p99=([\d.]+)\s+max=([\d.]+)"
)

# name -> (label, env delta, textureQuality)
# Texture quality is persisted rather than environment-driven. Every benchmark run also selects
# the Custom preset: named presets own textureQuality at startup and would otherwise silently
# replace the value requested here (for example, persisted High always forces Half).
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
    # --- OVERLAY LAYERS ---------------------------------------------------------------------
    # The rows above price GRAPHICS. They cannot answer "what does the marker overlay cost me",
    # which is a CPU question: every layer adds entities that two visibility systems walk on every
    # camera move, and the loot layer alone is 2k+ markers on streets. `nocluster` is the one to
    # watch -- it disables the declutter that keeps distant markers from all drawing at once, so it
    # should read WORSE than baseline, and by how much is the value of that pass.
    ("lay_base",        "LAYERS: defaults (loot on, clustering on)", {},                            1),
    ("lay_noloot",      "LAYERS: loot overlay OFF",                {"EFT_LAYERS": "noloot"},        1),
    ("lay_nocluster",   "LAYERS: dense declutter OFF",             {"EFT_LAYERS": "nocluster"},     1),
    ("lay_spawns",      "LAYERS: + pmc/scav/boss spawns",          {"EFT_LAYERS": "pmc,scav,boss"}, 1),
    ("lay_nav",         "LAYERS: + extract/door/interact/lock",
     {"EFT_LAYERS": "extract,door,interact,lock"},                                                  1),
    ("lay_zones",       "LAYERS: + hazard/minefield/sniper/botzone",
     {"EFT_LAYERS": "hazard,minefield,sniper,botzone"},                                             1),
    ("lay_patrol",      "LAYERS: + patrol polylines",              {"EFT_LAYERS": "patrol"},        1),
    ("lay_all",         "LAYERS: everything on",
     {"EFT_LAYERS": "pmc,scav,boss,extract,door,interact,lock,hazard,switch,transit,stationary,"
                    "loose,minefield,sniper,botzone,patrol,airdrop,ritual,player,showinactive"},    1),
    # --- stacked presets, to check the knobs compose ---
    ("stack_low",       "STACK: quarter tex + no shadows/bloom/GI/lights + grass culled",
     {"EFT_SHADOWS": "0", "EFT_BLOOM": "0", "EFT_GI": "0", "EFT_LIGHTS": "0",
      "EFT_CULL_PX": "4,1000", "EFT_PARALLAX": "0", "EFT_VIGNETTE": "0", "EFT_FOG": "0"}, 2),
    ("stack_medium",    "STACK: half tex + no shadows + grass culled",
     {"EFT_SHADOWS": "0", "EFT_CULL_PX": "2,600"},                                                  1),
    # --- the ULTRA matrix -------------------------------------------------------------------------
    # The rows above are deltas from the SHIPPED look (High: half textures, SSAO off). They cannot
    # answer "what is expensive in MY session" for someone on Ultra, because Ultra turns SSAO on and
    # textures to Full, and a knob's cost is not independent of the others. These rows re-baseline on
    # Ultra so each delta is what THAT user would actually get back by turning the knob off.
    # Every run selects Custom, so GfxSettings::default() + the env delta is the real config:
    # default is shadows ON, grass ON, bloom ON, lights ON, SSAO OFF -- hence SSAO is opt-IN here.
    ("u_base",          "ULTRA: full tex + SSAO on (the Ultra preset)",
     {"EFT_SSAO": "1"},                                                                             0),
    ("u_no_ssao",       "ULTRA minus SSAO",                        {},                              0),
    ("u_no_shadows",    "ULTRA minus sun shadows",
     {"EFT_SSAO": "1", "EFT_SHADOWS": "0"},                                                         0),
    ("u_no_both",       "ULTRA minus SSAO AND sun shadows",        {"EFT_SHADOWS": "0"},            0),
    ("u_no_grass",      "ULTRA minus foliage/grass",
     {"EFT_SSAO": "1", "EFT_CULL_PX": "1.5,1000"},                                                  0),
    ("u_grass_shadows", "ULTRA + grass casts shadows",
     {"EFT_SSAO": "1", "EFT_GRASS_SHADOWS": "1"},                                                   0),
    ("u_no_lights",     "ULTRA minus realtime lights",
     {"EFT_SSAO": "1", "EFT_LIGHTS": "0"},                                                          0),
    # Distance-LOD A/B. Only meaningful on an --alllod pack (one built WITHOUT it has a single shell
    # per group, so both rows are identical). EFT_LOD=0 selects cull mode 0 = draw only the default
    # (finest-present) shell, which is exactly what a non-alllod pack renders — so this pair isolates
    # distance-LOD on ONE pack, with identical geometry, textures and camera.
    ("u_lod_off",       "ULTRA, distance-LOD OFF (finest shell only)",
     {"EFT_SSAO": "1", "EFT_LOD": "0"},                                                             0),
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


def read_config():
    try:
        with open(CONFIG, encoding="utf-8-sig") as fh:
            return json.load(fh)
    except Exception:
        return {}


def write_bench_config(q):
    cfg = read_config()
    # 4 = QualityPreset::Custom. A named preset owns textureQuality in main.rs, so changing the
    # texture field alone does not change the renderer and produces mislabeled benchmark rows.
    cfg["qualityPreset"] = 4.0
    cfg["textureQuality"] = float(q)
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    tmp = CONFIG + f".bench-{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    os.replace(tmp, CONFIG)


class ConfigSandbox:
    """Restore atlas.config.json byte-for-byte, including the originally-missing case."""

    def __enter__(self):
        self.existed = os.path.exists(CONFIG)
        self.original = None
        if self.existed:
            with open(CONFIG, "rb") as fh:
                self.original = fh.read()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.existed:
            os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
            tmp = CONFIG + f".restore-{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(self.original)
            os.replace(tmp, CONFIG)
        elif os.path.exists(CONFIG):
            os.unlink(CONFIG)


def run_one(name, label, delta, texq, pack, secs):
    write_bench_config(texq)
    env = dict(os.environ)
    env.update({
        "EFT_BENCH": str(secs),
        "EFT_UNCAPPED": "1",
        # NOT EFT_HIDDEN: with no window the GPU-driven upload never finishes, so the bench gate
        # (`gpu_load.in_progress() == false`) never opens and the run hangs until timeout. A real
        # window at a FIXED size is also the honest thing to measure -- fill cost is part of it.
        "EFT_WIN": WINDOW,
    })
    env.update(CAMERA)
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
    resolved_cfg = read_config()
    # Record the camera: a static row and a flythrough row are not comparable, and without this the
    # two are indistinguishable once they are sitting in the same JSON file.
    rec = {"name": name, "label": label, "texQuality": texq, "env": delta, "camera": dict(CAMERA),
           "requestedSettings": {"qualityPreset": 4, "textureQuality": texq},
           "resolvedSettings": {
               "qualityPreset": int(resolved_cfg.get("qualityPreset", -1)),
               "textureQuality": int(resolved_cfg.get("textureQuality", -1)),
           },
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
    # Camera scenario. Mutually exclusive; --fly/--orbit are the ones that can price sun shadows,
    # because a static camera never makes the cascades refit.
    cam = ap.add_mutually_exclusive_group()
    cam.add_argument("--pose", default=None, metavar="x,y,z,yaw,pitch",
                     help=f"static camera (default {POSE})")
    cam.add_argument("--fly", default=None, metavar="x1,y1,z1>x2,y2,z2@secs",
                     help="ping-pong a straight path, looking forward")
    cam.add_argument("--orbit", default=None, metavar="cx,cy,cz,radius,height,degps",
                     help="circle a target point, looking at it")
    ap.add_argument("--baseline", default="baseline", metavar="NAME",
                    help="config name the summary deltas are relative to (use u_base for the "
                         "Ultra matrix)")
    args = ap.parse_args()

    WINDOW = args.win
    if args.fly:
        CAMERA.clear(); CAMERA["EFT_FLY"] = args.fly
    elif args.orbit:
        CAMERA.clear(); CAMERA["EFT_ORBIT"] = args.orbit
    elif args.pose:
        CAMERA["EFT_POSE"] = args.pose
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    todo = [c for c in CONFIGS if not want or c[0] in want]
    out_path = args.out or os.path.join(HERE, "docs", f"GFX_BENCH_{WINDOW}.json")

    print(f"benchmarking {len(todo)} configs x {args.secs}s on {args.pack} @ {WINDOW} "
          f"camera={CAMERA}", flush=True)
    results = []
    # The benchmark temporarily edits the same config used by the menu. Always restore the user's
    # exact bytes, even on Ctrl-C, timeout, a failed viewer launch, or a malformed results path.
    with ConfigSandbox():
        for i, (name, label, delta, texq) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {name:16} {label}", flush=True)
            rec = run_one(name, label, delta, texq, args.pack, args.secs)
            if "fps" in rec:
                print(f"        fps={rec['fps']:.1f} avg={rec['avg_ms']:.2f}ms p95={rec['p95']:.2f}ms "
                      f"vram={rec['vram_mib']} MiB ({rec['wall_s']}s)", flush=True)
            else:
                print(f"        FAILED: {rec.get('error')}", flush=True)
            results.append(rec)
            out_dir = os.path.dirname(os.path.abspath(out_path))
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1)

    # Summary table, deltas relative to baseline.
    base = next((r for r in results if r["name"] == args.baseline and "fps" in r), None)
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
    print("restored atlas.config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
