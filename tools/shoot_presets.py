"""Capture the settings panel + one identical scene at each quality preset.

Each preset is reproduced with the SAME env knobs `QualityPreset::apply` sets, plus the persisted
`textureQuality`, so the shots show exactly what picking that preset in the UI does. Same camera,
same window size, same settle -- the only variable is the preset.
"""
import os, sys, io, json, time, subprocess, argparse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(HERE, "target", "release", "atlas.exe")
CONFIG = os.path.join(os.environ.get("APPDATA", ""), "atlas", "atlas.config.json")

# Mixed content: foliage, buildings, shadowed facades, distant props -- so the foliage/shadow/
# texture differences are all visible in one frame.
POSE = os.environ.get("SHOT_POSE", "239.2,32.7,-335.3,50.0,-10.1")
WINDOW = "1600x1000"

PRESETS = {
    # name: (env delta matching QualityPreset::apply, textureQuality)
    "ultra":  ({"EFT_SSAO": "1"}, 0),
    "high":   ({}, 1),
    "medium": ({"EFT_SHADOWS": "0", "EFT_CULL_PX": "2,600"}, 1),
    "low":    ({"EFT_SHADOWS": "0", "EFT_BLOOM": "0", "EFT_LIGHTS": "0",
                "EFT_CULL_PX": "4,1000"}, 2),
}


def set_tex_quality(q):
    try:
        cfg = json.load(open(CONFIG, encoding="utf-8-sig")) if os.path.exists(CONFIG) else {}
    except Exception:
        cfg = {}
    cfg["textureQuality"] = float(q)
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    json.dump(cfg, open(CONFIG, "w", encoding="utf-8"), indent=1)


def shoot(out_png, extra_env, texq, pack, settle, log_tag):
    set_tex_quality(texq)
    env = dict(os.environ)
    env.update({"EFT_POSE": POSE, "EFT_WIN": WINDOW, "EFT_SHOT": out_png,
                "EFT_SHOT_SETTLE": str(settle)})
    env.update(extra_env)
    logs = os.path.join(HERE, "docs", "_bench_logs")
    os.makedirs(logs, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(logs, f"shot_{log_tag}.log"), "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen([EXE, pack], env=env, stdout=lf, stderr=subprocess.STDOUT)
        # Poll for the PNG rather than waiting on exit: the viewer does not reliably terminate
        # after writing a shot, so waiting on the process just burns the whole timeout every time.
        # Give the file a beat to finish flushing once it appears, then stop the process.
        while p.poll() is None and time.time() - t0 < 300:
            if os.path.exists(out_png) and os.path.getsize(out_png) > 10000:
                time.sleep(1.5)
                break
            time.sleep(0.5)
        if p.poll() is None:
            p.kill()
            p.wait(timeout=20)
    ok = os.path.exists(out_png) and os.path.getsize(out_png) > 10000
    print(f"  {'OK ' if ok else 'FAIL'} {os.path.basename(out_png)}  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="packs/interchange.eftpack")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--settle", type=int, default=260)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("settings panel (Graphics section expanded):", flush=True)
    env = dict(os.environ)
    shoot(os.path.join(args.outdir, "menu_graphics.png"),
          {"EFT_GFX_OPEN": "1"}, 1, args.pack, args.settle, "menu")

    print("scene at each preset:", flush=True)
    for name, (delta, texq) in PRESETS.items():
        shoot(os.path.join(args.outdir, f"preset_{name}.png"), delta, texq, args.pack,
              args.settle, name)

    set_tex_quality(1)
    print("\ndone ->", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
