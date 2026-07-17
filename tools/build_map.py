"""One-command map pack builder for the viewer's start menu.

Runs the full pipeline for a map whose DATASET already exists (eft_assets/<map>/scene.json —
full game extraction is a separate, much longer step; the menu surfaces that case).
Stages print `[STAGE i/N] name` markers and stream child output unbuffered so the menu's
progress panel can display them live. Exit 0 = pack ready (stamped). ASCII output only.

Usage: python tools/build_map.py <map> [--dry-run]
Env: EFT_PY_UNITY (UnityPy python), EFT_PY_BAKE (CUDA/warp python), EFT_TARKMAP_ROOT.
"""

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.dirname(HERE)
BBP = os.environ.get("EFT_TARKMAP_ROOT", r"C:\Users\user\beamng_blender_pipeline")
PY = sys.executable or "python"
PY_UNITY = os.environ.get("EFT_PY_UNITY", r"C:\Users\user\anaconda3\python.exe")
PY_BAKE = os.environ.get("EFT_PY_BAKE", r"C:\Users\user\anaconda3\envs\5090\python.exe")

# *_Light scene index per map (BuildSettings scene list). streets/ground_zero split lights
# across many scenes and are handled by the batch fleet — the menu build skips lights there.
LIGHT_LEVELS = {
    "interchange": 64,
    "lighthouse": 191,
    "factory": 69,
    "customs": 13,
    "woods": 167,
    "shoreline": 41,
    "reserve": 146,
    "labs": 114,
    "labyrinth": 551,
}
INDOOR_NO_GRASS = {"factory", "labs", "labyrinth"}


def run(stage, total, name, cmd, cwd, optional=False):
    print(f"[STAGE {stage}/{total}] {name}", flush=True)
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="ascii:replace")
    p = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="ascii", errors="replace",
    )
    for line in p.stdout:
        print("  " + line.rstrip(), flush=True)
    rc = p.wait()
    dt = time.time() - t0
    if rc != 0:
        if optional:
            print(f"[STAGE {stage}/{total}] {name}: FAILED rc={rc} ({dt:.0f}s) - optional, continuing", flush=True)
            return False
        print(f"[BUILD FAILED] stage '{name}' rc={rc} after {dt:.0f}s", flush=True)
        sys.exit(rc or 1)
    print(f"[STAGE {stage}/{total}] {name}: done ({dt:.0f}s)", flush=True)
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print("usage: build_map.py <map> [--dry-run]")
        sys.exit(2)
    m = args[0]
    dataset = os.path.join(BBP, "eft_assets", m)
    out_dir = os.path.join(BBP, "tarkmap", "out", m)
    pack = os.path.join(VIEWER, "packs", f"{m}.eftpack")
    total = 8

    print(f"[BUILD] map={m} dataset={dataset}", flush=True)
    if dry:
        for i, name in enumerate(
            ["check dataset", "extract lights", "bake lighting (GPU)", "assemble pack",
             "grass", "gameplay zones", "item icons", "stamp fingerprint"], 1):
            print(f"[STAGE {i}/{total}] {name}", flush=True)
            time.sleep(0.6)
            print(f"[STAGE {i}/{total}] {name}: done (0s)", flush=True)
        print("[BUILD OK] dry run", flush=True)
        return

    # 1: dataset present?
    print(f"[STAGE 1/{total}] check dataset", flush=True)
    if not os.path.isfile(os.path.join(dataset, "scene.json")):
        print(f"[BUILD FAILED] no dataset at {dataset} - run the full game extraction first "
              f"(extraction/README.md, eft_extract_v2)", flush=True)
        sys.exit(3)
    print(f"[STAGE 1/{total}] check dataset: done", flush=True)

    # 2: lights (optional; some maps have none / are fleet-handled)
    lv = LIGHT_LEVELS.get(m)
    if lv is not None and not any(
        f.startswith("lights_") for f in os.listdir(dataset) if f.endswith(".json")
    ):
        run(2, total, "extract lights",
            [PY_UNITY, os.path.join(BBP, "warp_viz", "eft_extract_lights.py"),
             "--level", str(lv), "--name", m],
            BBP, optional=True)
    else:
        print(f"[STAGE 2/{total}] extract lights: skipped (present or n/a)", flush=True)

    # 3: GPU SH bake (the long stage; skip if a fresh volume2 already exists)
    v2 = os.path.join(out_dir, "volume2.bin")
    if not os.path.isfile(v2):
        run(3, total, "bake lighting (GPU)",
            [PY_BAKE, os.path.join(BBP, "tarkmap", "tools", "bake_volume2.py"), m],
            os.path.join(BBP, "tarkmap"))
    else:
        print(f"[STAGE 3/{total}] bake lighting: skipped (volume2 exists)", flush=True)
    # promote volume2.* -> volume.* (assemble reads volume.*)
    for src, dst in [("volume2.bin", "volume.bin"), ("volume2.json", "volume.json"),
                     ("volume2.vis.bin", "volume.vis.bin")]:
        s = os.path.join(out_dir, src)
        if os.path.isfile(s):
            shutil.copyfile(s, os.path.join(out_dir, dst))

    # 4: assemble the pack (atomic; auto-ships loot/tasks/grade sidecars)
    run(4, total, "assemble pack", [PY, "-m", "eft_pipeline.assemble_bevy", m], VIEWER)

    # 5: grass (outdoor maps)
    if m in INDOOR_NO_GRASS:
        print(f"[STAGE 5/{total}] grass: skipped (indoor map)", flush=True)
    else:
        ok = run(5, total, "grass: extract density grids",
                 [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_grass.py"),
                  "--name", m], VIEWER, optional=True)
        if ok:
            run(5, total, "grass: build grass.bin",
                [PY, "-m", "eft_pipeline.build_grass", m, "--pack"], VIEWER, optional=True)

    # 6: typed gameplay zones (exfils/mines/snipers/doors/loose loot)
    run(6, total, "gameplay zones",
        [PY_UNITY, os.path.join(VIEWER, "extraction", "intel", "extract_gamedata.py"), m],
        VIEWER, optional=True)

    # 7: item icons (network; cached into the pack)
    run(7, total, "item icons",
        [PY, os.path.join(VIEWER, "extraction", "intel", "fetch_icons.py"), m],
        VIEWER, optional=True)

    # 8: stamp the game fingerprint (menu update detection)
    run(8, total, "stamp fingerprint",
        [PY, os.path.join(HERE, "stamp_fingerprint.py"), pack], VIEWER)

    print("[BUILD OK] pack ready", flush=True)


if __name__ == "__main__":
    main()
