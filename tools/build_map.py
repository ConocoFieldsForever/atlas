"""One-command map pack builder for the viewer's start menu.

Runs the full pipeline for a map whose DATASET already exists (<EFT_ASSETS_ROOT>/<dataset>/scene.json —
full game extraction is a separate, much longer step; the menu surfaces that case).
Stages print `[STAGE i/N] name` markers and stream child output unbuffered so the menu's
progress panel can display them live. Exit 0 = pack ready (stamped). ASCII output only.

Usage: python tools/build_map.py <map> [--dry-run]
Env (contract per extraction/README.md; unset -> legacy dev-machine defaults):
  EFT_TARKMAP_ROOT = the dir CONTAINING maps/ and out/ (a "tarkmap dir")
  EFT_ASSETS_ROOT  = the datasets dir (default: <EFT_TARKMAP_ROOT>/../eft_assets)
  EFT_PY_UNITY / EFT_PY_BAKE = UnityPy / CUDA-warp pythons (default: legacy anaconda
  interpreters when present on this machine, else this python)
"""

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.dirname(HERE)
# EFT_TARKMAP_ROOT is the tarkmap dir ITSELF (holds maps/ + out/), NOT the parent workspace.
TK = os.environ.get("EFT_TARKMAP_ROOT", r"C:\Users\user\beamng_blender_pipeline\tarkmap")
ASSETS = os.environ.get("EFT_ASSETS_ROOT") or os.path.normpath(
    os.path.join(TK, os.pardir, "eft_assets"))
PY = sys.executable or "python"


def _stage_python(envvar, legacy):
    """Interpreter for a stage: explicit env override > legacy anaconda path (keeps the
    original dev machine working unchanged) > whatever python is running this script."""
    p = os.environ.get(envvar)
    if p:
        return p
    return legacy if os.path.isfile(legacy) else PY


PY_UNITY = _stage_python("EFT_PY_UNITY", r"C:\Users\user\anaconda3\python.exe")
PY_BAKE = _stage_python("EFT_PY_BAKE", r"C:\Users\user\anaconda3\envs\5090\python.exe")

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
    # pass the contract values as-is (TK = the maps/+out/ dir, ASSETS = the datasets dir)
    env.setdefault("EFT_TARKMAP_ROOT", TK)
    env.setdefault("EFT_ASSETS_ROOT", ASSETS)
    p = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
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


def dataset_name(m):
    """DATASET folder name = the map config's source.root basename (map id 'interchange' ->
    dataset 'interchange_v2'). Workspace config (TK/maps) wins over the kit copy, matching
    extract_gamedata's resolution order. Falls back to the map id if no config is readable."""
    for p in (os.path.join(TK, "maps", m, "config.json"),
              os.path.join(VIEWER, "extraction", "maps", m, "config.json")):
        if os.path.isfile(p):
            try:
                root = json.load(open(p, encoding="utf-8"))["source"]["root"]
            except Exception as e:
                print(f"[BUILD] WARNING: unreadable config {p} ({e}) - using map id as "
                      f"dataset name", flush=True)
                return m
            return os.path.basename(os.path.normpath(root.replace("/", os.sep)))
    return m


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print("usage: build_map.py <map> [--dry-run]")
        sys.exit(2)
    m = args[0]
    dsname = dataset_name(m)
    dataset = os.path.join(ASSETS, dsname)
    # out/ stays keyed by MAP ID: bake_volume2 / extract_gamedata / assemble_bevy all write
    # and read TK/out/<map id> (they resolve the dataset via the map config themselves).
    out_dir = os.path.join(TK, "out", m)
    pack = os.path.join(VIEWER, "packs", f"{m}.eftpack")
    total = 8

    print(f"[BUILD] map={m} dataset={dsname} dataset_dir={dataset}", flush=True)
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
    if lv is None and m in ("streets", "ground_zero") and not any(
        f.startswith("lights_") for f in os.listdir(dataset) if f.endswith(".json")
    ):
        print(f"[STAGE 2/{total}] WARNING: {m} splits lights across many scenes and none are "
              f"extracted - the bake will be SKY-ONLY (dark interiors). Run the fleet light "
              f"merge first for full lighting.", flush=True)
    if lv is not None and not any(
        f.startswith("lights_") for f in os.listdir(dataset) if f.endswith(".json")
    ):
        # portable kit extractor (env-driven drop-in for the beamng warp_viz original);
        # --name is the DATASET folder name, it writes ASSETS/<dataset>/lights_<lv>.json
        run(2, total, "extract lights",
            [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_lights.py"),
             "--level", str(lv), "--name", dsname],
            VIEWER, optional=True)
    else:
        print(f"[STAGE 2/{total}] extract lights: skipped (present or n/a)", flush=True)

    # 3: GPU SH bake (the long stage; skip if a fresh volume2 already exists)
    v2 = os.path.join(out_dir, "volume2.bin")
    if not os.path.isfile(v2):
        # portable kit baker: takes the MAP ID positionally, reads EFT_TARKMAP_ROOT itself
        # (run() passes it) and writes TK/out/<map id>/volume2.*; cwd-independent.
        run(3, total, "bake lighting (GPU)",
            [PY_BAKE, os.path.join(VIEWER, "extraction", "bake", "bake_volume2.py"), m],
            VIEWER)
    else:
        print(f"[STAGE 3/{total}] bake lighting: skipped (volume2 exists)", flush=True)
    # promote volume2.* -> volume.* (assemble reads volume.*). vis.bin is NOT promoted:
    # nothing in the native viewer reads it (legacy web-viewer artifact; provenance audit).
    for src, dst in [("volume2.bin", "volume.bin"), ("volume2.json", "volume.json")]:
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
                  "--name", dsname], VIEWER, optional=True)
        if ok:
            run(5, total, "grass: build grass.bin",
                [PY, "-m", "eft_pipeline.build_grass", "--pack", pack], VIEWER, optional=True)

    # 6: typed gameplay zones (exfils/mines/snipers/doors/loose loot). The extractor writes
    # to tarkmap/out/<map>/gamedata.json and only PRINTS the copy step - do the copy here.
    if run(6, total, "gameplay zones",
           [PY_UNITY, os.path.join(VIEWER, "extraction", "intel", "extract_gamedata.py"), m],
           VIEWER, optional=True):
        gd = os.path.join(out_dir, "gamedata.json")
        if os.path.isfile(gd):
            shutil.copyfile(gd, os.path.join(pack, "gamedata.json"))
            print("  gamedata.json -> pack", flush=True)

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
