"""One-command map pack builder for the viewer's start menu.

Runs the full pipeline for a map. If the DATASET is missing (<EFT_ASSETS_ROOT>/<dataset>/scene.json)
it first runs the ONE-TIME full game extraction inline (the long step - game must be CLOSED), then
assembles the pack. Levels for that extraction come from the map config's source.levels.
Stages print `[STAGE i/N] name` markers and stream child output unbuffered so the menu's
progress panel can display them live. Exit 0 = pack ready (stamped). ASCII output only.

Usage: python tools/build_map.py <map> [--dry-run] [--self-contained]
  --self-contained: redistribution PR3 — passed through to assemble_bevy + build_grass so
  the emitted pack copies its textures/sidecars in and references them pack-relative.
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

# Robust output: a child pipeline stage can emit a non-ASCII byte (a material/mesh name — EFT
# assets include Cyrillic), and our stdout is a cp1252 pipe/file. The BUNDLED embeddable Python
# IGNORES PYTHONIOENCODING (its ._pth disables env-var handling), so the child's non-ASCII survived
# our ascii-replace read as U+FFFD and crashed the build printing it (UnicodeEncodeError) mid-
# assemble, before stages 5-9 (gamedata/POI, icons, fingerprint) could run. Force UTF-8 (+replace)
# on our own streams so printing any line is always safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
    # Factory 1.0 rework (the shipped roster's "Factory"): the live lights are in
    # Factory_Rework_Day_Light.unity == BuildSettings level 526 (night bank = 541). Without this
    # entry stage 2 skips light extraction and the SH bake is SKY-ONLY (dark, unlit interiors).
    "factory_rework": 526,
    "customs": 13,
    "woods": 167,
    "shoreline": 41,
    "reserve": 146,
    "labs": 114,
    "labyrinth": 551,
}
INDOOR_NO_GRASS = {"factory", "factory_rework", "labs", "labyrinth"}


def run(stage, total, name, cmd, cwd, optional=False):
    print(f"[STAGE {stage}/{total}] {name}", flush=True)
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    # PYTHONUTF8=1 asks children to emit UTF-8 (respected by the venv Python; the embeddable one
    # ignores it, but our own stdout is UTF-8 above and we read the child as UTF-8 below, so a
    # non-ASCII line is handled either way instead of crashing the build).
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    # pass the contract values as-is (TK = the maps/+out/ dir, ASSETS = the datasets dir)
    env.setdefault("EFT_TARKMAP_ROOT", TK)
    env.setdefault("EFT_ASSETS_ROOT", ASSETS)
    p = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    for line in p.stdout:
        print("  " + line.rstrip(), flush=True)
    rc = p.wait()
    dt = time.time() - t0
    # Machine-readable per-phase timing (captured by the viewer to weight the ETA + spot slow stages).
    print(f"[TIMING] {name}={dt:.1f}", flush=True)
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


def dataset_levels(m):
    """Comma-separated Unity level indices for the map (map config's source.levels) — the input to
    the one-time full extraction. Empty string when unreadable (caller errors clearly)."""
    for p in (os.path.join(TK, "maps", m, "config.json"),
              os.path.join(VIEWER, "extraction", "maps", m, "config.json")):
        if os.path.isfile(p):
            try:
                lv = json.load(open(p, encoding="utf-8"))["source"]["levels"]
                return ",".join(str(int(x)) for x in lv)
            except Exception as e:
                print(f"[BUILD] WARNING: cannot read source.levels from {p} ({e})", flush=True)
                return ""
    return ""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    self_contained = "--self-contained" in sys.argv
    sc_flag = ["--self-contained"] if self_contained else []
    # --alllod (or EFT_ALLLOD=1): keep EVERY LOD level in the dataset + pack (instead of the default
    # LOD0-only resolve) so the viewer can offer a forced-LOD selector. ~47% bigger; opt-in. NOTE:
    # only takes effect on a FRESH extraction -- delete the existing LOD0 dataset first, else the
    # stage-1 "dataset exists" check reuses the LOD0 dataset.
    all_lod = "--alllod" in sys.argv or os.environ.get("EFT_ALLLOD", "").strip() == "1"
    alllod_extract = ["--alllod"] if all_lod else []
    keeplods_flag = ["--keep-lods"] if all_lod else []
    if not args:
        print("usage: build_map.py <map> [--dry-run] [--self-contained]")
        sys.exit(2)
    m = args[0]
    dsname = dataset_name(m)
    dataset = os.path.join(ASSETS, dsname)
    # out/ stays keyed by MAP ID: bake_volume2 / extract_gamedata / assemble_bevy all write
    # and read TK/out/<map id> (they resolve the dataset via the map config themselves).
    out_dir = os.path.join(TK, "out", m)
    pack = os.path.join(VIEWER, "packs", f"{m}.eftpack")
    total = 9

    print(f"[BUILD] map={m} dataset={dsname} dataset_dir={dataset}", flush=True)
    if dry:
        # --self-contained is noted on the stages it changes (assemble + grass emit
        # pack-relative, copied-in textures/sidecars instead of absolute references).
        sc_note = " (self-contained)" if self_contained else ""
        for i, name in enumerate(
            ["check dataset", "extract lights", "bake lighting (GPU)",
             "assemble pack" + sc_note, "grass" + sc_note,
             "gameplay zones", "item icons", "bake nav grid (GPU)",
             "stamp fingerprint"], 1):
            print(f"[STAGE {i}/{total}] {name}", flush=True)
            time.sleep(0.6)
            print(f"[STAGE {i}/{total}] {name}: done (0s)", flush=True)
        print("[BUILD OK] dry run", flush=True)
        return

    # 1: dataset present? If not, run the ONE-TIME full game extraction inline (the long step:
    #    game/launcher must be CLOSED, tens of minutes to hours, 1-6 GB on disk). Folded into BUILD
    #    so one click goes from "no data" to a playable pack. Resumable - a re-run skips already
    #    exported meshes/textures.
    print(f"[STAGE 1/{total}] check dataset", flush=True)
    if not os.path.isfile(os.path.join(dataset, "scene.json")):
        levels = dataset_levels(m)
        if not levels:
            print(f"[BUILD FAILED] no dataset at {dataset} and no source.levels in the map config "
                  f"- cannot auto-extract (see extraction/README.md)", flush=True)
            sys.exit(3)
        print(f"[STAGE 1/{total}] no dataset yet - running the ONE-TIME full extraction. CLOSE the "
              f"game and launcher first (file locks). This can take a long time.", flush=True)
        # extract_parallel splits the levels across cores (reusing the unchanged eft_extract_v2 per
        # chunk) then merges — big maps go multi-core. EFT_JOBS=1 forces the plain serial extractor.
        run(1, total, "extract dataset (geometry + textures)",
            [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "extract_parallel.py"),
             "--levels", levels, "--name", dsname] + alllod_extract, VIEWER)
        if m not in INDOOR_NO_GRASS:
            run(1, total, "extract grass density",
                [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_grass.py"),
                 "--levels", levels, "--name", dsname], VIEWER, optional=True)
        if not os.path.isfile(os.path.join(dataset, "scene.json")):
            print(f"[BUILD FAILED] extraction finished but no scene.json at {dataset} - check the "
                  f"log above (is UnityPy installed for EFT_PY_UNITY? is EFT_GAME_DATA correct and "
                  f"the game closed?)", flush=True)
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
        # OPTIONAL: the SH bake needs an NVIDIA CUDA GPU + warp-lang. Without them (or on any bake
        # error) the build continues and the pack renders with flat ambient light (README: "No CUDA
        # GPU? Skip this step - the viewer still runs"). A mandatory bake wrongly failed the whole
        # build for anyone without the GPU bake deps.
        run(3, total, "bake lighting (GPU)",
            [PY_BAKE, os.path.join(VIEWER, "extraction", "bake", "bake_volume2.py"), m],
            VIEWER, optional=True)
    else:
        print(f"[STAGE 3/{total}] bake lighting: skipped (volume2 exists)", flush=True)
    # promote volume2.* -> volume.* (assemble reads volume.*). vis.bin is NOT promoted:
    # nothing in the native viewer reads it (legacy web-viewer artifact; provenance audit).
    for src, dst in [("volume2.bin", "volume.bin"), ("volume2.json", "volume.json")]:
        s = os.path.join(out_dir, src)
        if os.path.isfile(s):
            shutil.copyfile(s, os.path.join(out_dir, dst))

    # 4: assemble the pack (atomic; auto-ships loot/tasks/grade sidecars)
    run(4, total, "assemble pack",
        [PY, "-m", "eft_pipeline.assemble_bevy", m] + sc_flag + keeplods_flag, VIEWER)

    # 5: grass (outdoor maps)
    if m in INDOOR_NO_GRASS:
        print(f"[STAGE 5/{total}] grass: skipped (indoor map)", flush=True)
    else:
        # The stage-1 inline extraction already extracts grass density on a FRESH build, so don't
        # scan the (huge, Streets = 217-level) terrain bundle a second time here - just pack it.
        tl = os.path.join(dataset, "terrain_layers")
        have_grids = os.path.isdir(tl) and any(
            f.startswith("grass_density_") and f.endswith(".bin") for f in os.listdir(tl))
        if have_grids:
            print(f"[STAGE 5/{total}] grass: density grids already present - skip re-extract", flush=True)
            ok = True
        else:
            gl = dataset_levels(m)
            grass_cmd = [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_grass.py"),
                         "--name", dsname]
            if gl:
                # pass the level list so the extractor finds the terrain bundle (without it, it
                # auto-detects over an empty list and silently skips -> no grass on fresh datasets).
                grass_cmd += ["--levels", gl]
            ok = run(5, total, "grass: extract density grids", grass_cmd, VIEWER, optional=True)
        if ok:
            run(5, total, "grass: build grass.bin",
                [PY, "-m", "eft_pipeline.build_grass", "--pack", pack] + sc_flag,
                VIEWER, optional=True)

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

    # 8: NAV GRID for the viewer's in-process CPU pathfinding (bake_nav.py, GPU). Baked from
    #    instanced_raw.glb into TK/out/<map>/nav.* then COPIED into the pack, so the viewer routes on
    #    the CPU with no server. OPTIONAL: a build on a non-CUDA machine (or without the glb) just skips
    #    it and that map won't route until baked on a GPU box — the pack is still valid. Re-bakes when
    #    nav.bin is missing or older than the glb (keeps routing in sync with the geometry).
    nav_bin = os.path.join(out_dir, "nav.bin")
    glb = os.path.join(out_dir, "instanced_raw.glb")
    need_nav = (not os.path.isfile(nav_bin)) or (
        os.path.isfile(glb) and os.path.getmtime(glb) > os.path.getmtime(nav_bin))
    if need_nav:
        if not os.path.isfile(glb):
            # foundation glb (instance-preserving) — bake_nav raycasts against it. Plain python.
            run(8, total, "nav: assemble instanced glb",
                [PY, os.path.join(TK, "assemble_instanced.py"), m], TK, optional=True)
        if os.path.isfile(glb):
            run(8, total, "nav: bake grid (GPU)",
                [PY_BAKE, os.path.join(TK, "bake_nav.py"), m], TK, optional=True)
        else:
            print(f"[STAGE 8/{total}] nav: skipped (no instanced_raw.glb; run assemble_instanced on a "
                  f"build box) - this map won't route yet", flush=True)
    else:
        print(f"[STAGE 8/{total}] nav: skipped (fresh nav.bin exists)", flush=True)
    # copy whatever nav files exist into the pack (grid + door/edge masks + params)
    ncopied = 0
    for f in ("nav.bin", "nav_door.bin", "nav_blk.bin", "nav.json"):
        s = os.path.join(out_dir, f)
        if os.path.isfile(s):
            shutil.copyfile(s, os.path.join(pack, f))
            ncopied += 1
    if ncopied:
        print(f"  nav grid -> pack ({ncopied} files)", flush=True)
    else:
        print("  nav grid: none packed (routing disabled for this map until baked on a GPU box)",
              flush=True)

    # 9: stamp the game fingerprint (menu update detection)
    run(9, total, "stamp fingerprint",
        [PY, os.path.join(HERE, "stamp_fingerprint.py"), pack], VIEWER)

    print("[BUILD OK] pack ready", flush=True)


if __name__ == "__main__":
    main()
