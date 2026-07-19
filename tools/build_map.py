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


def find_atlas_exe():
    """Locate the built viewer binary that hosts `bake-nav` (the PORTABLE CPU nav baker). Order:
    EFT_ATLAS_EXE (the viewer hands its own running exe path when it launches a build) > the cargo
    target dirs (a dev build) > beside the repo / dist bundle. Returns a path or None."""
    exe = "atlas.exe" if os.name == "nt" else "atlas"
    env = os.environ.get("EFT_ATLAS_EXE")
    if env and os.path.isfile(env):
        return env
    for c in (os.path.join(VIEWER, "target", "release", exe),
              os.path.join(VIEWER, "target", "debug", exe),
              os.path.join(VIEWER, exe),
              os.path.join(VIEWER, "dist", exe),
              os.path.join(HERE, exe)):
        if os.path.isfile(c):
            return c
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    self_contained = "--self-contained" in sys.argv
    sc_flag = ["--self-contained"] if self_contained else []
    # FORCED REFRESH (menu UPDATE): after an EFT patch the plain build would SKIP extraction
    # (stage 1 sees the old scene.json), reuse stale lights/SH/nav, then stamp the pack with the
    # CURRENT fingerprint -> the menu flips to READY over stale geometry (release-blocker). --force
    # / EFT_FORCE_REBUILD invalidates every game-derived cache below so all stages re-run against
    # the live game files before the stamp. It deletes the CACHE GATES (scene.json, volume/nav/glb,
    # light sidecars), never the big mesh/texture exports or the existing .eftpack, so a failed
    # re-extract can't leave the user with nothing (the old pack stays playable until stage 4).
    force = "--force" in sys.argv or os.environ.get("EFT_FORCE_REBUILD", "").strip() == "1"
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
    if force and not dry:
        # Invalidate the game-derived cache gates so stages 1/2/3/8 re-run instead of "exists ->
        # skip". Best-effort: a missing file is fine; a locked one just means that stage re-runs
        # anyway on its own exists-check (which now also honors `force`).
        print(f"[BUILD] forced refresh: invalidating stale game-derived caches for {m}", flush=True)
        stale = [os.path.join(dataset, "scene.json"),
                 os.path.join(out_dir, "volume2.bin"), os.path.join(out_dir, "volume.bin"),
                 os.path.join(out_dir, "nav.bin"), os.path.join(out_dir, "instanced_raw.glb")]
        if os.path.isdir(dataset):
            stale += [os.path.join(dataset, f) for f in os.listdir(dataset)
                      if f.startswith("lights_") and f.endswith(".json")]
        for s in stale:
            try:
                if os.path.isfile(s):
                    os.remove(s)
            except OSError as e:
                print(f"  [force] could not remove {s}: {e}", flush=True)
    if dry:
        # --self-contained is noted on the stages it changes (assemble + grass emit
        # pack-relative, copied-in textures/sidecars instead of absolute references).
        sc_note = " (self-contained)" if self_contained else ""
        for i, name in enumerate(
            ["check dataset", "extract lights", "bake lighting (GPU)",
             "assemble pack" + sc_note, "grass" + sc_note,
             "gameplay zones", "item icons", "bake nav grid (CPU)",
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
    if force or not os.path.isfile(os.path.join(dataset, "scene.json")):
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
    if lv is not None and (force or not any(
        f.startswith("lights_") for f in os.listdir(dataset) if f.endswith(".json")
    )):
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
    if force or not os.path.isfile(v2):
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

    # 8: NAV GRID for the viewer's in-process CPU pathfinding. Baked by the PORTABLE Rust baker
    #    (`atlas bake-nav <pack>`) directly from the assembled pack's world triangles via a CPU BVH
    #    raycast — no CUDA, no instanced_raw.glb. It runs on ANY machine (AMD/NVIDIA/no-GPU), so
    #    routing is produced BY DEFAULT, and writes nav.bin/nav.json/nav_door.bin straight into the
    #    pack (same layout the old CUDA bake_nav.py emitted, same tuning constants -> same quality).
    #    Only skipped when no built viewer exe can be found (a kit without a compiled binary).
    atlas_exe = find_atlas_exe()
    if atlas_exe:
        run(8, total, "bake nav grid (CPU)",
            [atlas_exe, "bake-nav", pack], VIEWER, optional=True)
    else:
        print(f"[STAGE 8/{total}] nav: skipped - viewer exe not found. Build it "
              f"(`cargo build --release`) or set EFT_ATLAS_EXE, then rebuild to enable routing.",
              flush=True)
    if os.path.isfile(os.path.join(pack, "nav.bin")):
        print("  nav grid: baked into pack (in-process CPU routing enabled)", flush=True)
    else:
        print("  nav grid: none (routing disabled for this map until the baker runs)", flush=True)

    # 9: stamp the game fingerprint (menu update detection)
    run(9, total, "stamp fingerprint",
        [PY, os.path.join(HERE, "stamp_fingerprint.py"), pack], VIEWER)

    # Post-build storage dedup: a texture shared by several maps is byte-identical in each dataset's
    # tex/ (source-identity naming), so it's stored once per map = pure waste. Hardlink the copies to
    # a single physical file -- transparent (files stay in place) + lossless (no visual/behaviour
    # change). Best-effort; never fail the build over it. (Re-run each build to re-link overwrites.)
    try:
        env = dict(os.environ, EFT_ASSETS_ROOT=ASSETS)
        subprocess.call([sys.executable, os.path.join(HERE, "dedup_textures.py")], env=env)
    except Exception as e:
        print(f"  [dedup] skipped: {e}", flush=True)

    # Lighting completeness (finding 3a): a map we KNOW ships realtime lights (in LIGHT_LEVELS, or
    # the multi-scene streets/ground_zero) that produced neither a light sidecar NOR an SH bake will
    # render with dark/flat interiors. Don't hide that behind a clean [BUILD OK] - surface it so the
    # menu log makes the gap obvious (the pack is still valid geometry, so exit stays 0).
    expects_light = (m in LIGHT_LEVELS) or (m in ("streets", "ground_zero"))
    have_lights = os.path.isdir(dataset) and any(
        f.startswith("lights_") and f.endswith(".json") for f in os.listdir(dataset))
    have_sh = os.path.isfile(os.path.join(out_dir, "volume.bin"))
    if expects_light and not (have_lights or have_sh):
        print(f"[BUILD WARN] no lighting for {m}: no *_Light extract and no SH bake - interiors "
              f"will be dark/flat. Run the light extract and/or the CUDA SH bake "
              f"(see extraction/README.md) then rebuild.", flush=True)
        print(f"[BUILD OK] pack ready (WARNING: no lighting for {m})", flush=True)
    else:
        print("[BUILD OK] pack ready", flush=True)


if __name__ == "__main__":
    main()
