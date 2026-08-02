"""One-command map pack builder for the viewer's start menu.

Runs the full pipeline for a map. If the DATASET is missing (<EFT_ASSETS_ROOT>/<dataset>/scene.json)
it first runs the ONE-TIME full game extraction inline (the long step - game must be CLOSED), then
assembles the pack. Levels for that extraction come from the map config's source.levels.
Stages print `[STAGE i/N] name` markers and stream child output unbuffered so the menu's
progress panel can display them live. Exit 0 = pack ready (stamped). ASCII output only.

Usage: python tools/build_map.py <map> [--dry-run] [--self-contained]
  --self-contained: redistribution PR3 — passed through to assemble_bevy + build_grass so
  the emitted pack copies its textures/sidecars in and references them pack-relative.
Env (contract per README.md; unset -> legacy dev-machine defaults):
  EFT_TARKMAP_ROOT = the dir CONTAINING maps/ and out/ (a "tarkmap dir")
  EFT_ASSETS_ROOT  = the datasets dir (default: <EFT_TARKMAP_ROOT>/../eft_assets)
  EFT_PY_UNITY / EFT_PY_BAKE = UnityPy / CUDA-warp pythons (default: legacy anaconda
  interpreters when present on this machine, else this python)
"""

import json
import os
import re
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
# Default: a sibling `tarkmap` beside EFT_ASSETS_ROOT — the SAME derivation the menu uses — so a
# bare CLI `build_map.py <map>` gets working stage-6 gamedata instead of silently pointing TK at
# the dead legacy dev path (which made `extract_gamedata` exit 1 and packs ship without doors/
# exfil/zone intel; the copy at stage 6 also reads <TK>/out/<map>/). Legacy path stays as the
# last-resort fallback when neither env is set.
TK = os.environ.get("EFT_TARKMAP_ROOT")
if not TK:
    _ar = os.environ.get("EFT_ASSETS_ROOT")
    TK = (os.path.normpath(os.path.join(_ar, os.pardir, "tarkmap")) if _ar
          else r"C:\Users\user\beamng_blender_pipeline\tarkmap")
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

# The `*_Light` BuildSettings scene indices per map are no longer hardcoded here -- they are DERIVED
# by tools/gen_maps.py and shipped in extraction/maps/manifest.json ("light_levels", a LIST so
# streets/ground_zero -- which split lighting across many district scenes -- get FULL lighting, not
# the single-scene the old scalar table allowed). build_map reads that list; if the map isn't in the
# manifest yet (a brand-new location a builder is adding) it falls back to deriving the list straight
# from the live BuildSettings via gen_maps. Lights stay OPTIONAL -- an empty list just skips the
# stage. There is no hardcoded INDOOR_NO_GRASS set anymore either: grass is data-driven (a map has
# grass iff its dataset actually yields density grids), so indoor/no-terrain maps are skipped by
# nature (see stage 5).
MANIFEST_PATH = os.path.join(VIEWER, "extraction", "maps", "manifest.json")


def _manifest_maps():
    """{id: entry} from the shipped roster manifest, or {} if unreadable (fallback derivation)."""
    try:
        return {m["id"]: m for m in json.load(
            open(MANIFEST_PATH, encoding="utf-8")).get("maps", [])}
    except Exception as e:
        print(f"[BUILD] note: could not read maps manifest ({e}) - deriving lights from "
              f"BuildSettings", flush=True)
        return {}


def _config_unity_location(m):
    """The map's game location folder (source.unity_location) from its config, or None -- the join
    key for the manifest-miss light derivation. Workspace config wins over the kit copy."""
    for p in (os.path.join(TK, "maps", m, "config.json"),
              os.path.join(VIEWER, "extraction", "maps", m, "config.json")):
        if os.path.isfile(p):
            try:
                return json.load(open(p, encoding="utf-8"))["source"].get("unity_location")
            except Exception:
                return None
    return None


def light_levels_for(m):
    """List of `*_Light` BuildSettings level indices to extract for map m. Manifest first (the
    shipped, committed roster); fall back to deriving from the LIVE BuildSettings for a map not yet
    in the manifest. Returns [] when nothing can be found -- lights are OPTIONAL, the build never
    fails on this."""
    entry = _manifest_maps().get(m)
    if entry is not None and entry.get("light_levels") is not None:
        return [int(x) for x in entry["light_levels"]]
    folder = _config_unity_location(m)
    if not folder:
        print(f"[BUILD] note: {m} not in the manifest and no source.unity_location in its config - "
              f"lights will be skipped (optional)", flush=True)
        return []
    try:
        out = subprocess.check_output(
            [PY_UNITY, os.path.join(HERE, "gen_maps.py"), "--lights-for", folder],
            text=True, encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL)
        levels = json.loads(out.strip().splitlines()[-1])
        print(f"[BUILD] derived light levels for {m} from BuildSettings folder '{folder}': "
              f"{levels}", flush=True)
        return [int(x) for x in levels]
    except Exception as e:
        print(f"[BUILD] note: could not derive lights for {m} ({e}) - skipped (optional)", flush=True)
        return []


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


def _config_levels(m):
    """The map config's hand-curated source.levels (fallback only). [] when unreadable."""
    for p in (os.path.join(TK, "maps", m, "config.json"),
              os.path.join(VIEWER, "extraction", "maps", m, "config.json")):
        if os.path.isfile(p):
            try:
                return [int(x) for x in json.load(open(p, encoding="utf-8"))["source"]["levels"]]
            except Exception as e:
                print(f"[BUILD] WARNING: cannot read source.levels from {p} ({e})", flush=True)
                return []
    return []


def dataset_levels(m):
    """Comma-separated Unity level indices for the map — the input to the one-time full extraction.
    DERIVED LIVE from BuildSettings (every non-service scene in the map's location folder) rather than
    the hand-curated config.source.levels, which drifts as the game adds scenes: reserve's config was
    missing level116 (Reserve_Base_DesignStuff -> vehicles/loot/props), so crates resting on that
    geometry floated. The derived list is the game's own truth and self-heals across updates + all maps.
    Falls back to (or UNIONS with) the config list if derivation is unavailable, so a config that
    intentionally adds an off-folder level still contributes and we never regress to fewer levels."""
    cfg = _config_levels(m)
    folder = _config_unity_location(m)
    derived = []
    if folder:
        try:
            out = subprocess.check_output(
                [PY_UNITY, os.path.join(HERE, "gen_maps.py"), "--levels-for", folder],
                text=True, stderr=subprocess.DEVNULL)
            derived = [int(x) for x in json.loads(out.strip().splitlines()[-1])]
        except Exception as e:
            print(f"[BUILD] note: could not derive levels for {m} from BuildSettings ({e}) - "
                  f"using config.source.levels", flush=True)
    levels = sorted(set(derived) | set(cfg))          # union: never fewer than the config had
    if derived and set(derived) - set(cfg):
        print(f"[BUILD] levels derived from BuildSettings folder '{folder}': +{sorted(set(derived)-set(cfg))} "
              f"beyond config (total {len(levels)})", flush=True)
    return ",".join(str(x) for x in levels)


def derive_sea_level(dataset):
    """GAME-TRUTH ocean height, derived from the dataset's scene.json at build time — no authored
    per-map constants. EFT coastal maps DO ship their ocean surface as geometry (shoreline: 14
    tiled `Shoreline_Sea_Water_*` planes, role='water', all at one height), so the sea is found
    structurally: among water-role, non-decal (puddles ride 'Decal/Water ...' shaders), FLAT
    (world-Y span <= 2 m, which excludes river cascades) instances, bin by world height (0.1 m);
    the SEA is the bin whose union XZ footprint is MAP-SCALE — >= 10% of the scene's own
    translation AABB. Lakes/canals/ponds are orders of magnitude smaller and never qualify, so
    inland maps return None (no synthesized sea). Extents are measured in raw Unity world space
    (the viewer's X-flip conjugation preserves areas and Y, so no bridge is needed here).
    Returns the sea height + 0.05 m — the viewer's horizon quad rides just above the shipped
    tiles; both draw with the same deep-water shading, so the overlap cannot z-fight visibly."""
    scene = os.path.join(dataset, "scene.json")
    if not os.path.isfile(scene):
        return None
    d = json.load(open(scene, encoding="utf-8"))
    inst = d["instances"] if isinstance(d, dict) else d
    if not inst:
        return None

    mesh_aabb = {}

    def aabb_of(mesh):
        """Local-space AABB of a dataset OBJ (v-lines only); None when unreadable."""
        if mesh not in mesh_aabb:
            box = None
            try:
                lo = [float("inf")] * 3
                hi = [float("-inf")] * 3
                with open(os.path.join(dataset, "meshes", mesh), encoding="utf-8",
                          errors="replace") as f:
                    for line in f:
                        if line.startswith("v "):
                            p = line.split()
                            for k in range(3):
                                v = float(p[k + 1])
                                lo[k] = min(lo[k], v)
                                hi[k] = max(hi[k], v)
                if lo[0] <= hi[0]:
                    box = (lo, hi)
            except Exception:
                box = None
            mesh_aabb[mesh] = box
        return mesh_aabb[mesh]

    sx = [it["m"][3] for it in inst]
    sz = [it["m"][11] for it in inst]
    scene_area = max(1.0, (max(sx) - min(sx)) * (max(sz) - min(sz)))
    bins = {}                                     # y-bin -> [minx, maxx, minz, maxz, y]
    def is_water_sub(sb):
        """True water surface: role water AND a water shader (or untextured water, sh=None, the
        shoreline sea tiles). Excludes puddle DECALS and role-water tagged 'Standard'-shader
        collision/occluder PROXIES (streets ships a map-wide TEMP_GROUND_COLIDER cube like that
        — same proxy signal the structural culls use)."""
        if sb.get("role") != "water":
            return False
        sh = sb.get("sh")
        if sh is None:
            return True
        return "water" in sh.lower() and "decal" not in sh.lower()

    for it in inst:
        if not any(is_water_sub(sb) for sb in (it.get("subs") or [])):
            continue
        box = aabb_of(it.get("mesh", ""))
        if box is None:
            continue
        mm = it["m"]
        wy = [float("inf"), float("-inf")]
        wx = [float("inf"), float("-inf")]
        wz = [float("inf"), float("-inf")]
        for cx in (box[0][0], box[1][0]):
            for cy in (box[0][1], box[1][1]):
                for cz in (box[0][2], box[1][2]):
                    x = mm[0] * cx + mm[1] * cy + mm[2] * cz + mm[3]
                    y = mm[4] * cx + mm[5] * cy + mm[6] * cz + mm[7]
                    z = mm[8] * cx + mm[9] * cy + mm[10] * cz + mm[11]
                    wx = [min(wx[0], x), max(wx[1], x)]
                    wy = [min(wy[0], y), max(wy[1], y)]
                    wz = [min(wz[0], z), max(wz[1], z)]
        if wy[1] - wy[0] > 2.0:                   # sloped/cascade water is never the sea surface
            continue
        y_surf = (wy[0] + wy[1]) * 0.5
        key = round(y_surf * 10.0)
        b = bins.setdefault(key, [wx[0], wx[1], wz[0], wz[1], y_surf])
        b[0] = min(b[0], wx[0])
        b[1] = max(b[1], wx[1])
        b[2] = min(b[2], wz[0])
        b[3] = max(b[3], wz[1])
        b[4] = max(b[4], y_surf)
    # QUALIFY THE SEA STRUCTURALLY, not by size.
    #
    # Size alone cannot separate an ocean from a lake: woods' lake clears any sane area bar (it
    # derived a 7.454 m "sea level" and the viewer flooded the whole map with a horizon quad).
    # The real, geometric difference is CONTAINMENT:
    #   * an ocean is UNBOUNDED — it runs out to the edge of the built scene and past it, so its
    #     footprint reaches the scene's own XZ boundary;
    #   * a lake/pond/river is ENCLOSED — terrain wraps all the way around it, so its footprint
    #     stops well short of the boundary on every side.
    # So: require the water to touch the scene AABB on at least one side. That is derived from the
    # game's own geometry, needs no per-map constant, and does not care how big the lake is.
    #
    # `scene_*` come from instance TRANSLATIONS, so sea tiles contribute to the bounds they are
    # then tested against. That is not circular in the failing direction: a sea that defines the
    # edge trivially touches it (correct), while a lake sitting inside the terrain footprint does
    # not (also correct).
    sx_min, sx_max = min(sx), max(sx)
    sz_min, sz_max = min(sz), max(sz)
    span_x = max(1.0, sx_max - sx_min)
    span_z = max(1.0, sz_max - sz_min)
    EDGE_FRAC = 0.02          # "touching" = within 2% of the scene span of that side
    MIN_AREA_FRAC = 0.10      # still require map-scale: a puddle at the map edge is not an ocean

    best = None
    for b in bins.values():
        area = (b[1] - b[0]) * (b[3] - b[2])
        if area < MIN_AREA_FRAC * scene_area:
            continue
        touches = (
            b[0] <= sx_min + EDGE_FRAC * span_x      # reaches the -X edge
            or b[1] >= sx_max - EDGE_FRAC * span_x   # +X
            or b[2] <= sz_min + EDGE_FRAC * span_z   # -Z
            or b[3] >= sz_max - EDGE_FRAC * span_z   # +Z
        )
        if not touches:
            continue                                  # enclosed water: a lake, not the sea
        if best is None or area > best[0]:
            best = (area, b[4])
    if best is None:
        return None
    return round(best[1] + 0.05, 3)


def finalize_pack_manifest(pack, dataset):
    """Reconcile the assembled manifest with what the pack + dataset ACTUALLY contain.

    assemble_bevy writes the sidecar table at stage 4, but several producers run AFTER it: the
    portable SH bake writes volume.bin/volume.json/volume_valid.bin straight into the pack, and a
    light sidecar can be extracted (or refreshed) later. Whatever the manifest recorded at assemble
    time is then permanently wrong, and every symptom is silent:

      * volume sidecars null  -> load_sh_volume finds nothing, real_volume=false, and the map
        renders with NO baked GI. Observed on factory_rework: the pack shipped a valid 1.6 MB
        indirect-only volume that the manifest never referenced, so interiors were near-black.
      * a light sidecar missing from lightsAll -> that whole bank never loads, and any power switch
        controlling it resolves to zero light groups ("Power (no lights)"). Observed on interchange,
        whose entire switch-controlled bank lived in an unreferenced sidecar.

    So this runs LAST and points every sidecar at the file that exists, preferring the pack's own
    copy. Absolute in-place references (the non-self-contained default) are left alone when they
    still resolve — this only repairs what is null, missing, or superseded by a pack-local file.
    """
    import glob as _glob
    mp = os.path.join(pack, "manifest.json")
    if not os.path.isfile(mp):
        return
    try:
        with open(mp, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        print(f"  manifest: WARNING could not reconcile sidecars ({e})", flush=True)
        return
    sc = man.setdefault("sidecars", {})
    fixed = []

    def _pack_has(name):
        return os.path.isfile(os.path.join(pack, name))

    # --- SH irradiance volume + probe validity -------------------------------------------------
    # volume_valid.bin is read by FIXED NAME (render/gpu_driven.rs), so recording it is
    # documentation rather than plumbing — but a null field next to a present file reads as "this
    # map has no validity data", which is exactly the confusion to avoid.
    # THE PACK'S OWN VOLUME ALWAYS WINS, and the reason is correctness rather than tidiness: the
    # bake writes volume.bin, volume.json and volume_valid.bin together, so those three AGREE about
    # the probe grid. Point `volume` at a copy from a different bake and the validity mask is
    # applied to the wrong grid — and it is not rejected, because the probe COUNT can match while
    # the origin and spacing do not. Interchange was in exactly that state: it loaded a build-tree
    # volume from 2026-07-19 while the pack's own volume_valid.bin came from the 07-27 re-bake,
    # same 401x13x302 count, origin off by 7.6 m in Z and spacing off by 0.05 m/cell — about 20 m
    # of drift by the far edge, masking 677,882 probes against geometry that is not where the mask
    # thinks it is. The visible result was a mall interior that read flat and unlit.
    #
    # An outside reference is by definition the one that can go stale; the pack is the shipping
    # unit. So: if the pack carries the file, use the pack's.
    for key, name in (("volume", "volume.bin"), ("volumeMeta", "volume.json"),
                      ("volumeVis", "volume_valid.bin")):
        # Compare the WHOLE value — an absolute path into the build tree ends in the same file
        # name as the pack's own, so a basename test would silently keep the stale one.
        if _pack_has(name) and sc.get(key) != name:
            sc[key] = name
            fixed.append(f"{key}={name}")

    # --- realtime light sidecars ---------------------------------------------------------------
    # Union of what the manifest lists, what the pack carries and what the dataset produced, keyed
    # by file name (the one thing an absolute path and a pack-relative one share). A pack-local
    # copy always wins so the pack stays movable.
    listed = list(sc.get("lightsAll") or ([sc["lights"]] if sc.get("lights") else []))
    by_name = {}
    for p in listed:
        by_name.setdefault(os.path.basename(p), p)
    for p in sorted(_glob.glob(os.path.join(dataset, "lights_*.json"))):
        if not p.endswith("_all.json"):
            by_name.setdefault(os.path.basename(p), p.replace("\\", "/"))
    for name in sorted(os.path.basename(p) for p in _glob.glob(os.path.join(pack, "lights_*.json"))):
        by_name[name] = name                    # pack-local copy supersedes any outside reference
    merged = [by_name[k] for k in sorted(by_name)]
    if merged and merged != listed:
        sc["lightsAll"] = merged
        if not sc.get("lights") or os.path.basename(str(sc["lights"])) not in by_name:
            sc["lights"] = merged[0]
        fixed.append(f"lightsAll={len(merged)} sidecar(s)")

    if not fixed:
        return
    try:
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(man, f)
        print(f"  manifest: reconciled sidecars -> {', '.join(fixed)}", flush=True)
    except OSError as e:
        print(f"  manifest: WARNING could not write reconciled sidecars ({e})", flush=True)


def verify_pack_lighting(pack):
    """Assert the pack ships a COHERENT lighting set, and say plainly what is wrong if not.

    Every symptom in this area is silent: nothing crashes, nothing logs an error, the map just
    renders wrong. Three failures have actually happened, so each gets an explicit check:

      * volume sidecars null while the pack holds a bake  -> no baked GI at all (factory_rework
        rendered near-black interiors this way).
      * volume from one bake, volume_valid.bin from another -> the validity mask is applied to a
        grid it was not computed for. NOT caught by any size check, because the probe COUNT can
        match while the origin and spacing do not; interchange drifted ~20 m at the far edge and
        masked 677,882 probes against geometry that was not there.
      * no volume_valid.bin -> every probe treated as valid, so light leaks through walls.

    Warn-only: a pack with imperfect lighting is still valid geometry, and failing the build over
    it would be worse than shipping it with a loud note.
    """
    mp = os.path.join(pack, "manifest.json")
    try:
        with open(mp, encoding="utf-8") as f:
            sc = json.load(f).get("sidecars", {})
    except (OSError, ValueError):
        return
    has = lambda n: os.path.isfile(os.path.join(pack, n))
    if not has("volume.bin"):
        print("[BUILD WARN] lighting: no SH volume in the pack - the map will render with the flat "
              "realtime fallback. Build the viewer (`cargo build --release`) so `atlas bake-sh` can "
              "run, then rebuild.", flush=True)
        return
    problems = []
    for key, name in (("volume", "volume.bin"), ("volumeMeta", "volume.json"),
                      ("volumeVis", "volume_valid.bin")):
        if has(name) and sc.get(key) != name:
            problems.append(f"manifest.{key} is {sc.get(key)!r}, not the pack's own {name}")
    if not has("volume_valid.bin"):
        problems.append("no volume_valid.bin - probe validity missing, so light leaks through "
                        "geometry (this pack predates validity; a rebuild produces it)")
    else:
        # One byte per probe. A length that disagrees with volume.json's dims means the two came
        # from different bakes -- the exact mismatch the viewer cannot detect.
        try:
            with open(os.path.join(pack, "volume.json"), encoding="utf-8") as f:
                meta = json.load(f)
            dims = meta.get("dims") or []
            want = dims[0] * dims[1] * dims[2] if len(dims) == 3 else None
            got = os.path.getsize(os.path.join(pack, "volume_valid.bin"))
            if want and got != want:
                problems.append(f"volume_valid.bin is {got} bytes but volume.json describes "
                                f"{want} probes - the two are from different bakes")
            if meta.get("direct") is not False:
                problems.append("volume.json has no \"direct\": false - this is a FULL bake, so the "
                                "viewer disables realtime practicals (EFT_BAKE=warp does this; the "
                                "default `atlas bake-sh --indirect-only` is what ships the "
                                "direct/indirect split)")
        except (OSError, ValueError, IndexError, TypeError) as e:
            problems.append(f"could not compare volume.json against volume_valid.bin ({e})")
    if problems:
        print("[BUILD WARN] lighting is incoherent in this pack:", flush=True)
        for p in problems:
            print(f"[BUILD WARN]   - {p}", flush=True)
    else:
        print("  lighting: volume + probe validity coherent, manifest points at the pack's own bake",
              flush=True)


def merge_gamedata_interactables(gd_path, dataset_dir, switch_levels=None):
    """Stage-6 gamedata ENRICHMENT, callable standalone. A freshly extracted gamedata.json
    lacks the pack's merged `switches` array, power tags and switch->door links — so any tool
    ADOPTING a re-extracted file into an already-built pack must run this first (a raw copy
    silently drops the Level Controls data; that regression is why this is a function).

    Folds the dataset's interact_<lv>.json records (stage 2) + gamedata's own typed point
    interactables (CardReader / RaidDialogEntryPoint) into `switches` — every interactable
    (power lever + alarms + buttons + water + triggers) with a `kind` (power|switch) — then
    tags POWER-GATED extracts, wires trigger-hash switch->door edges, and resolves requirement
    item names via the cached static dump (offline-safe). Mutates gd_path IN PLACE only when
    there is something to merge. switch_levels=None -> every interact_*.json in dataset_dir."""
    try:
        data = json.load(open(gd_path, encoding="utf-8"))
        if switch_levels is None:
            switch_levels = sorted(
                int(mm.group(1)) for f in os.listdir(dataset_dir)
                if (mm := re.fullmatch(r"interact_(\d+)\.json", f)))
        sw = []
        for lv in switch_levels:
            p = os.path.join(dataset_dir, f"interact_{lv}.json")
            if os.path.isfile(p):
                sw.extend(json.load(open(p, encoding="utf-8")))
        # gamedata's own typed POINT-interactables ride the same `switches` array: identical
        # record shape (kind tags them), so the LEVEL CONTROLS panel + click pipeline pick them
        # up with zero viewer plumbing. gamedata pos is viewer-bridged; the switch world_pos
        # contract is RAW Unity, so the X-flip is undone here.
        def _disp(n):
            s = re.sub(r"^(INTERACTIVE_|SBG_|Node_)", "", str(n or ""))
            s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).replace("_", " ").strip()
            return (s[:1].upper() + s[1:]) if s else "Interactable"
        for key, kind in (("card_readers", "card_reader"), ("dialogs", "dialog")):
            for i, r in enumerate(data.get(key) or []):
                p = r.get("pos") or [0.0, 0.0, 0.0]
                rec = {"id": f"gd:{r.get('lv')}:{kind}:{i}", "level": r.get("lv"),
                       "kind": kind, "world_pos": [-p[0], p[1], p[2]],
                       "label": _disp(r.get("name")), "count": 0, "targets": []}
                names = [it["n"] for it in r.get("items") or [] if it.get("n")]
                ids = ([it["id"] for it in r.get("items") or [] if it.get("id")]
                       or r.get("item_ids") or [])
                if ids:
                    rec["item_id"] = ids[0]
                    rec["item_ids"] = ids
                if names:
                    rec["item_name"] = names[0]
                    rec["item_names"] = names
                sw.append(rec)
        if sw:
            data["switches"] = sw
            # Tag POWER-GATED extracts: a switch's exfil target (by GameObject name) means
            # that extract "requires power". Feeds the viewer's "Requires switch" card line.
            ex_by_go = {e.get("go"): e for e in data.get("exfils", []) if e.get("go")}
            n_tag = 0
            for s in sw:
                for t in s.get("targets", []):
                    if "Exfil" in t.get("type", "") and t.get("name") in ex_by_go:
                        ex_by_go[t["name"]]["requires_power"] = True
                        n_tag += 1
            # Wire switch->door edges via the serialized TRIGGER-HASH link (newer maps):
            # a Switch's trigger "Open_01_<hash>" and the door it drives carry the SAME
            # digit hash (extract_interact `link` <-> extract_gamedata door `links`) —
            # byte-derived on both sides, zero name matching. The door also learns which
            # interactable controls it (`controlled_by` = the switch label).
            door_by_link = {}
            for dr in data.get("doors", []):
                for L in dr.get("links", []):
                    door_by_link.setdefault(L, []).append(dr)
            n_link = 0
            for s in sw:
                for dr in door_by_link.get(s.get("link") or "", []):
                    s.setdefault("targets", []).append({
                        "type": "EFT.Interactive.Door",
                        "name": dr.get("id") or dr.get("name"),
                        "world_pos": dr.get("pos"), "via": "trigger-link"})
                    dr["controlled_by"] = s.get("label")
                    n_link += 1
            # Requirement ITEMS: a switch payload can serialize a required 24-hex item
            # template id (extract_interact `item_id` — e.g. the frozen hatch wants the
            # cutting torch). Resolve display names via the cached tarkov.dev static dump
            # (offline-safe: disk cache or skip) so the viewer can show the requirement
            # with its icon — stage 7 caches the PNG under the same name slug.
            iids = sorted({s["item_id"] for s in sw if s.get("item_id")})
            if iids:
                try:
                    _intel = os.path.join(VIEWER, "extraction", "intel")
                    if _intel not in sys.path:
                        sys.path.insert(0, _intel)
                    import tarkov_static
                    got = {it["id"]: it["name"]
                           for it in tarkov_static.load_static_items(ids=iids).get("items") or []
                           if it.get("id") and it.get("name")}
                    n_nm = 0
                    for s in sw:
                        nm = got.get(s.get("item_id") or "")
                        if nm:
                            s["item_name"] = nm
                            n_nm += 1
                    print(f"  resolved {n_nm} switch requirement item name(s) "
                          f"({len(iids)} id(s))", flush=True)
                except Exception as e:
                    print(f"  note: switch item-name resolution skipped ({e})", flush=True)
            json.dump(data, open(gd_path, "w", encoding="utf-8"))
            npow = sum(1 for s in sw if s.get("kind") == "power")
            print(f"  merged {len(sw)} interactable(s) [{npow} power] into gamedata.json "
                  f"({n_tag} power-gated extract(s) tagged, {n_link} switch->door "
                  f"link(s))", flush=True)
    except Exception as e:
        print(f"  note: could not merge interactables into gamedata.json ({e})", flush=True)


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
    # The GAME'S OWN pathfinding parameters (Unity NavMeshProjectSettings: agent radius/height/
    # slope/climb, ledgeDropHeight, minRegionArea) + the area and layer tables. Engine-global, not
    # per-map, so it lands in the SHARED pack tier and every map's nav bake reads the same numbers.
    # Without it `nav_bake` silently falls back to hand-tuned constants -- the guesswork this is
    # meant to replace -- so it runs on every build and is cheap (one globalgamemanagers read).
    shared_dir = os.path.join(VIEWER, "packs", "shared")
    total = 9

    print(f"[BUILD] map={m} dataset={dsname} dataset_dir={dataset}", flush=True)
    # Hand the built atlas exe to every child (extraction, bakes) so the GPU steps that shell out to it
    # -- e.g. the vendor-neutral terrain composite (`atlas bake-terrain`) in stage-1 extraction -- can
    # find it in CLI builds too, not only when the viewer launches the build. No-op if already set or
    # no exe is built (those steps then use their CPU fallback).
    if not os.environ.get("EFT_ATLAS_EXE"):
        _ax = find_atlas_exe()
        if _ax:
            os.environ["EFT_ATLAS_EXE"] = _ax
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
        # Resolve + show the manifest-driven light levels so the plan reflects what stage 2 will do
        # (proves the roster lookup / day-night pick / streets-GZ list without any heavy work).
        dry_lights = light_levels_for(m)
        light_note = (f" (levels {dry_lights})" if dry_lights
                      else " (none known -> sky-only bake)")
        # Mirror the REAL stage-1 plan: a missing dataset means the build would run the ONE-TIME
        # extraction's three sub-passes, so the dry run rehearses their exact markers — stage
        # names AND sample byte-weighted [SUBPROGRESS] lines. This is what makes the viewer's
        # whole first-build progress path (fresh weight table, sub-stage windows, ETA) testable
        # in seconds; without it that path only ever ran on a real multi-hour extraction, which
        # is why its regressions went unnoticed. Same fidelity rationale as dry_lights above.
        fresh_sim = force or not os.path.isfile(os.path.join(dataset, "scene.json"))
        def _dry_pass(name, subs=()):
            print(f"[STAGE 1/{total}] {name}", flush=True)
            for s in subs:
                time.sleep(0.3)
                print(s, flush=True)
            time.sleep(0.6)
            print(f"[STAGE 1/{total}] {name}: done (0s)", flush=True)
        for i, name in enumerate(
            ["check dataset", "extract lights" + light_note, "bake lighting (GPU)",
             "assemble pack" + sc_note, "grass" + sc_note,
             "gameplay zones", "item icons", "bake nav grid (CPU)",
             "stamp fingerprint"], 1):
            if i == 1 and fresh_sim:
                print(f"[STAGE 1/{total}] check dataset", flush=True)
                _dry_pass("extract dataset (geometry + textures)",
                          ["[SUBPROGRESS] extract levels 3/217 bytes 536870912/5463154688",
                           "[SUBPROGRESS] extract levels 120/217 bytes 2936012800/5463154688",
                           "[SUBPROGRESS] extract levels 210/217 bytes 5348024320/5463154688"])
                _dry_pass("extract grass density")
                _dry_pass("extract physics colliders",
                          ["[SUBPROGRESS] colliders levels 100/217 bytes 2726297600/5463154688"])
                print(f"[STAGE 1/{total}] check dataset: done", flush=True)
                continue
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
                  f"- cannot auto-extract (see README.md)", flush=True)
            sys.exit(3)
        print(f"[STAGE 1/{total}] no dataset yet - running the ONE-TIME full extraction. CLOSE the "
              f"game and launcher first (file locks). This can take a long time.", flush=True)
        # extract_parallel splits the levels across cores (reusing the unchanged eft_extract_v2 per
        # chunk) then merges — big maps go multi-core. EFT_JOBS=1 forces the plain serial extractor.
        run(1, total, "extract dataset (geometry + textures)",
            [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "extract_parallel.py"),
             "--levels", levels, "--name", dsname] + alllod_extract, VIEWER)
        # Grass density is extracted for EVERY map here; indoor/no-terrain maps simply yield no
        # grids and are skipped at pack time (stage 5) -- no hardcoded indoor list.
        run(1, total, "extract grass density",
            [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_grass.py"),
             "--levels", levels, "--name", dsname], VIEWER, optional=True)
        # PHYSICS COLLIDERS -- the world the player actually collides with. The dataset above is
        # built from MeshRenderers, so it only holds geometry you can SEE; most of the collision
        # world has no renderer at all (interchange: 131,945 of 141,347 colliders), which leaves the
        # nav bake blind to invisible walls, kerbs, railings and blockers. Unity bakes its own
        # navmesh from exactly this (NavMeshSurface.m_UseGeometry = PhysicsColliders). Map-agnostic;
        # optional, so a map that yields none still builds.
        run(1, total, "extract physics colliders",
            [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_colliders.py"),
             "--levels", levels, "--name", dsname], VIEWER, optional=True)
        if not os.path.isfile(os.path.join(dataset, "scene.json")):
            print(f"[BUILD FAILED] extraction finished but no scene.json at {dataset} - check the "
                  f"log above (is UnityPy installed for EFT_PY_UNITY? is EFT_GAME_DATA correct and "
                  f"the game closed?)", flush=True)
            sys.exit(3)
    print(f"[STAGE 1/{total}] check dataset: done", flush=True)

    # 1b: THE GAME'S PATHFINDING PARAMETERS (global, cheap, every build). Unity's
    #     NavMeshProjectSettings holds the agent descriptors the nav bake must match -- radius,
    #     height, slope, climb, and the two that matter most, ledgeDropHeight and
    #     maxJumpAcrossDistance (both 0 on every EFT agent, i.e. the game's navmesh has no drop or
    #     jump links at all). It is an ENGINE type, so it reads despite the encrypted il2cpp
    #     metadata. Absent, `nav_bake` falls back to hand-tuned constants -- the guesswork this
    #     replaces -- so it is refreshed on every build rather than cached per map.
    run(1, total, "extract nav agent settings (global)",
        [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_nav.py"),
         "--out", shared_dir], VIEWER, optional=True)

    # 2a: POWER SWITCHES (optional, map-agnostic) -- scan the map's geometry levels for a power lever
    #     (an EFT.Interactive.Switch whose serialized PPtr[] resolves entirely to LampController).
    #     Writes switches_<lv>.json for each level that has one; those switch-bearing levels are then
    #     folded into the light extraction below so their controlled lights (which ship OFF and are
    #     absent from the *_Light scene) get extracted + group-tagged for the viewer's power toggle.
    switch_levels = []
    # dataset_levels() returns a COMMA-SEPARATED STRING ("52,54,...,520"); parse it to an int LIST here
    # (iterating the raw string fed the switch scanner single characters -> "--levels 5,2,,,5,4,..." ->
    # it scanned bogus levels 5/2/4/... , found no lever, and DELETED the real switches_*.json sidecars).
    geom_levels = [int(x) for x in dataset_levels(m).split(",") if x.strip()]
    if geom_levels:
        sw_missing = [lv for lv in geom_levels
                      if force or not os.path.isfile(os.path.join(dataset, f"interact_{lv}.json"))]
        if sw_missing:
            # extract_interact = superset of eft_extract_switches: keeps EVERY interactable Switch
            # (power lever + alarms + floor/call buttons + water-plane + keycard/exfil triggers),
            # classified name-free. Writes interact_<lv>.json (the power records match the old format,
            # so the stage-6 merge + eft_extract_lights light-grouping keep working).
            run(2, total, "scan interactables",
                [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "extract_interact.py"),
                 "--levels", ",".join(str(lv) for lv in sw_missing), "--name", dsname],
                VIEWER, optional=True)
        switch_levels = sorted(lv for lv in geom_levels
                               if os.path.isfile(os.path.join(dataset, f"interact_{lv}.json")))
        if switch_levels:
            print(f"[STAGE 2/{total}] interactable levels: {switch_levels}", flush=True)

    # 2: lights (optional) -- extract EVERY `*_Light` scene the map uses. The level LIST comes from
    #    the manifest (or a BuildSettings-derived fallback), so streets/ground_zero -- which split
    #    lighting across many district scenes -- now get full lighting, not just one scene. Switch-
    #    bearing levels are appended so the switch-controlled (default-off) banks are extracted too.
    levels_light = sorted(set(light_levels_for(m)) | set(switch_levels))
    if not levels_light:
        print(f"[STAGE 2/{total}] extract lights: none known for {m} - the bake will be SKY-ONLY "
              f"(dark interiors) unless a light sidecar already exists.", flush=True)
    else:
        # extract each level whose sidecar is missing (or all, on --force). --name is the DATASET
        # folder; the extractor writes ASSETS/<dataset>/lights_<lv>.json. Optional: a failure on any
        # single scene doesn't fail the build.
        todo = [lv for lv in levels_light
                if force or not os.path.isfile(os.path.join(dataset, f"lights_{lv}.json"))]
        if not todo:
            print(f"[STAGE 2/{total}] extract lights: skipped (all {len(levels_light)} sidecar(s) "
                  f"present)", flush=True)
        for lv in todo:
            run(2, total, f"extract lights (level {lv})",
                [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_lights.py"),
                 "--level", str(lv), "--name", dsname],
                VIEWER, optional=True)

    # 3: SH irradiance-volume bake. DEFAULT = the PORTABLE Rust baker (`atlas bake-sh`), which runs
    #    POST-ASSEMBLE (it reads the assembled pack's world triangles + BVH, exactly like bake-nav) on
    #    ANY machine -- AMD/Intel/no-GPU, no CUDA, no warp-lang -- so EVERY build ships baked lighting
    #    instead of the flat realtime fallback. That step is below, right after assemble. Set
    #    EFT_BAKE=warp to instead use the author-side CUDA baker (bake_volume2.py: NVIDIA + warp-lang,
    #    adds the diffuse bounce), which runs HERE (pre-assemble, from the dataset) and whose volume is
    #    promoted into the pack by assemble.
    # 2b: GRADE LUT — the game's own colour grading, extracted from the player's OWN install.
    #     It is game content, so nothing ships it: every build regenerates it into TK/out, where
    #     assemble promotes it into packs/shared as grade_lut.bin. Cheap (seconds) and cached by
    #     mtime, so a rebuild is a no-op once present unless --force. If the game's LUT cannot be
    #     read, the parameter-fitted reconstruction (no game files at all) stands in, so a build
    #     never fails for want of a grade.
    lut_out = os.path.join(TK, "out", "eft_grade_lut.bin")
    if force or not os.path.isfile(lut_out):
        os.makedirs(os.path.dirname(lut_out), exist_ok=True)
        made = run(2, total, "extract grade LUT (from your game install)",
                   [PY_UNITY, os.path.join(VIEWER, "extraction", "grade", "make_grade_lut_game.py"),
                    lut_out],
                   VIEWER, optional=True)
        if not os.path.isfile(lut_out):
            run(2, total, "grade LUT (fitted reconstruction — no game files)",
                [PY_UNITY, os.path.join(VIEWER, "extraction", "grade", "make_grade_lut.py"),
                 os.path.join(VIEWER, "extraction", "grade", "eft_grade_fit.json"),
                 lut_out.replace(".bin", ".png")],
                VIEWER, optional=True)
    else:
        print(f"[STAGE 2/{total}] grade LUT: present ({lut_out})", flush=True)

    bake_mode = os.environ.get("EFT_BAKE", "").strip().lower()
    if bake_mode == "warp":
        v2 = os.path.join(out_dir, "volume2.bin")
        if force or not os.path.isfile(v2):
            # portable kit baker: takes the MAP ID positionally, reads EFT_TARKMAP_ROOT itself
            # (run() passes it) and writes TK/out/<map id>/volume2.*; cwd-independent.
            # OPTIONAL: needs an NVIDIA CUDA GPU + warp-lang. Without them (or on any bake error) the
            # build continues and the post-assemble portable baker below fills in the volume instead.
            run(3, total, "bake lighting (warp/CUDA)",
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
    else:
        # NOT a [STAGE] marker: this announces work that has not started (the portable SH bake runs
        # AFTER assemble). Emitted as one it moved the loader bar into stage 3's window before stage 4
        # had begun, and `max_frac` then clamped the whole 1205s assemble at that value - the bar sat
        # at 74.9% for two thirds of a rebuild. The real marker is printed below when the bake runs.
        print("  lighting: portable SH bake (GPU auto, CPU fallback) runs after assemble", flush=True)

    # 4: assemble the pack (atomic; auto-ships loot/tasks/grade sidecars)
    run(4, total, "assemble pack",
        [PY, "-m", "eft_pipeline.assemble_bevy", m] + sc_flag + keeplods_flag, VIEWER)

    # 3 (portable, post-assemble): bake the SH irradiance volume with the PORTABLE Rust baker. It runs
    #    HERE, not up at stage 3, because it reads the ASSEMBLED pack's world triangles + BVH (shared
    #    with bake-nav) -- no CUDA, no warp-lang, no dataset re-read -- so it runs on ANY machine and
    #    every build ships baked lighting instead of the flat realtime fallback. Writes volume.json/
    #    volume.bin straight into the pack (the exact format the viewer's load_sh_volume reads). Skipped
    #    in warp mode (the CUDA volume is already in the pack) UNLESS that bake produced nothing, in
    #    which case this is the fallback so the pack still ships lighting. Skipped only when no built
    #    viewer exe can be found (a kit without a compiled binary).
    _stage3_started = time.time()
    if bake_mode != "warp" or not os.path.isfile(os.path.join(pack, "volume.bin")):
        atlas_exe = find_atlas_exe()
        if atlas_exe:
            run(3, total, "bake lighting (portable SH: GPU auto w/ CPU fallback, direct/indirect split)",
                [atlas_exe, "bake-sh", pack, "--indirect-only"], VIEWER, optional=True)
        else:
            print(f"[STAGE 3/{total}] lighting: skipped - viewer exe not found. Build it "
                  f"(`cargo build --release`) or set EFT_ATLAS_EXE, then rebuild to bake lighting.",
                  flush=True)
    # "A volume.bin exists" is NOT "this build baked one". The sidecar migration carries the
    # PREVIOUS pack's volume across a rebuild, so when bake-sh failed (it is optional) this line
    # cheerfully reported success over a volume baked for the OLD geometry — a grid whose very
    # dimensions no longer match the pack. Report what actually happened, and say plainly when the
    # lighting on disk is stale.
    _vol = os.path.join(pack, "volume.bin")
    if os.path.isfile(_vol):
        _fresh = os.path.getmtime(_vol) >= _stage3_started
        if _fresh:
            print("  lighting: SH irradiance volume baked into pack", flush=True)
        else:
            print("  lighting: WARNING - the bake did NOT produce a volume; the pack is carrying "
                  "the PREVIOUS build's volume.bin, which was baked for different geometry. "
                  "Re-run `atlas bake-sh <pack> --indirect-only` (add EFT_BAKE_CPU=1 if the GPU "
                  "bake keeps losing the device) before trusting this map's lighting.", flush=True)
        # The manifest is reconciled against the pack's real contents by finalize_pack_manifest()
        # at the end of the build — one place, covering the volume AND the light sidecars. (This
        # used to be an inline volume-only patch here, which is why a light sidecar that appeared
        # after assemble stayed invisible.)
    else:
        print("  lighting: none (flat realtime fallback until the baker runs)", flush=True)

    # SEA: derive the ocean height FROM THE GAME DATA (dataset scene.json) and patch it into the
    # assembled manifest — never authored per-map. The viewer synthesizes its horizon quad at
    # manifest.seaLevel; absent -> inland map, no quad. (Own step, not gated on the volume bake.)
    try:
        _slv = derive_sea_level(dataset)
        _mp = os.path.join(pack, "manifest.json")
        if os.path.isfile(_mp):
            _m = json.load(open(_mp, encoding="utf-8"))
            if _slv is not None:
                _m["seaLevel"] = _slv
                print(f"  sea: manifest seaLevel = {_slv} (derived from scene water planes)", flush=True)
            elif "seaLevel" in _m:
                del _m["seaLevel"]                 # no sea in the game data -> no synthesized sea
                print("  sea: no map-scale water plane in the scene - seaLevel removed", flush=True)
            json.dump(_m, open(_mp, "w", encoding="utf-8"))
    except Exception as _e:
        print(f"  sea: WARNING seaLevel derivation failed ({_e})", flush=True)

    # 5: grass -- DATA-DRIVEN: a map has grass iff its dataset actually yields density grids. Indoor/
    #    no-terrain maps (Factory/Labs/Labyrinth) produce none and are skipped automatically -- no
    #    hardcoded indoor list. The stage-1 inline extraction already produced grids on a FRESH build,
    #    so don't rescan the (huge, Streets = 217-level) terrain bundle if they're already present.
    tl = os.path.join(dataset, "terrain_layers")

    def _have_grids():
        return os.path.isdir(tl) and any(
            f.startswith("grass_density_") and f.endswith(".bin") for f in os.listdir(tl))

    if _have_grids():
        print(f"[STAGE 5/{total}] grass: density grids already present - skip re-extract", flush=True)
    else:
        gl = dataset_levels(m)
        grass_cmd = [PY_UNITY, os.path.join(VIEWER, "extraction", "unity", "eft_extract_grass.py"),
                     "--name", dsname]
        if gl:
            # pass the level list so the extractor finds the terrain bundle (without it, it
            # auto-detects over an empty list and silently skips -> no grass on fresh datasets).
            grass_cmd += ["--levels", gl]
        run(5, total, "grass: extract density grids", grass_cmd, VIEWER, optional=True)
    if _have_grids():
        run(5, total, "grass: build grass.bin",
            [PY, "-m", "eft_pipeline.build_grass", "--pack", pack] + sc_flag,
            VIEWER, optional=True)
    else:
        print(f"[STAGE 5/{total}] grass: none (no density grids - indoor/no-terrain map)", flush=True)

    # 6: typed gameplay zones (exfils/mines/snipers/doors/loose loot). The extractor writes
    # to tarkmap/out/<map>/gamedata.json and only PRINTS the copy step - do the copy here.
    #
    # Pass the DERIVED level list, same as geometry (stage 1) and grass (stage 5). Without it the
    # extractor falls back to the hand-curated config.source.levels, which omits the *_DesignStuff
    # scene on every map but woods/factory_rework -- and DesignStuff is where the loot lives
    # (interchange lv52: 902 LootableContainer + 63 LootPoint; reserve lv116: 992; streets' twelve
    # City_*_DesignStuff scenes: 1278). Geometry already scanned those levels, so the props were
    # RENDERED while their typed records were missing: 5 of 907 containers on interchange, 0 of 992
    # on reserve. Same drift the dataset_levels() docstring calls out, one stage later.
    gd_cmd = [PY_UNITY, os.path.join(VIEWER, "extraction", "intel", "extract_gamedata.py"), m]
    if geom_levels:                                   # already derived for stage 2; don't re-shell
        gd_cmd.append("--levels=" + ",".join(str(lv) for lv in geom_levels))
    _stage6_started = time.time()
    _gd_ok = run(6, total, "gameplay zones", gd_cmd, VIEWER, optional=True)
    if _gd_ok:
        gd = os.path.join(out_dir, "gamedata.json")
        if os.path.isfile(gd):
            merge_gamedata_interactables(gd, dataset, switch_levels)
            shutil.copyfile(gd, os.path.join(pack, "gamedata.json"))
            print("  gamedata.json -> pack", flush=True)
    # "A gamedata.json exists" is NOT "this build produced one" — the same hole the SH volume had
    # at stage 3, one stage later. This stage is optional, and on failure the whole copy block above
    # is skipped, so the sidecar migration carries the PREVIOUS build's doors/exfils/zones/loot
    # points across the atomic swap and the build still prints [BUILD OK] with nothing said. Seen
    # for real: extract_gamedata died rc=3221225477 (0xC0000005) 53 s in, and the shipped pack kept
    # gamedata from an earlier extraction — harmless while the dataset is unchanged, silently wrong
    # after a game update. Compare against the stage start and say plainly which one is on disk.
    _gd_pack = os.path.join(pack, "gamedata.json")
    if os.path.isfile(_gd_pack):
        if os.path.getmtime(_gd_pack) < _stage6_started:
            print("  gamedata: WARNING - this build did NOT produce gameplay data; the pack is "
                  "carrying the PREVIOUS build's gamedata.json (doors, exfils, zones, loot points). "
                  "It matches the older extraction, not this one. Re-run "
                  f"`extraction/intel/extract_gamedata.py {m}` before trusting this map's intel.",
                  flush=True)
        elif not _gd_ok:
            print("  gamedata: WARNING - the stage reported failure but left a fresh "
                  "gamedata.json; treat this map's intel as incomplete.", flush=True)
    elif not _gd_ok:
        print("  gamedata: WARNING - no gameplay data in the pack (the stage failed and there was "
              "no previous build to carry across); doors/exfils/zones/loot will be missing.",
              flush=True)

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

    # Reconcile the manifest with the pack LAST — after the post-assemble SH bake, the gamedata
    # merge and the icon fetch have all had their say. Everything that writes into the pack after
    # stage 4 lands before this point, so a freshly built map never ships a sidecar table that
    # disagrees with its own directory.
    finalize_pack_manifest(pack, dataset)
    verify_pack_lighting(pack)

    # 9: stamp the game fingerprint (menu update detection)
    run(9, total, "stamp fingerprint",
        [PY, os.path.join(HERE, "stamp_fingerprint.py"), pack], VIEWER)

    # Lighting completeness (finding 3a): a map we KNOW ships realtime lights (any derived
    # light_levels, now including the multi-scene streets/ground_zero) that produced neither a light
    # sidecar NOR an SH bake will render with dark/flat interiors. Don't hide that behind a clean
    # [BUILD OK] - surface it so the menu log makes the gap obvious (the pack is still valid
    # geometry, so exit stays 0).
    expects_light = bool(levels_light)
    have_lights = os.path.isdir(dataset) and any(
        f.startswith("lights_") and f.endswith(".json") for f in os.listdir(dataset))
    have_sh = os.path.isfile(os.path.join(out_dir, "volume.bin"))
    if expects_light and not (have_lights or have_sh):
        print(f"[BUILD WARN] no lighting for {m}: no *_Light extract and no SH bake - interiors "
              f"will be dark/flat. Run the light extract and/or the CUDA SH bake "
              f"(see README.md) then rebuild.", flush=True)
        print(f"[BUILD OK] pack ready (WARNING: no lighting for {m})", flush=True)
    else:
        print("[BUILD OK] pack ready", flush=True)

    # Post-build storage dedup: a texture shared by several maps is byte-identical in each dataset's
    # tex/ (source-identity naming), so it's stored once per map = pure waste. Hardlink the copies to
    # one physical file -- transparent + lossless. Pure HOUSEKEEPING, so it runs AFTER [BUILD OK] and
    # FULLY DETACHED (no wait): sha1-hashing GBs on a slow drive took minutes, and running it before
    # the OK line kept the UI on "BUILDING" long after the pack was ready (field report). Output goes
    # to devnull so a late finish can't interleave into the NEXT build's log. Idempotent + per-file
    # best-effort, so an overlap with a subsequent build or an early process exit is harmless.
    try:
        env = dict(os.environ, EFT_ASSETS_ROOT=ASSETS)
        flags = 0x00000008 | 0x08000000 if os.name == "nt" else 0  # DETACHED | CREATE_NO_WINDOW
        subprocess.Popen(
            [sys.executable, os.path.join(HERE, "dedup_textures.py")], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags)
        print("  [dedup] storage dedup backgrounded (housekeeping; does not block the pack)", flush=True)
    except Exception as e:
        print(f"  [dedup] skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
