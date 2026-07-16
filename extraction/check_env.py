#!/usr/bin/env python
"""Smoke-check for the portable EFT extraction kit (see extraction/README.md).

  python extraction/check_env.py                     verify python deps + env vars + paths
  python extraction/check_env.py --init D:\\eft_work  ALSO scaffold the workspace there
                                                     (tarkmap/maps + tarkmap/out + eft_assets,
                                                      copies the map configs and the grade LUT)

Exit code 0 = ready to extract. Non-zero = at least one hard failure (each printed with a fix).
Warp/scipy problems are warnings only (they gate the SH bake / grade-LUT bake, not extraction).
"""
import os, sys, shutil, glob

KIT = os.path.dirname(os.path.abspath(__file__))            # <repo>/extraction
REPO = os.path.dirname(KIT)                                 # <repo> (eft_native_viewer)

fails, warns = [], []


def ok(msg):
    print(f"  [ok]   {msg}")


def fail(msg, fix):
    fails.append(msg)
    print(f"  [FAIL] {msg}\n         fix: {fix}")


def warn(msg, fix):
    warns.append(msg)
    print(f"  [warn] {msg}\n         note: {fix}")


def check_deps():
    print("\n== python dependencies ==")
    if sys.version_info < (3, 9):
        fail(f"python {sys.version.split()[0]} is too old", "install Python 3.10+ (3.12 recommended)")
    else:
        ok(f"python {sys.version.split()[0]}")
    # hard requirements (extraction cannot run without these)
    for mod, pipname in (("numpy", "numpy"), ("PIL", "Pillow"), ("UnityPy", "UnityPy")):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{pipname} {ver}")
            if mod == "UnityPy" and not ver.startswith("1.25"):
                warn(f"UnityPy {ver} != pinned 1.25.x",
                     "the extractors were validated on UnityPy 1.25.0; API shifts between minors. "
                     "pip install UnityPy==1.25.0 if extraction misbehaves")
        except ImportError:
            fail(f"{pipname} not importable", f"pip install -r extraction/requirements.txt  (missing: {pipname})")
    # soft requirements (gate individual steps only)
    try:
        import scipy  # noqa: F401
        ok(f"scipy {scipy.__version__} (grade-LUT baker)")
    except ImportError:
        warn("scipy not importable", "only needed to RE-bake the grade LUT (a prebuilt "
             "extraction/grade/eft_grade_lut.bin ships with the kit). pip install scipy to rebuild it")
    try:
        import warp as wp
        ndev = 0
        try:
            ndev = wp.get_cuda_device_count()
        except Exception:
            pass
        if ndev > 0:
            ok(f"warp-lang {wp.__version__}, {ndev} CUDA device(s) (SH volume bake ready)")
        else:
            warn(f"warp-lang {wp.__version__} imported but no CUDA device found",
                 "bake_volume2.py needs an NVIDIA GPU + CUDA driver; without it skip the SH bake "
                 "(the viewer still runs, just without baked GI)")
    except ImportError:
        warn("warp-lang not importable", "only needed for bake_volume2.py (SH irradiance volume). "
             "pip install warp-lang (requires an NVIDIA GPU + CUDA driver at runtime)")
    # the pack emitter's vendored core must resolve (bake + assemble both import it)
    sys.path.insert(0, REPO)
    try:
        from eft_pipeline.tarkmap_core.config import MapConfig  # noqa: F401
        ok("eft_pipeline.tarkmap_core importable (pack emitter / bake core)")
    except Exception as e:
        fail(f"eft_pipeline.tarkmap_core not importable ({e})",
             "run from a full checkout of the eft_native_viewer repo (extraction/ must sit next to eft_pipeline/)")


def check_env():
    print("\n== environment variables ==")
    gd = os.environ.get("EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
    src = "EFT_GAME_DATA" if "EFT_GAME_DATA" in os.environ else "default"
    if not os.path.isdir(gd):
        fail(f"game data dir not found: {gd} ({src})",
             r"set EFT_GAME_DATA to your game's EscapeFromTarkov_Data dir, e.g. "
             r'setx EFT_GAME_DATA "D:\Games\EFT\EscapeFromTarkov_Data"')
    elif not os.path.exists(os.path.join(gd, "globalgamemanagers")):
        fail(f"{gd} exists but has no globalgamemanagers ({src})",
             "EFT_GAME_DATA must be the EscapeFromTarkov_Data dir itself (the one containing "
             "globalgamemanagers, level0, level1, ..., sharedassets*.assets), not the install root")
    else:
        nlev = len(glob.glob(os.path.join(gd, "level*")))
        ok(f"EFT_GAME_DATA -> {gd} ({src}, {nlev} level files)")

    tk = os.environ.get("EFT_TARKMAP_ROOT")
    if not tk:
        fail("EFT_TARKMAP_ROOT is not set",
             r"point it at your workspace tarkmap dir, e.g.  setx EFT_TARKMAP_ROOT D:\eft_work\tarkmap  "
             "(create the workspace with: python extraction/check_env.py --init D:\\eft_work)")
    else:
        maps = os.path.join(tk, "maps")
        outd = os.path.join(tk, "out")
        ncfg = len(glob.glob(os.path.join(maps, "*", "config.json")))
        if ncfg == 0:
            fail(f"EFT_TARKMAP_ROOT={tk} but {maps} has no map configs",
                 f"copy the kit's map configs there:  xcopy /e /i \"{os.path.join(KIT, 'maps')}\" \"{maps}\"  "
                 "(or rerun check_env.py --init)")
        else:
            ok(f"EFT_TARKMAP_ROOT -> {tk} ({ncfg} map configs)")
        if not os.path.isdir(outd):
            warn(f"{outd} does not exist yet", "it is created by the bake/intel steps; mkdir it now if you like")
        lut = os.path.join(outd, "eft_grade_lut.bin")
        if not os.path.exists(lut):
            warn(f"no grade LUT at {lut}",
                 f"copy \"{os.path.join(KIT, 'grade', 'eft_grade_lut.bin')}\" there (or rerun --init); "
                 "assemble_bevy ships it into every pack (in-game color grading)")
        else:
            ok("grade LUT present in tarkmap/out")

    ar = os.environ.get("EFT_ASSETS_ROOT") or (
        os.path.join(os.path.dirname(tk), "eft_assets") if tk else os.path.join(os.getcwd(), "eft_assets"))
    src = "EFT_ASSETS_ROOT" if "EFT_ASSETS_ROOT" in os.environ else "derived"
    ok(f"datasets will be written to {ar} ({src})")
    if tk and os.path.normcase(os.path.abspath(ar)) != os.path.normcase(
            os.path.abspath(os.path.join(os.path.dirname(tk), "eft_assets"))):
        warn("EFT_ASSETS_ROOT is not <EFT_TARKMAP_ROOT>\\..\\eft_assets",
             "the map configs + pack emitter resolve datasets against the tarkmap parent dir; keep the "
             "standard layout (workspace\\tarkmap + workspace\\eft_assets) unless you know what you're doing")


def init_workspace(root):
    print(f"\n== scaffolding workspace at {root} ==")
    tk = os.path.join(root, "tarkmap")
    maps_dst = os.path.join(tk, "maps")
    out_dst = os.path.join(tk, "out")
    assets = os.path.join(root, "eft_assets")
    os.makedirs(out_dst, exist_ok=True)
    os.makedirs(assets, exist_ok=True)
    if not os.path.isdir(maps_dst):
        shutil.copytree(os.path.join(KIT, "maps"), maps_dst)
        print(f"  copied map configs -> {maps_dst}")
    else:
        print(f"  {maps_dst} already exists (left untouched)")
    lut_dst = os.path.join(out_dst, "eft_grade_lut.bin")
    if not os.path.exists(lut_dst):
        shutil.copy2(os.path.join(KIT, "grade", "eft_grade_lut.bin"), lut_dst)
        print(f"  copied grade LUT -> {lut_dst}")
    print("\n  now set the env vars (new shells pick them up):")
    print(f"    setx EFT_TARKMAP_ROOT \"{tk}\"")
    print(f"    setx EFT_ASSETS_ROOT \"{assets}\"")
    print("    setx EFT_GAME_DATA \"<your EscapeFromTarkov_Data dir>\"")
    # make the env checks below see the new workspace even before setx takes effect
    os.environ.setdefault("EFT_TARKMAP_ROOT", tk)
    os.environ.setdefault("EFT_ASSETS_ROOT", assets)


def main():
    if "--init" in sys.argv:
        i = sys.argv.index("--init")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--init needs a workspace dir, e.g.  python extraction/check_env.py --init D:\\eft_work")
        init_workspace(os.path.abspath(sys.argv[i + 1]))
    check_deps()
    check_env()
    print(f"\n{'READY' if not fails else 'NOT READY'}: {len(fails)} failure(s), {len(warns)} warning(s)")
    if fails:
        print("fix the [FAIL] items above, then re-run:  python extraction/check_env.py")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
