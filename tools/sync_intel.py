"""One-command tarkov.dev INTEL refresh for the start menu's SYNC button.

Re-pulls the community data the viewer's overlays run on — loot.json (containers+ev, loose loot,
locks/keys, extracts), tasks.json (the full quest catalog + zones), and the item-icon cache — and
ships them into packs/shared/ where every pack resolves them. Prints `[STAGE i/N]` markers exactly
like build_map.py so the menu can stream progress. Exit 0 = data refreshed. ASCII output only.

  python tools/sync_intel.py

Env (same contract as build_map.py; unset -> legacy dev-machine defaults):
  EFT_TARKMAP_ROOT = the dir CONTAINING maps/ and out/ (builders write <TK>/out/*.json)
Network: tarkov.dev GraphQL + icon CDN only. Re-run per wipe or whenever prices feel stale."""

import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.dirname(HERE)
TK = os.environ.get("EFT_TARKMAP_ROOT")
PY = sys.executable or "python"
SHARED = os.path.join(VIEWER, "packs", "shared")
BUILD_OUT = os.path.join(TK, "out") if TK else SHARED


def run(stage, total, name, cmd, optional=False):
    print(f"[STAGE {stage}/{total}] {name}", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="ascii:replace")
    env["EFT_INTEL_OUT_DIR"] = BUILD_OUT
    if TK:
        env.setdefault("EFT_TARKMAP_ROOT", TK)
    p = subprocess.Popen(cmd, cwd=VIEWER, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="ascii", errors="replace")
    for line in p.stdout:
        print("  " + line.rstrip(), flush=True)
    rc = p.wait()
    dt = time.time() - t0
    if rc != 0:
        if optional:
            print(f"[STAGE {stage}/{total}] {name}: FAILED rc={rc} ({dt:.0f}s) - optional, continuing",
                  flush=True)
            return False
        print(f"[SYNC FAILED] stage '{name}' rc={rc} after {dt:.0f}s", flush=True)
        sys.exit(rc or 1)
    print(f"[STAGE {stage}/{total}] {name}: done ({dt:.0f}s)", flush=True)
    return True


def main():
    total = 4
    os.makedirs(SHARED, exist_ok=True)

    # 1: loot intel (containers+ev / loose / locks / extracts, all maps in one file)
    run(1, total, "loot intel (tarkov.dev)",
        [PY, os.path.join(VIEWER, "extraction", "intel", "build_loot.py")])

    # 2: task catalog (all maps, one file)
    run(2, total, "task catalog (tarkov.dev)",
        [PY, os.path.join(VIEWER, "extraction", "intel", "build_tasks.py")])

    # 3: ship into packs/shared. With no legacy tarkmap tree, builders already wrote here.
    print(f"[STAGE 3/{total}] ship to packs/shared", flush=True)
    shipped = 0
    for f in ("loot.json", "tasks.json"):
        src = os.path.join(BUILD_OUT, f)
        if os.path.isfile(src):
            dst = os.path.join(SHARED, f)
            if os.path.normcase(os.path.abspath(src)) != os.path.normcase(os.path.abspath(dst)):
                shutil.copyfile(src, dst)
            print(f"  {f} -> packs/shared ({os.path.getsize(src)/1e6:.1f} MB)", flush=True)
            shipped += 1
        else:
            print(f"  WARNING: {src} missing (builder failed?)", flush=True)
    if shipped == 0:
        print("[SYNC FAILED] nothing shipped", flush=True)
        sys.exit(1)
    print(f"[STAGE 3/{total}] ship to packs/shared: done (0s)", flush=True)

    # 4: item icons for every BUILT pack's map (incremental; network failures skip per icon)
    packs = [d[:-8] for d in os.listdir(os.path.join(VIEWER, "packs"))
             if d.endswith(".eftpack")] if os.path.isdir(os.path.join(VIEWER, "packs")) else []
    if packs:
        ok = True
        for m in sorted(packs):
            ok &= run(4, total, f"icons: {m}",
                      [PY, os.path.join(VIEWER, "extraction", "intel", "fetch_icons.py"), m],
                      optional=True)
    else:
        print(f"[STAGE 4/{total}] icons: skipped (no packs built yet)", flush=True)

    print("[SYNC OK] tarkov.dev intel refreshed", flush=True)


if __name__ == "__main__":
    main()
