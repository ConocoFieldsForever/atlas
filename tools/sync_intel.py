#!/usr/bin/env python
"""One-command tarkov.dev intel refresh for the start menu's SYNC button.

Uses tarkov.dev's supported pre-generated JSON catalogs, revalidated with ETags. The parent process
fetches each required catalog at most once, builders read the shared cache without more HTTP, and
unchanged inputs do not rebuild loot/tasks. Item/task artwork remains incremental.

  python tools/sync_intel.py

Env (same contract as build_map.py; unset -> standalone viewer defaults):
  EFT_TARKMAP_ROOT       dir containing maps/ and out/ (builders write <root>/out/*.json)
  EFT_TARKOV_JSON_CACHE persistent raw-catalog cache (defaults to packs/shared/.tarkov-json-cache)
"""

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.dirname(HERE)
INTEL = os.path.join(VIEWER, "extraction", "intel")
if INTEL not in sys.path:
    sys.path.insert(0, INTEL)
import tarkov_static

TK = os.environ.get("EFT_TARKMAP_ROOT")
PY = sys.executable or "python"
SHARED = os.path.join(VIEWER, "packs", "shared")
BUILD_OUT = os.path.join(TK, "out") if TK else SHARED
STATE = os.path.join(SHARED, ".tarkov-sync.json")
SYNC_FORMAT = 1

# Eight explicit dependencies. Do not fetch json.tarkov.dev/endpoints or any catalog Atlas
# does not consume. The *_en catalogs resolve the compact translation keys in their base catalog.
CATALOGS = (
    "maps", "maps_en", "items", "items_en",
    "tasks", "tasks_en", "traders", "traders_en",
)
LOOT_DEPS = {"maps", "maps_en", "items", "items_en"}
TASK_DEPS = {"tasks", "tasks_en", "traders", "traders_en", "items", "items_en", "maps"}


def read_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_state(changed, downloaded):
    doc = {
        "format": SYNC_FORMAT,
        "checked": int(time.time()),
        "changed": sorted(changed),
        "downloadedBytes": downloaded,
    }
    tmp = STATE + f".{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="ascii") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, STATE)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


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


def skip(stage, total, name):
    print(f"[STAGE {stage}/{total}] {name}: unchanged - skipped", flush=True)


def sync(changed, format_changed):
    total = 4
    loot_out = os.path.join(BUILD_OUT, "loot.json")
    tasks_out = os.path.join(BUILD_OUT, "tasks.json")
    rebuild_loot = format_changed or not os.path.isfile(loot_out) or bool(changed & LOOT_DEPS)
    rebuild_tasks = format_changed or not os.path.isfile(tasks_out) or bool(changed & TASK_DEPS)

    if rebuild_loot:
        run(1, total, "loot intel (json.tarkov.dev)",
            [PY, os.path.join(INTEL, "build_loot.py")])
    else:
        skip(1, total, "loot intel")

    if rebuild_tasks:
        run(2, total, "task catalog (json.tarkov.dev)",
            [PY, os.path.join(INTEL, "build_tasks.py")])
    else:
        skip(2, total, "task catalog")

    # With no legacy tarkmap tree, builders already wrote to packs/shared. Copy only a rebuilt file;
    # an ETag-only check leaves the prior artifact and its honest data-age mtime untouched.
    print(f"[STAGE 3/{total}] ship to packs/shared", flush=True)
    for filename, rebuilt in (("loot.json", rebuild_loot), ("tasks.json", rebuild_tasks)):
        src = os.path.join(BUILD_OUT, filename)
        dst = os.path.join(SHARED, filename)
        if not os.path.isfile(src):
            print(f"[SYNC FAILED] required output missing: {src}", flush=True)
            sys.exit(1)
        needs_copy = rebuilt or not os.path.isfile(dst)
        if needs_copy and os.path.normcase(os.path.abspath(src)) != os.path.normcase(os.path.abspath(dst)):
            shutil.copyfile(src, dst)
            print(f"  {filename} -> packs/shared ({os.path.getsize(src) / 1e6:.1f} MB)", flush=True)
        elif needs_copy:
            print(f"  {filename} built in packs/shared ({os.path.getsize(src) / 1e6:.1f} MB)", flush=True)
        else:
            print(f"  {filename}: unchanged", flush=True)
    print(f"[STAGE 3/{total}] ship to packs/shared: done (0s)", flush=True)

    # One process handles all packs: it parses items.json once and deduplicates shared icon refs.
    packs_dir = os.path.join(VIEWER, "packs")
    packs = sorted(d[:-8] for d in os.listdir(packs_dir)
                   if d.endswith(".eftpack") and os.path.isdir(os.path.join(packs_dir, d))) \
        if os.path.isdir(packs_dir) else []
    if packs:
        run(4, total, f"icons: {len(packs)} built map(s)",
            [PY, os.path.join(INTEL, "fetch_icons.py"), *packs], optional=True)
    else:
        print(f"[STAGE 4/{total}] icons: skipped (no packs built yet)", flush=True)


def main():
    os.makedirs(SHARED, exist_ok=True)
    os.environ.setdefault("EFT_TARKOV_JSON_CACHE", os.path.join(SHARED, ".tarkov-json-cache"))

    print(f"[SYNC] revalidating {len(CATALOGS)} required json.tarkov.dev catalogs", flush=True)
    changed, downloaded = tarkov_static.prefetch(CATALOGS)
    # Children use the documents just checked above without issuing duplicate conditional requests.
    os.environ["EFT_TARKOV_JSON_CACHE_READY"] = "1"
    format_changed = read_state().get("format") != SYNC_FORMAT
    print(f"[SYNC] {len(changed)} changed; {downloaded / 1e6:.2f} MB downloaded", flush=True)

    sync(changed, format_changed)
    write_state(changed, downloaded)
    print("[SYNC OK] tarkov.dev intel checked", flush=True)


if __name__ == "__main__":
    main()
