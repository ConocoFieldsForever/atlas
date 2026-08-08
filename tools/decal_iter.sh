#!/bin/bash
# FAST decal iteration: re-project ONE level's decals and re-assemble, skipping the nav/SH bakes
# that dominate a full build. ~3 min instead of ~25. Usage: tools/decal_iter.sh [levels]
set -e
cd "$(dirname "$0")/.."
LEVELS="${1:-62}"
export EFT_ASSETS_ROOT="$PWD/eft_assets"
# the dir holding maps/ and out/, NOT the pack output dir (build_map.py:13). Pointing this at
# packs/ went unnoticed while the assembler reconstructed the path instead of reading the setting.
export EFT_TARKMAP_ROOT="$PWD/tarkmap"
echo "[iter] projecting decals for level(s) $LEVELS"
venv/Scripts/python.exe extraction/intel/extract_decals.py interchange \
    --dataset="$PWD/eft_assets/interchange_v2" --levels="$LEVELS" 2>&1 | tail -2
echo "[iter] assembling pack (no nav/SH re-bake)"
# NEVER filter this: a grep here once hid an atomic-publish failure, so the pack silently kept
# the PREVIOUS geometry while every log line said the projection had changed. Show the tail raw
# and fail loudly on a non-zero exit.
set -o pipefail
venv/Scripts/python.exe -m eft_pipeline.assemble_bevy interchange --self-contained 2>&1 | tail -12
echo "[iter] done"
