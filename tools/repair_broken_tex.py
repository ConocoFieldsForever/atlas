#!/usr/bin/env python3
"""Re-extract half-written texture PNGs from the game's own asset files.

A killed extraction run (before texpool gained atomic writes) left PNGs whose first ~41
bytes are a valid IHDR + IDAT header and whose remaining megabytes are NTFS preallocation
zeros. PIL's lazy open() reads only the header, so the corruption survived every visual
check, and exp_tex's skip-if-exists guard then preserved the files through every rebuild —
the viewer renders them as the deliberate magenta failed-decode placeholder.

The pixel data still exists in the game install; this re-reads it. The filename encodes the
source: <m_Name san()>__<source file stem>_<path_id>.png (exp_tex's naming). Whether a file
is a NORMAL map (needs the DXT5nm unswizzle) is taken from the packs' own materials.json
(`normal` vs `albedo` references), never from name suffixes.

    venv\Scripts\python.exe tools\repair_broken_tex.py            # scan + repair all maps
    venv\Scripts\python.exe tools\repair_broken_tex.py --dry-run  # report only

Exits non-zero if any broken file could not be repaired (never silently drops one).
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detect_game_dir():
    """EFT install root from the uninstall registry key (same source the viewer's
    detect_game_dir trusts), falling back to the stock install path."""
    try:
        import winreg
        for hive_key in (
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov",
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov",
        ):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, hive_key) as k:
                    loc = winreg.QueryValueEx(k, "InstallLocation")[0]
                    if loc and os.path.isdir(loc):
                        return loc
            except OSError:
                continue
    except ImportError:
        pass
    return r"C:\Battlestate Games\Escape from Tarkov"


def png_complete(fp):
    """A fully-written PNG ends with an IEND chunk; the half-written casualties end in zeros."""
    try:
        with open(fp, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size < 60:
                return False
            f.seek(-16, 2)
            return b"IEND" in f.read()
    except OSError:
        return False


def unswizzle_normal(img):
    """Unity DXT5nm packs normal.X in ALPHA, Y in GREEN, R~const. Rebuild standard RGB normal.
    Byte-for-byte the extractor's own rule (eft_extract_v2.unswizzle_normal)."""
    from PIL import Image
    a = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    if a[..., 0].mean() > 0.95 and a[..., 0].std() < 0.06:  # red ~constant => DXT5nm
        X = a[..., 3] * 2 - 1
        Y = a[..., 1] * 2 - 1
        Z = np.sqrt(np.clip(1 - X * X - Y * Y, 0, 1))
        out = np.stack([X * .5 + .5, Y * .5 + .5, Z * .5 + .5], -1)
        return Image.fromarray((out * 255).astype(np.uint8), "RGB")
    return img


def normal_referenced(repo):
    """Basenames referenced as `normal` (and as `albedo`) by ANY built pack's materials.json —
    the authoritative classification for the unswizzle decision."""
    normals, albedos = set(), set()
    for mj in glob.glob(os.path.join(repo, "packs", "*.eftpack", "materials.json")):
        try:
            mats = json.load(open(mj, encoding="utf-8"))
        except Exception as e:
            print(f"  warn: unreadable {mj}: {e}")
            continue
        for m in mats:
            n = m.get("normal")
            a = m.get("albedo")
            if n:
                normals.add(os.path.basename(n).lower())
            if a:
                albedos.add(os.path.basename(a).lower())
    return normals, albedos


def index_game_files(game):
    """stem(lower) -> full path for every serialized-asset candidate in the install.
    Skips streamed payloads (.resS/.resource) — UnityPy pulls those in by itself."""
    idx = {}
    for root, _dirs, files in os.walk(game):
        for fn in files:
            if fn.endswith((".resS", ".resource", ".manifest", ".dll", ".json", ".txt", ".png")):
                continue
            stem = fn[:-7] if fn.endswith(".assets") else fn
            idx.setdefault(stem.lower(), os.path.join(root, fn))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--game", default=None)
    args = ap.parse_args()

    broken = []
    for fp in glob.glob(os.path.join(REPO, "eft_assets", "*", "tex", "*.png")):
        if not png_complete(fp):
            broken.append(fp)
    print(f"[scan] {len(broken)} broken PNGs across eft_assets/*/tex")
    if not broken:
        return 0
    for fp in broken:
        print("   ", os.path.relpath(fp, REPO))
    if args.dry_run:
        return 1

    game = args.game or detect_game_dir()
    if not os.path.isdir(game):
        print(f"[fail] game dir not found: {game}")
        return 1

    import UnityPy
    normals, albedos = normal_referenced(REPO)
    game_idx = index_game_files(game)
    env_cache = {}
    failed = []

    for fp in broken:
        base = os.path.basename(fp)[:-4]
        try:
            name_part, src = base.rsplit("__", 1)
            stem, pid_s = src.rsplit("_", 1)
            pid = int(pid_s)
        except ValueError:
            print(f"[fail] {base}: filename does not carry __<stem>_<pid>")
            failed.append(fp)
            continue
        src_fp = game_idx.get(stem.lower())
        if not src_fp:
            print(f"[fail] {base}: no game file with stem '{stem}'")
            failed.append(fp)
            continue
        if src_fp not in env_cache:
            env_cache[src_fp] = UnityPy.load(src_fp)
        env = env_cache[src_fp]
        obj = next(
            (o for o in env.objects if o.path_id == pid and o.type.name == "Texture2D"),
            None,
        )
        if obj is None:
            print(f"[fail] {base}: no Texture2D path_id {pid} in {os.path.basename(src_fp)}")
            failed.append(fp)
            continue
        try:
            img = obj.read().image
        except Exception as e:
            print(f"[fail] {base}: decode from game data failed: {e}")
            failed.append(fp)
            continue
        lower = os.path.basename(fp).lower()
        is_normal = lower in normals and lower not in albedos
        if is_normal:
            img = unswizzle_normal(img)
        tmp = fp + ".tmp"
        img.save(tmp, format="PNG")
        os.replace(tmp, fp)  # atomic: never another half-written file
        if not png_complete(fp):
            print(f"[fail] {base}: rewrite still incomplete?!")
            failed.append(fp)
            continue
        kind = "normal" if is_normal else "albedo"
        print(f"[ok]   {base}  <- {os.path.basename(src_fp)} pid {pid}  ({kind}, {img.size[0]}x{img.size[1]})")

    print(f"[done] repaired {len(broken) - len(failed)}/{len(broken)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
