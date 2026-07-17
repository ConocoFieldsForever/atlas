"""Stamp packs with the game-install fingerprint they were built from.

The viewer's start menu recomputes the same fingerprint at launch and flags packs whose
stamp no longer matches (= the game updated since extraction -> re-extract needed).

Fingerprint = FNV-1a 64 over "name|size|mtime_s;" for every top-level file in
EscapeFromTarkov_Data whose name matches level*/ *.assets / *.resS / *.resource,
sorted by name. Stat-only (no file reads) so it runs in milliseconds. The Rust
side (viewer/src/menu.rs game_fingerprint) MUST implement the identical digest.

Usage: python tools/stamp_fingerprint.py [pack_dir ...]   (default: packs/*.eftpack)
"""

import json
import os
import sys
import time

GAME_DATA = os.environ.get(
    "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data"
)


def game_fingerprint(game_data: str) -> str | None:
    try:
        entries = []
        with os.scandir(game_data) as it:
            for e in it:
                if not e.is_file():
                    continue
                n = e.name
                if not (
                    n.startswith("level")
                    or n.endswith(".assets")
                    or n.endswith(".resS")
                    or n.endswith(".resource")
                ):
                    continue
                st = e.stat()
                entries.append((n, st.st_size, st.st_mtime_ns // 1_000_000_000))
    except OSError:
        return None
    if not entries:
        return None
    entries.sort()
    h = 0xCBF29CE484222325
    for n, size, mt in entries:
        for b in f"{n}|{size}|{mt};".encode():
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def main() -> None:
    fp = game_fingerprint(GAME_DATA)
    if fp is None:
        print(f"ERROR: no game files at {GAME_DATA}")
        sys.exit(1)
    print(f"game fingerprint: {fp}")
    packs = sys.argv[1:] or [
        os.path.join("packs", d) for d in os.listdir("packs") if d.endswith(".eftpack")
    ]
    for p in packs:
        mpath = os.path.join(p, "manifest.json")
        if not os.path.isfile(mpath):
            print(f"  skip {p}: no manifest.json")
            continue
        man = json.load(open(mpath, encoding="utf-8"))
        man["sourceFingerprint"] = fp
        man["sourceStampedAt"] = int(time.time())
        tmp = mpath + ".tmp"
        json.dump(man, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, mpath)
        print(f"  stamped {p}")


if __name__ == "__main__":
    main()
