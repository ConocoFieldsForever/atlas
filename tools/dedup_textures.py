"""Deduplicate identical extracted assets ACROSS datasets by hardlinking.

The extractor names textures/terrain-layers by SOURCE IDENTITY (same source asset -> same filename +
byte-identical content), so a texture shared by several maps is stored once PER MAP in each dataset's
tex/ dir -- e.g. ~270 MB duplicated between just Streets and Factory. This is pure on-disk waste: the
files are identical.

This tool groups every <EFT_ASSETS_ROOT>/*/{tex,terrain_layers}/ file by (size, sha1(content)) and,
for each group of identical files on DIFFERENT inodes, replaces the duplicates with HARDLINKS to one
master copy. Result: one physical copy on disk, but the file still appears in every dataset's dir, so
NOTHING changes for the extractor, the assembler, or the viewer -- zero visual/behavioural change,
just less disk. NTFS + POSIX both support hardlinks (os.link). Idempotent; safe to re-run after a
build (a re-extraction overwrites a link with a fresh file, which this simply re-links next run).

  python tools/dedup_textures.py [assets_dir] [--dry-run]

Env: EFT_ASSETS_ROOT overrides the assets dir (default: ./eft_assets).
"""
import hashlib
import os
import sys
from collections import defaultdict

ROOT = os.environ.get("EFT_ASSETS_ROOT")
if not ROOT:
    ROOT = next((a for a in sys.argv[1:] if not a.startswith("-")), "eft_assets")
DRY = "--dry-run" in sys.argv
SUBDIRS = ("tex", "terrain_layers")


def sha1(path, buf=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"[dedup] assets dir not found: {ROOT} (set EFT_ASSETS_ROOT)")

    # Collect candidate files across every dataset's tex/ + terrain_layers/.
    files = []
    for ds in sorted(os.listdir(ROOT)):
        for sub in SUBDIRS:
            subp = os.path.join(ROOT, ds, sub)
            if os.path.isdir(subp):
                for fn in os.listdir(subp):
                    fp = os.path.join(subp, fn)
                    if os.path.isfile(fp):
                        files.append(fp)
    print(f"[dedup] scanning {len(files)} files under {ROOT}", flush=True)

    # Group identical content (size first as a cheap pre-filter, then sha1).
    by_size = defaultdict(list)
    for fp in files:
        try:
            by_size[os.stat(fp).st_size].append(fp)
        except OSError:
            pass
    by_key = defaultdict(list)
    for size, group in by_size.items():
        if len(group) < 2:
            continue  # unique size -> unique content, no dup possible
        for fp in group:
            try:
                st = os.stat(fp)
                by_key[(size, sha1(fp))].append((fp, st.st_ino))
            except OSError:
                pass

    deduped = freed = 0
    for (size, _digest), group in by_key.items():
        if len(group) < 2:
            continue
        master_fp, master_ino = group[0]
        for fp, ino in group[1:]:
            if ino == master_ino:
                continue  # already the same physical file (hardlinked)
            deduped += 1
            freed += size
            if DRY:
                continue
            try:
                tmp = fp + ".deduptmp"
                os.link(master_fp, tmp)   # create link first; only unlink the dup if it succeeds
                os.replace(tmp, fp)       # atomic swap -> never leaves the file missing on a crash
            except OSError as e:
                print(f"  skip {fp}: {e}", flush=True)
                try:
                    os.remove(fp + ".deduptmp")
                except OSError:
                    pass
                deduped -= 1
                freed -= size

    verb = "would free" if DRY else "freed"
    print(f"[dedup] {deduped} duplicate files -> hardlinks; {verb} {freed / 1048576:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
