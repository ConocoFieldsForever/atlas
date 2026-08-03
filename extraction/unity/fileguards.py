"""Completeness guards + atomic writes for extracted artifacts.

ONE definition, imported by every exporter. These started life inside `eft_extract_v2.py`; the
collider exporter and the terrain-albedo path each carried their own `not exists or getsize == 0`
copy, and when the real guards were fixed those copies were not. That is the whole reason this
module exists: a size test cannot see the failure mode these files actually have.

The failure: a run killed mid-write leaves NTFS preallocation. The directory entry carries the
FINAL size while the data was never flushed, so the file is megabytes of NUL. `getsize() == 0` is
false, the skip-if-exists guard trusts it, and no amount of re-extraction ever repairs it -- streets
carried 8,962 zero-filled OBJs (968 MiB) that survived full rebuilds. So every check here reads the
bytes that prove the file was finished, and every write goes through `atomic_write` so the state can
only be complete-or-absent in the first place.
"""

import os
import threading


def atomic_write(final_path, write_fn):
    """Write `final_path` atomically: `write_fn(tmp)` fills a sibling temp, then `os.replace` makes
    it appear complete-or-not-at-all. Byte-for-byte identical on the happy path. Thread-unique temp
    name because texture saves run on a pool."""
    tmp = f"{final_path}.tmp{os.getpid()}_{threading.get_ident()}"
    try:
        write_fn(tmp)
        os.replace(tmp, final_path)                # atomic on the same directory/volume
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)                     # don't leave the partial temp lying around
        except OSError:
            pass
        raise


def write_text_utf8(path, text):
    """EFT ships Cyrillic asset names; cp1252 would truncate the file at the first one."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def png_complete(fp):
    """True only for a fully-written PNG (tail carries IEND). PIL's lazy open() reads only the
    header, so a NUL-filled body passes every casual check; the viewer then draws it as its magenta
    failed-decode placeholder. Missing/unreadable/short -> False (extract it)."""
    try:
        with open(fp, "rb") as f:
            f.seek(0, 2)
            if f.tell() < 60:
                return False
            f.seek(-16, 2)
            return b"IEND" in f.read()
    except OSError:
        return False


def obj_complete(fp):
    """True only for a fully-written mesh OBJ.

    `mesh.export()` writes `g <name>` first and ends every face line with a newline, so head+tail
    catches NUL-fill and truncation alike at two small reads -- no full parse on the reuse path.
    Missing/unreadable/short -> False (re-export it).
    """
    try:
        with open(fp, "rb") as f:
            f.seek(0, 2)
            if f.tell() < 8:
                return False
            f.seek(0)
            if f.read(2) not in (b"g ", b"v "):     # NUL-filled / garbage head
                return False
            f.seek(-1, 2)
            return f.read(1) in b"\r\n"            # truncated mid-line -> incomplete
    except OSError:
        return False


def drop_incomplete(fp):
    """Remove a half-written casualty so os.link / skip guards can never resurrect it."""
    try:
        os.remove(fp)
    except OSError:
        pass
