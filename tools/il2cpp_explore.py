#!/usr/bin/env python
"""IL2CPP global-metadata explorer -- read the game's OWN type/field/string tables.

WHY: every MonoBehaviour payload this repo decodes is currently recovered EMPIRICALLY -- byte
offsets found by column statistics over dumps (see extraction/intel/extract_gamedata.py: Door
KeyId + DoorState, Switch trigger names, StationaryWeapon arcs, LootPoint template ids). The
DECLARED truth for those layouts lives in il2cpp_data/Metadata/global-metadata.dat: every type,
every field, in declaration order -- which is exactly the order Unity's serializer writes a
MonoBehaviour payload in. Read that and the guesswork goes away (see docs/IL2CPP.md).

THE CATCH: EFT ships that file ENCRYPTED. Its first dword is 0x63153607, not il2cpp's 0xFAB11BAF
sanity magic, and the body is high-entropy -- no plain parser can read it, and one that tries
will happily print garbage. So this tool does two separate jobs:
  * against REAL metadata (a decrypted dump from Il2CppDumper or a memory dump of the running
    process) it parses the header, strings, types and fields;
  * against the SHIPPED file it says ENCRYPTED plainly and exits non-zero, and `scan` still
    works -- a raw printable-run sweep that assumes nothing about structure.

Version support: the header is read as its version-stable leading section table, and record
strides for typeDefinitions/fields are DERIVED from the file (a stride must tile its section
exactly AND make every sampled name resolve to a readable identifier) rather than hardcoded off
a version table this tool cannot test. That covers the v24..v31 layout drift -- v24.0's rgctx
pair, the byref type index dropped in v27 -- without guessing. Anything that fails to validate
is reported as unresolved instead of being printed wrong.

Usage:
  python tools/il2cpp_explore.py header               # magic/version/section table, or ENCRYPTED
  python tools/il2cpp_explore.py strings [SUBSTR]     # identifier + string-literal tables
  python tools/il2cpp_explore.py types [SUBSTR]       # namespace.Name of every type definition
  python tools/il2cpp_explore.py fields <TypeName>    # a type's fields, in declaration order
  python tools/il2cpp_explore.py scan [SUBSTR]        # raw byte scan; the only one that works
                                                      # on the encrypted shipped file
  options: --file=PATH (metadata to read)  --limit=N (max rows, default 200; 0 = all)
Env: EFT_IL2CPP_METADATA overrides the file; else <EFT_GAME_DATA>/il2cpp_data/Metadata/
     global-metadata.dat, with EFT_GAME_DATA defaulting to the stock install path.
ASCII-only output (this console is cp1252).
"""

import os
import re
import struct
import sys

MAGIC = 0xFAB11BAF
MIN_RUN = 6                                    # shortest printable run `scan` will report
DATA = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
DEFAULT_META = os.environ.get("EFT_IL2CPP_METADATA") or os.path.join(
    DATA, "il2cpp_data", "Metadata", "global-metadata.dat")

# After (magic, version) the header is 20 consecutive (offset:uint32, size:int32) pairs in this
# order -- the same order in every version this tool targets. NOTE each "size" is a BYTE COUNT,
# not an item count: items = size // sizeof(record). A pair that does not fit inside the file is
# DROPPED rather than trusted, so a bad guess surfaces as a missing section, never as nonsense.
SECTIONS = ("stringLiteral", "stringLiteralData", "string", "events", "properties", "methods",
            "parameterDefaultValues", "fieldDefaultValues", "fieldAndParameterDefaultValueData",
            "fieldMarshaledSizes", "parameters", "fields", "genericParameters",
            "genericParameterConstraints", "genericContainers", "nestedTypes", "interfaces",
            "vtableMethods", "interfaceOffsets", "typeDefinitions")

# Stride candidates, most-likely first. Il2CppTypeDefinition is 88 bytes from v27 on; older v24.x
# adds a byref type index (92) and, earliest of all, an rgctx start/count pair (100).
# Il2CppFieldDefinition is {nameIndex, typeIndex, token} = 12 from v27 on, 16 in v24.x (it also
# carried a custom-attribute index). The probe checks these before the generic sweep.
TD_SIZES = (88, 92, 100, 96, 84, 104, 108)
FD_SIZES = (12, 16, 20, 8, 24)
# Il2CppTypeDefinition tail is stable across v24..v31: 8 uint16 counts then bitfield+token
# (2 uint32) = 24 bytes, and the 8 int32 *Start indices sit immediately before them. Counting
# BACK from the record end therefore locates fieldStart/field_count without knowing the version.
FIELD_START_FROM_END = 56       # int32 fieldStart  (first of the 8 *Start indices)
FIELD_COUNT_FROM_END = 20       # uint16 field_count (third of the 8 uint16 counts)


def _num(fmt, buf, off, size):
    """One little-endian scalar, or None if it would read off the end. Never raises."""
    if off is None or off < 0 or off + size > len(buf):
        return None
    return struct.unpack_from(fmt, buf, off)[0]


def _u32(buf, off):
    return _num("<I", buf, off, 4)


def _i32(buf, off):
    return _num("<i", buf, off, 4)


def _u16(buf, off):
    return _num("<H", buf, off, 2)


def _a(text):
    """ASCII-safe for this cp1252 console: EFT identifiers/literals include Cyrillic."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _looks_like_ident(name):
    """Cheap gate used to validate a candidate record stride. Deliberately strict: il2cpp names
    start with a letter/underscore (or '<' for compiler-generated ones) and stay printable."""
    return bool(name) and len(name) < 512 and (name[0].isalpha() or name[0] in "_<$")


class Meta(object):
    """A metadata file: parsed if it is plain il2cpp, otherwise flagged and left alone."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            self.buf = fh.read()
        self.size = len(self.buf)
        self.magic = _u32(self.buf, 0)
        self.version = _i32(self.buf, 4)
        self.plain = self.magic == MAGIC and self.version is not None and 0 < self.version < 100
        self.sec = {}
        self._stride = {}
        if self.plain:
            for i, name in enumerate(SECTIONS):
                off, size = _u32(self.buf, 8 + 8 * i), _i32(self.buf, 12 + 8 * i)
                if off and size is not None and 0 <= size and off + size <= self.size:
                    self.sec[name] = (off, size)

    def blob(self, name):
        s = self.sec.get(name)
        return self.buf[s[0]:s[0] + s[1]] if s else None

    def cstr(self, index):
        """Identifier at a byte index into the `string` blob. None if it is not a clean string --
        the caller treats that as 'unresolved', which is how a wrong stride gets rejected."""
        blob = self.blob("string")
        if blob is None or index is None or not 0 <= index < len(blob):
            return None
        end = blob.find(b"\0", index)
        if end < 0 or end - index > 4096:
            return None
        try:
            name = blob[index:end].decode("utf-8")
        except UnicodeDecodeError:
            return None
        return name if name and all(ord(c) >= 32 for c in name) else None

    def stride(self, section, candidates, samples=48):
        """Record size for `section`, derived from the file: it must tile the section exactly and
        the first int32 of every sampled record must resolve to an identifier-looking name. Two
        independent checks over ~48 records make a false positive very unlikely; None means the
        layout is unknown and the caller must report that rather than print rows."""
        if section in self._stride:
            return self._stride[section]
        got = None
        if section in self.sec:
            off, size = self.sec[section]
            for cand in list(candidates) + [n for n in range(8, 129, 4) if n not in candidates]:
                if size < cand or size % cand:
                    continue
                total = size // cand
                step = max(1, total // samples)
                idx = list(range(0, total, step))
                if all(_looks_like_ident(self.cstr(_i32(self.buf, off + i * cand)))
                       for i in idx) and idx:
                    got = cand
                    break
        self._stride[section] = got
        return got

    def n_records(self, section, candidates):
        st = self.stride(section, candidates)
        return None if st is None else self.sec[section][1] // st

    def types(self):
        """(index, namespace, name, record_offset, stride) per type definition; empty if the
        record layout could not be pinned down."""
        st = self.stride("typeDefinitions", TD_SIZES)
        if st is None:
            return
        off, size = self.sec["typeDefinitions"]
        for i in range(size // st):
            base = off + i * st
            yield (i, self.cstr(_i32(self.buf, base + 4)) or "",
                   self.cstr(_i32(self.buf, base)) or "?", base, st)

    def fields_of(self, base, st):
        """(name, typeIndex, declaration slot) for one type record, or None if the field table
        does not validate. Field BYTE offsets are not in metadata -- see docs/IL2CPP.md."""
        n_fields = self.n_records("fields", FD_SIZES)
        start = _i32(self.buf, base + st - FIELD_START_FROM_END)
        count = _u16(self.buf, base + st - FIELD_COUNT_FROM_END)
        if n_fields is None or start is None or count is None:
            return None
        if start < 0 or count < 0 or start + count > n_fields:
            return None
        fd_off, fd_st = self.sec["fields"][0], self.stride("fields", FD_SIZES)
        out = []
        for k in range(count):
            rec = fd_off + (start + k) * fd_st
            name = self.cstr(_i32(self.buf, rec))
            if name is None:
                return None                      # partial garbage is worse than an honest None
            out.append((name, _i32(self.buf, rec + 4), start + k))
        return out


def load(path):
    if not os.path.isfile(path):
        print("[il2cpp] ERROR: no such file: %s" % _a(path))
        print("[il2cpp] set EFT_IL2CPP_METADATA or EFT_GAME_DATA, or pass --file=PATH")
        raise SystemExit(1)
    return Meta(path)


def banner(meta):
    print("[il2cpp] file    : %s (%s bytes)" % (_a(meta.path), format(meta.size, ",")))
    print("[il2cpp] magic   : 0x%08x  (il2cpp sanity is 0x%08x)" % (meta.magic or 0, MAGIC))


def require_plain(meta):
    """Every parse command funnels through here, so an encrypted file gets one clear verdict
    instead of four different crashes."""
    if meta.plain:
        return
    banner(meta)
    print("[il2cpp] VERDICT : ENCRYPTED / not plain il2cpp metadata -- cannot parse.")
    print("[il2cpp] EFT ships an obfuscated global-metadata.dat. To use header/strings/types/")
    print("[il2cpp] fields you must supply a DECRYPTED dump (Il2CppDumper output, or metadata")
    print("[il2cpp] carved from the running process's memory) via --file=PATH.")
    print("[il2cpp] What DOES work on this file today: `scan` (raw printable-byte sweep).")
    raise SystemExit(2)


def cmd_header(meta, _arg, limit):
    require_plain(meta)
    banner(meta)
    print("[il2cpp] version : %d" % meta.version)
    print("[il2cpp] sections: offset / size-in-BYTES (%d of %d resolved inside the file)"
          % (len(meta.sec), len(SECTIONS)))
    for name in SECTIONS:
        s = meta.sec.get(name)
        print("  %-34s %s" % (name, ("0x%08x  %12s" % (s[0], format(s[1], ",")))
                              if s else "(does not fit in file -- dropped)"))
    for label, section, cands in (("typeDefinitions", "typeDefinitions", TD_SIZES),
                                  ("fields", "fields", FD_SIZES)):
        st, n = meta.stride(section, cands), meta.n_records(section, cands)
        print("[il2cpp] %-16s: %s" % (label, "stride %d bytes -> %s records" % (st, format(n, ","))
                                      if st else "stride UNRESOLVED (unknown layout)"))


def cmd_strings(meta, needle, limit):
    require_plain(meta)
    key = (needle or "").lower()
    shown = total = 0
    blob = meta.blob("string")
    if blob is not None:
        pos = 0
        for part in blob.split(b"\0"):
            text = part.decode("utf-8", "replace")
            if text and key in text.lower():
                total += 1
                if not limit or shown < limit:
                    print("  ident 0x%08x  %s" % (pos, _a(text)))
                    shown += 1
            pos += len(part) + 1
    lits, data = meta.sec.get("stringLiteral"), meta.blob("stringLiteralData")
    if lits and data is not None:
        for i in range(lits[1] // 8):
            length, at = _u32(meta.buf, lits[0] + i * 8), _u32(meta.buf, lits[0] + i * 8 + 4)
            if length is None or at is None or at + length > len(data) or length > 1 << 20:
                continue
            text = data[at:at + length].decode("utf-8", "replace")
            if key in text.lower():
                total += 1
                if not limit or shown < limit:
                    print("  lit   #%-8d  %s" % (i, _a(text)))
                    shown += 1
    if blob is None and not lits:
        print("[il2cpp] no string sections resolved -- header layout not understood.")
    print("[il2cpp] %s of %s matching strings shown%s"
          % (format(shown, ","), format(total, ","),
             (" (--limit=%d; --limit=0 for all)" % limit) if limit and total > shown else ""))


def cmd_types(meta, needle, limit):
    require_plain(meta)
    if meta.stride("typeDefinitions", TD_SIZES) is None:
        print("[il2cpp] typeDefinitions layout UNRESOLVED -- no stride tiles the section with")
        print("[il2cpp] readable names. Metadata version %s may be unsupported." % meta.version)
        raise SystemExit(3)
    key = (needle or "").lower()
    shown = total = 0
    for i, ns, name, _base, _st in meta.types():
        full = ("%s.%s" % (ns, name)) if ns else name
        if key in full.lower():
            total += 1
            if not limit or shown < limit:
                print("  #%-6d %s" % (i, _a(full)))
                shown += 1
    print("[il2cpp] %s of %s matching types shown" % (format(shown, ","), format(total, ",")))


def cmd_fields(meta, needle, limit):
    require_plain(meta)
    if not needle:
        print("[il2cpp] usage: fields <TypeName>")
        raise SystemExit(1)
    key = needle.lower()
    exact = [t for t in meta.types()
             if t[2].lower() == key or ("%s.%s" % (t[1], t[2])).lower() == key]
    hits = exact or [t for t in meta.types() if key in ("%s.%s" % (t[1], t[2])).lower()]
    if not hits:
        print("[il2cpp] no type matches '%s'" % _a(needle))
        raise SystemExit(3)
    for i, ns, name, base, st in hits[:limit or len(hits)]:
        print("[type] #%-6d %s" % (i, _a(("%s.%s" % (ns, name)) if ns else name)))
        rows = meta.fields_of(base, st)
        if rows is None:
            print("  (field table did not validate for this record -- reporting nothing rather")
            print("   than guessing; metadata version %s layout may differ)" % meta.version)
            continue
        for slot, (fname, tidx, gidx) in enumerate(rows):
            print("  %2d  %-40s typeIndex=%-6s (field #%d)" % (slot, _a(fname), tidx, gidx))
        print("  %d field(s). Declaration ORDER is what Unity serializes in; the declared type"
              % len(rows))
        print("  NAME and the runtime byte offset live in GameAssembly.dll, not here.")
    if len(hits) > (limit or len(hits)):
        print("[il2cpp] %d more matches (raise --limit)" % (len(hits) - limit))


def cmd_scan(meta, needle, limit):
    """Works on ANY blob, encrypted included: no structure is assumed, so nothing is 'parsed'.
    Treat every hit as a coincidence until proven otherwise."""
    print("[il2cpp] RAW BYTE SCAN -- not parsed metadata. Printable runs >= %d bytes in %s"
          % (MIN_RUN, _a(os.path.basename(meta.path))))
    key = (needle or "").lower().encode("utf-8", "replace")
    shown = total = 0
    ascii_re = re.compile(b"[\x20-\x7e]{%d,}" % MIN_RUN)
    wide_re = re.compile(b"(?:[\x20-\x7e]\x00){%d,}" % MIN_RUN)
    for tag, rx, strip in (("ascii", ascii_re, False), ("utf16", wide_re, True)):
        for m in rx.finditer(meta.buf):
            raw = m.group(0)[::2] if strip else m.group(0)
            if key and key not in raw.lower():
                continue
            total += 1
            if not limit or shown < limit:
                print("  %s 0x%08x  %s" % (tag, m.start(), _a(raw.decode("ascii", "replace"))))
                shown += 1
    print("[il2cpp] %s of %s runs shown%s" % (format(shown, ","), format(total, ","),
          (" (--limit=%d; --limit=0 for all)" % limit) if limit and total > shown else ""))


COMMANDS = {"header": cmd_header, "strings": cmd_strings, "types": cmd_types,
            "fields": cmd_fields, "scan": cmd_scan}


def main(argv):
    path, limit, rest = DEFAULT_META, 200, []
    it = iter(argv)
    for arg in it:
        for flag, cast in (("--file", str), ("--limit", int)):
            if arg == flag:
                val = next(it, None)
            elif arg.startswith(flag + "="):
                val = arg.split("=", 1)[1]
            else:
                continue
            if val is None:
                print("[il2cpp] %s needs a value" % flag)
                return 1
            if flag == "--file":
                path = cast(val)
            else:
                limit = max(0, int(val))
            break
        else:
            rest.append(arg)
    if not rest or rest[0] not in COMMANDS:
        print(__doc__.split("Usage:", 1)[1].strip() if "Usage:" in __doc__ else "")
        return 1
    COMMANDS[rest[0]](load(path), rest[1] if len(rest) > 1 else None, limit)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
