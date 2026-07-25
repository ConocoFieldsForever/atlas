# IL2CPP metadata — what it would give us, and why we can't read EFT's

Tool: `tools/il2cpp_explore.py` (stdlib only).

## What this file is

EFT is built with Unity's IL2CPP backend: C# is compiled to C++, and everything the runtime needs
to know about *types* — every class, every field, in **declaration order** — lives in
`il2cpp_data/Metadata/global-metadata.dat`.

That declaration order is exactly the order Unity's serializer writes a MonoBehaviour payload in.
So a readable metadata file turns every payload layout in this repo from *recovered* into
*declared*.

## Why we can't read the shipped one

EFT ships it **encrypted**. Measured, not assumed:

| check | expected (plain il2cpp) | EFT's file |
|---|---|---|
| first dword (sanity magic) | `0xFAB11BAF` | `0x63153607` |
| printable runs >= 6 bytes | tens of thousands of real identifiers | 52,058 runs, **all noise** (`a&1-'\{;#L[a`, `4r;qu;9`, …) |
| searching those runs for `Door` | many hits | **zero** |

The second row matters: the file is not merely header-obfuscated with a readable body. 52,058
printable runs in 27 MB is roughly what uniform random bytes produce, and none of them is an
identifier. The whole payload is encrypted, so *no* parser and *no* string scan can recover names
from the file as shipped.

## Using the explorer

```
python tools/il2cpp_explore.py header            # magic/version/section table, or the ENCRYPTED verdict
python tools/il2cpp_explore.py strings [SUBSTR]  # identifier + string-literal tables
python tools/il2cpp_explore.py types   [SUBSTR]  # namespace.Name of every type definition
python tools/il2cpp_explore.py fields  <Type>    # a type's fields, in declaration order
python tools/il2cpp_explore.py scan    [SUBSTR]  # raw printable-byte sweep (works on ANY file)
    --file=PATH   metadata to read (default: the game's own, i.e. the encrypted one)
    --limit=N     max rows (default 200; 0 = all)
```

Against the stock install, `header` prints the ENCRYPTED verdict and exits non-zero, the parsing
commands decline rather than emit garbage, and `scan` works but (per the table above) finds only
noise.

To get the real thing, point it at a **decrypted dump** — Il2CppDumper output, or metadata carved
from the running process's memory, where the header is restored:

```
python tools/il2cpp_explore.py --file=D:\dump\global-metadata.dat types Door
python tools/il2cpp_explore.py --file=D:\dump\global-metadata.dat fields Door
```

`EFT_IL2CPP_METADATA` sets the default path; otherwise it is
`<EFT_GAME_DATA>/il2cpp_data/Metadata/global-metadata.dat`.

## What readable metadata would buy this project

Every MonoBehaviour field we consume today was recovered **empirically** — byte offsets found by
column statistics across dumps, then validated against known-good cases, with defensive checks so a
bad read degrades to `null` instead of shipping garbage. That works, but it is fragile across game
patches and it cost real debugging (see the Icebreaker door layout change in `docs/DOORS.md` §3).

Declared field offsets would replace guesswork in:

| decoder | what we recover by hand today |
|---|---|
| `dec_door` | KeyId, Id, `EDoorState` @ IdEnd+92, signed open angle @ IdEnd+56, and the newer trigger-name block that shifts all of it |
| `extract_interact` | Switch trigger names, required item template id, interaction verb |
| `dec_stationary` | mounted weapon template id, traverse/elevation arcs |
| `dec_lootpoint` / containers | item/category template id arrays |
| `dec_card_reader` | accepted keycard ids |
| light controller | spot angle / intensity / range (recovered by column stats) |

It would also let us decode components we currently skip because the layout is unknown, and make
the extractors self-checking across patches (field moved => detect it, instead of silently reading
the wrong four bytes).

## You do not need to decrypt it — two routes go around it

Static disassembly of `GameAssembly.dll` (capstone, no process touched) established that the
cipher is CUSTOM, not AES (no S-boxes or AES-NI anywhere in the binary; loader at VA
`0x180592dd0`, control-flow-flattened with self-generated RWX stubs), and that it **rekeys every
patch** — the first dword was `e82b66cd` in the 2026-06-16 build and `63153607` in 1.0.6.5.46221.
Recovering the transform would be days of work invalidated by each update. It was deliberately
not pursued, because two sources give the same information without touching it:

1. **Field NAMES — the pre-1.0 Mono `Assembly-CSharp.dll`.** EFT was a Mono game until 1.0
   (2025-11-15). That assembly is ordinary .NET metadata: real class and field names, in
   declaration order, statically deobfuscatable (`sp-tarkov/assembly-tool` + de4dot), no
   encryption. Declaration order is what Unity's serializer writes, so it maps onto the very
   offsets in the table above — it CONFIRMS and names them (layouts still drift between versions,
   so it supplements the empirical decoders rather than replacing them).
2. **Tuned lighting VALUES — AssetRipper/AssetStudio on the engine types.** Fog, exposure,
   colour-grade, LUT and bloom are serialized Unity asset data, not il2cpp fields: PostProcessingV2
   `ColorGrading`/`Bloom`/`PostProcessProfile` and URP/HDRP `Volume` are ENGINE types with
   hardcoded type-trees, readable with zero decryption and zero process contact.

**Runtime dumping is off the table.** Never dump, attach to, or scan the retail live process:
BattlEye's kernel driver catches debuggers, cross-process reads and injected threads, and retail
"offline/PVE" is not BE-free. Il2CppDumper and Cpp2IL both reject an encrypted `.dat`, so neither
is a shortcut around any of this.

See `docs/IL2CPP_OPPORTUNITIES.md` §6 for the current-version re-verification and the ranked
follow-ups.

## Practical note

None of this is required for the viewer to work — the empirical decoders are validated and shipping.
Treat readable metadata as a robustness upgrade, and re-run `tools/audit_doors.py` (and the other
extractor validations) after any game patch, which is the cheap defence we have without it.
