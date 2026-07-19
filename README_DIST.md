# Atlas — distribution readme

A native raid-planning map viewer for Escape from Tarkov. Renders maps extracted from YOUR
OWN game install; overlays loot, extracts, hazards, quests and more.

## Quick start (viewer-only bundle)

1. Unzip anywhere. Keep `atlas.exe`, `assets/`, and `packs/` side by side.
2. Drop `.eftpack` map folders into `packs/` (build your own — see below; packs are never
   distributed, they contain data derived from the game).
3. Run `atlas.exe` → the stash screen lists maps; PLAY opens one.

Controls: WASD + RMB mouse-look, Q/E up/down, Shift fast. Double-click identifies geometry.
Right panel: layers, search, quests, routes, graphics settings.

## Building your own packs (full bundle)

Requirements: Windows 10+, your own EFT install, ~2–10 GB disk per map. **Python is bundled**
(`python\` beside the exe) — nothing to install. Optional for baked lighting: NVIDIA GPU (CUDA)
— without it maps render with flat ambient.

```powershell
.\atlas.exe            # in the menu: click INSTALL DEPS once (uses the bundled Python to fetch
                       # UnityPy/numpy/Pillow), set GAME INSTALL if not autodetected, then BUILD.
# Advanced/offline: .\bootstrap.ps1 does the same deps install from a shell.
# GPU lighting bake (optional):  .\python\python.exe -m pip install -r extraction\requirements-bake.txt
```

The BUILD button runs the full pipeline with live progress (lights → lighting bake →
assembly → grass → gameplay zones → icons → fingerprint). "GAME FILES UPDATED" on a row
means the game patched since that pack was built — press UPDATE. Loot values and the quest
layer are fetched from tarkov.dev automatically after your first successful build (needs
internet), so they populate without a separate step.

> **Self-built packs vs. prebuilt:** geometry, lighting, grass, gameplay zones, loot, quests,
> extracts and hazards all build fully here. Two extras are dev-pipeline-only and are NOT
> produced by this kit: **in-viewer routing** (the Navigate tab needs a separate nav-grid
> bake) and the **game color grade** (needs a LUT extracted from the game's `resources.assets`).
> Maps still render and plan fully without them — colors are just un-graded and point-to-point
> routing is unavailable on packs you built yourself.

## Environment toggles (common)

| var | effect |
|---|---|
| `EFT_PACK` | pack dir to open directly (skip menu) |
| `EFT_RENDER=m0` | fallback instanced render path |
| `EFT_SHADOWS=1` | opt-in sun shadows |
| `EFT_GRADE=0` / `EFT_GRADE_EXPOSURE=x` | disable/adjust the game color grade |
| `EFT_FOG=0` | disable distance haze |
| `EFT_UNCAPPED=1` | uncap FPS (benchmarking) |
| `EFT_HIDDEN=1` + `EFT_SHOT=out.png` | headless screenshot |
| `EFT_GAME_DATA` | game install override (else autodetect/menu) |
| `EFT_TEX_BC=0` | disable BC texture compression |

## Data sources & credits

Geometry, lighting, grass, zones: extracted from YOUR game files, on your machine.
Prices, quests, item icons: [tarkov.dev](https://tarkov.dev) — free community API; data is
fetched during pack builds, cached under `packs/shared/`.

See `LICENSE-NOTES.md` for what may and may not be redistributed.
