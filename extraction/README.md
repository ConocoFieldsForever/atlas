# EFT extraction kit — build viewer packs from YOUR OWN game install

Self-contained, portable copy of the extraction pipeline. Nothing in here assumes the
original developer's machine: every path comes from three environment variables. You need
a legal Escape from Tarkov install, Python, and (for the lighting bake) an NVIDIA GPU.

The chain this kit covers, end to end:

```
game install (EscapeFromTarkov_Data)
  └─ 1. unity/eft_extract_v2.py      → dataset: scene.json + meshes/ + tex/ + terrain_layers/
     1b. unity/eft_extract_grass.py  → grass density grids + billboard textures (outdoor maps)
     1c. unity/eft_extract_lights.py → lights_<N>.json (real in-game lights)
  └─ 2. bake/bake_volume2.py         → SH irradiance volume (volume2.* → promote to volume.*)   [CUDA]
  └─ 3. intel/extract_semantics.py   → semantics.json (extracts/doors/zones POIs)
     3b. intel/build_loot.py         → loot.json   (tarkov.dev, no game files)
     3c. intel/build_tasks.py        → tasks.json  (tarkov.dev, no game files)
     3d. grade/eft_grade_lut.bin     → prebuilt color-grade LUT (ships with the kit)
  └─ 4. python -m eft_pipeline.assemble_bevy <map>   → <map>.eftpack   (script already in this repo)
     4b. python -m eft_pipeline.build_grass --pack …  → grass.bin in the pack (outdoor maps)
  └─ 5. cargo run --release -- packs\<map>.eftpack    → the viewer
```

Loot, tasks and the grade LUT are picked up **automatically** by the pack step (they are
copied into every pack from `tarkmap/out/`). `semantics.json` is the only sidecar you copy
next to the pack yourself (step 8 below).

---

## 0. Requirements

- Windows, **Python 3.10+** (3.12 validated), and for the viewer build: stable **Rust 1.88+**.
- `pip install -r extraction\requirements.txt`
  (UnityPy is **pinned to 1.25.0** — its API shifts between minors.)
- **SH volume bake only:** an NVIDIA GPU with a CUDA driver (`warp-lang`). Everything else is CPU.
- **Disk:** an extracted dataset is roughly 1–6 GB per map (OBJ meshes + full-res PNG textures;
  big outdoor maps are the high end). A finished `.eftpack` is **0.06–1 GB per map**. The pack
  references the dataset's textures **in place**, so keep the dataset on disk after packing.
  Budget ~10 GB free before extracting a large map.
- Run extraction while the game/launcher is **closed** (file locks).

## 1. One-time setup

Pick a workspace anywhere (example: `D:\eft_work`) and let the kit scaffold it:

```powershell
cd <this repo>            # the eft_native_viewer checkout
python extraction\check_env.py --init D:\eft_work
```

That creates the layout the whole pipeline expects and copies in the map configs + grade LUT:

```
D:\eft_work\
  tarkmap\
    maps\<map>\config.json    per-map config (levels, dataset name, coordinate matrix, culls)
    out\                      bake + intel outputs land here; eft_grade_lut.bin copied in
  eft_assets\                 extracted datasets land here (one folder per map)
```

Then set the environment variables (new terminals pick them up):

```powershell
setx EFT_GAME_DATA   "C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data"
setx EFT_TARKMAP_ROOT "D:\eft_work\tarkmap"
setx EFT_ASSETS_ROOT  "D:\eft_work\eft_assets"     # optional: this is the derived default
```

| variable | meaning | default |
|---|---|---|
| `EFT_GAME_DATA` | the game's `EscapeFromTarkov_Data` dir (contains `globalgamemanagers`, `level*`, `sharedassets*`) | standard `C:\Battlestate Games\…` install path |
| `EFT_TARKMAP_ROOT` | workspace `tarkmap` dir (holds `maps\` + `out\`) | **required** — no default |
| `EFT_ASSETS_ROOT` | where datasets are written | `<EFT_TARKMAP_ROOT>\..\eft_assets` |

Verify in a **new** terminal:

```powershell
python extraction\check_env.py
```

Fix any `[FAIL]` lines (they each print the fix) until it says `READY`.

## 2. Extract the dataset — worked example: **factory**

Each map's Unity level indices live in its config — read them, don't guess:

```powershell
python -c "import json;print(json.load(open(r'extraction\maps\factory\config.json'))['source'])"
# -> levels: [2, 69, 70, 177, 398], root: eft_assets/factory
```

The `--name` you pass MUST match the config's `source.root` basename (`factory` here;
note interchange's dataset is named `interchange_v2`).

```powershell
python extraction\unity\eft_extract_v2.py --levels 2,69,70,177,398 --name factory
```

Writes `EFT_ASSETS_ROOT\factory\{scene.json, meshes\, tex\, terrain_layers\}`. Expect tens of
minutes for a big map; the script prints per-level progress. It is resumable — already-exported
meshes/textures are skipped on a re-run.

## 3. Extract the grass density grids (outdoor maps)

```powershell
python extraction\unity\eft_extract_grass.py --levels 2,69,70,177,398 --name factory
```

With `--levels` it auto-detects which level carries the terrain bundle; indoor maps (Factory,
Labs, Labyrinth) print "no TerrainData — skip", which is correct — just move on. For outdoor
maps this writes `terrain_layers\grass_density_*.bin`, the grass billboard PNGs and `grass.json`
(the pack's grass step and its albedo pick both need these).

## 4. Extract the real lights

Lights live in a separate `*_light` Unity scene per map. Level indices (current game build —
a game patch can shift them):

| map | `--level` |
|---|---|
| customs | 13 |
| shoreline | 41 |
| interchange | 64 |
| factory | 69 |
| labs | 114 |
| reserve | 146 |
| woods | 167 |
| lighthouse | 191 |
| labyrinth | 551 |
| streets | 367–383 (split across many scenes) |
| ground_zero | 494–503 (split across many scenes) |

```powershell
python extraction\unity\eft_extract_lights.py --level 69 --name factory
```

Writes `EFT_ASSETS_ROOT\factory\lights_69.json` (only lights that are LIVE in-game; add
`--all` to also keep disabled night-only banks, tagged `on:false`, in a separate file).

**Streets / Ground Zero:** the light set is split across many scenes. Run the extractor once
per level in the range; each run writes its own `lights_<N>.json`. The bake and the pack
consume **one** lights file (first `lights_*.json` alphabetically), so merge them: each file is
a flat JSON array — concatenate the arrays into a single `lights_<firstlevel>.json` and delete
(or rename to `.bak`) the rest.

## 5. Bake the SH irradiance volume (CUDA)

```powershell
python extraction\bake\bake_volume2.py factory
```

Reads the dataset's `scene.json` + `lights_*.json`, ray-traces on the GPU (NVIDIA Warp), and
writes `EFT_TARKMAP_ROOT\out\factory\volume2.{json,bin,vis.bin}` plus debug slice PNGs.
Expect minutes to ~half an hour depending on GPU and map size. The last line prints
`[VERIFY] PASS/FAIL` — if it passed, **promote** the outputs to the names the pack references:

```powershell
cd $env:EFT_TARKMAP_ROOT\out\factory
copy /y volume2.bin volume.bin
copy /y volume2.json volume.json
copy /y volume2.vis.bin volume.vis.bin
```

Tunables (env vars, defaults are right for a first bake): `LIGHT_SCALE`, `SKY_SCALE`,
`SUN_SCALE`, `N_DIR`, `XZ_TARGET`, `Y_SPACING`. `--rebuild-mesh` forces the cached bake mesh
to rebuild after you re-extract a dataset. No CUDA GPU? Skip this step — the viewer still
runs, just without baked GI.

## 6. Intel: semantics, loot, tasks

```powershell
python extraction\intel\extract_semantics.py factory        # game files; levels default from the map config
python extraction\intel\extract_gamedata.py factory         # game files; TYPED exfils/minefields/doors (ground truth)
python extraction\intel\build_loot.py                       # internet (tarkov.dev), all maps at once
python extraction\intel\build_tasks.py                      # internet (tarkov.dev), all maps at once
```

Outputs land in `EFT_TARKMAP_ROOT\out\` (`<map>\semantics.json`, `<map>\gamedata.json`,
`loot.json`, `tasks.json`). Re-run loot/tasks each wipe — prices, containers and quests shift.

`extract_gamedata.py` reads the map's TYPED gameplay MonoBehaviours (ExfiltrationPoint /
Minefield / SniperFiringZone / Door / Trunk / TransitPoint / StationaryWeapon /
SpawnPointMarker) instead of name-matching GameObjects; in the viewer its exfils (with
faction + collider footprint outlines) replace the tarkov.dev extract layer, and minefields /
sniper zones get their own outline layers. If the map's config levels carry no exfils it
auto-probes the sibling scenes from BuildSettings for the logic scene (factory: level 68).
Copy the output next to the pack like semantics.json (step 8).

## 7. Grade LUT (already done)

The kit ships the prebuilt `extraction\grade\eft_grade_lut.bin` (baked from the game's OWN
grading LUT) and `--init` already copied it to `EFT_TARKMAP_ROOT\out\`. Nothing to do.

To regenerate it: `python extraction\grade\make_grade_lut_game.py` (the game-LUT source strip
ships next to the script; its docstring explains re-extracting the strip from your own
`resources.assets`). `make_grade_lut.py` is the older reconstructed-fit LUT (legacy look,
needs scipy) — kept for provenance, not what packs ship.

## 8. Assemble the pack

From the **repo root** (so the vendored `eft_pipeline` resolves), with the env vars set:

```powershell
cd <this repo>
python -m eft_pipeline.assemble_bevy factory --out .\packs\factory.eftpack
```

`loot.json`, `tasks.json` and `grade_lut.bin` are copied into the pack automatically from
`EFT_TARKMAP_ROOT\out\`; the SH volume / lights / terrain layers are referenced in place.
Then:

```powershell
# grass into the pack (OUTDOOR maps only — errors out with "0 clumps" on indoor maps, that's fine)
python -m eft_pipeline.build_grass --pack .\packs\factory.eftpack

# semantics + typed game data next to the pack (the viewer looks for <pack>\semantics.json /
# <pack>\gamedata.json, or set EFT_SEMANTICS_JSON / EFT_GAMEDATA_JSON)
copy "$env:EFT_TARKMAP_ROOT\out\factory\semantics.json" .\packs\factory.eftpack\
copy "$env:EFT_TARKMAP_ROOT\out\factory\gamedata.json" .\packs\factory.eftpack\
```

Note: if `build_grass` warns about a missing cross-map fallback albedo, it means your dataset's
`terrain_layers\` has no `grass_*_D.png` — re-run step 3 for that map.

## 9. Run the viewer

```powershell
cargo run --release -- .\packs\factory.eftpack
```

First build compiles Bevy/wgpu (a few minutes, several GB in `target\`). Controls: RMB =
mouse-look, WASD = move, Q/E = down/up, Shift = fast. See the repo README for the optional
`--features egui` stats overlay and env toggles (`EFT_SHADOWS=1`, `EFT_DETAIL_FADE`, …).

---

## Troubleshooting

- **`EFT_TARKMAP_ROOT is not set`** — step 1. Every script that needs it says so explicitly.
- **UnityPy errors on load / weird attribute errors** — check `pip show UnityPy` is 1.25.0
  (the pin exists because minor versions change the object API).
- **A level file is "missing"** — level indices shift when the game patches. The geometry lists
  in `extraction\maps\<map>\config.json` and the lights table above are current as of 2026-07.
  If a patch moved them, the level count in `EFT_GAME_DATA` changes; scan for the new `*_light`
  scene by extracting candidate levels with `eft_extract_lights.py` (a wrong level yields 0-few
  lights, the right one hundreds+).
- **Huge maps (Streets)** — extraction takes hours and the dataset is the multi-GB end. Packs
  over 2 GiB of geometry are split into parts automatically by the emitter and merged by the
  viewer; nothing extra to do.
- **`bake_volume2` FAIL verdict** — usually a missing/empty `lights_*.json` (step 4) or a
  dataset extracted with different culls than the config expects. Re-check step 4 output counts.
- **check_env warns about warp/CUDA** — only the SH bake needs it; all other steps run anywhere.
