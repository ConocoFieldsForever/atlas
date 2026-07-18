# Atlas

A fast, native **map viewer and raid planner for Escape from Tarkov**. It renders
each map in 3D from *your own* game files and overlays the things you actually plan
around — extracts, loot, keys, hazards, quest objectives — with search and
point-to-point routing on top.

Built in Rust + [Bevy](https://bevyengine.org) with a GPU-driven renderer, so it
opens instantly and flies around a full map at high frame rates. It is a native
desktop app — no browser, no web server.

> **Legality:** Atlas only renders data extracted from *your own legally-owned*
> Escape from Tarkov install, on your machine. Map data is never shipped with the
> app. See [LICENSE-NOTES.md](LICENSE-NOTES.md).

---

## What it does

- **3D maps** — walk or fly any extracted map with game-accurate geometry,
  lighting, terrain, grass, and color grading.
- **Raid overlays** — extracts, loose loot, loot containers, locked doors + the
  keys/keycards that open them, switches, transits, and hazards.
- **Quests** — objective markers and a quest tracker fed by live
  [tarkov.dev](https://tarkov.dev) data.
- **Routing** — click a destination and get a walkable path (CPU A\* over a baked
  nav grid that respects slopes, steps, drops, and player width — it won't cheat
  through gaps you couldn't fit through).
- **Search** — jump to any item, container, key, or extract by name.
- **Instant map switching** — swap maps without restarting; the app keeps
  rendering while the next one loads.

---

## Is it portable? Do I need Python?

**To just *view* maps: yes, fully portable — and no, no Python.** The viewer is a
single self-contained `atlas.exe` (~90 MB). Unzip it, keep `atlas.exe`, `assets/`,
and your `packs/` folder side by side, and run it. No installer, no admin rights,
no runtime to install. All it needs is:

- **Windows 10 or 11**
- **a GPU that supports Vulkan or DirectX 12** — any reasonably modern discrete or
  integrated GPU qualifies.

You can hand that folder to a friend and it just runs.

**To *build* new maps from game files: you need Python.** The **BUILD** / **UPDATE**
buttons in the menu run a Python extraction pipeline against your EFT install, so
that side needs:

- **Python 3.10+** and the pipeline kit (`eft_pipeline/`, `tools/`, `extraction/`)
  sitting next to the exe,
- **your own Escape from Tarkov install**,
- *(optional)* an **NVIDIA GPU** — used only for the baked-lighting step; without it
  maps still build, just with flatter ambient light.

If the Python kit isn't present, the viewer still launches and views packs normally
— only the build/update features go dark.

**Why can't maps just be shared?** A built map (`.eftpack`) contains geometry,
textures, and lighting derived from Battlestate Games' copyrighted files, so it
isn't redistributable. Each person builds their own from their own install — that's
the part that needs Python. (See [LICENSE-NOTES.md](LICENSE-NOTES.md).)

---

## Quick start

### Just viewing (someone gave you a build + packs)

1. Unzip anywhere. Keep `atlas.exe`, `assets/`, and `packs/` together.
2. Run `atlas.exe`. The menu lists the maps in `packs/`; press **PLAY**.

### Building your own maps

1. Get the full source/kit (this repo) with Python 3.10+ installed.
2. One-time setup:
   ```powershell
   .\bootstrap.ps1        # creates a venv + installs the Python deps + checks your env
   ```
3. Run `atlas.exe`. If your EFT folder isn't auto-detected, set **GAME INSTALL** at
   the bottom of the menu, then press **BUILD** on any map row. Progress streams live
   (lights → lighting bake → assembly → grass → zones → icons → fingerprint).
4. A row marked **GAME FILES UPDATED** means the game patched since that map was
   built — press **UPDATE** to rebuild it.

*(Optional, better lighting):*
```powershell
.\venv\Scripts\pip install -r extraction\requirements-bake.txt   # NVIDIA/CUDA bake
```

### Building from a dev checkout

```powershell
cargo run --release -p atlas                 # menu
cargo run --release -p atlas -- .\packs\interchange.eftpack   # open a pack directly
```
First build compiles Bevy/wgpu (a few minutes, several GB in `target/`). Needs
stable Rust 1.88+.

---

## Controls

| input | action |
|---|---|
| **WASD** | move |
| **RMB (hold)** | mouse-look |
| **Q / E** | down / up |
| **Shift** | move fast |
| **double-click** | identify the geometry / object under the cursor |
| **house icon (top-left rail)** | back to the map menu |

The right-hand panel holds layers, search, quests, routes, and graphics settings.

---

## Handy environment toggles

| var | effect |
|---|---|
| `EFT_PACK=<dir>` | open a pack directly, skipping the menu |
| `EFT_SHADOWS=1` | opt-in real-time sun shadows |
| `EFT_GRADE=0` | disable the game color grade |
| `EFT_FOG=0` | disable distance haze |
| `EFT_UNCAPPED=1` | uncap the frame rate (benchmarking) |
| `EFT_HIDDEN=1` + `EFT_SHOT=out.png` | headless screenshot |
| `EFT_GAME_DATA=<dir>` | override the EFT install location |

A fuller list lives in [README_DIST.md](README_DIST.md).

---

## For developers

The renderer design, the `.eftpack` format contract, and the extraction pipeline
are documented in **[ARCHITECTURE.md](ARCHITECTURE.md)**. In short: the Python
`eft_pipeline` is the scene source of truth and emits `.eftpack` v1; the Rust
`viewer/` crate is a pure consumer that reads every stride/offset from
`manifest.json` (emitter and loader can't drift). Version pins live in the
workspace `Cargo.toml`.

- `viewer/` — the `atlas` binary crate (Bevy app, renderer, UI, pathfinding).
- `eft_pipeline/` — the Python `.eftpack` emitter.
- `tools/`, `extraction/` — the build/extraction/intel scripts the menu shells out to.
- `packs/` — built maps (never committed; game-derived).

---

## Credits & data

Geometry, lighting, terrain, and grass are extracted from *your* game files on
*your* machine. Prices, quests, and item icons come from the free community API at
[tarkov.dev](https://tarkov.dev) — credit them in anything public. What may and may
not be redistributed is spelled out in [LICENSE-NOTES.md](LICENSE-NOTES.md).
