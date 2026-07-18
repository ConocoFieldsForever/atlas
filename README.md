# Atlas

A fast, native **map viewer and raid planner for Escape from Tarkov**. It renders
each map in 3D and overlays the things you actually plan around — extracts, loot,
keys, hazards, quest objectives — with search and point-to-point routing on top.

It's a normal desktop app: double-click and go. No browser, no account, no
internet required to use it.

> **Legal, in one line:** Atlas only shows maps built from *your own* copy of
> Escape from Tarkov, on your own PC. Map data is never bundled with the app. See
> [LICENSE-NOTES.md](LICENSE-NOTES.md).

---

## Will it run on my PC?

You need:

- **Windows 10 or 11** (64-bit)
- **A graphics card from roughly the last 10 years** (anything that supports
  Vulkan or DirectX 12 — basically any modern laptop or desktop GPU)
- **About 1 GB of free space** for the app, plus **1–10 GB per map** you install

You do **not** need to install Python, a runtime, or anything else just to run
Atlas and view maps.

---

## Getting started (2 minutes)

1. **Unzip** the Atlas folder anywhere you like (Desktop is fine). Keep the files
   together — `atlas.exe`, the `assets` folder, and the `packs` folder must stay
   side by side.
2. **Double-click `atlas.exe`.** The menu opens.
   - The first time, Windows may show a blue **"Windows protected your PC"** box
     because the app isn't code-signed. Click **More info → Run anyway.** (This is
     normal for indie software; nothing is being installed.)
3. You'll see the map list. To actually open a map, you need one installed — see
   the next section.

---

## Getting maps to view

A "map" in Atlas is a folder called a **pack** (ending in `.eftpack`). You get one
of two ways:

### Option A — Someone shares a pack with you (easiest)

If a friend sends you a built map pack:

1. Drop the `.eftpack` folder into the **`packs`** folder next to `atlas.exe`.
2. Restart Atlas (or press it if already listed). The map now shows as **READY**.
3. Press **PLAY**.

That's it — no Python, no game files needed on your end.

### Option B — Build maps from your own game (more involved)

Atlas can build maps straight from your Escape from Tarkov install. This is the
"full kit" and asks a bit more of you:

- You need the **full Atlas download** (the one that includes `bootstrap.ps1`, and
  the `tools` / `eft_pipeline` / `extraction` folders), not the viewer-only zip.
- You need **Python 3.10 or newer** installed.
- You need **Escape from Tarkov installed** on the same PC.

Then:

1. Right-click `bootstrap.ps1` → **Run with PowerShell** (one-time setup — it
   installs what the builder needs and checks your setup).
2. Launch `atlas.exe`. At the bottom of the menu:
   - **GAME INSTALL** — point it at your Tarkov folder (usually auto-fills; if not,
     paste the path and press **SET**).
   - **EXTRACTED ASSETS** — press **CHOOSE…** and pick a folder with plenty of free
     space (the extracted map data lands here — budget **~1–6 GB per map**).
3. **Close the game and its launcher** (the extractor needs the files unlocked),
   then press **BUILD** on a map row. The **first** build of a map runs a one-time
   extraction from your game files and can take a while (tens of minutes for a big
   map); it then assembles the pack. Progress streams live. When it finishes the
   row turns **READY** and you can **PLAY**. Re-building or building other maps
   afterward is much quicker.
4. If a row later says **GAME FILES UPDATED**, the game got patched — press
   **UPDATE** to rebuild it.

> The extraction is resumable — if it's interrupted, pressing BUILD again picks up
> where it left off. An NVIDIA graphics card makes the lighting look better but
> isn't required.

---

## Controls

| Do this | To |
|---|---|
| **W A S D** | move around |
| **Hold right mouse button + move mouse** | look |
| **Q / E** | go down / up |
| **Hold Shift** | move faster |
| **Double-click** something | identify what it is |
| **House icon** (top-left) | go back to the map menu |

The panel on the right has layers (loot, extracts, quests…), search, and routing.
Click a destination to get a walkable route to it.

---

## Troubleshooting

**The menu opens but every map says "NOT INSTALLED."**
That's expected on a fresh copy — Atlas ships with no maps. Get one via Option A or
B above.

**Windows / my antivirus warns about the app.**
Atlas isn't code-signed, so Windows SmartScreen and some antivirus tools flag it as
"unknown." It's safe to allow (**More info → Run anyway**). If antivirus quarantines
`atlas.exe`, add an exception for the Atlas folder.

**The window opens black, or the app closes immediately.**
Update your graphics drivers (NVIDIA/AMD/Intel) — Atlas needs Vulkan or DirectX 12.
If it still won't start, install Microsoft's free **Visual C++ Redistributable
(x64)** and try again.

**The BUILD button does nothing or shows an error.**
Building maps needs the full kit (Option B): the `tools`/`eft_pipeline` folders
beside the exe, Python installed, and your Tarkov install. The viewer-only download
can open and view packs but can't build them.

**BUILD says "no dataset" / the extraction fails.**
Make sure you set **EXTRACTED ASSETS** (a folder with free space) and **GAME
INSTALL** at the bottom of the menu, and that you ran `bootstrap.ps1` (it installs
UnityPy, which the extractor needs). The **game and its launcher must be closed** —
they lock the files the extractor reads.

**The first BUILD looks stuck on the first step for a long time.**
That's the one-time extraction working — it can take tens of minutes on a big map.
Watch the streaming log lines; it's making progress. Later builds skip this step.

**It feels like it's holding back / low frame rate.**
By design, Atlas caps to your monitor's refresh rate and goes nearly idle when its
window isn't focused, so it won't hog your GPU while your game is in front. Keep the
Atlas window focused for full speed.

---

## For developers

The renderer design, the pack format, and the extraction pipeline are documented in
**[ARCHITECTURE.md](ARCHITECTURE.md)**. Quick build from source:

```powershell
cargo run --release -p atlas                                  # menu
cargo run --release -p atlas -- .\packs\interchange.eftpack   # open a pack directly
```

Needs stable Rust 1.88+. A fuller list of environment toggles is in
[README_DIST.md](README_DIST.md).

---

## Credits & data

Map geometry, lighting, and terrain are extracted from *your* game files on *your*
PC. Prices, quests, and item icons come from the free community API at
[tarkov.dev](https://tarkov.dev) — please credit them in anything public. What may
and may not be shared is spelled out in [LICENSE-NOTES.md](LICENSE-NOTES.md).
