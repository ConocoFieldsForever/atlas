# What may and may not be redistributed

This project renders maps for raid planning from data the END USER extracts from their own
legally owned Escape from Tarkov installation.

**Never redistribute (never in the repo, never in a zip, never in CI):**
- `packs/*.eftpack` and everything inside them (geometry, textures, lighting volumes,
  gamedata) — derived from Battlestate Games' copyrighted game files.
- The extracted datasets (`eft_assets/…`) and any intermediate artifact of the extraction
  (scene.json, meshes, lights, density grids, baked volumes).
- The game files themselves, obviously.

**Redistributable:**
- The viewer executable, shaders, and this toolchain's source code.
- The python extraction/pipeline kit (code only — it produces the non-redistributable data
  on the user's machine, from the user's install).

**tarkov.dev data** (loot/quest catalogs, item icons): fetched by the END USER at
build/run time via their API/CDN, not bundled. tarkov.dev's API is free to use; credit
them in anything public. https://tarkov.dev

**Known gray area:** `extraction/grade/eft_grade_lut.bin` is a 1 MB display color
transform fitted from the game's own post-processing profile. It is committed for
convenience; if that is ever a concern, delete it and regenerate locally with
`extraction/grade/make_grade_lut_game.py` from your own install.

CI builds the executable and zips the code kit only. All pack building happens on end-user
machines.
