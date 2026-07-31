# Notices — what this project contains, and what it deliberately does not

Atlas is a **tool**. It reads a copy of Escape from Tarkov that you already own, on your own
machine, and builds local data files ("packs") that the viewer renders. It is not a game client
and it does not distribute the game.

## No game content is redistributed

This repository and its release archives contain **no assets from Escape from Tarkov**: no
meshes, textures, sounds, animations, bundles, scene data, item databases, or derivatives of any
of these. Everything of that kind is produced on your machine from your own installation and is
written to paths that are excluded from version control (`packs/`, `out/`, `eft_assets/`).

Concretely:

- `packs/` and `out/` are git-ignored in full. They hold extracted geometry, textures, lighting
  bakes, gameplay data, item/bot tables and assembled kits — all game-derived, all local.
- Release archives ship the viewer, its shaders, and the extraction/build scripts. Any
  game-derived file that a build step may have produced locally is removed before packaging
  (see `scripts/release.ps1`).
- Screenshots in this repository are renders produced by the tool. They depict the game's art
  and are included only to document the software's output.

If you find any file here that is game content or a derivative of it, that is a bug — please
open an issue and it will be removed.

## Third-party data fetched at build time

Some build steps download publicly mirrored copies of the game's item, bot and customization
tables to your machine so the tool can interpret your installation's assets. That data is
fetched at build time into ignored paths, is never committed, and is never redistributed by this
project. You are responsible for your own use of it.

## Third-party code

`third_party/bevy_render/` is a vendored fork of a Bevy Engine crate, used under its own MIT and
Apache-2.0 licenses; both license files are included alongside it.

## Trademarks

"Escape from Tarkov" and "Battlestate Games" are trademarks of their respective owners. This
project is unofficial, unaffiliated, and not endorsed by Battlestate Games.
