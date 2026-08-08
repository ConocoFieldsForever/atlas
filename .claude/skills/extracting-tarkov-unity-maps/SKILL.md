---
name: extracting-tarkov-unity-maps
description: >
  Documents the Escape From Tarkov Unity asset extraction pipeline: game files to dataset
  (scene.json + OBJ + PNG) to .eftpack (manifest.json, meshes.bin, instances.bin, materials.json,
  sidecars) to renderer. Covers the instance placement formula and the ban on TRS decomposition,
  the handedness conjugation, the OBJ vertex X-negation, the texture V-flip, terrain and MicroSplat,
  StaticDeferredDecal projectors, characters and skinned meshes, sky, particles, water, colliders,
  the light and SH irradiance bake, IL2CPP MonoBehaviour raw-payload reading, and the failure
  signature of every broken invariant. Use when extracting, placing, reimplementing (a Blender or
  Unreal importer), or debugging EFT/Unity map geometry, textures, decals, lighting, gameplay data,
  or pack builds, and BEFORE editing any placement, coordinate, UV, or handedness convention.
  Keywords: Tarkov, EFT, Unity, UnityPy, scene.json, eftpack, decal projector, handedness,
  conjugation, shear, V-flip, MicroSplat, SH irradiance volume, IL2CPP.
---

# Extracting Tarkov Unity maps

**Read `docs/extraction/README.md` now.** It is the navigation layer and it routes to ten
reference documents covering geometry and placement, textures and materials, terrain and the colour
grade, decals, game data, colliders and the semantic name layer, characters and animation, sky and
particles and water, lighting and the irradiance bake, and the build stages and pack format.

This file is only a discovery shim. The documentation lives in the repository, in plain markdown
with no tool-specific format, so that any agent, any importer author, and any person reading the
code can use the same source. Do not answer pipeline questions from this file alone.

The rules most often got wrong, so that a wrong turn is caught before the full read:

- Apply the raw 3x3 to vertices. **Never** decompose a world matrix to translation/rotation/scale;
  about 4% of instances carry legitimate shear that decomposition silently discards.
- Handedness is a **conjugation** `M' = G·M·G⁻¹` with `G = diag(-1,1,1)`, applied exactly once.
  Not a premultiply, and never together with reflecting the mesh vertices.
- OBJ vertices are already X-negated. That fixes local shape, not world handedness.
- The texture V-flip is baked into the UVs after tiling, once. There is never a U-flip.
- Decal projectors span local X and Z and project along local Y.
