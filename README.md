# eft_native_viewer

Native desktop **Escape from Tarkov map viewer** — Rust + Bevy (GPU-driven),
replacing the web three.js `tarkmap`. This repo is **code-only**: it *consumes*
the `.eftpack` v1 packs emitted by `eft_pipeline/assemble_bevy.py` and references
textures / terrain / lights / SH volume **in place** under
`C:\Users\user\beamng_blender_pipeline\eft_assets` (nothing is copied here —
disk is tight).

The extraction pipeline (UnityPy extractor → `assemble_bevy.py`) is the scene
**source of truth**; this engine is a pure **consumer**.

## Layout

```
eft_native_viewer/
  Cargo.toml               workspace root (pinned deps, profiles)
  eft_pipeline/            the .eftpack emitter (Python)
    __init__.py
    assemble_bevy.py       forks tarkmap/assemble_instanced.py → emits .eftpack v1
    tarkmap_core/          correctness code VENDORED verbatim from the upstream tarkmap
      __init__.py
      instmath.py  culls.py  objio.py  matsig.py   (verbatim)
      config.py    (verbatim EXCEPT ROOT/MAPS_DIR repointed at the in-place tree)
  viewer/
    Cargo.toml             the eft_viewer binary crate
    src/
      main.rs              Bevy app: window, fly camera, load .eftpack, install render plugin
      eftpack.rs           .eftpack v1 loader — reads strides/offsets FROM manifest.json
      render/
        mod.rs
        instancing.rs      M0 WORKING custom instanced draw (first-pixel)
        gpu_driven.rs      M1 GPU-driven design (cull/indirect POD layouts + frustum math)
    assets/shaders/
      instancing_m0.wgsl   the WIRED M0 shader (full 3×4 affine + cofactor normals)
      instanced.wgsl       M3 material shader (bindless + SH-GI + vert-paint)   [not yet wired]
      cull.wgsl grade.wgsl sh_gi.wgsl splat.wgsl                                [M1–M4 ports]
      SHADER_PORT_MAP.md
```

## Build the pack, then run

The pack is built on demand (one `.eftpack` per map — disk is tight). Run the
emitter as a **module from the repo root** so the vendored core resolves:

```powershell
cd C:\Users\user\eft_native_viewer

# 1) build the interchange pack (M0 TARGET). No --limit = full map.
python -m eft_pipeline.assemble_bevy interchange --out .\packs\interchange.eftpack

# 2) build + run the viewer against it
cargo run --release -- .\packs\interchange.eftpack
```

`--limit N` caps the number of (mesh,material) groups for a fast smoke pack
(truncates bounds + instances — not a usable full render).

Toolchain: stable Rust (1.82+). First build compiles Bevy/wgpu (a few minutes,
several GB in `target/` — mind the disk).

Controls: **RMB** = mouse-look · **WASD** = move · **Q/E** = down/up · **Shift** =
fast. With no pack argument the window still opens (empty) so you can verify the
app boots.

### Optional egui overlay

A small stats overlay is behind a non-default `egui` feature (kept off by default
so the base build never blocks on a `bevy_egui` release lagging a fresh Bevy
point version):

```powershell
cargo run --release --features egui -- <pack-dir>
```

## Version pinning

Engine is **locked to the Bevy 0.17 era** (verified resolved: bevy 0.17.3, wgpu
26.0.1, glam 0.30.10, bevy_egui 0.37.1). Pins live in the workspace root
`Cargo.toml`:

| crate      | pin    | note |
|------------|--------|------|
| bevy       | 0.17   | GPU-driven render path; all GPU types come via `bevy::render` re-exports |
| glam       | 0.30   | `Affine3A` carries the full 3×4 incl. shear |
| bytemuck   | 1.20   | POD instance / indirect-arg structs |
| image      | 0.25   | texture import (M2) |
| serde/json | 1      | manifest / materials parsing |
| bevy_egui  | 0.37   | optional overlay; align to the Bevy release if it drifts |

The viewer crate has **no direct `wgpu` dependency** — every GPU type is used
through `bevy::render` (its re-exported wgpu), so two wgpu versions can never be
linked. If you add low-level wgpu, take it from Bevy's re-export.

## What the viewer does today (M0)

- **Loads a full `.eftpack`** honoring the self-describing contract: vertex and
  instance **strides + field offsets are read from `manifest.json`**, never
  hardcoded, so the emitter and loader cannot drift. It also reads the emitter's
  **`conventions`** block (uvVFlipBaked / uvTilingBaked / normalMapGreenFlip) so a
  shader can't double-apply a flip. `materials.json` `emissive` parses as the
  `{texture,factor,hdr}` object the emitter actually writes (the earlier
  `Option<String>` typing crashed the whole load on any lit material).
- **Draws the pack** via the M0 **custom instanced path** (`render/instancing.rs`,
  the Bevy-0.17 low-level custom-instancing pattern): one Bevy `Mesh` + one
  instanced draw per unique eftpack mesh. The vertex shader applies the **FULL
  ROW-MAJOR 3×4 affine (incl shear/mirror) to RAW verts** and transforms normals
  by the **cofactor matrix** (det·inverse-transpose). **The #1 rule is obeyed:
  no TRS-decompose, ever.** Mirrors (det<0) render correctly with **zero baking**
  because the pipeline is **double-sided** (`cull_mode = None`) and the cofactor
  matrix flips the normal sign for det<0.
- **Fly camera** framed on the pack's world-AABB `bounds`; flat sun+ambient
  lambert so geometry reads honestly. (Textures/materials are M2–M4.)

## Design center (locked)

Low-overhead **GPU-driven instancing**, **not** meshlets/virtual-geometry
(largely moot for this already-instanced low-poly data — p50 ~384 tris, ~10.5k
unique meshes stored once). The M1 path (`render/gpu_driven.rs`) is:

1. one shared vertex buffer + one shared index buffer (unique meshes stored once),
2. one instance storage buffer (`Vec<GpuInstance>`),
3. bindless material/texture indexing (BC7 albedo / BC5 normal binding arrays),
4. a **compute cull** that frustum-tests each instance's *shear-correct* world
   bounding sphere (max column-norm radius scale — no decompose), applies the
   screen-height LOD predicate (ported `_lod.js`), and **compacts** survivors +
   builds `DrawIndexedIndirectArgs`,
5. **indirect multidraw** of the compacted set.

## Milestone punch list (to full parity)

- **M0 (this scaffold): first-pixel.** Custom instanced draw, full-affine placement,
  cofactor normals, double-sided mirror-correct. **Needs a `cargo build` pass** to
  shake out any 0.17.3 API drift (the crate resolves but has not been compiled).
- **M1: GPU-driven.** Move from per-mesh instanced draws to the compute
  frustum(+occlusion) cull → compaction → indirect multidraw in `gpu_driven.rs`
  (shared buffers, prefix-sum for per-batch `first_instance`, mesh→batch map, HZB).
  Fold in screen-height LOD. Switch the phase to Opaque3d/binned.
- **M2: materials/textures.** Bindless BC7 albedo / BC5 normal arrays, per-submesh
  material routing, `_MainTex * _Color`, cutout, glass/water, emissive.
- **M3: EFT shaders (`instanced.wgsl`).** Vert-paint 3-layer height splat, per-pixel
  normal mapping (DirectX green-flip honored per convention), SH-L1 GI from the
  volume as a real 3D texture.
- **M4: display chain.** Grade 3D-LUT + `Tonemapping::None` on a non-sRGB target,
  analytic sky/IBL, eye adaptation.
- **M5: terrain + grass.** MicroSplat terrain, deterministic grass, depth-bias
  z-fight polish, semantics.

## .eftpack v1 contract (consumed here)

```
<pack>/manifest.json   version, dataset, bounds[6], vertex{stride,attrs[]},
                       instance{stride,fields[]}, meshes[]{id,name,vtx/idx off+count,
                       submeshes[]{materialId,idxStart,idxCount}}, counts, roots[],
                       conventions{uvVFlipBaked,uvTilingBaked,normalMapGreenFlip,...},
                       sidecars{terrainLayers,lights,volume,semantics,...}  (abs paths, in place)
<pack>/meshes.bin      interleaved verts (pos f32x3 @0, normal f32x3 @12, uv f32x2 @24,
                       color unorm8x4 @32; stride 36) then u32 indices
<pack>/instances.bin   fixed-stride records (stride 80): affine f32x12 (ROW-MAJOR 3×4 incl
                       shear) @0, meshId u32 @48, lodGroup i32 @52, lodIndex i32 @56,
                       rootId u32 @60, flags u32 @64
<pack>/materials.json  [{id, role:opaque|cutout|glass|decal|water, albedo, normal,
                        uvXform[4], alphaMode, alphaCutoff, tint[4], metallic?, roughness?,
                        normalScale, normalGreenFlip, doubleSided, emissive:{texture,factor,hdr}|null,
                        roughnessFromAlbedoAlpha, specMap?, vp?}]
```
```
instance.flags bits:  0x1 MIRROR (det<0)   0x2 TERRAIN   0x4 BAKED_WORLD
```
