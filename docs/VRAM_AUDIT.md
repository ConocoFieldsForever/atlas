# VRAM audit — where Atlas's GPU memory goes, and what an overlay user can turn off

Context: Atlas runs as an overlay **while EFT itself owns the GPU**. Streets of Tarkov is the
worst case — game + Atlas together have been observed near the 32 GiB ceiling of this machine's
RTX 5090 (`nvidia-smi` snapshots this session: 28,031 MiB used with the game in raid; 7,704 MiB
with the game at menu and no Atlas). GPU contention has already produced one TDR-class crash
(`viewer/src/gpu_lease.rs:1-11`), and the lease file's own doc states the policy this audit
serves: *fail SAFE — VRAM headroom is a first-class goal*. Frame-rate is already throttled when
unfocused (`main.rs:850-857`) and when the overlay is hidden (`overlay.rs:316-327`), but **nothing
is ever deallocated** — every throttle is scheduling-only.

All numbers below are measured from the shipped packs on disk (PNG headers + manifest sums +
`instances.bin` decoded by its manifest layout) and from the allocation sites in the code. Where a
number is a model (BC-compressed size with full mip chain) the formula is stated. Measurement
script: 16 B per 4×4 block per mip level for BC3/BC5, `mips = floor(log2(maxdim))+1`.

---

## 1. Where the VRAM actually goes

### 1.1 Allocation inventory (GPU-driven path, `render/gpu_driven.rs`)

| Allocation | Site | Size law | Streets (measured) |
|---|---|---|---|
| **Vertex buffer** `eft_gpu_vertex` | `:3290` (async) / `:3444` (sync) | verts × **52 B** (`DRAW_VERTEX_STRIDE`, `:167`) | 57,086,691 verts → **2,831 MiB** |
| **Index buffer** `eft_gpu_index` | `:3296` / `:3449` | indices × 4 B (u32) | 178,725,858 → **682 MiB** |
| **Instance SSBO** `eft_gpu_instances` | `:3457` | instances × **80 B** (`InstanceGpuRecord`, `:104`) | 173,260 → 13.2 MiB |
| **Mesh meta SSBO** | `:3593` | meshes × 32 B (`MeshMeta`, `:122`) | 47,304 → 1.5 MiB |
| **Visible buffer** | `:3598` | instances × 4 B | 0.7 MiB |
| **Indirect ×2** (opaque + blend) | `:3604`, `:3610` | meshes × 20 B × 2 | 1.9 MiB |
| **LOD centers SSBO** | `:3633` | lodGroups × 16 B | 137,311 → 2.2 MiB |
| **Material table SSBO** | `:3662` | materials × **192 B** (`GpuMaterial`, `:250`) | 17,803 → 3.3 MiB |
| **Albedo bindless array** | upload at `:4624` (BC3) / `:4698` (RGBA8) | Σ per-texture BC3 + full mips | 2,675 uniques → **3,241 MiB** |
| **Normal bindless array** | `:4662` (BC5) | Σ per-texture BC5 + full mips | 2,021 uniques → **1,761 MiB** |
| **Detail/parallax/emissive** (appended to same arrays) | material build | Σ BC3 | 98 uniques → 52 MiB |
| **SH volume** — 3× `Rgba16Float` 3D textures | `:3838` | nx·ny·nz × 8 B × 3 | streets ships **no volume sidecar → 3×1×1×1 dummy (0)**; interchange 401×13×302 → 37.8 MiB |
| **Shadow atlas** `eft_shadow_depth` | `:3933` | `SHADOW_MAP_SIZE`² × `Depth32Float` × 2 cascades (`:613-615`, 2048²) | **32 MiB — allocated even when shadows are OFF** (layout-stability note `:607-610`) |
| Light records + CSR grid | `:4111-4120`, build `:458-590` | 48 B/light + CSR; 8 B degenerate grid on full-bake packs; ≤ ~16 MiB at the 4 M-cell clamp (`LIGHT_GRID_MAX_CELLS`, `:413`) | < 1 MiB |
| Grade LUT (64³ `Rgba16Float`) | `grade.rs:172-190` | fixed | 2.0 MiB |
| Skybox cubemap (6×128², HDR) | `main.rs:972-974` | fixed | ~0.8 MiB |
| SSAO | `ssao.rs:76-81` | 96 B uniform, **no render target** (writes into the HDR ping-pong) | ~0 |

**View targets** (per-frame, resolution-scaled — window is 1600×1000 logical, `main.rs:738`):

| Target | Law | @1600×1000 |
|---|---|---|
| MSAA color, `Rgba16Float` ×4 samples | px × 8 B × 4 | 51.2 MB |
| HDR resolve ping-pong ×2 | px × 8 B × 2 | 25.6 MB |
| Depth `Depth32Float` ×4 samples (+`TEXTURE_BINDING`, `main.rs:1128`) | px × 4 B × 4 | 25.6 MB |
| Bloom mip pyramid (`Rg11b10Ufloat`, max mip 512) | ~px/3 | ~2.2 MB |
| **Subtotal** | | **~105 MB** |

MSAA is **Bevy's default Sample4 — no write site exists anywhere in the codebase** (only read
sites: `instancing.rs:205`, `gpu_driven.rs:5372`, `ssao.rs:164`). HDR is forced by the `Hdr`
marker on the single camera (`main.rs:1136`). There is no render-scale mechanism; the only thing
that resizes the target is the overlay itself (`overlay.rs:250-257` shrinks the window to
`size_frac` of the monitor — overlay-up is actually *smaller* than the desktop window).

### 1.2 Key structural facts

* **Everything is resident, always.** Every mesh referenced by ≥1 instance is uploaded in full
  (orphan meshes are skipped, `gpu_driven.rs:2133-2137`; on streets all 47,304 meshes are
  referenced). Every unique texture in `materials.json` is uploaded at **full source resolution
  with a full mip chain**, BC3 (albedo, `bc3_compress_chain` `:4431`) / BC5 (normals, `:4495`)
  when ≥64 px (`bc_wanted` `:4603`), regardless of visibility or camera position. Culling
  (frustum + screen-size + distance-LOD) selects what is *drawn*, never what is *resident*.
  There is no residency budget and no eviction.
* **No vertex duplication in the 36 B→52 B repack.** Per-vertex material tagging needs no
  splitting because submeshes reference disjoint vertex sets (measured claim in the source,
  `gpu_driven.rs:2167-2176`). So GPU geometry = verts×52 + idx×4, exactly.
* **The async load streams uploads over frames** (`:3234-3433`) — a load-time smoothness fix
  only; final residency is identical to the sync path.
* **Streets is a lean (LOD0-only) pack**: decoded `instances.bin` lodIndex histogram
  `{-1: 5,593, 0: 167,667}`. There are no coarse shells to fall back to (see §2 "LOD floor").
  factory_rework is the only all-LOD pack (`{0: 32,535, 1: 4,616, 2: 1,678, 3: 492}`).
* The **shared BC texcache** (`packs/shared/texcache`, 8,857 entries, ~11 GB) stores the
  *concatenated per-mip* BC payload with a `[w,h,mips]` header (`texcache_write` `:4571`) —
  this is what makes a "skip top mips" toggle nearly free (§2).
* CPU RAM (not VRAM, but the same machine under load): the repacked `CpuData` staging blob IS
  freed after upload, but with a two-copy overlap window during the async load
  (main-world copy dropped by `free_cpu_staging` `:1323-1350`, render-world copy at `:3201-3203`).
  `Pack.meshes_bin` (2.77 GB on streets) is **never freed** — `LoadedPack(Arc<Pack>)` lives for
  the app's lifetime and pick/nav/walk-ground/bakes read it in place (`eftpack.rs:1157,1164`).
  PERF_UPLOAD_SPEC tracks the copy-reduction work.

### 1.3 Per-pack resident-VRAM estimate (as the code loads them today)

| Pack | Geometry (vtx+idx) | Textures (BC + mips) | SH vol | Buffers/misc | **Total** |
|---|---|---|---|---|---|
| **streets** | 2,831 + 682 = **3,513 MiB** | **5,027 MiB** (alb 3,241 / nrm 1,761 / extra 52 — 4,794 uniques) | 0 (none shipped) | ~25 MiB | **~8.5 GiB** + 32 MiB shadow + ~120 MiB targets ≈ **8.7 GiB** |
| interchange | 760 + 189 = 949 MiB | 1,877 MiB | 37.8 MiB | ~15 MiB | ~2.9 GiB |
| factory_rework | 265 + 61 = 326 MiB | 1,457 MiB | 1.6 MiB | ~10 MiB | ~1.8 GiB |
| icebreaker | 102 + 25 = 127 MiB | 550 MiB | 2.2 MiB | ~12 MiB | ~0.7 GiB |

Texture resolution census (streets, from PNG headers): albedo max-dim histogram
`{2048: 231, 1024: 1596, 512: 727, 256: 101, ≤128: 20}`; normals
`{2048: 43, 1024: 1093, 512: 784, 256: 85, ≤128: 16}`. Interchange has four 4096² albedos.

**Verdict: on streets it's ~59% textures, ~40% geometry, and everything else is noise.**
The two levers that matter are texture mip residency and (much harder) geometry bytes.

---

## 2. Menu toggles evaluated

### 2.1 Texture resolution cap — the big one (skip top mip level(s) at upload)

* **Saving (streets, measured per-texture):** full **5,027 MiB** → skip-1-mip **1,257 MiB
  (−3.77 GiB)** → skip-2 **314 MiB (−4.71 GiB)**. (A fixed 1024 px cap saves only ~0.9 GiB
  because most textures are already 1024 — the per-texture "drop N mips" form is strictly
  better.) Interchange: 1,877 → 469 → 117 MiB.
* **Cost: S.** The BC payloads are *already stored as concatenated mip chains* both in the
  shared texcache and in `TexCpu::{Bc3,Bc5,Raw}`. Uploading from mip N is: compute the byte
  offset of level N (the loop in `bc3_payload_len`, `gpu_driven.rs:4544-4549`), create the
  texture with dims `(w>>N, h>>N)` and `mips - N` levels, pass the sliced payload. Hooks:
  `upload_prepared` (`:4852-4871`) + the three `upload_*` helpers (`:4614`, `:4653`, `:4688`);
  the setting must be read at map-load time (it changes what is uploaded, so applying it means a
  pack reload — acceptable; see §3c). Guard: only skip while both dims stay ≥ 4-multiples and
  ≥ 64 px (BC block constraint; game textures are POT so this is nearly always true), and never
  skip terrain control maps (`ctrl_tex_linear` — they're blend weights, `:3267`).
* **No warm-cache invalidation:** the texcache entry is unchanged; only the upload slices it.
  Do NOT re-encode at lower res (rejected pack-time BC pre-encode precedent, PERF_FINDINGS §Rejected).
* Interaction with culling path: none (the cull never touches textures). M0 fallback: no-op
  (M0 is untextured, `instancing.rs:1-21`).
* Visual cost: half-res textures in a 55%-of-monitor overlay window are near-invisible;
  anisotropy (8×, `:3769`) still applies to the remaining chain.

### 2.2 Normal maps off

* **Saving:** streets **−1,761 MiB** at full res (−440 MiB when combined with skip-1). The
  shader already has a complete no-normal-map path (`normal_index == NO_NORMAL` → geometric
  normal, `:198-200`), so this is: don't spawn the normal prep tasks (`:3274-3283`), push the
  1×1 dummy for every slot (`make_dummy_normal_texture` `:4398`), `normal_count = 1`.
* **Cost: S** (load-time setting, like 2.1). Interaction: none — the bindless array stays valid.
* Visual cost: flat surface response; noticeable up close, irrelevant for map-navigation use.

### 2.3 LOD floor (draw/upload only coarse shells)

* **Honest answer: unavailable on streets.** Streets ships LOD0 only (§1.2); there are no
  coarser shells to keep. On the one all-LOD pack (factory_rework), uploading only lod ≥ 1
  would save roughly the LOD0-exclusive geometry (its LOD0-only meshes are 4.04 M of 5.35 M
  verts) — but factory is already the smallest problem (326 MiB geometry).
* Going all-LOD on streets to enable this is the **wrong direction for VRAM**: docs/LOD_AUDIT.md
  §3.2 measured all-LOD streets at ~3.87 GiB meshes.bin (×1.47) — multi-LOD **adds** memory to
  save raster time. A hypothetical "coarsest-shell-only" pack build would genuinely shrink
  geometry, but it requires the `--alllod` re-extract (≈10 min parallel, LOD_AUDIT §1.3) plus a
  new pack variant. **Defer**; not a menu toggle.

### 2.4 Draw distance / far-plane cut

* **VRAM saved: 0** (residency is unconditional). It is a GPU-*time* lever, and the best form is
  already identified in LOD_AUDIT §3.4: honouring every LODGroup's own last `srh` as a cull
  (not just the 442 `lastIsBillboard` groups, `gpu_driven.rs:1435-1440`) cuts drawn triangles
  to **2–9%** on interchange with zero bytes and no format change. That belongs in the same
  Performance section as a "Draw distance" slider mapped to `lod_bias`/cull-height scale
  (cull uniform is live, no rebuild — `CullUniform.lod_params`, `:150-154`). Effort M (cull WGSL
  + plumb). For overlay GPU-sharing this is the top *frame-time* item even though it saves no VRAM.

### 2.5 Disable SH ambient volumes

* **Saving: 0 on streets** (ships no volume sidecar — the loader already warns and binds a
  1×1×1 dummy, `:3820-3828`); 37.8 MiB on interchange (the largest). Sampling cost is trivial.
  **Not worth a toggle.**

### 2.6 Render-resolution scale

* No mechanism exists (no `scale_factor_override`, no off-screen target — everything renders to
  the swapchain-derived `ViewTarget`). The overlay already shrinks the window physically
  (`overlay.rs:250-257`), which is the same thing with better text rendering. Target VRAM is
  only ~105 MB anyway. **Skip as a VRAM lever**; revisit only as a frame-time lever (M effort:
  `RenderTarget::Image` + blit, or window-resolution downscale).

### 2.7 MSAA off

* **Saving:** ~64 MB @1600×1000 (MSAA color 38.4 + MSAA depth 19.2 + resolve overhead), scaling
  linearly with pixels (~147 MB overhead at 1440p full-screen). Also a real raster-time save on
  a 178 M-index scene.
* **Cost: S.** `Msaa` is a per-camera component in Bevy 0.17; insert `Msaa::Off` in
  `apply_gfx_camera` (`main.rs:467-534`), which already does the insert/remove dance for Bloom/
  DoF/chroma. Both pipeline paths re-specialize automatically (they key on the view's `Msaa`,
  `gpu_driven.rs:5372`, `instancing.rs:205`).
* **One coupling:** SSAO requires a multisampled depth view and silently no-ops at 1 sample
  (`ssao.rs:164-166`). SSAO defaults off (`render/mod.rs:148`), so: grey out SSAO when MSAA is off.

### 2.8 Shadow map size / off

* Toggle exists (`GfxSettings.shadows` → `sync_gfx_shadow_toggle`, `gpu_driven.rs:690`), but the
  2048²×2 `Depth32Float` atlas is **allocated unconditionally** for group(3) layout stability
  (`:607-610`) — toggling saves GPU time (4.4–6.0 ms/frame on the 5090 bench, PERF_FINDINGS) but
  **0 bytes**. Making size a load-time setting (2048→1024 = 32→8 MiB, or a 1×1 dummy when
  vetoed at load) is S–M. VRAM impact is small; treat it as the frame-time toggle it already is.

### 2.9 HDR off

* Would halve the ~77 MB of color targets (Rgba16Float→Rgba8) but breaks Bloom (`#[require(Hdr)]`),
  the grade LUT chain (`grade.rs:312` skips non-HDR views) and emissive glow — the entire
  post stack is built around the HDR target (`main.rs:1134-1140`). **Reject: high blast radius,
  ~40 MB.**

---

## 3. Is asset streaming feasible?

**(a) On-demand texture residency from GPU visibility feedback — feasible but not first.**
wgpu 26 has **no sparse/tiled resources and no partial texture residency**; "streaming" a mip
means creating a *new* texture + view and rebuilding the bindless bind group (~4,800 views,
`:3802-3813`). That rebuild is fine at low frequency (it's one render-thread call), but the
feedback loop (readback of per-material visibility from the cull pass) adds a GPU→CPU path the
codebase deliberately has none of (PERF_UPLOAD_SPEC: "no hot-path readbacks"). A **simpler
CPU-side variant needs no readback at all**: resident base = skip-2 everywhere (314 MiB), promote
textures whose *materials' instances* come within R meters of the camera (instance translations
are already CPU-side) under a fixed budget, demote LRU beyond it. That is the only true-streaming
design worth building here, and only after §2.1 ships and proves insufficient. Effort M–L.

**(b) Per-region mesh residency — reject.** There is no spatial partitioning in the pack:
instances are a flat array (`GpuInstance` carries no cell/region id, `eftpack.rs:409-418`),
geometry is two monolithic buffers addressed by `MeshMeta.base_vertex/first_index`, `mesh_meta`
assumes each mesh's instances are *contiguous* (`instance_base/instance_count`,
`gpu_driven.rs:2382-2390`), and meshes are shared by instances map-wide (a fence mesh appears
everywhere). The raw ingredients for binning do exist — per-instance world spheres are already
computed CPU-side (`:2328-2338`) and the light grid's CSR two-pass (`:524-574`) is a reusable
template — but residency would still need a buffer sub-allocator, indirection rebuilds, and a
defrag story inside the GPU-driven indirect path — L effort, high risk, and it fights the
architecture. Only terrain has a genuine grid identity (MicroSplat `Slice_R_C` tiles). The 3.5 GiB
geometry floor is better attacked by **vertex compaction** (52 B → ~28 B: oct-encoded normals
u16×2, f16 UVs, unorm8 color — saves ~1.3 GiB on streets, deliberately deferred in
PERF_UPLOAD_SPEC's "do not implement" list, so it needs its own sign-off) — a format project,
not streaming.

**(c) Overlay-mode preset reload — recommended pragmatic path.** The infrastructure already
exists: in-place pack swap (`EFT_SWITCH` soak-tested), async streamed build (`:3234`), warm
shared texcache (reload is seconds, not the cold 40 s). A "low-VRAM profile" is then just §2.1 +
§2.2 settings applied at load, and "apply" = reload current map. No new architecture.

**(d) wgpu 26 reality check.** Available: mip-sliced uploads (§2.1), bind-group recreation,
per-texture replace in the binding_array. Not available: sparse binding, residency priorities,
memory-budget queries (wgpu exposes no DXGI/NVML budget; if auto-degradation is ever wanted, poll
NVML out-of-band like `nvidia-smi` does). Anything relying on `MAP_WRITE` primary buffers is
already ruled out (PERF_UPLOAD_SPEC constraint 2).

---

## 4. Quick-wins ranking (streets numbers)

| # | Toggle | VRAM saved (streets) | Effort | Hooks |
|---|---|---|---|---|
| 1 | **Texture quality: Half** (skip 1 mip/texture) | **−3.77 GiB** (5.03→1.26) | S | `upload_prepared` `gpu_driven.rs:4852` + `upload_bc3/bc5/rgba8_chain` `:4614/:4653/:4688`; load-time |
| 2 | **Texture quality: Quarter** (skip 2) | **−4.71 GiB** | S | same |
| 3 | **Normal maps off** | −1.76 GiB (full-res) / −0.33 GiB (at Half) | S | skip tasks `:3274`; dummy `:4398` |
| 4 | **MSAA off** | −64 MB @1600×1000 (+raster time) | S | `apply_gfx_camera` `main.rs:467`; grey out SSAO (`ssao.rs:164`) |
| 5 | **Shadows off** (existing) + shrink/skip atlas | −32 MiB, −4.4–6 ms/frame | S–M | `SHADOW_MAP_SIZE` `gpu_driven.rs:613`; group(3) note `:607` |
| 6 | Draw distance (LODGroup last-`srh` cull + `lod_bias`) | 0 VRAM, −90%+ drawn triangles | M | `gpu_cull.wgsl` + `CullUniform.lod_params` (live) |
| 7 | SH volumes off | 0 (streets) / −38 MiB (interchange) | S | not worth it |
| 8 | Vertex compaction 52→28 B | −1.3 GiB | L (format, gated) | own design doc |

With the low-VRAM profile (Half textures + normals off + MSAA off + shadows off):
**streets ≈ 3.5 GiB geometry + 0.85 GiB textures + ~0.15 GiB rest ≈ 4.5 GiB — about half of
today's ~8.7 GiB.** Quarter + normals-off floors it at ~3.9 GiB; beyond that only the geometry
projects (compaction, coarse-pack) move the needle.

## 5. Recommended "Performance" settings section

Where: the menu settings SidePanel is `menu.rs:2150-2297` (`settings_tab: Option<u8>`; no tab
enum — a literal 3-entry tabs array). Exact hook list:

| step | site |
|---|---|
| i18n key ("Performance") | `i18n.rs:111-113` (enum) + `:315-317` (strings) |
| tab label (becomes index 3) | `menu.rs:2175-2179` |
| tab body arm (insert BEFORE the `_ =>` catch-all — it currently swallows ≥2) | `menu.rs:2191` / `:2277` |
| add `ResMut<GfxSettings>` to `menu_ui` | `menu.rs:1559` param list |
| live-apply + deferred persist pattern | mirror `menu.rs:2289-2304` |
| persistence helpers (flat `atlas.config.json` keys) | `menu.rs:996-1026` (`config_*_pub`/`save_config_*_pub`) |
| startup defaults | `main.rs:816-827` (where `GfxSettings::default()` is built) |

Note: **`GfxSettings` is not persisted today** (env defaults + session only) — the Performance
tab is its first persistence consumer. The in-raid Graphics panel (`ui.rs:1295-1447`) should
mirror the same toggles; copy the shadows pattern end-to-end (`ui.rs:1340` checkbox →
`GfxSettings.shadows` → `ExtractResourcePlugin` `main.rs:831` → render-world
`sync_gfx_shadow_toggle` `gpu_driven.rs:692`). Load-time settings (texture quality, normal maps,
shadow-atlas size) are snapshotted by the build at kickoff (`gpu_driven.rs:3256`); changing them
prompts "Apply = reload map".

Build these five, in order:

1. **Texture quality: Full / Half / Quarter** (default Full desktop, **Half in overlay mode**).
2. **Normal maps: On / Off** (default On desktop, **Off in overlay mode**).
3. **Shadows: On / Off** — already exists (`GfxSettings.shadows`); move/mirror into Performance,
   add atlas-size reduction while there.
4. **MSAA: 4× / Off** (default 4× desktop, **Off in overlay mode**).
5. **Draw distance** slider (maps to `lod_bias` + the new last-`srh` cull) — the frame-time
   companion so the overlay stops costing GPU the game needs.

**Overlay auto-profile:** when `OverlayConfig.enabled` (`overlay.rs:38-40`) is true at startup,
default the four toggles to the overlay values above unless the user has explicitly overridden
them (store an `Option` per setting; `None` = follow profile). This makes the first overlay run
land at ~4.5 GiB on streets with zero user action, while a desktop-mode launch keeps full quality.
Also worth logging at load: one line summarizing estimated resident VRAM (the loader already knows
every size it allocates) so future audits are a grep, not a spreadsheet.
