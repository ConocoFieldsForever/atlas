## Contents

- [1. What a StaticDeferredDecal is](#1-what-a-staticdeferreddecal-is)
- [2. Finding the projectors without typetrees](#2-finding-the-projectors-without-typetrees)
- [3. The MonoBehaviour payload - exact byte layout](#3-the-monobehaviour-payload--exact-byte-layout)
- [4. Material and texture resolution (one-hop external table)](#4-material-and-texture-resolution-one-hop-external-table)
- [5. The projector transform](#5-the-projector-transform)
- [6. The projector box: axes, atlas rect, UV](#6-the-projector-box-axes-atlas-rect-uv)
- [7. The projection bake](#7-the-projection-bake)
- [8. The emitted mesh convention](#8-the-emitted-mesh-convention)
- [9. Instance record schema and downstream UV composition](#9-instance-record-schema-and-downstream-uv-composition)
- [10. Invariants and failure signatures](#10-invariants-and-failure-signatures)
- [11. Tunables](#11-tunables)
- [12. Old patterns](#12-old-patterns)

**Marking convention.** Figures and observations tagged **[unverified]** are restatements of measurements recorded in source comments. They are not reproducible from the code itself and were not re-measured; treat them as provenance, not as data.

---

## 1. What a StaticDeferredDecal is

`StaticDeferredDecal` is a custom IL2CPP MonoBehaviour. It carries no mesh, no MeshFilter and no MeshRenderer. It is a **projector**: a GameObject whose world transform defines an oriented box, plus a rectangle into a texture atlas. At render time the engine's deferred pass rasterises the box, reads the depth buffer, reconstructs the world position of every covered pixel, transforms it into the box's local space, and - if it lands inside the box and the surface faces the projection axis - writes the atlas cell's texel over the G-buffer albedo.

Consequences for extraction:

- **A MeshRenderer walk cannot see it.** The component list on the projector GameObject contains a Transform and a MonoBehaviour, nothing renderable. Any extractor that enumerates `MeshRenderer` / `MeshFilter` / `SkinnedMeshRenderer` produces zero output for these objects. All spray paint, graffiti, painted road markings, blood/oil stains and tyre marks in the map are invisible to such a walk. On the reference map there are **1,737** of them across its level files (`extraction/intel/extract_decals.py:5`) **[unverified]** - note that `extraction/intel/extract_decals.py:79` gives **1,269** for the same map (quoted in §6). Both comments exist; the source reconciles them nowhere, so neither count should be relied on.
- **A name search cannot find them either.** Projector GameObject names are recycled from whatever was duplicated to make them. The projector that paints the writings atlas at the reference checkpoint is named `decal_carshadow (70)` (`extraction/intel/extract_decals.py:8`). Do not filter by name; filter by resolved script class.
- **The image is not on the receiving surface's material.** The receiving wall/plate/road has its own untouched material. Nothing about the receiver records that a decal is painted on it.

The extraction goal is to convert each projector into ordinary geometry: triangles that lie on the receiving surfaces, with UVs into the atlas cell, so a downstream importer needs no projector support at all.

---

## 2. Finding the projectors without typetrees

IL2CPP builds ship **no script typetrees for user MonoBehaviours**. The serialized object's type has only the base-class fields; the derived fields are an opaque byte tail. Two things follow.

**(a) Read the header with the assert disabled.**
`o.read_typetree(check_read=False)` (`extraction/intel/extract_decals.py:318`) parses only the base `MonoBehaviour` fields - `m_GameObject`, `m_Enabled`, `m_Script`, `m_Name` - and does **not** raise when the reader stops before end-of-stream. With the default `check_read=True` every one of these objects throws and the walk yields nothing.

**(b) Resolve the class name through MonoScript.**
`m_Script` is a `PPtr<MonoScript>` = `(m_FileID: i32, m_PathID: i64)`. The class name lives on the `MonoScript` object, whose `m_ClassName` string **is** typetree-readable (MonoScript is a builtin type).

Resolution rule (`extraction/intel/extract_decals.py:321-327`):

```
fid, spid = m_Script.m_FileID, m_Script.m_PathID
if fid == 0:                       # MonoScript is in THIS file
    cls = local_monoscripts[spid].m_ClassName
else:                              # one hop through the file's externals table
    fname = externals[fid - 1].name        # NOTE: 1-based index
    cls   = monoscripts_of(fname)[spid].m_ClassName
keep if cls == "StaticDeferredDecal"
```

The **externals table is 1-based**: `m_FileID == 0` means "this file", `m_FileID == n` means `externals[n-1]`. Off-by-one here resolves every class name to the wrong script and the walk silently returns zero projectors.

The externals table is fetched off the first sub-file that exposes one (`extraction/intel/extract_decals.py:228-234`); MonoScript maps per external file are cached by filename (`extraction/intel/extract_decals.py:113-128`).

---

## 3. The MonoBehaviour payload - exact byte layout

The derived fields are read directly out of the object's raw serialized bytes. Little-endian, Unity's 4-byte serialization alignment, **no 8-byte alignment on the i64**.

Field *semantics* below are inferred from the reader, not confirmed against a parsed game asset **[unverified]**. The layout is internally consistent: `<4f` at 0 plus `<iq` at 16 is exactly the 28 bytes the length gate demands.

**Header size** (`extraction/intel/extract_decals.py:351`):

```
hsize = (12 + 4 + 12 + 4 + len(m_Name.encode("utf-8")) + 3) & ~3
         │    │    │    │    │
         │    │    │    │    └── m_Name UTF-8 bytes
         │    │    │    └─────── i32 string length prefix
         │    │    └──────────── PPtr<MonoScript>  : i32 fileID + i64 pathID = 12
         │    └───────────────── m_Enabled          : u8 + 3 pad = 4
         └────────────────────── PPtr<GameObject>   : i32 fileID + i64 pathID = 12
                                 ( + 3 ) & ~3  = round the total up to a 4-byte boundary
```

`pl = raw[hsize:]` (`extraction/intel/extract_decals.py:352`), minimum length **28 bytes** (`if len(pl) < 28:`, `extraction/intel/extract_decals.py:355`).

**Payload** (`extraction/intel/extract_decals.py:358-359`):

| offset | dtype | field |
|---|---|---|
| 0  | `<f4` | `x1` - atlas rect left, in **texture PIXELS** |
| 4  | `<f4` | `y1` - atlas rect bottom, in pixels, **origin bottom-left** |
| 8  | `<f4` | `x2` - atlas rect right |
| 12 | `<f4` | `y2` - atlas rect top |
| 16 | `<i4` | `PPtr<Material>.m_FileID` |
| 20 | `<i8` | `PPtr<Material>.m_PathID` |
| 28 | - | end of the fields this pipeline consumes |

Unpacked as `struct.unpack_from("<4f", pl, 0)` and `struct.unpack_from("<iq", pl, 16)`. The `<` prefix selects **standard size, no native alignment**, which is what puts the i64 at offset 20 rather than 24. Using native `@`/`=` alignment reads the pathID from offset 24 and yields garbage pointers on every decal.

**Validity gates** (`extraction/intel/extract_decals.py:364-366`), verbatim, where `ext` is the file's externals table:

```
bad_rect = not all(isfinite(v) and -64 <= v <= 16384 for v in (x1,y1,x2,y2))
           or x2 <= x1 or y2 <= y1
bad_ptr  = mfid <= 0 or not ext or mfid > len(ext)
```

`not ext` is a **falsiness** test, not an identity test, and that matters: UnityPy's `SerializedFile` initialises `externals` to an empty list, not `None` (`venv/Lib/site-packages/UnityPy/files/SerializedFile.py:244`). Rewriting the gate as `ext is None` lets an empty table through to `len(ext)` and indexes `ext[mfid - 1]` on nothing.

The rect range is deliberately loose on the low end. Authored cells routinely carry small **negative insets** (−2, −8 px are common) and can overrun the texture by a pixel; this is authored bleed, not corruption. Rejecting on `0 <= v` discards whole atlas families (`extraction/intel/extract_decals.py:360-363`). Clamping happens later, at UV time.

`mfid <= 0` means only **external** material pointers are accepted; a decal whose material lives in the same file as the projector is currently dropped as a bad pointer.

---

## 4. Material and texture resolution (one-hop external table)

Given `(material_file, material_pathID)`:

1. Load `material_file`, index its objects by `path_id`, confirm the object type is `Material`, read its typetree (`extraction/intel/extract_decals.py:146-151`).
2. Take that **material file's own** externals table (`extraction/intel/extract_decals.py:154-159`).
3. Walk `m_SavedProperties.m_TexEnvs`. Entries arrive either as `(key, value)` pairs or as `{"first":…, "second":…}` depending on reader version - handle both (`extraction/intel/extract_decals.py:161-162`).
4. Keep **only** `_MainTex` and `_BumpMap` (`extraction/intel/extract_decals.py:167-168`). Array/auxiliary slots reuse pathIDs and collide in the export naming scheme.
5. Resolve `m_Texture = (m_FileID, m_PathID)`:
   - `m_FileID == 0` → the Texture2D is in `material_file`.
   - `m_FileID == n > 0` → **one hop**: `externals[n-1].name` is the file holding the Texture2D; load it and index by pathID (`extraction/intel/extract_decals.py:173-185`). **Most decal materials resolve only this way.**
6. Decode and write the texture once as `tex/<m_Name>__<srcfile without ".assets">_<pathID>.png` (`extraction/intel/extract_decals.py:189-197`).
7. Read `(W, H)` from the written PNG for `_MainTex` (`extraction/intel/extract_decals.py:199-202`). These pixel dimensions are what the atlas rect is measured against; getting them from anywhere else (a header field, a guess) desynchronises every UV.

**Scoping hazard.** The per-material loader must bind the material's own asset environment to a **local name distinct from the level-loop variable**. Naming it `env` inside a nested function makes Python resolve it from the enclosing scope at call time, after the level loop has rebound `env` - so material lookups search the *level* file instead of the material's file and silently return `None`. The signature is a large "unresolved materials" count with no exception (`extraction/intel/extract_decals.py:135-140`).

---

## 5. The projector transform

Build the world TRS by walking Transform parents, in **raw Unity space** (left-handed, Y-up, metres).

Quaternion `(x,y,z,w)` → 3×3 (`extraction/intel/extract_decals.py:97-103`):

```
R = [[1-2(y²+z²),   2(xy - zw),   2(xz + yw)],
     [  2(xy + zw), 1-2(x²+z²),   2(yz - xw)],
     [  2(xz - yw),   2(yz + xw), 1-2(x²+y²)]]
```

Parent composition (`extraction/intel/extract_decals.py:273-301`):

```
pos   = parent_pos + R(parent_rot) @ (local_pos * parent_scale)     # component-wise scale first
rot   = parent_rot ⊗ local_rot                                      # Hamilton product, parent on the left
scale = parent_scale * local_scale                                  # component-wise
```

Hamilton product as written:

```
x = w1x2 + x1w2 + y1z2 − z1y2
y = w1y2 − x1z2 + y1w2 + z1x2
z = w1z2 + x1y2 − y1x2 + z1w2
w = w1w2 − x1x2 − y1y2 − z1z2
```

Root case (`m_Father.m_PathID == 0`, or a father not present in the file): local TRS **is** world TRS.

This composition tracks scale component-wise and therefore does not model the shear a rotated non-uniform parent scale would introduce. Decal projectors sit in shallow, axis-aligned hierarchies where this holds; a receiver-geometry extractor must **not** use this shortcut (see the "never TRS-decompose" rule).

Final matrix emitted per decal (`extraction/intel/extract_decals.py:391-403`):

```
M3 = R(rot) @ diag(scale)
m  = row-major 4×4:  [M3[0,:], pos.x,
                      M3[1,:], pos.y,
                      M3[2,:], pos.z,
                      0,0,0,   1]
```

This is **RAW Unity space**, exactly like every other instance matrix in the scene. The global handedness conjugation is owned by the assembler; applying it here applies it twice.

**Active state.** A projector is live when every GameObject on its Transform parent chain has `m_IsActive == true` (walk capped at 64 levels, `extraction/intel/extract_decals.py:303-311`) **and** the MonoBehaviour's `m_Enabled` is set (`extraction/intel/extract_decals.py:387`). Inactive projectors are still emitted, tagged `drop: true`.

**What actually consumes `drop`.** The bake does not: `project_decals` iterates *every* decal, `drop: true` included, and writes baked OBJs for them (`extraction/intel/decal_project.py:167`). The assembler is what discards them, wholesale, at load: `_live = [d for d in _dec if not d.get('drop')]` (`eft_pipeline/assemble_bevy.py:702`). So a `drop: true` projector costs bake time and disk, and only a consumer that skips the assembler's filter can ever use one.

---

## 6. The projector box: axes, atlas rect, UV

### Which axis is which

The box spans **local X and local Z as the image plane** and projects along **local Y**. This is *not* the XY-plane/along-Z that "decal projector" suggests, and it was measured rather than assumed (`extraction/intel/extract_decals.py:76-85`). The measurements themselves are **[unverified]**:

- Across all 1,269 projectors on the reference map, the atlas cell's aspect ratio matches `|col_X| / |col_Z|` about three times better than `|col_X| / |col_Y|` - mean absolute log-ratio error **0.43 vs 1.34**.
- The reference checkpoint's sprays sit in a box scaled `(3.53, 0.63, 1.13)` against a `469 × 149` px cell; cell aspect `3.15` versus `3.53 / 1.13 = 3.12`.

Column roles in the bake (`extraction/intel/decal_project.py:172-174`):

```
ax = M[:,0]   image across   -> U
ay = M[:,1]   projection axis (the depth/"thickness" axis)
az = M[:,2]   image along-V
```

Half extents and unit axes (`extraction/intel/decal_project.py:175`, `:178-179`; the degenerate/oversize reject at `:176-177` sits between them - Step 2 of §7):

```
hx, hy, hz = |ax|/2, |ay|/2, |az|/2                # :175
ux, uy, uz = ax/(2hx), ay/(2hy), az/(2hz)          # :178  unit vectors
reach      = ‖(hx, hy, hz)‖                        # :179  box half-diagonal, for broad-phase
```

**Failure signature if you build the image plane in XY:** every decal lies on its side - words run vertically down walls, road markings stand up out of the tarmac.

### Atlas rect → UV

`(x1,y1,x2,y2)` are **pixels**, and `y` is measured from the texture's **BOTTOM** edge (Unity's texture convention, not the image/raster top-left convention).

Clamp the authored bleed into the texture first (`extraction/intel/extract_decals.py:406-410`):

```
cx1 = max(0, x1);  cy1 = max(0, y1)
cx2 = min(W, x2);  cy2 = min(H, y2)
reject if cx2 <= cx1 or cy2 <= cy1
```

Then (`extraction/intel/extract_decals.py:411-417`):

```
su = (cx2 - cx1) / W        # U scale
sv = (cy2 - cy1) / H        # V scale
ou =  cx1 / W               # U offset
ov =  cy1 / H               # V offset   -- NO FLIP
```

`ov` passes through **unflipped**. Applying `ov = 1 - cy2/H` selects the mirrored row band: on the reference atlas the top rows render where the game paints the bottom rows (`extraction/intel/extract_decals.py:413-416`; the confirming photograph of the real location is a field observation, **[unverified]**).

### U orientation under conjugation

The map's global orientation fix is a similarity conjugation, but **the bake and the assembler get their matrix from different places**:

- The bake **hardcodes** `G = diag(-1, 1, 1)` (`extraction/intel/decal_project.py:21`) and applies `M' = G · M · G`, `T' = G · T`.
- The assembler hardcodes nothing: it reads `coordinates.global_matrix` from the map config (`eft_pipeline/assemble_bevy.py:902` → `eft_pipeline/tarkmap_core/config.py:121-126`, defaulting to `diag(-1,1,1,1)` when the key is absent) and conjugates as `M' = G4 · M · G4⁻¹` (`eft_pipeline/tarkmap_core/instmath.py:15`, `:21-22`).

The two agree only because the **default** `G` is its own inverse. A config that sets `coordinates.global_matrix` to anything else silently desynchronises the hardcoded bake from the assembler - no error, decals simply land in a different frame from the geometry.

For the default `G` (`G = G⁻¹`, `det G = -1`), because `G e0 = -e0` while `G e1 = e1` and `G e2 = e2`:

```
M'[:,0] = -(G · M[:,0])       <-- X column reverses direction
M'[:,1] =   G · M[:,1]
M'[:,2] =   G · M[:,2]
```

**The flip is in X only.** So the bake's U must run *against* the conjugated box X, and V passes straight through (`extraction/intel/decal_project.py:251-255`):

```
u = 0.5 - (p · ux) / (2·hx)
v = 0.5 + (p · uz) / (2·hz)
```

with `p` the vertex position **relative to the box centre**, in the conjugated frame. Both land in `[0,1]` because the polygon has already been clipped to the box.

**Failure signature if U is read directly (`0.5 + …`):** every glyph is mirrored - the text reads backwards ("ЯATNU" for "UNTAR"). Geometry, placement and depth are all correct; only the lettering is reversed. That is diagnostic: a mirrored *image* with correct *placement* is a U-axis sign error, not a matrix error.

---

## 7. The projection bake

Emitting one flat quad at the box centre is wrong whenever the box spans surfaces at different depths: the nearer surface occludes the quad and the far half of the artwork vanishes (`extraction/intel/decal_project.py:5-7`). These decals are static, on static geometry, so the projection is evaluated **once**, offline, producing the same pixels the runtime deferred pass would.

Entry point: `project_decals(dataset, decals)` (`extraction/intel/decal_project.py:91`).

### Step 1 - receiver broad-phase

For every non-dropped scene instance: read the mesh's **local AABB** from the OBJ's `v` lines only (cheap; most meshes are never fully parsed - `extraction/intel/decal_project.py:106-124`), conjugate the instance (`extraction/intel/decal_project.py:143-144`):

```
M3 = G · M · G          T = G · t
```

Transform the 8 local corners, take the world AABB centre and its corner radius, store as a bounding sphere (`extraction/intel/decal_project.py:145-151`).

Per decal, candidate receivers are `‖center_i − C‖ ≤ radius_i + reach` (`extraction/intel/decal_project.py:181`).

**Everything in the bake happens in the conjugated frame** - both the receiver and the projector. Clipping receiver geometry against a *raw* projector matrix places the geometry in a mirrored frame; the decal lands on the correct object but leaning the opposite way to the surface it paints. That is the classic signature of a handedness flip applied to positions but not orientations (`extraction/intel/decal_project.py:138-142`).

### Step 2 - box rejection

Skip the decal entirely if `min(hx,hy,hz) < 1e-4` or `max(hx,hy,hz) > MAX_BOX_M = 200.0` m (`extraction/intel/decal_project.py:176-177`). A handful of authored projectors are degenerate or kilometre-scale. This `continue` increments **no counter** - see the accounting note in §11.

### Step 3 - receiver gate (tight) and candidate slab (loose)

Per receiver, with `rel = W − C` and box coordinates `bx = rel·ux`, `by = rel·uy`, `bz = rel·uz`:

```
touches = |bx| <= hx  AND  |by| <= hy  AND  |bz| <= hz          # tight, authored box
```

If nothing touches, the instance is not a receiver - skip it (`extraction/intel/decal_project.py:202-204`). This gate is what makes the generous depth bound below safe: the depth reach then only ever extends **along a surface the projector genuinely lands on**, never onto new geometry further down the ray.

```
inside  = |bx| <= 1.2·hx  AND  |by| <= DEPTH_REACH·hy  AND  |bz| <= 1.2·hz
```

Candidate triangles are those with **any** vertex inside (`extraction/intel/decal_project.py:205-208`). Clipping fixes the rest.

### Step 4 - per-triangle facing cull

```
n = normalize(cross(v1 - v0, v2 - v0))      # skip if ‖cross‖ < 1e-12
keep if  n · uy >= FACING_MIN (= 0.50, i.e. 60°)
```

(`extraction/intel/decal_project.py:211-228`)

Two independent things are being enforced:

- **Sign** - the decal paints the faces the projector lands *on*. Conjugation flips the effective sense of `uy`; getting the sign wrong moves every decal to the far face of its surface. Signature: the artwork disappears from the side you are standing on and is visible only from behind the plate (`extraction/intel/decal_project.py:222-226`).
- **Magnitude** - the cutoff stops a decal **smearing** down surfaces it only grazes. At `0.2` (78°) the artwork stretched into long streaks along angled plates and their support legs **[unverified]**. Without any cull, a box sitting on dense terrain clips thousands of invisible ground triangles (1.42 M for one level's decals **[unverified]**) and back faces get painted through (`extraction/intel/decal_project.py:29-32`, `:220-224`).

### Step 5 - clip to the box (Sutherland–Hodgman)

Convert the triangle to centre-relative coordinates `p = v − C`, then clip against six half-spaces (`extraction/intel/decal_project.py:237-243`), each pass keeping `dot(p, normal) <= dist` (`extraction/intel/decal_project.py:72-88`):

| plane pair | distance |
|---|---|
| `±ux` | `hx` |
| `±uz` | `hz` |
| `±uy` | `hy · DEPTH_REACH` (= 2.5·hy) |

Abort the triangle as soon as the polygon drops below 3 vertices.

**The image rectangle (X, Z) is clipped exactly** - that is what frames the artwork. **The depth axis (Y) is deliberately generous.** Authored boxes are routinely thinner than the surface they paint: the reference checkpoint plate sits 24° off perpendicular, so its face sweeps **1.44 m** through a **0.63 m** box **[unverified]**. A hard slab clip cuts the lettering **mid-glyph** - the top or bottom of a word is sliced off along a straight line that does not follow any surface edge (`extraction/intel/decal_project.py:38-42`, `:230-236`).

`DEPTH_REACH = 2.5` covers that. The source records that `4.0` was tried and let decals run down adjacent walls **[unverified]**.

### Step 6 - emit vertices, UVs, winding

Per clipped-polygon vertex (`extraction/intel/decal_project.py:246-255`):

```
world_pos = C + p + n · SURFACE_OFFSET_M          # SURFACE_OFFSET_M = 0.012 m
uv        = (0.5 - (p·ux)/(2hx),  0.5 + (p·uz)/(2hz))
```

The 12 mm offset is along the **receiving surface's own normal**, not the projection axis - it must win the depth test against the exact surface it is painted on, and 12 mm is small enough to stay invisible at grazing angles (`extraction/intel/decal_project.py:23-24`).

Triangulate as a fan and **derive each triangle's winding from the surface normal** (`extraction/intel/decal_project.py:256-265`):

```
for k in 1 .. len(poly)-2:
    tn = cross(poly[k] - poly[0], poly[k+1] - poly[0])
    emit (0, k, k+1)  if  tn · n >= 0
    else (0, k+1, k)
```

Copying the source triangle's order is **not** sufficient - the clip can reverse a polygon. Hard-coding a flip leaves the receiving surfaces blank (all faces culled as back faces). Deriving from `n` is correct in either handedness (`extraction/intel/decal_project.py:256-259`).

### Step 7 - per-decal caps

- **Empty** (`len(faces) == 0`) → the decal is dropped from the output entirely (`extraction/intel/decal_project.py:267-269`).
- **Too dense** (`len(faces) > MAX_TRIS_PER_DECAL = 4000`) → reported and skipped rather than silently doubling the pack. A projector enclosing terrain or foliage contributes an unbounded triangle count (`extraction/intel/decal_project.py:33-36`, `:270-272`).

---

## 8. The emitted mesh convention

### The pre-bake shared quad (`decal_quad__gen.obj`)

Written by `write_quad` (`extraction/intel/extract_decals.py:74-94`), called **unconditionally** at `extraction/intel/extract_decals.py:109`. It is the bake's fallback as much as its opt-out: `project_decals` returns the flat-quad instances **unchanged** on two non-`EFT_DECAL_FLAT` paths - no `scene.json` (`extraction/intel/decal_project.py:96-98`, "no scene.json - cannot project; keeping flat quads") and no usable instance geometry (`extraction/intel/decal_project.py:153-155`). A pipeline that never sets `EFT_DECAL_FLAT` can still ship flat quads.

- Four vertices in the local **XZ plane** at `y = +0.01` (z-fight guard), corners `(±0.5, ±0.5)`.
- **X is negated** when written - the extractor OBJ convention (UnityPy `mesh.export()`: negate X, reverse winding), matching every other mesh in the dataset.
- `vt` = the unit square `(0,0) (1,0) (1,1) (0,1)`.
- **Both windings** are emitted (4 triangles referencing the same 4 vt) so no cull mode can hide a decal.

### The baked mesh (`decal_bake_%05d__gen.obj`)

Written by `extraction/intel/decal_project.py:273-285`.

- Vertices are written **verbatim, with NO X negation** (`extraction/intel/decal_project.py:277-281`). They are already in the assembler's **final** frame, because the receiver was conjugated before clipping. Negating X here mirrors the decal straight back off the surface it was just fitted to.
- The instance matrix becomes **identity** (`extraction/intel/decal_project.py:291`). This is safe under the conjugation rule because `G · I · G⁻¹ = I` - conjugating identity is identity, so the assembler's global fix is a no-op on baked decals and the world-space vertices pass through untouched.
- `vt` values are real `[0,1]` box coordinates; face lines are `f v/vt v/vt v/vt` (`extraction/intel/decal_project.py:282-285`).
- The submesh's face count `n` is updated to `len(faces_out)` (`extraction/intel/decal_project.py:292`).

**Invariant:** *(final-space vertices) × (identity instance matrix)* must hold together. If an importer applies any global flip to the baked vertices, or the extractor negates X, the decal mirrors off its surface.

---

## 9. Instance record schema and downstream UV composition

One record per surviving projector (`extraction/intel/extract_decals.py:420-437`), written to `decals.json` as `{"instances": [...]}`, in the same schema as the main scene instance list so every downstream stage treats decals as ordinary geometry:

```json
{
  "mesh": "decal_bake_00042__gen.obj",
  "m":    [16 floats, row-major 4×4],
  "kind": "mesh",
  "root": "DECALS_PROJECTED",
  "lv":   <level index>,
  "drop": <true if the projector or any ancestor is inactive>,
  "subs": [{
    "tex":  "<atlas png basename>",
    "nrm":  "<normal map basename or null>",
    "col":  null,
    "sh":   "p0/DeferredDecal",
    "uv":   [su, sv, ou, ov],
    "cut":  null,
    "n":    <triangle count>,
    "role": "decal"
  }]
}
```

`sh` is a synthetic shader tag, not read from the material. `role: "decal"` is the load-bearing field.

**UV composition downstream** (`eft_pipeline/assemble_bevy.py:1002` then `:1006`):

```
uv  = uv_mesh * (su, sv) + (ou, ov)        # atlas cell selection (Unity _ST semantics)
uv.y = 1.0 - uv.y                          # bottom-left origin -> top-left raster origin
```

Composed end to end, a baked vertex with local `v ∈ [0,1]` samples texture row (from the top):

```
row_from_top = H - (cy1 + v · (cy2 − cy1))
```

so `v = 0` (the `−uz` side of the box) reads the rect's `cy1` edge and `v = 1` (the `+uz` side) reads `cy2`. The V flip belongs **once**, at texture-fetch time, and only there.

**Consumer contract.** `role: "decal"` must map to alpha **BLEND** (`eft_pipeline/assemble_bevy.py:310-311`) and to a small coplanar depth separation. The reference renderer pushes clip-space `z += 1.0e-3 · w` on the decal colour pass only (`viewer/assets/shaders/gpu_draw.wgsl:992`, `:1000-1001`) rather than using a rasterizer depth bias. The stated reason - that a depth-bias `constant` on a `Depth32Float` target scales as `constant · 2^(exponent(z)−23)` and therefore drifts with camera distance - is the shader comment's own rationale (`viewer/assets/shaders/gpu_draw.wgsl:979-982`) and is **[unverified]** here. In a Blender-style importer the equivalent is: alpha-blended material, backface culling off or winding-correct, and a small polygon offset on top of the 12 mm geometric offset already baked in.

**Normal-map albedo guard.** Some "decal" materials are bevel *normal* maps. Painting them as albedo turns every edge blue. `albedo_is_normalmap` (`eft_pipeline/assemble_bevy.py:196-207`) reduces the albedo to a single pixel - `im.convert('RGB').resize((8,8)).resize((1,1)).getpixel((0,0))` - and classifies on that pixel (`:204-205`):

```
b > 200 and abs(r-128) < 45 and abs(g-128) < 45 and b > r + 55 and b > g + 55
```

This is a **channel-relation** test, not a distance to `[128,128,255]`: it demands a strongly blue-dominant pixel with mid-grey R and G, each channel bounded independently. It runs over every `role: "decal"` submesh with a `tex` at `eft_pipeline/assemble_bevy.py:778-782`. When dropping a submesh, **mark it** (`sb['drop_nm_decal'] = True`), never remove it - a removed sub takes its `n` out of the running face cursor and shifts every later submesh's face range earlier by `n`.

---

## 10. Invariants and failure signatures

The visual signatures in this table are historical field observations recorded in the source comments; they describe what was seen when each invariant was broken and cannot be re-derived from the code **[unverified]**. The invariants themselves are code.

| # | Invariant | Break it → visible signature |
|---|---|---|
| 1 | Image plane is local **XZ**, projection axis is local **Y** | Every decal lies on its side; words run vertically |
| 2 | The extractor emits **raw** Unity matrices; conjugation is the assembler's job | Double flip - decals land on the mirrored side of the map |
| 3 | The bake conjugates **both** projector and receiver (`M' = G·M·G`, `T' = G·T`) | Decal lands on the right object but leans the opposite way to the surface |
| 4 | `u = 0.5 − (p·ux)/(2hx)` - U runs **against** the conjugated box X | **Mirrored text** ("ЯATNU" for "UNTAR"); placement otherwise perfect |
| 5 | V passes through **unflipped** at rect→UV time; the single flip happens at fetch | Wrong atlas **row band** selected - a different word/stain of the same atlas appears |
| 6 | `n · uy >= FACING_MIN` with the correct **sign** | Decal on the **back face** - invisible from the side you stand on, visible from behind |
| 7 | `FACING_MIN >= 0.5` (60°) | **Smearing**: artwork stretched into long streaks down grazing/angled surfaces and their legs |
| 8 | Facing cull present at all | Thousands of invisible terrain triangles baked; back faces painted through |
| 9 | Depth clip uses `DEPTH_REACH · hy`, not `hy` | **Letters cut mid-glyph** along a straight line that follows no surface edge |
| 10 | `DEPTH_REACH <= ~2.5` | Decal runs off its plate and down the adjacent wall |
| 11 | Tight `touches` gate before the loose `inside` slab | The generous depth reach finds *different* geometry further along the ray |
| 12 | Winding derived from `n · tn`, per fan triangle | Receiving plates render **blank** (all back faces, culled) |
| 13 | Baked verts written **without** X negation, instance matrix identity | Decal mirrors straight off the surface it was just fitted to |
| 14 | Bake at all (vs. one flat quad at box centre) | Far half of a word **vanishes** behind the nearer of two staggered plates |
| 15 | Payload `<iq` at offsets 16/20 (no native alignment) | Every material pathID is garbage; 100% unresolved materials |
| 16 | Externals table indexed `[fid - 1]` | Class names and materials resolve to the wrong file; zero projectors found |
| 17 | `read_typetree(check_read=False)` | Every MonoBehaviour throws; zero projectors found |
| 18 | Rect accepts small negatives (`-64 <= v`) | Whole atlas families rejected as "bad payload" |
| 19 | Material env bound to a **function-local** name | Silent mass "unresolved materials"; no exception, no traceback |
| 20 | Submeshes are **marked**, not removed, when dropped | Later submeshes read face ranges shifted by `n`; the last submesh renders a see-through hole |
| 21 | The bake's `G` and the assembler's `coordinates.global_matrix` are the same transform | Config override desynchronises baked decals from geometry, silently |

---

## 11. Tunables

All in `extraction/intel/decal_project.py:20-42`:

| constant | value | meaning |
|---|---|---|
| `G3` | `diag(-1, 1, 1)` | map handedness flip, **hardcoded** at `extraction/intel/decal_project.py:21`; the assembler instead reads `coordinates.global_matrix` and conjugates with `G4 @ M @ G4⁻¹` - see §6 |
| `SURFACE_OFFSET_M` | `0.012` m | nudge along the **receiving surface normal** |
| `MAX_BOX_M` | `200.0` m | reject degenerate/kilometre-scale authored boxes |
| `FACING_MIN` | `0.50` (60°) | facing cutoff; also the anti-smear guard |
| `MAX_TRIS_PER_DECAL` | `4000` | runaway guard; over-cap decals are reported and skipped |
| `DEPTH_REACH` | `2.5` | multiplier on `hy` for the depth clip and the loose candidate slab |
| loose slab X/Z margin | `1.2` | `extraction/intel/decal_project.py:205` |
| degenerate-triangle epsilon | `1e-12` | `extraction/intel/decal_project.py:215` |
| degenerate-box epsilon | `1e-4` | `extraction/intel/decal_project.py:176` |

Diagnostics: `EFT_DECAL_TRACE=<GameObject name substring>` follows a single projector through every gate (`extraction/intel/extract_decals.py:332-342`); `EFT_DECAL_DEBUG=1` prints rejection reasons; `EFT_DECAL_FLAT=1` skips the bake and keeps flat quads for comparison (`extraction/intel/extract_decals.py:447`).

Reported counters: emitted / seen / bad payloads / unresolved materials / inactive-kept (`extraction/intel/extract_decals.py:458-460`) and baked / triangles / empty / too-dense (`extraction/intel/decal_project.py:298-299`). A healthy run has `bad payloads` and `unresolved materials` both small relative to `seen`; a large `unresolved materials` count with zero exceptions is invariant #19.

**The bake counters do not balance.** A decal rejected by the degenerate/oversize-box test (`extraction/intel/decal_project.py:176-177`) leaves the loop without incrementing `empty` or `heavy`, so `baked + empty + heavy` is less than the number of decals fed in, by the box-reject count. Do not use the counters to prove nothing was silently lost.

---

## 12. Old patterns

- **The header docstring of `extraction/intel/extract_decals.py:13-14` states the image spans local XY and projects along local Z.** That predates the aspect-ratio measurement. The correct convention - image plane **XZ**, projection axis **Y** - is documented with its evidence at `extraction/intel/extract_decals.py:76-85` and implemented everywhere (`extraction/intel/decal_project.py:172-174`). Trust the code, not that sentence.
- **A dead second `G3`.** `extraction/intel/extract_decals.py:71` defines `G3 = np.diag([-1.0, 1.0, 1.0])` and nothing in that file reads it - the extractor emits raw matrices (§5). Only `extraction/intel/decal_project.py:21` has a live copy. Do not take the extractor's constant as evidence that the extractor conjugates.
- **Flat quads.** Before the projection bake, each decal was one quad at the box centre with the shared `decal_quad__gen.obj` mesh and the projector's raw matrix. Still reachable behind `EFT_DECAL_FLAT=1` for A/B comparison, and still the automatic fallback when the bake has nothing to project against (§8). It cannot paint a decal spanning surfaces at different depths.
- **Strict rect validation (`0 <= v <= texture size`).** Discarded authored bleed and with it whole atlas families. Replaced by the loose accept + clamp at UV time.
- **Hard-coded winding flip on baked triangles.** Left every receiving surface blank. Replaced by per-triangle derivation from the surface normal.