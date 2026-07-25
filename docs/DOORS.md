# Doors — what the game data gives us, and how the viewer uses it

Everything here is **derived from the game's own files**. No door behaviour is authored by hand,
name-matched, or eyeballed. This document records *where each fact comes from* so the next person
can re-verify it instead of trusting a constant.

Sources, in order of authority:

1. **Unity scene objects** (`levelN` files, read with UnityPy) — transforms, hierarchy, renderers.
   Fully readable. This is where almost everything below comes from.
2. **MonoBehaviour payloads** (the same files) — script fields. Readable as BYTES only, because
   EFT is IL2CPP with an encrypted `global-metadata.dat`, so UnityPy cannot generate script
   typetrees. Layouts are recovered empirically and validated (see below and `docs/IL2CPP.md`).
3. **il2cpp metadata** — would give the *declared* field layout and remove the guesswork. Ships
   encrypted; see `docs/IL2CPP.md` and `tools/il2cpp_explore.py`.

---

## 1. Which objects are doors

Typed, not name-matched: a door is a GameObject carrying a MonoBehaviour whose MonoScript class is
one of `Door`, `KeycardDoor`, `SlidingDoor`, `Trunk`, `ExfiltrationDoor`, `DoorSwitch`
(`DOOR_CLASSES` in `extraction/intel/extract_gamedata.py`). MonoScript is an engine type, so its
class name is always readable even though the script *body* is not.

Only `Door` / `KeycardDoor` / `DoorSwitch` are treated as **swing** doors. Trunks, sliding doors and
exfil doors move differently and are marked `swing: false` so the viewer never rotates them.

## 2. The hinge — position and axis

* **Pivot** = the door component's Transform world origin, bridged into viewer space with the same
  `G3 = diag(-1,1,1)` X-flip the geometry pipeline uses. Shipped as `doors[].pos`.
* **Axis** = the door's **local +Z**. In the viewer this is the instance affine's column 2.

Verified against the game, not assumed: on streets, `Inside_Door_Wood_23_L_260-180_door_L` is
authored **open**, and its Unity `m_LocalRotation` is exactly **+94.00° about local +Z** — matching
the open angle in its payload (94.0) to two decimals. Every **shut** door in the same scene sits at
local rotation **identity**, which also proves the pack's baked matrix is the CLOSED pose.

## 3. The open angle

Recovered from the `Door` payload at `IdEnd + 56` as a signed float, validated on 97 open
Interchange doors (within 0.15°) and re-confirmed by the local-rotation check above. Shipped as
`doors[].open_angle` (degrees).

Two payload layouts exist and both are parsed (`dec_door`):

* **Classic** (every pre-Icebreaker map): `[20B][dword 0][0x0F layer][KeyId str][12B][Id str][tail]`.
* **Trigger-block** (Icebreaker+): the same, but with `N` interaction-trigger name strings
  (`Open_01_<hash>`, `Quest_Complete_<hash>`) inserted *before* the KeyId block. Reading these with
  the classic fixed offsets returned a trigger name as the KeyId and lost state + angle — that was
  the "15 dead doors on the Icebreaker deck" bug. The `0x0F` layer dword anchors the walk; a
  KeyId is always empty or a 24-hex template id, which is how the two layouts are told apart.

## 4. Swing DIRECTION (the sign)

The viewer world is an **X-mirror** of Unity's (`G = diag(-1,1,1)`, `det = -1`). Conjugation maps a
rotation to `R(G·a, −θ)` — a mirror reverses rotational sense. Concretely, for `G = diag(-1,1,1)`
and a hinge along Z:

```
G · R(z, +90°) · G⁻¹  =  R(z, −90°)
```

So the authored Unity angle **must be negated** in viewer space, or every door in the game opens
the wrong way. This is `open_rad = -d.open_angle.to_radians()` in `render/gpu_driven.rs`.

## 5. Which parts swing (the glass problem)

A door is **not one mesh**. The Door component sits on the swinging **leaf** GameObject, and the
parts that swing are the renderers in **that leaf's transform subtree** — while the frame is a
*sibling* that must stay put. Verified on streets `Inside_Door_Wood_23_L_260-180`:

```
parent: Inside_Door_Wood_23_L_260-180_glass (1)
  ├── Inside_Door_Wood_23_L_260-180_LOD0            <- FRAME (sibling, static)
  ├── Inside_Door_Wood_23_L_260-180_glass_LOD0      <- frame glass (sibling, static)
  └── Inside_Door_Wood_23_L_260-180_door_L          <- the Door component lives here
        ├── ..._door_L_LOD0                          <- panel      } these
        ├── ..._door_L_glass_LOD0                    <- ITS glass   } swing
        └── ..._door_L_wood_LOD0                     <- inlay      } together
```

The viewer previously matched **one** instance — the nearest to the pivot — so a door's glass
stayed behind when the panel swung. The extractor now ships `doors[].parts` = `[[mesh name,
[x,y,z]], …]` for every drawn renderer in that subtree, and the viewer animates all of them about
the shared hinge. A proximity radius would be wrong: the static frame is only ~1.4 m away.

Parts are filtered to renderers the game actually **draws** (`m_Enabled`, and `m_CastShadows != 3`
i.e. not ShadowsOnly) — the same Unity-visibility rule the geometry cull uses. This drops shadow
proxies and keeps generic 3ds-max names (`Box001`, the ballistic panels) from false-matching an
unrelated instance a metre away.

## 6. Initial state

`EDoorState` flags from the payload at `IdEnd + 92`: `1 locked · 2 shut · 4 open · 8 interacting ·
16 breach`. Validated on 299 lighthouse doors (the column reads only {1,2,4,16}; keyed doors are
always 1).

A door authored **open** ships its OPEN pose baked into the instance matrix. The viewer therefore
derives the closed pose by rotating it *back* by the open angle and animates from there — treating
the shipped matrix as "closed" rendered such a door at **double** its open angle.

## 7. Keys

`KeyId` is a 24-hex item template id read from the payload (empty when the door needs no key).
Resolved to a display name + icon through the tarkov.dev static item dump at build time.

## 8. What drives a door (switches)

Newer maps serialize interaction-trigger names on both sides of the relationship: a `Door` carries
`Open_01_<hash>` and the `Switch` that drives it carries the **same digit hash** in its own trigger
string. The build merge joins them into `switch -> door` edges with zero name matching, which is
how the LEVEL CONTROLS panel can say what an interactable opens.

---

## How to audit this yourself

`tools/audit_doors.py <map>` re-derives the ground truth straight from the Unity scenes and reports
disagreements with what the pack ships — per door: payload angle vs the authored local rotation,
axis, state, part counts, and whether each part resolves to a pack instance. Run it after any
change to `dec_door` or the door pipeline.

## Known gaps

* **Sliding doors / trunks** are extracted and marked non-swing, but the viewer does not animate
  them yet — they need translation (and for trunks, a lid hinge) rather than a Z rotation.
* **Multi-leaf doors** (`_Door_L` + `_Door_R`) are independent Door components and animate
  independently, which matches the game.
* The payload offsets in §3/§6 remain empirically recovered. Readable il2cpp metadata would replace
  them with declared field offsets — see `docs/IL2CPP.md`.
