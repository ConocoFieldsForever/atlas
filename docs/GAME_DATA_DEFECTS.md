# Defects found in EFT's own shipped data

Found by cross-checking the extracted scene graph, loot tables, collider world and baked nav grid
against each other. Everything here is a statement about the GAME's data, not about our extraction —
where the two could not be separated, that is said explicitly.

Confidence is about the DATA. None of it has been confirmed by observing the running game, and that
distinction is kept throughout.

---

## 1. Interchange ships four INVISIBLE water volumes  ⭐ the real one

**Confidence: high on the data.** Four independent lines of evidence agree.

Four `Shoreline_Lake_Water_02_LOD0` objects on level 63 are physically water but visually absent.

| evidence | finding |
|---|---|
| MeshRenderer | `m_Materials` is an **empty array** — no material, so nothing is drawn |
| how anomalous | level 63 has 48,663 renderers WITH materials and exactly **4 without**: these |
| across the map | of 1,624 material-less renderers on all 14 levels, 1,620 are `AreaLight`/`AreaLightGI`/`*_lanterns_*_Area` placeholders (invisible by design). These 4 are the only non-placeholders |
| sibling collider | each has `Shoreline_Lake_Water_02_BALLISTIC_water`: a MeshCollider on Unity **layer 4 `Water`** with `EFT.Ballistics.BallisticCollider` **surface type 25** (wood is 28) |
| material table | the game ships **13 water materials, one per map** — `Lighthouse_Water`, `City_Water`, `Reserve_Water_bunker`, `Laboratory_Water_FX`, `Wastewater`, `Sandbox_Water4Advanced`… and **none for Interchange** |

So the ballistics and surface systems treat these as water while the renderer draws nothing. Expected
symptom: bullet splash effects, and water-typed surface behaviour, at spots where the player sees dry
ground.

**They are reachable.** All four sit within 4 m of walkable nav floor, two of them 12 m and 22 m from
a player spawn point:

| world position | floor within 4 m | nearest spawn |
|---|---|---|
| `156.4, 18.6, -380.5` | yes (Δy 2.7 m) | 22 m |
| `392.2, 16.6, -166.1` | yes (Δy 1.3 m) | 92 m |
| `-97.5, 21.0, -384.3` | yes (Δy 0.6 m) | **12 m** |
| `-471.7, 18.7, -95.2` | yes (Δy 1.0 m) | 130 m |

**Likely cause**: `Shoreline_Lake_Water_02` is a Shoreline asset. No renderer of that family appears
on Shoreline's own levels at all, so these look like leftovers placed on Interchange whose material
assignment never survived — the ballistic/collider half of the prefab shipped, the visual half did
not.

---

## 2. Minor / single instances

Each is one object; listed for completeness, none is worth a report on its own.

- **Factory Rework**: 1 lootable container lies OUTSIDE the level-border polygon (36 verts).
- **Interchange**: 1 container (`scontainer_Blue_Barrel_Base_Cap`, y≈19.7) has no nav floor within
  3 m. Ambiguous — could equally be a gap in our bake.
- **Inactive but shipped**: 1 container on Interchange, 2 on Woods.
- **73 Interchange containers** carry templates absent from every catalog we pull
  (`5ad74cf586f774391278f6f0` ×72, `67614e3a6a90e4f10b0b140d` ×1). Almost certainly a gap in OUR
  catalog rather than a game defect — noted so it is not mistaken for one.

---

## 3. Checked and found CLEAN

Worth recording, because "we looked and it was fine" is a result:

- **Spawn points**: 0 of 278 (Interchange), 0 of 368 (Woods), 0 of 160 (Factory) lack a nav floor —
  no spawns inside geometry.
- **Level border**: 0 containers and 0 spawns outside it on Interchange (993-vertex polygon) and
  Woods (1,429).
- **Loose-loot item templates**: every referenced template resolves to a real item. No dangling
  references to removed content.
- **Loot-group spawn counts**: 0 of 19 groups declare a `max` larger than their member count. The
  data is internally consistent everywhere we can check it.

---

## 3b. Duplication-class integrity: scanned, nothing found

**Scope, stated up front.** EFT's real item-duplication surface is server-side — inventory
transactions, raid-exit reconciliation, netcode — and **none of it exists in the client files**. What
is checkable here is whether objects the game identifies by a string id are uniquely identified,
because an id collision is how one physical thing ends up tracked as two. That is the class of defect
this section covers, and it is narrower than "are there dupe bugs".

| check | result |
|---|---|
| Container instance ids colliding within one level | **none** — 907 `LootableContainer` components on interchange, 907 shipped, zero lost to dedupe |
| Loot-point GUIDs colliding within one level | **none** |
| Switch ids repeated | none |
| Loot-group `max` exceeding member count | 0 of 19 |
| Quest objectives with the same kind+name at >1 position | present, and almost certainly intentional — see below |

The container count is the strongest of these. Our extractor dedupes containers **by id**, so a
collision would have silently dropped one — the per-level game count matching the shipped count
exactly means no two containers in a level share an id.

**Quest objectives at multiple positions** — Woods has `em_quest4_1` at 3 places,
`ny25_quest_6_woods_houseinvillage` at 4; Factory has `nf2024_4_zone_kill1` at 4. This reads as
ordinary design: an objective that accepts any one of several locations. Whether credit is granted
per-objective or per-location is **server-side logic that is not in these files**, so it cannot be
determined here, and it is recorded as an observation rather than a finding.

Stationary weapons "sharing" a `weapon_id` is a template id (two mounts of the same weapon type),
not an instance collision.

---

## 4. NOT a defect — investigated and dismissed

**Containers stacked at identical coordinates** (27 extra on Interchange, 24 Woods, 6 Factory).
Initially looked like duplicated loot. They are `card_file_box_01..04` — always four, always type
`Drawer`, always distinct container ids, always the same origin. It is a **filing cabinet with four
drawers**, each drawer its own lootable; the visual offset lives in the mesh, not the transform. The
same pattern appears on all three maps. Intentional.

It does matter for US though, and is not currently handled: four co-located containers produce four
overlapping map markers, and the loot planner treats them as four separate stops at one point —
inflating both the marker count and the route's stop budget. Worth clustering.
