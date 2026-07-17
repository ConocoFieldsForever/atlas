# UI Roadmap — toolbar, camera modes, task revamp

Living TODO for the viewer UI overhaul. Check items off as they land; keep the "Status"
column honest. Ordered roughly by dependency + payoff.

## 0. Structure — vertical icon toolbar (foundation for everything else)

A **thin, one-button-wide vertical toolbar** pinned to the LEFT edge of the right panel
(between the 3D view and the existing `SidePanel::right`). Each icon selects WHICH settings
group the right panel shows — the panel becomes a switchable container instead of one long
scroll. State: a `RightPanelTab` enum resource (`Layers | Camera | Tasks | …`), default
`Layers` (current behavior). Tarkov gear-screen styling (square, charcoal, beige active tab).

- [x] `RightPanelTab` resource + toolbar widget (egui vertical strip, vector icon buttons).
- [x] Route the existing MAP LAYERS content behind the eye tab (tab-gated early return).
- [x] Active-tab highlight; tooltips on hover.
- Icons: vector-drawn in egui (no image assets — keep the redistribution story clean), or a
  tiny embedded glyph set. Camera / eye / clipboard-check shapes.

## 1. Camera tab (icon: camera)

### 1a. Camera settings (right panel, under the Camera tab)
- [x] **FOV** slider (wired to Projection via CameraSettings + apply_camera_fov).
- [x] **Exposure** slider (GfxSettings.grade_exposure). TODO: fold in bloom/fog/sky-refl from
      "Graphics (experimental)" later.
- [ ] Near/far, and a "reset framing" button (re-run content_anchor).

### 1b. Fly-cam scroll-wheel speed  [DONE]
- [ ] Scroll wheel adjusts fly-cam `speed` (multiplicative, clamped). Up = faster, down =
      slower. Persist across the session. HUD or tab shows the current speed.

### 1c. Walking camera mode  [DONE]
- [x] Walk mode (CameraSettings.walk_mode / EFT_WALK=1): eye-height 1.7 m, yaw-only WASD on
      the ground plane, jump (Space) + gravity, sprint (shift). walk_ground.rs: a prebuilt
      2.5D XZ-bucketed WORLD-triangle grid (near-horizontal faces only), ground_height =
      highest surface below feet+STEP_UP -> correct floor in multi-story buildings. Built
      lazily on first activation (~1.2s, 24.5M tris on lighthouse) via ComputeTaskPool.
- [x] Scroll in walk mode scales walk_speed (clamp 1.5..12); jump_velocity = sqrt(2*G*K*speed)
      so apex height scales linearly with speed -> one scroll juices move + hop together.
- [x] Sprint (shift) x1.8 in walk mode.

## 2. Visibility tab (icon: eye)

- [ ] Move the MAP LAYERS visibility toggles (loot classes, spawns, extracts, hazards,
      minefields, sniper zones, quests, min-value, hide-inactive, search) behind the `eye` tab.
      This is largely re-homing the current panel content; the `Layers` tab in section 0 and
      this `eye` tab may be the SAME thing — decide during 0 (probably: eye IS the layers tab).

## 3. Tasks tab (icon: quest/clipboard) — AGENT REVAMP (big)

Spawn a dedicated agent to TOTALLY revamp the task/quest UI. Requirements:
- [ ] **Track tasks**: check on/off, per-task active state (exists today via `QuestTracker`;
      revamp the presentation).
- [ ] **Visualize required items**: for each task, show the items you need to hand in /
      collect, ideally with icons (see icon sourcing below).
- [ ] **Task + subtask locations**: place/route to the task's objective zones (gamedata typed
      quest triggers + tasks.json zones already exist — surface them richly; show subtask
      objectives distinctly).
- [ ] **Filter tasks by map**: only show tasks for the current map (or a chosen map).
- [ ] Cool visualizations: item-requirement checklists, objective progress, trader grouping,
      Kappa/Lightkeeper flags, level gating — mine tasks.json for what's available.

### Icon sourcing (for task items — investigate, agent)
- Prefer **tarkov.dev item icons** (already have `fetch_icons.py` → `packs/shared/icons/`;
  the loose-loot/inspect cards use them). Reuse that path for task-required items.
- **Distribution caveat**: tarkov.dev icons may be a redistribution concern. The agent must
  check PAST CHATS/history for whether we ever successfully exported item icons/images FROM
  THE GAME FILES (the earlier loose-loot agent concluded the client renders icons at runtime
  from 3D prefabs and ships NO 2D sprites — confirm/re-investigate; there was a runtime
  `%TEMP%/Icon Cache` of rendered PNGs). If game-file export is viable, prefer it so the
  bundle needs no tarkov.dev redistribution. Document the verdict.

### Icon sourcing — VERDICT (Tasks-tab agent, 2026-07-17)

**Confirmed: the client ships NO redistributable 2D item sprites; tarkov.dev CDN stays the only
practical source.** Re-verified against the transcripts and the code:

- Item bundles are 3D mesh/texture assets. The game RENDERS grid icons at runtime into
  `%TEMP%/Battlestate Games/EscapeFromTarkov/Icon Cache/live/<n>.png`, keyed by **opaque numeric
  hashes with no template-id index** (transcript `0467e5fb…`, line 65: *"renders each item icon on
  demand from a 3D prefab and writes it into that cache"*). The earlier loose-loot/stashscan work
  had to build `iconmatch.py` (Sobel-edge cosine matching) **and** fall back to OCR-of-name to tie
  those PNGs back to item ids — direct proof there is no usable game-file index from icon → item.
  The provenance is already documented in `fetch_icons.py` (header) and `viewer/src/inspect.rs`.
- **Game-file export is NOT viable for a redistributable, item-keyed set.** The runtime cache is
  (a) unindexed (opaque hashes) and (b) incomplete (only items the player has seen). A "proper"
  export would mean loading every item prefab in Unity and rendering each icon offline — a heavy
  bespoke renderer — and the output is STILL game-derived art, i.e. the same redistribution status
  as tarkov.dev (arguably worse, since tarkov.dev is a community-run, attribution-licensed source).
- **Pipeline = `fetch_icons.py` → `packs/shared/icons/<slug>.png`, cached at BUILD time; the exe
  ships no game-derived assets.** The panel reuses `inspect`'s icon cache (mirrored as
  `TaskIconCache`); items with no cached slug **degrade to a text chip** (full name on hover).
- **Gap found + closed:** `fetch_icons.py` previously fetched only loose-loot pools, door keys and
  lock/jackpot items, so only **128 of 3,563** distinct task-item names resolved. Extended it to
  also pull **task items** — per-map by default (~35–61 names/map, e.g. ground_zero 61, customs 53,
  lighthouse 35) and `--tasks-all` for the full 3,563-name catalog (chunked at 400/req). Re-run
  `python extraction/intel/fetch_icons.py <map>` per pack (or once with `--tasks-all`) to backfill.

## Notes / constraints
- egui 0.32 via bevy_egui 0.37; EguiPrimaryContextPass; ASCII glyph whitelist for labels.
- layers_panel is at/near the 16-system-param limit — bundle new params into a SystemParam.
- No game-derived assets shipped in the exe (vector icons or user-extracted-at-runtime only).
- Keep everything hideable; keep the Tarkov gear-screen visual language.

## Loot-run planner — shipped v1 + roadmap (2026-07-17)

Shipped v1 (planner.rs, Navigation tab "LOOT PLAN"):
- [x] Orienteering solver: cheapest-insertion by value density -> 2-opt -> real A* legs
      with budget repair; ONE bounded Dijkstra reachability field prunes off-mesh loot
      (solve ~250 ms for 10 stops on interchange).
- [x] Knobs: min item value, stop cap, walking budget; honors avoid-bosses/PMCs/scavs.
- [x] Ends at the best extract (re-picked after stops settle); ordered stop list with
      values + leg metres; gold stop orbs; tour drawn via the route machinery.
- [x] Menu INTEL strip: tarkov.dev sync age (loot/tasks/icons) + SYNC NOW job
      (tools/sync_intel.py -> build_loot + build_tasks + fetch_icons -> packs/shared).

Roadmap (in rough priority order):
- [ ] TIME model, not just distance: loot.json containers carry `t` (loot seconds); budget
      the raid in MINUTES (walk speed + looting time + extract buffer), show ETA per stop.
- [ ] Key awareness: skip locked-room loot unless its key is marked owned (locks/keys data
      exists per map); "keys I have" checklist that unlocks those candidates.
- [ ] Container-type weighting: safes/registers vs crates value multipliers; dedupe stops
      within N metres into one "cluster" stop (5x adjacent fuel tanks = one stop).
- [ ] Task integration: fold TRACKED task objectives into the tour as mandatory stops
      (prize-collecting with required nodes) — quest + loot in one run.
- [ ] Multiple plan variants like routes (greedy vs balanced vs safe) with value/dist/risk
      trade-off list; risk score = summed avoid-zone exposure along the tour.
- [ ] Spawn-aware start presets: pick an actual PMC spawn as start (spawn list exists).
- [ ] Numbered stop badges on the map (world-to-viewport egui overlay like inspect cards).
- [ ] Persist/share plans (JSON in the pack dir; "copy plan" for squadmates).
- [ ] Live re-plan: after walking past stop K, re-solve the remainder from here.
- [ ] Data: fetch --tasks-all icons once (3.5k items) so every stop row gets item art;
      surface per-map loot-data age on the map cards (already on the INTEL strip).
