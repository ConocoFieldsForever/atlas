# UI Roadmap — toolbar, camera modes, task revamp

Living TODO for the viewer UI overhaul. Check items off as they land; keep the "Status"
column honest. Ordered roughly by dependency + payoff.

## 0. Structure — vertical icon toolbar (foundation for everything else)

A **thin, one-button-wide vertical toolbar** pinned to the LEFT edge of the right panel
(between the 3D view and the existing `SidePanel::right`). Each icon selects WHICH settings
group the right panel shows — the panel becomes a switchable container instead of one long
scroll. State: a `RightPanelTab` enum resource (`Layers | Camera | Tasks | …`), default
`Layers` (current behavior). Tarkov gear-screen styling (square, charcoal, beige active tab).

- [ ] `RightPanelTab` resource + toolbar widget (egui vertical strip, icon buttons).
- [ ] Route the existing MAP LAYERS content behind the `Layers` tab (no behavior change).
- [ ] Active-tab highlight; tooltips on hover.
- Icons: vector-drawn in egui (no image assets — keep the redistribution story clean), or a
  tiny embedded glyph set. Camera / eye / clipboard-check shapes.

## 1. Camera tab (icon: camera)

### 1a. Camera settings (right panel, under the Camera tab)
- [ ] **FOV** slider (perspective vertical FOV; wire to `Projection`).
- [ ] **Exposure** slider (already have `GfxSettings.grade_exposure` — surface it here; today
      only env `EFT_GRADE_EXPOSURE`). Consider also bloom intensity, fog, sky-refl (some
      already exist under "Graphics (experimental)" — decide: move them here or cross-link).
- [ ] Near/far, and a "reset framing" button (re-run content_anchor).

### 1b. Fly-cam scroll-wheel speed  ← QUICK WIN, do first
- [ ] Scroll wheel adjusts fly-cam `speed` (multiplicative, clamped). Up = faster, down =
      slower. Persist across the session. HUD or tab shows the current speed.

### 1c. Walking camera mode
- [ ] Toggle **Walk mode** vs **Fly mode**. Walk = eye-height constrained (~1.7 m over the
      ground), WASD along the ground plane, mouse-look, **jump** (space), gravity.
      - Ground follow: sample terrain/mesh height under the camera. Options: (a) reuse the
        pathfind/nav mesh if available, (b) raycast down against world geometry, (c) simple
        terrain-height sample. Pick the cheapest that feels right; a raid planner doesn't need
        perfect collision — floor-follow + jump is enough.
- [ ] **Scroll wheel in walk mode** scales walk/sprint speed AND jump height together by a
      factor (scroll up = faster + jumps higher = "more player-like / juiced"). One knob,
      fun. Clamp to sane bounds.
- [ ] Sprint (shift) in walk mode; the scroll factor multiplies both walk and sprint.

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

## Notes / constraints
- egui 0.32 via bevy_egui 0.37; EguiPrimaryContextPass; ASCII glyph whitelist for labels.
- layers_panel is at/near the 16-system-param limit — bundle new params into a SystemParam.
- No game-derived assets shipped in the exe (vector icons or user-extracted-at-runtime only).
- Keep everything hideable; keep the Tarkov gear-screen visual language.
