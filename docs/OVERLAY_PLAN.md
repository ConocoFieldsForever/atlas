# Overlay — putting the map over the running game, and what the pinned stack will actually allow

The ask: press `~` while Escape from Tarkov is in the foreground, get Atlas on top of it, rotate and
fly the map with the controls we already have, press `~` again and hand input back to the game.

This document records what is **reachable** on the exact versions this repo pins, what is **not**, and
where the boundary sits between a map viewer and something that gets a user banned. Every
version-specific claim below was read out of the vendored crate sources under
`~/.cargo/registry/src/index.crates.io-*`, not from memory — the Bevy/winit/wgpu APIs move fast enough
that recalled knowledge is worthless here. Anti-cheat claims carry their source tier: **[DOC]** vendor
or publisher document, **[RE]** independent reverse-engineering, **[FORUM]** unverified community
claim, **[UNKNOWN]** searched and found nothing either way.

Three findings drive the whole design, and two of them are negative:

1. **Per-pixel window transparency is not reachable under this repo's Vulkan-only policy.** Requesting
   it is not a degraded experience — it is a hard wgpu validation error, which under `panic = "abort"`
   is a process abort.
2. **The click-through idle mode the ask implies is the exact window fingerprint anti-cheats look for
   in an ESP overlay**, and winit's API for it sets precisely the two style bits that fingerprint
   names. We must not ship it.
3. **Everything else the feature needs is already live-settable** on Bevy 0.17.3 with no new
   dependency, no `unsafe`, and no window rebuild.

So the recommendation is a **bounded, opaque, borderless, always-on-top, mouse-interactive panel** —
not a full-screen transparent HUD. It delivers the ask (map over the game, `~` to toggle, rotate +
WASD) while being structurally distinguishable from a cheat overlay at the OS level.

---

## 1. Versions this design is pinned against

Read from `Cargo.lock` and `Cargo.toml`. Nothing here is inferred.

| crate | version | lock line |
|---|---|---|
| `bevy` / `bevy_window` / `bevy_winit` | 0.17.3 | `Cargo.lock:440, 1458, 1475` |
| `winit` | 0.30.13 | `Cargo.lock:5854` |
| `wgpu` (pinned `=26.0.1`, feature-unified with Bevy's) | 26.0.1 | `Cargo.lock:5349`, `viewer/Cargo.toml` |
| `bevy_egui` / `egui` | 0.37.1 / 0.32.3 | `Cargo.lock:685, 2089` |
| `raw-window-handle` | 0.6.2 | `Cargo.lock:4157` |
| `windows-sys` | 0.52.0, 0.59.0, 0.60.2, 0.61.2 | `Cargo.lock:5680, 5688, 5697, 5706` |
| `windows` | 0.58.0, 0.61.3 | `Cargo.lock:5508, 5518` |

Backends are **Vulkan only** on Windows and Linux — `viewer/src/render/mod.rs:246-263`, because DX12
panics at pipeline creation on Bevy's own `downsample_depth.wgsl` (wgpu#5683) before any render path
runs. Section 3 is entirely a consequence of that line.

Release profile is `panic = "abort"` (`Cargo.toml`), so a wgpu validation error is not a recoverable
`Result` at the app layer — it is a dead process. That raises the cost of every "just try it and see"
in this design.

---

## 2. The window: what is live-settable, and what needs a rebuild

`bevy_winit` 0.17.3 propagates a specific, enumerable subset of `Window` field changes to the live
winit window each frame, in `changed_windows` (`bevy_winit-0.17.3/src/system.rs:301`) and
`changed_cursor_options` (`system.rs:569`). Everything else is read once in
`WinitWindows::create_window` (`winit_windows.rs:56`). The crate documents the split itself at
`system.rs:293-300`:

```
/// - [`Window::transparent`] cannot be changed after the window is created.
/// - [`Window::focused`] cannot be manually changed to `false` after the window is created.
```

| `Window` field | live at runtime? | evidence |
|---|---|---|
| `window_level` (`AlwaysOnTop`) | **yes** | `system.rs:486-488` -> `set_window_level` |
| `decorations` | **yes** | `system.rs:411-415` -> `set_decorations` |
| `visible` | **yes** | `system.rs:519-521` -> `set_visible` |
| `position` | **yes** | `system.rs:444-460` -> `set_outer_position` |
| `resolution` | **yes** | `system.rs:358-400` -> `request_inner_size` |
| `mode` (`BorderlessFullscreen`) | **yes** | `system.rs:317-356` -> `set_fullscreen` |
| `present_mode` | **yes** (surface reconfigured) | `bevy_render-0.17.3/src/view/window/mod.rs:385` |
| `CursorOptions::hit_test` (click-through) | yes — **but see §4** | `system.rs:606-616` -> `set_cursor_hittest` |
| `focused` | **one way only**: `false -> true` calls `focus_window()` | `system.rs:482-484` |
| `transparent` | **no — actively reverted** | `system.rs:490-494` |
| `composite_alpha_mode` | **no — silently inert** | `view/window/mod.rs:385` |
| `skip_taskbar`, `clip_children`, `name` (class name) | **no**, creation only | `winit_windows.rs:136-143`, `:242-249` |

Two of those rows are load-bearing enough to quote verbatim.

`transparent` is not merely ignored; Bevy writes the old value back and warns
(`bevy_winit-0.17.3/src/system.rs:490-494`):

```rust
// Currently unsupported changes
if window.transparent != cache.transparent {
    window.transparent = cache.transparent;
    warn!("Winit does not currently support updating transparency after window creation.");
}
```

`composite_alpha_mode` is worse — it is read into the `SurfaceConfiguration` once
(`bevy_render-0.17.3/src/view/window/mod.rs:358-369`) and the surface is only ever reconfigured on
`size_changed || present_mode_changed` (`:385`). Changing it at runtime produces no warning, no error,
and no effect. That is a silent trap worth naming here so nobody loses an afternoon to it.

**Consequence for the design:** the entire Desktop <-> Overlay presentation flip — borderless,
topmost, repositioned, resized, shown/hidden — is four field writes on the existing `PrimaryWindow`
entity. No window rebuild, no second window, no new dependency, no `unsafe`. That is the whole of
phase 1.

The repo currently sets **none** of these. `main.rs:730-757` configures only `title`, `resolution`,
`present_mode`, `visible`, `position`, `skip_taskbar`, and a repo-wide grep for
`WindowLevel|decorations|transparent:|composite_alpha` finds no `.rs` hits. This is greenfield.

---

## 3. Per-pixel transparency is not reachable on Vulkan/Windows, and asking for it aborts

This is the finding that reshapes the ask, so it gets the full chain.

A per-pixel-transparent window needs the swapchain to composite with alpha — `PreMultiplied` or
`PostMultiplied`. `bevy_window` documents the requirement at `window.rs:216-226`: *"You should also set
the window `composite_alpha_mode` to `CompositeAlphaMode::PostMultiplied`."*

wgpu's Vulkan backend reports whatever the driver's `VkSurfaceCapabilitiesKHR::supportedCompositeAlpha`
says, mapped one-for-one — `wgpu-hal-26.0.6/src/vulkan/adapter.rs:2611` calls
`conv::map_vk_composite_alpha(caps.supported_composite_alpha)` (`conv.rs:503-516`). On Windows,
`VK_KHR_win32_surface` conventionally reports **only** `VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR`; alpha
compositing of a swapchain on Windows goes through DXGI/DirectComposition, which is the DX12 path this
repo has disabled. **This specific claim is the one item in the document I could not verify on
hardware** — it is driver-reported, so it must be confirmed by probing
`Surface::get_capabilities().alpha_modes` on the target adapter before any transparency work starts.
See `## Known gaps`.

What is *not* uncertain is the failure mode if the probe comes back OPAQUE-only. `wgpu-core`
`device/global.rs:1786-1822`:

```rust
let fallbacks = match config.composite_alpha_mode {
    wgt::CompositeAlphaMode::Auto => &[Opaque, Inherit][..],
    _ => {
        return Err(E::UnsupportedAlphaMode {
            requested: config.composite_alpha_mode,
            available: caps.composite_alpha_modes.clone(),
        });
    }
};
```

Only `Auto` degrades. An **explicit** `PostMultiplied` on an adapter that does not support it is a
validation error (`wgpu-core-26.0.1/src/present.rs:105-108`, classified `ErrorType::Validation` at
`:140-141`), raised inside `configure_surface` during Bevy's window setup — and with `panic = "abort"`
that is `0xC0000409`, not a fallback. The same class of abort this repo already fought once, recorded
in `viewer/src/gpu_lease.rs:1-26`.

There is a second, independent reason it would not work even if the swapchain cooperated. winit's
Windows implementation of `with_transparent(true)` is not a swapchain setting at all — it is
`DwmEnableBlurBehindWindow` with an empty blur region (`winit-0.30.13/src/platform_impl/windows/
window.rs:1228-1247`), and it is skipped entirely when `no_redirection_bitmap` is set. Bevy never calls
`with_no_redirection_bitmap` (grep of `bevy_winit-0.17.3` finds only `with_skip_taskbar` and
`with_clip_children`, `winit_windows.rs:136-143`), so we would get the DWM blur-behind path — which
composites nothing useful behind an opaque swapchain.

And even if both of those resolved favourably: `transparent` is creation-only (§2), so a runtime `~`
toggle could not reach it. It would have to be `true` from launch for every session, including normal
desktop use.

**Verdict: do not pursue per-pixel transparency.** Section 10 describes what we ship instead, and
section 12 records the two transparency variants that were considered and rejected.

---

## 4. Click-through is the ESP fingerprint. We are not shipping it

The ask's "click-through when idle, input-capturing when active" model is the standard overlay
pattern, and on this stack it is a one-line change: `CursorOptions::hit_test = false`, live-settable at
`bevy_winit-0.17.3/src/system.rs:606-616`.

It is also, precisely, the thing to avoid.

winit implements click-through by setting `WindowFlags::IGNORE_CURSOR_EVENT`, which expands to
(`winit-0.30.13/src/platform_impl/windows/window_state.rs:300-301`):

```rust
if self.contains(WindowFlags::IGNORE_CURSOR_EVENT) {
    style_ex |= WS_EX_TRANSPARENT | WS_EX_LAYERED;
}
```

and `WindowLevel::AlwaysOnTop` adds `WS_EX_TOPMOST` at `:277`.

The published external-overlay detection heuristic is exactly that combination: a top-level window
sized to the game's client area carrying `WS_EX_LAYERED | WS_EX_TRANSPARENT`, usually with
`WS_EX_TOPMOST`, confirmed by a click-through "nudge test" **[FORUM/RE]**
(guidedhacking.com/threads/how-to-detect-external-overlays.14629). And BattlEye is documented as
enumerating top-level windows with `GetTopWindow()` and pattern-matching their titles against cheat
signatures, with *"window handles inside of the game process ... excluded from the aforementioned
enumeration"* **[RE]** (secret.club/2019/02/10/battleye-anticheat.html) — i.e. the enumeration exists
specifically to inspect external windows like ours.

`WS_EX_TOPMOST` alone is unremarkable desktop behaviour — Task Manager's always-on-top, sticky notes,
RTSS. The fingerprint is the *conjunction* of full-screen-sized, layered, transparent-to-input, and
topmost. We take topmost and decline the other three.

This costs the user something real, and the doc should be straight about it: with no click-through,
the overlay panel eats mouse clicks in its own rectangle while it is up. Since the design also gives it
focus while up (§7), that is consistent rather than surprising — the overlay is up or it is not.

---

## 5. Tarkov's display mode: what the user must set, and what happens if they don't

No ordinary top-level window can be drawn over a game holding a **true exclusive-fullscreen**
swapchain; the display is owned by that swapchain and the DWM is out of the composition path. This is
an OS/driver property, not an anti-cheat one, and no amount of `WS_EX_TOPMOST` changes it.

EFT offers Fullscreen / Borderless / Windowed. Behaviour per mode:

| EFT setting | overlay appears? | notes |
|---|---|---|
| **Borderless** (recommended) | **yes** | DWM composites both; topmost wins. This is the supported configuration. |
| **Windowed** | **yes** | Same path as borderless. |
| **Fullscreen** (exclusive) | **no**, or a mode-flicker | Best case the overlay is invisible; worst case showing it forces the game to drop out of exclusive mode, causing a black flash and a stall — mid-raid, that is worse than the feature is worth. |

Two hard requirements fall out:

1. **Atlas must detect and tell the user, not fail silently.** If the user is in exclusive fullscreen,
   pressing `~` looks like a broken hotkey. Detection is imperfect from outside the process; the
   pragmatic answer is a one-time notice in the overlay settings UI stating the requirement, plus a log
   line whenever the overlay is shown.
2. **Elevation.** EFT runs elevated. Windows UIPI restricts cross-integrity-level window interaction,
   and community guidance for EFT overlays is explicitly to run the overlay elevated too **[FORUM]**
   (theglobalgaming.com). This affects the global hotkey more than the window (§6). Atlas should not
   silently self-elevate; it should detect that it is *not* elevated, and say so if the global hotkey
   fails to register or fails to fire.

---

## 6. The global hotkey: `RegisterHotKey`, and why not a keyboard hook

Bevy only sees key events while its window is focused, so an in-app `Backquote` binding solves the
"dismiss" direction but not the "summon" direction. Summoning needs a system-wide hotkey.

**`Backquote` is free.** A full sweep of `KeyCode::` across `viewer/src` returns 16 hits in three
files, all accounted for: WASD/QE/Shift for the fly camera (`main.rs:1288-1310`), WASD/Shift/Space for
walk mode (`main.rs:1362-1384`), Shift as a pick/place modifier (`pick.rs:170`, `inspect.rs:211`), and
a single `Escape` that cancels marker-place mode (`pick.rs:157`). No F-keys are bound. There is no
egui-side key handling at all (`egui::Key::` / `consume_key` return zero matches). Nothing collides.

Three mechanisms exist for the system-wide half:

| mechanism | footprint | verdict |
|---|---|---|
| **`RegisterHotKey`** (user32) | registers one key combo with the OS; sees *no* other keystrokes; nothing installed globally, no cross-process presence | **recommended** |
| `SetWindowsHookEx(WH_KEYBOARD_LL)` | a global hook DLL/callback in the desktop input path; observes **every** keystroke system-wide | **rejected** — see below |
| `global-hotkey` crate | wraps `RegisterHotKey` on Windows | **rejected** — new dependency for a ~40-line call we can make directly |

**Why not the low-level hook.** I found **[UNKNOWN]** — no vendor statement, ban report, or RE writeup
naming `WH_KEYBOARD_LL` as an anti-cheat detection vector, and enormous legitimate usage (Discord and
Steam push-to-talk, OBS hotkeys, PowerToys, screen readers, IMEs). But the anti-cheat literature that
*does* exist around keyboard APIs is all about **synthesised** input — the `LLKHF_INJECTED` /
`LLKHF_LOWER_IL_INJECTED` flags and `GetCurrentInputMessageSource()`. A low-level hook sits adjacent to
that surface and buys us nothing `RegisterHotKey` does not already deliver. Taking the API with the
smaller observable footprint is free; take it. Notably, a shipping third-party EFT overlay makes the
same call and advertises it as a safety property, explicitly listing "Hook the game window / low-level
keyboard" among the things it does not do **[DOC, third-party]** (tarkov.aquapado.com).

**No new dependency is required.** `windows-sys 0.52.0` is already compiled into the tree as winit's
own Windows dependency (`Cargo.lock:5680`, `winit-0.30.13/Cargo.toml:525-553`), and winit already
enables both features we need — `Win32_UI_Input_KeyboardAndMouse` (`:547`, `RegisterHotKey` /
`UnregisterHotKey`) and `Win32_UI_WindowsAndMessaging` (`:552`, `PeekMessageW` / `WM_HOTKEY`). Adding

```toml
[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.52", features = [
    "Win32_Foundation", "Win32_UI_Input_KeyboardAndMouse", "Win32_UI_WindowsAndMessaging",
] }
```

unifies onto the already-built `0.52.0` and adds **zero** packages to the graph. (Do not reach for
`windows = "0.62"` or newer: the tree already carries `windows` 0.58.0 and 0.61.3, and a third major
version would be a real cost for no benefit.)

**Threading.** `RegisterHotKey` delivers `WM_HOTKEY` to the *registering thread's* message queue.
Bevy/winit owns the main thread's message pump, and posting into it is intrusive. The clean shape is a
dedicated thread that registers the hotkey, runs its own `GetMessageW` loop, and forwards presses over
an `mpsc::channel` — which is **exactly the shape `viewer/src/game_watch.rs` already uses** for the log
watcher (`game_watch.rs:110-116`: spawn a named thread, `Sender`/`Receiver`, a Bevy system drains
`try_iter()` each frame). Copy that pattern; do not invent a second one.

---

## 7. Anti-cheat and fair play

This is a first-class constraint, not a footnote. Atlas is a legitimate viewer, and the design's job is
to keep it *visibly* legitimate at the OS level.

### 7.1 What Atlas does today, stated precisely

`viewer/src/game_watch.rs:1-17` is already the honest statement of the data surface: it reads
`Logs\log_*\*application.log` (which map is loading), `*notifications.log` (task status pushes), and the
filenames in `Documents\Escape From Tarkov\Screenshots` — into which EFT itself bakes the local
player's world position and view quaternion. Nothing touches the game process. The overlay adds a
window and a hotkey; it must add nothing else.

### 7.2 The vendor documents

BattlEye's support page is unusually direct **[DOC]** (battleye.com/support):

> "No one is banned for using non-hack programs (like Fraps, overlays, etc.), picking up or using
> hacked in-game items, weapons or vehicles, being on a server at the same time as a cheater, or other
> passive non-cheating activity."

and their FAQ **[DOC]** (battleye.com/support/faq):

> "Generally we only ever ban for the use of actual cheats/hacks or components of such hacks which are
> designed to intentionally bypass BE's protection. ... For example, non-cheat overlays and visual
> enhancement tools like Reshade or SweetFX are generally supported ... We might decide to kick (not
> ban) you at some point for using a specific program (such as macro tools), but that won't
> automatically flag you as a cheater."

Two things carry: overlays are named as fine, and the escalation ceiling for grey-area utilities is
**kick, not ban** — in BattlEye's own words.

The documented failure mode for legitimate software in BattlEye titles is being **blocked, kicked, or
crashed** — never banned. DisplayFusion's `AppHook64_*.dll` was blocked in EFT after two separate
patches with no bans **[DOC]** (Steam community thread); MSI Dragon Center produces a "Corrupted Memory
#0" kick; ReShade is blocked in specific titles — all from the BattlEye FAQ. I located **no credible,
evidenced case of an EFT ban caused by an overlay, hotkey utility, or capture software.**

### 7.3 The contractual edge, which is sharper than the technical one

BSG's license agreement clause 4.3.4 prohibits, among other things, *"use of outside software that
captures, collects, counts or otherwise 'retrieves' information"*, with permission reserved to the
developers' discretion **[DOC, with a caveat]**. The caveat matters: the EFT-domain English agreement
would not resolve for me (404 / JS shell); this text is from the Arena agreement at
`arena.tarkov.com/legal/license_agreement`, which BSG's own support KB cites by the same clause number.
**Re-verify against the EFT-specific URL before quoting this in user-facing material.**

A tool that parses EFT log files and screenshot filenames *is* outside software that retrieves
information. The mitigations are real but they are mitigations, not permission: the data is written to
disk by the game for the user; BSG deliberately created the screenshot-coordinate mechanism (their bug-
reporting guidance instructs users to take in-game screenshots *"so that the filename contains the
coordinates"*); and BSG has never enforced against the large public ecosystem doing exactly this
(TarkovMonitor, tarkov.dev, TarkovMapTracker). BSG's only direct statement on third-party software
**[DOC]** (forum.escapefromtarkov.com/topic/157871) turns on the verbs *replace / override / modify* —
all write-side, all things Atlas does not do.

**State this plainly to users: that is non-enforcement, not authorisation.** No BSG statement exists
about TarkovMonitor or tarkov.dev in either direction **[UNKNOWN]**. The nearest thing to a
counterexample is TarkovMonitor issue #162, which its own maintainers titled *"implausible claim of ban
due to use"* and whose body is content-free.

### 7.4 Where the line is

| | radar / ESP cheat | Atlas with an overlay |
|---|---|---|
| data source | live client<->server packets, or the game's runtime heap | files EFT wrote to the user's own disk |
| whose position | **every player and AI on the server** | **only the local player** |
| freshness | continuous, real-time, unprompted | discrete, only when the user presses the screenshot key in raid |
| access mechanism | packet capture / DMA / injected ESP | `ReadFile` on the user's Documents folder |
| BSG's countermeasure | packet encryption bought specifically to kill radar; BattlEye strips process handles and hooks `ReadProcessMemory` into its driver **[RE]** | none — BSG *added* the coordinate mechanism |

The "only your own position, only when you deliberately act, at the cost of a screenshot keypress"
property is the entire ethical argument. It should be stated in the UI, not just in this file.

### 7.5 Rules the design must not break

Absolute, no-exceptions list. Anything here would move Atlas from "viewer" to "cheat":

- **No `OpenProcess` / `ReadProcessMemory` / `WriteProcessMemory` against the game.** BattlEye strips
  process handles of read/write access and redirects RPM/WPM to its BEDaisy driver **[RE]**. This is
  squarely "actual cheats" by BattlEye's own criterion.
- **No DLL injection, no hooking the game.** This is the one axis where an overlay could be materially
  worse off than the status quo: an unknown DLL entering a BE-protected process is exactly the category
  BE treats as suspicious-until-checked, and BE monitors even Steam's own `gameoverlayui.exe` for
  vtable manipulation and thread-suspension **[RE]**. There is no upside for a map viewer. Atlas
  renders in its own window and its own process.
- **No packet capture.**
- **No `WS_EX_TRANSPARENT` click-through, no full-screen-sized overlay** (§4).
- **No self-elevation, no driver, no service, no obfuscation, no packing.** Ship a normal signed
  desktop binary.
- **No synthetic input into the game** — no `SendInput`, no `PostMessage`/`SendMessage` to EFT's window.
  That is the documented detection surface (`LLKHF_INJECTED`), and no part of this feature needs it.

### 7.6 Positive obligations

- Keep the window **title and class name honest and boring** — "Atlas" is already correct, and window
  titles are literally scanned **[RE]**. Do not make the overlay window title dynamic or cryptic.
- Keep the overlay **bounded, movable, and mouse-interactive** — visually and structurally a map panel,
  not a HUD layer.
- Keep the taskbar entry. `skip_taskbar` is already wired to `EFT_HIDDEN` only (`main.rs:751-753`) and
  the overlay must not set it: a topmost window that hides itself from the taskbar is strictly more
  cheat-shaped and strictly less usable.
- Document the file-access surface in the shipped README, the way `game_watch.rs:1-17` already does in
  source.

### 7.7 Honest verdict

**Low risk, not zero risk, and the residual risk is contractual rather than technical.**

- **Known safe, vendor-documented:** overlay windows as a class; reading files on your own disk; not
  being banned for passive non-cheating activity.
- **Probably fine, unverified — and the doc should say "unverified":** `RegisterHotKey` (no evidence
  either way, and a shipping EFT overlay markets the choice as its safety property); a small topmost
  panel; consuming screenshot-filename coordinates.
- **Known fatal:** the §7.5 list.

**Two things are the user's decision, not ours**, and the UI should present them as such rather than
burying them: (a) whether they accept a tool that is tolerated-but-not-blessed under clause 4.3.4, and
(b) whether they want it on screen during a raid at all. Ship the overlay **off by default**, behind an
explicit opt-in that states the risk in one sentence, and never auto-enable it from a game-link event.

---

## 8. Input routing

**The model: the overlay is either up and focused, or hidden.** There is no "visible but ignoring
input" state — that state is the click-through mode §4 rules out, and eliminating it also eliminates
most of the ambiguity about where keystrokes are going.

- **Summon (`~` while the game is focused).** The hotkey thread posts to the channel; the Bevy system
  shows the window (`visible = true`) and raises it. Raising requires focus, which winit will give us
  via `Window::focus_window()` — documented as *"This method steals input focus from other
  applications"* (`winit-0.30.13/src/window.rs:1303-1312`). Here that is the intent, not a bug. Bevy
  exposes it as the `false -> true` transition on `Window::focused` (`bevy_winit/src/system.rs:482-484`);
  note the reverse is not available — `Window::focused` *"cannot be set unfocused after creation"*
  (`bevy_window/src/window.rs:227-237`).
- **Controls while up.** Nothing new is needed. `flycam_look` (RMB-drag + `AccumulatedMouseMotion`),
  `flycam_move` (WASD/QE/Shift), and `cursor_grab` already run as a chained set gated by
  `run_if(not(resource_exists::<menu::MenuState>))` (`main.rs:885-890`), and already respect the
  existing UI-focus gates `PointerOnUi` / `UiWantsKeyboard` (`inspect.rs:137-143`, read at
  `main.rs:1225, 1251, 1274, 1339`). **Hook the overlay into that existing gate; do not add a second
  focus flag.**
- **Dismiss (`~` again).** Set `visible = false`. Windows then activates the next window in z-order,
  which in the intended configuration is the game. This is the whole return path, and it needs no Win32
  call against EFT's window.
- **Explicitly rejected fallback:** `FindWindowW` + `SetForegroundWindow` on EFT's HWND to force focus
  back. It is user32-only and touches no process handle, so it is not in the §7.5 category — but it is
  reaching for the game's window when hiding our own already achieves the goal. If hide-only proves
  unreliable in testing, this is the escalation, and it should be recorded as a deliberate decision
  rather than slipped in.
- **Alt-tab / game focus loss.** If the user alt-tabs away from a visible overlay, it stays visible but
  unfocused, and `WinitSettings::unfocused_mode` drops it to ~2 Hz (§9). That is correct for GPU
  sharing and looks stuttery — which is fine, because it is a transient state the user resolves by
  pressing `~`. Optionally: auto-hide on `WindowFocusLost`. Recommend **not** doing that in phase 1; it
  makes the overlay vanish when the user clicks anything else, which is worse than a stuttery panel.
- **EFT in exclusive fullscreen.** The hotkey still fires and the window still shows — the user simply
  cannot see it, and their keystrokes now go to Atlas instead of the game. This is the single most
  confusing failure mode in the feature (§13) and is the reason §5's user-facing notice is a
  requirement rather than a nicety.

---

## 9. Performance coexistence

Atlas and EFT share one GPU, and this repo has already paid for getting that wrong once. The overlay
makes the sharing permanent.

**What already exists, and must not be broken.** `main.rs:842-852` installs the only `WinitSettings`
in the repo:

```rust
if !uncapped {
    app.insert_resource(bevy::winit::WinitSettings {
        focused_mode: bevy::winit::UpdateMode::Continuous,
        unfocused_mode: bevy::winit::UpdateMode::reactive_low_power(
            std::time::Duration::from_millis(500),
        ),
    });
}
```

paired with `PresentMode::AutoVsync` by default (`main.rs:682-687`). The comment there already states
the intent: *"so with the game in the foreground the viewer stops churning the GPU."* **The overlay
design inherits this for free and should not touch it.** Hidden overlay = unfocused = ~2 Hz reactive
redraw. Visible overlay = focused = continuous, vsync-capped. That is the correct answer to
"frame-rate capping / vsync-off-when-hidden / rendering only when visible" and it is already written.

Two consequences to respect:

- **`EFT_UNCAPPED=1` disables the gate** (`main.rs:845`). Overlay mode must never set it, and the
  settings UI should not expose it. Document that the two are incompatible.
- **`visible = false` is stronger than unfocused.** A hidden window produces no redraw requests at all,
  which is the desired hidden-state cost.

**The GPU lease.** `viewer/src/gpu_lease.rs` is held for the process lifetime from `main.rs:708-713`,
and `sh_bake.rs:748-752` checks it to route a bake to the CPU backend. An overlay means Atlas is
running for hours alongside the game, so **every bake started during that window takes the CPU path.**
That is correct — it is precisely the TDR the lease exists to prevent (`gpu_lease.rs:4-19`) — but it is
a behaviour change worth stating: map builds queued from the menu while the overlay is in use will be
slower, by design. No code change needed; a line in the settings UI would prevent a bug report.

**VRAM is the honest gap.** The existing knobs are draw-time selectors, not residency controls:

| knob | default | what it does | frees VRAM? |
|---|---|---|---|
| `EFT_CULL_PX="gen,grass"` | `1.5,4.0` | screen-size cull threshold in px (`render/mod.rs:109-115`, consumed `gpu_driven.rs:4922-4930`) | no |
| `EFT_LOD` | on | distance-LOD master; no-op on lean LOD0-only packs (`mod.rs:170-176`) | no |
| `EFT_LOD_BIAS` | `1.0` | holds finer shells farther (>1) / switches coarse sooner (<1); clamped 0.05..64, non-finite -> 1.0 (`gpu_driven.rs:4953-4972`) | no |
| `EFT_LOD_FORCE` | `-1` | debug: pin every group to shell N (`cs_cull` mode 2) | no |
| `EFT_LOAD_BUDGET_MS` | `6.0` | per-frame render-thread upload budget (`gpu_driven.rs:4876-4885`) | no |

All of these change which instances are *drawn*; the buffers stay resident. **Reducing the overlay's
VRAM footprint while hidden would require unloading the pack**, which means a reload cost on every
summon and is out of scope for this design. The right move is to be honest in the UI: running Atlas
alongside EFT costs the pack's VRAM for as long as it is open, and on a large map (Streets) on an 8 GB
card that may matter. Users on tight VRAM should close Atlas between raids rather than leave the
overlay resident. **Measure this before shipping** — see `## Known gaps`.

Recommended default for overlay mode: reuse the existing `GfxSettings` slider, and set `lod_bias`
toward the coarse end (e.g. 0.75) when the overlay is enabled, exposed and overridable. This reduces
draw cost while the overlay is up without inventing a knob.

---

## 10. What already exists, and what the overlay must not duplicate

The live game link is done and it is the reason this feature is worth building at all.
`viewer/src/game_watch.rs` already:

- follows the raid — `scene preset path:maps/<bundle>.bundle` in `application.log` maps through
  `bundle_to_map` (`:42-57`) to an in-place `MapSwitch` (`:158-180`), so the overlay is showing the
  right map without the user choosing one;
- turns each in-raid screenshot into a position fix — the world position and view quaternion baked into
  the filename, bridged to viewer space with the pipeline-wide X-flip, `viewer = (-x, y, z)`
  (`:490-518`);
- drives the "Screenshot to locate current position" toggle, which sets `cam_cmd.eye` to stand the
  camera in the player's eyes (`:184-194`), gated by an `AtomicBool` the menu flips live (`:66-77`);
- syncs task tracking from `notifications.log` and clears the player marker on `UserMatchOver`.

**The overlay adds presentation only.** It must not re-poll the screenshot folder, re-parse logs, or
add a second camera-jump path. The correct integration is: `~` summons the window; whatever
`game_watch` last put on screen is already there.

**One live bug this design depends on, which must be fixed first.**
`apply_camera_command` (`main.rs:537-540`) opens with a `fly_to`-only fast path:

```rust
fn apply_camera_command(mut cmd: ResMut<CameraCommand>, mut q: Query<(&mut Transform, &mut FlyCam)>) {
    if cmd.fly_to.is_none() {
        return; // read-only fast path: a take() through DerefMut would dirty change detection
    }
```

but `game_watch.rs:190` and `:193` set **only** `cam_cmd.eye` and never `fly_to`. The
screenshot-to-eyes pose is therefore unreachable through that guard. This is the single highest-value
thing the overlay showcases — press screenshot in raid, `~`, and you are standing where you stand — so
the guard must become `if cmd.fly_to.is_none() && cmd.eye.is_none()` before phase 1 lands. Filed here
rather than fixed, per this document's design-only scope.

Config persistence follows the established pattern exactly (`menu.rs:988-1035`, `paths.rs:113-130`): a
flat `atlas.config.json`, one key per setting, a `config_bool` reader with the default at the read site
and a `#[must_use] -> bool` saver. Copy `config_screenshot_locate` / `save_config_screenshot_locate`
verbatim in shape.

---

## 11. Recommended design

**A bounded, opaque, borderless, always-on-top Atlas window, toggled by `~`, that takes focus while up
and hides to return it.**

State machine — two states, one resource:

```
Desktop                                   Overlay
-------                                   -------
decorations   = true                      decorations   = false
window_level  = Normal                    window_level  = AlwaysOnTop
position      = <user's normal pos>       position      = <saved overlay pos>
resolution    = <user's normal size>      resolution    = <saved overlay size, default 1100x700>
visible       = true                      visible       = true  (shown) / false (dismissed)
```

- Every one of those is live-settable (§2). No rebuild, no second window, no `unsafe`, no Win32 for the
  window itself.
- `transparent` stays `false` for the reasons in §3. `hit_test` stays `true` for the reasons in §4.
- The overlay window keeps its taskbar entry, its honest title, and its normal class name (§7.6).
- Off by default; opt-in with a one-sentence risk statement; persisted under an `overlay*` key set in
  `atlas.config.json`.

The user-visible result: EFT in Borderless, `~` brings up a map panel over the game with the current
raid map already loaded and the player marker already placed, WASD/RMB-look work as they always have,
`~` again puts them back in the game.

---

## 12. Rejected alternatives

| rejected | why |
|---|---|
| **Per-pixel transparent full-screen HUD** | Vulkan-on-Windows almost certainly reports OPAQUE-only composite alpha, and an explicit unsupported mode is a wgpu validation error -> `panic = "abort"` (§3). Also creation-only, so `~` could not toggle it. |
| **Uniform window alpha via `SetLayeredWindowAttributes(LWA_ALPHA)`** | The genuine middle ground: `WS_EX_LAYERED` *without* `WS_EX_TRANSPARENT` on a bounded, hit-testing panel is not the ESP fingerprint (§4), and it is what many ordinary desktop utilities do. Rejected **for now**, not permanently: it needs a raw HWND, and layered-window compositing over a Vulkan swapchain is unverified on this stack. Revisit as an optional phase only after the core works. |
| **Click-through when idle** | Sets `WS_EX_TRANSPARENT \| WS_EX_LAYERED` (`winit .../window_state.rs:300-301`) — the documented external-overlay detection signature (§4). |
| **`WindowMode::BorderlessFullscreen` for the overlay** | Live-settable and easy, but a full-screen-sized topmost window over the game is half the ESP fingerprint even without transparency, and it makes the game completely invisible while up — which defeats the point of an overlay. |
| **DLL injection into EFT (the Discord/Steam/Overwolf approach)** | The only genuinely "proper" overlay technique, and categorically off-limits: unknown DLLs in a BE-protected process are what BE is built to scrutinise (§7.5). No upside for a map viewer. |
| **`SetWindowsHookEx(WH_KEYBOARD_LL)`** | Sees every keystroke system-wide and sits adjacent to the injected-input detection surface, for zero capability gain over `RegisterHotKey` (§6). |
| **`global-hotkey` crate** | Wraps `RegisterHotKey` on Windows. A new dependency and a new supply-chain surface for a call we can make directly against a `windows-sys` already in the tree (§6). |
| **A second Bevy window dedicated to the overlay** | Would allow a permanently-`transparent` overlay window alongside a normal desktop window, sidestepping the creation-only limit. Rejected: multi-window camera targeting plus a second `bevy_egui` context is a large change, and §3 means the transparency it would buy is unreachable anyway. |
| **`SetForegroundWindow` on EFT's HWND to return focus** | Hiding our own window already returns focus. Kept as a documented escalation if hide-only proves unreliable (§8), not a starting position. |
| **New per-overlay LOD/cull knobs** | `GfxSettings` already carries `lod_bias`, `cull_px`, `lod_force` (`render/mod.rs:107-181`). Overlay mode should set existing knobs, not add parallel ones (§9). |

---

## 13. Phased implementation plan

Smallest useful first. Each phase is independently shippable and independently revertible.

### Phase 0 — unblock (prerequisite, ~5 lines)
Fix the `apply_camera_command` guard so `cmd.eye` is reachable (§10). Without this the overlay's best
demo does not work.
**Files:** `viewer/src/main.rs` (`:537-540`).

### Phase 1 — in-app overlay toggle (no new deps, no `unsafe`, no Win32)
A new `viewer/src/overlay.rs` with an `OverlayPlugin`:
- `#[derive(Resource)] struct OverlayState { on: bool, shown: bool, saved: Option<(IVec2, UVec2)> }`
- one system reading `KeyCode::Backquote` (confirmed free, §6) that flips `on`, gated on
  `!ui_kb.0` the way `pick.rs:157` gates `Escape`;
- one system that, on change, writes `decorations` / `window_level` / `position` / `resolution` /
  `visible` on the `PrimaryWindow` entity and restores the saved desktop geometry on exit;
- config accessors mirroring `config_screenshot_locate` (§10).

Useful on its own: alt-tab to Atlas, press `~`, get a borderless topmost map panel that stays above the
game when you click back. Proves the whole window half of the design with zero risk surface.
**Files:** `viewer/src/overlay.rs` (new), `viewer/src/main.rs` (add plugin, register resource),
`viewer/src/menu.rs` (config accessors + the opt-in checkbox with its risk sentence).

### Phase 2 — global hotkey (the feature the user actually asked for)
A `RegisterHotKey` thread modelled on `game_watch.rs:110-116`: register `VK_OEM_3` (`~`) with
`MOD_NOREPEAT`, run a `GetMessageW` loop, forward `WM_HOTKEY` over an `mpsc` channel, `UnregisterHotKey`
on shutdown. A Bevy system drains it with `try_iter()` and drives the same `OverlayState` phase 1 built.
Show must also raise: `Window::focused = true` (§8). Registration failure (another app owns `~`, or a
UIPI/elevation problem) must surface as a visible message, not a silent no-op.
**Files:** `viewer/src/overlay.rs`, `viewer/Cargo.toml` (the zero-package `windows-sys 0.52` line, §6).

### Phase 3 — coexistence polish
Overlay-aware defaults: nudge `GfxSettings::lod_bias` coarser while the overlay is up (§9); the
exclusive-fullscreen notice and the elevation hint (§5); a settings row for overlay size/position; the
VRAM/CPU-bake notes in the UI (§9). Verify the `WinitSettings` interaction end-to-end with EFT actually
running.
**Files:** `viewer/src/overlay.rs`, `viewer/src/menu.rs`, `viewer/src/render/mod.rs` (read-only use of
existing `GfxSettings`).

### Phase 4 — optional, only if wanted after phases 1-3 ship
Uniform window alpha via `SetLayeredWindowAttributes(LWA_ALPHA)` on the raw HWND, obtained from the
`RawHandleWrapper` component (`bevy_window-0.17.3/src/raw_handle.rs:87-89`, inserted at
`bevy_winit/src/system.rs:96-101`; `raw-window-handle 0.6.2` is already in the lock at `Cargo.lock:4157`).
Slider-controlled, defaulting to fully opaque, never combined with `WS_EX_TRANSPARENT`. **Prototype
first** — layered compositing over a Vulkan swapchain is unverified here, and the failure mode may be a
black or torn window rather than a graceful decline.
**Files:** `viewer/src/overlay.rs`, `viewer/Cargo.toml` (`raw-window-handle = "0.6"`).

---

## 14. What could go wrong for a user

Ordered by how likely they are to hit it.

1. **EFT is in exclusive Fullscreen.** They press `~`, nothing appears, and their WASD now flies a map
   camera they cannot see instead of moving their character — mid-raid. **The worst failure in the
   feature.** Mitigation: the §5 notice, and a strong argument for making the first-run overlay opt-in
   say "set EFT to Borderless" in the same sentence as the risk statement.
2. **The hotkey does not register.** Another app already owns `~` (`RegisterHotKey` is
   first-come-first-served), or EFT's elevation blocks it. Must produce a visible message.
3. **The overlay opens over the game at a bad moment.** Any mid-raid focus switch costs them seconds.
   This is inherent to the feature; the mitigation is that dismissal is one keypress and the summon key
   is deliberate.
4. **VRAM pressure on a large map.** Streets resident in Atlas plus EFT on an 8 GB card may cost the
   game frames or stutter. Unmeasured (§9, `## Known gaps`).
5. **A GPU-driven bake takes the slow CPU path** because the viewer holds the interactive lease
   (`gpu_lease.rs`, `sh_bake.rs:748-752`). Correct behaviour, surprising if undocumented.
6. **Alt-tab leaves a stuttery overlay** at ~2 Hz until they press `~` (§8). Cosmetic.
7. **A future EFT patch removes or perturbs the screenshot-filename coordinates.** The mechanism is an
   undocumented implementation detail. `game_watch.rs:496-498` already bails when fewer than three
   floats parse, so it degrades rather than breaks — keep it that way.
8. **BSG changes its stance on third-party tools.** Clause 4.3.4 gives them the discretion (§7.3), and
   the current position is non-enforcement rather than permission. This is the risk the user must be
   allowed to accept knowingly, which is why the feature ships off by default.
9. **A BattlEye false positive.** No evidenced overlay-caused EFT ban was located, the documented
   failure mode for legitimate software is block/kick/crash rather than ban, and BattlEye states it does
   not ban for passive non-cheating activity (§7.2). But global bans are permanent and appeals succeed
   *"really really rare[ly]"* **[DOC, snippet only]**. The honest framing for the user is: very unlikely,
   not impossible, and the consequence if it happens is severe.

---

## Known gaps

- **Not verified on hardware: the Vulkan composite-alpha capability** on the target adapters. §3's
  conclusion rests on the conventional Windows behaviour of `VK_KHR_win32_surface` reporting
  OPAQUE-only. Confirm by logging `Surface::get_capabilities().alpha_modes` from the existing probe in
  `render::gpu_driven_supported` (`render/mod.rs:279-309`) before anyone attempts transparency. If the
  adapter *does* report `PostMultiplied`, §3's first argument falls — but the creation-only limit and
  the winit DWM-blur path still stand, and §4 still rules out click-through.
- **Not measured: the overlay's VRAM cost alongside a running EFT.** Section 9's claim that the LOD
  knobs are draw-time selectors that do not free residency follows from reading `gpu_driven.rs`'s
  upload path, but the actual footprint per map and the effect on EFT's frame time are unmeasured.
  Measure on Streets before recommending the feature to users on 8 GB cards.
- **Not verified: layered-window alpha over a Vulkan swapchain** (phase 4). Prototype before designing
  a UI around it.
- **Not verified firsthand: the EFT-domain license agreement text.** Clause 4.3.4 was read from the
  Arena mirror (§7.3). Re-verify before quoting it to users.
- **Not verified firsthand:** the peer-reviewed BattlEye technique inventory (Sabt et al., MATE 2025,
  ACM DL returned 403) and BSG's ban-appeal statement (fetch blocked, search snippet only). The RE
  citations in §4 and §7 are independent blog analysis, not vendor documentation, and are labelled
  **[RE]** throughout for that reason.
- **Untested: whether hiding the window reliably returns focus to EFT** in every configuration
  (§8). If it does not, the `SetForegroundWindow` escalation needs its own decision, not a silent
  patch.
- **Unaddressed: multi-monitor.** A user with EFT on one monitor and space on another does not need an
  overlay at all — they need Atlas topmost on the second monitor, which phase 1 already delivers as a
  side effect. Worth stating in the UI so those users skip the risk discussion entirely.

## How to audit this yourself

Every version claim above cites a vendored source path. To re-check after a `cargo update`:

```
rg -n "cannot be changed after the window is created" ~/.cargo/registry/src/*/bevy_winit-*/src/system.rs
rg -n "IGNORE_CURSOR_EVENT" -A2 ~/.cargo/registry/src/*/winit-*/src/platform_impl/windows/window_state.rs
rg -n "UnsupportedAlphaMode" -B12 ~/.cargo/registry/src/*/wgpu-core-*/src/device/global.rs
rg -n "KeyCode::" viewer/src            # confirm Backquote is still free
```

If the first command stops matching, re-read §2 before assuming `transparent` became live-settable.
