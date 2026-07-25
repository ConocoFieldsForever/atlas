//! OVERLAY MODE — summon the map over the running game with one key, dismiss it with the same key.
//!
//! THE ONE ARCHITECTURAL RULE: there is no second window, no second renderer, no second App. This
//! mutates fields on the EXISTING `PrimaryWindow` entity (`decorations` / `window_level` /
//! `position` / `resolution` / `visible`), all of which Bevy applies live. Everything the viewer
//! already draws — the map, the live game link's player fix, the camera standing in the player's
//! eyes — is simply the same render target, re-framed. See docs/OVERLAY_PLAN.md for why the
//! rejected alternatives (a transparent click-through HUD, a second window) are not viable here:
//! per-pixel alpha under the Vulkan-only + `panic = "abort"` policy is an ABORT risk rather than a
//! degraded look, and click-through sets the documented external-ESP window fingerprint.
//!
//! EFT MUST RUN BORDERLESS. No ordinary window can appear over exclusive fullscreen; that is a
//! platform rule, not a bug we can code around. The failure is nasty (the user presses `~`, gets
//! nothing visible, and their WASD is flying a camera they cannot see) so `OverlayConfig` carries
//! the notice the UI shows.
//!
//! HOW IT IS SUMMONED: the player's own IN-GAME SCREENSHOT KEY. EFT writes the position into the
//! screenshot filename, the game-watch thread turns it into a fix, and the fix raises the overlay
//! (game_watch::apply_game_events). We deliberately carry NO global hotkey: `RegisterHotKey`
//! CONSUMES its key machine-wide (the game never sees it), a keyboard hook would observe every
//! keystroke, and `SendInput` is the injection surface anti-cheat watches — the screenshot flow
//! needs none of them. Dismissal is the big BACK TO TARKOV button (or `~` while Atlas has focus),
//! which minimises Atlas so Windows hands the foreground back to the game.

use bevy::prelude::*;
use bevy::window::{MonitorSelection, PrimaryWindow, WindowLevel, WindowPosition};
use bevy::winit::UpdateMode;

/// Everything the overlay's behaviour is configured by — ONE struct so the menu UI can bind to it
/// directly and `atlas.config.json` has one obvious home for these keys. Runtime state (is it up
/// right now, what geometry did we come from) lives in `OverlayState`, deliberately separate: this
/// is the part worth persisting and showing in a settings panel.
///
/// Persisted flat, one key per field, matching the established `config_bool` pattern in menu.rs.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct OverlayConfig {
    /// Master switch. OFF by default: overlaying a game is the user's call to make, not ours
    /// (docs/OVERLAY_PLAN.md §7 — low but non-zero risk, and the residual risk is contractual).
    pub enabled: bool,
    /// Keep the window above the game while the overlay is up. Off = a normal window that the
    /// game will cover when it regains focus (useful on a second monitor).
    pub always_on_top: bool,
    /// Drop the window chrome while the overlay is up.
    pub borderless: bool,
    /// Overlay size as a FRACTION of the primary monitor (0.2..=1.0). A panel, not a takeover —
    /// the point is to read the map while the game is still visible around it.
    pub size_frac: Vec2,
    /// Where the panel sits, as a fraction of the leftover screen space (0,0 = top-left,
    /// 1,1 = bottom-right, 0.5,0.5 = centred).
    pub anchor: Vec2,
    /// Cap the frame rate while the overlay is up so Atlas leaves headroom for the game. 0 = no
    /// cap. This matters: Atlas and EFT share one GPU, and a TDR from contention is what killed
    /// the viewer before (see gpu_lease.rs).
    pub fps_cap: u32,
    /// Stop rendering entirely while the overlay is hidden (the window is not visible anyway).
    /// The single cheapest thing we can do for the game's frame rate.
    pub pause_when_hidden: bool,
    /// Summon the overlay automatically when a SCREENSHOT position fix arrives (RECOMMENDED, and
    /// the cheapest way to get "one key = screenshot + overlay").
    ///
    /// WHY THIS EXISTS INSTEAD OF A PASS-THROUGH HOTKEY: `RegisterHotKey` CONSUMES its key — the
    /// game never sees it, and Windows offers no pass-through flag. Making the game screenshot as
    /// well would need either a low-level keyboard hook (observes every keystroke on the machine)
    /// or synthetic input (SendInput — injecting keys into a game is precisely what anti-cheat
    /// looks for, and docs/OVERLAY_PLAN.md rules it out). Turning the flow around costs neither:
    /// the user presses THEIR OWN in-game screenshot key, EFT handles it natively with nothing
    /// intercepted, and the file EFT writes is our trigger — we already parse it for the position.
    /// One keypress, screenshot taken, overlay up, camera standing in the player's eyes.
    pub show_on_screenshot: bool,
    /// On dismiss, hand the keyboard back to the GAME by minimising Atlas (Windows has no
    /// "focus that other process" call; minimising activates whatever is behind us, which is the
    /// game). Off = Atlas stays on the desktop as a normal window when dismissed.
    pub return_focus_to_game: bool,
    /// Delete each screenshot once we've taken its position fix (RECOMMENDED). EFT writes one
    /// file per press and never cleans up, so locating yourself a few times a raid piles up
    /// full-resolution PNGs forever. Only files we actually CONSUMED are removed — never anything
    /// we didn't parse, and nothing at all while screenshot-locate is off. Lives here because the
    /// menu shows it beside the overlay's other live-link settings.
    pub delete_processed_shots: bool,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            always_on_top: true,
            borderless: true,
            size_frac: Vec2::new(0.55, 0.6),
            anchor: Vec2::new(1.0, 0.0), // top-right: out of the way of most HUDs
            fps_cap: 60,
            pause_when_hidden: true,
            show_on_screenshot: true,
            return_focus_to_game: true,
            delete_processed_shots: true,
        }
    }
}

impl OverlayConfig {
    /// Load from atlas.config.json, falling back to `Default` per field so a partial/older config
    /// still works (same forgiving shape as the other settings readers).
    pub fn load() -> Self {
        let d = Self::default();
        Self {
            enabled: crate::menu::config_bool_pub("overlayEnabled").unwrap_or(d.enabled),
            always_on_top: crate::menu::config_bool_pub("overlayAlwaysOnTop")
                .unwrap_or(d.always_on_top),
            borderless: crate::menu::config_bool_pub("overlayBorderless").unwrap_or(d.borderless),
            size_frac: Vec2::new(
                crate::menu::config_f32_pub("overlayWidthFrac").unwrap_or(d.size_frac.x),
                crate::menu::config_f32_pub("overlayHeightFrac").unwrap_or(d.size_frac.y),
            ),
            anchor: Vec2::new(
                crate::menu::config_f32_pub("overlayAnchorX").unwrap_or(d.anchor.x),
                crate::menu::config_f32_pub("overlayAnchorY").unwrap_or(d.anchor.y),
            ),
            fps_cap: crate::menu::config_f32_pub("overlayFpsCap").unwrap_or(d.fps_cap as f32) as u32,
            pause_when_hidden: crate::menu::config_bool_pub("overlayPauseWhenHidden")
                .unwrap_or(d.pause_when_hidden),
            show_on_screenshot: crate::menu::config_bool_pub("overlayShowOnScreenshot")
                .unwrap_or(d.show_on_screenshot),
            return_focus_to_game: crate::menu::config_bool_pub("overlayReturnFocus")
                .unwrap_or(d.return_focus_to_game),
            delete_processed_shots: crate::menu::config_bool_pub("deleteProcessedShots")
                .unwrap_or(d.delete_processed_shots),
        }
    }

    /// Write every field back. Returns false if the config file could not be written (the caller
    /// surfaces that — a silently unsaved setting is the bug we already fixed once elsewhere).
    #[must_use]
    pub fn save(&self) -> bool {
        let mut ok = crate::menu::save_config_bool_pub("overlayEnabled", self.enabled);
        ok &= crate::menu::save_config_bool_pub("overlayAlwaysOnTop", self.always_on_top);
        ok &= crate::menu::save_config_bool_pub("overlayBorderless", self.borderless);
        ok &= crate::menu::save_config_f32_pub("overlayWidthFrac", self.size_frac.x);
        ok &= crate::menu::save_config_f32_pub("overlayHeightFrac", self.size_frac.y);
        ok &= crate::menu::save_config_f32_pub("overlayAnchorX", self.anchor.x);
        ok &= crate::menu::save_config_f32_pub("overlayAnchorY", self.anchor.y);
        ok &= crate::menu::save_config_f32_pub("overlayFpsCap", self.fps_cap as f32);
        ok &= crate::menu::save_config_bool_pub("overlayPauseWhenHidden", self.pause_when_hidden);
        ok &= crate::menu::save_config_bool_pub("overlayShowOnScreenshot", self.show_on_screenshot);
        ok &= crate::menu::save_config_bool_pub("overlayReturnFocus", self.return_focus_to_game);
        ok &= crate::menu::save_config_bool_pub("deleteProcessedShots", self.delete_processed_shots);
        ok
    }

    /// Clamp anything a hand-edited config could put out of range, so a bad value can't produce a
    /// 1-pixel or off-screen window.
    pub fn sanitized(mut self) -> Self {
        self.size_frac = self.size_frac.clamp(Vec2::splat(0.2), Vec2::splat(1.0));
        self.anchor = self.anchor.clamp(Vec2::ZERO, Vec2::ONE);
        self.fps_cap = self.fps_cap.min(360);
        self
    }
}

/// Live overlay state (not persisted).
#[derive(Resource, Default)]
pub struct OverlayState {
    /// Is the overlay currently summoned?
    pub shown: bool,
    /// Bumped whenever something asks for the overlay to be brought to the FRONT, even if it is
    /// already `shown`. Windows can put us behind (or minimise us) without changing `shown` at
    /// all -- notably when Tarkov takes exclusive fullscreen back after a screenshot -- and
    /// because `apply_overlay` is change-gated on `shown`, nothing ever re-raised the window.
    /// The symptom is the overlay being "open but invisible": state says shown, the window
    /// exists, and the OS is not showing it. Consumers compare against their last seen value.
    pub raise_nonce: u32,
}

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        // EFT_OVERLAY_SUMMON=1 is the baton a menu-mode relaunch passes so the NEW instance comes
        // up with the overlay already showing (screenshot taken at the start menu -> relaunch into
        // the raid map -> panel up, camera already pinned by EFT_POSE). Consumed and REMOVED here
        // so a later PLAY relaunch doesn't inherit a stale summon.
        let summon = std::env::var("EFT_OVERLAY_SUMMON").is_ok_and(|v| v.trim() == "1");
        if summon {
            std::env::remove_var("EFT_OVERLAY_SUMMON");
            // The handoff's EFT_POSE has done its job once `setup` (Startup) read it. Drop it in
            // PostStartup so the camera is free afterwards (main.rs gates on its presence) and a
            // later PLAY relaunch doesn't inherit a stale pose.
            app.add_systems(PostStartup, || std::env::remove_var("EFT_POSE"));
        }
        app.insert_resource(OverlayConfig::load().sanitized())
            .insert_resource(OverlayState { shown: summon, raise_nonce: 0 })
            .add_systems(Update, (toggle_overlay, apply_overlay).chain());
        app.add_systems(bevy_egui::EguiPrimaryContextPass, overlay_return_button);
    }
}

/// `~` toggles the overlay. Gated on the UI not wanting the keyboard, exactly like pick.rs gates
/// Escape, so typing `~` into the search box doesn't summon/dismiss.
fn toggle_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    cfg: Res<OverlayConfig>,
    ui_kb: Option<Res<crate::inspect::UiWantsKeyboard>>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut state: ResMut<OverlayState>,
) {
    if !cfg.enabled || menu.is_some() {
        return; // opt-in only, and never while the start menu owns the screen
    }
    if ui_kb.map(|k| k.0).unwrap_or(false) {
        return;
    }
    if keys.just_pressed(KeyCode::Backquote) {
        state.shown = !state.shown;
    }
}

/// Push `OverlayState` onto the real window. Runs only on a change (the window fields are all
/// live-settable, so this is a handful of writes, never a rebuild).
fn apply_overlay(
    cfg: Res<OverlayConfig>,
    state: Res<OverlayState>,
    mut q: Query<&mut Window, With<PrimaryWindow>>,
    // Primary monitor preferred (the game is almost always there); any monitor as a fallback so
    // the overlay still sizes itself if the primary marker hasn't been spawned yet.
    monitors: Query<&bevy::window::Monitor, With<bevy::window::PrimaryMonitor>>,
    any_monitor: Query<&bevy::window::Monitor>,
    mut winit: ResMut<bevy::winit::WinitSettings>,
    mut last: Local<Option<bool>>,
    mut last_nonce: Local<Option<u32>>,
    mut saved: Local<Option<(WindowPosition, Vec2)>>,
) {
    // Re-run on a shown/hidden transition, on a settings change, OR on an explicit re-raise
    // request (see `OverlayState::raise_nonce`) -- the last one is what makes a second screenshot
    // pull the window back to the front after the game has taken the foreground.
    if *last == Some(state.shown) && !cfg.is_changed() && *last_nonce == Some(state.raise_nonce) {
        return;
    }
    *last_nonce = Some(state.raise_nonce);
    // Whether the overlay was ACTUALLY up before this run. The hide branch below must only touch
    // the window on a real shown->hidden transition: on the first frame (`last` = None) and on
    // every settings tweak (cfg change marks this system dirty) `shown` is simply still false, and
    // treating that as "dismiss" minimised Atlas at startup and whenever a slider moved.
    let was_shown = last.unwrap_or(false);
    *last = Some(state.shown);
    let Ok(mut win) = q.single_mut() else { return };

    if state.shown {
        // Remember the desktop layout ONCE (a config change while shown must not overwrite it).
        if saved.is_none() {
            *saved = Some((win.position, Vec2::new(win.resolution.width(), win.resolution.height())));
        }
        if cfg.borderless {
            win.decorations = false;
        }
        win.window_level = if cfg.always_on_top { WindowLevel::AlwaysOnTop } else { WindowLevel::Normal };
        // Raise + take focus: summoned from a raid the game owns the foreground, so an always-on-top
        // window that never asks for focus would appear without receiving the WASD that follows.
        win.visible = true;
        win.set_minimized(false); // we may have minimised ourselves to hand the game focus back
        win.focused = true;       // ask Windows to raise + give US the keyboard
        // Panel geometry from the monitor size: `size_frac` of it, `anchor` sliding it within the
        // leftover space (0,0 = top-left .. 1,1 = bottom-right). Falls back to centring when the
        // monitor size isn't known yet (first frames), which is never wrong, only less precise.
        if let Some(mon) = monitors.iter().next().or_else(|| any_monitor.iter().next()) {
            let (mw, mh) = (mon.physical_width as f32, mon.physical_height as f32);
            let w = (mw * cfg.size_frac.x).round().max(320.0);
            let h = (mh * cfg.size_frac.y).round().max(240.0);
            win.resolution.set(w, h);
            let x = ((mw - w) * cfg.anchor.x).round() as i32 + mon.physical_position.x;
            let y = ((mh - h) * cfg.anchor.y).round() as i32 + mon.physical_position.y;
            win.position = WindowPosition::At(IVec2::new(x, y));
        } else {
            win.position = WindowPosition::Centered(MonitorSelection::Primary);
        }
        // Leave the GAME headroom: once the user clicks back to EFT we are unfocused, and an
        // unthrottled Atlas would keep rendering the map at full rate on the same GPU. `fps_cap`
        // sets that unfocused rate; focused stays continuous so the overlay itself feels normal.
        winit.focused_mode = UpdateMode::Continuous;
        winit.unfocused_mode = if cfg.fps_cap > 0 {
            UpdateMode::Reactive {
                wait: std::time::Duration::from_secs_f32(1.0 / cfg.fps_cap as f32),
                react_to_device_events: false,
                react_to_user_events: true,
                react_to_window_events: true,
            }
        } else {
            UpdateMode::Continuous
        };
        info!(
            "overlay: shown (borderless={}, always_on_top={}, {:.0}%x{:.0}% of monitor, unfocused cap {} fps)",
            cfg.borderless,
            cfg.always_on_top,
            cfg.size_frac.x * 100.0,
            cfg.size_frac.y * 100.0,
            cfg.fps_cap
        );
    } else {
        // Overlay OFF entirely: behave like a stock desktop app. Restoring Continuous here (this
        // system only runs on change) undoes any throttle a previous enable left behind, and a
        // user who never opted in never sees their unfocused frame rate touched.
        if !cfg.enabled {
            winit.focused_mode = UpdateMode::Continuous;
            winit.unfocused_mode = UpdateMode::Continuous;
            return;
        }
        // Window mutations only on a REAL dismiss (see `was_shown` above) — never at startup and
        // never because a settings checkbox redrew us.
        if was_shown {
            win.window_level = WindowLevel::Normal;
            // Always restore chrome + desktop geometry first, so if the user brings Atlas back
            // from the taskbar by hand they get a normal draggable window, not a chromeless
            // overlay-shaped slab.
            win.decorations = true;
            if let Some((pos, res)) = saved.take() {
                win.position = pos;
                win.resolution.set(res.x, res.y);
            }
            if cfg.return_focus_to_game {
                // GIVE THE GAME THE KEYBOARD BACK. Windows won't let an app hand focus to another
                // process directly, but MINIMISING ourselves makes the OS activate whatever is
                // behind us -- which, summoned from a raid, is Tarkov. Without this the user
                // dismisses the overlay and their WASD still goes nowhere.
                win.set_minimized(true);
            }
            info!(
                "overlay: hidden (desktop window restored, focus to game={}, unfocused idle={})",
                cfg.return_focus_to_game, cfg.pause_when_hidden
            );
        }
        // Hidden but armed: idle hard when unfocused so a dismissed overlay costs the game nothing.
        winit.focused_mode = UpdateMode::Continuous;
        winit.unfocused_mode = if cfg.pause_when_hidden {
            UpdateMode::Reactive {
                wait: std::time::Duration::from_millis(500),
                react_to_device_events: false,
                react_to_user_events: true,
                react_to_window_events: true,
            }
        } else {
            UpdateMode::Continuous
        };
    }
}

/// The BIG way back. While the overlay is up, one oversized button sits bottom-centre: click it
/// and Atlas dismisses + minimises, so Windows hands the foreground (and the keyboard) straight
/// back to Tarkov. It exists because the polite ways out (`~`, Alt-Tab) all assume the user
/// remembers a binding mid-raid; a fat labelled button assumes nothing.
fn overlay_return_button(
    mut contexts: bevy_egui::EguiContexts,
    menu: Option<Res<crate::menu::MenuState>>,
    lang: Res<crate::i18n::Lang>,
    mut state: ResMut<OverlayState>,
) {
    use crate::i18n::{t, K};
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};

    if !state.shown || menu.is_some() {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let lg = *lang;
    egui::Area::new(egui::Id::new("overlay_return_btn"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -16.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                let btn = egui::Button::new(
                    RichText::new(t(lg, K::BackToTarkov))
                        .size(17.0)
                        .strong()
                        .color(theme::TEXT_BRIGHT),
                )
                .fill(theme::DANGER)
                .corner_radius(6.0)
                .min_size(egui::vec2(280.0, 46.0));
                if ui.add(btn).clicked() {
                    state.shown = false; // apply_overlay minimises + returns focus per config
                }
                ui.add_space(2.0);
                ui.label(RichText::new(t(lg, K::OverlayReopenHint)).size(10.0).color(theme::MUTED));
            });
        });
}
