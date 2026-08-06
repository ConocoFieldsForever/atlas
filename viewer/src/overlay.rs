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
//! EFT MUST NOT BE IN EXCLUSIVE FULLSCREEN. No ordinary window can appear over it; that is a
//! platform rule, not a bug we can code around. WINDOWED and BORDERLESS both work: the panel is
//! placed against the game's CLIENT rect, so it centres on the picture in either mode (in windowed
//! the frame's title bar and borders are excluded, which a window-rect placement got wrong). The
//! exclusive-fullscreen failure is nasty — the user presses `~`, sees nothing, and their WASD is
//! flying a camera they cannot see — so `OverlayConfig` carries the notice the UI shows.
//!
//! HOW IT IS SUMMONED: the player's own IN-GAME SCREENSHOT KEY. EFT writes the position into the
//! screenshot filename, the game-watch thread turns it into a fix, and the fix raises the overlay
//! (game_watch::apply_game_events). We deliberately carry NO global hotkey: `RegisterHotKey`
//! CONSUMES its key machine-wide (the game never sees it), a keyboard hook would observe every
//! keystroke, and `SendInput` is the injection surface anti-cheat watches — the screenshot flow
//! needs none of them. Dismissal is the big BACK TO TARKOV button (or `~` while Atlas has focus),
//! which minimises Atlas so Windows hands the foreground back to the game.

use bevy::prelude::*;
use bevy::camera::SubCameraView;
use bevy::window::{MonitorSelection, PrimaryWindow, WindowLevel, WindowPosition};
use bevy::winit::UpdateMode;

/// Where the overlay should place itself: the GAME's window rect when we can see it, else the
/// monitor. `(origin, size)` in physical desktop pixels.
pub(crate) type TargetRect = (IVec2, Vec2);

/// The part of Tarkov's full camera image covered by the Atlas window. Bevy turns this into an
/// asymmetric perspective frustum, so every visible 3D pixel is projected at the same screen
/// coordinate it occupied in the game instead of treating the smaller overlay as a new full view.
///
/// Retained while hidden because Tarkov may be briefly iconic during the next screenshot/focus
/// handoff; `OverlayState::shown` decides whether it is active.
#[derive(Resource, Default)]
struct OverlayViewSlice(Option<SubCameraView>);

fn view_slice(
    game_origin: IVec2,
    game_size: Vec2,
    overlay_origin: IVec2,
    overlay_size: UVec2,
) -> Option<SubCameraView> {
    let full_size = game_size.round().as_uvec2();
    if full_size.x == 0 || full_size.y == 0 || overlay_size.x == 0 || overlay_size.y == 0 {
        return None;
    }
    Some(SubCameraView {
        full_size,
        // Desktop/window coordinates grow downward, matching SubCameraView's offset convention.
        offset: (overlay_origin - game_origin).as_vec2(),
        size: overlay_size,
    })
}

/// Unity titles a game's window with its PRODUCT NAME, and EFT's is recorded first-party in
/// `EscapeFromTarkov_Data/app.info` (line 1 company, line 2 product): `EscapeFromTarkov`.
/// Case-insensitive and space-insensitive so a spaced or differently-cased variant still matches,
/// but anchored at the start so an unrelated window that merely mentions the game — a browser tab,
/// a wiki page, our own title bar — can never be mistaken for it.
fn title_is_game(title: &str) -> bool {
    let norm = title.trim().to_ascii_lowercase().replace(' ', "");
    let Some(rest) = norm.strip_prefix("escapefromtarkov") else {
        return false;
    };
    // The token must END the title or be followed by a separator. A bare `starts_with` also
    // accepted "Escape From Tarkov Wiki - Chrome" (spaces removed: "escapefromtarkovwiki..."),
    // which would have parked the overlay on a browser window; Unity's own suffixes are
    // punctuation-led (" - Direct3D 11"), so they still match.
    rest.is_empty() || !rest.starts_with(|c: char| c.is_ascii_alphanumeric())
}

/// Locate EFT's top-level window so the overlay can centre itself ON THE GAME rather than on the
/// monitor. Returns its physical rect, or `None` when the game isn't up.
///
/// This is a WINDOW-MANAGER query, deliberately not a process one: it enumerates top-level windows
/// and reads a title and a rectangle. It opens no process handle, reads no game memory, installs no
/// hook and synthesises no input — so it stays on the right side of the line this module and
/// `game_watch` draw (see the module header: the rejected techniques are click-through, per-pixel
/// alpha, `RegisterHotKey`, keyboard hooks and `SendInput`). Any window manager, capture tool or
/// screen recorder makes exactly these calls.
///
/// Declared against user32 directly rather than pulling in the `windows` crate: two functions and a
/// callback do not justify a dependency, and this file is the only caller.
#[cfg(windows)]
pub(crate) fn game_window_rect() -> Option<TargetRect> {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct Point {
        x: i32,
        y: i32,
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn EnumWindows(cb: extern "system" fn(*mut c_void, isize) -> i32, param: isize) -> i32;
        fn GetWindowTextW(hwnd: *mut c_void, buf: *mut u16, len: i32) -> i32;
        fn GetWindowThreadProcessId(hwnd: *mut c_void, pid: *mut u32) -> u32;
        fn GetClientRect(hwnd: *mut c_void, rect: *mut Rect) -> i32;
        fn ClientToScreen(hwnd: *mut c_void, pt: *mut Point) -> i32;
        fn IsWindowVisible(hwnd: *mut c_void) -> i32;
        fn IsIconic(hwnd: *mut c_void) -> i32;
    }

    /// Collected by the callback: the first visible window whose title matches the game.
    struct Found(Option<Rect>);

    extern "system" fn enum_cb(hwnd: *mut c_void, param: isize) -> i32 {
        // SAFETY: `param` is the &mut Found we passed to EnumWindows, valid for the whole call.
        let found = unsafe { &mut *(param as *mut Found) };
        if found.0.is_some() {
            return 0; // stop enumerating
        }
        unsafe {
            // A minimised game is not something to centre on — fall through to the monitor.
            if IsWindowVisible(hwnd) == 0 || IsIconic(hwnd) != 0 {
                return 1;
            }
            // NEVER read the title of one of our OWN windows. They cannot be the game, and
            // `GetWindowTextW` is only cached for OTHER processes -- for the calling process it
            // SendMessages WM_GETTEXT, which on Atlas's own (same-thread) window dispatches
            // reentrantly into winit's wndproc from inside `app.update()` and deadlocks the main
            // thread solid: zero CPU, Responding=false, frozen at the summon. The bug hid for as
            // long as summons only ever happened while the GAME was foreground, because
            // EnumWindows walks in Z-order and the search stopped at the game before reaching us;
            // the first summon that ran AFTER Atlas took focus put Atlas on top of the Z-order
            // and enumerated it first.
            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, &mut pid);
            if pid == std::process::id() {
                return 1;
            }
            let mut buf = [0u16; 256];
            let n = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
            if n <= 0 {
                return 1;
            }
            let title = String::from_utf16_lossy(&buf[..n as usize]);
            if title_is_game(&title) {
                // CLIENT area, not the window frame. In WINDOWED mode GetWindowRect includes the
                // title bar and borders, so centring on it puts the panel a title-bar's height too
                // high and off-centre by the border widths — visibly wrong against the game's
                // picture. GetClientRect gives the render area; ClientToScreen puts its origin on
                // the desktop. Borderless collapses to the same rect, so one path covers both.
                let mut c = Rect::default();
                if GetClientRect(hwnd, &mut c) == 0 {
                    return 1;
                }
                let mut origin = Point { x: 0, y: 0 };
                if ClientToScreen(hwnd, &mut origin) == 0 {
                    return 1;
                }
                found.0 = Some(Rect {
                    left: origin.x,
                    top: origin.y,
                    right: origin.x + (c.right - c.left),
                    bottom: origin.y + (c.bottom - c.top),
                });
                return 0;
            }
        }
        1
    }

    let mut found = Found(None);
    unsafe { EnumWindows(enum_cb, &mut found as *mut Found as isize) };
    let r = found.0?;
    let (w, h) = ((r.right - r.left) as f32, (r.bottom - r.top) as f32);
    // A minimised window reports a degenerate/off-screen rect; treat anything implausible as "no
    // game window" so we fall back to the monitor instead of placing the overlay off-screen.
    if w < 320.0 || h < 240.0 {
        return None;
    }
    Some((IVec2::new(r.left, r.top), Vec2::new(w, h)))
}

#[cfg(not(windows))]
pub(crate) fn game_window_rect() -> Option<TargetRect> {
    None
}

/// True while a Direct3D EXCLUSIVE-fullscreen application owns the screen.
///
/// This is the one state in which the overlay CANNOT work: an exclusive-fullscreen swapchain owns
/// the display, so no always-on-top window composites over it. Raising anyway is actively harmful —
/// the game loses keyboard input or minimises outright (the classic exclusive-fullscreen reaction
/// to losing the foreground) while the user sees nothing appear, mid-raid.
///
/// `SHQueryUserNotificationState` is the documented query for exactly this; QUNS_RUNNING_D3D_FULL_SCREEN
/// (3) is the D3D-exclusive state. Borderless-windowed fullscreen — where the overlay works fine —
/// reports QUNS_ACCEPTS_NOTIFICATIONS instead, so this does not fire there. Any failure returns
/// false (assume we may raise): a detection glitch must not silently disable the overlay.
#[cfg(windows)]
fn d3d_exclusive_fullscreen() -> bool {
    #[link(name = "shell32")]
    unsafe extern "system" {
        fn SHQueryUserNotificationState(state: *mut i32) -> i32;
    }
    const QUNS_RUNNING_D3D_FULL_SCREEN: i32 = 3;
    let mut state = 0i32;
    // SAFETY: out-param is a valid i32 for the whole call; S_OK (0) means `state` was written.
    let hr = unsafe { SHQueryUserNotificationState(&mut state) };
    hr == 0 && state == QUNS_RUNNING_D3D_FULL_SCREEN
}

#[cfg(not(windows))]
fn d3d_exclusive_fullscreen() -> bool {
    false
}

/// Flash this process's taskbar button — the honest alternative to stealing focus from a game we
/// cannot draw over. The user sees Atlas asking for attention and alt-tabs when they choose to.
#[cfg(windows)]
fn flash_atlas_taskbar() {
    use std::ffi::c_void;
    #[link(name = "user32")]
    unsafe extern "system" {
        fn EnumWindows(cb: extern "system" fn(*mut c_void, isize) -> i32, param: isize) -> i32;
        fn GetWindowThreadProcessId(hwnd: *mut c_void, pid: *mut u32) -> u32;
        fn IsWindowVisible(hwnd: *mut c_void) -> i32;
        fn FlashWindowEx(info: *mut FlashInfo) -> i32;
    }
    #[repr(C)]
    struct FlashInfo {
        cb_size: u32,
        hwnd: *mut c_void,
        flags: u32,
        count: u32,
        timeout: u32,
    }
    struct Found(*mut c_void);
    extern "system" fn cb(hwnd: *mut c_void, param: isize) -> i32 {
        let found = unsafe { &mut *(param as *mut Found) };
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(hwnd, &mut pid) };
        if pid == std::process::id() && unsafe { IsWindowVisible(hwnd) } != 0 {
            found.0 = hwnd;
            return 0;
        }
        1
    }
    let mut found = Found(std::ptr::null_mut());
    unsafe { EnumWindows(cb, &mut found as *mut Found as isize) };
    if found.0.is_null() {
        return;
    }
    const FLASHW_TRAY: u32 = 0x2;
    const FLASHW_TIMERNOFG: u32 = 0xC; // flash until the window comes to the foreground
    let mut info = FlashInfo {
        cb_size: std::mem::size_of::<FlashInfo>() as u32,
        hwnd: found.0,
        flags: FLASHW_TRAY | FLASHW_TIMERNOFG,
        count: 0,
        timeout: 0,
    };
    // SAFETY: `info` is a correctly sized FLASHWINFO for a window we just enumerated.
    unsafe { FlashWindowEx(&mut info) };
}

#[cfg(not(windows))]
fn flash_atlas_taskbar() {}

/// Bring this process's real top-level window to the foreground and VERIFY that Windows granted
/// keyboard focus.
///
/// Do not route this through Bevy/Winit's `Window::focused` write on Windows. Winit 0.30 implements
/// its force-focus fallback by synthesizing an Alt keypress with `SendInput`; this project
/// deliberately never injects input around the game. This uses only Windows activation APIs,
/// targets Atlas's own HWND, and retries across the screenshot/fullscreen transition.
#[cfg(windows)]
fn request_atlas_focus() -> bool {
    use std::ffi::c_void;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn EnumWindows(cb: extern "system" fn(*mut c_void, isize) -> i32, param: isize) -> i32;
        fn GetWindowThreadProcessId(hwnd: *mut c_void, pid: *mut u32) -> u32;
        fn IsWindowVisible(hwnd: *mut c_void) -> i32;
        fn GetForegroundWindow() -> *mut c_void;
        fn BringWindowToTop(hwnd: *mut c_void) -> i32;
        fn ShowWindowAsync(hwnd: *mut c_void, command: i32) -> i32;
        fn SetForegroundWindow(hwnd: *mut c_void) -> i32;
        fn AttachThreadInput(id_attach: u32, id_attach_to: u32, attach: i32) -> i32;
        fn PeekMessageW(
            message: *mut NativeMessage,
            hwnd: *mut c_void,
            min: u32,
            max: u32,
            remove: u32,
        ) -> i32;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThreadId() -> u32;
    }

    #[repr(C)]
    struct NativeMessage {
        hwnd: *mut c_void,
        message: u32,
        w_param: usize,
        l_param: isize,
        time: u32,
        point_x: i32,
        point_y: i32,
        private: u32,
    }

    struct Found(*mut c_void);
    extern "system" fn enum_cb(hwnd: *mut c_void, param: isize) -> i32 {
        let found = unsafe { &mut *(param as *mut Found) };
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(hwnd, &mut pid) };
        if pid == std::process::id() && unsafe { IsWindowVisible(hwnd) } != 0 {
            found.0 = hwnd;
            return 0;
        }
        1
    }

    let mut found = Found(std::ptr::null_mut());
    unsafe { EnumWindows(enum_cb, &mut found as *mut Found as isize) };
    let hwnd = found.0;
    if hwnd.is_null() {
        return false;
    }
    unsafe {
        let foreground = GetForegroundWindow();
        if foreground == hwnd {
            return true;
        }
        // Activate away from Atlas's event loop. The helper briefly joins the foreground input
        // queue, activates ONLY Atlas's HWND, and detaches. Atlas's own event queue is never joined
        // or blocked, so it remains free to process the resulting activation messages.
        use std::sync::atomic::{AtomicBool, Ordering};
        static ACTIVATION_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
        if !ACTIVATION_IN_FLIGHT.swap(true, Ordering::AcqRel) {
            let hwnd_value = hwnd as usize;
            std::thread::spawn(move || {
                let hwnd = hwnd_value as *mut c_void;
                let foreground = GetForegroundWindow();
                let helper_thread = GetCurrentThreadId();
                let foreground_thread =
                    GetWindowThreadProcessId(foreground, std::ptr::null_mut());

                // AttachThreadInput requires the caller to own a message queue. A no-remove peek
                // creates it without consuming any message or key.
                let mut message: NativeMessage = std::mem::zeroed();
                PeekMessageW(
                    &mut message,
                    std::ptr::null_mut(),
                    0,
                    0,
                    0,
                );
                let attached_foreground = foreground_thread != 0
                    && foreground_thread != helper_thread
                    && AttachThreadInput(helper_thread, foreground_thread, 1) != 0;

                const SW_RESTORE: i32 = 9;
                ShowWindowAsync(hwnd, SW_RESTORE);
                SetForegroundWindow(hwnd);
                BringWindowToTop(hwnd);

                if attached_foreground {
                    AttachThreadInput(helper_thread, foreground_thread, 0);
                }
                // Rate-limit retries if the foreground changes during the handoff.
                std::thread::sleep(std::time::Duration::from_millis(50));
                ACTIVATION_IN_FLIGHT.store(false, Ordering::Release);
            });
        }
        false
    }
}

#[cfg(not(windows))]
fn request_atlas_focus() -> bool {
    false
}

/// User-selectable key that restores Atlas's ordinary desktop window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayExitHotkey {
    NumpadEnter,
    Enter,
    Escape,
    F10,
    F11,
    F12,
}

impl OverlayExitHotkey {
    pub const ALL: [Self; 6] = [
        Self::NumpadEnter,
        Self::Enter,
        Self::Escape,
        Self::F10,
        Self::F11,
        Self::F12,
    ];

    pub fn config_value(self) -> &'static str {
        match self {
            Self::NumpadEnter => "numpad_enter",
            Self::Enter => "enter",
            Self::Escape => "escape",
            Self::F10 => "f10",
            Self::F11 => "f11",
            Self::F12 => "f12",
        }
    }

    pub fn from_config(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "numpad_enter" | "numpadenter" | "right_enter" => Some(Self::NumpadEnter),
            "enter" => Some(Self::Enter),
            "escape" | "esc" => Some(Self::Escape),
            "f10" => Some(Self::F10),
            "f11" => Some(Self::F11),
            "f12" => Some(Self::F12),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::NumpadEnter => "Right Enter (numpad)",
            Self::Enter => "Enter",
            Self::Escape => "Escape",
            Self::F10 => "F10",
            Self::F11 => "F11",
            Self::F12 => "F12",
        }
    }

    fn key_code(self) -> KeyCode {
        match self {
            Self::NumpadEnter => KeyCode::NumpadEnter,
            Self::Enter => KeyCode::Enter,
            Self::Escape => KeyCode::Escape,
            Self::F10 => KeyCode::F10,
            Self::F11 => KeyCode::F11,
            Self::F12 => KeyCode::F12,
        }
    }
}

/// Everything the overlay's behaviour is configured by — ONE struct so the menu UI can bind to it
/// directly and `atlas.config.json` has one obvious home for these keys. Runtime state (is it up
/// right now, what geometry did we come from) lives in `OverlayState`, deliberately separate: this
/// is the part worth persisting and showing in a settings panel.
///
/// Persisted flat, one key per field, matching the established config helper pattern in menu.rs.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct OverlayConfig {
    /// Master switch. OFF by default: overlaying a game is the user's call to make, not ours
    /// (docs/OVERLAY_PLAN.md §7 — low but non-zero risk, and the residual risk is contractual).
    pub enabled: bool,
    /// How the overlay window presents itself. Replaces the independent `always_on_top` +
    /// `borderless` booleans; see [`OverlayPresentation`] for why they had to be collapsed.
    pub presentation: OverlayPresentation,
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
    /// DISMISS the overlay: hide/minimize and (with `return_focus_to_game`) hand the keyboard back
    /// to the game, WITHOUT leaving overlay mode — screenshot summon stays armed and the next fix
    /// brings the panel back where it was. Configurable in the menu's overlay settings (the
    /// "Hide-overlay hotkey" picker). Leaving overlay mode entirely is the on-overlay EXIT button,
    /// not a key: it used to be this hotkey, which read as the overlay breaking out into a desktop
    /// window every time you tried to get back to the game.
    pub exit_hotkey: OverlayExitHotkey,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            presentation: OverlayPresentation::Borderless,
            size_frac: Vec2::new(0.55, 0.6),
            // CENTRED over the game window. The old top-right default put the panel under the
            // stamina/hydration cluster on wide monitors, and on a multi-monitor desk it anchored
            // to the primary screen rather than the one EFT was on. Centre is where the eye already
            // is when you summon a map, and `anchor` still moves it if you want a corner.
            anchor: Vec2::splat(0.5),
            fps_cap: 60,
            pause_when_hidden: true,
            show_on_screenshot: true,
            return_focus_to_game: true,
            delete_processed_shots: true,
            exit_hotkey: OverlayExitHotkey::NumpadEnter,
        }
    }
}

/// The anchor the overlay shipped with before it centred on the game window.
const LEGACY_ANCHOR: Vec2 = Vec2::new(1.0, 0.0); // top-right

/// Read the persisted anchor, migrating anyone still sitting on the OLD default exactly once.
///
/// Changing a `Default` does nothing for an existing install: `atlas.config.json` already holds
/// `overlayAnchorX/Y`, so every current user kept getting the panel jammed against the right edge
/// no matter what the code's default said. A value that is byte-for-byte the old default was never
/// a choice — it is just what the app wrote out — so it moves to centre. Anything else is a real
/// preference and is left alone, and the one-shot flag means a user who deliberately picks
/// top-right afterwards keeps it.
fn load_anchor(default: Vec2) -> Vec2 {
    let stored = Vec2::new(
        crate::menu::config_f32_pub("overlayAnchorX").unwrap_or(default.x),
        crate::menu::config_f32_pub("overlayAnchorY").unwrap_or(default.y),
    );
    if crate::menu::config_bool_pub("overlayAnchorMigrated").unwrap_or(false) {
        return stored;
    }
    let _ = crate::menu::save_config_bool_pub("overlayAnchorMigrated", true);
    if (stored - LEGACY_ANCHOR).abs().max_element() < 1e-3 {
        let _ = crate::menu::save_config_f32_pub("overlayAnchorX", default.x);
        let _ = crate::menu::save_config_f32_pub("overlayAnchorY", default.y);
        info!("overlay: anchor migrated from the old top-right default to centre-on-game");
        return default;
    }
    stored
}

/// How the overlay window presents itself over the game.
///
/// WHY THIS IS ONE CHOICE AND NOT THREE CHECKBOXES. Transparency is not an independent flag that
/// can be ORed onto the others: DWM decides whether a window composites per-pixel at
/// `CreateWindowEx`, and measurement says it only does so for a specific CONJUNCTION of creation
/// attributes. Same app, same alpha mode, same backdrop, only the creation flags differing:
///
/// | created as                          | result                          |
/// |-------------------------------------|---------------------------------|
/// | decorated + resizable               | `(255,255,255)` blended to white|
/// | undecorated + resizable             | `(53,53,53)` opaque             |
/// | undecorated + fixed + normal z       | `(53,53,53)` opaque             |
/// | undecorated + fixed + always-on-top | `(41,255,41)` CORRECT           |
///
/// As three booleans that is 8 combinations, exactly one of which is transparent, and the other
/// seven fail SILENTLY -- no error, no log, no crash, just an opaque window. Worse, the setting
/// cannot be honoured after the fact: a window created decorated and later set `decorations=false`
/// stays blended-to-white forever, which is precisely what the old summon-time code did on every
/// summon. Encoding the valid conjunctions as named modes makes the broken combinations
/// unrepresentable rather than merely documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayPresentation {
    /// Ordinary decorated window at normal z-order. The game covers it when it takes focus, which
    /// is what you want on a second monitor.
    Windowed,
    /// Undecorated panel held above the game, opaque. The default, and what every existing install
    /// has been running.
    Borderless,
    /// Undecorated, fixed-size, always-on-top window that composites per-pixel, so the game shows
    /// through wherever Atlas draws nothing. Must be chosen BEFORE the window exists, which is why
    /// it applies at the next map launch rather than at the next summon.
    Transparent,
}

impl OverlayPresentation {
    /// Config token. Stored as a string so a future mode does not have to fit a bool.
    pub fn config_value(self) -> &'static str {
        match self {
            Self::Windowed => "windowed",
            Self::Borderless => "borderless",
            Self::Transparent => "transparent",
        }
    }

    fn from_config(s: &str) -> Option<Self> {
        match s.trim() {
            "windowed" => Some(Self::Windowed),
            "borderless" => Some(Self::Borderless),
            "transparent" => Some(Self::Transparent),
            _ => None,
        }
    }

    /// Window chrome is dropped.
    pub fn borderless(self) -> bool {
        !matches!(self, Self::Windowed)
    }

    /// Held above the game.
    pub fn always_on_top(self) -> bool {
        matches!(self, Self::Borderless | Self::Transparent)
    }

    /// Per-pixel alpha, and therefore launch-gated.
    pub fn transparent(self) -> bool {
        matches!(self, Self::Transparent)
    }
}

impl OverlayConfig {
    /// Whether the overlay machinery is ON for this session. The master `enabled` switch is the
    /// consent gate for the opaque panel; a TRANSPARENT launch is that consent given a different
    /// way, so either one engages. ONE method, because the last time this predicate lived in two
    /// places they drifted: `apply_overlay` learned the transparent-launch clause and
    /// `apply_overlay_view_slice` did not, so a transparent session with the master switch off
    /// summoned the window but never published its view slice -- the panel lens-shift won the
    /// camera instead and the whole scene sat offset from the game's picture. (Same disease as
    /// the SSAO/EftPrepassLabel crash: two sites that must agree, with nothing forcing them to.)
    pub fn engaged(&self, transparent_launch: bool) -> bool {
        self.enabled || (transparent_launch && !crate::automated_finite_job())
    }

    /// Load from atlas.config.json, falling back to `Default` per field so a partial/older config
    /// still works (same forgiving shape as the other settings readers).
    pub fn load() -> Self {
        let d = Self::default();
        Self {
            enabled: crate::menu::config_bool_pub("overlayEnabled").unwrap_or(d.enabled),
            // MIGRATION: `overlayPresentation` is the key now, but installs predating it only have
            // the two booleans. Read the new key first and derive from the old pair when it is
            // absent, so an existing user's window does not silently change shape on upgrade.
            // Neither old value could mean Transparent, so no migration can invent it -- the mode
            // is only ever reachable by asking for it.
            presentation: crate::menu::config_str_pub("overlayPresentation")
                .as_deref()
                .and_then(OverlayPresentation::from_config)
                .unwrap_or_else(|| {
                    match crate::menu::config_bool_pub("overlayBorderless") {
                        Some(false) => OverlayPresentation::Windowed,
                        Some(true) => OverlayPresentation::Borderless,
                        None => d.presentation,
                    }
                }),
            size_frac: Vec2::new(
                crate::menu::config_f32_pub("overlayWidthFrac").unwrap_or(d.size_frac.x),
                crate::menu::config_f32_pub("overlayHeightFrac").unwrap_or(d.size_frac.y),
            ),
            anchor: load_anchor(d.anchor),
            fps_cap: crate::menu::config_f32_pub("overlayFpsCap").unwrap_or(d.fps_cap as f32) as u32,
            pause_when_hidden: crate::menu::config_bool_pub("overlayPauseWhenHidden")
                .unwrap_or(d.pause_when_hidden),
            show_on_screenshot: crate::menu::config_bool_pub("overlayShowOnScreenshot")
                .unwrap_or(d.show_on_screenshot),
            return_focus_to_game: crate::menu::config_bool_pub("overlayReturnFocus")
                .unwrap_or(d.return_focus_to_game),
            delete_processed_shots: crate::menu::config_bool_pub("deleteProcessedShots")
                .unwrap_or(d.delete_processed_shots),
            exit_hotkey: crate::menu::config_str_pub("overlayExitHotkey")
                .as_deref()
                .and_then(OverlayExitHotkey::from_config)
                .unwrap_or(d.exit_hotkey),
        }
    }

    /// Write every field back. Returns false if the config file could not be written (the caller
    /// surfaces that — a silently unsaved setting is the bug we already fixed once elsewhere).
    #[must_use]
    pub fn save(&self) -> bool {
        let mut ok = crate::menu::save_config_bool_pub("overlayEnabled", self.enabled);
        ok &= crate::menu::save_config_str_pub(
            "overlayPresentation",
            self.presentation.config_value(),
        );
        // The superseded booleans are still written, and deliberately so: a user who downgrades
        // to an older build would otherwise land on its `borderless: true` default regardless of
        // what they had chosen. Writing both keeps the two representations agreeing in the one
        // direction that can be known.
        ok &= crate::menu::save_config_bool_pub(
            "overlayAlwaysOnTop",
            self.presentation.always_on_top(),
        );
        ok &= crate::menu::save_config_bool_pub(
            "overlayBorderless",
            self.presentation.borderless(),
        );
        ok &= crate::menu::save_config_f32_pub("overlayWidthFrac", self.size_frac.x);
        ok &= crate::menu::save_config_f32_pub("overlayHeightFrac", self.size_frac.y);
        ok &= crate::menu::save_config_f32_pub("overlayAnchorX", self.anchor.x);
        ok &= crate::menu::save_config_f32_pub("overlayAnchorY", self.anchor.y);
        ok &= crate::menu::save_config_f32_pub("overlayFpsCap", self.fps_cap as f32);
        ok &= crate::menu::save_config_bool_pub("overlayPauseWhenHidden", self.pause_when_hidden);
        ok &= crate::menu::save_config_bool_pub("overlayShowOnScreenshot", self.show_on_screenshot);
        ok &= crate::menu::save_config_bool_pub("overlayReturnFocus", self.return_focus_to_game);
        ok &= crate::menu::save_config_bool_pub("deleteProcessedShots", self.delete_processed_shots);
        ok &= crate::menu::save_config_str_pub(
            "overlayExitHotkey",
            self.exit_hotkey.config_value(),
        );
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

/// Winit's reactive update mode is the renderer-side overlay frame limiter. Keep this in one
/// helper so focused and unfocused windows cannot accidentally drift back to different policies.
/// Device events (raw mouse motion) do not wake an extra frame; ordinary keyboard/window events
/// still do, which keeps input responsive while the timer supplies the steady redraw cadence.
fn overlay_update_mode(fps_cap: u32) -> UpdateMode {
    if fps_cap > 0 {
        UpdateMode::Reactive {
            wait: std::time::Duration::from_secs_f32(1.0 / fps_cap as f32),
            react_to_device_events: false,
            react_to_user_events: true,
            react_to_window_events: true,
        }
    } else {
        UpdateMode::Continuous
    }
}

/// Live overlay state (not persisted).
#[derive(Resource, Default)]
pub struct OverlayState {
    /// Is the overlay currently summoned?
    pub shown: bool,
    /// The user explicitly left overlay presentation but kept Atlas open as an ordinary decorated,
    /// resizable window. Screenshot summon remains armed and clears this flag next time.
    pub windowed: bool,
    /// Bumped whenever something asks for the overlay to be brought to the FRONT, even if it is
    /// already `shown`. Windows can put us behind (or minimise us) without changing `shown` at
    /// all -- notably when Tarkov takes exclusive fullscreen back after a screenshot -- and
    /// because `apply_overlay` is change-gated on `shown`, nothing ever re-raised the window.
    /// The symptom is the overlay being "open but invisible": state says shown, the window
    /// exists, and the OS is not showing it. Consumers compare against their last seen value.
    pub raise_nonce: u32,
}

/// True while the overlay is presenting OVER THE GAME: an ESP or transparent session,
/// summoned, not exited to a window. The map-session furniture (tab rail, side panels, the pick
/// hint) reads THIS ONE RESOURCE instead of recomputing the predicate, because twice today a
/// second copy of an overlay predicate drifted from the first (`engaged` in the view slice; the
/// `cfg.enabled` summon gate) and each time the failure was silent misbehaviour in the field.
/// A raid overlay wants the game visible, not a settings workstation: what stays is what earns
/// its pixels over live play -- the labels, the position HUD, the link banners, and the way back.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
pub struct OverlayFocus(pub bool);

/// Sole writer of [`OverlayFocus`]. Runs before `apply_overlay` in the chain so panels and the
/// window agree within a frame.
fn update_overlay_focus(
    cfg: Res<OverlayConfig>,
    state: Res<OverlayState>,
    esp: Res<crate::EspMode>,
    transparent: Res<crate::TransparentWindow>,
    mut focus: ResMut<OverlayFocus>,
) {
    let v = (esp.0 || transparent.0)
        && cfg.engaged(transparent.0)
        && state.shown
        && !state.windowed;
    if focus.0 != v {
        focus.0 = v;
    }
}

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        // EFT_OVERLAY_SUMMON=1 is the baton a menu-mode relaunch passes so the NEW instance comes
        // up with the overlay already showing (screenshot taken at the start menu -> relaunch into
        // the raid map -> panel up, camera already pinned by EFT_POSE). Consumed and REMOVED here
        // so a later PLAY relaunch doesn't inherit a stale summon.
        let summon = std::env::var("EFT_OVERLAY_SUMMON").is_ok_and(|v| v.trim() == "1");
        // A transparent launch IS the overlay: the user chose a window that only exists to sit
        // over the game, so it comes up summoned instead of waiting for a screenshot that the
        // master `enabled` switch may never have armed.
        if summon {
            std::env::remove_var("EFT_OVERLAY_SUMMON");
            // The handoff's EFT_POSE has done its job once `setup` (Startup) read it. Drop it in
            // PostStartup so the camera is free afterwards (main.rs gates on its presence) and a
            // later PLAY relaunch doesn't inherit a stale pose.
            app.add_systems(PostStartup, || {
                std::env::remove_var("EFT_POSE");
                std::env::remove_var("EFT_GAME_FOV");
            });
        }
        // A finite EFT_SHOT/EFT_BENCH job measures or captures — the desk-tool overlay must not
        // shape its frame clock. The persisted overlayEnabled=true would otherwise apply the
        // hidden-idle throttle (Reactive 500 ms) to the unfocused scripted window, and every
        // wall-clock number comes out as exactly 2 fps no matter the map. Resources still exist
        // (menu-mode consumers take them by value), the config is just forced off; the startup
        // focus grab is skipped for the same reason — a script must not yank the foreground.
        let automated = crate::automated_finite_job();
        let mut cfg = OverlayConfig::load().sanitized();
        if automated && cfg.enabled {
            info!("overlay: disabled for this finite EFT_SHOT/EFT_BENCH job (config untouched)");
            cfg.enabled = false;
        }
        app.insert_resource(cfg)
            .insert_resource(OverlayState {
                shown: summon && !automated,
                windowed: false,
                raise_nonce: 0,
            })
            .init_resource::<OverlayViewSlice>()
            .init_resource::<OverlayFocus>();
        if automated {
            app.add_systems(
                Update,
                (toggle_overlay, update_overlay_focus, apply_overlay, apply_overlay_view_slice)
                    .chain(),
            );
        } else {
            app.add_systems(
                Update,
                (
                    focus_atlas_on_startup,
                    summon_transparent_launch,
                    toggle_overlay,
                    update_overlay_focus,
                    apply_overlay,
                    apply_overlay_view_slice,
                )
                    .chain(),
            );
        }
        app.add_systems(
            bevy_egui::EguiPrimaryContextPass,
            (overlay_return_button, idle_badge),
        );
    }
}

/// A normal process launch should own the keyboard just as surely as a screenshot summon. Bevy's
/// initial focused flag describes the desired/cache state, not what Windows actually granted, so
/// verify the real foreground HWND for up to two seconds while the native window is appearing.
fn focus_atlas_on_startup(mut retries: Local<Option<u8>>) {
    let remaining = retries.get_or_insert(120);
    if *remaining == 0 {
        return;
    }
    if request_atlas_focus() {
        *remaining = 0;
        info!("startup: Atlas foreground focus confirmed");
    } else {
        *remaining = remaining.saturating_sub(1);
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
    if state.shown && keys.just_pressed(cfg.exit_hotkey.key_code()) {
        // DISMISS, do not exit overlay mode. This used to also set `windowed = true`, which tore
        // the window back to a decorated desktop app — reported as "the key exits overlay mode
        // instead of minimizing the overlay". Leaving `windowed` false routes through the dismiss
        // path in `apply_overlay`: with `return_focus_to_game` (default on) Atlas minimizes in
        // place and the game gets the keyboard back, while overlay mode stays armed — the next
        // screenshot (or `~`) summons it straight back to the same spot. Leaving overlay mode
        // entirely is a deliberate, mouse-sized decision: the EXIT button on the overlay itself.
        state.shown = false;
    } else if keys.just_pressed(KeyCode::Backquote) {
        // `~` remains the fast raid handoff. If Atlas is currently in ordinary window mode,
        // treat it as an explicit re-summon rather than hiding the desktop window.
        if state.windowed {
            state.windowed = false;
            state.shown = true;
            state.raise_nonce = state.raise_nonce.wrapping_add(1);
        } else {
            state.shown = !state.shown;
        }
    }
}

/// Summon a TRANSPARENT launch once the window has actually settled -- deliberately NOT at
/// Startup. The first attempt summoned on frame 1, and `apply_overlay`'s geometry pass
/// (SetWindowPos + restore + focus, all synchronous window-thread work) landed while winit was
/// still initializing and the first swapchain configure was in flight; the process froze solid
/// inside the driver -- zero CPU, `Responding = false`, log dead 0.2 s after "Creating new
/// window". Every summon that has ever worked came minutes into a settled session, so this waits
/// for one: thirty frames is imperceptible to a person and long past the first present.
fn summon_transparent_launch(
    mut st: ResMut<OverlayState>,
    t: Res<crate::TransparentWindow>,
    frames: Res<bevy::diagnostic::FrameCount>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    if !t.0 || crate::automated_finite_job() {
        *done = true;
        return;
    }
    if frames.0 >= 30 {
        *done = true;
        st.shown = true;
        st.raise_nonce = st.raise_nonce.wrapping_add(1);
        info!("overlay: transparent launch settled -- summoning");
    }
}

/// Push `OverlayState` onto the real window. Runs only on a change (the window fields are all
/// live-settable, so this is a handful of writes, never a rebuild).
fn apply_overlay(
    transparent: Res<crate::TransparentWindow>,
    cfg: Res<OverlayConfig>,
    state: Res<OverlayState>,
    mut q: Query<&mut Window, With<PrimaryWindow>>,
    // Primary monitor preferred (the game is almost always there); any monitor as a fallback so
    // the overlay still sizes itself if the primary marker hasn't been spawned yet.
    monitors: Query<&bevy::window::Monitor, With<bevy::window::PrimaryMonitor>>,
    any_monitor: Query<&bevy::window::Monitor>,
    mut winit: ResMut<bevy::winit::WinitSettings>,
    mut last_active: Local<Option<bool>>,
    mut last_nonce: Local<Option<u32>>,
    mut raise_retries: Local<u8>,
    mut focus_confirmed: Local<bool>,
    mut saved: Local<Option<(WindowPosition, UVec2)>>,
    mut overlay_rect: Local<Option<(WindowPosition, UVec2)>>,
    mut view: ResMut<OverlayViewSlice>,
    // One warning per exclusive-fullscreen episode, not one per retry frame.
    mut fs_warned: Local<bool>,
) {
    // Re-run on a shown/hidden transition, on a settings change, OR on an explicit re-raise
    // request (see `OverlayState::raise_nonce`) -- the last one is what makes a second screenshot
    // pull the window back to the front after the game has taken the foreground.
    // A transparent launch counts as enabled: the separate master switch exists so a PANEL never
    // surprises anyone over their game, but a user who chose Transparent in the menu has already
    // said exactly that. Requiring both silently ate the screenshot summon (field log: "summoning
    // the overlay" fired, no "overlay: shown" ever followed) with nothing telling the user why.
    let engaged = cfg.engaged(transparent.0);
    let active = engaged && state.shown && !state.windowed;
    let active_changed = *last_active != Some(active);
    let nonce_changed = *last_nonce != Some(state.raise_nonce);
    let config_changed = cfg.is_changed();
    if active && nonce_changed {
        // A screenshot arrives while Tarkov is still finishing its own capture/fullscreen
        // transition. One immediate raise can be overwritten by the game a frame later, leaving
        // OverlayState::shown true but the OS window minimized/behind. Retry for up to roughly two
        // seconds at 60 fps, stopping immediately once Windows confirms Atlas owns the keyboard.
        *raise_retries = 120;
        *focus_confirmed = false;
    } else if !active {
        *raise_retries = 0;
        *focus_confirmed = false;
    }
    if !active_changed && !config_changed && !nonce_changed && *raise_retries == 0 {
        return;
    }
    *last_nonce = Some(state.raise_nonce);
    // Whether the overlay was ACTUALLY up before this run. The hide branch below must only touch
    // the window on a real shown->hidden transition: on the first frame (`last` = None) and on
    // every settings tweak (cfg change marks this system dirty) `shown` is simply still false, and
    // treating that as "dismiss" minimised Atlas at startup and whenever a slider moved.
    let was_active = last_active.unwrap_or(false);
    *last_active = Some(active);
    let Ok(mut win) = q.single_mut() else { return };

    // THE ONE STATE WHERE RAISING IS WORSE THAN DOING NOTHING: an exclusive-fullscreen D3D game
    // owns the display, so the panel cannot composite over it no matter what we do — but taking
    // the foreground still costs the player their keyboard (or minimises the game) mid-raid, with
    // nothing visible in return. Ask for attention instead of seizing it, and say so once.
    let fs_blocked = active && d3d_exclusive_fullscreen();
    if fs_blocked {
        *raise_retries = 0;
        if !*fs_warned {
            *fs_warned = true;
            flash_atlas_taskbar();
            warn!(
                "overlay: the game is in EXCLUSIVE fullscreen \u{2014} the panel cannot draw over \
                 it, so Atlas is NOT taking focus (that would cost you keyboard input mid-raid). \
                 Atlas is flashing in the taskbar; switch EFT to Borderless/Windowed in its video \
                 settings for the overlay to appear over the game."
            );
        }
    } else if !active {
        *fs_warned = false;
    }

    // Follow-up raises after the full transition. Re-applying size/position/window level every
    // frame would churn the swapchain — exactly the surface path we are trying to keep stable.
    if active && !active_changed && !config_changed {
        if fs_blocked {
            return; // exclusive fullscreen: never raise, never re-ask for focus
        }
        win.visible = true;
        win.set_minimized(false);
        // `win.focused = true` is not a native-focus guarantee: it mutates Bevy's event-fed cache,
        // and Winit only acts on a false->true cache transition. Ask Windows directly every retry,
        // then stop as soon as the foreground HWND proves Atlas owns keyboard input.
        if request_atlas_focus() {
            *raise_retries = 0;
            if !*focus_confirmed {
                info!("overlay: Atlas foreground focus confirmed");
                *focus_confirmed = true;
            }
        } else {
            *raise_retries = raise_retries.saturating_sub(1);
        }
        return;
    }

    if active {
        // Reopens after BACK TO TARKOV use the exact previous overlay rectangle. Re-querying the
        // game here made placement depend on a focus race: Tarkov was sometimes visible (anchor to
        // the game) and sometimes briefly iconic (fall back to the monitor).
        let reuse_overlay_rect =
            saved.is_some() && overlay_rect.is_some() && !config_changed;
        // Remember the desktop layout ONCE (a config change while shown must not overwrite it).
        if saved.is_none() {
            *saved = Some((win.position, win.resolution.physical_size()));
        }
        win.decorations = !cfg.presentation.borderless();
        win.resizable = false;
        win.window_level = if cfg.presentation.always_on_top() {
            WindowLevel::AlwaysOnTop
        } else {
            WindowLevel::Normal
        };
        // Raise + take focus: summoned from a raid the game owns the foreground, so an always-on-top
        // window that never asks for focus would appear without receiving the WASD that follows.
        win.visible = true;
        win.set_minimized(false); // we may have minimised ourselves to hand the game focus back
        // The native request may be one frame early (before Winit applies visible/unminimized), so
        // the retry branch above keeps asking until Windows confirms Atlas is the foreground HWND.
        // Skipped entirely under exclusive fullscreen (see fs_blocked): geometry is still applied
        // so the panel is correct the moment the player alt-tabs, but we never seize the keyboard.
        if fs_blocked {
            // nothing: the taskbar flash above is the whole notification
        } else if request_atlas_focus() {
            *raise_retries = 0;
            if !*focus_confirmed {
                info!("overlay: Atlas foreground focus confirmed");
                *focus_confirmed = true;
            }
        } else {
            *raise_retries = raise_retries.saturating_sub(1);
        }
        // Panel geometry is measured against THE GAME'S WINDOW when we can see it, and the monitor
        // otherwise. Anchoring to the monitor was subtly wrong in two ways the user hits in
        // practice: on a multi-monitor desk the overlay landed on the PRIMARY monitor even when EFT
        // was running on another one, and in borderless-windowed the panel drifted off the game
        // entirely. Centring on the game's own rect is right in every configuration, and collapses
        // to the old behaviour when the game isn't up (the rect then IS the monitor, since the
        // overlay only functions over a borderless game — see the module header).
        let on_game = game_window_rect();
        let target: Option<TargetRect> = on_game.or_else(|| {
            monitors
                .iter()
                .next()
                .or_else(|| any_monitor.iter().next())
                .map(|mon| {
                    (
                        mon.physical_position,
                        Vec2::new(mon.physical_width as f32, mon.physical_height as f32),
                    )
                })
        });
        if let Some((origin, size)) = target {
            // Clamp to the target as well as to a usable minimum. A WINDOWED game can be smaller
            // than the 320x240 floor implies — at 1280x720 a 55%x60% panel is 704x432, fine, but a
            // user running the game in a small window would otherwise get a panel wider than the
            // thing it is supposed to sit on. `min` keeps it inside; `max` keeps it legible; the
            // min-of-max ordering means a genuinely tiny game window yields a panel that matches it
            // rather than one that overhangs.
            // Transparent mode covers the game EXACTLY. The panel fractions exist so an opaque
            // rectangle does not eat the whole screen, but a transparent window hides nothing --
            // and full cover is what makes the view slice the identity mapping, so a marker sits
            // on the very pixel of the thing it marks in the player's own field of view. Anything
            // smaller crops the game camera's frustum and every label is offset by the margin.
            let (fw, fh) = if transparent.0 {
                (Vec2::ONE, Vec2::ONE)
            } else {
                (cfg.size_frac, cfg.size_frac)
            };
            let w = (size.x * fw.x).round().max(320.0).min(size.x.max(320.0));
            let h = (size.y * fh.y).round().max(240.0).min(size.y.max(240.0));
            // The game rect and window position are PHYSICAL desktop pixels. `WindowResolution::set`
            // takes logical pixels and silently multiplies by DPI, making both placement and the
            // view crop drift on a scaled monitor. Keep this entire calculation in physical pixels.
            win.resolution.set_physical_resolution(w as u32, h as u32);
            // `anchor` still slides the panel inside the leftover space, so a user who prefers a
            // corner keeps it; the DEFAULT is now the centre.
            let x = ((size.x - w) * cfg.anchor.x).round() as i32 + origin.x;
            let y = ((size.y - h) * cfg.anchor.y).round() as i32 + origin.y;
            win.position = WindowPosition::At(IVec2::new(x, y));
        } else {
            win.position = WindowPosition::Centered(MonitorSelection::Primary);
        }
        if reuse_overlay_rect {
            let (pos, res) = overlay_rect.as_ref().expect("checked above");
            win.position = *pos;
            win.resolution.set_physical_resolution(res.x, res.y);
        }
        *overlay_rect = Some((win.position, win.resolution.physical_size()));
        // Build the exact game-screen crop after FINAL overlay geometry (including a cached reopen).
        // If Tarkov is briefly minimized during the focus handoff, retain the previous slice and
        // reuse it; the overlay rectangle itself is deliberately retained for the same reason.
        if let (Some((game_origin, game_size)), WindowPosition::At(overlay_origin)) =
            (on_game, win.position)
        {
            if let Some(slice) = view_slice(
                game_origin,
                game_size,
                overlay_origin,
                win.resolution.physical_size(),
            ) {
                view.0 = Some(slice);
            }
        }
        // Leave the GAME headroom for the entire time the overlay is shown. Atlas deliberately
        // takes focus so WASD/mouse cannot leak into Tarkov; limiting only `unfocused_mode` meant
        // that normal focused use ignored the user's cap and rendered at monitor refresh/full
        // speed. Apply the same policy on both sides of the focus transition. Standalone mode is
        // restored to Continuous in the cfg.enabled=false branch below.
        winit.focused_mode = overlay_update_mode(cfg.fps_cap);
        winit.unfocused_mode = overlay_update_mode(cfg.fps_cap);
        // Say WHICH rect we anchored to. "The overlay opened somewhere unexpected" is otherwise
        // undiagnosable from a log, and the two cases look identical on a single-monitor desk.
        // Report the EFFECTIVE mode and coverage, not the config's: EFT_TRANSPARENT=1 overrides
        // the config, and transparent mode covers 100% regardless of the panel fractions -- a log
        // that says "Borderless, 55%x60%" for a transparent full-cover window sends whoever reads
        // it down the wrong path.
        info!(
            "overlay: shown ({}, {:.0}%x{:.0}% of {}, cap {} fps)",
            if transparent.0 { "Transparent".to_string() } else { format!("{:?}", cfg.presentation) },
            if transparent.0 { 100.0 } else { cfg.size_frac.x * 100.0 },
            if transparent.0 { 100.0 } else { cfg.size_frac.y * 100.0 },
            if reuse_overlay_rect {
                "the previous overlay rectangle".to_string()
            } else {
                match (on_game, target) {
                    (Some(_), Some((o, s))) =>
                        format!("the GAME window {}x{} at {},{}", s.x as i32, s.y as i32, o.x, o.y),
                    (None, Some((o, s))) => format!(
                        "the monitor {}x{} at {},{} (game window not found)",
                        s.x as i32, s.y as i32, o.x, o.y
                    ),
                    _ => "the primary monitor (size unknown)".to_string(),
                }
            },
            cfg.fps_cap
        );
    } else {
        // Overlay OFF entirely: behave like a stock desktop app. Restoring Continuous here (this
        // system only runs on change) undoes any throttle a previous enable left behind, and a
        // user who never opted in never sees their unfocused frame rate touched.
        if state.windowed || !engaged {
            // A transparent-created window must NEVER be re-decorated: DWM latched the
            // conjunction at creation and a decorated transparent window blends to white with no
            // way back (see OverlayPresentation). Hide it instead -- the user asked for a normal
            // window, and an invisible one until the next summon is the nearest honest state.
            if transparent.0 {
                win.visible = false;
                *overlay_rect = None;
                view.0 = None;
                winit.focused_mode = UpdateMode::Continuous;
                winit.unfocused_mode = UpdateMode::Continuous;
                if state.windowed {
                    info!(
                        "overlay: hidden (transparent windows cannot become ordinary windows;                          relaunch without transparent mode for a desktop window)"
                    );
                }
                return;
            }
            win.window_level = WindowLevel::Normal;
            win.decorations = true;
            win.resizable = true;
            win.visible = true;
            win.set_minimized(false);
            if let Some((pos, res)) = saved.take() {
                win.position = pos;
                win.resolution.set_physical_resolution(res.x, res.y);
            }
            *overlay_rect = None;
            view.0 = None;
            winit.focused_mode = UpdateMode::Continuous;
            winit.unfocused_mode = UpdateMode::Continuous;
            if state.windowed {
                info!("overlay: exited to the ordinary resizable Atlas window");
            }
            return;
        }
        // Window mutations only on a REAL dismiss (see `was_active` above) — never at startup and
        // never because a settings checkbox redrew us.
        if was_active {
            win.window_level = WindowLevel::Normal;
            if cfg.return_focus_to_game {
                // Minimize in place. Restoring the desktop geometry before minimizing is what made
                // the next screenshot reopen in a different position. Windows still gives Tarkov
                // the keyboard, while `overlay_rect` retains the exact rectangle for the reopen.
                win.set_minimized(true);
            } else if transparent.0 {
                // Same rule as the exit path: never re-decorate a transparent window. Minimize is
                // the dismiss that preserves the conjunction.
                win.set_minimized(true);
            } else {
                // No automatic handoff: restore the ordinary desktop window immediately.
                win.decorations = true;
                win.resizable = true;
                if let Some((pos, res)) = saved.take() {
                    win.position = pos;
                    win.resolution.set_physical_resolution(res.x, res.y);
                }
                *overlay_rect = None;
                view.0 = None;
            }
            info!(
                "overlay: hidden (rectangle preserved={}, focus to game={}, unfocused idle={})",
                cfg.return_focus_to_game,
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

#[cfg(test)]
mod overlay_update_mode_tests {
    use super::*;

    #[test]
    fn positive_overlay_cap_uses_the_requested_timer() {
        match overlay_update_mode(50) {
            UpdateMode::Reactive {
                wait,
                react_to_device_events,
                react_to_user_events,
                react_to_window_events,
            } => {
                assert_eq!(wait, std::time::Duration::from_millis(20));
                assert!(!react_to_device_events);
                assert!(react_to_user_events);
                assert!(react_to_window_events);
            }
            other => panic!("expected capped reactive mode, got {other:?}"),
        }
    }

    #[test]
    fn zero_overlay_cap_is_uncapped() {
        assert!(matches!(overlay_update_mode(0), UpdateMode::Continuous));
    }
}

/// Apply the asymmetric projection only while the overlay is visible. The camera position,
/// rotation, and Tarkov FOV still come from `game_watch`; this supplies the final missing datum:
/// which rectangle of the full game image the Atlas window has replaced.
fn apply_overlay_view_slice(
    cfg: Res<OverlayConfig>,
    state: Res<OverlayState>,
    transparent: Res<crate::TransparentWindow>,
    view: Res<OverlayViewSlice>,
    mut active: ResMut<crate::render::OverlaySlice>,
) {
    // Publishes INTENT. `crate::render::apply_view_slice` is the sole writer of
    // `Camera::sub_camera_view` and decides precedence, because this used to write the camera
    // directly and so did `ui::fit_camera_viewport` -- two writers for one field, in two different
    // schedules. `fit_camera_viewport` runs in EguiPrimaryContextPass, which is LATER in the frame
    // than this Update system, so the panel lens-shift overwrote the overlay's asymmetric frustum
    // every single frame (and cleared it outright whenever no side panel was up). That frustum is
    // the entire perspective match: without it a marker no longer lands on the game pixel it
    // belongs to, which is the one thing the overlay exists to guarantee.
    let desired = if cfg.engaged(transparent.0) && state.shown && !state.windowed {
        view.0
    } else {
        None
    };
    if active.0 != desired {
        active.0 = desired;
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
    cfg: Res<OverlayConfig>,
    mut state: ResMut<OverlayState>,
) {
    use crate::i18n::{t, K};
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};

    if !state.shown || state.windowed || menu.is_some() {
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
                ui.add_space(5.0);
                let exit = egui::Button::new(
                    RichText::new(t(lg, K::OverlayExitWindow))
                        .size(12.0)
                        .strong()
                        .color(theme::TEXT_BRIGHT),
                )
                .fill(theme::CARD)
                .corner_radius(5.0)
                .min_size(egui::vec2(280.0, 32.0));
                if ui.add(exit).clicked() {
                    state.shown = false;
                    state.windowed = true;
                }
                ui.label(
                    RichText::new(format!(
                        "{} \u{00B7} {}",
                        cfg.exit_hotkey.label(),
                        t(lg, K::OverlayExitHint)
                    ))
                    .size(9.0)
                    .color(theme::MUTED),
                );
                // `~` also dismisses, and it was documented only in a menu tooltip — on a
                // tenkeyless/laptop keyboard the NumpadEnter default does not exist at all, so
                // the overlay itself has to name a key the user definitely has.
                ui.label(
                    RichText::new("~ also hides the overlay").size(9.0).color(theme::MUTED),
                );
            });
        });
}

/// "Atlas is idling on purpose" badge.
///
/// With the overlay armed but dismissed, an UNFOCUSED Atlas is deliberately throttled to a 500 ms
/// reactive tick so it costs the game nothing — including when the window is fully visible on a
/// second monitor. Measured user reaction to an unannounced 2 fps window: "it's frozen/broken".
/// The reactive mode still redraws on window events, so this badge paints; it disappears the
/// moment the window is focused (and therefore no longer throttled).
#[cfg(feature = "egui")]
fn idle_badge(
    mut contexts: bevy_egui::EguiContexts,
    menu: Option<Res<crate::menu::MenuState>>,
    cfg: Res<OverlayConfig>,
    state: Res<OverlayState>,
    windows: Query<&Window>,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};

    // Only the exact state that throttles: overlay armed, dismissed, pause-when-hidden on, and
    // the window not focused. Anything else redraws normally and needs no explanation.
    if menu.is_some() || state.shown || state.windowed || !cfg.enabled || !cfg.pause_when_hidden {
        return;
    }
    let Ok(win) = windows.single() else { return };
    if win.focused || !win.visible {
        return; // focused = full speed; invisible = nothing to explain
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Area::new(egui::Id::new("atlas_idle_badge"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(12.0, -12.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme::CARD)
                .stroke(egui::Stroke::new(1.0, theme::BORDER_STRONG))
                .inner_margin(egui::Margin::symmetric(10, 6))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("idling to leave the game its frame rate")
                            .size(11.0)
                            .color(theme::TEXT_BRIGHT),
                    );
                    ui.label(
                        RichText::new(
                            "click this window (or take a screenshot in raid) to resume",
                        )
                        .size(9.0)
                        .color(theme::MUTED),
                    );
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_enter_is_the_default_exit_hotkey_and_round_trips() {
        assert_eq!(
            OverlayConfig::default().exit_hotkey,
            OverlayExitHotkey::NumpadEnter
        );
        for key in OverlayExitHotkey::ALL {
            assert_eq!(
                OverlayExitHotkey::from_config(key.config_value()),
                Some(key)
            );
        }
    }

    #[test]
    fn centered_overlay_maps_to_the_same_game_pixels() {
        let slice = view_slice(
            IVec2::new(100, 50),
            Vec2::new(1920.0, 1080.0),
            IVec2::new(532, 266),
            UVec2::new(1056, 648),
        )
        .expect("valid slice");
        assert_eq!(slice.full_size, UVec2::new(1920, 1080));
        assert_eq!(slice.offset, Vec2::new(432.0, 216.0));
        assert_eq!(slice.size, UVec2::new(1056, 648));
    }

    #[test]
    fn rejects_degenerate_view_slices() {
        assert!(
            view_slice(IVec2::ZERO, Vec2::ZERO, IVec2::ZERO, UVec2::new(800, 600)).is_none()
        );
        assert!(
            view_slice(IVec2::ZERO, Vec2::new(1920.0, 1080.0), IVec2::ZERO, UVec2::ZERO)
                .is_none()
        );
    }

    /// The window title we match on is the game's own Unity PRODUCT NAME, which it records in
    /// `EscapeFromTarkov_Data/app.info` (line 1 company, line 2 product). Assert the matcher
    /// accepts whatever that file actually says rather than a name someone typed here — if BSG
    /// ever renames the product, this fails instead of the overlay silently drifting back to
    /// monitor-centring. Skips cleanly where the game isn't installed.
    #[test]
    fn matcher_accepts_the_games_own_product_name() {
        let dir = crate::menu::detect_game_dir();
        let Ok(info) = std::fs::read_to_string(std::path::Path::new(&dir).join("app.info")) else {
            return; // game not installed here — nothing to check against
        };
        let Some(product) = info.lines().nth(1).map(str::trim).filter(|s| !s.is_empty()) else {
            return;
        };
        assert!(
            title_is_game(product),
            "app.info product {product:?} no longer matches the overlay's window-title rule"
        );
    }

    /// A window that merely mentions the game must not capture the overlay.
    #[test]
    fn matcher_rejects_lookalikes() {
        assert!(title_is_game("EscapeFromTarkov"));
        assert!(title_is_game("escape from tarkov"));
        assert!(title_is_game("EscapeFromTarkov - Direct3D 11"));
        assert!(!title_is_game("Atlas"));
        assert!(!title_is_game("Escape From Tarkov Wiki - Chrome"));
        assert!(!title_is_game(""));
    }

    /// The Win32 declarations must actually work: enumerating top-level windows has to run without
    /// tripping the FFI and return either nothing or a plausible rect — never a degenerate one.
    /// (The game is usually not running while tests are, so `None` is the expected pass.)
    #[test]
    #[cfg(windows)]
    fn window_lookup_runs_and_returns_a_sane_rect() {
        if let Some((_origin, size)) = game_window_rect() {
            assert!(size.x >= 320.0 && size.y >= 240.0, "degenerate rect {size:?}");
            assert!(size.x < 32_768.0 && size.y < 32_768.0, "implausible rect {size:?}");
        }
    }

    /// The WINDOWED case, checked against a real window rather than reasoned about: with a stand-in
    /// titled like the game on screen, `EFT_TEST_CLIENT_RECT="x,y,w,h"` carries that window's true
    /// CLIENT rect and this asserts we report exactly it. That is the whole point of the
    /// GetClientRect/ClientToScreen path — a GetWindowRect placement returns the outer frame, which
    /// in windowed mode is offset by the title bar and wider by the borders, so the overlay would
    /// sit visibly high and off-centre over the game's picture. Skipped when the var is unset.
    #[test]
    #[cfg(windows)]
    fn windowed_lookup_reports_the_client_rect_not_the_frame() {
        let Ok(spec) = std::env::var("EFT_TEST_CLIENT_RECT") else { return };
        let v: Vec<i32> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        assert_eq!(v.len(), 4, "EFT_TEST_CLIENT_RECT must be x,y,w,h");
        let (origin, size) = game_window_rect().expect("stand-in game window not found");
        assert_eq!(
            (origin.x, origin.y, size.x as i32, size.y as i32),
            (v[0], v[1], v[2], v[3]),
            "reported rect is the window FRAME, not the client area"
        );
    }
}
