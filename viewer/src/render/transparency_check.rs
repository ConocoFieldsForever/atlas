//! TRANSPARENT-MODE STARTUP SELF-CHECK — because every way this feature fails is silent.
//!
//! The measurements behind the transparent overlay found no failure mode that announces itself.
//! A window created with the wrong attribute conjunction composites opaque with no error. A
//! surface that lacks the blending alpha mode degrades to Opaque with only a log line (the
//! vendored guard). A post pass that drops the alpha channel — the grade shader did exactly this
//! for months — blacks out the see-through with nothing to catch it. In every case the user sees
//! "transparent mode is on and the window is a solid slab", and nothing in a log says why.
//!
//! So transparent launches verify themselves once, from the inside, at two layers:
//!
//!  1. WINDOW ATTRIBUTES: the DWM conjunction (undecorated + non-resizable + always-on-top) is
//!     re-read from the live `Window` component a few frames in. This catches the future edit
//!     that flips one of them after creation — which cannot be repaired at runtime, only
//!     reported — before anyone spends an evening bisecting graphics settings.
//!  2. RENDERED ALPHA: one GPU readback of the presented frame, sampling the four corners
//!     (the map HUD keeps the CENTRE busy; corners are where the game should show through in ESP
//!     mode). If every corner comes back opaque, the alpha channel died somewhere between the
//!     clear colour and the swapchain, and the one-line verdict names the usual suspects.
//!
//! The check runs ONCE per launch, logs a PASS at info so a healthy run says so, and never
//! repeats — a per-frame readback would be a measurable cost for a diagnostic that cannot change
//! after startup.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::PrimaryWindow;

/// Frame to run the check on. Late enough that the swapchain has real content on every render
/// path (the GPU-driven path presents its first true frame only after "GPU buffers built"), early
/// enough that a broken launch is diagnosed while the user is still looking at it.
const CHECK_FRAME: u32 = 90;

#[derive(Resource, Default)]
struct CheckState {
    fired: bool,
}

fn schedule_check(
    mut state: ResMut<CheckState>,
    frames: Res<bevy::diagnostic::FrameCount>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut commands: Commands,
) {
    if state.fired || frames.0 < CHECK_FRAME {
        return;
    }
    state.fired = true;

    // Layer 1: the creation conjunction, re-read from the live component. A mismatch here is
    // unfixable at runtime (DWM latched it at CreateWindowEx), so the message says what to do
    // rather than pretending a toggle could save the session.
    if let Ok(win) = windows.single() {
        let ok = !win.decorations
            && !win.resizable
            && win.window_level == bevy::window::WindowLevel::AlwaysOnTop
            && win.transparent;
        if ok {
            info!("transparency self-check 1/2: window attributes hold the DWM conjunction");
        } else {
            error!(
                "transparency self-check 1/2 FAILED: decorations={} resizable={} level={:?} \
                 transparent={} — DWM latches these at window creation, so this window will \
                 composite OPAQUE for its whole life. Something mutated the window after launch; \
                 the overlay summon path is the usual suspect.",
                win.decorations, win.resizable, win.window_level, win.transparent
            );
        }
    }

    // Layer 2: does the frame we present actually carry alpha?
    commands.spawn(Screenshot::primary_window()).observe(check_alpha);
}

fn check_alpha(captured: On<ScreenshotCaptured>) {
    let img = &captured.event().image;
    let (w, h) = (img.width(), img.height());
    if w < 8 || h < 8 {
        return;
    }
    // Four corners, 4 px in from the edge; in ESP mode these are backdrop unless a panel covers
    // them, and one uncovered transparent corner is enough to prove the channel is alive.
    let corners = [(4, 4), (w - 5, 4), (4, h - 5), (w - 5, h - 5)];
    let mut alphas = [255u8; 4];
    for (i, (x, y)) in corners.into_iter().enumerate() {
        // 4 bytes/px for every swapchain format we configure (Bgra8/Rgba8); alpha is byte 3.
        let idx = ((y * w + x) * 4 + 3) as usize;
        if let Some(data) = img.data.as_ref() {
            if let Some(a) = data.get(idx).copied() {
                alphas[i] = a;
            }
        }
    }
    if alphas.iter().any(|a| *a < 250) {
        info!(
            "transparency self-check 2/2: rendered alpha is live (corner alphas {:?})",
            alphas
        );
    } else {
        error!(
            "transparency self-check 2/2 FAILED: every sampled corner is opaque (alphas {:?}). \
             The alpha channel died between the clear colour and the swapchain. Usual suspects: \
             a post pass writing alpha 1.0 (grade.wgsl was the historical offender), an opaque \
             egui CentralPanel, or the surface silently downgraded to Opaque (see the \
             'composite alpha mode' warning above, if any).",
            alphas
        );
    }
}

/// Registered only on transparent launches; on every other launch the plugin is absent and the
/// cost is zero.
pub struct TransparencyCheckPlugin;

impl Plugin for TransparencyCheckPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CheckState>().add_systems(Update, schedule_check);
    }
}
