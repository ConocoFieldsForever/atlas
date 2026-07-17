//! eft::menu_fx — vector-drawn start-menu decor + custom widgets. Everything here is pure
//! egui painter geometry (rects/circles/lines/polys): no image assets, nothing game-derived.
//!
//! * [`eft_loading_bar`] — game-loader-style segmented progress bar for the build panel:
//!   khaki segments with 1px edges, pulsing frontier segment, stage text + percent upper-left,
//!   ESTIMATED TIME mm:ss upper-right, small midpoint tick.
//! * [`security_camera`] — main-menu-vibe wall CCTV in the top-right that servo-tracks the
//!   mouse cursor (yaw/pitch clamped cone, exponential slew), blinking red LED, idle patrol
//!   sweep when the pointer leaves the window.
//!
//! Palette mirrors menu.rs (near-black field / charcoal panels / bone text) but runs a step
//! dimmer so the decor stays background. ASCII-only labels (glyph whitelist).

use bevy_egui::egui::{self, Color32, Rect, Shape, Stroke, StrokeKind, pos2, vec2};

// ---- shared palette (kept in step with menu.rs) ----
const KHAKI: Color32 = Color32::from_rgb(207, 203, 184); // filled loader segment (bone/khaki)
const RED: Color32 = Color32::from_rgb(176, 65, 62); // menu BAD / camera LED
const TXT: Color32 = Color32::from_rgb(199, 195, 183);
const DIM: Color32 = Color32::from_rgb(110, 107, 100);

fn scale_rgb(c: Color32, f: f32) -> Color32 {
    Color32::from_rgb(
        (c.r() as f32 * f) as u8,
        (c.g() as f32 * f) as u8,
        (c.b() as f32 * f) as u8,
    )
}

/// EFT-loader-style segmented progress bar.
///
/// * `frac` in [0,1] — stage-derived progress; fully filled segments are khaki, the frontier
///   segment pulses (~1.5 Hz) while work is running, the rest stay dark charcoal.
/// * `stage_text` — "LOADING OBJECTS..." style line, drawn upper-left with the percent under it.
/// * `elapsed_secs` — build wall time; drives the ESTIMATED TIME readout upper-right
///   (naive `elapsed/frac - elapsed`, shown as "--:--" until `frac` is meaningful).
/// * `failed` — recolors filled segments the menu red and freezes the pulse.
pub fn eft_loading_bar(
    ui: &mut egui::Ui,
    frac: f32,
    stage_text: &str,
    elapsed_secs: f32,
    failed: bool,
) {
    use egui::RichText;
    const SEGS: usize = 12; // the game loader reads as ~10-14 wide segments
    const GAP: f32 = 2.0;
    const BAR_H: f32 = 18.0;
    const EMPTY: Color32 = Color32::from_rgb(23, 23, 21);
    const EDGE: Color32 = Color32::from_rgb(43, 42, 38);
    const FRAME: Color32 = Color32::from_rgb(74, 72, 65);

    let frac = frac.clamp(0.0, 1.0);
    let eta = if failed || frac < 0.05 || elapsed_secs < 2.0 {
        "--:--".to_string() // no sane estimate yet (or never will be)
    } else if frac >= 1.0 {
        "00:00".to_string()
    } else {
        let est = (elapsed_secs / frac - elapsed_secs).clamp(0.0, 99.0 * 60.0 + 59.0);
        format!("{:02}:{:02}", (est / 60.0) as u32, (est % 60.0) as u32)
    };

    // Header row: stage + percent upper-left, ESTIMATED TIME + mm:ss upper-right.
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(
                RichText::new(stage_text)
                    .color(if failed { RED } else { TXT })
                    .size(12.0)
                    .strong(),
            );
            ui.label(RichText::new(format!("{:.0}%", frac * 100.0)).color(DIM).size(11.0));
        });
        ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
            ui.label(RichText::new("ESTIMATED TIME").color(DIM).size(10.0));
            ui.label(RichText::new(eta).color(TXT).size(12.0).strong());
        });
    });

    ui.add_space(6.0); // headroom for the midpoint tick above the frame
    let (rect, _) =
        ui.allocate_exact_size(vec2(ui.available_width(), BAR_H), egui::Sense::hover());
    let p = ui.painter();
    p.rect_stroke(rect, 0.0, Stroke::new(1.0, FRAME), StrokeKind::Inside);
    // Midpoint tick (the game loader's 50% marker) — a small notch above the frame. With an
    // even segment count the center also lands exactly on a gap, which reads as a divider.
    let cx = rect.center().x.floor() + 0.5;
    p.line_segment(
        [pos2(cx, rect.top() - 5.0), pos2(cx, rect.top() - 1.0)],
        Stroke::new(1.0, DIM),
    );

    let inner = rect.shrink(2.0);
    let seg_w = (inner.width() - GAP * (SEGS as f32 - 1.0)) / SEGS as f32;
    if seg_w < 2.0 {
        return; // panel squeezed to nothing — keep the frame, skip segments
    }
    let t = ui.input(|i| i.time) as f32;
    let frontier = (frac * SEGS as f32).floor() as usize; // == SEGS when frac hits 1.0
    let base = if failed { RED } else { KHAKI };
    for s in 0..SEGS {
        let x0 = inner.left() + s as f32 * (seg_w + GAP);
        let r = Rect::from_min_size(pos2(x0, inner.top()), vec2(seg_w, inner.height()));
        if s < frontier {
            p.rect_filled(r, 0.0, base);
        } else if s == frontier && frac < 1.0 {
            // Frontier segment: brightness pulse ~1.5 Hz. The sine is sharpened (^1.6) so it
            // dwells dim and snaps bright — reads mechanical rather than "breathing".
            let pulse = if failed {
                0.5 // static half-bright red stub marks where the build died
            } else {
                let w = 0.5 + 0.5 * (t * 1.5 * std::f32::consts::TAU).sin();
                0.35 + 0.65 * w.powf(1.6)
            };
            p.rect_filled(r, 0.0, scale_rgb(base, pulse));
            p.rect_stroke(r, 0.0, Stroke::new(1.0, EDGE), StrokeKind::Inside);
        } else {
            p.rect_filled(r, 0.0, EMPTY);
            p.rect_stroke(r, 0.0, Stroke::new(1.0, EDGE), StrokeKind::Inside);
        }
    }
}

/// Wall-mounted CCTV camera decor: paints directly with the panel painter, so call it BEFORE
/// the map list and it layers behind everything added later. It creates no widget and senses
/// nothing — pure painter output can never steal pointer input.
///
/// Tracking model (2D fake-3D, screen coords, y down):
/// * pitch = angular offset of the cursor from the rest direction (down-left), clamped to
///   +-25 deg — rotates the whole body in screen space (the mount is side-on to us).
/// * yaw   = lateral position of the cursor across the panel mapped to +-40 deg — shown by
///   foreshortening the body and skewing the front face + lens sideways along the normal.
/// * both angles slew with a frame-rate-independent exponential (servo feel, not glued).
/// * pointer outside the window -> slow idle patrol sweep instead.
///
/// FUTURE (comment only, deliberately not implemented): a variant could swap the vector body
/// for a user-extracted Tagilla-helmet sprite from packs/shared/menu/ when present. No asset
/// loading here — vector-only is what ships.
pub fn security_camera(ui: &egui::Ui, panel: Rect) {
    use std::f32::consts::{PI, TAU};
    const BODY: Color32 = Color32::from_rgb(42, 42, 40); // #2a2a28
    const DARK: Color32 = Color32::from_rgb(28, 28, 27); // #1c1c1b
    const EDGE: Color32 = Color32::from_rgb(19, 19, 18);
    const SEAM: Color32 = Color32::from_rgb(52, 51, 47);
    const METAL: Color32 = Color32::from_rgb(35, 35, 34);

    let ctx = ui.ctx();
    let t = ctx.input(|i| i.time) as f32;
    let dt = ctx.input(|i| i.stable_dt).min(0.1);

    // Mount: plate on the panel's right edge, arm reaching left-down to the pivot joint.
    let plate = Rect::from_min_max(
        pos2(panel.right() - 12.0, panel.top() + 24.0),
        pos2(panel.right() - 4.0, panel.top() + 62.0),
    );
    let pivot = pos2(plate.left() - 34.0, plate.center().y + 10.0);

    // ---- target pose ----
    const YAW_MAX: f32 = 40.0 * PI / 180.0;
    const PITCH_MAX: f32 = 25.0 * PI / 180.0;
    let rest = 2.62_f32; // rad (~150 deg): down-left, toward the middle of the map list
    let (yaw_t, pitch_t) = match ctx.input(|i| i.pointer.hover_pos()) {
        Some(m) => {
            let v = m - pivot;
            // pitch: wrapped screen-space angle to the cursor, relative to rest
            let mut d = v.y.atan2(v.x) - rest;
            while d > PI {
                d -= TAU;
            }
            while d < -PI {
                d += TAU;
            }
            // yaw: how far across the panel the cursor sits. Directly under the camera
            // (v.x ~ 0) turns it toward the wall (+), far screen-left turns it into the
            // scene (-). Linear in x reads better than atan2-vs-depth at this size.
            let yn = (1.0 + 2.0 * v.x / panel.width().max(1.0)).clamp(-1.0, 1.0);
            (yn * YAW_MAX, d.clamp(-PITCH_MAX, PITCH_MAX))
        }
        // Idle patrol: slow side-to-side sweep (~12 s period) with a faint nod.
        None => (
            (t * 0.5).sin() * YAW_MAX * 0.85,
            (t * 0.23).sin() * 0.05 - 0.06,
        ),
    };

    // ---- servo slew: exponential approach, frame-rate independent (alpha = 1-e^-k*dt) ----
    let id = egui::Id::new("menu_fx_cam_pose");
    let (mut yaw, mut pitch) =
        ctx.data(|d| d.get_temp::<(f32, f32)>(id)).unwrap_or((0.0, -0.05));
    let a = 1.0 - (-6.0 * dt).exp();
    yaw += (yaw_t - yaw) * a;
    pitch += (pitch_t - pitch) * a;
    ctx.data_mut(|d| d.insert_temp(id, (yaw, pitch)));

    // ---- geometry ----
    let ang = rest + pitch;
    let u = egui::Vec2::angled(ang); // body forward axis
    let n = vec2(-u.y, u.x); // body normal; +n is the "top" side for the rest pose
    let len = 56.0 * (0.78 + 0.22 * yaw.cos()); // yaw foreshortens the body
    let (wb, wf) = (13.0_f32, 10.0_f32); // back / front half-widths (tapered box)
    let sk = n * (yaw.sin() * 9.0); // yaw skews the front face + lens laterally

    let b = pivot + vec2(1.0, 9.0); // body back-center hangs under the bracket joint
    let f = b + u * len;
    let fc = f + sk; // front-face (lens) center

    let p = ui.painter();

    // Faint surveillance cone out of the lens — barely-there, sells the tracking.
    p.add(Shape::convex_polygon(
        vec![
            fc,
            fc + egui::Vec2::angled(ang + 0.15) * 88.0,
            fc + egui::Vec2::angled(ang - 0.15) * 88.0,
        ],
        Color32::from_rgba_unmultiplied(200, 196, 180, 6),
        Stroke::NONE,
    ));

    // Wall plate + screws.
    p.rect_filled(plate, 0.0, DARK);
    p.rect_stroke(plate, 0.0, Stroke::new(1.0, SEAM), StrokeKind::Inside);
    for dy in [-13.0, 13.0] {
        p.circle_filled(pos2(plate.center().x, plate.center().y + dy), 1.3, SEAM);
    }
    // Arm + pivot joint + short bracket down to the body.
    p.line_segment([plate.left_center(), pivot], Stroke::new(4.0, METAL));
    p.line_segment([pivot, b], Stroke::new(2.5, METAL));
    p.circle_filled(pivot, 4.5, DARK);
    p.circle_stroke(pivot, 4.5, Stroke::new(1.0, SEAM));

    // Body: tapered quad from back (at the bracket) to the skewed front face.
    p.add(Shape::convex_polygon(
        vec![b + n * wb, fc + n * wf, fc - n * wf, b - n * wb],
        BODY,
        Stroke::new(1.0, EDGE),
    ));
    // Darker rear cap (power/cable block).
    let bp = |du: f32, w: f32| b + u * du + n * w;
    p.add(Shape::convex_polygon(
        vec![
            bp(-3.0, wb + 1.5),
            bp(7.0, wb + 1.5),
            bp(7.0, -(wb + 1.5)),
            bp(-3.0, -(wb + 1.5)),
        ],
        DARK,
        Stroke::new(1.0, EDGE),
    ));
    // Panel seams across the housing (skew interpolated toward the front).
    for fs in [0.42_f32, 0.68] {
        let c = b + u * (len * fs) + sk * fs;
        let w = wb + (wf - wb) * fs - 1.0;
        p.line_segment([c + n * w, c - n * w], Stroke::new(1.0, METAL));
    }
    // Hood / sun visor over the lens (protrudes past the front on the top side).
    let hp = |du: f32, w: f32| fc + u * du + n * w;
    p.add(Shape::convex_polygon(
        vec![
            hp(-2.0, wf + 1.5),
            hp(7.0, wf + 1.5),
            hp(7.0, wf - 3.0),
            hp(-2.0, wf - 3.0),
        ],
        DARK,
        Stroke::new(1.0, EDGE),
    ));
    // Lens: dark glass, ring, dim curved glint.
    p.circle_filled(fc, 6.0, Color32::from_rgb(16, 16, 16));
    p.circle_stroke(fc, 6.0, Stroke::new(1.0, SEAM));
    p.circle_filled(fc, 3.2, Color32::from_rgb(24, 25, 26));
    p.circle_filled(
        fc + vec2(-1.8, -1.8),
        1.3,
        Color32::from_rgba_unmultiplied(150, 158, 158, 90),
    );

    // Record LED, top-rear: sharp 0.8 s on/off blink with a soft glow halo, independent of
    // the tracking pose.
    let led = b + u * 9.0 + n * (wb - 3.5);
    if t.rem_euclid(0.8) < 0.4 {
        p.circle_filled(led, 7.5, Color32::from_rgba_unmultiplied(RED.r(), RED.g(), RED.b(), 18));
        p.circle_filled(led, 4.5, Color32::from_rgba_unmultiplied(RED.r(), RED.g(), RED.b(), 55));
        p.circle_filled(led, 2.1, Color32::from_rgb(226, 82, 74));
    } else {
        p.circle_filled(led, 1.8, Color32::from_rgb(58, 34, 32));
    }
}
