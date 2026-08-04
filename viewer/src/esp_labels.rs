//! ESP LABELS — what a player can actually read in two seconds, over a live raid.
//!
//! The constraint that shapes everything here is EXPOSURE TIME, not screen space. Summoning the
//! overlay means standing still in a raid behind an opaque panel. So the question is never "how
//! much can we show" but "what is worth the two seconds", and every rule below is a way of
//! throwing information away on purpose.
//!
//! WHAT A PLAYER WANTS, in the order they want it:
//!   1. Is it worth walking to?      -> value, and only for things that have one
//!   2. How far?                     -> metres, integer, no decimals
//!   3. Is it even on my floor?      -> the height delta, which on a multi-storey map like
//!                                      Interchange decides whether "18 m" means 18 m or a
//!                                      staircase hunt. This is the single highest-value glyph
//!                                      on the screen and it costs one arrow.
//!   4. What is it?                  -> the name, truncated hard
//!
//! WHAT IT REFUSES TO DO:
//!   * Label everything. 906 loot containers and 2,790 POI markers is wallpaper, and wallpaper is
//!     read as noise and then ignored — the failure mode is not "cluttered", it is "the player
//!     stops looking". A hard budget (default 12) means the twelve are worth reading.
//!   * Label what is already unreadable. Beyond `max_dist` a marker is a pixel; it is counted in
//!     the summary line instead, which is more useful than a label nobody can resolve.
//!   * Draw a card per marker. `inspect.rs`'s per-marker `egui::Area` is right for six and fatal
//!     for three hundred: one foreground painter draws all of them.
//!
//! Ordering is by VALUE where a value exists and by distance otherwise, because "the nearest
//! thing" and "the thing worth walking to" are different questions and only the second one makes
//! somebody move.

use bevy::prelude::*;

/// Tunables, all user-facing eventually; the defaults are the ones that survive a raid.
#[derive(Resource)]
pub struct EspLabels {
    /// Hard cap on world-space labels. The whole point.
    pub budget: usize,
    /// Past this, a marker is a pixel — count it, do not label it.
    pub max_dist: f32,
    /// Screen-space declutter cell, logical points. One label per cell.
    pub cell_px: f32,
    /// Show the bottom-left NEAREST list.
    pub show_list: bool,
}

impl Default for EspLabels {
    fn default() -> Self {
        Self { budget: 12, max_dist: 250.0, cell_px: 64.0, show_list: true }
    }
}

/// One marker that survived every filter, ready to draw.
struct Cand {
    screen: Vec2,
    dist: f32,
    dy: f32,
    value: Option<i64>,
    title: String,
    accent: Color,
}

/// `18k` / `250k` / `1.2M` — a value you can read at a glance beats one you have to parse.
fn short_value(v: i64) -> String {
    match v {
        v if v >= 1_000_000 => format!("{:.1}M", v as f32 / 1.0e6),
        v if v >= 1_000 => format!("{}k", v / 1_000),
        v => format!("{v}"),
    }
}

/// Truncate on GRAPHEME-ish boundaries cheaply; marker titles are ASCII-ish in practice but a
/// Cyrillic name must not panic a slice.
fn short_title(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

#[allow(clippy::too_many_arguments)]
pub fn draw_esp_labels(
    mut contexts: bevy_egui::EguiContexts,
    esp: Res<crate::EspMode>,
    cfg: Res<EspLabels>,
    menu: Option<Res<crate::menu::MenuState>>,
    cams: Query<(&Camera, &GlobalTransform), With<crate::render::CullCamera>>,
    markers: Query<(
        &GlobalTransform,
        &crate::inspect::MarkerInfo,
        &InheritedVisibility,
        Option<&crate::poi::MarkerValue>,
    )>,
) {
    if !esp.0 || menu.is_some() {
        return; // labels are the ESP presentation; the 3D map has the world to read instead
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let Ok((camera, cam_tf)) = cams.single() else { return };
    let eye = cam_tf.translation();

    // Gather -> cull -> declutter -> budget. Each stage only ever removes.
    let mut cands: Vec<Cand> = Vec::new();
    for (tf, info, vis, value) in &markers {
        if !vis.get() {
            continue; // the layer toggles already decided this; never second-guess them here
        }
        let p = tf.translation();
        let dist = p.distance(eye);
        if dist > cfg.max_dist || dist < 0.5 {
            continue;
        }
        let Ok(screen) = camera.world_to_viewport(cam_tf, p) else {
            continue; // behind the camera
        };
        cands.push(Cand {
            screen,
            dist,
            dy: p.y - eye.y,
            value: value.map(|v| v.0).filter(|v| *v > 0),
            title: info.title.clone(),
            accent: info.accent,
        });
    }

    // Rank by VALUE PER METRE, not by value. Sorting on raw worth filled the screen with twelve
    // identical "Weapon box 26k" rows differing only in how far away they were, which tells the
    // player nothing and reads as a bug. Value per metre answers what they are actually asking --
    // what is worth the walk from HERE -- so a 26k box at 44 m outranks the same box at 216 m and
    // the list stops being one repeated row.
    let score = |c: &Cand| c.value.map(|v| v as f32 / c.dist.max(1.0));
    cands.sort_by(|a, b| match (score(a), score(b)) {
        (Some(x), Some(y)) => y.total_cmp(&x),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.dist.total_cmp(&b.dist),
    });

    // One per screen cell, keeping the first — which after the sort is the most valuable, or the
    // nearest among the unvalued. Two labels on top of each other are worth less than one.
    let mut taken: std::collections::HashSet<(i32, i32)> = Default::default();
    let mut shown: Vec<&Cand> = Vec::new();
    let mut hidden = 0usize;
    for c in &cands {
        let key = ((c.screen.x / cfg.cell_px) as i32, (c.screen.y / cfg.cell_px) as i32);
        if shown.len() < cfg.budget && taken.insert(key) {
            shown.push(c);
        } else {
            hidden += 1;
        }
    }

    // The free central region; the side panels own the rest, and a label drawn under them (or off
    // the window) is budget spent on nothing.
    let free = ctx.available_rect();
    let painter = ctx.layer_painter(bevy_egui::egui::LayerId::new(
        bevy_egui::egui::Order::Foreground,
        bevy_egui::egui::Id::new("esp_labels"),
    ));
    use bevy_egui::egui::{self, Align2, Color32, FontId, Stroke};

    for c in &shown {
        if !free.contains(egui::pos2(c.screen.x, c.screen.y)) {
            continue;
        }
        let col = {
            let s = c.accent.to_srgba();
            Color32::from_rgb((s.red * 255.0) as u8, (s.green * 255.0) as u8, (s.blue * 255.0) as u8)
        };
        // `Safe  34m ↑7  120k` — one line, fixed order, so the eye learns where to look once and
        // then always finds the number in the same place.
        let mut text = short_title(&c.title, 18);
        text.push_str(&format!("  {}m", c.dist.round() as i32));
        // Only when it MATTERS: a 2 m step is not worth a glyph, a storey is.
        if c.dy.abs() >= 2.0 {
            // ASCII, not arrows: the shipped UI font has no U+2191/U+2193 and draws them as tofu
            // boxes, so `^7m` beats a glyph nobody can read. Same reason the Assets tab paints its
            // own disclosure triangles instead of using U+25B8.
            text.push_str(&format!(" {}{}m", if c.dy > 0.0 { '^' } else { 'v' }, c.dy.abs().round() as i32));
        }
        if let Some(v) = c.value {
            text.push_str(&format!("  {}", short_value(v)));
        }
        let pos = egui::pos2(c.screen.x, c.screen.y);
        let galley = painter.layout_no_wrap(text, FontId::proportional(12.0), col);
        // Flip the plate to the left when it would overhang, so a label near the right edge stays
        // readable instead of being clipped mid-number.
        let flip = pos.x + 16.0 + galley.size().x > free.max.x;
        // A plate, always. Over live game pixels there is no guaranteed contrast, and text with a
        // bright frame behind it is text nobody can read.
        let plate_w = galley.size().x + 8.0;
        let r = egui::Rect::from_min_size(
            pos + egui::vec2(if flip { -8.0 - plate_w } else { 8.0 }, -galley.size().y * 0.5),
            galley.size() + egui::vec2(8.0, 4.0),
        );
        painter.rect_filled(r, 2.0, Color32::from_rgba_unmultiplied(11, 13, 16, 210));
        painter.rect_stroke(r, 2.0, Stroke::new(1.0, col), egui::StrokeKind::Inside);
        painter.galley(r.min + egui::vec2(4.0, 2.0), galley, col);
        // A dot ON the marker, so the label is attached to a place rather than floating.
        painter.circle_filled(pos, 2.5, col);
        painter.circle_stroke(pos, 3.5, Stroke::new(1.0, Color32::from_rgba_unmultiplied(11, 13, 16, 220)));
    }

    // What was thrown away, said out loud. A budget that hides things silently is a budget the
    // user cannot reason about — and "43 more" is itself information about where they are.
    if cfg.show_list && (!shown.is_empty() || hidden > 0) {
        let mut y = free.max.y - 96.0;
        let x = free.min.x + 10.0;
        if hidden > 0 {
            painter.text(
                egui::pos2(x, y + 74.0),
                Align2::LEFT_BOTTOM,
                format!("+{hidden} more within {}m", cfg.max_dist.round() as i32),
                FontId::proportional(11.0),
                Color32::from_rgb(122, 118, 108),
            );
        }
        // The five best, always in the same corner, so it can be read without hunting.
        for c in shown.iter().take(5) {
            let line = match c.value {
                Some(v) => format!("{}  {}m  {}", short_title(&c.title, 16), c.dist.round() as i32, short_value(v)),
                None => format!("{}  {}m", short_title(&c.title, 16), c.dist.round() as i32),
            };
            let g = painter.layout_no_wrap(line, FontId::proportional(12.0), Color32::from_rgb(214, 208, 196));
            let r = egui::Rect::from_min_size(egui::pos2(x, y), g.size() + egui::vec2(8.0, 3.0));
            painter.rect_filled(r, 2.0, Color32::from_rgba_unmultiplied(11, 13, 16, 190));
            painter.galley(r.min + egui::vec2(4.0, 1.0), g, Color32::from_rgb(214, 208, 196));
            y += 16.0;
        }
    }
}
