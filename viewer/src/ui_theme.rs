//! ui_theme.rs — the SINGLE source of truth for the viewer's UI design language.
//!
//! Every panel (menu.rs, ui.rs, tasks_panel.rs, inspect.rs, menu_fx.rs) consumes the tokens and
//! widget helpers here instead of re-declaring its own color/size constants. Before this module the
//! same handful of colors were copy-pasted per file and had DRIFTED (BONE was 215,211,203 in two
//! files but 208,200,178 in a third; DIM had three different values; CARD/BORDER/ACCENT/state colors
//! all forked). Reconciling them into one coherent palette is the whole point.
//!
//! ## Palette provenance (authentic EFT anchors)
//! The values below were reconciled against colors SAMPLED from Escape from Tarkov's own shipped UI
//! chrome — `EscapeFromTarkov_Data/StreamingAssets/Windows/tempsplash.bundle` (the game's 1024x768
//! loading screen, `tempSplash` Texture2D):
//!   * background field pixels cluster at **RGB(10,10,10)** (8,8,8 .. 16,16,16) -> [`BG`] / near-black.
//!   * the brightest wordmark pixels average **RGB(234,232,223)** -> [`TEXT_BRIGHT`] (warm bone).
//!   * EVERY neutral EFT samples carries a warm bias (R >= G > B): 56,56,48 / 64,64,56 / 234,232,223.
//!     So all greys/text here lean warm — the single biggest correction vs. the old cool-grey drift.
//! EFT's own UI architecture (from `globalgamemanagers.assets` MonoScript class names) is a
//! centralized `ColorPalette`/`ColorScheme` feeding skinnable panels (`AbstractSkin`) and square
//! `DefaultUiButtonNewStyle` buttons — i.e. exactly the flat, hard-cornered, one-palette model this
//! module implements. The interface sprite atlases + TMP font ("Bender" family, uppercase + wide
//! tracking) are runtime-downloaded and not in the base install, so their exact chrome pixels aren't
//! extractable; the tokens are the project's prior EFT-tuned values reconciled toward the authentic
//! anchors above. NO game asset ships in the exe — all icons here are painter-drawn (procedural).
//!
//! egui 0.32 via bevy_egui 0.37. Everything is square-cornered (EFT UI is all hard corners).
//! ASCII + the whitelisted glyphs only (x -> `\u{00D7}`, dot `\u{25CF}`, etc).

#![cfg(feature = "egui")]
// A design-system module deliberately exposes a COMPLETE, coherent token vocabulary (the full type
// and spacing scale, every state color, the widget helpers) so panels reach for the right token
// instead of a literal. Not every token is consumed on day one; that is by design, not dead weight.
#![allow(dead_code)]

use bevy_egui::egui::{self, Color32, CornerRadius, Margin, Response, RichText, Stroke, StrokeKind};

// ============================================================================================
// COLOR TOKENS — the authoritative palette. Each documents its ROLE. Warm-neutral (R >= G > B).
// ============================================================================================

// ---- Surfaces (near-black field -> charcoal panels; darkest chrome first) ----
/// App background field, behind every panel. Authentic EFT loading-screen field (sampled 10,10,10).
pub const BG: Color32 = Color32::from_rgb(10, 10, 10);
/// Header bars + the vertical toolbar rail — the darkest interactive chrome (one step above [`BG`]).
pub const RAIL: Color32 = Color32::from_rgb(16, 16, 15);
/// Side-panel body fill + the menu's map-row base. The dominant surface the eye reads content on.
pub const PANEL: Color32 = Color32::from_rgb(20, 20, 19);
/// Raised card / task row fill — one value lighter than [`PANEL`] so cards lift off the panel.
pub const CARD: Color32 = Color32::from_rgb(25, 25, 23);
/// Hover / active-tab fill (rail active swatch, row hover) — lighter still, still charcoal.
pub const CARD_HOVER: Color32 = Color32::from_rgb(33, 32, 29);
/// Inset wells: text-edit backgrounds, the loader's empty segments (darker than [`PANEL`]).
pub const INSET: Color32 = Color32::from_rgb(14, 14, 13);
/// Inspect billboard fill (translucent so the 3D scene reads behind it). Warm charcoal at ~92%
/// (premultiplied form — the only const-constructible rgba; reads like RAIL charcoal over the scene).
pub const CARD_TRANSLUCENT: Color32 = Color32::from_rgba_premultiplied(16, 16, 15, 236);
/// HUD readout background (top-left pose readout) — black at ~60% so it never fights the scene.
pub const HUD_BG: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 153);

// ---- Borders (warm steel; 1px everywhere) ----
/// Standard 1px card/panel border (sampled EFT steel ~48,47,44, warmed).
pub const BORDER: Color32 = Color32::from_rgb(47, 46, 42);
/// Emphasized border / the loader-bar frame — a brighter steel for structural emphasis.
pub const BORDER_STRONG: Color32 = Color32::from_rgb(70, 68, 61);
/// Faint inner dividers + separators (a hair above [`PANEL`]); the loader segment edges.
pub const SEAM: Color32 = Color32::from_rgb(33, 32, 29);

// ---- Text (warm bone ramp: brightest -> faint) ----
/// Brightest text: headings, live values, the menu wordmark. Authentic EFT wordmark ~234,232,223.
pub const TEXT_BRIGHT: Color32 = Color32::from_rgb(228, 226, 217);
/// Primary body text ("bone"). Reconciles the old 215,211,203 / 208,200,178 fork into one warm bone.
pub const BONE: Color32 = Color32::from_rgb(212, 208, 196);
/// Interactive beige/gold: the PLAY button, the active toolbar tab, key values. Warm gold-beige.
pub const BEIGE: Color32 = Color32::from_rgb(199, 178, 153);
/// Section-header text (collapsing-header titles, trader groups) — warm mid-grey.
pub const SECTION: Color32 = Color32::from_rgb(156, 154, 144);
/// Secondary labels / hints / muted copy (one dim level; warm).
pub const MUTED: Color32 = Color32::from_rgb(122, 118, 108);
/// Faintest text: marker counts, footnotes/provenance, disabled/off states. Dim warm grey, but kept
/// bright enough to stay legible on charcoal at 9-10pt (was 92,89,82 — too low-contrast for counts).
pub const FAINT: Color32 = Color32::from_rgb(108, 104, 95);

// ---- Accent + state (semantic; used CONSISTENTLY for the same meaning everywhere) ----
/// Primary accent: amber-gold. Panel titles ("MAP LAYERS"/"TASKS"), highlights, in-progress hints.
pub const ACCENT: Color32 = Color32::from_rgb(232, 194, 122);
/// State: ready / done / success / found-in-raid / server-running. Muted tactical green.
pub const OK: Color32 = Color32::from_rgb(127, 178, 108);
/// State: caution / "game files updated" / update-available. Amber-orange.
pub const WARN: Color32 = Color32::from_rgb(200, 140, 50);
/// State: destructive / failed. Muted brick red (the DELETE button fill).
pub const DANGER: Color32 = Color32::from_rgb(176, 65, 62);
/// Brighter red for error TEXT on the dark field (the plain [`DANGER`] is too dark to read as text).
pub const DANGER_TEXT: Color32 = Color32::from_rgb(206, 96, 84);

// ---- Special accents (item / quest semantics) ----
/// Keycards + quest-pickup items. Matches the on-map violet marker (poi Color::srgb 0.72,0.45,0.92).
pub const VIOLET: Color32 = Color32::from_rgb(184, 115, 235);
/// Tracked-task highlight (the quest-purple used when a task is being tracked).
pub const TRACKED: Color32 = Color32::from_rgb(150, 138, 232);
/// Kappa-required tag (deep gold).
pub const KAPPA: Color32 = Color32::from_rgb(212, 175, 95);
/// Lightkeeper-required tag (cyan).
pub const CYAN: Color32 = Color32::from_rgb(120, 200, 210);

// ============================================================================================
// TYPE SCALE — every font size the UI uses (pt). No panel should hardcode a raw size.
// ============================================================================================
/// Menu wordmark ("ATLAS").
pub const SIZE_DISPLAY: f32 = 22.0;
/// Menu map-row title (per-map name).
pub const SIZE_ROW_TITLE: f32 = 18.0;
/// Panel titles ("MAP LAYERS", "TASKS", "CAMERA").
pub const SIZE_TITLE: f32 = 16.0;
/// Card titles (inspect billboard title, task-card name).
pub const SIZE_CARD_TITLE: f32 = 15.0;
/// Emphasized body / a bold row label.
pub const SIZE_BODY: f32 = 13.0;
/// Standard control labels (checkboxes, combo captions).
pub const SIZE_LABEL: f32 = 12.0;
/// Small labels / secondary values / section-header size.
pub const SIZE_SMALL: f32 = 11.0;
/// Captions, counts, tags.
pub const SIZE_CAPTION: f32 = 10.0;
/// Footnotes / provenance / the smallest hint text.
pub const SIZE_TINY: f32 = 9.0;

// ============================================================================================
// SPACING SCALE — the vertical/horizontal rhythm. xs/sm/md/lg + the standard margins.
// ============================================================================================
pub const SP_XS: f32 = 2.0;
pub const SP_SM: f32 = 4.0;
pub const SP_MD: f32 = 8.0;
pub const SP_LG: f32 = 14.0;
/// Standard inter-widget spacing (x, y) — the panel rhythm.
pub const ITEM_SPACING: egui::Vec2 = egui::Vec2::new(8.0, 6.0);
/// Side-panel inner margin.
pub const MARGIN_PANEL: i8 = 14;
/// Card / row inner margin.
pub const MARGIN_CARD: i8 = 8;

// ============================================================================================
// GLOBAL STYLE — set ONCE per frame so egui's own defaults already match the theme (square
// corners, spacing, selection color, widget fills + text colors). Called from the first UI
// system in each mode (menu_ui + toolbar_panel). Modifies the existing (dark) style in place —
// it never REPLACES the Style (replacing with a fresh default flips egui to its light theme and
// paints a fullscreen pale layer, the historical bug).
// ============================================================================================
/// Install an INTENTIONAL tactical UI font (once) — not egui's rounded default. Prefers Bahnschrift
/// (techy condensed DIN), falling through common Windows faces, then egui's built-in if none load.
/// Only the Proportional family is replaced; Monospace stays default (the install-path / build log).
static FONTS_INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn install_fonts_once(ctx: &egui::Context) {
    use std::sync::atomic::Ordering;
    if FONTS_INSTALLED.swap(true, Ordering::Relaxed) {
        return;
    }
    use egui::{FontData, FontDefinitions, FontFamily};
    for path in [
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            let mut fonts = FontDefinitions::default();
            fonts
                .font_data
                .insert("atlas_ui".to_owned(), std::sync::Arc::new(FontData::from_owned(bytes)));
            fonts
                .families
                .entry(FontFamily::Proportional)
                .or_default()
                .insert(0, "atlas_ui".to_owned());
            ctx.set_fonts(fonts);
            return;
        }
    }
}

pub fn apply_global_style(ctx: &egui::Context) {
    install_fonts_once(ctx);
    ctx.style_mut(|s| {
        // ---- hard corners everywhere (EFT UI has no rounded widgets) ----
        let z = CornerRadius::ZERO;
        for w in [
            &mut s.visuals.widgets.noninteractive,
            &mut s.visuals.widgets.inactive,
            &mut s.visuals.widgets.hovered,
            &mut s.visuals.widgets.active,
            &mut s.visuals.widgets.open,
        ] {
            w.corner_radius = z;
        }
        s.visuals.window_corner_radius = z;
        s.visuals.menu_corner_radius = z;

        // ---- rhythm ----
        s.spacing.item_spacing = ITEM_SPACING;
        s.spacing.button_padding = egui::vec2(6.0, 3.0);

        // ---- surfaces (dark; NEVER light — see the note above) ----
        s.visuals.panel_fill = PANEL;
        s.visuals.window_fill = PANEL;
        s.visuals.window_stroke = Stroke::new(1.0, BORDER);
        s.visuals.extreme_bg_color = INSET; // text-edit / slider trough wells
        s.visuals.faint_bg_color = CARD;

        // ---- widget fills + text so bare ui.button()/checkbox()/label() already match ----
        let v = &mut s.visuals.widgets;
        v.noninteractive.bg_fill = PANEL;
        v.noninteractive.weak_bg_fill = PANEL;
        v.noninteractive.fg_stroke = Stroke::new(1.0, BONE); // default label text = bone
        v.noninteractive.bg_stroke = Stroke::new(1.0, BORDER); // separators / dividers (visible steel)

        v.inactive.bg_fill = CARD; // button rest
        v.inactive.weak_bg_fill = CARD;
        v.inactive.bg_stroke = Stroke::new(1.0, BORDER);
        v.inactive.fg_stroke = Stroke::new(1.0, BONE);

        v.hovered.bg_fill = CARD_HOVER;
        v.hovered.weak_bg_fill = CARD_HOVER;
        v.hovered.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
        v.hovered.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);

        v.active.bg_fill = CARD_HOVER;
        v.active.weak_bg_fill = CARD_HOVER;
        v.active.bg_stroke = Stroke::new(1.0, BEIGE);
        v.active.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);

        // ---- selection = subtle beige wash (text selection + selected selectable_label) ----
        s.visuals.selection.bg_fill = Color32::from_rgba_premultiplied(70, 63, 48, 120);
        s.visuals.selection.stroke = Stroke::new(1.0, BEIGE);
    });
}

// ============================================================================================
// WIDGET HELPERS — the shared building blocks. Every panel draws cards/headers/buttons/chips/
// swatches THROUGH these so they look and behave identically.
// ============================================================================================

/// A side-panel frame (square, [`PANEL`] fill, [`MARGIN_PANEL`] inner margin). The tabbed content
/// panels pass this to their `SidePanel`.
pub fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .inner_margin(Margin::same(MARGIN_PANEL))
        .corner_radius(0.0)
}

/// Panel title row text (accent, [`SIZE_TITLE`], bold) — "MAP LAYERS" / "TASKS" / "CAMERA".
pub fn title(text: &str) -> RichText {
    RichText::new(text).color(ACCENT).size(SIZE_TITLE).strong()
}

/// Section-header text for a `CollapsingHeader` (or a plain section label): NAME plus a faint
/// `\u{00B7} count` suffix when the section holds anything. Warm [`SECTION`] grey, bold.
pub fn section_header(name: &str, count: usize) -> RichText {
    let s = if count > 0 {
        format!("{name}   \u{00B7} {count}")
    } else {
        name.to_string()
    };
    RichText::new(s).color(SECTION).size(SIZE_LABEL).strong()
}

/// A charcoal card: square [`CARD`] fill + 1px [`BORDER`], [`MARGIN_CARD`] inner margin. `border`
/// lets a caller emphasize (e.g. a tracked task uses [`TRACKED`]); pass [`BORDER`] for the default.
pub fn card<R>(
    ui: &mut egui::Ui,
    border: Color32,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0, border))
        .inner_margin(Margin::same(MARGIN_CARD))
        .corner_radius(0.0)
        .show(ui, add)
}

/// A filled action button (square). `fill`/`text_color` pick the variant; see [`primary_button`] /
/// [`danger_button`] / [`warn_button`]. Returns the builder so callers can `add` or `add_sized`.
pub fn button_filled(text: &str, fill: Color32, text_color: Color32) -> egui::Button<'static> {
    // A bright rim (the fill lightened) crisply defines each button against the now-translucent menu
    // + the glowing globe behind it, so DELETE/PLAY/UPDATE stay clearly readable.
    let rim = Color32::from_rgb(
        (fill.r() as u16 + 78).min(255) as u8,
        (fill.g() as u16 + 78).min(255) as u8,
        (fill.b() as u16 + 78).min(255) as u8,
    );
    egui::Button::new(
        RichText::new(text.to_owned()).color(text_color).strong().size(15.5).extra_letter_spacing(0.6),
    )
    .fill(fill)
    .stroke(Stroke::new(1.5, rim))
    .corner_radius(0.0)
}

/// The primary affirmative button — beige fill, black text (the menu PLAY button).
pub fn primary_button(text: &str) -> egui::Button<'static> {
    button_filled(text, BEIGE, Color32::BLACK)
}

/// The destructive button — brick-red fill, black text (the menu DELETE / CONFIRM button).
pub fn danger_button(text: &str) -> egui::Button<'static> {
    button_filled(text, DANGER, Color32::BLACK)
}

/// The caution button — amber fill, black text (the menu UPDATE button).
pub fn warn_button(text: &str) -> egui::Button<'static> {
    button_filled(text, WARN, Color32::BLACK)
}

/// A small text chip: [`SIZE_CAPTION`] label in `color`, used where an item icon is unavailable and
/// for inline tags. (Kept text-only; the caller owns any surrounding layout.)
pub fn chip(ui: &mut egui::Ui, text: &str, color: Color32) -> Response {
    ui.label(RichText::new(text).size(SIZE_CAPTION).color(color))
}

/// A label/value stat pair: MUTED caption label + a BRIGHT value beside it (the HUD + menu rows).
pub fn stat_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(SIZE_SMALL).color(MUTED));
        ui.label(RichText::new(value).size(SIZE_BODY).color(TEXT_BRIGHT));
    });
}

/// A right-aligned faint marker count for a row (no-op when zero). Consistent density readout.
pub fn count_tag(ui: &mut egui::Ui, n: usize) {
    if n == 0 {
        return;
    }
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.label(RichText::new(n.to_string()).size(SIZE_CAPTION).color(FAINT));
    });
}

/// A filled swatch dot (`\u{25CF}`) in `color` at [`SIZE_LABEL`] — the on-map legend key in-panel.
pub fn swatch(ui: &mut egui::Ui, color: Color32) -> Response {
    ui.label(RichText::new("\u{25CF}").color(color).size(SIZE_LABEL))
}

/// A data-completeness tick label (menu): [`OK`] when present, [`FAINT`] when absent.
pub fn tick(on: bool, text: &str) -> RichText {
    RichText::new(text).size(SIZE_SMALL).color(if on { OK } else { FAINT })
}

/// Vivid, distinct legend color per loot class (doubles as the on-map marker key). Moved here so the
/// panel + any future legend read the SAME table.
pub fn loot_class_color(cls: &str) -> Color32 {
    match cls {
        "weapon" => Color32::from_rgb(214, 92, 72),
        "medical" => Color32::from_rgb(92, 200, 122),
        "safe" => Color32::from_rgb(235, 190, 74),
        "register" => Color32::from_rgb(84, 162, 235),
        "bag" => Color32::from_rgb(205, 150, 92),
        "crate" => Color32::from_rgb(196, 162, 108),
        "tech" => Color32::from_rgb(176, 112, 226),
        "furniture" => Color32::from_rgb(162, 138, 116),
        "stash" => Color32::from_rgb(150, 150, 150),
        "body" => Color32::from_rgb(222, 74, 74),
        _ => Color32::from_rgb(180, 180, 180),
    }
}

/// egui swatch color for a POI layer — derived from `poi::poi_look` (READ-only) so the in-panel dot
/// matches the on-map marker exactly.
pub fn poi_color(l: crate::poi::PoiLayer) -> Color32 {
    color32(crate::poi::poi_look(l).0)
}

/// bevy `Color` -> egui `Color32` (sRGB bytes). Shared so every marker->swatch conversion matches.
pub fn color32(color: bevy::prelude::Color) -> Color32 {
    let s = color.to_srgba();
    Color32::from_rgb(
        (s.red * 255.0) as u8,
        (s.green * 255.0) as u8,
        (s.blue * 255.0) as u8,
    )
}

// ============================================================================================
// TOOLBAR RAIL — the vertical icon rail buttons. Vector icons drawn with the painter (no assets).
// ============================================================================================

/// One 32x32 rail icon button. Draws the active/hover background, paints the vector icon in the
/// active-beige / idle-muted color, and returns true on click. `kind`: 0 = eye (visibility),
/// 1 = camera, 2 = tasks/checklist, 3 = navigation (route pin).
pub fn rail_button(ui: &mut egui::Ui, active: bool, kind: u8, tip: &str) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::click());
    // Idle icons sit at SECTION grey (brighter than the old dim MUTED): the rail is easy to miss, so
    // keep all three tabs clearly visible; the ACTIVE one gets beige + a bg fill + an indicator bar.
    let col = if active { BEIGE } else { SECTION };
    if active {
        ui.painter().rect_filled(rect, 0.0, CARD_HOVER);
        // Active-tab indicator: a beige bar on the panel-facing (left) edge, so the rail reads as a
        // tab strip and the current tab is obvious at a glance (the #1 discoverability fix).
        let bar = egui::Rect::from_min_max(rect.left_top(), egui::pos2(rect.left() + 2.5, rect.bottom()));
        ui.painter().rect_filled(bar, 0.0, BEIGE);
    } else if resp.hovered() {
        ui.painter().rect_filled(rect, 0.0, CARD);
    }
    paint_tool_icon(ui.painter(), rect, kind, col);
    resp.on_hover_text(tip).clicked()
}

/// Vector icon inside `rect`: 0 = eye, 1 = camera, 2 = tasks/checklist, 3 = navigation (a location
/// pin with a dashed route leading to it). Painter primitives only (no image assets — keeps the
/// shippable exe free of game-derived art).
pub fn paint_tool_icon(painter: &egui::Painter, rect: egui::Rect, kind: u8, c: Color32) {
    let ctr = rect.center();
    let s = Stroke::new(1.6, c);
    match kind {
        3 => {
            // navigation: dashed route from bottom-left up to a location pin at top-right
            let p = |x: f32, y: f32| egui::pos2(ctr.x + x, ctr.y + y);
            for (a, b) in [((-9.0, 9.0), (-6.0, 7.0)), ((-4.0, 5.5), (-1.0, 3.5)), ((1.0, 2.0), (3.0, 0.5))] {
                painter.line_segment([p(a.0, a.1), p(b.0, b.1)], Stroke::new(1.5, c));
            }
            // pin: head ring + tail down to its tip
            painter.circle_stroke(p(4.5, -5.0), 3.6, s);
            painter.circle_filled(p(4.5, -5.0), 1.2, c);
            painter.line_segment([p(1.6, -2.6), p(4.5, 1.2)], s);
            painter.line_segment([p(7.4, -2.6), p(4.5, 1.2)], s);
            return;
        }
        _ => {}
    }
    match kind {
        0 => {
            // eye: lens ellipse outline + pupil
            let pts: Vec<egui::Pos2> = (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.0 * std::f32::consts::TAU;
                    egui::pos2(ctr.x + t.cos() * 9.0, ctr.y + t.sin() * 5.0)
                })
                .collect();
            painter.add(egui::Shape::closed_line(pts, s));
            painter.circle_filled(ctr, 2.6, c);
        }
        1 => {
            // camera: body + top viewfinder bump + lens
            let body = egui::Rect::from_center_size(ctr, egui::vec2(20.0, 13.0));
            painter.rect_stroke(body, 1.0, s, StrokeKind::Middle);
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(ctr.x - 5.0, body.top() - 3.0),
                    egui::vec2(7.0, 3.0),
                ),
                0.0,
                c,
            );
            painter.circle_stroke(ctr, 4.2, s);
            painter.circle_filled(egui::pos2(body.right() - 2.5, body.top() + 2.5), 1.0, c);
        }
        4 => {
            // home / back-to-menu: a simple house (roof + body + door)
            let p = |x: f32, y: f32| egui::pos2(ctr.x + x, ctr.y + y);
            painter.line_segment([p(-9.0, -0.5), p(0.0, -8.5)], s);
            painter.line_segment([p(0.0, -8.5), p(9.0, -0.5)], s);
            painter.rect_stroke(
                egui::Rect::from_min_max(p(-7.0, -0.5), p(7.0, 9.0)),
                0.0,
                s,
                StrokeKind::Middle,
            );
            painter.rect_stroke(
                egui::Rect::from_min_max(p(-2.5, 3.5), p(2.5, 9.0)),
                0.0,
                Stroke::new(1.3, c),
                StrokeKind::Middle,
            );
        }
        _ => {
            // tasks: three checklist rows, first with a check
            for r in 0..3 {
                let y = rect.top() + 9.0 + r as f32 * 7.5;
                let bx = egui::Rect::from_min_size(
                    egui::pos2(rect.left() + 5.0, y - 2.5),
                    egui::vec2(5.0, 5.0),
                );
                painter.rect_stroke(bx, 0.0, Stroke::new(1.3, c), StrokeKind::Middle);
                if r == 0 {
                    painter.line_segment(
                        [egui::pos2(bx.left() + 1.0, y), egui::pos2(bx.center().x, bx.bottom() - 1.0)],
                        Stroke::new(1.3, c),
                    );
                    painter.line_segment(
                        [egui::pos2(bx.center().x, bx.bottom() - 1.0), egui::pos2(bx.right(), bx.top())],
                        Stroke::new(1.3, c),
                    );
                }
                painter.line_segment(
                    [egui::pos2(bx.right() + 3.0, y), egui::pos2(rect.right() - 5.0, y)],
                    Stroke::new(1.4, if r == 0 { c } else { FAINT }),
                );
            }
        }
    }
}

// ============================================================================================
// OBJECTIVE GLYPHS — tiny vector icons for the Tasks panel's subtask gutter, one per objective
// TYPE, so an objective with no cached item art still reads at a glance (KILL = crosshair, EXFIL =
// door+arrow, MARK = flag, GO TO = pin, HAND IN / PLANT = deposit, FIND / PICK UP = magnifier).
// Painter primitives only (no assets), matching `paint_tool_icon`. `kind` is the raw tasks.json
// objective type (same strings `tasks_panel::obj_tag` switches on).
// ============================================================================================

/// A centred completion check (used when an objective is marked done — replaces its gutter icon).
pub fn paint_check(painter: &egui::Painter, rect: egui::Rect, c: Color32) {
    let ctr = rect.center();
    let s = Stroke::new(2.2, c);
    painter.line_segment(
        [egui::pos2(ctr.x - 6.0, ctr.y + 0.5), egui::pos2(ctr.x - 1.5, ctr.y + 5.5)],
        s,
    );
    painter.line_segment(
        [egui::pos2(ctr.x - 1.5, ctr.y + 5.5), egui::pos2(ctr.x + 7.0, ctr.y - 6.0)],
        s,
    );
}

/// A ~18px objective-type glyph centred in `rect`, drawn in `c`.
pub fn paint_obj_glyph(painter: &egui::Painter, rect: egui::Rect, kind: &str, c: Color32) {
    let ctr = rect.center();
    let s = Stroke::new(1.5, c);
    let p = |x: f32, y: f32| egui::pos2(ctr.x + x, ctr.y + y);
    match kind {
        // KILL — crosshair (ring + 4 ticks).
        "shoot" => {
            painter.circle_stroke(ctr, 5.5, s);
            for (a, b) in [((-9.0, 0.0), (-5.5, 0.0)), ((9.0, 0.0), (5.5, 0.0)),
                           ((0.0, -9.0), (0.0, -5.5)), ((0.0, 9.0), (0.0, 5.5))] {
                painter.line_segment([p(a.0, a.1), p(b.0, b.1)], s);
            }
            painter.circle_filled(ctr, 1.2, c);
        }
        // EXFIL — a doorway with an arrow leaving it.
        "extract" => {
            painter.rect_stroke(
                egui::Rect::from_center_size(p(-3.0, 0.0), egui::vec2(8.0, 15.0)),
                0.0, s, StrokeKind::Middle,
            );
            painter.line_segment([p(-1.0, 0.0), p(8.0, 0.0)], s);
            painter.line_segment([p(4.0, -3.5), p(8.0, 0.0)], s);
            painter.line_segment([p(4.0, 3.5), p(8.0, 0.0)], s);
        }
        // MARK — a pennant flag.
        "mark" => {
            painter.line_segment([p(-6.0, -8.0), p(-6.0, 8.0)], s);
            painter.add(egui::Shape::convex_polygon(
                vec![p(-6.0, -8.0), p(7.0, -4.0), p(-6.0, 0.0)],
                c,
                Stroke::NONE,
            ));
        }
        // GO TO / VISIT — a map pin.
        "visit" => {
            painter.circle_stroke(p(0.0, -3.0), 5.0, s);
            painter.circle_filled(p(0.0, -3.0), 1.6, c);
            painter.line_segment([p(-3.4, 0.4), p(0.0, 8.5)], s);
            painter.line_segment([p(3.4, 0.4), p(0.0, 8.5)], s);
        }
        // HAND IN — down-arrow into an inbox tray.
        "giveItem" | "giveQuestItem" => {
            painter.line_segment([p(-7.0, 7.0), p(7.0, 7.0)], s);
            painter.line_segment([p(-7.0, 3.0), p(-7.0, 7.0)], s);
            painter.line_segment([p(7.0, 3.0), p(7.0, 7.0)], s);
            painter.line_segment([p(0.0, -8.0), p(0.0, 3.0)], s);
            painter.line_segment([p(-3.5, -0.5), p(0.0, 3.0)], s);
            painter.line_segment([p(3.5, -0.5), p(0.0, 3.0)], s);
        }
        // PLANT — down-arrow into a box.
        "plantItem" | "plantQuestItem" => {
            painter.rect_stroke(
                egui::Rect::from_center_size(p(0.0, 4.0), egui::vec2(14.0, 9.0)),
                0.0, s, StrokeKind::Middle,
            );
            painter.line_segment([p(0.0, -9.0), p(0.0, -1.0)], s);
            painter.line_segment([p(-3.0, -4.0), p(0.0, -1.0)], s);
            painter.line_segment([p(3.0, -4.0), p(0.0, -1.0)], s);
        }
        // FIND / PICK UP — a magnifier.
        "findItem" | "findQuestItem" => {
            painter.circle_stroke(p(-2.0, -2.0), 5.0, s);
            painter.line_segment([p(1.8, 1.8), p(7.0, 7.0)], Stroke::new(2.0, c));
        }
        // BUILD — a small bolt/gear dot.
        "buildWeapon" => {
            painter.circle_stroke(ctr, 5.5, s);
            for k in 0..6 {
                let a = k as f32 / 6.0 * std::f32::consts::TAU;
                painter.line_segment(
                    [p(a.cos() * 5.5, a.sin() * 5.5), p(a.cos() * 8.0, a.sin() * 8.0)],
                    s,
                );
            }
        }
        // Everything else (SKILL / TRADER / QUEST / SELL / USE / XP / DO) — a neutral ringed dot.
        _ => {
            painter.circle_stroke(ctr, 4.5, s);
            painter.circle_filled(ctr, 1.6, c);
        }
    }
}
