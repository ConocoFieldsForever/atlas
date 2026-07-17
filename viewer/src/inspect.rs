//! inspect.rs — CLICK-TO-INSPECT billboard cards for overlay markers.
//!
//! Left-click a loot cube (loot.rs) or a POI sphere (poi.rs) and a floating info
//! card pops up, anchored in screen space over that marker, naming what it is; a
//! "\u{00D7}" in the card's corner dismisses it. Several cards can be open at once —
//! each fresh marker click opens another, each "\u{00D7}" closes only its own. Clicking
//! empty space or the terrain does NOT clear cards (this is a sticky annotation
//! tool, not a hover tooltip).
//!
//! The flow is split across two systems on two schedules on purpose:
//!   * System A `pick_markers` (Update, NOT egui) casts a world ray from the cursor
//!     against each marker's world hit-sphere (`PickRadius`) — the SAME ray_sphere
//!     idiom as pick.rs — and pushes the nearest VISIBLE marker into `OpenCards`.
//!     It bails when the egui pointer is over any panel (`PointerOnUi`) so a click
//!     on the layers panel never leaks through to the world.
//!   * System B `draw_cards` (EguiPrimaryContextPass) projects each open marker to
//!     the screen with `Camera::world_to_viewport` and draws its egui billboard.
//!     It MUST run in `EguiPrimaryContextPass`: in plain `Update` the egui context
//!     has no fonts yet and `ctx_mut()` panics (bevy_egui 0.37). It also publishes
//!     `PointerOnUi` for the next frame's pick to read.
//!
//! `MarkerInfo` / `PickRadius` live here and are attached to markers at spawn time
//! in loot.rs / poi.rs; the small text helpers (`money`, `prettify`, `titlecase`)
//! are shared from here so those spawners can format card copy without egui.

use crate::render::CullCamera;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

/// The contents of one info card. Attached to every clickable marker at spawn.
/// Read only by the egui `draw_cards`; without the `egui` feature the fields are
/// still written by the spawners but never displayed (hence the gated allow).
#[derive(Component, Clone)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct MarkerInfo {
    pub title: String,
    pub subtitle: String,
    pub detail: Vec<String>,
    pub accent: Color,
}

/// World-space hit-sphere radius for the click raycast, set at spawn (clamped up a
/// little for tiny markers so doors/interactables are still easy to hit).
#[derive(Component)]
pub struct PickRadius(pub f32);

/// Icon SLUG for the card's title row — the basename of `<pack>/icons/<slug>.png`, cached
/// there at build time by extraction/intel/fetch_icons.py (the client ships no icon sprites;
/// it renders inventory icons at runtime from the item's 3D prefab, so the pack carries small
/// tarkov.dev PNGs instead and the viewer stays offline). Attached by poi.rs to loose-loot
/// and lock markers; a missing file is silently no-icon. Kept even without the `egui` feature
/// so the spawners compile unchanged (only `draw_cards` reads it).
#[derive(Component)]
#[cfg_attr(not(feature = "egui"), allow(dead_code))]
pub struct MarkerIcon(pub String);

/// The `<slug>.png` name contract, shared with fetch_icons.py `slug()`: lowercase, ASCII
/// alphanumerics pass through, every other run of chars collapses to one '-', trimmed.
pub fn icon_slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut dash = false;
    for c in name.chars() {
        let l = c.to_ascii_lowercase();
        if l.is_ascii_alphanumeric() {
            out.push(l);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Lazy per-slug icon texture cache for the cards. `None` entries memoize a miss (no file /
/// bad PNG) so the disk is probed at most once per slug; the egui `TextureHandle`s keep their
/// textures alive for the app's life (tens of 64-128 px icons — negligible).
#[cfg(feature = "egui")]
#[derive(Resource, Default)]
struct IconCache {
    /// `<pack>/icons`, resolved from `LoadedPack` on first use.
    root: Option<std::path::PathBuf>,
    /// `packs/shared/icons` — the cross-map icon store (icons are item-keyed, not map-keyed).
    shared: Option<std::path::PathBuf>,
    tex: std::collections::HashMap<String, Option<bevy_egui::egui::TextureHandle>>,
}

#[cfg(feature = "egui")]
impl IconCache {
    fn get(
        &mut self,
        ctx: &bevy_egui::egui::Context,
        pack: &Option<Res<crate::render::LoadedPack>>,
        slug: &str,
    ) -> Option<bevy_egui::egui::TextureHandle> {
        use bevy_egui::egui;
        if self.root.is_none() {
            self.root = pack.as_ref().map(|p| p.0.root.join("icons"));
            self.shared = pack
                .as_ref()
                .and_then(|p| p.0.root.parent())
                .map(|p| p.join("shared").join("icons"));
        }
        let root = self.root.as_ref()?;
        if let Some(hit) = self.tex.get(slug) {
            return hit.clone();
        }
        let file = format!("{slug}.png");
        let loaded = image::open(root.join(&file))
            .or_else(|_| {
                image::open(self.shared.as_ref().map(|s| s.join(&file)).unwrap_or_default())
            })
            .ok()
            .map(|img| {
                let rgba = img.into_rgba8();
                let (w, h) = rgba.dimensions();
                let ci = egui::ColorImage::from_rgba_unmultiplied(
                    [w as usize, h as usize],
                    rgba.as_raw(),
                );
                ctx.load_texture(format!("icon:{slug}"), ci, egui::TextureOptions::LINEAR)
            });
        self.tex.insert(slug.to_string(), loaded.clone());
        loaded
    }
}

/// The markers whose cards are currently open (entity ids). Deduped on insert;
/// "\u{00D7}" removes one; entries whose entity no longer resolves are dropped by
/// `draw_cards`.
#[derive(Resource, Default)]
struct OpenCards(Vec<Entity>);

/// True when the egui pointer is over ANY egui area (layers panel, stats window,
/// or a card). Set by `draw_cards`, read by `pick_markers` so world clicks that
/// land on UI don't also open/pick a marker behind it.
#[derive(Resource, Default)]
pub struct PointerOnUi(pub bool);

/// True while egui wants KEYBOARD input (marker search box, quest filters, sliders being typed
/// into) — the flycam must not fly when the user types 'wasd' into a text field.
#[derive(Resource, Default)]
pub struct UiWantsKeyboard(pub bool);

pub struct InspectPlugin;

impl Plugin for InspectPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OpenCards>()
            .init_resource::<PointerOnUi>()
            .init_resource::<UiWantsKeyboard>()
            .add_systems(Update, pick_markers);
        #[cfg(feature = "egui")]
        app.init_resource::<IconCache>();
        // The card UI is egui; it MUST live in EguiPrimaryContextPass (see module
        // doc). With the `egui` feature off there's simply no card UI, but the
        // component/resources still exist so loot.rs/poi.rs compile.
        #[cfg(feature = "egui")]
        app.add_systems(bevy_egui::EguiPrimaryContextPass, draw_cards);
        // Headless-QA aid: `EFT_INSPECT=<n>` auto-opens n cards a few frames in, so a
        // screenshot run (which can't click) can still show the billboards. Off by default.
        #[cfg(feature = "egui")]
        app.add_systems(Update, debug_autoselect);
    }
}

/// System A — cursor ray vs marker hit-spheres. On a left press (and only when the
/// pointer isn't over UI) select the nearest VISIBLE marker and open its card.
#[allow(clippy::type_complexity)]
fn pick_markers(
    mouse: Res<ButtonInput<MouseButton>>,
    pointer_on_ui: Res<PointerOnUi>,
    keys: Res<ButtonInput<KeyCode>>,
    place: Res<crate::pathfind::PlaceMode>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<CullCamera>>,
    markers: Query<(Entity, &GlobalTransform, &PickRadius, &ViewVisibility, &MarkerInfo)>,
    mut open: ResMut<OpenCards>,
) {
    if !mouse.just_pressed(MouseButton::Left) || pointer_on_ui.0 {
        return;
    }
    // A position-placement click (armed place mode, or the shift-click shortcut) must not ALSO
    // open a marker card — one click, one action (pick.rs owns that click).
    if place.0 || keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        return;
    };
    let ro: Vec3 = ray.origin;
    let rd: Vec3 = *ray.direction; // unit world direction

    // Nearest ray_sphere hit among markers whose layer is toggled ON and on-screen
    // (ViewVisibility is the resolved bool — NOT `Visibility`, which can be Inherited).
    let mut best: Option<(Entity, f32)> = None;
    for (e, tf, radius, view_vis, _info) in &markers {
        if !view_vis.get() {
            continue;
        }
        if let Some(t) = ray_sphere(ro, rd, tf.translation(), radius.0) {
            if best.map_or(true, |(_, bt)| t < bt) {
                best = Some((e, t));
            }
        }
    }

    // Hit -> open a card (deduped). Miss -> leave existing cards alone.
    if let Some((e, _)) = best {
        if !open.0.contains(&e) {
            open.0.push(e);
        }
    }
}

/// System B — draw one egui billboard per open card and publish `PointerOnUi`.
#[cfg(feature = "egui")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn draw_cards(
    mut contexts: bevy_egui::EguiContexts,
    cameras: Query<(&Camera, &GlobalTransform), With<CullCamera>>,
    markers: Query<(
        &GlobalTransform,
        &MarkerInfo,
        &PickRadius,
        Option<&crate::poi::MarkerValue>,
        Option<&MarkerIcon>,
    )>,
    mut open: ResMut<OpenCards>,
    mut pointer_on_ui: ResMut<PointerOnUi>,
    mut ui_kb: ResMut<UiWantsKeyboard>,
    mut route: MessageWriter<crate::pathfind::RouteRequest>,
    mut plan: ResMut<crate::ui::PlanList>,
    mut icons: ResMut<IconCache>,
    pack: Option<Res<crate::render::LoadedPack>>,
) {
    use bevy_egui::egui::{self, Align, Align2, Button, Layout, RichText};

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let Ok((camera, cam_tf)) = cameras.single() else {
        // No camera yet (first frame): still refresh the UI-hover flag and bail.
        pointer_on_ui.0 = ctx.is_pointer_over_area() || ctx.wants_pointer_input();
        ui_kb.0 = ctx.wants_keyboard_input();
        return;
    };

    let mut to_close: Vec<Entity> = Vec::new();
    for &e in open.0.iter() {
        let Ok((tf, info, radius, val, icon)) = markers.get(e) else {
            to_close.push(e); // marker despawned — drop its stale card
            continue;
        };
        // Icon beside the title (loose loot / lock keys); missing file = silently absent.
        let icon_tex = icon.and_then(|i| icons.get(ctx, &pack, &i.0));
        // Anchor above the marker so the card floats clear of the geometry.
        let world_pos = tf.translation() + Vec3::Y * radius.0.max(1.5);
        let Ok(screen) = camera.world_to_viewport(cam_tf, world_pos) else {
            continue; // behind the camera this frame — keep the card, skip drawing
        };

        let accent = color32(info.accent);
        egui::Area::new(egui::Id::new(("inspect_card", e.index())))
            .fixed_pos(egui::pos2(screen.x, screen.y))
            .pivot(Align2::CENTER_BOTTOM)
            .show(ctx, |ui| {
                // Square, warm-charcoal translucent billboard (was the ONE rounded card in the UI —
                // squared to match the EFT hard-corner design language). Tokens from ui_theme.
                let frame = egui::Frame::new()
                    .fill(crate::ui_theme::CARD_TRANSLUCENT)
                    .inner_margin(egui::Margin::same(10))
                    .corner_radius(0.0)
                    .stroke(egui::Stroke::new(1.0, crate::ui_theme::BORDER));
                frame.show(ui, |ui| {
                    ui.set_min_width(150.0);
                    ui.set_max_width(260.0);
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 3.0);

                    // Title (accent, bold) + a "\u{00D7}" dismiss pushed to the far right;
                    // the item icon (when the marker carries one) leads the row, scaled to
                    // a 22 px title-height square-ish thumb with its aspect kept.
                    ui.horizontal(|ui| {
                        if let Some(tex) = &icon_tex {
                            let sz = tex.size_vec2();
                            let s = 22.0 / sz.y.max(1.0);
                            ui.image((tex.id(), sz * s));
                        }
                        ui.label(
                            RichText::new(&info.title)
                                .color(accent)
                                .size(crate::ui_theme::SIZE_CARD_TITLE)
                                .strong(),
                        );
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let x = Button::new(
                                RichText::new("\u{00D7}").size(18.0).color(crate::ui_theme::SECTION),
                            )
                            .frame(false);
                            if ui.add(x).clicked() {
                                to_close.push(e);
                            }
                        });
                    });
                    // Subtitle (muted, smaller).
                    ui.label(
                        RichText::new(&info.subtitle)
                            .color(crate::ui_theme::MUTED)
                            .size(crate::ui_theme::SIZE_SMALL),
                    );
                    // Detail lines (bright bone, small).
                    for d in &info.detail {
                        ui.label(
                            RichText::new(d).color(crate::ui_theme::TEXT_BRIGHT).size(crate::ui_theme::SIZE_LABEL),
                        );
                    }
                    // One-click route from your position to this marker (in-process CPU A*), plus
                    // pin/unpin into the raid plan (the panel's "Raid plan" section lists the
                    // pins; ui::PlanList). Pin state keys off the marker ENTITY, so re-clicking
                    // the same marker toggles rather than duplicating.
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        if ui.small_button("route here").clicked() {
                            route.write(crate::pathfind::RouteRequest {
                                start: None,
                                dests: vec![tf.translation()],
                                optimize_order: false,
                                labels: vec![info.title.clone()],
                                ..Default::default()
                            });
                        }
                        let pinned = plan.pins.iter().any(|p| p.entity == e);
                        if ui
                            .small_button(if pinned { "unpin" } else { "pin" })
                            .on_hover_text("toggle in the Raid plan list (layers panel)")
                            .clicked()
                        {
                            if pinned {
                                plan.pins.retain(|p| p.entity != e);
                            } else {
                                plan.pins.push(crate::ui::PlanPin {
                                    entity: e,
                                    title: info.title.clone(),
                                    pos: tf.translation(),
                                    value: val.map(|v| v.0).unwrap_or(0),
                                });
                            }
                        }
                    });
                });
            });
    }

    // Drop closed ("\u{2715}") and unresolved cards.
    if !to_close.is_empty() {
        open.0.retain(|e| !to_close.contains(e));
    }

    // GLOBAL to the egui context, so it also covers the layers panel and the stats
    // window — exactly what the world raycast wants to ignore.
    pointer_on_ui.0 = ctx.is_pointer_over_area() || ctx.wants_pointer_input();
    ui_kb.0 = ctx.wants_keyboard_input();
}

/// Headless-QA aid: with `EFT_INSPECT` set, ~80 frames in (just before the frame-90
/// screenshot), open cards for the markers nearest screen-center that are on-screen &
/// visible — so a click-free screenshot run can still show the billboards. `EFT_INSPECT`
/// value is the card count (default 6). Runs once, then disables itself. No-op unset.
#[cfg(feature = "egui")]
fn debug_autoselect(
    mut frame: Local<u32>,
    mut done: Local<bool>,
    cameras: Query<(&Camera, &GlobalTransform), With<CullCamera>>,
    markers: Query<(Entity, &GlobalTransform, &ViewVisibility), With<MarkerInfo>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut open: ResMut<OpenCards>,
) {
    if *done {
        return;
    }
    *frame += 1;
    if *frame < 80 {
        return;
    }
    *done = true;
    let Ok(want) = std::env::var("EFT_INSPECT") else {
        return;
    };
    let n: usize = want.trim().parse().unwrap_or(6);
    let Ok((camera, cam_tf)) = cameras.single() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let (w, h) = (window.width(), window.height());
    let center = Vec2::new(w * 0.5, h * 0.5);
    let mut cands: Vec<(Entity, f32)> = Vec::new();
    for (e, tf, vv) in &markers {
        if !vv.get() {
            continue;
        }
        if let Ok(sp) = camera.world_to_viewport(cam_tf, tf.translation()) {
            if sp.x >= 0.0 && sp.x <= w && sp.y >= 0.0 && sp.y <= h {
                cands.push((e, sp.distance(center)));
            }
        }
    }
    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (e, _) in cands.into_iter().take(n) {
        if !open.0.contains(&e) {
            open.0.push(e);
        }
    }
    info!("inspect: EFT_INSPECT opened {} debug cards", open.0.len());
}

/// Ray vs sphere. `rd` MUST be unit length. Returns the nearest non-negative entry
/// `t` (0 if the origin is inside the sphere), or None if the sphere is behind the
/// ray or missed. Copied from pick.rs so the inspect raycast matches the debug pick.
#[inline]
fn ray_sphere(ro: Vec3, rd: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ro - center;
    let b = oc.dot(rd);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c; // a == 1 (rd unit)
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t0 = -b - sq;
    let t1 = -b + sq;
    if t1 < 0.0 {
        return None; // whole sphere behind the ray
    }
    Some(if t0 >= 0.0 { t0 } else { 0.0 })
}

/// bevy `Color` -> egui `Color32` (sRGB bytes). Delegates to the shared `ui_theme::color32` so every
/// marker->swatch/accent conversion in the app matches.
#[cfg(feature = "egui")]
fn color32(color: Color) -> bevy_egui::egui::Color32 {
    crate::ui_theme::color32(color)
}

/// Thousands-separated money string, e.g. 262500 -> "262,500" (sign preserved).
pub fn money(v: i64) -> String {
    let digits = v.unsigned_abs().to_string();
    let bytes = digits.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    if v < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// Prettify a raw GameObject name: strip a trailing `_LOD0`, turn `_` into spaces,
/// and collapse runs of whitespace. Deliberately simple.
pub fn prettify(name: &str) -> String {
    let base = name.strip_suffix("_LOD0").unwrap_or(name);
    base.replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Capitalize the first letter of each word (e.g. "killa" -> "Killa").
pub fn titlecase(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
