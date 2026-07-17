//! navigate_panel.rs — the NAVIGATION tab (right-panel router slot, 4th rail icon).
//!
//! The raid-planning flow in three steps, top to bottom:
//!   1. YOUR POSITION — one primary button arms a click-to-place mode; the next click on the map
//!      drops the gold "you are here" pin (pick.rs does the raycast; Esc cancels). No hotkeys, no
//!      modifier keys to remember — the button IS the affordance. While armed, a floating hint
//!      banner sits over the viewport so the mode is unmissable.
//!   2. EXTRACTS — a table of every extract on the map (faction-coloured, distance-annotated).
//!      Clicking a row routes from your position to that extract; NEAREST chains through all of
//!      them and takes the shortest. Rows work even while the Extracts overlay layer is hidden.
//!   3. ROUTE — the live route readout (walkable metres / computing / error) with a clear button.
//!
//! All colors/typography come from ui_theme (single source of truth). Routing itself is the
//! in-process CPU A* (nav.rs); this panel only writes `RouteRequest`s.

#![cfg(feature = "egui")]

use bevy::prelude::*;
use bevy_egui::egui::{self, Color32, RichText};

use crate::pathfind::{PlaceMode, RouteRequest, RouteResult, RouteStatus, ServerStatus, StartPoint};
use crate::poi::{PoiLayer, SceneInactive, ZoneWall};
use crate::render::CullCamera;
use crate::ui::RightPanelTab;
use crate::ui_theme as theme;

/// Panel-local state: the last extract row routed to (highlighted so "what did I click" is obvious).
#[derive(Default)]
pub struct NavUiState {
    selected: Option<Entity>,
}

/// One extract row, resolved from the marker entities each frame (cheap: a handful of extracts).
struct Row {
    entity: Entity,
    name: String,
    accent: Color32,
    pos: Vec3,
    dist: f32,
    inactive: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn navigate_tab(
    mut contexts: bevy_egui::EguiContexts,
    tab: Res<RightPanelTab>,
    menu: Option<Res<crate::menu::MenuState>>,
    server: Res<crate::pathfind::PathfindServer>,
    mut start_pt: ResMut<StartPoint>,
    mut place: ResMut<PlaceMode>,
    mut route: MessageWriter<RouteRequest>,
    route_result: Res<RouteResult>,
    mut cam_cmd: ResMut<crate::CameraCommand>,
    extracts: Query<
        (
            Entity,
            &PoiLayer,
            &GlobalTransform,
            &crate::inspect::MarkerInfo,
            Option<&SceneInactive>,
        ),
        Without<ZoneWall>,
    >,
    cams: Query<&Transform, With<CullCamera>>,
    mut ui_state: Local<NavUiState>,
) {
    if menu.is_some() {
        return; // start-menu mode owns the screen
    }
    // Leaving the tab keeps an armed place-mode live on purpose: you arm it, swing the camera,
    // click. But if the tab isn't active we still draw the armed banner so Esc/cancel stays
    // discoverable.
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    if *tab != RightPanelTab::Navigate {
        if place.0 {
            place_banner(ctx, &mut place);
        }
        return;
    }

    let ready = server.status == ServerStatus::Running;
    // Distance reference: the placed pin (stable), else the camera. Also decides row ordering —
    // with a placed pin the sort is by distance (stable + instantly useful); with the camera
    // fallback we sort by name so rows don't reshuffle while flying.
    let cam_pos = cams.single().map(|t| t.translation).unwrap_or(Vec3::ZERO);
    let ref_pos = start_pt.0.unwrap_or(cam_pos);

    let mut rows: Vec<Row> = extracts
        .iter()
        .filter(|(_, l, _, _, _)| **l == PoiLayer::Extract)
        .map(|(e, _, gt, info, inactive)| {
            let pos = gt.translation();
            Row {
                entity: e,
                name: pretty_name(&info.title),
                accent: theme::color32(info.accent),
                pos,
                dist: pos.distance(ref_pos),
                inactive: inactive.is_some(),
            }
        })
        .collect();
    if start_pt.0.is_some() {
        rows.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    } else {
        rows.sort_by(|a, b| a.name.cmp(&b.name));
    }

    egui::SidePanel::right("map_layers")
        .resizable(false)
        .frame(theme::panel_frame())
        .default_width(272.0)
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = theme::ITEM_SPACING;
            ui.label(theme::title("NAVIGATION"));
            ui.add_space(theme::SP_SM);

            // ---- no nav data: one clear warning, everything else still usable ----
            if !ready {
                theme::card(ui, theme::WARN, |ui| {
                    ui.label(
                        RichText::new("No route data for this map")
                            .size(theme::SIZE_LABEL)
                            .strong()
                            .color(theme::WARN),
                    );
                    ui.label(
                        RichText::new("Rebuild the map from the start menu to enable routing.")
                            .size(theme::SIZE_CAPTION)
                            .color(theme::MUTED),
                    );
                });
                ui.add_space(theme::SP_SM);
            }

            // ===== 1 · YOUR POSITION =====
            ui.label(theme::section_header("YOUR POSITION", 0));
            theme::card(ui, theme::BORDER_STRONG, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(6.0, 5.0);
                match start_pt.0 {
                    Some(p) => {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("\u{25CF}").color(GOLD).size(12.0));
                            ui.label(
                                RichText::new(format!("Placed at  {:.0}, {:.0}", p.x, p.z))
                                    .size(theme::SIZE_LABEL)
                                    .color(theme::TEXT_BRIGHT),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui
                                    .small_button(RichText::new("remove").size(10.0))
                                    .on_hover_text("routes start at the camera again")
                                    .clicked()
                                {
                                    start_pt.0 = None;
                                }
                            });
                        });
                    }
                    None => {
                        ui.label(
                            RichText::new("Not placed \u{2014} routes start at your camera")
                                .size(theme::SIZE_SMALL)
                                .color(theme::MUTED),
                        );
                    }
                }
                let full = egui::vec2(ui.available_width(), 26.0);
                if place.0 {
                    // Armed: the button flips to an amber cancel.
                    if ui
                        .add_sized(full, theme::warn_button("CLICK THE MAP\u{2026}  (cancel)"))
                        .on_hover_text("click anywhere on the map to drop your pin \u{00B7} Esc cancels")
                        .clicked()
                    {
                        place.0 = false;
                    }
                } else if ui
                    .add_sized(
                        full,
                        theme::primary_button(if start_pt.0.is_some() {
                            "MOVE POSITION"
                        } else {
                            "PLACE ON MAP"
                        }),
                    )
                    .on_hover_text("then click anywhere on the map to drop your pin")
                    .clicked()
                {
                    place.0 = true;
                }
            });

            ui.add_space(theme::SP_MD);

            // ===== 2 · EXTRACTS =====
            ui.horizontal(|ui| {
                ui.label(theme::section_header("EXTRACTS", rows.len()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            ready && !rows.is_empty(),
                            egui::Button::new(RichText::new("nearest").size(10.0)),
                        )
                        .on_hover_text("route to whichever extract is closest by foot")
                        .clicked()
                    {
                        ui_state.selected = None;
                        route.write(RouteRequest {
                            start: None,
                            dests: rows.iter().map(|r| r.pos).collect(),
                            optimize_order: true, // chain puts the nearest first
                        });
                    }
                });
            });
            if rows.is_empty() {
                ui.label(
                    RichText::new("no extracts found on this map")
                        .size(theme::SIZE_SMALL)
                        .italics()
                        .color(theme::MUTED),
                );
            }
            egui::ScrollArea::vertical()
                .id_salt("nav_extracts")
                .auto_shrink([false, true])
                .max_height(ui.available_height() - 90.0)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 3.0);
                    for r in &rows {
                        let selected = ui_state.selected == Some(r.entity);
                        let border = if selected { theme::ACCENT } else { theme::BORDER };
                        let resp = theme::card(ui, border, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("\u{25CF}").color(r.accent).size(11.0));
                                let name_col = if r.inactive { theme::FAINT } else { theme::BONE };
                                // Right side FIRST (distance + off tag), then the name truncates
                                // into whatever remains — a long name can never overlap the metres.
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new(format!("{:.0} m", r.dist))
                                                .size(theme::SIZE_CAPTION)
                                                .color(theme::FAINT),
                                        );
                                        if r.inactive {
                                            ui.label(
                                                RichText::new("off")
                                                    .size(theme::SIZE_TINY)
                                                    .color(theme::FAINT),
                                            )
                                            .on_hover_text("inactive in the current scene");
                                        }
                                        ui.with_layout(
                                            egui::Layout::left_to_right(egui::Align::Center),
                                            |ui| {
                                                ui.add(
                                                    egui::Label::new(
                                                        RichText::new(&r.name)
                                                            .size(theme::SIZE_LABEL)
                                                            .color(name_col),
                                                    )
                                                    .truncate()
                                                    .selectable(false),
                                                )
                                                .on_hover_text(&r.name);
                                            },
                                        );
                                    },
                                );
                            });
                        });
                        // The whole row is the click target: route from your position to it.
                        let row = resp
                            .response
                            .interact(egui::Sense::click())
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .on_hover_text(if ready {
                                "route here from your position"
                            } else {
                                "routing unavailable (no nav data)"
                            });
                        if row.clicked() && ready {
                            ui_state.selected = Some(r.entity);
                            route.write(RouteRequest {
                                start: None,
                                dests: vec![r.pos],
                                optimize_order: false,
                            });
                        }
                        // Middle-ground affordance: double-click flies the camera to the extract.
                        if row.double_clicked() {
                            cam_cmd.fly_to = Some(r.pos);
                        }
                    }
                });

            // ===== 3 · ROUTE =====
            match &route_result.status {
                RouteStatus::Idle => {}
                RouteStatus::Pending => {
                    ui.add_space(theme::SP_SM);
                    ui.label(
                        RichText::new("computing route\u{2026}")
                            .size(theme::SIZE_SMALL)
                            .color(theme::ACCENT),
                    );
                }
                RouteStatus::Ok => {
                    ui.add_space(theme::SP_SM);
                    theme::card(ui, theme::OK, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("ROUTE  {:.0} m", route_result.dist))
                                    .size(theme::SIZE_BODY)
                                    .strong()
                                    .color(theme::TEXT_BRIGHT),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button(RichText::new("clear").size(10.0)).clicked() {
                                    ui_state.selected = None;
                                    route.write(RouteRequest {
                                        start: None,
                                        dests: Vec::new(),
                                        optimize_order: false,
                                    });
                                }
                            });
                        });
                        ui.label(
                            RichText::new("walkable distance \u{00B7} drawn on the map")
                                .size(theme::SIZE_TINY)
                                .color(theme::MUTED),
                        );
                    });
                }
                RouteStatus::Error(e) => {
                    ui.add_space(theme::SP_SM);
                    ui.label(
                        RichText::new(e.as_str())
                            .size(theme::SIZE_CAPTION)
                            .color(theme::DANGER_TEXT),
                    );
                }
            }
        });

    // Armed-mode banner over the viewport (drawn after the panel so it centers in the free area).
    if place.0 {
        place_banner(ctx, &mut place);
    }
}

/// Gold matching the on-map "you are here" pin gizmo.
const GOLD: Color32 = Color32::from_rgb(255, 209, 51);

/// Raw internal ids ("interchange_secret_extraction  [Secret]") -> display ("Interchange Secret
/// Extraction  [Secret]"): only underscored names are touched — proper names pass through as-is.
fn pretty_name(name: &str) -> String {
    if !name.contains('_') {
        return name.to_string();
    }
    name.split(' ')
        .map(|tok| {
            if !tok.contains('_') {
                return tok.to_string();
            }
            tok.split('_')
                .filter(|w| !w.is_empty())
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Floating "click the map" banner, centered over the 3D viewport while place-mode is armed.
/// Clicking the banner cancels (as does Esc — handled by pick.rs).
fn place_banner(ctx: &egui::Context, place: &mut PlaceMode) {
    let avail = ctx.available_rect();
    let resp = egui::Area::new(egui::Id::new("nav_place_banner"))
        .order(egui::Order::Foreground)
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(avail.center().x, avail.top() + 18.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme::CARD_TRANSLUCENT)
                .stroke(egui::Stroke::new(1.0, GOLD))
                .inner_margin(egui::Margin::symmetric(14, 8))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("CLICK THE MAP TO PLACE YOUR POSITION")
                            .size(theme::SIZE_LABEL)
                            .strong()
                            .color(GOLD),
                    );
                    ui.label(
                        RichText::new("Esc or click here to cancel")
                            .size(theme::SIZE_TINY)
                            .color(theme::MUTED),
                    );
                });
        });
    if resp.response.interact(egui::Sense::click()).clicked() {
        place.0 = false;
    }
}
