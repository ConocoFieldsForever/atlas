//! navigate_panel.rs — the NAVIGATION tab (right-panel router slot, 4th rail icon).
//!
//! The raid-planning flow in three steps, top to bottom:
//!   1. YOUR POSITION — one primary button arms a click-to-place mode; the next click on the map
//!      drops the gold "you are here" pin (pick.rs does the raycast; Esc or the banner's cancel
//!      button aborts). No hotkeys to remember — the button IS the affordance. Moving/removing the
//!      pin auto-clears any drawn route (it started from the old spot; pathfind.rs).
//!   2. EXTRACTS — a table of every extract (faction-coloured painter dot — no font glyphs — plus a
//!      separated faction tag and a `~straight-line` distance). Clicking a row computes the walkable
//!      route to it; ROUTE NEAREST EXTRACT solves one A* per ACTIVE extract and keeps the shortest
//!      (true nearest-by-foot, not a tour). Rows work even while the Extracts overlay is hidden.
//!   3. ROUTE — a labelled result card: WHERE the route goes + walkable metres; the matching row is
//!      highlighted from `RouteResult::dest_label` (so "nearest" highlights its winner too).
//!
//! All colors/typography come from ui_theme (single source of truth). Routing itself is the
//! in-process CPU A* (nav.rs); this panel only writes `RouteRequest`s.

#![cfg(feature = "egui")]

use bevy::prelude::*;
use bevy_egui::egui::{self, Color32, RichText};

use crate::pathfind::{
    PlaceMode, RouteOpts, RouteRequest, RouteResult, RouteStatus, ServerStatus, StartPoint,
};
use crate::poi::{PoiLayer, SceneInactive, ZoneWall};
use crate::render::CullCamera;
use crate::ui::RightPanelTab;
use crate::ui_theme as theme;

/// Panel-local state: the row whose route is being COMPUTED right now (immediate feedback while
/// `RouteStatus::Pending`; once Ok the highlight is driven by `RouteResult::dest_label` instead)
/// + the loot-plan knobs.
pub struct NavUiState {
    pending: Option<Entity>,
    plan_min_value: i64,
    plan_stops: usize,
    plan_budget: f32,
    /// Last MapEpoch we reacted to — on a swap, `pending` (an extract Entity from the OLD map) is
    /// cleared so it can't highlight a wrong row / recycled id on the new map.
    last_epoch: u64,
}
impl Default for NavUiState {
    fn default() -> Self {
        Self {
            pending: None,
            plan_min_value: 100_000,
            plan_stops: 10,
            plan_budget: 1500.0,
            last_epoch: 0,
        }
    }
}

/// One extract row, resolved from the marker entities each frame (cheap: a handful of extracts).
struct Row {
    entity: Entity,
    /// Prettified display name, faction tag stripped ("NW Exfil").
    name: String,
    /// Faction tag without brackets ("PMC" / "Scav" / "All" / ""), shown separated + dim.
    tag: String,
    /// The label sent with route requests and echoed back in `RouteResult::dest_label`.
    label: String,
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
    mut route_result: ResMut<RouteResult>,
    mut route_opts: ResMut<RouteOpts>,
    plan: Res<crate::planner::PlanResult>,
    mut plan_req: MessageWriter<crate::planner::PlanRequest>,
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
    epoch: Res<crate::render::MapEpoch>,
    mut ui_state: Local<NavUiState>,
) {
    if menu.is_some() {
        return; // start-menu mode owns the screen
    }
    // In-place map swap: forget the pending extract row (its Entity is from the OLD map).
    if epoch.0 != ui_state.last_epoch {
        ui_state.pending = None;
        ui_state.last_epoch = epoch.0;
    }
    // Leaving the tab keeps an armed place-mode live on purpose: you arm it, swing the camera,
    // click. The banner (with its cancel button) stays visible either way.
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
            let (raw_name, tag) = split_tag(&info.title);
            let name = pretty_name(&raw_name);
            let label = if tag.is_empty() { name.clone() } else { format!("{name} [{tag}]") };
            Row {
                entity: e,
                name,
                tag,
                label,
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
        rows.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.tag.cmp(&b.tag)));
    }
    let active_n = rows.iter().filter(|r| !r.inactive).count();

    egui::SidePanel::right("map_layers")
        .resizable(false)
        .frame(theme::panel_frame())
        .default_width(300.0)
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
                    // Honest state (finding 2): routing needs a baked nav grid this pack doesn't
                    // carry, and the ordinary menu build can't produce one — so don't promise a
                    // rebuild enables it. When a pack DOES ship nav files the `ready` path above
                    // takes over and this card never shows.
                    ui.label(
                        RichText::new("Routing has not been built for this map yet \u{2014} extract, POI browsing and camera flight still work.")
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
                            dot(ui, GOLD, 4.5);
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

            // ---- avoid options: soft-avoid danger zones; when any is on, every route computes
            // Direct / Cautious / Wide-berth variants (listed with distances under ROUTE). ----
            ui.horizontal(|ui| {
                ui.label(RichText::new("avoid").size(theme::SIZE_SMALL).color(theme::MUTED));
                ui.checkbox(&mut route_opts.avoid_boss, RichText::new("bosses").size(theme::SIZE_SMALL))
                    .on_hover_text("detour around boss spawn areas when a reasonable path exists");
                ui.checkbox(&mut route_opts.avoid_pmc, RichText::new("PMCs").size(theme::SIZE_SMALL))
                    .on_hover_text("detour around PMC spawn areas");
                ui.checkbox(&mut route_opts.avoid_scav, RichText::new("scavs").size(theme::SIZE_SMALL))
                    .on_hover_text("detour around scav spawn areas");
            });

            // ---- live search visualization: replay the A* flood as it converges on a single
            // destination (cosmetic; only affects single-extract routes, not tours/loot plans). ----
            ui.checkbox(
                &mut route_opts.visualize,
                RichText::new("visualize search").size(theme::SIZE_SMALL).color(theme::MUTED),
            )
            .on_hover_text("watch the pathfinder's wavefront expand and converge on the next single-destination route");

            // ---- flagship action: true nearest-by-foot (one A* per ACTIVE extract, keep the
            // shortest — NOT a tour through all of them). ----
            let full = egui::vec2(ui.available_width(), 26.0);
            if ui
                .add_enabled(
                    ready && active_n > 0,
                    egui::Button::new(
                        RichText::new("ROUTE NEAREST EXTRACT")
                            .size(theme::SIZE_LABEL)
                            .strong()
                            .color(if ready && active_n > 0 { theme::ACCENT } else { theme::FAINT }),
                    )
                    .min_size(full)
                    .corner_radius(0.0),
                )
                .on_hover_text("compares the walkable route to every active extract and takes the shortest")
                .clicked()
            {
                ui_state.pending = None;
                let act: Vec<&Row> = rows.iter().filter(|r| !r.inactive).collect();
                route.write(RouteRequest {
                    start: None,
                    dests: act.iter().map(|r| r.pos).collect(),
                    labels: act.iter().map(|r| r.label.clone()).collect(),
                    nearest_of: true,
                    ..Default::default()
                });
            }

            // ===== LOOT PLAN (orienteering: max value under a walking budget, ends at an extract) =====
            ui.add_space(theme::SP_SM);
            egui::CollapsingHeader::new(theme::section_header("LOOT PLAN", plan.stops.len()))
                .id_salt("nav_lootplan")
                .default_open(false)
                // Auto-open while a plan is computing / live (collapses again on clear).
                .open((!matches!(plan.status, crate::planner::PlanStatus::Idle)).then_some(true))
                .show(ui, |ui| {
                    use crate::planner::{PlanRequest, PlanStatus};
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                    // knobs
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("min value").size(theme::SIZE_SMALL).color(theme::MUTED));
                        egui::ComboBox::from_id_salt("plan_minv")
                            .selected_text(format!("{}k \u{20BD}", ui_state.plan_min_value / 1000))
                            .show_ui(ui, |ui| {
                                for v in [50_000i64, 100_000, 150_000, 200_000, 300_000] {
                                    ui.selectable_value(&mut ui_state.plan_min_value, v, format!("{}k \u{20BD}", v / 1000));
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("stops").size(theme::SIZE_SMALL).color(theme::MUTED));
                        ui.add(egui::Slider::new(&mut ui_state.plan_stops, 4..=18));
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("budget").size(theme::SIZE_SMALL).color(theme::MUTED));
                        ui.add(egui::Slider::new(&mut ui_state.plan_budget, 500.0..=3000.0).suffix(" m").step_by(50.0));
                    });
                    let full = egui::vec2(ui.available_width(), 26.0);
                    if ui
                        .add_enabled(ready, egui::Button::new(
                            RichText::new("PLAN LOOT RUN").size(theme::SIZE_LABEL).strong()
                                .color(if ready { theme::ACCENT } else { theme::FAINT }))
                            .min_size(full).corner_radius(0.0))
                        .on_hover_text("pick the highest-value loot tour that fits the budget, ending at an extract \u{00B7} honors the avoid options")
                        .clicked()
                    {
                        plan_req.write(PlanRequest {
                            min_value: ui_state.plan_min_value,
                            max_stops: ui_state.plan_stops,
                            budget_m: ui_state.plan_budget,
                        });
                    }
                    match &plan.status {
                        PlanStatus::Idle => {}
                        PlanStatus::Pending => {
                            ui.label(RichText::new("optimizing\u{2026}").size(theme::SIZE_SMALL).color(theme::ACCENT));
                        }
                        PlanStatus::Error(e) => {
                            ui.label(RichText::new(e.as_str()).size(theme::SIZE_CAPTION).color(theme::DANGER_TEXT));
                        }
                        PlanStatus::Ok => {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!(
                                        "\u{2248}{}k \u{20BD}  \u{00B7}  {:.0} m  \u{00B7}  exits {}",
                                        plan.total_value / 1000,
                                        plan.total_dist,
                                        plan.extract
                                    ))
                                    .size(theme::SIZE_SMALL)
                                    .color(theme::OK),
                                );
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button(RichText::new("clear").size(10.0)).clicked() {
                                        plan_req.write(PlanRequest { min_value: 0, max_stops: 0, budget_m: 0.0 });
                                        route.write(RouteRequest::default());
                                    }
                                });
                            });
                            for (i, st) in plan.stops.iter().enumerate() {
                                let row = ui
                                    .horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!("{:>2}.", i + 1))
                                                .size(theme::SIZE_CAPTION)
                                                .color(theme::FAINT),
                                        );
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            ui.label(
                                                RichText::new(format!("+{:.0} m", st.leg))
                                                    .size(theme::SIZE_TINY)
                                                    .color(theme::FAINT),
                                            );
                                            ui.label(
                                                RichText::new(format!("{}k", st.value / 1000))
                                                    .size(theme::SIZE_CAPTION)
                                                    .color(theme::BEIGE),
                                            );
                                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                ui.add(
                                                    egui::Label::new(
                                                        RichText::new(&st.name)
                                                            .size(theme::SIZE_CAPTION)
                                                            .color(theme::BONE),
                                                    )
                                                    .truncate()
                                                    .selectable(false),
                                                );
                                            });
                                        });
                                    })
                                    .response
                                    .interact(egui::Sense::click())
                                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                                    .on_hover_text("fly to this stop");
                                if row.clicked() {
                                    cam_cmd.fly_to = Some(st.pos);
                                }
                            }
                        }
                    }
                });

            ui.add_space(theme::SP_MD);

            // ===== 2 · EXTRACTS =====
            ui.horizontal(|ui| {
                ui.label(theme::section_header("EXTRACTS", rows.len()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new("click a row to route there")
                            .size(theme::SIZE_TINY)
                            .color(theme::FAINT),
                    );
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
            let routed_label = (route_result.status == RouteStatus::Ok)
                .then(|| route_result.dest_label.clone())
                .flatten();
            let list_h = (ui.available_height() - 96.0).max(60.0);
            egui::ScrollArea::vertical()
                .id_salt("nav_extracts")
                .auto_shrink([false, true])
                .max_height(list_h)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 3.0);
                    // Organized by WHO can use the extract: your faction's first, then Scav,
                    // then shared/special — each group under a small dim header.
                    let groups: [(&str, Vec<&Row>); 3] = [
                        ("PMC", rows.iter().filter(|r| r.tag == "PMC").collect()),
                        ("SCAV", rows.iter().filter(|r| r.tag == "Scav").collect()),
                        (
                            "SHARED & SPECIAL",
                            rows.iter().filter(|r| r.tag != "PMC" && r.tag != "Scav").collect(),
                        ),
                    ];
                    for (gname, grows) in &groups {
                        if grows.is_empty() {
                            continue;
                        }
                        ui.add_space(2.0);
                        ui.label(
                            RichText::new(*gname)
                                .size(theme::SIZE_TINY)
                                .strong()
                                .color(theme::FAINT),
                        );
                        for r in grows {
                            let r: &Row = r;
                        // Highlight: the destination of the CURRENT route (label match — also
                        // covers "nearest" picking its winner), or the row being computed.
                        let is_routed = routed_label.as_deref() == Some(r.label.as_str());
                        let is_pending = route_result.status == RouteStatus::Pending
                            && ui_state.pending == Some(r.entity);
                        let border = if is_routed {
                            theme::OK
                        } else if is_pending {
                            theme::ACCENT
                        } else {
                            theme::BORDER
                        };
                        let resp = theme::card(ui, border, |ui| {
                            ui.horizontal(|ui| {
                                dot(ui, r.accent, 4.0);
                                let name_col = if r.inactive { theme::FAINT } else { theme::BONE };
                                // Right side FIRST (distance + tags), then the name truncates into
                                // what remains — a long name can never overlap the metres.
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new(format!("~{:.0} m", r.dist))
                                                .size(theme::SIZE_CAPTION)
                                                .color(theme::MUTED),
                                        )
                                        .on_hover_text(
                                            "straight-line distance \u{2014} click the row for the walkable route",
                                        );
                                        if r.inactive {
                                            ui.label(
                                                RichText::new("off")
                                                    .size(theme::SIZE_TINY)
                                                    .color(theme::FAINT),
                                            )
                                            .on_hover_text("inactive in the current scene");
                                        }
                                        // Faction tag only in the mixed group — inside the pure
                                        // PMC/SCAV groups the header already says it.
                                        if !r.tag.is_empty() && r.tag != "PMC" && r.tag != "Scav" {
                                            ui.label(
                                                RichText::new(&r.tag)
                                                    .size(theme::SIZE_TINY)
                                                    .color(theme::FAINT),
                                            )
                                            .on_hover_text("extract faction");
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
                                                .on_hover_text(&r.label);
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
                                "route here from your position \u{00B7} double-click to fly the camera"
                            } else {
                                "routing unavailable (no nav data)"
                            });
                        // Hover feedback: a faint wash over the row (the card was already painted,
                        // so overlay it — cheap and obvious).
                        if row.hovered() {
                            ui.painter().rect_filled(
                                resp.response.rect,
                                0.0,
                                Color32::from_rgba_premultiplied(255, 255, 255, 5),
                            );
                        }
                        if row.double_clicked() {
                            // Fly the camera to the extract (kept OFF the single-click so a route
                            // click never yanks the camera).
                            cam_cmd.fly_to = Some(r.pos);
                        } else if row.clicked() && ready {
                            ui_state.pending = Some(r.entity);
                            route.write(RouteRequest {
                                start: None,
                                dests: vec![r.pos],
                                labels: vec![r.label.clone()],
                                ..Default::default()
                            });
                        }
                        }
                    }
                });

            // ===== 3 · ROUTE =====
            let status = route_result.status.clone();
            match &status {
                RouteStatus::Idle => {}
                RouteStatus::Pending => {
                    ui.add_space(theme::SP_SM);
                    ui.label(theme::section_header("ROUTE", 0));
                    ui.label(
                        RichText::new("computing\u{2026}")
                            .size(theme::SIZE_SMALL)
                            .color(theme::ACCENT),
                    );
                }
                RouteStatus::Ok => {
                    ui.add_space(theme::SP_SM);
                    ui.label(theme::section_header("ROUTE", 0));
                    let dest = route_result.dest_label.clone();
                    let opts_list: Vec<(&'static str, f32)> =
                        route_result.options.iter().map(|o| (o.name, o.dist)).collect();
                    let selected = route_result.selected;
                    theme::card(ui, theme::OK, |ui| {
                        ui.horizontal(|ui| {
                            // WHERE the route goes (the whole point of the card), then the metres.
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button(RichText::new("clear").size(10.0)).clicked() {
                                    ui_state.pending = None;
                                    route.write(RouteRequest::default()); // empty dests = clear
                                }
                                ui.with_layout(
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(dest.as_deref().unwrap_or("Route"))
                                                    .size(theme::SIZE_BODY)
                                                    .strong()
                                                    .color(theme::TEXT_BRIGHT),
                                            )
                                            .truncate(),
                                        );
                                    },
                                );
                            });
                        });
                        if opts_list.len() <= 1 {
                            ui.label(
                                RichText::new(format!(
                                    "{:.0} m walkable \u{00B7} drawn on the map",
                                    route_result.dist
                                ))
                                .size(theme::SIZE_CAPTION)
                                .color(theme::OK),
                            );
                        } else {
                            // Variant list: click one to draw it (the others stay as dim
                            // alternates on the map). Longer = safer.
                            for (i, (name, dist)) in opts_list.iter().enumerate() {
                                let sel = i == selected;
                                let resp = ui
                                    .horizontal(|ui| {
                                        dot(
                                            ui,
                                            if sel { theme::color32(bevy::prelude::Color::srgb(0.25, 1.0, 0.45)) } else { theme::FAINT },
                                            3.2,
                                        );
                                        ui.label(
                                            RichText::new(*name)
                                                .size(theme::SIZE_SMALL)
                                                .strong()
                                                .color(if sel { theme::TEXT_BRIGHT } else { theme::MUTED }),
                                        );
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    RichText::new(format!("{dist:.0} m"))
                                                        .size(theme::SIZE_SMALL)
                                                        .color(if sel { theme::OK } else { theme::MUTED }),
                                                );
                                            },
                                        );
                                    })
                                    .response
                                    .interact(egui::Sense::click())
                                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                                    .on_hover_text("draw this variant");
                                if resp.clicked() {
                                    route_result.select(i);
                                }
                            }
                            ui.label(
                                RichText::new("variants differ by how hard they avoid danger zones")
                                    .size(theme::SIZE_TINY)
                                    .color(theme::MUTED),
                            );
                        }
                    });
                }
                RouteStatus::Error(e) => {
                    ui.add_space(theme::SP_SM);
                    ui.label(theme::section_header("ROUTE", 0));
                    theme::card(ui, theme::DANGER, |ui| {
                        ui.label(
                            RichText::new("NO ROUTE")
                                .size(theme::SIZE_LABEL)
                                .strong()
                                .color(theme::DANGER_TEXT),
                        );
                        ui.label(
                            RichText::new(e.as_str())
                                .size(theme::SIZE_CAPTION)
                                .color(theme::MUTED),
                        );
                    });
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

/// A filled circle drawn with the painter (NOT a font glyph — the \u{25CF} bullet renders as a
/// hollow box in this font at small sizes, which read like leftover checkboxes).
fn dot(ui: &mut egui::Ui, color: Color32, radius: f32) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(radius * 2.0 + 2.0, radius * 2.0 + 2.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), radius, color);
}

/// "NW Exfil  [PMC]" -> ("NW Exfil", "PMC"); titles without a trailing [tag] pass through whole.
fn split_tag(title: &str) -> (String, String) {
    let t = title.trim_end();
    if t.ends_with(']') {
        if let Some(i) = t.rfind('[') {
            let name = t[..i].trim_end().to_string();
            let tag = t[i + 1..t.len() - 1].trim().to_string();
            if !name.is_empty() {
                return (name, tag);
            }
        }
    }
    (t.to_string(), String::new())
}

/// Raw internal ids ("interchange_secret_extraction") -> display ("Interchange Secret Extraction");
/// only underscored names are touched — proper names pass through as-is.
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
/// Carries a REAL cancel button (a text-label click zone was dead over the labels); Esc works too
/// (handled by pick.rs, respecting text-field focus).
fn place_banner(ctx: &egui::Context, place: &mut PlaceMode) {
    let avail = ctx.available_rect();
    egui::Area::new(egui::Id::new("nav_place_banner"))
        .order(egui::Order::Foreground)
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(avail.center().x, avail.top() + 18.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme::CARD_TRANSLUCENT)
                .stroke(egui::Stroke::new(1.0, GOLD))
                .inner_margin(egui::Margin::symmetric(14, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("CLICK THE MAP TO PLACE YOUR POSITION")
                                .size(theme::SIZE_LABEL)
                                .strong()
                                .color(GOLD),
                        );
                        if ui
                            .button(RichText::new("cancel").size(theme::SIZE_CAPTION))
                            .on_hover_text("or press Esc")
                            .clicked()
                        {
                            place.0 = false;
                        }
                    });
                });
        });
}
