//! jobs.rs — the background JOB WORKER: run the map-build / tarkov.dev-sync pipelines from ANYWHERE
//! (the start menu OR while a map is open in-raid) through ONE queue + one worker.
//!
//! Clean-worker architecture (the maintainable core):
//!   * `Job`        — what to run (build a map / sync intel). Snapshots the paths it needs at
//!                    enqueue time (a later game-dir edit can't mutate queued work) and knows how
//!                    to spawn its child process.
//!   * `JobWorker`  — the resource: a FIFO queue + the single in-flight job + the last FINISHED job
//!                    (kept with its full log so a frontend can show the lingering outcome panel).
//!                    Runs at most one child at a time (these are heavy GPU/Unity pipelines).
//!   * `pump_jobs`  — the one system: reap a finished child, then start the next queued one.
//!   * `job_panel`  — a compact floating egui window (in-raid only) to enqueue builds / sync and
//!                    watch progress, so maps process in the background WITHOUT leaving the map.
//!
//! BOTH frontends feed this one worker: the start menu (`menu::menu_ui`) and the in-raid panel
//! below. The child-process streaming (stdout tail + `[STAGE i/N]` parsing + reap) lives in
//! `menu::BuildJob` — this module is the queue/worker around it, not a second copy.

use crate::menu::BuildJob;
use bevy::prelude::*;
use std::collections::VecDeque;

/// A unit of background work. `BuildMap` runs `tools/build_map.py <map>` against the game install
/// captured when it was enqueued; `SyncIntel` runs `tools/sync_intel.py` (no map/game args).
#[derive(Clone)]
pub enum Job {
    BuildMap { map: String, game_dir: String },
    SyncIntel,
    /// Install the Python build dependencies (venv + UnityPy/numpy/Pillow) from the menu.
    InstallDeps,
}

impl Job {
    pub fn label(&self) -> String {
        match self {
            Job::BuildMap { map, .. } => format!("build {map}"),
            Job::SyncIntel => "sync intel".to_string(),
            Job::InstallDeps => "install deps".to_string(),
        }
    }
    /// The map key when this is a build (drives the menu's per-row BUILDING state), else None.
    pub fn build_key(&self) -> Option<&str> {
        match self {
            Job::BuildMap { map, .. } => Some(map.as_str()),
            Job::SyncIntel | Job::InstallDeps => None,
        }
    }
    fn spawn(&self) -> std::io::Result<BuildJob> {
        match self {
            Job::BuildMap { map, game_dir } => BuildJob::spawn(map, game_dir),
            Job::SyncIntel => BuildJob::spawn_intel(),
            Job::InstallDeps => BuildJob::spawn_setup(),
        }
    }
}

/// The single background worker: one running child at a time + a queue behind it. Exists in BOTH
/// menu and raid mode (a global resource), so either frontend can feed it.
#[derive(Resource, Default)]
pub struct JobWorker {
    queue: VecDeque<Job>,
    /// The in-flight job (its kind + running child).
    current: Option<(Job, BuildJob)>,
    /// The most recently FINISHED job, kept with its full log for the lingering outcome panel
    /// until a frontend dismisses it (menu CLOSE). A new completion clears the dismissed flag.
    last: Option<(Job, BuildJob)>,
    last_dismissed: bool,
    /// Monotonic count of finished jobs — a frontend detects "a job just completed" (→ rescan)
    /// by watching this change, without consuming a shared event.
    pub completed: u64,
    /// Transient "failed to start" message (rare: python kit missing / spawn error).
    pub spawn_error: Option<String>,
}

impl JobWorker {
    /// Queue a job (deduped by label: won't double-queue the same map build / a second sync while
    /// one is already running or pending).
    pub fn enqueue(&mut self, job: Job) {
        let l = job.label();
        let running = self.current.as_ref().map(|(j, _)| j.label() == l).unwrap_or(false);
        if !running && !self.queue.iter().any(|j| j.label() == l) {
            self.queue.push_back(job);
        }
    }
    pub fn busy(&self) -> bool {
        self.current.is_some()
    }
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    // ---- in-flight job ----
    pub fn current_job(&self) -> Option<&BuildJob> {
        self.current.as_ref().map(|(_, b)| b)
    }
    /// The map key if a BuildMap is in flight (else None — e.g. a sync is running).
    pub fn current_build_key(&self) -> Option<&str> {
        self.current.as_ref().and_then(|(j, _)| j.build_key())
    }
    pub fn current_is_sync(&self) -> bool {
        matches!(self.current.as_ref().map(|(j, _)| j), Some(Job::SyncIntel))
    }
    pub fn current_is_install(&self) -> bool {
        matches!(self.current.as_ref().map(|(j, _)| j), Some(Job::InstallDeps))
    }
    /// (label, latest `[STAGE …]` marker) of the running job, for a compact readout.
    pub fn status(&self) -> Option<(String, String)> {
        self.current.as_ref().map(|(j, b)| (j.label(), b.snapshot(1).0))
    }
    pub fn cancel_current(&mut self) {
        if let Some((_, j)) = &self.current {
            j.cancel();
        }
    }

    // ---- last finished job (lingers until dismissed) ----
    pub fn last_job(&self) -> Option<&BuildJob> {
        if self.last_dismissed {
            return None;
        }
        self.last.as_ref().map(|(_, b)| b)
    }
    pub fn last_is_build(&self) -> bool {
        !self.last_dismissed && matches!(self.last.as_ref().map(|(j, _)| j), Some(Job::BuildMap { .. }))
    }
    pub fn last_is_sync(&self) -> bool {
        !self.last_dismissed && matches!(self.last.as_ref().map(|(j, _)| j), Some(Job::SyncIntel))
    }
    pub fn last_is_install(&self) -> bool {
        !self.last_dismissed && matches!(self.last.as_ref().map(|(j, _)| j), Some(Job::InstallDeps))
    }
    /// (label, ok) of the most recent finished (undismissed) job.
    pub fn last_outcome(&self) -> Option<(String, bool)> {
        if self.last_dismissed {
            return None;
        }
        self.last.as_ref().map(|(j, b)| (j.label(), b.snapshot(0).3))
    }
    pub fn dismiss_last(&mut self) {
        self.last_dismissed = true;
    }
}

pub struct JobsPlugin;
impl Plugin for JobsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<JobWorker>().add_systems(Update, pump_jobs);
        #[cfg(feature = "egui")]
        app.add_systems(bevy_egui::EguiPrimaryContextPass, job_panel);
    }
}

/// The one worker tick: reap a finished child, then start the next queued job.
fn pump_jobs(mut w: ResMut<JobWorker>) {
    // Reap the running job if its child has exited.
    let finished = w.current.as_ref().map(|(_, j)| j.snapshot(1).2).unwrap_or(false);
    if finished {
        if let Some(done) = w.current.take() {
            w.last = Some(done);
            w.last_dismissed = false;
            w.completed = w.completed.wrapping_add(1);
        }
    }
    // Start the next queued job when idle.
    if w.current.is_none() {
        if let Some(job) = w.queue.pop_front() {
            match job.spawn() {
                Ok(child) => {
                    w.spawn_error = None;
                    w.current = Some((job, child));
                }
                Err(e) => w.spawn_error = Some(format!("{}: {e}", job.label())),
            }
        }
    }
}

/// Compact floating window (in-raid only) to queue background builds / sync + watch progress.
#[cfg(feature = "egui")]
fn job_panel(
    mut contexts: bevy_egui::EguiContexts,
    menu: Option<Res<crate::menu::MenuState>>,
    mut worker: ResMut<JobWorker>,
    mut ui_state: bevy::ecs::system::Local<JobPanelState>,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};
    if menu.is_some() {
        return; // the start menu owns build/sync there; this is the in-raid frontend
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let busy = worker.busy();
    // Collapsed: a small unobtrusive pill bottom-left; expands to the full controls on click.
    egui::Window::new(RichText::new("MAP PROCESSING").size(theme::SIZE_CAPTION).color(theme::BEIGE))
        .id(egui::Id::new("jobs_panel"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(12.0, -12.0))
        .resizable(false)
        .collapsible(true)
        .default_open(false)
        .frame(
            egui::Frame::new()
                .fill(theme::CARD_TRANSLUCENT)
                .stroke(egui::Stroke::new(1.0, if busy { theme::ACCENT } else { theme::BORDER }))
                .inner_margin(egui::Margin::same(8)),
        )
        .show(ctx, |ui| {
            ui.set_max_width(260.0);
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);

            // ---- running job status ----
            if let Some((label, stage)) = worker.status() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("\u{25CF}").color(theme::ACCENT).size(11.0));
                    ui.label(RichText::new(label).size(theme::SIZE_SMALL).strong().color(theme::TEXT_BRIGHT));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(RichText::new("cancel").size(10.0)).clicked() {
                            worker.cancel_current();
                        }
                    });
                });
                ui.label(RichText::new(stage).size(theme::SIZE_TINY).color(theme::MUTED));
                let q = worker.queued();
                if q > 0 {
                    ui.label(RichText::new(format!("{q} queued")).size(theme::SIZE_TINY).color(theme::FAINT));
                }
            } else if let Some((label, ok)) = worker.last_outcome() {
                ui.label(
                    RichText::new(format!("{} {}", if ok { "\u{2713}" } else { "\u{00D7}" }, label))
                        .size(theme::SIZE_SMALL)
                        .color(if ok { theme::OK } else { theme::DANGER_TEXT }),
                );
            } else if let Some(err) = &worker.spawn_error {
                ui.label(RichText::new(err.as_str()).size(theme::SIZE_SMALL).color(theme::DANGER_TEXT));
            } else {
                ui.label(RichText::new("idle").size(theme::SIZE_SMALL).color(theme::MUTED));
            }

            ui.separator();

            // ---- enqueue a build for any known map ----
            ui.horizontal(|ui| {
                let cur = crate::menu::KNOWN_MAPS
                    .iter()
                    .find(|(k, _)| *k == ui_state.pick)
                    .map(|(_, n)| *n)
                    .unwrap_or("Interchange");
                egui::ComboBox::from_id_salt("jobs_map")
                    .selected_text(RichText::new(cur).size(theme::SIZE_SMALL))
                    .show_ui(ui, |ui| {
                        for (k, name) in crate::menu::KNOWN_MAPS {
                            ui.selectable_value(&mut ui_state.pick, k.to_string(), *name);
                        }
                    });
                if ui.small_button(RichText::new("build").size(10.0))
                    .on_hover_text("run the full build pipeline for this map in the background")
                    .clicked()
                {
                    let m = if ui_state.pick.is_empty() { "interchange".to_string() } else { ui_state.pick.clone() };
                    worker.enqueue(Job::BuildMap { map: m, game_dir: crate::menu::detect_game_dir() });
                }
            });
            if ui.small_button(RichText::new("sync tarkov.dev intel").size(10.0))
                .on_hover_text("re-pull loot values, tasks and icons")
                .clicked()
            {
                worker.enqueue(Job::SyncIntel);
            }
            ui.label(
                RichText::new("builds run in the background \u{2014} switch maps to play a finished one")
                    .size(theme::SIZE_TINY)
                    .italics()
                    .color(theme::MUTED),
            );
        });
}

/// Panel-local combo selection.
#[cfg(feature = "egui")]
#[derive(Default)]
pub struct JobPanelState {
    pick: String,
}
