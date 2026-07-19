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
//!
//! The START MENU (`menu::menu_ui`) is the ONLY frontend that feeds this worker (the in-raid
//! "MAP PROCESSING" pill was removed). The child-process streaming (stdout tail + `[STAGE i/N]`
//! parsing + reap) lives in `menu::BuildJob` — this module is the queue/worker around it.

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
        // NOTE: the in-raid "MAP PROCESSING" floating panel was removed — builds/sync are driven
        // from the START MENU only. The worker + queue below still exist globally so the menu
        // frontend can enqueue and watch progress; nothing enqueues while a map is open anymore.
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

// (removed) `job_panel` + `JobPanelState`: the in-raid "MAP PROCESSING" floating pill. Builds and
// intel sync are driven from the START MENU only now; the JobWorker above remains the shared queue
// the menu frontend feeds and polls.
