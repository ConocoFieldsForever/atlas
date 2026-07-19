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
    /// `force` = the menu UPDATE path: run `build_map.py --force` so extraction/lights/SH/nav
    /// re-run against the current game files instead of reusing stale extracted data (release
    /// blocker: a plain build skips extraction when the old scene.json still exists). Plain BUILD
    /// passes `force = false` and stays incremental.
    /// `background` = the "Process in background" toggle (default ON): the child is spawned DETACHED
    /// with its stdout/stderr redirected to a log FILE (not an inherited pipe), so it SURVIVES the
    /// app being closed mid-build. A later launch reattaches to it via the `build_<map>.running.json`
    /// sidecar. `false` keeps the legacy inherited-pipe behavior (dies when Atlas closes).
    BuildMap { map: String, game_dir: String, force: bool, background: bool },
    SyncIntel,
    /// Install the Python build dependencies (venv + UnityPy/numpy/Pillow) from the menu.
    InstallDeps,
}

impl Job {
    pub fn label(&self) -> String {
        match self {
            // UPDATE and BUILD share a label so the worker won't run both for one map at once.
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
            Job::BuildMap { map, game_dir, force, background } => {
                BuildJob::spawn(map, game_dir, *force, *background)
            }
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
    /// one is already running or pending). FORCE-AWARE for map builds: an UPDATE (`force=true`,
    /// re-extract against patched game files) must NEVER be swallowed by a queued/running plain BUILD
    /// (`force=false`) for the same map — that silently skips the user's requested re-extraction and
    /// can flip the row to READY over stale geometry. So a forced build UPGRADES a queued non-force
    /// build in place, and appends after a running non-force build rather than being dropped.
    pub fn enqueue(&mut self, job: Job) {
        let l = job.label();
        let incoming_force = matches!(&job, Job::BuildMap { force: true, .. });
        // Same-label job already QUEUED?
        if let Some(pos) = self.queue.iter().position(|j| j.label() == l) {
            let queued_force = matches!(&self.queue[pos], Job::BuildMap { force: true, .. });
            if incoming_force && !queued_force {
                self.queue[pos] = job; // upgrade the pending plain build to a forced re-extract
            }
            // else: an equal-or-stronger job is already queued → dedup (drop the incoming one).
            return;
        }
        // Same-label job currently RUNNING?
        if let Some((cur, _)) = self.current.as_ref().filter(|(j, _)| j.label() == l) {
            let running_force = matches!(cur, Job::BuildMap { force: true, .. });
            if incoming_force && !running_force {
                // A plain build is in flight but the user asked for a forced re-extract: queue it so
                // it runs AFTER (don't drop it; don't cancel the in-flight extraction mid-way).
                self.queue.push_back(job);
            }
            // else: an equal-or-stronger job is already running → dedup.
            return;
        }
        // No conflict — queue it.
        self.queue.push_back(job);
    }
    pub fn busy(&self) -> bool {
        self.current.is_some()
    }

    /// Adopt a build started by a PREVIOUS Atlas process (survived an app-close because it was
    /// spawned detached with file output). The menu's startup reattach scan builds a tail-only
    /// `BuildJob` around the still-alive child (its PID + log file) and hands it here so the live
    /// panel + BUILDING row light up again. No-op if a job is already in flight (only one at a time).
    pub fn reattach(&mut self, job: Job, child: BuildJob) {
        if self.current.is_none() {
            self.current = Some((job, child));
        }
    }

    /// Inject an ALREADY-FINISHED build as the lingering "last" outcome — used by the startup scan
    /// when a detached build died while Atlas was closed WITHOUT completing (its pack never appeared):
    /// surfaces the interrupted result + log so the user can resume/rebuild. Bumps `completed` so the
    /// frontend re-scans exactly as it would for a job that finished in-process.
    pub fn set_finished(&mut self, job: Job, child: BuildJob) {
        self.last = Some((job, child));
        self.last_dismissed = false;
        self.completed = self.completed.wrapping_add(1);
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
