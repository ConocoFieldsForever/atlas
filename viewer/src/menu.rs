//! eft::menu — Tarkov-style start menu / map manager ("stash screen").
//!
//! Shown when the viewer is launched with NO pack (bare `atlas`). Scans `packs/` and
//! presents one card per known map: install state, pack size on disk, built age, tarkov.dev
//! intel sync age, data-completeness ticks, and a game-update check — each pack's manifest
//! carries the `sourceFingerprint` of the game install it was extracted from (stamped by
//! `assemble_bevy`/`tools/stamp_fingerprint.py`); the menu recomputes the fingerprint from the
//! live install (stat-only, milliseconds) and flags mismatches as "GAME UPDATED - REBUILD".
//!
//! PLAY reuses the map-switch mechanism (spawn self with the pack argv + exit) so the render
//! path keeps its build-once buffers. DELETE removes the pack dir after an explicit confirm.

use bevy::prelude::*;

/// One known/installed map row.
pub struct MapEntry {
    /// Display title ("Streets of Tarkov").
    pub title: String,
    /// Dataset/dir stem ("streets").
    pub key: String,
    /// packs/<key>.eftpack when present on disk.
    pub pack_dir: Option<String>,
    pub size_bytes: u64,
    /// Age in days of manifest.json (pack build) and loot.json (tarkov.dev sync).
    pub built_days: Option<f64>,
    pub intel_days: Option<f64>,
    pub has_volume: bool,
    pub has_grass: bool,
    pub has_gamedata: bool,
    pub has_icons: bool,
    /// None = pack unstamped (unknown vintage); Some(true) = matches the live install.
    pub fp_match: Option<bool>,
    /// A pack dir exists but is STRUCTURALLY complete/loadable (finding 4): manifest.json parses AND
    /// meshes.bin + instances.bin are present. A false here means PLAY would open a blank window, so
    /// the menu shows a DAMAGED badge + offers rebuild/delete instead of PLAY. Always true for a
    /// not-installed row (no pack dir to be broken).
    pub valid: bool,
}

/// How the build panel reaches the running child. `Owned` = we hold the `Child` this process
/// spawned (pipe OR detached-file mode while Atlas stays open) — reap via `try_wait`, cancel via
/// PID + a `.kill()` fallback. `Detached` = a build started by a PREVIOUS Atlas process that we
/// REATTACHED to on startup: we only have its PID (its `Child` died with the old process), so we
/// poll liveness and cancel via `taskkill /T /PID`.
enum ProcHandle {
    Owned(std::sync::Arc<std::sync::Mutex<std::process::Child>>),
    Detached(u32),
}

/// A running `tools/build_map.py <map>` pipeline: stdout+stderr stream into `log` from a
/// reader/tailer thread; the UI shows the tail + the latest `[STAGE i/N]` marker. One at a time.
pub struct BuildJob {
    pub key: String,
    proc: ProcHandle,
    /// When background mode wrote a `build_<map>.running.json` sidecar, its path — removed by the
    /// reaper the moment the child exits so a stale sidecar can't trigger a phantom reattach.
    sidecar: Option<std::path::PathBuf>,
    log: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    /// Success flag — stored from the child's exit code. Read as the terminal outcome (`ok`).
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The single ATOMIC terminal signal: set true once the child has exited. Stored AFTER `done`
    /// so any thread that observes `exited==true` also observes the final `done` value — closing the
    /// race where a successful build was read as "finished but not-ok" (finding 10). `finished` is
    /// read from this, not from the streamed "[exit]" log line (which a reader thread pushes on its
    /// own schedule, independent of when `done` is stored).
    exited: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Some(exit-ok) once the child has been reaped.
    pub result: Option<bool>,
    /// Wall-clock build start — feeds the loading bar's ESTIMATED TIME readout.
    pub started: std::time::Instant,
    /// Monotonic max progress fraction seen for THIS build (as f32 bits). The loading bar must never
    /// regress, but a nested sub-script (`assemble_bevy`, the bake) emits its OWN `[STAGE i/M]` with a
    /// different M, which `build_frac` would map to a LOWER value — hence the "50% then back down"
    /// jumps. Clamping to this max fixes it. AtomicU32 = interior mutability through the `&BuildJob`
    /// the menu polls; a new build is a new BuildJob (starts at 0), so nothing leaks across builds.
    max_frac: std::sync::atomic::AtomicU32,
    /// Sticky "this build ran the ONE-TIME full extraction" latch, set by `build_view` the first
    /// time the stage marker names a fresh-extraction-only stage ("extract dataset ..."). It picks
    /// the first-build weight table in `build_frac`: on a fresh build stage 1 alone measured
    /// ~64% of wall-clock (streets, 2026-08-02) and the SH bake another ~19% — under the
    /// standard table the ETA overestimated all through a first build, then collapsed. Sticky so
    /// the table stays consistent AFTER stage 1 hands off (the marker text moves on). A reattach
    /// that lands post-extraction misses the latch and falls back to the standard table — today's
    /// behavior, and by then the bar is past 0.55 anyway.
    fresh_extract: std::sync::atomic::AtomicBool,
}

/// Best-effort per-platform spawn tweaks applied before `.spawn()`.
///
/// Windows: suppress the console window (CREATE_NO_WINDOW) always; for `detached` builds also
/// fully detach (DETACHED_PROCESS) so the child survives an Atlas close.
///
/// Unix: a child already survives its parent exiting (no console to suppress), so there's
/// nothing to do for that part. Instead, put it in its OWN process group (`setpgid(0, 0)`) —
/// `cancel()` below signals the whole group (negative pid) so grandchildren (eft_extract_v2,
/// the GPU bake, pip) die too, not just the direct python child. Done for every spawn, not just
/// `detached`, since the pipe path can be cancelled too.
fn prep_child(cmd: &mut std::process::Command, detached: bool) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        cmd.creation_flags(if detached { DETACHED_PROCESS | CREATE_NO_WINDOW } else { CREATE_NO_WINDOW });
    }
    #[cfg(unix)]
    {
        let _ = detached;
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
}

impl BuildJob {
    pub fn spawn(key: &str, game_dir: &str, force: bool, background: bool) -> std::io::Result<Self> {
        // GUI builds are ALWAYS --self-contained: the pack copies its textures/sidecars in and
        // references them pack-relative, so it stays valid when shipped to a friend (without it,
        // assemble_bevy bakes absolute machine paths and the pack loads untextured elsewhere).
        // `force` (menu UPDATE) adds --force so build_map.py re-extracts every game-derived artifact
        // instead of reusing stale data (release blocker).
        let mut extra: Vec<&str> = vec!["--self-contained"];
        if force {
            extra.push("--force");
        }
        // `background` (the "Process in background" toggle, default ON): only MAP builds detach — a
        // build is the long pipeline worth surviving an app-close; intel/deps are quick and have no
        // row to reattach, so they always use the inherited-pipe path.
        Self::spawn_script(key, game_dir, "tools/build_map.py", true, &extra, background)
    }

    /// The menu's tarkov.dev INTEL refresh (tools/sync_intel.py) — same streaming-job shape as a
    /// map build, no map args.
    pub fn spawn_intel() -> std::io::Result<Self> {
        Self::spawn_script("__intel__", "", "tools/sync_intel.py", false, &[], false)
    }

    /// One-click Python dependency setup (tools/setup_deps.py) — creates a venv + installs the
    /// extraction requirements, streamed into the build panel. Key "__deps__" tags the panel.
    pub fn spawn_setup() -> std::io::Result<Self> {
        Self::spawn_script("__deps__", "", "tools/setup_deps.py", false, &[], false)
    }

    fn spawn_script(
        key: &str,
        game_dir: &str,
        script: &str,
        key_as_args: bool,
        extra_args: &[&str],
        background: bool,
    ) -> std::io::Result<Self> {
        use std::io::BufRead;
        let root = crate::paths::repo_root().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "python kit not found (tools/ beside the exe or in the cwd)",
            )
        })?;
        // The datasets dir + writable working dir the pipeline reads from env (previously unset ->
        // the kit fell back to the original dev machine's hardcoded path -> "no dataset"). Ensure
        // out/ exists so bake/intel/sync can write there.
        let assets_root = detect_assets_dir();
        let tarkmap_root = detect_tarkmap_dir();
        let _ = std::fs::create_dir_all(std::path::Path::new(&tarkmap_root).join("out"));
        let mut cmd = std::process::Command::new(crate::paths::python_exe(root));
        cmd.current_dir(root) // kit-relative paths inside the script resolve from its root
            .env("EFT_GAME_DATA", game_dir) // menu-selected install drives the pipeline
            .env("EFT_ASSETS_ROOT", &assets_root)
            .env("EFT_TARKMAP_ROOT", &tarkmap_root)
            .arg(script);
        // Hand the build the EXACT running viewer exe so its nav stage can invoke the portable
        // `atlas bake-nav` baker without guessing at target/ paths (finding: routing by default).
        if let Ok(exe) = std::env::current_exe() {
            cmd.env("EFT_ATLAS_EXE", exe);
        }
        // Settings ▸ General ▸ "Force CPU processing": the env var inherits through python down to
        // every `atlas bake-*` child — sh bake takes the rayon path, the terrain composite declines
        // so the extractor uses its numpy fallback. Read from config at spawn time so the latest
        // setting always applies without re-plumbing state through the job system.
        if config_force_cpu_process() {
            cmd.env("EFT_BAKE_CPU", "1");
        }
        if key_as_args {
            cmd.args(key.split([',', ' ']).filter(|s| !s.is_empty()));
        }
        cmd.args(extra_args);
        let log = std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::<String>::with_capacity(512),
        ));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let exited = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        if background {
            // ---- DETACHED + FILE OUTPUT: the build SURVIVES an Atlas close. ----
            // stdout+stderr are redirected to build_<key>.log (NOT an inherited pipe that would
            // break — and kill build_map.py — the moment the parent exits), stdin is null, and the
            // child is spawned DETACHED (no console, not tied to Atlas's process/stdio). Result: if
            // the user closes Atlas mid-build, the python pipeline keeps writing the log + the pack.
            let logs = build_logs_dir();
            let _ = std::fs::create_dir_all(&logs);
            let log_path = build_log_path(key);
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&log_path)?;
            let file2 = file.try_clone()?;
            prep_child(&mut cmd, true);
            let child = cmd
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::from(file))
                .stderr(std::process::Stdio::from(file2))
                .spawn()?;
            let pid = child.id();
            // Sidecar { pid, map, started, force } so a later launch can DETECT + reattach/clean this
            // build. `force` is recovered from the args we're about to run with.
            let sidecar = build_running_path(key);
            let force = extra_args.contains(&"--force");
            let manifest = serde_json::json!({
                "pid": pid,
                "map": key,
                "started": now_epoch(),
                "force": force,
            });
            let _ =
                std::fs::write(&sidecar, serde_json::to_string_pretty(&manifest).unwrap_or_default());
            // The live panel TAILS the file (same [STAGE]/[SUBPROGRESS] parsing as the pipe path).
            spawn_log_tailer(log_path, log.clone(), exited.clone());
            // While Atlas stays open we still OWN the child, so reap via try_wait (accurate exit code)
            // and drop the sidecar the instant it exits.
            let child = std::sync::Arc::new(std::sync::Mutex::new(child));
            spawn_owned_reaper(
                child.clone(),
                log.clone(),
                Some(sidecar.clone()),
                done.clone(),
                exited.clone(),
            );
            return Ok(Self {
                key: key.to_string(),
                proc: ProcHandle::Owned(child),
                sidecar: Some(sidecar),
                log,
                done,
                exited,
                result: None,
                started: std::time::Instant::now(),
                max_frac: std::sync::atomic::AtomicU32::new(0),
                fresh_extract: std::sync::atomic::AtomicBool::new(false),
            });
        }

        // ---- Legacy inherited-pipe path (toggle OFF; intel/deps always). Streams stdout+stderr
        // through PIPEs read by reader threads. Dies with Atlas (the pipe closes) — acceptable for
        // the OFF toggle and the quick intel/deps jobs. ----
        prep_child(&mut cmd, false);
        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        for pipe in [
            child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let log = log.clone();
            std::thread::spawn(move || {
                let rd = std::io::BufReader::new(pipe);
                for line in rd.lines().map_while(Result::ok) {
                    push_capped(&log, line);
                }
            });
        }
        let child = std::sync::Arc::new(std::sync::Mutex::new(child));
        spawn_owned_reaper(child.clone(), log.clone(), None, done.clone(), exited.clone());
        Ok(Self {
            key: key.to_string(),
            proc: ProcHandle::Owned(child),
            sidecar: None,
            log,
            done,
            exited,
            result: None,
            started: std::time::Instant::now(),
            max_frac: std::sync::atomic::AtomicU32::new(0),
            fresh_extract: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// REATTACH to a build that a PREVIOUS Atlas process started detached (toggle ON) and left
    /// still-running when the app was closed. We only have its PID + log file (its `Child` died with
    /// the old process), so the panel tails the file and a liveness reaper watches the PID — when it
    /// dies we decide success from the pack/`[BUILD OK]` marker. `started` is reconstructed from the
    /// sidecar so the ETA readout is continuous.
    pub fn attach(
        key: &str,
        pid: u32,
        log_path: std::path::PathBuf,
        sidecar: std::path::PathBuf,
        started: std::time::Instant,
    ) -> Self {
        let log = std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::<String>::with_capacity(512),
        ));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let exited = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Tail from offset 0 so the panel back-fills the build's history (last 500 lines) and finds
        // the latest [STAGE] marker immediately.
        spawn_log_tailer(log_path, log.clone(), exited.clone());
        spawn_detached_reaper(
            pid,
            key.to_string(),
            log.clone(),
            Some(sidecar.clone()),
            done.clone(),
            exited.clone(),
        );
        Self {
            key: key.to_string(),
            proc: ProcHandle::Detached(pid),
            sidecar: Some(sidecar),
            log,
            done,
            exited,
            result: None,
            started,
            max_frac: std::sync::atomic::AtomicU32::new(0),
            fresh_extract: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Build an ALREADY-FINISHED job from a detached build's leftover log — used by the startup scan
    /// when a build died while Atlas was closed WITHOUT producing its pack (interrupted). Loads the
    /// log tail so the panel can show what happened; `ok` is the (usually false) terminal outcome.
    pub fn from_finished_log(key: &str, log_path: &std::path::Path, ok: bool) -> Self {
        let log = std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::<String>::with_capacity(512),
        ));
        if let Ok(content) = std::fs::read_to_string(log_path) {
            let lines: Vec<&str> = content.lines().collect();
            let start = lines.len().saturating_sub(500);
            for line in &lines[start..] {
                push_capped(&log, (*line).to_string());
            }
        }
        Self {
            key: key.to_string(),
            proc: ProcHandle::Detached(0),
            sidecar: None,
            log,
            done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(ok)),
            exited: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            result: Some(ok),
            started: std::time::Instant::now(),
            max_frac: std::sync::atomic::AtomicU32::new(0),
            fresh_extract: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn cancel(&self) {
        // Kill the WHOLE process tree. build_map.py / setup_deps.py spawn heavy child processes
        // (eft_extract_v2, the GPU bake, pip). Killing only the direct child would orphan them —
        // the extraction would keep churning after CANCEL. The direct .kill() below is a
        // fallback if the PID has already been reaped. Works for a reattached (Detached) build
        // too: its PID is still reachable by taskkill / the process-group signal.
        let pid = match &self.proc {
            ProcHandle::Owned(c) => c.lock().ok().map(|c| c.id()),
            ProcHandle::Detached(p) => Some(*p),
        };
        if let Some(pid) = pid.filter(|p| *p != 0) {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .creation_flags(0x0800_0000)
                    .output();
            }
            #[cfg(unix)]
            {
                // Every child is spawned in its own process group (see `prep_child`), so
                // signalling the NEGATIVE pid hits the whole tree, not just the direct child.
                let _ = std::process::Command::new("kill")
                    .args(["-9", &format!("-{pid}")])
                    .output();
            }
        }
        if let ProcHandle::Owned(c) = &self.proc {
            if let Ok(mut c) = c.lock() {
                let _ = c.kill();
            }
        }
    }

    /// Entire captured log (for the COPY LOG button — the panel only shows a tail).
    pub fn full_log(&self) -> String {
        let l = self.log.lock().unwrap();
        let mut s = String::with_capacity(l.iter().map(|x| x.len() + 1).sum());
        for line in l.iter() {
            s.push_str(line);
            s.push('\n');
        }
        s
    }

    /// (last stage marker, tail lines, finished?, success?)
    pub fn snapshot(&self, tail: usize) -> (String, Vec<String>, bool, bool) {
        let l = self.log.lock().unwrap();
        let stage = l
            .iter()
            .rev()
            .find(|s| s.starts_with("[STAGE") || s.starts_with("[BUILD") || s.starts_with("[SYNC"))
            .cloned()
            .unwrap_or_else(|| "starting...".into());
        let lines: Vec<String> = l.iter().rev().take(tail).cloned().collect();
        // Terminal state from the atomics, NOT the streamed log (finding 10): `exited` is published
        // AFTER `done`, so reading them in this order can never see finished-without-the-final-ok.
        use std::sync::atomic::Ordering::SeqCst;
        let finished = self.exited.load(SeqCst);
        let ok = self.done.load(SeqCst);
        (stage, lines.into_iter().rev().collect(), finished, ok)
    }
}

// ---- background-build plumbing (detached child + log-file tail + reattach) --------------------
//
// `<packs>/logs/build_<map>.log`      : the detached child's stdout+stderr (survives an app-close).
// `<packs>/logs/build_<map>.running.json` : { pid, map, started, force } — a live build's sidecar,
//                                       written at spawn and removed by the reaper on exit; the ONLY
//                                       durable record a fresh Atlas launch uses to reattach/clean.

/// Where a background build streams its log + drops its running sidecar. Under `packs/` so it lives
/// beside the pack it produces and is writable wherever packs are (finding 12 location rules apply).
fn build_logs_dir() -> std::path::PathBuf {
    crate::paths::packs_root().join("logs")
}
fn build_log_path(key: &str) -> std::path::PathBuf {
    build_logs_dir().join(format!("build_{key}.log"))
}
fn build_running_path(key: &str) -> std::path::PathBuf {
    build_logs_dir().join(format!("build_{key}.running.json"))
}

/// Wall-clock seconds since the epoch (for the running-sidecar `started` stamp). 0 on the (never
/// observed on Windows) clock-before-epoch error — panic=abort forbids unwrapping the Result.
fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Append a line to the shared log ring, capped at 500 (matches the old inline closure). A poisoned
/// mutex (only possible if a holder panicked — impossible under panic=abort) is silently skipped.
fn push_capped(
    log: &std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    line: String,
) {
    if let Ok(mut l) = log.lock() {
        if l.len() >= 500 {
            l.pop_front();
        }
        l.push_back(line);
    }
}

/// Is a process with this PID currently running? Uses `tasklist` (no extra crate / no winapi): the
/// CSV row for a live PID contains the quoted id; the "no tasks" notice does not.
#[cfg(windows)]
fn pid_alive(pid: u32) -> bool {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .creation_flags(0x0800_0000)
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\"")),
        Err(_) => false,
    }
}

/// Is a process with this PID currently running? `kill -0` sends no signal, just checks
/// existence + permission — no extra crate needed for a Unix "does this PID exist" probe.
#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A pack for `key` looks present on disk (its manifest exists). The rigorous validity check lives
/// in `scan`; this is just the completion signal for a reattached build whose exit code we never saw.
fn pack_present(key: &str) -> bool {
    crate::paths::packs_root().join(format!("{key}.eftpack")).join("manifest.json").is_file()
}

/// Does the captured log contain `marker` anywhere in its (capped) tail?
fn log_has(
    log: &std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    marker: &str,
) -> bool {
    log.lock().map(|l| l.iter().any(|s| s.contains(marker))).unwrap_or(false)
}

/// Read all currently-available bytes from a growing log file into `pending`, emitting each COMPLETE
/// (`\n`-terminated) line into `log` — byte-accurate so a `[STAGE …]` marker is never torn across a
/// read boundary. A partial trailing line stays in `pending` for the next call.
fn drain_log_file(
    file: &mut std::fs::File,
    pending: &mut Vec<u8>,
    log: &std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
) {
    use std::io::Read;
    let mut buf = [0u8; 8192];
    loop {
        match file.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                pending.extend_from_slice(&buf[..n]);
                while let Some(pos) = pending.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = pending.drain(..=pos).collect();
                    let s = String::from_utf8_lossy(&line);
                    push_capped(log, s.trim_end_matches(['\r', '\n']).to_string());
                }
            }
        }
    }
}

/// Tail a growing log FILE into `log` until `exited` is set and the file is fully drained. Used by
/// BOTH a fresh detached-file build and a reattached one, so the panel + `snapshot` read `log`
/// exactly as they did for the inherited-pipe path.
fn spawn_log_tailer(
    path: std::path::PathBuf,
    log: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    exited: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering::SeqCst;
    std::thread::spawn(move || {
        // The file is created by the spawner before this thread starts (or already exists on
        // reattach); open defensively with a few retries in case we win the race.
        let mut file = None;
        for _ in 0..100 {
            match std::fs::File::open(&path) {
                Ok(f) => {
                    file = Some(f);
                    break;
                }
                Err(_) => {
                    if exited.load(SeqCst) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
        let Some(mut file) = file else { return };
        let mut pending: Vec<u8> = Vec::new();
        loop {
            drain_log_file(&mut file, &mut pending, &log);
            if exited.load(SeqCst) {
                // The child has fully exited — one last drain catches its final flushed lines.
                drain_log_file(&mut file, &mut pending, &log);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        // Emit any trailing line the process wrote without a newline.
        if !pending.is_empty() {
            let s = String::from_utf8_lossy(&pending);
            let t = s.trim_end_matches(['\r', '\n']);
            if !t.is_empty() {
                push_capped(&log, t.to_string());
            }
        }
    });
}

/// Reap a child we OWN (`try_wait`): store the exit-code success, publish `exited` (AFTER `done` —
/// finding 10 ordering), stream an `[exit N]` line, and remove the running sidecar (if any). Serves
/// both the inherited-pipe and the detached-file-while-open paths.
fn spawn_owned_reaper(
    child: std::sync::Arc<std::sync::Mutex<std::process::Child>>,
    log: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    sidecar: Option<std::path::PathBuf>,
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    exited: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering::SeqCst;
    std::thread::spawn(move || loop {
        let status = match child.lock() {
            Ok(mut c) => c.try_wait(),
            Err(_) => break,
        };
        match status {
            Ok(Some(st)) => {
                done.store(st.success(), SeqCst);
                exited.store(true, SeqCst);
                push_capped(
                    &log,
                    format!("[exit {}]", st.code().map_or("?".into(), |c| c.to_string())),
                );
                if let Some(s) = &sidecar {
                    let _ = std::fs::remove_file(s);
                }
                break;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(300)),
            Err(_) => break,
        }
    });
}

/// Reap a REATTACHED build (we hold only its PID) by polling liveness. When the PID dies we can't
/// read an exit code, so success = the pack appeared OR the log shows `[BUILD OK]`. Removes the
/// sidecar so it can't trigger a phantom reattach next launch.
fn spawn_detached_reaper(
    pid: u32,
    key: String,
    log: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    sidecar: Option<std::path::PathBuf>,
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    exited: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering::SeqCst;
    std::thread::spawn(move || loop {
        if exited.load(SeqCst) {
            break;
        }
        if !pid_alive(pid) {
            // Give the tailer a moment to catch the final flushed lines ([BUILD OK] / traceback).
            std::thread::sleep(std::time::Duration::from_millis(700));
            let ok = pack_present(&key) || log_has(&log, "[BUILD OK]");
            done.store(ok, SeqCst);
            exited.store(true, SeqCst);
            if let Some(s) = &sidecar {
                let _ = std::fs::remove_file(s);
            }
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
    });
}

/// STARTUP REATTACH SCAN (called once on the first menu frame). Look for `build_<map>.running.json`
/// sidecars left by a detached build from a previous Atlas run:
///   * PID still ALIVE  -> reattach as the in-flight job (panel tails its log, row shows BUILDING).
///   * PID gone, pack present / `[BUILD OK]` -> it finished while we were closed: drop the sidecar +
///     log (the normal pack rescan already shows the row installed).
///   * PID gone, no pack -> interrupted: surface it as a finished-failed outcome (with its log) so
///     the user can resume/rebuild, and drop the sidecar.
/// Only ONE build can be in flight, so the first alive one is adopted; any others are left untouched.
fn reattach_builds(worker: &mut crate::jobs::JobWorker, game_dir: &str) {
    let Ok(rd) = std::fs::read_dir(build_logs_dir()) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(map) = name.strip_prefix("build_").and_then(|s| s.strip_suffix(".running.json"))
        else {
            continue;
        };
        let sidecar = e.path();
        let Some((pid, started_epoch, force)) = read_running_sidecar(&sidecar) else {
            let _ = std::fs::remove_file(&sidecar); // corrupt/partial sidecar — clean it up
            continue;
        };
        let log_path = build_log_path(map);
        if pid_alive(pid) {
            if !worker.busy() {
                let started = instant_from_epoch(started_epoch);
                let bj = BuildJob::attach(map, pid, log_path, sidecar.clone(), started);
                worker.reattach(
                    crate::jobs::Job::BuildMap {
                        map: map.to_string(),
                        game_dir: game_dir.to_string(),
                        force,
                        background: true,
                    },
                    bj,
                );
            }
            // else: a job is already in flight — leave this alive build's sidecar for a later launch.
            continue;
        }
        // PID gone.
        let finished_ok = pack_present(map) || file_has_build_ok(&log_path);
        let _ = std::fs::remove_file(&sidecar);
        if finished_ok {
            let _ = std::fs::remove_file(&log_path); // the pack rescan shows the row installed
        } else {
            let bj = BuildJob::from_finished_log(map, &log_path, false);
            worker.set_finished(
                crate::jobs::Job::BuildMap {
                    map: map.to_string(),
                    game_dir: game_dir.to_string(),
                    force,
                    background: true,
                },
                bj,
            );
        }
    }
}

/// Parse a `build_<map>.running.json` sidecar -> (pid, started_epoch, force). None if unreadable.
fn read_running_sidecar(path: &std::path::Path) -> Option<(u32, u64, bool)> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let pid = v.get("pid").and_then(|x| x.as_u64())? as u32;
    let started = v.get("started").and_then(|x| x.as_u64()).unwrap_or(0);
    let force = v.get("force").and_then(|x| x.as_bool()).unwrap_or(false);
    Some((pid, started, force))
}

/// Reconstruct a build's start `Instant` from its epoch stamp so the ETA readout stays continuous
/// across the app restart. Falls back to "now" if the clock math underflows.
fn instant_from_epoch(started_epoch: u64) -> std::time::Instant {
    let now = now_epoch();
    let elapsed = now.saturating_sub(started_epoch);
    std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(elapsed))
        .unwrap_or_else(std::time::Instant::now)
}

/// Whole-file check for the `[BUILD OK]` success marker (the reattach scan's log evidence, used when
/// the pack isn't detectable). Reads the file directly since there's no in-memory ring yet.
fn file_has_build_ok(log_path: &std::path::Path) -> bool {
    std::fs::read_to_string(log_path).map(|s| s.contains("[BUILD OK]")).unwrap_or(false)
}

/// Present ONLY in menu mode (bare launch, no pack): drives the fullscreen menu UI and
/// suppresses the in-raid panels (ui.rs checks for this resource).
#[derive(Resource)]
pub struct MenuState {
    pub entries: Vec<MapEntry>,
    pub game_fp: Option<String>,
    pub game_dir: String,
    pub total_bytes: u64,
    /// Index pending delete confirmation.
    pub confirm_delete: Option<usize>,
    /// Index showing rebuild instructions.
    pub show_rebuild: Option<usize>,
    /// Build panel: raw log tail visible? Collapsed by default; auto-expands on failure.
    pub show_log: bool,
    /// INTEL/sync strip: raw sync-log tail visible? The sync streams into the worker log like a
    /// build, but `build_view` ignores sync jobs (no map row), so the build panel never surfaces it
    /// — this drives an inline Show log / Copy log under the sync row instead. Auto-opens on failure.
    pub show_sync_log: bool,
    /// Footer editor buffer for the game-install path.
    pub game_dir_edit: String,
    /// Where extracted datasets live (EFT_ASSETS_ROOT) + its footer editor buffer, and whether the
    /// user has explicitly configured it (false => show the first-run onboarding banner).
    pub assets_dir: String,
    pub assets_dir_edit: String,
    pub assets_ok: bool,
    /// Whether the build python has UnityPy/numpy/Pillow (None probe failure => treat as ok so we
    /// don't nag when there's no kit to build with anyway).
    pub deps_ok: bool,
    /// Whether the Python build KIT (tools/build_map.py etc.) is present beside the exe / in the cwd
    /// (`repo_root().is_some()`). A shipped-lite bundle has NO kit, so BUILD/UPDATE must be disabled
    /// there (finding 11) — otherwise clicking them just hits a spawn error mid-pipeline.
    pub build_kit_available: bool,
    /// INTEL strip stats: (loot.json age days, tasks.json age days, cached icon count).
    pub intel: (Option<f64>, Option<f64>, usize),
    /// Outcome note of the last finished sync ("refreshed" / failure), shown until the next one.
    pub sync_note: Option<(String, bool)>,
    /// `JobWorker.completed` value we last reacted to — a change means a job finished, so rescan
    /// the pack list / intel and (for a sync) set the note. Builds/syncs now run on the shared
    /// worker, not owned by MenuState.
    pub seen_completed: u64,
    /// EFT_MENU_BUILD auto-build map key, enqueued once on the first menu frame (CLI/testing hook).
    pub autobuild: Option<String>,
    /// Set when a config write FAILED (finding 12) — shown as a warning by the footer so the user
    /// knows a setting won't persist, instead of it silently reverting on the next launch.
    pub config_err: Option<String>,
    /// "Process in background" toggle (default ON): a MAP build detaches + logs to a file so it keeps
    /// running even if Atlas is closed. Persisted in atlas.config.json; the footer checkbox flips it.
    pub process_in_background: bool,
    /// "Force CPU processing" (default OFF): map builds bake lighting/terrain on the CPU instead of
    /// the GPU (spawner exports EFT_BAKE_CPU=1). For machines whose GPU driver dies in the bakes.
    pub force_cpu_process: bool,
    /// "Screenshot to locate current position" — poll the EFT screenshot folder and move the
    /// camera onto each new position fix (see `config_screenshot_locate`).
    pub screenshot_locate: bool,
    /// Overlay settings (the menu edits them; `overlay::OverlayPlugin` reads the same keys back
    /// out of atlas.config.json at startup). One struct so a settings panel can bind to it.
    pub overlay: crate::overlay::OverlayConfig,
    /// Texture quality: 0 = full, 1 = half, 2 = quarter (mip levels dropped at upload).
    pub tex_quality: u8,
    /// Quality preset index (see `render::QualityPreset`). Chosen in the MENU and carried into the
    /// scene by main.rs, because texture quality is applied while a map loads.
    pub quality_preset: u8,
    /// Which SETTINGS tab the right-hand panel shows (0 = Overlay, 1 = Live link, 2 = General).
    /// `None` = the panel is collapsed and the map list has the full width.
    pub settings_tab: Option<u8>,
    /// One-shot guard: on the first menu frame we scan for detached builds a previous run left running
    /// and reattach/surface them. Set true after that scan so it never re-runs.
    pub reattached: bool,
}

/// Shared tarkov.dev data freshness: (loot.json age d, tasks.json age d, icon count).
pub fn intel_status() -> (Option<f64>, Option<f64>, usize) {
    let sh = crate::paths::shared_dir();
    (
        age_days(&sh.join("loot.json")),
        age_days(&sh.join("tasks.json")),
        std::fs::read_dir(sh.join("icons")).map(|d| d.count()).unwrap_or(0),
    )
}

// The map roster (dataset key -> display name) is no longer hardcoded here: it is generated from
// the game's BuildSettings scene list into extraction/maps/manifest.json and read via
// `crate::maps::known_pairs()` / `crate::maps::roster()`. Packs on disk that aren't in the roster
// still show up (title falls back to the key). Factory resolves to the 1.0 rework (id
// `factory_rework`) in the manifest, the version the live game loads.

/// FNV-1a 64 over "name|size|mtime_s;" of the game's top-level asset files, sorted by name.
/// MUST stay byte-identical to tools/stamp_fingerprint.py (the python stamper).
pub fn game_fingerprint(game_data: &str) -> Option<String> {
    let mut entries: Vec<(String, u64, u64)> = Vec::new();
    for e in std::fs::read_dir(game_data).ok()? {
        let Ok(e) = e else { continue };
        let Ok(md) = e.metadata() else { continue };
        if !md.is_file() {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        if !(name.starts_with("level")
            || name.ends_with(".assets")
            || name.ends_with(".resS")
            || name.ends_with(".resource"))
        {
            continue;
        }
        let mtime = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())?;
        entries.push((name, md.len(), mtime));
    }
    if entries.is_empty() {
        return None;
    }
    entries.sort();
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for (n, size, mt) in &entries {
        for b in format!("{n}|{size}|{mt};").bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1_0000_0001_B3);
        }
    }
    Some(format!("{h:016x}"))
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(path) {
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                if md.is_dir() {
                    total += dir_size(&e.path());
                } else {
                    total += md.len();
                }
            }
        }
    }
    total
}

fn age_days(path: &std::path::Path) -> Option<f64> {
    let md = std::fs::metadata(path).ok()?;
    let m = md.modified().ok()?;
    Some(m.elapsed().ok()?.as_secs_f64() / 86_400.0)
}

/// Scan packs/ and build the menu model. Called at startup and after a delete.
pub fn scan(game_fp: &Option<String>) -> (Vec<MapEntry>, u64) {
    let mut entries: Vec<MapEntry> = Vec::new();
    let mut total = 0u64;
    let installed = |key: &str| {
        crate::paths::packs_root()
            .join(format!("{key}.eftpack"))
            .to_string_lossy()
            .into_owned()
    };
    // Known roster first, then any extra packs on disk.
    let mut extra: Vec<String> = std::fs::read_dir(crate::paths::packs_root())
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.strip_suffix(".eftpack").map(str::to_string)
                })
                .filter(|k| crate::maps::known_pairs().iter().all(|(key, _)| key != k))
                .collect()
        })
        .unwrap_or_default();
    extra.sort();
    let roster = crate::maps::known_pairs()
        .iter()
        .map(|(k, t)| (*k, *t))
        .chain(extra.iter().map(|k| (k.as_str(), k.as_str())));
    for (key, title) in roster {
        let dir = installed(key);
        let p = std::path::Path::new(&dir);
        let manifest = p.join("manifest.json");
        if !manifest.is_file() {
            entries.push(MapEntry {
                title: title.to_string(),
                key: key.to_string(),
                pack_dir: None,
                size_bytes: 0,
                built_days: None,
                intel_days: None,
                has_volume: false,
                has_grass: false,
                has_gamedata: false,
                has_icons: false,
                fp_match: None,
                valid: true, // not installed: nothing to be broken
            });
            continue;
        }
        let size = dir_size(p);
        total += size;
        // Parse the manifest once; a parse failure is itself a "damaged pack" signal (finding 4).
        let man_parsed: Option<serde_json::Value> = std::fs::read_to_string(&manifest)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        // Required binary payloads a loadable pack must carry. Without them PLAY opens a blank
        // window (the async load fails) — so mark the pack invalid rather than showing it READY.
        let valid = man_parsed.is_some()
            && p.join("meshes.bin").is_file()
            && p.join("instances.bin").is_file();
        let man: serde_json::Value = man_parsed.unwrap_or_default();
        let stamped = man.get("sourceFingerprint").and_then(|v| v.as_str());
        let fp_match = match (stamped, game_fp.as_deref()) {
            (Some(s), Some(g)) => Some(s == g),
            _ => None,
        };
        let has_volume = man
            .get("sidecars")
            .and_then(|s| s.get("volume"))
            .and_then(|v| v.as_str())
            .map(|v| std::path::Path::new(v).exists() || p.join(v).exists())
            .unwrap_or(false)
            || p.join("volume.bin").exists();
        entries.push(MapEntry {
            title: title.to_string(),
            key: key.to_string(),
            pack_dir: Some(dir.clone()),
            size_bytes: size,
            built_days: age_days(&manifest),
            intel_days: age_days(&p.join("loot.json"))
                .or_else(|| age_days(&crate::paths::shared_dir().join("loot.json"))),
            has_volume,
            has_grass: p.join("grass.bin").exists(),
            has_gamedata: p.join("gamedata.json").exists(),
            has_icons: p.join("icons").is_dir()
                || crate::paths::shared_dir().join("icons").is_dir(),
            fp_match,
            valid,
        });
    }
    (entries, total)
}

/// Small persisted viewer config beside the EXE (portable-app style; cwd fallback for old
/// installs) — resolution in paths::config_path.
fn config_path() -> std::path::PathBuf {
    crate::paths::config_path()
}

/// Read atlas.config.json, tolerating a UTF-8 BOM.
///
/// `serde_json` rejects a leading BOM, and a rejected parse means EVERY setting silently
/// reverts to its default - language, game dir, overlay layout, quality preset, the lot. Any
/// editor that writes UTF-8-with-BOM (Notepad, PowerShell's `Out-File -Encoding utf8`) produces
/// one, and the failure is invisible: the file looks perfectly fine. Strip it on read.
fn read_config_text() -> Option<String> {
    let t = std::fs::read_to_string(config_path()).ok()?;
    Some(t.strip_prefix('\u{feff}').map(str::to_string).unwrap_or(t))
}

fn config_game_dir() -> Option<String> {
    let v: serde_json::Value =
        serde_json::from_str(&read_config_text()?).ok()?;
    v.get("gameData").and_then(|s| s.as_str()).map(str::to_string)
}

pub fn save_config_game_dir(dir: &str) -> bool {
    save_config_str("gameData", dir)
}

/// Generic single-key read from atlas.config.json (the game-dir helpers are the canonical example).
fn config_str(key: &str) -> Option<String> {
    let v: serde_json::Value =
        serde_json::from_str(&read_config_text()?).ok()?;
    v.get(key).and_then(|s| s.as_str()).map(str::to_string)
}

/// Generic single-key read-modify-write into atlas.config.json (preserves other keys). Returns
/// false when the write FAILED (finding 12): the path is now the writable user-profile location,
/// but a failure is no longer swallowed — it's logged and the boolean lets the caller warn the user
/// their setting won't persist, instead of the old silent revert-on-restart.
#[must_use]
fn save_config_str(key: &str, val: &str) -> bool {
    let path = config_path();
    // MUST go through `read_config_text` (BOM-tolerant) like the readers do. With a raw read, a
    // BOM'd config fails to parse here, this starts from `{}` and the write then DESTROYS every
    // other setting - game dir, language, overlay layout, the lot. Making only the readers
    // BOM-tolerant turned a visible all-defaults revert into silent data loss on first save.
    let mut v: serde_json::Value = read_config_text()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    v[key] = serde_json::Value::String(val.to_string());
    // Ensure the parent dir exists (%APPDATA%\atlas may not yet) before writing.
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap_or_default()) {
        Ok(()) => true,
        Err(e) => {
            error!("menu: could not save config to {}: {e}", path.display());
            false
        }
    }
}

/// String variants used by modules that own settings in the shared atlas.config.json.
pub fn config_str_pub(key: &str) -> Option<String> {
    config_str(key)
}
#[must_use]
pub fn save_config_str_pub(key: &str, val: &str) -> bool {
    save_config_str(key, val)
}

/// Generic single-key BOOL read from atlas.config.json (mirrors `config_str`). None when the key is
/// absent / not a bool — callers supply the default.
/// Public wrappers so other modules (overlay.rs) can persist their own settings through the SAME
/// atlas.config.json read-modify-write path instead of inventing a second config file.
pub fn config_bool_pub(key: &str) -> Option<bool> {
    config_bool(key)
}
pub fn save_config_bool_pub(key: &str, val: bool) -> bool {
    save_config_bool(key, val)
}
/// f32 variants (overlay geometry / fps cap). Stored as JSON numbers.
pub fn config_f32_pub(key: &str) -> Option<f32> {
    let v: serde_json::Value =
        serde_json::from_str(&read_config_text()?).ok()?;
    v.get(key).and_then(|n| n.as_f64()).map(|n| n as f32)
}
#[must_use]
pub fn save_config_f32_pub(key: &str, val: f32) -> bool {
    let path = config_path();
    // MUST go through `read_config_text` (BOM-tolerant) like the readers do. With a raw read, a
    // BOM'd config fails to parse here, this starts from `{}` and the write then DESTROYS every
    // other setting - game dir, language, overlay layout, the lot. Making only the readers
    // BOM-tolerant turned a visible all-defaults revert into silent data loss on first save.
    let mut v: serde_json::Value = read_config_text()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    v[key] = serde_json::json!(val);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap_or_default()) {
        Ok(()) => true,
        Err(e) => {
            error!("menu: could not save config to {}: {e}", path.display());
            false
        }
    }
}

/// Persist a named quality preset and the texture tier it owns in ONE read-modify-write.
///
/// These used to be two independent config writes. A second Atlas process or another settings
/// system could update the file between them, leaving combinations such as `Ultra + Quarter`.
/// Rendering already treated the preset as authoritative, but the menu then displayed the stale
/// texture tier and looked broken. One write makes the pair indivisible.
#[must_use]
pub fn save_quality_preset_pub(preset: crate::render::QualityPreset) -> bool {
    let path = config_path();
    let mut v: serde_json::Value = read_config_text()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    v["qualityPreset"] = serde_json::json!(preset.index() as f32);
    if let Some(q) = preset.tex_quality() {
        v["textureQuality"] = serde_json::json!(q as f32);
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap_or_default()) {
        Ok(()) => true,
        Err(e) => {
            error!("menu: could not save quality preset to {}: {e}", path.display());
            false
        }
    }
}

fn config_bool(key: &str) -> Option<bool> {
    let v: serde_json::Value =
        serde_json::from_str(&read_config_text()?).ok()?;
    v.get(key).and_then(|b| b.as_bool())
}

/// Generic single-key BOOL read-modify-write into atlas.config.json (preserves other keys). Returns
/// false when the write FAILED, exactly like `save_config_str` (the caller warns the user).
#[must_use]
fn save_config_bool(key: &str, val: bool) -> bool {
    let path = config_path();
    // MUST go through `read_config_text` (BOM-tolerant) like the readers do. With a raw read, a
    // BOM'd config fails to parse here, this starts from `{}` and the write then DESTROYS every
    // other setting - game dir, language, overlay layout, the lot. Making only the readers
    // BOM-tolerant turned a visible all-defaults revert into silent data loss on first save.
    let mut v: serde_json::Value = read_config_text()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    v[key] = serde_json::Value::Bool(val);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap_or_default()) {
        Ok(()) => true,
        Err(e) => {
            error!("menu: could not save config to {}: {e}", path.display());
            false
        }
    }
}

/// "Process in background" toggle (default ON): whether a MAP build detaches + streams to a log file
/// so it SURVIVES closing Atlas. Persisted in atlas.config.json under `processInBackground`.
pub fn config_process_in_background() -> bool {
    config_bool("processInBackground").unwrap_or(true)
}
pub fn save_config_process_in_background(on: bool) -> bool {
    save_config_bool("processInBackground", on)
}

/// "Screenshot to locate current position" (default ON): whether the game link POLLS the EFT
/// screenshot folder. EFT writes the player's world position + view quaternion into every
/// screenshot filename, so a screenshot taken in raid is a free position fix — Atlas puts the
/// camera exactly where the player stands, looking the same way. Off = the folder is never read.
/// Persisted in atlas.config.json under `screenshotLocate`.
/// "Force CPU processing" (default OFF): map builds bake lighting + terrain on the CPU — the
/// spawner exports EFT_BAKE_CPU=1 into the build child's environment. The override for machines
/// whose GPU driver crashes or TDRs inside the compute bakes (e.g. a bad Vulkan ICD). Persisted
/// in atlas.config.json under `forceCpuProcessing`.
pub fn config_force_cpu_process() -> bool {
    config_bool("forceCpuProcessing").unwrap_or(false)
}
pub fn save_config_force_cpu_process(on: bool) -> bool {
    save_config_bool("forceCpuProcessing", on)
}

pub fn config_screenshot_locate() -> bool {
    config_bool("screenshotLocate").unwrap_or(true)
}
pub fn save_config_screenshot_locate(on: bool) -> bool {
    save_config_bool("screenshotLocate", on)
}

/// True once the user (or an env var) has chosen where extracted datasets live — drives the
/// first-run onboarding gate. The default location existing on disk does NOT count as "configured".
pub fn assets_configured() -> bool {
    std::env::var("EFT_ASSETS_ROOT").map(|d| !d.is_empty()).unwrap_or(false)
        || config_str("assetsRoot").map(|d| !d.is_empty()).unwrap_or(false)
}

/// The extracted-datasets dir (`EFT_ASSETS_ROOT` the Python kit reads): env > saved config >
/// `<exe>/eft_assets` default. The full game extraction WRITES here and every build READS here.
pub fn detect_assets_dir() -> String {
    if let Ok(d) = std::env::var("EFT_ASSETS_ROOT") {
        if !d.is_empty() {
            return d;
        }
    }
    if let Some(d) = config_str("assetsRoot").filter(|d| !d.is_empty()) {
        return d;
    }
    crate::paths::exe_dir().join("eft_assets").to_string_lossy().into_owned()
}

pub fn save_config_assets_dir(dir: &str) -> bool {
    save_config_str("assetsRoot", dir)
}

/// UI language override ("en"/"ru") persisted by the menu's language toggle (None = auto-detect).
pub fn config_lang() -> Option<String> {
    config_str("lang")
}
pub fn save_config_lang(tag: &str) -> bool {
    save_config_str("lang", tag)
}

/// Shared EN|RU language switch, drawn as a self-contained FOREGROUND `Area` anchored to a window
/// corner. Used by BOTH the start menu and the in-raid viewer so the control is identical and —
/// crucially — can NEVER be clipped by a panel's vertical overflow: it floats independent of any
/// panel content flow. Pure UI: takes the current `Lang` by value and RETURNS the newly-picked lang
/// (so the caller mutates its `ResMut` only on a real change, avoiding per-frame change detection).
/// Anchor RIGHT_BOTTOM in the menu, LEFT_BOTTOM in-raid (clears the right-edge toolbar/layers rail).
pub fn lang_switch_area(
    ctx: &bevy_egui::egui::Context,
    current: crate::i18n::Lang,
    id: &str,
    anchor: bevy_egui::egui::Align2,
    offset: bevy_egui::egui::Vec2,
) -> Option<crate::i18n::Lang> {
    use bevy_egui::egui::{self, Color32, RichText, Stroke};
    use crate::i18n::{t, Lang, K};
    use crate::ui_theme as theme;
    let mut picked = None;
    egui::Area::new(egui::Id::new(id))
        .anchor(anchor, offset)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme::CARD_TRANSLUCENT)
                .stroke(Stroke::new(1.0, theme::BORDER))
                .inner_margin(egui::Margin::symmetric(8, 5))
                .corner_radius(0.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(t(current, K::LangLabel)).size(10.0).color(theme::MUTED));
                        for l in [Lang::En, Lang::Ru] {
                            let active = current == l;
                            let btn = egui::Button::new(
                                RichText::new(l.code())
                                    .size(12.0)
                                    .strong()
                                    .color(if active { theme::BONE } else { theme::MUTED }),
                            )
                            .fill(if active { theme::RAIL } else { Color32::TRANSPARENT })
                            .stroke(Stroke::new(1.0, if active { theme::BEIGE } else { theme::BORDER }))
                            .corner_radius(0.0);
                            if ui.add(btn).on_hover_text(t(current, K::LanguageTip)).clicked() && !active {
                                picked = Some(l);
                            }
                        }
                    });
                });
        });
    picked
}

/// Whether the build python has the extraction packages (UnityPy/numpy/Pillow). Runs a quick import
/// probe under paths::python_exe. None when there's no kit/python to probe (shipped-lite / no build).
pub fn deps_ready() -> Option<bool> {
    let root = crate::paths::repo_root()?;
    let mut cmd = std::process::Command::new(crate::paths::python_exe(root));
    cmd.current_dir(root).arg("-c").arg("import UnityPy, numpy, PIL");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok()?;
    Some(out.status.success())
}

/// The writable tarkmap working dir (`EFT_TARKMAP_ROOT`; holds `out/` bake+intel outputs and,
/// optionally, `maps/` configs — the kit falls back to the shipped `extraction/maps` when absent):
/// env > saved config > a sibling `tarkmap` beside the assets dir (matches the pipeline's expected
/// layout so `assemble_bevy` resolves the SH-volume dir correctly).
pub fn detect_tarkmap_dir() -> String {
    if let Ok(d) = std::env::var("EFT_TARKMAP_ROOT") {
        if !d.is_empty() {
            return d;
        }
    }
    if let Some(d) = config_str("tarkmapRoot").filter(|d| !d.is_empty()) {
        return d;
    }
    let assets = detect_assets_dir();
    std::path::Path::new(&assets)
        .parent()
        .map(|p| p.join("tarkmap"))
        .unwrap_or_else(|| crate::paths::exe_dir().join("tarkmap"))
        .to_string_lossy()
        .into_owned()
}

/// Looks like an EscapeFromTarkov_Data dir? (has level files / globalgamemanagers)
pub fn valid_game_dir(dir: &str) -> bool {
    let p = std::path::Path::new(dir);
    p.join("globalgamemanagers").exists()
        || p.join("level0").exists()
        || std::fs::read_dir(p)
            .map(|rd| {
                rd.flatten()
                    .any(|e| e.file_name().to_string_lossy().starts_with("sharedassets"))
            })
            .unwrap_or(false)
}

/// One registry string value via `reg query` (no winreg dependency). Windows only.
#[cfg(windows)]
fn reg_value(key: &str, value: &str) -> Option<String> {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("reg")
        .args(["query", key, "/v", value])
        .creation_flags(0x0800_0000)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let v = s
        .lines()
        .find(|l| l.trim_start().starts_with(value))?
        .split("REG_SZ")
        .nth(1)?
        .trim()
        .to_string();
    (!v.is_empty()).then_some(v)
}

/// BSG launcher registry entry -> "<InstallLocation>\EscapeFromTarkov_Data". Windows only — there
/// is no Windows registry to query on Linux/macOS.
#[cfg(windows)]
fn registry_game_dir() -> Option<String> {
    let loc = reg_value(
        r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov",
        "InstallLocation",
    )?;
    Some(format!(r"{loc}\EscapeFromTarkov_Data"))
}

/// STEAM install -> ".../steamapps/common/<name>/EscapeFromTarkov_Data". EFT ships on Steam
/// too, and Steam users have no BSG-launcher registry entry. Appid-free route: locate the
/// Steam root (registry on Windows, the default install dirs on Linux), then check EVERY
/// library listed in `steamapps/libraryfolders.vdf` — secondary libraries on other drives are
/// the common case ("D:\SteamLibrary"). Folder-name variants cover Steam's capitalization.
fn steam_game_dir() -> Option<String> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    #[cfg(windows)]
    {
        // Per-user SteamPath (forward slashes) then machine-wide InstallPath, then the default.
        if let Some(p) = reg_value(r"HKCU\Software\Valve\Steam", "SteamPath") {
            roots.push(std::path::PathBuf::from(p.replace('/', r"\")));
        }
        if let Some(p) = reg_value(r"HKLM\SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath") {
            roots.push(std::path::PathBuf::from(p));
        }
        roots.push(std::path::PathBuf::from(r"C:\Program Files (x86)\Steam"));
    }
    #[cfg(not(windows))]
    {
        if let Some(home) = std::env::var_os("HOME") {
            let home = std::path::Path::new(&home);
            roots.push(home.join(".steam/steam"));
            roots.push(home.join(".local/share/Steam"));
        }
    }
    // Steam roots are themselves libraries; libraryfolders.vdf adds the rest.
    let mut libs = roots.clone();
    for r in &roots {
        let Ok(txt) = std::fs::read_to_string(r.join("steamapps").join("libraryfolders.vdf"))
        else {
            continue;
        };
        for line in txt.lines() {
            // VDF rows look like:  "path"    "D:\\SteamLibrary"
            let l = line.trim();
            let Some(rest) = l.strip_prefix("\"path\"") else { continue };
            let p = rest.trim().trim_matches('"').replace("\\\\", "\\");
            if !p.is_empty() {
                libs.push(std::path::PathBuf::from(p));
            }
        }
    }
    for lib in libs {
        for name in ["Escape from Tarkov", "Escape From Tarkov", "EscapeFromTarkov"] {
            let d = lib
                .join("steamapps")
                .join("common")
                .join(name)
                .join("EscapeFromTarkov_Data");
            let s = d.to_string_lossy().into_owned();
            if valid_game_dir(&s) {
                return Some(s);
            }
        }
    }
    None
}

/// Autodetect priority: EFT_GAME_DATA env > saved config > BSG launcher registry > STEAM
/// (registry/default roots + every library in libraryfolders.vdf) > drive probe. The registry
/// lookups and drive-letter probe are Windows-only; on Linux the Steam/Proton default roots
/// and common Wine/Lutris prefixes are probed, and the folder picker remains the fallback.
pub fn detect_game_dir() -> String {
    if let Ok(d) = std::env::var("EFT_GAME_DATA") {
        if !d.is_empty() {
            return d;
        }
    }
    if let Some(d) = config_game_dir().filter(|d| valid_game_dir(d)) {
        return d;
    }
    // Steam is platform-shared: Proton installs on Linux land in the same library layout.
    #[cfg(windows)]
    {
        if let Some(d) = registry_game_dir().filter(|d| valid_game_dir(d)) {
            return d;
        }
        if let Some(d) = steam_game_dir() {
            return d;
        }
        for drive in ["C", "D", "E", "F", "G"] {
            for tail in [
                r"\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data",
                r"\Battlestate Games\EFT\EscapeFromTarkov_Data",
                r"\Games\Escape from Tarkov\EscapeFromTarkov_Data",
                // Registry-less Steam leftovers: default + the common secondary-library name.
                r"\Program Files (x86)\Steam\steamapps\common\Escape from Tarkov\EscapeFromTarkov_Data",
                r"\SteamLibrary\steamapps\common\Escape from Tarkov\EscapeFromTarkov_Data",
            ] {
                let d = format!("{drive}:{tail}");
                if valid_game_dir(&d) {
                    return d;
                }
            }
        }
        r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data".to_string()
    }
    #[cfg(not(windows))]
    {
        if let Some(d) = steam_game_dir() {
            return d;
        }
        if let Some(home) = std::env::var_os("HOME") {
            let home = std::path::Path::new(&home);
            for root in [
                // Steam/Proton prefixes keep the game under the LIBRARY (covered by
                // steam_game_dir above); these are the non-Steam Wine/Lutris guesses.
                home.join(".wine/drive_c"),
                home.join(".local/share/lutris/runners/wine"), // not a prefix root; harmless miss
                home.join("Games/escape-from-tarkov/drive_c"), // common Lutris default prefix
            ] {
                for tail in [
                    "Battlestate Games/Escape from Tarkov/EscapeFromTarkov_Data",
                    "Battlestate Games/EFT/EscapeFromTarkov_Data",
                ] {
                    let d = root.join(tail);
                    let s = d.to_string_lossy().into_owned();
                    if valid_game_dir(&s) {
                        return s;
                    }
                }
            }
        }
        String::new()
    }
}

/// One-time, menu-startup extraction of the real CCTV menu prop (menu_fx 3D decor):
/// if packs/shared/menu/camera.{bin,png} are missing and the game install resolves, run
/// tools/extract_menu_prop.py SYNCHRONOUSLY (dataset-first, UnityPy game-file fallback)
/// before the menu scans packs. Bounded wait (~30 s): on timeout the child is left to
/// finish on its own (files then exist for the NEXT launch) and this launch just uses the
/// vector camera. Any failure falls through silently to the vector camera — the menu
/// always draws. Runs before Bevy's first frame, so a short block here is invisible.
#[allow(dead_code)] // retired CCTV-prop extractor; kept for reference (backdrop is 2D now)
fn ensure_menu_prop(game_dir: &str) {
    let menu_dir = crate::paths::shared_dir().join("menu");
    if menu_dir.join("camera.bin").is_file() && menu_dir.join("camera.png").is_file() {
        return; // already extracted on this machine — never re-run
    }
    if !valid_game_dir(game_dir) {
        eprintln!("menu prop: game install not found - keeping the vector camera");
        return;
    }
    let Some(root) = crate::paths::repo_root() else {
        return; // no python kit beside the exe/cwd — shipped-lite bundle, vector camera
    };
    let script = root.join("tools").join("extract_menu_prop.py");
    if !script.is_file() {
        return;
    }
    eprintln!("menu prop: extracting the real CCTV prop (one-time, local-only)...");
    let mut cmd = std::process::Command::new(crate::paths::python_exe(root));
    cmd.current_dir(root)
        .env("EFT_GAME_DATA", game_dir)
        .arg("tools/extract_menu_prop.py")
        .arg("--out")
        .arg(&menu_dir)
        .stdout(std::process::Stdio::inherit()) // its ASCII [menu-prop] lines go to our console
        .stderr(std::process::Stdio::inherit());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let child = cmd.spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("menu prop: could not start extractor: {e}");
            return;
        }
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!(
                    "menu prop: extractor finished ({})",
                    if status.success() { "ok" } else { "failed - vector camera" }
                );
                return;
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    // Never kill it: let it finish in the background so the files are
                    // there next launch; this launch simply keeps the vector camera.
                    eprintln!("menu prop: extractor still running after 30s - vector camera this launch");
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            Err(e) => {
                eprintln!("menu prop: extractor wait failed: {e}");
                return;
            }
        }
    }
}

pub fn build_state() -> MenuState {
    let game_dir = detect_game_dir();
    // (The old real-CCTV menu prop is retired — the backdrop is the 2D triangle field / opt-in 3D
    // scenes now — so we no longer block first-frame on a synchronous extract_menu_prop run.)
    let game_fp = game_fingerprint(&game_dir);
    let (entries, total_bytes) = scan(&game_fp);
    // EFT_MENU_BUILD=<map>[,--dry-run] auto-starts a build on menu open (CLI/testing hook);
    // enqueued on the shared worker on the first frame (build_state has no worker access here).
    let autobuild = std::env::var("EFT_MENU_BUILD").ok().filter(|s| !s.is_empty());
    let assets_dir = detect_assets_dir();
    MenuState {
        entries,
        game_fp,
        game_dir_edit: game_dir.clone(),
        game_dir,
        assets_dir_edit: assets_dir.clone(),
        assets_dir,
        assets_ok: assets_configured(),
        deps_ok: deps_ready().unwrap_or(true),
        build_kit_available: crate::paths::repo_root().is_some(),
        total_bytes,
        confirm_delete: None,
        show_rebuild: None,
        show_log: false,
        show_sync_log: false,
        intel: intel_status(),
        sync_note: None,
        seen_completed: 0,
        autobuild,
        config_err: None,
        process_in_background: config_process_in_background(),
        force_cpu_process: config_force_cpu_process(),
        screenshot_locate: config_screenshot_locate(),
        overlay: crate::overlay::OverlayConfig::load().sanitized(),
        // A named preset owns texture quality. Derive the visible texture button from it too, so
        // even a stale/hand-edited file cannot show (for example) Ultra + Quarter for one frame.
        tex_quality: {
            let preset = crate::render::QualityPreset::from_index(
                (config_f32_pub("qualityPreset").unwrap_or(2.0) as u8).min(4),
            );
            preset
                .tex_quality()
                .unwrap_or_else(|| {
                    // Clamp: an out-of-range hand edit must not leave every button unselected.
                    (config_f32_pub("textureQuality").unwrap_or(1.0) as u8).min(2)
                })
        },
        quality_preset: (config_f32_pub("qualityPreset").unwrap_or(2.0) as u8).min(4), // High
        // EFT_SETTINGS_TAB=<0|1|2> opens a settings tab on launch (0 Overlay, 1 Live link,
        // 2 General) — screenshots / QA of the menu without hand-driving the UI.
        settings_tab: std::env::var("EFT_SETTINGS_TAB")
            .ok()
            .and_then(|s| s.trim().parse::<u8>().ok())
            .filter(|t| *t <= 2),
        reattached: false,
    }
}

fn fmt_size(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0} MB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        "-".to_string()
    }
}

fn fmt_age(days: Option<f64>) -> String {
    match days {
        Some(d) if d < 1.0 => "today".to_string(),
        Some(d) => format!("{:.0}d ago", d),
        None => "-".to_string(),
    }
}

/// Weighted build progress in [0,1]. The extraction dominates wall-clock (~55% of a build), so weight
/// phases by their typical relative duration (codex perf ranking) instead of "stage i of N" — that
/// equal-stage assumption is what pinned the ESTIMATED TIME at 99:59 (extraction counted as 1/9).
/// `stage` is the latest `[STAGE i/N]` / `[BUILD OK]` marker; `sub` is an optional in-stage fraction
/// (0..1) from a `[SUBPROGRESS]` marker, so the bar moves DURING the long extraction.
///
/// In-stage fraction from a `[SUBPROGRESS] ...` line: the LAST whitespace token is `<done>/<total>`,
/// unit-agnostic — the emitters put the human-readable level count first and the machine-read
/// byte ratio last, and older logs' plain `extract <d>/<t>` parses identically. `None` for
/// non-SUBPROGRESS lines and degenerate totals.
fn subprogress_frac(line: &str) -> Option<f32> {
    let rest = line.split("[SUBPROGRESS]").nth(1)?;
    let tok = rest.split_whitespace().last()?; // "<d>/<t>"
    let (d, t) = tok.split_once('/')?;
    let (d, t) = (d.trim().parse::<f32>().ok()?, t.trim().parse::<f32>().ok()?);
    (t > 0.0).then(|| (d / t).clamp(0.0, 1.0))
}

/// `fresh` = this build is running the ONE-TIME full extraction (the `fresh_extract` latch). The
/// standard 55% extraction weight was ranked on builds whose dataset already exists; on a fresh
/// build stage 1 alone is ~64% of wall-clock and the SH bake another ~19%, so the same table
/// made `elapsed/frac` overestimate the whole way through (the naive ETA then saturated its old
/// mm:ss ceiling at 99:59). The fresh table is MEASURED — `[TIMING]` lines of a real streets
/// first build (2026-08-02, 6C/12T, total 2h26m): stage 1 5578s (dataset 2899 / grass 6 /
/// colliders 2673), lights 114s, assemble 724s, SH bake 1625s, grass 11s, zones 196s, icons
/// 129s, nav 387s, stamp 5s. One sample, one map, one machine — but a measurement, not a guess.
///
/// Weights are per-stage WINDOWS rather than a cumulative array because the portable pipeline
/// runs stage 3 (SH bake) AFTER stage 4 (assemble): wall-clock order is 1,2,4,3,5..9, which a
/// cumulative-by-stage-number table cannot express — it froze the bar (and pinned the ETA)
/// through the whole 27-minute bake while the `max_frac` guard absorbed the regression.
fn build_frac(stage: &str, sub: Option<f32>, fresh: bool) -> f32 {
    if stage.starts_with("[BUILD OK]") {
        return 1.0;
    }
    let parsed = stage
        .strip_prefix("[STAGE ")
        .and_then(|s| s.split(']').next())
        .and_then(|s| {
            let (i, n) = s.split_once('/')?;
            Some((i.trim().parse::<f32>().ok()?, n.trim().parse::<f32>().ok()?))
        });
    let Some((i, n)) = parsed else {
        return 0.0;
    };
    let done = stage.contains(": done") || stage.contains(": skipped");
    let base = if done { 1.0 } else { sub.unwrap_or(0.5).clamp(0.0, 1.0) };
    // Stage 1 of a map build is THREE serial passes over the game files (dataset -> grass ->
    // colliders), each re-emitting "[STAGE 1/N]" with its own name. Treating them as one pool
    // meant the bar finished stage 1's span with the dataset pass and then sat frozen through
    // grass + colliders (on streets, up to an hour of "28%" / "55%" with a live build behind
    // it). Give each pass its slice of the stage-1 span; the `sub` signal moves the bar WITHIN
    // the slice. Splits are duration estimates on a fresh streets build (the dominant case —
    // these sub-stages only run on a fresh extraction). Unmatched stage-1 names (check dataset,
    // nav agent settings) keep the whole-span behavior; `max_frac` already absorbs any regress.
    let in_frac = if (n - 9.0).abs() < 0.5 && (i - 1.0).abs() < 0.5 {
        // Measured sub-pass shares of stage 1 (streets: 2899s / 6s / 2673s). Colliders is
        // nearly HALF the stage — it runs serial while the dataset extract runs one process
        // per core, which is also why parallelizing it is the biggest first-build win left.
        // A pass that EMITS [SUBPROGRESS] starts at the bottom of its slice, not the middle:
        // `base`'s 0.5 fallback is a guess made before the first reading arrives, and on the
        // colliders pass that guess (48.65%) lands just ABOVE the real first reading (48.63%),
        // so `max_frac` latches it and the pass's own signal is clamped away for its whole run.
        // The grass pass emits nothing, so it keeps the midpoint guess inside its 1% slice.
        let signalled = if done { 1.0 } else { sub.unwrap_or(0.0).clamp(0.0, 1.0) };
        if stage.contains("extract dataset") {
            0.52 * signalled
        } else if stage.contains("grass density") {
            0.52 + 0.01 * base
        } else if stage.contains("physics colliders") {
            0.53 + 0.47 * signalled
        } else {
            // Unmatched stage-1 names (check dataset, nav agent settings) are BOOKENDS, not a
            // pass with a duration -- they must sit at the START of the span, not the middle of
            // it. With `base` here the first marker of a fresh build ("check dataset", no
            // [SUBPROGRESS] yet, so base falls to the 0.5 default) claimed 0.5 * 0.636 = 31.8%,
            // and `max_frac` latched it: the dataset pass's own readings (3.3%, 17.8%, 32.4%)
            // were then all discarded, so the bar did not move until the extract was ~96% done.
            // On streets that is ~45 min parked at 31.8% at the very start of a first build --
            // the same "reads as a hang" this table exists to remove. `done` still ends the span.
            if done { 1.0 } else { 0.0 }
        }
    } else {
        base
    };
    // Per-stage (start, end) windows of the overall bar, indexed by stage NUMBER (1-based).
    let windows: &[(f32, f32)] = if (n - 9.0).abs() < 0.5 && fresh {
        // FIRST build, measured (see the doc comment). Stage 3's window sits AFTER stage 4's
        // because that is the wall-clock order the portable pipeline actually runs them in.
        &[
            (0.000, 0.636), // 1 extract (dataset + grass + colliders)
            (0.636, 0.649), // 2 lights + interactables + LUT
            (0.732, 0.917), // 3 SH bake — runs post-assemble
            (0.649, 0.732), // 4 assemble
            (0.917, 0.918), // 5 grass pack
            (0.918, 0.941), // 6 zones
            (0.941, 0.955), // 7 icons
            (0.955, 0.999), // 8 nav
            (0.999, 1.000), // 9 stamp
        ]
    } else if (n - 9.0).abs() < 0.5 {
        // REBUILD (dataset already extracted) — the case every build after the first one takes.
        // The table this replaces was ranked for a FRESH build and gave stage 1 the 0..0.55 span
        // because extraction dominates one. On a rebuild extraction does not run at all: stage 1
        // is `check dataset`, about a second, so the bar leapt to 55% immediately and, with the
        // stage-2/3/4 windows on top, read 84% after 57s of a 31-minute build. The ETA
        // (elapsed/frac) said ~68s with half an hour to go, and only became truthful in the last
        // four minutes.
        //
        // MEASURED on a real streets rebuild (2026-08-02, total 1878s): check 1s, lights+intel
        // 56s, assemble 1205s, SH bake 289s, grass 2s, zones 75s, icons 1s, nav 247s, stamp 2s.
        // Windows are cumulative in WALL-CLOCK order (1,2,4,3,5..9) for the same reason the fresh
        // table is: the portable pipeline bakes AFTER assembling, and a table ordered by stage
        // NUMBER put the bake's window behind the bar, freezing it at 88% for the whole 289s bake
        // while `max_frac` hid the contradiction. Same defect the fresh table fixed, left behind
        // on the path that runs far more often.
        &[
            (0.000, 0.001), // 1 check dataset (no extraction on a rebuild)
            (0.001, 0.030), // 2 lights + interactables + LUT
            (0.672, 0.826), // 3 SH bake — runs post-assemble
            (0.030, 0.672), // 4 assemble
            (0.826, 0.827), // 5 grass
            (0.827, 0.867), // 6 zones
            (0.867, 0.868), // 7 icons
            (0.868, 0.999), // 8 nav
            (0.999, 1.000), // 9 stamp
        ]
    } else if (n - 3.0).abs() < 0.5 {
        &[(0.00, 0.05), (0.05, 0.98), (0.98, 1.00)] // deps install: venv, pip, verify
    } else {
        return ((i - 1.0 + in_frac) / n).clamp(0.0, 1.0); // unknown pipeline: linear
    };
    let (lo, hi) = windows[(i as usize).clamp(1, windows.len()) - 1];
    (lo + (hi - lo) * in_frac).clamp(0.0, 1.0)
}

/// Localized age ("today" / "3 d ago") for the map-row cards.
fn fmt_age_lg(lg: crate::i18n::Lang, days: Option<f64>) -> String {
    use crate::i18n::{t, K};
    match days {
        Some(d) if d < 1.0 => t(lg, K::Today).to_string(),
        Some(d) => format!("{:.0} {}", d, t(lg, K::DAgo)),
        None => "-".to_string(),
    }
}

/// A per-frame snapshot of the build to show in the menu's loader panel — either the in-flight
/// build or the most recent finished one (which lingers until CLOSE). Owned plain data so the
/// egui closures don't hold a borrow on the worker.
#[cfg(feature = "egui")]
struct BuildView {
    key: String,
    stage: String,
    tail: Vec<String>,
    started_secs: f32,
    finished: bool,
    ok: bool,
    full_log: String,
    /// Monotonic weighted progress in [0,1] (see `build_view` — never regresses).
    frac: f32,
}

#[cfg(feature = "egui")]
fn build_view(w: &crate::jobs::JobWorker) -> Option<BuildView> {
    // The running build/deps-install takes precedence; otherwise a just-finished one lingers until
    // dismissed. Deps installs stream the same [STAGE]/[BUILD OK] markers, so they reuse the panel.
    let job = if w.current_build_key().is_some() || w.current_is_install() {
        w.current_job()
    } else if w.last_is_build() || w.last_is_install() {
        w.last_job()
    } else {
        None
    }?;
    let (stage, tail, finished, ok) = job.snapshot(12);
    // Auto-dismiss a successfully finished MAP build: its row flips to installed via the completion
    // rescan (see the `worker.completed` check in `render`), so the loader panel should DISAPPEAR
    // rather than linger at 100% — the user wants the bar visible only WHILE processing. Deps-install
    // success stays (there is no map row to appear, so its "done" line is the only feedback); ALL
    // failures stay visible (error + CLOSE) so nothing is silently hidden.
    if finished && ok && job.key != "__deps__" {
        return None;
    }
    // Weighted progress, clamped MONOTONIC on the persistent BuildJob so nested sub-script `[STAGE i/M]`
    // markers can't drop the bar. The sub-fraction moves the bar WITHIN the long extraction stage via
    // the parallel extractor's `[SUBPROGRESS] <done>/<total>` marker.
    let sub = if stage.starts_with("[STAGE 1/") {
        // Take the newest [SUBPROGRESS] FROM THE PASS THAT IS RUNNING. Stage 1 is three serial
        // passes with three independent denominators, so an unfiltered search hands the colliders
        // marker the dataset pass's trailing ~1.0: 0.53 + 0.47*0.98 = 0.99 of the span, which
        // `max_frac` latches at 63%. Every later colliders reading is lower and gets clamped away,
        // freezing the bar for the whole (~45 min) pass. The emitters already tag their lines, so
        // match on the tag; anything untagged (older logs' plain `extract <d>/<t>`) still parses.
        let tag = if stage.contains("physics colliders") { "colliders" } else { "extract" };
        tail.iter()
            .rev()
            .filter(|l| l.contains(tag))
            .find_map(|l| subprogress_frac(l))
    } else {
        None
    };
    // Latch the fresh-extraction flag off the stage marker itself: "extract dataset" is a stage
    // name only the ONE-TIME full extraction path emits (a cached dataset skips straight through
    // "check dataset"). Sticky on the job so the first-build weight table keeps applying after
    // stage 1 hands off to the tail stages.
    let fresh = {
        use std::sync::atomic::Ordering::Relaxed;
        if !job.fresh_extract.load(Relaxed)
            && stage.starts_with("[STAGE 1/")
            && stage.contains("extract dataset")
        {
            job.fresh_extract.store(true, Relaxed);
        }
        job.fresh_extract.load(Relaxed)
    };
    let raw = if finished && ok { 1.0 } else { build_frac(&stage, sub, fresh) };
    let frac = {
        use std::sync::atomic::Ordering::Relaxed;
        let prev = f32::from_bits(job.max_frac.load(Relaxed));
        let v = raw.max(prev).clamp(0.0, 1.0);
        job.max_frac.store(v.to_bits(), Relaxed);
        v
    };
    Some(BuildView {
        key: job.key.clone(),
        stage,
        tail,
        started_secs: job.started.elapsed().as_secs_f32(),
        finished,
        ok,
        full_log: job.full_log(),
        frac,
    })
}

/// Fullscreen menu UI (EguiPrimaryContextPass). Only registered/active in menu mode.
#[cfg(feature = "egui")]
pub fn menu_ui(
    mut contexts: bevy_egui::EguiContexts,
    state: Option<ResMut<MenuState>>,
    mut switch: ResMut<crate::MapSwitch>,
    // The shared background worker: menu build/sync buttons enqueue onto the SAME queue the
    // in-raid MAP PROCESSING panel uses (one worker, one code path).
    mut worker: ResMut<crate::jobs::JobWorker>,
    // Present only when the real-asset 3D CCTV spawned (menu_fx::spawn_menu_prop): flips
    // the CentralPanel transparent so the 3D world shows through, and suppresses the
    // vector-drawn camera (exactly one of the two decors ever renders).
    prop3d: Option<Res<crate::menu_fx::MenuCamProp>>,
    // UI language (EN/RU); the footer toggle flips + persists it.
    mut lang: ResMut<crate::i18n::Lang>,
    // GitHub update check: the status drives the top-right version indicator + the update modal;
    // `update_check.dismissed` is the LATER-this-session flag.
    update_status: Res<crate::update::UpdateStatus>,
    mut update_check: ResMut<crate::update::UpdateCheck>,
    // The LIVE overlay settings. The menu edits `MenuState.overlay` and persists it, but the
    // running app reads this RESOURCE -- without pushing the change here, ticking the box only
    // took effect on the next launch (the toggle looked broken).
    mut overlay_cfg: ResMut<crate::overlay::OverlayConfig>,
    // Settings writes waiting for the pointer to be released: sliders fire `.changed()` every
    // frame of a drag, and each save() is 12 read-parse-rewrite cycles of atlas.config.json.
    // Apply lives immediately; persist once, when the drag ends.
    mut settings_save_pending: Local<bool>,
) {
    use bevy_egui::egui::{self, Color32, RichText};
    use crate::i18n::{map_title, t, K};
    use crate::jobs::Job;
    let Some(mut state) = state else { return };
    let real_prop = prop3d.is_some();
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let lg = *lang; // Copy for reads; the toggle writes *lang

    // First-frame startup scan: reattach to (or surface the result of) a detached build a PREVIOUS
    // Atlas run left going when it was closed. Runs before the completion-rescan below so a build
    // that finished-while-closed re-scans into an installed row, and a still-running one lights up
    // the live panel + BUILDING row this same frame.
    if !state.reattached {
        state.reattached = true;
        reattach_builds(&mut worker, &state.game_dir);
    }

    // First-frame EFT_MENU_BUILD auto-build (enqueue once onto the shared worker).
    if let Some(map) = state.autobuild.take() {
        worker.enqueue(Job::BuildMap {
            map,
            game_dir: state.game_dir.clone(),
            force: false,
            background: state.process_in_background,
        });
    }

    // A job finished on the worker since we last looked: rescan the pack list + intel (a build
    // may have produced/updated a pack), and reflect a finished SYNC / failed BUILD in the UI.
    if worker.completed != state.seen_completed {
        state.seen_completed = worker.completed;
        let (entries, total) = scan(&state.game_fp);
        state.entries = entries;
        state.total_bytes = total;
        state.intel = intel_status();
        // A deps install may have just finished — re-probe so the status flips to ready without a
        // restart (python_exe now resolves to the freshly-created venv).
        state.deps_ok = deps_ready().unwrap_or(true);
        let ok = worker.last_outcome().map(|(_, ok)| ok).unwrap_or(false);
        if worker.last_is_sync() {
            // "refreshed" only if the sync exited 0 AND fresh intel is actually VISIBLE where we scan
            // (state.intel was just re-read above). A green "refreshed" next to "synced never" means
            // the script wrote loot.json somewhere shared_dir() can't see — report that instead of
            // lying. (After the packs_root content-check fix both point at the same dir.)
            let intel_present = state.intel.0.is_some() || state.intel.1.is_some();
            let refreshed = ok && intel_present;
            state.sync_note = Some(if refreshed {
                (t(lg, K::IntelRefreshed).to_string(), true)
            } else {
                (t(lg, K::SyncFailed).to_string(), false)
            });
            if !refreshed {
                // Surface the failing stage (or the exit-0-but-not-visible packs_root divergence)
                // without a click — same reflex as a failed build (see below).
                state.show_sync_log = true;
            }
        } else if worker.last_is_build() && !ok {
            state.show_log = true; // surface the failing stage without a click
        } else if worker.last_is_build() && ok {
            // A pack's loot values + quest layer come from the tarkov.dev intel sync, which the
            // build pipeline itself never runs. On a fresh machine the first successful build has
            // no shared intel sidecar yet, so the map would load with empty loot/quest layers until
            // the user happened to click "sync intel". Kick off a one-time sync automatically so
            // those layers populate. Once loot.json exists this won't fire again; a completing sync
            // re-enters this handler as last_is_sync (above), so there's no loop.
            if !crate::paths::shared_dir().join("loot.json").exists() {
                worker.enqueue(Job::SyncIntel);
            }
        }
    }

    // The menu animates without input now (camera LED blink / servo slew / idle patrol and
    // the loading-bar pulse): keep frames coming even when no events arrive, at a faster
    // cadence while a pipeline build is streaming.
    ctx.request_repaint_after(std::time::Duration::from_millis(if worker.busy() {
        50
    } else {
        80
    }));

    // EFT gear-screen scheme, all tokens from the single source of truth (ui_theme). Local aliases
    // keep the menu body readable; no drifted literal values live here anymore.
    use crate::ui_theme as theme;
    const BG: Color32 = theme::BG; // near-black field (authentic EFT loading-screen 10,10,10)
    const HEADER: Color32 = theme::RAIL; // header bar + build-panel fill
    const CARD: Color32 = theme::CARD; // map-row card (raised over BG, warm charcoal)
    const BORDER: Color32 = theme::BORDER; // 1px warm steel
    const BONE: Color32 = theme::BONE;
    const BEIGE: Color32 = theme::BEIGE;
    const DIM: Color32 = theme::MUTED;
    const OK: Color32 = theme::OK;
    const WARN: Color32 = theme::WARN;
    const BAD: Color32 = theme::DANGER;

    // Theme egui's own defaults once (square corners, spacing, widget fills + text, selection).
    // Modifies the existing dark style in place — never replaces it (that flips egui to its light
    // theme and paints a fullscreen pale layer, the historical bug).
    theme::apply_global_style(ctx);

    // Backdrop: the 2D reactive triangle field (default). Skipped when an env flag selects one of the
    // 3D backdrops instead (EFT_MENU_EXFIL / EFT_MENU_TERRAIN), so they don't double up.
    let use_3d_bg = std::env::var("EFT_MENU_EXFIL").map(|v| v.trim() == "1").unwrap_or(false)
        || std::env::var("EFT_MENU_TERRAIN").map(|v| v.trim() == "1").unwrap_or(false);
    if !use_3d_bg {
        crate::menu_fx::triangle_field(ctx);
    }

    // Worker intents: collected inside the egui closures (which only READ the worker) and applied
    // once after, so the ResMut is never aliased mid-closure. `enqueue_build` is the map key to
    // build; `cancel_current` cancels whatever job is in flight (build or sync).
    let mut enqueue_sync = false;
    // (map key, force): force = the UPDATE path (build_map.py --force re-extracts stale data).
    let mut enqueue_build: Option<(String, bool)> = None;
    let mut enqueue_install = false;
    let mut cancel_current = false;
    let mut dismiss_build = false;
    // Per-frame snapshots so the closures don't hold a borrow on the worker.
    let wk_build_key = worker.current_build_key().map(|s| s.to_string());
    let bv = build_view(&worker);

    egui::TopBottomPanel::top("menu_header")
        .frame(egui::Frame::new().fill(HEADER).inner_margin(egui::Margin::symmetric(24, 10)))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("ATLAS").color(BONE).size(theme::SIZE_DISPLAY).strong());
                ui.add_space(14.0);
                ui.label(RichText::new(format!("|  {}", t(lg, K::Map))).color(BEIGE).size(13.0));
                ui.label(RichText::new(t(lg, K::SelectLocation)).color(DIM).size(13.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // SETTINGS opens the tabbed panel beside the map list (overlay / live link /
                    // general). Toggles, so the same button closes it.
                    if ui
                        .selectable_label(
                            state.settings_tab.is_some(),
                            RichText::new(t(lg, K::Settings)).color(BEIGE).size(12.0),
                        )
                        .on_hover_text("Overlay, live game link and general options")
                        .clicked()
                    {
                        state.settings_tab = if state.settings_tab.is_some() { None } else { Some(0) };
                    }
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new(fmt_size(state.total_bytes)).color(BEIGE).size(13.0),
                    );
                    ui.label(RichText::new(t(lg, K::PacksOnDisk)).color(DIM).size(11.0));
                });
            });

            // ---- INTEL strip: tarkov.dev data freshness + one-click refresh. Overlays (loot
            // values, tasks, icons) are only as good as their last sync — surface it. ----
            ui.add_space(4.0);
            // Whether a sync has run this session (current or last job) — gates the inline log
            // toggle below. The sync writes no <packs>/logs file, so this is its only log surface.
            let sync_has_log = worker.current_is_sync() || worker.last_is_sync();
            ui.horizontal(|ui| {
                ui.label(RichText::new(t(lg, K::Intel)).color(BEIGE).size(11.0).strong());
                let (loot_d, tasks_d, icons) = state.intel;
                let ru = lg == crate::i18n::Lang::Ru;
                let age_txt = |d: Option<f64>| match d {
                    Some(d) if d < 1.0 => {
                        format!("{:.0} {}", (d * 24.0).max(1.0), if ru { "ч назад" } else { "h ago" })
                    }
                    Some(d) => format!("{d:.0} {}", if ru { "д назад" } else { "d ago" }),
                    None => t(lg, K::Never).to_string(),
                };
                // Stale warning: prices move per wipe/week — amber past 7 days, red past 21.
                let worst = loot_d.unwrap_or(f64::MAX).max(tasks_d.unwrap_or(f64::MAX));
                let age_col = if worst > 21.0 {
                    theme::DANGER_TEXT
                } else if worst > 7.0 {
                    theme::WARN
                } else {
                    theme::OK
                };
                ui.label(
                    RichText::new(format!(
                        "{} {}  \u{00B7}  {} {}  \u{00B7}  {} {}",
                        t(lg, K::Synced),
                        age_txt(loot_d),
                        t(lg, K::TasksLabel),
                        age_txt(tasks_d),
                        icons,
                        t(lg, K::Icons),
                    ))
                    .color(age_col)
                    .size(11.0),
                );
                // Sync job lifecycle on the shared worker: idle button -> streaming stage ->
                // outcome note (the note is set by the completion handler at the top of menu_ui).
                if worker.current_is_sync() {
                    let stage = worker.status().map(|(_, s)| s).unwrap_or_default();
                    ui.label(
                        RichText::new(format!("{}  {stage}", t(lg, K::Syncing)))
                            .color(theme::ACCENT)
                            .size(11.0),
                    );
                    if ui.small_button(RichText::new(t(lg, K::CancelLower)).size(10.0)).clicked() {
                        cancel_current = true;
                    }
                } else {
                    if ui
                        .add(egui::Button::new(RichText::new(t(lg, K::SyncNow)).size(11.0).color(BEIGE)))
                        .on_hover_text(t(lg, K::SyncTip))
                        .clicked()
                    {
                        state.sync_note = None;
                        enqueue_sync = true;
                    }
                    if let Some((note, ok)) = &state.sync_note {
                        ui.label(
                            RichText::new(note.as_str())
                                .color(if *ok { theme::OK } else { theme::DANGER_TEXT })
                                .size(11.0),
                        );
                    } else if let Some(err) = &worker.spawn_error {
                        ui.label(RichText::new(err.as_str()).color(theme::DANGER_TEXT).size(11.0));
                    }
                }
                // Show/Copy the sync's streamed log. build_view() skips sync jobs so the build panel
                // never carries these buttons for a sync, and no <packs>/logs file is written — this
                // inline toggle is the only way to read/share a failed sync's [SYNC FAILED] stage.
                if sync_has_log {
                    if ui
                        .small_button(RichText::new(
                            t(lg, if state.show_sync_log { K::HideLog } else { K::ShowLog }),
                        ).size(10.0))
                        .clicked()
                    {
                        state.show_sync_log = !state.show_sync_log;
                    }
                    if ui.small_button(RichText::new(t(lg, K::CopyLog)).size(10.0)).clicked() {
                        let job = if worker.current_is_sync() { worker.current_job() } else { worker.last_job() };
                        if let Some(job) = job {
                            ui.ctx().copy_text(job.full_log());
                        }
                    }
                }
            });
            // Sync log tail, inline under the INTEL row (capped ScrollArea like the build panel so a
            // long log scrolls INSIDE instead of shoving the map list up). Auto-opens on failure.
            if state.show_sync_log && sync_has_log {
                let job = if worker.current_is_sync() { worker.current_job() } else { worker.last_job() };
                if let Some(job) = job {
                    let (_, tail, _, _) = job.snapshot(200);
                    ui.add_space(4.0);
                    egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                        for line in &tail {
                            ui.label(RichText::new(line).color(DIM).size(11.0).monospace());
                        }
                    });
                }
            }
        });

    // With the real 3D prop active the CentralPanel goes TRANSPARENT: the 3D world behind
    // (menu-mode ClearColor is the same #090909, set in main.rs) becomes the field and the
    // prop shows through the right gutter. Header/cards keep their own opaque fills, so the
    // list looks identical either way.
    // ---- Footer + live build panel, PINNED as bottom panels (never clipped on a short window) ----
    // egui reserves each bottom panel's height and hands the CentralPanel only what's left, so these
    // controls stay reachable at ANY window height while the map LIST scrolls in the middle. Declared
    // BEFORE the CentralPanel (egui requirement), footer FIRST so it sits at the very bottom and the
    // build panel stacks just above it. Transparent frames keep the 3D globe showing behind them.
    egui::TopBottomPanel::bottom("menu_footer")
        .frame(egui::Frame::new().fill(Color32::TRANSPARENT).inner_margin(egui::Margin::symmetric(24, 8)))
        .show(ctx, |ui| {
            ui.add_space(8.0);
            ui.separator();
            // "Process in background" (default ON): keep a MAP build running even if Atlas is closed
            // (it detaches + streams to a log file; a later launch reattaches). Persisted immediately.
            ui.horizontal(|ui| {
                let mut bg = state.process_in_background;
                if ui
                    .checkbox(&mut bg, RichText::new(t(lg, K::ProcessInBackground)).color(BONE).size(11.0))
                    .on_hover_text(t(lg, K::ProcessInBackgroundTip))
                    .changed()
                {
                    state.process_in_background = bg;
                    state.config_err = (!save_config_process_in_background(bg))
                        .then(|| t(lg, K::ConfigSaveFailed).to_string());
                }
            });
            // "Screenshot to locate current position" (default ON): poll the EFT screenshot folder
            // and put the camera on each new fix. The \u{2139} info marker carries the one thing users trip over —
            // the in-game screenshot key being swallowed by the Windows Snipping Tool.
            ui.horizontal(|ui| {
                let mut loc = state.screenshot_locate;
                if ui
                    .checkbox(
                        &mut loc,
                        RichText::new(t(lg, K::ScreenshotLocate)).color(BONE).size(11.0),
                    )
                    .on_hover_text(t(lg, K::ScreenshotLocateTip))
                    .changed()
                {
                    state.screenshot_locate = loc;
                    crate::game_watch::set_screenshot_locate(loc);
                    state.config_err = (!save_config_screenshot_locate(loc))
                        .then(|| t(lg, K::ConfigSaveFailed).to_string());
                }
                ui.label(RichText::new("\u{2139}").color(DIM).size(12.0))
                    .on_hover_text(t(lg, K::ScreenshotLocateTip));
            });
            // Sub-option: house-keeping for the screenshots the locator consumes. Indented under
            // its parent and greyed out with it — it only means anything while the folder is
            // being polled. (Same control as Settings -> Live link; both write OverlayConfig.)
            ui.horizontal(|ui| {
                ui.add_space(22.0);
                ui.add_enabled_ui(state.screenshot_locate, |ui| {
                    let mut del = state.overlay.delete_processed_shots;
                    if ui
                        .checkbox(
                            &mut del,
                            RichText::new(t(lg, K::DeleteShots)).color(BONE).size(11.0),
                        )
                        .on_hover_text(t(lg, K::DeleteShotsTip))
                        .changed()
                    {
                        state.overlay.delete_processed_shots = del;
                        *overlay_cfg = state.overlay.clone(); // live, no relaunch needed
                        crate::game_watch::set_delete_processed_shots(del);
                        state.config_err = (!state.overlay.save())
                            .then(|| t(lg, K::ConfigSaveFailed).to_string());
                    }
                });
            });
            // Overlay mode (opt-in, default OFF). The tooltip carries the two things that decide
            // whether it works at all: EFT must be BORDERLESS, and this is the user's call to make.
            // Only the master switch lives here for now; the geometry/throttle fields of
            // `OverlayConfig` are ready for a proper settings panel.
            ui.horizontal(|ui| {
                let mut ov = state.overlay.enabled;
                if ui
                    .checkbox(&mut ov, RichText::new(t(lg, K::OverlayEnable)).color(BONE).size(11.0))
                    .on_hover_text(t(lg, K::OverlayEnableTip))
                    .changed()
                {
                    state.overlay.enabled = ov;
                    *overlay_cfg = state.overlay.clone(); // live, no relaunch needed
                    state.config_err = (!state.overlay.save())
                        .then(|| t(lg, K::ConfigSaveFailed).to_string());
                }
                ui.label(RichText::new("\u{2139}").color(DIM).size(12.0))
                    .on_hover_text(t(lg, K::OverlayEnableTip));
            });
            // Build dependencies: the pipeline needs UnityPy/numpy/Pillow. INSTALL DEPS sets them up
            // (venv + pip) from here without closing the app; progress streams into the panel above.
            ui.horizontal(|ui| {
                ui.label(RichText::new(t(lg, K::BuildDeps)).color(DIM).size(11.0));
                if worker.current_is_install() {
                    let stage = worker.status().map(|(_, s)| s).unwrap_or_default();
                    ui.label(
                        RichText::new(format!("{}  {stage}", t(lg, K::Installing)))
                            .color(theme::ACCENT)
                            .size(11.0),
                    );
                } else if state.deps_ok {
                    ui.label(RichText::new(t(lg, K::DepsReady)).color(OK).size(11.0));
                } else {
                    ui.label(RichText::new(t(lg, K::DepsMissing)).color(WARN).size(11.0));
                    if ui
                        .add_enabled(
                            !worker.busy(),
                            egui::Button::new(RichText::new(t(lg, K::InstallDeps)).color(BONE)),
                        )
                        .on_hover_text(t(lg, K::InstallDepsTip))
                        .clicked()
                    {
                        enqueue_install = true;
                    }
                }
            });
            // Game install path: autodetected (env > saved > registry > probe), editable here;
            // SET validates, persists to atlas.config.json and re-fingerprints the packs.
            ui.horizontal(|ui| {
                ui.label(RichText::new(t(lg, K::GameInstall)).color(DIM).size(11.0));
                let mut edit = state.game_dir_edit.clone();
                ui.add(
                    egui::TextEdit::singleline(&mut edit)
                        .desired_width(520.0)
                        .font(egui::TextStyle::Monospace),
                );
                state.game_dir_edit = edit;
                let dirty = state.game_dir_edit != state.game_dir;
                if ui
                    .add_enabled(dirty, egui::Button::new(RichText::new(t(lg, K::Set)).color(BONE)))
                    .clicked()
                {
                    if valid_game_dir(&state.game_dir_edit) {
                        state.game_dir = state.game_dir_edit.clone();
                        state.config_err = (!save_config_game_dir(&state.game_dir))
                            .then(|| t(lg, K::ConfigSaveFailed).to_string());
                        state.game_fp = game_fingerprint(&state.game_dir);
                        let (entries, total) = scan(&state.game_fp);
                        state.entries = entries;
                        state.total_bytes = total;
                    } else {
                        // An error!() line is invisible in a GUI: the button simply looked dead.
                        // Reuse the same warning strip every other settings failure uses.
                        error!("menu: '{}' does not look like EscapeFromTarkov_Data", state.game_dir_edit);
                        state.config_err = Some(
                            "That folder is not EscapeFromTarkov_Data \u{2014} pick the folder \
                             that contains globalgamemanagers."
                                .to_string(),
                        );
                    }
                }
                match &state.game_fp {
                    Some(fp) => ui.label(
                        RichText::new(format!("[{}]", &fp[..8])).color(OK).size(11.0),
                    ),
                    None => ui.label(
                        RichText::new(t(lg, K::GameNotFound)).color(WARN).size(11.0),
                    ),
                };
            });

            // Extracted-assets dir (EFT_ASSETS_ROOT): where the one-time full extraction writes the
            // datasets that BUILD reads. On first run explain it; CHOOSE opens a native folder picker.
            if !state.assets_ok {
                ui.add_space(2.0);
                ui.label(RichText::new(t(lg, K::FirstRunBanner)).color(WARN).size(11.0));
            } else {
                // Even after the folder is chosen (the verbose banner above is gone), keep a short
                // reminder that a map's FIRST build is a large one-time extraction — the user was
                // surprised by the extraction cost / thought a finished deps install had built a map.
                ui.add_space(2.0);
                ui.label(RichText::new(t(lg, K::FirstBuildHint)).color(DIM).size(10.0));
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new(t(lg, K::ExtractedAssets)).color(DIM).size(11.0));
                if ui.button(RichText::new(t(lg, K::Choose)).color(BONE)).clicked() {
                    let mut dlg = rfd::FileDialog::new().set_title(t(lg, K::FolderTitle));
                    if std::path::Path::new(&state.assets_dir).is_dir() {
                        dlg = dlg.set_directory(&state.assets_dir);
                    }
                    if let Some(p) = dlg.pick_folder() {
                        let dir = p.to_string_lossy().into_owned();
                        state.assets_dir = dir.clone();
                        state.assets_dir_edit = dir.clone();
                        state.config_err = (!save_config_assets_dir(&dir))
                            .then(|| t(lg, K::ConfigSaveFailed).to_string());
                        state.assets_ok = true;
                    }
                }
                let mut edit = state.assets_dir_edit.clone();
                ui.add(
                    egui::TextEdit::singleline(&mut edit)
                        .desired_width(420.0)
                        .font(egui::TextStyle::Monospace),
                );
                state.assets_dir_edit = edit;
                let dirty = state.assets_dir_edit != state.assets_dir;
                if ui
                    .add_enabled(dirty, egui::Button::new(RichText::new(t(lg, K::Set)).color(BONE)))
                    .clicked()
                {
                    state.assets_dir = state.assets_dir_edit.clone();
                    state.config_err = (!save_config_assets_dir(&state.assets_dir))
                        .then(|| t(lg, K::ConfigSaveFailed).to_string());
                    state.assets_ok = true;
                }
                if state.assets_ok {
                    ui.label(RichText::new(t(lg, K::IsSet)).color(OK).size(11.0));
                } else {
                    ui.label(RichText::new(t(lg, K::UsingDefault)).color(WARN).size(11.0));
                }
            });
        });

    // Build progress panel: shown only while a build/deps job runs or its finished panel lingers. Its
    // content is capped in an inner ScrollArea so a long streaming log scrolls INSIDE the panel
    // instead of growing it up over the map list.
    let mut toggle_log = false;
    if bv.is_some() {
        let show_log = state.show_log;
        egui::TopBottomPanel::bottom("menu_build")
            .frame(egui::Frame::new().fill(Color32::TRANSPARENT).inner_margin(egui::Margin::symmetric(24, 6)))
            .show(ctx, |ui| {
            // The title row + progress bar are PINNED in the panel (always fully visible while a build
            // runs); only the raw log below scrolls, in its OWN capped ScrollArea. Previously the whole
            // panel was wrapped in one ScrollArea, so a long log could scroll the loading bar off-screen.
            if let Some(bv) = &bv {
                let (stage, tail, key) = (&bv.stage, &bv.tail, &bv.key);
                let (finished, ok) = (bv.finished, bv.ok);
                let failed = finished && !ok;
                ui.add_space(10.0);
                egui::Frame::new()
                    .fill(HEADER)
                    .stroke(egui::Stroke::new(1.0, BORDER))
                    .inner_margin(10.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let title = if key == "__deps__" {
                                t(lg, K::InstallingDeps).to_string()
                            } else {
                                format!("{}: {}", t(lg, K::Building), map_title(lg, key, key).to_uppercase())
                            };
                            ui.label(RichText::new(title).color(BONE).strong());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if finished {
                                        let col = if ok { OK } else { BAD };
                                        let txt = t(lg, if ok { K::Done } else { K::Failed });
                                        ui.label(RichText::new(txt).color(col).strong());
                                        if ui.button(t(lg, K::Close)).clicked() {
                                            dismiss_build = true;
                                        }
                                    } else if ui
                                        .button(RichText::new(t(lg, K::Cancel)).color(BAD))
                                        .clicked()
                                    {
                                        cancel_current = true;
                                    }
                                    // The tail is hidden by default — the loader bar carries
                                    // the status; the raw log is one click away.
                                    if ui
                                        .button(t(lg, if show_log { K::HideLog } else { K::ShowLog }))
                                        .clicked()
                                    {
                                        toggle_log = true;
                                    }
                                    // Full captured log (the panel shows only a tail) — for
                                    // diagnosing which stage failed / sharing the output.
                                    if ui.button(t(lg, K::CopyLog)).clicked() {
                                        ui.ctx().copy_text(bv.full_log.clone());
                                    }
                                },
                            );
                        });
                        // Weighted, MONOTONIC progress — computed once in build_view (phases weighted by
                        // real relative duration so the ETA stops overshooting on the long extraction;
                        // clamped so a nested sub-script's [STAGE i/M] marker can't jump the bar backward).
                        let frac = bv.frac;
                        // "LOADING OBJECTS..." style stage line for the loader bar: the text
                        // between the [STAGE] marker and its status suffix, uppercased and
                        // ASCII-whitelisted (menu glyph set is plain ASCII only).
                        let stage_txt = if failed {
                            t(lg, K::BuildFailed).to_string()
                        } else if finished {
                            // Only a deps-install reaches here now (a successful MAP build auto-dismisses
                            // in build_view), so say "dependencies installed" rather than "BUILD COMPLETE"
                            // — the user was confused that a finished deps install looked like a built map.
                            if key == "__deps__" {
                                t(lg, K::DepsDone).to_string()
                            } else {
                                t(lg, K::BuildComplete).to_string()
                            }
                        } else {
                            let mut en = stage
                                .split(']')
                                .nth(1)
                                .unwrap_or("")
                                .split(':')
                                .next()
                                .unwrap_or("")
                                .trim()
                                .to_ascii_uppercase();
                            en.retain(|c| c.is_ascii_graphic() || c == ' ');
                            // The raw log is ASCII-only; the Russian stage name comes from our map
                            // (not the log), so it renders fine past the ASCII whitelist above.
                            let mut s = crate::i18n::stage_ru(lg, &en).map(str::to_string).unwrap_or(en);
                            if s.is_empty() {
                                s = t(lg, K::Starting).to_string();
                            }
                            // char-safe cap (Cyrillic is multi-byte; String::truncate would panic).
                            let capped: String = s.chars().take(38).collect();
                            format!("{capped}...")
                        };
                        ui.add_space(8.0);
                        crate::menu_fx::eft_loading_bar(ui, frac, &stage_txt, bv.started_secs, failed, lg);
                        if show_log {
                            ui.add_space(6.0);
                            // Only the streaming log scrolls (capped) — the loading bar above stays pinned.
                            egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                                for line in tail {
                                    ui.label(
                                        RichText::new(line).color(DIM).size(11.0).monospace(),
                                    );
                                }
                            });
                        }
                    });
            }
            });
    }
    if toggle_log {
        state.show_log = !state.show_log;
    }

    // ---- SETTINGS: a tabbed panel beside the map list ------------------------------------------
    // Collapsed by default (the map list is the point of this screen). Everything binds to
    // `MenuState.overlay` (`overlay::OverlayConfig`) and every change is pushed to the LIVE
    // resource as well as persisted — writing only the file is what made the toggle look broken.
    if let Some(tab) = state.settings_tab {
        egui::SidePanel::right("menu_settings")
            .default_width(330.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::CARD_TRANSLUCENT)
                    .inner_margin(egui::Margin::symmetric(16, 14)),
            )
            .show(ctx, |ui| {
                let mut cfg = state.overlay.clone();
                let mut dirty = false;
                ui.horizontal(|ui| {
                    ui.label(theme::title(t(lg, K::Settings)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("close").clicked() {
                            state.settings_tab = None;
                        }
                    });
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let tabs = [
                        t(lg, K::SettingsTabOverlay),
                        t(lg, K::SettingsTabLive),
                        t(lg, K::SettingsTabGeneral),
                    ];
                    for (i, name) in tabs.iter().enumerate() {
                        if ui
                            .selectable_label(tab == i as u8, RichText::new(*name).size(11.0))
                            .clicked()
                        {
                            state.settings_tab = Some(i as u8);
                        }
                    }
                });
                ui.separator();
                ui.add_space(4.0);
                egui::ScrollArea::vertical().show(ui, |ui| match tab {
                    0 => {
                        dirty |= ui
                            .checkbox(&mut cfg.enabled, RichText::new(t(lg, K::OverlayEnable)).size(11.0))
                            .on_hover_text(t(lg, K::OverlayEnableTip))
                            .changed();
                        ui.label(
                            RichText::new(t(lg, K::OverlayBorderlessNote)).size(10.0).color(DIM),
                        );
                        ui.add_space(8.0);
                        // LAUNCH configuration, deliberately OUTSIDE the enabled gate below:
                        // the presentation mode and ESP apply when a map is next opened, not when
                        // the overlay is summoned, and greying them behind the summon master
                        // switch hid the only settings a transparent-mode user actually needs.
                        // Choosing Transparent is itself the overlay opt-in (apply_overlay treats
                        // a transparent launch as engaged), so gating it on `enabled` was
                        // circular.
                        {
                            // ONE mode choice, not independent toggles: of the 8 combinations the
                            // old booleans allowed, transparency works under exactly one, and DWM
                            // fails the other seven silently (see OverlayPresentation). A radio
                            // row makes the invalid states unrepresentable instead of documented.
                            use crate::overlay::OverlayPresentation as P;
                            ui.label(RichText::new(t(lg, K::OverlayPresentationLabel)).size(10.0).color(DIM));
                            for (mode, label, tip) in [
                                (P::Windowed, K::OverlayModeWindowed, K::OverlayModeWindowedTip),
                                (P::Borderless, K::OverlayModeBorderless, K::OverlayModeBorderlessTip),
                                (P::Transparent, K::OverlayModeTransparent, K::OverlayModeTransparentTip),
                            ] {
                                let mut resp = ui.radio_value(
                                    &mut cfg.presentation,
                                    mode,
                                    RichText::new(t(lg, label)).size(11.0),
                                );
                                resp = resp.on_hover_text(t(lg, tip));
                                dirty |= resp.changed();
                            }
                            // The window's shape is latched at creation, so a change here cannot
                            // take effect on an already-open map. The menu relaunches on PLAY,
                            // which is exactly when this applies -- say so instead of letting the
                            // user wonder why the open window did not change.
                            if cfg.presentation == P::Transparent {
                                ui.label(
                                    RichText::new(t(lg, K::OverlayNextLaunchNote)).size(10.0).color(DIM),
                                );
                            }
                            ui.add_space(8.0);
                            let mut esp = config_bool("espMode").unwrap_or(false);
                            if ui
                                .checkbox(&mut esp, RichText::new(t(lg, K::EspModeLabel)).size(11.0))
                                .on_hover_text(t(lg, K::EspModeTip))
                                .changed()
                            {
                                let _ = save_config_bool_pub("espMode", esp);
                            }
                            ui.add_space(8.0);
                        }
                        ui.add_enabled_ui(cfg.enabled, |ui| {
                            // Size and anchor shape the PANEL modes; Transparent covers the whole
                            // game window by definition, so these sliders would do nothing there.
                            // Disable rather than let them lie (the Graphics-panel lesson).
                            let panel_geometry = cfg.presentation
                                != crate::overlay::OverlayPresentation::Transparent;
                            ui.add_enabled_ui(panel_geometry, |ui| {
                            ui.label(RichText::new(t(lg, K::OverlayPanelSize)).size(10.0).color(DIM));
                            dirty |= ui.add(egui::Slider::new(&mut cfg.size_frac.x, 0.2..=1.0).text("width")).changed();
                            dirty |= ui.add(egui::Slider::new(&mut cfg.size_frac.y, 0.2..=1.0).text("height")).changed();
                            ui.label(RichText::new(t(lg, K::OverlayPanelPos)).size(10.0).color(DIM));
                            dirty |= ui.add(egui::Slider::new(&mut cfg.anchor.x, 0.0..=1.0).text("x")).changed();
                            dirty |= ui.add(egui::Slider::new(&mut cfg.anchor.y, 0.0..=1.0).text("y")).changed();
                            });
                            ui.add_space(8.0);
                            ui.label(RichText::new(t(lg, K::OverlayPerf)).size(10.0).color(DIM));
                            let mut cap = cfg.fps_cap as f32;
                            if ui
                                .add(egui::Slider::new(&mut cap, 0.0..=360.0).text(t(lg, K::OverlayFpsCap)))
                                .on_hover_text(t(lg, K::OverlayFpsCapTip))
                                .changed()
                            {
                                cfg.fps_cap = cap.round() as u32;
                                dirty = true;
                            }
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(t(lg, K::OverlayExitHotkey))
                                        .size(10.0)
                                        .color(DIM),
                                );
                                egui::ComboBox::from_id_salt("overlay_exit_hotkey")
                                    .selected_text(cfg.exit_hotkey.label())
                                    .show_ui(ui, |ui| {
                                        for key in crate::overlay::OverlayExitHotkey::ALL {
                                            dirty |= ui
                                                .selectable_value(
                                                    &mut cfg.exit_hotkey,
                                                    key,
                                                    key.label(),
                                                )
                                                .changed();
                                        }
                                    });
                            });
                            dirty |= ui
                                .checkbox(&mut cfg.pause_when_hidden, RichText::new(t(lg, K::OverlayIdleHidden)).size(11.0))
                                .on_hover_text(t(lg, K::OverlayIdleHiddenTip))
                                .changed();
                            ui.add_space(8.0);
                            ui.separator();
                            // THE one-keypress flow: the player's OWN screenshot key does both.
                            dirty |= ui
                                .checkbox(
                                    &mut cfg.show_on_screenshot,
                                    RichText::new(t(lg, K::OverlayOpenOnShot)).size(11.0),
                                )
                                .on_hover_text(t(lg, K::OverlayOpenOnShotTip))
                                .changed();
                            dirty |= ui
                                .checkbox(
                                    &mut cfg.return_focus_to_game,
                                    RichText::new(t(lg, K::OverlayReturnFocus)).size(11.0),
                                )
                                .on_hover_text(t(lg, K::OverlayReturnFocusTip))
                                .changed();
                        });
                    }
                    1 => {
                        // LINK HEALTH, first: every watcher failure is non-fatal, so a dead link
                        // (no game dir, no Logs folder, a changed log format) used to look exactly
                        // like a healthy idle one. Show what the watcher actually resolved.
                        {
                            use std::sync::atomic::Ordering::Relaxed;
                            let h = &crate::game_watch::LINK_HEALTH;
                            let row = |ui: &mut egui::Ui, ok: bool, label: &str| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(if ok { "\u{2713}" } else { "\u{2717}" })
                                            .color(if ok { OK } else { WARN })
                                            .size(11.0),
                                    );
                                    ui.label(RichText::new(label).size(11.0).color(DIM));
                                });
                            };
                            let alive = h.ticks.load(Relaxed) > 0;
                            egui::Frame::new().fill(theme::INSET).inner_margin(8.0).show(ui, |ui| {
                                ui.label(RichText::new(t(lg, K::LinkHealth)).size(11.0).color(BONE));
                                row(ui, alive, t(lg, K::LinkWatcher));
                                row(ui, h.game_dir.load(Relaxed), t(lg, K::LinkGameDir));
                                row(ui, h.logs_dir.load(Relaxed), t(lg, K::LinkLogsDir));
                                row(ui, h.app_log.load(Relaxed), t(lg, K::LinkAppLog));
                                row(ui, h.shots_dir.load(Relaxed), t(lg, K::LinkShotsDir));
                                let ev = h.events.load(Relaxed);
                                ui.label(
                                    RichText::new(format!("{} {ev}", t(lg, K::LinkEvents)))
                                        .size(10.0)
                                        .color(if ev > 0 { DIM } else { WARN }),
                                );
                                if h.app_log.load(Relaxed) && ev == 0 {
                                    ui.label(
                                        RichText::new(t(lg, K::LinkNoEventsHint))
                                            .size(10.0)
                                            .color(WARN),
                                    );
                                }
                            });
                            ui.add_space(6.0);
                        }
                        let loc_now = {
                            let mut loc = state.screenshot_locate;
                            if ui
                                .checkbox(&mut loc, RichText::new(t(lg, K::ScreenshotLocate)).size(11.0))
                                .on_hover_text(t(lg, K::ScreenshotLocateTip))
                                .changed()
                            {
                                state.screenshot_locate = loc;
                                crate::game_watch::set_screenshot_locate(loc);
                                // Was `let _ =`: on a read-only folder the toggle silently
                                // reverted next launch. Surface it like every other save does.
                                state.config_err = (!save_config_screenshot_locate(loc))
                                    .then(|| t(lg, K::ConfigSaveFailed).to_string());
                            }
                            loc
                        };
                        // Sub-option of screenshot-locate: it only means anything while the folder
                        // is being polled, so it lives INDENTED under its parent and greys out
                        // with it (a live checkbox for a dead feature reads as a bug).
                        ui.horizontal(|ui| {
                            ui.add_space(22.0); // indent under the parent checkbox
                            ui.add_enabled_ui(loc_now, |ui| {
                                dirty |= ui
                                    .checkbox(
                                        &mut cfg.delete_processed_shots,
                                        RichText::new(t(lg, K::DeleteShots)).size(11.0),
                                    )
                                    .on_hover_text(t(lg, K::DeleteShotsTip))
                                    .changed();
                            });
                        });
                        ui.add_space(8.0);
                        ui.label(RichText::new(t(lg, K::LiveLinkNote)).size(10.0).color(DIM));
                    }
                    _ => {
                        let mut bg = state.process_in_background;
                        if ui
                            .checkbox(&mut bg, RichText::new(t(lg, K::ProcessInBackground)).size(11.0))
                            .on_hover_text(t(lg, K::ProcessInBackgroundTip))
                            .changed()
                        {
                            state.process_in_background = bg;
                            state.config_err = (!save_config_process_in_background(bg))
                                .then(|| t(lg, K::ConfigSaveFailed).to_string());
                        }
                        ui.add_space(6.0);
                        // FORCE CPU PROCESSING — bake lighting/terrain on the CPU (EFT_BAKE_CPU=1
                        // exported to the build child). The escape hatch for GPUs whose driver
                        // crashes/TDRs in the compute bakes (e.g. a broken Vulkan ICD).
                        let mut cpu = state.force_cpu_process;
                        if ui
                            .checkbox(&mut cpu, RichText::new(t(lg, K::ForceCpuProcess)).size(11.0))
                            .on_hover_text(t(lg, K::ForceCpuProcessTip))
                            .changed()
                        {
                            state.force_cpu_process = cpu;
                            if !save_config_force_cpu_process(cpu) {
                                state.config_err = Some(
                                    t(lg, K::ConfigSaveFailed).to_string(),
                                );
                            }
                        }
                        ui.add_space(10.0);
                        // QUALITY PRESET — set HERE, before a map loads, and carried into the
                        // scene. It has to live in the menu: texture quality is applied while
                        // textures upload, so picking it in-raid cannot change what was uploaded.
                        // main.rs seeds both the mip-skip and GfxSettings from `qualityPreset`.
                        // Every number in the summaries is measured (docs/GFX_BENCH_*.json).
                        ui.label(RichText::new(t(lg, K::Quality)).size(11.0))
                            .on_hover_text(t(lg, K::QualityTip));
                        ui.horizontal(|ui| {
                            for p in crate::render::QualityPreset::ALL {
                                let sum = match p {
                                    crate::render::QualityPreset::Low => K::QualityLowSum,
                                    crate::render::QualityPreset::Medium => K::QualityMediumSum,
                                    crate::render::QualityPreset::High => K::QualityHighSum,
                                    crate::render::QualityPreset::Ultra => K::QualityUltraSum,
                                    crate::render::QualityPreset::Custom => K::QualityCustomSum,
                                };
                                if ui
                                    .selectable_label(
                                        state.quality_preset == p.index(),
                                        RichText::new(p.label()).size(11.0),
                                    )
                                    .on_hover_text(t(lg, sum))
                                    .clicked()
                                {
                                    state.quality_preset = p.index();
                                    // A non-Custom preset owns texture quality; keep the persisted
                                    // value and the visible Texture-quality row in agreement.
                                    if let Some(q) = p.tex_quality() {
                                        state.tex_quality = q;
                                        crate::render::gpu_driven::set_tex_mip_skip(q);
                                    }
                                    if !save_quality_preset_pub(p) {
                                        state.config_err = Some(
                                            t(lg, K::ConfigSaveFailed).to_string(),
                                        );
                                    }
                                }
                            }
                        });
                        {
                            let p = crate::render::QualityPreset::from_index(state.quality_preset);
                            let sum = match p {
                                crate::render::QualityPreset::Low => K::QualityLowSum,
                                crate::render::QualityPreset::Medium => K::QualityMediumSum,
                                crate::render::QualityPreset::High => K::QualityHighSum,
                                crate::render::QualityPreset::Ultra => K::QualityUltraSum,
                                crate::render::QualityPreset::Custom => K::QualityCustomSum,
                            };
                            ui.label(RichText::new(t(lg, sum)).size(10.0).color(DIM));
                        }
                        ui.add_space(8.0);
                        // TEXTURE QUALITY — the one VRAM lever that matters (docs/VRAM_AUDIT.md:
                        // textures are ~59% of streets' residency; Half reclaims ~3.8 GiB there).
                        ui.label(RichText::new(t(lg, K::TexQuality)).size(11.0))
                            .on_hover_text(t(lg, K::TexQualityTip));
                        ui.horizontal(|ui| {
                            for (i, key) in
                                [K::TexQualityFull, K::TexQualityHalf, K::TexQualityQuarter]
                                    .into_iter()
                                    .enumerate()
                            {
                                if ui
                                    .selectable_label(
                                        state.tex_quality == i as u8,
                                        RichText::new(t(lg, key)).size(11.0),
                                    )
                                    .on_hover_text(t(lg, K::TexQualityTip))
                                    .clicked()
                                {
                                    state.tex_quality = i as u8;
                                    crate::render::gpu_driven::set_tex_mip_skip(i as u8);
                                    // Hand-picking a texture quality deviates from whatever preset
                                    // was active, so the preset becomes Custom rather than lying.
                                    if crate::render::QualityPreset::from_index(state.quality_preset)
                                        .tex_quality()
                                        != Some(i as u8)
                                    {
                                        state.quality_preset =
                                            crate::render::QualityPreset::Custom.index();
                                        if !save_config_f32_pub(
                                            "qualityPreset",
                                            crate::render::QualityPreset::Custom.index() as f32,
                                        ) {
                                            state.config_err = Some(
                                                "settings could not be saved (read-only folder?)"
                                                    .to_string(),
                                            );
                                        }
                                    }
                                    if !save_config_f32_pub("textureQuality", i as f32) {
                                        state.config_err = Some(
                                            "settings could not be saved (read-only folder?)"
                                                .to_string(),
                                        );
                                    }
                                }
                            }
                        });
                        ui.label(
                            RichText::new(t(lg, K::TexQualityNote)).size(10.0).color(DIM),
                        );
                    }
                });
                if dirty {
                    let cfg = cfg.sanitized();
                    state.overlay = cfg.clone();
                    *overlay_cfg = cfg.clone(); // live: no relaunch needed
                    crate::game_watch::set_delete_processed_shots(cfg.delete_processed_shots);
                    *settings_save_pending = true; // persisted below, once the pointer is up
                }
            });
    }
    // Deferred settings persist: live-apply happened the frame the value changed; the file write
    // waits for the drag to end so a slider doesn't rewrite atlas.config.json hundreds of times.
    if *settings_save_pending && !ctx.input(|i| i.pointer.any_down()) {
        *settings_save_pending = false;
        state.config_err = (!state.overlay.save())
            .then(|| t(lg, K::ConfigSaveFailed).to_string());
    }

    egui::CentralPanel::default()
        .frame(
            // TRANSPARENT so the 3D neon globe (menu_fx::spawn_menu_globe, glowing in the 3D world
            // behind egui) shows through. The ClearColor (#090909) is the dark field it glows on.
            egui::Frame::new()
                .fill(Color32::TRANSPARENT)
                .inner_margin(24.0),
        )
        .show(ctx, |ui| {
            let _ = real_prop; // the CCTV prop is retired; the 3D neon globe is the backdrop now

            let mut delete_now: Option<usize> = None;
            let mut rescan = false;
            let mut set_confirm: Option<usize> = None;
            // Explicit disarm for the delete confirm. Without it an accidentally armed row
            // stayed armed indefinitely: the flag was only ever cleared BY deleting.
            let mut clear_confirm = false;
            let mut set_rebuild: Option<usize> = None;
            // (map key, force): BUILD -> force=false (incremental), UPDATE -> force=true.
            let mut start_build: Option<(String, bool)> = None;
            let confirm_idx = state.confirm_delete;
            // Which map (if any) is being built RIGHT NOW — marks that row BUILDING and blocks the
            // other BUILD buttons. A finished build no longer blocks (its panel lingers until CLOSE).
            let building_key = wk_build_key.clone();
            // BUILD needs the extraction deps installed AND a valid game dir; gate the button on both
            // (Copy locals so they reach the inner button closure without borrowing `state`). Without
            // this a user can click BUILD before INSTALL DEPS / before setting GAME INSTALL and hit a
            // confusing mid-pipeline failure. deps_ok is true when deps are present OR unprobed.
            // Also require the build KIT (finding 11): a shipped-lite bundle has no tools/ so BUILD
            // and UPDATE must both stay disabled instead of spawn-erroring. UPDATE used to bypass
            // this entirely — it now shares `can_build`.
            let can_build = state.build_kit_available && state.deps_ok && state.game_fp.is_some();
            // PLAY is gated on a tarkov.dev sync having happened at least once (loot/spawns/intel come
            // from it — playing before a sync gives an empty map). `intel` is (loot age, tasks age,
            // icons); a present loot age means the sync ran. Copied to a local so it reaches the PLAY
            // button inside the card loop without borrowing `state`.
            let intel_synced = state.intel.0.is_some();
            // The map LIST fills whatever height the header + the pinned bottom panels (build +
            // footer) leave: a plain ScrollArea::vertical() caps itself to that and scrolls the rows.
            // (The old fixed `reserve` estimate is gone — the footer + build panel are their own
            // bottom panels now, so they can no longer be clipped and need no height reservation.)
            // Card fill: FULLY opaque so no backdrop lines show through the menu rows and the
            // vignette (Background layer, behind every opaque panel) only ever darkens the blue
            // background between/around them, never the UI text.
            let card_bg = CARD;
            // Name the ONE condition that's actually failing (the old combined "install deps AND
            // set game install" text told a user with deps ready + no install button to install
            // deps they already had).
            let setup_key = if !state.build_kit_available {
                K::BuildKitMissing
            } else if !state.deps_ok {
                K::DepsFirst
            } else {
                K::SetGameFirst
            };
            if !can_build {
                // Persistent INLINE reason (not just the disabled-button hover tooltip) so a
                // non-technical user actually sees WHY every BUILD is greyed out and what to do.
                egui::Frame::new()
                    .fill(theme::INSET)
                    .stroke(egui::Stroke::new(1.0, WARN))
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(format!("\u{26A0} {}", t(lg, setup_key)))
                                .color(WARN)
                                .size(12.0),
                        );
                    });
                ui.add_space(8.0);
            }
            if !intel_synced {
                // Global reason PLAY is disabled on every row until a tarkov.dev sync runs (SYNC NOW,
                // top) — inline, not just a hover, so a non-technical user knows what to do.
                egui::Frame::new()
                    .fill(theme::INSET)
                    .stroke(egui::Stroke::new(1.0, WARN))
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(format!("\u{26A0} {}", t(lg, K::PlayNeedsSync)))
                                .color(WARN)
                                .size(12.0),
                        );
                    });
                ui.add_space(8.0);
            }
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Right gutter: keep the map rows clear of the globe backdrop's right side.
                ui.set_max_width((ui.available_width() - 166.0).max(430.0));
                for i in 0..state.entries.len() {
                    let e = &state.entries[i];
                    let installed = e.pack_dir.is_some();
                    egui::Frame::new()
                        .fill(card_bg)
                        .stroke(egui::Stroke::new(1.0, BORDER))
                        .corner_radius(0.0)
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.set_min_height(34.0);
                                let title = RichText::new(map_title(lg, &e.key, &e.title))
                                    .size(theme::SIZE_ROW_TITLE)
                                    .strong()
                                    .color(if installed { BEIGE } else { DIM });
                                ui.add_sized([220.0, 30.0], egui::Label::new(title));
                                // Status badge.
                                let (txt, col) = if !installed {
                                    (t(lg, K::NotInstalled), DIM)
                                } else if !e.valid {
                                    // Pack dir present but manifest/meshes/instances broken or
                                    // missing (finding 4): never show READY — PLAY would blank out.
                                    (t(lg, K::Damaged), BAD)
                                } else {
                                    match e.fp_match {
                                        Some(true) => (t(lg, K::Ready), OK),
                                        // Game-file hashes changed since this pack was built.
                                        Some(false) => (t(lg, K::GameFilesUpdated), WARN),
                                        None => (t(lg, K::ReadyUnstamped), WARN),
                                    }
                                };
                                ui.label(RichText::new(txt).color(col).size(12.0).strong());
                                ui.add_space(10.0);
                                if installed {
                                    ui.label(RichText::new(fmt_size(e.size_bytes)).color(BEIGE));
                                    ui.label(
                                        RichText::new(format!("{} {}", t(lg, K::BuiltLabel), fmt_age_lg(lg, e.built_days)))
                                            .color(DIM)
                                            .size(11.0),
                                    );
                                    ui.label(
                                        RichText::new(format!("{} {}", t(lg, K::IntelLabel), fmt_age_lg(lg, e.intel_days)))
                                            .color(DIM)
                                            .size(11.0),
                                    );
                                    // Each flag says what the PACK carries, which is not always
                                    // what the player sees on screen (a greyed "grass sim" on a
                                    // visibly grassy map read as a bug) - so each one explains
                                    // itself on hover.
                                    let tick = |on: bool, s: &str| theme::tick(on, s);
                                    ui.label(tick(e.has_volume, t(lg, K::TickLight)))
                                        .on_hover_text(t(lg, K::TickLightTip));
                                    ui.label(tick(e.has_grass, t(lg, K::TickGrass)))
                                        .on_hover_text(t(lg, K::TickGrassTip));
                                    ui.label(tick(e.has_gamedata, t(lg, K::TickZones)))
                                        .on_hover_text(t(lg, K::TickZonesTip));
                                    ui.label(tick(e.has_icons, t(lg, K::TickIcons)))
                                        .on_hover_text(t(lg, K::TickIconsTip));
                                }
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let this_building =
                                            building_key.as_deref() == Some(e.key.as_str());
                                        let any_building = building_key.is_some();
                                        if installed {
                                            // Only a VALID pack is playable (finding 4). A damaged
                                            // pack offers REBUILD (plain BUILD) where PLAY would be,
                                            // so PLAY can never open a blank window; DELETE stays
                                            // available below either way.
                                            if e.valid {
                                                // Force a tarkov.dev sync before playing: PLAY is
                                                // disabled until intel has been synced at least once.
                                                //
                                                // ALSO disabled while ANY map build runs, and that
                                                // one is about the GPU, not about data. The build's
                                                // SH bake is a separate `atlas bake-sh` worker that
                                                // wants the GPU; the viewer takes an exclusive
                                                // interactive-GPU lease for its whole lifetime
                                                // (gpu_lease.rs). Playing mid-build therefore costs
                                                // one of two ways: start the viewer first and the
                                                // bake quietly drops to the CPU path (observed: 70
                                                // CPU-minutes and 9 GB on interchange instead of a
                                                // GPU pass), or start it during a GPU bake and both
                                                // saturate one adapter — which is the Windows TDR
                                                // reset that lost the viewer's device and aborted
                                                // the process. The lease already prevents the
                                                // crash; this stops the user walking into either.
                                                let play = theme::primary_button(t(lg, K::Play));
                                                // A missing tarkov.dev sync costs LAYERS (loot
                                                // prices, task metadata), not the map — gating
                                                // PLAY on it meant a user who built a map offline
                                                // could never open it, with no override anywhere.
                                                // The inline banner above already explains the
                                                // degraded state; let them in.
                                                let playable = !any_building;
                                                let resp = ui
                                                    .add_enabled_ui(playable, |ui| {
                                                        ui.add_sized([84.0, 30.0], play)
                                                    })
                                                    .inner;
                                                if playable {
                                                    let resp = if intel_synced {
                                                        resp
                                                    } else {
                                                        resp.on_hover_text(t(lg, K::PlayNeedsSync))
                                                    };
                                                    if resp.clicked() {
                                                        switch.0 = e.pack_dir.clone();
                                                    }
                                                } else {
                                                    resp.on_disabled_hover_text(t(lg, K::PlayBusyBuilding));
                                                }
                                            } else {
                                                let reb = theme::warn_button(if this_building {
                                                    "..."
                                                } else {
                                                    t(lg, K::Build)
                                                });
                                                let resp = ui
                                                    .add_enabled_ui(!any_building && can_build, |ui| {
                                                        ui.add_sized([84.0, 30.0], reb)
                                                    })
                                                    .inner;
                                                let resp = if !can_build {
                                                    resp.on_disabled_hover_text(t(lg, setup_key))
                                                } else {
                                                    resp
                                                };
                                                if resp.clicked() {
                                                    start_build = Some((e.key.to_string(), false));
                                                }
                                            }
                                            // UPDATE sits BETWEEN delete and play: shown when the
                                            // game-file hashes no longer match the pack's stamp —
                                            // re-runs the pipeline so the data catches up. (Not for a
                                            // damaged pack — REBUILD above already covers that.)
                                            if e.valid && e.fp_match == Some(false) {
                                                let upd = theme::warn_button(if this_building {
                                                    "..."
                                                } else {
                                                    t(lg, K::RebuildPack)
                                                });
                                                // UPDATE needs the kit+deps just like BUILD
                                                // (finding 11) and re-extracts stale data via
                                                // --force (finding 1 / release blocker).
                                                let resp = ui
                                                    .add_enabled_ui(!any_building && can_build, |ui| {
                                                        ui.add_sized([84.0, 30.0], upd).on_hover_text(t(lg, K::UpdateTip))
                                                    })
                                                    .inner;
                                                let resp = if !can_build {
                                                    resp.on_disabled_hover_text(t(lg, setup_key))
                                                } else {
                                                    resp
                                                };
                                                if resp.clicked() {
                                                    start_build = Some((e.key.to_string(), true));
                                                }
                                            }
                                            // Tarkov-style destructive button: red fill, black text.
                                            let del_btn = |s: &str| theme::danger_button(s);
                                            if confirm_idx == Some(i) {
                                                // CONFIRM used to replace DELETE in the same slot
                                                // under the same cursor, so the second click of a
                                                // habitual double-click deleted the pack outright.
                                                // A cancel sits where CONFIRM now is, and CONFIRM
                                                // moves right — the armed row is also escapable,
                                                // which it never was (the flag only cleared by
                                                // deleting).
                                                if ui
                                                    .add_sized([84.0, 30.0], egui::Button::new(
                                                        RichText::new(t(lg, K::Cancel)).color(BONE),
                                                    ))
                                                    .clicked()
                                                {
                                                    clear_confirm = true;
                                                }
                                                if ui
                                                    .add_sized([120.0, 30.0], del_btn(t(lg, K::Confirm)))
                                                    .on_hover_text(t(lg, K::DeleteConfirmTip))
                                                    .clicked()
                                                {
                                                    delete_now = Some(i);
                                                }
                                            } else if ui
                                                .add_enabled_ui(!this_building, |ui| {
                                                    ui.add_sized([84.0, 30.0], del_btn(t(lg, K::Delete)))
                                                })
                                                .inner
                                                .clicked()
                                            {
                                                set_confirm = Some(i);
                                            }
                                        } else {
                                            let build_label = if this_building {
                                                format!("{}...", t(lg, K::Building))
                                            } else {
                                                t(lg, K::Build).to_string()
                                            };
                                            let b = egui::Button::new(RichText::new(build_label));
                                            let resp = ui.add_enabled(!any_building && can_build, b);
                                            let resp = if !can_build {
                                                resp.on_disabled_hover_text(t(lg, setup_key))
                                            } else {
                                                resp
                                            };
                                            if resp.clicked() {
                                                start_build = Some((e.key.to_string(), false));
                                            }
                                        }
                                    },
                                );
                            });
                        });
                    ui.add_space(6.0);
                }
            });

            if clear_confirm {
                state.confirm_delete = None;
            }
            if let Some(i) = set_confirm {
                state.confirm_delete = Some(i);
            }
            if let Some(i) = set_rebuild {
                state.show_rebuild = Some(i);
            }
            if let Some(i) = delete_now {
                if let Some(dir) = state.entries[i].pack_dir.clone() {
                    // Safety: only ever delete inside the resolved packs root (entry paths are
                    // built from it, but belt-and-braces against future edits).
                    let p = std::path::Path::new(&dir);
                    if p.starts_with(crate::paths::packs_root()) {
                        match std::fs::remove_dir_all(p) {
                            Ok(()) => info!("menu: deleted {dir}"),
                            Err(e) => error!("menu: delete {dir} failed: {e}"),
                        }
                    } else {
                        error!("menu: refusing to delete outside packs root: {dir}");
                    }
                    rescan = true;
                }
                state.confirm_delete = None;
            }
            if rescan {
                let (entries, total) = scan(&state.game_fp);
                state.entries = entries;
                state.total_bytes = total;
            }

            let _ = set_rebuild; // (legacy instructions window removed — BUILD runs the pipeline)

            // Kick a build: enqueue on the shared worker after this closure (one at a time;
            // tools/build_map.py streams staged output; the completion handler up top rescans).
            if let Some((key, force)) = start_build {
                info!(
                    "menu: queueing {} '{key}' via tools/build_map.py",
                    if force { "UPDATE (forced re-extract)" } else { "build" }
                );
                enqueue_build = Some((key, force));
                state.show_log = false; // fresh panel starts with the log collapsed
            }
        });

    // Language switch — floated on its OWN foreground Area anchored to the window bottom-right so a
    // short window's (non-scrolling) CentralPanel overflow can NEVER clip it. It used to be the last
    // widget inside the panel flow and fell off the bottom on small windows. Shared with the in-raid
    // viewer (`lang_switch_area`) so the control is identical in both places.
    if let Some(l) = lang_switch_area(
        ctx,
        *lang,
        "menu_lang_toggle",
        egui::Align2::RIGHT_BOTTOM,
        egui::vec2(-20.0, -14.0),
    ) {
        *lang = l;
        if !save_config_lang(l.tag()) {
            state.config_err =
                Some("language change could not be saved (read-only folder?)".to_string());
        }
    }

    // A failed config write (finding 12) — surface it so the user knows the setting won't persist
    // instead of it silently reverting on restart. Floated bottom-left, clear of the lang toggle.
    if let Some(msg) = state.config_err.clone() {
        egui::Area::new(egui::Id::new("menu_config_err"))
            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(20.0, -14.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.label(RichText::new(format!("\u{26A0} {msg}")).color(theme::DANGER_TEXT).size(11.0));
            });
    }

    // A build/deps job that failed to even START (a rare spawn error — missing python kit, bad path)
    // used to be shown ONLY inside the intel row, so a BUILD / INSTALL DEPS that never launched read
    // as a dead click. Surface it globally, centred just above the footer, whatever the job type.
    if let Some(err) = worker.spawn_error.clone() {
        egui::Area::new(egui::Id::new("menu_spawn_err"))
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -42.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(format!("\u{26A0} {err}"))
                        .color(theme::DANGER_TEXT)
                        .size(12.0)
                        .strong(),
                );
            });
    }

    // ---- version indicator + update modal (menu-only GitHub check) --------------------------------
    // Both float on their OWN foreground Areas so a short window's overflow can't clip them, matching
    // the lang-toggle / config-err pattern above. All colors/frames come from `theme` (ui_theme).
    use crate::update::UpdateStatus;
    let available = matches!(&*update_status, UpdateStatus::Available { .. });

    // 1. Top-right version tag. Muted normally; an ACCENT dot + "update available" when a newer
    //    release exists. Anchored just below the header so it clears the header's size readout.
    egui::Area::new(egui::Id::new("menu_version"))
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-18.0, 52.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if available {
                    // A small accent badge dot before the label.
                    let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot.center(), 3.5, theme::ACCENT);
                    ui.label(
                        RichText::new(t(lg, K::UpdateAvailable))
                            .size(11.0)
                            .strong()
                            .color(theme::ACCENT),
                    );
                }
                ui.label(RichText::new(crate::update::APP_TAG).size(11.0).color(DIM));
            });
        });

    // 2. The update MODAL — shown while Available AND not dismissed this session. A dim backdrop
    //    (Middle order, below the card) swallows clicks to the menu behind it; the card (Foreground)
    //    reuses the MapLoadError/confirm-dialog idiom: a CARD-filled Frame with an ACCENT stroke,
    //    primary vs secondary theme buttons.
    if available && !update_check.dismissed {
        let (tag, url) = match &*update_status {
            UpdateStatus::Available { tag, url } => (tag.clone(), url.clone()),
            _ => (String::new(), String::new()),
        };
        let mut do_update = false;
        let mut do_later = false;

        // Dim backdrop.
        egui::Area::new(egui::Id::new("update_modal_dim"))
            .order(egui::Order::Middle)
            .fixed_pos(egui::Pos2::ZERO)
            .interactable(true)
            .show(ctx, |ui| {
                let screen = ctx.screen_rect();
                ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(170));
                ui.allocate_rect(screen, egui::Sense::click()); // block the menu behind
            });

        // The card.
        egui::Area::new(egui::Id::new("update_modal"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(CARD)
                    .stroke(egui::Stroke::new(1.0, theme::ACCENT))
                    .inner_margin(egui::Margin::symmetric(22, 18))
                    .corner_radius(0.0)
                    .show(ui, |ui| {
                        ui.set_max_width(440.0);
                        ui.horizontal(|ui| {
                            let (dot, _) =
                                ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(dot.center(), 4.0, theme::ACCENT);
                            ui.label(
                                RichText::new(t(lg, K::UpdateTitle))
                                    .size(theme::SIZE_TITLE)
                                    .strong()
                                    .color(theme::TEXT_BRIGHT),
                            );
                        });
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new(t(lg, K::UpdateBody).replace("{}", &tag))
                                .size(theme::SIZE_BODY)
                                .color(BONE),
                        );
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(t(lg, K::UpdateWarn)).size(theme::SIZE_SMALL).color(WARN),
                        );
                        ui.add_space(16.0);
                        ui.horizontal(|ui| {
                            // Primary: UPDATE (beige) opens the release page in the browser.
                            if ui.add_sized([120.0, 30.0], theme::primary_button(t(lg, K::Update))).clicked() {
                                do_update = true;
                            }
                            ui.add_space(8.0);
                            // Secondary: LATER (neutral outlined) dismisses for this session.
                            let later = egui::Button::new(
                                RichText::new(t(lg, K::UpdateLater)).size(15.5).strong().color(BONE),
                            )
                            .fill(HEADER)
                            .stroke(egui::Stroke::new(1.0, BORDER))
                            .corner_radius(0.0);
                            if ui.add_sized([120.0, 30.0], later).clicked() {
                                do_later = true;
                            }
                        });
                    });
            });

        if do_update {
            crate::update::open_url(&url);
            update_check.dismissed = true; // stop nagging once they've gone to the release page
        } else if do_later {
            update_check.dismissed = true;
        }
    }

    // ---- apply the worker intents collected above (single point of mutation) ----
    if enqueue_install {
        worker.enqueue(Job::InstallDeps);
    }
    if enqueue_sync {
        worker.enqueue(Job::SyncIntel);
    }
    if let Some((map, force)) = enqueue_build {
        worker.enqueue(Job::BuildMap {
            map,
            game_dir: state.game_dir.clone(),
            force,
            background: state.process_in_background,
        });
    }
    if cancel_current {
        worker.cancel_current();
    }
    if dismiss_build {
        worker.dismiss_last();
    }
}

#[cfg(test)]
mod tests {
    use super::{build_frac, subprogress_frac};

    // ---- subprogress_frac: BOTH marker generations must parse to the same ratio semantics.
    // The byte-weighted emitters (extract_parallel / eft_extract_colliders) put the machine
    // token LAST precisely so an already-shipped viewer keeps working — this is that contract.

    /// Replay a whole marker stream through `build_frac` + the `max_frac` clamp exactly as
    /// `build_view` drives them, and assert the property the bar actually has to have: it never
    /// goes backwards, and it never sits at one value across a pass boundary. The parser tests
    /// above check that a LINE is read correctly; this checks that the CURVE is usable, which is
    /// where both of the freezes this table was written to remove actually lived.
    fn replay(stream: &[&str]) -> Vec<f32> {
        let (mut tail, mut stage, mut mx, mut out) = (Vec::new(), String::new(), 0.0f32, Vec::new());
        let mut fresh = false;
        for line in stream {
            tail.push(line.to_string());
            if line.starts_with("[STAGE") || line.starts_with("[BUILD") {
                stage = line.to_string();
            }
            if stage.is_empty() {
                continue;
            }
            if stage.starts_with("[STAGE 1/") && stage.contains("extract dataset") {
                fresh = true;
            }
            let sub = if stage.starts_with("[STAGE 1/") {
                let tag = if stage.contains("physics colliders") { "colliders" } else { "extract" };
                tail.iter().rev().filter(|l| l.contains(tag)).find_map(|l| subprogress_frac(l))
            } else {
                None
            };
            mx = mx.max(build_frac(&stage, sub, fresh));
            out.push(mx);
        }
        out
    }

    /// The exact stream `tools/build_map.py --dry-run` emits on the fresh path.
    const DRY_RUN_FRESH: &[&str] = &[
        "[STAGE 1/9] check dataset",
        "[STAGE 1/9] extract dataset (geometry + textures)",
        "[SUBPROGRESS] extract levels 3/217 bytes 536870912/5463154688",
        "[SUBPROGRESS] extract levels 120/217 bytes 2936012800/5463154688",
        "[SUBPROGRESS] extract levels 210/217 bytes 5348024320/5463154688",
        "[STAGE 1/9] extract dataset (geometry + textures): done (0s)",
        "[STAGE 1/9] extract grass density",
        "[STAGE 1/9] extract grass density: done (0s)",
        "[STAGE 1/9] extract physics colliders",
        "[SUBPROGRESS] colliders levels 100/217 bytes 2726297600/5463154688",
        "[STAGE 1/9] extract physics colliders: done (0s)",
        "[STAGE 1/9] check dataset: done",
        "[STAGE 2/9] extract lights",
        "[STAGE 2/9] extract lights: done (0s)",
    ];

    #[test]
    fn bar_never_goes_backwards() {
        let f = replay(DRY_RUN_FRESH);
        for w in f.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "bar regressed: {:?}", f);
        }
    }

    #[test]
    fn first_build_does_not_open_a_third_full() {
        // "check dataset" is a bookend, not a pass: a fresh build must START near zero.
        let f = replay(DRY_RUN_FRESH);
        assert!(f[0] < 0.02, "fresh build opened at {:.3}", f[0]);
    }

    #[test]
    fn dataset_progress_is_not_discarded() {
        // The three dataset [SUBPROGRESS] lines must each move the bar. Before the bookend fix
        // they computed 3.3% / 17.8% / 32.4% against a latched 31.8% and were clamped away.
        let f = replay(DRY_RUN_FRESH);
        assert!(f[2] < f[3] && f[3] < f[4], "dataset readings did not advance: {:?}", &f[..5]);
    }

    #[test]
    fn colliders_pass_moves_the_bar() {
        // The regression this fix targets: the colliders marker must not inherit the dataset
        // pass's trailing ~1.0 and latch the top of the stage-1 span.
        let f = replay(DRY_RUN_FRESH);
        let at_marker = f[8]; // "[STAGE 1/9] extract physics colliders"
        let at_reading = f[9]; // its first own [SUBPROGRESS]
        assert!(at_marker < 0.60, "colliders marker jumped to {at_marker:.3} on stale sub");
        assert!(at_reading > at_marker, "colliders reading did not advance the bar");
    }

    #[test]
    fn both_tables_order_the_bake_after_the_assemble() {
        // The portable pipeline runs stage 3 (SH bake) AFTER stage 4 (assemble). A table ordered
        // by stage NUMBER puts the bake's window behind the bar and `max_frac` freezes it there
        // for the whole bake. Assert the ordering directly in both tables so neither regresses.
        for fresh in [true, false] {
            let assemble_end = build_frac("[STAGE 4/9] assemble pack: done (0s)", None, fresh);
            let bake_start = build_frac("[STAGE 3/9] bake lighting", Some(0.0), fresh);
            assert!(
                bake_start >= assemble_end - 1e-6,
                "fresh={fresh}: bake starts at {bake_start:.3} but assemble ended at {assemble_end:.3}"
            );
        }
    }

    #[test]
    fn rebuild_does_not_jump_to_half_on_the_first_marker() {
        // A rebuild skips extraction entirely, so stage 1 must not claim the extraction span.
        let f = build_frac("[STAGE 1/9] check dataset: done", None, false);
        assert!(f < 0.01, "rebuild opened at {f:.3}");
    }

    #[test]
    fn subprogress_parses_legacy_level_count() {
        let f = subprogress_frac("[SUBPROGRESS] extract 12/217").unwrap();
        assert!((f - 12.0 / 217.0).abs() < 1e-6);
    }

    #[test]
    fn subprogress_parses_byte_weighted_lines() {
        // the emitters' real shape: raw-byte ratio last
        let f = subprogress_frac("[SUBPROGRESS] extract levels 12/217 bytes 884998144/5463154688")
            .unwrap();
        assert!((f - 884998144.0 / 5463154688.0).abs() < 1e-6);
        // the token is unit-agnostic — floats parse identically
        let f = subprogress_frac("[SUBPROGRESS] colliders levels 3/200 bytes 50.0/4000.0").unwrap();
        assert!((f - 0.0125).abs() < 1e-6);
    }

    #[test]
    fn subprogress_rejects_noise_and_degenerate_totals() {
        assert_eq!(subprogress_frac("  [p3] level410: 7765 colliders in 60.0s"), None);
        assert_eq!(subprogress_frac("[SUBPROGRESS] extract levels 0/0 bytes 0/0"), None);
        assert_eq!(subprogress_frac("[SUBPROGRESS] extract garbage"), None);
    }

    // ---- build_frac: weight tables + the stage-1 sub-pass windows.

    #[test]
    fn build_ok_is_full() {
        assert_eq!(build_frac("[BUILD OK] pack ready", None, false), 1.0);
        assert_eq!(build_frac("[BUILD OK] pack ready", None, true), 1.0);
    }

    #[test]
    fn stage1_extract_window_standard_table() {
        // extraction sub-pass owns [0, 0.52] of stage 1's 0.55 span (measured: colliders is
        // nearly half the stage)
        // fresh=true: these three sub-passes only ever run during the ONE-TIME extraction, so the
        // fresh window (0..0.636) is the only table they can be reached under. Pinning them against
        // the rebuild table asserted a state the pipeline cannot produce.
        let f = build_frac("[STAGE 1/9] extract dataset (geometry + textures)", Some(1.0), true);
        assert!((f - 0.636 * 0.52).abs() < 1e-4);
    }

    #[test]
    fn stage1_colliders_without_signal_sits_at_the_START_of_its_window() {
        // the frozen-"28%" repro: colliders running, no SUBPROGRESS in the tail yet.
        //
        // This assertion CHANGED from 0.55 * (0.53 + 0.47*0.5) to the start of the slice, and the
        // midpoint it used to pin is why the pass still froze. The colliders pass emits progress
        // now (this PR added it), so "no signal" only ever means "the first line has not arrived
        // yet" -- a few seconds. Guessing the midpoint during those seconds puts the bar at
        // 48.65%, the first real reading is 48.63%, and `max_frac` keeps the guess: the pass then
        // reports for ~45 min with every reading clamped away. A pass that will report starts at
        // the bottom of its slice; only the signal moves it.
        let f = build_frac("[STAGE 1/9] extract physics colliders", None, true);
        assert!((f - 0.636 * 0.53).abs() < 1e-4);
    }

    #[test]
    fn fresh_table_dominates_extraction() {
        let f = build_frac("[STAGE 1/9] extract dataset (geometry + textures)", Some(0.5), true);
        assert!((f - 0.636 * 0.52 * 0.5).abs() < 1e-4);
        // stage-1 fully done on the fresh (measured) table = 0.636, not 0.55
        let f = build_frac("[STAGE 1/9] extract physics colliders: done (2673s)", None, true);
        assert!((f - 0.636).abs() < 1e-4);
    }

    #[test]
    fn fresh_windows_follow_wall_clock_order_through_the_bake() {
        // the portable pipeline runs stage 3 (SH bake) AFTER stage 4 (assemble); the fresh
        // windows encode that, so the raw frac no longer regresses (and the bar no longer
        // freezes) through the 27-minute bake.
        let assemble_done = build_frac("[STAGE 4/9] assemble pack: done (724s)", None, true);
        let bake_running = build_frac("[STAGE 3/9] bake lighting (portable SH)", None, true);
        let bake_done = build_frac("[STAGE 3/9] bake lighting (portable SH): done (1625s)", None, true);
        assert!((assemble_done - 0.732).abs() < 1e-4);
        assert!(bake_running > assemble_done);
        assert!((bake_done - 0.917).abs() < 1e-4);
    }

    #[test]
    fn fresh_sequence_is_monotonic_through_stage1() {
        // raw frac (before the max_frac guard) must never regress across the stage-1 handoffs;
        // the guard only needs to absorb the out-of-order stage-3/4 bake markers, not stage 1.
        let seq: &[(&str, Option<f32>)] = &[
            ("[STAGE 1/9] extract dataset (geometry + textures)", Some(0.1)),
            ("[STAGE 1/9] extract dataset (geometry + textures)", Some(0.9)),
            ("[STAGE 1/9] extract dataset (geometry + textures): done (7200s)", None),
            ("[STAGE 1/9] extract grass density", None),
            ("[STAGE 1/9] extract grass density: done (300s)", None),
            ("[STAGE 1/9] extract physics colliders", Some(0.2)),
            ("[STAGE 1/9] extract physics colliders", Some(0.9)),
            ("[STAGE 1/9] extract physics colliders: done (2400s)", None),
            ("[STAGE 4/9] assemble pack", None),
        ];
        let mut prev = 0.0;
        for (stage, sub) in seq {
            let f = build_frac(stage, *sub, true);
            assert!(f >= prev, "regressed at {stage}: {f} < {prev}");
            prev = f;
        }
    }

    #[test]
    fn deps_and_unknown_pipelines_unchanged() {
        let f = build_frac("[STAGE 2/3] pip install: done (30s)", None, false);
        assert!((f - 0.98).abs() < 1e-4);
        let f = build_frac("[STAGE 2/5] some future pipeline", None, false);
        assert!((f - 0.3).abs() < 1e-4); // linear fallback
    }
}
