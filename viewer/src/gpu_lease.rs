//! Interactive-GPU LEASE — keeps the heavy compute WORKER off the GPU while the viewer is
//! rendering on it.
//!
//! WHY THIS EXISTS. The SH bake already runs as a separate worker process (`atlas bake-sh`,
//! spawned detached by the build pipeline) — that is the right shape: a fault in the worker
//! can't corrupt the viewer's address space. But process isolation does NOT isolate the *device*.
//! Windows TDR gives a GPU operation ~2 seconds; miss it and the driver resets the adapter, and
//! every context on it is lost — including the viewer's. That is exactly what happened: the bake
//! saturated the GPU while the viewer rendered, the adapter reset, the viewer's wgpu device was
//! lost, and (with `panic = "abort"`) wgpu's internal panic aborted the process (0xC0000409),
//! taking its child build down with it.
//!
//! Two defences, both applied:
//!   1. Keep every dispatch well under the TDR budget — `sh_bake_gpu::run_batched` sizes batches
//!      adaptively from the measured wall time (and its FIRST batch from BVH depth).
//!   2. THIS: never do heavy GPU compute *concurrently* with interactive rendering. The viewer
//!      holds a lease for its whole lifetime; the worker checks it and quietly uses the CPU
//!      backend instead (a few minutes even on the largest map — see docs/DOORS.md's sibling
//!      note in the bake logs). A CLI bake with no viewer running keeps the fast GPU path.
//!
//! HOW THE LEASE IS HELD (crash-safe, no stale state). The holder keeps a file OPEN with
//! `share_mode(0)` — Windows refuses any other open while that handle lives, and the OS closes
//! the handle when the process dies *by any means*, including an abort or a kill. So there is no
//! pid file to go stale and nothing to clean up after a crash. On non-Windows this degrades to
//! "never busy": the whole failure mode is the Windows TDR path, and a false "free" only costs
//! the old behaviour.

use std::fs::File;
use std::path::PathBuf;

/// The lease file. This must NOT live under `packs_root()`: that root deliberately depends on
/// executable layout and the launcher's current working directory, so two copies of the same
/// viewer could resolve different lease files and both believe they owned the one physical GPU.
/// Keep it in one per-user runtime directory instead. Every Atlas viewer/baker on the account now
/// contends on the same lock regardless of whether it was launched from Explorer, PowerShell, a
/// release bundle, or the repo.
fn lease_path() -> PathBuf {
    let root = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("APPDATA"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("atlas");
    if let Err(e) = std::fs::create_dir_all(&root) {
        eprintln!(
            "[gpu-lease] could not create runtime directory {}: {e}",
            root.display()
        );
    }
    root.join("gpu-interactive.lease")
}

/// Take the interactive-GPU lease. Hold the returned handle for as long as the process renders —
/// dropping it (or dying) releases it. `None` means we could not take it, which is NOT fatal:
/// the viewer renders regardless, it just can't advertise the lease.
#[cfg(windows)]
pub fn acquire() -> Option<File> {
    use std::os::windows::fs::OpenOptionsExt;
    let p = lease_path();
    match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .share_mode(0) // exclusive: any other open fails while this handle lives
        .open(&p)
    {
        Ok(f) => {
            eprintln!("[gpu-lease] ACQUIRED {}", p.display());
            Some(f)
        }
        Err(e) => {
            // Not fatal (we still render), but it means a bake worker won't see us and may
            // contend for the adapter -- so say so loudly rather than failing silently.
            eprintln!(
                "[gpu-lease] FAILED to acquire {}: {e} (kind={:?}, os={:?})",
                p.display(),
                e.kind(),
                e.raw_os_error()
            );
            None
        }
    }
}

#[cfg(not(windows))]
pub fn acquire() -> Option<File> {
    None
}

/// Process-wide lease handle. `OnceLock` so the first `hold()` wins and the file stays open for
/// the rest of the process — the OS releases it on exit by any means, including an abort.
static HELD: std::sync::OnceLock<Option<File>> = std::sync::OnceLock::new();

/// Take the lease if we haven't already. Idempotent, so every entry point that starts RENDERING A
/// MAP can call it without coordinating.
///
/// Why this exists rather than one `acquire()` at startup: the menu and the viewer are the SAME
/// process, and the lease used to be taken unconditionally for its whole lifetime. So a build
/// launched from the menu always found the GPU "busy" — held by an idle settings screen — and every
/// such bake silently took the CPU backend. Interchange cost 6m34s of CPU that way. The lease is
/// meant to protect an interactive MAP view from a TDR, and a menu is not that; it is now taken
/// when a map actually loads (`hold`), including the in-place PLAY switch out of the menu.
pub fn hold(reason: &str) -> bool {
    let first = HELD.get().is_none();
    let _ = HELD.set(acquire());
    let held = HELD.get().map(|o| o.is_some()).unwrap_or(false);
    if first {
        eprintln!("[gpu-lease] holding = {held} ({reason})");
    }
    held
}

/// Is an interactive viewer currently holding the GPU? Called by the bake worker to decide
/// between the GPU and CPU backends.
#[cfg(windows)]
pub fn busy() -> bool {
    use std::os::windows::fs::OpenOptionsExt;
    let p = lease_path();
    if !p.exists() {
        eprintln!("[gpu-lease] no lease file at {} -> GPU is free", p.display());
        return false;
    }
    // If we can open it exclusively, nobody holds it. ANY failure means we could not prove the
    // GPU is free, so we treat it as held: the cost of a false "busy" is a slower CPU bake, the
    // cost of a false "free" is a TDR that kills the viewer. Fail SAFE, not fast.
    match std::fs::OpenOptions::new().write(true).share_mode(0).open(&p) {
        Ok(_) => {
            eprintln!("[gpu-lease] {} opened exclusively -> no viewer holds the GPU", p.display());
            false
        }
        Err(e) => {
            eprintln!(
                "[gpu-lease] {} is HELD ({e}; kind={:?}, os={:?}) -> a viewer owns the GPU",
                p.display(),
                e.kind(),
                e.raw_os_error()
            );
            true
        }
    }
}

#[cfg(not(windows))]
pub fn busy() -> bool {
    false
}
