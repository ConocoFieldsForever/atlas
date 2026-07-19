//! Build script. Two jobs:
//!   1. Capture the git short SHA into `ATLAS_GIT_SHA` so the app can build its release tag
//!      (`v<cargo_version>-<sha>`) and compare it against the newest GitHub release (see
//!      `update.rs`). Falls back to "unknown" (which disables the update check) when git is
//!      unavailable / this isn't a checkout.
//!   2. On Windows, embed the app icon (`resources/atlas.ico`) into the .exe so it shows in
//!      Explorer, the taskbar, and a desktop shortcut. A no-op on every other target.
use std::process::Command;

fn main() {
    // --- 1. git short SHA -> ATLAS_GIT_SHA -----------------------------------------------------
    let sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=ATLAS_GIT_SHA={sha}");
    // Rebuild when HEAD moves so the embedded SHA stays current. `--git-path` resolves the real
    // HEAD path relative to this crate dir (handles worktrees + the .git living at the repo root,
    // one level up from viewer/). Best-effort: skip the directive if git can't tell us.
    if let Some(head) = Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        println!("cargo:rerun-if-changed={head}");
    }

    // --- 2. Windows icon embed -----------------------------------------------------------------
    println!("cargo:rerun-if-changed=resources/atlas.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("resources/atlas.ico");
        if let Err(e) = res.compile() {
            // Never fail the whole build over the icon: if the resource compiler is unavailable the
            // exe simply ships without an embedded icon (the runtime window icon still applies).
            println!("cargo:warning=atlas icon embed skipped: {e}");
        }
    }
}
