//! Menu-only GitHub update check. On the START MENU we fire ONE token-less GET against the public
//! `ConocoFieldsForever/atlas` releases API, off the main thread, and compare the newest release's
//! `tag_name` to this build's tag. If they differ we surface a themed "update available" modal
//! (menu.rs). Everything here is best-effort and offline-safe:
//!   * never blocks the menu from rendering (a std::thread does the blocking I/O; the main thread
//!     polls a shared slot each frame),
//!   * any error / timeout / non-2xx / unparseable body => `Unknown` (no modal, no nag, no spam),
//!   * a build with no embedded git SHA (`ATLAS_GIT_SHA == "unknown"`, e.g. a tarball build) skips
//!     the check entirely,
//!   * panic=abort: NO unwrap/expect on the network or JSON path — every failure folds to `Unknown`.
//! No token is ever sent (the repo is public) and nothing is reported back — it's one read.

use bevy::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// This build's release tag, in the exact format the release workflow tags with:
/// `v<cargo_version>-<git_short_sha>` (e.g. `v0.1.0-15061f1`). The SHA is embedded by build.rs.
pub const APP_TAG: &str = concat!("v", env!("CARGO_PKG_VERSION"), "-", env!("ATLAS_GIT_SHA"));

/// This build's HEAD commit date (ISO 8601 UTC, e.g. `2026-07-20T13:00:00Z`), embedded by build.rs.
/// Used to recency-gate the update prompt: if this build is NEWER than the latest release's publish
/// time, it's ahead of every release (a local dev build) and must not be nagged. `"unknown"` on a
/// build with no git (then the gate is skipped and the plain tag-inequality decides, as before).
pub const APP_BUILD_TIME: &str = env!("ATLAS_BUILD_TIME");

/// The newest release, per GitHub. NOT `/releases/latest` — every atlas release is a pre-release,
/// which that endpoint hides; the plain list (newest first) surfaces pre-releases too.
const RELEASES_URL: &str =
    "https://api.github.com/repos/ConocoFieldsForever/atlas/releases?per_page=1";

/// Result of the update check. Read by the menu UI; default `Unknown`.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub enum UpdateStatus {
    /// Not checked yet, or the check failed / was skipped. No modal, no indicator badge.
    #[default]
    Unknown,
    /// The newest published release matches this build.
    UpToDate,
    /// A newer release exists — `tag` is its `tag_name`, `url` its `html_url` (opened on UPDATE).
    Available { tag: String, url: String },
}

impl UpdateStatus {
    /// Short label for the one-line "update check: <status>" info log.
    fn log_label(&self) -> String {
        match self {
            UpdateStatus::Unknown => "unknown".to_string(),
            UpdateStatus::UpToDate => "up to date".to_string(),
            UpdateStatus::Available { tag, .. } => format!("available ({tag})"),
        }
    }
}

/// Async plumbing + session UI state for the check. The background thread writes the result ONCE
/// into `slot`; `poll_update` drains it into the `UpdateStatus` resource (and logs it once).
#[derive(Resource)]
pub struct UpdateCheck {
    /// Filled once by the worker thread; `None` until the check finishes. `Arc<Mutex<..>>` (not an
    /// mpsc Receiver, which is `!Sync` and so can't live in a Bevy resource).
    slot: Arc<Mutex<Option<UpdateStatus>>>,
    /// LATER pressed this session -> suppress the modal (the top-right indicator stays).
    pub dismissed: bool,
}

impl UpdateCheck {
    /// Kick off the check (once). Spawns the network thread unless this build has no embedded SHA,
    /// in which case the result is `Unknown` immediately (dev/tarball build — nothing to compare).
    fn start() -> Self {
        let slot: Arc<Mutex<Option<UpdateStatus>>> = Arc::new(Mutex::new(None));
        if env!("ATLAS_GIT_SHA") == "unknown" {
            info!("update check: skipped (no embedded git sha)");
            if let Ok(mut g) = slot.lock() {
                *g = Some(UpdateStatus::Unknown);
            }
        } else {
            let out = slot.clone();
            // Detached: the menu renders immediately; this blocks only its own thread.
            std::thread::spawn(move || {
                let status = check_latest();
                if let Ok(mut g) = out.lock() {
                    *g = Some(status);
                }
            });
        }
        Self { slot, dismissed: false }
    }
}

impl Default for UpdateCheck {
    fn default() -> Self {
        Self::start()
    }
}

/// Perform the blocking GET + compare. Returns `Unknown` on ANY failure (offline, timeout, non-2xx,
/// malformed JSON, missing fields). Never panics.
fn check_latest() -> UpdateStatus {
    let resp = match ureq::get(RELEASES_URL)
        // GitHub rejects requests with no UA; `Accept` pins the stable v3 JSON shape.
        .set("User-Agent", "atlas")
        .set("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(6))
        .call()
    {
        Ok(r) => r,
        Err(_) => return UpdateStatus::Unknown, // offline / DNS / TLS / non-2xx: stay quiet
    };
    let body = match resp.into_string() {
        Ok(b) => b,
        Err(_) => return UpdateStatus::Unknown,
    };
    classify(&body, APP_TAG, APP_BUILD_TIME)
}

/// Pure compare: given the raw releases-list JSON body, this build's tag, and this build's commit
/// time, decide the status. GitHub returns the list newest-first, so `first` is the newest release.
/// Same tag => up to date. Different tag => an update EXISTS, but we only prompt if the release is
/// actually newer than this build: if this build's commit time is later than the release's
/// `published_at`, we're a dev build AHEAD of every release and stay quiet (that's the "always shows
/// the modal on a locally-built binary" fix). ISO-8601 UTC timestamps sort lexicographically ==
/// chronologically, so a plain string compare is exact — no date parsing. Unit-testable, no network.
fn classify(body: &str, app_tag: &str, build_time: &str) -> UpdateStatus {
    let val: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return UpdateStatus::Unknown,
    };
    let first = match val.as_array().and_then(|a| a.first()) {
        Some(f) => f,
        None => return UpdateStatus::Unknown, // no releases yet
    };
    let tag = match first.get("tag_name").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return UpdateStatus::Unknown,
    };
    if tag == app_tag {
        return UpdateStatus::UpToDate;
    }
    // Recency gate: don't nag a build that is AHEAD of the newest release (a local dev build whose
    // SHA differs but whose commit is newer than when that release was published). Skipped when the
    // build time is unknown or the release omits `published_at` — then the tag inequality decides.
    if build_time != "unknown" {
        if let Some(published_at) = first.get("published_at").and_then(|v| v.as_str()) {
            if build_time > published_at {
                return UpdateStatus::UpToDate;
            }
        }
    }
    // A genuinely newer release: carry its browser URL (fall back to the repo releases page if absent).
    let url = first
        .get("html_url")
        .and_then(|v| v.as_str())
        .unwrap_or("https://github.com/ConocoFieldsForever/atlas/releases")
        .to_string();
    UpdateStatus::Available { tag: tag.to_string(), url }
}

/// Drain the worker's result into the `UpdateStatus` resource exactly once, logging the outcome.
/// Cheap no-op every frame after (the slot is emptied on the first successful drain).
fn poll_update(check: Res<UpdateCheck>, mut status: ResMut<UpdateStatus>) {
    if let Ok(mut slot) = check.slot.lock() {
        if let Some(new) = slot.take() {
            info!("update check: {}", new.log_label());
            *status = new;
        }
    }
}

/// Open a URL in the user's default browser. Best-effort; never panics if it fails.
pub fn open_url(url: &str) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // `cmd /C start "" <url>` — the empty "" is the (ignored) window title so a url with spaces
        // isn't mistaken for one. CREATE_NO_WINDOW (0x0800_0000) keeps a console from flashing.
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .creation_flags(0x0800_0000)
            .spawn();
    }
    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

/// Menu-only: register the update-check resources + the poll system. Added exclusively in menu
/// mode (main.rs), so the check never runs in-raid.
pub struct UpdatePlugin;

impl Plugin for UpdatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UpdateStatus>() // default Unknown
            .init_resource::<UpdateCheck>() // Default::default() spawns the check thread once
            .add_systems(Update, poll_update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: &str = "2026-01-01T00:00:00Z"; // an early build time (before the test releases)

    // A different tag with NO published_at (recency gate can't apply) => Available, carrying the url.
    #[test]
    fn stale_tag_flips_to_available() {
        let body = r#"[{"tag_name":"v9.9.9-deadbee","html_url":"https://example.com/r","prerelease":true}]"#;
        match classify(body, "v0.1.0-15061f1", T0) {
            UpdateStatus::Available { tag, url } => {
                assert_eq!(tag, "v9.9.9-deadbee");
                assert_eq!(url, "https://example.com/r");
            }
            other => panic!("expected Available, got {other:?}"),
        }
    }

    // Same tag => UpToDate (no modal).
    #[test]
    fn matching_tag_is_up_to_date() {
        let body = r#"[{"tag_name":"v0.1.0-15061f1","html_url":"https://example.com/r"}]"#;
        assert_eq!(classify(body, "v0.1.0-15061f1", T0), UpdateStatus::UpToDate);
    }

    // A build whose commit is NEWER than the release's publish time is AHEAD of it => UpToDate,
    // even though the tags differ. (The "always shows the modal on a local dev build" fix.)
    #[test]
    fn build_ahead_of_release_is_up_to_date() {
        let body = r#"[{"tag_name":"v0.1.0-old1234","published_at":"2026-07-17T13:58:18Z"}]"#;
        assert_eq!(
            classify(body, "v0.1.0-new5678", "2026-07-20T09:00:00Z"),
            UpdateStatus::UpToDate
        );
    }

    // A release published AFTER this build => genuinely newer => Available.
    #[test]
    fn newer_release_prompts_available() {
        let body = r#"[{"tag_name":"v0.2.0-abc","published_at":"2026-08-01T00:00:00Z","html_url":"https://example.com/r"}]"#;
        match classify(body, "v0.1.0-new5678", "2026-07-20T09:00:00Z") {
            UpdateStatus::Available { tag, .. } => assert_eq!(tag, "v0.2.0-abc"),
            other => panic!("expected Available, got {other:?}"),
        }
    }

    // Unknown build time (tarball build) can't recency-gate => plain tag inequality decides (Available).
    #[test]
    fn unknown_build_time_keeps_tag_inequality() {
        let body = r#"[{"tag_name":"v0.1.0-old1234","published_at":"2026-07-17T13:58:18Z"}]"#;
        match classify(body, "v0.1.0-new5678", "unknown") {
            UpdateStatus::Available { .. } => {}
            other => panic!("expected Available, got {other:?}"),
        }
    }

    // Offline-safe: garbage / empty / missing-field bodies never panic and fold to Unknown.
    #[test]
    fn malformed_bodies_are_unknown() {
        assert_eq!(classify("not json", "v0.1.0-x", T0), UpdateStatus::Unknown);
        assert_eq!(classify("[]", "v0.1.0-x", T0), UpdateStatus::Unknown);
        assert_eq!(classify(r#"[{"prerelease":true}]"#, "v0.1.0-x", T0), UpdateStatus::Unknown);
    }

    // Available with no html_url still resolves (falls back to the releases page).
    #[test]
    fn available_without_url_falls_back() {
        let body = r#"[{"tag_name":"v2.0.0-abc1234"}]"#;
        match classify(body, "v0.1.0-15061f1", T0) {
            UpdateStatus::Available { url, .. } => {
                assert!(url.contains("ConocoFieldsForever/atlas/releases"));
            }
            other => panic!("expected Available, got {other:?}"),
        }
    }
}
