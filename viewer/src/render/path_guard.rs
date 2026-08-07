//! RENDER-PATH CRASH SENTINEL — how Atlas decides a render path is unsafe on THIS machine.
//!
//! The thing it replaces was a driver-string blacklist: `driver_info.contains("LLPC")` forced every
//! matching adapter onto the Standard path. That was wrong twice over.
//!
//! It was wrong about the FACT. "(LLPC)" is not an exotic ICD — it is what the ordinary AMD Radeon
//! Adrenalin driver reports in `driverInfo` on Windows (Adrenalin 25.11.1 reports "25.11.1 (LLPC)";
//! the RX 9070 XT in issue #9 reports "26.7.1 (LLPC)"). The comment justifying the rule described
//! it as an "AMDVLK-lineage ICD, not the standard Adrenalin compiler", but AMDVLK is a discontinued,
//! Linux-oriented project. So the rule did not catch a rare quirk: it caught EVERY AMD user on
//! Windows and sent all of them to the slowest renderer.
//!
//! It was wrong about the CAUSE. The device loss it was written for (RX 7800 XT, 2026-07-25) was
//! root-caused two days later in f6a5b0c: `gpu_shadow.wgsl` read the 192-byte material table at a
//! 176-byte stride, so every material index past 0 decoded garbage and an unclamped `albedo_index`
//! reached a `binding_array` out of range. AMD faults on that; NVIDIA returns zeros, which is why
//! the dev box never reproduced it. The stride is now pinned in both shaders and asserted by
//! `material_stride_tests`, and bindless indices are clamped at every use. The blacklist outlived
//! the bug by a release and nobody noticed, because the people it hurt could not tell the
//! difference between "slow" and "normal".
//!
//! WHY A FILE, AND NOT A `catch`. `wgpu-core`'s `handle_error_fatal` is `-> !` and panics directly,
//! and `Cargo.toml` sets `panic = "abort"`. A lost device therefore cannot be caught in-process --
//! there is no unwind, no handler, no chance to downgrade and retry. The only thing that outlives an
//! abort is something already on disk. So: write a marker naming the path about to be attempted;
//! delete it once that path has demonstrably worked. A marker found at startup means the last run
//! died while attempting it, and the next rung down is used instead.
//!
//! This is the same shape as a browser's GPU crash counter, and it is strictly better than a
//! blacklist for the property we actually want: it is vendor-agnostic (an NVIDIA or Intel driver
//! that starts faulting is caught too, and no such rule exists today), it costs nothing when
//! nothing is wrong, and it SELF-HEALS -- when a driver update fixes the fault, the marker simply
//! stops appearing and the user is back on the fast path with no code change and no new release.
//! A driver-string rule can only ever be relaxed by shipping a new binary.
//!
//! FALSE POSITIVES ARE THE REAL RISK, so the marker is cleared on two independent signals: a clean
//! exit, and sustained successful rendering with a pack loaded. Either one is proof the dangerous
//! window was survived. A `kill -9` mid-session is covered by the second, a quit from the menu by
//! the first. The cost of a miss is one launch on a slower-but-working renderer, and the user is
//! told exactly why in a line they can act on.

use bevy::prelude::*;
use std::path::PathBuf;

use super::RenderPath;

/// Frames of successful rendering, with a pack loaded, that count as "this path works here".
///
/// The field crashes died on frame 1 of the gpu-driven pipelines (44 ms and 49 ms in two of the
/// seven captured sessions), all of them AFTER "GPU buffers + bind groups built". So the danger
/// window opens when a pack starts drawing, not at device creation, and a marker cleared at
/// startup would clear before the risk. 120 frames is far past frame 1 while still only a couple
/// of seconds of normal play -- and it is deliberately counted in FRAMES, not seconds, because the
/// overlay idles at 2 fps while the game has focus and a wall-clock rule would fire there without
/// having drawn anything.
const FRAMES_TO_TRUST: u32 = 120;

/// `%APPDATA%\atlas\render-path.attempt`. `None` only when even the user-data root cannot be
/// resolved, in which case the guard disables itself rather than guessing at a writable location.
fn marker_path() -> Option<PathBuf> {
    crate::paths::user_data_dir_pub().map(|d| d.join("render-path.attempt"))
}

fn token(p: RenderPath) -> &'static str {
    match p {
        RenderPath::GpuDriven => "gpu",
        RenderPath::Standard => "std",
        RenderPath::M0Instanced => "m0",
    }
}

fn from_token(s: &str) -> Option<RenderPath> {
    match s.trim() {
        "gpu" => Some(RenderPath::GpuDriven),
        "std" => Some(RenderPath::Standard),
        "m0" => Some(RenderPath::M0Instanced),
        _ => None,
    }
}

/// The next path to try after `p` has proven unsafe here.
///
/// Standard is the ONLY automatic fallback. M0 used to be the last rung, and both times it was
/// reached in the field the reaction was the same: the map is white (M0 is untextured BY DESIGN,
/// a first-pixel dev path -- `instancing.rs` scope note), so the user reads the rescue as the
/// breakage. A fallback that looks broken is not a fallback. Standard keeps textures; if Standard
/// itself is dying, staying on it and failing visibly beats a white map that looks like success.
/// M0 remains reachable explicitly via EFT_RENDER=m0.
fn downgrade(p: RenderPath) -> Option<RenderPath> {
    match p {
        RenderPath::GpuDriven => Some(RenderPath::Standard),
        RenderPath::Standard | RenderPath::M0Instanced => None,
    }
}

/// Every path left unproven by a previous run, newest last.
fn unproven() -> Vec<RenderPath> {
    let Some(path) = marker_path() else { return Vec::new() };
    let Ok(raw) = std::fs::read_to_string(&path) else { return Vec::new() };
    raw.lines().filter_map(from_token).collect()
}

/// Choose the path to actually use, skipping any that a previous run died attempting.
///
/// The marker holds a SET, not a single token, and that distinction is load-bearing. With one
/// token the file cannot separate "this path crashed" from "this is merely the path we happened to
/// be running", so any abrupt kill demoted the user one rung -- including a force-kill from the
/// menu, which would drop somebody to the untextured M0 path over a crash that never happened.
/// Worse, a single token cannot chain honestly: recovering by walking down from whatever token was
/// last written can alternate between two rungs forever instead of converging.
///
/// Walking DOWN FROM THE PROBE'S CHOICE and skipping known-bad rungs fixes both. It terminates
/// (the rung list is finite and strictly descending), it never upgrades past what the adapter can
/// actually support (the walk starts at the probe's answer, so a card lacking the bindless
/// features is still held at M0), and an unrelated stale entry cannot drag a working path down.
///
/// `explicit` (the user passed `EFT_RENDER=` or a CLI token) suppresses this entirely -- when
/// someone has forced a path, silently running a different one is worse than crashing, because
/// they have no way to tell which of the two they are looking at.
pub fn resolve_after_crash(chosen: RenderPath, explicit: bool) -> RenderPath {
    let bad = unproven();
    if bad.is_empty() {
        return chosen;
    }
    if explicit {
        eprintln!(
            "render path: a previous run died attempting {bad:?}, but EFT_RENDER forces \
             {chosen:?} -- honouring the override and not downgrading."
        );
        return chosen;
    }
    let mut candidate = chosen;
    while bad.contains(&candidate) {
        match downgrade(candidate) {
            Some(next) => candidate = next,
            // Everything from here down has already failed. Stay put: there is nothing safer
            // left, and refusing to start at all would be worse than trying the last rung again.
            None => break,
        }
    }
    if candidate != chosen {
        eprintln!(
            "render path: a previous run died while starting the {chosen:?} path, so this one \
             uses {candidate:?} instead. This is automatic and temporary -- {chosen:?} is retried \
             after any launch that works, so a driver update restores it with no action from you. \
             Force a path with EFT_RENDER=gpu|std|m0."
        );
    }
    candidate
}

/// Record that `p` is being attempted, keeping any earlier unproven entries.
///
/// Must be called BEFORE the device is created: the whole point is to survive an abort during that
/// attempt. Appending rather than overwriting is what lets repeated failures accumulate into the
/// known-bad set instead of each run forgetting the last.
pub fn mark_attempt(p: RenderPath) {
    let Some(path) = marker_path() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut set = unproven();
    if !set.contains(&p) {
        set.push(p);
    }
    let body: Vec<&str> = set.into_iter().map(token).collect();
    let _ = std::fs::write(&path, body.join("\n"));
}

/// Clear the marker: this path has demonstrably worked here.
pub fn clear_marker() {
    if let Some(path) = marker_path() {
        let _ = std::fs::remove_file(path);
    }
}

/// Frames drawn with a pack loaded, and whether the marker has already been cleared.
#[derive(Resource, Default)]
struct TrustCounter {
    frames: u32,
    cleared: bool,
}

/// Clear the marker once the risky window has been survived, or on a clean exit.
///
/// Both signals are needed. Exit alone would leave a force-killed session looking like a crash;
/// frames alone would leave a menu-only session (never loads a pack, so never counts a frame)
/// looking like one too. Together they mean the marker survives only an actual abort.
fn trust_current_path(
    mut c: ResMut<TrustCounter>,
    pack: Option<Res<super::LoadedPack>>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut exits: MessageReader<bevy::app::AppExit>,
) {
    if c.cleared {
        return;
    }
    if exits.read().next().is_some() {
        c.cleared = true;
        clear_marker();
        return;
    }
    // Frames count in a MAP session only once the pack is loaded (the danger window this sentinel
    // exists for opens when the gpu-driven pipelines first draw a pack, so clearing earlier would
    // blind it) -- but a MENU session has no pack and never will, so for it the frames alone are
    // the proof. Without the menu arm, every force-killed menu session left a permanent false
    // "crash" marker, and the next map launch was downgraded for it: the field symptom was a user
    // launching interchange and getting the untextured M0 path with nothing wrong on their machine.
    if pack.is_none() && menu.is_none() {
        return;
    }
    c.frames += 1;
    if c.frames >= FRAMES_TO_TRUST {
        c.cleared = true;
        clear_marker();
    }
}

pub struct RenderPathGuardPlugin;

impl Plugin for RenderPathGuardPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TrustCounter>().add_systems(Update, trust_current_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_round_trip() {
        for p in [RenderPath::GpuDriven, RenderPath::Standard, RenderPath::M0Instanced] {
            assert_eq!(from_token(token(p)), Some(p), "{p:?} must survive a write/read cycle");
        }
    }

    #[test]
    fn downgrade_terminates() {
        // Every rung must reach the end; a cycle here would loop a crashing user forever.
        let mut p = Some(RenderPath::GpuDriven);
        let mut steps = 0;
        while let Some(cur) = p {
            p = downgrade(cur);
            steps += 1;
            assert!(steps <= 2, "downgrade chain does not terminate");
        }
        assert_eq!(steps, 2, "gpu -> std -> done");
    }

    #[test]
    fn gpu_falls_back_to_standard_not_m0() {
        // Standard keeps textures; M0 is untextured by design. A user recovering from a crash
        // should not also lose their materials.
        assert_eq!(downgrade(RenderPath::GpuDriven), Some(RenderPath::Standard));
    }

    #[test]
    fn unknown_marker_contents_are_ignored() {
        // A truncated or hand-edited marker must not be read as a path.
        assert_eq!(from_token(""), None);
        assert_eq!(from_token("vulkan"), None);
    }

    /// The selection rule, extracted so it can be tested without touching the filesystem.
    fn pick(chosen: RenderPath, bad: &[RenderPath]) -> RenderPath {
        let mut c = chosen;
        while bad.contains(&c) {
            match downgrade(c) {
                Some(n) => c = n,
                None => break,
            }
        }
        c
    }

    #[test]
    fn a_crash_on_the_fast_path_steps_down_one_rung() {
        assert_eq!(pick(RenderPath::GpuDriven, &[RenderPath::GpuDriven]), RenderPath::Standard);
    }

    #[test]
    fn repeated_crashes_stay_on_standard_never_the_white_map() {
        // Standard failing too means stay and fail VISIBLY: an auto-selected M0 is an untextured
        // map that reads as breakage (field-reported twice as "launched with no color").
        let bad = [RenderPath::GpuDriven, RenderPath::Standard];
        assert_eq!(pick(RenderPath::GpuDriven, &bad), RenderPath::Standard);
    }

    #[test]
    fn a_stale_entry_for_another_path_does_not_demote_a_working_one() {
        // The false positive the single-token version had: a force-kill while on Standard must
        // not cost a user the GPU-driven path, which never failed.
        assert_eq!(pick(RenderPath::GpuDriven, &[RenderPath::Standard]), RenderPath::GpuDriven);
    }

    #[test]
    fn the_probe_ceiling_is_never_exceeded() {
        // An adapter the probe held at M0 (missing bindless/indirect features) must stay there.
        // The walk only ever descends, so a stale entry can never promote it to a path the
        // hardware cannot run.
        assert_eq!(pick(RenderPath::M0Instanced, &[RenderPath::GpuDriven]), RenderPath::M0Instanced);
    }

    #[test]
    fn everything_failing_still_yields_a_path() {
        // Refusing to start would be worse than retrying the last rung.
        let all = [RenderPath::GpuDriven, RenderPath::Standard, RenderPath::M0Instanced];
        assert_eq!(pick(RenderPath::GpuDriven, &all), RenderPath::Standard);
    }
}
