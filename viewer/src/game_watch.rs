//! Live game link — TarkovMonitor-style passive file watching (no game hooks).
//!
//! Everything here reads files EFT itself writes; nothing touches the game process:
//!   * `<game>\Logs\log_*\*application_NNN.log`   — `scene preset path:maps/<bundle>.bundle` tells us
//!     which map is loading -> in-place map swap (MapSwitch) so Atlas follows the raid.
//!   * `<game>\Logs\log_*\*push-notifications_NNN.log` — task status push messages (started/failed/finished,
//!     `message.type` 10/11/12, task id = `message.templateId` before the space) -> auto-track /
//!     auto-complete in the Tasks tab; `UserMatchOver` -> clear the player marker.
//!   * `Documents\Escape From Tarkov\Screenshots` — EFT embeds the player's WORLD POSITION and
//!     rotation quaternion in every screenshot filename ("...]_x, y, z_qx, qy, qz, qw (0).png").
//!     Each new screenshot becomes a live player fix: marker + facing in the 3D world, and the
//!     pathfinder's "you are here" pin, so routes start from the player (press the screenshot key
//!     in raid to update). Only YOUR OWN position — same mechanism tarkov.dev's map page uses.
//!
//! A background thread polls (~0.7 s) and sends parsed events over an mpsc channel; Bevy systems
//! apply them. Coordinates bridge with the same X-flip the whole pipeline uses:
//! viewer = (-x, y, z). Disable entirely with `EFT_GAME_LINK=0`.
use bevy::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;

// ---------------------------------------------------------------------------------------------
// Events from the watcher thread
// ---------------------------------------------------------------------------------------------

#[cfg_attr(test, derive(Debug))]
enum GameEvent {
    /// A raid map started loading in the game (atlas map id, already bundle->id resolved).
    MapLoading(String),
    /// A new screenshot fix: viewer-space position, FLATTENED facing for the map marker, and the
    /// full 3-D view direction (pitch included) for standing in the player's eyes. Both None when
    /// the filename carries no quaternion.
    PlayerFix { pos: Vec3, fwd: Option<Vec3>, look: Option<Vec3>, game_hour: Option<f32> },
    /// Task status push: 10 = started, 11 = failed, 12 = finished.
    Task { id: String, status: i64 },
    /// The player's real in-game vertical FOV (application log, `Game settings:` JSON dump at
    /// boot and on every settings apply). Matching it makes the screenshot-eye view frame the
    /// world exactly like the game does.
    Fov(f32),
    /// A raid STARTED. Carries the wall-clock instant the `GameStarted` line was written, which
    /// is the only raid-start timestamp EFT puts on disk.
    RaidStart(std::time::SystemTime),
    /// The key the player has bound to MakeScreenshot, from the `Control settings` dump. The
    /// overlay is summoned by that key, so the hint must name the real one rather than a default
    /// the user may never have kept.
    ScreenshotKey(String),
    /// The raid ended — the last fix is stale.
    RaidEnd,
    /// The local profile's side for the upcoming/current raid. This comes only from
    /// GroupMatchRaidSettings; uppercase `Side` fields elsewhere describe other profiles.
    RaidSide(RaidSide),
    /// Atlas consumed a screenshot AND deleted it (delete_processed_shots). Reported so the app
    /// can tell the user ONCE that their file was removed: deleting someone's screenshots is not
    /// something a settings-tooltip they never opened counts as consent for.
    ShotDeleted(String),
}

/// Scene-preset bundle name -> our pack id. GAME-DERIVED via the embedded manifest
/// (`maps::bundle_to_id`): gen_maps reads each `maps/*.bundle`'s own scene list and joins its
/// location folder to the roster. Replaces the hardcoded TarkovMonitor MapBundles copy that
/// lived here — which had silently omitted `icebreaker.bundle`, so Icebreaker raids never
/// auto-detected. Bundles for locations we don't ship (terminal, arena) aren't in the roster.
pub(crate) fn bundle_to_map(bundle: &str) -> Option<&'static str> {
    crate::maps::bundle_to_id(bundle)
}

// ---------------------------------------------------------------------------------------------
// Bevy side
// ---------------------------------------------------------------------------------------------

/// "Screenshot to locate current position" — shared between the Bevy side (menu toggle) and the
/// watcher THREAD, which polls it every tick. Seeded from the persisted config at plugin build so
/// the thread honours the setting from the first poll.
static SCREENSHOT_LOCATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Read by the watcher thread each tick.
fn screenshot_locate() -> bool {
    SCREENSHOT_LOCATE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Flip the screenshot-locate setting live (the menu checkbox calls this the moment it changes;
/// no restart needed).
pub fn set_screenshot_locate(on: bool) {
    SCREENSHOT_LOCATE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Delete each screenshot once its position fix has been taken. Same live-flag shape.
static DELETE_SHOTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_delete_processed_shots(on: bool) {
    DELETE_SHOTS.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The latest live player fix, in viewer space.
pub struct PlayerFixState {
    pub pos: Vec3,
    /// Flattened heading (y = 0) — what the world marker draws. The full 3-D view direction
    /// (with pitch) is consumed straight off the event by the camera and not stored.
    pub fwd: Option<Vec3>,
    /// `Time::elapsed_secs` when the fix arrived (drives the marker pulse).
    pub at: f32,
    /// In-game time of day, decimal hours, from the trailing field of the screenshot filename
    /// (`..._14.54 (0).png`). It was always parsed and thrown away. EFT runs game time at 7x real
    /// time, so this is the only way to show the player the clock THEY are playing against;
    /// deriving it from the raid clock would be wrong by that factor.
    pub game_hour: Option<f32>,
}

/// Local raid side as named by EFT's `raidSettings.side` log field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaidSide {
    Pmc,
    Scav,
}

impl RaidSide {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pmc => "PMC",
            Self::Scav => "Scav",
        }
    }

    /// Component-type eligibility. `secret` remains visible for both sides because the
    /// SecretExfiltrationPoint class does not encode a faction.
    pub fn allows_extract(self, faction: &str) -> bool {
        let f = faction.to_ascii_lowercase();
        if f == "shared" || f == "secret" || f.contains('+') {
            return true;
        }
        match self {
            Self::Pmc => f == "pmc",
            Self::Scav => f == "scav" || f == "savage",
        }
    }
}

/// MANUAL raid-side choice, for planning when the game is not running.
///
/// Side filtering used to exist ONLY when the live link had parsed `GroupMatchRaidSettings`, i.e.
/// only while a raid was actually loading. At the desk — the primary planning case — a PMC saw
/// Scav-only extracts in "nearest extract" and in loot plans, and could be routed to an exit they
/// cannot use. The live value stays AUTHORITATIVE; this only fills the gap when it is absent.
#[derive(Resource, Default)]
pub struct SideChoice(pub Option<RaidSide>);

impl SideChoice {
    pub fn load() -> Self {
        Self(match crate::menu::config_str_pub("raidSide").as_deref() {
            Some("pmc") => Some(RaidSide::Pmc),
            Some("scav") => Some(RaidSide::Scav),
            _ => None,
        })
    }

    /// Persist; returns false if the config could not be written (caller surfaces it).
    pub fn save(&self) -> bool {
        crate::menu::save_config_str_pub(
            "raidSide",
            match self.0 {
                Some(RaidSide::Pmc) => "pmc",
                Some(RaidSide::Scav) => "scav",
                None => "",
            },
        )
    }
}

/// The side to filter by: the LIVE raid side when the logs know it, else the user's manual choice,
/// else None (show everything — never guess).
pub fn effective_side(
    link: Option<&GameLink>,
    choice: Option<&SideChoice>,
) -> Option<RaidSide> {
    link.and_then(|l| l.raid_side).or_else(|| choice.and_then(|c| c.0))
}

/// A running raid, from `|application|GameStarted:`.
///
/// EFT writes no raid countdown anywhere on disk, so this is derived: the log line's own timestamp
/// is the start, and the duration comes from the map's `raid_minutes` (the client's own
/// `EscapeTimeLimit`, already carried by `poi::MapIntelMeta`). `remaining()` is therefore only as
/// honest as that pair, which is why it returns `None` rather than guessing when the duration for
/// this map is unknown.
#[derive(Clone, Copy, Debug)]
pub struct RaidClock {
    /// When `GameStarted` was logged.
    pub started: std::time::SystemTime,
}

impl RaidClock {
    /// Wall-clock seconds since the raid began.
    pub fn elapsed_s(&self) -> f32 {
        self.started.elapsed().map(|d| d.as_secs_f32()).unwrap_or(0.0)
    }

    /// Seconds left, given the map's raid length. `None` when the duration is unknown: a countdown
    /// that invents its own end time is worse than no countdown, because it is the number a player
    /// would decide when to run on.
    pub fn remaining_s(&self, raid_minutes: Option<f32>) -> Option<f32> {
        raid_minutes.map(|m| (m * 60.0 - self.elapsed_s()).max(0.0))
    }
}

#[derive(Resource)]
pub struct GameLink {
    rx: Mutex<Receiver<GameEvent>>,
    pub player: Option<PlayerFixState>,
    /// The player's own MakeScreenshot binding, when the logs have named it.
    pub screenshot_key: Option<String>,
    /// The running raid, if the logs say one is in progress. Also the guard that makes
    /// `PrepareSelectedProfileLocally` mean "raid over": that line ALSO fires at login and on
    /// every return to the menu, so it is only an end signal while a clock is actually running.
    pub in_raid: Option<RaidClock>,
    /// The map the LOGS say the player is on, parsed but NOT yet applied. The switch is deferred
    /// until the user actually summons the overlay: yanking the viewer onto another map while
    /// they are reading a different one (or browsing the menu) is worse than being a beat late.
    pub pending_map: Option<String>,
    /// The raid's map has no built pack (map id). The HUD/overlay shows this — a silent no-op
    /// here is the worst outcome, because the user is standing in a raid watching a map that is
    /// NOT where they are.
    pub unbuilt_map: Option<String>,
    /// Authoritative when present. None means this session did not log GroupMatchRaidSettings;
    /// consumers must keep both factions visible instead of guessing.
    pub raid_side: Option<RaidSide>,
    /// The FIRST screenshot this session that Atlas consumed and deleted, until the user
    /// acknowledges it. Deleting a player's files is disclosed in a settings tooltip they have
    /// probably never opened, so say it once, on screen, with the way to turn it off.
    pub deleted_notice: Option<String>,
    /// Set once the notice has been shown+dismissed, so it never nags again this session.
    pub deleted_notice_done: bool,
    /// The raid is on a map whose pack is missing AND the user dismissed the full banner. A
    /// compact pill keeps saying so (ui::wrong_map_pill) — dismissing must not restore the silent
    /// wrong-map state the banner exists to break. Cleared when the raid ends.
    pub wrong_map: Option<String>,
}

pub struct GameWatchPlugin;

impl Plugin for GameWatchPlugin {
    fn build(&self, app: &mut App) {
        if std::env::var("EFT_GAME_LINK").is_ok_and(|v| v.trim() == "0") {
            info!("game link: disabled (EFT_GAME_LINK=0)");
            return;
        }
        // A scripted EFT_SHOT/EFT_BENCH run must not race the player's own session: the watcher
        // consumes + deletes their screenshots, and a live raid would queue a map swap into a run
        // that was told exactly which pack to measure. No watcher thread, no GameLink resource —
        // every consumer takes Option<GameLink> and stands down.
        if crate::automated_finite_job() {
            info!("game link: disabled (finite EFT_SHOT/EFT_BENCH job)");
            return;
        }
        // Seed the shared flag from the persisted menu setting before the thread starts, so a
        // user who turned screenshot-locate OFF never gets a single poll of the folder.
        set_screenshot_locate(crate::menu::config_screenshot_locate());
        set_delete_processed_shots(
            crate::menu::config_bool_pub("deleteProcessedShots").unwrap_or(true),
        );
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("eft-game-watch".into())
            .spawn(move || watcher_thread(tx))
            .ok();
        app.insert_resource(SideChoice::load());
        app.insert_resource(GameLink {
            rx: Mutex::new(rx),
            player: None,
            screenshot_key: None,
            in_raid: None,
            pending_map: None,
            unbuilt_map: None,
            raid_side: None,
            deleted_notice: None,
            deleted_notice_done: false,
            wrong_map: None,
        })
            .add_systems(
                Update,
                (apply_game_events, sync_map_on_overlay_show, draw_player_marker),
            );
    }
}

/// Drain the watcher channel and apply each event to the app's existing machinery: MapSwitch for
/// the in-place swap, StartPoint + the marker for the player fix, PlayerProgress for tasks.
fn apply_game_events(
    mut link: ResMut<GameLink>,
    mut start_pt: ResMut<crate::pathfind::StartPoint>,
    mut progress: ResMut<crate::progress::PlayerProgress>,
    catalog: Option<Res<crate::tasks_panel::TaskCatalog>>,
    route_result: Option<Res<crate::pathfind::RouteResult>>,
    // Reader + writer of the same message type conflict as bare params (B0002); a ParamSet
    // sequences the two accesses.
    mut routes: ParamSet<(
        MessageReader<crate::pathfind::RouteRequest>,
        MessageWriter<crate::pathfind::RouteRequest>,
    )>,
    mut last_route: Local<Option<crate::pathfind::RouteRequest>>,
    mut cam_cmd: ResMut<crate::CameraCommand>,
    overlay_cfg: Option<Res<crate::overlay::OverlayConfig>>,
    mut overlay_state: Option<ResMut<crate::overlay::OverlayState>>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut sw: ResMut<crate::MapSwitch>,
    mut cam_settings: ResMut<crate::CameraSettings>,
    mut toggles: ResMut<crate::ui::LayerToggles>,
    time: Res<Time>,
) {
    // Shadow-read every route request the UI sends (readers have independent cursors, so this does
    // not consume them): remember the latest real one so a new player fix can re-issue it from the
    // new position — live "route from me" without any UI change.
    for req in routes.p0().read() {
        if !req.dests.is_empty() {
            *last_route = Some(req.clone());
        } else {
            *last_route = None; // an explicit clear also stops re-routing
        }
    }

    let events: Vec<GameEvent> = match link.rx.lock() {
        Ok(rx) => rx.try_iter().collect(),
        Err(_) => return,
    };
    for ev in events {
        match ev {
            GameEvent::MapLoading(id) => {
                link.player = None; // a new raid invalidates the old fix
                // RECORD ONLY. `sync_map_on_overlay_show` applies it when the overlay is summoned.
                if link.pending_map.as_deref() != Some(id.as_str()) {
                    info!("game link: raid is on '{id}' (will load when the overlay is opened)");
                    link.pending_map = Some(id.clone());
                }
                continue;
            }
            GameEvent::PlayerFix { pos, fwd, look, game_hour } => {
                info!("game link: player fix at {:.1},{:.1},{:.1}", pos.x, pos.y, pos.z);
                link.player = Some(PlayerFixState { pos, fwd, at: time.elapsed_secs(), game_hour });
                // "Screenshot to locate current position": stand in the player's EYES. EFT bakes
                // the camera position + view quaternion into the filename, and both are already
                // bridged to viewer space, so this is a 1:1 pose -- no framing, no offset. Without
                // a facing (an older screenshot name with no quaternion) we keep the current look
                // direction and only move.
                // ONE KEYPRESS: the player pressed THEIR OWN in-game screenshot key, EFT wrote
                // the file, and we just turned it into a position. Summoning the overlay here is
                // what makes that single press mean "show me where I am" — no key interception,
                // no injected input, nothing touching the game (see OverlayConfig::show_on_screenshot).
                let summon =
                    overlay_cfg.as_ref().is_some_and(|c| c.enabled && c.show_on_screenshot);
                if menu.is_some() {
                    // At the START MENU a summon means: relaunch into the raid map (menu-mode
                    // MapSwitch IS the PLAY path — new process, menu torn down) with the pose and
                    // "come up shown" handed over via env vars the child inherits. The child
                    // consumes and REMOVES both (overlay.rs), so later relaunches stay clean.
                    if summon {
                        if let Some(id) = link.pending_map.clone() {
                            let dir = crate::paths::packs_root().join(format!("{id}.eftpack"));
                            if dir.join("manifest.json").is_file() {
                                if let Some(d) = look.or(fwd) {
                                    let yaw = (-d.x).atan2(-d.z).to_degrees();
                                    let pitch = d.y.asin().to_degrees();
                                    std::env::set_var(
                                        "EFT_POSE",
                                        format!("{},{},{},{yaw},{pitch}", pos.x, pos.y, pos.z),
                                    );
                                }
                                // The child watcher will rediscover this from the application log,
                                // but carry the already-known value across the relaunch so its first
                                // rendered map frame has Tarkov's exact projection.
                                std::env::set_var("EFT_GAME_FOV", cam_settings.fov_deg.to_string());
                                std::env::set_var("EFT_OVERLAY_SUMMON", "1");
                                info!(
                                    "game link: screenshot at the menu -> relaunching into '{id}' \
with the overlay up"
                                );
                                sw.0 = Some(dir.to_string_lossy().into_owned());
                            } else {
                                warn!(
                                    "game link: screenshot at the menu but '{id}' has no built pack"
                                );
                                link.unbuilt_map = Some(id);
                            }
                        }
                    }
                    continue; // no camera/marker work behind the start menu
                }
                if summon {
                    if let Some(st) = overlay_state.as_mut() {
                        // ALWAYS ask for a re-raise, even when already shown: the game reclaims the
                        // foreground (and on exclusive fullscreen can push us behind entirely)
                        // between screenshots, and a summon that only fires on the rising edge
                        // leaves the overlay flagged shown but invisible.
                        st.raise_nonce = st.raise_nonce.wrapping_add(1);
                        st.windowed = false;
                        if !st.shown {
                            st.shown = true;
                            info!("game link: screenshot fix -> summoning the overlay");
                        }
                    }
                }
                // A fresh screenshot fix is an explicit "locate me" — turn the player marker on
                // (it defaults OFF so stale fixes from past raids don't haunt the map).
                toggles.player_marker = true;
                if let Some(dir) = look.or(fwd) {
                    cam_cmd.eye = Some((pos, dir));
                } else {
                    // No quaternion in the name (older screenshot): move, keep looking as we were.
                    cam_cmd.eye = Some((pos, Vec3::ZERO));
                }
                // The pathfinder's "you are here" pin: every route (route-here / route tracked /
                // navigate tab) starts from it when set. Moving it clears any drawn route
                // (clear_route_on_start_move), so re-issue the last request from the new fix to
                // keep a live route following the player.
                start_pt.0 = Some(pos);
                if let (Some(req), Some(res)) = (last_route.as_ref(), route_result.as_ref()) {
                    use crate::pathfind::RouteStatus as RS;
                    if matches!(res.status, RS::Ok | RS::Pending) {
                        let mut req = req.clone();
                        req.start = Some(pos);
                        routes.p1().write(req);
                    }
                }
            }
            GameEvent::Task { id, status } => {
                match status {
                    10 => {
                        // Started -> auto-track: its markers appear on the map (QuestTracker.active
                        // mirrors progress.tracked each frame).
                        if progress.tracked.insert(id.clone()) {
                            info!("game link: task started - tracking {id}");
                        }
                    }
                    11 => {
                        progress.tracked.remove(&id);
                    }
                    12 => {
                        // Finished -> untrack + mark every objective done (per-objective keys, the
                        // same obj_key the Tasks tab checkboxes write).
                        progress.tracked.remove(&id);
                        if let Some(cat) = catalog.as_ref() {
                            if let Some(task) = cat.tasks.iter().find(|t| t.id == id) {
                                for (i, o) in task.objectives.iter().enumerate() {
                                    progress.done.insert(crate::tasks_panel::obj_key(&id, o, i));
                                }
                                info!("game link: task finished - {}", task.name);
                            }
                        }
                    }
                    _ => {}
                }
            }
            GameEvent::Fov(v) => {
                // Adopt the game's own FOV so the eye view frames like the game. Guarded so we
                // only dirty CameraSettings (and re-apply the projection) on a real change; the
                // camera tab can still override until the next Tarkov settings dump.
                if (cam_settings.fov_deg - v).abs() > 0.25 {
                    info!("game link: matching the game's FOV ({v} deg)");
                    cam_settings.fov_deg = v;
                }
            }
            GameEvent::ScreenshotKey(k) => {
                if link.screenshot_key.as_deref() != Some(k.as_str()) {
                    info!("game link: screenshot key is {k}");
                    link.screenshot_key = Some(k);
                }
            }
            GameEvent::RaidStart(at) => {
                if link.in_raid.is_none() {
                    info!("game link: raid started");
                }
                link.in_raid = Some(RaidClock { started: at });
            }
            GameEvent::RaidEnd => {
                // Reached from `PrepareSelectedProfileLocally` while a clock runs. UserMatchOver,
                // which used to be the ONLY trigger, appears in 0 of 310 log folders on this
                // machine -- so everything below had effectively stopped being cleared, and the
                // stale-map failure the comment describes was live rather than guarded against.
                link.in_raid = None;
                link.unbuilt_map = None;
                link.player = None;
                link.raid_side = None;
                // The deferred swap target dies with the raid too. Leaving it set meant a `~`
                // press or any parseable screenshot HOURS later still loaded that raid's map as
                // if it were live — and at the start menu it relaunched the whole process into
                // it. "Pending" only ever meant "the raid the logs say you are in right now".
                link.pending_map = None;
                link.wrong_map = None;
            }
            GameEvent::RaidSide(side) => {
                if link.raid_side != Some(side) {
                    info!("game link: raid side is {} (GroupMatchRaidSettings)", side.label());
                    link.raid_side = Some(side);
                }
            }
            GameEvent::ShotDeleted(name) => {
                // Once per session, and only until acknowledged.
                if !link.deleted_notice_done && link.deleted_notice.is_none() {
                    link.deleted_notice = Some(name);
                }
            }
        }
    }
}

/// Draw the live player marker: pulsing ground ring + facing arrow + a vertical beacon so the
/// player is findable from any camera height. Gizmos = immediate mode, nothing to clean up.
/// Gated by the panel's "Player marker (game link)" toggle (default OFF — a fix can outlive its
/// raid, and a stale green beacon otherwise haunts the map).
fn draw_player_marker(
    link: Res<GameLink>,
    toggles: Res<crate::ui::LayerToggles>,
    mut gizmos: Gizmos,
    time: Res<Time>,
) {
    if !toggles.player_marker {
        return;
    }
    let Some(fix) = &link.player else { return };
    let p = fix.pos;
    let t = time.elapsed_secs();
    let col = Color::srgb(0.15, 1.0, 0.55); // bright signal green
    let dim = Color::srgba(0.15, 1.0, 0.55, 0.35);
    // Pulsing ring (fresh fix pulses fast, settles after ~5 s).
    let age = (t - fix.at).max(0.0);
    let pulse = 1.0 + 0.25 * (t * if age < 5.0 { 6.0 } else { 1.5 }).sin();
    let r = 0.9 * pulse;
    let n = 24;
    let ring: Vec<Vec3> = (0..=n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            p + Vec3::new(a.cos() * r, 0.15, a.sin() * r)
        })
        .collect();
    gizmos.linestrip(ring, col);
    // Facing arrow (flattened forward from the screenshot quaternion).
    if let Some(fwd) = fix.fwd {
        gizmos.arrow(p + Vec3::Y * 0.2, p + Vec3::Y * 0.2 + fwd * 3.0, col);
    }
    // Vertical beacon: visible from the fly camera far above.
    gizmos.line(p, p + Vec3::Y * 30.0, dim);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_side_and_match_end_are_emitted_in_log_order() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut pending = String::new();
        let log = concat!(
            "Got notification | GroupMatchRaidSettings\n",
            "{\n",
            "  \"raidSettings\": {\"side\": \"Savage\"}\n",
            "}\n",
            "Got notification | UserMatchOver\n",
            "{\n",
            "  \"type\": \"userMatchOver\"\n",
            "}\n",
            "Got notification | GroupMatchRaidSettings\n",
            "{\n",
            "  \"raidSettings\": {\"side\": \"Pmc\"}\n",
            "}\n",
        );
        parse_notifications(&mut pending, log, &tx);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(matches!(events.as_slice(), [
            GameEvent::RaidSide(RaidSide::Scav),
            GameEvent::RaidEnd,
            GameEvent::RaidSide(RaidSide::Pmc),
        ]));
    }

    #[test]
    fn unrelated_uppercase_side_never_identifies_the_local_player() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut pending = String::new();
        parse_notifications(
            &mut pending,
            concat!(
                "Got notification | GroupMatchRaidSettings\n",
                "{\n",
                "  \"extendedProfile\": {\"Info\": {\"Side\": \"Savage\"}},\n",
                "  \"raidSettings\": {\"location\": \"Interchange\"}\n",
                "}\n",
            ),
            &tx,
        );
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn faction_filter_keeps_shared_and_unknown_secret_extracts() {
        assert!(RaidSide::Pmc.allows_extract("pmc"));
        assert!(!RaidSide::Pmc.allows_extract("scav"));
        assert!(RaidSide::Scav.allows_extract("scav"));
        assert!(!RaidSide::Scav.allows_extract("pmc"));
        assert!(RaidSide::Pmc.allows_extract("shared"));
        assert!(RaidSide::Scav.allows_extract("secret"));
    }
}

// ---------------------------------------------------------------------------------------------
// Watcher thread (std only): tail the two logs + scan the screenshots folder.
// ---------------------------------------------------------------------------------------------

/// LIVE-LINK HEALTH, published by the watcher thread for the settings UI.
///
/// Every failure in the watcher is non-fatal by design (it just keeps looping), which meant a dead
/// link — no game dir, no Logs folder, a BSG log-format change — was indistinguishable from a
/// healthy one that simply had nothing to report. The user only ever noticed as "the overlay
/// stopped following my raid". These counters make the state observable.
#[derive(Default)]
pub struct LinkHealth {
    /// Game install resolved (`detect_game_dir` returned something).
    pub game_dir: std::sync::atomic::AtomicBool,
    /// A `log_*` folder was found under the install.
    pub logs_dir: std::sync::atomic::AtomicBool,
    /// An `application_*.log` is being tailed right now.
    pub app_log: std::sync::atomic::AtomicBool,
    /// The screenshots folder was found.
    pub shots_dir: std::sync::atomic::AtomicBool,
    /// Watcher poll ticks completed — proves the thread is alive at all.
    pub ticks: std::sync::atomic::AtomicU64,
    /// Log lines the parsers have RECOGNIZED (map/fov/side/task). Zero after a long session with
    /// the game running is the signature of a log-format change.
    pub events: std::sync::atomic::AtomicU64,
}

pub static LINK_HEALTH: LinkHealth = LinkHealth {
    game_dir: std::sync::atomic::AtomicBool::new(false),
    logs_dir: std::sync::atomic::AtomicBool::new(false),
    app_log: std::sync::atomic::AtomicBool::new(false),
    shots_dir: std::sync::atomic::AtomicBool::new(false),
    ticks: std::sync::atomic::AtomicU64::new(0),
    events: std::sync::atomic::AtomicU64::new(0),
};

fn health_set(f: &std::sync::atomic::AtomicBool, v: bool) {
    f.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Tail state for one log file: byte offset consumed + partial-line/JSON carry-over.
#[derive(Default)]
struct Tail {
    path: Option<PathBuf>,
    offset: u64,
    pending: String,
}

fn watcher_thread(tx: Sender<GameEvent>) {
    let mut app_tail = Tail::default();
    let mut notif_tail = Tail::default();
    let mut shots_dir: Option<PathBuf> = None;
    // Only screenshots taken AFTER launch are fixes (old files in the folder are history).
    let mut last_shot = std::time::SystemTime::now();
    let mut game_dir = String::new();
    let mut tick: u64 = 0;
    loop {
        // Re-resolve the game install + screenshots folder occasionally (cheap registry/config
        // probes; the user can point Atlas at the game after launch).
        if tick % 20 == 0 {
            game_dir = crate::menu::detect_game_dir();
            if shots_dir.is_none() {
                shots_dir = find_screenshots_dir();
            }
        }
        tick += 1;
        LINK_HEALTH.ticks.store(tick, std::sync::atomic::Ordering::Relaxed);
        health_set(&LINK_HEALTH.game_dir, !game_dir.is_empty());
        health_set(&LINK_HEALTH.shots_dir, shots_dir.is_some());

        if !game_dir.is_empty() {
            let folder = latest_log_folder(Path::new(&game_dir));
            health_set(&LINK_HEALTH.logs_dir, folder.is_some());
            if let Some(folder) = folder {
                // Match the log CHANNEL, not a full filename: EFT writes
                // "<stamp> application_000.log" and "<stamp> push-notifications_000.log", so the
                // old needles ("application.log" / "notifications.log") never matched and the
                // whole log half of the link was silently dead -- no task tracking, no raid map
                // auto-switch. `retarget` already picks the newest match, which handles the
                // _000/_001 rotation for free.
                retarget(&mut app_tail, &folder, "application");
                retarget(&mut notif_tail, &folder, "notifications");
                health_set(&LINK_HEALTH.app_log, app_tail.path.is_some());
                if let Some(chunk) = read_new(&mut app_tail) {
                    parse_application(&mut app_tail.pending, &chunk, &tx);
                }
                if let Some(chunk) = read_new(&mut notif_tail) {
                    parse_notifications(&mut notif_tail.pending, &chunk, &tx);
                }
            }
        }
        // "Screenshot to locate current position" (menu toggle). When off we don't read the
        // folder at all — and we keep the watermark current so re-enabling it doesn't replay
        // every screenshot taken while it was off; only the NEXT one counts as a fix.
        if screenshot_locate() {
            if let Some(dir) = &shots_dir {
                scan_screenshots(dir, &mut last_shot, &tx);
            }
        } else {
            last_shot = std::time::SystemTime::now();
        }
        std::thread::sleep(std::time::Duration::from_millis(700));
    }
}

/// `<game>\Logs` (or `<game>\build\Logs`) -> the most recently modified `log_*` folder.
fn latest_log_folder(game: &Path) -> Option<PathBuf> {
    // `detect_game_dir` resolves the DATA folder (it validates on globalgamemanagers/sharedassets),
    // but EFT writes its logs beside the EXE, one level UP: <install>\Logs, not
    // <install>\EscapeFromTarkov_Data\Logs. Missing that candidate meant the log folder was never
    // found and the whole log half of the link did nothing -- no task tracking, no map follow --
    // regardless of the filename matching below. Try every layout, nearest first.
    let parent_logs = game.parent().map(|p| p.join("Logs"));
    let root = [
        Some(game.join("Logs")),
        parent_logs,
        Some(game.join("build").join("Logs")),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.is_dir())?;
    std::fs::read_dir(root)
        .ok()?
        .flatten()
        .filter(|e| {
            e.file_name().to_string_lossy().starts_with("log_")
                && e.file_type().map(|t| t.is_dir()).unwrap_or(false)
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// Point a tail at the newest file in `folder` whose name contains `needle` (EFT rotates to
/// `*_000.log` etc.); switching files or a shrunken file resets the offset.
fn retarget(tail: &mut Tail, folder: &Path, needle: &str) {
    let newest = std::fs::read_dir(folder)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_ascii_lowercase();
            n.contains(needle) && n.ends_with(".log")
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path());
    let Some(newest) = newest else { return };
    if tail.path.as_deref() != Some(newest.as_path()) {
        tail.path = Some(newest);
        tail.offset = 0;
        tail.pending.clear();
    }
}

/// Read everything past the tail's offset (None = no new bytes). A file smaller than the offset
/// (rotation/truncation) restarts from 0.
fn read_new(tail: &mut Tail) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let path = tail.path.as_ref()?;
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    if len < tail.offset {
        tail.offset = 0;
    }
    if len == tail.offset {
        return None;
    }
    f.seek(SeekFrom::Start(tail.offset)).ok()?;
    let mut buf = Vec::with_capacity((len - tail.offset) as usize);
    f.read_to_end(&mut buf).ok()?;
    tail.offset = len;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// application.log: line-oriented. `scene preset path:maps/<bundle>.bundle` = a raid map loading.
/// `2026-08-04 07:15:55.659|...` -> SystemTime, interpreting the stamp as LOCAL time (EFT writes
/// local). Returns None on anything that does not match, so a BSG format change degrades to "the
/// raid started when we noticed it" rather than to a wrong clock.
fn log_line_time(line: &str) -> Option<std::time::SystemTime> {
    let stamp = line.split('|').next()?.trim();
    let (date, rest) = stamp.split_once(' ')?;
    let mut d = date.split('-');
    let (y, mo, da) = (
        d.next()?.parse::<i32>().ok()?,
        d.next()?.parse::<u32>().ok()?,
        d.next()?.parse::<u32>().ok()?,
    );
    let hms = rest.split('.').next()?;
    let mut t = hms.split(':');
    let (h, mi, se) = (
        t.next()?.parse::<u32>().ok()?,
        t.next()?.parse::<u32>().ok()?,
        t.next()?.parse::<u32>().ok()?,
    );
    if !(1..=12).contains(&mo) || !(1..=31).contains(&da) || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    // Days since the Unix epoch (civil-from-days, Howard Hinnant's algorithm). No chrono dep.
    let (yy, mm) = if mo <= 2 { (y - 1, mo + 9) } else { (y, mo - 3) };
    let era = if yy >= 0 { yy } else { yy - 399 } / 400;
    let yoe = (yy - era * 400) as i64;
    let doy = ((153 * mm as i64 + 2) / 5) + da as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era as i64 * 146_097 + doe - 719_468;
    let secs_local = days * 86_400 + (h as i64) * 3600 + (mi as i64) * 60 + se as i64;
    // Local -> UTC without a tz crate: measure this machine's current offset once, from the same
    // clock both sides of the comparison use. Good to the second except across a DST boundary
    // mid-raid, where the clock would jump by an hour and still be self-consistent afterwards.
    let now = std::time::SystemTime::now();
    let now_unix = now.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64;
    let offset = local_utc_offset_secs(now_unix);
    let unix = secs_local - offset;
    if unix < 0 {
        return None;
    }
    std::time::UNIX_EPOCH.checked_add(std::time::Duration::from_secs(unix as u64))
}

/// This machine's current UTC offset in seconds, derived from the OS's own local-time formatting
/// of a known instant. Avoids a timezone dependency for the one thing we need it for.
fn local_utc_offset_secs(now_unix: i64) -> i64 {
    #[cfg(windows)]
    {
        // GetTimeZoneInformation reports Bias/DaylightBias in MINUTES west of UTC.
        #[repr(C)]
        struct Tzi {
            bias: i32,
            _std_name: [u16; 32],
            _std_date: [u16; 8],
            std_bias: i32,
            _dlt_name: [u16; 32],
            _dlt_date: [u16; 8],
            dlt_bias: i32,
        }
        unsafe extern "system" {
            fn GetTimeZoneInformation(info: *mut Tzi) -> u32;
        }
        const TIME_ZONE_ID_DAYLIGHT: u32 = 2;
        let mut tzi: Tzi = unsafe { std::mem::zeroed() };
        let rc = unsafe { GetTimeZoneInformation(&mut tzi) };
        if rc == u32::MAX {
            return 0;
        }
        let extra = if rc == TIME_ZONE_ID_DAYLIGHT { tzi.dlt_bias } else { tzi.std_bias };
        let _ = now_unix;
        -((tzi.bias + extra) as i64) * 60
    }
    #[cfg(not(windows))]
    {
        let _ = now_unix;
        0
    }
}

fn parse_application(pending: &mut String, chunk: &str, tx: &Sender<GameEvent>) {
    pending.push_str(chunk);
    // Keep the partial trailing line for the next read; process only complete lines.
    let upto = match pending.rfind('\n') {
        Some(i) => i + 1,
        None => return,
    };
    // Only the LAST preset in the chunk matters. The first read of a log tails from offset 0 —
    // deliberately, so launching Atlas mid-raid still finds the current map — but that chunk holds
    // every raid of the session, and emitting each one made the viewer load them in sequence
    // (streets, THEN interchange) before landing on the right one: minutes of wasted loading for
    // maps the player left long ago. Subsequent reads are small tails, where "last" is simply the
    // newest anyway.
    let mut latest: Option<&'static str> = None;
    let mut latest_fov: Option<f32> = None;
    // Raid start/end. Only the LAST of each in the chunk matters, for the same reason the map
    // preset takes the last: the first read of a log tails from offset 0 and holds every raid of
    // the session, so emitting each in turn would walk the app through hours of dead raids.
    let mut raid_start: Option<std::time::SystemTime> = None;
    let mut raid_end = false;
    let mut shot_key: Option<String> = None;
    for line in pending[..upto].lines() {
        // `2026-08-04 07:15:55.659|1.1.0.0.46624|Info|application|GameStarted:51.54(0) real:...`
        // The numbers on the line are load timings, not a clock -- the LINE'S timestamp is the
        // raid start, and it is the only one EFT writes down.
        if line.contains("|application|GameStarted:") {
            raid_start = log_line_time(line).or(Some(std::time::SystemTime::now()));
            raid_end = false; // this start supersedes any earlier end in the same chunk
        }
        // Fires at login and 3x on every return to the menu, so it is an END signal only while a
        // raid clock is running -- enforced by the consumer, which ignores it when `in_raid` is
        // None. Chosen because UserMatchOver, the documented trigger, appears in 0 of 310 log
        // folders here and 0 of the last 8 sessions.
        // LAST-ONE-WINS, by position in the chunk. Not "a start anywhere beats an end": the first
        // read of a log tails from offset 0, so an ordinary finished raid is `GameStarted ... then
        // PrepareSelectedProfileLocally`, and treating the start as the winner reported a raid
        // that ended hours ago as live (observed: the HUD showing RAID 0:00 on launch). Clearing
        // `raid_start` here is what makes the end the survivor.
        if line.contains("|application|PrepareSelectedProfileLocally") {
            raid_end = true;
            raid_start = None;
        }
        // `"MakeScreenshot","variants":[{"keyCode":["KeypadEnter"]}]` inside the Control settings
        // JSON dump. Line-keyed like FieldOfView above, so no JSON reassembly.
        if let Some(rest) = line.split("\"MakeScreenshot\"").nth(1) {
            if let Some(after) = rest.split("\"keyCode\"").nth(1) {
                let key: String = after
                    .split('"')
                    .nth(1)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !key.is_empty() && key.len() < 32 {
                    shot_key = Some(key);
                }
            }
        }
        if let Some(rest) = line.split("scene preset path:maps/").nth(1) {
            if let Some(bundle) = rest.split(".bundle").next() {
                if let Some(id) = bundle_to_map(bundle.trim()) {
                    latest = Some(id);
                }
            }
        }
        // `"FieldOfView": 50` inside the multi-line `Game settings:` JSON dump. Keyed on the
        // line, not the block, so we need no JSON reassembly; the last dump in the chunk wins
        // (a settings change mid-session re-dumps the whole block).
        if let Some(rest) = line.split("\"FieldOfView\":").nth(1) {
            let num: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if let Ok(v) = num.parse::<f32>() {
                if (30.0..=120.0).contains(&v) {
                    latest_fov = Some(v);
                }
            }
        }
    }
    if let Some(k) = shot_key {
        let _ = tx.send(GameEvent::ScreenshotKey(k));
        LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(at) = raid_start {
        let _ = tx.send(GameEvent::RaidStart(at));
        LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if raid_end {
        let _ = tx.send(GameEvent::RaidEnd);
        LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(id) = latest {
        let _ = tx.send(GameEvent::MapLoading(id.to_string()));
        LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(v) = latest_fov {
        let _ = tx.send(GameEvent::Fov(v));
        LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    pending.drain(..upto);
    cap(pending);
}

/// notifications.log: `Got notification | <kind>` followed by a multi-line JSON block (closing
/// brace at column 0). Parse the stream in order so a historical UserMatchOver correctly clears
/// the preceding raid side, while a later GroupMatchRaidSettings starts the next one.
fn parse_notifications(pending: &mut String, chunk: &str, tx: &Sender<GameEvent>) {
    pending.push_str(chunk);
    const MARK: &str = "Got notification | ";
    loop {
        let Some(mi) = pending.find(MARK) else {
            // No marker at all: nothing buffered matters beyond a partial marker at the very end.
            cap(pending);
            return;
        };
        let kind_start = mi + MARK.len();
        let Some(kind_end) = pending[kind_start..].find('\n').map(|o| kind_start + o) else {
            pending.drain(..mi);
            cap(pending);
            return;
        };
        let kind = pending[kind_start..kind_end].trim().to_string();
        let Some(js) = pending[kind_end..].find('{').map(|o| kind_end + o) else {
            pending.drain(..mi);
            cap(pending);
            return;
        };
        // The JSON block ends at the first close brace at column 0 (same rule TarkovMonitor's
        // `^{[\s\S]+?^}` regex uses).
        let Some(je) = pending[js..].find("\n}").map(|o| js + o + 2) else {
            pending.drain(..mi);
            cap(pending);
            return; // incomplete JSON - wait for the next chunk
        };
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&pending[js..je]) {
            match kind.as_str() {
                "ChatMessageReceived" => {
                    let msg = &v["message"];
                    let ty = msg["type"].as_i64().unwrap_or(0);
                    if (10..=12).contains(&ty) {
                        if let Some(tpl) = msg["templateId"].as_str() {
                            let id = tpl.split(' ').next().unwrap_or(tpl).to_string();
                            if !id.is_empty() {
                                let _ = tx.send(GameEvent::Task { id, status: ty });
                            LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                }
                "GroupMatchRaidSettings" => {
                    let side = match v["raidSettings"]["side"].as_str() {
                        Some("Pmc") => Some(RaidSide::Pmc),
                        Some("Savage") => Some(RaidSide::Scav),
                        _ => None,
                    };
                    if let Some(side) = side {
                        let _ = tx.send(GameEvent::RaidSide(side));
                            LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                "UserMatchOver" => {
                    let _ = tx.send(GameEvent::RaidEnd);
                            LINK_HEALTH.events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                _ => {}
            }
        }
        pending.drain(..je);
    }
}

/// Runaway guard: a malformed buffer (marker with no JSON ever completing) must not grow forever.
fn cap(pending: &mut String) {
    const MAX: usize = 1 << 20;
    if pending.len() > MAX {
        let cut = pending.len() - MAX / 2;
        pending.drain(..cut);
    }
}

/// EFT saves screenshots under Documents (possibly OneDrive-redirected).
fn find_screenshots_dir() -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE").ok()?;
    for base in [
        Path::new(&home).join("Documents"),
        Path::new(&home).join("OneDrive").join("Documents"),
    ] {
        let d = base.join("Escape From Tarkov").join("Screenshots");
        if d.is_dir() {
            return Some(d);
        }
    }
    None
}

/// New *.png since the last scan -> parse the position baked into the filename. The newest file
/// wins (one fix per scan is enough at a 0.7 s cadence).
fn scan_screenshots(dir: &Path, last: &mut std::time::SystemTime, tx: &Sender<GameEvent>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut newest: Option<(std::time::SystemTime, String)> = None;
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.to_ascii_lowercase().ends_with(".png") {
            continue;
        }
        let Ok(modified) = e.metadata().and_then(|m| m.modified()) else { continue };
        if modified > *last && newest.as_ref().map(|(t, _)| modified > *t).unwrap_or(true) {
            newest = Some((modified, name));
        }
    }
    let Some((t, name)) = newest else { return };
    *last = t;
    // Filename: "2026-07-21[14-30]_-123.45, 6.78, 90.12_0.0, 0.7, 0.0, 0.7 (0).png". The date/time
    // prefix contains no decimal-point numbers, so the first 3 floats are the position and the
    // next 4 the rotation quaternion.
    let f = floats_in(&name);
    if f.len() < 3 {
        return;
    }
    let (x, y, z) = (f[0], f[1], f[2]);
    let pos = Vec3::new(-x, y, z); // unity -> viewer (the pipeline-wide X-flip)
    // Unity forward = q * (0,0,1), X-flipped into viewer space. `fwd` is flattened to a heading
    // (the marker's arrow); `look` keeps the PITCH so the camera can sit in the player's eyes.
    let dirs = (f.len() >= 7).then(|| {
        let (qx, qy, qz, qw) = (f[3], f[4], f[5], f[6]);
        let fx = 2.0 * (qx * qz + qw * qy);
        let fy = 2.0 * (qy * qz - qw * qx);
        let fz = 1.0 - 2.0 * (qx * qx + qy * qy);
        (
            Vec3::new(-fx, 0.0, fz).normalize_or_zero(),
            Vec3::new(-fx, fy, fz).normalize_or_zero(),
        )
    });
    let nz = |v: Vec3| (v != Vec3::ZERO).then_some(v);
    // f[7] is EFT's in-game time of day in decimal hours, and it has always been parsed and
    // dropped. Game time runs at 7x real time, so it cannot be derived from the raid clock.
    let game_hour = f.get(7).copied().filter(|h| (0.0..24.0).contains(h));
    let _ = tx.send(GameEvent::PlayerFix {
        pos,
        fwd: dirs.and_then(|(f, _)| nz(f)),
        look: dirs.and_then(|(_, l)| nz(l)),
        game_hour,
    });
    // House-keeping: EFT never cleans these up, and locating yourself a few times a raid leaves a
    // pile of full-resolution PNGs. Delete ONLY the file we just consumed — we parsed a position
    // out of it, so it has served its purpose. Anything we could not parse is left alone.
    if DELETE_SHOTS.load(std::sync::atomic::Ordering::Relaxed) {
        let p = dir.join(&name);
        match std::fs::remove_file(&p) {
            Ok(()) => {
                info!("game link: consumed + deleted screenshot '{name}'");
                let _ = tx.send(GameEvent::ShotDeleted(name.clone()));
            }
            Err(e) => warn!("game link: could not delete '{name}': {e}"),
        }
    }
}

/// All `-?\d+\.\d+` decimals in `s`, in order (no regex dependency).
fn floats_in(s: &str) -> Vec<f32> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let start = i;
        let mut j = i + (b[i] == b'-') as usize;
        let ds = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > ds && j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
            let mut k = j + 1;
            while k < b.len() && b[k].is_ascii_digit() {
                k += 1;
            }
            if let Ok(v) = s[start..k].parse() {
                out.push(v);
            }
            i = k;
        } else {
            i = j.max(start + 1);
        }
    }
    out
}

/// Apply the raid map ON OVERLAY SUMMON — not the moment the log says so.
///
/// The user asked for this explicitly and it is the right call: Atlas is also a desk tool, and
/// having it yank itself onto another map while you are reading one (or browsing the menu) is
/// worse than loading a beat later. So the watcher only RECORDS `pending_map`, and the rising edge
/// of the overlay being shown is what commits it: summoning the overlay always means "show me
/// where I am now". A map with no pack raises the in-raid prompt (process / cancel) instead.
fn sync_map_on_overlay_show(
    overlay: Option<Res<crate::overlay::OverlayState>>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut link: ResMut<GameLink>,
    loaded: Option<Res<crate::render::LoadedPack>>,
    mut sw: ResMut<crate::MapSwitch>,
    mut was_shown: Local<bool>,
) {
    // Defense in depth: both summon paths already refuse in menu mode, but if `shown` ever went
    // true there anyway, MapSwitch would take the RELAUNCH path (spawn self + exit). Never here.
    if menu.is_some() {
        return;
    }
    let shown = overlay.map(|o| o.shown).unwrap_or(false);
    let rising = shown && !*was_shown;
    *was_shown = shown;
    if !rising {
        return;
    }
    let Some(id) = link.pending_map.clone() else { return };
    let dir = crate::paths::packs_root().join(format!("{id}.eftpack"));
    if !dir.join("manifest.json").is_file() {
        warn!("game link: overlay opened on '{id}' but no pack is built");
        link.unbuilt_map = Some(id);
        return;
    }
    link.unbuilt_map = None;
    let current = loaded.as_ref().and_then(|p| {
        p.0.root.file_name()?.to_str()?.strip_suffix(".eftpack").map(str::to_string)
    });
    // "Already there" is judged by the LOADED pack alone, and "already switching" by MapSwitch
    // itself — a remembered last-auto-switch would wrongly veto returning to the raid map after
    // the user browsed to a different one by hand.
    if current.as_deref() == Some(id.as_str()) || sw.0.is_some() {
        return;
    }
    info!("game link: overlay opened - loading the raid map '{id}'");
    sw.0 = Some(dir.to_string_lossy().into_owned());
}

#[cfg(test)]
mod raid_clock_tests {
    use super::*;

    /// The timestamp parser against the REAL line format, including the local->UTC step. A wrong
    /// offset here shows up as a raid clock that is hours out, which looks like a live countdown
    /// and is not one.
    #[test]
    fn log_line_time_parses_the_real_format() {
        let line = "2026-08-04 07:15:55.659|1.1.0.0.46624|Info|application|GameStarted:51.54(0) real:65.29(0) diff:13.75";
        let t = log_line_time(line).expect("should parse");
        // Round-trip through the same offset the parser used, so the assertion does not itself
        // depend on the machine's timezone.
        let unix = t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let local = unix + local_utc_offset_secs(now_unix);
        // 2026-08-04 07:15:55 local, expressed as seconds-of-day.
        assert_eq!(local.rem_euclid(86_400), 7 * 3600 + 15 * 60 + 55);
    }

    #[test]
    fn log_line_time_rejects_junk() {
        assert!(log_line_time("not a log line").is_none());
        assert!(log_line_time("2026-13-99 99:99:99.000|x|Info|application|GameStarted:1").is_none());
    }

    /// `PrepareSelectedProfileLocally` fires at LOGIN and 3x on every return to the menu, so a
    /// chunk that contains a start AFTER an end must not report the end: that is the ordinary
    /// login-then-raid sequence, and treating it as "raid over" would clear the live raid.
    #[test]
    fn a_start_after_an_end_wins_within_one_chunk() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut pending = String::new();
        let chunk = "2026-08-04 07:13:47.654|v|Info|application|PrepareSelectedProfileLocally ProfileId:x\n\
                     2026-08-04 07:15:55.659|v|Info|application|GameStarted:51.54(0) real:65.29(0)\n";
        parse_application(&mut pending, chunk, &tx);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(
            events.iter().any(|e| matches!(e, GameEvent::RaidStart(_))),
            "expected a RaidStart, got {events:?}"
        );
        assert!(
            !events.iter().any(|e| matches!(e, GameEvent::RaidEnd)),
            "the login PrepareSelectedProfileLocally must not read as a raid end: {events:?}"
        );
    }

    /// A cold tail of a whole session is `GameStarted ... PrepareSelectedProfileLocally`. Reading
    /// that as a live raid is what put `RAID 0:00` on the HUD at launch for a raid that had ended
    /// hours earlier. The timer must appear only while one is actually running, so the LAST event
    /// in the chunk decides.
    #[test]
    fn a_finished_raid_does_not_report_as_live() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut pending = String::new();
        let done = "2026-08-04 07:15:55.659|v|Info|application|GameStarted:51.54(0) real:65.29(0)
                    2026-08-04 07:40:37.481|v|Info|application|PrepareSelectedProfileLocally ProfileId:x
";
        parse_application(&mut pending, done, &tx);
        let ev: Vec<_> = rx.try_iter().collect();
        assert!(
            ev.iter().any(|e| matches!(e, GameEvent::RaidEnd)),
            "a finished raid must report RaidEnd, got {ev:?}"
        );
        assert!(
            !ev.iter().any(|e| matches!(e, GameEvent::RaidStart(_))),
            "a finished raid must NOT report a live start: {ev:?}"
        );
    }

    #[test]
    fn an_end_alone_is_reported() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut pending = String::new();
        parse_application(
            &mut pending,
            "2026-08-04 07:40:37.481|v|Info|application|PrepareSelectedProfileLocally ProfileId:x\n",
            &tx,
        );
        let events: Vec<_> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(e, GameEvent::RaidEnd)), "{events:?}");
    }
}
