//! Live game link — TarkovMonitor-style passive file watching (no game hooks).
//!
//! Everything here reads files EFT itself writes; nothing touches the game process:
//!   * `<game>\Logs\log_*\*application.log`   — `scene preset path:maps/<bundle>.bundle` tells us
//!     which map is loading -> in-place map swap (MapSwitch) so Atlas follows the raid.
//!   * `<game>\Logs\log_*\*notifications.log` — task status push messages (started/failed/finished,
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

enum GameEvent {
    /// A raid map started loading in the game (atlas map id, already bundle->id resolved).
    MapLoading(String),
    /// A new screenshot fix: viewer-space position + flattened facing (None if no quaternion).
    PlayerFix { pos: Vec3, fwd: Option<Vec3> },
    /// Task status push: 10 = started, 11 = failed, 12 = finished.
    Task { id: String, status: i64 },
    /// The raid ended (UserMatchOver) — the last fix is stale.
    RaidEnd,
}

/// Scene-preset bundle name -> our pack id (TarkovMonitor's MapBundles table, mapped to the atlas
/// roster). Bundles for maps we don't ship (terminal, icebreaker) are simply absent.
fn bundle_to_map(bundle: &str) -> Option<&'static str> {
    Some(match bundle {
        "city_preset" => "streets",
        "customs_preset" => "customs",
        "factory_day_preset" | "factory_night_preset" => "factory_rework",
        "laboratory_preset" | "laboratory_dark_preset" => "labs",
        "labyrinth_preset" => "labyrinth",
        "lighthouse_preset" => "lighthouse",
        "rezerv_base_preset" => "reserve",
        "sandbox_preset" | "sandbox_high_preset" => "ground_zero",
        "shopping_mall" => "interchange",
        "shoreline_preset" => "shoreline",
        "woods_preset" => "woods",
        _ => return None,
    })
}

// ---------------------------------------------------------------------------------------------
// Bevy side
// ---------------------------------------------------------------------------------------------

/// The latest live player fix, in viewer space.
pub struct PlayerFixState {
    pub pos: Vec3,
    pub fwd: Option<Vec3>,
    /// `Time::elapsed_secs` when the fix arrived (drives the marker pulse).
    pub at: f32,
}

#[derive(Resource)]
pub struct GameLink {
    rx: Mutex<Receiver<GameEvent>>,
    pub player: Option<PlayerFixState>,
    /// Last map id we auto-switched to (debounce: the preset line appears once per load, but a
    /// re-log or log re-read must not bounce the viewer back).
    last_auto: Option<String>,
}

pub struct GameWatchPlugin;

impl Plugin for GameWatchPlugin {
    fn build(&self, app: &mut App) {
        if std::env::var("EFT_GAME_LINK").is_ok_and(|v| v.trim() == "0") {
            info!("game link: disabled (EFT_GAME_LINK=0)");
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("eft-game-watch".into())
            .spawn(move || watcher_thread(tx))
            .ok();
        app.insert_resource(GameLink { rx: Mutex::new(rx), player: None, last_auto: None })
            .add_systems(Update, (apply_game_events, draw_player_marker));
    }
}

/// Drain the watcher channel and apply each event to the app's existing machinery: MapSwitch for
/// the in-place swap, StartPoint + the marker for the player fix, PlayerProgress for tasks.
fn apply_game_events(
    mut link: ResMut<GameLink>,
    menu: Option<Res<crate::menu::MenuState>>,
    loaded: Option<Res<crate::render::LoadedPack>>,
    mut sw: ResMut<crate::MapSwitch>,
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
                // In the menu the PLAY flow owns pack loading (and relaunches); only auto-swap the
                // live 3D viewer, and only when the target pack is actually built.
                if menu.is_some() {
                    continue;
                }
                let dir = crate::paths::packs_root().join(format!("{id}.eftpack"));
                if !dir.join("manifest.json").is_file() {
                    info!("game link: raid map '{id}' detected but no pack is built - staying put");
                    continue;
                }
                let current = loaded.as_ref().and_then(|p| {
                    p.0.root.file_name()?.to_str()?.strip_suffix(".eftpack").map(str::to_string)
                });
                if current.as_deref() == Some(id.as_str()) || link.last_auto.as_deref() == Some(id.as_str())
                {
                    continue; // already there / already switching
                }
                info!("game link: raid loading on '{id}' - switching map in place");
                link.last_auto = Some(id.clone());
                sw.0 = Some(dir.to_string_lossy().into_owned());
            }
            GameEvent::PlayerFix { pos, fwd } => {
                info!("game link: player fix at {:.1},{:.1},{:.1}", pos.x, pos.y, pos.z);
                link.player = Some(PlayerFixState { pos, fwd, at: time.elapsed_secs() });
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
            GameEvent::RaidEnd => {
                link.player = None;
            }
        }
    }
}

/// Draw the live player marker: pulsing ground ring + facing arrow + a vertical beacon so the
/// player is findable from any camera height. Gizmos = immediate mode, nothing to clean up.
fn draw_player_marker(link: Res<GameLink>, mut gizmos: Gizmos, time: Res<Time>) {
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

// ---------------------------------------------------------------------------------------------
// Watcher thread (std only): tail the two logs + scan the screenshots folder.
// ---------------------------------------------------------------------------------------------

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

        if !game_dir.is_empty() {
            if let Some(folder) = latest_log_folder(Path::new(&game_dir)) {
                retarget(&mut app_tail, &folder, "application.log");
                retarget(&mut notif_tail, &folder, "notifications.log");
                if let Some(chunk) = read_new(&mut app_tail) {
                    parse_application(&mut app_tail.pending, &chunk, &tx);
                }
                if let Some(chunk) = read_new(&mut notif_tail) {
                    parse_notifications(&mut notif_tail.pending, &chunk, &tx);
                }
            }
        }
        if let Some(dir) = &shots_dir {
            scan_screenshots(dir, &mut last_shot, &tx);
        }
        std::thread::sleep(std::time::Duration::from_millis(700));
    }
}

/// `<game>\Logs` (or `<game>\build\Logs`) -> the most recently modified `log_*` folder.
fn latest_log_folder(game: &Path) -> Option<PathBuf> {
    let mut root = game.join("Logs");
    if !root.is_dir() {
        root = game.join("build").join("Logs");
        if !root.is_dir() {
            return None;
        }
    }
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
        .filter(|e| e.file_name().to_string_lossy().contains(needle))
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
fn parse_application(pending: &mut String, chunk: &str, tx: &Sender<GameEvent>) {
    pending.push_str(chunk);
    // Keep the partial trailing line for the next read; process only complete lines.
    let upto = match pending.rfind('\n') {
        Some(i) => i + 1,
        None => return,
    };
    for line in pending[..upto].lines() {
        if let Some(rest) = line.split("scene preset path:maps/").nth(1) {
            if let Some(bundle) = rest.split(".bundle").next() {
                if let Some(id) = bundle_to_map(bundle.trim()) {
                    let _ = tx.send(GameEvent::MapLoading(id.to_string()));
                }
            }
        }
    }
    pending.drain(..upto);
    cap(pending);
}

/// notifications.log: `Got notification | ChatMessageReceived` followed by a multi-line JSON block
/// (closing brace at column 0). Task pushes carry message.type 10/11/12 + templateId "<taskid> 0".
fn parse_notifications(pending: &mut String, chunk: &str, tx: &Sender<GameEvent>) {
    if chunk.contains("UserMatchOver") {
        let _ = tx.send(GameEvent::RaidEnd);
    }
    pending.push_str(chunk);
    const MARK: &str = "Got notification | ChatMessageReceived";
    loop {
        let Some(mi) = pending.find(MARK) else {
            // No marker at all: nothing buffered matters beyond a partial marker at the very end.
            cap(pending);
            return;
        };
        let Some(js) = pending[mi..].find('{').map(|o| mi + o) else {
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
            let msg = &v["message"];
            let ty = msg["type"].as_i64().unwrap_or(0);
            if (10..=12).contains(&ty) {
                if let Some(tpl) = msg["templateId"].as_str() {
                    let id = tpl.split(' ').next().unwrap_or(tpl).to_string();
                    if !id.is_empty() {
                        let _ = tx.send(GameEvent::Task { id, status: ty });
                    }
                }
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
    let fwd = (f.len() >= 7).then(|| {
        // Unity forward = q * (0,0,1), then flatten to a heading and X-flip into viewer space.
        let (qx, qy, qz, qw) = (f[3], f[4], f[5], f[6]);
        let fx = 2.0 * (qx * qz + qw * qy);
        let fz = 1.0 - 2.0 * (qx * qx + qy * qy);
        Vec3::new(-fx, 0.0, fz).normalize_or_zero()
    });
    let _ = tx.send(GameEvent::PlayerFix { pos, fwd: fwd.filter(|v| *v != Vec3::ZERO) });
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
