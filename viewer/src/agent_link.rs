//! eft::agent_link — a standardized, lockstep control interface to the drone sim, so external
//! programs (RL trainers, scripted policies, notebooks) can fly the FPV drone and practice
//! target-tracking tasks against the real map geometry.
//!
//! Transport: newline-delimited JSON over TCP on 127.0.0.1 (local only, off by default — enable
//! with the camera-tab checkbox or `EFT_AGENT=<port>`). One client at a time. Every request gets
//! exactly one JSON reply line, so the protocol is trivially usable from any language; a
//! Gymnasium-compatible Python wrapper lives in `tools/drone_env.py` and the full spec in
//! `docs/AGENT_LINK.md`.
//!
//! LOCKSTEP: the sim only advances inside a `step` request (fixed dt, fixed substeps, seeded RNG)
//! — physics is decoupled from the render loop, so training runs as fast as the socket allows and
//! is reproducible. The viewer merely *visualizes* the session each frame (spectate camera +
//! target gizmos). A map switch invalidates the session (the collision grids are rebuilt).
//!
//! The "thing to track" is a simulated ground target (a walking-human proxy: capsule-sized,
//! ground-following, wall-blocked) with three motion modes — static, waypoint patrol, seeded
//! random wander — all stepped with the same grid queries the walk camera uses.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use serde_json::{json, Value};

use crate::drone::{self, ControlMode, DroneAction, DroneParams, DroneState};
use crate::render::{CullCamera, LoadedPack, MapEpoch};
use crate::walk_ground::{GroundData, GroundGrid, STEP_UP};

/// Target body: feet→center offset (m) — observations report the torso center of a human-sized
/// target, and gizmos draw a capsule of roughly this size.
const TARGET_CENTER: f32 = 0.9;

// ---------------------------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------------------------

/// UI/env control surface (ECS side). `status` is a mirror for the camera panel.
#[derive(Resource)]
pub struct AgentLinkCtl {
    pub enabled: bool,
    pub port: u16,
    pub status: String,
}

impl Default for AgentLinkCtl {
    fn default() -> Self {
        let env = std::env::var("EFT_AGENT").ok();
        let port = env
            .as_deref()
            .and_then(|v| v.trim().parse::<u16>().ok())
            .unwrap_or(7878);
        Self {
            enabled: env.is_some(),
            port,
            status: String::from("off"),
        }
    }
}

/// The world shared between the TCP server thread (steps it) and the ECS (visualizes it).
#[derive(Resource, Clone)]
pub struct AgentShared(pub Arc<Mutex<AgentWorld>>);

/// Simulated target the drone is asked to track.
pub struct TargetSim {
    pub mode: TargetMode,
    /// Feet position (world).
    pub feet: Vec3,
    pub vel: Vec3,
    pub speed: f32,
}

pub enum TargetMode {
    Static,
    Waypoints { pts: Vec<Vec3>, next: usize, looped: bool },
    Wander { center: Vec3, radius: f32, goal: Vec3 },
}

impl TargetSim {
    pub fn center(&self) -> Vec3 {
        self.feet + Vec3::Y * TARGET_CENTER
    }
}

/// Everything a session needs. Physics state + config lives here so the server thread can step it
/// under one short lock; the ECS reads pos/quat for spectate + gizmos.
pub struct AgentWorld {
    pub connected: bool,
    /// A reset has been received and the session drives the drone (spectate + gizmos on).
    pub active: bool,
    pub spectate: bool,
    pub drone: DroneState,
    pub params: DroneParams,
    pub control: ControlMode,
    pub target: TargetSim,
    pub dt: f32,
    pub t: f64,
    pub wind: Vec3,
    pub action_noise: f32,
    pub latency: usize,
    /// Pilot/home position (the reset spawn) — anchor of the analog video-link RF model.
    pub home: Vec3,
    /// Video-link usable range (m) for the obs `signal` field.
    pub range: f32,
    queue: VecDeque<DroneAction>,
    rng: u64,
    // Mirrored in by agent_sync each frame:
    pub grid: Option<Arc<GroundData>>,
    pub map: String,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub fov_deg: f32,
    pub aspect: f32,
    pub frame_req: Option<String>,
    // Last-step event flags (reported in obs).
    pub collided: bool,
    pub impact: f32,
}

impl Default for AgentWorld {
    fn default() -> Self {
        Self {
            connected: false,
            active: false,
            spectate: true,
            drone: DroneState::default(),
            params: DroneParams::default(),
            control: ControlMode::Rates,
            target: TargetSim {
                mode: TargetMode::Static,
                feet: Vec3::ZERO,
                vel: Vec3::ZERO,
                speed: 0.0,
            },
            dt: 0.02,
            t: 0.0,
            wind: Vec3::ZERO,
            action_noise: 0.0,
            latency: 0,
            home: Vec3::ZERO,
            range: 350.0,
            queue: VecDeque::new(),
            rng: 0x9E3779B97F4A7C15,
            grid: None,
            map: String::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
            fov_deg: 90.0,
            aspect: 16.0 / 9.0,
            frame_req: None,
            collided: false,
            impact: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Deterministic RNG (xorshift64*) — seeded per reset; no external dep.
// ---------------------------------------------------------------------------------------------

fn rng_next(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

/// Uniform [0,1).
fn rng_f32(s: &mut u64) -> f32 {
    (rng_next(s) >> 40) as f32 / (1u64 << 24) as f32
}

/// Uniform [-1,1).
fn rng_pm(s: &mut u64) -> f32 {
    rng_f32(s) * 2.0 - 1.0
}

// ---------------------------------------------------------------------------------------------
// ECS systems
// ---------------------------------------------------------------------------------------------

/// Start the server when enabled; every frame, hand the current grid/map/aspect into the shared
/// world, invalidate the session on a map switch, service frame-capture requests, and mirror a
/// status line for the UI.
pub fn agent_sync(
    mut commands: Commands,
    mut ctl: ResMut<AgentLinkCtl>,
    shared: Option<Res<AgentShared>>,
    grid: Option<Res<GroundGrid>>,
    pack: Option<Res<LoadedPack>>,
    epoch: Res<MapEpoch>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut last_epoch: Local<Option<u64>>,
) {
    let Some(shared) = shared else {
        // First frame with the link enabled: bind the listener and publish the shared world.
        // (Once started it stays up for the process lifetime — local-only and idle-cheap.)
        if ctl.enabled {
            let world = Arc::new(Mutex::new(AgentWorld::default()));
            match start_server(ctl.port, world.clone()) {
                Ok(()) => {
                    commands.insert_resource(AgentShared(world));
                    info!("agent_link: listening on 127.0.0.1:{}", ctl.port);
                    ctl.status = format!("listening :{}", ctl.port);
                }
                Err(e) => {
                    ctl.status = format!("listen failed: {e}");
                    ctl.enabled = false;
                }
            }
        }
        return;
    };
    let mut w = shared.0.lock().unwrap();

    // Map switch → session invalid (old-map coordinates + a stale grid make no sense).
    if *last_epoch != Some(epoch.0) {
        if last_epoch.is_some() && w.active {
            w.active = false;
            info!("agent_link: session ended (map switch)");
        }
        w.grid = None;
        *last_epoch = Some(epoch.0);
    }
    if w.grid.is_none() {
        if let Some(g) = &grid {
            if g.has_ceilings {
                w.grid = Some(g.0.clone());
            }
        }
    }
    if let Some(p) = &pack {
        if w.map != p.0.manifest.map {
            w.map = p.0.manifest.map.clone();
        }
        let b = p.0.manifest.bounds;
        w.bounds_min = Vec3::new(b[0], b[1], b[2]);
        w.bounds_max = Vec3::new(b[3], b[4], b[5]);
    }
    if let Ok(win) = windows.single() {
        let h = win.height().max(1.0);
        w.aspect = win.width().max(1.0) / h;
    }
    if let Some(path) = w.frame_req.take() {
        use bevy::render::view::screenshot::{save_to_disk, Screenshot};
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path.clone()));
        info!("agent_link: frame -> {path}");
    }
    ctl.status = match (w.connected, w.active) {
        (false, _) => format!("listening :{}", ctl.port),
        (true, false) => "client connected".into(),
        (true, true) => format!("session live  t={:.1}s", w.t),
    };
}

/// Draw the session: target capsule ring + vertical beacon, drone marker when not spectating.
pub fn agent_gizmos(
    shared: Option<Res<AgentShared>>,
    mut gizmos: Gizmos,
    cam: Query<&Transform, With<CullCamera>>,
) {
    let Some(shared) = shared else { return };
    let w = shared.0.lock().unwrap();
    if !w.active {
        return;
    }
    let red = Color::srgb(1.0, 0.35, 0.25);
    let c = w.target.center();
    gizmos.sphere(Isometry3d::from_translation(c), 0.35, red);
    gizmos.line(w.target.feet, w.target.feet + Vec3::Y * 12.0, Color::srgba(1.0, 0.35, 0.25, 0.4));
    // Heading tick so an approach angle is readable.
    if w.target.vel.length_squared() > 0.01 {
        gizmos.arrow(c, c + w.target.vel.normalize() * 1.2, red);
    }
    let spectating = w.spectate;
    if !spectating || cam.single().map(|t| (t.translation - w.drone.pos).length() > 1.0).unwrap_or(true) {
        let cyan = Color::srgb(0.25, 0.9, 1.0);
        gizmos.sphere(Isometry3d::from_translation(w.drone.pos), drone::DRONE_RADIUS * 2.0, cyan);
        let fwd = w.drone.quat * Vec3::NEG_Z;
        gizmos.arrow(w.drone.pos, w.drone.pos + fwd * 1.0, cyan);
    }
}

// ---------------------------------------------------------------------------------------------
// TCP server
// ---------------------------------------------------------------------------------------------

fn start_server(port: u16, world: Arc<Mutex<AgentWorld>>) -> std::io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    std::thread::Builder::new()
        .name("agent-link".into())
        .spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                if let Ok(peer) = stream.peer_addr() {
                    info!("agent_link: client {peer} connected");
                }
                {
                    world.lock().unwrap().connected = true;
                }
                let _ = serve_client(stream, &world);
                let mut w = world.lock().unwrap();
                w.connected = false;
                w.active = false;
                info!("agent_link: client disconnected, session ended");
            }
        })
        .map(|_| ())
}

fn serve_client(stream: TcpStream, world: &Arc<Mutex<AgentWorld>>) -> std::io::Result<()> {
    stream.set_nodelay(true).ok();
    let mut writer = stream.try_clone()?;
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(req) => {
                let mut w = world.lock().unwrap();
                handle(&req, &mut w)
            }
            Err(e) => json!({"ok": false, "err": format!("bad json: {e}")}),
        };
        writer.write_all(reply.to_string().as_bytes())?;
        writer.write_all(b"\n")?;
        if reply.get("bye").is_some() {
            break;
        }
    }
    Ok(())
}

fn vec3_of(v: Option<&Value>) -> Option<Vec3> {
    let arr = v?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some(Vec3::new(
        arr[0].as_f64()? as f32,
        arr[1].as_f64()? as f32,
        arr[2].as_f64()? as f32,
    ))
}

fn jvec(v: Vec3) -> Value {
    json!([v.x, v.y, v.z])
}

/// Horizontal (XZ) distance between two world points.
fn dist_xz(a: Vec3, b: Vec3) -> f32 {
    let (dx, dz) = (a.x - b.x, a.z - b.z);
    (dx * dx + dz * dz).sqrt()
}

fn handle(req: &Value, w: &mut AgentWorld) -> Value {
    match req.get("cmd").and_then(Value::as_str) {
        Some("hello") => json!({
            "ok": true, "proto": 1, "app": "atlas",
            "map": w.map, "grid_ready": w.grid.is_some(),
            "bounds": [jvec(w.bounds_min), jvec(w.bounds_max)],
            "params": params_json(&w.params),
        }),
        Some("params") => json!({"ok": true, "params": params_json(&w.params)}),
        Some("spectate") => {
            w.spectate = req.get("on").and_then(Value::as_bool).unwrap_or(true);
            json!({"ok": true, "spectate": w.spectate})
        }
        Some("frame") => match req.get("path").and_then(Value::as_str) {
            Some(p) => {
                w.frame_req = Some(p.to_string());
                json!({"ok": true, "queued": true,
                       "note": "async window capture; poll the file. Spectate must be on for the drone's view."})
            }
            None => json!({"ok": false, "err": "frame needs a 'path'"}),
        },
        Some("reset") => reset(req, w),
        Some("step") => step_cmd(req, w),
        Some("obs") => {
            if !w.active {
                json!({"ok": false, "err": "no active session (send reset)"})
            } else {
                obs(w)
            }
        }
        Some("bye") => json!({"ok": true, "bye": true}),
        Some(other) => json!({"ok": false, "err": format!("unknown cmd '{other}'")}),
        None => json!({"ok": false, "err": "missing 'cmd'"}),
    }
}

fn params_json(p: &DroneParams) -> Value {
    json!({
        "mass": p.mass, "gravity": p.gravity, "max_thrust": p.max_thrust,
        "hover_throttle": p.hover_throttle(),
        "drag_q_axial": p.drag_q_axial, "drag_q_side": p.drag_q_side, "drag_l": p.drag_l,
        "max_rate_rad": [p.max_rate.x, p.max_rate.y, p.max_rate.z],
        "rate_tau": p.rate_tau, "rate_tau_yaw": p.rate_tau_yaw, "thrust_tau": p.thrust_tau,
        "propwash": p.propwash,
        "max_tilt_rad": p.max_tilt, "tilt_tau": p.tilt_tau,
        "cam_tilt_rad": p.cam_tilt, "radius": drone::DRONE_RADIUS,
        "restitution": p.restitution, "contact_tau": p.contact_tau,
        "crash_speed": p.crash_speed,
    })
}

/// Highest surface under (x,z) scanning from the top of the map. None = void.
fn top_ground(g: &GroundData, x: f32, z: f32, top_y: f32) -> Option<f32> {
    g.ground_height(x, z, top_y, STEP_UP)
}

fn reset(req: &Value, w: &mut AgentWorld) -> Value {
    let Some(grid) = w.grid.clone() else {
        return json!({"ok": false, "err": "collision grid not built yet (open a map + enable the agent link, then retry)"});
    };
    let g = &*grid;
    w.rng = req.get("seed").and_then(Value::as_u64).unwrap_or(0x5EED) | 1;
    w.dt = req
        .get("dt")
        .and_then(Value::as_f64)
        .map(|v| (v as f32).clamp(0.001, 0.1))
        .unwrap_or(0.02);
    w.params = DroneParams::default();
    if let Some(pw) = req.get("propwash").and_then(Value::as_f64) {
        w.params.propwash = (pw as f32).clamp(0.0, 4.0);
    }
    w.control = match req.get("control").and_then(Value::as_str) {
        Some("angle") => ControlMode::Angle,
        _ => ControlMode::Rates,
    };
    w.wind = req
        .get("wind")
        .and_then(|v| v.as_array())
        .filter(|a| a.len() == 2)
        .map(|a| {
            Vec3::new(
                a[0].as_f64().unwrap_or(0.0) as f32,
                0.0,
                a[1].as_f64().unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::ZERO);
    w.action_noise = req.get("action_noise").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    w.latency = req.get("latency").and_then(Value::as_u64).unwrap_or(0).min(16) as usize;
    w.fov_deg = req
        .get("fov_deg")
        .and_then(Value::as_f64)
        .map(|v| (v as f32).clamp(30.0, 150.0))
        .unwrap_or(90.0);
    w.queue.clear();
    w.t = 0.0;
    w.collided = false;
    w.impact = 0.0;

    let top_y = w.bounds_max.y + 10.0;
    let center = (w.bounds_min + w.bounds_max) * 0.5;

    // Drone spawn: explicit pos, or hover 12 m over the highest surface at map center.
    let dcfg = req.get("drone");
    let dpos = vec3_of(dcfg.and_then(|d| d.get("pos")));
    let dyaw = dcfg
        .and_then(|d| d.get("yaw"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    let dpos = dpos.unwrap_or_else(|| {
        let y = top_ground(g, center.x, center.z, top_y).unwrap_or(center.y);
        Vec3::new(center.x, y + 12.0, center.z)
    });
    w.drone = DroneState::spawn(dpos, dyaw);
    w.home = dpos;
    w.range = req
        .get("range")
        .and_then(Value::as_f64)
        .map(|v| (v as f32).clamp(10.0, 5000.0))
        .unwrap_or(350.0);

    // Target spawn + motion mode.
    let tcfg = req.get("target");
    let tmode = tcfg
        .and_then(|t| t.get("mode"))
        .and_then(Value::as_str)
        .unwrap_or("wander");
    let speed = tcfg
        .and_then(|t| t.get("speed"))
        .and_then(Value::as_f64)
        .map(|v| (v as f32).clamp(0.0, 8.0))
        .unwrap_or(1.8);
    let feet = vec3_of(tcfg.and_then(|t| t.get("pos"))).or_else(|| {
        // Sample a grounded point 8..30 m from the drone.
        for _ in 0..64 {
            let ang = rng_f32(&mut w.rng) * std::f32::consts::TAU;
            let dist = 8.0 + rng_f32(&mut w.rng) * 22.0;
            let (x, z) = (dpos.x + ang.cos() * dist, dpos.z + ang.sin() * dist);
            if let Some(y) = top_ground(g, x, z, top_y) {
                return Some(Vec3::new(x, y, z));
            }
        }
        None
    });
    let Some(feet) = feet else {
        return json!({"ok": false, "err": "could not find ground for the target near the drone spawn"});
    };
    let mode = match tmode {
        "static" => TargetMode::Static,
        "waypoints" => {
            let pts: Vec<Vec3> = tcfg
                .and_then(|t| t.get("points"))
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|p| vec3_of(Some(p))).collect())
                .unwrap_or_default();
            if pts.is_empty() {
                return json!({"ok": false, "err": "waypoints mode needs 'points': [[x,y,z],..]"});
            }
            let looped = tcfg
                .and_then(|t| t.get("loop"))
                .and_then(Value::as_bool)
                .unwrap_or(true);
            TargetMode::Waypoints { pts, next: 0, looped }
        }
        _ => {
            let radius = tcfg
                .and_then(|t| t.get("radius"))
                .and_then(Value::as_f64)
                .map(|v| (v as f32).clamp(2.0, 200.0))
                .unwrap_or(30.0);
            TargetMode::Wander { center: feet, radius, goal: feet }
        }
    };
    w.target = TargetSim { mode, feet, vel: Vec3::ZERO, speed };
    w.active = true;
    obs(w)
}

fn step_cmd(req: &Value, w: &mut AgentWorld) -> Value {
    if !w.active {
        return json!({"ok": false, "err": "no active session (send reset)"});
    }
    let Some(grid) = w.grid.clone() else {
        w.active = false;
        return json!({"ok": false, "err": "grid dropped (map switch?) — session ended"});
    };
    let act = req.get("action").and_then(Value::as_array);
    let Some(act) = act.filter(|a| a.len() == 4) else {
        return json!({"ok": false, "err": "step needs 'action': [roll,pitch,yaw,throttle]"});
    };
    let action = DroneAction {
        roll: act[0].as_f64().unwrap_or(0.0) as f32,
        pitch: act[1].as_f64().unwrap_or(0.0) as f32,
        yaw: act[2].as_f64().unwrap_or(0.0) as f32,
        throttle: act[3].as_f64().unwrap_or(0.0) as f32,
    };
    let repeat = req.get("repeat").and_then(Value::as_u64).unwrap_or(1).clamp(1, 200);
    let g = &*grid;
    w.collided = false;
    w.impact = 0.0;
    for _ in 0..repeat {
        // Actuation latency: the command that reaches the motors is `latency` control ticks old.
        w.queue.push_back(action);
        let mut eff = if w.queue.len() > w.latency {
            w.queue.pop_front().unwrap()
        } else {
            DroneAction::default()
        };
        if w.action_noise > 0.0 {
            eff.roll += rng_pm(&mut w.rng) * w.action_noise;
            eff.pitch += rng_pm(&mut w.rng) * w.action_noise;
            eff.yaw += rng_pm(&mut w.rng) * w.action_noise;
            eff.throttle += rng_pm(&mut w.rng) * w.action_noise;
        }
        // ≤2.5 ms physics substeps under the control dt (dt=0.001 → a true 1 kHz loop).
        let n = ((w.dt / 0.0025).ceil() as u32).max(1);
        let sub = w.dt / n as f32;
        for _ in 0..n {
            let out = drone::step(&mut w.drone, &w.params, eff, w.control, w.wind, Some(g), sub);
            w.collided |= out.collided;
            w.impact = w.impact.max(out.impact);
        }
        step_target(w, g);
        w.t += w.dt as f64;
    }
    obs(w)
}

/// Advance the target one control tick: walk toward the current goal on the ground grid, slide
/// along walls, ground-follow, and re-goal (wander) / advance (waypoints) on arrival or block.
fn step_target(w: &mut AgentWorld, g: &GroundData) {
    let dt = w.dt;
    let speed = w.target.speed;
    if speed <= 0.0 {
        w.target.vel = Vec3::ZERO;
        return;
    }
    let feet = w.target.feet;
    let goal = match &mut w.target.mode {
        TargetMode::Static => None,
        TargetMode::Waypoints { pts, next, looped } => {
            let gp = pts[*next];
            if dist_xz(gp, feet) < 0.6 {
                if *next + 1 < pts.len() {
                    *next += 1;
                } else if *looped {
                    *next = 0;
                }
            }
            Some(pts[*next])
        }
        TargetMode::Wander { center, radius, goal } => {
            if dist_xz(*goal, feet) < 0.8 {
                // New seeded goal in the disc; keep trying until it lands on ground.
                for _ in 0..16 {
                    let ang = rng_f32(&mut w.rng) * std::f32::consts::TAU;
                    let r = rng_f32(&mut w.rng).sqrt() * *radius;
                    let cand = Vec3::new(center.x + ang.cos() * r, feet.y, center.z + ang.sin() * r);
                    if g.ground_height(cand.x, cand.z, feet.y + 20.0, STEP_UP).is_some() {
                        *goal = cand;
                        break;
                    }
                }
            }
            Some(*goal)
        }
    };
    let Some(goal) = goal else {
        w.target.vel = Vec3::ZERO;
        return;
    };
    let dir = Vec3::new(goal.x - feet.x, 0.0, goal.z - feet.z);
    if dir.length_squared() < 1e-6 {
        w.target.vel = Vec3::ZERO;
        return;
    }
    let stepv = dir.normalize() * speed * dt;
    let mut new = feet + stepv;
    // Wall slide (human capsule), then ground-follow with the walk camera's step allowance.
    let fixed = g.resolve_walls(Vec3::new(new.x, feet.y + 1.0, new.z), feet.y);
    new.x = fixed.x;
    new.z = fixed.y;
    match g.ground_height(new.x, new.z, feet.y, STEP_UP) {
        Some(y) => new.y = y,
        None => {
            // Void ahead (ledge): don't walk off; force a re-goal next tick for wander.
            if let TargetMode::Wander { goal, .. } = &mut w.target.mode {
                *goal = feet;
            }
            w.target.vel = Vec3::ZERO;
            return;
        }
    }
    // Blocked hard against a wall → wander picks a new goal so it doesn't grind forever.
    if dist_xz(new, feet) < 0.1 * speed * dt {
        if let TargetMode::Wander { goal, .. } = &mut w.target.mode {
            *goal = feet;
        }
    }
    w.target.vel = (new - feet) / dt;
    w.target.feet = new;
}

/// Observation: full drone kinematics + target kinematics in world AND camera frame, plus the
/// target's normalized image-plane coordinates under the session's pinhole model (fov_deg +
/// window aspect + FPV cam uptilt) — everything a tracking policy or reward needs.
fn obs(w: &AgentWorld) -> Value {
    let d = &w.drone;
    let cam_rot = d.quat * Quat::from_rotation_x(w.params.cam_tilt);
    let tcenter = w.target.center();
    let rel_world = tcenter - d.pos;
    let rel_cam = cam_rot.inverse() * rel_world;
    let depth = -rel_cam.z;
    let tanv = (w.fov_deg.to_radians() * 0.5).tan();
    let (ndc, in_fov) = if depth > 0.05 {
        let u = rel_cam.x / (depth * tanv * w.aspect);
        let v = rel_cam.y / (depth * tanv);
        (json!([u, v]), u.abs() <= 1.0 && v.abs() <= 1.0)
    } else {
        (Value::Null, false)
    };
    let vel_cam = cam_rot.inverse() * (w.target.vel - d.vel);
    let up = d.quat * Vec3::Y;
    let fwd = cam_rot * Vec3::NEG_Z;
    let oob = d.pos.y < w.bounds_min.y - 100.0;
    // Analog video-link quality: free-space range falloff from the reset/home point, attenuated
    // per wall/floor crossing on the home→drone line (same model as the viewer's VTX shader).
    let signal = {
        let dist = w.home.distance(d.pos);
        let mut s = (1.0 - (dist / w.range).powf(1.6)).clamp(0.0, 1.0);
        if let Some(g) = &w.grid {
            s *= 0.72f32.powi(g.segment_crossings(w.home, d.pos, 8) as i32);
        }
        s
    };
    json!({
        "ok": true,
        "t": w.t,
        "drone": {
            "pos": jvec(d.pos), "vel": jvec(d.vel),
            "quat": [d.quat.x, d.quat.y, d.quat.z, d.quat.w],
            "rate": [d.rate.x, d.rate.y, d.rate.z],
            "up": jvec(up), "cam_fwd": jvec(fwd),
        },
        "target": {
            "pos": jvec(tcenter), "feet": jvec(w.target.feet), "vel": jvec(w.target.vel),
            "rel_cam": jvec(rel_cam), "relvel_cam": jvec(vel_cam),
            "dist": rel_world.length(),
            "ndc": ndc, "in_fov": in_fov,
        },
        "collided": w.collided, "impact": w.impact,
        "crashed": d.crashed, "oob": oob, "signal": signal,
    })
}
