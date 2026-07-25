//! eft::drone — FPV quadcopter flight dynamics for the drone camera mode and the agent link.
//!
//! One physics core, two consumers:
//!  - `drone_move` (main.rs) flies it manually from keyboard+mouse in CamMode::Drone, and
//!  - `agent_link` steps it in LOCKSTEP over TCP so an external trainer can fly it.
//!
//! The model is a deliberately honest small-quad simulation (5" freestyle class), not an arcade
//! glide: real gravity (9.81, unlike the walk camera's game-feel 20), thrust only along body-up,
//! quadratic airframe drag (top speed ~50 m/s and dead-stick terminal fall ~25 m/s both emerge
//! from the same thrust/gravity/drag balance), first-order
//! motor/attitude response (commands are chased, never teleported), angle- and rate-(acro-)mode
//! control, and sphere collision against the map's ground/wall/ceiling grids with a crash
//! threshold on impact speed. What it does NOT model (documented in docs/AGENT_LINK.md): motor
//! torques/inertia tensor asymmetry, prop wash / ground effect, battery sag.
//!
//! Conventions: viewer world is Y-up, metres, right-handed; body frame = camera frame (+X right,
//! +Y up = thrust axis, -Z forward). Stick signs follow Mode-2 RC convention: pitch + = nose
//! down/forward, roll + = right, yaw + = turn right, throttle 0..1.

use bevy::prelude::*;

use crate::walk_ground::GroundData;

/// Hull radius (m) for collision — a 5" quad with props spans ~0.3 m.
pub const DRONE_RADIUS: f32 = 0.16;

/// Flight-dynamics parameters. Defaults model a ~680 g 5" freestyle quad.
#[derive(Clone, Copy, Debug)]
pub struct DroneParams {
    /// All-up weight (kg).
    pub mass: f32,
    /// Real gravity (m/s²).
    pub gravity: f32,
    /// Max total thrust (N). Default ≈ 4.2:1 thrust-to-weight.
    pub max_thrust: f32,
    /// Quadratic drag coefficient (N per (m/s)²). Sets BOTH the flat-out top speed
    /// (sqrt(max_thrust/drag_q) ≈ 50 m/s ≈ 180 km/h) and the dead-stick terminal fall speed
    /// (sqrt(mass·g/drag_q) ≈ 25 m/s, approached over several seconds — a cut quad must visibly
    /// keep accelerating as it drops, not plateau instantly).
    pub drag_q: f32,
    /// Linear drag (N per m/s) — rotor-disk drag, dominates at low speed.
    pub drag_l: f32,
    /// Max commanded body rates (rad/s): (pitch, yaw, roll). FPV-typical ~700°/s pitch/roll.
    pub max_rate: Vec3,
    /// First-order time constant (s) for actual PITCH/ROLL rate chasing the command — thrust
    /// differential is strong, so this is fast.
    pub rate_tau: f32,
    /// YAW rate time constant (s) — yaw comes from prop TORQUE only, visibly lazier than
    /// pitch/roll on every real quad.
    pub rate_tau_yaw: f32,
    /// Motor spool time constant (s): collective thrust chases the throttle command, it doesn't
    /// jump — punch-outs ramp, chops decay.
    pub thrust_tau: f32,
    /// Propwash gain (0 disables): attitude turbulence when descending into the prop's own wake
    /// at low horizontal airspeed — the signature FPV wobble on a falling half-throttle quad.
    pub propwash: f32,
    /// Angle mode: max tilt from level (rad).
    pub max_tilt: f32,
    /// Angle mode: attitude time constant (s).
    pub tilt_tau: f32,
    /// FPV camera uptilt (rad) used for the agent's target-projection math.
    pub cam_tilt: f32,
    /// Velocity kept along the surface tangent after a soft contact (0..1).
    pub restitution: f32,
    /// Impact speed (m/s, into the surface) above which the drone is CRASHED.
    pub crash_speed: f32,
}

impl Default for DroneParams {
    fn default() -> Self {
        Self {
            mass: 0.68,
            gravity: 9.81,
            max_thrust: 28.0,
            drag_q: 0.011,
            drag_l: 0.06,
            max_rate: Vec3::new(
                860f32.to_radians(),
                400f32.to_radians(),
                860f32.to_radians(),
            ),
            rate_tau: 0.045,
            rate_tau_yaw: 0.11,
            thrust_tau: 0.05,
            propwash: 1.0,
            max_tilt: 40f32.to_radians(),
            tilt_tau: 0.12,
            cam_tilt: 18f32.to_radians(),
            restitution: 0.25,
            crash_speed: 7.0,
        }
    }
}

impl DroneParams {
    /// Throttle fraction that exactly holds altitude when level.
    pub fn hover_throttle(&self) -> f32 {
        (self.mass * self.gravity / self.max_thrust).clamp(0.0, 1.0)
    }
}

/// How stick inputs map to attitude.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ControlMode {
    /// Sticks command tilt angles (self-levels, integrated yaw) — DJI-style, easy to fly/learn.
    Angle,
    /// Sticks command body rates directly (acro) — what real FPV pilots fly.
    Rates,
}

/// Normalized stick action. pitch/roll/yaw in [-1,1], throttle in [0,1].
#[derive(Clone, Copy, Debug, Default)]
pub struct DroneAction {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub throttle: f32,
}

impl DroneAction {
    pub fn clamped(self) -> Self {
        Self {
            roll: self.roll.clamp(-1.0, 1.0),
            pitch: self.pitch.clamp(-1.0, 1.0),
            yaw: self.yaw.clamp(-1.0, 1.0),
            throttle: self.throttle.clamp(0.0, 1.0),
        }
    }
}

/// Airframe state.
#[derive(Clone, Copy, Debug)]
pub struct DroneState {
    pub pos: Vec3,
    pub vel: Vec3,
    pub quat: Quat,
    /// Body rates (rad/s), axes (X=pitch, Y=yaw, Z=roll-carrier — see step()).
    pub rate: Vec3,
    /// Spooled collective thrust (normalized 0..1) — chases the throttle command (motor lag).
    pub thrust: f32,
    /// Sim-time accumulator (s) — phase source for the deterministic propwash oscillators.
    pub t: f32,
    /// Angle mode's integrated heading (rad, world yaw about +Y).
    pub yaw_cmd: f32,
    pub crashed: bool,
}

impl Default for DroneState {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            quat: Quat::IDENTITY,
            rate: Vec3::ZERO,
            thrust: 0.0,
            t: 0.0,
            yaw_cmd: 0.0,
            crashed: false,
        }
    }
}

impl DroneState {
    /// Level hover pose at `pos` facing world-yaw `yaw`.
    pub fn spawn(pos: Vec3, yaw: f32) -> Self {
        Self {
            pos,
            quat: Quat::from_axis_angle(Vec3::Y, yaw),
            yaw_cmd: yaw,
            ..default()
        }
    }
}

/// What one physics step reported back.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepOut {
    pub collided: bool,
    /// Peak speed INTO a surface this step (m/s); 0 when no contact.
    pub impact: f32,
}

/// Advance the airframe one substep. `wind` is the world-frame air velocity (drag acts on
/// airspeed, so a tailwind pushes). `grid` = the map collision grids; None = fly through
/// everything (grid still building).
pub fn step(
    st: &mut DroneState,
    p: &DroneParams,
    action: DroneAction,
    mode: ControlMode,
    wind: Vec3,
    grid: Option<&GroundData>,
    dt: f32,
) -> StepOut {
    let a = action.clamped();
    let mut out = StepOut::default();

    st.t += dt;

    // --- Motor spool: collective thrust chases the throttle command (punch-outs ramp) --------
    let th_cmd = if st.crashed { 0.0 } else { a.throttle };
    st.thrust += (th_cmd - st.thrust) * (1.0 - (-dt / p.thrust_tau).exp());

    // --- Propwash: descending into the prop's own wake at low horizontal airspeed makes real
    // quads wobble. Deterministic band-limited disturbance (incommensurate sine bank keyed off
    // sim time) scaled by descent-into-disk × slow-forward-flight × spooled thrust.
    let air = st.vel - wind;
    let up = st.quat * Vec3::Y;
    let wash = if p.propwash > 0.0 {
        let sink = (-air.dot(up)).max(0.0); // m/s falling through the disk
        let fwd_speed = (air - up * air.dot(up)).length();
        let i = (sink * 0.28).min(1.0) * (1.0 - fwd_speed / 9.0).max(0.0) * st.thrust.max(0.15);
        let t = st.t;
        Vec3::new(
            (t * 47.0).sin() + 0.6 * (t * 113.0 + 1.3).sin(),
            0.3 * ((t * 61.0 + 0.7).sin()),
            (t * 53.0 + 2.1).sin() + 0.6 * (t * 97.0 + 0.4).sin(),
        ) * (i * p.propwash * 2.4)
    } else {
        Vec3::ZERO
    };

    // --- Attitude ---------------------------------------------------------------------------
    // Stick sign → body rotation sign (right-handed, camera axes): nose-down = -X, turn-right =
    // -Y, roll-right = -Z (positive Z rotation lifts the right side = roll LEFT).
    match mode {
        ControlMode::Rates => {
            let target = Vec3::new(
                -a.pitch * p.max_rate.x,
                -a.yaw * p.max_rate.y,
                -a.roll * p.max_rate.z,
            );
            // Pitch/roll respond via thrust differential (fast); yaw only via prop torque (lazy).
            let a_rp = 1.0 - (-dt / p.rate_tau).exp();
            let a_y = 1.0 - (-dt / p.rate_tau_yaw).exp();
            st.rate.x += (target.x - st.rate.x) * a_rp;
            st.rate.z += (target.z - st.rate.z) * a_rp;
            st.rate.y += (target.y - st.rate.y) * a_y;
            // Propwash lands ON the gyro (st.rate) as a torque impulse scaled by dt/rate_tau —
            // substep-size invariant (steady-state gyro wobble ≈ the wash amplitude itself),
            // and the next substep's chase fights it: the closed loop oscillates, like a real
            // quad's PID in its own wake.
            st.rate += wash * (dt / p.rate_tau);
            st.quat = (st.quat * Quat::from_scaled_axis(st.rate * dt)).normalize();
            // Keep yaw_cmd tracking reality so a mode switch back to Angle doesn't snap.
            let fwd = st.quat * Vec3::NEG_Z;
            if fwd.x * fwd.x + fwd.z * fwd.z > 1e-6 {
                st.yaw_cmd = (-fwd.x).atan2(-fwd.z);
            }
        }
        ControlMode::Angle => {
            st.yaw_cmd -= a.yaw * p.max_rate.y * dt;
            let desired = Quat::from_axis_angle(Vec3::Y, st.yaw_cmd)
                * Quat::from_axis_angle(Vec3::X, -a.pitch * p.max_tilt)
                * Quat::from_axis_angle(Vec3::Z, -a.roll * p.max_tilt);
            let alpha = 1.0 - (-dt / p.tilt_tau).exp();
            let prev = st.quat;
            st.quat = (prev.slerp(desired, alpha) * Quat::from_scaled_axis(wash * dt)).normalize();
            // Report the achieved body rate (for observations / mode-switch continuity).
            st.rate = (prev.inverse() * st.quat).to_scaled_axis() / dt.max(1e-6);
        }
    }

    // --- Translation ------------------------------------------------------------------------
    let thrust_dir = st.quat * Vec3::Y;
    let thrust = st.thrust * p.max_thrust;
    let drag = -(air * (air.length() * p.drag_q + p.drag_l));
    let accel = thrust_dir * (thrust / p.mass) + drag / p.mass - Vec3::Y * p.gravity;
    st.vel += accel * dt; // semi-implicit Euler
    st.pos += st.vel * dt;

    // --- Collision --------------------------------------------------------------------------
    if let Some(g) = grid {
        let (fixed, hit) = g.resolve_sphere(st.pos, DRONE_RADIUS);
        if let Some(n) = hit {
            let vn = st.vel.dot(n);
            if vn < 0.0 {
                out.impact = -vn;
                st.vel -= n * vn * (1.0 + p.restitution);
                // Ground friction-ish damping on the tangential remainder so a downed quad
                // doesn't ice-skate forever.
                st.vel *= 0.92;
            }
            st.pos = fixed;
            out.collided = true;
            if out.impact > p.crash_speed {
                st.crashed = true;
            }
        }
    }
    out
}

/// Betaflight-style rate curve: stick deflection in [-1,1] → commanded body rate (rad/s).
/// `rc_rate` scales the center sensitivity (200°/s per unit at full stick, BF convention),
/// `expo` softens the center (x·|x|³ blend), `super_rate` sharpens the ends
/// (1/(1−|x|·super) blow-up). Defaults 1.0 / 0.0 / 0.7 ≈ a stock freestyle profile (~667°/s).
pub fn bf_rate(stick: f32, rc_rate: f32, expo: f32, super_rate: f32) -> f32 {
    let x = stick.clamp(-1.0, 1.0);
    let xe = x * x.abs().powi(3) * expo + x * (1.0 - expo);
    let mut deg = 200.0 * rc_rate * xe;
    let s = super_rate.clamp(0.0, 0.99);
    if s > 0.0 {
        deg *= 1.0 / (1.0 - xe.abs() * s);
    }
    deg.to_radians()
}

/// Per-camera manual-flight rig (lives on the CullCamera entity next to WalkState). `live=false`
/// means "respawn at the camera pose on the next drone-mode frame" — mode switches and map swaps
/// just clear it.
#[derive(Component, Default)]
pub struct DroneRig {
    pub live: bool,
    pub state: DroneState,
    /// Last spawn pose, for R-reset and the fell-out-of-world backstop.
    pub spawn_pos: Vec3,
    pub spawn_yaw: f32,
    /// Low-passed keyboard "virtual sticks" (roll, pitch, yaw) — keys are ±1 square waves; a
    /// ~90 ms rise turns them into flyable ramps. Gamepad axes bypass this.
    pub kb_stick: Vec3,
    /// Acro-mode throttle STICK position (0..1) — keyboard ramps it, gamepad sets it directly.
    pub throttle: f32,
}
