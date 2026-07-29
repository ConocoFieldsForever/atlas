//! eft::drone — FPV quadcopter flight dynamics for the drone camera mode and the agent link.
//!
//! One physics core, two consumers:
//!  - `drone_move` (main.rs) flies it manually from keyboard+mouse in CamMode::Drone, and
//!  - `agent_link` steps it in LOCKSTEP over TCP so an external trainer can fly it.
//!
//! The model is a deliberately honest small-quad simulation (5" freestyle class), not an arcade
//! glide: real gravity (9.81, unlike the walk camera's game-feel 20), thrust only along body-up,
//! ANISOTROPIC body-frame airframe drag (the prop-disk axis has ~6× the frontal area of the
//! edge-on axes, so top speed ~26 m/s and dead-stick terminal fall ~12.5 m/s emerge from the same
//! thrust/gravity/drag balance without fighting each other), first-order
//! motor/attitude response (commands are chased, never teleported), angle- and rate-(acro-)mode
//! control, and sphere collision against the map's ground/wall/ceiling grids with a crash
//! threshold on impact speed. What it does NOT model (documented in docs/AGENT_LINK.md): motor
//! torques/inertia tensor asymmetry, ground effect, battery sag. (Propwash IS modelled — see
//! `propwash` below.)
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
    /// Parasitic quadratic drag ACROSS the prop disks — body +Y, the airframe's high-drag axis
    /// (four 5" disks + the frame's plan area ≈ 0.06 m², Cd ≈ 1.15). Sets the dead-stick terminal
    /// fall, sqrt(mass·g/drag_q_axial) ≈ 12.5 m/s, approached over several seconds — a cut quad
    /// must visibly keep accelerating as it drops, not plateau instantly. It also sets the flat-out
    /// top speed (≈ 26 m/s ≈ 95 km/h), because holding altitude at speed forces ~50° of pitch and
    /// that swings this axis into the airflow.
    pub drag_q_axial: f32,
    /// Parasitic quadratic drag EDGE-ON to the disks — body X (lateral) and Z (fore/aft), where
    /// only the frame edge faces the flow (≈ 0.01 m², ~6× less area than the disk axis). This is
    /// the axis a level quad coasts on after a throttle chop, so it — not `drag_q_axial` — governs
    /// how far a dead-stick quad carries its forward momentum.
    pub drag_q_side: f32,
    /// Rotor H-force at FULL throttle (N per m/s): the spinning discs' in-plane drag, linear in
    /// airspeed. Scales with rpm (≈ sqrt of the spooled thrust) and acts only in the disk PLANE
    /// (body X/Z), which is where the H-force physically points — so it fades out with the motors
    /// and leaves a dead-stick quad nothing extra to brake against.
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
    /// Fraction of the impact (surface-normal) speed thrown back on a bounce: 0 = stick, 1 =
    /// perfectly elastic.
    pub restitution: f32,
    /// Time constant (s) for tangential speed rubbing off while in contact, so a downed quad
    /// skids to a stop in ~0.5 s instead of ice-skating. Applied as exp(-dt/τ), which makes ground
    /// friction SUBSTEP-INVARIANT — the same contact costs the same speed at 30 fps and 240 fps.
    pub contact_tau: f32,
    /// Impact speed (m/s, into the surface) above which the drone is CRASHED.
    pub crash_speed: f32,
}

impl Default for DroneParams {
    fn default() -> Self {
        Self {
            mass: 0.68,
            gravity: 9.81,
            max_thrust: 28.0,
            drag_q_axial: 0.043,
            drag_q_side: 0.0075,
            drag_l: 0.12,
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
            contact_tau: 0.15,
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
        // Scales with the SPOOLED thrust with no floor: wash is the props' own wake, so motors
        // that have spooled down have no wake to fall into and a dead-stick quad rides smooth.
        let i = (sink * 0.28).min(1.0) * (1.0 - fwd_speed / 9.0).max(0.0) * st.thrust;
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
    // Parasitic drag is ANISOTROPIC and lives in the BODY frame — the standard multirotor form
    // D = -C·|v|·v with C = diag(side, axial, side). A quad presents ~6× the area through its prop
    // disks as it does edge-on, so a single isotropic coefficient cannot set the dead-stick fall
    // and the forward coast at once: tuning it for the fall makes a cut quad brake absurdly hard,
    // and tuning it for the coast makes it drop like a stone. Orientation now decides, which is
    // also why nosing over to shed drag finally does something.
    // On top of that, the rotors' in-plane H-force: linear in airspeed, proportional to rpm
    // (≈ sqrt of spooled thrust), and confined to the disk plane — it dies with the motors, so a
    // dead-stick quad coasts on frame drag alone.
    let air_b = st.quat.inverse() * air;
    let speed = air.length();
    let rotor_h = p.drag_l * st.thrust.max(0.0).sqrt();
    let drag_b = -Vec3::new(
        air_b.x * (speed * p.drag_q_side + rotor_h),
        air_b.y * (speed * p.drag_q_axial),
        air_b.z * (speed * p.drag_q_side + rotor_h),
    );
    let drag = st.quat * drag_b;
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
                st.vel = resolve_contact(st.vel, n, p, dt);
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

/// Resolve one surface contact: bounce the normal component off `restitution`, rub the tangential
/// remainder off on `contact_tau` so a downed quad doesn't ice-skate forever.
///
/// The tangential decay MUST be exponential in `dt`. A flat per-call multiplier ties ground
/// friction to the substep count and therefore to framerate, and since gravity re-presses the hull
/// into the surface on every substep it fires ~1000×/s — enough to erase all horizontal speed the
/// instant a skid begins. Split out from `step` so that invariance is directly testable.
#[inline]
fn resolve_contact(vel: Vec3, n: Vec3, p: &DroneParams, dt: f32) -> Vec3 {
    let vn = vel.dot(n);
    let tangent = (vel - n * vn) * (-dt / p.contact_tau).exp();
    tangent - n * (vn * p.restitution)
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

/// Pins the EMERGENT numbers of the flight model — the ones a pilot actually feels — rather than
/// the coefficients, which are only a means to them. Every figure below is quoted for a ~680 g 5"
/// freestyle quad and can be checked against a real one.
#[cfg(test)]
mod tests {
    use super::*;

    /// Params with propwash off, so the drag tests measure drag and nothing else.
    fn quiet() -> DroneParams {
        DroneParams { propwash: 0.0, ..DroneParams::default() }
    }

    /// Fly with no collision grid for `secs` at a true 1 kHz, holding `action`.
    fn fly(st: &mut DroneState, p: &DroneParams, action: DroneAction, secs: f32) {
        for _ in 0..(secs / 0.001) as u32 {
            step(st, p, action, ControlMode::Rates, Vec3::ZERO, None, 0.001);
        }
    }

    /// Motors dead, belly-flat: the quad falls through its own prop disks, the airframe's
    /// highest-drag axis. Real 5" quads settle at 12-14 m/s. The old isotropic model gave 22 —
    /// and that excess sink is what dragged forward speed off a coasting quad, since quadratic
    /// drag couples the axes through |air|.
    #[test]
    fn dead_stick_terminal_fall_is_realistic() {
        let p = quiet();
        let mut st = DroneState::default();
        fly(&mut st, &p, DroneAction::default(), 15.0);
        let fall = -st.vel.y;
        assert!((11.0..14.0).contains(&fall), "flat-fall terminal {fall:.1} m/s, want 11-14");
    }

    /// THE regression this model exists to fix: chop the throttle at 25 m/s and the quad must
    /// coast, not brake. The old isotropic drag left 7.7 m/s after 3 s; anisotropic leaves ~13.
    #[test]
    fn cut_power_keeps_forward_momentum() {
        let p = quiet();
        let mut st = DroneState { vel: Vec3::new(0.0, 0.0, -25.0), thrust: p.hover_throttle(), ..default() };
        fly(&mut st, &p, DroneAction::default(), 3.0);
        let fwd = -st.vel.z;
        assert!(fwd > 12.0, "kept only {fwd:.1} m/s of 25 after a 3 s dead-stick coast");
        assert!(fwd < 18.0, "kept {fwd:.1} m/s — drag has gone too soft to feel like an airframe");
    }

    /// Drag must depend on ATTITUDE: the same airspeed costs far more through the disks (body +Y)
    /// than edge-on. Without this the whole model collapses back to one isotropic coefficient and
    /// nosing over to shed drag does nothing.
    #[test]
    fn drag_is_orientation_dependent() {
        let p = quiet();
        let v = Vec3::new(0.0, 0.0, -25.0);
        let mut edge = DroneState { vel: v, ..default() };
        // Pitched 90° nose-up puts body +Y — the prop-disk axis — straight into the airflow.
        let mut disk =
            DroneState { vel: v, quat: Quat::from_axis_angle(Vec3::X, -std::f32::consts::FRAC_PI_2), ..default() };
        fly(&mut edge, &p, DroneAction::default(), 0.25);
        fly(&mut disk, &p, DroneAction::default(), 0.25);
        let (lost_edge, lost_disk) = (25.0 + edge.vel.z, 25.0 + disk.vel.z);
        assert!(
            lost_disk > lost_edge * 3.0,
            "disk-on lost {lost_disk:.2} m/s vs edge-on {lost_edge:.2} — attitude barely matters"
        );
    }

    /// Ground friction must not depend on the host's framerate. One 16 ms contact has to cost the
    /// same tangential speed as sixteen 1 ms ones; the old per-call 0.92 multiplier made a 30 fps
    /// machine ~4× grippier than a 240 fps one.
    #[test]
    fn contact_friction_is_substep_invariant() {
        let p = quiet();
        let (n, slide) = (Vec3::Y, Vec3::new(10.0, 0.0, 0.0));
        let coarse = resolve_contact(slide, n, &p, 0.016);
        let mut fine = slide;
        for _ in 0..16 {
            fine = resolve_contact(fine, n, &p, 0.001);
        }
        assert!(
            (coarse.x - fine.x).abs() < 0.01,
            "16 ms in one step leaves {:.3} m/s but in sixteen leaves {:.3}",
            coarse.x,
            fine.x
        );
    }

    /// The other end of the drag calibration. `drag_q_axial` is pinned by the terminal fall, but it
    /// also caps top speed, because holding altitude at speed forces the quad to pitch far over and
    /// swing its disks into the airflow — so once the fall is pinned this has to come out plausible
    /// on its own or the anisotropy is wrong. It lands at ~96 km/h (52° of pitch): the conservative
    /// end of the real 5" range, which runs ~100-130 km/h for a fast build. The old isotropic model
    /// let it run to 169 km/h, which no 680 g freestyle quad does.
    #[test]
    fn top_speed_stays_in_the_real_5in_envelope() {
        let p = quiet();
        // Full throttle, hold a pitch, run to steady state. Bisect for the pitch whose steady
        // state exactly holds altitude — that is the max SUSTAINED level speed.
        let steady = |pitch: f32| {
            let q = Quat::from_axis_angle(Vec3::X, -pitch);
            let mut st = DroneState { quat: q, thrust: 1.0, ..default() };
            for _ in 0..40000 {
                step(&mut st, &p, DroneAction { throttle: 1.0, ..default() }, ControlMode::Rates, Vec3::ZERO, None, 0.001);
                st.quat = q; // hold the attitude
            }
            st.vel
        };
        let (mut lo, mut hi) = (0.05f32, 1.5f32);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if steady(mid).y > 0.0 { lo = mid } else { hi = mid }
        }
        let top = -steady(0.5 * (lo + hi)).z;
        assert!(
            (22.0..32.0).contains(&top),
            "flat-out level speed {top:.1} m/s ({:.0} km/h), want 22-32 (80-115 km/h)",
            top * 3.6
        );
    }

    /// Propwash is the props' own wake, so motors that have spooled down have none to fall into.
    /// A dead-stick quad must ride smooth even while sinking hard — the condition that otherwise
    /// maximises wash.
    #[test]
    fn propwash_dies_with_the_motors() {
        let p = DroneParams::default(); // propwash ON — that is the point
        let sinking = |thrust: f32| {
            let mut st = DroneState { vel: Vec3::new(0.0, -8.0, 0.0), thrust, ..default() };
            let mut peak: f32 = 0.0;
            for _ in 0..500 {
                step(&mut st, &p, DroneAction { throttle: thrust, ..default() }, ControlMode::Rates, Vec3::ZERO, None, 0.001);
                peak = peak.max(st.rate.length());
            }
            peak
        };
        assert!(sinking(0.0) < 1e-3, "dead motors still shook the airframe");
        assert!(sinking(0.5) > 0.5, "half-throttle descent lost its propwash entirely");
    }
}
