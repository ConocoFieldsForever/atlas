# Agent Link — standardized drone-sim control protocol

The viewer embeds an FPV quadcopter simulation (`viewer/src/drone.rs`) flying against the real
extracted map collision geometry, plus a simulated walking ground target. The **agent link**
exposes that sim over a tiny lockstep protocol so external programs — RL trainers, scripted
policies, notebooks — can fly the drone and practice **target-tracking** tasks reproducibly.
This is a *game-world simulation interface*; positions are viewer-world metres (Y-up) on the
loaded `.eftpack` map.

- Transport: **newline-delimited JSON over TCP, 127.0.0.1 only**. One client at a time.
  Every request line gets exactly one reply line.
- Enable: camera tab ▸ *AGENT LINK* checkbox, or launch with `EFT_AGENT=7878` (value = port).
- **Lockstep**: the sim advances **only** inside a `step` request — fixed `dt`, fixed 5 ms
  physics substeps, seeded RNG. Training speed is bounded by the socket, not the render loop
  (hundreds of steps/s), and runs are reproducible for a given seed + action sequence.
- The viewer visualizes a live session: red gizmo = target, cyan = drone; in **Drone FPV**
  camera mode with *Spectate* on, the camera rides the agent's drone.
- A map switch invalidates the session (`reset` again once the new grid is built).
- Python wrapper: `tools/drone_env.py` (`AgentLink` raw client + Gymnasium-compatible
  `EftDroneEnv`; `python tools/drone_env.py` is a smoke test).

## Commands

### `hello`
`{"cmd":"hello"}` →
`{"ok":true,"proto":1,"app":"atlas","map":"interchange","grid_ready":true,"bounds":[[...],[...]],"params":{...}}`

`grid_ready:false` means the collision grid is still building (or no map open) — `reset` will fail
until it flips true.

### `reset`
```json
{"cmd":"reset", "seed":7, "dt":0.02, "control":"rates",
 "drone":{"pos":[x,y,z], "yaw":0.0},
 "target":{"mode":"wander", "speed":1.8, "radius":30.0},
 "wind":[wx,wz], "action_noise":0.0, "latency":0, "fov_deg":90}
```
All fields optional except `cmd`. Defaults: seeded fixed, `dt=0.02` (50 Hz control), `rates`
control, drone spawned 12 m above the highest surface at map centre, target grounded 8–30 m away.
Reply = first observation.

- `control`: `"rates"` (acro: sticks command body rates — realistic FPV) or `"angle"`
  (self-leveling: sticks command tilt; much easier for a first policy).
- `target.mode`:
  - `"static"` — stands still (`pos` optional).
  - `"wander"` — seeded random walk on the ground within `radius` m of its spawn; walks floors,
    slides along walls, won't step off ledges. `speed` m/s (default 1.8, human walk).
  - `"waypoints"` — patrols `points:[[x,y,z],…]` (`loop:true` default).
- **Domain-randomization knobs** (the standard sim-to-real levers):
  `wind` = constant world XZ air velocity (m/s, drag acts on airspeed);
  `action_noise` = uniform ±noise added to each effective stick channel;
  `latency` = actuation delay in whole control ticks (command reaches the motors N ticks late).
- `fov_deg`: vertical FOV of the *observation* pinhole model (with the live window aspect and the
  FPV cam uptilt) used for `ndc` / `in_fov`.

### `step`
`{"cmd":"step","action":[roll,pitch,yaw,throttle],"repeat":1}` → observation.

Sticks in `[-1,1]`, throttle in `[0,1]` (`params.hover_throttle` ≈ 0.24 hovers when level).
Mode-2 signs: pitch + = nose forward/down, roll + = right, yaw + = turn right. `repeat` applies
the same action for N control ticks (frame-skip); `collided`/`impact` aggregate over them.

### `obs` / `params` / `spectate` / `frame` / `bye`
- `{"cmd":"obs"}` — current observation without stepping.
- `{"cmd":"params"}` — flight-model constants (mass, thrust, drag, rates, taus, hull radius…).
- `{"cmd":"spectate","on":true}` — camera follows the agent drone (viewer must be in Drone mode).
- `{"cmd":"frame","path":"C:\\out\\f.png"}` — queue an async window screenshot (poll the file).
  It captures the *window* (UI included) — a debug/imitation aid, not a per-step pixel pipe.
  The fast path is the state observation.
- `{"cmd":"bye"}` — polite close.

## Observation

```json
{"ok":true, "t":1.24,
 "drone": {"pos":[..],"vel":[..],"quat":[x,y,z,w],"rate":[..],"up":[..],"cam_fwd":[..]},
 "target":{"pos":[..],"feet":[..],"vel":[..],
           "rel_cam":[..],"relvel_cam":[..],"dist":9.7,
           "ndc":[u,v] | null, "in_fov":true},
 "collided":false, "impact":0.0, "crashed":false, "oob":false}
```

- Camera frame = body frame tilted up by the FPV cam angle: +X right, +Y up, **-Z forward**
  (`rel_cam[2] < 0` ⇒ target ahead). `ndc` is the target centre on the image plane, `[-1,1]`
  each axis; `null` when behind the camera.
- `target.pos` is the torso centre (feet + 0.9 m); the target is a human-sized proxy.
- `collided` = any contact this step; `impact` = peak speed into the surface (m/s);
  `crashed` = impact exceeded `params.crash_speed` (thrust dead — episode over, send `reset`);
  `oob` = fell far below the map bounds.

## Flight model (and its honest limits)

Modeled: real gravity 9.81; thrust only along body-up (max ≈ 4.2× weight — 5" freestyle class)
with a **motor-spool lag** (`thrust_tau` 50 ms — punch-outs ramp); **anisotropic body-frame
airframe drag** (the standard multirotor form `D = -C·|v|·v`, `C = diag(side, axial, side)`): the
prop-disk axis presents ~6× the frontal area of the edge-on axes, so `drag_q_axial` sets the
dead-stick terminal fall ≈ 12.5 m/s *and* — because holding altitude at speed forces ~50° of pitch,
swinging that axis into the flow — the flat-out top speed ≈ 26 m/s, while the much smaller
`drag_q_side` governs how far a cut quad carries its forward momentum. Attitude therefore changes
drag, so nosing over to shed it works. On top sits the rotors' in-plane H-force (`drag_l`, linear
in airspeed, ∝ rpm ≈ √thrust, disk-plane only), which fades out with the motors; **per-axis**
first-order rate response —
pitch/roll via thrust differential (`rate_tau` 45 ms), yaw via prop torque only
(`rate_tau_yaw` 110 ms, visibly lazier, like every real quad); **propwash** — deterministic
band-limited attitude turbulence when descending into the prop's own wake at low forward speed
(`propwash` reset knob, 0 disables; scales with spooled thrust, so a dead-stick quad rides smooth);
angle & acro control; sphere collision (r = 0.16 m) against the map's **ground + wall + ceiling**
triangle grids, with a normal-direction `restitution`, tangential contact friction on a time
constant (`contact_tau` — substep-invariant, so ground friction does not vary with the host's
framerate) and a crash threshold; constant wind, action noise, actuation latency.

Agent actions map **linearly** to body-rate commands (±`max_rate`); the Betaflight rate curve
(RC rate / expo / super rate) is a manual-flight input shaping and does not apply to the wire
protocol — apply your own curve client-side if you want stick-feel parity with a human pilot.

Not modeled (yet): per-motor torques / inertia tensor, ground effect, battery sag, gusting wind,
target line-of-sight occlusion (`in_fov` is geometric only — a wall between drone and target
does not blank it; raycast LOS is a natural follow-up), and per-step pixel observations
(state-vector obs by design; use `frame` sparingly for imitation/debug).

## Determinism

Same map pack + same `reset` payload (incl. `seed`) + same action sequence ⇒ same trajectory
(fixed-dt f32 physics, xorshift64* RNG, no wall-clock anywhere; propwash is a sim-time sine
bank, not random). Physics substeps are ≤2.5 ms regardless of `dt`, so changing `dt` changes
the control rate, not the integration. `dt` down to **0.001 (a true 1 kHz loop)** is supported
for training low-level rate controllers; the default 0.02 (50 Hz) is right for tracking policies.
