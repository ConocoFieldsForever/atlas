//! Driving the character from the walk camera, and the third-person boom.
//!
//! SCOPE: this does not run EFT's Animator. It maps the viewer's existing [`WalkState`] onto the
//! animator PARAMETERS the extracted graph is steered by, picks one state with a small explicit
//! machine, and evaluates that state's real blend tree. So the poses and the blend geometry are the
//! game's; the state selection is ours. That boundary is deliberate — reproducing 13 layers with
//! additive aiming and synced layers is a separate project, and pretending otherwise would make the
//! difference invisible.
//!
//! ## Why the camera boom is removed in `PreUpdate`
//!
//! `walk_move` (main.rs) treats the camera transform AS the player's eye: it reads the translation,
//! runs ground/wall physics on it, and writes it back. If a third-person offset were left in that
//! transform, the next frame's physics would run from the boom position and the player would slide
//! backwards forever. So the offset is removed in `PreUpdate` (which always precedes `Update`, where
//! `walk_move` lives) and re-applied in `PostUpdate` before transform propagation. This mirrors the
//! pattern `walk_ground` already uses for head bob — `tf.translation.y -= ws.last_bob` at the top of
//! the frame — and needs no changes to `walk_move` itself.

use super::anim::{accumulate_clip, eval_tree, PoseAccumulator, WeightedClip};
use super::pack::CharacterPack;
use super::rig::{CharacterRoot, CharacterMesh};
use super::{ActiveCharacter, CharacterSettings};
use crate::render::CullCamera;
use crate::walk_ground::{self, WalkState};
use crate::{CamMode, CameraSettings};
use bevy::prelude::*;
use std::collections::HashMap;

/// State paths in Tagilla's `TagillaBotAnimController`. Named here rather than in the pack because
/// WHICH state to play for a given movement situation is this module's decision, not data.
/// A missing path degrades to the idle state, so a character whose graph uses different paths is
/// visibly wrong rather than crashing.
pub(crate) mod states {
    pub const IDLE: &str = "Base Layer.Stand.Idle_Aim";
    pub const MOVE: &str = "Base Layer.StateMachine_Move.MOVE";
    pub const JUMP_IDLE: &str = "Base Layer.JUMP.Jump_Idle";
    pub const JUMP_MOVE: &str = "Base Layer.JUMP.Jump_Move";
    /// The airborne hold AFTER a jump's launch. EFT has a dedicated state for this
    /// (`idle_jump_loop`); routing a jump's descent through `Fall` instead is what made the apex
    /// read as a snap, since `Fall` is a 2-frame near-static pose meant for dropping off ledges.
    pub const JUMP_LOOP: &str = "Base Layer.JUMP.Jump_Loop";
    /// Falling WITHOUT having jumped — walked off an edge.
    pub const FALL: &str = "Base Layer.JUMP.Fall";
    pub const LAND_IDLE: &str = "Base Layer.JUMP.Land_Idle";
    pub const LAND_MOVE: &str = "Base Layer.JUMP.Land_Move";
}

/// Speed below which the character is considered standing still (m/s).
const MOVE_EPSILON: f32 = 0.15;
/// How long a landing state plays before handing back to idle/move (s).
const LAND_TIME: f32 = 0.25;
/// Cross-fade used when the graph has no transition for a pair our state machine invents.
const DEFAULT_FADE: f32 = 0.15;
/// Clamp on graph-declared fades: 0 would be a cut, and a very long one smears locomotion.
const FADE_RANGE: (f32, f32) = (0.05, 0.45);

/// Per-camera third-person offset bookkeeping. `applied` is what was added to the camera transform
/// last frame and must be subtracted before physics reads it again.
#[derive(Component, Default)]
pub struct CameraBoom {
    pub applied: Vec3,
}

/// `PreUpdate`: hand the true eye position back to the physics in `walk_move`.
pub fn unboom_camera(mut q: Query<(&mut Transform, &mut CameraBoom), With<CullCamera>>) {
    for (mut tf, mut boom) in &mut q {
        if boom.applied != Vec3::ZERO {
            tf.translation -= boom.applied;
            boom.applied = Vec3::ZERO;
        }
    }
}

/// Extract the yaw of a camera transform (rotation about +Y), ignoring pitch.
///
/// Bevy's forward is the transform's -Z. `walk_move` builds its movement basis as
/// `fwd = (-sin yaw, 0, -cos yaw)`, so inverting that is `yaw = atan2(-fwd.x, -fwd.z)` and the
/// character's facing stays consistent with the direction WASD actually pushes.
fn yaw_of(tf: &Transform) -> f32 {
    let f = tf.forward();
    (-f.x).atan2(-f.z)
}

fn yaw_of_dir(d: Vec3) -> f32 {
    (-d.x).atan2(-d.z)
}

/// Build the animator parameter set from the walk state.
///
/// `Direct_X` / `Direct_Y` are the movement direction expressed in BODY-LOCAL space, which is what
/// makes the 9-way directional blend do anything: the body faces where you look, and strafing then
/// selects the sidestep quadrants exactly as it does in game.
fn build_params(
    ws: &WalkState,
    settings: &CameraSettings,
    body_yaw: f32,
    landing: bool,
    aiming: f32,
) -> (HashMap<String, f32>, f32) {
    let v = ws.horizontal_velocity;
    let speed = v.length();

    // World -> body-local. Body forward at `body_yaw` is (-sin, -cos); right is (cos, -sin).
    let (sy, cy) = body_yaw.sin_cos();
    let fwd = Vec2::new(-sy, -cy);
    let right = Vec2::new(cy, -sy);
    let (dx, dy) = if speed > 1e-4 {
        let n = v / speed;
        (n.dot(right), n.dot(fwd))
    } else {
        (0.0, 0.0)
    };

    let mut p = HashMap::new();
    p.insert("Direct_X".into(), dx);
    p.insert("Direct_Y".into(), dy);
    p.insert("Direct".into(), dy);
    // `Speed` is EFT's NORMALISED gait dial (its own SPEED_MIN/SPEED_MAX consts are 0..1), not m/s.
    let norm = (speed / walk_ground::MAX_WALK_SPEED).clamp(0.0, 1.0);
    p.insert("Speed".into(), if ws.sprinting { 1.0 } else { norm });
    p.insert("SprintSpeed".into(), if ws.sprinting { 1.0 } else { 0.0 });
    p.insert("Sprint".into(), if ws.sprinting { 1.0 } else { 0.0 });
    p.insert("SprintInertia".into(), 0.0);
    // Pose level: 1 = standing. EFT's own constants are PRONE_POSE 0, CROUCH_POSE 0.5. The walk
    // camera has no crouch/prone input yet, so this is pinned standing rather than guessed.
    p.insert("Level".into(), 1.0);
    p.insert("Prone".into(), 0.0);
    p.insert("Tilt".into(), 0.0);
    p.insert("RotateSpeed".into(), 0.0);
    p.insert("IsJumping".into(), if !ws.grounded && ws.vy > 0.0 { 1.0 } else { 0.0 });
    p.insert("FallingDown".into(), if !ws.grounded && ws.vy < 0.0 { 1.0 } else { 0.0 });
    p.insert("Landing".into(), if landing { 1.0 } else { 0.0 });
    p.insert("InertFloat".into(), norm);
    p.insert("SidestepFloat".into(), dx.abs());
    p.insert("SidebackSpeed".into(), 1.0);
    // Weapon-driven lanes: Tagilla's blends branch on these and 0 selects his base (hammer) set.
    p.insert("WeaponTypeFloat".into(), 0.0);
    p.insert("Weapon_3rd".into(), 0.0);
    p.insert("ThirdPersonFloat".into(), 1.0);
    p.insert("WeapSizeModifier".into(), 0.0);
    p.insert("TransitionMultiplier".into(), 1.0);
    // The game's own aim signal. Set so the graph sees the right value; the pose it selects lives
    // on the additive aim LAYERS, which the evaluator does not blend yet.
    p.insert("isAiming".into(), aiming);
    p.insert("Aim_angle".into(), 0.0);
    let _ = settings;
    (p, speed)
}

/// Pick the state to play. Explicit and small on purpose — see the module doc.
///
/// `jumped` distinguishes a jump from walking off a ledge. It matters: a jump's descent belongs in
/// `Jump_Loop`, not `Fall`. Previously the apex (where `vy` crosses zero) cut straight from the
/// launch clip into `Fall`, a 2-frame static pose — which is the "snaps to a default position at the
/// apex" symptom.
fn select_state(ws: &WalkState, speed: f32, land_timer: f32, jumped: bool) -> &'static str {
    let moving = speed > MOVE_EPSILON;
    if !ws.grounded {
        if !jumped {
            return states::FALL;
        }
        return if ws.vy > 0.0 {
            if moving { states::JUMP_MOVE } else { states::JUMP_IDLE }
        } else {
            states::JUMP_LOOP
        };
    }
    if land_timer > 0.0 {
        return if moving { states::LAND_MOVE } else { states::LAND_IDLE };
    }
    if moving {
        states::MOVE
    } else {
        states::IDLE
    }
}

/// Resolve a state's blend tree to weighted clips, plus the state's playback speed.
pub(crate) fn gather(
    pack: &CharacterPack,
    state_path: &str,
    params: &HashMap<String, f32>,
    out: &mut Vec<WeightedClip>,
) -> f32 {
    out.clear();
    let Some(st) = pack.state(state_path).or_else(|| pack.state(states::IDLE)) else {
        return 1.0;
    };
    if let Some(Some(tree)) = st.trees.first() {
        eval_tree(tree, params, out);
    }
    st.speed
}

/// Blend-weighted root speed, for rate-matching playback so feet do not skate.
pub(crate) fn blended_root_speed(pack: &CharacterPack, leaves: &[WeightedClip]) -> f32 {
    leaves
        .iter()
        .filter_map(|l| pack.clip_by_controller_id(l.clip_id).map(|c| c.root_speed * l.weight))
        .sum()
}

/// `PostUpdate`: place the character at the player's feet, pose it, then apply the boom.
pub fn drive_character(
    time: Res<Time>,
    aim_blend: Res<AimBlend>,
    settings: Res<CameraSettings>,
    cs: Res<CharacterSettings>,
    active: Option<Res<ActiveCharacter>>,
    grid: Option<Res<walk_ground::GroundGrid>>,
    mut land_timer: Local<f32>,
    mut was_airborne: Local<bool>,
    mut jumped: Local<bool>,
    mut acc: Local<Option<PoseAccumulator>>,
    // These three touch `&mut Transform` and Bevy must be able to PROVE they cannot overlap, hence
    // the mutual `Without` filters rather than relying on "a bone is never the root in practice".
    mut cam: Query<(&mut Transform, &WalkState, &mut CameraBoom), With<CullCamera>>,
    mut root_q: Query<
        (&mut CharacterRoot, &mut Transform, &mut Visibility),
        (Without<CullCamera>, Without<super::rig::CharacterBone>),
    >,
    mut bone_q: Query<
        &mut Transform,
        (
            With<super::rig::CharacterBone>,
            Without<CullCamera>,
            Without<CharacterRoot>,
        ),
    >,
    mut mesh_vis: Query<
        (&mut Visibility, Option<&crate::character::rig::MeshView>),
        (With<CharacterMesh>, With<PlayerMesh>, Without<CharacterRoot>),
    >,
) {
    let Some(active) = active else { return };
    let pack: &CharacterPack = &active.pack;
    let Ok((mut cam_tf, ws, mut boom)) = cam.single_mut() else { return };
    let Ok((mut root, mut root_tf, mut root_vis)) = root_q.get_mut(active.root) else { return };

    let dt = time.delta_secs().min(0.1);

    // Walk mode only: in fly/drone the character would be dragged through the air.
    let walking = settings.mode == CamMode::Walk && cs.enabled;
    // The rig itself is present whenever we are walking; which GEOMETRY is drawn depends on the
    // view. EFT's first-person arms are a separate asset (the `hands/` bundles) that binds this
    // same rig, so first-person shows those instead of showing nothing -- and never shows the
    // third-person body, whose own arms would otherwise appear inside the FPV ones.
    let want_root = if walking { Visibility::Inherited } else { Visibility::Hidden };
    if *root_vis != want_root {
        *root_vis = want_root;
    }
    let shown = if cs.third_person {
        crate::character::rig::MeshView::Third
    } else {
        crate::character::rig::MeshView::First
    };
    for (mut v, view) in &mut mesh_vis {
        // A pack with no first-person hands has nothing to show in that view; its third-person
        // meshes stay hidden, which is the old behaviour.
        let on = walking && view.copied().unwrap_or(crate::character::rig::MeshView::Third) == shown;
        let want = if on { Visibility::Inherited } else { Visibility::Hidden };
        if *v != want {
            *v = want;
        }
    }
    if !walking {
        return;
    }

    // ---- landing edge ----
    if *was_airborne && ws.grounded {
        *land_timer = LAND_TIME;
    }
    *was_airborne = !ws.grounded;
    *land_timer = (*land_timer - dt).max(0.0);

    // ---- place ----
    // The camera's Y carries last frame's cosmetic head bob; the body must not inherit it or the
    // whole character pumps up and down.
    let feet = cam_tf.translation
        - Vec3::new(0.0, walk_ground::EYE_HEIGHT + ws.last_bob, 0.0);
    let cam_yaw = yaw_of(&cam_tf);

    // ---- heading: the body's own facing, deliberately NOT the camera's ----
    // The camera orbits this pivot; the body only turns for its own reasons. Reading body yaw off
    // the camera (the old behaviour) meant mouse-look spun the character in place and the camera
    // could never appear to orbit him.
    if !root.heading_init {
        root.heading = cam_yaw;
        root.heading_init = true;
    }
    let v = ws.horizontal_velocity;
    let target_yaw = match cs.heading_mode {
        super::HeadingMode::FaceCamera => cam_yaw,
        super::HeadingMode::FaceMovement => {
            if v.length() > MOVE_EPSILON {
                // Movement is already camera-relative (walk_move builds its basis from the camera
                // yaw), so this turns him toward where he is actually travelling.
                yaw_of_dir(Vec3::new(v.x, 0.0, v.y))
            } else {
                root.heading // idle: hold facing
            }
        }
    };
    // Shortest-arc approach at a bounded rate, so a 180 reversal sweeps instead of snapping.
    let mut delta = (target_yaw - root.heading + std::f32::consts::PI)
        .rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    let max_step = cs.turn_rate_deg.to_radians() * dt;
    delta = delta.clamp(-max_step, max_step);
    root.heading = (root.heading + delta).rem_euclid(std::f32::consts::TAU);
    let body_yaw = root.heading;

    // Rotate the character so its DERIVED forward axis points along the body yaw. No magic 180:
    // `pack.forward` was measured from the forward-walk clip's root motion.
    let facing = Quat::from_rotation_y(body_yaw - yaw_of_dir(pack.forward));
    root_tf.translation = feet;
    root_tf.rotation = facing;

    // ---- parameters + state ----
    let (params, speed) = build_params(ws, &settings, body_yaw, *land_timer > 0.0, aim_blend.0);
    // A jump is a launch off the ground with upward velocity; walking off a ledge is not.
    if ws.grounded {
        *jumped = false;
    } else if ws.vy > 0.1 {
        *jumped = true;
    }
    let next = select_state(ws, speed, *land_timer, *jumped);
    if root.state != next {
        // Cross-fade instead of cutting, using the graph's OWN duration where it has one.
        let fade = pack
            .transition_time(&root.state, next)
            .filter(|d| *d > 0.0)
            .unwrap_or(DEFAULT_FADE)
            .clamp(FADE_RANGE.0, FADE_RANGE.1);
        debug!(
            "character state {} -> {next} over {fade:.3}s (speed {speed:.2}, grounded {}, vy {:.2})",
            if root.state.is_empty() { "<none>" } else { &root.state },
            ws.grounded,
            ws.vy
        );
        if !root.state.is_empty() {
            root.prev_state = std::mem::take(&mut root.state);
            root.prev_time = root.state_time;
            root.fade = 0.0;
            root.fade_len = fade;
        }
        root.state = next.to_string();
        root.state_time = 0.0;
    }

    // Bind-pose diagnostic: leave the rig exactly as spawned.
    if cs.freeze_bind_pose {
        return;
    }

    // ---- blend trees -> weighted clips, for the incoming and outgoing states ----
    let mut leaves: Vec<WeightedClip> = Vec::new();
    let state_speed = gather(pack, &root.state, &params, &mut leaves);
    if leaves.is_empty() {
        return;
    }
    let mut prev_leaves: Vec<WeightedClip> = Vec::new();
    let mut prev_speed = 1.0;
    let fading = root.fade < 1.0 && !root.prev_state.is_empty();
    if fading {
        prev_speed = gather(pack, &root.prev_state.clone(), &params, &mut prev_leaves);
    }

    // ---- advance time, rate-matched to the blend's own root motion ----
    // Without this the legs cycle at the clip's authored speed while the body moves at the walk
    // speed, which reads as skating. With it, footfalls track ground speed.
    let rate_for = |leaves: &[WeightedClip]| {
        let r = blended_root_speed(pack, leaves);
        if r > 0.1 && speed > MOVE_EPSILON {
            (speed / r).clamp(0.25, 3.0)
        } else {
            1.0
        }
    };
    root.state_time += dt * rate_for(&leaves) * state_speed;
    if fading {
        root.prev_time += dt * rate_for(&prev_leaves) * prev_speed;
        root.fade = if root.fade_len > 1e-4 {
            (root.fade + dt / root.fade_len).min(1.0)
        } else {
            1.0
        };
    }

    // ---- accumulate (incoming and outgoing together = the cross-fade) ----
    let acc = acc.get_or_insert_with(|| PoseAccumulator::new(pack.bones.len()));
    acc.clear();
    // Smoothstep so the fade eases in and out rather than moving at constant angular rate.
    let t = root.fade.clamp(0.0, 1.0);
    let w_in = if fading { t * t * (3.0 - 2.0 * t) } else { 1.0 };
    for l in &leaves {
        if let Some(clip) = pack.clip_by_controller_id(l.clip_id) {
            accumulate_clip(acc, clip, root.state_time, l.weight * w_in);
        }
    }
    if fading {
        for l in &prev_leaves {
            if let Some(clip) = pack.clip_by_controller_id(l.clip_id) {
                accumulate_clip(acc, clip, root.prev_time, l.weight * (1.0 - w_in));
            }
        }
    }
    if root.fade >= 1.0 {
        root.prev_state.clear();
    }
    // Split the borrow: `locals` is the scratch buffer and `bones` the entity list, and the write
    // loop below needs both at once.
    let CharacterRoot { bones, locals, .. } = &mut *root;
    acc.resolve(pack, locals);

    // ---- write bone transforms ----
    // No root-motion special case here: the emitter STRIPS root motion out of the bone tracks into
    // its own channel, so every track is pure local animation. Earlier this code zeroed bone 0's
    // translation, but the carrier is actually bone 1 (`Root_Joint`) — `walk_aim_0` travels 4.003 m
    // along it — so the clip slid the whole skeleton out from under the camera while the walk physics
    // moved the character too.
    for (i, e) in bones.iter().enumerate() {
        let Ok(mut tf) = bone_q.get_mut(*e) else { continue };
        let (p, r, s) = locals[i];
        tf.translation = p;
        tf.rotation = r;
        tf.scale = s;
    }

    // ---- third-person boom ----
    if !cs.third_person {
        return;
    }
    let pivot = feet + Vec3::new(0.0, cs.boom_pivot_height, 0.0);
    let back = -cam_tf.forward().as_vec3();
    let mut dist = cs.boom_distance;
    if let Some(g) = &grid {
        // Pull in until the line of sight stops crossing geometry, so the camera does not sit inside
        // a wall when you back into one.
        while dist > 0.4 && g.segment_crossings(pivot, pivot + back * dist, 4) > 0 {
            dist -= 0.25;
        }
    }
    let target = pivot + back * dist;
    let offset = target - cam_tf.translation;
    cam_tf.translation += offset;
    boom.applied = offset;
}

/// The sight's eye anchor, ready to aim through.
///
/// `local` is the anchor in the WEAPON's space and `bone` the socket the weapon hangs on, so the
/// world pose is `bone.global * local` — which follows the animation for free, exactly as the
/// gun itself does.
#[derive(Resource)]
pub struct PlayerAim {
    pub bone: Entity,
    pub local: Transform,
    pub fov_deg: Option<f32>,
}

/// How far into the aim we are, 0 = hip, 1 = fully on the sight.
#[derive(Resource, Default)]
pub struct AimBlend(pub f32);

/// Aiming. Holds `isAiming` for the animator and, under `EFT_ADS_EYE=1`, snaps the eye onto the
/// optic's own anchor.
///
/// WHY THE EYE SNAP IS NOT THE DEFAULT: the game does not move your eye to the sight, it plays an
/// ADDITIVE aim pose that brings the sight up to your eye — `Additive_Aiming` and `Additive_ISaim`
/// are real layers in this controller, and `Aim_angle` is referenced 51 times across it. Dragging
/// the camera to the weapon instead puts it wherever the gun currently is, which at low ready is
/// inside the receiver. The anchor itself is correct and game-derived
/// (`OpticSight.ScopeTransform`, measured 14.6 cm behind the lens on the weapon axis); what is
/// missing is layer support in the pose evaluator, which neither `anim` nor `pack` models yet.
pub fn aim_down_sights(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    ui: Res<crate::inspect::UiWantsKeyboard>,
    cs: Res<CharacterSettings>,
    settings: Res<CameraSettings>,
    aim: Option<Res<PlayerAim>>,
    mut blend: ResMut<AimBlend>,
    bones: Query<&GlobalTransform, With<super::rig::CharacterBone>>,
    mut cam: Query<(&mut Transform, &mut Projection), With<crate::CullCamera>>,
) {
    let Some(aim) = aim else { return };
    // `EFT_ADS=1` holds the sight up without a mouse, so a headless capture can frame it.
    let forced = std::env::var("EFT_ADS").map(|v| v.trim() == "1").unwrap_or(false);
    let want = !ui.0
        && cs.enabled
        && settings.mode == CamMode::Walk
        && (forced || mouse.pressed(MouseButton::Right));
    // Ease in and out rather than snapping, the way a weapon actually comes up.
    let rate = if want { 9.0 } else { 12.0 };
    let target = if want { 1.0 } else { 0.0 };
    blend.0 += (target - blend.0) * (1.0 - (-rate * time.delta_secs()).exp());
    if blend.0 <= 0.001 {
        return;
    }
    if !std::env::var("EFT_ADS_EYE").map(|v| v.trim() == "1").unwrap_or(false) {
        return;
    }
    let Ok(bone_gt) = bones.get(aim.bone) else { return };
    let Ok((mut cam_tf, mut proj)) = cam.single_mut() else { return };
    let world = bone_gt.mul_transform(aim.local);
    let (_, want_rot, want_pos) = world.to_scale_rotation_translation();
    cam_tf.translation = cam_tf.translation.lerp(want_pos, blend.0);
    cam_tf.rotation = cam_tf.rotation.slerp(want_rot, blend.0);
    // MAGNIFICATION IS NOT A SCREEN ZOOM. `ScopeCameraData.FieldOfView` (5.03 deg on the G33)
    // drives the optic's own camera, whose image the game renders INTO the lens circle -- the
    // rest of the screen keeps its normal field of view. Applying it to the main camera would be
    // a ~12x zoom of everything, which is not what aiming looks like. The value is carried in the
    // pack for a future scope-camera pass; `EFT_ADS_ZOOM=1` opts the whole screen in meanwhile.
    if std::env::var("EFT_ADS_ZOOM").map(|v| v.trim() == "1").unwrap_or(false) {
        if let (Some(fov), Projection::Perspective(p)) = (aim.fov_deg, &mut *proj) {
            let hip = settings.fov_deg.to_radians();
            p.fov = hip + (fov.to_radians().max(0.02) - hip) * blend.0;
        }
    }
}

/// Marker: this mesh belongs to the character the camera is attached to. NPC meshes never carry
/// it, so switching your own view leaves every other character alone.
#[derive(Component)]
pub struct PlayerMesh;

/// Toggle first/third person.
pub fn toggle_view(
    keys: Res<ButtonInput<KeyCode>>,
    ui_kb: Res<crate::inspect::UiWantsKeyboard>,
    mut cs: ResMut<CharacterSettings>,
) {
    if ui_kb.0 {
        return;
    }
    if keys.just_pressed(KeyCode::KeyV) {
        cs.third_person = !cs.third_person;
        info!(
            "character view: {}",
            if cs.third_person { "third-person" } else { "first-person" }
        );
    }
}
