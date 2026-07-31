//! npc.rs — animated AI patrols, exactly where the game puts them.
//!
//! Spawns scavs (any `.eftchar` pack) on the pack's own `patrol_ways` — the waypoint polylines
//! the game's bots walk, extracted from its AI scene data — and drives them with the same
//! four-layer character stack the walk camera uses ([`character::pack`]/[`rig`]/[`anim`] plus a
//! small agent driver here instead of [`character::drive`]'s player input). Movement speed is
//! slaved to the blend's root motion (the game's own 2.5 m/s walk), so feet do not skate and
//! nothing is an authored constant. At each waypoint the agent pauses briefly, then walks on;
//! routes loop ping-pong like the game's patrols.
//!
//! `EFT_NPC=0` disables; `EFT_NPC_CHAR=<id>` overrides the default `scav` pack id.

use crate::character::anim::{accumulate_clip, PoseAccumulator, WeightedClip};
use crate::character::drive::{blended_root_speed, gather, states};
use crate::character::pack::CharacterPack;
use crate::character::rig::{self, CharacterBone, CharacterRoot};
use bevy::mesh::skinning::SkinnedMeshInverseBindposes;
use bevy::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

pub struct NpcPlugin;

impl Plugin for NpcPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (teardown_npcs, spawn_npcs).chain().run_if(npcs_need_rebuild),
        )
        // Same slot as the player driver: pose after game logic, before transform propagation.
        .add_systems(PostUpdate, drive_npcs.before(bevy::transform::TransformSystems::Propagate));
    }
}

fn npcs_need_rebuild(
    epoch: Res<crate::render::MapEpoch>,
    pack: Option<Res<crate::render::LoadedPack>>,
) -> bool {
    epoch.is_changed() || pack.is_some_and(|p| p.is_added())
}

/// One patrolling agent: its route (the game's waypoints), where it is along it, and its
/// dwell clock at the current waypoint.
#[derive(Component)]
struct Npc {
    route: Vec<Vec3>,
    /// Current leg START index; the agent walks route[leg] -> route[leg+dir].
    leg: usize,
    /// +1 forward, -1 backward (ping-pong at the ends, like the game's patrols).
    dir: i32,
    /// Distance covered along the current leg (m).
    dist: f32,
    /// Seconds left standing at the waypoint before walking the next leg.
    dwell: f32,
    /// Body yaw (rad), turned smoothly toward the walk direction.
    heading: f32,
}

/// The shared character data every NPC instance samples from.
#[derive(Resource)]
struct NpcCharacter(Arc<CharacterPack>);

fn teardown_npcs(mut commands: Commands, q: Query<Entity, With<Npc>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
    commands.remove_resource::<NpcCharacter>();
}

fn spawn_npcs(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut ibms: ResMut<Assets<SkinnedMeshInverseBindposes>>,
    pack: Option<Res<crate::render::LoadedPack>>,
) {
    if std::env::var("EFT_NPC").map(|v| v.trim() == "0").unwrap_or(false) {
        return;
    }
    let Some(pack) = pack else { return };
    // Patrol routes from the pack's own gamedata (the game's AI scene data).
    let routes = load_patrol_ways(&pack.0.root);
    if routes.is_empty() {
        info!("npc: no patrol_ways in gamedata — no patrols to walk");
        return;
    }
    // The character pack: default `scav`, overridable. Missing pack = a log, not an error —
    // the map is fully usable without NPCs.
    let id = std::env::var("EFT_NPC_CHAR").unwrap_or_else(|_| "scav".into());
    let dir = std::path::PathBuf::from("out").join("characters").join(id.trim());
    let cpack = match crate::character::pack::load(&dir) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            info!(
                "npc: character pack {} not loadable ({e}) — run \
                 extraction/characters/build_character.py --character scav",
                dir.display()
            );
            return;
        }
    };
    let mut n = 0usize;
    for route in &routes {
        if route.len() < 2 {
            continue;
        }
        let root = rig::spawn(
            &cpack,
            0,
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut images,
            &mut ibms,
        );
        let start = route[0];
        commands.entity(root).insert((
            Transform::from_translation(start),
            Npc {
                route: route.clone(),
                leg: 0,
                dir: 1,
                dist: 0.0,
                // Stagger initial dwell so simultaneous routes don't step in lockstep.
                dwell: 0.5 + (n as f32) * 0.7,
                heading: 0.0,
            },
        ));
        n += 1;
    }
    if n > 0 {
        info!("npc: {n} patrol agent(s) walking the game's own patrol_ways");
        commands.insert_resource(NpcCharacter(cpack));
    }
}

/// `patrol_ways` -> world polylines (already in viewer space; the extractor conjugates).
fn load_patrol_ways(root: &std::path::Path) -> Vec<Vec<Vec3>> {
    let Ok(txt) = std::fs::read_to_string(root.join("gamedata.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for w in v
        .get("patrol_ways")
        .and_then(|x| x.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
    {
        let pts: Vec<Vec3> = w
            .get("points")
            .and_then(|p| p.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| p.as_array())
                    .filter(|p| p.len() >= 3)
                    .map(|p| {
                        Vec3::new(
                            p[0].as_f64().unwrap_or(0.0) as f32,
                            p[1].as_f64().unwrap_or(0.0) as f32,
                            p[2].as_f64().unwrap_or(0.0) as f32,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        if pts.len() >= 2 {
            out.push(pts);
        }
    }
    out
}

/// Seconds an agent stands at a waypoint before walking on. A behavior constant the scene data
/// does not carry (the game's value lives in its AI logic); modest and obviously provisional.
const WAYPOINT_DWELL_S: f32 = 3.0;
/// Turn rate toward the walk direction (rad/s).
const TURN_RATE: f32 = 3.0;

/// Advance every agent along its route and pose it — the NPC counterpart of `drive_character`,
/// minus input, camera and jumping. Movement rate comes from the BLEND's root-motion speed, so
/// playback and travel can't disagree (no foot skating), exactly like the player driver.
fn drive_npcs(
    time: Res<Time>,
    cpack: Option<Res<NpcCharacter>>,
    mut acc: Local<Option<PoseAccumulator>>,
    mut scratch: Local<Vec<WeightedClip>>,
    mut params: Local<HashMap<String, f32>>,
    mut root_q: Query<(&mut Npc, &mut CharacterRoot, &mut Transform), Without<CharacterBone>>,
    mut bone_q: Query<&mut Transform, (With<CharacterBone>, Without<Npc>)>,
) {
    let Some(cpack) = cpack else { return };
    let pack: &CharacterPack = &cpack.0;
    let dt = time.delta_secs().min(0.1);

    for (mut npc, mut root, mut tf) in &mut root_q {
        // ---- agent step: dwell at waypoints, else walk the current leg ----
        let moving = if npc.dwell > 0.0 {
            npc.dwell -= dt;
            false
        } else {
            true
        };
        let (a, b) = {
            let i = npc.leg;
            let j = (npc.leg as i32 + npc.dir) as usize;
            (npc.route[i], npc.route[j])
        };
        let leg_vec = b - a;
        let leg_len = leg_vec.length().max(1.0e-3);
        let walk_dir = leg_vec / leg_len;

        // ---- animator: same parameter/state machinery as the player ----
        params.clear();
        let speed_norm = if moving { 1.0 } else { 0.0 };
        params.insert("Speed".into(), speed_norm);
        params.insert("InputSpeed".into(), speed_norm);
        params.insert("MoveSpeed".into(), speed_norm);
        params.insert("WalkSpeed".into(), speed_norm);
        params.insert("InputDirection".into(), 0.0);
        params.insert("Direction".into(), 0.0);
        let want = if moving { states::MOVE } else { states::IDLE };
        if root.state != want {
            root.prev_state = std::mem::replace(&mut root.state, want.to_string());
            root.prev_time = root.state_time;
            root.state_time = 0.0;
            root.fade = 0.0;
            root.fade_len = 0.25;
        }
        let state_path = root.state.clone();
        let st_speed = gather(pack, &state_path, &params, &mut scratch);
        let root_speed = blended_root_speed(pack, &scratch).max(0.0);

        // ---- move: travel at the blend's own root-motion speed ----
        if moving && root_speed > 1.0e-3 {
            npc.dist += root_speed * dt;
            if npc.dist >= leg_len {
                // waypoint reached: step the leg, ping-pong at the ends, dwell.
                npc.dist = 0.0;
                npc.dwell = WAYPOINT_DWELL_S;
                let next = npc.leg as i32 + npc.dir;
                let last = npc.route.len() as i32 - 1;
                if next <= 0 {
                    npc.leg = 0;
                    npc.dir = 1;
                } else if next >= last {
                    npc.leg = last as usize;
                    npc.dir = -1;
                } else {
                    npc.leg = next as usize;
                }
            }
        }
        let t = (npc.dist / leg_len).clamp(0.0, 1.0);
        tf.translation = a + leg_vec * t;
        // Face the walk direction, turning at a bounded rate.
        let want_yaw = (-walk_dir.x).atan2(-walk_dir.z);
        let mut d = want_yaw - npc.heading;
        while d > std::f32::consts::PI {
            d -= std::f32::consts::TAU;
        }
        while d < -std::f32::consts::PI {
            d += std::f32::consts::TAU;
        }
        npc.heading += d.clamp(-TURN_RATE * dt, TURN_RATE * dt);
        tf.rotation = Quat::from_rotation_y(npc.heading);

        // ---- clock: rate-match playback to travel (the player driver's rule) ----
        let rate = if moving && root_speed > 1.0e-3 {
            st_speed
        } else {
            st_speed
        };
        root.state_time += dt * rate;
        root.prev_time += dt * rate;
        root.fade = (root.fade + dt / root.fade_len.max(1.0e-3)).min(1.0);
        let fading = !root.prev_state.is_empty() && root.fade < 1.0;

        // ---- accumulate + resolve + write bones (the drive_character recipe) ----
        let acc = acc.get_or_insert_with(|| PoseAccumulator::new(pack.bones.len()));
        acc.clear();
        let ft = root.fade.clamp(0.0, 1.0);
        let w_in = if fading { ft * ft * (3.0 - 2.0 * ft) } else { 1.0 };
        for l in scratch.iter() {
            if let Some(clip) = pack.clip_by_controller_id(l.clip_id) {
                accumulate_clip(acc, clip, root.state_time, l.weight * w_in);
            }
        }
        if fading {
            let prev_path = root.prev_state.clone();
            let mut prev_leaves: Vec<WeightedClip> = Vec::new();
            gather(pack, &prev_path, &params, &mut prev_leaves);
            for l in &prev_leaves {
                if let Some(clip) = pack.clip_by_controller_id(l.clip_id) {
                    accumulate_clip(acc, clip, root.prev_time, l.weight * (1.0 - w_in));
                }
            }
        }
        if root.fade >= 1.0 {
            root.prev_state.clear();
        }
        let CharacterRoot { bones, locals, .. } = &mut *root;
        acc.resolve(pack, locals);
        for (i, e) in bones.iter().enumerate() {
            let Ok(mut btf) = bone_q.get_mut(*e) else { continue };
            let (p, r, s) = locals[i];
            btf.translation = p;
            btf.rotation = r;
            btf.scale = s;
        }
    }
}
