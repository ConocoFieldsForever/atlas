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
        .add_systems(PostUpdate, drive_npcs.before(bevy::transform::TransformSystems::Propagate))
        .add_systems(Update, sync_character_light.run_if(npcs_need_rebuild))
        .add_systems(Update, attach_weapons);
    }
}

fn npcs_need_rebuild(
    epoch: Res<crate::render::MapEpoch>,
    pack: Option<Res<crate::render::LoadedPack>>,
) -> bool {
    epoch.is_changed() || pack.is_some_and(|p| p.is_added())
}

/// One agent: a plan (the game's own target points) and the nav-routed path currently walked.
#[derive(Component)]
struct Npc {
    /// Plan targets — patrol_ways waypoints (ping-pong) or a core-point group (cycle).
    targets: Vec<Vec3>,
    at: usize,
    /// +1/-1 for patrol ping-pong; wanderers always cycle forward.
    dir: i32,
    ping_pong: bool,
    /// The current leg, as a NAV-ROUTED polyline from the grid built with the game's own agent
    /// parameters — so agents take doors, ramps and stairs, never a chord through a wall.
    /// Recomputed lazily (the nav grid streams in after spawn); empty = needs (re)planning.
    path: Vec<Vec3>,
    leg: usize,
    /// Distance covered along the current path leg (m).
    dist: f32,
    /// Seconds left standing at the current target before walking on.
    dwell: f32,
    /// Body yaw (rad), turned smoothly toward the walk direction.
    heading: f32,
}

/// The shared character data every NPC instance samples from.
#[derive(Resource)]
struct NpcCharacter(Arc<CharacterPack>);

/// The weapon every agent carries (one `.eftweap`, shared handles).
#[derive(Resource)]
struct NpcWeapon(Arc<crate::character::weapon::WeaponPack>);

/// Marks an agent whose weapon has been parented, so the attach runs once per agent.
#[derive(Component)]
struct WeaponAttached;

/// Parent the weapon's parts to each agent's `Weapon_root` bone. A separate system because the
/// bone entities are created by deferred commands during spawn and are only readable afterwards;
/// from then on the weapon follows the animation with no per-frame work.
fn attach_weapons(
    mut commands: Commands,
    weapon: Option<Res<NpcWeapon>>,
    cpack: Option<Res<NpcCharacter>>,
    agents: Query<(Entity, &CharacterRoot), (With<Npc>, Without<WeaponAttached>)>,
) {
    let (Some(weapon), Some(cpack)) = (weapon, cpack) else { return };
    let Some(bi) = cpack
        .0
        .bones
        .iter()
        .position(|b| b.name == crate::character::weapon::WEAPON_BONE)
    else {
        return;
    };
    for (e, root) in &agents {
        let Some(&bone) = root.bones.get(bi) else { continue };
        for (mesh, mat) in &weapon.0.parts {
            let child = commands
                .spawn((Mesh3d(mesh.clone()), MeshMaterial3d(mat.clone()), Transform::IDENTITY))
                .id();
            commands.entity(bone).add_child(child);
        }
        commands.entity(e).insert(WeaponAttached);
    }
}

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
    // Wander plans: the game's own bot interest points, grouped by ITS `cg` core-group id.
    let groups = load_core_groups(&pack.0.root);
    // The agent's WEAPON: `.eftweap` packs built by extraction/characters/build_weapon.py from
    // BSG's own item tree. EFT_NPC_WEAPON names one explicitly; otherwise the first pack present
    // in out/weapons is used. (Per-agent rolled kits from bot_loadouts.json land next; the roll
    // itself already exists in extraction/characters/loadout.py.)
    let weapon = std::env::var("EFT_NPC_WEAPON")
        .ok()
        .map(|id| crate::character::weapon::weapon_dir(&id))
        .or_else(|| {
            std::fs::read_dir("out/weapons").ok().and_then(|rd| {
                let mut dirs: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.join("manifest.json").is_file())
                    .collect();
                dirs.sort();
                dirs.into_iter().next()
            })
        })
        .and_then(|d| crate::character::weapon::load(&d, &mut meshes, &mut materials, &mut images));

    let mut n = 0usize;
    let mut spawn_agent = |targets: Vec<Vec3>, ping_pong: bool,
                           commands: &mut Commands,
                           meshes: &mut Assets<Mesh>,
                           materials: &mut Assets<StandardMaterial>,
                           images: &mut Assets<Image>,
                           ibms: &mut Assets<SkinnedMeshInverseBindposes>| {
        let root = rig::spawn(&cpack, 0, commands, meshes, materials, images, ibms);
        let start = targets[0];
        commands.entity(root).insert((
            Transform::from_translation(start),
            Npc {
                targets,
                at: 0,
                dir: 1,
                ping_pong,
                path: Vec::new(),
                leg: 0,
                dist: 0.0,
                // Stagger initial dwell so agents don't step in lockstep.
                dwell: 0.5 + (n as f32) * 0.9,
                heading: 0.0,
            },
        ));
        n += 1;
    };
    for route in &routes {
        if route.len() >= 2 {
            spawn_agent(route.clone(), true, &mut commands, &mut meshes, &mut materials, &mut images, &mut ibms);
        }
    }
    // One wanderer per core group with enough points to circulate; capped so big maps stay light.
    const MAX_WANDERERS: usize = 8;
    let mut wanderers = 0usize;
    for pts in groups {
        if pts.len() >= 3 && wanderers < MAX_WANDERERS {
            spawn_agent(pts, false, &mut commands, &mut meshes, &mut materials, &mut images, &mut ibms);
            wanderers += 1;
        }
    }
    if let Some(w) = weapon {
        commands.insert_resource(NpcWeapon(std::sync::Arc::new(w)));
    }
    if n > 0 {
        info!(
            "npc: {n} agent(s) — {} on patrol_ways, {wanderers} circulating core-point groups",
            n - wanderers
        );
        commands.insert_resource(NpcCharacter(cpack));
    }
}

/// `core_points` grouped by the game's own `cg` (core-group) id -> wander circuits.
fn load_core_groups(root: &std::path::Path) -> Vec<Vec<Vec3>> {
    let Ok(txt) = std::fs::read_to_string(root.join("gamedata.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return Vec::new();
    };
    let mut by_cg: std::collections::BTreeMap<i64, Vec<Vec3>> = Default::default();
    for c in v
        .get("core_points")
        .and_then(|x| x.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
    {
        let (Some(pos), Some(cg)) = (
            c.get("pos").and_then(|p| p.as_array()).filter(|p| p.len() >= 3),
            c.get("cg").and_then(|x| x.as_i64()),
        ) else {
            continue;
        };
        by_cg.entry(cg).or_default().push(Vec3::new(
            pos[0].as_f64().unwrap_or(0.0) as f32,
            pos[1].as_f64().unwrap_or(0.0) as f32,
            pos[2].as_f64().unwrap_or(0.0) as f32,
        ));
    }
    by_cg.into_values().collect()
}

/// Point the Bevy key light (which lights the SKINNED characters — they go through Bevy's PBR
/// path, not the map's gpu_driven shading) along the map's REAL sun, and match its ambient to the
/// map's GI. Without this the characters were lit from a fixed authored angle unrelated to the
/// world: a hard terminator across the skull, shading that disagreed with every surface behind
/// them. The direction is the same `sun_dir` the SH bake and the map's cascades use, so bodies
/// and world now agree. (`configure_lighting` does this too, but ONLY on the Standard path.)
fn sync_character_light(
    pack: Option<Res<crate::render::LoadedPack>>,
    gfx: Option<Res<crate::render::GfxSettings>>,
    mut ambient: ResMut<AmbientLight>,
    mut lights: Query<(&mut DirectionalLight, &mut Transform)>,
) {
    let Some(pack) = pack else { return };
    let sun = pack
        .0
        .manifest
        .sidecars
        .volume_meta
        .as_deref()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|txt| serde_json::from_str::<serde_json::Value>(&txt).ok())
        .and_then(|v| {
            let a = v.get("sun_dir")?.as_array()?.clone();
            Some(Vec3::new(
                a.first()?.as_f64()? as f32,
                a.get(1)?.as_f64()? as f32,
                a.get(2)?.as_f64()? as f32,
            ))
        });
    let Some(sun) = sun.filter(|s| s.length_squared() > 1.0e-6) else {
        return;
    };
    let sun = sun.normalize();
    // The map's own GI scale keeps bodies in step with the world when the user rides the slider.
    let gi = gfx.as_ref().map(|g| g.gi_intensity).unwrap_or(1.0).clamp(0.1, 3.0);
    ambient.color = Color::srgb(0.72, 0.74, 0.80);
    ambient.brightness = 900.0 * gi;
    for (mut dl, mut tf) in &mut lights {
        dl.illuminance = 9000.0 * gi;
        // `sun_dir` points TOWARD the sun; a directional light looks ALONG its travel.
        *tf = Transform::from_translation(sun * 100.0).looking_at(Vec3::ZERO, Vec3::Y);
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
    nav: Option<Res<crate::pathfind::Nav>>,
    mut scratch_nav: Local<Option<crate::nav::PooledScratch>>,
    mut acc: Local<Option<PoseAccumulator>>,
    mut scratch: Local<Vec<WeightedClip>>,
    mut prev_scratch: Local<Vec<WeightedClip>>,
    mut params: Local<HashMap<String, f32>>,
    mut replans: Local<u32>,
    mut root_q: Query<(&mut Npc, &mut CharacterRoot, &mut Transform), Without<CharacterBone>>,
    mut bone_q: Query<&mut Transform, (With<CharacterBone>, Without<Npc>)>,
) {
    let Some(cpack) = cpack else { return };
    let pack: &CharacterPack = &cpack.0;
    let dt = time.delta_secs().min(0.1);
    let grid = nav.as_ref().and_then(|n| n.0.as_ref());
    *replans = 0; // per-frame A* budget (see below)

    for (mut npc, mut root, mut tf) in &mut root_q {
        // ---- plan: (re)route the current leg through the NAV GRID when none is active ----
        // A* over a map-sized grid costs milliseconds, and several agents finishing a leg on the
        // SAME frame stacked those costs into a visible hitch. Budget ONE replan per frame; an
        // agent waiting its turn simply stands (it is already dwelling at a waypoint anyway).
        if npc.path.len() < 2 && *replans >= 1 {
            continue;
        }
        if npc.path.len() < 2 {
            *replans += 1;
            let from = tf.translation;
            let next = npc.targets[((npc.at as i32 + npc.dir).rem_euclid(npc.targets.len() as i32)) as usize];
            npc.path = if let Some(g) = grid {
                let sn = scratch_nav.get_or_insert_with(|| crate::nav::pooled_scratch(g.nodes()));
                match g.path(from, next, &mut *sn, None) {
                    Some((mut poly, _)) if poly.len() >= 2 => {
                        // A* returns a polyline through grid-CELL CENTRES, so poly[0] is the
                        // snapped cell, not where the agent is standing — adopting it verbatim
                        // teleported the body up to a cell sideways at every replan. Walk from
                        // the true position into the routed line instead.
                        if poly[0].distance_squared(from) > 0.01 {
                            poly.insert(0, from);
                        }
                        poly
                    }
                    // unreachable by grid: straight fallback rather than a frozen agent.
                    _ => vec![from, next],
                }
            } else {
                vec![from, next] // nav still streaming in: straight fallback, replanned next target
            };
            npc.leg = 0;
            npc.dist = 0.0;
        }
        // ---- agent step: dwell at plan targets, else walk the current path leg ----
        let moving = if npc.dwell > 0.0 {
            npc.dwell -= dt;
            false
        } else {
            true
        };
        let (a, b) = (npc.path[npc.leg], npc.path[npc.leg + 1]);
        let leg_vec = b - a;
        let leg_len = leg_vec.length().max(1.0e-3);

        // ---- animator: same parameter/state machinery as the player ----
        // The controller's REAL parameters (read from the extracted graph, not invented): MOVE is
        // a 2D freeform tree on Direct_X/Direct_Y (movement direction in body space), whose
        // children are 2D trees on Speed (walk..run) and Level (stance). Feeding names that do
        // not exist left every axis at 0 — no forward component — so the blend resolved to the
        // backpedal/strafe/crouch corners: bots moonwalked and duck-walked.
        params.clear();
        params.insert("Direct_X".into(), 0.0); // no strafe: agents turn to face their path
        params.insert("Direct_Y".into(), if moving { 1.0 } else { 0.0 }); // forward
        params.insert("Speed".into(), if moving { 0.5 } else { 0.0 }); // 0.5 = walk, 1 = run
        // Level is the STANCE axis and 1 is STANDING, not 0 — the tree's Level-0 row is
        // crouch_slow_aim / crouch_aim / crouch_run_aim, its Level-1 row walk_aim_slow /
        // walk_aim / run_aim. Feeding 0 duck-walked every agent down the route.
        params.insert("Level".into(), 1.0);
        params.insert("Sprint".into(), 0.0);
        params.insert("Tilt".into(), 0.0);
        let want = if moving { states::MOVE } else { states::IDLE };
        if root.state != want {
            root.prev_state = std::mem::replace(&mut root.state, want.to_string());
            root.prev_time = root.state_time;
            root.state_time = 0.0;
            root.fade = 0.0;
            root.fade_len = 0.25;
        }
        // `gather` borrows the pack, not the root — take the &str directly (this used to clone
        // two Strings per agent per frame).
        // Move the String out, gather, put it back: no clone, no unsafe, no borrow conflict.
        let sp = std::mem::take(&mut root.state);
        let st_speed = gather(pack, &sp, &params, &mut scratch);
        root.state = sp;
        let root_speed = blended_root_speed(pack, &scratch).max(0.0);

        // ---- move: travel at the blend's own root-motion speed ----
        // NOTE the ordering below. `a`/`leg_vec` above describe the leg as it was at the TOP of
        // this frame; the block may advance to the next leg, so the position write MUST re-read
        // the current leg afterwards. Writing with the stale `a` while `dist` had been reset to 0
        // snapped the body back to the START of the leg it had just finished, for exactly one
        // frame, at every leg boundary — the twitch every few steps.
        if moving && root_speed > 1.0e-3 {
            npc.dist += root_speed * dt;
            if npc.dist >= leg_len {
                // Carry the OVERSHOOT into the next leg instead of discarding it: dropping it
                // stalled the agent by a fraction of a step at every vertex.
                let over = npc.dist - leg_len;
                npc.dist = over;
                if npc.leg + 2 < npc.path.len() {
                    // interior polyline vertex: keep walking, no dwell (it's one route leg).
                    npc.leg += 1;
                } else {
                    npc.dist = 0.0;
                    // PLAN target reached: advance the plan, dwell, and force a replan.
                    npc.dwell = WAYPOINT_DWELL_S;
                    let n = npc.targets.len() as i32;
                    let next = npc.at as i32 + npc.dir;
                    if npc.ping_pong {
                        if next <= 0 {
                            npc.at = 0;
                            npc.dir = 1;
                        } else if next >= n - 1 {
                            npc.at = (n - 1) as usize;
                            npc.dir = -1;
                        } else {
                            npc.at = next as usize;
                        }
                    } else {
                        npc.at = next.rem_euclid(n) as usize;
                    }
                    npc.path.clear();
                    npc.leg = 0;
                }
            }
        }
        // Re-read the CURRENT leg (it may have advanced above) and place the body on it.
        let (ca, cb) = if npc.path.len() >= 2 && npc.leg + 1 < npc.path.len() {
            (npc.path[npc.leg], npc.path[npc.leg + 1])
        } else {
            (a, b)
        };
        let cvec = cb - ca;
        let clen = cvec.length().max(1.0e-3);
        let t = (npc.dist / clen).clamp(0.0, 1.0);
        tf.translation = ca + cvec * t;
        let walk_dir = cvec / clen;
        // Face the walk direction, turning at a bounded rate.
        // The rig's forward is +Z (manifest `characterForward`, derived from walk_aim_0's root
        // motion), and rotation_y(yaw) maps +Z to (sin yaw, 0, cos yaw) — so the yaw that faces
        // `d` is atan2(d.x, d.z). The old (-x, -z) form is the CAMERA convention (cameras look
        // down -Z); using it here pointed every body 180 degrees away from its travel, which is
        // why they moonwalked down the route with a correct forward-walk animation playing.
        let want_yaw = walk_dir.x.atan2(walk_dir.z);
        let mut d = want_yaw - npc.heading;
        while d > std::f32::consts::PI {
            d -= std::f32::consts::TAU;
        }
        while d < -std::f32::consts::PI {
            d += std::f32::consts::TAU;
        }
        npc.heading += d.clamp(-TURN_RATE * dt, TURN_RATE * dt);
        tf.rotation = Quat::from_rotation_y(npc.heading);

        // ---- clock: rate-match playback to TRAVEL (the player driver's rule) ----
        // Both branches used to be `st_speed`, i.e. no rate matching at all: the walk cycle ran at
        // the clip's authored rate while the body moved at the blend's root speed, so the feet
        // slipped and the gait read as stuttering. Playing at (travel / root_speed) locks the
        // stride to the ground exactly like the player's driver does.
        let rate = if moving && root_speed > 1.0e-3 {
            st_speed * (root_speed / root_speed.max(1.0e-3)).clamp(0.25, 4.0)
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
            let pp = std::mem::take(&mut root.prev_state);
            gather(pack, &pp, &params, &mut prev_scratch);
            root.prev_state = pp;
            for l in prev_scratch.iter() {
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
