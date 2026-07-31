//! eft::character — playable characters (Tagilla and friends) for the walk camera.
//!
//! Consumes the `.eftchar` packs emitted by `extraction/characters/build_character.py`. Four layers,
//! each independently replaceable:
//!
//! | module | role |
//! |---|---|
//! | [`pack`]  | `.eftchar` -> engine-agnostic data (skeleton, meshes, clips, state table) |
//! | [`rig`]   | that data -> Bevy entities: bone hierarchy + `SkinnedMesh` draws |
//! | [`anim`]  | clip sampling, blend-tree evaluation, weighted pose accumulation |
//! | [`drive`] | `WalkState` -> animator parameters -> state -> pose; third-person boom |
//!
//! Nothing here is Tagilla-specific: every character binds to the same 79-bone rig, so pointing
//! `EFT_CHARACTER` at another pack is the whole cost of swapping who you play.
//!
//! Skinning is Bevy's own (`bevy_mesh::skinning::SkinnedMesh` + the `bevy_pbr` joint palette), which
//! coexists with the map's custom `gpu_driven` pass the same way the POI markers and loot cubes do —
//! those are ordinary `Mesh3d` entities too.

pub mod anim;
pub mod drive;
pub mod pack;
pub mod rig;
pub mod weapon;

use bevy::mesh::skinning::SkinnedMeshInverseBindposes;
use bevy::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

/// The loaded character and its spawned root entity.
#[derive(Resource)]
pub struct ActiveCharacter {
    pub pack: Arc<pack::CharacterPack>,
    pub root: Entity,
}

/// What the character's body faces, which decides whether the camera orbits him or spins him.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeadingMode {
    /// Body turns toward the direction it is MOVING, and holds its facing while idle. Mouse-look
    /// then orbits the camera around a character who keeps his own heading. This is the ordinary
    /// third-person feel and the default.
    FaceMovement,
    /// Body follows the camera yaw, so looking around turns the character on the spot. This is how
    /// EFT itself behaves (your body faces where you aim) and it is what makes the 9-way directional
    /// blend meaningful, since strafing then genuinely plays sidestep clips.
    FaceCamera,
}

/// Runtime knobs. Defaults put you behind the character's shoulder in walk mode.
#[derive(Resource)]
pub struct CharacterSettings {
    /// Master switch. Off = the walk camera behaves exactly as it did before this module existed.
    pub enabled: bool,
    /// Third-person (body visible) vs first-person (body hidden). Toggled with V.
    pub third_person: bool,
    /// Boom length in metres.
    pub boom_distance: f32,
    /// Height above the feet that the boom pivots around — roughly chest height so the camera looks
    /// over the shoulder rather than at the ground.
    pub boom_pivot_height: f32,
    /// Which LOD to spawn; falls back to the pack's `defaultLod`.
    pub lod: Option<u32>,
    /// What the body faces. See [`HeadingMode`].
    pub heading_mode: HeadingMode,
    /// How fast the body turns toward its target facing (deg/s). EFT carries a per-state
    /// `RotationSpeedClamp` in the extracted `PlayerStateContainer` metadata, which is the
    /// game-derived version of this; it is in the pack for whoever wires it up.
    pub turn_rate_deg: f32,
    /// Diagnostic (`EFT_CHARACTER_BIND=1`): hold the rig in its bind pose and apply no clip.
    /// Separates "the skeleton/skinning is wrong" from "the pose pipeline is wrong", which look
    /// identical on screen.
    pub freeze_bind_pose: bool,
}

impl Default for CharacterSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            // `EFT_CHARACTER_VIEW=first` starts in first person. V still toggles; this exists so a
            // headless capture can frame the weapon without a keypress.
            third_person: !matches!(
                std::env::var("EFT_CHARACTER_VIEW").as_deref().map(str::trim),
                Ok("first") | Ok("fpv") | Ok("1")
            ),
            boom_distance: 2.6,
            boom_pivot_height: 1.45,
            lod: None,
            heading_mode: match std::env::var("EFT_CHARACTER_HEADING").as_deref().map(str::trim) {
                Ok("camera") => HeadingMode::FaceCamera,
                _ => HeadingMode::FaceMovement,
            },
            turn_rate_deg: 540.0,
            freeze_bind_pose: std::env::var("EFT_CHARACTER_BIND")
                .map(|v| v.trim() == "1")
                .unwrap_or(false),
        }
    }
}

/// Where to look for the character pack.
///
/// `EFT_CHARACTER` may be either a pack directory or a bare character id resolved under
/// `out/characters/<id>`. Unset = no character, and the walk camera is untouched.
fn pack_dir() -> Option<PathBuf> {
    let raw = std::env::var("EFT_CHARACTER").ok()?;
    let raw = raw.trim();
    if raw.is_empty() || raw == "0" {
        return None;
    }
    let direct = PathBuf::from(raw);
    if direct.join("manifest.json").is_file() {
        return Some(direct);
    }
    // Repo-relative default output location of build_character.py.
    let by_id = PathBuf::from("out").join("characters").join(raw);
    if by_id.join("manifest.json").is_file() {
        return Some(by_id);
    }
    warn!("EFT_CHARACTER={raw:?}: no manifest.json at {} or {}", direct.display(), by_id.display());
    None
}

fn load_character(
    mut commands: Commands,
    cs: Res<CharacterSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut ibms: ResMut<Assets<SkinnedMeshInverseBindposes>>,
) {
    let Some(dir) = pack_dir() else { return };
    let loaded = match pack::load(&dir) {
        Ok(p) => p,
        Err(e) => {
            // A malformed pack is a build problem, not a reason to take the viewer down.
            error!("character pack {} failed to load: {e:#}", dir.display());
            return;
        }
    };
    let lod = cs.lod.unwrap_or(loaded.default_lod);
    let spawned = rig::spawn(
        &loaded,
        lod,
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut images,
        &mut ibms,
    );
    attach_weapon(&loaded, &spawned, &mut commands, &mut meshes, &mut materials, &mut images);
    // Mark OUR meshes. The first/third-person switch is a property of the character you are
    // looking through, not of every character in the world.
    for &e in &spawned.meshes {
        commands.entity(e).insert(drive::PlayerMesh);
    }
    commands.insert_resource(ActiveCharacter { pack: Arc::new(loaded), root: spawned.root });
}

/// Put a weapon in the character's hands, on the rig's own `Weapon_root` socket.
///
/// The same socket the NPCs use and the same anchor `build_weapon.py` bakes against, so the gun
/// sits in the hands in both views: third-person you watch yourself carry it, first-person it is
/// what you see past the FPV hands. `EFT_PLAYER_WEAPON` names a pack explicitly; otherwise the
/// first one present in `out/weapons` is used, and no weapon at all is simply empty hands.
///
/// Done here rather than in a later system because `Commands::spawn` hands back usable entity ids
/// immediately -- the bone entities are already parentable, so the weapon is in place on the very
/// first frame the character exists.
fn attach_weapon(
    pack: &pack::CharacterPack,
    spawned: &rig::SpawnedRig,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
) {
    let Some(bi) = pack.bones.iter().position(|b| b.name == weapon::WEAPON_BONE) else {
        warn!("character rig has no {:?} bone -- no weapon attached", weapon::WEAPON_BONE);
        return;
    };
    let Some(&bone) = spawned.bones.get(bi) else { return };
    let dir = match std::env::var("EFT_PLAYER_WEAPON") {
        Ok(id) => Some(weapon::weapon_dir(id.trim())),
        Err(_) => first_weapon_pack(),
    };
    let Some(dir) = dir else { return };
    let Some(wp) = weapon::load(&dir, meshes, materials, images) else { return };
    for (mesh, mat) in &wp.parts {
        let child = commands
            .spawn((Mesh3d(mesh.clone()), MeshMaterial3d(mat.clone()), Transform::IDENTITY))
            .id();
        commands.entity(bone).add_child(child);
    }
    info!("player weapon: {} part(s) from {}", wp.parts.len(), dir.display());
}

/// First `.eftweap` pack on disk, in name order — a stable default when none is named.
fn first_weapon_pack() -> Option<PathBuf> {
    let mut dirs: Vec<_> = std::fs::read_dir("out/weapons")
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    dirs.sort();
    dirs.into_iter().next()
}

/// Attach the boom bookkeeping to the cull camera once it exists.
fn attach_boom(
    mut commands: Commands,
    q: Query<Entity, (With<crate::render::CullCamera>, Without<drive::CameraBoom>)>,
) {
    for e in &q {
        commands.entity(e).insert(drive::CameraBoom::default());
    }
}

pub struct CharacterPlugin;

impl Plugin for CharacterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CharacterSettings>()
            // PostStartup: the cull camera is spawned in main's Startup `setup`.
            .add_systems(PostStartup, (load_character, attach_boom))
            .add_systems(Update, (attach_boom, drive::toggle_view))
            // The boom MUST come off before `walk_move` (Update) reads the camera as the player's
            // eye; PreUpdate guarantees that without needing to order against a private system.
            .add_systems(PreUpdate, drive::unboom_camera)
            // Pose and re-boom after all movement, before transforms propagate.
            .add_systems(
                PostUpdate,
                drive::drive_character.before(bevy::transform::TransformSystems::Propagate),
            );
    }
}
