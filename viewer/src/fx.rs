//! fx.rs — LOOPING PARTICLE EFFECTS (fires / smoke / steam / sparks).
//!
//! Renders the game's own persistent ParticleSystems from the `particles.json` sidecar
//! (extraction/unity/eft_extract_particles.py — no reassembly needed, it heals built packs).
//! Every number is extracted data: the flipbook atlas + grid + rate (UVModule), start
//! color/size/lifetime/speed (InitialModule), emission rate, the material's tint and its blend
//! family (Additive vs Alpha Blended, from the shader name the game authored).
//!
//! Render model, v1: per emitter, a small cluster of camera-facing quads sharing one animated
//! flipbook material (frames advance together; positions/phases differ — a fire's flames flicker
//! in step but rise independently, which reads right). Quads rise with startSpeed and fall with
//! gravityModifier, looping over startLifetime. EFT_FX=0 disables the whole overlay.

use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use serde::Deserialize;

pub struct FxPlugin;

impl Plugin for FxPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (teardown_fx, spawn_fx).chain().run_if(fx_needs_rebuild),
        )
        .add_systems(Update, (animate_quads, animate_flipbooks, billboard_quads));
    }
}

fn fx_needs_rebuild(
    epoch: Res<crate::render::MapEpoch>,
    pack: Option<Res<crate::render::LoadedPack>>,
) -> bool {
    epoch.is_changed() || pack.is_some_and(|p| p.is_added())
}

#[derive(Deserialize)]
struct FxFile {
    emitters: Vec<FxEmitter>,
}

#[derive(Deserialize)]
struct FxEmitter {
    pos: [f32; 3],
    tex: String,
    #[serde(default)]
    shader: String,
    #[serde(default = "one4")]
    tint: [f32; 4],
    #[serde(default = "one")]
    lifetime: f32,
    #[serde(default)]
    speed: f32,
    #[serde(default = "one")]
    size: f32,
    #[serde(default = "one4")]
    color: [f32; 4],
    #[serde(default)]
    gravity: f32,
    #[serde(default = "four")]
    rate: f32,
    #[serde(default = "one_u")]
    tiles: [u32; 2],
    #[serde(default)]
    #[serde(rename = "uvEnabled")]
    uv_enabled: bool,
    #[serde(default = "one")]
    #[serde(rename = "uvCycles")]
    uv_cycles: f32,
    #[serde(default = "third")]
    #[serde(rename = "shapeRadius")]
    shape_radius: f32,
}

fn one() -> f32 {
    1.0
}
fn four() -> f32 {
    4.0
}
fn third() -> f32 {
    0.3
}
fn one4() -> [f32; 4] {
    [1.0; 4]
}
fn one_u() -> [u32; 2] {
    [1, 1]
}

/// One rising/looping billboard quad of an emitter.
#[derive(Component)]
struct FxQuad {
    base: Vec3,
    /// Loop phase offset in [0,1) — quads of one emitter are spread across the cycle.
    phase: f32,
    lifetime: f32,
    speed: f32,
    gravity: f32,
    size: f32,
}

/// Per-emitter flipbook driver: advances the SHARED material's uv_transform.
#[derive(Component)]
struct FxFlipbook {
    mat: Handle<StandardMaterial>,
    tiles: UVec2,
    /// frames per second, derived as frames x cycles / lifetime (Unity's lifetime time mode).
    fps: f32,
}

fn teardown_fx(mut commands: Commands, q: Query<Entity, Or<(With<FxQuad>, With<FxFlipbook>)>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn spawn_fx(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    pack: Option<Res<crate::render::LoadedPack>>,
) {
    if std::env::var("EFT_FX").map(|v| v.trim() == "0").unwrap_or(false) {
        return;
    }
    let Some(pack) = pack else { return };
    let path = pack.0.root.join("particles.json");
    let Ok(txt) = std::fs::read_to_string(&path) else {
        info!("fx: no particles.json (run eft_extract_particles.py) — no effects overlay");
        return;
    };
    let fx: FxFile = match serde_json::from_str(&txt) {
        Ok(v) => v,
        Err(e) => {
            warn!("fx: particles.json parse failed: {e}");
            return;
        }
    };

    let quad = meshes.add(Rectangle::new(1.0, 1.0));
    // Atlas image cache: emitters share the handful of fx textures.
    let mut tex_cache: std::collections::HashMap<String, Option<Handle<Image>>> = Default::default();
    let (mut n_quads, mut n_emit) = (0usize, 0usize);
    for e in &fx.emitters {
        let handle = tex_cache
            .entry(e.tex.clone())
            .or_insert_with(|| {
                let p = pack.0.root.join(&e.tex);
                image::open(&p).ok().map(|img| {
                    images.add(Image::from_dynamic(
                        img,
                        true, // fx atlases are authored sRGB color
                        RenderAssetUsages::RENDER_WORLD,
                    ))
                })
            })
            .clone();
        let Some(handle) = handle else { continue };
        // Blend family from the game's own shader name: the additive families glow (fire,
        // sparks, glow sprites); the rest alpha-blend (smoke).
        let sh = e.shader.to_ascii_lowercase();
        let additive = sh.contains("additive") || sh.contains("hdrfire");
        let rgba = [
            e.color[0] * e.tint[0],
            e.color[1] * e.tint[1],
            e.color[2] * e.tint[2],
            (e.color[3] * e.tint[3]).clamp(0.0, 1.0),
        ];
        let tiles = UVec2::new(e.tiles[0].max(1), e.tiles[1].max(1));
        // The cluster ACCUMULATES: n overlapping quads of the same flame sum toward the grade
        // LUT's clip plateau (blown cream blobs) where the game's HDR tonemap rolls off softly.
        // Normalize so the cluster's total energy ~ 2x one quad regardless of n.
        let n = ((e.rate * e.lifetime).ceil() as usize).clamp(1, 10);
        let norm = (2.0 / n as f32).min(1.0);
        let mat = materials.add(StandardMaterial {
            base_color_texture: Some(handle),
            base_color: Color::linear_rgba(rgba[0], rgba[1], rgba[2], rgba[3] * norm),
            alpha_mode: if additive { AlphaMode::Add } else { AlphaMode::Blend },
            unlit: true,
            cull_mode: None,
            uv_transform: bevy::math::Affine2::from_scale_angle_translation(
                Vec2::new(1.0 / tiles.x as f32, 1.0 / tiles.y as f32),
                0.0,
                Vec2::ZERO,
            ),
            ..default()
        });
        let frames = (tiles.x * tiles.y).max(1) as f32;
        commands.spawn(FxFlipbook {
            mat: mat.clone(),
            tiles,
            fps: if e.uv_enabled && frames > 1.0 {
                (frames * e.uv_cycles.max(0.1)) / e.lifetime.max(0.05)
            } else {
                0.0
            },
        });
        n_emit += 1;
        // Enough quads to cover the loop continuously, bounded so a 100-rate fire doesn't spawn
        // a hundred entities: phases are spread evenly, so even the cap reads as a full flame.
        for i in 0..n {
            // Deterministic per-quad jitter (no RNG dependency): golden-ratio scatter.
            let h = (i as f32 * 0.618_034) % 1.0;
            let ang = h * std::f32::consts::TAU;
            let jitter = Vec3::new(ang.cos(), 0.0, ang.sin()) * e.shape_radius * h.sqrt();
            commands.spawn((
                Mesh3d(quad.clone()),
                MeshMaterial3d(mat.clone()),
                Transform::from_translation(Vec3::from(e.pos) + jitter)
                    .with_scale(Vec3::splat(e.size.max(0.05))),
                FxQuad {
                    base: Vec3::from(e.pos) + jitter,
                    phase: i as f32 / n as f32,
                    lifetime: e.lifetime.max(0.05),
                    speed: e.speed,
                    gravity: e.gravity,
                    size: e.size.max(0.05),
                },
            ));
            n_quads += 1;
        }
    }
    info!(
        "fx: {n_emit} looping emitters -> {n_quads} flipbook billboards ({} atlases) from {}",
        tex_cache.values().filter(|v| v.is_some()).count(),
        path.display()
    );
}

/// Rise with startSpeed, fall with gravityModifier, loop over startLifetime; grow slightly and
/// vanish at wrap so the loop seam reads as turbulence rather than a pop.
fn animate_quads(time: Res<Time>, mut q: Query<(&FxQuad, &mut Transform)>) {
    let now = time.elapsed_secs();
    for (fx, mut tf) in &mut q {
        let t = ((now / fx.lifetime + fx.phase) % 1.0) * fx.lifetime;
        let y = fx.speed * t - 0.5 * fx.gravity * 9.81 * t * t;
        tf.translation = fx.base + Vec3::Y * y;
        // life fraction: ease in fast, fade the scale near the wrap.
        let lf = t / fx.lifetime;
        let grow = 0.75 + 0.45 * lf;
        let fade = (1.0 - lf).min(lf * 8.0).clamp(0.0, 1.0);
        tf.scale = Vec3::splat(fx.size * grow * (0.35 + 0.65 * fade));
    }
}

/// Advance each emitter's shared flipbook frame (row-major over the atlas grid).
fn animate_flipbooks(
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    q: Query<&FxFlipbook>,
) {
    let now = time.elapsed_secs();
    for fb in &q {
        if fb.fps <= 0.0 {
            continue;
        }
        let frames = (fb.tiles.x * fb.tiles.y).max(1);
        let f = (now * fb.fps) as u32 % frames;
        let (fx, fy) = (f % fb.tiles.x, f / fb.tiles.x);
        if let Some(m) = materials.get_mut(&fb.mat) {
            m.uv_transform.translation = Vec2::new(
                fx as f32 / fb.tiles.x as f32,
                fy as f32 / fb.tiles.y as f32,
            );
        }
    }
}

/// Face every quad toward the camera (billboards; the extractor drops stretched render modes'
/// orientation on purpose — v1 treats them as billboards too).
fn billboard_quads(
    cam: Query<&GlobalTransform, With<crate::render::CullCamera>>,
    mut q: Query<&mut Transform, With<FxQuad>>,
) {
    let Ok(cam) = cam.single() else { return };
    let rot = cam.to_scale_rotation_translation().1;
    for mut tf in &mut q {
        tf.rotation = rot;
    }
}
