//! Clip sampling, blend-tree evaluation, and pose accumulation.
//!
//! The emitter resampled every Unity curve encoding onto ONE uniform grid, so sampling here is a
//! frame index plus a lerp — there is deliberately no streamed/dense/constant handling in the
//! viewer, and adding a character can never introduce a new decode path.
//!
//! Blending is weighted accumulation per bone, not a chain of pairwise lerps: a 2D blend can have
//! nine active clips and pairwise lerping them is both slower and order-dependent. Rotations
//! accumulate as sign-aligned quaternion sums then normalise (nlerp), which for the small angular
//! spreads a locomotion blend produces is visually indistinguishable from proper slerp and is
//! linear in the number of clips.

use super::pack::{BlendNodeManifest, CharacterPack, ClipData};
use bevy::prelude::*;
use glam::{Quat, Vec3, Vec4};
use std::collections::HashMap;

/// Per-bone local transform accumulator. `weight` is tracked PER BONE because clips in a blend
/// legitimately drive different bone subsets — a sidestep clip may not touch the fingers, and
/// normalising by the blend's total weight instead would shrink those bones toward zero.
pub struct PoseAccumulator {
    pub pos: Vec<Vec3>,
    pub rot: Vec<Vec4>,
    pub scale: Vec<Vec3>,
    pub pos_weight: Vec<f32>,
    pub rot_weight: Vec<f32>,
    pub scale_weight: Vec<f32>,
    /// First non-zero rotation per bone, used to align later quaternions onto the same hemisphere.
    rot_ref: Vec<Option<Vec4>>,
}

impl PoseAccumulator {
    pub fn new(bones: usize) -> Self {
        Self {
            pos: vec![Vec3::ZERO; bones],
            rot: vec![Vec4::ZERO; bones],
            scale: vec![Vec3::ZERO; bones],
            pos_weight: vec![0.0; bones],
            rot_weight: vec![0.0; bones],
            scale_weight: vec![0.0; bones],
            rot_ref: vec![None; bones],
        }
    }

    pub fn clear(&mut self) {
        self.pos.fill(Vec3::ZERO);
        self.rot.fill(Vec4::ZERO);
        self.scale.fill(Vec3::ZERO);
        self.pos_weight.fill(0.0);
        self.rot_weight.fill(0.0);
        self.scale_weight.fill(0.0);
        self.rot_ref.fill(None);
    }

    fn add_rotation(&mut self, bone: usize, q: Quat, w: f32) {
        let mut v = Vec4::new(q.x, q.y, q.z, q.w);
        match self.rot_ref[bone] {
            Some(r) => {
                if r.dot(v) < 0.0 {
                    v = -v;
                }
            }
            None => self.rot_ref[bone] = Some(v),
        }
        self.rot[bone] += v * w;
        self.rot_weight[bone] += w;
    }

    /// Resolve into local transforms, falling back to the bind pose wherever nothing contributed.
    pub fn resolve(&self, pack: &CharacterPack, out: &mut [(Vec3, Quat, Vec3)]) {
        for (i, bone) in pack.bones.iter().enumerate() {
            let p = if self.pos_weight[i] > 1e-5 {
                self.pos[i] / self.pos_weight[i]
            } else {
                bone.local_pos
            };
            let r = if self.rot_weight[i] > 1e-5 {
                let v = self.rot[i];
                let q = Quat::from_xyzw(v.x, v.y, v.z, v.w);
                if q.length_squared() > 1e-8 {
                    q.normalize()
                } else {
                    bone.local_rot
                }
            } else {
                bone.local_rot
            };
            let s = if self.scale_weight[i] > 1e-5 {
                self.scale[i] / self.scale_weight[i]
            } else {
                bone.local_scale
            };
            out[i] = (p, r, s);
        }
    }
}

/// Sample one clip at `time` seconds and add it to the accumulator with weight `w`.
pub fn accumulate_clip(acc: &mut PoseAccumulator, clip: &ClipData, time: f32, w: f32) {
    if w <= 1e-5 || clip.frame_count == 0 {
        return;
    }
    let frames = clip.frame_count;
    let (i0, i1, a) = if frames == 1 || clip.duration <= 1e-6 {
        (0usize, 0usize, 0.0f32)
    } else {
        // frame_count = round(duration * rate) + 1, so the last frame sits exactly at `duration`.
        let t = if clip.looping {
            time.rem_euclid(clip.duration)
        } else {
            time.clamp(0.0, clip.duration)
        };
        let f = (t / clip.duration) * (frames - 1) as f32;
        let i0 = f.floor() as usize;
        let i0 = i0.min(frames - 1);
        let i1 = (i0 + 1).min(frames - 1);
        (i0, i1, f - i0 as f32)
    };

    for track in &clip.tracks {
        let b = track.bone;
        if !track.position.is_empty() {
            let p = track.position[i0].lerp(track.position[i1], a);
            acc.pos[b] += p * w;
            acc.pos_weight[b] += w;
        }
        if !track.rotation.is_empty() {
            let q0 = track.rotation[i0];
            let q1 = track.rotation[i1];
            // Within one clip the emitter already made the sequence sign-continuous, so a plain
            // nlerp between adjacent frames takes the short arc.
            let q = if a <= 0.0 { q0 } else { q0.slerp(q1, a) };
            acc.add_rotation(b, q, w);
        }
        if !track.scale.is_empty() {
            let s = track.scale[i0].lerp(track.scale[i1], a);
            acc.scale[b] += s * w;
            acc.scale_weight[b] += w;
        }
    }
}

/// A resolved blend-tree leaf: which controller clip id, and how much of it.
#[derive(Debug, Clone, Copy)]
pub struct WeightedClip {
    pub clip_id: i64,
    pub weight: f32,
}

/// Flatten a blend tree into weighted leaves for the given animator parameters.
pub fn eval_tree(
    node: &BlendNodeManifest,
    params: &HashMap<String, f32>,
    out: &mut Vec<WeightedClip>,
) {
    eval_node(node, params, 1.0, out);
    // Renormalise: gaps (a leaf whose clip was not extracted) and float error both leave the sum
    // slightly off, and an unnormalised pose reads as the character sinking or shrinking.
    let total: f32 = out.iter().map(|c| c.weight).sum();
    if total > 1e-5 {
        for c in out.iter_mut() {
            c.weight /= total;
        }
    }
}

fn eval_node(
    node: &BlendNodeManifest,
    params: &HashMap<String, f32>,
    weight: f32,
    out: &mut Vec<WeightedClip>,
) {
    if weight <= 1e-5 {
        return;
    }
    if node.kind == "clip" {
        if let Some(id) = node.clip {
            if id >= 0 {
                out.push(WeightedClip { clip_id: id, weight });
            }
        }
        return;
    }
    if node.children.is_empty() {
        return;
    }
    let p = |name: &str| params.get(name).copied().unwrap_or(0.0);

    let child_weights: Vec<f32> = match node.kind.as_str() {
        "1d" => weights_1d(node, p(&node.param_x)),
        "direct" => node
            .children
            .iter()
            .map(|_| 1.0 / node.children.len() as f32)
            .collect(),
        // Every 2D flavour goes through gradient-band interpolation. That IS Unity's algorithm for
        // Freeform Cartesian; for the two Directional flavours Unity uses a polar variant, so this
        // is an APPROXIMATION there. It is well behaved because EFT's directional children sit on a
        // unit circle, but it is not bit-exact with the game.
        _ => weights_gradient_band(node, Vec2::new(p(&node.param_x), p(&node.param_y))),
    };

    for (child, w) in node.children.iter().zip(child_weights) {
        eval_node(child, params, weight * w, out);
    }
}

/// Linear interpolation between the two bracketing thresholds; clamped outside the range.
fn weights_1d(node: &BlendNodeManifest, x: f32) -> Vec<f32> {
    let n = node.children.len();
    let mut w = vec![0.0; n];
    let t: Vec<f32> = node.children.iter().map(|c| c.threshold).collect();
    if n == 1 {
        w[0] = 1.0;
        return w;
    }
    if x <= t[0] {
        w[0] = 1.0;
        return w;
    }
    if x >= t[n - 1] {
        w[n - 1] = 1.0;
        return w;
    }
    for i in 0..n - 1 {
        if x >= t[i] && x <= t[i + 1] {
            let span = t[i + 1] - t[i];
            let a = if span.abs() < 1e-6 { 0.0 } else { (x - t[i]) / span };
            w[i] = 1.0 - a;
            w[i + 1] = a;
            return w;
        }
    }
    w[n - 1] = 1.0;
    w
}

/// Unity's Gradient Band Interpolation. For each child i, its weight is driven down by every other
/// child j in proportion to how far the sample point has moved from i toward j; the minimum over j
/// wins. Falls back to nearest-child when the sample lands outside every band.
fn weights_gradient_band(node: &BlendNodeManifest, sample: Vec2) -> Vec<f32> {
    let pos: Vec<Vec2> = node
        .children
        .iter()
        .map(|c| c.position.map(|p| Vec2::new(p[0], p[1])).unwrap_or(Vec2::ZERO))
        .collect();
    let n = pos.len();
    let mut w = vec![0.0f32; n];
    for i in 0..n {
        let mut wi = 1.0f32;
        for j in 0..n {
            if i == j {
                continue;
            }
            let v_ij = pos[j] - pos[i];
            let len2 = v_ij.length_squared();
            if len2 < 1e-8 {
                continue;
            }
            let v_ip = sample - pos[i];
            let t = 1.0 - v_ip.dot(v_ij) / len2;
            wi = wi.min(t.clamp(0.0, 1.0));
            if wi <= 0.0 {
                break;
            }
        }
        w[i] = wi;
    }
    let total: f32 = w.iter().sum();
    if total > 1e-5 {
        for x in w.iter_mut() {
            *x /= total;
        }
    } else if n > 0 {
        // Degenerate (e.g. all children coincident): snap to the nearest.
        let mut best = 0usize;
        let mut best_d = f32::MAX;
        for (i, p) in pos.iter().enumerate() {
            let d = (sample - *p).length_squared();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        w[best] = 1.0;
    }
    w
}
