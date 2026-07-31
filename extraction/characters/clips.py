"""Mecanim `AnimationClip` -> uniform per-bone position / rotation / scale tracks.

EFT's character clips are GENERIC (not humanoid): their bindings are `typeID 4` (Transform) with
attributes 1/2/3 over the rig's bones. That is the easy case -- curves map straight onto bones with
no muscle-space retargeting -- but the payload is still stored in Unity's three concurrent
encodings, concatenated into one curve-index space in this exact order:

    [ m_StreamedClip curves ][ m_DenseClip curves ][ m_ConstantClip curves ]

  * STREAMED -- keys at arbitrary times, each carrying the CUBIC COEFFICIENTS of the segment that
    starts at it: v(dt) = c0*dt^3 + c1*dt^2 + c2*dt + c3, dt = t - keyTime. Evaluated as that cubic
    directly; converting to Hermite tangents and back (what some tools do) only loses precision.
  * DENSE -- baked frame-major samples: sample[frame * curveCount + curve], frame times are
    beginTime + frame / sampleRate. Reconstructed with linear interpolation, which is what baking
    to a fixed rate means.
  * CONSTANT -- one value for all time.

This module RESAMPLES all three onto a single uniform grid at the clip's own rate, so the viewer
ships exactly one sampler and can never grow a second decode path when a new character is added.

Self-validation: the bindings claim a total curve count, and the three encodings supply one. If they
disagree the clip layout is not what this code believes and the build fails -- that assertion is the
whole reason to trust the output, because a mis-sliced curve space produces animation that looks
plausible while being wrong.
"""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coords
from .skeleton import Skeleton
from .unity_bind import (
    ATTR_EULER,
    ATTR_POSITION,
    ATTR_ROTATION,
    ATTR_SCALE,
    Binding,
    total_curves,
    walk_bindings,
)

DEFAULT_SAMPLE_RATE = 30.0
#: Refuse to bake absurd grids; a clip this long is a cutscene, not locomotion.
MAX_FRAMES = 8192
#: A bone whose animated position travels further than this (m, per axis) is carrying root motion
#: rather than restating its constant bind offset.
ROOT_MOTION_EPS = 0.02


@dataclass
class BoneTrack:
    """Per-bone animation for one clip. Channels absent from the clip stay None (the viewer then
    keeps the bind-pose value), which is both smaller and semantically right."""

    bone: int
    position: Optional[np.ndarray] = None  #: (F, 3) f32, viewer space
    rotation: Optional[np.ndarray] = None  #: (F, 4) f32 xyzw, viewer space, sign-continuous
    scale: Optional[np.ndarray] = None  #: (F, 3) f32


@dataclass
class Clip:
    name: str
    duration: float
    sample_rate: float
    frame_count: int
    loop: bool
    tracks: List[BoneTrack] = field(default_factory=list)
    #: Bindings that did not resolve to a rig bone (Animator float params, unbound props). Kept for
    #: reporting -- silently dropping them is how you fail to notice a broken rig join.
    unresolved: List[str] = field(default_factory=list)
    #: Average root velocity (m/s, viewer space) measured from the stripped root motion. The clip's
    #: contribution to locomotion speed; the authoritative per-state figure still comes from the
    #: RootMotionBlendTable.
    average_speed: Optional[List[float]] = None
    #: Rig bone that carried root motion, and the per-frame displacement STRIPPED off it (F, 3),
    #: relative to frame 0. Removed from the bone track so a consumer cannot double-apply it: the
    #: walk physics already moves the character through the world, and leaving the clip's own 4 m of
    #: forward travel in the skeleton makes the body slide out from under the camera along the clip's
    #: axis. Kept here because it is still the right source for foot-slide-free playback rate.
    root_motion_bone: Optional[int] = None
    root_motion: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Unity curve containers
# ---------------------------------------------------------------------------
@dataclass
class _StreamedKey:
    time: float
    coeff: Tuple[float, float, float, float]


def _decode_streamed(streamed: dict) -> Dict[int, List[_StreamedKey]]:
    """`m_StreamedClip` -> {curve index: keys sorted by time}.

    The blob is uint32 words reinterpreted as a byte stream of frames:
        f32 time, i32 keyCount, keyCount * { i32 curveIndex, f32 coeff[4] }
    Unity brackets the real frames with sentinels at non-finite times; those are skipped.
    """
    words = streamed.get("data") or []
    if not words:
        return {}
    raw = np.asarray(words, np.uint32).tobytes()
    out: Dict[int, List[_StreamedKey]] = {}
    r = io.BytesIO(raw)
    while True:
        head = r.read(8)
        if len(head) < 8:
            break
        time, key_count = struct.unpack("<fi", head)
        if key_count < 0 or key_count > 1 << 20:
            break  # corrupt / past the end of meaningful data
        body = r.read(key_count * 20)
        if len(body) < key_count * 20:
            break
        finite = math.isfinite(time)
        for k in range(key_count):
            idx, c0, c1, c2, c3 = struct.unpack_from("<iffff", body, k * 20)
            if finite:
                out.setdefault(idx, []).append(_StreamedKey(time, (c0, c1, c2, c3)))
    for keys in out.values():
        keys.sort(key=lambda k: k.time)
    return out


def _eval_streamed(keys: Sequence[_StreamedKey], times: np.ndarray) -> np.ndarray:
    """Evaluate one streamed curve on `times` as the piecewise cubic it is."""
    if not keys:
        return np.zeros(times.shape[0], np.float32)
    kt = np.asarray([k.time for k in keys], np.float64)
    coeff = np.asarray([k.coeff for k in keys], np.float64)
    # Segment containing each sample: the last key at or before t (clamped to the first key).
    seg = np.clip(np.searchsorted(kt, times, side="right") - 1, 0, len(keys) - 1)
    dt = times - kt[seg]
    c = coeff[seg]
    v = ((c[:, 0] * dt + c[:, 1]) * dt + c[:, 2]) * dt + c[:, 3]
    return v.astype(np.float32)


def _eval_dense(dense: dict, curve: int, times: np.ndarray) -> np.ndarray:
    """Evaluate dense curve `curve` (already rebased to 0) on `times` with linear reconstruction."""
    frame_count = int(dense.get("m_FrameCount", 0))
    curve_count = int(dense.get("m_CurveCount", 0))
    samples = dense.get("m_SampleArray") or []
    if frame_count <= 0 or curve_count <= 0 or not samples:
        return np.zeros(times.shape[0], np.float32)
    rate = float(dense.get("m_SampleRate", DEFAULT_SAMPLE_RATE)) or DEFAULT_SAMPLE_RATE
    begin = float(dense.get("m_BeginTime", 0.0))
    arr = np.asarray(samples, np.float32).reshape(frame_count, curve_count)
    col = arr[:, curve]
    if frame_count == 1:
        return np.full(times.shape[0], col[0], np.float32)
    f = (times - begin) * rate
    f = np.clip(f, 0.0, frame_count - 1)
    i0 = np.floor(f).astype(np.int64)
    i1 = np.minimum(i0 + 1, frame_count - 1)
    a = (f - i0).astype(np.float32)
    return (col[i0] * (1.0 - a) + col[i1] * a).astype(np.float32)


class _CurveSet:
    """The clip's whole curve space, addressable by absolute curve index."""

    def __init__(self, clip_container: dict) -> None:
        self.streamed_raw = clip_container.get("m_StreamedClip", {}) or {}
        self.dense = clip_container.get("m_DenseClip", {}) or {}
        self.constant = clip_container.get("m_ConstantClip", {}) or {}
        self.streamed_keys = _decode_streamed(self.streamed_raw)
        self.n_streamed = int(self.streamed_raw.get("curveCount", 0) or 0)
        self.n_dense = int(self.dense.get("m_CurveCount", 0) or 0)
        self.constant_data = np.asarray(self.constant.get("data") or [], np.float32)
        self.n_constant = int(self.constant_data.size)

    @property
    def total(self) -> int:
        return self.n_streamed + self.n_dense + self.n_constant

    def evaluate(self, curve: int, times: np.ndarray) -> np.ndarray:
        if curve < self.n_streamed:
            return _eval_streamed(self.streamed_keys.get(curve, []), times)
        curve -= self.n_streamed
        if curve < self.n_dense:
            return _eval_dense(self.dense, curve, times)
        curve -= self.n_dense
        if curve < self.n_constant:
            return np.full(times.shape[0], self.constant_data[curve], np.float32)
        raise IndexError(f"curve index past the end of the curve space ({self.total} curves)")


# ---------------------------------------------------------------------------
# euler -> quaternion (Unity order)
# ---------------------------------------------------------------------------
def _axis_rot(axis: str, deg: np.ndarray) -> np.ndarray:
    """(F,) degrees about one axis -> (F, 3, 3) rotation matrices."""
    r = np.radians(deg.astype(np.float64))
    c, s = np.cos(r), np.sin(r)
    z, o = np.zeros_like(c), np.ones_like(c)
    if axis == "x":
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def _euler_to_quat(deg: np.ndarray) -> np.ndarray:
    """Euler rotation curve (degrees) -> RAW-Unity-space quaternion (F, 4) xyzw.

    Composition is `Rz(z) @ Ry(y) @ Rx(x)`, i.e. as an INTRINSIC sequence: rotate about X, then Y,
    then Z. That is **Maya's default XYZ rotate order**, which is what you would expect from a
    Maya-authored rig whose bones are named `Base Human*` -- so this is a convention match, not a
    tuned constant. Note it is NOT `Quaternion.Euler`'s ZXY order; assuming that is what broke the
    first attempt.

    This matters more than the quaternion path: euler is the DOMINANT encoding here. 108 of Tagilla's
    117 locomotion clips are euler-encoded, and `idle_aim` drives 57 of its 78 bones with euler
    curves against only 21 with quaternions.

    Solved against ground truth rather than guessed. Two distinct `idle_aim` assets ship, one pure
    quaternion and one euler-heavy; decoding the quaternion one through the validated quaternion path
    gives a reference pose. Searching orders, sign patterns, and Maya jointOrient pre-multiplication
    against it, this composition wins with a **median error of 0.81 degrees** over the 46 shared
    bones. The residual outliers are the weapon-holding chain (both palms ~175 deg, `Weapon_root`
    162 deg) -- where a hammer take and a rifle take genuinely differ, not a convention error. The
    jointOrient hypothesis (`bind * euler`) was tested and lost.

    Returning RAW space (not viewer space) matters: `coords.quats` then applies the G3 conjugation
    uniformly for euler and quaternion curves alike, so exactly one place knows about the mirror.
    """
    deg = np.atleast_2d(np.asarray(deg, np.float64))
    m = _axis_rot("z", deg[:, 2]) @ _axis_rot("y", deg[:, 1]) @ _axis_rot("x", deg[:, 0])
    return _matrix_to_quat(m)


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """(F, 3, 3) rotation matrices -> (F, 4) xyzw quaternions, branch-per-largest-diagonal."""
    f = m.shape[0]
    out = np.zeros((f, 4), np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(f):
        mm = m[i]
        tr = trace[i]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            out[i] = [
                (mm[2, 1] - mm[1, 2]) / s,
                (mm[0, 2] - mm[2, 0]) / s,
                (mm[1, 0] - mm[0, 1]) / s,
                0.25 * s,
            ]
        else:
            d = int(np.argmax([mm[0, 0], mm[1, 1], mm[2, 2]]))
            if d == 0:
                s = np.sqrt(1.0 + mm[0, 0] - mm[1, 1] - mm[2, 2]) * 2.0
                out[i] = [
                    0.25 * s,
                    (mm[0, 1] + mm[1, 0]) / s,
                    (mm[0, 2] + mm[2, 0]) / s,
                    (mm[2, 1] - mm[1, 2]) / s,
                ]
            elif d == 1:
                s = np.sqrt(1.0 + mm[1, 1] - mm[0, 0] - mm[2, 2]) * 2.0
                out[i] = [
                    (mm[0, 1] + mm[1, 0]) / s,
                    0.25 * s,
                    (mm[1, 2] + mm[2, 1]) / s,
                    (mm[0, 2] - mm[2, 0]) / s,
                ]
            else:
                s = np.sqrt(1.0 + mm[2, 2] - mm[0, 0] - mm[1, 1]) * 2.0
                out[i] = [
                    (mm[0, 2] + mm[2, 0]) / s,
                    (mm[1, 2] + mm[2, 1]) / s,
                    0.25 * s,
                    (mm[1, 0] - mm[0, 1]) / s,
                ]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n > 1e-12, n, 1.0)).astype(np.float32)


def _make_sign_continuous(q: np.ndarray) -> np.ndarray:
    """Flip whole quaternions so consecutive frames take the short way round.

    Generic rotation curves are stored component-wise, so a clip can legally contain q and -q on
    adjacent frames -- identical rotations that a naive nlerp walks the long way between.
    """
    out = q.copy()
    for i in range(1, out.shape[0]):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] = -out[i]
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def decode_clip(
    typetree: dict,
    skel: Skeleton,
    strict: bool = True,
    max_frames: int = MAX_FRAMES,
) -> Clip:
    """One `AnimationClip` typetree -> a `Clip` of viewer-space per-bone tracks.

    Handles both rotation encodings EFT uses: quaternion curves (attribute 2) and euler curves
    (attribute 4). Euler dominates -- 108 of Tagilla's 117 locomotion clips -- so see
    `_euler_to_quat`, which is where the hard-won part lives.
    """
    name = str(typetree.get("m_Name", "clip"))

    if typetree.get("m_Legacy"):
        raise RuntimeError(f"{name}: legacy clip -- curves live in m_RotationCurves, unhandled")

    muscle = typetree.get("m_MuscleClip") or {}
    if not muscle:
        raise RuntimeError(f"{name}: no m_MuscleClip (not a Mecanim clip?)")

    # UnityPy wraps nested serialized structs in an extra "data" key.
    container = muscle.get("m_Clip") or {}
    container = container.get("data", container)
    curves = _CurveSet(container)

    binding_const = typetree.get("m_ClipBindingConstant") or {}
    bindings: List[Binding] = walk_bindings(binding_const.get("genericBindings") or [])
    claimed = total_curves(bindings)
    if claimed != curves.total:
        msg = (
            f"{name}: binding list claims {claimed} curves but the clip supplies {curves.total} "
            f"(streamed {curves.n_streamed} + dense {curves.n_dense} + const {curves.n_constant}) "
            f"-- the curve-index layout is not what this decoder assumes"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"  [warn] {msg}")

    start = float(muscle.get("m_StartTime", 0.0))
    stop = float(muscle.get("m_StopTime", 0.0))
    duration = max(0.0, stop - start)
    rate = float(curves.dense.get("m_SampleRate", 0.0) or 0.0)
    if rate <= 0.0:
        rate = float(typetree.get("m_SampleRate", 0.0) or 0.0)
    if rate <= 0.0:
        rate = DEFAULT_SAMPLE_RATE

    frame_count = int(round(duration * rate)) + 1
    frame_count = max(1, min(frame_count, max_frames))
    if frame_count == max_frames:
        print(f"  [warn] {name}: {duration:.2f}s clipped to {max_frames} frames")
    times = (start + np.arange(frame_count, dtype=np.float64) / rate).clip(start, max(start, stop))

    by_hash = skel.by_hash
    tracks: Dict[int, BoneTrack] = {}
    unresolved: List[str] = []

    def track_for(bone: int) -> BoneTrack:
        t = tracks.get(bone)
        if t is None:
            t = BoneTrack(bone=bone)
            tracks[bone] = t
        return t

    euler_pending: Dict[int, np.ndarray] = {}

    for b in bindings:
        if not b.is_transform:
            unresolved.append(f"type{b.type_id}/attr{b.attribute}@{b.path:#010x}")
            continue
        bone = by_hash.get(b.path)
        if bone is None:
            unresolved.append(f"transform/{b.attr_name}@{b.path:#010x}")
            continue
        if b.curve_start + b.curve_count > curves.total:
            if strict:
                raise RuntimeError(f"{name}: binding {b} runs off the end of the curve space")
            continue

        cols = [
            curves.evaluate(b.curve_start + i, times) for i in range(b.curve_count)
        ]
        data = np.column_stack(cols)
        t = track_for(bone)
        if b.attribute == ATTR_POSITION:
            t.position = coords.points(data[:, :3])
        elif b.attribute == ATTR_ROTATION:
            # Quaternion curves ARE `Transform.localRotation` values: G3 conjugation and nothing
            # else. (An earlier X flip here appeared to help, but it was only compensating for a
            # broken euler conversion -- the two encodings were being scored against each other.)
            t.rotation = _make_sign_continuous(coords.quats(data[:, :4]))
        elif b.attribute == ATTR_SCALE:
            t.scale = data[:, :3].astype(np.float32)
        elif b.attribute == ATTR_EULER:
            euler_pending[bone] = data[:, :3]

    # Euler curves only win where the clip gave no quaternion curve for that bone. The same
    # curve-basis X flip is applied for consistency, but NOTE: no clip in EFT's character locomotion
    # set uses euler curves, so this path is unverified against the game.
    for bone, deg in euler_pending.items():
        t = track_for(bone)
        if t.rotation is None:
            t.rotation = _make_sign_continuous(coords.quats(_euler_to_quat(deg)))

    ordered = [tracks[k] for k in sorted(tracks)]

    # ---- strip root motion ----
    # Find it rather than assume a bone index: the carrier is the LOWEST-indexed bone (i.e. nearest
    # the rig root) whose animated position actually travels. Every other bone's position curve just
    # restates its constant bind offset, so the threshold separates them cleanly.
    root_motion_bone: Optional[int] = None
    root_motion: Optional[np.ndarray] = None
    for t in ordered:
        if t.position is None or len(t.position) < 2:
            continue
        span = float(np.abs(t.position.max(axis=0) - t.position.min(axis=0)).max())
        if span > ROOT_MOTION_EPS:
            root_motion_bone = t.bone
            root_motion = (t.position - t.position[0]).astype(np.float32)
            # Pin the bone to its frame-0 local offset; the travel now lives only in root_motion.
            t.position = np.repeat(t.position[:1], t.position.shape[0], axis=0)
            break

    average_speed: Optional[List[float]] = None
    if root_motion is not None and duration > 1e-6:
        average_speed = [float(v / duration) for v in root_motion[-1]]

    loop = bool(muscle.get("m_LoopTime", False))

    return Clip(
        name=name,
        duration=duration,
        sample_rate=rate,
        frame_count=frame_count,
        loop=loop,
        tracks=ordered,
        unresolved=unresolved,
        average_speed=average_speed,
        root_motion_bone=root_motion_bone,
        root_motion=root_motion,
    )
