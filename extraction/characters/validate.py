"""Anatomical validation of a decoded clip. The guard that catches a wrong rotation basis.

A misread rotation curve does not produce garbage — it produces a character that is *connected*,
unit-quaternion clean, and completely wrong. Positions match the bind pose exactly, scales are 1.0,
every quaternion is normalised, and the figure is still folded double. Nothing in the decode itself
can detect that.

What CAN detect it is the skeleton's own geometry: a standing human's feet are on the floor, the head
is at head height, and the spine points up. So compose the pose and measure. See
`clips._curve_quat_to_transform` for the bug this exists to catch.

Bounds are deliberately loose — wide enough for a crouch run or a prone clip to pass, tight enough
that a 60-degree basis error cannot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .clips import Clip
from .skeleton import Skeleton, _trs

#: Names used for the measurements. Absent bones simply skip that check.
PELVIS = "Base HumanPelvis"
HEAD = "Base HumanHead"
FEET = ("Base HumanLFoot", "Base HumanRFoot")
#: (shoulder, hand) pairs. Arm reach is checked because the FIRST version of this validator measured
#: only pelvis/head/feet POSITIONS and happily passed a pose whose head was twisted down onto the
#: chest and whose arms were inside the torso -- the euler-conversion bug. Positions of three bones
#: do not constrain a skeleton.
ARMS = (
    ("Base HumanLCollarbone", "Base HumanLPalm"),
    ("Base HumanRCollarbone", "Base HumanRPalm"),
)

#: The lowest foot must be near the floor. Generous upward slack covers mid-stride and jump clips.
FOOT_Y_RANGE = (-0.20, 0.45)
#: Head height above the character origin. Covers prone (low) through standing.
HEAD_Y_RANGE = (0.15, 2.05)
#: Pelvis->head angle from +Y. A crouch run reaches ~28 deg; the X-basis bug produced 60-130 deg.
MAX_TILT_DEG = 50.0
#: Distance pelvis->foot, i.e. leg reach. Catches the "legs folded to half length" signature.
LEG_REACH_RANGE = (0.35, 1.15)
#: Distance collarbone->palm. A straight arm is ~0.62 m on this rig and a fully folded one ~0.20 m,
#: so anything outside this is a broken arm chain rather than a pose.
ARM_REACH_RANGE = (0.15, 0.80)
#: A hand should stay near the torso's vertical span, not end up under the feet or above the head.
HAND_Y_RANGE = (-0.10, 2.20)


@dataclass
class PoseMeasurement:
    clip: str
    frame: int
    head_y: float
    min_foot_y: float
    tilt_deg: float
    leg_reach: float
    #: (label, collarbone->palm distance, palm world Y) per arm.
    arms: List[Tuple[str, float, float]] = None  # type: ignore[assignment]

    def problems(self, prone: bool = False) -> List[str]:
        out: List[str] = []
        for label, reach, hand_y in self.arms or []:
            if not ARM_REACH_RANGE[0] <= reach <= ARM_REACH_RANGE[1]:
                out.append(f"{label} shoulder->hand {reach:.3f} m (want {ARM_REACH_RANGE})")
            if not HAND_Y_RANGE[0] <= hand_y <= HAND_Y_RANGE[1]:
                out.append(f"{label} hand at y={hand_y:.3f} (want {HAND_Y_RANGE})")
        if not FOOT_Y_RANGE[0] <= self.min_foot_y <= FOOT_Y_RANGE[1]:
            out.append(f"lowest foot at y={self.min_foot_y:.3f} (want {FOOT_Y_RANGE})")
        if not HEAD_Y_RANGE[0] <= self.head_y <= HEAD_Y_RANGE[1]:
            out.append(f"head at y={self.head_y:.3f} (want {HEAD_Y_RANGE})")
        if not prone and self.tilt_deg > MAX_TILT_DEG:
            out.append(f"spine tilted {self.tilt_deg:.1f} deg from vertical (max {MAX_TILT_DEG})")
        if not LEG_REACH_RANGE[0] <= self.leg_reach <= LEG_REACH_RANGE[1]:
            out.append(f"pelvis->foot reach {self.leg_reach:.3f} m (want {LEG_REACH_RANGE})")
        return out


def _world_matrices(skel: Skeleton, clip: Clip, frame: int) -> np.ndarray:
    """Forward-compose the pose at `frame`: clip values where present, bind pose elsewhere."""
    n = len(skel)
    pos = [skel.local_pos[i].astype(np.float64) for i in range(n)]
    rot = [skel.local_rot[i].astype(np.float64) for i in range(n)]
    scl = [skel.local_scale[i].astype(np.float64) for i in range(n)]
    for t in clip.tracks:
        f = min(frame, clip.frame_count - 1)
        if t.position is not None and len(t.position):
            pos[t.bone] = t.position[min(f, len(t.position) - 1)].astype(np.float64)
        if t.rotation is not None and len(t.rotation):
            rot[t.bone] = t.rotation[min(f, len(t.rotation) - 1)].astype(np.float64)
        if t.scale is not None and len(t.scale):
            scl[t.bone] = t.scale[min(f, len(t.scale) - 1)].astype(np.float64)

    out = np.zeros((n, 4, 4), np.float64)
    for i in range(n):
        local = _trs(pos[i], rot[i], scl[i])
        p = skel.parents[i]
        out[i] = local if p < 0 else out[p] @ local
    return out


def measure(skel: Skeleton, clip: Clip, frame: int = 0) -> Optional[PoseMeasurement]:
    """Measure one frame. Returns None if the rig lacks the reference bones."""
    names = skel.by_name
    if PELVIS not in names or HEAD not in names:
        return None
    feet = [names[f] for f in FEET if f in names]
    if not feet:
        return None
    w = _world_matrices(skel, clip, frame)
    pelvis = w[names[PELVIS]][:3, 3]
    head = w[names[HEAD]][:3, 3]
    foot_ys = [float(w[f][1, 3]) for f in feet]
    spine = head - pelvis
    norm = float(np.linalg.norm(spine))
    tilt = 0.0 if norm < 1e-6 else float(np.degrees(np.arccos(np.clip(spine[1] / norm, -1.0, 1.0))))
    reach = max(float(np.linalg.norm(pelvis - w[f][:3, 3])) for f in feet)
    arms: List[Tuple[str, float, float]] = []
    for shoulder, hand in ARMS:
        if shoulder in names and hand in names:
            sp = w[names[shoulder]][:3, 3]
            hp = w[names[hand]][:3, 3]
            arms.append((hand, float(np.linalg.norm(hp - sp)), float(hp[1])))
    return PoseMeasurement(
        arms=arms,
        clip=clip.name,
        frame=frame,
        head_y=float(head[1]),
        min_foot_y=min(foot_ys),
        tilt_deg=tilt,
        leg_reach=reach,
    )


def validate_clips(
    skel: Skeleton,
    clips: Sequence[Clip],
    strict: bool = True,
) -> Tuple[int, List[str]]:
    """Measure a few frames of each given clip. Returns (checked, complaints).

    The caller chooses WHICH clips by resolving controller clip ids from standing-locomotion states —
    deliberately not by name, because a name can resolve to an additive-delta twin whose poses are
    deltas and are meaningless to compose absolutely. Only standing/walking clips belong here anyway;
    "feet on the floor, spine up" says nothing about a prone or vault clip. A handful is enough: the
    rotation basis is global, so if it is wrong it is wrong everywhere.
    """
    complaints: List[str] = []
    checked = 0
    for clip in clips:
        prone = "prone" in clip.name.lower()
        frames = sorted({0, clip.frame_count // 3, 2 * clip.frame_count // 3})
        for f in frames:
            mm = measure(skel, clip, f)
            if mm is None:
                continue
            checked += 1
            for p in mm.problems(prone=prone):
                complaints.append(f"{clip.name} frame {f}: {p}")
    if complaints and strict:
        joined = "\n  ".join(complaints[:12])
        more = f"\n  ... and {len(complaints) - 12} more" if len(complaints) > 12 else ""
        raise RuntimeError(
            "decoded clips fail anatomical validation — the rotation basis is almost certainly "
            f"wrong (see clips._curve_quat_to_transform):\n  {joined}{more}"
        )
    return checked, complaints
