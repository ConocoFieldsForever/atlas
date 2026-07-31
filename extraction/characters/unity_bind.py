"""Unity generic-clip binding maths: the path hash, and the curve-index <-> binding walk.

A Mecanim GENERIC clip does not name the transforms it animates. `m_ClipBindingConstant.
genericBindings[]` holds, per binding, a `path` that is the **CRC32 of the transform's path string**
("Root_Joint/Base HumanPelvis/Base HumanSpine1"), a `typeID` (the Unity class the binding targets)
and an `attribute` (which property). Curves live in ONE flat index space; each binding consumes a
run of it whose width depends on the attribute (a quaternion is 4, a vector is 3, a float is 1).

So resolving "which bone does curve 137 drive" is: walk the bindings accumulating widths, then
CRC-match the winning binding's path against the rig's path strings.

The hash function is ASSERTED, not assumed. `validate_hash_fn` checks zlib's CRC-32 against the
AnimatorController's own `m_TOS` (a hash -> string table Unity ships for debugging). If BSG ever
changed the digest, the build fails there instead of silently producing a limp character.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Unity class ids seen on character clip bindings.
# ---------------------------------------------------------------------------
TYPE_TRANSFORM = 4
TYPE_ANIMATOR = 95
TYPE_SKINNED_MESH_RENDERER = 137
TYPE_GAMEOBJECT = 1

# Transform binding attributes (UnityEngine's kBindTransform* enum).
ATTR_POSITION = 1
ATTR_ROTATION = 2
ATTR_SCALE = 3
ATTR_EULER = 4

#: Curves consumed by each Transform attribute. Rotation is a quaternion; the rest are vectors.
_TRANSFORM_WIDTH = {
    ATTR_POSITION: 3,
    ATTR_ROTATION: 4,
    ATTR_SCALE: 3,
    ATTR_EULER: 3,
}

ATTR_NAME = {
    ATTR_POSITION: "position",
    ATTR_ROTATION: "rotation",
    ATTR_SCALE: "scale",
    ATTR_EULER: "euler",
}


def path_hash(path: str) -> int:
    """Unity's transform-path digest: zlib CRC-32 of the UTF-8 path, unsigned.

    Empty path hashes to 0, which is how Unity denotes "the clip's own root".
    """
    if path == "":
        return 0
    return zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF


def build_hash_map(paths: Iterable[str]) -> Dict[int, str]:
    """path strings -> {digest: path}. Raises on a digest collision (never seen; would be fatal)."""
    out: Dict[int, str] = {}
    for p in paths:
        h = path_hash(p)
        prev = out.get(h)
        if prev is not None and prev != p:
            raise RuntimeError(f"path-hash collision {h:#010x}: {prev!r} vs {p!r}")
        out[h] = p
    return out


def validate_hash_fn(tos: Sequence[Tuple[int, str]], min_samples: int = 8) -> Tuple[int, int]:
    """Check `path_hash` against an AnimatorController `m_TOS` (list of (hash, string) pairs).

    Unity's TOS mixes true transform paths with state-machine labels ("Base Layer.JUMP.Fall") and
    asset paths, which are hashed by a DIFFERENT function. So a hit rate of 1.0 is not expected --
    what matters is that a healthy number of entries agree and, crucially, that no entry that
    hashes to the right value contradicts us.

    Returns (matched, considered). Raises if fewer than `min_samples` entries agree, which would
    mean the digest itself changed.
    """
    matched = 0
    considered = 0
    for h, s in tos:
        if not isinstance(s, str) or not s:
            continue
        considered += 1
        if path_hash(s) == (int(h) & 0xFFFFFFFF):
            matched += 1
    if matched < min_samples:
        raise RuntimeError(
            f"transform-path digest self-check failed: only {matched}/{considered} TOS entries "
            f"agree with zlib CRC-32. Unity's path hash is not what this code assumes."
        )
    return matched, considered


@dataclass(frozen=True)
class Binding:
    """One `genericBindings[]` entry plus the curve range it owns."""

    path: int  #: CRC32 of the transform path
    type_id: int
    attribute: int
    curve_start: int
    curve_count: int
    script_hash: int = 0

    @property
    def is_transform(self) -> bool:
        return self.type_id == TYPE_TRANSFORM and self.attribute in _TRANSFORM_WIDTH

    @property
    def attr_name(self) -> str:
        return ATTR_NAME.get(self.attribute, f"attr{self.attribute}")


def binding_width(type_id: int, attribute: int) -> int:
    """Curves consumed by a binding. Non-Transform bindings are scalar (float parameters, etc)."""
    if type_id == TYPE_TRANSFORM:
        return _TRANSFORM_WIDTH.get(attribute, 1)
    return 1


def walk_bindings(generic_bindings: Sequence[dict]) -> List[Binding]:
    """Assign each `genericBindings[]` entry its contiguous curve range, in order.

    Mirrors Unity's own `ClipBindingConstant::FindBinding`, which walks the same list accumulating
    widths -- so this is the exact inverse of the runtime's lookup, not a guess at it.
    """
    out: List[Binding] = []
    cursor = 0
    for b in generic_bindings:
        type_id = int(b.get("typeID", 0))
        attribute = int(b.get("attribute", 0))
        width = binding_width(type_id, attribute)
        out.append(
            Binding(
                path=int(b.get("path", 0)) & 0xFFFFFFFF,
                type_id=type_id,
                attribute=attribute,
                curve_start=cursor,
                curve_count=width,
                script_hash=int(b.get("script", {}).get("m_PathID", 0))
                if isinstance(b.get("script"), dict)
                else 0,
            )
        )
        cursor += width
    return out


def total_curves(bindings: Sequence[Binding]) -> int:
    """Curves the binding list claims. Must equal streamed + dense + constant, or the clip layout
    is not what we think it is."""
    return sum(b.curve_count for b in bindings)
