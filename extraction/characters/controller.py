"""`AnimatorController` + `PlayerStateContainer` -> a locomotion state table.

SCOPE, stated up front: this does NOT reimplement Unity's Animator. EFT's player graph is 10 layers
with additive aiming, synced layers and body masks; faithfully re-running that is a different
project. What the viewer needs to drive a character from its own `WalkState` is smaller:

  * the PARAMETERS the graph is steered by, with defaults;
  * the STATES, each with its clip (or its blend tree's clips + thresholds) and playback speed;
  * the TRANSITIONS between them, with durations and conditions;
  * BSG's own per-state gameplay metadata from the `PlayerStateContainer` MonoBehaviours --
    `RotationSpeedClamp`, `StateSensitivity`, `DisableRootMotion`, `AnimationAuthority`,
    `AdditionalDirectionInfo`. This is the bridge between the animation graph and the movement state
    machine, and it is the reason to read the controller at all rather than hand-authoring a blend.

Extracted as data. A viewer-side blender consumes it; a future, more faithful runtime can consume
the same table without re-extraction.

Unity nests serialized structs under an extra "data" key when read through a type tree, so every
access goes through `_d`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .unity_bind import validate_hash_fn

#: Animator parameter type ids.
PARAM_TYPES = {1: "float", 3: "int", 4: "bool", 5: "trigger", 9: "bool"}


def _d(node) -> dict:
    """Unwrap UnityPy's nested-struct "data" indirection."""
    if isinstance(node, dict) and set(node.keys()) == {"data"}:
        return node["data"]
    return node if isinstance(node, dict) else {}


@dataclass
class Parameter:
    name: str
    type: str
    default: object = None


#: Unity `BlendTreeType`. A node with no children and a real clip id is a LEAF regardless of the
#: type byte, which for leaves is left at 0.
BLEND_TYPES = {
    0: "1d",
    1: "2d_simple_directional",
    2: "2d_freeform_directional",
    3: "2d_freeform_cartesian",
    4: "direct",
}


@dataclass
class BlendNode:
    """A node in a blend tree. Trees NEST -- EFT's `MOVE` is a 9-way directional blend on
    (Direct_X, Direct_Y) whose every direction is itself a 2D blend on (Speed, Level), so a
    flat "root's children" read returns nine nodes with no clips at all. Hence recursion.

    `threshold` / `position` describe where THIS node sits in its PARENT's blend space, which keeps
    the geometry with the node instead of in a parallel array the consumer has to re-zip.
    """

    kind: str  #: "clip" for a leaf, else a BLEND_TYPES value
    clip: int = -1  #: leaf only; index into the controller's m_AnimationClips
    param_x: str = ""
    param_y: str = ""
    threshold: float = 0.0  #: position in the parent's 1D space
    position: Tuple[float, float] = (0.0, 0.0)  #: position in the parent's 2D space
    timescale: float = 1.0
    cycle_offset: float = 0.0
    mirror: bool = False
    children: List["BlendNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.kind == "clip"

    def leaf_clips(self) -> List[int]:
        """Every clip id reachable from here, depth-first. This is what a clipSet expands to."""
        if self.is_leaf:
            return [self.clip] if self.clip >= 0 else []
        out: List[int] = []
        for c in self.children:
            out.extend(c.leaf_clips())
        return out

    def to_manifest(self) -> dict:
        out: dict = {"kind": self.kind}
        if self.is_leaf:
            out["clip"] = self.clip
        else:
            if self.param_x:
                out["paramX"] = self.param_x
            if self.param_y:
                out["paramY"] = self.param_y
            out["children"] = [c.to_manifest() for c in self.children]
        if self.threshold:
            out["threshold"] = self.threshold
        if self.position != (0.0, 0.0):
            out["position"] = list(self.position)
        if self.timescale != 1.0:
            out["timescale"] = self.timescale
        if self.cycle_offset:
            out["cycleOffset"] = self.cycle_offset
        if self.mirror:
            out["mirror"] = True
        return out


@dataclass
class Transition:
    target: str  #: destination state name ("" for exit)
    duration: float = 0.0
    offset: float = 0.0
    exit_time: float = 0.0
    has_exit_time: bool = False
    #: (parameter, mode, threshold) triples, mode as Unity's condition enum.
    conditions: List[Tuple[str, int, float]] = field(default_factory=list)


@dataclass
class State:
    name: str  #: leaf name, e.g. "MOVE"
    full_path: str  #: "Base Layer.StateMachine_Move.MOVE"
    layer: int
    speed: float = 1.0
    loop: bool = False
    mirror: bool = False
    cycle_offset: float = 0.0
    speed_param: str = ""
    #: One entry per SYNCHRONIZED LAYER SLOT of the owning state machine (`m_BlendTreeConstant
    #: IndexArray`), so a state can drive different trees on the base layer and on each synced
    #: layer. `None` where that slot has no tree. Slot 0 is the owning layer.
    trees: List[Optional[BlendNode]] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    #: BSG's PlayerStateContainer fields, verbatim, when a container names this state.
    gameplay: Dict[str, object] = field(default_factory=dict)

    @property
    def tree(self) -> Optional[BlendNode]:
        """The owning layer's tree -- what a simple consumer wants."""
        return self.trees[0] if self.trees else None

    def leaf_clips(self) -> List[int]:
        out: List[int] = []
        for t in self.trees:
            if t is not None:
                out.extend(t.leaf_clips())
        return out


@dataclass
class Layer:
    index: int
    name: str
    state_machine: int
    default_weight: float = 0.0
    blending: str = "override"
    ik_pass: bool = False
    #: Unity `m_StateMachineSynchronizedLayerIndex`. Several layers legitimately share one state
    #: machine (EFT has `Base Layer`, `Sync_SprintHands` and `TagillaSyncLayerForRegularOperations`
    #: all on state machine 0); the FIRST layer referencing a machine owns it, the rest are synced
    #: views. Getting this backwards attributes every base-layer state to the last synced layer.
    synchronized_layer: int = 0
    owns_state_machine: bool = True


@dataclass
class ControllerTable:
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    layers: List[Layer] = field(default_factory=list)
    states: List[State] = field(default_factory=list)
    #: index -> clip name, from the controller's own m_AnimationClips PPtrs resolved by the caller.
    clip_names: List[str] = field(default_factory=list)

    def state_by_name(self, name: str) -> Optional[State]:
        for s in self.states:
            if s.name == name or s.full_path == name:
                return s
        return None

    def to_manifest(self) -> dict:
        return {
            "name": self.name,
            "parameters": [{"name": p.name, "type": p.type, "default": p.default} for p in self.parameters],
            "layers": [
                {
                    "index": l.index,
                    "name": l.name,
                    "stateMachine": l.state_machine,
                    "defaultWeight": l.default_weight,
                    "blending": l.blending,
                    "ikPass": l.ik_pass,
                    "synchronizedLayer": l.synchronized_layer,
                    "ownsStateMachine": l.owns_state_machine,
                }
                for l in self.layers
            ],
            "clipNames": self.clip_names,
            "states": [
                {
                    "name": s.name,
                    "fullPath": s.full_path,
                    "layer": s.layer,
                    "speed": s.speed,
                    "loop": s.loop,
                    "mirror": s.mirror,
                    "cycleOffset": s.cycle_offset,
                    "speedParam": s.speed_param,
                    "trees": [None if t is None else t.to_manifest() for t in s.trees],
                    "transitions": [
                        {
                            "target": t.target,
                            "duration": t.duration,
                            "offset": t.offset,
                            "exitTime": t.exit_time,
                            "hasExitTime": t.has_exit_time,
                            "conditions": [list(c) for c in t.conditions],
                        }
                        for t in s.transitions
                    ],
                    "gameplay": s.gameplay,
                }
                for s in self.states
            ],
        }


def _leaf(full_path: str) -> str:
    return full_path.rsplit(".", 1)[-1] if full_path else ""


def parse_controller(
    typetree: dict,
    clip_names: Sequence[str],
    state_containers: Sequence[dict],
    validate_hashes: bool = True,
) -> ControllerTable:
    """Build the state table from an `AnimatorController` typetree.

    `clip_names` must be the resolved names of `m_AnimationClips`, in order -- state/blend-tree clip
    ids index into it. `state_containers` are the bundle's `PlayerStateContainer` typetrees.
    """
    name = str(typetree.get("m_Name", "controller"))
    tos_pairs: List[Tuple[int, str]] = [
        (int(h) & 0xFFFFFFFF, str(s)) for h, s in (typetree.get("m_TOS") or [])
    ]
    tos = dict(tos_pairs)
    if validate_hashes and tos_pairs:
        # Confirms the transform-path digest used everywhere else in this package.
        validate_hash_fn(tos_pairs)

    ctrl = _d(typetree.get("m_Controller") or {})
    table = ControllerTable(name=name, clip_names=list(clip_names))

    # ---- parameters ----
    values = _d(ctrl.get("m_Values") or {})
    defaults = _d(ctrl.get("m_DefaultValues") or {})
    float_d = defaults.get("m_FloatValues") or []
    int_d = defaults.get("m_IntValues") or []
    bool_d = defaults.get("m_BoolValues") or []
    for entry in values.get("m_ValueArray") or []:
        e = _d(entry)
        pid = int(e.get("m_ID", 0)) & 0xFFFFFFFF
        ptype = PARAM_TYPES.get(int(e.get("m_Type", 0)), f"type{e.get('m_Type')}")
        idx = int(e.get("m_Index", 0))
        default: object = None
        if ptype == "float" and idx < len(float_d):
            default = float(float_d[idx])
        elif ptype == "int" and idx < len(int_d):
            default = int(int_d[idx])
        elif ptype in ("bool", "trigger") and idx < len(bool_d):
            default = bool(bool_d[idx])
        table.parameters.append(
            Parameter(name=tos.get(pid, f"param_{pid:#010x}"), type=ptype, default=default)
        )
    def pname(pid: int) -> str:
        return tos.get(int(pid) & 0xFFFFFFFF, "")

    # ---- layers ----
    blend_modes = {0: "override", 1: "additive"}
    sm_to_layer: Dict[int, int] = {}
    for i, layer in enumerate(ctrl.get("m_LayerArray") or []):
        l = _d(layer)
        smi = int(l.get("m_StateMachineIndex", 0))
        # FIRST layer to reference a state machine owns it; later ones are synced views. Building
        # this map with a comprehension instead would let the last synced layer claim the machine.
        owns = smi not in sm_to_layer
        if owns:
            sm_to_layer[smi] = i
        table.layers.append(
            Layer(
                index=i,
                name=pname(l.get("m_Binding", 0)) or f"Layer {i}",
                state_machine=smi,
                default_weight=float(l.get("m_DefaultWeight", 0.0)),
                blending=blend_modes.get(int(l.get("(int&)m_LayerBlendingMode", 0) or 0), "override"),
                ik_pass=bool(l.get("m_IKPass", False)),
                synchronized_layer=int(l.get("m_StateMachineSynchronizedLayerIndex", 0)),
                owns_state_machine=owns,
            )
        )

    # ---- per-state gameplay metadata, joined by name ----
    gameplay_by_name: Dict[str, Dict[str, object]] = {}
    for sc in state_containers:
        nm = str(sc.get("Name", "") or "")
        if not nm:
            continue
        gameplay_by_name[nm] = {
            k: sc[k]
            for k in (
                "Type",
                "IsDefaultState",
                "AdditionalDirectionInfo",
                "RotationSpeedClamp",
                "StateSensitivity",
                "CanInteract",
                "DisableRootMotion",
                "CreateUniqueMovementStateObject",
                "AnimationAuthority",
            )
            if k in sc
        }

    # ---- states ----
    for smi, sm in enumerate(ctrl.get("m_StateMachineArray") or []):
        machine = _d(sm)
        layer_idx = sm_to_layer.get(smi, smi)
        for sc in machine.get("m_StateConstantArray") or []:
            st = _d(sc)
            full = tos.get(int(st.get("m_FullPathID", 0)) & 0xFFFFFFFF, "")
            leaf = tos.get(int(st.get("m_NameID", 0)) & 0xFFFFFFFF, "") or _leaf(full)
            state = State(
                name=leaf,
                full_path=full,
                layer=layer_idx,
                speed=float(st.get("m_Speed", 1.0)),
                loop=bool(st.get("m_Loop", False)),
                mirror=bool(st.get("m_Mirror", False)),
                cycle_offset=float(st.get("m_CycleOffset", 0.0)),
                speed_param=pname(st.get("m_SpeedParamID", 0)),
                gameplay=gameplay_by_name.get(leaf, {}),
            )
            state.trees = _parse_trees(st, pname)
            for tc in st.get("m_TransitionConstantArray") or []:
                t = _d(tc)
                dest_full = tos.get(int(t.get("m_DestinationState", 0)) & 0xFFFFFFFF, "")
                conditions: List[Tuple[str, int, float]] = []
                for cc in t.get("m_ConditionConstantArray") or []:
                    c = _d(cc)
                    conditions.append(
                        (
                            pname(c.get("m_EventID", 0)),
                            int(c.get("m_ConditionMode", 0)),
                            float(c.get("m_EventThreshold", 0.0)),
                        )
                    )
                state.transitions.append(
                    Transition(
                        target=_leaf(dest_full) or dest_full,
                        duration=float(t.get("m_TransitionDuration", 0.0)),
                        offset=float(t.get("m_TransitionOffset", 0.0)),
                        exit_time=float(t.get("m_ExitTime", 0.0)),
                        has_exit_time=bool(t.get("m_HasExitTime", False)),
                        conditions=conditions,
                    )
                )
            table.states.append(state)

    return table


#: Unity writes an absent clip id as unsigned -1.
_NO_CLIP = 0xFFFFFFFF


def _parse_trees(state: dict, pname) -> List[Optional[BlendNode]]:
    """State -> one blend tree per synchronized layer slot.

    `m_BlendTreeConstantArray` lives on the STATE (not the state machine), and
    `m_BlendTreeConstantIndexArray` has one entry per synced layer slot indexing into it, with -1
    meaning "this state contributes nothing on that layer". EFT's `MOVE` is `[0, -1, 1]`: a tree on
    the base layer, nothing on the sprint-hands sync slot, a different tree on the third.
    """
    trees_raw = state.get("m_BlendTreeConstantArray") or []
    idx_array = [int(i) for i in (state.get("m_BlendTreeConstantIndexArray") or [])]
    if not trees_raw:
        return []
    if not idx_array:
        idx_array = [0]

    out: List[Optional[BlendNode]] = []
    for slot in idx_array:
        if slot < 0 or slot >= len(trees_raw):
            out.append(None)
            continue
        nodes = [_d(n) for n in (_d(trees_raw[slot]).get("m_NodeArray") or [])]
        out.append(_build_node(nodes, 0, pname, set()) if nodes else None)
    return out


def _build_node(
    nodes: Sequence[dict], index: int, pname, seen: set
) -> Optional[BlendNode]:
    """Recursively materialise node `index`. `seen` guards against a malformed cyclic graph."""
    if index < 0 or index >= len(nodes) or index in seen:
        return None
    seen = seen | {index}
    raw = nodes[index]

    child_idx = [int(i) for i in (raw.get("m_ChildIndices") or [])]
    clip_id = int(raw.get("m_ClipID", _NO_CLIP)) & 0xFFFFFFFF
    duration = float(raw.get("m_Duration", 1.0) or 1.0)

    node = BlendNode(
        kind="clip" if not child_idx else BLEND_TYPES.get(int(raw.get("m_BlendType", 0)), "1d"),
        clip=-1 if clip_id == _NO_CLIP else clip_id,
        param_x="" if not child_idx else pname(raw.get("m_BlendEventID", 0)),
        param_y="" if not child_idx else pname(raw.get("m_BlendEventYID", 0)),
        timescale=duration,
        cycle_offset=float(raw.get("m_CycleOffset", 0.0)),
        mirror=bool(raw.get("m_Mirror", False)),
    )
    if not child_idx:
        return node

    thresholds = [
        float(v)
        for v in (_d(raw.get("m_Blend1dData") or {}).get("m_ChildThresholdArray") or [])
    ]
    positions = [
        (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
        for p in (_d(raw.get("m_Blend2dData") or {}).get("m_ChildPositionArray") or [])
    ]

    for n, ci in enumerate(child_idx):
        child = _build_node(nodes, ci, pname, seen)
        if child is None:
            continue
        if n < len(thresholds):
            child.threshold = thresholds[n]
        if n < len(positions):
            child.position = positions[n]
        node.children.append(child)
    return node
