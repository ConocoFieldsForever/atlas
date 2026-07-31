"""Build a `.eftchar` pack for one registered character.

    python extraction/characters/build_character.py --list
    python extraction/characters/build_character.py --character tagilla
    python extraction/characters/build_character.py --character tagilla --clips all
    python extraction/characters/build_character.py --character tagilla --skip-clips   # geometry only

Paths come from the environment, as elsewhere in extraction/:
    EFT_GAME_DATA  the game's EscapeFromTarkov_Data dir
    EFT_CHAR_OUT   output root (default <repo>/out/characters)

Ordering matters and is deliberate: the SKELETON is read first because every later join keys off its
path strings, and the CONTROLLER is parsed before the clips so its `m_TOS` can self-validate the
path digest before thousands of clip bindings are resolved against it. If the digest is wrong the
build stops in a second rather than emitting a pack full of unbound tracks.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Set

# Allow both `python extraction/characters/build_character.py` and `-m extraction.characters...`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from extraction.characters import clips as clips_mod  # noqa: E402
    from extraction.characters import controller as controller_mod  # noqa: E402
    from extraction.characters import pack as pack_mod  # noqa: E402
    from extraction.characters import skin as skin_mod  # noqa: E402
    from extraction.characters import validate as validate_mod  # noqa: E402
    from extraction.characters.skeleton import load_skeleton  # noqa: E402
else:
    from . import clips as clips_mod
    from . import controller as controller_mod
    from . import pack as pack_mod
    from . import skin as skin_mod
    from . import validate as validate_mod
    from .skeleton import load_skeleton

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REGISTRY = os.path.join(HERE, "characters.json")

EFT_DATA = os.environ.get(
    "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data"
)
CHAR_ROOT = os.path.join(EFT_DATA, "StreamingAssets", "Windows", "assets", "content", "characters")
OUT_ROOT = os.environ.get("EFT_CHAR_OUT") or os.path.join(REPO, "out", "characters")


def load_registry() -> dict:
    with open(REGISTRY, encoding="utf-8") as fh:
        return json.load(fh)


def game_build() -> str:
    """Read the installed build id so a pack records where it came from."""
    info = os.path.join(EFT_DATA, "app.info")
    if os.path.exists(info):
        try:
            with open(info, encoding="utf-8", errors="replace") as fh:
                txt = fh.read().strip().replace("\n", " ")
            if txt:
                return txt[:120]
        except OSError:
            pass
    return "unknown"


WINDOWS_ROOT = os.path.join(EFT_DATA, "StreamingAssets", "Windows")


def _resolve(rel: str) -> str:
    """Bundle path -> absolute. Relative to the characters root, unless it starts with `assets/`,
    which resolves from StreamingAssets/Windows so equipment outside the character tree is reachable.
    """
    base = WINDOWS_ROOT if rel.startswith("assets/") else CHAR_ROOT
    path = os.path.join(base, rel.replace("/", os.sep))
    if not os.path.exists(path):
        raise FileNotFoundError(f"bundle not found: {path}")
    return path


def _script_name(obj) -> str:
    try:
        return obj.read().m_Script.read().m_Name
    except Exception:
        return ""


class ControllerBundle:
    """A controller plus everything needed to resolve its clip references.

    `clips` is indexed by CLIP ID -- the `m_AnimationClips` PPtr order that states and blend trees
    reference -- and holds UnityPy object readers, so both the name and the typetree of a clip come
    from one place regardless of which bundle it physically lives in.
    """

    def __init__(self, ctrl_tt: dict, clips: List[object], containers: List[dict], rels: List[str]):
        self.typetree = ctrl_tt
        self.clips = clips
        self.containers = containers
        self.bundles = rels

    def name_of(self, clip_id: int) -> str:
        if not (0 <= clip_id < len(self.clips)) or self.clips[clip_id] is None:
            return ""
        try:
            return str(self.clips[clip_id].read().m_Name)  # type: ignore[union-attr]
        except Exception:
            return ""

    def names(self) -> List[str]:
        return [self.name_of(i) for i in range(len(self.clips))]


def load_controller(spec: dict, reg: dict) -> ControllerBundle:
    """Load the controller bundle TOGETHER WITH the shared animation bundles.

    They must share one UnityPy environment: `m_AnimationClips` entries with a non-zero `m_FileID`
    are external references, and UnityPy can only follow those to a file it has also loaded.
    """
    import UnityPy

    rels = [spec["controller"]] + list(reg["_shared"].get("animations") or [])
    env = UnityPy.load(*[_resolve(r) for r in rels])

    controllers: Dict[str, object] = {}
    containers: List[dict] = []
    for obj in env.objects:
        tn = obj.type.name
        if tn == "AnimatorController":
            controllers[str(obj.read_typetree().get("m_Name", ""))] = obj
        elif tn == "MonoBehaviour" and _script_name(obj) == "PlayerStateContainer":
            containers.append(obj.read_typetree())

    wanted = spec.get("controllerName")
    if wanted and wanted in controllers:
        ctrl_obj = controllers[wanted]
    elif len(controllers) == 1:
        ctrl_obj = next(iter(controllers.values()))
    else:
        raise SystemExit(
            f"{rels[0]} holds {len(controllers)} AnimatorControllers "
            f"({', '.join(sorted(controllers))}); set \"controllerName\" in characters.json"
        )

    # Resolve clip ids through the PARSED object so UnityPy follows cross-file PPtrs for us.
    clips: List[object] = []
    parsed = ctrl_obj.read()  # type: ignore[union-attr]
    for ptr in getattr(parsed, "m_AnimationClips", None) or []:
        reader = None
        try:
            deref = ptr.deref()
            reader = deref if deref is not None and deref.type.name == "AnimationClip" else None
        except Exception:
            reader = None
        clips.append(reader)

    return ControllerBundle(ctrl_obj.read_typetree(), clips, containers, rels)  # type: ignore[union-attr]


def dump_states(character: str, grep: Optional[str]) -> None:
    """Print a character's animator states with their clips -- how you author a clipSet."""
    reg = load_registry()
    spec = reg["characters"][character]
    cb = load_controller(spec, reg)
    clip_names = cb.names()
    table = controller_mod.parse_controller(
        cb.typetree, clip_names, cb.containers, validate_hashes=True
    )

    unnamed = sum(1 for n in clip_names if not n)
    print(f"controller {table.name!r} from {' + '.join(cb.bundles)}")
    print(f"  {len(table.layers)} layers, {len(table.states)} states, "
          f"{len(table.parameters)} parameters, {len(clip_names)} clip slots"
          + (f" ({unnamed} UNRESOLVED)" if unnamed else " (all resolved)"))
    print("\nparameters:")
    for p in table.parameters:
        print(f"  {p.type:8s} {p.name}  default={p.default}")
    print("\nlayers:")
    for l in table.layers:
        print(f"  [{l.index}] {l.name!r} sm={l.state_machine} weight={l.default_weight} "
              f"blend={l.blending}")
    print("\nstates:")
    needle = grep.lower() if grep else None
    for s in sorted(table.states, key=lambda s: (s.layer, s.full_path)):
        if needle and needle not in s.full_path.lower() and needle not in s.name.lower():
            continue
        kids = ""
        if s.tree is not None:
            t = s.tree
            leaves = s.leaf_clips()
            names = [
                clip_names[c] if 0 <= c < len(clip_names) else f"<{c}>" for c in leaves[:4]
            ]
            params = "/".join(p for p in (t.param_x, t.param_y) if p) or "-"
            nested = sorted(
                {
                    "/".join(p for p in (c.param_x, c.param_y) if p)
                    for c in t.children
                    if not c.is_leaf
                }
            )
            kids = f"  {t.kind}[{params}]"
            if nested:
                kids += " x " + ",".join(f"{n}" for n in nested)
            kids += f" {len(leaves)} clips {names}"
            if len(leaves) > 4:
                kids += " ..."
        gp = ""
        if s.gameplay:
            gp = "  gameplay=" + ",".join(
                f"{k}={v}" for k, v in s.gameplay.items() if k in ("Type", "DisableRootMotion",
                                                                   "RotationSpeedClamp")
            )
        print(f"  L{s.layer} {s.full_path or s.name}  speed={s.speed} loop={s.loop}"
              f"{kids}{gp}")


def build(
    character: str,
    clip_set: Optional[str],
    skip_clips: bool,
    lods: Optional[Sequence[int]],
    strict: bool,
    out_dir: Optional[str],
) -> str:
    reg = load_registry()
    chars = reg["characters"]
    if isinstance(character, dict):
        # A RESOLVED SPEC (extraction/characters/appearance.py): the bot's own weighted
        # appearance roll, resolved through the game's customization table. This is the
        # authoritative path — characters.json entries are hand-authored and only kept for
        # named one-offs and for facts the tables do not carry.
        spec = dict(character)
        spec.setdefault("clipSets", reg.get("clipSets") or {})
        character = spec.get("id") or spec.get("displayName", "rolled").replace(" ", "_").replace("#", "")
    elif character in chars:
        spec = chars[character]
    else:
        raise SystemExit(
            f"unknown character {character!r}; known: {', '.join(sorted(chars))} "
            f"(or build a rolled bot with --bot <type> [--seed N])")

    t0 = time.time()
    used_bundles: List[str] = []

    # ---- 1. canonical rig ----
    skel_rel = reg["_shared"]["skeleton"]
    print(f"[skeleton] {skel_rel}")
    skel = load_skeleton(_resolve(skel_rel))
    used_bundles.append(skel_rel)
    print(f"  {len(skel)} bones, root={skel.names[0]!r}, deepest path len="
          f"{max(len(p.split('/')) for p in skel.paths)}")

    # ---- 2. body parts ----
    lod_filter = tuple(lods) if lods else None
    meshes: List[skin_mod.SkinMesh] = []
    materials: List[skin_mod.Material] = []
    images: Dict[str, object] = {}
    for rel in spec["parts"]:
        print(f"[part] {rel}")
        part_name = os.path.splitext(os.path.basename(rel))[0]
        res = skin_mod.load_part(
            _resolve(rel), part_name, skel, material_base=len(materials), strict=strict, lods=lod_filter
        )
        for name, img in res.images.items():
            images.setdefault(name, img)
        materials.extend(res.materials)
        meshes.extend(res.meshes)
        used_bundles.append(rel)
        for m in res.meshes:
            print(f"  {m.name}: {m.vertex_count} verts, {len(m.submeshes)} submesh(es), "
                  f"{len(m.bound_bones)} bones")

    # ---- 2b. rigid equipment (helmet / facecover / cap) ----
    attachments: List[skin_mod.Attachment] = []
    for eq in spec.get("equipment") or []:
        rel = eq["bundle"]
        bone_name = eq["bone"]
        bone = skel.by_name.get(bone_name)
        if bone is None:
            print(f"  [warn] equipment {rel}: rig has no bone {bone_name!r} -- skipped")
            continue
        print(f"[equipment] {rel} -> {bone_name} (bone {bone})")
        atts, mats, imgs = skin_mod.load_attachment(
            _resolve(rel), bone, material_base=len(materials), lods=lod_filter
        )
        for name, img in imgs.items():
            images.setdefault(name, img)
        materials.extend(mats)
        attachments.extend(atts)
        used_bundles.append(rel)
        for a in atts:
            print(f"  {a.name}: {a.vertex_count} verts, {len(a.submeshes)} submesh(es)")

    # ---- 3. controller (before clips: its TOS validates the path digest) ----
    ctrl_table: Optional[controller_mod.ControllerTable] = None
    clip_objs: List[dict] = []
    clip_name_order: List[str] = []
    if not skip_clips and spec.get("controller"):
        print(f"[controller] {spec['controller']}")
        cb = load_controller(spec, reg)
        used_bundles.extend(cb.bundles)
        print(f"  using {cb.typetree.get('m_Name')!r}, "
              f"{len(cb.containers)} PlayerStateContainer(s)")

        # m_AnimationClips PPtr order defines the clip ids that states/blend trees reference.
        clip_name_order = cb.names()
        unnamed = sum(1 for n in clip_name_order if not n)
        print(f"  {len(clip_name_order)} clip slots"
              + (f", {unnamed} unresolved" if unnamed else ", all resolved"))

        ctrl_table = controller_mod.parse_controller(
            cb.typetree, clip_name_order, cb.containers, validate_hashes=True
        )
        print(f"  {len(ctrl_table.layers)} layers, {len(ctrl_table.states)} states, "
              f"{len(ctrl_table.parameters)} parameters")

        # ---- 4. clip selection, in CLIP-ID space ----
        # Selecting by id rather than by name matters: a clip set names STATES, a state's blend
        # trees reference clip ids, and several ids can legitimately alias one clip asset.
        set_name = clip_set or spec.get("defaultClipSet") or "all"
        # A character's own sets shadow the global ones: state paths are graph-specific.
        available = {**reg.get("clipSets", {}), **spec.get("clipSets", {})}
        wanted_ids: Optional[Set[int]] = None
        if set_name != "all":
            if set_name not in available:
                raise SystemExit(
                    f"unknown clip set {set_name!r} for {character}; "
                    f"known: {', '.join(sorted(available))}"
                )
            state_names = available[set_name]
            if state_names is not None:
                wanted_ids = set()
                missing: List[str] = []
                for sname in state_names:
                    st = ctrl_table.state_by_name(sname)
                    if st is None:
                        missing.append(sname)
                        continue
                    wanted_ids.update(c for c in st.leaf_clips() if c >= 0)
                if missing:
                    print(f"  [warn] clip set {set_name!r}: no state named {missing}")
                if not wanted_ids:
                    raise SystemExit(
                        f"clip set {set_name!r} resolved to 0 clips. State names differ between the "
                        f"player graph and the bot graphs -- run --dump-states to see this "
                        f"character's actual states."
                    )

        # Deduplicate by ASSET (path_id), not by name: two distinct clip assets can share a name.
        # Tagilla's graph has two different `crouch_run_aim_0`s, and only one is an absolute-pose
        # clip -- so everything downstream must key off the controller clip id, never the name.
        seen_paths: Set[int] = set()
        for cid, reader in enumerate(cb.clips):
            if reader is None:
                continue
            if wanted_ids is not None and cid not in wanted_ids:
                continue
            if reader.path_id in seen_paths:
                continue
            seen_paths.add(reader.path_id)
            clip_objs.append((reader.path_id, reader.read_typetree()))
        dup_names = len(clip_objs) - len({tt.get("m_Name") for _, tt in clip_objs})
        print(f"[clips] set={set_name}, "
              + (f"{len(wanted_ids)} id(s) -> " if wanted_ids else "")
              + f"{len(clip_objs)} unique asset(s)"
              + (f", {dup_names} sharing a name with another" if dup_names else ""))

    # ---- 5. decode clips ----
    decoded: List[clips_mod.Clip] = []
    unresolved_total = 0
    index_by_path: Dict[int, int] = {}
    for path_id, tt in clip_objs:
        try:
            c = clips_mod.decode_clip(tt, skel, strict=strict)
        except Exception as exc:
            msg = f"clip {tt.get('m_Name')!r}: {exc}"
            if strict:
                raise SystemExit(f"[fatal] {msg}")
            print(f"  [warn] skipped {msg}")
            continue
        unresolved_total += len(c.unresolved)
        index_by_path[path_id] = len(decoded)
        decoded.append(c)
    rooted = sum(1 for c in decoded if c.root_motion is not None)
    print(f"  root motion stripped from {rooted}/{len(decoded)} clips")

    # controller clip id -> index into the emitted clips[], or -1 when not extracted. THE
    # authoritative lookup; names are for humans only.
    clip_index_by_id: List[int] = []
    if not skip_clips and spec.get("controller"):
        for reader in cb.clips:
            clip_index_by_id.append(
                index_by_path.get(reader.path_id, -1) if reader is not None else -1
            )
    if decoded:
        bones = sorted({t.bone for c in decoded for t in c.tracks})
        print(f"  decoded {len(decoded)} clips, {len(bones)} distinct bones animated, "
              f"{unresolved_total} non-bone bindings ignored")

        # ---- 5b. anatomical validation ----
        # A wrong rotation basis produces clean, connected, unit-quaternion garbage that nothing in
        # the decode can detect. Measure the composed pose instead.
        #
        # Reference clips are resolved BY CLIP ID from the states that actually drive standing
        # locomotion -- not by name. Name lookup would sometimes pick the additive-delta twin of a
        # clip, whose poses are deltas and are meaningless to compose absolutely.
        ref_states = spec.get("validateStates") or [
            "Base Layer.Stand.Idle_Aim",
            "Base Layer.StateMachine_Move.MOVE",
        ]
        ref_ids: List[int] = []
        for sname in ref_states:
            st = ctrl_table.state_by_name(sname) if ctrl_table else None
            if st is not None:
                ref_ids.extend(st.leaf_clips())
        to_check: List[clips_mod.Clip] = []
        seen_idx: Set[int] = set()
        for cid in ref_ids:
            if 0 <= cid < len(clip_index_by_id):
                di = clip_index_by_id[cid]
                if di >= 0 and di not in seen_idx:
                    seen_idx.add(di)
                    to_check.append(decoded[di])
        if to_check:
            checked, complaints = validate_mod.validate_clips(skel, to_check, strict=strict)
            if complaints:
                print(f"  [warn] {len(complaints)} anatomical complaint(s) across {checked} poses")
                for c in complaints[:5]:
                    print(f"    {c}")
            else:
                print(f"  pose validation OK ({checked} poses over {len(to_check)} clips "
                      f"reached from {len(ref_states)} state(s))")
        else:
            print("  [warn] no reference clip resolved -- pose validation SKIPPED")

    # ---- 6. derive the character's forward axis ----
    # Do NOT hardcode a facing offset. The forward-walk clip's own root motion IS the character's
    # forward axis, in the pack's already-conjugated space, so the viewer can align "character
    # forward" to "movement direction" with no magic 180.
    forward_axis: Optional[List[float]] = None
    fwd_clip_name = spec.get("forwardClip", "walk_aim_0")
    fwd_clip = next((c for c in decoded if c.name == fwd_clip_name), None)
    if fwd_clip is not None and fwd_clip.average_speed:
        v = np.asarray(fwd_clip.average_speed, np.float64)
        v[1] = 0.0  # forward is horizontal
        n = float(np.linalg.norm(v))
        if n > 1e-3:
            forward_axis = [round(float(x), 6) for x in (v / n)]
            print(f"[forward] derived from {fwd_clip_name!r} root motion: {forward_axis} "
                  f"({n:.2f} m/s)")
    if forward_axis is None and not skip_clips:
        print(f"  [warn] no forward axis derived (clip {fwd_clip_name!r} absent or static); "
              f"the viewer will fall back to +Z")

    # ---- 7. write ----
    target = out_dir or os.path.join(OUT_ROOT, character)
    manifest = pack_mod.write_pack(
        target,
        pack_mod.BuildInfo(
            character=character,
            display_name=spec.get("displayName", character),
            game_build=game_build(),
            bundles=used_bundles,
        ),
        skel,
        meshes,
        materials,
        decoded,
        ctrl_table,
        images,
        default_lod=int(spec.get("lod", 0)),
        extra={"characterForward": forward_axis or [0.0, 0.0, 1.0],
               "characterForwardDerived": forward_axis is not None},
        clip_index_by_id=clip_index_by_id,
        attachments=attachments,
    )
    blob = manifest["blobs"]
    print(
        f"[done] {target}\n"
        f"  {len(meshes)} meshes, {len(materials)} materials, {len(manifest['textures'])} textures, "
        f"{len(decoded)} clips\n"
        f"  skin.bin {blob['skin']['totalByteLength']/1e6:.1f} MB, "
        f"anim.bin {blob['anim']['totalByteLength']/1e6:.1f} MB, "
        f"{time.time()-t0:.1f}s"
    )
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", help="registry id, e.g. tagilla")
    ap.add_argument("--list", action="store_true", help="list registered characters and clip sets")
    ap.add_argument("--clips", help="clip set name (default: the character's defaultClipSet)")
    ap.add_argument("--skip-clips", action="store_true", help="geometry only -- fast pipeline check")
    ap.add_argument("--lod", type=int, action="append", help="keep only these LODs (repeatable)")
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="downgrade join/layout failures to warnings. Diagnostics only: a non-strict pack may "
        "contain a silently wrong bone remap.",
    )
    ap.add_argument("--out", help="explicit output dir")
    ap.add_argument(
        "--dump-states",
        action="store_true",
        help="print the character's animator states, parameters and per-state clips, then exit. "
        "This is how you author a clipSet -- state names differ between the player graph and the "
        "bot graphs.",
    )
    ap.add_argument("--grep", help="with --dump-states, filter states by substring")
    ap.add_argument("--bot", help="bot type from the game's own tables (assault, pmcusec, "
                                  "bosskilla, ...) — appearance is ROLLED, not authored")
    ap.add_argument("--seed", type=int, default=0, help="roll seed for --bot (default 0)")
    args = ap.parse_args()

    if args.list:
        reg = load_registry()
        print("characters:")
        for k, v in sorted(reg["characters"].items()):
            print(f"  {k:12s} {v.get('displayName','')}  parts={len(v['parts'])}")
        print("clip sets:", ", ".join(sorted(reg["clipSets"])))
        return

    if args.bot:
        from appearance import resolve as resolve_appearance
        reg = load_registry()
        spec = resolve_appearance(args.bot, args.seed, clip_sets=reg.get("clipSets"))
        spec["id"] = f"{args.bot}_{args.seed}"
        print(f"[appearance] {args.bot} #{args.seed}: "
              + ", ".join(f"{k}={v['name']}" for k, v in spec["appearance"].items()))
        build(character=spec, clip_set=args.clips, skip_clips=args.skip_clips, lods=args.lod,
              strict=not args.no_strict, out_dir=args.out)
        return

    if not args.character:
        ap.error("--character, --bot or --list is required")

    if args.dump_states:
        dump_states(args.character, args.grep)
        return

    build(
        character=args.character,
        clip_set=args.clips,
        skip_clips=args.skip_clips,
        lods=args.lod,
        strict=not args.no_strict,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
