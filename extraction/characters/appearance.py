#!/usr/bin/env python3
"""Resolve WHAT A BOT LOOKS LIKE, from the game's own data. The single source of truth.

A character in EFT is not a hand-picked set of prefabs — it is a weighted roll over that bot
type's `appearance` table (body / feet / hands / head), whose ids resolve through
`customization.json` to prefab bundles. This module does exactly that and nothing else, so every
consumer (the character builder, the kit baker, the viewer's roster) sees the same answer.

Why this exists: the roster in characters.json used to carry HAND-PICKED parts. Measured against
the game, that scav was wrong — it wore `head_civilian_1`, which is not in the scav appearance
table at all (the game rolls wild_head_1/2/3/drozd/misha), and it had no hands. Anything a bot
wears is now derived; characters.json survives only for the few facts the tables do not carry.

The one thing appearance data does NOT provide is which ANIMATOR drives the body. The game ships
exactly three bot controllers (base / boar / tagilla) plus their root-motion tables, so the
choice is a bounded lookup over what exists on disk, not an open guess: a bot type whose name
contains a controller's stem uses it, everything else uses base. The result is reported so a
wrong pick is visible rather than silent.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SHARED = os.path.join(REPO, "packs", "shared")

#: Where the game keeps character content, relative to StreamingAssets/Windows.
CHAR_ROOT = "assets/content/characters"
CTRL_DIR = f"{CHAR_ROOT}/controllers/animationcontrollers"
RM_DIR = f"{CHAR_ROOT}/rootmotiontable"

#: Appearance slots, in the order the rig wants them. Body first: it carries the most bones.
SLOTS = ("body", "feet", "hands", "head")

#: Slots that contribute GEOMETRY to a third-person character.
#:
#: `hands` is deliberately absent. The game's hands slot points at the FIRST-PERSON hand prefabs
#: (assets/content/hands/..., e.g. wild_body_1_firsthands), which bind the FPV hands skeleton —
#: a different rig — and are what the player sees down their own arms. A third-person body mesh
#: already includes its arms and hands, so building the hands slot would both fail the rig check
#: and duplicate geometry. It is still RESOLVED and reported, because it is part of the roll and
#: a first-person view would need exactly it.
BUILD_SLOTS = ("body", "feet", "head")


class AppearanceError(RuntimeError):
    pass


def _load(name):
    p = os.path.join(SHARED, name)
    if not os.path.exists(p):
        raise AppearanceError(
            f"{name} is not banked — run extraction/characters/fetch_bot_db.py")
    return json.load(open(p, encoding="utf-8"))


def load_sources():
    """(bots, customization) — the two tables this module reads."""
    return _load("bot_loadouts.json"), _load("customization.json")


def weighted_pick(rng, table):
    """One id from the game's own {id: weight} table."""
    if not table:
        return None
    ids, ws = zip(*[(k, max(float(v), 0.0)) for k, v in table.items()])
    total = sum(ws)
    if total <= 0:
        return rng.choice(ids)
    r, acc = rng.random() * total, 0.0
    for i, w in zip(ids, ws):
        acc += w
        if r <= acc:
            return i
    return ids[-1]


def controller_for(bot_type, game_root):
    """(controller_rel, rootmotion_rel) for a bot — a lookup over the controllers that EXIST.

    The stems present on disk are the whole candidate set; a bot whose name contains one uses it.
    Everything else takes base, which is what the vast majority of bots animate with.
    """
    ctrl_dir = os.path.join(game_root, CTRL_DIR.replace("/", os.sep))
    stems = []
    if os.path.isdir(ctrl_dir):
        for f in os.listdir(ctrl_dir):
            if f.endswith("botanimcontroller.bundle"):
                stems.append(f[: -len("botanimcontroller.bundle")])
    stems = sorted(stems, key=len, reverse=True)  # longest first: 'tagilla' before ''
    name = (bot_type or "").lower()
    pick = next((s for s in stems if s and s != "base" and s in name), "base")
    ctrl = f"controllers/animationcontrollers/{pick}botanimcontroller.bundle"
    rm = f"rootmotiontable/{pick}botrootmotiontable.bundle"
    # Fall back to base when a stem has a controller but no matching root-motion table.
    if not os.path.exists(os.path.join(game_root, CHAR_ROOT.replace("/", os.sep),
                                       rm.replace("/", os.sep))):
        rm = "rootmotiontable/basebotrootmotiontable.bundle"
    return ctrl, rm


def resolve(bot_type, seed=0, bots=None, cust=None, game_root=None, clip_sets=None):
    """A build spec for one rolled bot: the same shape characters.json entries have, so the
    existing builder consumes it unchanged.

    Returns {displayName, parts[], controller, rootMotion, appearance{slot: {...}}, source}.
    """
    if bots is None:
        bots, cust = load_sources()
    game_root = game_root or os.path.join(
        os.environ.get("EFT_GAME_DATA",
                       r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data"),
        "StreamingAssets", "Windows")
    b = bots.get(bot_type)
    if b is None:
        raise AppearanceError(f"unknown bot type {bot_type!r} — see packs/shared/bot_loadouts.json")
    ap = b.get("appearance") or {}
    rng = random.Random(f"appearance:{bot_type}:{seed}")

    parts, chosen = [], {}
    for slot in SLOTS:
        iid = weighted_pick(rng, ap.get(slot) or {})
        if not iid:
            continue
        entry = (cust or {}).get(iid) or {}
        props = entry.get("_props") or {}
        rel = (props.get("Prefab") or {}).get("path")
        if not rel:
            continue
        # Paths in customization.json are StreamingAssets-relative; the builder's part loader
        # resolves against the characters root, so strip that prefix when present.
        part = rel[len(CHAR_ROOT) + 1:] if rel.startswith(CHAR_ROOT + "/") else rel
        exists = os.path.exists(os.path.join(game_root, rel.replace("/", os.sep)))
        chosen[slot] = {"id": iid, "name": entry.get("_name"), "prefab": rel,
                        "bodyPart": props.get("BodyPart"), "present": exists,
                        "built": slot in BUILD_SLOTS}
        if exists and slot in BUILD_SLOTS:
            parts.append(part)
    if not parts:
        # Distinguish the two failures, because they need opposite responses: an EMPTY table is
        # the source data saying this bot type has no appearance of its own (BSG's `*test`
        # placeholders, and a few followers that never spawn standalone) — nothing to fix, and
        # inventing a body for it would be exactly the guessing this module exists to prevent.
        # A populated table that resolved to nothing means the prefabs moved or the game root
        # is wrong, which IS a defect.
        if not any(ap.get(s) for s in SLOTS):
            raise AppearanceError(
                f"{bot_type}: has no appearance table in the source data (all slots empty) — "
                f"this bot type does not define a body of its own")
        raise AppearanceError(
            f"{bot_type}: appearance table has entries but none resolved to a prefab on disk "
            f"under {game_root} — rolled {', '.join(sorted(chosen)) or 'nothing'}")

    ctrl, rm = controller_for(bot_type, game_root)
    spec = {
        "displayName": f"{bot_type} #{seed}",
        "parts": parts,
        "controller": ctrl,
        "rootMotion": rm,
        "defaultClipSet": "locomotion",
        "lod": 0,
        "appearance": chosen,
        "source": {"bot": bot_type, "seed": seed, "derivedFrom": "appearance + customization"},
    }
    if clip_sets:
        spec["clipSets"] = clip_sets
    return spec


if __name__ == "__main__":
    import sys
    bots, cust = load_sources()
    bt = sys.argv[1] if len(sys.argv) > 1 else "assault"
    for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 2):
        s = resolve(bt, seed, bots, cust)
        print(f"[{bt} #{seed}] controller={os.path.basename(s['controller'])}")
        for slot, v in s["appearance"].items():
            mark = "" if v["present"] else "   (bundle MISSING)"
            if not v["built"]:
                mark += "   (first-person only — not built)"
            print(f"    {slot:6s} {v['name']}{mark}")
