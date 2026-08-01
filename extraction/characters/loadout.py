#!/usr/bin/env python3
"""Roll a bot's kit from the GAME'S OWN generation tables. Nothing here is authored.

Sources (banked under packs/shared/, all BSG data):
  * bot_loadouts.json — per bot type (`assault` = scav, `pmcbear`, `pmcusec`):
      inventory.equipment.<Slot> = {itemId: WEIGHT}   the game's own spawn frequencies
      inventory.mods[parentTpl][slot] = [allowed child ids]   what THAT bot may bolt on
  * globals.json ItemPresets — BSG's factory weapon builds (the fallback and the base tree)
  * item_templates.json — `_props.Slots[]` with per-slot Filters (the legality check)

Determinism: every roll is seeded from (bot type, agent index), so a given NPC keeps the same kit
across frames, across a reload, and across machines.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appearance

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SHARED = os.path.join(REPO, "packs", "shared")

def _config():
    """Policy (which slots are wearable, which appearance slots resolve) lives in the same
    config the fetcher uses — no slot list is hardcoded in the roller."""
    import fetch_bot_db
    return fetch_bot_db.load_config()


def load_tables():
    items = json.load(open(os.path.join(SHARED, "item_templates.json"), encoding="utf-8"))
    bots = json.load(open(os.path.join(SHARED, "bot_loadouts.json"), encoding="utf-8"))
    cust = {}
    cp = os.path.join(SHARED, "customization.json")
    if os.path.exists(cp):
        cust = json.load(open(cp, encoding="utf-8"))
    presets = {}
    gp = os.path.join(SHARED, "globals.json")
    if os.path.exists(gp):
        g = json.load(open(gp, encoding="utf-8"))
        for p in (g.get("ItemPresets") or {}).values():
            its = p.get("_items") or []
            if not its:
                continue
            root = its[0].get("_tpl")
            if p.get("_encyclopedia") or root not in presets:
                presets[root] = its
    return items, bots, presets, cust


def weighted_pick(rng, table):
    """One id from {id: weight} using the game's own weights."""
    if not table:
        return None
    ids, ws = zip(*[(k, max(float(v), 0.0)) for k, v in table.items()])
    total = sum(ws)
    if total <= 0:
        return rng.choice(ids)
    r = rng.random() * total
    acc = 0.0
    for i, w in zip(ids, ws):
        acc += w
        if r <= acc:
            return i
    return ids[-1]


def slot_filter(items, tpl, slot_name):
    """The template's own allow-list for a slot (the legality check)."""
    t = items.get(tpl) or {}
    for s in (t.get("_props") or {}).get("Slots") or []:
        if s.get("_name") == slot_name:
            fs = (s.get("_props") or {}).get("filters") or [{}]
            return set(fs[0].get("Filter") or [])
    return set()


def required_slots(items, tpl):
    t = items.get(tpl) or {}
    out = []
    for s in (t.get("_props") or {}).get("Slots") or []:
        out.append((s.get("_name"), bool(s.get("_required"))))
    return out


def build_weapon_tree(rng, items, bot_mods, presets, weapon_tpl, depth=0, seen=None):
    """{parentTpl: {slot: childTpl}} for one weapon.

    The FACTORY preset is the base (BSG's own default build). Where the bot's own mod table
    offers candidates for a slot, roll among the ones the parent template actually allows —
    that is how the game varies a bot's gun. Every emitted edge is validated against the
    parent's Filter, cycles and runaway depth are rejected.
    """
    seen = seen if seen is not None else set()
    install = {}
    if depth > 8 or weapon_tpl in seen:
        return install
    seen = seen | {weapon_tpl}

    # Factory baseline: parent tpl -> {slot: child tpl}
    factory = {}
    for it in presets.get(weapon_tpl) or []:
        pass
    pre = presets.get(weapon_tpl) or []
    by_id = {it["_id"]: it for it in pre}
    for it in pre:
        par, slot = it.get("parentId"), it.get("slotId")
        if par and slot and par in by_id:
            factory.setdefault(by_id[par]["_tpl"], {})[slot] = it["_tpl"]

    def rec(tpl, d):
        if d > 8 or tpl in install:
            return
        chosen = {}
        base = factory.get(tpl, {})
        for slot_name, required in required_slots(items, tpl):
            allowed = slot_filter(items, tpl, slot_name)
            cands = [c for c in (bot_mods.get(tpl, {}).get(slot_name) or []) if c in allowed]
            pick = None
            if cands:
                pick = cands[rng.randrange(len(cands))]
            elif base.get(slot_name) in allowed:
                pick = base.get(slot_name)          # the factory part
            # NO ARBITRARY PARTS. Filling a required slot with `sorted(allowed)[0]` picks by
            # alphabet, which is how a fixed A2 rear sight ended up installed under an HHS-1 --
            # a build the game would never produce. If neither this bot's table nor BSG's factory
            # preset names a part, the slot stays empty: an incomplete gun is honest, an invented
            # one is not.
            elif required and allowed:
                pick = None
            if pick:
                chosen[slot_name] = pick
        if chosen:
            install[tpl] = chosen
            for child in chosen.values():
                rec(child, d + 1)

    rec(weapon_tpl, depth)
    return install


def roll(bot_type, seed, items=None, bots=None, presets=None, cust=None, cfg=None):
    """A full kit for one agent: appearance, weapon (with its mod tree), and worn equipment —
    every choice drawn from THIS bot type's own tables."""
    if items is None:
        items, bots, presets, cust = load_tables()
    cfg = cfg or _config()
    b = bots.get(bot_type) or {}
    inv = b.get("inventory") or {}
    eq = inv.get("equipment") or {}
    mods = inv.get("mods") or {}
    # chances.equipment: the PERCENT chance a slot is filled at all. Without it every bot wore
    # every slot, which is not how the game spawns them (a scav often has no helmet).
    fill = (b.get("chances") or {}).get("equipment") or {}
    rng = random.Random(f"{bot_type}:{seed}")

    weapon = weighted_pick(rng, eq.get("FirstPrimaryWeapon") or {})
    kit = {
        "bot": bot_type,
        "seed": seed,
        "weapon": weapon,
        "weaponName": (items.get(weapon) or {}).get("_name") if weapon else None,
        "weaponTree": build_weapon_tree(rng, items, mods, presets, weapon) if weapon else {},
        "worn": {},
    }
    for slot in cfg.get("wearableSlots", []):
        # Roll the slot's own fill chance first (absent = always attempt, matching the tables
        # that omit a slot they always fill).
        chance = fill.get(slot)
        if chance is not None and rng.random() * 100.0 > float(chance):
            continue
        pick = weighted_pick(rng, eq.get(slot) or {})
        if pick:
            kit["worn"][slot] = {
                "id": pick,
                "name": (items.get(pick) or {}).get("_name"),
                "prefab": (((items.get(pick) or {}).get("_props") or {}).get("Prefab") or {}).get("path"),
            }
    # APPEARANCE is resolved by appearance.py, which OWNS that question -- the same call the
    # character builder makes, so a kit and the body it is worn on can never disagree about
    # which meshes the bot has. Seeded identically, so kit N and character N are the same bot.
    kit["appearance"] = appearance.resolve(bot_type, seed, bots, cust)["appearance"]
    return kit


if __name__ == "__main__":
    import sys
    items, bots, presets, cust = load_tables()
    bt = sys.argv[1] if len(sys.argv) > 1 else "assault"
    for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
        k = roll(bt, seed, items, bots, presets, cust)
        mods = sum(len(v) for v in k["weaponTree"].values())
        print(f"[{bt} #{seed}] {k['weaponName']} + {mods} mods")
        for slot, a in k.get("appearance", {}).items():
            print(f"    ~{slot:11s} {a['name']}")
        for slot, w in k["worn"].items():
            print(f"    {slot:12s} {w['name']}")
