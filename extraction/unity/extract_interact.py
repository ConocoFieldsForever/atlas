"""Extract ALL interactable switches on a map — the power lever AND every other
`EFT.Interactive.Switch` (alarm / floor-button / call-button / water-plane / ...), not just the
power lever. Purely typed Unity components (zero name matching), so it works identically on every map.

This is a superset of eft_extract_switches.py: it reuses that module's typed-component machinery but
DROPS the power-only filter, so the viewer can surface every interactable — not only the one that
owns a light bank. Each record is CLASSIFIED without name rules:
  kind = "power"  when the switch's trailing PPtr array resolves ENTIRELY to LampController (it owns
                  that light bank — full lamp/light resolution + gated targets, same as before), else
  kind = "switch" for every other interactable Switch (alarm, buttons, water-plane, ...), kept with
                  its class-validated target edges (exfils/doors/transits it gates) + label + world pos.

  python extraction/unity/extract_interact.py --levels 520 --name interchange_v2
  python extraction/unity/extract_interact.py --levels 518 --name reserve

Writes <dataset>/interact_<level>.json: a flat array of
  {id, level, switch_go, group, world_pos:[x,y,z], label, kind, count, controlled_lamp_gos:[...],
   controlled_light_gos:[...], targets:[{type,target_go,name,world_pos}]}
The power records are byte-identical to eft_extract_switches's (same join key `group` = "<lv>:<GO>",
so eft_extract_lights still tags the controlled lights); the non-power records add the rest.
"""
import os, re, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eft_extract_switches as S

# Organizational node names that carry no meaning for a human label. This is a DISPLAY rule only
# (uniform across every map, applied after prefix stripping) — never a cull or control rule.
GENERIC_SEGMENTS = {"logic", "oo", "interactive", "switch", "node"}
STRIP_PREFIXES = ("INTERACTIVE_", "SBG_", "Node_")


def informative(seg):
    """The human-meaningful remainder of a hierarchy segment, or None (pure organizer node)."""
    s = seg or ""
    for pre in STRIP_PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
    return s if s and s.lower() not in GENERIC_SEGMENTS else None


def read_cstr(buf, off):
    """length-prefixed printable utf8 string + 4-aligned end offset; (None, off) when implausible."""
    if off + 4 > len(buf):
        return None, off
    ln = int.from_bytes(buf[off:off + 4], "little")
    if ln <= 0 or ln > 256 or off + 4 + ln > len(buf):
        return None, off
    try:
        s = buf[off + 4:off + 4 + ln].decode("utf8")
    except UnicodeDecodeError:
        return None, off
    if not all(31 < ord(c) < 127 for c in s):
        return None, off
    return s, (off + 4 + ln + 3) & ~3


def payload_strings(raw):
    """Every plausible length-prefixed string in a MonoBehaviour payload, in serialized order.
    For a Switch that is (validated on Icebreaker level704): [0] the interaction TRIGGER name
    ("Open_01_<hash>" / "Try_Repair_01_<hash>"), then its serialized Id, optionally a 24-hex
    required-ITEM template id (the frozen hatch wants the cutting torch), and LAST the in-game
    interaction VERB ("Use"/"Open"/"Place")."""
    out = []
    off = 32
    n = len(raw or b"")
    while off + 4 <= n:
        s, e = read_cstr(raw, off)
        if s and len(s) >= 3:
            out.append(s)
            off = e
        else:
            off += 4
    return out


def dissect_strings(strs):
    """(trigger, link, item_id, verb) from a Switch payload's string sequence, all best-effort."""
    trigger = strs[0] if strs else None
    link = None
    if trigger and "_" in trigger:
        tail = trigger.rsplit("_", 1)[1]
        if tail.isdigit() and len(tail) >= 6:
            link = tail
    item_id = next((s for s in strs
                    if len(s) == 24 and all(c in "0123456789abcdef" for c in s)), None)
    verb = strs[-1] if len(strs) >= 2 and len(strs[-1]) <= 12 and "_" not in strs[-1] else None
    return trigger, link, item_id, verb


def build_labels(records, name_path_of, map_token):
    """Human label per record: '<context> - <action>', both straight from the hierarchy/payload.
    action = the switch GO's own informative name (trailing 'Logic' dropped), else the trigger
    stem, else the verb. context = the nearest informative ancestor; when two records collide on
    the same label, both extend context upward one ancestor at a time until unique (that is what
    separates the three identical CPU-panel repairs by their Room_01/02/03 organizer). The map-name
    token ('Icebreaker_...') is stripped for display. Purely cosmetic — `go_name` keeps the raw."""
    def disp(seg):
        s = informative(seg) or seg
        toks = s.split("_")
        if toks and toks[0].lower() == map_token:
            toks = toks[1:]
        return " ".join(t for t in toks if t)

    metas = []
    for r in records:
        path = name_path_of(r["switch_go"])            # root -> ... -> own GO
        own = informative(path[-1]) if path else None
        ancestors = [p for p in (path[:-1] if path else []) if informative(p)]
        if own:
            action = re.sub(r"_?[Ll]ogic$", "", own).strip("_") or own
        else:
            stem = re.sub(r"(_\d+)+$", "", r.get("trigger") or "").strip("_")
            action = stem or r.get("verb") or "switch"
        metas.append({"rec": r, "action": disp(action), "anc": ancestors, "depth": 1})

    def label_of(m):
        ctx = [disp(a) for a in m["anc"][-m["depth"]:]]
        lbl = " · ".join(ctx + [m["action"]]) if ctx else m["action"]
        return lbl[:1].upper() + lbl[1:]

    for _ in range(4):                                 # extend colliding contexts up to 3 ancestors
        seen = {}
        for m in metas:
            seen.setdefault(label_of(m), []).append(m)
        dupes = [ms for ms in seen.values() if len(ms) > 1]
        if not dupes:
            break
        for ms in dupes:
            for m in ms:
                if m["depth"] < len(m["anc"]):
                    m["depth"] += 1
    for m in metas:
        m["rec"]["label"] = label_of(m)


def find_interactables(level, objs, tfm, gos, monos, sc, map_token=""):
    """All interactable Switch records in an already-loaded level. Power levers keep their full
    lamp/light/target resolution (via find_power_switches); every OTHER Switch is added with its
    class-validated target edges but no light bank, PLUS the structural context a human label
    needs: the GO hierarchy path, the payload's trigger name (+ its digit-hash `link`, the join
    key to the doors it drives), any 24-hex required-item id, and the interaction verb. `label`
    is built from that context (build_labels); `go_name` keeps the raw GameObject name."""
    # Power levers: full records (controlled lamp bank + resolved Unity Lights + gated targets).
    power = S.find_power_switches(level, objs, tfm, gos, monos, sc)
    for p in power:
        p["kind"] = "power"
    power_gos = {p["switch_go"] for p in power}
    out = list(power)

    # GO -> ancestor name path (root -> ... -> GO), via the Transform parent chain.
    tf_info = {}
    for tpid, t in tfm.items():
        try:
            td = t.read_typetree()
            tf_info[tpid] = ((td.get("m_Father") or {}).get("m_PathID"),
                             (td.get("m_GameObject") or {}).get("m_PathID"))
        except Exception:
            pass
    _names = {}

    def name_of(gpid):
        if gpid not in _names:
            try:
                _names[gpid] = gos[gpid].read_typetree().get("m_Name", "?")
            except Exception:
                _names[gpid] = "?"
        return _names[gpid]

    def go_tf(go_pid):
        try:
            for comp in gos[go_pid].read_typetree().get("m_Component", []):
                cp = comp.get("component") or comp.get("second") or {}
                if cp.get("m_PathID") in tfm:
                    return cp.get("m_PathID")
        except Exception:
            pass
        return None

    def name_path_of(go_pid):
        names = []
        tp = go_tf(go_pid)
        depth = 0
        while tp is not None and depth < 64:
            depth += 1
            fa, g = tf_info.get(tp, (None, None))
            if g is not None:
                names.append(name_of(g))
            tp = fa
        return list(reversed(names))

    # Every OTHER EFT.Interactive.Switch (alarm / floor / call button / water plane / ...): keep it,
    # with the same class-validated PPtr target edges (exfils/doors/transits it gates) + context.
    plain = []
    for pid, mb in monos.items():
        if S.mono_class(mb, sc) != S.SWITCH_CLASS:
            continue
        sgo = S.mono_go(mb)
        if sgo in power_gos:
            continue                                   # already captured as a power lever above
        try:
            raw = mb.get_raw_data()
        except Exception:
            raw = None
        targets = []
        for t in S.decode_scalar_targets(raw, monos, sc, gos):
            tgo = t["target_go"]
            targets.append({
                "type": t["type"],
                "target_go": tgo,
                "name": name_of(tgo),
                "world_pos": S.world_pos(tgo, gos, tfm),
            })
        trigger, link, item_id, verb = dissect_strings(payload_strings(raw))
        rec = {
            "id": f"unity:{level}:mb:{pid}",
            "level": level,
            "switch_go": sgo,
            "group": f"{level}:{sgo}",
            "world_pos": S.world_pos(sgo, gos, tfm),
            "label": name_of(sgo),                     # placeholder; build_labels overwrites
            "go_name": name_of(sgo),
            "path": name_path_of(sgo),
            "kind": "switch",
            "count": 0,
            "controlled_lamp_gos": [],
            "controlled_light_gos": [],
            "targets": targets,
        }
        if trigger:
            rec["trigger"] = trigger
        if link:
            rec["link"] = link
        if item_id:
            rec["item_id"] = item_id
        if verb:
            rec["verb"] = verb
        plain.append(rec)
    build_labels(plain, name_path_of, map_token)
    out.extend(plain)
    return out


def extract_level(level, map_token=""):
    env = S.load_level(level)
    objs, tfm, gos, monos, sc = S.build_maps(env)
    return find_interactables(level, objs, tfm, gos, monos, sc, map_token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True, help="comma-separated level indices to scan (e.g. 520)")
    ap.add_argument("--name", required=True, help="output dataset folder name")
    args = ap.parse_args()
    out = os.path.join(S.OUTROOT, args.name)
    os.makedirs(out, exist_ok=True)
    # first token of the dataset name = the map-name prefix on its GameObjects ("Icebreaker_...");
    # stripped from DISPLAY labels only.
    map_token = (args.name.split("_")[0] or "").lower()
    total = 0
    for lv in [int(x) for x in args.levels.split(",") if x.strip()]:
        try:
            items = extract_level(lv, map_token)
        except Exception as e:
            print(f"level{lv}: scan failed ({e})", flush=True)
            continue
        fp = os.path.join(out, f"interact_{lv}.json")
        if items:
            json.dump(items, open(fp, "w"))
            total += len(items)
            npow = sum(1 for x in items if x["kind"] == "power")
            print(f"  level{lv}: {len(items)} interactable(s) [{npow} power, {len(items) - npow} other] -> {fp}",
                  flush=True)
            for x in items:
                tgt = (" -> " + ",".join(t["type"].split(".")[-1] for t in x["targets"])) if x["targets"] else ""
                extra = f"  {x['count']} lamps/{len(x['controlled_light_gos'])} lights" if x["kind"] == "power" else ""
                print(f"    [{x['kind']}] '{x['label']}' GO {x['switch_go']} @ {x['world_pos']}{extra}{tgt}",
                      flush=True)
        elif os.path.isfile(fp):
            os.remove(fp)                              # no interactables on this level -> drop stale sidecar
    print(f"interact scan: {total} interactable switch(es) across "
          f"{len([x for x in args.levels.split(',') if x.strip()])} level(s)", flush=True)


if __name__ == "__main__":
    main()
