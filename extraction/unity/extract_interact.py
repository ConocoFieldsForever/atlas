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
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eft_extract_switches as S


def find_interactables(level, objs, tfm, gos, monos, sc):
    """All interactable Switch records in an already-loaded level. Power levers keep their full
    lamp/light/target resolution (via find_power_switches); every OTHER Switch is added with its
    class-validated target edges but no light bank. No name matching anywhere."""
    # Power levers: full records (controlled lamp bank + resolved Unity Lights + gated targets).
    power = S.find_power_switches(level, objs, tfm, gos, monos, sc)
    for p in power:
        p["kind"] = "power"
    power_gos = {p["switch_go"] for p in power}
    out = list(power)

    # Every OTHER EFT.Interactive.Switch (alarm / floor / call button / water plane / ...): keep it,
    # with the same class-validated PPtr target edges (exfils/doors/transits it gates) + label.
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
        try:
            label = gos[sgo].read_typetree().get("m_Name", "switch")
        except Exception:
            label = "switch"
        targets = []
        for t in S.decode_scalar_targets(raw, monos, sc, gos):
            tgo = t["target_go"]
            try:
                tname = gos[tgo].read_typetree().get("m_Name", "")
            except Exception:
                tname = ""
            targets.append({
                "type": t["type"],
                "target_go": tgo,
                "name": tname,
                "world_pos": S.world_pos(tgo, gos, tfm),
            })
        out.append({
            "id": f"unity:{level}:mb:{pid}",
            "level": level,
            "switch_go": sgo,
            "group": f"{level}:{sgo}",
            "world_pos": S.world_pos(sgo, gos, tfm),
            "label": label,
            "kind": "switch",
            "count": 0,
            "controlled_lamp_gos": [],
            "controlled_light_gos": [],
            "targets": targets,
        })
    return out


def extract_level(level):
    env = S.load_level(level)
    objs, tfm, gos, monos, sc = S.build_maps(env)
    return find_interactables(level, objs, tfm, gos, monos, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True, help="comma-separated level indices to scan (e.g. 520)")
    ap.add_argument("--name", required=True, help="output dataset folder name")
    args = ap.parse_args()
    out = os.path.join(S.OUTROOT, args.name)
    os.makedirs(out, exist_ok=True)
    total = 0
    for lv in [int(x) for x in args.levels.split(",") if x.strip()]:
        try:
            items = extract_level(lv)
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
