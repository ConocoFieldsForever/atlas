"""Unity NavMesh project settings -> `nav_agents.json` (the game's own pathfinding recipe).

WHY
---
Our nav bake used hand-tuned constants (climb 0.38 m, slope 48 deg, headroom 1.8 m) that were
reverse-guessed from bot descriptors. They do not need guessing: Unity stores the authoritative
build settings in `NavMeshProjectSettings` inside `globalgamemanagers`, and both that and
`TagManager` are ENGINE types with hardcoded type trees -- fully readable even though EFT's
il2cpp `global-metadata.dat` is encrypted.

WHAT EFT ACTUALLY SHIPS (verified)
----------------------------------
    agent               radius  height  slope  climb  ledgeDrop  jumpAcross  minRegion  cellSize
    Humanoid            0.300   1.70    48.0   0.38   0.0        0.0         2.0        0.1667
    Big                 0.660   1.91    28.9   0.25   0.0        0.0         2.0        0.1667
    Small               0.310   1.78    48.0   0.33   0.0        0.0         2.0        0.1667
    HumanoidDormitory   0.300   1.78    43.0   0.25   0.0        0.0         2.0        0.1667
    HumanoidWoodCutter  0.329   1.78    48.0   0.25   0.0        0.0         2.0        0.1667

The two zeros matter more than anything else here. In Unity, `ledgeDropHeight` and
`maxJumpAcrossDistance` are the ONLY things that generate drop-down and jump-across off-mesh links.
At 0.0 on every agent, the game's navmesh contains no drops and no jumps at all: a bot can only
move where the surface is continuous within `agentClimb`. Our router allowed a free 2.0 m fall in
any direction, which is what let routes leave the ground and traverse things like a fuel tanker.

Area table (costs feed A*): 0 Walkable 1.0, 1 Not Walkable, 2 Jump 2.0, 3 Sitdown 1.0,
4 Danger 2.0, 5 Terrain 1.0. (`Sitdown` and `Danger` are BSG additions.) The cross-check that
validated the `NavMeshModifier` byte layout: interchange's `terrain` object sets m_Area=5, and
area 5 is named "Terrain".

Written to the SHARED pack tier, not a map pack -- these are global engine settings.

    python extraction/unity/eft_extract_nav.py                 # -> packs/shared/nav_agents.json
    python extraction/unity/eft_extract_nav.py --out <dir>
"""
import os, sys, json, argparse

EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")

# Fields copied verbatim from Unity's NavMeshBuildSettings.
AGENT_FIELDS = ("agentTypeID", "agentRadius", "agentHeight", "agentSlope", "agentClimb",
                "ledgeDropHeight", "maxJumpAcrossDistance", "minRegionArea", "cellSize",
                "tileSize", "manualCellSize", "manualTileSize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("packs", "shared"))
    args = ap.parse_args()
    import UnityPy

    env = UnityPy.load(os.path.join(EFTDATA, "globalgamemanagers"))
    agents, areas, layers = [], [], {}

    for o in env.objects:
        if o.type.name == "NavMeshProjectSettings":
            d = o.read_typetree()
            names = d.get("m_SettingNames") or []
            for i, s in enumerate(d.get("m_Settings") or []):
                rec = {"name": names[i] if i < len(names) else f"agent{i}"}
                for k in AGENT_FIELDS:
                    if k in s:
                        v = s[k]
                        rec[k] = float(v) if isinstance(v, float) else v
                agents.append(rec)
            for i, a in enumerate(d.get("areas") or []):
                if a.get("name"):
                    areas.append({"index": i, "name": a["name"], "cost": float(a.get("cost", 1.0))})
        elif o.type.name == "TagManager":
            for i, n in enumerate(o.read_typetree().get("layers") or []):
                if n:
                    layers[str(i)] = n

    if not agents:
        print("NavMeshProjectSettings not found -- nothing written", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "nav_agents.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump({"agents": agents, "areas": areas, "layers": layers}, fh, indent=1)

    print(f"wrote {fp}")
    print(f"  {len(agents)} agent type(s):")
    for a in agents:
        print(f"    {a['name']:<20} r={a.get('agentRadius'):.3f} h={a.get('agentHeight'):.2f} "
              f"slope={a.get('agentSlope'):.1f} climb={a.get('agentClimb'):.2f} "
              f"ledgeDrop={a.get('ledgeDropHeight'):.2f} jump={a.get('maxJumpAcrossDistance'):.2f} "
              f"minRegion={a.get('minRegionArea'):.1f} cell={a.get('cellSize'):.4f}")
    print(f"  {len(areas)} named area(s): " + ", ".join(f"{a['index']}={a['name']}({a['cost']})" for a in areas))
    print(f"  {len(layers)} named layer(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
