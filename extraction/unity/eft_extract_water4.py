#!/usr/bin/env python
"""Extract EVERY Water4-family material's full parameter set -> packs/shared/water4.json.

Phase 0 of docs/GRAPHICS_PLAN.md: Gerstner displacement (Phase 5) and any faithful water shading
need the game's authored values — _GAmplitude, _GDirectionAB/CD, WaveSpeed, _BumpTiling,
_BaseColor/_DepthColor, _Extinction, foam/distortion parameters — extracted, not invented. The
viewer currently hardcodes ONE _DepthColor lifted by hand from Sandbox_Water4Advanced; this makes
the whole family available, keyed by material name, so consumers can bind per-map values.

Scans resources.assets + every sharedassets*.assets for materials whose shader resolves to the
Water4 family (FX/Water4, FX/SimpleWater4, Hidden/Water/*). Colors, floats and the texture NAMES
(not the textures — those ship via the normal pack pipeline) are captured verbatim.

Per the plan: values that do not exist are simply ABSENT — nothing here fabricates a default.

  python extraction/unity/eft_extract_water4.py          -> packs/shared/water4.json
Re-run per game patch.
"""
import json
import os
import sys
import glob
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(os.environ.get("EFT_INTEL_OUT_DIR") or os.path.join(REPO, "packs", "shared"),
                   "water4.json")
GAME = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")


def kv(pairs):
    out = []
    for p in pairs or []:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            out.append((p[0], p[1]))
        elif isinstance(p, dict):
            out.append((p.get("first"), p.get("second")))
    return out


def color4(v):
    if isinstance(v, dict):
        return [v.get("r", 0.0), v.get("g", 0.0), v.get("b", 0.0), v.get("a", 0.0)]
    return v


def main():
    import UnityPy
    files = [os.path.join(GAME, "resources.assets")] + sorted(
        glob.glob(os.path.join(GAME, "sharedassets*.assets")))
    mats = {}
    n_files = 0
    for path in files:
        if not os.path.isfile(path):
            continue
        try:
            env = UnityPy.load(path)
        except Exception:
            continue
        n_files += 1
        base = os.path.basename(path)
        for o in env.objects:
            if o.type.name != "Material":
                continue
            try:
                d = o.read_typetree()
            except Exception:
                continue
            # Shader membership: resolve the shader name when the PPtr resolves; otherwise fall
            # back to the property signature (WaveSpeed + _GAmplitude only exist on Water4).
            props = d.get("m_SavedProperties") or {}
            colors = dict(kv(props.get("m_Colors")))
            if "WaveSpeed" not in colors and "_GAmplitude" not in colors:
                continue
            name = d.get("m_Name") or f"mat_{o.path_id}"
            floats = dict(kv(props.get("m_Floats")))
            texes = {}
            for k, v in kv(props.get("m_TexEnvs")):
                tex = (v or {}).get("m_Texture") or {}
                # Only the reference identity; actual textures ship via the pack pipeline.
                texes[k] = {"fileID": tex.get("m_FileID"), "pathID": tex.get("m_PathID"),
                            "scale": color4((v or {}).get("m_Scale")),
                            "offset": color4((v or {}).get("m_Offset"))}
            rec = {
                "asset": base,
                "colors": {k: color4(v) for k, v in colors.items()},
                "floats": floats,
                "textures": texes,
            }
            # Several maps re-ship a material under the same name; keep the RICHEST record.
            prev = mats.get(name)
            if prev is None or len(rec["colors"]) + len(rec["floats"]) > (
                    len(prev["colors"]) + len(prev["floats"])):
                mats[name] = rec
        del env
    if not mats:
        print("[water4] no Water4-family materials found - not writing", flush=True)
        return 1
    doc = {
        "schema": 1,
        "source": "EscapeFromTarkov_Data resources.assets + sharedassets* (Material typetrees)",
        "note": "Authored Water4 parameters, verbatim. Gerstner: _GAmplitude/_GSteepness/"
                "_GSpeed/_GDirectionAB/_GDirectionCD. Absent keys were absent in the game.",
        "built": int(time.time()),
        "materials": mats,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"[water4] {len(mats)} material(s) from {n_files} asset file(s) -> {OUT}", flush=True)
    for n in sorted(mats):
        g = mats[n]["colors"].get("_GAmplitude")
        print(f"[water4]   {n:34s} gerstner={'yes' if g else 'no '} asset={mats[n]['asset']}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
