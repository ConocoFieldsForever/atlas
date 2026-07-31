#!/usr/bin/env python3
"""Standalone LOOPING-particle extractor -> <pack>/particles.json + <pack>/tex_fx/*.png.

The game's persistent effects (the burning TerraGroup building, smoke columns, steam jets,
sparks) are plain Unity ParticleSystems whose data fully describes the look: the flipbook
atlas texture and its grid (UVModule), start color/size/lifetime/speed (InitialModule),
emission rate, and the renderer's material. This script reads exactly that — nothing is
authored — and writes a pack-local sidecar the viewer renders as camera-facing flipbook
billboards. Like the grass/sky sidecars it needs NO reassembly: it heals every built pack.

Kept: looping AND playOnAwake AND activeInHierarchy systems (a fire burns from raid start).
One-shot effects (muzzle flashes, hit sparks) are gameplay-driven and meaningless in a map
viewer — skipped.

usage: eft_extract_particles.py --pack packs/ground_zero.eftpack --levels 466,467,...
"""
import argparse
import json
import os
import sys

import UnityPy

EFTDATA = os.environ.get("EFT_GAME_DATA",
                         r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")


def g(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def curve_scalar(mm, default=0.0):
    """MinMaxCurve -> a representative constant (mode 0 scalar; else the max scalar)."""
    if not isinstance(mm, dict):
        return default
    v = mm.get("scalar")
    if v is None:
        v = mm.get("maxScalar", default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def gradient_rgba(mg, default=(1.0, 1.0, 1.0, 1.0)):
    """MinMaxGradient -> a representative rgba (maxColor; the gradient's first key as fallback)."""
    if not isinstance(mg, dict):
        return list(default)
    c = mg.get("maxColor") or mg.get("minColor")
    if isinstance(c, dict):
        return [round(float(c.get(k, 1.0)), 4) for k in ("r", "g", "b", "a")]
    return list(default)


def gradient_keys(mg):
    """MinMaxGradient -> up to 6 [t, r, g, b, a] keys of maxGradient (the OVER-LIFETIME look:
    fire's white-yellow core aging into orange then fading smoke). None when there is no real
    gradient. Unity stores color keys (rgb + time) and alpha keys (a + time) separately —
    merge on the union of their times."""
    if not isinstance(mg, dict):
        return None
    grad = mg.get("maxGradient") or mg.get("minGradient")
    if not isinstance(grad, dict):
        return None
    ckeys, akeys = [], []
    n_c = int(grad.get("m_NumColorKeys", 0) or 0)
    n_a = int(grad.get("m_NumAlphaKeys", 0) or 0)
    for i in range(min(n_c, 8)):
        k = grad.get(f"key{i}")
        t = grad.get(f"ctime{i}", 0)
        if isinstance(k, dict):
            ckeys.append((float(t) / 65535.0, [float(k.get(c, 1.0)) for c in ("r", "g", "b")]))
    for i in range(min(n_a, 8)):
        k = grad.get(f"key{i}")
        t = grad.get(f"atime{i}", 0)
        if isinstance(k, dict):
            akeys.append((float(t) / 65535.0, float(k.get("a", 1.0))))
    if not ckeys and not akeys:
        return None

    def sample_c(t):
        if not ckeys:
            return [1.0, 1.0, 1.0]
        ckeys.sort()
        for (t0, c0), (t1, c1) in zip(ckeys, ckeys[1:]):
            if t0 <= t <= t1:
                f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                return [c0[j] + (c1[j] - c0[j]) * f for j in range(3)]
        return ckeys[0][1] if t <= ckeys[0][0] else ckeys[-1][1]

    def sample_a(t):
        if not akeys:
            return 1.0
        akeys.sort()
        for (t0, a0), (t1, a1) in zip(akeys, akeys[1:]):
            if t0 <= t <= t1:
                f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                return a0 + (a1 - a0) * f
        return akeys[0][1] if t <= akeys[0][0] else akeys[-1][1]

    times = sorted({0.0, 1.0, *(t for t, _ in ckeys), *(t for t, _ in akeys)})[:6]
    return [[round(t, 3)] + [round(v, 4) for v in sample_c(t)] + [round(sample_a(t), 4)]
            for t in times]


def curve_keys(mm):
    """MinMaxCurve -> up to 4 [t, v] keys of maxCurve (size-over-lifetime), or None."""
    if not isinstance(mm, dict):
        return None
    cur = (mm.get("maxCurve") or {}).get("m_Curve")
    if not isinstance(cur, list) or len(cur) < 2:
        return None
    out = []
    for k in cur[:4]:
        if isinstance(k, dict):
            out.append([round(float(k.get("time", 0.0)), 3), round(float(k.get("value", 1.0)), 4)])
    return out if len(out) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--levels", required=True)
    args = ap.parse_args()
    pack = args.pack
    fx_dir = os.path.join(pack, "tex_fx")
    os.makedirs(fx_dir, exist_ok=True)

    emitters = []
    tex_cache = {}  # texture path_id -> pack-relative png path (or None on failure)
    n_seen = n_loop = 0

    for lv in [int(x) for x in args.levels.split(",") if x.strip()]:
        p = os.path.join(EFTDATA, f"level{lv}")
        if not os.path.exists(p):
            continue
        env = UnityPy.load(p)

        # transform tables for world position + activeInHierarchy (local m_IsActive chain).
        tf = {}      # tf pid -> (father pid, local TRS dict, go pid)
        go_act = {}  # go pid -> m_IsActive
        go2tf = {}
        for o in env.objects:
            if o.type.name == "Transform":
                d = o.read_typetree()
                tf[o.path_id] = (
                    (d.get("m_Father") or {}).get("m_PathID", 0),
                    d,
                    (d.get("m_GameObject") or {}).get("m_PathID", 0),
                )
                go2tf[(d.get("m_GameObject") or {}).get("m_PathID", 0)] = o.path_id
            elif o.type.name == "GameObject":
                d = o.read_typetree()
                go_act[o.path_id] = bool(d.get("m_IsActive", True))

        def world_pos_scale(tpid):
            """World position + a uniform scale magnitude from the father-chain TRS product.
            Rotation is ignored on purpose: the viewer re-billboards every quad anyway."""
            import numpy as np
            M = np.eye(4)
            hops = 0
            t = tpid
            while t and hops < 64:
                father, d, _ = tf.get(t, (0, None, 0))
                if d is None:
                    break
                lp = d.get("m_LocalPosition") or {}
                lr = d.get("m_LocalRotation") or {}
                ls = d.get("m_LocalScale") or {}
                x, y, z, w = (float(lr.get(k, 0.0)) for k in ("x", "y", "z", "w"))
                # quaternion -> rotation matrix
                R = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ])
                S = np.diag([float(ls.get(k, 1.0)) for k in ("x", "y", "z")])
                L = np.eye(4)
                L[:3, :3] = R @ S
                L[:3, 3] = [float(lp.get(k, 0.0)) for k in ("x", "y", "z")]
                M = L @ M
                t = father
                hops += 1
            pos = M[:3, 3]
            sc = float(np.cbrt(abs(max(np.linalg.det(M[:3, :3]), 1e-9))))
            return pos, sc

        def active_chain(go_pid):
            t = go2tf.get(go_pid, 0)
            hops = 0
            while t and hops < 64:
                father, _, gp = tf.get(t, (0, None, 0))
                if gp and not go_act.get(gp, True):
                    return False
                t = father
                hops += 1
            return True

        # renderer per GameObject (material + render mode)
        rend = {}
        for o in env.objects:
            if o.type.name != "ParticleSystemRenderer":
                continue
            try:
                d = o.read_typetree()
                gp = (d.get("m_GameObject") or {}).get("m_PathID", 0)
                mats = d.get("m_Materials") or []
                rend[gp] = (o, mats[0] if mats else None, int(d.get("m_RenderMode", 0) or 0))
            except Exception:
                continue

        for o in env.objects:
            if o.type.name != "ParticleSystem":
                continue
            try:
                d = o.read_typetree()
            except Exception:
                continue
            n_seen += 1
            gp = (d.get("m_GameObject") or {}).get("m_PathID", 0)
            # looping is the persistence signal; playOnAwake is NOT required — fire prefabs
            # trigger some of their looping children (sparks, embers) from scripts.
            if not bool(d.get("looping")):
                continue
            if not active_chain(gp):
                continue
            r = rend.get(gp)
            if not r or r[1] is None:
                continue
            n_loop += 1
            # material: shader name + main texture + tint
            tex_rel, shader, tint = None, "", [1.0, 1.0, 1.0, 1.0]
            try:
                ro = r[0].read()
                mat = ro.m_Materials[0].read()
                try:
                    sho = mat.m_Shader.read()
                    pf = getattr(sho, "m_ParsedForm", None)
                    shader = getattr(pf, "m_Name", None) or getattr(sho, "m_Name", "") or ""
                except Exception:
                    pass
                te = mat.m_SavedProperties.m_TexEnvs
                slots = dict(te.items() if hasattr(te, "items") else te)
                for key in ("_MainTex", "_BaseMap", "_Tex"):
                    s = slots.get(key)
                    tp = getattr(s, "m_Texture", None) if s is not None else None
                    pid = getattr(tp, "path_id", 0)
                    if pid:
                        if pid not in tex_cache:
                            try:
                                timg = tp.read()
                                name = "".join(
                                    c if c.isalnum() or c in "-_" else "_"
                                    for c in (getattr(timg, "m_Name", None) or f"fx_{pid}"))
                                out = f"fx_{name}_{pid & 0xFFFFFFFF}.png"
                                timg.image.save(os.path.join(fx_dir, out))
                                tex_cache[pid] = f"tex_fx/{out}"
                            except Exception:
                                tex_cache[pid] = None
                        tex_rel = tex_cache[pid]
                        break
                cs = mat.m_SavedProperties.m_Colors
                cd = dict(cs.items() if hasattr(cs, "items") else cs)
                tc = cd.get("_TintColor") or cd.get("_Color")
                if tc is not None:
                    tint = [round(float(getattr(tc, k, 1.0)), 4) for k in ("r", "g", "b", "a")]
            except Exception:
                pass
            if not tex_rel:
                continue  # nothing to draw without an atlas
            init = d.get("InitialModule") or {}
            uv = d.get("UVModule") or {}
            emis = d.get("EmissionModule") or {}
            shape = d.get("ShapeModule") or {}
            colmod = d.get("ColorModule") or {}
            sizemod = d.get("SizeModule") or {}
            lights = d.get("LightsModule") or {}
            pos, wscale = world_pos_scale(go2tf.get(gp, 0))
            rec_extra = {}
            # OVER-LIFETIME look (the part a constant color cannot fake): the game's own
            # color gradient + size curve, sampled to a few keys.
            if g(colmod, "enabled"):
                gk = gradient_keys(g(colmod, "gradient"))
                if gk:
                    rec_extra["colorOverLife"] = gk
            if g(sizemod, "enabled"):
                ck = curve_keys(g(sizemod, "curve"))
                if ck:
                    rec_extra["sizeOverLife"] = ck
            # The game's own "this effect casts light" signal + the referenced Light's values.
            if g(lights, "enabled"):
                lrec = {"ratio": round(float(g(lights, "ratio", 0.0) or 0.0), 3),
                        "intensity": round(curve_scalar(g(lights, "intensityCurve"), 1.0), 3),
                        "range": round(curve_scalar(g(lights, "rangeCurve"), 3.0), 3)}
                try:
                    lp = g(lights, "light") or {}
                    if lp.get("m_PathID"):
                        for lo in env.objects:
                            if lo.path_id == lp["m_PathID"] and lo.type.name == "Light":
                                ld = lo.read_typetree()
                                lc = ld.get("m_Color") or {}
                                lrec["color"] = [round(float(lc.get(k, 1.0)), 4)
                                                 for k in ("r", "g", "b")]
                                lrec["intensity"] = round(
                                    lrec["intensity"] * float(ld.get("m_Intensity", 1.0) or 1.0), 3)
                                lrec["range"] = round(
                                    max(lrec["range"], float(ld.get("m_Range", 0.0) or 0.0)), 3)
                                break
                except Exception:
                    pass
                rec_extra["light"] = lrec
            if int(g(uv, "timeMode", 0) or 0) == 1:
                rec_extra["uvFpsMode"] = 1
            emitters.append({
                **rec_extra,
                # the viewer/pack X-flip (diag(-1,1,1)) — same convention as every sidecar.
                "pos": [round(-float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
                "lv": lv,
                "tex": tex_rel,
                "shader": shader,
                "tint": tint,
                "renderMode": r[2],
                "lifetime": round(curve_scalar(init.get("startLifetime"), 3.0), 3),
                "speed": round(curve_scalar(init.get("startSpeed"), 0.5), 3),
                "size": round(curve_scalar(init.get("startSize"), 1.0) * wscale, 3),
                "color": gradient_rgba(init.get("startColor")),
                "gravity": round(curve_scalar(init.get("gravityModifier"), 0.0), 3),
                "rate": round(curve_scalar(g(emis, "rateOverTime"), 4.0), 3),
                "maxParticles": int(g(init, "maxNumParticles", 64) or 64),
                "tiles": [int(g(uv, "tilesX", g(uv, "xTile", 1)) or 1),
                          int(g(uv, "tilesY", g(uv, "yTile", 1)) or 1)],
                "uvEnabled": bool(g(uv, "enabled", False)),
                "uvFps": round(float(g(uv, "fps", 30.0) or 30.0), 2),
                "uvCycles": round(curve_scalar(g(uv, "cycles"), 1.0), 3),
                "shapeRadius": round(float(g(g(shape, "radius"), "value",
                                             g(shape, "radius", 0.3) if not isinstance(g(shape, "radius"), dict) else 0.3) or 0.3), 3),
            })

    out_path = os.path.join(pack, "particles.json")
    json.dump({"emitters": emitters, "note": "looping ParticleSystems (game data; viewer flipbook billboards)"},
              open(out_path, "w"), indent=1)
    print(f"[particles] {len(emitters)} looping emitters kept of {n_loop} looping / {n_seen} total; "
          f"{sum(1 for v in tex_cache.values() if v)} atlas textures -> {out_path}")


if __name__ == "__main__":
    main()
