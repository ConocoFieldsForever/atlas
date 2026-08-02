"""Index the game's Unity bundles for the viewer's Assets tab.

The viewer cannot read bundles itself (UnityPy is Python), so it shells out here. Everything is
written under `<pack>/assets/` — the viewer reads those files, never the bundles:

  summary <pack>                       levels + per-type object counts        (once, ~40s)
  catalog <pack>                       GLOBAL search index + script/type       (once, ~2min)
                                       histograms across every level
  level   <pack> <lv>                  one level's scene graph                (on demand, cached)
  dump    <pack> <lv> <pid>            one object's typetree                  (on demand)
  asset   <pack> <origin> <fid> <pid>  resolve + preview a shared asset       (on demand)

Scale is why this is split up. Streets references 238 level bundles holding 4.2M objects between
them, and level233 alone has 203,381; building everything eagerly would cost minutes and hundreds of
MB for a tree the user opens two nodes of. A level bundle loads in ~0.2s, so `level` is cheap enough
to run on the click that needs it.

THE JOIN. The pack's instances carry `par` = _fold32(parent Transform path_id) and `lv`. Every
GameObject here records the same fold of its own Transform, so picked geometry resolves to the exact
source object with no name matching. Keep `_fold32` identical to eft_pipeline/assemble_bevy.py.

WHERE THE ASSETS ACTUALLY ARE. Level bundles hold ONLY scene objects (GameObject/Transform/
MeshFilter/MeshRenderer/collider/MonoBehaviour) plus PhysicMaterial — zero Mesh, Texture2D or
Material. Those live in `sharedassets*.assets` and are reached by PPtr, where `m_FileID` indexes the
*referring file's* externals table. That last part is the trap: a Material found in
sharedassets161 resolves its texture PPtrs against sharedassets161's externals, NOT the level's. So
`asset` takes the ORIGIN file explicitly and reports the origin of what it resolved, letting the UI
follow a chain (renderer -> material -> texture) without ever guessing a file.

PARTIAL TYPE TREES. EFT bundles ship type trees, but il2cpp stripping leaves many custom
MonoBehaviours describing only the four base fields. `dump` reports the shortfall in bytes rather
than presenting a truncated read as the whole object -- see `_read_object`.
"""
import collections
import json
import os
import re
import struct
import sys

EFTDATA = os.environ.get(
    "EFT_GAME_DATA", r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")

# Components worth showing as tree children of their GameObject. Everything else in a level bundle
# is a shared ASSET (Mesh/Texture2D/Material/Shader/...) that no single GameObject owns; those are
# listed per-level under "assets" instead of being duplicated under every user.
_ASSET_TYPES = {
    "Mesh", "Texture2D", "Material", "Shader", "Sprite", "AnimationClip", "AudioClip",
    "Cubemap", "Font", "MonoScript", "TextAsset", "AssetBundle", "PhysicMaterial",
    "RenderTexture", "Avatar", "AnimatorController", "LightingSettings", "Texture2DArray",
}


def _fold32(x):
    """Fold a signed 64-bit Unity path_id to the u32 the pack carries. MUST match assemble_bevy.py's
    `_fold32` -- this is the pick->object join key. 0 stays 0 = 'no ancestor'."""
    x = int(x or 0)
    return int((x ^ (x >> 32)) & 0xFFFFFFFF)


def _bundle(lv):
    p = os.path.join(EFTDATA, f"level{lv}" if str(lv).isdigit() else str(lv))
    if not os.path.exists(p):
        raise SystemExit(f"no such bundle: {p}")
    return p


def _pack_levels(pack):
    """The level bundles this pack's geometry actually came from, read straight off instances.bin.

    A map references a couple hundred of the game's 766 level bundles; indexing the rest would be
    noise. Falls back to every level bundle on disk if the pack has no instances (never expected).
    """
    import numpy as np
    idt = np.dtype([("affine", "<f4", (12,)), ("meshId", "<u4"), ("lodGroup", "<i4"),
                    ("lodIndex", "<i4"), ("rootId", "<u4"), ("flags", "<u4"),
                    ("par", "<u4"), ("par2", "<u4"), ("lv", "<u4")])
    a = np.fromfile(os.path.join(pack, "instances.bin"), dtype=idt)
    lv = sorted({int(v) for v in a["lv"].tolist() if int(v) > 0})
    return lv


def _out_dir(pack):
    d = os.path.join(pack, "assets")
    os.makedirs(d, exist_ok=True)
    return d


def _name_of(d):
    """m_Name, tolerating the two places Unity hides it: the extra nested "data" key MonoBehaviour
    typetrees sometimes carry, and `m_ParsedForm` on a Shader (whose top-level m_Name is empty)."""
    if not isinstance(d, dict):
        return ""
    for holder in (d, d.get("data"), d.get("m_ParsedForm")):
        if isinstance(holder, dict) and holder.get("m_Name"):
            return holder["m_Name"]
    return ""


# MonoScript class names, cached across levels (externals repeat heavily). Ported from
# extraction/intel/extract_gamedata.py rather than reinvented -- `scripts` in explore_bundles.py
# prints "<script 4141>" precisely because it lacks this cross-file step.
_ms_idx = {}


def _monoscript_index(path):
    import UnityPy
    if path not in _ms_idx:
        idx = {}
        if os.path.exists(path):
            try:
                e = UnityPy.load(path)
                for o in e.objects:
                    if o.type.name == "MonoScript":
                        try:
                            idx[o.path_id] = o.read_typetree().get("m_ClassName")
                        except Exception:
                            pass
                del e
            except Exception:
                pass
        _ms_idx[path] = idx
    return _ms_idx[path]


def _resolver(env, objs):
    """(m_FileID, m_PathID) -> MonoScript class name. m_FileID 0 is this file; >0 indexes the
    serialized file's externals table (m_Script usually points at globalgamemanagers.assets)."""
    sf = next((f for f in env.files.values() if hasattr(f, "objects")), None)
    externals = list(getattr(sf, "externals", []) or [])
    local = {}
    for o in objs:
        if o.type.name == "MonoScript":
            try:
                local[o.path_id] = o.read_typetree().get("m_ClassName")
            except Exception:
                pass

    def resolve(fid, pid):
        try:
            if not fid:
                return local.get(pid)
            base = os.path.basename(getattr(externals[fid - 1], "path", "").replace("\\", "/"))
            return _monoscript_index(os.path.join(EFTDATA, base)).get(pid)
        except Exception:
            return None
    return resolve


# ---------------------------------------------------------------------------
# Cross-file resolution. `m_FileID` is an index into the REFERRING file's externals table, so every
# lookup has to name the file the pointer came from -- see the module docstring.
# ---------------------------------------------------------------------------
_files = {}          # basename -> (env, externals[])


def _load_named(name):
    """Load a game-data file by basename, cached for the life of the process."""
    import UnityPy
    if name not in _files:
        p = os.path.join(EFTDATA, name)
        if not os.path.exists(p):
            _files[name] = (None, [])
        else:
            env = UnityPy.load(p)
            sf = next((f for f in env.files.values() if hasattr(f, "objects")), None)
            ext = [os.path.basename(str(getattr(e, "path", "")).replace(chr(92), "/"))
                   for e in (getattr(sf, "externals", []) or [])]
            _files[name] = (env, ext)
    return _files[name]


_obj_idx = {}        # basename -> {path_id: obj}


def _objects_of(name):
    """path_id -> object for one file, built once. A level can reference thousands of shared meshes;
    scanning `env.objects` per reference would make that quadratic."""
    if name not in _obj_idx:
        env, _ = _load_named(name)
        _obj_idx[name] = {o.path_id: o for o in env.objects} if env is not None else {}
    return _obj_idx[name]


def _holder_of(origin, fid):
    """Which file a PPtr with this m_FileID lands in, or None if the externals table is short."""
    if not fid:
        return origin
    _, ext = _load_named(origin)
    return ext[fid - 1] if fid - 1 < len(ext) else None


def _deref(origin, fid, pid):
    """Follow a PPtr from `origin` (a file basename). Returns (obj, holder_basename) or (None, None).

    `fid == 0` means the object lives in `origin` itself; otherwise it indexes origin's externals.
    The holder is returned so a caller can follow the NEXT pointer from the right file.
    """
    if not pid:
        return None, None
    holder = _holder_of(origin, fid)
    if holder is None:
        return None, None
    return _objects_of(holder).get(pid), holder


def _pptr(d, key):
    """(m_FileID, m_PathID) out of a typetree field, or (0, 0)."""
    v = (d or {}).get(key) or {}
    if not isinstance(v, dict):
        return 0, 0
    return int(v.get("m_FileID") or 0), int(v.get("m_PathID") or 0)


def _named_from(origin, fid, pid):
    """Just the m_Name of a PPtr target (used to label MeshFilter/Material rows)."""
    o, holder = _deref(origin, fid, pid)
    if o is None:
        return "", ""
    try:
        return _name_of(o.read_typetree(check_read=False)), holder
    except Exception:
        return "", holder


# Unity TextureFormat enum -> label. Only the values EFT actually ships are worth naming; anything
# else is reported as its raw number rather than guessed at.
_TEXFMT = {
    1: "Alpha8", 3: "RGB24", 4: "RGBA32", 5: "ARGB32", 7: "RGB565", 10: "DXT1", 12: "DXT5",
    13: "RGBA4444", 14: "BGRA32", 17: "RHalf", 18: "RGHalf", 19: "RGBAHalf", 20: "RFloat",
    21: "RGFloat", 22: "RGBAFloat", 24: "RGB9e5Float", 26: "BC6H", 27: "BC7", 28: "BC4",
    29: "BC5", 34: "R8", 41: "R16", 45: "BC6H", 46: "BC7", 47: "ASTC_4x4", 48: "ASTC_5x5",
}

# Component types worth a bit in the global search mask. Order is the BIT ORDER and must stay
# stable -- the viewer's chip list indexes it. Bit 31 is "something else".
COMP_BITS = [
    "MeshRenderer", "MeshFilter", "MeshCollider", "BoxCollider", "SphereCollider",
    "CapsuleCollider", "LODGroup", "Light", "MonoBehaviour", "ParticleSystem", "Animator",
    "AudioSource", "Rigidbody", "Terrain", "ReflectionProbe", "SkinnedMeshRenderer",
    "Camera", "Canvas", "NavMeshObstacle", "OcclusionArea", "LightProbeGroup", "TerrainCollider",
]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def cmd_summary(pack):
    """Per-level object + type counts for every level the pack uses. Type names come off the object
    header, so this needs no typetree reads and runs ~0.2s per level."""
    import UnityPy
    levels = _pack_levels(pack)
    out = {"levels": [], "gameData": EFTDATA}
    for i, lv in enumerate(levels):
        p = os.path.join(EFTDATA, f"level{lv}")
        if not os.path.exists(p):
            out["levels"].append({"lv": lv, "objects": 0, "missing": True, "types": {}})
            continue
        try:
            env = UnityPy.load(p)
            c = collections.Counter(o.type.name for o in env.objects)
            out["levels"].append({"lv": lv, "objects": sum(c.values()), "types": dict(c.most_common())})
            del env
        except Exception as e:
            out["levels"].append({"lv": lv, "objects": 0, "error": str(e)[:200], "types": {}})
        if i % 25 == 0:
            print(f"[assets] summary {i}/{len(levels)}", file=sys.stderr, flush=True)
    fp = os.path.join(_out_dir(pack), "summary.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(fp)


# ---------------------------------------------------------------------------
# catalog — the GLOBAL index. One pass over every level the pack uses, producing what the tab needs
# to answer "find X" and "what kinds of things exist" WITHOUT opening a bundle.
# ---------------------------------------------------------------------------
# search.bin record: lv u32 | pathId i64 | fold u32 | nameOff u32 | nameLen u16 | scriptId u16 |
#                    compMask u32   == 28 bytes, padded to 32 so a viewer-side slice is aligned.
REC = struct.Struct("<IqIIHHI")
REC_PAD = 32
assert REC.size <= REC_PAD


def cmd_catalog(pack):
    """Build the global GameObject search index + script/component histograms.

    Only GameObjects go in the index: they are what a person searches for. Components are reachable
    from their owner and are summarised as a per-object bitmask plus the histograms.
    """
    import UnityPy
    levels = _pack_levels(pack)
    scripts = {}          # class name -> id
    script_hist = collections.Counter()
    comp_hist = collections.Counter()
    names = bytearray()
    name_at = {}          # name -> offset (dedup: EFT reuses names heavily)
    recs = bytearray()
    n_go = 0
    lv_counts = {}

    for i, lv in enumerate(levels):
        p = os.path.join(EFTDATA, f"level{lv}")
        if not os.path.exists(p):
            continue
        try:
            env = UnityPy.load(p)
            objs = env.objects
            resolve = _resolver(env, objs)
            go_tt, tf_tt = {}, {}
            comp_of = collections.defaultdict(list)   # GameObject pid -> [type names]
            # ALL distinct script classes on each object, not just the first: ~5% of
            # script-carrying objects have two or more, and keeping only one would silently drop
            # them from a "list every object with this script" query the catalog offers.
            script_of = collections.defaultdict(set)  # GameObject pid -> {script class}
            for o in objs:
                t = o.type.name
                comp_hist[t] += 1
                try:
                    if t == "GameObject":
                        go_tt[o.path_id] = o.read_typetree(check_read=False)
                        continue
                    if t in ("Transform", "RectTransform"):
                        tf_tt[o.path_id] = o.read_typetree(check_read=False)
                        continue
                    if t in _ASSET_TYPES:
                        continue
                    d = o.read_typetree(check_read=False)
                except Exception:
                    continue
                _, gp = _pptr(d, "m_GameObject")
                if not gp:
                    continue
                comp_of[gp].append(t)
                if t == "MonoBehaviour":
                    fid, pid = _pptr(d, "m_Script")
                    cls = resolve(fid, pid) or ""
                    if cls:
                        script_hist[cls] += 1
                        script_of[gp].add(cls)
            go2tf = {}
            for pid, d in tf_tt.items():
                _, g = _pptr(d, "m_GameObject")
                if g:
                    go2tf[g] = pid
            lv_counts[lv] = len(go_tt)
            for gp, d in go_tt.items():
                nm = _name_of(d) or ""
                b = nm.encode("utf-8", "replace")[:255]
                off = name_at.get(nm)
                if off is None:
                    off = len(names)
                    names += b
                    name_at[nm] = off
                mask = 0
                for t in comp_of.get(gp, ()):
                    try:
                        mask |= 1 << COMP_BITS.index(t)
                    except ValueError:
                        mask |= 1 << 31
                # One record per DISTINCT script (or a single script-less record). The viewer
                # de-duplicates by (lv, pathId) for plain text queries, so the extra rows are only
                # visible to a script filter — which is exactly what needs them.
                fold = _fold32(go2tf.get(gp, 0))
                for cls in (sorted(script_of.get(gp) or ()) or [None]):
                    if cls is not None and cls not in scripts:
                        scripts[cls] = len(scripts)
                    sid = scripts.get(cls, 0xFFFF) if cls else 0xFFFF
                    rec = REC.pack(lv, gp, fold, off, len(b), min(sid, 0xFFFF), mask)
                    recs += rec + b"\0" * (REC_PAD - REC.size)
                n_go += 1
            del env
        except Exception as e:
            print(f"[assets] level{lv} failed: {e}", file=sys.stderr)
        if i % 20 == 0:
            print(f"[assets] catalog {i}/{len(levels)}  ({n_go:,} objects)", file=sys.stderr, flush=True)

    d = _out_dir(pack)
    with open(os.path.join(d, "search.bin"), "wb") as f:
        f.write(recs)
    with open(os.path.join(d, "search_names.bin"), "wb") as f:
        f.write(names)
    # scriptId -> name, in id order
    inv = [""] * len(scripts)
    for k, v in scripts.items():
        inv[v] = k
    cat = {
        "count": n_go,
        "recPad": REC_PAD,
        "scriptNames": inv,
        "compBits": COMP_BITS,
        "scripts": script_hist.most_common(),
        "components": comp_hist.most_common(),
        "levelObjects": {str(k): v for k, v in lv_counts.items()},
    }
    fp = os.path.join(d, "catalog.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(cat, f, separators=(",", ":"))
    print(fp)


# ---------------------------------------------------------------------------
# asset — resolve one shared asset (Mesh / Texture2D / Material / PhysicMaterial) for PREVIEW
# ---------------------------------------------------------------------------
# Preview geometry cap. A shipped mesh can run to hundreds of thousands of triangles; the panel
# thumbnail cannot show that and the JSON would dwarf everything else. Truncation is REPORTED
# (`trisShown` vs `tris`) rather than silently pretending the preview is the whole mesh.
MAX_PREVIEW_TRIS = 30000


def _mesh_preview(o):
    """Counts + bounds from the type tree (cheap, exact), geometry from export() (only if needed)."""
    d = o.read_typetree(check_read=False)
    subs = d.get("m_SubMeshes") or []
    tris = sum(int(s.get("indexCount") or 0) for s in subs) // 3
    # The MESH's own vertex count, not the sum over submeshes: submeshes may reference overlapping
    # vertex ranges, and summing them reports more vertices than the mesh has.
    verts = int((d.get("m_VertexData") or {}).get("m_VertexCount") or 0)
    if not verts:
        verts = sum(int(s.get("vertexCount") or 0) for s in subs)
    aabb = d.get("m_LocalAABB") or {}
    c, e = aabb.get("m_Center") or {}, aabb.get("m_Extent") or {}
    out = {
        "kind": "mesh",
        "tris": tris,
        "verts": verts,
        "submeshes": len(subs),
        "readable": bool(d.get("m_IsReadable")),
        "bounds": {
            "c": [round(float(c.get(k, 0.0)), 4) for k in "xyz"],
            "e": [round(float(e.get(k, 0.0)), 4) for k in "xyz"],
        },
    }
    # Geometry for the thumbnail. UnityPy leaves m_Vertices unpacked (the data is in m_VertexData /
    # m_StreamData), so export() is the supported way to get positions out.
    #
    # OBJ indexes position and UV SEPARATELY (`f v/vt/vn`), and one position can carry several UVs
    # along a seam. A renderer needs one array per attribute, so each distinct `v/vt` corner becomes
    # one output vertex — otherwise the texture tears along every seam.
    try:
        pos_raw, uv_raw = [], []
        verts, uvs, idx, remap = [], [], [], {}
        for line in o.read().export().splitlines():
            if line.startswith("v "):
                p = line.split()
                pos_raw.append([round(float(p[1]), 4), round(float(p[2]), 4), round(float(p[3]), 4)])
            elif line.startswith("vt "):
                p = line.split()
                uv_raw.append([round(float(p[1]), 5), round(float(p[2]), 5)])
            elif line.startswith("f "):
                if len(idx) // 3 >= MAX_PREVIEW_TRIS:
                    continue
                for corner in line.split()[1:4]:
                    j = remap.get(corner)
                    if j is None:
                        parts = corner.split("/")
                        vi = int(parts[0]) - 1 if parts[0] else -1
                        ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                        j = len(verts)
                        verts.append(pos_raw[vi] if 0 <= vi < len(pos_raw) else [0.0, 0.0, 0.0])
                        uvs.append(uv_raw[ti] if 0 <= ti < len(uv_raw) else [0.0, 0.0])
                        remap[corner] = j
                    idx.append(j)
        out["positions"] = verts
        out["uvs"] = uvs
        out["indices"] = idx
        out["trisShown"] = len(idx) // 3
    except Exception as e:
        out["geomError"] = f"{type(e).__name__}: {e}"[:200]
    return out


def _texture_preview(o, out_dir, tag):
    d = o.read_typetree(check_read=False)
    res = {
        "kind": "texture",
        "w": int(d.get("m_Width") or 0),
        "h": int(d.get("m_Height") or 0),
        "mips": int(d.get("m_MipCount") or 0),
        "formatId": int(d.get("m_TextureFormat") or 0),
    }
    res["format"] = _TEXFMT.get(res["formatId"], f"fmt{res['formatId']}")
    try:
        img = o.read().image
        if img is None:
            res["error"] = "no decodable image data"
            return res
        w, h = img.size
        scale = min(1.0, 512.0 / max(w, h, 1))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        fp = os.path.join(out_dir, f"tex_{tag}.png")
        img.convert("RGBA").save(fp)
        res["thumb"] = fp
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:200]
    return res


def _material_preview(o, holder):
    d = o.read_typetree()
    sp = d.get("m_SavedProperties") or {}

    def pairs(key):
        out = []
        for it in (sp.get(key) or []):
            if isinstance(it, (list, tuple)) and len(it) == 2:
                out.append((it[0], it[1]))
        return out

    sfid, spid = _pptr(d, "m_Shader")
    shader, _ = _named_from(holder, sfid, spid)
    slots = []
    for name, val in pairs("m_TexEnvs"):
        tf, tp = _pptr(val, "m_Texture")
        if not tp:
            continue
        tn, th = _named_from(holder, tf, tp)
        sc = (val or {}).get("m_Scale") or {}
        slots.append({"slot": name, "tex": tn, "origin": holder, "fileId": tf, "pathId": tp,
                      "scale": [round(float(sc.get("x", 1.0)), 3), round(float(sc.get("y", 1.0)), 3)]})
    colors = []
    for name, v in pairs("m_Colors"):
        if isinstance(v, dict):
            colors.append({"name": name,
                           "rgba": [round(float(v.get(k, 0.0)), 4) for k in ("r", "g", "b", "a")]})
    floats = [{"name": n, "v": round(float(v), 4)} for n, v in pairs("m_Floats")
              if isinstance(v, (int, float))]
    return {"kind": "material", "shader": shader or "(unresolved)", "slots": slots,
            "colors": colors, "floats": floats}


def _physic_preview(o):
    d = o.read_typetree(check_read=False)
    combine = {0: "Average", 1: "Minimum", 2: "Multiply", 3: "Maximum"}
    return {
        "kind": "physicMaterial",
        "dynamicFriction": round(float(d.get("dynamicFriction") or 0.0), 4),
        "staticFriction": round(float(d.get("staticFriction") or 0.0), 4),
        "bounciness": round(float(d.get("bounciness") or 0.0), 4),
        "frictionCombine": combine.get(int(d.get("frictionCombine") or 0), "?"),
        "bounceCombine": combine.get(int(d.get("bounceCombine") or 0), "?"),
    }


def cmd_asset(pack, origin, fid, pid):
    """Resolve a PPtr from `origin` and emit a preview payload for whatever it points at."""
    fid, pid = int(fid), int(pid)
    o, holder = _deref(origin, fid, pid)
    d = _out_dir(pack)
    tag = f"{holder or 'x'}_{pid}".replace(".", "_").replace(os.sep, "_")
    if o is None:
        res = {"kind": "missing", "origin": origin, "fileId": fid, "pathId": pid,
               "error": "the referenced file is not in this game install"}
    else:
        t = o.type.name
        try:
            if t == "Mesh":
                res = _mesh_preview(o)
            elif t in ("Texture2D", "Sprite", "Cubemap"):
                res = _texture_preview(o, d, tag)
            elif t == "Material":
                res = _material_preview(o, holder)
            elif t == "PhysicMaterial":
                res = _physic_preview(o)
            else:
                res = {"kind": "other", "fields": _read_object(o).get("fields")}
        except Exception as e:
            res = {"kind": "error", "error": f"{type(e).__name__}: {e}"[:300]}
        try:
            res["name"] = _name_of(o.read_typetree(check_read=False))
        except Exception:
            res["name"] = ""
        res["type"] = t
    res.update({"origin": holder or origin, "pathId": pid, "srcFile": holder or origin})
    fp = os.path.join(d, f"asset_{tag}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(res, f, separators=(",", ":"), default=str)
    print(fp)


# ---------------------------------------------------------------------------
# level
# ---------------------------------------------------------------------------
def cmd_level(pack, lv):
    """One level's scene graph: GameObjects (hierarchy + components) and shared assets.

    Emitted flat -- `nodes` is an array and every link is an index into it -- because the viewer
    expands lazily and a nested JSON would force it to walk the whole tree to find one node.
    """
    import UnityPy
    lv = int(lv)
    env = UnityPy.load(_bundle(lv))
    objs = env.objects
    resolve = _resolver(env, objs)

    by_pid = {}          # path_id -> object
    go_tt = {}           # GameObject path_id -> typetree
    tf_tt = {}           # Transform  path_id -> typetree
    comp_of = {}         # component path_id -> owner GameObject path_id
    assets = []          # shared assets (no owning GameObject)
    counts = collections.Counter()

    for o in objs:
        counts[o.type.name] += 1
        by_pid[o.path_id] = o

    for o in objs:
        t = o.type.name
        try:
            if t == "GameObject":
                go_tt[o.path_id] = o.read_typetree(check_read=False)
            elif t in ("Transform", "RectTransform"):
                tf_tt[o.path_id] = o.read_typetree(check_read=False)
        except Exception:
            pass

    # component -> owner, plus the ONE fact that makes each component row worth reading: the mesh a
    # MeshFilter points at, how many materials a renderer has, how many levels an LODGroup holds.
    # A row saying only "MeshFilter" tells the user nothing they could not already see.
    script_of = {}
    value_of = {}         # component path_id -> short label
    ref_of = {}           # component path_id -> {"o":origin,"f":fileId,"p":pathId} for preview
    mesh_refs = {}        # component path_id -> (fileId, pathId) resolved in a batch below
    origin = f"level{lv}"
    for o in objs:
        t = o.type.name
        if t in ("GameObject",) or t in _ASSET_TYPES:
            continue
        try:
            d = o.read_typetree(check_read=False)
        except Exception:
            continue
        _, gp = _pptr(d, "m_GameObject")
        if gp:
            comp_of[o.path_id] = gp
        if t == "MonoBehaviour":
            fid, pid = _pptr(d, "m_Script")
            script_of[o.path_id] = resolve(fid, pid) or ""
        elif t == "MeshFilter":
            f, p = _pptr(d, "m_Mesh")
            if p:
                mesh_refs[o.path_id] = (f, p)
                ref_of[o.path_id] = {"o": origin, "f": f, "p": p}
        elif t in ("MeshRenderer", "SkinnedMeshRenderer"):
            mats = [m for m in (d.get("m_Materials") or []) if isinstance(m, dict)]
            value_of[o.path_id] = f"{len(mats)} material" + ("s" if len(mats) != 1 else "")
            if mats:
                ref_of[o.path_id] = {"o": origin, "f": int(mats[0].get("m_FileID") or 0),
                                     "p": int(mats[0].get("m_PathID") or 0)}
        elif t == "LODGroup":
            n = len(d.get("m_LODs") or [])
            value_of[o.path_id] = f"{n} level" + ("s" if n != 1 else "")
        elif t.endswith("Collider"):
            bits = []
            if d.get("m_IsTrigger"):
                bits.append("trigger")
            if d.get("m_Convex"):
                bits.append("convex")
            f, p = _pptr(d, "m_Material")
            if p:
                ref_of[o.path_id] = {"o": origin, "f": f, "p": p}
                bits.append("physic material")
            if bits:
                value_of[o.path_id] = " · ".join(bits)
        elif t == "Light":
            kind = {0: "spot", 1: "directional", 2: "point", 4: "area"}.get(
                int(d.get("m_Type") or -1), "light")
            value_of[o.path_id] = f"{kind} · {float(d.get('m_Intensity') or 0.0):.2f}"

    # Batch-resolve the mesh NAMES: group by holding file so each shared file loads once.
    by_holder = collections.defaultdict(list)
    for cpid, (f, p) in mesh_refs.items():
        h = _holder_of(origin, f)
        if h:
            by_holder[h].append((cpid, p))
    for h, items in by_holder.items():
        idx = _objects_of(h)
        for cpid, p in items:
            t = idx.get(p)
            if t is None:
                continue
            try:
                value_of[cpid] = _name_of(t.read_typetree(check_read=False))
            except Exception:
                pass

    # GameObject path_id -> its Transform (for hierarchy + the fold32 join key)
    go2tf, tf_children, tf_father = {}, {}, {}
    for pid, d in tf_tt.items():
        g = (d.get("m_GameObject") or {}).get("m_PathID")
        if g:
            go2tf[g] = pid
        tf_father[pid] = (d.get("m_Father") or {}).get("m_PathID") or 0
        tf_children[pid] = [c.get("m_PathID") for c in (d.get("m_Children") or []) if c.get("m_PathID")]

    # ---- build the flat node array ----------------------------------------
    nodes = []
    idx_of_go = {}

    for pid, d in go_tt.items():
        tf = go2tf.get(pid, 0)
        idx_of_go[pid] = len(nodes)
        nodes.append({
            "t": "GameObject",
            "n": _name_of(d) or "(unnamed)",
            "p": pid,
            "a": 1 if d.get("m_IsActive") else 0,
            "f": _fold32(tf) if tf else 0,   # THE pick join key (fold of the Transform path_id)
            "tf": tf,
            "c": [],                          # component node indices (filled below)
            "k": [],                          # child GameObject node indices (filled below)
        })

    # components as child nodes of their GameObject
    for pid, gp in comp_of.items():
        o = by_pid.get(pid)
        if o is None:
            continue
        gi = idx_of_go.get(gp)
        rec = {"t": o.type.name, "p": pid, "n": script_of.get(pid, ""),
               "sz": int(getattr(o, "byte_size", 0) or 0)}
        if value_of.get(pid):
            rec["v"] = value_of[pid]
        if ref_of.get(pid):
            rec["r"] = ref_of[pid]
        if gi is None:
            nodes.append(rec)          # orphan component: still browsable, just not under a GO
            continue
        rec["go"] = gi
        nodes[gi]["c"].append(len(nodes))
        nodes.append(rec)

    # hierarchy: walk each GameObject's Transform up to its father's GameObject; no father = a root
    roots = []
    for gp, gi in idx_of_go.items():
        tf = go2tf.get(gp, 0)
        fa = tf_father.get(tf, 0)
        pg = (tf_tt.get(fa) or {}).get("m_GameObject", {}).get("m_PathID") if fa else None
        pi = idx_of_go.get(pg) if pg else None
        if pi is None:
            roots.append(gi)
        else:
            nodes[pi]["k"].append(gi)

    for o in objs:
        if o.type.name in _ASSET_TYPES:
            try:
                nm = _name_of(o.read_typetree(check_read=False))
            except Exception:
                nm = ""
            assets.append({"t": o.type.name, "p": o.path_id, "n": nm,
                           "sz": int(getattr(o, "byte_size", 0) or 0)})

    roots.sort(key=lambda i: nodes[i]["n"].lower())
    assets.sort(key=lambda a: (a["t"], a["n"].lower()))
    out = {"lv": lv, "counts": dict(counts.most_common()), "nodes": nodes,
           "roots": roots, "assets": assets}
    fp = os.path.join(_out_dir(pack), f"lv{lv}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(fp)


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------
_SHORT = re.compile(r"Expected to read (\d+) bytes, but only read (\d+) bytes")


def _read_object(o):
    """Read one object's typetree, and be HONEST about how much of it the bundle describes.

    A strict read raises ValueError when the type tree runs out before the object's bytes do -- the
    il2cpp-stripped MonoBehaviour case. Falling back to the lenient read yields the fields that ARE
    described, but presenting those alone would claim a four-field object where the game has one with
    44 more bytes of state. So the shortfall is returned as data and the UI states it.
    """
    size = int(getattr(o, "byte_size", 0) or 0)
    try:
        return {"fields": o.read_typetree(), "complete": True, "size": size}
    except ValueError as e:
        m = _SHORT.search(str(e))
        try:
            fields = o.read_typetree(check_read=False)
        except Exception as e2:
            return {"fields": None, "complete": False, "size": size, "error": f"{type(e2).__name__}: {e2}"[:300]}
        if m:
            exp, got = int(m.group(1)), int(m.group(2))
            return {"fields": fields, "complete": False, "size": exp, "read": got,
                    "undescribed": exp - got}
        return {"fields": fields, "complete": False, "size": size, "error": str(e)[:300]}
    except Exception as e:
        return {"fields": None, "complete": False, "size": size, "error": f"{type(e).__name__}: {e}"[:300]}


def cmd_dump(pack, lv, pid):
    import UnityPy
    lv, pid = int(lv), int(pid)
    env = UnityPy.load(_bundle(lv))
    objs = env.objects
    o = next((x for x in objs if x.path_id == pid), None)
    if o is None:
        raise SystemExit(f"path_id {pid} not found in level{lv}")
    res = _read_object(o)
    res["lv"] = lv
    res["pathId"] = pid
    res["type"] = o.type.name
    if o.type.name == "MonoBehaviour":
        f = res.get("fields") or {}
        s = f.get("m_Script") or {}
        res["script"] = _resolver(env, objs)(s.get("m_FileID", 0), s.get("m_PathID", 0)) or ""
        res["scriptRef"] = {"fileId": s.get("m_FileID", 0), "pathId": s.get("m_PathID", 0)}
    fp = os.path.join(_out_dir(pack), f"dump_{lv}_{pid}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(res, f, separators=(",", ":"), default=str)
    print(fp)


def cmd_albedo(pack, lv, go_pid):
    """The base-colour texture of one GameObject, resolved end to end in ONE subprocess.

    Skinning the mesh thumbnail needs renderer -> material -> _MainTex -> pixels, and each hop lands
    in a different file. Doing it here rather than as three round trips from the viewer keeps the
    chain (and its `m_FileID`-is-relative-to-the-referrer rule) in one place.
    """
    lv, go_pid = int(lv), int(go_pid)
    origin = f"level{lv}"
    d = _out_dir(pack)
    res = {"kind": "albedo", "lv": lv, "gameObject": go_pid}
    idx = _objects_of(origin)
    go = idx.get(go_pid)
    if go is None:
        res["error"] = "GameObject not in this level"
    else:
        try:
            comps = (go.read_typetree(check_read=False) or {}).get("m_Component") or []
            rend = None
            for c in comps:
                # the entry is either {"component": {PPtr}} or the PPtr itself, depending on version
                ptr = c.get("component") if isinstance(c, dict) and "component" in c else c
                pid = (ptr or {}).get("m_PathID") if isinstance(ptr, dict) else None
                obj = idx.get(pid) if pid else None
                if obj is not None and obj.type.name in ("MeshRenderer", "SkinnedMeshRenderer"):
                    rend = obj
                    break
            if rend is None:
                res["error"] = "no renderer on this object"
            else:
                rd = rend.read_typetree(check_read=False)
                mats = [m for m in (rd.get("m_Materials") or []) if isinstance(m, dict)]
                if not mats:
                    res["error"] = "renderer has no material"
                else:
                    mo, mh = _deref(origin, int(mats[0].get("m_FileID") or 0),
                                    int(mats[0].get("m_PathID") or 0))
                    if mo is None:
                        res["error"] = "material file not in this install"
                    else:
                        md = mo.read_typetree()
                        res["material"] = _name_of(md)
                        texenvs = ((md.get("m_SavedProperties") or {}).get("m_TexEnvs") or [])
                        slot = None
                        for it in texenvs:
                            if isinstance(it, (list, tuple)) and len(it) == 2 and it[0] == "_MainTex":
                                slot = it[1]
                                break
                        tf, tp = _pptr(slot or {}, "m_Texture")
                        if not tp:
                            res["error"] = "material has no _MainTex"
                        else:
                            to, th = _deref(mh, tf, tp)
                            if to is None:
                                res["error"] = "texture file not in this install"
                            else:
                                tex = _texture_preview(to, d, f"albedo_{lv}_{go_pid}")
                                res.update(tex)
                                res["kind"] = "albedo"
                                res["texture"] = _name_of(to.read_typetree(check_read=False))
                                res["srcFile"] = th
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"[:300]
    fp = os.path.join(d, f"albedo_{lv}_{go_pid}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(res, f, separators=(",", ":"), default=str)
    print(fp)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"summary": cmd_summary, "catalog": cmd_catalog, "level": cmd_level,
     "dump": cmd_dump, "asset": cmd_asset, "albedo": cmd_albedo}[cmd](*rest)
