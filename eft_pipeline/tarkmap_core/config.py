"""Declarative per-map config -- VENDORED into eft_native_viewer/eft_pipeline/tarkmap_core.

This is the upstream tarkmap/tarkmap/config.py with ONE change: the ROOT / MAPS_DIR
anchors are repointed at the in-place beamng_blender_pipeline/tarkmap tree instead of
being derived from this file's own location. The new native-viewer repo is code-only and
references the map configs + datasets IN PLACE (disk is tight; nothing is copied), so:

  * maps/<id>/config.json is read from  <beamng>/tarkmap/maps/<id>/config.json
  * a config's relative source.root (e.g. "eft_assets/interchange_v2") resolves against
    <beamng> exactly as upstream did (ROOT/.. == the beamng dir).

If you later copy the maps/ tree and datasets into this repo, set EFT_TARKMAP_ROOT to the
new tarkmap dir (or drop a maps/ dir next to eft_pipeline and flip ROOT to derive locally).
Everything below the anchors is verbatim upstream logic.
"""
import json, os, functools
import numpy as np
print = functools.partial(print, flush=True)

SCHEMA_VERSION = 1

# --- in-place anchors (the ONLY divergence from upstream config.py) ------------------------
# Override with EFT_TARKMAP_ROOT to point at a relocated tarkmap tree.
_DEFAULT_UPSTREAM = r"C:\Users\user\beamng_blender_pipeline\tarkmap"
ROOT = os.environ.get("EFT_TARKMAP_ROOT", _DEFAULT_UPSTREAM)                 # <beamng>/tarkmap
MAPS_DIR = os.path.join(ROOT, 'maps')

DEFAULTS = {
    "schema_version": SCHEMA_VERSION,
    "coordinates": {"up": "y", "source_handedness": "left", "target_handedness": "right",
                    "global_matrix": None,
                    "flip_winding_on_negative_determinant": True},
    "tiling": {"strategy": "fixed_world_size", "tile_size_m": 128.0, "split_crossing_triangles": True,
               "bounds": "filtered_content_percentile"},
    "lod": {"method": "meshoptimizer", "metric": "screen_error", "sse_px": 16.0,
            "levels": [{"target_error": 0.0}, {"target_error": 0.02}, {"target_error": 0.06}],
            "preserve_boundaries": True, "preserve_material_borders": True, "hysteresis": 1.25},
    "ao": {"mode": "vertex", "samples": 64, "radius": 6.0, "intensity": 0.55,
           "sun": [0.35, 0.82, 0.45], "ambient": 0.35, "bake": "occlusion_only"},
    "textures": {"max_size": 1024,
                 "albedo": {"encode": "basis-lz", "colorspace": "srgb"},
                 "normal": {"encode": "uastc", "colorspace": "linear"},
                 "atlas": True, "dedup": True},
    "tiers": {"mobile": {"normals": False, "lightmap": True, "max_draws": 250, "max_active_vram_mb": 60},
              "desktop": {"normals": True, "ssao": True, "max_draws": 1500}},
    "validation": {"unknown_shader": "warn", "missing_texture": "warn", "missing_texture_area_pct": 8.0},
    "scene_filters": [], "material_rules": [], "terrain": {"type": "microsplat"},
    "cull": {"keep_root_prefix": None, "drop_root_regex": None, "drop_shadow_submeshes": True},
    "budget_mb": 40,
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class MapConfig:
    def __init__(self, d, path):
        self.path = path
        self.d = _deep_merge(DEFAULTS, d)
        self._validate()

    @classmethod
    def load(cls, map_id, maps_dir=MAPS_DIR):
        p = os.path.join(maps_dir, map_id, 'config.json')
        if not os.path.exists(p):
            raise FileNotFoundError(f"no config for map '{map_id}' at {p}")
        return cls(json.load(open(p, encoding='utf-8')), p)

    def _validate(self):
        d = self.d
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version {d.get('schema_version')} != {SCHEMA_VERSION}")
        for req in ("id", "name", "source"):
            if req not in d: raise ValueError(f"config missing required key: {req}")
        src = d["source"]
        if "root" not in src: raise ValueError("source.root required")
        if not os.path.isabs(src["root"]):
            # a relative source.root (e.g. "eft_assets/interchange_v2") resolves against the
            # datasets dir: EFT_ASSETS_ROOT when set (its leading "eft_assets" component is
            # the datasets dir itself, so strip it), else the tarkmap parent as upstream did.
            rel = src["root"].replace("\\", "/")
            assets = os.environ.get("EFT_ASSETS_ROOT")
            if assets:
                parts = [p for p in rel.split("/") if p]
                if parts and parts[0] == "eft_assets":
                    parts = parts[1:]
                src["root"] = os.path.normpath(os.path.join(assets, *parts))
            else:
                src["root"] = os.path.normpath(os.path.join(ROOT, "..", rel))   # resolve rel to workspace dir
        for sub, default in (("scene", "scene.json"), ("mesh_dir", "meshes"), ("texture_dir", "tex")):
            src.setdefault(sub, default)

    # ---- typed access ----
    def get(self, dotted, default=None):
        cur = self.d
        for k in dotted.split('.'):
            if not isinstance(cur, dict) or k not in cur: return default
            cur = cur[k]
        return cur

    @property
    def id(self): return self.d["id"]
    @property
    def name(self): return self.d["name"]
    @property
    def dataset(self): return self.d["source"]["root"]
    def src_path(self, *parts): return os.path.join(self.dataset, *parts)

    def coord_matrix(self):
        m = self.get("coordinates.global_matrix")
        return np.array(m, np.float64).reshape(4, 4) if m else np.eye(4)

    def __repr__(self): return f"<MapConfig {self.id} dataset={self.dataset}>"


def list_maps(maps_dir=MAPS_DIR):
    if not os.path.isdir(maps_dir): return []
    return sorted(d for d in os.listdir(maps_dir) if os.path.exists(os.path.join(maps_dir, d, 'config.json')))


if __name__ == '__main__':
    import sys
    for mid in (sys.argv[1:] or list_maps()):
        c = MapConfig.load(mid)
        print(f"{c}  tile={c.get('tiling.tile_size_m')}m  ao={c.get('ao.mode')}  dataset_exists={os.path.isdir(c.dataset)}")
