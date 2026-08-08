"""Sidecar classification and per-consumer input ids, shared by the assembler and the build driver.

A .eftpack is assembled in one step but FINISHED by five more (SH bake, grass, gamedata, icons,
nav). Everything those later stages write lands in the live pack, so the atomic swap has to carry
them across a rebuild or every geometry change would throw away hours of bake time. That intent is
right. Carrying them SILENTLY is not: nav, volume and grass are all derived FROM the assembled
world, so after a rebuild they can describe a world the pack no longer contains.

WHY THE ID IS PER-CONSUMER AND NOT ONE "geometry id"

A single hash over the geometry blobs marks wrongly in both directions:

  * it OVER-marks. A decal-only rebuild changes meshes.bin, so nav.bin would be declared stale and
    a ~30-minute re-bake demanded for spray paint that nav does not even consume (nav_bake skips
    role="decal" faces).
  * it UNDER-marks. The SH volume is baked against the LIGHTS, which are not geometry at all. Re-aim
    every spot cone and a geometry hash does not move, so a genuinely stale volume.bin is carried
    forward and reported as current.

Marking wrongly in either direction is how a warning gets trained into background noise. So each
consumer gets an id over the inputs IT actually reads, and nothing else.
"""
import hashlib
import json
import os

# Files assemble_bevy itself produces. No other stage writes these.
ASSEMBLER_OWNED = frozenset({
    'manifest.json', 'meshes.bin', 'instances.bin', 'materials.json',
    'colliders.bin', 'collider_meshes.bin', 'lod_integrity.json'})

# Derived from the pack by a stage that runs AFTER the swap -> mapped to the inputs each one reads.
# The key is the sidecar file; the value is the input set from _INPUT_SETS below.
GEOMETRY_DERIVED = {
    'nav.bin': 'nav', 'nav.json': 'nav', 'nav_blk.bin': 'nav',
    'nav_door.bin': 'nav', 'nav_wallcell.bin': 'nav',
    'volume.bin': 'light', 'volume.json': 'light',
    'volume_valid.bin': 'light', 'volume.vis.bin': 'light',
    'grass.bin': 'terrain', 'grass_sidecar.json': 'terrain',
    # gamedata is extracted from the game files, then JOINED to instances.bin's folded par/par2/lv
    # ancestry, so a rebuilt instance table breaks the join even though the game data did not move.
    'gamedata.json': 'ancestry', 'semantics.json': 'ancestry',
}

# Catalogs, network fetches and FX authored elsewhere: a rebuild cannot stale these.
INDEPENDENT = frozenset({
    'loot.json', 'tasks.json', 'grade_lut.bin', 'particles.json', 'icons', 'tex_fx',
    # asset dirs a self-contained pack carries: extracted art, not derived from the assembled world
    'tex', 'terrain_layers'})

# What each consumer actually reads. Blob names are hashed by content; 'lights' means the dataset's
# lights_*.json, which live outside the pack and are the SH bake's other half.
_INPUT_SETS = {
    # nav walks the render meshes AND the physics tier, but skips role="decal" faces.
    'nav': {'blobs': ('meshes.bin', 'instances.bin', 'colliders.bin', 'collider_meshes.bin'),
            'lights': False, 'skip_decal_materials': True},
    # the SH bake traces the same geometry and integrates the practical lights against it.
    'light': {'blobs': ('meshes.bin', 'instances.bin'),
              'lights': True, 'skip_decal_materials': True},
    # grass is seeded from the terrain instances only.
    'terrain': {'blobs': ('meshes.bin', 'instances.bin'),
                'lights': False, 'skip_decal_materials': True},
    # The join key is the instance table's ancestry columns. Decal instances are appended with a
    # synthetic root and no gameplay ancestry, so they cannot break the join and must not mark it.
    'ancestry': {'blobs': (), 'lights': False, 'skip_decal_materials': True},
}


def classify(name):
    """'assembler' | 'geometry' | 'independent' | 'unknown' for one pack entry."""
    if name in ASSEMBLER_OWNED:
        return 'assembler'
    if name in GEOMETRY_DERIVED:
        return 'geometry'
    if name in INDEPENDENT:
        return 'independent'
    # Extracted art and per-level intel that ride along in the pack. The grass prototype textures
    # and the light sidecars come from the game files, not from the assembled world, so a geometry
    # rebuild cannot stale them. (The SH volume DOES depend on the lights; that edge is covered by
    # the 'light' consumer's input id, not by this classification.)
    if name.startswith('grass_') and name.endswith('.png'):
        return 'independent'
    if name.startswith('lights_') and name.endswith('.json'):
        return 'independent'
    return 'unknown'


def _hash_file(h, path):
    h.update(os.path.basename(path).encode('utf-8'))
    if not os.path.isfile(path):
        return
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)


def _decal_material_ids(pack_dir):
    """Material ids whose role is 'decal'. Read from materials.json: the manifest carries only a
    count, and the role is the only thing that distinguishes paint from geometry."""
    try:
        with open(os.path.join(pack_dir, 'materials.json'), encoding='utf-8') as f:
            mats = json.load(f)
    except Exception:
        return set()
    if isinstance(mats, dict):
        mats = mats.get('materials') or []
    return {m.get('id') for m in mats
            if isinstance(m, dict) and m.get('role') == 'decal'}


def _mesh_table(manifest, decal_mats, skip_decals):
    """The geometry summary a consumer sees, as a list of per-mesh tuples.

    METADATA, not blob bytes, and deliberately so. Hashing meshes.bin whole costs a multi-GB read
    on every assemble and, worse, cannot be filtered: adding a spray-paint quad would move the id
    that nav is judged by, even though nav skips decal faces entirely. The offsets and counts here
    move whenever real geometry does, because every mesh after a changed one shifts.
    """
    rows = []
    for m in manifest.get('meshes', []):
        subs = m.get('submeshes') or []
        if skip_decals and subs and all(s.get('materialId', s.get('material_id')) in decal_mats
                                        for s in subs):
            continue                     # a wholly decal mesh: paint, not geometry
        rows.append([m.get('id'), m.get('name'), m.get('vtxOffset'), m.get('vtxCount'),
                     m.get('idxOffset'), m.get('idxCount')])
    return rows


def input_ids(pack_dir, manifest, dataset_dir=None):
    """{consumer: 'sha256:...'} over the inputs each consumer reads.

    Re-running the assembler on unchanged inputs reproduces every id exactly.
    """
    decal_mats = _decal_material_ids(pack_dir)

    lights_sig = None
    if dataset_dir and os.path.isdir(dataset_dir):
        lh = hashlib.sha256()
        for fn in sorted(os.listdir(dataset_dir)):
            if fn.startswith('lights_') and fn.endswith('.json') \
                    and not fn.endswith('_all.json') and not fn.endswith('.bak'):
                _hash_file(lh, os.path.join(dataset_dir, fn))
        lights_sig = lh.hexdigest()

    out = {}
    for consumer, spec in _INPUT_SETS.items():
        h = hashlib.sha256()
        skip = spec['skip_decal_materials']
        h.update(json.dumps({
            'v': 1,
            'bounds': manifest.get('bounds'),
            'colliderCount': manifest.get('colliderCount'),
            'instanceStride': manifest.get('instance', {}).get('stride'),
            # A decal adds an instance, so the raw count would move the nav id. Count only the
            # instances whose mesh survived the decal filter.
            'meshes': _mesh_table(manifest, decal_mats, skip),
        }, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        if 'colliders.bin' in spec['blobs']:
            # The physics tier has no decals in it, so it can be hashed by content cheaply.
            for n in ('colliders.bin', 'collider_meshes.bin'):
                _hash_file(h, os.path.join(pack_dir, n))
        if spec['lights']:
            h.update((lights_sig or 'no-lights').encode('utf-8'))
        out[consumer] = 'sha256:' + h.hexdigest()[:32]
    return out


def stale_sidecars(pack_dir, prev_ids, new_ids):
    """Names of migrated sidecars whose inputs moved. Empty on a first build (no previous ids)."""
    if not prev_ids:
        return []
    bad = []
    for name, consumer in sorted(GEOMETRY_DERIVED.items()):
        if not os.path.exists(os.path.join(pack_dir, name)):
            continue
        p, n = prev_ids.get(consumer), new_ids.get(consumer)
        if p and n and p != n:
            bad.append(name)
    return bad


REBUILD_HINT = {
    'nav': 'atlas bake-nav <pack>',
    'light': 'atlas bake-sh <pack>',
    'terrain': 'python -m eft_pipeline.build_grass --pack <pack>',
    'ancestry': 'python extraction/intel/extract_gamedata.py <map>',
}
