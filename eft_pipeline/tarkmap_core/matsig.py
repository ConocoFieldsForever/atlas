"""Material-signature grouping keys for the assembler. Split out of assemble_instanced.py (2026-07-01, verbatim).

Grouping by mesh NAME ALONE was a bug: EFT reuses ONE geometry with DIFFERENT per-instance textures (freight
containers/doors/grates ship blue/green/red), but EXT_mesh_gpu_instancing binds ONE material per node, and the emit
loop took the FIRST instance's subs for the whole group -> every container of a mesh rendered the first instance's
colour (a blue container came out red at LOD0 / green at LOD1, since each LOD is a separate obj with its own
first-instance colour). 219/17692 interchange meshes are mixed-texture. Keying grouping on the per-sub MATERIAL
SIGNATURE splits each colour into its own instanced node with the correct material; same-material instances still
instance together (near-zero cost for the 99% single-colour meshes). Geometry for a multi-colour mesh is emitted once
per colour (identical accessors dedup downstream)."""


def vp_sig(vp):
    if not vp: return None
    return (tuple((ly.get('tex'), ly.get('nrm'), tuple(ly.get('uv') or ()), tuple(ly.get('col') or ())) for ly in (vp.get('layers') or [])), vp.get('heights'), vp.get('blend'))


def _cut_key(s):
    """Authored _Cutoff for the dedup key. 0.0 is a REAL value ("discard nothing") and must not
    collapse onto the 0.5 default -- `or 0.5` did exactly that, so a cut=0.0 material and an
    otherwise identical cut=0.5 material shared ONE signature and ONE materials.json record.
    Only a missing cut defaults (decals.json writes "cut": None)."""
    c = s.get('cut')
    return round(float(0.5 if c is None else c), 3)


def sub_sig(subs):
    """Everything material_for()/the UV-bake keys on: tex/nrm/col/shader/role/cutoff/uv-tiling/vert-paint + the
    faithfulness extras (emissive/gloss/metal/bumpScale — materials differing only in these must NOT collapse)
    (+ face count so a differing geometry split can't collapse)."""
    return tuple((s.get('tex'), s.get('nrm'), tuple(s.get('col') or ()), s.get('sh'), s.get('role'),
                  _cut_key(s), tuple(s.get('uv') or (1, 1, 0, 0)), vp_sig(s.get('vp')), s.get('n', -1),
                  s.get('emis'), tuple(s.get('emisCol') or ()), s.get('gloss'), s.get('metal'), s.get('bumpScale'),
                  # real-specular + detail-map fields (2026-07): materials differing only in spec/smoothness source
                  # or detail layering must NOT collapse (different roughness/detail textures + extras)
                  s.get('spec'), s.get('smA'), s.get('detA'), s.get('detN'), tuple(s.get('detAuv') or ()),
                  tuple(s.get('detNuv') or ()), s.get('detAI'), s.get('detNS'))
                 for s in (subs or []))
