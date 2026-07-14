"""tarkmap_core -- correctness code VENDORED verbatim from beamng_blender_pipeline/tarkmap/tarkmap.

These modules are the tarkov-unity-extraction "correctness catalog" reused UNCHANGED by
assemble_bevy.py so the native pack cannot drift from the proven web/UE placement math:

  * instmath.py  -- make_conjugator (G@M@G^-1 similarity conjugation), det3, trs (WEB-ONLY,
                    unused by the Bevy fork), bake_into (degenerate/pinv world-bake fallback).
  * culls.py     -- Culls.filter (structural + off-map-backdrop cull), keep_submesh.
  * objio.py     -- load_obj / load_vcol.
  * matsig.py    -- sub_sig (the (mesh, material-signature) grouping key).
  * config.py    -- MapConfig (VENDORED WITH ONE EDIT: ROOT/MAPS_DIR repointed at the in-place
                    beamng tree; see its docstring).

Re-vendor with:  copy <beamng>/tarkmap/tarkmap/{instmath,culls,objio,matsig}.py here verbatim
(config.py is the patched copy -- do NOT overwrite it with upstream).
"""
