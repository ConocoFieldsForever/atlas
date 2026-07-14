"""eft_pipeline -- the .eftpack emitter for the native Bevy EFT map viewer.

Run the emitter as a module from the repo root so the vendored core imports resolve:

    python -m eft_pipeline.assemble_bevy interchange

Contents:
  * assemble_bevy.py  -- forks tarkmap/assemble_instanced.py; emits the self-describing
                         .eftpack v1 (manifest.json + meshes.bin + instances.bin + materials.json).
  * tarkmap_core/     -- the correctness code vendored verbatim (see its __init__).
"""
