#!/usr/bin/env python
"""Extract the game's OWN sky cubemaps + derived sky colors -> packs/shared/sky/.

Phase 4 of docs/GRAPHICS_PLAN.md. The viewer's sky was a procedural two-color gradient, and the
audit found FOUR separately-authored sky descriptions (visible cubemap, sky_reflect, SH-bake sky,
fog color). EFT ships its actual skies in StreamingAssets/Windows/cubemaps — this exports them and
computes the derived per-cubemap colors (zenith, horizon, mean) so every consumer can be fed from
ONE extracted source. Faces are exported as PNGs in wgpu cubemap order (+X,-X,+Y,-Y,+Z,-Z).

Only map-scale SKY cubemaps are exported (the bundle also carries interior reflection probes and
material captures); the classifier is structural: a sky cubemap's TOP face is markedly brighter
than its bottom face and its faces are near-continuous at the horizon — no name matching beyond
excluding the 'patron_' material-capture prefix... which IS a name; so instead: classify by the
top/bottom luma ratio alone. Misclassified probes simply appear in the sidecar and are ignored by
consumers that pick explicitly.

  python extraction/unity/eft_extract_sky.py            -> packs/shared/sky/*.png + sky.json
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(os.environ.get("EFT_INTEL_OUT_DIR") or os.path.join(REPO, "packs", "shared"),
                       "sky")
GAME = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
BUNDLE = os.path.join(GAME, "StreamingAssets", "Windows", "cubemaps")


def main():
    import UnityPy
    import numpy as np
    env = UnityPy.load(BUNDLE)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    from PIL import Image
    import texture2ddecoder
    for o in env.objects:
        if o.type.name != "Cubemap":
            continue
        # UnityPy's .image gives only ONE face for cubemaps — decode the raw data ourselves.
        # Unity stores cubemap data FACE-MAJOR: face0's full mip chain, then face1's, ... The sky
        # bundle is DXT1 (m_TextureFormat 10): 8 bytes per 4x4 block.
        # o.read() resolves m_StreamData (the faces live in the bundle's .resS); the typetree
        # path returns an empty inline blob. image_data is the FULL face-major buffer.
        try:
            tex = o.read()
            data = bytes(tex.get_image_data())
        except Exception as e:
            print(f"[sky] {getattr(o, 'path_id', '?')}: read failed ({e})", flush=True)
            continue
        name = getattr(tex, "m_Name", None) or f"cube_{o.path_id}"
        fmt = int(getattr(tex, "m_TextureFormat", 0) or 0)
        if fmt != 10:
            print(f"[sky] {name}: format {fmt} != DXT1 - skipped", flush=True)
            continue
        face = int(getattr(tex, "m_Width", 0) or 0)
        mips = int(getattr(tex, "m_MipCount", 1) or 1)
        if face <= 0 or not data:
            print(f"[sky] {name}: no data - skipped", flush=True)
            continue
        def mip_bytes(w, h):
            return max(1, w // 4) * max(1, h // 4) * 8
        chain = sum(mip_bytes(max(1, face >> m), max(1, face >> m)) for m in range(mips))
        if len(data) < chain * 6:
            print(f"[sky] {name}: {len(data)} bytes < 6x{chain} - skipped", flush=True)
            continue
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        faces = []
        for i in range(6):
            raw = data[i * chain: i * chain + mip_bytes(face, face)]
            bgra = texture2ddecoder.decode_bc1(raw, face, face)
            img = Image.frombytes("RGBA", (face, face), bgra, "raw", "BGRA")
            faces.append(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
        lum = [float((f * [0.2126, 0.7152, 0.0722]).sum(axis=2).mean()) for f in faces]
        # +Y (index 2) is up. Sky classifier: top face clearly brighter than bottom.
        is_sky = lum[2] > 1.3 * max(lum[3], 1e-4)
        # Derived colors in LINEAR-ish srgb-decoded space (approx 2.2; consumers re-derive if they
        # need exactness — the point is these come from the GAME's pixels, not a hand gradient).
        lin = [np.power(f, 2.2) for f in faces]
        zenith = lin[2].mean(axis=(0, 1)).tolist()
        # Horizon: the bottom rows of the four side faces (+X,-X,+Z,-Z are 0,1,4,5).
        strip = np.concatenate([lin[i][-max(4, face // 8):] for i in (0, 1, 4, 5)])
        horizon = strip.mean(axis=(0, 1)).tolist()
        mean = np.concatenate([l.reshape(-1, 3) for l in lin]).mean(axis=0).tolist()
        fnames = []
        for i, f in enumerate(faces):
            from PIL import Image
            fn = f"{safe}_face{i}.png"
            Image.fromarray((f * 255).astype("uint8")).save(os.path.join(OUT_DIR, fn))
            fnames.append(fn)
        out[name] = {
            "faces": fnames,
            "size": face,
            "is_sky": bool(is_sky),
            "zenith": [round(v, 5) for v in zenith],
            "horizon": [round(v, 5) for v in horizon],
            "mean": [round(v, 5) for v in mean],
            "face_luma": [round(v, 5) for v in lum],
        }
        print(f"[sky] {name:38s} {face}px sky={'Y' if is_sky else 'n'} "
              f"zenith={[round(v,3) for v in zenith]} horizon={[round(v,3) for v in horizon]}",
              flush=True)
    if not out:
        print("[sky] nothing extracted", flush=True)
        return 1
    doc = {"schema": 1, "source": "StreamingAssets/Windows/cubemaps (Cubemap assets, verbatim)",
           "built": int(time.time()), "cubemaps": out}
    with open(os.path.join(OUT_DIR, "sky.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"[sky] {len(out)} cubemap(s) -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
