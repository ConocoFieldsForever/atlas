"""Bake the authoritative EFT display chain into a 3D LUT for the web viewers.

Ports blender_side/50_environment.py + 60_enhance.grade_lut_fit() EXACTLY:
    linear (pre-exposed) -> Hejl-Dawson (EFT-fitted constants 6.2/0.05/0.8, 0.004/0.06;
    bakes the display encode - output is display-referred, NEVER re-encode) ->
    per-channel film curves -> fitted Fahrenheit stage (3x3 matrix + 16-pt curves,
    mixed at fit.mix) -> clamp.
Exposure (eye adaptation) and vignette stay in the shader (dynamic / spatial).

Input domain shaper: c_linear = 4*u^2 (u in [0,1], 64 steps) - Hejl saturates well
below 4, so the shaper spends resolution where the curve lives.
Layout: 512x512 PNG, 8x8 tiles of 64x64; slice b: tile (b%8, b//8); in-tile x=R, y=G
(same family as the game's vintage LUT: blue = row*8+col).

  python extraction/grade/make_grade_lut.py [path/to/eft_grade_fit.json] [out.png]

Portable kit notes: this bakes the LEGACY RECONSTRUCTED look (Hejl/film/Fahrenheit fit; upstream
keeps it as eft_grade_lut_recon.bin). The shipped extraction/grade/eft_grade_lut.bin is the NEWER
AUTHENTIC game LUT baked by make_grade_lut_game.py (2026-07-13) -- regenerate THAT one with
make_grade_lut_game.py, not this script. This stays in the kit for the legacy look + provenance.
The fit (eft_grade_fit.json) ships next to this script; it is map-agnostic (global display chain).
"""
import json, os, sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from PIL import Image

# portable kit: default fit ships NEXT TO this script; override via argv[1] or EFT_GRADE_FIT.
FIT = (sys.argv[1] if len(sys.argv) > 1 else
       os.environ.get("EFT_GRADE_FIT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "eft_grade_fit.json")))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "eft_grade_lut.png")

N = 64
fit = json.load(open(FIT))
W = np.array(fit["weights"], np.float64)          # 3x3, rows = output channel
MIX = float(fit.get("mix", 0.498))

def hejl(c):
    x = np.maximum(c - 0.004, 0.0)
    return (x * (6.2 * x + 0.05)) / (x * (6.2 * x + 0.8) + 0.06)

# film curves (50_environment.py pcur): per-channel control points, PCHIP (monotone
# cubic ~ Blender's auto-handle curve for these smooth ramps)
FILM = [
    [(0.0, 0.0393), (1.0, 1.0)],
    [(0.0, 0.0347), (0.7996, 0.9309), (1.0, 1.0)],
    [(0.0, 0.0133), (1.0, 0.998)],
]
film_f = [PchipInterpolator([p[0] for p in pts], [p[1] for p in pts]) for pts in FILM]
fitc_f = [PchipInterpolator(np.linspace(0, 1, len(c)), np.asarray(c, np.float64))
          for c in fit["curves"]]

u = np.arange(N) / (N - 1)
c = 4.0 * u * u                                   # shaper: linear input per axis sample
r, g, b = np.meshgrid(c, c, c, indexing="ij")     # [R,G,B] axes
rgb = np.stack([r, g, b], -1).reshape(-1, 3)      # (N^3, 3) linear

h = hejl(rgb)                                     # display-referred after Hejl
hf = np.stack([np.clip(film_f[k](np.clip(h[:, k], 0, 1)), 0, 1) for k in range(3)], -1)
v = hf @ W.T                                      # fitted matrix (rows = out channel)
vc = np.stack([np.clip(fitc_f[k](np.clip(v[:, k], 0, 1)), 0, 1) for k in range(3)], -1)
out = np.clip(hf * (1 - MIX) + vc * MIX, 0.0, 1.0)

# pack: slice b -> tile (b%8, b//8); in-tile (x=R idx, y=G idx). out index order is [r,g,b].
img = np.zeros((512, 512, 3), np.float64)
cube = out.reshape(N, N, N, 3)                    # [ri, gi, bi]
for bi in range(N):
    tx, ty = bi % 8, bi // 8
    img[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64] = cube[:, :, bi].transpose(1, 0, 2)  # y=G, x=R
Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(OUT)
# raw RGBA bytes for the viewers (DataTexture: exact, no PNG flipY/colorSpace ambiguity)
rgba = np.concatenate([(img * 255 + 0.5).astype(np.uint8),
                       np.full((512, 512, 1), 255, np.uint8)], -1)
open(OUT.replace('.png', '.bin'), 'wb').write(rgba.tobytes())
print(f"baked {OUT} (+.bin) from {FIT} (mix {MIX}, rmse {fit.get('rmse')})")
