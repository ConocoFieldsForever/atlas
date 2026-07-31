"""Bake the viewer display LUT from the GAME'S OWN grading LUT (authentic replacement for the
reconstructed eft_grade_lut.bin, whose provenance was never auditable).

Source: LUT-amidgenofbluegreen2lighterblack (resources.assets pathid 524, extracted to
out/interchange/_postfx/lut_amidgen_bluegreen.png).
PPv2 LDR strip layout SOLVED EMPIRICALLY (output-gradient attribution, 2026-07-13):
image[row, tile*32 + x] with x = R input, tile = B input, row = G input (UnityPy export
orientation, no flip). Greys can't disambiguate this — the per-axis output gradients can
(dR/dx 0.024, dB/dtile 0.017, dG/drow 0.022).

Chain baked into the output (what the game does to a linear scene color):
  linear c (post-exposure) -> clamp 0..1 -> sRGB encode (EFT hard-clips highlights -
  authentic LDR grading) -> 32^3 game LUT (trilinear) -> display-encoded out.

Output format matches make_grade_lut.py EXACTLY so _grade.js is unchanged:
  512x512 RGBA8, 8x8 tiles of 64x64; slice b -> tile(b%8, b//8); in-tile x=R, y=G;
  input shaper c_lin = 4*u^2 (u = i/63); row 0 = data row 0 (raw DataTexture bytes).

Usage: python extraction/grade/make_grade_lut_game.py [src.png] [out.bin]

Portable kit notes: THIS is the baker that produced the shipped extraction/grade/eft_grade_lut.bin
(make_grade_lut.py bakes the older RECONSTRUCTED fit LUT — a different, legacy look). The source
strip ships next to this script; to re-extract it from YOUR OWN install instead, pull the Texture2D
named 'LUT-amidgenofbluegreen2lighterblack' out of <EFT_GAME_DATA>/resources.assets with UnityPy
and save it as a PNG (32 x 1024).
"""
import sys, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# args: [SRC.png] [OUT.bin], or just OUT.bin. SRC is optional — when it is missing the strip is
# EXTRACTED from the player's own installation below, which is the point: no game texture ships
# with this tool.
_args = sys.argv[1:]
if len(_args) == 1 and _args[0].lower().endswith('.bin'):
    SRC, OUT = os.path.join(HERE, 'lut_amidgen_bluegreen.png'), _args[0]
else:
    SRC = _args[0] if _args else os.path.join(HERE, 'lut_amidgen_bluegreen.png')
    OUT = _args[1] if len(_args) > 1 else os.path.join(HERE, 'eft_grade_lut.bin')


def _extract_from_game(dst_png):
    """Pull the game's own grading LUT strip out of the installed resources.assets.

    Located BY NAME (the game's own asset name), never by a hardcoded path id, so a game update
    that renumbers assets still resolves. Requires EFT_GAME_DATA (or the default install path).
    """
    import UnityPy
    data = os.environ.get(
        'EFT_GAME_DATA', r'C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data')
    name_hint = os.environ.get('EFT_GRADE_LUT_NAME', 'amidgen').lower()
    src = os.path.join(data, 'resources.assets')
    if not os.path.exists(src):
        raise SystemExit(f'no resources.assets at {src} — set EFT_GAME_DATA')
    env = UnityPy.load(src)
    best = None
    for o in env.objects:
        if o.type.name != 'Texture2D':
            continue
        try:
            t = o.read()
        except Exception:
            continue
        nm = (getattr(t, 'm_Name', '') or '')
        if name_hint in nm.lower():
            best = (nm, t)
            break
    if best is None:
        raise SystemExit(f'no Texture2D matching {name_hint!r} in {src}')
    nm, tex = best
    os.makedirs(os.path.dirname(dst_png) or '.', exist_ok=True)
    tex.image.save(dst_png)
    print(f'extracted grade LUT {nm!r} -> {dst_png}')
    return dst_png


if not os.path.exists(SRC):
    SRC = _extract_from_game(os.path.join(os.path.dirname(OUT) or '.', 'lut_grade_src.png'))

strip = np.asarray(Image.open(SRC).convert('RGB')).astype(np.float32) / 255.0   # 32 x 1024 x 3
assert strip.shape == (32, 1024, 3), strip.shape
# strip[row=G, tile*32+x] with x=R, tile=B  ->  game[r, g, b]
game = strip.reshape(32, 32, 32, 3).transpose(2, 0, 1, 3)   # [g,b,r] -> [r,g,b]
# verify the transpose: strip.reshape gives [g, b, x=r]; transpose(2,0,1) -> [r, g, b]

def srgb_encode(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)

def sample_game(x):
    """Trilinear sample of the 32^3 game LUT. x: (...,3) display-referred 0..1 -> (...,3)."""
    f = np.clip(x, 0, 1) * 31.0
    i0 = np.floor(f).astype(np.int32)
    i1 = np.minimum(i0 + 1, 31)
    t = f - i0
    tr, tg, tb = t[..., 0:1], t[..., 1:2], t[..., 2:3]
    def G(a, b_, c_):
        return game[a, b_, c_]
    c000 = G(i0[..., 0], i0[..., 1], i0[..., 2]); c100 = G(i1[..., 0], i0[..., 1], i0[..., 2])
    c010 = G(i0[..., 0], i1[..., 1], i0[..., 2]); c110 = G(i1[..., 0], i1[..., 1], i0[..., 2])
    c001 = G(i0[..., 0], i0[..., 1], i1[..., 2]); c101 = G(i1[..., 0], i0[..., 1], i1[..., 2])
    c011 = G(i0[..., 0], i1[..., 1], i1[..., 2]); c111 = G(i1[..., 0], i1[..., 1], i1[..., 2])
    c00 = c000 * (1 - tr) + c100 * tr; c10 = c010 * (1 - tr) + c110 * tr
    c01 = c001 * (1 - tr) + c101 * tr; c11 = c011 * (1 - tr) + c111 * tr
    c0 = c00 * (1 - tg) + c10 * tg; c1 = c01 * (1 - tg) + c11 * tg
    return c0 * (1 - tb) + c1 * tb

# 64^3 input grid in the viewer's shaper space
u = np.arange(64, dtype=np.float32) / 63.0
c_lin = 4.0 * u * u                                          # linear channel values 0..4
R, G_, B = np.meshgrid(c_lin, c_lin, c_lin, indexing='ij')   # [ir, ig, ib]
disp = srgb_encode(np.stack([R, G_, B], axis=-1))            # display-referred LUT input
out = sample_game(disp)                                      # [ir, ig, ib, 3]

# pack into the 512x512 8x8-tile flipbook: slice b -> tile(b%8, b//8); in-tile x=R, y=G
img = np.zeros((512, 512, 4), np.uint8)
img[..., 3] = 255
for b in range(64):
    tx, ty = (b % 8) * 64, (b // 8) * 64
    img[ty:ty + 64, tx:tx + 64, :3] = np.clip(np.round(out[:, :, b].transpose(1, 0, 2) * 255), 0, 255).astype(np.uint8)  # rows=G, cols=R

img.tofile(OUT)
print('wrote', OUT, img.shape, 'from', os.path.basename(SRC))
for v in (0.02, 0.05, 0.18, 0.5, 1.0, 2.0, 4.0):
    idx = int(round(np.sqrt(min(v / 4.0, 1.0)) * 63))
    tx, ty = (idx % 8) * 64, (idx // 8) * 64
    px = img[ty + idx, tx + idx, :3]
    print(f'  linear {v:>5} -> display {tuple(int(x) for x in px)}')
