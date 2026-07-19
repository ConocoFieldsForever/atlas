"""Atlas app icon: a dark tactical MAP — topographic contours + a glowing plotted route ending in a
location pin (on-brand for a map viewer with routing). Drawn 4x-supersampled with PIL, downscaled LANCZOS,
emitted as a multi-size .ico + preview PNGs."""
import math, random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

S = 512            # final master
SS = 4             # supersample factor
W = S * SS

# --- palette (from viewer/src/ui_theme.rs) ---
BG_C  = (17, 20, 18)      # center of the bg gradient (slight green-charcoal)
BG_E  = (7, 8, 7)         # edge / near-black
BEIGE = (199, 178, 153)   # app accent
BONE  = (212, 208, 196)
MUTED = (110, 106, 97)
BORDER= (58, 56, 50)
AMBER = (235, 196, 128)   # warm pin glow

def with_a(c, a): return (c[0], c[1], c[2], a)

# --- rounded-square dark radial-gradient background ---
radius = int(W * 0.225)
yy, xx = np.mgrid[0:W, 0:W]
cx, cy = W * 0.44, W * 0.40
r = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (W * 0.78), 0, 1) ** 1.15
grad = np.zeros((W, W, 3), np.uint8)
for i in range(3):
    grad[..., i] = (BG_C[i] + (BG_E[i] - BG_C[i]) * r).astype(np.uint8)
bg = Image.fromarray(np.dstack([grad, np.full((W, W), 255, np.uint8)]), "RGBA")
mask = Image.new("L", (W, W), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, W - 1], radius=radius, fill=255)
bg.putalpha(mask)
canvas = bg

def layer():
    return Image.new("RGBA", (W, W), (0, 0, 0, 0))

# --- faint graticule grid (map feel), clipped to the rounded square ---
gl = layer(); gd = ImageDraw.Draw(gl)
step = W // 8
for i in range(1, 8):
    gd.line([(i * step, 0), (i * step, W)], fill=with_a(MUTED, 16), width=1 * SS)
    gd.line([(0, i * step), (W, i * step)], fill=with_a(MUTED, 16), width=1 * SS)
gl.putalpha(Image.composite(gl.getchannel("A"), Image.new("L", (W, W), 0), mask))
canvas = Image.alpha_composite(canvas, gl)

# --- topographic contour lines (nested, irregular closed loops) ---
cl = layer(); cd = ImageDraw.Draw(cl)
random.seed(11)
def contour(ox, oy, rad, jitter, alpha, width):
    pts = []; n = 96
    ph = [random.uniform(0, math.tau) for _ in range(4)]
    for k in range(n + 1):
        a = math.tau * k / n
        rr = rad * (1 + jitter * (0.5 * math.sin(3 * a + ph[0]) + 0.3 * math.sin(2 * a + ph[1]) + 0.2 * math.sin(5 * a + ph[2])))
        pts.append((ox + rr * math.cos(a), oy + rr * math.sin(a)))
    cd.line(pts, fill=with_a(BEIGE, alpha), width=width, joint="curve")
for j, rad in enumerate((0.11, 0.17, 0.24, 0.32, 0.41, 0.51)):
    contour(W * 0.42, W * 0.46, W * rad, 0.12, 44 - j * 4, 2 * SS)  # inner rings brighter = elevation depth
cl.putalpha(Image.composite(cl.getchannel("A"), Image.new("L", (W, W), 0), mask))
canvas = Image.alpha_composite(canvas, cl)

# --- the plotted ROUTE (hero): smooth path with waypoint nodes -> location pin ---
route = [(0.17, 0.74), (0.30, 0.585), (0.455, 0.63), (0.575, 0.45), (0.66, 0.285)]
route = [(x * W, y * W) for x, y in route]
def catmull(pts, samples=24):
    P = [pts[0]] + pts + [pts[-1]]; out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for s in range(samples + 1):
            t = s / samples; t2 = t * t; t3 = t2 * t
            x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    return out
rp = catmull(route)

# soft glow beneath the route
gw = layer(); ImageDraw.Draw(gw).line(rp, fill=with_a(BEIGE, 150), width=9 * SS, joint="curve")
canvas = Image.alpha_composite(canvas, gw.filter(ImageFilter.GaussianBlur(10 * SS)))
# route stroke
rd = ImageDraw.Draw(canvas)
rd.line(rp, fill=with_a(BONE, 255), width=int(5.0 * SS), joint="curve")
# waypoint nodes (hollow)
for (x, y) in route[:-1]:
    rr = 7 * SS
    rd.ellipse([x - rr, y - rr, x + rr, y + rr], fill=with_a(BG_E, 255), outline=with_a(BEIGE, 255), width=2 * SS)

# --- destination location PIN (teardrop) with amber glow ---
px, py = route[-1]
halo = layer(); ImageDraw.Draw(halo).ellipse([px - 46 * SS, py - 46 * SS, px + 46 * SS, py + 46 * SS], fill=with_a(AMBER, 130))
canvas = Image.alpha_composite(canvas, halo.filter(ImageFilter.GaussianBlur(16 * SS)))
pd = ImageDraw.Draw(canvas)
# teardrop = circle head + triangle tip pointing down to (px,py)
head_cy = py - 30 * SS; head_r = 26 * SS
pd.polygon([(px - head_r * 0.72, head_cy + head_r * 0.55), (px + head_r * 0.72, head_cy + head_r * 0.55), (px, py + 4 * SS)], fill=with_a(AMBER, 255))
pd.ellipse([px - head_r, head_cy - head_r, px + head_r, head_cy + head_r], fill=with_a(AMBER, 255))
pd.ellipse([px - 10 * SS, head_cy - 10 * SS, px + 10 * SS, head_cy + 10 * SS], fill=with_a(BG_C, 255))

# --- compass 'N' tick, top-right (subtle map mark) ---
nx, ny = W * 0.78, W * 0.20
cdt = ImageDraw.Draw(canvas)
cdt.line([(nx, ny + 22 * SS), (nx, ny - 22 * SS)], fill=with_a(MUTED, 150), width=2 * SS)
cdt.polygon([(nx, ny - 30 * SS), (nx - 8 * SS, ny - 14 * SS), (nx + 8 * SS, ny - 14 * SS)], fill=with_a(BEIGE, 200))

# --- inner rim / vignette for depth ---
rim = layer(); ImageDraw.Draw(rim).rounded_rectangle([2 * SS, 2 * SS, W - 1 - 2 * SS, W - 1 - 2 * SS], radius=radius - 2 * SS, outline=with_a(BORDER, 200), width=3 * SS)
rim.putalpha(Image.composite(rim.getchannel("A"), Image.new("L", (W, W), 0), mask))
canvas = Image.alpha_composite(canvas, rim)
# subtle top highlight
hl = layer(); ImageDraw.Draw(hl).rounded_rectangle([4 * SS, 4 * SS, W - 5 * SS, W - 5 * SS], radius=radius - 3 * SS, outline=with_a((255, 255, 255), 14), width=2 * SS)
hl.putalpha(Image.composite(hl.getchannel("A"), Image.new("L", (W, W), 0), mask))
canvas = Image.alpha_composite(canvas, hl)

# --- output ---
final = canvas.resize((S, S), Image.LANCZOS)
final.save("atlas_icon.png")
final.resize((128, 128), Image.LANCZOS).save("atlas_icon_128.png")
final.resize((32, 32), Image.LANCZOS).save("atlas_icon_32.png")
# contact sheet of small sizes for legibility check
sheet = Image.new("RGBA", (16 + 256 + 128 + 64 + 48 + 32 + 16 + 7 * 16, 256 + 32), (30, 30, 30, 255))
x = 16
for sz in (256, 128, 64, 48, 32, 16):
    sheet.alpha_composite(final.resize((sz, sz), Image.LANCZOS), (x, 16))
    x += sz + 16
sheet.save("atlas_icon_sheet.png")
final.save("atlas.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("wrote atlas_icon.png, atlas.ico (multi-size), preview sheet")
