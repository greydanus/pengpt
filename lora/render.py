"""Render pen.py's SVGs (polyline paths) to PNG, stacked with labels: python lora/render.py out.png a.svg:label b.svg:label ..."""
import re, sys
from PIL import Image, ImageDraw, ImageFont

def load(path, scale=3):
    s = open(path).read()
    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s).groups())
    img = Image.new("RGB", (int(w * scale), int(h * scale)), "white")
    d = ImageDraw.Draw(img)
    for p in re.findall(r'<path d="([^"]+)"', s):
        pts = [(float(x) * scale, float(y) * scale) for x, y in re.findall(r"([\d.]+) ([\d.]+)", p)]
        if len(pts) == 1 or len(set(pts)) == 1:
            x, y = pts[0]; d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(26, 35, 126))
        else:
            d.line(pts, fill=(26, 35, 126), width=int(2.2 * scale), joint="curve")
    return img

out, items = sys.argv[1], [a.rsplit(":", 1) for a in sys.argv[2:]]
imgs = [(load(p), lab) for p, lab in items]
font = ImageFont.load_default(28)
W, H = max(i.width for i, _ in imgs), sum(i.height + 40 for i, _ in imgs)
sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet); y = 0
for im, lab in imgs:
    d.text((10, y + 4), lab, fill=(120, 120, 120), font=font); sheet.paste(im, (0, y + 40)); y += im.height + 40
sheet.save(out); print(out)
