import argparse
import json
import os
import string
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + "\"'(),.?!"
SHORT = "acemnorsuvwxz"
XH = 120
NB8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def thin(img):
    img = np.pad(img.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, -dy, 0), -dx, 1) for dy, dx in NB8]
            b = sum(p)
            a = sum((p[i] == 0) & (p[(i + 1) % 8] == 1) for i in range(8))
            c1 = p[0] * p[2] * p[4] if step == 0 else p[0] * p[2] * p[6]
            c2 = p[2] * p[4] * p[6] if step == 0 else p[0] * p[4] * p[6]
            kill = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & (c1 == 0) & (c2 == 0)
            if kill.any():
                img[kill], changed = 0, True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def crossings(sk):
    p = [np.roll(np.roll(np.pad(sk, 1), -dy, 0), -dx, 1)[1:-1, 1:-1] for dy, dx in NB8]
    return sum(~p[i] & p[(i + 1) % 8] for i in range(8)) * sk


def neighbors(sk, p):
    h, w = sk.shape
    return [(p[0] + dy, p[1] + dx) for dy, dx in NB8 if 0 <= p[0] + dy < h and 0 <= p[1] + dx < w and sk[p[0] + dy, p[1] + dx]]


def prune(sk, length, thick):
    sk = sk.copy()
    for _ in range(3):
        kind, removed = crossings(sk), False
        for y, x in np.argwhere(kind == 1):
            path, prev, cur = [(y, x)], None, (y, x)
            while len(path) <= 6 * length:
                nb = [q for q in neighbors(sk, cur) if q != prev and q not in path]
                if len(nb) != 1:
                    break
                prev, cur = cur, nb[0]
                if kind[cur] >= 3:
                    break
                path.append(cur)
            if kind[cur] >= 3 and len(path) <= max(length, 1.3 * thick[cur]):
                for q in path:
                    sk[q] = False
                removed = True
        if not removed:
            return sk
    return sk


def walk(sk, start, first, node, nid, visited):
    path, prev, cur = [start, first], start, first
    visited[first] = True
    while True:
        cand = [r for r in neighbors(sk, cur) if r != prev and r not in path[-3:]]
        hit = [r for r in cand if node[r] and (nid[r] != nid[start] or len(path) > 4)]
        if hit:
            return path + [hit[0]]
        cand = sorted((r for r in cand if not visited[r] and not node[r]), key=lambda r: abs(r[0] - cur[0]) + abs(r[1] - cur[1]))
        if not cand:
            return path
        prev, cur = cur, cand[0]
        visited[cur] = True
        path.append(cur)


def edges(sk):
    kind = crossings(sk)
    node = sk & (kind != 2)
    nid, _ = ndimage.label(node, np.ones((3, 3)))
    visited, out = np.zeros_like(sk), []
    for p in map(tuple, np.argwhere(node)):
        for q in neighbors(sk, p):
            if not node[q] and not visited[q]:
                out.append(walk(sk, p, q, node, nid, visited))
    loops, n = ndimage.label(sk & ~node & ~visited, np.ones((3, 3)))
    for k in range(1, n + 1):
        pts = np.argwhere(loops == k)
        start = tuple(pts[pts[:, 0].argmin()])
        nb = neighbors(sk, start)
        path = walk(sk, start, nb[0], node, nid, visited) if nb else [start]
        out.append(path + [path[0]])
    touched = {nid[e[0]] for e in out} | {nid[e[-1]] for e in out}
    for k in set(range(1, nid.max() + 1)) - touched:
        out.append([tuple(np.round(np.argwhere(nid == k).mean(0)).astype(int))])
    return out, nid


def strokes_of(sk, bridge=0, tangent_len=25):
    es, nid = edges(sk)
    closed = [len(e) > 2 and e[0] == e[-1] and not nid[e[0]] for e in es]
    count = {}
    for i, e in enumerate(es):
        if len(e) > 1 and not closed[i]:
            for p in (e[0], e[-1]):
                count[nid[p]] = count.get(nid[p], 0) + 1
    root = {}
    find = lambda k: k if root.get(k, k) == k else find(root[k])
    skip = set()
    for i, e in enumerate(es):
        a, b = nid[e[0]], nid[e[-1]]
        if len(e) > 1 and not closed[i] and a and b and count[a] >= 3 and count[b] >= 3 and len(e) <= bridge:
            if find(a) != find(b):
                root[find(a)] = find(b)
            skip.add(i)
    ends = {}
    for i, e in enumerate(es):
        if len(e) > 1 and not closed[i] and i not in skip:
            ends.setdefault(find(nid[e[0]]), []).append((i, 0))
            ends.setdefault(find(nid[e[-1]]), []).append((i, 1))

    def tangent(i, side):
        pts = np.asarray(es[i] if side == 0 else es[i][::-1], float)
        v = pts[min(len(pts) - 1, tangent_len)] - pts[0]
        return v / (np.linalg.norm(v) + 1e-9)

    link = {}
    for node, group in ends.items():
        if node == 0:
            continue
        for d, a, b in sorted((tangent(*a) @ tangent(*b), a, b) for a, b in combinations(group, 2)):
            if d > -0.5:
                break
            if a not in link and b not in link:
                link[a], link[b] = b, a
    done, out = set(skip), []

    def run(i, side):
        pts = []
        while True:
            done.add(i)
            seg = es[i] if side == 0 else es[i][::-1]
            pts += seg if not pts else seg[1:]
            if (i, 1 - side) not in link:
                return pts, False
            i, side = link[(i, 1 - side)]
            if i in done:
                return pts, True
    for i, side in [(i, s) for i, e in enumerate(es) if len(e) > 1 and not closed[i] for s in (0, 1)]:
        if i not in done and (i, side) not in link:
            out.append(run(i, side))
    for i, e in enumerate(es):
        if i not in done:
            done.add(i)
            out.append(run(i, 0) if len(e) > 1 and not closed[i] else (e, len(e) > 1))
    return [(np.asarray(p, float)[:, ::-1], c) for p, c in out]


def orient(pts, closed):
    if len(pts) < 2:
        return pts
    if closed:
        pts = pts[:-1]
        area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        pts = pts[::-1] if area > 0 else pts
        top = np.argmin(pts[:, 1])
        return np.vstack([np.roll(pts, -top, 0), np.roll(pts, -top, 0)[:1]])
    lead = lambda p: p[1] + 0.5 * p[0]
    return pts[::-1] if lead(pts[-1]) < lead(pts[0]) else pts


def dot(p, thick, n=10):
    r = max(1.5, 0.6 * thick[int(round(p[1])), int(round(p[0]))])
    a = np.pi / 2 + np.linspace(0, 2 * np.pi, n + 1)
    return np.column_stack([p[0] + r * np.cos(a), p[1] - r * np.sin(a)])


def delayed(s, others):
    ext = np.ptp(s, 0) if len(s) > 1 else np.zeros(2)
    if ext.max() < 0.2 * XH:
        return True
    if ext[0] > 2 * ext[1] and len(s) > 4:
        mid = s[len(s) // 5: -len(s) // 5 or None]
        return any(len(o) > 1 and np.min(np.hypot(*(mid[:, None] - o[None]).transpose(2, 0, 1))) < 0.06 * XH for o in others)
    return False


def smooth(pts, w=5):
    if len(pts) <= 4 * w:
        return pts
    k = np.ones(w) / w
    out = pts.copy()
    out[w // 2: -(w // 2)] = np.column_stack([np.convolve(pts[:, i], k, "valid") for i in (0, 1)])
    return np.vstack([out[::3], out[-1:]]) if len(out) % 3 != 1 else out[::3]


def font_at_xheight(path, xh=XH):
    probe = ImageFont.truetype(path, 100)
    h = np.median([np.ptp(probe.getbbox(c, anchor="ls")[1::2]) for c in SHORT])
    return ImageFont.truetype(path, max(8, int(round(100 * xh / h))))


def glyph_strokes(font, c, fixes=None):
    l, t, r, b = font.getbbox(c, anchor="ls")
    pad = 20
    img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 255)
    ox, oy = pad - l, pad - t
    ImageDraw.Draw(img).text((ox, oy), c, font=font, fill=0, anchor="ls")
    ink = np.asarray(img) < 128
    if not ink.any():
        return [], img, (ox, oy)
    thick = ndimage.distance_transform_edt(ink)
    width = 2 * np.median(thick[ink])
    sk = prune(thin(ink), int(1.5 * width), thick)
    raw = [orient(p, closed) if len(p) > 1 else dot(p[0], thick) for p, closed in strokes_of(sk, int(6 * width + 4))]
    raw = [s for s in raw if len(s) > 2 or np.ptp(s, 0).max() > 0.08 * XH]
    late = [delayed(s, [o for o in raw if o is not s]) for s in raw]
    order = sorted(range(len(raw)), key=lambda i: (late[i], raw[i][0][0]))
    out = [smooth(raw[i]) - [ox, oy] for i in order]
    for fix in (fixes or {}).get(c, []):
        op, k = fix.split(":") if ":" in fix else (fix, "0")
        k = int(k)
        if op == "reverse":
            out[k] = out[k][::-1]
        elif op == "first":
            out.insert(0, out.pop(k))
        elif op == "last":
            out.append(out.pop(k))
        elif op == "drop":
            out.pop(k)
    return out, img, (ox, oy)


def load_fixes(path):
    return json.load(open(path)) if path and os.path.exists(path) else {}


def compose(font, glyphs, text, join=0.06 * XH):
    out = []
    for i, c in enumerate(text):
        x = font.getlength(text[:i + 1]) - font.getlength(c)
        for s in glyphs.get(c, []):
            s = s + [x, 0]
            if out and len(s) > 1 and len(out[-1]) > 1 and np.hypot(*(out[-1][-1] - s[0])) < join:
                out[-1] = np.vstack([out[-1], s[1:]])
            else:
                out.append(s)
    return out


def sheet_command(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font, fixes = font_at_xheight(args.font), load_fixes(args.fixes)
    chars = args.chars or CHARS
    cols = 13
    fig, axes = plt.subplots((len(chars) + cols - 1) // cols, cols, figsize=(cols * 1.5, 1.9 * ((len(chars) + cols - 1) // cols)))
    for ax in axes.flat:
        ax.axis("off")
    for ax, c in zip(axes.flat, chars):
        strokes, img, (ox, oy) = glyph_strokes(font, c, fixes)
        ax.imshow(img, cmap="gray", alpha=0.18, extent=(-ox, img.size[0] - ox, img.size[1] - oy, -oy))
        colors = plt.cm.plasma(np.linspace(0, 0.85, max(1, len(strokes))))
        for k, (s, col) in enumerate(zip(strokes, colors)):
            ax.plot(s[:, 0], s[:, 1], color=col, lw=1.6)
            ax.plot(*s[0], "o", color=col, ms=4)
            if len(s) > 3:
                m = len(s) * 3 // 5
                ax.annotate("", s[min(m + 1, len(s) - 1)], s[m - 1], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=11))
            ax.text(s[0][0] - 6, s[0][1] - 6, str(k), fontsize=7, color=col, ha="right")
        ax.set_title(f"{c}  ({len(strokes)})", fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle(f"{os.path.basename(args.font)}: strokes numbered in drawing order, dot = start, arrow = direction", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=90)
    print(f"sheet: {args.out}")


def words_command(args):
    font, fixes = font_at_xheight(args.font), load_fixes(args.fixes)
    glyphs = {c: glyph_strokes(font, c, fixes)[0] for c in CHARS}
    rows = []
    for text in (w.strip() for w in open(args.words) if w.strip()):
        strokes = compose(font, glyphs, text)
        rows.append({"text": text, "strokes": [s.round(2).tolist() for s in strokes], "baseline": 0.0, "style": args.style,
                     "writer": args.style, "source": os.path.basename(args.font), "license": args.license})
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"wrote {len(rows)} words from glyph strokes to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="pen strokes straight from a font's glyphs: skeleton, stroke order, word assembly")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sheet", help="every glyph's strokes, numbered and arrowed, for checking by eye")
    p.add_argument("--font", required=True)
    p.add_argument("--fixes", default="")
    p.add_argument("--chars", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("words", help="assemble words from glyph strokes, as rows like tools/ink.py qc writes")
    p.add_argument("--font", required=True)
    p.add_argument("--fixes", default="")
    p.add_argument("--words", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--license", required=True)
    args = parser.parse_args()
    {"sheet": sheet_command, "words": words_command}[args.command](args)


if __name__ == "__main__":
    main()
