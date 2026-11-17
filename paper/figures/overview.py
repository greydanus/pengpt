import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, ".")
import pengpt as P
from strokes import Strokes, to_strokes

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 11})
INK, SOFT, WARM, COOL, GREEN, LILAC = "#111111", "#f4f1ea", "#f6d9a8", "#bfe0f2", "#cfe8c8", "#dcd3ee"
TOKENS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2"]


def word(path, row):
    return P.to_points(Strokes(path).strokes(row)).astype(float)


def draw(ax, points, color=INK, lw=1.6, **kw):
    for s in to_strokes(points):
        ax.plot(s[:, 0], -s[:, 1], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", **kw)


def clean(ax, title=""):
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=12.5, pad=5)


def owners(tok, tokens):
    down, out = False, []
    for k, t in enumerate(tokens):
        for b in tok.expand([t]):
            if b == P.DOWN:
                down = True
                out.append(k)
            elif b == P.UP:
                out += [k] * down
                down = False
            elif down:
                out.append(k)
    return np.array(out)


def box(ax, x, y, w, h, text, color, size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.025", fc=color, ec="#444444", lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333", shrinkA=0, shrinkB=0, **kw))


SOURCE, EARLIER, SHOWN = "data/loc.hand.parquet", (9674, 9641), 9725


CACHE = "paper/figures/overview.json"


def make(checkpoint="out/ablate_hand/base/best.pt"):
    tok = P.load_for_sampling(checkpoint, "cpu")[1].stroke_tok
    ink = word(SOURCE, SHOWN)
    ink[:, :2] -= ink[:, :2].min(0)
    tokens, base, rng = tok.encode_word(ink), tok.encode_base(ink), np.random.default_rng(4)
    walks = {k: np.cumsum([[0, 0]] + [P.DIRECTIONS[m] for m in tok.expand([t]) if 0 <= m < 8], 0) * [1, -1] for k, t in enumerate(tokens)}
    bent = [P.augment(ink, P.DataConfig(augment=1.6, rotate=4, stretch=0.2, shear=0.25), rng, False) for _ in range(4)]
    kept = dict(ink=ink, raw=tok.decode_word(base, smooth=False), smooth=tok.decode_word(tokens), own=owners(tok, tokens), grid=tok.grid,
                moves=len(base), tokens=len(tokens), walks={str(k): walks[k] for k in sorted(sorted(walks, key=lambda k: -len(walks[k]))[:6])},
                bent=bent, hand=[word(SOURCE, row) for row in (*EARLIER, SHOWN)])
    plain = lambda v: v.tolist() if hasattr(v, "tolist") else {k: plain(x) for k, x in v.items()} if isinstance(v, dict) else [plain(x) for x in v] if isinstance(v, list) else v
    json.dump(plain(kept), open(CACHE, "w"))


def main(out="paper/figures/overview.pdf", at=0.30):
    if not os.path.exists(CACHE) or "--fresh" in sys.argv:
        make()
    K = json.load(open(CACHE))
    ink, raw, smooth, own = (np.array(K[k], float) for k in ("ink", "raw", "smooth", "own"))
    own = own.astype(int)

    fig = plt.figure(figsize=(11.0, 3.5))
    grid = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.05, 1.0, 1.15, 2.5], height_ratios=[0.42, 1], wspace=0.12, hspace=0.25,
                            left=0.01, right=0.995, top=0.90, bottom=0.14)

    ax = fig.add_subplot(grid[0, 0])
    draw(ax, ink, lw=2.0)
    clean(ax, "a)  Pen strokes")
    g, span = K["grid"], 3.6
    near = raw[int((at - 0.07) * len(raw)):int((at + 0.07) * len(raw)), :2]
    centre, r = np.rint((near.min(0) + near.max(0)) / 2 / g) * g * [1, -1], span * g
    ax.add_patch(Rectangle(centre - r, 2 * r, 2 * r, fc="none", ec=TOKENS[1], lw=0.9, zorder=5))

    zoom = fig.add_subplot(grid[1, 0])
    for k in np.arange(-5, 6):
        zoom.axhline(centre[1] + k * g, color="#cccccc", lw=0.5, zorder=0)
        zoom.axvline(centre[0] + k * g, color="#cccccc", lw=0.5, zorder=0)
    draw(zoom, ink, color="#bbbbbb", lw=5)
    walk = raw[:, :2] * [1, -1]
    for a, b, pa, pb in zip(walk[:-1], walk[1:], raw[:-1, 2], raw[1:, 2]):
        if pa == 1 and pb == 1 and max(np.abs(a - centre).max(), np.abs(b - centre).max()) < span * g:
            zoom.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=TOKENS[0], lw=1.2, mutation_scale=7, shrinkA=0, shrinkB=0))
            zoom.plot(*a, "o", color=INK, ms=2.2)
    zoom.set_xlim(centre[0] - span * g, centre[0] + span * g)
    zoom.set_ylim(centre[1] - span * g, centre[1] + span * g)
    clean(zoom)
    zoom.text(0.0, -0.08, "b)  A walk on a grid", transform=zoom.transAxes, ha="left", va="top", fontsize=12.5)
    zoom.text(0.5, -0.24, "8 directions,\nDOWN and UP", transform=zoom.transAxes, ha="center", va="top", fontsize=10.5)
    zoom.add_patch(Rectangle((0, 0), 1, 1, transform=zoom.transAxes, fc="none", ec=TOKENS[1], lw=1.2, clip_on=False))
    for side in (-1, 1):
        fig.add_artist(ConnectionPatch((centre[0] + side * r, centre[1] - r), ((side + 1) / 2, 1.0), "data", "axes fraction",
                                       axesA=ax, axesB=zoom, color=TOKENS[1], lw=0.7))

    col = grid[:, 1].subgridspec(4, 3, height_ratios=[1.0, 0.6, 1, 1], wspace=0.1, hspace=0.1)
    ax = fig.add_subplot(col[0, :])
    for a, b, k in zip(smooth[:-1], smooth[1:], own[1:]):
        if a[2] == 1:
            ax.plot([a[0], b[0]], [-a[1], -b[1]], color=TOKENS[k % len(TOKENS)], lw=2.4, solid_capstyle="butt")
    clean(ax, "c)  BPE merges")
    note = fig.add_subplot(col[1, :])
    note.axis("off")
    note.text(0.5, 0.95, f"{K['moves']} moves $\\rightarrow$ {K['tokens']} tokens\nSix of them:", ha="center", va="top", fontsize=10.5)
    walks = {int(k): np.array(v, float) for k, v in K["walks"].items()}
    longest = sorted(walks)
    reach = max(np.ptp(walks[k], 0).max() for k in longest) / 2 + 1
    for n, k in enumerate(longest):
        ax = fig.add_subplot(col[2 + n // 3, n % 3])
        mid = (walks[k].min(0) + walks[k].max(0)) / 2
        ax.plot(*walks[k].T, color=TOKENS[k % len(TOKENS)], lw=2.2, solid_capstyle="round", solid_joinstyle="round")
        ax.plot(*walks[k][0], "o", color=INK, ms=2.5)
        ax.set_xlim(mid[0] - reach, mid[0] + reach)
        ax.set_ylim(mid[1] - reach, mid[1] + reach)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#cccccc")

    col = grid[:, 2].subgridspec(5, 1, hspace=0.0)
    for k in range(5):
        ax = fig.add_subplot(col[k])
        bent = ink if k == 0 else np.array(K["bent"][k - 1], float)
        draw(ax, bent, color=INK if k == 0 else "#30506e", lw=1.3)
        clean(ax, "d)  Augment" if k == 0 else "")
        ax.set_anchor("N" if k == 0 else "C")
        ax.margins(0.08)

    ax = fig.add_subplot(grid[:, 3])
    y = 0.0
    for k, w in enumerate(np.array(w, float) for w in K["hand"]):
        w[:, 0] -= w[:, 0].min()
        w[:, 1] += y
        draw(ax, w, color="#999999" if k < 2 else TOKENS[1], lw=1.5)
        if k < 2:
            ax.text(w[:, 0].max() + 0.05, -(y - 0.05), "WORD", ha="left", va="center", fontsize=8.5, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.25", fc=SOFT, ec="#999999", lw=0.6))
        y += 0.33
    ax.margins(0.03)
    clean(ax, "e)  Words in one hand")
    ax.text(0.5, -0.06, "Up to four words by one\nwriter. Earlier words (grey)\nset the hand for the next (red).",
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5)

    ax = fig.add_subplot(grid[:, 4])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.12)
    ax.axis("off")
    ax.set_title("f)  Decoder with cross-attention", loc="left", fontsize=12.5, pad=5)
    ax.add_patch(FancyBboxPatch((0.40, 0.25), 0.56, 0.52, boxstyle="round,pad=0.01,rounding_size=0.03", fc=SOFT, ec="#888888", lw=0.7))
    ax.text(0.945, 0.215, "$\\times N$", fontsize=10.5, va="center", ha="right")
    box(ax, 0.44, 0.28, 0.48, 0.11, "Self-attention\n(rotary: step, pen position)", WARM, 9.5)
    box(ax, 0.44, 0.45, 0.48, 0.11, "Cross-attention", WARM, 10.5)
    box(ax, 0.44, 0.62, 0.48, 0.11, "MLP (ReLU$^2$)", COOL, 10.5)
    box(ax, 0.44, 0.83, 0.48, 0.08, "Softmax over stroke tokens", GREEN, 10.5)
    box(ax, 0.44, 0.07, 0.48, 0.10, "Token embedding\n+ pen position (sin, cos)", LILAC, 9.5)
    box(ax, 0.02, 0.45, 0.30, 0.11, "Told", LILAC, 10.5)
    for a, b in (((0.68, 0.17), (0.68, 0.28)), ((0.68, 0.39), (0.68, 0.45)), ((0.68, 0.56), (0.68, 0.62)), ((0.68, 0.73), (0.68, 0.83)),
                 ((0.32, 0.505), (0.44, 0.505)), ((0.68, 0.91), (0.68, 0.97)), ((0.68, 0.0), (0.68, 0.07))):
        arrow(ax, a, b)
    ax.text(0.68, 0.985, "next stroke token", ha="center", va="bottom", fontsize=10.5)
    ax.text(0.68, -0.01, "stroke tokens so far", ha="center", va="top", fontsize=10.5)
    ax.text(0.17, 0.40, "writer or source vector,\nthen text bytes, or a\nSigLIP 2 vector of a\ncaption or picture", ha="center", va="top",
            fontsize=9.5)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=110)
    print(out)


if __name__ == "__main__":
    main()
