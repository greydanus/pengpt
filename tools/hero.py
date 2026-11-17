import argparse
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path[:0] = ["inference", "tools"]
import figures
import pen as mini

INK, MUTED = "#0b0b0b", "#555555"


def phrase(ax, words, W=3.0, H=0.7, fill=0.9):
    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    if not any(len(w) for w in words):
        return
    points = np.concatenate(mini.layout(words, line_width=99))
    down = points[points[:, 2] == 1, :2]
    lo, size = down.min(0), np.ptp(down, 0)
    s = min((0.42 * W * H / (size[0] * size[1])) ** 0.5, fill * W / size[0], fill * H / size[1])
    for a, b in mini.strokes(points):
        q = (points[a:b, :2] - lo - size / 2) * s
        q = q if b - a > 1 else np.r_[q, q + 1e-3]
        ax.plot(W / 2 + q[:, 0], H / 2 - q[:, 1], color=INK, lw=1.0, solid_capstyle="round", solid_joinstyle="round")


def heading(fig, x, y, text):
    fig.text(x, y, text, fontsize=11.5, color=MUTED, style="italic", va="center")


def main():
    p = argparse.ArgumentParser(description="the README's first picture: one phrase in 21 hands held out of training, and twelve drawings, "
                                            "laid out from the samples the paper's first figure keeps")
    p.add_argument("--hands", default="paper/figures/unseen_hands.json")
    p.add_argument("--sketches", default="paper/figures/samples_sketches.json")
    p.add_argument("--out", default="static/hero.png")
    args = p.parse_args()
    written = [[np.array(w, float).reshape(-1, 3) for w in cell["words"] if len(w)] for cell in json.load(open(args.hands))]
    kept = json.load(open(args.sketches))
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    left, right = fig.add_gridspec(1, 2, wspace=0.05, left=0.01, right=0.99, top=0.93, bottom=0.075)
    n = len(written) // 3
    cells = left.subgridspec(n, 3, wspace=0, hspace=0)
    heading(fig, 0.012, 0.965, f"Handwriting: {len(written)} writers the model never trained on")
    for k, words in enumerate(written):
        phrase(fig.add_subplot(cells[k % n, k // n]), words)
    rows = right.subgridspec(4, 4, height_ratios=[1, 0.22, 1, 1], wspace=0.08, hspace=0.5)
    heading(fig, 0.512, 0.965, "Drawings: things it was trained to draw")
    ax = fig.add_subplot(rows[1, :])
    ax.axis("off")
    ax.text(0.0, 0.35, "Drawings: combinations it was never shown", fontsize=11.5, color=MUTED, style="italic", va="center")
    for row, group in ((0, kept["known"]), (2, kept["unseen"])):
        for k, cell in enumerate(group):
            ax = fig.add_subplot(rows[row + k // 4, k % 4])
            ax.axis("off")
            figures.ink(ax, figures.points(cell["drawings"][cell["pick"]]), 1.1, 0.02)
            ax.text(0.5, -0.05, textwrap.fill(cell["text"], 19), transform=ax.transAxes, fontsize=9, color=MUTED, ha="center", va="top",
                    linespacing=1.05)
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
