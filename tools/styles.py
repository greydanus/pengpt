import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "inference")
import pen as mini

TEMPERATURES = [0, 0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
HANDS = """
tbgxs:0 sq7uc:0 8auhv:0 ut98b:0 jp86a:0 6uh8z:0
m9ul3:0 f6he9:0 b09qm:0 x011f:0 q6afc:0 oi5r0:0
6ua43:1 ekavj:1 js8ik:2 m85xs:3 gat3e:2 p8m7r:1
f912z:1 245ak:4 9vutm:1 xq3p1:1 3is8l:2 ii45a:3
4axot:1 e3uew:2 qt7yy:1 90zp0:4 y81ha:4 gzjyt:2
vox5i:1 uypxa:2 ydv1l:2 9bbpe:4 xtvxy:2 47erk:2
v14c0:3 m2ui6:3 uvud3:2 pa3ub:4 fuf1r:1 kon7q:1
m6a7q:4 1ywyk:2 5ta4r:1 gvrbu:3 8u952:2 uflhz:2
"""


def main():
    p = argparse.ArgumentParser(description="one phrase in 48 hands: print to cursive across, neat to messy down")
    p.add_argument("--hand", default="hand.npz")
    p.add_argument("--text", default="Written by PenGPT")
    p.add_argument("--out", default="static/styles.png")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    pen, grid = mini.Pen(args.hand), [row.split() for row in HANDS.split("\n") if row]
    rows, cols, W, H, fill = len(grid), len(grid[0]), 3.0, 0.78, 0.84
    fig, axes = plt.subplots(rows, cols, figsize=(W * cols, H * rows), squeeze=False)
    fig.patch.set_facecolor("white")
    for r, (row, t) in enumerate(zip(grid, TEMPERATURES)):
        for c, cell in enumerate(row):
            writer, seed = cell.split(":")
            ax = axes[r][c]
            ax.axis("off")
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect("equal")
            words = pen.write(args.text, writer, temperature=t, seed=int(seed))
            if not any(len(w) for w in words):
                continue
            points = np.concatenate(mini.layout(words, line_width=99))
            down = points[points[:, 2] == 1, :2]
            lo, size = down.min(0), np.ptp(down, 0)
            s = min((0.42 * W * H / (size[0] * size[1])) ** 0.5, fill * W / size[0], fill * H / size[1])
            for a, b in mini.strokes(points):
                q = (points[a:b, :2] - lo - size / 2) * s
                q = q if b - a > 1 else np.r_[q, q + 1e-3]
                ax.plot(W / 2 + q[:, 0], H / 2 - q[:, 1], color="#0b0b0b", lw=1.25, solid_capstyle="round", solid_joinstyle="round")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.93, bottom=0.005, wspace=0.0, hspace=0.0)
    fig.text(0.515, 0.965, "print  →  cursive", ha="center", va="center", fontsize=13, color="0.25")
    fig.text(0.012, 0.47, "messy  ←  neat", rotation=90, ha="center", va="center", fontsize=13, color="0.25")
    fig.savefig(args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
