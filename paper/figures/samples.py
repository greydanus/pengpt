import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path[:0] = ["inference", "tools"]
import figures
import pen as mini
import styles

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"]})
INK, MUTED = "#0b0b0b", "#444444"
COLUMNS, ROWS = (0, 2, 5), (0, 2, 3, 4, 5, 6, 7)
KNOWN = [("a cat", 6), ("a house", 8), ("a sunflower", 7), ("a tennis racket", 8)]
UNSEEN = ["an elephant with wings", "a snail on wheels", "a pegasus", "a penguin wearing a top hat",
          "a snowman with antlers", "a shoe house", "a whale on wheels", "a pumpkin with cat ears"]


def hands():
    grid = [row.split() for row in styles.HANDS.split("\n") if row]
    return [([grid[r][c] for c in COLUMNS], styles.TEMPERATURES[r]) for r in ROWS]


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
        ax.plot(W / 2 + q[:, 0], H / 2 - q[:, 1], color=INK, lw=0.55, solid_capstyle="round", solid_joinstyle="round")


def handwriting(fig, box, written, title):
    n = len(written) // 3
    cells = box.subgridspec(n + 1, 3, height_ratios=[0.75] + [1] * n, wspace=0, hspace=0)
    ax = fig.add_subplot(cells[0, :])
    ax.axis("off")
    ax.text(0.0, 0.5, title, fontsize=8, color=MUTED, style="italic", va="center")
    for k, words in enumerate(written):
        phrase(fig.add_subplot(cells[1 + k % n, k // n]), words)


def trained(pen, text):
    rows = hands()
    return [pen.write(text, row[c].split(":")[0], temperature=t, seed=int(row[c].split(":")[1])) for c in range(3) for row, t in rows]


def unseen(path="paper/figures/unseen_hands.json"):
    return [[np.array(w, float).reshape(-1, 3) for w in cell["words"] if len(w)] for cell in json.load(open(path))]


def drawings(fig, box, static="static"):
    kept = "paper/figures/samples_sketches.json"
    if not os.path.exists(kept) or "--fresh" in sys.argv:
        ink = {c["text"]: [] for c in json.load(open(f"{static}/fidelity_ink.json"))}
        for c in json.load(open(f"{static}/fidelity_ink.json")):
            ink[c["text"]].append(c)
        ood = {c["text"]: c for c in json.load(open(f"{static}/ood.json"))}
        only = lambda c: dict(text=c["text"], pick=0, drawings=[c["drawings"][c["pick"]]])
        json.dump(dict(known=[only(ink[t][k]) for t, k in KNOWN], unseen=[only(ood[t]) for t in UNSEEN]), open(kept, "w"))
    kept = json.load(open(kept))
    groups = (("PenGPT sketch model (simple captions)", kept["known"]), ("PenGPT sketch model (compositional held-out captions)", kept["unseen"]))
    rows = box.subgridspec(5, 4, height_ratios=[0.16, 1, 0.16, 1, 1], wspace=0.08, hspace=0.52)
    for g, (note, cells) in enumerate(groups):
        ax = fig.add_subplot(rows[2 * g, :])
        ax.axis("off")
        ax.text(0.0, 0.3, note, fontsize=8, color=MUTED, style="italic", va="center")
        for k, cell in enumerate(cells):
            ax = fig.add_subplot(rows[2 * g + 1 + k // 4, k % 4])
            ax.axis("off")
            figures.ink(ax, figures.points(cell["drawings"][cell["pick"]]), 0.6, 0.02)
            ax.text(0.5, -0.04, textwrap.fill(cell["text"], 17), transform=ax.transAxes, fontsize=6.3, color=MUTED, ha="center", va="top",
                    linespacing=1.05)


def main(out="paper/figures/samples.pdf", hand="hand.npz", text="Written by PenGPT", held_out=False):
    fig = plt.figure(figsize=(3.3, 5.1))
    top, bottom = fig.add_gridspec(2, 1, height_ratios=[1.88, 3.15], hspace=0.04, left=0.0, right=1.0, top=1.0, bottom=0.045)
    if held_out:
        handwriting(fig, top, unseen(), "PenGPT handwriting model (held-out author styles)")
    else:
        handwriting(fig, top, trained(mini.Pen(hand), text), "PenGPT handwriting model (21 of its 1,192 author styles)")
    drawings(fig, bottom)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(out)


if __name__ == "__main__":
    main(held_out="--trained" not in sys.argv, out="paper/figures/samples_trained.pdf" if "--trained" in sys.argv else "paper/figures/samples.pdf")
