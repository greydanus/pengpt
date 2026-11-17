import argparse
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "inference")
import pen as mini

UNIT = 240
INKS = np.geomspace(2.5, 18, 12)
THINGS = ["a cat", "a house", "a rabbit", "a squirrel", "a sunflower", "a tennis racket"]
UNSEEN = """a teapot with a face|a mushroom with a face|a cactus with a face|a lighthouse with a face|a pear with a face|a teapot with legs
an apple with legs|a house with legs|a cup with legs|a cactus with legs|a six-legged horse|a snail on wheels|a whale on wheels
a teapot with wings|an elephant with wings|a pegasus|a snowman with wings|a pumpkin with wings|a rabbit with wings|a jackalope
a penguin wearing a top hat|an owl wearing a top hat|a duck wearing a top hat|a pumpkin wearing a top hat|an owl wearing glasses
a snowman with antlers|a snowman with rabbit ears|a snowman with cat ears|a pumpkin with cat ears|a gnome|a mushroom house
a pumpkin house|a strawberry house|a teacup house|a shoe house|a turtle with a house on its back|a turtle with a castle on its back
a whale with a sailboat on its back|an elephant-shaped teapot|a duck-shaped teapot""".replace("\n", "|").split("|")
ASKED = {"ood": [dict(text=t) for t in UNSEEN], "fidelity_ink": [dict(text=t, ink=round(float(k), 4)) for t in THINGS for k in INKS]}
SHAPE = {"ood": (4, 10), "fidelity_ink": (6, 12)}


def ink(ax, p, lw, pad):
    for a, b in mini.strokes(p):
        s = p[a:b, :2] if b - a > 1 else np.r_[p[a:b, :2], p[a:b, :2] + 1e-3]
        ax.plot(s[:, 0], -s[:, 1], color="#0b0b0b", lw=lw, solid_capstyle="round", solid_joinstyle="round")
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    c, half = (lo + hi) / 2, max(hi - lo) / 2 * (1 + pad) + 1e-6
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(-c[1] - half, -c[1] + half)
    ax.set_aspect("equal")


def points(drawing):
    return np.array(drawing, float).reshape(-1, 3) / [UNIT, UNIT, 1]


def fresh(name, model, n, seed):
    pen, cells = mini.Pen(mini.fetch(model)), []
    for i, cell in enumerate(ASKED[name]):
        drawings = [pen.draw(cell["text"], temperature=0.4, rng=np.random.default_rng([seed, i, s]), ink=cell.get("ink")) for s in range(n)]
        cells.append(dict(cell, mirrored=False, pick=0, drawings=[np.rint(p * [UNIT, UNIT, 1]).astype(int).tolist() for p in drawings]))
        print(f"{i + 1}/{len(ASKED[name])} {cell['text']}", flush=True)
    return cells


def candidates(cells, path):
    n = max(len(c["drawings"]) for c in cells)
    fig, axes = plt.subplots(len(cells), n, figsize=(1.3 * n, 1.45 * len(cells)), squeeze=False)
    for i, (row, cell) in enumerate(zip(axes, cells)):
        for s, ax in enumerate(row):
            ax.axis("off")
            if s < len(cell["drawings"]) and len(cell["drawings"][s]):
                ink(ax, points(cell["drawings"][s]), 0.8, 0.12)
            ax.set_title(f"{i}: {cell['text']} {cell.get('ink', '')}" if s == 0 else str(s), fontsize=6, pad=1, loc="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.2 / len(cells), bottom=0.002, wspace=0.05, hspace=0.25)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def figure(name, cells, path):
    rows, cols = SHAPE[name]
    fig, axes = plt.subplots(rows, cols, figsize=(18, 9.6))
    for k, (ax, cell) in enumerate(zip(axes.ravel(), cells)):
        ax.axis("off")
        p = points(cell["drawings"][cell["pick"]])
        if name == "ood":
            ink(ax, p, 1.1, 0.04)
            ax.text(0.5, -0.03, textwrap.fill(cell["text"], 20), transform=ax.transAxes, fontsize=9.5, ha="center", va="top", color="#444", linespacing=1.15)
        else:
            ink(ax, p, 0.8, 0.12)
            if k < cols:
                ax.set_title(f"ink {cell['ink']:.1f}", fontsize=7.5, pad=14)
            if k % cols == 0:
                ax.text(-0.02, 1.0, cell["text"], transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    fig.patch.set_facecolor("white")
    if name == "ood":
        fig.subplots_adjust(left=0.008, right=0.992, top=0.992, bottom=0.05, wspace=0.1, hspace=0.24)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.05, hspace=0.22)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="two figures of the drawing model, drawn from the drawings kept beside them (static/ood.json, "
                                "static/fidelity_ink.json). --fresh draws new candidates with the model instead: look at the sheet it writes, set "
                                "each cell's pick in the json, and run again without --fresh")
    p.add_argument("name", choices=list(ASKED))
    p.add_argument("--stem", default="", help="reads STEM.json and writes STEM.png; static/NAME, or out/NAME with --fresh")
    p.add_argument("--fresh", type=int, default=0, help="draw this many candidates of each cell and write them to STEM.json and STEM_candidates.png")
    p.add_argument("--model", default="sketch.npz", help="downloaded from the GitHub release if it is not there")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    stem = args.stem or ("out/" if args.fresh else "static/") + args.name
    if args.fresh:
        cells = fresh(args.name, args.model, args.fresh, args.seed)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            f.write("[\n" + ",\n".join(json.dumps(c, separators=(",", ":")) for c in cells) + "\n]\n")
    cells = json.load(open(stem + ".json"))
    if max(len(c["drawings"]) for c in cells) > 1:
        candidates(cells, stem + "_candidates.png")
    figure(args.name, cells, stem + ".png")
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
