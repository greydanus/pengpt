import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path[:0] = ["inference", "tools", "."]
import figures
import pen as mini
from strokes import Strokes, held_out

SOURCES = ["quickdraw", "tuberlin", "sketchy", "things", "seva", "controlsketch", "icons"]


def captions(n, rng, holdout=20):
    found = []
    for source in SOURCES:
        f = Strokes(f"data/{source}.sketch.parquet").files[0]
        rows = np.flatnonzero(held_out(f, holdout))
        found += [(source, str(f.tables["label"][f.cols["label"][i]])) for i in rng.choice(rows, 400, replace=False)]
    picked = [found[i] for i in rng.choice(len(found), n, replace=False)]
    return [(s, ("an " if t[0].lower() in "aeio" else "a ") + t.replace("_", " ")) for s, t in picked]


def make(out, n=40, seed=0, model="sketch.npz"):
    rng, pen = np.random.default_rng(seed), mini.Pen(model)
    cells = []
    for k, (source, text) in enumerate(captions(n, rng)):
        p = pen.draw(text, temperature=0.4, rng=np.random.default_rng([seed, k]))
        cells.append(dict(text=text, source=source, drawing=np.rint(p * [240, 240, 1]).astype(int).tolist()))
        print(k, source, text, flush=True)
    json.dump(cells, open(out + ".json", "w"))


def main(out="paper/figures/random_sketches"):
    if not os.path.exists(out + ".json") or "--fresh" in sys.argv:
        make(out)
    cells = json.load(open(out + ".json"))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"]})
    fig, axes = plt.subplots(4, 10, figsize=(6.75, 3.4))
    for ax, cell in zip(axes.ravel(), cells):
        ax.axis("off")
        figures.ink(ax, figures.points(cell["drawing"]), 0.55, 0.04)
        ax.text(0.5, -0.04, textwrap.fill(cell["text"], 16), transform=ax.transAxes, fontsize=5.5, ha="center", va="top", color="#444")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.06, wspace=0.12, hspace=0.32)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=170)


if __name__ == "__main__":
    main()
