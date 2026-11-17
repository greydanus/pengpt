import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path[:0] = [".", "inference", "tools", "paper/figures"]
import samples

ROWS, CACHE = [0, 3, 8, 11, 14, 17], "paper/figures/transfer.json"


def make(checkpoint="out/ablate_hand/base/best.pt", shown=3):
    import pengpt as P
    import unseen_hands as U
    device = P.resolve_device("auto")
    model, dataset, cfg, ckpt = P.load_for_sampling(checkpoint, device)
    known, picked, cells = pickle.load(open(U.CACHE, "rb")), json.load(open("paper/figures/unseen_hands.json")), []
    for k in ROWS:
        writer, rng = known[picked[k]["writer"]], np.random.default_rng(k)
        fine = [i for i, t in enumerate(writer["texts"]) if t.isalpha() and 3 <= len(t) <= 8]
        targets = [int(i) for i in rng.choice(fine, shown, replace=False)]
        asked = {writer["texts"][i].lower() for i in targets}
        other = [i for i, t in enumerate(writer["texts"]) if t.lower() not in asked]
        rest = dict(writer, points=[writer["points"][i] for i in other], texts=[writer["texts"][i] for i in other])
        before = U.primer(dataset.stroke_tok, rest, rng)
        words = U.write(model, dataset, cfg, before, [writer["texts"][i] for i in targets], 0, 1, k)
        cells.append(dict(text=[writer["texts"][i] for i in targets], primed=[w for w, _ in before],
                          real=[writer["points"][i].tolist() for i in targets], model=[w[0] for w in words]))
        print(cells[-1]["text"], "primed on", cells[-1]["primed"], flush=True)
    json.dump(cells, open(CACHE, "w"))


def extent(words):
    points = np.concatenate(samples.mini.layout(words, line_width=99))
    return points, points[points[:, 2] == 1, :2]


def line(ax, words, colour, s, W=3.0, H=0.5):
    points, down = extent(words)
    lo, size = down.min(0), np.ptp(down, 0)
    for a, b in samples.mini.strokes(points):
        q = (points[a:b, :2] - lo - size / 2) * s
        q = q if b - a > 1 else np.r_[q, q + 1e-3]
        ax.plot(W / 2 + q[:, 0], H / 2 - q[:, 1], color=colour, lw=0.7, solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")


def main(out="paper/figures/transfer.pdf"):
    if not os.path.exists(CACHE) or "--fresh" in sys.argv:
        make()
    cells = json.load(open(CACHE))
    half = (len(cells) + 1) // 2
    fig = plt.figure(figsize=(3.3, 0.55 * half + 0.12 * (half - 1)))
    grid = fig.add_gridspec(half, 2, left=0.005, right=0.995, top=1.0, bottom=0.0, wspace=0.09, hspace=0.22)
    points = lambda words: [np.array(w, float).reshape(-1, 3) for w in words if len(w)]
    for k, cell in enumerate(cells):
        top, under = grid[k % half, k // half].subgridspec(2, 1, hspace=0.0).subplots()
        real, made = points(cell["real"]), points(cell["model"])
        s = min(min(0.97 * 3.0 / np.ptp(extent(w)[1], 0)[0], 0.92 * 0.5 / np.ptp(extent(w)[1], 0)[1]) for w in (real, made))
        line(top, real, "#1f5fa8", s)
        line(under, made, "#0b0b0b", s)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(out)


if __name__ == "__main__":
    main()
