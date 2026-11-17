import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path[:0] = [".", "tools"]

RUNS = [("base", "all vectors"), ("image_label", "picture and label"), ("image_only", "picture only")]
CACHE = "paper/figures/captions.json"


def make(runs="out/ablate_sketch", n=16, guidance=2.0, temperature=0.6, seed=0):
    import guidance as G
    import pengpt as P
    from vectors import encoder
    device = P.resolve_device("auto")
    P.Vectors.encoder = encoder()
    asked = G.prompts(os.path.join(runs, "phrases.txt"), n, seed)
    drawn = {}
    for name, _ in RUNS:
        model, dataset, cfg, _ = P.load_for_sampling(os.path.join(runs, name, "best.pt"), device)
        drawn[name] = [[np.round(np.asarray(s), 4).tolist() for s in d] for d in G.draw(model, dataset, asked, guidance, temperature, seed)]
        print(name, [len(d) for d in drawn[name]], flush=True)
    json.dump(dict(prompts=asked, drawn=drawn), open(CACHE, "w"))


def main(out="paper/figures/captions.pdf", cols=8, first=None):
    if not os.path.exists(CACHE) or "--fresh" in sys.argv:
        make()
    kept = json.load(open(CACHE))
    kept["prompts"] = kept["prompts"][:first]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"]})
    bands = -(-len(kept["prompts"]) // cols)
    fig = plt.figure(figsize=(6.75, 3.05 * bands))
    grid = fig.add_gridspec(bands, 1, left=0.035, right=0.998, top=0.995, bottom=0.003, hspace=0.06)
    for b in range(bands):
        cells = grid[b].subgridspec(len(RUNS) + 1, cols, height_ratios=[0.62] + [1] * len(RUNS), wspace=0.06, hspace=0.04)
        for c in range(cols):
            k = b * cols + c
            if k >= len(kept["prompts"]):
                continue
            ax = fig.add_subplot(cells[0, c])
            ax.axis("off")
            ax.text(0.5, 0.0, textwrap.fill(kept["prompts"][k], 21), fontsize=5.2, ha="center", va="bottom", linespacing=1.05, color="#222222")
            for r, (name, label) in enumerate(RUNS):
                ax = fig.add_subplot(cells[r + 1, c])
                for s in kept["drawn"][name][k]:
                    s = np.asarray(s, float).reshape(-1, 2) if len(s) and len(s[0]) == 2 else np.asarray(s, float)[:, :2]
                    ax.plot(s[:, 0], -s[:, 1], color="#0b0b0b", lw=0.6, solid_capstyle="round", solid_joinstyle="round")
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_color("#dddddd")
                    side.set_linewidth(0.4)
                if c == 0:
                    ax.set_ylabel(label, fontsize=6.5, labelpad=2)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=170)
    print(out)


if __name__ == "__main__":
    main()
    main("paper/figures/captions_top.pdf", first=8)
