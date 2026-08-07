"""Draw named objects from a model trained on a drawing corpus.

    python draw.py --checkpoint out/quickdraw/best.pt --labels cat,car,fish
    python draw.py --checkpoint out/quickdraw/best.pt --labels cat --rows 8

Handwriting wants sample.py, which lays words out as a paragraph. Here each
sample is a standalone object, so labels run across and repeats run down.

Several samples per label is the point. One sample cannot tell a model that
ignores its prompt from a model having a bad draw, and a column of repeats
answers that immediately.
"""

import argparse

import matplotlib.pyplot as plt
import torch

from pengpt import SampleParams, generate, plot_words
from pengpt.model import load_for_sampling
from train import resolve_device


def draw_grid(model, dataset, labels, rows=4, temperature=1.0, seed=0):
    params = SampleParams(temperature=temperature,
                          max_tokens=dataset.cfg.max_seq_length - 1)
    fig, axes = plt.subplots(rows, len(labels),
                             figsize=(1.9 * len(labels), 1.9 * rows),
                             squeeze=False)
    for col, text in enumerate(labels):
        for row in range(rows):
            torch.manual_seed(seed + row * 31 + 3)
            plot_words(generate(model, dataset, text, params), params,
                       ax=axes[row][col], color="k")
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.24 / rows])

    # One height for every label. An axis title sits relative to that axis's
    # contents, so a column of short drawings would place its label lower.
    top = max(ax.get_position().y1 for ax in axes[0])
    for col, text in enumerate(labels):
        box = axes[0][col].get_position()
        fig.text(box.x0 + box.width / 2, top + 0.012, text,
                 ha="center", va="bottom", fontsize=10)
    return fig, top


def main():
    p = argparse.ArgumentParser(description="Draw objects from a pengpt model")
    p.add_argument("--checkpoint", type=str, default="out/quickdraw/best.pt")
    p.add_argument("--labels", type=str, required=True,
                   help="comma-separated category names")
    p.add_argument("--rows", type=int, default=4, help="samples per label")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="drawings.png")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    device = resolve_device(args.device)
    model, dataset, _, ckpt = load_for_sampling(args.checkpoint, device,
                                                n_examples=4000)

    alphabet = set(dataset.char_tok.alphabet)
    for label in labels:
        missing = sorted(set(label) - alphabet)
        if missing:
            print(f"WARNING: {label!r} contains {missing}, which this model's "
                  f"alphabet does not have; those encode as padding")

    fig, top = draw_grid(model, dataset, labels, args.rows, args.temperature,
                         args.seed)
    fig.text(0.5, top + 0.055, f"step {ckpt['step']:,}", ha="center",
             va="bottom", fontsize=11)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
