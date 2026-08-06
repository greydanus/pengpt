"""Render real handwriting beside what the model generates for the same text.

    python compare.py --checkpoint out_run1/best.pt --out static/compare.png
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pengpt import DataConfig, create_datasets, load_checkpoint, SampleParams
from pengpt.sampling import layout_words
from train import resolve_device


def draw(ax, words, color, params):
    if not words:
        ax.axis("off")
        return
    placed = layout_words(words, params)
    points = np.vstack(placed)
    pen_down = points[:, 2] == 1
    for chunk in np.split(points, np.flatnonzero(~pen_down) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            ax.plot(chunk[:, 0], -chunk[:, 1], color=color, linewidth=1.4,
                    solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="out_run1/best.pt")
    p.add_argument("--out", type=str, default="static/compare.png")
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = DataConfig(**ckpt["data_config"])
    cfg.train_size, cfg.test_size = 200, 200
    _, test, stroke_tok, char_tok = create_datasets(cfg, merges=ckpt["merges"])

    params = SampleParams(temperature=args.temperature,
                          max_tokens=cfg.max_seq_length - 1)
    fig, axes = plt.subplots(args.num, 2, figsize=(15, 1.9 * args.num))
    for i in range(args.num):
        x, c, _ = test[i]
        text = test.text_for(i)

        target = stroke_tok.decode(x.numpy())
        draw(axes[i, 0], target, "black", params)
        axes[i, 0].set_title(f'target: "{text}"', fontsize=10, loc="left")

        context = c.unsqueeze(0).to(device)
        idx = torch.full((1, 1), stroke_tok.PAD, dtype=torch.long, device=device)
        out = model.generate(idx, context, max_new_tokens=params.max_tokens,
                             temperature=params.temperature, do_sample=True,
                             end_token=stroke_tok.END, pad_token=stroke_tok.PAD)
        generated = stroke_tok.decode(out[0].cpu().numpy()[1:])
        draw(axes[i, 1], generated, "crimson", params)
        axes[i, 1].set_title(f"generated ({len(generated)} words)", fontsize=10, loc="left")

    fig.suptitle(f"step {ckpt['step']}  |  black = real handwriting, red = model",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
