"""Render real handwriting beside what the model generates for the same text.

    python compare.py --checkpoint out/best.pt --out static/compare.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pengpt import DataConfig, create_datasets, load_checkpoint, SampleParams
from pengpt.sampling import generate, plot_words
from train import resolve_device


def main():
    p = argparse.ArgumentParser(description="Compare real handwriting to generations")
    p.add_argument("--checkpoint", type=str, default="out/best.pt")
    p.add_argument("--out", type=str, default="static/compare.png")
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = DataConfig(**ckpt["data_config"])
    cfg.train_size, cfg.test_size = 200, 200
    _, test, stroke_tok, _ = create_datasets(cfg, merges=ckpt["merges"])
    params = SampleParams(temperature=args.temperature,
                          max_tokens=cfg.max_seq_length - 1)

    fig, axes = plt.subplots(args.num, 2, figsize=(15, 1.9 * args.num))
    for i in range(args.num):
        x, _, _ = test[i]
        text = test.text_for(i)
        plot_words(stroke_tok.decode(x.numpy()), params, ax=axes[i, 0], color="black")
        axes[i, 0].set_title(f'target: "{text}"', fontsize=10, loc="left")

        words = generate(model, test, text, params)
        plot_words(words, params, ax=axes[i, 1], color="crimson")
        axes[i, 1].set_title(f"generated ({len(words)} words)", fontsize=10, loc="left")

    fig.suptitle(f"step {ckpt['step']}  |  black = real handwriting, red = model",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
