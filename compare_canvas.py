import argparse
import os

import torch

from pengpt.model import load_for_sampling
from pengpt.sampling import SampleParams, generate, plot_words
from train import resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", default="out/cursive/best.pt")
    p.add_argument("--after", default="out/cursive_canvas/best.pt")
    p.add_argument("--out", default="out/cursive_canvas/compare.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    b_model, b_ds, b_cfg, b_ckpt = load_for_sampling(args.before, device, 200)
    a_model, a_ds, a_cfg, a_ckpt = load_for_sampling(args.after, device, 200)

    prompts = []
    for i in range(min(len(b_ds), 2000)):
        t = b_ds.text_for(i)
        if t and t not in prompts:
            prompts.append(t)
        if len(prompts) >= 6:
            break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(prompts), 2, figsize=(10, 2.0 * len(prompts)))
    bp = SampleParams(max_tokens=b_cfg.max_seq_length - 1, seed=0, verbose=False)
    ap = SampleParams(max_tokens=a_cfg.max_seq_length - 1, seed=0, verbose=False)
    for r, text in enumerate(prompts):
        torch.manual_seed(0)
        bw = generate(b_model, b_ds, text, bp)
        torch.manual_seed(0)
        aw = generate(a_model, a_ds, text, ap)
        plot_words(bw, bp, ax=axes[r][0], color="C0")
        plot_words(aw, ap, ax=axes[r][1], color="C2")
        axes[r][0].set_ylabel(text, fontsize=8, rotation=0, ha="right", va="center")
        if r == 0:
            axes[r][0].set_title(f"scribe  step {b_ckpt.get('step')}", fontsize=10)
            axes[r][1].set_title(f"pen+canvas  step {a_ckpt.get('step')}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
