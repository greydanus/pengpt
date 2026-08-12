import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pengpt.model import load_for_sampling
from pengpt.sampling import SampleParams, generate, plot_words
from train import resolve_device

from polyline.data import create_datasets
from polyline.tokenizer import PolylineTokenizer
from pengpt.config import DataConfig
from pengpt.model import load_checkpoint


def load_poly(path, device):
    model, ckpt = load_checkpoint(path, device)
    cfg = DataConfig(**{k: v for k, v in ckpt["data_config"].items()
                        if k in DataConfig.__dataclass_fields__})
    meta = ckpt.get("polyline", {})
    tok = PolylineTokenizer(grid=meta.get("grid", cfg.grid),
                            epsilon=meta.get("epsilon", 0.010),
                            max_run=meta.get("max_run", 16),
                            max_chunk_verts=meta.get("max_chunk_verts", 4),
                            merges=ckpt.get("merges") or [])
    cfg.train_size = cfg.test_size = 200
    _, dataset, _, _ = create_datasets(cfg, tokenizer=tok)
    assert dataset.char_tok.alphabet == ckpt["alphabet"]
    return model, dataset, cfg, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scribe", default="out/cursive/best.pt")
    p.add_argument("--poly", default="out/polyline_cursive/best.pt")
    p.add_argument("--out", default="polyline/compare_samples.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    s_model, s_ds, s_cfg, s_ckpt = load_for_sampling(args.scribe, device, n_examples=200)
    p_model, p_ds, p_cfg, p_ckpt = load_poly(args.poly, device)

    prompts = []
    for i in range(min(len(s_ds), 2000)):
        t = s_ds.text_for(i)
        if t and t not in prompts:
            prompts.append(t)
        if len(prompts) >= 6:
            break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(prompts), 2, figsize=(10, 2.0 * len(prompts)))
    s_params = SampleParams(max_tokens=s_cfg.max_seq_length - 1, seed=0, verbose=False)
    p_params = SampleParams(max_tokens=p_cfg.max_seq_length - 1, seed=0, verbose=False)
    for r, text in enumerate(prompts):
        torch.manual_seed(0)
        sw = generate(s_model, s_ds, text, s_params)
        torch.manual_seed(0)
        pw = generate(p_model, p_ds, text, p_params)
        plot_words(sw, s_params, ax=axes[r][0], color="C0")
        plot_words(pw, p_params, ax=axes[r][1], color="C3")
        axes[r][0].set_ylabel(text, fontsize=8, rotation=0, ha="right", va="center")
        if r == 0:
            axes[r][0].set_title(f"scribe  step {s_ckpt.get('step')}", fontsize=10)
            axes[r][1].set_title(f"polyline  step {p_ckpt.get('step')}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
