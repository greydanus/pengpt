"""Generate handwriting from a trained checkpoint.

    python sample.py --checkpoint out/best.pt --text "The quick brown fox" --out sample.png

Regenerate specific words (by index, printed with --show_indices) in a second
pass by re-running with --redo "3,7".
"""

import argparse

import torch

from pengpt import (DataConfig, create_datasets, load_checkpoint,
                    SampleParams, generate_paragraph, plot_paragraph)
from train import resolve_device


def main():
    p = argparse.ArgumentParser(description="Sample handwriting from a pengpt model")
    p.add_argument("--checkpoint", type=str, default="out/best.pt")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--out", type=str, default="sample.png")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--n_at_a_time", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show_indices", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)

    # Rebuild the dataset for warmup seeds; shrink it since we only need a few.
    data_cfg = DataConfig(**ckpt["data_config"])
    data_cfg.train_size, data_cfg.test_size = 1000, 100
    _, dataset, _, char_tok = create_datasets(data_cfg)
    assert char_tok.alphabet == ckpt["alphabet"], \
        "dataset alphabet does not match the checkpoint; use the training dataset file"

    params = SampleParams(temperature=args.temperature, top_k=args.top_k,
                          do_sample=not args.greedy, n_at_a_time=args.n_at_a_time,
                          n_context_words=data_cfg.num_words, seed=args.seed)
    word_offsets = generate_paragraph(model, dataset, args.text, params)

    fig, _ = plot_paragraph(word_offsets, args.text, params, show_indices=args.show_indices)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
