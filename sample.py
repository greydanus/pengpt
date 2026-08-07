"""Generate handwriting from a trained checkpoint.

    python sample.py --checkpoint out/cursive/best.pt --text "The quick brown fox"

If a word comes out wrong, find its index with --show_indices and regenerate
just that word: --redo 3,7
"""

import argparse

from pengpt import SampleParams, generate_paragraph, plot_paragraph
from pengpt.model import load_for_sampling
from train import resolve_device


def main():
    p = argparse.ArgumentParser(description="Sample handwriting from a pengpt model")
    p.add_argument("--checkpoint", type=str, default="out/cursive/best.pt")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--out", type=str, default="sample.png")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--n_at_a_time", type=int, default=2)
    p.add_argument("--redo", type=str, default="",
                   help="comma-separated word indices to regenerate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show_indices", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, dataset, cfg, _ = load_for_sampling(args.checkpoint, device, n_examples=100)

    params = SampleParams(temperature=args.temperature, top_k=args.top_k,
                          do_sample=not args.greedy, n_at_a_time=args.n_at_a_time,
                          max_tokens=cfg.max_seq_length - 1, seed=args.seed)
    words = generate_paragraph(model, dataset, args.text, params)
    if args.redo:
        redo = [int(i) for i in args.redo.split(",") if i.strip()]
        words = generate_paragraph(model, dataset, args.text, params,
                                   words=words, redo=redo)

    fig, _ = plot_paragraph(words, args.text, params, show_indices=args.show_indices)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
