"""Write the opening of the Iliad, the paragraph the original project used.

    python iliad.py --checkpoint out/best.pt

Butler's translation, the same passage as cursivetransformer's sample_v70.
"""

import argparse

from pengpt import SampleParams, generate_paragraph, plot_paragraph
from pengpt.model import load_for_sampling
from train import resolve_device

TEXT = ("Sing, O goddess, the anger of Achilles son of Peleus, that brought "
        "countless ills upon the Achaeans. Many a brave soul did it send "
        "hurrying down to Hades, and many a hero did it yield a prey to dogs "
        "and vultures, for so were the counsels of Jove fulfilled.")


def main():
    p = argparse.ArgumentParser(description="Generate the Iliad opening")
    p.add_argument("--checkpoint", type=str, default="out/best.pt")
    p.add_argument("--out", type=str, default="static/iliad.png")
    p.add_argument("--text", type=str, default=TEXT)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n_at_a_time", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--redo", type=str, default="")
    p.add_argument("--show_indices", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, dataset, cfg, ckpt = load_for_sampling(args.checkpoint, device, n_examples=100)
    params = SampleParams(temperature=args.temperature, n_at_a_time=args.n_at_a_time,
                          max_tokens=cfg.max_seq_length - 1, seed=args.seed)

    words = generate_paragraph(model, dataset, args.text, params)
    if args.redo:
        redo = [int(i) for i in args.redo.split(",") if i.strip()]
        words = generate_paragraph(model, dataset, args.text, params,
                                   words=words, redo=redo)

    fig, _ = plot_paragraph(words, args.text, params, figsize=(13, 7),
                            show_indices=args.show_indices)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved {args.out}  (checkpoint step {ckpt['step']})")


if __name__ == "__main__":
    main()
