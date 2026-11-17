import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import pengpt as P


def main():
    p = argparse.ArgumentParser(description="how well a handwriting model takes up the hand of a writer it never saw: word score alone and "
                                            "after three of their words, and the rank of the true writer among 16 as the source of those words")
    p.add_argument("checkpoints", nargs="+", help="checkpoints, or patterns of them")
    p.add_argument("--n", type=int, default=16, help="writers the true one is ranked among, all from one source")
    p.add_argument("--prime", type=int, default=3, help="words of a writer the model reads before the word that is scored")
    p.add_argument("--targets", type=int, default=200, help="words scored from each source")
    p.add_argument("--out", default="out/transfer.md")
    args = p.parse_args()
    device = P.resolve_device("auto")
    lines = [f"| Run | Alone | After {args.prime} words | Gain | Style rank of {args.n} | True writer first | By source, style rank |", "|" + "---|" * 7]
    for path in [q for pattern in args.checkpoints for q in sorted(glob.glob(pattern))]:
        model, _, cfg, ckpt = P.load_for_sampling(path, device)
        test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
        scores = {k: P.unseen_score(model, d, device, (0, args.prime)) for k, d in test.unseen.items()}
        ranks = {k: P.style_rank(model, d, device, args.n, args.prime, args.targets) for k, d in test.unseen.items()}
        ranks = {k: v for k, v in ranks.items() if v}
        if not ranks:
            print(f"{path}: too few held-out writers with {args.prime + 1} words")
            continue
        alone, after = (float(np.mean([s[i] for s in scores.values()])) for i in (0, 1))
        rank, first = (float(np.mean([r[i] for r in ranks.values()])) for i in (0, 1))
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        lines.append(f"| {name} | {alone:.4f} | {after:.4f} | {alone - after:+.4f} | {rank:.2f} | {first:.1%} | "
                     + ", ".join(f"{k} {r[0]:.2f} ({r[2]})" for k, r in ranks.items()) + " |")
        print(lines[-1], flush=True)
    text = "\n".join(lines) + (f"\n\nWriters held out of training. Word scores are nats a pen move. Style rank: a held-out word is scored after "
                               f"{args.prime} other words by its own writer and after {args.prime} words by each of {args.n - 1} other held-out "
                               f"writers of the same source; 1 is perfect and {(args.n + 1) / 2:g} is chance. Words scored are in brackets.\n")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
