import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pengpt.data import load_examples
from pengpt.tokenizer import ScribeTokenizer, learn_merges
from pengpt.sampling import draw
from polyline.roundtrip import chamfer

from chunk.tokenizer import (ChunkTokenizer, RdpChunkTokenizer,
                             learn_codebook, learn_rdp_codebook)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/bigbank_3500.json.zip")
    p.add_argument("--grid", type=float, default=0.020)
    p.add_argument("--n_codes", type=int, default=512)
    p.add_argument("--match_eps", type=float, default=1.0)
    p.add_argument("--mode", default="rdp", choices=("exact", "kmeans", "rdp"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="chunk/roundtrip.png")
    args = p.parse_args()

    words = [e["points"] for e in load_examples(args.dataset)]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(words))
    n_test = min(1000, max(10, int(0.05 * len(words))))
    train_ix, test_ix = perm[:-n_test], perm[-n_test:]
    train_w = [words[i] for i in train_ix]

    base = ScribeTokenizer(grid=args.grid)
    sm = learn_merges([base.encode_word(words[i]) for i in train_ix[:600]], 512)
    scribe = ScribeTokenizer(grid=args.grid, merges=sm)

    if args.mode == "rdp":
        print("learning rdp-chunk codebook")
        book = learn_rdp_codebook(train_w, grid=args.grid, n_codes=args.n_codes,
                                  seed=args.seed)
        tok = RdpChunkTokenizer(grid=args.grid, match_eps=args.match_eps, codebook=book)
    else:
        print("learning step n-gram codebook")
        books = learn_codebook(train_w, grid=args.grid, n_codes=args.n_codes,
                               seed=args.seed, mode=args.mode)
        tok = ChunkTokenizer(grid=args.grid, match_eps=args.match_eps,
                             codebook=books, n_codes=args.n_codes)

    rows = []
    hits = miss = 0
    for i in test_ix:
        w = words[i]
        st = scribe.encode_word(w)
        ct = tok.encode_word(w)
        if hasattr(tok, "last_hits"):
            h, m = tok.last_hits
            hits += h
            miss += m
        rows.append((len(st), chamfer(w, scribe.decode_word(st)),
                     len(ct), chamfer(w, tok.decode_word(ct))))
    rows = np.array(rows)
    print(f"test n={len(rows)}  mode={args.mode}  K={args.n_codes}  eps={args.match_eps}")
    print(f"{'':12s} {'tokens/word':>14s} {'recon':>12s}")
    print(f"{'scribe':12s} {rows[:, 0].mean():8.1f}  p90 {np.percentile(rows[:, 0], 90):5.0f}   "
          f"{rows[:, 1].mean():.4f}  ({rows[:, 1].mean() / args.grid:.2f} grid)")
    print(f"{'chunk':12s} {rows[:, 2].mean():8.1f}  p90 {np.percentile(rows[:, 2], 90):5.0f}   "
          f"{rows[:, 3].mean():.4f}  ({rows[:, 3].mean() / args.grid:.2f} grid)")
    print(f"compression  {rows[:, 0].mean() / max(rows[:, 2].mean(), 1e-6):.2f}x")
    print(f"recon ratio  {rows[:, 3].mean() / max(rows[:, 1].mean(), 1e-6):.2f}x scribe error")
    if hits + miss:
        print(f"chunk hit    {100 * hits / (hits + miss):.0f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pick = [int(i) for i in test_ix[:6]]
    fig, axes = plt.subplots(len(pick), 3, figsize=(8, 2.1 * len(pick)))
    titles = ["original", "scribe", "chunk vq"]
    for r, i in enumerate(pick):
        st = scribe.encode_word(words[i])
        ct = tok.encode_word(words[i])
        decoded = [words[i], scribe.decode_word(st), tok.decode_word(ct)]
        ns = [None, len(st), len(ct)]
        for c, (pts, n) in enumerate(zip(decoded, ns)):
            ax = axes[r][c]
            draw(ax, pts, color="k" if c == 0 else ("C0" if c == 1 else "C2"))
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if n is not None:
                ax.text(0.02, 0.08, f"{n} tok", transform=ax.transAxes, fontsize=8)
    fig.suptitle(f"chunk {args.mode}  K={args.n_codes}  eps={args.match_eps}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
