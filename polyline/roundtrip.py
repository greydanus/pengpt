import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pengpt.data import load_examples
from pengpt.tokenizer import ScribeTokenizer, learn_merges
from pengpt.sampling import draw

from polyline.tokenizer import PolylineTokenizer, learn_merges


def chamfer(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[a[:, 2] == 1][:, :2] if a.ndim == 2 and a.shape[1] == 3 else np.asarray(a)[:, :2]
    b = b[b[:, 2] == 1][:, :2] if b.ndim == 2 and b.shape[1] == 3 else np.asarray(b)[:, :2]
    if len(a) == 0 or len(b) == 0:
        return np.inf
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/bigbank_3500.json.zip")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--grid", type=float, default=0.020)
    p.add_argument("--epsilon", type=float, default=0.010)
    p.add_argument("--max_run", type=int, default=16)
    p.add_argument("--n_merges", type=int, default=256)
    p.add_argument("--out", default="polyline/roundtrip.png")
    args = p.parse_args()

    examples = load_examples(args.dataset, args.n)
    words = [e["points"] for e in examples]

    base = ScribeTokenizer(grid=args.grid)
    merges = learn_merges([base.encode_word(w) for w in words[:600]], args.n_merges)
    scribe = ScribeTokenizer(grid=args.grid, merges=merges)
    raw_poly = PolylineTokenizer(grid=args.grid, epsilon=args.epsilon, max_run=args.max_run)
    poly_merges = learn_merges(
        [raw_poly.encode_word(w) for w in words[:600]], args.n_merges,
        reserved=(raw_poly.DOWN, raw_poly.UP))
    poly = PolylineTokenizer(grid=args.grid, epsilon=args.epsilon,
                             max_run=args.max_run, merges=poly_merges)

    rows = []
    for w in words:
        st = scribe.encode_word(w)
        pt = poly.encode_word(w)
        sd = scribe.decode_word(st)
        pd = poly.decode_word(pt)
        rows.append((len(st), chamfer(w, sd), len(pt), chamfer(w, pd)))
    rows = np.array(rows)
    print(f"n={len(words)}  grid={args.grid}  poly eps={args.epsilon}  "
          f"scribe merges={len(merges)}  poly merges={len(poly_merges)}")
    print(f"{'':12s} {'tokens/word':>14s} {'recon':>12s}")
    print(f"{'scribe':12s} {rows[:, 0].mean():8.1f}  p90 {np.percentile(rows[:, 0], 90):5.0f}   "
          f"{rows[:, 1].mean():.4f}  ({rows[:, 1].mean() / args.grid:.2f} grid)")
    print(f"{'polyline':12s} {rows[:, 2].mean():8.1f}  p90 {np.percentile(rows[:, 2], 90):5.0f}   "
          f"{rows[:, 3].mean():.4f}  ({rows[:, 3].mean() / args.grid:.2f} grid)")
    print(f"compression  {rows[:, 0].mean() / max(rows[:, 2].mean(), 1e-6):.2f}x fewer tokens")
    print(f"recon ratio  {rows[:, 3].mean() / max(rows[:, 1].mean(), 1e-6):.2f}x scribe error")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pick = [0, 3, 7, 12, 20, 28]
    pick = [i for i in pick if i < len(words)]
    fig, axes = plt.subplots(len(pick), 3, figsize=(8, 2.1 * len(pick)))
    titles = ["original", "scribe", "polyline"]
    for r, i in enumerate(pick):
        st = scribe.encode_word(words[i])
        pt = poly.encode_word(words[i])
        decoded = [words[i], scribe.decode_word(st), poly.decode_word(pt)]
        ns = [None, len(st), len(pt)]
        for c, (pts, n) in enumerate(zip(decoded, ns)):
            ax = axes[r][c]
            draw(ax, pts, color="k" if c == 0 else ("C0" if c == 1 else "C3"))
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if n is not None:
                ax.text(0.02, 0.08, f"{n} tok", transform=ax.transAxes, fontsize=8)
    fig.suptitle(f"cursive roundtrip  grid={args.grid}  eps={args.epsilon}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
