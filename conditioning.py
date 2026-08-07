"""How strongly does the text prompt determine what the model draws?

    python conditioning.py --checkpoint out/quickdraw/best.pt --labels cat,apple,car,fish

Score one real drawing under every candidate label and see where its true label
ranks. If conditioning works the true label fits best, and the rank is
interpretable on its own: 1 of N is perfect, (N+1)/2 is chance.

Worth measuring separately from loss. A drawing model can post a healthy,
falling loss while ignoring its prompt entirely -- it learns the average shape
of the corpus and draws that regardless -- and samples alone are ambiguous early
in training, when everything looks like a scribble whether or not the prompt is
being read.

The reported spread, between the best and worst label's loss, says whether the
model merely prefers the right label or is actually driven by it. A high rank
with a tiny spread means the prompt is being read but barely steers generation,
which looks like every prompt producing the same drawing.

Score several drawings per label, which is what --per_label sets. One drawing
per label is not a small sample of the right thing, it is the wrong measurement:
per-label ranks are strongly bimodal, and a model can rank triangle 1st and
mouth 20th at the same checkpoint. Whichever single drawing came first then
decides the answer. One earlier version capped the number of labels rather than
the drawings per label, and on one checkpoint reported 3.0 where the converged
value was 9.7 -- the difference between "conditioning works" and "barely above
chance".

Read the per-label breakdown, not only the mean. A mean near chance can hide a
model that has learned a third of the categories well and the rest not at all,
and that is a different problem from one that has learned nothing.
"""

import argparse
import collections

import numpy as np
import torch

from pengpt.model import load_for_sampling
from train import resolve_device

DEFAULT_LABELS = ("cat,apple,car,fish,tree,house,star,umbrella,clock,ladder,"
                  "banana,bicycle")


def main():
    p = argparse.ArgumentParser(description="Measure prompt conditioning")
    p.add_argument("--checkpoint", type=str, default="out/quickdraw/best.pt")
    p.add_argument("--labels", type=str, default=DEFAULT_LABELS)
    p.add_argument("--per_label", type=int, default=8,
                   help="drawings per label; one is not enough, see module docstring")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    device = resolve_device(args.device)
    model, dataset, cfg, ckpt = load_for_sampling(args.checkpoint, device,
                                                  n_examples=4000)
    char_tok = dataset.char_tok

    found = collections.defaultdict(list)
    for i in range(len(dataset)):
        text = dataset.text_for(i)
        if text in labels and len(found[text]) < args.per_label:
            found[text].append(i)
        if len(found) >= len(labels) and all(
                len(v) >= args.per_label for v in found.values()):
            break
    if not found:
        raise SystemExit("none of those labels appear in the dataset")

    ranks, spreads = [], []
    by_label = {}
    for true, indices in sorted(found.items()):
        for i in indices:
            x, _, y = dataset[i]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            scored = []
            for label in labels:
                context = torch.from_numpy(
                    char_tok.encode(label, cfg.max_text_length)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    _, loss = model(x, context, y)
                scored.append((label, loss.item()))
            scored.sort(key=lambda r: r[1])
            rank = [l for l, _ in scored].index(true) + 1
            spread = (scored[-1][1] - scored[0][1]) / scored[0][1] * 100
            ranks.append(rank)
            spreads.append(spread)
            by_label.setdefault(true, []).append(rank)

    for true, rs in sorted(by_label.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {true:12s} mean rank {np.mean(rs):4.1f}/{len(labels)}"
              f"   over {len(rs)} drawings")

    chance = (len(labels) + 1) / 2
    print(f"\nstep {ckpt['step']}: mean rank {np.mean(ranks):.1f}/{len(labels)}"
          f" (chance {chance:.1f}), mean spread {np.mean(spreads):.1f}%")


if __name__ == "__main__":
    main()
