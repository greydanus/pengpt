"""How strongly does the text prompt determine what the model draws?

    python conditioning.py --checkpoint out/best.pt --labels cat,apple,car,fish

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

Score enough drawings. On six the mean rank swung from 2.3 to 4.0 between two
checkpoints a thousand steps apart, which reads as a regression and is not one:
twenty drawings against twenty labels put the same model at 5.4 against a chance
of 10.5, steadily.
"""

import argparse

import numpy as np
import torch

from pengpt.model import load_for_sampling
from train import resolve_device

DEFAULT_LABELS = ("cat,apple,car,fish,tree,house,star,umbrella,clock,ladder,"
                  "banana,bicycle")


def main():
    p = argparse.ArgumentParser(description="Measure prompt conditioning")
    p.add_argument("--checkpoint", type=str, default="out/best.pt")
    p.add_argument("--labels", type=str, default=DEFAULT_LABELS)
    p.add_argument("--n", type=int, default=25,
                   help="drawings to score; below about 20 the rank is noise")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    device = resolve_device(args.device)
    model, dataset, cfg, ckpt = load_for_sampling(args.checkpoint, device,
                                                  n_examples=4000)
    char_tok = dataset.char_tok

    found = {}
    for i in range(len(dataset)):
        text = dataset.text_for(i)
        if text in labels and text not in found:
            found[text] = i
        if len(found) >= args.n:
            break
    if not found:
        raise SystemExit("none of those labels appear in the dataset")

    ranks, spreads = [], []
    for true, i in found.items():
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
        print(f"  {true:12s} true label ranked {rank:2d}/{len(labels)}"
              f"   spread {spread:5.1f}%")

    chance = (len(labels) + 1) / 2
    print(f"\nstep {ckpt['step']}: mean rank {np.mean(ranks):.1f}/{len(labels)}"
          f" (chance {chance:.1f}), mean spread {np.mean(spreads):.1f}%")


if __name__ == "__main__":
    main()
