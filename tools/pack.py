import argparse
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

import pengpt as P

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
import pen

TIERS = (3, 4, 5, 6, 7, 8)


def rounded(w, bits):
    scale = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 1)
    return torch.round(w / scale) * scale


def held_out(test, n, device, size=32):
    return [tuple(t.to(device) for t in P.trim(test.batch(range(k * size, (k + 1) * size)), test.stroke_tok.PAD)) for k in range(n)]


def loss_of(model, batches):
    with torch.inference_mode():
        nll = lambda X, C, Y, W: F.cross_entropy(model(X, C, source=W)[0].transpose(1, 2), Y, ignore_index=P.IGNORE_INDEX,
                                                 reduction="sum").item()
        total = sum(nll(*batch) for batch in batches)
    return total / sum((Y != P.IGNORE_INDEX).sum().item() for _, _, Y, _ in batches)


def harm_of(model, batches):
    mats = {n: p for n, p in model.named_parameters() if p.ndim == 2}
    harm = {n: dict.fromkeys(TIERS, 0.0) for n in mats}
    moved = {n: {b: (rounded(p.detach(), b) - p.detach()) ** 2 for b in TIERS} for n, p in mats.items()}
    for X, C, Y, W in batches:
        model.zero_grad()
        model(X, C, Y, W)[1].backward()
        for n, p in mats.items():
            for b in TIERS:
                harm[n][b] += float((moved[n][b] * p.grad ** 2).sum()) if p.grad is not None else 0.0
    return mats, harm


def plan_of(mats, harm, bits):
    plan, size = dict.fromkeys(mats, TIERS[-1]), {n: p.numel() for n, p in mats.items()}
    while sum(plan[n] * size[n] for n in mats) > bits * sum(size.values()):
        tier = {n: TIERS.index(plan[n]) for n in mats}
        steps = [((harm[n][TIERS[i - 1]] - harm[n][TIERS[i]]) / size[n], n, TIERS[i - 1]) for n, i in tier.items() if i]
        if not steps:
            break
        _, n, plan[n] = min(steps)
    return plan


def with_plan(model, mats, plan):
    saved = {n: p.detach().clone() for n, p in mats.items()}
    with torch.no_grad():
        for n, p in mats.items():
            p.copy_(rounded(saved[n], plan[n]))
    return saved


def pack(checkpoint, out, bits=7.0, passes=24, compare=False):
    device = P.resolve_device("auto")
    model, _, cfg, ckpt = P.load_for_sampling(checkpoint, device)
    test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
    model.eval()
    batches = held_out(test, 2 * passes, device)
    fit, check = batches[:passes], batches[passes:]
    mats, harm = harm_of(model, fit)
    plain = loss_of(model, check)
    plans = {f"{bits:g} bits a weight, by what each matrix can bear": plan_of(mats, harm, bits)}
    if compare:
        plans.update({f"{b} bits for every matrix": dict.fromkeys(mats, b) for b in TIERS if b >= bits - 1})
    for name, plan in plans.items():
        saved = with_plan(model, mats, plan)
        mean = sum(plan[n] * p.numel() for n, p in mats.items()) / sum(p.numel() for p in mats.values())
        print(f"{name}: {mean:.2f} bits a weight, held-out loss {loss_of(model, check):.4f} against {plain:.4f} unpacked", flush=True)
        with torch.no_grad():
            for n, p in mats.items():
                p.copy_(saved[n])
    plan = next(iter(plans.values()))
    total = sum(p.numel() for p in mats.values())
    share = {b: sum(p.numel() for n, p in mats.items() if plan[n] == b) / total for b in TIERS}
    print("matrices by bits: " + ", ".join(f"{b}: {share[b]:.0%}" for b in TIERS))
    pen.export(checkpoint, out, plan)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")
    return plan


def renamed(names, keep=(), seed=0):
    rng = np.random.default_rng(seed)
    order = [0] + [int(i) + 1 for i in rng.permutation(len(names) - 1)]
    ids = list(dict.fromkeys("".join(c) for c in rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), (4 * len(names), 5))))
    fresh = iter(ids)
    return order, [names[i] if names[i] in ("", *keep) else next(fresh) for i in order]


def anonymous(path, keep=(), seed=0):
    z = dict(np.load(path))
    names = [str(n) for n in z["writers"]]
    order, new = renamed(names, keep, seed)
    z["writers"] = np.array(new)
    for k in ("writer_emb.weight", "writer_emb.weight.scale"):
        z[k] = z[k][order]
    np.savez_compressed(path, **z)
    return dict(zip((names[i] for i in order), new))


def publish(checkpoint, out, hidden=False, keep=(), seed=0):
    c = torch.load(checkpoint, map_location="cpu", weights_only=True)
    c = {k: v for k, v in c.items() if k != "optimizer"}
    names = list(c.get("sources", c.get("writers", [""])))
    if hidden:
        order, c["sources"] = renamed(names, keep, seed)
        table = next(k for k in ("source_emb.weight", "writer_emb.weight") if k in c["model"])
        c["model"] = {**c["model"], table: c["model"][table][order].clone()}
        c["data_config"] = {**c["data_config"], "dataset": f"released.{P.kind(P.restore(c['data_config']))}.parquet"}
    torch.save(c, out)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, step {c['step']}, {len(names) - 1} sources"
          + ", named as --anonymous names them" * hidden)


def main():
    p = argparse.ArgumentParser(description="pack a checkpoint into the small file inference/pen.py runs: 3 to 8 bits for each matrix, fewer where "
                                            "rounding it moves the held-out loss least, to an average of --bits")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="hand.npz")
    p.add_argument("--bits", type=float, default=7.0)
    p.add_argument("--passes", type=int, default=24, help="held-out batches of 32 the choice is made on; as many again check it")
    p.add_argument("--compare", action="store_true", help="also score the same number of bits for every matrix")
    p.add_argument("--anonymous", action="store_true", help="give the writers five-character ids for names, in a shuffled order")
    p.add_argument("--keep", default="", help="with --anonymous, writers that keep their names, separated by commas")
    p.add_argument("--publish", default="", help="write the checkpoint itself here, as it can be released, and pack nothing: no optimizer, and "
                                                 "with --anonymous the writer ids, their order and no dataset names, as in the packed file")
    args = p.parse_args()
    if args.publish:
        return publish(args.checkpoint, args.publish, args.anonymous, args.keep.split(","))
    pack(args.checkpoint, args.out, args.bits, args.passes, args.compare)
    if args.anonymous:
        anonymous(args.out, args.keep.split(","))


if __name__ == "__main__":
    main()
