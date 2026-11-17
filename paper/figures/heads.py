import sys

import numpy as np
import torch

sys.path[:0] = [".", "tools"]
import interp
import pengpt as P
from strokes import to_strokes

REACH, BINS, REFERENCE = 6, 5, (3, 2)


def lifts(test, i):
    chosen, words, _ = test.draw(np.random.default_rng([test.seed, int(i)]))
    letters = sum(ch.isalnum() for j in chosen for ch in test.bank_texts[j])
    return sum(len([s for s in to_strokes(w) if len(s) > 2]) for w in words) / max(letters, 1)


def batches(test, rows, device, size=64):
    X, C, Y, W = P.trim(test.batch(rows), test.stroke_tok.PAD)
    return [tuple(t[k:k + size].to(device) for t in (X, C, Y, W)) for k in range(0, len(rows), size)]


def progress(at):
    out = np.zeros(len(at))
    starts = np.flatnonzero(np.r_[True, at[1:] != at[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(at)]):
        out[a:b] = (np.arange(b - a) + 0.5) / (b - a)
    return out


def reading(model, groups, pad=0):
    L, H = model.cfg.n_layer, model.cfg.n_head
    offsets = np.zeros((L, H, BINS, 2 * REACH + 1))
    writer, counts = np.zeros((L, H, BINS)), np.zeros(BINS)
    with torch.inference_mode():
        for X, C, Y, W in groups:
            model(X, C, source=W)
            att = torch.stack([b.cross_attn.weights for b in model.blocks]).float().cpu().numpy()
            n = att.shape[-1] - C.shape[1]
            for b in range(X.shape[0]):
                keep, length = (Y[b] >= 0).cpu().numpy(), int((C[b] != pad).sum())
                a = att[:, b][:, :, keep]
                at = a[REFERENCE[0], REFERENCE[1]][:, n:n + length].argmax(-1)
                stage = np.minimum((progress(at) * BINS).astype(int), BINS - 1)
                for t in range(a.shape[2]):
                    lo, hi = max(0, at[t] - REACH), min(length, at[t] + REACH + 1)
                    offsets[:, :, stage[t], lo - at[t] + REACH:hi - at[t] + REACH] += a[:, :, t, n + lo:n + hi]
                    writer[:, :, stage[t]] += a[:, :, t, :n].sum(-1)
                    counts[stage[t]] += 1
    return offsets / counts[None, None, :, None], writer / counts


def knockout(model, groups, device):
    count = sum((Y >= 0).sum().item() for _, _, Y, _ in groups)
    base = P.nll(model, groups, device) / count
    cost = {}
    for kind in ("cross_attn", "attn"):
        cost[kind] = np.zeros((model.cfg.n_layer, model.cfg.n_head))
        for (l, h), _ in np.ndenumerate(cost[kind]):
            a = getattr(model.blocks[l], kind)
            a.off = torch.tensor([h], device=device)
            with torch.inference_mode():
                cost[kind][l, h] = P.nll(model, groups, device) / count - base
            a.off = None
    return base, cost


def main(checkpoint="out/ablate_hand/base/best.pt", out="paper/figures/heads.npz", n=1536, each=512):
    device = P.resolve_device("auto")
    model, ds, cfg, ckpt = P.load_for_sampling(checkpoint, device)
    test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
    interp.instrument(model)
    model.eval()
    style = np.array([lifts(test, i) for i in range(n)])
    order = np.argsort(style)
    kept = dict(cursive_below=style[order[each - 1]], print_above=style[order[-each]])
    for name, rows in (("cursive", order[:each]), ("print", order[-each:])):
        groups = batches(test, [int(i) for i in rows], device)
        kept[f"{name}_offsets"], kept[f"{name}_writer"] = reading(model, groups)
        print(name, "read", flush=True)
    np.savez(out, **kept)


if __name__ == "__main__":
    main()
