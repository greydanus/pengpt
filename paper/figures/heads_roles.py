import json
import sys

import numpy as np
import torch

sys.path[:0] = [".", "tools", "paper/figures"]
import heads
import interp
import pengpt as P

ROLES = ["pen in the air", "touching down", "drawing", "lifting"]
SETS = {"four letter readers": [("cross_attn", 3, 0), ("cross_attn", 3, 2), ("cross_attn", 4, 1), ("cross_attn", 4, 3)],
        "self-attention L1h2": [("attn", 0, 1)], "self-attention L2h3": [("attn", 1, 2)]}


def roles(st, y):
    out, down = [], False
    for t in y:
        if t >= st.PAD or t in (st.WORD, st.END, st.BOS):
            out.append(4)
            continue
        base = list(st.expand([int(t)]))
        down = True if P.DOWN in base else False if P.UP in base else down
        out.append(1 if P.DOWN in base else 3 if P.UP in base else 2 if down else 0)
    return np.array(out)


def per_token(model, groups):
    out = []
    with torch.inference_mode():
        for X, C, Y, W in groups:
            lp = model(X, C, source=W)[0].float().log_softmax(-1)
            out.append(-lp.gather(-1, Y.clamp(min=0)[..., None])[..., 0].cpu().numpy().ravel())
    return np.concatenate(out)


def main(checkpoint="out/ablate_hand/base/best.pt", out="paper/figures/heads_roles.json", n=1536, each=512):
    device = P.resolve_device("auto")
    model, ds, cfg, ckpt = P.load_for_sampling(checkpoint, device)
    test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
    interp.instrument(model)
    model.eval()
    st, style = ds.stroke_tok, np.array([heads.lifts(test, i) for i in range(n)])
    order, result = np.argsort(style), {}
    for name, rows in (("cursive", order[:each]), ("print", order[-each:])):
        groups = heads.batches(test, [int(i) for i in rows], device)
        role = np.concatenate([roles(st, y) for _, _, Y, _ in groups for y in Y.cpu().numpy()])
        kept = np.concatenate([(Y >= 0).cpu().numpy().ravel() for _, _, Y, _ in groups])
        base = per_token(model, groups)
        result[name] = dict(share=[float((role[kept] == r).mean()) for r in range(4)], base=[float(base[kept & (role == r)].mean()) for r in range(4)])
        for label, members in SETS.items():
            for kind, l, h in members:
                a = getattr(model.blocks[l], kind)
                a.off = torch.tensor(sorted({h, *([] if getattr(a, "off", None) is None else a.off.tolist())}), device=device)
            cost = per_token(model, groups) - base
            for kind, l, h in members:
                getattr(model.blocks[l], kind).off = None
            result[name][label] = dict(by_role=[float(cost[kept & (role == r)].mean()) for r in range(4)], overall=float(cost[kept].mean()))
        print(name, result[name], flush=True)
    json.dump(result, open(out, "w"))


if __name__ == "__main__":
    main()
