import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "font.size": 8})
sys.path[:0] = [".", "tools", "paper/figures"]
import heads
import interp
import pengpt as P

REACH, SHOW = 6, 4
INKS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2"]
CHOSEN = tuple(int(v) for v in sys.argv[1].split(",")) if len(sys.argv) > 1 else heads.REFERENCE
PICKS = {"cursive": (3,), "print": (8,)}
NAMED = {(3, 2), (4, 3)}
SHOWN = [((1, 0), "wide window"), ((1, 2), "writer and text"), ((2, 0), "the next letter"), ((3, 2), "this letter"), ((4, 3), "this and the next")]


EXAMPLES = "paper/figures/heads_examples.json"


def cached():
    if not os.path.exists(EXAMPLES) or "--fresh" in sys.argv:
        kept = [dict(style=style, text=text, strokes=[[pts.tolist(), owner.tolist()] for pts, owner in strokes],
                     att={f"{l},{h}": np.round(att[l, h], 4).tolist() for (l, h), _ in SHOWN}, first=int(first))
                for style, text, strokes, att, first in examples()]
        json.dump(kept, open(EXAMPLES, "w"))
    out = []
    for c in json.load(open(EXAMPLES)):
        att = {tuple(int(v) for v in k.split(",")): np.array(a) for k, a in c["att"].items()}
        out.append((c["style"], c["text"], [(np.array(p), np.array(o)) for p, o in c["strokes"]], att, c["first"]))
    return out


def runs(letters, strokes, text, least=3):
    at = letters.argmax(-1).copy()
    edges = np.flatnonzero(np.r_[True, at[1:] != at[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < least and 0 < a and b < len(at) and at[a - 1] == at[b]:
            at[a:b] = at[b]
    edges = np.flatnonzero(np.r_[True, at[1:] != at[:-1], True])
    whole = lambda a, b: any(a <= owner.min() - 1 and owner.max() - 1 <= b for _, owner in strokes)
    found = [(a, b - 1, at[a]) for a, b in zip(edges[:-1], edges[1:])
             if b - a >= least and letters[a:b, at[a]].mean() > (0.15 if whole(a, b - 1) else 0.2)]
    shape = lambda pts: " (cross)" if np.ptp(pts[:, 0]) > 1.3 * np.ptp(pts[:, 1]) else ""
    out = []
    for a, b, c in found:
        inside = [(pts[(owner - 1 >= a) & (owner - 1 <= b)], owner[(owner - 1 >= a) & (owner - 1 <= b)] - 1) for pts, owner in strokes]
        inside = [(pts, cols) for pts, cols in inside if len(np.unique(cols)) >= 2]
        again = sum(1 for _, _, other in found if other == c) > 1
        if text[c] == "t" and len(inside) > 1:
            out += [(cols.min(), cols.max(), c, shape(pts)) for pts, cols in inside]
        elif text[c] == "t" and again and inside:
            out.append((a, b, c, shape(max(inside, key=lambda item: len(item[0]))[0])))
        else:
            out.append((a, b, c, ""))
    longest = {}
    for a, b, c, part in out:
        if not part and b - a > longest.get(c, (0, 0, -1))[2]:
            longest[c] = (a, b, b - a)
    out = [(a, b, c, part) for a, b, c, part in out if part or longest[c][:2] == (a, b)]
    for c in range(len(text)):
        if c not in longest and letters[:, c].max() > 0.15:
            t = int(letters[:, c].argmax())
            out.append((t, t, c, ""))
    return out


def examples(checkpoint="out/ablate_hand/base/best.pt", n=1536, head=heads.REFERENCE):
    model, ds, cfg, ckpt = P.load_for_sampling(checkpoint, "cpu")
    test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
    interp.instrument(model)
    st, paths, found = ds.stroke_tok, interp.token_paths(ds.stroke_tok), {"cursive": [], "print": []}
    for i in range(n):
        chosen, _, _ = test.draw(np.random.default_rng([test.seed, i]))
        text = test.bank_texts[chosen[0]]
        if len(chosen) == 1 and text.isalpha() and text.islower() and 5 <= len(text) <= 7:
            lifts = heads.lifts(test, i)
            if lifts < 0.45 or lifts > 1.25:
                found["cursive" if lifts < 0.45 else "print"].append((i, text))
    print({k: [(j, t) for j, (_, t) in enumerate(v)][:14] for k, v in found.items()})
    out = []
    for style, picks in PICKS.items():
        for j in picks:
            i, text = found[style][j]
            X, C, Y, W = heads.batches(test, [i], "cpu")[0]
            with torch.inference_mode():
                model(X, C, source=W)
            att = torch.stack([blk.cross_attn.weights[0] for blk in model.blocks]).float().numpy()
            seq, first, keep = X[0].numpy(), att.shape[-1] - C.shape[1], (Y[0] >= 0).numpy()
            out.append((style, text, interp.trace(st, seq, paths)[2], att[:, :, keep][..., :first + len(text)], first))
    return out

COLOURS = {"four letter readers": "#d62728", "self-attention L1h2": "#1f77b4", "self-attention L2h3": "#6baed6"}
ROLES = ["pen in\nthe air", "touching\ndown", "drawing", "lifting"]


def main(out="paper/figures/heads.pdf"):
    r, knock = np.load("paper/figures/heads.npz"), json.load(open("paper/figures/heads.json"))
    offsets = (r["cursive_offsets"].mean(2) + r["print_offsets"].mean(2)) / 2
    writer = (r["cursive_writer"].mean(2) + r["print_writer"].mean(2)) / 2
    L, H = writer.shape
    table = np.c_[writer.reshape(-1, 1), np.full((L * H, 1), np.nan), offsets.reshape(L * H, -1)[:, REACH - SHOW:REACH + SHOW + 1]]
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
    fig = plt.figure(figsize=(6.75, 3.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.5, 1.7], wspace=0.13, left=0.06, right=0.995, top=0.91, bottom=0.17)

    ax = fig.add_subplot(grid[0])
    ax.imshow(table, aspect="auto", cmap="Blues", vmin=0, vmax=0.6, interpolation="nearest")
    ax.set_xticks([0, *range(2, 2 * SHOW + 3)])
    ax.set_xticklabels(["writer", *[f"{k:+d}" if k else "0" for k in range(-SHOW, SHOW + 1)]], fontsize=6.5)
    ax.get_xticklabels()[0].set_rotation(90)
    ax.set_yticks([l * H + (H - 1) / 2 for l in range(L)])
    ax.set_yticklabels([f"layer {l + 1}" for l in range(L)], fontsize=6.5)
    for l in range(1, L):
        ax.axhline(l * H - 0.5, color="#8a8a8a", lw=0.5)
    ax.tick_params(length=0)
    ax.set_xlabel("letters before and after the one being written")
    ax.set_title("a)  What each head reads", loc="left", fontsize=9)
    for name, (l, h), k, x, y, ha in (("next", (2, 0), 1, 2.9, 4.6, "left"), ("this letter", (3, 2), 0, -1.0, 12.2, "right"),
                                      ("this + next", (4, 3), 1, 1.9, 22.5, "left")):
        ax.annotate(name, (2 + SHOW + k + (-0.45 if ha == "right" else 0), l * H + h), xytext=(2 + SHOW + x, y), fontsize=6.5, color="#b03030",
                    ha=ha, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#b03030"))
    for side in ax.spines.values():
        side.set_visible(False)

    bx = fig.add_subplot(grid[1])
    for style, colour, shift in (("cursive", "#1f77b4", -0.2), ("print", "#d62728", 0.2)):
        bx.barh(np.arange(L * H) + shift, np.array(knock[style]["knock_cross_attn"]).ravel(), 0.4, color=colour, label=style)
    bx.set_ylim(L * H - 0.5, -0.5)
    bx.set_yticks([])
    for l in range(1, L):
        bx.axhline(l * H - 0.5, color="#8a8a8a", lw=0.5)
    bx.set_xlabel("cost of removing\n(nats a token)")
    bx.set_title("b)  If removed", loc="left", fontsize=9)
    bx.legend(fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.02, 0.008), handlelength=0.9, labelspacing=0.1, borderpad=0, borderaxespad=0)
    bx.spines[["top", "right", "left"]].set_visible(False)

    words = cached()
    cells = grid[2].subgridspec(len(SHOWN) + 1, len(words), height_ratios=[1.2] + [1.9 if head in NAMED else 0.75 for head, _ in SHOWN], hspace=0.5, wspace=0.06)
    for j, (style, text, strokes, att, first) in enumerate(words):
        ax = fig.add_subplot(cells[0, j])
        for pts, _ in strokes:
            ax.plot(pts[:, 0], -pts[:, 1], color="#1f77b4" if style == "cursive" else "#d62728", lw=1.2, solid_capstyle="round")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(("c)  Five heads over time" if j == 0 else "") , loc="left", fontsize=9, pad=8)
        for k, ((l, h), name) in enumerate(SHOWN):
            ax = fig.add_subplot(cells[k + 1, j])
            ax.imshow(att[(l, h)].T, aspect="auto", cmap="Blues", vmin=0, vmax=0.6, interpolation="nearest")
            ax.set_xticks([])
            lettered = j == 0 and (l, h) in NAMED
            ax.set_yticks(range(att[SHOWN[0][0]].shape[-1]) if lettered else [])
            ax.set_yticklabels(["w", *text] if lettered else [], fontsize=4.4)
            ax.tick_params(length=0, pad=1)
            ax.set_title(f"L{l + 1}h{h + 1}: {name}" if j == 0 else f"L{l + 1}h{h + 1}", loc="left", fontsize=6.3, pad=1.5, color="#333333")
            for a, b, c, part in runs(att[(l, h)][:, first:], strokes, text) if (l, h) in NAMED else []:
                mid, row, wide = (a + b) / 2, first + c, len(part) > 0
                x = min(max(mid, 5.5 if wide else 0), att[SHOWN[0][0]].shape[0] - (6 if wide else 1))
                ax.annotate(text[c] + part, (mid, row + 0.5 if wide and row < 2 else row - 0.5), xytext=(x, row + 1.7 if wide and row < 2 else row - (1.75 if wide else 1.05)), fontsize=5.3, color="#0b2e5c", ha="center",
                            va="center", arrowprops=dict(arrowstyle="-", lw=0.35, color="#0b2e5c", shrinkA=1.5, shrinkB=0) if wide else None)
            for side in ax.spines.values():
                side.set_linewidth(0.4)
            if k == len(SHOWN) - 1:
                ax.set_xlabel("pen tokens, in order", fontsize=7)
    out = out if CHOSEN == heads.REFERENCE else out.replace("heads", "heads_alt")
    fig.savefig(out, dpi=600)
    fig.savefig(out.replace(".pdf", ".png"), dpi=400)
    print(out)


if __name__ == "__main__":
    main()
