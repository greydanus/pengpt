import argparse
import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter

import pengpt as P

INK, MUTED, FAINT, BG = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TIME = LinearSegmentedColormap.from_list("time", ["#9ec5f4", "#2a78d6", "#0d366b"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.titlelocation": "left", "axes.titlecolor": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": FAINT,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False})


def recording_attention(self, x, source=None, rotary=None, mask=None, cache=None):
    q = P._split_heads(self.q(x), self.n_head)
    k, v = (P._split_heads(t, self.n_head) for t in self.kv(x if source is None else source).split(x.size(-1), dim=2))
    q, k = self.q_norm(q), self.k_norm(k)
    att = (q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5
    self.weights = att.masked_fill(~mask, float("-inf")).softmax(-1)
    return self.proj(P._merge_heads(self.weights @ v))


def instrument(model):
    hidden = []
    for block in model.blocks:
        block.cross_attn.forward = types.MethodType(recording_attention, block.cross_attn)
        block.register_forward_hook(lambda m, i, o: hidden.append(o.detach()))
    return hidden


def token_paths(st):
    n = st.vocab_size - 4
    return [np.cumsum([P.DIRECTIONS[b] for b in st.expand([v])], 0) if v < n and v not in (st.DOWN, st.UP)
            else np.zeros((0, 2), int) for v in range(st.vocab_size)]


def write(model, ds, text, writer):
    st = ds.stroke_tok
    ctx = torch.from_numpy(ds.encode_text(text))[None]
    w = torch.tensor([P.writer_id(ds, writer)])
    seq = model.generate(torch.tensor([[st.BOS]]), ctx, ds.cfg.max_seq_length - 1, do_sample=False,
                         end_token=st.END, pad_token=st.PAD, writer=w)[0]
    if (seq == st.END).any():
        seq = seq[:int(torch.nonzero(seq == st.END)[0]) + 1]
    with torch.inference_mode():
        logits, _ = model(seq[None], ctx, writer=w)
    att = torch.stack([b.cross_attn.weights[0] for b in model.blocks]).numpy()
    return seq.numpy(), logits[0].softmax(-1).numpy(), att


def trace(st, seq, paths, gap=6):
    pos, word, down = np.zeros((len(seq), 2)), np.zeros(len(seq), int), np.zeros(len(seq), bool)
    p, k, d = np.zeros(2), 0, False
    for t, tok in enumerate(seq):
        if tok == st.WORD:
            k += int(t > 0 and seq[t - 1] != st.WORD)
            p, d = np.zeros(2), False
        elif tok in (st.DOWN, st.UP):
            d = tok == st.DOWN
        elif len(paths[tok]):
            p = p + paths[tok][-1]
        pos[t], word[t], down[t] = p, k, d
    right = 0.0
    for i in range(word.max() + 1):
        ink = pos[(word == i) & down]
        if len(ink):
            shift = right - ink[:, 0].min() + gap * (i > 0)
            pos[word == i, 0] += shift
            right = ink[:, 0].max() + shift
    strokes, cur = [], None
    for t, tok in enumerate(seq):
        if tok == st.DOWN:
            cur = ([pos[t]], [t])
        elif cur and len(paths[tok]):
            cur[0].extend(pos[t - 1] + paths[tok])
            cur[1].extend([t] * len(paths[tok]))
        elif cur and tok in (st.UP, st.WORD, st.END):
            if len(cur[0]) > 1:
                strokes.append((P._smooth_strokes(np.c_[cur[0], np.ones(len(cur[0]))])[:, :2], np.array(cur[1])))
            cur = None
    return pos, down, strokes


def ink_axes(ax, strokes, color=FAINT, lw=3.0):
    for pts, _ in strokes:
        ax.plot(pts[:, 0], -pts[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_attention(text, seq, att, strokes, st, chars, path, label):
    pen = np.array([t not in (st.WORD, st.END, st.PAD, st.BOS) for t in seq])
    a = att[..., :len(chars)]
    ent = -(a * np.log(a + 1e-12)).sum(-1)[:, :, pen].mean(-1)
    l, h = np.unravel_index(np.argmin(ent), ent.shape)
    w = a[l, h]
    fig = plt.figure(figsize=(12, 3.2 + 0.2 * len(chars)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 0.2 * len(chars)], width_ratios=[1, 0.1], hspace=0.25, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    ink_axes(ax, strokes, lw=0.5)
    letter = w.argmax(-1)
    for pts, owner in strokes:
        seg = np.stack([pts[:-1], pts[1:]], 1) * [1, -1]
        ax.add_collection(LineCollection(seg, colors=[(BLUE, ORANGE)[letter[o] % 2] for o in owner[1:]], lw=2.2, capstyle="round"))
    top = -min(pts[:, 1].min() for pts, _ in strokes) + 4
    for c in sorted(set(letter[pen])):
        xs = np.concatenate([pts[owner == t] for pts, owner in strokes for t in np.flatnonzero(letter == c) if (owner == t).any()] or [np.zeros((0, 2))])
        if len(xs) and chars[c].strip():
            ax.text(xs[:, 0].mean(), top, chars[c], ha="center", fontsize=11, color=(BLUE, ORANGE)[c % 2], fontweight="bold")
    ax.set_ylim(-max(pts[:, 1].max() for pts, _ in strokes) - 2, top + 5)
    ax.set_title(f"Which letter each stroke reads  ·  {label}  ·  layer {l + 1}, head {h + 1}")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(w.T, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(chars)), [c if c.strip() else "space" for c in chars], fontsize=7)
    ax.set_xlabel("pen token")
    for t in np.flatnonzero((seq == st.WORD) & (np.r_[0, seq[:-1]] != st.WORD)):
        ax.axvline(t, color=ORANGE, lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(ent, cmap="Blues_r", vmin=0, vmax=max(2.5, ent.max()), aspect="auto")
    for (i, j), e in np.ndenumerate(ent):
        ax.text(j, i, f"{e:.1f}", ha="center", va="center", fontsize=7, color=INK if e > 1.2 else "white")
    ax.set_xticks(range(ent.shape[1]), [f"h{j + 1}" for j in range(ent.shape[1])])
    ax.set_yticks(range(ent.shape[0]), [f"L{i + 1}" for i in range(ent.shape[0])])
    ax.set_title("entropy (nats)", fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def step_density(p, paths, radius=13, res=3, sigma=1.5):
    n = 2 * radius * res + 1
    d = np.zeros((n, n))
    for v in np.flatnonzero(p > 1e-3):
        q = np.vstack([[0, 0], paths[v]])
        pts = np.concatenate([q[i] + (q[i + 1] - q[i]) * np.linspace(0, 1, res, endpoint=False)[:, None] for i in range(len(q) - 1)] + [q[-1:]])
        ij = np.round((pts[1:] + radius) * res).astype(int)
        np.add.at(d, (ij[:, 1], ij[:, 0]), p[v] / len(ij))
    d = gaussian_filter(d, sigma)
    flat = np.sort(d.ravel())[::-1]
    return d, flat[np.searchsorted(np.cumsum(flat), 0.5 * flat.sum())]


def plot_next_pen(seq, probs, pos, down, strokes, st, paths, path, label, zooms=5):
    move = np.array([len(q) > 0 for q in paths])
    steps = [t for t in range(len(seq) - 1) if down[t] and seq[t] != st.WORD and move[seq[t + 1]]]
    norm = Normalize(0, len(seq))
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, zooms, height_ratios=[2.3, 1], hspace=0.2)
    ax = fig.add_subplot(gs[0, :])
    ink_axes(ax, strokes, color=FAINT, lw=2.5)
    radius, res = 13, 3
    grid = np.arange(-radius, radius + 1e-9, 1 / res)
    for t in steps:
        p = probs[t] * move
        d, level = step_density(p / p.sum(), paths, radius, res)
        X, Y = np.meshgrid(pos[t, 0] + grid, -(pos[t, 1] + grid))
        color = TIME(norm(t))
        ax.contourf(X, Y, d, levels=[level, d.max() + 1e-12], colors=[color], alpha=0.35, zorder=2)
        ax.contour(X, Y, d, levels=[level], colors=[color], linewidths=0.6, zorder=3)
    lift = probs[:, st.UP]
    hot = [t for t in range(len(seq) - 1) if down[t] and lift[t] > 0.2]
    ax.scatter(pos[hot, 0], -pos[hot, 1], s=60 * lift[hot], facecolors="none", edgecolors=ORANGE, lw=1.4, zorder=4)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, TIME), ax=ax, fraction=0.012, pad=0.01)
    cb.set_label("pen token", color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title(f"Next-pen distributions (50% region of each step)  ·  orange rings: P(lift)  ·  {label}")
    for i, t in enumerate(np.array(steps)[np.linspace(4, len(steps) - 4, zooms).astype(int)]):
        ax = fig.add_subplot(gs[1, i])
        for pts, _ in strokes:
            ax.plot(pts[:, 0] - pos[t, 0], -(pts[:, 1] - pos[t, 1]), color=FAINT, lw=3, solid_capstyle="round")
        p = probs[t] * move
        p = p / p.sum()
        for rank, v in enumerate(np.argsort(-p)[:4][::-1]):
            q = np.vstack([[0, 0], paths[v]])
            ax.plot(q[:, 0], -q[:, 1], color=TIME(norm(t)), lw=0.8 + 6 * p[v], alpha=0.35 + 0.65 * p[v] / p.max(), solid_capstyle="round")
            if rank == 3:
                ax.text(q[-1, 0], -q[-1, 1], f" {p[v]:.2f}", fontsize=7, color=MUTED, va="center")
        q = np.vstack([[0, 0], paths[seq[t + 1]]])
        ax.plot(q[:, 0], -q[:, 1], color=INK, lw=0.9, ls=(0, (2, 1.5)))
        ax.plot(0, 0, "o", color=INK, ms=3)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"step {t}", fontsize=8, color=MUTED)
    fig.text(0.01, 0.02, "zooms: top 4 candidate strokes, width ∝ probability; dashed = written", fontsize=7, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def ridge_r2(X, y, groups, alpha=1.0, folds=5):
    pred = np.zeros_like(y, dtype=float)
    for f in range(folds):
        test = groups % folds == f
        A = np.c_[X[~test], np.ones((~test).sum())]
        coef = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[~test])
        pred[test] = np.c_[X[test], np.ones(test.sum())] @ coef
    return 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12), pred


def plot_tokens(model, st, paths, path):
    moves = [v for v in range(st.vocab_size) if len(paths[v])]
    E = model.tok_emb.weight.detach().numpy()[moves]
    Z = E - E.mean(0)
    pc = Z @ np.linalg.svd(Z, full_matrices=False)[2][:2].T
    end = np.array([paths[v][-1] for v in moves], float)
    angle = np.arctan2(-end[:, 1], end[:, 0])
    length = np.array([len(paths[v]) for v in moves], float)
    ids = np.arange(len(moves))
    r2 = {name: ridge_r2(E, y, ids)[0] for name, y in (("cos", np.cos(angle)), ("sin", np.sin(angle)), ("length", length))}
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    span = np.ptp(pc, 0).max()
    scale = 0.006 * span
    cmap = plt.get_cmap("twilight")
    for (x, y), v, a in zip(pc, moves, angle):
        q = np.vstack([[0, 0], paths[v]]).astype(float)
        q = (q - q.mean(0)) * scale
        ax.plot(x + q[:, 0], y - q[:, 1], color=cmap((a % (2 * np.pi)) / (2 * np.pi)), lw=1.4 if len(paths[v]) > 1 else 2.4,
                solid_capstyle="round")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 360), cmap), ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("stroke direction (°, 0 = right, 90 = up)", color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title(f"Token embeddings: each token drawn as its stroke\n"
                 f"linear R² from embedding: direction {np.mean([r2['cos'], r2['sin']]):.2f}, length {r2['length']:.2f}")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_probes(model, test, hidden, path):
    st, paths = test.stroke_tok, token_paths(test.stroke_tok)
    items = [test.make_item(st.encode_word(test.bank_points[i]), test.bank_texts[i], int(test.writer_ids[i])) for i in test.indices]
    X, C, Y, W = P.trim(tuple(torch.stack(t) for t in zip(*items)), st.PAD)
    emb = []
    model.embed = (lambda f: lambda idx, xy: emb.append(f(idx, xy)) or emb[-1])(model.embed)
    hidden.clear()
    with torch.inference_mode():
        model(X, C, writer=W)
    layers = [emb[0].numpy()] + [h.numpy() for h in hidden]
    X = X.numpy()
    keep = (X != st.PAD) & (X != st.BOS) & (X != st.END)
    rows = {"x": [], "height": [], "progress": [], "down": []}
    for b in range(len(X)):
        pos, down, _ = trace(st, X[b], paths)
        idx = np.flatnonzero(keep[b])
        rows["x"] += list(pos[idx, 0] / max(1, pos[idx, 0].max()))
        rows["height"] += list(pos[idx, 1])
        rows["progress"] += list(idx / len(idx))
        rows["down"] += list(down[idx].astype(float))
    groups = np.repeat(np.arange(len(X)), X.shape[1])[keep.ravel()]
    names = ["input"] + [f"layer {i + 1}" for i in range(len(hidden))]
    labels = {"height": "pen height", "x": "pen x in word", "progress": "progress through word", "down": "pen down (accuracy)"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ends = []
    for (key, y), color in zip(rows.items(), (BLUE, ORANGE, AQUA, YELLOW)):
        y = np.array(y)
        scores = []
        for H in layers:
            r2, pred = ridge_r2(H.reshape(-1, H.shape[-1])[keep.ravel()], y, groups)
            scores.append(np.mean((pred > 0.5) == (y > 0.5)) if key == "down" else r2)
        ax.plot(range(len(names)), scores, color=color, lw=2, marker="o", ms=5, markeredgecolor=BG)
        ends.append([scores[-1], labels[key], color])
    for i, e in enumerate(sorted(ends)):
        e[0] = max(e[0], sorted(ends)[i - 1][0] + 0.07) if i else e[0]
        ax.text(len(names) - 0.9, e[0], e[1], color=e[2], va="center", fontsize=8)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.2, len(names) + 0.9)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("linear probe R²")
    ax.grid(axis="y", color=FAINT, lw=0.6)
    ax.set_title("What each layer encodes (held-out words)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="interpretability plots for a pengpt checkpoint")
    parser.add_argument("--checkpoint", default="out/cursive/best.pt")
    parser.add_argument("--text", default="goddess countless")
    parser.add_argument("--writer", default="sam")
    parser.add_argument("--out", default="out/interp")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    model, ds, cfg, ckpt = P.load_for_sampling(args.checkpoint)
    model.eval()
    hidden = instrument(model)
    st, paths = ds.stroke_tok, token_paths(ds.stroke_tok)
    label = args.label or os.path.basename(os.path.dirname(args.checkpoint))
    seq, probs, att = write(model, ds, args.text, args.writer)
    pos, down, strokes = trace(st, seq, paths)
    chars = (["writer"] if model.cfg.n_writers else []) + list(args.text)
    plot_attention(args.text, seq, att, strokes, st, chars, f"{args.out}/attention.png", label)
    plot_next_pen(seq, probs, pos, down, strokes, st, paths, f"{args.out}/next_pen.png", label)
    plot_tokens(model, st, paths, f"{args.out}/tokens.png")
    _, test, _, _ = P.create_datasets(cfg, merges=ckpt["merges"], writers=ckpt.get("writers"))
    plot_probes(model, test, hidden, f"{args.out}/probes.png")
    print(f"wrote {args.out}/attention.png, next_pen.png, tokens.png, probes.png")


if __name__ == "__main__":
    main()
