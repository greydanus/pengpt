import argparse
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.ticker import PercentFormatter
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import pengpt as P

sys.path.insert(0, "inference")
import pen as mini

INK, MUTED, FAINT, BG = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.titlelocation": "left", "axes.titlecolor": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": FAINT,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False})


def recording_attention(self, x, source=None, rotary=None, mask=None, cache=None):
    q = P._split_heads(self.q(x), self.n_head)
    k, v = (P._split_heads(t, self.n_head) for t in self.kv(x if source is None else source).split(x.size(-1), dim=2))
    q, k = self.q_norm(q), self.k_norm(k)
    if rotary is not None:
        q, k = P.apply_rotary(q, *rotary), P.apply_rotary(k, *rotary)
    if cache is not None:
        buf, t = cache
        buf.index_copy_(3, t, torch.stack([k, v]))
        k, v = buf
    mask = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool, device=x.device).tril() if mask is None else mask
    self.weights = ((q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5).masked_fill(~mask, float("-inf")).softmax(-1)
    w = self.weights if getattr(self, "off", None) is None else self.weights.index_fill(1, self.off, 0)
    return self.proj(P._merge_heads(w @ v))


def instrument(model):
    hidden = [None] * len(model.blocks)
    for k, block in enumerate(model.blocks):
        for a in (block.attn, block.cross_attn):
            a.forward = types.MethodType(recording_attention, a)
        block.register_forward_hook(lambda m, i, o, k=k: hidden.__setitem__(k, o.detach()))
    return hidden


def token_paths(st):
    n = st.vocab_size - 4
    return [np.cumsum([P.DIRECTIONS[b] for b in st.expand([v])], 0) if v < n and v not in (st.DOWN, st.UP)
            else np.zeros((0, 2), int) for v in range(st.vocab_size)]


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


def plot_attention(seq, att, strokes, st, chars, path, label):
    pen = np.array([t not in (st.WORD, st.END, st.PAD, st.BOS) for t in seq])
    a = att[..., :len(chars)]
    ent = -(a * np.log(a + 1e-12)).sum(-1)[:, :, pen].mean(-1)
    first, last = int(chars[0] == "writer"), len(chars) - int(chars[-1] == "end")
    letters = a[..., first:last]
    order = letters[:, :, pen].argmax(-1)
    reads = letters.sum(-1)[:, :, pen].mean(-1) * (np.diff(order, axis=-1) >= 0).mean(-1)
    l, h = np.unravel_index(np.argmax(reads), reads.shape)
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
        if len(xs) and chars[c].strip() and len(chars[c]) == 1:
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


def plot_heads(att, seq, st, chars, path, label):
    pen = np.array([t not in (st.WORD, st.END, st.PAD, st.BOS) for t in seq])
    a = att[..., :len(chars)][:, :, pen]
    L, H = a.shape[:2]
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H, 1.25 * L), squeeze=False)
    for l in range(L):
        for h in range(H):
            ax, mass = axes[l, h], a[l, h].mean(0)
            ax.imshow(a[l, h].T, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            most = chars[int(mass.argmax())] if mass.max() > 0.5 else "letters in turn" if (np.diff(a[l, h].argmax(-1)) >= 0).mean() > 0.9 else "mixed"
            ax.set_title(f"L{l + 1} h{h + 1}: {most}", fontsize=7, pad=2)
    fig.suptitle(f"Every cross-attention head: what each pen token (across) reads of the writer and the letters (down)  ·  {label}", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
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
    items = [test.make_item(st.encode_word(test.bank_points[i]), test.encode_text(test.bank_texts[i]), int(test.source_ids[i])) for i in test.indices[:1500]]
    X, C, Y, W = P.trim(tuple(torch.stack(t) for t in zip(*items)), st.PAD)
    emb = [None]
    model.embed = (lambda f: lambda idx, xy: emb.__setitem__(0, f(idx, xy)) or emb[0])(model.embed)
    with torch.inference_mode():
        model(X, C, source=W)
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


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd0", "#d6457a", "#4aa3a2", "#7a6a55", "#5b6b8c", "#b0492c", "#0b0b0b"]
HEAT = LinearSegmentedColormap.from_list("heat", [(1, 1, 1, 0), (0.62, 0.77, 0.96, 0.85), (0.16, 0.47, 0.84, 0.95), (0.05, 0.21, 0.42, 1)])
WARM = LinearSegmentedColormap.from_list("warm", ["#dcdbd6", "#f2b28f", "#eb6834", "#7a1f05"])


def ink(ax, points, at=(0, 0), scale=1.0, color=INK, lw=0.8):
    for s in P.to_strokes(points):
        s = s if len(s) > 1 else np.r_[s, s + 1e-3]
        ax.plot(at[0] + scale * s[:, 0], at[1] - scale * s[:, 1], color=color, lw=lw, solid_capstyle="round")


def fitted(points, size):
    lo, hi = points[:, :2].min(0), points[:, :2].max(0)
    return (points - [*(lo + hi) / 2, 0]) * [*[size / max((hi - lo).max(), 1e-6)] * 2, 1]


def bare(ax, title=""):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def mosaic(xy, cols, rows):
    u = (xy - xy.min(0)) / np.ptp(xy, 0)
    cells = np.stack(np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows), -1).reshape(-1, 2)
    i, j = linear_sum_assignment(cdist(u, cells, "sqeuclidean"))
    return (cells[j[np.argsort(i)]] * [cols, rows])


def dots(ax, xy, color, names, counts, title):
    ax.scatter(xy[:, 0], xy[:, 1], s=7, c=color, linewidths=0.2, edgecolors=BG)
    for n, c, k in zip(names, COLORS, counts):
        ax.scatter([], [], s=16, color=c, label=f"{n}  {k:,}")
    bare(ax, title)
    ax.title.set_fontsize(8.5)


def tiles(ax, xy, cols, rows, wide, things, color, lw, title):
    at = mosaic(xy, cols, rows) * [wide, 1]
    for p, c, points in zip(at, color, things):
        if len(points):
            ink(ax, fitted(points, 0.9 * wide) * [1, min(1, 0.92 / max(np.ptp(points[:, 1]) * 0.9 * wide / max(np.ptp(points[:, :2], 0).max(), 1e-6), 1e-6)), 1], p, color=c, lw=lw)
    ax.set_xlim(0, cols * wide)
    ax.set_ylim(0, rows)
    bare(ax, title)
    ax.title.set_fontsize(9)


def plot_writers(pen, path, word="hello", cols=24, rows=50, wide=2.7):
    names = pen.writers[1:]
    xy = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=0).fit_transform(mini.rms(pen.w["writer_emb.weight"][1:]))
    corpus = [n.split("_")[0] if "_" in n else "" for n in names]
    corpora = sorted(set(corpus), key=lambda c: -corpus.count(c))
    color = np.array([COLORS[corpora.index(c)] for c in corpus])
    fig = plt.figure(figsize=(19, 10.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.05], wspace=0.02)
    a = fig.add_subplot(gs[0])
    dots(a, xy, color, corpora, [corpus.count(c) for c in corpora], f"{len(names):,} writer vectors  ·  UMAP, cosine distance  ·  colour: corpus")
    if "sam" in names:
        i = names.index("sam")
        a.annotate("sam", xy[i], xy[i] + [0.8, 0.8], color=INK, fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, handletextpad=0.1, columnspacing=1.2)
    words = [(pen.write(word, n) or [np.zeros((0, 3))])[0] for n in names]
    tiles(fig.add_subplot(gs[1]), xy, cols, rows, wide, words, color, 0.55,
          f"every writer's “{word}”, each moved to the nearest free cell of a grid laid over the map on the left")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.01)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_drawings(model, test, path, device, each=300, layer=8, side=36):
    st = test.stroke_tok
    picked = [(n, i) for n, part in test.parts.items() for i in part.indices[:each]]
    blank, kept = test.encode_text(""), []
    hook = model.blocks[layer - 1].register_forward_hook(lambda m, i, o: kept.append(o))
    pooled = []
    for k in range(0, len(picked), 32):
        items = [test.make_item(st.encode_word(test.bank_points[i]), blank, 0) for _, i in picked[k:k + 32]]
        X, C, _, W = P.trim(tuple(torch.stack(t) for t in zip(*items)), st.PAD)
        kept.clear()
        with torch.inference_mode():
            model(X.to(device), C.to(device), source=W.to(device))
        m = (X != st.PAD).to(device)[..., None]
        pooled.append(((kept[0] * m).sum(1) / m.sum(1)).float().cpu().numpy())
    hook.remove()
    H, names = np.concatenate(pooled), list(test.parts)
    source = np.array([names.index(n) for n, _ in picked])
    level = H - np.stack([H[source == k].mean(0) for k in range(len(names))])[source]
    color = np.array(COLORS)[source]
    shown = np.sort(np.random.default_rng(0).permutation(len(picked))[:side * side])
    things = [test.bank_points[picked[i][1]] for i in shown]
    fig = plt.figure(figsize=(20, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.52, 1, 1], wspace=0.02, hspace=0.12)
    titles = (("as the model holds them: by source", "placed by the model's own activations: drawings sort by where they come from, which is how they are drawn"),
              ("each source's average taken away", "each source's average taken away: drawings sort by what they show"))
    for k, (X, (short, long)) in enumerate(zip((H, level), titles)):
        xy = umap.UMAP(n_neighbors=20, min_dist=0.5, metric="cosine", random_state=0).fit_transform(X)
        a = fig.add_subplot(gs[k, 0])
        dots(a, xy, color, names, np.bincount(source), short)
        tiles(fig.add_subplot(gs[:, 1 + k]), xy[shown], side, side, 1, things, color[shown], 0.4, long)
    a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3, handletextpad=0.1, columnspacing=1.0)
    fig.suptitle(f"{len(picked):,} held-out drawings, shown to the model with no prompt  ·  UMAP of layer-{layer} activations averaged over the pen tokens  ·  "
                 f"right: {len(shown):,} of them, each moved to the nearest free cell of a grid laid over the map", fontsize=9.5, x=0.005, ha="left")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.925, bottom=0.085)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def made(model, ds, text, params, device):
    st = ds.stroke_tok
    new = P.generate(model, ds, text, params)
    end = np.flatnonzero(new == st.END)
    seq = torch.tensor([st.BOS, *new[:end[0] + 1 if end.size else len(new)]])
    seq = seq[seq != st.PAD]
    ctx = torch.from_numpy(ds.encode_text(text))[None].to(device)
    w = torch.tensor([P.source_id(ds, params.source)], device=device)
    with torch.inference_mode():
        logits, _ = model(seq[None].to(device), ctx, source=w)
    return seq.numpy(), logits[0].float().softmax(-1).cpu().numpy(), ctx, w


def futures(model, st, paths, seq, t, pos, down, ctx, w, n, new, stroke, seed=0):
    g = torch.Generator(device=ctx.device).manual_seed(seed)
    out = model.generate(torch.from_numpy(seq[:t + 1]).to(ctx.device).repeat(n, 1), ctx.repeat(n, 1), new, source=w.repeat(n), generator=g)
    lines = []
    for row in out[:, t + 1:].cpu().numpy():
        p, d, cur, drawn = pos[t].copy(), bool(down[t]), [pos[t].copy()], []
        for tok in row:
            if tok in (st.END, st.WORD, st.PAD) or (stroke and tok == st.UP and (drawn or len(cur) > 1)):
                break
            if tok == st.DOWN:
                d, cur = True, [p.copy()]
            elif tok == st.UP:
                drawn += [np.array(cur)] * (d and len(cur) > 1)
                d = False
            elif len(paths[tok]):
                cur += list(p + paths[tok]) if d else []
                p = p + paths[tok][-1]
        lines.append(drawn + [np.array(cur)] * (d and len(cur) > 1))
    return lines


def chance(lines, lo, hi, res=2, sigma=1.2):
    shape = np.ceil((hi - lo) * res).astype(int)[::-1] + 1
    total = np.zeros(shape)
    for strokes in lines:
        hit = np.zeros(shape, bool)
        for s in strokes:
            dense = np.concatenate([s[i] + (s[i + 1] - s[i]) * np.linspace(0, 1, 2 * res, endpoint=False)[:, None] for i in range(len(s) - 1)] + [s[-1:]])
            ij = np.clip(np.round((dense - lo) * res).astype(int), 0, shape[::-1] - 1)
            hit[ij[:, 1], ij[:, 0]] = True
        total += gaussian_filter(hit.astype(float), sigma)
    return total / len(lines)


def bar(ax, x, y, length, label):
    ax.plot([x, x + length], [y, y], color=MUTED, lw=1.2, solid_capstyle="butt", zorder=6)
    ax.annotate(label, (x, y), (0, 2), textcoords="offset points", ha="left", va="bottom", fontsize=6, color=MUTED, zorder=6)


def plot_next(model, ds, text, params, path, device, label, unit, frames=6, n=128, new=48, stroke=True):
    st, paths = ds.stroke_tok, token_paths(ds.stroke_tok)
    seq, probs, ctx, w = made(model, ds, text, params, device)
    pos, down, strokes = trace(st, seq, paths)
    move = np.array([len(q) > 0 for q in paths])
    spots = np.flatnonzero(seq[:-2] == st.UP) if stroke else np.flatnonzero(down[:-2] & move[seq[1:-1]] & move[seq[:-2]])
    spots = spots[np.linspace(0, len(spots) - 1, frames + 2).astype(int)[1:-1]] if stroke else spots[np.linspace(3, len(spots) - 4, frames).astype(int)]
    allpts = np.concatenate([p for p, _ in strokes])
    lo, hi = allpts.min(0) - (9 if stroke else 5), allpts.max(0) + (9 if stroke else 5)
    found = []
    for t in spots:
        p = probs[t] * move
        lines = futures(model, st, paths, seq, t, pos, down, ctx, w, n, new, stroke)
        found.append((t, p, lines, chance(lines, lo, hi, sigma=2.0 if stroke else 1.2)))
    reach = int(max(np.abs(paths[v]).max() for _, p, _, _ in found for v in np.flatnonzero(p > 2e-3))) + 4
    top = max(c.max() for *_, c in found)
    fig = plt.figure(figsize=(14, 4.9) if stroke else (14, 4.3))
    gs = fig.add_gridspec(2, frames + 1, height_ratios=[1, 0.52], width_ratios=[1] * frames + [0.05], hspace=0.12, wspace=0.05) if stroke else \
        fig.add_gridspec(frames // 2, 5, width_ratios=[5, 1.15, 5, 1.15, 0.1], hspace=0.3, wspace=0.04)
    for k, (t, p, lines, c) in enumerate(found):
        ax = fig.add_subplot(gs[0, k] if stroke else gs[k // 2, 2 * (k % 2)])
        for pts, owner in strokes:
            ax.plot(pts[:, 0], -pts[:, 1], color=FAINT, lw=0.9, solid_capstyle="round", zorder=1)
            seen = pts[owner <= t]
            ax.plot(seen[:, 0], -seen[:, 1], color=INK, lw=1.1, solid_capstyle="round", zorder=4)
        ax.imshow(c, extent=(lo[0], hi[0] + 0.5, -hi[1] - 0.5, -lo[1]), origin="upper", cmap=HEAT, vmin=0, vmax=top, zorder=3)
        if stroke:
            starts = np.array([s[0][0] for s in lines if s])
            ax.scatter(starts[:, 0], -starts[:, 1], s=5, color=ORANGE, alpha=0.45, linewidths=0, zorder=5)
        ax.plot(pos[t, 0], -pos[t, 1], "o", color=ORANGE, ms=4, markeredgecolor=BG, markeredgewidth=0.6, zorder=6)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(-hi[1], -lo[1])
        bare(ax)
        ax.set_title(f"after stroke {(seq[:t + 1] == st.UP).sum()} of {(seq == st.UP).sum()}" if stroke else f"after token {t} of {len(seq) - 2}", fontsize=7.5, color=MUTED, pad=2)
        if k == 0:
            bar(ax, lo[0] + 2, -hi[1] + 2, unit[0], unit[1])
        ax = fig.add_subplot(gs[1, k] if stroke else gs[k // 2, 2 * (k % 2) + 1])
        d, _ = step_density(p / p.sum(), paths, reach, 3, 1.5)
        for pts, owner in strokes:
            seen = pts[owner <= t]
            ax.plot(seen[:, 0], -seen[:, 1], color=INK, lw=1.3, solid_capstyle="round")
        ax.imshow(d, extent=(pos[t, 0] - reach, pos[t, 0] + reach, -pos[t, 1] - reach, -pos[t, 1] + reach), origin="upper", cmap=HEAT, zorder=3, vmax=np.quantile(d, 0.999))
        ax.plot(pos[t, 0], -pos[t, 1], "o", color=ORANGE, ms=4, markeredgecolor=BG, markeredgewidth=0.6, zorder=4)
        ax.set_xlim(pos[t, 0] - reach, pos[t, 0] + reach)
        ax.set_ylim(-pos[t, 1] - reach, -pos[t, 1] + reach)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set(visible=True, color=FAINT)
        other = f"touch {probs[t, st.DOWN]:.2f}" if not down[t] else f"lift {probs[t, st.UP]:.2f}"
        ax.set_title(f"next token · {other}", fontsize=6.5, color=MUTED, pad=2)
        if k == 0:
            bar(ax, pos[t, 0] - reach + 1.5, -pos[t, 1] - reach + 1.5, 5, "5 grid steps")
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, top), HEAT), cax=fig.add_subplot(gs[0, -1] if stroke else gs[:, -1]), format=PercentFormatter(1, 0 if top > 0.1 else 1))
    cb.set_label(f"chance the next {'stroke' if stroke else f'{new} tokens'} ink{'s' * stroke} a spot", color=MUTED, fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Where the pen goes next  ·  {label}, “{text}”  ·  black: drawn so far, grey: what came after, orange: the pen{', and where each sampled next stroke starts' * stroke}\n"
                 f"large panels: {n} sampled continuations  ·  boxes: every possible next token laid where it would land, weighted by its probability",
                 fontsize=8.5, x=0.01, y=0.995, ha="left", va="top")
    fig.subplots_adjust(left=0.01, right=0.955, top=0.86 if stroke else 0.84, bottom=0.02)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def held(test, n, device, size=32):
    X, C, Y, W = P.trim(test.batch(range(n)), test.stroke_tok.PAD)
    return [tuple(t[k:k + size].to(device) for t in (X, C, Y, W)) for k in range(0, n, size)]


def scored(model, batches):
    out = []
    with torch.inference_mode():
        for X, C, Y, W in batches:
            lp = model(X, C, source=W)[0].float().log_softmax(-1)
            out.append(-lp.gather(-1, Y.clamp(min=0)[..., None])[..., 0].cpu().numpy())
    return np.concatenate(out)


ROLES = ["travel, pen up", "touch down", "first move of a stroke", "rest of the stroke", "lift", "end"]


def roles(st, X, Y, move):
    down, d = np.zeros(X.shape, bool), np.zeros(len(X), bool)
    for t in range(X.shape[1]):
        d = np.where(X[:, t] == st.DOWN, True, np.where(np.isin(X[:, t], (st.UP, st.WORD, st.BOS)), False, d))
        down[:, t] = d
    m, r = move[np.clip(Y, 0, None)] & (Y >= 0), np.full(Y.shape, -1)
    r[m & ~down], r[Y == st.DOWN], r[m & down & (X == st.DOWN)], r[m & down & (X != st.DOWN)], r[Y == st.UP] = 0, 1, 2, 3, 4
    r[np.isin(Y, (st.END, st.WORD))] = 5
    return r


def plot_surprise(cases, path, bins=10):
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 3.3 * len(cases)), gridspec_kw=dict(width_ratios=[1.25, 1, 1.25]), squeeze=False)
    for row, (label, without, model, test, batches, blind) in zip(axes, cases):
        st, paths = test.stroke_tok, token_paths(test.stroke_tok)
        move = np.array([len(q) > 0 for q in paths])
        X, Y = (torch.cat([b[i] for b in batches]).cpu().numpy() for i in (0, 2))
        told, untold, r = scored(model, batches), scored(model, blind), roles(st, X, Y, move)
        real = r >= 0
        ax, y = row[0], np.arange(len(ROLES))[::-1]
        a, b = ([v[r == k].mean() if (r == k).any() else 0 for k in range(len(ROLES))] for v in (told, untold))
        ax.barh(y + 0.2, b, 0.38, color=FAINT, label=without)
        ax.barh(y - 0.2, a, 0.38, color=BLUE, label="as trained")
        for k in range(len(ROLES)):
            ax.text(max(a[k], b[k]) + 0.03, y[k], f"{told[r == k].sum() / told[real].sum():.0%} of all nats, {(r == k).sum() / real.sum():.0%} of tokens", va="center", fontsize=6.5, color=MUTED)
        ax.set_yticks(y, ROLES)
        ax.set_xlim(0, 1.75 * max(max(a), max(b)))
        ax.set_xlabel("nats per token, held out")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.set_title(f"{label}: where the uncertainty is  ·  {told[real].mean():.2f} nats per token")
        length = real.sum(1, keepdims=True)
        at = np.minimum((np.arange(X.shape[1])[None] * bins // np.maximum(length, 1)), bins - 1)
        ax, mid = row[1], (np.arange(bins) + 0.5) / bins
        a, b = ([v[real & (at == k)].mean() for k in range(bins)] for v in (told, untold))
        ax.plot(mid, b, color=MUTED, lw=1.5, ls=(0, (3, 2)), label=without)
        ax.plot(mid, a, color=BLUE, lw=2, label="as trained")
        ax.fill_between(mid, a, b, color=BLUE, alpha=0.12, lw=0)
        ax.set_ylim(0, None)
        ax.set_xlabel("how far through the sequence")
        ax.set_ylabel("nats per token")
        ax.legend(frameon=False, fontsize=7)
        gain = (untold - told)[real]
        ax.set_title(f"what it is told is worth {gain.mean():.2f} nats a token;\n{(untold - told)[real & (at < bins // 5)].sum() / gain.sum():.0%} of that in the first fifth", fontsize=8.5)
        ax, i = row[2], int(np.argsort(length[:, 0])[len(X) // 2])
        seq = X[i][:int(length[i, 0]) + 1]
        cost = np.r_[0, told[i]][:len(seq)]
        pos, _, strokes = trace(st, seq, paths)
        for pts, owner in strokes:
            seg = np.stack([pts[:-1], pts[1:]], 1) * [1, -1]
            ax.add_collection(LineCollection(seg, colors=WARM(np.clip(cost[owner[1:]] / 8, 0, 1)), lw=1.8, capstyle="round"))
        waiting = 0.0
        for t, tok in enumerate(seq):
            waiting = waiting + cost[t] if tok not in (st.DOWN,) and not any((owner == t).any() for _, owner in strokes) else waiting
            if tok == st.DOWN:
                ax.scatter(pos[t, 0], -pos[t, 1], s=6 * (waiting + cost[t]), facecolors="none", edgecolors=BLUE, lw=0.9, zorder=5)
                waiting = 0.0
        ax.autoscale()
        bare(ax)
        ax.set_title("one held-out example: nats each token cost;\nring area: nats spent travelling to a stroke's start", fontsize=8.5)
        cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 8), WARM), ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("nats", color=MUTED, fontsize=7)
        cb.ax.tick_params(labelsize=6.5, length=2)
        cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_knockout(cases, path):
    fig, axes = plt.subplots(1, 2 * len(cases), figsize=(3.6 * 2 * len(cases), 5.2), squeeze=False)
    count = lambda batches: sum((Y >= 0).sum().item() for _, _, Y, _ in batches)
    for k, (label, model, batches) in enumerate(cases):
        dev = batches[0][0].device
        instrument(model)
        base = P.nll(model, batches, dev) / count(batches)
        costs = []
        for kind in ("cross_attn", "attn"):
            cost = np.zeros((model.cfg.n_layer, model.cfg.n_head))
            for (l, h), _ in np.ndenumerate(cost):
                a = getattr(model.blocks[l], kind)
                a.off = torch.tensor([h], device=dev)
                with torch.inference_mode():
                    cost[l, h] = P.nll(model, batches, dev) / count(batches) - base
                a.off = None
            costs.append(cost)
        top = max(c.max() for c in costs)
        for j, cost in enumerate(costs):
            ax = axes[0, 2 * k + j]
            ax.imshow(cost, cmap="Oranges", norm=PowerNorm(0.5, 0, top), aspect="auto")
            for (l, h), c in np.ndenumerate(cost):
                ax.text(h, l, f"{c:.2f}"[1 if 0 <= c < 1 else 0:] if abs(c) >= 0.005 else "·", ha="center", va="center", fontsize=6.5, color="white" if c > 0.4 * top else INK)
            ax.set_xticks(range(cost.shape[1]), [f"h{h + 1}" for h in range(cost.shape[1])])
            ax.set_yticks(range(cost.shape[0]), [f"L{l + 1}" for l in range(cost.shape[0])])
            ax.tick_params(length=0, labelsize=7)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{label}: heads that read {'what it was told' if j == 0 else 'its own ink'}" + f"\n{base:.2f} nats per token with every head" * (j == 0), fontsize=8.5, loc="left")
    fig.suptitle("Silencing one attention head at a time: the nats per held-out token it costs  ·  one colour scale a model, square-root spaced  ·  a dot is under 0.005", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_looks(model, ds, text, params, path, device, label, layers=(2, 5, 8, 11), frames=6):
    st, paths = ds.stroke_tok, token_paths(ds.stroke_tok)
    seq, _, _, _ = made(model, ds, text, params, device)
    pos, down, strokes = trace(st, seq, paths)
    att = [model.blocks[l - 1].attn.weights[0].float().cpu().numpy() for l in layers]
    marks = np.flatnonzero(down)
    marks = marks[np.linspace(0, len(marks) - 1, frames + 1).astype(int)[1:]]
    fig, axes = plt.subplots(len(layers), frames, figsize=(2.6 * frames, 2.6 * len(layers)), squeeze=False)
    for r, a in enumerate(att):
        for k, t in enumerate(marks):
            ax, weight = axes[r, k], a[:, t].max(0)
            weight = weight / max(weight[1:t].max(), 1e-9)
            for pts, owner in strokes:
                seg = np.stack([pts[:-1], pts[1:]], 1) * [1, -1]
                past = owner[1:] < t
                ax.add_collection(LineCollection(seg[~past], colors="#eeede9", lw=0.8, capstyle="round"))
                ax.add_collection(LineCollection(seg[past], colors=WARM(np.clip(weight[owner[1:][past]], 0, 1)), lw=1.6, capstyle="round"))
            ax.plot(pos[t, 0], -pos[t, 1], "o", color="#2a78d6", ms=4, zorder=5)
            ax.autoscale()
            bare(ax)
            ax.set_title(f"layer {layers[r]}, token {t}" if k == 0 else f"token {t}", fontsize=7, color=MUTED)
    fig.suptitle(f"What the pen looks back at  ·  {label}: “{text}”\nblue dot: where the pen is  ·  earlier ink is darker the more any self-attention head of that layer reads it",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hand_blends(pen, pairs, path, text="quick brown fox", steps=7):
    fig, axes = plt.subplots(steps, len(pairs), figsize=(5.2 * len(pairs), 0.85 * steps + 0.6), squeeze=False)
    E = mini.rms(pen.w["writer_emb.weight"])
    for c, (a, b) in enumerate(pairs):
        for r, m in enumerate(np.linspace(0, 1, steps)):
            words = mini.layout(pen.write(text, (1 - m) * E[pen.writers.index(a)] + m * E[pen.writers.index(b)]))
            ax = axes[r, c]
            for p in words:
                ink(ax, p, color=tuple((1 - m) * np.array([0.16, 0.47, 0.84]) + m * np.array([0.92, 0.41, 0.2])), lw=1.1)
            bare(ax)
            ax.text(-0.02, 0.5, a if r == 0 else b if r == steps - 1 else f"{m:.2f}", transform=ax.transAxes, ha="right", va="center", fontsize=7, color=MUTED)
    fig.suptitle("One hand turning into another: the writer vector slides from the top writer to the bottom one, most likely ink at every step", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def slerp(a, b, m):
    w = np.arccos(np.clip(a @ b, -1, 1))
    return (np.sin((1 - m) * w) * a + np.sin(m * w) * b) / np.sin(w)


def plot_sketch_blends(pen, pairs, path, steps=7, temperature=0.5, seed=3, pushes=(0, 1, 1.5, 2, 3, 5, 8), pushed="a bicycle"):
    fig, axes = plt.subplots(len(pairs) + 1, steps, figsize=(1.9 * steps, 2.0 * (len(pairs) + 1)), squeeze=False)
    for r, (a, b) in enumerate(pairs):
        va, vb = pen.told(a), pen.told(b)
        for k, m in enumerate(np.linspace(0, 1, steps)):
            ink(axes[r, k], pen.draw(slerp(va, vb, m), "", temperature, None, np.random.default_rng(seed)), lw=0.9)
            bare(axes[r, k])
            axes[r, k].set_title(a if k == 0 else b if k == steps - 1 else f"{m:.2f}", fontsize=7, color=MUTED)
    for k, g in enumerate(pushes):
        ink(axes[-1, k], pen.draw(pushed, "", temperature, None, np.random.default_rng(seed), guidance=g), lw=0.9)
        bare(axes[-1, k])
        axes[-1, k].set_title(f"“{pushed}”, guidance {g}" if k == 0 else f"guidance {g}", fontsize=7, color=MUTED)
    fig.suptitle("One drawing turning into another: the SigLIP 2 vector the model is told slides from the left words to the right ones, same random draws\n"
                 "last row: one prompt, pushed harder towards its words (0 ignores them)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="figures of what the two models learned: which letters a stroke reads, maps of writers and drawings, where the pen goes next, where the nats go, what each head is worth, blends")
    p.add_argument("--hand", default="hand.pt")
    p.add_argument("--sketch", default="sketch.pt")
    p.add_argument("--hand_packed", default="hand.npz")
    p.add_argument("--sketch_packed", default="sketch.npz")
    p.add_argument("--words", default="siglip-text.npz")
    p.add_argument("--only", default="attention,tokens,probes,writers,hand_blends,hand_next,drawings,sketch_next,surprise,knockout,looks,sketch_blends")
    p.add_argument("--text", default="the pen remembers")
    p.add_argument("--prompt", default="a cat")
    p.add_argument("--writer", default="sam")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="out/interp")
    args = p.parse_args()
    only, out = args.only.split(","), args.out
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(4)
    if {"writers", "hand_blends"} & set(only):
        pen = mini.Pen(args.hand_packed)
        if "writers" in only:
            plot_writers(pen, f"{out}/writers.png")
        if "hand_blends" in only:
            rng = np.random.default_rng(args.seed)
            others = [str(n) for n in rng.choice(pen.writers[1:], 5, replace=False)]
            plot_hand_blends(pen, [(args.writer, others[0]), (others[1], others[2]), (others[3], others[4])], f"{out}/hand_blends.png")
    surprise, knock = [], []
    if {"attention", "tokens", "probes", "hand_next", "surprise", "knockout"} & set(only):
        model, ds, cfg, ckpt = P.load_for_sampling(args.hand)
        model.eval()
        params = P.SampleParams(do_sample=False, source=args.writer, max_tokens=cfg.max_seq_length - 1)
        if "hand_next" in only:
            plot_next(model, ds, args.text, params, f"{out}/hand_next.png", "cpu", "handwriting", (0.123 / cfg.grid, "x-height"), new=24, stroke=False)
        if {"probes", "surprise", "knockout"} & set(only):
            test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
        if {"attention", "tokens", "probes"} & set(only):
            hidden, st, paths = instrument(model), ds.stroke_tok, token_paths(ds.stroke_tok)
            if "attention" in only:
                seq = made(model, ds, args.text, params, "cpu")[0]
                att = torch.stack([b.cross_attn.weights[0] for b in model.blocks]).numpy()
                chars = ["writer"] * bool(model.cfg.n_sources) + list(args.text) + ["end"] * bool(cfg.text_end)
                plot_attention(seq, att, trace(st, seq, paths)[2], st, chars, f"{out}/attention.png", "handwriting")
                plot_heads(att, seq, st, chars, f"{out}/heads.png", "handwriting")
            if "tokens" in only:
                plot_tokens(model, st, paths, f"{out}/tokens.png")
            if "probes" in only:
                plot_probes(model, test, hidden, f"{out}/probes.png")
        if {"surprise", "knockout"} & set(only):
            batches = held(test, 384, "cpu", 128)
            surprise.append(("handwriting", "not told the writer", model, test, batches, [(X, C, Y, torch.zeros_like(W)) for X, C, Y, W in batches]))
            knock.append(("handwriting", model, batches))
    if {"drawings", "sketch_next", "looks", "surprise", "knockout"} & set(only):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, ds, cfg, ckpt = P.load_for_sampling(args.sketch, device)
        model.eval()
        params = P.SampleParams(temperature=0.6, max_tokens=cfg.max_seq_length - 1, seed=args.seed)
        if {"drawings", "surprise", "knockout"} & set(only):
            test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
        if "drawings" in only:
            plot_drawings(model, test, f"{out}/drawings.png", device)
        if "sketch_next" in only:
            torch.manual_seed(args.seed)
            plot_next(model, ds, args.prompt, params, f"{out}/sketch_next.png", device, "drawings", (0.1 / cfg.grid, "a tenth of the drawing"))
        if {"surprise", "knockout"} & set(only):
            batches = held(test, 256, device, 32)
            blank = torch.from_numpy(test.encode_text("")).to(device)
            surprise.append(("drawings", "not told what to draw", model, test, batches, [(X, blank.expand(len(X), -1), Y, W) for X, C, Y, W in batches]))
            knock.append(("drawings", model, batches[:3]))
    if "surprise" in only:
        plot_surprise(surprise, f"{out}/surprise.png")
    if "knockout" in only:
        plot_knockout(knock, f"{out}/knockout.png")
    if "looks" in only:
        instrument(model)
        torch.manual_seed(args.seed)
        plot_looks(model, ds, args.prompt, params, f"{out}/looks.png", device, "drawings")
    if "sketch_blends" in only:
        from words import Words
        pen = mini.Pen(args.sketch_packed)
        pen.encoder = Words(args.words)
        plot_sketch_blends(pen, [("a cat", "a bicycle"), ("a house", "a sailing boat"), ("a fish", "a bird")], f"{out}/sketch_blends.png")
    print(f"wrote {', '.join(only)} to {out}/")


if __name__ == "__main__":
    main()
