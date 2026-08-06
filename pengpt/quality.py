"""Rank drawings by quality, so a crowd-sourced corpus can be filtered.

Quick, Draw! contains careful cats next to single scribbles, and keeping only
the best quarter means ranking. Ranking needs *relative* judgements: asked to
rate drawings in isolation a judge puts almost everything in the middle, leaving
no resolution in the top tail where the filtering decision happens.

    1. Render a sample of drawings and embed them with a vision model.
    2. Have a judge order that sample, and fit Bradley-Terry to the resulting
       comparisons, turning wins and losses into a latent quality score.
    3. Fit a linear probe from embedding to score. Embedding the rest of the
       corpus is cheap and needs no further judgements.

Judging is O(sample) rather than O(corpus), which is what makes this affordable
at 50M drawings. Filtering is per class so class balance survives.

Show a judge many drawings at once and ask for an ordering, rather than asking
about pairs: one call orders thirty items where pairwise needs hundreds. Coarse
tiers beat a strict total order, since "these five are excellent and those five
are junk" is reliable where "is #43 better than #44" is noise. JUDGE_RUBRIC is
the criteria those judgements should follow.

Do not train the probe on Quick, Draw!'s own `recognized` flag. It measures
whether a classifier saw the expected category, not whether the drawing is good,
and it correlates with human quality judgements at only +0.10 -- it rejects
careful full-body cats for not being cat faces.

Geometric features are kept as a fallback for when no vision model is available;
they are much weaker (+0.24 against judged quality, against +0.42 for DINOv2).
"""

JUDGE_RUBRIC = """Order these drawings from best to worst as training examples.

Reward: the subject is recognizable, its parts are present and connected (a dog
with legs and ears, a car with wheels and windows), and the strokes are
deliberate.

Penalize, hardest first:
  - any drawing with letters or words written on it, however neat the rest is
  - scribbles, and drawings that are one line or a few disconnected fragments
  - missing essential parts, so the subject is only guessable from the label

Words on the canvas need to be called out explicitly like this. A vision
embedding ranks such drawings anywhere from the 22nd to the 76th percentile,
because a cat with "MEOW" written over it still looks largely like a cat, and a
stroke-geometry rule that catches them flags a fifth of clean drawings too. The
judge is the only reliable filter for writing, so the rubric has to ask.
"""

import numpy as np


def render(points, px=64, linewidth=1.0):
    """Draw one trajectory as a square greyscale image.

    64 pixels is enough to tell a considered drawing from a scribble -- above
    that only the line weight changes -- and small images keep embedding cheap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    points = np.asarray(points, dtype=float)
    fig, ax = plt.subplots(figsize=(px / 100, px / 100), dpi=100)
    down = points[:, 2] == 1
    for chunk in np.split(points, np.flatnonzero(~down) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            ax.plot(chunk[:, 0], -chunk[:, 1], "k-", linewidth=linewidth,
                    solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((px, px))


class DinoEmbedder:
    """Vision embeddings for rendered drawings."""

    def __init__(self, name="facebook/dinov2-small", device=None, px=64):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.px = px

    def embed(self, points_list, batch_size=64, verbose=False):
        images = [render(p, self.px) for p in points_list]
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = self.processor(images=images[i:i + batch_size],
                                       return_tensors="pt").to(self.device)
                out.append(self.model(**batch).pooler_output.float().cpu().numpy())
                if verbose:
                    print(f"  embedded {min(i + batch_size, len(images))}/{len(images)}",
                          flush=True)
        return np.concatenate(out)


def features(points):
    """Cheap geometric descriptors, for use without a vision model."""
    points = np.asarray(points, dtype=float)
    down = points[points[:, 2] == 1]
    if len(down) < 3:
        return np.zeros(6)

    spans = _stroke_slices(points[:, 2])
    extent = max(np.ptp(down[:, 0]), np.ptp(down[:, 1]), 1e-6)
    lengths = np.array([_arc_length(points[a:b, :2]) for a, b in spans]) if spans else np.zeros(1)
    ink = lengths.sum()

    grid = np.zeros((8, 8), dtype=bool)
    ix = np.clip(((down[:, 0] - down[:, 0].min()) / extent * 7).astype(int), 0, 7)
    iy = np.clip(((down[:, 1] - down[:, 1].min()) / extent * 7).astype(int), 0, 7)
    grid[ix, iy] = True

    return np.array([
        len(spans),
        ink / extent,
        float((lengths < 0.05 * extent).sum()),
        grid.mean(),
        len(down) / max(ink / extent, 1e-6),
        float(np.ptp(down[:, 0]) / max(np.ptp(down[:, 1]), 1e-6)),
    ])


def _stroke_slices(pen):
    down = (np.asarray(pen) == 1).astype(np.int8)
    if not down.any():
        return []
    edges = np.diff(np.r_[0, down, 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _arc_length(xy):
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def bradley_terry(n_items, comparisons, iters=300, reg=1e-2, lr=0.1):
    """Latent quality from pairwise wins.

    comparisons: (winner, loser) index pairs. Returns one standardized score per
    item, higher is better. The regularizer keeps items with few or one-sided
    comparisons from running away.
    """
    wins = np.zeros(n_items)
    played = np.zeros((n_items, n_items))
    for w, l in comparisons:
        wins[w] += 1
        played[w, l] += 1
        played[l, w] += 1

    scores = np.zeros(n_items)
    for _ in range(iters):
        expected = np.zeros(n_items)
        for i in np.flatnonzero(played.any(1)):
            partners = np.flatnonzero(played[i])
            p = 1.0 / (1.0 + np.exp(np.clip(scores[partners] - scores[i], -30, 30)))
            expected[i] = (played[i, partners] * p).sum()
        scores += lr * (wins - expected - reg * scores)
    return (scores - scores.mean()) / (scores.std() + 1e-9)


class LinearProbe:
    """Ridge regression from embedding to judged quality."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = self.mean = self.std = None

    def fit(self, X, scores):
        X = np.asarray(X, dtype=float)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.weights = np.linalg.solve(A, Z.T @ np.asarray(scores, dtype=float))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        return Z @ self.weights

    def agreement(self, X, scores):
        """Spearman against held-out judgements; the number that matters."""
        return spearman(self.score(X), np.asarray(scores))


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_probe(path="data/quickdraw_probe.npz"):
    """A probe already calibrated on judged Quick, Draw! drawings.

    Fitted on 90 drawings ordered into quality tiers by hand, embedded with
    dinov2-small at 64px. Five-fold cross-validation against those tiers gives
    spearman +0.42, against +0.24 for geometric features. Re-fit rather than
    reuse this if you change the embedder or the render size.
    """
    data = np.load(path)
    probe = LinearProbe(alpha=float(data["alpha"]))
    probe.weights, probe.mean, probe.std = data["weights"], data["mean"], data["std"]
    return probe


def select_per_class(scores, labels, fraction=0.25):
    """Indices of the best `fraction` within each class, so balance survives.

    Filtering a whole corpus at once would keep whichever classes the scorer
    happens to rate highly and drop the rest.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    keep = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        order = idx[np.argsort(scores[idx])[::-1]]
        keep.append(order[:max(1, int(round(fraction * len(order))))])
    return np.sort(np.concatenate(keep))
