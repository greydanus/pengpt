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

Show a judge fifteen drawings at once and ask for an ordering, rather than
asking about pairs: one look orders the batch where pairwise needs hundreds of
questions. Coarse tiers beat a strict total order, since "these five are
excellent and those five are junk" is reliable where "is #43 better than #44" is
noise. JUDGE_RUBRIC is the criteria those judgements should follow.

Calibrated on 210 drawings judged this way, the probe is a good coarse filter
and a poor fine ranking, which is all filtering needs. Held out it ranks +0.55
against the judged tiers; the quarter it keeps is 83% good-or-better against a
48% base rate with none of the 25 judged junk drawings surviving; and it agrees
with the judge on 86% of pairs two tiers apart but only 71% of adjacent pairs.

Three things were tried and dropped, recorded here so they are not retried:

  - Quick, Draw!'s own `recognized` flag as a training target. It measures
    whether a classifier saw the expected category, not whether the drawing is
    good, and agrees with judged quality at only +0.10 -- it rejects careful
    full-body cats for not being cat faces.
  - Hand-built stroke geometry (stroke count, ink ratio, coverage) instead of an
    embedding: +0.13 against judged quality where CLIP reaches +0.55, and it
    lets four of 25 junk drawings through the cut.
  - A probe per class rather than one shared probe. It looked better on 90
    judgements and worse on 210, where car fell from +0.36 shared to +0.18
    per-class: too few examples per class for 512 dimensions.
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


def render(points, px=64, linewidth=2, supersample=8, pad=2):
    """Draw one trajectory as a square image, ready to embed.

    Small and thin wins, which is not obvious: 64px with fine strokes ranks
    +0.59 against judged quality, where the same drawings rendered with thick
    strokes score +0.19. Heavy strokes merge a dog's legs and fill in a cat's
    face, destroying the detail the ranking depends on. Bigger canvases and
    inverted polarity do not help either.

    Antialiasing does matter, so this draws large and downsamples rather than
    rasterizing directly at 64px, which costs ranking quality (+0.47). Going
    through PIL rather than matplotlib is 13x faster for the same result, and
    rendering is otherwise half the cost of scoring a corpus.
    """
    from PIL import Image, ImageDraw

    points = np.asarray(points, dtype=float)
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        return Image.new("RGB", (px, px), "white")

    big = px * supersample
    margin = pad * supersample
    x0, y0 = down[:, 0].min(), down[:, 1].min()
    scale = (big - 2 * margin) / max(np.ptp(down[:, 0]), np.ptp(down[:, 1]), 1e-6)

    image = Image.new("L", (big, big), 255)
    draw = ImageDraw.Draw(image)
    for chunk in np.split(points, np.flatnonzero(points[:, 2] == 0) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            draw.line([(margin + (x - x0) * scale, margin + (y - y0) * scale)
                       for x, y in chunk[:, :2]],
                      fill=0, width=int(linewidth * supersample))
    return image.resize((px, px), Image.LANCZOS).convert("RGB")


class Embedder:
    """Vision embeddings for rendered drawings.

    CLIP is the default. Measured against hand-judged tiers it reaches spearman
    +0.59 where dinov2-small reaches +0.44 and dinov2-base +0.30, which is what
    you would expect: CLIP's training data is full of line art and clip art,
    while DINOv2's is photographs.
    """

    def __init__(self, name="openai/clip-vit-base-patch32", device=None, px=64,
                 linewidth=1.0, fp16=False):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.is_clip = "clip" in name.lower()
        self.fp16 = fp16 and self.device != "cpu"
        if self.fp16:
            self.model = self.model.half()
        self.px = px
        self.linewidth = linewidth

    def embed(self, points_list, batch_size=64, verbose=False):
        images = [render(p, self.px, self.linewidth) for p in points_list]
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = self.processor(images=images[i:i + batch_size],
                                       return_tensors="pt").to(self.device)
                if self.fp16:
                    batch["pixel_values"] = batch["pixel_values"].half()
                feats = (self.model.get_image_features(**batch) if self.is_clip
                         else self.model(**batch).pooler_output)
                out.append(feats.float().cpu().numpy())
                if verbose:
                    print(f"  embedded {min(i + batch_size, len(images))}/{len(images)}",
                          flush=True)
        return np.concatenate(out)


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

    Fitted on 210 drawings ordered into quality tiers by hand, embedded with
    CLIP at 64px. Held out it ranks +0.55 against those tiers, and the quarter
    it keeps is 83% good-or-better against a 48% base rate, with none of the 25
    judged junk drawings surviving. Re-fit rather than reuse this if you change
    the embedder or the render settings, which the file records.
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
