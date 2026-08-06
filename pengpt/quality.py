"""Rank drawings by quality, so a corpus can be filtered to its best fraction.

Crowd-sourced drawing corpora are uneven: Quick, Draw! contains careful cats
next to single scribbles. Keeping only the best few percent means ranking, and
ranking needs *relative* judgements. Asked to rate drawings 1-5 in isolation, a
judge puts almost everything at 3, which gives no resolution in the top tail
where the filtering decision actually happens.

So the pipeline is:

  1. Sample a few hundred drawings and collect pairwise "which is better"
     judgements on them. Any judge works -- a person, a VLM, a classifier.
  2. Fit Bradley-Terry to those comparisons, turning wins and losses into a
     latent quality score per drawing.
  3. Regress those scores onto cheap geometric features, which costs nothing to
     evaluate. The result scores an entire corpus without further judgements.

Step 3 is what makes this affordable: judging is O(sample), not O(corpus).
Quality is graded, so `select` takes a fraction rather than a threshold.
"""

import numpy as np


def features(points):
    """Cheap geometric descriptors of one trajectory.

    Chosen to separate a considered drawing from a scribble without knowing what
    the subject is: how many strokes it took, how much ink relative to size,
    whether it is littered with specks, and how much of the bounding box the ink
    actually visits.
    """
    points = np.asarray(points, dtype=float)
    down = points[points[:, 2] == 1]
    if len(down) < 3:
        return np.zeros(6)

    spans = _stroke_slices(points[:, 2])
    extent = max(np.ptp(down[:, 0]), np.ptp(down[:, 1]), 1e-6)
    lengths = np.array([_arc_length(points[a:b, :2]) for a, b in spans]) if spans else np.zeros(1)
    ink = lengths.sum()

    # how much of the bounding box the ink visits, on a coarse grid
    grid = np.zeros((8, 8), dtype=bool)
    ix = np.clip(((down[:, 0] - down[:, 0].min()) / extent * 7).astype(int), 0, 7)
    iy = np.clip(((down[:, 1] - down[:, 1].min()) / extent * 7).astype(int), 0, 7)
    grid[ix, iy] = True

    return np.array([
        len(spans),                                  # stroke count
        ink / extent,                                # ink relative to size
        float((lengths < 0.05 * extent).sum()),      # speck strokes
        grid.mean(),                                 # spatial coverage
        len(down) / max(ink / extent, 1e-6),         # points per unit ink
        float(np.ptp(down[:, 0]) / max(np.ptp(down[:, 1]), 1e-6)),   # aspect ratio
    ])


def _stroke_slices(pen):
    down = (np.asarray(pen) == 1).astype(np.int8)
    if not down.any():
        return []
    edges = np.diff(np.r_[0, down, 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _arc_length(xy):
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def bradley_terry(n_items, comparisons, iters=200, reg=1e-2):
    """Latent quality from pairwise wins.

    comparisons: iterable of (winner, loser) index pairs. Returns a score per
    item, standardized, where higher is better. The regularizer keeps items with
    few or one-sided comparisons from running off to infinity.
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
        for i in range(n_items):
            partners = np.flatnonzero(played[i])
            if len(partners) == 0:
                continue
            p = 1.0 / (1.0 + np.exp(scores[partners] - scores[i]))
            expected[i] = (played[i, partners] * p).sum()
        grad = wins - expected - reg * scores
        scores += 0.1 * grad
    return (scores - scores.mean()) / (scores.std() + 1e-9)


class QualityScorer:
    """Learns to predict judged quality from geometry, then scores a whole corpus."""

    def __init__(self):
        self.weights = None
        self.mean = None
        self.std = None

    def fit(self, points_list, scores):
        X = np.stack([features(p) for p in points_list])
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        X = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        self.weights, *_ = np.linalg.lstsq(X, np.asarray(scores), rcond=None)
        return self

    def score(self, points_list):
        X = np.stack([features(p) for p in points_list])
        X = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        return X @ self.weights

    def agreement(self, points_list, scores):
        """Spearman correlation with held-out judgements; the honest check."""
        predicted = self.score(points_list)
        a = np.argsort(np.argsort(predicted)).astype(float)
        b = np.argsort(np.argsort(np.asarray(scores))).astype(float)
        return float(np.corrcoef(a, b)[0, 1])


def select(points_list, scorer, fraction=0.05):
    """Indices of the best `fraction` of a corpus, best first."""
    order = np.argsort(scorer.score(points_list))[::-1]
    return order[:max(1, int(round(fraction * len(order))))]
