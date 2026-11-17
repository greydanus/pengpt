import numpy as np

THETA = np.linspace(-np.pi, np.pi, 220)
RADII = np.concatenate([[0], np.linspace(0.0001, 0.060, 30), np.geomspace(0.06001, 0.90, 120)])


def nearest(values, bins):
    i = np.searchsorted(bins, values).clip(1, len(bins) - 1)
    return i - (values - bins[i - 1] < bins[i] - values)


class PolarTokenizer:

    def __init__(self, grid, every, moves):
        self.grid, self.every, self.moves, self.merges = grid, every, moves, []
        n = len(THETA) + 2 * len(RADII)
        self.PAD, self.END, self.WORD, self.BOS = n, n + 1, n + 2, n + 3
        self.vocab_size = n + 4

    def token_deltas(self):
        return np.zeros((self.vocab_size, 2), dtype=np.int64)

    def thin(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        down = points[:, 2] == 1
        start = np.flatnonzero(down & ~np.r_[False, down[:-1]])
        index = np.arange(len(points))
        along = index - np.maximum.accumulate(np.where(np.isin(index, start), index, 0))
        return points[~down | (along % self.every == 0) | np.r_[~down[1:], True]]

    def encode_word(self, points):
        p = self.thin(points)
        if not len(p):
            return np.zeros(0, dtype=np.int64)
        d = np.diff(p[:, :2], axis=0, prepend=np.zeros((1, 2)))
        r = len(THETA) + nearest(np.hypot(*d.T), RADII) + len(RADII) * (p[:, 2] != 1)
        return np.c_[nearest(np.arctan2(d[:, 1], d[:, 0]), THETA), r].ravel().astype(np.int64)

    def encode_many(self, words):
        return [self.encode_word(w) for w in words]

    def before_end(self, tokens):
        tokens = np.asarray(tokens)
        end = np.flatnonzero(tokens == self.END)
        return tokens[:end[0]] if end.size else tokens

    def decode(self, tokens):
        tokens = self.before_end(tokens)
        return self.decode_word(tokens[tokens < self.PAD])

    def decode_words(self, tokens):
        tokens = self.before_end(tokens)
        tokens = tokens[(tokens != self.PAD) & (tokens != self.BOS)]
        words = np.split(tokens, np.flatnonzero(tokens == self.WORD))
        return [self.decode_word(w[w != self.WORD]) for w in words if (w != self.WORD).any()]

    def decode_word(self, tokens, smooth=True):
        tokens = np.asarray(tokens, dtype=np.int64).ravel()
        tokens = tokens[tokens < self.PAD]
        theta, r = tokens[:len(tokens) // 2 * 2].reshape(-1, 2).T
        if not len(r):
            return np.zeros((0, 3))
        r = (r - len(THETA)).clip(0, 2 * len(RADII) - 1)
        angle, radius = THETA[theta.clip(0, len(THETA) - 1)], RADII[r % len(RADII)]
        x, y = np.cumsum(radius * np.cos(angle)), np.cumsum(radius * np.sin(angle))
        return np.c_[x, y, (r < len(RADII)).astype(float)]
