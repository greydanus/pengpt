"""Geometry and tokenization for pen strokes.

Conventions
-----------
- A *word* is an (N, 3) float array of absolute pen positions ``(x, y, pen)``.
  ``pen`` is 1 while the pen touches the paper; a ``pen == 0`` point is a move.
  y grows downward and one unit of y roughly spans the writing area, with the
  text baseline at y = 0 (see data.load_examples).
- *Offsets* are per-step pen movements in polar form ``(r, theta, pen)``.
- Each offset becomes exactly two tokens: the direction token first, then the
  combined (magnitude, pen state) token — "point, then shoot". Words are
  separated by a pair of WORD tokens so that the theta/r alternation is
  preserved across word boundaries.
"""

import numpy as np


def word_to_offsets(points, prev_word=None):
    """Convert absolute points (N, 3) to polar offsets (N, 3) = (r, theta, pen).

    If prev_word is given, the first offset encodes the carriage jump from the
    end of the previous word: horizontally from the previous word's rightmost
    point to this word's leftmost point, vertically between the raw endpoints.
    """
    offsets = np.zeros_like(points)
    offsets[1:, :2] = np.diff(points[:, :2], axis=0)
    if prev_word is not None:
        offsets[0, 0] = (prev_word[:, 0].max() - prev_word[-1, 0]) + \
                        (points[0, 0] - points[:, 0].min())
        offsets[0, 1] = points[0, 1] - prev_word[-1, 1]
    r = np.hypot(offsets[:, 0], offsets[:, 1])
    theta = np.arctan2(offsets[:, 1], offsets[:, 0])
    return np.column_stack([r, theta, points[:, 2]])


def offsets_to_points(offsets):
    """Invert word_to_offsets (up to the starting position, which is lost)."""
    dx = offsets[:, 0] * np.cos(offsets[:, 1])
    dy = offsets[:, 0] * np.sin(offsets[:, 1])
    xy = np.cumsum(np.column_stack([dx, dy]), axis=0)
    return np.column_stack([xy, offsets[:, 2]])


class StrokeTokenizer:
    """Quantizes polar offsets into a small discrete vocabulary.

    Token layout: [pen-down radii | pen-up radii | thetas | PAD | END | WORD].
    Radius bins are dense near zero (most pen movements are tiny) and
    geometric out to 0.9; theta bins are uniform over [-pi, pi].
    """

    def __init__(self, n_theta_bins=220):
        self.theta_bins = np.linspace(-np.pi, np.pi, n_theta_bins)
        self.r_bins = np.concatenate([
            [0.0],
            np.linspace(1e-4, 0.06, 30),
            np.geomspace(0.06001, 0.90, 120),
        ])
        self.n_r = len(self.r_bins)
        self.n_theta = n_theta_bins
        self.theta_offset = 2 * self.n_r  # radii come first, one copy per pen state
        self.PAD = self.theta_offset + self.n_theta
        self.END = self.PAD + 1
        self.WORD = self.PAD + 2
        self.vocab_size = self.PAD + 3

    @property
    def separator(self):
        return np.array([self.WORD, self.WORD], dtype=np.int64)

    def encode_word(self, offsets):
        """Polar offsets (N, 3) -> 1D int64 array of 2N tokens."""
        r_idx = (np.digitize(offsets[:, 0], self.r_bins) - 1).clip(0, self.n_r - 1)
        r_idx[offsets[:, 2] == 0] += self.n_r
        theta_idx = (np.digitize(offsets[:, 1], self.theta_bins) - 1).clip(0, self.n_theta - 1)
        pairs = np.column_stack([self.theta_offset + theta_idx, r_idx])
        return pairs.ravel().astype(np.int64)

    def encode_words(self, word_offsets):
        """Encode a list of words, joined by WORD separator pairs."""
        parts = []
        for i, offsets in enumerate(word_offsets):
            if i > 0:
                parts.append(self.separator)
            parts.append(self.encode_word(offsets))
        return np.concatenate(parts)

    def decode(self, tokens):
        """1D tokens -> list of per-word polar offset arrays.

        Truncates at the first END token, ignores PAD, splits on WORD tokens,
        and clips any out-of-range indices, so arbitrary model output is safe
        to decode.
        """
        tokens = np.asarray(tokens)
        ends = np.flatnonzero(tokens == self.END)
        if ends.size:
            tokens = tokens[:ends[0]]
        tokens = tokens[tokens != self.PAD]
        chunks = np.split(tokens, np.flatnonzero(tokens == self.WORD))
        words = []
        for chunk in chunks:
            chunk = chunk[chunk != self.WORD]
            if len(chunk):
                words.append(self._decode_word(chunk))
        return words

    def _decode_word(self, tokens):
        pairs = tokens[:(len(tokens) // 2) * 2].reshape(-1, 2)
        theta = self.theta_bins[(pairs[:, 0] - self.theta_offset).clip(0, self.n_theta - 1)]
        r_idx = pairs[:, 1].clip(0, 2 * self.n_r - 1)
        pen = (r_idx < self.n_r).astype(float)
        r = self.r_bins[r_idx % self.n_r]
        return np.column_stack([r, theta, pen])


class CharTokenizer:
    """Maps characters to integer ids; 0 is reserved for padding."""

    PAD = 0

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.stoi = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(alphabet) + 1

    def encode(self, text, length=None):
        ids = [self.stoi.get(ch, self.PAD) for ch in text]
        if length is not None:
            ids = ids[:length] + [self.PAD] * max(0, length - len(ids))
        return np.array(ids, dtype=np.int64)

    def decode(self, ids):
        ids = np.asarray(ids)
        pads = np.flatnonzero(ids == self.PAD)
        if pads.size:
            ids = ids[:pads[0]]
        return "".join(self.itos.get(int(i), "") for i in ids)
