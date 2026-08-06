"""ScribeTokens: a fixed 10-token vocabulary for any pen trajectory.

Reference: "ScribeTokens: Fixed-Vocabulary Tokenization of Digital Ink"
(arXiv:2603.02805).

Coordinates are quantized to an integer grid and motion becomes a walk on that
grid: eight compass directions (Freeman chain codes) plus two pen-state tokens.

    0:right 1:up-right 2:up 3:up-left 4:left 5:down-left 6:down 7:down-right
    8:DOWN (pen touches paper)   9:UP (pen lifts)

Movement between grid points is decomposed with Bresenham's line algorithm, so
any path is representable. Byte pair encoding then merges recurring runs, which
is what makes sequences short.

Two properties matter. Sampling invariance: two recordings of the same shape
tokenize identically however densely the hardware sampled them, so a mouse, a
digitizer and a public dataset all land in one representation. No
out-of-vocabulary: a grid walk always decomposes into base tokens, unlike bin
tables that must be retuned per dataset and silently clip outliers.

A word is an (N, 3) array of (x, y, pen); pen == 1 while touching the paper, and
rows with pen == 0 mark a lift, so strokes are the maximal runs of pen == 1.
"""

import collections

import numpy as np

DIRECTIONS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1),
                       (-1, 0), (-1, -1), (0, -1), (1, -1)])
DOWN, UP = 8, 9
N_BASE = 10

_DIR_LOOKUP = np.full((3, 3), -1, dtype=np.int64)
for _i, (_dx, _dy) in enumerate(DIRECTIONS):
    _DIR_LOOKUP[_dx + 1, _dy + 1] = _i


def bresenham_steps(x0, y0, x1, y1):
    """Unit direction indices walking (x0, y0) -> (x1, y1).

    Tie-breaking must stay fixed: two encodings of one straight line have to
    agree, or BPE learns separate merges for a single shape.
    """
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    out, x, y = [], int(x0), int(y0)
    while x != x1 or y != y1:
        e2 = 2 * err
        step_x = step_y = 0
        if e2 > -dy:
            err -= dy; step_x = sx
        if e2 < dx:
            err += dx; step_y = sy
        x += step_x; y += step_y
        out.append(_DIR_LOOKUP[step_x + 1, step_y + 1])
    return np.array(out, dtype=np.int64)


def _walk(grid_xy):
    out = []
    for (x0, y0), (x1, y1) in zip(grid_xy[:-1], grid_xy[1:]):
        dx, dy = int(x1 - x0), int(y1 - y0)
        if dx == 0 and dy == 0:
            continue
        if dx == 0 or dy == 0 or abs(dx) == abs(dy):
            n = max(abs(dx), abs(dy))
            direction = _DIR_LOOKUP[(0 if dx == 0 else (1 if dx > 0 else -1)) + 1,
                                    (0 if dy == 0 else (1 if dy > 0 else -1)) + 1]
            out.extend([int(direction)] * n)
        else:
            out.extend(int(t) for t in bresenham_steps(x0, y0, x1, y1))
    return out


def _stroke_spans(pen):
    down = (np.asarray(pen) == 1).astype(np.int8)
    if not down.any():
        return []
    edges = np.diff(np.r_[0, down, 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _smooth_strokes(points, window=5):
    out = points.copy()
    for start, stop in _stroke_spans(points[:, 2]):
        if stop - start < window:
            continue
        seg = points[start:stop, :2]
        pad = window // 2
        padded = np.vstack([np.repeat(seg[:1], pad, axis=0), seg,
                            np.repeat(seg[-1:], pad, axis=0)])
        kernel = np.ones(window) / window
        smoothed = np.column_stack([np.convolve(padded[:, 0], kernel, "valid"),
                                    np.convolve(padded[:, 1], kernel, "valid")])
        smoothed[0], smoothed[-1] = seg[0], seg[-1]
        out[start:stop, :2] = smoothed
    return out


class ScribeTokenizer:

    def __init__(self, grid=0.012, merges=None):
        self.grid = grid
        self.merges = [(int(a), int(b), int(c)) for a, b, c in (merges or [])]
        self._pairs = {(a, b): c for a, b, c in self.merges}
        self._inverse = {c: (a, b) for a, b, c in self.merges}
        n = N_BASE + len(self.merges)
        self.PAD, self.END, self.WORD = n, n + 1, n + 2
        self.vocab_size = n + 3

    def encode_word(self, points):
        points = np.asarray(points, dtype=float)
        grid_xy = np.rint(points[:, :2] / self.grid).astype(np.int64)
        out, previous_end = [], None
        for start, stop in _stroke_spans(points[:, 2]):
            if previous_end is not None:
                out.extend(_walk([previous_end, grid_xy[start]]))
            out.append(DOWN)
            out.extend(_walk(grid_xy[start:stop]))
            out.append(UP)
            previous_end = grid_xy[stop - 1]
        return self.apply_merges(np.array(out, dtype=np.int64))

    def encode_words(self, words):
        parts = []
        for i, word in enumerate(words):
            if i:
                parts.append(np.array([self.WORD, self.WORD], dtype=np.int64))
            parts.append(self.encode_word(word))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def apply_merges(self, tokens):
        if not self._pairs:
            return tokens
        pairs, out, i, n = self._pairs, [], 0, len(tokens)
        while i < n:
            token = int(tokens[i])
            j = i + 1
            while j < n:
                merged = pairs.get((token, int(tokens[j])))
                if merged is None:
                    break
                token = merged
                j += 1
            out.append(token)
            i = j
        return np.array(out, dtype=np.int64)

    def decode(self, tokens):
        tokens = np.asarray(tokens)
        end = np.flatnonzero(tokens == self.END)
        if end.size:
            tokens = tokens[:end[0]]
        tokens = tokens[tokens != self.PAD]
        words = []
        for chunk in np.split(tokens, np.flatnonzero(tokens == self.WORD)):
            chunk = chunk[chunk != self.WORD]
            if len(chunk):
                words.append(self.decode_word(chunk))
        return words

    def decode_word(self, tokens, smooth=True):
        """Walk the grid, emitting a point wherever the pen is down.

        Eight-direction movement on a grid leaves staircase artifacts on gentle
        curves; the moving average removes them, shifting points by well under
        one cell so it cannot invent structure the tokens did not encode.
        """
        points, x, y, down = [], 0, 0, False
        for t in self.expand(tokens):
            if t == DOWN:
                down = True
                points.append((x, y, 1.0))
            elif t == UP:
                if down:
                    points.append((x, y, 0.0))
                down = False
            elif 0 <= t < 8:
                x += DIRECTIONS[t, 0]
                y += DIRECTIONS[t, 1]
                if down:
                    points.append((x, y, 1.0))
        if not points:
            return np.zeros((0, 3))
        out = np.array(points, dtype=float)
        out[:, :2] *= self.grid
        return _smooth_strokes(out) if smooth else out

    def expand(self, tokens):
        out = list(np.asarray(tokens).ravel())
        while any(t in self._inverse for t in out):
            nxt = []
            for t in out:
                nxt.extend(self._inverse[t]) if t in self._inverse else nxt.append(t)
            out = nxt
        return out


def learn_merges(token_sequences, n_merges=512, min_count=20):
    """Learn BPE merges over direction tokens.

    DOWN and UP are never merged, so stroke boundaries stay explicit and every
    merged token still decomposes into base tokens.
    """
    sequences = [list(s) for s in token_sequences]
    merges, next_id = [], N_BASE
    for _ in range(n_merges):
        counts = collections.Counter()
        for s in sequences:
            for pair in zip(s, s[1:]):
                if DOWN not in pair and UP not in pair:
                    counts[pair] += 1
        if not counts:
            break
        (a, b), count = counts.most_common(1)[0]
        if count < min_count:
            break
        merges.append((a, b, next_id))
        for i, s in enumerate(sequences):
            out, j = [], 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                    out.append(next_id); j += 2
                else:
                    out.append(s[j]); j += 1
            sequences[i] = out
        next_id += 1
    return merges


class CharTokenizer:

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
