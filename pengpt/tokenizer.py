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

Two properties matter. Insensitivity to sampling rate: recording the same shape
more finely barely changes the tokens, and changes them not at all once samples
land in adjacent grid cells, since both recordings then walk the same cells and
Bresenham has no gap to bridge. Point density in raw pen data is an artifact of
the recorder, so this is what lets a mouse, a digitizer and a public dataset
share one representation. No out-of-vocabulary: a grid walk always decomposes
into base tokens, unlike bin tables that must be retuned per dataset and
silently clip outliers.

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
    grid_xy = np.asarray(grid_xy, dtype=np.int64)
    if len(grid_xy) < 2:
        return []
    deltas = np.diff(grid_xy, axis=0)
    moved = (deltas[:, 0] != 0) | (deltas[:, 1] != 0)
    deltas = deltas[moved]
    if not len(deltas):
        return []
    # Dense recordings land in the same or an adjacent cell, so nearly every
    # move is a single direction token; Bresenham only matters for the rare
    # multi-cell jump. Taking the all-unit case wholesale keeps the per-point
    # Python loop off the training hot path.
    if (np.abs(deltas) <= 1).all():
        return _DIR_LOOKUP[deltas[:, 0] + 1, deltas[:, 1] + 1].tolist()
    out = []
    for i in np.flatnonzero(moved):
        (x0, y0), (x1, y1) = grid_xy[i], grid_xy[i + 1]
        if abs(x1 - x0) <= 1 and abs(y1 - y0) <= 1:
            out.append(int(_DIR_LOOKUP[x1 - x0 + 1, y1 - y0 + 1]))
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
        # BOS is what generation starts from. Without it the first sampled token
        # is conditioned on a prefix that never occurs in training, and that
        # token fixes the pen's entry point and the word's height above the
        # baseline -- the two things a whole sample hangs on.
        self.PAD, self.END, self.WORD, self.BOS = n, n + 1, n + 2, n + 3
        self.vocab_size = n + 4

    def token_deltas(self):
        """(vocab_size, 2) net pen displacement of every token, in grid cells.

        Base direction tokens move one cell, pen-state and special tokens move
        nothing, and a merged token moves by the sum of its children -- so the
        pen's absolute position is the running sum of these deltas over any
        token sequence, merged or not. This is what lets a model be told where
        the pen is without changing the token stream.
        """
        deltas = np.zeros((self.vocab_size, 2), dtype=np.int64)
        deltas[:len(DIRECTIONS)] = DIRECTIONS
        for a, b, c in self.merges:
            deltas[c] = deltas[a] + deltas[b]
        return deltas

    def encode_word(self, points):
        """Tokens for one word, starting from the baseline at x = 0.

        The walk begins at the origin rather than at the word's first point, so
        the initial pen-up move carries the word's height above or below the
        baseline. Words that start at the cap line (digits, most capitals) and
        words that start on the baseline are then distinguishable from the
        tokens alone, with no per-alphabet table.
        """
        points = np.asarray(points, dtype=float)
        grid_xy = np.rint(points[:, :2] / self.grid).astype(np.int64)
        out, previous_end = [], np.array([0, 0], dtype=np.int64)
        for start, stop in _stroke_spans(points[:, 2]):
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
        """Apply merges in the order they were learned, as BPE requires.

        One pass per rule, mirroring learn_merges, so encoding reproduces what
        training produced. Taking whichever rule happens to match first in a
        left-to-right scan is a different algorithm: on a rule set holding
        (0,1)->10 and (2,0)->11, the sequence [2, 0, 1] greedily becomes
        [11, 1], where BPE gives [2, 10]. Priority order is what makes the
        earlier, more frequent rule win. The two disagreed on every word of the
        bundled corpus, and the greedy reading cost a third of the compression
        the merges were learned to provide (99 tokens per word against 67).

        Rules whose pair is absent are skipped rather than scanned for, which is
        most of them once a sequence is partly merged.
        """
        if not self._pairs:
            return tokens
        out = np.asarray(tokens, dtype=np.int64).copy()
        for a, b, merged in self.merges:
            matches = np.flatnonzero((out[:-1] == a) & (out[1:] == b))
            if not matches.size:
                continue
            # A left-to-right pass merges non-overlapping pairs: a match
            # directly after an applied one shares its second token and is
            # skipped. Only adjacent matches (runs like [a,a,a] under a rule
            # (a,a)) need the sequential resolution; otherwise every match
            # stands.
            if matches.size > 1 and (np.diff(matches) == 1).any():
                kept, last = [], -2
                for i in matches:
                    if i > last + 1:
                        kept.append(i); last = i
                matches = np.array(kept)
            out[matches] = merged
            remove = np.ones(len(out), dtype=bool)
            remove[matches + 1] = False
            out = out[remove]
        return out

    def decode(self, tokens):
        tokens = np.asarray(tokens)
        end = np.flatnonzero(tokens == self.END)
        if end.size:
            tokens = tokens[:end[0]]
        tokens = tokens[(tokens != self.PAD) & (tokens != self.BOS)]
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

    Fewer than n_merges may come back: merging stops once no pair occurs
    min_count times, since a rule that fires on a handful of words costs a
    vocabulary entry without shortening anything. On the bundled data this
    settles around 350 merges however many are asked for, and on Quick, Draw!
    around 170 -- sketch strokes are shorter and less repetitive than cursive,
    so there is simply less to merge. Raising --n_merges past that does
    nothing.
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
