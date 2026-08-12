import collections
import math

import numpy as np

from pengpt.tokenizer import _stroke_spans


def rdp(points, epsilon):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        start, end = points[i], points[j]
        chord = end - start
        length = float(np.hypot(*chord))
        if j - i < 2:
            continue
        if length < 1e-12:
            distances = np.hypot(*(points[i + 1:j] - start).T)
        else:
            offsets = points[i + 1:j] - start
            distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / length
        k = int(np.argmax(distances))
        if distances[k] > epsilon:
            mid = i + 1 + k
            keep[mid] = True
            stack.append((i, mid))
            stack.append((mid, j))
    return points[keep]


def _split_delta(dx, dy, max_run):
    steps = max(1, math.ceil(abs(dx) / max_run), math.ceil(abs(dy) / max_run))
    out = []
    x = y = 0
    for i in range(1, steps + 1):
        nx, ny = int(round(dx * i / steps)), int(round(dy * i / steps))
        if nx != x or ny != y:
            out.append((nx - x, ny - y))
            x, y = nx, ny
    return out or [(0, 0)]


class PolylineTokenizer:

    def __init__(self, grid=0.020, epsilon=0.010, max_run=16, max_chunk_verts=4,
                 merges=None):
        self.grid = float(grid)
        self.epsilon = float(epsilon)
        self.max_run = int(max_run)
        self.max_chunk_verts = max(2, int(max_chunk_verts))
        self.side = 2 * self.max_run + 1
        self.n_disp = self.side * self.side
        self.DOWN = self.n_disp
        self.UP = self.n_disp + 1
        self.merges = [(int(a), int(b), int(c)) for a, b, c in (merges or [])]
        self._pairs = {(a, b): c for a, b, c in self.merges}
        self._inverse = {c: (a, b) for a, b, c in self.merges}
        n = self.n_disp + 2 + len(self.merges)
        self.PAD, self.END, self.WORD, self.BOS = n, n + 1, n + 2, n + 3
        self.vocab_size = n + 4

    def _disp_id(self, dx, dy):
        dx = int(np.clip(dx, -self.max_run, self.max_run))
        dy = int(np.clip(dy, -self.max_run, self.max_run))
        return (dx + self.max_run) * self.side + (dy + self.max_run)

    def _id_disp(self, tid):
        dx = tid // self.side - self.max_run
        dy = tid % self.side - self.max_run
        return int(dx), int(dy)

    def token_deltas(self):
        deltas = np.zeros((self.vocab_size, 2), dtype=np.int64)
        for tid in range(self.n_disp):
            deltas[tid] = self._id_disp(tid)
        for a, b, c in self.merges:
            deltas[c] = deltas[a] + deltas[b]
        return deltas

    def apply_merges(self, tokens):
        if not self._pairs:
            return np.asarray(tokens, dtype=np.int64)
        out = np.asarray(tokens, dtype=np.int64).copy()
        for a, b, merged in self.merges:
            matches = np.flatnonzero((out[:-1] == a) & (out[1:] == b))
            if not matches.size:
                continue
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

    def expand(self, tokens):
        out = list(np.asarray(tokens).ravel())
        while any(t in self._inverse for t in out):
            nxt = []
            for t in out:
                nxt.extend(self._inverse[t]) if t in self._inverse else nxt.append(t)
            out = nxt
        return out

    def _quantize(self, xy):
        return np.rint(np.asarray(xy, dtype=float) / self.grid).astype(np.int64)

    def _segments(self, a, b):
        dx, dy = int(b[0] - a[0]), int(b[1] - a[1])
        if dx == 0 and dy == 0:
            return []
        return [self._disp_id(x, y) for x, y in _split_delta(dx, dy, self.max_run)]

    def stroke_chunks(self, xy):
        cells = self._quantize(xy)
        keep = np.ones(len(cells), dtype=bool)
        if len(cells) > 1:
            keep[1:] = np.any(cells[1:] != cells[:-1], axis=1)
        cells = cells[keep]
        if len(cells) == 0:
            return []
        verts = np.rint(rdp(cells.astype(float), self.epsilon / self.grid)).astype(np.int64)
        if len(verts) > 1:
            keep = np.ones(len(verts), dtype=bool)
            keep[1:] = np.any(verts[1:] != verts[:-1], axis=1)
            verts = verts[keep]
        if len(verts) <= self.max_chunk_verts:
            return [verts]
        chunks, i = [], 0
        step = self.max_chunk_verts - 1
        while i < len(verts) - 1:
            j = min(i + self.max_chunk_verts, len(verts))
            chunks.append(verts[i:j])
            i += step
        return chunks

    def encode_word(self, points):
        points = np.asarray(points, dtype=float)
        out = []
        prev = np.array([0, 0], dtype=np.int64)
        for start, stop in _stroke_spans(points[:, 2]):
            raw = points[start:stop, :2]
            if len(raw) < 2:
                continue
            chunks = self.stroke_chunks(raw)
            if not chunks:
                continue
            first = chunks[0][0]
            out.extend(self._segments(prev, first))
            out.append(self.DOWN)
            for chunk in chunks:
                if len(chunk) == 1:
                    continue
                for a, b in zip(chunk[:-1], chunk[1:]):
                    out.extend(self._segments(a, b))
            out.append(self.UP)
            prev = chunks[-1][-1]
        tokens = np.array(out, dtype=np.int64)
        return self.apply_merges(tokens) if self.merges else tokens

    def encode_words(self, words):
        parts = []
        for i, word in enumerate(words):
            if i:
                parts.append(np.array([self.WORD, self.WORD], dtype=np.int64))
            parts.append(self.encode_word(word))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def decode_word(self, tokens, smooth=True, spacing=None):
        if spacing is None:
            spacing = self.grid
        x = y = 0
        down = False
        vertices = []
        for t in self.expand(tokens):
            t = int(t)
            if t == self.DOWN:
                down = True
                vertices.append((x, y, 1.0))
            elif t == self.UP:
                if down:
                    vertices.append((x, y, 0.0))
                down = False
            elif 0 <= t < self.n_disp:
                dx, dy = self._id_disp(t)
                x += dx
                y += dy
                if down:
                    vertices.append((x, y, 1.0))
        if not vertices:
            return np.zeros((0, 3))
        verts = np.array(vertices, dtype=float)
        verts[:, :2] *= self.grid
        if not smooth:
            return verts
        return _densify(verts, spacing)

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


def learn_merges(token_sequences, n_merges=256, min_count=20, reserved=()):
    reserved = set(reserved)
    sequences = [list(s) for s in token_sequences]
    if not sequences:
        return []
    next_id = max((max(s) for s in sequences if s), default=-1) + 1
    merges = []
    for _ in range(n_merges):
        counts = collections.Counter()
        for s in sequences:
            for pair in zip(s, s[1:]):
                if pair[0] not in reserved and pair[1] not in reserved:
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


def _densify(points, spacing):
    out, stroke = [], []

    def flush(stroke):
        s = np.asarray(stroke)
        if len(s) < 2:
            return list(s)
        dist = np.r_[0.0, np.cumsum(np.hypot(*np.diff(s[:, :2], axis=0).T))]
        if dist[-1] < 1e-9:
            return [s[0], s[-1]]
        n = max(2, int(round(dist[-1] / spacing)) + 1)
        t = np.linspace(0.0, dist[-1], n)
        xy = np.column_stack([np.interp(t, dist, s[:, 0]),
                              np.interp(t, dist, s[:, 1]),
                              np.ones(n)])
        return list(xy)

    for point in points:
        if point[2] == 1:
            stroke.append(point)
        else:
            if stroke:
                out.extend(flush(stroke))
                stroke = []
            out.append(point)
    if stroke:
        out.extend(flush(stroke))
    return np.array(out)
