import numpy as np

from pengpt.tokenizer import _stroke_spans, DIRECTIONS, _DIR_LOOKUP
from polyline.tokenizer import PolylineTokenizer, _split_delta

LENGTHS = (12, 8, 6, 4)


def _dedup_cells(cells):
    cells = np.asarray(cells, dtype=np.int64)
    if len(cells) < 2:
        return cells
    keep = np.ones(len(cells), dtype=bool)
    keep[1:] = np.any(cells[1:] != cells[:-1], axis=1)
    return cells[keep]


def _cell_path(xy, grid):
    cells = np.rint(np.asarray(xy, dtype=float) / grid).astype(np.int64)
    return _dedup_cells(cells)


def _windows(cells, length):
    cells = np.asarray(cells, dtype=np.int64)
    n = len(cells)
    if n < length + 1:
        return np.zeros((0, length + 1, 2), dtype=np.float64)
    out = np.empty((n - length, length + 1, 2), dtype=np.float64)
    for i in range(n - length):
        out[i] = cells[i:i + length + 1] - cells[i]
    return out


def _kmeans(x, k, rng, iters=20):
    n = len(x)
    k = min(k, n)
    if k == 0:
        return np.zeros((0,) + x.shape[1:], dtype=np.float64)
    flat = x.reshape(n, -1)
    # k-means++
    centers = np.empty((k, flat.shape[1]), dtype=np.float64)
    centers[0] = flat[rng.integers(n)]
    closest = np.full(n, np.inf)
    for j in range(1, k):
        d = ((flat - centers[j - 1]) ** 2).sum(1)
        closest = np.minimum(closest, d)
        w = closest / (closest.sum() + 1e-12)
        centers[j] = flat[rng.choice(n, p=w)]
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = flat[m].mean(0)
    return centers.reshape((k,) + x.shape[1:])


def _nearest(win, book):
    if len(book) == 0:
        return 0, np.inf
    d = book - win[None]
    err = np.hypot(d[..., 0], d[..., 1]).max(axis=1)
    j = int(err.argmin())
    return j, float(err[j])


def learn_codebook(words, grid=0.020, n_codes=256, seed=0, max_windows=40000,
                   mode="exact"):
    buckets = {L: [] for L in LENGTHS}
    for w in words:
        pts = np.asarray(w, dtype=float)
        for a, b in _stroke_spans(pts[:, 2]):
            cells = _cell_path(pts[a:b, :2], grid)
            for L in LENGTHS:
                win = _windows(cells, L)
                if len(win):
                    buckets[L].append(win)
    books = {}
    rng = np.random.default_rng(seed)
    for L in LENGTHS:
        if not buckets[L]:
            books[L] = np.zeros((0, L + 1, 2))
            continue
        x = np.concatenate(buckets[L], axis=0)
        if mode == "exact":
            keys, inverse = np.unique(np.rint(x).astype(np.int16).reshape(len(x), -1),
                                      axis=0, return_inverse=True)
            counts = np.bincount(inverse)
            order = np.argsort(-counts)[:n_codes]
            books[L] = keys[order].reshape(-1, L + 1, 2).astype(np.float64)
            print(f"  exact L={L}: {len(x)} windows, {len(keys)} unique, "
                  f"kept {len(books[L])} (top cover {counts[order].sum() / len(x):.1%})")
        else:
            if len(x) > max_windows:
                x = x[rng.choice(len(x), size=max_windows, replace=False)]
            means = _kmeans(x, n_codes, rng)
            medoids = np.empty_like(means)
            for j, c in enumerate(means):
                d = ((x - c) ** 2).sum(axis=(1, 2))
                medoids[j] = x[d.argmin()]
            books[L] = medoids
            print(f"  medoid L={L}: {len(x)} windows -> {len(books[L])} codes")
    return books


def _resample_poly(verts, n=8):
    verts = np.asarray(verts, dtype=float)
    if len(verts) == 1:
        return np.repeat(verts, n, axis=0)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(verts, axis=0).T))]
    if d[-1] < 1e-9:
        return np.repeat(verts[:1], n, axis=0)
    t = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(t, d, verts[:, 0]),
                            np.interp(t, d, verts[:, 1])])


def learn_rdp_codebook(words, grid=0.020, epsilon=0.020, n_codes=512, seed=0,
                       samples=8):
    poly = PolylineTokenizer(grid=grid, epsilon=epsilon, max_chunk_verts=4)
    feats = []
    for w in words:
        pts = np.asarray(w, dtype=float)
        for a, b in _stroke_spans(pts[:, 2]):
            raw = pts[a:b, :2]
            if len(raw) < 2:
                continue
            for chunk in poly.stroke_chunks(raw):
                if len(chunk) < 2:
                    continue
                shape = _resample_poly(chunk.astype(float), samples)
                feats.append(shape - shape[0])
    if not feats:
        return np.zeros((0, samples, 2))
    x = np.stack(feats)
    rng = np.random.default_rng(seed)
    means = _kmeans(x, n_codes, rng)
    medoids = np.empty_like(means)
    for j, c in enumerate(means):
        d = ((x - c) ** 2).sum(axis=(1, 2))
        medoids[j] = x[d.argmin()]
    print(f"  rdp-chunk book: {len(x)} chunks -> {len(medoids)} medoids")
    return medoids


class ChunkTokenizer:

    def __init__(self, grid=0.020, match_eps=1.0, codebook=None, n_codes=256):
        self.grid = float(grid)
        self.match_eps = float(match_eps)
        self.n_codes = int(n_codes)
        self.codebook = codebook or {L: np.zeros((0, L + 1, 2)) for L in LENGTHS}
        self._offset = {}
        n = 10
        for L in LENGTHS:
            self._offset[L] = n
            n += self.n_codes
        self.n_base = n
        self.DOWN, self.UP = 8, 9
        self.PAD, self.END, self.WORD, self.BOS = n, n + 1, n + 2, n + 3
        self.vocab_size = n + 4
        self.merges = []

    def code_id(self, length, k):
        return self._offset[length] + int(k)

    def parse_code(self, tid):
        tid = int(tid)
        for L in LENGTHS:
            o = self._offset[L]
            if o <= tid < o + self.n_codes:
                return L, tid - o
        return None

    def token_deltas(self):
        deltas = np.zeros((self.vocab_size, 2), dtype=np.int64)
        deltas[:8] = DIRECTIONS
        for L in LENGTHS:
            book = self.codebook[L]
            for k, shape in enumerate(book):
                if k >= self.n_codes:
                    break
                end = np.rint(shape[-1]).astype(np.int64)
                deltas[self.code_id(L, k)] = end
        return deltas

    def _encode_cells(self, cells):
        cells = _dedup_cells(cells)
        if len(cells) < 2:
            return []
        out = []
        i = 0
        last = len(cells) - 1
        while i < last:
            hit = None
            for L in LENGTHS:
                if i + L > last:
                    continue
                win = (cells[i:i + L + 1] - cells[i]).astype(np.float64)
                book = self.codebook[L]
                if len(book) == 0:
                    continue
                k, err = _nearest(win, book)
                if err <= self.match_eps:
                    hit = (L, k)
                    break
            if hit is not None:
                L, k = hit
                out.append(self.code_id(L, k))
                i += L
                continue
            d = cells[i + 1] - cells[i]
            if abs(int(d[0])) <= 1 and abs(int(d[1])) <= 1 and (d[0] != 0 or d[1] != 0):
                out.append(int(_DIR_LOOKUP[int(d[0]) + 1, int(d[1]) + 1]))
            else:
                from polyline.tokenizer import _split_delta
                for dx, dy in _split_delta(int(d[0]), int(d[1]), 1):
                    if dx or dy:
                        out.append(int(_DIR_LOOKUP[dx + 1, dy + 1]))
            i += 1
        return out

    def encode_word(self, points):
        points = np.asarray(points, dtype=float)
        out = []
        prev = np.array([0, 0], dtype=np.int64)
        for start, stop in _stroke_spans(points[:, 2]):
            cells = _cell_path(points[start:stop, :2], self.grid)
            if len(cells) == 0:
                continue
            travel = np.vstack([prev, cells[0]])
            out.extend(self._encode_cells(travel))
            out.append(self.DOWN)
            out.extend(self._encode_cells(cells))
            out.append(self.UP)
            prev = cells[-1]
        return np.array(out, dtype=np.int64)

    def encode_words(self, words):
        parts = []
        for i, word in enumerate(words):
            if i:
                parts.append(np.array([self.WORD, self.WORD], dtype=np.int64))
            parts.append(self.encode_word(word))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def decode_word(self, tokens, smooth=True):
        x = y = 0
        down = False
        pts = []
        for t in np.asarray(tokens).ravel():
            t = int(t)
            if t == self.DOWN:
                down = True
                pts.append((x, y, 1.0))
            elif t == self.UP:
                if down:
                    pts.append((x, y, 0.0))
                down = False
            elif 0 <= t < 8:
                x += int(DIRECTIONS[t, 0])
                y += int(DIRECTIONS[t, 1])
                if down:
                    pts.append((x, y, 1.0))
            else:
                parsed = self.parse_code(t)
                if parsed is None:
                    continue
                L, k = parsed
                book = self.codebook[L]
                if k >= len(book):
                    continue
                shape = np.rint(book[k]).astype(np.int64)
                ox, oy = x, y
                for p in shape[1:]:
                    x = ox + int(p[0])
                    y = oy + int(p[1])
                    if down:
                        pts.append((x, y, 1.0))
        if not pts:
            return np.zeros((0, 3))
        out = np.array(pts, dtype=float)
        out[:, :2] *= self.grid
        if not smooth:
            return out
        from polyline.tokenizer import _densify
        return _densify(out, self.grid)

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


class RdpChunkTokenizer:

    def __init__(self, grid=0.020, epsilon=0.020, match_eps=1.0, codebook=None,
                 max_run=16, max_chunk_verts=4, samples=8):
        self.grid = float(grid)
        self.epsilon = float(epsilon)
        self.match_eps = float(match_eps)
        self.samples = int(samples)
        self.poly = PolylineTokenizer(grid=grid, epsilon=epsilon, max_run=max_run,
                                     max_chunk_verts=max_chunk_verts)
        self.codebook = np.asarray(codebook if codebook is not None else np.zeros((0, samples, 2)))
        self.n_codes = len(self.codebook)
        self.n_disp = self.poly.n_disp
        self.DOWN = self.n_disp
        self.UP = self.n_disp + 1
        self.code0 = self.n_disp + 2
        n = self.code0 + self.n_codes
        self.PAD, self.END, self.WORD, self.BOS = n, n + 1, n + 2, n + 3
        self.vocab_size = n + 4
        self.merges = []

    def token_deltas(self):
        deltas = np.zeros((self.vocab_size, 2), dtype=np.int64)
        deltas[:self.n_disp] = self.poly.token_deltas()[:self.n_disp]
        for k, shape in enumerate(self.codebook):
            end = np.rint(shape[-1]).astype(np.int64)
            deltas[self.code0 + k] = end
        return deltas

    def _match(self, chunk):
        if self.n_codes == 0 or len(chunk) < 2:
            return None
        feat = _resample_poly(chunk.astype(float), self.samples)
        feat = feat - feat[0]
        k, err = _nearest(feat, self.codebook)
        if err <= self.match_eps:
            return k
        return None

    def _disp_tokens(self, a, b):
        return self.poly._segments(a, b)

    def encode_word(self, points):
        points = np.asarray(points, dtype=float)
        out = []
        prev = np.array([0, 0], dtype=np.int64)
        n_hit = n_miss = 0
        for start, stop in _stroke_spans(points[:, 2]):
            raw = points[start:stop, :2]
            if len(raw) < 2:
                continue
            chunks = self.poly.stroke_chunks(raw)
            if not chunks:
                continue
            first = chunks[0][0]
            out.extend(self._disp_tokens(prev, first))
            out.append(self.DOWN)
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                k = self._match(chunk)
                if k is not None:
                    out.append(self.code0 + k)
                    n_hit += 1
                else:
                    for a, b in zip(chunk[:-1], chunk[1:]):
                        out.extend(self._disp_tokens(a, b))
                    n_miss += 1
            out.append(self.UP)
            prev = chunks[-1][-1]
        self.last_hits = (n_hit, n_miss)
        return np.array(out, dtype=np.int64)

    def encode_words(self, words):
        parts = []
        for i, word in enumerate(words):
            if i:
                parts.append(np.array([self.WORD, self.WORD], dtype=np.int64))
            parts.append(self.encode_word(word))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def decode_word(self, tokens, smooth=True):
        x = y = 0
        down = False
        pts = []
        for t in np.asarray(tokens).ravel():
            t = int(t)
            if t == self.DOWN:
                down = True
                pts.append((x, y, 1.0))
            elif t == self.UP:
                if down:
                    pts.append((x, y, 0.0))
                down = False
            elif 0 <= t < self.n_disp:
                dx, dy = self.poly._id_disp(t)
                x += dx
                y += dy
                if down:
                    pts.append((x, y, 1.0))
            elif self.code0 <= t < self.code0 + self.n_codes:
                shape = self.codebook[t - self.code0]
                ox, oy = x, y
                for p in np.rint(shape[1:]).astype(np.int64):
                    x = ox + int(p[0])
                    y = oy + int(p[1])
                    if down:
                        pts.append((x, y, 1.0))
        if not pts:
            return np.zeros((0, 3))
        out = np.array(pts, dtype=float)
        out[:, :2] *= self.grid
        if not smooth:
            return out
        from polyline.tokenizer import _densify
        return _densify(out, self.grid)

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

