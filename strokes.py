import json
import os
import zlib

import numpy as np

UNIT, GROUP = 1e-3, 64


def normalize(strokes, with_size=False):
    strokes = [np.asarray(s, float).reshape(-1, 2) for s in strokes]
    strokes = [s[np.isfinite(s).all(1)] for s in strokes]
    strokes = [s for s in strokes if len(s)]
    size = np.ptp(np.concatenate(strokes), 0).max() if strokes else 0
    if not size > 0:
        return (None, 0.0) if with_size else None
    lo = np.concatenate(strokes).min(0)
    out = [(s - lo) / size for s in strokes]
    return (out, float(size)) if with_size else out


def ink_length(strokes):
    return float(sum(np.hypot(*np.diff(s, axis=0).T).sum() for s in strokes))


def quantize(stroke, unit=UNIT):
    q = np.rint(np.asarray(stroke, float) / unit).astype(np.int64)
    q = q[np.r_[True, (np.diff(q, axis=0) != 0).any(1)]]
    d = np.diff(q, axis=0)
    n = np.maximum(1, -(-np.abs(d).max(1) // 127)) if len(d) else np.zeros(0, np.int64)
    if (n > 1).any():
        seg = np.repeat(np.arange(len(d)), n)
        k = np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n) + 1
        q = np.vstack([q[:1], q[seg] + np.rint(d[seg] * (k / n[seg])[:, None]).astype(np.int64)])
    return q


EXT, EXTS = ".sketch.parquet", (".sketch.parquet", ".hand.parquet")
ABOUT = ("One drawing a row, scaled so that its longer side is 1. Every drawing is a list of pen strokes. x0 and y0 hold where each stroke starts "
         "and dx and dy the steps between its points, all in units of 0.001 of the longer side, y down: a stroke's points are its start plus the "
         "running sum of its steps. stroke_len is the number of points in each stroke, so a stroke of n points takes n - 1 steps, in order, from dx "
         "and dy.")


def stem(path):
    path = str(path)
    return next((path[:-len(e)] for e in EXTS if path.endswith(e)), path)


def pack(records, unit=UNIT):
    strokes = [quantize(s, unit) for r in records for s in r["strokes"]]
    lens = np.array([len(s) for s in strokes], np.int64)
    points = np.concatenate(strokes)
    starts = np.cumsum(lens) - lens
    steps = np.diff(points, axis=0, prepend=points[:1])
    assert np.abs(points[starts]).max() < 2 ** 15, "ink too far from the origin for int16; normalize it first"
    keep = np.ones(len(points), bool)
    keep[starts] = False
    return points[starts].astype(np.int16), steps[keep].astype(np.int8), fit(lens), fit(np.array([len(r["strokes"]) for r in records]))


def fit(a):
    return a.astype(next(t for t in (np.uint8, np.uint16, np.uint32) if a.max(initial=0) <= np.iinfo(t).max))


def number(raw):
    whole = all(type(v) is int for v in raw)
    return fit(np.array(raw)) if whole and min(raw) >= 0 else np.array(raw, np.int64 if whole else np.float32)


def save_strokes(records, path, unit=UNIT, about=ABOUT):
    first, steps, lens, n_strokes = pack(records, unit)
    columns = {k: [r.get(k) for r in records] for k in sorted({k for r in records for k in r if k != "strokes"})}
    write_parquet(path, first, steps, lens, n_strokes,
                  {k: number(raw) if all(type(v) in (int, float) for v in raw) else raw for k, raw in columns.items()}, unit, about)


def write_table(table, path, about, group=10_000):
    import pyarrow.parquet as pq
    table = table.replace_schema_metadata({k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in about.items()})
    repeated = [f.name for f in table.schema if str(f.type) in ("string", "large_string") and f.name != "id"]
    tmp = str(path) + ".part"
    pq.write_table(table, tmp, compression="zstd", compression_level=19, row_group_size=group, use_dictionary=repeated, write_statistics=group > 1000)
    os.replace(tmp, path)


def write_parquet(path, first, steps, lens, n_strokes, columns, unit=UNIT, about=ABOUT):
    import pyarrow as pa
    per_stroke = np.r_[0, np.cumsum(n_strokes.astype(np.int64))].astype(np.int32)
    per_step = np.r_[0, np.cumsum(lens.astype(np.int64) - 1)][per_stroke].astype(np.int32)
    lists = lambda offsets, values: pa.ListArray.from_arrays(offsets, pa.array(np.ascontiguousarray(values)))
    cols = {k: pa.array(v) for k, v in columns.items()}
    cols.update(x0=lists(per_stroke, first[:, 0]), y0=lists(per_stroke, first[:, 1]), stroke_len=lists(per_stroke, lens.astype(np.uint16)),
                dx=lists(per_step, steps[:, 0]), dy=lists(per_step, steps[:, 1]))
    write_table(pa.table(cols), path, {"unit": str(unit), "about": about}, GROUP)


STROKES = ("x0", "y0", "stroke_len", "dx", "dy")


def read_parquet(path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    nested = lambda kind: pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind)
    t = pq.read_table(path, columns=[f.name for f in pq.read_schema(path) if f.name in STROKES or not nested(f.type)])
    flat = {k: t[k].combine_chunks() for k in STROKES}
    first = np.c_[flat["x0"].values.to_numpy(), flat["y0"].values.to_numpy()]
    steps = np.c_[flat["dx"].values.to_numpy(), flat["dy"].values.to_numpy()]
    lens, tables, cols = flat["stroke_len"].values.to_numpy(), {}, {}
    for k in t.column_names:
        col = t[k]
        if k in STROKES:
            continue
        if (pa.types.is_integer(col.type) or pa.types.is_floating(col.type)) and not col.null_count:
            cols[k] = col.to_numpy()
            continue
        uniq, index = np.unique([json.dumps(v, ensure_ascii=False) for v in col.to_pylist()], return_inverse=True)
        tables[k], cols[k] = [json.loads(u) for u in uniq], index
    about = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    return float(about.get("unit", UNIT)), first, steps, lens, np.diff(flat["stroke_len"].offsets.to_numpy()), tables, cols


def read_strokes(path, strokes=True):
    f = StrokeFile(path)
    cols = {k: [f.tables[k][i] for i in v] if k in f.tables else v.tolist() for k, v in f.cols.items()}
    rows = [dict(zip(cols, v)) for v in zip(*cols.values())]
    if strokes:
        lens = f.stroke_len.astype(np.int64)
        starts = np.cumsum(lens) - lens
        deltas = np.zeros((lens.sum(), 2), np.int64)
        rest = np.ones(len(deltas), bool)
        rest[starts] = False
        deltas[rest], deltas[starts] = f.steps, f.first
        total = np.cumsum(deltas, axis=0)
        before = np.vstack([np.zeros((1, 2), np.int64), total])[starts]
        flat = ((total - np.repeat(before, lens, axis=0)) * f.unit).astype(np.float32)
        parts = np.split(flat, np.cumsum(lens)[:-1])
        for i, r in enumerate(rows):
            r["strokes"] = parts[f.stroke[i]:f.stroke[i + 1]]
    return rows


def add_text(path, columns, about=None):
    import pyarrow as pa
    f = StrokeFile(path)
    table = pa.table({"id": [f.tables["id"][j] for j in f.cols["id"]], **{k: f.column(k) for k in f.texts},
                      **{k: list(v) for k, v in columns.items()}})
    write_table(table, f.side, {"about": {**f.about(), **(about or {})}})


class StrokeFile:

    def __init__(self, path):
        import pyarrow.parquet as pq
        self.path, self.held = str(path), {}
        self.unit, self.first, self.steps, self.stroke_len, n_strokes, self.tables, self.cols = read_parquet(self.path)
        self.side = stem(self.path) + ".captions.parquet"
        self.texts = sorted(n for n in pq.read_schema(self.side).names if n != "id") if os.path.exists(self.side) else []
        self.stroke = np.r_[0, np.cumsum(n_strokes.astype(np.int64))]
        self.step = np.r_[0, np.cumsum(self.stroke_len.astype(np.int64) - 1)]

    def __len__(self):
        return len(self.stroke) - 1

    def meta(self, i):
        return {k: self.tables[k][v[i]] if k in self.tables else v[i].item() for k, v in self.cols.items()}

    def column(self, key):
        if key not in self.held:
            import pyarrow.parquet as pq
            table = pq.read_table(self.side, columns=["id", key])
            ids, values, mine = table["id"].to_pylist(), table[key].to_pylist(), [self.tables["id"][j] for j in self.cols["id"]]
            if ids != mine:
                by = dict(zip(ids, values))
                values = [by.get(i) or "" for i in mine]
            self.held[key] = values
        return self.held[key]

    def text(self, i, key):
        return self.column(key)[i]

    def captions(self, i):
        return {k: v for k in self.texts for v in [self.text(i, k)] if v}

    def about(self):
        import pyarrow.parquet as pq
        meta = pq.read_schema(self.side).metadata if os.path.exists(self.side) else None
        return json.loads(meta[b"about"]) if meta and b"about" in meta else {}

    def strokes(self, i):
        return [(np.vstack([self.first[s:s + 1], self.steps[self.step[s]:self.step[s + 1]]]).cumsum(0, dtype=np.int32) * self.unit).astype(np.float32)
                for s in range(self.stroke[i], self.stroke[i + 1])]


class Strokes:

    def __init__(self, *paths):
        self.files = [StrokeFile(p) for p in paths]
        self.end = np.cumsum([len(f) for f in self.files])
        assert len({i for f in self.files for i in f.tables["id"]}) == len(self), "two drawings share an id"

    def __len__(self):
        return int(self.end[-1])

    def at(self, i):
        k = int(np.searchsorted(self.end, i, "right"))
        return self.files[k], int(i - (self.end[k - 1] if k else 0))

    def meta(self, i):
        f, j = self.at(i)
        return f.meta(j)

    def strokes(self, i):
        f, j = self.at(i)
        return f.strokes(j)

    def captions(self, i):
        f, j = self.at(i)
        return f.captions(j)

    def __getitem__(self, i):
        return {**self.meta(i), "strokes": self.strokes(i)}


def held_out(f, holdout):
    return np.array([zlib.crc32(f.tables["id"][j].encode()) % holdout == 0 for j in f.cols["id"]])


def to_points(strokes):
    return np.concatenate([np.r_[np.c_[s, np.ones(len(s))], [[*s[-1], 0]]] for s in strokes]).astype(np.float32)


def to_strokes(points):
    points = np.asarray(points, float).reshape(-1, 3)
    edges = np.diff(np.r_[0, points[:, 2] == 1, 0])
    return [points[a:b, :2] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def corners(s, tol):
    keep, todo = {0, len(s) - 1}, [(0, len(s) - 1)]
    while todo:
        i, j = todo.pop()
        if j - i < 2:
            continue
        d, p = s[j] - s[i], s[i + 1:j] - s[i]
        L2 = d[0] * d[0] + d[1] * d[1]
        t = np.clip((p[:, 0] * d[0] + p[:, 1] * d[1]) / L2, 0, 1) if L2 else np.zeros(len(p))
        ex, ey = p[:, 0] - t * d[0], p[:, 1] - t * d[1]
        far = np.sqrt(ex * ex + ey * ey)
        k = int(far.argmax())
        if far[k] > tol:
            keep.add(i + 1 + k)
            todo += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(keep)


def chaikin(c, closed=False, times=4):
    for _ in range(times):
        if len(c) < 3:
            break
        a, b = (c, np.roll(c, -1, 0)) if closed else (c[:-1], c[1:])
        cut = np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], 1).reshape(-1, 2)
        if not closed:
            cut[0], cut[-1] = c[0], c[-1]
        c = cut
    return np.concatenate([c, c[:1]]) if closed else c


def lengths(c):
    d = np.diff(c, axis=0)
    return np.concatenate([[0], np.cumsum(np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]))])


def resample(c, n):
    L = lengths(c)
    t = L[-1] * np.arange(n + 1) / n
    i = np.minimum(np.searchsorted(L[1:], t), len(c) - 2)
    span = L[i + 1] - L[i]
    f = np.where(span > 0, np.minimum(1, (t - L[i]) / np.where(span > 0, span, 1)), 0)
    return c[i] + f[:, None] * (c[i + 1] - c[i])


def curve(s, grid):
    u = s[np.concatenate([[True], (np.diff(s, axis=0) != 0).any(1)])]
    if not grid or len(u) < 3:
        return s
    c = u[corners(u, grid / 2)]
    m = len(c)
    a, b = c[1:-1] - c[:-2], c[2:] - c[1:-1]
    sharp = lambda a, b: a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] <= -0.5 * np.sqrt((a[..., 0] * a[..., 0] + a[..., 1] * a[..., 1]) * (b[..., 0] * b[..., 0] + b[..., 1] * b[..., 1]))
    cuts = [0, *(1 + np.flatnonzero(sharp(a, b))), m - 1]
    gap = c[-1] - c[0]
    if m > 4 and len(cuts) == 2 and np.sqrt(gap[0] * gap[0] + gap[1] * gap[1]) < grid and not sharp(c[0] - c[-2], c[1] - c[0]):
        pieces = [chaikin(c[:-1], True)]
    else:
        pieces = [chaikin(c[i:j + 1]) for i, j in zip(cuts[:-1], cuts[1:])]
    ends = np.cumsum([lengths(q)[-1] for q in pieces])
    if not ends[-1]:
        return s
    n = np.maximum(np.diff(np.concatenate([[0], np.floor(ends / ends[-1] * (len(s) - 1) + 0.5)])).astype(int), 1)
    while n.sum() > len(s) - 1:
        n[n.argmax()] -= 1
    return np.concatenate([resample(q, k)[1 if i else 0:] for i, (q, k) in enumerate(zip(pieces, n))])


def curves(points, grid):
    out = np.array(points, float).reshape(-1, 3)
    edges = np.diff(np.r_[0, out[:, 2] == 1, 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b, :2] = curve(out[a:b, :2], grid)
    return out


def plot(ax, strokes, color="k", lw=0.9):
    for st in strokes:
        st = np.asarray(st)
        if len(st) > 1:
            ax.plot(st[:, 0], -st[:, 1], color=color, lw=lw, solid_capstyle="round")
        elif len(st):
            ax.plot(st[:, 0], -st[:, 1], "o", color=color, ms=1.6 * lw, markeredgewidth=0)
    ax.set_aspect("equal")
    ax.axis("off")


def grid(drawings, titles, cols, out, size=1.6, colors=None, lw=1.0, dpi=130, fontsize=7, loc="center", heading=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = -(-len(drawings) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * (1.15 if heading else 1)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (ax, strokes, title) in enumerate(zip(axes.flat, drawings, titles)):
        plot(ax, strokes, colors[k] if colors else "k", lw)
        ax.set_title(title, fontsize=fontsize, pad=1, loc=loc)
    if heading:
        fig.suptitle(heading, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.06 * 1.6 / (size * rows) - 0.004 * fontsize, wspace=0.05, hspace=0.2)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(out)
