import argparse
import collections
import gzip
import json
import os
import queue
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DataConfig:
    dataset: str = "data/bigbank_3500.json.zip"
    max_examples: int = 0
    train_size: int = 500_000
    test_size: int = 3_000
    max_words: int = 8
    max_seq_length: int = 512
    max_text_length: int = 50
    grid: float = 0.020
    n_merges: int = 512
    augment: str = "handwriting"
    spacing: float = 0.0
    spacing_jitter: float = 0.20
    scale_jitter: float = 0.15
    rotate: float = 0.0
    shear_min: float = -0.22
    shear_max: float = -0.18
    hflip: float = 0.0
    tremor: float = 0.0
    stroke_dropout: float = 0.0
    seed: int = 1337


@dataclass
class ModelConfig:
    n_layer: int = 5
    n_head: int = 2
    n_embd: int = 64
    pen_pos_bands: int = 8
    pen_pos_jitter: int = 32
    vocab_size: int = -1
    block_size: int = -1
    context_vocab_size: int = -1
    context_block_size: int = -1


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 16
    learning_rate: float = 6e-3
    matrix_lr: float = 0.04
    weight_decay: float = 1e-4
    warmup: int = 200
    warmdown: float = 0.4
    grad_clip: float = 1.0
    eval_every: int = 1_000
    showcase: str = ""
    print_every: int = 100
    device: str = "auto"
    compile: bool = True
    out_dir: str = "out/default"
    resume: str = ""
    wandb: bool = False
    wandb_project: str = "pengpt"
    wandb_entity: str = ""
    wandb_run_name: str = ""


DERIVED_FIELDS = {"vocab_size", "block_size", "context_vocab_size",
                  "context_block_size"}

CHOICES = {"augment": ("none", "general", "handwriting")}


def add_config_args(parser):
    for cls in (DataConfig, ModelConfig, TrainConfig):
        for f in fields(cls):
            if f.name in DERIVED_FIELDS:
                continue
            if isinstance(f.default, bool):
                parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction,
                                    default=f.default)
            else:
                parser.add_argument(f"--{f.name}", type=type(f.default),
                                    default=f.default, choices=CHOICES.get(f.name))


def configs_from_args(args):
    args = vars(args)
    build = lambda cls: cls(**{f.name: args[f.name] for f in fields(cls) if f.name in args})
    return build(DataConfig), build(ModelConfig), build(TrainConfig)


DIRECTIONS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1),
                       (-1, 0), (-1, -1), (0, -1), (1, -1)])
DOWN, UP = 8, 9
N_BASE = 10

_DIR_LOOKUP = np.full((3, 3), -1, dtype=np.int64)
for _i, (_dx, _dy) in enumerate(DIRECTIONS):
    _DIR_LOOKUP[_dx + 1, _dy + 1] = _i


def bresenham_steps(x0, y0, x1, y1):
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
        self.DOWN, self.UP = DOWN, UP
        self.PAD, self.END, self.WORD, self.BOS = n, n + 1, n + 2, n + 3
        self.vocab_size = n + 4

    def token_deltas(self):
        deltas = np.zeros((self.vocab_size, 2), dtype=np.int64)
        deltas[:len(DIRECTIONS)] = DIRECTIONS
        for a, b, c in self.merges:
            deltas[c] = deltas[a] + deltas[b]
        return deltas

    def encode_word(self, points):
        return self.apply_merges(self.encode_base(points))

    def encode_many(self, base_words):
        if not self._pairs or not base_words:
            return [np.asarray(w, dtype=np.int64) for w in base_words]
        flat = np.concatenate([np.r_[w, self.PAD] for w in base_words]).astype(np.int64)
        merged = self.apply_merges(flat)
        return [w[:-1] for w in np.split(merged, np.flatnonzero(merged == self.PAD) + 1)[:-1]]

    def encode_base(self, points):
        points = np.asarray(points, dtype=float)
        grid_xy = np.rint(points[:, :2] / self.grid).astype(np.int64)
        out, previous_end = [], np.array([0, 0], dtype=np.int64)
        for start, stop in _stroke_spans(points[:, 2]):
            out.extend(_walk([previous_end, grid_xy[start]]))
            out.append(DOWN)
            out.extend(_walk(grid_xy[start:stop]))
            out.append(UP)
            previous_end = grid_xy[stop - 1]
        return np.array(out, dtype=np.int64)

    def encode_words(self, words):
        parts = []
        for i, word in enumerate(words):
            if i:
                parts.append(np.array([self.WORD] * 2, dtype=np.int64))
            parts.append(self.encode_word(word))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def apply_merges(self, tokens):
        if not self._pairs:
            return tokens
        out = np.asarray(tokens, dtype=np.int64).copy()
        for a, b, merged in self.merges:
            matches = np.flatnonzero((out[:-1] == a) & (out[1:] == b))
            if not matches.size:
                continue
            if matches.size > 1 and (np.diff(matches) == 1).any():
                starts = np.r_[True, np.diff(matches) != 1]
                run_start = np.flatnonzero(starts)[np.cumsum(starts) - 1]
                matches = matches[(np.arange(matches.size) - run_start) % 2 == 0]
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


class ByteTokenizer:

    PAD = 0
    vocab_size = 257

    def encode(self, text, length):
        ids = [b + 1 for b in text.encode()][:length]
        return np.array(ids + [self.PAD] * (length - len(ids)), dtype=np.int64)

    def decode(self, ids):
        ids = [int(i) for i in ids]
        ids = ids[:ids.index(self.PAD)] if self.PAD in ids else ids
        return bytes(i - 1 for i in ids).decode(errors="replace")


IGNORE_INDEX = -1
Y_BASELINE = 0.65
INK_HEIGHT = 0.22


def check_scale(examples, grid, n=200):
    heights = [np.quantile(d[:, 1], 0.95) - np.quantile(d[:, 1], 0.05)
               for d in (e["points"][e["points"][:, 2] == 1] for e in examples[:n])
               if len(d) > 2]
    if not heights:
        return
    ratio = float(np.median(heights)) / INK_HEIGHT
    if not 0.5 < ratio < 2:
        print(f"WARNING: ink is {ratio:.1f}x the usual size. "
              f"Sequences will be about {ratio:.1f}x as long; "
              f"pass --grid {grid * ratio:.4f} to compensate, or rescale the data.")


def read_items(path, limit=None):
    path = str(path)
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            with zf.open(zf.namelist()[0]) as f:
                yield from json.load(f)
        return
    opener = gzip.open if path.endswith(".gz") else open
    if ".jsonl" in path:
        with opener(path, "rt") as f:
            for n, line in enumerate(f):
                if limit and n >= limit:
                    return
                if line.strip():
                    yield json.loads(line)
    else:
        with opener(path, "rt") as f:
            yield from json.load(f)


def load_examples(path, limit=None):
    examples = []
    for item in read_items(path, limit):
        points = np.array(item["points"], dtype=float)
        meta = item.get("metadata", {})
        if "aspectRatio" in meta:
            points[:, 0] *= meta["aspectRatio"]
            points[:, 1] -= Y_BASELINE
        points[:, 0] -= points[0, 0]
        text = item.get("text", meta.get("asciiSequence", ""))
        examples.append({"text": text, "points": points})
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def resample(points, spacing):
    out, stroke = [], []

    def flush(stroke):
        s = np.asarray(stroke)
        if len(s) < 3:
            return list(s)
        dist = np.r_[0.0, np.cumsum(np.hypot(*np.diff(s[:, :2], axis=0).T))]
        if dist[-1] < 1e-9:
            return [s[0], s[-1]]
        n = max(2, int(round(dist[-1] / spacing)) + 1)
        t = np.linspace(0.0, dist[-1], n)
        return list(np.column_stack([np.interp(t, dist, s[:, 0]),
                                     np.interp(t, dist, s[:, 1]),
                                     np.ones(n)]))

    for point in points:
        if point[2] == 1:
            stroke.append(point)
        else:
            if stroke:
                out.extend(flush(stroke)); stroke = []
            out.append(point)
    if stroke:
        out.extend(flush(stroke))
    return np.array(out)


def prepare_word(points, cfg, rng=None):
    spacing = cfg.spacing
    if rng is not None and cfg.augment != "none":
        points = points.copy()
        if cfg.augment == "handwriting":
            points[:, 0] += rng.uniform(cfg.shear_min, cfg.shear_max) * points[:, 1]
        if cfg.rotate:
            a = np.deg2rad(rng.uniform(-cfg.rotate, cfg.rotate))
            c, s = np.cos(a), np.sin(a)
            points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]])
        points[:, 0] *= rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
        points[:, 1] *= rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
        spacing *= rng.uniform(1 - cfg.spacing_jitter, 1 + cfg.spacing_jitter)
    return resample(points, spacing) if spacing > 0 else points


def _tremor(points, rng, amp, wavelength=0.055, end_scale=0.006):
    out = points.copy()
    for start, stop in _stroke_spans(points[:, 2]):
        seg = out[start:stop, :2]
        if len(seg) < 3:
            continue
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        if d[-1] < 1e-6:
            continue
        knots = max(3, int(d[-1] / wavelength) + 2)
        offset = np.interp(d, np.linspace(0, d[-1], knots), rng.normal(0, amp, knots))
        tangent = np.gradient(seg, axis=0)
        norm = np.hypot(tangent[:, 0], tangent[:, 1]) + 1e-9
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / norm[:, None]
        seg += offset[:, None] * normal
        for end, t in ((0, seg[0] - seg[1]), (-1, seg[-1] - seg[-2])):
            delta = rng.uniform(-end_scale, end_scale)
            if delta > 0:
                seg[end] += t / (np.hypot(*t) + 1e-9) * delta
        out[start:stop, :2] = seg
    return out


def augment_drawing(points, text, cfg, rng):
    if cfg.tremor > 0:
        points = _tremor(points, rng, rng.uniform(0.3, 1.0) * cfg.tremor)
    if cfg.stroke_dropout > 0:
        spans = _stroke_spans(points[:, 2])
        if len(spans) > 1:
            drop = rng.random(len(spans)) < cfg.stroke_dropout
            if drop.all():
                drop[rng.integers(len(spans))] = False
            keep = np.ones(len(points), dtype=bool)
            for (start, stop), d in zip(spans, drop):
                if d:
                    lift = stop < len(points) and points[stop, 2] == 0
                    keep[start:stop + (1 if lift else 0)] = False
            points = points[keep]
    if cfg.hflip > 0 and rng.random() < cfg.hflip:
        lowered = text.lower()
        if "left" not in lowered and "right" not in lowered:
            points = points.copy()
            points[:, 0] = points[:, 0].max() + points[:, 0].min() - points[:, 0]
    return points


class PenDataset(Dataset):

    def __init__(self, bank_points, bank_texts, indices, stroke_tok, char_tok, cfg,
                 length, augment=True, name="", seed=0):
        self.bank_points = bank_points
        self.bank_texts = bank_texts
        self.indices = np.asarray(indices)
        self.stroke_tok = stroke_tok
        self.char_tok = char_tok
        self.cfg = cfg
        self.length = length
        self.augment = augment
        self.name = name
        self.seed = seed

    def encode_text(self, text):
        return self.char_tok.encode(text, self.cfg.max_text_length)

    def __len__(self):
        return self.length

    def draw_words(self, rng):
        draw = min(self.cfg.max_words, len(self.indices))
        limit = rng.integers(1, draw + 1)
        chosen = rng.choice(self.indices, size=draw, replace=False)[:limit]
        base = []
        for i in chosen:
            word = prepare_word(self.bank_points[i], self.cfg,
                                rng if self.augment else None)
            if self.augment:
                word = augment_drawing(word, self.bank_texts[i], self.cfg, rng)
            base.append(self.stroke_tok.encode_base(word))
        return chosen, base

    def pack(self, chosen, words):
        block = self.cfg.max_seq_length
        kept, parts, total = [], [], 0
        for i, tokens in zip(chosen, words):
            extra = len(tokens) + (2 if parts else 0)
            if parts and total + extra > block - 2:
                break
            if parts:
                parts.append(np.array([self.stroke_tok.WORD] * 2, dtype=np.int64))
            parts.append(tokens)
            total += extra
            kept.append(i)
        return np.concatenate(parts), " ".join(self.bank_texts[i] for i in kept)

    def pick_words(self, rng):
        chosen, base = self.draw_words(rng)
        return self.pack(chosen, self.stroke_tok.encode_many(base))

    def batch(self, indices):
        draws = [self.draw_words(np.random.default_rng([self.seed, int(i)])) for i in indices]
        merged = iter(self.stroke_tok.encode_many([w for _, base in draws for w in base]))
        items = [self.make_item(*self.pack(chosen, [next(merged) for _ in base]))
                 for chosen, base in draws]
        return tuple(torch.stack(t) for t in zip(*items))

    def __getitem__(self, idx):
        return self.make_item(*self.pick_words(np.random.default_rng([self.seed, idx])))

    def make_item(self, tokens, text):
        st, block = self.stroke_tok, self.cfg.max_seq_length
        x = torch.full((block,), st.PAD, dtype=torch.long)
        y = torch.full((block,), IGNORE_INDEX, dtype=torch.long)
        n = min(len(tokens), block - 2)
        x[0] = st.BOS
        x[1:n + 1] = torch.from_numpy(tokens[:n])
        x[n + 1] = st.END
        y[:n + 1] = x[1:n + 2]
        if len(tokens) > n:
            y[n] = IGNORE_INDEX
        return x, torch.from_numpy(self.encode_text(text)), y


def create_datasets(cfg, merges=None):
    examples = load_examples(cfg.dataset, cfg.max_examples or None)
    check_scale(examples, cfg.grid)
    bank_points = [e["points"] for e in examples]
    bank_texts = [e["text"] for e in examples]
    char_tok = ByteTokenizer()

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(examples))
    n_test = min(1000, max(10, int(0.05 * len(examples))))
    train_ix, test_ix = perm[:-n_test], perm[-n_test:]

    if merges is None:
        base = ScribeTokenizer(grid=cfg.grid)
        sample = rng.choice(train_ix, size=min(600, len(train_ix)), replace=False)
        corpus = [base.encode_word(prepare_word(bank_points[i], cfg, rng)) for i in sample]
        merges = learn_merges(corpus, cfg.n_merges)
        print(f"Learned {len(merges)} BPE merges")
    stroke_tok = ScribeTokenizer(grid=cfg.grid, merges=merges)

    def build(ix, n, name, seed):
        return PenDataset(bank_points, bank_texts, ix, stroke_tok, char_tok, cfg,
                          length=n, augment=cfg.augment != "none", name=name, seed=seed)

    train_dataset = build(train_ix, cfg.train_size, "train", cfg.seed)
    test_dataset = build(test_ix, cfg.test_size, "test", cfg.seed + 1)
    print(f"Word bank: {len(train_ix)} train / {len(test_ix)} test words; "
          f"stroke vocab {stroke_tok.vocab_size}")
    return train_dataset, test_dataset, stroke_tok, char_tok


def infinite_batches(dataset, batch_size, seed=0, prefetch=4):
    rng = np.random.default_rng(seed)
    q = queue.Queue(maxsize=prefetch)

    def produce():
        while True:
            q.put(dataset.batch(rng.integers(0, len(dataset), batch_size)))

    threading.Thread(target=produce, daemon=True).start()
    while True:
        yield q.get()


def _split_heads(t, n_head):
    B, T, C = t.shape
    return t.view(B, T, n_head, C // n_head).transpose(1, 2)


def _merge_heads(t):
    B, nh, T, hs = t.shape
    return t.transpose(1, 2).contiguous().view(B, T, nh * hs)


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rotary_tables(T, head_dim, device, base=10_000):
    inv_freq = base ** -(torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(T, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rotary(x, cos, sin):
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.kv = nn.Linear(cfg.n_embd, 2 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.q_norm = nn.RMSNorm(cfg.n_embd // cfg.n_head)
        self.k_norm = nn.RMSNorm(cfg.n_embd // cfg.n_head)
        self.n_head = cfg.n_head

    def forward(self, x, source=None, rotary=None, mask=None, cache=None):
        q = _split_heads(self.q(x), self.n_head)
        k, v = (_split_heads(t, self.n_head)
                for t in self.kv(x if source is None else source).split(x.size(-1), dim=2))
        if rotary is not None:
            q, k = apply_rotary(q, *rotary), apply_rotary(k, *rotary)
        q, k = self.q_norm(q), self.k_norm(k)
        if cache is not None:
            buf, slot = cache
            buf.copy_(torch.where(slot, torch.stack([k, v]), buf))
            k, v = buf
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        return self.proj(_merge_heads(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.cross_attn = Attention(cfg)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, context, rotary, ctx_mask, self_mask=None, cache=None):
        x = x + self.attn(rms_norm(x), rotary=rotary, mask=self_mask, cache=cache)
        x = x + self.cross_attn(rms_norm(x), context, mask=ctx_mask)
        return x + self.fc_out(F.relu(self.fc(rms_norm(x))).square())


class PenTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "fill in the derived ModelConfig fields first"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.ctx_emb = nn.Embedding(cfg.context_vocab_size, cfg.n_embd)
        self.ctx_pos_emb = nn.Embedding(cfg.context_block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.resid_scale = nn.Parameter(torch.ones(cfg.n_layer))
        self.x0_scale = nn.Parameter(torch.zeros(cfg.n_layer))
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.register_buffer("pen_deltas", torch.zeros(cfg.vocab_size, 2, dtype=torch.long))
        if cfg.pen_pos_bands > 0:
            self.pen_pos_proj = nn.Linear(4 * cfg.pen_pos_bands, cfg.n_embd, bias=False)
        self.init_weights()
        print(f"PenTransformer parameters: {sum(p.numel() for p in self.parameters()):,}")

    @torch.no_grad()
    def init_weights(self):
        s = 3 ** 0.5 * self.cfg.n_embd ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "fc_out.weight")):
                nn.init.zeros_(p)
            elif name.endswith("fc.weight"):
                nn.init.uniform_(p, -0.4 * s, 0.4 * s)
            elif p.ndim == 2 and (name.startswith("blocks.") or name.startswith("pen_pos")):
                nn.init.uniform_(p, -s, s)
        for emb in (self.tok_emb, self.ctx_emb, self.ctx_pos_emb):
            nn.init.normal_(emb.weight, std=0.8)
        nn.init.normal_(self.head.weight, std=0.001)

    def embed(self, idx, pen_xy):
        x = self.tok_emb(idx)
        if self.cfg.pen_pos_bands > 0:
            k = torch.arange(self.cfg.pen_pos_bands, device=idx.device)
            angles = pen_xy[..., None].float() * (torch.pi / 2 ** (k + 1))
            x = x + self.pen_pos_proj(torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2))
        return rms_norm(x)

    def encode_context(self, context):
        c = rms_norm(self.ctx_emb(context) + self.ctx_pos_emb.weight[:context.size(1)])
        mask = (context != 0)[:, None, None, :]
        return c, mask | ~mask.any(-1, keepdim=True)

    def logits(self, x, c, ctx_mask, rotary, self_mask=None, caches=None):
        x0 = x
        for i, block in enumerate(self.blocks):
            x = block(self.resid_scale[i] * x + self.x0_scale[i] * x0, c, rotary, ctx_mask,
                      self_mask, caches[i] if caches else None)
        return 15 * torch.tanh(self.head(rms_norm(x)) / 15)

    def forward(self, idx, context, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"
        pen_xy = self.pen_deltas[idx].cumsum(dim=1)
        if self.training and self.cfg.pen_pos_jitter > 0:
            j = self.cfg.pen_pos_jitter
            pen_xy = pen_xy + torch.randint(-j, j + 1, (B, 1, 2), device=idx.device)
        rotary = rotary_tables(T, self.cfg.n_embd // self.cfg.n_head, idx.device)
        logits = self.logits(self.embed(idx, pen_xy), *self.encode_context(context), rotary)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(),
                                   ignore_index=IGNORE_INDEX)
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, context, max_new_tokens, temperature=1.0, top_k=None,
                 do_sample=True, end_token=None, pad_token=None):
        was_training = self.training
        self.eval()
        B, T0 = idx.shape
        n, dev = min(T0 + max_new_tokens, self.cfg.block_size), idx.device
        head_dim = self.cfg.n_embd // self.cfg.n_head
        caches = [torch.zeros(2, B, self.cfg.n_head, n, head_dim, device=dev) for _ in self.blocks]
        cos, sin = rotary_tables(n, head_dim, dev)
        c, ctx_mask = self.encode_context(context)
        slots, t = torch.arange(n, device=dev), torch.zeros(1, dtype=torch.long, device=dev)
        pen_xy = torch.zeros(B, 1, 2, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        tokens = list(idx.split(1, dim=1))
        for step in range(n - 1):
            pen_xy = pen_xy + self.pen_deltas[tokens[step]]
            rotary = cos.index_select(0, t), sin.index_select(0, t)
            mask = (slots <= t)[None, None, None]
            slot = (slots == t)[:, None]
            logits = self.logits(self.embed(tokens[step], pen_xy), c, ctx_mask, rotary, mask,
                                 [(buf, slot) for buf in caches])[:, -1]
            t += 1
            if step < T0 - 1:
                continue
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = (torch.multinomial(F.softmax(logits, dim=-1), 1) if do_sample
                   else logits.argmax(dim=-1, keepdim=True))
            if end_token is not None:
                nxt[done] = pad_token if pad_token is not None else end_token
                done |= nxt.squeeze(1) == end_token
            tokens.append(nxt)
            if end_token is not None and done.all():
                break
        self.train(was_training)
        return torch.cat(tokens, dim=1)


POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def orthogonalize(G):
    if G.size(-2) > G.size(-1):
        return orthogonalize(G.mT).mT
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in POLAR_EXPRESS:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        params = list(params)
        groups = [dict(params=[p for p in params if p.shape == shape])
                  for shape in sorted({p.shape for p in params})]
        super().__init__(groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group["params"]
            grads = torch.stack([p.grad for p in params])
            state = self.state[params[0]]
            if "buf" not in state:
                state["buf"] = torch.zeros_like(grads)
            state["buf"].lerp_(grads, 1 - group["momentum"])
            update = orthogonalize(grads.lerp_(state["buf"], group["momentum"]))
            rows, cols = params[0].shape
            lr = group["lr"] * max(1.0, rows / cols) ** 0.5
            for p, u in zip(params, update):
                p.mul_(1 - lr * group["weight_decay"]).add_(u.to(p.dtype), alpha=-lr)


def make_optimizers(model, cfg):
    is_matrix = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
    opts = [torch.optim.AdamW([p for n, p in model.named_parameters() if not is_matrix(n, p)],
                              lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                              betas=(0.9, 0.99), eps=1e-8),
            Muon([p for n, p in model.named_parameters() if is_matrix(n, p)], lr=cfg.matrix_lr)]
    for opt in opts:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return opts


def lr_multiplier(step, cfg):
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    warmdown = max(1, int(cfg.warmdown * cfg.steps))
    return min(1.0, max(0.0, (cfg.steps - step) / warmdown))


def attach_pen_deltas(model, stroke_tok):
    model.pen_deltas.copy_(torch.as_tensor(stroke_tok.token_deltas(),
                                               device=model.pen_deltas.device))


def save_checkpoint(path, model, data_config, merges=None, optimizers=None,
                    step=None, best_loss=None):
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.cfg),
        "data_config": data_config,
        "merges": [[int(a), int(b), int(c)] for a, b, c in (merges or [])],
        "step": int(step) if step is not None else None,
        "best_loss": float(best_loss) if best_loss is not None else None,
    }
    if optimizers is not None:
        checkpoint["optimizer"] = [opt.state_dict() for opt in optimizers]
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = PenTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint


def load_for_sampling(path, device="cpu"):
    model, checkpoint = load_checkpoint(path, device)
    cfg = DataConfig(**checkpoint["data_config"])
    stroke_tok = ScribeTokenizer(grid=cfg.grid, merges=checkpoint["merges"])
    attach_pen_deltas(model, stroke_tok)
    dataset = PenDataset([], [], [], stroke_tok, ByteTokenizer(), cfg, length=0)
    return model, dataset, cfg, checkpoint


@dataclass
class SampleParams:
    temperature: float = 1.0
    top_k: int = None
    do_sample: bool = True
    max_tokens: int = 512
    n_at_a_time: int = 4
    space_width: float = 0.16
    line_width: float = 8.0
    line_height: float = 0.55
    seed: int = 42
    linewidth: float = 1.3
    verbose: bool = True


def generate(model, dataset, text, params=None):
    params = params or SampleParams()
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    context = torch.from_numpy(dataset.encode_text(text)).unsqueeze(0).to(device)
    idx = torch.full((1, 1), st.BOS, dtype=torch.long, device=device)
    out = model.generate(idx, context, max_new_tokens=params.max_tokens,
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END, pad_token=st.PAD)
    return st.decode(out[0].cpu().numpy()[1:])


def draw(ax, points, color="b", linewidth=1.3):
    points = np.asarray(points, dtype=float)
    if len(points):
        pen_down = points[:, 2] == 1
        for chunk in np.split(points, np.flatnonzero(~pen_down) + 1):
            chunk = chunk[chunk[:, 2] == 1]
            if len(chunk) > 1:
                ax.plot(chunk[:, 0], -chunk[:, 1], color=color, linewidth=linewidth,
                        solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def layout_words(words, params=None):
    params = params or SampleParams()
    placed, x, y = [], 0.0, 0.0
    for points in words:
        points = np.asarray(points, dtype=float).copy()
        if len(points) == 0:
            placed.append(np.array([[x, y, 0.0]]))
            continue
        points[:, 0] -= points[:, 0].min()
        width = points[:, 0].max()
        if x > 0 and x + width > params.line_width:
            x, y = 0.0, y + params.line_height
        points[:, 0] += x
        points[:, 1] += y
        placed.append(points)
        x += width + params.space_width
    return placed


def plot_words(words, params=None, title="", figsize=(12, 2), dpi=150, ax=None,
               color="b"):
    params = params or SampleParams()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    placed = layout_words(words, params)
    draw(ax, np.vstack(placed) if placed else np.zeros((0, 3)), color, params.linewidth)
    if title:
        ax.set_title(title)
    return fig, ax, placed


def plot_paragraph(words, params=None, figsize=(12, 8), dpi=200, show_indices=False):
    params = params or SampleParams()
    fig, ax, placed = plot_words(words, params, figsize=figsize, dpi=dpi)
    if show_indices:
        for i, points in enumerate(placed):
            ax.text(points[:, 0].min() - 0.08, -points[0, 1] + 0.15, str(i), fontsize=8)
    return fig, ax


def generate_paragraph(model, dataset, text, params=None, words=None, redo=None):
    params = params or SampleParams()
    fits = max(1, (params.max_tokens - 1) // 120)
    if params.n_at_a_time > fits:
        print(f"  n_at_a_time={params.n_at_a_time} likely overflows a "
              f"{params.max_tokens}-token block; using {fits}")
        params = replace(params, n_at_a_time=fits)
    torch.manual_seed(params.seed)
    prompt_words = text.strip().split()

    def for_chunk(chunk):
        out = generate(model, dataset, " ".join(chunk), params)[:len(chunk)]
        if params.verbose:
            print(f"  {' '.join(chunk)}")
        return out + [np.zeros((0, 3))] * (len(chunk) - len(out))

    if words is None:
        words = []
        for i in range(0, len(prompt_words), params.n_at_a_time):
            words += for_chunk(prompt_words[i:i + params.n_at_a_time])
    else:
        for i in redo or []:
            if i < len(prompt_words):
                words[i] = for_chunk([prompt_words[i]])[0]
    return words


def save_progress(model, dataset, out_dir, step, showcase=(), n=6, samples=3):
    X, C, _ = dataset.batch(range(n))
    C = torch.cat([C] + [torch.from_numpy(dataset.encode_text(t))[None] for t in showcase])
    st, device, rows = dataset.stroke_tok, next(model.parameters()).device, len(C)
    gen = lambda k, do_sample: model.generate(
        torch.full((rows * k, 1), st.BOS, device=device), C.repeat_interleave(k, 0).to(device),
        dataset.cfg.max_seq_length - 1, do_sample=do_sample, end_token=st.END, pad_token=st.PAD).cpu()
    torch.manual_seed(3)
    greedy, out = gen(1, False), gen(samples, True)
    fig, axes = plt.subplots(rows, samples + 2, figsize=(3 * (samples + 2), 1.1 * rows),
                             squeeze=False, sharex="row", sharey="row")
    params = SampleParams(line_width=2.5)
    for i in range(rows):
        drawings = [X[i] if i < n else None, greedy[i]] + list(out[i * samples:(i + 1) * samples])
        for j, (ax, tokens) in enumerate(zip(axes[i], drawings)):
            words = [] if tokens is None else st.decode(tokens.numpy())
            plot_words(words, params, ax=ax, color=("darkgreen", "navy")[j] if j < 2 else "k")
            if tokens is None:
                ax.text(0.5, 0.5, "showcase prompt:\nno real drawing", transform=ax.transAxes,
                        ha="center", va="center", color="0.6", fontsize=8)
        axes[i][0].set_title(dataset.char_tok.decode(C[i]), loc="left", fontsize=8, pad=2)
    fig.suptitle(f"step {step:,}: real test example (green), greedy (blue), samples (black)",
                 fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.03, hspace=0.3)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=100, max_batches=10):
    model.eval()
    if not hasattr(dataset, "_eval_batches"):
        n = min(len(dataset), batch_size * max_batches)
        dataset._eval_batches = [dataset.batch(range(i, min(i + batch_size, n)))
                                 for i in range(0, n, batch_size)]
    total = count = 0
    for X, C, Y in dataset._eval_batches:
        _, loss = model(X.to(device), C.to(device), Y.to(device))
        n_tok = (Y != IGNORE_INDEX).sum().item()
        total += loss.item() * n_tok
        count += n_tok
    model.train()
    return total / count


def train(data_cfg, model_cfg, train_cfg):
    device = resolve_device(train_cfg.device)
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_cfg.out_dir, "best.pt")

    ckpt = None
    if train_cfg.resume:
        _, ckpt = load_checkpoint(train_cfg.resume, device)
        resumed_cfg = DataConfig(**ckpt["data_config"])
        if asdict(resumed_cfg) != asdict(data_cfg):
            changed = {k: (v, getattr(data_cfg, k))
                       for k, v in asdict(resumed_cfg).items()
                       if getattr(data_cfg, k) != v}
            print(f"Using the checkpoint's data config; ignoring {changed} "
                  f"(checkpoint value, command-line value)")
        data_cfg = resumed_cfg

    torch.manual_seed(data_cfg.seed)
    train_dataset, test_dataset, stroke_tok, char_tok = create_datasets(
        data_cfg, merges=ckpt["merges"] if ckpt else None)
    model_cfg.vocab_size = stroke_tok.vocab_size
    model_cfg.block_size = data_cfg.max_seq_length
    model_cfg.context_vocab_size = char_tok.vocab_size
    model_cfg.context_block_size = data_cfg.max_text_length

    model = PenTransformer(model_cfg).to(device)
    attach_pen_deltas(model, stroke_tok)
    optimizers = make_optimizers(model, train_cfg)

    step, best_loss = 0, None
    if ckpt:
        model.load_state_dict(ckpt["model"])
        for opt, state in zip(optimizers, ckpt["optimizer"]):
            opt.load_state_dict(state)
        step, best_loss = ckpt["step"], ckpt["best_loss"]
        print(f"Resumed from {train_cfg.resume} at step {step} (best loss {best_loss})")

    run = None
    if train_cfg.wandb:
        import wandb
        run = wandb.init(project=train_cfg.wandb_project,
                         entity=train_cfg.wandb_entity or None,
                         name=train_cfg.wandb_run_name or None,
                         config={**asdict(data_cfg), **asdict(model_cfg), **asdict(train_cfg)})

    batches = infinite_batches(train_dataset, train_cfg.batch_size, seed=data_cfg.seed + step)

    step_model = torch.compile(model) if train_cfg.compile and device != "cpu" else model
    t_start = t_print = time.time()
    while step < train_cfg.steps:
        X, C, Y = [t.to(device, non_blocking=True) for t in next(batches)]
        _, loss = step_model(X, C, Y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        mult = lr_multiplier(step, train_cfg)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * mult
                if isinstance(opt, Muon):
                    group["momentum"] = 0.85 + 0.10 * min(1.0, step / 300)
            opt.step()
        step += 1

        if run:
            run.log({"train_loss_step": loss.item(), "step": step})
        if step % train_cfg.print_every == 0:
            ms = (time.time() - t_print) * 1000 / train_cfg.print_every
            print(f"step {step} | loss {loss.item():.4f} | {ms:.0f} ms/step | lr x{mult:.3f}")
            t_print = time.time()
        if step % train_cfg.eval_every and step < train_cfg.steps:
            continue

        train_loss = evaluate(model, train_dataset, device)
        test_loss = evaluate(model, test_dataset, device)
        rank = text_rank(model, test_dataset, device)
        print(f"step {step} | train loss {train_loss:.4f} | test loss {test_loss:.4f}"
              f" | text rank {rank:.2f}/32 | {(time.time() - t_start) / 60:.1f} min")
        if run:
            run.log({"train_loss": train_loss, "test_loss": test_loss, "text_rank": rank,
                     "step": step})

        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss
            print(f"New best test loss; saving checkpoint to {checkpoint_path}")
            save_checkpoint(checkpoint_path, model, asdict(data_cfg),
                            stroke_tok.merges, optimizers, step, best_loss)
            if run:
                artifact = wandb.Artifact("best_checkpoint", type="model")
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact)

        save_checkpoint(os.path.join(train_cfg.out_dir, "last.pt"), model,
                        asdict(data_cfg), stroke_tok.merges,
                        optimizers, step, best_loss)

        path = save_progress(model, test_dataset, os.path.join(train_cfg.out_dir, "progress"), step,
                             [t for t in train_cfg.showcase.split("|") if t])
        print(f"  wrote {path}")
        if run:
            run.log({"progress": wandb.Image(path), "step": step})
        t_print = time.time()

    if run:
        run.finish()


def sample_command(args):
    device = resolve_device(args.device)
    model, dataset, cfg, _ = load_for_sampling(args.checkpoint, device)
    params = SampleParams(temperature=args.temperature, top_k=args.top_k,
                          do_sample=not args.greedy, n_at_a_time=args.n_at_a_time,
                          max_tokens=cfg.max_seq_length - 1, seed=args.seed)
    words = generate_paragraph(model, dataset, args.text, params)
    if args.redo:
        redo = [int(i) for i in args.redo.split(",") if i.strip()]
        words = generate_paragraph(model, dataset, args.text, params,
                                   words=words, redo=redo)
    fig, _ = plot_paragraph(words, params, show_indices=args.show_indices)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


@torch.inference_mode()
def text_rank(model, dataset, device, n=32):
    X, C, Y = (t.to(device) for t in dataset.batch(range(n)))
    model.eval()
    nll = torch.stack([F.cross_entropy(model(X, C[j].expand_as(C))[0].transpose(1, 2), Y,
                                       ignore_index=IGNORE_INDEX, reduction="none").sum(1)
                       for j in range(n)], dim=1)
    model.train()
    return 1 + (nll < nll.diagonal()[:, None]).sum(1).float().mean().item()


def rank_command(args):
    device = resolve_device(args.device)
    _, _, cfg, ckpt = load_for_sampling(args.checkpoint, device)
    model, _ = load_checkpoint(args.checkpoint, device)
    _, test, stroke_tok, _ = create_datasets(cfg, merges=ckpt["merges"])
    attach_pen_deltas(model, stroke_tok)
    print(f"step {ckpt['step']}: true text ranks {text_rank(model, test, device, args.n):.2f} "
          f"of {args.n} (chance {(args.n + 1) / 2:.1f})")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pengpt: a minimal transformer for pen strokes")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train a model on one dataset")
    train_p.add_argument("--preset", type=str, default="",
                         help="per-dataset settings from utils.PRESETS "
                              "(cursive, quickdraw, icons); "
                              "explicit flags override the preset")
    add_config_args(train_p)

    sample_p = sub.add_parser("sample", help="generate from a trained checkpoint")
    sample_p.add_argument("--checkpoint", type=str, default="out/cursive/best.pt")
    sample_p.add_argument("--text", type=str, required=True)
    sample_p.add_argument("--out", type=str, default="sample.png")
    sample_p.add_argument("--temperature", type=float, default=1.0)
    sample_p.add_argument("--top_k", type=int, default=None)
    sample_p.add_argument("--greedy", action="store_true")
    sample_p.add_argument("--n_at_a_time", type=int, default=2)
    sample_p.add_argument("--redo", type=str, default="",
                          help="comma-separated word indices to regenerate")
    sample_p.add_argument("--seed", type=int, default=42)
    sample_p.add_argument("--show_indices", action="store_true")
    sample_p.add_argument("--device", type=str, default="auto")

    rank_p = sub.add_parser("rank", help="prompt conditioning: rank each test drawing's "
                                         "true text among other test texts")
    rank_p.add_argument("--checkpoint", type=str, default="out/cursive/best.pt")
    rank_p.add_argument("--n", type=int, default=32)
    rank_p.add_argument("--device", type=str, default="auto")

    known, _ = parser.parse_known_args(argv)
    if getattr(known, "preset", ""):
        from utils import PRESETS
        if known.preset not in PRESETS:
            raise SystemExit(f"unknown preset {known.preset!r}; "
                             f"choose from {sorted(PRESETS)}")
        train_p.set_defaults(**PRESETS[known.preset])
    args = parser.parse_args(argv)

    commands = {"train": lambda a: train(*configs_from_args(a)),
                "sample": sample_command, "rank": rank_command}
    commands[args.command](args)


if __name__ == "__main__":
    main()
