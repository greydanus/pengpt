# pengpt: a minimal transformer for pen strokes, in one file.
#
# Ink becomes tokens -- a walk on a grid (8 directions + pen DOWN/UP), shortened
# by BPE -- and a small GPT-style decoder predicts the next move, steered by
# cross-attention over the character-level text prompt. The file reads top to
# bottom as the pipeline runs: config, tokenizer, data, model, sampling, commands.
# Per-dataset presets and corpus-building tools live in utils.py.
#
#   python pengpt.py train --preset cursive    # ~45 min on an M-series laptop
#   python pengpt.py sample --checkpoint out/cursive/best.pt --text "hello world"

import argparse
import collections
import gzip
import json
import math
import os
import textwrap
import time
import zipfile
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # figures only ever go to files; interactive backends stall headless runs
import matplotlib.pyplot as plt


@dataclass
class DataConfig:
    dataset: str = "data/bigbank_3500.json.zip"
    max_examples: int = 0
    train_size: int = 500_000
    test_size: int = 3_000
    max_words: int = 8              # words packed per training sequence
    max_seq_length: int = 512
    max_text_length: int = 50
    grid: float = 0.020             # tokenizer cell size: reconstruction error vs sequence length
    n_merges: int = 512
    augment: str = "handwriting"
    spacing: float = 0.0            # resample to uniform arc length; 0 = data already uniform
    spacing_jitter: float = 0.20
    scale_jitter: float = 0.15
    rotate: float = 0.0
    shear_min: float = -0.22
    shear_max: float = -0.18
    hflip: float = 0.0              # mirror drawings; skipped when the caption says left/right
    tremor: float = 0.0             # humanize ruler-perfect ink (icon sets); wrong for human ink
    stroke_dropout: float = 0.0     # drop whole strokes; scenes survive it, words do not
    seed: int = 1337


@dataclass
class ModelConfig:
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 64
    pen_pos_bands: int = 8          # fourier features of absolute pen position; 0 disables
    pen_pos_jitter: int = 32        # train-time canvas offset, so layout cannot be memorized
    vocab_size: int = -1
    block_size: int = -1
    context_vocab_size: int = -1
    context_block_size: int = -1


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    warmup: int = 200
    grad_clip: float = 1.0
    eval_every: int = 1_000
    print_every: int = 100
    num_workers: int = 0            # loading never bottlenecks; workers deadlock on macOS py3.14
    device: str = "auto"
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


def _filter_config(cls, d):
    known = {f.name for f in fields(cls)}
    dropped = {k: v for k, v in d.items() if k not in known and v}
    if dropped:
        print(f"WARNING: checkpoint used removed features, ignoring {dropped}")
    return {k: v for k, v in d.items() if k in known}


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
    err = dx - dy  # fixed tie-breaking: one line must always encode one way, or BPE splits it
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
    if (np.abs(deltas) <= 1).all():  # dense recordings: every move is one direction token
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
            deltas[c] = deltas[a] + deltas[b]  # a merged token moves by the sum of its children
        return deltas

    def encode_word(self, points):
        points = np.asarray(points, dtype=float)
        grid_xy = np.rint(points[:, :2] / self.grid).astype(np.int64)
        out, previous_end = [], np.array([0, 0], dtype=np.int64)  # start at the baseline origin, so tokens carry word height
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
                parts.append(np.array([self.WORD] * 2, dtype=np.int64))
            parts.append(self.encode_word(word))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def apply_merges(self, tokens):
        if not self._pairs:
            return tokens
        out = np.asarray(tokens, dtype=np.int64).copy()
        for a, b, merged in self.merges:  # one pass per rule, in learned order: earlier rules win shared tokens
            matches = np.flatnonzero((out[:-1] == a) & (out[1:] == b))
            if not matches.size:
                continue
            if matches.size > 1 and (np.diff(matches) == 1).any():  # runs like [a,a,a] under rule (a,a)
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
                if DOWN not in pair and UP not in pair:  # never merge across a stroke boundary
                    counts[pair] += 1
        if not counts:
            break
        (a, b), count = counts.most_common(1)[0]
        if count < min_count:  # a rule that rarely fires costs vocabulary without shortening anything
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


IGNORE_INDEX = -1
Y_BASELINE = 0.65
INK_HEIGHT = 0.22


def check_scale(examples, grid, n=200):
    heights = [np.quantile(d[:, 1], 0.95) - np.quantile(d[:, 1], 0.05)
               for d in (e["points"][e["points"][:, 2] == 1] for e in examples[:n])
               if len(d) > 2]
    if not heights:
        return
    ratio = float(np.median(heights)) / INK_HEIGHT  # token cost scales with ink size over grid size
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
        if "aspectRatio" in meta:  # collect.html format
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
            points[:, 0] += rng.uniform(cfg.shear_min, cfg.shear_max) * points[:, 1]  # italic slant; presumes a baseline
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
        seg += offset[:, None] * normal  # smooth waver along the path, not per-point jitter
        for end, t in ((0, seg[0] - seg[1]), (-1, seg[-1] - seg[-2])):
            delta = rng.uniform(-end_scale, end_scale)
            if delta > 0:
                seg[end] += t / (np.hypot(*t) + 1e-9) * delta
        out[start:stop, :2] = seg
    return out


def augment_drawing(points, text, cfg, rng):
    if cfg.tremor > 0:
        points = _tremor(points, rng, rng.uniform(0.3, 1.0) * cfg.tremor)  # careful hands to casual ones
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
        if "left" not in lowered and "right" not in lowered:  # the text must never lie about the picture
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

    def pick_words(self, rng):
        block = self.cfg.max_seq_length
        draw = min(self.cfg.max_words, len(self.indices))
        limit = rng.integers(1, draw + 1)  # sometimes stop short, so END decorrelates from block position
        chosen, parts, total = [], [], 0
        for i in rng.choice(self.indices, size=draw, replace=False)[:limit]:
            word = prepare_word(self.bank_points[i], self.cfg,
                                rng if self.augment else None)
            if self.augment:
                word = augment_drawing(word, self.bank_texts[i], self.cfg, rng)
            tokens = self.stroke_tok.encode_word(word)
            extra = len(tokens) + (2 if parts else 0)
            if parts and total + extra > block - 2:  # never truncate mid-word; leave room for BOS and END
                break
            if parts:
                parts.append(np.array([self.stroke_tok.WORD] * 2, dtype=np.int64))
            parts.append(tokens)
            total += extra
            chosen.append(i)
        return np.concatenate(parts), " ".join(self.bank_texts[i] for i in chosen)

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, idx])
        tokens, text = self.pick_words(rng)
        st, block = self.stroke_tok, self.cfg.max_seq_length
        x = torch.full((block,), st.PAD, dtype=torch.long)
        y = torch.full((block,), IGNORE_INDEX, dtype=torch.long)
        n = min(len(tokens), block - 2)
        x[0] = st.BOS  # generation seeds with the same BOS, so its first step was seen in training
        x[1:n + 1] = torch.from_numpy(tokens[:n])
        x[n + 1] = st.END
        y[:n + 1] = x[1:n + 2]
        if len(tokens) > n:
            y[n] = IGNORE_INDEX  # a word cut by the block did not actually end there
        return x, torch.from_numpy(self.encode_text(text)), y

    def text_for(self, idx):
        return self.pick_words(np.random.default_rng([self.seed, idx]))[1]


def create_datasets(cfg, merges=None):
    examples = load_examples(cfg.dataset, cfg.max_examples or None)
    check_scale(examples, cfg.grid)
    bank_points = [e["points"] for e in examples]
    bank_texts = [e["text"] for e in examples]
    alphabet = " " + "".join(sorted(set("".join(bank_texts)) - {" "}))
    char_tok = CharTokenizer(alphabet)

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
          f"stroke vocab {stroke_tok.vocab_size}; "
          f"alphabet ({len(alphabet)} chars): {alphabet!r}")
    return train_dataset, test_dataset, stroke_tok, char_tok


def infinite_batches(dataset, **kwargs):
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True,
                                             num_samples=int(1e10))
    loader = DataLoader(dataset, sampler=sampler, **kwargs)
    while True:
        yield from loader


def _split_heads(t, n_head):
    B, T, C = t.shape
    return t.view(B, T, n_head, C // n_head).transpose(1, 2)


def _merge_heads(t):
    B, nh, T, hs = t.shape
    return t.transpose(1, 2).contiguous().view(B, T, nh * hs)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head

    def forward(self, x):
        q, k, v = (_split_heads(t, self.n_head) for t in self.qkv(x).split(x.size(-1), dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(_merge_heads(y))


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.kv = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head

    def forward(self, x, context, mask=None):
        q = _split_heads(self.q(x), self.n_head)
        k, v = (_split_heads(t, self.n_head) for t in self.kv(context).split(x.size(-1), dim=2))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(_merge_heads(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.cross_attn = CrossAttention(cfg)
        self.ln3 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, context, ctx_mask=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), context, ctx_mask)
        x = x + self.mlp(self.ln3(x))
        return x


class PenTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "fill in the derived ModelConfig fields first"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.ctx_emb = nn.Embedding(cfg.context_vocab_size, cfg.n_embd)
        self.ctx_pos_emb = nn.Embedding(cfg.context_block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.pen_pos_bands > 0:
            self.register_buffer("pen_deltas",
                                 torch.zeros(cfg.vocab_size, 2, dtype=torch.long))
            self.pen_pos_proj = nn.Linear(4 * cfg.pen_pos_bands, cfg.n_embd, bias=False)
        self.apply(self._init_weights)
        print(f"PenTransformer parameters: {sum(p.numel() for p in self.parameters()):,}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _pen_features(self, idx):
        positions = self.pen_deltas[idx].cumsum(dim=1)  # inclusive cumsum: pen after each token, still causal
        if self.training and self.cfg.pen_pos_jitter > 0:
            jitter = self.cfg.pen_pos_jitter
            positions = positions + torch.randint(-jitter, jitter + 1,
                                                  (idx.size(0), 1, 2), device=idx.device)
        k = torch.arange(self.cfg.pen_pos_bands, device=idx.device)
        angles = positions[..., None].float() * (torch.pi / (2.0 * 2.0 ** k))
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)

    def forward(self, idx, context, targets=None):
        T = idx.size(1)
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        if self.cfg.pen_pos_bands > 0:
            x = x + self.pen_pos_proj(self._pen_features(idx))

        ctx_pos = torch.arange(context.size(1), device=idx.device)
        c = self.ctx_emb(context) + self.ctx_pos_emb(ctx_pos)
        ctx_mask = (context != 0)[:, None, None, :]
        ctx_mask = ctx_mask | ~ctx_mask.any(-1, keepdim=True)  # all-pad prompt: attend uniformly rather than NaN

        for block in self.blocks:
            x = block(x, c, ctx_mask)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=IGNORE_INDEX)
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, context, max_new_tokens, temperature=1.0, top_k=None,
                 do_sample=True, end_token=None, pad_token=None):
        was_training = self.training
        self.eval()
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond, context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            if do_sample:
                idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            if end_token is not None:
                idx_next[done] = pad_token if pad_token is not None else end_token
                done |= idx_next.squeeze(1) == end_token
            idx = torch.cat([idx, idx_next], dim=1)
            if end_token is not None and done.all():
                break
        self.train(was_training)
        return idx


def attach_pen_deltas(model, stroke_tok):
    if hasattr(model, "pen_deltas"):
        model.pen_deltas.copy_(torch.as_tensor(stroke_tok.token_deltas(),
                                               device=model.pen_deltas.device))


def save_checkpoint(path, model, alphabet, data_config, merges=None, optimizer=None,
                    scheduler=None, step=None, best_loss=None):
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.cfg),
        "alphabet": alphabet,
        "data_config": data_config,
        "merges": [[int(a), int(b), int(c)] for a, b, c in (merges or [])],
        "step": int(step) if step is not None else None,
        "best_loss": float(best_loss) if best_loss is not None else None,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    saved = _filter_config(ModelConfig, checkpoint["model_config"])
    saved.setdefault("pen_pos_bands", 0)  # absent key = checkpoint predates the feature
    model = PenTransformer(ModelConfig(**saved))
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    return model, checkpoint


def load_for_sampling(path, device="cpu", n_examples=200):
    model, checkpoint = load_checkpoint(path, device)
    cfg = DataConfig(**_filter_config(DataConfig, checkpoint["data_config"]))
    cfg.train_size = cfg.test_size = n_examples
    _, dataset, stroke_tok, char_tok = create_datasets(cfg, merges=checkpoint["merges"])
    assert char_tok.alphabet == checkpoint["alphabet"], \
        "dataset alphabet does not match the checkpoint; use the training dataset"
    attach_pen_deltas(model, stroke_tok)
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
    idx = torch.full((1, 1), st.BOS, dtype=torch.long, device=device)  # nothing from the training set seeds the sample
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
        points[:, 1] += y  # words carry their own height, so layout only advances and wraps
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
        for i in range(0, len(prompt_words), params.n_at_a_time):  # neighbours act as context
            words += for_chunk(prompt_words[i:i + params.n_at_a_time])
    else:
        for i in redo or []:
            if i < len(prompt_words):
                words[i] = for_chunk([prompt_words[i]])[0]
    return words


def progress_prompts(dataset, n=8):
    seen, out = set(), []
    for i in range(min(len(dataset), 2000)):
        text = dataset.text_for(i)  # prompts from the corpus itself are always in vocabulary
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= n:
            break
    if not out:
        raise ValueError(f"{dataset.name or 'dataset'} yielded no text to prompt "
                         "with; progress images need labelled examples")
    return out


def _cached_prompts(dataset):
    if not hasattr(dataset, "_progress_prompts"):
        dataset._progress_prompts = progress_prompts(dataset)
    return dataset._progress_prompts


def save_progress(model, dataset, out_dir, step, prompts=None, temperature=1.0,
                  rows=4):
    prompts = prompts or _cached_prompts(dataset)  # fixed prompts and seed: consecutive images differ only by the model
    os.makedirs(out_dir, exist_ok=True)
    params = SampleParams(temperature=temperature,
                          max_tokens=dataset.cfg.max_seq_length - 1)
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    contexts = torch.stack([torch.from_numpy(dataset.encode_text(text))
                            for text in prompts for _ in range(rows)]).to(device)
    torch.manual_seed(3)
    idx = torch.full((len(contexts), 1), st.BOS, dtype=torch.long, device=device)
    out = model.generate(idx, contexts, max_new_tokens=params.max_tokens,
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END,
                         pad_token=st.PAD)

    fig, axes = plt.subplots(rows, len(prompts),
                             figsize=(1.9 * len(prompts), 1.9 * rows),
                             squeeze=False)
    for col in range(len(prompts)):
        for row in range(rows):
            words = st.decode(out[col * rows + row].cpu().numpy()[1:])
            plot_words(words, params, ax=axes[row][col], color="k")
    fig.suptitle(f"step {step:,}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    top = max(ax.get_position().y1 for ax in axes[0])
    for col, text in enumerate(prompts):
        box = axes[0][col].get_position()
        fig.text(box.x0 + box.width / 2, top + 0.012,
                 "\n".join(textwrap.wrap(text, width=24)),
                 ha="center", va="bottom", fontsize=8)
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def save_samples(model, dataset, out_dir=".", num=3, do_sample=True):
    os.makedirs(out_dir, exist_ok=True)
    params = SampleParams(do_sample=do_sample,
                          max_tokens=dataset.cfg.max_seq_length - 1)
    paths = []
    for i in range(num):
        text = dataset.text_for(i)
        words = generate(model, dataset, text, params)
        fig, _, _ = plot_words(words, params, title=f'{dataset.name} {i}: "{text}"')
        path = os.path.join(out_dir,
                            f"{dataset.name}_{'sample' if do_sample else 'greedy'}_{i}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


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
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    losses = []
    for i, (X, C, Y) in enumerate(loader):
        _, loss = model(X.to(device), C.to(device), Y.to(device))
        losses.append(loss.item())
        if i + 1 >= max_batches:
            break
    model.train()
    return sum(losses) / len(losses)


def train(data_cfg, model_cfg, train_cfg):
    device = resolve_device(train_cfg.device)
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_cfg.out_dir, "best.pt")

    ckpt = None
    if train_cfg.resume:
        _, ckpt = load_checkpoint(train_cfg.resume, device)
        resumed_cfg = DataConfig(**_filter_config(DataConfig, ckpt["data_config"]))
        if asdict(resumed_cfg) != asdict(data_cfg):
            changed = {k: (v, getattr(data_cfg, k))
                       for k, v in asdict(resumed_cfg).items()
                       if getattr(data_cfg, k) != v}
            print(f"Using the checkpoint's data config; ignoring {changed} "
                  f"(checkpoint value, command-line value)")
        data_cfg = resumed_cfg  # merges and alphabet define token ids; the checkpoint's config wins

    torch.manual_seed(data_cfg.seed)
    train_dataset, test_dataset, stroke_tok, char_tok = create_datasets(
        data_cfg, merges=ckpt["merges"] if ckpt else None)
    if ckpt:
        assert char_tok.alphabet == ckpt["alphabet"], \
            "dataset alphabet does not match the checkpoint; resume needs its dataset"
    model_cfg.vocab_size = stroke_tok.vocab_size
    model_cfg.block_size = data_cfg.max_seq_length
    model_cfg.context_vocab_size = char_tok.vocab_size
    model_cfg.context_block_size = data_cfg.max_text_length

    model = PenTransformer(model_cfg).to(device)
    attach_pen_deltas(model, stroke_tok)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate,
                                  weight_decay=train_cfg.weight_decay,
                                  betas=(0.9, 0.99), eps=1e-8)

    def lr_at(step):
        if step < train_cfg.warmup:
            return (step + 1) / train_cfg.warmup
        t = (step - train_cfg.warmup) / max(1, train_cfg.steps - train_cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    step, best_loss = 0, None
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best_loss = ckpt["step"], ckpt["best_loss"]
        print(f"Resumed from {train_cfg.resume} at step {step} (best loss {best_loss})")

    run = None
    if train_cfg.wandb:
        import wandb
        run = wandb.init(project=train_cfg.wandb_project,
                         entity=train_cfg.wandb_entity or None,
                         name=train_cfg.wandb_run_name or None,
                         config={**asdict(data_cfg), **asdict(model_cfg), **asdict(train_cfg)})

    batches = infinite_batches(train_dataset, batch_size=train_cfg.batch_size,
                               pin_memory=(device == "cuda"),
                               num_workers=train_cfg.num_workers)

    while step < train_cfg.steps:
        t0 = time.time()
        X, C, Y = [t.to(device) for t in next(batches)]
        _, loss = model(X, C, Y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if run:
            run.log({"train_loss_step": loss.item(), "step": step})
        if step % train_cfg.print_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | {(time.time()-t0)*1000:.0f} ms/step"
                  f" | lr {scheduler.get_last_lr()[0]:.6f}")
        if step % train_cfg.eval_every != 0:
            continue

        train_loss = evaluate(model, train_dataset, device)
        test_loss = evaluate(model, test_dataset, device)
        print(f"step {step} | train loss {train_loss:.4f} | test loss {test_loss:.4f}")
        if run:
            run.log({"train_loss": train_loss, "test_loss": test_loss, "step": step})

        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss
            print(f"New best test loss; saving checkpoint to {checkpoint_path}")
            save_checkpoint(checkpoint_path, model, char_tok.alphabet, asdict(data_cfg),
                            stroke_tok.merges, optimizer, scheduler, step, best_loss)
            if run:
                artifact = wandb.Artifact("best_checkpoint", type="model")
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact)

        save_checkpoint(os.path.join(train_cfg.out_dir, "last.pt"), model,  # resume wants where the run stopped, not best
                        char_tok.alphabet, asdict(data_cfg), stroke_tok.merges,
                        optimizer, scheduler, step, best_loss)

        paths = [save_progress(model, test_dataset,
                               os.path.join(train_cfg.out_dir, "progress"), step)]
        if step % (train_cfg.eval_every * 4) == 0:
            paths += save_samples(model, test_dataset,
                                  os.path.join(train_cfg.out_dir, "samples"),
                                  num=3, do_sample=True)
        print(f"  wrote {paths[0]}")
        if run:
            run.log({os.path.basename(p): wandb.Image(p) for p in paths})

    if run:
        run.finish()


def sample_command(args):
    device = resolve_device(args.device)
    model, dataset, cfg, _ = load_for_sampling(args.checkpoint, device, n_examples=100)
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


RANK_LABELS = ("cat,apple,car,fish,tree,house,star,umbrella,clock,ladder,"
               "banana,bicycle")


def rank_command(args):
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    device = resolve_device(args.device)
    model, dataset, cfg, ckpt = load_for_sampling(args.checkpoint, device,
                                                  n_examples=4000)

    found = collections.defaultdict(list)
    for i in range(len(dataset)):
        text = dataset.text_for(i)
        if text in labels and len(found[text]) < args.per_label:
            found[text].append(i)
        if len(found) >= len(labels) and all(
                len(v) >= args.per_label for v in found.values()):
            break
    if not found:
        raise SystemExit("none of those labels appear in the dataset")

    ranks, spreads, by_label = [], [], {}
    for true, indices in sorted(found.items()):
        for i in indices:
            x, _, y = dataset[i]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            scored = []
            for label in labels:  # score one real drawing under every candidate label
                context = torch.from_numpy(
                    dataset.encode_text(label)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    _, loss = model(x, context, y)
                scored.append((label, loss.item()))
            scored.sort(key=lambda r: r[1])
            ranks.append([l for l, _ in scored].index(true) + 1)
            spreads.append((scored[-1][1] - scored[0][1]) / scored[0][1] * 100)
            by_label.setdefault(true, []).append(ranks[-1])

    for true, rs in sorted(by_label.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {true:12s} mean rank {np.mean(rs):4.1f}/{len(labels)}"
              f"   over {len(rs)} drawings")
    chance = (len(labels) + 1) / 2
    print(f"\nstep {ckpt['step']}: mean rank {np.mean(ranks):.1f}/{len(labels)}"
          f" (chance {chance:.1f}), mean spread {np.mean(spreads):.1f}%")


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

    rank_p = sub.add_parser("rank", help="measure prompt conditioning: rank the "
                                         "true label among candidates")
    rank_p.add_argument("--checkpoint", type=str, default="out/quickdraw/best.pt")
    rank_p.add_argument("--labels", type=str, default=RANK_LABELS)
    rank_p.add_argument("--per_label", type=int, default=8,
                        help="drawings per label; per-label ranks are bimodal, one is not enough")
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
