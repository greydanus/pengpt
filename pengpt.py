import argparse
import glob
import json
import os
import queue
import sys
import threading
import time
import zlib
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from strokes import Strokes, curves, grid, held_out, plot, stem, to_points, to_strokes


@dataclass
class DataConfig:
    dataset: str = "data/bigbank.hand.parquet"
    test: str = ""
    tokens: str = ""
    vocab: str = ""
    holdout: int = 20
    max_train: int = 0
    max_words: int = 4
    max_seq_length: int = 256
    max_text_length: int = 50
    text_end: bool = True
    grid: float = 0.02
    n_merges: int = 512
    tokenizer: str = "scribe"
    augment: float = 1.0
    rotate: float = 2.0
    stretch: float = 0.10
    shear: float = 0.10
    perspective: float = 0.0
    warp: float = 0.0
    scale: float = 0.0
    shift: float = 0.0
    stroke_swap: float = 0.0
    stroke_reverse: float = 0.0
    loop_start: float = 0.0
    min_strokes: int = 4
    label_drop: float = 0.0
    hint: float = 0.0
    vectors: str = ""
    vec_columns: str = ""
    vec_two: float = 0.4
    vec_pair: float = 0.3
    vec_kind: float = 0.85
    seed: int = 1337


@dataclass
class ModelConfig:
    arch: str = "pen"
    window: int = 10
    n_layer: int = 5
    n_head: int = 2
    n_embd: int = 64
    pen_pos_bands: int = 8
    pen_pos_jitter: int = 32
    pen_rotary: bool = False
    source_dropout: float = 0.1
    dropout: float = 0.0
    n_sources: int = 0
    context_dim: int = 0
    context_slots: int = 0
    context_width: int = 0
    context_kinds: int = 0
    vocab_size: int = -1
    block_size: int = -1
    context_vocab_size: int = -1
    context_block_size: int = -1


@dataclass
class TrainConfig:
    steps: int = 12_000
    stop: int = 0
    batch_size: int = 32
    batch_tokens: int = 0
    pad: int = 64
    learning_rate: float = 6e-3
    embedding_lr: float = 6e-3
    matrix_lr: float = 0.04
    weight_decay: float = 1e-4
    warmup: int = 200
    warmdown: float = 0.8
    grad_clip: float = 1.0
    eval_every: int = 1_000
    sample_every: int = 5_000
    showcase: str = ""
    showcase_source: str = ""
    print_every: int = 100
    loader_workers: int = 6
    gpus: int = 1
    device: str = "auto"
    compile: bool = True
    graphs: bool = False
    muon_half: bool = True
    mixed: bool = True
    keep: bool = False
    train_seed: int = 0
    out_dir: str = "out/default"
    resume: str = ""
    wandb: bool = False
    wandb_project: str = "pengpt"
    wandb_entity: str = ""
    wandb_run_name: str = ""


CLASSES = (DataConfig, ModelConfig, TrainConfig)
DERIVED_FIELDS = {"vocab_size", "block_size", "context_vocab_size", "context_block_size", "n_sources", "context_dim", "context_slots",
                  "context_width", "context_kinds"}
X_HEIGHT, INK_HEIGHT, SPACING, SHORT_LETTERS = 0.123, 0.22, 0.01, set("acemnorsuvwxz")


def kind(cfg):
    return "sketch" if all(p.endswith(".sketch.parquet") for p, _ in dataset_specs(cfg.dataset)) else "hand"


def add_config_args(parser, classes=None):
    for cls in classes or CLASSES:
        for f in fields(cls):
            if f.name in DERIVED_FIELDS:
                continue
            if isinstance(f.default, bool):
                parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction,
                                    default=f.default)
            else:
                parser.add_argument(f"--{f.name}", type=type(f.default),
                                    default=f.default)


def configs_from_args(args, classes=None):
    args = vars(args)
    build = lambda cls: cls(**{f.name: args[f.name] for f in fields(cls) if f.name in args})
    return tuple(build(cls) for cls in classes or CLASSES)


DIRECTIONS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
DOWN, UP, N_BASE = 8, 9, 10
IGNORE_INDEX = -1
KINDS = ("image", "label", "name", "frame", "steps", "kind", "phrase", "short", "dense", "ask", "tags", "vague", "pose", "verdict",
         "manner", "parts", "sentence", "outline", "terse")

_DIR_LOOKUP = np.full((3, 3), -1, dtype=np.int64)
for _i, (_dx, _dy) in enumerate(DIRECTIONS):
    _DIR_LOOKUP[_dx + 1, _dy + 1] = _i


def _stroke_spans(pen):
    down = (np.asarray(pen) == 1).astype(np.int8)
    if not down.any():
        return []
    edges = np.diff(np.concatenate([[0], down, [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _smooth_strokes(points, window=3):
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

    def __init__(self, grid=DataConfig.grid, merges=None):
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

    def moves(self, points):
        return len(self.encode_base(points))

    def encode_word(self, points):
        return self.apply_merges(self.encode_base(points))

    def encode_many(self, words):
        base = self.encode_bases(words)
        if not self._pairs or not base:
            return base
        flat = np.concatenate([np.concatenate([w, [self.PAD]]) for w in base])
        merged = self.apply_merges(flat)
        return [w[:-1] for w in np.split(merged, np.flatnonzero(merged == self.PAD) + 1)[:-1]]

    def encode_base(self, points):
        return self.encode_bases([points])[0]

    def encode_bases(self, words):
        words = [np.asarray(w, dtype=float).reshape(-1, 3) for w in words]
        points, word = np.concatenate(words), np.repeat(np.arange(len(words)), [len(w) for w in words])
        down = points[:, 2] == 1
        if not down.any():
            return [np.zeros(0, dtype=np.int64) for _ in words]
        starts_word = np.concatenate([[True], word[1:] != word[:-1]])
        pen_was_up = ~np.concatenate([[False], down[:-1]])
        hop = (down & (starts_word | pen_was_up))[down]
        cell, word = np.rint(points[down, :2] / self.grid).astype(np.int64), word[down]
        first = np.concatenate([[True], word[1:] != word[:-1]])
        d = cell - np.where(first[:, None], 0, np.roll(cell, 1, 0))
        last = np.concatenate([hop[1:], [True]])
        n = np.abs(d).max(1)
        size = n + hop + last
        off = np.cumsum(size) - size
        out = np.empty(size.sum(), dtype=np.int64)
        seg = np.repeat(np.arange(len(d)), n)
        i = np.arange(len(seg)) - (np.cumsum(n) - n)[seg]
        long, short = n[seg], np.abs(d).min(1)[seg]
        minor = lambda k: -((long - 2 * k * short) // (2 * long))
        along = np.abs(d[seg]) == long[:, None]
        unit = np.sign(d[seg]) * np.where(along, 1, (minor(i + 1) - minor(i))[:, None])
        out[off[seg] + i] = _DIR_LOOKUP[tuple((unit + 1).T)]
        out[off[hop] + n[hop]] = DOWN
        out[off[last] + n[last] + hop[last]] = UP
        per_word = np.bincount(word, weights=size, minlength=len(words)).astype(np.int64)
        return np.split(out, np.cumsum(per_word)[:-1])

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
        return curves(_smooth_strokes(out), self.grid) if smooth else out

    def expand(self, tokens):
        out = list(np.asarray(tokens).ravel())
        while any(t in self._inverse for t in out):
            nxt = []
            for t in out:
                nxt.extend(self._inverse[t]) if t in self._inverse else nxt.append(t)
            out = nxt
        return out


def learn_merges(token_sequences, n_merges=512, min_count=20):
    s = np.concatenate([np.r_[t, UP] for t in token_sequences]).astype(np.int64)
    vocab, merges = N_BASE + n_merges, []
    for merged in range(N_BASE, vocab):
        a, b = s[:-1], s[1:]
        ok = (a != UP) & (a != DOWN) & (b != UP) & (b != DOWN)
        pairs, counts = np.unique(a[ok] * vocab + b[ok], return_counts=True)
        if not len(counts) or counts.max() < min_count:
            break
        merges.append((*divmod(int(pairs[counts.argmax()]), vocab), merged))
        s = ScribeTokenizer(merges=merges[-1:]).apply_merges(s)
    return merges


ARROWS = {"→": (1, 0), "↗": (1, 1), "↑": (0, 1), "↖": (-1, 1), "←": (-1, 0), "↙": (-1, -1), "↓": (0, -1), "↘": (1, -1)}


def read_arrow_merges(path, n):
    ids = {arrow: int(_DIR_LOOKUP[step[0] + 1, step[1] + 1]) for arrow, step in ARROWS.items()}
    merges = []
    for line in open(path, encoding="utf-8"):
        pair = line.split()
        if len(merges) == n:
            break
        if len(pair) == 2 and pair[0] in ids and pair[1] in ids and pair[0] + pair[1] not in ids:
            ids[pair[0] + pair[1]] = N_BASE + len(merges)
            merges.append((ids[pair[0]], ids[pair[1]], N_BASE + len(merges)))
    return merges


class Tokink(ScribeTokenizer):

    def __init__(self, grid, path, n):
        super().__init__(grid, read_arrow_merges(path, n))
        key = np.array([a * self.vocab_size + b for a, b, _ in self.merges], dtype=np.int64)
        self._rank = np.argsort(key)
        self._key = key[self._rank]

    def encode_many(self, words):
        return [self.apply_merges(w) for w in self.encode_bases(words)]

    def apply_merges(self, tokens):
        out, none = np.asarray(tokens, dtype=np.int64).copy(), len(self._key)
        while len(out) > 1 and none:
            pair = out[:-1] * self.vocab_size + out[1:]
            at = np.searchsorted(self._key, pair).clip(max=none - 1)
            rank = np.where(self._key[at] == pair, self._rank[at], none)
            first = rank.min()
            if first == none:
                break
            matches = np.flatnonzero(rank == first)
            if matches.size > 1 and (np.diff(matches) == 1).any():
                starts = np.r_[True, np.diff(matches) != 1]
                run_start = np.flatnonzero(starts)[np.cumsum(starts) - 1]
                matches = matches[(np.arange(matches.size) - run_start) % 2 == 0]
            out[matches] = N_BASE + first
            keep = np.ones(len(out), dtype=bool)
            keep[matches + 1] = False
            out = out[keep]
        return out


def stroke_tokenizer(cfg, merges=None):
    if cfg.tokenizer == "polar":
        from polar import PolarTokenizer
        return PolarTokenizer(cfg.grid, max(1, round(cfg.grid / SPACING)), ScribeTokenizer().moves)
    if cfg.tokenizer == "tokink":
        return Tokink(cfg.grid, cfg.tokens, cfg.n_merges)
    return ScribeTokenizer(grid=cfg.grid, merges=merges)


class ByteTokenizer:

    PAD, END = 0, 257

    def __init__(self, end=True):
        self.end = end
        self.vocab_size = 258 if end else 257

    def encode(self, text, length):
        ids = [b + 1 for b in text.encode()][:length - self.end] + [self.END] * self.end
        return np.array(ids + [self.PAD] * (length - len(ids)), dtype=np.int64)

    def decode(self, ids):
        ids = [int(i) for i in ids if int(i) != self.END]
        ids = ids[:ids.index(self.PAD)] if self.PAD in ids else ids
        return bytes(i - 1 for i in ids).decode(errors="replace")


def densify(points, step):
    xy, pen = points[:, :2], points[:, 2]
    d = np.diff(xy, axis=0)
    n = np.where(pen[:-1] == 1, np.maximum(1, np.ceil(np.hypot(*d.T) / step)), 1).astype(int)
    seg = np.repeat(np.arange(len(n)), n)
    t = (np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n)) / n[seg]
    return np.vstack([np.c_[xy[seg] + d[seg] * t[:, None], pen[seg]], points[-1:]])


def alter(points, cfg, rng):
    points = np.asarray(points, float)
    strokes = [points[a:b, :2] for a, b in _stroke_spans(points[:, 2])]
    if len(strokes) >= cfg.min_strokes and cfg.stroke_swap > 0:
        longest, i = int(np.argmax([np.hypot(*np.diff(s, axis=0).T).sum() for s in strokes])), 0
        while i < len(strokes) - 1:
            if longest not in (i, i + 1) and rng.random() < cfg.stroke_swap:
                strokes[i], strokes[i + 1], i = strokes[i + 1], strokes[i], i + 1
            i += 1
    for k, s in enumerate(strokes):
        if cfg.loop_start > 0 and len(s) > 3 and np.hypot(*(s[0] - s[-1])) < 0.03 and rng.random() < cfg.loop_start:
            j = int(rng.integers(1, len(s) - 1))
            s = np.vstack([s[j:-1], s[:j + 1]])
        if cfg.stroke_reverse > 0 and rng.random() < cfg.stroke_reverse:
            s = s[::-1]
        strokes[k] = s
    return to_points(strokes).astype(float)


def bend(cfg, rng):
    k = cfg.augment
    u = lambda a, size=None: rng.uniform(-k * a, k * a, size)
    wave = rng.normal(size=(3, 2))
    wave *= rng.uniform(0.4, 1.0, (3, 1)) / np.hypot(*wave.T)[:, None]
    phase, push, tilt = rng.uniform(0, 2 * np.pi, 3), rng.normal(size=(3, 2)) * k * cfg.warp / 3 ** 0.5, u(cfg.perspective, 2)
    angle, shear = np.deg2rad(u(cfg.rotate)), u(cfg.shear)
    c, s = np.cos(angle), np.sin(angle)
    turn, size = np.array([[1, 0], [shear, 1]]) @ np.array([[c, s], [-s, c]]), np.exp(u(cfg.stretch, 2) + u(cfg.scale))
    return wave, phase, push, tilt, turn, size, u(cfg.shift, 2), rng.uniform(-0.5, 0.5, 2) * cfg.grid


def augment(points, cfg, rng, centre=True, how=None):
    wave, phase, push, tilt, turn, size, shift, nudge = how or bend(cfg, rng)
    curved = cfg.warp > 0 or cfg.perspective > 0
    out = densify(np.asarray(points, float), 2 * cfg.grid) if curved else np.array(points, float)
    mid = lambda p: (p.min(0) + p.max(0)) / 2 if centre else 0
    p = out[:, :2] - mid(out[:, :2])
    if curved:
        p = p + np.sin(2 * np.pi * p @ wave.T + phase) @ push
        p = p / (1 + p @ tilt)[:, None]
    p = p @ turn * size
    out[:, :2] = p - mid(p) + shift + nudge
    return out


def centred(strokes):
    points = to_points(strokes)
    points[:, :2] -= (points[:, :2].min(0) + points[:, :2].max(0)) / 2
    return points


def measure(points):
    points = np.asarray(points, float)
    spans = _stroke_spans(points[:, 2])
    size = max(np.ptp(points[:, :2], 0).max(), 1e-6)
    return sum(np.hypot(*np.diff(points[a:b, :2], axis=0).T).sum() for a, b in spans) / size, len(spans)


def hint_digits(ink=None, strokes=None):
    ink = None if ink is None else int(np.clip(round(2 * ink), 0, 99))
    strokes = None if strokes is None else int(np.clip(strokes, 0, 99))
    return [d for n in (ink, strokes) for d in ((None, None) if n is None else divmod(n, 10))]


class Bank:

    def __init__(self, strokes, centre):
        self.strokes, self.centre, self.held = strokes, centre, {}

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, i):
        if self.centre:
            return centred(self.strokes.strokes(int(i)))
        if i not in self.held:
            self.held[i] = to_points(self.strokes.strokes(int(i)))
        return self.held[i]


class Labels:

    def __init__(self, cfg):
        self.cfg, self.tok = cfg, ByteTokenizer(cfg.text_end)
        self.tok.vocab_size += 40 * (cfg.hint > 0)

    def of(self, i, text, rng=None):
        return text

    def encode(self, told, ink=None, strokes=None):
        cfg = self.cfg
        length = cfg.max_text_length + cfg.text_end
        if cfg.hint <= 0:
            return self.tok.encode(told, length)
        hints = [0 if d is None else 258 + 10 * k + d for k, d in enumerate(hint_digits(ink, strokes))]
        return np.r_[self.tok.encode(told, length - 4), hints].astype(np.int64)

    def model_fields(self):
        return dict(context_vocab_size=self.tok.vocab_size, context_block_size=self.cfg.max_text_length + self.cfg.text_end)


class Vectors:
    encoder = None

    def __init__(self, cfg, dim, kinds=0, columns=None):
        self.cfg, self.dim, self.kinds, self.columns, self.typed, self.width = cfg, dim, kinds, columns or {}, {}, dim + kinds

    def worded(self, text):
        if text not in self.typed:
            from vectors import encoder
            Vectors.encoder = Vectors.encoder or encoder(self.cfg.vectors)
            self.typed[text] = Vectors.encoder.texts([text])[0].astype(np.float32)
        return self.typed[text]

    def of(self, i, text, rng=None):
        cfg, D = self.cfg, self.dim
        out, kinds = np.zeros((2, D), np.float32), np.zeros((2, self.kinds), np.float32)
        have = lambda c: self.columns[c].there(i)
        one = [c for c in self.columns if self.columns[c].shape[1:] == (D,) and have(c)]
        two = [c for c in self.columns if self.columns[c].shape[1:] == (2, D) and have(c)]
        if rng is None:
            picks = [next((c for c in ("frame", "label") if c in one), (one or ["image"])[0])]
        elif two and (not one or rng.random() < cfg.vec_pair):
            picks = [two[rng.integers(len(two))]]
        else:
            picks = rng.permutation(one)[:1 + (rng.random() < cfg.vec_two)]
        named = self.kinds and (rng is None or rng.random() < cfg.vec_kind)
        for k, c in enumerate(picks):
            v = self.columns[c][i]
            out[k:k + v.ndim] = v
            if named and c in KINDS[:self.kinds]:
                kinds[k:k + v.ndim, KINDS.index(c)] = 1
        kinds[~out.any(1)] = 0
        return np.r_[(out[0] + out[1]) / np.linalg.norm(out[0] + out[1]) if out[1].any() else out[0], kinds.max(0)]

    def encode(self, told, ink=None, strokes=None):
        if isinstance(told, str):
            vec = np.zeros(self.width, np.float32)
            if told:
                vec[:self.dim] = self.worded(told)
        else:
            vec = np.r_[told, np.zeros(max(0, self.width - len(told)), np.float32)]
        if self.cfg.hint > 0:
            flags = np.zeros(40, np.float32)
            flags[[10 * k + d for k, d in enumerate(hint_digits(ink, strokes)) if d is not None]] = 1
            vec = np.r_[vec, flags]
        return vec

    def model_fields(self):
        return dict(Labels(self.cfg).model_fields(), context_dim=self.width + 40 * (self.cfg.hint > 0), context_slots=1,
                    context_width=self.dim, context_kinds=self.kinds)


class PenDataset:
    ask = (None, None)

    def __init__(self, bank_points, bank_texts, indices, stroke_tok, told, cfg, augment=True, seed=0, source_ids=None, weights=None,
                 sources=("",)):
        self.bank_points, self.bank_texts, self.indices = bank_points, bank_texts, np.asarray(indices, int)
        self.source_ids = np.zeros(len(bank_texts), int) if source_ids is None else np.asarray(source_ids)
        self.sources = list(sources)
        w = np.ones(len(bank_texts)) if weights is None else np.asarray(weights, float)
        ids = self.source_ids[self.indices]
        self.groups = [self.indices[ids == g] for g in np.unique(ids)]
        p = np.array([w[g].sum() for g in self.groups])
        self.group_p = p / p.sum()
        self.stroke_tok, self.told, self.cfg, self.augment, self.seed = stroke_tok, told, cfg, augment, seed
        self.hand = kind(cfg) == "hand"

    def encode_text(self, told, ink=None, strokes=None):
        return self.told.encode(told, *(self.ask if ink is None and strokes is None else (ink, strokes)))

    def draw(self, rng):
        cfg, group = self.cfg, self.groups[rng.choice(len(self.groups), p=self.group_p)]
        n = min(cfg.max_words, len(group))
        limit = rng.integers(1, n + 1) if n > 1 else 1
        chosen = rng.choice(group, size=n, replace=False)[:limit] if n > 1 else [rng.choice(group)]
        words = [self.bank_points[i] for i in chosen]
        if self.augment:
            altering, how = cfg.stroke_swap or cfg.stroke_reverse or cfg.loop_start, bend(cfg, rng) if self.hand else None
            words = [augment(alter(w, cfg, rng) if altering else w, cfg, rng, not self.hand, how) for w in words]
        return chosen, words, rng

    def join(self, chosen, tokens, words, rng):
        cfg, kept, parts, total = self.cfg, [], [], 0
        text = lambda rows: " ".join(self.bank_texts[j] for j in rows)
        for i, t in zip(chosen, tokens):
            extra = len(t) + (2 if parts else 0)
            if parts and (total + extra > cfg.max_seq_length - 2 or len(text(kept + [i]).encode()) > cfg.max_text_length):
                break
            parts += [np.array([self.stroke_tok.WORD] * 2, dtype=np.int64)] * bool(parts) + [t]
            total += extra
            kept.append(i)
        told = text(kept)
        if not self.augment:
            told = self.told.of(kept[0], told)
        elif rng.random() < cfg.label_drop or not told:
            told = ""
        else:
            told = self.told.of(kept[0], told, rng)
        hints = []
        if self.augment and cfg.hint > 0:
            hints = [v if rng.random() < cfg.hint else None for v in measure(np.concatenate(words[:len(kept)]))]
        return np.concatenate(parts), self.encode_text(told, *hints), int(self.source_ids[kept[0]])

    def make_item(self, tokens, context, source=0):
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
        return x, torch.from_numpy(context), y, torch.tensor(source)

    def batch(self, indices):
        draws = [self.draw(np.random.default_rng([self.seed, int(i)])) for i in indices]
        merged = iter(self.stroke_tok.encode_many([w for _, words, _ in draws for w in words]))
        items = [self.make_item(*self.join(chosen, [next(merged) for _ in words], words, rng)) for chosen, words, rng in draws]
        return tuple(torch.stack(t) for t in zip(*items))


def dataset_specs(spec):
    out = []
    for part in spec.split(","):
        path, _, weight = part.strip().partition(":")
        out.append((path, float(weight or 1)))
    return out


def name_of(path):
    return os.path.basename(stem(path))


def learn_tokens(cfg, each=400, fresh=False):
    if cfg.tokens and os.path.exists(cfg.tokens) and not fresh:
        saved = json.load(open(cfg.tokens))
        if saved["grid"] != cfg.grid or saved["n_merges"] != cfg.n_merges:
            raise SystemExit(f"{cfg.tokens} was learned at grid {saved['grid']} with {saved['n_merges']} merges; "
                             f"pass --tokens with another path to learn a vocabulary for grid {cfg.grid} with {cfg.n_merges}")
        return saved["merges"]
    paths = sorted(glob.glob(cfg.vocab)) or [p for p, _ in dataset_specs(cfg.dataset)]
    rng, base, corpus = np.random.default_rng(cfg.seed), ScribeTokenizer(grid=cfg.grid), []
    for path in paths:
        bank = Bank(Strokes(path), True)
        rows = rng.choice(len(bank), min(each, len(bank)), replace=False)
        corpus += base.encode_bases([augment(bank[i], cfg, rng) for i in rows])
    merges, names = learn_merges(corpus, cfg.n_merges), [name_of(p) for p in paths]
    if cfg.tokens:
        with open(cfg.tokens, "w") as f:
            json.dump({"grid": cfg.grid, "n_merges": cfg.n_merges, "sources": names, "merges": merges}, f)
    print(f"Learned {len(merges)} merges from up to {each} drawings of each of {', '.join(names)}"
          + f"; saved to {cfg.tokens}" * bool(cfg.tokens))
    return merges


def need_vectors(cfg):
    from vectors import have
    missing = [p for p, _ in dataset_specs(cfg.dataset) if not have(p, cfg.vectors)]
    if missing:
        raise SystemExit(f"no {cfg.vectors} vectors for {', '.join(missing)}: "
                         f"each drawing file needs its .vectors.parquet beside it")


def told_by(cfg):
    if not cfg.vectors:
        return Labels(cfg)
    from vectors import columns, together
    need_vectors(cfg)
    parts = [columns(p, cfg.vectors) for p, _ in dataset_specs(cfg.dataset)]
    shared = {c: together([p[c] for p in parts]) for c in parts[0] if all(c in p for p in parts)}
    if cfg.vec_columns:
        shared = {c: v for c, v in shared.items() if c in cfg.vec_columns.split(",")}
    if "image" not in shared:
        raise SystemExit(f"a vector table of {cfg.dataset} has no picture vectors")
    return Vectors(cfg, shared["image"].shape[-1], len(KINDS) * (cfg.vec_kind > 0), shared)


def sketch_datasets(cfg, merges=None, sources=None):
    specs = dataset_specs(cfg.dataset)
    data = Strokes(*[p for p, _ in specs])
    names, sizes = [name_of(p) for p, _ in specs], [len(f) for f in data.files]
    texts = [f.tables["label"][j] for f in data.files for j in f.cols["label"]]
    held = np.concatenate([held_out(f, cfg.holdout) for f in data.files])
    source = np.repeat(np.arange(len(names)), sizes)
    sources = sources or ["", *names]
    source_ids = np.array([sources.index(n) if n in sources else 0 for n in names])[source]
    weights = np.repeat([w for _, w in specs], sizes)
    stroke_tok, told = ScribeTokenizer(grid=cfg.grid, merges=merges or learn_tokens(cfg)), told_by(cfg)

    def build(ix, seed, augmented):
        return PenDataset(Bank(data, True), texts, ix, stroke_tok, told, cfg, augment=augmented, seed=seed,
                          source_ids=source_ids, weights=weights, sources=sources)

    rng = np.random.default_rng(cfg.seed)
    parts = {n: rng.permutation(np.flatnonzero(held & (source == k)))[:1000] for k, n in enumerate(names)}
    seen = {n: rng.permutation(np.flatnonzero(~held & (source == k)))[:300] for k, n in enumerate(names)}
    kept = np.flatnonzero(~held)
    kept = np.sort(np.random.default_rng(cfg.seed + 2).permutation(kept)[:cfg.max_train]) if cfg.max_train else kept
    train, test = build(kept, cfg.seed, cfg.augment > 0), build(np.concatenate(list(parts.values())), cfg.seed + 1, False)
    test.parts, test.unseen = {n: build(ix, cfg.seed + 1, False) for n, ix in parts.items()}, {}
    test.seen = {n: build(ix, cfg.seed + 1, False) for n, ix in seen.items()}
    print(f"{len(kept):,} train / {held.sum():,} held-out drawings from {', '.join(names)}; stroke vocab {stroke_tok.vocab_size}")
    return train, test, stroke_tok


def check(texts, points, grid, name="", n=500, seed=0):
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(texts), min(n, len(texts)), replace=False)
    short = [i for i, t in enumerate(texts) if len(t) >= 2 and set(t) <= SHORT_LETTERS]
    short = [short[i] for i in rng.choice(len(short), min(2000, len(short)), replace=False)] if short else []
    downs = [d[d[:, 2] == 1] for d in (np.asarray(points(i), float) for i in short) if (d[:, 2] == 1).sum() >= 3]
    heights = [np.quantile(d[:, 1], 0.98) - np.quantile(d[:, 1], 0.02) for d in downs]
    bottoms = [np.quantile(d[:, 1], 0.98) for d in downs]
    steps, tokens, chars, sizes, base = [], 0, 0, [], ScribeTokenizer(grid=grid)
    for i in sample:
        p = np.asarray(points(i), float)
        if (p[:, 2] == 1).sum() < 3:
            continue
        same = (p[1:, 2] == 1) & (p[:-1, 2] == 1)
        steps.append(np.hypot(*np.diff(p[:, :2], axis=0)[same].T))
        sizes.append(np.ptp(p[p[:, 2] == 1, 1]))
        tokens, chars = tokens + len(base.encode_base(p)), chars + len(texts[i])
    steps = np.concatenate(steps) if steps else np.zeros(1)
    median = lambda values: float(np.median(values)) if len(values) >= 5 else None
    report = {"x_height": median(heights), "baseline": median(bottoms), "spacing": float(np.median(steps)),
              "long_steps": float(np.mean(steps > grid)), "tokens_per_char": tokens / max(chars, 1)}
    x = f"{report['x_height']:.3f}" if report["x_height"] is not None else "n/a"
    b = f"{report['baseline']:+.3f}" if report["baseline"] is not None else "n/a"
    print(f"{name}: x-height {x} (target {X_HEIGHT}), baseline {b}, point spacing {report['spacing']:.4f} "
          f"(target {SPACING}), {100 * report['long_steps']:.0f}% steps > grid, {report['tokens_per_char']:.0f} tokens/char")
    errors = []
    if report["x_height"] is not None and not 0.6 < report["x_height"] / X_HEIGHT < 1.7:
        errors.append(f"x-height {x} is {report['x_height'] / X_HEIGHT:.2f}x the target {X_HEIGHT}; rescale the ink")
    if report["baseline"] is not None and abs(report["baseline"]) > 0.06:
        errors.append(f"baseline sits at {b}; words should rest on y = 0")
    if report["x_height"] is None and sizes and not 0.5 < float(np.median(sizes)) / INK_HEIGHT < 2:
        errors.append(f"ink is {float(np.median(sizes)) / INK_HEIGHT:.1f}x the usual size; rescale it")
    if report["long_steps"] > 0.05:
        print(f"  WARNING ({name}): {100 * report['long_steps']:.0f}% of in-stroke steps skip a grid cell; "
              f"resampling to {SPACING} spacing would match the other datasets")
    for err in errors:
        print(f"  ERROR ({name}): {err}")
    return report, errors


def words_of(f, cfg):
    texts, writers = ([f.tables[k][j] for j in f.cols[k]] for k in ("label", "writer"))
    good = np.array([len(t.encode()) <= cfg.max_text_length and len(t.split()) == 1 for t in texts])
    if not good.all():
        print(f"Dropping {(~good).sum()} words from {f.path} whose text holds a space or exceeds "
              f"max_text_length {cfg.max_text_length}")
    return texts, writers, good


def hand_datasets(cfg, merges=None, sources=None):
    specs, fixed = dataset_specs(cfg.dataset), dataset_specs(cfg.test) if cfg.test else []
    data = Strokes(*[p for p, _ in specs + fixed])
    bank, start = Bank(data, False), np.r_[0, data.end]
    texts, writers, weights, train_ix, test_ix, parts, unseen = [], [], [], [], [], {}, {}
    for k, ((path, weight), f, offset) in enumerate(zip(specs + fixed, data.files, start)):
        t, w, good = words_of(f, cfg)
        rows, name = np.flatnonzero(good), name_of(path)
        if cfg.holdout and not fixed and len({w[i] for i in rows}) >= cfg.holdout:
            new = np.array([zlib.crc32(w[i].encode()) % cfg.holdout == 0 for i in rows])
            unseen[name] = offset + rows[new][np.random.default_rng(cfg.seed).permutation(new.sum())[:500]]
            rows = rows[~new]
        if check([t[i] for i in rows], lambda i: bank[offset + rows[i]], cfg.grid, os.path.basename(path))[1]:
            raise SystemExit(f"{path} is not in the shared coordinate system; see the errors above")
        kept = np.zeros(len(t), bool)
        kept[rows] = True
        texts, writers, weights = texts + t, writers + [x if k else None for k, x in zip(kept, w)], weights + [weight] * len(t)
        perm = offset + rows[np.random.default_rng(cfg.seed).permutation(len(rows))]
        n_test = min(1000, len(rows) // 20) if not fixed else len(rows) * (k >= len(specs))
        train_ix += list(perm[:len(perm) - n_test][:cfg.max_train or None])
        test_ix += list(perm[len(perm) - n_test:])
        if n_test:
            parts[name] = perm[len(perm) - n_test:]
    if not test_ix:
        raise SystemExit("too few words to hold some out; need at least 20 in some dataset")
    sources = sources or [""] + sorted({w for w in writers if w})
    lookup = {w: i for i, w in enumerate(sources)}
    source_ids, told, rng = np.array([lookup.get(w, 0) for w in writers]), Labels(cfg), np.random.default_rng(cfg.seed)
    if cfg.tokenizer != "scribe":
        merges = []
    if merges is None and cfg.tokens and os.path.exists(cfg.tokens):
        merges = learn_tokens(cfg)
    if merges is None:
        base, p = ScribeTokenizer(grid=cfg.grid), np.asarray(weights)[train_ix] / np.sum(np.asarray(weights)[train_ix])
        sample = rng.choice(train_ix, size=min(3000, len(train_ix)), replace=False, p=p)
        merges = learn_merges([base.encode_word(augment(bank[i], cfg, rng, False)) for i in sample], cfg.n_merges)
        print(f"Learned {len(merges)} BPE merges")
    stroke_tok = stroke_tokenizer(cfg, merges)

    def build(ix, seed, augmented, ids=source_ids):
        return PenDataset(bank, texts, ix, stroke_tok, told, cfg, augment=augmented, seed=seed, source_ids=ids,
                          weights=weights, sources=sources)

    train, test = build(train_ix, cfg.seed, cfg.augment > 0), build(test_ix, cfg.seed + 1, False)
    test.parts, test.seen = {name: build(ix, cfg.seed + 1, False) for name, ix in parts.items()}, {}
    test.unseen = {name: by_writer(build, ix, data) for name, ix in unseen.items() if len(ix)}
    print(f"Word bank: {len(train_ix)} train / {len(test_ix)} test words from {len(sources) - 1} writers; "
          f"{sum(len(d.groups) for d in test.unseen.values())} writers held out; stroke vocab {stroke_tok.vocab_size}")
    return train, test, stroke_tok


def by_writer(build, rows, data):
    ids = np.zeros(len(data), int)
    ids[rows] = np.unique([data.meta(i)["writer"] for i in rows], return_inverse=True)[1]
    return build(rows, 0, False, ids)


def create_datasets(cfg, merges=None, sources=None):
    return (hand_datasets if kind(cfg) == "hand" else sketch_datasets)(cfg, merges, sources)


def fit(model_cfg, dataset):
    model_cfg.n_sources, model_cfg.vocab_size = len(dataset.sources), dataset.stroke_tok.vocab_size
    model_cfg.block_size = dataset.cfg.max_seq_length
    for name, value in dataset.told.model_fields().items():
        setattr(model_cfg, name, value)
    return model_cfg


def _produce(dataset, batch_size, seed, q, pool=8, step=64, tokens=0):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    pad = dataset.stroke_tok.PAD
    held = None
    try:
        while True:
            new = [t.numpy() for t in dataset.batch(rng.integers(0, 2 ** 62, pool * batch_size))]
            X, C, Y, W = new if held is None else [np.concatenate(pair) for pair in zip(held, new)]
            length = (X != pad).sum(1)
            order = length.argsort()
            width = lambda longest: min(X.shape[1], -(-longest // step) * step)
            fits = lambda end: max(1, tokens // width(length[order[end - 1]]))
            groups, end = ([], len(order)) if tokens else (list(order.reshape(pool, batch_size)), 0)
            while end and fits(end) <= end:
                size = fits(end)
                groups.append(order[end - size:end])
                end -= size
            for k in rng.permutation(len(groups)):
                rows = groups[k]
                n = width(length[rows].max())
                q.put((X[rows, :n], C[rows], Y[rows, :n], W[rows]))
            held = [a[order[:end]] for a in (X, C, Y, W)] if end else None
    except Exception as e:
        q.put(e)


def infinite_batches(dataset, batch_size, seed=0, prefetch=4, workers=1, tokens=0, step=64):
    import multiprocessing as mp
    kwargs = {"tokens": tokens, "step": step}
    if workers > 1 and "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
        q = ctx.Queue(maxsize=prefetch * workers)
        for k in range(workers):
            ctx.Process(target=_produce, args=(dataset, batch_size, [seed, k], q), kwargs=kwargs, daemon=True).start()
    else:
        q = queue.Queue(maxsize=prefetch)
        threading.Thread(target=_produce, args=(dataset, batch_size, seed, q), kwargs=kwargs, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, Exception):
            raise item
        yield tuple(torch.from_numpy(a) for a in item)


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


def pen_rotary_tables(t, pen_xy, head_dim, base=10_000, shortest=4.0, longest=512.0):
    n = head_dim // 4
    m = (head_dim // 2 - n) // 2
    k = torch.arange(n, device=t.device).float()
    seq = torch.outer(t.float(), base ** -(k / n)).expand(pen_xy.size(0), -1, -1)
    waves = 2 * torch.pi / (shortest * (longest / shortest) ** (k[:m] / max(m - 1, 1)))
    angles = torch.cat([seq, (pen_xy[..., None].float() * waves).flatten(-2)], dim=-1)[:, None]
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
        q, k = self.q_norm(q), self.k_norm(k)
        if rotary is not None:
            q, k = apply_rotary(q, *rotary), apply_rotary(k, *rotary)
        if cache is not None:
            buf, t = cache
            buf.index_copy_(3, t, torch.stack([k, v]))
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
        self.dropout = cfg.dropout

    def forward(self, x, context, rotary, ctx_mask, self_mask=None, cache=None):
        drop = lambda branch: F.dropout(branch, self.dropout, self.training)
        x = x + drop(self.attn(rms_norm(x), rotary=rotary, mask=self_mask, cache=cache))
        x = x + drop(self.cross_attn(rms_norm(x), context, mask=ctx_mask))
        return x + drop(self.fc_out(F.relu(self.fc(rms_norm(x))).square()))


class PenTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "fill in the derived ModelConfig fields first"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.ctx_emb = nn.Embedding(cfg.context_vocab_size, cfg.n_embd)
        self.ctx_pos_emb = nn.Embedding(cfg.context_block_size, cfg.n_embd)
        if cfg.context_slots > 0:
            self.ctx_proj = nn.Linear(cfg.context_width, 4 * cfg.n_embd, bias=False)
            self.ctx_slot = nn.Embedding(cfg.context_slots, 4 * cfg.n_embd)
            told = cfg.context_slots * (cfg.context_width + cfg.context_kinds)
            if cfg.context_kinds:
                self.ctx_kind = nn.Linear(cfg.context_kinds, 4 * cfg.n_embd, bias=False)
            if cfg.context_dim > told:
                self.hint_proj = nn.Linear(cfg.context_dim - told, cfg.n_embd, bias=False)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.resid_scale = nn.Parameter(torch.ones(cfg.n_layer))
        self.x0_scale = nn.Parameter(torch.zeros(cfg.n_layer))
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.n_sources > 0:
            self.source_emb = nn.Embedding(cfg.n_sources, cfg.n_embd)
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
        if self.cfg.n_sources > 0:
            nn.init.normal_(self.source_emb.weight, std=0.8)

    def embed(self, idx, pen_xy):
        x = self.tok_emb(idx)
        if self.cfg.pen_pos_bands > 0:
            k = torch.arange(self.cfg.pen_pos_bands, device=idx.device)
            angles = pen_xy[..., None].float() * (torch.pi / 2 ** (k + 1))
            x = x + self.pen_pos_proj(torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2))
        return rms_norm(x)

    def told_bytes(self, context):
        return rms_norm(self.ctx_emb(context) + self.ctx_pos_emb.weight[:context.size(1)]), context != 0

    def told_vectors(self, context):
        K, D, N, B = self.cfg.context_slots, self.cfg.context_width, self.cfg.context_kinds, len(context)
        told, end = context[:, :K * D].float().view(B, K, D), K * (D + N)
        c = self.ctx_proj(told) + self.ctx_slot.weight
        if N:
            c = c + self.ctx_kind(context[:, K * D:end].float().view(B, K, N))
        c, mask = rms_norm(c.view(B, 4 * K, -1)), (told.abs().amax(-1) > 0).repeat_interleave(4, 1)
        if context.size(1) > end:
            flags = context[:, end:].float()
            c = torch.cat([c, rms_norm(self.hint_proj(flags))[:, None]], 1)
            mask = torch.cat([mask, flags.any(1, keepdim=True)], 1)
        return c, mask

    def encode_context(self, context, source=None):
        c, mask = self.told_vectors(context) if context.is_floating_point() else self.told_bytes(context)
        if self.cfg.n_sources > 0:
            w = torch.zeros(len(context), dtype=torch.long, device=context.device) if source is None else source
            if self.training or self.cfg.source_dropout >= 1:
                w = torch.where(torch.rand(w.shape, device=w.device) < self.cfg.source_dropout, 0, w)
            c = torch.cat([rms_norm(self.source_emb(w))[:, None], c], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        return c, (mask | ~mask.any(-1, keepdim=True))[:, None, None, :]

    def logits(self, x, c, ctx_mask, rotary, self_mask=None, caches=None):
        x0 = x
        for i, block in enumerate(self.blocks):
            x = block(self.resid_scale[i] * x + self.x0_scale[i] * x0, c, rotary, ctx_mask,
                      self_mask, caches[i] if caches else None)
        return 15 * torch.tanh(self.head(rms_norm(x)) / 15)

    def forward(self, idx, context, targets=None, source=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"
        pen_xy = self.pen_deltas[idx].cumsum(dim=1)
        if self.training and self.cfg.pen_pos_jitter > 0:
            j = self.cfg.pen_pos_jitter
            pen_xy = pen_xy + torch.randint(-j, j + 1, (B, 1, 2), device=idx.device)
        head_dim = self.cfg.n_embd // self.cfg.n_head
        rotary = (pen_rotary_tables(torch.arange(T, device=idx.device), pen_xy, head_dim) if self.cfg.pen_rotary
                  else rotary_tables(T, head_dim, idx.device))
        logits = self.logits(self.embed(idx, pen_xy), *self.encode_context(context, source), rotary)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                   ignore_index=IGNORE_INDEX)
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, context, max_new_tokens, temperature=1.0, top_k=None,
                 do_sample=True, end_token=None, pad_token=None, source=None, generator=None, guidance=1.0, blank=None):
        was_training = self.training
        self.eval()
        B, T0 = idx.shape
        G = 1 if guidance == 1 or blank is None else 2
        if G == 2:
            context = torch.cat([context, blank.to(context.device).expand(B, -1)])
            source = None if source is None else source.repeat(2)
        n, dev = min(T0 + max_new_tokens, self.cfg.block_size), idx.device
        head_dim = self.cfg.n_embd // self.cfg.n_head
        caches = [torch.zeros(2, G * B, self.cfg.n_head, n, head_dim, device=dev) for _ in self.blocks]
        cos, sin = rotary_tables(n, head_dim, dev)
        c, ctx_mask = self.encode_context(context, source)
        slots, t = torch.arange(n, device=dev), torch.zeros(1, dtype=torch.long, device=dev)
        pen_xy = torch.zeros(B, 1, 2, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        tokens = list(idx.split(1, dim=1))
        for step in range(n - 1):
            pen_xy = pen_xy + self.pen_deltas[tokens[step]]
            if self.cfg.pen_rotary:
                rotary = pen_rotary_tables(t, pen_xy, head_dim)
            else:
                rotary = cos.index_select(0, t), sin.index_select(0, t)
            mask = (slots <= t)[None, None, None]
            x = self.embed(tokens[step], pen_xy)
            if G == 2:
                x, rotary = x.repeat(2, 1, 1), tuple(r.repeat(2, 1, 1, 1) if r.dim() == 4 else r for r in rotary)
            logits = self.logits(x, c, ctx_mask, rotary, mask,
                                 [(buf, t) for buf in caches])[:, -1]
            t += 1
            if step < T0 - 1:
                continue
            if G == 2:
                told, untold = logits.log_softmax(-1).chunk(2)
                logits = untold + guidance * (told - untold)
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.where(torch.as_tensor(do_sample, device=dev).view(-1, 1),
                              torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator),
                              logits.argmax(dim=-1, keepdim=True))
            if end_token is not None:
                nxt[done] = pad_token if pad_token is not None else end_token
                done |= nxt.squeeze(1) == end_token
            tokens.append(nxt)
            if end_token is not None and step % 16 == 0 and done.all():
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


def orthogonalize(G, half=True):
    if G.size(-2) > G.size(-1):
        return orthogonalize(G.mT, half).mT
    X = G.bfloat16() if half and G.is_cuda else G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in POLAR_EXPRESS:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, half=True):
        params = list(params)
        groups = [dict(params=[p for p in params if p.shape == shape])
                  for shape in sorted({p.shape for p in params})]
        super().__init__(groups, dict(lr=lr, momentum=momentum))
        self.half = half

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group["params"]
            grads = torch.stack([p.grad for p in params])
            state = self.state[params[0]]
            if "buf" not in state:
                state["buf"] = torch.zeros_like(grads)
            state["buf"].lerp_(grads, 1 - group["momentum"])
            update = orthogonalize(grads.lerp_(state["buf"], group["momentum"]), self.half)
            rows, cols = params[0].shape
            lr = group["lr"] * max(1.0, rows / cols) ** 0.5
            for p, u in zip(params, update):
                p.add_(u.to(p.dtype), alpha=-lr)


def build(cfg):
    if cfg.arch == "graves":
        from graves import GravesLSTM
        return GravesLSTM(cfg)
    return PenTransformer(cfg)


def pen_optimizers(model, cfg):
    is_matrix = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
    matrices = [p for n, p in model.named_parameters() if is_matrix(n, p)]
    rest = [(n, p) for n, p in model.named_parameters() if not is_matrix(n, p)]
    groups = [dict(params=[p for n, p in rest if "emb" in n], lr=cfg.embedding_lr),
              dict(params=[p for n, p in rest if "emb" not in n])]
    adamw = torch.optim.AdamW(groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.99), eps=1e-8,
                              fused=True)
    return [adamw, Muon(matrices, lr=cfg.matrix_lr, half=cfg.muon_half)]


def make_optimizers(model, cfg):
    opts = model.optimizers(cfg) if hasattr(model, "optimizers") else pen_optimizers(model, cfg)
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
    model.pen_deltas.copy_(torch.as_tensor(stroke_tok.token_deltas(), device=model.pen_deltas.device))


def save_checkpoint(path, model, data_config, merges=None, optimizers=None,
                    step=None, best_loss=None, sources=None):
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.cfg),
        "data_config": data_config,
        "merges": [[int(a), int(b), int(c)] for a, b, c in (merges or [])],
        "sources": list(sources or [""]),
        "step": int(step) if step is not None else None,
        "best_loss": float(best_loss) if best_loss is not None else None,
    }
    if optimizers is not None:
        checkpoint["optimizer"] = [opt.state_dict() for opt in optimizers]
    torch.save(checkpoint, path)


OLD_NAMES = {"n_writers": "n_sources", "writer_dropout": "source_dropout", "scale_jitter": "stretch", "shear_max": "shear"}


def read_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    saved = {OLD_NAMES.get(k, k): v for k, v in checkpoint["model_config"].items()}
    assert not saved.get("pen_pos_word") and not saved.get("vision"), f"{path} uses an option this code no longer has"
    assert saved.get("context_slots") or not saved.get("context_dim"), \
        f"{path} is told CLIP vectors, which this code no longer reads"
    known = {f.name: saved[f.name] for f in fields(ModelConfig) if f.name in saved}
    checkpoint["model_config"] = ModelConfig(**{"pen_pos_jitter": 0, **known})
    checkpoint["model"] = {k.replace("writer_emb.", "source_emb."): v for k, v in checkpoint["model"].items()}
    checkpoint["sources"] = checkpoint.get("sources", checkpoint.get("writers", [""]))
    return checkpoint


def load_checkpoint(path, device="cpu"):
    checkpoint = read_checkpoint(path, device)
    model = build(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def restore(saved):
    saved = {"text_end": False, "holdout": 0, "augment": True, **{OLD_NAMES.get(k, k): v for k, v in saved.items()}}
    saved["augment"] = 0.0 if saved["augment"] in ("none", False) else float(saved["augment"])
    return DataConfig(**{f.name: saved[f.name] for f in fields(DataConfig) if f.name in saved})


def load_for_sampling(path, device="cpu"):
    model, ckpt = load_checkpoint(path, device)
    cfg, m = restore(ckpt["data_config"]), model.cfg
    tok = stroke_tokenizer(cfg, ckpt["merges"])
    told = Vectors(cfg, m.context_width, m.context_kinds) if m.context_slots else Labels(cfg)
    return model, PenDataset([], [], [], tok, told, cfg, sources=ckpt["sources"]), cfg, ckpt


@dataclass
class SampleParams:
    temperature: float = 1.0
    top_k: int = None
    do_sample: bool = True
    max_tokens: int = 512
    guidance: float = None
    source: str = ""
    n_at_a_time: int = 2
    burn_in: int = 1
    space_width: float = 0.16
    line_width: float = 8.0
    line_height: float = 0.55
    seed: int = 42
    linewidth: float = 1.3


def source_id(dataset, name):
    if name and name not in dataset.sources:
        raise SystemExit(f"unknown source {name!r}; known: {', '.join(w for w in dataset.sources if w)}")
    return dataset.sources.index(name) if name else 0


def generate(model, dataset, text, params=None, prefix=()):
    params = params or SampleParams()
    st, device = dataset.stroke_tok, next(model.parameters()).device
    trained = dataset.cfg.label_drop > 0
    guidance = (2.0 if trained else 1.0) if params.guidance is None else params.guidance
    if guidance != 1 and not trained:
        raise SystemExit("guidance needs a model trained with --label_drop; this one was not")
    seq = [st.BOS] + [t for word in prefix for t in [*word, st.WORD, st.WORD]]
    out = model.generate(torch.tensor([seq], device=device), torch.from_numpy(dataset.encode_text(text))[None].to(device),
                         max_new_tokens=params.max_tokens + 1 - len(seq), temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END, pad_token=st.PAD,
                         source=torch.tensor([source_id(dataset, params.source)], device=device), guidance=guidance,
                         blank=torch.from_numpy(dataset.encode_text(""))[None])
    return out[0, len(seq):].cpu().numpy()


def sketch(model, dataset, text, params=None):
    params = params or SampleParams(temperature=0.6, max_tokens=dataset.cfg.max_seq_length - 1)
    return dataset.stroke_tok.decode(generate(model, dataset, text, params))


def write(model, dataset, text, params=None, prefix=()):
    st, new = dataset.stroke_tok, generate(model, dataset, text, params, prefix)
    end = np.flatnonzero(new == st.END)
    new = new[:end[0]] if end.size else new
    new = new[new != st.PAD]
    tokens = [w[w != st.WORD] for w in np.split(new, np.flatnonzero(new == st.WORD))]
    tokens = [t for t in tokens if len(t)][:None if end.size else -1]
    return [st.decode_word(t) for t in tokens], tokens


def write_paragraph(model, dataset, text, params=None, words=None, redo=None):
    params = params or SampleParams()
    torch.manual_seed(params.seed)
    asked = text.strip().split()
    if words is not None:
        for i in redo or []:
            words[i] = (write(model, dataset, asked[i], params)[0] + [np.zeros((0, 3))])[0]
        return words
    words, prior, prior_tokens = [], [], []
    for i in range(0, len(asked), params.n_at_a_time):
        chunk = asked[i:i + params.n_at_a_time]
        if len(" ".join(prior + chunk).encode()) > dataset.cfg.max_text_length:
            prior, prior_tokens = [], []
        out, tokens = write(model, dataset, " ".join(prior + chunk), params, prior_tokens)
        for word in chunk[len(out):]:
            more, more_tokens = write(model, dataset, word, params)
            out, tokens = out + (more or [np.zeros((0, 3))])[:1], tokens + more_tokens[:1]
        words += out[:len(chunk)]
        complete = params.burn_in and len(tokens) >= len(chunk)
        prior, prior_tokens = (chunk[-params.burn_in:], tokens[:len(chunk)][-params.burn_in:]) if complete else ([], [])
    return words


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


def plot_words(ax, words, params=None, color="b", show_indices=False):
    params = params or SampleParams()
    placed = layout_words(words, params)
    plot(ax, to_strokes(np.vstack(placed)) if placed else [], color, params.linewidth)
    for i, points in enumerate(placed if show_indices else []):
        ax.text(points[:, 0].min() - 0.08, -points[0, 1] + 0.15, str(i), fontsize=8)
    return placed


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def trim(batch, pad):
    X, C, Y, W = batch
    n = int((X != pad).sum(1).max())
    return X[:, :n], C, Y[:, :n], W


def by_length(batch, pad, size=256):
    order = (batch[0] != pad).sum(1).argsort()
    return [trim(tuple(t[rows] for t in batch), pad) for rows in order.split(size)]


def nll(model, batches, device):
    total = 0.0
    for X, C, Y, W in batches:
        k, step = 0, 16
        while k < len(X):
            rows = slice(k, k + step)
            logits, _ = model(X[rows].to(device), C[rows].to(device), source=W[rows].to(device))
            total += F.cross_entropy(logits.transpose(1, 2), Y[rows].to(device), ignore_index=IGNORE_INDEX, reduction="sum").item()
            k, step = k + step, max(16, 2 ** 28 // (logits.size(1) * logits.size(2)))
    return total


@torch.inference_mode()
def evaluate(model, dataset, device, n=300):
    st = dataset.stroke_tok
    if not hasattr(dataset, "_eval"):
        tokens = [st.encode_word(dataset.bank_points[i]) for i in dataset.indices]
        told = lambda i: dataset.encode_text(dataset.told.of(i, dataset.bank_texts[i]))
        words = [dataset.make_item(t, told(i), int(dataset.source_ids[i])) for t, i in zip(tokens, dataset.indices)]
        words = by_length(tuple(torch.stack(t) for t in zip(*words)), st.PAD)
        packed = by_length(dataset.batch(range(n)), st.PAD)
        n_tokens = sum((Y != IGNORE_INDEX).sum().item() for _, _, Y, _ in packed)
        n_moves = sum(st.moves(dataset.bank_points[i]) for i in dataset.indices)
        dataset._eval = packed, n_tokens, words, n_moves
    packed, n_tokens, words, n_moves = dataset._eval
    model.eval()
    loss, score = nll(model, packed, device) / n_tokens, nll(model, words, device) / n_moves
    model.train()
    return loss, score


@torch.inference_mode()
def unseen_score(model, dataset, device, context=(0, 1, 3)):
    st, cfg = dataset.stroke_tok, dataset.cfg
    if not hasattr(dataset, "_unseen"):
        rng = np.random.default_rng(cfg.seed)
        tok = {i: st.encode_word(dataset.bank_points[i]) for i in dataset.indices}
        sets, moves = [[] for _ in context], 0
        for group in dataset.groups:
            for i in group if len(group) > max(context) else []:
                before, items = rng.choice(group[group != i], max(context), replace=False), []
                for k in context:
                    text = " ".join(dataset.bank_texts[j] for j in [*before[:k], i])
                    tokens = np.concatenate([t for j in before[:k] for t in (tok[j], [st.WORD] * 2)] + [tok[i]]).astype(np.int64)
                    if len(tokens) > cfg.max_seq_length - 2 or len(text.encode()) > cfg.max_text_length:
                        break
                    x, c, y, w = dataset.make_item(tokens, dataset.encode_text(text))
                    y[:len(tokens) - len(tok[i])] = IGNORE_INDEX
                    items.append((x, c, y, w))
                if len(items) == len(context):
                    moves += st.moves(dataset.bank_points[i])
                    for kept, item in zip(sets, items):
                        kept.append(item)
        dataset._unseen = [by_length(tuple(torch.stack(t) for t in zip(*items)), st.PAD) for items in sets], moves
    sets, moves = dataset._unseen
    model.eval()
    scores = [nll(model, batches, device) / moves for batches in sets]
    model.train()
    return scores


@torch.inference_mode()
def style_rank(model, dataset, device, n=16, prime=3, targets=200):
    st, cfg, rng, tok = dataset.stroke_tok, dataset.cfg, np.random.default_rng(dataset.cfg.seed), {}
    groups = [g for g in dataset.groups if len(g) > prime]
    if len(groups) < n:
        return None
    encode = lambda i: tok.setdefault(i, st.encode_word(dataset.bank_points[i]))

    def primed(before, i):
        text = " ".join(dataset.bank_texts[j] for j in [*before, i])
        tokens = np.concatenate([t for j in before for t in (encode(j), [st.WORD] * 2)] + [encode(i)]).astype(np.int64)
        if len(tokens) > cfg.max_seq_length - 2 or len(text.encode()) > cfg.max_text_length:
            return None
        x, c, y, w = dataset.make_item(tokens, dataset.encode_text(text))
        y[:len(tokens) - len(encode(i))] = IGNORE_INDEX
        return x, c, y, w

    ranks = []
    model.eval()
    for _ in range(targets):
        own, *others = rng.choice(len(groups), n, replace=False)
        i, *same = rng.choice(groups[own], prime + 1, replace=False)
        items = [primed(same, i)] + [primed(rng.choice(groups[g], prime, replace=False), i) for g in others]
        if any(item is None for item in items):
            continue
        X, C, Y, W = (t.to(device) for t in trim(tuple(torch.stack(t) for t in zip(*items)), st.PAD))
        logits, _ = model(X, C, source=W)
        cost = F.cross_entropy(logits.transpose(1, 2), Y, ignore_index=IGNORE_INDEX, reduction="none").sum(1)
        ranks.append(1 + (cost[1:] < cost[0]).sum().item())
    model.train()
    return (float(np.mean(ranks)), float(np.mean(np.array(ranks) == 1)), len(ranks)) if ranks else None


@torch.inference_mode()
def text_rank(model, dataset, device, n=16):
    X, C, Y, W = (t.to(device) for t in trim(dataset.batch(range(n)), dataset.stroke_tok.PAD))
    model.eval()
    cost = torch.stack([F.cross_entropy(model(x.expand(n, -1), C, source=w.expand(n))[0].transpose(1, 2), y.expand(n, -1),
                                        ignore_index=IGNORE_INDEX, reduction="none").sum(1) for x, y, w in zip(X, Y, W)])
    model.train()
    return 1 + (cost < cost.diagonal()[:, None]).sum(1).float().mean().item()


def save_progress(model, dataset, out_dir, step, showcase=(), n=6, samples=3, showcase_source=""):
    draws = [dataset.draw(np.random.default_rng([dataset.seed, k])) for k in range(n)]
    X, C, _, W = dataset.batch(range(n))
    C = torch.cat([C] + [torch.from_numpy(dataset.encode_text(t))[None] for t in showcase])
    W = torch.cat([W, torch.full((len(showcase),), source_id(dataset, showcase_source))])
    st, device, rows, k, hand = dataset.stroke_tok, next(model.parameters()).device, len(C), samples + 1, dataset.hand
    out = model.generate(torch.full((rows * k, 1), st.BOS, device=device), C.repeat_interleave(k, 0).to(device),
                         dataset.cfg.max_seq_length - 1, do_sample=(torch.arange(rows * k) % k > 0).to(device),
                         end_token=st.END, pad_token=st.PAD, source=W.repeat_interleave(k, 0).to(device),
                         generator=torch.Generator(device).manual_seed(3)).cpu().numpy()
    names = [dataset.told.tok.decode(C[i]) if hand else dataset.bank_texts[draws[i][0][0]] for i in range(n)]
    names += list(showcase)
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    heading = f"step {step:,}: a real held-out example (green), greedy (blue), samples (black)"
    if not hand:
        drawings, titles = [], []
        for i in range(rows):
            drawings += [to_strokes(st.decode(X[i].numpy())) if i < n else []]
            drawings += [to_strokes(st.decode(t)) for t in out[i * k:(i + 1) * k]]
            titles += [f"{names[i][:28]}{'' if i < n else ' (no real drawing)'}", "greedy"] + [""] * samples
        grid(drawings, titles, samples + 2, path, colors=["darkgreen", "navy", *["k"] * samples] * rows, heading=heading)
        return path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(rows, samples + 2, figsize=(3 * (samples + 2), 1.1 * rows), squeeze=False, sharex="row",
                             sharey="row")
    for i in range(rows):
        row = [X[i].numpy() if i < n else None] + list(out[i * k:(i + 1) * k])
        for j, (ax, tokens) in enumerate(zip(axes[i], row)):
            words = [] if tokens is None else st.decode_words(tokens)
            plot_words(ax, words, SampleParams(line_width=2.5), ("darkgreen", "navy")[j] if j < 2 else "k")
            if tokens is None:
                ax.text(0.5, 0.5, "showcase prompt:\nno real writing", transform=ax.transAxes, ha="center", va="center",
                        color="0.6", fontsize=8)
        axes[i][0].set_title(names[i], loc="left", fontsize=8, pad=2)
    fig.suptitle(heading, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.03, hspace=0.3)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def distribute(cfg):
    if cfg.gpus > 1 and WORLD == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        where = ["--standalone"] if sys.platform != "darwin" else ["--nnodes=1", "--rdzv-backend=c10d",
                                                                   "--rdzv-endpoint=127.0.0.1:29511", "--local-addr=127.0.0.1"]
        os.execv(sys.executable, [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={cfg.gpus}", *where, *sys.argv])
    if WORLD == 1:
        return resolve_device(cfg.device)
    if RANK:
        sys.stdout = open(os.devnull, "w")
    seen = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(map(str, range(WORLD)))
    os.environ["CUDA_VISIBLE_DEVICES"] = seen.split(",")[int(os.environ["LOCAL_RANK"])]
    device = resolve_device(cfg.device)
    dist.init_process_group("nccl" if device == "cuda" else "gloo")
    cfg.steps, cfg.stop, cfg.warmup = -(-cfg.steps // WORLD), -(-cfg.stop // WORLD), max(100, cfg.warmup // WORLD)
    for name in ("learning_rate", "embedding_lr", "matrix_lr"):
        setattr(cfg, name, getattr(cfg, name) * WORLD ** 0.5)
    print(f"{WORLD} GPUs: {cfg.steps} steps of {WORLD} x {cfg.batch_size} sequences, learning rates x{WORLD ** 0.5:.2f}")
    return device


def sync_grads(model):
    if WORLD > 1:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        for g, mean in zip(grads, flat.split([g.numel() for g in grads])):
            g.copy_(mean.view_as(g))


def warm(step_model, model, batch_size, device, mixed=False, tokens=0):
    t0, widths = time.time(), list(range(64, model.cfg.block_size + 1, 64))
    for mine in (widths[RANK % len(widths):][:1], widths):
        for T in mine:
            X = torch.zeros(max(1, tokens // T) if tokens else batch_size, T, dtype=torch.long, device=device)
            if model.cfg.context_dim:
                C = torch.zeros(len(X), model.cfg.context_dim, device=device)
            else:
                C = torch.zeros(len(X), model.cfg.context_block_size, dtype=torch.long, device=device)
            with torch.autocast("cuda", torch.bfloat16, enabled=mixed):
                loss = step_model(X, C, X.clone(), X[:, 0].clone())[1]
            loss.backward()
            model.zero_grad(set_to_none=True)
        dist.barrier()
    print(f"compiled {len(widths)} batch widths in {time.time() - t0:.0f}s")


def train(data_cfg, model_cfg, train_cfg):
    device = distribute(train_cfg)
    torch.set_float32_matmul_precision("high")
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_cfg.out_dir, "best.pt")

    ckpt = None
    if train_cfg.resume:
        ckpt = read_checkpoint(train_cfg.resume, device)
        model_cfg = ckpt["model_config"]
        resumed_cfg = restore(ckpt["data_config"])
        if asdict(resumed_cfg) != asdict(data_cfg):
            changed = {k: (v, getattr(data_cfg, k))
                       for k, v in asdict(resumed_cfg).items()
                       if getattr(data_cfg, k) != v}
            print(f"Using the checkpoint's data config; ignoring {changed} "
                  f"(checkpoint value, command-line value)")
        data_cfg = resumed_cfg

    torch.manual_seed(data_cfg.seed + train_cfg.train_seed)
    train_dataset, test_dataset, stroke_tok = create_datasets(
        data_cfg, merges=ckpt["merges"] if ckpt else None, sources=ckpt.get("sources") if ckpt else None)

    source_id(train_dataset, train_cfg.showcase_source)

    model = build(fit(model_cfg, train_dataset)).to(device)
    attach_pen_deltas(model, stroke_tok)
    optimizers = make_optimizers(model, train_cfg)
    torch.manual_seed(data_cfg.seed + train_cfg.train_seed + RANK)

    step, best_loss = 0, None
    if ckpt:
        model.load_state_dict(ckpt["model"])
        for opt, state in zip(optimizers, ckpt["optimizer"]):
            opt.load_state_dict(state)
        step, best_loss = ckpt["step"], ckpt["best_loss"]
        print(f"Resumed from {train_cfg.resume} at step {step} (best loss {best_loss})")

    run = None
    if train_cfg.wandb and RANK == 0:
        import wandb
        run = wandb.init(project=train_cfg.wandb_project,
                         entity=train_cfg.wandb_entity or None,
                         name=train_cfg.wandb_run_name or None,
                         config={**asdict(data_cfg), **asdict(model_cfg), **asdict(train_cfg)})

    batch_seed = data_cfg.seed + step + 10_000 * train_cfg.train_seed + 1_000_003 * RANK
    batches = infinite_batches(train_dataset, train_cfg.batch_size, seed=batch_seed, tokens=train_cfg.batch_tokens,
                               workers=train_cfg.loader_workers if device == "cuda" else 1, step=train_cfg.pad)

    mode = "reduce-overhead" if train_cfg.graphs and device == "cuda" else None
    step_model = torch.compile(model, mode=mode) if train_cfg.compile and device != "cpu" else model
    mixed = train_cfg.mixed and device == "cuda"
    if WORLD > 1 and step_model is not model:
        warm(step_model, model, train_cfg.batch_size, device, mixed, train_cfg.batch_tokens)
    loss_sum, n_loss = torch.zeros((), device=device), 0
    t_start = t_print = time.time()
    t_data = t_evals = 0.0
    end = min(train_cfg.steps, train_cfg.stop or train_cfg.steps)
    while step < end:
        t0 = time.time()
        X, C, Y, W = [t.to(device, non_blocking=True) for t in next(batches)]
        t_data += time.time() - t0
        with torch.autocast("cuda", torch.bfloat16, enabled=mixed):
            _, loss = step_model(X, C, Y, W)
        model.zero_grad(set_to_none=True)
        loss.backward()
        sync_grads(model)
        loss_sum += loss.detach()
        n_loss += 1
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

        if step % train_cfg.print_every == 0:
            ms = (time.time() - t_print) * 1000 / train_cfg.print_every
            print(f"step {step} | loss {loss.item():.4f} | {ms:.0f} ms/step | lr x{mult:.3f}")
            if run:
                run.log({"train_loss_step": loss.item(), "step": step})
            t_print = time.time()
        if RANK or step % train_cfg.eval_every and step < end:
            continue

        t_eval = time.time()
        train_loss, loss_sum, n_loss = (loss_sum / max(n_loss, 1)).item(), loss_sum * 0, 0
        results = {k: evaluate(model, d, device) for k, d in test_dataset.parts.items()}
        rank = text_rank(model, test_dataset, device)
        losses, scores = ({k: r[i] for k, r in results.items()} for i in (0, 1))
        test_loss, score = float(np.mean(list(losses.values()))), float(np.mean(list(scores.values())))
        seen = {k: evaluate(model, d, device)[0] for k, d in test_dataset.seen.items() if k in losses and len(d.indices)}
        gap = float(np.mean([losses[k] - v for k, v in seen.items()])) if seen else None
        unseen = {k: unseen_score(model, d, device) for k, d in test_dataset.unseen.items()}
        row = {"step": step, "minutes": (time.time() - t_start) / 60, "train_loss": train_loss, "test_loss": test_loss,
               "gap": gap, "word_score": score, "text_rank": rank, "test_loss_by": losses, "seen_loss_by": seen,
               "word_score_by": scores, "unseen_by": unseen}
        with open(os.path.join(train_cfg.out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        if run:
            run.log({"train_loss": train_loss, "test_loss": test_loss, "gap": gap, "word_score": score, "text_rank": rank,
                     "step": step, **{f"test_loss/{k}": v for k, v in losses.items()},
                     **{f"word_score/{k}": v for k, v in scores.items()}})

        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss
            print(f"New best test loss; saving checkpoint to {checkpoint_path}")
            save_checkpoint(checkpoint_path, model, asdict(data_cfg), stroke_tok.merges, None, step, best_loss,
                            train_dataset.sources)
            if run:
                artifact = wandb.Artifact("best_checkpoint", type="model")
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact)

        save_checkpoint(os.path.join(train_cfg.out_dir, "last.pt"), model,
                        asdict(data_cfg), stroke_tok.merges,
                        optimizers, step, best_loss, train_dataset.sources)
        if train_cfg.keep:
            save_checkpoint(os.path.join(train_cfg.out_dir, f"step_{step:06d}.pt"), model, asdict(data_cfg),
                            stroke_tok.merges, None, step, test_loss, train_dataset.sources)

        if step % train_cfg.sample_every == 0 or step >= end:
            path = save_progress(model, test_dataset, os.path.join(train_cfg.out_dir, "progress"), step,
                                 [t for t in train_cfg.showcase.split("|") if t], showcase_source=train_cfg.showcase_source)
            print(f"  wrote {path}")
            if run:
                run.log({"progress": wandb.Image(path), "step": step})
        t_evals += time.time() - t_eval
        total = time.time() - t_start
        print(f"step {step} | train loss {train_loss:.4f} | test loss {test_loss:.4f}"
              f"{'' if gap is None else f' | gap {gap:+.4f}'} | word score {score:.4f} | text rank {rank:.2f}/16 | "
              f"{total / 60:.1f} min, {100 * t_evals / total:.1f}% in evals, {100 * t_data / total:.1f}% waiting for data")
        if len(losses) > 1:
            print("  " + " | ".join(f"{k}: loss {losses[k]:.4f}, word score {scores[k]:.4f}" for k in losses))
        if unseen:
            print("  unseen writers, alone / after one / after three of their words | " +
                  " | ".join(f"{k}: {a:.4f} / {b:.4f} / {c:.4f}" for k, (a, b, c) in unseen.items()))
        t_print = time.time()

    if run:
        run.finish()
    if WORLD > 1:
        dist.destroy_process_group()


def lengths_command(args):
    cfg = DataConfig(**{f.name: getattr(args, f.name) for f in fields(DataConfig)})
    tok, rng, pad = ScribeTokenizer(grid=cfg.grid, merges=learn_tokens(cfg)), np.random.default_rng(cfg.seed), args.pad
    block = cfg.max_seq_length
    print(f"{'source':16} {'drawings':>9} {'median':>7} {'99th':>6} {'cut at ' + str(block):>11} {'in a batch':>11} "
          f"{f'steps for {args.passes} passes':>20}")
    for path in sorted(glob.glob(cfg.vocab)):
        bank = Bank(Strokes(path), True)
        rows = rng.choice(len(bank), min(args.sample, len(bank)), replace=False)
        n = np.array([len(t) for t in tok.encode_many([bank[i] for i in rows])])
        width = np.minimum(block, -(-(np.minimum(n, block - 2) + 2) // pad) * pad)
        each = args.batch_tokens / width.mean()
        steps = int(round(args.passes * len(bank) * (1 - 1 / cfg.holdout) / each, -2))
        print(f"{name_of(path):16} {len(bank):9,} {np.median(n):7.0f} {np.percentile(n, 99):6.0f} "
              f"{np.mean(n > block - 2):11.1%} {each:11.0f} {steps:20,}")


def look_command(args):
    cfg = DataConfig(**{f.name: getattr(args, f.name) for f in fields(DataConfig)})
    tok, rng, drawings, titles = ScribeTokenizer(grid=cfg.grid), np.random.default_rng(args.seed), [], []
    for path, _ in dataset_specs(cfg.dataset):
        sketches = Strokes(path)
        for i in rng.choice(len(sketches), args.n, replace=False):
            points = centred(sketches.strokes(int(i)))
            bent = lambda: tok.decode_word(tok.encode_word(augment(alter(points, cfg, rng), cfg, rng)))
            drawings += [points] + [bent() for _ in range(args.cols - 1)]
            titles += [f"{name_of(path)}: {sketches.meta(int(i))['label'][:24]}"] + [""] * (args.cols - 1)
    grid([to_strokes(d) for d in drawings], titles, args.cols, args.out)


def train_command(args):
    data_cfg, model_cfg, train_cfg = configs_from_args(args)
    if data_cfg.vectors:
        need_vectors(data_cfg)
    train(data_cfg, model_cfg, train_cfg)


def sample_command(args):
    device = resolve_device(args.device)
    model, dataset, cfg, _ = load_for_sampling(args.checkpoint, device)
    temperature = args.temperature if args.temperature is not None else 1.0 if dataset.hand else 0.6
    params = SampleParams(temperature=temperature, top_k=args.top_k, do_sample=not args.greedy, guidance=args.guidance,
                          max_tokens=cfg.max_seq_length - 1, source=args.source, n_at_a_time=args.n_at_a_time,
                          burn_in=args.burn_in, seed=args.seed)
    if not dataset.hand:
        dataset.ask = (args.ink, args.strokes)
        torch.manual_seed(args.seed)
        texts = [t for t in args.text.split("|") for _ in range(args.n)]
        return grid([to_strokes(sketch(model, dataset, t, params)) for t in texts], texts, args.n, args.out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    words = write_paragraph(model, dataset, args.text, params)
    if args.redo:
        redo = [int(i) for i in args.redo.split(",") if i.strip()]
        words = write_paragraph(model, dataset, args.text, params, words=words, redo=redo)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    plot_words(ax, words, params, show_indices=args.show_indices)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


def rank_command(args):
    device = resolve_device(args.device)
    model, _, cfg, ckpt = load_for_sampling(args.checkpoint, device)
    test = create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
    rank = text_rank(model, test, device, args.n)
    print(f"step {ckpt['step']}: true text ranks {rank:.2f} of {args.n} (chance {(args.n + 1) / 2:.1f})")


def unseen_command(args):
    device = resolve_device(args.device)
    model, dataset, cfg, ckpt = load_for_sampling(args.checkpoint, device)
    for path, _ in dataset_specs(args.dataset):
        data = Strokes(path)
        texts, writers, good = words_of(data.files[0], cfg)
        rows = np.array([i for i in np.flatnonzero(good) if writers[i] not in ckpt["sources"]], int)
        if not len(rows):
            print(f"{path}: every writer was in training")
            continue
        rows = rows[np.random.default_rng(cfg.seed).permutation(len(rows))[:args.n]]
        build = lambda ix, seed, augmented, ids: PenDataset(Bank(data, False), texts, ix, dataset.stroke_tok, dataset.told,
                                                            cfg, augment=False, source_ids=ids)
        held = by_writer(build, rows, data)
        alone, one, three = unseen_score(model, held, device)
        print(f"{path}: {len(rows)} words from {len(held.groups)} writers the model never saw: "
              f"word score {alone:.4f} alone, {one:.4f} after one of their words, {three:.4f} after three")


def check_command(args):
    failed = []
    for path, _ in dataset_specs(args.dataset):
        data = Bank(Strokes(path), False)
        texts = [data.strokes.meta(i)["label"] for i in range(len(data))]
        failed += [path] * bool(check(texts, lambda i: data[i], args.grid, os.path.basename(path))[1])
    if failed:
        raise SystemExit(f"not in the shared coordinate system: {', '.join(failed)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="pengpt: a small transformer that holds a pen; it writes by hand and it draws")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train on one or more weighted .hand.parquet or .sketch.parquet files")
    train_p.add_argument("--preset", type=str, default="", help="a recipe from utils.PRESETS; explicit flags override it")
    train_p.add_argument("--profile", type=str, default="",
                         help="the machine, from utils.PROFILES (cpu, m5, a100, 8xa100, h100, 8xh100): the recipe that works "
                              "best on it; explicit flags override it")
    train_p.add_argument("--model", type=str, default="hand", choices=["hand", "sketch"], help="which model --profile trains")
    add_config_args(train_p)

    sample_p = sub.add_parser("sample", help="write text or draw labels from a checkpoint")
    sample_p.add_argument("--checkpoint", type=str, default="out/cursive/best.pt")
    sample_p.add_argument("--text", type=str, required=True, help="the text to write, or labels to draw separated by |")
    sample_p.add_argument("--out", type=str, default="out/sample.png")
    sample_p.add_argument("--source", "--writer", type=str, default="",
                          help="write in this writer's hand, or draw in the manner of this source")
    sample_p.add_argument("--temperature", type=float, default=None, help="default 1 for writing and 0.6 for drawing")
    sample_p.add_argument("--top_k", type=int, default=None)
    sample_p.add_argument("--greedy", action="store_true")
    sample_p.add_argument("--seed", type=int, default=None)
    sample_p.add_argument("--device", type=str, default="auto")
    sample_p.add_argument("--n_at_a_time", type=int, default=2, help="writing: words per model call")
    sample_p.add_argument("--burn_in", type=int, default=1,
                          help="writing: words already written that each next call continues from")
    sample_p.add_argument("--redo", type=str, default="", help="writing: comma-separated word indices to write again")
    sample_p.add_argument("--show_indices", action="store_true")
    sample_p.add_argument("--n", type=int, default=8, help="drawing: drawings of each label")
    sample_p.add_argument("--ink", type=float, default=None,
                          help="drawing: ask for this much ink, in lengths of the longer side")
    sample_p.add_argument("--strokes", type=int, default=None, help="drawing: ask for this many strokes")
    sample_p.add_argument("--guidance", type=float, default=None,
                          help="drawing: how hard a drawing is pushed towards its label; 1 is not at all")

    rank_p = sub.add_parser("rank", help="how well a checkpoint follows its prompt: the rank of each test example's own text "
                                         "among the others")
    rank_p.add_argument("--checkpoint", type=str, default="out/cursive/best.pt")
    rank_p.add_argument("--n", type=int, default=32)
    rank_p.add_argument("--device", type=str, default="auto")

    unseen_p = sub.add_parser("unseen", help="word score on writers a checkpoint never trained on: each word alone, and "
                                             "after one and three other words in the same hand")
    unseen_p.add_argument("--checkpoint", type=str, default="out/cursive/best.pt")
    unseen_p.add_argument("--dataset", type=str, required=True)
    unseen_p.add_argument("--n", type=int, default=2000)
    unseen_p.add_argument("--device", type=str, default="auto")

    check_p = sub.add_parser("check", help="handwriting files share one frame: x-height, baseline, point spacing")
    check_p.add_argument("--dataset", type=str, required=True)
    check_p.add_argument("--grid", type=float, default=DataConfig.grid)

    look_p = sub.add_parser("look", help="drawings beside augmented copies, as the model sees them")
    add_config_args(look_p, (DataConfig,))
    look_p.add_argument("--n", type=int, default=6)
    look_p.add_argument("--cols", type=int, default=8)
    look_p.add_argument("--out", type=str, default="out/augment.png")

    tokens_p = sub.add_parser("tokens", help="learn the stroke vocabulary over every file of --vocab and write it to --tokens")
    add_config_args(tokens_p, (DataConfig,))

    lengths_p = sub.add_parser("lengths", help="tokens per drawing for every file of --vocab, and the steps that make a "
                                               "number of passes")
    add_config_args(lengths_p, (DataConfig,))
    lengths_p.add_argument("--sample", type=int, default=4000)
    lengths_p.add_argument("--passes", type=int, default=60)
    lengths_p.add_argument("--batch_tokens", type=int, default=65_536)
    lengths_p.add_argument("--pad", type=int, default=16)

    known, _ = parser.parse_known_args(argv)
    import utils
    from utils import PRESETS, PROFILES, SKETCH
    recipe = {}
    if getattr(known, "profile", ""):
        if known.profile not in PROFILES:
            raise SystemExit(f"unknown profile {known.profile!r}; choose from {sorted(PROFILES)}")
        recipe = utils.recipe(known.profile, known.model)
    preset = getattr(known, "preset", "") or recipe.pop("preset", "")
    if preset and preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    train_p.set_defaults(**{**PRESETS.get(preset, {}), **recipe})
    for p in (look_p, tokens_p, lengths_p):
        p.set_defaults(**{k: v for k, v in SKETCH.items() if k in {f.name for f in fields(DataConfig)}})
    args = parser.parse_args(argv)
    if args.command == "sample" and args.seed is None:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["data_config"]
        args.seed = 0 if kind(restore(saved)) == "sketch" else 42
    as_data = lambda a: DataConfig(**{f.name: getattr(a, f.name) for f in fields(DataConfig)})
    {"train": train_command, "sample": sample_command, "rank": rank_command, "unseen": unseen_command, "check": check_command,
     "look": look_command, "lengths": lengths_command, "tokens": lambda a: learn_tokens(as_data(a), fresh=True)}[args.command](args)


if __name__ == "__main__":
    main()
