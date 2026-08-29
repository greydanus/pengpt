"""pengpt: a minimal transformer for pen strokes, in one file.

Give it a corpus of pen trajectories -- handwriting, sketches, anything drawn
with a stylus or mouse -- and it learns to generate more of them, conditioned
on text. The same core algorithm trains on every supported dataset; per-dataset
settings live in utils.PRESETS.

    python pengpt.py train --preset cursive
    python pengpt.py train --preset quickdraw
    python pengpt.py sample --checkpoint out/cursive/best.pt --text "hello world"
    python pengpt.py rank --checkpoint out/quickdraw/best.pt --labels cat,car,fish

The file reads top to bottom as the pipeline runs: configuration, tokenizer,
data, model, sampling, then the three commands.
"""

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
# Every figure here is written to a file, never shown. Left to choose for
# itself, matplotlib picks an interactive backend where one is available, which
# on macOS opens a GUI window per eval from inside a headless training run: it
# logs ApplePersistenceIgnoreState, takes about thirty seconds to write what Agg
# writes in one, and can block the run outright when no session is attached.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
#
# Defaults come from wall-clock ablations on bigbank_3500, scored in bits per
# pen-point on held-out words. Two are worth knowing about because they are
# specific to the machine rather than to the task:
#
# - batch_size is small. Throughput on MPS is flat in batch size (47 ktok/s at
#   8, 58 at 128), so a larger batch buys no tokens per second and simply takes
#   fewer optimizer steps in the same time. On hardware that does scale, raise it.
# - grid trades reconstruction error against sequence length. 0.020 keeps error
#   inside the width of a pen stroke while costing 99 tokens per word.
#
# Comparing two runs: reported loss is nats per token, which is only comparable
# between runs whose tokenizers agree. Anything that changes tokens per example
# -- grid, n_merges, how merges apply -- rescales it, so multiply by tokens per
# example first.
#
# The defaults suit handwriting. Drawing corpora want --max_words 1
# --augment general and per-corpus lengths; utils.PRESETS records the settled
# values per dataset.
# ---------------------------------------------------------------------------


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
    # Probability of mirroring a drawing left-right. Skipped for any example
    # whose caption says "left" or "right", so the text never lies about the
    # picture. A mirrored scene is a valid scene; mirrored handwriting is not,
    # so leave this off for text corpora.
    hflip: float = 0.0
    # Hand-tremor amplitude in ink units, for designer-drawn sources (icon
    # sets, procedural sketches) whose ruler-perfect lines would otherwise
    # teach a drafting machine's hand. 0 disables.
    tremor: float = 0.0
    # Probability of deleting each stroke independently (at least one always
    # survives). A scene of sixty strokes minus three still matches its
    # caption; a word minus a letter does not, so again: drawings only.
    stroke_dropout: float = 0.0
    seed: int = 1337


@dataclass
class ModelConfig:
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 64
    # Fourier features of the pen's absolute canvas position, added to the
    # token embedding. The tokens are relative motion, so without this the
    # model must integrate the whole walk with attention to know whether two
    # strokes connect; with it, position is an input rather than a
    # computation. On by default; the measured gain is real but small (~0.02
    # nats per token on scene sketches, less on short drawings), so
    # --pen_pos_bands 0 opts out. Wavelengths are 4 * 2^k grid cells; the
    # longest period (4 * 2^(bands-1)) must exceed the largest within-sample
    # position range plus twice the jitter, or positions alias.
    pen_pos_bands: int = 8
    # Training-time random canvas offset in grid cells. Absolute layout can't
    # be memorized when the origin moves, but within-sample geometry -- which
    # strokes touch, what is already drawn where -- survives translation.
    pen_pos_jitter: int = 32
    vocab_size: int = -1
    block_size: int = -1
    context_vocab_size: int = -1
    context_block_size: int = -1


DERIVED_FIELDS = {"vocab_size", "block_size", "context_vocab_size",
                  "context_block_size"}

CHOICES = {"augment": ("none", "general", "handwriting")}


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
    # 0 loads data on the main process. The models are small enough that
    # loading is never the bottleneck, and DataLoader worker spawn deadlocks
    # before step 1 on some macOS + recent-Python combinations.
    num_workers: int = 0
    device: str = "auto"
    out_dir: str = "out/default"
    resume: str = ""
    wandb: bool = False
    wandb_project: str = "pengpt"
    wandb_entity: str = ""
    wandb_run_name: str = ""


def add_config_args(parser):
    """Generate one CLI from the three config dataclasses."""
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

    def build(cls):
        return cls(**{f.name: args[f.name] for f in fields(cls) if f.name in args})

    return build(DataConfig), build(ModelConfig), build(TrainConfig)


def _filter_config(cls, d):
    """Keep only the fields cls still has, reporting anything dropped.

    Checkpoints written before the repo was consolidated carry config keys for
    removed features (text encoders, pen-position features, source labels).
    Their weights still load -- load_state_dict is non-strict -- but a key that
    was actually in use deserves a warning rather than silence.
    """
    known = {f.name for f in fields(cls)}
    dropped = {k: v for k, v in d.items() if k not in known and v}
    if dropped:
        print(f"WARNING: checkpoint used removed features, ignoring {dropped}")
    return {k: v for k, v in d.items() if k in known}


# ---------------------------------------------------------------------------
# Tokenizer
#
# ScribeTokens ("ScribeTokens: Fixed-Vocabulary Tokenization of Digital Ink",
# arXiv:2603.02805): coordinates are quantized to an integer grid and motion
# becomes a walk on that grid -- eight compass directions (Freeman chain codes)
# plus two pen-state tokens.
#
#     0:right 1:up-right 2:up 3:up-left 4:left 5:down-left 6:down 7:down-right
#     8:DOWN (pen touches paper)   9:UP (pen lifts)
#
# Movement between grid points is decomposed with Bresenham's line algorithm,
# so any path is representable. Byte pair encoding then merges recurring runs,
# which is what makes sequences short.
#
# Two properties matter. Insensitivity to sampling rate: recording the same
# shape more finely barely changes the tokens, and changes them not at all once
# samples land in adjacent grid cells. Point density in raw pen data is an
# artifact of the recorder, so this is what lets a mouse, a digitizer and a
# public dataset share one representation. No out-of-vocabulary: a grid walk
# always decomposes into base tokens, unlike bin tables that must be retuned
# per dataset and silently clip outliers.
#
# A word is an (N, 3) array of (x, y, pen); pen == 1 while touching the paper,
# and rows with pen == 0 mark a lift, so strokes are the maximal runs of
# pen == 1.
# ---------------------------------------------------------------------------

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
        self.DOWN, self.UP = DOWN, UP
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


# ---------------------------------------------------------------------------
# Data
#
# On-disk format is a JSON list (optionally zipped) of examples:
#
#     {"text": "hello", "points": [[x, y, pen], ...]}
#
# Data from collect.html instead carries a metadata dict (asciiSequence,
# aspectRatio); both forms are normalized on load. Each example is one word,
# and training examples pack random words together until the block is full, so
# a few thousand words yield effectively unlimited distinct examples and no
# example is ever truncated mid-word.
#
# On resampling: point density in raw pen data is often an artifact of capture
# hardware, and resampling to uniform arc length removes it. The bundled
# bigbank data does not need this -- collect.html records a point every time
# the pen has moved a fixed distance -- hence cfg.spacing defaults to 0. Set it
# for time-sampled sources such as IAM, where density really varies with speed.
# ---------------------------------------------------------------------------

IGNORE_INDEX = -1
Y_BASELINE = 0.65

INK_HEIGHT = 0.22


def ink_height(examples, n=200):
    """Median vertical extent of the ink, ignoring outliers."""
    heights = [np.quantile(d[:, 1], 0.95) - np.quantile(d[:, 1], 0.05)
               for d in (e["points"][e["points"][:, 2] == 1] for e in examples[:n])
               if len(d) > 2]
    return float(np.median(heights)) if heights else 0.0


def check_scale(examples, grid, n=200):
    """Warn when ink size and grid size disagree.

    The grid is a fixed distance, so tokens per example scale with how large the
    drawing is: the same shapes at twice the size cost twice the sequence. What
    matters is the ratio of ink size to grid, not either alone. INK_HEIGHT is
    simply the scale the bundled data happens to use; a dataset at a different
    scale is fine as long as --grid moves with it.
    """
    height = ink_height(examples, n)
    if height <= 0:
        return
    ratio = height / INK_HEIGHT
    if not 0.5 < ratio < 2:
        print(f"WARNING: ink is {ratio:.1f}x the usual size (height {height:.2f}). "
              f"Sequences will be about {ratio:.1f}x as long; "
              f"pass --grid {grid * ratio:.4f} to compensate, or rescale the data.")


def read_items(path, limit=None):
    """Yield raw items from .json, .zip, .jsonl, or either of the last gzipped.

    JSON Lines matters at corpus scale: a 12M-drawing array cannot be parsed
    into memory at once, but one line at a time can, and `limit` then caps how
    much of a large corpus is used.
    """
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
    """Place points at equal distances along each pen-down stroke.

    It is the path that is sampled uniformly, not the chords: the straight-line
    distance between consecutive outputs is at most spacing, and less wherever
    the path curves between them. Pen-up rows are lift markers, not movements,
    so they pass through untouched.
    """
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
    """Canonicalize one trajectory, augmenting it when an rng is given.

    Augmentations come in two tiers. The general ones -- rescaling, rotation,
    and jittered resampling -- make sense for any pen data, since a drawing is
    still the same drawing slightly larger or slightly turned. Shear is
    handwriting-only: it is always negative, so it imposes an italic slant
    rather than jittering, and it presumes a baseline to slant about. On a
    sketch it is a distortion, so cfg.augment selects the tier.
    """
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
    """Hand tremor and endpoint slop, per pen-down stroke.

    Designer-drawn sources (icon sets, procedural sketches) are ruler-perfect:
    exact circles, straight lines, corners that close. A model trained on them
    learns a drafting machine's hand. This bridges toward human ink: smooth
    low-frequency noise perpendicular to each path -- knots every ~wavelength
    of arc, interpolated, so lines wave the way a freehand line waves rather
    than jittering per point -- plus endpoints extended or trimmed by a few
    millimetres of canvas, so corners overshoot or fall short.

    amp is in ink units (INK_HEIGHT is 0.22, so 0.004 is ~2% of letter height,
    about a third of the default token grid: visible waver, same drawing).
    """
    out = points.copy()
    for start, stop in _stroke_spans(points[:, 2]):
        seg = out[start:stop, :2]
        if len(seg) < 3:
            continue
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        if d[-1] < 1e-6:
            continue
        knots = max(3, int(d[-1] / wavelength) + 2)
        knot_val = rng.normal(0, amp, knots)
        offset = np.interp(d, np.linspace(0, d[-1], knots), knot_val)
        tangent = np.gradient(seg, axis=0)
        norm = np.hypot(tangent[:, 0], tangent[:, 1]) + 1e-9
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / norm[:, None]
        seg += offset[:, None] * normal
        # Endpoint slop: overshoot along the end tangent, or stop short.
        for end, t in ((0, seg[0] - seg[1]), (-1, seg[-1] - seg[-2])):
            delta = rng.uniform(-end_scale, end_scale)
            if delta > 0:
                seg[end] += t / (np.hypot(*t) + 1e-9) * delta
        out[start:stop, :2] = seg
    return out


def augment_drawing(points, text, cfg, rng):
    """Drawing-only augmentations that need the caption or stroke structure.

    These run after prepare_word's geometric jitter. All are off by default
    and belong to drawing corpora: mirrored or stroke-dropped handwriting
    stops matching its transcript, but a scene stays a scene. Which of them
    suit a dataset is the preset's decision -- tremor humanizes designer
    geometry but has no business on human ink, and a dropped stroke can be
    the x in an icon labelled "ticket x".
    """
    if cfg.tremor > 0:
        # Amplitude varies per drawing: a corpus of one fixed waver is as
        # synthetic as no waver. Sampling U(0.3, 1) x tremor spans careful
        # hands to casual ones without reaching the level that mangles small
        # elements (measured: 2x the default visibly distorts a plus sign).
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
                    # The lift row after the run belongs to the stroke.
                    lift = stop < len(points) and points[stop, 2] == 0
                    keep[start:stop + (1 if lift else 0)] = False
            points = points[keep]
    if cfg.hflip > 0 and rng.random() < cfg.hflip:
        lowered = text.lower()
        if "left" not in lowered and "right" not in lowered:
            points = points.copy()
            # Reflect within the bounding box, so coordinates keep their range.
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
        """Pack words until the block is full, sometimes stopping short.

        Filling greedily every time teaches the model that END arrives when the
        block is nearly full, rather than when the prompt runs out: it then runs
        on past a short prompt, repeating strokes to fill the space. Choosing a
        word count up front decorrelates END from position in the block.
        """
        block = self.cfg.max_seq_length
        # A bank smaller than max_words is normal on the test split, which is
        # only 5% of the corpus; drawing without replacement must not ask for
        # more words than exist.
        draw = min(self.cfg.max_words, len(self.indices))
        limit = rng.integers(1, draw + 1)
        chosen, parts, total = [], [], 0
        for i in rng.choice(self.indices, size=draw, replace=False)[:limit]:
            word = prepare_word(self.bank_points[i], self.cfg,
                                rng if self.augment else None)
            if self.augment:
                word = augment_drawing(word, self.bank_texts[i], self.cfg, rng)
            tokens = self.stroke_tok.encode_word(word)
            extra = len(tokens) + (2 if parts else 0)
            if parts and total + extra > block - 2:   # room for BOS and END
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

        # x is [BOS, tokens..., END]; y is x shifted left, so position 0 learns
        # to predict the first real token from BOS alone. Generation seeds with
        # that same BOS, so its first step is one the model has seen.
        x = torch.full((block,), st.PAD, dtype=torch.long)
        y = torch.full((block,), IGNORE_INDEX, dtype=torch.long)
        n = min(len(tokens), block - 2)
        x[0] = st.BOS
        x[1:n + 1] = torch.from_numpy(tokens[:n])
        x[n + 1] = st.END
        y[:n + 1] = x[1:n + 2]
        # A word longer than the block is cut, and the cut is not where the
        # drawing ends: supervising END there teaches that drawings stop
        # wherever the block does. Everything before the cut is still real.
        if len(tokens) > n:
            y[n] = IGNORE_INDEX
        c = self.encode_text(text)
        return x, torch.from_numpy(c), y

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
                          length=n, augment=cfg.augment != "none", name=name,
                          seed=seed)

    train_dataset = build(train_ix, cfg.train_size, "train", cfg.seed)
    test_dataset = build(test_ix, cfg.test_size, "test", cfg.seed + 1)
    print(f"Word bank: {len(train_ix)} train / {len(test_ix)} test words; "
          f"stroke vocab {stroke_tok.vocab_size}; "
          f"alphabet ({len(alphabet)} chars): {alphabet!r}")
    return train_dataset, test_dataset, stroke_tok, char_tok


class InfiniteDataLoader:

    def __init__(self, dataset, **kwargs):
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True,
                                                 num_samples=int(1e10))
        self.loader = DataLoader(dataset, sampler=sampler, **kwargs)
        self.iterator = iter(self.loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


# ---------------------------------------------------------------------------
# Model
#
# A GPT-style decoder with cross-attention over a character context, in the
# makemore/nanoGPT lineage: pre-LayerNorm blocks of (causal self-attention,
# cross-attention, MLP). Attention uses scaled_dot_product_attention, which
# dispatches to flash attention where available.
#
# The prompt reaches the model only through cross-attention, never as a class
# index, so the same architecture takes a word to write or the name of an
# object to draw, and an unseen label still says something through its
# characters.
# ---------------------------------------------------------------------------


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
            # Net displacement per token in grid cells; filled from the
            # tokenizer by attach_pen_deltas and restored from the state dict
            # on load.
            self.register_buffer("pen_deltas",
                                 torch.zeros(cfg.vocab_size, 2, dtype=torch.long))
            self.pen_pos_proj = nn.Linear(4 * cfg.pen_pos_bands, cfg.n_embd,
                                          bias=False)
        self.apply(self._init_weights)
        print(f"PenTransformer parameters: {sum(p.numel() for p in self.parameters()):,}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _pen_features(self, idx):
        """Fourier features of the pen's absolute position after each token.

        Positions are the inclusive cumsum of per-token displacements, so a
        token's feature reflects the pen after that token -- causal, since the
        next-token prediction at position t may see everything through t.
        """
        positions = self.pen_deltas[idx].cumsum(dim=1)
        if self.training and self.cfg.pen_pos_jitter > 0:
            jitter = self.cfg.pen_pos_jitter
            positions = positions + torch.randint(-jitter, jitter + 1,
                                                  (idx.size(0), 1, 2),
                                                  device=idx.device)
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
        # A prompt of only padding (every char outside the alphabet) would
        # mask every key, and softmax over an empty row is NaN. Attending
        # uniformly to padding instead degrades to unconditional generation.
        ctx_mask = ctx_mask | ~ctx_mask.any(-1, keepdim=True)

        for block in self.blocks:
            x = block(x, c, ctx_mask)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, context, max_new_tokens, temperature=1.0, top_k=None,
                 do_sample=True, end_token=None, pad_token=None):
        """Autoregressively extend idx (B, T). If end_token is given, sequences
        that emit it are padded out with pad_token and generation stops early
        once every sequence has finished."""
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
    """Copy the tokenizer's per-token displacement table into the model."""
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
    """Returns (model, checkpoint_dict). The checkpoint carries the alphabet
    and data config needed to rebuild matching tokenizers."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    saved = _filter_config(ModelConfig, checkpoint["model_config"])
    # A checkpoint from before the feature existed has no pen_pos_bands key;
    # letting it default on would pair trained weights with an untrained
    # position projection. Absent key means the feature was off.
    saved.setdefault("pen_pos_bands", 0)
    model = PenTransformer(ModelConfig(**saved))
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    return model, checkpoint


def load_for_sampling(path, device="cpu", n_examples=200):
    """Rebuild everything a checkpoint needs to generate: model, dataset, config.

    The dataset comes back because generation reads the tokenizers and the text
    of held-out examples from it; the BPE merges and alphabet ride along in the
    checkpoint, so the tokenizer always matches the trained model.
    """
    model, checkpoint = load_checkpoint(path, device)
    cfg = DataConfig(**_filter_config(DataConfig, checkpoint["data_config"]))
    cfg.train_size = cfg.test_size = n_examples
    _, dataset, stroke_tok, char_tok = create_datasets(cfg, merges=checkpoint["merges"])
    assert char_tok.alphabet == checkpoint["alphabet"], \
        "dataset alphabet does not match the checkpoint; use the training dataset"
    attach_pen_deltas(model, stroke_tok)
    return model, dataset, cfg, checkpoint


# ---------------------------------------------------------------------------
# Sampling and plotting
#
# Everything here builds on two primitives: `generate` turns a text prompt into
# per-word point arrays, and `draw` puts point arrays on a matplotlib axis.
# ---------------------------------------------------------------------------


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
    """Text prompt -> list of per-word (N, 3) point arrays.

    Nothing from the training set seeds the sequence: the model starts empty and
    the prompt alone drives it, so a sample cannot copy strokes it was shown.
    """
    params = params or SampleParams()
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    context = torch.from_numpy(
        dataset.encode_text(text)).unsqueeze(0).to(device)
    idx = torch.full((1, 1), st.BOS, dtype=torch.long, device=device)
    out = model.generate(idx, context, max_new_tokens=params.max_tokens,
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END, pad_token=st.PAD)
    return st.decode(out[0].cpu().numpy()[1:])


def draw(ax, points, color="b", linewidth=1.3):
    """Draw absolute pen points (N, 3), lifting the pen where pen == 0."""
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
    """Place per-word point arrays on a page, left to right with wrapping.

    Each word carries its own height above the baseline, so this only advances
    horizontally and wraps lines; no per-alphabet table is involved.
    """
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
    """Lay out per-word arrays and draw them on one axis."""
    params = params or SampleParams()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    placed = layout_words(words, params)
    draw(ax, np.vstack(placed) if placed else np.zeros((0, 3)), color, params.linewidth)
    if title:
        ax.set_title(title)
    return fig, ax, placed


def plot_paragraph(words, text="", params=None, figsize=(12, 8), dpi=200,
                   show_indices=False, include_title=False):
    params = params or SampleParams()
    fig, ax, placed = plot_words(words, params, figsize=figsize, dpi=dpi)
    if show_indices:
        for i, points in enumerate(placed):
            ax.text(points[:, 0].min() - 0.08, -points[0, 1] + 0.15, str(i), fontsize=8)
    if include_title and text:
        ax.set_title("\n".join(textwrap.wrap(text, width=83)), loc="left", fontsize=13)
    return fig, ax


def generate_paragraph(model, dataset, text, params=None, words=None, redo=None):
    """Generate a paragraph, n_at_a_time words per model call.

    Words come in small groups because neighbours act as context. Groups of up
    to four recover every word; larger groups overflow the block -- six words
    cost more than 512 tokens about 70% of the time -- and the words that do not
    fit are silently dropped.

    Pass a previous result as `words` plus indices as `redo` to regenerate only
    the words that came out wrong.
    """
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


def progress_prompts(dataset, n=8):
    """Prompts drawn from the dataset's own text, so they are always in vocabulary.

    Hardcoded prompts silently break on a new corpus: the handwriting defaults
    contain digits and a capital S, and Quick, Draw!'s 35-character alphabet has
    neither, so those characters encoded as padding and two of three progress
    panels showed nothing meaningful for a whole run.

    A dataset with no usable text raises rather than falling back to those
    defaults, which would reintroduce exactly that failure somewhere quieter.
    """
    seen, out = set(), []
    for i in range(min(len(dataset), 2000)):
        text = dataset.text_for(i)
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
    """Same prompts every eval, so the strip tracks the model not the prompt."""
    if not hasattr(dataset, "_progress_prompts"):
        dataset._progress_prompts = progress_prompts(dataset)
    return dataset._progress_prompts


def save_progress(model, dataset, out_dir, step, prompts=None, temperature=1.0,
                  rows=4):
    """A grid of samples: one column per prompt, `rows` samples down each.

    Several samples per prompt is what makes the picture readable. One sample
    cannot distinguish a model that ignores its prompt from one that drew a
    poor sample, and a column of four shows immediately whether the prompt
    controls the shape or the model is drawing the corpus average whatever it
    is asked for.

    Prompts and seeds are fixed across evals, so consecutive images differ only
    by the model.
    """
    prompts = prompts or _cached_prompts(dataset)
    os.makedirs(out_dir, exist_ok=True)
    params = SampleParams(temperature=temperature,
                          max_tokens=dataset.cfg.max_seq_length - 1)

    fig, axes = plt.subplots(rows, len(prompts),
                             figsize=(1.9 * len(prompts), 1.9 * rows),
                             squeeze=False)
    # One batched generate for the whole grid instead of rows x cols separate
    # calls: at long block sizes the sequential version dominated eval time
    # (each call re-runs its full prefix every token). A fixed seed keeps the
    # grid deterministic per eval, so consecutive images still differ only by
    # the model.
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    contexts = torch.stack([
        torch.from_numpy(dataset.encode_text(text))
        for text in prompts for _ in range(rows)]).to(device)
    torch.manual_seed(3)
    idx = torch.full((len(contexts), 1), st.BOS, dtype=torch.long, device=device)
    out = model.generate(idx, contexts, max_new_tokens=params.max_tokens,
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END,
                         pad_token=st.PAD)
    for col in range(len(prompts)):
        for row in range(rows):
            words = st.decode(out[col * rows + row].cpu().numpy()[1:])
            plot_words(words, params, ax=axes[row][col], color="k")
    fig.suptitle(f"step {step:,}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Label the columns on the figure rather than per axis. An axis title sits
    # relative to that axis's contents, so a column of short drawings puts its
    # label lower than the rest; placing them at one figure height keeps them
    # aligned however tall the samples happen to be.
    top = max(ax.get_position().y1 for ax in axes[0])
    for col, text in enumerate(prompts):
        box = axes[0][col].get_position()
        # Wrap to the column width, or sentence-length prompts run into their
        # neighbors. va="bottom" grows extra lines upward, into space
        # bbox_inches="tight" then reclaims.
        wrapped = "\n".join(textwrap.wrap(text, width=24))
        fig.text(box.x0 + box.width / 2, top + 0.012, wrapped,
                 ha="center", va="bottom", fontsize=8)
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def save_samples(model, dataset, out_dir=".", num=3, do_sample=True):
    """Generate from test prompts and save one PNG per example."""
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


# ---------------------------------------------------------------------------
# Commands: train, sample, rank
# ---------------------------------------------------------------------------


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

    # Resuming has to rebuild the *checkpoint's* tokenizer, not one re-derived
    # from the command line. Merges and alphabet decide what every token id
    # means, so re-learning them from a config that differs in --n_merges,
    # --grid, --dataset or --seed loads weights against a vocabulary they were
    # never trained on -- silently, whenever the sizes happen to still match.
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
        data_cfg = resumed_cfg

    torch.manual_seed(data_cfg.seed)   # after resume, so it is the run's own seed

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

    loader = InfiniteDataLoader(train_dataset, batch_size=train_cfg.batch_size,
                                pin_memory=(device == "cuda"),
                                num_workers=train_cfg.num_workers)

    while step < train_cfg.steps:
        t0 = time.time()
        X, C, Y = [t.to(device) for t in loader.next()]
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

        if step % train_cfg.eval_every == 0:
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
                    import wandb
                    artifact = wandb.Artifact("best_checkpoint", type="model")
                    artifact.add_file(checkpoint_path)
                    run.log_artifact(artifact)

            # Resuming needs where the run stopped, which is not where it was
            # last best: on a long run those diverge by thousands of steps, and
            # --resume best.pt silently replays them.
            save_checkpoint(os.path.join(train_cfg.out_dir, "last.pt"), model,
                            char_tok.alphabet, asdict(data_cfg), stroke_tok.merges,
                            optimizer, scheduler, step, best_loss)

            progress = save_progress(model, test_dataset,
                                     os.path.join(train_cfg.out_dir, "progress"),
                                     step)
            paths = [progress]
            if step % (train_cfg.eval_every * 4) == 0:
                paths += save_samples(model, test_dataset,
                                      os.path.join(train_cfg.out_dir, "samples"),
                                      num=3, do_sample=True)
            print(f"  wrote {progress}")
            if run:
                import wandb
                run.log({os.path.basename(p): wandb.Image(p) for p in paths})

    if run:
        run.finish()


def sample_command(args):
    """Generate a paragraph of handwriting (or a drawing) from a checkpoint."""
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

    fig, _ = plot_paragraph(words, args.text, params, show_indices=args.show_indices)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


RANK_LABELS = ("cat,apple,car,fish,tree,house,star,umbrella,clock,ladder,"
               "banana,bicycle")


def rank_command(args):
    """How strongly does the text prompt determine what the model draws?

    Score one real drawing under every candidate label and see where its true
    label ranks. If conditioning works the true label fits best, and the rank
    is interpretable on its own: 1 of N is perfect, (N+1)/2 is chance.

    Worth measuring separately from loss. A drawing model can post a healthy,
    falling loss while ignoring its prompt entirely -- it learns the average
    shape of the corpus and draws that regardless -- and samples alone are
    ambiguous early in training, when everything looks like a scribble whether
    or not the prompt is being read.

    The reported spread, between the best and worst label's loss, says whether
    the model merely prefers the right label or is actually driven by it. A
    high rank with a tiny spread means the prompt is being read but barely
    steers generation, which looks like every prompt producing the same
    drawing.

    Score several drawings per label (--per_label). One drawing per label is
    not a small sample of the right thing, it is the wrong measurement:
    per-label ranks are strongly bimodal, and a model can rank triangle 1st
    and mouth 20th at the same checkpoint. Read the per-label breakdown, not
    only the mean.
    """
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

    ranks, spreads = [], []
    by_label = {}
    for true, indices in sorted(found.items()):
        for i in indices:
            x, _, y = dataset[i]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            scored = []
            for label in labels:
                context = torch.from_numpy(
                    dataset.encode_text(label)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    _, loss = model(x, context, y)
                scored.append((label, loss.item()))
            scored.sort(key=lambda r: r[1])
            rank = [l for l, _ in scored].index(true) + 1
            spread = (scored[-1][1] - scored[0][1]) / scored[0][1] * 100
            ranks.append(rank)
            spreads.append(spread)
            by_label.setdefault(true, []).append(rank)

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
                        help="drawings per label; one is not enough, "
                             "per-label ranks are strongly bimodal")
    rank_p.add_argument("--device", type=str, default="auto")

    known, _ = parser.parse_known_args(argv)
    if getattr(known, "preset", ""):
        from utils import PRESETS
        if known.preset not in PRESETS:
            raise SystemExit(f"unknown preset {known.preset!r}; "
                             f"choose from {sorted(PRESETS)}")
        train_p.set_defaults(**PRESETS[known.preset])
    args = parser.parse_args(argv)

    if args.command == "train":
        train(*configs_from_args(args))
    elif args.command == "sample":
        sample_command(args)
    elif args.command == "rank":
        rank_command(args)


if __name__ == "__main__":
    main()
