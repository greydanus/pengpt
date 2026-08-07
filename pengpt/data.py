"""Dataset loading, augmentation, and the PenDataset used for training.

On-disk format is a JSON list (optionally zipped) of examples:

    {"text": "hello", "points": [[x, y, pen], ...]}

Data from collect.html instead carries a metadata dict (asciiSequence,
aspectRatio); both forms are normalized on load. Each example is one word, and
training examples pack random words together until the block is full, so a few
thousand words yield effectively unlimited distinct examples and no example is
ever truncated mid-word.

On resampling. Point density in raw pen data is often an artifact of capture
hardware, and resampling to uniform arc length removes it. The bundled bigbank
data does not need this: collect.html records a point every time the pen has
moved a fixed distance, so its spacing is already uniform at ~0.011, and
resampling only discards detail. Hence cfg.spacing defaults to 0. Set it for
time-sampled sources such as IAM, where density really does vary with speed.
"""

import gzip
import json
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import ScribeTokenizer, CharTokenizer, learn_merges

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


def _read(path, limit=None):
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
    for item in _read(path, limit):
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
        c = torch.from_numpy(self.char_tok.encode(text, self.cfg.max_text_length))
        return x, c, y

    def text_for(self, idx):
        return self.pick_words(np.random.default_rng([self.seed, idx]))[1]


def create_datasets(cfg, merges=None):
    examples = load_examples(cfg.dataset, getattr(cfg, 'max_examples', 0) or None)
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
