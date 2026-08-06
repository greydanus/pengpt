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

import json
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import ScribeTokenizer, CharTokenizer, learn_merges

IGNORE_INDEX = -1
Y_BASELINE = 0.65


INK_HEIGHT = 0.22


def check_scale(examples, n=200):
    """Warn if ink is not at the scale a fixed grid assumes.

    Token cost per word is proportional to ink size, so a corpus scaled 2x
    larger costs 2x the sequence length for the same shapes. Every dataset must
    arrive normalized to roughly the same height; convert.py does this, and
    this catches the cases that skipped it.
    """
    heights = []
    for e in examples[:n]:
        down = e["points"][e["points"][:, 2] == 1]
        if len(down) > 2:
            heights.append(np.quantile(down[:, 1], 0.95) - np.quantile(down[:, 1], 0.05))
    if not heights:
        return
    median = float(np.median(heights))
    if not 0.5 * INK_HEIGHT < median < 2 * INK_HEIGHT:
        print(f"WARNING: median ink height is {median:.2f}, expected ~{INK_HEIGHT}. "
              f"Sequences will be ~{median / INK_HEIGHT:.1f}x the usual length; "
              f"rescale the data or set --grid {0.012 * median / INK_HEIGHT:.4f}")


def load_examples(path):
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            with zf.open(zf.namelist()[0]) as f:
                raw = json.load(f)
    else:
        with open(path) as f:
            raw = json.load(f)

    examples = []
    for item in raw:
        points = np.array(item["points"], dtype=float)
        meta = item.get("metadata", {})
        if "aspectRatio" in meta:
            points[:, 0] *= meta["aspectRatio"]
            points[:, 1] -= Y_BASELINE
        points[:, 0] -= points[0, 0]
        text = item.get("text", meta.get("asciiSequence", ""))
        examples.append({"text": text, "points": points})
    print(f"Loaded {len(examples)} examples from {path}")
    check_scale(examples)
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


def normalize(points, cfg):
    return resample(points, cfg.spacing) if cfg.spacing > 0 else points


def augment_word(points, cfg, rng):
    points = points.copy()
    points[:, 0] += rng.uniform(cfg.shear_min, cfg.shear_max) * points[:, 1]
    points[:, 0] *= rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
    points[:, 1] *= rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
    if cfg.spacing > 0:
        jitter = rng.uniform(1 - cfg.spacing_jitter, 1 + cfg.spacing_jitter)
        points = resample(points, cfg.spacing * jitter)
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

    def __len__(self):
        return self.length

    def pick_words(self, rng):
        block = self.cfg.max_seq_length
        chosen, parts, total = [], [], 0
        for i in rng.choice(self.indices, size=self.cfg.max_words, replace=False):
            word = (augment_word(self.bank_points[i], self.cfg, rng) if self.augment
                    else normalize(self.bank_points[i], self.cfg))
            tokens = self.stroke_tok.encode_word(word)
            extra = len(tokens) + (2 if parts else 0)
            if parts and total + extra > block - 1:
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
        n = min(len(tokens), block - 1)
        x[:n] = torch.from_numpy(tokens[:n])
        x[n] = st.END
        y[:n] = x[1:n + 1]
        y[n] = st.END
        c = torch.from_numpy(self.char_tok.encode(text, self.cfg.max_text_length))
        return x, c, y

    def text_for(self, idx):
        return self.pick_words(np.random.default_rng([self.seed, idx]))[1]


def create_datasets(cfg, merges=None):
    examples = load_examples(cfg.dataset)
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
        corpus = [base.encode_word(augment_word(bank_points[i], cfg, rng)) for i in sample]
        merges = learn_merges(corpus, cfg.n_merges)
        print(f"Learned {len(merges)} BPE merges")
    stroke_tok = ScribeTokenizer(grid=cfg.grid, merges=merges)

    def build(ix, n, augment, name, seed):
        return PenDataset(bank_points, bank_texts, ix, stroke_tok, char_tok, cfg,
                          length=n, augment=augment and cfg.augment, name=name, seed=seed)

    train_dataset = build(train_ix, cfg.train_size, True, "train", cfg.seed)
    test_dataset = build(test_ix, cfg.test_size, True, "test", cfg.seed + 1)
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
