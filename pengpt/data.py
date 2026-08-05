"""Dataset loading, augmentation, and the PenDataset used for training.

The on-disk format is a JSON list (optionally zipped) of examples:

    {"text": "hello", "points": [[x, y, pen], ...]}

Data collected with collect.html instead carries a ``metadata`` dict
(``asciiSequence``, ``aspectRatio``); load_examples normalizes both forms.
Each example is one *word*; training examples are made by stitching together
random combinations of num_words words.
"""

import json
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import StrokeTokenizer, CharTokenizer, word_to_offsets

IGNORE_INDEX = -1   # loss is not computed at these target positions
Y_BASELINE = 0.65   # collect.html canvas baseline, as a fraction of its height


def load_examples(path):
    """Load a dataset file and return a list of {'text', 'points'} dicts."""
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
        if "aspectRatio" in meta:  # collect.html format: normalize to baseline coords
            points[:, 0] *= meta["aspectRatio"]
            points[:, 1] -= Y_BASELINE
        points[:, 0] -= points[0, 0]
        text = item.get("text", meta.get("asciiSequence", ""))
        examples.append({"text": text, "points": points})
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def make_combos(n_bank, n_combos, n_words, rng):
    """Random index tuples into the word bank; words are stitched lazily."""
    n_words = min(n_words, n_bank)
    return [rng.choice(n_bank, size=n_words, replace=False) for _ in range(n_combos)]


def downsample_word(points, fraction, drop_prob, rng):
    """Remove ~fraction of the points inside each pen-down stroke.

    Stroke endpoints are always kept: losing them creates gaps between strokes
    that should join. drop_prob adds extra per-point dropout so that a letter's
    position in the token sequence is decorrelated from its position on paper.
    """
    if fraction <= 0:
        return points

    def thin(stroke):
        n_keep = max(2, int(len(stroke) * (1 - fraction)))
        idx = np.linspace(0, len(stroke) - 1, n_keep, dtype=int)
        kept = [stroke[i] for i in idx]
        if drop_prob > 0:
            kept = [p for j, p in enumerate(kept)
                    if j in (0, len(kept) - 1) or rng.random() > drop_prob]
        return kept

    out, stroke = [], []
    for point in points:
        if point[2] == 1:
            stroke.append(point)
        else:
            if stroke:
                out.extend(thin(stroke))
                stroke = []
            out.append(point)
    if stroke:
        out.extend(thin(stroke))
    return np.array(out)


def augment_word(points, cfg, rng):
    """Shear (slant), rescale, and downsample one word. Returns a new array."""
    points = points.copy()
    points[:, 0] += rng.uniform(cfg.shear_min, cfg.shear_max) * points[:, 1]
    points[:, 0] *= rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
    points[:, 1] *= rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
    fraction = cfg.downsample_mean + cfg.downsample_width * (rng.random() - 0.5)
    return downsample_word(points, fraction, cfg.drop_prob, rng)


class PenDataset(Dataset):
    """Serves (stroke tokens, char context, shifted targets) triples.

    Holds the word bank once and materializes each combo on the fly, so memory
    stays proportional to the bank rather than to train_size.
    """

    def __init__(self, bank_points, bank_texts, combos, stroke_tok, char_tok, cfg,
                 augment=True, name=""):
        self.bank_points = bank_points
        self.bank_texts = bank_texts
        self.combos = combos
        self.stroke_tok = stroke_tok
        self.char_tok = char_tok
        self.cfg = cfg
        self.augment = augment
        self.name = name
        self._counter = 0  # varies augmentation when an index repeats

    def __len__(self):
        return len(self.combos)

    def text_for(self, idx):
        return " ".join(self.bank_texts[i] for i in self.combos[idx])

    def encode_points(self, word_points):
        """List of per-word point arrays -> 1D stroke-token array."""
        offsets = [word_to_offsets(w, word_points[i - 1] if i > 0 else None)
                   for i, w in enumerate(word_points)]
        return self.stroke_tok.encode_words(offsets)

    def __getitem__(self, idx):
        words = [self.bank_points[i] for i in self.combos[idx]]
        if self.augment:
            rng = np.random.default_rng([self.cfg.seed, idx, self._counter])
            self._counter = (self._counter + 1) % 100_000
            words = [augment_word(w, self.cfg, rng) for w in words]
        tokens = self.encode_points(words)

        st, L = self.stroke_tok, min(len(tokens), self.cfg.max_seq_length - 1)
        x = torch.full((self.cfg.max_seq_length,), st.PAD, dtype=torch.long)
        y = torch.full((self.cfg.max_seq_length,), IGNORE_INDEX, dtype=torch.long)
        x[:L] = torch.from_numpy(tokens[:L])
        x[L] = st.END
        y[:L] = x[1:L + 1]  # next-token targets, ending with END; padding is ignored

        c = torch.from_numpy(self.char_tok.encode(self.text_for(idx), self.cfg.max_text_length))
        return x, c, y


def create_datasets(cfg):
    """Load the word bank, split it, and build train/test combo datasets."""
    examples = load_examples(cfg.dataset)
    bank_points = [e["points"] for e in examples]
    bank_texts = [e["text"] for e in examples]

    # The character vocabulary comes from the data itself.
    alphabet = " " + "".join(sorted(set("".join(bank_texts)) - {" "}))
    char_tok = CharTokenizer(alphabet)
    stroke_tok = StrokeTokenizer()

    # Hold out whole words (not just combos) so test examples are truly unseen.
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(examples))
    n_test = min(1000, max(10, int(0.05 * len(examples))))
    train_ix, test_ix = perm[:-n_test], perm[-n_test:]

    def build(ix, n_combos, augment, name):
        combos = [ix[c] for c in make_combos(len(ix), n_combos, cfg.num_words, rng)]
        return PenDataset(bank_points, bank_texts, combos, stroke_tok, char_tok,
                          cfg, augment=augment and cfg.augment, name=name)

    train_dataset = build(train_ix, cfg.train_size, True, "train")
    test_dataset = build(test_ix, cfg.test_size, True, "test")
    print(f"Word bank: {len(train_ix)} train / {len(test_ix)} test words; "
          f"{len(train_dataset)} train / {len(test_dataset)} test combos; "
          f"alphabet ({len(alphabet)} chars): {alphabet!r}")
    return train_dataset, test_dataset, stroke_tok, char_tok


class InfiniteDataLoader:
    """A DataLoader that never runs out (random sampling with replacement)."""

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
