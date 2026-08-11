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

from .tokenizer import ScribeTokenizer, CharTokenizer, learn_merges, _stroke_spans

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
        examples.append({"text": text, "points": points,
                         "source": item.get("meta", {}).get("source", "")})
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
    learns a drafting machine's hand. This bridges toward human ink the same
    way sketch_style.humanize does for the physics corpus: smooth low-frequency
    noise perpendicular to each path -- knots every ~wavelength of arc,
    interpolated, so lines wave the way a freehand line waves rather than
    jittering per point -- plus endpoints extended or trimmed by a few
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


HUMAN_INK = {"sketchy", "quickdraw", "fscoco"}
NO_FLIP = {"icons"}
NO_DROPOUT = {"icons"}


def augment_drawing(points, text, cfg, rng, source=""):
    """Drawing-only augmentations that need the caption or stroke structure.

    These run after prepare_word's geometric jitter. All are off by default
    and belong to drawing corpora: mirrored or stroke-dropped handwriting
    stops matching its transcript, but a scene stays a scene.

    Sources gate what applies: tremor humanizes designer geometry, so human
    ink skips it; icons skip mirroring (letterform labels) and stroke dropout
    (a dropped stroke can be the x in "ticket x").
    """
    if cfg.tremor > 0 and source not in HUMAN_INK:
        # Amplitude varies per drawing: a corpus of one fixed waver is as
        # synthetic as no waver. Sampling U(0.3, 1) x tremor spans careful
        # hands to casual ones without reaching the level that mangles small
        # elements (measured: 2x the default visibly distorts a plus sign).
        points = _tremor(points, rng, rng.uniform(0.3, 1.0) * cfg.tremor)
    if cfg.stroke_dropout > 0 and source not in NO_DROPOUT:
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
    if cfg.hflip > 0 and source not in NO_FLIP and rng.random() < cfg.hflip:
        lowered = text.lower()
        if "left" not in lowered and "right" not in lowered:
            points = points.copy()
            # Reflect within the bounding box, so coordinates keep their range.
            points[:, 0] = points[:, 0].max() + points[:, 0].min() - points[:, 0]
    return points


class PenDataset(Dataset):

    def __init__(self, bank_points, bank_texts, indices, stroke_tok, char_tok, cfg,
                 length, augment=True, name="", seed=0, text_encoder=None,
                 bank_embeds=None, bank_sources=None):
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
        self.text_encoder = text_encoder
        self.bank_embeds = bank_embeds
        self.bank_sources = bank_sources

    def sources(self):
        if self.bank_sources is None:
            return []
        return sorted({self.bank_sources[i] for i in self.indices} - {""})

    def for_source(self, source):
        keep = [i for i in self.indices if self.bank_sources[i] == source]
        return PenDataset(self.bank_points, self.bank_texts, keep,
                          self.stroke_tok, self.char_tok, self.cfg,
                          length=min(self.length, 1000), augment=self.augment,
                          name=f"{self.name}_{source}", seed=self.seed,
                          text_encoder=self.text_encoder,
                          bank_embeds=self.bank_embeds,
                          bank_sources=self.bank_sources)

    def encode_text(self, text):
        if self.text_encoder is not None:
            return self.text_encoder.encode(text, self.cfg.max_text_length)
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
                word = augment_drawing(
                    word, self.bank_texts[i], self.cfg, rng,
                    source=self.bank_sources[i] if self.bank_sources else "")
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

    def bank_word_for(self, idx):
        """The bank index that __getitem__(idx) will draw, without tokenizing.

        Replays pick_words' first two rng draws, which is only well-defined at
        max_words == 1 (one word, no packing loop). Length-bucketed batching
        uses this to group samples of similar size for a fraction of the cost
        of actually encoding them; a test pins the replay to pick_words.
        """
        assert self.cfg.max_words == 1
        rng = np.random.default_rng([self.seed, idx])
        rng.integers(1, min(1, len(self.indices)) + 1)  # pick_words' limit draw
        return int(rng.choice(self.indices, size=1, replace=False)[0])

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
        if self.bank_embeds is not None:
            drop = self.augment and rng.random() < self.cfg.embed_dropout
            if not drop:
                c = c.copy()
                live = c.any(axis=1)
                d = self.text_encoder.clip_dim
                c[live, :d] = self.bank_embeds[self.bank_word_for(idx)]
        return x, torch.from_numpy(c), y

    def text_for(self, idx):
        return self.pick_words(np.random.default_rng([self.seed, idx]))[1]


def filter_holdout(examples, holdout):
    held = {w.strip().lower() for w in holdout.split(",") if w.strip()}
    if not held:
        return examples
    kept = [e for e in examples if not (held & set(e["text"].lower().split()))]
    print(f"Holdout {sorted(held)}: {len(examples) - len(kept)} examples removed")
    return kept


def create_datasets(cfg, merges=None, text_encoder=None):
    examples = load_examples(cfg.dataset, getattr(cfg, 'max_examples', 0) or None)
    bank_embeds = None
    if getattr(cfg, "clip_image_embeds", ""):
        bank_embeds = np.load(cfg.clip_image_embeds)[:len(examples)]
        assert len(bank_embeds) == len(examples), \
            "clip_image_embeds rows do not match the dataset"
    held = {w.strip().lower() for w in getattr(cfg, "holdout", "").split(",")
            if w.strip()}
    if held:
        keep = [i for i, e in enumerate(examples)
                if not (held & set(e["text"].lower().split()))]
        print(f"Holdout {sorted(held)}: {len(examples) - len(keep)} examples removed")
        examples = [examples[i] for i in keep]
        if bank_embeds is not None:
            bank_embeds = bank_embeds[keep]
    check_scale(examples, cfg.grid)
    bank_points = [e["points"] for e in examples]
    bank_texts = [e["text"] for e in examples]

    alphabet = " " + "".join(sorted(set("".join(bank_texts)) - {" "}))
    char_tok = CharTokenizer(alphabet)
    if text_encoder == "clip+char":
        from .textenc import build_text_encoder
        text_encoder = build_text_encoder("clip+char", char_tok=char_tok)

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

    bank_sources = [e["source"] for e in examples]
    if not any(bank_sources):
        bank_sources = None

    def build(ix, n, name, seed):
        return PenDataset(bank_points, bank_texts, ix, stroke_tok, char_tok, cfg,
                          length=n, augment=cfg.augment != "none", name=name,
                          seed=seed, text_encoder=text_encoder,
                          bank_embeds=bank_embeds, bank_sources=bank_sources)

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


class _BucketBatches:
    """Endless batches of dataset indices, grouped by drawing size.

    Random batches are padded to their longest member, and the longest of
    sixteen random scenes is nearly always near the block size -- so almost
    nothing is saved by trimming them. Sorting a shuffled window by point
    count (a cheap proxy for token count) before chunking makes batches
    length-homogeneous, which is what lets the training loop cut most of the
    padding. Sampling stays uniform: every index in a window is used exactly
    once, windows are drawn at random, and batch order is shuffled.
    """

    def __init__(self, dataset, batch_size, window_batches=64, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.window = batch_size * window_batches
        self.rng = np.random.default_rng(seed)
        # Unaugmented token counts, once per bank word. Raw point count is a
        # poor proxy -- BPE compresses drawings unevenly -- and fuzzy sorting
        # leaves batches nearly as padded as random ones. Augmentation moves a
        # length ~15%, which sorting tolerates.
        self.sizes = np.array([len(dataset.stroke_tok.encode_word(p))
                               for p in dataset.bank_points])

    def __iter__(self):
        while True:
            indices = self.rng.integers(0, len(self.dataset), size=self.window)
            proxy = self.sizes[[self.dataset.bank_word_for(i) for i in indices]]
            indices = indices[np.argsort(proxy, kind="stable")]
            batches = [indices[i:i + self.batch_size].tolist()
                       for i in range(0, len(indices), self.batch_size)]
            self.rng.shuffle(batches)
            yield from batches


class BucketedInfiniteLoader:
    """InfiniteDataLoader that groups similar-length drawings per batch."""

    def __init__(self, dataset, batch_size, seed=0, **kwargs):
        self.loader = DataLoader(dataset,
                                 batch_sampler=_BucketBatches(dataset, batch_size,
                                                              seed=seed),
                                 **kwargs)
        self.iterator = iter(self.loader)

    def next(self):
        return next(self.iterator)
