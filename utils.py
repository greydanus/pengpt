"""Per-dataset presets, loaders, and special tools.

pengpt.py is the algorithm; this file is everything that is specific to one
dataset. PRESETS holds the settled training settings per corpus, and the CLI
holds the tools that build each corpus in the first place:

    python utils.py download --out_dir qd_raw          # fetch Quick, Draw!
    python utils.py convert --quickdraw cat.ndjson --out data/cats.json
    python utils.py rank --raw_dir qd_raw --out data/quickdraw_top25.jsonl
    python utils.py icons --raw_dir data/raw --out data/icons.jsonl

The quality-ranking tools need `transformers` and `Pillow`; the icons tool
needs `svgelements`. Both are imported lazily so training never requires them.
"""

import argparse
import collections
import glob
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pengpt import INK_HEIGHT


# ---------------------------------------------------------------------------
# Presets: the settled `python pengpt.py train` settings per dataset.
#
# All four use one algorithm; only lengths, augmentation, and model size move.
# The drawing corpora share --max_words 1 (each example is one trajectory, so
# the WORD separator is inert) and --augment general (no shear: an italic
# slant presumes a baseline and is a distortion on a sketch). Sequence and
# text lengths sit near each corpus's p99 -- leaving them at the cursive
# defaults spends most of every sequence on padding.
# ---------------------------------------------------------------------------

PRESETS = {
    # Bundled cursive handwriting from collect.html; the config defaults were
    # ablated on this corpus, so the preset only names it.
    "cursive": dict(
        dataset="data/bigbank_3500.json.zip",
        pen_pos_bands=0,  # A/B'd Aug 2026: position features buy 0.02 nats of loss but cost spelling
        out_dir="out/cursive",
    ),
    # quickdraw_balanced_fixed = quickdraw_balanced after the cumsum-corruption
    # repair; the unrepaired file has 24% of drawings as diagonal staircases.
    # Repaired token stats: p50 39 / p99 136, so a 192 block covers 99.6%.
    "quickdraw": dict(
        dataset="data/quickdraw_balanced_fixed.jsonl.gz",
        max_words=1, augment="general",
        max_text_length=24, max_seq_length=192,
        n_layer=6, n_embd=128, learning_rate=1e-3, batch_size=32,
        out_dir="out/quickdraw",
    ),
    # Six stroke-native icon sets aggregated by `python utils.py icons`.
    # Designer geometry: the finer grid keeps small icon detail (0.020 is too
    # coarse for it), and tremor + rotation bridge ruler-perfect lines toward
    # human ink. Merged token stats: p50 33 / p99 74 -> block 128.
    "icons": dict(
        dataset="data/icons.jsonl",
        max_words=1, augment="general",
        grid=0.012, max_text_length=36, max_seq_length=128,
        tremor=0.004, rotate=2.0,
        n_layer=4, n_embd=96, learning_rate=1e-3, batch_size=32,
        out_dir="out/icons",
    ),
}


# ---------------------------------------------------------------------------
# Converting external data to the pengpt format
#
# Target format is a JSON list (or JSON Lines) of
# {"text": str, "points": [[x, y, pen], ...]} with y growing downward, the
# baseline near y = 0, and ink normalized to INK_HEIGHT.
# ---------------------------------------------------------------------------


def to_absolute(points):
    """Return absolute (x, y, pen), converting from deltas when they look like it.

    Delta-encoded ink has near-zero mean displacement and a tiny coordinate
    range; absolute ink spans the writing area. The ratio separates them by
    orders of magnitude, so the test does not need a tuned threshold.
    """
    points = np.asarray(points, dtype=float).copy()
    span = points[:, :2].max(0) - points[:, :2].min(0)
    step = np.abs(np.diff(points[:, :2], axis=0)).mean() if len(points) > 1 else 0.0
    if step > 0 and span.max() < 12 * step:
        points[:, :2] = np.cumsum(points[:, :2], axis=0)
    return points


def normalize(points, pen_down_is=1, height_quantile=0.95, absolute=False):
    """Scale to the bundled data's conventions: x starts at 0, baseline at y = 0.

    Ink is scaled to INK_HEIGHT because token cost per word is proportional to
    ink size: a corpus twice as large costs twice the sequence length for the
    same shapes, and the tokenizer's grid is a fixed distance.

    Pass absolute=True when the source format is known to store absolute
    coordinates. The delta autodetection in to_absolute reads sparse absolute
    drawings -- a square is five points after RDP simplification, so its span is
    small relative to its mean step -- as delta-encoded, and cumsum turns them
    into a diagonal staircase. That corrupted 24% of a filtered Quick, Draw!
    corpus before it was caught, concentrated in exactly the categories whose
    drawings are few straight lines.
    """
    points = np.asarray(points, dtype=float).copy() if absolute else to_absolute(points)
    if pen_down_is != 1:
        points[:, 2] = 1.0 - points[:, 2]
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        down = points
    height = (np.quantile(down[:, 1], height_quantile)
              - np.quantile(down[:, 1], 1 - height_quantile))
    points[:, :2] *= INK_HEIGHT / max(height, 1e-6)
    points[:, 0] -= points[0, 0]
    points[:, 1] -= np.quantile(points[points[:, 2] == 1][:, 1], 0.95)
    return points


def convert_quickdraw(path, max_items=None, categories=None):
    """Google Quick, Draw! simplified ndjson -> pengpt format.

    Each line holds one drawing as a list of strokes, each stroke a pair of
    coordinate arrays. Pen lifts are implicit in the stroke boundaries, which is
    already how pengpt reads a trajectory, so the conversion is mostly a matter
    of flattening and inserting a lift marker between strokes.

    The label is the category, so a model trained on this draws a named object
    rather than a word. Train with the quickdraw preset.
    """
    examples = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if categories and item.get("word") not in categories:
            continue
        points = []
        for stroke in item["drawing"]:
            xs, ys = stroke[0], stroke[1]
            for x, y in zip(xs, ys):
                points.append([float(x), float(y), 1.0])
            points.append([float(xs[-1]), float(ys[-1]), 0.0])
        if len(points) < 4:
            continue
        examples.append({
            "text": str(item.get("word", "")),
            "points": normalize(np.array(points), absolute=True).round(4).tolist(),
        })
        if max_items and len(examples) >= max_items:
            break
    return examples


# ---------------------------------------------------------------------------
# Downloading Quick, Draw!
# ---------------------------------------------------------------------------

QD_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
QD_LIST = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"


def _get(url, dest=None):
    """Fetch a URL, falling back to curl where Python's SSL roots are missing."""
    try:
        if dest:
            urllib.request.urlretrieve(url, dest)
            return None
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode()
    except Exception:
        import subprocess
        cmd = ["curl", "-fsSL", url] + (["-o", dest] if dest else [])
        out = subprocess.run(cmd, capture_output=not dest, check=True)
        return None if dest else out.stdout.decode()


def quickdraw_categories():
    return [line.strip() for line in _get(QD_LIST).splitlines() if line.strip()]


def fetch_category(name, out_dir, retries=3):
    """Download one category, leaving an existing complete file alone."""
    path = os.path.join(out_dir, f"{name}.ndjson")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return name, os.path.getsize(path), "cached"
    url = QD_BASE + urllib.parse.quote(name) + ".ndjson"
    partial = path + ".part"
    for attempt in range(retries):
        try:
            _get(url, partial)
            os.replace(partial, path)          # only a complete file gets the real name
            return name, os.path.getsize(path), "ok"
        except Exception as e:
            if attempt == retries - 1:
                if os.path.exists(partial):
                    os.remove(partial)
                return name, 0, f"failed: {e}"
    return name, 0, "failed"


def download_command(args):
    os.makedirs(args.out_dir, exist_ok=True)
    names = ([c.strip() for c in args.categories.split(",") if c.strip()]
             or quickdraw_categories())
    print(f"{len(names)} categories -> {args.out_dir}")

    done = failed = total = 0
    with ThreadPoolExecutor(args.threads) as pool:
        for name, size, status in pool.map(lambda n: fetch_category(n, args.out_dir), names):
            done += 1
            total += size
            if status.startswith("failed"):
                failed += 1
                print(f"[{done}/{len(names)}] {name:28s} {status}", flush=True)
            elif done % 25 == 0 or done == len(names):
                print(f"[{done}/{len(names)}] {total / 1e9:5.1f} GB", flush=True)

    print(f"\n{done - failed} files, {total / 1e9:.1f} GB in {args.out_dir}")
    if failed:
        print(f"{failed} failed; run again to retry just those")


# ---------------------------------------------------------------------------
# Ranking drawings by quality, so a crowd-sourced corpus can be filtered
#
# Quick, Draw! contains careful cats next to single scribbles, and keeping only
# the best quarter means ranking. Ranking needs *relative* judgements: asked to
# rate drawings in isolation a judge puts almost everything in the middle,
# leaving no resolution in the top tail where the filtering decision happens.
#
#     1. Render a sample of drawings and embed them with a vision model.
#     2. Have a judge order that sample, and fit Bradley-Terry to the resulting
#        comparisons, turning wins and losses into a latent quality score.
#     3. Fit a linear probe from embedding to score. Embedding the rest of the
#        corpus is cheap and needs no further judgements.
#
# Judging is O(sample) rather than O(corpus), which is what makes this
# affordable at 50M drawings. Filtering is per class so class balance survives.
#
# A calibrated probe ships in data/quickdraw_probe.npz, fitted on 210
# hand-judged drawings. Held out, the quarter it keeps is 83% good-or-better
# against a 48% base rate with none of the judged junk surviving: a reliable
# coarse filter rather than a fine ranking.
#
# Three things were tried and dropped, recorded here so they are not retried:
#
#   - Quick, Draw!'s own `recognized` flag as a training target. It measures
#     whether a classifier saw the expected category, not whether the drawing
#     is good, and agrees with judged quality at only +0.10.
#   - Hand-built stroke geometry (stroke count, ink ratio, coverage) instead
#     of an embedding: +0.13 against judged quality where CLIP reaches +0.55.
#   - A probe per class rather than one shared probe: too few examples per
#     class for 512 dimensions.
# ---------------------------------------------------------------------------

JUDGE_RUBRIC = """Order these drawings from best to worst as training examples.

Reward: the subject is recognizable, its parts are present and connected (a dog
with legs and ears, a car with wheels and windows), and the strokes are
deliberate.

Penalize, hardest first:
  - any drawing with letters or words written on it, however neat the rest is
  - scribbles, and drawings that are one line or a few disconnected fragments
  - missing essential parts, so the subject is only guessable from the label

Words on the canvas need to be called out explicitly like this. A vision
embedding ranks such drawings anywhere from the 22nd to the 76th percentile,
because a cat with "MEOW" written over it still looks largely like a cat, and a
stroke-geometry rule that catches them flags a fifth of clean drawings too. The
judge is the only reliable filter for writing, so the rubric has to ask.
"""


def render(points, px=64, linewidth=2, supersample=8, pad=2):
    """Draw one trajectory as a square image, ready to embed.

    Small and thin wins, which is not obvious: 64px with fine strokes ranks
    +0.59 against judged quality, where the same drawings rendered with thick
    strokes score +0.19. Heavy strokes merge a dog's legs and fill in a cat's
    face, destroying the detail the ranking depends on.

    Antialiasing does matter, so this draws large and downsamples rather than
    rasterizing directly at 64px, which costs ranking quality (+0.47).
    """
    from PIL import Image, ImageDraw

    points = np.asarray(points, dtype=float)
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        return Image.new("RGB", (px, px), "white")

    big = px * supersample
    margin = pad * supersample
    x0, y0 = down[:, 0].min(), down[:, 1].min()
    scale = (big - 2 * margin) / max(np.ptp(down[:, 0]), np.ptp(down[:, 1]), 1e-6)

    image = Image.new("L", (big, big), 255)
    draw = ImageDraw.Draw(image)
    for chunk in np.split(points, np.flatnonzero(points[:, 2] == 0) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            draw.line([(margin + (x - x0) * scale, margin + (y - y0) * scale)
                       for x, y in chunk[:, :2]],
                      fill=0, width=int(linewidth * supersample))
    return image.resize((px, px), Image.LANCZOS).convert("RGB")


_WORKER = {}


def _default_workers(device):
    """Workers only where the GPU actually outruns one CPU core.

    Measured: an H100 runs the model at ~25k images/s while a core renders and
    normalizes at ~700, so it needs feeding. MPS runs it at ~400, which one core
    already keeps up with, and there the pool costs more in start-up and
    pickling than it returns -- 260 images/s becomes 94.
    """
    if device != "cuda":
        return 1
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def _init_worker(name, px, linewidth):
    from transformers import AutoImageProcessor
    _WORKER["processor"] = AutoImageProcessor.from_pretrained(name)
    _WORKER["px"], _WORKER["linewidth"] = px, linewidth


def _prepare_worker(chunk):
    images = [render(p, _WORKER["px"], _WORKER["linewidth"]) for p in chunk]
    return _WORKER["processor"](images=images, return_tensors="pt")["pixel_values"]


class Embedder:
    """Vision embeddings for rendered drawings.

    CLIP is the default. Measured against hand-judged tiers it reaches spearman
    +0.59 where dinov2-small reaches +0.44 and dinov2-base +0.30, which is what
    you would expect: CLIP's training data is full of line art and clip art,
    while DINOv2's is photographs.
    """

    def __init__(self, name="openai/clip-vit-base-patch32", device=None, px=64,
                 linewidth=1.0, fp16=False, workers=0):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.name = name
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.is_clip = "clip" in name.lower()
        self.fp16 = fp16 and self.device != "cpu"
        if self.fp16:
            self.model = self.model.half()
        self.px = px
        self.linewidth = linewidth
        self.workers = workers if workers else _default_workers(self.device)
        self._pool = None

    def embed(self, points_list, batch_size=256, verbose=False):
        """Embed drawings, preparing batches on worker processes.

        On a fast GPU the bottleneck is entirely CPU-side: workers render and
        normalize so the GPU sees a queue rather than one core feeding it.
        """
        out = []
        with self.torch.inference_mode():
            for pixels, n in self._batches(points_list, batch_size):
                pixels = pixels.to(self.device, non_blocking=True)
                if self.fp16:
                    pixels = pixels.half()
                feats = (self.model.get_image_features(pixel_values=pixels)
                         if self.is_clip else self.model(pixel_values=pixels))
                # transformers 5 returns an output object where 4 returned a
                # tensor, and non-CLIP models pool separately
                if not hasattr(feats, "float"):
                    feats = (feats.image_embeds if getattr(feats, "image_embeds", None)
                             is not None else feats.pooler_output)
                out.append(feats.float().cpu().numpy())
                if verbose:
                    print(f"  embedded {n}", flush=True)
        return np.concatenate(out) if out else np.zeros((0, 1))

    def _batches(self, points_list, batch_size):
        """Yield (pixel_values, count) batches, rendered and normalized ahead.

        The pool is created once and reused, since a caller scoring a corpus
        calls embed() per chunk and paying process start-up each time would
        cost more than the parallelism returns.
        """
        chunks = [points_list[i:i + batch_size]
                  for i in range(0, len(points_list), batch_size)]
        if self.workers <= 1 or len(chunks) <= 1:
            for chunk in chunks:
                yield self._prepare(chunk), len(chunk)
            return
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(
                self.workers, initializer=_init_worker,
                initargs=(self.name, self.px, self.linewidth))
        for pixels in self._pool.map(_prepare_worker, chunks, chunksize=1):
            yield pixels, len(pixels)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _prepare(self, chunk):
        images = [render(p, self.px, self.linewidth) for p in chunk]
        return self.processor(images=images, return_tensors="pt")["pixel_values"]


def bradley_terry(n_items, comparisons, iters=300, reg=1e-2, lr=0.1):
    """Latent quality from pairwise wins.

    comparisons: (winner, loser) index pairs. Returns one standardized score per
    item, higher is better. The regularizer keeps items with few or one-sided
    comparisons from running away.
    """
    wins = np.zeros(n_items)
    played = np.zeros((n_items, n_items))
    for w, l in comparisons:
        wins[w] += 1
        played[w, l] += 1
        played[l, w] += 1

    scores = np.zeros(n_items)
    for _ in range(iters):
        expected = np.zeros(n_items)
        for i in np.flatnonzero(played.any(1)):
            partners = np.flatnonzero(played[i])
            p = 1.0 / (1.0 + np.exp(np.clip(scores[partners] - scores[i], -30, 30)))
            expected[i] = (played[i, partners] * p).sum()
        scores += lr * (wins - expected - reg * scores)
    return (scores - scores.mean()) / (scores.std() + 1e-9)


class LinearProbe:
    """Ridge regression from embedding to judged quality."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = self.mean = self.std = None

    def fit(self, X, scores):
        X = np.asarray(X, dtype=float)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.weights = np.linalg.solve(A, Z.T @ np.asarray(scores, dtype=float))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        return Z @ self.weights

    def agreement(self, X, scores):
        """Spearman against held-out judgements; the number that matters."""
        return spearman(self.score(X), np.asarray(scores))


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_probe(path="data/quickdraw_probe.npz"):
    """A probe already calibrated on judged Quick, Draw! drawings.

    Fitted on 210 drawings ordered into quality tiers by hand, embedded with
    CLIP at 64px. Re-fit rather than reuse this if you change the embedder or
    the render settings, which the file records.
    """
    data = np.load(path)
    probe = LinearProbe(alpha=float(data["alpha"]))
    probe.weights, probe.mean, probe.std = data["weights"], data["mean"], data["std"]
    return probe


def select_per_class(scores, labels, fraction=0.25):
    """Indices of the best `fraction` within each class, so balance survives.

    Filtering a whole corpus at once would keep whichever classes the scorer
    happens to rate highly and drop the rest.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    keep = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        order = idx[np.argsort(scores[idx])[::-1]]
        keep.append(order[:max(1, int(round(fraction * len(order))))])
    return np.sort(np.concatenate(keep))


RANK_CHUNK = 20_000


def iter_quickdraw(path, limit=None):
    """Yield (points, label) from a Quick, Draw! ndjson, skipping truncated lines."""
    label = os.path.basename(path).split(".")[0]
    seen = 0
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.endswith("}"):
                continue
            try:
                drawing = json.loads(line)["drawing"]
            except (ValueError, KeyError):
                continue
            points = []
            for stroke in drawing:
                xs, ys = stroke[0], stroke[1]
                if not xs:
                    continue
                points.extend([float(x), float(y), 1.0] for x, y in zip(xs, ys))
                points.append([float(xs[-1]), float(ys[-1]), 0.0])
            if len(points) >= 4:
                yield np.array(points), label
                seen += 1
                if limit and seen >= limit:
                    return


def score_category(path, embedder, probe, fraction, limit, batch_size):
    """Best `fraction` of one category, as pengpt example dicts.

    Chunked so a large category never loads at once: each chunk keeps its own
    best share, which approximates a global top-fraction closely enough for
    filtering and bounds memory at RANK_CHUNK drawings.
    """
    kept, seen, chunk = [], 0, []
    label = None

    def flush(chunk):
        if not chunk:
            return []
        scores = probe.score(embedder.embed(chunk, batch_size=batch_size))
        n = max(1, int(round(fraction * len(chunk))))
        return [chunk[i] for i in np.argsort(scores)[::-1][:n]]

    for points, label in iter_quickdraw(path, limit):
        chunk.append(points)
        seen += 1
        if len(chunk) >= RANK_CHUNK:
            kept.extend(flush(chunk))
            chunk = []
    kept.extend(flush(chunk))
    examples = [{"text": label, "points": normalize(p, absolute=True).round(4).tolist()}
                for p in kept]
    return examples, seen


def rank_command(args):
    """Score Quick, Draw! and keep the best fraction of each class.

    Built to run over the whole 50M-drawing corpus without holding it: output
    streams to JSON Lines as each category finishes, categories are read in
    chunks, and finished categories are recorded so --resume picks up after a
    crash rather than repeating an hour of paid GPU time.
    """
    settings = np.load(args.probe)
    embedder = Embedder(name=str(settings["embedder"]), px=int(settings["px"]),
                        linewidth=float(settings["linewidth"]), fp16=args.fp16,
                        workers=args.workers)
    print(f"{embedder.device}, fp16={embedder.fp16}, {embedder.workers} render workers")
    probe = load_probe(args.probe)

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.ndjson")))
    if not paths:
        raise SystemExit(f"no .ndjson files in {args.raw_dir}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out):
        if args.overwrite:
            open(args.out, "w").close()
        elif args.resume:
            with open(args.out) as f:
                done = {json.loads(line)["text"] for line in f if line.strip()}
            print(f"resuming: {len(done)} categories already done")
        else:
            raise SystemExit(
                f"{args.out} already exists. Pass --resume to continue it, or "
                f"--overwrite to start again; appending to it silently would "
                f"duplicate every category it already holds.")

    todo = [p for p in paths
            if os.path.basename(p).split(".")[0] not in done]
    print(f"{len(todo)} of {len(paths)} categories to score, "
          f"keeping the best {args.fraction:.0%} of each")

    kept_total, seen_total, t0 = 0, 0, time.time()
    with open(args.out, "a") as out:
        for n, path in enumerate(todo, 1):
            label = os.path.basename(path).split(".")[0]
            examples, seen = score_category(path, embedder, probe, args.fraction,
                                            args.limit_per_class, args.batch_size)
            for example in examples:
                out.write(json.dumps(example) + "\n")
            out.flush()
            os.fsync(out.fileno())
            kept_total += len(examples)
            seen_total += seen
            elapsed = time.time() - t0
            rate = seen_total / elapsed
            eta = (len(todo) - n) * (seen_total / n) / rate / 60
            print(f"[{n}/{len(todo)}] {label:22s} {seen:7,} -> {len(examples):6,}"
                  f" | {rate:5.0f}/s | {elapsed / 60:5.1f} min elapsed"
                  f" | eta {eta:5.0f} min", flush=True)

    print(f"\nkept {kept_total:,} of {seen_total:,} scored "
          f"({kept_total / max(seen_total, 1):.0%}) in "
          f"{(time.time() - t0) / 60:.1f} min")
    if done:
        print(f"plus {len(done)} categories from an earlier run")
    print(f"-> {args.out}")
    embedder.close()


# ---------------------------------------------------------------------------
# Aggregating stroke-native icon sets into one pen corpus
#
# Six open icon sets ship SVGs whose paths are literal pen centerlines
# (fill="none" stroke=...): Lucide, Tabler outline, Feather, Iconoir regular,
# Heroicons outline, and Akar. Phosphor and RemixIcon flatten their strokes to
# filled outlines and are permanently unsuitable -- tracing a filled outline
# draws the shape's boundary twice, which no pen does.
#
# Beyond parsing, three things make the output *sketchable* rather than merely
# vector:
#
# - Per-element strictness: even inside stroke-native sets, the odd element
#   carries a visible fill. Any such icon is rejected whole rather than
#   partially converted, since a half-drawn icon has a label its ink no longer
#   matches.
# - Sketchability limits: an icon with 30+ strokes or a cloud of sub-cell
#   dashes is a rendering, not a sketch anyone would pen. Caps on stroke count,
#   tiny-stroke count, and token cost keep every example drawable in one
#   sitting. Rejected icons are written to a preview sheet, never silently
#   dropped.
# - Pen-travel ordering: SVG export order jumps around the canvas; a person
#   sketches nearby strokes consecutively. Strokes are greedily reordered (and
#   flipped) to minimize pen-up travel, which is both more human and cheaper --
#   pen-up moves cost real tokens in ScribeTokens.
#
# Sets sharing ancestry draw many icons identically (Lucide forked Feather), so
# same-label icons are deduped on geometry: near-identical renditions collapse
# to one, while genuinely different depictions of "camera" survive as style
# variety, the way Quick, Draw! keeps many drawings per category.
# ---------------------------------------------------------------------------

# (set name, subdir of raw_dir, license) -- order is dedupe priority: when two
# sets draw a label identically, the earlier set's rendition is kept.
ICON_SETS = [
    ("lucide", "lucide/icons", "ISC"),
    ("tabler", "tabler-icons/icons/outline", "MIT"),
    ("iconoir", "iconoir/icons/regular", "MIT"),
    ("heroicons", "heroicons/optimized/24/outline", "MIT"),
    ("akar", "akar-icons/src/svg", "MIT"),
    ("feather", "feather/icons", "MIT"),
]

MAX_STROKES = 24          # a person will not pen 30 strokes for one icon
# Calibrated against known icons: gear teeth and sun rays run ~8 tiny strokes
# and are classic pen sketches; dotted-circle glyphs start at 12 and are
# dot-by-dot tedium. 10 splits the two populations.
MAX_TINY = 10             # dashes/dots shorter than ~2 grid cells
TINY_LEN = 0.025          # in normalized units; INK_HEIGHT is 0.22
MAX_POINTS = 2200         # caps token cost; ~p99 of the pool at step 0.25
SAMPLE_STEP = 0.25        # SVG user units between samples (24px viewBox)


def _visible(color):
    """True when an svgelements color paints something."""
    return color is not None and color.value is not None and color.alpha != 0


def svg_to_strokes(path):
    """One SVG -> list of (N, 2) stroke arrays, or a rejection reason string.

    Paint is checked per element: a Shape with a visible fill means the icon is
    not purely stroke-native, and the whole file is rejected rather than
    converted into ink that no longer matches its label.
    """
    from svgelements import SVG, Path, Shape

    try:
        svg = SVG.parse(path)
    except Exception:
        return "unparseable"
    strokes = []
    for element in svg.elements():
        if not isinstance(element, Shape):
            continue
        if _visible(getattr(element, "fill", None)):
            return "has_fill"
        if not _visible(getattr(element, "stroke", None)):
            continue                      # invisible helper geometry
        shape = Path(element)
        shape.reify()
        for sub in shape.as_subpaths():
            sub = Path(sub)
            length = sub.length(error=1e-4)
            if length < 1e-6:
                continue
            n = max(2, int(np.ceil(length / SAMPLE_STEP)))
            strokes.append(np.array(sub.npoint(np.linspace(0, 1, n)), dtype=float))
    if not strokes:
        return "no_strokes"
    return strokes


def order_strokes(strokes):
    """Greedy pen-travel ordering: nearest next stroke, either end first.

    SVG export order can hop across the canvas; a person sketches what is near
    the pen. Start from the stroke closest to the top-left (where most people
    begin) and repeatedly take the unvisited stroke whose nearer endpoint is
    closest to the current pen position, flipping it when its far end is the
    near one. Pen-up travel is real token cost in ScribeTokens, so this is
    thrift as well as realism.
    """
    remaining = list(range(len(strokes)))
    starts = np.array([s[0] for s in strokes])
    ends = np.array([s[-1] for s in strokes])
    corner = np.vstack([starts, ends]).min(axis=0)
    first = min(remaining, key=lambda i: np.hypot(*(starts[i] - corner)))
    out = [strokes[first]]
    remaining.remove(first)
    pen = out[-1][-1]
    while remaining:
        d_start = [np.hypot(*(starts[i] - pen)) for i in remaining]
        d_end = [np.hypot(*(ends[i] - pen)) for i in remaining]
        k = int(np.argmin(np.minimum(d_start, d_end)))
        i = remaining.pop(k)
        stroke = strokes[i] if d_start[k] <= d_end[k] else strokes[i][::-1]
        out.append(stroke)
        pen = stroke[-1]
    return out


def strokes_to_points(strokes):
    """Ordered strokes -> normalized (N, 3) with pen-lift marker rows."""
    rows = []
    for s in strokes:
        for x, y in s:
            rows.append([x, y, 1.0])
        rows.append([s[-1][0], s[-1][1], 0.0])
    return normalize(np.array(rows), absolute=True).round(4)


def sketchability(points):
    """Rejection reason for an icon nobody would pen, or None if fine."""
    down = points[points[:, 2] == 1]
    lifts = int((points[:, 2] == 0).sum())
    if lifts > MAX_STROKES:
        return "too_many_strokes"
    if len(points) > MAX_POINTS:
        return "too_much_ink"
    if len(down) < 4:
        return "degenerate"
    tiny = 0
    for chunk in np.split(points, np.flatnonzero(points[:, 2] == 0) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            arc = np.hypot(*np.diff(chunk[:, :2], axis=0).T).sum()
            if arc < TINY_LEN:
                tiny += 1
    if tiny > MAX_TINY:
        return "too_many_tiny_strokes"
    return None


def icon_label(set_name, filename):
    """Filename -> caption. Tabler brand icons read better as "<name> logo"."""
    name = os.path.splitext(os.path.basename(filename))[0]
    words = name.replace("-", " ").replace("_", " ").strip()
    if set_name == "tabler" and words.startswith("brand "):
        words = words[len("brand "):] + " logo"
    return words


def _signature(points, n=32):
    """Fixed-length shape signature for near-duplicate detection.

    Pen-down points resampled by cumulative arc length to n points, scaled to a
    unit box. Two renditions of one icon from a shared ancestor land within a
    few percent of each other; independent drawings of the same label do not.
    """
    down = points[points[:, 2] == 1][:, :2]
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(down, axis=0).T))]
    t = np.linspace(0, d[-1], n)
    sig = np.column_stack([np.interp(t, d, down[:, 0]),
                           np.interp(t, d, down[:, 1])])
    sig -= sig.min(axis=0)
    return sig / max(sig.max(), 1e-9)


def dedupe_icons(examples, threshold=0.035):
    """Drop later same-label icons whose geometry matches an earlier one."""
    by_label = collections.defaultdict(list)
    kept, dropped = [], 0
    for ex in examples:
        sig = _signature(np.array(ex["points"]))
        dup = any(np.abs(sig - other).mean() < threshold
                  for other in by_label[ex["text"]])
        if dup:
            dropped += 1
            continue
        by_label[ex["text"]].append(sig)
        kept.append(ex)
    return kept, dropped


def icon_preview(examples, out_path, n=48, title=""):
    import matplotlib.pyplot as plt
    from pengpt import draw

    rng = np.random.default_rng(0)
    picks = rng.choice(len(examples), size=min(n, len(examples)), replace=False)
    cols = 8
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.3 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, idx in enumerate(picks):
        ex = examples[idx]
        ax = axes[k // cols][k % cols]
        draw(ax, np.array(ex["points"]), color="k", linewidth=1.0)
        ax.set_title(f'{ex["text"][:26]}\n({ex["meta"]["set"]})', fontsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def icons_command(args):
    examples, rejects = [], []
    reasons = collections.Counter()
    for set_name, subdir, license_ in ICON_SETS:
        files = sorted(glob.glob(os.path.join(args.raw_dir, subdir, "*.svg")))
        n_ok = 0
        for f in files:
            strokes = svg_to_strokes(f)
            if isinstance(strokes, str):
                reasons[f"{set_name}:{strokes}"] += 1
                continue
            if not args.no_reorder:
                strokes = order_strokes(strokes)
            points = strokes_to_points(strokes)
            reason = sketchability(points)
            item = {"text": icon_label(set_name, f),
                    "points": points.tolist(),
                    "meta": {"set": set_name, "license": license_}}
            if reason:
                reasons[f"{set_name}:{reason}"] += 1
                rejects.append(item)
                continue
            examples.append(item)
            n_ok += 1
        print(f"{set_name:10s} {n_ok:5d} kept of {len(files):5d}")

    examples, n_dupes = dedupe_icons(examples)
    print(f"\ndeduped {n_dupes} near-identical same-label renditions")
    if reasons:
        print("rejections:")
        for k, v in reasons.most_common():
            print(f"  {k:36s} {v}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    labels = {ex["text"] for ex in examples}
    print(f"\nwrote {len(examples)} icons ({len(labels)} distinct labels) "
          f"to {args.out}")

    base = os.path.splitext(args.out)[0]
    icon_preview(examples, base + "_preview.png",
                 title=f"accepted ({len(examples)})")
    if rejects:
        icon_preview(rejects, base + "_rejected.png",
                     title=f"rejected ({len(rejects)}) -- verify nothing good is lost")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def convert_command(args):
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    examples = convert_quickdraw(args.quickdraw, args.max_items, cats or None)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(examples, f)
    print(f"Wrote {len(examples)} examples to {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="pengpt dataset tools")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download the Quick, Draw! corpus")
    dl.add_argument("--out_dir", type=str, default="qd_raw")
    dl.add_argument("--categories", type=str, default="",
                    help="comma-separated subset; default is all 345")
    dl.add_argument("--threads", type=int, default=16)

    cv = sub.add_parser("convert", help="convert a Quick, Draw! ndjson to pengpt format")
    cv.add_argument("--quickdraw", type=str, required=True,
                    help="Quick, Draw! simplified .ndjson (one category per file)")
    cv.add_argument("--categories", type=str, default="",
                    help="comma-separated categories to keep")
    cv.add_argument("--max_items", type=int, default=None)
    cv.add_argument("--out", type=str, default="data/converted.json")

    rk = sub.add_parser("rank", help="filter Quick, Draw! to its best drawings")
    rk.add_argument("--raw_dir", type=str, required=True,
                    help="directory of Quick, Draw! simplified .ndjson files")
    rk.add_argument("--out", type=str, default="data/quickdraw_top25.jsonl")
    rk.add_argument("--fraction", type=float, default=0.25)
    rk.add_argument("--limit_per_class", type=int, default=None)
    rk.add_argument("--probe", type=str, default="data/quickdraw_probe.npz")
    rk.add_argument("--batch_size", type=int, default=256)
    rk.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                    help="half precision embedding; 1.7x faster, same ranking")
    rk.add_argument("--workers", type=int, default=0,
                    help="processes rendering batches ahead of the GPU; "
                         "0 picks a default from the device")
    rk.add_argument("--resume", action="store_true",
                    help="skip categories already present in the output")
    rk.add_argument("--overwrite", action="store_true",
                    help="discard an existing output file and start again")

    ic = sub.add_parser("icons", help="aggregate stroke-native icon sets into one corpus")
    ic.add_argument("--raw_dir", type=str, default="data/raw")
    ic.add_argument("--out", type=str, default="data/icons.jsonl")
    ic.add_argument("--no_reorder", action="store_true",
                    help="keep SVG file order instead of pen-travel order")

    args = parser.parse_args(argv)
    {"download": download_command, "convert": convert_command,
     "rank": rank_command, "icons": icons_command}[args.command](args)


if __name__ == "__main__":
    main()
