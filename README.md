# pengpt

A minimal, general-purpose transformer for pen strokes. Give it a corpus of pen
trajectories — handwriting, sketches, anything drawn with a stylus or mouse —
and it learns to generate more of them, conditioned on text.

This is a ground-up rewrite of
[cursivetransformer](https://github.com/greydanus/cursivetransformer)
([paper](https://arxiv.org/abs/2504.00051) ·
[blog post](https://greydanus.github.io/2025/03/30/cursive-transformer/)),
rebuilt around a representation that is not specific to one writer or one
dataset.

![hero](static/hero.png)

The whole model lives in two files. `pengpt.py` is the algorithm — tokenizer,
data pipeline, transformer, training loop, sampling — and `utils.py` is
everything specific to one dataset: a preset of settled training settings per
corpus, plus the tools that build each corpus in the first place. The same
core algorithm trains on every dataset; only lengths, augmentation, and model
size move between presets.

## Quality over a training run

![progress](static/progress.png)

The same three prompts, rendered at each eval. Stroke control and slant arrive
almost immediately; spelling follows. All three prompts are correct by step
10,000, about 25 minutes in, and the rest of the run sharpens the penmanship.

## Quickstart

```bash
pip install -e .          # torch, numpy, matplotlib
pip install -e ".[dev]"   # + pytest
pytest tests/             # a few seconds
```

Train — about 45 minutes on an M-series laptop, no GPU rental needed:

```bash
python pengpt.py train --preset cursive
```

The presets are `cursive`, `quickdraw`, and `icons`; each names its
dataset, lengths, augmentation, and model size, and any explicit flag
overrides the preset. Each run owns a folder under `out/`:

```
out/cursive/
  best.pt        lowest test loss so far
  last.pt        the most recent eval, for --resume
  progress/      one grid per eval: 8 prompts across, 4 samples down
  samples/       full-size renders, every fourth eval
```

Progress images use the same prompts and seeds every time, so consecutive
images differ only by the model. Four samples per prompt is what makes them
readable: one sample cannot tell a model ignoring its prompt from a model
having a bad draw.

Add `--wandb --wandb_entity you` for Weights & Biases logging (optional).
Resume with `--resume out/cursive/last.pt` — `last.pt`, not `best.pt`, which
lags whenever test loss has stopped improving. Resuming rebuilds the tokenizer
from the checkpoint rather than from the command line, and says so when a data
flag disagrees: merges and alphabet decide what every token id means, so a
re-derived vocabulary would load the weights against ids they were never
trained on. For a faster run, `--n_layer 3` roughly halves the time and scored
within noise of the default in ablations.

Checkpoints from before the repo was consolidated around `pengpt.py` may have
been trained with features that no longer exist (alternate text encoders,
pen-position inputs); the loader warns and ignores those config keys, and a
checkpoint that actually used them should be retrained.

### Drawings rather than handwriting

```bash
python pengpt.py train --preset quickdraw
```

The preset is the settings a drawing corpus needs: `--max_words 1` because
each sample is a single object, `--augment general` because shear is an italic
slant that presumes a baseline and is a distortion on a sketch, and
`--max_text_length 24 --max_seq_length 192` near the corpus's longest label
and p99 token count — leaving these at the cursive defaults spends most of
every sequence on padding.

`quickdraw_balanced_fixed` is `quickdraw_balanced` after repairing a
delta-detection bug that cumsum'd 24% of drawings into diagonal staircases
(see `normalize` in `utils.py`). At the repaired token statistics
(p50 = 39, p99 = 136), a 192-token block covers 99.6% of drawings.

Loss alone will not tell you whether the prompt is being read — a model can
post a falling loss while drawing the corpus average for every prompt. Measure
it directly: score one real drawing under every candidate label and see where
its true label ranks (1 of N is perfect, (N+1)/2 is chance):

```bash
python pengpt.py rank --checkpoint out/quickdraw/best.pt --per_label 8
```

Generate handwriting from a trained model:

```bash
python pengpt.py sample --checkpoint out/cursive/best.pt --text "The quick brown fox jumps over the lazy dog"
```

If a word comes out misspelled, note its index (`--show_indices`) and regenerate
just that word with `--redo 3,7`.

## How it works

**Tokenization** is
[ScribeTokens](https://arxiv.org/abs/2603.02805). Pen coordinates are quantized
to an integer grid, and motion becomes a walk on that grid: eight compass
directions plus `DOWN` and `UP`. Movement between grid points is decomposed with
Bresenham's line algorithm, so any path is representable by ten base symbols.
Byte pair encoding then merges recurring runs — a common curve becomes a single
token — which is what keeps sequences short.

![roundtrip](static/roundtrip.png)

*Grey is the original, red is decoded from tokens. `grid=0.020` is the default:
reconstruction error stays inside the width of a pen stroke at 99 tokens per word.*

**Model**: a small GPT-style decoder (~410k params at the cursive default,
~1.7M at the quickdraw preset) with cross-attention over the character
embedding of the text prompt, in the makemore/nanoGPT lineage.

The prompt reaches the model only through that cross-attention, never as a class
index, so the same architecture takes a word to write or the name of an object
to draw, and an unseen label still says something through its characters.

One optional input rides along with the token embeddings: Fourier features of
the pen's absolute canvas position, computed from a per-token displacement
table the tokenizer provides (a merged token moves by the sum of its
children). The tokens themselves are relative motion, so without this the
model must integrate the whole walk with attention to know whether two strokes
connect. It is on by default and cheap (one small bias-free linear layer), but
the measured gain is modest — about 0.02 nats per token on scene sketches,
less on short drawings — so `--pen_pos_bands 0` turns it off. A random canvas
offset at training time keeps absolute layout from being memorized.

**Data**: each example is one word, an `(N, 3)` array of `(x, y, pen)`.
Training examples pack random words together until the block is full, so a few
thousand words become effectively unlimited examples and nothing is ever
truncated mid-word.

## Why ScribeTokens

The original used polar `(theta, r)` bins, two tokens per pen point. Measured on
bigbank_3500, ScribeTokens is better on every axis at once:

| | tokens/word | reconstruction error | bits/pen-point | vocab |
|---|---|---|---|---|
| polar bins | 185 | 0.0116 | 13.09 | 525 |
| ScribeTokens | **99** | **0.0061** | **5.94** | **306** |

Trained head to head at matched wall clock, ScribeTokens reached **3.43
bits/pen-point against the polar tokenizer's 7.16** — a 2.1× advantage, while
running *fewer* optimizer steps. BPE is doing real work here: at equal block
sizes, 256 merges scored 1.29 against 2.12 with no merges at all.

Together with a coarser grid and a shorter block, this took a full training run
from **21.5 hours to under an hour** on an M-series laptop. No rented GPU is
needed at this scale.

Two properties matter beyond the numbers:

- **Insensitivity to sampling rate.** Recording the same shape more finely
  changes the tokens little, and changes them not at all once samples are
  dense enough to land in adjacent grid cells. Point density in raw pen data is
  an artifact of the recorder, so a mouse, a 100 Hz digitizer, and a
  preprocessed public dataset land in nearly the same representation. Measured
  at the default grid, two samplings of one curve reconstruct to within two
  grid cells of each other; exact token equality needs a coarser grid than the
  default.
- **No out-of-vocabulary.** A grid walk always decomposes into base tokens. Bin
  tables have to be retuned per dataset and silently clip whatever falls
  outside them.

This also removes machinery the old code needed. Because ScribeTokens encodes
absolute grid position, words carry their own height and the hardcoded
`STARTS_AT_BOTTOM` / `STARTS_AT_TOP` character tables — one writer's alphabet,
meaningless on any other dataset — are gone, along with the inter-word carriage
jump formula. Generation no longer needs to be seeded with real strokes from the
training set.

## Writing a paragraph

```bash
python pengpt.py sample --checkpoint out/cursive/best.pt --text "Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans."
```

![iliad](static/iliad.png)

Long text is generated a few words per model call and laid out with wrapping.
If a word comes out wrong, find its index with `--show_indices` and regenerate
only that word with `--redo 3,7`.

## Training on your own pen data

The dataset format is a JSON list (optionally zipped) or JSON Lines, one item
per word:

```json
{"text": "hello", "points": [[0.0, 0.1, 1], [0.01, 0.12, 1], ...]}
```

`pen` is 1 while the pen is down; a `pen = 0` point is a lift. y grows downward,
the baseline sits near y = 0, and lowercase letters are roughly 0.1–0.3 units
tall.

- **Collect your own**: open `collect.html` in a browser, write with a
  mouse/trackpad, export JSON. This is how the bundled `bigbank_3500` dataset
  (3,500 words, one author) was made.
- **Convert existing datasets**: `normalize` in `utils.py` brings absolute or
  delta-encoded points to the expected conventions.
- **Irregular point density?** Set `--spacing 0.02` to resample to uniform arc
  length. The bundled data does not need this: `collect.html` records a point
  every time the pen has moved a fixed distance, so its spacing is already
  uniform at ~0.011 and resampling only discards detail. Time-sampled sources
  like IAM do need it.
- The character vocabulary is derived from the data and stored in the
  checkpoint, along with the BPE merges.

## Pen data that is not handwriting

Nothing in the tokenizer knows about letters or words — it encodes pen motion on
a grid. Sketches, signatures, diagrams and gestures work with two settings:

```bash
python pengpt.py train --dataset data/sketches.json --max_words 1 --augment general
```

`--max_words 1` treats each example as one trajectory, which makes the `WORD`
separator inert. `--augment general` drops the shear: it is always negative, so
it imposes an italic slant rather than jittering, and it presumes a baseline to
slant about. `--augment none` disables augmentation entirely, and `--rotate 10`
adds rotation, which suits data with no canonical upright.

Ink scale is the one thing to get right. The grid is a fixed distance, so tokens
per example scale with how large the drawing is; `normalize` in `utils.py`
rescales to the bundled data's scale, and training warns with a suggested
`--grid` if a dataset arrives at a different one.

Measured on the bundled cursive, the shear earns its place: removing it costs
0.79 → 0.95 bits per pen-point even with 37% more optimizer steps. That is a
statement about this dataset, not about pen data in general.

## Quick, Draw!

[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) is 50M
doodles across 345 categories under CC-BY-4.0. Its format is already what pengpt
expects — strokes with pen lifts at the boundaries — so the converter is thin:

```bash
python utils.py convert --quickdraw cat.ndjson --out data/cats.json
python pengpt.py train --dataset data/cats.json --max_words 1 --augment general
```

Much of it is sloppy, and mean quality matters more than raw count, so
`utils.py rank` keeps the best fraction of each category:

```bash
python utils.py download --out_dir qd_raw
python utils.py rank --raw_dir qd_raw --out data/top25.jsonl --resume
```

It renders each drawing, embeds it with CLIP, and scores it with a probe
calibrated on 210 hand-judged drawings (`data/quickdraw_probe.npz`). Held out,
the quarter it keeps is 83% good-or-better against a 48% base rate, with none
of the judged junk surviving — a good coarse filter, not a fine ranking.
Selection is per category so class balance survives, output streams to JSON
Lines, and `--resume` picks up after a crash. Requires `transformers` and
`Pillow` (`pip install -e ".[quality]"`).

Throughput is 204 drawings/s here, so the full corpus is about 68 hours; a
rented GPU does it in one to three, since 92% of the time is CLIP inference.

## Icons

Six open icon sets ship SVGs whose paths are literal pen centerlines — Lucide,
Tabler outline, Feather, Iconoir regular, Heroicons outline, and Akar.
`utils.py icons` aggregates them into one corpus of 8,770 sketchable icons
across 7,541 compositional labels ("arrow down left from circle"), rejecting
filled shapes, un-pennable geometry (30+ strokes, dot-by-dot dashes), and
near-identical renditions that shared ancestry produces (Lucide forked
Feather). Strokes are greedily reordered to minimize pen-up travel, the way a
person sketches. Requires `svgelements`.

```bash
python utils.py icons --raw_dir data/raw --out data/icons.jsonl
python pengpt.py train --preset icons
```

The preset uses a finer grid (0.012 — the default 0.020 is too coarse for
small icon detail) and `--tremor 0.004 --rotate 2`: designer geometry is
ruler-perfect, and a model trained on it learns a drafting machine's hand, so
tremor bridges it toward human ink.

## Repo map

```
pengpt.py            the algorithm: config, ScribeTokens + BPE, data pipeline,
                     PenTransformer, sampling, and the train/sample/rank commands
utils.py             per-dataset presets, converters, and corpus-building tools
                     (Quick, Draw! download/convert/rank, icon aggregation)
collect.html         self-contained data collection page
tests/               pytest suite; pins tokenizer, data, and model properties
```

By Sam Greydanus. Cursive model and dataset from the cursivetransformer project
with Zachary Wimpee.
