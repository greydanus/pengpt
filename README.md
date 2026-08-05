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

## Quickstart

```bash
pip install -e .          # torch, numpy, matplotlib
pip install -e ".[dev]"   # + pytest
pytest tests/             # ~1 second
```

Train:

```bash
python train.py --dataset data/bigbank_3500.json.zip --out_dir out
```

Sample images and the best checkpoint land in `out/` as training goes. Add
`--wandb --wandb_entity you` for Weights & Biases logging (optional). Resume
with `--resume out/best.pt`.

Generate handwriting from a trained model:

```bash
python sample.py --checkpoint out/best.pt --text "The quick brown fox jumps over the lazy dog"
```

If a word comes out misspelled, note its index (`--show_indices`) and regenerate
just that word with `generate_paragraph(..., words=prev, redo=[3, 7])`.

## How it works

**Tokenization** (`pengpt/tokenizer.py`) is
[ScribeTokens](https://arxiv.org/abs/2603.02805). Pen coordinates are quantized
to an integer grid, and motion becomes a walk on that grid: eight compass
directions plus `DOWN` and `UP`. Movement between grid points is decomposed with
Bresenham's line algorithm, so any path is representable by ten base symbols.
Byte pair encoding then merges recurring runs — a common curve becomes a single
token — which is what keeps sequences short.

![roundtrip](static/roundtrip.png)

*Grey is the original, red is decoded from tokens. `grid=0.012` is the default.*

**Model** (`pengpt/model.py`): a small GPT-style decoder (~430k params at
default size) with cross-attention over the character embedding of the text
prompt, in the makemore/nanoGPT lineage.

**Data** (`pengpt/data.py`): each example is one word, an `(N, 3)` array of
`(x, y, pen)`. Training examples pack random words together until the block is
full, so a few thousand words become effectively unlimited examples and nothing
is ever truncated mid-word.

## Why ScribeTokens

The original used polar `(theta, r)` bins, two tokens per pen point. Measured on
bigbank_3500, ScribeTokens is better on every axis at once:

| | tokens/word | reconstruction error | bits/pen-point | vocab |
|---|---|---|---|---|
| polar bins | 184.8 | 0.0116 | 13.09 | 525 |
| ScribeTokens | **82.3** | **0.0077** | **5.94** | **148** |

Trained head to head at matched wall clock, ScribeTokens reached **3.43
bits/pen-point against the polar tokenizer's 7.16** — a 2.1× advantage, while
running *fewer* optimizer steps. BPE is doing real work here: at equal block
sizes, 256 merges scored 1.29 against 2.12 with no merges at all.

Two properties matter beyond the numbers:

- **Sampling invariance.** Two recordings of the same shape tokenize
  identically however densely the hardware sampled them. A mouse, a 100 Hz
  digitizer, and a preprocessed public dataset all land in one representation,
  so a single model can train across all of them.
- **No out-of-vocabulary.** A grid walk always decomposes into base tokens. Bin
  tables have to be retuned per dataset and silently clip whatever falls
  outside them.

This also removes machinery the old code needed. Because ScribeTokens encodes
absolute grid position, words carry their own height and the hardcoded
`STARTS_AT_BOTTOM` / `STARTS_AT_TOP` character tables — one writer's alphabet,
meaningless on any other dataset — are gone, along with the inter-word carriage
jump formula. Generation no longer needs to be seeded with real strokes from the
training set.

## Training on your own pen data

The dataset format is a JSON list (optionally zipped), one item per word:

```json
{"text": "hello", "points": [[0.0, 0.1, 1], [0.01, 0.12, 1], ...]}
```

`pen` is 1 while the pen is down; a `pen = 0` point is a lift. y grows downward,
the baseline sits near y = 0, and lowercase letters are roughly 0.1–0.3 units
tall.

- **Collect your own**: open `collect.html` in a browser, write with a
  mouse/trackpad, export JSON. This is how the bundled `bigbank_3500` dataset
  (3,500 words, one author) was made.
- **Convert existing datasets**: see `pengpt/convert.py`.
- **Irregular point density?** Set `--spacing 0.02` to resample to uniform arc
  length. The bundled data does not need this: `collect.html` records a point
  every time the pen has moved a fixed distance, so its spacing is already
  uniform at ~0.011 and resampling only discards detail. Time-sampled sources
  like IAM do need it.
- The character vocabulary is derived from the data and stored in the
  checkpoint, along with the BPE merges.

## Repo map

```
pengpt/config.py     dataclass configs + CLI
pengpt/tokenizer.py  ScribeTokens + BPE + char tokenizer
pengpt/data.py       loading, augmentation, PenDataset
pengpt/model.py      PenTransformer + checkpoint I/O
pengpt/sampling.py   generation, paragraph layout, plotting
pengpt/convert.py    external dataset converters
train.py, sample.py  CLI entry points
collect.html         self-contained data collection page
deprecated/          the previous polar tokenizer
```

By Sam Greydanus. Cursive model and dataset from the cursivetransformer project
with Zachary Wimpee.
