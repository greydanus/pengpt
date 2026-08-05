# pengpt

A minimal, general-purpose transformer for pen strokes. Give it a corpus of pen
trajectories — handwriting, sketches, anything drawn with a stylus or mouse —
and it learns to generate more of them, conditioned on text.

This is a ground-up rewrite of
[cursivetransformer](https://github.com/greydanus/cursivetransformer)
([paper](https://arxiv.org/abs/2504.00051) ·
[blog post](https://greydanus.github.io/2025/03/30/cursive-transformer/)),
keeping the recipe that produced the paper results while making the code
smaller, cleaner, and easier to point at new datasets.

![hero](static/hero.png)

## Quickstart

```bash
git clone <this-repo> && cd pengpt
pip install -e .          # torch, numpy, matplotlib
pip install -e ".[dev]"   # + pytest
pytest tests/             # ~1 second
```

Train (defaults reproduce the paper recipe — a few hours on an A100):

```bash
python train.py --dataset data/bigbank_3500.json.zip --out_dir out
```

Sample images and the best checkpoint land in `out/` as training goes. Add
`--wandb --wandb_entity you --wandb_project pengpt` for Weights & Biases
logging (optional; everything also logs locally). Resume with
`--resume out/best.pt`.

Generate handwriting from a trained model:

```bash
python sample.py --checkpoint out/best.pt --text "The quick brown fox jumps over the lazy dog"
```

If a word comes out misspelled, note its index (`--show_indices`) and
regenerate just that word from Python with
`generate_paragraph(..., word_offsets=prev, redo=[3, 7])`.

## How it works

- **Data**: each example is one handwritten word, an `(N, 3)` array of
  `(x, y, pen)` points. Training examples are made by stitching random
  combinations of `num_words` words, which turns a few thousand words into
  hundreds of thousands of distinct examples.
- **Tokenization** (`pengpt/tokenizer.py`): points → per-step offsets → polar
  `(r, theta)` → two tokens per point: a direction token, then a combined
  (magnitude, pen state) token — "point, then shoot". Words are separated by a
  pair of `WORD` tokens; sequences end with `END`. Vocabulary: 525 tokens.
- **Model** (`pengpt/model.py`): a small GPT-style decoder (~420k params at
  default size) with cross-attention over the character embedding of the text
  prompt, in the makemore/nanoGPT lineage.
- **Augmentation** (`pengpt/data.py`): per-word shear (slant), x/y rescaling,
  and randomized stroke downsampling that always preserves stroke endpoints.
  The randomized downsampling rate decorrelates letter identity from token
  position and matters a lot for generalization.

## Training on your own pen data

The dataset format is a JSON list (optionally zipped), one item per word:

```json
{"text": "hello", "points": [[0.0, 0.1, 1], [0.01, 0.12, 1], ...]}
```

`pen` is 1 while the pen is down; a `pen = 0` point is a move. y grows
downward, the baseline sits near y = 0, and lowercase letters are roughly
0.1–0.3 units tall (match the bundled data's scale).

- **Collect your own**: open `collect.html` in a browser, write words with a
  mouse/trackpad, export JSON. This is how the bundled `bigbank_3500` dataset
  (3,500 words, one author) was made.
- **Convert existing datasets**: see `pengpt/convert.py` (includes a BRUSH
  converter and a `--probe` mode for inspecting unknown pickle formats). For
  sentence-level datasets, train with `--num_words 1` and a larger
  `--max_text_length`.
- The character vocabulary is derived from the dataset automatically and
  stored in the checkpoint.

## What changed vs. cursivetransformer

Same tokenizer bins, architecture shape, and hyperparameters; the differences
are engineering:

- Attention uses `F.scaled_dot_product_attention` (flash attention) instead of
  hand-rolled masks — faster and less memory.
- Loss is masked after the `END` token instead of being computed over padding
  (~40% of every batch was trivial pad-prediction), and generation stops at
  `END` instead of always running the full window.
- Word combos are stored as index tuples and materialized lazily: dataset
  construction went from ~10 GB of RAM to ~16 MB, and startup from minutes to
  seconds.
- Augmentation uses a local `np.random.Generator` instead of reseeding global
  numpy state per item.
- wandb is optional, the alphabet is derived from data rather than hardcoded,
  checkpoints are self-describing (config + alphabet included), and there is a
  test suite.

## Repo map

```
pengpt/config.py     dataclass configs + CLI
pengpt/tokenizer.py  geometry + stroke/char tokenizers
pengpt/data.py       loading, augmentation, PenDataset
pengpt/model.py      PenTransformer + checkpoint I/O
pengpt/sampling.py   generation, paragraph layout, plotting
pengpt/convert.py    external dataset converters
train.py, sample.py  CLI entry points
collect.html         self-contained data collection page
```

By Sam Greydanus. Cursive model and dataset from the cursivetransformer
project with Zachary Wimpee.
