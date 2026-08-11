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

## Quality over a training run

![progress](static/progress.png)

The same three prompts, rendered at each eval. Stroke control and slant arrive
almost immediately; spelling follows. All three prompts are correct by step
10,000, about 25 minutes in, and the rest of the run sharpens the penmanship.

## Quickstart

```bash
pip install -e .          # torch, numpy, matplotlib
pip install -e ".[dev]"   # + pytest
pytest tests/             # ~1 second
```

Train — about 45 minutes on an M-series laptop, no GPU rental needed:

```bash
python train.py --dataset data/bigbank_3500.json.zip --out_dir out/cursive
```

Each run owns a folder under `out/`:

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
having a bad draw. Stack them with `python progress.py`, or put real
handwriting beside generations for the same text with `python compare.py`.

Add `--wandb --wandb_entity you` for Weights & Biases logging (optional).
Resume with `--resume out/cursive/last.pt` -- `last.pt`, not `best.pt`, which
lags whenever test loss has stopped improving. Resuming rebuilds the tokenizer
from the checkpoint rather than from the command line, and says so when a data
flag disagrees: merges and alphabet decide what every token id means, so a
re-derived vocabulary would load the weights against ids they were never
trained on. For a faster run, `--n_layer 3` roughly halves the time and scored
within noise of the default in ablations.

Checkpoints from before commit `1693c3f` no longer load. That commit is titled
for a change to `conditioning.py` but also rewrote the token stream: BPE merges
now apply in learned order, and a BOS token was added. Both change what the ids
mean and shift `vocab_size`, so a model from before it has to be retrained.

### Drawings rather than handwriting

The defaults are tuned for cursive. A drawing corpus needs three of them
changed, because its samples are single objects with short labels:

```bash
python train.py --dataset data/quickdraw_balanced_fixed.jsonl.gz \
  --max_words 1 --augment general \
  --max_text_length 24 --max_seq_length 192 \
  --n_layer 6 --n_embd 128 --learning_rate 1e-3 --batch_size 32 \
  --out_dir out/quickdraw
```

`quickdraw_balanced_fixed` is `quickdraw_balanced` after `repair_quickdraw.py`:
the original was written through a delta-detection bug that cumsum'd 24% of
drawings into diagonal staircases (see `normalize` in `pengpt/convert.py`).
At the repaired token statistics (p50 = 39, p99 = 136), a 192-token block
covers 99.6% of drawings; 384 spent most of every sequence on padding.

`--augment general` drops shear, which is an italic slant that presumes a
baseline and is a distortion on a sketch. Set `--max_text_length` near the
longest label and `--max_seq_length` near the p99 token count; leaving them at
the cursive defaults spends most of every sequence on padding.

Loss alone will not tell you whether the prompt is being read -- a model can
post a falling loss while drawing the corpus average for every prompt. Measure
it directly:

```bash
python conditioning.py --checkpoint out/quickdraw/best.pt --per_label 8
```

Generate handwriting from a trained model:

```bash
python sample.py --checkpoint out/cursive/best.pt --text "The quick brown fox jumps over the lazy dog"
```

If a word comes out misspelled, note its index (`--show_indices`) and regenerate
just that word with `--redo 3,7`.

## How it works

**Tokenization** (`pengpt/tokenizer.py`) is
[ScribeTokens](https://arxiv.org/abs/2603.02805). Pen coordinates are quantized
to an integer grid, and motion becomes a walk on that grid: eight compass
directions plus `DOWN` and `UP`. Movement between grid points is decomposed with
Bresenham's line algorithm, so any path is representable by ten base symbols.
Byte pair encoding then merges recurring runs — a common curve becomes a single
token — which is what keeps sequences short.

![roundtrip](static/roundtrip.png)

*Grey is the original, red is decoded from tokens. `grid=0.020` is the default:
reconstruction error stays inside the width of a pen stroke at 99 tokens per word.*

**Model** (`pengpt/model.py`): a small GPT-style decoder (~410k params at the
cursive default, ~1.7M at the drawing settings above) with cross-attention over
the character embedding of the text prompt, in the makemore/nanoGPT lineage.

The prompt reaches the model only through that cross-attention, never as a class
index, so the same architecture takes a word to write or the name of an object
to draw, and an unseen label still says something through its characters.

**Data** (`pengpt/data.py`): each example is one word, an `(N, 3)` array of
`(x, y, pen)`. Training examples pack random words together until the block is
full, so a few thousand words become effectively unlimited examples and nothing
is ever truncated mid-word.

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
python iliad.py --checkpoint out/cursive/best.pt
```

![iliad](static/iliad.png)

Long text is generated a few words per model call and laid out with wrapping.
If a word comes out wrong, find its index with `--show_indices` and regenerate
only that word with `--redo 3,7`.

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

## Pen data that is not handwriting

Nothing in the tokenizer knows about letters or words — it encodes pen motion on
a grid. Sketches, signatures, diagrams and gestures work with two settings:

```bash
python train.py --dataset data/sketches.json --max_words 1 --augment general
```

`--max_words 1` treats each example as one trajectory, which makes the `WORD`
separator inert. `--augment general` drops the shear: it is always negative, so
it imposes an italic slant rather than jittering, and it presumes a baseline to
slant about. `--augment none` disables augmentation entirely, and `--rotate 10`
adds rotation, which suits data with no canonical upright.

Ink scale is the one thing to get right. The grid is a fixed distance, so tokens
per example scale with how large the drawing is; `convert.py` normalizes to the
bundled data's scale, and `train.py` warns with a suggested `--grid` if a dataset
arrives at a different one.

Measured on the bundled cursive, the shear earns its place: removing it costs
0.79 → 0.95 bits per pen-point even with 37% more optimizer steps. That is a
statement about this dataset, not about pen data in general.

## Physics sketches

`physics_sketch.py` generates physics word problems procedurally: one parameter
draw yields a caption, a pen sketch of that setup, and a question whose answer
follows from the physics. Ground truth is exact by construction.

```bash
python physics_sketch.py --n 20000 --max_text_length 160 --out data/physics_v1.jsonl
python physics_sketch.py --preview     # per-archetype validation sheets
python train.py --dataset data/physics_v1.jsonl \
  --max_words 1 --augment general \
  --max_text_length 160 --max_seq_length 160 \
  --n_layer 6 --n_embd 128 --learning_rate 1e-3 --batch_size 32 \
  --out_dir out/physics
```

`--max_text_length 160` is not optional. These captions compare two clauses and
the deciding one is usually the second, so the cursive default of 50 cuts the
prompt before the information that fixes the answer: 33% of `physics_v0` was
trained on a prompt shared by examples with opposite labels. Loss falls anyway.
Generation now refuses to write a file whose captions its `--max_text_length`
would make ambiguous, and prints the value to train with.

`--max_seq_length 160` covers the p99: 278 merges take the drawings from a p99
of 265 raw tokens to 124.

### How much does the picture matter?

The natural failure of a procedural generator is that the caption states every
attribute, so text alone answers the question and the sketch is redundant. A
scratchpad can then only break even. Measured per archetype as *caption-only
accuracy* — how often the caption alone picks out the answer:

| regime | archetypes | caption-only | reading |
|---|---|---|---|
| stated | lever, ramp, scale, … | 100% | picture redundant |
| elided | lever, stack with `--detail elided` | 80% | picture helps |
| geometry | reach, fit, shadow | 50–55% | picture required |

`--detail elided` withholds one attribute from the caption and leaves it in the
drawing alone — the lever's distances, the stack's shift. The geometry
archetypes go further and are *always* elided: the caption names the objects
("a ladder leans from the ground toward the top of a wall") and the arrangement
lives entirely in the sketch, so text alone is at chance.

That is why `reach`, `fit` and `shadow` have one or two captions each. For them
low caption diversity is correct — their variety is in the drawing. The caption
floor test exempts them, and a separate test asserts the opposite property:
that their captions stay *insufficient*.

```bash
python physics_sketch.py --n 28000 --style plain --detail elided \
  --max_text_length 170 --out data/physics_v2.jsonl
```

36% of that corpus needs the picture. `meta["elided"]` names the withheld
attribute, so the picture-needed subset can be scored on its own.

### Composing attributes, not comparing one

Each archetype scores a small set of binned attributes and rejects draws that
land too near a tie, so the caption always decides the answer with a margin.
What makes an archetype *diverse* is composing several attributes rather than
comparing one between two objects: a single attribute admits only 3 sizes
choose 2, times 2 orderings, which is six captions no matter how many examples
are generated. `ramp`, `pendulum` and `drop` each had exactly that ceiling.

| archetype | attributes | captions |
|---|---|---|
| lever | size × distance × noun, both sides | 992 |
| spring | load × stiffness × noun, both springs | 992 |
| magnet | strength × distance × bolt material | 192 |
| roll | push × surface × cart load | 116 |
| scale | size × noun, both pans | 96 |
| pendulum | length × bob mass × release angle | 96 |
| drop | height × roll speed × ball weight | 96 |
| pulley | size × noun, both sides | 96 |
| stack | block count × shift × base width | 90 |
| float | material × fluid × shape × size | 90 |
| ramp | height × surface roughness | 62 |

Eleven archetypes across statics, buoyancy, springs, pulleys, rolling friction
and magnetism, plus three geometry archetypes — so no single heuristic
(bigger is heavier, taller is faster) covers the corpus. `float` deliberately
contradicts the bigger-is-heavier habit the lever and scale reward: a big cork
floats and a small steel nut sinks.

Some of those attributes are causal and some are deliberately not.
Speed at the bottom of a ramp really does depend on height and roughness, and a
tower really does tip when its top block passes the base edge. But a pendulum's
period does not depend on bob mass, and fall time does not depend on how fast a
ball rolls off or how heavy it is. Those are stated in the caption and drawn in
the sketch anyway, so answering requires knowing which stated attribute matters
rather than reading off the one number that varies. Every archetype's answers
stay balanced, so none is guessable from its prior.

`physics_sketch.py` prints per-archetype caption counts on every run, and the
test suite pins a floor under them.

### Multi-step chains: depth as the independent variable

The archetypes above are all one step, so a scratchpad has nothing to hold.
Chains make reasoning depth the only thing that varies:

```bash
python physics_sketch.py --n 24000 --depths 1,2,3,4 --style plain \
  --max_text_length 420 --out data/physics_chains.jsonl
python physics_sketch.py --preview --depths 1,2,3,4 --style plain
```

A chain stage *consumes* the previous stage's output and *produces* one for the
next — a gate that routes a ball, a slope that turns a side into a speed, a gap
that turns a speed into a landing, a seesaw that turns a landing back into a
side. Depth 1 is a bare lever, so the sweep starts from the same one-step
problem the standalone archetypes pose, and everything else — vocabulary,
drawing style, answer format — is held constant.

Each example carries per-step ground truth in `meta["steps"]`, so an
*intermediate* state can be verified rather than only the final answer. Dense
process reward is the reason to generate these rather than scrape them.

**A stage caption never names the carry it received.** Stages say "that side"
and describe a transformation ("sends the ball out the opposite way") rather
than a state. An earlier version named it, and the final clause alone then
predicted the answer 95% of the time at depth 4 — every earlier stage was
decorative and a "depth 4" example was a depth-1 example wearing a label. A
stage that ignores its carry (a gap nothing can clear) is also refused as a
chain's last step, for the same reason. It is now ~57%, near the floor for a
binary answer, and a test pins it.

### Keeping sketches minimal

`--style plain` drops the ink that carries no information about the answer.
Ground hatching costs 111 tokens against 38 for a plain line, and a hatched
crate 37 against 20 for a box, so decoration was most of a short scene.
Roughness ticks stay in plain mode — they are the only thing in the drawing
that carries the ramp's answer — but use fewer marks.

Chain stages are also drawn at `CHAIN_SCALE`, since token cost is proportional
to ink size at a fixed grid and a stage inside a chain is a schematic element
rather than a whole picture. Together these took a depth-4 sketch from a p99 of
568 tokens to 118, and made cost linear in depth:

| depth | sketch tokens (p50 / p99) | caption chars (p99) |
|---|---|---|
| 1 | 18 / 22 | 105 |
| 2 | 41 / 69 | 258 |
| 3 | 64 / 93 | 330 |
| 4 | 86 / 118 | 401 |

Linearity is the point. If deeper problems also carried more decorative ink,
sketch length would be confounded with reasoning depth and a scaling curve
could not separate the two.

## Quick, Draw!

[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) is 50M
doodles across 345 categories under CC-BY-4.0. Its format is already what pengpt
expects — strokes with pen lifts at the boundaries — so the converter is thin:

```bash
python -m pengpt.convert --quickdraw cat.ndjson --out data/cats.json
python train.py --dataset data/cats.json --max_words 1 --augment general
```

Much of it is sloppy, and mean quality matters more than raw count, so
`rank_quickdraw.py` keeps the best fraction of each category:

```bash
python rank_quickdraw.py --raw_dir qd_raw --out data/top25.jsonl --resume
```

It renders each drawing, embeds it with CLIP, and scores it with a probe
calibrated on 210 hand-judged drawings (`pengpt/quality.py`). Held out, the
quarter it keeps is 83% good-or-better against a 48% base rate, with none of the
judged junk surviving — a good coarse filter, not a fine ranking. Selection is
per category so class balance survives, output streams to JSON Lines, and
`--resume` picks up after a crash.

Throughput is 204 drawings/s here, so the full corpus is about 68 hours; a
rented GPU does it in one to three, since 92% of the time is CLIP inference.

## Repo map

```
pengpt/config.py     dataclass configs + CLI
pengpt/tokenizer.py  ScribeTokens + BPE + char tokenizer
pengpt/data.py       loading, augmentation, PenDataset
pengpt/model.py      PenTransformer + checkpoint I/O
pengpt/sampling.py   generation, paragraph layout, plotting
pengpt/convert.py    external dataset converters
pengpt/quality.py    rank drawings, to filter a crowd-sourced corpus
train.py             training loop
sample.py            write arbitrary text
iliad.py             write the Iliad opening
compare.py           real handwriting beside generations
progress.py          contact sheet of samples over a run
rank_quickdraw.py    filter Quick, Draw! to its best drawings
physics_sketch.py    procedural physics problems: caption, sketch, answer
sketch_style.py      hand-style variation for procedural sketches
ink_icons.py         aggregate stroke-native icon sets into one pen corpus
ink_sketchy.py       convert the Sketchy database (75k object sketches)
collect.html         self-contained data collection page
```

By Sam Greydanus. Cursive model and dataset from the cursivetransformer project
with Zachary Wimpee.
