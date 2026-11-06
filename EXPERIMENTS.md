# Experiments

Cursive results on `bigbank_3500` (one writer, 3,325 train / 175 test words), M3 Pro laptop, `--preset cursive`, 5,000 steps unless noted. Nov 2026.

**Score** = nats per base pen move on the 175 clean (unaugmented) held-out words, writer `sam`. It doesn't depend on the tokenizer or the packing, unlike test loss. Lower is better. **Greedy** = phrases or words spelled right in greedy samples, counted by eye.

## Augmentation

| Augmentation | Score |
|---|---|
| none | 0.775 (overfits: test loss rises after ~1k steps) |
| old: fixed slant −0.22..−0.18 per word | 0.834 |
| **now**: per sequence, slant ±0.1, scale ±10%, rotate ±2°, offset ±½ grid | **0.602** |

## Schedule

20k steps with warmdown 0.4: 0.590 (the original model scored 0.788). The test loss plateaued while the learning rate was held flat, so warmdown is now 0.8.

## Mixing in BRUSH

| Mix (cursive : BRUSH) | Score | Cursive only |
|---|---|---|
| 10:1, old augmentation | 0.809 | 0.830 |
| 3:1, current augmentation | 0.615 | 0.602 |

With current augmentation, BRUSH no longer helps the cursive score. Writer styles came out distinct but mostly illegible.

## Data scaling (training words)

| Words | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,325 |
|---|---|---|---|---|---|---|
| Score | 0.631 | 0.619 | 0.608 | 0.604 | 0.606 | 0.605 |
| Greedy, of 25 words | 16 | 18 | 14 | 18 | 16 | – |

The score flattens around 2,000–2,500 words. A 3,325-word run on the old warmdown 0.4 scored 0.602 with 22/25 greedy; the 3,325 score above is from the warmdown 0.8 run (the packing experiment's 8-word run).

Seed check (`--train_seed`, greedy = phrases right of 8):

| Words | Scores | Greedy |
|---|---|---|
| 2,000 | 0.608, 0.617 (mean 0.613) | 1, 3 |
| 3,325 | 0.603, 0.605, 0.605 (mean 0.604) | 5, 6, 3 |

A same-seed rerun differs by 0.003 (MPS nondeterminism). All the words beats 2,000 on both measures, so 2,000 isn't saturated.

## English equivalent of bigbank

Which part of the gain depends on a word's own letters? Per held-out word, the loss across the scaling models is explained by:

| Model | R² |
|---|---|
| one curve for every word | 0.172 |
| exposure to the word's letters | **0.187** |
| exposure to its letter pairs | 0.161 |
| letter exposures shuffled across words | 0.143 (95th pct 0.165) |

Most of the gain is shared by all words; letter exposure adds a little and letter pairs nothing. Exposure model: mean over a word's characters of (count + 1)^−β, best β 0.95.

Bigbank is two-thirds random strings over 70 characters. English words are sampled by `wordfreq` frequency; "as written" adds 12% capitals, ~10% trailing punctuation, 1% numbers. English words needed to match 2,000 bigbank words:

| Judged on | β 0.5 | β 0.95 | β 1.5 |
|---|---|---|---|
| English as written | 2,030 | **3,555** | 8,480 |
| English, lowercase | 1,356 | **1,628** | 2,658 |
| bigbank-style strings | 40,322 | never | never |

Rare capitals, digits and punctuation set the as-written number. Counting ink instead (English words average 4.5 characters, bigbank 5.3), 2,000 bigbank words ≈ 2,350 English words. Rule of thumb: ~3,500 English words for a writer you want reproduced well, plus some bigbank-style strings or synthetic font writers for rare characters. Not modeled: English also teaches whole common words, which only lowers the number.

## Burn-in generation

Writing two words per call, with the previous word's ink and text carried in as context (`--burn_in 1`): 17/17 Iliad words correct, vs 16/17 without.

## Packing (`--max_words`, block size)

Each sequence packs 1 to `max_words` words, drawn uniformly. Scores by context:
- **1 word**: each test word alone.
- **deploy**: two at a time, given the true previous word as unscored context, as `sample` writes.
- **8 words**: fixed packed sequences of 7.3 words on average.

| max_words / block | ms/step | min | words/seq | tokens/seq | 1 word | deploy (1st, 2nd) | 8 words | text rank | greedy /8 |
|---|---|---|---|---|---|---|---|---|---|
| 2 / 256 | 15 | 2.0 | 1.50 | 95 | 0.615 | 0.785 (0.634, 0.951) | 1.217 | 1.00 | 4 |
| 3 / 384 | 22 | 2.9 | 1.98 | 128 | 0.612 | 0.617 (0.614, 0.620) | 0.984 | 1.00 | 5 |
| 4 / 512 | 31 | 3.6 | 2.50 | 162 | 0.608 | 0.610 (0.607, 0.614) | 0.756 | 1.19 | 7 |
| **8 / 512** | 31 | 3.8 | 4.31 | 277 | **0.605** | **0.606** (0.605, 0.608) | **0.612** | 1.00 | 6 |

- Models break past the longest sequence they trained on. 2-word packing fails under burn-in, because burn-in plus 2 words makes a 3-word sequence.
- Step time follows block size, not word count, so 8 words gets 1.7× the ink of 4 words for the same cost.
- On sequences short enough for every model, the settings are within 0.01 of each other. 8 words stayed the default until the matched-time test below.

## Packing at matched training time (Nov 6)

4,000 steps unless noted. Greedy counts are words right in the pangram (of 9) and the Iliad opening (of 17).

| Setup | Training time | Score | Greedy pangram | Greedy Iliad |
|---|---|---|---|---|
| 8 words / block 512 / batch 16 | 2.2 min | 0.6107 | 3 | ~6 |
| **4 words / block 256 / batch 32** | 2.0 min | **0.5980** | 9 | ~15 |
| same, `--train_seed 1` | 1.8 min | **0.5957** | 9 | 16 |
| 4 words / block 256 / batch 16, 8,000 steps | 2.8 min | 0.5970 | – | – |

Four words wins by ~0.013 (4× seed noise) on both seeds, and the 8-word model's letters leaking across word boundaries ("teh azyy vegg") disappear. A 20k-step 8-word run scored 0.590 in ~14 min; this gets within 0.006 in 2 min. Now the default. Block 256 holds 146 tokens on average (57% full, vs 54% for 8 words at 512).

## Augmentation ablation (Nov 6, 8 words / block 512)

| Setting | Score |
|---|---|
| **current** (slant ±0.1, scale ±10%, rotate ±2°, offset ±½ cell) | **0.6107** |
| half ranges (slant ±0.05, scale ±5%, rotate ±1°) | 0.6101 |
| current + per-word jitter (¼ of the ranges, on top) | 0.6143 |
| geometry only (no offset) | 0.6160 |
| wider (slant ±0.2, scale ±15%, rotate ±3°) | 0.6268 |
| offset only | 0.6271 (overfits) |

Even a ±0.05-cell shift re-tokenizes 55% of a word's tokens, so any augmentation re-tokenizes. But offset alone overfits: the geometry matters, and wider ranges lose texture. Changing augmentation changes the learned merges, so compare by score, not test loss (geometry-only had the lowest test loss but a worse score).

## Pen-position RoPE (Nov 6, 8 words / block 512)

Rotating 4 of each head's 16 rotary pairs by pen x and 4 by pen y, instead of the additive Fourier features: score 0.6125 vs 0.6107, test loss 1.9931 vs 1.9794, greedy pangram 1/9 vs 3/9, with more letters leaking across words. Not kept.

## Eval cost

Evals used ~28% of a run (a train-loss pass, padded batches, two generation passes for the progress image every eval). Now train loss is the running mean of step losses, eval batches are trimmed to their real length, and the progress image renders every `--sample_every` (5,000) steps in one batched generation: ~1 s per eval, ~4% of training.

## Step time (Nov 6, 4 words / block 256 / batch 32, M3 Pro)

A training step was ~26 ms: forward ~7 ms, backward ~25 ms, optimizer ~5 ms (timed on synthetic batches; absolute numbers run high under load, ratios held). The backward has no single slow op. Removing pieces one at a time:

| Removed or replaced | Step change |
|---|---|
| cross-attention (all layers) | −7.1 ms |
| QK-norm | −2.0 ms |
| hand-written attention instead of SDPA | ~0 (backward −4.5, forward +4) |
| cross-attention's padding mask, RMSNorm kernel, embedding grads, pen features, logit soft cap | ~0 each |

Fused AdamW cut the AdamW update from 2.0 to 0.2 ms (26.0 vs 27.7 ms/step), now the default. Compiling the optimizer saves another ~0.6 ms but would recompile as the learning rate changes; not done.

4,000-step runs at the defaults:

| Change | Params | ms/step | Score |
|---|---|---|---|
| none (three seeds) | 395k | 25.8 | 0.5957–0.5986 |
| 1 head | 396k | 24.7 | 0.6094 |
| fp16 autocast + grad scaling | 395k | 57.3 | 0.5990 |
| n_embd 96, 1 head | 839k | 29.2 | 0.5943 |

1 head saves 4%, not the 15% a synthetic benchmark showed, and costs 0.011. fp16 matches quality but runs 2.2× slower on MPS. The 96-wide 1-head model ties the 20k-step run's score at 4k steps; one seed, within ~0.003 of the baseline, so not adopted yet.

## Steps (Nov 6)

At the 4-word defaults, 20k steps (10.5 min) scored 0.5943 against 0.5957–0.5986 at 4k steps (~2 min). Test loss flattened after ~12k steps while train loss kept falling (1.84 vs 1.90 at 20k).

## Decode smoothing (Nov 6)

Decoding averages each stroke's grid walk over a few points to hide the 0.02-grid staircase. A 5-point window spans ~0.1 units, close to a full x-height, and visibly rounded corners the real ink has (pointed S tops, angular z, kinks in k). A 3-point window keeps most of that sharpness with slight jaggedness on long straight strokes; it is now the default. This only changes drawing, not training or checkpoints.

## Repeated 4k cycles and the overfitting point (Nov 6)

The 4k schedule (200-step warmup, linear decay to 0 over the last 80%) repeated back to back, resuming each cycle from the last. Scores at cycle ends, where the learning rate has decayed to 0:

| Cycle end | Train loss | Test loss | Score |
|---|---|---|---|
| 4k | 1.9653 | 1.8949 | 0.5958 |
| 8k | 1.9100 | 1.8674 | 0.5885 |
| **12k** | 1.8833 | **1.8600** | **0.5866** |
| 16k | 1.8660 | 1.8617 | 0.5871 |
| 20k | 1.8531 | 1.8664 | 0.5891 |
| 24k | 1.8424 | 1.8717 | 0.5904 |

Held-out scores bottom out after three cycles (12k steps, ~6.5 min) and rise for three cycles after, while train loss keeps falling: overfitting starts around 12k steps at these defaults. Three cycles beat the single 20k-step run (0.5943) by 0.008. A single 12k-step run with one decay did better still: test loss 1.8551, score **0.5849**, in 6.1 min. The length mattered, not the restarts, so the default is now 12,000 steps; `--steps 4000` stays the quick-experiment setting.

## Grid resolution (Nov 6)

p95 distance from decoded strokes to the original ink, as % of x-height, with tokens per word (BPE learned per grid):

| Grid | Cells per x-height | PatrickHand traces | BRUSH | bigbank | Tokens/word (traces) |
|---|---|---|---|---|---|
| 0.02 | 6.1 | 6.8 | 6.5 | 8.1 | 76 |
| 0.015 | 8.2 | 5.2 | 4.9 | 4.5 | 89 |
| 0.0125 | 9.8 | 4.3 | 4.0 | 4.1 | 98 |
| 0.01 | 12.3 | 3.5 | 3.3 | 4.2 | 107 |
| 0.0075 | 16.4 | 2.7 | 2.5 | 2.4 | 119 |

Decode smoothing can't recover detail below the grid: unsmoothed decoding already sits at about half a cell. 0.0125 is the candidate for traced fonts: its p95 error (~0.005 units) is about one InkSight pixel. Finer grids mostly encode tracer noise. Word score is per base pen move, so it doesn't compare across grids.
