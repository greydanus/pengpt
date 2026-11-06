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
- On sequences short enough for every model, the settings are within 0.01 of each other. 8 words stays the default.
