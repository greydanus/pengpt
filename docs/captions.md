# What the caption vectors buy the drawing model

The drawing model is told SigLIP 2 vectors while it trains: the vector of the drawing's own picture, of its label, and of 17 kinds of
generated caption. The captions cost the most to make and multiply the size of the dataset, so this note asks whether they are needed,
or whether a model trained on the picture vector alone could be told the vector of typed text when it draws.

**Answer.** They are needed for typed text. A model trained on pictures alone draws the wrong thing when told a text vector (its own
prompt ranked first 6% of the time, against 75%). A model trained on pictures and labels matches the full model on single-word labels
(77.5% against 75.4%) and fails on longer phrases (9.5% against 43.0%). The captions buy free-text prompting and nothing else: on the
held-out pen-move score the three models are within 0.016 nats of each other.

## Runs

All runs are the sketch preset on one A100 (12 layers, 512 wide, 54.0M parameters, 467,973 training and 9,697 held-out drawings from
ten sources), 54,000 steps, evaluated every 3,000, scored at the checkpoint with the lowest held-out loss. They differ in one flag.

| Run | Flag | Told while training |
|---|---|---|
| base | | picture, label and 17 caption kinds |
| seed | `--train_seed 1` | the same; the noise to read the rest by |
| image_label | `--vec_columns image,label` | picture and label |
| image_only | `--vec_columns image` | picture |
| no_guidance | `--label_drop 0` | as base, the vector never dropped |
| no_augment | `--augment 0` | as base, drawings never moved or scaled |

## Typed prompts

Each model draws typed prompts at temperature 0.6, two drawings a prompt. Each drawing is rendered and given to SigLIP 2 with all the
prompts of its set; the score is how often its own prompt is the best match. Two sets: the 345 Quick, Draw! labels, and 300 held-out
captions of the `short` kind, 30 from each source (`out/ablate_sketch/phrases.txt`). "Shifted" adds the mean picture vector less the
mean text vector (length 1.041, from 3,000 training drawings) to each typed vector and renormalises it, the cheapest way to carry
text across the gap between the two kinds of vector.

### 345 labels

| Run | Guidance | Cosine to its prompt | Its prompt ranked first | In the first five | Strokes |
|---|---|---|---|---|---|
| base | 1 | 0.1151 | 54.9% | 73.3% | 21.1 |
| base | 2 | 0.1304 | 75.4% | 89.6% | 14.8 |
| seed | 1 | 0.1180 | 57.5% | 76.5% | 23.3 |
| seed | 2 | 0.1304 | 75.1% | 90.7% | 13.0 |
| image_label | 1 | 0.1186 | 57.5% | 77.4% | 19.4 |
| image_label | 2 | 0.1325 | 77.5% | 91.7% | 15.1 |
| image_only | 1 | 0.0602 | 4.5% | 11.6% | 143.8 |
| image_only | 2 | 0.0625 | 6.1% | 14.2% | 84.1 |
| image_only, shifted | 1 | 0.0814 | 18.8% | 38.0% | 60.9 |
| image_only, shifted | 2 | 0.0839 | 22.3% | 43.8% | 26.3 |
| no_guidance | 1 | 0.1191 | 57.7% | 77.4% | 14.5 |
| no_augment | 1 | 0.0781 | 14.3% | 28.8% | 71.2 |
| no_augment | 2 | 0.0612 | 3.9% | 11.7% | 3.9 |

### 300 held-out caption phrases

| Run | Guidance | Cosine to its prompt | Its prompt ranked first | In the first five | Strokes |
|---|---|---|---|---|---|
| base | 1 | 0.1022 | 28.8% | 51.2% | 59.5 |
| base | 2 | 0.1223 | 43.0% | 70.0% | 50.2 |
| seed | 1 | 0.1053 | 31.2% | 52.8% | 59.7 |
| seed | 2 | 0.1224 | 43.7% | 71.5% | 44.7 |
| image_label | 1 | 0.0717 | 6.8% | 16.8% | 74.6 |
| image_label | 2 | 0.0815 | 9.5% | 24.0% | 64.5 |
| image_only | 1 | 0.0666 | 5.0% | 12.2% | 149.7 |
| image_only | 2 | 0.0725 | 4.3% | 17.2% | 90.6 |
| image_only, shifted | 1 | 0.0871 | 11.0% | 26.7% | 114.5 |
| image_only, shifted | 2 | 0.0974 | 17.3% | 36.8% | 54.7 |
| no_guidance | 1 | 0.1103 | 32.7% | 58.7% | 44.8 |
| no_augment | 1 | 0.0926 | 17.3% | 37.5% | 75.8 |
| no_augment | 2 | 0.0645 | 6.2% | 17.7% | 3.4 |

Chance is 0.3% on either set. base and seed differ by at most 2.6 points at guidance 1 and 0.7 at guidance 2.

What to read from it:

- **Pictures alone do not carry over to text.** image_only told a text vector draws long scribbles (144 strokes a drawing against 21)
  that match no prompt. The shift recovers a third of the way on labels (22.3%) and less on phrases (17.3%).
- **Labels are enough for labels.** image_label is level with base on the 345 labels, inside the seed noise.
- **Captions are what make phrases work.** On phrases image_label falls to 9.5%, base holds 43.0%.
- The phrase set favours base a little: its prompts are held-out captions of a kind base trained on. The drawings and the captions
  themselves were never seen.
- Which of the 17 caption kinds matter was not tested. One or two may do.

## Held-out pen moves

Nats per pen move on the held-out drawings, each model told the vectors it trained with; lower is better.

| Run | Best step | Word score | Against base | Text rank of 16 | Train less test gap at step 48,000 |
|---|---|---|---|---|---|
| base | 54,000 | 0.5914 | | 1.88 | +0.067 |
| seed | 54,000 | 0.5901 | -0.0012 | 1.38 | +0.065 |
| no_guidance | 54,000 | 0.5910 | -0.0004 | 1.38 | +0.066 |
| image_label | 54,000 | 0.5956 | +0.0042 | 1.62 | +0.068 |
| image_only | 54,000 | 0.5800 | -0.0113 | 1.00 | +0.136 |
| no_augment | 9,000 | 0.6039 | +0.0125 | 1.12 | +1.901 |

By source, in the same order as `tools/ablate.py table sketch`:

| Run | controlsketch | creative | differsketching | icons | photosketch | quickdraw | seva | sketchy | things | tuberlin |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.5349 | 0.6538 | 0.6210 | 0.3248 | 0.7527 | 0.5202 | 0.6043 | 0.6559 | 0.6322 | 0.6141 |
| seed | 0.5334 | 0.6525 | 0.6196 | 0.3231 | 0.7519 | 0.5198 | 0.6038 | 0.6538 | 0.6314 | 0.6119 |
| no_guidance | 0.5338 | 0.6535 | 0.6215 | 0.3240 | 0.7528 | 0.5200 | 0.6042 | 0.6547 | 0.6322 | 0.6133 |
| image_label | 0.5343 | 0.6584 | 0.6247 | 0.3356 | 0.7577 | 0.5226 | 0.6091 | 0.6576 | 0.6396 | 0.6166 |
| image_only | 0.5294 | 0.6452 | 0.6158 | 0.3081 | 0.7478 | 0.5034 | 0.5875 | 0.6422 | 0.6174 | 0.6034 |
| no_augment | 0.5527 | 0.6967 | 0.6598 | 0.2905 | 0.7832 | 0.4681 | 0.6170 | 0.6864 | 0.6471 | 0.6374 |

These rows are not a like-for-like test of the captions: image_only is always told the picture, the most informative vector, while
base is told a mix. They do show that the picture vector alone gives the best pen-move score, which is why a picture-only model looks
strong until text is typed. no_augment's last-step score is 0.8946: it overfits from step 9,000 on.

## Why pictures alone fail: where the vectors lie

The model trained before these runs (`out/sketch8/sketch8/best.pt`, the base recipe) scored on 1,000 held-out drawings, 100 a source,
told one vector column at a time.

| Told | Nats per pen move |
|---|---|
| image | 0.5972 |
| dense | 0.6001 |
| frame | 0.6001 |
| sentence | 0.6006 |
| parts | 0.6008 |
| tags | 0.6009 |
| short | 0.6009 |
| ask | 0.6016 |
| phrase | 0.6017 |
| terse | 0.6022 |
| name | 0.6022 |
| outline | 0.6025 |
| verdict | 0.6036 |
| pose | 0.6036 |
| label | 0.6036 |
| vague | 0.6047 |
| kind | 0.6054 |
| manner | 0.6107 |
| nothing | 0.6121 |

The picture is worth 0.015 nats a move over nothing, the label 0.009, the best captions 0.012.

SigLIP 2 vectors of the same 1,000 drawings and their text:

| Text | Cosine to its own picture | To another drawing's picture | After centring each side | Own text nearest |
|---|---|---|---|---|
| label | 0.114 | 0.033 | 0.329 | 28.6% |
| frame | 0.193 | 0.050 | 0.387 | 56.5% |
| phrase | 0.134 | -0.000 | 0.355 | 43.8% |
| short | 0.147 | 0.002 | 0.365 | 50.9% |
| dense | 0.159 | 0.033 | 0.339 | 47.2% |
| sentence | 0.169 | 0.034 | 0.359 | 46.5% |

One picture's cosine to another picture is 0.726, the mean picture vector has length 0.852, and the mean picture's cosine to the mean
label is 0.044. Pictures of sketches sit in a narrow cone, nearly at right angles to every text vector, so a model that has only seen
that cone is far out of its training range when told text. The shift moves text into the cone but does not match its spread.

## Guidance, on the released model

`sketch.pt`, the 345 labels, four drawings each, temperature 0.6.

| Guidance | Cosine to its prompt | Its prompt ranked first | In the first five | Strokes |
|---|---|---|---|---|
| 1 | 0.1216 | 60.7% | 81.0% | 14.6 |
| 1.5 | 0.1299 | 74.3% | 90.1% | 11.7 |
| 2 | 0.1321 | 76.7% | 91.7% | 11.1 |
| 3 | 0.1328 | 77.9% | 91.8% | 9.3 |
| 4 | 0.1305 | 76.0% | 89.6% | 9.5 |

Guidance 2 gets nearly all of the gain. Training with label dropout to allow it costs nothing: no_guidance and base agree to 0.0004
nats on pen moves, and at guidance 1 no_guidance is 2.8 points ahead on labels, about the seed noise.

## Running it again

```
tools/captions.py prompts --out out/ablate_sketch
tools/ablate.py plan sketch --only base,seed,image_label,image_only,no_guidance,no_augment
tools/guidance.py --checkpoint RUN/best.pt --guidance 1,2 --n 2 --dataset data/quickdraw.sketch.parquet --prompts 345 --out RUN/score_labels
tools/guidance.py --checkpoint RUN/best.pt --guidance 1,2 --n 2 --dataset out/ablate_sketch/phrases.txt --prompts 300 --out RUN/score_phrases
tools/guidance.py ... --shift out/ablate_sketch/shift.npy --out RUN/score_labels_shift
tools/captions.py table --out out/ablate_sketch
tools/captions.py told --checkpoint RUN/best.pt
```

`pengpt.py` carries one flag for this, `--vec_columns`. The drawings behind every score are in `RUN/score_*/guidance_*.png`, 32 a grid.
