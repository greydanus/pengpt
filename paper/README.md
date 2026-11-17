# The paper

`paper/render.sh` builds `pengpt.pdf` from what is checked in (it needs `tectonic`). `paper/build.sh` first redraws the overview
figure from a checkpoint under `out/` and rewrites the tables, then builds.

`paper/results` holds the numbers behind the tables: for each run the best and the last row of `metrics.jsonl`, and the score
files `tools/transfer.py`, `tools/legible.py` and `tools/guidance.py` wrote. `python paper/tables.py` works from it on any machine.

Run every script from the top of the repo.

| In the paper | Made by | Reads |
|---|---|---|
| `tables/*.tex`: ablations, the full handwriting table, guidance | `paper/tables.py` | `paper/results`, which it first refreshes from `out/` where the runs are present |
| Typed-prompt and vector tables in the appendix | written by hand from `docs/captions.md` | `tools/captions.py table`, `tools/captions.py told` |
| `figures/overview.pdf` | `paper/figures/overview.py` | `figures/overview.json`; `--fresh` remakes it from `out/ablate_hand/base/best.pt` and `data/loc.hand.parquet` |
| `figures/samples.pdf` | `paper/figures/samples.py` | `figures/unseen_hands.json`, `figures/samples_sketches.json`; `--fresh` recuts the sketches from `static/fidelity_ink.json` and `static/ood.json` |
| `figures/unseen_hands.json` | `paper/figures/unseen_hands.py` | `out/ablate_hand/base/best.pt`, `figures/unseen_picks.json` |
| `figures/transfer.pdf` | `paper/figures/transfer.py` | `figures/transfer.json`; `--fresh` writes the words again with `out/ablate_hand/base/best.pt` |
| `figures/heads.pdf` | `paper/figures/heads_plot.py` | `figures/heads.npz`, `heads.json`, `heads_examples.json`; `heads.py` and `--fresh` remake them from `out/ablate_hand/base/best.pt` |
| `figures/captions.pdf`, `figures/captions_top.pdf` (the first eight phrases, in the body) | `paper/figures/captions.py` | `figures/captions.json`; `--fresh` redraws it from `out/ablate_sketch/{base,image_label,image_only}/best.pt` |
| `figures/random_sketches.pdf` | `paper/figures/random_sketches.py` | `figures/random_sketches.json`; `--fresh` redraws it with `sketch.npz` |
| `figures/writers.png` | `tools/interp.py`, copied from `static/` | `hand.pt` |
| `figures/styles.png` | `tools/styles.py`, copied from `static/` | `hand.npz` |
| `figures/styles_plain.png`, `figures/writers_plain.png` | `paper/figures/crop.py` | the two above, with their own axis labels and title cut off |
| `figures/iliad.png` | copied from `static/`; no script in this repo | |
