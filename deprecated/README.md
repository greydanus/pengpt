# deprecated

Drawing corpora, parked while pengpt focuses on handwriting. Not maintained or tested.

- `quickdraw.py`: Quick, Draw! download, convert, and CLIP-probe ranking (`quickdraw_probe.npz`). Needs `transformers`, `Pillow`.
- `icons.py`: line-icon SVG sets to strokes. Needs `svgelements`.

Run from the repo root, e.g. `python -m deprecated.quickdraw rank --raw_dir qd_raw --out data/top25.jsonl`.
The old presets, which go in `utils.PRESETS` to use again:

```python
"quickdraw": dict(dataset="data/quickdraw_balanced_fixed.jsonl.gz", max_words=1, augment="general",
                  max_text_length=24, max_seq_length=192, n_layer=6, n_embd=128, batch_size=32, out_dir="out/quickdraw"),
"icons": dict(dataset="data/icons.jsonl", max_words=1, augment="general", grid=0.012, max_text_length=36,
              max_seq_length=128, tremor=0.004, n_layer=4, n_embd=96, batch_size=32, out_dir="out/icons"),
```

`check_data` in `pengpt.py` measures x-height, which rejects these normalized drawings; skip it to train them.
