import glob

TEXT = "Sing, O Muse, of Odysseus|The quick brown fox|Ithaca, 1184 BC."
HAND = dict(showcase=TEXT, showcase_source="sam")
SKETCH = dict(
    tokens="data/sketch_tokens.json", vocab="data/*.sketch.parquet", holdout=50, max_words=1, max_seq_length=768,
    grid=0.0125, n_merges=1024,
    rotate=6.0, stretch=0.12, shear=0.10, perspective=0.12, warp=0.012, scale=0.10, shift=0.05,
    stroke_swap=0.15, stroke_reverse=0.15, loop_start=0.15, label_drop=0.1, hint=0.2,
    pen_pos_jitter=0, pen_rotary=True, batch_tokens=65_536, pad=16,
    showcase="cat|bicycle|The Eiffel Tower", keep=True)
HANDS = ("archive", "bigbank", "cedar", "csafe", "github", "loc", "notes")
EVERY = {"hand": ",".join(p for p in (f"data/{n}.hand.parquet" for n in HANDS) if glob.glob(p)),
         "sketch": ",".join(sorted(glob.glob("data/*.sketch.parquet")))}
BIGBANK = "data/bigbank.hand.parquet"

PRESETS = {
    "cursive": dict(HAND, dataset=BIGBANK, out_dir="out/cursive"),
    "mix": dict(HAND, dataset=EVERY["hand"], n_layer=8, n_embd=192, n_head=4, source_dropout=0.5, embedding_lr=0.06,
                batch_size=128, steps=60_000, sample_every=20_000, loader_workers=10, graphs=True, out_dir="out/mix"),
    "graves": dict(HAND, dataset=BIGBANK, arch="graves", n_layer=3, n_embd=400, learning_rate=1e-3, weight_decay=0.0,
                   compile=False, mixed=False, out_dir="out/graves"),
    "polar": dict(HAND, dataset=BIGBANK, tokenizer="polar", max_seq_length=1024, pen_pos_bands=0, out_dir="out/polar"),
    "quickdraw": dict(SKETCH, dataset="data/quickdraw.sketch.parquet", n_layer=12, n_embd=384, n_head=6, steps=14_400,
                      eval_every=1_200, sample_every=2_400, out_dir="out/quickdraw"),
    "sketch": dict(SKETCH, dataset=EVERY["sketch"], vectors="siglip", n_layer=12, n_embd=512, n_head=8, steps=144_000,
                   eval_every=2_000, sample_every=4_000, loader_workers=11, out_dir="out/sketch"),
}

PROFILES = {
    "m5": dict(hand=dict(preset="cursive"),
               sketch=dict(preset="quickdraw", n_layer=8, n_embd=256, n_head=4, steps=6_000, eval_every=1_000,
                           sample_every=2_000, batch_tokens=32_768)),
    "cpu": dict(like="m5", compile=False, hand=dict(steps=4_000),
                sketch=dict(n_layer=4, n_embd=128, steps=2_000, eval_every=500, sample_every=1_000, batch_tokens=16_384)),
    "a100": dict(hand=dict(preset="mix"), sketch=dict(preset="quickdraw")),
    "8xa100": dict(like="a100", gpus=8,
                   hand=dict(batch_size=256, steps=30_000, eval_every=1_250, out_dir="out/mix8"),
                   sketch=dict(preset="sketch", eval_every=1_500)),
    "h100": dict(like="a100"),
    "8xh100": dict(like="8xa100"),
}


def recipe(profile, model):
    p = PROFILES[profile]
    inherited = recipe(p["like"], model) if "like" in p else {}
    shared = {k: v for k, v in p.items() if k not in ("like", "hand", "sketch")}
    return {**inherited, **shared, **p.get(model, {})}
