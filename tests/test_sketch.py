import json
import os
from dataclasses import fields

import numpy as np
import pytest
import torch

import pengpt as P
from pengpt import augment, centred, densify
from strokes import save_strokes as save_sketches
from strokes import to_points
from strokes import to_strokes as strokes_of
from utils import SKETCH

BASE = {k: v for k, v in SKETCH.items() if k in {f.name for f in fields(P.DataConfig)}}
ARGS = [a for k, v in SKETCH.items() if k not in ("tokens", "vocab", "showcase", "keep", "pen_rotary") for a in (f"--{k}", str(v))] + ["--pen_rotary", "--keep"]


def SketchConfig(**kw):
    return P.DataConfig(**{**BASE, "dataset": "a.sketch.parquet", **kw})


def save_column(path, name, column, vectors, model):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from strokes import StrokeFile
    from vectors import fields as columns_of
    from vectors import order, said, table_path
    f, out = StrokeFile(path), table_path(path)
    old = pq.read_table(out) if os.path.exists(out) else None
    kept = {n: old[n] for n in old.column_names[1:]} if old else {}
    told = json.loads(old.schema.metadata[b"said"]) if old else {}
    a = np.rint(vectors / np.maximum(np.abs(vectors).max(-1, keepdims=True), 1e-12) * 127).astype(np.int8)
    for k in range(a.shape[1] if a.ndim == 3 else 1):
        part = np.ascontiguousarray(a[:, k] if a.ndim == 3 else a)
        kept[f"{column}_{k + 1}" if a.ndim == 3 else column] = pa.FixedSizeListArray.from_arrays(pa.array(part.ravel()), part.shape[1])
    told[column] = said(f, column)
    table = pa.table({"id": pa.array([f.tables["id"][j] for j in f.cols["id"]]), **kept})
    meta = {"encoder": model, "order": str(order(f)), "said": json.dumps({c: told[c] for c in columns_of(table.schema)})}
    pq.write_table(table.replace_schema_metadata(meta), out, row_group_size=64)


def house():
    return [np.array([[0, 1], [0, 0.4], [1, 0.4], [1, 1], [0, 1.0]]), np.array([[0, 0.4], [0.5, 0], [1, 0.4]]), np.array([[0.4, 1], [0.4, 0.7]])]


def test_densify_keeps_the_path_and_the_pen():
    points = to_points(house())
    dense = densify(points, 0.05)
    a, b = strokes_of(points), strokes_of(dense)
    assert len(a) == len(b) and (dense[:, 2] == 0).sum() == 3
    for s, d in zip(a, b):
        assert np.allclose(s[0], d[0]) and np.allclose(s[-1], d[-1])
        assert np.hypot(*np.diff(d, axis=0).T).max() <= 0.05 + 1e-6
        assert np.isclose(np.hypot(*np.diff(s, axis=0).T).sum(), np.hypot(*np.diff(d, axis=0).T).sum())


def test_no_augmentation_only_centres_within_half_a_cell():
    cfg = SketchConfig(augment=0.0)
    out = augment(centred(house()), cfg, np.random.default_rng(0))
    centre = (out[:, :2].min(0) + out[:, :2].max(0)) / 2
    assert np.abs(centre).max() <= cfg.grid / 2 + 1e-9
    assert np.allclose(np.ptp(out[:, :2], 0), 1.0)


def test_strokes_that_touch_still_touch_and_shapes_stay_close():
    cfg = SketchConfig()
    for seed in range(50):
        s = strokes_of(augment(centred(house()), cfg, np.random.default_rng(seed)))
        assert np.allclose(s[0][0], s[0][-1]) and np.allclose(s[1][0], s[0][np.argmin(np.hypot(*(s[0] - s[1][0]).T))])
        size = np.ptp(np.concatenate(s), 0)
        assert 0.7 < size.min() and size.max() < 1.45


def test_augmentation_is_set_by_the_seed_and_shifts_the_drawing():
    cfg = SketchConfig()
    a, b = (augment(centred(house()), cfg, np.random.default_rng(3)) for _ in range(2))
    assert np.array_equal(a, b)
    centres = np.array([(lambda p: (p.min(0) + p.max(0)) / 2)(augment(centred(house()), cfg, np.random.default_rng(k))[:, :2]) for k in range(200)])
    assert 0.02 < np.abs(centres).max() <= cfg.shift + cfg.grid / 2 + 1e-9


@pytest.mark.parametrize("pen_rotary", [False, True])
def test_cached_generation_matches_the_forward_pass(pen_rotary):
    torch.manual_seed(0)
    tok = P.ScribeTokenizer(grid=0.0125)
    cfg = P.ModelConfig(n_layer=2, n_head=2, n_embd=32, pen_pos_jitter=0, pen_rotary=pen_rotary, vocab_size=tok.vocab_size, block_size=24,
                             context_vocab_size=258, context_block_size=8)
    model = P.PenTransformer(cfg)
    torch.nn.init.normal_(model.head.weight, std=0.5)
    P.attach_pen_deltas(model, tok)
    context = torch.randint(1, 200, (2, 8))
    out = model.generate(torch.full((2, 1), tok.BOS), context, 20, do_sample=False)
    model.eval()
    with torch.inference_mode():
        logits, _ = model(out[:, :-1], context)
    assert torch.equal(logits.argmax(-1), out[:, 1:])


def test_pen_rotary_sees_where_the_pen_is():
    t, xy = torch.arange(3), torch.tensor([[[0, 0], [5, 0], [5, 0]]])
    cos, _ = P.pen_rotary_tables(t, xy, 32)
    assert cos.shape == (1, 1, 3, 16)
    assert torch.allclose(cos[0, 0, 1, 8:], cos[0, 0, 2, 8:]) and not torch.allclose(cos[0, 0, 0, 8:], cos[0, 0, 1, 8:])


def files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    for name, n in (("a", 300), ("b", 200)):
        save_sketches([{"id": f"{name}/{i}", "label": ("cat", "dog")[i % 2], "strokes": [s + rng.normal(0, 0.01, s.shape) for s in house()]}
                       for i in range(n)], f"{name}.sketch.parquet")
    return SketchConfig(dataset="a.sketch.parquet,b.sketch.parquet:2", vocab="*.sketch.parquet", tokens="tokens.json", n_merges=16,
                        max_seq_length=512)


def test_datasets_share_one_vocabulary_and_hold_drawings_out(tmp_path, monkeypatch):
    cfg = files(tmp_path, monkeypatch)
    train, test, tok = P.create_datasets(cfg)
    chars = train.told.tok
    assert train.sources == ["", "a", "b"] and set(test.parts) == {"a", "b"}
    assert not set(train.indices) & set(test.indices) and len(train.indices) + sum(len(d.indices) for d in test.parts.values()) == 500
    X, C, Y, W = train.batch(range(32))
    assert X.shape == (32, 512) and (X[:, 0] == tok.BOS).all() and set(W.tolist()) == {1, 2}
    assert {chars.decode(c) for c in C} == {"cat", "dog", ""}
    assert (Y == tok.END).sum() == 32
    again = P.create_datasets(cfg)[2]
    assert again.merges == tok.merges
    with pytest.raises(SystemExit):
        P.create_datasets(SketchConfig(**{**cfg.__dict__, "grid": 0.02}))


def test_a_drawing_survives_the_tokens(tmp_path, monkeypatch):
    cfg = files(tmp_path, monkeypatch)
    _, test, tok = P.create_datasets(cfg)
    X = test.batch([0])[0][0].numpy()
    back = np.vstack(tok.decode(X))
    assert np.allclose(np.ptp(back[:, :2], 0), 1.0, atol=0.05) and np.abs(back[:, :2]).max() < 0.55


def test_training_runs_and_resumes(tmp_path, monkeypatch):
    files(tmp_path, monkeypatch)
    args = ["train", "--dataset", "a.sketch.parquet,b.sketch.parquet", "--vocab", "*.sketch.parquet", "--tokens", "tokens.json", "--n_merges", "16",
            "--max_seq_length", "512", "--n_layer", "1", "--n_embd", "32", "--n_head", "2", "--batch_size", "4", "--eval_every", "2",
            "--sample_every", "2", "--showcase", "cat", "--device", "cpu", "--no-compile", "--out_dir", "run"]
    P.main(args[:1] + ARGS + args[1:] + ["--steps", "2"])
    P.main(args[:1] + ARGS + args[1:] + ["--steps", "4", "--resume", "run/last.pt"])
    P.main(["sample", "--checkpoint", "run/best.pt", "--text", "cat|dog", "--n", "2", "--out", "s.png", "--device", "cpu"])
    assert (tmp_path / "s.png").exists() and (tmp_path / "run" / "progress").exists()
    rows = [json.loads(line) for line in open(tmp_path / "run" / "metrics.jsonl")]
    assert all(np.isfinite(r["gap"]) and set(r["seen_loss_by"]) == set(r["test_loss_by"]) == {"a", "b"} for r in rows)
    assert all(abs(r["gap"] - np.mean([r["test_loss_by"][k] - r["seen_loss_by"][k] for k in "ab"])) < 1e-9 for r in rows)


def test_a_token_budget_gives_short_drawings_bigger_batches_and_loses_none():
    class Fake:
        class stroke_tok:
            PAD = 0

        def __init__(self):
            self.served = 0

        def batch(self, indices):
            rng = np.random.default_rng(self.served)
            n = np.minimum(rng.geometric(0.01, len(indices)) + 5, 256)
            X = np.zeros((len(indices), 256), np.int64)
            for row, k in zip(X, n):
                row[:k] = 1
            ids = np.arange(self.served, self.served + len(indices))
            self.served += len(indices)
            return tuple(torch.from_numpy(a) for a in (X, ids, X.copy(), ids))

    class Stop(Exception):
        pass

    class Q:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)
            if len(self.items) > 400:
                raise Stop

    q, data = Q(), Fake()
    try:
        P._produce(data, 64, 0, q, tokens=4096)
    except Stop:
        pass
    batches = [b for b in q.items if not isinstance(b, Exception)]
    sizes = {X.shape for X, _, _, _ in batches}
    assert all(rows == 4096 // width and width % 64 == 0 for rows, width in sizes) and len(sizes) == 4
    assert all((X[:, -64:] != 0).any() for X, _, _, _ in batches)
    seen = np.concatenate([c for _, c, _, _ in batches])
    assert len(set(seen)) == len(seen) and data.served - len(seen) < 4096 // 64 + 8 * 64


def many():
    return [np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                      [0, 0.0]])] + [np.array([[0.2, 0.2 + 0.1 * k], [0.5, 0.2 + 0.1 * k], [0.8, 0.25 + 0.1 * k]]) for k in range(6)]


def test_swaps_reversals_and_loop_starts_keep_every_point():
    cfg = SketchConfig(stroke_swap=0.5, stroke_reverse=0.5, loop_start=1.0)
    before = centred(many())
    key = lambda p: sorted(map(tuple, np.round(np.asarray(p, float)[p[:, 2] == 1][:, :2], 4).tolist()))
    orders = set()
    for seed in range(40):
        out = P.alter(before, cfg, np.random.default_rng(seed))
        s = strokes_of(out)
        assert len(s) == 7 and len(s[0]) == 5 and np.allclose(s[0][0], s[0][-1])
        assert set(key(out)) == set(key(before))
        orders.add(tuple(round(float(x[:, 1].mean()), 3) for x in s[1:]))
    assert len(orders) > 5


def test_dropped_labels_reach_the_text_in_training_only(tmp_path, monkeypatch):
    cfg = files(tmp_path, monkeypatch)
    cfg = SketchConfig(**{**cfg.__dict__, "label_drop": 0.5})
    train, test, tok = P.create_datasets(cfg)
    chars = train.told.tok
    texts = {chars.decode(c) for c in train.batch(range(64))[1]}
    assert texts == {"", "cat", "dog"}
    assert {chars.decode(c) for c in test.batch(range(8))[1]} <= {"cat", "dog"}


def test_ink_and_stroke_hints_are_given_a_fifth_of_the_time_and_never_on_held_out_drawings(tmp_path, monkeypatch):
    cfg = files(tmp_path, monkeypatch)
    assert P.hint_digits(4.6, 17) == [0, 9, 1,
                                              7] and P.hint_digits() == [None] * 4 and P.hint_digits(1e9, 500) == [9, 9, 9, 9]
    assert P.hint_digits(None, 3) == [None, None, 0, 3]
    ink, strokes = P.measure(centred(house()))
    assert strokes == 3 and 4.5 < ink < 5.2
    train, test, tok = P.create_datasets(cfg)
    chars = train.told.tok
    assert chars.vocab_size == 298
    C = train.batch(range(400))[1].numpy()
    assert C.shape[1] == 51 and 0.1 < (C[:, -4] > 0).mean() < 0.3 and 0.1 < (C[:, -1] > 0).mean() < 0.3 and ((C[:, -4] > 0) == (C[:, -3] > 0)).all()
    assert all(set(C[:, -4 + k]) <= {0, *range(258 + 10 * k, 268 + 10 * k)}
               for k in range(4)) and {chars.decode(c[:-4]) for c in C} == {"cat", "dog", ""}
    assert (test.batch(range(16))[1].numpy()[:, -4:] == 0).all()
    test.ask = (4.6, 17)
    assert test.encode_text("cat")[-4:].tolist() == [258, 277, 279, 295]
    assert P.restore({"grid": 0.0125}).hint == 0 and P.restore({"grid": 0.0125}).label_drop == 0


def test_a_drawing_that_runs_out_of_tokens_is_still_drawn():
    st = P.ScribeTokenizer(grid=0.0125)
    data = P.PenDataset([], [], [], st, P.Labels(cfg := SketchConfig(label_drop=0.0, hint=0.0)), cfg)

    class Model:
        tail = []

        def parameters(self):
            yield torch.zeros(1)

        def generate(self, idx, context, max_new_tokens, **kw):
            return torch.cat([idx, torch.tensor([[8, 0, 0, 2, 2, 9, 8, 4, 4] + self.tail])], 1)

    cut, params = Model(), P.SampleParams(max_tokens=12)
    assert len(P.sketch(cut, data, "cat", params)) == 9
    cut.tail = [9, st.END]
    assert len(P.sketch(cut, data, "cat", params)) == 10


def test_a_model_is_told_one_vector_the_mean_when_there_are_two(tmp_path, monkeypatch):
    from strokes import add_text
    files(tmp_path, monkeypatch)
    rng, D = np.random.default_rng(0), 8
    for name, n in (("a", 300), ("b", 200)):
        add_text(f"{name}.sketch.parquet", {k: [f"{k} {i}" for i in range(n)] for k in ("name", "frame", "short", "steps")})
        for column, shape in (("image", (n, D)), ("label", (n, D)), ("name", (n, D)), ("frame", (n, D)), ("short", (n, D)), ("steps", (n, 2, D))):
            v = rng.normal(size=shape).astype(np.float32)
            if column == "steps":
                v[::2, 1] = 0
            save_column(f"{name}.sketch.parquet", "siglip", column, v, "fake")
    base = dict(dataset="a.sketch.parquet,b.sketch.parquet", vocab="*.sketch.parquet", tokens="tokens.json", n_merges=16, vectors="siglip")
    train, test, tok = P.create_datasets(SketchConfig(**base))
    assert set(train.told.columns) == {"image", "label", "name", "frame", "short", "steps"}
    C = train.batch(range(600))[1].numpy()
    N = len(P.KINDS)
    told, kinds, flags = C[:, :D], C[:, D:D + N], C[:, D + N:]
    there = np.abs(told).max(-1) > 0
    assert set(kinds.sum(1)) == {0.0, 1.0, 2.0} and not kinds[~there].any() and 0.75 < kinds[there].any(1).mean() < 0.95
    assert not kinds[:, [P.KINDS.index(k) for k in P.KINDS if k not in train.told.columns]].any()
    assert C.shape == (600, D + N + 40) and set(flags.sum(1)) == {0.0, 2.0, 4.0} and 0.03 < (~there).mean() < 0.2
    assert np.allclose(np.linalg.norm(told[there], axis=-1), 1, atol=1e-3)
    frames = np.stack([train.told.columns["frame"][i] for i in range(500)])
    E = test.batch(range(16))[1].numpy()
    assert (E[:, D:D + N].argmax(1) == P.KINDS.index("frame")).all() and (E[:, D:D + N].sum(1) == 1).all()
    assert all((np.abs(frames - e).max(1) < 1e-4).any() for e in E[:, :D])

    class Words:
        def texts(self, texts):
            return np.stack([np.eye(D, dtype=np.float32)[len(t) % D] for t in texts])
    monkeypatch.setattr(P.Vectors, "encoder", Words())
    args = ["train", "--dataset", "a.sketch.parquet,b.sketch.parquet", "--vocab", "*.sketch.parquet", "--tokens", "tokens.json", "--n_merges", "16",
            "--vectors", "siglip",
            "--n_layer", "1", "--n_embd", "32", "--n_head", "2", "--batch_tokens", "0", "--batch_size", "4", "--eval_every", "2",
            "--sample_every", "2",
            "--showcase", "cat", "--device", "cpu", "--no-compile", "--out_dir", "run", "--steps", "2"]
    args = args[:1] + ARGS + args[1:]
    P.main(args)
    model, data, _ = P.load_for_sampling("run/last.pt")[:3]
    assert model.cfg.context_slots == 1 and model.cfg.context_width == D and model.cfg.context_kinds == N and model.cfg.context_dim == D + N + 40
    assert data.encode_text("a cat")[:D].any() and not data.encode_text("").any()
    assert np.array_equal(data.encode_text(data.encode_text("a cat")[:D]), data.encode_text("a cat"))
    c, mask = model.encode_context(torch.from_numpy(np.stack([data.encode_text("a cat"), data.encode_text("")])))
    assert c.shape == (2, 1 + 4 + 1, 32) and mask[0, 0, 0].tolist() == [True] * 5 + [False] and mask[1, 0, 0].sum() == 1
    P.main(["sample", "--checkpoint", "run/last.pt", "--text", "cat|dog", "--n", "2", "--out", "s.png", "--device", "cpu"])
    assert (tmp_path / "s.png").exists()
    P.main([*args, "--out_dir", "run3", "--vec_kind", "0"])
    model, data, _ = P.load_for_sampling("run3/last.pt")[:3]
    assert model.cfg.context_kinds == 0 and model.cfg.context_dim == D + 40 and not hasattr(model, "ctx_kind") and len(data.encode_text("a cat")) == D + 40
