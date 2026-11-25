import numpy as np
import pytest
import torch

from strokes import plot, save_strokes, to_strokes
from pengpt import (restore, augment, bend, DataConfig, ModelConfig, PenTransformer, PenDataset, Labels, create_datasets, check,
                    load_for_sampling, source_id, write, write_paragraph, SampleParams, ScribeTokenizer, ByteTokenizer, learn_merges,
                    IGNORE_INDEX, save_checkpoint, load_checkpoint, attach_pen_deltas, DIRECTIONS, DOWN, UP,
                    _DIR_LOOKUP, _stroke_spans)


def bresenham_steps(x0, y0, x1, y1):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    out, x, y = [], int(x0), int(y0)
    while x != x1 or y != y1:
        e2 = 2 * err
        step_x = step_y = 0
        if e2 > -dy:
            err -= dy; step_x = sx
        if e2 < dx:
            err += dx; step_y = sy
        x += step_x; y += step_y
        out.append(_DIR_LOOKUP[step_x + 1, step_y + 1])
    return np.array(out, dtype=np.int64)


def item(ds, i):
    return tuple(t[0] for t in ds.batch([i]))


def save_words(rows, path):
    save_strokes([{"id": f"{r['writer']}/{k}", "label": r["text"], "writer": r["writer"], "strokes": r["strokes"]} for k, r in enumerate(rows)],
                 path, 2e-3, "words")


def check_data(examples, grid):
    return check([e["text"] for e in examples], lambda i: examples[i]["points"], grid)


def make_word(n=40, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=0.02, size=(n, 2)) + [0.01, 0.0]
    points = np.cumsum(steps, axis=0)
    pen = np.ones(n)
    pen[n // 2] = 0
    return np.column_stack([points, pen])


def test_bresenham_lands_on_endpoint():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.integers(-40, 40, 2), rng.integers(-40, 40, 2)
        steps = bresenham_steps(a[0], a[1], b[0], b[1])
        end = a + DIRECTIONS[steps].sum(0) if len(steps) else a
        assert tuple(end) == tuple(b)
        assert all(0 <= s < 8 for s in steps)


def test_encode_decode_roundtrip():
    st = ScribeTokenizer(grid=0.01)
    word = make_word()
    tokens = st.encode_word(word)
    assert tokens.min() >= 0 and tokens.max() < st.vocab_size
    out = st.decode_words(np.concatenate([tokens, [st.END]]))
    assert len(out) == 1
    orig = word[word[:, 2] == 1][:, :2]
    rec = out[0][out[0][:, 2] == 1][:, :2]
    orig, rec = orig - orig[0], rec - rec[0]
    d = np.hypot(orig[:, None, 0] - rec[None, :, 0], orig[:, None, 1] - rec[None, :, 1])
    assert d.min(1).mean() < 3 * st.grid


def test_decoded_strokes_are_curves_that_keep_their_points_their_ends_and_their_sharp_tips():
    from inference import pen as mini
    from pengpt import _smooth_strokes
    from strokes import curves
    st, rng = ScribeTokenizer(grid=0.02), np.random.default_rng(0)
    for _ in range(30):
        tokens = np.concatenate([[DOWN, *(np.cumsum(rng.choice([-1, 0, 0, 0, 1], rng.integers(0, 60))) % 8), UP, 0, 0] for _ in range(5)])
        raw, out = st.decode_word(tokens, smooth=False), st.decode_word(tokens)
        assert out.shape == raw.shape and (out[:, 2] == raw[:, 2]).all() and np.abs(out[:, :2] - raw[:, :2]).max() < 6 * st.grid
        for a, b in _stroke_spans(raw[:, 2]):
            assert np.allclose(out[[a, b - 1], :2], raw[[a, b - 1], :2]) or np.hypot(*(raw[a, :2] - raw[b - 1, :2])) < st.grid
        assert np.array_equal(out, mini.curves(_smooth_strokes(raw), st.grid))
    stem = st.decode_word([DOWN, *[6] * 10, *[2] * 10, UP])
    assert np.ptp(stem[:, 1]) > 9.3 * st.grid and np.abs(stem[:, 0]).max() < 1e-9
    ring = [d for d in (0, 1, 2, 3, 4, 5, 6, 7) for _ in range(4)]
    oval = st.decode_word([DOWN, *ring, UP])[:-1, :2]
    turn = np.diff(np.unwrap(np.arctan2(*np.diff(oval, axis=0).T[::-1])))
    assert np.abs(turn).max() < 0.5 and np.allclose(oval[0], oval[-1])
    dot = np.array([[0.1, 0.2, 1], [0.1, 0.2, 0]])
    assert np.array_equal(curves(dot, 0.02), dot)


def test_baseline_height_survives_roundtrip():
    st = ScribeTokenizer(grid=0.01)
    high = make_word(seed=1)
    high[:, 1] -= 0.20
    low = make_word(seed=1)
    for word in (high, low):
        decoded = st.decode_words(np.concatenate([st.encode_word(word), [st.END]]))[0]
        ink_in = word[word[:, 2] == 1][:, 1]
        ink_out = decoded[decoded[:, 2] == 1][:, 1]
        assert abs(ink_out.min() - ink_in.min()) < 5 * st.grid
        assert abs(ink_out.max() - ink_in.max()) < 5 * st.grid


def test_pen_state_structure():
    st = ScribeTokenizer(grid=0.01)
    tokens = list(st.encode_word(make_word()))
    assert tokens.count(DOWN) == tokens.count(UP) == 2
    assert tokens.index(DOWN) < tokens.index(UP)


def test_merges_are_lossless():
    st = ScribeTokenizer(grid=0.01)
    words = [make_word(seed=s) for s in range(20)]
    base = [st.encode_word(w) for w in words]
    merges = learn_merges(base, n_merges=64, min_count=2)
    st2 = ScribeTokenizer(grid=0.01, merges=merges)
    assert st2.vocab_size > st.vocab_size
    for w, b in zip(words, base):
        merged = st2.encode_word(w)
        assert len(merged) <= len(b)
        assert list(st2.expand(merged)) == list(b)


def test_merges_apply_in_learned_order():
    st = ScribeTokenizer(grid=0.01, merges=[(0, 1, 10), (2, 0, 11)])
    assert list(st.apply_merges(np.array([2, 0, 1]))) == [2, 10]


def test_merges_match_sequential_application():
    st0 = ScribeTokenizer(grid=0.01)
    words = [make_word(n=60, seed=s) for s in range(20)]
    base = [st0.encode_word(w) for w in words]
    merges = learn_merges(base, n_merges=128, min_count=2)
    st = ScribeTokenizer(grid=0.01, merges=merges)

    def sequential(seq):
        s = [int(t) for t in seq]
        for a, b, c in merges:
            out, j = [], 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                    out.append(c); j += 2
                else:
                    out.append(s[j]); j += 1
            s = out
        return s

    for b in base:
        assert list(st.apply_merges(b)) == sequential(b)


def test_word_separation_and_end():
    st = ScribeTokenizer(grid=0.01)
    words = [make_word(seed=s) for s in range(3)]
    tokens = np.concatenate([np.r_[[st.WORD] * 2 * bool(i), st.encode_word(w)] for i, w in enumerate(words)]).astype(int)
    tokens = np.concatenate([tokens, [st.END], tokens[:7], [st.PAD] * 5])
    assert len(st.decode_words(tokens)) == 3


def curve(n, freq=3.0, amp=0.2):
    t = np.linspace(0, 1, n)
    return np.column_stack([t, amp * np.sin(freq * t), np.ones(n)])


def test_sampling_invariance_when_samples_do_not_skip_cells():
    st = ScribeTokenizer(grid=0.05)
    t = np.linspace(0, 1, 120)
    at_cell_scale = np.column_stack([t, 0.5 * t, np.ones(len(t))])
    t2 = np.linspace(0, 1, 1200)
    ten_times_denser = np.column_stack([t2, 0.5 * t2, np.ones(len(t2))])
    assert np.array_equal(st.encode_word(at_cell_scale),
                          st.encode_word(ten_times_denser))


def test_sampling_density_changes_tokens_when_grid_is_fine():
    st = ScribeTokenizer(grid=0.008)
    assert not np.array_equal(st.encode_word(curve(60)), st.encode_word(curve(600)))


def test_sampling_density_stays_close_at_the_default_grid():
    st = ScribeTokenizer(grid=0.012)
    a = st.decode_words(np.concatenate([st.encode_word(curve(60)), [st.END]]))[0]
    b = st.decode_words(np.concatenate([st.encode_word(curve(600)), [st.END]]))[0]
    a, b = a[a[:, 2] == 1][:, :2], b[b[:, 2] == 1][:, :2]
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    assert 0.5 * (d.min(1).mean() + d.min(0).mean()) < 2 * st.grid


def test_scale_changes_token_count_proportionally():
    st = ScribeTokenizer(grid=0.01)
    word = make_word(n=60)
    small = len(st.encode_word(word * np.array([1.0, 1.0, 1.0])))
    large = len(st.encode_word(word * np.array([2.0, 2.0, 1.0])))
    assert 1.6 * small < large < 2.5 * small


def test_decode_survives_random_tokens():
    st = ScribeTokenizer(grid=0.01)
    rng = np.random.default_rng(0)
    for _ in range(20):
        junk = rng.integers(0, st.vocab_size, size=rng.integers(1, 200))
        words = st.decode_words(junk)
        assert isinstance(words, list)
        for w in words:
            assert w.ndim == 2 and w.shape[1] == 3


def test_empty_and_degenerate_input():
    st = ScribeTokenizer(grid=0.01)
    assert len(st.encode_word(np.zeros((0, 3)))) == 0
    pen_up_only = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
    assert len(st.encode_word(pen_up_only)) == 0
    single = np.array([[0.0, 0.0, 1.0]])
    assert len(st.decode_words(np.concatenate([st.encode_word(single), [st.END]]))) <= 1


def test_one_bend_moves_every_word_of_a_sequence_together_and_keeps_the_baseline():
    cfg = DataConfig()
    word = make_word()
    how = bend(cfg, np.random.default_rng(7))
    a, b = augment(word, cfg, None, False, how), augment(word + [0.3, 0, 0], cfg, None, False, how)
    c = augment(word, cfg, np.random.default_rng(7), False)
    assert np.array_equal(a, c) and not np.allclose(a[:, :2], word[:, :2]) and np.abs(a[:, :2] - word[:, :2]).max() < 0.2
    assert np.allclose((b - a)[:, 1], (b - a)[0, 1], atol=1e-9) and np.allclose((b - a)[:, 0], (b - a)[0, 0], atol=1e-9)
    assert np.array_equal(a[:, 2], word[:, 2]) and len(a) == len(word)


def test_works_without_word_structure():
    t = np.linspace(0, 4 * np.pi, 200)
    shapes = []
    for s in range(6):
        r = np.random.default_rng(s)
        xy = np.column_stack([np.cos(t) * 0.1 + r.normal(0, 0.005, 200),
                              np.sin(2 * t) * 0.1 + r.normal(0, 0.005, 200)])
        pen = np.ones(200); pen[100] = 0
        shapes.append(np.column_stack([xy, pen]))
    labels = [f"shape{s}" for s in range(6)]

    cfg = DataConfig(max_seq_length=512, max_words=1, max_text_length=20, grid=0.012)
    st = ScribeTokenizer(grid=cfg.grid)
    st = ScribeTokenizer(grid=cfg.grid,
                         merges=learn_merges([st.encode_word(p) for p in shapes],
                                             64, min_count=2))

    ds = PenDataset(shapes, labels, np.arange(6), st, Labels(cfg), cfg, augment=False)

    x, c, _, _ = item(ds, 0)
    assert (x != st.PAD).sum() > 20
    assert st.WORD not in set(x.tolist())
    assert len(st.decode_words(x.numpy())) == 1


def test_byte_tokenizer():
    ct = ByteTokenizer()
    ids = ct.encode("bad café", length=12)
    assert ids.shape == (12,) and ids[-1] == ct.PAD and ids.max() < ct.vocab_size
    assert ids[len("bad café".encode())] == ct.END
    assert ct.decode(ids) == "bad café"
    assert ct.decode(ct.encode("Kmxqngyo?", length=5)) == "Kmxq"
    old = ByteTokenizer(end=False)
    assert old.vocab_size == 257 and ct.END not in old.encode("bad", length=6)
    assert old.decode(old.encode("Kmxqngyo?", length=4)) == "Kmxq"


def test_old_checkpoints_load_without_text_end():
    assert restore({"max_text_length": 50}).text_end is False
    assert DataConfig().text_end is True


@pytest.fixture
def tiny_dataset():
    cfg = DataConfig(max_seq_length=256, max_text_length=20, max_words=3,
                     grid=0.01, seed=0)
    bank = [make_word(seed=s) for s in range(6)]
    texts = ["cab", "face", "bag", "dad", "egg", "fed"]
    st = ScribeTokenizer(grid=cfg.grid)
    return PenDataset(bank, texts, np.arange(6), st, Labels(cfg), cfg, augment=True)


def test_dataset_item(tiny_dataset):
    x, c, y, _ = item(tiny_dataset, 0)
    st, cfg = tiny_dataset.stroke_tok, tiny_dataset.cfg
    assert x.shape == (cfg.max_seq_length,) and c.shape == (cfg.max_text_length + cfg.text_end,)
    assert x[0] == st.BOS
    end = (x == st.END).nonzero().item()
    assert y[end - 1] == st.END
    assert (y[end:] == IGNORE_INDEX).all()
    assert (x[end + 1:] == st.PAD).all()
    assert set(tiny_dataset.told.tok.decode(c).split()) <= set(tiny_dataset.bank_texts)


def test_targets_are_inputs_shifted_by_one(tiny_dataset):
    x, _, y, _ = item(tiny_dataset, 0)
    st = tiny_dataset.stroke_tok
    end = (x == st.END).nonzero().item()
    assert torch.equal(y[:end], x[1:end + 1])


def test_generation_prefix_occurs_in_training(tiny_dataset):
    st = tiny_dataset.stroke_tok
    firsts = {item(tiny_dataset, i)[0][0].item() for i in range(8)}
    assert firsts == {st.BOS}


def test_dataset_never_truncates_mid_word(tiny_dataset):
    st = tiny_dataset.stroke_tok
    for i in range(8):
        x, _, _, _ = item(tiny_dataset, i)
        end = (x == st.END).nonzero()
        assert len(end) == 1, "exactly one END expected"
        assert end.item() < tiny_dataset.cfg.max_seq_length


def test_truncated_word_does_not_supervise_end():
    cfg = DataConfig(max_seq_length=16, max_text_length=8, max_words=1, grid=0.005)
    st = ScribeTokenizer(grid=cfg.grid)
    long_word = make_word(n=200)
    ds = PenDataset([long_word], ["cab"], np.arange(1), st, Labels(cfg), cfg, augment=False)
    x, _, y, _ = item(ds, 0)
    end = (x == st.END).nonzero().item()
    assert end == cfg.max_seq_length - 1
    assert y[end - 1] == IGNORE_INDEX
    assert (y[:end - 1] != IGNORE_INDEX).all()


def test_bank_smaller_than_max_words():
    cfg = DataConfig(max_seq_length=256, max_text_length=20, max_words=8,
                     grid=0.01, seed=0)
    st, ct = ScribeTokenizer(grid=0.01), ByteTokenizer()
    for n_bank in (1, 2, 7):
        bank = [make_word(seed=s) for s in range(n_bank)]
        texts = ["cab", "face", "bag", "dad", "egg", "fed", "gag"][:n_bank]
        ds = PenDataset(bank, texts, np.arange(n_bank), st, Labels(cfg), cfg, augment=False)
        x, c, _, _ = item(ds, 0)
        assert (x != st.PAD).sum() > 2
        assert len(ct.decode(c).split()) <= n_bank


def test_model_forward_and_generate(tiny_dataset):
    st, ct, cfg = tiny_dataset.stroke_tok, tiny_dataset.told.tok, tiny_dataset.cfg
    mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size,
                       block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                       context_block_size=cfg.max_text_length + 1)
    model = PenTransformer(mcfg)
    x, c, y, _ = (t.unsqueeze(0) for t in item(tiny_dataset, 0))
    logits, loss = model(x, c, y)
    assert logits.shape == (1, cfg.max_seq_length, st.vocab_size)
    assert torch.isfinite(loss)

    out = model.generate(x[:, :10], c, max_new_tokens=20, end_token=st.END, pad_token=st.PAD)
    assert out.shape[1] <= 30
    assert st.decode_words(out[0].numpy()) is not None


def test_checkpoint_roundtrip(tiny_dataset, tmp_path):
    st, ct, cfg = tiny_dataset.stroke_tok, tiny_dataset.told.tok, tiny_dataset.cfg
    mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size,
                       block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                       context_block_size=cfg.max_text_length + 1)
    model = PenTransformer(mcfg)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, {"dataset": "x"}, st.merges)
    reloaded, ckpt = load_checkpoint(str(path))
    assert reloaded.cfg == model.cfg
    for k, v in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[k], v)

    x, c, y, _ = (t.unsqueeze(0) for t in item(tiny_dataset, 0))
    model.eval(), reloaded.eval()
    with torch.inference_mode():
        a, _ = model(x, c, y)
        b, _ = reloaded(x, c, y)
    assert torch.allclose(a, b)


def _encode_reference(points, grid):
    xy = np.rint(points[:, :2] / grid).astype(int)
    out, prev = [], (0, 0)
    for start, stop in _stroke_spans(points[:, 2]):
        path = [prev] + [tuple(p) for p in xy[start:stop]]
        for i, ((x0, y0), (x1, y1)) in enumerate(zip(path[:-1], path[1:])):
            if (x0, y0) != (x1, y1):
                out.extend(int(t) for t in bresenham_steps(x0, y0, x1, y1))
            if i == 0:
                out.append(DOWN)
        out.append(UP)
        prev = path[-1]
    return out


def test_vectorized_tokenizer_matches_reference():
    rng = np.random.default_rng(7)
    st = ScribeTokenizer(grid=0.02)
    for trial in range(200):
        k = rng.integers(1, 60)
        steps = rng.integers(-1, 2, size=(k, 2)) * 0.02
        steps[rng.random(k) < 0.1] *= rng.integers(2, 9)
        points = np.c_[np.cumsum(steps, axis=0), rng.random(k) < 0.8]
        assert st.encode_base(points).tolist() == _encode_reference(points, st.grid), trial


def test_token_deltas_track_pen_position():
    words = [make_word(n=60, seed=s) for s in range(10)]
    st0 = ScribeTokenizer(grid=0.01)
    merges = learn_merges([st0.encode_word(w) for w in words], 64, min_count=2)
    st = ScribeTokenizer(grid=0.01, merges=merges)
    deltas = st.token_deltas()
    rng = np.random.default_rng(0)
    points = np.column_stack([np.cumsum(rng.normal(0, 0.05, (30, 2)), axis=0),
                              np.ones(30)])
    points[9, 2] = 0
    tokens = st.encode_word(points)
    summed = deltas[tokens].sum(axis=0)
    grid_end = np.rint(points[-1, :2] / st.grid).astype(int)
    assert (summed == grid_end).all()


def test_pen_pos_features_are_causal_and_checkpoint_safe(tiny_dataset, tmp_path):
    st, ct, cfg = tiny_dataset.stroke_tok, tiny_dataset.told.tok, tiny_dataset.cfg
    mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, pen_pos_bands=6,
                       vocab_size=st.vocab_size, block_size=cfg.max_seq_length,
                       context_vocab_size=ct.vocab_size,
                       context_block_size=cfg.max_text_length + 1)
    model = PenTransformer(mcfg)
    attach_pen_deltas(model, st)
    model.eval()
    x, c, y, _ = (t.unsqueeze(0) for t in item(tiny_dataset, 0))
    logits, loss = model(x, c, y)
    assert torch.isfinite(loss)

    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % st.vocab_size
    logits2, _ = model(x2, c, y)
    assert torch.allclose(logits[0, :-1], logits2[0, :-1], atol=1e-5)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, {"dataset": "x"}, st.merges)
    reloaded, _ = load_checkpoint(str(path))
    assert (reloaded.pen_deltas == model.pen_deltas).all()

    mcfg0 = ModelConfig(n_layer=2, n_head=2, n_embd=32, pen_pos_bands=0,
                        vocab_size=st.vocab_size, block_size=cfg.max_seq_length,
                        context_vocab_size=ct.vocab_size,
                        context_block_size=cfg.max_text_length + 1)
    logits0, _ = PenTransformer(mcfg0)(x, c, y)
    assert logits0.shape == logits.shape


@pytest.mark.parametrize("variant", [{}, {"pen_rotary": True}])
def test_cached_generation_matches_full_recompute(tiny_dataset, variant):
    st, ct, cfg = tiny_dataset.stroke_tok, tiny_dataset.told.tok, tiny_dataset.cfg
    mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, n_sources=3, **variant, vocab_size=st.vocab_size,
                       block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                       context_block_size=cfg.max_text_length + 1)
    torch.manual_seed(0)
    model = PenTransformer(mcfg)
    for p in model.parameters():
        p.data.normal_(0, 0.3)
    attach_pen_deltas(model, st)
    model.eval()
    x, c, _, _ = (t[:2] for t in tiny_dataset.batch(range(2)))
    w = torch.tensor([1, 2])
    n = int((x[0] == st.WORD).nonzero()[0]) + 3 if variant else 3
    assert not variant or (x[0, :n] == st.WORD).any()
    prefix = x[:, :n]
    cached = model.generate(prefix, c, max_new_tokens=40, do_sample=False, source=w)
    full = prefix
    with torch.inference_mode():
        for _ in range(40):
            logits, _ = model(full, c, source=w)
            full = torch.cat([full, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(cached, full)


def test_mixing_weights_and_single_writer_sequences():
    cfg = DataConfig(max_seq_length=256, max_text_length=20, max_words=3, grid=0.01, seed=0)
    bank = [make_word(seed=s) for s in range(8)]
    texts = ["ab", "cd", "ef", "ga", "bc", "de", "fa", "gb"]
    writer_ids = [1, 1, 1, 1, 2, 2, 2, 2]
    weights = [10 / 4] * 4 + [1 / 4] * 4
    ds = PenDataset(bank, texts, np.arange(8), ScribeTokenizer(grid=0.01), Labels(cfg), cfg, augment=False, source_ids=writer_ids, weights=weights, sources=["", "a", "b"])
    seen = []
    for i in range(400):
        _, c, _, w = item(ds, i)
        words = ByteTokenizer().decode(c).split()
        assert {writer_ids[texts.index(t)] for t in words} == {int(w)}
        seen.append(int(w))
    assert 0.85 < np.mean(np.array(seen) == 1) < 0.96


def test_packed_text_never_exceeds_prompt_length():
    cfg = DataConfig(max_seq_length=512, max_text_length=12, max_words=8, grid=0.05, seed=0)
    bank = [make_word(n=10, seed=s) for s in range(8)]
    texts = ["word" + str(s) for s in range(8)]
    ds = PenDataset(bank, texts, np.arange(8), ScribeTokenizer(grid=0.05), Labels(cfg), cfg, augment=False)
    for i in range(50):
        text = ByteTokenizer().decode(item(ds, i)[1])
        assert 0 < len(text.encode()) <= cfg.max_text_length and len(text.split()) >= 1


def test_writer_vector_changes_output(tiny_dataset):
    st, cfg = tiny_dataset.stroke_tok, tiny_dataset.cfg
    model = PenTransformer(ModelConfig(n_layer=1, n_head=2, n_embd=32, n_sources=3, vocab_size=st.vocab_size,
                                       block_size=cfg.max_seq_length, context_vocab_size=258,
                                       context_block_size=cfg.max_text_length + 1))
    for p in model.parameters():
        p.data.normal_(0, 0.3)
    model.eval()
    x, c, _, _ = (t.unsqueeze(0) for t in item(tiny_dataset, 0))
    a, _ = model(x, c, source=torch.tensor([1]))
    b, _ = model(x, c, source=torch.tensor([2]))
    unknown, _ = model(x, c)
    assert not torch.allclose(a, b) and torch.isfinite(unknown).all()


def tiny_ink(path, n, writer_names, text="ab"):
    rows = [{"text": f"{text}{i % 7}", "strokes": [np.c_[np.linspace(0, 0.3, 20), 0.1 * np.sin(np.linspace(0, 3, 20) + i)].tolist()],
             "style": "s", "writer": writer_names[i % len(writer_names)], "source": "t", "license": "MIT"} for i in range(n)]
    save_words(rows, str(path))


def test_create_datasets_mixes_weighted_datasets(tmp_path):
    tiny_ink(tmp_path / "a.hand.parquet", 40, ["sam"])
    tiny_ink(tmp_path / "b.hand.parquet", 200, ["w1", "w2", ""], text="cd")
    cfg = DataConfig(dataset=f"{tmp_path / 'a.hand.parquet'}:9,{tmp_path / 'b.hand.parquet'}:1", max_seq_length=256,
                     max_text_length=20, max_words=3, grid=0.02, n_merges=8)
    train, test, _ = create_datasets(cfg)
    assert train.sources == ["", "sam", "w1", "w2"]
    assert len(train.indices) == 38 + 190 and len(test.indices) == 2 + 10
    sam = [k for k, g in enumerate(train.groups) if train.source_ids[g[0]] == 1]
    assert len(sam) == 1 and np.isclose(train.group_p[sam[0]], 38 * 9 / (38 * 9 + 190))


def test_load_for_sampling_and_generate(tmp_path, tiny_dataset):
    st, cfg = tiny_dataset.stroke_tok, tiny_dataset.cfg
    model = PenTransformer(ModelConfig(n_layer=1, n_head=2, n_embd=32, n_sources=2, vocab_size=st.vocab_size,
                                       block_size=cfg.max_seq_length, context_vocab_size=258,
                                       context_block_size=cfg.max_text_length + 1))
    attach_pen_deltas(model, st)
    from dataclasses import asdict
    save_checkpoint(str(tmp_path / "m.pt"), model, asdict(cfg), st.merges, sources=["", "sam"])
    loaded, ds, _, _ = load_for_sampling(str(tmp_path / "m.pt"))
    assert source_id(ds, "sam") == 1 and source_id(ds, "") == 0
    words, tokens = write(loaded, ds, "ab", SampleParams(max_tokens=30, source="sam"))
    assert isinstance(words, list) and len(words) == len(tokens)
    para = write_paragraph(loaded, ds, "ab cd ef ga bc", SampleParams(max_tokens=200, source="sam",
                                                                         n_at_a_time=1, burn_in=1))
    assert len(para) == 5


def fake_words(n=60, x_height=0.12, baseline=0.0, text="mann", seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = np.linspace(0, 1, 60)
        y = baseline - x_height * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * rng.uniform(0.9, 1.1)
        out.append({"text": text, "points": np.c_[t * 0.6, y, np.ones(60)]})
    return out


def test_check_data_accepts_style_and_rejects_bad_scale():
    assert check_data(fake_words(), 0.02)[1] == []
    assert check_data(fake_words(x_height=0.12 * 1.4), 0.02)[1] == []
    assert check_data(fake_words(x_height=0.12 * 3), 0.02)[1]
    assert check_data(fake_words(x_height=0.12 * 0.4), 0.02)[1]
    assert check_data(fake_words(baseline=0.15), 0.02)[1]
    caps = fake_words(x_height=0.2, text="MANN")
    assert check_data(caps, 0.02)[0]["x_height"] is None and check_data(caps, 0.02)[1] == []
    assert check_data(fake_words(x_height=2.0, text="MANN"), 0.02)[1]


def test_trimmed_eval_matches_full_length_loss(tiny_dataset):
    from pengpt import trim, IGNORE_INDEX
    st, ct, cfg = tiny_dataset.stroke_tok, tiny_dataset.told.tok, tiny_dataset.cfg
    mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size, block_size=cfg.max_seq_length,
                       context_vocab_size=ct.vocab_size, context_block_size=cfg.max_text_length + 1)
    torch.manual_seed(0)
    model = PenTransformer(mcfg).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.3)
    X, C, Y, W = tiny_dataset.batch(range(4))
    Xt, Ct, Yt, Wt = trim((X, C, Y, W), st.PAD)
    assert Xt.shape[1] < X.shape[1] and (Yt != IGNORE_INDEX).sum() == (Y != IGNORE_INDEX).sum()
    with torch.inference_mode():
        assert torch.allclose(model(X, C, Y)[1], model(Xt, Ct, Yt)[1], atol=1e-5)


def test_generate_drops_a_word_cut_off_by_the_block(tmp_path, tiny_dataset):
    st, cfg = tiny_dataset.stroke_tok, tiny_dataset.cfg
    model = PenTransformer(ModelConfig(n_layer=1, n_head=2, n_embd=32, n_sources=2, vocab_size=st.vocab_size,
                                       block_size=cfg.max_seq_length, context_vocab_size=258,
                                       context_block_size=cfg.max_text_length + 1))
    attach_pen_deltas(model, st)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.weight[0] = 1.0
    words, tokens = write(model, tiny_dataset, "ab cd", SampleParams(max_tokens=20, do_sample=False))
    assert words == [] and tokens == []
    para = write_paragraph(model, tiny_dataset, "ab cd ef", SampleParams(max_tokens=20, do_sample=False))
    assert len(para) == 3 and all(len(w) == 0 for w in para)


def test_a_one_point_dot_is_drawn():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = ScribeTokenizer(grid=0.02)
    word = np.array([[0, 0, 1], [0.1, -0.05, 1], [0.2, 0, 1], [0.2, 0, 0], [0.26, 0.0, 1], [0.26, 0.0, 0]], float)
    pts = st.decode_words(np.r_[st.encode_word(word), st.END])[0]
    fig, ax = plt.subplots()
    plot(ax, to_strokes(pts))
    assert any(len(l.get_xdata()) == 1 for l in ax.lines) and len(ax.lines) == 2
    plt.close(fig)


def test_batches_are_cut_to_their_longest_sequence_and_keep_every_token(tiny_dataset):
    from pengpt import infinite_batches
    st, block = tiny_dataset.stroke_tok, tiny_dataset.cfg.max_seq_length
    batches = infinite_batches(tiny_dataset, 4, seed=0)
    widths = set()
    for _ in range(16):
        X, C, Y, W = next(batches)
        widths.add(X.shape[1])
        assert X.shape == Y.shape and len(X) == len(C) == len(W) == 4 and X.shape[1] % 64 == 0 and X.shape[1] <= block
        assert ((X == st.END).sum(1) == 1).all() and (X != st.PAD).sum(1).max() > X.shape[1] - 64
    assert len(widths) > 1


def test_each_dataset_keeps_its_own_test_part_and_two_word_labels_are_dropped(tmp_path):
    tiny_ink(tmp_path / "a.hand.parquet", 40, ["sam"])
    tiny_ink(tmp_path / "b.hand.parquet", 200, ["w1", "w2"], text="c d")
    tiny_ink(tmp_path / "c.hand.parquet", 100, ["w3"], text="ef")
    cfg = DataConfig(dataset=",".join(str(tmp_path / f"{k}.hand.parquet") for k in "ac"), max_text_length=20, grid=0.02, n_merges=8)
    _, test, _ = create_datasets(cfg)
    assert {k: len(d.indices) for k, d in test.parts.items()} == {"a": 2, "c": 5} and len(test.indices) == 7
    assert {test.bank_texts[i][:2] for i in test.parts["c"].indices} == {"ef"}
    with pytest.raises(SystemExit):
        create_datasets(DataConfig(dataset=str(tmp_path / "b.hand.parquet"), max_text_length=20, grid=0.02, n_merges=8))


def test_held_out_writers_never_train_and_are_scored_as_the_unknown_writer(tmp_path):
    import zlib
    from pengpt import unseen_score
    names = [f"w{k}" for k in range(60)]
    tiny_ink(tmp_path / "a.hand.parquet", 600, names)
    tiny_ink(tmp_path / "b.hand.parquet", 60, ["sam"], text="cd")
    cfg = DataConfig(dataset=f"{tmp_path / 'a.hand.parquet'},{tmp_path / 'b.hand.parquet'}", max_text_length=20, grid=0.02, n_merges=8)
    train, test, st = create_datasets(cfg)
    ct = train.told.tok
    held = {w for w in names if zlib.crc32(w.encode()) % 20 == 0}
    assert held and not held & set(train.sources) and "sam" in train.sources and list(test.unseen) == ["a"]
    assert len(test.unseen["a"].groups) == len(held) and len(test.unseen["a"].indices) == 10 * len(held)
    model = PenTransformer(ModelConfig(n_layer=1, n_head=2, n_embd=32, n_sources=len(train.sources), vocab_size=st.vocab_size,
                                       block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                                       context_block_size=cfg.max_text_length + 1))
    alone, one, three = unseen_score(model, test.unseen["a"], "cpu")
    assert np.isfinite([alone, one, three]).all() and len({alone, one, three}) == 3
    assert not create_datasets(DataConfig(dataset=cfg.dataset, max_text_length=20, grid=0.02, n_merges=8, holdout=0))[1].unseen


def test_one_process_training_is_untouched_by_the_multi_gpu_path(tiny_dataset):
    from pengpt import TrainConfig, distribute, sync_grads
    cfg = TrainConfig(steps=800, warmup=200)
    cfg.device = "cpu"
    assert distribute(cfg) == "cpu" and (cfg.steps, cfg.warmup, cfg.matrix_lr) == (800, 200, TrainConfig.matrix_lr)
    st, c = tiny_dataset.stroke_tok, tiny_dataset.cfg
    model = PenTransformer(ModelConfig(n_layer=1, n_head=2, n_embd=32, vocab_size=st.vocab_size, block_size=c.max_seq_length,
                                       context_vocab_size=258, context_block_size=c.max_text_length + 1))
    x, ctx, y, _ = (t.unsqueeze(0) for t in item(tiny_dataset, 0))
    model(x, ctx, y)[1].backward()
    before = [p.grad.clone() for p in model.parameters()]
    sync_grads(model)
    assert all(torch.equal(a, p.grad) for a, p in zip(before, model.parameters()))


def test_a_profile_loads_its_preset_and_explicit_flags_win(monkeypatch):
    import pengpt
    seen = []
    monkeypatch.setattr(pengpt, "train", lambda d, m, t: seen.append((d, m, t)))
    pengpt.main(["train", "--profile", "8xa100", "--steps", "500"])
    pengpt.main(["train", "--profile", "m5"])
    pengpt.main(["train", "--profile", "a100", "--preset", "cursive"])
    (d8, m8, t8), (d5, m5, t5), (da, ma, ta) = seen
    assert (t8.gpus, t8.steps, m8.n_embd, t8.batch_size, t8.graphs) == (8, 500, 192, 256, True)
    assert (t5.gpus, m5.n_embd, d5.dataset) == (1, 64, "data/bigbank.hand.parquet")
    assert (ta.gpus, ma.n_embd, da.dataset) == (1, 64, "data/bigbank.hand.parquet")
    with pytest.raises(SystemExit):
        pengpt.main(["train", "--profile", "cray"])


def test_a_machine_is_another_machine_with_a_few_changes(monkeypatch):
    import pengpt
    import utils
    assert utils.recipe("8xh100", "hand") == utils.recipe("8xa100", "hand") and utils.recipe("h100", "sketch") == utils.recipe("a100", "sketch")
    assert utils.recipe("8xa100", "sketch") == dict(preset="sketch", gpus=8, eval_every=1_500)
    assert utils.recipe("8xa100", "hand") == dict(preset="mix", gpus=8, batch_size=256, steps=30_000, eval_every=1_250, out_dir="out/mix8")
    cpu, m5 = utils.recipe("cpu", "sketch"), utils.recipe("m5", "sketch")
    assert cpu["preset"] == m5["preset"] == "quickdraw" and cpu["n_head"] == m5["n_head"] and cpu["compile"] is False and cpu["n_embd"] == 128
    monkeypatch.setitem(utils.PROFILES, "8xb200", dict(like="8xh100", sketch=dict(batch_tokens=131_072)))
    seen = []
    monkeypatch.setattr(pengpt, "train", lambda d, m, t: seen.append((d, m, t)))
    pengpt.main(["train", "--profile", "8xb200", "--model", "sketch", "--dataset", "a.sketch.parquet", "--vectors", ""])
    pengpt.main(["train", "--profile", "cpu", "--model", "sketch"])
    (d, m, t), (dc, mc, tc) = seen
    assert (t.gpus, t.batch_tokens, t.eval_every, m.n_embd, d.max_seq_length) == (8, 131_072, 1_500, 512, 768)
    assert (mc.n_layer, mc.n_embd, mc.n_head, tc.steps, tc.compile, dc.grid) == (4, 128, 4, 2_000, False, 0.0125)
