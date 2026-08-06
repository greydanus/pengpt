import numpy as np
import pytest
import torch

from pengpt import (DataConfig, ModelConfig, PenTransformer, PenDataset,
                    ScribeTokenizer, CharTokenizer, learn_merges)
from pengpt.data import IGNORE_INDEX, prepare_word, resample
from pengpt.tokenizer import bresenham_steps, DIRECTIONS, DOWN, UP


def make_word(n=40, seed=0):
    """A synthetic word: a smooth random walk with one pen lift in the middle."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=0.02, size=(n, 2)) + [0.01, 0.0]
    points = np.cumsum(steps, axis=0)
    pen = np.ones(n)
    pen[n // 2] = 0  # one pen-up move
    return np.column_stack([points, pen])


def test_bresenham_lands_on_endpoint():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.integers(-40, 40, 2), rng.integers(-40, 40, 2)
        steps = bresenham_steps(a[0], a[1], b[0], b[1])
        end = a + DIRECTIONS[steps].sum(0) if len(steps) else a
        assert tuple(end) == tuple(b)
        assert all(0 <= s < 8 for s in steps)      # only direction tokens


def test_encode_decode_roundtrip():
    st = ScribeTokenizer(grid=0.01)
    word = make_word()
    tokens = st.encode_word(word)
    assert tokens.min() >= 0 and tokens.max() < st.vocab_size
    out = st.decode(np.concatenate([tokens, [st.END]]))
    assert len(out) == 1
    # reconstruction stays within a couple of grid cells of the original
    orig = word[word[:, 2] == 1][:, :2]
    rec = out[0][out[0][:, 2] == 1][:, :2]
    orig, rec = orig - orig[0], rec - rec[0]
    d = np.hypot(orig[:, None, 0] - rec[None, :, 0], orig[:, None, 1] - rec[None, :, 1])
    assert d.min(1).mean() < 3 * st.grid


def test_baseline_height_survives_roundtrip():
    """A word's height above the baseline is carried by the tokens.

    Words that start at the cap line (digits) and words that start on the
    baseline must stay distinguishable after decoding, with no per-alphabet
    table to tell them apart.
    """
    st = ScribeTokenizer(grid=0.01)
    high = make_word(seed=1)
    high[:, 1] -= 0.20
    low = make_word(seed=1)
    for word in (high, low):
        decoded = st.decode(np.concatenate([st.encode_word(word), [st.END]]))[0]
        ink_in = word[word[:, 2] == 1][:, 1]
        ink_out = decoded[decoded[:, 2] == 1][:, 1]
        assert abs(ink_out.min() - ink_in.min()) < 5 * st.grid
        assert abs(ink_out.max() - ink_in.max()) < 5 * st.grid


def test_pen_state_structure():
    """Every stroke is delimited by DOWN ... UP, and only by those."""
    st = ScribeTokenizer(grid=0.01)
    tokens = list(st.encode_word(make_word()))
    assert tokens.count(DOWN) == tokens.count(UP) == 2   # two strokes, one lift
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
        assert len(merged) <= len(b)                    # never longer
        assert list(st2.expand(merged)) == list(b)      # and exactly reversible


def test_word_separation_and_end():
    st = ScribeTokenizer(grid=0.01)
    words = [make_word(seed=s) for s in range(3)]
    tokens = st.encode_words(words)
    # append END plus garbage; decode must ignore everything after END
    tokens = np.concatenate([tokens, [st.END], tokens[:7], [st.PAD] * 5])
    assert len(st.decode(tokens)) == 3


def curve(n, freq=3.0, amp=0.2):
    t = np.linspace(0, 1, n)
    return np.column_stack([t, amp * np.sin(freq * t), np.ones(n)])


def test_sampling_invariance_when_samples_do_not_skip_cells():
    """Identical tokens for two samplings of one shape.

    Invariance requires consecutive samples to land in the same or an adjacent
    grid cell. Then both recordings walk the same cells in the same order and
    Bresenham has no gap to bridge. Sampling more finely than that changes
    nothing, which is the property that lets one model see data from recorders
    of different rates.
    """
    st = ScribeTokenizer(grid=0.05)
    t = np.linspace(0, 1, 120)
    at_cell_scale = np.column_stack([t, 0.5 * t, np.ones(len(t))])
    t2 = np.linspace(0, 1, 1200)
    ten_times_denser = np.column_stack([t2, 0.5 * t2, np.ones(len(t2))])
    assert np.array_equal(st.encode_word(at_cell_scale),
                          st.encode_word(ten_times_denser))


def test_sampling_density_changes_tokens_when_grid_is_fine():
    """The converse, pinned down so the limit is not forgotten."""
    st = ScribeTokenizer(grid=0.008)
    assert not np.array_equal(st.encode_word(curve(60)), st.encode_word(curve(600)))


def test_sampling_density_stays_close_at_the_default_grid():
    """Even without exact invariance, both encodings track the same curve."""
    st = ScribeTokenizer(grid=0.012)
    a = st.decode(np.concatenate([st.encode_word(curve(60)), [st.END]]))[0]
    b = st.decode(np.concatenate([st.encode_word(curve(600)), [st.END]]))[0]
    a, b = a[a[:, 2] == 1][:, :2], b[b[:, 2] == 1][:, :2]
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    assert 0.5 * (d.min(1).mean() + d.min(0).mean()) < 2 * st.grid


def test_scale_changes_token_count_proportionally():
    """Token cost tracks ink size, which is why datasets must be normalized."""
    st = ScribeTokenizer(grid=0.01)
    word = make_word(n=60)
    small = len(st.encode_word(word * np.array([1.0, 1.0, 1.0])))
    large = len(st.encode_word(word * np.array([2.0, 2.0, 1.0])))
    assert 1.6 * small < large < 2.5 * small


def test_decode_survives_random_tokens():
    """Model output early in training is noise; decoding must not raise."""
    st = ScribeTokenizer(grid=0.01)
    rng = np.random.default_rng(0)
    for _ in range(20):
        junk = rng.integers(0, st.vocab_size, size=rng.integers(1, 200))
        words = st.decode(junk)
        assert isinstance(words, list)
        for w in words:
            assert w.ndim == 2 and w.shape[1] == 3


def test_empty_and_degenerate_input():
    st = ScribeTokenizer(grid=0.01)
    assert len(st.encode_word(np.zeros((0, 3)))) == 0
    pen_up_only = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
    assert len(st.encode_word(pen_up_only)) == 0
    single = np.array([[0.0, 0.0, 1.0]])
    assert len(st.decode(np.concatenate([st.encode_word(single), [st.END]]))) <= 1


def test_prepare_word_is_deterministic_and_optional():
    cfg = DataConfig(spacing=0.0)
    word = make_word(n=50)
    plain = prepare_word(word, cfg)
    assert np.array_equal(plain, word)          # no rng means no change

    a = prepare_word(word, cfg, np.random.default_rng(7))
    b = prepare_word(word, cfg, np.random.default_rng(7))
    assert np.array_equal(a, b)                 # same seed, same augmentation
    assert not np.array_equal(a, word)
    assert np.array_equal(a[:, 2], word[:, 2])  # pen states untouched


def test_works_without_word_structure():
    """Data that is not handwriting: one trajectory per example, no words.

    A sketch or a signature has no word boundaries, so the WORD token and the
    packing loop must be inert rather than in the way. Nothing here should need
    a code path of its own -- only max_words=1.
    """
    rng = np.random.default_rng(0)
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
    ct = CharTokenizer(" " + "".join(sorted(set("".join(labels)) - {" "})))
    ds = PenDataset(shapes, labels, np.arange(6), st, ct, cfg, length=6,
                    augment=False, name="sketch")

    x, c, _ = ds[0]
    assert (x != st.PAD).sum() > 20
    assert st.WORD not in set(x.tolist())      # no word separators at all
    assert len(st.decode(x.numpy())) == 1      # one trajectory back


def test_char_tokenizer():
    ct = CharTokenizer(" abcdefgh")
    ids = ct.encode("bad cafe", length=12)
    assert ids.shape == (12,) and ids[-1] == ct.PAD
    assert ct.decode(ids) == "bad cafe"
    assert ct.decode(ct.encode("abz")) == "ab"  # unknown chars become PAD, ending decode


def test_resample_is_uniform_and_keeps_lifts():
    word = make_word(n=80)
    out = resample(word, 0.02)
    assert (out[:, 2] == 0).sum() == (word[:, 2] == 0).sum()   # lifts survive

    # On a straight path, arc and chord coincide, so spacing is exactly uniform.
    line = np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.ones(50)])
    steps = np.diff(resample(line, 0.05)[:, 0])
    assert np.allclose(steps[:-1], 0.05, atol=1e-6)

    # On a curved path the chords are shorter than the spacing, never longer.
    first_lift = np.flatnonzero(out[:, 2] == 0)[0]
    chords = np.hypot(*np.diff(out[:first_lift, :2], axis=0).T)
    assert chords.max() <= 0.02 + 1e-9

    # denser spacing yields more points
    assert len(resample(word, 0.01)) > len(resample(word, 0.04))


@pytest.fixture
def tiny_dataset():
    cfg = DataConfig(max_seq_length=256, max_text_length=20, max_words=3,
                     grid=0.01, seed=0)
    bank = [make_word(seed=s) for s in range(6)]
    texts = ["cab", "face", "bag", "dad", "egg", "fed"]
    st = ScribeTokenizer(grid=cfg.grid)
    ct = CharTokenizer(" abcdefg")
    return PenDataset(bank, texts, np.arange(6), st, ct, cfg, length=8,
                      augment=True, name="tiny")


def test_dataset_item(tiny_dataset):
    x, c, y = tiny_dataset[0]
    st, cfg = tiny_dataset.stroke_tok, tiny_dataset.cfg
    assert x.shape == (cfg.max_seq_length,) and c.shape == (cfg.max_text_length,)
    end = (x == st.END).nonzero().item()          # exactly one END in the input
    assert y[end] == st.END                       # targets end with END...
    assert (y[end + 1:] == IGNORE_INDEX).all()    # ...then loss is masked
    assert (x[end + 1:] == st.PAD).all()
    assert tiny_dataset.char_tok.decode(c) == tiny_dataset.text_for(0)


def test_dataset_never_truncates_mid_word(tiny_dataset):
    """Packing stops before overflow, so a word is never cut in half."""
    st = tiny_dataset.stroke_tok
    for i in range(len(tiny_dataset)):
        x, _, _ = tiny_dataset[i]
        end = (x == st.END).nonzero()
        assert len(end) == 1, "exactly one END expected"
        assert end.item() < tiny_dataset.cfg.max_seq_length


def test_model_forward_and_generate(tiny_dataset):
    st, ct, cfg = tiny_dataset.stroke_tok, tiny_dataset.char_tok, tiny_dataset.cfg
    mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size,
                       block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                       context_block_size=cfg.max_text_length)
    model = PenTransformer(mcfg)
    x, c, y = (t.unsqueeze(0) for t in tiny_dataset[0])
    logits, loss = model(x, c, y)
    assert logits.shape == (1, cfg.max_seq_length, st.vocab_size)
    assert torch.isfinite(loss)

    out = model.generate(x[:, :10], c, max_new_tokens=20, end_token=st.END, pad_token=st.PAD)
    assert out.shape[1] <= 30
    assert st.decode(out[0].numpy()) is not None  # arbitrary output decodes safely
