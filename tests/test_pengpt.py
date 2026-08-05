import numpy as np
import pytest
import torch

from pengpt import (DataConfig, ModelConfig, PenTransformer, PenDataset,
                    ScribeTokenizer, CharTokenizer, learn_merges)
from pengpt.data import IGNORE_INDEX, augment_word, resample
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
