import numpy as np
import pytest
import torch

from pengpt import (DataConfig, ModelConfig, PenTransformer, PenDataset,
                    StrokeTokenizer, CharTokenizer, word_to_offsets, offsets_to_points)
from pengpt.data import IGNORE_INDEX, augment_word, downsample_word, make_combos


def make_word(n=40, seed=0):
    """A synthetic word: a smooth random walk with one pen lift in the middle."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=0.02, size=(n, 2)) + [0.01, 0.0]
    points = np.cumsum(steps, axis=0)
    pen = np.ones(n)
    pen[n // 2] = 0  # one pen-up move
    return np.column_stack([points, pen])


def test_offsets_roundtrip():
    word = make_word()
    points = offsets_to_points(word_to_offsets(word))
    assert np.allclose(points[:, :2], word[:, :2] - word[0, :2], atol=1e-9)
    assert np.array_equal(points[:, 2], word[:, 2])


def test_stroke_tokenizer_roundtrip():
    st = StrokeTokenizer()
    word = make_word()
    offsets = word_to_offsets(word)
    decoded = st.decode(st.encode_word(offsets))
    assert len(decoded) == 1
    out = decoded[0]
    assert out.shape == offsets.shape
    assert np.array_equal(out[:, 2], offsets[:, 2])                     # pen state exact
    assert np.all(np.abs(out[:, 1] - offsets[:, 1]) < 2 * np.pi / 219)  # theta within a bin
    assert np.all(np.abs(out[:, 0] - offsets[:, 0]) < 0.0021 + 0.025 * offsets[:, 0])


def test_word_separation_and_end():
    st = StrokeTokenizer()
    words = [word_to_offsets(make_word(seed=s)) for s in range(3)]
    tokens = st.encode_words(words)
    # append END and post-END garbage; decode should ignore both
    tokens = np.concatenate([tokens, [st.END], tokens[:7], [st.PAD] * 5])
    decoded = st.decode(tokens)
    assert len(decoded) == 3
    for original, out in zip(words, decoded):
        assert len(out) == len(original)


def test_char_tokenizer():
    ct = CharTokenizer(" abcdefgh")
    ids = ct.encode("bad cafe", length=12)
    assert ids.shape == (12,) and ids[-1] == ct.PAD
    assert ct.decode(ids) == "bad cafe"
    assert ct.decode(ct.encode("abz")) == "ab"  # unknown chars become PAD, which ends decoding


def test_downsample_keeps_endpoints():
    word = make_word(n=60)
    rng = np.random.default_rng(0)
    out = downsample_word(word, fraction=0.7, drop_prob=0.05, rng=rng)
    assert len(out) < len(word)
    assert np.allclose(out[0], word[0]) and np.allclose(out[-1], word[-1])
    assert (out[:, 2] == 0).sum() == (word[:, 2] == 0).sum()  # pen lifts survive


@pytest.fixture
def tiny_dataset():
    cfg = DataConfig(max_seq_length=256, max_text_length=20, num_words=2, seed=0)
    bank_points = [make_word(seed=s) for s in range(6)]
    bank_texts = ["cab", "face", "bag", "dad", "egg", "fed"]
    ct = CharTokenizer(" abcdefg")
    st = StrokeTokenizer()
    combos = make_combos(6, 8, 2, np.random.default_rng(0))
    return PenDataset(bank_points, bank_texts, combos, st, ct, cfg, augment=True, name="tiny")


def test_dataset_item(tiny_dataset):
    x, c, y = tiny_dataset[0]
    st, cfg = tiny_dataset.stroke_tok, tiny_dataset.cfg
    assert x.shape == (cfg.max_seq_length,) and c.shape == (cfg.max_text_length,)
    end = (x == st.END).nonzero().item()          # exactly one END in the input
    assert y[end - 1] == st.END                   # targets end with END...
    assert (y[end:] == IGNORE_INDEX).all()        # ...then loss is masked
    assert (x[end + 1:] == st.PAD).all()
    assert tiny_dataset.char_tok.decode(c) == tiny_dataset.text_for(0)


def test_augmentation_is_deterministic(tiny_dataset):
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = augment_word(tiny_dataset.bank_points[0], tiny_dataset.cfg, rng1)
    b = augment_word(tiny_dataset.bank_points[0], tiny_dataset.cfg, rng2)
    assert np.array_equal(a, b)


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
