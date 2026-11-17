import numpy as np
import torch

from pengpt import DataConfig, ModelConfig, ScribeTokenizer, build, create_datasets, fit, attach_pen_deltas, style_rank, stroke_tokenizer
from test_hand import make_word, tiny_ink


def test_polar_tokens_decode_to_the_ink_they_encode():
    tok = stroke_tokenizer(DataConfig(tokenizer="polar"))
    word = make_word(60)
    word[:, :2] -= word[0, :2]
    tokens = tok.encode_word(word)
    kept, back = tok.thin(word), tok.decode_word(tokens)
    assert len(tokens) == 2 * len(kept) and tokens.max() < tok.PAD
    assert np.abs(back[:, :2] - kept[:, :2]).max() < 0.02 and (back[:, 2] == (kept[:, 2] == 1)).all()
    assert tok.moves(word) == len(ScribeTokenizer().encode_base(word))


def test_polar_runs_through_the_dataset_and_the_model(tmp_path):
    tiny_ink(tmp_path / "a.hand.parquet", 80, ["w1", "w2"])
    cfg = DataConfig(dataset=str(tmp_path / "a.hand.parquet"), tokenizer="polar", max_text_length=20, max_seq_length=128)
    train, _, st = create_datasets(cfg)
    X, C, Y, W = train.batch(range(4))
    assert X.max() < st.vocab_size and st.merges == []
    model = build(fit(ModelConfig(n_layer=1, n_head=2, n_embd=32, pen_pos_bands=0), train))
    attach_pen_deltas(model, st)
    assert torch.isfinite(model(X, C, Y, W)[1])


def test_graves_steps_one_token_at_a_time_as_it_does_all_at_once(tmp_path):
    tiny_ink(tmp_path / "a.hand.parquet", 80, ["w1", "w2"])
    cfg = DataConfig(dataset=str(tmp_path / "a.hand.parquet"), max_text_length=20, max_seq_length=64, n_merges=8)
    train, _, st = create_datasets(cfg)
    model = build(fit(ModelConfig(arch="graves", n_layer=3, n_embd=24, window=4, source_dropout=0.0), train)).eval()
    X, C, Y, W = train.batch(range(3))
    whole, loss = model(X[:, :20], C, Y[:, :20], W)
    state, steps = None, []
    for t in range(20):
        logits, state = model.run(X[:, t:t + 1], C, W, state)
        steps.append(logits)
    assert torch.isfinite(loss) and torch.allclose(whole, torch.cat(steps, 1), atol=1e-5)
    out = model.generate(X[:, :1], C, 12, end_token=st.END, pad_token=st.PAD, source=W)
    assert out.shape[0] == 3 and 1 < out.shape[1] <= 13 and out.max() < st.vocab_size


def test_style_rank_scores_a_word_after_its_own_writer_and_after_others(tmp_path):
    tiny_ink(tmp_path / "a.hand.parquet", 600, [f"w{k}" for k in range(60)])
    tiny_ink(tmp_path / "b.hand.parquet", 60, ["sam"], text="cd")
    cfg = DataConfig(dataset=f"{tmp_path / 'a.hand.parquet'},{tmp_path / 'b.hand.parquet'}", max_text_length=20, grid=0.02, n_merges=8,
                     holdout=3)
    train, test, st = create_datasets(cfg)
    model = build(fit(ModelConfig(n_layer=1, n_head=2, n_embd=32), train))
    attach_pen_deltas(model, st)
    held = test.unseen["a"]
    rank, first, n = style_rank(model, held, "cpu", n=4, prime=2, targets=20)
    assert n == 20 and 1 <= rank <= 4 and 0 <= first <= 1
    assert style_rank(model, held, "cpu", n=len(held.groups) + 1) is None
