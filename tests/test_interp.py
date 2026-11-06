import numpy as np
import torch

from pengpt import ByteTokenizer, DataConfig, ModelConfig, PenDataset, PenTransformer, ScribeTokenizer, attach_pen_deltas
from test_pengpt import make_word
from tools import interp


def tiny():
    cfg = DataConfig(max_seq_length=64, max_text_length=12, max_words=2, grid=0.01, seed=0)
    st = ScribeTokenizer(grid=cfg.grid)
    texts = ["cab", "face", "bag", "dad", "egg", "fed"]
    ds = PenDataset([make_word(seed=s) for s in range(6)], texts, np.arange(6), st, ByteTokenizer(), cfg, length=4,
                    augment=False, writer_ids=np.ones(6, int), writers=["", "sam"])
    torch.manual_seed(0)
    model = PenTransformer(ModelConfig(n_layer=2, n_head=2, n_embd=32, n_writers=2, vocab_size=st.vocab_size,
                                       block_size=cfg.max_seq_length, context_vocab_size=257,
                                       context_block_size=cfg.max_text_length)).eval()
    attach_pen_deltas(model, st)
    return model, ds


def test_token_paths_end_at_token_displacement():
    st = ScribeTokenizer(grid=0.01, merges=[(0, 0, 10), (10, 1, 11)])
    paths = interp.token_paths(st)
    deltas = st.token_deltas()
    for v, q in enumerate(paths):
        if len(q):
            assert np.array_equal(q[-1], deltas[v])
    assert len(paths[st.DOWN]) == len(paths[st.UP]) == len(paths[st.PAD]) == 0


def test_attention_rows_are_distributions_and_plots_render(tmp_path):
    model, ds = tiny()
    hidden = interp.instrument(model)
    st, paths = ds.stroke_tok, interp.token_paths(ds.stroke_tok)
    seq, probs, att = interp.write(model, ds, "cab", "sam")
    assert att.shape[:3] == (2, 2, len(seq))
    assert np.allclose(att.sum(-1), 1, atol=1e-5) and np.allclose(probs.sum(-1), 1, atol=1e-5)
    pos, down, strokes = interp.trace(st, seq, paths)
    interp.plot_tokens(model, st, paths, str(tmp_path / "tokens.png"))
    interp.plot_probes(model, ds, hidden, str(tmp_path / "probes.png"))
    assert (tmp_path / "tokens.png").exists() and (tmp_path / "probes.png").exists()
