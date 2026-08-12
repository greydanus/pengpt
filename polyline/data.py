import numpy as np

from pengpt.data import (PenDataset, check_scale, load_examples, prepare_word)
from pengpt.tokenizer import CharTokenizer

from .tokenizer import PolylineTokenizer, learn_merges


def create_datasets(cfg, tokenizer=None, merges=None, text_encoder=None):
    examples = load_examples(cfg.dataset, getattr(cfg, "max_examples", 0) or None)
    check_scale(examples, cfg.grid)
    bank_points = [e["points"] for e in examples]
    bank_texts = [e["text"] for e in examples]
    alphabet = " " + "".join(sorted(set("".join(bank_texts)) - {" "}))
    char_tok = CharTokenizer(alphabet)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(examples))
    n_test = min(1000, max(10, int(0.05 * len(examples))))
    train_ix, test_ix = perm[:-n_test], perm[-n_test:]

    if tokenizer is None:
        epsilon = getattr(cfg, "epsilon", 0.010)
        max_run = getattr(cfg, "max_run", 16)
        max_chunk_verts = getattr(cfg, "max_chunk_verts", 4)
        n_merges = getattr(cfg, "n_merges", 256)
        if merges is None and n_merges:
            base = PolylineTokenizer(grid=cfg.grid, epsilon=epsilon, max_run=max_run,
                                     max_chunk_verts=max_chunk_verts)
            sample = rng.choice(train_ix, size=min(600, len(train_ix)), replace=False)
            corpus = [base.encode_word(prepare_word(bank_points[i], cfg, rng))
                      for i in sample]
            merges = learn_merges(corpus, n_merges,
                                  reserved=(base.DOWN, base.UP))
            print(f"Learned {len(merges)} polyline BPE merges")
        tokenizer = PolylineTokenizer(
            grid=cfg.grid, epsilon=epsilon, max_run=max_run,
            max_chunk_verts=max_chunk_verts, merges=merges or [],
        )

    bank_sources = [e["source"] for e in examples]
    if not any(bank_sources):
        bank_sources = None

    def build(ix, n, name, seed):
        return PenDataset(bank_points, bank_texts, ix, tokenizer, char_tok, cfg,
                          length=n, augment=cfg.augment != "none", name=name,
                          seed=seed, text_encoder=text_encoder,
                          bank_sources=bank_sources)

    train_dataset = build(train_ix, cfg.train_size, "train", cfg.seed)
    test_dataset = build(test_ix, cfg.test_size, "test", cfg.seed + 1)
    print(f"Word bank: {len(train_ix)} train / {len(test_ix)} test words; "
          f"polyline vocab {tokenizer.vocab_size} "
          f"(eps={tokenizer.epsilon}, max_run={tokenizer.max_run}); "
          f"alphabet ({len(alphabet)} chars): {alphabet!r}")
    return train_dataset, test_dataset, tokenizer, char_tok
