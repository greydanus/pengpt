import numpy as np

from pengpt import ARROWS, DIRECTIONS, DOWN, UP, ScribeTokenizer, Tokink, learn_merges


def walks(n, seed=0):
    rng = np.random.default_rng(seed)
    return [np.r_[DOWN, rng.choice(8, rng.integers(5, 60), p=[0.3, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.1]), UP] for _ in range(n)]


def test_tokink_reads_arrow_merges_and_merges_like_the_scribe_tokenizer(tmp_path):
    merges = learn_merges(walks(400), n_merges=200, min_count=2)
    name = {int(np.flatnonzero((DIRECTIONS == step).all(1))[0]): arrow for arrow, step in ARROWS.items()}
    for a, b, c in merges:
        name[c] = name[a] + name[b]
    path = tmp_path / "merges.txt"
    path.write_text("#version: 0.2\n" + "".join(f"{name[a]} {name[b]}\n" for a, b, _ in merges), encoding="utf-8")
    slow, fast = ScribeTokenizer(merges=merges), Tokink(0.02, path, len(merges))
    assert fast.merges == slow.merges and fast.vocab_size == slow.vocab_size
    for walk in walks(200, seed=1):
        assert np.array_equal(fast.apply_merges(walk), slow.apply_merges(walk))
        assert fast.expand(fast.apply_merges(walk)) == list(walk)
    assert len(Tokink(0.02, path, 50).merges) == 50
    rng = np.random.default_rng(2)
    words = [np.c_[np.cumsum(rng.normal(0.01, 0.02, (30, 2)), 0), np.ones(30)] for _ in range(20)]
    assert all(np.array_equal(a, b) for a, b in zip(fast.encode_many(words), slow.encode_many(words)))
