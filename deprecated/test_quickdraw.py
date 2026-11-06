import numpy as np


def test_bradley_terry_recovers_an_ordering():
    from deprecated.quickdraw import bradley_terry, spearman
    truth = np.arange(20, dtype=float)
    rng = np.random.default_rng(0)
    comparisons = []
    for _ in range(600):
        i, j = rng.choice(20, 2, replace=False)
        p = 1 / (1 + np.exp(-(truth[i] - truth[j]) / 4))
        comparisons.append((i, j) if rng.random() < p else (j, i))
    assert spearman(bradley_terry(20, comparisons), truth) > 0.9


def test_select_per_class_keeps_class_balance():
    from deprecated.quickdraw import select_per_class
    labels = np.array(["cat"] * 40 + ["car"] * 40)
    scores = np.r_[np.arange(40), np.arange(40) * -1.0]
    keep = select_per_class(scores, labels, fraction=0.25)
    assert len(keep) == 20
    assert (labels[keep] == "cat").sum() == (labels[keep] == "car").sum() == 10
