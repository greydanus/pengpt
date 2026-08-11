import numpy as np
import pytest
import torch

from pengpt import (DataConfig, ModelConfig, PenTransformer, PenDataset,
                    ScribeTokenizer, CharTokenizer, learn_merges)
from pengpt.data import IGNORE_INDEX, prepare_word, resample
from pengpt.model import save_checkpoint, load_checkpoint
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


def test_merges_apply_in_learned_order():
    """Encoding must reproduce what learn_merges produced, not just something
    reversible.

    Rule order is priority order. A left-to-right scan that takes whichever
    rule matches first is reversible and shorter than the base sequence, so
    losslessness alone does not catch it -- but it is a different encoding, and
    it leaves a third of the compression on the table.
    """
    st = ScribeTokenizer(grid=0.01, merges=[(0, 1, 10), (2, 0, 11)])
    # rule (0,1) was learned first, so it wins the shared 0.
    assert list(st.apply_merges(np.array([2, 0, 1]))) == [2, 10]


def test_merges_match_sequential_application():
    """The same check against learned merges on many words."""
    st0 = ScribeTokenizer(grid=0.01)
    words = [make_word(n=60, seed=s) for s in range(20)]
    base = [st0.encode_word(w) for w in words]
    merges = learn_merges(base, n_merges=128, min_count=2)
    st = ScribeTokenizer(grid=0.01, merges=merges)

    def sequential(seq):
        s = [int(t) for t in seq]
        for a, b, c in merges:                      # one pass per rule, in order
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


def test_normalize_absolute_bypasses_delta_detection():
    """Sparse absolute drawings must not be read as deltas.

    A square is five points after RDP simplification, so its span is small
    relative to its mean step and the delta heuristic fires; cumsum then turns
    it into a diagonal staircase. This corrupted 24% of a filtered Quick, Draw!
    corpus. absolute=True is how a caller that knows its format opts out.
    """
    from pengpt.convert import normalize
    square = np.array([[0, 0, 1], [255, 0, 1], [255, 255, 1],
                       [0, 255, 1], [0, 0, 1]], dtype=float)
    out = normalize(square.copy(), absolute=True)
    assert np.allclose(out[0, :2], out[-1, :2])       # the square stays closed
    assert (np.diff(out[:, 0]) < 0).any()             # a cumsum'd square is monotone


def test_physics_captions_must_determine_their_answer():
    """A truncated physics caption must not leave the answer ambiguous.

    These captions compare two clauses and the deciding one is usually second,
    so the cursive default of 50 characters cut 33% of physics_v0 down to a
    prompt shared by examples with opposite labels. Loss still falls on that
    data, so only an explicit check catches it.
    """
    import physics_sketch

    examples = physics_sketch.generate(300, seed=0)
    longest = max(len(e["text"]) for e in examples)

    # At the full caption length nothing is truncated, so nothing is ambiguous.
    assert physics_sketch.check_captions(examples, longest) == longest

    # Half a caption cuts before the deciding clause on every archetype.
    with pytest.raises(SystemExit, match="no longer determines the answer"):
        physics_sketch.check_captions(examples, longest // 2)


def test_every_physics_archetype_composes_several_attributes():
    """No archetype may collapse to a handful of captions.

    Distinct captions, not example count, bound what a text-conditioned model
    can learn: examples sharing a caption differ only in stroke style. Three
    archetypes once compared a single binned attribute between two objects and
    so could only ever produce six captions each, however many examples were
    generated. This pins the floor so that cannot come back unnoticed.
    """
    import collections

    import physics_sketch

    rng = np.random.default_rng(0)
    captions = collections.defaultdict(set)
    answers = collections.defaultdict(collections.Counter)
    for i in range(12_000):
        kind = list(physics_sketch.SPECS)[i % len(physics_sketch.SPECS)]
        spec = physics_sketch.SPECS[kind](rng)
        if spec is not None:
            captions[kind].add(spec["text"])
            answers[kind][spec["answer"]] += 1

    for kind in physics_sketch.SPECS:
        # Geometry archetypes carry their variety in the drawing rather than
        # the caption -- that is the point of them -- so only the caption-borne
        # archetypes are held to a caption-count floor.
        if kind not in physics_sketch.GEOMETRY_KINDS:
            assert len(captions[kind]) >= 40, \
                f"{kind} has too few distinct captions"
        # No archetype may be guessable from its answer prior alone.
        counts = answers[kind]
        assert max(counts.values()) / sum(counts.values()) < 0.7, \
            f"{kind} answers are imbalanced: {dict(counts)}"


def test_geometry_archetypes_need_the_picture():
    """Geometry captions must NOT determine the answer; the drawing must.

    The rest of the corpus states every attribute, so the caption alone answers
    it and a sketch can at best break even. These archetypes are the other
    regime -- easy with the picture, near chance without -- which is where a
    scratchpad can actually pay off. If a caption here ever becomes sufficient,
    the archetype has stopped testing what it exists to test.
    """
    import collections

    import physics_sketch

    rng = np.random.default_rng(0)
    for kind in physics_sketch.GEOMETRY_KINDS:
        by_caption = collections.defaultdict(collections.Counter)
        for _ in range(4000):
            spec = physics_sketch.SPECS[kind](rng)
            if spec is not None:
                by_caption[spec["text"]][spec["answer"]] += 1
        total = sum(sum(c.values()) for c in by_caption.values())
        best = sum(max(c.values()) for c in by_caption.values())
        assert best / total < 0.75, (
            f"{kind} is {best / total:.0%} solvable from the caption alone; "
            f"the drawing should be carrying that information")


def test_chain_terminal_stages_are_balanced():
    """Each chain question type must be balanced on its own.

    An aggregate check hides a skewed stage behind the mix: a buoyancy stage
    inheriting a 60/40 float prior stays invisible when most chains end on a
    seesaw. Balance is per terminal stage or it is not balance.
    """
    import collections

    import physics_sketch

    rng = np.random.default_rng(0)
    by_stage = collections.defaultdict(collections.Counter)
    for _ in range(6000):
        spec = physics_sketch.spec_chain(rng, 4)
        if spec is not None:
            by_stage[spec["steps"][-1]["stage"]][spec["answer"]] += 1

    assert len(by_stage) >= 5, "chains should end on a variety of stages"
    for stage, counts in by_stage.items():
        total = sum(counts.values())
        if total < 100:                     # too few to judge
            continue
        assert max(counts.values()) / total < 0.7, \
            f"chain stage {stage} is imbalanced: {dict(counts)}"


def test_chain_depth_is_real_not_decorative():
    """A chain's final clause must not give away the answer.

    If it does, every earlier stage is decorative and a "depth 4" example is a
    depth-1 example wearing a label -- which would silently invalidate any
    scaling study that uses depth as its independent variable. Stages once
    named the side they received ("the left chute feeds..."), and the final
    clause alone then predicted the answer 95% of the time.
    """
    import collections

    import physics_sketch

    rng = np.random.default_rng(0)
    by_last = collections.defaultdict(collections.Counter)
    steps_seen = []
    for _ in range(3000):
        spec = physics_sketch.spec_chain(rng, 4)
        if spec is None:
            continue
        by_last[spec["text"].split(", then ")[-1]][spec["answer"]] += 1
        steps_seen.append(len(spec["steps"]))

    assert steps_seen and all(n == 4 for n in steps_seen)
    total = sum(sum(c.values()) for c in by_last.values())
    best = sum(max(c.values()) for c in by_last.values())
    assert best / total < 0.7, (
        f"final clause alone predicts the answer {best / total:.0%} of the time")


def test_chain_carries_per_step_ground_truth():
    """Each step records its input and output, for process-level verification.

    Verifying only the final answer gives one bit per example; the point of
    generating chains procedurally is that every intermediate state is known,
    so credit can be assigned per step.
    """
    import physics_sketch

    rng = np.random.default_rng(0)
    example = None
    while example is None:
        example = physics_sketch.make_example("chain3", rng)

    steps = example["meta"]["steps"]
    assert example["meta"]["depth"] == 3 and len(steps) == 3
    # Each step consumes what the one before it produced.
    for earlier, later in zip(steps, steps[1:]):
        assert later["in"] == earlier["out"]
    assert steps[-1]["out"] == example["meta"]["answer"]


def test_plain_style_shortens_sketches():
    """Plain style must cut sketch length without changing the labels.

    In a depth sweep the sketch should hold reasoning state and nothing else;
    decorative ink that grows with depth would confound sketch length with
    reasoning depth.
    """
    import physics_sketch
    from pengpt.tokenizer import ScribeTokenizer

    st = ScribeTokenizer(grid=0.020)

    def token_p99(style):
        physics_sketch.STYLE = style
        rng = np.random.default_rng(0)
        lengths = []
        while len(lengths) < 60:
            ex = physics_sketch.make_example("lever", rng)
            if ex is not None:
                lengths.append(len(st.encode_word(np.array(ex["points"]))))
        return np.percentile(lengths, 99)

    try:
        assert token_p99("plain") < token_p99("rich")
    finally:
        physics_sketch.STYLE = "rich"


def test_bradley_terry_recovers_an_ordering():
    from pengpt.quality import bradley_terry, spearman
    truth = np.arange(20, dtype=float)
    rng = np.random.default_rng(0)
    comparisons = []
    for _ in range(600):
        i, j = rng.choice(20, 2, replace=False)
        p = 1 / (1 + np.exp(-(truth[i] - truth[j]) / 4))
        comparisons.append((i, j) if rng.random() < p else (j, i))
    assert spearman(bradley_terry(20, comparisons), truth) > 0.9


def test_select_per_class_keeps_class_balance():
    from pengpt.quality import select_per_class
    labels = np.array(["cat"] * 40 + ["car"] * 40)
    scores = np.r_[np.arange(40), np.arange(40) * -1.0]   # opposite orderings
    keep = select_per_class(scores, labels, fraction=0.25)
    assert len(keep) == 20
    assert (labels[keep] == "cat").sum() == (labels[keep] == "car").sum() == 10


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
    assert x[0] == st.BOS                         # generation starts here too
    end = (x == st.END).nonzero().item()          # exactly one END in the input
    assert y[end - 1] == st.END                   # END is the last thing predicted...
    assert (y[end:] == IGNORE_INDEX).all()        # ...and nothing is predicted from it
    assert (x[end + 1:] == st.PAD).all()
    assert tiny_dataset.char_tok.decode(c) == tiny_dataset.text_for(0)


def test_targets_are_inputs_shifted_by_one(tiny_dataset):
    """The supervised span must line up exactly, or every token is off by one."""
    x, _, y = tiny_dataset[0]
    st = tiny_dataset.stroke_tok
    end = (x == st.END).nonzero().item()
    assert torch.equal(y[:end], x[1:end + 1])


def test_generation_prefix_occurs_in_training(tiny_dataset):
    """What generate() seeds with has to be what position 0 saw in training.

    Seeding from a token the model never saw first leaves the opening move --
    the pen's entry point and the word's height above the baseline -- to a
    prefix with no training support.
    """
    st = tiny_dataset.stroke_tok
    firsts = {tiny_dataset[i][0][0].item() for i in range(len(tiny_dataset))}
    assert firsts == {st.BOS}


def test_dataset_never_truncates_mid_word(tiny_dataset):
    """Packing stops before overflow, so a word is never cut in half."""
    st = tiny_dataset.stroke_tok
    for i in range(len(tiny_dataset)):
        x, _, _ = tiny_dataset[i]
        end = (x == st.END).nonzero()
        assert len(end) == 1, "exactly one END expected"
        assert end.item() < tiny_dataset.cfg.max_seq_length


def test_truncated_word_does_not_supervise_end():
    """When a word overflows the block, the END written at the cut must not be
    a training target: the drawing did not actually end there."""
    cfg = DataConfig(max_seq_length=16, max_text_length=8, max_words=1, grid=0.005)
    st, ct = ScribeTokenizer(grid=cfg.grid), CharTokenizer(" abc")
    long_word = make_word(n=200)
    ds = PenDataset([long_word], ["cab"], np.arange(1), st, ct, cfg, length=1,
                    augment=False)
    x, _, y = ds[0]
    end = (x == st.END).nonzero().item()
    assert end == cfg.max_seq_length - 1          # cut at the block edge
    assert y[end - 1] == IGNORE_INDEX             # END not supervised
    assert (y[:end - 1] != IGNORE_INDEX).all()    # everything before it is


def test_bank_smaller_than_max_words():
    """max_words is a ceiling, not a requirement.

    The test split is 5% of a corpus and floors at ten words, so a bank with
    fewer words than max_words is ordinary rather than exotic; drawing without
    replacement must not ask for more than exists.
    """
    cfg = DataConfig(max_seq_length=256, max_text_length=20, max_words=8,
                     grid=0.01, seed=0)
    st, ct = ScribeTokenizer(grid=0.01), CharTokenizer(" abcdefg")
    for n_bank in (1, 2, 7):
        bank = [make_word(seed=s) for s in range(n_bank)]
        texts = ["cab", "face", "bag", "dad", "egg", "fed", "gag"][:n_bank]
        ds = PenDataset(bank, texts, np.arange(n_bank), st, ct, cfg, length=4,
                        augment=False)
        x, c, _ = ds[0]
        assert (x != st.PAD).sum() > 2
        assert len(ct.decode(c).split()) <= n_bank


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


class _StubTextEncoder:
    dim = 8

    def encode(self, text, length):
        out = np.zeros((length, 8), dtype=np.float32)
        for i, b in enumerate(text.encode()[:length]):
            out[i, b % 8] = 1.0
            out[i, (b // 8) % 8] += 0.5
        return out


def test_float_context_path(tiny_dataset):
    st, cfg = tiny_dataset.stroke_tok, tiny_dataset.cfg
    tiny_dataset.text_encoder = _StubTextEncoder()
    try:
        x, c, y = tiny_dataset[0]
        assert c.dtype == torch.float32
        assert c.shape == (cfg.max_text_length, 8)
        mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size,
                           block_size=cfg.max_seq_length, context_vocab_size=1,
                           context_block_size=cfg.max_text_length, context_dim=8)
        model = PenTransformer(mcfg)
        logits, loss = model(x.unsqueeze(0), c.unsqueeze(0), y.unsqueeze(0))
        assert torch.isfinite(loss)
        empty = torch.zeros(1, cfg.max_text_length, 8)
        _, loss2 = model(x.unsqueeze(0), empty, y.unsqueeze(0))
        assert torch.isfinite(loss2)
        out = model.generate(x.unsqueeze(0)[:, :6], c.unsqueeze(0),
                             max_new_tokens=8, end_token=st.END, pad_token=st.PAD)
        assert out.shape[1] <= 14
    finally:
        tiny_dataset.text_encoder = None


def test_clip_char_encode_structure(tiny_dataset):
    from pengpt.textenc import CharClipEncoder
    enc = CharClipEncoder.__new__(CharClipEncoder)
    enc.char_tok = tiny_dataset.char_tok
    enc.clip_dim = 4
    enc.dim = 4 + tiny_dataset.char_tok.vocab_size
    enc._global = lambda text: np.array([1, 0, 0, 0], dtype=np.float32)
    out = enc.encode("cab", 8)
    ct = tiny_dataset.char_tok
    assert out.shape == (8, enc.dim)
    assert (out[:3, 0] == 1).all() and (out[3:] == 0).all()
    for i, ch in enumerate("cab"):
        assert out[i, 4 + ct.stoi[ch]] == 1.0
        assert out[i, 4:].sum() == 1.0


def test_holdout_filter():
    from pengpt.data import filter_holdout
    examples = [{"text": "a big wolf"}, {"text": "wolfhound"},
                {"text": "a cat"}, {"text": "gray wolf pup"}]
    kept = filter_holdout(examples, "wolf")
    assert [e["text"] for e in kept] == ["wolfhound", "a cat"]
    assert filter_holdout(examples, "") is examples


def test_token_deltas_track_pen_position(tiny_dataset):
    """The per-token displacement table must integrate to the true pen path,
    through BPE merges, or pen-position features would silently lie."""
    st = tiny_dataset.stroke_tok
    deltas = st.token_deltas()
    rng = np.random.default_rng(0)
    points = np.column_stack([np.cumsum(rng.normal(0, 0.05, (30, 2)), axis=0),
                              np.ones(30)])
    points[9, 2] = 0  # a lift mid-way
    tokens = st.encode_word(points)
    summed = deltas[tokens].sum(axis=0)
    grid_end = np.rint(points[-1, :2] / st.grid).astype(int)
    assert (summed == grid_end).all()


def test_pen_pos_features_are_causal_and_checkpoint_safe(tiny_dataset, tmp_path):
    st, ct, cfg = tiny_dataset.stroke_tok, tiny_dataset.char_tok, tiny_dataset.cfg
    mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size,
                       block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                       context_block_size=cfg.max_text_length, pen_pos_bands=6)
    model = PenTransformer(mcfg)
    model.pen_deltas.copy_(torch.tensor(st.token_deltas()))
    model.eval()
    x, c, y = (t.unsqueeze(0) for t in tiny_dataset[0])
    logits, loss = model(x, c, y)
    assert torch.isfinite(loss)

    # Causality: changing a suffix token must not change earlier logits.
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % st.vocab_size
    logits2, _ = model(x2, c, y)
    assert torch.allclose(logits[0, :-1], logits2[0, :-1], atol=1e-5)

    # Round-trip through a checkpoint, table included.
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, ct.alphabet, {"dataset": "x"}, st.merges)
    reloaded, _ = load_checkpoint(str(path))
    assert (reloaded.pen_deltas == model.pen_deltas).all()

    # A no-position config must still load checkpoints that never had one.
    mcfg0 = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size,
                        block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                        context_block_size=cfg.max_text_length)
    model0 = PenTransformer(mcfg0)
    logits0, _ = model0(x, c, y)
    assert logits0.shape == logits.shape


def _apply_merges_reference(merges, pairs, tokens):
    """The pre-vectorization implementation, kept to pin exact equivalence."""
    if not pairs:
        return np.asarray(tokens)
    out = [int(t) for t in tokens]
    present = set(zip(out, out[1:]))
    for a, b, merged in merges:
        if (a, b) not in present:
            continue
        nxt, i, n = [], 0, len(out)
        while i < n:
            if i + 1 < n and out[i] == a and out[i + 1] == b:
                nxt.append(merged); i += 2
            else:
                nxt.append(out[i]); i += 1
        out = nxt
        present = set(zip(out, out[1:]))
    return np.array(out, dtype=np.int64)


def _walk_reference(grid_xy):
    out = []
    for (x0, y0), (x1, y1) in zip(grid_xy[:-1], grid_xy[1:]):
        if x0 != x1 or y0 != y1:
            out.extend(int(t) for t in bresenham_steps(x0, y0, x1, y1))
    return out


def test_vectorized_tokenizer_matches_reference(tiny_dataset):
    from pengpt.tokenizer import _walk
    st = tiny_dataset.stroke_tok
    rng = np.random.default_rng(7)
    for trial in range(30):
        # Walks with dense unit steps, repeats, and occasional long jumps.
        steps = rng.integers(-1, 2, size=(rng.integers(2, 60), 2))
        steps[rng.random(len(steps)) < 0.1] *= rng.integers(2, 9)
        grid = np.cumsum(steps, axis=0)
        assert _walk(grid) == _walk_reference(grid)

        # Token streams engineered to hit overlapping-run merges too.
        tokens = rng.integers(0, st.vocab_size // 2, size=rng.integers(2, 200))
        tokens[rng.random(len(tokens)) < 0.3] = tokens[0]  # force runs
        got = st.apply_merges(np.asarray(tokens))
        want = _apply_merges_reference(st.merges, st._pairs, tokens)
        assert (got == want).all(), trial


def test_augment_drawing():
    from types import SimpleNamespace
    from pengpt.data import augment_drawing
    rng = np.random.default_rng(0)
    pts = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0],
                    [2, 2, 1], [3, 2, 1], [3, 2, 0],
                    [5, 1, 1], [6, 1, 1], [6, 1, 0]], dtype=float)

    # Defaults are inert.
    cfg = SimpleNamespace(stroke_dropout=0.0, hflip=0.0, tremor=0.0)
    assert (augment_drawing(pts, "a scene", cfg, rng) == pts).all()

    # Dropout removes whole strokes with their lift rows, never all of them.
    cfg = SimpleNamespace(stroke_dropout=0.999, hflip=0.0, tremor=0.0)
    out = augment_drawing(pts, "a scene", cfg, rng)
    assert len(out) in (3, 6) and (out[:, 2] == 1).sum() >= 2
    assert (out[-1, 2] == 0)

    # Flip reflects x within the bbox, but never when the caption is sided.
    cfg = SimpleNamespace(stroke_dropout=0.0, hflip=1.0, tremor=0.0)
    out = augment_drawing(pts, "a scene", cfg, rng)
    assert np.isclose(out[:, 0].min(), pts[:, 0].min())
    assert np.isclose(out[:, 0].max(), pts[:, 0].max())
    assert np.isclose(out[0, 0], 6.0)
    kept = augment_drawing(pts, "a tree on the left", cfg, rng)
    assert (kept == pts).all()

    # Tremor perturbs a dense stroke without changing its structure: same
    # point count, pen states untouched, displacement bounded near the
    # amplitude (perpendicular noise plus endpoint slop, never a redraw).
    line = np.column_stack([np.linspace(0, 1, 60), np.zeros(60), np.ones(60)])
    line[-1, 2] = 0
    cfg = SimpleNamespace(stroke_dropout=0.0, hflip=0.0, tremor=0.004)
    out = augment_drawing(line.copy(), "a scene", cfg, np.random.default_rng(1))
    assert out.shape == line.shape
    assert (out[:, 2] == line[:, 2]).all()
    assert not np.allclose(out[:, :2], line[:, :2])
    assert np.abs(out[:, 1]).max() < 6 * cfg.tremor
    # Two-point strokes have no interior to wave; they pass through untouched.
    assert (augment_drawing(pts, "a scene", cfg,
                            np.random.default_rng(2)) == pts).all()


def test_bucketed_loader(tiny_dataset):
    from pengpt.data import BucketedInfiniteLoader

    # bank_word_for must replay exactly the draw __getitem__ makes.
    if tiny_dataset.cfg.max_words == 1:
        for idx in range(20):
            bank = tiny_dataset.bank_word_for(idx)
            assert tiny_dataset.text_for(idx) == tiny_dataset.bank_texts[bank]

    ds = tiny_dataset
    ds.cfg.max_words = 1
    for idx in range(20):
        bank = ds.bank_word_for(idx)
        assert ds.bank_texts[bank] in ds.text_for(idx) or \
            ds.text_for(idx) == ds.bank_texts[bank]

    loader = BucketedInfiniteLoader(ds, batch_size=4, num_workers=0)
    x, c, y = loader.next()
    assert x.shape[0] == 4 and x.shape[1] == ds.cfg.max_seq_length
