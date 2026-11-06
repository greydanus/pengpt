import numpy as np

from tools.glyphs import apply_ops, compose, cut


def entry(cursive):
    line = lambda x0, x1, y=0.0: [[x0, y], [x1, y]]
    return {"cursive": cursive, "glyphs": {
        "n": {"strokes": [line(0, 1)], "marks": []},
        "i": {"strokes": [line(0, 0.5), [[0.3, -1.2]]], "marks": [1]},
        "c": {"strokes": [line(0, 0.2), line(0.3, 1)], "marks": []},
        "H": {"strokes": [line(0, 0.1), line(0.5, 0.6)], "marks": []},
        "!": {"strokes": [line(0, 0.1), [[0.05, 0.1]]], "marks": []}}}


def test_cursive_word_is_one_stroke_with_marks_after():
    strokes = compose(entry(True), "nicn", [0, 1, 2, 3])
    assert len(strokes) == 2
    assert len(strokes[0]) == 10 and np.allclose(strokes[-1], [[1.3, -1.2]])


def test_space_and_punctuation_break_the_pen():
    strokes = compose(entry(True), "ni n!", [0, 1, 2, 3, 4])
    assert len(strokes) == 5
    assert np.allclose(strokes[1], [[1.3, -1.2]])


def test_capital_joins_only_when_close():
    near = compose(entry(True), "Hn", [0, 0.9])
    far = compose(entry(True), "Hn", [0, 3])
    assert len(near) == 2 and len(far) == 3


def test_print_keeps_every_stroke_in_place():
    strokes = compose(entry(False), "nic", [0, 1, 2])
    assert len(strokes) == 5
    assert np.allclose(strokes[2], [[1.3, -1.2]])


def test_fix_ops():
    a, b, c = (np.array([[k, 0.0], [k + 0.5, 0.0]]) for k in range(3))
    out = apply_ops([a, b, c], ["order:2,0,1", "join:1,2", "reverse:0"])
    assert len(out) == 2 and np.allclose(out[0], c[::-1]) and np.allclose(out[1], np.vstack([a, b]))


class Font:
    def getlength(self, text):
        return 10.0 * len(text)


def test_cut_keeps_the_middle_letter_and_its_descender():
    row = {"text": "ngn", "pos": 1, "baseline": 50, "x0": 0}
    word = np.array([[2, 50], [9, 40], [12, 45], [18, 40], [15, 70], [8, 75], [21, 48], [28, 40]], float)
    (mid,) = cut(row, [word], Font(), 10)
    assert np.allclose(mid[0], [12, 45]) and np.allclose(mid[-1], [8, 75])


def test_dots_snap_to_the_ink_and_missing_dots_are_added():
    from tools.glyphs import snap_dots
    stem, near = np.array([[50.0, 40], [50, 100]]), np.array([[57.0, 22], [58, 23]])
    out = snap_dots([stem, near], [(60.0, 20.0, 10.0), (90.0, 95.0, 8.0)], xh=100)
    assert len(out) == 3 and np.allclose(out[0], stem)
    assert np.allclose(out[1].mean(0), [60, 20], atol=1) and np.allclose(np.hypot(*(out[1] - [60, 20]).T), 5)
    assert np.allclose(out[2][0], [90, 91])
    v = np.array([[0.0, -1.0], [0.5, 0.0], [1.0, -1.0]])
    out = apply_ops([v], ["split:0,1", "reverse:0"])
    assert np.allclose(out[0], v[1::-1]) and np.allclose(out[1], v[1:])
