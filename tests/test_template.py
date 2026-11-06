import os

import numpy as np
import pytest

from tools.template import compare, normalize


def test_template_flags_order_and_direction():
    stem, bar = np.array([[0.0, -1.0], [0.0, 0.0]]), np.array([[-0.4, -1.0], [0.4, -1.0]])
    template = [s.tolist() for s in normalize([stem, bar])]
    assert compare([stem, bar], template) == (None, None)
    assert compare([bar, stem], template)[1] == ["order:1,0"]
    assert compare([stem[::-1], bar], template)[1] == ["reverse:0"]
    assert "strokes" in compare([stem], template)[0]


@pytest.mark.skipif(not os.path.exists("data/raw/fonts/PatrickHand-Regular.ttf"), reason="font not downloaded")
def test_glyph_library_spacing_and_layout():
    from tools.datasets import glyph_library
    entry = {"font": "PatrickHand-Regular.ttf", "size": 200, "xh": 93.0, "cursive": False,
             "glyphs": {"o": {"strokes": [[[0, 0], [0.5, 0.1], [1, 0], [0.5, -1], [0, 0]]], "marks": []}}}
    lib = glyph_library(entry, "OFL-1.1")
    steps = np.hypot(*np.diff(np.asarray(lib["glyphs"]["o"]["strokes"][0]), axis=0).T)
    assert np.allclose(steps, lib["step"], rtol=0.1) and lib["width"]["o"] > 0 and len(lib["sha256"]) == 64
