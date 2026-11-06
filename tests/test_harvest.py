import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("skimage")
from tools.harvest import TEXT, pick, shrink, variants

WAVE = [(20, 60), (60, 20), (100, 60), (140, 20), (180, 60)]


def word(fill="black"):
    img = Image.new("RGB", (200, 80), "white")
    ImageDraw.Draw(img).line(WAVE, fill=fill, width=5)
    return img


def test_shrink_caps_oversized_crops():
    img, s = shrink(Image.new("RGB", (2000, 900)))
    assert img.height <= 320 and max(img.size) <= 1024 and np.isclose(s, img.height / 900, atol=0.01)
    assert shrink(Image.new("RGB", (300, 100)))[1] == 1.0


def test_variants_find_faint_pencil_and_reject_blank_images():
    v = variants(word(fill=(150, 150, 150)))
    assert (v["clean"] < 100).sum() > 100 and (v["mono"] < 100).sum() > 100
    with pytest.raises(ValueError):
        variants(Image.new("RGB", (200, 80), "white"))


def test_pick_keeps_the_candidate_on_the_ink():
    clean = variants(word())["clean"]
    off = [[(x, y + 20) for x, y in WAVE]]
    best = pick((clean, "wave", [("mono", TEXT, off, ""), ("clean", TEXT, [WAVE], "")], None))
    assert best["key"][0] and best["candidate"] == "clean/text"


def test_source_rates_score_each_writer_by_its_qc_pass_rate():
    from tools.harvest import source_rates
    rows = [({"writer": "a"}, True)] * 3 + [({"writer": "a"}, False)] + [({"writer": "b"}, False)] * 2 + [({"source": "page.jpg"}, True)]
    assert source_rates(rows) == {"a": 0.75, "b": 0.0, "page.jpg": 1.0}
