import json
from argparse import Namespace

import numpy as np
from PIL import Image, ImageDraw

from pengpt import load_examples, read_ink, save_ink, strokes_to_points
from tools.ink import (detokenize, segment_command, file_hash, levenshtein, on_ink, qc_command, read_jsonl,
                       replace_rows, score, split_line, write_jsonl)
from utils import INK_HEIGHT, export_traces, points_to_strokes


def ink_row(spans, h=20, width=400):
    row = np.zeros((h, width), bool)
    for a, b in spans:
        row[5:15, a:b] = True
    return row


def test_split_line_cuts_at_widest_gaps():
    row = ink_row([(10, 40), (43, 70), (110, 150), (190, 210), (213, 240)])
    assert split_line(row, ["one", "two", "three"], 20) == [(7, 73), (107, 153), (187, 243)]


def test_split_line_merges_punctuation_into_its_word():
    row = ink_row([(10, 70), (78, 81), (130, 200)])
    assert split_line(row, ["word.", "next"], 20) == [(7, 84), (127, 203)]


def test_split_line_refuses_ambiguous_lines():
    assert split_line(ink_row([(10, 40), (60, 90), (110, 140)]), ["ab", "cd"], 20) is None
    assert split_line(ink_row([(10, 40)]), ["two", "words"], 20) is None


def test_split_line_rejects_widths_that_disagree_with_letter_counts():
    row = ink_row([(10, 20), (60, 300)])
    assert split_line(row, ["longword", "a"], 20) is None


def test_detokenize_strokes_and_bounds():
    tok = lambda *ids: "".join(f"<ink_token_{i}>" for i in ids)
    strokes = detokenize("<extra_id_0> word " + tok(450, 10, 225 + 20, 11, 225 + 21, 450, 5, 225 + 5, 300, 225 + 6))
    assert strokes == [[(10, 20), (11, 21)], [(5, 5)]]


def test_levenshtein():
    assert levenshtein("score", "score") == 0
    assert levenshtein("pcose", "score") == 2
    assert levenshtein("", "abc") == 3


def test_score_separates_faithful_from_wrong_traces(tmp_path):
    path = [(20, 60), (60, 20), (100, 60), (140, 20), (180, 60)]
    img = Image.new("RGB", (200, 80), "white")
    ImageDraw.Draw(img).line(path, fill="black", width=4)
    img.save(tmp_path / "w.png")
    item = {"image": str(tmp_path / "w.png")}
    good = score(item, {"strokes": [path]})
    half = score(item, {"strokes": [path[:3]]})
    off = score(item, {"strokes": [[(x, y + 15) for x, y in path]]})
    assert good["coverage"] > 0.95 and good["precision"] > 0.95
    assert half["coverage"] < 0.6 and half["precision"] > 0.95
    assert off["precision"] < 0.5
    assert score(item, {"strokes": []})["coverage"] == 0


def test_strokes_points_roundtrip():
    strokes = [[[0.0, 0.0], [0.1, 0.05], [0.2, 0.0]], [[0.3, -0.1]], [[0.4, 0.0], [0.5, 0.1]]]
    assert points_to_strokes(strokes_to_points(strokes)) == strokes


def test_export_keeps_relative_size_and_baseline(tmp_path):
    rows = [{"text": "a", "strokes": [[[0, 100], [10, 80], [20, 100]]], "baseline": 100,
             "style": "s", "source": "f.ttf", "license": "OFL", "qc": {"coverage": 1.0, "precision": 1.0}},
            {"text": "l", "strokes": [[[50, 100], [55, 40], [60, 100]]], "baseline": 100,
             "style": "s", "source": "f.ttf", "license": "OFL", "qc": {"coverage": 1.0, "precision": 1.0}}]
    (tmp_path / "acc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    export_traces(str(tmp_path / "acc.jsonl"), str(tmp_path / "out.ink.npz"))
    a, l = load_examples(str(tmp_path / "out.ink.npz"))
    height = lambda e: np.ptp(e["points"][:, 1])
    assert np.isclose(height(l) / height(a), 3.0, rtol=1e-2)
    assert np.isclose(a["points"][:, 1].max(), 0) and np.isclose(l["points"][:, 1].max(), 0)
    assert np.isclose(np.median([height(a), height(l)]) * 0.9, INK_HEIGHT, rtol=0.2)
    assert a["points"][:, 0].min() == 0


def test_on_ink_drops_strokes_in_the_blank_margin():
    img = Image.new("RGB", (100, 40), "white")
    ImageDraw.Draw(img).line([(20, 20), (80, 20)], fill="black", width=3)
    word, margin_dot = [(20, 20), (80, 20)], [(2, 2)]
    assert on_ink([margin_dot, word], img) == [word]


def test_export_puts_first_pen_point_at_x_zero(tmp_path):
    row = {"text": "o", "strokes": [[[30, 100], [10, 90], [20, 80], [40, 95]]], "baseline": 100,
           "style": "s", "source": "f.ttf", "license": "OFL", "qc": {"coverage": 1.0, "precision": 1.0}}
    (tmp_path / "acc.jsonl").write_text(json.dumps(row) + "\n")
    export_traces(str(tmp_path / "acc.jsonl"), str(tmp_path / "out.ink.npz"))
    (word,) = load_examples(str(tmp_path / "out.ink.npz"))
    assert word["points"][0, 0] == 0 and word["points"][:, 0].min() < 0


def test_replace_rows_is_idempotent(tmp_path):
    path = str(tmp_path / "manifest.jsonl")
    page_a = [{"image": "a1.png", "source": "a"}, {"image": "a2.png", "source": "a"}]
    replace_rows(path, page_a, "a")
    replace_rows(path, [{"image": "b1.png", "source": "b"}], "b")
    replace_rows(path, page_a, "a")
    assert sorted(r["image"] for r in read_jsonl(path)) == ["a1.png", "a2.png", "b1.png"]


def test_qc_ignores_traces_of_a_changed_image(tmp_path):
    img = Image.new("RGB", (200, 80), "white")
    path = [(20, 60), (60, 20), (100, 60), (140, 20), (180, 60)]
    ImageDraw.Draw(img).line(path, fill="black", width=4)
    image = str(tmp_path / "w.png")
    img.save(image)
    manifest, traces, out = (str(tmp_path / f) for f in ("m.jsonl", "t.jsonl", "acc.jsonl"))
    write_jsonl(manifest, [{"image": image, "text": "wave", "source": "s"}])
    write_jsonl(traces, [{"image": image, "hash": file_hash(image), "prompt": "p", "strokes": [path], "read": ""}])
    args = Namespace(manifest=manifest, traces=traces, out=out, sheet="",
                     min_coverage=0.88, min_precision=0.9, max_read_error=0.34, max_length_ratio=1.3)
    qc_command(args)
    assert len(read_jsonl(out)) == 1
    ImageDraw.Draw(img).line([(10, 10), (190, 10)], fill="black", width=4)
    img.save(image)
    qc_command(args)
    assert read_jsonl(out) == []


def test_ink_npz_roundtrip_and_limit(tmp_path):
    rows = [{"text": f"w{i} é", "strokes": [[[0.0, -0.1 * i], [0.123, 0.05]], [[0.3, 0.0]]],
             "style": "s", "writer": str(i % 2), "source": "src", "license": "MIT"} for i in range(5)]
    save_ink(rows, str(tmp_path / "d.ink.npz"))
    back = read_ink(str(tmp_path / "d.ink.npz"))
    assert [r["text"] for r in back] == [r["text"] for r in rows] and back[3]["writer"] == "1"
    assert all(np.allclose(np.asarray(a), b) for r, q in zip(rows, back) for a, b in zip(r["strokes"], q["strokes"]))
    assert len(load_examples(str(tmp_path / "d.ink.npz"), limit=2)) == 2


def test_empty_strokes_are_skipped():
    points = strokes_to_points([[[0.0, 0.0], [0.1, 0.1]], [], [[0.2, 0.0]]])
    assert points.shape == (5, 3) and points[:, 2].tolist() == [1, 1, 0, 1, 0]


def test_score_flags_traces_drawn_twice_and_counts_faint_ink(tmp_path):
    path = [(20, 60), (60, 20), (100, 60), (140, 20), (180, 60)]
    img = Image.new("RGB", (200, 80), "white")
    ImageDraw.Draw(img).line(path, fill=(170, 170, 170), width=4)
    img.save(tmp_path / "faint.png")
    item = {"image": str(tmp_path / "faint.png")}
    once, twice = score(item, {"strokes": [path]}), score(item, {"strokes": [path, path]})
    assert once["coverage"] > 0.95 and 0.7 < once["length_ratio"] < 1.3
    assert twice["length_ratio"] > 1.6
    assert on_ink([path, [(100, 75)]], img) == [path]


def test_segment_writes_a_manifest_and_reruns_idempotently(tmp_path):
    page = Image.new("RGB", (420, 200), (245, 240, 230))
    draw = ImageDraw.Draw(page)
    for y, xs in ((60, [(20, 120), (170, 260)]), (140, [(20, 90), (150, 300)])):
        for a, b in xs:
            draw.line([(x, y + 12 * np.sin(x / 7)) for x in range(a, b)], fill=(40, 40, 60), width=3)
    page.save(tmp_path / "page.png")
    (tmp_path / "t.txt").write_text("hello there\nhi everyone\n")
    args = Namespace(page=str(tmp_path / "page.png"), transcript=str(tmp_path / "t.txt"), overview="",
                     out=str(tmp_path / "words"), style="test", license="test")
    segment_command(args)
    segment_command(args)
    rows = read_jsonl(str(tmp_path / "words" / "manifest.jsonl"))
    assert [r["text"] for r in rows] == ["hello", "there", "hi", "everyone"]
    assert all(isinstance(r["baseline"], int) and r["baseline"] > 0 for r in rows)
