import json
import string
from argparse import Namespace

import numpy as np
from PIL import Image, ImageDraw

from pengpt import INK_HEIGHT, load_examples, read_ink, save_ink
from tools.ink import (detokenize, segment_command, file_hash, levenshtein, on_ink, qc_command, read_jsonl,
                       plain, replace_rows, score, split_line, synth_words, write_jsonl)
from tools.datasets import load_traces, points_to_strokes, standardize


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


def test_points_to_strokes_splits_at_lifts():
    points = [[0.0, 0.0, 1], [0.1, 0.05, 1], [0.1, 0.05, 0], [0.3, -0.1, 1], [0.3, -0.1, 0]]
    assert points_to_strokes(points) == [[[0.0, 0.0], [0.1, 0.05]], [[0.3, -0.1]]]


def export_traces(accepted, out):
    save_ink(standardize(load_traces(accepted)), out)


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
                     min_coverage=0.85, min_mark=0.5, min_precision=0.9, max_retrace=0.35, max_mark_length=1.95, max_read_error=0.34, max_length_ratio=1.6)
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


def test_collect_export_converts_to_shared_scale(tmp_path, monkeypatch):
    import tools.datasets as datasets
    monkeypatch.setattr(datasets, "RAW", str(tmp_path))
    (tmp_path / "collect").mkdir()
    entries = []
    for i, text in enumerate(["moon", "sun", "cow", "ear", "vase", "swan"]):
        t = np.linspace(0, 6 * np.pi, 80)
        pts = [[0.2 + 0.01 * k, 0.6 - 0.05 * (1 + np.sin(a)), 1] for k, a in enumerate(t)]
        entries.append({"metadata": {"author": "me", "asciiSequence": text, "aspectRatio": 2.0},
                        "points": pts + [[pts[-1][0], pts[-1][1], 0]]})
    (tmp_path / "collect" / "export.json").write_text(json.dumps(entries))
    save_ink(datasets.standardize(datasets.load_collect()), str(tmp_path / "c.ink.npz"))
    words = load_examples(str(tmp_path / "c.ink.npz"))
    assert [w["writer"] for w in words] == ["me"] * 6 and words[0]["points"][0, 0] == 0
    from pengpt import check_data
    assert check_data(words, 0.02)[1] == []


def test_synth_words_match_the_bigbank_mix():
    words = synth_words(20000, seed=1)
    assert words == synth_words(20000, seed=1) and words != synth_words(20000, seed=2)
    share = lambda f: np.mean([bool(f(w)) for w in words])
    core = lambda w: w.strip("\"'(),.?!")
    assert 0.23 < share(lambda w: core(w).replace(".", "").replace("'", "").isdigit()) < 0.27
    assert 0.17 < share(lambda w: core(w).isalpha() and core(w).isupper()) < 0.21
    assert 0.12 < share(lambda w: w[-1] in ",.?!") < 0.2
    assert 0.08 < share(lambda w: "(" in w or ")" in w) < 0.12
    assert set("".join(words)) <= set(string.ascii_letters + string.digits + "\"'(),.?!")


def test_score_catches_a_skipped_period(tmp_path):
    img = Image.new("RGB", (240, 80), "white")
    word = [(20, 60), (60, 20), (100, 60), (140, 20), (180, 60)]
    ImageDraw.Draw(img).line(word, fill="black", width=4)
    ImageDraw.Draw(img).ellipse([(206, 56), (211, 61)], fill="black")
    img.save(tmp_path / "w.png")
    item = {"image": str(tmp_path / "w.png")}
    full, skipped = score(item, {"strokes": [word, [(209, 59)]]}), score(item, {"strokes": [word]})
    assert full["min_mark"] > 0.5 and skipped["min_mark"] == 0.0
    assert skipped["coverage"] > 0.85


def loop(cx, cy, rx, ry, turns=1.0, n=120):
    t = np.linspace(0, 2 * np.pi * turns, int(n * turns))
    return [(cx + rx * np.sin(a), cy - ry * np.cos(a)) for a in t]


def test_retrace_flags_a_doubled_loop_but_not_a_crossing(tmp_path):
    img = Image.new("RGB", (120, 160), "white")
    ImageDraw.Draw(img).line(loop(60, 80, 25, 50), fill="black", width=5)
    img.save(tmp_path / "o.png")
    eight = Image.new("RGB", (120, 160), "white")
    path = [(60 + 22 * np.sin(2 * a), 80 - 55 * np.cos(a)) for a in np.linspace(0, 2 * np.pi, 240)]
    ImageDraw.Draw(eight).line(path, fill="black", width=5)
    eight.save(tmp_path / "8.png")
    once = score({"image": str(tmp_path / "o.png")}, {"strokes": [loop(60, 80, 25, 50)]})
    twice = score({"image": str(tmp_path / "o.png")}, {"strokes": [loop(60, 80, 25, 50, turns=1.9)]})
    crossing = score({"image": str(tmp_path / "8.png")}, {"strokes": [path]})
    assert once["retrace"] < 0.1 and crossing["retrace"] < 0.2 and twice["retrace"] > 0.4


def test_mark_length_flags_one_letter_drawn_twice_beside_itself(tmp_path):
    img = Image.new("RGB", (400, 120), "white")
    letters = [[(x, 20), (x, 100)] for x in range(40, 400, 50)]
    for s in letters:
        ImageDraw.Draw(img).line(s, fill="black", width=5)
    img.save(tmp_path / "w.png")
    beside = letters + [[(42, 20), (42, 100)]]
    q = score({"image": str(tmp_path / "w.png")}, {"strokes": beside})
    assert q["length_ratio"] < 1.6 and q["max_mark_length"] > 1.95


def test_plain_treats_same_glyph_readings_as_equal():
    assert plain("0..") == plain("O..") and plain("Café") == plain("cafe")
    assert plain("cat") != plain("cot")
