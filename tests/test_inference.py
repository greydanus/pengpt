from dataclasses import asdict

import numpy as np
import pytest
import torch

from PIL import Image, ImageDraw

import pengpt as P
from inference import hand
from inference import pen as mini


def test_layout_wraps_lines_and_svg_holds_one_path_per_stroke(tmp_path):
    word = np.array([[0, 0, 1], [0.2, 0.1, 1], [0.2, 0.1, 0], [0.3, 0, 1], [0.3, 0, 0]], float)
    placed = mini.layout([word, word, word], line_width=1.0)
    assert [p[0, 1] for p in placed] == [0, 0, 0.55] and placed[1][0, 0] > 0.3
    mini.svg(placed, tmp_path / "w.svg")
    assert open(tmp_path / "w.svg").read().count("<path") == 6


def test_pen_draws_the_same_greedy_strokes_as_a_sketch_model_with_pen_rotary(tmp_path):
    torch.manual_seed(0)
    merges = [(0, 0, 10), (10, 6, 11), (2, 2, 12), (12, 0, 13)]
    st = P.ScribeTokenizer(grid=0.0125, merges=merges)
    cfg = P.ModelConfig(n_layer=2, n_head=2, n_embd=32, n_sources=2, pen_rotary=True, pen_pos_jitter=0, vocab_size=st.vocab_size, block_size=40,
                        context_vocab_size=258, context_block_size=51)
    model = P.PenTransformer(cfg)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    P.attach_pen_deltas(model, st)
    P.save_checkpoint(tmp_path / "m.pt", model, asdict(P.DataConfig(dataset="a.sketch.parquet", max_seq_length=40)), merges, step=1, sources=["", "quickdraw"])
    mini.export(tmp_path / "m.pt", tmp_path / "m.npz")
    pen = mini.Pen(tmp_path / "m.npz")
    assert pen.pen_rotary and pen.writers == ["", "quickdraw"]
    model.load_state_dict({**model.state_dict(), **{k.replace("writer_emb.", "source_emb."): torch.from_numpy(v) for k, v in pen.w.items()}})
    _, ds, data_cfg, _ = P.load_for_sampling(tmp_path / "m.pt")
    for source, text in [("", "cat"), ("quickdraw", "The Eiffel Tower")]:
        params = P.SampleParams(do_sample=False, max_tokens=data_cfg.max_seq_length - 1, source=source)
        idx = torch.tensor([[st.BOS]])
        context = torch.from_numpy(ds.encode_text(text))[None]
        out = model.generate(idx, context, params.max_tokens, do_sample=False, end_token=st.END, pad_token=st.PAD,
                             source=torch.tensor([P.source_id(ds, source)]))[0, 1:].numpy()
        end = np.flatnonzero(out == st.END)
        new, done = pen.raw(text, source, guidance=1.0)
        assert np.array_equal(new, out[:end[0] + 1] if end.size else out) and done == bool(end.size)
        assert pen.draw(text, source, 0.0).shape[1] == 3
    assert len(mini.sheet([np.zeros((0, 3)), np.array([[0.0, 0, 1], [1, 1, 1], [1, 1, 0]])], 2)) == 1


def test_guided_drawing_is_the_same_in_numpy_and_torch_and_refused_without_label_dropping(tmp_path):
    torch.manual_seed(0)
    st = P.ScribeTokenizer(grid=0.0125, merges=[(0, 0, 10)])
    cfg = P.ModelConfig(n_layer=2, n_head=2, n_embd=32, n_sources=2, pen_rotary=True, pen_pos_jitter=0, vocab_size=st.vocab_size, block_size=24,
                        context_vocab_size=258, context_block_size=51)
    model = P.PenTransformer(cfg)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    P.attach_pen_deltas(model, st)
    for name, drop in (("plain", 0.0), ("dropped", 0.1)):
        P.save_checkpoint(tmp_path / f"{name}.pt", model, asdict(P.DataConfig(dataset="a.sketch.parquet", max_seq_length=24, label_drop=drop)), st.merges,
                                  step=1, sources=["", "a"])
        mini.export(tmp_path / f"{name}.pt", tmp_path / f"{name}.npz")
    plain, pen = mini.Pen(tmp_path / "plain.npz"), mini.Pen(tmp_path / "dropped.npz")
    assert np.array_equal(plain.raw("cat")[0], pen.raw("cat", guidance=1.0)[0])
    with pytest.raises(ValueError):
        plain.raw("cat", guidance=2.0)
    model.load_state_dict({**model.state_dict(), **{k.replace("writer_emb.", "source_emb."): torch.from_numpy(v) for k, v in pen.w.items()}})
    _, ds, data_cfg, _ = P.load_for_sampling(tmp_path / "dropped.pt")
    blank = torch.from_numpy(ds.encode_text(""))[None]
    for text, source, g in (("cat", "", 2.0), ("The Eiffel Tower", "a", 3.5), ("dog", "a", 1.0)):
        out = model.generate(torch.tensor([[st.BOS]]), torch.from_numpy(ds.encode_text(text))[None], 23, do_sample=False, end_token=st.END,
                             pad_token=st.PAD,
                             source=torch.tensor([P.source_id(ds, source)]), guidance=g, blank=blank)[0, 1:].numpy()
        end = np.flatnonzero(out == st.END)
        assert np.array_equal(pen.raw(text, source, guidance=g)[0], out[:end[0] + 1] if end.size else out)
    two = model.generate(torch.full((2, 1), st.BOS), torch.stack([torch.from_numpy(ds.encode_text(x)) for x in ("cat", "dog")]), 23, do_sample=False,
                         end_token=st.END, pad_token=st.PAD, guidance=2.0, blank=blank)
    assert two.shape[0] == 2
    assert len(P.generate(model, ds, "cat", P.SampleParams(do_sample=False, max_tokens=23)))


def test_a_model_packed_earlier_still_draws_the_same_tokens_and_a_newer_format_is_refused(tmp_path):
    import json
    pen = mini.Pen("tests/fixtures/tiny.model.npz")
    for want in json.load(open("tests/fixtures/tiny.model.json")):
        assert pen.raw(want["text"], want["writer"], guidance=want["guidance"])[0].tolist() == want["tokens"]
    z = dict(np.load("tests/fixtures/tiny.model.npz"))
    np.savez(tmp_path / "new.npz", **{**z, "meta": json.dumps({**json.loads(str(z["meta"])), "format": mini.FORMAT + 1})})
    with pytest.raises(SystemExit):
        mini.Pen(tmp_path / "new.npz")


def test_ink_and_stroke_hints_reach_a_label_model_the_same_in_numpy_and_torch(tmp_path):
    torch.manual_seed(0)
    st = P.ScribeTokenizer(grid=0.0125, merges=[(0, 0, 10)])
    data = P.DataConfig(dataset="a.sketch.parquet", max_seq_length=24, hint=0.2, label_drop=0.1)
    told = P.Labels(data)
    model = P.PenTransformer(P.ModelConfig(n_layer=2, n_head=2, n_embd=32, n_sources=2, pen_rotary=True, pen_pos_jitter=0, vocab_size=st.vocab_size,
                                           block_size=24, **told.model_fields()))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    P.attach_pen_deltas(model, st)
    P.save_checkpoint(tmp_path / "m.pt", model, asdict(data), st.merges, step=1, sources=["", "a"])
    mini.export(tmp_path / "m.pt", tmp_path / "m.npz", bits=8)
    pen = mini.Pen(tmp_path / "m.npz")
    model.load_state_dict({**model.state_dict(), **{k.replace("writer_emb.", "source_emb."): torch.from_numpy(v) for k, v in pen.w.items()}})
    _, ds, _, _ = P.load_for_sampling(tmp_path / "m.pt")
    for text, ink, strokes in (("cat", None, None), ("a very long label that runs past the room the hints leave it", 4.6, 17), ("dog", None, 5)):
        ds.ask = (ink, strokes)
        out = model.generate(torch.tensor([[st.BOS]]), torch.from_numpy(ds.encode_text(text))[None], 23, do_sample=False, end_token=st.END,
                             pad_token=st.PAD, source=torch.tensor([1]), guidance=2.0, blank=torch.from_numpy(ds.encode_text(""))[None])[0, 1:].numpy()
        end = np.flatnonzero(out == st.END)
        assert np.array_equal(pen.raw(text, "a", guidance=2.0, ink=ink, strokes=strokes)[0], out[:end[0] + 1] if end.size else out)


def test_find_words_boxes_each_word_and_skips_ink_cut_by_the_edge():
    img = Image.new("RGB", (900, 300), "white")
    d = ImageDraw.Draw(img)
    for x0 in (60, 330, 640):
        for k in range(4):
            d.ellipse([x0 + 45 * k, 120, x0 + 45 * k + 34, 160], outline="black", width=4)
    d.line([0, 40, 80, 40], fill="black", width=4)
    crops, h = hand.find_words(img)
    assert len(crops) == 3 and all(150 < c.width < 260 for c in crops)


def test_fit_style_lowers_the_loss_and_the_pen_writes_with_the_vector(tmp_path):
    torch.manual_seed(0)
    merges = [(0, 0, 10), (10, 6, 11)]
    st = P.ScribeTokenizer(merges=merges)
    cfg = P.ModelConfig(n_layer=2, n_head=2, n_embd=32, n_sources=3, vocab_size=st.vocab_size, block_size=64,
                        context_vocab_size=258, context_block_size=51)
    model = P.PenTransformer(cfg)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    P.attach_pen_deltas(model, st)
    P.save_checkpoint(tmp_path / "m.pt", model, asdict(P.DataConfig(max_seq_length=64)), merges, step=1, sources=["", "ann", "bo"])
    hand.mini.export(tmp_path / "m.pt", tmp_path / "pen.npz")
    pen = hand.mini.Pen(tmp_path / "pen.npz")
    word = np.array([[0, 0, 1], [0.1, 0, 1], [0.1, -0.1, 1], [0.1, -0.1, 0], [0.2, 0, 1], [0.3, 0.06, 1], [0.3, 0.06, 0]], float)
    style, temperature = hand.fit_style(pen, ["ab", "ba"], [word, word * [1, 0.5, 1]], steps=5)
    assert style.shape == (32,) and not np.allclose(style, pen.w["writer_emb.weight"][0]) and 0.05 <= temperature <= 1.5
    assert [t.tolist() for t in pen.tokens("ab", pen.w["writer_emb.weight"][2])] == [t.tolist() for t in pen.tokens("ab", "bo")]


def test_draw_renders_soft_strokes_at_the_requested_scale():
    word = np.array([[0, 0, 1], [0.4, -0.2, 1], [0.4, -0.2, 0]], float)
    img = np.asarray(hand.draw([word], scale=200, width=4.0, ink=60))
    assert img.shape[1] in range(120, 135) and img.min() < 90 and len(np.unique(img)) > 20


def test_load_photo_turns_a_big_rotated_colour_photo_into_a_small_upright_grey_one(tmp_path):
    img = Image.new("RGB", (4000, 3000), (230, 220, 200))
    ImageDraw.Draw(img).rectangle([0, 0, 1000, 400], fill=(40, 30, 60))
    exif = img.getexif()
    exif[274] = 6
    img.save(tmp_path / "p.jpg", exif=exif)
    out, plain = hand.load_photo(tmp_path / "p.jpg")
    assert out.mode == "L" and plain.mode == "RGB" and out.size == plain.size == (1500, 2000)
    a = np.asarray(out)
    assert a[100, 1400] < 60 and a[1900, 100] > 200


def test_packing_gives_each_matrix_the_bits_it_can_bear_and_the_pen_still_writes(tmp_path, monkeypatch):
    from strokes import save_strokes
    from tools import pack
    monkeypatch.chdir(tmp_path)
    word = lambda k: [np.array([[0, 0], [0.05, -0.1], [0.1, 0], [0.15 + 0.01 * (k % 5), -0.12]]), np.array([[0.2, -0.05], [0.3, -0.02]])]
    save_strokes([{"id": f"w/{k}", "label": ("man", "mom", "ran")[k % 3], "writer": f"w{k % 2}", "strokes": word(k)} for k in range(200)],
                 "h.hand.parquet", 2e-3, "words")
    P.main(["train", "--dataset", "h.hand.parquet", "--n_merges", "8", "--n_layer", "2", "--n_embd", "32", "--batch_size", "8", "--steps", "2",
            "--eval_every", "2", "--device", "cpu", "--no-compile", "--out_dir", "run"])
    plan = pack.pack("run/last.pt", "small.npz", bits=5.0, passes=2)
    size = {n: p.numel() for n, p in P.load_checkpoint("run/last.pt")[0].named_parameters() if p.ndim == 2}
    assert set(plan) == set(size) and set(plan.values()) <= set(pack.TIERS) and len(set(plan.values())) > 1
    assert sum(plan[n] * size[n] for n in size) <= 5.0 * sum(size.values())
    z = np.load("small.npz")
    assert all(np.abs(z[n.replace("source_emb.", "writer_emb.")]).max() <= 2 ** (b - 1) - 1 for n, b in plan.items())
    pen = mini.Pen("small.npz")
    assert pen.writers == ["", "w0", "w1"] and all(w.shape[1] == 3 for w in pen.write("man ran", "w1"))


def test_a_model_told_vectors_draws_the_same_in_numpy_and_torch(tmp_path):
    torch.manual_seed(2)
    st, D, N = P.ScribeTokenizer(grid=0.0125, merges=[(0, 0, 10), (10, 6, 11)]), 8, len(P.KINDS)
    data = P.DataConfig(dataset="a.sketch.parquet", max_seq_length=24, vectors="siglip", hint=0.2, label_drop=0.1)
    told = P.Vectors(data, D, N)
    model = P.PenTransformer(P.ModelConfig(n_layer=2, n_head=2, n_embd=32, n_sources=2, pen_rotary=True, pen_pos_jitter=0, vocab_size=st.vocab_size,
                                           block_size=24, **told.model_fields()))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    P.attach_pen_deltas(model, st)
    P.save_checkpoint(tmp_path / "m.pt", model, asdict(data), st.merges, step=1, sources=["", "quickdraw"])
    mini.export(tmp_path / "m.pt", tmp_path / "m.npz", bits=8)
    pen = mini.Pen(tmp_path / "m.npz")
    assert pen.vector == D and pen.format == 2
    model.load_state_dict({**model.state_dict(), **{k.replace("writer_emb.", "source_emb."): torch.from_numpy(v) for k, v in pen.w.items()}})
    _, ds, _, _ = P.load_for_sampling(tmp_path / "m.pt")
    vectors = torch.nn.functional.normalize(torch.randn(3, D), dim=-1).numpy()
    for v, source, ink, strokes, g in ((vectors[0], "", None, None, 2.0), (vectors[1], "quickdraw", 4.6, 17, 3.0), (vectors[2], "", None, 5, 1.0)):
        ds.ask = (ink, strokes)
        out = model.generate(torch.tensor([[st.BOS]]), torch.from_numpy(ds.encode_text(v))[None], 23, do_sample=False, end_token=st.END,
                             pad_token=st.PAD, source=torch.tensor([P.source_id(ds, source)]), guidance=g,
                             blank=torch.from_numpy(ds.encode_text(""))[None])[0, 1:].numpy()
        end = np.flatnonzero(out == st.END)
        assert np.array_equal(pen.raw(v, source, guidance=g, ink=ink, strokes=strokes)[0], out[:end[0] + 1] if end.size else out)
    pen.encoder = lambda text: vectors[len(text) % 3]
    assert pen.draw("a cat", temperature=0.0).shape[1] == 3



def test_a_published_checkpoint_has_the_packed_file_s_writers_in_its_order_and_no_dataset_names(tmp_path):
    from tools import pack
    torch.manual_seed(0)
    st = P.ScribeTokenizer(merges=[(0, 0, 10)])
    names = ["", "cedar_0001", "notes_ann", "sam", "loc_bo"]
    model = P.PenTransformer(P.ModelConfig(n_layer=1, n_head=2, n_embd=32, n_sources=len(names), vocab_size=st.vocab_size, block_size=64,
                                           context_vocab_size=258, context_block_size=51))
    P.attach_pen_deltas(model, st)
    data = asdict(P.DataConfig(dataset="data/cedar.hand.parquet,data/notes.hand.parquet", max_seq_length=64))
    P.save_checkpoint(tmp_path / "m.pt", model, data, st.merges, optimizers=P.make_optimizers(model, P.TrainConfig()), step=1, sources=names)
    mini.export(tmp_path / "m.pt", tmp_path / "m.npz")
    told = pack.anonymous(tmp_path / "m.npz", ("sam",))
    pack.publish(tmp_path / "m.pt", tmp_path / "public.pt", True, ("sam",))
    pen, c = mini.Pen(tmp_path / "m.npz"), torch.load(tmp_path / "public.pt", weights_only=True)
    assert c["sources"] == pen.writers and "optimizer" not in c and c["data_config"]["dataset"] == "released.hand.parquet"
    assert set(c["sources"]) & set(names) == {"", "sam"} and not any(n.encode() in open(tmp_path / "public.pt", "rb").read() for n in names if n not in ("", "sam"))
    for old, new in told.items():
        row = c["model"]["source_emb.weight"][c["sources"].index(new)]
        assert torch.equal(row, model.source_emb.weight[names.index(old)]) and np.abs(pen.w["writer_emb.weight"][pen.writers.index(new)] - row.numpy()).max() < 0.02
    loaded, ds, _, _ = P.load_for_sampling(tmp_path / "public.pt")
    assert ds.hand and P.source_id(ds, "sam") == c["sources"].index("sam")
    pack.publish(tmp_path / "m.pt", tmp_path / "plain.pt")
    plain = torch.load(tmp_path / "plain.pt", weights_only=True)
    assert plain["sources"] == names and "optimizer" not in plain and plain["data_config"]["dataset"] == data["dataset"]


def test_a_missing_file_is_fetched_from_the_release_under_its_name_there_and_a_file_at_hand_is_left_alone(tmp_path, monkeypatch, capsys):
    asked = []
    monkeypatch.setattr(mini, "FETCHED", [])
    monkeypatch.setattr(mini.urllib.request, "urlretrieve", lambda url, to: (asked.append(url), open(to, "w").write("w"))[0])
    path = mini.fetch(f"{tmp_path}/inksight/weights.npz", "inksight-weights.npz")
    assert asked == [f"https://github.com/{mini.REPO}/releases/download/{mini.RELEASE}/inksight-weights.npz"] and open(path).read() == "w"
    assert not list(tmp_path.glob("inksight/*.part")) and mini.fetch(path, "inksight-weights.npz") == path and len(asked) == 1
    mini.fetch(f"{tmp_path}/hand.npz")
    assert asked[-1].endswith("/hand.npz")
    said = capsys.readouterr().out
    assert said.count("30 to 60 seconds") == 1 and said.index("30 to 60 seconds") < said.index("downloading it") and mini.FETCHED == ["inksight-weights.npz", "hand.npz"]
