import json
from types import SimpleNamespace

import pytest
from PIL import Image

from tools.transcribe import Transcriber, label


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        n = kw["messages"][0]["content"][1]["text"].split("There are ")[1].split(" ")[0]
        cells = [{"n": k + 1, "text": f"w{k + 1}", "legibility": self.reply} for k in range(int(n))]
        return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=2000, output_tokens=500),
                               content=[SimpleNamespace(type="text", text=json.dumps({"cells": cells}))])


def test_transcriber_maps_cells_back_in_order_and_tracks_cost():
    tr = Transcriber(client=FakeClient(7))
    out = tr([Image.new("RGB", (60, 30), "white") for _ in range(45)])
    assert len(out) == 45 and out[0] == ("w1", 7) and out[20] == ("w1", 7) and out[44] == ("w5", 7)
    assert tr.client.calls == 3 and tr.spent == pytest.approx(3 * (2000 * 2 + 500 * 10) / 1e6)


def test_transcriber_stops_at_its_cost_cap():
    tr = Transcriber(client=FakeClient(7), max_cost=0.0)
    tr.spent = 0.01
    with pytest.raises(RuntimeError):
        tr([Image.new("RGB", (60, 30))])


def test_label_keeps_human_labels_and_drops_illegible_words():
    rows = [{"image": "a.png", "text": "cat"}, {"image": "b.png"}, {"image": "c.png"}]
    kept, dropped = label(rows, [("cot", 8), ("dog", 7), ("x", 2)], min_legibility=4)
    assert [(r["text"], r["read"]) for r in kept] == [("cat", "cot"), ("dog", "dog")]
    assert dropped[0]["dropped"] == "legibility 2"
