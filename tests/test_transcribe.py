import json
from types import SimpleNamespace

import pytest
from PIL import Image

from tools.transcribe import Transcriber, screen


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


def test_screen_drops_the_worst_sources_then_illegible_words():
    rows = [{"image": f"{i}.png", "writer": w} for i, w in enumerate(["good"] * 6 + ["bad"] * 2 + ["mid"] * 2)]
    reads = [("x", 9)] * 5 + [("y", 2)] + [("z", 3)] * 2 + [("q", 6)] * 2
    kept, dropped, score = screen(rows, reads, drop=0.2, min_legibility=4)
    assert score["bad"] == 3 and {r["writer"] for r in kept} == {"good", "mid"}
    assert sorted(r["dropped"].split()[0] for r in dropped) == ["legibility", "source", "source"]
    assert all(r["text"] for r in kept)
