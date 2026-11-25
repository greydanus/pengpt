import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from inference import pen as mini

SCRIPT = """
import fs from 'node:fs'
import {Pen, npz, accelerate} from '%s'
const b = fs.readFileSync('tests/fixtures/tiny.model.npz')
const pen = new Pen(await npz(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)))
const draw = w => [...pen.raw(w.text, w.writer, [], 0, null, null, w.guidance)]
const wanted = JSON.parse(fs.readFileSync('tests/fixtures/tiny.model.json')), plain = wanted.map(draw)
await accelerate(pen)
const walks = JSON.parse(fs.readFileSync('%s'))
console.log(JSON.stringify([plain, wanted.map(draw), walks.map(w => [...pen.decode(w)])]))
"""


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_the_browser_pen_draws_the_same_tokens_and_the_same_curves_as_the_numpy_pen_with_and_without_its_wasm_kernel(tmp_path):
    page = open("web/index.html", encoding="utf-8").read()
    (tmp_path / "pen.mjs").write_text(page[page.index('id="pen">') + 9:page.index("</script>")], encoding="utf-8")
    rng, turn = np.random.default_rng(0), lambda rng, n: np.cumsum(rng.choice([-1, 0, 0, 0, 1], n)) % 8
    walks = [np.concatenate([[mini.DOWN, *turn(rng, rng.integers(0, 60)), mini.UP, *turn(rng, 3)] for _ in range(6)]).tolist() for _ in range(40)]
    walks += [[mini.DOWN, *[0] * 6, *[2] * 6, *[4] * 6, *[6] * 6, mini.UP], [mini.DOWN, *[6] * 9, *[2] * 9, mini.UP], [mini.DOWN, mini.UP], []]
    (tmp_path / "walks.json").write_text(json.dumps(walks))
    out = subprocess.run(["node", "--input-type=module", "-e", SCRIPT % (tmp_path / "pen.mjs", tmp_path / "walks.json")], capture_output=True, text=True, check=True)
    wanted = [w["tokens"] for w in json.load(open("tests/fixtures/tiny.model.json"))]
    plain, fast, drawn = json.loads(out.stdout)
    assert [plain, fast] == [wanted, wanted]
    pen = mini.Pen("tests/fixtures/tiny.model.npz")
    for walk, got in zip(walks, drawn):
        want = pen.decode(walk)
        assert len(got) == want.size and np.abs(np.array(got).reshape(-1, 3) - want).max(initial=0) < 1e-6
