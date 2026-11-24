import json
import os
import shutil
import subprocess

import pytest

SCRIPT = """
import fs from 'node:fs'
import {Pen, npz, accelerate} from '%s'
const b = fs.readFileSync('tests/fixtures/tiny.model.npz')
const pen = new Pen(await npz(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)))
const draw = w => [...pen.raw(w.text, w.writer, [], 0, null, null, w.guidance)]
const wanted = JSON.parse(fs.readFileSync('tests/fixtures/tiny.model.json')), plain = wanted.map(draw)
await accelerate(pen)
console.log(JSON.stringify([plain, wanted.map(draw)]))
"""


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_the_browser_pen_draws_the_same_tokens_as_the_numpy_pen_with_and_without_its_wasm_kernel():
    out = subprocess.run(["node", "--input-type=module", "-e", SCRIPT % os.path.abspath("web/pen.js")], capture_output=True, text=True, check=True)
    wanted = [w["tokens"] for w in json.load(open("tests/fixtures/tiny.model.json"))]
    assert json.loads(out.stdout) == [wanted, wanted]
