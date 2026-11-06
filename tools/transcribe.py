import argparse
import base64
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from tools.ink import read_jsonl, write_jsonl

MODEL, PRICE = "claude-sonnet-5-5", (2.0, 10.0)
PER, COLS, CELL_W, CELL_H, MIN_HEIGHT = 20, 2, 520, 120, 16
PROMPT = ("Each numbered cell holds one handwritten word, number or token, cropped from a scan. Transcribe each cell exactly "
          "as written: keep capitalization, digits and punctuation (commas, periods, quotes, apostrophes, parentheses, hyphens). "
          "Do not correct spelling or add punctuation that isn't there. If a cell is hard to read, give your best guess. "
          "Also rate each cell's legibility from 1 to 10 as source data for learning to write by hand: 10 = crisp, high-contrast "
          "ink with every stroke clear; 7 = clearly legible with minor blur or noise; 5 = readable but faint, blurry or smudged; "
          "3 = hard to read, strokes partly lost; 1 = illegible, cut off or not handwriting.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["cells"],
          "properties": {"cells": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["n", "text", "legibility"],
              "properties": {"n": {"type": "integer"}, "text": {"type": "string"}, "legibility": {"type": "integer"}}}}}}


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def sheet(images):
    out = Image.new("RGB", (CELL_W * COLS, CELL_H * ((len(images) + COLS - 1) // COLS)), "white")
    d, font = ImageDraw.Draw(out), ImageFont.load_default(size=22)
    for k, img in enumerate(images):
        s = min((CELL_W - 60) / img.width, (CELL_H - 12) / img.height, 2.0)
        img = img.convert("RGB").resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        x, y = (k % COLS) * CELL_W, (k // COLS) * CELL_H
        d.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], outline=(200, 200, 200))
        d.text((x + 8, y + 8), str(k + 1), fill=(200, 0, 0), font=font)
        out.paste(img, (x + 50, y + (CELL_H - img.height) // 2))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


class Transcriber:

    def __init__(self, model=MODEL, max_cost=5.0, workers=6, client=None):
        if client is None:
            import anthropic
            load_env()
            client = anthropic.Anthropic()
        self.client, self.model, self.max_cost, self.workers = client, model, max_cost, workers
        self.spent, self.lock = 0.0, threading.Lock()

    def read(self, images):
        if self.spent > self.max_cost:
            raise RuntimeError(f"transcription stopped: ${self.spent:.2f} spent, cap ${self.max_cost:.2f}")
        data = base64.standard_b64encode(sheet(images)).decode()
        r = self.client.beta.messages.create(
            model=self.model, max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                                                   {"type": "text", "text": f"{PROMPT} There are {len(images)} cells."}]}])
        with self.lock:
            self.spent += r.usage.input_tokens / 1e6 * PRICE[0] + r.usage.output_tokens / 1e6 * PRICE[1]
        if r.stop_reason == "refusal":
            return [("", 0)] * len(images)
        text = next(b.text for b in r.content if b.type == "text")
        cells = {c["n"]: (c["text"].strip(), int(c["legibility"])) for c in json.loads(text)["cells"]}
        return [cells.get(k + 1, ("", 0)) for k in range(len(images))]

    def __call__(self, images):
        groups = [images[i:i + PER] for i in range(0, len(images), PER)]
        with ThreadPoolExecutor(self.workers) as ex:
            return [cell for group in ex.map(self.read, groups) for cell in group]


def label(rows, reads, min_legibility=4):
    kept, dropped = [], []
    for r, (text, leg) in zip(rows, reads):
        row = {**r, "text": r.get("text") or text, "read": text, "legibility": leg}
        reason = f"legibility {leg}" if leg < min_legibility else None if row["text"] else "no transcription"
        (dropped if reason else kept).append({**row, "dropped": reason} if reason else row)
    return kept, dropped


def main():
    p = argparse.ArgumentParser(description="label handwriting crops before tracing: Sonnet transcribes each word from the raw crop "
                                            "and rates its legibility; human labels are kept, illegible words and tiny crops are dropped")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min_legibility", type=int, default=4, help="drop words rated below this (1-10)")
    p.add_argument("--max_cost", type=float, default=5.0)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    rows, small = [], []
    for r in read_jsonl(args.manifest):
        (rows if Image.open(r["image"]).height >= MIN_HEIGHT else small).append(r)
    tr = Transcriber(args.model, args.max_cost)
    reads = tr([Image.open(r["image"]) for r in rows])
    kept, dropped = label(rows, reads, args.min_legibility)
    dropped += [{**r, "dropped": f"smaller than {MIN_HEIGHT} px"} for r in small]
    write_jsonl(args.out, kept)
    write_jsonl(os.path.splitext(args.out)[0] + "_dropped.jsonl", dropped)
    print(f"kept {len(kept)} of {len(rows) + len(small)} words ({len(dropped)} dropped) for ${tr.spent:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
