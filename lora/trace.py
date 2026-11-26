"""Labelled word crops -> pen strokes (InkSight) -> data/me.hand.parquet, the training set.

For every crop align.py marked ok whose label is one word, InkSight traces the crop told the word
("Derender the ink: <word>"), on two cleaned versions of the image, and the repo's own judge (pick()
in inference/tracing.py: ink coverage, precision, retracing, missed marks) keeps the better trace or
rejects both. Traces are cached in data/me/traces.jsonl, so a run can stop and resume. Passing words
are put on a common baseline and x-height, page by page (standardize()), and saved in the layout
strokes.py reads. A contact sheet of traced words, data/me/traced.png, is drawn for a look.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "inference")]
from strokes import save_strokes  # noqa: E402
from tracing import TEXT, InkSight, pick, shrink, standardize, trace_round, variants  # noqa: E402


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def contact_sheet(rows, out, n=120, cols=8, cell=(260, 110)):
    rows = rows[:n]
    img = Image.new("RGB", (cols * cell[0], -(-len(rows) // cols) * cell[1]), "white")
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        x0, y0 = (i % cols) * cell[0], (i // cols) * cell[1]
        pts = np.concatenate([np.asarray(s) for s in r["strokes"]])
        lo, hi = pts.min(0), pts.max(0)
        s = min((cell[0] - 20) / max(hi[0] - lo[0], 1e-6), (cell[1] - 34) / max(hi[1] - lo[1], 1e-6))
        for st in r["strokes"]:
            p = [(x0 + 10 + (x - lo[0]) * s, y0 + 6 + (y - lo[1]) * s) for x, y in st]
            if len(p) > 1:
                d.line(p, fill=(20, 20, 20), width=2)
        d.text((x0 + 8, y0 + cell[1] - 22), r["text"], fill=(0, 110, 0))
    img.save(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="data/me")
    p.add_argument("--out", default="data/me.hand.parquet")
    p.add_argument("--writer", default="me")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="trace at most this many new crops (0: all)")
    p.add_argument("--chunk", type=int, default=256, help="crops traced between saves")
    args = p.parse_args()

    labels = [x for x in load_jsonl(os.path.join(args.data, "labels.jsonl")) if x["status"] == "ok" and " " not in x["text"]]
    cache_path = os.path.join(args.data, "traces.jsonl")
    done = {x["id"]: x for x in load_jsonl(cache_path)}
    todo = [x for x in labels if x["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(labels)} one-word crops labelled ok, {len(done)} traced before, {len(todo)} to trace")

    if todo:
        tracer = InkSight(os.path.join(ROOT, "inksight"))
        t0 = time.time()
        for c0 in range(0, len(todo), args.chunk):
            chunk = todo[c0:c0 + args.chunk]
            preps, keep = [], []
            for x in chunk:
                crop = Image.open(os.path.join(args.data, "words", x["id"] + ".png")).convert("RGB")
                img, scale = shrink(crop)
                try:
                    preps.append({"row": {"text": x["text"]}, "scale": scale, "variants": variants(img)})
                    keep.append(x)
                except ValueError:
                    done[x["id"]] = {"id": x["id"], "page": x["page"], "text": x["text"], "passed": False, "why": "no ink"}
            found = trace_round(tracer, preps, [("clean", TEXT), ("mono", TEXT)], args.batch)
            with open(cache_path, "a") as f:
                for x, prep, cands in zip(keep, preps, found):
                    best = pick((prep["variants"]["clean"], x["text"], cands, None)) or {}
                    ok = bool(best.get("key", (False,))[0])
                    rec = {"id": x["id"], "page": x["page"], "text": x["text"], "passed": ok, "candidate": best.get("candidate"),
                           "qc": best.get("qc"), "strokes": [(np.asarray(s, float) / prep["scale"]).round(2).tolist() for s in best.get("strokes", [])]}
                    done[x["id"]] = rec
                    f.write(json.dumps(rec) + "\n")
            n = c0 + len(chunk)
            rate = (time.time() - t0) / n
            passed = sum(v.get("passed", False) for v in done.values())
            print(f"  {n}/{len(todo)} traced, {passed} passed so far, {rate:.2f}s a crop, about {rate * (len(todo) - n) / 60:.0f} min left", flush=True)

    good = [v for v in done.values() if v.get("passed") and v.get("strokes")]
    print(f"{len(good)} of {len(done)} traces passed the judge")
    rows = standardize([{"text": v["text"], "strokes": [np.asarray(s, float) for s in v["strokes"]], "frame": v["page"],
                         "writer": args.writer, "style": args.writer, "source": "scan", "license": "", "baseline": None} for v in good])
    records = [{"id": f"{args.writer}/{v['id']}", "label": r["text"], "writer": args.writer, "strokes": r["strokes"]} for v, r in zip(good, rows)]
    save_strokes(records, args.out, unit=2e-3)
    contact_sheet(rows, os.path.join(args.data, "traced.png"))
    print(f"{len(records)} words -> {args.out}; a sample drawn in {os.path.join(args.data, 'traced.png')}")


if __name__ == "__main__":
    main()
