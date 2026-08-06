"""Score every Quick, Draw! drawing and keep the best fraction of each class.

    python rank_quickdraw.py --raw_dir qd_raw --out data/quickdraw_top25.json

Reads one .ndjson per category, renders each drawing small and thin, embeds it
with CLIP, scores it with the calibrated probe, and keeps the top fraction
within each category so class balance survives. Writes pengpt's dataset format.

The probe ships in data/quickdraw_probe.npz, fitted on 90 hand-judged drawings.
Held out, the quarter it keeps is 82% good-or-better against a 48% base rate,
with none of the judged junk surviving -- a reliable coarse filter rather than a
fine-grained ranking.

Categories stream one at a time and only embeddings are held in memory, so the
full 50M-drawing corpus does not need to fit at once. Pass --limit_per_class to
sample rather than score everything.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from pengpt.convert import normalize
from pengpt.quality import Embedder, load_probe


def read_category(path, limit=None):
    """Quick, Draw! ndjson -> (points list, label). Skips truncated lines."""
    points, label = [], os.path.basename(path).split(".")[0]
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.endswith("}"):
                continue
            item = json.loads(line)
            strokes = []
            for xs, ys in ((s[0], s[1]) for s in item["drawing"]):
                for x, y in zip(xs, ys):
                    strokes.append([float(x), float(y), 1.0])
                strokes.append([float(xs[-1]), float(ys[-1]), 0.0])
            if len(strokes) >= 4:
                points.append(np.array(strokes))
            if limit and len(points) >= limit:
                break
    return points, label


def main():
    p = argparse.ArgumentParser(description="Rank and filter Quick, Draw!")
    p.add_argument("--raw_dir", type=str, required=True,
                   help="directory of Quick, Draw! simplified .ndjson files")
    p.add_argument("--out", type=str, default="data/quickdraw_top25.json")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--limit_per_class", type=int, default=None)
    p.add_argument("--probe", type=str, default="data/quickdraw_probe.npz")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    settings = np.load(args.probe)
    embedder = Embedder(name=str(settings["embedder"]), px=int(settings["px"]),
                        linewidth=float(settings["linewidth"]))
    probe = load_probe(args.probe)

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.ndjson")))
    if not paths:
        raise SystemExit(f"no .ndjson files in {args.raw_dir}")
    print(f"{len(paths)} categories, keeping the best {args.fraction:.0%} of each")

    kept, seen, t0 = [], 0, time.time()
    for n, path in enumerate(paths, 1):
        points, label = read_category(path, args.limit_per_class)
        if not points:
            continue
        scores = probe.score(embedder.embed(points, batch_size=args.batch_size))
        order = np.argsort(scores)[::-1][:max(1, int(round(args.fraction * len(points))))]
        for i in order:
            kept.append({"text": label,
                         "points": normalize(points[i]).round(4).tolist()})
        seen += len(points)
        rate = seen / (time.time() - t0)
        print(f"[{n}/{len(paths)}] {label:24s} {len(points):7d} -> {len(order):6d} "
              f"| {rate:.0f} drawings/s | eta {(len(paths) - n) * len(points) / rate / 60:.0f} min",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(kept, f)
    print(f"\nkept {len(kept):,} of {seen:,} drawings in {(time.time() - t0) / 60:.1f} min")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
