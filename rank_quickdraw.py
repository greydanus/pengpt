"""Score Quick, Draw! drawings and keep the best fraction of each class.

    python rank_quickdraw.py --raw_dir qd_raw --out data/quickdraw_top25.jsonl

Reads one .ndjson per category, renders each drawing, embeds it with CLIP,
scores it with the calibrated probe, and keeps the top fraction within each
category so class balance survives.

Built to run over the whole 50M-drawing corpus without holding it:

  - output streams to JSON Lines as each category finishes, because the kept
    quarter of the full corpus is about 12 GB and will not sit in a list
  - categories are read in chunks, so a 145k-drawing file never lands in memory
    at once
  - finished categories are recorded, so --resume picks up after a crash rather
    than repeating an hour of paid GPU time

The probe ships in data/quickdraw_probe.npz, calibrated on 210 hand-judged
drawings. Held out, the quarter it keeps is 83% good-or-better against a 48%
base rate with none of the judged junk surviving: a reliable coarse filter
rather than a fine-grained ranking.

Use pengpt.data.load_examples on the .jsonl output; --to_json writes a single
JSON array instead, which is easier to inspect but must fit in memory.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from pengpt.convert import normalize
from pengpt.quality import Embedder, load_probe

CHUNK = 20_000


def iter_drawings(path, limit=None):
    """Yield (points, label) from a Quick, Draw! ndjson, skipping truncated lines."""
    label = os.path.basename(path).split(".")[0]
    seen = 0
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.endswith("}"):
                continue
            try:
                drawing = json.loads(line)["drawing"]
            except (ValueError, KeyError):
                continue
            points = []
            for stroke in drawing:
                xs, ys = stroke[0], stroke[1]
                if not xs:
                    continue
                points.extend([float(x), float(y), 1.0] for x, y in zip(xs, ys))
                points.append([float(xs[-1]), float(ys[-1]), 0.0])
            if len(points) >= 4:
                yield np.array(points), label
                seen += 1
                if limit and seen >= limit:
                    return


def score_category(path, embedder, probe, fraction, limit, batch_size):
    """Best `fraction` of one category, as pengpt example dicts.

    Chunked so a large category never loads at once: each chunk keeps its own
    best share, which approximates a global top-fraction closely enough for
    filtering and bounds memory at CHUNK drawings.
    """
    kept, seen, chunk = [], 0, []
    label = None

    def flush(chunk):
        if not chunk:
            return []
        scores = probe.score(embedder.embed(chunk, batch_size=batch_size))
        n = max(1, int(round(fraction * len(chunk))))
        return [chunk[i] for i in np.argsort(scores)[::-1][:n]]

    for points, label in iter_drawings(path, limit):
        chunk.append(points)
        seen += 1
        if len(chunk) >= CHUNK:
            kept.extend(flush(chunk))
            chunk = []
    kept.extend(flush(chunk))
    examples = [{"text": label, "points": normalize(p).round(4).tolist()}
                for p in kept]
    return examples, seen


def main():
    p = argparse.ArgumentParser(description="Rank and filter Quick, Draw!")
    p.add_argument("--raw_dir", type=str, required=True,
                   help="directory of Quick, Draw! simplified .ndjson files")
    p.add_argument("--out", type=str, default="data/quickdraw_top25.jsonl")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--limit_per_class", type=int, default=None)
    p.add_argument("--probe", type=str, default="data/quickdraw_probe.npz")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="half precision embedding; 1.7x faster, same ranking")
    p.add_argument("--resume", action="store_true",
                   help="skip categories already present in the output")
    p.add_argument("--overwrite", action="store_true",
                   help="discard an existing output file and start again")
    p.add_argument("--to_json", action="store_true",
                   help="also write a single JSON array (must fit in memory)")
    args = p.parse_args()

    settings = np.load(args.probe)
    embedder = Embedder(name=str(settings["embedder"]), px=int(settings["px"]),
                        linewidth=float(settings["linewidth"]), fp16=args.fp16)
    probe = load_probe(args.probe)

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.ndjson")))
    if not paths:
        raise SystemExit(f"no .ndjson files in {args.raw_dir}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out):
        if args.overwrite:
            open(args.out, "w").close()
        elif args.resume:
            with open(args.out) as f:
                done = {json.loads(line)["text"] for line in f if line.strip()}
            print(f"resuming: {len(done)} categories already done")
        else:
            raise SystemExit(
                f"{args.out} already exists. Pass --resume to continue it, or "
                f"--overwrite to start again; appending to it silently would "
                f"duplicate every category it already holds.")

    todo = [p for p in paths
            if os.path.basename(p).split(".")[0] not in done]
    print(f"{len(todo)} of {len(paths)} categories to score, "
          f"keeping the best {args.fraction:.0%} of each")

    kept_total, seen_total, t0 = 0, 0, time.time()
    with open(args.out, "a") as out:
        for n, path in enumerate(todo, 1):
            label = os.path.basename(path).split(".")[0]
            examples, seen = score_category(path, embedder, probe, args.fraction,
                                            args.limit_per_class, args.batch_size)
            for example in examples:
                out.write(json.dumps(example) + "\n")
            out.flush()
            os.fsync(out.fileno())
            kept_total += len(examples)
            seen_total += seen
            elapsed = time.time() - t0
            rate = seen_total / elapsed
            eta = (len(todo) - n) * (seen_total / n) / rate / 60
            print(f"[{n}/{len(todo)}] {label:22s} {seen:7,} -> {len(examples):6,}"
                  f" | {rate:5.0f}/s | {elapsed / 60:5.1f} min elapsed"
                  f" | eta {eta:5.0f} min", flush=True)

    print(f"\nkept {kept_total:,} of {seen_total:,} scored "
          f"({kept_total / max(seen_total, 1):.0%}) in "
          f"{(time.time() - t0) / 60:.1f} min")
    if done:
        print(f"plus {len(done)} categories from an earlier run")
    print(f"-> {args.out}")

    if args.to_json:
        target = os.path.splitext(args.out)[0] + ".json"
        with open(args.out) as f:
            everything = [json.loads(line) for line in f if line.strip()]
        with open(target, "w") as f:
            json.dump(everything, f)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
