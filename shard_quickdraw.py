"""Score one shard of Quick, Draw!, end to end, in a single process.

    for i in $(seq 0 47); do
      nohup python shard_quickdraw.py $i 48 > sh_$i.log 2>&1 &
    done

Each process takes every Nth category and renders, embeds, scores and writes by
itself. Nothing is shared and nothing is sent between processes, which is the
point: a worker pool has to ship every rendered batch back to one parent, about
37 MB of pickled arrays per batch, and that left a 224-core machine 98% idle.
Independent processes reached 29,000 drawings/s where a 96-worker pool managed
2,289.

Two things matter as much as the sharding:

  - **Run from local disk.** With the virtualenv on network storage, 48
    interpreters importing transformers over FUSE took minutes each and blocked
    in `request_wait_answer`. Copy the venv, the code and the corpus to local
    disk or a RAM disk first; import goes from minutes to seconds.
  - **Keep the resume set small.** Reading a multi-gigabyte output file at
    startup to see what is done costs every process a large network read. Pass
    --done with a file of category names instead.

Shards write separate files; concatenate them when they finish.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from pengpt.convert import normalize
from pengpt.fastembed import FastEmbedder
from pengpt.quality import LinearProbe
from rank_quickdraw import iter_drawings


def load_probe(path):
    data = np.load(path)
    probe = LinearProbe(alpha=float(data["alpha"]))
    probe.weights, probe.mean, probe.std = data["weights"], data["mean"], data["std"]
    return probe, data


def main():
    p = argparse.ArgumentParser(description="Score one shard of Quick, Draw!")
    p.add_argument("shard", type=int)
    p.add_argument("n_shards", type=int)
    p.add_argument("--raw_dir", type=str, default="/dev/shm/qd")
    p.add_argument("--out_dir", type=str, default="/dev/shm/shards")
    p.add_argument("--probe", type=str, default="data/quickdraw_probe_fast.npz")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--done", type=str, default="",
                   help="file of category names already finished elsewhere")
    args = p.parse_args()

    probe, settings = load_probe(args.probe)
    emb = FastEmbedder(px=int(settings["px"]), linewidth=int(settings["linewidth"]),
                       supersample=int(settings["supersample"]), workers=1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"top25_{args.shard:02d}.jsonl")

    done = set()
    if os.path.exists(out_path) and os.path.getsize(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["text"] for line in f if line.strip()}
    if args.done and os.path.exists(args.done):
        with open(args.done) as f:
            done |= {line.strip() for line in f if line.strip()}

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.ndjson")))
    mine = [p for i, p in enumerate(paths)
            if i % args.n_shards == args.shard
            and os.path.basename(p)[:-7] not in done]
    print(f"shard {args.shard}: {len(mine)} categories", flush=True)

    kept_total = seen_total = 0
    t0 = time.time()

    def best_of(chunk):
        if not chunk:
            return []
        scores = probe.score(emb.embed(chunk, batch_size=args.batch_size))
        keep = max(1, int(round(args.fraction * len(chunk))))
        return [chunk[i] for i in np.argsort(scores)[::-1][:keep]]

    with open(out_path, "a") as out:
        for n, path in enumerate(mine, 1):
            label = os.path.basename(path)[:-7]
            seen, chunk = 0, []

            def write(points_list):
                nonlocal kept_total
                for points in points_list:
                    out.write(json.dumps({
                        "text": label,
                        "points": normalize(points).round(4).tolist()}) + "\n")
                    kept_total += 1

            for points, _ in iter_drawings(path):
                chunk.append(points)
                seen += 1
                if len(chunk) >= args.chunk:
                    write(best_of(chunk))
                    chunk = []
            write(best_of(chunk))
            out.flush()
            os.fsync(out.fileno())

            seen_total += seen
            elapsed = time.time() - t0
            rate = seen_total / elapsed
            eta = (len(mine) - n) * (seen_total / n) / rate / 60
            print(f"s{args.shard} [{n}/{len(mine)}] {label:20s} {seen:7,}"
                  f" | {rate:6.0f}/s | eta {eta:5.0f}m", flush=True)

    print(f"SHARD {args.shard} DONE kept {kept_total:,} of {seen_total:,} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
