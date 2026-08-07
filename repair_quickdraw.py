"""Undo the cumsum corruption in Quick, Draw! files written before the fix.

    python repair_quickdraw.py --src data/quickdraw_balanced.jsonl.gz \
        --out data/quickdraw_balanced_fixed.jsonl.gz

Before `normalize(..., absolute=True)`, the delta autodetection in to_absolute
read sparse absolute drawings -- a square is five points after RDP
simplification, so its span is small relative to its mean step -- as
delta-encoded and cumsum'd them into a diagonal staircase. That hit 24% of the
balanced corpus, concentrated in the categories whose drawings are a few
straight lines (ladder 96%, triangle 95%, square 92%).

The corruption is exactly invertible. Raw Quick, Draw! coordinates are
nonnegative, so a cumsum'd drawing is nondecreasing in both x and y at every
row, which genuine drawings essentially never are: that is the detector. And
since stored = affine(cumsum(original)), each row of diff(stored) is one
original point up to the affine scale, which renormalizing restores. Only the
first point of each corrupted drawing is lost.

The one false positive is a drawing that genuinely moves down-right the whole
way (some of the `line` category). Its repair collapses to a single spot, so
anything whose repaired ink has no extent keeps its original points instead.

Built to run over the 12.7M-example filtered corpus in minutes, not hours:
lines that need no repair pass through verbatim rather than being re-parsed
into JSON on the way out, and each worker gzip-compresses its own chunk --
a gzip file is a valid concatenation of gzip members, so compression
parallelizes with everything else and the writer only appends bytes.
"""

import argparse
import gzip
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pengpt.data import INK_HEIGHT


def is_corrupt(points):
    d = np.diff(points[:, :2], axis=0)
    return len(points) > 2 and (d >= -2e-4).all()


def repair(points):
    """diff undoes the cumsum; renormalize to the bundled scale. None if the
    repair collapses, which marks a genuine monotone drawing, not a corrupt one."""
    xy = np.diff(points[:, :2], axis=0)
    out = np.column_stack([xy, points[1:, 2]])
    down = out[out[:, 2] == 1]
    if len(down) < 2 or np.ptp(down[:, 0]) + np.ptp(down[:, 1]) < 1e-6:
        return None
    h = np.quantile(down[:, 1], 0.95) - np.quantile(down[:, 1], 0.05)
    out[:, :2] *= INK_HEIGHT / max(h, 1e-6)
    out[:, 0] -= out[0, 0]
    out[:, 1] -= np.quantile(out[out[:, 2] == 1][:, 1], 0.95)
    return out


def repair_chunk(lines):
    """(compressed jsonl bytes, examples, repaired) for one block of lines."""
    out, repaired = [], 0
    for line in lines:
        points = np.array(json.loads(line)["points"], dtype=float)
        if is_corrupt(points):
            fixed = repair(points)
            if fixed is not None:
                item = json.loads(line)
                item["points"] = fixed.round(4).tolist()
                line = json.dumps(item)
                repaired += 1
        out.append(line)
    data = gzip.compress(("\n".join(out) + "\n").encode(), compresslevel=6)
    return data, len(out), repaired


def read_chunks(path, chunk):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        block = []
        for line in f:
            line = line.strip()
            if line:
                block.append(line)
            if len(block) >= chunk:
                yield block
                block = []
        if block:
            yield block


def main():
    p = argparse.ArgumentParser(description="Repair cumsum-corrupted Quick, Draw! files")
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 3))
    p.add_argument("--chunk", type=int, default=10_000)
    args = p.parse_args()

    n = repaired = 0
    t0 = time.time()
    # A bounded window of in-flight chunks: Pool.imap's feeder thread would read
    # the whole file ahead of the workers, so submission is throttled by hand.
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "wb") as out:
        window = deque()
        for block in read_chunks(args.src, args.chunk):
            window.append(pool.submit(repair_chunk, block))
            if len(window) >= 2 * args.workers:
                data, count, fixed = window.popleft().result()
                out.write(data)
                n += count
                repaired += fixed
                if n % 500_000 < args.chunk:
                    rate = n / (time.time() - t0)
                    print(f"  {n:,} examples, {repaired:,} repaired, {rate:,.0f}/s",
                          flush=True)
        for future in window:
            data, count, fixed = future.result()
            out.write(data)
            n += count
            repaired += fixed
    print(f"{n:,} examples -> {args.out}: {repaired:,} repaired "
          f"({repaired / max(n, 1):.1%}) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
