"""Convert the Sketchy database (HF kmewhort/sketchy-svgs) to pengpt JSONL.

    python ink_sketchy.py --raw_dir data/raw/sketchy --out data/sketchy.jsonl

Sketchy is 75k crowd-drawn object sketches over 125 ImageNet categories,
drawn on tablets. The SVGs preserve temporal stroke order -- path order is the
order the worker actually drew -- so unlike icon sets, NO reordering is
applied here: real human order is the one thing a designed corpus cannot fake,
and it is exactly what a generative pen model should learn.

Each SVG carries a Sketchy ANNOTATION comment with the category (Synset
Label), a short worker caption (Worker Tag, e.g. "blimp flying"), and the
dataset's own validity flags (Sanity Check). Sketches the original authors
flagged as Error (wrong object) or Ambiguous are dropped; Context and Shading
flags are kept but recorded in meta, since context strokes and shading are
legitimate ink that merely differs in style. The worker tag rides along in
meta as a richer caption for later experiments.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq
from svgelements import SVG, Path, Shape

from pengpt.convert import normalize

REJECT_FLAGS = ("Error", "Ambiguous")


def parse_annotation(svg_text):
    """Category, worker tag, and sanity flags from the annotation comment."""
    label = re.search(r"Synset Label:\s*(.+)", svg_text)
    tag = re.search(r"Worker Tag:\s*(.+)", svg_text)
    flags = []
    block = re.search(r"Sanity Check:\s*\n(.*?)\n\s*Comment:", svg_text, re.S)
    if block:
        flags = [line.strip() for line in block.group(1).splitlines()
                 if line.strip()]
    return (label.group(1).strip() if label else "",
            tag.group(1).strip() if tag else "", flags)


def _visible(color):
    return color is not None and color.value is not None and color.alpha != 0


def svg_to_points(svg_text, target_points=1200):
    """SVG markup -> normalized (N, 3), strokes in the order they were drawn.

    Sampling step scales with total ink length so every sketch lands near
    target_points regardless of canvas size; ScribeTokens is insensitive to
    density, so this only bounds file size, not fidelity.
    """
    svg = SVG.parse_string(svg_text) if hasattr(SVG, "parse_string") else None
    if svg is None:
        import io
        svg = SVG.parse(io.StringIO(svg_text))
    subpaths = []
    total = 0.0
    for element in svg.elements():
        if not isinstance(element, Shape):
            continue
        if not _visible(getattr(element, "stroke", None)):
            continue
        shape = Path(element)
        shape.reify()
        for sub in shape.as_subpaths():
            sub = Path(sub)
            length = sub.length(error=1e-3)
            if length > 1e-6:
                subpaths.append((sub, length))
                total += length
    if not subpaths:
        return None
    step = max(total / target_points, 1e-3)
    rows = []
    for sub, length in subpaths:
        n = max(2, int(np.ceil(length / step)))
        pts = sub.npoint(np.linspace(0, 1, n))
        for x, y in pts:
            rows.append([float(x), float(y), 1.0])
        rows.append([rows[-1][0], rows[-1][1], 0.0])
    if len(rows) < 4:
        return None
    return normalize(np.array(rows), absolute=True).round(4)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw_dir", type=str, default="data/raw/sketchy")
    p.add_argument("--out", type=str, default="data/sketchy.jsonl")
    p.add_argument("--limit", type=int, default=0, help="stop after N rows (debug)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {args.raw_dir}")

    n_in = n_out = 0
    flag_counts = {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out:
        for path in files:
            split = "test" if "test" in os.path.basename(path) else "train"
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=256, columns=["svg"]):
                for svg_text in batch.column("svg").to_pylist():
                    n_in += 1
                    label, tag, flags = parse_annotation(svg_text)
                    for f in flags:
                        flag_counts[f] = flag_counts.get(f, 0) + 1
                    if not label:
                        continue
                    if any(bad in f for f in flags for bad in REJECT_FLAGS):
                        continue
                    points = svg_to_points(svg_text)
                    if points is None:
                        continue
                    meta = {"split": split}
                    if tag and tag not in ("{}",) and tag.lower() != label.lower():
                        meta["tag"] = tag
                    kept = [f for f in flags if f != "Saved"]
                    if kept:
                        meta["flags"] = kept
                    out.write(json.dumps({"text": label,
                                          "points": points.tolist(),
                                          "meta": meta}) + "\n")
                    n_out += 1
                    if n_out % 5000 == 0:
                        print(f"  {n_out:,} converted of {n_in:,} seen", flush=True)
                    if args.limit and n_out >= args.limit:
                        break
                if args.limit and n_out >= args.limit:
                    break
            if args.limit and n_out >= args.limit:
                break
    print(f"wrote {n_out:,} of {n_in:,} sketches to {args.out}")
    print("sanity flags seen:", dict(sorted(flag_counts.items(),
                                            key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
