"""Convert external pen-stroke datasets to the pengpt JSON format.

Target format is a JSON list of {"text": str, "points": [[x, y, pen], ...]}
with y growing downward, the baseline near y = 0, and lowercase letters roughly
0.1-0.3 units tall.

Most public datasets store deltas rather than absolute positions, and most
record a pen-state flag whose polarity is the opposite of ours. Both are
detected and reported by --probe, which prints what a file contains and how the
converter would read it. Run it on one file before converting a whole corpus.

Sentence-level datasets (BRUSH, IAM) hold one multi-word sample per item rather
than one word, so raise --max_text_length when training on them. Sources that
sample in time rather than in distance also need --spacing set, to bring their
point density into the same canonical form as the bundled data.
"""

import argparse
import json
import os
import pickle

import numpy as np

from .data import INK_HEIGHT


def to_absolute(points):
    """Return absolute (x, y, pen), converting from deltas when they look like it.

    Delta-encoded ink has near-zero mean displacement and a tiny coordinate
    range; absolute ink spans the writing area. The ratio separates them by
    orders of magnitude, so the test does not need a tuned threshold.
    """
    points = np.asarray(points, dtype=float).copy()
    span = points[:, :2].max(0) - points[:, :2].min(0)
    step = np.abs(np.diff(points[:, :2], axis=0)).mean() if len(points) > 1 else 0.0
    if step > 0 and span.max() < 12 * step:
        points[:, :2] = np.cumsum(points[:, :2], axis=0)
    return points


def normalize(points, pen_down_is=1, height_quantile=0.95):
    """Scale to the bundled data's conventions: x starts at 0, baseline at y = 0.

    Ink is scaled to INK_HEIGHT because token cost per word is proportional to
    ink size: a corpus twice as large costs twice the sequence length for the
    same shapes, and the tokenizer's grid is a fixed distance.
    """
    points = to_absolute(points)
    if pen_down_is != 1:
        points[:, 2] = 1.0 - points[:, 2]
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        down = points
    height = (np.quantile(down[:, 1], height_quantile)
              - np.quantile(down[:, 1], 1 - height_quantile))
    points[:, :2] *= INK_HEIGHT / max(height, 1e-6)
    points[:, 0] -= points[0, 0]
    points[:, 1] -= np.quantile(points[points[:, 2] == 1][:, 1], 0.95)
    return points


def convert_brush(root, max_items=None, pen_down_is=1):
    examples = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            try:
                with open(path, "rb") as f:
                    item = pickle.load(f)
                sentence, drawing = item[0], np.asarray(item[1], dtype=float)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            examples.append({
                "text": str(sentence),
                "points": normalize(drawing, pen_down_is).round(4).tolist(),
            })
            if max_items and len(examples) >= max_items:
                return examples
    return examples


def probe(path):
    """Print what a file holds and how this converter would read it."""
    with open(path, "rb") as f:
        item = pickle.load(f)
    print(f"type: {type(item).__name__}, "
          f"len: {len(item) if hasattr(item, '__len__') else '?'}")
    for i, part in enumerate(item):
        if isinstance(part, str):
            print(f"  [{i}]: str {part[:60]!r}")
            continue
        arr = np.asarray(part)
        print(f"  [{i}]: array {arr.shape} {arr.dtype}")
        if arr.ndim == 2 and arr.shape[1] >= 3:
            span = arr[:, :2].max(0) - arr[:, :2].min(0)
            step = np.abs(np.diff(arr[:, :2], axis=0)).mean()
            flag = arr[:, 2]
            print(f"        xy span {span.round(3)}, mean step {step:.4f}")
            print(f"        -> reads as {'DELTAS' if span.max() < 12 * step else 'ABSOLUTE'}")
            print(f"        col 2 values {np.unique(flag)[:5]}, "
                  f"fraction==1 {np.mean(flag == 1):.3f}")
            print(f"        -> if that fraction is small, pass --pen_down_is 0")


def main():
    p = argparse.ArgumentParser(description="Convert datasets to pengpt format")
    p.add_argument("--brush_dir", type=str, help="root of an extracted BRUSH dataset")
    p.add_argument("--probe", type=str, help="print the structure of one file")
    p.add_argument("--out", type=str, default="data/converted.json")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--pen_down_is", type=int, default=1, choices=(0, 1))
    args = p.parse_args()

    if args.probe:
        probe(args.probe)
        return
    if args.brush_dir:
        examples = convert_brush(args.brush_dir, args.max_items, args.pen_down_is)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(examples, f)
        print(f"Wrote {len(examples)} examples to {args.out}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
