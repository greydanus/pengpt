"""Convert external pen-stroke datasets to the pengpt JSON format.

The target format is a JSON list of {"text": str, "points": [[x, y, pen], ...]}
with y growing downward, the baseline near y = 0, and letter sizes matching the
bundled data (lowercase letters roughly 0.1-0.3 units tall). If your source
data has different conventions, eyeball a few converted examples with
pengpt.sampling.plot_points before training.

BRUSH (https://github.com/brownvc/decoupled-style-descriptors) support assumes
each drawing file is a pickle of [sentence, drawing, label] with drawing an
(N, 3) array of (x, y, pen). Run --probe on one file first to confirm the
layout, since this converter was written from the format docs, not tested
against a download.

Sentence-level datasets like BRUSH have one multi-word example per item, so
train with --num_words 1 and a longer --max_text_length.
"""

import argparse
import json
import os
import pickle

import numpy as np


def normalize(points, height_quantile=0.95):
    """Shift/scale points so x starts at 0 and the ink height is ~0.5 units,
    with the vertical center of mass at y = 0."""
    points = np.asarray(points, dtype=float)
    down = points[points[:, 2] == 1] if (points[:, 2] == 1).any() else points
    height = np.quantile(down[:, 1], height_quantile) - np.quantile(down[:, 1], 1 - height_quantile)
    scale = 0.5 / max(height, 1e-6)
    points[:, :2] *= scale
    points[:, 0] -= points[0, 0]
    points[:, 1] -= np.median(down[:, 1]) * scale
    return points


def convert_brush(root, max_items=None):
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
            examples.append({"text": str(sentence),
                             "points": normalize(drawing).round(4).tolist()})
            if max_items and len(examples) >= max_items:
                return examples
    return examples


def probe(path):
    with open(path, "rb") as f:
        item = pickle.load(f)
    print(f"type: {type(item)}, len: {len(item) if hasattr(item, '__len__') else '?'}")
    for i, part in enumerate(item):
        arr = np.asarray(part) if not isinstance(part, str) else part
        desc = f"str {part[:60]!r}" if isinstance(part, str) else f"array {arr.shape} {arr.dtype}"
        print(f"  [{i}]: {desc}")


def main():
    p = argparse.ArgumentParser(description="Convert datasets to pengpt format")
    p.add_argument("--brush_dir", type=str, help="root of an extracted BRUSH dataset")
    p.add_argument("--probe", type=str, help="print the structure of one pickle file")
    p.add_argument("--out", type=str, default="data/converted.json")
    p.add_argument("--max_items", type=int, default=None)
    args = p.parse_args()

    if args.probe:
        probe(args.probe)
        return
    if args.brush_dir:
        examples = convert_brush(args.brush_dir, args.max_items)
        with open(args.out, "w") as f:
            json.dump(examples, f)
        print(f"Wrote {len(examples)} examples to {args.out}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
