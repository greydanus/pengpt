"""Convert stroke-native SVGs (Lucide, Tabler, OpenMoji black) to pengpt JSONL.

Icon sets drawn with `fill="none" stroke=...` are already pen data: every path
is a stroke, so unlike ink_scrape.py there is no raster, no skeleton, and no
tracing loss. Paths are sampled at uniform parameter density (ScribeTokens is
insensitive to sampling density, so uniform-t is as good as uniform arc
length here) and normalized to the bundled data's scale.

Parsing uses svgelements, which handles the compressed path syntax
(".5.5", fused arc flags) that icon sets minify into.

The label is the file name with hyphens as spaces: "a-arrow-down.svg" ->
"a arrow down".

    python ink_svg.py --svgs "path/to/icons/*.svg" --out data/lucide.jsonl
"""

import argparse
import glob
import json
import os

import numpy as np
from svgelements import SVG, Path, Shape

from pengpt.convert import normalize


def svg_to_points(path, step=0.25):
    """One SVG file -> (N, 3) array in pengpt conventions, or None if empty.

    `step` is the sample spacing in SVG user units (Lucide's viewBox is 24x24,
    so 0.25 gives ~100 points across a full-width stroke).
    """
    svg = SVG.parse(path)
    points = []
    for element in svg.elements():
        if not isinstance(element, Shape):
            continue
        shape = Path(element)
        shape.reify()  # apply transforms
        for sub in shape.as_subpaths():
            sub = Path(sub)
            length = sub.length(error=1e-4)
            if length < 1e-6:
                continue
            n = max(2, int(np.ceil(length / step)))
            for x, y in sub.npoint(np.linspace(0, 1, n)):
                points.append([float(x), float(y), 1.0])
            points.append([points[-1][0], points[-1][1], 0.0])
    if len(points) < 4:
        return None
    return normalize(np.array(points), absolute=True).round(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--svgs", required=True, help="glob of stroke-based SVG files")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--step", type=float, default=0.25,
                        help="sample spacing in SVG user units")
    parser.add_argument("--label_map", default=None,
                        help="JSON dict mapping file basename (no extension) to "
                             "label; files without an entry are skipped")
    args = parser.parse_args()

    label_map = json.load(open(args.label_map)) if args.label_map else None

    files = sorted(glob.glob(os.path.expanduser(args.svgs)))
    if not files:
        raise SystemExit(f"no files match {args.svgs}")

    kept, skipped = 0, 0
    with open(args.out, "w") as f:
        for file in files:
            base = os.path.splitext(os.path.basename(file))[0]
            if label_map is not None and base not in label_map:
                skipped += 1
                continue
            try:
                points = svg_to_points(file, step=args.step)
            except Exception as e:
                print(f"skip {os.path.basename(file)}: {e}")
                skipped += 1
                continue
            if points is None:
                skipped += 1
                continue
            name = label_map[base] if label_map else base.replace("-", " ")
            f.write(json.dumps({"text": name, "points": points.tolist()}) + "\n")
            kept += 1
    print(f"wrote {kept} ({skipped} skipped) to {args.out}")

    lengths = [len(json.loads(l)["text"]) for l in open(args.out)]
    print(f"label length max {max(lengths)}, so --max_text_length {max(lengths)}")


if __name__ == "__main__":
    main()
