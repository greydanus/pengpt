"""Convert raster images to pen strokes: image -> line drawing -> skeleton -> polylines.

The scrape pipeline for building a text->drawing corpus from images that were
never drawn with a pen. Each image is reduced to a line drawing (threshold for
clipart that already is one, Canny edges for photos), skeletonized to 1px
centerlines, traced into polylines, simplified, and written in the pengpt
format with pen lifts between strokes.

Stroke order is unobserved in a raster image, so it is imposed here: longest
stroke first, then greedy nearest-endpoint, which keeps pen-up travel (and so
ScribeTokens cost) low. Human stroke order is semantically meaningful in ways
this heuristic is not; a learned ordering model trained on Quick, Draw! is the
known upgrade path.

    python ink_scrape.py --images "clipart/*.png" --out data/scraped.jsonl
    python ink_scrape.py --images "clipart/*.png" --preview preview.png
"""

import argparse
import glob
import json
import os

import numpy as np
import cv2
from skimage.morphology import skeletonize

from pengpt.convert import normalize


def load_line_drawing(path, size=512, mode="auto"):
    """Read an image and return a binary line-drawing mask (True = ink).

    Line art (dark strokes on a light ground) is thresholded directly, so the
    stroke centerline survives. Anything else goes through Canny, which traces
    contours instead -- outlines, not sketches, but a usable v0 for photos.
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"could not read {path}")
    # Composite alpha onto white, or transparent clipart reads as solid black.
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3:4].astype(float) / 255.0
        rgb = rgba[:, :, :3].astype(float)
        gray = cv2.cvtColor((rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY)
    scale = size / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_fraction = (otsu > 0).mean()
    if mode == "auto":
        mode = "lineart" if dark_fraction < 0.35 else "edges"
    if mode == "lineart":
        binary = otsu > 0
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # Close 1px gaps so the skeleton is connected rather than confetti.
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        binary = edges > 0
    return skeletonize(binary)


_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skeleton):
    """Walk a 1px skeleton into polylines of (x, y) pixel coordinates.

    Endpoints are walked first so open curves come out whole; whatever remains
    is loops, which get opened at an arbitrary pixel. A walk stops at a
    junction rather than continuing through it, so a Y becomes three strokes
    meeting at a point -- which is also how a pen would draw it.
    """
    pixels = set(zip(*np.nonzero(skeleton)))
    neighbors = {p: [q for q in ((p[0] + dr, p[1] + dc) for dr, dc in _OFFSETS)
                     if q in pixels] for p in pixels}
    degree = {p: len(n) for p, n in neighbors.items()}
    unvisited = set(pixels)
    strokes = []

    def walk(start):
        path = [start]
        unvisited.discard(start)
        current = start
        while True:
            options = [n for n in neighbors[current] if n in unvisited]
            if not options:
                break
            # Prefer 4-connected continuation, which follows the drawn line
            # instead of cutting corners at diagonal pixel pairs.
            current = min(options, key=lambda n: abs(n[0] - current[0]) + abs(n[1] - current[1]))
            path.append(current)
            unvisited.discard(current)
            if degree[current] >= 3:
                break
        return path

    for group in (1, 3, 2):  # endpoints, junctions, then loop remnants
        for p in list(pixels):
            if p in unvisited and (degree[p] == group or (group == 2 and degree[p] >= 2)):
                stroke = walk(p)
                if len(stroke) >= 3:
                    strokes.append([(float(c), float(r)) for r, c in stroke])
    return strokes


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker simplification."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    chord = end - start
    length = np.hypot(*chord)
    if length < 1e-12:
        distances = np.hypot(*(points - start).T)
    else:
        offsets = points - start
        distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / length
    index = int(np.argmax(distances))
    if distances[index] > epsilon:
        left = rdp(points[:index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def order_strokes(strokes):
    """Longest first, then greedy nearest endpoint, reversing when closer."""
    remaining = sorted(strokes, key=lambda s: -len(s))
    if not remaining:
        return []
    ordered = [remaining.pop(0)]
    while remaining:
        tip = np.asarray(ordered[-1][-1])
        best, best_dist, best_flip = 0, np.inf, False
        for i, stroke in enumerate(remaining):
            for flip, end in ((False, stroke[0]), (True, stroke[-1])):
                d = np.hypot(*(np.asarray(end) - tip))
                if d < best_dist:
                    best, best_dist, best_flip = i, d, flip
        stroke = remaining.pop(best)
        ordered.append(stroke[::-1] if best_flip else stroke)
    return ordered


def image_to_points(path, size=512, mode="auto", epsilon=1.2, min_span=6.0):
    """Full pipeline for one image -> (N, 3) array in pengpt conventions."""
    skeleton = load_line_drawing(path, size=size, mode=mode)
    strokes = trace_skeleton(skeleton)
    simplified = []
    for stroke in strokes:
        s = rdp(stroke, epsilon)
        span = np.hypot(*(s.max(0) - s.min(0)))
        if len(s) >= 2 and span >= min_span:
            simplified.append(s)
    if not simplified:
        return None
    points = []
    for stroke in order_strokes(simplified):
        for x, y in stroke:
            points.append([x, y, 1.0])
        points.append([stroke[-1][0], stroke[-1][1], 0.0])
    return normalize(np.array(points), absolute=True).round(4)


def preview(paths, results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(paths)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    axes = np.atleast_2d(axes)
    for i, (path, points) in enumerate(zip(paths, results)):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(os.path.basename(path), fontsize=8)
        if points is not None:
            starts = np.where(np.r_[1, points[:-1, 2] == 0])[0]
            for a, b in zip(starts, np.r_[starts[1:], len(points)]):
                seg = points[a:b][points[a:b, 2] == 1]
                axes[1, i].plot(seg[:, 0], seg[:, 1], "k-", linewidth=1)
            axes[1, i].invert_yaxis()
            axes[1, i].set_aspect("equal")
        for ax in (axes[0, i], axes[1, i]):
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", required=True, help="glob of input images")
    parser.add_argument("--out", default=None, help="output JSONL path")
    parser.add_argument("--mode", default="auto", choices=["auto", "lineart", "edges"])
    parser.add_argument("--size", type=int, default=512, help="working resolution")
    parser.add_argument("--epsilon", type=float, default=1.2, help="RDP tolerance, px")
    parser.add_argument("--preview", default=None, help="write a comparison grid PNG")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.expanduser(args.images)))
    if not paths:
        raise SystemExit(f"no images match {args.images}")

    results = []
    for path in paths:
        try:
            points = image_to_points(path, size=args.size, mode=args.mode,
                                     epsilon=args.epsilon)
        except ValueError as e:
            print(f"skip: {e}")
            points = None
        results.append(points)
        if points is not None:
            n_strokes = int((points[:, 2] == 0).sum())
            print(f"{os.path.basename(path)}: {len(points)} points, {n_strokes} strokes")

    if args.out:
        kept = 0
        with open(args.out, "w") as f:
            for path, points in zip(paths, results):
                if points is None:
                    continue
                f.write(json.dumps({"text": "", "points": points.tolist(),
                                    "meta": {"source": os.path.basename(path)}}) + "\n")
                kept += 1
        print(f"wrote {kept}/{len(paths)} to {args.out}")

    if args.preview:
        preview(paths, results, args.preview)


if __name__ == "__main__":
    main()
