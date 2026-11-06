import argparse
import collections
import glob
import json
import os

import numpy as np

from deprecated.quickdraw import normalize


ICON_SETS = [
    ("lucide", "lucide/icons", "ISC"),
    ("tabler", "tabler-icons/icons/outline", "MIT"),
    ("iconoir", "iconoir/icons/regular", "MIT"),
    ("heroicons", "heroicons/optimized/24/outline", "MIT"),
    ("akar", "akar-icons/src/svg", "MIT"),
    ("feather", "feather/icons", "MIT"),
]

MAX_STROKES = 24
MAX_TINY = 10
TINY_LEN = 0.025
MAX_POINTS = 2200
SAMPLE_STEP = 0.25


def _visible(color):
    return color is not None and color.value is not None and color.alpha != 0


def svg_to_strokes(path):
    from svgelements import SVG, Path, Shape

    try:
        svg = SVG.parse(path)
    except Exception:
        return "unparseable"
    strokes = []
    for element in svg.elements():
        if not isinstance(element, Shape):
            continue
        if _visible(getattr(element, "fill", None)):
            return "has_fill"
        if not _visible(getattr(element, "stroke", None)):
            continue
        shape = Path(element)
        shape.reify()
        for sub in shape.as_subpaths():
            sub = Path(sub)
            length = sub.length(error=1e-4)
            if length < 1e-6:
                continue
            n = max(2, int(np.ceil(length / SAMPLE_STEP)))
            strokes.append(np.array(sub.npoint(np.linspace(0, 1, n)), dtype=float))
    if not strokes:
        return "no_strokes"
    return strokes


def order_strokes(strokes):
    remaining = list(range(len(strokes)))
    starts = np.array([s[0] for s in strokes])
    ends = np.array([s[-1] for s in strokes])
    corner = np.vstack([starts, ends]).min(axis=0)
    first = min(remaining, key=lambda i: np.hypot(*(starts[i] - corner)))
    out = [strokes[first]]
    remaining.remove(first)
    pen = out[-1][-1]
    while remaining:
        d_start = [np.hypot(*(starts[i] - pen)) for i in remaining]
        d_end = [np.hypot(*(ends[i] - pen)) for i in remaining]
        k = int(np.argmin(np.minimum(d_start, d_end)))
        i = remaining.pop(k)
        stroke = strokes[i] if d_start[k] <= d_end[k] else strokes[i][::-1]
        out.append(stroke)
        pen = stroke[-1]
    return out


def strokes_to_points(strokes):
    rows = []
    for s in strokes:
        for x, y in s:
            rows.append([x, y, 1.0])
        rows.append([s[-1][0], s[-1][1], 0.0])
    return normalize(np.array(rows)).round(4)


def sketchability(points):
    down = points[points[:, 2] == 1]
    lifts = int((points[:, 2] == 0).sum())
    if lifts > MAX_STROKES:
        return "too_many_strokes"
    if len(points) > MAX_POINTS:
        return "too_much_ink"
    if len(down) < 4:
        return "degenerate"
    tiny = 0
    for chunk in np.split(points, np.flatnonzero(points[:, 2] == 0) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            arc = np.hypot(*np.diff(chunk[:, :2], axis=0).T).sum()
            if arc < TINY_LEN:
                tiny += 1
    if tiny > MAX_TINY:
        return "too_many_tiny_strokes"
    return None


def icon_label(set_name, filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    words = name.replace("-", " ").replace("_", " ").strip()
    if set_name == "tabler" and words.startswith("brand "):
        words = words[len("brand "):] + " logo"
    return words


def _signature(points, n=32):
    down = points[points[:, 2] == 1][:, :2]
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(down, axis=0).T))]
    t = np.linspace(0, d[-1], n)
    sig = np.column_stack([np.interp(t, d, down[:, 0]),
                           np.interp(t, d, down[:, 1])])
    sig -= sig.min(axis=0)
    return sig / max(sig.max(), 1e-9)


def dedupe_icons(examples, threshold=0.035):
    by_label = collections.defaultdict(list)
    kept, dropped = [], 0
    for ex in examples:
        sig = _signature(np.array(ex["points"]))
        dup = any(np.abs(sig - other).mean() < threshold
                  for other in by_label[ex["text"]])
        if dup:
            dropped += 1
            continue
        by_label[ex["text"]].append(sig)
        kept.append(ex)
    return kept, dropped


def icon_preview(examples, out_path, n=48, title=""):
    import matplotlib.pyplot as plt
    from pengpt import draw

    rng = np.random.default_rng(0)
    picks = rng.choice(len(examples), size=min(n, len(examples)), replace=False)
    cols = 8
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.3 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, idx in enumerate(picks):
        ex = examples[idx]
        ax = axes[k // cols][k % cols]
        draw(ax, np.array(ex["points"]), color="k", linewidth=1.0)
        ax.set_title(f'{ex["text"][:26]}\n({ex["meta"]["set"]})', fontsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def icons_command(args):
    examples, rejects = [], []
    reasons = collections.Counter()
    for set_name, subdir, license_ in ICON_SETS:
        files = sorted(glob.glob(os.path.join(args.raw_dir, subdir, "*.svg")))
        n_ok = 0
        for f in files:
            strokes = svg_to_strokes(f)
            if isinstance(strokes, str):
                reasons[f"{set_name}:{strokes}"] += 1
                continue
            if not args.no_reorder:
                strokes = order_strokes(strokes)
            points = strokes_to_points(strokes)
            reason = sketchability(points)
            item = {"text": icon_label(set_name, f),
                    "points": points.tolist(),
                    "meta": {"set": set_name, "license": license_}}
            if reason:
                reasons[f"{set_name}:{reason}"] += 1
                rejects.append(item)
                continue
            examples.append(item)
            n_ok += 1
        print(f"{set_name:10s} {n_ok:5d} kept of {len(files):5d}")

    examples, n_dupes = dedupe_icons(examples)
    print(f"\ndeduped {n_dupes} near-identical same-label renditions")
    if reasons:
        print("rejections:")
        for k, v in reasons.most_common():
            print(f"  {k:36s} {v}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    labels = {ex["text"] for ex in examples}
    print(f"\nwrote {len(examples)} icons ({len(labels)} distinct labels) "
          f"to {args.out}")

    base = os.path.splitext(args.out)[0]
    icon_preview(examples, base + "_preview.png",
                 title=f"accepted ({len(examples)})")
    if rejects:
        icon_preview(rejects, base + "_rejected.png",
                     title=f"rejected ({len(rejects)}) -- verify nothing good is lost")


def main(argv=None):
    parser = argparse.ArgumentParser(description="aggregate stroke-native icon sets into one corpus")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/icons.jsonl")
    parser.add_argument("--no_reorder", action="store_true",
                    help="keep SVG file order instead of pen-travel order")

    icons_command(parser.parse_args(argv))


if __name__ == "__main__":
    main()
