"""Track sample quality over training as a single contact sheet.

train.py writes out/cursive/progress/step_NNNNNN.png at each eval. This stacks them so
quality over time is visible at a glance.

    python progress.py --dir out/cursive/progress --out static/progress.png
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="out/cursive/progress")
    p.add_argument("--out", type=str, default="static/progress.png")
    p.add_argument("--max_rows", type=int, default=10)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "step_*.png")))
    if not paths:
        print(f"No step_*.png in {args.dir}")
        return
    if len(paths) > args.max_rows:
        stride = len(paths) / args.max_rows
        paths = [paths[min(int(i * stride), len(paths) - 1)] for i in range(args.max_rows)]

    fig, axes = plt.subplots(len(paths), 1, figsize=(13, 1.5 * len(paths)))
    axes = [axes] if len(paths) == 1 else axes
    for ax, path in zip(axes, paths):
        step = re.search(r"step_(\d+)", path).group(1)
        ax.imshow(mpimg.imread(path))
        ax.set_ylabel(f"{int(step):,}", fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=28)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("sample quality over training (label = step)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Saved {args.out} ({len(paths)} checkpoints)")


if __name__ == "__main__":
    main()
