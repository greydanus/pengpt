import argparse
import glob
import pickle
import random

import numpy as np

from pengpt import INK_HEIGHT, save_ink

RAW = "data/raw"


class SafeUnpickler(pickle.Unpickler):
    allowed = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("builtins", "list"), ("builtins", "str")}

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def split_strokes(points, ends):
    strokes, start = [], 0
    for i in np.flatnonzero(ends):
        strokes.append(points[start:i + 1])
        start = i + 1
    if start < len(points):
        strokes.append(points[start:])
    return [s for s in strokes if len(s)]


def standardize(samples):
    by_writer = {}
    for s in samples:
        by_writer.setdefault(s["writer"], []).append(s)
    out = []
    for group in by_writer.values():
        heights = [np.ptp(np.quantile(np.concatenate(s["strokes"])[:, 1], [0.05, 0.95])) for s in group]
        k = INK_HEIGHT / max(np.median(heights), 1e-9)
        for s in group:
            x0 = min(st[:, 0].min() for st in s["strokes"])
            strokes = [np.column_stack([(st[:, 0] - x0) * k, (st[:, 1] - s["baseline"]) * k]).round(5).tolist()
                       for st in s["strokes"]]
            out.append({"text": s["text"], "strokes": strokes, "style": s["style"],
                        "writer": s["writer"], "source": s["source"], "license": s["license"]})
    return out


def brush_words(drawing, sentence, label):
    xy, ends = drawing[:, :2].astype(float), drawing[:, 2] > 0.5
    char = label.argmax(1)
    words, i = [], 0
    for token in sentence.split(" "):
        j = sentence.index(token, i)
        if token:
            words.append((token, j, j + len(token)))
        i = j + len(token)
    for token, a, b in words:
        keep = (char >= a) & (char < b)
        if keep.sum() < 3:
            continue
        idx = np.flatnonzero(keep)
        breaks = ends[idx] | np.r_[np.diff(idx) > 1, True]
        yield token, split_strokes(xy[idx], breaks)


def load_brush():
    samples = []
    for path in sorted(glob.glob(f"{RAW}/brush/BRUSH/*/*")):
        if "_resample" in path:
            continue
        writer = path.split("/")[-2]
        with open(path, "rb") as f:
            sentence, drawing, label = SafeUnpickler(f).load()
        for token, strokes in brush_words(np.asarray(drawing), sentence, np.asarray(label)):
            samples.append({"text": token, "strokes": strokes, "baseline": 100.0, "writer": f"brush_{writer}",
                            "style": f"brush_{writer}", "source": "BRUSH",
                            "license": "BRUSH terms: non-commercial research only"})
    return samples


def bottom(strokes):
    return float(np.quantile(np.concatenate(strokes)[:, 1], 0.95))


def load_uji():
    samples, lines = [], open(f"{RAW}/uji/ujipenchars2.txt", encoding="utf-8").read().splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("WORD "):
            continue
        _, text, tag = line.split(maxsplit=2)
        n = int(lines[i + 1].split()[1])
        strokes = [np.array(lines[i + 2 + k].split("#")[1].split(), float).reshape(-1, 2) for k in range(n)]
        writer = tag.split("_")[-1].split("-")[0]
        samples.append({"text": text, "strokes": strokes, "baseline": bottom(strokes), "writer": f"uji_{writer}",
                        "style": f"uji_{writer}", "source": "UJIpenchars2 (UCI)", "license": "CC BY 4.0"})
    return samples


def load_mathwriting():
    import xml.etree.ElementTree as ET
    ns = {"i": "http://www.w3.org/2003/InkML"}
    samples = []
    for path in sorted(glob.glob(f"{RAW}/mathwriting/*/train/*.inkml") + glob.glob(f"{RAW}/mathwriting/*/valid/*.inkml")
                       + glob.glob(f"{RAW}/mathwriting/*/test/*.inkml")):
        root = ET.parse(path).getroot()
        label = next(a.text for a in root.findall("i:annotation", ns) if a.get("type") == "normalizedLabel")
        strokes = [np.array([p.split()[:2] for p in t.text.split(",")], float) for t in root.findall("i:trace", ns)]
        strokes = [s for s in strokes if len(s)]
        if label and strokes:
            samples.append({"text": label, "strokes": strokes, "baseline": bottom(strokes), "writer": "mathwriting",
                            "style": "mathwriting", "source": "MathWriting 2024", "license": "CC BY-NC-SA 4.0"})
    return samples


LOADERS = {"brush": load_brush, "uji": load_uji, "mathwriting": load_mathwriting}


def sheet(records, path, n=40, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = random.Random(seed).sample(records, min(n, len(records)))
    cols = 5
    fig, axes = plt.subplots((len(pick) + cols - 1) // cols, cols, figsize=(3 * cols, 1.0 * len(pick) / cols + 0.6),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, pick):
        for s in r["strokes"]:
            s = np.asarray(s)
            ax.plot(s[:, 0], -s[:, 1], color="black", lw=1.0) if len(s) > 1 else ax.plot(s[:, 0], -s[:, 1], "k.", ms=2)
        ax.set_aspect("equal")
        ax.set_title(f"{r['text']}  ({r['writer']})", fontsize=7, loc="left", pad=1)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, hspace=0.6, wspace=0.05)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="convert downloaded pen datasets to pengpt stroke records")
    parser.add_argument("names", nargs="+", choices=sorted(LOADERS))
    args = parser.parse_args()
    for name in args.names:
        records = standardize(LOADERS[name]())
        out = f"{RAW}/{name}/{name}.ink.npz"
        save_ink(records, out)
        sheet(records, f"{RAW}/{name}/{name}_samples.png")
        writers = len({r["writer"] for r in records})
        print(f"{name}: {len(records)} samples from {writers} writers -> {out}")


if __name__ == "__main__":
    main()
