import argparse
import glob
import json
import os
import pickle
import random
import zipfile
import zlib

import numpy as np

from pengpt import INK_HEIGHT, SHORT_LETTERS, SPACING, X_HEIGHT, _stroke_spans, save_ink

RAW = "data/raw"


class SafeUnpickler(pickle.Unpickler):
    allowed = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct"), ("builtins", "list"), ("builtins", "str")}

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


def catmull_rom(p, per=16):
    p = p[np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 1e-9]]
    if len(p) < 3:
        return p
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    p0, p1, p2, p3 = q[:-3], q[1:-2], q[2:-1], q[3:]
    t1 = np.hypot(*(p1 - p0).T) ** 0.5
    t2 = t1 + np.hypot(*(p2 - p1).T) ** 0.5
    t3 = t2 + np.hypot(*(p3 - p2).T) ** 0.5
    t = t1[:, None] + (t2 - t1)[:, None] * np.linspace(0, 1, per, endpoint=False)
    lerp = lambda pa, pb, ta, tb: ((tb[:, None, None] - t[..., None]) * pa[:, None] + (t[..., None] - ta[:, None, None]) * pb[:, None]) / (tb - ta)[:, None, None]
    zero = np.zeros_like(t1)
    a1, a2, a3 = lerp(p0, p1, zero, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3)
    b1 = ((t2[:, None] - t)[..., None] * a1 + (t - zero[:, None])[..., None] * a2) / t2[:, None, None]
    b2 = ((t3[:, None] - t)[..., None] * a2 + (t - t1[:, None])[..., None] * a3) / (t3 - t1)[:, None, None]
    c = ((t2[:, None] - t)[..., None] * b1 + (t - t1[:, None])[..., None] * b2) / (t2 - t1)[:, None, None]
    return np.vstack([c.reshape(-1, 2), p[-1:]])


def resample_stroke(p, spacing=SPACING):
    if len(p) < 2:
        return p
    dense = catmull_rom(np.asarray(p, float))
    if len(dense) < 2:
        return dense[:1]
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(dense, axis=0).T))]
    s = np.linspace(0, cum[-1], max(1, int(round(cum[-1] / spacing))) + 1)
    return np.column_stack([np.interp(s, cum, dense[:, 0]), np.interp(s, cum, dense[:, 1])])


def group_xheight(group):
    words = [np.ptp(np.quantile(np.concatenate(s["strokes"])[:, 1], [0.02, 0.98])) for s in group
             if len(s["text"]) >= 2 and set(s["text"]) <= SHORT_LETTERS]
    if len(words) >= 5:
        return float(np.median(words)), "words"
    letters = [h for s in group for h in s.get("xh", [])]
    if len(letters) >= 5:
        return float(np.median(letters)), "letters"
    return None, "none"


def standardize(samples):
    by_frame = {}
    for s in samples:
        by_frame.setdefault(s["frame"], []).append(s)
    out, how = [], {}
    for frame, group in by_frame.items():
        xh, how[frame] = group_xheight(group)
        if xh:
            k = X_HEIGHT / xh
            if how[frame] in ("words", "letters"):
                trial = [[resample_stroke(np.column_stack([st[:, 0] * k, st[:, 1] * k])) for st in s["strokes"]] for s in group
                         if set(s["text"]) <= SHORT_LETTERS and len(s["text"]) >= (2 if how[frame] == "words" else 1)]
                got = np.median([np.ptp(np.quantile(np.concatenate(t)[:, 1], [0.02, 0.98])) for t in trial])
                k *= X_HEIGHT / got
        else:
            heights = [np.ptp(np.quantile(np.concatenate(s["strokes"])[:, 1], [0.05, 0.95])) for s in group]
            k = INK_HEIGHT / max(np.median(heights), 1e-9)
        for s in group:
            x0, y0 = s["strokes"][0][0, 0], s["baseline"]
            strokes = [resample_stroke(np.column_stack([(st[:, 0] - x0) * k, (st[:, 1] - y0) * k])).round(5).tolist()
                       for st in s["strokes"]]
            out.append({"text": s["text"], "strokes": strokes, "style": s["style"],
                        "writer": s["writer"], "source": s["source"], "license": s["license"]})
    print(f"one scale per recording frame, x-height measured from: {how}")
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
        xh = [np.ptp(xy[char == c, 1]) for c in range(a, b) if sentence[c] in SHORT_LETTERS and (char == c).sum() > 2]
        yield token, split_strokes(xy[idx], breaks), xh


DESCENDERS = set("gjpqyQ,;")


def densify(strokes, step):
    out = []
    for s in strokes:
        s = np.asarray(s, float)
        if len(s) < 2:
            out.append(s)
            continue
        seg = np.diff(s, axis=0)
        n = np.maximum(1, (np.hypot(*seg.T) / step).astype(int))
        out.append(np.concatenate([s[i] + seg[i] * np.linspace(0, 1, n[i], endpoint=False)[:, None]
                                   for i in range(len(seg))] + [s[-1:]]))
    return np.concatenate(out)


def band_bottom(strokes, bins=40):
    y = np.concatenate(strokes)[:, 1]
    y = densify(strokes, max(np.ptp(y), 1e-6) / 200)[:, 1]
    hist, edges = np.histogram(y, bins=bins)
    hist = np.convolve(hist, np.ones(3) / 3, "same")
    j = top = hist.argmax()
    while j + 1 < bins and hist[j + 1] >= 0.5 * hist[top]:
        j += 1
    return float(edges[j + 1])


def baseline(strokes, text):
    if set(text) & DESCENDERS:
        return band_bottom(strokes)
    return float(np.quantile(np.concatenate(strokes)[:, 1], 0.95))


def load_brush():
    samples = []
    zf = zipfile.ZipFile(f"{RAW}/brush/brush.zip")
    names = sorted(n for n in zf.namelist() if n.startswith("BRUSH/") and n.count("/") == 2
                   and not n.endswith("/") and "_resample" not in n and ".DS_Store" not in n)
    tops = {}
    for name in names:
        writer = name.split("/")[1]
        sentence, drawing, label = SafeUnpickler(zf.open(name)).load()
        tops.setdefault(writer, []).append(float(np.max(np.asarray(drawing)[:, 1])))
        drawing, label = np.asarray(drawing), np.asarray(label)
        flat = np.array([sentence[c] not in DESCENDERS and sentence[c] != " " for c in label.argmax(1)])
        base = (float(np.quantile(drawing[flat, 1], 0.95)) if flat.sum() > 10
                else baseline([drawing[:, :2].astype(float)], sentence))
        for token, strokes, xh in brush_words(drawing, sentence, label):
            samples.append({"text": token, "strokes": strokes, "baseline": base, "xh": xh, "writer": f"brush_{writer}",
                            "style": f"brush_{writer}", "source": "BRUSH",
                            "license": "BRUSH terms: non-commercial research only"})
    big = {w for w, t in tops.items() if np.median(t) > 140}
    for s in samples:
        s["frame"] = "brush_big_frame" if s["writer"][6:] in big else "brush_frame"
    return samples


def load_uji():
    samples, lines = [], open(f"{RAW}/uji/ujipenchars2.txt", encoding="utf-8").read().splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("WORD "):
            continue
        _, text, tag = line.split(maxsplit=2)
        n = int(lines[i + 1].split()[1])
        strokes = [np.array(lines[i + 2 + k].split("#")[1].split(), float).reshape(-1, 2) for k in range(n)]
        writer = tag.split("_")[-1].split("-")[0]
        xh = [np.ptp(np.concatenate(strokes)[:, 1])] if text in SHORT_LETTERS else []
        samples.append({"text": text, "strokes": strokes, "baseline": baseline(strokes, text), "xh": xh, "frame": "uji", "writer": f"uji_{writer}",
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
            samples.append({"text": label, "strokes": strokes, "baseline": baseline(strokes, label), "frame": "mathwriting", "writer": "mathwriting",
                            "style": "mathwriting", "source": "MathWriting 2024", "license": "CC BY-NC-SA 4.0"})
    return samples


def points_to_strokes(points):
    points = np.asarray(points, dtype=float)
    return [points[a:b, :2].round(5).tolist() for a, b in _stroke_spans(points[:, 2])]


def load_collect():
    samples = []
    for path in sorted(glob.glob(f"{RAW}/collect/*.json")):
        for item in json.load(open(path)):
            meta = item["metadata"]
            points = np.asarray(item["points"], float) * [meta["aspectRatio"], 1, 1]
            strokes = [np.asarray(s) for s in points_to_strokes(points)]
            text, author = meta["asciiSequence"], meta["author"]
            samples.append({"text": text, "strokes": strokes, "baseline": baseline(strokes, text), "frame": f"collect_{author}",
                            "writer": author, "style": author, "source": "collect.html", "license": "unspecified"})
    return samples


def load_traces(path=f"{RAW}/traces/accepted.jsonl"):
    samples = []
    for r in map(json.loads, open(path)):
        strokes = [np.asarray(s, float) for s in r["strokes"] if len(s)]
        samples.append({"text": r["text"], "strokes": strokes, "baseline": r["baseline"], "frame": r["source"],
                        "writer": r.get("writer", r["source"]), "style": r["style"], "source": r["source"],
                        "license": r["license"]})
    return samples


def glyph_samples(entry, words, xs, name, license):
    from tools.glyphs import compose
    return [{"text": w, "strokes": [np.asarray(s, float) for s in compose(entry, w, x)], "baseline": 0.0, "frame": name,
             "writer": name, "style": name, "source": entry["font"], "license": license} for w, x in zip(words, xs)]


def font_license(font):
    for path in glob.glob(f"{RAW}/*/images/manifest.jsonl"):
        row = json.loads(open(path).readline())
        if row.get("source") == font:
            return row["license"]
    return "unspecified"


def load_glyphs(name, n):
    from tools.glyphs import STORE, layout, load
    from tools.ink import synth_words
    entry = load(STORE)[name]
    words = synth_words(n, seed=zlib.crc32(name.encode()) % 2 ** 31)
    return glyph_samples(entry, words, [layout(entry, w) for w in words], name, font_license(entry["font"]))


LOADERS = {"brush": load_brush, "uji": load_uji, "mathwriting": load_mathwriting, "collect": load_collect,
           "traces": load_traces}


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


def glyphs_main(args):
    from tools.glyphs import FIXES, load
    fixes = load(FIXES)
    names = args.fonts or [k for k, f in fixes.items() if f.get("signed_off")]
    for name in names:
        if not fixes.get(name, {}).get("signed_off"):
            raise SystemExit(f"{name}: no human sign-off on its stroke library in {FIXES}")
        records = standardize(load_glyphs(name, args.n))
        out = f"{RAW}/glyphs/{name}/{name}.ink.npz"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_ink(records, out)
        sheet(records, f"{RAW}/glyphs/{name}/{name}_samples.png")
        print(f"{name}: {len(records)} words -> {out}")


def main():
    parser = argparse.ArgumentParser(description="convert downloaded pen datasets to pengpt stroke records")
    parser.add_argument("names", nargs="+", choices=sorted(LOADERS) + ["glyphs"])
    parser.add_argument("--fonts", nargs="*", default=[], help="glyphs: store names; default every font signed off in tools/glyph_fixes.json")
    parser.add_argument("--n", type=int, default=3000, help="glyphs: words per font")
    args = parser.parse_args()
    if args.names == ["glyphs"]:
        return glyphs_main(args)
    for name in args.names:
        records = standardize(LOADERS[name]())
        out = f"{RAW}/{name}/{name}.ink.npz"
        tmp = f"{RAW}/{name}/{name}.tmp.ink.npz"
        save_ink(records, tmp)
        os.replace(tmp, out)
        sheet(records, f"{RAW}/{name}/{name}_samples.png")
        writers = len({r["writer"] for r in records})
        print(f"{name}: {len(records)} samples from {writers} writers -> {out}")


if __name__ == "__main__":
    main()
