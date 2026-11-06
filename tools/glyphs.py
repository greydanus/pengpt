import argparse
import json
import os
import string
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from tools.ink import file_hash, ink_baseline, on_ink, read_jsonl, write_jsonl

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + "\"'(),.?!"
SHORT = "acemnorsuvwxz"
PROMPTS = ("Derender the ink: {text}", "Recognize and derender.")
STORE, FIXES = "data/glyphs.json", "tools/glyph_fixes.json"
SIZE, PAD = 200, 8


def slug(font_path):
    return os.path.splitext(os.path.basename(font_path))[0].lower().replace("-", "_")


def advance(font, text, i):
    return font.getlength(text[:i + 1]) - font.getlength(text[i])


def contexts(c, cursive):
    if cursive and c in string.ascii_lowercase:
        return [("low", "n" + c + "n", 1), ("high", "o" + c + "n", 1)]
    if cursive and c in string.ascii_uppercase:
        return [("low", c + "n", 0)]
    return [("low", c, 0)]


def render(font_path, out, cursive, alone=""):
    font = ImageFont.truetype(font_path, SIZE)
    os.makedirs(f"{out}/images", exist_ok=True)
    rows = []
    for k, c in enumerate(CHARS):
        for variant, text, pos in contexts(c, cursive and c not in alone):
            left, top, right, bottom = font.getbbox(text, anchor="ls")
            img = Image.new("RGB", (right - left + 2 * PAD, bottom - top + 2 * PAD), "white")
            ImageDraw.Draw(img).text((PAD - left, PAD - top), text, font=font, fill="black", anchor="ls")
            path = f"{out}/images/{k:02d}{'' if variant == 'low' else '_' + variant}.png"
            img.save(path)
            rows.append({"image": path, "char": c, "variant": variant, "text": text, "pos": pos, "baseline": PAD - top, "x0": PAD - left})
    write_jsonl(f"{out}/manifest.jsonl", rows)
    return rows


def trace(rows, out, weights, batch=16):
    from tools.inksight import InkSight
    model = InkSight(weights)
    done = {(r["hash"], r["prompt"]) for r in read_jsonl(out)}
    for prompt in PROMPTS:
        todo = [r for r in rows if (file_hash(r["image"]), prompt) not in done]
        t = time.time()
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            got = model.trace([Image.open(r["image"]) for r in chunk], [r["text"] for r in chunk], prompt)
            write_jsonl(out, [{"image": r["image"], "hash": file_hash(r["image"]), "prompt": prompt, "strokes": s, "read": read}
                              for r, (s, read) in zip(chunk, got)], "a")
        print(f"[{prompt}] {len(todo)} glyphs in {time.time() - t:.0f}s")


def cut(row, strokes, font, xh):
    text, pos = row["text"], row["pos"]
    base = row["baseline"]
    lo = row["x0"] + advance(font, text, pos) if pos else -1e9
    hi = row["x0"] + advance(font, text, pos + 1) if pos + 1 < len(text) else 1e9
    out = []
    for s in strokes:
        mid = ((s[:, 0] >= lo) & (s[:, 0] <= hi)) | (s[:, 1] < base - 1.15 * xh) | (s[:, 1] > base + 0.15 * xh)
        if mid.any():
            a, b = np.flatnonzero(mid)[[0, -1]]
            if b > a or len(s) == 1:
                out.append(s[a:b + 1])
    return out


def dots(image, lo, hi, xh):
    ink = np.asarray(Image.open(image).convert("L")) < 128
    lab, n = ndimage.label(ink)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        r = np.sqrt(len(ys) / np.pi)
        if r < 0.25 * xh and max(np.ptp(xs), np.ptp(ys)) < 1.6 * min(np.ptp(xs), np.ptp(ys)) + 2 and lo <= xs.mean() <= hi:
            out.append((xs.mean(), ys.mean(), r))
    return out


def circle(x, y, r, n=10):
    a = np.pi / 2 + np.linspace(0, 2 * np.pi, n + 1)
    return np.column_stack([x + r * np.cos(a), y - r * np.sin(a)])


def snap_dots(strokes, found, xh):
    out = list(strokes)
    for x, y, r in found:
        near = [k for k, p in enumerate(out) if np.ptp(p, 0).max() < 0.4 * xh and np.hypot(*(p.mean(0) - [x, y])) < r + 0.15 * xh]
        dot = circle(x, y, 0.5 * r)
        if near:
            out[near[0]] = dot
            out = [p for k, p in enumerate(out) if k not in near[1:]]
        else:
            out.append(dot)
    return out


def is_mark(s, c):
    ext = np.ptp(s, 0) if len(s) > 1 else np.zeros(2)
    return (c in "ij" and ext.max() < 0.4) or (c in "tx" and ext[0] > 1.5 * ext[1] and ext[1] < 0.5)


def apply_ops(strokes, ops):
    for op in ops:
        name, arg = op.split(":")
        k = [int(a) for a in arg.split(",")]
        if name == "reverse":
            strokes[k[0]] = strokes[k[0]][::-1]
        elif name == "order":
            strokes = [strokes[i] for i in k]
        elif name == "drop":
            strokes.pop(k[0])
        elif name == "join":
            strokes[k[0]] = np.vstack([strokes[k[0]], strokes[k[1]]])
            strokes.pop(k[1])
    return strokes


def build(font_path, fixes):
    s = slug(font_path)
    out, fix = f"data/raw/glyphs/{s}", fixes.get(s, {})
    font = ImageFont.truetype(font_path, SIZE)
    xh = np.median([np.ptp(font.getbbox(c, anchor="ls")[1::2]) for c in SHORT])
    shift = ink_baseline(font)
    rows = dict(sorted(((r["image"], r) for r in read_jsonl(f"{out}/manifest.jsonl")), key=lambda kv: kv[1].get("variant") == "high"))
    traces = {(r["image"], r["prompt"]): r for r in read_jsonl(f"{out}/traces.jsonl")}
    glyphs = {}
    for image, row in rows.items():
        c, g = row["char"], fix.get("glyphs", {}).get(row["char"], {})
        high = row.get("variant") == "high"
        t = traces[(image, g.get("prompt", PROMPTS[0]))]
        strokes = [np.asarray(p, float) for p in on_ink(t["strokes"], Image.open(image)) if len(p)]
        strokes = cut(row, strokes, font, xh) if high else apply_ops(cut(row, strokes, font, xh), g.get("ops", []))
        lo = row["x0"] + advance(font, row["text"], row["pos"]) if row["pos"] else -1e9
        hi = row["x0"] + advance(font, row["text"], row["pos"] + 1) if row["pos"] + 1 < len(row["text"]) else 1e9
        if c in "ij.!?":
            strokes = snap_dots(strokes, dots(image, lo, hi, xh), xh)
        origin = [row["x0"] + advance(font, row["text"], row["pos"]), row["baseline"] + shift]
        strokes = [(p - origin) / xh for p in strokes]
        if fix.get("cursive") and c.islower():
            strokes = [p for p in strokes if len(p) == 1 or np.ptp(p, 0).max() > 0.15 or is_mark(p, c)]
        marks = [k for k, p in enumerate(strokes) if is_mark(p, c)] if high else g.get("marks", [k for k, p in enumerate(strokes) if is_mark(p, c)])
        if high:
            glyphs[c]["high"] = {"strokes": [p.round(4).tolist() for p in strokes], "marks": marks}
        else:
            glyphs[c] = {**glyphs.get(c, {}), "strokes": [p.round(4).tolist() for p in strokes], "marks": marks}
    return {"font": os.path.basename(font_path), "size": SIZE, "xh": float(xh), "cursive": bool(fix.get("cursive")),
            "reviewed": bool(fix.get("reviewed")), "glyphs": glyphs}


def against_grain(entry):
    out = []
    for c, g in entry["glyphs"].items():
        for k, s in enumerate(map(np.asarray, g["strokes"])):
            if len(s) < 3 or np.hypot(*(s[0] - s[-1])) < 0.15:
                continue
            dx, dy = s[-1] - s[0]
            if dy < -0.4 and (abs(dy) > 1.5 * abs(dx) or not entry["cursive"] and abs(dx) < 2 * abs(dy)):
                out.append(f"{c}[{k}] up")
            elif abs(dx) > 1.5 * abs(dy) and dx < -0.4:
                out.append(f"{c}[{k}] leftward")
    return out


def curve(a, b, n=8):
    p0, p1 = a[-1], b[0]
    gap = np.hypot(*(p1 - p0))
    if gap < 0.05 or len(a) < 3 or len(b) < 3:
        return np.zeros((0, 2))
    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
    m0, m1 = unit(a[-1] - a[-3]) * gap, unit(b[2] - b[0]) * gap
    t = np.linspace(0, 1, n + 2)[1:-1, None]
    return (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1


def walk(s):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(s, axis=0).T))]


def splice(tail, s, reach=0.6, close=0.08):
    a, b = len(tail) - 1 - np.flatnonzero(walk(tail[::-1]) <= reach), np.flatnonzero(walk(s) <= reach)
    d = np.hypot(*(tail[a][:, None] - s[b][None]).transpose(2, 0, 1))
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return np.vstack([tail[:a[i] + 1], s[b[j]:]]) if d[i, j] < close else None


def backtrack(tail, q, reach=1.5, close=0.15):
    k = np.flatnonzero(walk(tail[::-1]) <= reach)[1:]
    if not len(k):
        return None
    d = np.hypot(*(tail[::-1][k] - q).T)
    i = k[np.argmin(d)]
    return tail[::-1][1:i + 1] if d.min() < close and i > 2 else None


def ride_in(p, s):
    head = s[:max(2, len(s) * 3 // 5)]
    j = int(np.argmin(np.hypot(*(head - p).T)))
    return np.vstack([s[j::-1], s[1:]]) if j > 0 else s


def join(tail, s, log=None):
    log = [] if log is None else log
    spliced = splice(tail, s)
    if spliced is not None:
        log.append(("splice", spliced[len(tail) - 1:len(tail)]))
        return spliced
    back = backtrack(tail, s[0])
    if back is not None:
        log.append(("backtrack", np.vstack([tail[-1:], back, s[:1]])))
        return np.vstack([tail, back, s])
    if s[0, 1] < tail[-1, 1] - 0.4 and len(s) > 3:
        rode = ride_in(tail[-1], s)
        if len(rode) > len(s):
            log.append(("ride in", rode[:len(rode) - len(s) + 1]))
        s = rode
    bridge = curve(tail, s)
    log.append(("bridge", np.vstack([tail[-1:], bridge, s[:1]])))
    return np.vstack([tail, bridge, s])


def compose(entry, text, xs, bridge=1.0, high=-0.55, log=None):
    g = entry["glyphs"]
    out, marks, prev = [], [], None
    for c, x in zip(text, xs):
        if c not in g:
            out, marks, prev = out + marks, [], None
            continue
        glyph = g[c]["high"] if "high" in g[c] and (prev or "").islower() and out and out[-1][-1, 1] < high else g[c]
        for k, s in enumerate(glyph["strokes"]):
            s = np.asarray(s) + [x, 0]
            if k in glyph["marks"]:
                (marks if entry["cursive"] else out).append(s)
                continue
            joined = k > 0 or (prev or "").islower() or ((prev or "").isupper() and np.hypot(*(out[-1][-1] - s[0])) < bridge)
            if entry["cursive"] and c.islower() and out and joined:
                out[-1] = join(out[-1], s, log)
            else:
                out.append(s)
        prev = c
    return out + marks


def layout(entry, text, font_dir="data/raw/fonts"):
    font = ImageFont.truetype(f"{font_dir}/{entry['font']}", entry["size"])
    return [advance(font, text, i) / entry["xh"] for i in range(len(text))]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def sheet(entry, path, words=("minimum", "Ithaca,", "quick brown", "Hello!", "jumpy (1184)", "Queen")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols, rows = 14, (len(CHARS) + 13) // 14
    fig = plt.figure(figsize=(18, 1.25 * rows + 1.5 * (len(words) + 1) // 2 + 0.6))
    grid = fig.add_gridspec(rows + (len(words) + 1) // 2, 2 * cols, height_ratios=[1.25] * rows + [1.5] * ((len(words) + 1) // 2),
                            left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.25, wspace=0.05)

    def draw(ax, strokes, marks=(), numbered=False):
        cmap = plt.cm.plasma if numbered else plt.cm.viridis
        for i, (s, col) in enumerate(zip(strokes, cmap(np.linspace(0, 0.85, max(1, len(strokes)))))):
            ax.plot(s[:, 0], -s[:, 1], color=col, lw=1.6, ls="--" if i in marks else "-", solid_capstyle="round")
            ax.plot(s[0, 0], -s[0, 1], "o", color=col, ms=3.5)
            if numbered and len(s) > 3:
                m = len(s) * 3 // 5
                ax.annotate("", (s[min(m + 1, len(s) - 1), 0], -s[min(m + 1, len(s) - 1), 1]), (s[m - 1, 0], -s[m - 1, 1]),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, mutation_scale=10))
            if numbered:
                ax.text(s[0, 0] - 0.06, -s[0, 1] + 0.06, str(i), fontsize=7, color=col, ha="right", clip_on=True)
        ax.set_aspect("equal")
        ax.axis("off")

    for k, c in enumerate(CHARS):
        ax = fig.add_subplot(grid[k // cols, 2 * (k % cols):2 * (k % cols) + 2])
        for y in (0, 1):
            ax.axhline(y, color="0.9", lw=0.8, zorder=0)
        draw(ax, [np.asarray(s) for s in entry["glyphs"][c]["strokes"]], entry["glyphs"][c]["marks"], True)
        ax.set_xlim(-0.25, 1.65)
        ax.set_ylim(-0.85, 2.05)
        ax.text(-0.2, -0.8, f"{c}  ({len(entry['glyphs'][c]['strokes'])})", fontsize=9, va="bottom")
    for j, w in enumerate(words):
        ax = fig.add_subplot(grid[rows + j // 2, cols * (j % 2):cols * (j % 2) + cols])
        strokes = compose(entry, w, layout(entry, w))
        draw(ax, strokes)
        pts = np.vstack(strokes)
        ax.set_xlim(pts[:, 0].min() - 0.2, pts[:, 0].min() + max(np.ptp(pts[:, 0]) + 0.4, 9))
        ax.set_ylim(-pts[:, 1].max() - 0.15, -pts[:, 1].min() + 0.15)
        ax.text(pts[:, 0].min() - 0.2, -pts[:, 1].min() + 0.1, f"{w!r}: {len(strokes)} strokes", fontsize=9, va="bottom")
    fig.suptitle(f"{entry['font']}: strokes numbered in drawing order, dot = start, arrow = direction, dashed = drawn after the word; "
                 "grey lines = baseline and x-height", fontsize=11)
    fig.savefig(path, dpi=80)


def trace_command(args):
    fixes = load(args.fixes)
    out = f"data/raw/glyphs/{slug(args.font)}"
    fix = fixes.get(slug(args.font), {})
    alone = "".join(c for c, g in fix.get("glyphs", {}).items() if g.get("context") is False)
    rows = render(args.font, out, bool(fix.get("cursive")), alone)
    trace(rows, f"{out}/traces.jsonl", args.weights, args.batch)


def build_command(args):
    entry = build(args.font, load(args.fixes))
    store = load(args.store)
    store[slug(args.font)] = entry
    json.dump(store, open(args.store, "w"))
    path = f"data/raw/glyphs/{slug(args.font)}/review.png"
    sheet(entry, path)
    print(f"{slug(args.font)}: {len(entry['glyphs'])} glyphs -> {args.store}; review sheet {path}")
    flags = against_grain(entry)
    if flags:
        print(f"  check by eye, strokes against the usual direction: {', '.join(flags)}")


def words_command(args):
    entry = load(args.store)[args.name]
    rows = []
    for text in (w.strip() for w in open(args.words) if w.strip()):
        strokes = compose(entry, text, layout(entry, text))
        rows.append({"text": text, "strokes": [s.round(4).tolist() for s in strokes], "baseline": 0.0, "style": args.name,
                     "writer": args.name, "source": entry["font"], "license": args.license})
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} words in {args.name} to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="pen strokes for a font: trace each glyph with InkSight, review, assemble words")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("trace", help="render every glyph (cursive: inside joining context) and trace it with InkSight")
    p.add_argument("--font", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--weights", default="inksight", help="folder with the port's weights.npz and spm.model")
    p.add_argument("--fixes", default=FIXES)
    p = sub.add_parser("build", help="apply review fixes, save the font's glyphs to the store, draw the review sheet")
    p.add_argument("--font", required=True)
    p.add_argument("--fixes", default=FIXES)
    p.add_argument("--store", default=STORE)
    p = sub.add_parser("words", help="assemble words from stored glyphs, as rows like tools/ink.py qc writes")
    p.add_argument("--name", required=True)
    p.add_argument("--words", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--store", default=STORE)
    args = parser.parse_args()
    {"trace": trace_command, "build": build_command, "words": words_command}[args.command](args)


if __name__ == "__main__":
    main()
