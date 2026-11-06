import argparse
import itertools
import json
from argparse import Namespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.glyphs import CHARS, FIXES, PROMPTS, STORE, build, build_command, corners, load, slug, walk
from tools.ink import file_hash, ink_baseline, judge, read_jsonl, write_jsonl

TEMPLATE = "tools/glyph_template.json"
RULES = Namespace(min_coverage=0.85, min_mark=0.5, min_precision=0.9, max_read_error=9.0, max_length_ratio=1.6,
                  max_retrace=0.35, max_mark_length=1.95)
WIDE = 4.0


def resample(s, n=12):
    s = np.asarray(s, float)
    d = walk(s) if len(s) > 1 else np.zeros(1)
    if d[-1] == 0:
        return np.repeat(s[:1], n, 0)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, s[:, 0]), np.interp(t, d, s[:, 1])], 1)


def normalize(strokes):
    p = np.vstack([np.asarray(s, float) for s in strokes])
    lo, size = p.min(0), np.maximum(np.ptp(p, 0), max(np.ptp(p, 0).max() * 0.6, 1e-6))
    return [resample((np.asarray(s, float) - lo) / size) for s in strokes]


def features(s):
    closed = np.linalg.norm(s[0] - s[-1]) < 0.25
    area = np.sum(s[:-1, 0] * s[1:, 1] - s[1:, 0] * s[:-1, 1])
    return s[0], s[-1], np.sign(area) if closed else 0


def stroke_cost(a, t):
    (a0, a1, aw), (t0, t1, tw) = features(a), features(t)
    w = 0.5 if aw and tw else 1.0
    return w * (np.linalg.norm(a0 - t0) + np.linalg.norm(a1 - t1)) / 2 + 0.5 * (aw * tw < 0)


def near(p, s):
    if len(s) < 2:
        return np.linalg.norm(s[0] - p)
    a, d = s[:-1], np.diff(s, axis=0)
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[:, None] * d - p, axis=1).min()


def duplicates(strokes, tol=0.06):
    on = lambda a, b: np.mean([near(p, b) for p in a]) < tol
    out = []
    for j, s in enumerate(strokes):
        if len(s) > 2 and any(i != j and i not in out and len(o) > 2 and on(s, o) and (walk(s)[-1], -j) <= (walk(o)[-1], -i)
                              for i, o in enumerate(strokes)):
            out.append(j)
    return out


def split_points(strokes, tol=0.08):
    size, out = np.ptp(np.vstack(strokes), 0).max(), []
    for k, s in enumerate(strokes):
        ends = [e for j, o in enumerate(strokes) if j != k for e in (o[0], o[-1]) if near(e, s) < tol]
        idx = set(corners(s)) | {int(np.linalg.norm(s - e, axis=1).argmin()) for e in ends}
        idx |= {int(f(s[:, d])) for f in (np.argmin, np.argmax) for d in (0, 1)}
        out += [(k, i) for i in sorted(idx) if min(walk(s[:i + 1])[-1], walk(s[i:])[-1]) > 0.15 * size]
    return out


def suggest(strokes, template, max_cost=0.3, penalty=0.04, tol=0.12):
    strokes, t = [np.asarray(s, float) for s in strokes], [np.asarray(s) for s in template]
    if len(strokes) == len(t) and np.mean([stroke_cost(a, b) for a, b in zip(normalize(strokes), t)]) <= max_cost:
        return [], True
    drops = duplicates(strokes)
    base = [s for k, s in enumerate(strokes) if k not in drops]
    ink = lambda a, b: np.linalg.norm(a - b) < tol or all(min(near(p, s) for s in base) < tol for p in a + np.linspace(0, 1, 12)[:, None] * (b - a))
    best = (np.inf, [])
    for split in [None] + split_points(base):
        pieces = base if split is None else base[:split[0]] + [base[split[0]][:split[1] + 1], base[split[0]][split[1]:]] + base[split[0] + 1:]
        n = len(pieces)
        if n < len(t) or n - len(t) > 2 or n > 5:
            continue
        norm = normalize(pieces)
        loop = split and np.linalg.norm(base[split[0]][0] - base[split[0]][-1]) < 0.25 * np.ptp(np.vstack(base), 0).max()
        for perm in itertools.permutations(range(n)):
            for flips in itertools.product((0, 1), repeat=n):
                seq = [pieces[i][::-1] if f else pieces[i] for i, f in zip(perm, flips)]
                nseq = [norm[i][::-1] if f else norm[i] for i, f in zip(perm, flips)]
                ok = [b for b in range(n - 1) if ink(seq[b][-1], seq[b + 1][0])]
                for joins in itertools.combinations(ok, n - len(t)):
                    if split and not loop and any({perm[b], perm[b + 1]} == {split[0], split[0] + 1} for b in joins):
                        continue
                    groups, cur = [], [nseq[0]]
                    for b in range(n - 1):
                        cur, groups = (cur + [nseq[b + 1]], groups) if b in joins else ([nseq[b + 1]], groups + [cur])
                    cost = np.mean([stroke_cost(np.vstack(g), s) for g, s in zip(groups + [cur], t)])
                    edits = sum(flips) + (perm != tuple(range(n))) + len(joins) + (split is not None)
                    if cost <= max_cost and cost + penalty * edits < best[0]:
                        ops = [f"split:{split[0]},{split[1]}"] if split else []
                        ops += [f"reverse:{i}" for i, f in zip(perm, flips) if f]
                        ops += [f"order:{','.join(map(str, perm))}"] if perm != tuple(range(n)) else []
                        ops += [f"join:{b},{b + 1}" for b in sorted(joins, reverse=True)]
                        best = (cost + penalty * edits, ops)
    return [f"drop:{k}" for k in sorted(drops, reverse=True)] + best[1], np.isfinite(best[0])


def retrace_wide(images, rows, out, weights):
    from tools.inksight import InkSight
    model = InkSight(weights)
    pads = [max(0, int(WIDE * Image.open(i).height) - Image.open(i).width) // 2 for i in images]
    wide = []
    for i, pad in zip(images, pads):
        img = Image.open(i).convert("RGB")
        canvas = Image.new("RGB", (img.width + 2 * pad, img.height), "white")
        canvas.paste(img, (pad, 0))
        wide.append(canvas)
    for prompt in PROMPTS:
        got = model.trace(wide, [rows[i]["char"] for i in images], prompt)
        write_jsonl(out, [{"image": i, "hash": file_hash(i), "prompt": f"{prompt} @wide", "read": read,
                           "strokes": [[[x - pad, y] for x, y in st] for st in strokes]}
                          for i, pad, (strokes, read) in zip(images, pads, got)], "a")


def choose(font_path, fixes, weights):
    name, d = slug(font_path), f"data/raw/glyphs/{slug(font_path)}"
    rows = {r["image"]: r for r in read_jsonl(f"{d}/manifest.jsonl")}
    seen = {(r["image"], r["prompt"]): r for r in read_jsonl(f"{d}/traces.jsonl")}
    q = lambda i, p: judge({**rows[i], "text": rows[i]["char"]}, seen[(i, p)], RULES)
    failed = [i for i in rows if max(q(i, p)[1]["coverage"] for p in PROMPTS) < 0.6]
    if failed and (failed[0], f"{PROMPTS[0]} @wide") not in seen:
        retrace_wide(failed, rows, f"{d}/traces.jsonl", weights)
        seen = {(r["image"], r["prompt"]): r for r in read_jsonl(f"{d}/traces.jsonl")}
    glyphs = fixes.setdefault(name, {"cursive": False, "glyphs": {}}).setdefault("glyphs", {})
    for i, row in rows.items():
        judged = {p: q(i, p) for (im, p) in seen if im == i}
        score = lambda p: (judged[p][1]["passed"], round(judged[p][1]["coverage"] * judged[p][1]["precision"], 2))
        pieces = {p: sum(len(s) > 1 for s in judged[p][0]["strokes"]) for p in PROMPTS}
        a, b = (judged[p][1] for p in PROMPTS)
        if i in failed:
            pick = max(judged, key=score)
        elif (b["passed"] and not a["passed"]) or (pieces[PROMPTS[0]] >= pieces[PROMPTS[1]] + 2 and b["coverage"] >= a["coverage"] - 0.02):
            pick = PROMPTS[1]
        else:
            continue
        g = glyphs.setdefault(row["char"], {})
        if "prompt" not in g and pick != PROMPTS[0]:
            g["prompt"], g["note"] = pick, (g.get("note", "") + f"; auto: {pick}").lstrip("; ")


def fix(font_path, fixes, template):
    name = slug(font_path)
    font_fix = fixes.setdefault(name, {"cursive": False, "glyphs": {}})
    prompts = {c: {"prompt": g["prompt"]} for c, g in font_fix.get("glyphs", {}).items() if "prompt" in g}
    raw = build(font_path, {**fixes, name: {**font_fix, "glyphs": prompts}})["glyphs"]
    glyphs = font_fix.setdefault("glyphs", {})
    for c in CHARS:
        if "ops" in glyphs.get(c, {}):
            continue
        ops, matched = suggest(raw[c]["strokes"], template["glyphs"][c])
        if ops:
            g = glyphs.setdefault(c, {})
            g["ops"], g["note"] = ops, (g.get("note", "") + "; auto: matched to the template").lstrip("; ")
            print(f"  {c}: {ops}")
        if not matched:
            print(f"  {c}: differs from the template, check by eye")


def library_sheet(entry, path, font_dir="data/raw/fonts"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font = ImageFont.truetype(f"{font_dir}/{entry['font']}", entry["size"])
    xh, shift = entry["xh"], ink_baseline(font)
    fig, axes = plt.subplots((len(CHARS) + 9) // 10, 10, figsize=(19, 16))
    for ax in axes.flat:
        ax.axis("off")
    for ax, c in zip(axes.flat, CHARS):
        left, top, right, bottom = font.getbbox(c, anchor="ls")
        img = Image.new("L", (right - left + 20, bottom - top + 20), 255)
        ImageDraw.Draw(img).text((10 - left, 10 - top), c, font=font, fill=0, anchor="ls")
        x0, base = 10 - left, 10 - top + shift
        ax.imshow(img, cmap="gray", alpha=0.18, extent=(-x0 / xh, (img.width - x0) / xh, -(img.height - base) / xh, base / xh))
        for y in (0, 1):
            ax.axhline(y, color="0.88", lw=0.7, zorder=0)
        g = entry["glyphs"][c]
        strokes = [np.asarray(s) for s in g["strokes"]]
        for i, (s, col) in enumerate(zip(strokes, plt.cm.plasma(np.linspace(0, 0.8, max(1, len(strokes)))))):
            ax.plot(s[:, 0], -s[:, 1], color=col, lw=1.8, ls="--" if i in g["marks"] else "-", solid_capstyle="round")
            ax.plot(s[0, 0], -s[0, 1], "o", color=col, ms=5)
            for m in (len(s) // 3, 2 * len(s) // 3) if len(s) > 4 else ():
                ax.annotate("", (s[m + 1, 0], -s[m + 1, 1]), (s[m - 1, 0], -s[m - 1, 1]),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=12))
            ax.text(s[0, 0] - 0.07, -s[0, 1] + 0.07, str(i), fontsize=8, color=col, ha="right", clip_on=True, weight="bold")
        ax.set_xlim(-0.3, 1.9)
        ax.set_ylim(-0.8, 2.1)
        ax.set_aspect("equal")
        ax.set_title(f"{c}   {len(strokes)} stroke{'s' * (len(strokes) != 1)}", fontsize=11)
    fig.suptitle(f"{entry['font']}: numbers = drawing order, dot = start, arrows = direction, dashed = drawn after the word", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=70)


def main():
    ap = argparse.ArgumentParser(description="print fonts: match a traced font to the stroke-order template and draw its library for sign-off")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("review", help="choose prompts, match glyphs to the template (keeps manual fixes), build, draw library.png")
    p.add_argument("--font", required=True)
    p.add_argument("--weights", default="data/raw/inksight")
    p = sub.add_parser("save", help="make a signed-off font the stroke-order template")
    p.add_argument("--font", required=True)
    args = ap.parse_args()
    if args.command == "save":
        entry = load(STORE)[slug(args.font)]
        glyphs = {c: [s.round(3).tolist() for s in normalize(entry["glyphs"][c]["strokes"])] for c in CHARS}
        return json.dump({"font": entry["font"], "glyphs": glyphs}, open(TEMPLATE, "w"))
    fixes = load(FIXES)
    if not fixes.get(slug(args.font), {}).get("reviewed"):
        choose(args.font, fixes, args.weights)
        fix(args.font, fixes, load(TEMPLATE))
        json.dump(fixes, open(FIXES, "w"), indent=2)
    build_command(Namespace(font=args.font, fixes=FIXES, store=STORE))
    path = f"data/raw/glyphs/{slug(args.font)}/library.png"
    library_sheet(load(STORE)[slug(args.font)], path)
    print(f"stroke library for sign-off: {path}")


if __name__ == "__main__":
    main()
