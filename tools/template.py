import argparse
import itertools
import json

import numpy as np

from tools.glyphs import CHARS, FIXES, STORE, build, corners, load

TEMPLATE = "tools/glyph_template.json"


def resample(s, n=12):
    s = np.asarray(s, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))] if len(s) > 1 else np.zeros(1)
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


def save(entry, path=TEMPLATE):
    glyphs = {c: [s.round(3).tolist() for s in normalize(entry["glyphs"][c]["strokes"])] for c in CHARS}
    json.dump({"font": entry["font"], "glyphs": glyphs}, open(path, "w"))


def stroke_cost(a, t):
    (a0, a1, aw), (t0, t1, tw) = features(a), features(t)
    w = 0.5 if aw and tw else 1.0
    return w * (np.linalg.norm(a0 - t0) + np.linalg.norm(a1 - t1)) / 2 + 0.5 * (aw * tw < 0)


def compare(strokes, template, max_cost=0.3):
    a, t = normalize(strokes), [np.asarray(s) for s in template]
    if len(a) != len(t):
        return f"{len(a)} strokes, template {len(t)}", None
    cost = lambda perm, flips: np.mean([stroke_cost(a[i][::-1] if f else a[i], t[j])
                                        for j, (i, f) in enumerate(zip(perm, flips))])
    perms = itertools.permutations(range(len(a))) if len(a) <= 4 else [tuple(range(len(a)))]
    best = min(((cost(p, f), p, f) for p in perms for f in itertools.product((0, 1), repeat=len(a))), key=lambda x: x[0])
    if best[0] > max_cost:
        return f"starts or ends elsewhere than the template (cost {best[0]:.2f})", None
    c, perm, flips = best
    if any(flips) or perm != tuple(range(len(a))):
        ops = [f"reverse:{i}" for i, f in zip(perm, flips) if f]
        ops += [f"order:{','.join(map(str, perm))}"] if perm != tuple(range(len(a))) else []
        return "stroke order or direction differs", ops
    return None, None


def near(p, s):
    if len(s) < 2:
        return np.linalg.norm(s[0] - p)
    a, d = s[:-1], np.diff(s, axis=0)
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[:, None] * d - p, axis=1).min()


def length(s):
    return np.linalg.norm(np.diff(s, axis=0), axis=1).sum()


def duplicates(strokes, tol=0.06):
    on = lambda a, b: np.mean([near(p, b) for p in a]) < tol
    out = []
    for j, s in enumerate(strokes):
        if len(s) > 2 and any(i != j and i not in out and len(o) > 2 and on(s, o) and (length(s), -j) <= (length(o), -i) for i, o in enumerate(strokes)):
            out.append(j)
    return out


def split_points(strokes, tol=0.08):
    size, out = np.ptp(np.vstack(strokes), 0).max(), []
    for k, s in enumerate(strokes):
        ends = [e for j, o in enumerate(strokes) if j != k for e in (o[0], o[-1]) if near(e, s) < tol]
        idx = set(corners(s)) | {int(np.linalg.norm(s - e, axis=1).argmin()) for e in ends}
        idx |= {int(f(s[:, d])) for f in (np.argmin, np.argmax) for d in (0, 1)}
        out += [(k, i) for i in sorted(idx) if min(length(s[:i + 1]), length(s[i:])) > 0.15 * size]
    return out


def suggest(strokes, template, max_cost=0.3, penalty=0.04, tol=0.12):
    strokes = [np.asarray(s, float) for s in strokes]
    if compare(strokes, template, max_cost) == (None, None):
        return [], True
    drops = duplicates(strokes)
    base = [s for k, s in enumerate(strokes) if k not in drops]
    t = [np.asarray(s) for s in template]
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
                    merged = [np.vstack(g) for g in groups + [cur]]
                    cost = np.mean([stroke_cost(a, b) for a, b in zip(merged, t)])
                    edits = sum(flips) + (perm != tuple(range(n))) + len(joins) + (split is not None)
                    if cost <= max_cost and cost + penalty * edits < best[0]:
                        ops = [f"split:{split[0]},{split[1]}"] if split else []
                        ops += [f"reverse:{i}" for i, f in zip(perm, flips) if f]
                        ops += [f"order:{','.join(map(str, perm))}"] if perm != tuple(range(n)) else []
                        ops += [f"join:{b},{b + 1}" for b in sorted(joins, reverse=True)]
                        best = (cost + penalty * edits, ops)
    return [f"drop:{k}" for k in sorted(drops, reverse=True)] + best[1], np.isfinite(best[0])


def raw_glyphs(name):
    fixes = load(FIXES)
    fix = fixes.get(name, {"cursive": False, "glyphs": {}})
    prompts = {c: {"prompt": g["prompt"]} for c, g in fix.get("glyphs", {}).items() if "prompt" in g}
    return build(f"data/raw/fonts/{load(STORE)[name]['font']}", {**fixes, name: {**fix, "glyphs": prompts}})["glyphs"]


def check(entry, template):
    out = {}
    for c in CHARS:
        msg, ops = compare(entry["glyphs"][c]["strokes"], template["glyphs"][c])
        if msg:
            out[c] = (msg, ops)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("save", help="save a signed-off font's strokes as the stroke-order template")
    p.add_argument("name")
    p = sub.add_parser("check", help="compare fonts' glyphs with the template, character by character")
    p.add_argument("names", nargs="+")
    p = sub.add_parser("fix", help="write fix ops that match each glyph to the template; glyphs with manual ops are kept")
    p.add_argument("names", nargs="+")
    p.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    store = load(STORE)
    if args.command == "save":
        save(store[args.name])
        return print(f"{args.name} -> {TEMPLATE}")
    template = load(TEMPLATE)
    if args.command == "fix":
        for name in args.names:
            raw, fixes = raw_glyphs(name), load(FIXES)
            glyphs = fixes.setdefault(name, {"cursive": False, "glyphs": {}}).setdefault("glyphs", {})
            for c in CHARS:
                ops, matched = suggest(raw[c]["strokes"], template["glyphs"][c])
                if "ops" in glyphs.get(c, {}) or not ops:
                    continue
                print(f"{name} {c}: {ops}" + ("" if matched else "  (still differs from the template)"))
                if not args.dry:
                    g = glyphs.setdefault(c, {})
                    g["ops"], g["note"] = ops, (g.get("note", "") + "; auto: matched to the template").lstrip("; ")
            if not args.dry:
                json.dump(fixes, open(FIXES, "w"), indent=2)
        return
    for name in args.names:
        found = check(store[name], template)
        print(f"{name}: {len(found)} glyphs differ from {template['font']}")
        for c, (msg, ops) in found.items():
            print(f"  {c}  {msg}" + (f"   suggest {ops}" if ops else ""))


if __name__ == "__main__":
    main()
