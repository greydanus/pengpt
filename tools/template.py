import argparse
import itertools
import json

import numpy as np

from tools.glyphs import CHARS, STORE, load

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
    lo, size = p.min(0), np.maximum(np.ptp(p, 0), max(np.ptp(p, 0).max() * 0.3, 1e-6))
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
    return (np.linalg.norm(a0 - t0) + np.linalg.norm(a1 - t1)) / 2 + 0.5 * (aw * tw < 0)


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
    args = ap.parse_args()
    store = load(STORE)
    if args.command == "save":
        save(store[args.name])
        return print(f"{args.name} -> {TEMPLATE}")
    template = load(TEMPLATE)
    for name in args.names:
        found = check(store[name], template)
        print(f"{name}: {len(found)} glyphs differ from {template['font']}")
        for c, (msg, ops) in found.items():
            print(f"  {c}  {msg}" + (f"   suggest {ops}" if ops else ""))


if __name__ == "__main__":
    main()
