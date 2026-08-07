"""Procedurally generate physics word problems with verifiable stroke sketches.

    python physics_sketch.py --n 20000 --out data/physics_v0.jsonl
    python physics_sketch.py --preview          # render a validation sheet

Each example is a real-world relational reasoning problem in three linked
parts: a natural-language caption of a physical setup, a pen-stroke sketch of
that setup, and a question whose answer follows from the physics. The three
are generated together from one parameter draw, so every example carries its
own ground truth and nothing unverifiable enters the geometry.

Design decisions, each load-bearing:

- **Parameters live in named bins.** "a big box far left" is the caption; the
  sketch draws the exact sampled size and position. Answers are decided by the
  bins with an enforced margin, so the caption alone determines the answer and
  no example is a near-tie a model could only guess.
- **The sketch depicts the setup, never the outcome.** A see-saw is drawn
  level even when a side must tip: the drawing is the premise and the answer
  is the reasoning target. Drawing the tilt would leak the label.
- **A ground line anchors every scene**, drawn first, then structures left to
  right. Deterministic stroke order is a convention the model can learn;
  load_examples shifts x so the first point lands at 0, which makes the
  ground line's left end a stable origin across the corpus.
- **Answers are one of two named entities** (left/right side, first/second
  ball), never free text, so downstream QA is exactly checkable.

The "text" field holds the caption for text->sketch training with train.py
(--max_words 1 --augment none). The question, answer, archetype and raw
parameters ride along in "meta" for the reasoning stages, which train a model
to answer from the caption plus its own sketch.
"""

import argparse
import json
import os

import numpy as np

# Scenes live in y-up coordinates with the ground at y = 0 and are flipped to
# the repo's y-down convention on export. Heights stay near the bundled ink
# scale (~0.22-0.44 tall) so --grid 0.020 prices a scene like a couple of words.
GROUND_Y = 0.0
SPACING = 0.012


def _resample(path, spacing=SPACING):
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    if d[-1] < spacing:
        return path[[0, -1]]
    t = np.linspace(0.0, d[-1], max(2, int(round(d[-1] / spacing)) + 1))
    return np.column_stack([np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])])


def seg(*xy_pairs):
    return np.asarray(xy_pairs, dtype=float)


def rect(cx, cy, w, h):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return seg((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))


def circle(cx, cy, r, n=28):
    a = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def triangle(cx, base_y, base_w, height):
    x0, x1 = cx - base_w / 2, cx + base_w / 2
    return seg((x0, base_y), (x1, base_y), (cx, base_y + height), (x0, base_y))


def scene_to_points(strokes):
    """Strokes (y-up) -> (N, 3) array in the repo's y-down convention.

    Each stroke's rows carry pen = 1 and are followed by one pen = 0 row
    duplicating the last point, the same lift-marker form collect.html writes.
    """
    rows = []
    for stroke in strokes:
        pts = _resample(stroke)
        for x, y in pts:
            rows.append([x, -y, 1.0])
        rows.append([pts[-1][0], -pts[-1][1], 0.0])
    return np.array(rows).round(4)


def ground(width=0.9):
    return seg((0.0, GROUND_Y), (width, GROUND_Y))


# Size and distance bins: name -> sampling range. Margins between bins are what
# make bin-level comparisons unambiguous.
SIZES = {"small": (0.030, 0.040), "medium": (0.052, 0.062), "big": (0.078, 0.092)}
SIZE_RANK = {"small": 0, "medium": 1, "big": 2}
DISTS = {"near the middle": (0.08, 0.13), "halfway out": (0.18, 0.23),
         "at the far end": (0.28, 0.33)}
DIST_RANK = {k: i for i, k in enumerate(DISTS)}


def _pick(rng, options, exclude=None):
    keys = [k for k in options if k != exclude]
    return keys[rng.integers(len(keys))]


def lever(rng):
    """Torque: which side of a level see-saw tips down."""
    sL, sR = _pick(rng, SIZES), _pick(rng, SIZES)
    dL, dR = _pick(rng, DISTS), _pick(rng, DISTS)
    # torque rank: size rank + distance rank, with ties re-rolled by the caller
    tL = SIZE_RANK[sL] + DIST_RANK[dL]
    tR = SIZE_RANK[sR] + DIST_RANK[dR]
    if tL == tR:
        return None
    cx, plank_y, half = 0.45, 0.10, 0.36
    strokes = [ground(), triangle(cx, GROUND_Y, 0.10, plank_y),
               seg((cx - half, plank_y), (cx + half, plank_y))]
    for side, size_name, dist_name in (("L", sL, dL), ("R", sR, dR)):
        s = rng.uniform(*SIZES[size_name])
        d = rng.uniform(*DISTS[dist_name])
        x = cx - d - s / 2 if side == "L" else cx + d + s / 2
        strokes.append(rect(x, plank_y + s / 2, s, s))
    answer = "left" if tL > tR else "right"
    text = (f"a {sL} box {dL} on the left and a {sR} box {dR} "
            f"on the right of a level seesaw")
    return strokes, text, "which side tips down?", answer


def ramp(rng):
    """Energy: the ball from the taller ramp is faster at the bottom."""
    h1_name, h2_name = _pick(rng, SIZES), None
    h2_name = _pick(rng, SIZES, exclude=h1_name)
    heights = {"small": (0.10, 0.13), "medium": (0.17, 0.20), "big": (0.25, 0.29)}
    strokes = [ground()]
    tops = []
    for i, (x0, name) in enumerate(((0.05, h1_name), (0.50, h2_name))):
        h = rng.uniform(*heights[name])
        w = h * rng.uniform(1.3, 1.6)
        strokes.append(seg((x0, GROUND_Y), (x0 + w, GROUND_Y), (x0, h), (x0, GROUND_Y)))
        # the ball rests on the slope near the apex: slope runs (x0,h)->(x0+w,0),
        # its upslope normal is (h,w)/|(h,w)|, and the center sits one radius out
        r = 0.022
        t = 0.12
        norm = np.hypot(w, h)
        strokes.append(circle(x0 + t * w + r * h / norm, (1 - t) * h + r * w / norm, r))
        tops.append(name)
    answer = "first" if SIZE_RANK[tops[0]] > SIZE_RANK[tops[1]] else "second"
    text = (f"a ball at the top of a {tops[0]} ramp and another ball "
            f"at the top of a {tops[1]} ramp")
    return strokes, text, "which ball is moving faster at the bottom?", answer


def stack(rng):
    """Support: does the offset tower stand or tip, and which way."""
    s = rng.uniform(0.085, 0.10)
    direction = "left" if rng.random() < 0.5 else "right"
    stable = rng.random() < 0.5
    # stable: small same-direction offsets; unstable: top box far past the edge
    step = rng.uniform(0.15, 0.30) * s if stable else rng.uniform(0.62, 0.75) * s
    sign = -1 if direction == "left" else 1
    cx = 0.45
    strokes = [ground()]
    for level in range(3):
        strokes.append(rect(cx + sign * step * level, GROUND_Y + s / 2 + s * level, s, s))
    offset_name = "slightly" if stable else "far"
    answer = "stands" if stable else f"tips {direction}"
    text = (f"three boxes stacked in a tower, each shifted {offset_name} "
            f"to the {direction} of the one below")
    return strokes, text, "does the tower stand or tip over?", answer


def pendulum(rng):
    """Period: the shorter pendulum swings faster."""
    l1, l2 = _pick(rng, SIZES), None
    l2 = _pick(rng, SIZES, exclude=l1)
    lengths = {"small": (0.12, 0.15), "medium": (0.20, 0.24), "big": (0.30, 0.35)}
    top = 0.44
    strokes = [seg((0.08, top), (0.82, top))]
    for x, name in ((0.28, l1), (0.62, l2)):
        length = rng.uniform(*lengths[name])
        r = 0.024
        strokes.append(seg((x, top), (x, top - length)))
        strokes.append(circle(x, top - length - r, r))
    answer = "first" if SIZE_RANK[l1] < SIZE_RANK[l2] else "second"
    text = (f"two pendulums hang from a beam, the first on a {l1} string "
            f"and the second on a {l2} string")
    return strokes, text, "which pendulum swings back and forth faster?", answer


def scale(rng):
    """Mass: the heavier pan of a level balance goes down."""
    sL = _pick(rng, SIZES)
    sR = _pick(rng, SIZES, exclude=sL)
    cx, beam_y, half = 0.45, 0.34, 0.26
    strokes = [ground(), seg((cx, GROUND_Y), (cx, beam_y)),
               seg((cx - half, beam_y), (cx + half, beam_y))]
    for x, name in ((cx - half, sL), (cx + half, sR)):
        pan_y = beam_y - 0.10
        strokes.append(seg((x, beam_y), (x, pan_y)))
        strokes.append(seg((x - 0.06, pan_y), (x + 0.06, pan_y)))
        s = rng.uniform(*SIZES[name])
        strokes.append(rect(x, pan_y + s / 2, s, s))
    answer = "left" if SIZE_RANK[sL] > SIZE_RANK[sR] else "right"
    text = (f"a balance scale holding a {sL} box on the left pan "
            f"and a {sR} box on the right pan")
    return strokes, text, "which pan goes down?", answer


def drop(rng):
    """Fall time: the ball on the lower shelf lands first."""
    h1 = _pick(rng, SIZES)
    h2 = _pick(rng, SIZES, exclude=h1)
    heights = {"small": (0.10, 0.13), "medium": (0.18, 0.22), "big": (0.28, 0.33)}
    strokes = [ground()]
    for x, name in ((0.18, h1), (0.62, h2)):
        h = rng.uniform(*heights[name])
        r = 0.022
        strokes.append(seg((x, h), (x + 0.14, h), (x + 0.14, h - 0.03)))  # shelf bracket
        strokes.append(circle(x + 0.04, h + r + 0.004, r))
    answer = "first" if SIZE_RANK[h1] < SIZE_RANK[h2] else "second"
    text = (f"one ball rests on a {h1}-height shelf and another ball "
            f"on a {h2}-height shelf, both about to roll off")
    return strokes, text, "which ball hits the ground first?", answer


ARCHETYPES = {"lever": lever, "ramp": ramp, "stack": stack,
              "pendulum": pendulum, "scale": scale, "drop": drop}


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    out = []
    while len(out) < n:
        name = names[len(out) % len(names)]
        made = ARCHETYPES[name](rng)
        if made is None:            # bin-level tie: reroll
            continue
        strokes, text, question, answer = made
        out.append({"text": text,
                    "points": scene_to_points(strokes).tolist(),
                    "meta": {"archetype": name, "question": question,
                             "answer": answer}})
    return out


def main():
    p = argparse.ArgumentParser(description="Generate the physics sketch dataset")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--out", type=str, default="data/physics_v0.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true",
                   help="render a validation sheet instead of writing a dataset")
    args = p.parse_args()

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pengpt.sampling import draw
        examples = generate(18, seed=args.seed)
        fig, axes = plt.subplots(3, 6, figsize=(22, 9))
        for ax, ex in zip(axes.ravel(), examples):
            draw(ax, np.array(ex["points"]), color="k", linewidth=1.0)
            m = ex["meta"]
            ax.set_title(f'{ex["text"]}\n{m["question"]} -> {m["answer"]}',
                         fontsize=7, wrap=True)
        fig.tight_layout()
        fig.savefig("physics_preview.png", dpi=110, bbox_inches="tight")
        print("wrote physics_preview.png")
        return

    examples = generate(args.n, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    from collections import Counter
    kinds = Counter(ex["meta"]["archetype"] for ex in examples)
    longest = max(len(ex["text"]) for ex in examples)
    print(f"wrote {len(examples)} examples to {args.out}")
    print(f"archetypes: {dict(kinds)}")
    print(f"longest caption: {longest} chars")


if __name__ == "__main__":
    main()
