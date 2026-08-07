"""Procedurally generate physics word problems with hand-styled stroke sketches.

    python physics_sketch.py --n 20000 --out data/physics_v1.jsonl
    python physics_sketch.py --preview          # per-archetype validation sheets

Each example is three linked parts generated from one parameter draw: a
natural-language caption of a physical setup, a pen-stroke sketch of that
setup, and a question whose answer follows from the physics. Ground truth is
exact by construction, so nothing unverifiable enters the geometry.

The generator is split in two, and the split is the design:

- **Specs** (`spec_*`) decide *what the scene is*: entities, their binned and
  exact parameters, the caption, question and answer. All correctness lives
  here. Parameters are quantized into named bins with enforced margins, so the
  caption decides the answer and no example is a near-tie.
- **Compilers** (`draw_*`) decide *how it is drawn*, and are deliberately
  non-deterministic: each object samples a depiction program (a weight may be
  a closed box, four separate strokes, a hatched crate, or a sack) and the
  whole scene is humanized with tremor, endpoint overshoot and stroke-order
  variation (see sketch_style.py). Two drawings of one spec should differ the
  way two people's napkin sketches differ.

Style noise is uncorrelated with the answer by construction -- a plank's
accidental tilt is random either way -- so it adds robustness pressure rather
than leaking labels. Sketches depict the setup, never the outcome: the see-saw
is drawn level even when a side must tip, because the drawing is the premise
and the answer is the reasoning target.

Structural lines are trimmed to their content (a plank ends just past its
outermost weight, the ground just past the structure), which keeps whitespace
and token count down. Only relative lengths carry meaning.

The "text" field trains text->sketch with train.py (--max_words 1). The
question, answer, and full spec ride along in "meta" for the reasoning stages
and for judge checklists.
"""

import argparse
import json
import os

import numpy as np

import sketch_style as S

GROUND_Y = 0.0
SPACING = 0.012

SIZES = {"small": (0.030, 0.040), "medium": (0.052, 0.062), "big": (0.078, 0.092)}
SIZE_RANK = {"small": 0, "medium": 1, "big": 2}
DISTS = {"near the middle": (0.08, 0.13), "halfway out": (0.18, 0.23),
         "at the far end": (0.28, 0.33)}
DIST_RANK = {k: i for i, k in enumerate(DISTS)}
HEIGHTS = {"small": (0.10, 0.13), "medium": (0.17, 0.21), "big": (0.26, 0.31)}
LENGTHS = {"small": (0.12, 0.15), "medium": (0.20, 0.24), "big": (0.30, 0.35)}


def _resample(path, spacing=SPACING):
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    if d[-1] < spacing:
        return path[[0, -1]]
    t = np.linspace(0.0, d[-1], max(2, int(round(d[-1] / spacing)) + 1))
    return np.column_stack([np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])])


def scene_to_points(strokes):
    """Strokes (y-up) -> (N, 3) y-down array with pen-lift marker rows."""
    rows = []
    for stroke in strokes:
        pts = _resample(stroke)
        for x, y in pts:
            rows.append([x, -y, 1.0])
        rows.append([pts[-1][0], -pts[-1][1], 0.0])
    return np.array(rows).round(4)


def _pick(rng, options, exclude=None):
    keys = [k for k in options if k != exclude]
    return keys[rng.integers(len(keys))]


def _weight(noun, rng, cx, cy, w, h):
    return S.WEIGHT_PROGRAMS[noun](cx, cy, w, h, rng)


def _noun(rng, stackable=False):
    options = S.BOX_NOUNS if stackable else list(S.WEIGHT_PROGRAMS)
    return options[rng.integers(len(options))]


def _plural(noun):
    return noun + ("es" if noun.endswith("x") else "s")


def _ball(rng, cx, cy, r):
    return S.BALL_PROGRAMS[rng.integers(len(S.BALL_PROGRAMS))](cx, cy, r, rng)


def _ground(rng, x0, x1):
    return S.GROUND_PROGRAMS[rng.integers(len(S.GROUND_PROGRAMS))](x0, x1, GROUND_Y, rng)


# ------------------------------------------------------------------- specs

def spec_lever(rng):
    sL, sR = _pick(rng, SIZES), _pick(rng, SIZES)
    dL, dR = _pick(rng, DISTS), _pick(rng, DISTS)
    tL = SIZE_RANK[sL] + DIST_RANK[dL]
    tR = SIZE_RANK[sR] + DIST_RANK[dR]
    if tL == tR:
        return None
    nL, nR = _noun(rng), _noun(rng)
    return {"kind": "lever", "nouns": (nL, nR),
            "size": (rng.uniform(*SIZES[sL]), rng.uniform(*SIZES[sR])),
            "dist": (rng.uniform(*DISTS[dL]), rng.uniform(*DISTS[dR])),
            "text": (f"a {sL} {nL} {dL} on the left and a {sR} {nR} {dR} "
                     f"on the right of a level seesaw"),
            "question": "which side tips down?",
            "answer": "left" if tL > tR else "right"}


def draw_lever(spec, rng):
    (s1, s2), (d1, d2) = spec["size"], spec["dist"]
    cx = 0.0
    plank_y = rng.uniform(0.09, 0.12)
    half = max(d1 + s1, d2 + s2) + 0.04
    strokes = [S.PIVOT_PROGRAMS[rng.integers(len(S.PIVOT_PROGRAMS))](
        cx, GROUND_Y, rng.uniform(0.07, 0.11), plank_y, rng),
        [np.array([(cx - half, plank_y), (cx + half, plank_y)])],
        _weight(spec["nouns"][0], rng, cx - d1 - s1 / 2, plank_y + s1 / 2, s1, s1),
        _weight(spec["nouns"][1], rng, cx + d2 + s2 / 2, plank_y + s2 / 2, s2, s2)]
    flat = [s for group in strokes for s in group]
    ground = _ground(rng, cx - half - 0.05, cx + half + 0.05)
    return ground + flat


def spec_ramp(rng):
    h1 = _pick(rng, SIZES)
    h2 = _pick(rng, SIZES, exclude=h1)
    return {"kind": "ramp",
            "height": (rng.uniform(*HEIGHTS[h1]), rng.uniform(*HEIGHTS[h2])),
            "text": (f"a ball at the top of a {h1} ramp and another ball "
                     f"at the top of a {h2} ramp"),
            "question": "which ball is moving faster at the bottom?",
            "answer": "first" if SIZE_RANK[h1] > SIZE_RANK[h2] else "second"}


def draw_ramp(spec, rng):
    strokes, x0 = [], 0.0
    for h in spec["height"]:
        w = h * rng.uniform(1.3, 1.6)
        strokes.append(np.array([(x0, GROUND_Y), (x0 + w, GROUND_Y),
                                 (x0, h), (x0, GROUND_Y)]))
        r = rng.uniform(0.020, 0.026)
        t = rng.uniform(0.08, 0.16)
        norm = np.hypot(w, h)
        strokes += _ball(rng, x0 + t * w + r * h / norm, (1 - t) * h + r * w / norm, r)
        x0 += w + rng.uniform(0.05, 0.09)
    return _ground(rng, -0.04, x0 - 0.02) + strokes


def spec_stack(rng):
    direction = "left" if rng.random() < 0.5 else "right"
    stable = rng.random() < 0.5
    s = rng.uniform(0.085, 0.10)
    step = rng.uniform(0.15, 0.30) if stable else rng.uniform(0.62, 0.75)
    noun = _noun(rng, stackable=True)
    return {"kind": "stack", "size": s, "step": step * s, "noun": noun,
            "direction": direction, "stable": stable,
            "text": (f"three {_plural(noun)} stacked in a tower, each shifted "
                     f"{'slightly' if stable else 'far'} to the {direction} "
                     f"of the one below"),
            "question": "does the tower stand or tip over?",
            "answer": "stands" if stable else f"tips {direction}"}


def draw_stack(spec, rng):
    s, sign = spec["size"], -1 if spec["direction"] == "left" else 1
    strokes = []
    for level in range(3):
        strokes += _weight(spec["noun"], rng, sign * spec["step"] * level,
                           GROUND_Y + s / 2 + s * level, s, s)
    xs = np.concatenate([st[:, 0] for st in strokes])
    return _ground(rng, xs.min() - 0.06, xs.max() + 0.06) + strokes


def spec_pendulum(rng):
    l1 = _pick(rng, SIZES)
    l2 = _pick(rng, SIZES, exclude=l1)
    return {"kind": "pendulum",
            "length": (rng.uniform(*LENGTHS[l1]), rng.uniform(*LENGTHS[l2])),
            "text": (f"two pendulums hang from a beam, the first on a {l1} "
                     f"string and the second on a {l2} string"),
            "question": "which pendulum swings back and forth faster?",
            "answer": "first" if SIZE_RANK[l1] < SIZE_RANK[l2] else "second"}


def draw_pendulum(spec, rng):
    top = max(spec["length"]) + rng.uniform(0.06, 0.10)
    gap = rng.uniform(0.24, 0.34)
    strokes = []
    for i, length in enumerate(spec["length"]):
        x = i * gap
        r = rng.uniform(0.020, 0.027)
        strokes.append(np.array([(x, top), (x, top - length)]))
        strokes += _ball(rng, x, top - length - r, r)
    beam = [np.array([(-0.07, top), (gap + 0.07, top)])]
    return beam + strokes


def spec_scale(rng):
    sL = _pick(rng, SIZES)
    sR = _pick(rng, SIZES, exclude=sL)
    nL, nR = _noun(rng), _noun(rng)
    return {"kind": "scale", "nouns": (nL, nR),
            "size": (rng.uniform(*SIZES[sL]), rng.uniform(*SIZES[sR])),
            "text": (f"a balance scale holding a {sL} {nL} on the left pan "
                     f"and a {sR} {nR} on the right pan"),
            "question": "which pan goes down?",
            "answer": "left" if SIZE_RANK[sL] > SIZE_RANK[sR] else "right"}


def draw_scale(spec, rng):
    s1, s2 = spec["size"]
    pan_w = max(s1, s2) / 2 + rng.uniform(0.015, 0.03)
    half = 2.4 * pan_w
    drop = rng.uniform(0.075, 0.10)
    beam_y = GROUND_Y + drop + max(s1, s2) + rng.uniform(0.05, 0.09)
    strokes = [np.array([(0.0, GROUND_Y), (0.0, beam_y)]),
               np.array([(-half, beam_y), (half, beam_y)])]
    for x, s in ((-half, s1), (half, s2)):
        pan_y = beam_y - drop
        strokes.append(np.array([(x, beam_y), (x, pan_y)]))
        strokes.append(np.array([(x - pan_w, pan_y), (x + pan_w, pan_y)]))
        strokes += _weight(spec["nouns"][0 if x < 0 else 1], rng,
                           x, pan_y + s / 2, s, s)
    return _ground(rng, -half - 0.06, half + 0.06) + strokes


def spec_drop(rng):
    h1 = _pick(rng, SIZES)
    h2 = _pick(rng, SIZES, exclude=h1)
    return {"kind": "drop",
            "height": (rng.uniform(*HEIGHTS[h1]), rng.uniform(*HEIGHTS[h2])),
            "text": (f"one ball rests on a {h1}-height shelf and another "
                     f"ball on a {h2}-height shelf, both about to roll off"),
            "question": "which ball hits the ground first?",
            "answer": "first" if SIZE_RANK[h1] < SIZE_RANK[h2] else "second"}


def draw_drop(spec, rng):
    strokes, x = [], 0.0
    for h in spec["height"]:
        shelf_w = rng.uniform(0.11, 0.15)
        r = rng.uniform(0.020, 0.026)
        leg_x = x + shelf_w * rng.uniform(0.75, 0.9)
        strokes.append(np.array([(x, h), (x + shelf_w, h)]))
        strokes.append(np.array([(leg_x, h), (leg_x, GROUND_Y)]))
        strokes += _ball(rng, x + rng.uniform(0.025, 0.05), h + r, r)
        x += shelf_w + rng.uniform(0.10, 0.16)
    return _ground(rng, -0.04, x - 0.04) + strokes


SPECS = {"lever": spec_lever, "ramp": spec_ramp, "stack": spec_stack,
         "pendulum": spec_pendulum, "scale": spec_scale, "drop": spec_drop}
DRAWERS = {"lever": draw_lever, "ramp": draw_ramp, "stack": draw_stack,
           "pendulum": draw_pendulum, "scale": draw_scale, "drop": draw_drop}


def make_example(kind, rng):
    spec = SPECS[kind](rng)
    if spec is None:
        return None
    strokes = DRAWERS[kind](spec, rng)
    strokes = S.humanize(strokes, rng)
    meta = {k: spec[k] for k in ("kind", "question", "answer")}
    return {"text": spec["text"],
            "points": scene_to_points(strokes).tolist(),
            "meta": meta}


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    names = list(SPECS)
    out = []
    while len(out) < n:
        made = make_example(names[len(out) % len(names)], rng)
        if made is not None:
            out.append(made)
    return out


def main():
    p = argparse.ArgumentParser(description="Generate the physics sketch dataset")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--out", type=str, default="data/physics_v1.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true",
                   help="render per-archetype validation sheets")
    args = p.parse_args()

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pengpt.sampling import draw
        rng = np.random.default_rng(args.seed)
        fig, axes = plt.subplots(len(SPECS), 8, figsize=(22, 2.1 * len(SPECS)))
        for row, kind in enumerate(SPECS):
            for col in range(8):
                ex = None
                while ex is None:
                    ex = make_example(kind, rng)
                ax = axes[row][col]
                draw(ax, np.array(ex["points"]), color="k", linewidth=1.0)
                m = ex["meta"]
                ax.set_title(f'{m["answer"]}', fontsize=7)
            axes[row][0].set_ylabel(kind, fontsize=10, rotation=0,
                                    ha="right", va="center", labelpad=8)
        fig.suptitle("physics_v1: eight draws per archetype (title = answer)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig("physics_preview.png", dpi=110, bbox_inches="tight")
        print("wrote physics_preview.png")
        return

    examples = generate(args.n, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    from collections import Counter
    kinds = Counter(ex["meta"]["kind"] for ex in examples)
    print(f"wrote {len(examples)} examples to {args.out}")
    print(f"archetypes: {dict(kinds)}")
    print(f"longest caption: {max(len(ex['text']) for ex in examples)} chars")


if __name__ == "__main__":
    main()
