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

# Sketch detail level. "plain" keeps every element that carries information
# about the answer and drops the rest: ground hatching, crate hatching, and
# extra depiction variety are decoration, and they are expensive. Measured at
# the default grid, hatched ground costs 111 tokens against 38 for a plain
# line, and a hatched crate 37 against 20 for a box -- so decoration was most
# of a short scene.
#
# This matters beyond throughput. In a depth sweep the sketch is supposed to
# hold the reasoning state and nothing else; if deeper problems also carry more
# decorative ink, sketch length is confounded with reasoning depth and a
# scaling curve cannot separate them. "plain" keeps the per-stage cost as close
# to constant as the geometry allows.
STYLE = "rich"


def _plain():
    return STYLE == "plain"


# How much of the setup the caption states.
#
# "full" states every attribute, so the caption alone determines the answer and
# the sketch is redundant. That is the wrong shape for measuring whether a
# scratchpad helps: if text suffices, a picture can only ever break even.
#
# "elided" removes one attribute from the caption and leaves it in the drawing
# alone. The example is then easy with the sketch and genuinely underdetermined
# without it -- which is the regime where a sketchpad can pay off. Whichever
# attribute is dropped is recorded in meta["elided"], so the two regimes can be
# compared on matched specs.
DETAIL = "full"


def _elide(rng):
    """True when this example should hide one attribute from the caption."""
    return DETAIL == "elided"

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
    """Draw a named weight.

    Plain mode draws the two expensive depictions with cheaper outlines -- a
    crate as a plain box (37 tokens against 20), a four-stroke block as one
    closed box -- while the caption keeps saying "crate" or "block". The noun
    is what the model must attend to; its hatching is not.
    """
    if _plain() and noun in ("crate", "block"):
        return S.box_closed(cx, cy, w, h, rng)
    return S.WEIGHT_PROGRAMS[noun](cx, cy, w, h, rng)


def _noun(rng, stackable=False):
    """Pick the noun for a weight.

    The noun appears in the caption, so the full vocabulary is always offered:
    restricting it in plain mode would cut caption diversity, which is the
    scarce resource here. Plain mode economizes on how a noun is *drawn*
    instead -- see _weight -- which costs tokens without costing captions.
    """
    options = S.BOX_NOUNS if stackable else list(S.WEIGHT_PROGRAMS)
    return options[rng.integers(len(options))]


def _plural(noun):
    return noun + ("es" if noun.endswith("x") else "s")


def _ball(rng, cx, cy, r):
    return S.BALL_PROGRAMS[rng.integers(len(S.BALL_PROGRAMS))](cx, cy, r, rng)


def _ground(rng, x0, x1):
    """The ground line. Hatching is decoration and costs ~3x a plain line."""
    if _plain():
        return S.ground_line(x0, x1, GROUND_Y, rng)
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
    # Distances are drawn to scale, so eliding them leaves the caption
    # underdetermined while the sketch still answers the question.
    if _elide(rng):
        text = (f"a {sL} {nL} on the left and a {sR} {nR} on the right of a "
                f"level seesaw")
        elided = "dist"
    else:
        text = (f"a {sL} {nL} {dL} on the left and a {sR} {nR} {dR} "
                f"on the right of a level seesaw")
        elided = None
    return {"kind": "lever", "nouns": (nL, nR), "elided": elided,
            "size": (rng.uniform(*SIZES[sL]), rng.uniform(*SIZES[sR])),
            "dist": (rng.uniform(*DISTS[dL]), rng.uniform(*DISTS[dR])),
            "text": text,
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


SURFACES = {"smooth": 0, "slightly rough": 1, "very rough": 2}


def spec_ramp(rng):
    """Speed at the bottom: height helps, friction hurts.

    Height alone is one binned attribute, which is why this archetype could
    only ever produce six captions. Surface roughness is a genuinely
    independent second factor -- on a frictionless ramp the final speed depends
    on height alone, and roughness removes energy -- so the two compose into an
    additive score the same way the lever's size and distance do.

    The score is ordinal, not a real energy calculation: bins are spaced widely
    enough (see HEIGHTS and the margin rejection below) that a taller-but-
    rougher ramp is genuinely ambiguous rather than secretly decidable, so
    those draws are rejected rather than guessed at.
    """
    h1, h2 = _pick(rng, SIZES), _pick(rng, SIZES)
    f1, f2 = _pick(rng, SURFACES), _pick(rng, SURFACES)
    t1 = SIZE_RANK[h1] - SURFACES[f1]
    t2 = SIZE_RANK[h2] - SURFACES[f2]
    if t1 == t2:
        return None
    return {"kind": "ramp",
            "height": (rng.uniform(*HEIGHTS[h1]), rng.uniform(*HEIGHTS[h2])),
            "rough": (SURFACES[f1], SURFACES[f2]),
            "text": (f"a ball at the top of a {h1} {f1} ramp and another ball "
                     f"at the top of a {h2} {f2} ramp"),
            "question": "which ball is moving faster at the bottom?",
            "answer": "first" if t1 > t2 else "second"}


def _ramp_hatch(x0, w, h, level, rng):
    """Tick marks along the slope: none when smooth, denser when rougher.

    Unlike ground hatching this is not decoration -- roughness decides the
    answer, so the ticks are the only thing in the drawing that carries it.
    Plain mode keeps them and merely uses fewer, since the count only has to be
    distinguishable between the three levels, not realistic.
    """
    if level <= 0:
        return []
    out, n = [], (2 * level if _plain() else 3 * level + 1)
    for t in np.linspace(0.12, 0.88, n):
        x, y = x0 + t * w, (1 - t) * h
        size = rng.uniform(0.010, 0.016)
        out.append(np.array([(x, y), (x - size * 0.5, y + size)], float))
    return out


def draw_ramp(spec, rng):
    strokes, x0 = [], 0.0
    for h, rough in zip(spec["height"], spec["rough"]):
        w = h * rng.uniform(1.3, 1.6)
        strokes.append(np.array([(x0, GROUND_Y), (x0 + w, GROUND_Y),
                                 (x0, h), (x0, GROUND_Y)]))
        strokes += _ramp_hatch(x0, w, h, rough, rng)
        r = rng.uniform(0.020, 0.026)
        t = rng.uniform(0.08, 0.16)
        norm = np.hypot(w, h)
        strokes += _ball(rng, x0 + t * w + r * h / norm, (1 - t) * h + r * w / norm, r)
        x0 += w + rng.uniform(0.05, 0.09)
    return _ground(rng, -0.04, x0 - 0.02) + strokes


COUNTS = {"three": 3, "four": 4, "five": 5}
SHIFTS = {"slightly": 0.22, "noticeably": 0.45, "far": 0.68}
# Wider than one block, so the base reads as a base in the drawing and can
# actually support an overhang rather than being a tick under the tower.
BASES = {"narrow": 1.4, "wide": 2.6}


def spec_stack(rng):
    """A tower tips when its top block's centre passes the base's edge.

    Three attributes compose: how many blocks, how far each is shifted, and how
    wide the base is. The rule is real rather than looked up -- total offset is
    (n - 1) * shift * size, and the tower stands while that stays inside half
    the base width -- so taller towers and narrower bases tip at shifts that a
    short wide one survives. A single stated attribute is never enough.

    Draws landing within a margin of the tipping point are rejected, so no
    example is a near-tie the caption cannot resolve.
    """
    direction = "left" if rng.random() < 0.5 else "right"
    count = _pick(rng, COUNTS)
    shift = _pick(rng, SHIFTS)
    base = _pick(rng, BASES)
    base_w = BASES[base]

    # Everything is in units of one block's width. The top block's centre sits
    # offset from the bottom block's centre; the tower stands while that centre
    # stays over the base, whose half-width is base_w / 2.
    n = COUNTS[count]
    offset = (n - 1) * SHIFTS[shift]
    margin = abs(offset - base_w / 2)
    if margin < 0.15:                          # too close to call
        return None
    stable = offset < base_w / 2
    s = rng.uniform(0.070, 0.085)
    noun = _noun(rng, stackable=True)
    # The shift is the attribute a drawing shows best -- an overhang is
    # obvious in a picture and fiddly in words -- so that is what is elided.
    if _elide(rng):
        text = (f"{count} {_plural(noun)} stacked on a {base} base, each "
                f"shifted to the {direction} of the one below")
        elided = "shift"
    else:
        text = (f"{count} {_plural(noun)} stacked on a {base} base, each "
                f"shifted {shift} to the {direction} of the one below")
        elided = None
    return {"kind": "stack", "size": s, "step": SHIFTS[shift] * s,
            "count": n, "base_w": base_w, "noun": noun, "elided": elided,
            "direction": direction, "stable": stable,
            "text": text,
            "question": "does the tower stand or tip over?",
            "answer": "stands" if stable else f"tips {direction}"}


def draw_stack(spec, rng):
    """The base is a shallow slab the tower sits on, drawn wide enough to see.

    Blocks start on top of it rather than on the ground, so the overhang the
    caption describes is the overhang in the picture.
    """
    s, sign = spec["size"], -1 if spec["direction"] == "left" else 1
    half = spec["base_w"] * s / 2
    slab_h = s * rng.uniform(0.16, 0.24)
    strokes = [np.array([(-half, GROUND_Y), (half, GROUND_Y),
                         (half, GROUND_Y + slab_h), (-half, GROUND_Y + slab_h),
                         (-half, GROUND_Y)], float)]
    for level in range(spec["count"]):
        strokes += _weight(spec["noun"], rng, sign * spec["step"] * level,
                           GROUND_Y + slab_h + s / 2 + s * level, s, s)
    xs = np.concatenate([st[:, 0] for st in strokes])
    return _ground(rng, xs.min() - 0.06, xs.max() + 0.06) + strokes


BOB_SIZES = {"light": (0.020, 0.025), "heavy": (0.034, 0.042)}
ANGLES = {"a little": 12.0, "a lot": 26.0}


def spec_pendulum(rng):
    """Period depends on length alone -- bob mass is a genuine distractor.

    Unlike the lever's size and distance, the two attributes here are not both
    causal: a heavy bob swings at exactly the same rate as a light one on the
    same string. Mass is varied in the caption and drawn at visibly different
    sizes anyway, so the only way to answer is to know that it does not matter.
    That makes it the one archetype where extra attributes test physics rather
    than composition, which is worth having in the mix.

    Equal lengths are rejected because then the answer really is undecidable.
    """
    l1, l2 = _pick(rng, SIZES), _pick(rng, SIZES)
    if l1 == l2:
        return None
    m1, m2 = _pick(rng, BOB_SIZES), _pick(rng, BOB_SIZES)
    a1, a2 = _pick(rng, ANGLES), _pick(rng, ANGLES)
    return {"kind": "pendulum",
            "length": (rng.uniform(*LENGTHS[l1]), rng.uniform(*LENGTHS[l2])),
            "bob": (rng.uniform(*BOB_SIZES[m1]), rng.uniform(*BOB_SIZES[m2])),
            "angle": (ANGLES[a1] * rng.uniform(0.85, 1.15),
                      ANGLES[a2] * rng.uniform(0.85, 1.15)),
            "text": (f"two pendulums hang from a beam, the first a {m1} bob on "
                     f"a {l1} string pulled back {a1} and the second a {m2} bob "
                     f"on a {l2} string pulled back {a2}"),
            "question": "which pendulum swings back and forth faster?",
            "answer": "first" if SIZE_RANK[l1] < SIZE_RANK[l2] else "second"}


def draw_pendulum(spec, rng):
    top = max(spec["length"]) + rng.uniform(0.06, 0.10)
    gap = rng.uniform(0.26, 0.36)
    strokes = []
    for i, (length, r, angle) in enumerate(
            zip(spec["length"], spec["bob"], spec["angle"])):
        x = i * gap
        # Drawn at its release angle: the swing amplitude is visible, and for
        # small swings it still does not affect the period.
        a = np.deg2rad(angle) * (1 if i else -1)
        bx, by = x + length * np.sin(a), top - length * np.cos(a)
        strokes.append(np.array([(x, top), (bx, by)]))
        strokes += _ball(rng, bx, by - r, r)
    beam = [np.array([(-0.10, top), (gap + 0.10, top)])]
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


SPEEDS = {"slowly": 0, "quickly": 1}
BALL_WEIGHTS = {"light": (0.019, 0.024), "heavy": (0.032, 0.039)}


def spec_drop(rng):
    """Fall time depends on height alone -- speed and weight are distractors.

    Two independent red herrings: a ball rolling off faster travels further
    horizontally but falls for the same time, and a heavier ball falls at the
    same rate as a light one. Both are stated in the caption and both are drawn
    (speed as a motion arrow, weight as bob size), so the archetype rewards
    knowing which stated attribute is causal rather than reading off the one
    number that varies.
    """
    h1, h2 = _pick(rng, SIZES), _pick(rng, SIZES)
    if h1 == h2:
        return None
    v1, v2 = _pick(rng, SPEEDS), _pick(rng, SPEEDS)
    w1, w2 = _pick(rng, BALL_WEIGHTS), _pick(rng, BALL_WEIGHTS)
    return {"kind": "drop",
            "height": (rng.uniform(*HEIGHTS[h1]), rng.uniform(*HEIGHTS[h2])),
            "speed": (SPEEDS[v1], SPEEDS[v2]),
            "radius": (rng.uniform(*BALL_WEIGHTS[w1]), rng.uniform(*BALL_WEIGHTS[w2])),
            "text": (f"a {w1} ball rolls {v1} off a {h1}-height shelf and a "
                     f"{w2} ball rolls {v2} off a {h2}-height shelf"),
            "question": "which ball hits the ground first?",
            "answer": "first" if SIZE_RANK[h1] < SIZE_RANK[h2] else "second"}


def _motion_arrow(x, y, fast, rng):
    """A short arrow showing which way the ball rolls; longer when quicker."""
    length = rng.uniform(0.030, 0.040) * (1.8 if fast else 1.0)
    head = length * 0.30
    return [np.array([(x, y), (x + length, y)], float),
            np.array([(x + length - head, y + head * 0.6), (x + length, y),
                      (x + length - head, y - head * 0.6)], float)]


def draw_drop(spec, rng):
    strokes, x = [], 0.0
    for h, fast, r in zip(spec["height"], spec["speed"], spec["radius"]):
        shelf_w = rng.uniform(0.11, 0.15)
        leg_x = x + shelf_w * rng.uniform(0.75, 0.9)
        strokes.append(np.array([(x, h), (x + shelf_w, h)]))
        strokes.append(np.array([(leg_x, h), (leg_x, GROUND_Y)]))
        ball_x = x + rng.uniform(0.025, 0.05)
        strokes += _ball(rng, ball_x, h + r, r)
        strokes += _motion_arrow(ball_x + r + 0.012, h + r, fast, rng)
        x += shelf_w + rng.uniform(0.10, 0.16)
    return _ground(rng, -0.04, x - 0.04) + strokes


# ------------------------------------------- archetypes beyond weights on planks
#
# The six above are all statics and gravity. These add domains with different
# governing rules, so a model cannot cover the corpus with one heuristic about
# bigger-is-heavier. Each still follows the same contract: bin the attributes,
# score them, reject near-ties, and let the caption decide the answer.

DENSITIES = {"cork": 0, "wood": 1, "plastic": 2, "steel": 4}
FLUIDS = {"water": 2, "oil": 1, "syrup": 3}


def spec_float(rng):
    """Does the block float or sink? Density against the fluid, not size.

    Size is varied and drawn but never decides the outcome -- a big cork floats
    and a small steel nut sinks -- so this archetype directly contradicts the
    bigger-is-heavier heuristic the lever and scale archetypes reward.
    """
    material = _pick(rng, DENSITIES)
    fluid = _pick(rng, FLUIDS)
    if DENSITIES[material] == FLUIDS[fluid]:
        return None
    floats = DENSITIES[material] < FLUIDS[fluid]
    # The material x fluid grid holds 6 floating pairs against 4 sinking ones,
    # so accepting every draw teaches a 60/40 prior. Reject the majority class
    # at 1 - 4/6 rather than reweighting, which would distort which
    # material-fluid combinations appear.
    if floats and rng.random() < 1 / 3:
        return None
    size = _pick(rng, SIZES)
    shape = _pick(rng, {"block": 0, "ball": 1, "bar": 2})
    return {"kind": "float", "material": material, "fluid": fluid,
            "size": rng.uniform(*SIZES[size]), "floats": floats, "shape": shape,
            "text": (f"a {size} {material} {shape} is lowered into a tank "
                     f"of {fluid}"),
            "question": f"does the {shape} float or sink?",
            "answer": "floats" if floats else "sinks"}


def draw_float(spec, rng):
    """A vessel with a waterline and the block drawn at its resting place."""
    w, h = 0.20, 0.17
    strokes = S.VESSEL_PROGRAMS[rng.integers(len(S.VESSEL_PROGRAMS))](
        0.0, GROUND_Y, w, h, rng)
    level = GROUND_Y + h * rng.uniform(0.62, 0.74)
    strokes += S.fill_level(0.0, GROUND_Y, w, level, rng)
    s = spec["size"]
    # Floating sits astride the waterline; sinking rests on the floor.
    cy = level if spec["floats"] else GROUND_Y + s / 2
    cx = rng.uniform(-0.03, 0.03)
    if spec["shape"] == "ball":
        strokes += _ball(rng, cx, cy, s / 2)
    elif spec["shape"] == "bar":
        strokes += S.box_closed(cx, cy, s * 1.7, s * 0.5, rng)
    else:
        strokes += S.box_closed(cx, cy, s, s, rng)
    return _ground(rng, -w / 2 - 0.05, w / 2 + 0.05) + strokes


STIFFNESS = {"soft": 1, "medium": 2, "stiff": 3}


def spec_spring(rng):
    """Which spring is compressed further? Load over stiffness.

    Two independent attributes, and the comparison is a ratio rather than a
    sum, so the additive-score habit from the lever does not transfer.
    """
    k1, k2 = _pick(rng, STIFFNESS), _pick(rng, STIFFNESS)
    w1, w2 = _pick(rng, SIZES), _pick(rng, SIZES)
    c1 = (SIZE_RANK[w1] + 1) / STIFFNESS[k1]
    c2 = (SIZE_RANK[w2] + 1) / STIFFNESS[k2]
    if abs(c1 - c2) < 0.25:
        return None
    n1, n2 = _noun(rng), _noun(rng)
    return {"kind": "spring", "nouns": (n1, n2),
            "load": (rng.uniform(*SIZES[w1]), rng.uniform(*SIZES[w2])),
            "squash": (min(c1 / 3, 0.6), min(c2 / 3, 0.6)),
            "text": (f"a {w1} {n1} rests on a {k1} spring and a {w2} {n2} "
                     f"rests on a {k2} spring"),
            "question": "which spring is squashed more?",
            "answer": "first" if c1 > c2 else "second"}


def draw_spring(spec, rng):
    """Two springs under load, each drawn at its own compressed height."""
    strokes, x = [], 0.0
    rest = 0.16
    program = S.SPRING_PROGRAMS[rng.integers(len(S.SPRING_PROGRAMS))]
    for noun, load, squash in zip(spec["nouns"], spec["load"], spec["squash"]):
        top = GROUND_Y + rest * (1 - squash)
        strokes += program(x, GROUND_Y, top, rng)
        strokes += _weight(noun, rng, x, top + load / 2, load, load)
        x += 0.24
    return _ground(rng, -0.07, x - 0.24 + 0.07) + strokes


def spec_pulley(rng):
    """Two weights over a pulley: the heavier side descends.

    Same question shape as the scale, but the mechanism is different and so is
    the drawing, so it tests whether the model reads the sketch rather than
    pattern-matching one layout.
    """
    sL, sR = _pick(rng, SIZES), _pick(rng, SIZES)
    if SIZE_RANK[sL] == SIZE_RANK[sR]:
        return None
    nL, nR = _noun(rng), _noun(rng)
    return {"kind": "pulley", "nouns": (nL, nR),
            "size": (rng.uniform(*SIZES[sL]), rng.uniform(*SIZES[sR])),
            "text": (f"a {sL} {nL} on the left and a {sR} {nR} on the right "
                     f"hang from a rope over a pulley"),
            "question": "which side goes down?",
            "answer": "left" if SIZE_RANK[sL] > SIZE_RANK[sR] else "right"}


def draw_pulley(spec, rng):
    """A pulley wheel with a rope down each side, drawn level."""
    r = rng.uniform(0.030, 0.040)
    top = 0.30
    half = max(r, rng.uniform(0.10, 0.14))
    strokes = S.pulley(0.0, top, r, rng)
    hang = top - rng.uniform(0.11, 0.15)
    # The rope leaves the wheel at its rim, not its centre, and runs over the
    # top: without that span the weights read as floating unattached.
    strokes.append(np.array([(-r, top), (0.0, top + r), (r, top)], float))
    for side, size, noun in ((-1, spec["size"][0], spec["nouns"][0]),
                             (1, spec["size"][1], spec["nouns"][1])):
        x = side * half
        strokes.append(np.array([(side * r, top), (x, top), (x, hang)], float))
        strokes += _weight(noun, rng, x, hang - size / 2, size, size)
    return strokes


PUSHES = {"gently": 0, "hard": 1}
CART_LOADS = {"empty": 0, "loaded": 1}


def spec_roll(rng):
    """Which cart rolls further? Push strength against surface drag.

    Wheels and a push arrow put a different idiom on the page: the answer is
    read from an annotation rather than from the shape of an object.
    """
    p1, p2 = _pick(rng, PUSHES), _pick(rng, PUSHES)
    s1, s2 = _pick(rng, SURFACES), _pick(rng, SURFACES)
    l1, l2 = _pick(rng, CART_LOADS), _pick(rng, CART_LOADS)
    # A heavier cart carries further for the same push on the same surface, so
    # load is a third independent factor rather than another distractor.
    t1 = PUSHES[p1] * 2 - SURFACES[s1] + CART_LOADS[l1]
    t2 = PUSHES[p2] * 2 - SURFACES[s2] + CART_LOADS[l2]
    if t1 == t2:
        return None
    return {"kind": "roll", "push": (PUSHES[p1], PUSHES[p2]),
            "rough": (SURFACES[s1], SURFACES[s2]),
            "load": (CART_LOADS[l1], CART_LOADS[l2]),
            "text": (f"a {l1} cart is pushed {p1} across {s1} ground and a "
                     f"{l2} cart is pushed {p2} across {s2} ground"),
            "question": "which cart rolls further?",
            "answer": "first" if t1 > t2 else "second"}


def draw_roll(spec, rng):
    """Two carts, each a body on wheels with a push arrow behind it."""
    strokes, x = [], 0.0
    wheel = S.WHEEL_PROGRAMS[rng.integers(len(S.WHEEL_PROGRAMS))]
    for push, rough, load in zip(spec["push"], spec["rough"], spec["load"]):
        r = 0.018
        body_w, body_h = 0.085, 0.042
        cy = GROUND_Y + 2 * r + body_h / 2
        strokes += S.box_closed(x, cy, body_w, body_h, rng)
        if load:                       # a loaded cart carries a box on top
            strokes += S.box_closed(x, cy + body_h / 2 + 0.020, 0.048, 0.040, rng)
        for side in (-1, 1):
            strokes += wheel(x + side * body_w * 0.30, GROUND_Y + r, r, rng)
        strokes += S.arrow(x - body_w / 2 - 0.055, cy,
                           0.038 * (1.7 if push else 1.0), 0.0, rng)
        if rough:
            for tick in np.linspace(x - 0.05, x + 0.05, 2 * rough):
                strokes.append(np.array([(tick, GROUND_Y),
                                         (tick - 0.014, GROUND_Y - 0.018)], float))
        x += 0.30
    return _ground(rng, -0.13, x - 0.30 + 0.11) + strokes


MAGNET_STRENGTHS = {"weak": 0, "strong": 2}
BOLT_MATERIALS = {"iron": 1, "steel": 1, "plastic": 0, "brass": 0}


def spec_magnet(rng):
    """Which nail is pulled in? Magnet strength falls off with distance.

    Strength and distance compose the way the lever's size and distance do, but
    the material attribute is a hard gate: a plastic bolt is never attracted,
    however strong the magnet. A rule with an override, rather than a pure
    score.
    """
    strength = _pick(rng, MAGNET_STRENGTHS)
    d1, d2 = _pick(rng, DISTS), _pick(rng, DISTS)
    if DIST_RANK[d1] == DIST_RANK[d2]:
        return None
    m1, m2 = _pick(rng, BOLT_MATERIALS), _pick(rng, BOLT_MATERIALS)
    # Material is a gate, not a score: a plastic bolt is never attracted
    # however close it sits or however strong the magnet. When only one bolt is
    # iron, distance stops mattering entirely -- a rule with an override.
    pull = [BOLT_MATERIALS[m1], BOLT_MATERIALS[m2]]
    if not any(pull):
        answer = "neither"
    elif pull[0] and not pull[1]:
        answer = "first"
    elif pull[1] and not pull[0]:
        answer = "second"
    else:
        answer = "first" if DIST_RANK[d1] < DIST_RANK[d2] else "second"
    return {"kind": "magnet", "strength": MAGNET_STRENGTHS[strength],
            "dist": (DIST_RANK[d1], DIST_RANK[d2]), "pull": pull,
            "text": (f"a {strength} magnet sits between a {m1} bolt {d1} on "
                     f"the left and a {m2} bolt {d2} on the right"),
            "question": "which bolt is pulled in?",
            "answer": answer}


def draw_magnet(spec, rng):
    """A horseshoe magnet with a bolt either side at its stated distance."""
    strokes = S.magnet_horseshoe(0.0, GROUND_Y + 0.03, 0.075, 0.085, rng)
    for i, rank in enumerate(spec["dist"]):
        side = -1 if i == 0 else 1
        d = (0.075, 0.115, 0.16)[rank]
        x = side * d
        # A bolt: a head with a shaft hanging below, drawn in one stroke so it
        # reads as one object at this size. Material is stated in the caption
        # rather than drawn -- iron and brass look alike on paper, and
        # inventing a visual tell would leak the gate the question turns on.
        strokes.append(np.array([(x - 0.011, GROUND_Y + 0.062),
                                 (x + 0.011, GROUND_Y + 0.062),
                                 (x, GROUND_Y + 0.062),
                                 (x, GROUND_Y + 0.018)], float))
    return _ground(rng, -0.21, 0.21) + strokes


# ------------------------------------------------------------- geometry
#
# Spatial questions where a drawing is genuinely the better medium: the setup
# is a handful of shapes in relative position, and describing it precisely
# enough in words takes more effort than looking at it. These are always
# generated elided -- the caption names the objects, the picture holds the
# arrangement -- so they sit at the "easy with the sketch, hard without" end of
# the corpus by construction rather than by a flag.


def spec_reach(rng):
    """Is the ladder long enough to reach the ledge?

    Two lengths compared through a right triangle rather than directly, so the
    comparison is geometric: a ladder set further out needs to be longer to
    reach the same height.
    """
    height = rng.uniform(0.16, 0.30)
    base = rng.uniform(0.08, 0.22)
    need = np.hypot(height, base)
    length = need * rng.uniform(0.72, 1.28)
    if abs(length - need) < 0.035:            # too close to call
        return None
    return {"kind": "reach", "height": height, "base": base, "length": length,
            "elided": "geometry",
            "text": "a ladder leans from the ground toward the top of a wall",
            "question": "does the ladder reach the top?",
            "answer": "reaches" if length >= need else "falls short"}


def draw_reach(spec, rng):
    """The wall, and the ladder drawn at its true length and angle."""
    h, b, L = spec["height"], spec["base"], spec["length"]
    strokes = [np.array([(0.0, GROUND_Y), (0.0, h)], float),
               np.array([(-0.03, h), (0.03, h)], float)]
    # The ladder keeps its foot at -b and runs at the angle a ladder of this
    # length would sit at, so its tip lands above or below the ledge honestly.
    angle = np.arccos(min(1.0, b / max(L, 1e-6)))
    strokes.append(np.array([(-b, GROUND_Y),
                             (-b + L * np.cos(angle), L * np.sin(angle))], float))
    return _ground(rng, -b - 0.05, 0.06) + strokes


def spec_fit(rng):
    """Does the box fit through the doorway?

    Width against width, but the box may be turned on its side, so the answer
    depends on which dimension is compared -- a question that is immediate in a
    picture and wordy in text.
    """
    door = rng.uniform(0.09, 0.16)
    w = rng.uniform(0.06, 0.20)
    h = rng.uniform(0.06, 0.20)
    turned = rng.random() < 0.5
    span = h if turned else w                 # the dimension facing the door
    if abs(span - door) < 0.022:
        return None
    return {"kind": "fit", "door": door, "w": w, "h": h, "turned": turned,
            "elided": "geometry",
            "text": ("a box is carried toward a doorway"
                     + (" on its side" if turned else "")),
            "question": "does the box fit through?",
            "answer": "fits" if span < door else "too wide"}


def draw_fit(spec, rng):
    """The doorway as two posts, and the box beside it at its orientation."""
    door, w, h = spec["door"], spec["w"], spec["h"]
    top = 0.26
    strokes = []
    for side in (-1, 1):
        x = side * door / 2
        strokes.append(np.array([(x, GROUND_Y), (x, top)], float))
    strokes.append(np.array([(-door / 2, top), (door / 2, top)], float))
    bw, bh = (h, w) if spec["turned"] else (w, h)
    strokes += S.box_closed(-0.22, GROUND_Y + bh / 2, bw, bh, rng)
    return _ground(rng, -0.34, 0.12) + strokes


def spec_shadow(rng):
    """Which pole casts the longer shadow? Height and sun angle.

    The sun's angle is drawn as a ray rather than named, so the caption cannot
    state it and the picture has to be read.
    """
    h1, h2 = _pick(rng, SIZES), _pick(rng, SIZES)
    if SIZE_RANK[h1] == SIZE_RANK[h2]:
        return None
    return {"kind": "shadow",
            "height": (rng.uniform(*HEIGHTS[h1]), rng.uniform(*HEIGHTS[h2])),
            "angle": rng.uniform(25.0, 55.0), "elided": "geometry",
            "text": "two poles stand in the sun",
            "question": "which pole has the longer shadow?",
            "answer": "first" if SIZE_RANK[h1] > SIZE_RANK[h2] else "second"}


def draw_shadow(spec, rng):
    """Poles with a sun ray down each, and the shadow it casts on the ground."""
    strokes, x = [], 0.0
    t = np.tan(np.deg2rad(spec["angle"]))
    for h in spec["height"]:
        strokes.append(np.array([(x, GROUND_Y), (x, h)], float))
        shadow = h * t
        strokes += S.arrow(x - 0.05, h + 0.05, 0.04, -0.04, rng)
        strokes.append(np.array([(x, GROUND_Y), (x + shadow, GROUND_Y)], float))
        strokes.append(np.array([(x + shadow, GROUND_Y),
                                 (x + shadow, GROUND_Y - 0.018)], float))
        x += shadow + 0.14
    return _ground(rng, -0.10, x - 0.06) + strokes


SPECS = {"lever": spec_lever, "ramp": spec_ramp, "stack": spec_stack,
         "pendulum": spec_pendulum, "scale": spec_scale, "drop": spec_drop,
         "float": spec_float, "spring": spec_spring, "pulley": spec_pulley,
         "roll": spec_roll, "magnet": spec_magnet,
         "reach": spec_reach, "fit": spec_fit, "shadow": spec_shadow}

# Archetypes whose caption deliberately does not determine the answer: the
# arrangement lives in the drawing. Held to different expectations from the
# rest -- low caption diversity is correct for them, and a caption that became
# sufficient would be the bug.
GEOMETRY_KINDS = ("reach", "fit", "shadow")
DRAWERS = {"lever": draw_lever, "ramp": draw_ramp, "stack": draw_stack,
           "pendulum": draw_pendulum, "scale": draw_scale, "drop": draw_drop,
           "float": draw_float, "spring": draw_spring, "pulley": draw_pulley,
           "roll": draw_roll, "magnet": draw_magnet,
           "reach": draw_reach, "fit": draw_fit, "shadow": draw_shadow}


# ------------------------------------------------------------------- chains
#
# The archetypes above are all one step: read the caption, make one comparison,
# answer. A scratchpad has nothing to hold at that depth, so it cannot help --
# measuring scratchpad benefit needs problems where intermediate state exists
# and must survive to the next step.
#
# A chain stage is a different shape from an archetype. It *consumes* a carry
# value from the previous stage and *produces* one for the next, so the stages
# compose into a dependency line rather than sitting side by side. `depth` is
# then the number of stages, and it is the only thing that changes between a
# depth-1 and a depth-4 example: same vocabulary, same drawing style, same
# answer format. That is what makes it a clean independent variable for a
# scaling study.
#
# Every stage records its own input and output in meta["steps"], so an
# intermediate state can be verified rather than only the final answer. That is
# the point of generating these procedurally: process reward needs per-step
# ground truth, which a scraped corpus cannot give you.
#
# The carry is a small discrete value -- a side ("left"/"right"), a survivor
# ("first"/"second"), or a speed level -- so a stage never has to serialize a
# continuous quantity into the caption.

CHAIN_GAP = 0.30
# Token cost scales with ink size at a fixed grid, and a chain stage is a
# schematic element rather than a whole picture, so stages are drawn smaller
# than the standalone archetypes.
CHAIN_SCALE = 0.62


# A stage caption must never name the carry it received. Naming it states the
# previous stage's answer, so the chain can be solved by reading the last
# clause alone and every earlier step becomes decorative -- depth stops being
# the variable it is supposed to be. Measured before this was fixed: the final
# clause alone predicted the answer 95% of the time at depth 4.
#
# So stages refer to the carry only as "that side" / "whichever side", and
# describe a *transformation* ("feeds the opposite side") rather than a state.


def stage_gate(carry, rng, index):
    """A two-door gate that either keeps the incoming side or swaps it.

    Pure routing with no physics of its own. Its job in a chain is to make the
    previous stage's answer *matter* to the next one -- the doors lead to
    different places, so losing the carry loses everything downstream. The
    caption states only whether the gate crosses over, never which side arrived.
    """
    swap = rng.random() < 0.5
    out = ("right" if carry == "left" else "left") if swap else carry
    text = ("that side feeds a gate below which sends the ball out "
            f"{'the opposite way' if swap else 'the same way'}")
    return {"stage": "gate", "text": text, "in": carry, "out": out,
            "swap": swap}


def stage_slope(carry, rng, index):
    """Two ramps side by side; the carried side names which one the ball takes.

    Produces a speed level (0 low, 1 high) from that ramp's height and
    roughness, so the next stage receives something physical rather than a
    label that was merely passed along.
    """
    sides = ("left", "right") if carry in ("left", "right") else ("first", "second")
    taken = 0 if carry in ("left", "first") else 1
    heights = [_pick(rng, SIZES), _pick(rng, SIZES)]
    roughs = [_pick(rng, SURFACES), _pick(rng, SURFACES)]
    score = SIZE_RANK[heights[taken]] - SURFACES[roughs[taken]]
    if score == 0:                                  # too close to call
        return None
    fast = score > 0
    # The carry names which ramp is taken, so the caption must not assert one:
    # "the ramp on that side" keeps the reference to the previous stage's
    # answer rather than leaking it.
    text = (f"a ball rolls off that side onto the ramp below it, one of two -- a "
            f"{heights[0]} {roughs[0]} ramp on the {sides[0]} and a "
            f"{heights[1]} {roughs[1]} ramp on the {sides[1]}")
    return {"stage": "slope", "text": text, "in": carry,
            "out": "fast" if fast else "slow",
            "height": heights, "rough": roughs, "taken": taken}


def stage_gap(carry, rng, index):
    """A ball leaves a ledge and either clears a gap or falls in.

    Consumes a speed level and produces a side, converting the carry back into
    something a gate or lever can route. Wide gaps need speed; narrow ones do
    not, so both carry values reach both outcomes and the answer is never
    guessable from the stage alone.
    """
    if carry not in ("fast", "slow"):
        return None
    # Three gap widths rather than two, so the outcome is balanced. With only
    # wide/narrow the rule (fast clears, or narrow clears) says "far" three
    # times in four, and that skew propagates down the chain: a later seesaw
    # inherits it through the far -> right mapping and its answers stop being
    # 50/50. An "impossible" gap that nothing clears restores the balance while
    # keeping both carry values relevant.
    width = ("narrow", "wide", "impossible")[rng.integers(3)]
    clears = {"narrow": True, "wide": carry == "fast", "impossible": False}[width]
    text = f"it launches off a ledge across a {width} gap"
    return {"stage": "gap", "text": text, "in": carry,
            "out": "far" if clears else "near", "width": width,
            # Only a wide gap actually consults the incoming speed. The other
            # two widths have the same outcome either way, so a chain ending on
            # one is answerable from its last clause alone; spec_chain refuses
            # to terminate on a stage that does not use its carry.
            "uses_carry": width == "wide"}


def stage_seesaw(carry, rng, index):
    """A ball lands on one end of a seesaw, tipping the far side down.

    Accepts either a side or a landing position and produces a side, so it can
    terminate a chain with the familiar "which side goes down" question.
    """
    landing = {"left": "left", "right": "right",
               "far": "right", "near": "left",
               "first": "left", "second": "right"}.get(carry)
    if landing is None:
        return None
    out = "right" if landing == "left" else "left"
    # "that end", not the named end: naming it would state the incoming carry.
    text = "it drops onto that end of a seesaw"
    return {"stage": "seesaw", "text": text, "in": carry, "out": out,
            "landing": landing}


def stage_tank(carry, rng, index):
    """The ball drops into a tank and either floats or sinks.

    Consumes a landing position and produces a float/sink state, bringing the
    buoyancy domain into chains. The material is stated but which tank the ball
    reaches is not, so the carry still has to survive.
    """
    if carry not in ("far", "near"):
        return None
    fluids = [_pick(rng, FLUIDS), _pick(rng, FLUIDS)]
    material = _pick(rng, DENSITIES)
    reached = 0 if carry == "near" else 1
    if DENSITIES[material] == FLUIDS[fluids[reached]]:
        return None
    floats = DENSITIES[material] < FLUIDS[fluids[reached]]
    # Same 6-to-4 float/sink imbalance as spec_float; rejected the same way.
    if floats and rng.random() < 1 / 3:
        return None
    text = (f"the {material} ball drops into one of two tanks, "
            f"{fluids[0]} on the near side and {fluids[1]} on the far side")
    return {"stage": "tank", "text": text, "in": carry,
            "out": "floats" if floats else "sinks",
            "fluids": fluids, "reached": reached}


def stage_lift(carry, rng, index):
    """A float/sink state decides which side of a pulley is loaded.

    Converts the buoyancy carry back into a side, so tank does not have to be
    terminal and chains can keep going through it.
    """
    if carry not in ("floats", "sinks"):
        return None
    swap = rng.random() < 0.5
    loaded = "left" if (carry == "floats") != swap else "right"
    text = ("that outcome decides which pan of a pulley is loaded -- "
            f"{'floating' if not swap else 'sinking'} loads the left")
    return {"stage": "lift", "text": text, "in": carry, "out": loaded,
            "swap": swap}


def stage_spring(carry, rng, index):
    """A speed carry compresses a spring a lot or a little.

    Produces a bounce height, adding a fourth carry vocabulary so chains are
    not confined to sides and speeds.
    """
    if carry not in ("fast", "slow"):
        return None
    stiff = _pick(rng, STIFFNESS)
    high = (carry == "fast") and STIFFNESS[stiff] < 3
    text = f"it slams into a {stiff} spring at the end of the track"
    return {"stage": "spring", "text": text, "in": carry,
            "out": "high" if high else "low", "stiff": STIFFNESS[stiff],
            # A stiff spring returns "low" whatever arrives, so it must not end
            # a chain -- same rule as an impossible gap.
            "uses_carry": STIFFNESS[stiff] < 3}


def stage_bounce(carry, rng, index):
    """A bounce height decides which of two shelves the ball lands on."""
    if carry not in ("high", "low"):
        return None
    swap = rng.random() < 0.5
    out = ("right" if carry == "high" else "left") if swap else \
          ("left" if carry == "high" else "right")
    text = ("it comes down on one of two shelves, the taller one on the "
            f"{'right' if swap else 'left'}")
    return {"stage": "bounce", "text": text, "in": carry, "out": out,
            "swap": swap}


CHAIN_STAGES = {"gate": stage_gate, "slope": stage_slope,
                "gap": stage_gap, "seesaw": stage_seesaw,
                "tank": stage_tank, "lift": stage_lift,
                "spring": stage_spring, "bounce": stage_bounce}

# Which stages accept which carry type. A chain is built by walking this graph,
# so a stage never receives a carry it cannot consume and no chain has to be
# thrown away for being ill-typed.
_ACCEPTS = {"gate": ("left", "right"),
            "slope": ("left", "right", "first", "second"),
            "gap": ("fast", "slow"),
            "seesaw": ("left", "right", "far", "near", "first", "second"),
            "tank": ("far", "near"),
            "lift": ("floats", "sinks"),
            "spring": ("fast", "slow"),
            "bounce": ("high", "low")}

CHAIN_QUESTIONS = {
    # depth 1 is the bare lever, so a chain sweep starts from the same one-step
    # problem the standalone archetypes pose
    "lever": "which side tips down?",
    "gate": "which side does the ball come out on?",
    "slope": "is the ball moving fast or slow at the bottom?",
    "gap": "does the ball land far or near?",
    "seesaw": "which side of the seesaw goes down?",
    "tank": "does the ball float or sink?",
    "lift": "which side goes down?",
    "spring": "does it bounce high or low?",
    "bounce": "which shelf does it land on?",
}


def spec_chain(rng, depth=2):
    """A chain of `depth` stages, each consuming the previous stage's output.

    The opening stage is a plain lever, so a depth-1 chain is exactly the
    one-step problem the other archetypes pose and the depth sweep starts from
    a familiar baseline.

    Returns None when a stage rejects its draw (a tie, or a carry it cannot
    take); the caller retries. Rejection is cheap and keeps every accepted
    chain fully determined at every step.
    """
    head = spec_lever(rng)
    if head is None:
        return None
    carry = head["answer"]
    parts = [head["text"]]
    steps = [{"stage": "lever", "in": None, "out": carry}]

    for i in range(depth - 1):
        options = [n for n, ok in _ACCEPTS.items() if carry in ok]
        stage = CHAIN_STAGES[options[rng.integers(len(options))]]
        made = stage(carry, rng, i)
        if made is None:
            return None
        # A final stage that ignores its carry makes the whole chain solvable
        # from its last clause, which is the failure this design exists to
        # avoid. Reject rather than emit a shallow example wearing a depth-N
        # label.
        if i == depth - 2 and not made.get("uses_carry", True):
            return None
        parts.append(made["text"])
        steps.append({"stage": made["stage"], "in": made["in"],
                      "out": made["out"], "spec": made})
        carry = made["out"]

    return {"kind": f"chain{depth}", "depth": depth, "steps": steps,
            "text": ", then ".join(parts),
            "question": CHAIN_QUESTIONS[steps[-1]["stage"]],
            "answer": carry}


def draw_chain(spec, rng):
    """Draw each stage in its own column, left to right along the chain.

    Reading order is the dependency order, so the picture carries the same
    structure the caption does. Each stage is drawn by a small compiler of its
    own; the shared ground line ties them into one scene.

    Stages are drawn at CHAIN_SCALE, smaller than the standalone archetypes.
    Token cost is proportional to ink size at a fixed grid, and inside a chain
    a stage is a schematic element rather than the whole picture, so the extra
    size bought nothing. Shrinking is what keeps per-stage cost low enough that
    a depth-4 sketch still fits a modest block.
    """
    strokes, x = [], 0.0
    for step in spec["steps"]:
        stage = step["stage"]
        if stage == "lever":
            sub = _draw_chain_lever(rng)
        elif stage == "gate":
            sub = _draw_gate(step["spec"], rng)
        elif stage == "slope":
            sub = _draw_slope(step["spec"], rng)
        elif stage == "gap":
            sub = _draw_gap(step["spec"], rng)
        elif stage == "tank":
            sub = _draw_tank(step["spec"], rng)
        elif stage == "lift":
            sub = _draw_lift(step["spec"], rng)
        elif stage == "spring":
            sub = _draw_spring_stage(step["spec"], rng)
        elif stage == "bounce":
            sub = _draw_bounce(step["spec"], rng)
        else:
            sub = _draw_seesaw(step["spec"], rng)
        strokes += [np.asarray(s, float) * CHAIN_SCALE + np.array([x, 0.0])
                    for s in sub]
        x += CHAIN_GAP
    return strokes


def _draw_chain_lever(rng):
    """The opening lever, as a plank on a pivot with a weight at each end.

    Deliberately simpler than draw_lever: inside a chain the head only has to
    show that a seesaw is the starting point, and its sizes and distances are
    already stated in the caption.
    """
    half, plank_y = 0.14, 0.10
    out = [np.array([(-half, plank_y), (half, plank_y)], float)]
    # The pivot is built to reach exactly plank_y, so the plank rests on its
    # apex rather than floating above it or cutting through it.
    out += S.PIVOT_PROGRAMS[rng.integers(len(S.PIVOT_PROGRAMS))](
        0.0, GROUND_Y, 0.07, plank_y, rng)
    for side in (-1, 1):
        out += S.box_closed(side * half * 0.7, plank_y + 0.026, 0.050, 0.050, rng)
    return out


def _draw_gate(spec, rng):
    """A funnel over two doors, with the taken door left open."""
    w, top, mid = 0.11, 0.20, 0.10
    out = [np.array([(-w, top), (0.0, mid)], float),
           np.array([(w, top), (0.0, mid)], float),
           np.array([(0.0, mid), (0.0, GROUND_Y + 0.02)], float)]
    for side in (-1, 1):
        out.append(np.array([(side * w, GROUND_Y + 0.06),
                             (side * w, GROUND_Y)], float))
    if spec["swap"]:
        out.append(np.array([(-w * 0.5, mid * 0.6), (w * 0.5, mid * 0.4)], float))
    return out


def _draw_slope(spec, rng):
    """Two ramps; the one the ball takes carries the ball."""
    out, x0 = [], -0.13
    for i, (hk, rk) in enumerate(zip(spec["height"], spec["rough"])):
        h = np.mean(HEIGHTS[hk])
        w = h * 1.4
        out.append(np.array([(x0, GROUND_Y), (x0 + w, GROUND_Y),
                             (x0, h), (x0, GROUND_Y)], float))
        out += _ramp_hatch(x0, w, h, SURFACES[rk], rng)
        if i == spec["taken"]:
            out += _ball(rng, x0 + 0.012, h + 0.022, 0.022)
        x0 += w + 0.05
    return out


def _draw_gap(spec, rng):
    """Two ledges separated by a gap, drawn at its stated width."""
    gap = {"narrow": 0.05, "wide": 0.11, "impossible": 0.18}[spec["width"]]
    h = 0.14
    return [np.array([(-0.10, h), (0.0, h)], float),
            np.array([(0.0, h), (0.0, GROUND_Y)], float),
            np.array([(gap, GROUND_Y), (gap, h * 0.6)], float),
            np.array([(gap, h * 0.6), (gap + 0.10, h * 0.6)], float)]


def _draw_tank(spec, rng):
    """Two vessels side by side, each with a waterline.

    Both are drawn and neither is marked: which one the ball reaches is the
    carry, so marking it would state the previous stage's answer.
    """
    out, w, h = [], 0.10, 0.11
    for i in (0, 1):
        cx = -0.08 + i * 0.16
        out += S.beaker(cx, GROUND_Y, w, h, rng)
        out += S.fill_level(cx, GROUND_Y, w, GROUND_Y + h * 0.68, rng)
    return out


def _draw_lift(spec, rng):
    """A small pulley with a pan hanging either side."""
    r, top, half = 0.028, 0.24, 0.09
    out = S.pulley(0.0, top, r, rng)
    for side in (-1, 1):
        x = side * half
        out.append(np.array([(x, top), (x, top - 0.10)], float))
        out.append(np.array([(x - 0.030, top - 0.10),
                             (x + 0.030, top - 0.10)], float))
    return out


def _draw_spring_stage(spec, rng):
    """A spring standing against a back wall, drawn at its rest length."""
    program = S.SPRING_PROGRAMS[rng.integers(len(S.SPRING_PROGRAMS))]
    out = [np.array([(0.09, GROUND_Y), (0.09, 0.16)], float)]
    out += program(0.045, GROUND_Y + 0.02, GROUND_Y + 0.12, rng, width=0.016,
                   coils=3)
    return out


def _draw_bounce(spec, rng):
    """Two shelves at different heights, the taller one on its stated side."""
    tall = 0.19 if spec["swap"] else 0.12
    short = 0.12 if spec["swap"] else 0.19
    out = []
    for x, h in ((-0.10, short), (0.10, tall)):
        out.append(np.array([(x - 0.05, h), (x + 0.05, h)], float))
        out.append(np.array([(x, h), (x, GROUND_Y)], float))
    return out


def _draw_seesaw(spec, rng):
    """A level plank on a pivot, drawn level: the drawing is the premise."""
    half, plank_y = 0.13, 0.075
    out = [np.array([(-half, plank_y), (half, plank_y)], float)]
    out += S.PIVOT_PROGRAMS[rng.integers(len(S.PIVOT_PROGRAMS))](
        0.0, GROUND_Y, 0.06, plank_y, rng)
    side = -1 if spec["landing"] == "left" else 1
    out += _ball(rng, side * half * 0.8, plank_y + 0.025, 0.022)
    return out


def make_example(kind, rng):
    """One example. `kind` is an archetype name, or "chainN" for a depth-N chain."""
    if kind.startswith("chain"):
        depth = int(kind[5:])
        spec = spec_chain(rng, depth)
        if spec is None:
            return None
        strokes = DRAWERS_CHAIN(spec, rng)
    else:
        spec = SPECS[kind](rng)
        if spec is None:
            return None
        strokes = DRAWERS[kind](spec, rng)
    strokes = S.humanize(strokes, rng)
    meta = {k: spec[k] for k in ("kind", "question", "answer")}
    if spec.get("elided"):
        # Which attribute the caption withholds, so the picture-needed subset
        # can be selected and scored on its own.
        meta["elided"] = spec["elided"]
    if "depth" in spec:
        meta["depth"] = spec["depth"]
        # Per-step ground truth, for verifying intermediate states rather than
        # only the final answer. This is what a scraped corpus cannot provide.
        meta["steps"] = [{k: s[k] for k in ("stage", "in", "out")}
                         for s in spec["steps"]]
    return {"text": spec["text"],
            "points": scene_to_points(strokes).tolist(),
            "meta": meta}


DRAWERS_CHAIN = draw_chain


def generate(n, seed=0, depths=None):
    """`depths` selects chain depths to mix in, e.g. (1, 2, 3, 4).

    With depths=None the six single-step archetypes are generated round-robin,
    as before. With depths given, chains of each requested depth are generated
    round-robin instead, so a depth sweep holds everything constant except the
    number of reasoning steps.
    """
    rng = np.random.default_rng(seed)
    names = [f"chain{d}" for d in depths] if depths else list(SPECS)
    out = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 200 * n + 10_000:
            raise SystemExit(
                f"only generated {len(out)} of {n} examples; a stage is "
                f"rejecting nearly every draw")
        made = make_example(names[len(out) % len(names)], rng)
        if made is not None:
            out.append(made)
    return out


def check_captions(examples, max_text_length):
    """Fail if --max_text_length would cut a caption before its answer.

    Captions here are compositional: the answer depends on a comparison between
    two clauses, and the deciding clause is usually the second one. Truncating
    at the cursive default of 50 characters left 33% of physics_v0 with a prompt
    that does not determine its own answer -- two examples with identical
    prompts and opposite labels, which is unlearnable noise rather than a hard
    example. Training loss falls anyway, so nothing downstream catches it.

    The strict check is not caption length but whether a truncated prompt still
    picks out one answer, so that is what is measured.

    Examples whose caption deliberately withholds an attribute (meta["elided"],
    including every geometry archetype) are exempt: their prompt is *supposed*
    to be underdetermined, with the drawing carrying the rest. Truncation
    damage is a bug; elision is the design.
    """
    examples = [e for e in examples if not e["meta"].get("elided")]
    if not examples:
        return 0
    longest = max(len(e["text"]) for e in examples)
    seen = {}
    for e in examples:
        seen.setdefault(e["text"][:max_text_length], set()).add(e["meta"]["answer"])
    ambiguous = {k: v for k, v in seen.items() if len(v) > 1}
    if ambiguous:
        n_hit = sum(1 for e in examples
                    if len(seen[e["text"][:max_text_length]]) > 1)
        example = next(iter(ambiguous))
        raise SystemExit(
            f"--max_text_length {max_text_length} truncates captions to a prefix "
            f"that no longer determines the answer: {len(ambiguous)} prefixes, "
            f"{n_hit:,} of {len(examples):,} examples ({n_hit / len(examples):.0%}).\n"
            f"  e.g. {example!r} -> {sorted(ambiguous[example])}\n"
            f"Longest caption is {longest} characters; pass "
            f"--max_text_length {longest} when training.")
    return longest


def main():
    p = argparse.ArgumentParser(description="Generate the physics sketch dataset")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--out", type=str, default="data/physics_v1.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_text_length", type=int, default=110,
                   help="the value you will pass to train.py; generation fails "
                        "if it would truncate a caption past its answer")
    p.add_argument("--preview", action="store_true",
                   help="render per-archetype validation sheets")
    p.add_argument("--depths", type=str, default="",
                   help="comma-separated chain depths, e.g. 1,2,3,4; "
                        "empty generates the single-step archetypes")
    p.add_argument("--style", type=str, default="rich", choices=("rich", "plain"),
                   help="plain drops decoration that carries no information "
                        "about the answer, for a shorter sketch")
    p.add_argument("--detail", type=str, default="full",
                   choices=("full", "elided"),
                   help="elided withholds one attribute from the caption, "
                        "leaving it in the drawing alone")
    args = p.parse_args()

    global STYLE, DETAIL
    STYLE, DETAIL = args.style, args.detail
    depths = tuple(int(d) for d in args.depths.split(",") if d.strip())

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pengpt.sampling import draw
        rng = np.random.default_rng(args.seed)
        kinds = [f"chain{d}" for d in depths] if depths else list(SPECS)
        fig, axes = plt.subplots(len(kinds), 8, figsize=(22, 2.1 * len(kinds)),
                                 squeeze=False)
        for row, kind in enumerate(kinds):
            for col in range(8):
                ex = None
                while ex is None:
                    ex = make_example(kind, rng)
                ax = axes[row][col]
                draw(ax, np.array(ex["points"]), color="k", linewidth=1.0)
                ax.set_title(f'{ex["meta"]["answer"]}', fontsize=7)
            axes[row][0].set_ylabel(kind, fontsize=10, rotation=0,
                                    ha="right", va="center", labelpad=8)
        fig.suptitle(f"physics ({args.style}): eight draws per row "
                     f"(title = answer)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig("physics_preview.png", dpi=110, bbox_inches="tight")
        print("wrote physics_preview.png")
        return

    examples = generate(args.n, seed=args.seed, depths=depths or None)
    check_captions(examples, args.max_text_length)
    # Report over every caption, elided ones included: they are exempt from the
    # ambiguity check but still must not be cut off during training.
    longest = max(len(e["text"]) for e in examples)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    kinds = Counter(ex["meta"]["kind"] for ex in examples)
    print(f"wrote {len(examples)} examples to {args.out}")
    print(f"archetypes: {dict(kinds)}")
    print(f"longest caption: {longest} chars "
          f"(train with --max_text_length {longest})")

    # Distinct captions, not example count, is what bounds what a text-conditioned
    # model can learn: examples sharing a caption differ only in stroke style, so
    # they teach drawing variation rather than a new prompt. Reported per
    # archetype because the total hides a lever with hundreds against a ramp with
    # six.
    print("\ndistinct captions per archetype (the diversity that matters):")
    for kind in sorted({ex["meta"]["kind"] for ex in examples}):
        texts = {ex["text"] for ex in examples if ex["meta"]["kind"] == kind}
        n = sum(1 for ex in examples if ex["meta"]["kind"] == kind)
        print(f"  {kind:10s} {len(texts):5d} unique over {n:6,} examples"
              f"  ({n / max(len(texts), 1):.0f}x repeat)")
    unique = len({ex["text"] for ex in examples})
    print(f"  {'TOTAL':10s} {unique:5d} unique over {len(examples):6,} examples")


if __name__ == "__main__":
    main()
