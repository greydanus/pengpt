"""Hand-style variation for procedural sketches: one spec, many drawings.

The dataset design separates what a scene *is* (the spec, where ground truth
lives) from how it is *drawn*. This module is the drawing half: given clean
polyline strokes, it produces the kind of variation a room of people drawing
the same physics diagram would: different depictions of the same object, hand
tremor, corners that overshoot or fall short, strokes drawn in a different
order. Verifiability is untouched because none of this changes the spec.

Three layers of variation, independently seeded:

- **Depiction programs.** A weight can be a closed box, a box of four separate
  strokes, a hatched crate, or a round-shouldered sack; a pivot can be a
  triangle, a wedge, or a post on a base; ground can be a line or a line with
  hatch ticks. Each archetype samples one program per object per drawing.
- **Hand tremor.** Smooth low-frequency noise perpendicular to each path,
  interpolated from knots every ~5% of arc length, so lines wave the way a
  freehand line waves instead of jittering per point.
- **Endpoint behavior.** Human strokes overshoot corners or stop short and
  leave gaps; closed shapes often do not quite close. Endpoints get extended
  or trimmed by a few millimeters of canvas, and whole strokes rotate by up
  to ~2 degrees.

`humanize(strokes, rng)` applies tremor + endpoints + order variation to any
stroke list; the depiction programs live with their archetypes.
"""

import numpy as np


def _arc_lengths(path):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]


def tremor(path, rng, amp=0.004, wavelength=0.055):
    """Smooth perpendicular noise, like a steady hand rather than a shaky one."""
    path = np.asarray(path, dtype=float)
    if len(path) < 3:
        return path
    d = _arc_lengths(path)
    if d[-1] < 1e-6:
        return path
    n_knots = max(3, int(d[-1] / wavelength) + 2)
    knot_pos = np.linspace(0, d[-1], n_knots)
    knot_val = rng.normal(0, amp, n_knots)
    knot_val[0] = knot_val[-1] = 0            # anchors stay put
    offset = np.interp(d, knot_pos, knot_val)
    tangent = np.gradient(path, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1]) + 1e-9
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / norm[:, None]
    return path + offset[:, None] * normal


def _densify(path, spacing=0.018):
    """Vertices every ~spacing along the path, so tremor can reach straight
    lines: a 2-point plank has nothing to bend until it has vertices."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path
    d = _arc_lengths(path)
    if d[-1] < spacing:
        return path
    t = np.linspace(0, d[-1], max(2, int(d[-1] / spacing) + 1))
    return np.column_stack([np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])])


def end_behavior(path, rng, scale=0.008):
    """Overshoot past each end, or stop short of it, along the end tangent."""
    path = np.asarray(path, dtype=float)
    if len(path) < 3:
        return path
    for end in (-1, 0):
        delta = rng.uniform(-scale, scale)
        d = _arc_lengths(path)
        if delta < 0:                      # fall short: trim arc length
            keep = (d < d[-1] + delta) if end == -1 else (d > -delta)
            if keep.sum() >= 2:
                path = path[keep]
        else:                              # overshoot: extend along the tangent
            tangent = path[-1] - path[-2] if end == -1 else path[0] - path[1]
            tangent = tangent / (np.hypot(*tangent) + 1e-9)
            extra = (path[-1] + tangent * delta)[None] if end == -1 \
                else (path[0] + tangent * delta)[None]
            path = np.vstack([path, extra]) if end == -1 else np.vstack([extra, path])
    return path


def humanize(strokes, rng, amp=0.004, keep_first=True, max_scene_deg=1.5):
    """Densify + tremor + end behavior per stroke, one small whole-scene
    rotation, and stroke-order variation.

    Rotation is applied to the scene as a body, never per stroke: contact
    relationships (a ball resting on a ramp) survive, hanging strings stay
    plausibly vertical, and a balance beam cannot acquire a tilt that reads
    as the answer. Style noise stays uncorrelated with every label.

    keep_first holds the first stroke (ground or beam, the scene's anchor)
    first in the ordering, so the origin convention survives styling.
    """
    styled = []
    for i, stroke in enumerate(strokes):
        s = _densify(np.asarray(stroke, dtype=float))
        first = keep_first and i == 0
        if not first:
            s = end_behavior(s, rng)
        s = tremor(s, rng, amp=amp * (0.5 if first else 1.0))
        styled.append(s)
    center = np.vstack(styled).mean(0)
    a = np.deg2rad(rng.uniform(-max_scene_deg, max_scene_deg))
    c, s_ = np.cos(a), np.sin(a)
    rot = np.array([[c, s_], [-s_, c]])
    styled = [(p - center) @ rot + center for p in styled]
    if len(styled) > 2:
        head = 1 if keep_first else 0
        tail = styled[head:]
        # swap a random adjacent pair: order varies without losing structure-first
        j = rng.integers(0, len(tail) - 1)
        tail[j], tail[j + 1] = tail[j + 1], tail[j]
        styled = styled[:head] + tail
    return styled


# ---------------------------------------------------------------- depictions

def box_closed(cx, cy, w, h, rng):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return [np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], float)]


def box_four_strokes(cx, cy, w, h, rng):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [np.array([corners[i], corners[(i + 1) % 4]], float) for i in range(4)]


def box_hatched(cx, cy, w, h, rng):
    strokes = box_closed(cx, cy, w, h, rng)
    for f in (0.3, 0.6):
        x = cx - w / 2 + f * w
        strokes.append(np.array([(x, cy - h / 2), (x + 0.25 * w, cy + h / 2)], float))
    return strokes


def sack(cx, cy, w, h, rng):
    """A round-shouldered lump: flat bottom, superellipse-ish top."""
    t = np.linspace(0, np.pi, 20)
    top = np.column_stack([cx + (w / 2) * np.cos(t)[::-1],
                           cy - h / 2 + h * np.sin(t)[::-1] ** 0.8])
    bottom = np.array([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2)], float)
    return [np.vstack([bottom[::-1], top]), ]


# Depiction programs carry the noun a caption should use for them, so language
# and drawing stay coherent: a sack is called a sack, a hatched box a crate.
WEIGHT_PROGRAMS = {"box": box_closed, "block": box_four_strokes,
                   "crate": box_hatched, "sack": sack}
BOX_NOUNS = ["box", "block", "crate"]        # things that stack flat


def ball_circle(cx, cy, r, rng, n=26):
    a = np.linspace(0, 2 * np.pi, n) + rng.uniform(0, 2 * np.pi)
    return [np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])]


def ball_open_circle(cx, cy, r, rng, n=26):
    """Drawn in one sweep that does not quite close."""
    a = np.linspace(0, 2 * np.pi * rng.uniform(0.9, 0.97), n) + rng.uniform(0, 6)
    return [np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])]


BALL_PROGRAMS = [ball_circle, ball_open_circle]


def pivot_triangle(cx, base_y, w, h, rng):
    return [np.array([(cx - w / 2, base_y), (cx + w / 2, base_y),
                      (cx, base_y + h), (cx - w / 2, base_y)], float)]


def pivot_wedge(cx, base_y, w, h, rng):
    return [np.array([(cx - w / 2, base_y), (cx, base_y + h), (cx + w / 2, base_y)],
                     float)]


def pivot_post(cx, base_y, w, h, rng):
    return [np.array([(cx, base_y), (cx, base_y + h)], float),
            np.array([(cx - w / 2, base_y), (cx + w / 2, base_y)], float)]


PIVOT_PROGRAMS = [pivot_triangle, pivot_wedge, pivot_post]


def ground_line(x0, x1, y, rng):
    return [np.array([(x0, y), (x1, y)], float)]


def ground_hatched(x0, x1, y, rng):
    strokes = ground_line(x0, x1, y, rng)
    for x in np.arange(x0 + 0.03, x1, 0.09):
        strokes.append(np.array([(x, y), (x - 0.02, y - 0.025)], float))
    return strokes


GROUND_PROGRAMS = [ground_line, ground_line, ground_hatched]  # hatching is rarer
