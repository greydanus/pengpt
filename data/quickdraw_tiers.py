"""The hand judgements the shipped quality probe is calibrated on.

Two samples of Quick, Draw! drawings, ordered into quality tiers by eye from
rendered sheets: the first 90 in batches of thirty, the second 120 in batches of
fifteen, which made each comparison more careful. Cross-tier pairs become the
comparisons Bradley-Terry consumes; see pengpt.quality.

Indices refer to the sampling order recorded when the sheets were rendered, so
these lists are a record of the judgements rather than something to re-run. The
criteria are in pengpt.quality.JUDGE_RUBRIC: reward a recognizable subject whose
parts are present and connected, penalize fragments, near-empty drawings, and
anything with words written on it.
"""

# best -> worst
SAMPLE_A = [
    # excellent: complete subject, parts present, cleanly drawn
    [0, 21, 28, 6, 9, 55, 34, 78, 67, 60, 32, 38, 45, 87, 83],
    # good: clearly the subject, minor sloppiness
    [2, 13, 24, 27, 36, 39, 42, 47, 50, 51, 52, 53, 54, 56, 57,
     58, 59, 63, 68, 70, 72, 73, 74, 79, 80, 82, 86, 88],
    # fair: recognizable but crude or incomplete
    [1, 3, 4, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 22, 26,
     29, 30, 31, 33, 35, 43, 44, 48, 49, 62, 64, 65, 69, 71, 75, 76, 81, 85],
    # poor: barely a subject, fragmentary
    [5, 17, 37, 41, 46, 84, 89],
    # junk: a line, a scribble, or text written on the canvas
    [20, 23, 25, 40, 61, 66, 77],
]

SAMPLE_B = [
    # excellent: complete subject, parts present, cleanly drawn
    [7, 11, 17, 25, 34, 44, 53, 58, 69, 75, 83, 92, 95, 104, 106, 107, 110, 111,
     116, 5, 3, 16, 20, 22, 27, 31, 46, 48, 61, 76, 87],
    # good: clearly the subject, minor sloppiness
    [0, 1, 6, 9, 10, 15, 21, 23, 26, 30, 32, 33, 40, 41, 45, 50, 52, 55, 56, 57,
     63, 65, 66, 67, 70, 71, 72, 73, 74, 78, 80, 81, 82, 84, 85, 90, 91, 93, 94,
     96, 98, 99, 102, 103, 105, 108, 113, 114, 115, 118, 119],
    # fair: recognizable but crude or incomplete
    [4, 8, 12, 13, 18, 19, 24, 29, 36, 37, 39, 42, 47, 51, 54, 59, 60, 62, 68,
     77, 86, 88, 89, 97, 100, 101, 117],
    # poor: barely a subject, fragmentary
    [28, 38, 43, 49, 64, 79, 109],
    # junk: scribble, or words written on the canvas
    [2, 14, 35, 112],
]


def comparisons(tiers):
    """Every cross-tier pair from one sample, better first."""
    out = []
    for i, better in enumerate(tiers):
        for worse in tiers[i + 1:]:
            for a in better:
                for b in worse:
                    out.append((a, b))
    return out


def tier_of(tiers, index):
    for t, group in enumerate(tiers):
        if index in group:
            return t
    raise KeyError(index)
