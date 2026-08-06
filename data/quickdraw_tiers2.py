"""Quality judgements over a fresh 120-drawing sample, judged in batches of 15.

Smaller batches than the first pass of 30, which makes each comparison more
careful, and a larger sample so the calibration rests on more than 90 examples.
Same rubric: reward a recognizable subject with its parts present and connected,
penalize fragments, near-empty drawings, and anything with words written on it.
"""

TIERS = [
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


def comparisons():
    """Every cross-tier pair, better first."""
    out = []
    for i, better in enumerate(TIERS):
        for worse in TIERS[i + 1:]:
            for a in better:
                for b in worse:
                    out.append((a, b))
    return out


def tier_of(idx):
    for t, group in enumerate(TIERS):
        if idx in group:
            return t
    raise KeyError(idx)
