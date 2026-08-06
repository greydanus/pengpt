"""My quality judgements over the 90-drawing sample, as ordered tiers.

Judged listwise from rendered sheets of 30, the way hkb's reranker works: a
judge shown many items at once orders them far more cheaply than one shown
pairs, and the ordering is what Bradley-Terry needs.

Tiers rather than a strict total order, because judging "is 43 better than 44"
is noise while "these five are excellent and those five are junk" is reliable.
Pairwise comparisons are then generated between tiers, which is exactly the
signal Bradley-Terry consumes.

What I rewarded: a recognizable subject with its parts present and connected --
a dog with legs and ears, a car with wheels and windows. What I penalized:
disconnected fragments, drawings that are mostly a single line, and anything
with words written on it.
"""

# best -> worst
TIERS = [
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
