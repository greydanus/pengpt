import numpy as np

from pengpt import INK_HEIGHT


PRESETS = {
    "cursive": dict(
        dataset="data/bigbank_3500.json.zip",
        showcase="Sing, O Muse, of Odysseus|The quick brown fox|Ithaca, 1184 BC.",
        out_dir="out/cursive",
    ),
    "quickdraw": dict(
        dataset="data/quickdraw_balanced_fixed.jsonl.gz",
        max_words=1, augment="general",
        max_text_length=24, max_seq_length=192,
        n_layer=6, n_embd=128, learning_rate=1e-3, batch_size=32,
        out_dir="out/quickdraw",
    ),
    "icons": dict(
        dataset="data/icons.jsonl",
        max_words=1, augment="general",
        grid=0.012, max_text_length=36, max_seq_length=128,
        tremor=0.004, rotate=2.0,
        n_layer=4, n_embd=96, learning_rate=1e-3, batch_size=32,
        out_dir="out/icons",
    ),
}


def to_absolute(points):
    points = np.asarray(points, dtype=float).copy()
    span = points[:, :2].max(0) - points[:, :2].min(0)
    step = np.abs(np.diff(points[:, :2], axis=0)).mean() if len(points) > 1 else 0.0
    if step > 0 and span.max() < 12 * step:
        points[:, :2] = np.cumsum(points[:, :2], axis=0)
    return points


def normalize(points, pen_down_is=1, height_quantile=0.95, absolute=False):
    points = np.asarray(points, dtype=float).copy() if absolute else to_absolute(points)
    if pen_down_is != 1:
        points[:, 2] = 1.0 - points[:, 2]
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        down = points
    height = (np.quantile(down[:, 1], height_quantile)
              - np.quantile(down[:, 1], 1 - height_quantile))
    points[:, :2] *= INK_HEIGHT / max(height, 1e-6)
    points[:, 0] -= points[0, 0]
    points[:, 1] -= np.quantile(points[points[:, 2] == 1][:, 1], 0.95)
    return points
