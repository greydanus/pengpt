import json
import zipfile

import numpy as np

from pengpt import INK_HEIGHT, _stroke_spans


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


def points_to_strokes(points):
    points = np.asarray(points, dtype=float)
    return [points[a:b, :2].round(5).tolist() for a, b in _stroke_spans(points[:, 2])]


def save_strokes(records, path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.split("/")[-1].removesuffix(".zip"), json.dumps(records))


def export_traces(accepted_path, out_path):
    rows = [json.loads(line) for line in open(accepted_path)]
    heights = {}
    for r in rows:
        ys = np.concatenate([np.asarray(s)[:, 1] for s in r["strokes"]])
        heights.setdefault(r["source"], []).append(np.quantile(ys, 0.95) - np.quantile(ys, 0.05))
    scale = {src: INK_HEIGHT / np.median(h) for src, h in heights.items()}
    records = []
    for r in rows:
        k, x0 = scale[r["source"]], r["strokes"][0][0][0]
        strokes = [[[round((x - x0) * k, 5), round((y - r["baseline"]) * k, 5)] for x, y in s] for s in r["strokes"]]
        records.append({"text": r["text"], "strokes": strokes, "style": r["style"], "source": r["source"],
                        "license": r["license"], "qc": {key: r["qc"][key] for key in ("coverage", "precision")}})
    save_strokes(records, out_path)
    print(f"exported {len(records)} words from {len(scale)} sources to {out_path}")

