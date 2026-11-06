import json

import numpy as np

from pengpt import INK_HEIGHT, _stroke_spans, save_ink


PRESETS = {
    "cursive": dict(
        dataset="data/bigbank_3500.ink.npz",
        showcase="Sing, O Muse, of Odysseus|The quick brown fox|Ithaca, 1184 BC.",
        showcase_writer="sam",
        out_dir="out/cursive",
    ),
    "quickdraw": dict(
        dataset="data/quickdraw_balanced_fixed.jsonl.gz",
        max_words=1, augment="general",
        max_text_length=24, max_seq_length=192,
        n_layer=6, n_embd=128, batch_size=32,
        out_dir="out/quickdraw",
    ),
    "icons": dict(
        dataset="data/icons.jsonl",
        max_words=1, augment="general",
        grid=0.012, max_text_length=36, max_seq_length=128,
        tremor=0.004,
        n_layer=4, n_embd=96, batch_size=32,
        out_dir="out/icons",
    ),
}


def normalize(points):
    points = np.asarray(points, dtype=float).copy()
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        down = points
    height = np.quantile(down[:, 1], 0.95) - np.quantile(down[:, 1], 0.05)
    points[:, :2] *= INK_HEIGHT / max(height, 1e-6)
    points[:, 0] -= points[0, 0]
    points[:, 1] -= np.quantile(points[points[:, 2] == 1][:, 1], 0.95)
    return points


def points_to_strokes(points):
    points = np.asarray(points, dtype=float)
    return [points[a:b, :2].round(5).tolist() for a, b in _stroke_spans(points[:, 2])]


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
        records.append({"text": r["text"], "strokes": strokes, "style": r["style"],
                        "writer": r.get("writer", r["source"]), "source": r["source"], "license": r["license"]})
    save_ink(records, out_path)
    print(f"exported {len(records)} words from {len(scale)} sources to {out_path}")

