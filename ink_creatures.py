import argparse
import json

import numpy as np

from pengpt.convert import normalize

FILES = [("bird", "creative_birds_json.txt", "data/birds.jsonl"),
         ("creature", "creative_creatures_json.txt", "data/creatures.jsonl")]


def to_points(all_strokes):
    rows = []
    for step in all_strokes:
        for stroke in step:
            if not stroke:
                continue
            pts = [stroke[0][:2]] + [seg[2:4] for seg in stroke]
            for x, y in pts:
                rows.append([float(x), float(y), 1.0])
            rows.append([rows[-1][0], rows[-1][1], 0.0])
    if len(rows) < 4:
        return None
    return normalize(np.array(rows), absolute=True).round(4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",
                   default="data/raw/doodlergan/DoodlerGAN/raw_data_clean")
    args = p.parse_args()

    for kind, src, out_path in FILES:
        data = json.load(open(f"{args.raw_dir}/{src}"))
        n = 0
        with open(out_path, "w") as out:
            for item in data:
                if item.get("good_sample") != 1:
                    continue
                points = to_points(item["all_strokes"])
                if points is None:
                    continue
                desc = (item.get("description") or "").strip()
                parts = sorted({p for p in item.get("partsUsed", [])
                                if p not in ("initial", "details")})
                text = desc if desc else f"{kind} with " + ", ".join(parts)
                out.write(json.dumps({
                    "text": text, "points": points.tolist(),
                    "meta": {"kind": kind, "parts": parts}}) + "\n")
                n += 1
        print(f"{kind}: wrote {n} to {out_path}")


if __name__ == "__main__":
    main()
