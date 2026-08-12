import argparse
import json
import re

import numpy as np

from pengpt.data import _read


def canonical(text):
    text = text.lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9 !?',.:;\"]", " ", text)
    return " ".join(text.split())


SOURCES = [
    ("doodle", "data/quickdraw_balanced_fixed.jsonl.gz", 120_000),
    ("doodle", "data/quickdraw_captioned.jsonl", 0),
    ("sketch", "data/sketchy_tags.jsonl", 0),
    ("icon", "data/icons_captioned.jsonl", 0),
    ("scene", "data/fscoco.json", 0),
    ("bird", "data/birds.jsonl", 0),
    ("creature", "data/creatures.jsonl", 0),
]

COLORS = {"black", "white", "grey", "gray", "brown", "red", "orange", "yellow",
          "green", "blue", "purple", "pink", "beige", "tan", "golden", "gold",
          "silver", "dark", "light"}


def strip_colors(text):
    kept = [w for w in text.split() if w not in COLORS]
    return " ".join(kept) if kept else text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/union.jsonl")
    args = p.parse_args()

    counts = {}
    with open(args.out, "w") as out:
        for source, path, cap in SOURCES:
            n_in = sum(1 for _ in _read(path)) if cap else 0
            stride = max(1, n_in // cap) if cap else 1
            kept = 0
            for i, item in enumerate(_read(path)):
                if i % stride:
                    continue
                raw = item["text"]
                if raw == "unrecognizable":
                    raw = item.get("meta", {}).get("label", "")
                text = canonical(raw)
                if source == "sketch":
                    text = strip_colors(text)
                if not text:
                    continue
                meta = item.get("meta", {})
                meta["source"] = source
                out.write(json.dumps({"text": f"{source}: {text}",
                                      "points": item["points"],
                                      "meta": meta}) + "\n")
                kept += 1
            counts[source] = counts.get(source, 0) + kept
    print(f"wrote {sum(counts.values()):,} examples to {args.out}: {counts}")


if __name__ == "__main__":
    main()
