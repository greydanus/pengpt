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
    ("quickdraw", "data/quickdraw_balanced_fixed.jsonl.gz", 120_000),
    ("sketchy", "data/sketchy_tags.jsonl", 0),
    ("icons", "data/icons.jsonl", 0),
]


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
                text = canonical(item["text"])
                if not text:
                    continue
                meta = item.get("meta", {})
                meta["source"] = source
                out.write(json.dumps({"text": text, "points": item["points"],
                                      "meta": meta}) + "\n")
                kept += 1
            counts[source] = kept
    print(f"wrote {sum(counts.values()):,} examples to {args.out}: {counts}")


if __name__ == "__main__":
    main()
