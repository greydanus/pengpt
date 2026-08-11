import argparse
import json
import os

import numpy as np

from pengpt.data import _read


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    from pengpt.fastembed import FastEmbedder
    embedder = FastEmbedder(device=args.device, workers=1, fp16=False)
    out_path = args.out or args.dataset.split(".json")[0] + ".clipimg.npy"

    chunks, buf, n = [], [], 0
    for item in _read(args.dataset):
        buf.append(np.asarray(item["points"], dtype=float))
        if len(buf) >= args.batch * 8:
            chunks.append(embedder.embed(buf, args.batch))
            n += len(buf)
            buf = []
            print(f"  {n:,} embedded", flush=True)
    if buf:
        chunks.append(embedder.embed(buf, args.batch))
        n += len(buf)
    embs = np.concatenate(chunks).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    np.save(out_path, embs)
    print(f"wrote {embs.shape} to {out_path}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
