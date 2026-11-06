import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage

from tools.ink import file_hash, flatten, ink_context, ink_mask, join_strokes, judge, read_jsonl, write_jsonl

TEXT, PLAIN = "Derender the ink: {text}", "Recognize and derender."
ROUNDS = [[("mono", PLAIN)], [("raw", PLAIN), ("clean", TEXT), ("mono", TEXT)]]
MAX_SIDE, MAX_HEIGHT = 1024, 320


def shrink(img):
    s = min(1.0, MAX_SIDE / max(img.size), MAX_HEIGHT / img.height)
    return (img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS) if s < 1 else img), s


def variants(img):
    from skimage.morphology import skeletonize
    norm = flatten(img)
    if norm.min() > 0.3:
        norm = (norm - norm.min()) / (1 - norm.min())
    mask = ink_mask(norm)
    if not mask.any():
        raise ValueError("no ink")
    lo = np.percentile(norm[mask], 10)
    clean = np.where(mask, ((norm - lo) / max(0.75 - lo, 1e-3)).clip(0, 1), 1.0)
    width = np.median(ndimage.distance_transform_edt(mask)[mask])
    mono = ndimage.distance_transform_edt(~skeletonize(mask)) <= max(1.5, width)
    return {"raw": np.asarray(img), "clean": (clean * 255).astype(np.uint8), "mono": np.where(mono, 0, 255).astype(np.uint8)}


def prepare(row):
    try:
        img, scale = shrink(Image.open(row["image"]).convert("RGB"))
        return {"row": row, "scale": scale, "variants": variants(img)}
    except Exception as e:
        return {"row": row, "error": f"prepare: {type(e).__name__}: {e}"}


def trace_round(model, preps, cands, batch):
    out = [[] for _ in preps]
    for variant, prompt in cands:
        for i in range(0, len(preps), batch):
            chunk = preps[i:i + batch]
            images = [Image.fromarray(p["variants"][variant]).convert("RGB") for p in chunk]
            for k, (strokes, read) in enumerate(model.trace(images, [p["row"]["text"] for p in chunk], prompt)):
                out[i + k].append((variant, prompt, strokes, read))
    return out


def pick(job):
    clean, text, cands, best = job
    try:
        ctx = ink_context(clean)
        for variant, prompt, strokes, read in cands:
            joined = join_strokes([s for s in strokes if len(s)], ctx)
            c, q = judge({"text": text}, {"strokes": joined, "read": read, "prompt": prompt}, ctx=ctx)
            key = (q["passed"], q["coverage"] * q["precision"])
            if best is None or key > tuple(best["key"]):
                best = {"key": key, "candidate": f"{variant}/{'text' if prompt == TEXT else 'plain'}", "strokes": c["strokes"],
                        "qc": {k: v for k, v in q.items() if k != "prompt"}}
        return best
    except Exception as e:
        return best or {"error": f"pick: {type(e).__name__}: {e}"}


def harvest(manifest, out, weights="inksight", batch=128, workers=os.cpu_count() or 4):
    os.makedirs(out, exist_ok=True)
    acc, rej = f"{out}/accepted.jsonl", f"{out}/rejected.jsonl"
    done = {r["hash"] for path in (acc, rej) for r in read_jsonl(path) if "hash" in r}
    rows = [{**r, "hash": file_hash(r["image"])} for r in read_jsonl(manifest)]
    rows = list({r["hash"]: r for r in rows if r["hash"] not in done}.values())
    chunks = [rows[i:i + batch] for i in range(0, len(rows), batch)]
    print(f"{len(rows)} new images in {len(chunks)} chunks ({len(done)} already done)")
    with Pool(workers) as pool:
        from tools.inksight import InkSight
        model = InkSight(weights)
        t0, n, n_ok, traced = time.time(), 0, 0, 0
        pending = pool.map_async(prepare, chunks[0]) if chunks else None
        for k in range(len(chunks)):
            preps = pending.get()
            if k + 1 < len(chunks):
                pending = pool.map_async(prepare, chunks[k + 1])
            ok = [p for p in preps if "error" not in p]
            best, todo = [None] * len(ok), list(range(len(ok)))
            for cands in ROUNDS:
                if not todo:
                    break
                found = trace_round(model, [ok[i] for i in todo], cands, batch)
                traced += len(todo) * len(cands)
                picks = pool.map(pick, [(ok[i]["variants"]["clean"], ok[i]["row"]["text"], f, best[i]) for i, f in zip(todo, found)])
                for i, b in zip(todo, picks):
                    best[i] = b
                todo = [i for i in todo if not (best[i] and best[i].get("key", (False,))[0])]
            accepted, rejected = [], [{**p["row"], "qc": {"reason": p["error"]}} for p in preps if "error" in p]
            for p, b in zip(ok, best):
                if b and "key" in b and b["key"][0]:
                    accepted.append({**p["row"], "strokes": [(np.asarray(s, float) / p["scale"]).round(2).tolist() for s in b["strokes"]],
                                     "candidate": b["candidate"], "qc": b["qc"]})
                else:
                    rejected.append({**p["row"], "qc": (b or {}).get("qc") or {"reason": (b or {}).get("error", "no trace")},
                                     "candidate": (b or {}).get("candidate")})
            write_jsonl(acc, accepted, "a")
            write_jsonl(rej, rejected, "a")
            n, n_ok = n + len(preps), n_ok + len(accepted)
            dt = time.time() - t0
            print(f"{n}/{len(rows)} words | {n_ok / n:.0%} accepted | {traced / n:.2f} traces/word | {dt / n * 1000:.0f} ms/word", flush=True)


def main():
    p = argparse.ArgumentParser(description="word images -> QC-passing pen strokes: image variants, adaptive best-of-N "
                                            "InkSight traces, QC, pen-lift joining; streaming and resumable")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inksight", default="inksight")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()
    harvest(args.manifest, args.out, args.inksight, args.batch, args.workers)


if __name__ == "__main__":
    main()
