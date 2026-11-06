import argparse
import glob
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from utils import normalize


def convert_quickdraw(path, max_items=None, categories=None):
    examples = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if categories and item.get("word") not in categories:
            continue
        points = []
        for stroke in item["drawing"]:
            xs, ys = stroke[0], stroke[1]
            for x, y in zip(xs, ys):
                points.append([float(x), float(y), 1.0])
            points.append([float(xs[-1]), float(ys[-1]), 0.0])
        if len(points) < 4:
            continue
        examples.append({
            "text": str(item.get("word", "")),
            "points": normalize(np.array(points)).round(4).tolist(),
        })
        if max_items and len(examples) >= max_items:
            break
    return examples


QD_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
QD_LIST = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"


def _get(url, dest=None):
    try:
        if dest:
            urllib.request.urlretrieve(url, dest)
            return None
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode()
    except Exception:
        import subprocess
        cmd = ["curl", "-fsSL", url] + (["-o", dest] if dest else [])
        out = subprocess.run(cmd, capture_output=not dest, check=True)
        return None if dest else out.stdout.decode()


def quickdraw_categories():
    return [line.strip() for line in _get(QD_LIST).splitlines() if line.strip()]


def fetch_category(name, out_dir, retries=3):
    path = os.path.join(out_dir, f"{name}.ndjson")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return name, os.path.getsize(path), "cached"
    url = QD_BASE + urllib.parse.quote(name) + ".ndjson"
    partial = path + ".part"
    for attempt in range(retries):
        try:
            _get(url, partial)
            os.replace(partial, path)
            return name, os.path.getsize(path), "ok"
        except Exception as e:
            if attempt == retries - 1:
                if os.path.exists(partial):
                    os.remove(partial)
                return name, 0, f"failed: {e}"
    return name, 0, "failed"


def download_command(args):
    os.makedirs(args.out_dir, exist_ok=True)
    names = ([c.strip() for c in args.categories.split(",") if c.strip()]
             or quickdraw_categories())
    print(f"{len(names)} categories -> {args.out_dir}")

    done = failed = total = 0
    with ThreadPoolExecutor(args.threads) as pool:
        for name, size, status in pool.map(lambda n: fetch_category(n, args.out_dir), names):
            done += 1
            total += size
            if status.startswith("failed"):
                failed += 1
                print(f"[{done}/{len(names)}] {name:28s} {status}", flush=True)
            elif done % 25 == 0 or done == len(names):
                print(f"[{done}/{len(names)}] {total / 1e9:5.1f} GB", flush=True)

    print(f"\n{done - failed} files, {total / 1e9:.1f} GB in {args.out_dir}")
    if failed:
        print(f"{failed} failed; run again to retry just those")


JUDGE_RUBRIC = """Order these drawings from best to worst as training examples.

Reward: the subject is recognizable, its parts are present and connected (a dog
with legs and ears, a car with wheels and windows), and the strokes are
deliberate.

Penalize, hardest first:
  - any drawing with letters or words written on it, however neat the rest is
  - scribbles, and drawings that are one line or a few disconnected fragments
  - missing essential parts, so the subject is only guessable from the label

Words on the canvas need to be called out explicitly like this. A vision
embedding ranks such drawings anywhere from the 22nd to the 76th percentile,
because a cat with "MEOW" written over it still looks largely like a cat, and a
stroke-geometry rule that catches them flags a fifth of clean drawings too. The
judge is the only reliable filter for writing, so the rubric has to ask.
"""


def render(points, px=64, linewidth=2, supersample=8, pad=2):
    from PIL import Image, ImageDraw

    points = np.asarray(points, dtype=float)
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        return Image.new("RGB", (px, px), "white")

    big = px * supersample
    margin = pad * supersample
    x0, y0 = down[:, 0].min(), down[:, 1].min()
    scale = (big - 2 * margin) / max(np.ptp(down[:, 0]), np.ptp(down[:, 1]), 1e-6)

    image = Image.new("L", (big, big), 255)
    draw = ImageDraw.Draw(image)
    for chunk in np.split(points, np.flatnonzero(points[:, 2] == 0) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            draw.line([(margin + (x - x0) * scale, margin + (y - y0) * scale)
                       for x, y in chunk[:, :2]],
                      fill=0, width=int(linewidth * supersample))
    return image.resize((px, px), Image.LANCZOS).convert("RGB")


_WORKER = {}


def _default_workers(device):
    if device != "cuda":
        return 1
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def _init_worker(name, px, linewidth):
    from transformers import AutoImageProcessor
    _WORKER["processor"] = AutoImageProcessor.from_pretrained(name)
    _WORKER["px"], _WORKER["linewidth"] = px, linewidth


def _prepare_worker(chunk):
    images = [render(p, _WORKER["px"], _WORKER["linewidth"]) for p in chunk]
    return _WORKER["processor"](images=images, return_tensors="pt")["pixel_values"]


class Embedder:

    def __init__(self, name="openai/clip-vit-base-patch32", device=None, px=64,
                 linewidth=1.0, fp16=False, workers=0):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.name = name
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.is_clip = "clip" in name.lower()
        self.fp16 = fp16 and self.device != "cpu"
        if self.fp16:
            self.model = self.model.half()
        self.px = px
        self.linewidth = linewidth
        self.workers = workers if workers else _default_workers(self.device)
        self._pool = None

    def embed(self, points_list, batch_size=256, verbose=False):
        out = []
        with self.torch.inference_mode():
            for pixels, n in self._batches(points_list, batch_size):
                pixels = pixels.to(self.device, non_blocking=True)
                if self.fp16:
                    pixels = pixels.half()
                feats = (self.model.get_image_features(pixel_values=pixels)
                         if self.is_clip else self.model(pixel_values=pixels))
                if not hasattr(feats, "float"):
                    feats = (feats.image_embeds if getattr(feats, "image_embeds", None)
                             is not None else feats.pooler_output)
                out.append(feats.float().cpu().numpy())
                if verbose:
                    print(f"  embedded {n}", flush=True)
        return np.concatenate(out) if out else np.zeros((0, 1))

    def _batches(self, points_list, batch_size):
        chunks = [points_list[i:i + batch_size]
                  for i in range(0, len(points_list), batch_size)]
        if self.workers <= 1 or len(chunks) <= 1:
            for chunk in chunks:
                yield self._prepare(chunk), len(chunk)
            return
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(
                self.workers, initializer=_init_worker,
                initargs=(self.name, self.px, self.linewidth))
        for pixels in self._pool.map(_prepare_worker, chunks, chunksize=1):
            yield pixels, len(pixels)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _prepare(self, chunk):
        images = [render(p, self.px, self.linewidth) for p in chunk]
        return self.processor(images=images, return_tensors="pt")["pixel_values"]


def bradley_terry(n_items, comparisons, iters=300, reg=1e-2, lr=0.1):
    wins = np.zeros(n_items)
    played = np.zeros((n_items, n_items))
    for w, l in comparisons:
        wins[w] += 1
        played[w, l] += 1
        played[l, w] += 1

    scores = np.zeros(n_items)
    for _ in range(iters):
        expected = np.zeros(n_items)
        for i in np.flatnonzero(played.any(1)):
            partners = np.flatnonzero(played[i])
            p = 1.0 / (1.0 + np.exp(np.clip(scores[partners] - scores[i], -30, 30)))
            expected[i] = (played[i, partners] * p).sum()
        scores += lr * (wins - expected - reg * scores)
    return (scores - scores.mean()) / (scores.std() + 1e-9)


class LinearProbe:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = self.mean = self.std = None

    def fit(self, X, scores):
        X = np.asarray(X, dtype=float)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.weights = np.linalg.solve(A, Z.T @ np.asarray(scores, dtype=float))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        return Z @ self.weights

    def agreement(self, X, scores):
        return spearman(self.score(X), np.asarray(scores))


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_probe(path="data/quickdraw_probe.npz"):
    data = np.load(path)
    probe = LinearProbe(alpha=float(data["alpha"]))
    probe.weights, probe.mean, probe.std = data["weights"], data["mean"], data["std"]
    return probe


def select_per_class(scores, labels, fraction=0.25):
    scores, labels = np.asarray(scores), np.asarray(labels)
    keep = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        order = idx[np.argsort(scores[idx])[::-1]]
        keep.append(order[:max(1, int(round(fraction * len(order))))])
    return np.sort(np.concatenate(keep))


RANK_CHUNK = 20_000


def iter_quickdraw(path, limit=None):
    label = os.path.basename(path).split(".")[0]
    seen = 0
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.endswith("}"):
                continue
            try:
                drawing = json.loads(line)["drawing"]
            except (ValueError, KeyError):
                continue
            points = []
            for stroke in drawing:
                xs, ys = stroke[0], stroke[1]
                if not xs:
                    continue
                points.extend([float(x), float(y), 1.0] for x, y in zip(xs, ys))
                points.append([float(xs[-1]), float(ys[-1]), 0.0])
            if len(points) >= 4:
                yield np.array(points), label
                seen += 1
                if limit and seen >= limit:
                    return


def score_category(path, embedder, probe, fraction, limit, batch_size):
    kept, seen, chunk = [], 0, []
    label = None

    def flush(chunk):
        if not chunk:
            return []
        scores = probe.score(embedder.embed(chunk, batch_size=batch_size))
        n = max(1, int(round(fraction * len(chunk))))
        return [chunk[i] for i in np.argsort(scores)[::-1][:n]]

    for points, label in iter_quickdraw(path, limit):
        chunk.append(points)
        seen += 1
        if len(chunk) >= RANK_CHUNK:
            kept.extend(flush(chunk))
            chunk = []
    kept.extend(flush(chunk))
    examples = [{"text": label, "points": normalize(p).round(4).tolist()}
                for p in kept]
    return examples, seen


def rank_command(args):
    settings = np.load(args.probe)
    embedder = Embedder(name=str(settings["embedder"]), px=int(settings["px"]),
                        linewidth=float(settings["linewidth"]), fp16=args.fp16,
                        workers=args.workers)
    print(f"{embedder.device}, fp16={embedder.fp16}, {embedder.workers} render workers")
    probe = load_probe(args.probe)

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.ndjson")))
    if not paths:
        raise SystemExit(f"no .ndjson files in {args.raw_dir}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out):
        if args.overwrite:
            open(args.out, "w").close()
        elif args.resume:
            with open(args.out) as f:
                done = {json.loads(line)["text"] for line in f if line.strip()}
            print(f"resuming: {len(done)} categories already done")
        else:
            raise SystemExit(
                f"{args.out} already exists. Pass --resume to continue it, or "
                f"--overwrite to start again; appending to it silently would "
                f"duplicate every category it already holds.")

    todo = [p for p in paths
            if os.path.basename(p).split(".")[0] not in done]
    print(f"{len(todo)} of {len(paths)} categories to score, "
          f"keeping the best {args.fraction:.0%} of each")

    kept_total, seen_total, t0 = 0, 0, time.time()
    with open(args.out, "a") as out:
        for n, path in enumerate(todo, 1):
            label = os.path.basename(path).split(".")[0]
            examples, seen = score_category(path, embedder, probe, args.fraction,
                                            args.limit_per_class, args.batch_size)
            for example in examples:
                out.write(json.dumps(example) + "\n")
            out.flush()
            os.fsync(out.fileno())
            kept_total += len(examples)
            seen_total += seen
            elapsed = time.time() - t0
            rate = seen_total / elapsed
            eta = (len(todo) - n) * (seen_total / n) / rate / 60
            print(f"[{n}/{len(todo)}] {label:22s} {seen:7,} -> {len(examples):6,}"
                  f" | {rate:5.0f}/s | {elapsed / 60:5.1f} min elapsed"
                  f" | eta {eta:5.0f} min", flush=True)

    print(f"\nkept {kept_total:,} of {seen_total:,} scored "
          f"({kept_total / max(seen_total, 1):.0%}) in "
          f"{(time.time() - t0) / 60:.1f} min")
    if done:
        print(f"plus {len(done)} categories from an earlier run")
    print(f"-> {args.out}")
    embedder.close()


def convert_command(args):
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    examples = convert_quickdraw(args.quickdraw, args.max_items, cats or None)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(examples, f)
    print(f"Wrote {len(examples)} examples to {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quick, Draw! corpus tools")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download the Quick, Draw! corpus")
    dl.add_argument("--out_dir", type=str, default="qd_raw")
    dl.add_argument("--categories", type=str, default="",
                    help="comma-separated subset; default is all 345")
    dl.add_argument("--threads", type=int, default=16)

    cv = sub.add_parser("convert", help="convert a Quick, Draw! ndjson to pengpt format")
    cv.add_argument("--quickdraw", type=str, required=True,
                    help="Quick, Draw! simplified .ndjson (one category per file)")
    cv.add_argument("--categories", type=str, default="",
                    help="comma-separated categories to keep")
    cv.add_argument("--max_items", type=int, default=None)
    cv.add_argument("--out", type=str, default="data/converted.json")

    rk = sub.add_parser("rank", help="filter Quick, Draw! to its best drawings")
    rk.add_argument("--raw_dir", type=str, required=True,
                    help="directory of Quick, Draw! simplified .ndjson files")
    rk.add_argument("--out", type=str, default="data/quickdraw_top25.jsonl")
    rk.add_argument("--fraction", type=float, default=0.25)
    rk.add_argument("--limit_per_class", type=int, default=None)
    rk.add_argument("--probe", type=str, default="data/quickdraw_probe.npz")
    rk.add_argument("--batch_size", type=int, default=256)
    rk.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                    help="half precision embedding; 1.7x faster, same ranking")
    rk.add_argument("--workers", type=int, default=0,
                    help="processes rendering batches ahead of the GPU; "
                         "0 picks a default from the device")
    rk.add_argument("--resume", action="store_true",
                    help="skip categories already present in the output")
    rk.add_argument("--overwrite", action="store_true",
                    help="discard an existing output file and start again")

    args = parser.parse_args(argv)
    {"download": download_command, "convert": convert_command,
     "rank": rank_command}[args.command](args)


if __name__ == "__main__":
    main()
