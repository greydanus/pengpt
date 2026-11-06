import argparse
import hashlib
import json
import os
import random
import re
import string
import time
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

SIZE = 224
UP = 4
INK_LEVEL = 200
MIN_ASPECT = float(os.environ.get("INK_MIN_ASPECT", 2.5))


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def write_jsonl(path, rows, mode="w"):
    with open(path, mode) as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def replace_rows(path, rows, source):
    write_jsonl(path, [r for r in read_jsonl(path) if r["source"] != source] + rows)


def file_hash(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]


def flatten(rgb):
    red = np.asarray(rgb.convert("RGB"))[..., 0].astype(float)
    return (red / np.maximum(ndimage.grey_closing(red, size=(31, 31)), 1)).clip(0, 1)


def ink_mask(norm, weak=0.80, strong=0.55):
    labels, _ = ndimage.label(norm < weak)
    return np.isin(labels, np.unique(labels[norm < strong])) & (labels > 0)


def tone(norm):
    t = ((norm - 0.35) / (0.92 - 0.35)).clip(0, 1) ** 1.4
    return Image.fromarray((ndimage.gaussian_filter(t, 0.7) * 255).astype(np.uint8)).convert("RGB")


def upsample(norm):
    h, w = norm.shape
    big = Image.fromarray((norm * 65535).astype(np.uint16)).resize((w * UP, h * UP), Image.BICUBIC)
    return np.asarray(big).astype(float) / 65535


LETTER_WEIGHTS = np.maximum([6, 1.5, 2.8, 4.3, 6, 2.2, 2, 6.1, 6, 4, 0.77, 4, 2.4, 6, 6, 1.9, 0.095, 6, 6.3, 6, 2.8,
                             0.98, 2.4, 4, 2, 0.074], 4)


def synth_words(n, seed=0):
    rng = np.random.default_rng(seed)
    lengths = np.arange(3, 11)
    p = np.maximum(1 - 0.85 * (lengths - 3) / 7, 0)
    p[0] *= 0.5
    p[1] /= 0.8
    p /= p.sum()
    letters = LETTER_WEIGHTS / LETTER_WEIGHTS.sum()
    words = []
    for _ in range(n):
        if rng.random() < 0.25:
            word = "".join(rng.choice(list(string.digits), rng.integers(1, 6)))
            if rng.random() < 0.3:
                i = rng.integers(0, len(word) + 1)
                word = word[:i] + "." + word[i:]
        else:
            word = "".join(rng.choice(list(string.ascii_lowercase), rng.choice(lengths, p=p), p=letters))
            word = word.upper() if rng.random() < 0.25 else word.capitalize() if rng.random() < 0.25 else word
        if rng.random() < 0.1:
            quote = rng.choice(['"', "'"])
            if quote == "'" and rng.random() < 0.5 and len(word) > 1:
                i = rng.integers(1, len(word))
                word = word[:i] + "'" + word[i:]
            else:
                word = quote + word if rng.random() < 0.5 else word + quote
        if rng.random() < 0.1:
            word = "(" + word if rng.random() < 0.5 else word + ")"
        if rng.random() < 0.8:
            for mark in ",.?!":
                if rng.random() < 0.05:
                    word += mark
                    break
        words.append(word)
    return words


def words_command(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    words = synth_words(args.n, args.seed)
    open(args.out, "w").write("\n".join(words) + "\n")
    print(f"wrote {len(words)} words to {args.out}: {' '.join(words[:12])} ...")


def ink_baseline(font):
    bottoms = []
    for c in "acemnorsuvwxz":
        img = Image.new("L", (4 * font.size, 4 * font.size), 255)
        ImageDraw.Draw(img).text((font.size, 3 * font.size), c, font=font, fill=0, anchor="ls")
        rows = np.nonzero(np.asarray(img) < INK_LEVEL)[0]
        if len(rows):
            bottoms.append(rows.max() - 3 * font.size)
    return int(round(np.median(bottoms)))


def render_command(args):
    font = ImageFont.truetype(args.font, args.size)
    shift = ink_baseline(font)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i, text in enumerate(w.strip() for w in open(args.words) if w.strip()):
        left, top, right, bottom = font.getbbox(text, anchor="ls")
        img = Image.new("RGB", (right - left + 16, bottom - top + 16), "white")
        ImageDraw.Draw(img).text((8 - left, 8 - top), text, font=font, fill="black", anchor="ls")
        path = f"{args.out}/{os.path.splitext(os.path.basename(args.font))[0]}_{i:05d}.png"
        img.save(path)
        rows.append({"image": path, "text": text, "baseline": 8 - top + shift, "style": args.style, "writer": args.style,
                     "source": os.path.basename(args.font), "license": args.license})
    replace_rows(f"{args.out}/manifest.jsonl", rows, os.path.basename(args.font))
    print(f"rendered {len(rows)} words to {args.out}/manifest.jsonl")


def split_line(ink_row, tokens, h):
    cols = ndimage.binary_dilation(ink_row.any(0), iterations=3)
    edges = np.flatnonzero(np.diff(np.r_[0, cols.astype(int), 0]))
    segs = []
    for seg in zip(edges[::2], edges[1::2]):
        if segs and (seg[1] - seg[0] < 0.3 * h or segs[-1][1] - segs[-1][0] < 0.3 * h):
            segs[-1] = (segs[-1][0], seg[1])
        else:
            segs.append(seg)
    k = len(tokens) - 1
    gaps = np.array([segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)])
    if not segs or k > len(gaps):
        return None
    cuts = np.sort(np.argsort(gaps)[::-1][:k]) if k else np.array([], int)
    rest = np.delete(gaps, cuts)
    if k and len(rest) and gaps[cuts].min() < 1.3 * rest.max():
        return None
    starts, ends = [0] + list(cuts + 1), list(cuts) + [len(segs) - 1]
    words = [(segs[i][0], segs[j][1]) for i, j in zip(starts, ends)]
    per_char = np.array([(e - s) / max(1, len(t.strip(".,;:=-"))) for (s, e), t in zip(words, tokens)])
    if len(words) > 1 and not (0.5 < (per_char / np.median(per_char))).all() & ((per_char / np.median(per_char)) < 2).all():
        return None
    return words


def shear(rows, center, slant, margin):
    out = np.zeros((rows.shape[0], rows.shape[1] + 2 * margin), rows.dtype)
    for i, row in enumerate(rows):
        shift = margin + int(round((i - center) * slant))
        out[i, shift:shift + rows.shape[1]] = row
    return out


def estimate_slant(ink, bands):
    def sharpness(t):
        m = int(np.ceil(abs(t) * ink.shape[0])) + 1
        return sum((shear(ink[a:b], (b - a) / 2, t, m).sum(0).astype(float) ** 2).sum() for a, b in bands)
    return max(np.linspace(-1.2, 1.2, 49), key=sharpness)


def segment_command(args):
    page = Image.open(args.page)
    norm = flatten(page)
    ink = ink_mask(norm)
    lines = [l.strip() for l in open(args.transcript) if l.strip()]
    from scipy.signal import find_peaks
    profile = ndimage.uniform_filter1d(ink.sum(1).astype(float), 9)
    peaks, props = find_peaks(profile, distance=max(5, ink.shape[0] // (2 * len(lines))), prominence=0)
    if len(peaks) < len(lines):
        raise SystemExit(f"found {len(peaks)} text lines but the transcript has {len(lines)}")
    peaks = np.sort(peaks[np.argsort(props["prominences"])[::-1][:len(lines)]])
    q = max(4, int(0.2 * (np.median(np.diff(peaks)) if len(peaks) > 1 else ink.shape[0])))
    bands = [(max(0, p - q), min(ink.shape[0], p + q)) for p in peaks]
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.page))[0]
    faint = ink_mask(norm, weak=0.90) & ndimage.binary_dilation(norm < 0.75, iterations=4)
    slant = estimate_slant(np.where(ink, 1 - norm, 0), bands)
    print(f"estimated slant: {np.degrees(np.arctan(slant)):.0f} degrees from vertical")
    ys_all, xs_all = np.mgrid[0:ink.shape[0], 0:ink.shape[1]]
    margin = int(np.ceil(abs(slant) * ink.shape[0])) + 1
    rows, skipped, boxes, cut_words = [], [], [], []
    for li, ((a, b), text) in enumerate(zip(bands, lines)):
        continued = li > 0 and lines[li - 1].endswith(("=", "-"))
        h = b - a
        tokens = text.split()
        center = (a + b) / 2
        alone = faint.copy()
        for a2, b2 in bands:
            if (a2, b2) != (a, b):
                alone[a2:b2] = False
        labels, _ = ndimage.label(alone, structure=np.ones((3, 3)))
        sheared_x = xs_all + np.rint((ys_all - center) * slant).astype(int) + margin
        ids = np.arange(1, labels.max() + 1)
        lo, hi = ndimage.minimum(sheared_x, labels, ids), ndimage.maximum(sheared_x, labels, ids)
        cy = np.asarray(ndimage.center_of_mass(alone, labels, ids)).reshape(-1, 2)[:, 0]
        words = split_line(shear(ink[a:b], center - a, slant, margin), tokens, h)
        if words is None:
            skipped.append(text)
            continue
        for wi, ((s, e), token) in enumerate(zip(words, tokens)):
            if token.endswith(("=", "-")) or (continued and wi == 0) or not token.strip(".,;:!?\"'"):
                continue
            pad = h // 4
            core = labels[a + h // 4:b - h // 4] * ((sheared_x[a + h // 4:b - h // 4] >= s) & (sheared_x[a + h // 4:b - h // 4] < e))
            left = (words[wi - 1][1] + s) // 2 if wi else s - pad
            right = (e + words[wi + 1][0]) // 2 if wi + 1 < len(words) else e + pad
            marks = ids[(lo >= left) & (hi < right) & (cy >= a - h) & (cy <= b + h)]
            mine = np.union1d(np.unique(core[core > 0]), marks)
            full = np.isin(labels, mine)
            keep = full & (sheared_x >= left) & (sheared_x < right)
            if not keep.any():
                continue
            if (full & ~keep).sum() > 0.1 * full.sum():
                cut_words.append(token)
                continue
            ys, xs = np.nonzero(keep)
            y0, y1 = max(0, ys.min() - pad), min(ink.shape[0], ys.max() + pad)
            x0, x1 = max(0, xs.min() - pad), min(ink.shape[1], xs.max() + pad)
            near = ndimage.binary_dilation(keep[y0:y1, x0:x1], iterations=2)
            word = np.where(near, norm[y0:y1, x0:x1], 1.0)
            boxes.append((x0, y0, x1, y1, token))
            n = upsample(word)
            path = f"{args.out}/{stem}_{li:02d}_{wi:02d}.png"
            tone(n).save(path)
            rows.append({"image": path, "text": token, "baseline": int((b - y0) * UP), "style": args.style,
                         "writer": args.style,
                         "source": os.path.basename(args.page), "license": args.license})
    replace_rows(f"{args.out}/manifest.jsonl", rows, os.path.basename(args.page))
    print(f"{stem}: {len(rows)} words from {len(lines) - len(skipped)}/{len(lines)} lines")
    if args.overview:
        overview(page, boxes, bands, lines, skipped, args.overview)
    for text in skipped:
        print(f"  skipped (could not split confidently): {text}")
    if cut_words:
        print(f"  skipped {len(cut_words)} words whose ink runs past the word boundary: {' '.join(cut_words)}")


def overview(page, boxes, bands, lines, skipped, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, h = page.size
    fig, ax = plt.subplots(figsize=(12, 12 * h / w))
    ax.imshow(page.convert("L"), cmap="gray")
    for x0, y0, x1, y1, token in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="tab:blue", lw=1))
        ax.text(x0, y0, token, color="tab:blue", fontsize=7, va="bottom")
    for (a, b), text in zip(bands, lines):
        if text in skipped:
            ax.text(2, (a + b) / 2, "SKIPPED LINE", fontsize=7, va="center",
                    bbox=dict(facecolor="white", edgecolor="black", lw=0.5))
    ax.axis("off")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"overview: {path}")


def scale_and_pad(img, min_aspect=MIN_ASPECT):
    pad = max(0, int(min_aspect * img.height) - img.width) // 2
    if pad:
        wide = Image.new("RGB", (img.width + 2 * pad, img.height), "white")
        wide.paste(img, (pad, 0))
        img = wide
    ratio = min(SIZE / img.width, SIZE / img.height)
    small = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    canvas = Image.new("RGB", (SIZE, SIZE), "black")
    dx, dy = (SIZE - small.width) // 2, (SIZE - small.height) // 2
    canvas.paste(small, (dx, dy))
    return canvas, ratio, dx + pad * ratio, dy


def detokenize(text):
    tokens = [int(t) for t in re.findall(r"<ink_token_(\d+)>", text)]
    strokes, stroke, i, start = [], [], 0, 2 * (SIZE + 1)
    while i < len(tokens):
        if tokens[i] == start:
            if stroke:
                strokes.append(stroke)
            stroke, i = [], i + 1
        elif i + 1 < len(tokens) and tokens[i + 1] != start:
            x, y = tokens[i], tokens[i + 1] - (SIZE + 1)
            if 0 <= x <= SIZE and 0 <= y <= SIZE:
                stroke.append((x, y))
            i += 2
        else:
            i += 1
    if stroke:
        strokes.append(stroke)
    return strokes


def on_ink(strokes, img, radius=3):
    ink = ndimage.binary_dilation(np.asarray(img.convert("L")) < INK_LEVEL, iterations=radius)
    h, w = ink.shape
    touches = lambda s: any(ink[min(max(int(round(y)), 0), h - 1), min(max(int(round(x)), 0), w - 1)] for x, y in s)
    return [s for s in strokes if touches(s)]


def export_command(args):
    import tensorflow as tf
    import tensorflow_text
    from huggingface_hub import snapshot_download
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    meta = tf.compat.v1.saved_model.loader.load(sess, ["serve"], snapshot_download("Derendering/InkSight-Small-p"))
    ops = [op for op in sess.graph.get_operations() if op.type == "VarHandleOp"]
    vals = sess.run([tf.raw_ops.ReadVariableOp(resource=op.outputs[0], dtype=op.get_attr("dtype")) for op in ops])
    os.makedirs(args.out, exist_ok=True)
    np.savez(f"{args.out}/weights.npz", **{op.get_attr("shared_name").decode(): v.astype(np.float32) for op, v in zip(ops, vals)})
    spm = next(n.attr["model"].s for f in meta.graph_def.library.function for n in f.node_def if n.op == "SentencepieceOp")
    open(f"{args.out}/spm.model", "wb").write(spm)
    print(f"exported {len(ops)} InkSight weights and its tokenizer to {args.out}")


def tf_tracer():
    import tensorflow as tf
    import tensorflow_text
    from huggingface_hub import from_pretrained_keras
    serve = from_pretrained_keras("Derendering/InkSight-Small-p").signatures["serving_default"]

    def trace(images, texts, prompt):
        padded = [scale_and_pad(img) for img in images]
        encoded = tf.stack([tf.reshape(tf.io.encode_jpeg(np.asarray(c)), (1,)) for c, _, _, _ in padded])
        out = serve(**{"input_text": tf.constant([prompt.replace("{text}", t) for t in texts]), "image/encoded": encoded})
        rows = []
        for o, (_, ratio, dx, dy) in zip(out["output_0"].numpy(), padded):
            raw = o[0].decode()
            rows.append(([[[(x - dx) / ratio, (y - dy) / ratio] for x, y in s] for s in detokenize(raw)],
                         re.sub(r"<[^>]+>", "", raw).strip()))
        return rows
    return trace


def torch_tracer(path):
    from tools.inksight import InkSight
    model = InkSight(path)
    return lambda images, texts, prompt: model.trace(images, texts, prompt)


def trace_command(args):
    trace = torch_tracer(args.inksight) if args.engine == "torch" else tf_tracer()
    items = list({it["hash"]: it for it in ({**it, "hash": file_hash(it["image"])} for it in read_jsonl(args.manifest))}.values())
    done = {(r.get("hash"), r["prompt"]) for r in read_jsonl(args.out)}
    t0, n = time.time(), 0
    for prompt in args.prompts:
        todo = sorted((it for it in items if (it["hash"], prompt) not in done), key=lambda it: len(it["text"]))
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            out = trace([Image.open(it["image"]).convert("RGB") for it in chunk], [it["text"] for it in chunk], prompt)
            write_jsonl(args.out, [{"image": it["image"], "hash": it["hash"], "prompt": prompt, "strokes": strokes,
                                    "read": read, "engine": args.engine} for it, (strokes, read) in zip(chunk, out)], "a")
            n += len(chunk)
            print(f"[{prompt}] {i + len(chunk)}/{len(todo)} | {(time.time() - t0) / n:.2f} s/word", flush=True)


def levenshtein(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def resample_path(strokes):
    pts = []
    for s in map(lambda s: np.asarray(s, float), strokes):
        if len(s) == 1:
            pts.append(s)
            continue
        seg = np.diff(s, axis=0)
        n = np.maximum(1, np.ceil(np.hypot(*seg.T)).astype(int))
        k = np.repeat(np.arange(len(seg)), n)
        t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / n[k]
        pts.append(s[k] + seg[k] * t[:, None])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def score(item, cand):
    gray = np.asarray(Image.open(item["image"]).convert("L"))
    ink = gray < INK_LEVEL
    pts = resample_path(cand["strokes"])
    if not ink.any() or not len(pts):
        return {"coverage": 0.0, "precision": 0.0, "length_ratio": 0.0, "min_mark": 0.0, "max_mark_length": 0.0, "retrace": 0.0, "points": 0}
    half = max(1.0, 2 * np.median(ndimage.distance_transform_edt(ink)[ink]) - 1)
    canvas = np.zeros(ink.shape, bool)
    ij = np.clip(np.rint(pts).astype(int), 0, [ink.shape[1] - 1, ink.shape[0] - 1])
    canvas[ij[:, 1], ij[:, 0]] = True
    path = ndimage.distance_transform_edt(~canvas) <= half + 1
    near = ndimage.distance_transform_edt(~ink) <= half / 2 + 2
    labels, k = ndimage.label(ink, structure=np.ones((3, 3)))
    ids = np.arange(1, k + 1)
    area = ndimage.sum(ink, labels, ids)
    marks = ndimage.mean(path, labels, ids)[area >= half ** 2]
    gap, (ni, nj) = ndimage.distance_transform_edt(~ink, return_indices=True)
    on = gap[ij[:, 1], ij[:, 0]] <= half / 2 + 2
    owner = labels[ni[ij[on, 1], ij[on, 0]], nj[ij[on, 1], ij[on, 0]]]
    drawn = np.bincount(owner, minlength=k + 1)[1:] * 2 * half / np.maximum(area, 1)
    pairs = cKDTree(pts).query_pairs(half, output_type="ndarray")
    again = np.zeros(len(pts), bool)
    again[pairs[pairs[:, 1] - pairs[:, 0] > 6 * half, 1]] = True
    big = (area >= 8 * half ** 2) & (np.bincount(owner, minlength=k + 1)[1:] > 0)
    retrace = np.bincount(owner, again[on], minlength=k + 1)[1:] / np.maximum(np.bincount(owner, minlength=k + 1)[1:], 1)
    return {"coverage": float((ink & path).sum() / ink.sum()),
            "min_mark": float(marks.min()) if len(marks) else 1.0,
            "max_mark_length": float(drawn[area >= 8 * half ** 2].max()) if (area >= 8 * half ** 2).any() else 0.0,
            "retrace": float(retrace[big].max()) if big.any() else 0.0,
            "precision": float(near[ij[:, 1], ij[:, 0]].mean()),
            "length_ratio": float(len(pts) * 2 * half / ink.sum()),
            "points": int(len(pts))}


def plain(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return text.lower().replace("0", "o")


def judge(item, cand, args):
    c = {**cand, "strokes": on_ink(cand["strokes"], Image.open(item["image"]))}
    q = score(item, c)
    label = plain(item["text"])
    q["read"] = c["read"]
    q["read_error"] = levenshtein(plain(c["read"]), label) / len(label) if c["read"] else None
    q["prompt"] = c["prompt"]
    q["passed"] = (q["coverage"] >= args.min_coverage and q["precision"] >= args.min_precision
                   and q["min_mark"] >= args.min_mark and q["length_ratio"] <= args.max_length_ratio
                   and q["retrace"] <= args.max_retrace and q["max_mark_length"] <= args.max_mark_length
                   and (q["read_error"] is None or q["read_error"] <= args.max_read_error))
    return c, q


def qc_command(args):
    items = {it["image"]: it for it in read_jsonl(args.manifest)}
    cands = {}
    for c in read_jsonl(args.traces):
        cands.setdefault(c.get("hash"), []).append(c)
    accepted, rejected = [], []
    for image, item in items.items():
        best = None
        for c in cands.get(file_hash(image), []):
            c, q = judge(item, c, args)
            key = (q["passed"], q["coverage"] * q["precision"])
            if best is None or key > best[0]:
                best = (key, q, c)
        if best is None:
            rejected.append({**item, "qc": {"reason": "no trace"}})
            continue
        _, q, c = best
        row = {**item, "strokes": c["strokes"], "qc": q}
        (accepted if q["passed"] else rejected).append(row)
    write_jsonl(args.out, accepted)
    write_jsonl(os.path.splitext(args.out)[0] + "_rejected.jsonl", rejected)
    write_jsonl(os.path.splitext(args.out)[0] + "_retry.jsonl",
                [{k: r[k] for k in ("image", "text", "baseline")} for r in rejected if "reason" not in r["qc"]])
    print(f"accepted {len(accepted)} / {len(items)} ({100 * len(accepted) / max(1, len(items)):.0f}%), "
          f"rejected {len(rejected)}")
    if args.sheet:
        review_sheet(accepted, rejected, args.sheet)


def peek_command(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = {file_hash(it["image"]): it for it in read_jsonl(args.manifest)}
    rows = [r for r in read_jsonl(args.traces) if r.get("hash") in items and r["prompt"].startswith(args.prompt)]
    if args.worst:
        sign = 1 if args.worst in ("coverage", "precision", "min_mark") else -1
        rows = sorted(rows, key=lambda r: sign * judge(items[r["hash"]], r, args)[1][args.worst])[:args.n]
    else:
        rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 1.5 * len(rows) + 0.3), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1]})
    for r, (a, b) in zip(rows, axes):
        item = items[r["hash"]]
        c, q = judge(item, r, args)
        kept = c["strokes"]
        a.imshow(Image.open(item["image"]).convert("L"), cmap="gray", alpha=0.3)
        for s in (s for s in r["strokes"] if s not in kept):
            s = np.asarray(s)
            a.plot(s[:, 0], s[:, 1], "r--", lw=1)
        for s, color in zip(kept, plt.cm.viridis(np.linspace(0, 0.9, max(1, len(kept))))):
            s = np.asarray(s)
            a.plot(s[:, 0], s[:, 1], color=color, lw=1.6)
            a.plot(s[0, 0], s[0, 1], "o", color=color, ms=3.5)
            b.plot(s[:, 0], -s[:, 1], color="k", lw=1.6, solid_capstyle="round")
        b.set_aspect("equal")
        for ax in (a, b):
            ax.axis("off")
        a.set_title(f"{'PASS' if q['passed'] else 'FAIL'}  {item['text']!r}  read {r['read']!r}", loc="left", fontsize=9,
                    color="darkgreen" if q["passed"] else "firebrick")
        b.set_title(f"{len(kept)} strokes  cov {q['coverage']:.2f}  mark {q['min_mark']:.2f}  prec {q['precision']:.2f}  "
                    f"len {q['length_ratio']:.2f}  retrace {q['retrace']:.2f}  {r.get('engine', 'tf')}", loc="left", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, hspace=0.5, wspace=0.05)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"peek: {len(rows)} random traces -> {args.out}")


def review_sheet(accepted, rejected, path, per_page=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("ACCEPTED", r) for r in accepted[:per_page]] + [("REJECTED", r) for r in rejected[:per_page]]
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 1.1 * len(rows) + 0.5), squeeze=False)
    for (verdict, r), (a, b) in zip(rows, axes):
        img = Image.open(r["image"])
        a.imshow(img)
        q = r["qc"]
        a.set_title(f"{verdict}  {r['text']!r}", loc="left", fontsize=8, pad=2)
        b.imshow(img, alpha=0.2)
        for s in r.get("strokes", []):
            s = np.asarray(s)
            if len(s):
                b.plot(s[:, 0], s[:, 1], color="black", lw=1.2)
        detail = (f"cov {q.get('coverage', 0):.2f}  mark {q.get('min_mark', 0):.2f}  prec {q.get('precision', 0):.2f}  "
                  f"len {q.get('length_ratio', 0):.2f}  "
                  f"read {q.get('read', '')!r}" if "coverage" in q else q.get("reason", ""))
        b.set_title(detail, loc="left", fontsize=7, pad=2)
        for ax in (a, b):
            ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.6, wspace=0.05)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    print(f"review sheet: {path}")


def main():
    parser = argparse.ArgumentParser(description="handwriting images to pen strokes: render or segment, "
                                                 "trace with InkSight, quality-check")
    sub = parser.add_subparsers(dest="command", required=True)
    meta = argparse.ArgumentParser(add_help=False)
    meta.add_argument("--style", required=True)
    meta.add_argument("--license", required=True)
    p = sub.add_parser("words", help="random words with the bigbank mix of capitals, digits and punctuation")
    p.add_argument("--n", type=int, default=3500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("render", parents=[meta], help="render a word list in a font")
    p.add_argument("--font", required=True)
    p.add_argument("--words", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=96)
    p = sub.add_parser("segment", parents=[meta], help="cut a page scan into labeled, cleaned word images")
    p.add_argument("--page", required=True)
    p.add_argument("--transcript", required=True, help="one line of text per written line")
    p.add_argument("--overview", default="", help="write the page with every crop box drawn on it")
    p.add_argument("--out", required=True)
    p = sub.add_parser("export", help="InkSight weights and tokenizer for the torch engine (needs TensorFlow)")
    p.add_argument("--out", default="inksight")
    p = sub.add_parser("trace", help="trace word images with InkSight; resumable")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--prompts", nargs="+", default=["Derender the ink: {text}"])
    p.add_argument("--engine", default="torch", choices=["torch", "tf"])
    p.add_argument("--inksight", default="inksight", help="torch engine: dir written by export")
    rules = argparse.ArgumentParser(add_help=False)
    rules.add_argument("--manifest", required=True)
    rules.add_argument("--traces", required=True)
    rules.add_argument("--min_coverage", type=float, default=0.85)
    rules.add_argument("--min_mark", type=float, default=0.5, help="every separate ink mark (i-dot, period) this covered")
    rules.add_argument("--min_precision", type=float, default=0.90)
    rules.add_argument("--max_read_error", type=float, default=0.34)
    rules.add_argument("--max_length_ratio", type=float, default=1.6)
    rules.add_argument("--max_retrace", type=float, default=0.35, help="share of a mark's trace that runs over its own path")
    rules.add_argument("--max_mark_length", type=float, default=1.95, help="trace length over ink length, per mark")
    p = sub.add_parser("qc", parents=[rules], help="score traces against the image ink; keep the best passing one")
    p.add_argument("--out", required=True)
    p.add_argument("--sheet", default="")
    p = sub.add_parser("peek", parents=[rules], help="random traces drawn over their images, with QC verdicts")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--worst", default="", help="show the n worst by this QC metric instead of random traces")
    p.add_argument("--prompt", default="", help="only traces whose prompt starts with this")
    p.add_argument("--out", default="peek.png")
    args = parser.parse_args()
    {"words": words_command, "render": render_command, "export": export_command, "segment": segment_command, "trace": trace_command,
     "qc": qc_command, "peek": peek_command}[args.command](args)


if __name__ == "__main__":
    main()
