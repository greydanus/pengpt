import argparse
import hashlib
import json
import os
import re
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

SIZE = 224
UP = 4
INK_LEVEL = 200


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


def tone(norm, keep):
    t = ((norm - 0.35) / (0.92 - 0.35)).clip(0, 1) ** 1.4
    img = ndimage.gaussian_filter(np.where(keep, t, 1.0), 0.7)
    return Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")


def upsample(norm):
    h, w = norm.shape
    big = Image.fromarray((norm * 65535).astype(np.uint16)).resize((w * UP, h * UP), Image.BICUBIC)
    return np.asarray(big).astype(float) / 65535


def render_command(args):
    font = ImageFont.truetype(args.font, args.size)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i, text in enumerate(w.strip() for w in open(args.words) if w.strip()):
        left, top, right, bottom = font.getbbox(text, anchor="ls")
        img = Image.new("RGB", (right - left + 16, bottom - top + 16), "white")
        ImageDraw.Draw(img).text((8 - left, 8 - top), text, font=font, fill="black", anchor="ls")
        path = f"{args.out}/{os.path.splitext(os.path.basename(args.font))[0]}_{i:05d}.png"
        img.save(path)
        rows.append({"image": path, "text": text, "baseline": 8 - top, "style": args.style,
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
    slant = estimate_slant(ink, bands)
    print(f"estimated slant: {np.degrees(np.arctan(slant)):.0f} degrees from vertical")
    ys_all, xs_all = np.mgrid[0:ink.shape[0], 0:ink.shape[1]]
    rows, skipped, boxes = [], [], []
    for li, ((a, b), text) in enumerate(zip(bands, lines)):
        continued = li > 0 and lines[li - 1].endswith(("=", "-"))
        h = b - a
        y0, y1 = max(0, a - h // 2), min(ink.shape[0], b + h // 2)
        tokens = text.split()
        center = (a + b) / 2
        alone = faint.copy()
        for a2, b2 in bands:
            if (a2, b2) != (a, b):
                alone[a2:b2] = False
        labels, _ = ndimage.label(alone, structure=np.ones((3, 3)))
        margin = int(np.ceil(abs(slant) * ink.shape[0])) + 1
        words = split_line(shear(ink[a:b], center - a, slant, margin), tokens, h)
        if words is None:
            skipped.append(text)
            continue
        for wi, ((s, e), token) in enumerate(zip(words, tokens)):
            if token.endswith(("=", "-")) or (continued and wi == 0) or not token.strip(".,;:!?\"'"):
                continue
            pad = h // 4
            sheared_x = xs_all + np.rint((ys_all - center) * slant).astype(int) + margin
            core = labels[a + h // 4:b - h // 4] * ((sheared_x[a + h // 4:b - h // 4] >= s) & (sheared_x[a + h // 4:b - h // 4] < e))
            mine = np.unique(core[core > 0])
            left = (words[wi - 1][1] + s) // 2 if wi else s - pad
            right = (e + words[wi + 1][0]) // 2 if wi + 1 < len(words) else e + pad
            keep = np.isin(labels, mine) & (sheared_x >= left) & (sheared_x < right)
            if not keep.any():
                continue
            ys, xs = np.nonzero(keep)
            y0, y1 = max(0, ys.min() - pad), min(ink.shape[0], ys.max() + pad)
            x0, x1 = max(0, xs.min() - pad), min(ink.shape[1], xs.max() + pad)
            near = ndimage.binary_dilation(keep, iterations=2)[y0:y1, x0:x1]
            word = np.where(near, norm[y0:y1, x0:x1], 1.0)
            boxes.append((x0, y0, x1, y1, token))
            n = upsample(word)
            path = f"{args.out}/{stem}_{li:02d}_{wi:02d}.png"
            tone(n, ink_mask(n) | (n < 0.92)).save(path)
            rows.append({"image": path, "text": token, "baseline": int((b - y0) * UP), "style": args.style,
                         "source": os.path.basename(args.page), "license": args.license})
    replace_rows(f"{args.out}/manifest.jsonl", rows, os.path.basename(args.page))
    print(f"{stem}: {len(rows)} words from {len(lines) - len(skipped)}/{len(lines)} lines")
    if args.overview:
        overview(page, boxes, bands, lines, skipped, args.overview)
    for text in skipped:
        print(f"  skipped (could not split confidently): {text}")


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


def scale_and_pad(img):
    ratio = min(SIZE / img.width, SIZE / img.height)
    small = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    canvas = Image.new("RGB", (SIZE, SIZE), "black")
    dx, dy = (SIZE - small.width) // 2, (SIZE - small.height) // 2
    canvas.paste(small, (dx, dy))
    return canvas, ratio, dx, dy


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


def trace_command(args):
    import tensorflow as tf
    import tensorflow_text
    from huggingface_hub import from_pretrained_keras
    serve = from_pretrained_keras("Derendering/InkSight-Small-p").signatures["serving_default"]
    items = [{**it, "hash": file_hash(it["image"])} for it in read_jsonl(args.manifest)]
    done = {(r.get("hash"), r["prompt"]) for r in read_jsonl(args.out)}
    t0, n = time.time(), 0
    for prompt in args.prompts:
        todo = [it for it in items if (it["hash"], prompt) not in done]
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            images = [Image.open(it["image"]).convert("RGB") for it in chunk]
            padded = [scale_and_pad(img) for img in images]
            encoded = tf.stack([tf.reshape(tf.io.encode_jpeg(np.asarray(c)), (1,)) for c, _, _, _ in padded])
            texts = tf.constant([prompt.format(text=it["text"]) for it in chunk])
            out = serve(**{"input_text": texts, "image/encoded": encoded})["output_0"].numpy()
            rows = []
            for it, img, o, (_, ratio, dx, dy) in zip(chunk, images, out, padded):
                raw = o[0].decode()
                strokes = on_ink([[[(x - dx) / ratio, (y - dy) / ratio] for x, y in s] for s in detokenize(raw)], img)
                rows.append({"image": it["image"], "hash": it["hash"], "prompt": prompt, "strokes": strokes,
                             "read": re.sub(r"<[^>]+>", "", raw).strip()})
            write_jsonl(args.out, rows, "a")
            n += len(chunk)
            print(f"[{prompt}] {i + len(chunk)}/{len(todo)} | {(time.time() - t0) / n:.1f} s/word", flush=True)


def levenshtein(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def score(item, cand):
    gray = np.asarray(Image.open(item["image"]).convert("L"))
    ink = gray < INK_LEVEL
    if not ink.any() or not cand["strokes"]:
        return {"coverage": 0.0, "precision": 0.0, "length_ratio": 0.0, "points": 0}
    width = max(2.0, 2 * np.median(ndimage.distance_transform_edt(ink)[ink]))
    canvas = Image.new("L", (gray.shape[1], gray.shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    pts = []
    for s in cand["strokes"]:
        s = np.asarray(s, float)
        if len(s) > 1:
            draw.line([tuple(p) for p in s], fill=255, width=1)
            seg = np.diff(s, axis=0)
            steps = np.maximum(1, np.ceil(np.hypot(*seg.T)).astype(int))
            pts += [s[k] + seg[k] * t for k in range(len(seg)) for t in np.linspace(0, 1, steps[k], endpoint=False)]
        else:
            draw.point(tuple(s[0]), fill=255)
            pts.append(s[0])
    path = ndimage.binary_dilation(np.asarray(canvas) > 0, iterations=int(np.ceil(width)))
    near = ndimage.binary_dilation(ink, iterations=int(np.ceil(width / 2)) + 2)
    pts = np.clip(np.rint(np.asarray(pts)).astype(int), 0, [gray.shape[1] - 1, gray.shape[0] - 1])
    return {"coverage": float((ink & path).sum() / ink.sum()),
            "precision": float(near[pts[:, 1], pts[:, 0]].mean()),
            "length_ratio": float(len(pts) * width / ink.sum()),
            "points": int(len(pts))}


def qc_command(args):
    items = {it["image"]: it for it in read_jsonl(args.manifest)}
    cands = {}
    for c in read_jsonl(args.traces):
        cands.setdefault(c.get("hash") or c["image"], []).append(c)
    accepted, rejected = [], []
    for image, item in items.items():
        best = None
        for c in cands.get(file_hash(image), []) + cands.get(image, []):
            c = {**c, "strokes": on_ink(c["strokes"], Image.open(image))}
            q = score(item, c)
            label = item["text"].lower()
            q["read"] = c["read"]
            q["read_error"] = levenshtein(c["read"].lower(), label) / len(label) if c["read"] else None
            q["prompt"] = c["prompt"]
            q["passed"] = (q["coverage"] >= args.min_coverage and q["precision"] >= args.min_precision
                           and q["length_ratio"] <= args.max_length_ratio
                           and (q["read_error"] is None or q["read_error"] <= args.max_read_error))
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
    write_jsonl(args.out.replace(".jsonl", "_rejected.jsonl"), rejected)
    print(f"accepted {len(accepted)} / {len(items)} ({100 * len(accepted) / max(1, len(items)):.0f}%), "
          f"rejected {len(rejected)}")
    if args.sheet:
        review_sheet(accepted, rejected, args.sheet)


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
        detail = (f"cov {q.get('coverage', 0):.2f}  prec {q.get('precision', 0):.2f}  "
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
    p = sub.add_parser("trace", help="trace word images with InkSight; resumable")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prompts", nargs="+", default=["Recognize and derender.", "Derender the ink: {text}"])
    p = sub.add_parser("qc", help="score traces against the image ink; keep the best passing one")
    p.add_argument("--manifest", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sheet", default="")
    p.add_argument("--min_coverage", type=float, default=0.88)
    p.add_argument("--min_precision", type=float, default=0.90)
    p.add_argument("--max_read_error", type=float, default=0.34)
    p.add_argument("--max_length_ratio", type=float, default=1.3)
    args = parser.parse_args()
    {"render": render_command, "segment": segment_command, "trace": trace_command,
     "qc": qc_command}[args.command](args)


if __name__ == "__main__":
    main()
