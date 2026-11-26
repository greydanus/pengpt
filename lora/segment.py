"""data/me/pages/*.png -> one crop per handwritten word, plus numbered contact sheets to label.

Writes data/me/words/<page>_<n>.png, data/me/words.jsonl (one record a word: id, page, box, line) and
data/me/sheets/sheet_<k>.png (numbered crops, in reading order, for transcription by eye).

Adapted from find_words() in inference/hand.py, with three changes for scanned assignment pages:
red pen (a grader's ticks) and the grey ruled lines are whitened before ink is found, every word is kept
(hand.py keeps the 16 widest), and short words ("a", "I", "of") pass the size filter.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference")]
from tracing import flatten, ink_mask  # noqa: E402


def stroke_width(mask):
    """Median pen width in pixels, from the distance transform along each stroke's middle."""
    from skimage.morphology import skeletonize
    d = ndimage.distance_transform_edt(mask)
    sk = skeletonize(mask)
    return float(2 * np.median(d[sk])) if sk.any() else 0.0


def track_rules(line, strips=32, seed=0.5, follow=0.25, drift=3, span=0.6, full=0.8):
    """Ruled lines as curves y(strip): each followed strip by strip across the page from a strong seed.

    `line` is a loose ink mask (lines are found in it; it may be fainter than the ink kept). In each
    narrow strip a curved line is flat, so it shows as a 1-3 px row that is dark across most of the
    strip. A seed is such a row; it is followed left and right, moving at most `drift` px a strip.
    Tracks that cover under `span` of the strips are text, not lines, and are dropped.
    """
    H, W = line.shape
    sw = max(8, W // strips)
    xs = list(range(0, W - sw // 2, sw))
    prof = np.stack([line[:, x:x + sw].mean(1) for x in xs], 1)  # H x n_strips: how full each row is in each strip
    tracks, taken = [], np.zeros(H, bool)
    for s0 in (len(xs) // 2, len(xs) // 4, 3 * len(xs) // 4):
        col = prof[:, s0]
        peaks = np.flatnonzero((col > seed) & (col >= np.r_[0, col[:-1]]) & (col >= np.r_[col[1:], 0]))
        for y0 in peaks:
            if taken[max(0, y0 - 2 * drift):y0 + 2 * drift + 1].any():
                continue
            ys, hits = {s0: y0}, 1
            for step in (1, -1):
                y = y0
                for s in range(s0 + step, len(xs) if step > 0 else -1, step):
                    lo, hi = max(0, y - drift), min(H, y + drift + 1)
                    k = lo + int(np.argmax(prof[lo:hi, s]))
                    if prof[k, s] > follow:
                        y, hits = k, hits + 1
                    ys[s] = y
            track = [ys[s] for s in range(len(xs))]
            fill = np.median([prof[y, s] for s, y in enumerate(track)])
            # a ruled line is solid across nearly every strip (median fill ~1.0); a row of text is
            # broken by the gaps between letters and words (median fill ~0.5 or less)
            if hits >= span * len(xs) and fill > full:
                if all(abs(np.median(track) - np.median(t)) > drift for t in tracks):
                    tracks.append(track)
                taken[max(0, y0 - drift):y0 + drift + 1] = True
    return tracks, xs, sw


def remove_rules(ink, line=None, **kw):
    """Erase ruled lines found by track_rules, keeping ink where a letter crosses them."""
    line = ink if line is None else line
    tracks, xs, sw = track_rules(line, **kw)
    ink = ink.copy()
    H, W = ink.shape
    for ys in tracks:
        for x0, y in zip(xs, ys):
            x1 = W if x0 == xs[-1] else x0 + sw
            band = line[max(0, y - 4):y + 5, x0:x1].mean(1) > 0.2  # the line's own rows, here
            if not band.any():
                continue
            top, bot = max(0, y - 4) + np.flatnonzero(band)[[0, -1]]
            top, bot = max(0, top - 1), min(H - 1, bot + 1)
            above = ink[max(0, top - 2), x0:x1]
            below = ink[min(H - 1, bot + 2), x0:x1]
            keep = ndimage.binary_dilation(above | below, iterations=1)  # a stroke passes through the line here
            ink[top:bot + 1, x0:x1] &= keep[None, :]
    return ink, len(tracks)


def clean_page(rgb, dark=110, redness=40):
    """Keep only the pen: drops red pen (a grader's), paper, show-through from the back, and ruled lines."""
    a = np.asarray(rgb.convert("RGB")).astype(int)
    gray = a.mean(-1)
    red = (a[..., 0] - np.maximum(a[..., 1], a[..., 2])) > redness
    loose = (gray < 190) & ~red  # lines are found in a looser mask, so a faint stretch does not break a track
    ink, n_rows = remove_rules((gray < dark) & ~red, loose)
    ink, n_cols = remove_rules(ink.T, loose.T, full=0.6)  # the margin line(s), down the left of the page
    ink = ink.T
    ink = ndimage.binary_opening(ink, structure=np.ones((1, 2))) | ndimage.binary_opening(ink, structure=np.ones((2, 1)))  # lone pixels
    out = np.where(ink, gray, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")


def find_words(img, min_h=0.45, max_h=3.4, min_w=0.5):
    norm = flatten(img)
    mask = ink_mask(norm)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n < 3:
        return [], 0
    area = ndimage.sum(mask, lab, np.arange(1, n + 1))
    big = area >= 0.25 * np.percentile(area, 90)  # letters, not specks of paper (specks outnumber letters on fine scans)
    h = np.median([b[0].stop - b[0].start for b, a in zip(ndimage.find_objects(lab), big) if a])
    bigmask = np.isin(lab, np.flatnonzero(big) + 1)
    edges = np.diff(np.pad(ndimage.binary_dilation(mask, structure=np.ones((int(0.6 * h), 1))), ((0, 0), (1, 1))).astype(np.int8), axis=1)
    gaps = np.concatenate([np.flatnonzero(r == 1)[1:] - np.flatnonzero(r == -1)[:-1] for r in edges[::4]])
    gaps = np.log(gaps[(gaps > 0.1 * h) & (gaps < 4 * h)])
    lo, hi = np.percentile(gaps, [25, 90]) if len(gaps) else (0, 0)
    for _ in range(20 * (hi > lo)):
        near = np.abs(gaps - lo) < np.abs(gaps - hi)
        lo, hi = gaps[near].mean(), gaps[~near].mean()
    gap = np.clip(np.exp((lo + hi) / 2), 0.3 * h, 1.5 * h) if hi > lo else 0.5 * h
    blob, _ = ndimage.label(ndimage.binary_dilation(mask, structure=np.ones((max(1, int(0.12 * h)), max(3, int(gap))))))
    words, pad = [], int(0.3 * h)
    for k, b in enumerate(ndimage.find_objects(blob), 1):
        ys, xs = np.nonzero((blob[b] == k) & mask[b])
        if not len(ys) or not ((blob[b] == k) & bigmask[b]).any():
            continue
        y0, y1 = b[0].start + ys.min(), b[0].start + ys.max() + 1
        x0, x1 = b[1].start + xs.min(), b[1].start + xs.max() + 1
        if not min_h * h <= y1 - y0 <= max_h * h or x1 - x0 < min_w * h:
            continue
        box = slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad)
        own = ndimage.binary_dilation((blob[box] == k) & mask[box], iterations=2)
        crop = Image.fromarray((np.where(own, norm[box], 1.0) * 255).astype(np.uint8))
        words.append({"box": [int(box[1].start), int(box[0].start), int(box[1].stop), int(box[0].stop)],
                      "core": [int(x0), int(y0), int(x1), int(y1)], "crop": crop})
    return words, float(h)


def reading_order(words, h):
    """Group words into lines by vertical centre, then left to right."""
    words = sorted(words, key=lambda w: (w["core"][1] + w["core"][3]) / 2)
    lines, cur, last = [], [], None
    for w in words:
        c = (w["core"][1] + w["core"][3]) / 2
        if last is not None and c - last > 0.6 * h:
            lines.append(cur)
            cur = []
        cur.append(w)
        last = c if not cur[:-1] else (last * (len(cur) - 1) + c) / len(cur)
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines):
        line.sort(key=lambda w: w["core"][0])
        for w in line:
            w["line"] = i
    return [w for line in lines for w in line]


def sheets(records, crops, out, per=40, cols=4, cell=(560, 150)):
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "sheet_*.png")):
        os.remove(f)
    font = ImageFont.load_default(26)
    paths = []
    for s in range(0, len(records), per):
        chunk = list(zip(records[s:s + per], crops[s:s + per]))
        rows = -(-len(chunk) // cols)
        img = Image.new("RGB", (cols * cell[0], rows * cell[1]), "white")
        d = ImageDraw.Draw(img)
        for i, (r, c) in enumerate(chunk):
            x, y = (i % cols) * cell[0], (i // cols) * cell[1]
            c = c.copy()
            c.thumbnail((cell[0] - 90, cell[1] - 16), Image.LANCZOS)
            img.paste(c, (x + 84, y + (cell[1] - c.height) // 2))
            d.text((x + 6, y + cell[1] // 2 - 14), str(r["n"]), fill=(200, 0, 0), font=font)
            d.rectangle([x, y, x + cell[0] - 1, y + cell[1] - 1], outline=(220, 220, 220))
        path = os.path.join(out, f"sheet_{s // per:03d}.png")
        img.save(path)
        paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pages", default="data/me/pages")
    p.add_argument("--out", default="data/me")
    p.add_argument("--per_sheet", type=int, default=40)
    args = p.parse_args()
    wdir = os.path.join(args.out, "words")
    os.makedirs(wdir, exist_ok=True)
    records, crops = [], []
    for path in sorted(glob.glob(os.path.join(args.pages, "*.png"))):
        page = os.path.splitext(os.path.basename(path))[0]
        words, h = find_words(clean_page(Image.open(path)))
        words = reading_order(words, h)
        for i, w in enumerate(words):
            wid = f"{page}_{i:03d}"
            w["crop"].save(os.path.join(wdir, f"{wid}.png"))
            records.append({"n": len(records), "id": wid, "page": page, "line": w["line"], "box": w["box"], "xheight_px": round(h, 1)})
            crops.append(w["crop"])
        print(f"{page}: {len(words)} words, {words[-1]['line'] + 1 if words else 0} lines, letter height {h:.0f}px")
    with open(os.path.join(args.out, "words.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    paths = sheets(records, crops, os.path.join(args.out, "sheets"), args.per_sheet)
    print(f"{len(records)} words -> {wdir}, {len(paths)} sheets -> {os.path.join(args.out, 'sheets')}")


if __name__ == "__main__":
    main()
