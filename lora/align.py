"""Match word crops to a page's transcript: the ground truth for each crop.

data/me/transcripts/<page>.txt holds the page as written (transcribed by eye), one line of the page
a line of the file. That text is the ground truth. To know which crop holds which transcript word,
the page's OCR (lora/ocr, Apple Vision; data/me/ocr.jsonl) is used only as an anchor:

  1. each OCR word belongs to the crop its box overlaps most;
  2. the transcript is aligned to the OCR words by spelling similarity (dynamic programming that
     tolerates OCR misreads and lets one OCR word match two transcript words or the reverse);
  3. a crop's label is the transcript text that lands on it, in order.

A crop with no transcript text is junk (a fragment, a crossed-out word, the header, a grader's mark).
A label is marked `check` when it is not one contiguous run of transcript words, or when the crop's
width does not fit the label's length (a fitted width a letter and a space, per page), or when OCR and
transcript disagree a lot. Writes data/me/labels.jsonl and review sheets data/me/review/<page>.png.
"""
import argparse
import glob
import json
import os
import re
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def tokens(path):
    words, line_of = [], []
    for i, line in enumerate(open(path)):
        for w in line.split():
            words.append(w)
            line_of.append(i)
    return words, line_of


def overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def assign_ocr(ocr_words, recs):
    """Index of the crop each OCR word lies in (by most overlap of its box), or -1."""
    out = []
    for w in ocr_words:
        b = w["box"]
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        ov = [overlap(b, r["box"]) / area for r in recs]
        k = int(np.argmax(ov)) if ov else -1
        out.append(k if k >= 0 and ov[k] > 0.25 else -1)
    return out


def align_words(T, O, gap_t=0.55, gap_o=0.45):
    """Align transcript words T to OCR words O. Returns, for each transcript word, an OCR index or -1."""
    n, m = len(T), len(O)
    D = np.full((n + 1, m + 1), np.inf)
    B = np.zeros((n + 1, m + 1, 2), int)
    D[0, 0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, arg = np.inf, (0, 0)
            if i and D[i - 1, j] + gap_t < best:
                best, arg = D[i - 1, j] + gap_t, (1, 0)
            if j and D[i, j - 1] + gap_o < best:
                best, arg = D[i, j - 1] + gap_o, (0, 1)
            if i and j:
                c = D[i - 1, j - 1] + (1 - sim(T[i - 1], O[j - 1]))
                if c < best:
                    best, arg = c, (1, 1)
            if i >= 2 and j:  # one OCR word read two transcript words ("opento")
                c = D[i - 2, j - 1] + (1 - sim(T[i - 2] + T[i - 1], O[j - 1])) + 0.1
                if c < best:
                    best, arg = c, (2, 1)
            if i and j >= 2:  # OCR split one word in two
                c = D[i - 1, j - 2] + (1 - sim(T[i - 1], O[j - 2] + O[j - 1])) + 0.1
                if c < best:
                    best, arg = c, (1, 2)
            D[i, j], B[i, j] = best, arg
    to, i, j = [-1] * n, n, m
    while i or j:
        di, dj = B[i, j]
        if di and dj:
            for q in range(di):
                to[i - 1 - q] = j - 1  # the last OCR word of the pair carries the crop
            if dj == 2:
                to[i - 1] = j - 2
        i, j = i - di, j - dj
    return to


def run_page(page, recs, T, line_of, ocr):
    O = [w["text"] for w in ocr["words"]]
    crop_of_o = assign_ocr(ocr["words"], recs)
    t2o = align_words(T, O)
    crop_words = {}
    for t, o in enumerate(t2o):
        if o >= 0 and crop_of_o[o] >= 0:
            crop_words.setdefault(crop_of_o[o], []).append(t)
    ocr_text = {}
    for o, k in enumerate(crop_of_o):
        if k >= 0:
            ocr_text.setdefault(k, []).append(O[o])
    h = recs[0]["xheight_px"]
    width = np.array([max(r["box"][2] - r["box"][0] - 0.6 * h, 0.2 * h) / h for r in recs])
    # fit a letter's and a space's width (in letter heights) on crops whose label is one clean run
    rows, ys = [], []
    for k, ts in crop_words.items():
        if ts == list(range(ts[0], ts[-1] + 1)):
            rows.append([sum(len(T[t]) for t in ts), len(ts) - 1])
            ys.append(width[k])
    letter, space = np.linalg.lstsq(np.array(rows, float), np.array(ys), rcond=None)[0] if len(rows) > 5 else (0.5, 0.5)
    letter, space = float(np.clip(letter, 0.2, 1.2)), float(np.clip(space, 0.1, 2.0))
    out = []
    for k, r in enumerate(recs):
        ts = crop_words.get(k, [])
        if not ts:
            out.append({"id": r["id"], "page": page, "text": "", "status": "junk", "ocr": " ".join(ocr_text.get(k, []))})
            continue
        text = " ".join(T[t] for t in ts)
        est = letter * sum(len(T[t]) for t in ts) + space * (len(ts) - 1)
        fit = float(abs(np.log(width[k] / est)))
        agree = sim(text, "".join(ocr_text.get(k, [])))
        why = []
        if ts != list(range(ts[0], ts[-1] + 1)) or len({line_of[t] for t in ts}) > 1:
            why.append("not one run")
        if fit > 0.3:
            why.append(f"width {width[k]:.1f}h vs {est:.1f}h")
        if agree < 0.6:
            why.append(f"ocr '{' '.join(ocr_text.get(k, []))}'")
        out.append({"id": r["id"], "page": page, "text": text, "status": "check" if why else "ok", "why": "; ".join(why),
                    "fit": round(fit, 3), "agree": round(agree, 3), "ocr": " ".join(ocr_text.get(k, []))})
    landed = {t for ts in crop_words.values() for t in ts}
    missed = [T[t] for t in range(len(T)) if t not in landed]
    return out, missed, letter, space


def review_sheet(labels, words_dir, out, cols=5, cell=(440, 150)):
    font = ImageFont.load_default(24)
    rows = -(-len(labels) // cols)
    img = Image.new("RGB", (cols * cell[0], max(1, rows) * cell[1]), "white")
    d = ImageDraw.Draw(img)
    color = {"ok": (0, 120, 0), "check": (230, 110, 0), "junk": (150, 150, 150)}
    for i, lab in enumerate(labels):
        x, y = (i % cols) * cell[0], (i // cols) * cell[1]
        c = Image.open(os.path.join(words_dir, lab["id"] + ".png"))
        c.thumbnail((cell[0] - 10, cell[1] - 46), Image.LANCZOS)
        img.paste(c, (x + 5, y + 4))
        d.text((x + 5, y + cell[1] - 38), f"{i} {lab['text'] or '(junk)'}", fill=color[lab["status"]], font=font)
        d.rectangle([x, y, x + cell[0] - 1, y + cell[1] - 1], outline=(225, 225, 225))
    img.save(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="data/me")
    p.add_argument("--pages", nargs="*", help="only these pages (default: every page with a transcript)")
    args = p.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.data, "words.jsonl"))]
    by_page = {}
    for r in recs:
        by_page.setdefault(r["page"], []).append(r)
    ocr = {}
    for l in open(os.path.join(args.data, "ocr.jsonl")):
        x = json.loads(l)
        ocr[os.path.splitext(os.path.basename(x["image"]))[0]] = x
    path = os.path.join(args.data, "labels.jsonl")
    labels = {}
    if os.path.exists(path) and args.pages:
        for l in open(path):
            x = json.loads(l)
            labels[x["id"]] = x
    os.makedirs(os.path.join(args.data, "review"), exist_ok=True)
    pages = args.pages or sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(args.data, "transcripts", "*.txt")))
    total = {"ok": 0, "check": 0, "junk": 0}
    for page in pages:
        T, line_of = tokens(os.path.join(args.data, "transcripts", page + ".txt"))
        out, missed, letter, space = run_page(page, by_page[page], T, line_of, ocr[page])
        for o in out:
            labels[o["id"]] = o
        n = {s: sum(o["status"] == s for o in out) for s in total}
        for s in total:
            total[s] += n[s]
        words_ok = sum(len(o["text"].split()) for o in out if o["status"] == "ok")
        print(f"{page}: {len(out)} crops: {n['ok']} ok ({words_ok}/{len(T)} words), {n['check']} check, {n['junk']} junk; "
              f"{len(missed)} words in no crop; letter {letter:.2f}h space {space:.2f}h")
        review_sheet(out, os.path.join(args.data, "words"), os.path.join(args.data, "review", page + ".png"))
    with open(path, "w") as f:
        for k in sorted(labels):
            f.write(json.dumps(labels[k]) + "\n")
    print(f"all: {total}")


if __name__ == "__main__":
    main()
