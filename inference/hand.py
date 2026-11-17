import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, UnidentifiedImageError
from scipy import ndimage

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import pen as mini
import pengpt as P
from tracing import PLAIN, TEXT, InkSight, Transcriber, flatten, ink_mask, load_env, pick, shrink, standardize, trace_round, variants


def load_photo(path, side=2000):
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        jpg = tempfile.NamedTemporaryFile(suffix=".jpg").name
        try:
            subprocess.run(["sips", "-s", "format", "jpeg", "-Z", str(side), str(path), "--out", jpg], check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit(f"{path} is not an image this can open (JPEG, PNG, WebP; HEIC needs pillow_heif or macOS)")
        img = Image.open(jpg)
    img.draft("RGB", (side, side))
    plain = ImageOps.exif_transpose(img).convert("RGB")
    plain.thumbnail((side, side), Image.LANCZOS)
    img = ImageOps.autocontrast(plain.convert("L"), cutoff=1)
    return (ImageOps.invert(img) if np.median(img) < 100 else img), plain


def find_words(img, most=16):
    norm = flatten(img)
    mask = ink_mask(norm)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n < 3:
        return [], 0
    area = ndimage.sum(mask, lab, np.arange(1, n + 1))
    h = np.median([b[0].stop - b[0].start for b, a in zip(ndimage.find_objects(lab), area) if a >= np.median(area)])
    edges = np.diff(np.pad(ndimage.binary_dilation(mask, structure=np.ones((int(0.6 * h), 1))), ((0, 0), (1, 1))).astype(np.int8), axis=1)
    gaps = np.concatenate([np.flatnonzero(r == 1)[1:] - np.flatnonzero(r == -1)[:-1] for r in edges[::4]])
    gaps = np.log(gaps[(gaps > 0.1 * h) & (gaps < 4 * h)])
    lo, hi = np.percentile(gaps, [25, 90]) if len(gaps) else (0, 0)
    for _ in range(20 * (hi > lo)):
        near = np.abs(gaps - lo) < np.abs(gaps - hi)
        lo, hi = gaps[near].mean(), gaps[~near].mean()
    gap = np.clip(np.exp((lo + hi) / 2), 0.3 * h, 1.5 * h) if hi > lo else 0.5 * h
    blob, _ = ndimage.label(ndimage.binary_dilation(mask, structure=np.ones((max(1, int(0.12 * h)), max(3, int(gap))))))
    crops, pad = [], int(0.3 * h)
    for k, b in enumerate(ndimage.find_objects(blob), 1):
        ys, xs = np.nonzero((blob[b] == k) & mask[b])
        y0, y1, x0, x1 = b[0].start + ys.min(), b[0].start + ys.max() + 1, b[1].start + xs.min(), b[1].start + xs.max() + 1
        if min(y0, x0) <= 1 or y1 >= mask.shape[0] - 1 or x1 >= mask.shape[1] - 1 or not 0.5 * h <= y1 - y0 <= 3.2 * h or x1 - x0 < 1.5 * h:
            continue
        box = slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad)
        own = ndimage.binary_dilation((blob[box] == k) & mask[box], iterations=2)
        crops.append(Image.fromarray((np.where(own, norm[box], 1.0) * 255).astype(np.uint8)))
        crops[-1].info["row"] = (y0 + y1) / 2
    return sorted(crops, key=lambda c: -c.width)[:most], h


def trace_words(tracer, crops, h, n=8, enough=5, min_legibility=6):
    preps = []
    for crop in crops:
        img, scale = shrink(crop.convert("RGB"))
        preps.append({"row": {"text": "x" * round(crop.width / h)}, "scale": scale, "variants": variants(img)})
    with ThreadPoolExecutor(1) as ex:
        reads = ex.submit(Transcriber(max_cost=0.5), crops)
        found = trace_round(tracer, preps, [("mono", PLAIN), ("raw", PLAIN)], 64)
        reads = reads.result()
    good = [i for i, (t, leg) in enumerate(reads) if leg >= min_legibility and 2 <= len(t) <= 12 and t.isascii() and " " not in t and any(c.islower() for c in t)]
    for i in good:
        preps[i]["row"]["text"] = reads[i][0]
    judge = lambda i, cands, best=None: pick((preps[i]["variants"]["clean"], reads[i][0], cands, best))
    best = {i: judge(i, found[i]) for i in good}
    again = [i for i in good if not best[i].get("key", (False,))[0]]
    if len(good) - len(again) < enough and again:
        for i, cands in zip(again, trace_round(tracer, [preps[i] for i in again], [("clean", TEXT), ("mono", TEXT)], 64)):
            best[i] = judge(i, cands, best[i])
    passed = sorted((i for i in good if best[i].get("key", (False,))[0]), key=lambda i: -best[i]["key"][1])[:n]
    words = [{"text": reads[i][0], "strokes": [np.asarray(s, float) / preps[i]["scale"] for s in best[i]["strokes"]], "image": crops[i]} for i in passed]
    rows = standardize([{**w, "frame": "me", "writer": "me", "style": "me", "source": "photo", "license": "", "baseline": None} for w in words]) if words else []
    for w, r in zip(words, rows):
        w["points"] = np.concatenate([np.r_[np.c_[s, np.ones(len(s))], [[*s[-1], 0]]] for s in map(np.asarray, r["strokes"])])
    return words, len(good)


def fit_style(pen, texts, points, steps=30, lr=0.3):
    cfg = P.ModelConfig(n_layer=pen.n_layer, n_head=pen.n_head, n_embd=pen.hd * pen.n_head, pen_pos_bands=pen.bands, pen_rotary=pen.pen_rotary,
                        n_sources=len(pen.writers), vocab_size=pen.BOS + 1, block_size=pen.block_size,
                        context_vocab_size=len(pen.w["ctx_emb.weight"]), context_block_size=len(pen.w["ctx_pos_emb.weight"]))
    with contextlib.redirect_stdout(None):
        model = P.PenTransformer(cfg).eval().requires_grad_(False)
    model.load_state_dict({**{k.replace("writer_emb.", "source_emb."): torch.from_numpy(v) for k, v in pen.w.items()},
                           "pen_deltas": torch.from_numpy(pen.deltas)})
    data = P.DataConfig(grid=pen.grid, max_seq_length=pen.max_seq_length, max_text_length=pen.max_text_length, text_end=pen.text_end)
    st = P.ScribeTokenizer(grid=pen.grid, merges=pen.merges)
    ds = P.PenDataset([], [], [], st, P.Labels(data), data)
    X, C, Y, _ = (torch.stack(t) for t in zip(*[ds.make_item(st.encode_word(p), ds.encode_text(text)) for p, text in zip(points, texts)]))
    n = int((X != st.PAD).sum(1).max())
    X, Y = X[:, :n], Y[:, :n]
    xy = model.pen_deltas[X].cumsum(1)
    rotary = P.pen_rotary_tables(torch.arange(n), xy, pen.hd) if pen.pen_rotary else P.rotary_tables(n, pen.hd, "cpu")
    x, (c, mask) = model.embed(X, xy), model.encode_context(C)
    v = model.source_emb.weight[0].clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    for step in range(steps + 1):
        logits = model.logits(x, torch.cat([P.rms_norm(v).expand(len(X), 1, -1), c[:, 1:]], 1), mask, rotary)
        if step < steps:
            opt.zero_grad()
            F.cross_entropy(logits.transpose(1, 2), Y, ignore_index=P.IGNORE_INDEX).backward()
            opt.step()
    logits = logits.detach()[Y != P.IGNORE_INDEX]
    logp = F.log_softmax(logits, -1)
    real, temps = -logp.gather(1, Y[Y != P.IGNORE_INDEX][:, None]).mean().item(), np.linspace(0.05, 1.5, 59)
    return v.detach().numpy(), float(np.interp(real, [-(F.softmax(logits / t, -1) * logp).sum(1).mean().item() for t in temps], temps))


def draw(words, scale, width, ink, pad=0.12):
    pts = np.concatenate(words)
    lo = pts[:, :2].min(0) - pad
    img = Image.new("L", tuple(int(v) for v in (pts[:, :2].max(0) + pad - lo) * 2 * scale), 255)
    d, r = ImageDraw.Draw(img), width
    for p in words:
        for a, b in mini.strokes(p):
            xy = [tuple(q) for q in (p[a:b, :2] - lo) * 2 * scale]
            d.line(xy, fill=ink, width=round(2 * r), joint="curve")
            for x, y in (xy[0], xy[-1]):
                d.ellipse([x - r, y - r, x + r, y + r], fill=ink)
    return img.filter(ImageFilter.GaussianBlur(0.35 * r)).resize((img.width // 2, img.height // 2), Image.LANCZOS)


def figure(words, same, written, plain):
    crops = [w["image"] for w in words]
    grey = [np.asarray(c) for c in crops]
    width = 2.2 * np.median(np.concatenate([ndimage.distance_transform_edt(g < 200)[g < 200] for g in grey]))
    scale = np.percentile([np.ptp(np.concatenate(w["strokes"])[:, 0]) / max(np.ptp(s[:, 0]), 1e-6) for w, s in zip(words, same) if len(s)], 50)
    starts = np.cumsum([0] + [c.width for c in crops])
    strip = Image.new("L", (int(starts[-1]), max(c.height for c in crops)), 255)
    for x, c in zip(starts, crops):
        strip.paste(c, (int(x), (strip.height - c.height) // 2))
    pen = dict(scale=scale, width=float(max(width, 1.5)), ink=int(np.percentile(np.concatenate([g[g < 200] for g in grey]), 5)))
    under, right = [], -np.inf
    for s, x, c in zip(same, starts, crops):
        if len(s):
            under.append(s + [max((x + c.width / 2) / scale - (s[:, 0].min() + s[:, 0].max()) / 2, right + 0.12 - s[:, 0].min()), 0, 0])
            right = under[-1][:, 0].max()
    rows = [("words found in it", strip, 0), ("the same words, written by the model", draw(under, **pen), int((min(u[:, 0].min() for u in under) - 0.12) * scale)),
            ("new text in the same hand", draw(mini.layout(written, max(strip.width, 1800) / scale), **pen), 0)]
    wide = max(im.width + x for _, im, x in rows)
    zoom = min(1.0, wide / plain.width)
    tall = min(int(2.5 * strip.height / zoom), plain.height)
    top = int(np.clip(np.median([c.info["row"] for c in crops]) - tall / 2, 0, plain.height - tall))
    dark = ndimage.uniform_filter1d(255 - np.asarray(plain.convert("L"), float).mean(1), max(3, tall // 20))
    quiet = lambda y: max(y - tall // 4, 0) + int(np.argmin(dark[max(y - tall // 4, 0):y + tall // 4 + 1]))
    top, end = quiet(top), quiet(top + tall)
    view = plain.crop((0, top, plain.width, end))
    rows.insert(0, ("the photo", view.resize((round(view.width * zoom), round(view.height * zoom)), Image.LANCZOS), 0))
    font, gap = ImageFont.load_default(max(18, strip.height // 5)), max(30, strip.height // 3)
    out = Image.new("RGB", (wide + 20, sum(im.height + gap for _, im, _ in rows) + 10), "white")
    d, y = ImageDraw.Draw(out), 0
    for title, im, x in rows:
        d.text((10, y + gap // 5), title, fill=(110, 110, 110), font=font)
        out.paste(im, (10 + x, y + gap))
        y += im.height + gap
    return out


def main():
    p = argparse.ArgumentParser(description="a photo of a few handwritten words -> a style vector -> any text in that hand")
    p.add_argument("photo")
    p.add_argument("text", nargs="?", default="The quick brown fox jumps over the lazy dog")
    p.add_argument("--model", default="hand.npz")
    p.add_argument("--tracer", default="inksight")
    p.add_argument("--temperature", type=float, default=None, help="default: matched to how unpredictable the photo's ink is to the model")
    p.add_argument("--top_k", type=int, default=5, help="sample among the k likeliest pen moves, which keeps the spelling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="hand")
    args = p.parse_args()
    t = [time.time()]
    lap = lambda: (t.append(time.time()), t[-1] - t[-2])[1]
    load_env()
    for ok, fix in [(os.path.exists(args.photo), f"no file at {args.photo}"),
                    (os.environ.get("ANTHROPIC_API_KEY"), "set ANTHROPIC_API_KEY (or put it in .env): Claude reads the words")]:
        if not ok:
            raise SystemExit(fix)
    for f in ("weights.npz", "spm.model"):
        mini.fetch(f"{args.tracer}/{f}", f"inksight-{f}")
    lap()
    photo, plain = load_photo(args.photo)
    print(f"photo loaded as {photo.width} x {photo.height} grayscale in {lap():.2f}s")
    crops, h = find_words(photo)
    if len(crops) < 3:
        raise SystemExit(f"found {len(crops)} words in the photo, need at least 3: photograph a few lines of dark handwriting on plain paper, with the whole words in frame")
    words, read = trace_words(InkSight(args.tracer), crops, h)
    print(f"{len(crops)} words boxed, {read} read clearly, {len(words)} traced cleanly in {lap():.1f}s: {' '.join(w['text'] for w in words)}")
    if read < 3:
        raise SystemExit(f"only {read} of {len(crops)} boxed words could be read: is the photo sideways, upside down or out of focus?")
    if len(words) < 3:
        raise SystemExit(f"{read} words were read but only {len(words)} could be traced cleanly, need at least 3: try a sharper, closer photo with more contrast")
    pen, texts = mini.Pen(args.model), [w["text"] for w in words]
    style, temperature = fit_style(pen, texts, [w["points"] for w in words])
    temperature = temperature if args.temperature is None else args.temperature
    np.save(f"{args.out}.npy", np.r_[style, temperature])
    print(f"style fit in {lap():.1f}s, temperature {temperature:.2f} -> {args.out}.npy")
    same, written = (pen.write(s, style, temperature, args.top_k, args.seed) for s in (" ".join(texts), args.text))
    figure(words, same, written, plain).save(f"{args.out}.png")
    mini.svg(mini.layout(written), f"{args.out}.svg", color="#333")
    print(f"written in {lap():.1f}s, {t[-1] - t[0]:.1f}s in all -> {args.out}.png, {args.out}.svg")


if __name__ == "__main__":
    main()
