from concurrent.futures import ThreadPoolExecutor
import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import threading
import unicodedata

from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree
import numpy as np
import torch
import torch.nn.functional as F


X_HEIGHT, INK_HEIGHT, SPACING, UNIT = 0.123, 0.22, 0.01, 2e-3


SHORT_LETTERS = set("acemnorsuvwxz")


AUTHOR_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789"


def author_id(writer):
    n = int.from_bytes(hashlib.sha256(writer.encode()).digest()[:8], "big")
    return "".join(AUTHOR_CHARS[n // len(AUTHOR_CHARS) ** k % len(AUTHOR_CHARS)] for k in range(5))


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def catmull_rom(p, per=16):
    p = p[np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 1e-9]]
    if len(p) < 3:
        return p
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    p0, p1, p2, p3 = q[:-3], q[1:-2], q[2:-1], q[3:]
    t1 = np.hypot(*(p1 - p0).T) ** 0.5
    t2 = t1 + np.hypot(*(p2 - p1).T) ** 0.5
    t3 = t2 + np.hypot(*(p3 - p2).T) ** 0.5
    t = t1[:, None] + (t2 - t1)[:, None] * np.linspace(0, 1, per, endpoint=False)
    lerp = lambda pa, pb, ta, tb: ((tb[:, None, None] - t[..., None]) * pa[:, None] + (t[..., None] - ta[:, None, None]) * pb[:, None]) / (tb - ta)[:, None, None]
    zero = np.zeros_like(t1)
    a1, a2, a3 = lerp(p0, p1, zero, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3)
    b1 = ((t2[:, None] - t)[..., None] * a1 + (t - zero[:, None])[..., None] * a2) / t2[:, None, None]
    b2 = ((t3[:, None] - t)[..., None] * a2 + (t - t1[:, None])[..., None] * a3) / (t3 - t1)[:, None, None]
    c = ((t2[:, None] - t)[..., None] * b1 + (t - t1[:, None])[..., None] * b2) / (t2 - t1)[:, None, None]
    return np.vstack([c.reshape(-1, 2), p[-1:]])


def resample_stroke(p, spacing=SPACING):
    if len(p) < 2:
        return p
    dense = catmull_rom(np.asarray(p, float))
    if len(dense) < 2:
        return dense[:1]
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(dense, axis=0).T))]
    s = np.linspace(0, cum[-1], max(1, int(round(cum[-1] / spacing))) + 1)
    return np.column_stack([np.interp(s, cum, dense[:, 0]), np.interp(s, cum, dense[:, 1])])


def group_xheight(group):
    words = [np.ptp(np.quantile(np.concatenate(s["strokes"])[:, 1], [0.02, 0.98])) for s in group
             if len(s["text"]) >= 2 and set(s["text"]) <= SHORT_LETTERS]
    if len(words) >= 5:
        return float(np.median(words)), "words"
    letters = [h for s in group for h in s.get("xh", [])]
    if len(letters) >= 5:
        return float(np.median(letters)), "letters"
    bands = [np.subtract(*core_band(s["strokes"], frac=0.25)[::-1]) for s in group if any(c.islower() for c in s["text"])]
    if len(bands) >= 3:
        return float(np.median(bands)), "band"
    return None, "none"


def standardize(samples):
    by_frame = {}
    for s in samples:
        by_frame.setdefault(s["frame"], []).append(s)
    out, how = [], {}
    for frame, group in by_frame.items():
        xh, how[frame] = group_xheight(group)
        if xh:
            k = X_HEIGHT / xh
            if how[frame] in ("words", "letters"):
                trial = [[resample_stroke(np.column_stack([st[:, 0] * k, st[:, 1] * k])) for st in s["strokes"]] for s in group
                         if set(s["text"]) <= SHORT_LETTERS and len(s["text"]) >= (2 if how[frame] == "words" else 1)]
                got = np.median([np.ptp(np.quantile(np.concatenate(t)[:, 1], [0.02, 0.98])) for t in trial])
                k *= X_HEIGHT / got
        else:
            heights = [np.ptp(np.quantile(np.concatenate(s["strokes"])[:, 1], [0.05, 0.95])) for s in group]
            k = INK_HEIGHT / max(np.median(heights), 1e-9)
        for s in group:
            x0 = s["strokes"][0][0, 0]
            y0 = s["baseline"] if s.get("baseline") is not None else baseline(s["strokes"], s["text"], X_HEIGHT / k)
            strokes = [resample_stroke(np.column_stack([(st[:, 0] - x0) * k, (st[:, 1] - y0) * k])).round(5).tolist()
                       for st in s["strokes"]]
            out.append({"text": s["text"], "strokes": strokes, "style": s["style"], "writer": s["writer"],
                        "author": s.get("author") or author_id(s["writer"]), "source": s["source"], "license": s["license"]})
    return out


def densify(strokes, step):
    out = []
    for s in strokes:
        s = np.asarray(s, float)
        if len(s) < 2:
            out.append(s)
            continue
        seg = np.diff(s, axis=0)
        n = np.maximum(1, (np.hypot(*seg.T) / step).astype(int))
        out.append(np.concatenate([s[i] + seg[i] * np.linspace(0, 1, n[i], endpoint=False)[:, None]
                                   for i in range(len(seg))] + [s[-1:]]))
    return np.concatenate(out)


def core_band(strokes, bins=40, frac=0.5):
    y = np.concatenate(strokes)[:, 1]
    y = densify(strokes, max(np.ptp(y), 1e-6) / 200)[:, 1]
    hist, edges = np.histogram(y, bins=bins)
    hist = np.convolve(hist, np.ones(3) / 3, "same")
    lo = hi = peak = hist.argmax()
    while hi + 1 < bins and hist[hi + 1] >= frac * hist[peak]:
        hi += 1
    while lo > 0 and hist[lo - 1] >= frac * hist[peak]:
        lo -= 1
    return float(edges[lo]), float(edges[hi + 1])


def baseline(strokes, text, xh):
    tall = xh * (1 if any(c.islower() for c in text) else 2)
    y = np.sort(densify(strokes, xh / 20)[:, 1])
    bottom = np.arange(y[0], y[-1] + xh / 40, xh / 40)
    ink = np.searchsorted(y, bottom, "right") - np.searchsorted(y, bottom - tall, "left")
    low = bottom[np.flatnonzero(ink >= 0.95 * ink.max())[-1]]
    return float(np.quantile(y[(y >= low - tall) & (y <= low)], 0.95))


SIZE = 224


INK_LEVEL = 200


MIN_ASPECT = 2.5


def flatten(rgb):
    red = np.asarray(rgb.convert("RGB"))[..., 0].astype(float)
    return (red / np.maximum(ndimage.grey_closing(red, size=(31, 31)), 1)).clip(0, 1)


def ink_mask(norm, weak=0.80, strong=0.55):
    labels, _ = ndimage.label(norm < weak)
    return np.isin(labels, np.unique(labels[norm < strong])) & (labels > 0)


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


def on_ink(strokes, img=None, radius=3, ctx=None):
    ink = ctx["touch"] if ctx else ndimage.binary_dilation(np.asarray(img.convert("L")) < INK_LEVEL, iterations=radius)
    h, w = ink.shape
    touches = lambda s: any(0 <= round(x) < w and 0 <= round(y) < h and ink[int(round(y)), int(round(x))] for x, y in s)
    return [s for s in strokes if touches(s)]


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


def ink_context(gray):
    ink = np.asarray(gray) < INK_LEVEL
    ctx = {"ink": ink, "touch": ndimage.binary_dilation(ink, iterations=3)}
    if not ink.any():
        return ctx
    gap, (ni, nj) = ndimage.distance_transform_edt(~ink, return_indices=True)
    labels, k = ndimage.label(ink, structure=np.ones((3, 3)))
    half = max(1.0, 2 * np.median(ndimage.distance_transform_edt(ink)[ink]) - 1)
    return {**ctx, "half": half, "gap": gap, "ni": ni, "nj": nj, "labels": labels, "k": k,
            "area": ndimage.sum(ink, labels, np.arange(1, k + 1))}


def drawn_area(pts, shape, half):
    raw = np.rint(pts).astype(int)
    inside = (raw[:, 0] >= 0) & (raw[:, 1] >= 0) & (raw[:, 0] < shape[1]) & (raw[:, 1] < shape[0])
    ij = np.clip(raw, 0, [shape[1] - 1, shape[0] - 1])
    canvas = np.zeros(shape, bool)
    canvas[ij[inside, 1], ij[inside, 0]] = True
    return ndimage.distance_transform_edt(~canvas) <= half + 1, inside, ij


def missed_marks(strokes, ctx, text):
    pts = resample_path(strokes) if strokes else np.zeros((0, 2))
    if "half" not in ctx or not len(pts):
        return []
    half, labels, k, area = ctx["half"], ctx["labels"], ctx["k"], ctx["area"]
    cover = ndimage.mean(drawn_area(pts, labels.shape, half)[0], labels, np.arange(1, k + 1))
    small = np.flatnonzero((area >= half ** 2) & (area < 32 * half ** 2))
    dots = sum(ch in "ij" or not ch.isalnum() for ch in text)
    out = []
    for i in small[np.argsort(-area[small])][:dots]:
        if cover[i] < RULES.min_mark:
            ys, xs = np.nonzero(labels == i + 1)
            p = np.c_[xs, ys].astype(float)
            m = p.mean(0)
            u = np.linalg.svd(p - m, full_matrices=False)[2][0]
            t = (p - m) @ u
            out.append([(m + t.min() * u).tolist(), (m + t.max() * u).tolist()])
    return out


def covered(q, args=None):
    args = args or RULES
    low = getattr(args, "min_coverage_read", args.min_coverage)
    return q["coverage"] >= args.min_coverage or (q["coverage"] >= low and q.get("read_error") is not None
                                                  and q["read_error"] <= getattr(args, "max_read_error_cov", 0.0))


def score(item, cand, ctx=None):
    c = ctx or ink_context(Image.open(item["image"]).convert("L"))
    ink, pts = c["ink"], resample_path(cand["strokes"])
    if not ink.any() or not len(pts):
        return {"coverage": 0.0, "precision": 0.0, "length_ratio": 0.0, "min_mark": 0.0, "max_mark_length": 0.0, "retrace": 0.0, "points": 0}
    half, gap, labels, k, area = c["half"], c["gap"], c["labels"], c["k"], c["area"]
    path, inside, ij = drawn_area(pts, ink.shape, half)
    marks, small = ndimage.mean(path, labels, np.arange(1, k + 1)), (area >= half ** 2) & (area < 32 * half ** 2)
    dots = sum(ch in "ij" or not ch.isalnum() for ch in item.get("text", ""))
    marks = np.concatenate([marks[area >= 32 * half ** 2], marks[small][np.argsort(-area[small])][:dots]])
    on = gap[ij[:, 1], ij[:, 0]] <= half / 2 + 2
    owner = labels[c["ni"][ij[on, 1], ij[on, 0]], c["nj"][ij[on, 1], ij[on, 0]]]
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
            "precision": float(((gap[ij[:, 1], ij[:, 0]] <= half / 2 + 2) & inside).mean()),
            "length_ratio": float(len(pts) * 2 * half / ink.sum()),
            "points": int(len(pts))}


def join_strokes(strokes, ctx):
    if not strokes or "half" not in ctx:
        return strokes
    half, near = ctx["half"], ctx["gap"] <= ctx["half"] / 2 + 2
    h, w = near.shape
    out = [np.asarray(strokes[0], float)]
    for s in map(lambda s: np.asarray(s, float), strokes[1:]):
        p, q = out[-1][-1], s[0]
        gap = np.hypot(*(q - p))
        seg = p + (q - p) * np.linspace(0, 1, max(2, int(gap) + 1))[:, None]
        ij = np.clip(np.rint(seg).astype(int), 0, [w - 1, h - 1])
        out[-1:] = [np.vstack([out[-1], s])] if gap <= 6 * half and near[ij[:, 1], ij[:, 0]].all() else [out[-1], s]
    return [s.tolist() for s in out]


def plain(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return text.lower().replace("0", "o")


def core(text):
    return "".join(ch for ch in plain(text) if ch.isalnum())


def clipped(read, text):
    r, t = core(read), core(text)
    return bool(r) and len(r) < len(t) and r in (t[1:], t[:-1])


RULES = argparse.Namespace(min_coverage=0.80, min_mark=0.5, min_precision=0.90, max_read_error=0.34, max_length_ratio=1.6,
                           max_retrace=0.35, max_mark_length=1.95, min_coverage_read=0.70, max_read_error_cov=0.15)


def judge(item, cand, args=RULES, ctx=None):
    ctx = ctx or ink_context(Image.open(item["image"]).convert("L"))
    c = {**cand, "strokes": on_ink(cand["strokes"], ctx=ctx)}
    added = missed_marks(c["strokes"], ctx, item.get("text", ""))
    c["strokes"] = [list(map(list, s)) for s in c["strokes"]] + added
    q = score(item, c, ctx)
    q["marks_added"] = len(added)
    label = plain(item["text"])
    q["read"] = c.get("read", "")
    q["read_error"] = levenshtein(plain(q["read"]), label) / max(1, len(label)) if q["read"] else None
    q["prompt"] = c.get("prompt", "")
    q["truncated"] = clipped(q["read"], item["text"])
    q["passed"] = (not q["truncated"] and covered(q, args) and q["precision"] >= args.min_precision
                   and q["min_mark"] >= args.min_mark and q["length_ratio"] <= args.max_length_ratio
                   and q["retrace"] <= args.max_retrace and q["max_mark_length"] <= args.max_mark_length
                   and (q["read_error"] is None or q["read_error"] <= args.max_read_error))
    return c, q


D, H, HD, L = 768, 12, 64, 12


TEXT_LEN, IMG_LEN, MAX_LEN = 128, 197, 1024


EOS, INK0, N_INK = 1, 19815, 12000


IMG_SEG, TXT_SEG = 2, 3


def rms(x, scale, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * scale


def bucket(rel, bidirectional, n_buckets=32, max_distance=128):
    n, ret = -rel, torch.zeros_like(rel)
    if bidirectional:
        n_buckets //= 2
        ret = ret + (n < 0).long() * n_buckets
        n = n.abs()
    else:
        n = n.clamp(min=0)
    exact = n_buckets // 2
    large = exact + (torch.log(n.float() / exact + 1.1920929e-07) / math.log(max_distance / exact) * (n_buckets - exact)).long()
    return ret + torch.where(n < exact, n, large.clamp(max=n_buckets - 1))


class InkSight:

    def __init__(self, path="inksight", device=None, dtype=torch.float32, kv_dtype=None):
        import sentencepiece
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        z = np.load(f"{path}/weights.npz")
        w = {k: torch.from_numpy(z[k]).to(device=device, dtype=dtype) for k in z.files}
        self.w, self.device, self.dtype = w, device, dtype
        self.kv_dtype = kv_dtype or {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, dtype)
        self.sp = sentencepiece.SentencePieceProcessor(model_file=f"{path}/spm.model")
        per_layer = lambda prefix, l: {k[len(prefix):]: v[:, l] for k, v in w.items() if k.startswith(prefix)}
        self.enc = [per_layer("encoder__encoder__", l) for l in range(L)]
        self.dec = [per_layer("decoder__decoder__", l) for l in range(L)]
        for wl in self.dec:
            wl["qkv"] = torch.cat([wl[f"self_attention__{n}__kernel"] for n in ("query", "key", "value")], 1)
        seg = torch.tensor([IMG_SEG] * IMG_LEN + [TXT_SEG] * TEXT_LEN, device=device)
        pos = torch.arange(IMG_LEN + TEXT_LEN, device=device)
        rel, same, pair = bucket(pos[None] - pos[:, None], True), (seg[:, None] == seg[None]).to(dtype), 16 * seg[:, None] + seg[None]
        self.enc_bias = [wl["relpos_bias__rel_embedding"][:, rel] * same + wl["multimodal_relpos_bias__rel_embedding"][:, pair]
                         for wl in self.enc]
        drel = bucket(-torch.arange(MAX_LEN, device=device), False)
        self.dec_bias = torch.stack([wl["relpos_bias__rel_embedding"][:, drel] for wl in self.dec])[:, None, :, None]
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    def vit(self, img):
        w, p = self.w, "vit_encoder__Transformer__"
        ln = lambda x, n: F.layer_norm(x.float(), (D,), w[f"{n}__scale"].float(), w[f"{n}__bias"].float(), 1e-6).to(self.dtype)
        x = F.conv2d(img, w["vit_encoder__embedding__kernel"].permute(3, 2, 0, 1), w["vit_encoder__embedding__bias"], stride=16)
        x = torch.cat([w["vit_encoder__cls"].expand(len(x), -1, -1), x.flatten(2).transpose(1, 2)], 1) + w[f"{p}posembed_input__pos_embedding"]
        for i in range(12):
            b = f"{p}encoderblock_{i}__"
            a = f"{b}MultiHeadDotProductAttention_0__"
            h = ln(x, f"{b}LayerNorm_0")
            q, k, v = (torch.einsum("btd,dhe->bhte", h, w[f"{a}{n}__kernel"]) + w[f"{a}{n}__bias"][:, None] for n in ("query", "key", "value"))
            x = x + torch.einsum("bhte,hed->btd", F.scaled_dot_product_attention(q, k, v), w[f"{a}out__kernel"]) + w[f"{a}out__bias"]
            h = F.gelu(ln(x, f"{b}LayerNorm_1") @ w[f"{b}MlpBlock_0__Dense_0__kernel"] + w[f"{b}MlpBlock_0__Dense_0__bias"], approximate="tanh")
            x = x + h @ w[f"{b}MlpBlock_0__Dense_1__kernel"] + w[f"{b}MlpBlock_0__Dense_1__bias"]
        return ln(x, f"{p}encoder_norm")

    def mlp(self, x, wl):
        h = rms(x, wl["pre_mlp_layer_norm__scale"])
        return x + (F.gelu(h @ wl["mlp__wi_0__kernel"], approximate="tanh") * (h @ wl["mlp__wi_1__kernel"])) @ wl["mlp__wo__kernel"]

    def heads(self, x, kernel):
        return (x @ kernel).view(len(x), -1, H, HD).transpose(1, 2)

    def encode(self, img, tokens):
        w, n = self.w, IMG_LEN + tokens.shape[1]
        x = torch.cat([self.vit(img) @ w["encoder__image_dense_embedder__embedders_image_dense__kernel"], w["token_embedder__embedding"][tokens]], 1)
        valid = torch.cat([torch.ones(len(x), IMG_LEN, dtype=torch.bool, device=x.device), tokens > 0], 1)
        mask = torch.zeros(valid.shape, device=x.device, dtype=self.dtype).masked_fill(~valid, -1e10)
        mask = mask[:, None, :, None] + mask[:, None, None, :]
        for wl, bias in zip(self.enc, self.enc_bias):
            h = rms(x, wl["pre_attention_layer_norm__scale"])
            q, k, v = (self.heads(h, wl[f"attention__{n}__kernel"]) for n in ("query", "key", "value"))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask + bias[:, :n, :n], scale=1.0)
            x = self.mlp(x + o.transpose(1, 2).reshape(x.shape) @ wl["attention__out__kernel"], wl)
        return rms(x, w["encoder__encoder_norm__scale"]), valid

    @torch.inference_mode()
    def generate(self, img, tokens, caps=None, max_len=MAX_LEN, graphs=None, kv_dtype=None, bucket_len=128, compact=True):
        w, B, dev = self.w, len(img), self.device
        graphs, kv_dtype = dev == "cuda" if graphs is None else graphs, kv_dtype or self.kv_dtype
        NEG = -3e4 if kv_dtype == torch.float16 else -1e10
        enc, valid = self.encode(img, tokens[:, :max(1, int((tokens > 0).sum(1).max()))])
        cross = [tuple(self.heads(enc, wl[f"encoder_decoder_attention__{n}__kernel"]).to(kv_dtype) for n in ("key", "value")) for wl in self.dec]
        cmask = torch.zeros(B, 1, 1, enc.shape[1], device=dev, dtype=kv_dtype).masked_fill(~valid[:, None, None], NEG)
        caps = torch.full((B,), max_len, device=dev) if caps is None else caps.to(dev).clamp(max=max_len)
        max_len = int(caps.max())
        kc = torch.zeros(L, B, H, max_len, HD, device=dev, dtype=kv_dtype)
        vc = torch.zeros_like(kc)
        tok, t = torch.zeros(B, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev)
        out, done = torch.zeros(B, max_len, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
        slots, rows = torch.arange(max_len, device=dev), torch.arange(B, device=dev)

        def step(n):
            x = w["token_embedder__embedding"][tok][:, None]
            d = t - slots[:n]
            bias = self.dec_bias[..., d.clamp(min=0)] + torch.zeros(n, device=dev, dtype=self.dtype).masked_fill(d < 0, NEG)
            for l, wl in enumerate(self.dec):
                h = rms(x, wl["pre_self_attention_layer_norm__scale"])
                q, k, v = (h[:, 0] @ wl["qkv"]).view(B, 3, H, HD).unbind(1)
                kc[l].index_copy_(2, t, k[:, :, None].to(kv_dtype))
                vc[l].index_copy_(2, t, v[:, :, None].to(kv_dtype))
                o = F.scaled_dot_product_attention(q[:, :, None].to(kv_dtype), kc[l, :, :, :n], vc[l, :, :, :n],
                                                   attn_mask=bias[l].to(kv_dtype), scale=1.0).to(self.dtype)
                x = x + o.transpose(1, 2).reshape(B, 1, D) @ wl["self_attention__out__kernel"]
                h = rms(x, wl["pre_cross_attention_layer_norm__scale"])
                o = F.scaled_dot_product_attention(self.heads(h, wl["encoder_decoder_attention__query__kernel"]).to(kv_dtype), *cross[l],
                                                   attn_mask=cmask, scale=1.0).to(self.dtype)
                x = self.mlp(x + o.transpose(1, 2).reshape(B, 1, D) @ wl["encoder_decoder_attention__out__kernel"], wl)
            nxt = (rms(x[:, 0], w["decoder__decoder_norm__scale"]) @ w["decoder__logits_dense__kernel"]).argmax(-1)
            nxt = torch.where(done, 0, torch.where(t + 1 >= caps, EOS, nxt))
            tok.copy_(nxt)
            out.index_put_((rows, t.expand(B)), nxt)
            done.logical_or_(nxt == EOS)
            t.add_(1)

        recorded, pool = {}, torch.cuda.graph_pool_handle() if graphs else None
        for s in range(max_len):
            n = min(max_len, (s // bucket_len + 1) * bucket_len) if graphs or dev != "cuda" else s + 1
            if not graphs:
                step(n)
            elif n in recorded:
                recorded[n].replay()
            else:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    step(n)
                torch.cuda.current_stream().wait_stream(side)
                recorded[n] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(recorded[n], pool=pool):
                    step(n)
            if s % 16 == 15:
                keep = (~done).nonzero()[:, 0]
                if not len(keep):
                    break
                if compact and len(keep) <= 0.75 * B:
                    B, recorded = len(keep), {}
                    pool = torch.cuda.graph_pool_handle() if graphs else None
                    rows, tok, caps, done, cmask = rows[keep], tok[keep], caps[keep], done[keep], cmask[keep]
                    kc, vc, cross = kc[:, keep], vc[:, keep], [(k[keep], v[keep]) for k, v in cross]
        return out[:, :s + 1]

    def trace(self, images, texts, prompt="Derender the ink: {text}", **kw):
        padded = [scale_and_pad(im.convert("RGB")) for im in images]
        px = torch.from_numpy(np.stack([np.asarray(c, np.float32) for c, *_ in padded])).to(self.device)
        prompts = [prompt] * len(texts) if isinstance(prompt, str) else prompt
        ids = [(self.sp.encode(p.replace("{text}", t)) + [EOS])[:TEXT_LEN] for p, t in zip(prompts, texts)]
        tokens = torch.tensor([i + [0] * (TEXT_LEN - len(i)) for i in ids], device=self.device)
        rows = []
        caps = torch.tensor([min(MAX_LEN, 200 + 90 * len(t)) for t in texts])
        for out, (_, ratio, dx, dy) in zip(self.generate(((px - 127.5) / 127.5).permute(0, 3, 1, 2).to(self.dtype), tokens, caps, **kw).tolist(), padded):
            out = out[:out.index(EOS)] if EOS in out else out
            ink = "".join(f"<ink_token_{i - INK0}>" for i in out if INK0 <= i < INK0 + N_INK)
            read = self.sp.decode([i for i in out if 2 < i < INK0 - 100])
            rows.append(([[[(x - dx) / ratio, (y - dy) / ratio] for x, y in st] for st in detokenize(ink)], read))
        return rows


TEXT, PLAIN = "Derender the ink: {text}", "Recognize and derender."


MAX_SIDE, MAX_HEIGHT = 1024, 320


WEAK, WHITE = 0.8, 0.95


def shrink(img):
    s = min(1.0, MAX_SIDE / max(img.size), MAX_HEIGHT / img.height)
    return (img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS) if s < 1 else img), s


def variants(img):
    from skimage.morphology import skeletonize
    norm = flatten(img)
    if 0.3 < norm.min() < 1:
        norm = (norm - norm.min()) / (1 - norm.min())
    mask = ink_mask(norm, weak=WEAK)
    if not mask.any():
        raise ValueError("no ink")
    lo = np.percentile(norm[mask], 10)
    clean = np.where(mask, ((norm - lo) / max(WHITE - lo, 1e-3)).clip(0, 1), 1.0)
    width = np.median(ndimage.distance_transform_edt(mask)[mask])
    mono = ndimage.distance_transform_edt(~skeletonize(mask)) <= max(1.5, width)
    return {"raw": np.asarray(img), "clean": (clean * 255).astype(np.uint8), "mono": np.where(mono, 0, 255).astype(np.uint8)}


def trace_round(model, preps, cands, batch):
    out = [[] for _ in preps]
    jobs = sorted(((i, v, pr) for i in range(len(preps)) for v, pr in cands), key=lambda j: len(preps[j[0]]["row"]["text"]))
    for k in range(0, len(jobs), batch):
        chunk = jobs[k:k + batch]
        images = [Image.fromarray(preps[i]["variants"][v]).convert("RGB") for i, v, _ in chunk]
        traces = model.trace(images, [preps[i]["row"]["text"] for i, _, _ in chunk], [pr for _, _, pr in chunk])
        for (i, v, pr), (strokes, read) in zip(chunk, traces):
            out[i].append((v, pr, strokes, read))
    return out


def pick(job):
    clean, text, cands, best = job
    try:
        ctx = ink_context(clean)
        reads = [r for _, prompt, _, r in cands if prompt == PLAIN and r] + ([best["qc"]["read"]] if best and best.get("qc", {}).get("read") else [])
        label = min(reads, key=lambda r: levenshtein(plain(r), plain(text)), default="")
        for variant, prompt, strokes, read in cands:
            joined = join_strokes([s for s in strokes if len(s)], ctx)
            c, q = judge({"text": text}, {"strokes": joined, "read": read or label, "prompt": prompt}, ctx=ctx)
            key = (q["passed"], q["coverage"] * q["precision"])
            if best is None or key > tuple(best["key"]):
                best = {"key": key, "candidate": f"{variant}/{'text' if prompt == TEXT else 'plain'}", "strokes": c["strokes"],
                        "qc": {k: v for k, v in q.items() if k != "prompt"}}
        return best
    except Exception as e:
        return best or {"error": f"pick: {type(e).__name__}: {e}"}


MODEL, PRICE = "claude-sonnet-5-5", (2.0, 10.0)


PER, COLS, CELL_W, CELL_H, MIN_HEIGHT = 20, 2, 520, 120, 16


PROMPT = ("Each numbered cell holds one handwritten word, number or token, cropped from a scan. Transcribe each cell exactly "
          "as written: keep capitalization, digits and punctuation (commas, periods, quotes, apostrophes, parentheses, hyphens). "
          "Do not correct spelling or add punctuation that isn't there. If a cell is hard to read, give your best guess. "
          "Also rate each cell's legibility from 1 to 10 as source data for learning to write by hand: 10 = crisp, high-contrast "
          "ink with every stroke clear; 7 = clearly legible with minor blur or noise; 5 = readable but faint, blurry or smudged; "
          "3 = hard to read, strokes partly lost; 1 = illegible, cut off or not handwriting.")


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["cells"],
          "properties": {"cells": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["n", "text", "legibility"],
              "properties": {"n": {"type": "integer"}, "text": {"type": "string"}, "legibility": {"type": "integer"}}}}}}


def sheet(images):
    out = Image.new("RGB", (CELL_W * COLS, CELL_H * ((len(images) + COLS - 1) // COLS)), "white")
    d, font = ImageDraw.Draw(out), ImageFont.load_default(size=22)
    for k, img in enumerate(images):
        s = min((CELL_W - 60) / img.width, (CELL_H - 12) / img.height, 2.0)
        img = img.convert("RGB").resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        x, y = (k % COLS) * CELL_W, (k // COLS) * CELL_H
        d.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], outline=(200, 200, 200))
        d.text((x + 8, y + 8), str(k + 1), fill=(200, 0, 0), font=font)
        out.paste(img, (x + 50, y + (CELL_H - img.height) // 2))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


class Transcriber:

    def __init__(self, model=MODEL, max_cost=5.0, workers=6, client=None):
        if client is None:
            import anthropic
            load_env()
            client = anthropic.Anthropic()
        self.client, self.model, self.max_cost, self.workers = client, model, max_cost, workers
        self.spent, self.lock = 0.0, threading.Lock()

    def read(self, images):
        if self.spent > self.max_cost:
            raise RuntimeError(f"transcription stopped: ${self.spent:.2f} spent, cap ${self.max_cost:.2f}")
        data = base64.standard_b64encode(sheet(images)).decode()
        r = self.client.beta.messages.create(
            model=self.model, max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                                                   {"type": "text", "text": f"{PROMPT} There are {len(images)} cells."}]}])
        with self.lock:
            self.spent += r.usage.input_tokens / 1e6 * PRICE[0] + r.usage.output_tokens / 1e6 * PRICE[1]
        if r.stop_reason == "refusal":
            return [("", 0)] * len(images)
        text = next(b.text for b in r.content if b.type == "text")
        cells = {c["n"]: (c["text"].strip(), int(c["legibility"])) for c in json.loads(text)["cells"]}
        return [cells.get(k + 1, ("", 0)) for k in range(len(images))]

    def __call__(self, images):
        groups = [images[i:i + PER] for i in range(0, len(images), PER)]
        with ThreadPoolExecutor(self.workers) as ex:
            return [cell for group in ex.map(self.read, groups) for cell in group]
