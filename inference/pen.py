import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

DIRECTIONS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
DOWN, UP, N_BASE = 8, 9, 10
EPS = np.finfo(np.float32).eps
REPO, RELEASE = "greydanus/pengpt", "models-2026-11-16"
FORMAT = 2
FETCHED = []


def rms(x):
    return x / np.sqrt(np.einsum("...i,...i->...", x, x)[..., None] / x.shape[-1] + EPS)


def rows(matrix, vectors):
    return np.stack([matrix @ v for v in vectors])


def strokes(points):
    edges = np.diff(np.concatenate([[0], points[:, 2], [0]]))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def corners(s, tol):
    keep, todo = {0, len(s) - 1}, [(0, len(s) - 1)]
    while todo:
        i, j = todo.pop()
        if j - i < 2:
            continue
        d, p = s[j] - s[i], s[i + 1:j] - s[i]
        L2 = d[0] * d[0] + d[1] * d[1]
        t = np.clip((p[:, 0] * d[0] + p[:, 1] * d[1]) / L2, 0, 1) if L2 else np.zeros(len(p))
        ex, ey = p[:, 0] - t * d[0], p[:, 1] - t * d[1]
        far = np.sqrt(ex * ex + ey * ey)
        k = int(far.argmax())
        if far[k] > tol:
            keep.add(i + 1 + k)
            todo += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(keep)


def chaikin(c, closed=False, times=4):
    for _ in range(times):
        if len(c) < 3:
            break
        a, b = (c, np.roll(c, -1, 0)) if closed else (c[:-1], c[1:])
        cut = np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], 1).reshape(-1, 2)
        if not closed:
            cut[0], cut[-1] = c[0], c[-1]
        c = cut
    return np.concatenate([c, c[:1]]) if closed else c


def lengths(c):
    d = np.diff(c, axis=0)
    return np.concatenate([[0], np.cumsum(np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]))])


def resample(c, n):
    L = lengths(c)
    t = L[-1] * np.arange(n + 1) / n
    i = np.minimum(np.searchsorted(L[1:], t), len(c) - 2)
    span = L[i + 1] - L[i]
    f = np.where(span > 0, np.minimum(1, (t - L[i]) / np.where(span > 0, span, 1)), 0)
    return c[i] + f[:, None] * (c[i + 1] - c[i])


def curve(s, grid):
    u = s[np.concatenate([[True], (np.diff(s, axis=0) != 0).any(1)])]
    if not grid or len(u) < 3:
        return s
    c = u[corners(u, grid / 2)]
    m = len(c)
    a, b = c[1:-1] - c[:-2], c[2:] - c[1:-1]
    sharp = lambda a, b: a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] <= -0.5 * np.sqrt((a[..., 0] * a[..., 0] + a[..., 1] * a[..., 1]) * (b[..., 0] * b[..., 0] + b[..., 1] * b[..., 1]))
    cuts = [0, *(1 + np.flatnonzero(sharp(a, b))), m - 1]
    gap = c[-1] - c[0]
    if m > 4 and len(cuts) == 2 and np.sqrt(gap[0] * gap[0] + gap[1] * gap[1]) < grid and not sharp(c[0] - c[-2], c[1] - c[0]):
        pieces = [chaikin(c[:-1], True)]
    else:
        pieces = [chaikin(c[i:j + 1]) for i, j in zip(cuts[:-1], cuts[1:])]
    ends = np.cumsum([lengths(q)[-1] for q in pieces])
    if not ends[-1]:
        return s
    n = np.maximum(np.diff(np.concatenate([[0], np.floor(ends / ends[-1] * (len(s) - 1) + 0.5)])).astype(int), 1)
    while n.sum() > len(s) - 1:
        n[n.argmax()] -= 1
    return np.concatenate([resample(q, k)[1 if i else 0:] for i, (q, k) in enumerate(zip(pieces, n))])


def curves(points, grid):
    out = np.array(points, float).reshape(-1, 3)
    edges = np.diff(np.r_[0, out[:, 2] == 1, 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b, :2] = curve(out[a:b, :2], grid)
    return out


def export(checkpoint, out, bits=8):
    import torch
    c = torch.load(checkpoint, map_location="cpu", weights_only=True)
    m, d = c["model_config"], c["data_config"]
    assert DIRECTIONS.tolist() == c["model"]["pen_deltas"][:8].tolist()
    assert not m.get("pen_pos_word") and not m.get("vision") and m.get("context_slots", 0) <= 1 and (m.get("context_slots") or not m.get("context_dim"))
    arrays = {}
    for name, w in c["model"].items():
        b = bits.get(name, 8) if isinstance(bits, dict) else bits
        name, w = name.replace("source_emb.", "writer_emb."), w.numpy()
        if w.ndim == 2 and w.dtype == np.float32:
            scale = np.maximum(np.abs(w).max(1, keepdims=True), 1e-12) / (2 ** (b - 1) - 1)
            arrays[name], arrays[name + ".scale"] = np.rint(w / scale).astype(np.int8), scale.astype(np.float32)
        elif name != "pen_deltas":
            arrays[name] = w
    meta = dict(format=2 if m.get("context_slots") else 1, vector=m.get("context_width", 0), vectors=d.get("vectors", ""), n_layer=m["n_layer"], n_head=m["n_head"], bands=m["pen_pos_bands"], block_size=m["block_size"],
                pen_rotary=bool(m.get("pen_rotary", False)),
                grid=d["grid"], max_text_length=d["max_text_length"], text_end=bool(d.get("text_end", False)),
                max_seq_length=d["max_seq_length"], step=c["step"], label_drop=float(d.get("label_drop", 0)), hint=float(d.get("hint", 0)))
    np.savez_compressed(out, meta=json.dumps(meta), merges=np.array(c["merges"], dtype=np.int16).reshape(-1, 3),
                        writers=np.array(c.get("sources", c.get("writers"))), **arrays)


def fetch(path, name=None):
    if not os.path.exists(path):
        name = name or os.path.basename(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not FETCHED:
            print("model weights are missing, so this run downloads them first: expect 30 to 60 seconds more, depending on your connection")
        FETCHED.append(name)
        print(f"{path} not found: downloading it from {REPO} release {RELEASE}")
        try:
            urllib.request.urlretrieve(f"https://github.com/{REPO}/releases/download/{RELEASE}/{name}", f"{path}.part")
        except OSError:
            subprocess.run(["gh", "release", "download", RELEASE, "--repo", REPO, "--pattern", name, "--output", f"{path}.part", "--clobber"], check=True)
        os.replace(f"{path}.part", path)
    return path


class Pen:

    def __init__(self, path):
        z = np.load(fetch(path))
        self.pen_rotary, self.label_drop, self.hint, self.format, self.vector, self.encoder = False, 0.0, 0.0, 1, 0, None
        self.__dict__.update(json.loads(str(z["meta"])))
        if self.format > FORMAT:
            raise SystemExit(f"{path} is in format {self.format} and this file reads up to format {FORMAT}; use the pen.py that came with the model")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        self.w = {k: z[k].astype(np.float32) * z[k + ".scale"] if k + ".scale" in z else z[k].astype(np.float32)
                  for k in z.files if k not in ("meta", "merges", "writers") and not k.endswith(".scale")}
        self.writers, self.merges = [str(x) for x in z["writers"]], z["merges"].tolist()
        n = N_BASE + len(z["merges"])
        self.PAD, self.END, self.WORD, self.BOS = n, n + 1, n + 2, n + 3
        self.paths = [DIRECTIONS[i:i + 1] for i in range(8)] + [np.zeros((0, 2), int)] * (n + 4 - 8)
        for a, b, merged in z["merges"]:
            self.paths[merged] = np.concatenate([self.paths[a], self.paths[b]])
        self.deltas = np.array([p.sum(0) for p in self.paths])
        self.paths = [p.cumsum(0) for p in self.paths]
        self.hd = self.w["tok_emb.weight"].shape[1] // self.n_head
        angles = np.outer(np.arange(self.block_size), 10_000 ** -(np.arange(0, self.hd, 2) / self.hd)).astype(np.float32)
        self.cos, self.sin = np.cos(angles), np.sin(angles)
        self.freq = (np.pi / 2 ** (np.arange(self.bands) + 1)).astype(np.float32)
        n = self.hd // 4
        m = (self.hd // 2 - n) // 2
        self.seq_freq = (10_000 ** -(np.arange(n) / n)).astype(np.float32)
        self.pen_freq = (2 * np.pi / (4.0 * 128.0 ** (np.arange(m) / max(m - 1, 1)))).astype(np.float32)
        self.layers = [{k: self.w[f"blocks.{i}.{k}.weight"] for k in
                        ("attn.q", "attn.kv", "attn.proj", "attn.q_norm", "attn.k_norm", "cross_attn.q", "cross_attn.kv",
                         "cross_attn.proj", "cross_attn.q_norm", "cross_attn.k_norm", "fc", "fc_out")} for i in range(self.n_layer)]

    def heads(self, x):
        return x.reshape(-1, self.n_head, self.hd).swapaxes(0, 1)

    def attend(self, q, k, v):
        a = (k @ q[..., None])[..., 0] * self.hd ** -0.5
        a = np.exp(a - a.max(-1, keepdims=True))
        return ((a / a.sum(-1, keepdims=True))[..., None, :] @ v).reshape(*q.shape[:-2], -1)

    def told(self, text):
        if not isinstance(text, str):
            return np.asarray(text, np.float32)
        if not text:
            return np.zeros(self.vector, np.float32)
        if self.encoder is None:
            from words import Words
            self.encoder = Words()
        return self.encoder(text)

    def rotary(self, t, pen):
        if not self.pen_rotary:
            return self.cos[t], self.sin[t]
        angles = np.concatenate([t * self.seq_freq, (pen[:, None].astype(np.float32) * self.pen_freq).ravel()])
        return np.cos(angles), np.sin(angles)

    def tokens(self, text, writer="", prefix=(), temperature=0.0, top_k=None, rng=None):
        new, done = self.raw(text, writer, prefix, temperature, top_k, rng)
        words = [u[u != self.WORD] for u in np.split(new[:-1] if done else new, np.flatnonzero((new[:-1] if done else new) == self.WORD))]
        return [u for u in words if len(u)][:None if done else -1]

    def raw(self, text, writer="", prefix=(), temperature=0.0, top_k=None, rng=None, guidance=None, ink=None, strokes=None):
        w, C, H, hd = self.w, self.w["tok_emb.weight"].shape[1], self.n_head, self.hd
        guidance = (2.0 if self.label_drop else 1.0) if guidance is None else guidance
        if guidance != 1 and not self.label_drop:
            raise ValueError("guidance needs a model trained with --label_drop; this one was not")
        v = w["writer_emb.weight"][self.writers.index(writer)] if isinstance(writer, str) else np.asarray(writer, np.float32)

        asked = [None if n is None else int(np.clip(n, 0, 99)) for n in (None if ink is None else round(2 * ink), strokes)] if self.hint else []
        hints = [(self.max_text_length + self.text_end - 4 + k, 258 + 10 * k + d) for k, d in
                 enumerate(d for n in asked for d in ((None, None) if n is None else divmod(n, 10))) if d is not None]

        def context(text):
            if self.vector:
                told, flags = self.told(text), np.zeros(40, np.float32)
                flags[[i - 258 for _, i in hints]] = 1
                c = np.concatenate([rms(v)[None], *[rms((w["ctx_proj.weight"] @ told + w["ctx_slot.weight"][0]).reshape(4, -1))] * bool(told.any()),
                                    *[rms(w["hint_proj.weight"] @ flags)[None]] * bool(flags.any())])
            else:
                ids = [b + 1 for b in text.encode()][:self.max_text_length - 4 * bool(self.hint)] + [257] * self.text_end
                at, ids = [*range(len(ids)), *(p for p, _ in hints)], ids + [i for _, i in hints]
                c = np.concatenate([rms(v)[None], rms(w["ctx_emb.weight"][ids] + w["ctx_pos_emb.weight"][at])])
            return [(rms(self.heads(kv[:, :C])) * L["cross_attn.k_norm"], self.heads(kv[:, C:])) for L in self.layers
                    for kv in [c @ L["cross_attn.kv"].T]]

        seq = [self.BOS] + [int(t) for word in prefix for t in [*word, self.WORD, self.WORD]]
        n = min(self.max_seq_length, self.block_size)
        cross = [context(t) for t in ([text] if guidance == 1 else [text, ""])]
        S = len(cross)
        K, V, pen = np.zeros((self.n_layer, S, H, n, hd), np.float32), np.zeros((self.n_layer, S, H, n, hd), np.float32), np.zeros(2, int)
        for t in range(n - 1):
            pen = pen + self.deltas[seq[t]]
            angles = pen[:, None].astype(np.float32) * self.freq
            x0 = rms(w["tok_emb.weight"][seq[t]] + (w["pen_pos_proj.weight"] @ np.concatenate([np.sin(angles), np.cos(angles)], 1).ravel()
                                                    if self.bands else 0))
            cos, sin = self.rotary(t, pen)
            x = np.repeat(x0[None], S, 0)
            for i, L in enumerate(self.layers):
                x = w["resid_scale"][i] * x + w["x0_scale"][i] * x0
                a = rms(x)
                kv = rows(L["attn.kv"], a)
                q, k = rms(rows(L["attn.q"], a).reshape(S, H, hd)) * L["attn.q_norm"], rms(kv[:, :C].reshape(S, H, hd)) * L["attn.k_norm"]
                q, k = (np.concatenate([u[..., :hd // 2] * cos + u[..., hd // 2:] * sin, u[..., hd // 2:] * cos - u[..., :hd // 2] * sin], -1)
                        for u in (q, k))
                K[i, :, :, t], V[i, :, :, t] = k, kv[:, C:].reshape(S, H, hd)
                x = x + rows(L["attn.proj"], self.attend(q, K[i, :, :, :t + 1], V[i, :, :, :t + 1]))
                q = rms(rows(L["cross_attn.q"], rms(x)).reshape(S, H, hd)) * L["cross_attn.q_norm"]
                x = x + np.stack([L["cross_attn.proj"] @ self.attend(q[s], *cross[s][i]) for s in range(S)])
                x = x + rows(L["fc_out"], np.maximum(rows(L["fc"], rms(x)), 0) ** 2)
            if t < len(seq) - 1:
                continue
            logits = 15 * np.tanh(rows(w["head.weight"], rms(x)) / 15)
            logits = logits - logits.max(1, keepdims=True)
            logits = logits - np.log(np.exp(logits).sum(1, keepdims=True))
            mixed = logits[0] if S == 1 else logits[1] + guidance * (logits[0] - logits[1])
            if temperature > 0:
                mixed = mixed / temperature
                if top_k:
                    mixed[mixed < np.sort(mixed)[-top_k]] = -np.inf
                p = np.exp(mixed - mixed.max())
                seq.append(int((rng or np.random.default_rng()).choice(len(p), p=p / p.sum())))
            else:
                seq.append(int(mixed.argmax()))
            if seq[-1] == self.END:
                break
        return np.array(seq[1 + sum(len(word) + 2 for word in prefix):], dtype=int), seq[-1] == self.END

    def decode(self, tokens):
        points, xy, down = [], np.zeros(2, int), False
        for t in tokens:
            if t == DOWN:
                down = True
                points.append([*xy, 1])
            elif t == UP:
                points += [[*xy, 0]] * down
                down = False
            elif len(self.paths[t]):
                points += [[*p, 1] for p in xy + self.paths[t]] * down
                xy = xy + self.paths[t][-1]
        out = np.array(points, dtype=float).reshape(-1, 3)
        out[:, :2] *= self.grid
        for a, b in strokes(out):
            if b - a >= 3:
                out[a + 1:b - 1, :2] = (out[a:b - 2, :2] + out[a + 1:b - 1, :2] + out[a + 2:b, :2]) / 3
        return curves(out, self.grid)

    def draw(self, text, source="", temperature=0.6, top_k=None, rng=None, guidance=None, ink=None, strokes=None):
        new, _ = self.raw(text, source, (), temperature, top_k, rng, guidance, ink, strokes)
        return self.decode(new[new < self.PAD])

    def write(self, text, writer="", temperature=0.0, top_k=None, seed=0, n_at_a_time=2):
        rng, texts, words, prior, prior_tokens = np.random.default_rng(seed), text.split(), [], [], []
        for i in range(0, len(texts), n_at_a_time):
            chunk = texts[i:i + n_at_a_time]
            if len(" ".join(prior + chunk).encode()) > self.max_text_length:
                prior, prior_tokens = [], []
            tokens = self.tokens(" ".join(prior + chunk), writer, prior_tokens, temperature, top_k, rng)
            for word in chunk[len(tokens):]:
                tokens += self.tokens(word, writer, (), temperature, top_k, rng)[:1] or [np.zeros(0, int)]
            words += [self.decode(t) for t in tokens[:len(chunk)]]
            prior, prior_tokens = (chunk[-1:], tokens[:len(chunk)][-1:]) if all(len(t) for t in tokens[:len(chunk)]) else ([], [])
        return words


def layout(words, line_width=8.0, line_height=0.55, space=0.16):
    placed, x, y = [], 0.0, 0.0
    for p in words:
        if not len(p):
            continue
        p = p - [p[:, 0].min(), 0, 0]
        if x > 0 and x + p[:, 0].max() > line_width:
            x, y = 0.0, y + line_height
        placed.append(p + [x, y, 0])
        x += p[:, 0].max() + space
    return placed


def sheet(drawings, cols, gap=0.3):
    placed = []
    for k, p in enumerate(drawings):
        if len(p):
            centre = (p[:, :2].min(0) + p[:, :2].max(0)) / 2
            placed.append(p - [*centre, 0] + [(k % cols) * (1 + gap), (k // cols) * (1 + gap), 0])
    return placed


def svg(words, path, scale=160, pad=0.3, stroke=2.2, color="#1a237e"):
    pts = np.concatenate(words)
    lo, hi = pts[:, :2].min(0) - pad, pts[:, :2].max(0) + pad
    lines = []
    for p in words:
        for a, b in strokes(p):
            xy = (p[a:b, :2] - lo) * scale
            xy = xy if b - a > 1 else np.concatenate([xy, xy + 0.01])
            lines.append('<path d="M' + " L".join(f"{u:.1f} {v:.1f}" for u, v in xy) + '"/>')
    w, h = (hi - lo) * scale
    open(path, "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">'
                          f'<rect width="100%" height="100%" fill="white"/><g fill="none" stroke="{color}" stroke-width="{stroke}" '
                          f'stroke-linecap="round" stroke-linejoin="round">' + "".join(lines) + "</g></svg>")


def main():
    p = argparse.ArgumentParser(description="a small pen: numpy only, and nothing from the rest of this repository. `write` writes text by hand and "
                                            "`draw` draws labels, each with a packed model, and saves an SVG; `export` packs a checkpoint")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", default="hand.npz")
    e.add_argument("--bits", type=int, default=8, help="8 for a handwriting model, 6 for a drawing model")
    w = sub.add_parser("write")
    w.add_argument("text", nargs="?", default="The quick brown fox jumps over the lazy dog")
    w.add_argument("--model", default="hand.npz", help="downloaded from the GitHub release if it is not there")
    w.add_argument("--writer", default="", help="a known writer, or a style vector saved by hand.py (me.npy)")
    w.add_argument("--temperature", type=float, default=None, help="0 writes the most likely ink (the default, unless the style file carries its own)")
    w.add_argument("--writers", action="store_true", help="list the writers and exit")
    w.add_argument("--out", default="hand.svg")
    d = sub.add_parser("draw")
    d.add_argument("text", nargs="?", default="cat", help="labels separated by |")
    d.add_argument("--model", default="sketch.npz", help="downloaded from the GitHub release if it is not there")
    d.add_argument("--n", type=int, default=6, help="drawings of each label")
    d.add_argument("--source", default="", help="draw in the manner of this source; empty for any")
    d.add_argument("--temperature", type=float, default=0.6, help="0 draws the most likely strokes")
    d.add_argument("--guidance", type=float, default=None, help="how hard a drawing is pushed towards its label; 1 is not at all")
    d.add_argument("--ink", type=float, default=None, help="ask for this much ink, in lengths of the longer side")
    d.add_argument("--strokes", type=int, default=None, help="ask for this many strokes")
    d.add_argument("--sources", action="store_true", help="list the sources and exit")
    d.add_argument("--out", default="sketch.svg")
    for q in (w, d):
        q.add_argument("--top_k", type=int, default=None)
        q.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.command == "export":
        export(args.checkpoint, args.out, args.bits)
        return print(f"-> {args.out}")
    pen, t = Pen(args.model), time.time()
    if getattr(args, "writers", False) or getattr(args, "sources", False):
        return print("\n".join(pen.writers[1:]))
    if args.command == "draw":
        rng = np.random.default_rng(args.seed)
        drawings = [pen.draw(text, args.source, args.temperature, args.top_k, rng, args.guidance, args.ink, args.strokes)
                    for text in args.text.split("|") for _ in range(args.n)]
        svg(sheet(drawings, args.n), args.out, stroke=1.6, color="#111")
        return print(f"{len(drawings)} drawings in {time.time() - t:.1f}s -> {args.out}")
    if args.writer.endswith(".npy"):
        style = np.load(args.writer)
        args.writer, args.top_k = style[:-1], args.top_k or 5
        args.temperature = args.temperature if args.temperature is not None else float(style[-1])
    words = pen.write(args.text, args.writer, args.temperature or 0.0, args.top_k, args.seed)
    svg(layout(words), args.out)
    print(f"{len(words)} words in {time.time() - t:.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
