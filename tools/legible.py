import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import pengpt as P
from strokes import to_strokes

READER = "microsoft/trocr-large-handwritten"


def picture(points, height=128, pad=24, width=5):
    from PIL import Image, ImageDraw
    strokes = to_strokes(points)
    if not strokes:
        return Image.new("RGB", (height, height), "white")
    p = np.concatenate(strokes)
    k = (height - 2 * pad) / max(np.ptp(p[:, 1]), 0.15)
    img = Image.new("RGB", (int(k * np.ptp(p[:, 0])) + 2 * pad, int(k * np.ptp(p[:, 1])) + 2 * pad), "white")
    d = ImageDraw.Draw(img)
    for s in strokes:
        pts = [tuple(q) for q in (s - p.min(0)) * k + pad]
        d.line(pts if len(pts) > 1 else pts * 2, fill="black", width=width, joint="curve")
    return img


class Reader:

    def __init__(self, device):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.device, self.processor = device, TrOCRProcessor.from_pretrained(READER)
        self.model = VisionEncoderDecoderModel.from_pretrained(READER).to(device).eval()

    @torch.inference_mode()
    def read(self, images, batch=32):
        out = []
        for k in range(0, len(images), batch):
            pixels = self.processor(images=images[k:k + batch], return_tensors="pt").pixel_values.to(self.device)
            out += self.processor.batch_decode(self.model.generate(pixels, max_new_tokens=32), skip_special_tokens=True)
        return out


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        last, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            last, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, last + (x != y))
    return row[-1]


def errors(read, texts):
    plain = lambda t: "".join(c for c in t.lower() if c.isalnum())
    read, texts = zip(*[(r, t) for r, t in zip(read, texts) if plain(t)])
    wrong = sum(distance(plain(r), plain(t)) for r, t in zip(read, texts)) / sum(len(plain(t)) for t in texts)
    return wrong, float(np.mean([plain(r) == plain(t) for r, t in zip(read, texts)]))


@torch.inference_mode()
def written(model, dataset, texts, sources, greedy, seed, batch=64):
    st, device, out = dataset.stroke_tok, next(model.parameters()).device, []
    for k in range(0, len(texts), batch):
        told = torch.from_numpy(np.stack([dataset.encode_text(t) for t in texts[k:k + batch]])).to(device)
        start = torch.full((len(told), 1), st.BOS, device=device)
        tokens = model.generate(start, told, dataset.cfg.max_seq_length - 1, do_sample=not greedy, end_token=st.END,
                                pad_token=st.PAD, source=torch.as_tensor(sources[k:k + batch], device=device),
                                generator=torch.Generator(device).manual_seed(seed + k))
        out += [st.decode(t) for t in tokens[:, 1:].cpu().numpy()]
    return out


def words(test, each):
    rows = np.concatenate([d.indices[:each] for d in test.parts.values()])
    part = np.repeat(list(test.parts), [len(d.indices[:each]) for d in test.parts.values()])
    return rows, part, [test.bank_texts[i] for i in rows], [int(test.source_ids[i]) for i in rows]


def main():
    p = argparse.ArgumentParser(description="can what a handwriting model writes be read: held-out words written by each checkpoint and "
                                            "read back by TrOCR, as character error and words read exactly, beside the real ink")
    p.add_argument("checkpoints", nargs="+", help="checkpoints, or patterns of them")
    p.add_argument("--each", type=int, default=60, help="held-out words from each source")
    p.add_argument("--dataset", default="", help="read held-out words from these files, not the checkpoint's own")
    p.add_argument("--test", default="", help="score every word of these files, in place of the checkpoint's own held-out words")
    p.add_argument("--sampled", action="store_true", help="sample at temperature 1 in place of the most likely ink")
    p.add_argument("--fit_only", action="store_true", help="only the nats per pen move of the words: nothing is written or read")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/legible.md")
    args = p.parse_args()
    device = P.resolve_device("auto")
    reader = None if args.fit_only else Reader(device)
    paths = [q for pattern in args.checkpoints for q in sorted(glob.glob(pattern))]
    lines, real = ["| Run | Nats a pen move | Character error | Words read exactly | By source, character error |", "|---|---|---|---|---|"], {}
    for path in paths:
        model, _, cfg, ckpt = P.load_for_sampling(path, device)
        cfg.dataset, cfg.test = args.dataset or cfg.dataset, args.test or cfg.test
        test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
        rows, part, texts, sources = words(test, args.each)
        run, fit = os.path.basename(os.path.dirname(os.path.abspath(path))), P.evaluate(model, test, device)[1]
        if args.fit_only:
            lines.append(f"| {run} | {fit:.4f} | | | |")
            print(lines[-1], flush=True)
            continue
        scored = {run: written(model, test, texts, sources, not args.sampled, args.seed)}
        if cfg.dataset not in real:
            real[cfg.dataset] = True
            scored = {f"real ink ({len(test.parts)} sources)": [test.bank_points[i] for i in rows], **scored}
        for name, inks in scored.items():
            read = reader.read([picture(ink) for ink in inks])
            by = {s: errors([r for r, q in zip(read, part) if q == s], [t for t, q in zip(texts, part) if q == s])[0] for s in test.parts}
            wrong, exact = errors(read, texts)
            lines.append(f"| {name} | {f'{fit:.4f}' if name == run else ''} | {wrong:.3f} | {exact:.1%} | " + ", ".join(f"{s} {v:.3f}" for s, v in by.items()) + " |")
            print(lines[-1], flush=True)
    text = "\n".join(lines) + f"\n\n{args.each} held-out words a source, {'sampled' if args.sampled else 'most likely ink'}, read by {READER}; letters and digits only, either case.\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
