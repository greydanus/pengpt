import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import pengpt as P
from strokes import Strokes, grid, to_strokes
from vectors import encoder


def picture(strokes, px=224, width=3 / 224, pad=0.10, up=2):
    from PIL import Image, ImageDraw
    big = up * px
    img = Image.new("L", (big, big), 255)
    if not len(strokes):
        return np.asarray(img.resize((px, px)))
    p, d = np.concatenate(strokes), ImageDraw.Draw(img)
    k = big * (1 - 2 * pad) / max(np.ptp(p, 0).max(), 1e-6)
    off, w = (big - k * np.ptp(p, 0)) / 2 - k * p.min(0), max(1, round(big * width))
    for s in strokes:
        pts = [tuple(q) for q in np.asarray(s) * k + off]
        if len(pts) > 1:
            d.line(pts, fill=0, width=w, joint="curve")
        for x, y in (pts[0], pts[-1]):
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=0)
    return np.asarray(img.resize((px, px), Image.LANCZOS))


@torch.inference_mode()
def draw(model, dataset, texts, guidance, temperature, seed):
    st, device = dataset.stroke_tok, next(model.parameters()).device
    told = torch.from_numpy(np.stack([dataset.encode_text(t) for t in texts])).to(device)
    start = torch.full((len(texts), 1), st.BOS, device=device)
    out = model.generate(start, told, dataset.cfg.max_seq_length - 1, temperature=temperature, end_token=st.END,
                         pad_token=st.PAD, generator=torch.Generator(device).manual_seed(seed), guidance=guidance,
                         blank=torch.from_numpy(dataset.encode_text(""))[None])
    return [to_strokes(st.decode(t)) for t in out[:, 1:].cpu().numpy()]


def prompts(path, n, seed):
    if path.endswith(".txt"):
        return [line.strip() for line in open(path) if line.strip()][:n]
    f = Strokes(path).files[0]
    labels = sorted(set(f.tables["label"]))
    return [labels[i] for i in np.random.default_rng(seed).choice(len(labels), min(n, len(labels)), replace=False)]


def main():
    p = argparse.ArgumentParser(description="what guidance buys a drawing model: the same prompts drawn at several strengths, each drawing "
                                            "scored by how well SigLIP 2 matches its picture to its prompt")
    p.add_argument("--checkpoint", default="sketch.pt")
    p.add_argument("--guidance", default="1,1.5,2,3,4")
    p.add_argument("--dataset", default="data/quickdraw.sketch.parquet", help="the prompts are labels of this file, or the lines of a .txt file")
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--n", type=int, default=4, help="drawings of each prompt at each strength")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--shift", default="", help="a .npy vector added to each typed prompt's vector before the model is told it: the mean "
                                               "picture vector less the mean text vector, for a model trained on pictures alone")
    p.add_argument("--out", default="out/guidance")
    args = p.parse_args()
    device = P.resolve_device("auto")
    model, dataset, cfg, _ = P.load_for_sampling(args.checkpoint, device)
    strengths = [float(v) for v in args.guidance.split(",")]
    if cfg.label_drop <= 0:
        print("this model was trained without --label_drop, so it has nothing to be guided against: drawing at 1 only")
        strengths = [1.0]
    clip = P.Vectors.encoder = encoder()
    asked = prompts(args.dataset, args.prompts, args.seed)
    texts, own = [t for t in asked for _ in range(args.n)], np.repeat(np.arange(len(asked)), args.n)
    words = clip.texts(asked).astype(np.float32)
    if args.shift:
        moved = words + np.load(args.shift)
        dataset.told.typed.update(zip(asked, moved / np.linalg.norm(moved, axis=1, keepdims=True)))
    os.makedirs(args.out, exist_ok=True)
    lines = ["| Guidance | Cosine to its prompt | Its prompt ranked first | In the first five | Strokes |", "|---|---|---|---|---|"]
    for g in strengths:
        drawings = []
        for k in range(0, len(texts), args.batch):
            drawings += draw(model, dataset, texts[k:k + args.batch], g, args.temperature, args.seed + k)
        seen = clip.images(np.stack([picture(d) for d in drawings])).astype(np.float32)
        match = seen @ words.T
        rank = (match > match[np.arange(len(own)), own][:, None]).sum(1)
        lines.append(f"| {g:g} | {match[np.arange(len(own)), own].mean():.4f} | {(rank == 0).mean():.1%} | {(rank < 5).mean():.1%} | "
                     f"{np.mean([len(d) for d in drawings]):.1f} |")
        print(lines[-1], flush=True)
        shown = [i for i in range(len(texts)) if i % args.n == 0][:32]
        grid([drawings[i] for i in shown], [texts[i][:24] for i in shown], 8, os.path.join(args.out, f"guidance_{g:g}.png"))
    text = "\n".join(lines) + f"\n\n{len(asked)} prompts, {args.n} drawings of each, temperature {args.temperature:g}, {args.checkpoint}.\n"
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
