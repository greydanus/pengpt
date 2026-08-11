"""Zero-shot eval: generate held-out labels, CLIP-rank them against trained ones.

    python zeroshot.py --checkpoint out/x/best.pt --held wolf,whale --trained cat,dog,fish
"""

import argparse

import numpy as np
import torch

from pengpt.fastembed import render_array
from pengpt.model import load_for_sampling
from pengpt.sampling import generate, SampleParams
from train import resolve_device


def clip_rank(images, candidate_labels, device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    texts = [f"a sketch of a {l}" for l in candidate_labels]
    with torch.inference_mode():
        t = proc(text=texts, return_tensors="pt", padding=True).to(device)
        tf = model.get_text_features(**t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        rgb = [np.stack([im] * 3, axis=-1) for im in images]
        p = proc(images=rgb, return_tensors="pt").to(device)
        imf = model.get_image_features(**p)
        imf = imf / imf.norm(dim=-1, keepdim=True)
    return (imf @ tf.T).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--held", required=True)
    p.add_argument("--trained", required=True)
    p.add_argument("--per_label", type=int, default=8)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, dataset, cfg, _ = load_for_sampling(args.checkpoint, device, n_examples=50)
    held = [l.strip() for l in args.held.split(",") if l.strip()]
    trained = [l.strip() for l in args.trained.split(",") if l.strip()]
    candidates = held + trained

    params = SampleParams(max_tokens=cfg.max_seq_length - 1)
    images, owners = [], []
    for label in candidates:
        for k in range(args.per_label):
            torch.manual_seed(31 * k + 7)
            words = generate(model, dataset, label, params)
            pts = np.vstack(words) if words else np.zeros((0, 3))
            images.append(render_array(pts, px=112))
            owners.append(label)

    sims = clip_rank(images, candidates, device)
    print(f"{'label':16s} {'top1':>5s} {'rank':>6s}")
    for group, labels in (("HELD-OUT", held), ("trained", trained)):
        top1s, ranks = [], []
        for label in labels:
            li = candidates.index(label)
            rows = [i for i, o in enumerate(owners) if o == label]
            order = np.argsort(-sims[rows], axis=1)
            rank = np.mean([list(o).index(li) + 1 for o in order])
            top1 = np.mean([o[0] == li for o in order])
            top1s.append(top1)
            ranks.append(rank)
            print(f"{label:16s} {top1:5.0%} {rank:6.1f}/{len(candidates)}")
        print(f"-- {group}: top1 {np.mean(top1s):.0%}, "
              f"mean rank {np.mean(ranks):.1f} (chance {(len(candidates)+1)/2:.1f})\n")


if __name__ == "__main__":
    main()
