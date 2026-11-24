import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))

SMALL = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def prompts(path):
    return [line.strip() for line in open(path, encoding="utf-8") if line.strip()]


def unit(a):
    return a / a.norm(dim=-1, keepdim=True)


def small(device=DEVICE):
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(SMALL), AutoModel.from_pretrained(SMALL).text_model.to(device)


def tokens(tok, texts, device=DEVICE):
    return tok([t.lower() for t in texts], padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)


@torch.inference_mode()
def label(args):
    from words import Words
    words, texts, out = Words(args.teacher), prompts(args.prompts), []
    model, remap = words.model.to(DEVICE).to(torch.float32 if DEVICE == "cpu" else torch.float16), words.remap.to(DEVICE)
    for k in range(0, len(texts), 64):
        ids = words.tokenizer([t.lower() for t in texts[k:k + 64]], padding="max_length", truncation=True, max_length=64, return_tensors="pt")["input_ids"]
        out.append(unit(model(input_ids=remap[ids.to(DEVICE)].long()).pooler_output.float()).cpu().numpy().astype(np.float16))
        if k % 6400 == 0:
            print(f"{k}/{len(texts)}", flush=True)
    np.save(args.out, np.concatenate(out))


def fit(args):
    texts, Y = prompts(args.prompts), torch.from_numpy(np.load(args.vectors).astype(np.float32))
    texts, held = texts[:len(Y)], max(1, len(Y) // 20)
    mu, (tok, enc) = Y[:-held].mean(0), small()
    head = torch.nn.Linear(enc.config.hidden_size, Y.shape[1]).to(DEVICE)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    told = lambda batch: head(enc(**tokens(tok, batch)).pooler_output) + mu.to(DEVICE)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": args.lr}, {"params": head.parameters(), "lr": 20 * args.lr}], weight_decay=0.01)
    start, rng = [g["lr"] for g in opt.param_groups], np.random.default_rng(0)
    for step in range(args.steps + 1):
        if step % 1000 == 0 or step == args.steps:
            enc.eval()
            with torch.inference_mode():
                P = unit(torch.cat([told(texts[len(Y) - held:][k:k + 256]).float().cpu() for k in range(0, held, 256)]))
            enc.train()
            near = (unit(P - mu) * unit(Y[-held:] - mu)).sum(1).mean()
            first = ((P @ Y[-held:].T).argmax(1) == torch.arange(held)).float().mean()
            print(f"step {step}: held-out prompts, cosine to the teacher with the mean taken away {near:.3f}, own vector nearest {first:.1%}", flush=True)
        if step == args.steps:
            break
        for g, lr in zip(opt.param_groups, start):
            g["lr"] = lr * min(1, (step + 1) / 200) * (0.5 + 0.5 * math.cos(math.pi * step / args.steps))
        rows = rng.integers(0, len(Y) - held, args.batch)
        p, y, m = told([texts[i] for i in rows]), Y[rows].to(DEVICE), mu.to(DEVICE)
        loss = (1 - F.cosine_similarity(p - m, y - m, dim=-1)).mean() + 10 * ((unit(p) - y) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*enc.parameters(), *head.parameters()], 1.0)
        opt.step()
    torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "mu": mu}, args.out)


def pack(args):
    c = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tok, enc = small("cpu")
    t = json.loads(tok.backend_tokenizer.to_str())["model"]
    vocab = [token for token, _ in sorted(t["vocab"].items(), key=lambda kv: kv[1])]
    merges = [m if isinstance(m, str) else " ".join(m) for m in t["merges"]]
    arrays = {}
    for k, w in {**c["enc"], "head.weight": c["head"]["weight"], "head.bias": c["head"]["bias"] + c["mu"]}.items():
        w = w.float().numpy()
        if w.ndim == 2:
            scale = np.maximum(np.abs(w).max(1, keepdims=True), 1e-12) / (2 ** ((args.table if "token_embedding" in k else args.bits) - 1) - 1)
            arrays[k], arrays[k + ".scale"] = np.rint(w / scale).astype(np.int8), scale.astype(np.float32)
        elif "position_ids" not in k:
            arrays[k] = w.astype(np.float32)
    cfg = enc.config
    meta = dict(source=SMALL, width=cfg.hidden_size, n_layer=cfg.num_hidden_layers, n_head=cfg.num_attention_heads, positions=cfg.max_position_embeddings,
                bos=vocab.index("<|startoftext|>"), eos=vocab.index("<|endoftext|>"), eps=cfg.layer_norm_eps, vector=len(c["mu"]))
    packed = np.frombuffer(json.dumps({"vocab": vocab, "merges": merges}, ensure_ascii=False).encode(), np.uint8)
    np.savez_compressed(args.out, meta=json.dumps(meta), tokenizer=packed, **arrays)
    print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser(description="a small reader of typed words for the browser: a TinyCLIP text tower taught to give the SigLIP 2 vector a drawing model is "
                                            "told. label works out the big model's vectors for a file of prompts, a line each; fit teaches the small one to "
                                            "match them; pack writes the file web/words.js reads")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("label")
    a.add_argument("--prompts", required=True)
    a.add_argument("--teacher", default="siglip-text.npz", help="downloaded from the GitHub release if it is not there")
    a.add_argument("--out", default="out/words/vectors.npy")
    a = sub.add_parser("fit")
    a.add_argument("--prompts", required=True)
    a.add_argument("--vectors", default="out/words/vectors.npy")
    a.add_argument("--steps", type=int, default=6000)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--lr", type=float, default=5e-5)
    a.add_argument("--out", default="out/words/words.pt")
    a = sub.add_parser("pack")
    a.add_argument("--checkpoint", default="out/words/words.pt")
    a.add_argument("--bits", type=int, default=6)
    a.add_argument("--table", type=int, default=4, help="bits for the table of token vectors, most of the file")
    a.add_argument("--out", default="words.npz")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    {"label": label, "fit": fit, "pack": pack}[args.command](args)


if __name__ == "__main__":
    main()
