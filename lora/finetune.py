"""LoRA fine-tuning of the released handwriting model on one writer: data/me.hand.parquet -> out/me_lora/.

The base model (hand.pt) stays frozen. A new writer slot is added to its source embedding (started at
the mean of the 1,192 writers it knows), and every linear layer in its blocks (self-attention, cross-
attention and MLP: q, kv, proj, fc, fc_out) gets a low-rank adapter W + (alpha/r) B A, with B started
at zero so training begins from the released model exactly. Only the adapters and the new writer
vector train. --rank 0 trains the writer vector alone, which is the baseline to beat.

Data, tokenizer and augmentation are the repo's own (hand_datasets with the released BPE merges);
5% of the words are held out. Every --eval_every steps the held-out `word score` (nats a pen move,
the repo's number to compare runs by; lower is better) is measured and the best step is kept. At the
end the adapters are merged into the weights and saved as an ordinary checkpoint, so

    python pengpt.py sample --checkpoint out/me_lora/best.pt --writer me --text "..."
    python -m tools.pack --checkpoint out/me_lora/best.pt --out me.npz

work as for the released model.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pengpt as P  # noqa: E402

TARGETS = ("attn.q", "attn.kv", "attn.proj", "cross_attn.q", "cross_attn.kv", "cross_attn.proj", "fc", "fc_out")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.rank, self.scale = base, rank, alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        lin.weight.data = self.base.weight.data + self.scale * (self.B @ self.A).data
        return lin


def add_lora(model, rank, alpha):
    wrapped = []
    for block in model.blocks:
        for name in TARGETS:
            parent, attr = (block, name) if "." not in name else (getattr(block, name.split(".")[0]), name.split(".")[1])
            lin = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(lin, rank, alpha))
            wrapped.append((parent, attr))
    return wrapped


def merge_lora(wrapped):
    for parent, attr in wrapped:
        setattr(parent, attr, getattr(parent, attr).merged())


def add_writer(model, init):
    old = model.source_emb
    new = nn.Embedding(old.num_embeddings + 1, old.embedding_dim).to(old.weight.device)
    new.weight.data[:-1] = old.weight.data
    new.weight.data[-1] = init
    model.source_emb = new
    model.cfg.n_sources += 1
    return new.num_embeddings - 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="hand.pt")
    p.add_argument("--dataset", default="data/me.hand.parquet")
    p.add_argument("--writer", default="me")
    p.add_argument("--out", default="out/me_lora")
    p.add_argument("--rank", type=int, default=8, help="0: train the writer vector alone")
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="adapters")
    p.add_argument("--writer_lr", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=6, help="evaluations without a new best before stopping")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = P.resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, _, base_cfg, ckpt = P.load_for_sampling(args.base, device)
    cfg = base_cfg
    cfg.dataset, cfg.test, cfg.holdout, cfg.max_train = args.dataset, "", 0, 0
    sources = list(ckpt["sources"]) + [args.writer]
    train, test, tok = P.create_datasets(cfg, merges=ckpt["merges"], sources=sources)
    me = add_writer(model, model.source_emb.weight.data[1:].mean(0))
    assert sources[me] == args.writer

    def score(tag):
        loss, word = P.evaluate(model, test, device, n=200)
        return loss, word

    # baselines: the model told nothing about the writer (source 0), as released
    test_src = test.source_ids.copy()
    test.source_ids[:] = 0
    model.eval()
    _, base_word = score("base")
    test.source_ids[:] = test_src
    if hasattr(test, "_eval"):
        del test._eval
    print(f"base model, no writer: held-out word score {base_word:.4f}")

    for prm in model.parameters():
        prm.requires_grad_(False)
    wrapped = add_lora(model, args.rank, args.alpha) if args.rank > 0 else []
    model.to(device)
    lora_params = [prm for n, prm in model.named_parameters() if n.endswith((".A", ".B"))]
    for prm in lora_params:
        prm.requires_grad_(True)
    model.source_emb.weight.requires_grad_(True)
    mask = torch.zeros_like(model.source_emb.weight)
    mask[me] = 1
    model.source_emb.weight.register_hook(lambda g: g * mask)  # only the new writer's row moves
    groups = [dict(params=[model.source_emb.weight], lr=args.writer_lr, weight_decay=0.0)]
    if lora_params:
        groups.append(dict(params=lora_params, lr=args.lr, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    n_train = sum(prm.numel() for prm in lora_params) + model.source_emb.embedding_dim
    print(f"training {n_train:,} parameters ({n_train / sum(q.numel() for n, q in model.named_parameters() if not n.endswith(('.A', '.B'))):.1%} of the model): "
          f"rank {args.rank}, {len(train.indices)} train words, {len(test.indices)} held out")

    log = open(os.path.join(args.out, "metrics.jsonl"), "w")
    best, best_step, since, t0 = float("inf"), -1, 0, time.time()
    rng = np.random.default_rng(args.seed)
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            _, word = score("lora")
            better = word < best
            if better:
                best, best_step, since = word, step, 0
                state = {"lora": {n: prm.detach().cpu() for n, prm in model.named_parameters() if n.endswith((".A", ".B"))},
                         "writer": model.source_emb.weight[me].detach().cpu(), "step": step, "word_score": word, "args": vars(args)}
                torch.save(state, os.path.join(args.out, "adapter.pt"))
            else:
                since += 1
            print(f"step {step:4d}  held-out word score {word:.4f}{'  *' if better else ''}  ({time.time() - t0:.0f}s)", flush=True)
            log.write(json.dumps({"step": step, "word_score": word}) + "\n")
            log.flush()
            model.train()
            if since >= args.patience:
                print(f"no new best in {args.patience} evaluations: stopping")
                break
        if step == args.steps:
            break
        mult = min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * mult
        X, C, Y, W = P.trim(train.batch(rng.integers(0, 2**31, args.batch_size)), tok.PAD)
        W = torch.full_like(W, me)
        _, loss = model(X.to(device), C.to(device), targets=Y.to(device), source=W.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], 1.0)
        opt.step()

    # restore the best adapters, merge them into the weights, save an ordinary checkpoint
    state = torch.load(os.path.join(args.out, "adapter.pt"), weights_only=False)
    with torch.no_grad():
        named = dict(model.named_parameters())
        for n, v in state["lora"].items():
            named[n].copy_(v.to(device))
        model.source_emb.weight[me] = state["writer"].to(device)
    merge_lora(wrapped)
    model.to(device).eval()
    P.save_checkpoint(os.path.join(args.out, "best.pt"), model, asdict(cfg), merges=ckpt["merges"], step=state["step"],
                      best_loss=state["word_score"], sources=sources)
    summary = {"base_word_score": base_word, "best_word_score": best, "best_step": best_step, "rank": args.rank,
               "trained_params": n_train, "train_words": int(len(train.indices)), "test_words": int(len(test.indices))}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"best held-out word score {best:.4f} at step {best_step} (base {base_word:.4f}) -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
