import argparse
import time

import torch

import pengpt
from pengpt import Muon


def main():
    p = argparse.ArgumentParser(description="where one training step's time goes on this GPU: data, forward and backward, clipping, each optimizer")
    pengpt.add_config_args(p)
    p.add_argument("--measure", type=int, default=150)
    p.add_argument("--warm", type=int, default=60)
    args = p.parse_args()
    data_cfg, model_cfg, train_cfg = pengpt.configs_from_args(args)
    train, _, tok = pengpt.create_datasets(data_cfg)
    torch.set_float32_matmul_precision("high")
    model = pengpt.PenTransformer(pengpt.fit(model_cfg, train)).cuda()
    pengpt.attach_pen_deltas(model, tok)
    opts = pengpt.make_optimizers(model, train_cfg)
    step_model = torch.compile(model) if train_cfg.compile else model
    batches = pengpt.infinite_batches(train, train_cfg.batch_size, workers=train_cfg.loader_workers, tokens=train_cfg.batch_tokens,
                                         step=train_cfg.pad)
    t, real, padded, drawn = {}, 0, 0, 0

    def lap(name, t0, on):
        torch.cuda.synchronize()
        if on:
            t[name] = t.get(name, 0) + time.time() - t0
        return time.time()

    seen, k, n = set(), 0, 0
    while k < args.measure:
        t0 = time.time()
        X, C, Y, W = next(batches)
        n += 1
        on = X.shape[1] in seen and n > args.warm
        seen.add(X.shape[1])
        t0 = lap("data", t0, on)
        X, C, Y, W = (a.cuda() for a in (X, C, Y, W))
        t0 = lap("to gpu", t0, on)
        with torch.autocast("cuda", torch.bfloat16, enabled=train_cfg.mixed):
            _, loss = step_model(X, C, Y, W)
        t0 = lap("forward", t0, on)
        model.zero_grad(set_to_none=True)
        loss.backward()
        t0 = lap("backward", t0, on)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        t0 = lap("clip", t0, on)
        for opt in opts:
            opt.step()
            t0 = lap("muon" if isinstance(opt, Muon) else "adamw", t0, on)
        if on:
            k, real, padded, drawn = k + 1, real + int((X != tok.PAD).sum()), padded + X.numel(), drawn + len(X)
    total = sum(t.values())
    print(f"batch {train_cfg.batch_size}, batch tokens {train_cfg.batch_tokens}, block {data_cfg.max_seq_length}, pad {train_cfg.pad}, mixed "
          f"{train_cfg.mixed}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters: {1000 * total / k:.1f} ms/step, {drawn / k:.0f} "
          f"drawings/step, {1000 * total / drawn:.3f} ms/drawing, {real / total / 1e3:,.0f}k tokens/s, {100 * (1 - real / padded):.0f}% of the batch "
          f"is padding, {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
    print("  " + ", ".join(f"{n} {1000 * v / k:.1f}" for n, v in t.items()))


if __name__ == "__main__":
    main()
