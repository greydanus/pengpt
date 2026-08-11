"""Train a PenTransformer.

    python train.py --dataset data/bigbank_3500.json.zip --out_dir out/cursive
    python train.py --wandb --wandb_entity you --wandb_project pengpt

See pengpt/config.py for the full set of options and their defaults.
"""

import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from pengpt import (DataConfig, parse_configs, create_datasets, InfiniteDataLoader,
                    BucketedInfiniteLoader, PenTransformer, save_checkpoint,
                    load_checkpoint, save_samples, save_progress,
                    save_mixed_progress)


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=100, max_batches=10):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    losses = []
    for i, (X, C, Y) in enumerate(loader):
        _, loss = model(X.to(device), C.to(device), Y.to(device))
        losses.append(loss.item())
        if i + 1 >= max_batches:
            break
    model.train()
    return sum(losses) / len(losses)


def main():
    data_cfg, model_cfg, train_cfg = parse_configs(description="Train a pengpt model")
    device = resolve_device(train_cfg.device)
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_cfg.out_dir, "best.pt")

    # Resuming has to rebuild the *checkpoint's* tokenizer, not one re-derived
    # from the command line. Merges and alphabet decide what every token id
    # means, so re-learning them from a config that differs in --n_merges,
    # --grid, --dataset or --seed loads weights against a vocabulary they were
    # never trained on -- silently, whenever the sizes happen to still match.
    ckpt = None
    if train_cfg.resume:
        _, ckpt = load_checkpoint(train_cfg.resume, device)
        resumed_cfg = DataConfig(**ckpt["data_config"])
        if asdict(resumed_cfg) != asdict(data_cfg):
            changed = {k: (v, getattr(data_cfg, k))
                       for k, v in asdict(resumed_cfg).items()
                       if getattr(data_cfg, k) != v}
            print(f"Using the checkpoint's data config; ignoring {changed} "
                  f"(checkpoint value, command-line value)")
        data_cfg = resumed_cfg

    torch.manual_seed(data_cfg.seed)   # after resume, so it is the run's own seed

    text_encoder = None
    if data_cfg.text_encoder == "clip+char":
        text_encoder = "clip+char"
    elif data_cfg.text_encoder != "char":
        from pengpt.textenc import build_text_encoder
        text_encoder = build_text_encoder(data_cfg.text_encoder, device)
    train_dataset, test_dataset, stroke_tok, char_tok = create_datasets(
        data_cfg, merges=ckpt["merges"] if ckpt else None,
        text_encoder=text_encoder)
    if ckpt:
        assert char_tok.alphabet == ckpt["alphabet"], \
            "dataset alphabet does not match the checkpoint; resume needs its dataset"
    model_cfg.vocab_size = stroke_tok.vocab_size
    model_cfg.block_size = data_cfg.max_seq_length
    model_cfg.context_vocab_size = char_tok.vocab_size
    model_cfg.context_block_size = data_cfg.max_text_length
    model_cfg.context_dim = (train_dataset.text_encoder.dim
                             if train_dataset.text_encoder else 0)

    model = PenTransformer(model_cfg).to(device)
    if model_cfg.pen_pos_bands > 0:
        # The displacement table is a property of the tokenizer; a resume's
        # state dict re-loads the same values.
        model.pen_deltas.copy_(torch.tensor(stroke_tok.token_deltas(),
                                            device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate,
                                  weight_decay=train_cfg.weight_decay,
                                  betas=(0.9, 0.99), eps=1e-8)
    def lr_at(step):
        if step < train_cfg.warmup:
            return (step + 1) / train_cfg.warmup
        t = (step - train_cfg.warmup) / max(1, train_cfg.steps - train_cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    step, best_loss = 0, None
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best_loss = ckpt["step"], ckpt["best_loss"]
        print(f"Resumed from {train_cfg.resume} at step {step} (best loss {best_loss})")

    run = None
    if train_cfg.wandb:
        import wandb
        run = wandb.init(project=train_cfg.wandb_project,
                         entity=train_cfg.wandb_entity or None,
                         name=train_cfg.wandb_run_name or None,
                         config={**asdict(data_cfg), **asdict(model_cfg), **asdict(train_cfg)})

    if train_cfg.bucket_batches:
        loader = BucketedInfiniteLoader(train_dataset,
                                        batch_size=train_cfg.batch_size,
                                        seed=data_cfg.seed,
                                        pin_memory=(device == "cuda"),
                                        num_workers=train_cfg.num_workers)
    else:
        loader = InfiniteDataLoader(train_dataset, batch_size=train_cfg.batch_size,
                                    pin_memory=(device == "cuda"),
                                    num_workers=train_cfg.num_workers)
    pad = None
    if train_cfg.bucket_batches:
        pad = stroke_tok.PAD

    src_tests = {s: test_dataset.for_source(s) for s in test_dataset.sources()}
    if src_tests:
        print(f"per-source eval: {sorted(src_tests)}")

    while step < train_cfg.steps:
        t0 = time.time()
        X, C, Y = [t.to(device) for t in loader.next()]
        if pad is not None:
            # Trim suffix padding to the batch's longest drawing. Exactly two
            # shapes (half block or full): the 2023-era rounding to 64 made
            # ~10 distinct shapes, and MPSGraph keeps a workspace per shape --
            # that is the unbounded-memory trap that shelved bucketing on MPS.
            # Two shapes bounds it at two workspaces while capturing most of
            # the win when the corpus mix is mostly-short (union: 66% fits the
            # half block). Exact either way: causal attention means real
            # tokens never look at trimmed positions, and the loss ignores
            # them.
            n = int((X != pad).sum(1).max())
            half = X.size(1) // 2
            n = half if n <= half else X.size(1)
            X, Y = X[:, :n], Y[:, :n]
        _, loss = model(X, C, Y)

        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        # The MPS allocator keeps separate cached pools per tensor shape, so
        # trimmed (varying-length) batches grow resident memory without bound
        # -- observed at 21GB and swapping by step ~150. Flushing the cache
        # periodically caps it; the flush costs far less than one step.
        if pad is not None and device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

        if run:
            run.log({"train_loss_step": loss.item(), "step": step})
        if step % train_cfg.print_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | {(time.time()-t0)*1000:.0f} ms/step"
                  f" | lr {scheduler.get_last_lr()[0]:.6f}")

        if step % train_cfg.eval_every == 0:
            train_loss = evaluate(model, train_dataset, device)
            test_loss = evaluate(model, test_dataset, device)
            print(f"step {step} | train loss {train_loss:.4f} | test loss {test_loss:.4f}")
            if run:
                run.log({"train_loss": train_loss, "test_loss": test_loss, "step": step})
            for s, ds_s in src_tests.items():
                loss_s = evaluate(model, ds_s, device, batch_size=50, max_batches=4)
                print(f"  {s:10s} test {loss_s:.4f}")
                if run:
                    run.log({f"test_loss_{s}": loss_s, "step": step})

            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                print(f"New best test loss; saving checkpoint to {checkpoint_path}")
                save_checkpoint(checkpoint_path, model, char_tok.alphabet, asdict(data_cfg),
                                stroke_tok.merges, optimizer, scheduler, step, best_loss)
                if run:
                    artifact = wandb.Artifact("best_checkpoint", type="model")
                    artifact.add_file(checkpoint_path)
                    run.log_artifact(artifact)

            # Resuming needs where the run stopped, which is not where it was
            # last best: on a long run those diverge by thousands of steps, and
            # --resume best.pt silently replays them.
            save_checkpoint(os.path.join(train_cfg.out_dir, "last.pt"), model,
                            char_tok.alphabet, asdict(data_cfg), stroke_tok.merges,
                            optimizer, scheduler, step, best_loss)

            if src_tests:
                for s, ds_s in src_tests.items():
                    save_progress(model, ds_s,
                                  os.path.join(train_cfg.out_dir, f"progress_{s}"),
                                  step)
                progress = save_mixed_progress(
                    model, src_tests, os.path.join(train_cfg.out_dir, "progress"),
                    step)
            else:
                progress = save_progress(model, test_dataset,
                                         os.path.join(train_cfg.out_dir, "progress"),
                                         step)
            paths = [progress]
            if step % (train_cfg.eval_every * 4) == 0:
                paths += save_samples(model, test_dataset,
                                      os.path.join(train_cfg.out_dir, "samples"),
                                      num=3, do_sample=True)
            print(f"  wrote {progress}")
            if run:
                run.log({os.path.basename(p): wandb.Image(p) for p in paths})

    if run:
        run.finish()


if __name__ == "__main__":
    main()
