"""Train a PenTransformer.

    python train.py --dataset data/bigbank_3500.json.zip --out_dir out
    python train.py --wandb --wandb_entity you --wandb_project pengpt

See pengpt/config.py for the full set of options and their defaults.
"""

import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from pengpt import (parse_configs, create_datasets, InfiniteDataLoader,
                    PenTransformer, save_checkpoint, load_checkpoint, save_samples,
                    save_progress)


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
    torch.manual_seed(data_cfg.seed)
    device = resolve_device(train_cfg.device)
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_cfg.out_dir, "best.pt")

    train_dataset, test_dataset, stroke_tok, char_tok = create_datasets(data_cfg)
    model_cfg.vocab_size = stroke_tok.vocab_size
    model_cfg.block_size = data_cfg.max_seq_length
    model_cfg.context_vocab_size = char_tok.vocab_size
    model_cfg.context_block_size = data_cfg.max_text_length

    model = PenTransformer(model_cfg).to(device)
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
    if train_cfg.resume:
        _, ckpt = load_checkpoint(train_cfg.resume, device)
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

    loader = InfiniteDataLoader(train_dataset, batch_size=train_cfg.batch_size,
                                pin_memory=(device == "cuda"),
                                num_workers=train_cfg.num_workers)

    while step < train_cfg.steps:
        t0 = time.time()
        X, C, Y = [t.to(device) for t in loader.next()]
        _, loss = model(X, C, Y)

        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

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

            progress = save_progress(model, test_dataset,
                                     os.path.join(train_cfg.out_dir, "progress"), step)
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
