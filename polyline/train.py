import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pengpt.config import DataConfig, ModelConfig, TrainConfig, DERIVED_FIELDS, CHOICES
from pengpt.model import PenTransformer, save_checkpoint, load_checkpoint
from pengpt.data import InfiniteDataLoader
from pengpt.sampling import save_progress, save_samples
from train import resolve_device, evaluate

from polyline.data import create_datasets


@dataclass
class PolylineDataConfig(DataConfig):
    epsilon: float = 0.020
    max_run: int = 16
    max_chunk_verts: int = 4


def parse():
    import argparse
    parser = argparse.ArgumentParser()
    for cls in (PolylineDataConfig, ModelConfig, TrainConfig):
        for f in fields(cls):
            if f.name in DERIVED_FIELDS:
                continue
            if isinstance(f.default, bool):
                parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction,
                                    default=f.default)
            else:
                parser.add_argument(f"--{f.name}", type=type(f.default),
                                    default=f.default, choices=CHOICES.get(f.name))
    args = vars(parser.parse_args())

    def build(cls):
        return cls(**{f.name: args[f.name] for f in fields(cls) if f.name in args})

    return build(PolylineDataConfig), build(ModelConfig), build(TrainConfig)


def main():
    data_cfg, model_cfg, train_cfg = parse()
    if data_cfg.max_seq_length == 512:
        data_cfg.max_seq_length = 256
        print("polyline: defaulting max_seq_length to 256 (override with --max_seq_length)")

    device = resolve_device(train_cfg.device)
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_cfg.out_dir, "best.pt")

    ckpt = None
    if train_cfg.resume:
        _, ckpt = load_checkpoint(train_cfg.resume, device)
        print(f"Resuming weights from {train_cfg.resume}")

    torch.manual_seed(data_cfg.seed)
    if ckpt and ckpt.get("polyline"):
        for k, v in ckpt["polyline"].items():
            if hasattr(data_cfg, k):
                setattr(data_cfg, k, v)
        print(f"Using tokenizer from checkpoint: {ckpt['polyline']}")
    merges = ckpt["merges"] if ckpt else None
    train_dataset, test_dataset, stroke_tok, char_tok = create_datasets(
        data_cfg, merges=merges)
    if ckpt:
        assert char_tok.alphabet == ckpt["alphabet"]

    model_cfg.vocab_size = stroke_tok.vocab_size
    model_cfg.block_size = data_cfg.max_seq_length
    model_cfg.context_vocab_size = char_tok.vocab_size
    model_cfg.context_block_size = data_cfg.max_text_length
    model_cfg.context_dim = 0

    model = PenTransformer(model_cfg).to(device)
    if model_cfg.pen_pos_bands > 0:
        model.pen_deltas.copy_(torch.tensor(stroke_tok.token_deltas(), device=device))
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
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt.get("step") or 0
        best_loss = ckpt.get("best_loss")
        print(f"Resumed at step {step} (best loss {best_loss})")

    loader = InfiniteDataLoader(train_dataset, batch_size=train_cfg.batch_size,
                                pin_memory=(device == "cuda"),
                                num_workers=train_cfg.num_workers)

    extra = {"polyline": {"grid": stroke_tok.grid, "epsilon": stroke_tok.epsilon,
                          "max_run": stroke_tok.max_run,
                          "max_chunk_verts": stroke_tok.max_chunk_verts}}

    def dump(path, loss):
        save_checkpoint(path, model, char_tok.alphabet, asdict(data_cfg),
                        stroke_tok.merges, optimizer, scheduler, step, loss)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob.update(extra)
        torch.save(blob, path)

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

        if step % train_cfg.print_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | {(time.time()-t0)*1000:.0f} ms/step"
                  f" | lr {scheduler.get_last_lr()[0]:.6f}")

        if step % train_cfg.eval_every == 0:
            train_loss = evaluate(model, train_dataset, device)
            test_loss = evaluate(model, test_dataset, device)
            print(f"step {step} | train loss {train_loss:.4f} | test loss {test_loss:.4f}")
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                print(f"New best test loss; saving {checkpoint_path}")
                dump(checkpoint_path, best_loss)
            dump(os.path.join(train_cfg.out_dir, "last.pt"), best_loss)
            progress = save_progress(model, test_dataset,
                                     os.path.join(train_cfg.out_dir, "progress"),
                                     step)
            print(f"  wrote {progress}")
            if step % (train_cfg.eval_every * 4) == 0:
                save_samples(model, test_dataset,
                             os.path.join(train_cfg.out_dir, "samples"),
                             num=3, do_sample=True)

    print("done")


if __name__ == "__main__":
    main()
