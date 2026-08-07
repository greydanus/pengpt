"""Configuration dataclasses and a CLI that is generated from them.

Defaults come from wall-clock ablations on bigbank_3500, scored in bits per
pen-point on held-out words. Two are worth knowing about because they are
specific to the machine rather than to the task:

- batch_size is small. Throughput here is flat in batch size (47 ktok/s at 8,
  58 at 128), so a larger batch buys no tokens per second and simply takes
  fewer optimizer steps in the same time. On hardware that does scale, raise it.
- grid trades reconstruction error against sequence length. 0.020 keeps error
  inside the width of a pen stroke while costing 99 tokens per word.

Every run writes to out/<experiment>/, holding best.pt, last.pt, progress/, and
samples/. Keeping them under one ignored parent is what stops a run's outputs
from being committed by accident.

The defaults suit handwriting. A drawing corpus such as Quick, Draw! wants
--max_words 1 --augment general, a max_text_length near its longest label, and
a max_seq_length near the p99 token count -- see README.
"""

import argparse
from dataclasses import dataclass, fields


@dataclass
class DataConfig:
    dataset: str = "data/bigbank_3500.json.zip"
    max_examples: int = 0
    train_size: int = 500_000
    test_size: int = 3_000
    max_words: int = 8
    max_seq_length: int = 512
    max_text_length: int = 50
    grid: float = 0.020
    n_merges: int = 512
    augment: str = "handwriting"
    spacing: float = 0.0
    spacing_jitter: float = 0.20
    scale_jitter: float = 0.15
    rotate: float = 0.0
    shear_min: float = -0.22
    shear_max: float = -0.18
    seed: int = 1337


@dataclass
class ModelConfig:
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 64
    vocab_size: int = -1
    block_size: int = -1
    context_vocab_size: int = -1
    context_block_size: int = -1


DERIVED_FIELDS = {"vocab_size", "block_size", "context_vocab_size", "context_block_size"}

CHOICES = {"augment": ("none", "general", "handwriting")}


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    warmup: int = 200
    grad_clip: float = 1.0
    eval_every: int = 1_000
    print_every: int = 100
    num_workers: int = 4
    device: str = "auto"
    out_dir: str = "out/default"
    resume: str = ""
    wandb: bool = False
    wandb_project: str = "pengpt"
    wandb_entity: str = ""
    wandb_run_name: str = ""


def parse_configs(argv=None, description="pengpt"):
    """Build one argparse CLI from the three dataclasses above."""
    parser = argparse.ArgumentParser(description=description)
    for cls in (DataConfig, ModelConfig, TrainConfig):
        for f in fields(cls):
            if f.name in DERIVED_FIELDS:
                continue
            if isinstance(f.default, bool):
                parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction,
                                    default=f.default)
            else:
                parser.add_argument(f"--{f.name}", type=type(f.default),
                                    default=f.default, choices=CHOICES.get(f.name))
    args = vars(parser.parse_args(argv))

    def build(cls):
        return cls(**{f.name: args[f.name] for f in fields(cls) if f.name in args})

    return build(DataConfig), build(ModelConfig), build(TrainConfig)
