"""Configuration dataclasses and a CLI that is generated from them.

Defaults reproduce the best cursivetransformer paper run
(https://arxiv.org/abs/2504.00051).
"""

import argparse
from dataclasses import dataclass, fields


@dataclass
class DataConfig:
    dataset: str = "data/bigbank_3500.json.zip"
    train_size: int = 500_000
    test_size: int = 3_000
    max_words: int = 8
    max_seq_length: int = 512
    max_text_length: int = 50
    grid: float = 0.020
    n_merges: int = 512
    augment: bool = True
    spacing: float = 0.0
    spacing_jitter: float = 0.20
    shear_min: float = -0.22
    shear_max: float = -0.18
    scale_jitter: float = 0.10
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


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    warmup: int = 200
    grad_clip: float = 1.0
    eval_every: int = 1_000
    print_every: int = 100
    num_workers: int = 4
    device: str = "auto"
    out_dir: str = "out"
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
                parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = vars(parser.parse_args(argv))

    def build(cls):
        return cls(**{f.name: args[f.name] for f in fields(cls) if f.name in args})

    return build(DataConfig), build(ModelConfig), build(TrainConfig)
