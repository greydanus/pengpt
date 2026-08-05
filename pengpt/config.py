"""Configuration dataclasses and a CLI that is generated from them.

Defaults reproduce the best cursivetransformer paper run
(https://arxiv.org/abs/2504.00051).
"""

import argparse
from dataclasses import dataclass, fields


@dataclass
class DataConfig:
    dataset: str = "data/bigbank_3500.json.zip"  # .json or .json.zip
    train_size: int = 497_000     # number of generated word-combo examples
    test_size: int = 3_000
    num_words: int = 4            # words stitched together per training example
    max_seq_length: int = 1050    # stroke tokens per example (2 per pen point)
    max_text_length: int = 50     # characters of ASCII context
    augment: bool = True
    downsample_mean: float = 0.65  # mean fraction of interior stroke points removed
    downsample_width: float = 0.10 # width of the uniform jitter on that fraction
    drop_prob: float = 0.05        # per-point random drop during downsampling
    shear_min: float = -0.22       # fixed negative shear = consistent italic slant
    shear_max: float = -0.18
    scale_jitter: float = 0.10     # +/- fraction of random x and y rescaling
    seed: int = 1337


@dataclass
class ModelConfig:
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 64
    # Derived from the dataset and tokenizers before the model is built:
    vocab_size: int = -1
    block_size: int = -1
    context_vocab_size: int = -1
    context_block_size: int = -1


DERIVED_FIELDS = {"vocab_size", "block_size", "context_vocab_size", "context_block_size"}


@dataclass
class TrainConfig:
    steps: int = 125_000
    batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    lr_decay: float = 0.5         # multiply lr by this ...
    lr_decay_every: int = 20_000  # ... every this many steps
    grad_clip: float = 1.0
    eval_every: int = 2_500       # also saves checkpoints and sample images
    print_every: int = 100
    num_workers: int = 4
    device: str = "auto"          # auto | cuda | mps | cpu
    out_dir: str = "out"
    resume: str = ""              # path to a checkpoint to resume from
    wandb: bool = False           # optional; local logging always works
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
