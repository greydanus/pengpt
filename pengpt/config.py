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

Comparing two runs. Reported loss is nats per token, which is only comparable
between runs whose tokenizers agree. Anything that changes tokens per example --
grid, n_merges, how merges apply -- rescales it, so the run packing more
information per token posts a higher loss for identical performance. Multiply by
tokens per example first: a merge-order change here moved 52.1 tokens at 2.3964
nats and 42.8 at 2.9197, which reads as a large regression and is 180.1 against
180.3 bits per example, the same number twice.

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
    # Probability of mirroring a drawing left-right. Skipped for any example
    # whose caption says "left" or "right", so the text never lies about the
    # picture. A mirrored scene is a valid scene; mirrored handwriting is not,
    # so leave this off for text corpora.
    hflip: float = 0.0
    tremor: float = 0.0
    # Probability of deleting each stroke independently (at least one always
    # survives). A scene of sixty strokes minus three still matches its
    # caption; a word minus a letter does not, so again: drawings only.
    stroke_dropout: float = 0.0
    text_encoder: str = "char"
    clip_image_embeds: str = ""
    embed_dropout: float = 0.1
    holdout: str = ""
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
    # Fourier features of the pen's absolute canvas position, added to the
    # token embedding. 0 disables (and matches checkpoints from before the
    # feature existed). The tokens are relative motion, so without this the
    # model must integrate the whole walk with attention to know whether two
    # strokes connect; with it, position is an input rather than a computation.
    pen_pos_bands: int = 0
    # Training-time random canvas offset in grid cells. Absolute layout can't
    # be memorized when the origin moves, but within-sample geometry -- which
    # strokes touch, what is already drawn where -- survives translation.
    pen_pos_jitter: int = 32
    # Fourier features of (pen - last DOWN), so the model sees the vector back
    # to the start of the current stroke. Closure is "return to this cell",
    # not "where am I on the page".
    pen_last_down: bool = False
    # Local ink occupancy around the pen. 0 disables. 16 means a 16x16 map.
    # local_cell is ScribeTokens cells per map pixel (can be fractional).
    # 1.25 is a ~20-cell window -- about 1.6x tighter than cell=2 -- with
    # bilinear splat so sub-cell structure survives.
    local_canvas: int = 0
    local_cell: float = 1.25
    # Read the occupancy map with a Linear (fast) instead of the tiny CNN.
    # CNN backward is the bulk of the canvas tax on MPS; Linear lands near
    # +10% step time, CNN near +25%. Old canvas checkpoints keep the CNN
    # via _model_config_from_ckpt.
    canvas_linear: bool = True
    context_dim: int = 0


DERIVED_FIELDS = {"vocab_size", "block_size", "context_vocab_size",
                  "context_block_size", "context_dim"}

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
    # Group similar-length drawings per batch and trim each batch to its
    # longest member (rounded up, so MPS reuses compiled shapes). Exact --
    # suffix padding never influences real tokens -- but only worthwhile for
    # single-drawing corpora whose lengths vary a lot; requires max_words 1.
    bucket_batches: bool = False
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
