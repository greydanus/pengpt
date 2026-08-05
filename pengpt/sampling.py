"""Generation and plotting: words, paragraphs, and training-time samples."""

import os
import textwrap
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

from .tokenizer import offsets_to_points

# Baseline alignment heuristics for the bigbank handwriting style: words whose
# first pen point sits at the baseline vs. at the cap line. Others ('1I2Z?')
# start somewhere in between and are left alone.
STARTS_AT_BOTTOM = "enaitoshrdx.vpukbgfcymzwlqjS,GJ"
STARTS_AT_TOP = "8049637OTA5N)EHR\"'(BCQLMWYUF!DXVKP"


@dataclass
class SampleParams:
    temperature: float = 1.0
    top_k: int = None
    do_sample: bool = True
    num_steps: int = 1050        # max stroke tokens generated per model call
    n_at_a_time: int = 2         # words generated per model call
    n_context_words: int = 4     # pad the ASCII context to this many words
    space_width: float = 0.16
    line_width: float = 8.0      # wrap lines longer than this
    line_height: float = 0.55
    letter_height: float = 0.35  # clip strokes that wander beyond this
    warmup_ix: int = None        # dataset example used to seed generation
    seed: int = 42
    linewidth: float = 1.3       # matplotlib stroke width
    verbose: bool = True


def plot_points(points, title="", fig=None, ax=None, figsize=(12, 2), dpi=150,
                linewidth=1.3):
    """Plot absolute pen points (N, 3), lifting the pen where pen == 0."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pen_down = points[:, 2] == 1
    for chunk in np.split(points, np.flatnonzero(~pen_down) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk):
            ax.plot(chunk[:, 0], 1 - chunk[:, 1], "b-", linewidth=linewidth)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return fig, ax


def layout_words(word_offsets, params, words=None):
    """Place per-word offset arrays on a page: returns a list of point arrays.

    Words advance left to right with wrapping; if the word list is given, each
    word is shifted vertically so its first point sits at the right height.
    """
    placed, x, y = [], 0.0, 0.0
    for i, offsets in enumerate(word_offsets):
        points = offsets_to_points(offsets)

        if words is not None and i < len(words) and len(points):
            first = words[i][0] if words[i] else ""
            if first in STARTS_AT_BOTTOM:
                points[:, 1] -= points[0, 1]
            elif first in STARTS_AT_TOP:
                points[:, 1] -= points[0, 1] + 0.18

        if x > params.line_width:
            x, y = 0.0, y + params.line_height

        if len(points) == 0:
            points = np.array([[x, y, 0.0]])
        else:
            points[:, 0] += x
            points[:, 1] = np.clip(points[:, 1], -params.letter_height,
                                   params.letter_height) + y
        placed.append(points)
        x = points[-1, 0] + params.space_width
    return placed


def plot_paragraph(word_offsets, text, params=None, figsize=(12, 8), dpi=200,
                   show_indices=False, include_title=False):
    params = params or SampleParams()
    placed = layout_words(word_offsets, params, words=text.split())
    fig, ax = plot_points(np.vstack(placed), figsize=figsize, dpi=dpi,
                          linewidth=params.linewidth)
    if show_indices:  # word indices, for picking which words to regenerate
        for i, points in enumerate(placed):
            ax.text(points[0, 0] - 0.12, 0.95 - points[0, 1], str(i), fontsize=8)
    if include_title:
        ax.set_title("\n".join(textwrap.wrap(text, width=83)), loc="left", fontsize=13)
    return fig, ax


def generate_words(model, dataset, words, params, rng):
    """Generate stroke offsets for a short list of words.

    The model always saw num_words-word examples in training, so we (a) seed
    generation with the strokes of one real word drawn from the dataset and
    prepend its text to the context ("warmup"), and (b) pad the context out to
    n_context_words with random words from the bank. Only the strokes for
    `words` are returned.
    """
    st, ct = dataset.stroke_tok, dataset.char_tok
    device = next(model.parameters()).device

    ix = params.warmup_ix if params.warmup_ix is not None else int(rng.integers(len(dataset)))
    if params.verbose:
        print(f"  {' '.join(words)} (warmup_ix={ix})")
    x, _, _ = dataset[ix]
    tokens = x.numpy()
    word_starts = np.flatnonzero(tokens == st.WORD)
    first_word_tokens = tokens[:word_starts[0]] if word_starts.size else tokens
    seed = np.concatenate([first_word_tokens, st.separator])

    context_words = [dataset.text_for(ix).split()[0]] + list(words)
    while len(context_words) < params.n_context_words + 1:
        context_words.append(dataset.bank_texts[int(rng.integers(len(dataset.bank_texts)))])

    idx = torch.from_numpy(seed).unsqueeze(0).to(device)
    context = torch.from_numpy(
        ct.encode(" ".join(context_words), dataset.cfg.max_text_length)).unsqueeze(0).to(device)

    out = model.generate(idx, context, max_new_tokens=params.num_steps - len(seed),
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END, pad_token=st.PAD)
    generated = st.decode(out[0].cpu().numpy()[len(seed):])

    generated = generated[:len(words)]
    generated += [np.zeros((0, 3))] * (len(words) - len(generated))
    return generated


def generate_paragraph(model, dataset, text, params=None, word_offsets=None, redo=None):
    """Generate a paragraph n_at_a_time words per model call.

    Pass the previous result as word_offsets plus a list of indices as redo to
    regenerate only misspelled words.
    """
    params = params or SampleParams()
    torch.manual_seed(params.seed)
    rng = np.random.default_rng(params.seed)

    words = text.strip().split()
    if word_offsets is None:
        word_offsets = []
        for i in range(0, len(words), params.n_at_a_time):
            chunk = words[i:i + params.n_at_a_time]
            word_offsets += generate_words(model, dataset, chunk, params, rng)
    else:
        for i in redo or []:
            if i < len(words):
                word_offsets[i] = generate_words(model, dataset, [words[i]], params, rng)[0]
    return word_offsets


def save_samples(model, dataset, out_dir=".", num=3, do_sample=True, warmup_tokens=50):
    """Seed with the first tokens of dataset examples, generate the rest, and
    save one PNG per example. Returns the file paths (for wandb etc.)."""
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    batch = [dataset[i] for i in range(num)]
    idx = torch.stack([x for x, _, _ in batch])[:, :warmup_tokens].to(device)
    context = torch.stack([c for _, c, _ in batch]).to(device)

    out = model.generate(idx, context, max_new_tokens=dataset.cfg.max_seq_length - warmup_tokens,
                         do_sample=do_sample, end_token=st.END, pad_token=st.PAD)

    os.makedirs(out_dir, exist_ok=True)
    params, paths = SampleParams(), []
    for i in range(num):
        text = dataset.text_for(i)
        word_offsets = st.decode(out[i].cpu().numpy())
        placed = layout_words(word_offsets, params)
        points = np.vstack(placed) if placed else np.zeros((1, 3))
        fig, _ = plot_points(points, title=f'{dataset.name} sample {i}: "{text}"')
        path = os.path.join(out_dir, f"{dataset.name}_{'sample' if do_sample else 'greedy'}_{i}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
