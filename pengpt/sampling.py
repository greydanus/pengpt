"""Generation and plotting: words, paragraphs, and training-time samples."""

import os
import textwrap
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt


@dataclass
class SampleParams:
    temperature: float = 1.0
    top_k: int = None
    do_sample: bool = True
    max_tokens: int = 512
    n_at_a_time: int = 2
    space_width: float = 0.16
    line_width: float = 8.0
    line_height: float = 0.55
    seed: int = 42
    linewidth: float = 1.3
    verbose: bool = True


def plot_points(points, title="", fig=None, ax=None, figsize=(12, 2), dpi=150,
                linewidth=1.3):
    """Plot absolute pen points (N, 3), lifting the pen where pen == 0."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pen_down = points[:, 2] == 1
    for chunk in np.split(points, np.flatnonzero(~pen_down) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            ax.plot(chunk[:, 0], -chunk[:, 1], "b-", linewidth=linewidth,
                    solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return fig, ax


def layout_words(words, params):
    """Place per-word point arrays on a page, left to right with wrapping.

    Each word carries its own vertical position (ScribeTokens encodes absolute
    grid coordinates), so no per-alphabet baseline table is needed: we only
    advance horizontally and wrap lines.
    """
    placed, x, y = [], 0.0, 0.0
    for points in words:
        points = np.asarray(points, dtype=float).copy()
        if len(points) == 0:
            placed.append(np.array([[x, y, 0.0]]))
            continue
        points[:, 0] -= points[:, 0].min()
        width = points[:, 0].max()
        if x > 0 and x + width > params.line_width:
            x, y = 0.0, y + params.line_height
        points[:, 0] += x
        points[:, 1] += y
        placed.append(points)
        x += width + params.space_width
    return placed


def plot_paragraph(words, text="", params=None, figsize=(12, 8), dpi=200,
                   show_indices=False, include_title=False):
    params = params or SampleParams()
    placed = layout_words(words, params)
    fig, ax = plot_points(np.vstack(placed), figsize=figsize, dpi=dpi,
                          linewidth=params.linewidth)
    if show_indices:
        for i, points in enumerate(placed):
            ax.text(points[:, 0].min() - 0.08, -points[0, 1] + 0.15, str(i), fontsize=8)
    if include_title and text:
        ax.set_title("\n".join(textwrap.wrap(text, width=83)), loc="left", fontsize=13)
    return fig, ax


def generate_words(model, dataset, words, params, rng):
    """Generate pen strokes for a short list of words.

    Generation is unconditional on any warmup strokes: the model starts from an
    empty sequence and the text prompt alone drives it, so nothing from the
    training set leaks into a sample.
    """
    st, ct = dataset.stroke_tok, dataset.char_tok
    device = next(model.parameters()).device
    if params.verbose:
        print(f"  {' '.join(words)}")

    text = " ".join(words)
    context = torch.from_numpy(
        ct.encode(text, dataset.cfg.max_text_length)).unsqueeze(0).to(device)
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    idx[0, 0] = st.PAD

    out = model.generate(idx, context, max_new_tokens=params.max_tokens,
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END, pad_token=st.PAD)
    generated = st.decode(out[0].cpu().numpy()[1:])
    generated = generated[:len(words)]
    generated += [np.zeros((0, 3))] * (len(words) - len(generated))
    return generated


def generate_paragraph(model, dataset, text, params=None, words=None, redo=None):
    """Generate a paragraph n_at_a_time words per model call.

    Pass a previous result as `words` plus a list of indices as `redo` to
    regenerate only the words that came out wrong.
    """
    params = params or SampleParams()
    torch.manual_seed(params.seed)
    rng = np.random.default_rng(params.seed)

    tokens = text.strip().split()
    if words is None:
        words = []
        for i in range(0, len(tokens), params.n_at_a_time):
            chunk = tokens[i:i + params.n_at_a_time]
            words += generate_words(model, dataset, chunk, params, rng)
    else:
        for i in redo or []:
            if i < len(tokens):
                words[i] = generate_words(model, dataset, [tokens[i]], params, rng)[0]
    return words


def save_samples(model, dataset, out_dir=".", num=3, do_sample=True):
    """Generate from test prompts and save one PNG per example."""
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    os.makedirs(out_dir, exist_ok=True)
    params, paths = SampleParams(do_sample=do_sample), []

    for i in range(num):
        text = dataset.text_for(i)
        context = torch.from_numpy(
            dataset.char_tok.encode(text, dataset.cfg.max_text_length)
        ).unsqueeze(0).to(device)
        idx = torch.full((1, 1), st.PAD, dtype=torch.long, device=device)
        out = model.generate(idx, context, max_new_tokens=dataset.cfg.max_seq_length - 1,
                             do_sample=do_sample, end_token=st.END, pad_token=st.PAD)
        words = st.decode(out[0].cpu().numpy()[1:])
        placed = layout_words(words, params)
        points = np.vstack(placed) if placed else np.zeros((1, 3))
        fig, _ = plot_points(points, title=f'{dataset.name} {i}: "{text}"')
        path = os.path.join(out_dir,
                            f"{dataset.name}_{'sample' if do_sample else 'greedy'}_{i}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
