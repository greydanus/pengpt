"""Generation and plotting: words, paragraphs, and training-time samples.

Everything here builds on two primitives: `generate` turns a text prompt into
per-word point arrays, and `draw` puts point arrays on a matplotlib axis.
"""

import os
import textwrap
from dataclasses import dataclass, replace

import numpy as np
import torch
import matplotlib
# Every figure here is written to a file, never shown. Left to choose for
# itself, matplotlib picks an interactive backend where one is available, which
# on macOS opens a GUI window per eval from inside a headless training run: it
# logs ApplePersistenceIgnoreState, takes about thirty seconds to write what Agg
# writes in one, and can block the run outright when no session is attached.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass
class SampleParams:
    temperature: float = 1.0
    top_k: int = None
    do_sample: bool = True
    max_tokens: int = 512
    n_at_a_time: int = 4
    space_width: float = 0.16
    line_width: float = 8.0
    line_height: float = 0.55
    seed: int = 42
    linewidth: float = 1.3
    verbose: bool = True


def generate(model, dataset, text, params=None):
    """Text prompt -> list of per-word (N, 3) point arrays.

    Nothing from the training set seeds the sequence: the model starts empty and
    the prompt alone drives it, so a sample cannot copy strokes it was shown.
    """
    params = params or SampleParams()
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    context = torch.from_numpy(
        dataset.encode_text(text)).unsqueeze(0).to(device)
    idx = torch.full((1, 1), st.BOS, dtype=torch.long, device=device)
    out = model.generate(idx, context, max_new_tokens=params.max_tokens,
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END, pad_token=st.PAD)
    return st.decode(out[0].cpu().numpy()[1:])


def draw(ax, points, color="b", linewidth=1.3):
    """Draw absolute pen points (N, 3), lifting the pen where pen == 0."""
    points = np.asarray(points, dtype=float)
    if len(points):
        pen_down = points[:, 2] == 1
        for chunk in np.split(points, np.flatnonzero(~pen_down) + 1):
            chunk = chunk[chunk[:, 2] == 1]
            if len(chunk) > 1:
                ax.plot(chunk[:, 0], -chunk[:, 1], color=color, linewidth=linewidth,
                        solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def layout_words(words, params=None):
    """Place per-word point arrays on a page, left to right with wrapping.

    Each word carries its own height above the baseline, so this only advances
    horizontally and wraps lines; no per-alphabet table is involved.
    """
    params = params or SampleParams()
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


def plot_words(words, params=None, title="", figsize=(12, 2), dpi=150, ax=None,
               color="b"):
    """Lay out per-word arrays and draw them on one axis."""
    params = params or SampleParams()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    placed = layout_words(words, params)
    draw(ax, np.vstack(placed) if placed else np.zeros((0, 3)), color, params.linewidth)
    if title:
        ax.set_title(title)
    return fig, ax, placed


def plot_paragraph(words, text="", params=None, figsize=(12, 8), dpi=200,
                   show_indices=False, include_title=False):
    params = params or SampleParams()
    fig, ax, placed = plot_words(words, params, figsize=figsize, dpi=dpi)
    if show_indices:
        for i, points in enumerate(placed):
            ax.text(points[:, 0].min() - 0.08, -points[0, 1] + 0.15, str(i), fontsize=8)
    if include_title and text:
        ax.set_title("\n".join(textwrap.wrap(text, width=83)), loc="left", fontsize=13)
    return fig, ax


def generate_paragraph(model, dataset, text, params=None, words=None, redo=None):
    """Generate a paragraph, n_at_a_time words per model call.

    Words come in small groups because neighbours act as context. Groups of up
    to four recover every word; larger groups overflow the block -- six words
    cost more than 512 tokens about 70% of the time -- and the words that do not
    fit are silently dropped.

    Pass a previous result as `words` plus indices as `redo` to regenerate only
    the words that came out wrong.
    """
    params = params or SampleParams()
    fits = max(1, (params.max_tokens - 1) // 120)
    if params.n_at_a_time > fits:
        print(f"  n_at_a_time={params.n_at_a_time} likely overflows a "
              f"{params.max_tokens}-token block; using {fits}")
        params = replace(params, n_at_a_time=fits)
    torch.manual_seed(params.seed)
    prompt_words = text.strip().split()

    def for_chunk(chunk):
        out = generate(model, dataset, " ".join(chunk), params)[:len(chunk)]
        if params.verbose:
            print(f"  {' '.join(chunk)}")
        return out + [np.zeros((0, 3))] * (len(chunk) - len(out))

    if words is None:
        words = []
        for i in range(0, len(prompt_words), params.n_at_a_time):
            words += for_chunk(prompt_words[i:i + params.n_at_a_time])
    else:
        for i in redo or []:
            if i < len(prompt_words):
                words[i] = for_chunk([prompt_words[i]])[0]
    return words


def progress_prompts(dataset, n=8):
    """Prompts drawn from the dataset's own text, so they are always in vocabulary.

    Hardcoded prompts silently break on a new corpus: the handwriting defaults
    contain digits and a capital S, and Quick, Draw!'s 35-character alphabet has
    neither, so those characters encoded as padding and two of three progress
    panels showed nothing meaningful for a whole run.

    A dataset with no usable text raises rather than falling back to those
    defaults, which would reintroduce exactly that failure somewhere quieter.
    """
    seen, out = set(), []
    for i in range(min(len(dataset), 2000)):
        text = dataset.text_for(i)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= n:
            break
    if not out:
        raise ValueError(f"{dataset.name or 'dataset'} yielded no text to prompt "
                         "with; progress images need labelled examples")
    return out


def _cached_prompts(dataset):
    """Same prompts every eval, so the strip tracks the model not the prompt."""
    if not hasattr(dataset, "_progress_prompts"):
        dataset._progress_prompts = progress_prompts(dataset)
    return dataset._progress_prompts


def save_progress(model, dataset, out_dir, step, prompts=None, temperature=1.0,
                  rows=4):
    """A grid of samples: one column per prompt, `rows` samples down each.

    Several samples per prompt is what makes the picture readable. One sample
    cannot distinguish a model that ignores its prompt from one that drew a
    poor sample, and a column of four shows immediately whether the prompt
    controls the shape or the model is drawing the corpus average whatever it
    is asked for.

    Prompts and seeds are fixed across evals, so consecutive images differ only
    by the model.
    """
    prompts = prompts or _cached_prompts(dataset)
    os.makedirs(out_dir, exist_ok=True)
    params = SampleParams(temperature=temperature,
                          max_tokens=dataset.cfg.max_seq_length - 1)

    fig, axes = plt.subplots(rows, len(prompts),
                             figsize=(1.9 * len(prompts), 1.9 * rows),
                             squeeze=False)
    # One batched generate for the whole grid instead of rows x cols separate
    # calls: at long block sizes the sequential version dominated eval time
    # (each call re-runs its full prefix every token). A fixed seed keeps the
    # grid deterministic per eval, so consecutive images still differ only by
    # the model.
    st = dataset.stroke_tok
    device = next(model.parameters()).device
    contexts = torch.stack([
        torch.from_numpy(dataset.encode_text(text))
        for text in prompts for _ in range(rows)]).to(device)
    torch.manual_seed(3)
    idx = torch.full((len(contexts), 1), st.BOS, dtype=torch.long, device=device)
    out = model.generate(idx, contexts, max_new_tokens=params.max_tokens,
                         temperature=params.temperature, top_k=params.top_k,
                         do_sample=params.do_sample, end_token=st.END,
                         pad_token=st.PAD)
    for col in range(len(prompts)):
        for row in range(rows):
            words = st.decode(out[col * rows + row].cpu().numpy()[1:])
            plot_words(words, params, ax=axes[row][col], color="k")
    fig.suptitle(f"step {step:,}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Label the columns on the figure rather than per axis. An axis title sits
    # relative to that axis's contents, so a column of short drawings puts its
    # label lower than the rest; placing them at one figure height keeps them
    # aligned however tall the samples happen to be.
    top = max(ax.get_position().y1 for ax in axes[0])
    for col, text in enumerate(prompts):
        box = axes[0][col].get_position()
        # Wrap to the column width, or sentence-length prompts (FS-COCO
        # captions) run into their neighbors. va="bottom" grows extra lines
        # upward, into space bbox_inches="tight" then reclaims.
        wrapped = "\n".join(textwrap.wrap(text, width=24))
        fig.text(box.x0 + box.width / 2, top + 0.012, wrapped,
                 ha="center", va="bottom", fontsize=8)
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def save_mixed_progress(model, datasets_by_source, out_dir, step, cols=6):
    os.makedirs(out_dir, exist_ok=True)
    sources = sorted(datasets_by_source)
    fig, axes = plt.subplots(len(sources), cols,
                             figsize=(2.1 * cols, 2.2 * len(sources)),
                             squeeze=False)
    for row, source in enumerate(sources):
        ds = datasets_by_source[source]
        prompts = _cached_prompts(ds)[:cols]
        params = SampleParams(max_tokens=ds.cfg.max_seq_length - 1)
        st = ds.stroke_tok
        device = next(model.parameters()).device
        contexts = torch.stack([torch.from_numpy(ds.encode_text(t))
                                for t in prompts]).to(device)
        torch.manual_seed(7)
        idx = torch.full((len(prompts), 1), st.BOS, dtype=torch.long, device=device)
        out = model.generate(idx, contexts, max_new_tokens=params.max_tokens,
                             temperature=params.temperature, end_token=st.END,
                             pad_token=st.PAD)
        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(prompts):
                words = st.decode(out[col].cpu().numpy()[1:])
                plot_words(words, params, ax=ax, color="k")
                ax.set_title(prompts[col][:28], fontsize=7)
        axes[row][0].set_ylabel(source, fontsize=10, rotation=0,
                                ha="right", va="center", labelpad=10)
    fig.suptitle(f"step {step:,}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"mixed_{step:06d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def save_samples(model, dataset, out_dir=".", num=3, do_sample=True):
    """Generate from test prompts and save one PNG per example."""
    os.makedirs(out_dir, exist_ok=True)
    params = SampleParams(do_sample=do_sample,
                          max_tokens=dataset.cfg.max_seq_length - 1)
    paths = []
    for i in range(num):
        text = dataset.text_for(i)
        words = generate(model, dataset, text, params)
        fig, _, _ = plot_words(words, params, title=f'{dataset.name} {i}: "{text}"')
        path = os.path.join(out_dir,
                            f"{dataset.name}_{'sample' if do_sample else 'greedy'}_{i}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
