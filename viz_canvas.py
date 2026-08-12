import argparse
import os

import numpy as np
import torch

from pengpt.data import load_examples
from pengpt.model import load_for_sampling
from pengpt.sampling import draw
from train import resolve_device


def _encode(st, points, block, device):
    tokens = st.encode_word(points)
    x = torch.full((1, block), st.PAD, dtype=torch.long)
    n = min(len(tokens), block - 2)
    x[0, 0] = st.BOS
    x[0, 1:n + 1] = torch.from_numpy(tokens[:n])
    x[0, n + 1] = st.END
    return x.to(device), n, tokens


def _stroke_chunks(points):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return []
    pen_down = points[:, 2] == 1
    chunks = []
    for chunk in np.split(points, np.flatnonzero(~pen_down) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk):
            chunks.append(chunk)
    return chunks


def _to_window(xy, pen, cell, grid, half):
    scale = cell * grid if cell * grid else 1.0
    return ((xy[..., 0] - pen[0]) / scale + half,
            (xy[..., 1] - pen[1]) / scale + half)


def _splat(img, u, v, radius=0.67):
    """Soft disk of radius cells (1.0 was the old bilinear footprint)."""
    n = img.shape[0]
    x0, y0 = int(np.floor(u - radius)), int(np.floor(v - radius))
    x1, y1 = int(np.ceil(u + radius)), int(np.ceil(v + radius))
    for ys in range(y0, y1 + 1):
        for xs in range(x0, x1 + 1):
            if 0 <= xs < n and 0 <= ys < n:
                w = 1.0 - np.hypot(xs - u, ys - v) / radius
                if w > 0:
                    img[ys, xs] += w


def raster_strokes(points, pen, n, cell, grid, radius=0.67):
    """Paint pen-down polylines onto an n×n window around the pen."""
    img = np.zeros((n, n), dtype=np.float64)
    half = n // 2
    for chunk in _stroke_chunks(points):
        u, v = _to_window(chunk[:, :2], pen, cell, grid, half)
        if len(chunk) == 1:
            _splat(img, float(np.asarray(u).reshape(-1)[0]),
                   float(np.asarray(v).reshape(-1)[0]), radius)
            continue
        for i in range(len(u) - 1):
            du, dv = float(u[i + 1] - u[i]), float(v[i + 1] - v[i])
            steps = max(int(np.ceil(max(abs(du), abs(dv)) * 3)), 1)
            for s in range(steps + 1):
                t = s / steps
                _splat(img, float(u[i]) + t * du, float(v[i]) + t * dv, radius)
    return np.clip(img, 0, 1)


def save_windows(model, stroke_tok, bank_points, bank_texts, out_path,
                 n=8, seed=7, block=None):
    """One contact sheet: full word, window crop, model occupancy map."""
    if getattr(model.cfg, "local_canvas", 0) <= 0:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    device = next(model.parameters()).device
    st = stroke_tok
    block = block or model.cfg.block_size
    rng = np.random.default_rng(seed)
    usable = [i for i, p in enumerate(bank_points) if len(np.asarray(p)) >= 4]
    if not usable:
        return None
    idxs = rng.choice(usable, size=min(n, len(usable)), replace=False)
    cell = float(model.cfg.local_cell)
    nmap = int(model.cfg.local_canvas)
    half = nmap // 2
    grid = st.grid
    # Display is zoomed out ~33% vs the trained window (wider FOV, same 16×16).
    view_cell = cell * (4.0 / 3.0)
    span = half * view_cell * grid
    was_training = model.training
    model.eval()

    fig, axes = plt.subplots(len(idxs), 3, figsize=(10.2, 2.15 * len(idxs)))
    if len(idxs) == 1:
        axes = np.array([axes])

    for row, wi in enumerate(idxs):
        points = bank_points[int(wi)]
        text = bank_texts[int(wi)] if bank_texts is not None else ""
        x, nw, tokens = _encode(st, points, block, device)
        with torch.no_grad():
            pos = model._positions(x)
            saved_cell = model.cfg.local_cell
            model.cfg.local_cell = view_cell
            maps = model._local_maps(x, pos)
            model.cfg.local_cell = saved_cell
        t = 1 + max(1, int(0.55 * nw))
        t = min(t, nw)
        decoded = st.decode_word(tokens)
        prefix = x[0, 1:t + 1].cpu().numpy()
        so_far = st.decode(np.concatenate([prefix, [st.END]]))
        so_far = np.vstack(so_far) if so_far else np.zeros((0, 3))
        pen = pos[0, t].cpu().numpy().astype(float) * grid

        ax = axes[row, 0]
        draw(ax, decoded, color="0.82", linewidth=1.5)
        draw(ax, so_far, color="k", linewidth=1.5)
        ax.add_patch(Rectangle((pen[0] - span, -(pen[1] + span)),
                               2 * span, 2 * span,
                               fill=False, edgecolor="C3", linewidth=1.3))
        ax.plot(pen[0], -pen[1], "C3+", markersize=11, markeredgewidth=1.6)
        ax.set_ylabel(f'#{int(wi)}  "{text}"', fontsize=8, rotation=0,
                      ha="right", va="center")
        if row == 0:
            ax.set_title("full word  (gray = not yet written)", fontsize=9)

        ax = axes[row, 1]
        draw(ax, decoded, color="0.82", linewidth=1.8)
        draw(ax, so_far, color="k", linewidth=1.8)
        ax.plot(pen[0], -pen[1], "C3+", markersize=12, markeredgewidth=1.6)
        ax.set_xlim(pen[0] - span, pen[0] + span)
        ax.set_ylim(-(pen[1] + span), -(pen[1] - span))
        ax.set_aspect("equal")
        ax.axis("off")
        if row == 0:
            ax.set_title("window at full resolution", fontsize=9)

        ax = axes[row, 2]
        m = maps[0, t].cpu().numpy()
        strokes = raster_strokes(so_far, pen, nmap, view_cell, grid, radius=0.67)
        # Data +y is down the page (same as draw()). origin=upper puts small y
        # — the tops of letters — at the top of the grid.
        ax.imshow(strokes, cmap="Greys", origin="upper", vmin=0, vmax=1,
                  interpolation="nearest")
        # Endpoint hits at the display zoom (trained model still uses `cell`).
        hits = np.zeros((*m.shape, 4))
        hits[..., 0] = 0.82
        hits[..., 3] = 0.55 * m
        ax.imshow(hits, origin="upper", interpolation="nearest")
        for chunk in _stroke_chunks(so_far):
            u, v = _to_window(chunk[:, :2], pen, view_cell, grid, half)
            ax.plot(u, v, color="0.12", linewidth=0.90, solid_capstyle="round",
                    solid_joinstyle="round")
        ax.plot(half, half, "r+", markersize=13, markeredgewidth=2)
        ax.set_xlim(-0.5, nmap - 0.5)
        ax.set_ylim(nmap - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, nmap, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nmap, 1), minor=True)
        ax.grid(which="minor", color="0.88", linewidth=0.35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if row == 0:
            ax.set_title(f"strokes in {nmap}×{nmap}  (red wash = model hits)",
                         fontsize=9)

    fig.suptitle(
        f"local canvas  {nmap}×{nmap}, {view_cell:.2g} scribe cells/pixel  "
        f"(window {nmap * view_cell:.2g} cells = {2 * span:.2f} ink;  "
        f"trained at {cell:g})  letter height ~0.22",
        fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    model.train(was_training)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out/cursive_canvas_lin/best.pt")
    p.add_argument("--local_canvas", type=int, default=16)
    p.add_argument("--local_cell", type=float, default=1.25)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="out/cursive_canvas_lin/samples/windows.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    if os.path.exists(args.checkpoint):
        model, dataset, cfg, _ = load_for_sampling(args.checkpoint, device, 20)
        st = dataset.stroke_tok
        examples = load_examples(cfg.dataset)
    else:
        from pengpt.config import DataConfig, ModelConfig
        from pengpt.data import create_datasets
        from pengpt.model import PenTransformer, attach_tokenizer_tables
        cfg = DataConfig(dataset="data/bigbank_3500.json.zip", train_size=8, test_size=4)
        _, dataset, st, ct = create_datasets(cfg)
        examples = load_examples(cfg.dataset)
        mcfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=st.vocab_size,
                           block_size=cfg.max_seq_length, context_vocab_size=ct.vocab_size,
                           context_block_size=cfg.max_text_length,
                           local_canvas=args.local_canvas, local_cell=args.local_cell)
        model = PenTransformer(mcfg).to(device)
        attach_tokenizer_tables(model, st)
    model.eval()
    model.cfg.local_canvas = args.local_canvas
    model.cfg.local_cell = args.local_cell

    points = [e["points"] for e in examples]
    texts = [e["text"] for e in examples]
    path = save_windows(model, st, points, texts, args.out,
                        n=args.n, seed=args.seed, block=cfg.max_seq_length)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
