"""A GPT-style decoder with cross-attention over a character context.

The architecture follows Karpathy's makemore/nanoGPT lineage: pre-LayerNorm
blocks of (causal self-attention, cross-attention, MLP). Attention uses
torch.nn.functional.scaled_dot_product_attention, which dispatches to flash
attention where available.
"""

import math
from dataclasses import asdict, fields

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig


def _split_heads(t, n_head):
    B, T, C = t.shape
    return t.view(B, T, n_head, C // n_head).transpose(1, 2)


def _merge_heads(t):
    B, nh, T, hs = t.shape
    return t.transpose(1, 2).contiguous().view(B, T, nh * hs)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head

    def forward(self, x):
        q, k, v = (_split_heads(t, self.n_head) for t in self.qkv(x).split(x.size(-1), dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(_merge_heads(y))


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.kv = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head

    def forward(self, x, context, mask=None):
        q = _split_heads(self.q(x), self.n_head)
        k, v = (_split_heads(t, self.n_head) for t in self.kv(context).split(x.size(-1), dim=2))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(_merge_heads(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.cross_attn = CrossAttention(cfg)
        self.ln3 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, context, ctx_mask=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), context, ctx_mask)
        x = x + self.mlp(self.ln3(x))
        return x


class PenTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "fill in the derived ModelConfig fields first"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        if cfg.context_dim > 0:
            self.ctx_proj = nn.Linear(cfg.context_dim, cfg.n_embd)
        else:
            self.ctx_emb = nn.Embedding(cfg.context_vocab_size, cfg.n_embd)
        self.ctx_pos_emb = nn.Embedding(cfg.context_block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        needs_pen = cfg.pen_pos_bands > 0 or cfg.pen_last_down or cfg.local_canvas > 0
        if needs_pen:
            # Filled from the tokenizer by attach_tokenizer_tables.
            self.register_buffer("pen_deltas",
                                 torch.zeros(cfg.vocab_size, 2, dtype=torch.long))
        if cfg.pen_pos_bands > 0 or cfg.pen_last_down:
            n_xy = 2 if cfg.pen_last_down and cfg.pen_pos_bands > 0 else 1
            if cfg.pen_pos_bands == 0:
                bands = 4
            else:
                bands = cfg.pen_pos_bands
            self._pen_feat_bands = bands
            self.pen_pos_proj = nn.Linear(4 * bands * n_xy, cfg.n_embd, bias=False)
        if cfg.pen_last_down or cfg.local_canvas > 0:
            self.register_buffer("down_id", torch.tensor(8, dtype=torch.long))
            self.register_buffer("up_id", torch.tensor(9, dtype=torch.long))
        if cfg.local_canvas > 0:
            self.register_buffer("ink_rel",
                                 torch.zeros(cfg.vocab_size, 48, 2, dtype=torch.long))
            self.register_buffer("ink_valid",
                                 torch.zeros(cfg.vocab_size, 48, dtype=torch.bool))
            if cfg.canvas_linear:
                self.canvas_proj = nn.Linear(cfg.local_canvas ** 2, cfg.n_embd,
                                             bias=False)
            else:
                hid = max(8, cfg.n_embd // 4)
                self.canvas_net = nn.Sequential(
                    nn.Conv2d(1, hid, 3, stride=2, padding=1),
                    nn.GELU(approximate="tanh"),
                    nn.Conv2d(hid, hid, 3, stride=2, padding=1),
                    nn.GELU(approximate="tanh"),
                    nn.AdaptiveAvgPool2d(4),
                    nn.Flatten(),
                    nn.Linear(hid * 16, cfg.n_embd, bias=False),
                )
        self.apply(self._init_weights)
        print(f"PenTransformer parameters: {sum(p.numel() for p in self.parameters()):,}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _positions(self, idx):
        positions = self.pen_deltas[idx].cumsum(dim=1)
        if self.training and self.cfg.pen_pos_jitter > 0:
            jitter = self.cfg.pen_pos_jitter
            offset = torch.randint(-jitter, jitter + 1, (idx.size(0), 1, 2),
                                   device=idx.device)
            positions = positions + offset
        return positions

    def _fourier(self, xy, bands):
        k = torch.arange(bands, device=xy.device)
        angles = xy[..., None].float() * (torch.pi / (2.0 * 2.0 ** k))
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)

    def _last_down_offset(self, idx, positions):
        T = idx.size(1)
        steps = torch.arange(T, device=idx.device)
        down_i = torch.where(idx == self.down_id, steps, torch.full_like(idx, -1))
        last_i = down_i.cummax(dim=1).values
        gather = last_i.clamp(min=0)
        last = torch.stack([
            positions[:, :, 0].gather(1, gather),
            positions[:, :, 1].gather(1, gather),
        ], dim=-1)
        last = torch.where(last_i.unsqueeze(-1) >= 0, last, torch.zeros_like(last))
        return positions - last

    def _drawing_mask(self, idx):
        """True where token t lays ink (DOWN, or a move while the pen is down)."""
        down = idx == self.down_id
        up = idx == self.up_id
        steps = torch.arange(idx.size(1), device=idx.device)
        last_down = torch.where(down, steps, torch.full_like(idx, -1)).cummax(1).values
        last_up = torch.where(up, steps, torch.full_like(idx, -1)).cummax(1).values
        pen_after = last_down > last_up
        pen_before = F.pad(pen_after[:, :-1], (1, 0), value=False)
        return pen_before | down

    def _local_maps(self, idx, positions):
        """(B, T, N, N) soft occupancy of ink so far, around the pen.

        Ink endpoints are bilinear-splatted onto an N x N map whose pixel
        pitch is local_cell ScribeTokens cells (fractional allowed). Same
        (B, T, hist) pairing as the binary map; four weighted writes instead
        of one hard assignment.
        """
        B, T = idx.shape
        n = self.cfg.local_canvas
        cell = float(self.cfg.local_cell) if self.cfg.local_cell else 1.0
        half = n // 2
        hist = min(96, T)
        drawn = self._drawing_mask(idx)
        ink = positions.float()
        z = ink.new_zeros(B, hist - 1, 2)
        zd = drawn.new_zeros(B, hist - 1)
        ink_h = torch.cat([z, ink], 1).unfold(1, hist, 1).permute(0, 1, 3, 2)
        drawn_h = torch.cat([zd, drawn], 1).unfold(1, hist, 1)
        d = (ink_h - ink[:, :, None, :]) / cell
        u = d[..., 0] + half
        v = d[..., 1] + half
        maps = ink.new_zeros(B, T, n * n)
        x0 = u.floor()
        y0 = v.floor()
        wx = u - x0
        wy = v - y0
        nsq = n * n
        # Four separate scatters beat one fat cat: MPS scatter scales poorly
        # in the hist*4 dimension.
        for dx, dy, w in (
            (0, 0, (1 - wx) * (1 - wy)),
            (1, 0, wx * (1 - wy)),
            (0, 1, (1 - wx) * wy),
            (1, 1, wx * wy),
        ):
            xs = x0 + dx
            ys = y0 + dy
            valid = drawn_h & (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
            slot = (ys * n + xs).long().clamp(0, nsq - 1)
            maps.scatter_add_(2, slot, w * valid.to(dtype=w.dtype))
        return maps.view(B, T, n, n).clamp_(0, 1)

    def _pen_features(self, idx):
        positions = self._positions(idx)
        parts = []
        bands = getattr(self, "_pen_feat_bands", self.cfg.pen_pos_bands)
        if self.cfg.pen_pos_bands > 0:
            parts.append(self._fourier(positions, self.cfg.pen_pos_bands))
        if self.cfg.pen_last_down:
            parts.append(self._fourier(self._last_down_offset(idx, positions), bands))
        return torch.cat(parts, dim=-1) if parts else None, positions

    def forward(self, idx, context, targets=None):
        T = idx.size(1)
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        if self.cfg.pen_pos_bands > 0 or self.cfg.pen_last_down or self.cfg.local_canvas > 0:
            feats, positions = self._pen_features(idx)
            if feats is not None:
                x = x + self.pen_pos_proj(feats)
            if self.cfg.local_canvas > 0:
                with torch.no_grad():
                    maps = self._local_maps(idx, positions)
                B = idx.size(0)
                if getattr(self.cfg, "canvas_linear", False):
                    x = x + self.canvas_proj(maps.reshape(B, T, -1))
                else:
                    x = x + self.canvas_net(maps.reshape(B * T, 1, maps.size(-2), maps.size(-1))
                                            ).view(B, T, -1)

        ctx_pos = torch.arange(context.size(1), device=idx.device)
        if context.dim() == 3:
            c = self.ctx_proj(context) + self.ctx_pos_emb(ctx_pos)
            ctx_mask = (context.abs().sum(-1) != 0)[:, None, None, :]
            ctx_mask = ctx_mask | ~ctx_mask.any(-1, keepdim=True)
        else:
            c = self.ctx_emb(context) + self.ctx_pos_emb(ctx_pos)
            ctx_mask = (context != 0)[:, None, None, :]
            # A prompt of only padding (every char outside the alphabet) would
            # mask every key, and softmax over an empty row is NaN. Attending
            # uniformly to padding instead degrades to unconditional generation.
            ctx_mask = ctx_mask | ~ctx_mask.any(-1, keepdim=True)

        for block in self.blocks:
            x = block(x, c, ctx_mask)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, context, max_new_tokens, temperature=1.0, top_k=None,
                 do_sample=True, end_token=None, pad_token=None):
        """Autoregressively extend idx (B, T). If end_token is given, sequences
        that emit it are padded out with pad_token and generation stops early
        once every sequence has finished."""
        was_training = self.training
        self.eval()
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond, context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            if do_sample:
                idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            if end_token is not None:
                idx_next[done] = pad_token if pad_token is not None else end_token
                done |= idx_next.squeeze(1) == end_token
            idx = torch.cat([idx, idx_next], dim=1)
            if end_token is not None and done.all():
                break
        self.train(was_training)
        return idx


def save_checkpoint(path, model, alphabet, data_config, merges=None, optimizer=None,
                    scheduler=None, step=None, best_loss=None):
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.cfg),
        "alphabet": alphabet,
        "data_config": data_config,
        "merges": [[int(a), int(b), int(c)] for a, b, c in (merges or [])],
        "step": int(step) if step is not None else None,
        "best_loss": float(best_loss) if best_loss is not None else None,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def attach_tokenizer_tables(model, stroke_tok):
    """Copy tokenizer geometry into the model's buffers."""
    if hasattr(model, "pen_deltas"):
        model.pen_deltas.copy_(torch.as_tensor(stroke_tok.token_deltas(),
                                               device=model.pen_deltas.device))
    if hasattr(model, "down_id"):
        model.down_id.fill_(int(stroke_tok.DOWN))
        model.up_id.fill_(int(stroke_tok.UP))
    if hasattr(model, "ink_rel"):
        rel, valid = stroke_tok.token_ink_cells(max_cells=model.ink_rel.size(1))
        model.ink_rel.copy_(torch.as_tensor(rel, device=model.ink_rel.device))
        model.ink_valid.copy_(torch.as_tensor(valid, device=model.ink_valid.device))


def _model_config_from_ckpt(d):
    cfg = {f.name: f.default for f in fields(ModelConfig)}
    cfg.update(d)
    # Checkpoints from before canvas_linear used the tiny CNN.
    if "canvas_linear" not in d and d.get("local_canvas", 0) > 0:
        cfg["canvas_linear"] = False
    return ModelConfig(**cfg)


def load_checkpoint(path, device="cpu"):
    """Returns (model, checkpoint_dict). The checkpoint carries the alphabet
    and data config needed to rebuild matching tokenizers."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = PenTransformer(_model_config_from_ckpt(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    return model, checkpoint


def load_for_sampling(path, device="cpu", n_examples=200):
    """Rebuild everything a checkpoint needs to generate: model, dataset, config.

    The dataset comes back because generation reads the tokenizers and the text
    of held-out examples from it; the BPE merges and alphabet ride along in the
    checkpoint, so the tokenizer always matches the trained model.
    """
    from .config import DataConfig
    from .data import create_datasets

    model, checkpoint = load_checkpoint(path, device)
    cfg = DataConfig(**checkpoint["data_config"])
    cfg.train_size = cfg.test_size = n_examples
    text_encoder = None
    if getattr(cfg, "text_encoder", "char") == "clip+char":
        text_encoder = "clip+char"
    elif getattr(cfg, "text_encoder", "char") != "char":
        from .textenc import build_text_encoder
        text_encoder = build_text_encoder(cfg.text_encoder, device)
    _, dataset, stroke_tok, char_tok = create_datasets(
        cfg, merges=checkpoint["merges"], text_encoder=text_encoder)
    assert char_tok.alphabet == checkpoint["alphabet"], \
        "dataset alphabet does not match the checkpoint; use the training dataset"
    attach_tokenizer_tables(model, stroke_tok)
    return model, dataset, cfg, checkpoint
