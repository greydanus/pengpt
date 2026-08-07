"""A GPT-style decoder with cross-attention over a character context.

The architecture follows Karpathy's makemore/nanoGPT lineage: pre-LayerNorm
blocks of (causal self-attention, cross-attention, MLP). Attention uses
torch.nn.functional.scaled_dot_product_attention, which dispatches to flash
attention where available.
"""

import math
from dataclasses import asdict

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
        self.ctx_emb = nn.Embedding(cfg.context_vocab_size, cfg.n_embd)
        self.ctx_pos_emb = nn.Embedding(cfg.context_block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        print(f"PenTransformer parameters: {sum(p.numel() for p in self.parameters()):,}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, context, targets=None):
        T = idx.size(1)
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        ctx_pos = torch.arange(context.size(1), device=idx.device)
        c = self.ctx_emb(context) + self.ctx_pos_emb(ctx_pos)
        ctx_mask = None
        if context.dim() == 2:
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
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


def load_checkpoint(path, device="cpu"):
    """Returns (model, checkpoint_dict). The checkpoint carries the alphabet
    and data config needed to rebuild matching tokenizers."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = PenTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
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
    _, dataset, stroke_tok, char_tok = create_datasets(cfg, merges=checkpoint["merges"])
    assert char_tok.alphabet == checkpoint["alphabet"], \
        "dataset alphabet does not match the checkpoint; use the training dataset"
    return model, dataset, cfg, checkpoint
