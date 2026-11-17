import torch
import torch.nn as nn
from torch.nn import functional as F

IGNORE_INDEX = -1


class GravesLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size > 0, "fill in the derived ModelConfig fields first"
        self.cfg = cfg
        E, K = cfg.n_embd, cfg.window
        self.tok_emb = nn.Embedding(cfg.vocab_size, E)
        if cfg.n_sources > 0:
            self.source_emb = nn.Embedding(cfg.n_sources, E)
        if cfg.context_slots > 0:
            self.told = nn.Linear(cfg.context_dim, E)
            self.begin = nn.Linear(cfg.context_dim, 2 * cfg.n_layer * E)
            self.first = nn.LSTM(2 * E, E, batch_first=True)
        else:
            self.ctx_emb = nn.Embedding(cfg.context_vocab_size, E)
            self.first = nn.LSTMCell(2 * E, E)
        self.window = nn.Linear(E, 3 * K)
        self.rest = nn.ModuleList(nn.LSTM(3 * E, E, batch_first=True) for _ in range(cfg.n_layer - 1))
        self.head = nn.Linear(cfg.n_layer * E, cfg.vocab_size)
        self.register_buffer("pen_deltas", torch.zeros(cfg.vocab_size, 2, dtype=torch.long))
        nn.init.zeros_(self.window.weight)
        with torch.no_grad():
            self.window.bias[2 * K:] = -3.0
        print(f"GravesLSTM parameters: {sum(p.numel() for p in self.parameters()):,}")

    def optimizers(self, cfg):
        return [torch.optim.AdamW(self.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))]

    def start(self, B, device):
        E, zeros = self.cfg.n_embd, lambda *shape: torch.zeros(*shape, device=device)
        return dict(first=(zeros(B, E), zeros(B, E)), kappa=zeros(B, self.cfg.window), w=zeros(B, E),
                    rest=[(zeros(1, B, E), zeros(1, B, E)) for _ in self.rest])

    def embed(self, idx, source=None):
        x = self.tok_emb(idx)
        if self.cfg.n_sources > 0:
            w = torch.zeros(len(idx), dtype=torch.long, device=idx.device) if source is None else source
            if self.training or self.cfg.source_dropout >= 1:
                w = torch.where(torch.rand(w.shape, device=w.device) < self.cfg.source_dropout, 0, w)
            x = x + self.source_emb(w)[:, None]
        return x

    def run_told(self, idx, context, source=None, state=None):
        x, B, E = self.embed(idx, source), len(idx), self.cfg.n_embd
        context = context.to(self.told.weight.dtype)
        w = self.told(context)[:, None].expand(-1, x.size(1), -1).to(x.dtype)
        if state is None:
            h = torch.tanh(self.begin(context)).view(B, self.cfg.n_layer, 2, E).permute(1, 2, 0, 3).to(x.dtype)
            state = [(h[l, 0][None].contiguous(), h[l, 1][None].contiguous()) for l in range(self.cfg.n_layer)]
        below, s = self.first(torch.cat([x, w], -1), state[0])
        kept, after = [below], [s]
        for lstm, s in zip(self.rest, state[1:]):
            below, s = lstm(torch.cat([x, below, w], -1), s)
            kept.append(below)
            after.append(s)
        return self.head(torch.cat(kept, -1)), after

    def run(self, idx, context, source=None, state=None):
        if context.is_floating_point():
            return self.run_told(idx, context, source, state)
        B, T = idx.shape
        x, c, mask = self.embed(idx, source), self.ctx_emb(context), (context != 0).float()
        u = torch.arange(context.size(1), device=idx.device).float()[None, None]
        state = state or self.start(B, idx.device)
        (h, m), kappa, w = state["first"], state["kappa"], state["w"]
        hs, ws = [], []
        for t in range(T):
            h, m = self.first(torch.cat([x[:, t], w], -1), (h, m))
            alpha, beta, step = self.window(h).float().exp().chunk(3, -1)
            kappa = kappa + step
            phi = (alpha[..., None] * torch.exp(-beta[..., None] * (kappa[..., None] - u) ** 2)).sum(1) * mask
            w = torch.einsum("bu,bue->be", phi.to(c.dtype), c)
            hs.append(h)
            ws.append(w)
        below, w_seq, kept, rest = torch.stack(hs, 1), torch.stack(ws, 1), [torch.stack(hs, 1)], []
        for lstm, s in zip(self.rest, state["rest"]):
            below, s = lstm(torch.cat([x, below, w_seq], -1), s)
            kept.append(below)
            rest.append(s)
        return self.head(torch.cat(kept, -1)), dict(first=(h, m), kappa=kappa, w=w, rest=rest)

    def forward(self, idx, context, targets=None, source=None):
        logits, _ = self.run(idx, context, source)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=IGNORE_INDEX)
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, context, max_new_tokens, temperature=1.0, top_k=None,
                 do_sample=True, end_token=None, pad_token=None, source=None, generator=None, guidance=1.0, blank=None):
        was_training = self.training
        self.eval()
        B, T0 = idx.shape
        n, dev = min(T0 + max_new_tokens, self.cfg.block_size), idx.device
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        tokens = list(idx.split(1, dim=1))
        guided = guidance != 1 and blank is not None and context.is_floating_point()
        if guided:
            context = torch.cat([context, blank.to(context).expand(B, -1)])
            source = None if source is None else source.repeat(2)
        twice = lambda t: t.repeat(2, 1) if guided else t
        mix = lambda l: l[B:] + guidance * (l[:B] - l[B:]) if guided else l
        logits, state = self.run(twice(idx), context, source)
        logits = mix(logits)
        for step in range(T0 - 1, n - 1):
            logits = logits[:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.where(torch.as_tensor(do_sample, device=dev).view(-1, 1),
                              torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator),
                              logits.argmax(dim=-1, keepdim=True))
            if end_token is not None:
                nxt[done] = pad_token if pad_token is not None else end_token
                done |= nxt.squeeze(1) == end_token
            tokens.append(nxt)
            if end_token is not None and done.all():
                break
            logits, state = self.run(twice(nxt), context, source, state)
            logits = mix(logits)
        self.train(was_training)
        return torch.cat(tokens, dim=1)
