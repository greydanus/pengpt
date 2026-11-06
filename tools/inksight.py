import math

import numpy as np
import torch
import torch.nn.functional as F

D, H, HD, L = 768, 12, 64, 12
TEXT_LEN, IMG_LEN, MAX_LEN = 128, 197, 1024
EOS, INK0, N_INK = 1, 19815, 12000
IMG_SEG, TXT_SEG = 2, 3


def rms(x, scale, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * scale


def bucket(rel, bidirectional, n_buckets=32, max_distance=128):
    n, ret = -rel, torch.zeros_like(rel)
    if bidirectional:
        n_buckets //= 2
        ret = ret + (n < 0).long() * n_buckets
        n = n.abs()
    else:
        n = n.clamp(min=0)
    exact = n_buckets // 2
    large = exact + (torch.log(n.float() / exact + 1.1920929e-07) / math.log(max_distance / exact) * (n_buckets - exact)).long()
    return ret + torch.where(n < exact, n, large.clamp(max=n_buckets - 1))


class InkSight:

    def __init__(self, path="inksight", device=None, dtype=torch.float32, kv_dtype=None):
        import sentencepiece
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        z = np.load(f"{path}/weights.npz")
        w = {k: torch.from_numpy(z[k]).to(device=device, dtype=dtype) for k in z.files}
        self.w, self.device, self.dtype = w, device, dtype
        self.kv_dtype = kv_dtype or (torch.bfloat16 if device == "cuda" else dtype)
        self.sp = sentencepiece.SentencePieceProcessor(model_file=f"{path}/spm.model")
        per_layer = lambda prefix, l: {k[len(prefix):]: v[:, l] for k, v in w.items() if k.startswith(prefix)}
        self.enc = [per_layer("encoder__encoder__", l) for l in range(L)]
        self.dec = [per_layer("decoder__decoder__", l) for l in range(L)]
        for wl in self.dec:
            wl["qkv"] = torch.cat([wl[f"self_attention__{n}__kernel"] for n in ("query", "key", "value")], 1)
        seg = torch.tensor([IMG_SEG] * IMG_LEN + [TXT_SEG] * TEXT_LEN, device=device)
        pos = torch.arange(IMG_LEN + TEXT_LEN, device=device)
        rel, same, pair = bucket(pos[None] - pos[:, None], True), (seg[:, None] == seg[None]).to(dtype), 16 * seg[:, None] + seg[None]
        self.enc_bias = [wl["relpos_bias__rel_embedding"][:, rel] * same + wl["multimodal_relpos_bias__rel_embedding"][:, pair]
                         for wl in self.enc]
        drel = bucket(-torch.arange(MAX_LEN, device=device), False)
        self.dec_bias = torch.stack([wl["relpos_bias__rel_embedding"][:, drel] for wl in self.dec])[:, None, :, None]
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    def vit(self, img):
        w, p = self.w, "vit_encoder__Transformer__"
        ln = lambda x, n: F.layer_norm(x.float(), (D,), w[f"{n}__scale"].float(), w[f"{n}__bias"].float(), 1e-6).to(self.dtype)
        x = F.conv2d(img, w["vit_encoder__embedding__kernel"].permute(3, 2, 0, 1), w["vit_encoder__embedding__bias"], stride=16)
        x = torch.cat([w["vit_encoder__cls"].expand(len(x), -1, -1), x.flatten(2).transpose(1, 2)], 1) + w[f"{p}posembed_input__pos_embedding"]
        for i in range(12):
            b = f"{p}encoderblock_{i}__"
            a = f"{b}MultiHeadDotProductAttention_0__"
            h = ln(x, f"{b}LayerNorm_0")
            q, k, v = (torch.einsum("btd,dhe->bhte", h, w[f"{a}{n}__kernel"]) + w[f"{a}{n}__bias"][:, None] for n in ("query", "key", "value"))
            x = x + torch.einsum("bhte,hed->btd", F.scaled_dot_product_attention(q, k, v), w[f"{a}out__kernel"]) + w[f"{a}out__bias"]
            h = F.gelu(ln(x, f"{b}LayerNorm_1") @ w[f"{b}MlpBlock_0__Dense_0__kernel"] + w[f"{b}MlpBlock_0__Dense_0__bias"], approximate="tanh")
            x = x + h @ w[f"{b}MlpBlock_0__Dense_1__kernel"] + w[f"{b}MlpBlock_0__Dense_1__bias"]
        return ln(x, f"{p}encoder_norm")

    def mlp(self, x, wl):
        h = rms(x, wl["pre_mlp_layer_norm__scale"])
        return x + (F.gelu(h @ wl["mlp__wi_0__kernel"], approximate="tanh") * (h @ wl["mlp__wi_1__kernel"])) @ wl["mlp__wo__kernel"]

    def heads(self, x, kernel):
        return (x @ kernel).view(len(x), -1, H, HD).transpose(1, 2)

    def encode(self, img, tokens):
        w = self.w
        x = torch.cat([self.vit(img) @ w["encoder__image_dense_embedder__embedders_image_dense__kernel"], w["token_embedder__embedding"][tokens]], 1)
        valid = torch.cat([torch.ones(len(x), IMG_LEN, dtype=torch.bool, device=x.device), tokens > 0], 1)
        mask = torch.zeros(valid.shape, device=x.device, dtype=self.dtype).masked_fill(~valid, -1e10)
        mask = mask[:, None, :, None] + mask[:, None, None, :]
        for wl, bias in zip(self.enc, self.enc_bias):
            h = rms(x, wl["pre_attention_layer_norm__scale"])
            q, k, v = (self.heads(h, wl[f"attention__{n}__kernel"]) for n in ("query", "key", "value"))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask + bias, scale=1.0)
            x = self.mlp(x + o.transpose(1, 2).reshape(x.shape) @ wl["attention__out__kernel"], wl)
        return rms(x, w["encoder__encoder_norm__scale"]), valid

    @torch.inference_mode()
    def generate(self, img, tokens, caps=None, max_len=MAX_LEN, graphs=None, kv_dtype=None, bucket_len=128):
        w, B, dev = self.w, len(img), self.device
        graphs, kv_dtype = dev == "cuda" if graphs is None else graphs, kv_dtype or self.kv_dtype
        enc, valid = self.encode(img, tokens)
        cross = [tuple(self.heads(enc, wl[f"encoder_decoder_attention__{n}__kernel"]).to(kv_dtype) for n in ("key", "value")) for wl in self.dec]
        cmask = torch.zeros(B, 1, 1, enc.shape[1], device=dev, dtype=kv_dtype).masked_fill(~valid[:, None, None], -1e10)
        caps = torch.full((B,), max_len, device=dev) if caps is None else caps.to(dev)
        kc = torch.zeros(L, B, H, max_len, HD, device=dev, dtype=kv_dtype)
        vc = torch.zeros_like(kc)
        tok, t = torch.zeros(B, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev)
        out, done = torch.zeros(B, max_len, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
        slots = torch.arange(max_len, device=dev)

        def step(n):
            x = w["token_embedder__embedding"][tok][:, None]
            d = t - slots[:n]
            bias = self.dec_bias[..., d.clamp(min=0)] + torch.zeros(n, device=dev, dtype=self.dtype).masked_fill(d < 0, -1e10)
            for l, wl in enumerate(self.dec):
                h = rms(x, wl["pre_self_attention_layer_norm__scale"])
                q, k, v = (h[:, 0] @ wl["qkv"]).view(B, 3, H, HD).unbind(1)
                kc[l].index_copy_(2, t, k[:, :, None].to(kv_dtype))
                vc[l].index_copy_(2, t, v[:, :, None].to(kv_dtype))
                o = F.scaled_dot_product_attention(q[:, :, None].to(kv_dtype), kc[l, :, :, :n], vc[l, :, :, :n],
                                                   attn_mask=bias[l].to(kv_dtype), scale=1.0).to(self.dtype)
                x = x + o.transpose(1, 2).reshape(B, 1, D) @ wl["self_attention__out__kernel"]
                h = rms(x, wl["pre_cross_attention_layer_norm__scale"])
                o = F.scaled_dot_product_attention(self.heads(h, wl["encoder_decoder_attention__query__kernel"]).to(kv_dtype), *cross[l],
                                                   attn_mask=cmask, scale=1.0).to(self.dtype)
                x = self.mlp(x + o.transpose(1, 2).reshape(B, 1, D) @ wl["encoder_decoder_attention__out__kernel"], wl)
            nxt = (rms(x[:, 0], w["decoder__decoder_norm__scale"]) @ w["decoder__logits_dense__kernel"]).argmax(-1)
            nxt = torch.where(done, 0, torch.where(t + 1 >= caps, EOS, nxt))
            tok.copy_(nxt)
            out.index_copy_(1, t, nxt[:, None])
            done.logical_or_(nxt == EOS)
            t.add_(1)

        recorded, pool = {}, torch.cuda.graph_pool_handle() if graphs else None
        for s in range(max_len):
            n = min(max_len, (s // bucket_len + 1) * bucket_len) if graphs or dev != "cuda" else s + 1
            if not graphs:
                step(n)
            elif n in recorded:
                recorded[n].replay()
            else:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    step(n)
                torch.cuda.current_stream().wait_stream(side)
                recorded[n] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(recorded[n], pool=pool):
                    step(n)
            if s % 16 == 15 and bool(done.all()):
                break
        return out[:, :s + 1]

    def trace(self, images, texts, prompt="Derender the ink: {text}"):
        from tools.ink import scale_and_pad, detokenize
        padded = [scale_and_pad(im.convert("RGB")) for im in images]
        px = torch.from_numpy(np.stack([np.asarray(c, np.float32) for c, *_ in padded])).to(self.device)
        ids = [(self.sp.encode(prompt.replace("{text}", t)) + [EOS])[:TEXT_LEN] for t in texts]
        tokens = torch.tensor([i + [0] * (TEXT_LEN - len(i)) for i in ids], device=self.device)
        rows = []
        caps = torch.tensor([min(MAX_LEN, 200 + 90 * len(t)) for t in texts])
        for out, (_, ratio, dx, dy) in zip(self.generate(((px - 127.5) / 127.5).permute(0, 3, 1, 2).to(self.dtype), tokens, caps).tolist(), padded):
            out = out[:out.index(EOS)] if EOS in out else out
            ink = "".join(f"<ink_token_{i - INK0}>" for i in out if INK0 <= i < INK0 + N_INK)
            read = self.sp.decode([i for i in out if 2 < i < INK0 - 100])
            rows.append(([[[(x - dx) / ratio, (y - dy) / ratio] for x, y in st] for st in detokenize(ink)], read))
        return rows

