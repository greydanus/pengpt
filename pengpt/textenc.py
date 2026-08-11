import numpy as np


class ByT5Encoder:

    def __init__(self, name="google/byt5-small", device="cpu"):
        import torch
        from transformers import T5EncoderModel
        self.torch = torch
        self.name = name
        self.device = device
        self.model = T5EncoderModel.from_pretrained(name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.config.d_model
        self._cache = {}

    def _states(self, text):
        if text not in self._cache:
            ids = [b + 3 for b in text.encode("utf-8")] + [1]
            batch = self.torch.tensor([ids], device=self.device)
            with self.torch.inference_mode():
                out = self.model(input_ids=batch).last_hidden_state
            self._cache[text] = out[0, :-1].float().cpu().numpy()
        return self._cache[text]

    def encode(self, text, length):
        parts = []
        for i, word in enumerate(text.split()):
            if i:
                parts.append(self._states(" "))
            parts.append(self._states(word))
        states = (np.concatenate(parts) if parts
                  else np.zeros((0, self.dim), dtype=np.float32))
        out = np.zeros((length, self.dim), dtype=np.float32)
        n = min(length, len(states))
        out[:n] = states[:n]
        return out


class HybridEncoder:

    def __init__(self, byt5_name="google/byt5-small",
                 clip_name="openai/clip-vit-base-patch32", device="cpu"):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        self.byt5 = ByT5Encoder(byt5_name, device)
        self.torch = torch
        self.device = device
        self.clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip_dim = self.clip.config.projection_dim
        self.dim = self.clip_dim + self.byt5.dim
        self.clip_tok = CLIPTokenizer.from_pretrained(clip_name)
        self._cache = {}

    def _global(self, text):
        if text not in self._cache:
            t = self.clip_tok([text], return_tensors="pt",
                              padding=True, truncation=True).to(self.device)
            with self.torch.inference_mode():
                v = self.clip.get_text_features(**t)[0].float().cpu().numpy()
            self._cache[text] = v / (np.linalg.norm(v) + 1e-8)
        return self._cache[text]

    def encode(self, text, length):
        chars = self.byt5.encode(text, length)
        out = np.zeros((length, self.dim), dtype=np.float32)
        live = np.abs(chars).sum(-1) != 0
        out[live, :self.clip_dim] = self._global(text)
        out[:, self.clip_dim:] = chars
        return out


class CharClipEncoder:

    def __init__(self, char_tok, clip_name="openai/clip-vit-base-patch32",
                 device="cpu"):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        self.char_tok = char_tok
        self.torch = torch
        self.device = device
        self.clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip_dim = self.clip.config.projection_dim
        self.dim = self.clip_dim + char_tok.vocab_size
        self.clip_tok = CLIPTokenizer.from_pretrained(clip_name)
        self._cache = {}

    def _word(self, word):
        if word not in self._cache:
            t = self.clip_tok([word], return_tensors="pt",
                              padding=True, truncation=True).to(self.device)
            with self.torch.inference_mode():
                v = self.clip.get_text_features(**t)[0].float().cpu().numpy()
            self._cache[word] = v
        return self._cache[word]

    def _global(self, text):
        vecs = [self._word(w) for w in text.split()]
        if not vecs:
            return np.zeros(self.clip_dim, dtype=np.float32)
        v = np.mean(vecs, axis=0)
        return v / (np.linalg.norm(v) + 1e-8)

    def encode(self, text, length):
        ids = self.char_tok.encode(text, length)
        out = np.zeros((length, self.dim), dtype=np.float32)
        live = ids > 0
        out[live, :self.clip_dim] = self._global(text)
        out[live, self.clip_dim + ids[live]] = 1.0
        return out


def build_text_encoder(name, device="cpu", char_tok=None):
    if name == "char":
        return None
    if name == "hybrid":
        return HybridEncoder(device=device)
    if name == "clip+char":
        return CharClipEncoder(char_tok, device=device)
    return ByT5Encoder(name, device)
