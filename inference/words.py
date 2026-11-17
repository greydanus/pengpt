import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pen import fetch


class Words:

    def __init__(self, path="siglip-text.npz"):
        from transformers import PreTrainedTokenizerFast, SiglipTextConfig, SiglipTextModel
        z = np.load(fetch(path))
        full = lambda k: z[k].astype(np.float32) * z[k + ".scale"] if k + ".scale" in z.files else z[k].astype(np.float32 if z[k].dtype.kind == "f" else z[k].dtype)
        meta, weights = json.loads(str(z["meta"])), {k: torch.from_numpy(full(k)) for k in z.files if k != "meta" and not k.endswith(".scale")}
        fills = {n: getattr(torch.Tensor, n) for n in ("normal_", "uniform_")}
        try:
            for n in fills:
                setattr(torch.Tensor, n, lambda t, *a, **k: t)
            self.model = SiglipTextModel(SiglipTextConfig(**meta["config"]))
        finally:
            for n, f in fills.items():
                setattr(torch.Tensor, n, f)
        names, self.remap = self.model.state_dict(), weights.pop("remap", None)
        self.model.load_state_dict({k if k in names else k.removeprefix("text_model."): v for k, v in weights.items()})
        self.model.eval()
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8").write(meta["tokenizer"])
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(d, "tokenizer.json"), pad_token="<pad>")

    def __call__(self, text):
        ids = self.tokenizer([text.lower()], padding="max_length", truncation=True, max_length=64, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            out = self.model(input_ids=ids if self.remap is None else self.remap[ids].long()).pooler_output[0]
        return (out / out.norm()).numpy().astype(np.float32)
