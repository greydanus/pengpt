"""High-throughput embedding for scoring a whole corpus.

pengpt.quality.Embedder is the reference path: PIL images through the
HuggingFace processor. It is clear and it matches how the probe was calibrated,
but it tops out around 400 drawings/s because every drawing goes through a
Python image object and a per-image normalization.

This module does the same thing at corpus scale. Two changes:

  - render straight into a uint8 array, skipping PIL's image protocol and the
    processor entirely; normalization becomes one tensor op per batch
  - feed the model 112px images rather than letting the processor upsample to
    224px. On an H100 the model runs at 26,600 img/s at 224px and 95,900 at
    96px, and 112px is where ranking quality peaks: it scored +0.62 against
    judged tiers where 224px scored +0.59 and 64px collapsed to +0.38.

**Scale it with independent processes, not a worker pool.** A pool sends every
rendered batch back to one parent -- 37 MB of pickled arrays per batch through a
single pipe -- which left a 224-core machine 98% idle and got *slower* as
workers were added: 96 workers managed 2,289 drawings/s where 48 separate
processes reached 29,000/s. Give each process its own shard of categories and
let it render, embed, score and write by itself. `workers` therefore defaults to
1; raise it only when a single process must saturate a GPU on its own.

**Keep everything on local disk.** The same run stalled for an hour on a rented
box because the virtualenv sat on network storage: 48 interpreters importing
transformers over FUSE took minutes each and blocked in `request_wait_answer`.
Copying the venv and the corpus to local disk took import from minutes to 2.5
seconds. Every symptom looked like CPU or GPU contention and none of it was.

Scores from this path track the reference path closely but not exactly, because
the model sees genuinely different inputs. Check with `agreement`, and re-fit
the probe on this path's embeddings rather than reusing one calibrated on the
reference path.
"""

import numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_CFG = {}


def render_array(points, px=64, linewidth=2, supersample=2, pad=2):
    """Draw one trajectory straight into a (px, px) uint8 array.

    Supersampling is what antialiases the strokes, and it dominates the cost:
    8x renders at ~900/s per core, 2x at ~4,600/s. Two is enough here because
    the strokes are already several pixels wide at 64px.
    """
    from PIL import Image, ImageDraw

    points = np.asarray(points, dtype=float)
    down = points[points[:, 2] == 1]
    if len(down) < 2:
        return np.full((px, px), 255, dtype=np.uint8)

    big = px * supersample
    margin = pad * supersample
    x0, y0 = down[:, 0].min(), down[:, 1].min()
    scale = (big - 2 * margin) / max(np.ptp(down[:, 0]), np.ptp(down[:, 1]), 1e-6)

    image = Image.new("L", (big, big), 255)
    draw = ImageDraw.Draw(image)
    for chunk in np.split(points, np.flatnonzero(points[:, 2] == 0) + 1):
        chunk = chunk[chunk[:, 2] == 1]
        if len(chunk) > 1:
            draw.line([(margin + (x - x0) * scale, margin + (y - y0) * scale)
                       for x, y in chunk[:, :2]],
                      fill=0, width=int(linewidth * supersample))
    if supersample > 1:
        image = image.resize((px, px), Image.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _init(px, linewidth, supersample):
    _CFG.update(px=px, linewidth=linewidth, supersample=supersample)


def _render_chunk(chunk):
    return np.stack([render_array(p, _CFG["px"], _CFG["linewidth"],
                                  _CFG["supersample"]) for p in chunk])


class FastEmbedder:
    """CLIP embeddings at corpus scale.

    The render pool is created lazily but always before the model touches CUDA,
    because a process pool forked after CUDA initialization hangs.
    """

    def __init__(self, name="openai/clip-vit-base-patch32", px=112, linewidth=2,
                 supersample=2, workers=1, device=None, fp16=True):
        import os
        import torch
        self.torch = torch
        self.px, self.linewidth, self.supersample = px, linewidth, supersample
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        self.workers = (workers if workers is not None
                        else min(32, max(1, (os.cpu_count() or 4) // 2)))
        self._pool = self._start_pool()
        self.model = self._load_model()

    def _start_pool(self):
        if self.workers <= 1:
            return None
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        return ctx.Pool(self.workers, initializer=_init,
                        initargs=(self.px, self.linewidth, self.supersample))

    def _load_model(self):
        # transformers 5 refuses to load the composite CLIP checkpoint into the
        # vision-only class (CLIPConfig has no hidden_size); the full model
        # costs the unused text tower but loads everywhere.
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(self.name)
        model = model.to(self.device).eval()
        return model.half() if self.fp16 else model

    def _to_tensor(self, arrays):
        """uint8 (N, px, px) -> normalized (N, 3, px, px) on device."""
        torch = self.torch
        x = torch.from_numpy(arrays).to(self.device, non_blocking=True)
        x = x.float().div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        mean = torch.as_tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.as_tensor(STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.half() if self.fp16 else x

    def embed(self, points_list, batch_size=1024):
        chunks = [points_list[i:i + batch_size]
                  for i in range(0, len(points_list), batch_size)]
        rendered = (self._pool.imap(_render_chunk, chunks, chunksize=1)
                    if self._pool else (_render_chunk_local(c, self) for c in chunks))
        out = []
        with self.torch.inference_mode():
            for arrays in rendered:
                x = self._to_tensor(arrays)
                feats = self.model.get_image_features(
                    pixel_values=x, interpolate_pos_encoding=(self.px != 224))
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512))

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None


def _render_chunk_local(chunk, embedder):
    return np.stack([render_array(p, embedder.px, embedder.linewidth,
                                  embedder.supersample) for p in chunk])


def agreement(points_list, probe, fast, reference):
    """Spearman between this path's scores and the reference path's.

    The two see different pixels, so they will not agree exactly; below about
    +0.95 the probe should be recalibrated against whichever path will be used.
    """
    from .quality import spearman
    return spearman(probe.score(fast.embed(points_list)),
                    probe.score(reference.embed(points_list)))
