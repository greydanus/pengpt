# RunPod runbook: union v3 at scale

## Pod setup

A100 80GB or H100. Everything on the pod's **local NVMe** — a venv or dataset
on network storage stalls imports for minutes (measured on a past H100 run:
48 interpreters importing transformers over FUSE; local disk took it to 2.5s).

```bash
python -m venv /workspace/venv && source /workspace/venv/bin/activate
git clone https://github.com/greydanus/pengpt && cd pengpt
pip install -e . && pip install transformers pillow
```

## Transfer from the laptop

```bash
rsync -av data/union.jsonl data/quickdraw_probe.npz <pod>:/workspace/pengpt/data/
```

union.jsonl is the v3 build (registers doodle/sketch/icon/scene/bird/creature,
Claude captions, ~250k examples). Rebuild instead with `python ink_union.py`
if the source jsonls are shipped. CLIP and ByT5 weights download on first use.

## Train

```bash
python train.py --dataset data/union.jsonl \
  --max_words 1 --augment general \
  --grid 0.012 --max_seq_length 512 --max_text_length 112 \
  --text_encoder clip+char \
  --holdout saxophone,windmill,penguin,pretzel,wheelchair,seagull \
  --hflip 0.5 --stroke_dropout 0.05 --rotate 3 --tremor 0.002 \
  --pen_pos_bands 7 --bucket_batches \
  --n_layer 12 --n_embd 512 --n_head 8 \
  --learning_rate 6e-4 --batch_size 64 \
  --steps 100000 --num_workers 8 --device cuda --out_dir out/union_gpu
```

Notes:
- ~40M params. If loss curves show train/test gap < 0.05 past 30k steps, the
  model is still capacity-bound: go wider before going longer (the 9M local
  run ended at gap 0.000).
- `--bucket_batches` is safe and fast on CUDA (2.55x measured); the MPS
  two-shape constraint does not apply.
- `--num_workers 8` is fine on Linux; the py3.14 hang is macOS-local.
- block 512 (not 448): GPU makes the fscoco/creature tails cheap.
- Register prefixes are in the data; prompt generation with them:
  "sketch: a duck", "icon: two circles side by side".

## Evaluate

```bash
python conditioning.py --checkpoint out/union_gpu/best.pt --per_label 8
python zeroshot.py --checkpoint out/union_gpu/best.pt \
  --held saxophone,windmill,penguin,pretzel,wheelchair,seagull,wolf,narwhal \
  --trained guitar,piano,duck,swan,hotdog,bicycle,chair,castle --per_label 8
```

Reference numbers to beat (9M local, v2 union, 40k steps): global test 2.687;
conditioning 1.2/12; zero-shot held-out top1 17% / rank 6.4 vs chance 8.5;
trained-label CLIP top1 12% (the register-mixture casualty — the register
tokens exist to fix exactly this number).
