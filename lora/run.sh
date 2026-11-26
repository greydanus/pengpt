#!/usr/bin/env bash
# Writer-vector baseline, then LoRA rank 8; logs in out/. Run inside tmux:  tmux new -d -s lora 'bash lora/run.sh'
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p out
$PY lora/finetune.py --rank 0 --out out/me_vector 2>&1 | grep --line-buffered -v "Redirects are" | tee out/me_vector.log
$PY lora/finetune.py --rank 8 --out out/me_lora   2>&1 | grep --line-buffered -v "Redirects are" | tee out/me_lora.log
echo "== summary"; cat out/me_vector/summary.json out/me_lora/summary.json
