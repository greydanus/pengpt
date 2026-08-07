#!/bin/bash
# Report conditioning rank at each new eval, so a run that is drawing the corpus
# average for every prompt is visible while it is still cheap to stop.
LABELS="cat,apple,car,fish,tree,house,star,umbrella,clock,ladder,banana,bicycle,pizza,book,snowman,penguin,mouth,toe,triangle,sun"
CKPT=${1:-out/quickdraw/last.pt}
seen=""
while true; do
  if [ -f "$CKPT" ]; then
    step=$(python3 -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])" 2>/dev/null)
    if [ -n "$step" ] && [ "$step" != "$seen" ]; then
      seen=$step
      python3 conditioning.py --checkpoint "$CKPT" --per_label 8 --labels "$LABELS" 2>/dev/null|tail -1
    fi
  fi
  sleep 120
done
