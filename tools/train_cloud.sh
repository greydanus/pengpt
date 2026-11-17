#!/bin/bash
set -e -o pipefail
cd "$(dirname "$0")/.."
export NAME=${NAME:-train} DIR=pengpt
export VM=${VM:-pengpt-$NAME}
PLAN=${PLAN:-tools/plan.txt} OUT=${OUT:-out/node}
runs=$(grep -v -E '^[[:space:]]*(#|$)' $PLAN || true)
if [ -z "${DATA:-}" ]; then
  DATA=$(echo "$runs" | grep -o -E 'data/[^ ,:]*\.(sketch|hand)\.parquet' || true)
  for p in $(echo "$runs" | grep -o -E -- '--preset[= ][^ ]+' | cut -c10- | sort -u); do
    DATA="$DATA $(python3 -c "from utils import PRESETS; print(' '.join(p.split(':')[0] for p in PRESETS['$p']['dataset'].split(',')))")"
  done
  DATA=$(echo $DATA | tr ' ' '\n' | sort -u | tr '\n' ' ')
fi
MORE="" SKETCHES=$(echo $DATA | tr ' ' '\n' | grep '\.sketch\.parquet$' || true)
if echo "$runs" | grep -q -E -- '--vectors|--preset[= ]sketch'; then
  for d in $SKETCHES; do MORE="$MORE ${d%.sketch.parquet}.vectors.parquet"; done
fi
for d in $SKETCHES; do c=${d%.sketch.parquet}.captions.parquet; [ -f $c ] && MORE="$MORE $c"; done
for f in $DATA $MORE; do [ -f $f ] || { echo "$f is not there; $PLAN needs it"; exit 1; }; done
[ "${CODE:-0}" != 1 ] || DATA="" MORE=""
mkdir -p out
ARGS=${ARGS:---sample_every 100000 --loader_workers 10}
[ "${REUSE:-0}" = 1 ] && [ -f out/${VM}_push.tgz ] || COPYFILE_DISABLE=1 tar czhf out/${VM}_push.tgz pengpt.py graves.py polar.py strokes.py vectors.py utils.py inference/pen.py tools/train_node.sh $PLAN data/sketch_tokens.json $DATA $(echo $MORE | tr ' ' '\n' | sort -u)
SPECS=${SPECS:-"spot:a2-highgpu-8g flex:a2-highgpu-8g spot:a2-ultragpu-8g flex:a2-ultragpu-8g"} tools/gcp.sh find
PUSH=out/${VM}_push.tgz JOB="ARGS='$ARGS' PLAN=$PLAN OUT=$OUT BORROWED=${BORROWED:-0} setsid nohup bash tools/train_node.sh" LOG=node.log \
  SHOW='for f in '$OUT'/*/train.log; do echo $(basename $(dirname $f)): $(grep -E "^step [0-9]+ \| train" $f | tail -n 1 | cut -c1-110); done' tools/gcp.sh watch
tools/gcp.sh ssh "cd ~/pengpt && for d in $OUT/*/; do [ -f \$d/best.pt ] && python3 inference/pen.py export --checkpoint \$d/best.pt --out \$d/model.npz > /dev/null; done; tar czf out.tgz \$(find $OUT -name train.log -o -name metrics.jsonl -o -name model.npz -o -name last.npz -o -name 'step_*.png') \$(for d in $OUT/*/; do [ -f \$d/model.npz ] || ls \$d/best.pt; done)"
tools/gcp.sh get pengpt/out.tgz out/${VM}_out.tgz
tar xzf out/${VM}_out.tgz
echo "pulled $(find $OUT -name model.npz -o -name best.pt | wc -l) runs -> $OUT"
[ "${DELETE:-0}" != 1 ] || tools/gcp.sh delete
