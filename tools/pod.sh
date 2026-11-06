#!/bin/bash
set -e -o pipefail
: ${HOST:?set HOST=root@ip} ${PORT:?set PORT}
O="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ${KEY:+-i $KEY -o IdentitiesOnly=yes}"
S=${S:-ssh -p $PORT $O $HOST}
DIR=${DIR:-data/raw/patrickhand}
R=${R:-/root/pengpt}
W=run/$(basename "$DIR")
PY=${PY:-/root/tf/bin/python}
PROMPT=${PROMPT:-Recognize and derender.}
MANIFEST=${MANIFEST:-$DIR/images/manifest.jsonl}
ENV='q=$(cat /sys/fs/cgroup/cpu.max 2> /dev/null | awk "\$1 != \"max\" {print int(\$1 / \$2)}"); export TF_NUM_INTRAOP_THREADS=${q:-8} TF_NUM_INTEROP_THREADS=2'
cmd=$1; shift || true
case $cmd in
setup)
  $S bash -s <<EOF
set -e
python3 -m venv /root/tf 2> /dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv /root/tf; }
/root/tf/bin/pip install -q 'tensorflow[and-cuda]==2.20.0' tensorflow-text==2.20.1 tf-keras==2.20.1 'huggingface_hub<1' pillow scipy
$PY -c "import tensorflow as tf; x = tf.reduce_sum(tf.ones((4096, 4096))); print(len(tf.config.list_physical_devices('GPU')), 'GPUs, test op ran on', x.device)"
$PY -c "import tensorflow_text; from huggingface_hub import from_pretrained_keras; from_pretrained_keras('Derendering/InkSight-Small-p')" > /dev/null 2>&1
echo SETUP OK
EOF
  ;;
push)
  COPYFILE_DISABLE=1 tar czf /tmp/pengpt_push.tgz --no-xattrs tools/ink.py "$DIR/images"
  scp -P $PORT $O /tmp/pengpt_push.tgz $HOST:/root/pengpt_push.tgz
  $S "mkdir -p $R/$W && tar xzf /root/pengpt_push.tgz -C $R && echo pushed \$(wc -l < $R/$DIR/images/manifest.jsonl) words" ;;
bench)
  $S bash -s <<EOF
$ENV
cd $R && mkdir -p $W && head -n ${WORDS:-256} $MANIFEST > $W/bench.jsonl
for b in ${BATCHES:-8 32 128}; do
  rm -f $W/bench_out.jsonl
  echo "batch \$b: \$(CUDA_VISIBLE_DEVICES=0 $PY -u -m tools.ink trace --manifest $W/bench.jsonl --out $W/bench_out.jsonl --batch \$b 2>&1 | tail -n 1)"
done
EOF
  ;;
trace)
  $S bash -s <<EOF
$ENV
cd $R && mkdir -p $W
N=\$(( \$(nvidia-smi -L | wc -l) ))
for g in \$(seq 0 \$((N - 1))); do
  if [ -f run/gpu\$g.pid ] && kill -0 \$(cat run/gpu\$g.pid) 2> /dev/null; then echo "GPU \$g busy: \$(cat run/gpu\$g.font)"; continue; fi
  awk -v n=\$N -v g=\$g 'NR % n == g' $MANIFEST > $W/manifest_\$g.jsonl
  CUDA_VISIBLE_DEVICES=\$g setsid nohup bash -c "$PY -u -m tools.ink trace --manifest $W/manifest_\$g.jsonl --out $W/traces_\$g.jsonl --prompts '$PROMPT' --batch ${BATCH:-16}; echo EXIT \\\$?" > $W/trace_\$g.log 2>&1 < /dev/null &
  echo \$! > run/gpu\$g.pid
  echo $W > run/gpu\$g.font
  echo "GPU \$g: \$(wc -l < $W/manifest_\$g.jsonl) words, prompt '$PROMPT'"
done
EOF
  ;;
retry)
  ${PYTHON:-python3} -c "
import json, sys
rows = [json.loads(l) for l in open('$DIR/accepted_rejected.jsonl')]
rows = [{k: r[k] for k in ('image', 'text', 'baseline')} for r in rows if r.get('qc', {}).get('reason') != 'no trace']
open('$DIR/retry.jsonl', 'w').writelines(json.dumps(r) + '\n' for r in rows)
print(len(rows), 'rejected words to retry')"
  scp -q -P $PORT $O "$DIR/retry.jsonl" $HOST:$R/$W/retry.jsonl
  MANIFEST=$W/retry.jsonl PROMPT=${PROMPT_RETRY:-"Derender the ink: {text}"} $0 trace ;;
status)
  $S bash -s <<EOF
cd $R
for f in $W/trace_*.log; do echo "\$f (\$(( \$(date +%s) - \$(stat -c %Y \$f) ))s since last line): \$(tail -n 1 \$f)"; done
echo "\$(cat $W/traces_*.jsonl 2> /dev/null | wc -l) traces for \$(wc -l < $DIR/images/manifest.jsonl) words"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2> /dev/null || true
EOF
  ;;
pull)
  $S "cat $R/$W/*_out.jsonl $R/$W/traces_*.jsonl 2> /dev/null; true" > "$DIR/traces.jsonl"
  echo "pulled $(wc -l < "$DIR/traces.jsonl") traces to $DIR/traces.jsonl" ;;
peek)
  $0 pull
  ${PYTHON:-python3} -m tools.ink peek --manifest "$DIR/images/manifest.jsonl" --traces "$DIR/traces.jsonl" \
    --n ${N:-12} ${SEED:+--seed $SEED} ${WORST:+--worst $WORST} --out "$DIR/peek.png" ;;
done)
  $S "cd $R && n=\$(ls $W/trace_*.log | wc -l) && [ \$n -gt 0 ] && [ \$n -eq \$(grep -l '^EXIT' $W/trace_*.log | wc -l) ]" ;;
watch)
  while true; do
    $0 status || true
    $0 pull || true
    if $0 done; then
      echo "all shards finished"
      [ "$STOP" = 1 ] && $0 stop
      break
    fi
    sleep ${EVERY:-600}
  done ;;
qc)
  ${PYTHON:-python3} -m tools.ink qc --manifest "$DIR/images/manifest.jsonl" --traces "$DIR/traces.jsonl" \
    --out "$DIR/accepted.jsonl" --sheet "$DIR/qc_sheet.png" ;;
kill)
  $S "cd $R && for f in run/gpu*.pid; do p=\$(cat \$f); kill -TERM -- -\$p 2> /dev/null || { pkill -TERM -P \$p; kill \$p; } 2> /dev/null || true; rm -f \$f; done; echo killed" ;;
stop)
  $S "source /etc/rp_environment && runpodctl stop pod \$RUNPOD_POD_ID" ;;
*)
  echo "usage: HOST=root@ip PORT=port [KEY=~/.ssh/key] [DIR=data/raw/<font>] tools/pod.sh setup|push|bench|trace|retry|status|pull|peek|done|watch|qc|kill|stop" ;;
esac
