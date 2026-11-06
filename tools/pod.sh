#!/bin/bash
set -e -o pipefail
: ${HOST:?set HOST=root@ip} ${PORT:?set PORT}
O="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ${KEY:+-i $KEY -o IdentitiesOnly=yes}"
S=${S:-ssh -p $PORT $O $HOST}
DIR=${DIR:-data/raw/patrickhand}
R=${R:-/root/pengpt}
W=run/$(basename "$DIR")
PY=${PY:-python3}
TFPY=/root/tf/bin/python
PROMPT=${PROMPT:-Derender the ink: {text}}
MANIFEST=${MANIFEST:-$DIR/images/manifest.jsonl}
ENV='q=$(cat /sys/fs/cgroup/cpu.max 2> /dev/null | awk "\$1 != \"max\" {print int(\$1 / \$2)}"); export TF_NUM_INTRAOP_THREADS=${q:-8} TF_NUM_INTEROP_THREADS=2'
cmd=$1; shift || true
case $cmd in
setup)
  scp -q -P $PORT $O tools/ink.py tools/inksight.py $HOST:/root/
  $S bash -s <<EOF
set -e
mkdir -p $R/tools && mv /root/ink.py /root/inksight.py $R/tools/
pip install -q --break-system-packages sentencepiece pillow scipy matplotlib scikit-image
python3 -m venv /root/tf 2> /dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv /root/tf; }
/root/tf/bin/pip install -q 'tensorflow[and-cuda]==2.20.0' tensorflow-text==2.20.1 tf-keras==2.20.1 'huggingface_hub<1' pillow scipy
cd $R && [ -f inksight/weights.npz ] || CUDA_VISIBLE_DEVICES= $TFPY -m tools.ink export --out inksight
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs, torch', torch.__version__)"
echo SETUP OK
EOF
  ;;
push)
  COPYFILE_DISABLE=1 tar czf /tmp/pengpt_push.tgz --no-xattrs tools/ink.py tools/inksight.py "$DIR/images"
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
  CUDA_VISIBLE_DEVICES=\$g setsid nohup bash -c "$PY -u -m tools.ink trace --manifest $W/manifest_\$g.jsonl --out $W/traces_\$g.jsonl --prompts '$PROMPT' --batch ${BATCH:-256}; echo EXIT \\\$?" > $W/trace_\$g.log 2>&1 < /dev/null &
  echo \$! > run/gpu\$g.pid
  echo $W > run/gpu\$g.font
  echo "GPU \$g: \$(wc -l < $W/manifest_\$g.jsonl) words, prompt '$PROMPT'"
done
EOF
  ;;
retry)
  $0 qc
  MANIFEST=$W/accepted_retry.jsonl PROMPT=${PROMPT_RETRY:-"Recognize and derender."} $0 trace ;;
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
sync)
  scp -q -P $PORT $O tools/ink.py tools/inksight.py $HOST:$R/tools/
  $S "cd $R && cat $W/*_out.jsonl $W/traces_*.jsonl 2> /dev/null > $W/all.jsonl; true" ;;
peek)
  $0 sync
  $S "cd $R && $PY -m tools.ink peek --manifest $DIR/images/manifest.jsonl --traces $W/all.jsonl \
      --n ${N:-12} ${SEED:+--seed $SEED} ${WORST:+--worst $WORST} ${ONLY:+--prompt '$ONLY'} --out $W/peek.png"
  scp -q -P $PORT $O $HOST:$R/$W/peek.png "$DIR/peek.png" && echo "sheet: $DIR/peek.png" ;;
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
  $0 sync
  $S "cd $R && $PY -m tools.ink qc --manifest $DIR/images/manifest.jsonl --traces $W/all.jsonl --out $W/accepted.jsonl"
  scp -q -P $PORT $O "$HOST:$R/$W/accepted*.jsonl" "$DIR/" ;;
kill)
  $S "cd $R && for f in run/gpu*.pid; do p=\$(cat \$f); kill -TERM -- -\$p 2> /dev/null || { pkill -TERM -P \$p; kill \$p; } 2> /dev/null || true; rm -f \$f; done; echo killed" ;;
stop)
  $S "source /etc/rp_environment && runpodctl stop pod \$RUNPOD_POD_ID" ;;
*)
  echo "usage: HOST=root@ip PORT=port [KEY=~/.ssh/key] [DIR=data/raw/<font>] tools/pod.sh setup|push|bench|trace|retry|status|pull|sync|peek|qc|done|watch|kill|stop" ;;
esac
