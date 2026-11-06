#!/bin/bash
set -e -o pipefail
: ${HOST:?set HOST=root@ip} ${PORT:?set PORT}
O="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ${KEY:+-i $KEY -o IdentitiesOnly=yes}"
S=${S:-ssh -p $PORT $O $HOST}
R=${R:-/root/pengpt}
MANIFEST=${MANIFEST:-}
NAME=${NAME:-$(basename "${MANIFEST%.jsonl}")}
OUT=${OUT:-data/raw/harvest/$NAME}
cmd=$1; shift || true
need() { : ${MANIFEST:?set MANIFEST=path/to/manifest.jsonl}; }
case $cmd in
setup)
  $S "mkdir -p $R/tools" && scp -q -P $PORT $O tools/*.py $HOST:$R/tools/
  $S bash -s <<EOF
set -e
pip install -q --break-system-packages sentencepiece pillow scipy matplotlib scikit-image
cd $R
if [ ! -f inksight/weights.npz ]; then
  python3 -m venv /root/tf 2> /dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv /root/tf; }
  /root/tf/bin/pip install -q 'tensorflow[and-cuda]==2.20.0' tensorflow-text==2.20.1 tf-keras==2.20.1 'huggingface_hub<1' pillow scipy
  CUDA_VISIBLE_DEVICES= /root/tf/bin/python -m tools.ink export --out inksight
fi
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs, torch', torch.__version__)"
echo SETUP OK
EOF
  ;;
push)
  need
  python3 -c "import json, sys; print('\n'.join(json.loads(l)['image'] for l in open(sys.argv[1])))" "$MANIFEST" > /tmp/pengpt_push.txt
  echo "$MANIFEST" >> /tmp/pengpt_push.txt
  ls tools/*.py >> /tmp/pengpt_push.txt
  COPYFILE_DISABLE=1 tar czf /tmp/pengpt_push.tgz --no-xattrs -T /tmp/pengpt_push.txt
  scp -q -P $PORT $O /tmp/pengpt_push.tgz $HOST:/root/pengpt_push.tgz
  $S "mkdir -p $R && tar xzf /root/pengpt_push.tgz -C $R && echo pushed \$(wc -l < $R/$MANIFEST) rows" ;;
harvest)
  need
  $S bash -s <<EOF
cd $R && mkdir -p run/$NAME
if [ -f run/$NAME/pid ] && kill -0 \$(cat run/$NAME/pid) 2> /dev/null; then echo "already running"; exit 0; fi
setsid nohup python3 -u -m tools.harvest --manifest $MANIFEST --out run/$NAME $* > run/$NAME/log 2>&1 < /dev/null &
echo \$! > run/$NAME/pid
echo "harvest $NAME started"
EOF
  ;;
status)
  $S "cd $R && tail -n 3 run/$NAME/log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" ;;
pull)
  mkdir -p "$OUT"
  for f in accepted.jsonl rejected.jsonl sources.json; do scp -q -P $PORT $O "$HOST:$R/run/$NAME/$f" "$OUT/" 2> /dev/null || true; done
  echo "pulled $(wc -l < "$OUT/accepted.jsonl") accepted words to $OUT" ;;
peek)
  need
  $S "cd $R && python3 -m tools.ink peek --manifest $MANIFEST --traces run/$NAME/accepted.jsonl --n ${N:-12} ${SEED:+--seed $SEED} --out run/$NAME/peek.png"
  mkdir -p "$OUT" && scp -q -P $PORT $O "$HOST:$R/run/$NAME/peek.png" "$OUT/peek.png" && echo "sheet: $OUT/peek.png" ;;
kill)
  $S "p=\$(cat $R/run/$NAME/pid) && kill -TERM -- -\$p 2> /dev/null || kill \$p; echo killed $NAME" ;;
stop)
  $S "source /etc/rp_environment && runpodctl stop pod \$RUNPOD_POD_ID" ;;
*)
  echo "usage: HOST=root@ip PORT=port [KEY=~/.ssh/key] MANIFEST=m.jsonl tools/pod.sh setup|push|harvest|status|pull|peek|kill|stop" ;;
esac
