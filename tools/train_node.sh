#!/bin/bash
set -u
cd ~/pengpt
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=1 PYTHONPATH=$PWD/pylib:${PYTHONPATH:-}
echo $$ > job.pid
for m in matplotlib pyarrow; do python3 -c "import $m" 2> /dev/null || python3 -m pip install -q --target pylib $m; done
rm -rf pylib/numpy pylib/numpy.libs pylib/numpy-*
inc=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
if [ ! -f $inc/Python.h ]; then
  [ -d ~/pyh/usr/include ] || (mkdir -p ~/pyh && cd ~/pyh && apt-get download -q libpython3-dev lib$(basename $inc)-dev > /dev/null 2>&1 && for f in *.deb; do dpkg -x $f .; done)
  export CPATH=$HOME/pyh/usr/include/$(basename $inc):$HOME/pyh/usr/include
fi
PLAN=${PLAN:-tools/plan.txt}
if [ "${BORROWED:-0}" != 1 ] && grep -q -E -- '--vectors|--preset[= ]sketch' $PLAN; then
  python3 -c "import torchaudio" 2> /dev/null || sudo python3 -m pip uninstall -y -q torchaudio > /dev/null 2>&1
  python3 -c "import transformers" 2> /dev/null || python3 -m pip install -q --user transformers
fi
OUT=${OUT:-out/node} gpus=$(nvidia-smi -L | wc -l) all=""
declare -a busy
failed=0
while read -r name args; do
  case "$name" in ""|\#*) continue ;; esac
  mkdir -p $OUT/$name
  resume=$([ -f $OUT/$name/last.pt ] && echo "--resume $OUT/$name/last.pt")
  case "$args" in
    *--gpus*) wait; python3 -u pengpt.py train --out_dir $OUT/$name ${ARGS:-} $args $resume >> $OUT/$name/train.log 2>&1 || failed=$(( failed + 1 )); continue ;;
  esac
  g=-1
  while [ $g -lt 0 ]; do
    for k in $(seq 0 $(( gpus - 1 ))); do
      if [ -z "${busy[$k]:-}" ] || ! kill -0 ${busy[$k]} 2> /dev/null; then g=$k; break; fi
    done
    [ $g -lt 0 ] && sleep 10
  done
  CUDA_VISIBLE_DEVICES=$g python3 -u pengpt.py train --out_dir $OUT/$name ${ARGS:-} $args $resume >> $OUT/$name/train.log 2>&1 &
  busy[$g]=$! all="$all $!"
done < $PLAN
for p in $all; do wait $p || failed=$(( failed + 1 )); done
[ $failed = 0 ] || { echo "FAILED: $failed runs in $OUT, see their train.log"; exit 1; }
echo "ALL DONE $OUT $(date)"
