#!/bin/bash
# A GPU node on Google Cloud for a job that can be resumed: find one (spot, then flex-start, across zones), watch a job on it through
# preemptions, copy files to and from it, delete it. tools/train_cloud.sh is the job. Settings of your own (PROJECT, SUBNET, OWNER,
# G for a gcloud that is not on the PATH) can live in tools/gcp.env, which git ignores.
set -e -o pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ ! -f $ROOT/tools/gcp.env ] || . $ROOT/tools/gcp.env
G=${G:-gcloud}
PROJECT=${PROJECT:-$($G config get-value project 2> /dev/null || true)}
[ -n "$PROJECT" ] || { echo "set PROJECT to a Google Cloud project (or gcloud config set project)"; exit 1; }
NAME=${NAME:-train}
VM=${VM:-pengpt-$NAME}
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east4-c us-west1-b us-west4-b us-west3-b europe-west4-a europe-west4-b asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-northeast1-a me-west1-b"}
# Biggest first. GPU=h100 asks for eight H100s before any A100s.
A100="spot:a2-highgpu-8g flex:a2-highgpu-8g spot:a2-ultragpu-8g flex:a2-ultragpu-8g spot:a2-highgpu-4g flex:a2-highgpu-4g spot:a2-highgpu-2g flex:a2-highgpu-2g spot:a2-highgpu-1g flex:a2-highgpu-1g"
[ "${GPU:-a100}" = h100 ] && H100="spot:a3-highgpu-8g flex:a3-highgpu-8g spot:a3-megagpu-8g flex:a3-megagpu-8g " || H100=""
SPECS=${SPECS:-$H100$A100}
STATE=$ROOT/out/gcp && mkdir -p $STATE
ZONE=${ZONE:-$(cat $STATE/$VM.zone 2>/dev/null || true)}
A="--project $PROJECT --zone $ZONE --tunnel-through-iap --quiet"
run() { perl -e 'alarm shift; exec @ARGV' ${TIMEOUT:-300} $G compute ssh $VM $A --command "$1"; }
state() { $G compute instances describe $VM --project $PROJECT --zone $1 --format="value(status)" 2>/dev/null || true; }
quiet() { grep -v -i -E "warning|numpy|tunnel|please see|^$" || true; }
run_id() { { tar tvzf $PUSH; echo "$JOB"; } | cksum | cut -d' ' -f1; }
cmd=$1; shift || true

case $cmd in
create)
  $G compute instances create $VM --project $PROJECT --zone $ZONE --machine-type ${TYPE:-a2-highgpu-8g} \
    ${SUBNET:+--subnet $SUBNET-${ZONE%-*}} --no-address --no-service-account --no-scopes --maintenance-policy TERMINATE \
    $(case ${PROV:-spot} in (flex) echo "--provisioning-model=FLEX_START --max-run-duration=${HOURS:-6}h --instance-termination-action=DELETE --request-valid-for-duration=10m --reservation-affinity=none" ;; (standard) echo "--provisioning-model=STANDARD" ;; (*) echo "--provisioning-model=SPOT --instance-termination-action=STOP" ;; esac) \
    --image-project deeplearning-platform-release --image-family pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --boot-disk-size 200GB --boot-disk-type pd-ssd --metadata install-nvidia-driver=True,enable-oslogin=TRUE \
    --labels purpose=pengpt${OWNER:+,owner=$OWNER} ;;
find)
  [ -n "$ZONE" ] && [ -n "$(state $ZONE)" ] && { echo "$VM already in $ZONE"; exit 0; }
  while true; do
    for spec in $SPECS; do
      set -- ${spec/:/ }
      for z in $ZONES; do
        PROV=$1 TYPE=$2 ZONE=$z $0 create > /dev/null 2>&1 || true
        # A create that fails for capacity still shows the instance for a few seconds while it is removed.
        for t in 1 2 3 4 5 6 7 8; do s=$(state $z); [ -z "$s" ] || [ "$s" = RUNNING ] && break; sleep 5; done
        if [ "$(state $z)" = RUNNING ]; then echo $z > $STATE/$VM.zone; echo "got $VM: $spec $z $(date)"; exit 0; fi
      done
    done
    echo "no capacity for $SPECS; retrying in 60 s"; sleep 60
  done ;;
watch)
  # Local babysitter for a job that can be resumed: PUSH=archive.tgz JOB="VAR=... bash tools/x_node.sh" LOG=x.log tools/gcp.sh watch
  # unpacks the archive in ~/$DIR on the node and runs the job there; restarts the node after a preemption and the job with it;
  # prints the lines the remote command SHOW gives every EVERY seconds; ends when the log says ALL DONE (that line is left in
  # out/gcp/$VM.done) and fails when it says FAILED. A line left in the log by an earlier job is not believed: the node keeps the
  # id of the job it was last given (the archive's listing and the JOB), and only a log written under this id counts.
  DIR=${DIR:-pengpt} down=0
  RUN=${RUN:-$(run_id)}
  rm -f $STATE/$VM.done
  while true; do
    ZONE=$(cat $STATE/$VM.zone)
    st=$(state $ZONE)
    if [ "$st" != RUNNING ]; then
      [ "${BORROWED:-0}" != 1 ] || { echo "$VM is ${st:-gone} and is not ours to restart"; exit 1; }
      down=$(( down + 1 ))
      echo "$(date +%H:%M) node is ${st:-gone}; starting it again (try $down)"
      if [ $down -gt 8 ]; then $0 delete 2>&1 | quiet || true; $0 find; down=0; continue; fi
      $G compute instances start $VM --project $PROJECT --zone $ZONE > /dev/null 2>&1 || true
      sleep ${WAIT:-45}; continue
    fi
    out=$(TIMEOUT=60 $0 ssh "cd ~/$DIR 2> /dev/null && { [ \"\$(cat run.id 2> /dev/null)\" = $RUN ] && { grep -E '^(ALL DONE|FAILED)' $LOG | tail -n 1; ${SHOW:-true}; }; [ -f job.pid ] && kill -0 \$(cat job.pid) 2> /dev/null && echo BUSY; }; echo UP" 2> /dev/null | quiet) || true
    echo "$out" | grep -q '^UP$' || { sleep 15; continue; }
    down=0
    echo "$(date +%H:%M)"; echo "$out" | grep -v -E '^(BUSY|UP)$' || true
    if echo "$out" | grep -q '^FAILED'; then echo "the job failed; $VM is left up for a look: NAME=$NAME tools/gcp.sh ssh 'tail ~/$DIR/$LOG'"; exit 1; fi
    if echo "$out" | grep -q '^ALL DONE'; then echo "$out" | grep '^ALL DONE' > $STATE/$VM.done; exit 0; fi
    if ! echo "$out" | grep -q '^BUSY$'; then
      $0 ssh "mkdir -p ~/$DIR" 2>&1 | quiet
      $0 put $PUSH $DIR/push.tgz
      $0 ssh "cd ~/$DIR && tar xzf push.tgz && rm push.tgz && echo $RUN > run.id && ($JOB > $LOG 2>&1 < /dev/null &); sleep 2" 2>&1 | quiet
      echo "$(date +%H:%M) job started on $VM ($ZONE)"
    fi
    sleep ${EVERY:-300}
  done ;;
put)
  d=$STATE/put.$$ && mkdir -p $d && split -b ${PIECE:-100m} $1 $d/part.
  $0 ssh "rm -rf ~/$2.parts && mkdir -p ~/$2.parts"
  ls $d | xargs -P ${STREAMS:-8} -I{} $G compute scp $d/{} $VM:~/$2.parts/{} $A > /dev/null 2>&1
  rm -rf $d
  $0 ssh "cat ~/$2.parts/part.* > ~/$2 && rm -rf ~/$2.parts && [ \"\$(cksum < ~/$2)\" = \"$(cksum < $1 | sed 's/^ *//')\" ]" 2>&1 | quiet ;;
get)
  $G compute scp $VM:~/$1 $2 $A 2>&1 | quiet ;;
ssh)
  run "$*" ;;
gpu)
  run "nvidia-smi dmon -s u -c ${1:-5}; uptime" ;;
list)
  $G compute instances list --project $PROJECT --filter="labels.purpose=pengpt" --format="table(name,zone.basename(),machineType.basename(),status,scheduling.provisioningModel)" ;;
keep)
  touch $STATE/$VM.keep; echo "$VM is kept: delete does nothing until tools/gcp.sh free" ;;
free)
  rm -f $STATE/$VM.keep ;;
delete)
  [ ! -f $STATE/$VM.keep ] || { echo "$VM is kept (tools/gcp.sh free lets it go)"; exit 0; }
  $G compute instances delete $VM --project $PROJECT --zone $ZONE --quiet && rm -f $STATE/$VM.zone ;;
*)
  echo "usage: [NAME=train] [VM=name] [GPU=h100] tools/gcp.sh find|list|ssh cmd|gpu [n]|put local remote|get remote local|keep|free|delete"
  echo "  a job that can be resumed: PUSH=archive.tgz JOB='bash tools/x_node.sh' LOG=x.log tools/gcp.sh watch" ;;
esac
