#!/usr/bin/env bash
# Parallel Phase-19 A/B test:
#   1) Build patched HCTR once on $BUILD_JID node (~1-2 min, NFS shared so the
#      result is visible from all reserved nodes)
#   2) In parallel: baseline run on $BASE_JID, phase19 run on $P19_JID
#   3) Print summary
#
# Usage:
#   bash run_phase19_ab_parallel.sh <build_jid> <baseline_jid> <phase19_jid>
#
# Example:
#   bash run_phase19_ab_parallel.sh 4652 4652 4975
set -e
BUILD_JID=${1:?build jobid required}
BASE_JID=${2:?baseline jobid required}
P19_JID=${3:?phase19 jobid required}
LOG_DIR=/home/chcai/sweep_log
mkdir -p "$LOG_DIR"

IMG=rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424
WORKSPACE_MNT="-v /home/chcai/training_results_v5.1/AMD/benchmarks/dlrm_dcnv2/implementations/hugectr_rocm_port:/workspace"
DATA_MNT="-v /apps/chcai/criteo_data/mlperf:/nfs_data:ro"
SCRIPTS_MNT="-v /home/chcai/training_results_v5.1/scripts:/scripts:ro"

DOCKER_BASE="docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host --tmpfs /ramdata:size=512g,exec,rw $WORKSPACE_MNT $DATA_MNT $SCRIPTS_MNT -w /workspace $IMG"

echo "===== STEP 1: BUILD on jid=$BUILD_JID ====="
BUILD_LOG=$LOG_DIR/phase19_build.log
srun --jobid="$BUILD_JID" --overlap -N1 -n1 -c 16 bash -c "
$DOCKER_BASE bash /scripts/_phase19_build_inner.sh
" > "$BUILD_LOG" 2>&1
echo "[ok] build done; tail:"
tail -8 "$BUILD_LOG"

echo
echo "===== STEP 2: PARALLEL RUNS  baseline=$BASE_JID  phase19=$P19_JID ====="
BASE_LOG=$LOG_DIR/phase19_baseline.log
P19_LOG=$LOG_DIR/phase19_phase19.log

run_one() {
    local jid=$1
    local label=$2
    local mcpy=$3
    local logf=$4
    srun --jobid="$jid" --overlap -N1 -n1 -c 16 bash -c "
$DOCKER_BASE bash -c '
echo \"=== stage data into /ramdata (tmpfs) ===\"
mkdir -p /ramdata/mlperf
T0=\$(date +%s)
cp /nfs_data/train_data.bin /ramdata/mlperf/ &
P1=\$!
cp /nfs_data/val_data.bin /ramdata/mlperf/ &
P2=\$!
wait \$P1 \$P2
T1=\$(date +%s)
echo \"=== staging done in \$((T1-T0))s ===\"
ln -sf /ramdata /criteo
bash /scripts/_phase19_run_inner.sh $label $mcpy 250
'
" > "$logf" 2>&1
}

run_one "$BASE_JID" baseline 0 "$BASE_LOG" &
PID_BASE=$!
run_one "$P19_JID"  phase19  1 "$P19_LOG"  &
PID_P19=$!
echo "PID_BASE=$PID_BASE  PID_P19=$PID_P19"
echo "tail -f $BASE_LOG  /  tail -f $P19_LOG"
wait $PID_BASE
EXIT_B=$?
wait $PID_P19
EXIT_P=$?

echo
echo "===== STEP 3: SUMMARY ====="
echo "baseline exit=$EXIT_B  phase19 exit=$EXIT_P"
for label in baseline phase19; do
    f=$LOG_DIR/phase19_${label}.log
    sps=$(grep -aE "Steady-state throughput" "$f" | tail -1 | awk '{print $(NF-2)}')
    pi=$(grep -aE "Per-iter time" "$f" | tail -1 | awk '{print $(NF-1)}')
    loss=$(grep -aE "Final loss" "$f" | tail -1 | awk '{print $NF}')
    echo "[$label]  sps=${sps:-?} M  per-iter=${pi:-?} ms  final_loss=${loss:-?}"
done
