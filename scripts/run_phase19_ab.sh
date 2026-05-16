#!/usr/bin/env bash
# Phase-19 A/B smoke test wrapper. Spins up a docker container on a slurm
# allocation and runs _phase19_ab_inner.sh which builds + runs baseline +
# Phase-19 (HCTR_USE_MEMCPY_STREAM=1) back-to-back.
#
# Usage:
#   bash run_phase19_ab.sh <jobid>   # uses /apps/chcai/criteo_data/mlperf
set -e
JOBID=${1:?"jobid required"}
LOG=/home/chcai/sweep_log/phase19_ab.log
mkdir -p /home/chcai/sweep_log

echo "===== PHASE-19 A/B ====="
echo "  jobid=$JOBID"
echo "  log:  $LOG"

srun --jobid="$JOBID" --overlap -N1 -n1 -c 16 bash -c "
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host \
    --tmpfs /ramdata:size=512g,exec,rw \
    -v /home/chcai/training_results_v5.1/AMD/benchmarks/dlrm_dcnv2/implementations/hugectr_rocm_port:/workspace \
    -v /apps/chcai/criteo_data/mlperf:/nfs_data:ro \
    -v /home/chcai/training_results_v5.1/scripts:/scripts:ro \
    -w /workspace rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424 \
    bash -c '
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
bash /scripts/_phase19_ab_inner.sh
'
" > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID  tail -f $LOG"
wait $PID
EXIT=$?
echo
echo "===== EXIT: $EXIT ====="
grep -aE "Phase-19 A/B SUMMARY|sps=" "$LOG" | tail -10
exit $EXIT
