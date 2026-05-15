#!/bin/bash
# Apple-to-apple HCTR run matching NV B200 submission setup:
#   - docker --tmpfs /ramdata:size=512g (in-container RAM disk)
#   - Stage 402 GB train + 81 GB val from NFS into /ramdata at startup
#   - Run HCTR pointing at /ramdata
#
# Usage:
#   bash run_apple_to_apple.sh <jobid> [trace_steps]
#     trace_steps = 0 (no trace, full perf run, default)
#                 = N (capture rocprofv3 trace, run N+15 iters total)
#
# Example:
#   bash run_apple_to_apple.sh 4653            # bs=1x clean perf run
#   bash run_apple_to_apple.sh 4653 5          # bs=1x, capture 5-step trace

set -e
JOBID=${1:?"jobid required"}
TRACE_STEPS=${2:-0}

# Default to bs=1x for apple-to-apple comparison with NV.
HCTR_BATCH=${HCTR_BATCH:-55296}
HCTR_MAX_ITER=${HCTR_MAX_ITER:-2000}

if [ "$TRACE_STEPS" -gt 0 ]; then
    HCTR_MAX_ITER=$((TRACE_STEPS + 15))  # 15 warmup + N measured
    HCTR_DISPLAY=5
    PROFILE_PREFIX="rocprofv3 --kernel-trace --hip-trace -d /trace_out --output-format csv --output-file a2a_bs${HCTR_BATCH}_${TRACE_STEPS}step --"
    TRACE_DIR=/home/chcai/trace_a2a_bs${HCTR_BATCH}_${TRACE_STEPS}step
    mkdir -p "$TRACE_DIR" && chmod 777 "$TRACE_DIR"
    TRACE_MOUNT="-v $TRACE_DIR:/trace_out"
else
    HCTR_DISPLAY=200
    PROFILE_PREFIX=""
    TRACE_MOUNT=""
fi

LOG=/home/chcai/sweep_log/a2a_bs${HCTR_BATCH}_$( [ "$TRACE_STEPS" -gt 0 ] && echo "trace${TRACE_STEPS}" || echo "perf" ).log
mkdir -p /home/chcai/sweep_log
echo "===== APPLE-TO-APPLE RUN ====="
echo "  jobid=$JOBID  bs=$HCTR_BATCH  iters=$HCTR_MAX_ITER  trace_steps=$TRACE_STEPS"
echo "  log: $LOG"
echo "  tmpfs: --tmpfs /ramdata:size=512g (matches NV --tmpfs /ramdata:size=250g pattern)"
echo "============================================="

srun --jobid="$JOBID" --overlap -N1 -n1 -c 16 bash -c "
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host \
    --tmpfs /ramdata:size=512g,exec,rw \
    -v /home/chcai/training_results_v5.1/AMD/benchmarks/dlrm_dcnv2/implementations/hugectr_rocm_port:/workspace \
    -v /apps/chcai/criteo_data/mlperf:/nfs_data:ro \
    \${HCTR_EXTRA_DOCKER_ARGS:-} \
    $TRACE_MOUNT \
    -e HCTR_NGPU=8 -e HCTR_BATCH=$HCTR_BATCH -e HCTR_EVAL_BATCH=131072 -e HCTR_EV_SIZE=128 \
    -e HCTR_LR=0.004 -e HCTR_MAX_ITER=$HCTR_MAX_ITER -e HCTR_DISPLAY=$HCTR_DISPLAY -e HCTR_MEM_CAP=200 \
    -e HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348' \
    -e HCTR_SHARDING_PLAN=auto -e OMP_NUM_THREADS=8 -e HCTR_USE_MULTI_HOT=1 \
    -e HCTR_USE_MLPERF_CRITEO=1 -e HCTR_USE_CUDA_GRAPH=1 \
    -e HCTR_DP_SHARD_THRESH=0.008 -e HCTR_MEM_COMM_WORK_RATIO=9 -e HCTR_READER_THREADS=1 \
    -e HCTR_ASYNC_WGRAD=0 \
    -e HCTR_PROFILE_PREFIX=\"$PROFILE_PREFIX\" \
    -w /workspace rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424 \
    bash -c '
echo \"=== stage data into /ramdata (tmpfs, ~3 min for 483 GB) ===\"
mkdir -p /ramdata/mlperf
T0=\$(date +%s)
cp /nfs_data/train_data.bin /ramdata/mlperf/ &
P1=\$!
cp /nfs_data/val_data.bin /ramdata/mlperf/ &
P2=\$!
wait \$P1 \$P2
T1=\$(date +%s)
echo \"=== staging done in \$((T1-T0))s ===\"
ls -lh /ramdata/mlperf/
df -h /ramdata
echo \"=== now run HCTR ===\"
ln -sf /ramdata /criteo
bash /workspace/run_b200_match.sh
'
" > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "Tail log: tail -f $LOG"
wait $PID
EXIT=$?
echo
echo "===== EXIT: $EXIT ====="
grep -aE "Steady|Final loss|Loss range|Per-iter|staging done|^total" "$LOG" | tail -10
exit $EXIT
