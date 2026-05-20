#!/usr/bin/env bash
# Unified DLRM-DCNv2 trace + convert pipeline for AMD MI350X / ROCm 7.2.
#
# Captures everything needed for a complete Perfetto view:
#   * --kernel-trace        compute kernel dispatches
#   * --hip-trace           HIP runtime API calls (for iter detection
#                           via hipGraphLaunch + flow arrows)
#   * --memory-copy-trace   HSA-level DMA copies (H2D / D2H / D2D —
#                           NV's "memcpy lane"; not captured by
#                           --kernel-trace alone)
#   * --rccl-trace          RCCL API calls so the converter can label
#                           rccl kernel events by operation type
#                           (allreduce / allgather / alltoall) and emit
#                           a per-window RCCL op-count summary
#
# Then post-processes the raw csvs into:
#   * one annotated Perfetto JSON per GPU (iter_steady_gpu{0..N-1}.json)
#   * one merged 8-GPU view (iter_steady_all8gpus.json)
#
# Usage:
#   trace_and_convert.sh <slurm_jobid> <start_iter> <end_iter> [tag]
#
# Args:
#   slurm_jobid : a long-running slurm reservation jobid to overlap into
#                 (the docker container runs there)
#   start_iter  : first iter to include in the converter window (0-based;
#                 typically 5 or higher to skip warmup)
#   end_iter    : exclusive upper bound of the window (e.g. 10 = iters 5..9)
#   tag         : optional label folded into output dirs / log names
#                 (default: "phase18_iter${start}_${end}")
#
# Examples:
#   # default 5-iter steady-state window (iters 5..9 of a 15-iter run)
#   bash trace_and_convert.sh 4975 5 10
#
#   # longer window for ring-drain analysis
#   bash trace_and_convert.sh 4975 25 30 ring_drain
#
# Outputs (host paths):
#   /home/chcai/sweep_log/trace_${TAG}.log
#   /home/chcai/rps_out/trace_${TAG}/iter_${start}_${end}_{kernel,hip_api,memory_copy,rccl_api}_trace.csv
#   /home/chcai/training_results_v5.1/results/perfetto_${TAG}/iter_steady_gpu{0..7}.json
#   /home/chcai/training_results_v5.1/results/perfetto_${TAG}/iter_steady_all8gpus.json
set -e

JOBID=${1:?"usage: $0 <jobid> <start_iter> <end_iter> [tag]"}
START_ITER=${2:?"start_iter (0-based) required"}
END_ITER=${3:?"end_iter (exclusive) required"}
TAG=${4:-phase18_iter${START_ITER}_${END_ITER}}

N_ITERS=$((END_ITER - START_ITER))
if [ "$N_ITERS" -le 0 ]; then
    echo "ERROR: end_iter ($END_ITER) must be > start_iter ($START_ITER)" >&2
    exit 1
fi
# Need at least END_ITER + 5 iters total so steady-state region is well-defined
# (rocprofv3 init + HCTR warmup typically need ~5 iters before the iter
# boundary detection from hipGraphLaunch stabilizes).
MAX_ITER=$((END_ITER + 5))
if [ "$MAX_ITER" -lt 15 ]; then
    MAX_ITER=15
fi

# Resolve paths
WS_HOST=/home/chcai/training_results_v5.1/AMD/benchmarks/dlrm_dcnv2/implementations/hugectr_rocm_port
SCRIPTS_HOST=/home/chcai/training_results_v5.1/scripts
RPS_HOST=/home/chcai/rps_out
LOG_HOST=/home/chcai/sweep_log
RESULTS_HOST=/home/chcai/training_results_v5.1/results
TRACE_SUBDIR=trace_${TAG}
OUT_SUBDIR=perfetto_${TAG}
IMG=rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424

mkdir -p "$LOG_HOST" "$RPS_HOST"
mkdir -p "$RESULTS_HOST/$OUT_SUBDIR"
# permit non-root container user (the docker image runs as root mapped to
# "nobody" via user namespace) to write into the host-owned results dir
chmod 777 "$RESULTS_HOST/$OUT_SUBDIR"

LOG="$LOG_HOST/trace_${TAG}.log"
mv "$LOG" "${LOG}.prev" 2>/dev/null || true

echo "===== trace_and_convert.sh ====="
echo "  jobid       = $JOBID"
echo "  iter window = ${START_ITER}..${END_ITER} (= $N_ITERS steady iters)"
echo "  MAX_ITER    = $MAX_ITER  (warmup before window + a few after)"
echo "  tag         = $TAG"
echo "  log         = $LOG"
echo "  raw csvs    = $RPS_HOST/$TRACE_SUBDIR/"
echo "  perfetto    = $RESULTS_HOST/$OUT_SUBDIR/"
echo "================================="

srun --jobid="$JOBID" --overlap -N1 -n1 -c 16 bash -c "
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host \
    --tmpfs /ramdata:size=512g,exec,rw \
    -v $WS_HOST:/workspace \
    -v /apps/chcai/criteo_data/mlperf:/nfs_data:ro \
    -v $SCRIPTS_HOST:/scripts:ro \
    -v $RPS_HOST:/rps_out \
    -v $RESULTS_HOST:/results \
    -e HCTR_ROCTX='${HCTR_ROCTX:-1}' \
    -e HCTR_STAGE_TRAIN_GB='${HCTR_STAGE_TRAIN_GB:-}' \
    -e HCTR_STAGE_VAL_GB='${HCTR_STAGE_VAL_GB:-}' \
    -w /workspace $IMG \
    bash /workspace/scripts/_trace_and_convert_inner.sh \
        '$START_ITER' '$END_ITER' '$MAX_ITER' '$TRACE_SUBDIR' '$OUT_SUBDIR'
" > "$LOG" 2>&1
EXIT=$?

echo
echo "===== trace_and_convert.sh DONE (exit=$EXIT) ====="
echo "--- last 30 lines of $LOG ---"
tail -30 "$LOG"
exit $EXIT
