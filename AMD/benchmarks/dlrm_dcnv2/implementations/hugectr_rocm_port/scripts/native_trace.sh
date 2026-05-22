#!/usr/bin/env bash
# Phase 20.10 (Path D, 2026-05-20): outer wrapper for the low-overhead
# HCTR native tracer. Same interface as trace_and_convert.sh but calls
# _native_trace_inner.sh (HCTR_NATIVE_TRACE=1, NO rocprofv3).
#
# Pairs with native_overlay_all_ranks.sh -- run that after this to merge
# the rocprofv3 cold-pass kernel metadata onto the native trace.
#
# Usage:
#   native_trace.sh <slurm_jobid> <begin_iter> <end_iter> [tag]
#
# Examples:
#   bash scripts/native_trace.sh 7259 5 10 5iter_native
#
# Env overrides honored:
#   HCTR_NATIVE_TRACE_DETAIL  0|1|2 (default 2; events/iter and overhead)
#   HCTR_STAGE_TRAIN_GB / HCTR_STAGE_VAL_GB  partial-stage knobs
#   HCTR_USE_CUDA_GRAPH  (default 1)
#   HCTR_DEDICATED_RCCL_STREAM  (default 0 for tracing; 1 in production)
#   EXHAUSTIVE=1  one-shot deep capture (DEEP=1, DETAIL=2, GRAPH_NODES=1,
#                 GRAPH_CLOCK=1, CUDA_GRAPH off). Use to harvest a
#                 phase->kernel map for later light traces. NOT a perf
#                 measurement -- iter wall inflates ~30-40%. Output dir is
#                 suffixed _exhaustive; the saved phase_kernel_map.json
#                 inside is the artifact to reuse via --load-map.
set -e

JOBID=${1:?"usage: $0 <jobid> <begin_iter> <end_iter> [tag]"}
BEGIN_ITER=${2:?"begin_iter (0-based, where to start tracing) required"}
END_ITER=${3:?"end_iter (exclusive) required"}
TAG=${4:-native_iter${BEGIN_ITER}_${END_ITER}}

# EXHAUSTIVE=1: maximum-detail capture. Forces DEEP=1 (no hipGraph capture
# -> per-phase ROCTX ranges visible on the network), DETAIL=2 (every
# event), GRAPH_NODES=1 (per-graph-node events for any remaining
# hipGraph), GRAPH_CLOCK=1 (in-graph clock-writer kernels for per-kernel
# timing). Tag is suffixed _exhaustive so the dir is distinguishable.
EXHAUSTIVE=${EXHAUSTIVE:-0}
if [ "$EXHAUSTIVE" = "1" ]; then
    case "$TAG" in
        *exhaustive*) ;;  # already labelled
        *) TAG="${TAG}_exhaustive" ;;
    esac
    export HCTR_NATIVE_TRACE_DEEP=1
    export HCTR_NATIVE_TRACE_DETAIL=2
    export HCTR_NATIVE_TRACE_GRAPH_NODES=1
    export HCTR_NATIVE_TRACE_GRAPH_CLOCK=1
    # DEEP=1 forces use_graph=false in pipeline.cpp; mirror it on the env
    # so any downstream check sees the same state.
    export HCTR_USE_CUDA_GRAPH=0
    echo "##########################################################"
    echo "# EXHAUSTIVE TRACE MODE -- not a perf measurement.       #"
    echo "#   DEEP=1, DETAIL=2, GRAPH_NODES=1, GRAPH_CLOCK=1.      #"
    echo "#   hipGraph disabled. Expect ~30-40% iter-wall inflation.#"
    echo "#   Output dir tagged _exhaustive. Reuse the resulting   #"
    echo "#   phase_kernel_map.json on light traces via --load-map.#"
    echo "##########################################################"
fi

N_ITERS=$((END_ITER - BEGIN_ITER))
if [ "$N_ITERS" -le 0 ]; then
    echo "ERROR: end_iter ($END_ITER) must be > begin_iter ($BEGIN_ITER)" >&2
    exit 1
fi

WS_HOST=/home/chcai/training_results_v5.1/AMD/benchmarks/dlrm_dcnv2/implementations/hugectr_rocm_port
RPS_HOST=/home/chcai/rps_out
LOG_HOST=/home/chcai/sweep_log
OUTDIR_HOST=$RPS_HOST/native_${TAG}
IMG=rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424

mkdir -p "$LOG_HOST" "$RPS_HOST"
rm -rf "$OUTDIR_HOST"
mkdir -p "$OUTDIR_HOST"
chmod 777 "$OUTDIR_HOST"

LOG="$LOG_HOST/native_${TAG}.log"
mv "$LOG" "${LOG}.prev" 2>/dev/null || true

echo "===== native_trace.sh ====="
echo "  jobid       = $JOBID"
echo "  trace iters = ${BEGIN_ITER}..${END_ITER} (= $N_ITERS iters)"
echo "  detail      = ${HCTR_NATIVE_TRACE_DETAIL:-2}"
echo "  tag         = $TAG"
echo "  log         = $LOG"
echo "  output dir  = $OUTDIR_HOST"
echo "  stage knobs = train=${HCTR_STAGE_TRAIN_GB:-full} val=${HCTR_STAGE_VAL_GB:-full}"
echo "============================"

srun --jobid="$JOBID" --overlap -N1 -n1 -c 16 bash -c "
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host \
    --tmpfs /ramdata:size=512g,exec,rw \
    -v $WS_HOST:/workspace \
    -v /apps/chcai/criteo_data/mlperf:/nfs_data:ro \
    -v $OUTDIR_HOST:/rps_out \
    -e HCTR_NATIVE_TRACE_MODE=1 \
    -e HCTR_NATIVE_TRACE_BEGIN='$BEGIN_ITER' \
    -e HCTR_NATIVE_TRACE_END='$END_ITER' \
    -e HCTR_NATIVE_TRACE_DETAIL='${HCTR_NATIVE_TRACE_DETAIL:-2}' \
    -e HCTR_NATIVE_TRACE_DEEP='${HCTR_NATIVE_TRACE_DEEP:-0}' \
    -e HCTR_NATIVE_TRACE_GRAPH_NODES='${HCTR_NATIVE_TRACE_GRAPH_NODES:-0}' \
    -e HCTR_NATIVE_TRACE_GRAPH_CLOCK='${HCTR_NATIVE_TRACE_GRAPH_CLOCK:-0}' \
    -e HCTR_USE_CUDA_GRAPH='${HCTR_USE_CUDA_GRAPH:-1}' \
    -e HCTR_DEDICATED_RCCL_STREAM='${HCTR_DEDICATED_RCCL_STREAM:-0}' \
    -e HCTR_STAGE_TRAIN_GB='${HCTR_STAGE_TRAIN_GB:-}' \
    -e HCTR_STAGE_VAL_GB='${HCTR_STAGE_VAL_GB:-}' \
    -e HCTR_HIPBLASLT_FUSED_EPILOGUES_FPROP='${HCTR_HIPBLASLT_FUSED_EPILOGUES_FPROP:-0}' \
    -w /workspace $IMG \
    bash /workspace/scripts/_native_trace_inner.sh
" > "$LOG" 2>&1
EXIT=$?

echo
echo "===== native_trace.sh DONE (exit=$EXIT) ====="
# _native_trace_inner.sh writes to /rps_out/native_trace_test; remap to
# our per-tag dir.
INNER_OUT="$OUTDIR_HOST/native_trace_test"
if [ -d "$INNER_OUT" ]; then
    mv "$INNER_OUT"/* "$OUTDIR_HOST"/ 2>/dev/null || true
    rmdir "$INNER_OUT" 2>/dev/null || true
fi
echo "--- last 30 lines of $LOG ---"
tail -30 "$LOG"
echo
echo "--- per-rank JSONs in $OUTDIR_HOST ---"
ls -lh "$OUTDIR_HOST"/hctr_native_trace_*.json 2>/dev/null
echo
echo "Next step: overlay rocprofv3 metadata + auto-merge into one 8-GPU trace"
echo "(the overlay script now calls merge_perfetto_traces.py and drops the"
echo " first and last iter to avoid hipEvent staleness / end-of-window):"
echo "  bash scripts/native_overlay_all_ranks.sh \\"
echo "      $OUTDIR_HOST \\"
echo "      /home/chcai/rps_out/trace_cold_v2/iter5_10 \\"
echo "      $BEGIN_ITER"
echo "Output: $OUTDIR_HOST/overlay_all8gpus_trimmed.json"
if [ "$EXHAUSTIVE" = "1" ]; then
    echo
    echo "EXHAUSTIVE mode hint: after overlay the saved map lives at"
    echo "  $OUTDIR_HOST/phase_kernel_map.json"
    echo "Reuse it on subsequent light traces by setting LOAD_MAP, e.g.:"
    echo "  LOAD_MAP=$OUTDIR_HOST/phase_kernel_map.json \\"
    echo "    bash scripts/native_overlay_all_ranks.sh <light_dir> - $BEGIN_ITER"
fi
exit $EXIT
