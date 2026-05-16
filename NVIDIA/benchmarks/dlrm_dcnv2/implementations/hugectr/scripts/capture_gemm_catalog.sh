#!/bin/bash
# Capture HCTR's per-iter GEMM call catalog by running a brief training
# session with cublaslt_gemm_logger_shim.so LD_PRELOADed. Every cublasLtMatmul
# call made during model.compile() + graph capture (the first iter) gets
# logged to <LOGDIR>/gemm_init_log.jsonl. Run build_gemm_catalog.py on the
# resulting JSONL to produce gemm_catalog.json, then pass that to
# nsys_to_perfetto_annotated.py via --gemm-catalog=<path> for full
# PyTorch-profiler-style M/N/K/dtype/op/epilogue/dlrm_layer annotation
# on every GEMM kernel in subsequent Perfetto JSON exports.
#
# This is a one-time setup; the catalog is deterministic for a given
# (model + batch size + dtype config) so it can be reused across many
# trace renders. ~5 min to capture.
#
# Required environment for the underlying srun + docker pipeline:
#   RESV         Slurm reservation owning the target B200 node (required)
#   NODE         Slurm node suffix, e.g. "1" for hungry-hippo-fin-03-1
#   IMAGE        docker image tag (default: mlperf-nvidia:recommendation-hugectr)
#   PERSIST_SRC  bind-mount source for the Criteo prefix (200 GB train + 18 GB val);
#                default /mnt/local_disk/home/chcai/criteo_full
#   LOGDIR       writable dir for run log + gemm_init_log.jsonl;
#                default /home/chcai/hctr_runs/results
#   CONFIG       HCTR config to source for hyper-params;
#                default config_b200_1x8_rr_bs1x_auto.sh
#
# Usage example:
#   RESV=gh-chcai-XXXX bash scripts/capture_gemm_catalog.sh 1
#
# Side note: the same shim object also intercepts cublasLtMatmul during
# steady-state replay, but since HCTR uses captured CUDA graphs all
# steady-state matmul calls bypass the host wrapper and go straight to
# cudaGraphLaunch -- so the log only fills up during init / first iter.
# Resulting JSONL is small (~300 lines, ~96 KB on our DLRM-DCNv2 model).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # ...hugectr/
LOGDIR="${LOGDIR:-/home/chcai/hctr_runs/results}"
IMAGE="${IMAGE:-mlperf-nvidia:recommendation-hugectr}"
PERSIST_SRC="${PERSIST_SRC:-/mnt/local_disk/home/chcai/criteo_full}"
CONFIG="${CONFIG:-config_b200_1x8_rr_bs1x_auto.sh}"
NODE="${1:?usage: $0 <node-suffix>  (must set RESV env var too)}"
RESV="${RESV:?must set RESV env var to a Slurm reservation owning the GPU node}"
NODE_HOSTNAME="${NODE_HOSTNAME_PATTERN:-hungry-hippo-fin-03}-${NODE}"

if [ ! -f "$SCRIPT_DIR/$CONFIG" ]; then
    echo "ERROR: config not found at $SCRIPT_DIR/$CONFIG" >&2
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/scripts/cublaslt_gemm_logger_shim.so" ]; then
    echo "ERROR: shim not built. Compile it first:" >&2
    echo "  docker run --rm -v $SCRIPT_DIR/scripts:/work \\" >&2
    echo "      $IMAGE bash -c 'cd /work && g++ -O2 -fPIC -shared \\" >&2
    echo "      -o cublaslt_gemm_logger_shim.so \\" >&2
    echo "      cublaslt_gemm_logger_shim.cpp -ldl -lpthread'" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$SCRIPT_DIR/$CONFIG"
DATESTAMP=$(date +%y%m%d%H%M%S)
LOG="$LOGDIR/gemm_catalog_capture_${NODE}_${DATESTAMP}.log"

mkdir -p "$LOGDIR"
rm -f "$LOGDIR/gemm_init_log.jsonl"   # shim appends; start clean

echo "=========================================="
echo "Capture GEMM catalog (cublaslt_gemm_logger_shim active)"
echo "Node       : $NODE_HOSTNAME"
echo "Reservation: $RESV"
echo "Image      : $IMAGE"
echo "Config     : $CONFIG  (MAX_ITER overridden to 500)"
echo "Catalog log: $LOGDIR/gemm_init_log.jsonl"
echo "Run log    : $LOG"
echo "=========================================="

env HOME="$HOME" SLURM_TIMELIMIT=00:15:00 srun \
  --reservation="$RESV" \
  --nodelist="$NODE_HOSTNAME" \
  --gres=gpu:8 --chdir="$HOME" \
  --overlap bash -c "
docker run --rm \\
    --runtime=nvidia --gpus all \\
    --network=host --ipc=host \\
    --ulimit memlock=-1 --ulimit stack=67108864 \\
    --shm-size=64g \\
    --tmpfs /ramdata:size=250g \\
    --cap-add=IPC_LOCK --cap-add=SYS_NICE \\
    --device=/dev/infiniband \\
    -v $SCRIPT_DIR:/workspace/dlrm \\
    -v $PERSIST_SRC:/persist:ro \\
    -v $LOGDIR:/results \\
    -w /workspace/dlrm \\
    -e BATCHSIZE=$BATCHSIZE -e BATCHSIZE_EVAL=$BATCHSIZE_EVAL \\
    -e LEARNING_RATE=$LEARNING_RATE -e USE_MIXED_PRECISION=$USE_MIXED_PRECISION \\
    -e SCALER=$SCALER -e SHARDING_PLAN=$SHARDING_PLAN \\
    -e MEM_COMM_BW_RATIO=$MEM_COMM_BW_RATIO -e GEN_LOSS_SUMMARY=$GEN_LOSS_SUMMARY \\
    -e MINIMUM_TRAINING_TIME=$MINIMUM_TRAINING_TIME \\
    -e DP_SHARDING_THRESHOLD=$DP_SHARDING_THRESHOLD \\
    -e USE_ALGORITHM_SEARCH=$USE_ALGORITHM_SEARCH \\
    -e MAX_ITER=500 -e DISPLAY_INTERVAL=100 \\
    -e EVAL_INTERVAL=2000000 \\
    -e CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS \\
    -e HCTR_DEFAULT_CONCURRENCY=$HCTR_DEFAULT_CONCURRENCY \\
    -e DGXNGPU=$DGXNGPU -e DGXNNODES=$DGXNNODES -e DGXSYSTEM=$DGXSYSTEM \\
    -e RUN_SCRIPT=$RUN_SCRIPT \\
    -e TRAIN_DATA=/ramdata/train_data.bin \\
    -e VAL_DATA=/ramdata/val_data.bin \\
    -e LD_PRELOAD=/workspace/dlrm/scripts/cublaslt_gemm_logger_shim.so \\
    -e SHIM_GEMM_LOG=/results/gemm_init_log.jsonl \\
    $IMAGE \\
    bash -c '
echo \"=== copy data ===\"
time cp /persist/train_data.bin /ramdata/train_data.bin
time cp /persist/val_data.bin   /ramdata/val_data.bin
echo \"=== run training briefly to populate gemm_init_log.jsonl ===\"
mpirun -n 1 --allow-run-as-root bash run_and_time.sh
echo \"=== summary ===\"
wc -l /results/gemm_init_log.jsonl
'
" 2>&1 | tee "$LOG"

echo "=========================================="
echo "Done."
echo "Catalog log: $LOGDIR/gemm_init_log.jsonl"
echo
echo "Next:"
echo "  python3 $SCRIPT_DIR/scripts/build_gemm_catalog.py \\"
echo "      $LOGDIR/gemm_init_log.jsonl $LOGDIR/gemm_catalog.json"
echo
echo "Then pass --gemm-catalog=$LOGDIR/gemm_catalog.json to"
echo "nsys_to_perfetto_annotated.py for full per-kernel shape annotation."
