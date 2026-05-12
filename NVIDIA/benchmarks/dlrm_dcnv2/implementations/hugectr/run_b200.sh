#!/bin/bash
# Single-node srun + docker launcher for the MLPerf HugeCTR DLRM-DCNv2 benchmark
# on our B200 cluster (hungry-hippo-fin-03-*). Bypasses Pyxis/Enroot and uses
# the existing srun + docker run pattern from launch_srun_nvidia.sh.
#
# Usage:
#   bash run_b200.sh \
#       --reservation gh-chcai-7e6de3a5 \
#       --nodelist hungry-hippo-fin-03-3 \
#       --config config_b200_1x8.sh \
#       --train-data /home/chcai/criteo_processed/train_data.bin \
#       --val-data   /home/chcai/criteo_processed/val_data.bin \
#       --image      mlperf-nvidia:recommendation-hugectr \
#       --logdir     /home/chcai/criteo_synth/results
#
# If --image-tar is provided and the image is not present locally on the node,
# it is loaded from disk first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RESERVATION=""
NODELIST=""
CONFIG="config_b200_1x8.sh"
IMAGE="mlperf-nvidia:recommendation-hugectr"
IMAGE_TAR=""
TRAIN_DATA=""
VAL_DATA=""
LOGDIR=""
TIMELIMIT="01:00:00"
NSYS_TRACE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --reservation)  RESERVATION="$2"; shift 2;;
        --nodelist)     NODELIST="$2"; shift 2;;
        --config)       CONFIG="$2"; shift 2;;
        --image)        IMAGE="$2"; shift 2;;
        --image-tar)    IMAGE_TAR="$2"; shift 2;;
        --train-data)   TRAIN_DATA="$2"; shift 2;;
        --val-data)     VAL_DATA="$2"; shift 2;;
        --logdir)       LOGDIR="$2"; shift 2;;
        --time)         TIMELIMIT="$2"; shift 2;;
        --nsys-trace)   NSYS_TRACE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

: "${RESERVATION:?--reservation required}"
: "${NODELIST:?--nodelist required}"
: "${TRAIN_DATA:?--train-data required}"
: "${VAL_DATA:?--val-data required}"
: "${LOGDIR:?--logdir required}"

mkdir -p "$LOGDIR"
DATESTAMP=$(date +%y%m%d%H%M%S)
LOGFILE="$LOGDIR/run_${DATESTAMP}.log"

CONFIG_PATH="$SCRIPT_DIR/$CONFIG"
[ -f "$CONFIG_PATH" ] || { echo "Config not found: $CONFIG_PATH"; exit 1; }

# Source the config in this shell to get DL hyperparameters; we'll forward them
# as docker -e env vars below. (config also defines DGXNGPU/DGXNNODES/etc.)
# shellcheck source=/dev/null
source "$CONFIG_PATH"

TRAIN_DATA_DIR="$(dirname "$TRAIN_DATA")"
TRAIN_DATA_BASE="$(basename "$TRAIN_DATA")"
VAL_DATA_DIR="$(dirname "$VAL_DATA")"
VAL_DATA_BASE="$(basename "$VAL_DATA")"

echo "=========================================="
echo "Reservation : $RESERVATION"
echo "Nodelist    : $NODELIST"
echo "Image       : $IMAGE"
[ -n "$IMAGE_TAR" ] && echo "Image tar   : $IMAGE_TAR"
echo "Config      : $CONFIG ($DGXNNODES nodes x $DGXNGPU GPUs, batch $BATCHSIZE)"
echo "Train data  : $TRAIN_DATA"
echo "Val data    : $VAL_DATA"
echo "Logfile     : $LOGFILE"
echo "Max iter    : ${MAX_ITER:-?}"
echo "Display     : ${DISPLAY_INTERVAL:-?}"
echo "Eval interval: ${EVAL_INTERVAL:-?}"
echo "=========================================="

env HOME=/home/chcai SLURM_TIMELIMIT="$TIMELIMIT" srun \
    --reservation="$RESERVATION" \
    --nodelist="$NODELIST" \
    --chdir=/tmp \
    bash -c "
set -e
# Load the image if a local docker daemon doesn't already have it.
if [ -n '${IMAGE_TAR}' ] && ! docker image inspect '${IMAGE}' >/dev/null 2>&1; then
    echo 'Loading image from ${IMAGE_TAR}...'
    if [[ '${IMAGE_TAR}' == *.zst ]]; then
        zstd -dc '${IMAGE_TAR}' | docker load
    else
        docker load -i '${IMAGE_TAR}'
    fi
fi

docker run --rm \
    --runtime=nvidia --gpus all \
    --network=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=64g \
    --cap-add=IPC_LOCK --cap-add=SYS_NICE \
    --device=/dev/infiniband \
    -v $SCRIPT_DIR:/workspace/dlrm \
    -v $TRAIN_DATA_DIR:/data:ro \
    -v $VAL_DATA_DIR:/data_val:ro \
    -v $LOGDIR:/results \
    -w /workspace/dlrm \
    -e BATCHSIZE -e BATCHSIZE_EVAL -e LEARNING_RATE \
    -e USE_MIXED_PRECISION -e SCALER -e SHARDING_PLAN \
    -e MEM_COMM_BW_RATIO -e GEN_LOSS_SUMMARY \
    -e DP_SHARDING_THRESHOLD -e MAX_ITER \
    -e DISPLAY_INTERVAL -e EVAL_INTERVAL \
    -e MINIMUM_TRAINING_TIME \
    -e DGXNGPU -e DGXNNODES -e DGXSYSTEM \
    -e RUN_SCRIPT \
    -e TRAIN_DATA=/data/$TRAIN_DATA_BASE \
    -e VAL_DATA=/data_val/$VAL_DATA_BASE \
    -e NSYS_TRACE='${NSYS_TRACE}' \
    -e NSYS_DELAY='${NSYS_DELAY:-30}' \
    -e NSYS_DURATION='${NSYS_DURATION:-5}' \
    -e USE_ALGORITHM_SEARCH \
    -e DLRM_BIND \
    -e NCCL_ALGO -e NCCL_PROTO -e NCCL_IB_DISABLE \
    -e NCCL_IB_HCA -e NCCL_IB_GID_INDEX -e NCCL_NET_GDR_LEVEL \
    -e NCCL_DEBUG -e NCCL_DEBUG_SUBSYS \
    -e NCCL_BUFFSIZE -e NCCL_MIN_NCHANNELS -e NCCL_MAX_NCHANNELS \
    -e NCCL_NCHANNELS_PER_NET_PEER -e CUDA_DEVICE_MAX_CONNECTIONS \
    -e NCCL_LAUNCH_MODE \
    -e NCCL_P2P_NET_CHUNKSIZE -e NCCL_NVLS_NCHANNELS \
    -e NCCL_MAX_P2P_NTHREADS -e NCCL_LL_THRESHOLD -e NCCL_LL128_BUFFSIZE \
    -e NCCL_LL128_ENABLE -e NCCL_NVLS_THRESHOLD \
    -e NCCL_RUNTIME_CONNECT \
    -e NCCL_GRAPH_MIXING_SUPPORT -e NCCL_CUMEM_ENABLE \
    -e NCCL_CHECKS_DISABLE \
    -e HCTR_DEFAULT_CONCURRENCY -e DENSE_UNIQUE_RATIO -e WGRAD_UNIQUE_RATIO \
    -e HCTR_RMM_SETTABLE \
    '${IMAGE}' \
    bash -c '
        if [ -n \"\$NSYS_TRACE\" ]; then
            echo \"NSYS tracing -> /results/\${NSYS_TRACE}.nsys-rep (delay \${NSYS_DELAY:-30}s, duration \${NSYS_DURATION:-5}s)\"
            export DLRM_BIND=\"nsys profile -t cuda,nvtx,osrt,cudnn,cublas --cuda-graph-trace=node --delay=\${NSYS_DELAY:-30} --duration=\${NSYS_DURATION:-5} --output=/results/\${NSYS_TRACE} --force-overwrite=true\"
        fi
        mpirun -n 1 --allow-run-as-root bash run_and_time.sh
    ' 2>&1
" 2>&1 | tee "$LOGFILE"

echo "=========================================="
echo "Done. Log: $LOGFILE"
