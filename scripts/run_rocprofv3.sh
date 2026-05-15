#!/bin/bash
# rocprofv3 kernel-trace profiling for the NaN reproducer.
#
# Usage:
#   bash run_rocprofv3.sh [STEPS] [PRECOND_FREQ] [GPU] [IMAGE]
#   bash run_rocprofv3.sh                      # 5 steps, precond every 2, GPU 0
#   bash run_rocprofv3.sh 200                  # 200 steps, precond every 2
#   bash run_rocprofv3.sh 5000 200             # 5000 steps, precond every 200
#   bash run_rocprofv3.sh 5000 200 3           # same, on GPU 3
#
# Precision knobs (same as run_docker.sh / run_nan_test.sh):
#   MODEL_DTYPE=bfloat16 bash run_rocprofv3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME="${SCRIPT_NAME:-repro3_precision_test.py}"
OUTPUT_DIR="${SCRIPT_DIR}/experiments"
RUN_TAG="rocprofv3_$(date +%Y-%m-%d_%H-%M-%S)"

NUM_STEPS=${1:-5}
PRECONDITION_FREQUENCY=${2:-2}
GPU=${3:-0}
DOCKER_IMAGE=${4:-rocm/pytorch-private:nan-repro}

HOST_OUTPUT="${OUTPUT_DIR}/${RUN_TAG}"
mkdir -p "$HOST_OUTPUT"
chmod 777 "$HOST_OUTPUT"

PRECISION_ENVS=()
for var in MODEL_DTYPE SHAMPOO_PRECONDITIONER_DTYPE SHAMPOO_COMMUNICATION_DTYPE RMSNORM_DTYPE AMP_DTYPE DISABLE_TF32; do
    if [ -n "${!var:-}" ]; then
        PRECISION_ENVS+=(-e "${var}=${!var}")
        echo "Precision: ${var}=${!var}"
    fi
done

echo "=== rocprofv3 profiling run ==="
echo "Steps: $NUM_STEPS | Precondition freq: $PRECONDITION_FREQUENCY | GPU: $GPU | Image: $DOCKER_IMAGE"
echo "Script: $SCRIPT_NAME"
echo "Output: $HOST_OUTPUT"
echo ""

docker run --rm --init \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --shm-size=16g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "${SCRIPT_DIR}:/repro" \
    -v "${HOST_OUTPUT}:/output" \
    -e "PRECONDITION_FREQUENCY=${PRECONDITION_FREQUENCY}" \
    -e ROCPROF_TMPDIR=/dev/shm \
    -e "HSA_TOOLS_DISABLE_REGISTER=1" \
    -e "LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}" \
    "${PRECISION_ENVS[@]}" \
    "${DOCKER_IMAGE}" \
    bash -c "
        echo 'Starting rocprofv3 profiling (${NUM_STEPS} steps, GPU ${GPU})...'
        HIP_VISIBLE_DEVICES=${GPU} NUM_STEPS=${NUM_STEPS} MASTER_PORT=29500 \
        rocprofv3 --kernel-trace \
        --output-format csv \
        -d /output/rocprof \
        -- python3 /repro/${SCRIPT_NAME} 2>&1 | tee /output/rocprof_run.log

        echo ''
        echo '=== rocprofv3 output files ==='
        find /output/rocprof/ -type f -exec ls -lh {} \; 2>/dev/null || echo 'No output files found'
    "

echo ""
echo "Results saved to: ${HOST_OUTPUT}"
