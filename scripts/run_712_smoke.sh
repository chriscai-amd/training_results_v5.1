#!/bin/bash
# Run bs=1x smoke test against ROCm 7.12 build.
set -e

apt-get update -qq 2>&1 | tail -1
apt-get install -y --no-install-recommends -qq \
    libnuma1 libtbb12 libaio1t64 libopenmpi3t64 openmpi-bin \
    python3-pip python3-dev libpython3-dev 2>&1 | tail -1

pip3 install --break-system-packages --quiet \
    pybind11 numpy mpi4py mlperf-logging 2>&1 | tail -2

# Set up ROCm 7.12
ROCM_DIR=/opt/rocm-7.12.0rc1
ln -sfn /apps/chcai/rocm_712_extracted ${ROCM_DIR}
ln -sfn ${ROCM_DIR} /opt/rocm
export PATH=${ROCM_DIR}/bin:${ROCM_DIR}/lib/llvm/bin:$PATH

# Critical: link our 7.12 build's hugectr.so + use 7.12 ROCm libs (not the
# default 7.2 the no_rocm container would otherwise hit if any are pre-linked)
export LD_LIBRARY_PATH=${ROCM_DIR}/lib:/apps/chcai/build_rocm712/lib
export PYTHONPATH=/apps/chcai/build_rocm712/lib:${PYTHONPATH:-}

# libaio shim (we already mounted host libaio.so.1.0.1 to /usr/lib path)
ln -sf /usr/lib/x86_64-linux-gnu/libaio.so.1.0.1 /usr/lib/x86_64-linux-gnu/libaio.so.1 2>/dev/null || true

# RCCL / HIP env from production run_b200_match.sh
export NCCL_PROTO=LL128
export NCCL_ALGO=Ring
export HIP_FORCE_DEV_KERNARG=1
export DEBUG_HIP_DYNAMIC_QUEUES=1
export DEBUG_HIP_BLOCK_SYNC=0
export NCCL_BUFFSIZE=8388608

# HCTR config
export HCTR_NGPU=8 HCTR_BATCH=55296 HCTR_EVAL_BATCH=131072
export HCTR_EV_SIZE=128 HCTR_LR=0.004 HCTR_MAX_ITER=1500 HCTR_DISPLAY=300
export HCTR_MEM_CAP=200 HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348'
export HCTR_SHARDING_PLAN=auto OMP_NUM_THREADS=8
export HCTR_USE_MULTI_HOT=1 HCTR_USE_MLPERF_CRITEO=1
export HCTR_USE_CUDA_GRAPH=1
export HCTR_DP_SHARD_THRESH=0.008 HCTR_MEM_COMM_WORK_RATIO=9
export HCTR_READER_THREADS=1
export HCTR_USE_SUBSAMPLED_CRITEO=1
export HCTR_USE_REAL_TABLE_SIZES=1

cd /host/hugectr_rocm_port

# Verify hugectr.so is the 7.12 build (linked against 7.12 amdhip64)
echo "=== libs hugectr.so depends on ==="
ldd /apps/chcai/build_rocm712/lib/hugectr.so 2>&1 | grep -aE "amdhip|hipblas|rccl|libomp" | head -10

echo
echo "=== Python sanity check ==="
python3 -c "
import sys
sys.path.insert(0, '/apps/chcai/build_rocm712/lib')
import hugectr
print('hugectr module imported OK from 7.12 build')
print('hugectr file:', hugectr.__file__)
"

echo
echo "=== bs=1x smoke test ==="
bash /host/hugectr_rocm_port/run_b200_match.sh 2>&1 | grep -aE "M sps|samples/sec|ms/iter|Iter|loss|Final|allocating.*GB|OOM|Error|Traceback|HIP Graph Segmented" | tail -30
