#!/usr/bin/env bash
# Inner script: run a single configuration (baseline or phase19).
# Args:
#   $1 = label (used in log filename)
#   $2 = HCTR_USE_MEMCPY_STREAM value (0 or 1)
#   $3 = MAX_ITER (default 250)
set -eu
LABEL=${1:?label required}
USE_MCPY=${2:?HCTR_USE_MEMCPY_STREAM value required}
ITERS=${3:-250}

apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
/usr/bin/python3 -m pip install --break-system-packages --quiet mpi4py mlperf-logging 2>&1 | tail -1 || true

export LD_LIBRARY_PATH=/opt/rocm/lib:/workspace/hugectr_hip/build_rocm72/lib
export PYTHONPATH=/workspace/hugectr_hip/build_rocm72/lib:${PYTHONPATH:-}
export NCCL_PROTO=LL128 NCCL_ALGO=Ring HIP_FORCE_DEV_KERNARG=1
# RCCL bootstrap: force loopback so we don't pick a misconfigured OOB
# benic interface (some reservations have a /32 with no route to host on
# 192.168.15.0). Confirmed needed on cv350-rck-g03-c10-08 (May 15 2026).
export NCCL_SOCKET_IFNAME=lo
export DEBUG_HIP_DYNAMIC_QUEUES=1 DEBUG_HIP_BLOCK_SYNC=0 NCCL_BUFFSIZE=8388608
export HCTR_USE_MULTI_HOT=1 HCTR_USE_MLPERF_CRITEO=1 HCTR_USE_CUDA_GRAPH=1
export HCTR_USE_SUBSAMPLED_CRITEO=1 HCTR_USE_REAL_TABLE_SIZES=1
export HCTR_DP_SHARD_THRESH=0.008 HCTR_MEM_COMM_WORK_RATIO=9 HCTR_READER_THREADS=1
export HCTR_ASYNC_WGRAD=0 HCTR_NGPU=8 HCTR_BATCH=55296 HCTR_EVAL_BATCH=131072
export HCTR_EV_SIZE=128 HCTR_LR=0.004 HCTR_DISPLAY=10 HCTR_MEM_CAP=200
export HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348'
export HCTR_SHARDING_PLAN=auto OMP_NUM_THREADS=8

export HCTR_MAX_ITER=$ITERS
export HCTR_USE_MEMCPY_STREAM=$USE_MCPY

echo "===== run [$LABEL] HCTR_USE_MEMCPY_STREAM=$USE_MCPY iters=$ITERS ====="
bash /workspace/run_b200_match.sh
echo
echo "===== [$LABEL] DONE ====="
