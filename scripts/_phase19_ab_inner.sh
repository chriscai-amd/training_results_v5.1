#!/usr/bin/env bash
# Inner script for Phase-19 A/B smoke test (runs inside container).
#   1) incremental build of HCTR with the swizzle memcpy-stream patch
#   2) run baseline (HCTR_USE_MEMCPY_STREAM=0) at bs=1x for ~250 iters
#   3) run Phase-19 (HCTR_USE_MEMCPY_STREAM=1) same config
#   4) print steady-state sps for both
set -eu

apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
/usr/bin/python3 -m pip install --break-system-packages --quiet pybind11 numpy mpi4py mlperf-logging 2>&1 | tail -1 || true

echo "===== Phase-19 build (incremental) ====="
cd /workspace/hugectr_hip/build_rocm72
export PATH=/opt/rocm/bin:$PATH
T0=$(date +%s)
cmake --build . -j 16 2>&1 | tail -8
T1=$(date +%s)
echo "[ok] build took $((T1-T0))s"
ls -lh lib/*.so 2>/dev/null

export LD_LIBRARY_PATH=/opt/rocm/lib:/workspace/hugectr_hip/build_rocm72/lib
export PYTHONPATH=/workspace/hugectr_hip/build_rocm72/lib:${PYTHONPATH:-}

# Standard env from run_b200_match.sh so we get apples-to-apples.
export NCCL_PROTO=LL128 NCCL_ALGO=Ring HIP_FORCE_DEV_KERNARG=1
export DEBUG_HIP_DYNAMIC_QUEUES=1 DEBUG_HIP_BLOCK_SYNC=0 NCCL_BUFFSIZE=8388608
export HCTR_USE_MULTI_HOT=1 HCTR_USE_MLPERF_CRITEO=1 HCTR_USE_CUDA_GRAPH=1
export HCTR_USE_SUBSAMPLED_CRITEO=1 HCTR_USE_REAL_TABLE_SIZES=1
export HCTR_DP_SHARD_THRESH=0.008 HCTR_MEM_COMM_WORK_RATIO=9 HCTR_READER_THREADS=1
export HCTR_ASYNC_WGRAD=0 HCTR_NGPU=8 HCTR_BATCH=55296 HCTR_EVAL_BATCH=131072
export HCTR_EV_SIZE=128 HCTR_LR=0.004 HCTR_DISPLAY=10 HCTR_MEM_CAP=200
export HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348'
export HCTR_SHARDING_PLAN=auto OMP_NUM_THREADS=8

run_one() {
    local label=$1
    local mcpy_val=$2
    local iters=${3:-250}
    local logf=/tmp/p19_${label}.log
    echo
    echo "===== run [$label]  HCTR_USE_MEMCPY_STREAM=$mcpy_val  iters=$iters ====="
    HCTR_MAX_ITER=$iters HCTR_USE_MEMCPY_STREAM=$mcpy_val \
        bash /workspace/run_b200_match.sh > "$logf" 2>&1 || true
    echo "[$label] tail:"
    grep -aE "Steady|Final loss|Loss range|Per-iter" "$logf" | tail -5
    echo "[$label] log: $logf"
}

# 250 iters: 50 warmup + ~200 steady measurement
run_one baseline 0 250
run_one phase19  1 250

echo
echo "===== Phase-19 A/B SUMMARY ====="
for label in baseline phase19; do
    sps=$(grep -aE "Steady-state throughput" /tmp/p19_${label}.log | tail -1 | awk '{print $(NF-2)}')
    pi=$(grep -aE "Per-iter time" /tmp/p19_${label}.log | tail -1 | awk '{print $(NF-1)}')
    loss=$(grep -aE "Final loss" /tmp/p19_${label}.log | tail -1 | awk '{print $NF}')
    echo "[$label]  sps=${sps:-?} M  per-iter=${pi:-?} ms  final_loss=${loss:-?}"
done
