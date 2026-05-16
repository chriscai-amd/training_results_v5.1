#!/usr/bin/env bash
# Quick sequential smoke: 1 baseline + 1 phase19, both ~120 iters, on one node.
set -eu
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
/usr/bin/python3 -m pip install --break-system-packages --quiet mpi4py mlperf-logging 2>&1 | tail -1 || true

export LD_LIBRARY_PATH=/opt/rocm/lib:/workspace/hugectr_hip/build_rocm72/lib
export PYTHONPATH=/workspace/hugectr_hip/build_rocm72/lib:${PYTHONPATH:-}
export NCCL_PROTO=LL128 NCCL_ALGO=Ring HIP_FORCE_DEV_KERNARG=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT
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
    local iters=${3:-120}
    local logf=/sweep_log/p19_smoke_${label}.log
    echo
    echo "===== run [$label]  HCTR_USE_MEMCPY_STREAM=$mcpy_val  iters=$iters ====="
    HCTR_MAX_ITER=$iters HCTR_USE_MEMCPY_STREAM=$mcpy_val \
        bash /workspace/run_b200_match.sh > "$logf" 2>&1 || true
    echo "[$label] tail (last 30 lines):"
    tail -30 "$logf"
}

run_one baseline 0 120
run_one phase19  1 120

echo
echo "===== SMOKE SUMMARY ====="
for label in baseline phase19; do
    f=/sweep_log/p19_smoke_${label}.log
    sps=$(grep -aE "Steady-state throughput" "$f" | tail -1 | awk '{print $(NF-2)}')
    pi=$(grep -aE "Per-iter time" "$f" | tail -1 | awk '{print $(NF-1)}')
    loss=$(grep -aE "Final loss" "$f" | tail -1 | awk '{print $NF}')
    fail=$(grep -aE "RuntimeError|Traceback" "$f" | head -1)
    echo "[$label]  sps=${sps:-?} M  per-iter=${pi:-?} ms  final_loss=${loss:-?}  ${fail:+FAIL}"
done
