#!/usr/bin/env bash
# Capture a 15-iter trace WITH --memory-copy-trace so HSA-level DMA copies
# (h2d / d2h / d2d) are recorded -- those don't show up under --kernel-trace
# alone, which is why our prior traces appeared to lack a memcpy lane.
# Process only GPU 0 (rank 0) to keep iteration fast.
set -eu

apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
/usr/bin/python3 -m pip install --break-system-packages --quiet mpi4py mlperf-logging 2>&1 | tail -1 || true

echo "=== stage data ==="
mkdir -p /ramdata/mlperf
T0=$(date +%s)
cp /nfs_data/train_data.bin /ramdata/mlperf/ &
cp /nfs_data/val_data.bin /ramdata/mlperf/ &
wait
T1=$(date +%s)
echo "=== staging done in $((T1-T0))s ==="
ln -sf /ramdata /criteo

echo "=== run rocprofv3 with --memory-copy-trace (15 iters; pick steady 5-9) ==="
TRACE_DIR=/rps_out/trace_memcpy
rm -rf "$TRACE_DIR"
mkdir -p "$TRACE_DIR"
chmod 777 "$TRACE_DIR"

cd /workspace/runtime_test/nvidia_frontend
export HCTR_MAX_ITER=15
export HCTR_DISPLAY=5
export NCCL_SOCKET_IFNAME=lo
export HCTR_USE_MULTI_HOT=1 HCTR_USE_MLPERF_CRITEO=1 HCTR_USE_CUDA_GRAPH=1
export HCTR_USE_SUBSAMPLED_CRITEO=1 HCTR_USE_REAL_TABLE_SIZES=1
export HCTR_DP_SHARD_THRESH=0.008 HCTR_MEM_COMM_WORK_RATIO=9 HCTR_READER_THREADS=1
export HCTR_ASYNC_WGRAD=0 HCTR_NGPU=8 HCTR_BATCH=55296 HCTR_EVAL_BATCH=131072
export HCTR_EV_SIZE=128 HCTR_LR=0.004 HCTR_MEM_CAP=200
export HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348'
export HCTR_SHARDING_PLAN=auto OMP_NUM_THREADS=8

HCTR_PROFILE_PREFIX="rocprofv3 --kernel-trace --hip-trace --memory-copy-trace --output-format csv -d $TRACE_DIR --output-file iter5_10 --" \
    bash /workspace/run_b200_match.sh

echo
echo "=== output CSVs (note the new memory_copy_trace.csv) ==="
ls -lh "$TRACE_DIR"
echo
echo "=== peek at memory_copy_trace.csv schema ==="
head -1 "$TRACE_DIR/iter5_10_memory_copy_trace.csv" 2>/dev/null || echo "  no memory_copy_trace.csv produced (maybe a different filename?)"
head -3 "$TRACE_DIR/iter5_10_memory_copy_trace.csv" 2>/dev/null | tail -2

echo
echo "=== convert GPU 0 only (skipping merge to save time) ==="
OUT=/results/post_phase18_memcpy_perfetto
rm -rf "$OUT" 2>/dev/null
mkdir -p "$OUT"
chmod 777 "$OUT"
python3 /workspace/scripts/rocprofv3_to_perfetto_annotated.py \
    "$TRACE_DIR/iter5_10" 0 5 5 "$OUT/iter_steady_gpu0.json" 2>&1 | tail -10

echo
echo "=== output files ==="
ls -lh "$OUT"
