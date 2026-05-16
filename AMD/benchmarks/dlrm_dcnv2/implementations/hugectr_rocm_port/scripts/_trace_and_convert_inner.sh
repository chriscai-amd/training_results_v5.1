#!/usr/bin/env bash
# In-container payload for trace_and_convert.sh.
#
# Runs HCTR for ${MAX_ITER} iterations under rocprofv3 with the full set
# of tracing flags (kernel + hip + memory-copy + rccl), then converts the
# raw csvs into per-GPU annotated Perfetto JSONs + a merged 8-GPU view
# for iters ${START_ITER}..${END_ITER}-1.
#
# Mounted dirs (from outer script):
#   /workspace -> hugectr_rocm_port (HCTR source + build + run_b200_match.sh)
#   /nfs_data  -> MLPerf Criteo on NFS (staged into tmpfs /ramdata)
#   /scripts   -> (legacy; kept for compat with other inner scripts)
#   /rps_out   -> /home/chcai/rps_out (raw rocprofv3 csv outputs land here)
#   /results   -> training_results_v5.1/results (final perfetto JSONs)
set -eu

START_ITER=${1:?"START_ITER required"}
END_ITER=${2:?"END_ITER required"}
MAX_ITER=${3:?"MAX_ITER required"}
TRACE_SUBDIR=${4:?"TRACE_SUBDIR required"}
OUT_SUBDIR=${5:?"OUT_SUBDIR required"}
N_ITERS=$((END_ITER - START_ITER))

apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
/usr/bin/python3 -m pip install --break-system-packages --quiet mpi4py mlperf-logging 2>&1 | tail -1 || true

echo "=== stage data (NFS -> tmpfs /ramdata) ==="
mkdir -p /ramdata/mlperf
T0=$(date +%s)
cp /nfs_data/train_data.bin /ramdata/mlperf/ &
cp /nfs_data/val_data.bin /ramdata/mlperf/ &
wait
T1=$(date +%s)
echo "=== staging done in $((T1-T0))s ==="
ln -sf /ramdata /criteo

TRACE_DIR=/rps_out/$TRACE_SUBDIR
rm -rf "$TRACE_DIR"
mkdir -p "$TRACE_DIR"
chmod 777 "$TRACE_DIR"

# canonical "iter5_10" style basename even when the window is different;
# anchored on START/END so the file naming reflects the actual window.
BASENAME="iter${START_ITER}_${END_ITER}"

echo "=== run HCTR under rocprofv3 (max_iter=$MAX_ITER; window=${START_ITER}..${END_ITER}) ==="
cd /workspace/runtime_test/nvidia_frontend
# Match run_apple_to_apple.sh environment so the trace is apples-to-apples
# with the perf-config defaults.
export HCTR_MAX_ITER=$MAX_ITER
export HCTR_DISPLAY=5
export NCCL_SOCKET_IFNAME=lo
export HCTR_USE_MULTI_HOT=1 HCTR_USE_MLPERF_CRITEO=1 HCTR_USE_CUDA_GRAPH=1
export HCTR_USE_SUBSAMPLED_CRITEO=1 HCTR_USE_REAL_TABLE_SIZES=1
export HCTR_DP_SHARD_THRESH=0.008 HCTR_MEM_COMM_WORK_RATIO=9 HCTR_READER_THREADS=1
export HCTR_ASYNC_WGRAD=0 HCTR_NGPU=8 HCTR_BATCH=55296 HCTR_EVAL_BATCH=131072
export HCTR_EV_SIZE=128 HCTR_LR=0.004 HCTR_MEM_CAP=200
export HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348'
export HCTR_SHARDING_PLAN=auto OMP_NUM_THREADS=8

# All four tracing flags together so the trace is COMPLETE (kernel
# dispatches + HIP API + DMA memcpys + RCCL collective API). Omitting
# --memory-copy-trace makes the data reader's H2D lane invisible;
# omitting --rccl-trace makes RCCL kernels indistinguishable by op type.
HCTR_PROFILE_PREFIX="rocprofv3 \
    --kernel-trace \
    --hip-trace \
    --memory-copy-trace \
    --rccl-trace \
    --output-format csv \
    -d $TRACE_DIR \
    --output-file $BASENAME --" \
    bash /workspace/run_b200_match.sh

echo
echo "=== raw CSVs produced ==="
ls -lh "$TRACE_DIR"

OUT=/results/$OUT_SUBDIR
rm -rf "$OUT" 2>/dev/null || true
mkdir -p "$OUT"
chmod 777 "$OUT"

echo
echo "=== convert per-GPU CSVs -> annotated Perfetto JSONs (window ${START_ITER}..${END_ITER}, $N_ITERS iters) ==="
for g in 0 1 2 3 4 5 6 7; do
    OUT_JSON="$OUT/iter_steady_gpu${g}.json"
    echo "  -- gpu $g -> $OUT_JSON --"
    python3 /workspace/scripts/rocprofv3_to_perfetto_annotated.py \
        "$TRACE_DIR/$BASENAME" "$g" "$START_ITER" "$N_ITERS" "$OUT_JSON" 2>&1 \
        | grep -E "kernels in window|DMA memcpy|RCCL API|RCCL operation-type|role=|memcpy_h2d" \
        | head -15
done

echo
echo "=== merge per-GPU JSONs into 8-GPU view ==="
python3 /workspace/scripts/merge_perfetto_gpus.py \
    "$TRACE_DIR/$BASENAME" \
    "$OUT/iter_steady_gpu*.json" \
    "$OUT/iter_steady_all8gpus.json" \
    "$START_ITER" "$N_ITERS" 2>&1 | tail -12

echo
echo "=== final output files ==="
ls -lh "$OUT"
