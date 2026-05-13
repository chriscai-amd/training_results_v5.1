#!/usr/bin/env bash
# Match the NVIDIA B200 reference run on AMD MI350X.
# Keep our existing day_0 binary (single-hot, 4.75 M train rows) but flip
# every other knob to the B200 reference values:
#   - 8 GPUs, global batch 55,296 (= 6912 / GPU), eval batch 1,048,576
#   - LR 0.004 constant, Adagrad, mixed precision (FP16) + scaler 16,348
#   - ev_size 128, real TABLE_SIZE_ARRAY (~204 M IDs, 3 caps at 40 M)
#   - auto sharding plan, 100 iters (perf only)
set -eu

apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 > /dev/null 2>&1 || true
pip install --quiet mpi4py mlperf-logging 2>&1 | tail -3 || true

export LD_LIBRARY_PATH=/opt/rocm/lib:/workspace/hugectr_hip/build_rocm72/lib
export PYTHONPATH=/workspace/hugectr_hip/build_rocm72/lib:${PYTHONPATH:-}

# ROCm port: RCCL tunings that bumped sustained perf from 11.76 -> 11.88 M sps
# at NV's batch (55,296) on /dev/shm. AMD's RCCL default is Simple proto, which
# is fine but LL128 is ~1 % faster for our small grouped-allreduce shapes
# (~6 MiB total for the 5 top-MLP wgrads + 3 bottom-MLP wgrads). Ring is
# already RCCL's default for 8-rank intra-node.
export NCCL_PROTO=${NCCL_PROTO:-LL128}
export NCCL_ALGO=${NCCL_ALGO:-Ring}

# ROCm port: HIP_FORCE_DEV_KERNARG=1 makes HIP write kernel-arg buffers
# directly into device-visible memory, avoiding a host-side staging
# buffer copy on every kernel launch. With HIP graphs this saves a few
# 100 ns per kernel; analogous to NV's CUDA_DEVICE_MAX_CONNECTIONS=64
# (which NV measured at +0.9 % on B200). On AMD MI350X with HCTR's
# deep CUDA-graph we measured the gain as within day-to-day noise
# (~+0.1 % across 3 trials), but it has no downside so we bake it in.
export HIP_FORCE_DEV_KERNARG=${HIP_FORCE_DEV_KERNARG:-1}

# ROCm port (May 2026): DEBUG_HIP_DYNAMIC_QUEUES=1 enables AMD CDNA3
# dynamic-queue allocation, which lets HIP grab additional hardware queues
# on demand instead of pinning everything to a fixed pool. On HCTR's
# 4-stream pipeline (compute / RCCL / copy / embedding) this gives
# significantly better stream concurrency: per the rocprofv3 trace, the
# union-of-busy-streams per iter drops from 12.10 ms -> 10.64 ms (-12 %),
# the per-iter wall drops from 13.36 ms -> 12.03 ms (-10 %), and the
# "other" stream's p90 inter-kernel gap drops from 15.3 ms -> 0.27 ms
# (-98 %). End-to-end: +6.0 % at bs4x (15.21 -> 16.12 M sps) and +8.9 %
# at bs1x (11.86 -> 12.92 M sps). Largest single-knob win we found in
# the entire env-var optimisation campaign. Loss is consistent across
# all 5 trials per config (0.288764 -> 0.288771-829, within FP16
# noise from kernel-order changes).
export DEBUG_HIP_DYNAMIC_QUEUES=${DEBUG_HIP_DYNAMIC_QUEUES:-1}

# ROCm port: DEBUG_HIP_BLOCK_SYNC=0 disables host-blocking on stream
# sync (default in newer ROCm is 1 = block on event waits). With
# DYN_QUEUES on, BLOCK_SYNC=0 stacks for an additional small win
# (16.12 -> 16.15 M sps at bs4x, +0.2 %).
export DEBUG_HIP_BLOCK_SYNC=${DEBUG_HIP_BLOCK_SYNC:-0}

# ROCm port: NCCL_BUFFSIZE=8388608 (8 MiB) reduced steady ms/iter by
# ~0.7 % at bs4x in our sweep -- baked in.
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-8388608}

# Match B200 reference dataset shape (table sizes). Three multi-hot tiers,
# from highest fidelity to lowest:
#   HCTR_USE_MLPERF_CRITEO=1     -> real MLPerf-published Criteo from R2
#                                    (downloaded from training.mlcommons-storage.org).
#                                    This is what NVIDIA's submission actually consumes.
#   HCTR_USE_FULL_CRITEO=1       -> our 24-day prime-mixed synthetic from HF day_*.gz
#                                    (482 M rows; close to NVIDIA's HF-subsample run).
#   default (HCTR_USE_MULTI_HOT) -> day_0-only synthetic.
# All three tiers share the 912 B/row, 214 keys/row format that Layer_t.MLP expects.
export HCTR_USE_SUBSAMPLED_CRITEO=1
export HCTR_USE_REAL_TABLE_SIZES=1
if [ "${HCTR_USE_MULTI_HOT:-0}" = "1" ]; then
    if [ "${HCTR_USE_MLPERF_CRITEO:-0}" = "1" ] && [ -d /criteo/mlperf ]; then
        DATA_DIR=/criteo/mlperf
        echo "[ok] data shape = MULTI-HOT REAL MLPERF CRITEO (R2-hosted, 214 keys/row, 912 B/row)"
    elif [ "${HCTR_USE_FULL_CRITEO:-0}" = "1" ] && [ -d /criteo/hugectr_bin_mh_full ]; then
        DATA_DIR=/criteo/hugectr_bin_mh_full
        echo "[ok] data shape = MULTI-HOT FULL CRITEO (24 days synthetic, 214 keys/row, 912 B/row)"
    else
        DATA_DIR=/criteo/hugectr_bin_mh
        echo "[ok] data shape = MULTI-HOT day_0 only (214 keys/row, 912 B/row)"
    fi
    # 4 (label) + 13*4 (dense) + sum(MULTI_HOT_SIZES)*4 = 4 + 52 + 214*4 = 912.
    BYTES_PER_ROW=912
else
    DATA_DIR=/criteo/hugectr_bin
    BYTES_PER_ROW=160
    echo "[ok] data shape = SINGLE-HOT (26 keys/row, 160 B/row)"
fi
export HCTR_TRAIN_NUM_SAMPLES=$(stat -c %s "$DATA_DIR/train_data.bin" | awk -v b=$BYTES_PER_ROW '{print int($1/b)}')
export HCTR_EVAL_NUM_SAMPLES=$(stat  -c %s "$DATA_DIR/val_data.bin"   | awk -v b=$BYTES_PER_ROW '{print int($1/b)}')
export HCTR_AUC_THRESHOLD=0.99
echo "[ok] TRAIN_NUM_SAMPLES=$HCTR_TRAIN_NUM_SAMPLES  EVAL_NUM_SAMPLES=$HCTR_EVAL_NUM_SAMPLES"

cd /workspace/runtime_test/nvidia_frontend
ulimit -c 0

NGPU=${HCTR_NGPU:-8}
BATCH=${HCTR_BATCH:-55296}
EVAL_BATCH=${HCTR_EVAL_BATCH:-1048576}
EV_SIZE=${HCTR_EV_SIZE:-128}
LR=${HCTR_LR:-0.004}
MAX_ITER=${HCTR_MAX_ITER:-100}
DISPLAY=${HCTR_DISPLAY:-10}
PRECISION_FLAGS=${HCTR_PRECISION_FLAGS:-"--use_mixed_precision --scaler 16348"}

echo "[run] ngpus=$NGPU batch=$BATCH eval_batch=$EVAL_BATCH ev_size=$EV_SIZE lr=$LR max_iter=$MAX_ITER"
echo "[run] precision_flags='$PRECISION_FLAGS'"

# MI350X has 288 GB HBM3; use 256 GB for embedding budget (default is 60 GB
# which OOMs against the real 204 M-ID TABLE_SIZE_ARRAY at ev_size=128).
MEM_CAP=${HCTR_MEM_CAP:-256}

ALGO_SEARCH_FLAG=""
if [ "${HCTR_ENABLE_ALGO_SEARCH:-0}" != "1" ]; then
    ALGO_SEARCH_FLAG="--disable_algorithm_search"
fi

${HCTR_PROFILE_PREFIX:-} python3 train.py $ALGO_SEARCH_FLAG \
    --batchsize "$BATCH" \
    --batchsize_eval "$EVAL_BATCH" \
    --ev_size "$EV_SIZE" \
    --lr "$LR" \
    --max_iter "$MAX_ITER" \
    --display_interval "$DISPLAY" \
    --eval_interval 999999 \
    --num_gpus_per_node "$NGPU" \
    --train_data "$DATA_DIR/train_data.bin" \
    --val_data   "$DATA_DIR/val_data.bin" \
    --sharding_plan "${HCTR_SHARDING_PLAN:-auto}" \
    --mem_comm_bw_ratio "${HCTR_MEM_COMM_BW_RATIO:-9}" \
    --mem_comm_work_ratio "${HCTR_MEM_COMM_WORK_RATIO:-9}" \
    --dp_sharding_threshold "${HCTR_DP_SHARD_THRESH:-0.008}" \
    --memory_cap_for_embedding "$MEM_CAP" \
    --gen_loss_summary \
    --optimizer "${HCTR_OPTIMIZER:-adagrad}" \
    $PRECISION_FLAGS 2>&1 | tee /tmp/b200match.log

echo ""
echo "============================================================"
echo "         B200-MATCH RUN SUMMARY (AMD MI350X x $NGPU)"
echo "============================================================"
SUMMARY_BS=$BATCH SUMMARY_INTER=$DISPLAY SUMMARY_NGPU=$NGPU \
SUMMARY_EVAL_BATCH=$EVAL_BATCH SUMMARY_EV_SIZE=$EV_SIZE \
SUMMARY_PRECISION="$PRECISION_FLAGS" \
python3 - <<'PY'
import os, re
log = open("/tmp/b200match.log").read()
iters = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
         for m in re.finditer(r"Iter: (\d+) Time\(\d+ iters\): ([\d.]+)s Loss: ([\d.]+)", log)]
bs    = int(os.environ["SUMMARY_BS"])
inter = int(os.environ["SUMMARY_INTER"])
ngpu  = int(os.environ["SUMMARY_NGPU"])
eval_batch = os.environ["SUMMARY_EVAL_BATCH"]
ev_size    = os.environ["SUMMARY_EV_SIZE"]
precision  = os.environ["SUMMARY_PRECISION"]
if iters:
    its, secs, losses = zip(*iters)
    sps = [(bs * inter) / s for s in secs]
    # Trim warm-up (first entry) and the partial-window outlier (last entry,
    # which often reports a ~1ms time because the run terminated mid-window).
    if len(sps) >= 3:
        steady = sps[1:-1]
    elif len(sps) > 1:
        steady = sps[1:]
    else:
        steady = sps
    avg = sum(steady) / len(steady)
    print(f"GPUs                       : {ngpu} x AMD MI350X (gfx950)")
    print(f"Global batch / per-GPU     : {bs:,} / {bs//ngpu:,}")
    print(f"Eval batch                 : {eval_batch}")
    print(f"Embedding dim              : {ev_size}")
    prec_label = "FP16 mixed (scaler 16348)" if "mixed_precision" in precision else "FP32"
    print(f"Precision                  : {prec_label}")
    print(f"Iterations completed       : {its[-1]}")
    print(f"Final loss (BCE)           : {losses[-1]:.6f}")
    print(f"Loss range over training   : {max(losses):.4f} -> {min(losses):.4f}")
    print(f"Steady-state throughput    : {avg/1e6:.3f} M samples/sec")
    print(f"Per-iter time (steady)     : {(bs*inter)/avg*1000/inter:.2f} ms/iter")
    sec = 4_195_197_692 / avg
    print(f"Projected full Criteo epoch: {sec:.1f} sec ({sec/60:.1f} min)")
PY
