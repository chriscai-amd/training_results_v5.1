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

# Match B200 reference dataset shape (table sizes). Pick single-hot vs
# synthetic multi-hot (sum=130 keys, 576 B/row) based on HCTR_USE_MULTI_HOT.
export HCTR_USE_SUBSAMPLED_CRITEO=1
export HCTR_USE_REAL_TABLE_SIZES=1
if [ "${HCTR_USE_MULTI_HOT:-0}" = "1" ]; then
    DATA_DIR=/criteo/hugectr_bin_mh
    BYTES_PER_ROW=576
    echo "[ok] data shape = MULTI-HOT (130 keys/row, 576 B/row)"
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

python3 train.py \
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
    --mem_comm_work_ratio 9 \
    --dp_sharding_threshold 0.008 \
    --memory_cap_for_embedding "$MEM_CAP" \
    --disable_algorithm_search \
    --gen_loss_summary \
    --optimizer "${HCTR_OPTIMIZER:-adagrad}" \
    $PRECISION_FLAGS 2>&1 | tee /tmp/b200match.log

echo ""
echo "============================================================"
echo "         B200-MATCH RUN SUMMARY (AMD MI350X x $NGPU)"
echo "============================================================"
python3 - <<PY
import re
log = open("/tmp/b200match.log").read()
iters = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
         for m in re.finditer(r"Iter: (\d+) Time\(\d+ iters\): ([\d.]+)s Loss: ([\d.]+)", log)]
bs = $BATCH
inter = $DISPLAY
if iters:
    its, secs, losses = zip(*iters)
    sps = [(bs * inter) / s for s in secs]
    steady = sps[1:] if len(sps) > 1 else sps
    avg = sum(steady) / len(steady)
    print(f"GPUs                       : $NGPU x AMD MI350X (gfx950)")
    print(f"Global batch / per-GPU     : {bs:,} / {bs//$NGPU:,}")
    print(f"Eval batch                 : $EVAL_BATCH")
    print(f"Embedding dim              : $EV_SIZE")
    print(f"Precision                  : {'FP16 mixed (scaler 16348)' if 'mixed_precision' in '$PRECISION_FLAGS' else 'FP32'}")
    print(f"Iterations completed       : {its[-1]}")
    print(f"Final loss (BCE)           : {losses[-1]:.6f}")
    print(f"Loss range over training   : {max(losses):.4f} -> {min(losses):.4f}")
    print(f"Steady-state throughput    : {avg/1e6:.3f} M samples/sec")
    print(f"Per-iter time (steady)     : {(bs*inter)/avg*1000/inter:.2f} ms/iter")
    sec = 4_195_197_692 / avg
    print(f"Projected full Criteo epoch: {sec:.1f} sec ({sec/60:.1f} min)")
PY
