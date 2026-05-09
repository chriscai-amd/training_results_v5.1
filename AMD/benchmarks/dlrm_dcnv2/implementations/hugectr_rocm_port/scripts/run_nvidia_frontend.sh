#!/usr/bin/env bash
# Launch the (lightly adapted) NVIDIA MLPerf DLRM-DCNv2 train.py inside the
# AMD nightly container, against our HugeCTR ROCm build + Criteo day_0 binary.
set -eu
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 > /dev/null 2>&1 || true
pip install --quiet mpi4py mlperf-logging 2>&1 | tail -3 || true

export LD_LIBRARY_PATH=/opt/rocm/lib:/workspace/hugectr_hip/build_rocm72/lib
export PYTHONPATH=/workspace/hugectr_hip/build_rocm72/lib:${PYTHONPATH:-}

# Tell NVIDIA's frontend to use our subsampled Criteo dataset shape.
export HCTR_USE_SUBSAMPLED_CRITEO=1
export HCTR_SLOT_SIZE=65536
export HCTR_TRAIN_NUM_SAMPLES=$(stat -c %s /criteo/hugectr_bin/train_data.bin | awk '{print int($1/160)}')
export HCTR_EVAL_NUM_SAMPLES=$(stat -c %s  /criteo/hugectr_bin/val_data.bin   | awk '{print int($1/160)}')
export HCTR_AUC_THRESHOLD=0.99   # never trip the early-stop in the smoke run
echo "[ok] TRAIN_NUM_SAMPLES=$HCTR_TRAIN_NUM_SAMPLES  EVAL_NUM_SAMPLES=$HCTR_EVAL_NUM_SAMPLES"

cd /workspace/runtime_test/nvidia_frontend
ulimit -c 0

# Use Adagrad, batch 8192, ev_size 16 (matches our pre-built smaller embeddings),
# round_robin sharding (default), 1 GPU on this node.
python3 train.py \
    --batchsize 8192 \
    --batchsize_eval 8192 \
    --ev_size 16 \
    --max_iter 1000 \
    --display_interval 100 \
    --eval_interval 999999 \
    --num_gpus_per_node 1 \
    --train_data /criteo/hugectr_bin/train_data.bin \
    --val_data   /criteo/hugectr_bin/val_data.bin \
    --sharding_plan round_robin \
    --disable_algorithm_search \
    --gen_loss_summary \
    --optimizer adagrad 2>&1 | tee /tmp/nvfront_full.log

echo ""
echo "============================================================"
echo "                  RUN SUMMARY (AMD MI350X)"
echo "============================================================"
python3 - <<'PY'
import re, sys, json
log = open("/tmp/nvfront_full.log").read()
iters = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
         for m in re.finditer(r"Iter: (\d+) Time\(\d+ iters\): ([\d.]+)s Loss: ([\d.]+)", log)]
if iters:
    its, secs, losses = zip(*iters)
    bs = 8192
    inter = its[1] - its[0] if len(its) > 1 else its[0]
    samples_per_iter = bs
    samples_per_sec = [(samples_per_iter * inter) / s for s in secs]
    # Skip the warm-up iter (first window contains compile + first kernel JIT).
    steady = samples_per_sec[1:] if len(samples_per_sec) > 1 else samples_per_sec
    avg = sum(steady) / len(steady)
    print(f"Iterations completed       : {its[-1]}")
    print(f"Final loss (BCE)           : {losses[-1]:.6f}")
    print(f"Loss range over training   : {max(losses):.4f} -> {min(losses):.4f}")
    print(f"Steady-state throughput    : {avg/1e6:.3f} M samples/sec")
    print(f"Per-iter time (steady)     : {(bs*inter)/avg*1e3:.2f} ms / {inter} iters ({(bs*inter)/avg*10:.3f} ms/iter)")
    full_criteo = 4_195_197_692
    if avg > 0:
        sec = full_criteo / avg
        print(f"Projected full Criteo epoch: {sec:.1f} sec ({sec/60:.1f} min) for {full_criteo:,} samples")
PY

