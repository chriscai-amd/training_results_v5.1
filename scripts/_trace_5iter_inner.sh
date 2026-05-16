#!/usr/bin/env bash
# Capture a 5-iter steady-state trace at ~iter 1000 with rocprofv3 --kernel-trace.
# Uses -P (collection-period) to trace only a 50ms window, near-zero overhead
# outside the window. With trace-overhead ~10ms/iter at bs=1x, 50ms = ~5 iters.
set -eu

echo "=== stage data ==="
mkdir -p /ramdata/mlperf
cp /nfs_data/train_data.bin /ramdata/mlperf/ &
cp /nfs_data/val_data.bin /ramdata/mlperf/ &
wait
ln -sf /ramdata /criteo

echo "=== run with rocprofv3 --kernel-trace (15 iters total, pick steady from logs) ==="
mkdir -p /rps_out/trace_15iter
chmod 777 /rps_out/trace_15iter
cd /workspace/runtime_test/nvidia_frontend
# Override HCTR_MAX_ITER to 15 for short trace run; iters 5-10 are steady state.
# The -P collection-period flag was tried but rocprofv3 produced no output
# even though the window timing was correct — buffer flush bug w/ kernel-trace.
export HCTR_MAX_ITER=15
export HCTR_DISPLAY=5
# IMPORTANT: --memory-copy-trace is a SEPARATE flag in rocprofv3 (it's not
# included in --kernel-trace or --hip-trace). Without it, host-to-device
# DMA copies (the data reader's placement_streams_ traffic = NV's "memcpy
# lane") are invisible to the trace, leading to misleading conclusions that
# AMD lacks a memcpy lane. Baked in here so we never miss it again.
HCTR_PROFILE_PREFIX="rocprofv3 --kernel-trace --hip-trace --memory-copy-trace --output-format csv -d /rps_out/trace_15iter --output-file iter5_10 --" \
    bash /workspace/run_b200_match.sh
