#!/usr/bin/env bash
# Phase 20.4 (2026-05-17): test HCTR native Perfetto emitter.
# Validates: (a) HCTR_NATIVE_TRACE=1 doesn't break training,
#            (b) iter wall stays within ~5% of production 4.05 ms,
#            (c) per-rank JSON file is produced and parseable.
set -eu

apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
/usr/bin/python3 -m pip install --break-system-packages --quiet mpi4py mlperf-logging 2>&1 | tail -1 || true

# Stage data into ramdata
echo "=== stage data ==="
mkdir -p /ramdata/mlperf
cp /nfs_data/train_data.bin /ramdata/mlperf/ &
cp /nfs_data/val_data.bin /ramdata/mlperf/ &
wait
ln -sf /ramdata /criteo

OUTDIR=/rps_out/native_trace_test
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"; chmod 777 "$OUTDIR"

cd /workspace/runtime_test/nvidia_frontend
export HCTR_MAX_ITER=25
export HCTR_DISPLAY=5
# Phase 20.7 fix: flush auto-fires at the LAST traced iter (no mid-trace
# blocking I/O). FLUSH_EVERY is now off-by-default; leave unset.
# Default: trace iters 5..10 (5-iter window after warmup)
export HCTR_NATIVE_TRACE_BEGIN="${HCTR_NATIVE_TRACE_BEGIN:-5}"
export HCTR_NATIVE_TRACE_END="${HCTR_NATIVE_TRACE_END:-10}"
# HCTR_NATIVE_TRACE_DEEP=1 disables graph capture for the network graph,
# giving per-network-segment timing (fwd/bmlp, fwd/tmlp, fwd/loss,
# bwd/tmlp, bwd/bmlp). Cost: ~30% iter-wall overhead.
echo "  HCTR_NATIVE_TRACE_BEGIN=$HCTR_NATIVE_TRACE_BEGIN"
echo "  HCTR_NATIVE_TRACE_END=$HCTR_NATIVE_TRACE_END"
echo "  HCTR_NATIVE_TRACE_DEEP=${HCTR_NATIVE_TRACE_DEEP:-0}"
export NCCL_SOCKET_IFNAME=lo
export HCTR_USE_MULTI_HOT=1 HCTR_USE_MLPERF_CRITEO=1
export HCTR_USE_CUDA_GRAPH="${HCTR_USE_CUDA_GRAPH:-1}"  # honor inbound env
export HCTR_USE_SUBSAMPLED_CRITEO=1 HCTR_USE_REAL_TABLE_SIZES=1
export HCTR_DP_SHARD_THRESH=0.008 HCTR_MEM_COMM_WORK_RATIO=9 HCTR_READER_THREADS=1
export HCTR_ASYNC_WGRAD=0 HCTR_NGPU=8 HCTR_BATCH=55296 HCTR_EVAL_BATCH=131072
export HCTR_EV_SIZE=128 HCTR_LR=0.004 HCTR_MEM_CAP=200
export HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348'
export HCTR_SHARDING_PLAN=auto OMP_NUM_THREADS=8
export HCTR_ROCTX=0  # native emitter is independent of ROCTX

# *** Native trace gate ***
# HCTR_NATIVE_TRACE_MODE env: 1=on, 0=off (sanity)
export HCTR_NATIVE_TRACE="${HCTR_NATIVE_TRACE_MODE:-1}"
export HCTR_NATIVE_TRACE_DIR="$OUTDIR"
export HCTR_NATIVE_TRACE_BASE="${HCTR_NATIVE_TRACE_BASE:-1}"  # record iter_base for cross-stream alignment

# Run WITHOUT rocprofv3 -- the whole point is no external profiler.
bash /workspace/run_b200_match.sh > "$OUTDIR/run_stdout.log" 2>&1

echo ""
echo "=== iter timings ==="
grep -aE 'Iter:|Steady|Per-iter' "$OUTDIR/run_stdout.log" | tail -10

echo ""
echo "=== HCTR_NATIVE_TRACE output files ==="
ls -lh "$OUTDIR"/hctr_native_trace_*.json 2>/dev/null || echo "(no JSON files produced!)"

echo ""
echo "=== quick JSON sanity check ==="
for f in "$OUTDIR"/hctr_native_trace_*.json; do
  [ -f "$f" ] || continue
  bytes=$(stat -c%s "$f")
  events=$(/usr/bin/python3 -c "import json,sys; d=json.load(open('$f')); print(len(d['traceEvents']))" 2>/dev/null || echo "PARSE_FAIL")
  echo "  $f: $bytes bytes, $events events"
done

echo ""
echo "=== unique phase names in rank 0 (first 30) ==="
/usr/bin/python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/hctr_native_trace_rank0.json'))
    names = sorted(set(e['name'] for e in d['traceEvents']))
    for n in names[:30]: print(' ', n)
    print('  total unique:', len(names))
except Exception as e:
    print('FAIL:', e)
"

echo ""
echo "=== iter durations from emitter ==="
/usr/bin/python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/hctr_native_trace_rank0.json'))
    iters = [e for e in d['traceEvents'] if e['cat']=='iter']
    print(f'  Found {len(iters)} iter events')
    for e in iters[-10:]:
        print(f'  iter={e[\"args\"][\"iter\"]}  dur={e[\"dur\"]:.1f} us')
except Exception as e:
    print('FAIL:', e)
"
