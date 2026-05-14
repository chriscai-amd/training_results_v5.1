#!/usr/bin/env bash
# Convert all-GPU rocprofv3 trace bundle into per-GPU annotated Perfetto JSONs.
#
# Usage:
#   bash rocprofv3_convert_all_gpus.sh <trace_prefix> [first_iter] [n_iters] [out_dir]
#
# Example:
#   bash rocprofv3_convert_all_gpus.sh \
#       /home/chcai/trace_a2a_bs55296_5step/a2a_bs55296_5step \
#       15 3 \
#       /home/chcai/traces/HCTR_amd
set -e
PREFIX=${1:?"trace prefix required, e.g. /path/to/trace_xx"}
FIRST_ITER=${2:-15}
N_ITERS=${3:-3}
OUT_DIR=${4:-$(dirname "$PREFIX")/perfetto}
NUM_GPUS=${HCTR_NGPU:-8}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
CONVERTER=$SCRIPT_DIR/rocprofv3_to_perfetto_annotated.py

mkdir -p "$OUT_DIR"
echo "Converting $NUM_GPUS GPUs from $PREFIX, iters $FIRST_ITER..$((FIRST_ITER+N_ITERS-1))"
echo "Output dir: $OUT_DIR"
echo

PREFIX_BASE=$(basename "$PREFIX")
for g in $(seq 0 $((NUM_GPUS - 1))); do
    OUT="$OUT_DIR/${PREFIX_BASE}.gpu${g}.iter${FIRST_ITER}-$((FIRST_ITER + N_ITERS - 1)).json"
    echo "=== GPU $g -> $OUT ==="
    python3 "$CONVERTER" "$PREFIX" "$g" "$FIRST_ITER" "$N_ITERS" "$OUT" 2>&1 | sed 's/^/    /'
    echo
done

echo
echo "All conversions done. Files:"
ls -lh "$OUT_DIR"/*.json | head -20
