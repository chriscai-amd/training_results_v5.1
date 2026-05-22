#!/usr/bin/env bash
# Phase 20.10 (Path A+D, 2026-05-20): apply rocprofv3 cold-pass kernel
# names onto all per-rank HCTR native traces, producing 8 overlaid
# Perfetto JSON files.
#
# This is the third step of the cold/hot/overlay workflow:
#
#   STEP 1 (one-shot, slow, ~+370% overhead) -- cold pass:
#     bash scripts/trace_and_convert.sh <jobid> <start> <end> cold
#     # produces /home/chcai/rps_out/trace_cold/iter_${start}_${end}_*.csv
#
#   STEP 2 (every production run, ~+3-30% overhead) -- native trace:
#     HCTR_NATIVE_TRACE=1 HCTR_NATIVE_TRACE_DETAIL=2 \
#         HCTR_NATIVE_TRACE_DIR=/home/chcai/native_out \
#         bash scripts/_native_trace_inner.sh  # or run_b200_match.sh
#     # produces /home/chcai/native_out/hctr_native_trace_rank{0..7}.json
#
#   STEP 3 (this script) -- overlay:
#     bash scripts/native_overlay_all_ranks.sh \
#         /home/chcai/native_out \
#         /home/chcai/rps_out/trace_cold/iter_${start}_${end} \
#         [cold_iter] [n_gpus]
#     # produces /home/chcai/native_out/overlay_rank{0..7}.json
#
# Args:
#   $1: native trace dir (contains hctr_native_trace_rank${R}.json)
#   $2: cold-pass rocprofv3 prefix (no _kernel_trace.csv suffix)
#   $3: cold iter to harvest mapping from (default: 5)
#   $4: number of GPUs / ranks to overlay (default: 8)
#
# Notes:
#   * The cold pass must have been captured with HCTR_ROCTX=1 so that
#     ROCTX marker ranges are present in marker_api_trace.csv. This
#     enables the joiner to bracket kernels by their owning HCTR phase.
#   * Path D in-tree resolution (register_phase_for_kernel) and Path A
#     (this script) are additive: Path D's args.kernel values are
#     preserved; Path A only fills in missing fields.
#   * Output is overlay_rank{R}.json. By default the joiner explodes the
#     captured [graph] network HIP-graph bucket into per-kernel slices
#     (tid=graph/kernels) using cold-pass kernel names. Pass
#     --no-explode-graph to native_trace_overlay.py to keep one aggregate.
set -eu

NATIVE_DIR=${1:?"usage: $0 <native_dir> <cold_prefix_or_-> [cold_iter] [n_gpus]"}
COLD_PREFIX=${2:?"cold-pass prefix required (or '-' if using LOAD_MAP env)"}
COLD_ITER=${3:-5}
NGPU=${4:-8}

# Phase 21.1 (2026-05-21): LOAD_MAP=/path/to/phase_kernel_map.json bypasses
# the cold-pass scan and reuses a previously saved map (typically from an
# EXHAUSTIVE run -- see native_trace.sh). When set, COLD_PREFIX may be "-".
LOAD_MAP=${LOAD_MAP:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOINER="$SCRIPT_DIR/native_trace_overlay.py"

if [ ! -d "$NATIVE_DIR" ]; then
    echo "ERROR: native_dir '$NATIVE_DIR' does not exist" >&2
    exit 1
fi
if [ ! -f "$JOINER" ]; then
    echo "ERROR: joiner script not found at $JOINER" >&2
    exit 1
fi
if [ -n "$LOAD_MAP" ]; then
    if [ ! -f "$LOAD_MAP" ]; then
        echo "ERROR: LOAD_MAP=$LOAD_MAP does not exist" >&2
        exit 1
    fi
    echo "===== LOAD_MAP set: reusing saved map, skipping cold-pass scan ====="
    echo "       map        = $LOAD_MAP"
else
    if [ ! -f "${COLD_PREFIX}_kernel_trace.csv" ]; then
        echo "ERROR: cold-pass kernel trace not found: ${COLD_PREFIX}_kernel_trace.csv" >&2
        echo "       Did you run 'trace_and_convert.sh ... cold' with HCTR_ROCTX=1?" >&2
        echo "       (Or pass LOAD_MAP=/path/to/phase_kernel_map.json to reuse" >&2
        echo "        a saved map from an EXHAUSTIVE run.)" >&2
        exit 1
    fi
    if [ ! -f "${COLD_PREFIX}_marker_api_trace.csv" ]; then
        echo "WARNING: ${COLD_PREFIX}_marker_api_trace.csv missing." >&2
        echo "         The cold pass should run with HCTR_ROCTX=1 so ROCTX phase" >&2
        echo "         ranges are recorded. Without them the joiner cannot bracket" >&2
        echo "         kernels to phase names and the overlay will be empty." >&2
    fi
fi

# Cache the phase->kernel mapping once (it's the same for every rank
# because rank 0..7 all run the same code path with the same hipGraphs).
# When LOAD_MAP is set, all ranks read that file and the cold-pass scan is
# skipped entirely. Otherwise the joiner derives the map from the cold
# pass on rank 0 and the same prefix is re-scanned for each rank (cheap;
# per-rank application of the map is the slow part).
MAP_JSON="$NATIVE_DIR/phase_kernel_map.json"

build_src_args() {
    # Echo the --cold-prefix/--gpu/--cold-iter or --load-map arg block.
    local rank=$1
    if [ -n "$LOAD_MAP" ]; then
        printf -- "--load-map %s" "$LOAD_MAP"
    else
        printf -- "--cold-prefix %s --gpu %s --cold-iter %s" \
            "$COLD_PREFIX" "$rank" "$COLD_ITER"
    fi
}

if [ -n "$LOAD_MAP" ]; then
    echo "===== rank 0 overlay (using saved map) ====="
    python3 "$JOINER" \
        --native-json "$NATIVE_DIR/hctr_native_trace_rank0.json" \
        $(build_src_args 0) \
        --out "$NATIVE_DIR/overlay_rank0.json"
else
    echo "===== building phase->kernel map from cold pass (gpu 0, iter $COLD_ITER) ====="
    python3 "$JOINER" \
        --native-json "$NATIVE_DIR/hctr_native_trace_rank0.json" \
        $(build_src_args 0) \
        --out "$NATIVE_DIR/overlay_rank0.json" \
        --dump-map "$MAP_JSON"
fi

for R in $(seq 1 $((NGPU - 1))); do
    NJ="$NATIVE_DIR/hctr_native_trace_rank${R}.json"
    OJ="$NATIVE_DIR/overlay_rank${R}.json"
    if [ ! -f "$NJ" ]; then
        echo "  rank $R: native JSON missing ($NJ); skipping" >&2
        continue
    fi
    echo "===== rank $R overlay ====="
    python3 "$JOINER" \
        --native-json "$NJ" \
        $(build_src_args "$R") \
        --out "$OJ"
done

echo
echo "===== overlay DONE ====="
ls -lh "$NATIVE_DIR"/overlay_rank*.json 2>/dev/null || true

# Phase 21.0 (2026-05-21): auto-merge 8 overlay files into one 8-GPU trace
# and drop distorted first/last iters (hipEvent staleness + end-of-window).
# Override trim with TRIM_EDGES env var; set TRIM_EDGES=0 to disable.
TRIM_EDGES=${TRIM_EDGES:-1}
echo
echo "===== merging overlay ranks (trim-edges=$TRIM_EDGES) ====="
python3 "$SCRIPT_DIR/merge_perfetto_traces.py" "$NATIVE_DIR" --trim-edges "$TRIM_EDGES"

echo
echo "Drag $NATIVE_DIR/overlay_all8gpus_trimmed.json into ui.perfetto.dev to view."
echo "(per-rank overlay_rank*.json still available for single-rank inspection)"
