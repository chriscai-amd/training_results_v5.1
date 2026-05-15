#!/bin/bash
# Bench representative MLP GEMM shapes — heuristic vs all-algos
set -u

bench_one() {
    local m=$1 n=$2 k=$3 transA=$4 transB=$5
    hipblaslt-bench -m "$m" -n "$n" -k "$k" \
        --transA "$transA" --transB "$transB" \
        --precision f16_r --compute_type f32_r \
        --bias_vector --bias_type f16_r --activation_type relu \
        --algo_method "$6" --requested_solution "$7" \
        --iters 50 --cold_iters 10 2>&1 \
      | awk -F',' '/^[ \t]+[TN],/ { print $NF }'
}

echo "shape | M | N | K | trans | heuristic_us | best_us | speedup"
for spec in \
    "top_L1_fwd:6912:1024:3456:T:N" \
    "top_L2_fwd:6912:1024:1024:T:N" \
    "top_L3_fwd:6912:512:1024:T:N" \
    "top_L4_fwd:6912:256:512:T:N" \
    "top_L5_fwd:6912:1:256:T:N" \
    "bot_L1_fwd:6912:512:13:T:N" \
    "bot_L2_fwd:6912:256:512:T:N" \
    "bot_L3_fwd:6912:128:256:T:N" \
    "top_L1_wgrad:3456:1024:6912:T:N" \
    "bot_L1_wgrad:13:512:6912:T:N" \
    ; do
    name=$(echo "$spec" | cut -d: -f1)
    m=$(echo "$spec" | cut -d: -f2)
    n=$(echo "$spec" | cut -d: -f3)
    k=$(echo "$spec" | cut -d: -f4)
    transA=$(echo "$spec" | cut -d: -f5)
    transB=$(echo "$spec" | cut -d: -f6)
    heur=$(bench_one "$m" "$n" "$k" "$transA" "$transB" heuristic 1 | head -1)
    best=$(bench_one "$m" "$n" "$k" "$transA" "$transB" all -1 | sort -n | head -1)
    if [ -n "$heur" ] && [ -n "$best" ]; then
        speedup=$(python3 -c "print(f'{float($heur)/float($best):.3f}')" 2>/dev/null || echo "?")
    else
        speedup=NA
    fi
    printf "  %-15s M=%5s N=%5s K=%5s  trans=%s%s  heur=%6s us  best=%6s us  speedup=%s\n" \
        "$name" "$m" "$n" "$k" "$transA" "$transB" "${heur:-?}" "${best:-?}" "$speedup"
done
