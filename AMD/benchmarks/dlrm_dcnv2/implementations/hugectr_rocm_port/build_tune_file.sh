#!/bin/bash
# Auto-generate HIPBLASLT tuning file for our bs4x MLP shapes.
set -u

mkdir -p /workspace/hbl_tuning
TUNE=/workspace/hbl_tuning/bs4x_tune.txt
rm -f "$TUNE"

run_one() {
    local m=$1 n=$2 k=$3 transA=$4 transB=$5 act=$6 beta=$7
    HIPBLASLT_TUNING_FILE="$TUNE" hipblaslt-bench \
        -m "$m" -n "$n" -k "$k" \
        --transA "$transA" --transB "$transB" \
        --precision f16_r --compute_type f32_r \
        --bias_vector --bias_type f16_r --activation_type "$act" \
        --beta "$beta" \
        --algo_method all --requested_solution -1 \
        --iters 100 --cold_iters 20 --rotating 512 2>&1 | tail -1
}

echo "=== generating tuning entries (12 unique fwd/bwd shapes at bs4x) ==="
# fwd: D = act(A^T @ B + bias),  beta=0
run_one 27648 1024 3456 T N relu 0 ; echo "  top_L1_fwd ok"
run_one 27648 1024 1024 T N relu 0 ; echo "  top_L2_fwd ok"
run_one 27648  512 1024 T N relu 0 ; echo "  top_L3_fwd ok"
run_one 27648  256  512 T N relu 0 ; echo "  top_L4_fwd ok"
run_one 27648    1  256 T N none 0 ; echo "  top_L5_fwd ok"
run_one 27648  512   13 T N relu 0 ; echo "  bot_L1_fwd ok"
run_one 27648  256  512 T N relu 0 ; echo "  bot_L2_fwd ok"
run_one 27648  128  256 T N relu 0 ; echo "  bot_L3_fwd ok"
# wgrad: dW = X^T @ dY, beta=1 (grad accumulation)
run_one  3456 1024 27648 T N none 1 ; echo "  top_L1_wgrad ok"
run_one  1024 1024 27648 T N none 1 ; echo "  top_L2_wgrad ok"
run_one    13  512 27648 T N none 1 ; echo "  bot_L1_wgrad ok"
# dgrad: dX = dY @ W, beta=1
run_one 27648 3456 1024 N N none 1 ; echo "  top_L1_dgrad ok"

echo
echo "=== tuning file contents ==="
ls -la "$TUNE"
echo "lines:"
wc -l "$TUNE"
echo "head:"
head -10 "$TUNE"
