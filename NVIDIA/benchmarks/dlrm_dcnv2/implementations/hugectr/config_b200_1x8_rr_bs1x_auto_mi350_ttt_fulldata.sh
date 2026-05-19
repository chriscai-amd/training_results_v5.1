#!/bin/bash
# TTT (time-to-train) configuration: full Criteo 1TB dataset, mi350 settings.
#
# Inherits ALL stable hyperparameters (SCALER=16348, MEM_COMM_BW_RATIO=9,
# CUDA_DEVICE_MAX_CONNECTIONS=64, USE_ALGORITHM_SEARCH=false, etc.) from the
# config_b200_1x8_rr_bs1x_auto_mi350_quick.sh used for the perf baseline --
# the only differences from the perf config are:
#   - MAX_ITER bumped from 8000 -> 80000 (1.05x epoch on the full 4.2 B
#     train samples, batch 55296)
#   - EVAL_INTERVAL dropped from 2000000 -> 2000 (so we get ~40 eval AUC
#     readings during the run)
#   - AUC_THRESHOLD raised to 0.95 (effectively disabled) so training
#     doesn't auto-stop when the MLPerf 0.80275 target is reached -- we
#     want the full learning curve to see whether AUC saturates or keeps
#     improving with more data.
source $(dirname ${BASH_SOURCE[0]})/config_b200_1x8_rr_bs1x_auto_mi350_quick.sh

# Overrides
export MAX_ITER=80000        # 1.05x epoch (epoch = 4,195,197,692 / 55296 = 75866 iter)
export EVAL_INTERVAL=2000    # ~40 eval points across the epoch
export AUC_THRESHOLD=0.95    # don't auto-stop at MLPerf 0.80275 target
