#!/bin/bash
# MI350X-aligned 4-stream layout attempt:
#   - keep train_intra/inter_iteration_overlap=True (gives the 3 EBC scheduler
#     side streams: emb "mp", emb "dp", sparse_prep -- exactly what MI350X has)
#   - async_wgrad=False (folds MLP wgrad into the main computation stream,
#     matching MI350X's "main carries mlp_fwd+bwd_dgrad+bwd_wgrad together")
# Expected: 4 HCTR-owned streams (main + mp + dp + sparse_prep) + 4 cuBLASLt
# internal helper streams (367/369/370/371) that we cannot eliminate from
# config alone -> 8 visible streams in the trace.
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

export RUN_SCRIPT="train_mi350.py"
export BATCHSIZE=55296
export BATCHSIZE_EVAL=1048576
export LEARNING_RATE=0.004
export USE_MIXED_PRECISION=true
export SCALER=16348
export SHARDING_PLAN=auto
export MEM_COMM_BW_RATIO=9
export GEN_LOSS_SUMMARY=true
export MINIMUM_TRAINING_TIME=0
export DP_SHARDING_THRESHOLD=0.008
export USE_ALGORITHM_SEARCH=false
export MAX_ITER=8000
export DISPLAY_INTERVAL=500
export EVAL_INTERVAL=2000000

export CUDA_DEVICE_MAX_CONNECTIONS=64
export HCTR_DEFAULT_CONCURRENCY=8

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=b200_1x8_rr_bs1x_auto_mi350
export WALLTIME_RUNANDTIME=300
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
