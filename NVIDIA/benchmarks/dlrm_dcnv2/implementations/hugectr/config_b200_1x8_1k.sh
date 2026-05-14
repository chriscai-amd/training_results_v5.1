#!/bin/bash
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
export RUN_SCRIPT="train.py"
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
export MAX_ITER=1000
export DISPLAY_INTERVAL=50
export EVAL_INTERVAL=2000000
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
