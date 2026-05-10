#!/bin/bash
# Single-node 1x8 B200 config with all known performance tunings applied.
# Differences vs config_b200_1x8.sh:
#   - SHARDING_PLAN=hier_auto      (better for NUMA-multi-domain layout)
#   - USE_ALGORITHM_SEARCH=false   (skip cuBLAS algo search; saves ~1s of first-iter)
#   - MAX_ITER=2000                (longer steady-state averaging)
#   - DISPLAY_INTERVAL=100

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params
export RUN_SCRIPT="train.py"
export BATCHSIZE=55296
export BATCHSIZE_EVAL=1048576
export LEARNING_RATE=0.004
export USE_MIXED_PRECISION=true
export SCALER=16348
export SHARDING_PLAN=uniform
export MEM_COMM_BW_RATIO=9
export GEN_LOSS_SUMMARY=true
export MINIMUM_TRAINING_TIME=0
export DP_SHARDING_THRESHOLD=0.008
export USE_ALGORITHM_SEARCH=false
export MAX_ITER=2000
export DISPLAY_INTERVAL=100
export EVAL_INTERVAL=2000000

## System run params
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=b200_1x8_uniform
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
