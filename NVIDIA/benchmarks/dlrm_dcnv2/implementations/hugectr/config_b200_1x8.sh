#!/bin/bash
# Single-node 8x B200 config for our cluster (hungry-hippo-fin-03-*).
# Adapted from config_GB200_2x4x6912.sh; same hyperparameters, but
# DGXNNODES=1 / DGXNGPU=8 to match a single 8-GPU NVIDIA B200 node.
# Use MAX_ITER=100 for short perf runs on subsampled / synthetic data.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params
export RUN_SCRIPT="train.py"
export BATCHSIZE=55296
export BATCHSIZE_EVAL=1048576
export LEARNING_RATE=0.004
export USE_MIXED_PRECISION=true
export SCALER=16348
export SHARDING_PLAN=auto
export MEM_COMM_BW_RATIO=9
export GEN_LOSS_SUMMARY=true
# eval thresholds disabled for short perf runs (set to 0 to skip)
export MINIMUM_TRAINING_TIME=0
export DP_SHARDING_THRESHOLD=0.008
# ensure short max_iter is honored; set EVAL_INTERVAL high so eval is skipped
export MAX_ITER=100
export DISPLAY_INTERVAL=10
export EVAL_INTERVAL=200000

## System run params
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
