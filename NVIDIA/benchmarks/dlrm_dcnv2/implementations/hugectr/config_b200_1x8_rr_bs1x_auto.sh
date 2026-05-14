#!/bin/bash
# bs=1x (MLPerf spec) with SHARDING_PLAN=auto on tmpfs - apples-to-apples vs ref
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
export USE_ALGORITHM_SEARCH=false
export MAX_ITER=2000
export DISPLAY_INTERVAL=100
export EVAL_INTERVAL=2000000

export CUDA_DEVICE_MAX_CONNECTIONS=64
export HCTR_DEFAULT_CONCURRENCY=8

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=b200_1x8_rr_bs1x_auto
export WALLTIME_RUNANDTIME=30
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
