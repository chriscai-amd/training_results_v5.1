#!/bin/bash
# Run one HCTR trial on a given slurm job + extra env vars.
# Usage: run_one_trial.sh <jobid> <bs> "<extra_env_string>" <max_iter>
# Outputs single line: "  steady=X.XXX M sps, per_iter=X.XX ms, loss=X.XXXXX"
set -u

JOBID=${1:?jobid}
BS=${2:?batchsize}
EXTRA=${3:-}
MAXITER=${4:-200}
LABEL=${5:-noname}

EXTRA_ENV=""
for kv in $EXTRA; do EXTRA_ENV="$EXTRA_ENV -e $kv"; done

out=$(srun --jobid=$JOBID --overlap -N1 -n1 -c 32 bash -c "
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host \
    -v /home/chcai/hugectr_rocm_port:/workspace \
    -v /dev/shm/criteo:/criteo \
    -e HCTR_NGPU=8 -e HCTR_BATCH=$BS -e HCTR_EVAL_BATCH=131072 \
    -e HCTR_EV_SIZE=128 -e HCTR_LR=0.004 -e HCTR_MAX_ITER=$MAXITER -e HCTR_DISPLAY=25 \
    -e HCTR_MEM_CAP=200 \
    -e HCTR_PRECISION_FLAGS='--use_mixed_precision --scaler 16348' \
    -e HCTR_SHARDING_PLAN=auto -e OMP_NUM_THREADS=8 \
    -e HCTR_USE_MULTI_HOT=1 -e HCTR_USE_MLPERF_CRITEO=1 \
    -e HCTR_USE_CUDA_GRAPH=1 \
    -e HCTR_DP_SHARD_THRESH=0.008 -e HCTR_MEM_COMM_WORK_RATIO=5 \
    -e HCTR_READER_THREADS=1 \
    $EXTRA_ENV \
    -w /workspace \
    rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424 \
    bash /workspace/run_b200_match.sh 2>&1")
steady=$(echo "$out" | grep -oP 'Steady-state throughput.*?\K[0-9.]+(?= M)')
per_iter=$(echo "$out" | grep -oP 'Per-iter time \(steady\).*?\K[0-9.]+(?= ms)')
loss=$(echo "$out" | grep -oP 'Final loss \(BCE\)\s*: \K[0-9.]+')
echo "[job=$JOBID bs=$BS $LABEL] steady=${steady:-N/A} M sps, per_iter=${per_iter:-N/A} ms, loss=${loss:-N/A}"
