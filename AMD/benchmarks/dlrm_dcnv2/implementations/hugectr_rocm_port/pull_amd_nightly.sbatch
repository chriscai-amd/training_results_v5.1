#!/bin/bash
#SBATCH --job-name=docker_pull
#SBATCH --partition=meta64
#SBATCH --nodelist=cv350-rck-g03-c07-18
#SBATCH --time=2:00:00
#SBATCH --output=/tmp/docker_pull_sbatch.log

set -x
date
docker pull rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424
echo "=== docker images after pull ==="
docker images
date
