# DLRM-DCNv2 on a single-node 8 × B200 cluster

This document describes how to reproduce a NVIDIA Merlin HugeCTR DLRM-DCNv2
run on a single-node 8 × B200 (SM 100) Slurm cluster that uses plain
`srun + docker` instead of Pyxis/Enroot, only allows HTTPS egress, and shares
home over NFS. It also reports the kernel-level perf breakdown we measured.

The path mirrors the published MLPerf Training v5.1 submission
(`tyche_ngpu8_ngc25.03_hugectr` for GB200 and `5.1-0040` for 8 × B200) but is
adapted for our cluster topology and constraints.

## Conventions used in commands

Commands below use shell-style placeholders:

```
$REPO_ROOT      checkout root of this repository
$DATA_ROOT      writable directory with ≥ 1 TB free for raw/processed Criteo data
<gpu-node>      Slurm hostname of an 8 × B200 worker node
<your-reservation>  active Slurm reservation owning <gpu-node>
```

Set them in your shell, e.g. `export REPO_ROOT=$(git rev-parse --show-toplevel)
DATA_ROOT=/scratch/criteo`.

## 1. Hardware / OS assumptions

```
GPUs           8 × NVIDIA Blackwell B200 (SM 100)
Host CPU       AMD EPYC (240 cores)
RAM            ≥ 1 TB
NFS            home directory shared across nodes
Slurm          QOS=reservation-only, plain srun + docker (no Pyxis/Enroot)
Egress         HTTPS only (no plain HTTP through to archive.ubuntu.com)
```

## 2. Files in this branch

| File | Status | What it does |
| ---- | ------ | ------------ |
| `config_b200_1x8.sh` | new | Single-node 1 × 8 B200 config; clone of `config_GB200_2x4x6912.sh` with `DGXNNODES=1 DGXNGPU=8`, `MAX_ITER=100`, `EVAL_INTERVAL=200000`. |
| `run_b200.sh` | new | `srun + docker run + mpirun -n 1` launcher. Bypasses Pyxis/Enroot. Optional `--image-tar` to load the docker image from NFS, and `--nsys-trace` / `--nsys-delay` / `--nsys-duration` for profiling. |
| `train_nsys.py` | new | Wrapper around `train.py` injecting a `ProfilerWindowCallback` (timer-driven `cudaProfilerStart`/`Stop`) for `--capture-range=cudaProfilerApi` traces. |
| `Dockerfile` | patched | Adds an apt `http://` → `https://` rewrite before `apt-get update` (cluster only allows HTTPS to `archive.ubuntu.com`). |
| `requirements.txt` | patched | Bumps `mpi4py` from `3.1.5` to `>=4.0.0` (3.1.5 is incompatible with the setuptools shipped in the `nvcr.io/nvidia/pytorch:25.03-py3` base image). |
| `.gitignore` | new | Local ignore for `**/__pycache__/`. |

## 3. Build the docker image (~3 min on a 240-core EPYC + buildkit cache)

```bash
SLURM_TIMELIMIT=02:00:00 srun \
    --reservation=<your-reservation> --nodelist=<gpu-node> \
    --chdir=$(pwd) \
    bash -c 'docker build --network=host -t mlperf-nvidia:recommendation-hugectr .'
```

`--network=host` is required so the docker build sandbox can reach the apt
mirror via the host network (the default bridge fails on this cluster).

Save the image as a portable tarball on NFS so other nodes can `docker load`
without rebuilding:

```bash
docker save mlperf-nvidia:recommendation-hugectr \
    | zstd -3 -T0 -o $DATA_ROOT/docker_images/mlperf-nvidia-recommendation-hugectr.tar.zst
# ~11 GB compressed (~37 GB uncompressed)
```

`run_b200.sh --image-tar <path>` will `zstd -dc | docker load` automatically
on any reservation node that doesn't already have the image.

## 4. Dataset prep

The reference Criteo 1 TB Click Logs is gated. Two practical sources:

1. **Hugging Face mirror** (open, ~36 GB compressed, ~10 % subsample of original):
   <https://huggingface.co/datasets/criteo/CriteoClickLogs>
2. **Original Criteo download** (`https://ailab.criteo.com/...`) — gated, full
   1 TB.

If using the HF mirror, note that `day_0.gz` and `day_1.gz` were deleted from
`main` on 2026-01-15. The LFS objects are still reachable via the
pre-deletion commit:

```python
import os
from huggingface_hub import snapshot_download, hf_hub_download
TARGET = os.path.join(os.environ["DATA_ROOT"], "criteo_1tb_raw_input_dataset_dir")
PRE_DELETE = "a1a472783e135c6786617567f51bb64bfc4b031a"
snapshot_download(
    "criteo/CriteoClickLogs", repo_type="dataset",
    local_dir=TARGET, max_workers=8,
)
for f in ("day_0.gz", "day_1.gz"):
    hf_hub_download(
        "criteo/CriteoClickLogs", filename=f, repo_type="dataset",
        revision=PRE_DELETE, local_dir=TARGET,
    )
```

The HF mirror's gzip streams are missing the trailer (`gzip -t` fails). Decompress
with `gzip -dc` (which still emits all the data) and trim the partial last
line of each `day_N` file:

```python
import os
RAW = os.path.join(os.environ["DATA_ROOT"], "criteo_1tb_raw_input_dataset_dir")
for i in range(24):
    f = os.path.join(RAW, f"day_{i}")
    sz = os.path.getsize(f)
    with open(f, "rb") as h:
        h.seek(max(0, sz - (1 << 20)))
        tail = h.read()
    last_nl = tail.rfind(b"\n")
    if last_nl != -1:
        os.truncate(f, sz - len(tail) + last_nl + 1)
```

Then run the standard MLPerf pipeline inside the docker image (same image as
training; Step 4-5 don't need HugeCTR but the container has all required
deps):

```bash
# Step 1-3 (TSV → npy → contiguous → shuffle)
docker run --rm --runtime=nvidia --gpus all --network=host --ipc=host --shm-size=64g \
    -v $REPO_ROOT/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr:/workspace/hugectr \
    -v $DATA_ROOT/criteo_1tb_raw_input_dataset_dir:/data/raw \
    -v $DATA_ROOT/criteo_1tb_temp_intermediate_files_dir:/data/temp \
    -v $DATA_ROOT/criteo_1tb_numpy_contiguous_shuffled_output_dataset_dir:/data/processed \
    -w /workspace/hugectr nvcr.io/nvidia/pytorch:26.01-py3 \
    bash -c 'pip install fbgemm-gpu-nightly torchrec pyarrow tqdm typing-inspect tensordict iopath pyre-extensions
             bash scripts/process_Criteo_1TB_Click_Logs_dataset.sh /data/raw /data/temp /data/processed'

# Step 4 (multi-hot npz)
docker run ... -v $DATA_ROOT/criteo_1tb_sparse_multi_hot:/data/multi_hot ... \
    python scripts/materialize_synthetic_multihot_dataset.py \
        --in_memory_binary_criteo_path /data/processed \
        --output_path /data/multi_hot \
        --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
        --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
        --multi_hot_distribution_type uniform

# Step 5 (raw .bin format for HugeCTR)
docker run ... -v $DATA_ROOT/criteo_1tb_multihot_raw:/data/raw ... \
    python scripts/convert_to_raw.py \
        --input_dir_labels_and_dense /data/processed \
        --input_dir_sparse_multihot /data/multi_hot \
        --output_dir /data/raw --stages train val test
```

Final outputs (sizes shown for the HF subsample):

```
$DATA_ROOT/criteo_1tb_multihot_raw/train_data.bin   ≈ 431 GB  (~748 M rows, days 0-22)
$DATA_ROOT/criteo_1tb_multihot_raw/val_data.bin     ≈  19 GB  (~32 M rows, day 23)
$DATA_ROOT/criteo_1tb_multihot_raw/test_data.bin    =   0 B   (split point > subsample)
```

Total prep took ~3.5 h on a 1 × 8 B200 + EPYC + 1 TB+ RAM node:

```
Step 1 (npy_preproc_criteo)            ~1 h 50 min
Step 2 (contiguous_preproc_criteo)     ~1 h 20 min
Step 3 (shuffle_preproc_criteo)        ~10 min
Step 4 (materialize_synthetic_multihot) ~10 min
Step 5 (convert_to_raw)                ~15 min
```

## 5. Run training

```bash
bash run_b200.sh \
    --reservation <your-reservation> \
    --nodelist <gpu-node> \
    --config config_b200_1x8.sh \
    --image mlperf-nvidia:recommendation-hugectr \
    --image-tar $DATA_ROOT/docker_images/mlperf-nvidia-recommendation-hugectr.tar.zst \
    --train-data $DATA_ROOT/criteo_1tb_multihot_raw/train_data.bin \
    --val-data   $DATA_ROOT/criteo_1tb_multihot_raw/val_data.bin \
    --logdir     $DATA_ROOT/criteo_synth/results \
    --time       01:00:00
```

Add `--nsys-trace <name>` to capture an nsys profile under
`<logdir>/<name>.nsys-rep`. The default `--nsys-delay 30 --nsys-duration 5`
captures a 5 s window starting 30 s after process spawn (covers init through
~iter 50 on this hardware).

## 6. What the run does

| Setting | Value |
| ------- | ----- |
| Global batch size | 55 296 |
| Eval batch size | 1 048 576 |
| Learning rate | 0.004 (constant) |
| Optimizer | adagrad (eps=1e-8, init_accu=0) |
| Precision | mixed (BF16 compute, FP32 master weights) |
| Loss scaler | 16 348 |
| Embedding dim | 128 |
| Embedding tables | 26 (3 × 40 M tables capped) |
| Bottom MLP | 13 → 512 → 256 → 128 |
| Top MLP | (concat) → 1024 → 1024 → 512 → 256 → 1 |
| Cross network | 3 layers, projection_dim = 512 |
| Sharding plan | `auto` |
| DP threshold | 0.008 GiB |
| `mem/comm bw ratio` | 9 |
| `MAX_ITER` | 100 (perf only — remove for time-to-AUC convergence) |

## 7. Performance results (1 × 8 B200, HF 10 % subsample)

### 7.1 Throughput

| Run | Wall (100 iters) | Throughput | Per-iter (avg) |
| --- | ---------------: | ---------: | -------------: |
| Unprofiled | 1.71 s | **3.24 M samples/s** | 17.1 ms |
| `nsys --cuda-graph-trace=node` | 1.83 s | 3.02 M samples/s | 18.3 ms |
| `nsys` default                  | 1.97 s | 2.80 M samples/s | 19.7 ms |

The first iter takes ~1.0 s for cuBLAS algorithm search + cuda-graph
instantiation. Iters 11+ run at 3–6 ms each, so steady-state throughput is
~9 M samples/s on 8 × B200, and the 100-iter average is dragged down by
the compile-heavy first iter.

### 7.2 Compute vs comm breakdown (per-GPU avg, training-window-only)

The training window is auto-detected via NCCL kernel density (200 ms bins
with ≥100 NCCL events) — see `breakdown_nsys.py`.

```
metric                  avg/GPU (ms)    sum 8 GPUs (ms)
────────────────────────────────────────────────────────
compute (busy)               196.88          1575.04          (86.1 % of wall)
comm total                    85.36           682.92
  exposed                     31.79           254.31          (13.9 % of wall)
  hidden                      53.58           428.60          (62.8 % of comm hidden)
wall (any-kernel)            228.67          1829.36
```

### 7.3 Top kernels in the training window (training-only, all 8 GPUs)

```
%      kernel                                       inst   total_ms   role
─────────────────────────────────────────────────────────────────────────
16.4%  ncclDevKernel_SendRecv                       3152     488      embedding all-to-all
 6.5%  ncclDevKernel_AllReduce_Sum_f16              800      195      DDP grad sync
 6.1%  embedding update4_kernel (Adam)              1600     181      sparse opt
 4.8%  cub::DeviceRadixSortOnesweep                 7120     144      sparse-index sort
 4.1%  HugeCTR vector_mul_fma3_align (fp16)         2376     123      fused FMA
 4.1%  cutlass3x_sm100_s128x256_bgrada (BF16 BWD)   2376     122      MLP backward
 3.5%  HugeCTR label_and_count_keys                  784     103      sparse prep (KJT)
 3.4%  embedding multi_to_one_reduce_vec4_v2         800     102      sparse fwd reduction
 3.0%  nvjet_hsh_128x192_64x7 (cuBLAS GEMM)         2376      88      MLP forward
 2.5%  ada_grad_update4_kernel                       800      73      dense optimizer
 2.3%  embedding multi_to_one_warp_per_ev (fp32)     792      70      sparse fwd
 2.1%  cutlass_80_s16816gemm_drelu (mixed BWD)      1584      62      MLP backward
 2.1%  HugeCTR vector_fma4_align8                   2376      61      fused FMA
 1.8%  nvjet_hsh_64x192_64x8 (cuBLAS GEMM)           891      54      MLP fwd
 1.6%  HugeCTR concat_bwd_kernel                    1584      47      interaction layer bwd
```

### 7.4 Aggregate buckets (% of GPU time)

```
NCCL                       ~23 %   (16.4 + 6.5)
embedding ops              ~17 %   (update4, multi_to_one, label_and_count)
sparse infra (sort, cub)    ~6 %   (radix sort + scan)
MLP fwd/bwd GEMMs          ~14 %   (cutlass_s128x256, nvjet_hsh, drelu)
elementwise FMA / fused    ~10 %   (vector_mul/fma)
optimizer (adagrad)         ~3 %
other (long tail)          ~27 %
```

DLRM-DCNv2 is **embedding/comm-bound, not compute-bound** on B200. GEMMs are
only ~14 % of GPU time; embedding lookup + sort + comm dominate.

## 8. Comparison vs published MLPerf v5.1 numbers

Reference: <https://developer.nvidia.com/deep-learning-performance-training-inference/training>

```
System          GPUs       MLPerf-ID  TTT (min)  TTT (sec)  est. throughput
────────────────────────────────────────────────────────────────────────────
G894-AD1        8 × B200   5.1-0040     2.3        138       ~22.8 M samples/s
Tyche           8 × GB200  5.1-0066     2.2        132       ~23.9 M samples/s
SRS-GB200-NVL72 64×GB200   5.0-0087    0.7         42       ~75   M samples/s
ours (this run) 8 × B200   —            —          —          3.24 M samples/s avg
                                                              ~9    M samples/s steady-state
```

We're **~39 % of the reference 8 × B200 throughput** in steady state. Likely
sources of the gap:

1. **Cold cuBLAS algorithm search** — first iter (~1.0 s of 1.71 s total) is
   dominated by autotune; 100-iter measurement is too short to amortize.
2. **10 % subsampled data** — embedding access patterns differ; the `auto`
   sharding planner is calibrated for full Criteo vocabulary distribution.
3. **No NCCL-NVLS / SHARP tuning** — `NCCL_NVLS_ENABLE=1` is set but the
   plugin stack hasn't been verified on this cluster.
4. **MLPerf submitters tune `BATCHSIZE`, `SCALER`, `SHARDING_PLAN`,
   `DP_SHARDING_THRESHOLD`, `LR`** for the specific hardware layout; defaults
   in `config_GB200_2x4x6912.sh` are GB200-tuned.
5. **DGX-class NVLink-switch tuning** — `nvidia-smi topo -m` may reveal
   topology differences vs G894-AD1 (the reference 8 × B200 system).

Path to closing the gap (in priority):

1. Run a longer measurement window (≥1000 iters) to amortize compile + algo
   search.
2. Try `SHARDING_PLAN=hier_auto`, sweep `DP_SHARDING_THRESHOLD`.
3. Verify SHARP / NVLS plugin status; sweep `NCCL_ALGO`, `NCCL_PROTO`.
4. Re-run on the **full** Criteo dataset for an apples-to-apples MLPerf
   comparison.

## 9. Profiling / debugging notes

- **`--cuda-graph-trace=node` is required** to see kernels that live inside
  HugeCTR's CUDA graphs (without it, the training-window kernels are
  invisible and only init-phase autotune kernels show up).
- **`--capture-range=cudaProfilerApi` is unreliable** in this combination —
  with HugeCTR's `use_cuda_graph=True`, we observed empty traces even with
  `cudaProfilerStart`/`Stop` correctly emitted by `train_nsys.py`. Time-based
  `--delay`/`--duration` is the reliable path.
- **NUMA `mbind`/`set_mempolicy` warnings** at startup are harmless — the
  container doesn't have `CAP_SYS_NICE`, but performance is not measurably
  affected.
- **Shallow clone OK** — this repo was cloned with
  `--depth=1 --filter=blob:none --no-checkout` and sparse-checkout limited to
  `NVIDIA/benchmarks/dlrm_dcnv2/`. Pushing new branches works fine; only the
  full history isn't available locally.
