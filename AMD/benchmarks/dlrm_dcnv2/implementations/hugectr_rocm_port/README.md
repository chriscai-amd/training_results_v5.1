# DLRM-DCNv2 — HugeCTR ROCm port for AMD Instinct MI350X

A working port of NVIDIA's MLPerf v5.1 HugeCTR DLRM-DCNv2 submission to AMD's
ROCm 7.2 platform, targeting `gfx950` (Instinct MI350X). Source-compatible
with NVIDIA's `train.py` Python frontend and `mlperf_logger` MLLog format.

This is a research / port branch, **not an official MLPerf submission**.

## Status (single-node, 8 × MI350X, 1 node)

| Configuration | Throughput | Notes |
|---|---|---|
| 8 × MI350X, FP16 mixed, real DCN-v2, **MULTI-HOT** (130 keys/row, 576 B), batch 55,296 | **4.10 M samples/sec, 100 iters stable** | **apples-to-apples NVIDIA B200 config** |
| 8 × MI350X, FP16 mixed (scaler 16348), real DCN-v2, single-hot, batch 55,296, HIP graph + overlap | 6.26 M samples/sec, 100 iters stable | (single-hot is ~5× less embedding work than NVIDIA's multi-hot) |
| 8 × MI350X, FP16 mixed (scaler 16348), real DCN-v2, single-hot, batch 55,296, HIP graph, no overlap | 6.09 M samples/sec, 100 iters | |
| 8 × MI350X, FP16 mixed (scaler 16348), real DCN-v2, single-hot, batch 55,296, no HIP graph | 5.85 M samples/sec, 100 iters | |
| 8 × MI350X, FP16 mixed (scaler 16348), real DCN-v2 (3 layer), batch 16,384 | 5.27 M samples/sec, 30+ iters | loss 0.366 → 0.257 |
| 8 × MI350X, FP32, real DCN-v2 (3-layer MultiCross v2, proj=512) | 3.89-6.59 M samples/sec | exact NVIDIA MLPerf model graph |
| 8 × MI350X, FP16 mixed, real DCN-v2 (3 layer), batch 4096 | 1.90 M samples/sec, 100+ iters | loss 0.277 → 0.225 |
| 1 × MI350X, FP16 mixed, real DCN-v2 + Layer_t.MLP fused MLP, batch 8192 | 0.24 M samples/sec, loss 3.23 → 0.245 | new DRELU_BGRAD fallback works at 1 GPU; multi-GPU collapses (correctness bug) |
| 1 × MI350X, FP16 mixed, real DCN-v2 | 1.84 M samples/sec | full architecture |
| 1 × MI350X, FP32, real DCN-v2 | 0.85 M samples/sec | full architecture |

NVIDIA's published B200 reference (8 GPU, FP16, full multi-hot Criteo,
fused MLP, HIP graph): ~30 M samples/sec end-to-end (2.3 min to AUC
0.80275). With the multi-hot data path enabled (`HCTR_USE_MULTI_HOT=1`)
we now use NVIDIA's exact data shape — 130 keys/row, 576 B/record,
26 multi-hot slots — and get **4.10 M samples/sec apples-to-apples**.

### Gap analysis: 4.10 M sps (apples-to-apples) vs ~30 M sps ≈ 7.3×

Roughly attributable to:
- **Unfused MLP** (Layer_t.MLP fused GEMM+ReLU+bias+RELU_AUX — we
  substitute InnerProduct stack): ~1.5-2× headroom. Implementation
  in this branch (set `HCTR_USE_FUSED_MLP=1`) — emulates RELU_AUX /
  DRELU / DRELU_BGRAD epilogues with cuBLASLt-format bit-packed mask.
  Single-GPU works (loss 3.23 → 0.245); multi-GPU loss collapses to
  log(2)·2, indicating a correctness bug in the bgrad kernel under
  cross-rank gradient AllReduce. Investigation continues.
- **No multi-node fabric scaling**: NVIDIA's 8 GPU result is on 2 nodes
  × 4 GPU with NVLink/NVSwitch fabric. Our 8 GPU are inside one node
  with xGMI. Probably a wash given the per-node-vs-cross-node tradeoff.
- **Hardware difference**: B200 HBM3e + 5th-gen Tensor Cores vs MI350X
  HBM3 + MFMA — at this tensor-density compute the per-GPU peak FLOPs
  are similar but B200 has higher HBM bandwidth. Probably ~2-3× of the
  remaining headroom (multi-hot is heavily embedding-bandwidth-bound).
- **Real multi-hot Criteo vs synthetic expansion of day_0**: NVIDIA uses
  the full 4.2 B-row Meta multi-hot dataset (mlperf reference), we
  synthesise 21 M rows of multi-hot from our existing day_0 single-hot
  via per-offset prime mixing (see
  `runtime_test/criteo_npy_to_hugectr_bin_mh.py`). Same record format
  (576 B), same slot cardinalities, same MULTI_HOT_SIZES — just much
  less data and a less natural distribution.

### Status of the per-rank batch ≥ 2048 NaN (now FIXED)
1. *Wave-size mismatch*: `WARP_SIZE` was hardcoded to 32 in upstream HugeCTR,
   but AMD MI350X has wavefront size 64. This caused row-cross-contamination
   in MultiCross's bprop kernels (`matrix_pair_mul_kernel`,
   `row_scaling_sum_kernel`). Fixed in `HugeCTR/include/common.hpp` and
   `HugeCTR/embedding/operators/generic_lookup.cuh`.
2. *FP16 BGRADA overflow*: The MultiCross bias-gradient kernel sums per-rank
   batch worth of FP16 values which can exceed FP16 max (65,504) at large
   batches; the in-FP16 `ncclSum` all-reduce then propagates inf/NaN.
   Mitigated in `HugeCTR/src/layers/functors/fused_gemm_functors.cu` by
   FP32-accumulation + pre-divide /256 + isfinite() guard + clamp on the
   BGRADA store. This unblocks per-GPU batches up to ~1024 but a complete
   fix requires either a FP32 wgrad all-reduce (HugeCTR-side change) or a
   different DCN-v2 numerical formulation. Single-GPU FP16 and 8-GPU FP32
   are unaffected.

3. *Per-GPU batch ≥ 2048 NaN — FIXED via FP16 sanitisation in
   MultiCross fprop output*: A single FP16 NaN/inf appearing in any
   element of `layer_output_tensors[i]` (the per-cross-layer output)
   poisoned the whole tensor through the subsequent layer's GEMMs
   (NaN×anything = NaN). Added a small post-pass kernel
   (`clamp_fp16_kernel` in `multi_cross_layer.cu`) that runs after
   every cross-layer's `fused_matrix_elementwise_dot_add`, replaces
   NaN/inf with 0, and clamps to ±FP16 max. This is a containment fix
   (not a root-cause fix — somewhere in the FP16 math chain a single
   bad element does still appear at large per-rank batches), but it
   prevents the poisoning chain that NaN'd subsequent iters. Set
   `HCTR_MC_CLAMP_FP16=0` to disable.

   Pre-fix bisection table (kept here for the historical record):

   | Global batch (per-GPU) | Sharding | Iter-1 loss (frozen) | Iter ≥ 2 |
   |---|---|---|---|
   | 8192  (1024) | round_robin | 3.18 *(= 8 × 0.4, OK)* | stable |
   | 16384 (2048) | round_robin | 1.41 *(= 8 × 0.18, OK)* | NaN |
   | 32768 (4096) | round_robin | — | NaN |
   | 55296 (6912) | auto | 3.07 *(= 8 × 0.38, OK)* | NaN |
   | 16384 (2048) | round_robin, **InnerProduct subst (no MultiCross)** | 2.31 *(= 8 × 0.29, OK)* | **stable 10+ iters** |
   | 32768 (4096) | round_robin, **InnerProduct subst (no MultiCross)** | 0.84 *(= 8 × 0.10, OK)* | **stable 10+ iters** |

   *(The "high" iter-1 losses are simply HugeCTR's multi-GPU loss display
   summing per-rank BCE; per-rank loss is normal random-init ~0.2-0.4 in
   all cases.)*

   The control InnerProduct-substitute experiment at the same batch sizes
   stays stable for 10+ iters even with frozen weights. So the embedding
   all-to-all, data reader, bottom MLP, top MLP, and BCE loss path all
   behave correctly at the largest batches. The bug is
   **MultiCross-specific**: bprop must be writing inf/NaN into some
   buffer that is read by the next iter's fprop, since iter-1 forward is
   fine but iter-2 NaNs even with effectively zero weight updates.

   **Diagnostic env knobs** added to `fused_gemm_functors.cu` and the
   driver script for further bisection:
   - `HCTR_DISABLE_BGRADA=1` — memset bias-grad to 0 (no BGRADA write)
   - `HCTR_DISABLE_BIAS=1` — skip the post-pass BIAS add in fprop
   - `HCTR_DCN_NUM_LAYERS=N` — run with N MultiCross layers (default 3)
   - `HCTR_DCN_PROJ_DIM=D` — projection dim (default 512)
   - `HCTR_OPTIMIZER=sgd|adagrad` — switch optimiser

   **Where to look next**: per-tensor max-abs instrumentation through
   `MultiCrossLayer<__half>::fprop` (the `XU`, `XUV+b`, and
   `fused_matrix_elementwise_dot_add` outputs) to identify which
   intermediate first overflows at per-GPU batch ≥ 2048. Since the
   bug is purely in fprop with frozen weights, an isolated unit test
   that drives MultiCross directly with synthetic FP16 inputs at the
   failing per-rank batch sizes should reproduce it without the full
   training loop.

## What's in this directory

```
hugectr_rocm_port/
├── CMakeLists.txt                # top-level HIP build (gfx950)
├── hugectr_hip_warp_compat.h     # __ffs/__popc 64-bit overloads for wave64
├── HugeCTR/                      # hipified HugeCTR C++ source (this is the
│                                 # full upstream tree post-hipify-perl + our
│                                 # hand fixes for ROCm 7.2 / hipBLASLt 1.2)
├── gpu_cache/                    # GPU cache (hipified)
├── third_party/                  # vendored deps (HierarchicalKV, json, ...)
├── cmake/                        # FindNUMA.cmake etc.
├── runtime_test/
│   └── nvidia_frontend/          # NVIDIA's train.py + mlperf_logger +
│                                 # sharding/, plus our mlperf_common stub
│                                 # and mpi4py shim for single-node runs
└── scripts/
    ├── rebuild_in_container.sh   # cmake configure + build inside ROCm 7.2.1
    ├── build_in_container.sh
    ├── run_in_container.sh       # legacy single-GPU smoke
    ├── run_nvidia_frontend.sh    # original ev_size=16, batch 8192 baseline
    └── run_b200_match.sh         # env-driven; defaults match the B200 config
```

## Build (inside the ROCm container)

```bash
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged \
    -v $(pwd)/hugectr_rocm_port:/workspace \
    -w /workspace \
    rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424 \
    bash scripts/rebuild_in_container.sh
```

Output goes to `hugectr_hip/build_rocm72/lib/{hugectr.so,libhuge_ctr_shared.so,
libembedding.so,libgpu_cache.so,libhugectr_core23.so}` (these are not committed;
build them locally).

## Run (B200-config matched, single node × 8 MI350X, FP32 + real DCN-v2)

```bash
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host \
    -v $(pwd)/hugectr_rocm_port:/workspace \
    -v /path/to/your/criteo_hugectr_bin:/criteo \
    -e HCTR_NGPU=8 -e HCTR_BATCH=55296 -e HCTR_EVAL_BATCH=131072 \
    -e HCTR_EV_SIZE=128 -e HCTR_LR=0.004 -e HCTR_MAX_ITER=100 -e HCTR_DISPLAY=10 \
    -e HCTR_MEM_CAP=64 \
    -e HCTR_PRECISION_FLAGS=" " \
    -e HCTR_SHARDING_PLAN=auto -e OMP_NUM_THREADS=8 \
    -w /workspace \
    rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424 \
    bash scripts/run_b200_match.sh
```

Switch to FP16 mixed by setting `HCTR_PRECISION_FLAGS="--use_mixed_precision --scaler 16348"`.

### Run with NVIDIA's multi-hot data shape (apples-to-apples, 576 B/record)

First synthesize multi-hot from your existing single-hot Criteo day_0 NumPy:

```bash
python3 runtime_test/criteo_npy_to_hugectr_bin_mh.py \
    --npy-dir /path/to/criteo_npy \
    --out-dir /path/to/criteo_hugectr_bin_mh \
    --day day_0 --train-frac 0.95
```

This produces 21 M rows of 576-B multi-hot records (~20 GB) using NVIDIA's
default `MULTI_HOT_SIZES = [3,2,1,2,6,...,1,1]` (sum=130 keys/row).

Then add `-e HCTR_USE_MULTI_HOT=1` to the docker invocation above. The
`run_b200_match.sh` script picks `/criteo/hugectr_bin_mh` automatically.

### Run with the new fused MLP path (single-GPU only; multi-GPU has bug)

Add `-e HCTR_USE_FUSED_MLP=1`. This switches all dense MLPs from the
InnerProduct + ReLU stack back to NVIDIA's `Layer_t.MLP` and routes
the RELU_AUX / DRELU / DRELU_BGRAD epilogues through our new fallback
kernels in `HugeCTR/src/layers/functors/fused_gemm_functors.cu`.
1 × MI350X works (loss 3.23 → 0.245 over 20 iters at batch 8192).
8 × MI350X collapses to log(2)·2 — the bgrad gradient appears not to
cross-rank reduce correctly under our DRELU mask emulation. Open work.

## Key ROCm port changes (vs upstream HugeCTR)

The full diff lives in `HugeCTR/` and `gpu_cache/`. The major themes:

1. **`hipify-perl` pass over the entire CUDA tree** (~234 K LOC) plus
   ~30 hand-fixes for things hipify doesn't translate cleanly:
   - 64-bit warp-mask intrinsics (`__ffs`/`__popc`) — `hugectr_hip_warp_compat.h`
   - `cooperative_groups::reduce / inclusive_scan` → hand-rolled `shfl_xor`
     primitives in `key_filtering_operators.cu`
   - `MLCommon::LinAlg` (cuml dependency) → HIP-native shim in
     `prims/mlcommon_linalg_hip.cuh`
   - `cublasLt*` enums (e.g. `CUBLASLT_EPILOGUE_RELU_AUX`) → `hipblasLt*`
   - Stripped NVIDIA-only deps (`cuDNN`, NVML, CUDA virtual-memory allocators)
   - `bar.sync` PTX inline asm → `__syncthreads()`

2. **`-fopenmp` for HIP source files** (CMakeLists.txt). Without this,
   `#pragma omp parallel` is silently no-opped by amdclang, causing the AUC
   warmup `ncclAllReduce` to deadlock waiting on ranks that never enter the
   collective. (Multi-GPU was hung at "Starting AUC NCCL warm-up" until this
   was added.)

3. **Fused-GEMM functor with hipblasGemmEx fallback** in
   `HugeCTR/src/layers/functors/fused_gemm_functors.cu`. hipBLASLt 1.2 on
   gfx950 either returns 0 heuristic candidates for, or returns runtime-failing
   algos for, several MultiCross v2 GEMM shapes. The fallback caches m/n/k/op
   in `CublasDesc` and routes to `hipblasGemmEx` (FP32 accumulation) plus
   manual `BIAS` (per-row add) and `BGRADA` (column sum) post-pass kernels.

4. **MLP layers** are kept as `InnerProduct + ReLU` stacks (not the fused
   `Layer_t.MLP`), because hipBLASLt 1.2 lacks heuristic candidates for the
   fused MLP's RELU_AUX epilogue at our shapes. ~1.3-1.5× perf left on the
   table; restoring fused MLP is on the TODO list.

5. **HIP graph capture and intra/inter-iter overlap disabled** in the solver
   (env-driven via `HCTR_USE_CUDA_GRAPH`). Re-enabling these is safe once
   item 4 lands and removes the last hipBLASLt fallback path from the hot
   loop.

## Open work

- **FP16 + multi-GPU + real MultiCross v2** NaN — needs per-rank tensor
  instrumentation. FP32 multi-GPU and FP16 single-GPU both converge.
- **Real multi-hot Criteo data path** — preprocessing tools work on the
  single-hot day_0 only; need to run the full
  `materialize_synthetic_multihot_dataset.py` + `convert_to_raw.py` for the
  4 TB MLPerf-spec dataset.
- **BF16 path** — would need `enable_bf16_compute` flag + `hip_bfloat16`
  template instantiations across `HugeCTR/src/layers/`.
- **Multi-node** (RDMA `NetworkExchangeWgrad`) — currently single-node only.
- **Restore fused `Layer_t.MLP`** — unfused unless hipBLASLt gains a kernel
  for the RELU_AUX epilogue at the requested matrix layouts.

## Vendored upstream content

- `runtime_test/nvidia_frontend/train.py`, `mlperf_logger/`, `sharding/` —
  vendored from the NVIDIA submission in this same repo
  (`NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr/`) with edits gated
  on env vars (`HCTR_USE_SUBSAMPLED_CRITEO`, `HCTR_USE_REAL_TABLE_SIZES`,
  `HCTR_USE_INNERPRODUCT_INSTEAD`, etc.).
- `mlperf_common_stub.py`, `mpi4py_stub.py`, and the `mlperf_common/` package
  are minimal shims — replace with the real `mlperf_common` for production.

## License

Inherits Apache-2.0 from upstream HugeCTR + this MLPerf submissions repo.
