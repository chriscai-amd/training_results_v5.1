# DLRM-DCNv2 — HugeCTR ROCm port for AMD Instinct MI350X

A working port of NVIDIA's MLPerf v5.1 HugeCTR DLRM-DCNv2 submission to AMD's
ROCm 7.2 platform, targeting `gfx950` (Instinct MI350X). Source-compatible
with NVIDIA's `train.py` Python frontend and `mlperf_logger` MLLog format.

This is a research / port branch, **not an official MLPerf submission**.

## Status (single-node, 8 × MI350X, 1 node)

| Configuration | Throughput | Notes |
|---|---|---|
| 8 × MI350X, FP32, real DCN-v2 (3-layer MultiCross v2, proj=512) | **3.89-6.59 M samples/sec** | exact NVIDIA MLPerf model graph |
| 8 × MI350X, FP16 mixed (scaler 16348), real DCN-v2, **per-GPU batch ≤ 1024** | **3.05 M samples/sec** | converges 80+ iters with WARP_SIZE / FP16-clamp fixes (see "Open work" for the larger-batch limitation) |
| 8 × MI350X, FP16 mixed (scaler 16348), InnerProduct-substitute interaction | **12.29 M samples/sec** | MLPs+optimizer match; substitute for cross net |
| 1 × MI350X, FP16 mixed, real DCN-v2 | 1.84 M samples/sec | full architecture |
| 1 × MI350X, FP32, real DCN-v2 | 0.85 M samples/sec | full architecture |

NVIDIA B200 reference (8 GPU, FP16, full multi-hot Criteo, fused MLP, HIP graph):
~30 M samples/sec end-to-end (2.3 min to AUC 0.80275).

**Open: 8 × MI350X with FP16 + real MultiCross at per-rank batch ≥ 2048
still NaNs after a few iters.** Two contributing root causes have been
identified and partially fixed:
1. *Wave-size mismatch*: `WARP_SIZE` was hardcoded to 32 in upstream HugeCTR,
   but AMD MI350X has wavefront size 64. This caused row-cross-contamination
   in MultiCross's bprop kernels (`matrix_pair_mul_kernel`,
   `row_scaling_sum_kernel`). Fixed in `HugeCTR/include/common.hpp` and
   `HugeCTR/embedding/operators/generic_lookup.cuh`.
2. *FP16 BGRADA overflow*: The MultiCross bias-gradient kernel sums per-rank
   batch worth of FP16 values which can exceed FP16 max (65,504) at large
   batches; the in-FP16 `ncclSum` all-reduce then propagates inf/NaN.
   Mitigated in `HugeCTR/src/layers/functors/fused_gemm_functors.cu` by
   FP32-accumulation + clamp on the BGRADA store. This unblocks per-GPU
   batches up to ~1024 but a complete fix requires either a FP32 wgrad
   all-reduce (HugeCTR-side change) or a different DCN-v2 numerical
   formulation. Single-GPU FP16 and 8-GPU FP32 are unaffected.

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
